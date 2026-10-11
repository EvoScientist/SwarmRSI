#!/usr/bin/env python3
"""Run one NatureBench cell with upstream AIDE, under our budget and evaluator.

This does not reimplement AIDE. It loads the pinned upstream Agent and its
draft/debug/improve policy unchanged, and supplies four things the comparison
needs:

  budget      AIDE's own agent.steps is replaced by the task's matched
              backbone_message_ids from our profile, so both methods face the
              same envelope. Its defaults (steps=20, num_drafts=5,
              debug_prob=0.5, max_debug_depth=3) are recorded, never used.
  execution   AideNatureExecution runs each candidate in the same bwrap
              sandbox and scores it with the same official evaluator we use.
  accounting  AideCallLedger caps calls, output tokens and wall time and keeps
              a per-call receipt.
  evidence    run_controller seals identity, checkpoint and a final snapshot.

Usage:
  run_aide_naturebench_cell.py --root <fresh dir> --task <task id>
      --profile <profile.json> --tasks-root <dir> --runtime-python <python>
      --upstream-root <aide checkout> --secret <openrouter json>
      [--model deepseek/deepseek-v4-pro] [--steps N]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

# Import the shared defaults whether or not the package is installed: these scripts
# are run straight from a checkout as often as from an install.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from evolvcoder.model_defaults import DEFAULT_BACKBONE  # noqa: E402

from types import SimpleNamespace

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "src"))

from evolvcoder.baselines.aide_controller import AideRuntime, run_controller  # noqa: E402
from evolvcoder.baselines.aide_execution import AideNatureExecution  # noqa: E402
from evolvcoder.baselines.aide_openrouter import make_openrouter_query  # noqa: E402
from evolvcoder.baselines.aide_provider_process import AideProviderProcess  # noqa: E402
from evolvcoder.baselines.aide_resources import AideOperationBudget  # noqa: E402
from evolvcoder.baselines.aide_session import AidePhaseCalls, AideSession  # noqa: E402
from evolvcoder.baselines.aide_state import AideJournalStore  # noqa: E402
from evolvcoder.baselines.aide_transport import AideCallLedger  # noqa: E402
from evolvcoder import naturebench_evaluator as nature_bridge  # noqa: E402


def tolerate_non_utf8_previews() -> None:
    """Upstream's data preview opens every .txt/.csv/.json file as UTF-8 and the run stops before its
    first model call when one is not (NB 00757: UnicodeDecodeError, 2026-09-28). Only that failure is
    caught: the file is then described the way upstream describes binary files (byte size), or its
    preview says it is not UTF-8 text. Every other file is previewed exactly as upstream does."""
    import humanize
    from aide.utils import data_preview as dp

    def guard(fn, fallback):
        def wrapped(*args, **kwargs):
            try:
                return fn(*args, **kwargs)
            except UnicodeDecodeError:
                return fallback(*args, **kwargs)
        return wrapped

    def size_only(f, *_, **__):
        s = Path(f).stat().st_size
        return s, humanize.naturalsize(s)

    dp.get_file_len_size = guard(dp.get_file_len_size, size_only)
    dp.preview_csv = guard(dp.preview_csv, lambda p, file_name, *a, **k: f"-> {file_name} is not UTF-8 text; no preview.")
    dp.preview_json = guard(dp.preview_json, lambda p, file_name, *a, **k: f"-> {file_name} is not UTF-8 text; no preview.")


def secret_key(path: Path) -> str:
    """A JSON {"api_key": ...} file (OpenRouter) or a plain-text key file (DeepSeek)."""
    text = path.read_text().strip()
    return json.loads(text)["api_key"] if text.startswith("{") else text


def parse() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True, help="fresh run directory")
    p.add_argument("--task", required=True)
    p.add_argument("--profile", type=Path, required=True)
    p.add_argument("--tasks-root", type=Path, required=True)
    p.add_argument("--runtime-python", type=Path, required=True)
    p.add_argument("--upstream-root", type=Path, required=True)
    p.add_argument("--secret", type=Path, required=True)
    p.add_argument("--model", default=DEFAULT_BACKBONE)
    p.add_argument("--base-url", default="https://openrouter.ai/api/v1",
                   help="OpenAI-compatible endpoint, e.g. https://api.deepseek.com")
    p.add_argument("--reasoning-param", choices=("openrouter", "reasoning_effort"), default="openrouter")
    p.add_argument("--tool-choice", choices=("named", "auto"), default="named")
    p.add_argument("--request-timeout", type=float, default=None,
                   help="per-request ceiling in seconds (default: the remaining wall)")
    p.add_argument("--evaluator-src", type=Path, default=None,
                   help="src directory whose evolvcoder/naturebench_evaluator.py scores candidates (default: this checkout)")
    p.add_argument("--steps", type=int, default=None,
                   help="override the budget-derived step count (audit only)")
    p.add_argument("--dry-run", action="store_true", help="wire everything, make no model call")
    return p.parse_args()


def main() -> int:
    args = parse()
    profile = json.loads(args.profile.read_text())
    task = profile["tasks"][args.task]
    limits = task["limits"]
    task_root = (args.tasks_root / args.task).resolve()

    sys.path.insert(0, str(args.upstream_root.resolve()))
    from aide.agent import Agent  # noqa: E402
    from aide.journal import Journal, Node  # noqa: E402
    from aide.interpreter import ExecutionResult  # noqa: E402
    from aide.utils.metric import MetricValue, WorstMetricValue  # noqa: E402
    from aide.backend import backend_openrouter, provider_to_query_func  # noqa: E402
    from omegaconf import OmegaConf  # noqa: E402
    tolerate_non_utf8_previews()

    cfg = OmegaConf.load(args.upstream_root / "aide/utils/config.yaml")
    upstream_defaults = {"steps": int(cfg.agent.steps), **{k: v for k, v in dict(cfg.agent.search).items()}}
    # Budget translation: one AIDE step is one backbone message, matching the
    # envelope the reference trace used on this task. Under the declared-budget
    # protocol the counts are null and only the wall clock binds, as it does for
    # BRS; the ledger needs positive integers, so a null count becomes a sentinel
    # far above anything four hours can reach, and the identity says so.
    uncapped = {"calls": 100_000, "output_tokens": 10**9, "score_attempts": 100_000}
    wall_only = limits["backbone_message_ids"] is None
    calls_cap = uncapped["calls"] if wall_only else int(limits["backbone_message_ids"])
    output_cap = (uncapped["output_tokens"] if limits["tokens"]["output"] is None
                  else int(limits["tokens"]["output"]))
    score_cap = (uncapped["score_attempts"] if limits["score_bearing_evaluations"] is None
                 else int(limits["score_bearing_evaluations"]))
    steps = int(args.steps if args.steps is not None else calls_cap)
    cfg.agent.steps = steps
    cfg.workspace_dir = str((args.root / "public-workspace").resolve())
    for entry in (cfg.agent.code, cfg.agent.feedback, cfg.report):
        entry.model = args.model

    identity = {
        "schema_version": "aide-naturebench-cell-v1",
        "benchmark": "naturebench", "task_id": args.task, "model": args.model,
        "upstream_root": str(args.upstream_root.resolve()),
        "budget": {"steps": steps, "model_calls": limits["backbone_message_ids"],
                   "output_tokens": limits["tokens"]["output"],
                   "wall_seconds": int(limits["wall_time_seconds"]),
                   "score_attempts": limits["score_bearing_evaluations"],
                   "wall_only_protocol": wall_only,
                   "sentinel_caps": uncapped if wall_only else None},
        "upstream_defaults_not_used": upstream_defaults,
        "provider_routing": {"provider": {**(profile.get("provider") or {}), "ignore": ["OpenInference"]},
                             "reasoning": {"effort": "medium", "exclude": True}, "per_call_output_tokens": 24000,
                             "retry_failed_provider_attempts": True,
                             "missing_tool_call_retries": 2,
                             "base_url": args.base_url, "reasoning_param": args.reasoning_param,
                             "tool_choice": args.tool_choice,
                             "request_timeout_seconds": args.request_timeout,
                             "evaluator_src": None if args.evaluator_src is None else str(args.evaluator_src),
                             "note": "BRS's NatureBench provider policy and planner reasoning; upstream Fireworks pin removed"},
        "claim_scope": "same model name and frozen task/evaluator contract; "
                       "AIDE's own step/draft defaults replaced by the matched envelope",
    }

    def environment_note() -> str:
        # AIDE's own prompt advertises a fixed package list (lightGBM, statsmodels, timm, ...)
        # that this runtime does not have. BRS's planner is given the measured inventory; AIDE
        # gets the same facts here, as task input, without touching its prompts.
        import subprocess
        freeze = subprocess.run([str(args.runtime_python), "-m", "pip", "list", "--format=freeze"],
                                capture_output=True, text=True, check=True).stdout
        packages = sorted({line.split("==")[0] for line in freeze.splitlines() if "==" in line}, key=str.lower)
        cache = os.environ.get("EVOLVCODER_MODEL_CACHE")
        paths = []
        if cache:
            root = Path(cache)
            paths += [f"/model_cache/hub/checkpoints/{p.name}" for p in sorted(root.glob("hub/checkpoints/*.pt*"))]
            paths += [f"/model_cache/transformers/{p.name} (from_pretrained('{p.name[8:].replace('--', '/')}'))"
                      for p in sorted(root.glob("transformers/models--*"))]
        return ("\n\n## Execution environment (measured; overrides any generic instructions)\n"
                "- Outputs: the official NatureBench evaluator reads ONLY the files your program writes into the "
                "directory named by the environment variable OUTPUT_DIR (./output), with exactly the names and formats "
                "the task description above requires. ./working/submission.csv is not graded for this task.\n"
                "- Inputs: the public data is in the directory named by DATA_DIR (also available as ./input).\n"
                "- The code is executed as a script body with its own global namespace in which __name__ is not "
                "\"__main__\"; code under `if __name__ == \"__main__\":` will not run.\n"
                "- No network access. Only these Python packages are installed: " + ", ".join(packages) + ".\n"
                + ("- Offline model files (TORCH_HOME=/model_cache, HF_HOME=TRANSFORMERS_CACHE=/model_cache/transformers, "
                   "HF_HUB_OFFLINE=1): " + "; ".join(paths) + ". Nothing else can be downloaded.\n" if paths else ""))

    # The scorer subprocess can come from another checkout so every method in a comparison is scored by
    # the same evaluator code (Snellius: source_22944f9, as BRS and Arbor). The runner itself is unchanged.
    evaluator_script = REPO / "src/evolvcoder/naturebench_evaluator.py"
    if args.evaluator_src is not None:
        evaluator_script = args.evaluator_src.resolve() / "evolvcoder/naturebench_evaluator.py"
        os.environ["PYTHONPATH"] = str(args.evaluator_src.resolve()) + (os.pathsep + os.environ["PYTHONPATH"] if os.environ.get("PYTHONPATH") else "")

    def initialize(root: Path) -> AideRuntime:
        helper = REPO / "scripts/run_aide_provider_once.py"
        provider = AideProviderProcess(
            python=Path(sys.executable), helper=helper,
            remaining_seconds=lambda: float(limits["wall_time_seconds"]),
            environment={**os.environ, "OPENROUTER_API_KEY": secret_key(args.secret), "AIDE_PROVIDER_BASE_URL": args.base_url},
        )
        ledger = AideCallLedger(
            root / "calls", model=args.model,
            max_calls=calls_cap,
            max_output_tokens=output_cap,
            wall_seconds=float(limits["wall_time_seconds"]),
            per_call_output_tokens=24000, create=provider,  # BRS planner cap; 16000 was used up by reasoning
            retry_failed_attempts=True,  # a failed call is counted (possibly billed) and retried, as for BRS
        )
        # Same response factory the session tests use: the ledger keeps raw JSON and the
        # upstream code expects OpenAI ChatCompletion objects.
        from openai.types.chat import ChatCompletion
        calls = AidePhaseCalls(ledger, ChatCompletion.model_validate)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls.create)))
        adapter = make_openrouter_query(
            client=client, backoff_create=lambda create, errors, **kw: create(**kw), timeout_exceptions=(),
            # Same provider policy and planner reasoning as BRS on NatureBench (profile["provider"],
            # planner_reasoning medium): unrestricted routing sent reviews to a 4096-token provider
            # and unbounded reasoning used all 16000 output tokens without a tool call.
            # OpenInference serves this model with a ~4.5k output cap, which require_parameters
            # does not exclude; every retry landed there on 00689.
            provider=({**(profile.get("provider") or {}), "ignore": ["OpenInference"]}
                      if "openrouter.ai" in args.base_url else None),
            reasoning={"effort": "medium", "exclude": True},
            missing_tool_call_retries=2,
            reasoning_param=args.reasoning_param, tool_choice=args.tool_choice,
            request_timeout=args.request_timeout,
        )
        backend_openrouter.query = adapter
        provider_to_query_func["openrouter"] = adapter
        budget = AideOperationBudget(ledger=ledger, artifacts=root / "artifacts",
                                     max_score_attempts=score_cap)
        executor = AideNatureExecution(
            workspace=root / "public-workspace", artifacts=root / "artifacts",
            task_root=task_root, runtime_python=args.runtime_python,
            upstream_root=args.upstream_root, dependency_root=REPO / "src",
            helper_path=REPO / "scripts/run_aide_interpreter_once.py",
            interpreter_sha256=json.loads(
                (REPO / "config/baseline_audits/naturebench_aide_fm_agent_reproduction_v1.json").read_text()
            )["aide"]["source_attestation"]["aide/interpreter.py"],
            # The same evaluator module BRS uses: it builds the offline candidate
            # sandbox (_candidate_command) and attests outputs (_attest), and its
            # CLI is the official scorer. bridge=None crashed at the first execution.
            bridge=nature_bridge, evaluator_script=evaluator_script,
            timeout_seconds=int(profile.get("screening_execution_limits", {}).get(
                "execution_validation_timeout_seconds", 1800)),
            operation_budget=budget,
        )
        executor.prepare()
        agent = Agent(task_desc=(task_root / "problem/README.md").read_text()[:18000] + environment_note(),
                      cfg=cfg, journal=Journal())
        session = AideSession(
            agent=agent, store=AideJournalStore(root / "checkpoint.json", identity=identity),
            calls=calls, executor=executor, node_factory=Node, journal_factory=Journal,
            metric_factory=MetricValue, worst_metric_factory=WorstMetricValue,
            execution_result_factory=ExecutionResult,
        )
        def report(journal):
            from aide.journal2report import journal2report
            return journal2report(journal, agent.task_desc, cfg.report)

        return AideRuntime(session=session, budget=budget, report=report, restore_factories={
            "node_factory": Node, "journal_factory": Journal, "metric_factory": MetricValue,
            "worst_metric_factory": WorstMetricValue,
        })

    if args.dry_run:
        args.root.mkdir(parents=True, exist_ok=True)
        print(json.dumps({"wired": True, "model_calls": 0, "identity": identity}, ensure_ascii=False, indent=1))
        return 0
    args.root.mkdir(parents=True, exist_ok=True)
    result = run_controller(root=args.root, identity=identity, initialize=initialize)
    print(json.dumps(result, ensure_ascii=False, default=str)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
