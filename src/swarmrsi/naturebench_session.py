#!/usr/bin/env python3
"""Complete NatureBench session migrated from the development runner.

Use naturebench_cli for explicit inputs, launch ownership and terminal-first
verification. No developer checkout or user-machine defaults are required.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import os
import math
import re
import shutil
import subprocess
import sys
import tempfile
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any

PACKAGE_ROOT = Path(__file__).resolve().parent
ROOT = PACKAGE_ROOT.parent

from swarmrsi.campaign import (  # noqa: E402
    ActionExecution, ProgramNode, ProposedAction, VSearchCoordinator,
    artifact_workspace_sha256,
)
from swarmrsi.coder_engine import ContractBoundedCoder  # noqa: E402
from swarmrsi.coder_models import (  # noqa: E402
    CoderConfig, CoderVersion, MutationInspiration, ProgramMutationContext,
    ScientificActionContract, ValidationCommand, make_selected_scientific_plan,
)
from swarmrsi.coder_openrouter import OpenRouterCoderModel  # noqa: E402
from swarmrsi.planner_protocol import (  # noqa: E402
    PROTOCOL_VERSION, compile_proposal_slots, cost_gate_error, normalize_proposal, proposal_legality_error,
)
from swarmrsi.diagnostics import diagnostic_artifact_fields, is_repair_seed  # noqa: E402
from swarmrsi.dynamic_visualization import write_dynamic_research_view  # noqa: E402
from swarmrsi.session_limits import SessionLimits  # noqa: E402
from swarmrsi.session_runtime import (  # noqa: E402
    SessionClock, safe_transport_usage, transport_snapshot, verify_transport_snapshot,
)
from swarmrsi.task_metrics import target as score_target, terminal_targets, planner_metric_instruction  # noqa: E402
from swarmrsi.contracts import (  # noqa: E402
    ActionCandidate, ActionOperator, ProposalRejected, ResearchDependency, ScientificFamily,
    canonical_sha256,
)
from swarmrsi.strategies import (  # noqa: E402
    EvidenceCoupledStrategy, HeterogeneousMultiIslandStrategy,
    ProposalDirective,
    MultiParentGraphStrategy,
    configured_tree_strategy,
)

# Set only by the installed CLI before running this session. Import is inert.
TASKS_ROOT: Path | None = None
ACTIVE_TASK_ID: str | None = None
RUNTIME: Path | None = None
CANDIDATE_RUNTIME: dict | None = None
SECRET: Path | None = None
SECRET_ENV: str | None = None
_CONTROLLER_ENVIRONMENT: dict[str, str] | None = None
MODEL_CACHE: Path | None = None
EVALUATOR = PACKAGE_ROOT / "naturebench_evaluator.py"
from .model_defaults import DEFAULT_BACKBONE, DEFAULT_BASE_URL
from .naturebench_evaluator import MODEL_CACHE_ENVIRONMENT

MODEL = DEFAULT_BACKBONE
BASE_URL = DEFAULT_BASE_URL
COMMON_SCIENTIFIC_IMPORT_ROOTS = (
    "numpy", "pandas", "scipy", "sklearn", "networkx", "h5py", "anndata",
    "scanpy", "torch", "tensorflow", "xgboost", "lightgbm", "igraph",
    "catboost", "esm", "polars", "statsmodels", "matplotlib", "seaborn", "joblib",
)


def sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def cell_source_hashes(starter: Path, profile: Path, search=None) -> dict[str, str]:
    """Use the same complete source binding for fresh cells and resumed cells."""
    # Include every package module and the official task authority itself.
    # Relative package keys work in both a wheel and an unpacked source tree.
    task_root = TASKS_ROOT / ACTIVE_TASK_ID
    paths = (
        *sorted(p for p in starter.rglob("*") if p.is_file()
                and "__pycache__" not in p.parts and p.suffix != ".pyc"),
        profile.resolve(),
        *sorted(PACKAGE_ROOT.rglob("*.py")),
        task_root / "metadata.json",
        *sorted(p for p in (task_root / "evaluation").rglob("*")
                if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"),
        *(p for p in (task_root / "problem/README.md",
                      task_root / "problem/data_description.md") if p.is_file()),
    )
    hashes = {
        str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path): sha(path)
        for path in paths
    }
    metadata = json.loads((task_root / 'metadata.json').read_text())
    if metadata.get('benchmark') == 'mlebench':
        from swarmrsi.mlebench_tasks import verify_assets
        custody_path = Path(metadata['custody'])
        custody = verify_assets(custody_path)
        hashes[str(custody_path)] = sha(custody_path)
        hashes.update({ref['path']: ref['sha256'] for ref in custody['refs']})
    if search is not None and callable(getattr(search, "source_hashes", None)):
        hashes.update(search.source_hashes())
    return hashes


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def root_score_bearing_count(*, valid: bool, metric: float | None) -> int:
    """Count only roots that reached a valid official scientific score."""
    return int(valid and metric is not None)


def action_start_envelope(
    *,
    remaining_calls: int,
    remaining_evals: int,
    remaining_input: float,
    remaining_output: int,
    remaining_tools: int,
    remaining_wall: float,
    screening_limits: dict[str, Any],
    observed: dict[str, float] | None = None,
) -> tuple[bool, dict[str, dict[str, float]]]:
    """Require enough residual budget to execute and validate one whole action.

    The configured minima are floors chosen once, by hand, for one backbone. What an
    action actually costs is a property of the backbone and the task: on MLE-Bench a
    v4-pro action averaged 24.5 model calls against a floor of 4, and on OpenProblems
    a flash action averaged 28,786 output tokens against a floor of 8,000. A gate set
    six times below the real cost admits actions that then die mid-flight -- the
    cross-modal cells recorded exactly that, `wall_budget_exhausted` and
    `output_token_budget_exhausted` on actions that had already been started.

    So when this run has finished enough actions to have a mean, that mean is used
    instead, and the configured floor stays as a lower bound. It needs no per-backbone
    tuning: a cheaper model measures itself cheaper and keeps going longer.
    """
    observed = observed or {}

    def minimum(name: str, configured: float) -> float:
        return max(float(configured), float(observed.get(name, 0.0)))

    action_calls = int(screening_limits.get("model_calls_per_action", 8))
    action_tools = int(screening_limits.get("tool_calls_per_action", 24))
    full_timeout = float(screening_limits.get("full_evaluation_timeout_seconds", 1200))
    required = {
        "model_calls": minimum("model_calls", min(action_calls, int(screening_limits.get(
            "minimum_model_calls_to_start_action", 4
        )))),
        "score_evaluations": 1.0,
        "output_tokens": minimum("output_tokens", float(screening_limits.get(
            "minimum_output_tokens_to_start_action", 8000
        ))),
        "tool_calls": minimum("tool_calls", min(action_tools, int(screening_limits.get(
            "minimum_tool_calls_to_start_action", 4
        )))),
        "wall_seconds": minimum("wall_seconds", float(screening_limits.get(
            "minimum_wall_seconds_to_start_action", full_timeout + 60
        ))),
    }
    # Opt-in: the wall floor is sized for an evaluation that runs the whole timeout,
    # but MLE actions measured about 300 s against an 1860 s floor, so runs stopped with
    # 40% of their wall unused. Once this run has a measured mean, the floor may come
    # down to 1.5x that mean (never below 600 s, never above the configured floor).
    if screening_limits.get("wall_floor_from_observed") and observed.get("wall_seconds"):
        measured = max(1.5 * float(observed["wall_seconds"]), 600.0)
        configured = float(screening_limits.get("minimum_wall_seconds_to_start_action", full_timeout + 60))
        required["wall_seconds"] = min(configured, measured)
    available = {
        "model_calls": float(remaining_calls),
        "score_evaluations": float(remaining_evals),
        "output_tokens": float(remaining_output),
        "tool_calls": float(remaining_tools),
        "wall_seconds": float(remaining_wall),
    }
    if screening_limits.get("enforce_context_read_token_limit", True):
        required["context_read_tokens"] = minimum("input_tokens", float(screening_limits.get(
            "minimum_context_read_tokens_to_start_action", 8000
        )))
        available["context_read_tokens"] = float(remaining_input)
    shortfalls = {
        name: {"required": requirement, "available": available[name]}
        for name, requirement in required.items()
        if available[name] < requirement
    }
    return not shortfalls, shortfalls


def transport_backoff_seconds(attempt: int) -> float:
    """Wait before the n-th re-send: 5 s, 20 s, 80 s, then 120 s each.

    Sized to the observed outages of the cluster's proxy upstream; a sub-second retry
    only lands inside the same outage. The ceiling was 60 s while those outages were
    believed to last tens of seconds. One lasted about twenty minutes and ended ten
    paid cells at once, so the ceiling and the budget now cover that case: twelve
    re-sends wait about nineteen minutes in total, against a four-hour cell.
    """
    return float(min(5.0 * (4 ** (attempt - 1)), 120.0))


REASONING_EFFORTS = ("high", "medium", "low")


def step_reasoning_down(model) -> bool:
    """Lower the model's reasoning effort one step; False when already at the floor.

    Works on the same attribute the transport serialises into the request body, so
    every later batch's journalled request shows the effort it actually ran at.
    A model with no reasoning setting is left alone: there is nothing to lower.
    """
    current = getattr(model, "reasoning", None)
    if not isinstance(current, dict) or current.get("effort") not in REASONING_EFFORTS:
        return False
    index = REASONING_EFFORTS.index(current["effort"])
    if index + 1 >= len(REASONING_EFFORTS):
        return False
    model.reasoning = {**current, "effort": REASONING_EFFORTS[index + 1]}
    return True


def observed_action_cost(accounting, *, minimum_samples: int = 2) -> dict[str, float]:
    """Mean cost of one completed action in this run, or {} while it is still guesswork.

    Totals divided by attempted actions, which both survive checkpointing, so a
    resumed run keeps the same view. Below `minimum_samples` the mean is dominated by
    whichever action happened to run first, so the configured floors are left alone.
    """
    attempted = len(getattr(accounting, "attempted_actions", ()) or ())
    if attempted < minimum_samples:
        return {}
    return {name: float(getattr(accounting, name, 0) or 0) / attempted
            for name in ("model_calls", "tool_calls", "output_tokens", "input_tokens", "wall_seconds")}


def expansion_start_envelope(
    *, planner_output_limit: int, remaining_calls: int, remaining_evals: int,
    remaining_input: float, remaining_output: int, remaining_tools: int,
    remaining_wall: float, screening_limits: dict[str, Any],
    observed: dict[str, float] | None = None,
) -> tuple[bool, dict[str, dict[str, float]], int]:
    """Reserve the existing Coder minimum before paying for a new proposal.

    Only expansion needs the extra one model call and at least one output
    token. Queued candidates still use action_start_envelope unchanged. Wall
    and input checks keep the original policy; this is not an end-to-end wall
    accounting repair or a new context-read cap.
    """
    if type(planner_output_limit) is not int or planner_output_limit < 1:
        raise ValueError("planner output limit must be a positive integer")
    ready, shortfalls = action_start_envelope(
        remaining_calls=remaining_calls - 1, remaining_evals=remaining_evals,
        remaining_input=remaining_input, remaining_output=remaining_output - 1,
        remaining_tools=remaining_tools, remaining_wall=remaining_wall,
        screening_limits=screening_limits, observed=observed,
    )
    reserved_output = math.ceil(float(screening_limits.get("minimum_output_tokens_to_start_action", 8000)))
    cap = min(planner_output_limit, remaining_output - reserved_output) if ready else 0
    return ready, shortfalls, cap


def normalize_planner_score(value: Any, default: float) -> float:
    """Normalize untrusted numeric or ordinal planner estimates to [0, 1]."""
    ordinal = {
        "very_low": 0.1, "low": 0.25, "medium": 0.5,
        "high": 0.75, "very_high": 0.9,
    }
    if isinstance(value, bool):
        result = default
    elif isinstance(value, str):
        key = value.strip().lower().replace("-", "_").replace(" ", "_")
        if key in ordinal:
            result = ordinal[key]
        else:
            try:
                result = float(value)
            except ValueError:
                result = default
    else:
        try:
            result = float(value)
        except (TypeError, ValueError):
            result = default
    if not math.isfinite(result):
        result = default
    return max(0.0, min(1.0, result))


def controller_action_operator(
    raw: Any, *, has_intervention_sources: bool,
) -> ActionOperator:
    """Keep typed graph semantics aligned with controller-owned recombination."""
    if has_intervention_sources:
        return ActionOperator.COMBINE_PROGRAMS
    try:
        return ActionOperator(str(raw))
    except ValueError:
        return ActionOperator.IMPROVE_PROGRAM


def execution_log_excerpt(artifact: Path, *, head: int, tail: int) -> str | None:
    """The node's own last full-run output (candidate-produced, public data only).

    Read from the execution evidence the evaluator bridge writes for every scored
    program; None for the root or for evidence written before logs were kept.
    """
    try:
        details = json.loads((Path(artifact) / ".swarmrsi/execution_validation.json").read_text())["details"]
    except Exception:
        return None
    log = details.get("candidate_execution_log")
    if not isinstance(log, str) or not log.strip():
        return None
    if len(log) <= head + tail + 100:
        return log
    return log[:head] + f"\n... [{len(log) - head - tail} characters omitted] ...\n" + log[-tail:]


def compact_evaluation_summary(evidence: dict[str, Any]) -> dict[str, Any]:
    details = evidence.get("details", {}) if isinstance(evidence.get("details"), dict) else {}
    return {
        "schema_version": evidence.get("schema_version"),
        "valid": bool(evidence.get("valid")),
        "aggregate_improvement": evidence.get("metrics", {}).get("aggregate_improvement"),
        "primary_scores": details.get("primary_scores"),
        "per_instance_improvement": details.get("per_instance_improvement"),
        "submission_output_attestation": details.get("submission_output_attestation"),
        "controller_evaluation_attempts": evidence.get("controller_evaluation_attempts", 1),
        "public_diagnostic": evidence.get("public_diagnostic"),
        "score_bearing_evaluation_attempted": bool(
            evidence.get(
                "score_bearing_evaluation_attempted",
                evidence.get("valid") is True
                and evidence.get("metrics", {}).get("aggregate_improvement") is not None,
            )
        ),
    }


def evaluation_evidence_retryable(evidence: Any) -> bool:
    """Retry only evaluator/infrastructure failures, never candidate failures."""
    if not isinstance(evidence, dict):
        return True
    if evidence.get("schema_version") != "external-domain-result-v1":
        return True
    if "valid" not in evidence or "score_bearing_evaluation_attempted" not in evidence:
        return True
    if evidence.get("valid") is True:
        metric = evidence.get("metrics", {}).get("aggregate_improvement")
        return (
            not isinstance(metric, (int, float))
            or isinstance(metric, bool)
            or not math.isfinite(metric)
        )
    diagnostic = evidence.get("public_diagnostic", {})
    category = diagnostic.get("category") if isinstance(diagnostic, dict) else None
    return category in {
        "evaluation_bridge", "official_evaluation", "artifact_attestation",
    }


def coder_sandbox_config(task_root: Path) -> dict[str, Any]:
    if CANDIDATE_RUNTIME is not None:
        from .candidate_execution import validate_binding
        spec, _ = validate_binding(CANDIDATE_RUNTIME)
        return {"wrapper": (), "rewrites": ((str(RUNTIME), spec.python),),
                "environment": (), "available": True}
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        return {
            "wrapper": (), "rewrites": (), "environment": (), "available": False,
        }
    # --unshare-pid: without it the candidate could enumerate the controller's
    # /proc/<pid>/fd (NB 02926 zero-call probe).
    wrapper: list[str] = [bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid"]
    for source in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc")):
        if source.exists():
            wrapper.extend(("--ro-bind", str(source), str(source)))
    if Path("/sys").is_dir():
        wrapper.extend(("--ro-bind", "/sys", "/sys"))
    runtime_root = RUNTIME.parent.parent.absolute()
    wrapper.extend((
        "--ro-bind", str(runtime_root), "/runtime",
        "--ro-bind", str((task_root / "problem/data").resolve()), "/public_data",
        "--bind", "{workspace}", "/workspace",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ))
    model_cache = MODEL_CACHE
    if model_cache is not None and model_cache.is_dir():
        wrapper.extend(("--ro-bind", str(model_cache), "/model_cache"))
    if Path("/proc/driver/nvidia").is_dir():
        wrapper.extend((
            "--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia",
        ))
    for device in sorted(Path("/dev").glob("nvidia*")):
        wrapper.extend(("--dev-bind", str(device), str(device)))
    wrapper.extend(("--chdir", "{sandbox_cwd}", "--"))
    sandbox_environment = [
        ("DATA_DIR", "/public_data"),
        ("OUTPUT_DIR", "/workspace/output"),
        ("TMPDIR", "/tmp"),
    ]
    if model_cache is not None and model_cache.is_dir():
        sandbox_environment.extend(MODEL_CACHE_ENVIRONMENT)
    return {
        "wrapper": tuple(wrapper),
        "rewrites": ((str(RUNTIME), "/runtime/bin/python"),),
        "environment": tuple(sandbox_environment),
        "available": True,
    }


def secret_key() -> str:
    if SECRET_ENV is not None:
        environment = _CONTROLLER_ENVIRONMENT if _CONTROLLER_ENVIRONMENT is not None else os.environ
        key = environment.get(SECRET_ENV)
        if not isinstance(key, str) or not key.strip():
            raise ValueError("configured controller credential environment variable is unavailable")
        return key
    text = SECRET.read_text().strip()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError:
        # A file holding just the key, with no JSON wrapping. Accepting it keeps a
        # hand-written credential file from failing at the first model call, long after
        # the run has been set up. Anything with whitespace inside is not a key.
        if text and not any(c.isspace() for c in text):
            return text
        raise ValueError("credential file is neither JSON nor a bare key")
    if not isinstance(raw, dict):
        raise ValueError("credential file must be a JSON object or a bare key")
    # The backbone is configurable, so the credential file is not necessarily an
    # OpenRouter one; a provider's own key lives under its own name. The generic
    # names are tried first so a file carrying both is unambiguous.
    for key in ("api_key", "key", "OPENROUTER_API_KEY", "DEEPSEEK_API_KEY"):
        if raw.get(key):
            return str(raw[key])
    raise ValueError(
        "credential file has no supported key field; expected one of "
        "api_key, key, OPENROUTER_API_KEY, DEEPSEEK_API_KEY")


def accelerator_inventory() -> dict[str, object]:
    """What compute the candidate will actually find, measured in its own runtime.

    The dependency inventory says torch is importable and stops there, so a plan
    could only infer a GPU by guessing. Across eighteen NatureBench cells not one
    winning program imported torch: every one was scikit-learn or hand-written
    numpy, on tasks whose own metadata asks for a GPU because the published method
    is a graph attention autoencoder or a neural set encoder. One program's own
    docstring named DeepSets and then implemented it in numpy. The gaps against the
    reference run from 0.2 to 1.5, and a card sat idle in every cell.
    """
    script = (
        "import json\n"
        "try:\n"
        "    import torch\n"
        "    ok = bool(torch.cuda.is_available())\n"
        "    n = torch.cuda.device_count() if ok else 0\n"
        "    name = torch.cuda.get_device_name(0) if n else None\n"
        "    total = torch.cuda.get_device_properties(0).total_memory if n else None\n"
        "    print(json.dumps({'cuda': ok, 'devices': n, 'name': name, 'total_memory_bytes': total,\n"
        "                      'torch_version': torch.__version__}))\n"
        "except Exception as exc:\n"
        "    print(json.dumps({'cuda': False, 'devices': 0, 'error': type(exc).__name__}))\n"
    )
    completed = runtime_probe(script)
    if completed.returncode != 0:
        return {"cuda": False, "devices": 0, "error": "probe_failed"}
    try:
        return json.loads(completed.stdout)
    except json.JSONDecodeError:
        return {"cuda": False, "devices": 0, "error": "unparseable_probe_output"}


def runtime_dependency_inventory() -> tuple[tuple[str, ...], tuple[str, ...]]:
    script = (
        "import importlib.util,json; roots="
        + repr(COMMON_SCIENTIFIC_IMPORT_ROOTS)
        + "; print(json.dumps({x: importlib.util.find_spec(x) is not None for x in roots}))"
    )
    completed = runtime_probe(script)
    if completed.returncode != 0:
        raise RuntimeError("cannot inventory the frozen candidate runtime")
    available_map = json.loads(completed.stdout)
    available = tuple(name for name in COMMON_SCIENTIFIC_IMPORT_ROOTS if available_map[name])
    unavailable = tuple(name for name in COMMON_SCIENTIFIC_IMPORT_ROOTS if not available_map[name])
    return available, unavailable


# A probe normally answers in a second or two. On a cold shared filesystem the
# first `import torch` of a runtime that lives on GPFS can take minutes, and
# s41551-024-01257-9 lost its first attempt that way: the accelerator probe hit
# the 30 s limit and the session died before its first model call. The first
# try warms part of the page cache, so one retry with a long limit is enough in
# practice, and a probe that fails twice still fails loudly rather than being
# reported as "no GPU": a planner told a GPU task has no GPU writes CPU code
# for the whole session, which costs more than failing before the first call.
RUNTIME_PROBE_TIMEOUT_SECONDS = 30
RUNTIME_PROBE_COLD_RETRY_SECONDS = 600


def _runtime_probe_once(script: str, timeout: int):
    """One probe attempt; returns None when the attempt timed out."""
    if CANDIDATE_RUNTIME is None:
        try:
            return subprocess.run((str(RUNTIME), "-c", script), capture_output=True, text=True,
                                  timeout=timeout, check=False)
        except subprocess.TimeoutExpired:
            return None
    from .candidate_execution import execute, validate_binding
    import uuid
    spec, state = validate_binding(CANDIDATE_RUNTIME)
    root = state.parent / "runtime_probes" / uuid.uuid4().hex
    workspace, public = root / "workspace", root / "public"
    workspace.mkdir(parents=True)
    public.mkdir()
    record = execute(CANDIDATE_RUNTIME, workspace=workspace, public_data=public,
                     argv=(spec.python, "-I", "-c", script), timeout=timeout)
    if record.get("harness_error") or record.get("cleanup_confirmed") is not True:
        raise RuntimeError("candidate runtime probe failed; inspect " + record["record_dir"])
    if record.get("timed_out"):
        return None
    return subprocess.CompletedProcess([spec.python, "-I", "-c", script], record["returncode"], record["stdout"], record["stderr"])


def runtime_probe(script: str):
    for timeout in (RUNTIME_PROBE_TIMEOUT_SECONDS, RUNTIME_PROBE_COLD_RETRY_SECONDS):
        completed = _runtime_probe_once(script, timeout)
        if completed is not None:
            return completed
    raise RuntimeError(
        f"candidate runtime probe timed out at {RUNTIME_PROBE_TIMEOUT_SECONDS} s and again at "
        f"{RUNTIME_PROBE_COLD_RETRY_SECONDS} s; the runtime filesystem is not serving imports"
    )


# Code and the controller-read internal_validation.json must never be hidden.
_UNIGNORABLE_SUFFIXES = frozenset({".py", ".json"})


def scratch_suffix_patterns(screening_limits: dict[str, Any]) -> tuple[str, ...]:
    """Opt-in: data files a coder's experiments leave in the workspace (feats.npz, model.pkl).

    Ignored like output/: never part of the artifact, never a contract violation, so
    run.py has to regenerate anything it needs. Without this a cached feats.npz voided
    a whole MLE action (nomad a0003) as path_outside_contract.
    """
    suffixes = tuple(screening_limits.get("scratch_output_suffixes", ()))
    for suffix in suffixes:
        if (not isinstance(suffix, str) or not suffix.startswith(".") or "/" in suffix
                or "*" in suffix or suffix.lower() in _UNIGNORABLE_SUFFIXES):
            raise ValueError(f"invalid scratch output suffix: {suffix!r}")
    return tuple(f"*{suffix}" for suffix in suffixes)


def repeated_carve_folds_enabled(task_root: Path, screening_limits: dict[str, Any]) -> int | None:
    """The carve fold count search scores on, or None for the single carve.

    Both the package's evaluation/folds and the profile's carve_folds >= 2 are
    required; with either absent nothing about the session changes (NatureBench
    packages never carry evaluation/folds).
    """
    folds = screening_limits.get("carve_folds")
    if folds is None:
        return None
    if type(folds) is not int or folds < 1:
        raise ValueError("carve_folds must be a positive integer")
    if folds < 2:
        return None
    if not (Path(task_root) / "evaluation/folds").is_dir():
        print(f"warning: carve_folds={folds} but {task_root} has no evaluation/folds; "
              "scoring on the single carve", file=sys.stderr)
        return None
    from .naturebench_evaluator import repeated_carve_folds
    available = repeated_carve_folds(task_root)
    if available < folds:
        raise ValueError(f"carve_folds={folds} but the task package holds {available} carve folds")
    return folds


def repeated_carve_arguments(folds: int | None, *, fold_timeout: int, deadline: float) -> tuple[str, ...]:
    if folds is None:
        return ()
    return ("--carve-folds", str(folds), "--carve-fold-timeout-seconds", str(int(fold_timeout)),
            "--carve-deadline-seconds", str(int(max(1, deadline))))


def candidate_evaluator_arguments():
    from .candidate_execution import evaluate_arguments
    return evaluate_arguments(CANDIDATE_RUNTIME)


def public_data_scale_profile(public_data: Path) -> dict[str, Any]:
    """Return a bounded, deterministic public-data scale summary for planning."""
    files = sorted(path for path in public_data.rglob("*") if path.is_file())
    rows: list[dict[str, Any]] = []
    for path in files[:240]:
        item: dict[str, Any] = {
            "path": path.relative_to(public_data).as_posix(),
            "bytes": path.stat().st_size,
        }
        if path.suffix.lower() in {".csv", ".tsv"} and path.stat().st_size <= 256_000_000:
            delimiter = "\t" if path.suffix.lower() == ".tsv" else ","
            try:
                with path.open(newline="", errors="replace") as handle:
                    reader = csv.DictReader(handle, delimiter=delimiter)
                    count = 0
                    sampled_widths: dict[str, int] = {
                        str(name): 0 for name in (reader.fieldnames or [])
                    }
                    for record in reader:
                        count += 1
                        if count <= 256:
                            for name, value in record.items():
                                sampled_widths[str(name)] = max(
                                    sampled_widths.get(str(name), 0), len(str(value or ""))
                                )
                item["rows"] = count
                item["columns"] = list(sampled_widths)
                item["sampled_max_text_width_by_column"] = sampled_widths
            except (OSError, csv.Error, UnicodeError):
                item["tabular_profile"] = "unavailable"
        rows.append(item)
    return {
        "file_count": len(files),
        "total_bytes": sum(path.stat().st_size for path in files),
        "files": rows,
        "truncated": len(files) > len(rows),
    }



INTERNAL_VALIDATION_NAME = "internal_validation.json"
# Present in the environment of the terminal execution only, under the opt-in policy.
SUBMISSION_FIT_ENV = "SWARMRSI_SUBMISSION_FIT"
SUBMISSION_FIT_POLICIES = ("legacy", "full_refit")

# What a candidate may write. INTERNAL_VALIDATION_NAME is here because the
# standing coder instruction asks for it: a controller-requested artifact that
# the contract forbids turns every obedient candidate into a contract violation,
# which is exactly what cost the whole peptide cell in R35.
CANDIDATE_WRITE_PATHS = ("run.py", "*.py", "src/*.py", "src/**/*.py", INTERNAL_VALIDATION_NAME)


def read_internal_validation(artifact: Path) -> dict[str, Any] | None:
    """A candidate's own estimate from public labels, or None when absent/invalid."""
    path = Path(artifact) / INTERNAL_VALIDATION_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    estimate = raw.get("estimate") if isinstance(raw, dict) else None
    if (not isinstance(raw, dict) or not isinstance(raw.get("metric_name"), str)
            or raw.get("direction") not in {"maximize", "minimize"}
            or type(estimate) not in (int, float) or not math.isfinite(estimate)):
        return None
    return {"metric_name": raw["metric_name"], "direction": raw["direction"], "estimate": float(estimate),
            "protocol": str(raw.get("protocol", ""))[:500]}


def internal_validation_gate(candidate_artifact: Path, parent_artifact: Path | None,
                             *, tolerance: float = 0.0,
                             policy: str = "legacy") -> dict[str, Any] | None:
    """Reject a candidate only when its own estimate is worse than its parent's.

    Both estimates must exist and describe the same metric and direction; the
    root has none, so a first-level candidate is always officially evaluated.
    A rejection spends no official evaluation and is never a refutation.
    """
    if policy not in {"legacy", "controller_only"}:
        raise ValueError("unknown internal validation policy")
    candidate = read_internal_validation(candidate_artifact)
    parent = read_internal_validation(parent_artifact) if parent_artifact is not None else None
    if candidate is None:
        return None
    record: dict[str, Any] = {"candidate": candidate, "parent": parent, "tolerance": tolerance, "rejected": False}
    if policy == "controller_only":
        # Candidate-written estimates cannot establish common rows, preprocessing,
        # or model binding. Even identical prose is not an attestation. Retain the
        # diagnostic but let the existing controller evaluator judge the child.
        # Explicitly mark trust so downstream ECLC cannot reuse this diagnostic.
        record["evidence_authority"] = "candidate_unattested"
        record["reason"] = "unattested_candidate_estimate"
        return record
    if parent is None or parent["metric_name"] != candidate["metric_name"] or parent["direction"] != candidate["direction"]:
        record["reason"] = "no_comparable_parent_estimate"
        return record
    sign = 1.0 if candidate["direction"] == "maximize" else -1.0
    shortfall = sign * (parent["estimate"] - candidate["estimate"])
    record["shortfall"] = shortfall
    if shortfall > tolerance:
        record["rejected"] = True
        record["reason"] = "candidate_estimate_below_parent"
    else:
        record["reason"] = "candidate_estimate_not_below_parent"
    return record


_DATA_PREVIEW_SCRIPT = r"""
import json, os, sys, pathlib
root = pathlib.Path(sys.argv[1]); budget = int(sys.argv[2]); out = []; used = 0
def add(text):
    global used
    if used + len(text) > budget: return False
    out.append(text); used += len(text); return True
def describe(obj, depth=0):
    shape = getattr(obj, "shape", None)
    if shape is not None and hasattr(obj, "dtype"):
        return f"array{tuple(shape)} {obj.dtype}"
    if hasattr(obj, "columns") and hasattr(obj, "shape"):
        return f"DataFrame{obj.shape} columns={list(obj.columns)[:16]}"
    if isinstance(obj, dict):
        items = list(obj.items())[:14]
        inner = ", ".join(f"{k!s}: {describe(v, depth + 1) if depth < 1 else type(v).__name__}" for k, v in items)
        return "dict{" + inner + ("" if len(obj) <= 14 else ", ...") + "}"
    if isinstance(obj, (list, tuple)):
        return f"{type(obj).__name__}[{len(obj)}]" + (f" of {describe(obj[0], depth + 1)}" if obj and depth < 1 else "")
    return type(obj).__name__ + (f" {str(obj)[:60]}" if isinstance(obj, (int, float, str)) else "")
files = sorted(p for p in root.rglob("*") if p.is_file())
seen_suffix = {}
for p in files:
    rel = p.relative_to(root).as_posix(); suf = p.suffix.lower()
    # One preview per directory and suffix: 988 CSVs of one kind need one example.
    key = (str(p.parent), suf)
    if seen_suffix.get(key, 0) >= 2: continue
    seen_suffix[key] = seen_suffix.get(key, 0) + 1
    size = p.stat().st_size; line = f"- {rel} ({size:,} bytes)"
    try:
        if suf in (".csv", ".tsv", ".txt") and size < 5e9:
            import pandas as pd
            sep = "\t" if suf == ".tsv" else None
            df = pd.read_csv(p, sep=sep, nrows=3, engine="python")
            cols = list(df.columns)
            line += f": columns[{len(cols)}]={cols[:24]}{'...' if len(cols) > 24 else ''}; first row={df.head(1).to_dict('records')[0] if len(df) else {}}"
        elif suf == ".npy":
            import numpy as np
            try:
                a = np.load(p, mmap_mode="r", allow_pickle=False); line += f": shape={a.shape} dtype={a.dtype}"
            except ValueError:
                # Pickled object arrays (e.g. a dict of arrays); benchmark-provided public data.
                if size < 2e8:
                    line += ": " + describe(np.load(p, allow_pickle=True).item() if np.load(p, allow_pickle=True).shape == () else np.load(p, allow_pickle=True))
                else:
                    line += ": pickled object array (too large to preview)"
        elif suf in (".pkl", ".pickle") and size < 2e8:
            import pickle
            with open(p, "rb") as f:
                line += ": " + describe(pickle.load(f))
        elif suf == ".mat" and size < 5e8:
            import scipy.io as sio
            line += ": " + describe({k: v for k, v in sio.loadmat(p).items() if not k.startswith("__")})
        elif suf == ".npz":
            import numpy as np
            with np.load(p, allow_pickle=False) as z:
                line += ": " + ", ".join(f"{k}{z[k].shape}" for k in list(z.files)[:12])
        elif suf in (".h5", ".hdf5", ".h5ad", ".loom"):
            import h5py
            with h5py.File(p, "r") as h:
                items = []
                h.visititems(lambda n, o: items.append(f"{n}{getattr(o, 'shape', '')}") if len(items) < 14 else None)
                line += ": " + ", ".join(items)
        elif suf == ".json" and size < 2e8:
            d = json.load(open(p))
            line += f": {type(d).__name__}" + (f" keys={list(d)[:16]}" if isinstance(d, dict) else f" len={len(d)} first={str(d[0])[:160] if d else ''}")
    except Exception as exc:
        line += f": (preview unavailable: {type(exc).__name__})"
    if not add(line[:600] + "\n"):
        add("- ... preview truncated ...\n"); break
print("".join(out))
"""


def public_data_preview(public_data: Path, *, budget_chars: int = 8000, timeout_seconds: int = 180) -> str | None:
    """Columns, shapes and a first row per data file, computed once by the controller.

    Candidates spent their first calls writing _probe/_inspect scripts to learn what
    the files contain, in every action. MLEvolve puts a data preview in every prompt.
    Public data only; failures give None and never block a session.
    """
    if not public_data.is_dir():
        return None
    try:
        completed = subprocess.run((str(RUNTIME), "-c", _DATA_PREVIEW_SCRIPT, str(public_data), str(budget_chars)),
                                   capture_output=True, text=True, timeout=timeout_seconds, check=False)
    except Exception:
        return None
    text = completed.stdout.strip()
    return text or None


def log_loss_metric_name(task_root: Path) -> str | None:
    """The task's public metric name when it is a log loss / cross-entropy, else None."""
    try:
        metric = json.loads((Path(task_root) / "metadata.json").read_text()).get("search_metric") or {}
    except (OSError, ValueError):
        return None
    name = str(metric.get("name", "")) if isinstance(metric, dict) else ""
    return name if re.search(r"log[\s_-]*loss|cross[\s_-]*entropy", name, re.IGNORECASE) else None


_TORCHVISION_WEIGHT_CALLS_SCRIPT = (
    "import json\n"
    "import torchvision.models as m\n"
    "calls = {}\n"
    "for name in m.list_models():\n"
    "    for w in m.get_model_weights(name):\n"
    "        calls.setdefault(w.url.rsplit('/', 1)[-1], f'torchvision.models.{name}(weights=\"{w.name}\")')\n"
    "print(json.dumps(calls))\n"
)


def torchvision_weight_calls() -> dict[str, str]:
    """Checkpoint file name -> the torchvision call that loads it, from the candidate runtime.

    The controller environment usually has no torchvision, so the mapping is read where the
    candidate runs; the cache inventory then falls back to bare file names if that fails.
    """
    if RUNTIME is None and CANDIDATE_RUNTIME is None:
        return {}
    try:
        completed = runtime_probe(_TORCHVISION_WEIGHT_CALLS_SCRIPT)
        if completed.returncode != 0:
            return {}
        calls = json.loads(completed.stdout.strip().splitlines()[-1])
    except Exception:
        return {}
    return {str(k): str(v) for k, v in calls.items()} if isinstance(calls, dict) else {}


def offline_model_inventory(cache: Path | None) -> dict[str, list[str]]:
    """What a sandbox with this model cache can load offline, for the coder to know.

    torchvision checkpoints are mapped to the call that loads them when the candidate
    runtime can say so (a file name such as vit_l_16_swag-4f3808c9.pth does not say which
    weights enum it is); Hugging Face snapshots are listed by repo id rather than by
    their blob and lock files. Anything else in the cache is listed by path.
    """
    if cache is None or not Path(cache).is_dir():
        return {}
    cache = Path(cache)
    files = sorted(p.name for p in (cache / "hub/checkpoints").glob("*.pth")) if (cache / "hub/checkpoints").is_dir() else []
    calls = torchvision_weight_calls() if files else {}
    torchvision = [calls.get(name, name) for name in files]
    hub = cache / "transformers"
    huggingface = sorted(
        p.name[len("models--"):].replace("--", "/") for p in hub.glob("models--*") if p.is_dir()
    ) if hub.is_dir() else []
    other = sorted(
        p.relative_to(cache).as_posix() for p in cache.rglob("*")
        if p.is_file() and not p.name.startswith(".")
        and p.relative_to(cache).parts[0] != "transformers"
        and p.parent != cache / "hub/checkpoints"
    )
    return {"torchvision": torchvision, "huggingface": huggingface, "other": other}


def public_task_context(
    task_root: Path, *, execution_timeout_seconds: int | None = None,
    internal_validation: bool = True,
    internal_validation_policy: str = "legacy",
    submission_fit_policy: str = "legacy",
    run_diagnostics: bool = True,
    compact: bool = False,
    write_first: bool = False,
    scratch_suffixes: tuple[str, ...] = (),
    shift_aware_validation: bool = False,
) -> str:
    """Materialize bounded public task context inside the coder workspace.

    compact keeps the file list once: the scale contract lists at most 40 files with a
    column count and the first 12 names instead of per-column widths, capped at 6,000
    characters, and the inventory is dropped when the contract already lists every file. Uncompacted, the two ran to 29,000 characters on some tasks and pushed the README
    out of what the coder received.
    """
    if internal_validation_policy not in {"legacy", "controller_only"}:
        raise ValueError("unknown internal validation policy")
    if submission_fit_policy not in SUBMISSION_FIT_POLICIES:
        raise ValueError("unknown submission fit policy")
    sections = [
        "CODER INTERFACE: You may inspect run.py, contract-allowed helper modules, "
        "TASK_CONTEXT.md, and the controller-owned read-only public_data/ mount. "
        "DATA_DIR points to the same public data during commands. Hidden evaluation "
        "files and labels remain inaccessible. " + (
            "The README, data description and public data preview below give the schemas: "
            "implement the fixed hypothesis directly and inspect public_data/ only to "
            "diagnose a failure you have observed."
            if write_first else
            "Use bounded inspection and local "
            "experiments to infer schemas before implementing the fixed hypothesis."
        )
    ]
    public_data = task_root / "problem/data"
    available, unavailable = runtime_dependency_inventory()
    sections.append(
        "## Frozen Python dependency inventory\n\n"
        "Confirmed available: " + ", ".join(available) + "\n\n"
        "Confirmed unavailable: " + ", ".join(unavailable) + "\n\n"
        "Use only confirmed-available third-party dependencies. The controller "
        "will reject imports from the confirmed-unavailable list."
    )
    accelerator = accelerator_inventory()
    if accelerator.get("cuda") and accelerator.get("devices"):
        gib = (accelerator.get("total_memory_bytes") or 0) / (1024 ** 3)
        sections.append(
            "## Accelerator\n\n"
            f"{accelerator['devices']} CUDA device(s) visible to the candidate: "
            f"{accelerator.get('name') or 'unnamed'}, {gib:.0f} GiB, "
            f"torch {accelerator.get('torch_version')}. This is measured inside the "
            "candidate runtime, not assumed.\n\n"
            "A method that needs one may use it. Training a model is in scope where the "
            "task calls for it and the execution timeout allows; move tensors to cuda "
            "explicitly, because torch defaults to CPU and a model left there will run "
            "far slower than the timeout assumes."
        )
    else:
        sections.append(
            "## Accelerator\n\nNo CUDA device is visible to the candidate"
            + (f" ({accelerator['error']})" if accelerator.get("error") else "")
            + ". Choose methods that are feasible on CPU within the execution timeout."
        )
    inventory = offline_model_inventory(MODEL_CACHE)
    if any(inventory.values()):
        lines = ["## Frozen offline model inventory", "",
                 "The candidate sandbox has no network. Only these pretrained assets exist, "
                 "read-only under /model_cache (TORCH_HOME=/model_cache, "
                 "HF_HOME=TRANSFORMERS_CACHE=/model_cache/transformers, offline mode on). "
                 "Use only these; a pretrained loader for anything unlisted tries a download "
                 "and fails."]
        if inventory["torchvision"]:
            lines += ["", "torchvision ImageNet weights:"] + [f"- {item}" for item in inventory["torchvision"]]
        if inventory["huggingface"]:
            lines += ["", "Hugging Face repos (from_pretrained with the repo id):"] + [
                f"- {item}" for item in inventory["huggingface"]]
        if inventory["other"]:
            lines += ["", "Other files:"] + [f"- {item}" for item in inventory["other"][:60]]
            if len(inventory["other"]) > 60:
                lines.append(f"- ... {len(inventory['other']) - 60} more")
        sections.append("\n".join(lines))
    else:
        sections.append(
            "## Frozen offline model inventory\n\nNo pretrained model cache is mounted and the "
            "sandbox has no network: pretrained weights are unavailable."
        )
    scale_profile = public_data_scale_profile(public_data)
    if compact:
        listed = []
        for item in scale_profile["files"][:40]:
            row = {key: value for key, value in item.items()
                   if key not in {"columns", "sampled_max_text_width_by_column"}}
            if "columns" in item:
                row["column_count"] = len(item["columns"])
                row["first_columns"] = item["columns"][:12]
            listed.append(row)
        scale_profile = {**scale_profile, "files": listed,
                         "truncated": scale_profile["file_count"] > len(listed)}
    feasibility = (
        f"Each complete candidate execution must finish within "
        f"{execution_timeout_seconds} seconds. "
        if execution_timeout_seconds is not None else
        "Candidate execution has a fixed controller timeout. "
    )
    if internal_validation:
        # Bounded on purpose: in R35 the unbounded wording made candidates run
        # k-fold cross-validation inside run.py, which exhausted action wall
        # budgets on the heavier tasks before a single program was scored.
        validation_instruction = (
            "## Internal validation before finish\n\n"
            "Before calling finish, write `internal_validation.json` at the workspace root: "
            '{"metric_name": <the task metric>, "direction": "maximize"|"minimize", '
            '"estimate": <number>, "protocol": <short text>}. Compute the estimate ONCE, on a '
            "single hold-out split of the public training labels, using the model run.py already "
            "trains; do not add cross-validation, repeated training, or any loop that multiplies "
            "run time. If it cannot be produced within a small fraction of the run's time, omit "
            "the file: the controller then evaluates officially as usual. The controller compares "
            "it with the parent's estimate and may skip the official evaluation when it is worse. "
            "Never read evaluation files, ground truth, or hidden labels."
        )
        if internal_validation_policy == "controller_only":
            validation_instruction = (
                "## Internal validation diagnostics\n\n"
                "If training already produces a public-label hold-out estimate, you may record it "
                "ONCE in `internal_validation.json`: "
                '{"metric_name": <the task metric>, "direction": "maximize"|"minimize", '
                '"estimate": <number>, "protocol": <short text>}. '
                "This is an optional, unattested diagnostic: it cannot reject a candidate or "
                "authorize lineage consolidation. Only controller-scored evidence governs those "
                "decisions. Do not add cross-validation or repeated training solely to produce "
                "this file; omit it when unavailable. Never read evaluation files, ground truth, "
                "or hidden labels."
            )
        sections.append(validation_instruction)
    if run_diagnostics:
        # What run.py prints on its full run is handed to the next planning and
        # coding step (first and last 2,500 characters). Make it worth reading.
        sections.append(
            "## Run diagnostics\n\n"
            "Print concise progress to stdout during the full run; the next step reads it to decide "
            "what to change. For each instance print: the public hold-out value of the task metric "
            "(and of each target or sub-metric when there are several); for iterative models the "
            "hold-out value every few epochs or rounds, so under- or over-fitting is visible; and "
            "the wall time of loading, training and prediction. Keep it under about 150 lines, print "
            "numbers rather than prose, and never print anything read from evaluation or hidden files."
        )
    if shift_aware_validation:
        # On the 2026-09-24 NB runs the hold-out was a random split while the test
        # held out whole species (64512), peptides (00619) or real instead of
        # simulated data (02870): the hold-out looked fine and the test did not.
        # Generic on purpose: it points at the public task text, never at a task.
        sections.append(
            "## Validation that resembles the test\n\n"
            "Before choosing the hold-out, read how README.md and data_description.md say the test "
            "inputs differ from the training data -- for example unseen species, organisms, "
            "peptides, proteins, molecules or scaffolds, other cell types, donors, batches, sites or "
            "time periods, or real instead of simulated data. Build the public hold-out so that it "
            "differs from the rest of the public training data in the same way (hold out whole "
            "groups by the matching key; when the test differs in several ways, pick the one the "
            "task text stresses). If the text states no difference, a random split is fine. Say in "
            "the run output which split was used and print the hold-out metric per instance. Use "
            "public training data only; never read evaluation files, ground truth or hidden labels."
        )
    if submission_fit_policy == "full_refit":
        # Without this the hold-out instruction above is the only word on training data,
        # and the natural reading of it is what every sealed neural program in the
        # MLE-Bench Lite campaign did: fit on the split's complement, validate on the
        # split, submit that same model. Five of twelve official grades came from
        # models that never saw 10-20% of the labels the terminal step handed them.
        sections.append(
            "## Submission contract\n\n"
            f"At the end of the search the controller re-executes this same run.py once more, "
            f"on the full official training set, with `{SUBMISSION_FIT_ENV}=full` in the "
            "environment; that execution's submission.csv is the one graded. The variable is "
            "absent during search. When it is set: keep the hold-out only to fix what training "
            "needs it for (the epoch or iteration count, early-stopping point, thresholds, "
            "hyperparameters), then fit the model that writes submission.csv on EVERY training "
            "row in DATA_DIR -- the hold-out rows included -- using the settings the hold-out "
            "selected. A fold ensemble that trains on every row across its folds also satisfies "
            "this. Record what was done by adding to `internal_validation.json`: "
            '"submission_fit": {"mode": "full"|"holdout_complement"|"fold_ensemble", '
            '"rows_available": <int>, "rows_fit": <int>}. '
            "When the variable is absent, behave exactly as the sections above describe. "
            "Never read evaluation files, ground truth, or hidden labels in either mode."
        )
    metric_name = log_loss_metric_name(task_root)
    if metric_name is not None:
        # Leaf e91/59bc252: carved log loss 0.005 became official 0.254. The program
        # picked a 1e-5 softmax temperature and a 1e-15 floor on a hold-out with one
        # or two rows per class and no mistakes; the unseen test paid for one error.
        sections.append(
            "## Log-loss tail risk\n\n"
            f"The metric ({metric_name}) charges -ln(p) for the true class, so a single "
            "unseen row predicted confidently wrong can cost more than every correct row "
            "together earns. A small hold-out that happens to contain no mistakes makes "
            "any sharpening look free: a lower softmax temperature, a tinier clipping floor, "
            "or post-hoc scaling toward 0/1. Choose such calibration settings by repeated "
            "splits or cross-validation over the training rows, not by one small hold-out, "
            "and never floor a predicted probability below 1e-5 before renormalizing."
        )
    if scratch_suffixes:
        # Ignored files never reach the artifact: a run.py that reads a cache it does not
        # rebuild would work during search and fail at the terminal full refit.
        sections.append(
            "## Scratch files\n\n"
            f"Files ending in {', '.join(scratch_suffixes)} are scratch: they are discarded "
            "after every coding turn and never become part of the program. run.py may write "
            "and read them within one execution, but must regenerate any it needs on every "
            "execution; never rely on one left over from an earlier run."
        )
    sections.append(
        "## Controller-measured public data scale and feasibility contract\n\n"
        + json.dumps(scale_profile, ensure_ascii=False, sort_keys=True)[:6000 if compact else 16000]
        + "\n\n"
        + feasibility
        + "Before proposing or implementing a method, estimate its dominant operation from "
        "these public sizes. Do not choose a representation, model, exhaustive primitive, "
        "cross-validation plan, or ensemble whose full-data execution lacks credible runtime "
        "headroom. If the fixed hypothesis is infeasible at this scale, return plan_blocked "
        "rather than silently substituting a different scientific method."
    )
    for path in (
        task_root / "problem/README.md",
        task_root / "problem/data_description.md",
    ):
        if path.is_file():
            sections.append(f"## {path.name}\n\n{path.read_text(errors='replace')[:18000]}")
    inventory = []
    for path in sorted(public_data.rglob("*")):
        if path.is_file():
            inventory.append(path.relative_to(public_data).as_posix())
        if len(inventory) >= 240:
            inventory.append("... inventory truncated ...")
            break
    if not (compact and not scale_profile["truncated"]):
        sections.append("## Public data inventory\n\n" + "\n".join(inventory))
    if run_diagnostics:
        preview = public_data_preview(public_data)
        if preview:
            sections.append("## Public data preview\n\nComputed by the controller: columns, shapes and a "
                            "first row, one or two files per directory and type. Use it instead of "
                            "writing inspection scripts.\n\n" + preview)
    return "\n\n".join(sections) + "\n"


def _root_scratch(run_dir: Path):
    """Scratch for one isolated starter evaluation.

    Local bubblewrap keeps the temporary directory. The container backend
    bind-mounts the workspace, so it lives under the run directory: an
    unresolved action must never lose its source under a live candidate.
    """
    if CANDIDATE_RUNTIME is None:
        return tempfile.TemporaryDirectory(prefix="swarmrsi-root-eval-")
    import uuid
    scratch = run_dir / "root_evaluation" / uuid.uuid4().hex
    scratch.mkdir(parents=True)
    return contextlib.nullcontext(str(scratch))


def evaluate_root(
    *, starter: Path, task_root: Path, run_dir: Path, timeout_seconds: int,
    reuse_evidence_path: Path | None = None,
    carve_folds: int | None = None, carve_deadline_seconds: float | None = None,
) -> tuple[float | None, bool, float, dict[str, Any], str]:
    """Evaluate an isolated copy of the starter and persist its evidence.

    With repeated carve the root is scored on the same K folds as every candidate,
    so improvements compare fold means with fold means.
    """
    evidence_path = run_dir / "root_evaluation.json"
    starter_sha256 = sha(starter / "run.py")
    if evidence_path.is_file():
        prior = json.loads(evidence_path.read_text())
        if prior.get("root_executable_sha256") != starter_sha256:
            raise ValueError("root evaluation is bound to a different starter")
        valid = bool(prior.get("valid"))
        metric = prior.get("metrics", {}).get("aggregate_improvement") if valid else None
        return (
            float(metric) if metric is not None else None,
            valid,
            float(prior.get("wall_seconds", 0.0)),
            compact_evaluation_summary(prior),
            sha(evidence_path),
        )
    if reuse_evidence_path is not None:
        reuse_evidence_path = reuse_evidence_path.resolve()
        source_manifest_path = reuse_evidence_path.parent / "manifest.json"
        if not reuse_evidence_path.is_file() or not source_manifest_path.is_file():
            raise ValueError("reused root evidence requires its source manifest")
        prior = json.loads(reuse_evidence_path.read_text())
        source_manifest = json.loads(source_manifest_path.read_text())
        hashes = source_manifest.get("source_hashes", {})
        if source_manifest.get("task_id") != task_root.name:
            raise ValueError("reused root evidence is bound to a different task")
        if prior.get("root_executable_sha256") != starter_sha256:
            raise ValueError("reused root evidence is bound to a different starter")
        if hashes.get("starter") != starter_sha256 or hashes.get("evaluator") != sha(EVALUATOR):
            raise ValueError("reused root evidence source hashes do not match this run")
        prior_folds = ((prior.get("details") or {}).get("repeated_carve") or {}).get("folds")
        if prior_folds != carve_folds:
            raise ValueError("reused root evidence was scored on a different number of carve folds")
        record = {
            **prior,
            "reused_from": str(reuse_evidence_path),
            "reused_evidence_sha256": sha(reuse_evidence_path),
        }
        atomic_json(evidence_path, record)
        valid = bool(record.get("valid"))
        metric = record.get("metrics", {}).get("aggregate_improvement") if valid else None
        return (
            float(metric) if metric is not None else None,
            valid,
            float(record.get("wall_seconds", 0.0)),
            compact_evaluation_summary(record),
            sha(evidence_path),
        )
    started = time.monotonic()
    with _root_scratch(run_dir) as temporary:
        workspace = Path(temporary) / "workspace"
        shutil.copytree(starter, workspace)
        raw_result = Path(temporary) / "result.json"
        completed = subprocess.run(
            (
                sys.executable, str(EVALUATOR),
                "--workspace", str(workspace),
                "--result", str(raw_result),
                "--task-root", str(task_root),
                "--runtime-python", str(RUNTIME),
                "--timeout-seconds", str(timeout_seconds),
                "--require-candidate-sandbox",
                *candidate_evaluator_arguments(),
                *repeated_carve_arguments(carve_folds, fold_timeout=timeout_seconds,
                                          deadline=carve_deadline_seconds or timeout_seconds),
            ),
            cwd=ROOT,
            capture_output=True,
            text=True,
            timeout=(carve_deadline_seconds if carve_folds is not None and carve_deadline_seconds
                     else timeout_seconds) + 30,
            check=False,
        )
        elapsed = time.monotonic() - started
        if raw_result.is_file():
            evidence = json.loads(raw_result.read_text())
        else:
            evidence = {
                "schema_version": "external-domain-result-v1",
                "valid": False,
                "public_diagnostic": {
                    "category": "evaluation_bridge",
                    "summary": (completed.stderr or completed.stdout or
                                f"returncode {completed.returncode}")[-2000:],
                },
            }
    record = {
        **evidence,
        "root_executable_sha256": starter_sha256,
        "wall_seconds": elapsed,
        "bridge_returncode": completed.returncode,
    }
    atomic_json(evidence_path, record)
    valid = bool(evidence.get("valid"))
    metric = evidence.get("metrics", {}).get("aggregate_improvement") if valid else None
    return (
        (float(metric) if metric is not None else None), valid, elapsed,
        compact_evaluation_summary(record), sha(evidence_path),
    )


class SessionDeadlineReached(TimeoutError):
    """The shared session deadline, not a fresh per-request timer, expired."""


class PlannerTransportFailed(RuntimeError):
    """A provider attempt ended without a response; no raw error text is retained."""

    def __init__(self, exception_type: str, resends: int, codes: dict | None = None) -> None:
        super().__init__("planner provider request failed")
        self.record = {"exception_type": exception_type, "resends": resends,
                       "provider_cost_status": "unknown", **(codes or {})}


class Planner:
    def __init__(self, model: OpenRouterCoderModel, seed: int, task_root: Path,
                 required_symbols: tuple[str, ...],
                 preserved_source_patterns: tuple[str, ...],
                 execution_timeout_seconds: int | None = None,
                 audit_root: Path | None = None, wisdom_context=None,
                 internal_validation: bool = True, cost_gate: bool = False,
                 max_transport_retries: int = 5,
                 internal_validation_policy: str = "legacy",
                 submission_fit_policy: str = "legacy",
                 execution_feedback: bool = True,
                 shift_aware_validation: bool = False) -> None:
        self.model, self.seed, self.batch = model, seed, 0
        self.execution_feedback = bool(execution_feedback)
        self.max_transport_retries = max(0, int(max_transport_retries))
        # Revision G cost gate: proposals must declare the complexity class of
        # their dominant operation and an estimated runtime; the controller
        # refuses full-pairwise methods on large public instances and estimates
        # without headroom under the execution timeout. Off by default so
        # revisions A-F reproduce byte for byte.
        self.cost_gate = bool(cost_gate)
        self.execution_timeout_seconds = execution_timeout_seconds
        self.max_public_rows: int | None = None
        if self.cost_gate:
            scale = public_data_scale_profile(task_root / "problem/data")
            rows = [int(f["rows"]) for f in scale.get("files", []) if isinstance(f.get("rows"), int)]
            self.max_public_rows = max(rows) if rows else None
        self.required_symbols = required_symbols
        self.preserved_source_patterns = preserved_source_patterns
        self.audit_root = audit_root
        self.wisdom_context = wisdom_context
        self._batch_records: list[dict[str, Any]] = []
        self.task_context = public_task_context(
            task_root, execution_timeout_seconds=execution_timeout_seconds,
            internal_validation=internal_validation,
            internal_validation_policy=internal_validation_policy,
            submission_fit_policy=submission_fit_policy,
            shift_aware_validation=shift_aware_validation,
        )[:30000]
        self.available_import_roots, self.unavailable_import_roots = (
            runtime_dependency_inventory()
        )
        metadata_path = task_root / 'metadata.json'
        self.search_metric = json.loads(metadata_path.read_text()).get('search_metric') if metadata_path.is_file() else None

    def _state_config_sha256(self) -> str:
        config = {
            "proposal_protocol": PROTOCOL_VERSION,
            "seed": self.seed,
            "task_context": self.task_context,
            "required_symbols": self.required_symbols,
            "preserved_source_patterns": self.preserved_source_patterns,
            "cost_gate": bool(getattr(self, "cost_gate", False)),
        }
        if getattr(self, 'search_metric', None) is not None:
            config['search_metric'] = self.search_metric
        wisdom = getattr(self, "wisdom_context", None)
        if wisdom is not None and wisdom.mode != "off":
            config["wisdom"] = wisdom.config
        return canonical_sha256(config)

    def checkpoint_state(self) -> dict[str, Any]:
        state = {
            "schema_version": "swarmrsi-planner-state-v1",
            "config_sha256": self._state_config_sha256(),
            "batch": self.batch,
            "batch_records": getattr(self, "_batch_records", []),
        }
        if getattr(self, "audit_root", None) is not None:
            state["transport_receipt"] = transport_snapshot(self.audit_root, self.batch)
        if getattr(self, "session_clock", None) is not None:
            state["session_clock"] = self.session_clock.snapshot()
        if getattr(self, "terminal_intent", None) is not None:
            state["terminal_intent"] = self.terminal_intent
        return state

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("terminal_intent") is not None:
            raise ValueError("terminal checkpoint cannot resume search")
        if "transport_receipt" in payload:
            verify_transport_snapshot(self.audit_root, payload["batch"], payload["transport_receipt"])
        if (
            payload.get("schema_version") != "swarmrsi-planner-state-v1"
            or payload.get("config_sha256") != self._state_config_sha256()
        ):
            raise ValueError("planner checkpoint configuration changed")
        batch = payload.get("batch")
        if type(batch) is not int or batch < 0:
            raise ValueError("planner checkpoint batch must be a nonnegative integer")
        records = payload.get("batch_records", [])
        if not isinstance(records, list) or len(records) != batch:
            raise ValueError("planner checkpoint lacks complete proposal records")
        for index, reference in enumerate(records):
            if "path" in reference:
                path = Path(reference["path"])
                audit_root = getattr(self, "audit_root", None)
                if audit_root is None or path.resolve().parent != audit_root.resolve():
                    raise ValueError("planner record is outside its audit root")
                if not path.is_file() or sha(path) != reference["sha256"]:
                    raise ValueError("planner proposal record missing or hash mismatch")
                record = json.loads(path.read_text())
                for kind in ("request", "response"):
                    source = audit_root / f"batch_{index:06d}.{kind}.json"
                    if not source.is_file() or sha(source) != record[f"{kind}_artifact_sha256"]:
                        raise ValueError("planner proposal journal missing or hash mismatch")
            else:
                record = reference["record"]
                if canonical_sha256(record) != reference["sha256"]:
                    raise ValueError("planner proposal record hash mismatch")
            if record.get("batch") != index:
                raise ValueError("planner proposal record batch mismatch")
        audit_root = getattr(self, "audit_root", None)
        expected_names = {f"batch_{i:06d}.{kind}.json" for i in range(batch)
                          for kind in ("request", "response", "record")}
        if audit_root is not None and any(
            path.name not in expected_names
            for path in audit_root.glob("batch_*.json")
        ):
            raise ValueError("uncheckpointed planner request; refusing silent model reissue")
        self._batch_records = records
        self.batch = batch

    @staticmethod
    def _migration_mutation_context(
        *, action_id: str, contract: ScientificActionContract,
        parent: ProgramNode, source_ids: tuple[str, ...],
        known: dict[str, ProgramNode], stage: int,
    ) -> ProgramMutationContext | None:
        """Bind real migrant programs to the coder, not only to the graph.

        Migration is deliberately evidence-only: source scores help the coder
        understand why a program was selected, but are never inherited by the
        destination realization.  Missing or unscored sources are protocol
        errors because a controller-issued migration packet must already be
        hash- and evaluation-bound.
        """
        if not source_ids:
            return None
        inspirations: list[MutationInspiration] = []
        lineage_evidence: list[str] = []
        for source_id in source_ids:
            source = known.get(source_id)
            if source is None:
                raise ValueError("migration source does not bind an existing program node")
            if source.native_metric is None or not math.isfinite(source.native_metric):
                raise ValueError("migration source must have a finite scientific score")
            source_code = source.artifact / "run.py"
            if not source_code.is_file() or sha(source_code) != source.executable_sha256:
                raise ValueError("migration source executable hash mismatch")
            inspirations.append(MutationInspiration(
                particle_id=source.id,
                program_sha256=source.executable_sha256,
                primary_metric=source.native_metric,
                code_excerpt=source_code.read_text(errors="replace")[-30_000:],
            ))
            lineage_evidence.append(
                f"{source.id}:program={source.executable_sha256}:"
                f"evaluation={source.evaluation_evidence_sha256}"
            )
        return ProgramMutationContext(
            request_id=f"migration_{action_id}",
            research_contract_sha256=contract.research_contract_sha256,
            kernel="cross_program_inspiration",
            stage=max(1, stage),
            annealing_lambda=0.5,
            parent_particle_id=parent.id,
            parent_program_sha256=parent.executable_sha256,
            inspirations=tuple(inspirations),
            lineage_evidence=tuple(lineage_evidence),
        )

    def _respond_with_transport_retry(self, request: dict):
        """Retry an identical request, retaining unknown usage for failed attempts.

        Completed-response accounting is unchanged. A transport failure does not
        establish zero provider cost. Write each attempt before entering transport
        and keep its outcome separately, including when a later retry succeeds.
        The optional absolute monotonic deadline must be owned by the session;
        this helper never creates or replenishes it.
        """
        budget = int(getattr(self, "max_transport_retries", 2))
        attempts = 0
        deadline = getattr(self, "session_deadline_monotonic", None)
        original_timeout = getattr(self.model, "request_total_timeout_seconds", None)
        can_bound = hasattr(self.model, "request_total_timeout_seconds")
        audit_root = getattr(self, "audit_root", None)
        request_sha = canonical_sha256(request)

        def remaining():
            return math.inf if deadline is None else deadline - time.monotonic()

        def record(kind, fields):
            if audit_root is None:
                return
            path = audit_root / "transport_attempts" / (
                f"batch_{getattr(self, 'batch', 0):06d}.attempt_{attempts:03d}.{kind}.json"
            )
            if path.exists():
                raise ValueError("planner transport attempt already exists; refusing reissue")
            atomic_json(path, {
                "schema_version": "swarmrsi-planner-transport-attempt-v1",
                "batch": getattr(self, "batch", 0), "attempt": attempts,
                "request_sha256": request_sha, "timestamp_unix": time.time(),
                **fields,
            })

        while True:
            left = remaining()
            if left <= 0:
                raise SessionDeadlineReached("session deadline reached before planner request")
            record("started", {"provider_cost_status": "unknown"})
            # Durable storage may itself be slow. Never reuse the pre-write
            # allowance to start a provider request after the session expired.
            left = remaining()
            if left <= 0:
                record("outcome", {"status": "request_not_sent",
                                   "provider_cost_status": "not_sent"})
                raise SessionDeadlineReached("session deadline reached before planner transport")
            try:
                if can_bound and deadline is not None:
                    self.model.request_total_timeout_seconds = (
                        left if original_timeout is None else min(original_timeout, left)
                    )
                result = self.model.respond(request)
            except Exception as exc:
                # Never persist raw exception strings: upstream messages can carry
                # URLs, headers or credentials. The type is sufficient attribution.
                from .coder_openrouter import http_failure_codes
                codes = http_failure_codes(exc)
                record("outcome", {"status": "transport_failed",
                                   "exception_type": type(exc).__name__,
                                   "provider_cost_status": "unknown", **codes})
                transient = ContractBoundedCoder._transient_transport_error(exc)
                if not transient:
                    if isinstance(exc, (RuntimeError, OSError)):
                        raise PlannerTransportFailed(type(exc).__name__, attempts, codes) from exc
                    raise
                if remaining() <= 0:
                    raise SessionDeadlineReached("session deadline reached during planner request") from exc
                if attempts >= budget:
                    if isinstance(exc, (RuntimeError, OSError)):
                        raise PlannerTransportFailed(type(exc).__name__, attempts) from exc
                    raise
            else:
                # Whitelist numeric usage only. Missing cost remains unknown even
                # when the provider returned a response and token counts.
                usage = result[1]
                safe_usage = safe_transport_usage(usage)
                record("outcome", {"status": "response_received", "usage": safe_usage,
                                   "provider_cost_status": "reported" if any(
                                       key in safe_usage for key in ("cost", "cost_usd")
                                   ) else "unknown"})
                return result
            finally:
                if can_bound:
                    self.model.request_total_timeout_seconds = original_timeout
            attempts += 1
            wait = transport_backoff_seconds(attempts)
            if wait >= remaining():
                raise SessionDeadlineReached("insufficient session time for planner retry backoff")
            print(f"planner transport retry {attempts}/{budget} after {wait:.0f}s", file=sys.stderr, flush=True)
            time.sleep(wait)

    def propose(
        self, *, version: str, nodes: tuple[ProgramNode, ...], view: dict,
        count: int, directives: tuple[ProposalDirective, ...] = (),
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> tuple[list[ProposedAction], dict[str, Any]]:
        known = {node.id: node for node in nodes}
        required_ids = {
            identifier
            for directive in directives
            for identifier in (directive.code_parent_id, *directive.intervention_source_ids)
        }
        ranked_nodes = sorted(
            nodes,
            key=lambda item: (
                item.id in required_ids,
                item.native_metric is not None,
                item.native_metric if item.native_metric is not None else float("-inf"),
                item.id,
            ),
            reverse=True,
        )[:16]
        summaries = []
        for rank, node in enumerate(ranked_nodes):
            code = (node.artifact / "run.py").read_text(errors="replace")[-10000:]
            # The best few nodes carry what their programs printed when run: training
            # curves, per-instance error, timing. Without it the planner sees only the
            # final score, and three reruns of one task stalled at the same value.
            log = (execution_log_excerpt(node.artifact, head=800, tail=1700)
                   if getattr(self, "execution_feedback", True) and rank < 4 else None)
            summaries.append({
                "id": node.id,
                "parent_id": node.parent_id,
                "native_metric": node.native_metric,
                "executable_sha256": node.executable_sha256,
                "hypothesis_id": node.hypothesis_id,
                "hypothesis_statement": node.hypothesis_statement,
                "scientific_family": node.scientific_family,
                "realized_family_sha256": node.realized_family_sha256,
                "family_realization_status": "declared_not_independently_verified",
                "implementation_observation": node.implementation_observation,
                "island_id": node.island_id,
                "evaluation_summary": node.evaluation_summary,
                "changed_paths": list(node.changed_paths),
                "code": code,
                **({"execution_log_excerpt": log} if log else {}),
            })
        diagnostic_summaries: list[dict[str, Any]] = []
        for row in diagnostics[-6:]:
            item = {key: value for key, value in row.items() if key != "artifact"}
            artifact = Path(str(row.get("artifact", "")))
            failed_run = artifact / "run.py"
            if is_repair_seed(row) and failed_run.is_file():
                item["failed_code"] = failed_run.read_text(errors="replace")[-9000:]
            diagnostic_summaries.append(item)
        proposal_slots = compile_proposal_slots(
            [asdict(item) for item in directives[:count]], list(diagnostics[-6:]),
        )
        cost_gate = bool(getattr(self, "cost_gate", False))
        for slot in proposal_slots:
            slot["cost_gate"] = cost_gate
        if len(proposal_slots) != count:
            raise ValueError("planner count must match controller slots")
        wisdom = getattr(self, "wisdom_context", None)
        retrieval = None
        if wisdom is not None and wisdom.mode != "off":
            query = json.dumps({"task": self.task_context, "parents": summaries,
                                "failures": diagnostic_summaries}, ensure_ascii=False, sort_keys=True)
            retrieval = wisdom.retrieve(query)
            for slot in proposal_slots:
                slot["wisdom_entry_ids"] = [e["entry_sha256"] for e in retrieval["entries"]]
        recent_events = list(view.get("events", []))[-60:]
        prompt = {
            "schema_version": "swarmrsi-scientific-planner-v3",
            "action_space_version": PROTOCOL_VERSION,
            "version": version, "seed": self.seed, "batch": self.batch,
            "task": self.task_context,
            "nodes": summaries,
            "diagnostic_failed_branches": diagnostic_summaries,
            "research_state": {
                "stream_sha256": view.get("stream_sha256"),
                "actions": view.get("actions", []),
                "recent_events": recent_events,
            },
            "proposal_slots": proposal_slots,
            "count": count,
            "instruction": (
                "Return exactly one proposal per proposal_slots entry, in the same order. The controller "
                "owns code_parent_id, island_id, and migration intervention_source_ids; do not return or "
                "change those fields. Return the slot_index and choose only its allowed_operators. "
                "diagnostic_source_action_id must be empty/omitted or one of evidence_diagnostic_ids. "
                "repair_program requires an ID in this slot's repair_diagnostic_ids; when that list "
                "is empty, repair_program is unavailable. Correcting or optimizing a valid program "
                "is improve_program, not failed-workspace repair. Each proposal needs hypothesis, "
                "description, operator, family with "
                "input_transform, primary_operator, output_construction, objective_or_test, and "
                "capacity_controls; plus predictions quality, information_gain, reuse, validity, cost. "
                "Within this batch, scientific family cores must be distinct. For a heterogeneous island, "
                "stay within that slot's scientific_role. A migration source is evidence/inspiration only: "
                "do not inherit its score, and the destination realization must be independently executed "
                "and evaluated. You may cite one diagnostic_source_action_id as hash-bound failure "
                "evidence. This citation alone never changes the code parent or seeds failed code. "
                "Use repair_program only to repair that failed implementation with its original "
                "code parent; otherwise use an appropriate scientific change operator and keep the "
                "failure as evidence only. Do not alter data, metric, output contract, seed, evaluator, or "
                "access labels. " + planner_metric_instruction(getattr(self, 'search_metric', None)) + "Use per-instance evidence, "
                "changed paths, failure diagnostics, and lineage to decide what to test; do not trust the "
                "planner's own predicted quality as measured evidence. Use only confirmed dependencies."
                + (
                    " COST DECLARATION (required): every proposal carries cost_bound = {dominant_operation, "
                    "complexity_class, estimated_seconds}. complexity_class is one of linear, n_log_n, "
                    "bounded_neighbourhood (top-k, kNN, batched, sketched, minibatch), quadratic_full (any "
                    "all-pairs matrix, dense kernel, exact assignment, full transport plan), cubic_or_worse. "
                    f"The largest public table has {getattr(self, 'max_public_rows', None)} rows and a candidate must finish "
                    f"within {getattr(self, 'execution_timeout_seconds', None)} seconds. The controller REFUSES quadratic_full or "
                    "cubic_or_worse on 20000+ rows and any estimated_seconds above half the timeout; a refused "
                    "proposal costs this slot its turn. If the mechanism you want is quadratic, propose its "
                    "bounded realization (neighbourhood, batched top-k, greedy assignment, sampling) and say so."
                    if getattr(self, "cost_gate", False) else ""
                )
            ),
        }
        if any(slot.get("research_episode") is not None for slot in proposal_slots):
            prompt["instruction"] += (
                " Some slots carry research_episode, controller-owned Bidirectional Research Search context. "
                "arm=H with mode=new_hypothesis: propose a hypothesis whose scientific family core differs from "
                "every covered_families entry and implement it on the controller-selected code parent; the "
                "listed opportunities are observed differential program behaviour you may cite as motivation. "
                "arm=H with mode=reactivate_hypothesis: the hypothesis is fixed by the controller (its id, statement "
                "and family are given); keep that mechanism, implement it on the selected code parent, use listed "
                "source programs only as typed inspiration, and say which open requirement the plan closes. "
                "arm=P: continue the program lineage as described in guidance. Requirement and coverage labels "
                "are controller bookkeeping about executed code, never scientific verdicts."
            )
        if retrieval is not None and retrieval["entries"]:
            prompt["cross_task_wisdom"] = retrieval
            prompt["instruction"] += (
                " Cross-task Wisdom contains source-bound observations and advisory precedents, never "
                "a transferable score or verdict. Independently implement and test any reused idea. "
                "Optionally list the supplied wisdom_entry_ids that informed each proposal; omit or use [] "
                "when none informed it. This is declared use, not proof of causal improvement."
            )
        request = {
            "version": "PLANNER", "seed": self.seed + self.batch, "planner_payload": prompt,
        }
        batch_index = self.batch
        audit_root = getattr(self, "audit_root", None)
        request_path = None
        if audit_root is not None:
            request_path = audit_root / f"batch_{batch_index:06d}.request.json"
            if any((audit_root / f"batch_{batch_index:06d}.{kind}.json").exists()
                   for kind in ("request", "response", "record")):
                raise ValueError("uncheckpointed planner request; refusing silent model reissue")
            atomic_json(request_path, request)
        response, usage = self._respond_with_transport_retry(request)
        # Keep an immutable JSON snapshot even for in-process model adapters.
        response = json.loads(json.dumps(response, ensure_ascii=False))
        usage = dict(usage)
        if audit_root is not None:
            response_path = audit_root / f"batch_{batch_index:06d}.response.json"
            atomic_json(response_path, {"response": response, "usage": usage})
        self.batch += 1
        raw_proposals = response.get("proposals", [])
        if not isinstance(raw_proposals, list):
            raw_proposals = []
        returned_proposals = len(raw_proposals)
        proposals: list[ProposedAction] = []
        rejected_proposals: list[dict[str, Any]] = []
        seen_family_sha256s: set[str] = set()
        diagnostics_by_id = {str(row.get("action_id")): row for row in diagnostics}
        for index, raw in enumerate(raw_proposals[:count]):
            if not isinstance(raw, dict):
                rejected_proposals.append({"index": index, "reason": "proposal_not_object"})
                continue
            if index >= len(directives):
                break
            directive = directives[index]
            parent = known.get(directive.code_parent_id)
            if parent is None:
                raise ValueError("planner slot parent is absent from controller state")
            raw = normalize_proposal(raw)
            error = proposal_legality_error(raw, proposal_slots[index])
            if not error and getattr(self, "cost_gate", False):
                error = cost_gate_error(raw, max_rows=getattr(self, "max_public_rows", None),
                                        timeout_seconds=getattr(self, "execution_timeout_seconds", None))
            if error:
                rejected_proposals.append({
                    "index": index, "reason": error,
                    "diagnostic_source_action_id": raw.get("diagnostic_source_action_id", ""),
                    "controller_code_parent_id": parent.id,
                    "allowed_operators": proposal_slots[index]["allowed_operators"],
                    "repair_diagnostic_ids": proposal_slots[index]["repair_diagnostic_ids"],
                })
                continue
            action_id = f"a{len(view.get('actions', [])) + index:04d}_{canonical_sha256(raw)[:10]}"
            family_raw = raw.get("family", {})
            if not isinstance(family_raw, dict):
                family_raw = {}
            controls = family_raw.get("capacity_controls", [])
            if isinstance(controls, str):
                controls = [controls]
            elif not isinstance(controls, (list, tuple)):
                controls = []
            family = ScientificFamily(
                input_transform=str(family_raw.get("input_transform", "unchanged"))[:500],
                primary_operator=str(family_raw.get("primary_operator", "unspecified"))[:500],
                output_construction=str(family_raw.get("output_construction", "hard labels"))[:500],
                objective_or_test=str(family_raw.get("objective_or_test", "unsupervised structure"))[:500],
                capacity_controls=tuple(str(x)[:300] for x in controls[:8]),
            )
            if family.sha256 in seen_family_sha256s:
                rejected_proposals.append({"index": index, "reason": "duplicate_family_in_batch"})
                continue
            dependencies = [ResearchDependency(parent.id, "code_parent", parent.executable_sha256)]
            for source_id in directive.intervention_source_ids:
                source = known.get(source_id)
                if source is not None and source.id != parent.id:
                    dependencies.append(ResearchDependency(
                        source.id, "intervention_source", source.executable_sha256
                    ))
            diagnostic_id = str(raw.get("diagnostic_source_action_id", ""))
            diagnostic = diagnostics_by_id.get(diagnostic_id)
            if diagnostic is not None and diagnostic.get("diagnostic_workspace_sha256"):
                dependencies.append(ResearchDependency(
                    diagnostic_id, "diagnostic_source",
                    str(diagnostic["diagnostic_workspace_sha256"]),
                ))
            pred = raw.get("predictions", {})
            if not isinstance(pred, dict):
                pred = {}
            clamp = lambda key, default: normalize_planner_score(pred.get(key), default)
            operator = ActionOperator(raw["operator"])
            if operator == ActionOperator.REPAIR_PROGRAM and (
                diagnostic is None or not is_repair_seed(diagnostic)
                or diagnostic.get("code_parent_id") != parent.id
            ):
                rejected_proposals.append({
                    "index": index, "reason": "repair_requires_original_parent_and_diagnostic",
                    "diagnostic_source_action_id": diagnostic_id,
                    "controller_code_parent_id": parent.id,
                })
                continue
            candidate = ActionCandidate(
                id=action_id, hypothesis_id=getattr(directive, "hypothesis_id", None) or f"h_{action_id}", operator=operator,
                dependencies=tuple(dependencies), family=family,
                predicted_program_quality=clamp("quality", .5),
                predicted_information_gain=clamp("information_gain", .5),
                predicted_reuse_value=clamp("reuse", .5),
                predicted_validity=clamp("validity", .7),
                expected_cost=max(.01, normalize_planner_score(pred.get("cost"), .5)),
                hypothesis_statement=str(raw.get("hypothesis", ""))[:4000],
                island_id=directive.island_id,
            )
            diagnostic_context = ""
            parent_log = (execution_log_excerpt(parent.artifact, head=2500, tail=2500)
                          if getattr(self, "execution_feedback", True) else None)
            if parent_log and diagnostic is None:
                diagnostic_context = (
                    "\n\nOutput the code parent printed on its last full run (public data only). "
                    "Read it first and state what it shows (under- or over-fitting, the weakest "
                    "instance or target, where time goes) before changing code:\n" + parent_log
                )
            if diagnostic is not None:
                diagnostic_context = (
                    ("\n\nRetained failed implementation to repair; keep its bound code parent:\n"
                     if operator == ActionOperator.REPAIR_PROGRAM else
                     "\n\nRetained failure evidence only. Start from the controller-selected valid "
                     "code parent; do not replace it with this failed implementation:\n")
                    + json.dumps({
                        key: value for key, value in diagnostic.items()
                        if key not in {"artifact", "validation"}
                    }, ensure_ascii=False, sort_keys=True)[:5000]
                )
            contract = ScientificActionContract(
                action_id=action_id,
                description=(
                    "Coder interface: inspect the read-only public_data mount and run bounded "
                    "experiments when useful. Implement only the selected hypothesis in run.py "
                    "and contract-allowed Python helper modules.\n\n"
                    + self.task_context[:8000] + "\n\n"
                    + str(raw.get("hypothesis", "")) + "\n"
                    + str(raw.get("description", ""))
                    + diagnostic_context
                )[:24000],
                allowed_write_paths=CANDIDATE_WRITE_PATHS,
                protected_paths=("TASK_CONTEXT.md",),
                required_symbols=self.required_symbols,
                forbidden_import_roots=(
                    "requests", "urllib", "subprocess", "socket",
                    *self.unavailable_import_roots,
                ),
                preserved_source_patterns=self.preserved_source_patterns,
                hypothesis_id=candidate.hypothesis_id,
                research_contract_sha256=candidate.action_sha256,
                selected_plan_json=make_selected_scientific_plan(
                    candidate, str(raw.get("description", ""))
                ),
                coder_limits=(getattr(directive, "research_episode", None) or {}).get("coder_limits"),
            )
            mutation_context = self._migration_mutation_context(
                action_id=action_id,
                contract=contract,
                parent=parent,
                source_ids=directive.intervention_source_ids,
                known=known,
                stage=self.batch,
            )
            proposals.append(ProposedAction(candidate, contract, mutation_context, index))
            seen_family_sha256s.add(family.sha256)
        for index in range(count, returned_proposals):
            rejected_proposals.append({"index": index, "reason": "extra_proposal_without_slot"})
        for index in range(returned_proposals, count):
            rejected_proposals.append({"index": index, "reason": "missing_proposal_for_slot"})
        response_sha256 = response.get("_raw_sha256")
        if not isinstance(response_sha256, str) or len(response_sha256) != 64:
            response_sha256 = canonical_sha256(response)
        diagnostic = {
                "requested_proposals": count,
                "returned_proposals": returned_proposals,
                "accepted_proposals": len(proposals),
                "rejected_proposals": rejected_proposals,
                "response_sha256": response_sha256,
                "provider_finish_reason": response.get("_provider_finish_reason"),
                "invalid_response": bool(response.get("_invalid_json"))
                or returned_proposals != count
                or len(proposals) != count,
                "parse_error": response.get("_parse_error"),
        }
        record = {
            "schema_version": "swarmrsi-planner-proposal-record-v1",
            "request_kind": "semantic_planner_payload; actual HTTP body is controller-bound in response transport metadata",
            "batch": batch_index, "request": request, "response": response,
            "usage": usage, "diagnostic": diagnostic,
            "accepted_slot_indices": [row.proposal_slot_index for row in proposals],
        }
        if retrieval is not None:
            use = [{"action_id": p.candidate.id, "slot_index": p.proposal_slot_index,
                    "declared_entry_ids": raw_proposals[p.proposal_slot_index].get("wisdom_entry_ids", [])}
                   for p in proposals]
            record["wisdom"] = {"retrieval": retrieval, "proposal_use": use,
                                "injected": bool(retrieval["entries"]), "causal_effect": "not_established"}
            diagnostic["wisdom"] = {"snapshot_sha256": retrieval["snapshot_sha256"],
                "retrieval_sha256": retrieval["retrieval_sha256"], "selected_count": retrieval["selected_count"],
                "injected": bool(retrieval["entries"]), "proposal_use": use, "authority": "advisory_only"}
        if audit_root is not None:
            record.update(request_artifact_sha256=sha(request_path),
                          response_artifact_sha256=sha(response_path))
            record_path = audit_root / f"batch_{batch_index:06d}.record.json"
            atomic_json(record_path, record)
            reference = {"path": str(record_path.resolve()), "sha256": sha(record_path)}
        else:
            reference = {"record": record, "sha256": canonical_sha256(record)}
        self._batch_records = [*getattr(self, "_batch_records", []), reference]
        # Full records never enter decision context; only their digest and
        # compact structural feedback are projected through the graph.
        diagnostic = {**diagnostic, "proposal_record_sha256": reference["sha256"],
                      "proposal_record_path": reference.get("path")}
        return proposals, {
            "model_calls": 1, **usage, "planner_diagnostic": diagnostic,
        }


class Executor:
    def __init__(self, model: OpenRouterCoderModel, run_dir: Path, seed: int,
                 task_root: Path, required_symbols: tuple[str, ...],
                 screening_limits: dict[str, Any]) -> None:
        self.model, self.run_dir, self.seed = model, run_dir, seed
        self.task_root, self.required_symbols = task_root, required_symbols
        self.screening_limits = screening_limits
        self.carve_folds = repeated_carve_folds_enabled(task_root, screening_limits)
        self.request_output_cap = getattr(model, "max_tokens_per_turn", 12000)
        self.remaining_calls = 8
        self.remaining_input = 46720
        self.remaining_output = 105376
        self.remaining_tools = 116
        self.remaining_wall = 10190.0

    def set_remaining(self, accounting: Any, limits: dict[str, Any]) -> None:
        budget = SessionLimits.from_legacy(limits, enforce_context=self.screening_limits.get("enforce_context_read_token_limit", True))
        allocation = budget.coder_allocation(accounting, self.screening_limits)
        self.remaining_calls = allocation['calls']
        self.remaining_input = allocation['input']
        self.remaining_output = allocation['output']
        self.model.max_tokens_per_turn = max(1, min(self.request_output_cap, self.remaining_output))
        self.remaining_tools = allocation['tools']
        self.remaining_wall = allocation['wall']
        clock = getattr(self, "session_clock", None)
        if clock is not None:
            self.remaining_wall = min(self.remaining_wall, clock.remaining())
            if self.remaining_wall < 1:
                raise SessionDeadlineReached("session deadline reached before action allocation")

    def _config(self, proposal: ProposedAction, parent: ProgramNode, step: int) -> CoderConfig:
        pyflakes = (str(RUNTIME), "-m", "pyflakes")
        result = ".swarmrsi/execution_validation.json"
        full_timeout = int(self.screening_limits.get("full_evaluation_timeout_seconds", 1200))
        # Repeated carve executes the candidate again on every extra fold, each under
        # the same timeout, so the scientific reserve covers K scorings.
        reserved_scientific_wall = min(
            float((getattr(self, "carve_folds", None) or 1) * full_timeout + 30), max(0.0, self.remaining_wall * 0.55)
        )
        coder_wall = max(1.0, self.remaining_wall - reserved_scientific_wall)
        finalization_reserve = self.screening_limits.get("coder_finalization_reserve_seconds")
        if finalization_reserve is not None:
            if (type(finalization_reserve) not in (int, float)
                    or not math.isfinite(finalization_reserve) or finalization_reserve <= 0):
                raise ValueError("coder finalization reserve must be finite and positive")
            # Tail actions receive less wall. Never enlarge the allocation or
            # leave a negative exploration interval; seal the effective value.
            finalization_reserve = min(finalization_reserve, coder_wall / 2)
        sandbox = coder_sandbox_config(self.task_root)
        if self.screening_limits.get("require_candidate_sandbox", True) and not sandbox["available"]:
            raise RuntimeError("candidate filesystem sandbox is required but unavailable")
        execution_timeout = int(self.screening_limits.get(
            "execution_validation_timeout_seconds",
            self.screening_limits.get("full_evaluation_timeout_seconds", 1200),
        ))
        smoke = (
            sys.executable, str(EVALUATOR), "--workspace", "{workspace}",
            "--result", "{workspace}/" + result, "--task-root", str(self.task_root),
            "--runtime-python", str(RUNTIME), "--timeout-seconds", str(execution_timeout),
            "--execution-only", "--require-candidate-sandbox",
            *candidate_evaluator_arguments(),
        )
        validation = (
            ValidationCommand("syntax", (str(RUNTIME), "-m", "compileall", "-q", "."), sandboxed=CANDIDATE_RUNTIME is not None),
            ValidationCommand("static", pyflakes + (".",), blocking=False, sandboxed=CANDIDATE_RUNTIME is not None),
            ValidationCommand(
                "import", (str(RUNTIME), "-c", "import runpy; runpy.run_path('run.py')"),
                sandboxed=True,
            ),
            ValidationCommand(
                "smoke", smoke,
                timeout_seconds=min(execution_timeout + 30, coder_wall),
                result_json_path=result,
                required_truthy_json_paths=("valid", "execution_valid"),
                max_harness_retries=int(
                    self.screening_limits.get("validation_harness_retries", 2)
                ),
            ),
        )
        return CoderConfig(
            version=CoderVersion.C3, seed=self.seed + step, model=getattr(self.model, "model", MODEL), route="openrouter",
            executable_parent="run.py", expected_parent_sha256=parent.executable_sha256,
            action=proposal.contract, validation=validation,
            mutation_context=proposal.mutation_context,
            max_model_calls=min(self.remaining_calls, (proposal.contract.coder_limits or {}).get("model_calls", self.remaining_calls)),
            max_tool_calls=min(self.remaining_tools, (proposal.contract.coder_limits or {}).get("tool_calls", self.remaining_tools)),
            max_input_tokens=self.remaining_input, max_output_tokens=self.remaining_output,
            max_wall_seconds=coder_wall,
            finalization_reserve_seconds=finalization_reserve,
            max_repairs=int(self.screening_limits.get("max_repairs_per_action", 6)),
            max_diagnostic_runs_after_checkpoint=int(self.screening_limits.get("max_diagnostic_runs_after_checkpoint", 2)),
            max_transport_retries=int(self.screening_limits.get("max_transport_retries", 8)),
            allowed_tools=(
                "list", "inspect", "search", "edit", "write", "append", "delete",
                "repair", "run", "diagnose",
            ),
            allowed_command_prefixes=((str(RUNTIME),), (sys.executable, str(EVALUATOR))),
            read_only_mounts=(("public_data", str(self.task_root / "problem/data")),),
            tool_command_wrapper=sandbox["wrapper"],
            tool_command_path_rewrites=sandbox["rewrites"],
            tool_sandbox_environment=sandbox["environment"],
            candidate_runtime=json.dumps(CANDIDATE_RUNTIME, sort_keys=True) if CANDIDATE_RUNTIME is not None else None,
            runtime_environment=(
                ("DATA_DIR", str(self.task_root / "problem/data")),
                ("OUTPUT_DIR", "{workspace}/output"),
                ("MPLBACKEND", "Agg"),
            ),
            runtime_artifact_directories=tuple(self.screening_limits.get(
                "runtime_artifact_directories", ()
            )),
            # Opt-in: top-level directories a coder's own smoke runs write to. Ignored
            # files never enter the artifact; without this a stray smoke_out/submission.csv
            # voided whole MLE actions (Random-acts a0002, Spooky a0011).
            ignored_workspace_patterns=(
                ".git/**", ".swarmrsi/**", "output/**", "**/__pycache__/**",
                ".pytest_cache/**", "*.pyc",
            ) + tuple(f"{directory}/**" for directory in self.screening_limits.get(
                "scratch_output_directories", ()
            )) + scratch_suffix_patterns(self.screening_limits),
            # Opt-in per profile: MLE sessions share this executor and keep their coder.
            write_first=bool(self.screening_limits.get("write_first_enabled", False)),
            task_context_chars=(
                int(self.screening_limits["coder_task_context_chars"])
                if self.screening_limits.get("coder_task_context_chars") is not None else None
            ),
            diff_mode=bool(self.screening_limits.get("coder_diff_mode", False)),
            no_silent_fallback=bool(self.screening_limits.get("coder_no_silent_fallback", False)),
            fresh_program_for_new_method=bool(self.screening_limits.get("coder_fresh_program_for_new_method", False)),
            max_mutation_chars=(
                int(self.screening_limits["coder_max_mutation_chars"])
                if self.screening_limits.get("coder_max_mutation_chars") is not None else None
            ),
            reasoning_off_after_truncations=(
                int(self.screening_limits["coder_reasoning_off_after_truncations"])
                if self.screening_limits.get("coder_reasoning_off_after_truncations") is not None else None
            ),
        )

    def config_sha256(self, *, proposal: ProposedAction, parent: ProgramNode, step: int) -> str:
        return self._config(proposal, parent, step).config_sha256

    def _diagnostic_seed(
        self, proposal: ProposedAction, parent: ProgramNode,
    ) -> tuple[Path | None, str | None]:
        if proposal.candidate.operator != ActionOperator.REPAIR_PROGRAM:
            return None, None
        dependencies = [
            item for item in proposal.candidate.dependencies
            if item.kind == "diagnostic_source"
        ]
        if len(dependencies) != 1:
            raise ValueError("repair action must bind exactly one diagnostic source")
        dependency = dependencies[0]
        from swarmrsi.retained_results import contained, read_json, verify_seal
        cells = contained(self.run_dir, "cells")
        matches = [
            path for path in cells.iterdir()
            if path.is_dir() and path.name.endswith("_" + dependency.source_id)
        ] if cells.is_dir() else []
        if len(matches) != 1:
            raise ValueError("repair diagnostic source cell is missing or ambiguous")
        source_cell = matches[0]
        outcome = read_json(contained(source_cell, "result.json"))
        verify_seal(outcome, "outcome_sha256")
        fields = diagnostic_artifact_fields(relpath=outcome.get("diagnostic_artifact_relpath"),
                                            digest=outcome.get("diagnostic_workspace_sha256"))
        diagnostic = contained(source_cell, outcome.get("diagnostic_artifact_relpath") or "artifact")
        if not is_repair_seed({**fields, "status": outcome["status"], "artifact": str(diagnostic),
                              "diagnostic_workspace_sha256": outcome.get("diagnostic_workspace_sha256")}):
            raise ValueError("diagnostic is evidence-only, not a retained failed repair seed")
        code_parents = [item for item in proposal.candidate.dependencies if item.kind == "code_parent"]
        if (len(code_parents) != 1 or code_parents[0].source_id != parent.id
                or code_parents[0].source_sha256 != parent.executable_sha256
                or outcome["action_id"] != dependency.source_id
                or outcome["parent_sha256"] != parent.executable_sha256
                or outcome["parent_workspace_sha256"] != parent.workspace_sha256
                or outcome.get("parent_preserved") is not True):
            raise ValueError("repair diagnostic original parent or action binding mismatch")
        if outcome["diagnostic_workspace_sha256"] != dependency.source_sha256:
            raise ValueError("repair diagnostic result hash mismatch")
        if not diagnostic.is_dir():
            raise ValueError("retained failed repair seed is missing")
        if artifact_workspace_sha256(diagnostic) != dependency.source_sha256:
            raise ValueError("repair diagnostic artifact hash mismatch")
        return diagnostic, dependency.source_sha256

    def execute(self, *, proposal: ProposedAction, parent: ProgramNode, step: int) -> ActionExecution:
        clock = getattr(self, "session_clock", None)
        if clock is not None and clock.remaining() < 1:
            raise SessionDeadlineReached("session deadline reached before coder execution")
        config = self._config(proposal, parent, step)
        cell = self.run_dir / "cells" / f"{step:03d}_{proposal.candidate.id}"
        diagnostic_seed, diagnostic_sha256 = self._diagnostic_seed(proposal, parent)
        coder = ContractBoundedCoder(config, self.model)
        if clock is not None:
            coder.session_deadline_monotonic = clock.deadline_monotonic
        outcome = coder.run(
            parent.artifact, cell,
            diagnostic_seed_root=diagnostic_seed,
            expected_diagnostic_workspace_sha256=diagnostic_sha256,
        )
        artifact = cell / "artifact"
        evidence_path = cell / "scientific_evaluation.json"
        evaluation_wall = 0.0
        score_evaluations = 0
        gate = None
        if outcome.status.value == "success" and outcome.implementation_real:
            gate = internal_validation_gate(
                artifact, parent.artifact,
                tolerance=float(self.screening_limits.get("internal_validation_tolerance", 0.0)),
                policy=self.screening_limits.get("internal_validation_policy", "legacy"),
            ) if self.screening_limits.get("internal_validation_enabled", True) else None
            if gate is not None:
                atomic_json(cell / "internal_validation_gate.json", gate)
        if gate is not None and gate["rejected"]:
            # Spend no official evaluation on a candidate that measured itself
            # below its parent on public labels; the run keeps its shots.
            evidence_path.unlink(missing_ok=True)
            atomic_json(evidence_path, {
                "schema_version": "swarmrsi-internal-validation-rejection-v1", "valid": False,
                "score_bearing_evaluation_attempted": False, "internal_validation_rejected": True,
                "internal_validation": gate,
                "public_diagnostic": {"category": "internal_validation_rejected",
                                      "summary": "candidate's public-label estimate is below its parent's"}})
        elif outcome.status.value == "success" and outcome.implementation_real:
            full_timeout = int(self.screening_limits.get("full_evaluation_timeout_seconds", 1200))
            max_retries = int(self.screening_limits.get("evaluator_harness_retries", 2))
            if not 0 <= max_retries <= 5:
                raise ValueError("evaluator_harness_retries must be in [0, 5]")
            started = time.monotonic()
            evidence: dict[str, Any] = {}
            attempt_records: list[dict[str, Any]] = []
            for attempt in range(1, max_retries + 2):
                left = clock.remaining() if clock is not None else math.inf
                if left < 1:
                    # Keep the paid coder result. Do not manufacture a scientific
                    # attempt or zero score when the evaluator was never started.
                    evidence = {"schema_version": "external-domain-result-v1",
                                "valid": False, "execution_valid": False,
                                "score_bearing_evaluation_attempted": bool(score_evaluations),
                                "controller_evaluation_attempts": len(attempt_records),
                                "controller_evaluation_attempt_records": list(attempt_records),
                                "public_diagnostic": {"category": "session_deadline",
                                    "summary": "no session time remains for another evaluator attempt"}}
                    atomic_json(evidence_path, evidence)
                    break
                attempt_timeout = min(full_timeout, int(left)) if math.isfinite(left) else full_timeout
                bridge_timeout = min(full_timeout + 30, left)
                carve_arguments: tuple[str, ...] = ()
                carve_folds = getattr(self, "carve_folds", None)
                if carve_folds is not None:
                    # Fold 0 is graded from the Coder's attested run; every extra fold
                    # executes (same timeout as that run) and grades. The bridge gets
                    # the whole envelope as a deadline and records how many folds fit.
                    fold_timeout = int(self.screening_limits.get(
                        "execution_validation_timeout_seconds", full_timeout))
                    budget = full_timeout + (carve_folds - 1) * (fold_timeout + full_timeout)
                    deadline = min(float(budget), left - 30)
                    carve_arguments = repeated_carve_arguments(
                        carve_folds, fold_timeout=fold_timeout, deadline=deadline)
                    bridge_timeout = min(max(1.0, deadline) + 30, left)
                evidence_path.unlink(missing_ok=True)
                try:
                    completed = subprocess.run(
                        (
                            sys.executable, str(EVALUATOR), "--workspace", str(artifact),
                            "--result", str(evidence_path), "--task-root", str(self.task_root),
                            "--runtime-python", str(RUNTIME), "--timeout-seconds", str(attempt_timeout),
                            "--score-existing-output",
                            "--require-candidate-sandbox",
                            *candidate_evaluator_arguments(),
                            *carve_arguments,
                        ),
                        cwd=ROOT, capture_output=True, text=True,
                        timeout=bridge_timeout, check=False,
                    )
                    if evidence_path.is_file():
                        try:
                            loaded = json.loads(evidence_path.read_text())
                            evidence = loaded if isinstance(loaded, dict) else {}
                        except (OSError, TypeError, ValueError, json.JSONDecodeError):
                            evidence = {}
                    else:
                        evidence = {}
                    if not evidence:
                        evidence = {
                            "schema_version": "external-domain-result-v1",
                            "valid": False,
                            "execution_valid": False,
                            "score_bearing_evaluation_attempted": False,
                            "public_diagnostic": {
                                "category": "evaluation_bridge",
                                "summary": (completed.stderr or completed.stdout or
                                            f"returncode {completed.returncode}")[-2000:],
                            },
                        }
                except subprocess.TimeoutExpired:
                    evidence = {
                        "schema_version": "external-domain-result-v1",
                        "valid": False,
                        "execution_valid": False,
                        # The bridge was invoked; on timeout the actual grader
                        # stage is uncertain. Conservatively consume its slot.
                        "score_bearing_evaluation_attempted": True,
                        "controller_evaluator_invoked": True,
                        "public_diagnostic": {
                            "category": "evaluation_bridge",
                            "summary": "controller evaluator bridge timed out",
                        },
                    }
                score_evaluations += int(bool(
                    evidence.get("score_bearing_evaluation_attempted")
                ))
                diagnostic = evidence.get("public_diagnostic", {})
                attempt_records.append({
                    "attempt": attempt,
                    "evidence_sha256": canonical_sha256(evidence),
                    "valid": bool(evidence.get("valid")),
                    "score_bearing_evaluation_attempted": bool(
                        evidence.get("score_bearing_evaluation_attempted")
                    ),
                    "failure_category": (
                        diagnostic.get("category") if isinstance(diagnostic, dict) else None
                    ),
                })
                evidence["controller_evaluation_attempts"] = attempt
                evidence["controller_evaluation_attempt_records"] = list(attempt_records)
                if not evaluation_evidence_retryable(evidence) or attempt > max_retries:
                    atomic_json(evidence_path, evidence)
                    break
            evaluation_wall = time.monotonic() - started
        evidence = json.loads(evidence_path.read_text()) if evidence_path.is_file() else {}
        # One published candidate is one scored evaluation. Harness retries after
        # an evaluator fault (evaluation_evidence_retryable) re-attempt the same
        # candidate; they are recorded in controller_evaluation_attempts but must
        # not count as extra submissions against the matched envelope, and the
        # research graph refuses a per-candidate count above one. G seed-1's
        # cross-modal cell died on exactly that after three attempts.
        score_evaluations = min(score_evaluations, 1)
        valid = bool(evidence.get("valid")) and outcome.status.value == "success"
        metric = evidence.get("metrics", {}).get("aggregate_improvement") if valid else None
        evaluation_sha = sha(evidence_path) if evidence_path.is_file() else None
        evaluation_summary = compact_evaluation_summary(evidence)
        return ActionExecution(
            outcome=outcome, artifact=artifact, native_metric=metric,
            artifact_valid=valid, score_evaluations=score_evaluations,
            evaluation_summary=evaluation_summary,
            evaluation_evidence_sha256=evaluation_sha,
            evaluation_evidence_path=(evidence_path if evidence_path.is_file() else None),
            evaluation_wall_seconds=evaluation_wall,
            internal_validation=gate,
        )


def strategy(name: str, seed: int, protocol: dict[str, Any] | None = None):
    if (protocol or {}).get("search") is not None:
        from .search_plugins import default_search_registry
        if name != "v1":
            raise ValueError("explicit search cannot be combined with a legacy version selector")
        if (protocol or {}).get("rsi_policy") is not None:
            raise ValueError("explicit search needs a compatible RSI policy adapter; Tree policy cannot be silently substituted")
        search = protocol["search"]
        drafts = int((protocol.get("screening_execution_limits") or {}).get("initial_root_drafts", 0))
        if drafts and search.get("name") != "bidirectional_research":
            raise ValueError("initial_root_drafts with an explicit search is supported for bidirectional_research only")
        # Not written into the spec: the run evidence compares the executed spec with
        # the policy's (random-c15-seed0 finished 55 actions and was then refused).
        return default_search_registry().create(
            search, seed=seed, session_parameters={"initial_root_drafts": drafts} if drafts else None)
    tree = configured_tree_strategy(seed=seed, config=(protocol or {}).get("tree_search"))
    if (protocol or {}).get("rsi_policy") is not None:
        from swarmrsi.tree_policy import PolicyTreeStrategy, TreePolicy
        if name != "v1":
            raise ValueError("learned Tree policies can only run on v1")
        tree = PolicyTreeStrategy(policy=TreePolicy.from_dict(protocol["rsi_policy"]), seed=seed,
                                  base_config=protocol.get("tree_search"))
    island_config = (protocol or {}).get("heterogeneous_islands", {})
    role_rows = island_config.get("roles")
    roles = None
    if role_rows is not None:
        roles = tuple(
            (str(item["id"]), str(item["scientific_role"]))
            for item in role_rows
        )
    heterogeneous = {
        "seed": seed,
        "population_size": int(island_config.get("population_size", 4)),
        "migration_interval": int(island_config.get("migration_interval", 1)),
        "migration_enabled": bool(island_config.get("migration_enabled", True)),
        "reserve_last_score_for_continuation": bool(
            island_config.get("reserve_last_score_for_continuation", False)
        ),
    }
    if roles is not None:
        heterogeneous["roles"] = roles
    drafts = int(((protocol or {}).get("screening_execution_limits") or {}).get("initial_root_drafts", 0))
    if name == "v4" and drafts:
        from .root_drafts import RootDraftingEvidenceStrategy
        return RootDraftingEvidenceStrategy(initial_root_drafts=drafts)
    if name == "v1" and drafts:
        from .root_drafts import RootDraftingContinuationTree
        from .strategies import ContinuationTreeStrategy
        if type(tree) is not ContinuationTreeStrategy:
            raise ValueError("initial_root_drafts on v1 needs tree_search.policy=continuation_tree and no rsi_policy")
        settings = {k: v for k, v in tree.config.items() if k not in ("policy", "seed")}
        return RootDraftingContinuationTree(initial_root_drafts=drafts, seed=seed, **settings)
    return {
        "v1": tree,
        "v2": MultiParentGraphStrategy(seed=seed),
        "v3": HeterogeneousMultiIslandStrategy(**heterogeneous),
        "v4": EvidenceCoupledStrategy(),
    }[name]


def target_stop_due(steps: int, minimum_steps: int, threshold: float | None, metric: float | None,
                    extra_after_target: int, first_reached_at: int | None) -> tuple[bool, int | None]:
    """Whether the target stop fires now, and the step count at which the target was first reached.

    The target counts once at least `minimum_steps` actions are done and `metric` beats `threshold`;
    the stop then waits `extra_after_target` further actions (0: stop at once).
    """
    hit = (steps >= minimum_steps and threshold is not None and metric is not None
           and metric > threshold)
    if hit and first_reached_at is None:
        first_reached_at = steps
    return hit and steps >= first_reached_at + extra_after_target, first_reached_at


def build_search_models(api_key: str, protocol: dict[str, Any],
                        screening_limits: dict[str, Any]):
    from .model_settings import request_controls
    common = {
        "api_key": api_key, "model": protocol.get("model", MODEL), "base_url": BASE_URL,
        "provider": protocol.get("provider"), "reasoning": protocol.get("reasoning"),
        "parallel_tool_calls": protocol.get("parallel_tool_calls"),
        # Optional per-model request capabilities (e.g. Opus 5.5: "auto", false).
        "tool_choice": protocol.get("tool_choice"),
        "reasoning_can_disable": protocol.get("reasoning_can_disable", True),
    }
    # The planner may reason less than the coder. Measured over 293 successful
    # batches, a planner call spent 3948 output tokens on average to emit a ~470
    # token proposal: nine tenths of it was reasoning, and output tokens were the
    # first budget to run out in 24 of 34 cells. `planner_reasoning` overrides the
    # shared setting for that role alone; absent, the shared one still applies.
    planner_common = dict(common, reasoning=protocol.get("planner_reasoning", protocol.get("reasoning")))
    return (
        OpenRouterCoderModel(
            **common, **request_controls(protocol, screening_limits, role="coder"),
            native_tools=bool(protocol.get("native_coder_tools", False)),
        ),
        OpenRouterCoderModel(
            **planner_common, **request_controls(protocol, screening_limits, role="planner"),
            native_tools=bool(protocol.get("native_planner_tools", False)),
        ),
    )


def run_session(args: argparse.Namespace) -> int:
    session_started_unix, session_started_monotonic = time.time(), time.monotonic()
    if any(value is None for value in (TASKS_ROOT, ACTIVE_TASK_ID, RUNTIME)) or (SECRET is None and SECRET_ENV is None):
        raise ValueError("Configure explicit inputs through naturebench_cli before running the session")
    protocol = json.loads(args.profile.read_text())
    model_name = protocol.get("model", MODEL)
    if not isinstance(model_name, str) or not model_name.strip():
        raise SystemExit("profile model must be a nonempty identifier")
    try:
        task = protocol["tasks"][args.task_id]
    except KeyError as exc:
        raise SystemExit(f"task {args.task_id!r} is absent from {args.profile}") from exc
    from .model_settings import request_controls
    screening_limits = {
        **protocol.get("screening_execution_limits", {}),
        **task.get("screening_execution_limits", {}),
    }
    request_controls(protocol, screening_limits, role="coder")
    planner_controls = request_controls(protocol, screening_limits, role="planner")
    task_root = TASKS_ROOT / args.task_id
    # Validate and freeze learning knobs before any provider call or root
    # evaluation. Off mode does not open a cross-task experience library.
    policy = strategy(args.version, args.seed, protocol)
    from swarmrsi.wisdom import WisdomContext
    wisdom_settings = dict(protocol.get("wisdom", {}))
    if wisdom_settings.get("snapshot_path") is not None:
        wisdom_settings["snapshot_path"] = Path(wisdom_settings["snapshot_path"])
    wisdom = WisdomContext(task_id=args.task_id, **wisdom_settings)
    profile_source_key = str(args.profile.resolve().relative_to(ROOT)) if args.profile.resolve().is_relative_to(ROOT) else str(args.profile.resolve())
    learning_context = {"schema_version": "swarmrsi-learning-context-v1",
                        "policy": protocol.get("rsi_policy"), "wisdom_mode": wisdom.mode,
                        "wisdom_snapshot_sha256": wisdom.config["snapshot_sha256"],
                        "wisdom_snapshot_ref": wisdom.snapshot_ref, "wisdom_config": wisdom.config,
                        "profile_source_key": profile_source_key}
    if protocol.get("search") is not None:
        from .search_policy import SearchPolicy, policy_from_record, search_spec_of
        from .search_plugins import SearchSpec
        executed = SearchSpec.from_dict(protocol["search"])
        learned = protocol.get("search_policy")
        if learned is None:
            learning_context["policy"] = SearchPolicy(executed).to_dict()
        else:
            parsed = policy_from_record(learned)
            if search_spec_of(parsed).to_dict() != executed.to_dict():
                raise SystemExit("learned search policy does not produce the configured search")
            learning_context["policy"] = learned
        learning_context["search_identity"] = policy.execution_identity
        learning_context["search_decision_config"] = policy.config
    starter = Path(task["starter"])
    if not starter.is_absolute():
        starter = args.profile.parent / starter
    starter = starter.resolve()
    required_symbols = tuple(task.get("required_symbols", ("main",)))
    preserved_source_patterns = tuple(task.get(
        "preserved_source_patterns", ("DATA_DIR", "OUTPUT_DIR")
    ))
    required = [starter / "run.py", EVALUATOR, task_root / "metadata.json", RUNTIME]
    if SECRET is not None:
        required.append(SECRET)
    elif SECRET_ENV not in (_CONTROLLER_ENVIRONMENT if _CONTROLLER_ENVIRONMENT is not None else os.environ):
        raise ValueError("configured controller credential environment variable is absent")
    if not all(path.exists() for path in required):
        raise SystemExit("missing frozen input: " + ", ".join(str(p) for p in required if not p.exists()))
    pyflakes_ready = runtime_probe("import pyflakes").returncode == 0
    if not pyflakes_ready:
        raise SystemExit(f"frozen runtime lacks required pyflakes module: {RUNTIME}")
    if not coder_sandbox_config(task_root)["available"]:
        raise SystemExit("configured candidate execution backend is unavailable; no model was called")
    if args.preflight:
        print(json.dumps({"ready": True, "zero_model_calls": True,
                          "task_id": args.task_id,
                          "source_hashes": {
                              str(p): sha(p) for p in required[:2]
                          }}, indent=2))
        return 0

    run_dir = args.run_dir.resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    graph_view_path = run_dir / "research_graph.html"
    if not graph_view_path.exists():
        write_dynamic_research_view(graph_view_path)
    manifest_path, checkpoint_path = run_dir / "manifest.json", run_dir / "checkpoint.json"
    limits = task["limits"]
    budget = SessionLimits.from_legacy(limits, enforce_context=screening_limits.get("enforce_context_read_token_limit", True))
    budget.validate_action_limits(screening_limits)
    clock_identity = {
        "task_id": args.task_id, "seed": args.seed, "model": model_name,
        "version": policy.name, "source_hashes": cell_source_hashes(starter, args.profile, policy),
        "learning_context": learning_context, "limits": limits,
    }
    runtime_controls = protocol.get("session_runtime_controls")
    controller_environment = _CONTROLLER_ENVIRONMENT if _CONTROLLER_ENVIRONMENT is not None else os.environ
    launch_timestamp = controller_environment.get("SWARMRSI_TRIAL_STARTED_AT_UNIX")
    clock = SessionClock(started_unix=session_started_unix,
                         started_monotonic=session_started_monotonic,
                         wall_limit=budget.limits.wall_seconds, identity=clock_identity,
                         runtime_controls=runtime_controls,
                         trial_started_unix=float(launch_timestamp) if launch_timestamp is not None else None)
    planner_output_limit = planner_controls["max_tokens_per_turn"]
    max_empty_proposal_batches = int(screening_limits.get(
        # An empty batch costs one planner call. At 2, s41592-023-02035-2 and
        # s42256-023-00619-3 each ended with three to four paid hours unused
        # after two batches refused for form. 4 keeps a stuck planner bounded.
        "max_consecutive_empty_proposal_batches", 4
    ))
    # One off-lane proposal used to end a run with most of its budget unused.
    # A refusal now retires that proposal; only a run of them ends the session.
    max_proposal_rejections = int(screening_limits.get(
        "max_consecutive_proposal_rejections", 3
    ))
    if planner_output_limit <= 0 or max_empty_proposal_batches <= 0 or max_proposal_rejections <= 0:
        raise SystemExit("planner output, empty-batch and rejection limits must be positive")
    api_key = secret_key()
    coder_model, planner_model = build_search_models(api_key, protocol, screening_limits)
    proposer = Planner(
        planner_model, args.seed, task_root, required_symbols,
        preserved_source_patterns,
        execution_timeout_seconds=int(screening_limits.get(
            "execution_validation_timeout_seconds",
            screening_limits.get("full_evaluation_timeout_seconds", 1200),
        )),
        audit_root=run_dir / "planner_batches",
        wisdom_context=wisdom,
        internal_validation=bool(screening_limits.get("internal_validation_enabled", True)),
        internal_validation_policy=str(screening_limits.get("internal_validation_policy", "legacy")),
        submission_fit_policy=str(screening_limits.get("submission_fit_policy", "legacy")),
        cost_gate=bool(screening_limits.get("cost_gate_enabled", False)),
        max_transport_retries=int(screening_limits.get("max_transport_retries", 8)),
        execution_feedback=bool(screening_limits.get("execution_feedback_enabled", True)),
        shift_aware_validation=bool(screening_limits.get("shift_aware_validation", False)),
    )
    executor = Executor(
        coder_model, run_dir, args.seed, task_root, required_symbols, screening_limits
    )
    version_name = policy.name
    if checkpoint_path.exists():
        if not manifest_path.is_file():
            raise SystemExit("checkpoint exists without manifest; refusing ambiguous resume")
        prior_manifest = json.loads(manifest_path.read_text())
        if (
            prior_manifest.get("schema_version") != "swarmrsi-live-run-v2"
            or prior_manifest.get("version") != version_name
            or prior_manifest.get("task_id") != args.task_id
            or prior_manifest.get("model") != model_name
        ):
            raise SystemExit("checkpoint manifest does not match requested cell")
        current_sources = cell_source_hashes(starter, args.profile, policy)
        if prior_manifest.get("learning_context") != learning_context:
            raise SystemExit("learning policy/Wisdom context changed since checkpoint")
        frozen_sources = prior_manifest.get("source_hashes", {})
        if any(frozen_sources.get(path) != digest for path, digest in current_sources.items()):
            raise SystemExit("source hash changed since checkpoint; refusing mixed-code resume")
        frozen_root_evidence = run_dir / "root_evaluation.json"
        if (
            not frozen_root_evidence.is_file()
            or sha(frozen_root_evidence)
            != prior_manifest.get("root_evaluation_evidence_sha256")
        ):
            raise SystemExit("root evaluation evidence changed since checkpoint")
        saved_checkpoint = json.loads(checkpoint_path.read_text())
        from .retained_results import verify_seal
        verify_seal(saved_checkpoint, "checkpoint_sha256")
        clock = SessionClock.restore(
            origin=prior_manifest.get("session_clock"),
            snapshot=saved_checkpoint.get("proposer_state", {}).get("session_clock"),
            wall_limit=budget.limits.wall_seconds, identity=clock_identity,
            runtime_controls=runtime_controls,
        )
        campaign = VSearchCoordinator.restore(strategy=policy, proposer=proposer,
                                              executor=executor,
                                              payload=saved_checkpoint)
    else:
        frozen = run_dir / "root"
        if frozen.exists():
            raise SystemExit("root exists without checkpoint; refusing ambiguous resume")
        if screening_limits.get("grader_readiness_gate", True):
            # Before the first model call: can the official evaluator run here at
            # all? Two paid sessions on 2026-09-22 could not score any candidate
            # (missing pyarrow, removed np.trapz) and spent hours finding out.
            from .naturebench_evaluator import grader_readiness
            readiness = grader_readiness(task_root, RUNTIME)
            atomic_json(run_dir / "grader_readiness.json", readiness)
            if not readiness["ready"]:
                raise SystemExit("official evaluator cannot run in the scoring runtime; zero model calls spent: "
                                 + "; ".join(readiness["problems"])[:1500])
        shutil.copytree(starter, frozen)
        context_path = frozen / "TASK_CONTEXT.md"
        context_options = dict(
            execution_timeout_seconds=int(screening_limits.get(
                "execution_validation_timeout_seconds",
                screening_limits.get("full_evaluation_timeout_seconds", 1200),
            )),
            internal_validation=bool(screening_limits.get("internal_validation_enabled", True)),
            internal_validation_policy=str(screening_limits.get("internal_validation_policy", "legacy")),
            submission_fit_policy=str(screening_limits.get("submission_fit_policy", "legacy")),
            run_diagnostics=bool(screening_limits.get("execution_feedback_enabled", True)),
            write_first=bool(screening_limits.get("write_first_enabled", False)),
            shift_aware_validation=bool(screening_limits.get("shift_aware_validation", False)),
        )
        if screening_limits.get("scratch_output_suffixes"):
            context_options["scratch_suffixes"] = tuple(screening_limits["scratch_output_suffixes"])
        context_text = public_task_context(task_root, **context_options)
        coder_context_cap = screening_limits.get("coder_task_context_chars")
        if coder_context_cap is not None and len(context_text) > int(coder_context_cap):
            # Only a context the coder could not receive whole is compacted; every
            # task that fits keeps exactly what it had, plus the new sections.
            context_text = public_task_context(task_root, compact=True, **context_options)
        context_path.write_text(context_text)
        root_timeout_seconds = int(task.get("root_evaluation_timeout_seconds", 3600))
        # The evaluator subprocess already reserves 30 seconds for its bridge.
        # Do not start even the root when that finite envelope no longer fits.
        root_available = clock.remaining() - 30
        if root_available < 1 and args.root_evidence is None:
            root_evidence = {
                "schema_version": "external-domain-result-v1", "valid": False,
                "execution_valid": False, "score_bearing_evaluation_attempted": False,
                "root_executable_sha256": sha(starter / "run.py"), "wall_seconds": 0.0,
                "public_diagnostic": {"category": "session_deadline",
                                      "summary": "insufficient time to start root evaluation"},
            }
            atomic_json(run_dir / "root_evaluation.json", root_evidence)
            root_metric, root_valid, root_wall = None, False, 0.0
            root_evaluation_summary = compact_evaluation_summary(root_evidence)
            root_evidence_sha = sha(run_dir / "root_evaluation.json")
        else:
            root_carve_folds = repeated_carve_folds_enabled(task_root, screening_limits)
            root_carve_deadline = None
            if root_carve_folds is not None:
                root_carve_deadline = float(root_carve_folds * root_timeout_seconds)
                if math.isfinite(root_available):
                    root_carve_deadline = max(1.0, min(root_carve_deadline, root_available))
            root_timeout_seconds = min(root_timeout_seconds, int(root_available)) if math.isfinite(root_available) else root_timeout_seconds
            root_metric, root_valid, root_wall, root_evaluation_summary, root_evidence_sha = evaluate_root(
                starter=starter,
                task_root=task_root,
                run_dir=run_dir,
                timeout_seconds=max(1, root_timeout_seconds),
                reuse_evidence_path=args.root_evidence,
                **({"carve_folds": root_carve_folds, "carve_deadline_seconds": root_carve_deadline}
                   if root_carve_folds is not None else {}),
            )
        root = ProgramNode(
            id="root", artifact=frozen, executable_sha256=sha(frozen / "run.py"),
            workspace_sha256=artifact_workspace_sha256(frozen),
            parent_id=None, hypothesis_id="h_root", native_metric=root_metric,
            artifact_valid=root_valid, realized_family_sha256=None,
            hypothesis_statement="Frozen starter program",
            evaluation_summary=root_evaluation_summary,
            evaluation_evidence_sha256=root_evidence_sha,
            evaluation_evidence_path=run_dir / "root_evaluation.json",
        )
        campaign = VSearchCoordinator(
            version=version_name, strategy=policy, proposer=proposer, executor=executor,
            root=root, max_model_calls=limits["backbone_message_ids"],
            max_score_evaluations=limits["score_bearing_evaluations"],
            max_tokens=budget.combined_tokens,
            proposal_batch_size=int(task.get(
                "proposal_batch_size", protocol.get("proposal_batch_size", 1)
            )),
        )
        # A failed generic starter is useful execution evidence, but it is not
        # a score-bearing scientific evaluation. Counting it would consume the
        # only candidate slot on comparator traces with one valid submission.
        campaign.accounting.score_evaluations = root_score_bearing_count(
            valid=root_valid, metric=root_metric
        )
        campaign.accounting.wall_seconds = root_wall
        atomic_json(manifest_path, {
            "schema_version": "swarmrsi-live-run-v2", "version": version_name,
            "seed": args.seed, "model": model_name, "task_id": args.task_id,
            "provider": protocol.get("provider"),
            "reasoning": protocol.get("reasoning"),
            "planner_reasoning": protocol.get("planner_reasoning", protocol.get("reasoning")),
            "native_coder_tools": coder_model.native_tools,
            "native_planner_tools": planner_model.native_tools,
            "screening_execution_limits": screening_limits,
            "proposal_batch_size": campaign.proposal_batch_size,
            "heterogeneous_islands": protocol.get("heterogeneous_islands"),
            "execution_scientific_evaluation_split": True,
            "public_data_read_only_mount": True,
            "failed_artifact_retention": True,
            "target": {"comparison": "gt", "threshold": score_target(task)},
            **({"search_metric": task["search_metric"]} if "search_metric" in task else {}),
            "limits": limits,
            "source_hashes": cell_source_hashes(starter, args.profile, policy),
            "learning_context": learning_context,
            "root_evaluation_evidence_sha256": root_evidence_sha,
            "started_at_unix": session_started_unix,
            "session_clock": clock.origin,
        })
        proposer.session_clock = clock
        atomic_json(checkpoint_path, campaign.checkpoint())

    proposer.session_clock = executor.session_clock = clock
    proposer.session_deadline_monotonic = clock.deadline_monotonic
    threshold = score_target(task)
    screening_action_cap = task.get(
        "screening_max_completed_actions",
        protocol.get("screening_max_completed_actions"),
    )
    if screening_action_cap is not None:
        screening_action_cap = int(screening_action_cap)
        if screening_action_cap <= 0:
            raise SystemExit("screening_max_completed_actions must be positive or absent")
    minimum_actions_before_target_stop = int(task.get(
        "minimum_completed_actions_before_target_stop",
        protocol.get("minimum_completed_actions_before_target_stop", 0),
    ))
    if minimum_actions_before_target_stop < 0:
        raise SystemExit("minimum_completed_actions_before_target_stop cannot be negative")
    target_stop_scope = str(task.get(
        "target_stop_scope", protocol.get("target_stop_scope", "overall")
    ))
    if target_stop_scope not in {"overall", "evolved"}:
        raise SystemExit("target_stop_scope must be 'overall' or 'evolved'")
    # Keep searching this many more actions after the target is first reached, so the run holds
    # spare target-beating programs in case its best is later ruled invalid. 0 = stop at once.
    extra_actions_after_target = int(task.get(
        "extra_actions_after_target", protocol.get("extra_actions_after_target", 0),
    ))
    if extra_actions_after_target < 0:
        raise SystemExit("extra_actions_after_target cannot be negative")
    target_first_reached_at = None
    stop_reason = None
    stop_details: dict[str, Any] | None = None
    consecutive_grader_environment_failures = 0
    while True:
        current_summary = campaign.summary()
        if clock.remaining() <= 0:
            stop_reason = "session_deadline_reached"
            break
        best = current_summary["best_native_metric"]
        target_metric = (
            current_summary["best_evolved_native_metric"]
            if target_stop_scope == "evolved"
            else best
        )
        target_stop, target_first_reached_at = target_stop_due(
            len(campaign.steps), minimum_actions_before_target_stop, threshold, target_metric,
            extra_actions_after_target, target_first_reached_at,
        )
        if target_stop:
            stop_reason = (
                "beat_claude_code_same_backbone_evolved"
                if target_stop_scope == "evolved" or current_summary["best_node_id"] != "root"
                else "beat_claude_code_same_backbone_root_retained"
            )
            if "search_metric" in task:
                stop_reason = "public_search_target_reached"
            break
        if screening_action_cap is not None and len(campaign.steps) >= screening_action_cap:
            stop_reason = "screening_action_cap_reached"
            break
        residual = clock.residual(budget, campaign.accounting)
        remaining_calls, remaining_evals, remaining_input, remaining_output, remaining_tools, remaining_wall = (
            residual[k] for k in ('remaining_calls', 'remaining_evals', 'remaining_input', 'remaining_output', 'remaining_tools', 'remaining_wall')
        )
        if min(
            remaining_calls, remaining_evals, remaining_input, remaining_output,
            remaining_tools, remaining_wall,
        ) <= 0:
            stop_reason = "session_deadline_reached" if clock.remaining() <= 0 else "matched_resource_envelope_exhausted"
            break
        unavailable = campaign.unavailable_action_sha256s()
        available = [a for a in campaign.actions.values()
                     if a.candidate.action_sha256 not in unavailable]
        if not available:
            expansion_ready, shortfalls, planner_cap = expansion_start_envelope(
                planner_output_limit=planner_output_limit,
                remaining_calls=remaining_calls, remaining_evals=remaining_evals,
                remaining_input=remaining_input, remaining_output=remaining_output,
                remaining_tools=remaining_tools, remaining_wall=remaining_wall,
                screening_limits=screening_limits,
                observed=(observed_action_cost(campaign.accounting)
                          if screening_limits.get("wall_floor_from_observed") else None),
            )
            if not expansion_ready:
                stop_reason = "insufficient_remaining_planner_and_action_envelope"
                stop_details = {"action_start_shortfalls_after_planner_reserve": shortfalls}
                break
            planner_model.max_tokens_per_turn = planner_cap
            try:
                added = campaign.expand()
            except SessionDeadlineReached:
                stop_reason = "session_deadline_reached"
                break
            except PlannerTransportFailed as exc:
                stop_reason = "planner_transport_failed"
                stop_details = dict(exc.record)
                break
            except ProposalRejected as exc:
                # Keep the charged planner response and its journal, not an older
                # pre-request checkpoint. No candidate execution or identity rewrite;
                # the batch registers nothing.
                rejection = campaign.record_proposal_rejection(exc, stage="expand")
                atomic_json(checkpoint_path, campaign.checkpoint())
                if campaign.consecutive_proposal_rejections >= max_proposal_rejections:
                    stop_reason = "search_proposal_rejected"
                    stop_details = {"proposal_rejection": rejection,
                                    "proposal_rejections": list(campaign.proposal_rejections)}
                    break
                continue
            atomic_json(checkpoint_path, campaign.checkpoint())
            atomic_json(run_dir / "live_summary.json", campaign.summary())
            if not campaign.consecutive_truncated_expansions:
                planner_model.reasoning_off_next = False
            if added == 0:
                if campaign.consecutive_truncated_expansions:
                    # The response hit the output cap mid-reasoning. Flash-class
                    # backbones do this on 14% of batches against 3% for v4-pro at
                    # the same cap, and two in a row used to end the run. Lower
                    # the planner's reasoning effort one step and retry; give up
                    # only once it has been cut off at the lowest effort too.
                    lowered = step_reasoning_down(planner_model)
                    if (not lowered and screening_limits.get("planner_reasoning_off_on_truncation")
                            and not getattr(planner_model, "reasoning_off_next", False)):
                        # Every truncated NB planner batch was reasoning that never ended
                        # (24,000 of 24,000 tokens); one batch without reasoning gets a
                        # proposal out, and the configured reasoning returns after it.
                        planner_model.reasoning_off_next = True
                        continue
                    if not lowered and campaign.consecutive_truncated_expansions >= 2:
                        stop_reason = "planner_truncated_proposal_batches"
                        stop_details = {
                            "consecutive_truncated_proposal_batches": campaign.consecutive_truncated_expansions,
                            "planner_reasoning": planner_model.reasoning,
                            "planner_reasoning_off_tried": bool(getattr(planner_model, "reasoning_off_next", False)),
                        }
                        break
                    continue
                if campaign.consecutive_empty_expansions >= max_empty_proposal_batches:
                    stop_reason = "planner_invalid_proposal_batches"
                    stop_details = {
                        "consecutive_empty_proposal_batches": (
                            campaign.consecutive_empty_expansions
                        )
                    }
                    break
                continue
        # Expansion itself consumes matched model/output budget. Recompute the
        # residual envelope and do not launch a candidate unless it can still
        # receive a bounded implementation-repair-validation cycle.
        residual = clock.residual(budget, campaign.accounting)
        remaining_calls, remaining_evals, remaining_input, remaining_output, remaining_tools, remaining_wall = (
            residual[k] for k in ('remaining_calls', 'remaining_evals', 'remaining_input', 'remaining_output', 'remaining_tools', 'remaining_wall')
        )
        action_ready, action_shortfalls = action_start_envelope(
            remaining_calls=remaining_calls,
            remaining_evals=remaining_evals,
            remaining_input=remaining_input,
            remaining_output=remaining_output,
            remaining_tools=remaining_tools,
            remaining_wall=remaining_wall,
            screening_limits=screening_limits,
            observed=observed_action_cost(campaign.accounting),
        )
        if not action_ready:
            stop_reason = "insufficient_remaining_action_envelope"
            stop_details = {"action_start_shortfalls": action_shortfalls}
            break
        try:
            executor.set_remaining(campaign.accounting, limits)
            row = campaign.step()
        except SessionDeadlineReached:
            stop_reason = "session_deadline_reached"
            break
        except PlannerTransportFailed as exc:
            stop_reason = "planner_transport_failed"
            stop_details = dict(exc.record)
            break
        except ProposalRejected as exc:
            rejection = campaign.record_proposal_rejection(exc, stage="select")
            atomic_json(checkpoint_path, campaign.checkpoint())
            if campaign.consecutive_proposal_rejections >= max_proposal_rejections:
                stop_reason = "search_proposal_rejected"
                stop_details = {"proposal_rejection": rejection,
                                "proposal_rejections": list(campaign.proposal_rejections)}
                break
            continue
        atomic_json(checkpoint_path, campaign.checkpoint())
        atomic_json(run_dir / "live_summary.json", campaign.summary())
        print(json.dumps(asdict(row), sort_keys=True), flush=True)
        diagnostic = (getattr(campaign, "last_evaluation_summary", None) or {}).get("public_diagnostic")
        if isinstance(diagnostic, dict) and diagnostic.get("category") == "grader_environment":
            consecutive_grader_environment_failures += 1
            # Two different candidates failing the same way means the scorer, not
            # the candidates. Every further action would be paid for and unscorable.
            if consecutive_grader_environment_failures >= 2:
                stop_reason = "grader_environment_failure"
                stop_details = {"consecutive_grader_environment_failures": consecutive_grader_environment_failures,
                                "public_diagnostic": diagnostic}
                break
        elif row.native_metric is not None or row.artifact_valid:
            consecutive_grader_environment_failures = 0
    final_summary = campaign.summary()
    if stop_reason == "planner_transport_failed":
        stop_details["incumbent_retained"] = final_summary["best_node_id"] is not None
        if final_summary["best_node_id"] is None:
            stop_reason = "planner_transport_failed_without_incumbent"
        # Preserve the pending request's unknown-cost receipt and forbid replay
        # even if a crash interrupts the following final.json write.
        proposer.terminal_intent = {"stop_reason": stop_reason, "stop_details": stop_details}
    if stop_reason == "session_deadline_reached":
        stop_details = {"incumbent_retained": final_summary["best_node_id"] is not None}
        if final_summary["best_node_id"] is None:
            stop_reason = "session_deadline_reached_without_incumbent"
        # Seal intent together with pending requests. A crash between checkpoint
        # and final must never turn into another attempt at the provider.
        proposer.terminal_intent = {"stop_reason": stop_reason, "stop_details": stop_details}
    if stop_reason == "search_proposal_rejected":
        stop_details["incumbent_retained"] = final_summary["best_node_id"] is not None
        if final_summary["best_node_id"] is None:
            stop_reason = "search_proposal_rejected_without_incumbent"
    best_evolved = final_summary["best_evolved_native_metric"]
    terminal_checkpoint = campaign.checkpoint()
    atomic_json(checkpoint_path, terminal_checkpoint)
    final = {**final_summary, "stop_reason": stop_reason,
             "stop_details": stop_details,
             **terminal_targets(task, final_summary['best_native_metric'], best_evolved),
             "minimum_completed_actions_before_target_stop": minimum_actions_before_target_stop,
             "target_stop_scope": target_stop_scope,
             "screening_max_completed_actions": screening_action_cap,
             "finished_at_unix": time.time(),
             "session_clock": terminal_checkpoint["proposer_state"].get("session_clock"),
             "planner_transport": terminal_checkpoint["proposer_state"].get("transport_receipt")}
    atomic_json(run_dir / "final.json", final)
    try:  # Cost evidence is a by-product; it must never change a finished run's result.
        from .run_cost import summarize_run_cost
        atomic_json(run_dir / "cost_summary.json", summarize_run_cost(run_dir))
    except Exception as exc:  # noqa: BLE001
        print(json.dumps({"cost_summary_error": f"{type(exc).__name__}: {exc}"[:300]}), flush=True)
    atomic_json(run_dir / "live_summary.json", final)
    print(json.dumps({"stop_reason": stop_reason,
                      "best_native_metric": final["best_native_metric"]}, sort_keys=True))
    return 1 if stop_reason in {"search_proposal_rejected_without_incumbent",
                               "session_deadline_reached_without_incumbent",
                               "planner_transport_failed_without_incumbent"} else 0


if __name__ == "__main__":
    raise SystemExit("Use swarmrsi run-naturebench with explicit runtime inputs.")
