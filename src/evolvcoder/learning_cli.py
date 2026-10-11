#!/usr/bin/env python3
"""Prepare, run, resume and inspect finite Tree RSI / frozen-Wisdom experiments.

All commands except run/ablate are zero-provider operations. Actual execution
requires --execute; this is a CLI switch, not a scientific acceptance gate.
"""
from __future__ import annotations

import argparse
from copy import deepcopy
import json
import os
from pathlib import Path


from .coder_openrouter import OpenRouterCoderModel
from .contracts import canonical_sha256
from .learning_evidence import (FeedbackPool, learning_winner_skips, read_run_evidence,
    retain_learning_winners, source_ref, verify_ref, verify_run_evidence)
from .learning_trials import SubprocessValidationRunner
from .installed_learning import InstalledValidationRunner, portable_profile
from . import naturebench_session
from .rsi import validate_margin
from .naturebench_cli import configure as configure_inner, executable_path, _overlap, validate_credential_env
from .retained_results import read_json, require, select_pending_tasks, verify_seal, write_immutable
from .rsi import OpenRouterPolicyProposer, RSIChain, ValidationPending
from .tree_policy import TreePolicy
from .wisdom import WisdomLibrary
from .model_settings import request_controls


def resolve(base: Path, value: str) -> Path:
    path = Path(value)
    return (base / path).absolute() if not path.is_absolute() else path


def fixed_profile(profile: dict, config: dict, root: Path) -> dict:
    profile = deepcopy(profile)
    if config["schema_version"] == "evolvcoder-installed-learning-v1":
        profile = portable_profile(profile, Path(config["_resolved_profile"]))
    wisdom = deepcopy(config.get("wisdom", {"mode": "readonly"}))
    require(wisdom.get("mode", "readonly") in {"off", "readonly"}, "campaign Wisdom mode must be off or readonly")
    wisdom.setdefault("mode", "readonly")
    require("snapshot_path" not in wisdom, "W0 is produced from the explicit feedback sources")
    if wisdom["mode"] == "readonly":
        wisdom["snapshot_path"] = str(root / "W0.json")
    profile["wisdom"] = wisdom
    profile.pop("rsi_policy", None)
    return profile


def verify_generated(record: dict) -> None:
    verify_seal(record, "campaign_sha256")
    root = Path(record["root"])
    original = read_json(verify_ref(record["input_refs"][1]))
    require(read_json(Path(record["profile"])) == fixed_profile(original, record["requested_config"], root),
            "generated profile differs from frozen campaign inputs")
    snapshot = read_json(root / "W0.json")
    verify_seal(snapshot, "snapshot_sha256")
    require(snapshot["snapshot_sha256"] == record["wisdom_snapshot_sha256"], "generated W0 changed")
    for ref in record.get("generated_refs", []):
        verify_ref(ref)


def prepare(config_path: Path, *, allow_create: bool = True) -> dict:
    config = read_json(config_path)
    require(config.get("schema_version") in {"evolvcoder-learning-campaign-v1", "evolvcoder-installed-learning-v1"}, "unknown campaign configuration")
    installed = config["schema_version"] == "evolvcoder-installed-learning-v1"
    required = {"schema_version", "root", "profile", "runtime_python", "feedback_sources"}
    required |= {"tasks_root"} if installed else {"source_root"}
    optional = {"retained_registry", "task_ids", "seeds", "rounds", "proposal_seed", "minimum_mean_delta",
                "stop_solved_tasks", "environment", "timeout_seconds", "wisdom", "proposal_output_tokens",
                "retained_learning_results"}
    if installed:
        optional |= {"model_cache", "base_url", "credential_file", "credential_env"}
    require(required.issubset(config) and not (set(config) - required - optional), "missing/unknown campaign fields")
    if installed:
        require((config.get("credential_file") is None) != (config.get("credential_env") is None),
                "configure exactly one credential file or environment name")
        if config.get("credential_env") is not None:
            validate_credential_env(config["credential_env"])
    base = config_path.resolve().parent
    root = resolve(base, config["root"])
    profile_path = resolve(base, config["profile"])
    if installed:
        # Non-executable resource identities use canonical filesystem paths.
        # Python launchers are deliberately handled separately below.
        root, profile_path = root.resolve(), profile_path.resolve()
    profile = read_json(profile_path)
    if installed:
        config = {**config, "_resolved_profile": str(profile_path)}
        validate_margin(config.get("minimum_mean_delta", 0.))
        for name, default in (("rounds", 3), ("timeout_seconds", 14400), ("proposal_output_tokens", 8192)):
            require(type(config.get(name, default)) is int and config.get(name, default) > 0, name + " must be positive integer")
        require(type(config.get("proposal_seed", 0)) is int and config.get("proposal_seed", 0) >= 0, "invalid proposal_seed")
        require(type(config.get("stop_solved_tasks", True)) is bool, "stop_solved_tasks must be boolean")
    campaign_path = root / "campaign.json"
    require(allow_create or campaign_path.exists(), "campaign is not prepared; status does not create it")
    refs = [source_ref(config_path), source_ref(profile_path)]
    if config.get("retained_registry"):
        registry_path = resolve(base, config["retained_registry"])
        queue = select_pending_tasks(profile, read_json(registry_path))
        refs.append(source_ref(registry_path))
    else:
        queue = {"pending_task_ids": list(profile["tasks"]), "retained_registry": None,
                 "scope": "no historical retained registry supplied"}
    requested = config.get("task_ids", list(profile["tasks"]))
    require(isinstance(requested, list) and bool(requested) and len(set(requested)) == len(requested)
            and set(requested).issubset(profile["tasks"]), "unknown/duplicate task selection")
    seeds = config.get("seeds", [0])
    require(isinstance(seeds, list) and bool(seeds) and all(type(s) is int and s >= 0 for s in seeds)
            and len(set(seeds)) == len(seeds), "invalid seed set")
    learned_paths = [resolve(base, p) for p in config.get("retained_learning_results", [])]
    learned_skips = learning_winner_skips(learned_paths, profile)
    refs.extend(source_ref(p) for p in learned_paths)
    pending = [task for task in requested if task in queue["pending_task_ids"] and task not in learned_skips]
    cells = [{"task_id": task, "seed": seed} for task in pending for seed in seeds]
    if installed:
        runtime_value = config["runtime_python"]
        runtime = executable_path(resolve(base, runtime_value) if "/" in runtime_value else runtime_value)
        protected = [config_path, profile_path, naturebench_session.PACKAGE_ROOT, runtime.parent.parent,
                     resolve(base, config["tasks_root"])]
        if config.get("credential_file") is not None:
            protected.append(resolve(base, config["credential_file"]))
        protected.extend(Path(t["starter"]) for t in portable_profile(profile, profile_path)["tasks"].values())
        for source in config["feedback_sources"]:
            protected.extend(resolve(base, source[key]) for key in ("run_dir", "source_root"))
        protected.extend(learned_paths)
        if config.get("retained_registry"):
            protected.append(resolve(base, config["retained_registry"]))
        if config.get("model_cache"):
            protected.append(resolve(base, config["model_cache"]))
        require(not any(_overlap(root, p) for p in protected), "campaign output overlaps source, task, runtime or retained evidence")
        for cell in cells:
            configure_inner(argparse.Namespace(profile=profile_path, tasks_root=resolve(base, config["tasks_root"]),
                runtime_python=runtime, credential_file=resolve(base, config["credential_file"]) if config.get("credential_file") is not None else None,
                credential_env=config.get("credential_env"),
                run_dir=root / "validation/trials/input-check/run", task_id=cell["task_id"], seed=cell["seed"],
                model_cache=resolve(base, config["model_cache"]) if config.get("model_cache") else None,
                base_url=config.get("base_url", "https://openrouter.ai/api/v1"), version="v1", preflight=True))
    initial_queue = {"requested_tasks": requested, "seeds": seeds,
                    "pending_cells": cells, "historical_queue": queue,
                    "retained_learning_skips": learned_skips,
                    "skip_scope": "historical results retain their original version; no new score is assigned"}
    if campaign_path.exists():
        record = read_json(campaign_path)
        verify_seal(record, "campaign_sha256")
        require(record["requested_config"] == config, "campaign configuration changed")
        for reference in record["input_refs"]:
            verify_ref(reference)
        require(record["cells"] == cells and record["input_refs"] == refs
                and read_json(root / "initial_queue.json") == initial_queue,
                "retained results or initial queue changed")
        verify_generated(record)
        return record
    root.mkdir(parents=True, exist_ok=True)
    write_immutable(root / "initial_queue.json", initial_queue)
    pool = FeedbackPool(root / "feedback")
    library = WisdomLibrary(root / "wisdom")
    sources = []
    for source in config["feedback_sources"]:
        require(set(source).issubset({"run_dir", "source_root", "original_run_root", "expected_final_sha256", "runtime_tags"}),
                "unknown feedback source field")
        evidence = read_run_evidence(resolve(base, source["run_dir"]), resolve(base, source["source_root"]),
                    original_run_root=source.get("original_run_root"), expected_final_sha256=source.get("expected_final_sha256"),
                    source_role="development")
        pool.add(evidence)
        library.ingest(evidence, runtime_tags=tuple(source.get("runtime_tags", ["python"])))
        sources.append({"run_id": evidence["run_id"], "evidence_sha256": evidence["evidence_sha256"]})
    require(bool(sources) or not cells, "RSI requires actual prior feedback; empty feedback is not an experiment")
    snapshot_path = root / "W0.json"
    snapshot = library.freeze(snapshot_path)
    profile = fixed_profile(profile, config, root)
    write_immutable(root / "fixed_profile.json", profile)
    body = {"schema_version": "evolvcoder-learning-campaign-record-v1", "requested_config": config,
            "input_refs": refs, "root": str(root), "source_root": str(naturebench_session.ROOT if installed else resolve(base, config["source_root"])),
            "runtime_python": str(runtime if installed else resolve(base, config["runtime_python"])), "profile": str(root / "fixed_profile.json"),
            "generated_refs": [source_ref(root / name) for name in ("fixed_profile.json", "W0.json", "initial_queue.json")],
            "cells": cells, "theta0": TreePolicy().to_dict(), "feedback_sources": sources,
            "wisdom_snapshot_sha256": snapshot["snapshot_sha256"], "rounds": config.get("rounds", 3),
            "proposal_seed": config.get("proposal_seed", 0), "minimum_mean_delta": config.get("minimum_mean_delta", 0.),
            "stop_solved_tasks": config.get("stop_solved_tasks", True),
            "targets": {t: profile["tasks"][t]["claude_code_score"] for t in pending},
            "claim_scope": "development adaptation, not unseen transfer; no historical winner relabeling"}
    if installed:
        body["installed_execution"] = {name: str(resolve(base, config[name]).resolve()) if config.get(name) else None
                                       for name in ("tasks_root", "credential_file", "model_cache")}
        body["installed_execution"]["base_url"] = config.get("base_url", "https://openrouter.ai/api/v1")
        if config.get("credential_env") is not None:
            body["installed_execution"]["credential_env"] = config["credential_env"]
    record = {**body, "campaign_sha256": canonical_sha256(body)}
    write_immutable(campaign_path, record)
    return record


def rsi_entry(record: dict, *, initialize: bool = False) -> dict | None:
    """Freeze the NEXT stage's queue after a fully closed Wisdom comparison.

    Preparation does not choose this boundary. Once RSI starts, its source and
    queue stay immutable; partial pairs or later results cannot edit them.
    """
    root = Path(record["root"])
    path = root / "rsi_entry.json"
    existing = read_json(path) if path.exists() else None
    require(existing is not None or not any(root.glob("rsi*/round_*/proposal_request.json")),
            "RSI entry missing after a proposal; do not reconstruct its stage queue")
    if existing is not None:
        verify_seal(existing, "entry_sha256")
        require(existing["campaign_sha256"] == record["campaign_sha256"], "RSI entry campaign changed")
        reference = existing["completed_wisdom_comparison"]
    elif initialize:
        comparison_path = root / "wisdom_comparison.json"
        require(comparison_path.exists() or not (root / "wisdom_comparison_request.json").exists(),
                "Wisdom comparison is not closed; preserve its pending or invalid arms before starting RSI")
        reference = source_ref(comparison_path) if comparison_path.exists() else None
    else:
        return None
    skipped = {}
    if reference is not None:
        require(verify_ref(reference).resolve() == (root / "wisdom_comparison.json").resolve(),
                "RSI entry refers to a different Wisdom stage")
        result = read_json(verify_ref(reference))
        verify_seal(result, "comparison_sha256")
        expected_request = {"cells": record["cells"], "policy": record["theta0"],
                            "W0": record["wisdom_snapshot_sha256"], "utility_field": "final.best_native_metric"}
        require(result["request"] == expected_request and result["comparison"] == compare_wisdom(
            result["on_runs"], result["off_runs"], record["cells"]), "RSI prior Wisdom comparison changed")
        for run in result["on_runs"]:
            require(run["comparison_context"]["fixed_profile"] == read_json(Path(record["profile"]))
                    and run["policy"] == record["theta0"], "Wisdom stage belongs to different campaign conditions")
        if record["stop_solved_tasks"]:
            for run in result["on_runs"] + result["off_runs"]:
                score = run["best_evolved_native_metric"]
                if score is not None and score > record["targets"][run["task_id"]]:
                    previous = skipped.get(run["task_id"])
                    if previous is None or score > previous["score"]:
                        skipped[run["task_id"]] = {"run_id": run["run_id"], "evidence_sha256": run["evidence_sha256"],
                            "score": score, "node_id": run["best_evolved_node_id"], "new_rsi_score": None}
    body = {"schema_version": "evolvcoder-rsi-entry-v1", "campaign_sha256": record["campaign_sha256"],
            "completed_wisdom_comparison": reference, "skipped_completed_stage_winners": skipped,
            "cells": [c for c in record["cells"] if c["task_id"] not in skipped],
            "chain_directory": "rsi_after_wisdom" if reference is not None else "rsi"}
    derived = {**body, "entry_sha256": canonical_sha256(body)}
    if existing is not None:
        require(existing == derived, "RSI stage entry differs from original completed evidence")
    elif initialize:
        write_immutable(path, derived)
    return derived


def components(record: dict, *, execute: bool = False, read_only: bool = False, initialize_learning: bool = False):
    verify_generated(record)
    require(bool(record["cells"]), "all selected tasks already have retained winners; nothing to launch")
    root = Path(record["root"])
    config = record["requested_config"]
    profile = read_json(Path(record["profile"]))
    entry = rsi_entry(record, initialize=initialize_learning)
    cells = entry["cells"] if entry else record["cells"]
    chain_root = root / (entry["chain_directory"] if entry else "rsi")
    chain = RSIChain(chain_root, theta0=TreePolicy.from_dict(record["theta0"]), cells=cells,
        rounds=record["rounds"], proposal_seed=record["proposal_seed"], minimum_mean_delta=record["minimum_mean_delta"],
        stop_solved_tasks=record["stop_solved_tasks"], targets=record["targets"],
        base_tree_config=profile.get("tree_search"), read_only=read_only) if cells else None
    runner_root = root / "validation"
    runner_class = InstalledValidationRunner if "installed_execution" in record else SubprocessValidationRunner
    if (runner_root / "runner.json").exists():
        validator = runner_class.resume(runner_root, allow_execution=execute)
        require(validator.config["cells"] == record["cells"]
                and validator.profile == profile, "validation runner differs from the frozen campaign")
        if "installed_execution" in record:
            require(validator.config["execution"] == {**record["installed_execution"], "runtime_python": record["runtime_python"]}
                    and validator.config["timeout_seconds"] == config.get("timeout_seconds", 14400)
                    and validator.config["execution_control_root"] == str(root),
                    "installed runner differs from requested campaign execution")
    else:
        require(not read_only, "validation runner is not prepared")
        if "installed_execution" in record:
            options = dict(record["installed_execution"])
            for key in ("tasks_root", "credential_file", "model_cache"):
                options[key] = Path(options[key]) if options[key] is not None else None
        else:
            options = {"source_root": Path(record["source_root"])}
        validator = runner_class(runner_root, **options,
            profile_path=Path(record["profile"]), runtime_python=Path(record["runtime_python"]), cells=record["cells"],
            environment=config.get("environment"), timeout_seconds=config.get("timeout_seconds", 14400), allow_execution=execute)
    return chain, validator, None if read_only else FeedbackPool(root / "feedback")


def proposer(record: dict, validator: SubprocessValidationRunner):
    # Credentials are read only for an explicitly requested live outer run;
    # no header/key is written to a campaign, prompt, or accounting file.
    profile = validator.profile
    controls = request_controls(profile, {}, role="rsi",
        role_output_cap=record["requested_config"].get("proposal_output_tokens", 8192))
    key_path = (record.get("installed_execution") or {}).get("credential_file") or validator.environment.get("EVOLVCODER_OPENROUTER_SECRET")
    key_env = (record.get("installed_execution") or {}).get("credential_env")
    if key_env is not None:
        validate_credential_env(key_env)
        environment = naturebench_session._CONTROLLER_ENVIRONMENT if naturebench_session._CONTROLLER_ENVIRONMENT is not None else os.environ
        key = environment.get(key_env)
        require(isinstance(key, str) and bool(key.strip()), "configured provider credential environment variable is unavailable")
    else:
        require(key_path is not None, "set EVOLVCODER_OPENROUTER_SECRET to the existing credential file")
        secret = read_json(Path(key_path))
        key = next((secret.get(k) for k in ("api_key", "key", "OPENROUTER_API_KEY") if secret.get(k)), None)
        require(isinstance(key, str) and bool(key), "credential file does not contain an OpenRouter key")
    model = OpenRouterCoderModel(api_key=key, model=profile["model"], base_url=(record.get("installed_execution") or {}).get("base_url", "https://openrouter.ai/api/v1"),
        **controls,
        provider=profile.get("provider"), reasoning=profile.get("reasoning"), native_tools=False,
    )
    return OpenRouterPolicyProposer(model)


class _LazyPolicyProposer:
    """A recorded response resumes without reopening provider credentials."""

    def __init__(self, factory):
        self.factory, self.instance = factory, None

    def propose(self, request):
        if self.instance is None:
            self.instance = self.factory()
        return self.instance.propose(request)


def compare_wisdom(on: list[dict], off: list[dict], cells: list[dict]) -> dict:
    require(bool(cells) and len(on) == len(off) == len(cells), "incomplete Wisdom comparison")
    pairs = []
    for cell, left, right in zip(cells, on, off):
        for run in (left, right):
            verify_run_evidence(run)
            require(run["source_role"] == "selection" and run["artifact_valid"]
                    and {"task_id": run["task_id"], "seed": run["seed"]} == cell, "invalid Wisdom comparison cell")
            require(run["planner_evidence"]["status"] == "verified_original_journals", "Wisdom comparison lacks original planner journals")
        require(left["policy"] == right["policy"], "Wisdom comparison must use the same frozen policy")
        contexts = []
        for run, mode in ((left, "readonly"), (right, "off")):
            context = deepcopy(run["comparison_context"])
            require(context["wisdom_mode"] == mode, "wrong Wisdom comparison arm")
            context.pop("wisdom_mode")
            context.pop("wisdom_snapshot_sha256")
            context["fixed_profile"].pop("wisdom", None)
            contexts.append(context)
        require(contexts[0] == contexts[1], "Wisdom arms differ beyond the declared context switch")
        pairs.append({**cell, "on_run_id": left["run_id"], "off_run_id": right["run_id"],
                      "on_evidence_sha256": left["evidence_sha256"], "off_evidence_sha256": right["evidence_sha256"],
                      "on_utility": left["utility"], "off_utility": right["utility"], "delta": left["utility"] - right["utility"],
                      "wisdom_injected_batches": left["planner_evidence"]["injected_batches"],
                      "activation_observed": left["planner_evidence"]["injected_batches"] > 0})
    return {"utility_field": "final.best_native_metric", "pairs": pairs,
            "mean_paired_delta": sum(p["delta"] for p in pairs) / len(pairs),
            "claim_scope": "paired development Wisdom effect, not policy or unseen-transfer effect"}


def wisdom_ablation(record: dict, on_runner: SubprocessValidationRunner, *, execute: bool) -> dict:
    require(on_runner.profile.get("wisdom", {}).get("mode") == "readonly", "Wisdom ablation requires a frozen W0 arm")
    root = Path(record["root"])
    require(not (root / "rsi_entry.json").exists() or (root / "wisdom_comparison.json").exists(),
            "start the optional Wisdom comparison before RSI; do not reopen its fixed task stage")
    off_path = root / "no_wisdom_profile.json"
    off_profile = deepcopy(on_runner.profile)
    off_profile["wisdom"] = {k: v for k, v in off_profile["wisdom"].items() if k != "snapshot_path"}
    off_profile["wisdom"]["mode"] = "off"
    write_immutable(off_path, off_profile)
    off_root = root / "validation_no_wisdom"
    if (off_root / "runner.json").exists():
        off_runner = type(on_runner).resume(off_root, allow_execution=execute)
    else:
        # Clone the already frozen environment, including ABSENT variables.
        # Re-running constructor capture here would make an observer's newer
        # shell part of only one ablation arm.
        off_config = {**deepcopy(on_runner.config), "base_profile": off_profile, "base_profile_source": source_ref(off_path)}
        if isinstance(on_runner, InstalledValidationRunner):
            InstalledValidationRunner.write_config(off_root, off_config)
        else:
            write_immutable(off_root / "runner.json", off_config)
        off_runner = type(on_runner).resume(off_root, allow_execution=execute)
    require(off_runner.environment == on_runner.environment and off_runner.profile == off_profile
            and off_runner.config["cells"] == record["cells"], "Wisdom control configuration changed")
    policy = TreePolicy.from_dict(record["theta0"])
    request = {"cells": record["cells"], "policy": policy.to_dict(),
               "W0": record["wisdom_snapshot_sha256"], "utility_field": "final.best_native_metric"}
    write_immutable(root / "wisdom_comparison_request.json", request)
    result_path = root / "wisdom_comparison.json"
    if result_path.exists():
        result = read_json(result_path)
        verify_seal(result, "comparison_sha256")
        require(result["request"] == request, "Wisdom comparison request changed")
        require(result["comparison"] == compare_wisdom(result["on_runs"], result["off_runs"], record["cells"]),
                "Wisdom comparison differs from original runs")
        return result
    require(execute, "ablate requires --execute; no implicit provider execution")
    # Both arms are planned before either executes. A win cannot delete only
    # the other arm; completed W0/theta0 cells are later reused by RSI.
    on, off, outcomes = [], [], []
    for cell in record["cells"]:
        for label, runner, records in (("off", off_runner, off), ("on", on_runner, on)):
            try:
                run = runner.evaluate(policy=policy, **cell)
                verify_run_evidence(run)
                require(run["policy"] == policy.to_dict() and run["source_role"] == "selection"
                        and {"task_id": run["task_id"], "seed": run["seed"]} == cell,
                        "Wisdom runner returned the wrong validation cell")
                accumulate(record, [run])
                records.append(run)
                outcomes.append({**cell, "arm": label, "status": "valid", "run_id": run["run_id"]})
            except (ValueError, RuntimeError, OSError) as exc:
                pending = isinstance(exc, ValidationPending)
                outcomes.append({**cell, "arm": label, "status": "pending" if pending else "invalid",
                                 "reason": f"{type(exc).__name__}: {exc}"})
                if pending:
                    return partial_wisdom(root, request, outcomes, on, off, on_runner, off_runner)
    if len(on) != len(off) or len(on) != len(record["cells"]):
        return partial_wisdom(root, request, outcomes, on, off, on_runner, off_runner)
    comparison = compare_wisdom(on, off, record["cells"])
    body = {"schema_version": "evolvcoder-wisdom-comparison-v1", "request": request,
            "on_runs": on, "off_runs": off, "comparison": comparison,
            "on_accounting": on_runner.accounting(), "off_accounting": off_runner.accounting()}
    result = {**body, "comparison_sha256": canonical_sha256(body)}
    write_immutable(result_path, result)
    return result


def partial_wisdom(root, request, outcomes, on, off, on_runner, off_runner):
    """Preserve incomplete arms, never turn a missing score into a mean."""
    attempted = {(r["task_id"], r["seed"], r["arm"]) for r in outcomes}
    missing = [{**cell, "arm": arm, "status": "not_started"} for cell in request["cells"]
               for arm in ("off", "on") if (cell["task_id"], cell["seed"], arm) not in attempted]
    body = {"schema_version": "evolvcoder-wisdom-partial-v1", "request": request,
            "comparison": {"status": "pending" if any(r["status"] == "pending" for r in outcomes) else "incomplete",
                           "mean_paired_delta": None, "outcomes": outcomes + missing},
            "on_runs": on, "off_runs": off,
            "on_accounting": on_runner.accounting(), "off_accounting": off_runner.accounting()}
    digest = canonical_sha256(body)
    write_immutable(root / "wisdom_attempts" / f"{digest}.json", {**body, "attempt_sha256": digest})
    return body


def accumulate(record: dict, runs: list[dict]) -> dict:
    root = Path(record["root"])
    pool, library = FeedbackPool(root / "feedback"), WisdomLibrary(root / "wisdom")
    for run in runs:
        pool.add(run)
        library.ingest(run)
    retained = retain_learning_winners(runs, read_json(Path(record["profile"])), root / "retained_winners")
    body = {"schema_version": "evolvcoder-retained-learning-collection-v1",
            "winner_refs": [source_ref(p) for p in sorted((root / "retained_winners").glob("*.json"))]}
    digest = canonical_sha256(body)
    collection = root / "retained_winner_collections" / f"{digest}.json"
    write_immutable(collection, {**body, "collection_sha256": digest})
    return {"retained_run_ids": [r["run"]["run_id"] for r in retained],
            "retained_winner_collection": str(collection),
            "active_W0_unchanged": True, "wisdom_accumulation_scope": "future snapshots only"}


def add_arguments(parser):
    parser.add_argument("stage", choices=["prepare", "preflight", "run", "ablate", "status", "export"])
    parser.add_argument("--config", required=True, type=Path)
    parser.add_argument("--execute", action="store_true", help="Allow actual provider/subprocess trials for run or ablate.")


def execute(args) -> int:
    if args.execute and args.stage not in {"run", "ablate"}:
        raise ValueError("--execute is only used with run or ablate")
    record = prepare(args.config, allow_create=args.stage not in {"status", "export"})
    if args.stage == "prepare" or not record["cells"]:
        print(json.dumps({"campaign_sha256": record["campaign_sha256"], "pending_cells": record["cells"],
                          "wisdom_snapshot_sha256": record["wisdom_snapshot_sha256"], "zero_model_calls": True}))
        return 0
    entry = rsi_entry(record)
    chain_directory = entry["chain_directory"] if entry else "rsi"
    if args.stage == "status" and entry is not None and not entry["cells"]:
        print(json.dumps({"status": "all_tasks_solved_before_rsi", "rsi_trials": 0,
                          "entry": entry, "zero_model_calls": True}))
        return 0
    if args.stage == "status" and not (Path(record["root"]) / chain_directory / "chain.json").exists():
        print(json.dumps({"status": "prepared_not_started", "pending_cells": record["cells"], "zero_model_calls": True}))
        return 0
    if args.stage == "run":
        require(args.execute, "run requires --execute; use preflight for zero-call checks")
    chain, validator, pool = components(record, execute=args.execute, read_only=args.stage in {"status", "export"},
                                       initialize_learning=args.stage == "run")
    if chain is None:
        print(json.dumps({"status": "all_tasks_solved_before_rsi", "rsi_trials": 0,
                          "entry": rsi_entry(record), "scope": "Wisdom-stage wins, not evidence of RSI benefit"}))
        return 0
    if args.stage == "ablate":
        comparison = wisdom_ablation(record, validator, execute=args.execute)["comparison"]
        print(json.dumps(comparison))
        return {"pending": 3, "incomplete": 1}.get(comparison.get("status"), 0)
    if args.stage == "preflight":
        results = [validator.preflight(policy=TreePolicy.from_dict(record["theta0"]), **cell) for cell in record["cells"]]
        print(json.dumps({"ready": all(r["ready"] for r in results), "cells": results, "zero_model_calls": True}))
        return 0 if all(r["ready"] for r in results) else 1
    if args.stage == "status":
        print(json.dumps({"state": chain.state(), "validation_accounting": validator.accounting()}))
        return 0
    if args.stage == "export":
        exported = chain.export_policy(Path(record["root"]) / "frozen_policy.json")
        print(json.dumps(exported))
        return 0
    require(args.execute, "run requires --execute; use preflight for zero-call checks")
    # Completed chains return without reading credentials or constructing a provider client.
    if chain.state()["finished"] or chain.state().get("halted"):
        state = chain.state()
    else:
        if "installed_execution" in record:
            # Environment readiness precedes the first outer provider call.
            checks = [validator.preflight(policy=TreePolicy.from_dict(chain.state()["current_policy"]), **cell)
                      for cell in chain.state()["active_cells"]]
            require(all(check["ready"] for check in checks), "installed learning preflight failed; no proposal requested")
        state = chain.run(_LazyPolicyProposer(lambda: proposer(record, validator)), validator, pool)
    accumulation = accumulate(record, [r for r in pool.snapshot()["records"] if r["source_role"] == "selection"])
    if state["finished"]:
        chain.export_policy(Path(record["root"]) / "frozen_policy.json")
    print(json.dumps({"state": state, "validation_accounting": validator.accounting(), "accumulation": accumulation}))
    return 1 if state.get("halted") else 3 if state.get("status") == "validation_pending" else 0


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    return execute(parser.parse_args(argv))


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except ValidationPending as exc:
        print(json.dumps({"status": "pending", "reason": str(exc)}))
        raise SystemExit(3)
