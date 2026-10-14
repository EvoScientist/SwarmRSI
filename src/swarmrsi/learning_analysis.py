"""Read-only RSI/Wisdom journal analysis, never an execution or promotion API."""
from __future__ import annotations

import hashlib
import ctypes
import errno
import json
import math
import os
from pathlib import Path
import tempfile
import sys
from types import SimpleNamespace

from .contracts import canonical_sha256
from .learning_evidence import verify_ref, verify_run_evidence
from .retained_results import require, verify_seal
from .rsi import RSIChain
from .tree_policy import PolicyTreeStrategy, TreePolicy
from .search_policy import SearchPolicy, policy_from_record, compiled_trial_profile
from .wisdom import verify_snapshot
from .validation_failure import verify_unavailable


class Capture:
    """Bind parsed bytes and reject changing journals instead of mixing snapshots."""
    def __init__(self):
        self.refs, self.inventories = {}, []
        self.protected_roots = set()

    def read(self, path: Path):
        require(path.is_file() and not path.is_symlink(), f"missing or symlink input: {path}")
        path = path.resolve()
        raw = path.read_bytes()
        ref = {"path": str(path), "sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw)}
        require(str(path) not in self.refs or self.refs[str(path)] == ref, "input changed during analysis")
        self.refs[str(path)] = ref
        return json.loads(raw)

    def watch(self, root: Path, pattern: str):
        paths = sorted(root.glob(pattern))
        self.inventories.append((root, pattern, paths))
        return paths

    def bind(self, value):
        if isinstance(value, dict):
            if {"path", "sha256", "bytes"}.issubset(value):
                verify_ref(value)
                key = str(Path(value["path"]).resolve())
                ref = {k: value[k] for k in ("path", "sha256", "bytes")}
                require(key not in self.refs or self.refs[key] == ref, "conflicting source reference")
                self.refs[key] = ref
            else:
                for child in value.values():
                    self.bind(child)
        elif isinstance(value, list):
            for child in value:
                self.bind(child)

    def verify(self):
        for ref in self.refs.values():
            verify_ref(ref)
        for root, pattern, paths in self.inventories:
            require(sorted(root.glob(pattern)) == paths, "journal advanced during analysis; export a new snapshot")


def _cost(value):
    require(value is None or type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "invalid reported cost")
    return value


def _report_cost(values):
    values = [_cost(value) for value in values]
    known = [value for value in values if value is not None]
    return {"reported_known_sum_usd": sum(known) if known else None,
            "reported_complete_sum_usd": sum(known) if values and len(known) == len(values) else None,
            "known_records": len(known), "unknown_records": len(values) - len(known),
            "actual_cost_usd": None, "actual_cost_null_reason": "provider_reconciled_spend_not_available",
            "scope": "controller-reported amounts; not a provider-reconciled bill"}


def _run(record, capture, runs):
    verify_run_evidence(record)
    capture.bind(record)
    key = record["run_id"]
    require(key not in runs or runs[key] == record, "conflicting copies of run evidence")
    runs[key] = record
    return key


def _runner_descriptor(root, capture):
    """Inspect original files, never resume execution or import historical code.

    An observer may have a newer/different installation. That does not permit
    substituting its sources or interpreter for the originals referenced here.
    """
    root = Path(root).resolve()
    config = capture.read(root / "runner.json")
    schema = config["schema_version"]
    installed = schema == "swarmrsi-installed-rsi-validation-runner-v1"
    require(installed or schema == "swarmrsi-rsi-validation-runner-v1", "unknown validation runner")
    source, runtime = Path(config["source_root"]), Path(config["runtime_python"])
    require(source.is_absolute() and runtime.is_absolute(), "original runner paths must be absolute")
    require(str(runtime.resolve()) == config["runtime_executable_resolved"], "runtime executable changed")
    capture.bind(config)
    original_profile = dict(capture.read(Path(config["base_profile_source"]["path"])))
    original_profile.pop("rsi_policy", None)
    require(original_profile == config["base_profile"], "runner base profile differs from original source")
    if installed:
        identity = capture.read(root / "runner_identity.json")
        require(set(identity) == {"source"} and Path(identity["source"]["path"]).resolve() == root / "runner.json",
                "runner identity path changed")
        capture.bind(identity)
        execution = config["execution"]
        require(set(execution) in (
            {"runtime_python", "tasks_root", "credential_file", "model_cache", "base_url"},
            {"runtime_python", "tasks_root", "credential_file", "credential_env", "model_cache", "base_url"}),
            "unknown installed execution fields")
        require((execution["credential_file"] is None) != (execution.get("credential_env") is None),
                "installed runner credential reference must be file or env name")
        for key in ("runtime_python", "tasks_root", "credential_file", "model_cache"):
            value = execution.get(key)
            require(value is None or isinstance(value, str) and Path(value).is_absolute(),
                    "original installed execution paths must be absolute")
        for key, executable in (("orchestrator_executable_ref", runtime),
                                ("candidate_executable_ref", Path(execution["runtime_python"]))):
            require(Path(config[key]["path"]).resolve() == executable.resolve(), "executable reference path changed")
        # Installed runners bind the whole original Python package. A newer
        # observer never needs to import it to prove its unchanged membership.
        package = source / "swarmrsi"
        expected = {Path(ref["path"]).resolve() for ref in config["source_refs"]
                    if Path(ref["path"]).resolve().is_relative_to(package.resolve())}
        actual = set(capture.watch(package, "**/*.py"))
        require(actual == expected and package / "naturebench_cli.py" in expected,
                "original installed package membership changed")
    capture.protected_roots.update((str(root), str(source.resolve())))
    for task in config["base_profile"].get("tasks", {}).values():
        if task.get("starter"):
            starter = Path(task["starter"])
            if not starter.is_absolute():
                starter = Path(config["base_profile_source"]["path"]).parent / starter
            capture.protected_roots.add(str(starter.resolve()))
    if installed:
        capture.protected_roots.add(str(Path(config["execution"]["tasks_root"]).resolve()))
    return SimpleNamespace(config=config, profile=config["base_profile"], runtime=runtime,
                           source_root=source, installed=installed)


def _original_command(runner, directory, identity):
    entry = ["-m", "swarmrsi.naturebench_cli"] if runner.installed else [
        str(runner.source_root / "scripts/run_swarmrsi_v_search.py")]
    command = [str(runner.runtime), *entry, "--version", "v1", "--task-id", identity["task_id"],
        "--seed", str(identity["seed"]), "--profile", str(directory / "profile.json"), "--run-dir", str(directory / "run")]
    if runner.installed:
        for key in ("runtime_python", "tasks_root", "credential_file", "credential_env", "base_url", "model_cache"):
            value = runner.config["execution"].get(key)
            if value is not None:
                command.extend(["--" + key.replace("_", "-"), value])
    return command


def _failure(record, capture):
    identity = verify_unavailable(record)
    capture.bind(record)
    trial = Path(record["trial_root"])
    runner = _runner_descriptor(trial.parent.parent, capture)
    require(set(identity) == {"runner_sha256", "policy", "task_id", "seed"}
            and {"task_id": identity["task_id"], "seed": identity["seed"]} in runner.config["cells"],
            "unavailable trial identity is outside original runner cells")
    selected = policy_from_record(identity["policy"])
    if not isinstance(selected, SearchPolicy):
        PolicyTreeStrategy(policy=selected, base_config=runner.profile.get("tree_search"))
    require(capture.read(trial / "profile.json") == compiled_trial_profile(runner.profile, selected),
            "unavailable trial profile changed")
    require(not capture.watch(trial / "run", "final.json"), "terminal appeared during analysis")
    launch = capture.read(trial / "launch.json")
    require(launch["command"] == _original_command(runner, trial, identity),
            "unavailable launch not bound to original runner")
    capture.protected_roots.add(str(trial.resolve()))
    return identity


def _proposal_transport(request, response):
    """Report unavailable legacy transport, but reject conflicting transport bytes."""
    parsed = response["response"]
    fields = {"_transport_request_body_json", "_transport_request_body_sha256", "_raw_content", "_raw_sha256"}
    if not fields.intersection(parsed):
        return "not_available"
    require(fields.issubset(parsed), "incomplete proposal transport evidence")
    for field, digest in (("_transport_request_body_json", "_transport_request_body_sha256"),
                          ("_raw_content", "_raw_sha256")):
        require(hashlib.sha256(parsed[field].encode()).hexdigest() == parsed[digest], "proposal transport hash mismatch")
    body = json.loads(parsed["_transport_request_body_json"])
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    expected = {"version": "RSI_POLICY", "seed": request["proposal_seed"], "policy_learning_request": request}
    require(len(users) == 1 and json.loads(users[0]["content"]) == expected, "proposal HTTP request mismatch")
    from .coder_openrouter import OpenRouterCoderModel
    decoder = OpenRouterCoderModel(api_key="offline-audit", model=body["model"],
        base_url="https://example.invalid", max_tokens_per_turn=1, native_tools=bool(body.get("tools")))
    decoded = decoder._decode_message(json.loads(parsed["_raw_content"]), "RSI_POLICY", parsed.get("_provider_finish_reason"))
    from .coder_openrouter import TRANSPORT_USAGE_KEYS
    require(decoded == {k: v for k, v in parsed.items()
                        if not k.startswith("_transport_request_body_") and k not in TRANSPORT_USAGE_KEYS},
            "proposal response differs from provider message")
    return "verified_original_transport_and_response"


def _chain(root, capture, runs):
    config = capture.read(root / "chain.json")
    chain = RSIChain(root, theta0=TreePolicy.from_dict(config["theta0"]), read_only=True,
        **{k: config[k] for k in ("cells", "rounds", "proposal_seed", "minimum_mean_delta",
                                "stop_solved_tasks", "targets", "base_tree_config")})
    directories = capture.watch(root, "round_*")
    journals = {}
    for directory in directories:
        require(directory.is_dir() and not directory.is_symlink(), "invalid round directory")
        require(directory.name in {f"round_{i:03d}" for i in range(config["rounds"])}, "unexpected round index")
        journals[directory.name] = {p.name: capture.read(p) for p in capture.watch(directory, "*.json")}
    completed, state = chain._finished(), chain.state()
    require(len(directories) <= len(completed) + 1, "orphaned future RSI journal")
    require([p.name for p in directories] == [f"round_{i:03d}" for i in range(len(directories))], "round journal gap")
    rows = []
    for index, directory in enumerate(directories):
        docs = journals[directory.name]
        if not docs:
            continue  # An empty directory is preparation, not an invented candidate.
        terminal = completed[index] if index < len(completed) else None
        incumbent = completed[index - 1]["retained_policy"] if index else config["theta0"]
        cells = terminal["cells"] if terminal else state["active_cells"]
        order = ["feedback.json", "proposal_request.json", "proposal_response.json", "candidate.json", "validation_request.json"]
        present = [i for i, name in enumerate(order) if name in docs]
        require(present == list(range(max(present) + 1)) if present else False, "orphaned proposal journal")
        snapshot = docs["feedback.json"]
        verify_seal(snapshot, "feedback_sha256")
        for record in snapshot["records"]:
            require(record["source_role"] != "heldout", "heldout feedback forbidden")
            _run(record, capture, runs)
        request = docs.get("proposal_request.json")
        if request is not None:
            require(request == chain._request(index, TreePolicy.from_dict(incumbent), cells, snapshot, completed[:index]),
                    "proposal request changed")
        response, candidate_record = docs.get("proposal_response.json"), docs.get("candidate.json")
        candidate = candidate_record["policy"] if candidate_record else None
        if candidate:
            policy = TreePolicy.from_dict(candidate)
            PolicyTreeStrategy(policy=policy, base_config=config["base_tree_config"])
            raw = response["response"]
            require(candidate_record["diff"] == policy.diff(TreePolicy.from_dict(incumbent))
                    and bool(candidate_record["diff"]) and raw["parameters"] == candidate["parameters"]
                    and raw["rationale"] == candidate_record["rationale"]
                    and raw["evidence_run_ids"] == candidate_record["evidence_run_ids"], "candidate provenance mismatch")
            require(bool(raw["evidence_run_ids"]) and set(raw["evidence_run_ids"]).issubset(
                {r["run_id"] for r in snapshot["records"]}), "candidate cites unknown feedback")
        if "validation_request.json" in docs:
            require(docs["validation_request.json"] == {"cells": cells, "candidate": candidate,
                "incumbent": incumbent, "utility_field": config["utility_field"]}, "validation request changed")
        received = []
        for name, record in sorted(docs.items()):
            if not name.startswith("received_"):
                continue
            require("validation_request.json" in docs, "received validation without request")
            parts = name.removesuffix(".json").split("_")
            require(len(parts) == 3 and parts[1] in {"incumbent", "candidate"} and parts[2].isdigit(), "invalid receipt name")
            role, cell_index = parts[1], int(parts[2])
            require(name == f"received_{role}_{cell_index:03d}.json" and cell_index < len(cells), "invalid receipt index")
            trusted = terminal is None or record in terminal[f"{role}_runs"]
            error = None
            try:
                require(trusted, "retained malformed return is not a verified trial measurement")
                require(record["source_role"] == "selection" and record["policy"] == (candidate if role == "candidate" else incumbent)
                        and {"task_id": record["task_id"], "seed": record["seed"]} == cells[cell_index], "received identity mismatch")
                _run(record, capture, runs)
            except (ValueError, OSError, KeyError, TypeError) as exc:
                trusted, error = False, str(exc)[:500]
            received.append({"role": role, "cell_index": cell_index, "status": "verified" if trusted else "unverified_return",
                "run_id": record["run_id"] if trusted else None, "null_reason": error,
                "source_ref": capture.refs[str((directory / name).resolve())]})
        status = terminal["status"] if terminal else ("validation_pending" if "validation_request.json" in docs else
            "proposal_response_recorded_pending_validation" if response else
            "proposal_result_unknown" if request else "feedback_recorded_not_proposed")
        trial = terminal["trial"] if terminal else None
        failure = terminal.get("validation_failure") if terminal else None
        if failure is not None:
            _failure(failure, capture)
        rows.append({"chain_id": state["chain_sha256"], "round": index,
            "candidate_id": candidate["policy_sha256"] if candidate else None,
            "candidate_id_null_reason": None if candidate else "no_registered_policy_candidate",
            "theta0_sha256": config["theta0"]["policy_sha256"], "incumbent_policy": incumbent,
            "candidate_policy": candidate, "retained_policy": terminal["retained_policy"] if terminal else None,
            "retained_policy_null_reason": None if terminal else "round_decision_pending",
            "status": status, "terminal": terminal is not None, "validation_cells": cells,
            "comparison_complete": trial is not None, "validation_failure": failure,
            "feedback_sha256": snapshot["feedback_sha256"], "feedback_run_ids": [r["run_id"] for r in snapshot["records"]],
            "visible_feedback_run_ids": [r["run_id"] for r in request["feedback"]["runs"]] if request else [],
            "feedback_omitted_runs": request["feedback"]["omitted_runs"] if request else None,
            "candidate_rationale": candidate_record["rationale"] if candidate_record else None,
            "candidate_cited_run_ids": candidate_record["evidence_run_ids"] if candidate_record else [],
            "received_validations": received, "trial": trial,
            "trial_null_reason": None if trial else "incomplete_or_invalid_pair_no_imputation",
            "decision_reason": terminal["error"] or trial["reason"] if terminal and (terminal["error"] or trial) else None,
            "proposal_usage": response["usage"] if response else None,
            "proposal_transport": _proposal_transport(request, response) if response else "not_available",
            "provenance_type": "controller_observed", "scientific_hypothesis_verdict": None,
            "source_refs": [capture.refs[str((directory / name).resolve())] for name in sorted(docs)]})
    return state, rows


def _runner(root, capture, runs):
    runner = _runner_descriptor(root, capture)
    config = runner.config
    rows = []
    for directory in capture.watch(root / "trials", "*"):
        require(directory.is_dir() and not directory.is_symlink(), "invalid trial directory")
        docs = {p.name: capture.read(p) for p in capture.watch(directory, "*.json")}
        identity = docs.get("identity.json")
        if identity is None:
            require(not docs, "trial records without original identity")
            rows.append({"runner_id": canonical_sha256(config), "trial_id": directory.name,
                "status": "preparing", "run_id": None, "identity": None, "launch_intent": False,
                "process_receipt": False, "reported_cost_usd": None, "cache_hits": 0, "source_refs": [],
                "wisdom_usage_status": "not_audited_no_verified_terminal", "wisdom_usage_batches": None})
            continue
        require(set(identity) == {"runner_sha256", "policy", "task_id", "seed"}
                and identity["runner_sha256"] == canonical_sha256(config)
                and directory.name == canonical_sha256(identity), "trial identity changed")
        require({"task_id": identity["task_id"], "seed": identity["seed"]} in config["cells"], "unplanned validation cell")
        selected = policy_from_record(identity["policy"])
        if not isinstance(selected, SearchPolicy):
            PolicyTreeStrategy(policy=selected, base_config=runner.profile.get("tree_search"))
        expected_profile = compiled_trial_profile(runner.profile, selected)
        if "profile.json" in docs:
            require(docs["profile.json"] == expected_profile, "trial profile changed")
        launch, process, run = docs.get("launch.json"), docs.get("process.json"), docs.get("evidence.json")
        if launch is not None:
            command = _original_command(runner, directory, identity)
            require(launch["runner_sha256"] == identity["runner_sha256"] and launch["command"] == command
                    and "profile.json" in docs, "launch not bound to original trial")
        require(process is None or launch is not None and process["host"] == launch["host"]
                and type(process["pid"]) is int and process["pid"] > 0, "unbound process receipt")
        require("exit.json" not in docs or launch is not None, "exit without launch")
        failure = docs.get("unavailable.json")
        if failure is not None:
            require(not run and Path(failure["trial_root"]).resolve() == directory.resolve()
                    and _failure(failure, capture) == identity, "unavailable evidence belongs to a different trial")
        if run:
            _run(run, capture, runs)
            require(run["task_id"] == identity["task_id"] and run["seed"] == identity["seed"]
                    and run["policy"] == identity["policy"] and run["source_role"] == "selection"
                    and Path(run["run_dir"]).resolve() == (directory / "run").resolve()
                    and Path(run["source_root"]).resolve() == runner.source_root
                    and run["comparison_context"]["fixed_profile"] == runner.profile, "evidence not bound to runner trial")
        cache = sorted(name for name in docs if name.startswith("cache_hit_"))
        require(cache == [f"cache_hit_{i:06d}.json" for i in range(len(cache))], "cache receipt gap")
        for name in cache:
            require(run is not None and docs[name] == {"run_id": run["run_id"],
                "evidence_sha256": run["evidence_sha256"], "additional_model_calls": 0}, "unbound cache receipt")
        status = "verified_terminal" if run else "validation_unavailable" if failure else \
            "exited_without_verified_receipt" if "exit.json" in docs else \
            "process_receipt_no_verified_terminal" if process else "launch_intent_process_unknown" if launch else "prepared_not_launched"
        rows.append({"runner_id": canonical_sha256(config), "trial_id": directory.name, "identity": identity,
            "run_id": run["run_id"] if run else None, "status": status,
            "validation_failure": failure,
            "launch_intent": launch is not None, "process_receipt": process is not None,
            "process_liveness": "not_checked", "reported_cost_usd": _cost(run["cost_usd"]) if run else None,
            "cost_null_reason": "not_reported_or_not_terminal" if run is None or run["cost_usd"] is None else None,
            "accounting": run["accounting"] if run else None, "cache_hits": len(cache),
            "wisdom_usage_status": run["planner_evidence"]["status"] if run else "not_audited_no_verified_terminal",
            "wisdom_usage_batches": len(run["planner_evidence"]["records"]) if run and
                run["planner_evidence"]["status"] == "verified_original_journals" else None,
            "source_refs": [capture.refs[str((directory / name).resolve())] for name in sorted(docs)]})
    return rows


def analyze_learning(*, chain_root: Path | None = None, runner_roots=(), wisdom_snapshot: Path | None = None, run_evidence=(),
                     _capture: Capture | None = None):
    """Build a verified snapshot. No API call, process check, execution, or input write."""
    capture, runs = _capture if _capture is not None else Capture(), {}
    require(chain_root is not None or runner_roots or wisdom_snapshot is not None or run_evidence,
            "analysis requires at least one explicit input")
    state, trials = _chain(chain_root.resolve(), capture, runs) if chain_root is not None else (None, [])
    roots = [Path(root).resolve() for root in runner_roots]
    require(len(set(roots)) == len(roots), "duplicate runner root")
    ledger = [row for root in roots for row in _runner(root, capture, runs)]
    for path in run_evidence:
        _run(capture.read(Path(path)), capture, runs)
    sources = []
    if wisdom_snapshot is not None:
        snapshot = capture.read(wisdom_snapshot)
        verify_snapshot(snapshot)
        capture.bind(snapshot)
        for record in snapshot["sources"].values():
            _run(record, capture, runs)
        sources = [{"snapshot_sha256": snapshot["snapshot_sha256"],
            **{k: entry[k] for k in ("entry_sha256", "source_task_id", "source_run_id", "source_action_id",
                                    "source_evidence_sha256", "source_role", "kind", "authority")},
            "source_ref": capture.refs[str(wisdom_snapshot.resolve())], "provenance_type": "controller_observed"}
            for entry in snapshot["entries"]]
    usage = []
    for run in runs.values():
        journal = run["planner_evidence"]
        for index, refs in enumerate(journal["records"]):
            record = capture.read(Path(refs["record"]["path"]))
            wisdom = record.get("wisdom")
            usage.append({"run_id": run["run_id"], "batch": index,
                "journal_status": journal["status"], "mode": run["comparison_context"]["wisdom_mode"],
                "snapshot_sha256": run["comparison_context"]["wisdom_snapshot_sha256"],
                "injected": wisdom["injected"] if wisdom else False,
                "retrieval": wisdom["retrieval"] if wisdom else None,
                "proposal_declared_use": wisdom["proposal_use"] if wisdom else [],
                "implemented_use": None, "implemented_use_null_reason": "requires_program_execution_analysis",
                "causal_effect": None, "causal_effect_null_reason": "injection_and_self_report_do_not_establish_effect",
                "source_refs": refs, "provenance_type": "controller_observed"})
    physical = {}
    for row in ledger:
        if row["run_id"] is not None:
            require(row["run_id"] not in physical, "one run bound to multiple physical trial identities")
            physical[row["run_id"]] = row
    run_rows = []
    for key, run in sorted(runs.items()):
        uses = [{"round": row["round"], "role": receipt["role"], "cell_index": receipt["cell_index"]}
                for row in trials for receipt in row["received_validations"] if receipt["run_id"] == key]
        run_rows.append({"run_id": key, "task_id": run["task_id"], "seed": run["seed"], "source_role": run["source_role"],
            "policy": run["policy"], "utility_field": run["utility_field"], "utility": run["utility"],
            "best_evolved_native_metric": run["best_evolved_native_metric"], "artifact_valid": run["artifact_valid"],
            "terminal_status": run["terminal_status"], "logical_validation_uses": uses,
            "physical_trial": physical.get(key), "physical_trial_null_reason": None if key in physical else "ledger_not_supplied_or_historical",
            "reported_cost_usd": _cost(run["cost_usd"]), "accounting": run["accounting"],
            "planner_evidence_status": run["planner_evidence"]["status"], "evidence_sha256": run["evidence_sha256"],
            "source_refs": run["records"], "provenance_type": "controller_observed"})
    proposal_values = [(row["proposal_usage"] or {}).get("cost_usd") for row in trials
                       if row["status"] != "feedback_recorded_not_proposed"]
    protected = {*(str(root) for root in roots), *capture.protected_roots}
    if chain_root is not None:
        protected.add(str(chain_root.resolve()))
    for record in runs.values():
        protected.update(str(Path(record[key]).resolve()) for key in ("run_dir", "source_root"))
    # Also protect source roots for prepared/failed runners, which have no run evidence.
    for root in roots:
        protected.add(str(Path(capture.read(root / "runner.json")["source_root"]).resolve()))
    if wisdom_snapshot is not None:
        protected.add(str(wisdom_snapshot.resolve().parent))
    not_audited = [{"runner_id": row["runner_id"], "trial_id": row["trial_id"],
                   "status": row["wisdom_usage_status"]} for row in ledger
                  if row["wisdom_usage_status"] != "verified_original_journals"]
    capture.verify()
    manifest = {"schema_version": "swarmrsi-learning-analysis-v1", "chain_state": state,
        "observed_round_rows": len(trials), "wisdom_source_entries": len(sources),
        "wisdom_usage_exported_terminal_batches": len(usage),
        "wisdom_usage_coverage": {"scope": "verified_terminal_run_evidence_only",
            "complete_for_supplied_ledgers": bool(ledger) and not not_audited,
            "incomplete": bool(not_audited) or not bool(ledger), "not_audited_trials": not_audited,
            "not_audited_trial_count": len(not_audited),
            "legacy_unattested_terminal_run_count": sum(run["planner_evidence"]["status"] !=
                "verified_original_journals" for run in runs.values()),
            "campaign_total_usage_batches": None,
            "null_reason": "no_campaign_wide_prefix_audit; failed_or_pending_trials_may_have_real_injections"},
        "protected_input_roots": sorted(protected),
        "source_refs": list(capture.refs.values()), "input_scope": "explicit chain, runner ledgers, snapshot and run evidence paths",
        "launch_intents": sum(row["launch_intent"] for row in ledger),
        "process_receipts": sum(row["process_receipt"] for row in ledger),
        "verified_physical_terminal_runs": len(physical), "cache_receipts": sum(row["cache_hits"] for row in ledger),
        "reported_unique_validation_cost": _report_cost([row["reported_cost_usd"] for row in ledger
                                                        if row["launch_intent"] or row["run_id"] is not None]),
        "reported_proposal_cost": _report_cost(proposal_values),
        "actual_total_cost_usd": None, "actual_total_cost_null_reason": "provider_reconciled_spend_not_available",
        "zero_model_calls": True, "zero_official_submissions": True,
        "claim_scope": "development journal analysis, not unseen transfer, scientific adjudication or demonstrated RSI/Wisdom benefit",
        "limitations": ["Pending trials do not receive a numeric result or a decision.",
            "Launch intent and process receipt do not prove provider billing or current liveness.",
            "Wisdom usage is terminal-evidence-only; pending/failed run journals are not audited here and are not zero usage.",
            "Legacy planner journals can be unavailable; no Wisdom use is invented for them.",
            "Run evidence paths must still resolve to original verified artifacts; no silent relocation."]}
    return {"manifest": manifest, "policy_trials": trials, "validation_runs": run_rows,
            "validation_ledger": ledger, "wisdom_sources": sources, "wisdom_usage": usage}


# A filesystem that rejects the no-replace flags reports one of these; an
# existing target is EEXIST and always remains a refusal, never a fallback.
NO_REPLACE_UNSUPPORTED = frozenset({errno.EINVAL, errno.ENOSYS, errno.ENOTSUP, errno.EOPNOTSUPP})
MANIFEST_NAME = "manifest.json"


def _native_no_replace(source: Path, destination: Path) -> bool:
    """One atomic no-replace rename; False means this filesystem lacks the flag.

    Linux renameat2(RENAME_NOREPLACE); macOS renamex_np(RENAME_EXCL); Windows
    rename already refuses an existing target. An existing destination raises
    FileExistsError here and is never downgraded to the fallback.
    """
    if sys.platform == "win32":
        os.rename(source, destination)
        return True
    libc = ctypes.CDLL(None, use_errno=True)
    if sys.platform == "darwin" and hasattr(libc, "renamex_np"):
        function = libc.renamex_np
        function.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_uint]
        args = (os.fsencode(source), os.fsencode(destination), 4)
    elif sys.platform.startswith("linux") and hasattr(libc, "renameat2"):
        function = libc.renameat2
        function.argtypes = [ctypes.c_int, ctypes.c_char_p, ctypes.c_int, ctypes.c_char_p, ctypes.c_uint]
        args = (-100, os.fsencode(source), -100, os.fsencode(destination), 1)
    else:
        return False
    function.restype = ctypes.c_int
    ctypes.set_errno(0)
    if function(*args) == 0:
        return True
    error = ctypes.get_errno()
    if error in NO_REPLACE_UNSUPPORTED:
        return False
    raise OSError(error, os.strerror(error), str(destination))


def _publish_no_replace(source: Path, destination: Path) -> str:
    """Publish a staged directory without ever replacing an existing target.

    Preferred: one atomic no-replace rename, so the complete bundle appears in
    a single step. Shared filesystems reject those flags (GPFS returns EINVAL),
    and there the destination name is claimed with mkdir, which is equally
    atomic against a competing writer. The payload is then moved in with
    manifest.json last, so an interrupted publication stays visibly incomplete
    instead of looking like a finished analysis. Never check-then-rename, and
    never replace an existing directory.
    """
    if _native_no_replace(source, destination):
        return "atomic_rename"
    os.mkdir(destination)
    names = sorted(path.name for path in source.iterdir())
    for name in [n for n in names if n != MANIFEST_NAME] + [n for n in names if n == MANIFEST_NAME]:
        os.replace(source / name, destination / name)
    handle = os.open(destination, os.O_RDONLY)
    try:
        os.fsync(handle)
    finally:
        os.close(handle)
    # Both publication paths end with the staging directory gone, so callers
    # never have to know which one the filesystem supported.
    os.rmdir(source)
    return "claimed_directory"


def export_learning_analysis(destination: Path, **kwargs):
    """Atomic, new-directory export; never replace a prior analysis or input."""
    destination = destination.absolute()
    input_roots = list(kwargs.get("runner_roots", ()))
    if kwargs.get("chain_root") is not None:
        input_roots.append(kwargs["chain_root"])
    for root in input_roots:
        require(not destination.resolve().is_relative_to(Path(root).resolve()), "output must be outside input journal roots")
    require(not destination.exists() and not destination.is_symlink(), "output exists; choose a new snapshot directory")
    capture = Capture()
    bundle = analyze_learning(**kwargs, _capture=capture)
    for root in bundle["manifest"]["protected_input_roots"]:
        require(not destination.resolve().is_relative_to(Path(root)), "output overlaps referenced input artifact or source root")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".learning-analysis-", dir=destination.parent) as temporary:
        staging = Path(temporary) / "bundle"
        staging.mkdir()
        outputs = {}
        for name, rows in bundle.items():
            if name == "manifest":
                continue
            raw = "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n" for row in rows).encode()
            (staging / f"{name}.jsonl").write_bytes(raw)
            outputs[f"{name}.jsonl"] = {"sha256": hashlib.sha256(raw).hexdigest(), "bytes": len(raw), "rows": len(rows)}
        manifest = {**bundle["manifest"], "outputs": outputs}
        manifest["analysis_sha256"] = canonical_sha256(manifest)
        (staging / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n")
        capture.verify()
        require(not destination.exists() and not destination.is_symlink(), "output appeared during export")
        _publish_no_replace(staging, destination)
    return manifest
