"""Run-derived observations for RSI and Wisdom, never scientific verdicts."""
from __future__ import annotations

import ast
import difflib
import hashlib
import json
import math
from pathlib import Path
from typing import Any

from .contracts import ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily, canonical_sha256
from .retained_results import (contained, file_sha, finite, output_attestation, read_json,
                               relocate, require, verify_events, verify_seal, write_immutable)
from .campaign import artifact_workspace_sha256


def source_ref(path: Path) -> dict:
    return {"path": str(path.resolve()), "sha256": file_sha(path), "bytes": path.stat().st_size}


def verify_ref(ref: dict) -> Path:
    path = Path(ref["path"])
    require(path.is_file() and not path.is_symlink() and file_sha(path) == ref["sha256"]
            and path.stat().st_size == ref["bytes"], "learning source reference mismatch")
    return path


def program_observation(parent_code: str, child_code: str, parent_summary: dict, child_summary: dict) -> dict:
    """Conservative source/output observations, not a plan-realization judge."""
    source_change = "different_source"
    if parent_code == child_code:
        source_change = "identical_source"
    else:
        try:
            old, new = ast.parse(parent_code), ast.parse(child_code)
            if ast.dump(old) == ast.dump(new):
                source_change = "ast_unchanged"
            else:
                old_names = {n.name for n in old.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef))}
                added = [n.name for n in new.body if isinstance(n, (ast.FunctionDef, ast.AsyncFunctionDef)) and n.name not in old_names]
                loaded = {n.id for n in ast.walk(new) if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load)}
                if added and not any(name in loaded for name in added):
                    source_change = "added_definitions_without_static_name_load"
        except SyntaxError:
            source_change = "source_parse_unavailable"
    old_output = (parent_summary or {}).get("submission_output_attestation")
    new_output = (child_summary or {}).get("submission_output_attestation")
    output_relation = "not_comparable"
    if old_output and new_output:
        output_relation = "identical_observed_outputs" if old_output == new_output else "different_observed_outputs"
    return {"source_change": source_change, "output_relation": output_relation,
            "declared_intervention_realization": "unverified",
            "hypothesis_disposition": "inconclusive",
            "scope": "Observed source and scored outputs only; no global equivalence or scientific refutation inferred."}


def failure_projection(validation: list[dict], *, maximum_rows: int = 4, text_limit: int = 2000) -> dict:
    failed = [v for v in validation if v.get("accepted") is False or v.get("timed_out") is True
              or v.get("returncode") not in (None, 0)]
    selected = failed[-maximum_rows:]
    rows = []
    for original in selected:
        row = {k: original.get(k) for k in ("tier", "attempt", "accepted", "returncode", "timed_out")}
        for key in ("stdout", "stderr"):
            value = str(original.get(key) or "")
            row[key] = value[-text_limit:]
            row[f"omitted_{key}_characters"] = max(0, len(value) - text_limit)
        rows.append(row)
    return {"failed_checks": rows, "validation_count": len(validation), "failed_count": len(failed),
            "omitted_failed_checks": max(0, len(failed) - len(selected))}


def bounded_change(parent: str, child: str, limit: int = 6000) -> dict:
    diff = "\n".join(difflib.unified_diff(parent.splitlines(), child.splitlines(), fromfile="parent", tofile="attempt", n=3))
    excerpt = diff if len(diff) <= limit else diff[:limit // 2] + "\n[omitted middle]\n" + diff[-limit // 2:]
    return {"program_diff_excerpt": excerpt, "omitted_diff_characters": max(0, len(diff) - limit)}


def verify_planner_transport(request: dict, parsed: dict, model: str | None = None) -> dict:
    """Bind semantic context and decoded decisions to actual adapter bytes."""
    from .coder_openrouter import OpenRouterCoderModel
    transport = parsed.get("_transport_request_body_json")
    raw = parsed.get("_raw_content")
    require(isinstance(transport, str) and isinstance(raw, str), "planner transport/raw evidence absent")
    require(hashlib.sha256(transport.encode()).hexdigest() == parsed.get("_transport_request_body_sha256")
            and hashlib.sha256(raw.encode()).hexdigest() == parsed.get("_raw_sha256"),
            "planner transport/raw hash mismatch")
    body = json.loads(transport)
    users = [m for m in body.get("messages", []) if m.get("role") == "user"]
    require(len(users) == 1 and json.loads(users[0]["content"]) == request,
            "planner HTTP payload differs from recorded semantic request")
    require(model is None or body.get("model") == model, "planner HTTP model mismatch")
    decoder = OpenRouterCoderModel(api_key="offline-audit", model=body["model"],
        base_url="https://example.invalid", max_tokens_per_turn=1, native_tools=bool(body.get("tools")))
    decoded = decoder._decode_message(json.loads(raw), "PLANNER", parsed.get("_provider_finish_reason"))
    # Usage/routing keys (provider, reasoning tokens, cost, cached tokens) come from the
    # response envelope, not the message; re-decoding the message cannot reproduce them.
    from .coder_openrouter import TRANSPORT_USAGE_KEYS
    require(decoded == {k: v for k, v in parsed.items()
                        if not k.startswith("_transport_request_body_") and k not in TRANSPORT_USAGE_KEYS},
            "planner decoded proposals differ from original provider message")
    return {"request_sha256": parsed["_transport_request_body_sha256"], "message_sha256": parsed["_raw_sha256"]}


def planner_evidence(run_dir: Path, origin: str, checkpoint: dict, manifest: dict) -> dict:
    """Verify actual provider journals and replay only the deterministic retrieval."""
    context = manifest.get("learning_context") or {}
    if context.get("schema_version") != "swarmrsi-learning-context-v1":
        return {"status": "legacy_not_attested", "records": [], "declared_use": {}}
    from .wisdom import WisdomContext
    config = context["wisdom_config"]
    mode = context["wisdom_mode"]
    snapshot = verify_ref(context["wisdom_snapshot_ref"]) if mode == "readonly" else None
    wisdom = WisdomContext(task_id=manifest["task_id"], mode=mode, snapshot_path=snapshot,
        runtime_tags=tuple(config["runtime_tags"]), max_entries=config["max_entries"], max_characters=config["max_characters"])
    require(wisdom.config == config and wisdom.config["snapshot_sha256"] == context["wisdom_snapshot_sha256"],
            "Wisdom run context mismatch")
    state = checkpoint["proposer_state"]
    require(type(state["batch"]) is int and len(state["batch_records"]) == state["batch"], "incomplete planner journal")
    records, declared_use, injected = [], {}, 0
    for index, reference in enumerate(state["batch_records"]):
        path = relocate(run_dir, origin, reference["path"])
        require(file_sha(path) == reference["sha256"], "planner record changed")
        record = read_json(path)
        require(record["batch"] == index, "planner batch ordering mismatch")
        request_path = contained(path.parent, f"batch_{index:06d}.request.json")
        response_path = contained(path.parent, f"batch_{index:06d}.response.json")
        request, response = read_json(request_path), read_json(response_path)
        require(file_sha(request_path) == record["request_artifact_sha256"]
                and file_sha(response_path) == record["response_artifact_sha256"]
                and request == record["request"] and response == {"response": record["response"], "usage": record["usage"]},
                "planner request/response binding mismatch")
        transport = verify_planner_transport(request, response["response"], manifest.get("model"))
        prompt = request["planner_payload"]
        if mode == "off":
            require("cross_task_wisdom" not in prompt and "wisdom" not in record, "off mode injected cross-task Wisdom")
        else:
            query = json.dumps({"task": prompt["task"], "parents": prompt["nodes"],
                                "failures": prompt["diagnostic_failed_branches"]}, ensure_ascii=False, sort_keys=True)
            retrieval = wisdom.retrieve(query)
            require(record["wisdom"]["retrieval"] == retrieval, "stored Wisdom retrieval not derived from frozen W0")
            require(record["wisdom"]["injected"] == bool(retrieval["entries"]), "Wisdom injection flag mismatch")
            if retrieval["entries"]:
                require(prompt.get("cross_task_wisdom") == retrieval, "Wisdom did not reach the actual planner request")
                injected += 1
            else:
                require("cross_task_wisdom" not in prompt, "empty retrieval gained extra prompt content")
            available = {e["entry_sha256"] for e in retrieval["entries"]}
            for use in record["wisdom"]["proposal_use"]:
                raw = response["response"]["proposals"][use["slot_index"]]
                require(use["declared_entry_ids"] == raw.get("wisdom_entry_ids", [])
                        and set(use["declared_entry_ids"]).issubset(available), "unbound declared Wisdom use")
                declared_use[use["action_id"]] = use["declared_entry_ids"]
        records.append({"record": source_ref(path), "request": source_ref(request_path),
                        "response": source_ref(response_path), "transport": transport})
    return {"status": "verified_original_journals", "records": records, "injected_batches": injected,
            "declared_use": declared_use, "causal_effect": "not_established_from_injection_or_self_report"}


def read_run_evidence(run_dir: Path, source_root: Path, *, original_run_root: str | None = None,
                      expected_final_sha256: str | None = None, source_role: str = "development") -> dict:
    """Re-derive a terminal result from source/score/artifact-bound original files.

    Supports relocated historical runs. Their legacy family claims are exposed
    only as declared plans, not accepted as realized scientific diversity.
    """
    require(source_role in {"development", "selection", "heldout"}, "unknown source role")
    run_dir, source_root = run_dir.resolve(), source_root.resolve()
    final_path = contained(run_dir, "final.json")
    if expected_final_sha256 is not None:
        require(file_sha(final_path) == expected_final_sha256, "terminal result hash changed")
    final, checkpoint, manifest = (read_json(contained(run_dir, name)) for name in ("final.json", "checkpoint.json", "manifest.json"))
    require(bool(final.get("stop_reason")) and final.get("finished_at_unix") is not None, "run is not terminal")
    verify_seal(checkpoint, "checkpoint_sha256")
    verify_events(checkpoint["events"])
    require(final["version"] == manifest["version"] == checkpoint["version"], "run version mismatch")
    for key in ("accounting", "steps", "diagnostics"):
        require(final[key] == checkpoint[key], "final/checkpoint mismatch")
    require(final["graph"]["events"] == checkpoint["events"], "final graph mismatch")
    require(final["search_strategy_state"] == checkpoint["strategy_state"], "final policy state mismatch")
    proposer_state = checkpoint.get("proposer_state") or {}
    if "session_clock" in manifest:
        from .session_runtime import SessionClock, _nonnegative
        require("transport_receipt" in proposer_state, "session transport receipt missing")
        clock_state = proposer_state.get("session_clock")
        require(isinstance(clock_state, dict) and final.get("session_clock") == clock_state,
                "final/checkpoint session clock mismatch")
        identity = {key: manifest[key] for key in (
            "task_id", "seed", "model", "version", "source_hashes", "learning_context", "limits")}
        profile_key = manifest["learning_context"]["profile_source_key"]
        clock_profile = Path(profile_key) if Path(profile_key).is_absolute() else contained(source_root, profile_key)
        runtime_controls = read_json(clock_profile).get("session_runtime_controls")
        expected_clock = SessionClock(started_unix=manifest["started_at_unix"],
                                      started_monotonic=0,
                                      wall_limit=manifest["limits"]["wall_time_seconds"], identity=identity,
                                      runtime_controls=runtime_controls,
                                      trial_started_unix=manifest["session_clock"].get("trial_started_at_unix"))
        require(clock_state.get("origin") == manifest["session_clock"] == expected_clock.origin,
                "session clock identity mismatch")
        recorded = _nonnegative(clock_state.get("recorded_at_unix"), "recorded time")
        _nonnegative(clock_state.get("elapsed_seconds"), "elapsed time")
        require(manifest["started_at_unix"] <= recorded <= final["finished_at_unix"],
                "session clock time ordering mismatch")
    if "transport_receipt" in proposer_state:
        from .session_runtime import verify_transport_snapshot
        receipt = proposer_state["transport_receipt"]
        require(final.get("planner_transport") == receipt, "final/checkpoint planner transport mismatch")
        batch_refs = proposer_state.get("batch_records", [])
        audit_root = (relocate(run_dir, original_run_root or str(run_dir), batch_refs[0]["path"]).parent
                      if batch_refs else contained(run_dir, "planner_batches"))
        verify_transport_snapshot(audit_root, proposer_state["batch"], receipt)
        if receipt["pending_request_batches"]:
            require(final["stop_reason"] in {
                "session_deadline_reached", "session_deadline_reached_without_incumbent",
                "planner_transport_failed", "planner_transport_failed_without_incumbent",
            } and proposer_state.get("terminal_intent") == {
                "stop_reason": final["stop_reason"], "stop_details": final["stop_details"],
            }, "pending planner request lacks terminal stop intent")
    if proposer_state.get("terminal_intent") is not None:
        require(proposer_state["terminal_intent"] == {
            "stop_reason": final["stop_reason"], "stop_details": final["stop_details"]},
            "terminal intent mismatch")
    require(final["summary_sha256"] == canonical_sha256({
        "version": final["version"], "steps": final["steps"],
        "graph_sha256": canonical_sha256([e["event_sha256"] for e in checkpoint["events"]]),
    }), "final summary seal mismatch")
    source_refs = []
    for relative, digest in manifest["source_hashes"].items():
        path = Path(relative) if Path(relative).is_absolute() else contained(source_root, relative)
        require(file_sha(path) == digest, f"historical source changed: {relative}")
        source_refs.append(source_ref(path))
    nodes = {n["id"]: n for n in checkpoint["nodes"]}
    require(len(nodes) == len(checkpoint["nodes"]) and "root" in nodes, "ambiguous program identities")
    origin = original_run_root or str(run_dir)
    artifacts, node_rows, measurements = {}, [], {}
    for node in nodes.values():
        artifact = relocate(run_dir, origin, node["artifact"])
        executable = contained(artifact, node.get("executable_relpath", "run.py"))
        require(file_sha(executable) == node["executable_sha256"]
                and artifact_workspace_sha256(artifact) == node["workspace_sha256"], "program artifact mismatch")
        artifacts[node["id"]] = executable
        if not node["artifact_valid"]:
            continue
        score = finite(node["native_metric"])
        evidence = relocate(run_dir, origin, node["evaluation_evidence_path"])
        require(file_sha(evidence) == node["evaluation_evidence_sha256"], "official evidence changed")
        measured = read_json(evidence)
        require(measured.get("valid") is True and measured.get("execution_valid") is True,
                "invalid measurement")
        require(measured.get("score_bearing_evaluation_attempted") is True, "no official score attempt")
        require(measured["metrics"]["aggregate_improvement"] == score
                and measured["details"]["candidate_executable_sha256"] == node["executable_sha256"],
                "score/artifact binding mismatch")
        root_missing = node["id"] == "root" and not (artifact / "output").is_dir()
        if not root_missing:
            require(output_attestation(artifact / "output") == measured["details"]["submission_output_attestation"],
                    "scored output bytes changed")
        measurements[node["id"]] = measured["details"]
        if node["id"] != "root":
            result = read_json(contained(artifact.parent, "result.json"))
            verify_seal(result, "outcome_sha256")
            trajectory = [json.loads(line) for line in contained(artifact.parent, "trajectory.jsonl").read_text().splitlines()]
            verify_events(trajectory)
            require(trajectory == result["events"] and result["trajectory_sha256"] == canonical_sha256(
                [e["event_sha256"] for e in trajectory]), "coder trajectory mismatch")
            require(result["action_id"] == node["id"] and result["status"] == "success"
                    and result["parent_preserved"] is True and not result["contract_violations"], "invalid coder outcome")
            require(result["final_sha256"] == node["executable_sha256"]
                    and result["final_workspace_sha256"] == node["workspace_sha256"], "coder artifact mismatch")
            require(node["parent_id"] in nodes and result["parent_sha256"] == nodes[node["parent_id"]]["executable_sha256"],
                    "coder parent lineage mismatch")
        node_rows.append({"id": node["id"], "score": score, "artifact": source_ref(executable),
                          "evaluation": source_ref(evidence),
                          "coder_outcome": source_ref(artifact.parent / "result.json") if node["id"] != "root" else None,
                          "coder_trajectory": source_ref(artifact.parent / "trajectory.jsonl") if node["id"] != "root" else None,
                          "output_custody": "original_root_output_not_retained" if root_missing else "verified_original_bytes"})
    valid_nodes = [n for n in nodes.values() if n["artifact_valid"] and n["native_metric"] is not None]
    expected_best = max(valid_nodes, key=lambda n: n["native_metric"]) if valid_nodes else None
    require(final["best_native_metric"] == (expected_best["native_metric"] if expected_best else None), "terminal best metric mismatch")
    if expected_best:
        require(nodes.get(final["best_node_id"]) in valid_nodes and nodes[final["best_node_id"]]["native_metric"] == expected_best["native_metric"],
                "terminal best node mismatch")
    evolved = [n for n in valid_nodes if n["id"] != "root"]
    best_evolved_metric = max((n["native_metric"] for n in evolved), default=None)
    require(final["best_evolved_native_metric"] == best_evolved_metric, "terminal evolved metric mismatch")
    if evolved:
        require(nodes.get(final["best_evolved_node_id"]) in evolved
                and nodes[final["best_evolved_node_id"]]["native_metric"] == best_evolved_metric, "terminal evolved node mismatch")
    actions = {row["candidate"]["id"]: row for row in checkpoint["actions"]}
    require(len(actions) == len(checkpoint["actions"]), "duplicate action identities")
    require(len({s["action_id"] for s in checkpoint["steps"]}) == len(checkpoint["steps"]), "duplicate completed steps")
    observations = []
    for step in checkpoint["steps"]:
        identifier = step["action_id"]
        action = actions[identifier]["candidate"]
        ActionCandidate(**{**action, "operator": ActionOperator(action["operator"]),
                           "family": ScientificFamily(**action["family"]),
                           "dependencies": tuple(ResearchDependency(**d) for d in action["dependencies"])})
        node = nodes.get(identifier)
        parent = nodes[step["code_parent_id"]]
        parent_dependency = next(d for d in action["dependencies"] if d["kind"] == "code_parent")
        require(parent_dependency["source_id"] == parent["id"]
                and parent_dependency["source_sha256"] == parent["executable_sha256"], "action parent mismatch")
        if node is not None:
            require(node["parent_id"] == parent["id"] and node["native_metric"] == step["native_metric"]
                    and node["artifact_valid"] == step["artifact_valid"], "observed step/node mismatch")
        row = {"action_id": identifier, "action_sha256": action["action_sha256"],
               "parent_id": parent["id"], "declared_hypothesis": action.get("hypothesis_statement", ""),
               "declared_family": action["family"], "operator": action["operator"],
               "coder_status": step["coder_status"], "recovery_route": step.get("recovery_route"),
               "artifact_valid": step["artifact_valid"], "native_metric": step["native_metric"],
               "parent_metric": parent["native_metric"], "scientific_disposition": "inconclusive"}
        if node is not None:
            code = artifacts[identifier].read_text(errors="replace")
            row.update(program=source_ref(artifacts[identifier]), code_excerpt=code[:6000],
                       omitted_code_characters=max(0, len(code) - 6000),
                       implementation_observation=program_observation(
                           artifacts[parent["id"]].read_text(errors="replace"), code,
                           measurements.get(parent["id"]), measurements.get(identifier)))
        else:
            diagnostic = next((d for d in checkpoint["diagnostics"] if d["action_id"] == identifier), None)
            if diagnostic is None:
                # A third state the audit did not model: the coder returned, the
                # artifact failed validation, the candidate was discarded, and no
                # diagnostic workspace was ever sealed. There is no node and no
                # diagnostic, so there is nothing to attest -- but there is also
                # nothing being claimed: no program, no score. An MLE-Bench run
                # reached terminal verification and died here on a bare
                # StopIteration. Anything that does claim a result must still have
                # custody, so only this exact shape is admitted.
                require(step["artifact_valid"] is False and step["native_metric"] is None,
                        "action without node or diagnostic claims a program or a score")
                row["scientific_disposition"] = "discarded_without_diagnostic"
                observations.append(row)
                continue
            failed = relocate(run_dir, origin, diagnostic["artifact"])
            require(artifact_workspace_sha256(failed) == diagnostic["diagnostic_workspace_sha256"], "failed source changed")
            result = read_json(contained(failed.parent, "result.json"))
            verify_seal(result, "outcome_sha256")
            require(result["action_id"] == identifier and result["trajectory_sha256"] == diagnostic["trajectory_sha256"],
                    "failed outcome binding mismatch")
            # The coordinator deliberately falls back to the final artifact
            # for plan_blocked/no-edit and evaluator-fault branches. A missing
            # separate failed workspace is not missing program custody.
            diagnostic_digest = result.get("diagnostic_workspace_sha256")
            selected_digest = diagnostic_digest or result["final_workspace_sha256"]
            require(result["status"] == diagnostic["status"]
                    and selected_digest == diagnostic["diagnostic_workspace_sha256"]
                    and result["parent_sha256"] == parent["executable_sha256"], "failed outcome source mismatch")
            attempted = contained(failed, parent.get("executable_relpath", "run.py"))
            if not diagnostic_digest:
                require(file_sha(attempted) == result["final_sha256"], "diagnostic final executable mismatch")
            validation = diagnostic.get("validation", [])
            require(validation == result.get("validation", []), "diagnostic validation differs from coder result")
            trajectory = [json.loads(line) for line in contained(failed.parent, "trajectory.jsonl").read_text().splitlines()]
            verify_events(trajectory)
            require(trajectory == result["events"] and canonical_sha256([e["event_sha256"] for e in trajectory]) == result["trajectory_sha256"],
                    "failed trajectory mismatch")
            code = attempted.read_text(errors="replace")
            evaluation_ref = None
            if diagnostic.get("evaluation_evidence_sha256"):
                evidence_path = contained(failed.parent, "scientific_evaluation.json")
                require(file_sha(evidence_path) == diagnostic["evaluation_evidence_sha256"], "failed evaluation evidence changed")
                evaluation_ref = source_ref(evidence_path)
            row.update(failure_reason=diagnostic.get("reason"), failure_validation=validation,
                       failure_summary=failure_projection(validation),
                       failure_stage=("scientific_evaluation" if step.get("recovery_route") == "retry_evaluator_same_candidate"
                                      else "planning_realization" if result["status"] == "plan_blocked" else "implementation"),
                       evaluation_evidence=evaluation_ref,
                       attempted_program=source_ref(attempted), code_excerpt=code[:6000],
                       omitted_code_characters=max(0, len(code) - 6000),
                       diagnostic_artifact_kind="separate_failed_workspace" if diagnostic_digest else "verified_final_workspace",
                       failed_result=source_ref(failed.parent / "result.json"),
                       failed_trajectory=source_ref(failed.parent / "trajectory.jsonl"),
                       implementation_observation={"declared_intervention_realization": "not_established",
                                                   "hypothesis_disposition": "inconclusive"})
        observations.append(row)
        if row.get("code_excerpt") is not None:
            child = artifacts[identifier].read_text(errors="replace") if node is not None else attempted.read_text(errors="replace")
            row.update(bounded_change(artifacts[parent["id"]].read_text(errors="replace"), child))
    context = manifest.get("learning_context") or {}
    journal = planner_evidence(run_dir, origin, checkpoint, manifest)
    for observation in observations:
        observation["wisdom_declared_entry_ids"] = journal["declared_use"].get(observation["action_id"], [])
    policy = context.get("policy")
    search_policy = False
    if policy:
        from .search_policy import SearchPolicy, matched_profile, policy_from_record, search_spec_of, validate_execution_identity
        parsed_policy = policy_from_record(policy)
        # A learned search policy executes its effective specification; a Tree
        # policy executes through the checkpoint's learned_policy instead.
        executed_search = search_spec_of(parsed_policy)
        search_policy = executed_search is not None
        if search_policy:
            state = checkpoint.get("strategy_state") or {}
            require(set(state) == {"schema_version", "identity", "state", "checkpoint_sha256"}
                    and state["schema_version"] == "swarmrsi-search-checkpoint-v1",
                    "search run lacks a complete plugin checkpoint")
            require(state["checkpoint_sha256"] == canonical_sha256({k: v for k, v in state.items() if k != "checkpoint_sha256"}),
                    "search checkpoint hash mismatch")
            identity = state["identity"]
            require(identity == context.get("search_identity"), "checkpoint search execution identity differs from manifest")
            validate_execution_identity(identity, parsed_policy, manifest["seed"])
            decision_config = context.get("search_decision_config")
            require(isinstance(decision_config, dict)
                    and set(decision_config) == {"search_plugin", "algorithm_config"}
                    and decision_config["search_plugin"] == identity,
                    "search decision configuration is not bound to the executing identity")
            decision_digest = canonical_sha256(decision_config)
            for event in checkpoint["events"]:
                if event["event_type"] == "search.action_selected":
                    require(event["payload"].get("policy_sha256") == decision_digest,
                            "selected action policy differs from executing search configuration")
            require(identity.get("schema_version") == "swarmrsi-search-plugin-v1"
                    and identity.get("spec") == executed_search.to_dict()
                    and identity.get("seed") == manifest["seed"] and isinstance(state["state"], dict),
                    "declared search differs from executing checkpoint")
            sources = identity.get("source_hashes")
            require(isinstance(sources, dict) and bool(sources)
                    and all(manifest["source_hashes"].get(p) == digest for p, digest in sources.items()),
                    "executed search implementation sources are not bound by the manifest")
        else:
            require((checkpoint.get("strategy_state") or {}).get("config", {}).get("learned_policy") == policy,
                    "declared policy was not the checkpoint's executing policy")
    compatibility = {key: manifest.get(key) for key in ("task_id", "seed", "model", "provider", "reasoning",
                     "limits", "screening_execution_limits", "native_coder_tools", "native_planner_tools", "target")}
    profile_key = context.get("profile_source_key")
    fixed_profile = None
    if policy:
        from .search_policy import SearchPolicy, matched_profile
        require(profile_key in manifest["source_hashes"], "policy run must bind its complete profile")
        profile_path = Path(profile_key) if Path(profile_key).is_absolute() else contained(source_root, profile_key)
        profile = read_json(profile_path)
        if search_policy:
            require(profile.get("search") == executed_search.to_dict() and profile.get("rsi_policy") is None,
                    "profile search policy mismatch")
            require(profile.get("search_policy") == (None if isinstance(parsed_policy, SearchPolicy) else policy),
                    "profile learned search policy mismatch")
        else:
            require(profile.get("rsi_policy") == policy, "profile policy mismatch")
        fixed_profile = matched_profile(profile, parsed_policy)
    compatibility.update(source_hashes={k: v for k, v in manifest["source_hashes"].items() if k != profile_key},
                         fixed_profile=fixed_profile,
                         root_executable_sha256=nodes["root"]["executable_sha256"],
                         wisdom_snapshot_sha256=context.get("wisdom_snapshot_sha256"),
                         wisdom_mode=context.get("wisdom_mode", "off"))
    body = {"schema_version": "swarmrsi-learning-run-v1", "provenance_type": "controller_observed",
            "run_id": canonical_sha256({"manifest": file_sha(run_dir / "manifest.json"), "final": file_sha(final_path)}),
            "run_dir": str(run_dir), "source_root": str(source_root), "original_run_root": origin,
            "source_role": source_role, "task_id": manifest["task_id"], "seed": manifest["seed"], "model": manifest["model"],
            "policy": policy, "compatibility_sha256": canonical_sha256(compatibility), "comparison_context": compatibility,
            "terminal_status": final["stop_reason"], "artifact_valid": expected_best is not None,
            "utility_field": "final.best_native_metric", "utility": final["best_native_metric"],
            "best_evolved_native_metric": best_evolved_metric,
            "best_evolved_node_id": final["best_evolved_node_id"],
            "accounting": final["accounting"], "cost_usd": final["accounting"].get("cost_usd"),
            "records": [source_ref(run_dir / name) for name in ("manifest.json", "checkpoint.json", "final.json")],
            "source_refs": source_refs, "programs": node_rows, "observations": observations, "planner_evidence": journal}
    if "transport_receipt" in proposer_state:
        body["planner_transport"] = proposer_state["transport_receipt"]
    if manifest.get('search_metric') is not None:
        body['metric_context'] = {**manifest['search_metric'], 'utility_direction': 'maximize',
                                  'legacy_native_metric_fields_are': 'public_search_utility',
                                  'official_terminal_score': None}
    return {**body, "evidence_sha256": canonical_sha256(body)}


def verify_run_evidence(record: dict) -> None:
    verify_seal(record, "evidence_sha256")
    require(record["schema_version"] == "swarmrsi-learning-run-v1", "unknown learning evidence schema")
    for reference in record["records"] + record["source_refs"]:
        verify_ref(reference)
    fresh = read_run_evidence(Path(record["run_dir"]), Path(record["source_root"]),
                              original_run_root=record["original_run_root"],
                              expected_final_sha256=record["records"][2]["sha256"], source_role=record["source_role"])
    require(fresh == record, "learning summary differs from source records")


class FeedbackPool:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)

    def add(self, record: dict) -> None:
        verify_run_evidence(record)
        require(record["source_role"] != "heldout", "heldout results cannot enter learning feedback")
        write_immutable(self.root / (record["run_id"] + ".json"), record)

    def snapshot(self) -> dict:
        records = [read_json(p) for p in sorted(self.root.glob("*.json"))]
        for record in records:
            verify_run_evidence(record)
            require(record["source_role"] != "heldout", "heldout feedback forbidden")
        body = {"schema_version": "swarmrsi-feedback-snapshot-v1", "records": records}
        return {**body, "feedback_sha256": canonical_sha256(body)}


def retain_learning_winners(records: list[dict], profile: dict, directory: Path) -> list[dict]:
    """Preserve verified new winners for later campaigns without relabeling.

    A policy may be rejected while one of its programs wins a task. Keep that
    program, but do not turn its score into proof of a superior search policy.
    """
    retained = []
    for run in records:
        verify_run_evidence(run)
        if run["source_role"] != "selection" or run["task_id"] not in profile["tasks"]:
            continue
        source_profile = run["comparison_context"].get("fixed_profile")
        require(source_profile is not None and source_profile["model"] == run["model"] == profile["model"],
                "retained learning run model mismatch")
        source_task = source_profile["tasks"][run["task_id"]]
        target_task = profile["tasks"][run["task_id"]]
        if "search_metric" in target_task:
            # Public validation is not a win over a Claude Code reference and
            # cannot create an official-test winner/reuse record.
            continue
        for key in ("claude_code_score", "transcript_sha256"):
            require(source_task.get(key) == target_task.get(key), "retained learning baseline mismatch")
        score, threshold = run["best_evolved_native_metric"], finite(target_task["claude_code_score"])
        if not run["artifact_valid"] or score is None or not finite(score) > threshold:
            continue
        body = {"schema_version": "swarmrsi-retained-learning-win-v1", "task_id": run["task_id"],
                "model": run["model"], "threshold": threshold, "transcript_sha256": target_task.get("transcript_sha256"),
                "winner_node_id": run["best_evolved_node_id"], "winner_metric": score, "run": run,
                "claim_scope": "original measured program retained; not a new-version or policy-superiority score"}
        row = {**body, "winner_sha256": canonical_sha256(body)}
        write_immutable(directory / (run["run_id"] + ".json"), row)
        retained.append(row)
    return retained


def learning_winner_skips(paths: list[Path], profile: dict) -> dict:
    skipped = {}
    for path in paths:
        require(path.is_file(), "provide an existing immutable winner/collection file; do not silently rescan or rerun")
        document = read_json(path)
        if document.get("schema_version") == "swarmrsi-retained-learning-collection-v1":
            verify_seal(document, "collection_sha256")
            files = [verify_ref(ref) for ref in document["winner_refs"]]
        else:
            files = [path]
        for file in files:
            row = read_json(file)
            verify_seal(row, "winner_sha256")
            require(row["schema_version"] == "swarmrsi-retained-learning-win-v1", "unknown learning winner schema")
            run = row["run"]
            verify_run_evidence(run)
            require(row["task_id"] == run["task_id"] and row["model"] == run["model"]
                    and row["winner_metric"] == run["best_evolved_native_metric"]
                    and row["winner_node_id"] == run["best_evolved_node_id"] and run["artifact_valid"]
                    and run["source_role"] == "selection", "retained winner differs from original measured run")
            original = run["comparison_context"]["fixed_profile"]["tasks"][run["task_id"]]
            require(row["threshold"] == original["claude_code_score"]
                    and row["transcript_sha256"] == original.get("transcript_sha256")
                    and finite(row["winner_metric"]) > finite(row["threshold"]), "retained winner did not exceed its original comparator")
            task = profile["tasks"].get(row["task_id"])
            if task and profile["model"] == row["model"] and task["claude_code_score"] == row["threshold"] and task.get("transcript_sha256") == row["transcript_sha256"]:
                current = skipped.get(row["task_id"])
                if current is None or row["winner_metric"] > current["winner_metric"]:
                    skipped[row["task_id"]] = {"winner_metric": row["winner_metric"], "run_id": run["run_id"],
                        "winner_sha256": row["winner_sha256"], "original_policy": run["policy"], "new_version_score": None}
    return skipped
