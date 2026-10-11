"""Synthetic file-bound analysis replay only; no real performance claims."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import pytest

from evolvcoder.contracts import canonical_sha256
from evolvcoder.learning_analysis import analyze_learning, export_learning_analysis
from evolvcoder.learning_evidence import read_run_evidence
from evolvcoder.retained_results import file_sha, read_json
from evolvcoder.rsi import ValidationPending
from evolvcoder.tree_policy import TreePolicy
from evolvcoder.wisdom import WisdomLibrary
from test_evolvcoder_learning import FileValidator, Proposer, make_run, setup_chain
from test_evolvcoder_learning_runtime import validation_runner
from test_evolvcoder_retained_results import put, seal


def inventory(root):
    return {str(p.relative_to(root)): (file_sha(p), p.stat().st_mtime_ns) for p in root.rglob("*") if p.is_file()}


def completed(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=3)
    policies = [TreePolicy(parent_exploration_weight=value) for value in (1., 2., 3.)]
    validator = FileValidator(tmp_path / "runs", {TreePolicy().sha256: .1, policies[0].sha256: .3,
        policies[1].sha256: .2, policies[2].sha256: .3})
    chain.run(Proposer(policies), validator, pool)
    return chain, pool, validator


def test_all_rounds_current_incumbent_and_logical_reuse(tmp_path):
    chain, _, validator = completed(tmp_path)
    before = inventory(tmp_path)
    with patch("subprocess.Popen", side_effect=AssertionError("no launch")), \
         patch("urllib.request.urlopen", side_effect=AssertionError("no model")):
        bundle = analyze_learning(chain_root=chain.root)
    assert inventory(tmp_path) == before
    rows = bundle["policy_trials"]
    assert [r["status"] for r in rows] == ["accepted", "rejected", "rejected"]
    assert rows[1]["incumbent_policy"] == rows[0]["candidate_policy"] == rows[2]["retained_policy"]
    assert rows[1]["trial"]["mean_delta"] == pytest.approx(-.1)
    assert rows[2]["trial"]["mean_delta"] == 0
    assert len(validator.runs) == 4
    assert len(bundle["validation_runs"]) == 5  # Four trials and one historical feedback source.
    best = [r for r in bundle["validation_runs"] if len(r["logical_validation_uses"]) == 3]
    assert len(best) == 1 and best[0]["physical_trial"] is None
    assert bundle["manifest"]["actual_total_cost_usd"] is None
    assert bundle["manifest"]["reported_unique_validation_cost"]["reported_known_sum_usd"] is None
    assert all(r["proposal_transport"] == "not_available" for r in rows)


def test_initialized_chain_is_not_trial_or_effect(tmp_path):
    chain, _ = setup_chain(tmp_path)
    bundle = analyze_learning(chain_root=chain.root)
    assert bundle["policy_trials"] == []
    assert bundle["manifest"]["chain_state"]["completed_rounds"] == 0
    assert not bundle["manifest"]["chain_state"]["finished"]
    assert bundle["manifest"]["reported_proposal_cost"]["reported_complete_sum_usd"] is None


def test_pending_partial_pair_keeps_verified_incumbent_without_delta(tmp_path):
    chain, pool = setup_chain(tmp_path)
    policy = TreePolicy(parent_exploration_weight=1.)
    validator = FileValidator(tmp_path / "runs", {TreePolicy().sha256: .1})
    with patch.object(validator, "evaluate", wraps=validator.evaluate) as original:
        def dispatched(**kwargs):
            if kwargs["policy"] == policy:
                raise ValidationPending("synthetic running candidate")
            return original(**kwargs)
        class Pending:
            evaluate = staticmethod(dispatched)
        result = chain.run_round(Proposer([policy]), Pending(), pool)
    assert result["status"] == "validation_pending"
    bundle = analyze_learning(chain_root=chain.root)
    row = bundle["policy_trials"][0]
    assert row["status"] == "validation_pending" and not row["terminal"]
    assert row["trial"] is row["retained_policy"] is None
    assert len(row["received_validations"]) == 1
    assert row["received_validations"][0]["status"] == "verified"


@pytest.mark.parametrize("prefix,status", [(1, "feedback_recorded_not_proposed"), (2, "proposal_result_unknown")])
def test_interrupted_proposal_prefix_is_not_a_candidate(tmp_path, prefix, status):
    chain, pool = setup_chain(tmp_path)
    directory = chain.root / "round_000"
    snapshot = pool.snapshot()
    put(directory / "feedback.json", snapshot)
    if prefix == 2:
        put(directory / "proposal_request.json", chain._request(0, TreePolicy(), chain.config["cells"], snapshot, []))
    row = analyze_learning(chain_root=chain.root)["policy_trials"][0]
    assert row["status"] == status and row["candidate_id"] is None
    assert row["proposal_usage"] is row["trial"] is None


def test_invalid_proposal_and_malformed_trial_return_are_preserved(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=2)
    class Bad:
        def evaluate(self, **kwargs):
            return {"invalid_synthetic_return": True}
    chain.run_round(Proposer([TreePolicy()]), Bad(), pool)
    with pytest.raises((KeyError, ValueError)):
        chain.run_round(Proposer([TreePolicy(parent_exploration_weight=1.)]), Bad(), pool)
    rows = analyze_learning(chain_root=chain.root)["policy_trials"]
    assert [r["status"] for r in rows] == ["invalid_proposal", "validation_pending"]
    assert not rows[1]["terminal"] and not rows[1]["comparison_complete"]
    assert rows[0]["candidate_id"] is None
    assert rows[1]["received_validations"][0]["status"] == "unverified_return"
    assert all(r["trial"] is None for r in rows)


@pytest.mark.parametrize("damage", ["incumbent", "missing_request", "source", "orphan", "future"])
def test_reject_changed_provenance(tmp_path, damage):
    chain, _, _ = completed(tmp_path)
    if damage == "incumbent":
        path = chain.root / "round_001/result.json"
        row = read_json(path); row["incumbent_policy"] = TreePolicy().to_dict()
        row.pop("round_sha256"); row["round_sha256"] = canonical_sha256(row); put(path, row)
    elif damage == "missing_request":
        (chain.root / "round_000/proposal_request.json").unlink()
    elif damage == "source":
        row = read_json(chain.root / "round_000/feedback.json")["records"][0]
        source = Path(row["run_dir"]) / "final.json"
        source.write_text(source.read_text() + "\n")
    elif damage == "orphan":
        (chain.root / "round_001").rename(chain.root / "orphaned")
    else:
        (chain.root / "round_009").mkdir()
    with pytest.raises((ValueError, OSError)):
        analyze_learning(chain_root=chain.root)


def test_wisdom_sources_are_not_invented_use(tmp_path):
    chain, pool = setup_chain(tmp_path)
    library = WisdomLibrary(tmp_path / "wisdom")
    library.ingest(pool.snapshot()["records"][0])
    snapshot = tmp_path / "W0.json"
    library.freeze(snapshot)
    before = inventory(tmp_path)
    bundle = analyze_learning(chain_root=chain.root, wisdom_snapshot=snapshot)
    assert len(bundle["wisdom_sources"]) == 1
    assert bundle["wisdom_sources"][0]["authority"] == "advisory_only"
    assert bundle["wisdom_usage"] == []
    assert bundle["validation_runs"][0]["planner_evidence_status"] == "legacy_not_attested"
    assert inventory(tmp_path) == before


def test_physical_ledger_distinguishes_intent_process_exit_and_cache(tmp_path):
    chain, _ = setup_chain(tmp_path / "chain_test")
    runner = validation_runner(tmp_path / "runner_test")
    root, command = runner._prepare(TreePolicy(), "fixture", 0)
    def observed():
        return analyze_learning(chain_root=chain.root, runner_roots=[runner.root])
    assert observed()["validation_ledger"][0]["status"] == "prepared_not_launched"
    put(root / "launch.json", {"command": command, "host": "synthetic", "runner_sha256": canonical_sha256(runner.config)})
    result = observed()
    assert result["manifest"]["launch_intents"] == 1 and result["manifest"]["process_receipts"] == 0
    assert result["validation_ledger"][0]["status"] == "launch_intent_process_unknown"
    put(root / "process.json", {"pid": 999999, "host": "synthetic", "start_identity": None})
    assert observed()["manifest"]["process_receipts"] == 1
    put(root / "exit.json", {"returncode": 17, "cost_usd": None})
    result = observed()
    assert result["validation_ledger"][0]["status"] == "exited_without_verified_receipt"
    assert result["manifest"]["reported_unique_validation_cost"]["unknown_records"] == 1
    assert result["validation_ledger"][0]["wisdom_usage_batches"] is None
    assert result["manifest"]["wisdom_usage_coverage"]["incomplete"] is True
    assert result["manifest"]["wisdom_usage_coverage"]["not_audited_trial_count"] == 1
    assert result["manifest"]["wisdom_usage_coverage"]["campaign_total_usage_batches"] is None
    put(root / "cache_hit_000000.json", {"run_id": "invented", "evidence_sha256": "a" * 64, "additional_model_calls": 0})
    with pytest.raises(ValueError, match="unbound cache"):
        observed()


@pytest.mark.parametrize("damage", ["identity", "profile", "launch"])
def test_reject_changed_runner_journals(tmp_path, damage):
    chain, _ = setup_chain(tmp_path / "chain_test")
    runner = validation_runner(tmp_path / "runner_test")
    root, command = runner._prepare(TreePolicy(), "fixture", 0)
    if damage == "identity":
        row = read_json(root / "identity.json"); row["seed"] = 42; put(root / "identity.json", row)
    elif damage == "profile":
        row = read_json(root / "profile.json"); row["changed"] = True; put(root / "profile.json", row)
    else:
        put(root / "launch.json", {"command": command + ["--changed"], "host": "synthetic", "runner_sha256": canonical_sha256(runner.config)})
    with pytest.raises(ValueError):
        analyze_learning(chain_root=chain.root, runner_roots=[runner.root])


def test_atomic_export_cli_and_refuse_overwrite(tmp_path, capsys):
    from evolvcoder.cli import main
    chain, _ = setup_chain(tmp_path / "inputs")
    before = inventory(tmp_path / "inputs")
    destination = tmp_path / "export"
    assert main(["analyze-learning", "--chain-root", str(chain.root), "--output", str(destination)]) == 0
    assert '"zero_model_calls": true' in capsys.readouterr().out
    manifest = read_json(destination / "manifest.json")
    assert manifest["outputs"]["policy_trials.jsonl"]["rows"] == 0
    for name, info in manifest["outputs"].items():
        assert file_sha(destination / name) == info["sha256"]
    assert inventory(tmp_path / "inputs") == before
    output_before = inventory(destination)
    with pytest.raises(ValueError, match="output exists"):
        export_learning_analysis(destination, chain_root=chain.root)
    assert inventory(destination) == output_before


@pytest.mark.parametrize("installed", [False, True])
def test_terminal_ledger_cache_is_not_duplicate_spend(tmp_path, monkeypatch, installed):
    from evolvcoder.learning_evidence import source_ref
    from evolvcoder.learning_trials import SubprocessValidationRunner
    chain, _ = setup_chain(tmp_path / "chain_test")
    record = make_run(tmp_path / "measurement")
    source = Path(record["source_root"])
    profile = record["comparison_context"]["fixed_profile"]
    config = {"schema_version": "evolvcoder-rsi-validation-runner-v1", "base_profile": profile,
        "source_root": str(source), "base_profile_source": source_ref(source / "profile.json"),
        "cells": [{"task_id": "won", "seed": 0}], "source_refs": record["source_refs"],
        "runtime_python": str(Path(sys.executable).absolute()), "environment": {},
        "runtime_executable_resolved": str(Path(sys.executable).resolve()), "timeout_seconds": 60}
    runner_root = tmp_path / "runner"
    if installed:
        from evolvcoder.installed_learning import InstalledValidationRunner
        from evolvcoder import naturebench_session
        # Synthetic original installed layout; no import or execution of it.
        entry = source / "evolvcoder/naturebench_cli.py"
        entry.parent.mkdir(); entry.write_text("# original synthetic installed entrypoint\n")
        config.update(schema_version=InstalledValidationRunner.SCHEMA,
            source_refs=[*config["source_refs"], source_ref(entry)],
            orchestrator_executable_ref=source_ref(Path(sys.executable)),
            candidate_executable_ref=source_ref(Path(sys.executable)),
            execution={"runtime_python": str(Path(sys.executable).absolute()), "tasks_root": str(tmp_path / "tasks"),
                       "credential_file": str(tmp_path / "missing-secret.json"), "model_cache": None,
                       "base_url": "https://example.invalid"})
        InstalledValidationRunner.write_config(runner_root, config)
        monkeypatch.setattr(naturebench_session, "ROOT", source)
        runner = InstalledValidationRunner.resume(runner_root)
    else:
        put(runner_root / "runner.json", config)
        runner = SubprocessValidationRunner.resume(runner_root)
    trial, command = runner._prepare(TreePolicy(), "won", 0)
    Path(record["run_dir"]).rename(trial / "run")
    # Synthetic reported amount, not an assertion of real provider charges.
    for name in ["checkpoint.json", "final.json"]:
        path = trial / "run" / name
        row = read_json(path); row["accounting"]["cost_usd"] = 1.25
        put(path, seal(row, "checkpoint_sha256") if name == "checkpoint.json" else row)
    record = read_run_evidence(trial / "run", source, original_run_root="/original/run", source_role="selection")
    put(trial / "evidence.json", record)
    put(trial / "launch.json", {"command": command, "host": "synthetic", "runner_sha256": canonical_sha256(config)})
    for index in range(3):
        put(trial / f"cache_hit_{index:06d}.json", {"run_id": record["run_id"],
            "evidence_sha256": record["evidence_sha256"], "additional_model_calls": 0})
    bundle = analyze_learning(chain_root=chain.root, runner_roots=[runner_root])
    manifest = bundle["manifest"]
    assert manifest["verified_physical_terminal_runs"] == 1 and manifest["cache_receipts"] == 3
    assert manifest["reported_unique_validation_cost"]["reported_complete_sum_usd"] == 1.25
    assert manifest["actual_total_cost_usd"] is None
    assert bundle["validation_runs"][0]["physical_trial"]["cache_hits"] == 3


def test_real_format_wisdom_request_usage_is_not_implementation_evidence(tmp_path):
    from evolvcoder.wisdom import WisdomContext
    from test_evolvcoder_incremental_feedback import campaign
    from learning_test_helpers import configure, raw
    from learning_test_helpers import client
    chain, pool = setup_chain(tmp_path / "chain_test")
    library = WisdomLibrary(tmp_path / "wisdom")
    ids = library.ingest(pool.snapshot()["records"][0])
    snapshot_path = tmp_path / "W0.json"
    library.freeze(snapshot_path)
    run = make_run(tmp_path / "destination", task_id="different")
    root = Path(run["run_dir"])
    loop = campaign(tmp_path / "loop")
    proposal = {**raw(), "wisdom_entry_ids": [ids[0]]}
    planner, calls = configure(loop, [proposal], audit_root=root / "planner")
    def adapter(request):
        response = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {
            "name": "submit_proposals", "arguments": json.dumps({"proposals": [proposal]})}}]},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}
        with patch("urllib.request.urlopen") as http:
            http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            return client().respond(request)
    planner.model.respond = adapter
    context = WisdomContext(task_id="different", mode="readonly", snapshot_path=snapshot_path)
    planner.wisdom_context, planner.task_context = context, "linear predictions public metric"
    assert loop.expand() == 1
    checkpoint = read_json(root / "checkpoint.json")
    checkpoint["proposer_state"] = loop.checkpoint()["proposer_state"]
    final = read_json(root / "final.json")
    final["planner_transport"] = checkpoint["proposer_state"]["transport_receipt"]
    put(root / "final.json", final)
    for ref in checkpoint["proposer_state"]["batch_records"]:
        ref["path"] = ref["path"].replace(str(root), "/original/run")
    put(root / "checkpoint.json", seal(checkpoint, "checkpoint_sha256"))
    manifest = read_json(root / "manifest.json")
    manifest["model"] = "test/model"
    manifest["learning_context"].update(schema_version="evolvcoder-learning-context-v1", wisdom_mode=context.mode,
        wisdom_config=context.config, wisdom_snapshot_sha256=context.config["snapshot_sha256"], wisdom_snapshot_ref=context.snapshot_ref)
    put(root / "manifest.json", manifest)
    record = read_run_evidence(root, Path(run["source_root"]), original_run_root="/original/run", source_role="selection")
    evidence = tmp_path / "destination_evidence.json"
    put(evidence, record)
    bundle = analyze_learning(chain_root=chain.root, wisdom_snapshot=snapshot_path, run_evidence=[evidence])
    usage = bundle["wisdom_usage"]
    assert len(usage) == 1 and usage[0]["injected"] is True
    assert usage[0]["proposal_declared_use"][0]["declared_entry_ids"] == [ids[0]]
    assert usage[0]["implemented_use"] is usage[0]["causal_effect"] is None


def test_input_race_fails_before_output_is_installed(tmp_path, monkeypatch):
    import evolvcoder.learning_analysis as module
    chain, _ = setup_chain(tmp_path / "inputs")
    original = module.Capture.verify
    def changed(capture):
        path = chain.root / "chain.json"
        path.write_text(path.read_text() + "\n")
        original(capture)
    monkeypatch.setattr(module.Capture, "verify", changed)
    output = tmp_path / "failed_export"
    with pytest.raises(ValueError, match="source reference mismatch"):
        export_learning_analysis(output, chain_root=chain.root)
    assert not output.exists()


def test_output_cannot_be_written_inside_journal(tmp_path):
    chain, _ = setup_chain(tmp_path)
    with pytest.raises(ValueError, match="outside input"):
        export_learning_analysis(chain.root / "export", chain_root=chain.root)
    assert not (chain.root / "export").exists()


@pytest.mark.parametrize("source_kind", ["feedback", "wisdom", "extra_evidence", "runner_source"])
@pytest.mark.parametrize("target_kind", ["artifact", "source"])
def test_export_cannot_invalidate_referenced_historical_inputs(tmp_path, source_kind, target_kind):
    from evolvcoder.learning_evidence import verify_run_evidence
    chain, pool = setup_chain(tmp_path / "chain_test")
    kwargs = {"chain_root": chain.root}
    if source_kind == "feedback":
        policy = TreePolicy(parent_exploration_weight=1.)
        validator = FileValidator(tmp_path / "runs", {TreePolicy().sha256: .1, policy.sha256: .2})
        chain.run_round(Proposer([policy]), validator, pool)
        record = read_json(chain.root / "round_000/feedback.json")["records"][0]
    elif source_kind == "runner_source":
        runner = validation_runner(tmp_path / "runner_test")
        kwargs["runner_roots"] = [runner.root]
        destination = runner.source_root / "export"
        before = inventory(tmp_path)
        with pytest.raises(ValueError, match="referenced input"):
            export_learning_analysis(destination, **kwargs)
        assert inventory(tmp_path) == before
        return
    else:
        record = make_run(tmp_path / "history", role="development")
        if source_kind == "wisdom":
            library = WisdomLibrary(tmp_path / "wisdom")
            library.ingest(record)
            snapshot = tmp_path / "wisdom/W0.json"
            library.freeze(snapshot)
            kwargs["wisdom_snapshot"] = snapshot
        else:
            evidence = tmp_path / "evidence.json"
            put(evidence, record)
            kwargs["run_evidence"] = [evidence]
    destination = Path(record["run_dir"]) / "cells/0_child/artifact/export" if target_kind == "artifact" else \
        Path(record["source_root"]) / "export"
    before = inventory(tmp_path)
    with pytest.raises(ValueError, match="referenced input"):
        export_learning_analysis(destination, **kwargs)
    assert inventory(tmp_path) == before and not destination.exists()
    verify_run_evidence(record)


@pytest.mark.parametrize("damage", [None, "http", "parsed", "missing"])
def test_proposal_transport_is_checked_without_a_model_call(damage):
    from evolvcoder.learning_analysis import _proposal_transport
    from learning_test_helpers import client
    request = {"proposal_seed": 0, "synthetic_feedback": True}
    answer = {"parameters": TreePolicy().to_dict()["parameters"], "rationale": "synthetic", "evidence_run_ids": ["synthetic"]}
    raw = {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}],
           "usage": {"prompt_tokens": 3, "completion_tokens": 5, "cost": .02}}
    with patch("urllib.request.urlopen") as http:
        http.return_value.__enter__.return_value.read.return_value = json.dumps(raw).encode()
        parsed, usage = client().respond({"version": "RSI_POLICY", "seed": 0, "policy_learning_request": request})
    if damage == "http":
        parsed["_transport_request_body_json"] += " "
    elif damage == "parsed":
        parsed["rationale"] = "not returned"
    elif damage == "missing":
        parsed.pop("_raw_content")
    with patch("urllib.request.urlopen", side_effect=AssertionError("no calls during analysis")):
        if damage:
            with pytest.raises(ValueError):
                _proposal_transport(request, {"response": parsed, "usage": usage})
        else:
            assert _proposal_transport(request, {"response": parsed, "usage": usage}) == "verified_original_transport_and_response"
