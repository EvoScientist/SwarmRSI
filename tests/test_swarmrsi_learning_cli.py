"""CLI preparation/resume and missing real-world failure shapes, no provider."""
from pathlib import Path
import sys

import pytest

from swarmrsi.campaign import artifact_workspace_sha256
from swarmrsi.contracts import canonical_sha256
from swarmrsi.learning_evidence import read_run_evidence
from swarmrsi.retained_results import file_sha, read_json
from swarmrsi.tree_policy import TreePolicy
from swarmrsi.wisdom import experience_entries
from swarmrsi import learning_cli as cli
from test_swarmrsi_learning import make_run
from test_swarmrsi_learning import FileValidator, Proposer
from test_swarmrsi_learning_runtime import validation_runner
from test_swarmrsi_retained_results import put, seal


def configuration(tmp_path):
    runtime = validation_runner(tmp_path / "runtime")
    profile_path = Path(runtime.config["base_profile_source"]["path"])
    profile = read_json(profile_path)
    profile.update(model="synthetic/model")
    profile["tasks"]["fixture"]["claude_code_score"] = .5
    put(profile_path, profile)
    history = make_run(tmp_path / "history", role="development")
    config = {"schema_version": "swarmrsi-learning-campaign-v1", "root": str(tmp_path / "campaign"),
              "source_root": str(runtime.source_root), "profile": str(profile_path),
              "runtime_python": str(Path(sys.executable).absolute()),
              "rounds": 2, "wisdom": {"mode": "readonly"}, "environment": {},
              "feedback_sources": [{"run_dir": history["run_dir"], "source_root": history["source_root"],
                                    "original_run_root": "/original/run"}]}
    path = tmp_path / "campaign_config.json"
    put(path, config)
    return path, config


def test_cli_prepares_real_file_bound_feedback_and_idempotent_w0(tmp_path, capsys):
    path, config = configuration(tmp_path)
    first = cli.prepare(path)
    assert first == cli.prepare(path)
    assert first["cells"] == [{"task_id": "fixture", "seed": 0}]
    assert len(read_json(Path(config["root"]) / "W0.json")["entries"]) == 1
    assert cli.main(["prepare", "--config", str(path)]) == 0
    assert '"zero_model_calls": true' in capsys.readouterr().out
    assert cli.main(["preflight", "--config", str(path)]) == 0
    assert '"ready": true' in capsys.readouterr().out


def test_cli_does_not_make_paid_calls_without_explicit_switch(tmp_path):
    path, _ = configuration(tmp_path)
    with pytest.raises(ValueError, match="run requires --execute"):
        cli.main(["run", "--config", str(path)])
    with pytest.raises(ValueError, match="ablate requires --execute"):
        cli.main(["ablate", "--config", str(path)])


def test_cli_refuses_silent_config_change_and_resumes_captured_environment(tmp_path, monkeypatch):
    path, config = configuration(tmp_path)
    record = cli.prepare(path)
    _, original, _ = cli.components(record)
    monkeypatch.setenv("SWARMRSI_RUNTIME_PYTHON", "/different/after-start/python")
    _, resumed, _ = cli.components(record)
    assert resumed.environment == original.environment
    config["rounds"] = 99
    put(path, config)
    with pytest.raises(ValueError, match="configuration changed"):
        cli.prepare(path)


def test_status_is_read_only_before_and_after_preflight(tmp_path, capsys):
    path, config = configuration(tmp_path)
    root = Path(config["root"])
    with pytest.raises(ValueError, match="not prepared"):
        cli.main(["status", "--config", str(path)])
    assert not root.exists()
    cli.prepare(path)
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    assert cli.main(["status", "--config", str(path)]) == 0
    assert "prepared_not_started" in capsys.readouterr().out
    assert before == {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    cli.main(["preflight", "--config", str(path)])
    before = {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}
    assert cli.main(["status", "--config", str(path)]) == 0
    assert before == {str(p): p.stat().st_mtime_ns for p in root.rglob("*")}


def test_retained_collection_is_bound_and_reverified_on_campaign_resume(tmp_path):
    path, config = configuration(tmp_path)
    collection = tmp_path / "old_winners.json"
    put(collection, seal({"schema_version": "swarmrsi-retained-learning-collection-v1",
                          "winner_refs": []}, "collection_sha256"))
    config["retained_learning_results"] = [str(collection)]
    put(path, config)
    record = cli.prepare(path)
    assert str(collection) in {r["path"] for r in record["input_refs"]}
    collection.unlink()
    with pytest.raises((ValueError, FileNotFoundError)):
        cli.prepare(path)


@pytest.mark.parametrize("field", ["model", "budget", "wisdom"])
def test_generated_profile_cannot_drift_before_first_runner(tmp_path, field):
    path, _ = configuration(tmp_path)
    record = cli.prepare(path)
    generated = Path(record["profile"])
    profile = read_json(generated)
    if field == "model":
        profile["model"] = "different/model"
    elif field == "budget":
        profile["tasks"]["fixture"]["limits"] = {"output": 999999}
    else:
        profile["wisdom"] = {"mode": "off"}
    put(generated, profile)
    with pytest.raises(ValueError, match="generated profile"):
        cli.prepare(path)
    with pytest.raises(ValueError, match="generated profile"):
        cli.components(record)
    assert not (Path(record["root"]) / "validation/runner.json").exists()


def test_wisdom_failed_arms_retained_without_false_mean_or_relaunch(tmp_path, monkeypatch):
    path, _ = configuration(tmp_path)
    record = cli.prepare(path)
    _, runner, _ = cli.components(record, execute=True)
    monkeypatch.setenv("SWARMRSI_MODEL_CACHE", "/changed/observer/cache")
    first = cli.wisdom_ablation(record, runner, execute=True)
    assert first["comparison"]["status"] == "incomplete"
    assert first["comparison"]["mean_paired_delta"] is None
    assert [r["status"] for r in first["comparison"]["outcomes"]] == ["invalid", "invalid"]
    off_root = Path(record["root"]) / "validation_no_wisdom"
    assert read_json(off_root / "runner.json")["environment"] == runner.environment
    assert not (Path(record["root"]) / "wisdom_comparison.json").exists()
    second = cli.wisdom_ablation(record, runner, execute=True)
    assert second["on_accounting"]["unique_launches"] == second["off_accounting"]["unique_launches"] == 1
    assert len(list((Path(record["root"]) / "wisdom_attempts").glob("*.json"))) >= 1


def test_wisdom_pair_rejects_non_wisdom_configuration_drift(monkeypatch):
    from copy import deepcopy
    # Synthetic structure-only fixture; real source/journal verification is
    # covered separately, these invented numbers are not benchmark evidence.
    monkeypatch.setattr(cli, "verify_run_evidence", lambda run: None)
    cell = {"task_id": "fixture", "seed": 0}
    base = {**cell, "source_role": "selection", "artifact_valid": True,
            "policy": TreePolicy().to_dict(), "planner_evidence": {
                "status": "verified_original_journals", "injected_batches": 1},
            "run_id": "on", "evidence_sha256": "synthetic", "utility": .2,
            "comparison_context": {"wisdom_mode": "readonly", "wisdom_snapshot_sha256": "W0",
                "fixed_profile": {"wisdom": {"mode": "readonly"}, "budget": 10}}}
    off = deepcopy(base)
    off.update(run_id="off", utility=.1)
    off["comparison_context"].update(wisdom_mode="off", wisdom_snapshot_sha256=None)
    off["comparison_context"]["fixed_profile"]["wisdom"] = {"mode": "off"}
    assert cli.compare_wisdom([base], [off], [cell])["mean_paired_delta"] == pytest.approx(.1)
    off["comparison_context"]["fixed_profile"]["budget"] = 11
    with pytest.raises(ValueError, match="differ beyond"):
        cli.compare_wisdom([base], [off], [cell])


def synthetic_closed_stage(tmp_path, monkeypatch, *, on_score=.2, off_score=.8, remaining=True):
    """Stage-routing fixture only; actual provenance and HTTP checks are tested separately."""
    from copy import deepcopy
    path, config = configuration(tmp_path)
    if remaining:
        profile_path = Path(config["profile"])
        profile = read_json(profile_path)
        profile["tasks"]["remaining"] = deepcopy(profile["tasks"]["fixture"])
        put(profile_path, profile)
    record = cli.prepare(path)
    _, original_runner, pool = cli.components(record)
    on, off = [], []
    for cell in record["cells"]:
        for rows, label, score in ((on, "on", on_score), (off, "off", off_score)):
            rows.append({**cell, "policy": record["theta0"], "run_id": label + cell["task_id"],
                "evidence_sha256": "synthetic-stage-evidence", "best_evolved_node_id": "child",
                "best_evolved_native_metric": score if cell["task_id"] == "fixture" else .1,
                "comparison_context": {"fixed_profile": read_json(Path(record["profile"]))}})
    comparison = {"fixture": "synthetic pair routing, not a scientific result"}
    monkeypatch.setattr(cli, "compare_wisdom", lambda a, b, c: comparison)
    request = {"cells": record["cells"], "policy": record["theta0"],
               "W0": record["wisdom_snapshot_sha256"], "utility_field": "final.best_native_metric"}
    result_path = Path(record["root"]) / "wisdom_comparison.json"
    put(result_path, seal({"request": request, "on_runs": on, "off_runs": off,
                          "comparison": comparison}, "comparison_sha256"))
    return path, record, original_runner, pool


@pytest.mark.parametrize("on_score,off_score", [(.2, .8), (.8, .2)])
def test_closed_wisdom_winner_never_runs_in_later_rsi_rounds(tmp_path, monkeypatch, on_score, off_score):
    path, record, original_runner, pool = synthetic_closed_stage(tmp_path, monkeypatch,
        on_score=on_score, off_score=off_score)
    chain, reused_runner, _ = cli.components(record, initialize_learning=True)
    assert chain.config["cells"] == [{"task_id": "remaining", "seed": 0}]
    assert reused_runner.config == original_runner.config  # Identical control cache identity.
    proposer = Proposer([TreePolicy(parent_exploration_weight=1), TreePolicy(parent_exploration_weight=2)])
    validator = FileValidator(tmp_path / "rsi_trials", {TreePolicy().sha256: .1,
        TreePolicy(parent_exploration_weight=1).sha256: .2, TreePolicy(parent_exploration_weight=2).sha256: .3})
    assert chain.run(proposer, validator, pool)["finished"]
    assert {task for _, task, _ in validator.requests} == {"remaining"}
    resumed, _, _ = cli.components(record, read_only=True)
    assert resumed.state() == chain.state()
    entry = cli.rsi_entry(record)
    assert entry["skipped_completed_stage_winners"]["fixture"]["new_rsi_score"] is None
    comparison_path = Path(record["root"]) / "wisdom_comparison.json"
    comparison_path.write_text(comparison_path.read_text() + "\n")
    with pytest.raises(ValueError, match="source reference"):
        cli.components(record)


def test_all_closed_stage_winners_skip_rsi_without_a_fake_improvement(tmp_path, monkeypatch, capsys):
    path, record, _, _ = synthetic_closed_stage(tmp_path, monkeypatch, remaining=False)
    monkeypatch.setattr(cli, "proposer", lambda *a: pytest.fail("must not call an RSI provider for solved tasks"))
    assert cli.main(["run", "--config", str(path), "--execute"]) == 0
    assert "all_tasks_solved_before_rsi" in capsys.readouterr().out
    assert not (Path(record["root"]) / "rsi_after_wisdom/round_000").exists()


def test_unclosed_stage_or_missing_entry_never_reconstructs_a_new_rsi_queue(tmp_path):
    path, _ = configuration(tmp_path)
    record = cli.prepare(path)
    root = Path(record["root"])
    put(root / "wisdom_comparison_request.json", {"incomplete": True})
    with pytest.raises(ValueError, match="not closed"):
        cli.components(record, initialize_learning=True)
    put(root / "rsi/round_000/proposal_request.json", {"prior_outbound": True})
    with pytest.raises(ValueError, match="entry missing"):
        cli.components(record)


@pytest.mark.parametrize("status,route,kind", [
    ("plan_blocked", "return_to_planner", "implementation_plan_blocked"),
    ("success", "retry_evaluator_same_candidate", "evaluation_chain_failure"),
])
def test_no_separate_failed_artifact_retains_real_program_and_correct_failure_owner(tmp_path, status, route, kind):
    original = make_run(tmp_path / "history", role="development")
    root = Path(original["run_dir"])
    c = read_json(root / "checkpoint.json")
    child = c["nodes"].pop()
    result_path = root / "cells/0_child/result.json"
    result = read_json(result_path)
    result.update(status=status, diagnostic_workspace_sha256=None, validation=[], reason="synthetic boundary")
    put(result_path, seal(result, "outcome_sha256"))
    c["steps"][0].update(artifact_valid=False, native_metric=None, coder_status=status, recovery_route=route)
    c["diagnostics"] = [{"action_id": "child", "artifact": child["artifact"],
        "status": status, "code_parent_id": "root", "diagnostic_workspace_sha256": child["workspace_sha256"],
        "trajectory_sha256": result["trajectory_sha256"], "validation": [], "reason": "synthetic boundary"}]
    put(root / "checkpoint.json", seal(c, "checkpoint_sha256"))
    f = read_json(root / "final.json")
    f.update(steps=c["steps"], diagnostics=c["diagnostics"], best_evolved_native_metric=None,
             best_evolved_node_id=None, best_node_id="root", best_native_metric=.05)
    f["summary_sha256"] = canonical_sha256({"version": f["version"], "steps": c["steps"],
                                           "graph_sha256": canonical_sha256([])})
    put(root / "final.json", f)
    evidence = read_run_evidence(root, Path(original["source_root"]), original_run_root="/original/run")
    observation = evidence["observations"][0]
    assert observation["diagnostic_artifact_kind"] == "verified_final_workspace"
    assert observation["attempted_program"]["sha256"] == child["executable_sha256"]
    assert observation["code_excerpt"]
    assert experience_entries(evidence)[0]["kind"] == kind
