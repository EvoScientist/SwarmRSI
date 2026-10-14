from __future__ import annotations
import copy
import json
from pathlib import Path

import pytest

from swarmrsi.campaign import artifact_workspace_sha256
from swarmrsi.contracts import canonical_sha256
from swarmrsi.retained_results import (
    build_registry, contained, file_sha, output_attestation, read_json,
    select_pending_tasks, verify_winner, write_immutable,
)


def put(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, sort_keys=True) + "\n")


def seal(record, field):
    record.pop(field, None)
    record[field] = canonical_sha256(record)
    return record


@pytest.fixture
def run(tmp_path):
    root, source = tmp_path / "archive", tmp_path / "source"
    source.mkdir()
    profile = {"model": "same-model", "tasks": {
        "won": {"claude_code_score": .1, "transcript_sha256": "c" * 64},
        "unwon": {"claude_code_score": .3, "transcript_sha256": "d" * 64}}}
    profile_path = source / "profile.json"
    put(profile_path, profile)
    (source / "controller.py").write_text("# frozen historical controller\n")
    nodes = []
    for identifier, score, relative in (("root", .05, "root"), ("child", .2, "cells/0_child/artifact")):
        artifact = root / relative
        artifact.mkdir(parents=True)
        (artifact / "run.py").write_text(f"print({score!r})\n")
        (artifact / "output").mkdir()
        (artifact / "output/pred.csv").write_text(f"prediction\n{score}\n")
        evidence_path = (artifact if identifier == "root" else artifact.parent) / "evaluation.json"
        sha = file_sha(artifact / "run.py")
        ev = {"valid": True, "execution_valid": True, "score_bearing_evaluation_attempted": True,
              "metrics": {"aggregate_improvement": score},
              "details": {"candidate_executable_sha256": sha,
                          "submission_output_attestation": output_attestation(artifact / "output")}}
        put(evidence_path, ev)
        # Evidence lives outside the executable workspace in production.
        if identifier == "root":
            evidence_path.rename(root / "root_evaluation.json")
            evidence_path = root / "root_evaluation.json"
        node = {"id": identifier, "parent_id": None if identifier == "root" else "root",
                "artifact_valid": True, "native_metric": score,
                "artifact": "/original/run/" + relative, "executable_relpath": "run.py",
                "executable_sha256": sha, "workspace_sha256": artifact_workspace_sha256(artifact),
                "evaluation_evidence_path": "/original/run/" + evidence_path.relative_to(root).as_posix(),
                "evaluation_evidence_sha256": file_sha(evidence_path)}
        nodes.append(node)
    result = {"action_id": "child", "status": "success", "parent_preserved": True,
              "contract_violations": [], "parent_sha256": nodes[0]["executable_sha256"],
              "final_sha256": nodes[1]["executable_sha256"],
              "final_workspace_sha256": nodes[1]["workspace_sha256"],
              "events": [], "trajectory_sha256": canonical_sha256([])}
    put(root / "cells/0_child/result.json", seal(result, "outcome_sha256"))
    (root / "cells/0_child/trajectory.jsonl").write_text("")
    checkpoint = {"version": "old-tree", "accounting": {"model_calls": 2}, "diagnostics": [],
                  "steps": [{"action_id": "child", "native_metric": .2, "artifact_valid": True}],
                  "events": [], "strategy_state": {"policy": "old"}, "nodes": nodes}
    put(root / "checkpoint.json", seal(checkpoint, "checkpoint_sha256"))
    final = {k: copy.deepcopy(checkpoint[k]) for k in ("version", "accounting", "steps", "diagnostics")}
    final.update({"graph": {"events": []}, "search_strategy_state": checkpoint["strategy_state"],
                  "target_threshold": .1, "finished_at_unix": 1, "stop_reason": "target",
                  "best_evolved_node_id": "child", "best_evolved_native_metric": .2,
                  "best_node_id": "child", "best_native_metric": .2, "root_native_metric": .05,
                  "summary_sha256": canonical_sha256({"version": "old-tree", "steps": checkpoint["steps"],
                                                      "graph_sha256": canonical_sha256([])})})
    put(root / "final.json", final)
    manifest = {"task_id": "won", "model": "same-model", "version": "old-tree", "seed": 0,
                "target": {"comparison": "gt", "threshold": .1},
                "source_hashes": {"profile.json": file_sha(profile_path), "controller.py": file_sha(source / "controller.py")}}
    put(root / "manifest.json", manifest)
    spec = {"archive": str(root), "source_archive": str(source), "task_id": "won", "campaign_id": "old",
            "original_run_root": "/original/run", "profile_source_path": "profile.json",
            "expected_final_sha256": file_sha(root / "final.json"),
            "expected_checkpoint_sha256": file_sha(root / "checkpoint.json"),
            "expected_manifest_sha256": file_sha(root / "manifest.json")}
    return spec, profile_path


def test_preserves_winner_and_full_queue_without_new_version_score(run):
    spec, profile = run
    registry = build_registry([spec], profile)
    winner = registry["winners"][0]
    assert winner["winner_score"] == .2 and winner["original_version"] == "old-tree"
    assert len(winner["nodes"]) == 2
    queue = select_pending_tasks(read_json(profile), registry)
    assert queue["task_count"] == 2 and queue["pending_task_ids"] == ["unwon"]
    assert next(row for row in queue["tasks"] if row["task_id"] == "won")["status"] == "skipped_historical_win"
    assert all(row["new_version_score"] is None for row in queue["tasks"])


@pytest.mark.parametrize("filename", ["cells/0_child/artifact/run.py", "cells/0_child/artifact/output/pred.csv",
                                     "cells/0_child/evaluation.json", "cells/0_child/trajectory.jsonl", "checkpoint.json", "final.json"])
def test_corruption_is_not_a_reason_to_respend(run, filename):
    spec, profile = run
    registry = build_registry([spec], profile)
    path = Path(spec["archive"]) / filename
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="changed"):
        select_pending_tasks(read_json(profile), registry)
    with pytest.raises((ValueError, json.JSONDecodeError)):
        verify_winner(spec, profile)


def test_source_bytes_not_just_source_hashes(run):
    spec, profile = run
    registry = build_registry([spec], profile)
    (Path(spec["source_archive"]) / "controller.py").write_text("changed")
    with pytest.raises(ValueError, match="source changed"):
        select_pending_tasks(read_json(profile), registry)


def test_full_archived_source_is_preserved_beyond_old_declared_subset(run):
    spec, profile = run
    path = Path(spec["source_archive"]) / "not_declared_by_old_manifest.py"
    path.write_text("# full repository archive, not a historical manifest binding\n")
    registry = build_registry([spec], profile)
    path.unlink()
    with pytest.raises(ValueError, match="full retained source archive changed"):
        select_pending_tasks(read_json(profile), registry)


@pytest.mark.parametrize("field,value", [("task_id", "unwon"), ("model", "new-model"),
                                       ("threshold", -.5), ("winner_score", 999), ("winner_node_id", "root")])
def test_resealing_registry_cannot_invent_a_win_without_original_records(run, field, value):
    spec, profile = run
    registry = build_registry([spec], profile)
    registry["winners"][0][field] = value
    seal(registry, "registry_sha256")
    with pytest.raises(ValueError):
        select_pending_tasks(read_json(profile), registry)


@pytest.mark.parametrize("value", [None, float("nan"), float("inf"), True])
def test_missing_or_nonfinite_score_never_wins(run, value):
    spec, profile = run
    p = Path(spec["archive"]) / "checkpoint.json"
    c = read_json(p)
    c["nodes"][1]["native_metric"] = value
    put(p, seal(c, "checkpoint_sha256"))
    spec["expected_checkpoint_sha256"] = file_sha(p)
    with pytest.raises(ValueError):
        verify_winner(spec, profile)


def test_root_only_is_not_evolved_success(run):
    spec, profile = run
    p = Path(spec["archive"]) / "checkpoint.json"
    c = read_json(p); c["nodes"][1]["artifact_valid"] = False
    put(p, seal(c, "checkpoint_sha256"))
    spec["expected_checkpoint_sha256"] = file_sha(p)
    with pytest.raises(ValueError, match="root-only"):
        verify_winner(spec, profile)


@pytest.mark.parametrize("key,value", [("model", "different"), ("task_id", "unwon")])
def test_task_and_model_mismatch(run, key, value):
    spec, profile = run
    path = Path(spec["archive"]) / "manifest.json"
    m = read_json(path); m[key] = value; put(path, m)
    spec["expected_manifest_sha256"] = file_sha(path)
    with pytest.raises(ValueError, match="task/model"):
        verify_winner(spec, profile)


def test_relocated_path_cannot_escape_or_follow_symlink(run):
    spec, profile = run
    root = Path(spec["archive"])
    with pytest.raises(ValueError):
        contained(root, "../source/controller.py")
    (root / "external").symlink_to(profile.parent, target_is_directory=True)
    with pytest.raises(ValueError, match="symlink"):
        build_registry([spec], profile)


def test_duplicate_registry_and_atomic_idempotence(run, tmp_path):
    spec, profile = run
    with pytest.raises(ValueError, match="duplicate"):
        build_registry([spec, spec], profile)
    path = tmp_path / "registry.json"
    registry = build_registry([spec], profile)
    write_immutable(path, registry)
    original = path.read_bytes()
    write_immutable(path, registry)
    with pytest.raises(FileExistsError):
        write_immutable(path, {"wrong": True})
    assert path.read_bytes() == original


def test_different_baseline_or_model_is_not_silently_skipped(run):
    spec, profile = run
    registry = build_registry([spec], profile)
    newer = read_json(profile); newer["model"] = "new-backbone"
    assert select_pending_tasks(newer, registry)["pending_task_ids"] == list(newer["tasks"])
    newer = read_json(profile); newer["tasks"]["won"]["claude_code_score"] = .4
    assert select_pending_tasks(newer, registry)["pending_task_ids"] == list(newer["tasks"])


def test_failed_workspace_is_verified_before_initial_inventory(run):
    spec, profile = run
    root = Path(spec["archive"])
    artifact = root / "cells/failed/failed_artifact"
    artifact.mkdir(parents=True)
    (artifact / "run.py").write_text("this is the retained failed program\n")
    c = read_json(root / "checkpoint.json")
    result = {"action_id": "failed", "status": "implementation_failed",
              "diagnostic_workspace_sha256": artifact_workspace_sha256(artifact),
              "parent_sha256": c["nodes"][0]["executable_sha256"],
              "events": [], "trajectory_sha256": canonical_sha256([])}
    put(artifact.parent / "result.json", seal(result, "outcome_sha256"))
    (artifact.parent / "trajectory.jsonl").write_text("")
    c["diagnostics"] = [{"action_id": "failed", "status": "implementation_failed", "code_parent_id": "root",
                         "artifact": "/original/run/cells/failed/failed_artifact",
                         "diagnostic_workspace_sha256": result["diagnostic_workspace_sha256"],
                         "trajectory_sha256": result["trajectory_sha256"]}]
    put(root / "checkpoint.json", seal(c, "checkpoint_sha256"))
    f = read_json(root / "final.json"); f["diagnostics"] = c["diagnostics"]; put(root / "final.json", f)
    spec.update(expected_checkpoint_sha256=file_sha(root / "checkpoint.json"),
                expected_final_sha256=file_sha(root / "final.json"))
    assert len(verify_winner(spec, profile)["diagnostics"]) == 1
    (artifact / "run.py").write_text("changed before registry construction")
    with pytest.raises(ValueError, match="failed workspace"):
        verify_winner(spec, profile)


def test_tied_evolved_score_is_not_a_win(run):
    spec, profile = run
    root = Path(spec["archive"])
    c = read_json(root / "checkpoint.json"); c["nodes"][1]["native_metric"] = .1
    put(root / "checkpoint.json", seal(c, "checkpoint_sha256"))
    f = read_json(root / "final.json"); f["best_evolved_native_metric"] = .1
    put(root / "final.json", f)
    spec.update(expected_checkpoint_sha256=file_sha(root / "checkpoint.json"),
                expected_final_sha256=file_sha(root / "final.json"))
    with pytest.raises(ValueError, match="strict evolved win"):
        verify_winner(spec, profile)
