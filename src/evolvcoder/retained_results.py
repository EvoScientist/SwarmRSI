"""Verified historical development winners and no-rerun queue selection.

This module never executes a candidate, invokes an evaluator, or supplies a new
ablation score. Original run records are not rewritten when an archive moves.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import tempfile
from typing import Any

from .campaign import artifact_workspace_sha256
from .contracts import canonical_sha256


def require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def file_sha(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def finite(value: Any) -> float:
    require(isinstance(value, (int, float)) and not isinstance(value, bool), "missing numeric score")
    require(math.isfinite(value), "non-finite score")
    return float(value)


def contained(root: Path, relative: str) -> Path:
    parts = PurePosixPath(relative)
    require(bool(relative) and not parts.is_absolute() and ".." not in parts.parts, "unsafe archive path")
    cursor = root
    require(not root.is_symlink(), "symlink archive root")
    for part in parts.parts:
        cursor = cursor / part
        require(not cursor.is_symlink(), "symlink in archive")
    require(cursor.resolve().is_relative_to(root.resolve()), "archive path escape")
    return cursor


def relocate(root: Path, original_root: str, original: str) -> Path:
    source = PurePosixPath(original)
    require(source.is_absolute() and ".." not in source.parts, "unsafe original path")
    try:
        relative = source.relative_to(PurePosixPath(original_root)).as_posix()
    except ValueError as exc:
        raise ValueError("record outside original run root") from exc
    return contained(root, relative)


def verify_seal(record: dict, field: str) -> None:
    body = dict(record)
    digest = body.pop(field, None)
    require(digest == canonical_sha256(body), f"invalid {field}")


def verify_events(events: list[dict]) -> None:
    previous = None
    for index, event in enumerate(events):
        verify_seal(event, "event_sha256")
        require(event["sequence"] == index and event["previous_event_sha256"] == previous,
                "invalid event chain")
        previous = event["event_sha256"]


def inventory(root: Path) -> list[dict]:
    require(root.is_dir() and not root.is_symlink(), "archive unavailable")
    rows = []
    for path in sorted(root.rglob("*")):
        require(not path.is_symlink(), "symlink in archive")
        if path.is_file():
            rows.append({"path": path.relative_to(root).as_posix(),
                         "sha256": file_sha(path), "bytes": path.stat().st_size})
        else:
            require(path.is_dir(), "non-regular archive entry")
    require(bool(rows), "empty archive")
    return rows


def output_attestation(root: Path) -> dict:
    """Exact submission-output-attestation-v1 byte protocol; no re-evaluation."""
    rows = inventory(root)
    digest = hashlib.sha256()
    for row in rows:
        digest.update(row["path"].encode())
        digest.update(b"\0")
        with contained(root, row["path"]).open("rb") as stream:
            for block in iter(lambda: stream.read(1024 * 1024), b""):
                digest.update(block)
        digest.update(b"\0")
    return {"schema_version": "submission-output-attestation-v1", "root": "output",
            "artifact_sha256": digest.hexdigest(), "file_count": len(rows),
            "total_bytes": sum(row["bytes"] for row in rows)}


def verify_winner(spec: dict, profile_path: Path) -> dict:
    """Audit a completed copied run against its independently supplied final hash.

    Full original source is retained separately; only source files declared in
    the original manifest are asserted to be historical runtime bindings.
    """
    root = Path(spec["archive"]).absolute()
    source_root = Path(spec["source_archive"]).absolute()
    profile = read_json(profile_path)
    for name in ("final", "checkpoint", "manifest"):
        require(file_sha(contained(root, name + ".json")) == spec[f"expected_{name}_sha256"],
                f"{name} hash mismatch")
    final, checkpoint, manifest = (read_json(contained(root, name)) for name in
                                  ("final.json", "checkpoint.json", "manifest.json"))
    task_id = spec["task_id"]
    require(task_id in profile["tasks"], "task absent from profile")
    task = profile["tasks"][task_id]
    threshold = finite(task["claude_code_score"])
    require(manifest["task_id"] == task_id and manifest["model"] == profile["model"], "task/model mismatch")
    require(manifest["target"] == {"comparison": "gt", "threshold": threshold}, "baseline target mismatch")
    require(final["target_threshold"] == threshold, "final target mismatch")
    require(final.get("finished_at_unix") is not None and final.get("stop_reason"), "run not terminal")
    require(final["version"] == checkpoint["version"] == manifest["version"], "version mismatch")
    profile_digest = file_sha(profile_path)
    require(manifest["source_hashes"].get(spec["profile_source_path"]) == profile_digest,
            "historical profile mismatch")
    for relative, digest in manifest["source_hashes"].items():
        require(file_sha(contained(source_root, relative)) == digest, f"source hash mismatch: {relative}")
    verify_seal(checkpoint, "checkpoint_sha256")
    verify_events(checkpoint["events"])
    for key in ("accounting", "steps", "diagnostics"):
        require(final[key] == checkpoint[key], f"final/checkpoint {key} mismatch")
    require(final["graph"]["events"] == checkpoint["events"], "final graph mismatch")
    require(final["search_strategy_state"] == checkpoint["strategy_state"], "strategy mismatch")
    require(final["summary_sha256"] == canonical_sha256({
        "version": final["version"], "steps": final["steps"],
        "graph_sha256": canonical_sha256([event["event_sha256"] for event in checkpoint["events"]]),
    }), "summary hash mismatch")
    nodes = {node["id"]: node for node in checkpoint["nodes"]}
    require(len(nodes) == len(checkpoint["nodes"]) and "root" in nodes, "invalid node identities")
    valid = [n for n in nodes.values() if n["artifact_valid"] is True and n["native_metric"] is not None]
    require(bool(valid), "no valid programs")
    for node in valid:
        finite(node["native_metric"])
    evolved = [node for node in valid if node["id"] != "root"]
    require(bool(evolved), "root-only result is not an evolved winner")
    winner = nodes.get(final["best_evolved_node_id"])
    require(winner in evolved and winner["native_metric"] == max(n["native_metric"] for n in evolved),
            "best evolved node mismatch")
    require(final["best_evolved_native_metric"] == winner["native_metric"], "evolved score mismatch")
    require(finite(winner["native_metric"]) > threshold, "not a strict evolved win")
    best = nodes.get(final["best_node_id"])
    require(best in valid and best["native_metric"] == max(n["native_metric"] for n in valid)
            and final["best_native_metric"] == best["native_metric"], "best incumbent mismatch")
    require(final["root_native_metric"] == nodes["root"]["native_metric"], "root score mismatch")

    # Preserve and verify every scored branch, not only the selected success.
    node_receipts = []
    for node in valid:
        artifact = relocate(root, spec["original_run_root"], node["artifact"])
        executable = contained(artifact, node["executable_relpath"])
        evidence = relocate(root, spec["original_run_root"], node["evaluation_evidence_path"])
        require(file_sha(executable) == node["executable_sha256"], "executable hash mismatch")
        require(artifact_workspace_sha256(artifact) == node["workspace_sha256"], "workspace hash mismatch")
        require(file_sha(evidence) == node["evaluation_evidence_sha256"], "evaluation hash mismatch")
        evaluation = read_json(evidence)
        require(evaluation.get("valid") is True and evaluation.get("execution_valid") is True,
                "invalid official evaluation")
        require(evaluation.get("score_bearing_evaluation_attempted") is True, "no official score attempt")
        require(evaluation["metrics"]["aggregate_improvement"] == node["native_metric"], "official score mismatch")
        require(evaluation["details"]["candidate_executable_sha256"] == node["executable_sha256"],
                "evaluation/executable mismatch")
        root_output_missing = node["id"] == "root" and not (artifact / "output").exists()
        attestation = evaluation["details"]["submission_output_attestation"]
        if not root_output_missing:
            require(output_attestation(artifact / "output") == attestation, "scored output mismatch")
        if node["id"] != "root":
            result = read_json(contained(artifact.parent, "result.json"))
            verify_seal(result, "outcome_sha256")
            trajectory = [json.loads(line) for line in contained(artifact.parent, "trajectory.jsonl").read_text().splitlines()]
            require(trajectory == result["events"], "coder trajectory mismatch")
            verify_events(trajectory)
            require(result["trajectory_sha256"] == canonical_sha256([e["event_sha256"] for e in trajectory]),
                    "coder trajectory seal mismatch")
            require(result["action_id"] == node["id"] and result["status"] == "success"
                    and result["parent_preserved"] is True and not result["contract_violations"],
                    "invalid coder result")
            require(result["final_sha256"] == node["executable_sha256"]
                    and result["final_workspace_sha256"] == node["workspace_sha256"], "coder artifact mismatch")
            require(node["parent_id"] in nodes and result["parent_sha256"] == nodes[node["parent_id"]]["executable_sha256"],
                    "parent lineage mismatch")
            steps = [step for step in checkpoint["steps"] if step["action_id"] == node["id"]]
            require(len(steps) == 1 and steps[0]["artifact_valid"] is True
                    and steps[0]["native_metric"] == node["native_metric"], "completed step mismatch")
        node_receipts.append({"id": node["id"], "score": node["native_metric"],
                              "artifact": str(artifact), "evidence": str(evidence),
                              "executable_sha256": node["executable_sha256"],
                              "evaluation_sha256": node["evaluation_evidence_sha256"],
                              "output_attestation": attestation,
                              "output_custody": "original_paid_root_outputs_not_retained" if root_output_missing else "verified_original_bytes"})
    diagnostic_receipts = []
    for diagnostic in checkpoint["diagnostics"]:
        artifact = relocate(root, spec["original_run_root"], diagnostic["artifact"])
        require(artifact_workspace_sha256(artifact) == diagnostic["diagnostic_workspace_sha256"],
                "failed workspace hash mismatch")
        result = read_json(contained(artifact.parent, "result.json"))
        verify_seal(result, "outcome_sha256")
        trajectory = [json.loads(line) for line in contained(artifact.parent, "trajectory.jsonl").read_text().splitlines()]
        verify_events(trajectory)
        require(trajectory == result["events"]
                and canonical_sha256([e["event_sha256"] for e in trajectory]) == diagnostic["trajectory_sha256"]
                and result["trajectory_sha256"] == diagnostic["trajectory_sha256"], "failed trajectory mismatch")
        require(result["action_id"] == diagnostic["action_id"] and result["status"] == diagnostic["status"],
                "failed result identity mismatch")
        require(result["diagnostic_workspace_sha256"] == diagnostic["diagnostic_workspace_sha256"]
                and result["parent_sha256"] == nodes[diagnostic["code_parent_id"]]["executable_sha256"],
                "failed result source mismatch")
        diagnostic_receipts.append({"action_id": diagnostic["action_id"], "artifact": str(artifact),
                                    "workspace_sha256": diagnostic["diagnostic_workspace_sha256"],
                                    "trajectory_sha256": diagnostic["trajectory_sha256"]})
    rows = inventory(root)
    source_rows = inventory(source_root)
    return {"task_id": task_id, "model": manifest["model"], "seed": manifest["seed"],
            "original_version": manifest["version"], "historical_campaign": spec["campaign_id"],
            "result_class": "retained_historical_development_win_not_new_version_measurement",
            "original_run_root": spec["original_run_root"], "archive": str(root),
            "source_archive": str(source_root), "declared_source_hashes": manifest["source_hashes"],
            "source_archive_inventory": source_rows, "source_archive_inventory_sha256": canonical_sha256(source_rows),
            "profile_source_path": spec["profile_source_path"],
            "profile_sha256": profile_digest, "claude_trace_sha256": task["transcript_sha256"],
            "threshold": threshold, "comparison": "gt", "winner_node_id": winner["id"],
            "winner_score": winner["native_metric"], "best_node_id": best["id"], "best_score": best["native_metric"],
            "final_sha256": spec["expected_final_sha256"], "checkpoint_sha256": file_sha(root / "checkpoint.json"),
            "manifest_sha256": file_sha(root / "manifest.json"), "nodes": node_receipts,
            "diagnostics": diagnostic_receipts,
            "archive_inventory": rows, "archive_inventory_sha256": canonical_sha256(rows),
            "input_custody": "original_campaign_identity; raw benchmark inputs are not duplicated in this archive"}


def build_registry(specs: list[dict], profile_path: Path) -> dict:
    require(bool(specs), "no winners specified")
    winners = [verify_winner(spec, profile_path) for spec in specs]
    identities = [(row["task_id"], row["model"], row["final_sha256"]) for row in winners]
    require(len(set(identities)) == len(identities), "duplicate retained run")
    body = {"schema_version": "evolvcoder-retained-winners-v1", "winners": winners}
    return {**body, "registry_sha256": canonical_sha256(body)}


def verify_registry(registry: dict) -> None:
    verify_seal(registry, "registry_sha256")
    require(registry["schema_version"] == "evolvcoder-retained-winners-v1", "unknown registry schema")
    for row in registry["winners"]:
        current = inventory(Path(row["archive"]))
        require(current == row["archive_inventory"] and canonical_sha256(current) == row["archive_inventory_sha256"],
                "retained archive changed; do not silently rerun")
        for relative, digest in row["declared_source_hashes"].items():
            require(file_sha(contained(Path(row["source_archive"]), relative)) == digest, "retained source changed")
        source_rows = inventory(Path(row["source_archive"]))
        require(source_rows == row["source_archive_inventory"]
                and canonical_sha256(source_rows) == row["source_archive_inventory_sha256"],
                "full retained source archive changed")
        # A self-sealed registry is not authority to rename a task or invent a
        # score. Re-derive the ENTIRE row from its retained original records.
        spec = {"archive": row["archive"], "source_archive": row["source_archive"],
                "original_run_root": row["original_run_root"], "task_id": row["task_id"],
                "campaign_id": row["historical_campaign"], "profile_source_path": row["profile_source_path"],
                **{f"expected_{name}_sha256": row[f"{name}_sha256"] for name in ("final", "checkpoint", "manifest")}}
        original = verify_winner(spec, contained(Path(row["source_archive"]), row["profile_source_path"]))
        require(original == row, "retained winner row not bound to original records")


def select_pending_tasks(profile: dict, registry: dict) -> dict:
    """Keep the full task ledger while excluding verified winners from new jobs."""
    verify_registry(registry)
    rows = []
    for task_id, task in profile["tasks"].items():
        matches = [row for row in registry["winners"] if row["task_id"] == task_id
                   and row["model"] == profile["model"] and row["threshold"] == task["claude_code_score"]
                   and row["claude_trace_sha256"] == task["transcript_sha256"]]
        winner = max(matches, key=lambda row: row["winner_score"]) if matches else None
        rows.append({"task_id": task_id, "status": "skipped_historical_win" if winner else "pending_new_validation",
                     "historical_final_sha256": winner["final_sha256"] if winner else None,
                     "historical_score": winner["winner_score"] if winner else None,
                     "new_version_score": None})
    body = {"schema_version": "evolvcoder-retained-winner-queue-v1", "model": profile["model"],
            "registry_sha256": registry["registry_sha256"], "profile_sha256": canonical_sha256(profile),
            "task_count": len(rows), "tasks": rows,
            "pending_task_ids": [row["task_id"] for row in rows if row["status"] == "pending_new_validation"]}
    return {**body, "queue_sha256": canonical_sha256(body)}


def write_immutable(path: Path, value: dict) -> None:
    """Atomic no-clobber publication; exact existing bytes are idempotent."""
    raw = (json.dumps(value, ensure_ascii=False, indent=2, sort_keys=True, allow_nan=False) + "\n").encode()
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    temporary = Path(name)
    try:
        with os.fdopen(descriptor, "wb") as stream:
            stream.write(raw)
            stream.flush()
            os.fsync(stream.fileno())
        try:
            os.link(temporary, path)
        except FileExistsError:
            if path.is_symlink() or path.read_bytes() != raw:
                raise FileExistsError(f"refusing to overwrite {path}")
    finally:
        temporary.unlink()
