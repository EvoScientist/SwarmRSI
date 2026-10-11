"""A settled, unavailable validation is not a scientific trial measurement."""
from __future__ import annotations

from copy import deepcopy
import os
from pathlib import Path

from .contracts import canonical_sha256
from .learning_evidence import source_ref, verify_ref
from .retained_results import read_json, require, verify_seal, write_immutable


def verify_unavailable(record: dict) -> dict:
    verify_seal(record, "failure_sha256")
    require(record["schema_version"] == "evolvcoder-unavailable-validation-v1"
            and record["reason"] == "settled_validation_missing_final",
            "unknown validation failure record")
    root = Path(record["trial_root"])
    require(root.is_absolute() and root.name == record["trial_id"], "validation failure root mismatch")
    required = {"identity.json", "launch.json", "exit.json"}
    refs = record["source_refs"]
    require(required.issubset(refs), "validation failure lacks original settlement records")
    for name, ref in refs.items():
        path = Path(name)
        require(not path.is_absolute() and ".." not in path.parts, "unsafe validation failure reference")
        require(verify_ref(ref).resolve() == (root / path).resolve(), "validation failure reference location changed")
    identity = read_json(root / "identity.json")
    runner = read_json(verify_ref(record["runner_ref"]))
    require(Path(record["runner_ref"]["path"]).resolve() == (root.parent.parent / "runner.json").resolve()
            and canonical_sha256(runner) == identity["runner_sha256"], "validation failure runner context changed")
    launch, ended = read_json(root / "launch.json"), read_json(root / "exit.json")
    require(canonical_sha256(identity) == record["trial_id"]
            and launch["runner_sha256"] == identity["runner_sha256"], "validation failure identity mismatch")
    require(type(ended.get("returncode")) is int, "validation process termination is unknown")
    require(not os.path.lexists(root / "run/final.json"), "terminal exists; must verify it rather than label unavailable")
    require(record["cost_usd"] is None and record["accounting"] is None
            and record["scientific_verdict"] is None, "unavailable validation cannot impute cost, usage or verdict")
    return identity


def unavailable_record(root: Path) -> dict:
    root = root.resolve()
    names = ["identity.json", "launch.json", "exit.json"]
    optional = ["process.json", "execution.log", "run/session_inputs.json", "run/session_state.json",
                "run/manifest.json", "run/checkpoint.json", "run/live_summary.json", "run/root_evaluation.json"]
    names += [name for name in optional if (root / name).is_file()]
    # Keep the already-persisted branches and proposal journals reachable. Raw
    # outputs stay in their original locations, never rewritten as a final.
    for pattern in ("run/planner_batches/*.json", "run/cells/*/result.json",
                    "run/cells/*/trajectory.jsonl", "run/cells/*/evaluation.json"):
        names += [str(path.relative_to(root)) for path in sorted(root.glob(pattern)) if path.is_file()]
    body = {"schema_version": "evolvcoder-unavailable-validation-v1",
            "trial_root": str(root), "trial_id": root.name,
            "runner_ref": source_ref(root.parent.parent / "runner.json"),
            "reason": "settled_validation_missing_final", "source_refs": {name: source_ref(root / name) for name in names},
            "cost_usd": None, "accounting": None, "scientific_verdict": None}
    record = {**body, "failure_sha256": canonical_sha256(body)}
    verify_unavailable(record)
    write_immutable(root / "unavailable.json", record)
    return record


class ValidationUnavailable(RuntimeError):
    """Required trial ended without a usable final; no implicit restart."""

    def __init__(self, evidence: dict):
        verify_unavailable(evidence)
        self.evidence = deepcopy(evidence)
        ended = read_json(Path(evidence["trial_root"]) / "exit.json")
        # No arbitrary provider error text or credentials in this message.
        super().__init__(f"validation produced no terminal evidence: returncode={ended['returncode']}; not relaunched")
