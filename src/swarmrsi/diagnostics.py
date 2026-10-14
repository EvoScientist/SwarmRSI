"""Controller-owned diagnostic artifact roles, separate from scientific status.

A digest identifies evidence; it does not grant permission to seed a repair.
Legacy rows remain readable evidence and acquire no implicit repair authority.
"""
from __future__ import annotations

from pathlib import PurePosixPath
from typing import Mapping, Any


FAILED_WORKSPACE = "retained_failed_workspace"
FINAL_PROGRAM = "retained_final_program_evidence"


def _sha(value: Any) -> bool:
    return isinstance(value, str) and len(value) == 64 and all(c in "0123456789abcdef" for c in value)


def diagnostic_artifact_fields(*, relpath: str | None, digest: str | None) -> dict:
    """Derive role from the actual CoderOutcome, never from model prose."""
    if relpath is None:
        if digest is not None:
            raise ValueError("diagnostic digest without retained failed workspace")
        return {"diagnostic_artifact_kind": FINAL_PROGRAM, "diagnostic_artifact_relpath": None}
    if relpath != "failed_artifact" or not _sha(digest):
        raise ValueError("invalid retained failed workspace contract")
    return {"diagnostic_artifact_kind": FAILED_WORKSPACE, "diagnostic_artifact_relpath": relpath}


def is_repair_seed(diagnostic: Mapping[str, Any]) -> bool:
    """Pure eligibility shared by admission and routing; Executor rechecks bytes.

    Evaluation failures are evidence about the evaluation chain, even when an
    attempted workspace was retained. They never become implementation repairs.
    """
    return (
        diagnostic.get("diagnostic_artifact_kind") == FAILED_WORKSPACE
        and diagnostic.get("diagnostic_artifact_relpath") == "failed_artifact"
        and diagnostic.get("status") in {"implementation_failed", "plan_blocked"}
        and _sha(diagnostic.get("diagnostic_workspace_sha256"))
        and isinstance(diagnostic.get("artifact"), str)
        and PurePosixPath(diagnostic["artifact"]).name == "failed_artifact"
    )
