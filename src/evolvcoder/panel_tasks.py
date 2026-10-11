"""Task packages for the TusoAI comparison panel, built from upstream templates.

TusoAI states a task in one file: `load_data` says how the data is read, `evaluate`
is the metric, and the initial `tuso_model` is the baseline method. Those are the
same three things this harness needs as `problem/data`, `evaluation/evaluator.py`
and `starter/run.py` -- the difference is that we also hold a split back from the
candidate and seal every file, so a score can be checked rather than believed.

This module is the generic half of that translation: custody, verification and the
artifact readers. The metric is read from the task's own metadata rather than fixed
here, because each of the six panel tasks has a different one. What stays
task-specific is only how the raw data is split, which each task's `prepare.py`
does in about thirty lines.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

SCHEMA = "evolvcoder-panel-custody-v1"
DIRECTIONS = ("minimize", "maximize")


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def check_metric(metric: dict) -> dict:
    if (not isinstance(metric, dict) or set(metric) != {"name", "direction", "role"}
            or not isinstance(metric["name"], str) or not metric["name"]
            or metric["direction"] not in DIRECTIONS or metric["role"] != "public_validation"):
        raise ValueError("panel metric must be {name, direction in minimize|maximize, role=public_validation}")
    return metric


def seal(root: Path, *, task_id: str, metric: dict, candidate_visible, controller_only,
         extra: dict | None = None) -> dict:
    """Write custody.json binding every file this task package ships.

    `candidate_visible` and `controller_only` are recorded separately because the
    split between them is the whole point: the adapter refuses to start if any
    candidate mount overlaps a controller-only path.
    """
    root = Path(root)
    check_metric(metric)
    visible, hidden = sorted(candidate_visible), sorted(controller_only)
    overlap = set(visible) & set(hidden)
    if overlap:
        raise ValueError(f"file listed as both candidate-visible and controller-only: {sorted(overlap)}")
    files = {}
    for relative in visible + hidden + ["metadata.json"]:
        path = root / relative
        if not path.is_file() or path.is_symlink():
            raise ValueError(f"custody needs a regular file at {relative}")
        files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    record = {"schema_version": SCHEMA, "task_id": task_id, "metric": dict(metric),
              "candidate_visible": visible, "controller_only": hidden, "files": files,
              **(extra or {})}
    record["custody_sha256"] = canonical(record)
    (root / "custody.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def verify_assets(custody_path: Path) -> dict:
    """Re-check the seal and every bound file before the task is used for anything."""
    custody_path = Path(custody_path)
    record = json.loads(custody_path.read_text())
    sealed = dict(record)
    digest = sealed.pop("custody_sha256", None)
    if record.get("schema_version") != SCHEMA or canonical(sealed) != digest:
        raise ValueError("panel custody schema or seal mismatch")
    check_metric(record["metric"])
    root = custody_path.parent
    for relative, expected in record["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"panel asset changed: {relative}")
    return record


def read_prediction(output_dir: Path, names):
    """Read whichever declared artifact the candidate wrote, in the declared order.

    Exactly one must be present: a candidate that writes two is not choosing, and
    scoring the first one found would hide that.
    """
    import numpy as np
    output_dir = Path(output_dir)
    found = [n for n in names if (output_dir / n).is_file()]
    if len(found) != 1:
        raise ValueError(f"expected exactly one of {list(names)} in the output directory, found {found}")
    name = found[0]
    path = output_dir / name
    if name.endswith(".npz"):
        from scipy.sparse import load_npz
        return load_npz(path)
    if name.endswith(".npy"):
        return np.load(path, allow_pickle=False)
    if name.endswith(".csv"):
        import pandas as pd
        return pd.read_csv(path)
    raise ValueError(f"unsupported prediction artifact {name}")
