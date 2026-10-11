"""OpenProblems denoising task preparation, custody and official scoring.

The molecule split and the metric are the official ones, restated here so the
controller can run them without the upstream nextflow stack:

* ``split_molecules`` is the official binomial molecule split used by the task's
  ``process_dataset`` step (originally from czbiohub/molecular-cross-validation,
  MIT). It is what produced the released train/test pair, and this module uses
  the same procedure again to carve the controller-held public validation split
  out of the public train counts.
* ``mse`` is the official ``mse`` metric: normalize each cell to 10,000 counts,
  ``log1p``, then mean squared error. It is reimplemented on numpy/scipy rather
  than scanpy/scprep so the controller needs no extra runtime; the formula is
  asserted against a fixture rather than assumed.

The private test counts never enter the task's public data, the candidate's
mounts, or any model context. A public validation score is a search signal, not
an official result.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

TASK = "openproblems-v1-tenx-1k-pbmc-denoising"
METRIC = {"name": "mse", "direction": "minimize", "role": "public_validation"}
SCHEMA = "evolvcoder-openproblems-custody-v1"
TARGET_SUM = 10_000.0
DEFAULT_TRAIN_FRAC = 0.9
DEFAULT_SEED = 0


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: dict) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def split_molecules(umis, data_split: float, overlap_factor: float = 0.0, random_state=None):
    """Official binomial molecule split (czbiohub molecular-cross-validation, MIT)."""
    import numpy as np
    if random_state is None:
        random_state = np.random.RandomState()
    umis_x_disjoint = random_state.binomial(umis, data_split - overlap_factor)
    umis_y_disjoint = random_state.binomial(
        umis - umis_x_disjoint, (1 - data_split) / (1 - data_split + overlap_factor))
    overlap = umis - umis_x_disjoint - umis_y_disjoint
    return umis_x_disjoint + overlap, umis_y_disjoint + overlap


def _dense(matrix):
    import numpy as np
    from scipy.sparse import issparse
    return np.asarray(matrix.todense() if issparse(matrix) else matrix, dtype=np.float64)


def normalize_cp10k_log1p(matrix):
    """normalize_total(target_sum=10_000) then log1p, matching the official metric."""
    import numpy as np
    values = _dense(matrix)
    totals = values.sum(axis=1, keepdims=True)
    # A cell with no counts stays zero instead of becoming nan, which is what
    # scanpy's normalize_total does with its zero-count guard.
    scale = np.divide(TARGET_SUM, totals, out=np.zeros_like(totals), where=totals > 0)
    return np.log1p(values * scale)


def mse(denoised, truth) -> float:
    """Official denoising mse between normalized predictions and held-out counts."""
    import numpy as np
    left, right = normalize_cp10k_log1p(denoised), normalize_cp10k_log1p(truth)
    if left.shape != right.shape:
        raise ValueError(f"prediction shape {left.shape} does not match the target {right.shape}")
    return float(np.mean((left - right) ** 2))


def read_counts(path: Path):
    import anndata as ad
    adata = ad.read_h5ad(Path(path))
    if "counts" not in adata.layers:
        raise ValueError(f"{Path(path).name} has no counts layer")
    return adata


def read_prediction(output_dir: Path):
    """Accept the sparse or dense denoised matrix the candidate wrote."""
    import numpy as np
    output_dir = Path(output_dir)
    sparse_path, dense_path = output_dir / "denoised.npz", output_dir / "denoised.npy"
    if sparse_path.is_file():
        from scipy.sparse import load_npz
        try:
            return load_npz(sparse_path)
        except ValueError:
            with np.load(sparse_path) as bundle:
                if "denoised" not in bundle:
                    raise ValueError("denoised.npz must contain a 'denoised' array")
                return bundle["denoised"]
    if dense_path.is_file():
        return np.load(dense_path)
    raise ValueError("candidate produced no denoised.npz or denoised.npy in OUTPUT_DIR")


def verify_assets(custody_path: Path) -> dict:
    """Re-check every bound file before the task is used for anything."""
    custody_path = Path(custody_path)
    record = json.loads(custody_path.read_text())
    sealed = dict(record)
    digest = sealed.pop("custody_sha256")
    if record.get("schema_version") != SCHEMA or canonical(sealed) != digest:
        raise ValueError("OpenProblems custody schema or seal mismatch")
    root = custody_path.parent
    for relative, expected in record["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"OpenProblems asset changed: {relative}")
    return record


EVALUATOR = '''"""Public validation scorer: controller-held split only, never the official test."""
import json
import os
from pathlib import Path

from evolvcoder.openproblems_tasks import METRIC, mse, read_counts, read_prediction

TASK_ROOT = Path(__file__).resolve().parent.parent
truth = read_counts(TASK_ROOT / "evaluation/public_validation.h5ad")
prediction = read_prediction(Path(os.environ["OUTPUT_DIR"]))
Path("score.json").write_text(json.dumps({METRIC["name"]: mse(prediction, truth.layers["counts"])}))
'''

README = '''# OpenProblems denoising: {task}

Denoise the single-cell molecule counts in `train.h5ad`.

* Input: `$DATA_DIR/train.h5ad`, an AnnData with `layers["counts"]` of shape {shape}.
* Output: write the denoised matrix to `$OUTPUT_DIR/denoised.npz` (scipy sparse) or
  `$OUTPUT_DIR/denoised.npy` (dense), with exactly the same shape as the input.
* Score: official OpenProblems `mse` (lower is better) after normalizing each cell to
  10,000 counts and applying `log1p`.

The score you receive during search is computed on a controller-held validation
split carved from the public counts with the official binomial molecule split
(train_frac={train_frac}, seed={seed}). The official held-out test counts are not
available here and are never part of this task's data.
'''

STARTER = '''"""Deterministic observed-count baseline for OpenProblems denoising."""
import os
from pathlib import Path

import numpy as np
from scipy.sparse import csr_matrix, issparse, save_npz

import anndata as ad


def denoise(counts):
    """Return the observed counts unchanged; a real method should improve on this."""
    return counts


def main():
    adata = ad.read_h5ad(Path(os.environ["DATA_DIR"]) / "train.h5ad")
    denoised = denoise(adata.layers["counts"])
    output = Path(os.environ["OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    if issparse(denoised):
        save_npz(output / "denoised.npz", csr_matrix(denoised))
    else:
        np.save(output / "denoised.npy", np.asarray(denoised, dtype=np.float32))


if __name__ == "__main__":
    main()
'''


def official_manifest(source: Path, manifest: Path | None = None) -> tuple[Path, dict]:
    """The sealed manifest that binds the released train/test bytes."""
    candidates = [manifest] if manifest is not None else [
        source / "dataset_manifest.json", source.parent / "dataset_manifest.json"]
    for path in candidates:
        if path is not None and Path(path).is_file():
            return Path(path), json.loads(Path(path).read_text())
    raise FileNotFoundError("no dataset_manifest.json beside or above the official assets")


def prepare(*, source: Path, destination: Path, seed: int = DEFAULT_SEED,
            train_frac: float = DEFAULT_TRAIN_FRAC, manifest_path: Path | None = None) -> dict:
    """Materialize a task package; the official test counts stay controller-only."""
    import anndata as ad
    import numpy as np
    from scipy.sparse import csr_matrix, issparse

    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("destination must be a new directory")
    manifest_path, manifest = official_manifest(source, manifest_path)
    for name, expected in manifest["sources"].items():
        actual = file_sha256(source / name)
        if actual != expected["sha256"]:
            raise ValueError(f"official asset {name} does not match its manifest hash")
    if not 0 < train_frac < 1:
        raise ValueError("train_frac must lie strictly between 0 and 1")

    train = read_counts(source / "train.h5ad")
    counts = train.layers["counts"]
    counts = csr_matrix(counts) if not issparse(counts) else counts.tocsr().copy()
    counts.data = np.asarray(counts.data).round().astype(int)
    sub_train_data, sub_val_data = split_molecules(
        counts.data, train_frac, 0.0, np.random.RandomState(seed))
    sub_train, sub_val = counts.copy(), counts.copy()
    sub_train.data, sub_val.data = sub_train_data, sub_val_data
    sub_train.eliminate_zeros()
    sub_val.eliminate_zeros()

    (destination / "problem/data").mkdir(parents=True)
    (destination / "evaluation").mkdir()
    (destination / "private").mkdir()
    ad.AnnData(layers={"counts": sub_train}, obs=train.obs[[]], var=train.var[[]],
               uns=dict(train.uns)).write_h5ad(destination / "problem/data/train.h5ad")
    ad.AnnData(layers={"counts": sub_val}, obs=train.obs[[]], var=train.var[[]]
               ).write_h5ad(destination / "evaluation/public_validation.h5ad")
    (destination / "evaluation/evaluator.py").write_text(EVALUATOR)
    (destination / "problem/README.md").write_text(README.format(
        task=TASK, shape=tuple(train.shape), train_frac=train_frac, seed=seed))
    (destination / "metadata.json").write_text(json.dumps({
        "task_id": TASK, "search_metric": METRIC, "search_target": None,
        "dataset_id": str(train.uns.get("dataset_id", "")),
        "shape": list(train.shape), "public_split": {
            "procedure": "official binomial molecule split (process_dataset/helper.split_molecules)",
            "train_frac": train_frac, "seed": seed, "source": "public train counts"},
        "official_test_visible_to_candidate": False}, indent=2, sort_keys=True) + "\n")
    for name in ("train.h5ad", "test.h5ad"):
        (destination / "private" / name).write_bytes((source / name).read_bytes())
    (destination / "private/dataset_manifest.json").write_bytes(manifest_path.read_bytes())
    (destination / "starter").mkdir()
    (destination / "starter/run.py").write_text(STARTER)

    files = {str(path.relative_to(destination)): {"sha256": file_sha256(path), "bytes": path.stat().st_size}
             for path in sorted(destination.rglob("*")) if path.is_file()}
    body = {"schema_version": SCHEMA, "task_id": TASK, "metric": METRIC,
            "official_manifest_sha256": canonical(manifest), "public_split_seed": seed,
            "public_split_train_frac": train_frac,
            "candidate_visible": ["problem/README.md", "problem/data/train.h5ad", "starter/run.py"],
            "controller_only": ["evaluation/public_validation.h5ad", "evaluation/evaluator.py",
                                "private/train.h5ad", "private/test.h5ad", "private/dataset_manifest.json"],
            "files": files}
    record = {**body, "custody_sha256": canonical(body)}
    (destination / "custody.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
    return record


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="materialize a task package from official assets")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--seed", type=int, default=DEFAULT_SEED)
    build.add_argument("--train-frac", type=float, default=DEFAULT_TRAIN_FRAC)
    build.add_argument("--manifest", type=Path, help="sealed dataset manifest; defaults beside or above --source")
    check = commands.add_parser("verify", help="re-check a prepared package against its custody")
    check.add_argument("--custody", type=Path, required=True)
    grade = commands.add_parser("score", help="score a denoised matrix against held-out counts")
    grade.add_argument("--output-dir", type=Path, required=True)
    grade.add_argument("--truth", type=Path, required=True)
    grade.add_argument("--result", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "prepare":
        record = prepare(source=args.source, destination=args.destination, seed=args.seed,
                         train_frac=args.train_frac, manifest_path=args.manifest)
        print(json.dumps({"task_id": record["task_id"], "files": len(record["files"]),
                          "custody_sha256": record["custody_sha256"], "model_calls": 0}))
        return 0
    if args.command == "verify":
        record = verify_assets(args.custody)
        print(json.dumps({"task_id": record["task_id"], "verified_files": len(record["files"])}))
        return 0
    truth = read_counts(args.truth)
    value = mse(read_prediction(args.output_dir), truth.layers["counts"])
    args.result.write_text(json.dumps({METRIC["name"]: value}) + "\n")
    print(json.dumps({METRIC["name"]: value}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
