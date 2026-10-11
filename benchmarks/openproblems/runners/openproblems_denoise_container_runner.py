#!/usr/bin/env python3
"""Trusted two-stage runner for the OpenProblems denoising evaluator.

The candidate stage has no test file.  The score stage has no candidate source.
This process separation makes ``candidate_saw_hidden_labels=false`` auditable.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import random
from pathlib import Path

import anndata as ad
import numpy as np
from scipy.sparse import issparse


def _dense(matrix) -> np.ndarray:
    return matrix.toarray() if issparse(matrix) else np.asarray(matrix)


def _log_cp10k(matrix) -> np.ndarray:
    values = _dense(matrix).astype(np.float64, copy=True)
    totals = values.sum(axis=1, keepdims=True)
    np.divide(values * 10_000.0, totals, out=values, where=totals != 0)
    values[totals[:, 0] == 0] = 0.0
    return np.log1p(values)


def run_candidate(
    solution: Path,
    train_path: Path,
    output: Path,
    *,
    max_cells: int | None = None,
) -> None:
    spec = importlib.util.spec_from_file_location("candidate_solution", solution)
    if spec is None or spec.loader is None:
        raise RuntimeError("could not import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    model = getattr(module, "tuso_model", None)
    if not callable(model):
        raise RuntimeError("solution.py must define callable tuso_model(adata)")

    source = ad.read_h5ad(train_path)
    if "counts" not in source.layers:
        raise RuntimeError("public train asset lacks layers['counts']")
    if max_cells is not None:
        if max_cells < 1:
            raise RuntimeError("max_cells must be positive")
        source = source[: min(max_cells, source.n_obs)].copy()
    candidate_input = ad.AnnData(
        X=source.layers["counts"].copy(),
        obs=source.obs.copy(),
        var=source.var.copy(),
    )
    candidate_input.obsm["train"] = source.layers["counts"].copy()
    np.random.seed(42)
    random.seed(42)
    result = model(candidate_input)
    if not isinstance(result, ad.AnnData):
        raise RuntimeError("tuso_model must return an AnnData object")
    if "denoised" not in result.obsm:
        raise RuntimeError("tuso_model omitted obsm['denoised']")
    prediction = _dense(result.obsm["denoised"])
    expected_shape = source.shape
    if prediction.shape != expected_shape:
        raise RuntimeError(
            f"prediction shape {prediction.shape} != expected {expected_shape}"
        )
    if not np.isfinite(prediction).all():
        raise RuntimeError("prediction contains non-finite values")
    if np.min(prediction) < 0:
        raise RuntimeError("prediction contains negative molecular counts")
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(output, denoised=prediction)


def score_prediction(prediction_path: Path, test_path: Path, output: Path) -> None:
    prediction = np.load(prediction_path, allow_pickle=False)["denoised"]
    target_data = ad.read_h5ad(test_path)
    if "counts" not in target_data.layers:
        raise RuntimeError("hidden test asset lacks layers['counts']")
    target = target_data.layers["counts"]
    if prediction.shape != target_data.shape:
        raise RuntimeError("persisted prediction shape does not match hidden target")
    target_log = _log_cp10k(target)
    prediction_log = _log_cp10k(prediction)
    mse = float(np.mean(np.square(target_log - prediction_log)))
    payload = {
        "score_1_minus_mse": 1.0 - mse,
        "mse": mse,
        "finite_fraction": float(np.isfinite(prediction).mean()),
        "nonnegative_fraction": float((prediction >= 0).mean()),
        "n_obs": int(prediction.shape[0]),
        "n_vars": int(prediction.shape[1]),
    }
    output.write_text(json.dumps(payload, sort_keys=True) + "\n", encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="mode", required=True)
    candidate = subparsers.add_parser("candidate")
    candidate.add_argument("--solution", type=Path, required=True)
    candidate.add_argument("--train", type=Path, required=True)
    candidate.add_argument("--output", type=Path, required=True)
    candidate.add_argument("--max-cells", type=int, default=None)
    score = subparsers.add_parser("score")
    score.add_argument("--prediction", type=Path, required=True)
    score.add_argument("--test", type=Path, required=True)
    score.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.mode == "candidate":
        run_candidate(
            args.solution,
            args.train,
            args.output,
            max_cells=args.max_cells,
        )
    else:
        score_prediction(args.prediction, args.test, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
