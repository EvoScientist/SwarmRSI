#!/usr/bin/env python3
"""Execute one SVG candidate on a truth-free AnnData object in Docker."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import anndata as ad
import numpy as np


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("swarmrsi_svg_solution", path)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise RuntimeError("solution.py must define tuso_model(adata)")
    return candidate


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    adata = ad.read_h5ad(args.input)
    if "true_spatial_var_score" in adata.var:
        raise RuntimeError("sealed input unexpectedly contains reference labels")
    expected_names = np.asarray(adata.var_names.astype(str), dtype=str)
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = _load_solution(args.solution)(adata)
    result = adata if result is None else result
    if "pred_spatial_var_score" not in result.var:
        raise RuntimeError("candidate omitted var['pred_spatial_var_score']")
    if not np.array_equal(np.asarray(result.var_names.astype(str)), expected_names):
        raise RuntimeError("candidate changed or reordered var_names")
    prediction = np.asarray(result.var["pred_spatial_var_score"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prediction=prediction,
        var_names=expected_names,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
