#!/usr/bin/env python3
"""Run one spatial-decomposition candidate without exposing true proportions."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import anndata as ad
import numpy as np


class DecompContractError(ValueError):
    pass


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("evocoder_decomp_solution", path)
    if spec is None or spec.loader is None:
        raise DecompContractError("cannot import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise DecompContractError(
            "solution.py must define tuso_model(single_cell_adata, spatial_adata)"
        )
    return candidate


def extract_prediction(result, spatial: ad.AnnData) -> np.ndarray:
    if isinstance(result, ad.AnnData):
        if not result.obs_names.equals(spatial.obs_names):
            raise DecompContractError("candidate changed or reordered spot obs_names")
        if "proportions_pred" not in result.obsm:
            raise DecompContractError("candidate omitted obsm['proportions_pred']")
        values = np.asarray(result.obsm["proportions_pred"], dtype=float)
    else:
        values = np.asarray(result, dtype=float)
    n_cell_types = len(np.asarray(spatial.uns["cell_type_names"]))
    expected_shape = (spatial.n_obs, n_cell_types)
    if values.shape != expected_shape:
        raise DecompContractError(
            f"proportions_pred shape mismatch: {values.shape} != {expected_shape}"
        )
    if not np.isfinite(values).all():
        raise DecompContractError("proportions_pred contains non-finite values")
    if np.any(values < -1e-12):
        raise DecompContractError("proportions_pred contains negative values")
    if not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=1e-6):
        raise DecompContractError("each proportions_pred row must sum to one")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--single-cell", type=Path, required=True)
    parser.add_argument("--spatial", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    single_cell = ad.read_h5ad(args.single_cell)
    spatial = ad.read_h5ad(args.spatial)
    if "cell_type" not in single_cell.obs:
        raise DecompContractError("single-cell input is missing obs['cell_type']")
    if "proportions_true" in spatial.obsm:
        raise DecompContractError("candidate-visible spatial input contains gold proportions")
    if "cell_type_names" not in spatial.uns:
        raise DecompContractError("spatial input is missing uns['cell_type_names']")
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = _load_solution(args.solution)(single_cell, spatial)
    prediction = extract_prediction(result, spatial)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prediction=prediction,
        obs_names=np.asarray(spatial.obs_names.astype(str), dtype=str),
        cell_type_names=np.asarray(spatial.uns["cell_type_names"]).astype(str),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
