#!/usr/bin/env python3
"""Run one visualization candidate on a gold-free AnnData input."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import anndata as ad
import numpy as np


GOLD_UNS = {
    "between_waypoint_distances",
    "label_centroids",
    "waypoint_centroid_distances",
    "between_centroid_distances",
}


class VisualContractError(ValueError):
    pass


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("swarmrsi_visual_solution", path)
    if spec is None or spec.loader is None:
        raise VisualContractError("cannot import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise VisualContractError("solution.py must define tuso_model(adata)")
    return candidate


def extract_embedding(result, source: ad.AnnData) -> np.ndarray:
    if isinstance(result, ad.AnnData):
        if not result.obs_names.equals(source.obs_names):
            raise VisualContractError("candidate changed or reordered obs_names")
        if "X_emb" not in result.obsm:
            raise VisualContractError("candidate omitted obsm['X_emb']")
        values = np.asarray(result.obsm["X_emb"], dtype=float)
    else:
        values = np.asarray(result, dtype=float)
    if values.ndim != 2 or values.shape[0] != source.n_obs:
        raise VisualContractError("X_emb must have shape (n_obs, n_components)")
    if values.shape[1] < 1:
        raise VisualContractError("X_emb has no components")
    if not np.isfinite(values).all():
        raise VisualContractError("X_emb contains non-finite values")
    return values


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = ad.read_h5ad(args.input)
    leaked_obs = {"cell_type", "is_waypoint"} & set(source.obs.columns)
    leaked_uns = GOLD_UNS & set(source.uns.keys())
    if leaked_obs or leaked_uns:
        raise VisualContractError(
            "candidate-visible input contains gold fields: "
            + ", ".join(sorted(leaked_obs | leaked_uns))
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = _load_solution(args.solution)(source)
    embedding = extract_embedding(result, source)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        embedding=embedding,
        obs_names=np.asarray(source.obs_names.astype(str), dtype=str),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
