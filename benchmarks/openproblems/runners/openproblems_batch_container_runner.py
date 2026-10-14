#!/usr/bin/env python3
"""Run and attest one OpenProblems batch-integration candidate."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse


class BatchContractError(ValueError):
    pass


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("swarmrsi_batch_solution", path)
    if spec is None or spec.loader is None:
        raise BatchContractError("cannot import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise BatchContractError("solution.py must define tuso_model(adata)")
    return candidate


def _finite(name: str, value) -> None:
    data = value.data if sparse.issparse(value) else np.asarray(value)
    if not np.isfinite(data).all():
        raise BatchContractError(f"{name} contains non-finite values")


def sanitize_result(source: ad.AnnData, result: ad.AnnData) -> ad.AnnData:
    if not isinstance(result, ad.AnnData):
        raise BatchContractError("batch candidate must return AnnData")
    if not result.obs_names.equals(source.obs_names):
        raise BatchContractError("candidate changed or reordered obs_names")
    output = ad.AnnData(obs=source.obs.iloc[:, 0:0].copy())
    output.uns["dataset_id"] = str(source.uns.get("dataset_id", "unknown"))
    output.uns["normalization_id"] = str(
        source.uns.get("normalization_id", "unknown")
    )
    output.uns["method_id"] = "swarmrsi_candidate"
    found = False
    if "corrected_counts" in result.layers:
        matrix = result.layers["corrected_counts"]
        if tuple(matrix.shape) != tuple(source.shape):
            raise BatchContractError("corrected_counts shape does not match input")
        _finite("corrected_counts", matrix)
        output = ad.AnnData(
            X=sparse.csr_matrix((source.n_obs, source.n_vars)),
            obs=source.obs.iloc[:, 0:0].copy(),
            var=source.var.iloc[:, 0:0].copy(),
        )
        output.layers["corrected_counts"] = matrix.copy()
        found = True
    if "X_emb" in result.obsm:
        embedding = np.asarray(result.obsm["X_emb"])
        if embedding.ndim != 2 or embedding.shape[0] != source.n_obs:
            raise BatchContractError("X_emb must have shape (n_obs, n_components)")
        if embedding.shape[1] < 1:
            raise BatchContractError("X_emb has no components")
        _finite("X_emb", embedding)
        output.obsm["X_emb"] = embedding.copy()
        found = True
    graph_keys = {"connectivities", "distances"} & set(result.obsp.keys())
    if graph_keys and graph_keys != {"connectivities", "distances"}:
        raise BatchContractError(
            "graph output requires both connectivities and distances"
        )
    if graph_keys:
        for key in ("connectivities", "distances"):
            graph = result.obsp[key]
            if tuple(graph.shape) != (source.n_obs, source.n_obs):
                raise BatchContractError(f"{key} must have shape (n_obs, n_obs)")
            _finite(key, graph)
            output.obsp[key] = graph.copy()
        found = True
    if not found:
        raise BatchContractError(
            "candidate must output corrected_counts, X_emb, or a neighbor graph"
        )
    output.uns["dataset_id"] = str(source.uns.get("dataset_id", "unknown"))
    output.uns["normalization_id"] = str(
        source.uns.get("normalization_id", "unknown")
    )
    output.uns["method_id"] = "swarmrsi_candidate"
    return output


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    source = ad.read_h5ad(args.input)
    if "cell_type" not in source.obs or "batch" not in source.obs:
        raise BatchContractError(
            "batch input requires obs['cell_type'] and obs['batch']"
        )
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = _load_solution(args.solution)(source.copy())
    clean = sanitize_result(source, result)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    clean.write_h5ad(args.output, compression="gzip")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
