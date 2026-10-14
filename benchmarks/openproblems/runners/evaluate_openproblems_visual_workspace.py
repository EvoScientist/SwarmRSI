#!/usr/bin/env python3
"""Replay raw Visual components without inventing TusoAI normalization."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Any

import anndata as ad
import numpy as np
from scipy.stats import pearsonr, spearmanr
from sklearn.manifold import trustworthiness
from sklearn.metrics import pairwise_distances


class VisualEvaluationError(ValueError):
    pass


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _processed_distances(embedding: np.ndarray, gold: ad.AnnData) -> dict[str, np.ndarray]:
    is_waypoint = gold.obs["is_waypoint"].to_numpy(dtype=bool)
    labels = np.unique(gold.obs["cell_type"].astype(str).to_numpy())
    centroids = np.stack([
        embedding[gold.obs["cell_type"].astype(str).to_numpy() == label].mean(axis=0)
        for label in labels
    ])
    between_waypoints = pairwise_distances(embedding[is_waypoint], metric="euclidean", n_jobs=-2)
    np.fill_diagonal(between_waypoints, 0)
    waypoint_centroids = pairwise_distances(embedding[is_waypoint], centroids, metric="euclidean", n_jobs=-2)
    np.fill_diagonal(waypoint_centroids, 0)
    between_centroids = pairwise_distances(centroids, metric="euclidean", n_jobs=-2)
    np.fill_diagonal(between_centroids, 0)
    return {
        "between_waypoint_distances": between_waypoints,
        "waypoint_centroid_distances": waypoint_centroids,
        "between_centroid_distances": between_centroids,
    }


def _calculate_radii(matrix: Any, n_neighbors: int, random_state: int) -> np.ndarray:
    try:
        from umap.umap_ import fuzzy_simplicial_set, nearest_neighbors
    except ImportError as exc:
        raise VisualEvaluationError("density preservation requires the pinned UMAP runtime") from exc
    indices, distances, _ = nearest_neighbors(matrix, n_neighbors, "euclidean", {}, False, random_state, verbose=False)
    graph, _, _, graph_distances = fuzzy_simplicial_set(
        matrix, n_neighbors, random_state, "euclidean", {}, indices, distances,
        verbose=False, return_dists=True,
    )
    graph = graph.tocoo(); graph.sum_duplicates(); graph.eliminate_zeros()
    mu_sum = np.zeros(graph.shape[1], dtype=np.float32)
    radii = np.zeros(graph.shape[1], dtype=np.float32)
    for row, col, weight in zip(graph.row, graph.col, graph.data):
        distance = graph_distances[row, col]
        radii[row] += weight * distance; radii[col] += weight * distance
        mu_sum[row] += weight; mu_sum[col] += weight
    return np.log(1e-8 + radii / mu_sum)


def score_raw_components(embedding: np.ndarray, gold: ad.AnnData, *, include_density: bool) -> dict[str, float | None]:
    required = {"between_waypoint_distances", "waypoint_centroid_distances", "between_centroid_distances"}
    if "cell_type" not in gold.obs or "is_waypoint" not in gold.obs or not required.issubset(gold.uns):
        raise VisualEvaluationError("gold solution lacks labels, waypoints, or reference distances")
    derived = _processed_distances(embedding, gold)
    components: dict[str, float | None] = {
        "trustworthiness": float(trustworthiness(gold.layers["normalized"], embedding, n_neighbors=15, metric="euclidean")),
        "waypoint_distance_correlation": float(spearmanr(gold.uns["between_waypoint_distances"], derived["between_waypoint_distances"], axis=None).correlation),
        "centroid_distance_correlation": float(spearmanr(gold.uns["waypoint_centroid_distances"], derived["waypoint_centroid_distances"], axis=None).correlation),
        "label_distance_correlation": float(spearmanr(gold.uns["between_centroid_distances"], derived["between_centroid_distances"], axis=None).correlation),
        "density_preservation": None,
    }
    if include_density:
        original_radii = _calculate_radii(gold.layers["normalized"], 30, 42)
        embedded_radii = _calculate_radii(embedding, 30, 42)
        components["density_preservation"] = float(pearsonr(original_radii, embedded_radii)[0])
    return components


def evaluate_dataset(*, solution_py: Path, dataset_path: Path, gold_path: Path,
                     runner: Path, seed: int, include_density: bool) -> dict[str, Any]:
    dataset = ad.read_h5ad(dataset_path)
    gold = ad.read_h5ad(gold_path)
    if not dataset.obs_names.equals(gold.obs_names):
        raise VisualEvaluationError("dataset and gold obs_names differ")
    with tempfile.TemporaryDirectory(prefix="swarmrsi-visual-") as temp_dir:
        output = Path(temp_dir) / "embedding.npz"
        completed = subprocess.run([
            sys.executable, str(runner), "--solution", str(solution_py),
            "--input", str(dataset_path), "--output", str(output), "--seed", str(seed),
        ], check=False, capture_output=True, text=True)
        if completed.returncode != 0:
            raise VisualEvaluationError("candidate runner failed: " + (completed.stderr or completed.stdout))
        artifact = np.load(output)
        if not np.array_equal(np.asarray(artifact["obs_names"]).astype(str),
                              np.asarray(dataset.obs_names.astype(str))):
            raise VisualEvaluationError("embedding artifact obs_names mismatch")
        embedding = np.asarray(artifact["embedding"], dtype=float)
    return {
        "components": score_raw_components(embedding, gold, include_density=include_density),
        "artifact_valid": True,
        "artifact_binding": {"candidate_sha256": sha256(solution_py), "dataset_sha256": sha256(dataset_path),
                             "gold_sha256": sha256(gold_path), "n_cells": int(dataset.n_obs),
                             "n_components": int(embedding.shape[1])},
        "tusoai_normalized_score": None,
        "normalization_status": "unresolved_for_correlation_outputs",
        "cross_metric_and_dataset_aggregate": None,
        "formal_main_table_eligible": False,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution-py", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--gold-solution", type=Path, required=True)
    parser.add_argument("--runner", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--include-density", action="store_true")
    parser.add_argument("--require-formal-aggregate", action="store_true")
    args = parser.parse_args()
    if args.require_formal_aggregate:
        raise SystemExit("formal Visual score is blocked: normalization and aggregation are unresolved")
    result = evaluate_dataset(solution_py=args.solution_py, dataset_path=args.dataset,
                              gold_path=args.gold_solution, runner=args.runner,
                              seed=args.seed, include_density=args.include_density)
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
