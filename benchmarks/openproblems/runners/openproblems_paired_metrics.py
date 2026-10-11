"""Frozen lightweight metrics for the paired OpenProblems reconstruction."""

from __future__ import annotations

from typing import Any, Iterable

import numpy as np
from scipy import sparse
from scipy.sparse.csgraph import connected_components
from sklearn.metrics import silhouette_samples, silhouette_score
from sklearn.neighbors import NearestNeighbors


class PairedMetricError(ValueError):
    pass


def finite_unit(value: Any, *, name: str) -> float:
    number = float(value)
    if not np.isfinite(number) or number < -1e-9 or number > 1.0 + 1e-9:
        raise PairedMetricError(f"{name} must be finite and in [0, 1]")
    return float(np.clip(number, 0.0, 1.0))


def equal_mean(values: Iterable[float], *, name: str) -> float:
    array = np.asarray(list(values), dtype=float)
    if array.ndim != 1 or len(array) == 0 or not np.isfinite(array).all():
        raise PairedMetricError(f"{name} requires non-empty finite values")
    return float(array.mean())


def denoise_control_normalized(*, method_mse: float, no_denoise_mse: float) -> float:
    method = float(method_mse)
    control = float(no_denoise_mse)
    if not np.isfinite(method) or not np.isfinite(control) or control <= 0:
        raise PairedMetricError("denoise MSE values are invalid")
    return float(np.clip((control - method) / control, 0.0, 1.0))


def label_dataset_score(components: dict[str, float]) -> float:
    keys = ("accuracy", "f1_macro", "f1_micro", "f1_weighted")
    if set(components) != set(keys):
        raise PairedMetricError("label component set drift")
    return equal_mean((finite_unit(components[key], name=key) for key in keys), name="label")


def _knn_connectivities(embedding: np.ndarray, *, neighbors: int = 15):
    values = np.asarray(embedding, dtype=float)
    if values.ndim != 2 or len(values) < 2 or not np.isfinite(values).all():
        raise PairedMetricError("batch embedding is invalid")
    neighbors_to_keep = min(neighbors, len(values) - 1)
    count = neighbors_to_keep + 1
    fitted = NearestNeighbors(n_neighbors=count, metric="euclidean", n_jobs=1).fit(values)
    indices = fitted.kneighbors(values, return_distance=False)
    rows = np.repeat(np.arange(len(values)), neighbors_to_keep)
    columns = indices[:, 1:].reshape(-1)
    graph = sparse.csr_matrix((np.ones(len(rows)), (rows, columns)), shape=(len(values), len(values)))
    return graph.maximum(graph.T)


def graph_connectivity_score(graph, labels: Iterable[str]) -> float:
    values = np.asarray(list(labels)).astype(str)
    if graph.shape != (len(values), len(values)):
        raise PairedMetricError("graph and label dimensions differ")
    scores: list[float] = []
    for label in sorted(set(values.tolist())):
        selected = np.flatnonzero(values == label)
        if len(selected) == 0:
            continue
        subgraph = graph[selected, :][:, selected]
        _, component = connected_components(subgraph, directed=False)
        largest = int(np.bincount(component).max())
        scores.append(largest / len(selected))
    return finite_unit(equal_mean(scores, name="graph connectivity"), name="graph connectivity")


def batch_components(embedding: Any, cell_type: Iterable[str], batch: Iterable[str], graph=None) -> dict[str, float]:
    values = np.asarray(embedding, dtype=float)
    cell_types = np.asarray(list(cell_type)).astype(str)
    batches = np.asarray(list(batch)).astype(str)
    if values.ndim != 2 or len(values) != len(cell_types) or len(values) != len(batches):
        raise PairedMetricError("batch inputs are not aligned")
    if len(set(cell_types.tolist())) < 2:
        raise PairedMetricError("ASW label requires at least two cell types")
    label_asw = (float(silhouette_score(values, cell_types, metric="euclidean")) + 1.0) / 2.0
    per_type: list[float] = []
    for label in sorted(set(cell_types.tolist())):
        selected = np.flatnonzero(cell_types == label)
        local_batches = batches[selected]
        unique = set(local_batches.tolist())
        if len(unique) < 2 or len(selected) <= len(unique):
            continue
        samples = silhouette_samples(values[selected], local_batches, metric="euclidean")
        per_type.append(float(np.mean(1.0 - np.abs(samples))))
    if not per_type:
        raise PairedMetricError("ASW batch has no cell type spanning multiple batches")
    connectivities = _knn_connectivities(values) if graph is None else graph
    return {
        "graph_connectivity": graph_connectivity_score(connectivities, cell_types),
        "asw_label": finite_unit(label_asw, name="asw_label"),
        "asw_batch": finite_unit(equal_mean(per_type, name="asw_batch"), name="asw_batch"),
    }


def batch_dataset_score(components: dict[str, float]) -> float:
    keys = ("graph_connectivity", "asw_label", "asw_batch")
    if set(components) != set(keys):
        raise PairedMetricError("batch component set drift")
    return equal_mean((finite_unit(components[key], name=key) for key in keys), name="batch")


def decomp_control_normalized(*, method_r2: float, random_control_r2: float) -> float:
    method = float(method_r2)
    control = float(random_control_r2)
    if not np.isfinite(method) or not np.isfinite(control) or control >= 1.0:
        raise PairedMetricError("decomposition R2 values are invalid")
    return float(np.clip((method - control) / (1.0 - control), 0.0, 1.0))


def visual_dataset_score(components: dict[str, float]) -> float:
    required = {
        "trustworthiness", "waypoint_distance_correlation",
        "centroid_distance_correlation", "label_distance_correlation",
        "density_preservation",
    }
    if set(components) != required:
        raise PairedMetricError("visual component set drift")
    trust = finite_unit(components["trustworthiness"], name="trustworthiness")
    distance = equal_mean(
        ((float(components[key]) + 1.0) / 2.0 for key in (
            "waypoint_distance_correlation", "centroid_distance_correlation", "label_distance_correlation"
        )),
        name="visual distance correlations",
    )
    density = (float(components["density_preservation"]) + 1.0) / 2.0
    return equal_mean((trust, finite_unit(distance, name="distance"), finite_unit(density, name="density")), name="visual")
