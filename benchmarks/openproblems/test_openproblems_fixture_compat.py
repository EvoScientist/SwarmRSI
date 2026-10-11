"""OpenProblems denoising task package, metric and adapter.

Needs anndata/scipy (the `openproblems` extra); the whole module skips without
them rather than asserting anything about a benchmark it cannot read.
"""
import json
from pathlib import Path

import pytest

pytest.importorskip("anndata", reason="OpenProblems assets need the 'openproblems' extra")
pytest.importorskip("scipy", reason="OpenProblems assets need the 'openproblems' extra")

import numpy as np
import anndata as ad
from scipy.sparse import csr_matrix

from evolvcoder.api import Application, TaskSpec
from evolvcoder.configuration import config_from_dict
from evolvcoder.openproblems_run import OpenProblemsAdapter
from evolvcoder.openproblems_tasks import (METRIC, TASK, canonical, mse, normalize_cp10k_log1p,
                                           prepare, read_prediction, split_molecules, verify_assets)


def official_assets(tmp_path, *, cells=12, genes=9, seed=3):
    """A small released train/test pair with the same shape contract as the real one."""
    random = np.random.RandomState(seed)
    counts = csr_matrix(random.poisson(2.0, size=(cells, genes)).astype(int))
    source = tmp_path / "official"
    source.mkdir()
    for name in ("train.h5ad", "test.h5ad"):
        ad.AnnData(shape=counts.shape, layers={"counts": counts.copy()}, uns={"dataset_id": "fixture", "dataset_organism": "homo_sapiens"}
                   ).write_h5ad(source / name)
    sources = {}
    for name in ("train.h5ad", "test.h5ad"):
        raw = (source / name).read_bytes()
        import hashlib
        sources[name] = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
    (source / "dataset_manifest.json").write_text(json.dumps({"sources": sources, "protocol": "fixture"}))
    return source


def test_metric_matches_the_official_normalize_and_log1p_formula():
    values = np.array([[1.0, 0.0, 3.0], [2.0, 2.0, 0.0]])
    normalized = normalize_cp10k_log1p(values)
    expected = np.log1p(values / values.sum(axis=1, keepdims=True) * 10_000.0)
    assert np.allclose(normalized, expected)
    # A cell with no counts stays zero rather than becoming nan.
    assert np.allclose(normalize_cp10k_log1p(np.zeros((1, 3))), 0.0)
    assert mse(values, values) == 0.0 and mse(csr_matrix(values), values) == 0.0
    # normalize_total makes the metric invariant to a per-cell scale, so a
    # uniformly rescaled prediction is the same prediction to this metric.
    assert mse(values, values * 2) == 0.0
    assert mse(values, values[::-1]) > 0.0
    with pytest.raises(ValueError, match="does not match"):
        mse(values, values[:, :2])


def test_official_molecule_split_is_conservative_and_reproducible():
    counts = np.random.RandomState(0).poisson(3.0, size=200).astype(int)
    left, right = split_molecules(counts, 0.9, 0.0, np.random.RandomState(7))
    again, _ = split_molecules(counts, 0.9, 0.0, np.random.RandomState(7))
    assert np.array_equal(left + right, counts) and np.array_equal(left, again)
    assert left.sum() > right.sum() > 0


def test_prepared_package_hides_the_official_test_and_seals_every_file(tmp_path):
    source = official_assets(tmp_path)
    record = prepare(source=source, destination=tmp_path / "task", seed=0, train_frac=0.9)
    root = tmp_path / "task"
    assert record["task_id"] == TASK and record["metric"] == METRIC
    assert verify_assets(root / "custody.json") == record
    # The candidate's data directory carries the public split only.
    assert sorted(p.name for p in (root / "problem/data").iterdir()) == ["train.h5ad"]
    public = ad.read_h5ad(root / "problem/data/train.h5ad").layers["counts"]
    held = ad.read_h5ad(root / "evaluation/public_validation.h5ad").layers["counts"]
    original = ad.read_h5ad(source / "train.h5ad").layers["counts"]
    assert public.shape == held.shape == original.shape
    assert np.array_equal((public + held).toarray(), original.toarray())
    assert public.sum() > held.sum() > 0
    for name in ("evaluation/public_validation.h5ad", "private/test.h5ad"):
        assert name in record["controller_only"] and name in record["files"]
    metadata = json.loads((root / "metadata.json").read_text())
    assert metadata["search_metric"] == METRIC and metadata["official_test_visible_to_candidate"] is False
    with pytest.raises(FileExistsError):
        prepare(source=source, destination=root)


def test_changed_official_asset_or_custody_is_refused(tmp_path):
    source = official_assets(tmp_path)
    prepare(source=source, destination=tmp_path / "task")
    root = tmp_path / "task"
    held = root / "evaluation/public_validation.h5ad"
    held.write_bytes(held.read_bytes() + b"\0")
    with pytest.raises(ValueError, match="asset changed"):
        verify_assets(root / "custody.json")
    record = json.loads((root / "custody.json").read_text())
    record["files"]["evaluation/public_validation.h5ad"]["sha256"] = "0" * 64
    (root / "custody.json").write_text(json.dumps(record))
    with pytest.raises(ValueError, match="seal mismatch"):
        verify_assets(root / "custody.json")
    (source / "train.h5ad").write_bytes((source / "train.h5ad").read_bytes() + b"\0")
    with pytest.raises(ValueError, match="manifest hash"):
        prepare(source=source, destination=tmp_path / "again")


def test_prediction_is_read_as_sparse_or_dense(tmp_path):
    from scipy.sparse import save_npz
    values = np.array([[1.0, 0.0], [0.0, 2.0]])
    output = tmp_path / "out"
    output.mkdir()
    with pytest.raises(ValueError, match="no denoised"):
        read_prediction(output)
    np.save(output / "denoised.npy", values)
    assert np.allclose(np.asarray(read_prediction(output)), values)
    save_npz(output / "denoised.npz", csr_matrix(values))
    assert np.allclose(read_prediction(output).toarray(), values)


def task_config(tmp_path, root, **overrides):
    profile = {"model": "provider/model", "tasks": {TASK: {
        "starter": "starter", "search_metric": METRIC, "search_target": None,
        "limits": {"backbone_message_ids": 10, "tool_calls": 10, "score_bearing_evaluations": 3,
                   "wall_time_seconds": 600, "tokens": {"input": 1000, "output": 1000}}}}}
    (root / "profile.json").write_text(json.dumps(profile))
    (root / "assets.json").write_text(json.dumps({
        "schema_version": "evolvcoder-openproblems-assets-v1", "profile": "profile.json",
        "tasks_root": str(root), "custody": "custody.json", "trial_timeout_seconds": 600}))
    data = {"benchmark": "openproblems", "task": TASK, "method": "evolvcoder",
            "model": {"model": "provider/model"}, "benchmark_config": str(root / "assets.json"),
            "output_root": str(tmp_path / "runs")}
    data.update(overrides)
    return config_from_dict(data, base=root)


def test_adapter_binds_custody_and_keeps_the_official_test_out_of_reach(tmp_path):
    source = official_assets(tmp_path)
    record = prepare(source=source, destination=tmp_path / "task")
    root = tmp_path / "task"
    config = task_config(tmp_path, root)
    task = Application().benchmarks.get("openproblems").task(TASK, config)
    assert isinstance(task, TaskSpec) and task.benchmark == "openproblems" and task.task_id == TASK
    assert task.metric_name == METRIC["name"] and task.metric_direction == "minimize"
    assert task.evaluator_id == "openproblems:" + record["custody_sha256"]
    assert task.public_data == root / "problem/data"
    described = " ".join(task.public_description.split())
    assert "official held-out test counts are not available here" in described
    # Neither the held-out split nor the official test is inside the public data.
    for private in (root / "evaluation", root / "private"):
        assert not private.is_relative_to(task.public_data)


@pytest.mark.parametrize("break_it", ["wisdom", "rsi", "threshold", "task_id"])
def test_unqualified_openproblems_configurations_are_refused(tmp_path, break_it):
    source = official_assets(tmp_path)
    prepare(source=source, destination=tmp_path / "task")
    root = tmp_path / "task"
    overrides, task_id = {}, TASK
    if break_it == "wisdom":
        overrides["learning"] = {"wisdom_enabled": True, "wisdom_snapshot": str(root / "metadata.json")}
    elif break_it == "rsi":
        overrides["learning"] = {"rsi_enabled": True, "feedback_store": str(tmp_path / "pool")}
    elif break_it == "task_id":
        task_id = "another-task"
    config = task_config(tmp_path, root, **overrides)
    if break_it == "threshold":
        profile = json.loads((root / "profile.json").read_text())
        profile["tasks"][TASK]["claude_code_score"] = 0.5
        (root / "profile.json").write_text(json.dumps(profile))
    with pytest.raises(ValueError):
        OpenProblemsAdapter().task(task_id, config)
