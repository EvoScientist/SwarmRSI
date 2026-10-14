"""Regression cases migrated from the P12 independent audit; synthetic only."""
import errno
import hashlib
import json
import os
from pathlib import Path

import pytest

from swarmrsi import learning_analysis as analysis
from swarmrsi.retained_results import read_json
from swarmrsi.tree_policy import TreePolicy
from swarmrsi.validation_failure import unavailable_record, ValidationUnavailable
from test_swarmrsi_learning import setup_chain, Proposer
from test_swarmrsi_retained_results import put
from test_installed_learning_analysis import prepare
from test_naturebench_entrypoint import inputs  # noqa: F401


def halted_chain(inputs, tmp_path):
    runner, trial, _ = prepare(inputs, "file")
    chain, pool = setup_chain(tmp_path / "chain", cells=[{"task_id": "fixture", "seed": 0}])
    put(trial / "exit.json", {"returncode": 17})
    failure = unavailable_record(trial)

    class Unavailable:
        def evaluate(self, **kwargs):
            raise ValidationUnavailable(failure)

    chain.run_round(Proposer([TreePolicy(parent_exploration_weight=1.)]), Unavailable(), pool)
    return chain, runner, trial


@pytest.mark.parametrize("damage", ["model", "policy", "missing"])
def test_chain_only_unavailable_verifies_original_executed_profile(inputs, tmp_path, damage):
    chain, _, trial = halted_chain(inputs, tmp_path)
    original = analysis.analyze_learning(chain_root=chain.root)
    assert original["policy_trials"][0]["status"] == "validation_unavailable"
    path = trial / "profile.json"
    if damage == "missing":
        path.rename(path.with_name("profile.original.json"))
    else:
        profile = read_json(path)
        profile["model" if damage == "model" else "rsi_policy"] = (
            "different/model" if damage == "model" else TreePolicy(parent_exploration_weight=2.).to_dict())
        put(path, profile)
    # Supplying the runner is not required by the public CLI. Chain-only
    # unavailable evidence must apply the same original-profile checks.
    with pytest.raises((ValueError, FileNotFoundError)):
        analysis.analyze_learning(chain_root=chain.root)


def test_export_cannot_write_into_original_starter_outside_installed_package(inputs, tmp_path):
    runner, _, _ = prepare(inputs, "file")
    chain, _ = setup_chain(tmp_path / "chain")
    starter = Path(runner.profile["tasks"]["fixture"]["starter"])
    assert not starter.is_relative_to(runner.source_root)
    destination = starter / "new-analysis"
    with pytest.raises(ValueError):
        analysis.export_learning_analysis(destination, chain_root=chain.root, runner_roots=[runner.root])
    assert not destination.exists()


def test_input_changes_during_output_staging_do_not_publish_stale_refs(tmp_path, monkeypatch):
    chain, _ = setup_chain(tmp_path / "chain")
    destination = tmp_path / "export"
    original_write = Path.write_bytes
    changed = []

    def during_staging(path, data):
        result = original_write(path, data)
        if not changed and any(p.startswith(".learning-analysis-") for p in path.parts):
            original = chain.root / "chain.json"
            original.write_text(original.read_text() + "\n")
            changed.append(True)
        return result

    monkeypatch.setattr(Path, "write_bytes", during_staging)
    with pytest.raises(ValueError):
        analysis.export_learning_analysis(destination, chain_root=chain.root)
    assert changed and not destination.exists()


def test_empty_destination_appearing_at_commit_is_not_replaced(tmp_path, monkeypatch):
    chain, _ = setup_chain(tmp_path / "chain")
    destination = tmp_path / "export"
    original_rename = analysis._publish_no_replace
    appeared = []

    def competing_writer(source, target, *args, **kwargs):
        if Path(target) == destination:
            destination.mkdir(mode=0o700)
            appeared.append(destination.stat().st_ino)
        return original_rename(source, target, *args, **kwargs)

    monkeypatch.setattr(analysis, "_publish_no_replace", competing_writer)
    with pytest.raises((ValueError, FileExistsError)):
        analysis.export_learning_analysis(destination, chain_root=chain.root)
    assert appeared and destination.stat().st_ino == appeared[0]
    assert not list(destination.iterdir())


@pytest.mark.parametrize("existing", [False, True])
def test_native_no_replace_directory_publication(tmp_path, existing):
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir(); (staging / "data").write_text("snapshot")
    if existing:
        target.mkdir(); inode = target.stat().st_ino
        with pytest.raises(FileExistsError):
            analysis._publish_no_replace(staging, target)
        assert target.stat().st_ino == inode and not list(target.iterdir())
        assert (staging / "data").read_text() == "snapshot"
    else:
        analysis._publish_no_replace(staging, target)
        assert (target / "data").read_text() == "snapshot" and not staging.exists()


@pytest.mark.parametrize("existing", [False, True])
def test_publication_claims_the_name_when_no_replace_flags_are_unsupported(tmp_path, monkeypatch, existing):
    """GPFS and similar shared filesystems reject RENAME_NOREPLACE with EINVAL."""
    monkeypatch.setattr(analysis, "_native_no_replace", lambda source, destination: False)
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir()
    (staging / "policy_trials.jsonl").write_text("{}\n")
    (staging / "manifest.json").write_text("{}\n")
    if existing:
        target.mkdir(); inode = target.stat().st_ino
        with pytest.raises(FileExistsError):
            analysis._publish_no_replace(staging, target)
        assert target.stat().st_ino == inode and not list(target.iterdir())
        assert sorted(p.name for p in staging.iterdir()) == ["manifest.json", "policy_trials.jsonl"]
    else:
        assert analysis._publish_no_replace(staging, target) == "claimed_directory"
        assert (target / "policy_trials.jsonl").read_text() == "{}\n"
        # Same postcondition as the native rename: the staged directory is gone.
        assert (target / "manifest.json").read_text() == "{}\n" and not staging.exists()


def test_interrupted_claimed_publication_leaves_no_manifest(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "_native_no_replace", lambda source, destination: False)
    original = os.replace

    def fail_on_manifest(source, destination, *args, **kwargs):
        if Path(destination).name == analysis.MANIFEST_NAME:
            raise OSError(errno.EIO, "interrupted publication")
        return original(source, destination, *args, **kwargs)

    monkeypatch.setattr(analysis.os, "replace", fail_on_manifest)
    staging, target = tmp_path / "staging", tmp_path / "target"
    staging.mkdir()
    (staging / "policy_trials.jsonl").write_text("{}\n")
    (staging / "manifest.json").write_text("{}\n")
    with pytest.raises(OSError):
        analysis._publish_no_replace(staging, target)
    # The claimed name is never a finished analysis without its sealed manifest.
    assert target.is_dir() and [p.name for p in target.iterdir()] == ["policy_trials.jsonl"]


def test_unsupported_filesystem_export_is_complete_and_still_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(analysis, "_native_no_replace", lambda source, destination: False)
    chain, _ = setup_chain(tmp_path / "chain")
    destination = tmp_path / "export"
    manifest = analysis.export_learning_analysis(destination, chain_root=chain.root)
    names = sorted(p.name for p in destination.iterdir())
    assert names == sorted([*manifest["outputs"], "manifest.json"])
    for name, info in manifest["outputs"].items():
        assert hashlib.sha256((destination / name).read_bytes()).hexdigest() == info["sha256"]
    assert json.loads((destination / "manifest.json").read_text()) == manifest
    with pytest.raises(ValueError, match="output exists"):
        analysis.export_learning_analysis(destination, chain_root=chain.root)


@pytest.mark.parametrize("error", [errno.EEXIST, errno.EACCES])
def test_real_publication_errors_are_not_treated_as_unsupported(tmp_path, monkeypatch, error):
    calls = []
    monkeypatch.setattr(analysis.os, "mkdir", lambda *a, **k: calls.append(a))

    def failing(*args, **kwargs):
        raise OSError(error, os.strerror(error), "target")

    monkeypatch.setattr(analysis, "_native_no_replace", failing)
    with pytest.raises(OSError):
        analysis._publish_no_replace(tmp_path, tmp_path / "target")
    assert not calls
