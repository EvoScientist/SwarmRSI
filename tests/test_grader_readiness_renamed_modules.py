"""Grader readiness reads ground truth pickled under a module later renamed to its private form.

NB 00916 ships clf_py36.pkl naming sklearn.svm.classes (sklearn.svm._classes since 0.22); its official
evaluator loads it with a compat unpickler, so readiness must not call it unreadable.
"""
import sys
from pathlib import Path

from swarmrsi.naturebench_evaluator import grader_readiness


def write_task(tmp_path: Path, pickled: bytes) -> Path:
    pkg = tmp_path / "lib/fakepkg/sub"
    pkg.mkdir(parents=True)
    (tmp_path / "lib/fakepkg/__init__.py").write_text("")
    (pkg / "__init__.py").write_text("")
    (pkg / "_classes.py").write_text("class Model:\n    pass\n")
    task = tmp_path / "task"
    (task / "evaluation/ground_truth").mkdir(parents=True)
    (task / "evaluation/evaluator.py").write_text("import json\nprint(json.dumps({}))\n")
    (task / "evaluation/ground_truth/model.pkl").write_bytes(pickled)
    return task


def readiness(tmp_path, pickled, monkeypatch):
    task = write_task(tmp_path, pickled)
    monkeypatch.setenv("PYTHONPATH", str(tmp_path / "lib"))
    return grader_readiness(task, Path(sys.executable))


def test_pickle_naming_the_renamed_public_module_is_readable(tmp_path, monkeypatch):
    # protocol-0 pickle of fakepkg.sub.classes.Model(), a module that only exists as fakepkg.sub._classes
    pickled = b"cfakepkg.sub.classes\nModel\np0\n(tRp1\n."
    result = readiness(tmp_path, pickled, monkeypatch)
    assert result["unreadable_ground_truth"] == [] and result["ready"], result


def test_a_really_missing_module_is_still_unreadable(tmp_path, monkeypatch):
    pickled = b"cfakepkg.sub.absent\nModel\np0\n(tRp1\n."
    result = readiness(tmp_path, pickled, monkeypatch)
    assert result["unreadable_ground_truth"] and not result["ready"], result
