"""The candidate's own run output reaches the next planning and coding step.

Three reruns of s42256-025-01002-0 stalled at -0.008/-0.013/-0.008: the planner saw
only final scores. MLEvolve hands every improve/debug step the parent's stdout and
stderr (first and last 2,500 characters); on the same backbone it keeps improving.
"""
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolvcoder import naturebench_evaluator as bridge
from evolvcoder import naturebench_session as session
from test_grader_environment import _task


def _run(tmp_path, task, workspace, *extra):
    result = tmp_path / ("result%d.json" % len(list(tmp_path.glob("result*.json"))))
    subprocess.run([sys.executable, "-m", "evolvcoder.naturebench_evaluator", "--workspace", str(workspace),
                    "--result", str(result), "--task-root", str(task), "--runtime-python", sys.executable,
                    "--timeout-seconds", "120", *extra], capture_output=True, text=True)
    return json.loads(result.read_text())


def _chatty(workspace, lines):
    (workspace / "run.py").write_text(
        "import os, pathlib, sys\n"
        f"for i in range({lines}):\n"
        "    print(f'epoch {i} val_auc 0.{i:04d}')\n"
        "print('OpenBLAS warning: precompiled NUM_THREADS exceeded', file=sys.stderr)\n"
        "print('train 12.5s predict 0.4s', file=sys.stderr)\n"
        "out = pathlib.Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True)\n"
        "(out / 'p.csv').write_text('a\\n1\\n')\n")


def test_execution_only_keeps_the_trimmed_run_log_and_scoring_carries_it(tmp_path):
    task, workspace = _task(tmp_path, "import json\njson.dump({'inst': {'AUC': 0.9}}, open('score.json', 'w'))\n")
    _chatty(workspace, 600)
    execution = _run(tmp_path, task, workspace, "--execution-only")
    log = execution["details"]["candidate_execution_log"]
    assert log.startswith("epoch 0 val_auc") and "epoch 599 val_auc" in log
    assert "characters omitted" in log and "OpenBLAS" not in log and "train 12.5s" in log
    assert len(log) < bridge.EXECUTION_LOG_HEAD + bridge.EXECUTION_LOG_TAIL + 200
    (workspace / ".evolvcoder").mkdir(exist_ok=True)
    (workspace / ".evolvcoder/execution_validation.json").write_text(json.dumps(execution))
    scored = _run(tmp_path, task, workspace, "--score-existing-output")
    assert scored["valid"] is True
    assert scored["details"]["candidate_execution_log"] == log
    assert session.execution_log_excerpt(workspace, head=100, tail=100).startswith("epoch 0")


def test_short_logs_are_kept_whole_and_silence_is_none():
    assert bridge._trim_log("epoch 1 val 0.5", "") == "epoch 1 val 0.5"
    assert bridge._trim_log("", "OpenBLAS warning: precompiled NUM_THREADS exceeded") is None


def test_excerpt_is_none_without_evidence(tmp_path):
    assert session.execution_log_excerpt(tmp_path, head=10, tail=10) is None


def test_task_context_asks_for_run_diagnostics_and_can_be_switched_off(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "runtime_dependency_inventory", lambda: ((), ()))
    monkeypatch.setattr(session, "accelerator_inventory", lambda: {"cuda": False, "devices": 0})
    task = tmp_path / "task"; (task / "problem/data").mkdir(parents=True)
    (task / "problem/README.md").write_text("# task\n")
    on = session.public_task_context(task)
    off = session.public_task_context(task, run_diagnostics=False)
    assert "## Run diagnostics" in on and "## Run diagnostics" not in off


def test_compact_context_keeps_the_readme_in_reach_and_write_first_drops_probe_advice(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "runtime_dependency_inventory", lambda: ((), ()))
    monkeypatch.setattr(session, "accelerator_inventory", lambda: {"cuda": False, "devices": 0})
    monkeypatch.setattr(session, "public_data_preview", lambda *a, **k: None)
    task = tmp_path / "task"; data = task / "problem/data"; data.mkdir(parents=True)
    (task / "problem/README.md").write_text("# task\nOBJECTIVE-MARKER\n")
    for i in range(200):
        (data / f"cell_{i:03d}.csv").write_text("gene_" + ",gene_".join(str(j) for j in range(60)) + "\n" + "1," * 59 + "1\n")
    full = session.public_task_context(task)
    small = session.public_task_context(task, compact=True, write_first=True)
    assert len(small) < len(full) * 0.6
    contract = small.split("## Controller-measured public data scale", 1)[1].split("## README", 1)[0]
    assert "sampled_max_text_width_by_column" not in contract and '"column_count": 60' in contract
    assert "## Public data inventory" in small  # 200 files > 40 listed, so the full list stays
    assert "OBJECTIVE-MARKER" in small
    assert "infer schemas before implementing" in full and "infer schemas before implementing" not in small
    assert "only to diagnose a failure" in small


def test_shift_aware_validation_section_is_opt_in_and_generic(tmp_path, monkeypatch):
    monkeypatch.setattr(session, "runtime_dependency_inventory", lambda: ((), ()))
    monkeypatch.setattr(session, "accelerator_inventory", lambda: {"cuda": False, "devices": 0})
    monkeypatch.setattr(session, "public_data_preview", lambda *a, **k: None)
    task = tmp_path / "task"; (task / "problem/data").mkdir(parents=True)
    (task / "problem/README.md").write_text("# task\n")
    off = session.public_task_context(task)
    on = session.public_task_context(task, shift_aware_validation=True)
    assert "## Validation that resembles the test" not in off
    section = on.split("## Validation that resembles the test", 1)[1]
    assert "unseen species" in section and "never read evaluation files" in section
