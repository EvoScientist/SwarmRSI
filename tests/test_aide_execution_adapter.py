from __future__ import annotations

import hashlib
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from swarmrsi.baselines.aide_execution import AideExecutionUnresolved, AideNatureExecution
from swarmrsi.baselines.aide_state import atomic_json, unseal


@pytest.fixture
def executor(tmp_path, monkeypatch):
    repo = Path(__file__).resolve().parents[1]
    spec = importlib.util.spec_from_file_location("nature_bridge", repo / "scripts/evaluate_naturebench_local_workspace.py")
    bridge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bridge)
    monkeypatch.setattr(bridge.shutil, "which", lambda command: "/usr/bin/bwrap")
    monkeypatch.delenv("SWARMRSI_MODEL_CACHE", raising=False)
    task = tmp_path / "task"
    (task / "problem/data").mkdir(parents=True)
    (task / "problem/data/public.txt").write_text("public input")
    (task / "evaluation").mkdir()
    (task / "evaluation/private.txt").write_text("hidden never mounted")
    source = tmp_path / "upstream"
    (source / "aide").mkdir(parents=True)
    (source / "aide/interpreter.py").write_text("# locked source fixture\n")
    deps = tmp_path / "dependencies"
    deps.mkdir()
    value = AideNatureExecution(
        workspace=tmp_path / "workspace", artifacts=tmp_path / "nodes", task_root=task,
        runtime_python=tmp_path / "runtime/bin/python", upstream_root=source,
        dependency_root=deps, helper_path=repo / "scripts/run_aide_interpreter_once.py",
        interpreter_sha256=hashlib.sha256((source / "aide/interpreter.py").read_bytes()).hexdigest(),
        bridge=bridge, evaluator_script=repo / "scripts/evaluate_naturebench_local_workspace.py",
        timeout_seconds=5,
    )
    value.prepare()
    return value


def fake_helper(executor, *, exc_type=None, produce_output=True, after=None):
    def execute(command, **kwargs):
        assert command[0] == "/usr/bin/bwrap"
        assert "/aide-source/aide" in command and str(executor.upstream_root) not in command
        assert str(executor.task_root / "evaluation") not in command
        assert not any("API_KEY" in key for key in kwargs["env"])
        cache = executor.workspace / "working/count.txt"
        count = int(cache.read_text()) + 1 if cache.exists() else 1
        cache.write_text(str(count))
        if produce_output:
            (executor.workspace / "output/predictions.txt").write_text(str(count))
        if after:
            after()
        receipt = {
            "schema_version": "aide-original-interpreter-receipt-v1",
            "code_sha256": hashlib.sha256(kwargs["input"].encode()).hexdigest(),
            "interpreter_sha256": executor.interpreter_sha256, "reset_session": True,
            "execution_result": {"term_out": ["real captured stdout\n", "real captured stderr\n"],
                "exec_time": 0.2, "exc_type": exc_type, "exc_info": None, "exc_stack": None},
        }
        return SimpleNamespace(stdout=json.dumps(receipt), stderr="helper stderr", returncode=0)
    return execute


def test_one_execution_per_node_cache_continuity_and_immutable_snapshots(executor, monkeypatch):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor))
    first = executor.execute(node_id="one", code="print(1)", execution_only=True)
    second = executor.execute(node_id="two", code="print(2)", execution_only=True)
    assert first["execution_result"]["term_out"] == ["real captured stdout\n", "real captured stderr\n"]
    assert (executor.artifacts / "one/artifact/output/predictions.txt").read_text() == "1"
    assert (executor.artifacts / "two/artifact/output/predictions.txt").read_text() == "2"
    assert (executor.artifacts / "one/workspace/working/count.txt").read_text() == "1"
    assert (executor.workspace / "working/count.txt").read_text() == "2"
    assert not (executor.workspace / ".swarmrsi").exists()
    assert not (executor.workspace / "run.py").exists()
    assert second["evaluation"]["valid"] is True
    with pytest.raises(FileExistsError):
        executor.execute(node_id="two", code="print(2)", execution_only=True)


@pytest.mark.parametrize("exc_type,output", [("ValueError", False), (None, False)])
def test_program_failure_retains_code_logs_and_never_scores(executor, monkeypatch, exc_type, output):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run",
                        fake_helper(executor, exc_type=exc_type, produce_output=output))
    value = executor.execute(node_id="failed", code="raise ValueError('bad')")
    assert value["evaluation"]["valid"] is False
    assert value["evaluation"]["score_bearing_evaluation_attempted"] is False
    assert (executor.artifacts / "failed/program.py").read_text() == "raise ValueError('bad')"
    assert (executor.artifacts / "failed/execution_receipt.json").is_file()
    assert not (executor.artifacts / "failed/official_evaluation.json").exists()


@pytest.mark.parametrize("damage", [None, "code", "raw_snapshot", "output", "started_without_capture"])
def test_completed_scoring_reconciles_without_reexecution(executor, monkeypatch, damage):
    helper = fake_helper(executor)
    invoked = []
    def run(command, **kwargs):
        invoked.append(command[0])
        if command[0] == "/usr/bin/bwrap":
            return helper(command, **kwargs)
        target = Path(command[command.index("--result") + 1])
        atomic_json(target, {"valid": True, "execution_valid": True,
                            "score_bearing_evaluation_attempted": True,
                            "metrics": {"aggregate_improvement": 0.25}})
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", run)
    bundle = executor.execute(node_id="recover", code="print(1)")
    assert len(invoked) == 2
    if damage == "code":
        (executor.artifacts / "recover/program.py").write_text("different")
    elif damage == "raw_snapshot":
        (executor.artifacts / "recover/workspace/working/count.txt").write_text("changed")
    elif damage == "output":
        (executor.artifacts / "recover/artifact/output/predictions.txt").write_text("changed")
    elif damage == "started_without_capture":
        (executor.artifacts / "recover/scorer_capture.json").unlink()
    if damage is None:
        assert executor.reconcile(node_id="recover", code="print(1)") == bundle
    else:
        with pytest.raises((ValueError, AideExecutionUnresolved)):
            executor.reconcile(node_id="recover", code="print(1)")
    assert len(invoked) == 2


@pytest.mark.parametrize("change_metric", [False, True])
def test_scorer_completed_before_bundle_has_hash_bound_recovery(executor, monkeypatch, change_metric):
    helper = fake_helper(executor)
    invoked = []
    def run(command, **kwargs):
        invoked.append(command[0])
        if command[0] == "/usr/bin/bwrap":
            return helper(command, **kwargs)
        target = Path(command[command.index("--result") + 1])
        atomic_json(target, {"valid": True, "execution_valid": True,
                            "score_bearing_evaluation_attempted": True,
                            "metrics": {"aggregate_improvement": 0.25}})
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", run)
    original_score = executor.score
    def score_then_interrupt(**kwargs):
        original_score(**kwargs)
        raise RuntimeError("interrupted after scorer receipt but before execution bundle")
    monkeypatch.setattr(executor, "score", score_then_interrupt)
    with pytest.raises(RuntimeError, match="interrupted"):
        executor.execute(node_id="score-completed", code="print(1)")
    assert not (executor.artifacts / "score-completed/execution_bundle.json").exists()
    target = executor.artifacts / "score-completed/official_evaluation.json"
    if change_metric:
        value = json.loads(target.read_text())
        value["metrics"]["aggregate_improvement"] = 900.0
        atomic_json(target, value)
        with pytest.raises(AideExecutionUnresolved, match="matching result hash"):
            executor.reconcile(node_id="score-completed", code="print(1)")
    else:
        recovered = executor.reconcile(node_id="score-completed", code="print(1)")
        assert recovered["evaluation"]["metrics"]["aggregate_improvement"] == 0.25
    assert len(invoked) == 2


def test_failed_execution_reconciles_without_any_scorer(executor, monkeypatch):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run",
                        fake_helper(executor, exc_type="ValueError", produce_output=False))
    bundle = executor.execute(node_id="recover-failure", code="raise ValueError()")
    def forbidden(*args, **kwargs):
        raise AssertionError("completed failed execution must never run or score again")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", forbidden)
    assert executor.reconcile(node_id="recover-failure", code="raise ValueError()") == bundle


def test_incomplete_execution_cannot_be_automatically_repeated(executor, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("interrupted before reliable receipt")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fail)
    with pytest.raises(OSError):
        executor.execute(node_id="uncertain", code="print(1)")
    with pytest.raises(AideExecutionUnresolved, match="do not rerun"):
        executor.reconcile(node_id="uncertain", code="print(1)")


def test_execution_only_qualification_cannot_silently_start_scoring_on_recovery(executor, monkeypatch):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor))
    executor.execute(node_id="execution-only", code="print(1)", execution_only=True)
    def forbidden(*args, **kwargs):
        raise AssertionError("qualification recovery cannot start a scientific evaluation")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", forbidden)
    with pytest.raises(AideExecutionUnresolved, match="qualification cannot"):
        executor.reconcile(node_id="execution-only", code="print(1)")


def test_execution_only_mode_retained_before_bundle_interrupt(executor, monkeypatch):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor))
    def interrupt(phase, payload):
        if phase == "execution_complete":
            raise RuntimeError("cut after complete execution before bundle")
    with pytest.raises(RuntimeError, match="cut after"):
        executor.execute(node_id="execution-only-cut", code="print(1)",
                         execution_only=True, phase_callback=interrupt)
    assert not (executor.artifacts / "execution-only-cut/execution_bundle.json").exists()
    def forbidden(*args, **kwargs):
        raise AssertionError("recovery must not execute or score")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", forbidden)
    with pytest.raises(AideExecutionUnresolved, match="qualification cannot"):
        executor.reconcile(node_id="execution-only-cut", code="print(1)")


@pytest.mark.parametrize("reserved_kind", ["symlink", "directory", "file"])
def test_candidate_controller_path_cannot_redirect_host_write(executor, monkeypatch, tmp_path, reserved_kind):
    escaped = tmp_path / "host-target"
    escaped.mkdir()
    sentinel = escaped / "keep.txt"
    sentinel.write_text("unchanged")
    def attack():
        path = executor.workspace / ".swarmrsi"
        if reserved_kind == "symlink":
            path.symlink_to(escaped, target_is_directory=True)
        elif reserved_kind == "directory":
            path.mkdir()
            (path / "execution_validation.json.tmp").symlink_to(sentinel)
        else:
            path.write_text("candidate object")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor, after=attack))
    result = executor.execute(node_id="host_write_attack", code="print(1)", execution_only=True)
    assert result["evaluation"]["valid"] is True
    assert sentinel.read_text() == "unchanged"
    assert sorted(path.name for path in escaped.iterdir()) == ["keep.txt"]
    retained = executor.artifacts / "host_write_attack/workspace/.swarmrsi"
    assert retained.exists() or retained.is_symlink()
    assert (executor.artifacts / "host_write_attack/program.py").is_file()
    safe_receipt = executor.artifacts / "host_write_attack/artifact/.swarmrsi/execution_validation.json"
    assert not safe_receipt.parent.is_symlink() and safe_receipt.is_file()


def test_output_symlink_is_not_a_valid_execution_artifact(executor, monkeypatch, tmp_path):
    target = tmp_path / "private-host-directory"
    target.mkdir()
    (target / "private.txt").write_text("do not read")
    def attack():
        (executor.workspace / "output").rmdir()
        (executor.workspace / "output").symlink_to(target, target_is_directory=True)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run",
                        fake_helper(executor, produce_output=False, after=attack))
    value = executor.execute(node_id="bad-output", code="print(1)", execution_only=True)
    assert value["evaluation"]["valid"] is False
    manifest = (executor.artifacts / "bad-output/workspace_snapshot.json").read_text()
    assert "do not read" not in manifest


def test_evaluator_failure_is_not_returned_as_buggy_candidate(executor, monkeypatch):
    helper = fake_helper(executor)
    def run(command, **kwargs):
        if command[0] == "/usr/bin/bwrap":
            return helper(command, **kwargs)
        target = Path(command[command.index("--result") + 1])
        atomic_json(target, {"valid": False, "execution_valid": True,
                            "score_bearing_evaluation_attempted": True,
                            "public_diagnostic": {"category": "official_evaluation"}})
        return SimpleNamespace(stdout="", stderr="", returncode=0)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", run)
    with pytest.raises(AideExecutionUnresolved, match="not_candidate_bug"):
        executor.execute(node_id="grader-failure", code="print(1)")
    receipt = unseal(json.loads((executor.artifacts / "grader-failure/execution_receipt.json").read_text()))
    assert receipt["execution_result"]["exc_type"] is None
    with pytest.raises(AideExecutionUnresolved, match="already exists"):
        executor.score(node_id="grader-failure")


def test_host_failure_still_retains_generated_program(executor, monkeypatch):
    def fail(*args, **kwargs):
        raise OSError("host launch failure")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fail)
    with pytest.raises(OSError):
        executor.execute(node_id="host-failure", code="print('retained')")
    assert (executor.artifacts / "host-failure/program.py").read_text() == "print('retained')"


def test_candidate_run_py_preserved_separately_from_original_source(executor, monkeypatch):
    def generated_auxiliary():
        (executor.workspace / "run.py").write_text("# auxiliary generated file")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run",
                        fake_helper(executor, after=generated_auxiliary))
    value = executor.execute(node_id="auxiliary", code="print('original')", execution_only=True)
    assert value["evaluation"]["valid"]
    assert (executor.artifacts / "auxiliary/workspace/run.py").read_text() == "# auxiliary generated file"
    assert (executor.artifacts / "auxiliary/artifact/run.py").read_text() == "print('original')"


def test_scoring_timeout_cannot_silently_repeat_attempt(executor, monkeypatch):
    helper = fake_helper(executor)
    scorer_calls = []
    def run(command, **kwargs):
        if command[0] == "/usr/bin/bwrap":
            return helper(command, **kwargs)
        scorer_calls.append(command)
        import subprocess
        raise subprocess.TimeoutExpired(command, 35)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", run)
    with pytest.raises(AideExecutionUnresolved, match="attempt_unresolved"):
        executor.execute(node_id="timed-grader", code="print(1)")
    assert (executor.artifacts / "timed-grader/score_started.json").is_file()
    with pytest.raises(AideExecutionUnresolved, match="reconcile"):
        executor.score(node_id="timed-grader")
    assert len(scorer_calls) == 1


@pytest.mark.parametrize("case", ["nested_escape", "cycle", "extra_mapping"])
def test_public_preview_input_validation(executor, tmp_path, case):
    public = executor.task_root / "problem/data"
    if case == "nested_escape":
        nested = public / "nested"
        nested.mkdir()
        (nested / "escape.txt").symlink_to(executor.task_root / "evaluation/private.txt")
    elif case == "cycle":
        (public / "cycle").symlink_to(public, target_is_directory=True)
    else:
        (executor.workspace / "input/extra").write_text("not in frozen mapping")
    with pytest.raises(ValueError):
        executor.prepare()


@pytest.mark.parametrize("kind", ["file", "symlink"])
def test_prepare_resume_preserves_candidate_owned_failures_for_debug(executor, monkeypatch, tmp_path, kind):
    def break_output():
        output = executor.workspace / "output"
        output.rmdir()
        if kind == "file":
            output.write_text("broken output path")
        else:
            output.symlink_to(tmp_path / "absent-target")
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run",
                        fake_helper(executor, produce_output=False, after=break_output))
    failed = executor.execute(node_id="broken", code="print(1)", execution_only=True)
    assert not failed["evaluation"]["valid"]
    executor.prepare()  # Must not reject something an uninterrupted debug can fix.
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor))
    repaired = executor.execute(node_id="repaired", code="print(2)", execution_only=True)
    assert repaired["evaluation"]["valid"]
    assert (executor.artifacts / "broken/workspace/output").is_file() or kind == "symlink"


def test_resolved_artifact_path_cannot_enter_candidate_workspace(executor, tmp_path):
    (executor.workspace / "nested").mkdir()
    alias = tmp_path / "artifact-alias"
    alias.symlink_to(executor.workspace / "nested", target_is_directory=True)
    with pytest.raises(ValueError, match="outside candidate"):
        AideNatureExecution(
            workspace=executor.workspace, artifacts=alias, task_root=executor.task_root,
            runtime_python=executor.runtime_python, upstream_root=executor.upstream_root,
            dependency_root=executor.dependency_root, helper_path=executor.helper_path,
            interpreter_sha256=executor.interpreter_sha256, bridge=executor.bridge,
            evaluator_script=executor.evaluator_script,
        )
