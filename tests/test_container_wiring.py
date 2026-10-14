"""Container backend wiring across the session, the evaluator and root evaluation.

No Docker daemon is used. candidate_execution.execute is replaced by a stand-in
that installs a genuine controller-owned action record for every call, so every
receipt verification here runs the production code path.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import pytest

from swarmrsi import candidate_execution as backend
from swarmrsi import naturebench_evaluator as evaluator
from swarmrsi import naturebench_session as session
from test_candidate_execution import PYTHON, completed_action, make_spec


@pytest.fixture
def bound(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    state = root / "run/candidate_runtime"; state.mkdir(mode=0o700, parents=True)
    spec = make_spec()
    binding = {"kind": "docker", "spec": asdict(spec), "state_root": str(state)}
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", binding)
    monkeypatch.setattr(session, "RUNTIME", Path(sys.executable))
    return SimpleNamespace(root=root, state=state, spec=spec, binding=binding)


def fake_execute(monkeypatch, *, exit_code=0, timed_out=False, status=None, on_run=None):
    """Stand in for the Docker run: a real, verifiable record is installed per call."""
    calls = []
    def execute(binding, *, workspace, public_data, argv, timeout, cwd="/workspace"):
        spec, state = backend.validate_binding(binding)
        workspace, public_data = Path(workspace).resolve(), Path(public_data).resolve()
        calls.append({"workspace": workspace, "public_data": public_data, "argv": tuple(argv), "timeout": timeout, "cwd": cwd})
        if on_run is not None:
            on_run(workspace)
        if status is not None:
            return {"status": status, "harness_error": "synthetic backend fault", "record_dir": str(state / ("x" * 32)),
                    "cleanup_confirmed": True, "returncode": None, "stdout": "", "stderr": ""}
        _, receipt, _ = completed_action(state.parent, monkeypatch, exit_code=exit_code, timed_out=timed_out,
                                         argv=argv, state=state, workspace=workspace, public=public_data)
        return receipt
    monkeypatch.setattr(backend, "execute", execute)
    return calls


def task_package(root):
    task = root / "tasks/fixture"
    (task / "problem/data").mkdir(parents=True); (task / "evaluation").mkdir()
    marker = task / "SCIENTIFIC_EVALUATOR_WAS_CALLED"
    (task / "evaluation/evaluator.py").write_text(
        "import json\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called')\n"
        "Path('score.json').write_text(json.dumps({'toy': {'accuracy': 1.0}}))\n")
    (task / "metadata.json").write_text(json.dumps({"performance_entries": [{"dataset_name": "toy", "metrics": [{
        "is_primary": True, "name": "accuracy", "metric_direction": "higher_is_better", "sota_score": [{"value": 0.5}]}]}]}))
    workspace = root / "candidate"; workspace.mkdir()
    (workspace / "run.py").write_text("raise SystemExit('must run inside the bound backend, never on the host')\n")
    return task, workspace, marker


def candidate_output(workspace):
    (workspace / "output").mkdir(exist_ok=True)
    (workspace / "output/answer.txt").write_text("ok")


def run_evaluator(monkeypatch, *args):
    monkeypatch.setattr(sys, "argv", ["evaluator", *map(str, args)])
    return evaluator.main()


def execution_only(monkeypatch, bound, task, workspace, result):
    return run_evaluator(monkeypatch, "--workspace", workspace, "--result", result, "--task-root", task,
                         "--runtime-python", sys.executable, "--execution-only", "--require-candidate-sandbox",
                         *backend.evaluate_arguments(bound.binding))


def test_evaluator_executes_in_bound_backend_and_reuses_receipt_by_content(bound, monkeypatch):
    task, workspace, marker = task_package(bound.root)
    calls = fake_execute(monkeypatch, on_run=candidate_output)
    result = workspace / ".swarmrsi/execution_validation.json"
    assert execution_only(monkeypatch, bound, task, workspace, result) == 0
    payload = json.loads(result.read_text())
    assert payload["valid"] is True and payload["execution_valid"] is True and not marker.exists()
    assert calls == [{"workspace": workspace, "public_data": task / "problem/data", "argv": (PYTHON, "run.py"),
                      "timeout": 1800, "cwd": "/workspace"}]
    details = payload["details"]
    assert details["candidate_sandboxed"] is True and details["candidate_runtime_binding"] == bound.binding
    assert details["candidate_runtime_receipt"]["status"] == "candidate_exited"
    assert Path(details["candidate_runtime_receipt"]["record_dir"]).parent == bound.state
    # The retained artifact is a copy of the temporary workspace: the receipt is
    # bound by command, public data, run.py hash and output attestation, not by path.
    artifact = bound.root / "cells/001/artifact"; shutil.copytree(workspace, artifact)
    scored = bound.root / "scientific.json"
    assert run_evaluator(monkeypatch, "--workspace", artifact, "--result", scored, "--task-root", task,
                         "--runtime-python", sys.executable, "--score-existing-output", "--require-candidate-sandbox",
                         *backend.evaluate_arguments(bound.binding)) == 0
    scientific = json.loads(scored.read_text())
    assert scientific["valid"] is True and scientific["metrics"]["aggregate_improvement"] == 1.0
    assert scientific["details"]["candidate_execution_reused"] is True and marker.exists()
    assert scientific["details"]["candidate_runtime_receipt"] == details["candidate_runtime_receipt"]
    assert scientific["details"]["candidate_runtime_binding"] == bound.binding and len(calls) == 1


@pytest.mark.parametrize("fault", ["receipt", "backend", "public_data", "argv", "no_binding"])
def test_reuse_rejects_foreign_receipts_and_backends(bound, monkeypatch, fault):
    task, workspace, marker = task_package(bound.root)
    fake_execute(monkeypatch, on_run=candidate_output)
    result = workspace / ".swarmrsi/execution_validation.json"
    assert execution_only(monkeypatch, bound, task, workspace, result) == 0
    binding, evidence = bound.binding, json.loads(result.read_text())
    if fault == "receipt":
        evidence["details"]["candidate_runtime_receipt"]["returncode"] = 1
    elif fault == "backend":
        binding = {**binding, "spec": {**binding["spec"], "cpus": 3}}
    elif fault == "public_data":
        other = bound.root / "other_public"; other.mkdir()
        _, receipt, _ = completed_action(bound.root, monkeypatch, state=bound.state, workspace=workspace, public=other)
        evidence["details"]["candidate_runtime_receipt"] = receipt
    elif fault == "argv":
        _, receipt, _ = completed_action(bound.root, monkeypatch, argv=[PYTHON, "-c", "pass"], state=bound.state,
                                         workspace=workspace, public=task / "problem/data")
        evidence["details"]["candidate_runtime_receipt"] = receipt
    else:
        binding = None
    result.write_text(json.dumps(evidence))
    scored = bound.root / "scientific.json"
    assert run_evaluator(monkeypatch, "--workspace", workspace, "--result", scored, "--task-root", task,
                         "--runtime-python", sys.executable, "--score-existing-output", "--require-candidate-sandbox",
                         *backend.evaluate_arguments(binding)) == 0
    scientific = json.loads(scored.read_text())
    assert scientific["valid"] is False and not marker.exists()
    assert scientific["public_diagnostic"]["category"] == "execution_evidence"


@pytest.mark.parametrize("status", ["infrastructure_error", "cancelled"])
def test_backend_faults_are_bridge_failures_not_candidate_faults(bound, monkeypatch, status):
    task, workspace, marker = task_package(bound.root)
    fake_execute(monkeypatch, status=status)
    result = bound.root / "execution.json"
    assert execution_only(monkeypatch, bound, task, workspace, result) == 0
    payload = json.loads(result.read_text())
    assert payload["valid"] is False and payload["execution_valid"] is False
    assert payload["public_diagnostic"]["category"] == "evaluation_bridge"
    assert payload["public_diagnostic"]["candidate_only"] is False and not marker.exists()


@pytest.mark.parametrize("kwargs,category", [({"exit_code": 3}, "candidate_execution"), ({"timed_out": True}, "candidate_timeout")])
def test_candidate_exit_and_timeout_are_candidate_faults(bound, monkeypatch, kwargs, category):
    task, workspace, marker = task_package(bound.root)
    fake_execute(monkeypatch, on_run=candidate_output, **kwargs)
    result = bound.root / "execution.json"
    assert execution_only(monkeypatch, bound, task, workspace, result) == 0
    payload = json.loads(result.read_text())
    assert payload["valid"] is False and payload["public_diagnostic"]["category"] == category
    assert payload["public_diagnostic"]["candidate_only"] is True and not marker.exists()


def test_session_sandbox_config_rewrites_host_python_to_image_python(bound):
    config = session.coder_sandbox_config(bound.root)
    assert config == {"wrapper": (), "rewrites": ((sys.executable, PYTHON),), "environment": (), "available": True}
    assert session.candidate_evaluator_arguments() == backend.evaluate_arguments(bound.binding)


def test_runtime_probe_runs_inside_bound_backend_with_persistent_records(bound, monkeypatch):
    calls = fake_execute(monkeypatch)
    completed = session.runtime_probe("import pyflakes")
    assert completed.returncode == 0 and completed.stdout == "synthetic stdout\n"
    call = calls[0]
    assert call["argv"] == (PYTHON, "-I", "-c", "import pyflakes") and call["timeout"] == 30
    assert call["workspace"].is_relative_to(bound.state.parent / "runtime_probes") and call["workspace"].is_dir()
    assert call["public_data"].parent == call["workspace"].parent and call["public_data"].is_dir()
    fake_execute(monkeypatch, status="infrastructure_error")
    with pytest.raises(RuntimeError, match="probe failed"):
        session.runtime_probe("import pyflakes")
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    assert session.runtime_probe("print(7)").stdout.strip() == "7" and len(calls) == 1


def test_root_evaluation_workspace_persists_under_run_dir_for_container_backend(bound, monkeypatch):
    starter = bound.root / "starter"; starter.mkdir(); (starter / "run.py").write_text("def main():\n    return 0\n")
    task_root = bound.root / "tasks/fixture"; (task_root / "problem/data").mkdir(parents=True)
    run_dir = bound.root / "run/session"; run_dir.mkdir(parents=True)
    seen = []
    def fake_run(argv, **kwargs):
        argv = list(argv)
        workspace = Path(argv[argv.index("--workspace") + 1]); result = Path(argv[argv.index("--result") + 1])
        seen.append((workspace, argv, (workspace / "run.py").is_file()))
        result.write_text(json.dumps({"schema_version": "external-domain-result-v1", "valid": True, "execution_valid": True,
                                      "score_bearing_evaluation_attempted": True, "metrics": {"aggregate_improvement": 0.25},
                                      "details": {"candidate_executable_sha256": session.sha(starter / "run.py"),
                                                  "submission_output_attestation": {"fixture": True}}}))
        return subprocess.CompletedProcess(argv, 0, "", "")
    monkeypatch.setattr(session, "subprocess", SimpleNamespace(run=fake_run, CompletedProcess=subprocess.CompletedProcess))
    metric, valid, _, _, _ = session.evaluate_root(starter=starter, task_root=task_root, run_dir=run_dir, timeout_seconds=5)
    workspace, argv, had_source = seen[0]
    assert metric == 0.25 and valid and had_source
    assert workspace.is_relative_to(run_dir / "root_evaluation") and (workspace / "run.py").is_file()
    assert argv[-2:] == list(backend.evaluate_arguments(bound.binding)) and "--require-candidate-sandbox" in argv
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    (run_dir / "root_evaluation.json").unlink()
    session.evaluate_root(starter=starter, task_root=task_root, run_dir=run_dir, timeout_seconds=5)
    workspace, argv, had_source = seen[1]
    assert had_source and not workspace.exists() and not workspace.is_relative_to(run_dir)
    assert argv[-1] == "--require-candidate-sandbox"


def test_runtime_probe_retries_a_cold_timeout_with_a_long_limit(bound, monkeypatch):
    # s41551-024-01257-9 lost its first attempt when a cold GPFS `import torch`
    # outlasted the 30 s probe. A timed-out first try is retried once, long.
    calls = []
    real = fake_execute(monkeypatch)
    inner = backend.execute
    def slow_then_fast(binding, **kwargs):
        calls.append(kwargs["timeout"])
        if len(calls) == 1:
            fake_execute(monkeypatch, timed_out=True)
            receipt = backend.execute(binding, **kwargs)
            monkeypatch.setattr(backend, "execute", slow_then_fast)
            return receipt
        return inner(binding, **kwargs)
    monkeypatch.setattr(backend, "execute", slow_then_fast)
    completed = session.runtime_probe("import torch")
    assert calls == [session.RUNTIME_PROBE_TIMEOUT_SECONDS, session.RUNTIME_PROBE_COLD_RETRY_SECONDS]
    assert completed.returncode == 0


def test_runtime_probe_that_times_out_twice_fails_loudly_not_as_no_gpu(bound, monkeypatch):
    fake_execute(monkeypatch, timed_out=True)
    with pytest.raises(RuntimeError, match="timed out"):
        session.runtime_probe("import torch")
    with pytest.raises(RuntimeError, match="timed out"):
        session.accelerator_inventory()


def test_local_runtime_probe_retries_after_subprocess_timeout(monkeypatch):
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    seen = []
    def fake_run(argv, **kwargs):
        seen.append(kwargs["timeout"])
        if len(seen) == 1:
            raise subprocess.TimeoutExpired(argv, kwargs["timeout"])
        return subprocess.CompletedProcess(argv, 0, "ok\n", "")
    monkeypatch.setattr(session.subprocess, "run", fake_run)
    assert session.runtime_probe("print(1)").stdout == "ok\n"
    assert seen == [session.RUNTIME_PROBE_TIMEOUT_SECONDS, session.RUNTIME_PROBE_COLD_RETRY_SECONDS]
    seen.clear()
    monkeypatch.setattr(session.subprocess, "run",
                        lambda argv, **kw: (_ for _ in ()).throw(subprocess.TimeoutExpired(argv, kw["timeout"])))
    with pytest.raises(RuntimeError, match="timed out"):
        session.runtime_probe("print(1)")
