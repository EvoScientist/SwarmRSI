from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest

from swarmrsi.baselines.aide_resources import (
    AideOperationBudget, aide_run_lock, run_bounded_process,
)
from swarmrsi.baselines.aide_session import AidePhaseCalls
from swarmrsi.baselines.aide_state import atomic_json, seal
from swarmrsi.baselines.aide_transport import AideCallLedger, AideResourceStop
from test_aide_execution_adapter import executor, fake_helper  # fixture, not an upstream replacement


def budget(root, artifacts, *, scores=2, wall=100.0):
    clock = [1000.0]
    ledger = AideCallLedger(
        root / "calls", model="test/model", max_calls=2, max_output_tokens=100,
        per_call_output_tokens=100, wall_seconds=wall,
        create=lambda **kw: {"model": "test/model", "usage": {
            "prompt_tokens": 1, "completion_tokens": 50}}, clock=lambda: clock[0],
    )
    return AideOperationBudget(ledger=ledger, artifacts=artifacts, max_score_attempts=scores), clock


@pytest.mark.parametrize("wall,expected", [(100, (20, 50)), (45, (15, 45)), (31, (1, 31))])
def test_external_timeout_includes_helper_overhead_inside_global_wall(tmp_path, wall, expected):
    bound, _ = budget(tmp_path, tmp_path / "artifacts", wall=wall)
    assert bound.external_timeout(kind="execute", configured_seconds=20) == expected
    assert bound.external_timeout(kind="score", configured_seconds=20) == expected


def test_expired_wall_and_changed_budget_cannot_reset_on_resume(tmp_path):
    bound, clock = budget(tmp_path, tmp_path / "artifacts")
    clock[0] += 101
    with pytest.raises(AideResourceStop, match="wall"):
        bound.before_new_search()
    with pytest.raises(AideResourceStop, match="wall"):
        bound.external_timeout(kind="score", configured_seconds=20)
    same = AideOperationBudget(ledger=bound.ledger, artifacts=bound.artifacts, max_score_attempts=2)
    assert same.remaining_wall() == -1
    with pytest.raises(ValueError, match="changed on resume"):
        AideOperationBudget(ledger=bound.ledger, artifacts=bound.artifacts, max_score_attempts=3)


def test_unresolved_started_score_consumes_cap(tmp_path):
    bound, _ = budget(tmp_path, tmp_path / "artifacts", scores=1)
    atomic_json(bound.artifacts / "unfinished/score_started.json", seal({"node_id": "unfinished"}))
    assert bound.score_attempts() == 1
    for kind in ("execute", "score"):
        with pytest.raises(AideResourceStop, match="score_attempt"):
            bound.external_timeout(kind=kind, configured_seconds=20)
    with pytest.raises(AideResourceStop, match="score_attempt"):
        bound.before_new_search()
    # Explicit execution-only fixtures do not secretly consume a score slot.
    assert bound.external_timeout(kind="execute", configured_seconds=20, execution_only=True) == (20, 50)


def test_accepted_response_replay_is_allowed_after_all_model_budgets_expire(tmp_path):
    bound, clock = budget(tmp_path, tmp_path / "artifacts")
    calls = AidePhaseCalls(bound.ledger, lambda response: response)
    request = {"model": "test/model"}
    with calls.phase(0):
        calls.create(**request)
        calls.create(**request)
    clock[0] += 101
    with pytest.raises(AideResourceStop):
        bound.before_new_search()
    with calls.phase(0):
        assert calls.create(**request)["usage"]["completion_tokens"] == 50
        assert calls.create(**request)["usage"]["completion_tokens"] == 50
    assert calls.count == 2


def test_real_score_gate_and_readonly_reconcile_after_expiry(executor, monkeypatch, tmp_path):
    bound, clock = budget(tmp_path / "controller", executor.artifacts, scores=1)
    executor.operation_budget = bound
    helper = fake_helper(executor)
    started = []

    def run(command, **kwargs):
        started.append(command[0])
        assert kwargs["timeout"] <= bound.remaining_wall()
        if command[0] == "/usr/bin/bwrap":
            return helper(command, **kwargs)
        target = Path(command[command.index("--result") + 1])
        atomic_json(target, {"valid": True, "execution_valid": True,
                            "score_bearing_evaluation_attempted": True,
                            "metrics": {"aggregate_improvement": -0.25}})
        return SimpleNamespace(stdout="", stderr="", returncode=0)

    monkeypatch.setattr(executor, "_run_external", run)
    result = executor.execute(node_id="scored", code="print(1)")
    assert bound.score_attempts() == 1 and len(started) == 2
    clock[0] += 101
    assert executor.reconcile(node_id="scored", code="print(1)") == result
    assert executor.verify_completed(node_id="scored", code="print(1)") == result
    with pytest.raises(AideResourceStop):
        executor.execute(node_id="not-started", code="print(2)")
    assert not (executor.artifacts / "not-started").exists()
    assert len(started) == 2


def test_first_score_on_reconcile_cannot_bypass_wall_gate(executor, monkeypatch, tmp_path):
    bound, clock = budget(tmp_path / "controller", executor.artifacts)
    executor.operation_budget = bound
    helper = fake_helper(executor, after=lambda: clock.__setitem__(0, clock[0] + 101))
    started = []

    def run(command, **kwargs):
        started.append(command)
        return helper(command, **kwargs)

    monkeypatch.setattr(executor, "_run_external", run)
    with pytest.raises(AideResourceStop, match="wall"):
        executor.execute(node_id="pending-score", code="print(1)")
    assert (executor.artifacts / "pending-score/execution_receipt.json").is_file()
    assert not (executor.artifacts / "pending-score/score_started.json").exists()
    with pytest.raises(AideResourceStop, match="wall"):
        executor.reconcile(node_id="pending-score", code="print(1)")
    assert len(started) == 1


@pytest.mark.parametrize("slow_path", ["phase_callback", "command_creation"])
def test_execute_does_not_dispatch_after_setup_consumes_deadline(executor, monkeypatch, tmp_path, slow_path):
    bound, clock = budget(tmp_path / "controller", executor.artifacts)
    executor.operation_budget = bound
    invoked = []
    monkeypatch.setattr("swarmrsi.baselines.aide_resources.run_bounded_process",
                        lambda *args, **kwargs: invoked.append(args))
    original_command = executor._command

    def command(*args, **kwargs):
        result = original_command(*args, **kwargs)
        if slow_path == "command_creation":
            clock[0] += 101
        return result

    def phase(name, payload):
        if name == "executing" and slow_path == "phase_callback":
            clock[0] += 101

    monkeypatch.setattr(executor, "_command", command)
    with pytest.raises(AideResourceStop, match="wall"):
        executor.execute(node_id="slow-setup", code="print(1)", phase_callback=phase)
    assert not invoked
    assert (executor.artifacts / "slow-setup/program.py").read_text() == "print(1)"
    assert (executor.artifacts / "slow-setup/execution_started.json").is_file()


def test_dispatch_shortens_both_helper_and_process_timeout(executor, monkeypatch, tmp_path):
    bound, clock = budget(tmp_path / "controller", executor.artifacts, wall=45)
    executor.operation_budget = bound
    executor.timeout_seconds = 20
    helper = fake_helper(executor)

    def phase(name, payload):
        if name == "executing":
            clock[0] += 10

    def run(command, **kwargs):
        assert command[command.index("--timeout-seconds") + 1] == "5"
        assert kwargs["timeout"] == 35
        return helper(command, **kwargs)

    monkeypatch.setattr("swarmrsi.baselines.aide_resources.run_bounded_process", run)
    assert executor.execute(node_id="shorter", code="print(1)", execution_only=True,
                            phase_callback=phase)["evaluation"]["valid"] is True


@pytest.mark.parametrize("slow_path", ["attestation", "receipt_write"])
def test_score_does_not_dispatch_after_evidence_preparation_consumes_deadline(executor, monkeypatch, tmp_path, slow_path):
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.subprocess.run", fake_helper(executor))
    executor.execute(node_id="score-setup", code="print(1)", execution_only=True)
    bound, clock = budget(tmp_path / "controller", executor.artifacts, scores=1)
    executor.operation_budget = bound
    invoked = []
    monkeypatch.setattr("swarmrsi.baselines.aide_resources.run_bounded_process",
                        lambda *args, **kwargs: invoked.append(args))
    original_attest, original_write = executor.bridge._attest, atomic_json

    def attest(path):
        result = original_attest(path)
        if slow_path == "attestation":
            clock[0] += 101
        return result

    def write(path, value):
        original_write(path, value)
        if path.name == "score_started.json" and slow_path == "receipt_write":
            clock[0] += 101

    monkeypatch.setattr(executor.bridge, "_attest", attest)
    monkeypatch.setattr("swarmrsi.baselines.aide_execution.atomic_json", write)
    with pytest.raises(AideResourceStop, match="wall"):
        executor.score(node_id="score-setup")
    assert not invoked and bound.score_attempts() == 1


def test_run_lock_second_process_rejected_without_state_mutation(tmp_path):
    src = Path(__file__).resolve().parents[1] / "src"
    code = (
        "from pathlib import Path; from swarmrsi.baselines.aide_resources import aide_run_lock; "
        f"ctx=aide_run_lock(Path({str(tmp_path)!r})); ctx.__enter__()"
    )
    with aide_run_lock(tmp_path):
        inode = (tmp_path / "controller.lock").stat().st_ino
        result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                                env={**os.environ, "PYTHONPATH": str(src)}, timeout=10)
        assert result.returncode != 0 and "active controller" in result.stderr
        assert (tmp_path / "controller.lock").stat().st_ino == inode
        assert sorted(path.name for path in tmp_path.iterdir()) == ["controller.lock"]
    with aide_run_lock(tmp_path):
        assert (tmp_path / "controller.lock").stat().st_ino == inode


@pytest.mark.parametrize("ignore_term", [False, True])
def test_process_timeout_kills_group_descendants(tmp_path, ignore_term):
    marker = tmp_path / "child-survived"
    child = (
        "import signal,time; from pathlib import Path; "
        + ("signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"time.sleep(2); Path({str(marker)!r}).write_text('should not happen')"
    )
    parent = (
        "import subprocess,sys,time; "
        + ("import signal; signal.signal(signal.SIGTERM, signal.SIG_IGN); " if ignore_term else "")
        + f"subprocess.Popen([sys.executable,'-c',{child!r}]); print('started',flush=True); time.sleep(10)"
    )
    started = time.monotonic()
    with pytest.raises(subprocess.TimeoutExpired) as error:
        run_bounded_process([sys.executable, "-c", parent], timeout=0.6, teardown_seconds=0.2)
    assert "started" in str(error.value.output)
    # The parent sleeps 10 s under a 0.6 s timeout; the bound only has to prove
    # the group was killed long before that. Interpreter spawn on a loaded
    # shared node has cost seconds, so 6 s replaces the 1.5 s of a quiet laptop.
    assert time.monotonic() - started < 6.0
    time.sleep(2.1)
    assert not marker.exists()


def test_normally_finished_parent_cannot_leave_background_child(tmp_path):
    marker = tmp_path / "orphan-survived"
    child = f"import time; from pathlib import Path; time.sleep(1); Path({str(marker)!r}).touch()"
    parent = (
        "import subprocess,sys; "
        f"subprocess.Popen([sys.executable,'-c',{child!r}],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL); "
        "print('done')"
    )
    result = run_bounded_process([sys.executable, "-c", parent], timeout=2)
    assert result.stdout.strip() == "done" and result.returncode == 0
    time.sleep(1.1)
    assert not marker.exists()
