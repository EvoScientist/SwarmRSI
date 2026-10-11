"""Complete session deadline/custody tests with virtual providers, never paid calls."""
import hashlib
import json
import math
from types import SimpleNamespace

import pytest

from evolvcoder import naturebench_cli as cli, naturebench_session as ns, session_runtime as sr
from evolvcoder.coder_openrouter import OpenRouterCoderModel
from evolvcoder.configuration import Limits
from evolvcoder.session_limits import SessionLimits
from test_naturebench_entrypoint import inputs, synthetic_root, forbid  # noqa: F401
from test_session_limits import usage, ACTION


def clock_fixture(monkeypatch):
    now = [0.0]
    monkeypatch.setattr(sr.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(sr.time, "time", lambda: 1000 + now[0])
    return now


def prepare(inputs, monkeypatch, *, valid=True, two_batches=False):
    now, received = clock_fixture(monkeypatch), []
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    raw["tasks"]["fixture"]["limits"]["wall_time_seconds"] = 100
    raw["screening_execution_limits"] = {"minimum_wall_seconds_to_start_action": 1}
    inputs.profile.write_text(json.dumps(raw))
    monkeypatch.setattr(ns, "runtime_probe", lambda _: SimpleNamespace(returncode=0))
    monkeypatch.setattr(ns, "runtime_dependency_inventory", lambda: ((), ()))
    monkeypatch.setattr(ns, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(ns, "public_task_context", lambda *a, **kw: "synthetic context")

    def root(**kw):
        value = synthetic_root(**kw)
        return value if valid else (None, False, value[2], None, value[4])

    monkeypatch.setattr(ns, "evaluate_root", root)
    decoder = OpenRouterCoderModel(api_key="never-sent", model="synthetic/model",
                                   base_url="https://example.invalid", native_tools=False, max_tokens_per_turn=100)

    class Model:
        native_tools = False
        request_total_timeout_seconds = 300.0

        def respond(self, request):
            received.append(self.request_total_timeout_seconds)
            if two_batches and len(received) == 1:
                now[0] = 95.0
                # Synthetic response bytes pass the real decoder and journal
                # verifier; neither the provider nor the grader is invoked.
                result = decoder._decode_message({"content": '{"proposals": []}'}, "PLANNER", "stop")
                body = json.dumps({"model": "synthetic/model",
                    "messages": [{"role": "user", "content": json.dumps(request)}]})
                result["_transport_request_body_json"] = body
                result["_transport_request_body_sha256"] = hashlib.sha256(body.encode()).hexdigest()
                return result, {"input_tokens": 11, "output_tokens": 13, "cost_usd": .2}
            now[0] += self.request_total_timeout_seconds
            raise TimeoutError("total response deadline exceeded")

    model = Model()
    monkeypatch.setattr(ns, "build_search_models", lambda *a: (SimpleNamespace(native_tools=True), model))
    return now, received


@pytest.mark.parametrize("valid", [True, False])
@pytest.mark.parametrize("two_batches", [True, False])
def test_complete_deadline_retains_incumbent_and_unknown_requests(inputs, monkeypatch, valid, two_batches):
    now, received = prepare(inputs, monkeypatch, valid=valid, two_batches=two_batches)
    if valid:
        assert cli.execute(inputs) == 0
    else:
        with pytest.raises(RuntimeError, match="session returned 1"):
            cli.execute(inputs)
    final = json.loads((inputs.run_dir / "final.json").read_text())
    cp = json.loads((inputs.run_dir / "checkpoint.json").read_text())
    assert received == ([100.0, 5.0] if two_batches else [100.0])
    assert final["best_node_id"] == ("root" if valid else None)
    assert final["stop_details"] == {"incumbent_retained": valid}
    assert final["accounting"]["model_calls"] == int(two_batches)
    assert final["accounting"]["wall_seconds"] == .01  # Old counter retains its meaning.
    assert final["session_clock"]["elapsed_seconds"] == 100.0
    receipt = final["planner_transport"]
    assert receipt == cp["proposer_state"]["transport_receipt"]
    assert receipt["unresolved_cost_attempts"] == 1
    assert receipt["known_cost_usd_subtotal"] == (.2 if two_batches else 0.0)
    assert receipt["pending_request_batches"] == [int(two_batches)]
    assert cp["proposer_state"]["terminal_intent"]["stop_reason"] == final["stop_reason"]
    monkeypatch.setattr(ns, "run_session", forbid)
    monkeypatch.setattr(ns, "build_search_models", forbid)
    inputs.credential_file.unlink()
    assert cli.execute(inputs) == (0 if valid else 1)
    side_file = next((inputs.run_dir / "planner_batches/transport_attempts").glob("*.started.json"))
    side_file.write_text(side_file.read_text() + " ")
    with pytest.raises(ValueError, match="transport receipt mismatch"):
        cli.execute(inputs)


def test_deadline_checkpoint_without_final_cannot_restart(inputs, monkeypatch):
    _, received = prepare(inputs, monkeypatch)
    write = ns.atomic_json

    def crash(path, value):
        if path.name == "final.json":
            raise OSError("synthetic interruption after terminal checkpoint")
        write(path, value)

    monkeypatch.setattr(ns, "atomic_json", crash)
    with pytest.raises(OSError, match="terminal checkpoint"):
        cli.execute(inputs)
    monkeypatch.setattr(ns, "atomic_json", write)
    assert not (inputs.run_dir / "final.json").exists()
    with pytest.raises(ValueError, match="unfinished prior launch"):
        cli.execute(inputs)
    # Even bypassing the CLI cannot restart a sealed deadline checkpoint.
    with pytest.raises(ValueError, match="terminal checkpoint"):
        ns.run_session(inputs)
    assert received == [100.0]


def test_new_terminal_cannot_omit_unknown_usage_receipt(inputs, monkeypatch):
    from evolvcoder.contracts import canonical_sha256
    prepare(inputs, monkeypatch)
    assert cli.execute(inputs) == 0
    cp_path, final_path = inputs.run_dir / "checkpoint.json", inputs.run_dir / "final.json"
    cp, final = json.loads(cp_path.read_text()), json.loads(final_path.read_text())
    cp["proposer_state"].pop("transport_receipt")
    cp.pop("checkpoint_sha256")
    cp["checkpoint_sha256"] = canonical_sha256(cp)
    final.pop("planner_transport")
    cp_path.write_text(json.dumps(cp))
    final_path.write_text(json.dumps(final))
    with pytest.raises(ValueError, match="transport receipt missing"):
        cli.execute(inputs)


def test_restore_clock_deducts_downtime_and_cannot_reset_identity(monkeypatch):
    now = clock_fixture(monkeypatch)
    clock = sr.SessionClock(started_unix=1000, started_monotonic=0, wall_limit=100, identity={"task": "A"})
    now[0] = 40
    saved = clock.snapshot()
    now[0] = 70
    restored = sr.SessionClock.restore(origin=clock.origin, snapshot=saved, wall_limit=100, identity={"task": "A"})
    assert restored.remaining() == 30
    now[0] = 80
    assert restored.remaining() == 20
    with pytest.raises(ValueError, match="identity"):
        sr.SessionClock.restore(origin=clock.origin, snapshot=saved, wall_limit=100, identity={"task": "B"})
    with pytest.raises(ValueError, match="missing"):
        sr.SessionClock.restore(origin=None, snapshot=saved, wall_limit=100, identity={"task": "A"})
    now[0] = 30
    with pytest.raises(ValueError, match="backwards"):
        sr.SessionClock.restore(origin=clock.origin, snapshot=saved, wall_limit=100, identity={"task": "A"})


@pytest.mark.parametrize("startup", [0.0, 20.0, 99.0])
def test_launch_origin_deducts_startup_from_earlier_work_deadline(monkeypatch, startup):
    now = clock_fixture(monkeypatch)
    now[0] = startup
    controls = sr.runtime_clock_controls(100, 100)
    clock = sr.SessionClock(started_unix=1000 + startup, started_monotonic=startup,
        wall_limit=100, identity={}, runtime_controls=controls, trial_started_unix=1000)
    assert clock.origin["wall_limit_seconds"] == 100
    assert controls["session_finalize_reserve_seconds"] == 5
    assert clock.remaining() == max(0.0, 95 - startup)
    assert clock.deadline_monotonic == max(startup, 95)
    saved = clock.snapshot()
    now[0] += 1
    restored = sr.SessionClock.restore(origin=clock.origin, snapshot=saved, wall_limit=100,
                                      identity={}, runtime_controls=controls)
    assert restored.remaining() == max(0.0, 94 - startup)


def test_zero_declared_execution_time_is_rejected():
    with pytest.raises(ValueError, match="does not fit"):
        sr.runtime_clock_controls(1, 100)


def test_clock_restricts_executor_without_changing_accounted_wall(monkeypatch, tmp_path):
    now = clock_fixture(monkeypatch)
    clock = sr.SessionClock(started_unix=1000, started_monotonic=0, wall_limit=100, identity={})
    budget = SessionLimits(Limits(wall_seconds=100))
    used = usage(wall_seconds=1)
    executor = ns.Executor(SimpleNamespace(), tmp_path, 0, tmp_path, (), ACTION)
    executor.session_clock = clock
    now[0] = 95
    assert clock.residual(budget, used)["remaining_wall"] == 5
    executor.set_remaining(used, budget.to_legacy())
    assert executor.remaining_wall == 5 and used.wall_seconds == 1
    now[0] = 100
    with pytest.raises(ns.SessionDeadlineReached):
        executor.set_remaining(used, budget.to_legacy())
    unlimited = sr.SessionClock(started_unix=1000, started_monotonic=0, wall_limit=None, identity={})
    assert unlimited.deadline_monotonic is None and math.isinf(unlimited.remaining())


@pytest.mark.parametrize("during_evaluation", [False, True])
def test_paid_coder_result_survives_evaluation_deadline(inputs, monkeypatch, during_evaluation):
    from test_internal_validation_gate import fake_coder, estimate
    from test_evolvcoder_incremental_feedback import campaign
    cli.configure(inputs)
    loop = campaign(inputs.profile.parent / "evaluation-case", scores=[])
    loop.expand()
    proposal, parent = next(iter(loop.actions.values())), loop.nodes["root"]
    now = clock_fixture(monkeypatch)
    base = fake_coder(proposal, parent, estimate(.2))

    class Coder(base):
        def run(self, *a, **kw):
            result = super().run(*a, **kw)
            now[0] = 95.0 if during_evaluation else 100.0
            return result

    monkeypatch.setattr(ns, "ContractBoundedCoder", Coder)
    monkeypatch.setattr(ns, "coder_sandbox_config", lambda _: {
        "available": True, "wrapper": (), "rewrites": (), "environment": ()})
    invoked = []

    def evaluate(*a, **kw):
        invoked.append(kw["timeout"])
        assert "5" == a[0][a[0].index("--timeout-seconds") + 1]
        now[0] = 100.0
        raise ns.subprocess.TimeoutExpired(a[0], kw["timeout"])

    monkeypatch.setattr(ns.subprocess, "run", evaluate)
    executor = ns.Executor(SimpleNamespace(model="synthetic"), inputs.run_dir, 0,
                           inputs.tasks_root / inputs.task_id, ("main",), ACTION)
    executor.session_clock = sr.SessionClock(started_unix=1000, started_monotonic=0, wall_limit=100, identity={})
    executor.set_remaining(usage(), SessionLimits(Limits()).to_legacy())
    execution = executor.execute(proposal=proposal, parent=parent, step=0)
    assert execution.outcome.model_calls == 1 and (execution.artifact / "run.py").is_file()
    assert not execution.artifact_valid and execution.native_metric is None
    assert execution.score_evaluations == int(during_evaluation)
    assert invoked == ([5.0] if during_evaluation else [])
