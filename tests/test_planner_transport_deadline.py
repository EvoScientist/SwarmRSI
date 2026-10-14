"""Zero-provider-call transport tests; session integration has separate gates."""
import json
from types import SimpleNamespace

import pytest

from swarmrsi import naturebench_session as ns


def planner(model, root, deadline=100):
    obj = ns.Planner.__new__(ns.Planner)
    obj.model, obj.audit_root, obj.batch = model, root, 0
    obj.max_transport_retries = 3
    obj.session_deadline_monotonic = deadline
    return obj


@pytest.mark.parametrize("configured", [300.0, None, 2.0])
def test_one_deadline_clips_and_restores_request(monkeypatch, tmp_path, configured):
    now, received = [95.0], []
    monkeypatch.setattr(ns.time, "monotonic", lambda: now[0])

    class Model:
        request_total_timeout_seconds = configured

        def respond(self, request):
            assert list(tmp_path.glob("transport_attempts/*.started.json"))
            received.append(self.request_total_timeout_seconds)
            now[0] = 100.0
            raise TimeoutError("total response deadline exceeded; SECRET")

    model = Model()
    with pytest.raises(ns.SessionDeadlineReached):
        planner(model, tmp_path)._respond_with_transport_retry({"same": True})
    assert received == [2.0 if configured == 2.0 else 5.0]
    assert model.request_total_timeout_seconds is configured
    outcome = next(tmp_path.glob("transport_attempts/*.outcome.json")).read_text()
    assert "SECRET" not in outcome
    assert json.loads(outcome)["provider_cost_status"] == "unknown"


def test_expired_before_request_writes_no_attempt(monkeypatch, tmp_path):
    monkeypatch.setattr(ns.time, "monotonic", lambda: 100.0)
    model = SimpleNamespace(respond=lambda _: pytest.fail("called expired transport"))
    with pytest.raises(ns.SessionDeadlineReached):
        planner(model, tmp_path)._respond_with_transport_retry({})
    assert not list(tmp_path.iterdir())


def test_successful_retry_does_not_erase_unknown_attempt(monkeypatch, tmp_path):
    now, received, slept = [10.0], [], []
    monkeypatch.setattr(ns.time, "monotonic", lambda: now[0])
    monkeypatch.setattr(ns, "transport_backoff_seconds", lambda _: 5.0)

    def sleep(seconds):
        assert model.request_total_timeout_seconds == 300.0
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(ns.time, "sleep", sleep)
    body = {"identical": [1]}

    class Model:
        request_total_timeout_seconds = 300.0

        def respond(self, request):
            assert request is body
            received.append(self.request_total_timeout_seconds)
            now[0] += 10
            if len(received) < 3:
                raise TimeoutError("temporary timed out")
            return {"ok": True}, {"cost_usd": 0.2, "input_tokens": 7, "headers": "SECRET"}

    model = Model()
    result = planner(model, tmp_path)._respond_with_transport_retry(body)
    assert result[0] == {"ok": True}
    assert received == [90.0, 75.0, 60.0] and slept == [5.0, 5.0]
    records = [json.loads(p.read_text()) for p in sorted(tmp_path.glob("transport_attempts/*.outcome.json"))]
    assert [r["provider_cost_status"] for r in records] == ["unknown", "unknown", "reported"]
    assert records[2]["usage"] == {"input_tokens": 7, "cost_usd": 0.2}
    assert len({r["request_sha256"] for r in records}) == 1
    assert model.request_total_timeout_seconds == 300.0


def test_insufficient_backoff_never_sleeps_or_retries(monkeypatch, tmp_path):
    monkeypatch.setattr(ns.time, "monotonic", lambda: 99.0)
    monkeypatch.setattr(ns.time, "sleep", lambda _: pytest.fail("slept beyond deadline"))

    def respond(_):
        raise TimeoutError("temporary timed out")

    with pytest.raises(ns.SessionDeadlineReached, match="backoff"):
        planner(SimpleNamespace(respond=respond), tmp_path)._respond_with_transport_retry({})
    assert len(list(tmp_path.glob("transport_attempts/*.started.json"))) == 1


def test_auth_failure_not_retried(monkeypatch, tmp_path):
    now = [0]
    monkeypatch.setattr(ns.time, "monotonic", lambda: now[0])

    def respond(_):
        now[0] = 101
        raise ValueError("authentication refused")

    with pytest.raises(ValueError, match="authentication"):
        planner(SimpleNamespace(respond=respond), tmp_path)._respond_with_transport_retry({})
    assert len(list(tmp_path.glob("transport_attempts/*.started.json"))) == 1


def test_slow_attempt_record_cannot_start_expired_transport(monkeypatch, tmp_path):
    now = [95.0]
    monkeypatch.setattr(ns.time, "monotonic", lambda: now[0])
    original = ns.atomic_json

    def slow_write(path, value):
        original(path, value)
        now[0] += 5.0

    monkeypatch.setattr(ns, "atomic_json", slow_write)
    model = SimpleNamespace(respond=lambda _: pytest.fail("called after slow write"))
    with pytest.raises(ns.SessionDeadlineReached):
        planner(model, tmp_path)._respond_with_transport_retry({})
    outcome = json.loads(next(tmp_path.glob("transport_attempts/*.outcome.json")).read_text())
    assert outcome["status"] == "request_not_sent"
    assert outcome["provider_cost_status"] == "not_sent"


def test_orphan_attempt_cannot_be_reissued(monkeypatch, tmp_path):
    monkeypatch.setattr(ns.time, "monotonic", lambda: 0)
    calls = []
    model = SimpleNamespace(respond=lambda _: (calls.append(1) or {}, {}))
    obj = planner(model, tmp_path)
    obj._respond_with_transport_retry({})
    with pytest.raises(ValueError, match="refusing reissue"):
        obj._respond_with_transport_retry({})
    assert len(calls) == 1
    outcome = json.loads(next(tmp_path.glob("transport_attempts/*.outcome.json")).read_text())
    assert outcome["provider_cost_status"] == "unknown"


@pytest.mark.parametrize("fail_kind,expected_calls", [("started", 0), ("outcome", 1)])
def test_storage_failure_never_hides_attempt_or_changes_timeout(
    monkeypatch, tmp_path, fail_kind, expected_calls,
):
    monkeypatch.setattr(ns.time, "monotonic", lambda: 95.0)
    original, calls = ns.atomic_json, []

    def fail_write(path, value):
        if path.name.endswith(f".{fail_kind}.json"):
            raise OSError("disk unavailable")
        original(path, value)

    monkeypatch.setattr(ns, "atomic_json", fail_write)
    model = SimpleNamespace(request_total_timeout_seconds=300.0,
                            respond=lambda _: (calls.append(1) or {}, {}))
    obj = planner(model, tmp_path)
    with pytest.raises(OSError, match="disk unavailable"):
        obj._respond_with_transport_retry({})
    assert len(calls) == expected_calls
    assert model.request_total_timeout_seconds == 300.0
    assert not list(tmp_path.glob("transport_attempts/*.outcome.json"))
    if fail_kind == "outcome":
        assert len(list(tmp_path.glob("transport_attempts/*.started.json"))) == 1
        monkeypatch.setattr(ns, "atomic_json", original)
        with pytest.raises(ValueError, match="refusing reissue"):
            obj._respond_with_transport_retry({})
        assert len(calls) == 1
