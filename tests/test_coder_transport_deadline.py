"""Default action request deadline invariants, with virtual time and no network."""
from types import SimpleNamespace

import pytest

from evolvcoder import coder_engine as ce, naturebench_session as ns


def engine(model):
    obj = ce.ContractBoundedCoder.__new__(ce.ContractBoundedCoder)
    obj.model = model
    obj.config = SimpleNamespace(max_transport_retries=3, max_wall_seconds=100.0)
    obj._append_event = lambda events, name, **kw: events.append((name, kw))
    return obj


@pytest.mark.parametrize("configured", [300.0, None])
def test_default_near_deadline_clips_request_and_restores_on_failure(monkeypatch, configured):
    now, payload, received = [99.0], {"unchanged": ["request"]}, []
    monkeypatch.setattr(ce, "time", SimpleNamespace(monotonic=lambda: now[0],
                                                  sleep=lambda _: pytest.fail("no retry time remains")))

    class Model:
        request_total_timeout_seconds = configured

        def respond(self, body):
            assert body is payload
            received.append(self.request_total_timeout_seconds)
            now[0] += self.request_total_timeout_seconds
            raise TimeoutError("total response deadline exceeded")

    model, events = Model(), []
    with pytest.raises(TimeoutError):
        engine(model)._respond_with_transport_retry(payload, events, started=0)
    assert received == [1.0]
    assert model.request_total_timeout_seconds is configured
    assert events == []
    assert payload == {"unchanged": ["request"]}


@pytest.mark.parametrize("configured,expected", [(300.0, [80.0, 65.0]), (30.0, [30.0, 30.0])])
def test_retry_recomputes_remaining_without_restarting_action_clock(monkeypatch, configured, expected):
    now, received, slept, payload = [20.0], [], [], {"request": "same"}

    def sleep(seconds):
        assert model.request_total_timeout_seconds == configured
        slept.append(seconds)
        now[0] += seconds

    monkeypatch.setattr(ce, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=sleep))
    monkeypatch.setattr(ns, "transport_backoff_seconds", lambda _: 10.0)

    class Model:
        request_total_timeout_seconds = configured

        def respond(self, body):
            assert body is payload
            received.append(self.request_total_timeout_seconds)
            if len(received) == 1:
                now[0] = 25.0
                raise TimeoutError("temporary timed out request")
            return {"ok": True}, {}

    model, events = Model(), []
    assert engine(model)._respond_with_transport_retry(payload, events, started=0) == ({"ok": True}, {})
    assert received == expected and slept == [10.0]
    assert len(events) == 1
    assert model.request_total_timeout_seconds == configured


def test_expired_action_never_starts_a_request(monkeypatch):
    monkeypatch.setattr(ce, "time", SimpleNamespace(monotonic=lambda: 100.0))
    model = SimpleNamespace(request_total_timeout_seconds=300.0,
                            respond=lambda _: pytest.fail("expired action called provider"))
    with pytest.raises(TimeoutError, match="deadline reached"):
        engine(model)._respond_with_transport_retry({}, [], started=0)
    assert model.request_total_timeout_seconds == 300.0


def test_shared_session_deadline_is_tighter_than_fresh_action_allowance(monkeypatch):
    now, observed = [95.0], []
    monkeypatch.setattr(ce, "time", SimpleNamespace(monotonic=lambda: now[0]))
    model = SimpleNamespace(request_total_timeout_seconds=300.0)

    def respond(_):
        observed.append(model.request_total_timeout_seconds)
        return {}, {}

    model.respond = respond
    coder = engine(model)
    coder.session_deadline_monotonic = 100.0
    coder._respond_with_transport_retry({}, [], started=95.0)
    assert observed == [5.0]  # Not a fresh 100-second action window.
    assert model.request_total_timeout_seconds == 300.0
