"""Coder transport re-send policy."""


def test_coder_waits_between_transport_resends(monkeypatch):
    """An immediate re-send lands inside the same outage.

    A proxy outage of several minutes ended ten paid cells at once while the coder
    spent its two re-sends in under a second. The planner already waited; this pins
    that the coder does too, and that the wait never exceeds the action's wall.
    """
    import swarmrsi.coder_engine as ce
    from swarmrsi.naturebench_session import transport_backoff_seconds

    slept = []
    monkeypatch.setattr(ce.time, "sleep", lambda s: slept.append(s))
    assert [transport_backoff_seconds(n) for n in (1, 2, 3, 4, 9)] == [5.0, 20.0, 80.0, 120.0, 120.0]

    class Boom:
        def __init__(self): self.calls = 0
        def respond(self, payload):
            self.calls += 1
            if self.calls <= 2:
                raise RuntimeError("URLError: <urlopen error Tunnel connection failed: 502 Parent proxy unreacheable>")
            return ("ok", {})

    class Cfg:
        max_transport_retries = 8
        max_wall_seconds = 3600.0

    engine = ce.ContractBoundedCoder.__new__(ce.ContractBoundedCoder)
    engine.model, engine.config = Boom(), Cfg()
    engine._append_event = lambda events, name, **kw: events.append((name, kw))
    events = []
    assert engine._respond_with_transport_retry({}, events, ce.time.monotonic()) == ("ok", {})
    assert slept == [5.0, 20.0] and len(events) == 2
    assert [e[1]["waited_seconds"] for e in events] == [5.0, 20.0]
