from __future__ import annotations

import json
import random
from types import SimpleNamespace

import pytest

from swarmrsi.baselines.aide_state import AideJournalStore, atomic_json, seal, unseal
from swarmrsi.baselines.aide_transport import (
    AideCallLedger, AideResourceStop, AideTransportUnresolved,
)
from swarmrsi.baselines.aide_openrouter import make_openrouter_query


class Metric:
    def __init__(self, value=None, *, maximize=None):
        self.value, self.maximize = value, maximize
        self.is_worst = value is None


class Node:
    def __init__(self, **kwargs):
        self.__dict__.update(dict(
            code="print('x')", plan="plan", step=None, id="root", ctime=1.25,
            _term_out=["ok\n"], exec_time=1.0, exc_type=None, exc_info=None,
            exc_stack=None, analysis="measured result", is_buggy=False,
            parent=None, metric=Metric(0.2, maximize=True),
        ) | kwargs)
        self.children = set()
        if self.parent is not None:
            self.parent.children.add(self)


class Journal:
    def __init__(self):
        self.nodes = []

    def append(self, node):
        node.step = len(self.nodes)
        self.nodes.append(node)


def restore(store):
    return store.restore(node_factory=Node, journal_factory=Journal,
                         metric_factory=Metric, worst_metric_factory=Metric)


def make_journal():
    journal = Journal()
    root = Node()
    journal.append(root)
    failed = Node(id="failed", parent=root, is_buggy=True, metric=Metric(),
                  exc_type="ValueError", exc_info={"args": ["bad shape"]},
                  exc_stack=[["runfile.py", 3, "<module>", "raise ValueError()"]])
    journal.append(failed)
    pending = Node(id="pending", parent=failed, metric=None, is_buggy=None,
                   _term_out=None, exec_time=None, analysis=None)
    return journal, pending


def test_pending_edges_preview_rng_and_exception_roundtrip(tmp_path):
    journal, pending = make_journal()
    store = AideJournalStore(tmp_path / "state.json", identity={"source": "fixed", "seed": 7})
    rng_before = random.getstate()
    try:
        random.seed(7)
        store.save(journal=journal, pending=pending, preview="original public preview",
                   phase="generated", metadata={"artifact": "node/pending"})
        expected_next = random.random()
        checkpoint = store.path.read_bytes()
        value = restore(store)
        assert [node.id for node in value.journal.nodes] == ["root", "failed"]
        assert value.pending is not None and value.pending.step is None
        assert value.pending.parent is value.journal.nodes[1]
        assert value.pending in value.pending.parent.children
        assert value.journal.nodes[1].metric.is_worst
        assert value.journal.nodes[1].exc_stack == journal.nodes[1].exc_stack
        assert value.preview == "original public preview"
        random.setstate(value.rng_state)
        assert random.random() == expected_next
        random.setstate(value.rng_state)
        store.save(journal=value.journal, pending=value.pending, preview=value.preview,
                   phase=value.phase, metadata=value.metadata)
        assert store.path.read_bytes() == checkpoint
    finally:
        random.setstate(rng_before)


def test_state_rejects_source_drift_and_content_corruption(tmp_path):
    journal, pending = make_journal()
    store = AideJournalStore(tmp_path / "state.json", identity={"source": "fixed"})
    store.save(journal=journal, pending=pending, preview=None, phase="generated")
    with pytest.raises(ValueError, match="identity changed"):
        restore(AideJournalStore(store.path, identity={"source": "changed"}))
    payload = json.loads(store.path.read_text())
    payload["nodes"][0]["code"] = "different"
    atomic_json(store.path, payload)
    with pytest.raises(ValueError, match="hash mismatch"):
        restore(store)


@pytest.mark.parametrize("damage", ["pending_edge", "order", "metric", "duplicate"])
def test_semantic_corruption_rejected_even_if_resealed(tmp_path, damage):
    journal, pending = make_journal()
    store = AideJournalStore(tmp_path / "state.json", identity={})
    store.save(journal=journal, pending=pending, preview=None, phase="generated")
    value = unseal(json.loads(store.path.read_text()))
    if damage == "pending_edge":
        value["nodes"][1]["children_ids"] = []
    elif damage == "order":
        value["nodes"][0]["step"] = 7
    elif damage == "metric":
        value["nodes"][1]["metric"]["value"] = 42.0
    else:
        value["pending"]["id"] = "failed"
    atomic_json(store.path, seal(value))
    with pytest.raises(ValueError):
        restore(store)


def response(*, tokens=7, usage=True, tool_json=None):
    message = SimpleNamespace(content="plan\n```python\nprint(1)\n```", tool_calls=[])
    if tool_json is not None:
        message.tool_calls = [SimpleNamespace(function=SimpleNamespace(
            name="submit_review", arguments=tool_json))]
    return SimpleNamespace(
        choices=[SimpleNamespace(message=message)],
        usage=SimpleNamespace(prompt_tokens=11, completion_tokens=tokens) if usage else None,
        model="deepseek/deepseek-v4-pro", created=1,
    )


class Provider:
    def __init__(self, items):
        self.items, self.requests = list(items), []

    def create(self, **kwargs):
        self.requests.append(kwargs)
        value = self.items.pop(0)
        if isinstance(value, BaseException):
            raise value
        return value


def ledger(tmp_path, provider, **changes):
    kwargs = dict(model="deepseek/deepseek-v4-pro", max_calls=3,
                  max_output_tokens=20, wall_seconds=100,
                  per_call_output_tokens=10, create=provider.create)
    return AideCallLedger(tmp_path / "calls", **(kwargs | changes))


REQUEST = {"model": "deepseek/deepseek-v4-pro", "messages": [{"role": "user", "content": "code"}]}


def test_provider_attempts_are_durable_and_remaining_output_is_reserved(tmp_path):
    provider = Provider([response(tokens=7), response(tokens=7), response(tokens=6)])
    calls = ledger(tmp_path, provider)
    for _ in range(3):
        calls.create(**REQUEST)
    assert [row["max_tokens"] for row in provider.requests] == [10, 10, 6]
    assert sum(row["output_tokens"] for row in calls.read()["attempts"]) == 20
    with pytest.raises(AideResourceStop):
        calls.create(**REQUEST)
    assert len(provider.requests) == 3


@pytest.mark.parametrize("malformed", ["{", "not-json"])
def test_malformed_review_does_not_lose_a_successful_response_usage(tmp_path, malformed):
    provider = Provider([response(tool_json=malformed)])
    calls = ledger(tmp_path, provider)
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls.create)))
    spec = SimpleNamespace(
        name="submit_review", as_openai_tool_dict={"type": "function"},
        openai_tool_choice_dict={"type": "function"},
    )
    query = make_openrouter_query(client=client,
        backoff_create=lambda create, exceptions, **kwargs: create(**kwargs),
        timeout_exceptions=())
    with pytest.raises(json.JSONDecodeError):
        query("system", "review", func_spec=spec, model=REQUEST["model"])
    record = calls.read()["attempts"][0]
    assert record["state"] == "completed" and record["output_tokens"] == 7
    assert (calls.root / "000000.response.json").is_file()


@pytest.mark.parametrize("failure", [TimeoutError("secret must not appear"), KeyboardInterrupt()])
def test_uncertain_transport_is_not_retried_or_forgotten_on_resume(tmp_path, failure):
    provider = Provider([failure, response()])
    calls = ledger(tmp_path, provider)
    with pytest.raises(AideTransportUnresolved):
        calls.create(**REQUEST)
    resumed = ledger(tmp_path, provider)
    with pytest.raises(AideTransportUnresolved):
        resumed.create(**REQUEST)
    assert len(provider.requests) == 1
    assert "secret must not appear" not in calls.path.read_text()


def test_missing_usage_saved_and_poisoned(tmp_path):
    provider = Provider([response(usage=False)])
    calls = ledger(tmp_path, provider)
    with pytest.raises(AideTransportUnresolved):
        calls.create(**REQUEST)
    assert (calls.root / "000000.response.json").is_file()
    assert calls.read()["attempts"][0]["state"] == "usage_unresolved"


def test_completed_response_replay_is_exact_and_zero_call(tmp_path):
    provider = Provider([response()])
    calls = ledger(tmp_path, provider)
    calls.create(**REQUEST)
    raw = calls.replay(0, logical_request=REQUEST, response_factory=lambda value: value)
    assert raw["usage"]["completion_tokens"] == 7
    assert len(provider.requests) == 1
    with pytest.raises(ValueError, match="differs"):
        calls.replay(0, logical_request=REQUEST | {"temperature": 0.1}, response_factory=lambda value: value)
    path = calls.root / "000000.response.json"
    value = unseal(json.loads(path.read_text()))
    value["response"]["usage"]["completion_tokens"] = 0
    atomic_json(path, seal(value))
    with pytest.raises(ValueError, match="ledger"):
        calls.replay(0, logical_request=REQUEST, response_factory=lambda value: value)


def test_cross_instance_call_ceiling_and_wall_ceiling(tmp_path):
    provider = Provider([response()])
    first = ledger(tmp_path, provider, max_calls=1)
    other = ledger(tmp_path, provider, max_calls=1)
    first.create(**REQUEST)
    with pytest.raises(AideResourceStop, match="model_call"):
        other.create(**REQUEST)
    clock = [100.0]
    timed = ledger(tmp_path / "timed", Provider([]), clock=lambda: clock[0])
    clock[0] += 101
    with pytest.raises(AideResourceStop, match="wall"):
        timed.create(**REQUEST)


def test_model_drift_request_rejected_before_provider(tmp_path):
    provider = Provider([])
    calls = ledger(tmp_path, provider)
    with pytest.raises(ValueError, match="unapproved backbone"):
        calls.create(**(REQUEST | {"model": "other-model"}))
    assert not provider.requests
