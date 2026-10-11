"""Original upstream Agent tests, with transport/execution mocked, not search.

Set AIDE_UPSTREAM_ROOT to the pinned checkout and put its real dependencies on
PYTHONPATH. Absence is an explicit skip locally, never a qualification pass.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import random
import sys
from types import SimpleNamespace

import pytest

from evolvcoder.baselines.aide_execution import AideExecutionUnresolved
from evolvcoder.baselines.aide_openrouter import assert_upstream_hashes, make_openrouter_query
from evolvcoder.baselines.aide_session import AidePhaseCalls, AideSession
from evolvcoder.baselines.aide_state import AideJournalStore, atomic_json, seal, unseal
from evolvcoder.baselines.aide_transport import AideCallLedger, AideTransportUnresolved

MODEL = "deepseek/deepseek-v4-pro"


class SimulatedInterruption(RuntimeError):
    pass


@pytest.fixture
def upstream(monkeypatch):
    location = os.environ.get("AIDE_UPSTREAM_ROOT")
    if not location:
        pytest.skip("real AIDE checkout/dependencies required for original Agent qualification")
    root = Path(location)
    repo = Path(__file__).resolve().parents[1]
    expected = json.loads((repo / "config/baseline_audits/naturebench_aide_fm_agent_reproduction_v1.json").read_text())["aide"]["source_attestation"]
    assert_upstream_hashes(root, expected)
    monkeypatch.syspath_prepend(str(root))
    monkeypatch.delenv("OPENAI_BASE_URL", raising=False)
    from aide import backend
    from aide.agent import Agent
    from aide.interpreter import ExecutionResult
    from aide.journal import Journal, Node
    from aide.utils.metric import MetricValue, WorstMetricValue
    from omegaconf import OmegaConf
    from openai.types.chat import ChatCompletion
    assert backend.determine_provider(MODEL) == "openrouter"
    yield SimpleNamespace(root=root, backend=backend, Agent=Agent, Journal=Journal,
                          Node=Node, MetricValue=MetricValue, WorstMetricValue=WorstMetricValue,
                          ExecutionResult=ExecutionResult, OmegaConf=OmegaConf,
                          response_factory=ChatCompletion.model_validate)
    assert_upstream_hashes(root, expected)


class Provider:
    def __init__(self, factory, *, malformed_review=None, extraction_retry=False):
        self.factory, self.requests = factory, []
        self.malformed_review, self.extraction_retry = malformed_review, extraction_retry
        self.generated = 0

    def create(self, **kwargs):
        self.requests.append(kwargs)
        assert kwargs["model"] == MODEL
        message = {"role": "assistant", "content": None}
        if kwargs.get("tools"):
            arguments = self.malformed_review or json.dumps({
                "is_bug": False, "metric": 999.0, "lower_is_better": True,
                "summary": "Original upstream review retained, even when the official metric disagrees.",
            })
            message["tool_calls"] = [{"id": "review", "type": "function", "function": {
                "name": "submit_review", "arguments": arguments,
            }}]
        else:
            self.generated += 1
            message["content"] = (
                "No code block on this first extraction attempt."
                if self.extraction_retry and self.generated == 1 else
                f"A concise proposed implementation.\n```python\nprint({self.generated})\n```"
            )
        return self.factory({
            "id": f"mock-{len(self.requests)}", "object": "chat.completion", "created": 1,
            "model": MODEL, "choices": [{"index": 0, "message": message, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7, "total_tokens": 18},
        })


class MockExecution:
    """Only the external execution/scoring boundary is replaced in these tests."""
    def __init__(self, root, scores):
        self.artifacts = root
        root.mkdir()
        self.scores, self.started = scores, []

    def execute(self, *, node_id, code, phase_callback):
        index = len(self.started)
        self.started.append(node_id)
        root = self.artifacts / node_id
        root.mkdir()
        (root / "program.py").write_text(code)
        phase_callback("executing", {"node_id": node_id})
        score = self.scores[index % len(self.scores)]
        result = {"term_out": [f"actual original ExecutionResult fixture {index}\n"],
                  "exec_time": 0.01, "exc_type": "ValueError" if score is None else None,
                  "exc_info": None, "exc_stack": None}
        if score is None:
            evaluation = {"valid": False, "execution_valid": False,
                          "score_bearing_evaluation_attempted": False,
                          "public_diagnostic": {"candidate_only": True, "category": "candidate_execution"}}
        else:
            evaluation = {"valid": True, "execution_valid": True,
                          "score_bearing_evaluation_attempted": True,
                          "metrics": {"aggregate_improvement": score}}
        bundle = {"node_id": node_id, "execution_result": result, "evaluation": evaluation,
                  "artifact": str(root), "snapshot_manifest": {"fixture": True}}
        atomic_json(root / "execution_bundle.json", seal(bundle))
        phase_callback("executed", {"node_id": node_id})
        return bundle

    def reconcile(self, *, node_id, code, phase_callback):
        root = self.artifacts / node_id
        if not (root / "execution_bundle.json").exists():
            raise AideExecutionUnresolved("incomplete execution must not be repeated")
        assert (root / "program.py").read_text() == code
        return unseal(json.loads((root / "execution_bundle.json").read_text()))

    def verify_completed(self, *, node_id, code):
        return self.reconcile(node_id=node_id, code=code, phase_callback=None)


class Harness:
    def __init__(self, root, upstream, monkeypatch, **provider_options):
        self.root, self.upstream, self.monkeypatch = root, upstream, monkeypatch
        self.provider = Provider(upstream.response_factory, **provider_options)
        self.executor = MockExecution(root / "artifacts", [None, 0.4, None, 0.2, 0.3, 0.45, 0.1, 0.48])
        self.store = AideJournalStore(root / "checkpoint.json", identity={"fixture": "original-agent", "seed": 17})
        self.cfg = upstream.OmegaConf.load(upstream.root / "aide/utils/config.yaml")
        self.cfg.workspace_dir = str(root / "public-workspace")
        workspace = Path(self.cfg.workspace_dir)
        workspace.mkdir()
        (workspace / "public.txt").write_text("public fixture only")
        for entry in (self.cfg.agent.code, self.cfg.agent.feedback, self.cfg.report):
            entry.model = MODEL
        assert self.cfg.agent.steps == 20
        assert dict(self.cfg.agent.search) == {"max_debug_depth": 3, "debug_prob": 0.5, "num_drafts": 5}
        random.seed(17)

    def session(self):
        upstream = self.upstream
        ledger = AideCallLedger(self.root / "calls", model=MODEL, max_calls=100,
                               max_output_tokens=10000, wall_seconds=10000,
                               per_call_output_tokens=100, create=self.provider.create)
        calls = AidePhaseCalls(ledger, upstream.response_factory)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls.create)))
        adapter = make_openrouter_query(
            client=client, backoff_create=lambda create, errors, **kw: create(**kw), timeout_exceptions=(),
        )
        self.monkeypatch.setattr(upstream.backend.backend_openrouter, "query", adapter)
        self.monkeypatch.setitem(upstream.backend.provider_to_query_func, "openrouter", adapter)
        agent = upstream.Agent(task_desc="Synthetic public task; official score is authoritative.",
                               cfg=self.cfg, journal=upstream.Journal())
        return AideSession(
            agent=agent, store=self.store, calls=calls, executor=self.executor,
            node_factory=upstream.Node, journal_factory=upstream.Journal,
            metric_factory=upstream.MetricValue, worst_metric_factory=upstream.WorstMetricValue,
            execution_result_factory=upstream.ExecutionResult,
        )


def test_twenty_original_steps_all_stages_and_official_best(tmp_path, upstream, monkeypatch):
    harness = Harness(tmp_path, upstream, monkeypatch)
    session = harness.session()
    for _ in range(20):
        session.step()
    nodes = session.agent.journal.nodes
    assert len(nodes) == 20 and session.phase == "ready"
    assert [node.stage_name for node in nodes[:5]] == ["draft"] * 5
    assert {node.stage_name for node in nodes} == {"draft", "debug", "improve"}
    assert len(harness.executor.started) == 20
    assert len(harness.provider.requests) == 40
    assert len(session.calls.ledger.read()["attempts"]) == 40
    assert session.agent.journal.get_best_node().metric.value == 0.48
    assert any(node.is_buggy for node in nodes)
    for index, node in enumerate(nodes):
        assert node.step == index
        assert node.analysis.startswith("Original upstream review")
        binding = unseal(json.loads((harness.executor.artifacts / node.id / "review_binding.json").read_text()))
        assert binding["llm_review"]["metric"] in (999.0, None)
        if not node.is_buggy:
            assert node.metric.value != 999.0 and node.metric.maximize is True
    restored = harness.session()
    assert [node.id for node in restored.agent.journal.nodes] == [node.id for node in nodes]
    assert restored.agent.data_preview == session.agent.data_preview


@pytest.mark.parametrize("phase", ["generating", "generated", "executed", "reviewing", "reviewed"])
def test_resume_original_pending_phase_once(tmp_path, upstream, monkeypatch, phase):
    harness = Harness(tmp_path, upstream, monkeypatch)
    session = harness.session()
    for _ in range(5):
        session.step()
    original_save = harness.store.save
    interrupted = []
    def save(**kwargs):
        original_save(**kwargs)
        if kwargs["phase"] == phase and not interrupted:
            interrupted.append(True)
            raise SimulatedInterruption(phase)
    monkeypatch.setattr(harness.store, "save", save)
    with pytest.raises(SimulatedInterruption):
        session.step()
    saved = unseal(json.loads(harness.store.path.read_text()))
    assert len(saved["nodes"]) == 5
    if phase != "generating":
        assert saved["pending"]["parent_id"] in {row["id"] for row in saved["nodes"]}
    resumed = harness.session()
    def no_new_search():
        raise AssertionError("resolving a pending node must not choose another parent")
    monkeypatch.setattr(resumed.agent, "search_policy", no_new_search)
    resumed.step()
    assert len(resumed.agent.journal.nodes) == 6
    assert len(harness.executor.started) == 6
    assert len(harness.provider.requests) == 12
    assert resumed.phase == "ready"


@pytest.mark.parametrize("malformed", ["{", "not-json"])
def test_malformed_review_resume_retains_branch_and_never_rebuys(tmp_path, upstream, monkeypatch, malformed):
    harness = Harness(tmp_path, upstream, monkeypatch, malformed_review=malformed)
    session = harness.session()
    with pytest.raises(json.JSONDecodeError):
        session.step()
    original_requests = len(harness.provider.requests)
    resumed = harness.session()
    with pytest.raises(json.JSONDecodeError):
        resumed.step()
    assert len(harness.provider.requests) == original_requests == 2
    assert len(harness.executor.started) == 1
    assert resumed.pending.code and not resumed.agent.journal.nodes
    assert len(resumed.calls.ledger.read()["attempts"]) == 2


def test_original_extraction_retry_is_charged_and_replayed(tmp_path, upstream, monkeypatch):
    harness = Harness(tmp_path, upstream, monkeypatch, extraction_retry=True)
    session = harness.session()
    original = session.calls.create
    interrupted = []
    def create(**kwargs):
        response = original(**kwargs)
        if not interrupted:
            interrupted.append(True)
            raise SimulatedInterruption("accepted response before extraction")
        return response
    # Backend client bound this method at construction; install a fresh query
    # using the wrapper solely to simulate a crash after durable acceptance.
    client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    adapter = make_openrouter_query(client=client,
        backoff_create=lambda provider, errors, **kw: provider(**kw), timeout_exceptions=())
    monkeypatch.setitem(upstream.backend.provider_to_query_func, "openrouter", adapter)
    with pytest.raises(SimulatedInterruption):
        session.step()
    resumed = harness.session()
    resumed.step()
    assert len(harness.provider.requests) == 3  # two original generation attempts, one review
    assert len(harness.executor.started) == 1
    assert len(resumed.agent.journal.nodes) == 1


def test_unknown_execution_does_not_start_another_candidate(tmp_path, upstream, monkeypatch):
    harness = Harness(tmp_path, upstream, monkeypatch)
    session = harness.session()
    original_save = harness.store.save
    def save(**kwargs):
        original_save(**kwargs)
        if kwargs["phase"] == "executing":
            raise SimulatedInterruption("execution acceptance unknown")
    monkeypatch.setattr(harness.store, "save", save)
    with pytest.raises(SimulatedInterruption):
        session.step()
    resumed = harness.session()
    with pytest.raises(AideExecutionUnresolved):
        resumed.step()
    assert len(harness.executor.started) == len(harness.provider.requests) == 1
    assert len(resumed.agent.journal.nodes) == 0


@pytest.mark.parametrize("damage", ["binding_missing", "bundle_changed", "code_changed", "anchor_missing"])
def test_ready_resume_validates_completed_history_before_new_calls(tmp_path, upstream, monkeypatch, damage):
    harness = Harness(tmp_path, upstream, monkeypatch)
    session = harness.session()
    session.step()
    node = session.agent.journal.nodes[0]
    root = harness.executor.artifacts / node.id
    if damage == "binding_missing":
        (root / "review_binding.json").unlink()
    elif damage == "code_changed":
        (root / "program.py").write_text("changed code")
    elif damage == "anchor_missing":
        state = unseal(json.loads(harness.store.path.read_text()))
        state["metadata"]["completed_evidence"].clear()
        atomic_json(harness.store.path, seal(state))
    else:
        bundle = unseal(json.loads((root / "execution_bundle.json").read_text()))
        bundle["execution_result"]["term_out"] = ["changed log"]
        atomic_json(root / "execution_bundle.json", seal(bundle))
    with pytest.raises((ValueError, AssertionError, FileNotFoundError)):
        harness.session()
    assert len(harness.provider.requests) == 2
    assert len(harness.executor.started) == 1
