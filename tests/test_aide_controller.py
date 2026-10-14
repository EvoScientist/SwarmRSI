"""The final controller drives actual pinned Agent/Journal/report methods.

Only provider and candidate/scorer boundaries are synthetic. Missing original
AIDE dependencies is a skip, never a remote qualification pass.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import random
from types import SimpleNamespace

import pytest

import swarmrsi.baselines.aide_controller as controller
from swarmrsi.baselines.aide_controller import AideRuntime, run_controller, verify_snapshot
from swarmrsi.baselines.aide_openrouter import make_openrouter_query
from swarmrsi.baselines.aide_resources import AideOperationBudget
from swarmrsi.baselines.aide_session import AidePhaseCalls, AideSession
from swarmrsi.baselines.aide_state import AideJournalStore, atomic_json, seal, unseal
from swarmrsi.baselines.aide_transport import AideCallLedger
from test_aide_original_session import MODEL, Provider, MockExecution, upstream


class Artifacts(MockExecution):
    def __init__(self, root, scores):
        self.artifacts = root / "artifacts"
        self.artifacts.mkdir(exist_ok=True)
        self.scores, self.started = scores, []
        self.workspace = root / "workspace"
        self.workspace.mkdir(exist_ok=True)

    def execute(self, *, node_id, code, phase_callback):
        result = super().execute(node_id=node_id, code=code, phase_callback=phase_callback)
        node_root = self.artifacts / node_id
        for name in ("workspace", "artifact"):
            path = node_root / name
            path.mkdir()
            (path / "run.py").write_text(code)
            (path / "output").mkdir()
            (path / "output/predictions.txt").write_text("synthetic output")
        (node_root / "workspace/cache.pkl").write_bytes(b"synthetic retained cache")
        (node_root / "workspace/helper.py").write_text("# retained helper\n")
        (node_root / "workspace/external-link").symlink_to("/not-readable-outside-fixture")
        (self.workspace / "working.txt").write_text("persistent working state")
        if result["evaluation"]["valid"]:
            atomic_json(node_root / "score_started.json", seal({"node_id": node_id}))
        return result


class Campaign:
    def __init__(self, upstream, monkeypatch, *, calls=100, scores=None, score_cap=30,
                 interrupt_reviewed=False, interrupt_report=False):
        self.upstream, self.monkeypatch = upstream, monkeypatch
        self.provider = Provider(upstream.response_factory)
        self.max_calls, self.score_cap = calls, score_cap
        self.scores = scores or [None, -0.4, None, -0.2, -0.3, -0.1]
        self.interrupt_reviewed, self.interrupt_report = interrupt_reviewed, interrupt_report
        self.runtime, self.executor = None, None
        self.clock = [1000.0]

    def initialize(self, root):
        up = self.upstream
        if self.executor is None:
            self.executor = Artifacts(root, self.scores)
        store = AideJournalStore(root / "checkpoint.json", identity={"fixture": "full-original-controller"})
        if self.interrupt_reviewed:
            save = store.save

            def interrupted(**kwargs):
                save(**kwargs)
                if kwargs["phase"] == "reviewed" and self.interrupt_reviewed:
                    self.interrupt_reviewed = False
                    raise RuntimeError("synthetic interruption after durable review")

            self.monkeypatch.setattr(store, "save", interrupted)
        cfg = up.OmegaConf.load(up.root / "aide/utils/config.yaml")
        cfg.workspace_dir = str(self.executor.workspace)
        for entry in (cfg.agent.code, cfg.agent.feedback, cfg.report):
            entry.model = MODEL
        if not store.path.exists():
            random.seed(17)
        ledger = AideCallLedger(root / "calls", model=MODEL, max_calls=self.max_calls,
            max_output_tokens=10000, wall_seconds=10000, per_call_output_tokens=5000,
            create=self.provider.create, clock=lambda: self.clock[0])
        calls = AidePhaseCalls(ledger, up.response_factory)
        client = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=calls.create)))
        adapter = make_openrouter_query(client=client,
            backoff_create=lambda create, errors, **kw: create(**kw), timeout_exceptions=())
        self.monkeypatch.setitem(up.backend.provider_to_query_func, "openrouter", adapter)
        factories = {"node_factory": up.Node, "journal_factory": up.Journal,
                     "metric_factory": up.MetricValue, "worst_metric_factory": up.WorstMetricValue}
        agent = up.Agent(task_desc="Synthetic public task only.", cfg=cfg, journal=up.Journal())
        session = AideSession(agent=agent, store=store, calls=calls, executor=self.executor,
            execution_result_factory=up.ExecutionResult, **factories)
        budget = AideOperationBudget(ledger=ledger, artifacts=self.executor.artifacts,
                                    max_score_attempts=self.score_cap)

        def report(journal):
            from aide.journal2report import journal2report
            text = journal2report(journal, agent.task_desc, cfg.report)
            if self.interrupt_report:
                self.interrupt_report = False
                raise RuntimeError("synthetic report interruption after accepted response")
            return text

        self.runtime = AideRuntime(session=session, budget=budget, restore_factories=factories, report=report)
        return self.runtime


def launch(root, campaign, *, resume=False, report=True):
    return run_controller(root=root, identity={"fixture": "controller-v1"},
                          initialize=campaign.initialize, resume=resume, generate_report=report)


def test_full_twenty_step_original_controller_negative_best_and_complete_export(tmp_path, upstream, monkeypatch):
    campaign = Campaign(upstream, monkeypatch)
    result = launch(tmp_path / "run", campaign)
    search = result["search"]
    assert search["terminal"] and search["completed_steps"] == search["configured_steps"] == 20
    assert search["stop_reason"] == "original_steps_completed"
    assert search["best"]["native_metric"] == -0.1
    assert result["report"]["state"] == "completed"
    assert len(campaign.provider.requests) == 41
    assert campaign.provider.requests[-1]["max_tokens"] == 4096
    assert all(row["model"] == MODEL for row in campaign.provider.requests)
    snapshot = unseal(json.loads((tmp_path / "run" / search["export"]["path"]).read_text()))
    assert len(snapshot["journal"]) == 20 and snapshot["search_call_end"] == 40
    assert {n.stage_name for n in campaign.runtime.session.agent.journal.nodes} == {"draft", "debug", "improve"}
    inventory = snapshot["artifacts"]["inventory"]
    assert sum(path.endswith("workspace/cache.pkl") for path in inventory) == 20
    assert sum(path.endswith("workspace/helper.py") for path in inventory) == 20
    links = [row for path, row in inventory.items() if path.endswith("external-link")]
    assert all(row == {"kind": "symlink", "target": "/not-readable-outside-fixture"} for row in links)
    verify_snapshot(tmp_path / "run", search["export"], campaign.runtime,
                    {"fixture": "controller-v1", "generate_report": True})


def test_report_replay_after_caps_does_not_reopen_search_or_mutate_search_final(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=41, interrupt_report=True)
    first = launch(root, campaign)
    assert first["report"]["state"] == "interrupted"
    digest = hashlib.sha256((root / "search_final.json").read_bytes()).hexdigest()
    calls_before = len(campaign.provider.requests)
    campaign.clock[0] += 10001
    monkeypatch.setattr(upstream.Agent, "step", lambda *a, **k: pytest.fail("terminal search restarted"))
    monkeypatch.setattr(AideOperationBudget, "before_new_search", lambda *a: pytest.fail("terminal got search gate"))
    result = launch(root, campaign, resume=True)
    assert result["report"]["state"] == "completed"
    assert result["search"] == first["search"]
    assert hashlib.sha256((root / "search_final.json").read_bytes()).hexdigest() == digest
    assert len(campaign.provider.requests) == calls_before == 41


def test_expired_budgets_still_append_durable_review_before_terminal(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5], interrupt_reviewed=True)
    result = launch(root, campaign, report=False)
    assert not result["search"]["terminal"] and result["search"]["completed_steps"] == 0
    assert not (root / "search_final.json").exists()
    campaign.clock[0] += 10001
    resumed = launch(root, campaign, resume=True, report=False)
    assert resumed["search"]["terminal"] and resumed["search"]["completed_steps"] == 1
    assert resumed["search"]["best"]["native_metric"] == -0.5
    assert len(campaign.provider.requests) == 2


def test_higher_scored_pending_does_not_enter_original_journal_best(tmp_path, upstream, monkeypatch):
    campaign = Campaign(upstream, monkeypatch, calls=3, scores=[-0.5, 99])
    result = launch(tmp_path / "run", campaign, report=False)
    assert result["search"]["terminal"] and result["search"]["completed_steps"] == 1
    assert result["search"]["best"]["native_metric"] == -0.5
    pending = result["search"]["pending"]
    assert not pending["selected_as_best"]
    assert pending["evaluation"]["metrics"]["aggregate_improvement"] == 99
    assert len(campaign.provider.requests) == 3


def test_no_good_nodes_is_none_not_a_fabricated_zero_score(tmp_path, upstream, monkeypatch):
    campaign = Campaign(upstream, monkeypatch, scores=[None])
    result = launch(tmp_path / "run", campaign, report=False)
    assert result["search"]["completed_steps"] == 20
    assert result["search"]["best"] is None


def test_score_budget_stop_keeps_original_steps_denominator(tmp_path, upstream, monkeypatch):
    campaign = Campaign(upstream, monkeypatch, scores=[-0.2], score_cap=2)
    result = launch(tmp_path / "run", campaign, report=False)
    assert result["search"]["completed_steps"] == 2
    assert result["search"]["configured_steps"] == 20
    assert result["search"]["stop_reason"] == "score_attempt_budget_exhausted"
    assert len(campaign.provider.requests) == 4


def test_terminal_missing_checkpoint_cannot_initialize_new_session(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5])
    launch(root, campaign, report=False)
    (root / "checkpoint.json").unlink()
    monkeypatch.setattr(campaign, "initialize", lambda root: pytest.fail("initialized after terminal checkpoint loss"))
    with pytest.raises(ValueError, match="lost its checkpoint"):
        launch(root, campaign, resume=True, report=False)


def test_explicit_resume_identity_and_terminal_artifact_integrity(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5])
    first = launch(root, campaign, report=False)
    with pytest.raises(ValueError, match="explicit resume"):
        launch(root, campaign, report=False)
    with pytest.raises(ValueError, match="identity changed"):
        launch(root, campaign, resume=True, report=True)
    raw = root / first["search"]["best"]["raw_workspace"] / "cache.pkl"
    raw.write_bytes(b"changed cache")
    with pytest.raises(ValueError, match="artifact/workspace changed"):
        launch(root, campaign, resume=True, report=False)


@pytest.mark.parametrize("suffix", ["request", "response"])
@pytest.mark.parametrize("corruption", ["missing", "resealed_tamper"])
def test_export_rejects_broken_original_call_binding_before_report(
        tmp_path, upstream, monkeypatch, suffix, corruption):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5])
    original = controller.export_snapshot

    def corrupt_then_export(root, runtime, identity):
        path = root / "calls" / f"000000.{suffix}.json"
        if corruption == "missing":
            path.unlink()
        else:
            value = unseal(json.loads(path.read_text()))
            value["tampered_after_ledger_acceptance"] = True
            atomic_json(path, seal(value))
        return original(root, runtime, identity)

    monkeypatch.setattr(controller, "export_snapshot", corrupt_then_export)
    with pytest.raises(ValueError):
        launch(root, campaign)
    assert len(campaign.provider.requests) == 2
    assert not (root / "search_final.json").exists()
    assert not (root / "report_state.json").exists()


def test_nonterminal_missing_checkpoint_cannot_initialize_new_session(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5], interrupt_reviewed=True)
    result = launch(root, campaign, report=False)
    assert not result["search"]["terminal"]
    (root / "checkpoint.json").unlink()
    monkeypatch.setattr(campaign, "initialize", lambda root: pytest.fail("reinitialized missing history"))
    with pytest.raises(ValueError, match="checkpoint missing after recorded activity"):
        launch(root, campaign, resume=True, report=False)
    assert len(campaign.provider.requests) == 2


@pytest.mark.parametrize("activity", ["provider", "execution"])
def test_abrupt_crash_without_final_still_requires_existing_checkpoint(tmp_path, activity):
    root = tmp_path / "run"
    root.mkdir()
    atomic_json(root / "identity.json", seal({"fixture": "crash", "generate_report": False}))
    if activity == "provider":
        atomic_json(root / "calls/ledger.json", seal({"attempts": [{"state": "completed"}]}))
    else:
        atomic_json(root / "artifacts/node1/execution_started.json", seal({"node_id": "node1"}))
    with pytest.raises(ValueError, match=f"checkpoint missing after {activity} activity"):
        run_controller(root=root, identity={"fixture": "crash"}, resume=True, generate_report=False,
                       initialize=lambda root: pytest.fail("initialized despite previous activity"))


def test_empty_initialization_can_resume_before_first_checkpoint(tmp_path, upstream, monkeypatch):
    root = tmp_path / "run"
    root.mkdir()
    atomic_json(root / "identity.json", seal({"fixture": "controller-v1", "generate_report": False}))
    atomic_json(root / "calls/ledger.json", seal({
        "schema_version": "aide-provider-ledger-v1", "started_at": 1000.0, "attempts": [],
        "identity": {"model": MODEL, "max_calls": 2, "max_output_tokens": 10000,
                     "wall_seconds": 10000, "per_call_output_tokens": 5000}}))
    campaign = Campaign(upstream, monkeypatch, calls=2, scores=[-0.5])
    result = launch(root, campaign, resume=True, report=False)
    assert result["search"]["terminal"] and result["search"]["completed_steps"] == 1
    assert len(campaign.provider.requests) == 2
