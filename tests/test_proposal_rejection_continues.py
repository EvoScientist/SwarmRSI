"""A semantic proposal refusal retires that proposal; the search continues.

Synthetic invariant tests only: no provider, GPU, or benchmark score calls.
"""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from swarmrsi import naturebench_cli as cli, naturebench_session as session
from swarmrsi.campaign import VSearchCoordinator
from swarmrsi.contracts import ProposalRejected, ScientificFamily
from test_brs_strategy import ResearchProposer, brs_campaign, brs, episode_of, run_steps, _eclc_params
from test_naturebench_entrypoint import inputs  # noqa: F401


def session_step(loop):
    """The production runner's queue rule, including retired proposals."""
    unavailable = loop.unavailable_action_sha256s()
    if not any(item.candidate.action_sha256 not in unavailable for item in loop.actions.values()):
        loop.expand()
    try:
        return loop.step()
    except ProposalRejected as exc:
        return loop.record_proposal_rejection(exc, stage="select")


def off_family_once(loop):
    original = loop.proposer.propose

    def changed(**request):
        loop.proposer.propose = original
        proposals, usage = original(**request)
        p = proposals[0]
        family = ScientificFamily("raw", "changed-core", "direct", "public metric")
        return [replace(p, candidate=replace(p.candidate, action_sha256="", family=family))], usage

    loop.proposer.propose = changed


def rejected_after_first_gain(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .8], parameters=_eclc_params())
    first, = run_steps(loop, 1)
    assert brs(loop).consolidation_lineages[first.action_id]["status"] == "armed"
    off_family_once(loop)
    executions = loop.executor.calls
    rejection = session_step(loop)
    assert isinstance(rejection, dict) and rejection["code"] == "consolidation_family_mismatch"
    assert loop.executor.calls == executions
    return loop, first, rejection


def test_consolidation_family_mismatch_retires_lane_and_search_continues(tmp_path):
    loop, first, rejection = rejected_after_first_gain(tmp_path)
    algorithm = brs(loop)
    rejected = loop.actions[rejection["action_id"]].candidate
    lane = algorithm.consolidation_lineages[first.action_id]
    assert lane["status"] == "retired_rejected"
    assert rejected.id not in algorithm.consolidation_by_action
    assert rejection["action_sha256"] == rejected.action_sha256 in loop.rejected_actions
    assert rejected.action_sha256 not in loop.accounting.attempted_actions
    assert loop.consecutive_proposal_rejections == 1

    row = session_step(loop)
    assert row.action_id != rejected.id
    assert episode_of(loop, row.action_id).mode != "evidence_consolidation"
    assert loop.consecutive_proposal_rejections == 0
    assert rejected.id not in {step.action_id for step in loop.steps}
    assert algorithm.consolidation_lineages[first.action_id]["status"] == "retired_rejected"
    events = [event.event_type for event in loop.view.events]
    assert "search.proposal_rejected" in events
    assert "search.frontier.consolidation_retired" in events
    assert loop.summary()["proposal_rejections"] == [rejection]


def test_resume_keeps_rejected_action_excluded(tmp_path):
    loop, first, rejection = rejected_after_first_gain(tmp_path)
    saved = json.loads(json.dumps(loop.checkpoint(), allow_nan=False))
    fresh = brs_campaign(tmp_path / "restored", [.6, .8], parameters=_eclc_params())
    restored = VSearchCoordinator.restore(strategy=fresh.strategy, proposer=fresh.proposer,
                                          executor=fresh.executor, payload=saved)
    assert json.loads(json.dumps(restored.checkpoint(), allow_nan=False)) == saved
    rejected_sha = loop.actions[rejection["action_id"]].candidate.action_sha256
    assert rejected_sha in restored.unavailable_action_sha256s()
    assert restored.proposal_rejections == [rejection]
    assert restored.consecutive_proposal_rejections == 1
    assert brs(restored).consolidation_lineages[first.action_id]["status"] == "retired_rejected"
    # The restored proposer restarts its id counter; keep ids distinct from the checkpoint.
    fresh.proposer.index = 100
    row = session_step(restored)
    assert row.action_id != rejection["action_id"]
    assert fresh.executor.calls == 1
    assert episode_of(restored, row.action_id).mode != "evidence_consolidation"


def test_expand_hypothesis_mismatch_registers_nothing_and_is_recorded(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .8], parameters=_eclc_params())
    run_steps(loop, 1)
    honest, loop.proposer = loop.proposer, ResearchProposer(substitute_hypothesis=True)
    loop.proposer.index = 50
    before = dict(loop.actions)
    with pytest.raises(ProposalRejected) as caught:
        loop.expand()
    assert caught.value.record["code"] == "controller_hypothesis_mismatch"
    assert loop.actions == before
    record = loop.record_proposal_rejection(caught.value, stage="expand")
    assert record["stage"] == "expand" and record["action_sha256"] is None
    loop.proposer = honest
    row = session_step(loop)
    assert row.action_id in loop.actions and loop.consecutive_proposal_rejections == 0


def test_session_stops_after_configured_consecutive_rejections(inputs, monkeypatch):
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    raw["screening_execution_limits"] = {"max_consecutive_proposal_rejections": 2}
    inputs.profile.write_text(json.dumps(raw))
    cli.configure(inputs)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic context")
    monkeypatch.setattr(session, "build_search_models", lambda *a: (
        SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    from test_naturebench_entrypoint import synthetic_root
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)

    def rejected(loop):
        loop.accounting.model_calls += 1
        raise ProposalRejected("synthetic semantic refusal", code="fixture_refusal", action_id="fixture")

    monkeypatch.setattr(session.VSearchCoordinator, "expand", rejected)
    assert cli.execute(inputs) == 0
    final = json.loads((inputs.run_dir / "final.json").read_text())
    checkpoint = json.loads((inputs.run_dir / "checkpoint.json").read_text())
    assert final["stop_reason"] == "search_proposal_rejected"
    assert final["stop_details"]["incumbent_retained"] is True
    assert final["accounting"]["model_calls"] == 2
    assert [r["stage"] for r in final["stop_details"]["proposal_rejections"]] == ["expand", "expand"]
    assert checkpoint["proposal_rejections"] == final["stop_details"]["proposal_rejections"]
    assert not final["steps"]
