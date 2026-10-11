"""Synthetic invariant tests only: no provider, GPU, or benchmark score calls."""
from dataclasses import replace
import json
from types import SimpleNamespace

import pytest

from evolvcoder import naturebench_cli as cli, naturebench_session as session
from evolvcoder.contracts import ActionOperator, ProposalRejected, ScientificFamily
from evolvcoder.campaign import VSearchCoordinator
from test_brs_strategy import brs_campaign, brs, run_steps, _eclc_params
from test_naturebench_entrypoint import inputs, synthetic_root, forbid  # noqa: F401


@pytest.mark.parametrize("kind", ["family", "hypothesis", "shape"])
@pytest.mark.parametrize("advanced", [False, True])
def test_eclc_refuses_semantic_changes_without_polluting_evidence(tmp_path, kind, advanced):
    loop = brs_campaign(tmp_path, [.4, .6, .8], parameters=_eclc_params())
    run_steps(loop, 2 if advanced else 1)
    algorithm = brs(loop)
    before = {k: v for k, v in algorithm.hypotheses.records.items()}
    best = loop.summary()["best_node_id"]
    calls, executions = loop.accounting.model_calls, loop.executor.calls
    original = loop.proposer.propose

    def changed(**request):
        proposals, usage = original(**request)
        p = proposals[0]
        change = {"family": ScientificFamily("raw", "changed-core", "direct", "public metric")}
        if kind == "hypothesis":
            change = {"hypothesis_id": "changed-hypothesis"}
        if kind == "shape":
            change = {"operator": ActionOperator.BRANCH_PROGRAM}
        # This is a freshly generated malformed fixture, not a runtime rewrite.
        candidate = replace(p.candidate, action_sha256="", **change)
        return [replace(p, candidate=candidate)], usage

    loop.proposer.propose = changed
    with pytest.raises(ProposalRejected) as caught:
        run_steps(loop, 1)
    assert caught.value.record["action_id"]
    assert loop.executor.calls == executions
    assert loop.accounting.model_calls == calls + 1
    assert loop.summary()["best_node_id"] == best
    assert algorithm.hypotheses.records == before
    assert algorithm.pending_episode_id is None
    assert caught.value.record["action_id"] not in algorithm.consolidation_by_action
    saved = json.loads(json.dumps(loop.checkpoint(), allow_nan=False))
    fresh = brs_campaign(tmp_path / "restored", [.8], parameters=_eclc_params())
    restored = VSearchCoordinator.restore(strategy=fresh.strategy, proposer=fresh.proposer,
                                          executor=fresh.executor, payload=saved)
    assert json.loads(json.dumps(restored.checkpoint(), allow_nan=False)) == saved
    assert fresh.executor.calls == 0
    assert restored.accounting.model_calls == calls + 1
    assert brs(restored).hypotheses.records == before


@pytest.mark.parametrize("boundary", ["expand", "step"])
@pytest.mark.parametrize("valid_root", [True, False])
def test_session_seals_rejection_with_charged_usage_and_no_false_incumbent(
        inputs, monkeypatch, boundary, valid_root):
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    inputs.profile.write_text(json.dumps(raw))
    cli.configure(inputs)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic context")
    monkeypatch.setattr(session, "build_search_models", lambda *a: (
        SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))

    def root(**kw):
        value = synthetic_root(**kw)
        return value if valid_root else (None, False, value[2], None, value[4])

    monkeypatch.setattr(session, "evaluate_root", root)

    def rejected(loop):
        loop.accounting.model_calls += 1
        loop.accounting.input_tokens += 300
        loop.accounting.output_tokens += 200
        raise ProposalRejected("synthetic semantic refusal", code="fixture_refusal", action_id="fixture")

    if boundary == "step":
        monkeypatch.setattr(session.VSearchCoordinator, "expand", lambda _: 1)
    monkeypatch.setattr(session.VSearchCoordinator, boundary, rejected)
    if valid_root:
        assert cli.execute(inputs) == 0
    else:
        with pytest.raises(RuntimeError, match="session returned 1"):
            cli.execute(inputs)
    final = json.loads((inputs.run_dir / "final.json").read_text())
    checkpoint = json.loads((inputs.run_dir / "checkpoint.json").read_text())
    assert final["stop_details"]["incumbent_retained"] is valid_root
    assert final["best_node_id"] == ("root" if valid_root else None)
    # Each refusal is retired and the search continues; only the default run of
    # three consecutive refusals seals the session.
    assert final["accounting"]["model_calls"] == checkpoint["accounting"]["model_calls"] == 3
    assert final["accounting"]["output_tokens"] == 600
    assert final["stop_details"]["proposal_rejection"]["code"] == "fixture_refusal"
    assert [r["code"] for r in final["stop_details"]["proposal_rejections"]] == ["fixture_refusal"] * 3
    assert checkpoint["consecutive_proposal_rejections"] == 3
    assert not final["steps"]
    # The installed entrypoint verifies the complete artifact/checkpoint record
    # before reuse. Rejected work cannot silently call the provider again.
    monkeypatch.setattr(session, "run_session", forbid)
    monkeypatch.setattr(session, "build_search_models", forbid)
    monkeypatch.setattr(session, "evaluate_root", forbid)
    inputs.credential_file.unlink()
    assert cli.execute(inputs) == (0 if valid_root else 1)
    (inputs.run_dir / "root/run.py").write_text("# tampered artifact\n")
    with pytest.raises(ValueError, match="program artifact mismatch"):
        cli.execute(inputs)


def test_session_does_not_swallow_integrity_errors(inputs, monkeypatch):
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    inputs.profile.write_text(json.dumps(raw))
    cli.configure(inputs)
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic context")
    monkeypatch.setattr(session, "build_search_models", lambda *a: (
        SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))

    def invalid(_):
        raise ValueError("proposal does not bind an existing executable parent")

    monkeypatch.setattr(session.VSearchCoordinator, "expand", invalid)
    with pytest.raises(ValueError, match="existing executable parent"):
        session.run_session(inputs)
    assert not (inputs.run_dir / "final.json").exists()
