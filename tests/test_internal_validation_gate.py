"""A candidate's own public-label estimate can save an official evaluation.

No model, no evaluator: the gate is a pure comparison, the lane is a pure
classification. Rejection spends nothing and refutes nothing.
"""
import json
from pathlib import Path

import pytest
from test_naturebench_entrypoint import inputs  # noqa: F401

from evolvcoder.coder_models import CoderExitStatus
from evolvcoder.naturebench_session import internal_validation_gate, read_internal_validation
from evolvcoder.routing import RecoveryRoute, classify_recovery_route
from evolvcoder.coder_models import CoderOutcome, CoderVersion


def coder_outcome(status):
    outcome = CoderOutcome(
        status=status, version=CoderVersion.C3, config_sha256="c" * 64, action_id="a0000",
        parent_sha256="p" * 64, parent_workspace_sha256="w" * 64, final_sha256="f" * 64,
        final_workspace_sha256="x" * 64, model_calls=1, tool_calls=1, input_tokens=1, output_tokens=1,
        wall_seconds=0.0, changed_paths=["run.py"], implementation_real=status == CoderExitStatus.SUCCESS,
        parent_preserved=True, reason="synthetic", trajectory_sha256="t" * 64)
    outcome.seal()
    return outcome


def artifact(tmp_path, name, record):
    root = tmp_path / name
    root.mkdir()
    if record is not None:
        (root / "internal_validation.json").write_text(json.dumps(record))
    return root


def estimate(value, *, direction="maximize", metric="aggregate_improvement"):
    return {"metric_name": metric, "direction": direction, "estimate": value, "protocol": "5-fold CV on public train"}


@pytest.mark.parametrize("direction,parent_score,child_score", [
    ("maximize", 0.9, 0.8), ("minimize", 0.1, 0.2)])
def test_controller_only_does_not_reject_unattested_child(tmp_path, direction, parent_score, child_score):
    from types import SimpleNamespace
    from evolvcoder.brs_strategy import BidirectionalResearchStrategy

    parent = artifact(tmp_path, "parent", estimate(parent_score, direction=direction))
    child = artifact(tmp_path, "child", estimate(child_score, direction=direction))
    assert internal_validation_gate(child, parent)["rejected"]
    gate = internal_validation_gate(child, parent, policy="controller_only")
    assert not gate["rejected"]
    assert gate["evidence_authority"] == "candidate_unattested"
    evidence = BidirectionalResearchStrategy._internal_validation_evidence(
        SimpleNamespace(internal_validation=gate))
    assert not evidence["comparable"]


def test_unknown_validation_policy_fails_closed(tmp_path):
    with pytest.raises(ValueError, match="unknown internal validation policy"):
        internal_validation_gate(tmp_path, None, policy="typo")


def test_controller_policy_reaches_planner_and_candidate_context(inputs):
    from types import SimpleNamespace
    from evolvcoder import naturebench_cli as cli, naturebench_session as session

    cli.configure(inputs)
    root = inputs.tasks_root / inputs.task_id
    context = session.public_task_context(root, internal_validation_policy="controller_only")
    assert "optional, unattested diagnostic" in context
    assert "may skip the official evaluation" not in context
    planner = session.Planner(SimpleNamespace(), 0, root, ("main",), (),
                              internal_validation_policy="controller_only")
    assert "optional, unattested diagnostic" in planner.task_context
    legacy = session.public_task_context(root)
    assert "may skip the official evaluation" in legacy
    with pytest.raises(ValueError, match="unknown internal validation policy"):
        session.public_task_context(root, internal_validation_policy="typo")


def test_gate_only_rejects_a_comparable_worse_estimate(tmp_path):
    parent = artifact(tmp_path, "parent", estimate(0.30))
    assert internal_validation_gate(artifact(tmp_path, "none", None), parent) is None
    first = internal_validation_gate(artifact(tmp_path, "first", estimate(0.10)), artifact(tmp_path, "root", None))
    assert first["rejected"] is False and first["reason"] == "no_comparable_parent_estimate"
    worse = internal_validation_gate(artifact(tmp_path, "worse", estimate(0.25)), parent)
    assert worse["rejected"] is True and worse["shortfall"] == pytest.approx(0.05)
    equal = internal_validation_gate(artifact(tmp_path, "equal", estimate(0.30)), parent)
    assert equal["rejected"] is False
    better = internal_validation_gate(artifact(tmp_path, "better", estimate(0.31)), parent)
    assert better["rejected"] is False
    tolerated = internal_validation_gate(artifact(tmp_path, "tolerated", estimate(0.25)), parent, tolerance=0.1)
    assert tolerated["rejected"] is False


def test_gate_respects_direction_and_refuses_incomparable_estimates(tmp_path):
    parent = artifact(tmp_path, "parent", estimate(0.20, direction="minimize", metric="mse"))
    higher = internal_validation_gate(artifact(tmp_path, "higher", estimate(0.25, direction="minimize", metric="mse")), parent)
    assert higher["rejected"] is True
    lower = internal_validation_gate(artifact(tmp_path, "lower", estimate(0.15, direction="minimize", metric="mse")), parent)
    assert lower["rejected"] is False
    other = internal_validation_gate(artifact(tmp_path, "other", estimate(0.9, metric="auc")), parent)
    assert other["rejected"] is False and other["reason"] == "no_comparable_parent_estimate"


@pytest.mark.parametrize("record", [
    "not json", {"metric_name": "m"}, {"metric_name": "m", "direction": "sideways", "estimate": 1.0},
    {"metric_name": "m", "direction": "maximize", "estimate": "1.0"},
    {"metric_name": "m", "direction": "maximize", "estimate": float("nan")}, [1, 2],
])
def test_malformed_estimates_are_ignored_not_trusted(tmp_path, record):
    root = tmp_path / "bad"
    root.mkdir()
    text = record if isinstance(record, str) else json.dumps(record)
    (root / "internal_validation.json").write_text(text)
    assert read_internal_validation(root) is None
    assert internal_validation_gate(root, tmp_path / "missing") is None


def test_rejection_lane_spends_nothing_and_never_repairs():
    route = classify_recovery_route(coder_outcome(CoderExitStatus.SUCCESS), artifact_valid=False, native_metric=None,
                                    parent_metric=0.1, internal_validation_rejected=True)
    assert route.route == "discard_underperforming_candidate"
    assert set(route.reason_codes) == {"internal_validation_below_parent", "no_official_evaluation_spent"}
    # A failed implementation is a failure whatever its self-estimate says.
    failed = classify_recovery_route(coder_outcome(CoderExitStatus.IMPLEMENTATION_FAILED), artifact_valid=False,
                                     native_metric=None, parent_metric=0.1, internal_validation_rejected=True)
    assert failed.route == "repair_implementation"
    assert RecoveryRoute("discard_underperforming_candidate", ("x",), "a" * 64).route_sha256


def fake_coder(proposal, parent, record):
    """A coder that writes one candidate program plus the given self-estimate."""
    from evolvcoder.coder_models import CoderOutcome, CoderVersion
    from evolvcoder.campaign import artifact_workspace_sha256
    from test_evolvcoder_incremental_feedback import file_sha

    class FakeCoder:
        def __init__(self, config, model):
            self.config = config

        def run(self, source_root, cell, **kwargs):
            artifact = cell / "artifact"
            artifact.mkdir(parents=True)
            (artifact / "run.py").write_text("print('candidate')\n")
            (artifact / "internal_validation.json").write_text(json.dumps(record))
            outcome = CoderOutcome(
                status=CoderExitStatus.SUCCESS, version=CoderVersion.C3, config_sha256=self.config.config_sha256,
                action_id=proposal.candidate.id, parent_sha256=parent.executable_sha256,
                parent_workspace_sha256=parent.workspace_sha256, final_sha256=file_sha(artifact / "run.py"),
                final_workspace_sha256=artifact_workspace_sha256(artifact), model_calls=1, tool_calls=1,
                input_tokens=1, output_tokens=1, wall_seconds=0.0, changed_paths=["run.py"],
                implementation_real=True, parent_preserved=True, reason="synthetic_success",
                trajectory_sha256="t" * 64)
            outcome.seal()
            return outcome

    return FakeCoder


def test_executor_skips_the_official_evaluator_for_a_self_rejected_candidate(inputs, monkeypatch, tmp_path):
    """The hook in the session executor: no evaluator subprocess, no score, a gate record."""
    from types import SimpleNamespace
    from evolvcoder import naturebench_cli as cli, naturebench_session as session
    from evolvcoder.coder_models import CoderOutcome, CoderVersion
    from evolvcoder.campaign import artifact_workspace_sha256
    from evolvcoder.session_limits import SessionLimits
    from evolvcoder.configuration import Limits
    from test_evolvcoder_incremental_feedback import campaign, file_sha
    from test_session_limits import ACTION, usage

    cli.configure(inputs)
    loop = campaign(inputs.profile.parent / "gate-case", scores=[])
    loop.expand()
    proposal = next(iter(loop.actions.values()))
    parent = loop.nodes["root"]
    (parent.artifact / "internal_validation.json").write_text(json.dumps(estimate(0.30)))

    monkeypatch.setattr(session, "ContractBoundedCoder", fake_coder(proposal, parent, estimate(0.25)))
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {
        "available": True, "wrapper": ("/synthetic/bwrap",), "rewrites": (), "environment": ()})
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **k: pytest.fail("official evaluator must not run"))
    executor = session.Executor(SimpleNamespace(model="synthetic"), inputs.run_dir, 0,
                                inputs.tasks_root / inputs.task_id, ("main",), ACTION)
    executor.set_remaining(usage(), SessionLimits(Limits()).to_legacy())
    execution = executor.execute(proposal=proposal, parent=parent, step=0)
    assert execution.score_evaluations == 0 and execution.native_metric is None and execution.artifact_valid is False
    assert execution.internal_validation["rejected"] is True
    assert execution.evaluation_summary["public_diagnostic"]["category"] == "internal_validation_rejected"
    assert execution.evaluation_summary["score_bearing_evaluation_attempted"] is False
    cell = inputs.run_dir / "cells" / f"000_{proposal.candidate.id}"
    assert json.loads((cell / "internal_validation_gate.json").read_text())["shortfall"] == pytest.approx(0.05)


@pytest.mark.parametrize("controller_only", [False, True])
def test_the_gate_can_be_ablated_off_and_then_everything_is_scored(tmp_path, monkeypatch, inputs, controller_only):
    """Ablation 4: with the gate disabled a worse self-estimate still reaches the evaluator."""
    from types import SimpleNamespace
    from evolvcoder import naturebench_cli as cli, naturebench_session as session
    from evolvcoder.session_limits import SessionLimits
    from evolvcoder.configuration import Limits
    from test_evolvcoder_incremental_feedback import campaign
    from test_session_limits import ACTION, usage

    cli.configure(inputs)
    loop = campaign(inputs.profile.parent / "ablation-case", scores=[])
    loop.expand()
    proposal = next(iter(loop.actions.values()))
    parent = loop.nodes["root"]
    (parent.artifact / "internal_validation.json").write_text(json.dumps(estimate(0.30)))
    monkeypatch.setattr(session, "ContractBoundedCoder", fake_coder(proposal, parent, estimate(0.25)))
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {
        "available": True, "wrapper": ("/synthetic/bwrap",), "rewrites": (), "environment": ()})
    invoked = []
    monkeypatch.setattr(session.subprocess, "run", lambda *a, **k: invoked.append(a) or SimpleNamespace(
        returncode=1, stdout="", stderr="synthetic evaluator stub"))
    executor = session.Executor(SimpleNamespace(model="synthetic"), inputs.run_dir, 0,
                                inputs.tasks_root / inputs.task_id, ("main",),
                                {**ACTION, "internal_validation_enabled": controller_only,
                                 "internal_validation_policy": "controller_only" if controller_only else "legacy"})
    executor.set_remaining(usage(), SessionLimits(Limits()).to_legacy())
    execution = executor.execute(proposal=proposal, parent=parent, step=0)
    # The gate was not consulted and the official evaluator was actually invoked.
    assert invoked
    gate_path = inputs.run_dir / "cells" / f"000_{proposal.candidate.id}" / "internal_validation_gate.json"
    if controller_only:
        assert execution.internal_validation["rejected"] is False
        assert execution.internal_validation["evidence_authority"] == "candidate_unattested"
        assert json.loads(gate_path.read_text()) == execution.internal_validation
    else:
        assert execution.internal_validation is None
        assert not gate_path.exists()


def test_candidate_cannot_self_attest_controller_evidence(tmp_path):
    claimed = {**estimate(0.1), "evidence_authority": "controller",
               "comparable": True, "rejected": True, "shortfall": 1.0}
    child = artifact(tmp_path, "claimed", claimed)
    parent = artifact(tmp_path, "parent", estimate(0.9))
    gate = internal_validation_gate(child, parent, policy="controller_only")
    assert gate["evidence_authority"] == "candidate_unattested"
    assert not gate["rejected"]
    assert "shortfall" not in gate
    assert "evidence_authority" not in gate["candidate"]


def test_the_gate_artifact_is_inside_the_action_contract(inputs):
    """R35 regression: every candidate that obeyed the instruction was a violation.

    The standing instruction asks the coder for internal_validation.json, so the
    contract the planner builds must permit exactly that path.
    """
    from evolvcoder import naturebench_session as session
    from evolvcoder.coder_models import ScientificActionContract

    assert session.INTERNAL_VALIDATION_NAME in session.CANDIDATE_WRITE_PATHS
    # The instruction that asks for it and the contract that permits it agree.
    context = session.public_task_context(inputs.tasks_root / inputs.task_id, execution_timeout_seconds=60)
    assert session.INTERNAL_VALIDATION_NAME in context
    # A candidate that writes it is inside the contract; an unrelated path is not.
    contract = ScientificActionContract(action_id="a", description="d",
                                        allowed_write_paths=session.CANDIDATE_WRITE_PATHS,
                                        protected_paths=("TASK_CONTEXT.md",))
    assert session.INTERNAL_VALIDATION_NAME not in contract.protected_paths
    from evolvcoder.coder_engine import ContractBoundedCoder
    allowed = ContractBoundedCoder._write_allowed
    probe = type("C", (), {"config": type("K", (), {"action": contract})()})()
    assert allowed(probe, session.INTERNAL_VALIDATION_NAME) is True
    assert allowed(probe, "run.py") is True
    assert allowed(probe, "notes.json") is False


def test_the_instruction_follows_the_switch_and_is_bounded(inputs):
    """R35: the unbounded wording put k-fold CV inside run.py and exhausted wall budgets."""
    from evolvcoder import naturebench_session as session
    task_root = inputs.tasks_root / inputs.task_id
    on = session.public_task_context(task_root, execution_timeout_seconds=60, internal_validation=True)
    off = session.public_task_context(task_root, execution_timeout_seconds=60, internal_validation=False)
    assert session.INTERNAL_VALIDATION_NAME in on and session.INTERNAL_VALIDATION_NAME not in off
    assert "## Internal validation" not in off
    for phrase in ("ONCE", "single hold-out split", "do not add cross-validation", "omit the file"):
        assert phrase in on
    # The planner threads the same switch into the context it gives the coder.
    from types import SimpleNamespace
    planner = session.Planner(SimpleNamespace(native_tools=True), 0, task_root, ("main",), (),
                              execution_timeout_seconds=60, internal_validation=False)
    assert session.INTERNAL_VALIDATION_NAME not in planner.task_context


def test_operational_allowlist_carries_an_egress_proxy_but_candidates_do_not(monkeypatch, tmp_path):
    """A host with no direct route needs the proxy; the candidate still gets none."""
    from evolvcoder.naturebench_cli import OPERATIONAL_ENVIRONMENT_KEYS
    for name in ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy"):
        assert name in OPERATIONAL_ENVIRONMENT_KEYS, name
    # A proxy variable names a route, so it may not be used to carry a credential.
    from evolvcoder.naturebench_cli import validate_credential_env
    with pytest.raises(ValueError):
        validate_credential_env("HTTPS_PROXY")
    # The candidate's environment drops every proxy name even before bwrap unshares the network.
    monkeypatch.setenv("HTTPS_PROXY", "http://127.0.0.1:3128")
    monkeypatch.setenv("no_proxy", "localhost")
    proxy_names = {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "no_proxy", "all_proxy"}
    import os
    kept = {k: v for k, v in os.environ.items()
            if not k.endswith("_API_KEY") and not k.endswith("_TOKEN")
            and k not in {"EVAL_SERVICE_URL"} and k not in proxy_names}
    assert not (set(kept) & proxy_names)


def test_submission_fit_policy_reaches_planner_and_candidate_context(inputs):
    """Opt-in contract: the coder is told the terminal re-execution must fit on every row."""
    from types import SimpleNamespace
    from evolvcoder import naturebench_cli as cli, naturebench_session as session

    cli.configure(inputs)
    root = inputs.tasks_root / inputs.task_id
    context = session.public_task_context(root, submission_fit_policy="full_refit")
    assert "## Submission contract" in context
    assert f"`{session.SUBMISSION_FIT_ENV}=full`" in context
    assert "fit the model that writes submission.csv on EVERY training row" in context
    assert '"submission_fit"' in context
    planner = session.Planner(SimpleNamespace(), 0, root, ("main",), (), submission_fit_policy="full_refit")
    assert "## Submission contract" in planner.task_context
    legacy = session.public_task_context(root)
    assert "## Submission contract" not in legacy and session.SUBMISSION_FIT_ENV not in legacy
    with pytest.raises(ValueError, match="unknown submission fit policy"):
        session.public_task_context(root, submission_fit_policy="typo")
