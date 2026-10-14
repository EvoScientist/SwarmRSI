"""Offline controller counterexamples, not reruns of NatureBench methods."""
from __future__ import annotations

import hashlib
import shutil
from dataclasses import replace

import pytest

from swarmrsi import ContractBoundedCoder, CoderExitStatus, CoderVersion, artifact_workspace_sha256
from swarmrsi.routing import classify_recovery_route
from test_swarmrsi_coder import ScriptedModel, config, workspace


class RecordingCoder(ContractBoundedCoder):
    """Observe gate reachability without executing data or external services."""

    def __init__(self, *args, accepted=True, **kwargs):
        super().__init__(*args, **kwargs)
        self.accepted = accepted
        self.validated_workspaces = []

    def _validate(self, root, events, *, deadline=None):
        self.validated_workspaces.append(self._workspace_sha(self._manifest(root)))
        return [{"tier": "smoke", "accepted": self.accepted,
                 "returncode": 0 if self.accepted else 1, "blocking": True}]


def repair_fixture(tmp_path):
    root = workspace(tmp_path)
    seed = tmp_path / "failed_seed"
    shutil.copytree(root, seed)
    (seed / "solution.py").write_text(
        "from helper import VALUE\n\ndef load_data():\n    return []\n\n"
        "def predict(x):\n    return VALUE\n\nprint(predict(None))\n"
    )
    (seed / "helper.py").write_text("VALUE = 'bad'\n")
    base = config(root, calls=10)
    cfg = replace(base, action=replace(
        base.action, allowed_write_paths=("solution.py", "helper.py"),
        required_call_names=(), required_source_patterns=(),
    ), config_sha256="")
    return root, seed, cfg


def run_seeded(coder, root, seed, out):
    return coder.run(root, out, diagnostic_seed_root=seed,
                     expected_diagnostic_workspace_sha256=artifact_workspace_sha256(seed))


def assert_failed_retention(outcome, root, seed, out):
    assert outcome.status != CoderExitStatus.SUCCESS
    assert outcome.parent_preserved is True
    assert outcome.final_workspace_sha256 == artifact_workspace_sha256(root)
    assert artifact_workspace_sha256(out / "artifact") == artifact_workspace_sha256(root)
    assert outcome.diagnostic_workspace_sha256 == artifact_workspace_sha256(seed)
    assert artifact_workspace_sha256(out / "failed_artifact") == artifact_workspace_sha256(seed)
    assert not outcome.implementation_real


@pytest.mark.parametrize("termination", [
    "finish", "duplicate", "call_cap", "tool_cap", "output_cap",
    "input_cap", "provider_error", "nonobject", "blocked",
])
def test_unchanged_failed_seed_is_not_new_work(tmp_path, termination):
    root, seed, cfg = repair_fixture(tmp_path)
    inspect = {"tool": "inspect", "path": "helper.py"}
    responses = [inspect]
    if termination == "finish":
        responses = [{"tool": "finish"}, {"tool": "blocked", "reason": "no repair"}]
    elif termination == "duplicate":
        responses = [inspect] * 4
    elif termination == "call_cap":
        cfg = replace(cfg, max_model_calls=1, config_sha256="")
    elif termination == "tool_cap":
        cfg = replace(cfg, max_tool_calls=1, config_sha256="")
    elif termination == "output_cap":
        cfg = replace(cfg, max_output_tokens=5, config_sha256="")
    elif termination == "input_cap":
        cfg = replace(cfg, max_input_tokens=10, config_sha256="")
    elif termination == "provider_error":
        responses = []  # Scripted StopIteration models a failed response.
    elif termination == "nonobject":
        responses = [[]]
    elif termination == "blocked":
        responses = [{"tool": "blocked", "reason": "cannot implement plan"}]
    coder = RecordingCoder(cfg, ScriptedModel(responses))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert coder.validated_workspaces == []
    assert_failed_retention(outcome, root, seed, out)
    route = classify_recovery_route(outcome, artifact_valid=False,
                                   native_metric=None, parent_metric=0.0)
    assert route.route not in {"exploit_verified_program", "new_hypothesis_or_alternative_realization"}


def test_failed_real_repair_rolls_back_to_verified_parent_not_seed(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    cfg = replace(cfg, max_repairs=0, config_sha256="")
    before = artifact_workspace_sha256(root)
    model = ScriptedModel([
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'worse'"},
        {"tool": "finish"},
    ])
    coder = RecordingCoder(cfg, model, accepted=False)
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert outcome.parent_preserved
    assert outcome.final_workspace_sha256 == before
    assert artifact_workspace_sha256(out / "artifact") == before
    assert outcome.diagnostic_workspace_sha256 != artifact_workspace_sha256(seed)
    assert (out / "failed_artifact/helper.py").read_text() == "VALUE = 'worse'\n"
    assert (seed / "helper.py").read_text() == "VALUE = 'bad'\n"
    assert artifact_workspace_sha256(root) == before


@pytest.mark.parametrize("termination", ["finish", "call_cap", "output_cap", "provider_error", "duplicate"])
def test_real_helper_repair_still_reaches_unchanged_validation(tmp_path, termination):
    root, seed, cfg = repair_fixture(tmp_path)
    edit = {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'good'"}
    responses = [edit]
    if termination == "finish":
        responses += [{"tool": "finish"}]
    elif termination == "call_cap":
        cfg = replace(cfg, max_model_calls=1, config_sha256="")
    elif termination == "output_cap":
        cfg = replace(cfg, max_output_tokens=5, config_sha256="")
    elif termination == "duplicate":
        responses += [{"tool": "inspect", "path": "helper.py"}] * 4
    # provider_error: exhausted scripted model raises after the real edit.
    coder = RecordingCoder(cfg, ScriptedModel(responses))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.implementation_real
    assert len(coder.validated_workspaces) == 1
    assert coder.validated_workspaces[0] == outcome.final_workspace_sha256
    assert (out / "artifact/helper.py").read_text() == "VALUE = 'good'\n"
    assert {"solution.py", "helper.py"}.issubset(outcome.changed_paths)


def test_noop_write_is_not_a_repair(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    cfg = replace(cfg, max_model_calls=1, config_sha256="")
    coder = RecordingCoder(cfg, ScriptedModel([
        {"tool": "write", "path": "helper.py", "content": (seed / "helper.py").read_text()},
    ]))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert coder.validated_workspaces == []
    assert_failed_retention(outcome, root, seed, out)


def test_explicit_diagnostic_run_is_still_available_before_mutation(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    coder = RecordingCoder(cfg, ScriptedModel([
        {"tool": "run", "argv": ["test-diagnostic-only"]},
        {"tool": "blocked", "reason": "profile supplied to planner"},
    ]))
    invocations = []

    def diagnostic_run(*args, **kwargs):
        invocations.append((args, kwargs))
        return {"returncode": 0, "timed_out": False, "stdout": "diagnostic"}

    coder._run_command = diagnostic_run
    outcome = run_seeded(coder, root, seed, tmp_path / "out")
    assert len(invocations) == 1
    assert coder.validated_workspaces == []
    assert outcome.status == CoderExitStatus.PLAN_BLOCKED


def test_bad_seed_hash_fails_before_model_or_validation(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    model = ScriptedModel([])
    coder = RecordingCoder(cfg, model)
    with pytest.raises(ValueError, match="diagnostic seed workspace hash mismatch"):
        coder.run(root, tmp_path / "out", diagnostic_seed_root=seed,
                  expected_diagnostic_workspace_sha256=hashlib.sha256(b"wrong").hexdigest())
    assert model.calls == 0
    assert coder.validated_workspaces == []


def test_real_repair_runs_actual_python_validation(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    coder = ContractBoundedCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'good'"},
        {"tool": "finish"},
    ]))
    outcome = run_seeded(coder, root, seed, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert all(v["accepted"] for v in outcome.validation)
    assert outcome.validation[-1]["stdout"].strip() == "good"


@pytest.mark.parametrize("version", [CoderVersion.C0, CoderVersion.C1, CoderVersion.C2])
def test_other_coder_versions_do_not_validate_unchanged_failed_seed(tmp_path, version):
    root, seed, cfg = repair_fixture(tmp_path)
    cfg = replace(cfg, version=version, max_model_calls=1, config_sha256="")
    response = {"tool": "finish"}
    if version == CoderVersion.C0:
        response = {"candidate_content": (seed / "solution.py").read_text()}
    elif version == CoderVersion.C1:
        response = {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'bad'"}
    coder = RecordingCoder(cfg, ScriptedModel([response]))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert coder.validated_workspaces == []
    assert_failed_retention(outcome, root, seed, out)


def test_penultimate_turn_does_not_validate_unchanged_seed(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    cfg = replace(cfg, max_model_calls=4, config_sha256="")
    coder = RecordingCoder(cfg, ScriptedModel([
        {"tool": "inspect", "path": "helper.py"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "blocked", "reason": "no implementation"},
    ]))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert coder.validated_workspaces == []
    assert_failed_retention(outcome, root, seed, out)


def test_edit_then_revert_to_seed_does_not_count_as_new_implementation(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    cfg = replace(cfg, max_tool_calls=2, config_sha256="")
    coder = RecordingCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'good'"},
        {"tool": "edit", "path": "helper.py", "old": "'good'", "new": "'bad'"},
    ]))
    out = tmp_path / "out"
    outcome = run_seeded(coder, root, seed, out)
    assert coder.validated_workspaces == []
    assert_failed_retention(outcome, root, seed, out)


def test_noop_finish_feedback_can_lead_to_real_repair(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)

    class FeedbackModel(ScriptedModel):
        def __init__(self):
            super().__init__([
                {"tool": "finish"},
                {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'good'"},
                {"tool": "finish"},
            ])
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            return super().respond(payload)

    model = FeedbackModel()
    coder = RecordingCoder(cfg, model)
    outcome = run_seeded(coder, root, seed, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert len(coder.validated_workspaces) == 1
    assert model.payloads[0]["feedback"]["repair_start"]["diagnostic_workspace_sha256"] == artifact_workspace_sha256(seed)
    assert model.payloads[1]["feedback"]["observation"]["error"] == "diagnostic_seed_requires_repair"
    checkpoint = next(e["payload"] for e in outcome.events if e["kind"] == "coder.checkpoint_created")
    assert checkpoint["initial_work_workspace_sha256"] == artifact_workspace_sha256(seed)
    assert checkpoint["parent_workspace_sha256"] == artifact_workspace_sha256(root)


@pytest.mark.parametrize("termination", [
    "duplicates", "repair_cap", "call_cap", "tool_cap", "output_cap",
    "input_cap", "provider_error", "nonobject", "C0", "C1",
])
def test_known_failed_workspace_is_not_automatically_validated_twice(tmp_path, termination):
    root, seed, cfg = repair_fixture(tmp_path)
    edit = {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'worse'"}
    inspect = {"tool": "inspect", "path": "helper.py"}
    responses = [edit, {"tool": "finish"}]
    if termination == "duplicates":
        cfg = replace(cfg, max_model_calls=12, config_sha256="")
        responses += [inspect] * 4
    elif termination == "repair_cap":
        cfg = replace(cfg, max_repairs=0, config_sha256="")
    elif termination == "call_cap":
        cfg = replace(cfg, max_model_calls=2, config_sha256="")
    elif termination == "tool_cap":
        cfg = replace(cfg, max_tool_calls=2, config_sha256="")
        responses += [inspect]
    elif termination == "output_cap":
        cfg = replace(cfg, max_output_tokens=15, config_sha256="")
        responses += [inspect]
    elif termination == "input_cap":
        cfg = replace(cfg, max_input_tokens=30, config_sha256="")
        responses += [inspect]
    elif termination == "nonobject":
        responses += [[]]
    elif termination == "C0":
        cfg = replace(cfg, version=CoderVersion.C0, max_model_calls=1, config_sha256="")
        responses = [{"candidate_content": (seed / "solution.py").read_text() + "\n# attempted fix\n"}]
    elif termination == "C1":
        cfg = replace(cfg, version=CoderVersion.C1, max_model_calls=1, config_sha256="")
        responses = [edit]
    coder = RecordingCoder(cfg, ScriptedModel(responses), accepted=False)
    outcome = run_seeded(coder, root, seed, tmp_path / "out")
    assert outcome.status != CoderExitStatus.SUCCESS
    assert len(coder.validated_workspaces) == 1
    assert outcome.validation[0]["accepted"] is False
    assert outcome.parent_preserved
    if termination == "C0":
        # One-shot failure already has no automatic terminal-validation path.
        assert outcome.reason == "one_shot_failed"
    else:
        assert any(e["kind"] == "coder.unchanged_failed_validation_not_repeated" for e in outcome.events)


def test_known_failure_allows_validation_after_actual_second_mutation(tmp_path):
    root, seed, cfg = repair_fixture(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'worse'"},
        {"tool": "finish"},
        {"tool": "edit", "path": "helper.py", "old": "'worse'", "new": "'good'"},
        {"tool": "finish"},
    ])

    class ChangingVerdictCoder(RecordingCoder):
        def _validate(self, root, events, *, deadline=None):
            self.accepted = bool(self.validated_workspaces)
            return super()._validate(root, events, deadline=deadline)

    coder = ChangingVerdictCoder(cfg, model)
    outcome = run_seeded(coder, root, seed, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert len(coder.validated_workspaces) == 2
    assert len(set(coder.validated_workspaces)) == 2
    assert outcome.repaired_after_failure
