"""Offline evidence/feedback tests, not model-effect or benchmark results."""
from __future__ import annotations

import hashlib
import json
import sys
from dataclasses import replace
from pathlib import Path

import pytest

from evolvcoder.command_feedback import CommandFailureJournal
from evolvcoder.coder_models import coder_sha256, CoderExitStatus
from test_evolvcoder_coder import ScriptedModel, config, workspace
from evolvcoder.coder_engine import ContractBoundedCoder


S = hashlib.sha256(b"source-one").hexdigest()
T = hashlib.sha256(b"source-two").hexdigest()
RUN = {"tool": "run", "argv": ["python", "-c", "import run; run.main()"], "timeout_seconds": 300}
ERROR = {"returncode": 137, "timed_out": False, "stdout": "", "stderr": "", "wall_seconds": 49.0}


def observe(journal, action=RUN, result=None, before=S, after=S, call=1):
    return journal.observe(action, ERROR if result is None else result,
                           before_sha256=before, after_sha256=after, tool_call=call)


def test_repeated_failures_ignore_only_wall_time_and_retain_raw_hash():
    journal = CommandFailureJournal()
    for n in range(11):
        raw = dict(ERROR, wall_seconds=49 + n / 100)
        receipt = observe(journal, result=raw, call=n+1)
        assert receipt["observation_sha256"] == coder_sha256(raw)
    row = journal.project(S)["entries"][0]
    assert row["repeat_count"] == 11
    assert row["classification"]["kind"] == "nonzero_exit"
    assert row["classification"]["scope"] == "unknown"
    assert row["last_tool_call"] == 11


@pytest.mark.parametrize("change", [
    {"stdout": "new progress"}, {"stderr": "new traceback"},
    {"runtime_artifacts": {"sha256": T}}, {"timed_out": True},
])
def test_other_observation_changes_are_not_exact_repeats(change):
    journal = CommandFailureJournal()
    observe(journal)
    observe(journal, result=ERROR | change, call=2)
    assert journal.project(S)["total_unique_failures"] == 2


def test_cwd_command_limits_and_source_states_are_distinct():
    journal = CommandFailureJournal()
    observe(journal)
    observe(journal, action=RUN | {"cwd": "other"}, call=2)
    observe(journal, action=RUN | {"timeout_seconds": 600}, call=3)
    observe(journal, before=T, after=T, call=4)
    observe(journal, before=S, after=T, call=5)
    view = journal.project(T)
    assert view["total_unique_failures"] == 5
    assert view["omitted_failures"] == 5 - len(view["entries"])
    assert all(x["state"] == "stale" for x in journal.project("f" * 64, max_chars=10000)["entries"])


def test_auxiliary_success_cannot_erase_other_failure_or_old_source_failure():
    journal = CommandFailureJournal()
    observe(journal)
    good = {"returncode": 0, "stdout": "ok", "stderr": "", "timed_out": False}
    observe(journal, action=RUN | {"argv": ["python", "-c", "print('ok')"]}, result=good, call=2)
    assert journal.project(S)["entries"][0]["state"] == "unresolved"
    observe(journal, result=good, before=T, after=T, call=3)
    assert journal.project(S)["entries"][0]["state"] == "unresolved"
    assert journal.project(T)["entries"][0]["state"] == "stale"
    observe(journal, result=good, call=4)
    row = journal.project(S)["entries"][0]
    assert row["state"] == "resolved_command_only"
    assert row["resolution"]["meaning"] == "same_command_succeeded_not_candidate_validated"
    observe(journal, call=5)
    assert journal.project(S)["entries"][0]["state"] == "unresolved"


@pytest.mark.parametrize("observation", [
    {"error": "implementation_checkpoint_required"},
    {"returncode": None, "harness_error": "command_not_allowlisted", "timed_out": False},
    ERROR | {"process_status": "running"},
    ERROR | {"execution_started": False},
])
def test_unexecuted_requests_and_live_polls_are_excluded(observation):
    journal = CommandFailureJournal()
    assert observe(journal, result=observation) is None
    assert journal.project(S) is None


def test_exception_scope_is_observation_not_scientific_attribution():
    journal = CommandFailureJournal()
    raw = dict(ERROR, returncode=1, stderr='  File "<string>", line 1\n    import os; for x in []: pass\n               ^^^\nSyntaxError: invalid syntax\n')
    observe(journal, action=RUN | {"argv": ["python", "-c", "import os; for x in []: pass"]}, result=raw)
    row = journal.project(S)["entries"][0]
    assert row["classification"]["scope"] == "inline_command"
    assert "candidate program" in row["classification"]["next_action"]
    observe(journal, action=RUN | {"argv": ["python", "-c", "x=1; for x in []: pass"]}, result=raw, call=2)
    view = journal.project(S)
    assert view["total_unique_failures"] == 2
    assert all(x["observed_family_failure_count"] == 2 for x in view["entries"])
    assert all(x["repeat_count"] == 1 for x in view["entries"])
    journal2 = CommandFailureJournal()
    observe(journal2, result=dict(raw, stderr="some text mentioning SyntaxError"))
    assert journal2.project(S)["entries"][0]["classification"]["scope"] == "unknown"
    journal3 = CommandFailureJournal()
    observe(journal3, action=RUN | {"argv": ["python", "-c", "exec('for x:')"]}, result=raw)
    assert journal3.project(S)["entries"][0]["classification"]["scope"] == "unknown"


def test_projection_bound_and_omissions_are_explicit():
    journal = CommandFailureJournal()
    for n in range(10):
        observe(journal, action=RUN | {"argv": ["python", "-c", str(n) + "x" * 10000]}, call=n+1)
    for budget in [1000, 3000, 6000]:
        view = journal.project(S, max_chars=budget)
        assert len(json.dumps(view, ensure_ascii=False, sort_keys=True)) <= budget
        assert len(view["entries"]) <= 3
        assert view["omitted_failures"] + len(view["entries"]) == 10
        assert view["entries"]
        assert all(x.get("detail_fields_omitted") or x["command_omitted_chars"] > 0 for x in view["entries"])
    with pytest.raises(ValueError): journal.project(S, max_chars=10)


def test_projection_and_receipt_never_alias_journal_or_input_action():
    journal = CommandFailureJournal()
    action = dict(RUN, argv=list(RUN["argv"]))
    receipt = observe(journal, action=action)
    original_id = receipt["failure_id"]
    receipt["classification"]["kind"] = "tampered"
    action["argv"].append("tampered")
    view = journal.project(S)
    assert view["entries"][0]["classification"]["kind"] == "nonzero_exit"
    assert "tampered" not in view["entries"][0]["command_excerpt"]
    view["entries"][0]["classification"]["kind"] = "tampered again"
    assert journal.project(S)["entries"][0]["classification"]["kind"] == "nonzero_exit"
    observe(journal, result=dict(ERROR, returncode=0), call=2)
    view = journal.project(S)
    view["entries"][0]["resolution"]["meaning"] = "tampered"
    row = journal.project(S)["entries"][0]
    assert row["failure_id"] == original_id
    assert row["resolution"]["meaning"] == "same_command_succeeded_not_candidate_validated"


def test_real_decode_failure_is_an_executed_runner_failure(tmp_path):
    root = workspace(tmp_path)
    cfg = replace(config(root), allowed_command_prefixes=((sys.executable,),), config_sha256="")
    coder = ContractBoundedCoder(cfg, ScriptedModel([]))
    argv = [sys.executable, "-c", "import os; os.write(1, bytes([255]))"]
    raw = coder._run_command(root, tuple(argv), timeout=5)
    assert raw["execution_started"] is True
    assert raw["returncode"] is None
    assert raw["harness_error"] == "command_output_decode_error"
    assert raw["undecoded_stream_tail_hex"] == "ff"
    journal = CommandFailureJournal()
    receipt = observe(journal, action={"tool": "run", "argv": argv}, result=raw)
    assert receipt["classification"]["kind"] == "runner_error"


def test_engine_keeps_failure_visible_after_inspect_and_allows_real_repair(tmp_path):
    root = workspace(tmp_path)
    class RecordingModel(ScriptedModel):
        def __init__(self):
            super().__init__([
                RUN, RUN,
                {"tool": "inspect", "path": "solution.py"},
                {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
                {"tool": "finish"},
            ])
            self.payloads = []
        def respond(self, payload):
            self.payloads.append(payload)
            return super().respond(payload)
    class ReplayCoder(ContractBoundedCoder):
        def _run_command(self, root, argv, timeout, cwd=".", *, sandboxed=False):
            if tuple(RUN["argv"]) == argv:
                return dict(ERROR)
            return super()._run_command(root, argv, timeout, cwd, sandboxed=sandboxed)
    model = RecordingModel()
    outcome = ReplayCoder(config(root, calls=10), model).run(root, tmp_path / "out")
    assert "explicit_run_feedback" not in model.payloads[0]
    row = model.payloads[3]["explicit_run_feedback"]["entries"][0]
    assert row["repeat_count"] == 2 and row["state"] == "unresolved"
    assert model.payloads[4]["explicit_run_feedback"]["entries"][0]["state"] == "stale"
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.parent_preserved
    raw = [e for e in outcome.events if e["kind"] == "coder.tool_finished" and e["payload"]["tool"] == "run"]
    assert len(raw) == 2 and all(e["payload"]["observation"] == ERROR for e in raw)
    assert len([e for e in outcome.events if e["kind"] == "coder.explicit_run_feedback"]) == 2


@pytest.mark.parametrize("case,expected_count,expected_family,expected_kind", [
    ("r31b", 11, 11, "nonzero_exit"), ("peptide_first", 3, 4, "observed_exception"),
    ("peptide_third", 7, 8, "observed_exception"),
])
def test_original_closed_trace_feedback_replay(case, expected_count, expected_family, expected_kind):
    repo = Path(__file__).resolve().parents[1]
    paths = {
        "r31b": "results/audits/tree_learning_r31b_20260914/closed_prefix_06/001_repair_0000_69b1b068f3/result.json",
        "peptide_first": "results/audits/tree_learning_r29b_20260914/peptide_wisdom_off/first_cell/result.json",
        "peptide_third": "results/audits/tree_learning_r29b_20260914/peptide_wisdom_off/third_cell/result.json",
    }
    path = repo / paths[case]
    if not path.exists(): pytest.skip("optional original development trace not distributed with package")
    data = json.loads(path.read_text())
    journal = CommandFailureJournal()
    action = {}
    state = S
    for e in data["events"]:
        if e["kind"] == "coder.model_response": action = e["payload"]["response"]
        if e["kind"] == "coder.tool_finished":
            p = e["payload"]
            if p["tool"] in {"edit", "write", "append", "repair"} and p["ok"]:
                state = coder_sha256(p["observation"])
            if p["tool"] == "run":
                journal.observe(action, p["observation"], before_sha256=state, after_sha256=state, tool_call=e["sequence"])
    # This is observation recurrence replay with mutation boundaries, not a
    # reconstructed workspace/hash custody claim or a model decision replay.
    view = journal.project(state, max_chars=20000)
    rows = [r for r in view["entries"] if r["repeat_count"] == expected_count]
    assert rows and rows[0]["classification"]["kind"] == expected_kind
    # In peptide traces the first timeout allowance differs (60 vs 120),
    # so exact command-context recurrence is 1+3 / 1+7, not 4 / 8.
    assert rows[0]["observed_family_failure_count"] == expected_family
