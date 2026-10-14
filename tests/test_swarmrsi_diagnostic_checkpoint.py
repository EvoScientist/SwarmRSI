"""R35 exact diagnostic reachability, with no providers or benchmark execution."""
import json
from pathlib import Path

import pytest

from swarmrsi.coder_engine import ContractBoundedCoder
from swarmrsi.coder_models import CoderExitStatus
from swarmrsi.command_feedback import CommandFailureJournal
from test_swarmrsi_coder import ScriptedModel, config, workspace

S, T = "a" * 64, "b" * 64
BAD = {"tool": "run", "argv": ["synthetic-probe", "bad"], "timeout_seconds": 30}
FAIL = {"execution_started": True, "returncode": 1, "timed_out": False,
        "stdout": "", "stderr": "FileNotFoundError: public_data", "wall_seconds": .1}
GOOD = {"execution_started": True, "returncode": 0, "timed_out": False,
        "stdout": "corrected diagnostic reached", "stderr": "", "wall_seconds": .1}


def record(journal, *, action=BAD, result=FAIL, before=S, after=S, call=1):
    return journal.observe(action, result, before_sha256=before, after_sha256=after, tool_call=call)


@pytest.mark.parametrize("change", [
    {"argv": ["synthetic-probe", "corrected"]}, {"cwd": "other"},
    {"timeout_seconds": 60},
])
def test_only_exact_command_is_blocked(change):
    journal = CommandFailureJournal()
    receipt = record(journal)
    repeated = journal.unchanged_failed_command(BAD, S)
    assert repeated["failure_id"] == receipt["failure_id"]
    assert repeated["last_observation_sha256"] == receipt["observation_sha256"]
    assert journal.unchanged_failed_command(BAD | change, S) is None
    repeated["failure_id"] = "mutated"
    assert journal.unchanged_failed_command(BAD, S)["failure_id"] == receipt["failure_id"]


@pytest.mark.parametrize("before,after,current", [(S, S, T), (S, T, T), (T, S, S)])
def test_stale_or_mutating_execution_does_not_gate_new_source(before, after, current):
    journal = CommandFailureJournal()
    record(journal, before=before, after=after)
    assert journal.unchanged_failed_command(BAD, current) is None


@pytest.mark.parametrize("result", [
    FAIL | {"execution_started": False},
    {k: v for k, v in FAIL.items() if k != "execution_started"},  # Legacy advisory only.
    FAIL | {"execution_started": True, "returncode": None},
    FAIL | {"harness_error": "command_output_decode_error"},
    FAIL | {"process_status": "running"},
    {"error": "invalid_model_action", "transport_error_category": "provider_failure"},
    {"error": "implementation_checkpoint_required"},
])
def test_nonexecution_nonterminal_legacy_and_harness_records_do_not_gate(result):
    journal = CommandFailureJournal()
    record(journal, result=result)
    assert journal.unchanged_failed_command(BAD, S) is None


@pytest.mark.parametrize("result", [
    FAIL | {"timed_out": True, "returncode": None},
    FAIL | {"returncode": 137, "stderr": ""},
])
def test_timeout_or_unknown_nonzero_can_gate_exact_repeat_without_scientific_verdict(result):
    journal = CommandFailureJournal()
    record(journal, result=result)
    assert journal.unchanged_failed_command(BAD, S)
    projection = journal.project(S)
    assert projection["authority"] == "diagnostic_only_not_candidate_or_scientific_verdict"
    assert projection["entries"][0]["classification"]["scope"] == "unknown"


def test_latest_harness_failure_does_not_revive_old_candidate_failure_gate():
    journal = CommandFailureJournal()
    record(journal)
    record(journal, result=FAIL | {"harness_error": "output_decode_error"}, call=2)
    assert journal.unchanged_failed_command(BAD, S) is None
    # A later concrete command failure, not the harness diagnosis, restores the guard.
    record(journal, call=3)
    assert journal.unchanged_failed_command(BAD, S)["last_tool_call"] == 3


def test_success_resolves_exact_command_not_unrelated_diagnostics():
    journal = CommandFailureJournal()
    record(journal)
    record(journal, action=BAD | {"argv": ["another-probe"]}, result=GOOD, call=2)
    assert journal.unchanged_failed_command(BAD, S)
    record(journal, result=GOOD, call=3)
    assert journal.unchanged_failed_command(BAD, S) is None
    record(journal, call=4)
    assert journal.unchanged_failed_command(BAD, S)["last_tool_call"] == 4


def test_lookup_is_independent_of_clipped_three_entry_prompt_projection():
    journal = CommandFailureJournal()
    receipt = record(journal)
    for index in range(2, 10):
        record(journal, action=BAD | {"argv": ["probe", str(index)]}, call=index)
    assert journal.project(S)["omitted_failures"] > 0
    assert all(row["failure_id"] != receipt["failure_id"] for row in journal.project(S)["entries"])
    assert journal.unchanged_failed_command(BAD, S)["failure_id"] == receipt["failure_id"]


@pytest.mark.parametrize("tool_request", [{"tool": "inspect"}, BAD | {"argv": []}, BAD | {"argv": "bad"}])
def test_invalid_or_other_tools_are_left_to_existing_executor(tool_request):
    journal = CommandFailureJournal()
    record(journal)
    assert journal.unchanged_failed_command(tool_request, S) is None


class ProbeCoder(ContractBoundedCoder):
    def __init__(self, cfg, model, *, responses):
        super().__init__(cfg, model)
        self.responses = responses
        self.invocations = []

    def _run_command(self, root, argv, timeout, cwd=".", *, sandboxed=False):
        if argv in self.responses:
            self.invocations.append((argv, timeout, cwd))
            return dict(self.responses[argv])
        return super()._run_command(root, argv, timeout, cwd, sandboxed=sandboxed)


def run_correction(tmp_path, bad, corrected, failure=FAIL):
    root = workspace(tmp_path)
    model = ScriptedModel([
        bad,
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "metric.json"},
        bad,  # Fourth call reaches the checkpoint, unchanged failure must not rerun.
        corrected,
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    cfg = config(root, calls=8)
    coder = ProbeCoder(cfg, model, responses={tuple(bad["argv"]): failure, tuple(corrected["argv"]): GOOD})
    outcome = coder.run(root, tmp_path / "out")
    observations = [e["payload"]["observation"] for e in outcome.events if e["kind"] == "coder.tool_finished"]
    assert [row[0] for row in coder.invocations] == [tuple(bad["argv"]), tuple(corrected["argv"])]
    assert any(o.get("error") == "unchanged_failed_command_at_implementation_checkpoint" for o in observations)
    assert any(o.get("stdout") == GOOD["stdout"] for o in observations)
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.parent_preserved and outcome.implementation_real
    assert outcome.model_calls <= cfg.max_model_calls and outcome.tool_calls <= cfg.max_tool_calls
    assert all(v["accepted"] for v in outcome.validation)
    return outcome


@pytest.mark.parametrize("bad_code,good_code", [
    ("DATA_DIR='public_data'; print(DATA_DIR)", "DATA_DIR='/public_data'; print(DATA_DIR)"),
    ("import os; for x in []: pass", "import os\nfor x in []:\n    pass"),
])
def test_changed_path_or_inline_syntax_command_reaches_existing_executor(tmp_path, bad_code, good_code):
    bad = {"tool": "run", "argv": ["python", "-c", bad_code], "timeout_seconds": 30}
    corrected = {**bad, "argv": ["python", "-c", good_code]}
    run_correction(tmp_path, bad, corrected)


def test_original_p5_corrected_public_path_is_not_blocked(tmp_path):
    repo = Path(__file__).resolve().parents[1]
    path = repo / "results/audits/installed_rsi_p5_20260915/first_two_actions/000_a0000_e01d46d6bf/result.json"
    if not path.exists():
        pytest.skip("optional original P5 trace not distributed with package")
    original = path.read_bytes()
    import hashlib
    assert hashlib.sha256(original).hexdigest() == "350615290b482548c7180559f4bd0e23bf03064b1c68f0a3e54e35d7d9b594f8"
    current, executed, corrected = None, None, None
    for event in json.loads(original)["events"]:
        if event["kind"] == "coder.model_response":
            current = event["payload"]["response"]
        if event["kind"] == "coder.tool_finished" and event["payload"]["tool"] == "run":
            observation = event["payload"]["observation"]
            if observation.get("execution_started") is True:
                executed = (current, observation)
            elif observation.get("error") == "implementation_checkpoint_required" and "/public_data" in current["argv"][-1]:
                corrected = current
                break
    assert executed and corrected
    run_correction(tmp_path, executed[0], corrected, executed[1])
    assert path.read_bytes() == original


def test_new_probe_after_checkpoint_without_prior_failure_is_executable(tmp_path):
    root = workspace(tmp_path)
    probe = BAD | {"argv": ["synthetic-probe", "new"]}
    coder = ProbeCoder(config(root, calls=6), ScriptedModel([
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "metric.json"},
        probe, {"tool": "blocked", "reason": "diagnostic only"},
    ]), responses={tuple(probe["argv"]): GOOD})
    outcome = coder.run(root, tmp_path / "out")
    assert len(coder.invocations) == 1
    assert outcome.status == CoderExitStatus.PLAN_BLOCKED
    assert not outcome.implementation_real
    assert outcome.validation == []  # A successful probe is not a new evaluated candidate.
