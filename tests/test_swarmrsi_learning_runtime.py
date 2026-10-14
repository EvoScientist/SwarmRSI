"""Zero-provider integration checks; subprocess failures are synthetic."""
import json
from pathlib import Path
import sys
from unittest.mock import patch

import jsonschema
import pytest

from swarmrsi.coder_openrouter import OpenRouterCoderModel
from swarmrsi.learning_trials import SubprocessValidationRunner
from swarmrsi.learning_evidence import planner_evidence, verify_planner_transport
from swarmrsi.retained_results import read_json, write_immutable
from swarmrsi.tree_policy import PolicyTreeStrategy, TreePolicy
from swarmrsi.wisdom import WisdomContext, WisdomLibrary
from swarmrsi.naturebench_session import strategy
from test_swarmrsi_incremental_feedback import campaign
from test_swarmrsi_learning import make_run
from learning_test_helpers import configure, raw
from learning_test_helpers import client


def test_rsi_proposer_requests_policy_json_not_coding_tools():
    policy = TreePolicy(parent_exploration_weight=1)
    answer = {"parameters": policy.to_dict()["parameters"], "rationale": "synthetic", "evidence_run_ids": ["fixture"]}
    response = {"choices": [{"message": {"content": json.dumps(answer)}, "finish_reason": "stop"}],
                "usage": {"prompt_tokens": 11, "completion_tokens": 13, "cost": .02}}
    with patch("urllib.request.urlopen") as http:
        http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
        parsed, usage = client(native=True).respond({"version": "RSI_POLICY", "policy_learning_request": {}})
    body = json.loads(http.call_args.args[0].data)
    assert "tools" not in body and body["response_format"] == {"type": "json_object"}
    assert "search policy" in body["messages"][0]["content"]
    assert parsed["parameters"] == answer["parameters"] and usage["cost_usd"] == .02
    assert json.loads(parsed["_transport_request_body_json"]) == body
    assert "_raw_sha256" in parsed and http.call_count == 1


def test_runtime_consumes_frozen_policy_and_rejects_other_search_versions():
    profile = {"tree_search": {"policy": "continuation_tree"},
               "rsi_policy": TreePolicy(max_scored_continuations=3).to_dict()}
    actual = strategy("v1", 5, profile)
    assert isinstance(actual, PolicyTreeStrategy)
    assert actual.config["max_scored_continuations"] == 3 and actual.config["seed"] == 5
    for version in ("v2", "v3", "v4"):
        with pytest.raises(ValueError, match="only run on v1"):
            strategy(version, 5, profile)


def test_actual_planner_receives_wisdom_and_graph_records_declared_use(tmp_path):
    source = make_run(tmp_path / "history", role="development")
    library = WisdomLibrary(tmp_path / "wisdom")
    ids = library.ingest(source)
    snapshot_path = tmp_path / "W0.json"
    library.freeze(snapshot_path)
    loop = campaign(tmp_path / "loop")
    proposal = {**raw(), "wisdom_entry_ids": [ids[0]]}
    planner, calls = configure(loop, [proposal], audit_root=tmp_path / "planner")
    def actual_adapter(request):
        calls.append(request)
        response = {"choices": [{"message": {"content": None, "tool_calls": [{"function": {
            "name": "submit_proposals", "arguments": json.dumps({"proposals": [proposal]})}}]},
            "finish_reason": "stop"}], "usage": {"prompt_tokens": 3, "completion_tokens": 5}}
        with patch("urllib.request.urlopen") as http:
            http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
            return client().respond(request)
    planner.model.respond = actual_adapter
    planner.wisdom_context = WisdomContext(task_id="different", mode="readonly", snapshot_path=snapshot_path)
    planner.task_context = "linear predictions public metric"
    assert loop.expand() == 1
    packet = calls[0]["planner_payload"]
    assert packet["cross_task_wisdom"]["entries"][0]["entry_sha256"] == ids[0]
    schema = OpenRouterCoderModel._planner_tool_definition(packet["proposal_slots"])["function"]["parameters"]
    jsonschema.validate({"proposals": [proposal]}, schema)
    record = read_json(tmp_path / "planner/batch_000000.record.json")
    assert record["wisdom"]["injected"] and record["wisdom"]["causal_effect"] == "not_established"
    assert record["wisdom"]["proposal_use"][0]["declared_entry_ids"] == [ids[0]]
    assert any(e.payload.get("wisdom", {}).get("injected") for e in loop.view.events)
    context = planner.wisdom_context
    manifest = {"task_id": "different", "learning_context": {
        "schema_version": "swarmrsi-learning-context-v1", "wisdom_mode": context.mode,
        "wisdom_config": context.config, "wisdom_snapshot_sha256": context.config["snapshot_sha256"],
        "wisdom_snapshot_ref": context.snapshot_ref}}
    verified = planner_evidence(tmp_path, str(tmp_path), loop.checkpoint(), manifest)
    assert verified["status"] == "verified_original_journals" and verified["injected_batches"] == 1
    assert list(verified["declared_use"].values()) == [[ids[0]]]
    changed = tmp_path / "planner/batch_000000.request.json"
    changed.write_text(changed.read_text() + "\n")
    with pytest.raises(ValueError, match="request/response binding"):
        planner_evidence(tmp_path, str(tmp_path), loop.checkpoint(), manifest)


@pytest.mark.parametrize("damage", ["missing", "http", "raw", "decoded"])
def test_planner_journal_requires_actual_http_and_provider_message(damage):
    import hashlib
    request = {"version": "PLANNER", "planner_payload": {"cross_task_wisdom": "synthetic"}}
    answer = {"choices": [{"message": {"content": '{"proposals": []}'}, "finish_reason": "stop"}]}
    with patch("urllib.request.urlopen") as http:
        http.return_value.__enter__.return_value.read.return_value = json.dumps(answer).encode()
        parsed, _ = client(native=False).respond(request)
    verify_planner_transport(request, parsed, "test/model")
    if damage == "missing":
        parsed.pop("_transport_request_body_json")
    elif damage == "http":
        body = json.loads(parsed["_transport_request_body_json"])
        body["messages"][1]["content"] = json.dumps({"version": "PLANNER", "planner_payload": {}})
        parsed["_transport_request_body_json"] = json.dumps(body)
        parsed["_transport_request_body_sha256"] = hashlib.sha256(json.dumps(body).encode()).hexdigest()
    elif damage == "raw":
        parsed["_raw_content"] += " "
    else:
        parsed["proposals"] = [{"wisdom_entry_ids": ["never-returned"]}]
    with pytest.raises(ValueError, match="planner"):
        verify_planner_transport(request, parsed, "test/model")


def test_off_preserves_identical_planner_payload_and_local_history(tmp_path):
    left, right = campaign(tmp_path / "left"), campaign(tmp_path / "right")
    a, ca = configure(left, [raw()])
    b, cb = configure(right, [raw()])
    b.wisdom_context = WisdomContext(task_id="won", mode="off", snapshot_path=tmp_path / "absent")
    assert a._state_config_sha256() == b._state_config_sha256()
    left.expand(); right.expand()
    assert ca == cb
    assert "nodes" in cb[0]["planner_payload"] and "research_state" in cb[0]["planner_payload"]


def validation_runner(tmp_path, *, execute=False):
    """A real child process that exits 17; never masquerades as scored evidence."""
    source = tmp_path / "source"
    (source / "src/swarmrsi").mkdir(parents=True)
    (source / "scripts").mkdir()
    (source / "starter").mkdir()
    (source / "starter/run.py").write_text("print('synthetic starter')\n")
    (source / "src/swarmrsi/fixture.py").write_text("# synthetic source binding\n")
    (source / "scripts/evaluate_naturebench_local_workspace.py").write_text("# no evaluator executed\n")
    (source / "scripts/run_swarmrsi_v_search.py").write_text(
        "import sys\nif '--preflight' in sys.argv:\n print('synthetic zero-call preflight')\n sys.exit(0)\nsys.exit(17)\n")
    profile_path = source / "profile.json"
    write_immutable(profile_path, {"tree_search": {"policy": "continuation_tree"},
                                   "tasks": {"fixture": {"starter": "starter"}}})
    return SubprocessValidationRunner(tmp_path / "validator", source_root=source,
        profile_path=profile_path, runtime_python=Path(sys.executable),
        cells=[{"task_id": "fixture", "seed": 0}], allow_execution=execute)


def test_real_subprocess_failure_is_recorded_once_and_never_respent(tmp_path):
    runner = validation_runner(tmp_path, execute=True)
    with pytest.raises(RuntimeError, match="returncode=17"):
        runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)
    with patch("subprocess.Popen") as spawn:
        with pytest.raises(RuntimeError, match="not relaunched"):
            runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)
        assert spawn.call_count == 0
    accounting = runner.accounting()
    assert accounting["unique_launches"] == 1 and accounting["unknown_cost_launches"] == 1
    assert accounting["trials"][0]["run_id"] is None


def test_validator_preflight_and_execution_switch_are_separate(tmp_path):
    runner = validation_runner(tmp_path)
    assert runner.preflight(policy=TreePolicy(), task_id="fixture", seed=0)["ready"]
    with pytest.raises(ValueError, match="execution is disabled"):
        runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)
    assert runner.accounting()["unique_launches"] == 0


def test_interpreter_entry_path_preserves_virtual_environment(tmp_path):
    runner = validation_runner(tmp_path)
    assert runner.runtime == Path(sys.executable).absolute()
    assert runner.config["runtime_executable_resolved"] == str(Path(sys.executable).resolve())
    _, command = runner._prepare(TreePolicy(), "fixture", 0)
    assert command[0] == str(Path(sys.executable).absolute())


def test_runtime_environment_is_frozen_even_when_previously_absent(tmp_path, monkeypatch):
    monkeypatch.delenv("SWARMRSI_NATUREBENCH_TASKS_ROOT", raising=False)
    monkeypatch.setenv("TORCH_HOME", "/synthetic/original-cache")
    runner = validation_runner(tmp_path)
    monkeypatch.setenv("SWARMRSI_NATUREBENCH_TASKS_ROOT", "/synthetic/other-tasks")
    monkeypatch.setenv("TORCH_HOME", "/synthetic/other-cache")
    env = runner._environment()
    assert "SWARMRSI_NATUREBENCH_TASKS_ROOT" not in env
    assert env["TORCH_HOME"] == "/synthetic/original-cache"


def test_terminal_path_precedes_any_inner_runner_invocation(tmp_path, monkeypatch):
    runner = validation_runner(tmp_path)
    root, _ = runner._prepare(TreePolicy(), "fixture", 0)
    (root / "run").mkdir()
    write_immutable(root / "run/final.json", {"test": "routing sentinel, not valid evaluation"})
    # Only test dispatch ordering here; source-valid receipts are covered in
    # test_swarmrsi_learning. No sentinel result reaches an RSI comparison.
    sentinel = {"run_id": "offline-routing-sentinel", "evidence_sha256": "a" * 64}
    monkeypatch.setattr(runner, "_completed", lambda *args: sentinel)
    with patch("subprocess.Popen") as spawn:
        assert runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0) == sentinel
        assert spawn.call_count == 0
    assert not (root / "launch.json").exists()
