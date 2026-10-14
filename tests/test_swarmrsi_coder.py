from __future__ import annotations

import hashlib
import http.server
import json
import shutil
import subprocess
import sys
import threading
import time
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path

import pytest

from swarmrsi import (
    ActionCandidate,
    ActionExecution,
    ActionOperator,
    CoderConfig,
    CoderExitStatus,
    CoderOutcome,
    CoderVersion,
    ContractBoundedCoder,
    ScientificActionContract,
    ValidationCommand,
    DynamicResearchView,
    HypothesisProgramTreeStrategy,
    OpenRouterCoderModel,
    ProgramNode,
    ProposedAction,
    ResearchDependency,
    ScientificFamily,
    VSearchCoordinator,
    artifact_workspace_sha256,
)
from swarmrsi.campaign import controller_repair_directive


def test_efficiency_repair_requires_complexity_change_and_measured_headroom():
    directive = controller_repair_directive("optimize_efficiency")
    assert "algorithmic-complexity defect" in directive
    assert "below 50%" in directive
    assert "public-data benchmark" in directive
    assert "Do not merely reduce restarts" in directive
    assert "Preserve the exact scientific hypothesis" in directive


def test_openrouter_total_deadline_terminates_slow_response():
    class SlowHandler(http.server.BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802
            time.sleep(1.0)
            self.send_response(200)
            self.end_headers()
            self.wfile.write(b'{}')

        def log_message(self, _format, *_args):
            return

    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), SlowHandler)
    server.daemon_threads = True
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    model = OpenRouterCoderModel(
        api_key="test-only",
        model="test/model",
        base_url=f"http://127.0.0.1:{server.server_port}",
        max_tokens_per_turn=10,
        request_timeout_seconds=5,
        request_total_timeout_seconds=0.2,
    )
    started = time.monotonic()
    try:
        with pytest.raises(TimeoutError, match="total response deadline"):
            model.respond({"version": "C3", "seed": 0})
    finally:
        server.shutdown()
        server.server_close()
    # The 0.2 s total deadline must fire before the 5 s per-request timeout;
    # 4 s still separates the two on a loaded shared node.
    assert time.monotonic() - started < 4.0


def test_openrouter_native_tool_call_is_normalized():
    response_payload = {
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "choices": [{"message": {"content": None, "tool_calls": [{
            "function": {"name": "inspect", "arguments": '{"path":"run.py"}'},
        }]}}],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(
        api_key="test-only", model="test/model", base_url="https://example.invalid",
        max_tokens_per_turn=100, native_tools=True,
        reasoning={"effort": "high", "exclude": True},
    )
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        action, usage = model.respond({"version": "C3", "seed": 0})
    request_body = json.loads(urlopen.call_args.args[0].data)
    assert request_body["tool_choice"] == "required"
    assert request_body["reasoning"] == {"effort": "high", "exclude": True}
    assert action == {"tool": "inspect", "arguments": {"path": "run.py"}}
    assert usage == {"input_tokens": 7, "output_tokens": 3}


def test_openrouter_planner_uses_typed_submit_proposals_tool():
    proposal = {
        "hypothesis": "test a bounded representation",
        "description": "implement the selected representation",
        "operator": "improve_program",
        "family": {
            "input_transform": "scaled",
            "primary_operator": "linear",
            "output_construction": "direct",
            "objective_or_test": "metric",
            "capacity_controls": ["fixed seed"],
        },
        "predictions": {
            "quality": 0.5, "information_gain": 0.5, "reuse": 0.5,
            "validity": 0.8, "cost": 0.2,
        },
    }
    response_payload = {
        "usage": {"prompt_tokens": 11, "completion_tokens": 17},
        "choices": [{
            "finish_reason": "tool_calls",
            "message": {"content": None, "tool_calls": [{
                "function": {
                    "name": "submit_proposals",
                    "arguments": json.dumps({"proposals": [proposal]}),
                },
            }]},
        }],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(
        api_key="test-only", model="test/model", base_url="https://example.invalid",
        max_tokens_per_turn=100, native_tools=True,
    )
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        parsed, usage = model.respond({
            "version": "PLANNER", "seed": 0, "planner_payload": {"count": 1},
        })
    request_body = json.loads(urlopen.call_args.args[0].data)
    assert request_body["tool_choice"] == "required"
    assert request_body["tools"][0]["function"]["name"] == "submit_proposals"
    assert "response_format" not in request_body
    assert {key: value for key, value in parsed.items() if not key.startswith("_")} == {"proposals": [proposal]}
    assert parsed["_provider_finish_reason"] == "tool_calls"
    assert json.loads(parsed["_raw_content"]) == response_payload["choices"][0]["message"]
    assert hashlib.sha256(parsed["_raw_content"].encode()).hexdigest() == parsed["_raw_sha256"]
    assert json.loads(parsed["_transport_request_body_json"]) == request_body
    assert hashlib.sha256(parsed["_transport_request_body_json"].encode()).hexdigest() == parsed["_transport_request_body_sha256"]
    assert usage == {"input_tokens": 11, "output_tokens": 17}


def test_openrouter_caps_each_turn_at_controller_remaining_output_budget():
    response_payload = {
        "usage": {"prompt_tokens": 7, "completion_tokens": 3},
        "choices": [{"message": {"content": None, "tool_calls": [{
            "function": {"name": "inspect", "arguments": '{"path":"run.py"}'},
        }]}}],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(
        api_key="test-only", model="test/model", base_url="https://example.invalid",
        max_tokens_per_turn=100, native_tools=True,
    )
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({
            "version": "C3", "seed": 0,
            "max_output_tokens_this_turn": 17,
        })
    request_body = json.loads(urlopen.call_args.args[0].data)
    assert request_body["max_tokens"] == 17


def test_coder_decrements_per_turn_output_ceiling(tmp_path):
    root = workspace(tmp_path)

    class CeilingModel(ScriptedModel):
        def __init__(self):
            super().__init__([
                {"tool": "inspect", "path": "solution.py"},
                {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
            ], usages=[
                {"input_tokens": 10, "output_tokens": 7},
                {"input_tokens": 10, "output_tokens": 3},
            ])
            self.ceilings = []

        def respond(self, payload):
            self.ceilings.append(payload["max_output_tokens_this_turn"])
            return super().respond(payload)

    model = CeilingModel()
    outcome = ContractBoundedCoder(
        config(root, calls=2, output_tokens=10), model
    ).run(root, tmp_path / "out")
    assert model.ceilings == [10, 3]
    assert outcome.output_tokens == 10
    assert outcome.status == CoderExitStatus.SUCCESS


def test_coder_replays_bounded_interaction_history_and_stops_duplicate_loop(tmp_path):
    root = workspace(tmp_path)

    class RepeatingModel:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            return {
                "tool": "inspect", "path": "solution.py",
                "start_line": 1, "end_line": 20,
            }, {"input_tokens": 10, "output_tokens": 5}

    model = RepeatingModel()
    outcome = ContractBoundedCoder(config(root, calls=10), model).run(
        root, tmp_path / "out"
    )
    assert outcome.status == CoderExitStatus.PLAN_BLOCKED
    assert outcome.reason == "non_progress_loop_after_repeated_duplicate_observations"
    assert outcome.model_calls == 4
    assert model.payloads[0]["interaction_history"] == []
    assert model.payloads[1]["interaction_history"][0]["model_action"]["tool"] == "inspect"
    assert any(
        event["kind"] == "coder.non_progress_loop_stopped"
        for event in outcome.events
    )


def test_hash_preconditioned_append_builds_large_module_safely(tmp_path):
    root = workspace(tmp_path)
    first = "def helper():\n"
    first_sha = hashlib.sha256(first.encode()).hexdigest()
    model = ScriptedModel([
        {"tool": "write", "path": "helper.py", "content": first},
        {
            "tool": "append", "path": "helper.py",
            "content": "    return 1\n", "expected_sha256": first_sha,
        },
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    base_cfg = config(root, calls=4)
    cfg = replace(
        base_cfg,
        action=replace(
            base_cfg.action,
            allowed_write_paths=("solution.py", "helper.py"),
        ),
        config_sha256="",
    )
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert (tmp_path / "out/artifact/helper.py").read_text() == first + "    return 1\n"


def test_changed_workspace_is_validated_when_model_enters_duplicate_loop(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "solution.py"},
    ])
    outcome = ContractBoundedCoder(config(root, calls=8), model).run(
        root, tmp_path / "out"
    )
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.reason == "controller_validation_after_non_progress_loop"
    assert outcome.model_calls == 5


def test_changed_workspace_is_validated_when_output_budget_stops_dialogue(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel(
        [{"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"}],
        usages=[{"input_tokens": 10, "output_tokens": 10}],
    )
    base_cfg = config(root, calls=3, output_tokens=10)
    outcome = ContractBoundedCoder(base_cfg, model).run(
        root, tmp_path / "out"
    )
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.reason == "controller_validation_after_early_termination"
    assert outcome.model_calls == 1
    assert any(
        event["kind"] == "coder.early_termination_validation_finished"
        and event["payload"]["validation_passed"] is True
        for event in outcome.events
    )


def test_invalid_workspace_is_not_promoted_when_output_budget_stops_dialogue(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel(
        [{"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'worse'"}],
        usages=[{"input_tokens": 10, "output_tokens": 10}],
    )
    outcome = ContractBoundedCoder(
        config(root, calls=3, output_tokens=10), model
    ).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert outcome.reason == "output_token_budget_exhausted"
    assert outcome.diagnostic_artifact_relpath == "failed_artifact"
    assert any(
        event["kind"] == "coder.early_termination_validation_finished"
        and event["payload"]["validation_passed"] is False
        for event in outcome.events
    )


class ScriptedModel:
    def __init__(self, responses, usages=None):
        self.responses = iter(responses)
        self.usages = iter(usages or [])
        self.calls = 0

    def respond(self, payload):
        self.calls += 1
        try:
            usage = next(self.usages)
        except StopIteration:
            usage = {"input_tokens": 10, "output_tokens": 5}
        return next(self.responses), usage


def workspace(tmp_path: Path) -> Path:
    root = tmp_path / "source"
    root.mkdir(parents=True)
    (root / "solution.py").write_text(
        "def load_data():\n    return 'bad'\n\ndef predict(x):\n    return x\n\nprint(predict(load_data()))\n",
        encoding="utf-8",
    )
    (root / "test_solution.py").write_text(
        "from solution import load_data, predict\n\ndef test_ok():\n    assert predict(load_data()) == 'good'\n",
        encoding="utf-8",
    )
    (root / "metric.json").write_text('{"metric":"frozen"}\n', encoding="utf-8")
    return root


def config(root: Path, version=CoderVersion.C3, *, calls=6, repairs=2, output_tokens=1000):
    return CoderConfig(
        version=version,
        seed=0,
        model="deepseek/deepseek-v4-pro",
        route="openrouter/provider-default",
        executable_parent="solution.py",
        expected_parent_sha256=hashlib.sha256((root / "solution.py").read_bytes()).hexdigest(),
        action=ScientificActionContract(
            action_id="replace-broken-loader",
            description="repair the loader without changing prediction or metric",
            allowed_write_paths=("solution.py",),
            protected_paths=("metric.json", "test_solution.py"),
            required_symbols=("load_data", "predict"),
            required_call_names=("predict", "load_data"),
            forbidden_import_roots=("scanpy", "loompy"),
            required_source_patterns=(r"return ['\"]good['\"]",),
            preserved_source_patterns=(r"def predict\(x\):",),
        ),
        validation=(
            ValidationCommand("syntax", (sys.executable, "-m", "py_compile", "solution.py")),
            ValidationCommand("unit", (sys.executable, "-m", "pytest", "-q", "test_solution.py")),
            ValidationCommand("smoke", (sys.executable, "solution.py")),
        ),
        max_model_calls=1 if version == CoderVersion.C0 else calls,
        max_tool_calls=10,
        max_input_tokens=10_000,
        max_output_tokens=output_tokens,
        # Generous on purpose: these cases exercise tool/validation behaviour, not
        # budget enforcement, and they spawn real subprocesses on shared hosts.
        # Wall-budget exhaustion is asserted by its own test with an explicit 0.2s.
        max_wall_seconds=600,
        max_repairs=repairs,
    )


def test_c3_persistent_inspect_search_edit_finish(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "inspect", "path": "solution.py", "start_line": 1, "end_line": 4},
        {"tool": "search", "path": ".", "query": "return 'bad'"},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root), model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.implementation_real and outcome.parent_preserved
    assert outcome.changed_paths == ["solution.py"]
    assert [item["tier"] for item in outcome.validation] == ["syntax", "unit", "smoke"]


def test_c3_checkpoint_keeps_read_limit_and_existing_sandbox_requirement(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "metric.json"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "run", "argv": [sys.executable, "-c", "print('new diagnostic')"]},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
    ])
    diagnostic_config = replace(
        config(root, calls=6), allowed_command_prefixes=((sys.executable,),), config_sha256=""
    )
    outcome = ContractBoundedCoder(diagnostic_config, model).run(
        root, tmp_path / "out"
    )
    observations = [
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished"
    ]
    assert any(
        item.get("error") == "implementation_checkpoint_required"
        for item in observations
    )
    # The new command reaches the existing runner, which must still reject this
    # deliberately unsandboxed fixture. Successful execution is covered by the
    # explicit mocked-runner reachability tests, not claimed by this case.
    run_observation = next(
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished" and event["payload"]["tool"] == "run"
    )
    assert run_observation["harness_error"] == "sandbox_not_configured"
    assert run_observation.get("execution_started") is not True
    assert outcome.status == CoderExitStatus.SUCCESS


def test_c3_returns_actionable_feedback_for_malformed_native_tool_call(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {
            "_invalid_json": True,
            "_parse_error": "JSONDecodeError: malformed arguments",
            "_raw_sha256": hashlib.sha256(b"malformed").hexdigest(),
        },
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root, calls=4), model).run(
        root, tmp_path / "out"
    )
    observations = [
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished"
    ]
    assert observations[0]["error"] == "invalid_model_action"
    assert "malformed arguments" in observations[0]["parse_error"]
    assert outcome.status == CoderExitStatus.SUCCESS


def test_c3_final_repair_must_integrate_helper_through_executable(tmp_path):
    root = workspace(tmp_path)
    base_cfg = config(root, calls=6)
    cfg = replace(
        base_cfg,
        action=replace(
            base_cfg.action,
            allowed_write_paths=("solution.py", "helper.py"),
        ),
        config_sha256="",
    )
    model = ScriptedModel([
        {"tool": "write", "path": "helper.py", "content": "VALUE = 'bad'\n"},
        {"tool": "inspect", "path": "helper.py"},
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'almost'"},
        {"tool": "inspect", "path": "helper.py"},
        {"tool": "edit", "path": "helper.py", "old": "'almost'", "new": "'good'"},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
    ])
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    observations = [
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished"
    ]
    assert any(
        item.get("error") == "executable_integration_required"
        for item in observations
    )
    assert outcome.status == CoderExitStatus.SUCCESS
    assert "solution.py" in outcome.changed_paths


def test_c3_failed_controller_validation_forces_a_repair_mutation(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {
            "tool": "edit",
            "path": "solution.py",
            "old": "return 'bad'",
            "new": "return ('good'",
        },
        {"tool": "inspect", "path": "solution.py", "start_line": 1, "end_line": 1},
        {"tool": "inspect", "path": "metric.json"},
        {"tool": "inspect", "path": "solution.py", "start_line": 2, "end_line": 2},
        {
            "tool": "run",
            "argv": [sys.executable, "-c", "print('avoid repair')"],
        },
        {
            "tool": "edit",
            "path": "solution.py",
            "old": "return ('good'",
            "new": "return 'good'",
        },
    ])
    outcome = ContractBoundedCoder(config(root, calls=6), model).run(
        root, tmp_path / "out"
    )
    observations = [
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished"
    ]
    forced = next(
        item for item in observations
        if item.get("error") == "repair_mutation_required"
    )
    assert forced["validation_failure"]["tier"] == "syntax"
    assert outcome.status == CoderExitStatus.SUCCESS


def test_advisory_static_failure_does_not_block_executable_candidate(tmp_path):
    root = workspace(tmp_path)
    base = config(root)
    cfg = replace(
        base,
        validation=(
            base.validation[0],
            ValidationCommand(
                "static", (sys.executable, "-c", "raise SystemExit(1)"),
                blocking=False,
            ),
            *base.validation[1:],
        ),
        config_sha256="",
    )
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.validation[1]["tier"] == "static"
    assert outcome.validation[1]["accepted"] is False
    assert outcome.validation[1]["blocking"] is False
    assert outcome.validation[-1]["tier"] == "smoke"
    assert outcome.validation[-1]["accepted"] is True


def test_repair_can_start_from_hash_bound_failed_multifile_workspace(tmp_path):
    root = workspace(tmp_path)
    diagnostic = tmp_path / "diagnostic"
    shutil.copytree(root, diagnostic)
    (diagnostic / "solution.py").write_text(
        "from helper import VALUE\n\ndef load_data():\n    return []\n\ndef predict(x):\n    return VALUE\n\nif __name__ == '__main__':\n    print(predict(None))\n",
        encoding="utf-8",
    )
    (diagnostic / "helper.py").write_text("VALUE = 'bad'\n", encoding="utf-8")
    (diagnostic / "output").mkdir()
    (diagnostic / "output" / "stale_prediction.csv").write_text(
        "must-not-be-reused\n", encoding="utf-8"
    )
    base = config(root)
    cfg = replace(
        base,
        action=replace(
            base.action,
            allowed_write_paths=("solution.py", "helper.py"),
            required_call_names=(),
            required_source_patterns=(),
        ),
        config_sha256="",
    )
    model = ScriptedModel([
        {"tool": "edit", "path": "helper.py", "old": "'bad'", "new": "'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(cfg, model).run(
        root,
        tmp_path / "out",
        diagnostic_seed_root=diagnostic,
        expected_diagnostic_workspace_sha256=artifact_workspace_sha256(diagnostic),
    )
    assert outcome.status == CoderExitStatus.SUCCESS
    assert {"helper.py", "solution.py"}.issubset(outcome.changed_paths)
    assert not (tmp_path / "out" / "artifact" / "output" / "stale_prediction.csv").exists()
    assert any(
        event["kind"] == "coder.diagnostic_seed_applied"
        for event in outcome.events
    )


def test_repair_rejects_mismatched_diagnostic_workspace_before_model_call(tmp_path):
    root = workspace(tmp_path)
    diagnostic = tmp_path / "diagnostic"
    shutil.copytree(root, diagnostic)
    model = ScriptedModel([])
    with pytest.raises(ValueError, match="diagnostic seed workspace hash mismatch"):
        ContractBoundedCoder(config(root), model).run(
            root,
            tmp_path / "out",
            diagnostic_seed_root=diagnostic,
            expected_diagnostic_workspace_sha256=hashlib.sha256(b"wrong").hexdigest(),
        )
    assert model.calls == 0


def test_c3_read_only_public_mount_is_inspectable_but_not_writable(tmp_path):
    root = workspace(tmp_path)
    public = tmp_path / "public"
    public.mkdir()
    public_file = public / "table.csv"
    public_file.write_text("feature,value\na,1\n", encoding="utf-8")
    cfg = replace(
        config(root),
        read_only_mounts=(("public_data", str(public.resolve())),),
        config_sha256="",
    )
    model = ScriptedModel([
        {"tool": "list", "path": "public_data"},
        {"tool": "inspect", "path": "public_data/table.csv"},
        {"tool": "write", "path": "public_data/table.csv", "content": "tampered"},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    observations = [
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished"
    ]
    assert any(item.get("entries", [{}])[0].get("path") == "public_data/table.csv"
               for item in observations if item.get("entries"))
    assert any("feature,value" in item.get("content", "") for item in observations)
    assert any("path_outside_contract" in item.get("error", "") for item in observations)
    assert public_file.read_text() == "feature,value\na,1\n"
    assert outcome.status == CoderExitStatus.SUCCESS


def test_c3_multi_file_contract_audits_helper_imports(tmp_path):
    root = workspace(tmp_path)
    action = replace(
        config(root).action,
        allowed_write_paths=("solution.py", "*.py"),
    )
    cfg = replace(config(root), action=action, config_sha256="")
    model = ScriptedModel([
        {"tool": "write", "path": "helper.py", "content": "import scanpy\nVALUE = 'good'\n"},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert "forbidden_import_present:scanpy" in outcome.contract_violations
    assert (tmp_path / "out/failed_artifact/helper.py").is_file()
    assert not (tmp_path / "out/artifact/helper.py").exists()


def test_execution_only_bridge_never_invokes_scientific_evaluator(tmp_path):
    task = tmp_path / "task"
    workspace_path = tmp_path / "candidate"
    data = task / "problem/data"
    evaluation = task / "evaluation"
    data.mkdir(parents=True)
    evaluation.mkdir(parents=True)
    workspace_path.mkdir()
    (workspace_path / "run.py").write_text(
        "import os\nfrom pathlib import Path\n"
        "counter=Path('execution_count.txt')\n"
        "counter.write_text(str(int(counter.read_text())+1) if counter.exists() else '1')\n"
        "Path(os.environ['OUTPUT_DIR']).joinpath('answer.txt').write_text('ok')\n"
    )
    marker = task / "SCIENTIFIC_EVALUATOR_WAS_CALLED"
    (evaluation / "evaluator.py").write_text(
        "import json\nfrom pathlib import Path\n"
        f"Path({str(marker)!r}).write_text('called')\n"
        "Path('score.json').write_text(json.dumps({'toy': {'accuracy': 1.0}}))\n"
    )
    (task / "metadata.json").write_text(json.dumps({
        "performance_entries": [{
            "dataset_name": "toy",
            "metrics": [{
                "is_primary": True, "name": "accuracy",
                "metric_direction": "higher_is_better",
                "sota_score": [{"value": 0.5}],
            }],
        }],
    }))
    result = workspace_path / ".swarmrsi/execution_validation.json"
    bridge = Path(__file__).resolve().parents[1] / "scripts/evaluate_naturebench_local_workspace.py"
    completed = subprocess.run([
        sys.executable, str(bridge), "--workspace", str(workspace_path),
        "--result", str(result), "--task-root", str(task),
        "--runtime-python", sys.executable, "--execution-only",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    payload = json.loads(result.read_text())
    assert payload["execution_valid"] is True
    assert payload["score_bearing_evaluation_attempted"] is False
    assert not marker.exists()
    assert (workspace_path / "execution_count.txt").read_text() == "1"

    scientific_result = tmp_path / "scientific.json"
    completed = subprocess.run([
        sys.executable, str(bridge), "--workspace", str(workspace_path),
        "--result", str(scientific_result), "--task-root", str(task),
        "--runtime-python", sys.executable, "--score-existing-output",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    scientific = json.loads(scientific_result.read_text())
    assert scientific["valid"] is True
    assert scientific["metrics"]["aggregate_improvement"] == 1.0
    assert scientific["details"]["candidate_execution_reused"] is True
    assert len(scientific["details"]["reused_execution_evidence_sha256"]) == 64
    assert marker.exists()
    assert (workspace_path / "execution_count.txt").read_text() == "1"

    (workspace_path / "output/answer.txt").write_text("tampered")
    tampered_result = tmp_path / "tampered.json"
    completed = subprocess.run([
        sys.executable, str(bridge), "--workspace", str(workspace_path),
        "--result", str(tampered_result), "--task-root", str(task),
        "--runtime-python", sys.executable, "--score-existing-output",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    tampered = json.loads(tampered_result.read_text())
    assert tampered["valid"] is False
    assert tampered["public_diagnostic"]["category"] == "execution_evidence"

    (workspace_path / "output/answer.txt").write_text("ok")
    (workspace_path / "run.py").write_text(
        (workspace_path / "run.py").read_text() + "# changed after execution\n"
    )
    changed_code_result = tmp_path / "changed_code.json"
    completed = subprocess.run([
        sys.executable, str(bridge), "--workspace", str(workspace_path),
        "--result", str(changed_code_result), "--task-root", str(task),
        "--runtime-python", sys.executable, "--score-existing-output",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    changed_code = json.loads(changed_code_result.read_text())
    assert changed_code["valid"] is False
    assert "run.py does not match" in changed_code["public_diagnostic"]["summary"]


def test_execution_only_bridge_emits_complete_failure_schema(tmp_path):
    task = tmp_path / "task"
    workspace_path = tmp_path / "candidate"
    (task / "problem/data").mkdir(parents=True)
    (task / "evaluation").mkdir(parents=True)
    workspace_path.mkdir()
    (workspace_path / "run.py").write_text("raise RuntimeError('candidate bug')\n")
    (task / "evaluation/evaluator.py").write_text("raise AssertionError('must not run')\n")
    (task / "metadata.json").write_text(json.dumps({"performance_entries": []}))
    result = tmp_path / "execution.json"
    bridge = Path(__file__).resolve().parents[1] / "scripts/evaluate_naturebench_local_workspace.py"
    completed = subprocess.run([
        sys.executable, str(bridge), "--workspace", str(workspace_path),
        "--result", str(result), "--task-root", str(task),
        "--runtime-python", sys.executable, "--execution-only",
    ], capture_output=True, text=True, check=False)
    assert completed.returncode == 0
    payload = json.loads(result.read_text())
    assert payload["valid"] is False
    assert payload["execution_valid"] is False
    assert payload["score_bearing_evaluation_attempted"] is False
    assert payload["public_diagnostic"]["category"] == "candidate_execution"


def test_c3_normalizes_nested_tool_arguments(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "arguments": {
            "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"
        }},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root), model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS


def test_openrouter_exposes_only_controller_allowed_native_tools(monkeypatch):
    from swarmrsi.coder_openrouter import OpenRouterCoderModel

    response_payload = {
        "usage": {},
        "choices": [{"message": {"tool_calls": [{
            "function": {"name": "finish", "arguments": "{}"}
        }]}}],
    }

    class Response:
        def __enter__(self): return self
        def __exit__(self, *_args): return False
        def read(self): return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(
        model="deepseek/deepseek-v4-pro", api_key="test", native_tools=True,
        base_url="https://example.invalid", max_tokens_per_turn=100,
    )
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({
            "version": "C3", "seed": 0,
            "allowed_tools": ["edit", "repair", "run"],
        })
    request_body = json.loads(urlopen.call_args.args[0].data)
    assert {tool["function"]["name"] for tool in request_body["tools"]} == {
        "edit", "repair", "run", "finish", "blocked"
    }


def test_c3_rejects_duplicate_observation_then_recovers(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "solution.py", "start_line": 1},
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root), model).run(root, tmp_path / "out")
    duplicate = [
        event for event in outcome.events
        if event["kind"] == "coder.tool_finished"
        and event["payload"].get("observation", {}).get("error") == "duplicate_observation"
    ]
    assert duplicate and outcome.status == CoderExitStatus.SUCCESS


def test_c3_validation_respects_total_wall_deadline(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root, calls=2)
    object.__setattr__(cfg, "max_wall_seconds", 0.2)
    object.__setattr__(cfg, "validation", (
        ValidationCommand("full", (sys.executable, "-c", "import time; time.sleep(5)"),
                          timeout_seconds=10),
    ))
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    started = time.monotonic()
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert outcome.validation[0]["timed_out"]
    # The command sleeps 5 s; the 0.2 s wall deadline must cut it well before that.
    # Interpreter spawn on a loaded shared node has taken over a second (the
    # revision G qualification on the cluster took 62 minutes for the whole suite),
    # so the bound only has to separate "deadline fired" from "sleep completed".
    assert time.monotonic() - started < 4.0


def test_c3_normalizes_common_edit_aliases(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "file_path": "solution.py",
         "old_string": "return 'bad'", "new_string": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root), model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS


def test_c3_directory_ignore_is_recursive(tmp_path):
    root = workspace(tmp_path)
    (root / "output/a").mkdir(parents=True)
    (root / "output/a/predictions.npy").write_bytes(b"before")
    cfg = config(root)
    object.__setattr__(cfg, "ignored_workspace_patterns", cfg.ignored_workspace_patterns + ("output/**",))
    coder = ContractBoundedCoder(cfg, ScriptedModel([]))
    assert coder._ignored("output/a/predictions.npy")
    assert "output/a/predictions.npy" not in coder._manifest(root)


def test_c3_repairs_after_controller_observed_failure(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return ('good'"},
        {"tool": "finish"},
        {"tool": "diagnose"},
        {"tool": "repair", "path": "solution.py", "old": "return ('good'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    outcome = ContractBoundedCoder(config(root), model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.repaired_after_failure
    assert any(event["kind"] == "coder.repair_requested" for event in outcome.events)


def test_c3_rejects_repeated_diagnosis_without_new_controller_evidence(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "diagnose"},
        {"tool": "diagnose"},
    ])
    outcome = ContractBoundedCoder(
        config(root, calls=2), model
    ).run(root, tmp_path / "out")
    diagnose_events = [
        event for event in outcome.events
        if event["kind"] == "coder.tool_finished"
        and event["payload"]["tool"] == "diagnose"
    ]
    assert diagnose_events[0]["payload"]["ok"] is True
    assert diagnose_events[1]["payload"]["ok"] is False
    assert diagnose_events[1]["payload"]["observation"]["error"] == "duplicate_diagnosis"


@pytest.mark.parametrize("response", [
    {"tool": "edit", "path": "metric.json", "old": "frozen", "new": "changed"},
    {"tool": "edit", "path": "../escape.py", "old": "x", "new": "y"},
])
def test_c3_rejects_out_of_contract_edits_and_rolls_back(tmp_path, response):
    root = workspace(tmp_path)
    outcome = ContractBoundedCoder(
        config(root, calls=1), ScriptedModel([response])
    ).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert outcome.parent_preserved
    assert outcome.final_sha256 == outcome.parent_sha256


def test_c3_rejects_unapproved_run_command(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print('escape')"]}])
    outcome = ContractBoundedCoder(config(root, calls=1), model).run(root, tmp_path / "out")
    event = next(item for item in outcome.events if item["kind"] == "coder.tool_finished")
    assert event["payload"]["observation"]["harness_error"] == "command_not_allowlisted"
    assert outcome.parent_preserved


def test_c3_allowed_model_command_still_fails_closed_without_sandbox(tmp_path):
    root = workspace(tmp_path)
    cfg = replace(
        config(root, calls=1),
        allowed_command_prefixes=((sys.executable,),),
        config_sha256="",
    )
    outcome = ContractBoundedCoder(
        cfg, ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print(1)"]}])
    ).run(root, tmp_path / "out")
    event = next(item for item in outcome.events if item["kind"] == "coder.tool_finished")
    assert event["payload"]["observation"]["harness_error"] == "sandbox_not_configured"
    assert outcome.parent_preserved


def test_c3_token_budget_is_controller_enforced_and_rolls_back(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel(
        [{"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"}],
        usages=[{"input_tokens": 10, "output_tokens": 101}],
    )
    outcome = ContractBoundedCoder(
        config(root, calls=2, output_tokens=100), model
    ).run(root, tmp_path / "out")
    assert outcome.reason == "token_budget_exhausted"
    assert outcome.parent_preserved
    assert outcome.final_sha256 == outcome.parent_sha256


def test_c3_validates_real_change_when_model_call_budget_ends(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
    ])
    outcome = ContractBoundedCoder(config(root, calls=1), model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.reason == "controller_final_validation_at_budget"
    assert outcome.implementation_real
    assert outcome.parent_preserved
    assert outcome.final_sha256 != outcome.parent_sha256


def test_c3_reserves_last_call_for_penultimate_validation_repair(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return ('good'"},
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "repair", "path": "solution.py", "old": "return ('good'", "new": "return 'good'"},
    ])
    outcome = ContractBoundedCoder(
        config(root, calls=3), model
    ).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    assert outcome.reason == "controller_final_validation_at_budget"
    assert outcome.repaired_after_failure
    assert any(event["kind"] == "coder.repair_requested" for event in outcome.events)


def test_c3_event_chain_and_atomic_outputs_are_verifiable(tmp_path):
    root = workspace(tmp_path)
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    out = tmp_path / "out"
    outcome = ContractBoundedCoder(config(root), model).run(root, out)
    events = [json.loads(line) for line in (out / "trajectory.jsonl").read_text().splitlines()]
    for index, event in enumerate(events):
        assert event["sequence"] == index
        assert event["previous_event_sha256"] == (events[index - 1]["event_sha256"] if index else None)
        check = dict(event)
        digest = check.pop("event_sha256")
        from swarmrsi.coder_models import coder_sha256
        assert digest == coder_sha256(check)
    assert outcome.trajectory_sha256
    assert not (out / "result.json.tmp").exists()
    assert not (out / "trajectory.jsonl.tmp").exists()


def test_c3_distinguishes_plan_blocked_and_evaluation_invalid(tmp_path):
    blocked_root = workspace(tmp_path / "blocked")
    blocked = ContractBoundedCoder(
        config(blocked_root), ScriptedModel([{"tool": "blocked", "reason": "contract impossible"}])
    ).run(blocked_root, tmp_path / "blocked-out")
    assert blocked.status == CoderExitStatus.PLAN_BLOCKED

    invalid_root = workspace(tmp_path / "invalid")
    cfg = config(invalid_root)
    object.__setattr__(cfg, "validation", (
        ValidationCommand("syntax", ("definitely-missing-executable",)),
    ))
    model = ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])
    invalid = ContractBoundedCoder(cfg, model).run(invalid_root, tmp_path / "invalid-out")
    assert invalid.status == CoderExitStatus.EVALUATION_INVALID
    assert invalid.parent_preserved
    assert invalid.diagnostic_workspace_sha256
    assert invalid.diagnostic_artifact_relpath == "failed_artifact"
    assert "return 'good'" in (tmp_path / "invalid-out/failed_artifact/solution.py").read_text()
    assert "return 'bad'" in (tmp_path / "invalid-out/artifact/solution.py").read_text()


def test_missing_required_evidence_field_is_candidate_failure_not_harness_error(tmp_path):
    root = workspace(tmp_path)
    command = (
        sys.executable, "-c",
        "from pathlib import Path; import json; p=Path('.swarmrsi/execution.json'); "
        "p.parent.mkdir(exist_ok=True); p.write_text(json.dumps({"
        "'valid':False,'public_diagnostic':{'category':'candidate_execution','summary':'bug'}}))",
    )
    cfg = replace(config(root, calls=2), validation=(ValidationCommand(
        "smoke", command, result_json_path=".swarmrsi/execution.json",
        required_truthy_json_paths=("valid", "execution_valid"),
    ),), config_sha256="")
    outcome = ContractBoundedCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])).run(root, tmp_path / "candidate-failure")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert outcome.validation[-1]["failure_category"] == "candidate_execution"
    assert "harness_error" not in outcome.validation[-1]


def test_validation_harness_retry_reuses_unchanged_candidate(tmp_path):
    root = workspace(tmp_path)
    (root / "validator.py").write_text(
        "from pathlib import Path\nimport json\n"
        "d=Path('.swarmrsi'); d.mkdir(exist_ok=True)\n"
        "c=d/'count'; n=int(c.read_text()) if c.exists() else 0; c.write_text(str(n+1))\n"
        "e=d/'execution.json'\n"
        "e.write_text('{' if n == 0 else json.dumps({'valid':True,'execution_valid':True}))\n"
    )
    command = ValidationCommand(
        "smoke", (sys.executable, "validator.py"),
        result_json_path=".swarmrsi/execution.json",
        required_truthy_json_paths=("valid", "execution_valid"),
        max_harness_retries=1,
    )
    cfg = replace(config(root), validation=(command,), config_sha256="")
    outcome = ContractBoundedCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])).run(root, tmp_path / "retried")
    assert outcome.status == CoderExitStatus.SUCCESS
    retries = [event for event in outcome.events
               if event["kind"] == "coder.validation_harness_retry"]
    assert len(retries) == 1
    assert outcome.validation[-1]["attempt"] == 2


def test_timed_out_validation_without_evidence_routes_as_candidate_timeout(tmp_path):
    root = workspace(tmp_path)
    command = ValidationCommand(
        "smoke", (sys.executable, "-c", "import time; time.sleep(1)"),
        timeout_seconds=0.05,
        result_json_path=".swarmrsi/execution.json",
        required_truthy_json_paths=("valid", "execution_valid"),
        max_harness_retries=2,
    )
    cfg = replace(
        config(root, calls=2), validation=(command,), config_sha256="",
    )
    outcome = ContractBoundedCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])).run(root, tmp_path / "timed-out")
    assert outcome.status == CoderExitStatus.IMPLEMENTATION_FAILED
    assert outcome.validation[-1]["timed_out"] is True
    assert outcome.validation[-1]["failure_category"] == "candidate_timeout"
    assert "harness_error" not in outcome.validation[-1]
    assert not any(event["kind"] == "coder.validation_harness_retry"
                   for event in outcome.events)


def test_parent_hash_is_bound_before_any_model_call(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root)
    (root / "solution.py").write_text("print('mutated')\n", encoding="utf-8")
    with pytest.raises(ValueError, match="parent hash"):
        ContractBoundedCoder(cfg, ScriptedModel([])).run(root, tmp_path / "out")


def test_c3_outcome_is_bound_into_dynamic_research_view(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root)
    # Align the controller action id with the coder action contract.
    action = ActionCandidate(
        id=cfg.action.action_id,
        hypothesis_id="h-loader",
        operator=ActionOperator.REPAIR_PROGRAM,
        dependencies=(ResearchDependency("parent", "code_parent", cfg.expected_parent_sha256),),
        family=ScientificFamily("raw", "loader", "direct", "unit"),
        predicted_program_quality=0.8,
        predicted_information_gain=0.4,
        predicted_reuse_value=0.5,
        predicted_validity=0.9,
        expected_cost=1.0,
    )
    view = DynamicResearchView()
    view.register_action(action)
    view.record_coder_started(
        action_id=action.id,
        executable_parent_sha256=cfg.expected_parent_sha256,
        coder_config_sha256=cfg.config_sha256,
    )
    outcome = ContractBoundedCoder(cfg, ScriptedModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ])).run(root, tmp_path / "out")
    event = view.record_coder_outcome(action_id=action.id, outcome=outcome)
    assert event.previous_event_sha256 == view.events[-2].event_sha256
    assert event.payload["trajectory_sha256"] == outcome.trajectory_sha256
    assert view.projection()["actions"][0]["latest_event"] == "coder.execution_finished"


def test_outer_planner_c3_evaluator_loop_binds_one_parent(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root)
    action = ActionCandidate(
        id=cfg.action.action_id,
        hypothesis_id="h-loader",
        # This fixture edits a valid parent with no failed-action diagnostic.
        # Explicit REPAIR_PROGRAM is reserved for a bound failed implementation.
        operator=ActionOperator.IMPROVE_PROGRAM,
        dependencies=(ResearchDependency("root", "code_parent", cfg.expected_parent_sha256),),
        family=ScientificFamily("raw", "loader", "direct", "unit"),
        predicted_program_quality=0.8,
        predicted_information_gain=0.4,
        predicted_reuse_value=0.5,
        predicted_validity=0.9,
        expected_cost=1.0,
    )
    proposed = ProposedAction(action, cfg.action)

    class Proposer:
        def propose(self, **kwargs):
            return [proposed], {"model_calls": 1, "input_tokens": 20, "output_tokens": 10}

    class Executor:
        def config_sha256(self, **kwargs):
            return cfg.config_sha256

        def execute(self, **kwargs):
            out_dir = tmp_path / "candidate"
            outcome = ContractBoundedCoder(cfg, ScriptedModel([
                {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
                {"tool": "finish"},
            ])).run(root, out_dir)
            return ActionExecution(outcome, out_dir / "artifact", 0.1, True, 1)

    coordinator = VSearchCoordinator(
        version="v1_hypothesis_program_tree",
        strategy=HypothesisProgramTreeStrategy(),
        proposer=Proposer(),
        executor=Executor(),
        root=ProgramNode(
            "root", root, cfg.expected_parent_sha256, "0" * 64, None,
            "root-hypothesis", 0.25, True, None,
            executable_relpath="solution.py",
        ),
        max_model_calls=10,
        max_score_evaluations=2,
        max_tokens=10_000,
    )
    assert coordinator.expand() == 1
    step = coordinator.step()
    assert step.artifact_valid and step.code_parent_id == "root"
    assert coordinator.summary()["best_native_metric"] == 0.25
    assert coordinator.summary()["best_node_id"] == "root"
    assert coordinator.summary()["best_evolved_native_metric"] == 0.1
    assert [event.event_type for event in coordinator.view.events][-5:] == [
        "coder.execution_started", "coder.execution_finished", "program.implementation_observed",
        "search.action_observed",
        "search.recovery_routed",
    ]
    assert step.recovery_route == "new_hypothesis_or_alternative_realization"


def test_controller_queues_one_same_hypothesis_repair_from_failed_artifact(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root)
    action = ActionCandidate(
        id=cfg.action.action_id,
        hypothesis_id="h-loader",
        operator=ActionOperator.IMPROVE_PROGRAM,
        dependencies=(ResearchDependency("root", "code_parent", cfg.expected_parent_sha256),),
        family=ScientificFamily("raw", "loader", "direct", "unit"),
        predicted_program_quality=0.6,
        predicted_information_gain=0.5,
        predicted_reuse_value=0.4,
        predicted_validity=0.5,
        expected_cost=1.0,
        hypothesis_statement="Repair the loader with the same method.",
    )
    proposal = ProposedAction(action, cfg.action)
    cell = tmp_path / "cell"
    failed = cell / "failed_artifact"
    failed.mkdir(parents=True)
    (failed / "run.py").write_text("def broken(:\n", encoding="utf-8")
    diagnostic_sha = artifact_workspace_sha256(failed)

    class Proposer:
        def propose(self, **kwargs):
            return [proposal], {"model_calls": 1, "input_tokens": 1, "output_tokens": 1}

    class Executor:
        def config_sha256(self, **kwargs):
            return cfg.config_sha256

        def execute(self, **kwargs):
            failed_outcome = CoderOutcome(
                status=CoderExitStatus.IMPLEMENTATION_FAILED,
                version=CoderVersion.C3,
                config_sha256=cfg.config_sha256,
                action_id=action.id,
                parent_sha256=cfg.expected_parent_sha256,
                parent_workspace_sha256=artifact_workspace_sha256(root),
                final_sha256=None,
                final_workspace_sha256=artifact_workspace_sha256(root),
                model_calls=2,
                tool_calls=2,
                input_tokens=10,
                output_tokens=10,
                wall_seconds=1.0,
                changed_paths=["run.py"],
                parent_preserved=True,
                reason="implementation_failed_after_validation",
                diagnostic_workspace_sha256=diagnostic_sha,
                diagnostic_artifact_relpath="failed_artifact",
                trajectory_sha256=hashlib.sha256(b"trajectory").hexdigest(),
            )
            failed_outcome.seal()
            return ActionExecution(
                failed_outcome, cell / "artifact", None, False, 0
            )

    coordinator = VSearchCoordinator(
        version="v1_hypothesis_program_tree",
        strategy=HypothesisProgramTreeStrategy(),
        proposer=Proposer(), executor=Executor(),
        root=ProgramNode(
            "root", root, cfg.expected_parent_sha256,
            artifact_workspace_sha256(root), None, "h-root", 0.0, True, None,
        ),
        max_model_calls=10, max_score_evaluations=2, max_tokens=10_000,
    )
    assert coordinator.expand() == 1
    step = coordinator.step()
    assert step.recovery_route == "repair_implementation"
    repairs = [
        item for item in coordinator.actions.values()
        if item.candidate.operator == ActionOperator.REPAIR_PROGRAM
    ]
    assert len(repairs) == 1
    repair = repairs[0]
    assert repair.candidate.hypothesis_id == action.hypothesis_id
    assert repair.candidate.family == action.family
    assert any(
        dependency.kind == "diagnostic_source"
        and dependency.source_id == action.id
        and dependency.source_sha256 == diagnostic_sha
        for dependency in repair.candidate.dependencies
    )
    assert "def broken(:" in repair.contract.description
    assert coordinator.strategy.select(coordinator._snapshot()).action_id == repair.candidate.id


def test_campaign_summary_preserves_scored_root_as_incumbent(tmp_path):
    root = workspace(tmp_path)
    cfg = config(root)

    class NeverCalled:
        def propose(self, **kwargs):
            raise AssertionError("summary must not call proposer")

        def config_sha256(self, **kwargs):
            raise AssertionError("summary must not call executor")

        def execute(self, **kwargs):
            raise AssertionError("summary must not call executor")

    coordinator = VSearchCoordinator(
        version="v1_hypothesis_program_tree",
        strategy=HypothesisProgramTreeStrategy(),
        proposer=NeverCalled(),
        executor=NeverCalled(),
        root=ProgramNode(
            "root", root, cfg.expected_parent_sha256, "0" * 64, None,
            "root-hypothesis", 0.25, True, None,
        ),
        max_model_calls=10,
        max_score_evaluations=2,
        max_tokens=10_000,
    )

    summary = coordinator.summary()
    assert summary["root_native_metric"] == 0.25
    assert summary["best_native_metric"] == 0.25
    assert summary["best_node_id"] == "root"
    assert summary["best_evolved_native_metric"] is None


def test_campaign_checkpoint_binds_strategy_artifact_workspace_and_evidence(tmp_path):
    artifact = tmp_path / "root"
    artifact.mkdir()
    (artifact / "run.py").write_text("def main():\n    return 1\n")
    evidence = tmp_path / "root_evaluation.json"
    evidence.write_text('{"valid":true}\n')
    root = ProgramNode(
        id="root", artifact=artifact,
        executable_sha256=hashlib.sha256((artifact / "run.py").read_bytes()).hexdigest(),
        workspace_sha256=artifact_workspace_sha256(artifact),
        parent_id=None, hypothesis_id="h-root", native_metric=0.0,
        artifact_valid=True, realized_family_sha256=None,
        evaluation_evidence_sha256=hashlib.sha256(evidence.read_bytes()).hexdigest(),
        evaluation_evidence_path=evidence,
    )

    class NeverCalled:
        def propose(self, **kwargs): raise AssertionError
        def config_sha256(self, **kwargs): raise AssertionError
        def execute(self, **kwargs): raise AssertionError

    policy = HypothesisProgramTreeStrategy(seed=0)
    campaign = VSearchCoordinator(
        version=policy.name, strategy=policy, proposer=NeverCalled(),
        executor=NeverCalled(), root=root, max_model_calls=10,
        max_score_evaluations=5, max_tokens=1000,
    )
    payload = json.loads(json.dumps(campaign.checkpoint()))
    restored = VSearchCoordinator.restore(
        strategy=HypothesisProgramTreeStrategy(seed=0), proposer=NeverCalled(),
        executor=NeverCalled(), payload=payload,
    )
    assert json.loads(json.dumps(restored.checkpoint())) == payload
    corrupted = json.loads(json.dumps(payload))
    corrupted["limits"]["max_tokens"] = 999
    with pytest.raises(ValueError, match="checkpoint hash mismatch"):
        VSearchCoordinator.restore(
            strategy=HypothesisProgramTreeStrategy(seed=0), proposer=NeverCalled(),
            executor=NeverCalled(), payload=corrupted,
        )


def container_binding(tmp_path):
    import os
    from test_candidate_execution import IMAGE, PYTHON
    state = tmp_path.resolve() / "candidate_runtime"; state.mkdir(mode=0o700)
    spec = {"image": IMAGE, "python": PYTHON, "cpus": 2, "memory_mb": 1024, "uid": max(1, os.getuid()),
            "gid": max(1, os.getgid()), "pids": 256, "tmp_mb": 128, "shm_mb": 64, "file_mb": 4096,
            "docker_socket": "/var/run/docker.sock"}
    return {"kind": "docker", "spec": spec, "state_root": str(state)}


def container_config(root, binding, public=None):
    return replace(
        config(root, calls=4), allowed_command_prefixes=((sys.executable,),), config_sha256="",
        candidate_runtime=json.dumps(binding, sort_keys=True),
        read_only_mounts=(("public_data", str(public)),) if public is not None else (),
        tool_command_path_rewrites=((sys.executable, "/opt/env/bin/python"),),
    )


def synthetic_record(binding, **overrides):
    return {"status": "candidate_exited", "returncode": 0, "execution_started": True, "timed_out": False,
            "stdout": "1\n", "stderr": "", "cleanup_confirmed": True, "harness_error": None,
            "record_dir": binding["state_root"] + "/" + "c" * 32, "request_sha256": "r" * 64,
            "result_sha256": "s" * 64, **overrides}


def run_observation(outcome):
    return next(
        event["payload"]["observation"] for event in outcome.events
        if event["kind"] == "coder.tool_finished" and event["payload"]["tool"] == "run"
    )


def test_c3_container_binding_routes_sandboxed_commands_through_bound_backend(tmp_path, monkeypatch):
    from swarmrsi import candidate_execution as backend
    root = workspace(tmp_path)
    public = tmp_path.resolve() / "public"; public.mkdir()
    binding = container_binding(tmp_path)
    calls = []
    def execute(seen, *, workspace, public_data, argv, timeout, cwd="/workspace"):
        calls.append({"binding": seen, "workspace": Path(workspace), "live": Path(workspace).is_dir(),
                      "public_data": Path(public_data), "argv": tuple(argv), "cwd": cwd, "timeout": timeout})
        return synthetic_record(seen)
    monkeypatch.setattr(backend, "execute", execute)
    outcome = ContractBoundedCoder(
        container_config(root, binding, public),
        ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print(1)"], "cwd": "."}]),
    ).run(root, tmp_path / "out")
    observation = run_observation(outcome)
    assert observation["returncode"] == 0 and observation["stdout"] == "1\n" and observation["execution_started"] is True
    assert observation["runtime_receipt"] == {"record_dir": binding["state_root"] + "/" + "c" * 32,
                                              "request_sha256": "r" * 64, "result_sha256": "s" * 64}
    call = calls[0]
    assert call["binding"] == binding and call["argv"] == ("/opt/env/bin/python", "-c", "print(1)")
    assert call["cwd"] == "/workspace" and call["public_data"] == public and 0 < call["timeout"] <= 60.0
    assert call["live"] and call["workspace"] != root.resolve() and not call["workspace"].exists()
    assert outcome.parent_preserved


@pytest.mark.parametrize("record,expected", [
    ({"status": "infrastructure_error", "harness_error": "candidate container is missing"}, "candidate container is missing"),
    ({"status": "cancelled", "harness_error": None}, "cancelled"),
])
def test_c3_container_backend_faults_are_harness_errors(tmp_path, monkeypatch, record, expected):
    from swarmrsi import candidate_execution as backend
    root = workspace(tmp_path)
    public = tmp_path.resolve() / "public"; public.mkdir()
    binding = container_binding(tmp_path)
    monkeypatch.setattr(backend, "execute", lambda seen, **kwargs: synthetic_record(seen, returncode=137, **record))
    outcome = ContractBoundedCoder(
        container_config(root, binding, public),
        ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print(1)"]}]),
    ).run(root, tmp_path / "out")
    observation = run_observation(outcome)
    assert observation["harness_error"] == expected and observation["returncode"] is None
    assert outcome.parent_preserved


def test_c3_container_backend_exceptions_and_missing_mount_are_harness_errors(tmp_path, monkeypatch):
    from swarmrsi import candidate_execution as backend
    from swarmrsi.container_runtime import RuntimeUnavailable
    root = workspace(tmp_path)
    public = tmp_path.resolve() / "public"; public.mkdir()
    binding = container_binding(tmp_path)
    def unresolved(seen, **kwargs):
        raise RuntimeUnavailable("unresolved runtime action; inspect/clean it before a new launch")
    monkeypatch.setattr(backend, "execute", unresolved)
    outcome = ContractBoundedCoder(
        container_config(root, binding, public),
        ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print(1)"]}]),
    ).run(root, tmp_path / "out")
    observation = run_observation(outcome)
    assert observation["harness_error"] == "RuntimeUnavailable" and observation["execution_started"] is False
    assert "unresolved" in observation["stderr"] and outcome.parent_preserved
    calls = []
    monkeypatch.setattr(backend, "execute", lambda *a, **k: calls.append(1))
    outcome = ContractBoundedCoder(
        container_config(root, binding),
        ScriptedModel([{"tool": "run", "argv": [sys.executable, "-c", "print(1)"]}]),
    ).run(root, tmp_path / "out2")
    observation = run_observation(outcome)
    assert observation["harness_error"] == "ValueError" and "public_data" in observation["stderr"] and not calls


def test_coder_config_identity_is_unchanged_without_candidate_runtime(tmp_path):
    root = workspace(tmp_path)
    base = config(root)
    assert base.candidate_runtime is None and "candidate_runtime" not in base.canonical()
    encoded = json.dumps(container_binding(tmp_path), sort_keys=True)
    bound = replace(base, candidate_runtime=encoded, config_sha256="")
    assert bound.canonical()["candidate_runtime"] == encoded and bound.config_sha256 != base.config_sha256
    for value in ({"kind": "docker"}, "not json", json.dumps({"kind": "bwrap"}), json.dumps({"kind": "docker"})):
        with pytest.raises((ValueError, TypeError)):
            replace(base, candidate_runtime=value, config_sha256="")


def test_c3_diagnostic_runs_are_bounded_once_implementation_is_due(tmp_path):
    """Revision-A peptide action a0003: 14 distinct probes, one write, no gain."""
    root = workspace(tmp_path)
    # calls=14 makes the checkpoint due at call 7: probes at calls 7 and 8 are
    # the two allowed diagnostics, the ones at 9 and 10 are refused.
    probes = [{"tool": "run", "argv": [sys.executable, "-c", f"print({n})"]} for n in range(8)]
    model = ScriptedModel([
        {"tool": "inspect", "path": "solution.py"},
        {"tool": "inspect", "path": "metric.json"},
        *probes,  # the third probe after the checkpoint is refused
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "run", "argv": [sys.executable, "-c", "print('after implementation')"]},
        {"tool": "finish"},
    ])
    bounded = replace(config(root, calls=14), allowed_command_prefixes=((sys.executable,),),
                      max_tool_calls=20, max_diagnostic_runs_after_checkpoint=2, config_sha256="")
    outcome = ContractBoundedCoder(bounded, model).run(root, tmp_path / "out")
    runs = [event["payload"]["observation"] for event in outcome.events
            if event["kind"] == "coder.tool_finished" and event["payload"]["tool"] == "run"]
    refused = [item for item in runs if item.get("error") == "implementation_checkpoint_required"]
    assert len(refused) == 2 and refused[0]["diagnostic_runs_after_checkpoint"] == 2
    assert "diagnostic runs" in refused[0]["instruction"]
    # Every refused probe is charged as a model turn but never executed.
    assert all(item.get("execution_started") is not True for item in refused)
    # Once the executable changed, a run is validation again and is accepted.
    assert runs[-1].get("error") != "implementation_checkpoint_required"
    assert outcome.status == CoderExitStatus.SUCCESS and outcome.changed_paths == ["solution.py"]
    with pytest.raises(ValueError, match="positive integer"):
        replace(config(root), max_diagnostic_runs_after_checkpoint=0, config_sha256="")


class FlakyModel(ScriptedModel):
    """Raises the given transport errors before each scripted response."""
    def __init__(self, responses, errors):
        super().__init__(responses)
        self.errors = list(errors)

    def respond(self, payload):
        if self.errors:
            raise self.errors.pop(0)
        return super().respond(payload)


def test_transient_transport_failure_is_resent_within_bounds(tmp_path, monkeypatch):
    """Cross-modal revision A lost action a0001 to a single IncompleteRead."""
    monkeypatch.setattr("swarmrsi.coder_engine.time.sleep", lambda seconds: None)
    root = workspace(tmp_path)
    model = FlakyModel([
        {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"},
        {"tool": "finish"},
    ], [RuntimeError("OpenRouter request failed: IncompleteRead(715 bytes read)")])
    bounded = replace(config(root), max_transport_retries=2, config_sha256="")
    outcome = ContractBoundedCoder(bounded, model).run(root, tmp_path / "out")
    assert outcome.status == CoderExitStatus.SUCCESS
    retried = [e for e in outcome.events if e["kind"] == "coder.model_response_retried"]
    assert len(retried) == 1 and "IncompleteRead" in retried[0]["payload"]["error"]
    # A re-send is not a model call: two responses, two calls charged.
    assert outcome.model_calls == 2 and model.calls == 2


@pytest.mark.parametrize("error, retries", [
    (RuntimeError("OpenRouter request failed: HTTP Error 401: Unauthorized"), 2),   # not transient
    (RuntimeError("OpenRouter request failed: IncompleteRead(1 bytes read)"), None),  # legacy: no re-send
])
def test_non_transient_or_legacy_failures_still_end_the_action(tmp_path, monkeypatch, error, retries):
    monkeypatch.setattr("swarmrsi.coder_engine.time.sleep", lambda seconds: None)
    root = workspace(tmp_path)
    model = FlakyModel([{"tool": "finish"}], [error])
    cfg = replace(config(root), max_transport_retries=retries, config_sha256="")
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert outcome.status != CoderExitStatus.SUCCESS
    assert not [e for e in outcome.events if e["kind"] == "coder.model_response_retried"]
    assert any(e["kind"] == "coder.model_response_failed" for e in outcome.events)


def test_transport_retries_are_bounded_and_then_fail_honestly(tmp_path, monkeypatch):
    monkeypatch.setattr("swarmrsi.coder_engine.time.sleep", lambda seconds: None)
    root = workspace(tmp_path)
    errors = [RuntimeError("OpenRouter request failed: RemoteDisconnected") for _ in range(3)]
    model = FlakyModel([{"tool": "finish"}], errors)
    cfg = replace(config(root), max_transport_retries=2, config_sha256="")
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert len([e for e in outcome.events if e["kind"] == "coder.model_response_retried"]) == 2
    assert any(e["kind"] == "coder.model_response_failed" for e in outcome.events)
    with pytest.raises(ValueError, match="\\[0, 12\\]"):
        replace(config(root), max_transport_retries=13, config_sha256="")


def test_write_first_changes_the_instruction_and_leaves_old_config_hashes_alone(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    base = config(root)
    assert "write_first" not in base.canonical()
    eager = dataclasses.replace(base, write_first=True, config_sha256="")
    assert eager.canonical()["write_first"] is True

    class Recording:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            return {"tool": "edit", "path": "solution.py", "old": "return 'bad'",
                    "new": "return 'good'"} if len(self.payloads) == 1 else {"tool": "finish"}, \
                {"input_tokens": 10, "output_tokens": 5}

    instructions = []
    for cfg in (base, eager):
        model = Recording()
        ContractBoundedCoder(cfg, model).run(root, tmp_path / f"out-{cfg.write_first}")
        instructions.append(model.payloads[0]["instruction"])
    assert "run bounded local experiments" in instructions[0]
    assert "Write first" not in instructions[0]
    assert "Write first" in instructions[1]
    assert "only to diagnose a failure" in instructions[1]


def test_task_context_cap_keeps_the_start_when_configured(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    (root / "TASK_CONTEXT.md").write_text("HEAD-MARKER\n" + "x" * 30_000 + "\nTAIL-MARKER\n")
    seen = {}

    class Recording:
        def respond(self, payload):
            seen.setdefault("first", payload)
            return {"tool": "finish"}, {"input_tokens": 10, "output_tokens": 5}

    for cap in (None, 25_000):
        seen.clear()
        cfg = dataclasses.replace(config(root), task_context_chars=cap, config_sha256="")
        assert ("task_context_chars" in cfg.canonical()) == (cap is not None)
        ContractBoundedCoder(cfg, Recording()).run(root, tmp_path / f"out-{cap}")
        text = json.dumps(seen["first"])
        if cap is None:
            assert "TAIL-MARKER" in text and "HEAD-MARKER" not in text
        else:
            assert "HEAD-MARKER" in text and "TAIL-MARKER" not in text and "characters truncated" in text



def test_reasoning_goes_off_for_one_turn_after_reasoning_only_truncations(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    truncated = {"_invalid_json": True, "_parse_error_category": "empty_response",
                 "_transport_error_category": "response_truncated", "_provider_finish_reason": "length"}

    class Truncating:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            if len(self.payloads) <= 2:
                return dict(truncated), {"input_tokens": 10, "output_tokens": 5}
            if len(self.payloads) == 3:
                return {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"}, \
                    {"input_tokens": 10, "output_tokens": 5}
            return {"tool": "finish"}, {"input_tokens": 10, "output_tokens": 5}

    for threshold in (None, 2):
        model = Truncating()
        cfg = dataclasses.replace(config(root, calls=8), reasoning_off_after_truncations=threshold, config_sha256="")
        outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / f"out-{threshold}")
        flags = [bool(p.get("reasoning_off_this_turn")) for p in model.payloads]
        if threshold is None:
            assert not any(flags)
        else:
            assert flags[:4] == [False, False, True, False]
            assert any(e["kind"] == "coder.reasoning_off_for_turn" for e in outcome.events)


def test_openrouter_turns_reasoning_off_on_request_and_records_the_provider():
    response_payload = {
        "provider": "Relace",
        "usage": {"prompt_tokens": 7, "completion_tokens": 3, "completion_tokens_details": {"reasoning_tokens": 0}},
        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
            "function": {"name": "inspect", "arguments": json.dumps({"path": "run.py"})}}]}}],
    }

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(
        api_key="test-only", model="test/model", base_url="https://example.invalid",
        max_tokens_per_turn=100, native_tools=True, reasoning={"effort": "low", "exclude": True},
    )
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        action, _ = model.respond({"version": "C3", "seed": 0, "reasoning_off_this_turn": True})
    body = json.loads(urlopen.call_args.args[0].data)
    assert body["reasoning"] == {"enabled": False}
    assert "reasoning_off_this_turn" not in body["messages"][1]["content"]
    assert action["_provider"] == "Relace" and action["_reasoning_tokens"] == 0
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({"version": "C3", "seed": 0})
    assert json.loads(urlopen.call_args.args[0].data)["reasoning"] == {"effort": "low", "exclude": True}


def test_mutation_limit_rejects_large_writes_and_asks_for_small_steps(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    big = "def load_data():\n    return 'good'\n" + "# pad\n" * 400

    class Stepwise:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            step = len(self.payloads)
            if step == 1:
                return {"tool": "write", "path": "solution.py", "content": big}, {"input_tokens": 10, "output_tokens": 5}
            if step == 2:
                return {"tool": "edit", "path": "solution.py", "old": "return 'bad'", "new": "return 'good'"}, \
                    {"input_tokens": 10, "output_tokens": 5}
            return {"tool": "finish"}, {"input_tokens": 10, "output_tokens": 5}

    cfg = dataclasses.replace(config(root, calls=6), max_mutation_chars=1000, config_sha256="")
    assert cfg.canonical()["max_mutation_chars"] == 1000
    assert "max_mutation_chars" not in config(root).canonical()
    model = Stepwise()
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert "Work in small steps" in model.payloads[0]["instruction"]
    assert "at most 1000 characters" in model.payloads[0]["instruction"]
    rejected = model.payloads[1]["interaction_history"][0]["tool_result"]
    assert rejected["ok"] is False and rejected["observation"]["error"] == "mutation_too_large"
    assert outcome.status == CoderExitStatus.SUCCESS
    plain = Stepwise()
    ContractBoundedCoder(config(root, calls=6), plain).run(root, tmp_path / "out-plain")
    assert "Work in small steps" not in plain.payloads[0]["instruction"]


def test_http_failure_codes_keep_digits_only():
    from swarmrsi.coder_openrouter import http_failure_codes

    exc = RuntimeError("OpenRouter request failed: HTTPError: HTTP Error 403: Forbidden [provider_error_code=403]")
    assert http_failure_codes(exc) == {"http_status": 403, "provider_error_code": 403}
    assert http_failure_codes(RuntimeError("socket closed")) == {}


def test_model_level_reasoning_off_flag_applies_to_the_next_request():
    response_payload = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                            "function": {"name": "inspect", "arguments": json.dumps({"path": "run.py"})}}]}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(api_key="t", model="m", base_url="https://example.invalid",
                                 max_tokens_per_turn=100, native_tools=True, reasoning={"effort": "low", "exclude": True})
    model.reasoning_off_next = True
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({"version": "C3", "seed": 0})
    assert json.loads(urlopen.call_args.args[0].data)["reasoning"] == {"enabled": False}
    model.reasoning_off_next = False
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({"version": "C3", "seed": 0})
    assert json.loads(urlopen.call_args.args[0].data)["reasoning"] == {"effort": "low", "exclude": True}


def test_diff_mode_applies_search_replace_patches_and_returns_failures():
    from swarmrsi.coder_engine import apply_search_replace

    text = "a = 1\nb = 2\n"
    assert apply_search_replace(text, "<<<<<<< SEARCH\nb = 2\n=======\nb = 3\n>>>>>>> REPLACE") == ("a = 1\nb = 3\n", [])
    new, problems = apply_search_replace(text, "<<<<<<< SEARCH\nc = 9\n=======\nc = 1\n>>>>>>> REPLACE")
    assert new is None and "found 0 times" in problems[0]
    assert apply_search_replace(text, "no blocks here")[0] is None


def test_diff_mode_coder_patches_validates_and_retries(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    bad = "<<<<<<< SEARCH\nreturn 'nope'\n=======\nreturn 'good'\n>>>>>>> REPLACE"
    good = "<<<<<<< SEARCH\n    return 'bad'\n=======\n    return 'good'\n>>>>>>> REPLACE"

    class Patching:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            patch = bad if len(self.payloads) == 1 else good
            return {"plan": "fix the loader", "patch": patch}, {"input_tokens": 10, "output_tokens": 5}

    cfg = dataclasses.replace(config(root, calls=4), diff_mode=True, config_sha256="")
    assert cfg.canonical()["diff_mode"] is True and "diff_mode" not in config(root).canonical()
    model = Patching()
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert model.payloads[0]["response_mode"] == "patch"
    assert "SEARCH" in model.payloads[0]["instruction"]
    first = model.payloads[1]["interaction_history"][0]["tool_result"]["observation"]
    assert first["error"] == "patch_not_applied"
    assert outcome.status == CoderExitStatus.SUCCESS and len(model.payloads) == 2
    assert "silent fallback" not in model.payloads[0]["instruction"]


def test_no_silent_fallback_is_opt_in_and_reaches_the_patch_instruction(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    good = "<<<<<<< SEARCH\n    return 'bad'\n=======\n    return 'good'\n>>>>>>> REPLACE"

    class Patching:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            return {"plan": "p", "patch": good}, {"input_tokens": 10, "output_tokens": 5}

    cfg = dataclasses.replace(config(root, calls=4), diff_mode=True, no_silent_fallback=True, config_sha256="")
    assert cfg.canonical()["no_silent_fallback"] is True and "no_silent_fallback" not in config(root).canonical()
    model = Patching()
    ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    assert "silent fallback hides the evidence" in model.payloads[0]["instruction"]
    assert "complete new program" not in model.payloads[0]["instruction"]


def test_fresh_program_for_new_method_is_opt_in_and_diff_mode_only(tmp_path):
    import dataclasses

    root = workspace(tmp_path)
    good = "<<<<<<< SEARCH\n    return 'bad'\n=======\n    return 'good'\n>>>>>>> REPLACE"

    class Patching:
        def __init__(self):
            self.payloads = []

        def respond(self, payload):
            self.payloads.append(payload)
            return {"plan": "p", "patch": good}, {"input_tokens": 10, "output_tokens": 5}

    cfg = dataclasses.replace(config(root, calls=4), diff_mode=True, fresh_program_for_new_method=True,
                              config_sha256="")
    assert cfg.canonical()["fresh_program_for_new_method"] is True
    assert "fresh_program_for_new_method" not in config(root).canonical()
    model = Patching()
    outcome = ContractBoundedCoder(cfg, model).run(root, tmp_path / "out")
    instruction = model.payloads[0]["instruction"]
    assert "complete new program that implements the declared method from scratch" in instruction
    assert "If it refines, return a patch" in instruction
    assert outcome.status == CoderExitStatus.SUCCESS  # a refinement patch is still accepted


def test_patch_mode_request_has_no_tools_and_asks_for_json():
    response_payload = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
                        "choices": [{"finish_reason": "stop", "message": {"content": json.dumps({"plan": "p", "patch": "x"})}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(response_payload).encode()

    model = OpenRouterCoderModel(api_key="t", model="m", base_url="https://example.invalid",
                                 max_tokens_per_turn=100, native_tools=True, reasoning={"effort": "low"})
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        action, _ = model.respond({"version": "C3", "seed": 0, "response_mode": "patch"})
    body = json.loads(urlopen.call_args.args[0].data)
    assert "tools" not in body and body["response_format"] == {"type": "json_object"}
    assert action["patch"] == "x"
