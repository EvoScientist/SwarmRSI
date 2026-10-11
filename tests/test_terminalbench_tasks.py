"""Terminal-Bench custody, trial materialization and adapter guards."""
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest

from evolvcoder import terminalbench_tasks as tb
from evolvcoder.terminalbench_run import TerminalBenchAdapter, ASSETS_SCHEMA
from evolvcoder.configuration import RuntimeConfig

ENV_IMAGE = "harborframework/terminal-bench:demo-environment-abc@sha256:" + "a" * 64
VER_IMAGE = "harborframework/terminal-bench:demo-verifier-abc@sha256:" + "b" * 64

TASK_TOML = f"""
schema_version = "1.0"
artifacts = ["/app/out.txt"]

[task]
name = "terminal-bench/demo"

[metadata]
category = "Software"
expert_time_estimate_hours = 2

[verifier]
environment_mode = "separate"
timeout_sec = 300.0

[verifier.environment]
cpus = 4
memory_mb = 4096
docker_image = "{VER_IMAGE}"

[agent]
timeout_sec = 28800.0

[environment]
build_timeout_sec = 900.0
cpus = 2
memory_mb = 8192
storage_mb = 10240
gpus = 0
docker_image = "{ENV_IMAGE}"
"""


@pytest.fixture
def upstream(tmp_path):
    root = tmp_path / "upstream/demo"
    (root / "environment").mkdir(parents=True)
    (root / "tests").mkdir()
    (root / "solution").mkdir()
    (root / "task.toml").write_text(TASK_TOML)
    (root / "instruction.md").write_text("Write hello into /app/out.txt.")
    (root / "environment/Dockerfile").write_text("FROM python:3.13-slim\n")
    (root / "tests/test_outputs.py").write_text("def test_secret():\n    assert True\n")
    (root / "solution/solve.sh").write_text("#!/bin/bash\necho oracle\n")
    return root


@pytest.fixture
def package(upstream, tmp_path):
    tb.prepare(source=upstream, destination=tmp_path / "tasks/demo")
    return tmp_path / "tasks/demo"


def test_package_seals_instruction_and_hides_the_suite(package, upstream):
    record = tb.verify_assets(package / "custody.json")
    assert record["task_id"] == "demo" and record["search_signal"] == "candidate_self_validation"
    assert record["environment_image"] == ENV_IMAGE and record["verifier_image"] == VER_IMAGE
    shipped = {p.relative_to(package).as_posix() for p in package.rglob("*") if p.is_file()}
    # The hidden suite and the oracle are referenced by digest, never copied.
    assert not any("test_outputs" in name or "solution" in name for name in shipped)
    assert "Write hello" in (package / "problem/instruction.md").read_text()
    metadata = json.loads((package / "metadata.json").read_text())
    assert metadata["leaderboard_comparable"] is True
    assert metadata["hidden_suite_visible_to_candidate"] is False
    assert metadata["resources"]["cpus"] == 2 and metadata["resources"]["memory_mb"] == 8192


@pytest.mark.parametrize("victim", ["problem/instruction.md", "evaluation/task_ref.json", "metadata.json"])
def test_changed_package_file_is_rejected(package, victim):
    (package / victim).write_text("tampered")
    with pytest.raises(ValueError, match="asset changed"):
        tb.verify_assets(package / "custody.json")


def test_unpinned_image_is_refused(upstream, tmp_path):
    (upstream / "task.toml").write_text(TASK_TOML.replace(f'"{ENV_IMAGE}"', '"some/image:latest"'))
    with pytest.raises(ValueError, match="not pinned to a digest"):
        tb.prepare(source=upstream, destination=tmp_path / "tasks/unpinned")


def test_materialize_puts_the_candidate_where_the_oracle_was(package, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / tb.SOLUTION_NAME).write_text("#!/bin/bash\necho hello > /app/out.txt\n")
    (candidate / "helper.py").write_text("x = 1\n")
    record = tb.materialize_trial(package=package, candidate=candidate, destination=tmp_path / "trial")
    trial = tmp_path / "trial"
    assert record["environment_image"] == ENV_IMAGE
    # Environment and hidden suite come from upstream unchanged; solution is ours.
    assert (trial / "tests/test_outputs.py").read_text() == "def test_secret():\n    assert True\n"
    assert "hello" in (trial / "solution/solve.sh").read_text()
    assert (trial / "solution/helper.py").is_file()
    assert (trial / "solution/solve.sh").stat().st_mode & 0o111


def test_materialize_refuses_a_candidate_without_an_entrypoint(package, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / "helper.py").write_text("x = 1\n")
    with pytest.raises(ValueError, match="no regular solve.sh"):
        tb.materialize_trial(package=package, candidate=candidate, destination=tmp_path / "trial")


def test_materialize_refuses_a_moved_upstream_task(package, upstream, tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    (candidate / tb.SOLUTION_NAME).write_text("#!/bin/bash\ntrue\n")
    (upstream / "tests/test_outputs.py").write_text("def test_secret():\n    assert False\n")
    with pytest.raises(ValueError, match="upstream .* changed"):
        tb.materialize_trial(package=package, candidate=candidate, destination=tmp_path / "trial")


def test_reward_and_ctrf_reading(tmp_path):
    logs = tmp_path / "logs"
    logs.mkdir()
    (logs / tb.REWARD_NAME).write_text("1\n")
    assert tb.read_reward(logs) == 1
    (logs / tb.REWARD_NAME).write_text("0.5")
    with pytest.raises(ValueError, match="rewards are 0 or 1"):
        tb.read_reward(logs)
    (logs / tb.CTRF_NAME).write_text(json.dumps({"results": {"summary": {"tests": 4, "passed": 3},
        "tests": [{"name": "a", "status": "passed"}, {"name": "b", "status": "passed"},
                  {"name": "c", "status": "passed"}, {"name": "d", "status": "failed"}]}}))
    report = tb.read_ctrf(logs)
    assert report["fraction"] == 0.75 and report["failed_names"] == ["d"]
    assert report["is_diagnostic_only"] is True


def test_missing_reward_is_an_unfinished_trial_not_a_zero(tmp_path):
    (tmp_path / "logs").mkdir()
    with pytest.raises(FileNotFoundError, match="verifier did not complete"):
        tb.read_reward(tmp_path / "logs")


def test_internal_validation_is_the_search_signal(tmp_path):
    candidate = tmp_path / "candidate"
    candidate.mkdir()
    assert tb.read_internal_validation(candidate) is None
    (candidate / tb.INTERNAL_VALIDATION_NAME).write_text(json.dumps(
        {"metric_name": "self_validation", "direction": "maximize", "estimate": 0.8, "protocol": "ran it"}))
    assert tb.read_internal_validation(candidate)["estimate"] == 0.8
    (candidate / tb.INTERNAL_VALIDATION_NAME).write_text(json.dumps({"estimate": "high"}))
    assert tb.read_internal_validation(candidate) is None


def test_campaign_resources_are_arithmetic(upstream):
    summary = tb.campaign_resources([upstream])
    assert summary["tasks"] == 1 and summary["distinct_images"] == 2
    assert summary["cpus_if_all_concurrent"] == 2
    assert summary["memory_gib_if_all_concurrent"] == 8.0
    assert summary["agent_timeout_hours"] == [8.0]
    assert summary["gpu_tasks"] == []


def _config(tmp_path, package, *, rsi=False):
    assets = tmp_path / "assets.json"
    profile = tmp_path / "profile.json"
    starter = package / "starter"
    profile.write_text(json.dumps({"model": "m", "tasks": {"demo": {
        "starter": str(starter), "custody": str(package / "custody.json"), "search_metric": tb.METRIC}}}))
    assets.write_text(json.dumps({"schema_version": ASSETS_SCHEMA, "profile": str(profile),
        "tasks_root": str(package.parent), "upstream_root": str(tmp_path / "upstream"),
        "harbor_python": sys.executable, "trial_timeout_seconds": 1800,
        "terminal_timeout_seconds": 1800}))
    return SimpleNamespace(benchmark="terminalbench", task="demo", benchmark_config=assets,
        runtime=RuntimeConfig(python=sys.executable), output_root=tmp_path / "runs",
        winner_registry=None, learning=SimpleNamespace(rsi_enabled=rsi, wisdom_enabled=False))


def test_adapter_returns_the_task_and_keeps_the_suite_out_of_reach(tmp_path, package):
    task = TerminalBenchAdapter().task("demo", _config(tmp_path, package))
    assert task.benchmark == "terminalbench" and task.task_id == "demo"
    assert task.metric_name == "self_validation" and task.metric_direction == "maximize"
    assert task.public_data == (package / "problem").resolve()
    assert task.evaluator_config == package / "evaluation/task_ref.json"
    assert "hidden pytest suite" in task.public_description


def test_adapter_refuses_a_diagnostic_package(tmp_path, upstream):
    tb.prepare(source=upstream, destination=tmp_path / "tasks/demo", diagnostic_signal=True)
    package = tmp_path / "tasks/demo"
    with pytest.raises(ValueError, match="not leaderboard-comparable"):
        TerminalBenchAdapter().task("demo", _config(tmp_path, package))


def test_adapter_refuses_unqualified_reuse(tmp_path, package):
    with pytest.raises(ValueError, match="learning/reuse is not yet qualified"):
        TerminalBenchAdapter().task("demo", _config(tmp_path, package, rsi=True))


def test_adapter_refuses_a_per_task_claude_threshold(tmp_path, package):
    config = _config(tmp_path, package)
    profile = json.loads(Path(config.benchmark_config).read_text())
    payload = json.loads(Path(profile["profile"]).read_text())
    payload["tasks"]["demo"]["claude_code_score"] = 0.5
    Path(profile["profile"]).write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="not a per-task Claude Code threshold"):
        TerminalBenchAdapter().task("demo", config)
