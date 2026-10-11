"""Installed entrypoint engineering tests; all scores/credentials are synthetic."""
import argparse
import json
import os
import subprocess
import sys

import pytest

from evolvcoder import naturebench_cli as cli
from evolvcoder import naturebench_session as session


@pytest.fixture
def inputs(tmp_path):
    starter = tmp_path / "starter"; starter.mkdir()
    (starter / "run.py").write_text("def main():\n    return 0\n")
    task = tmp_path / "tasks/fixture"
    (task / "problem/data").mkdir(parents=True)
    (task / "evaluation").mkdir()
    (task / "evaluation/evaluator.py").write_text("# synthetic authority, never run\n")
    (task / "metadata.json").write_text('{}')
    (task / "problem/README.md").write_text("Synthetic entrypoint test only")
    profile = {"model": "synthetic/model", "target_stop_scope": "overall",
               "wisdom": {"mode": "off"}, "tasks": {"fixture": {
        "starter": "starter", "required_symbols": ["main"], "preserved_source_patterns": [],
        "claude_code_score": -1.0,
        "limits": {"backbone_message_ids": 20, "tool_calls": 20, "score_bearing_evaluations": 4,
                   "wall_time_seconds": 6000, "tokens": {"input": 100000, "output": 30000}}}}}
    path = tmp_path / "profile.json"; path.write_text(json.dumps(profile))
    secret = tmp_path / "credential.json"; secret.write_text('{"api_key":"synthetic-never-sent"}')
    return argparse.Namespace(profile=path, tasks_root=tmp_path / "tasks", task_id="fixture",
                              runtime_python=sys.executable, credential_file=secret,
                              run_dir=tmp_path / "runs/new", model_cache=None,
                              base_url="https://example.invalid", version="v1", seed=0, preflight=False)


def forbid(*args, **kwargs):
    pytest.fail("provider or scoring must not run on this path")


@pytest.mark.parametrize("name", ["continuation_tree", "multi_parent_graph"])
def test_configure_binds_selected_plugin_sources_with_fresh_sessions(inputs, name):
    profile = json.loads(inputs.profile.read_text())
    profile["search"] = {"name": name}
    inputs.profile.write_text(json.dumps(profile))
    protocol, context = cli.configure(inputs)
    selected = session.strategy("v1", 0, protocol)
    assert all(context["source_hashes"][path] == digest
               for path, digest in selected.source_hashes().items())
    assert selected.checkpoint_state() == session.strategy("v1", 0, protocol).checkpoint_state()


@pytest.mark.parametrize("spec", [{"name": "missing_search"},
                                    {"name": "continuation_tree", "parameters": {"typo": 1}}])
def test_configure_rejects_invalid_selected_plugin(inputs, spec):
    profile = json.loads(inputs.profile.read_text())
    profile["search"] = spec
    inputs.profile.write_text(json.dumps(profile))
    with pytest.raises((TypeError, ValueError)):
        cli.configure(inputs)


def test_profile_paths_and_installed_source_authority_are_bound(inputs):
    protocol, context = cli.configure(inputs)
    refs = context["source_hashes"]
    assert "evolvcoder/file_observation.py" in refs
    assert "evolvcoder/naturebench_cli.py" in refs
    assert "evolvcoder/naturebench_evaluator.py" in refs
    assert str(inputs.tasks_root / "fixture/evaluation/evaluator.py") in refs
    assert str(inputs.tasks_root / "fixture/metadata.json") in refs
    assert not any(k.startswith(("src/", "scripts/")) for k in refs)
    assert "synthetic-never-sent" not in json.dumps(context)
    before = refs[str(inputs.tasks_root / "fixture/metadata.json")]
    (inputs.tasks_root / "fixture/metadata.json").write_text('{"changed":true}')
    assert cli.configure(inputs)[1]["source_hashes"][str(inputs.tasks_root / "fixture/metadata.json")] != before


def test_runtime_symlink_is_not_dereferenced(tmp_path):
    launcher = tmp_path / "venv/bin/python"; launcher.parent.mkdir(parents=True)
    launcher.symlink_to(sys.executable)
    assert cli.executable_path(str(launcher)) == launcher
    assert cli.executable_path(str(launcher)) != launcher.resolve()


def test_environment_allowlist_blocks_arbitrary_credential_and_restores(monkeypatch):
    monkeypatch.setenv("MODEL_CREDENTIAL", "synthetic-secret")
    monkeypatch.setenv("UNRELATED_PRIVATE", "synthetic-private")
    monkeypatch.setenv("PYTHONPATH", "/old/checkout")
    monkeypatch.setenv("OMP_NUM_THREADS", "2")
    with cli.isolated_environment(None):
        assert "MODEL_CREDENTIAL" not in os.environ
        assert "UNRELATED_PRIVATE" not in os.environ and "PYTHONPATH" not in os.environ
        assert os.environ["OMP_NUM_THREADS"] == "2"
    assert os.environ["MODEL_CREDENTIAL"] == "synthetic-secret"


def test_lock_rejects_concurrent_entry(inputs):
    with cli.run_lock(inputs.run_dir):
        with pytest.raises(ValueError, match="already owned"):
            with cli.run_lock(inputs.run_dir):
                pytest.fail("second lock acquired")


def test_pending_launch_never_replays(inputs, monkeypatch):
    _, context = cli.configure(inputs)
    inputs.run_dir.mkdir(parents=True)
    (inputs.run_dir / "session_inputs.json").write_text(json.dumps(context))
    (inputs.run_dir / "session_state.json").write_text('{"status":"running"}')
    monkeypatch.setattr(session, "run_session", forbid)
    with pytest.raises(ValueError, match="unfinished prior launch"):
        cli.execute(inputs)


@pytest.mark.parametrize("artifact", ["root/run.py", "planner_batches/batch_000000.request.json"])
def test_missing_state_cannot_replay_prior_artifacts(inputs, monkeypatch, artifact):
    _, context = cli.configure(inputs)
    inputs.run_dir.mkdir(parents=True)
    (inputs.run_dir / "session_inputs.json").write_text(json.dumps(context))
    orphan = inputs.run_dir / artifact; orphan.parent.mkdir(parents=True)
    orphan.write_text("synthetic prior artifact")
    monkeypatch.setattr(session, "run_session", forbid)
    with pytest.raises(ValueError, match="orphan replay"):
        cli.execute(inputs)
    assert orphan.read_text() == "synthetic prior artifact"


@pytest.mark.parametrize("cache", ["tasks", "tasks/fixture", "tasks/fixture/evaluation"])
def test_model_cache_cannot_mount_private_task_assets(inputs, cache):
    inputs.model_cache = inputs.profile.parent / cache
    with pytest.raises(ValueError, match="private evaluator"):
        cli.configure(inputs)


def test_task_symlink_does_not_bypass_output_overlap(inputs):
    task = inputs.tasks_root / "fixture"
    target = inputs.profile.parent / "actual_task"
    task.rename(target); task.symlink_to(target, target_is_directory=True)
    inputs.run_dir = target / "new_output"
    with pytest.raises(ValueError, match="run-dir must not overlap"):
        cli.configure(inputs)


def test_foreign_directory_not_taken_over(inputs, monkeypatch):
    inputs.run_dir.mkdir(parents=True)
    (inputs.run_dir / "user.txt").write_text("keep")
    monkeypatch.setattr(session, "run_session", forbid)
    with pytest.raises(ValueError, match="refusing takeover"):
        cli.execute(inputs)
    assert (inputs.run_dir / "user.txt").read_text() == "keep"


@pytest.mark.parametrize("target", ["starter", "tasks/fixture/problem/data", "runs/new"])
def test_secret_cannot_be_in_candidate_mount(inputs, target):
    inputs.credential_file = inputs.profile.parent / target / "key.json"
    with pytest.raises(ValueError, match="credential-file"):
        cli.configure(inputs)


@pytest.mark.parametrize("key,value", [("backbone_message_ids", "20"), ("tool_calls", True),
                                        ("wall_time_seconds", float("inf")), ("dollar_cap", 2.0)])
def test_unsupported_or_invalid_limits_fail_before_launch(inputs, monkeypatch, key, value):
    p = json.loads(inputs.profile.read_text()); p["tasks"]["fixture"]["limits"][key] = value
    inputs.profile.write_text(json.dumps(p))
    monkeypatch.setattr(session, "run_session", forbid)
    with pytest.raises(ValueError):
        cli.execute(inputs)
    assert not inputs.run_dir.exists()


def synthetic_root(*, starter, run_dir, **kwargs):
    # We do not call a benchmark evaluator; this deliberately synthetic record
    # exercises the real session initialization, checkpoint and final verifier.
    record = {"schema_version": "external-domain-result-v1", "valid": True,
              "execution_valid": True, "score_bearing_evaluation_attempted": True,
              "metrics": {"aggregate_improvement": 0.1},
              "details": {"candidate_executable_sha256": session.sha(starter / "run.py"),
                          "submission_output_attestation": {"fixture": True}}}
    path = run_dir / "root_evaluation.json"; session.atomic_json(path, record)
    return .1, True, .01, session.compact_evaluation_summary(record), session.sha(path)


def test_full_initialization_and_terminal_first_without_provider_or_root_replay(inputs, monkeypatch, capsys):
    from types import SimpleNamespace
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "Synthetic public context")
    calls = []
    def models(*args):
        calls.append("constructed")
        return (SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True))
    monkeypatch.setattr(session, "build_search_models", models)
    assert cli.execute(inputs) == 0
    assert calls == ["constructed"]
    first = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert first["best_native_metric"] == .1 and first["best_evolved_native_metric"] is None
    assert first["reused_terminal"] is False
    assert first["usage"]["model_calls"] == 0
    inputs.credential_file.unlink()  # No secret is needed to read a terminal.
    monkeypatch.setattr(session, "build_search_models", forbid)
    monkeypatch.setattr(session, "evaluate_root", forbid)
    assert cli.execute(inputs) == 0
    assert json.loads(capsys.readouterr().out)["reused_terminal"] is True
    final = inputs.run_dir / "final.json"; value = json.loads(final.read_text())
    value["best_native_metric"] = 999; final.write_text(json.dumps(value))
    with pytest.raises(ValueError):
        cli.execute(inputs)


def test_changed_inputs_rejected_before_provider(inputs, monkeypatch):
    _, context = cli.configure(inputs)
    inputs.run_dir.mkdir(parents=True)
    (inputs.run_dir / "session_inputs.json").write_text(json.dumps(context))
    inputs.seed = 1
    monkeypatch.setattr(session, "run_session", forbid)
    with pytest.raises(ValueError, match="inputs changed"):
        cli.execute(inputs)


def test_real_installed_command_and_preflight_dispatch(inputs):
    environment = dict(os.environ); environment.pop("PYTHONPATH", None)
    result = subprocess.run([sys.executable, "-m", "evolvcoder.cli", "run-naturebench", "--help"],
                            cwd=inputs.profile.parent, env=environment, capture_output=True, text=True)
    assert result.returncode == 0 and "--runtime-python" in result.stdout
    assert "--credential-file" in result.stdout and "--preflight" in result.stdout
    assert "old evocoder" not in result.stderr
    command = [sys.executable, "-m", "evolvcoder.cli", "run-naturebench",
               "--profile", str(inputs.profile), "--tasks-root", str(inputs.tasks_root),
               "--task-id", inputs.task_id, "--runtime-python", str(inputs.runtime_python),
               "--credential-file", str(inputs.credential_file), "--run-dir", str(inputs.run_dir),
               "--base-url", "https://example.invalid", "--preflight"]
    checked = subprocess.run(command, cwd=inputs.profile.parent, env=environment,
                             capture_output=True, text=True, timeout=30)
    assert not inputs.run_dir.exists()
    assert "synthetic-never-sent" not in checked.stdout + checked.stderr
    if checked.returncode == 0:
        assert json.loads(checked.stdout)["zero_model_calls"] is True
    else:
        assert "configured candidate execution backend is unavailable; no model was called" in checked.stderr


def test_cli_sdk_config_is_not_silently_accepted():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--config", "generic.toml"])
    assert exc.value.code == 2


def test_configure_binds_candidate_runtime_and_rejects_overlap_or_model_cache(inputs, tmp_path, monkeypatch):
    from dataclasses import asdict
    from test_candidate_execution import make_spec
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    binding = {"kind": "docker", "spec": asdict(make_spec()), "state_root": str(tmp_path.resolve() / "records")}
    profile = json.loads(inputs.profile.read_text()); profile["candidate_runtime"] = binding
    inputs.profile.write_text(json.dumps(profile))
    protocol, context = cli.configure(inputs)
    assert session.CANDIDATE_RUNTIME == binding and context["candidate_runtime"] == binding
    cache = tmp_path / "cache"; cache.mkdir()
    with pytest.raises(ValueError, match="model-cache"):
        cli.configure(argparse.Namespace(**{**vars(inputs), "model_cache": cache}))
    for state_root in (inputs.run_dir / "candidate_runtime", inputs.tasks_root / "fixture/records", "relative/records"):
        profile["candidate_runtime"] = {**binding, "state_root": str(state_root)}
        inputs.profile.write_text(json.dumps(profile))
        with pytest.raises(ValueError):
            cli.configure(inputs)
    profile["candidate_runtime"] = {**binding, "spec": {**binding["spec"], "image": "research:local"}}
    inputs.profile.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="immutable"):
        cli.configure(inputs)
