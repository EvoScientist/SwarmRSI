"""Controller-only env/file credentials; all values and scores are synthetic."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evolvcoder import learning_cli, naturebench_cli as cli, naturebench_session as session
from evolvcoder.installed_learning import InstalledValidationRunner, portable_profile
from evolvcoder.retained_results import read_json
from evolvcoder.tree_policy import TreePolicy
from test_installed_learning import config_for
from test_naturebench_entrypoint import inputs, synthetic_root


SECRET_NAME = "ARBITRARY_MODEL_AUTH_FOR_TEST"
SECRET_VALUE = "synthetic-provider-secret-never-persist"


@pytest.fixture(autouse=True)
def preserve_globals(monkeypatch):
    # configure's legacy globals are process-owned. Restore them for isolated
    # tests; public SDK process isolation is tested at the unified entrypoint.
    for name in ("SECRET", "SECRET_ENV", "_CONTROLLER_ENVIRONMENT", "TASKS_ROOT", "ACTIVE_TASK_ID", "RUNTIME", "MODEL_CACHE", "BASE_URL"):
        monkeypatch.setattr(session, name, getattr(session, name))


def use_env(inputs):
    inputs.credential_file = None
    inputs.credential_env = SECRET_NAME
    return inputs


@pytest.mark.parametrize("name", [
    "PATH", "HOME", "SLURM_JOB_ID", "CUDA_VISIBLE_DEVICES", "a=b", "bad-name", "",
    "PYTHONHOME", "PYTHONPATH", "PYTHONINSPECT", "PYTHONWARNINGS", "PYTHONSTARTUP",
    "LD_PRELOAD", "LD_AUDIT", "LD_DEBUG", "DYLD_INSERT_LIBRARIES", "DYLD_PRINT_LIBRARIES",
    "BASH_ENV", "ENV", "GCONV_PATH", "GLIBC_TUNABLES", "MALLOC_TRACE",
    "EVOLVCODER_MODEL_CACHE", "TORCH_HOME", "MPLBACKEND", "SSLKEYLOGFILE",
])
def test_reject_unsafe_credential_names_before_configure(inputs, name):
    use_env(inputs).credential_env = name
    with pytest.raises(ValueError, match="credential-env"):
        cli.configure(inputs)


def test_exclusive_credential_modes_and_old_file_context(inputs):
    _, context = cli.configure(inputs)
    assert "credential_env" not in context
    assert context["credential_file"] == str(inputs.credential_file)
    inputs.credential_env = SECRET_NAME
    with pytest.raises(ValueError, match="exactly one"):
        cli.configure(inputs)
    inputs.credential_env = None
    inputs.credential_file = None
    with pytest.raises(ValueError, match="exactly one"):
        cli.configure(inputs)


def test_provider_private_environment_and_actual_child_isolation(inputs, monkeypatch):
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    monkeypatch.setenv("ANOTHER_PRIVATE_TOKEN", "another-synthetic-secret")
    _, context = cli.configure(use_env(inputs))
    assert context["credential_env"] == SECRET_NAME
    assert SECRET_VALUE not in json.dumps(context)
    assert session.secret_key() == SECRET_VALUE
    with cli.isolated_environment(None):
        assert SECRET_NAME not in os.environ
        assert session.secret_key() == SECRET_VALUE
        # Actual child, no mocked subprocess/environment. This is the same
        # ambient environment inherited by candidate/evaluator orchestration.
        child = subprocess.run([sys.executable, "-c", "import os,json; print(json.dumps(dict(os.environ)))"],
                               capture_output=True, text=True, check=True)
        assert SECRET_NAME not in json.loads(child.stdout)
        assert SECRET_VALUE not in child.stdout
        assert "another-synthetic-secret" not in child.stdout
        with cli.isolated_environment(None):
            assert session.secret_key() == SECRET_VALUE
    assert os.environ[SECRET_NAME] == SECRET_VALUE
    assert session._CONTROLLER_ENVIRONMENT is None


def test_full_inner_initialization_env_and_terminal_reentry_no_key(inputs, monkeypatch, capsys):
    use_env(inputs)
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "Synthetic public task")
    consumed = []
    def models(key, profile, settings):
        consumed.append(key)
        assert SECRET_NAME not in os.environ
        return SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)
    monkeypatch.setattr(session, "build_search_models", models)
    assert cli.execute(inputs) == 0
    assert consumed == [SECRET_VALUE]
    for path in inputs.run_dir.rglob("*"):
        if path.is_file():
            assert SECRET_VALUE.encode() not in path.read_bytes(), str(path)
    monkeypatch.delenv(SECRET_NAME)
    monkeypatch.setattr(session, "secret_key", Mock(side_effect=AssertionError("key access on terminal")))
    assert cli.execute(inputs) == 0
    assert consumed == [SECRET_VALUE]
    assert SECRET_VALUE not in capsys.readouterr().out


def env_runner(inputs):
    use_env(inputs)
    profile_path = inputs.profile.parent / "fixed.json"
    profile_path.write_text(json.dumps(portable_profile(read_json(inputs.profile), inputs.profile)))
    return InstalledValidationRunner(inputs.profile.parent / "validation", profile_path=profile_path,
        runtime_python=Path(inputs.runtime_python), tasks_root=inputs.tasks_root,
        credential_env=SECRET_NAME, cells=[{"task_id": inputs.task_id, "seed": 0}], base_url=inputs.base_url)


def test_installed_env_name_command_and_runtime_only_injection(inputs, monkeypatch):
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    runner = env_runner(inputs)
    _, command = runner._prepare(TreePolicy(), inputs.task_id, 0)
    assert command[command.index("--credential-env") + 1] == SECRET_NAME
    assert "--credential-file" not in command
    assert SECRET_NAME not in runner.environment
    assert SECRET_VALUE not in json.dumps(runner.config) + json.dumps(command)
    # Existence-only preflight must not access the value. Simulate an external
    # key agent that supports presence checks but denies reads.
    class PresenceOnly(dict):
        def __getitem__(self, key):
            raise AssertionError("preflight read a secret value")
    monkeypatch.setattr(session, "_CONTROLLER_ENVIRONMENT", PresenceOnly({SECRET_NAME: SECRET_VALUE}))
    assert runner._preflight_environment()[SECRET_NAME] == "preflight-placeholder-not-a-key"
    monkeypatch.setattr(session, "_CONTROLLER_ENVIRONMENT", None)
    assert runner._environment()[SECRET_NAME] == SECRET_VALUE
    captured = deepcopy(runner.config)
    monkeypatch.setenv(SECRET_NAME, "rotated-synthetic-key")
    resumed = InstalledValidationRunner.resume(runner.root)
    assert resumed.config == captured
    assert resumed._environment()[SECRET_NAME] == "rotated-synthetic-key"
    assert SECRET_VALUE not in (runner.root / "runner.json").read_text()


def test_learning_env_preparation_and_outer_provider_config(inputs, monkeypatch):
    path, config = config_for(inputs)
    config.pop("credential_file")
    config["credential_env"] = SECRET_NAME
    path.write_text(json.dumps(config))
    # Preparation does not need the value and does not invent a secret file.
    monkeypatch.delenv(SECRET_NAME, raising=False)
    record = learning_cli.prepare(path)
    assert record["installed_execution"]["credential_env"] == SECRET_NAME
    _, runner, _ = learning_cli.components(record)
    assert runner.config["execution"]["credential_file"] is None
    constructor = Mock(return_value=SimpleNamespace())
    monkeypatch.setattr(learning_cli, "OpenRouterCoderModel", constructor)
    with pytest.raises(ValueError, match="credential environment variable is unavailable"):
        learning_cli.proposer(record, runner)
    constructor.assert_not_called()
    monkeypatch.setenv(SECRET_NAME, SECRET_VALUE)
    learning_cli.proposer(record, runner)
    assert constructor.call_args.kwargs["api_key"] == SECRET_VALUE
    for file in Path(record["root"]).rglob("*.json"):
        assert SECRET_VALUE not in file.read_text(), str(file)


def test_preflight_missing_env_has_no_provider_or_scoring(inputs, monkeypatch):
    use_env(inputs).preflight = True
    monkeypatch.delenv(SECRET_NAME, raising=False)
    monkeypatch.setattr(session, "secret_key", Mock(side_effect=AssertionError("secret read")))
    monkeypatch.setattr(session, "evaluate_root", Mock(side_effect=AssertionError("scoring")))
    with pytest.raises(ValueError, match="credential environment variable is absent"):
        cli.execute(inputs)
    assert not inputs.run_dir.exists()


def test_parser_env_is_name_not_file():
    parser = argparse.ArgumentParser()
    cli.add_arguments(parser)
    args = parser.parse_args(["--profile", "/profile.json", "--tasks-root", "/tasks", "--task-id", "fixture",
                              "--runtime-python", sys.executable, "--credential-env", SECRET_NAME, "--run-dir", "/runs"])
    assert args.credential_file is None and args.credential_env == SECRET_NAME
