import json
from pathlib import Path

import pytest

from swarmrsi.configuration import ConfigurationError, Limits, RuntimeConfig, config_from_dict, load_config


def payload():
    return {"benchmark": "naturebench", "task": "example", "model": {"model": "provider/model"}}


def test_configuration_is_portable_and_round_trips(tmp_path):
    data = payload()
    data.update(output_root="outputs", benchmark_config="tasks/config.json", winner_registry="winners.json")
    data["learning"] = {"wisdom_enabled": True, "wisdom_snapshot": "wisdom.json", "rsi_enabled": True,
                        "feedback_store": "history", "policy": "policy.json"}
    config = config_from_dict(data, base=tmp_path)
    assert config.output_root == tmp_path / "outputs"
    assert config.learning.feedback_store == tmp_path / "history"
    assert config.model.model == "provider/model"
    assert config_from_dict(config.to_dict(), base=Path("/")) == config
    assert data["output_root"] == "outputs"  # caller-owned input is not mutated


@pytest.mark.parametrize("field", ["model_calls", "input_tokens", "output_tokens", "tool_calls", "evaluations", "wall_seconds", "cost_usd"])
@pytest.mark.parametrize("value", [True, False, 0, -1, "100", float("nan"), float("inf")])
def test_invalid_limits_do_not_silently_disable_enforcement(field, value):
    with pytest.raises(ConfigurationError):
        Limits(**{field: value})


@pytest.mark.parametrize("field", ["model_calls", "input_tokens", "output_tokens", "tool_calls", "evaluations"])
def test_integral_limits_reject_fractions(field):
    with pytest.raises(ConfigurationError):
        Limits(**{field: 1.5})


def test_no_currency_ceiling_is_supported():
    assert Limits(model_calls=100, cost_usd=None).cost_usd is None


@pytest.mark.parametrize("section", [None, "model", "runtime", "limits", "learning"])
def test_unknown_fields_are_errors(tmp_path, section):
    data = payload()
    target = data if section is None else data.setdefault(section, {})
    target["typo"] = 1
    with pytest.raises(ConfigurationError, match="unknown"):
        config_from_dict(data, base=tmp_path)


@pytest.mark.parametrize("changes", [
    {"api_key": "do-not-print-me"},
    {"api_key_env": None},
    {"api_key_file": "secret.json"},
    {"api_key_env": "bad=value"},
    {"base_url": "https://user:password@host/v1"},
    {"base_url": "https://host/v1?api_key=secret"},
    {"base_url": "file:///tmp/test"},
    {"temperature": True},
    {"temperature": float("nan")},
    {"temperature": 3},
    {"max_output_tokens": 0},
    {"concurrency": False},
    {"requests_per_minute": "10"},
])
def test_invalid_model_configuration(tmp_path, changes):
    data = payload(); data["model"].update(changes)
    with pytest.raises(ConfigurationError) as error:
        config_from_dict(data, base=tmp_path)
    assert "do-not-print-me" not in str(error.value)


def test_secret_file_reference_is_not_read(tmp_path):
    data = payload(); data["model"].update(api_key_env=None, api_key_file="not-created.json")
    config = config_from_dict(data, base=tmp_path)
    assert config.model.api_key_file == tmp_path / "not-created.json"
    assert not config.model.api_key_file.exists()


@pytest.mark.parametrize("runtime", [
    {"kind": "unknown"}, {"kind": "container"}, {"kind": "slurm"},
    {"kind": "local", "host": "cluster"}, {"kind": "container", "image": "x", "account": "p"},
    {"cpus": 0}, {"gpus": True}, {"gpus": -1}, {"memory_mb": 1.5},
])
def test_invalid_runtime(tmp_path, runtime):
    data = payload(); data["runtime"] = runtime
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)


def test_slurm_and_container_configuration_parse_without_claiming_readiness(tmp_path):
    for runtime in ({"kind": "container", "image": "research:local"},
                    {"kind": "slurm", "host": "cluster", "account": "project", "partition": "gpu", "gpus": 1}):
        data = payload(); data["runtime"] = runtime
        assert config_from_dict(data, base=tmp_path).runtime.kind == runtime["kind"]


@pytest.mark.parametrize("learning", [
    {"wisdom_enabled": "false"}, {"wisdom_enabled": True},
    {"rsi_enabled": "false"}, {"rsi_enabled": True}, {"rsi_rounds": 0},
])
def test_learning_requires_explicit_persistent_inputs(tmp_path, learning):
    data = payload(); data["learning"] = learning
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)


@pytest.mark.parametrize("value", [True, -1, 1.5, "0"])
def test_seed_is_not_coerced(tmp_path, value):
    data = payload(); data["seed"] = value
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)


def test_json_duplicate_keys_and_nonfinite_values_rejected(tmp_path):
    path = tmp_path / "config.json"
    for content in ('{"task":"a","task":"b"}', '{"x": NaN}', '{"model":{"model":"a","model":"b"}}'):
        path.write_text(content)
        with pytest.raises(ConfigurationError):
            load_config(path)


def test_toml_and_json_have_same_semantics(tmp_path):
    json_path = tmp_path / "config.json"; json_path.write_text(json.dumps(payload()))
    toml_path = tmp_path / "config.toml"
    toml_path.write_text('benchmark = "naturebench"\ntask = "example"\n[model]\nmodel = "provider/model"\n')
    assert load_config(toml_path) == load_config(json_path)


def test_cli_checks_schema_only_and_does_not_read_credentials(tmp_path, capsys, monkeypatch):
    from swarmrsi.cli import main
    monkeypatch.setenv("OPENROUTER_API_KEY", "secret-not-in-output")
    path = tmp_path / "run.json"; path.write_text(json.dumps(payload()))
    assert main(["check-config", str(path)]) == 0
    output = capsys.readouterr().out
    report = json.loads(output)
    assert report["configuration_valid"] and report["runtime_ready"] is None
    assert "secret-not-in-output" not in output
    path.write_text('{"bad":true}')
    assert main(["check-config", str(path)]) == 2


def test_configuration_checks_do_not_modify_input_tree(tmp_path):
    path = tmp_path / "config.json"; path.write_text(json.dumps(payload()))
    before = {p.name: p.read_bytes() for p in tmp_path.iterdir()}
    load_config(path)
    assert {p.name: p.read_bytes() for p in tmp_path.iterdir()} == before


def test_local_python_path_is_config_relative_and_preserves_venv_symlink(tmp_path, monkeypatch):
    config_dir = tmp_path / "config"; config_dir.mkdir()
    launcher = config_dir / ".venv/bin/python"; launcher.parent.mkdir(parents=True)
    launcher.symlink_to("/usr/bin/python3")
    data = payload(); data["runtime"] = {"python": ".venv/bin/python"}
    monkeypatch.chdir(tmp_path)
    config = config_from_dict(data, base=config_dir)
    assert config.runtime.python == str(launcher)
    assert config.runtime.python != str(launcher.resolve())
    assert config_from_dict(config.to_dict(), base=tmp_path) == config


def test_bare_python_command_remains_target_path_lookup(tmp_path):
    data = payload(); data["runtime"] = {"python": "python3.12"}
    assert config_from_dict(data, base=tmp_path).runtime.python == "python3.12"


def test_sdk_cannot_bypass_relative_executable_resolution():
    with pytest.raises(ConfigurationError, match="resolved"):
        RuntimeConfig(python=".venv/bin/python")


@pytest.mark.parametrize("kind", ["container", "slurm"])
def test_remote_relative_executable_uses_target_root_not_local_config(tmp_path, kind):
    runtime = {"kind": kind, "python": ".venv/bin/python", "work_root": "/cluster/project"}
    runtime.update({"image": "research:local"} if kind == "container" else {"account": "p", "partition": "gpu"})
    data = payload(); data["runtime"] = runtime
    config = config_from_dict(data, base=tmp_path)
    assert config.runtime.python == "/cluster/project/.venv/bin/python"
    assert config_from_dict(config.to_dict(), base=tmp_path) == config
    del data["runtime"]["work_root"]
    with pytest.raises(ConfigurationError, match="work_root"):
        config_from_dict(data, base=tmp_path)


@pytest.mark.parametrize("runtime", [
    {"kind": "container", "image": "x", "work_root": "relative"},
    {"kind": "container", "image": "x", "python": "~/venv/bin/python"},
])
def test_remote_paths_do_not_expand_on_observer_host(tmp_path, runtime):
    data = payload(); data["runtime"] = runtime
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)


def test_local_serialization_omits_container_and_container_options_round_trip(tmp_path):
    from swarmrsi.configuration import ContainerOptions
    local = config_from_dict(payload(), base=tmp_path)
    assert local.runtime.container is None and "container" not in local.to_dict()["runtime"]
    data = payload(); data["runtime"] = {"kind": "container", "image": "sha256:" + "a" * 64, "cpus": 2, "memory_mb": 1024,
                                        "container": {"python": "/opt/env/bin/python", "uid": 1000, "gid": 1000}}
    config = config_from_dict(data, base=tmp_path)
    assert config.runtime.container == ContainerOptions(python="/opt/env/bin/python", uid=1000, gid=1000)
    assert config.to_dict()["runtime"]["container"] == {"python": "/opt/env/bin/python", "uid": 1000, "gid": 1000,
                                                        "pids": 256, "tmp_mb": 512, "shm_mb": 64, "file_mb": 4096}
    assert config_from_dict(config.to_dict(), base=tmp_path) == config


@pytest.mark.parametrize("runtime", [
    {"kind": "local", "container": {"python": "/opt/env/bin/python", "uid": 1000, "gid": 1000}},
    {"kind": "container", "image": "x", "container": {"python": "opt/env/bin/python", "uid": 1000, "gid": 1000}},
    {"kind": "container", "image": "x", "container": {"python": "/opt/../bin/python", "uid": 1000, "gid": 1000}},
    {"kind": "container", "image": "x", "container": {"python": "/opt/env/bin/python", "uid": 0, "gid": 1000}},
    {"kind": "container", "image": "x", "container": {"python": "/opt/env/bin/python", "uid": 1000, "gid": 1000, "unknown": 1}},
    {"kind": "container", "image": "x", "container": {"python": "/opt/env/bin/python", "uid": 1000}},
])
def test_container_options_require_container_runtime_and_explicit_fields(tmp_path, runtime):
    data = payload(); data["runtime"] = runtime
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)
