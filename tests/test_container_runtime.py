from dataclasses import asdict, replace
import json
import os
from pathlib import Path
import pytest

from evolvcoder import container_runtime as runtime


@pytest.fixture
def spec():
    return runtime.ContainerSpec("sha256:" + "a" * 64, "/opt/env/bin/python", 2, 1024,
                                 max(1, os.getuid()), max(1, os.getgid()), tmp_mb=128)


@pytest.fixture
def action(tmp_path, spec):
    root = tmp_path.resolve()
    for name in ("state", "workspace", "public"):
        (root / name).mkdir(mode=0o700)
    return runtime.make_request(spec, action_id="b" * 32, state_root=root / "state",
                                workspace=root / "workspace", public_data=root / "public",
                                argv=[spec.python, "run.py"], timeout=10)


def install_request(action):
    directory = Path(action["state_root"]) / action["action_id"]
    directory.mkdir(mode=0o700)
    runtime._write(directory / "request.json", action)
    return directory


def inspected(action):
    spec = runtime.ContainerSpec.from_dict(action["spec"])
    command = runtime.create_command(action)
    return {
        "Id": "d" * 64, "Image": spec.image, "Name": "/" + action["name"],
        "Config": {"User": f"{spec.uid}:{spec.gid}", "Entrypoint": ["/usr/bin/env"],
                   "Cmd": command[command.index(spec.image) + 1:], "Labels": {runtime.LABEL: action["action_id"]}},
        "HostConfig": {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                       "NanoCpus": 2_000_000_000, "Memory": 1024**3, "MemorySwap": 1024**3,
                       "PidsLimit": spec.pids, "ShmSize": spec.shm_mb * 1024**2,
                       "CapAdd": None, "CapDrop": ["ALL"], "SecurityOpt": ["no-new-privileges"],
                       "RestartPolicy": {"Name": "no"},
                       "Tmpfs": {"/tmp": "rw,nosuid,nodev,size=134217728,mode=1777"}},
        "Mounts": [{"Destination": "/workspace", "Source": action["workspace"], "RW": True, "Type": "bind"},
                   {"Destination": "/public_data", "Source": action["public_data"], "RW": False, "Type": "bind"}],
        "State": {"Running": False, "Paused": False, "ExitCode": 0, "OOMKilled": False,
                  "Error": "", "StartedAt": "2026-09-15T00:00:00Z"},
    }


@pytest.mark.parametrize("change", [
    {"image": "image:latest"}, {"image": "sha256:abc"}, {"python": "python3"},
    {"python": "/opt/../bin/python"}, {"uid": 0}, {"gid": 0}, {"cpus": True},
    {"memory_mb": 0}, {"tmp_mb": 1000}, {"pids": -1}, {"docker_socket": "tcp://server"},
])
def test_spec_rejects_unsafe_or_implicit_values(spec, change):
    with pytest.raises(ValueError):
        replace(spec, **change)


def test_explicit_image_python_resources_and_environment(action):
    argv = runtime.create_command(action)
    assert "--network" in argv and argv[argv.index("--network") + 1] == "none"
    assert argv[argv.index("--entrypoint") + 1] == "/usr/bin/env"
    assert argv[argv.index(action["spec"]["image"]) + 1] == "-i"
    assert "--privileged" not in argv and "--gpus" not in argv
    assert "--memory-swap" in argv and "--pids-limit" in argv
    assert not any("docker.sock," in arg or "state,dst" in arg for arg in argv)
    assert argv[-2:] == ["/opt/env/bin/python", "run.py"]


@pytest.mark.parametrize("environment", [{"OPENROUTER_API_KEY": "secret"}, {"LD_PRELOAD": "x"},
                                         {"PYTHONPATH": "/controller"}, {"HOME": "/controller"}])
def test_secret_and_startup_environment_not_inherited(spec, environment):
    with pytest.raises(ValueError):
        runtime.candidate_environment(spec, environment)


def test_mounts_and_records_must_be_disjoint(action, spec):
    with pytest.raises(ValueError, match="disjoint"):
        runtime.make_request(spec, action_id="a" * 32, state_root=Path(action["state_root"]),
                             workspace=Path(action["state_root"]), public_data=Path(action["public_data"]),
                             argv=[spec.python], timeout=10)


@pytest.mark.parametrize("config", [{"Volumes": {"/hidden": {}}}, {"Env": ["LD_PRELOAD=/bad.so"]},
                                    {"Env": ["ACCESS_TOKEN=secret"]}])
def test_image_implicit_mounts_and_secrets_rejected(spec, config):
    with pytest.raises(runtime.RuntimeUnavailable):
        runtime.validate_image(spec, {"Id": spec.image, "Config": config})


def test_image_id_must_match(spec):
    with pytest.raises(runtime.RuntimeUnavailable):
        runtime.validate_image(spec, {"Id": "sha256:" + "e" * 64})


def test_observed_daemon_binding_valid(action):
    assert runtime.verify_container(action, inspected(action))["image"] == action["spec"]["image"]


@pytest.mark.parametrize("key,value", [("Privileged", True), ("MemorySwap", -1), ("PidsLimit", 0),
                                     ("NetworkMode", "host"), ("CapAdd", ["SYS_ADMIN"]),
                                     ("Tmpfs", {}), ("Devices", [{"PathOnHost": "/dev/nvidia0"}])])
def test_observed_daemon_binding_not_flags_alone(action, key, value):
    row = inspected(action)
    row["HostConfig"][key] = value
    with pytest.raises(runtime.RuntimeUnavailable):
        runtime.verify_container(action, row)


def test_foreign_container_is_never_killed(action, monkeypatch):
    row = inspected(action)
    row["Config"]["Labels"][runtime.LABEL] = "other"
    calls = []
    def host(argv, **kwargs):
        calls.append(argv)
        return json.dumps([row]) if "inspect" in argv else "123\n"
    monkeypatch.setattr(runtime, "_host", host)
    with pytest.raises(runtime.RuntimeUnavailable, match="not owned"):
        runtime.inspect_owned(action)
    assert len(calls) == 2 and all("kill" not in call for call in calls)


def test_program_exit_125_is_candidate_not_launch_error(action, monkeypatch):
    directory = install_request(action)
    row = inspected(action)
    row["State"]["ExitCode"] = 125
    (directory / "container.cid").write_text(row["Id"])
    runtime._write(directory / "binding.json", {"id": row["Id"]})
    runtime._write(directory / "worker.json", {"execution_started": True, "timed_out": False, "harness_error": None})
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: row)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    assert result["status"] == "candidate_exited" and result["returncode"] == 125
    assert result["cleanup_confirmed"] is True


def test_daemon_unavailable_is_not_cleanup_success(action, monkeypatch):
    directory = install_request(action)
    def unavailable(_):
        raise runtime.RuntimeUnavailable("daemon unavailable")
    monkeypatch.setattr(runtime, "inspect_owned", unavailable)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    assert result["status"] == "infrastructure_error" and not result["cleanup_confirmed"]
    assert result["returncode"] is None


def test_worker_death_stops_owned_container_without_replay(action, monkeypatch):
    directory = install_request(action)
    row = inspected(action)
    (directory / "container.cid").write_text(row["Id"])
    row["State"]["Running"] = True
    calls = []
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: row)
    def host(argv, **kwargs):
        calls.append(argv)
        row["State"]["Running"] = False
        return ""
    monkeypatch.setattr(runtime, "_host", host)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    assert result["cleanup_confirmed"] and result["status"] == "infrastructure_error"
    assert len(calls) == 1 and calls[0][-2:] == ["kill", row["Id"]]
    assert result["harness_error"] == "worker_interrupted_before_receipt"


def test_unresolved_action_prevents_new_launch(action, spec, monkeypatch):
    install_request(action)
    monkeypatch.setattr(runtime, "preflight", lambda _: pytest.fail("must not get past pending record"))
    with pytest.raises(runtime.RuntimeUnavailable, match="unresolved"):
        runtime.launch(spec, state_root=Path(action["state_root"]), workspace=Path(action["workspace"]),
                       public_data=Path(action["public_data"]), argv=[spec.python], timeout=10)


def test_dispatch_failure_is_uncertain_not_replayed(action, spec, monkeypatch):
    monkeypatch.setattr(runtime, "preflight", lambda _: {})
    def host(*args, **kwargs):
        raise runtime.RuntimeUnavailable("lost response")
    monkeypatch.setattr(runtime, "_host", host)
    with pytest.raises(runtime.RuntimeUnavailable, match="dispatch uncertain"):
        runtime.launch(spec, state_root=Path(action["state_root"]), workspace=Path(action["workspace"]),
                       public_data=Path(action["public_data"]), argv=[spec.python], timeout=10)
    attempts = list(Path(action["state_root"]).glob("*/dispatch_error.json"))
    assert len(attempts) == 1 and runtime.status(attempts[0].parent)["status"] == "pending"


def test_supervisor_deadline_and_cleanup_are_independent(action):
    command = runtime._supervisor_command(action, Path(action["state_root"]) / action["action_id"])
    assert "--property=RuntimeMaxSec=30" in command
    assert "--property=KillMode=control-group" in command
    assert any(arg.startswith("--property=ExecStopPost=") and '"cleanup"' in arg for arg in command)
    assert "--wait" not in command and "--scope" not in command


def test_request_tampering_rejected(action):
    directory = install_request(action)
    action["argv"] = ["/bin/sh"]
    runtime._write(directory / "request.json", action)
    with pytest.raises(runtime.RuntimeUnavailable, match="integrity"):
        runtime.status(directory)


def test_forged_cleanup_record_cannot_release_next_action(action, spec, monkeypatch):
    directory = install_request(action)
    runtime._write(directory / "result.json", {"cleanup_confirmed": True})
    monkeypatch.setattr(runtime, "preflight", lambda _: pytest.fail("forged cleanup must not release next action"))
    with pytest.raises(runtime.RuntimeUnavailable):
        runtime.launch(spec, state_root=Path(action["state_root"]), workspace=Path(action["workspace"]),
                       public_data=Path(action["public_data"]), argv=[spec.python], timeout=10)


def test_modified_result_code_and_evidence_are_rejected(action, monkeypatch):
    directory = install_request(action)
    row = inspected(action)
    row["State"]["ExitCode"] = 125
    (directory / "container.cid").write_text(row["Id"])
    runtime._write(directory / "binding.json", {"id": row["Id"]})
    runtime._write(directory / "worker.json", {"execution_started": True, "timed_out": False, "harness_error": None})
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: row)
    runtime.cleanup(directory)
    value = runtime.status(directory)
    value["returncode"] = 0
    runtime._write(directory / "result.json", value)
    with pytest.raises(runtime.RuntimeUnavailable, match="integrity"):
        runtime.status(directory)
    value.pop("result_sha256")
    value["result_sha256"] = runtime._digest(value)
    runtime._write(directory / "result.json", value)
    with pytest.raises(runtime.RuntimeUnavailable, match="contradicts"):
        runtime.status(directory)


@pytest.mark.parametrize("mount", ["workspace", "public_data"])
def test_mount_substitution_prevents_create(action, monkeypatch, mount):
    directory = install_request(action)
    path = Path(action[mount])
    path.rename(path.with_name(path.name + "_original"))
    private = path.with_name("private")
    private.mkdir()
    path.symlink_to(private, target_is_directory=True)
    monkeypatch.setattr(runtime, "_host", lambda *a, **k: pytest.fail("changed mount must not reach daemon"))
    runtime.worker(directory)
    value = runtime._read(directory / "worker.json")
    assert value["execution_started"] is False and value["harness_error"]
    assert not (directory / "create.json").exists()


def test_container_substitution_never_becomes_cleanup_target(action, monkeypatch):
    directory = install_request(action)
    (directory / "container.cid").write_text("e" * 64)
    runtime._write(directory / "binding.json", {"id": "e" * 64})
    row = inspected(action)
    def host(argv, **kwargs):
        assert "kill" not in argv
        return json.dumps([row]) if "inspect" in argv else "123\n"
    monkeypatch.setattr(runtime, "_host", host)
    with pytest.raises(runtime.RuntimeUnavailable, match="ID changed"):
        runtime.inspect_owned(action)


@pytest.mark.parametrize("new_status", ["cancelled", "infrastructure_error"])
def test_non_candidate_status_cannot_choose_weaker_verification(action, monkeypatch, new_status):
    directory = install_request(action)
    row = inspected(action)
    row["State"]["ExitCode"] = 125
    (directory / "container.cid").write_text(row["Id"])
    runtime._write(directory / "binding.json", {"id": row["Id"]})
    runtime._write(directory / "worker.json", {"execution_started": True, "timed_out": False, "harness_error": None})
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: row)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    result.pop("result_sha256")
    result.update(status=new_status, returncode=0)
    result["result_sha256"] = runtime._digest(result)
    runtime._write(directory / "result.json", result)
    with pytest.raises(runtime.RuntimeUnavailable, match="contradicts"):
        runtime.status(directory)


def test_absence_is_recorded_and_verified(action, monkeypatch):
    directory = install_request(action)
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: None)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    assert result["status"] == "infrastructure_error" and result["cleanup_confirmed"]
    assert runtime._read(directory / "cleanup_observation.json")["observation"] == "absent"
    assert "cleanup_observation.json" in result["evidence_refs"]


def test_cancel_of_completed_action_is_read_only(action, monkeypatch):
    directory = install_request(action)
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: None)
    runtime.cleanup(directory)
    result = runtime.status(directory)
    assert runtime.cancel(directory) == result
    assert not (directory / "cancel.json").exists()


def test_submission_fit_flag_is_allowlisted_but_secrets_still_are_not(spec):
    environment = runtime.candidate_environment(spec, {"EVOLVCODER_SUBMISSION_FIT": "full"})
    assert environment["EVOLVCODER_SUBMISSION_FIT"] == "full" and environment["DATA_DIR"] == "/public_data"
    with pytest.raises(ValueError):
        runtime.candidate_environment(spec, {"EVOLVCODER_SUBMISSION_FIT": "full", "OPENROUTER_API_KEY": "x"})
