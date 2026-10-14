"""Shared candidate backend binding: identity, argument transport and receipt binding.

No Docker daemon is used. Every action record is installed through the real
container_runtime request/cleanup/status projection, so each receipt check
below re-derives the receipt exactly as production does.
"""
from dataclasses import asdict
import json
import os
from pathlib import Path
import shutil
from types import SimpleNamespace
import uuid

import pytest

from swarmrsi import candidate_execution as backend
from swarmrsi import container_runtime as runtime
from swarmrsi.configuration import config_from_dict
from test_configuration import payload
from test_container_runtime import inspected, install_request

IMAGE = "sha256:" + "a" * 64
PYTHON = "/opt/env/bin/python"


def container_runtime_dict(**overrides):
    value = {"kind": "container", "image": IMAGE, "cpus": 2, "memory_mb": 1024,
             "container": {"python": PYTHON, "uid": max(1, os.getuid()), "gid": max(1, os.getgid()), "tmp_mb": 128}}
    value.update(overrides)
    return value


def context_for(tmp_path, runtime_dict):
    data = payload(); data["runtime"] = runtime_dict
    return SimpleNamespace(config=config_from_dict(data, base=tmp_path), output_dir=tmp_path.resolve() / "run")


def make_spec():
    return runtime.ContainerSpec(IMAGE, PYTHON, 2, 1024, max(1, os.getuid()), max(1, os.getgid()), tmp_mb=128)


def completed_action(root, monkeypatch, *, exit_code=0, timed_out=False, argv=None,
                     state=None, workspace=None, public=None):
    """Install a finished action whose receipt the real status projection accepts."""
    root = Path(root).resolve()
    state = Path(state) if state else root / "state"
    workspace = Path(workspace) if workspace else root / "workspace"
    public = Path(public) if public else root / "public"
    for directory in (state, workspace, public):
        directory.mkdir(mode=0o700, parents=True, exist_ok=True)
    spec = make_spec()
    request = runtime.make_request(spec, action_id=uuid.uuid4().hex, state_root=state, workspace=workspace,
                                   public_data=public, argv=list(argv or [PYTHON, "run.py"]), timeout=10)
    directory = install_request(request)
    row = inspected(request); row["State"]["ExitCode"] = exit_code
    (directory / "container.cid").write_text(row["Id"])
    runtime._write(directory / "binding.json", {"id": row["Id"]})
    runtime._write(directory / "worker.json", {"execution_started": True, "timed_out": timed_out, "harness_error": None})
    (directory / "stdout.log").write_bytes(b"synthetic stdout\n")
    monkeypatch.setattr(runtime, "inspect_owned", lambda _: row)
    runtime.cleanup(directory)
    binding = {"kind": "docker", "spec": asdict(spec), "state_root": str(state)}
    return binding, runtime.status(directory), SimpleNamespace(spec=spec, state=state, workspace=workspace,
                                                              public=public, directory=directory)


def test_local_runtime_has_no_binding_and_container_binds_explicit_cpu_options(tmp_path):
    assert backend.for_context(context_for(tmp_path, {"kind": "local"})) is None
    context = context_for(tmp_path, container_runtime_dict())
    binding = backend.for_context(context)
    assert binding == {"kind": "docker", "state_root": str(tmp_path.resolve() / "run/candidate_runtime"),
                       "spec": {"image": IMAGE, "python": PYTHON, "cpus": 2, "memory_mb": 1024,
                                "uid": max(1, os.getuid()), "gid": max(1, os.getgid()), "pids": 256, "tmp_mb": 128,
                                "shm_mb": 64, "file_mb": 4096, "docker_socket": "/var/run/docker.sock"}}
    assert backend.validate_binding(binding) == (make_spec(), tmp_path.resolve() / "run/candidate_runtime")


@pytest.mark.parametrize("change", [
    {"container": None}, {"host": "gpu-node"}, {"gpus": 1}, {"memory_mb": None}, {"image": "research:local"},
    {"container": {"python": PYTHON, "uid": 1, "gid": 1, "tmp_mb": 1000}},
])
def test_incomplete_or_remote_container_runtime_is_rejected_not_defaulted(tmp_path, change):
    context = context_for(tmp_path, container_runtime_dict(**change))
    with pytest.raises(ValueError):
        backend.for_context(context)


@pytest.mark.parametrize("change", [
    {"kind": "bwrap"}, {"extra": 1}, {"state_root": "relative/state"}, {"spec": {"image": IMAGE}},
    {"spec": {**container_runtime_dict()["container"], "image": IMAGE, "cpus": 2, "memory_mb": 1024, "docker_socket": "tcp://x"}},
])
def test_binding_shape_is_explicit(tmp_path, change):
    binding = {**backend.for_context(context_for(tmp_path, container_runtime_dict())), **change}
    with pytest.raises((ValueError, TypeError)):
        backend.validate_binding(binding)


def test_binding_state_root_must_be_canonical(tmp_path):
    root = tmp_path.resolve()
    (root / "real").mkdir(mode=0o700)
    (root / "link").symlink_to(root / "real", target_is_directory=True)
    binding = backend.for_context(context_for(tmp_path, container_runtime_dict()))
    with pytest.raises(ValueError, match="canonical"):
        backend.validate_binding({**binding, "state_root": str(root / "link")})


def test_evaluator_arguments_carry_the_exact_binding(tmp_path):
    assert backend.evaluate_arguments(None) == ()
    binding = backend.for_context(context_for(tmp_path, container_runtime_dict()))
    flag, encoded = backend.evaluate_arguments(binding)
    assert flag == "--candidate-runtime-json" and json.loads(encoded) == binding
    assert encoded == json.dumps(binding, sort_keys=True, separators=(",", ":"))


def test_execute_resolves_host_paths_and_forwards_exact_binding(tmp_path, monkeypatch):
    binding = backend.for_context(context_for(tmp_path, container_runtime_dict()))
    root = tmp_path.resolve()
    (root / "workspace").mkdir(); (root / "public").mkdir()
    (root / "link").symlink_to(root / "workspace", target_is_directory=True)
    calls = []
    monkeypatch.setattr(runtime, "run", lambda spec, **kwargs: calls.append((spec, kwargs)) or {"status": "synthetic"})
    assert backend.execute(binding, workspace=root / "link", public_data=root / "public",
                           argv=[PYTHON, "run.py"], timeout=5) == {"status": "synthetic"}
    spec, kwargs = calls[0]
    assert spec == make_spec()
    assert kwargs == {"state_root": Path(binding["state_root"]), "workspace": root / "workspace",
                      "public_data": root / "public", "argv": (PYTHON, "run.py"), "timeout": 5, "cwd": "/workspace"}


def test_receipt_is_rederived_and_bound_to_command_data_and_workspace(tmp_path, monkeypatch):
    binding, receipt, action = completed_action(tmp_path, monkeypatch)
    verified = backend.verify_receipt(binding, receipt, argv=[PYTHON, "run.py"],
                                      public_data=action.public, workspace=action.workspace)
    assert verified == receipt and verified["status"] == "candidate_exited"
    assert verified["returncode"] == 0 and verified["stdout"] == "synthetic stdout\n"
    assert backend.verify_receipt(binding, receipt) == receipt
    for kwargs, message in (({"argv": [PYTHON, "other.py"]}, "different command"),
                            ({"public_data": action.workspace}, "different public data"),
                            ({"workspace": tmp_path.resolve() / "copy"}, "different workspace")):
        with pytest.raises(ValueError, match=message):
            backend.verify_receipt(binding, receipt, **kwargs)


@pytest.mark.parametrize("fault", ["returncode", "stdout", "status", "record_dir", "missing", "spec"])
def test_receipt_edits_and_foreign_records_are_rejected(tmp_path, monkeypatch, fault):
    binding, receipt, action = completed_action(tmp_path, monkeypatch)
    forged = dict(receipt)
    if fault == "returncode":
        forged["returncode"] = 1
    elif fault == "stdout":
        forged["stdout"] = ""
    elif fault == "status":
        forged["status"] = "candidate_timeout"
    elif fault == "record_dir":
        other = tmp_path.resolve() / "other_state"; other.mkdir(mode=0o700)
        shutil.copytree(action.directory, other / action.directory.name)
        forged["record_dir"] = str(other / action.directory.name)
    elif fault == "missing":
        forged = None
    else:
        binding = {**binding, "spec": {**binding["spec"], "cpus": 3}}
    with pytest.raises(ValueError):
        backend.verify_receipt(binding, forged)


def test_sibling_actions_are_not_interchangeable(tmp_path, monkeypatch):
    binding, first, action = completed_action(tmp_path, monkeypatch)
    _, second, _ = completed_action(tmp_path, monkeypatch, argv=[PYTHON, "-c", "pass"], state=action.state,
                                    workspace=action.workspace, public=action.public)
    assert backend.verify_receipt(binding, second, argv=[PYTHON, "-c", "pass"]) == second
    with pytest.raises(ValueError, match="different command"):
        backend.verify_receipt(binding, second, argv=[PYTHON, "run.py"])
    with pytest.raises(ValueError, match="differs from original"):
        backend.verify_receipt(binding, {**second, "record_dir": first["record_dir"]})


def test_unresolved_or_pending_actions_block_receipt_use(tmp_path, monkeypatch):
    root = tmp_path.resolve()
    state, workspace, public = root / "state", root / "workspace", root / "public"
    for directory in (state, workspace, public):
        directory.mkdir(mode=0o700)
    spec = make_spec()
    binding = {"kind": "docker", "spec": asdict(spec), "state_root": str(state)}
    pending = install_request(runtime.make_request(spec, action_id=uuid.uuid4().hex, state_root=state, workspace=workspace,
                                                   public_data=public, argv=[PYTHON, "run.py"], timeout=10))
    receipt = runtime.status(pending)
    assert receipt["status"] == "pending"
    with pytest.raises(runtime.RuntimeUnavailable, match="unresolved"):
        backend.verify_receipt(binding, receipt)
    def unavailable(_):
        raise runtime.RuntimeUnavailable("daemon unavailable")
    monkeypatch.setattr(runtime, "inspect_owned", unavailable)
    runtime.cleanup(pending)
    receipt = runtime.status(pending)
    assert receipt["status"] == "infrastructure_error" and receipt["cleanup_confirmed"] is False
    with pytest.raises(runtime.RuntimeUnavailable, match="unresolved"):
        backend.verify_receipt(binding, receipt)


def test_execute_forwards_an_explicit_environment_only_when_given(tmp_path, monkeypatch):
    binding = backend.for_context(context_for(tmp_path, container_runtime_dict()))
    root = tmp_path.resolve(); (root / "workspace").mkdir(); (root / "public").mkdir()
    calls = []
    monkeypatch.setattr(runtime, "run", lambda spec, **kwargs: calls.append(kwargs) or {"status": "synthetic"})
    backend.execute(binding, workspace=root / "workspace", public_data=root / "public",
                    argv=[PYTHON, "run.py"], timeout=5, environment={"SWARMRSI_SUBMISSION_FIT": "full"})
    assert calls[0]["environment"] == {"SWARMRSI_SUBMISSION_FIT": "full"}
    backend.execute(binding, workspace=root / "workspace", public_data=root / "public", argv=[PYTHON, "run.py"], timeout=5)
    assert "environment" not in calls[1]
