"""Shared candidate backend binding used by Coder and benchmark adapters.

This layer does not select hypotheses, edit programs, or grade results. Local
bubblewrap remains at its existing call sites; explicit container bindings use
the same controller-owned backend for inventory, diagnostics and execution.
"""
from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path

from . import container_runtime as docker


def validate_binding(value):
    if not isinstance(value, dict) or set(value) != {"kind", "spec", "state_root"} or value["kind"] != "docker":
        raise ValueError("invalid candidate runtime binding")
    spec = docker.ContainerSpec.from_dict(value["spec"])
    root = Path(value["state_root"])
    if not root.is_absolute() or root.resolve() != root:
        raise ValueError("candidate runtime records require an absolute canonical host path")
    return spec, root


def for_context(context):
    runtime = context.config.runtime
    if runtime.kind == "local":
        return None
    if runtime.kind != "container" or runtime.container is None:
        raise ValueError("container execution requires explicit candidate options; no remote/Slurm fallback")
    if runtime.host is not None or runtime.gpus:
        raise ValueError("this container backend supports local CPU execution; remote/GPU bindings are not implemented")
    options = asdict(runtime.container)
    spec = docker.ContainerSpec(image=runtime.image, cpus=runtime.cpus, memory_mb=runtime.memory_mb, **options)
    return {"kind": "docker", "spec": asdict(spec), "state_root": str(context.output_dir / "candidate_runtime")}


def evaluate_arguments(binding):
    if binding is None:
        return ()
    validate_binding(binding)
    return ("--candidate-runtime-json", json.dumps(binding, sort_keys=True, separators=(",", ":")))


def execute(binding, *, workspace, public_data, argv, timeout, cwd="/workspace", environment=None):
    spec, state_root = validate_binding(binding)
    # Forwarded only when given, so a caller that passes nothing makes exactly the
    # historical call and every receipt sealed against it still verifies.
    extra = {"environment": environment} if environment is not None else {}
    return docker.run(spec, state_root=state_root, workspace=Path(workspace).resolve(),
                      public_data=Path(public_data).resolve(), argv=tuple(argv), timeout=timeout, cwd=cwd, **extra)


def verify_receipt(binding, receipt, *, argv=None, public_data=None, workspace=None):
    """Re-derive a receipt from its controller-owned record and bind the request.

    ``argv`` and ``public_data`` are stable for every caller. ``workspace`` is
    bound only where the executed directory still exists at the same host path
    (the frozen MLE terminal program). A Coder artifact is a retained copy of a
    temporary workspace, so reuse callers bind run.py and output by content.
    """
    spec, state_root = validate_binding(binding)
    if not isinstance(receipt, dict) or not isinstance(receipt.get("record_dir"), str):
        raise ValueError("candidate runtime receipt is missing")
    directory = Path(receipt["record_dir"])
    if directory.parent != state_root or directory.resolve() != directory:
        raise ValueError("candidate runtime receipt is outside the bound controller root")
    actual = docker.status(directory)
    if actual != receipt:
        raise ValueError("candidate runtime receipt differs from original execution")
    request = docker._load_action(directory)
    if request["spec"] != asdict(spec):
        raise ValueError("candidate runtime image or resource identity changed")
    if argv is not None and request["argv"] != list(argv):
        raise ValueError("candidate runtime receipt executed a different command")
    if public_data is not None and request["public_data"] != str(Path(public_data).resolve()):
        raise ValueError("candidate runtime receipt used different public data")
    if workspace is not None and request["workspace"] != str(Path(workspace).resolve()):
        raise ValueError("candidate runtime receipt executed a different workspace")
    if actual.get("cleanup_confirmed") is not True:
        raise docker.RuntimeUnavailable("candidate runtime termination is unresolved")
    return actual
