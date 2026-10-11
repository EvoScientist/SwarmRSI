"""Controller-owned Docker execution with an independent, bounded supervisor.

Linux Docker + a systemd user manager are explicit prerequisites. The caller
does not own the container's lifetime: a transient service owns it, and its
ExecStopPost verifies termination even if the caller or worker is SIGKILLed.
There is no local/unsandboxed fallback and no automatic replay of an action.

This module intentionally uses only the standard library and can be executed
by absolute filename under ``python -I``. Candidate Python is a separate path
inside an immutable image, never a host executable or the controller venv.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
from dataclasses import asdict, dataclass
import fcntl
import hashlib
import json
import math
import os
from pathlib import Path, PurePosixPath
import re
import selectors
import shlex
import signal
import stat
import subprocess
import sys
import time
import uuid


LABEL = "org.evolvcoder.action"
MAX_LOG_BYTES = 8 * 1024 * 1024
EVIDENCE_FILES = frozenset({"worker.json", "binding.json", "final_container.json", "container.cid",
                            "stdout.log", "stderr.log", "cleanup_observation.json", "cancel.json"})
ENV_KEYS = frozenset({
    "LANG", "LC_ALL", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS", "MPLBACKEND", "PYTHONUNBUFFERED", "PYTHONDONTWRITEBYTECODE",
    # Set by the terminal step only, and only under the opt-in submission-fit policy:
    # tells run.py this execution produces the submission, so the model it writes out
    # must be fit on every training row it is given, not on the hold-out's complement.
    "EVOLVCODER_SUBMISSION_FIT",
})


class RuntimeUnavailable(RuntimeError):
    """Trusted infrastructure failure, not evidence against a candidate."""


def _digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def _write(path: Path, value):
    temporary = path.with_name(path.name + "." + uuid.uuid4().hex + ".tmp")
    with temporary.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, indent=2, sort_keys=True)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _read(path: Path):
    return json.loads(path.read_text())


def _positive(value, name, *, integer=True):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be finite and positive")


@dataclass(frozen=True)
class ContainerSpec:
    image: str
    python: str
    cpus: int
    memory_mb: int
    uid: int
    gid: int
    pids: int = 256
    tmp_mb: int = 512
    shm_mb: int = 64
    file_mb: int = 4096
    docker_socket: str = "/var/run/docker.sock"

    def __post_init__(self):
        if not isinstance(self.image, str) or re.fullmatch(r"sha256:[0-9a-f]{64}", self.image) is None:
            raise ValueError("container image must be an immutable local sha256 image ID, not a tag")
        if (not isinstance(self.python, str) or not self.python.startswith("/")
                or ".." in PurePosixPath(self.python).parts or any(c in self.python for c in "\x00\n\r")):
            raise ValueError("candidate python must be an absolute image path")
        for name in ("cpus", "memory_mb", "uid", "gid", "pids", "tmp_mb", "shm_mb", "file_mb"):
            _positive(getattr(self, name), name)
        if self.tmp_mb + self.shm_mb >= self.memory_mb:
            raise ValueError("scratch must fit within the candidate memory ceiling")
        if not isinstance(self.docker_socket, str) or not self.docker_socket.startswith("/") or any(c in self.docker_socket for c in "\x00\n\r"):
            raise ValueError("only an explicit local Unix Docker socket is supported")

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict):
            raise ValueError("container specification must be an object")
        return cls(**value)


def _host_env():
    # Never hand provider credentials or DOCKER_HOST/context overrides to a CLI.
    uid = os.getuid()
    return {"PATH": "/usr/bin:/bin", "HOME": str(Path.home()), "LANG": "C.UTF-8",
            "XDG_RUNTIME_DIR": f"/run/user/{uid}",
            "DBUS_SESSION_BUS_ADDRESS": f"unix:path=/run/user/{uid}/bus"}


def _host(argv, *, timeout=15):
    try:
        result = subprocess.run(_control_argv(argv), capture_output=True, text=True, errors="replace",
                                timeout=timeout, env=_host_env(), check=False)
    except (OSError, subprocess.TimeoutExpired) as exc:
        raise RuntimeUnavailable(f"host control command unavailable: {type(exc).__name__}") from exc
    if result.returncode:
        raise RuntimeUnavailable(f"host control command failed ({result.returncode}): {result.stderr[-2000:]}")
    return result.stdout


def _control_argv(argv):
    """Refresh an already-granted socket group only for the Docker CLI.

    A long-lived user manager can predate docker group membership. ``sg`` is
    the system's normal membership-checked utility; this neither grants group
    membership nor changes the user manager or daemon. Never use a shell to
    interpret candidate argv: shlex.join preserves every argument literally.
    """
    if len(argv) < 3 or argv[:2] != ["/usr/bin/docker", "--host"]:
        return argv
    socket_path = Path(argv[2].removeprefix("unix://"))
    if os.access(socket_path, os.R_OK | os.W_OK) or not socket_path.exists():
        return argv
    import grp
    import pwd
    group = grp.getgrgid(socket_path.stat().st_gid)
    user = pwd.getpwuid(os.getuid())
    if group.gr_gid not in os.getgrouplist(user.pw_name, user.pw_gid):
        raise RuntimeUnavailable("controller account is not authorized for the Docker socket group")
    return ["/usr/bin/sg", group.gr_name, "-c", shlex.join(argv)]


def _docker(spec):
    return ["/usr/bin/docker", "--host", "unix://" + spec.docker_socket]


def _overlap(a, b):
    return a == b or a in b.parents or b in a.parents


def _directory(path):
    path = Path(path)
    if not path.is_absolute() or path.is_symlink() or path.resolve() != path or not path.is_dir():
        raise ValueError("mounts must be existing absolute canonical directories without symlink components")
    if path == Path("/") or path in {Path.home(), Path("/tmp"), Path("/var"), Path("/data"), Path("/usr"), Path("/etc")}:
        raise ValueError("broad host directories must not be mounted into candidates")
    if any(c in str(path) for c in ",\x00\n\r"):
        raise ValueError("mount paths cannot contain Docker CSV delimiters or control characters")
    return path


def _mount_identity(path):
    path = _directory(path)
    metadata = path.stat()
    return {"device": metadata.st_dev, "inode": metadata.st_ino}


def _verify_mounts(request):
    for key in ("workspace", "public_data", "state_root"):
        if _mount_identity(request[key]) != request["mount_identities"][key]:
            raise RuntimeUnavailable(f"{key} directory identity changed after dispatch")


def candidate_environment(spec, extra=None):
    extra = dict(extra or {})
    if set(extra) - ENV_KEYS or any(not isinstance(v, str) or "\x00" in v for v in extra.values()):
        raise ValueError("candidate environment must use the explicit non-secret allowlist")
    environment = {
        "PATH": str(PurePosixPath(spec.python).parent) + ":/usr/local/bin:/usr/bin:/bin",
        "HOME": "/tmp", "TMPDIR": "/tmp", "MPLCONFIGDIR": "/tmp/matplotlib",
        "DATA_DIR": "/public_data", "OUTPUT_DIR": "/workspace/output",
        "LANG": "C.UTF-8", "MPLBACKEND": "Agg", "PYTHONUNBUFFERED": "1",
        "PYTHONDONTWRITEBYTECODE": "1", "CUDA_VISIBLE_DEVICES": "",
        **{name: str(spec.cpus) for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")},
        **extra,
    }
    return environment


def make_request(spec, *, action_id, state_root, workspace, public_data, argv, timeout, cwd="/workspace", environment=None):
    if not isinstance(spec, ContainerSpec):
        raise ValueError("explicit ContainerSpec required")
    if re.fullmatch(r"[0-9a-f]{32}", action_id) is None:
        raise ValueError("invalid action identity")
    _positive(timeout, "timeout", integer=False)
    state_root = _directory(state_root)
    workspace = _directory(workspace)
    public_data = _directory(public_data)
    if any(_overlap(a, b) for a, b in ((workspace, public_data), (workspace, state_root), (public_data, state_root))):
        raise ValueError("public data, workspace and controller runtime records must be disjoint")
    if (not isinstance(argv, (list, tuple)) or not argv
            or any(not isinstance(item, str) or "\x00" in item for item in argv)):
        raise ValueError("candidate argv must be a nonempty sequence of strings")
    if (not isinstance(cwd, str) or not (cwd == "/workspace" or cwd.startswith("/workspace/"))
            or ".." in PurePosixPath(cwd).parts or "\x00" in cwd):
        raise ValueError("candidate cwd must stay inside /workspace")
    value = {"schema_version": "evolvcoder-container-action-v1", "action_id": action_id,
             "name": "evolvcoder-" + action_id, "unit": "evolvcoder-" + action_id + ".service",
             "spec": asdict(spec), "state_root": str(state_root), "workspace": str(workspace),
             "public_data": str(public_data), "argv": list(argv), "cwd": cwd,
             "mount_identities": {key: _mount_identity(path) for key, path in
                                  (("state_root", state_root), ("workspace", workspace), ("public_data", public_data))},
             "environment": candidate_environment(spec, environment),
             "timeout_seconds": float(timeout), "created_at": time.time(),
             "controller_uid": os.getuid(), "controller_python": sys.executable,
             "driver": str(Path(__file__).resolve()),
             "driver_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest()}
    value["deadline"] = value["created_at"] + timeout
    value["request_sha256"] = _digest(value)
    return value


def create_command(request):
    spec = ContainerSpec.from_dict(request["spec"])
    command = [*_docker(spec), "create", "--pull", "never", "--name", request["name"],
               "--cidfile", str(Path(request["state_root"]) / request["action_id"] / "container.cid"),
               "--label", LABEL + "=" + request["action_id"], "--restart", "no",
               "--network", "none", "--read-only", "--user", f"{spec.uid}:{spec.gid}",
               "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
               "--cpus", str(spec.cpus), "--memory", str(spec.memory_mb * 1024**2),
               "--memory-swap", str(spec.memory_mb * 1024**2), "--pids-limit", str(spec.pids),
               "--shm-size", str(spec.shm_mb * 1024**2),
               "--ulimit", f"fsize={spec.file_mb * 1024**2}:{spec.file_mb * 1024**2}",
               "--tmpfs", f"/tmp:rw,nosuid,nodev,size={spec.tmp_mb * 1024**2},mode=1777",
               "--mount", f"type=bind,src={request['public_data']},dst=/public_data,readonly,bind-propagation=rprivate",
               "--mount", f"type=bind,src={request['workspace']},dst=/workspace,bind-propagation=rprivate",
               "--workdir", request["cwd"], "--log-driver", "none", "--no-healthcheck",
               "--entrypoint", "/usr/bin/env", spec.image, "-i"]
    command.extend(f"{k}={v}" for k, v in sorted(request["environment"].items()))
    return command + request["argv"]


def validate_image(spec, image):
    if image.get("Id") != spec.image:
        raise RuntimeUnavailable("configured image identity changed")
    config = image.get("Config") or {}
    if config.get("Volumes"):
        raise RuntimeUnavailable("candidate image must not declare implicit anonymous volumes")
    # env -i clears runtime inheritance, but the env loader itself runs first.
    # Disallow image-controlled loader startup and embedded credential fields.
    for item in config.get("Env") or []:
        key = item.partition("=")[0]
        if key.startswith(("LD_", "DYLD_")) or any(word in key.upper() for word in ("SECRET", "TOKEN", "PASSWORD", "API_KEY")):
            raise RuntimeUnavailable("image includes a forbidden startup/credential environment field")
    return {"image": spec.image, "image_config_sha256": _digest(config),
            "environment_keys": sorted(x.partition("=")[0] for x in config.get("Env") or []),
            "image_content_review_required": True}


def inspect_owned(request):
    spec = ContainerSpec.from_dict(request["spec"])
    # List success with zero matches is proof of absence; arbitrary inspect
    # failure (including daemon outage) is NOT proof that a container stopped.
    names = _host([*_docker(spec), "container", "ls", "-a", "--filter", "name=^/" + request["name"] + "$", "--format", "{{.ID}}"])
    if not names.strip():
        return None
    rows = json.loads(_host([*_docker(spec), "inspect", request["name"]]))
    if len(rows) != 1:
        raise RuntimeUnavailable("container identity is ambiguous")
    row = rows[0]
    if row.get("Name") != "/" + request["name"] or row.get("Config", {}).get("Labels", {}).get(LABEL) != request["action_id"]:
        raise RuntimeUnavailable("container is not owned by this action; refusing cleanup")
    directory = Path(request["state_root"]) / request["action_id"]
    cidfile = directory / "container.cid"
    if not cidfile.is_file() or cidfile.is_symlink():
        raise RuntimeUnavailable("container ID was not durably captured; refusing uncertain ownership")
    cid = cidfile.read_text().strip()
    if re.fullmatch(r"[0-9a-f]{64}", cid) is None or row.get("Id") != cid:
        raise RuntimeUnavailable("container ID changed; refusing substituted container")
    binding = directory / "binding.json"
    if binding.is_file() and _read(binding).get("id") != cid:
        raise RuntimeUnavailable("container ID differs from its original execution binding")
    return row


def verify_container(request, row):
    spec = ContainerSpec.from_dict(request["spec"])
    host, config = row["HostConfig"], row["Config"]
    expected = {"NetworkMode": "none", "ReadonlyRootfs": True, "Privileged": False,
                "NanoCpus": spec.cpus * 1_000_000_000, "Memory": spec.memory_mb * 1024**2,
                "MemorySwap": spec.memory_mb * 1024**2, "PidsLimit": spec.pids,
                "ShmSize": spec.shm_mb * 1024**2}
    if row.get("Image") != spec.image or any(host.get(k) != v for k, v in expected.items()):
        raise RuntimeUnavailable("daemon did not bind requested image/resource/isolation controls")
    if (config.get("User") != f"{spec.uid}:{spec.gid}" or host.get("CapAdd")
            or {x.upper() for x in host.get("CapDrop") or []} != {"ALL"}
            or "no-new-privileges" not in (host.get("SecurityOpt") or [])
            or host.get("RestartPolicy", {}).get("Name") != "no"
            or host.get("PidMode") == "host" or host.get("IpcMode") == "host"):
        raise RuntimeUnavailable("daemon candidate privilege policy differs")
    mounts = {(m["Destination"], m["Source"], m["RW"], m["Type"]) for m in row.get("Mounts", []) if m["Type"] != "tmpfs"}
    if mounts != {("/workspace", request["workspace"], True, "bind"), ("/public_data", request["public_data"], False, "bind")}:
        raise RuntimeUnavailable("daemon candidate mounts differ")
    if (host.get("Tmpfs") != {"/tmp": f"rw,nosuid,nodev,size={spec.tmp_mb * 1024**2},mode=1777"}
            or host.get("Devices") or host.get("DeviceRequests")):
        raise RuntimeUnavailable("unexpected scratch or device access")
    if config.get("Entrypoint") != ["/usr/bin/env"] or config.get("Cmd") != create_command(request)[create_command(request).index(spec.image) + 1:]:
        raise RuntimeUnavailable("daemon candidate command differs")
    return {"id": row["Id"], "image": row["Image"], "host_config": host,
            "user": config["User"], "mounts": row["Mounts"], "state": row["State"]}


def preflight(spec):
    if sys.platform != "linux" or os.getuid() == 0 or spec.uid != os.getuid() or spec.gid != os.getgid():
        raise RuntimeUnavailable("container runtime requires a non-root Linux controller and matching candidate UID/GID")
    socket_path = Path(spec.docker_socket)
    if not socket_path.exists() or not stat.S_ISSOCK(socket_path.stat().st_mode):
        raise RuntimeUnavailable("configured local Docker socket is unavailable")
    _host(["/usr/bin/systemctl", "--user", "show-environment"])
    info = json.loads(_host([*_docker(spec), "info", "--format", "{{json .}}"], timeout=30))
    if not any("seccomp" in item for item in info.get("SecurityOptions", [])):
        raise RuntimeUnavailable("Docker default seccomp support is required")
    image = json.loads(_host([*_docker(spec), "image", "inspect", spec.image]))[0]
    return {**validate_image(spec, image), "supervisor": "systemd-user",
            "docker_version": info.get("ServerVersion"), "security_options": info["SecurityOptions"],
            "controller_python": sys.executable, "candidate_python": spec.python,
            "candidate_memory_bytes": spec.memory_mb * 1024**2,
            "controller_memory_is_separate": True,
            "workspace_total_disk_quota": None, "workspace_disk_limit_semantics": "per_file_fsize_not_total_volume_quota"}


@contextmanager
def _lock(root, *, blocking=False):
    root = Path(root)
    root.mkdir(mode=0o700, parents=True, exist_ok=True)
    _directory(root)
    metadata = root.stat()
    if metadata.st_uid != os.getuid() or stat.S_IMODE(metadata.st_mode) & 0o077:
        raise ValueError("runtime state root must be controller-owned with mode 0700")
    with (root / ".lock").open("a+") as stream:
        fcntl.flock(stream, fcntl.LOCK_EX | (0 if blocking else fcntl.LOCK_NB))
        yield


def _load_action(directory):
    directory = Path(directory)
    _directory(directory)
    request = _read(directory / "request.json")
    expected_hash = request.pop("request_sha256")
    if _digest(request) != expected_hash:
        raise RuntimeUnavailable("runtime action request integrity failed")
    request["request_sha256"] = expected_hash
    if (request["controller_uid"] != os.getuid() or directory.parent != Path(request["state_root"])
            or directory.name != request["action_id"]
            or request["name"] != "evolvcoder-" + directory.name
            or request["unit"] != request["name"] + ".service"):
        raise RuntimeUnavailable("runtime action ownership mismatch")
    return request


def _supervisor_command(request, directory):
    python, driver = request["controller_python"], request["driver"]
    # systemd command-string quoting is NOT shell quoting. Our executable/record
    # paths are quoted with double quotes and specifier/variable expansion is
    # escaped; no shell is involved in either service command.
    def systemd_quote(item):
        return '"' + str(item).replace("\\", "\\\\").replace('"', '\\"').replace("%", "%%").replace("$", "$$") + '"'
    cleanup = " ".join(systemd_quote(x) for x in (python, "-I", "-B", driver, "cleanup", str(directory)))
    return ["/usr/bin/systemd-run", "--user", "--quiet", "--service-type=exec", "--unit=" + request["unit"],
            "--property=Restart=no", "--property=RuntimeMaxSec=" + str(math.ceil(request["timeout_seconds"] + 20)),
            "--property=TimeoutStopSec=45", "--property=KillMode=control-group",
            "--property=WorkingDirectory=" + str(directory), "--property=UMask=0077",
            "--property=ExecStopPost=" + cleanup,
            python, "-I", "-B", driver, "worker", str(directory)]


def launch(spec, *, state_root, workspace, public_data, argv, timeout, cwd="/workspace", environment=None):
    """Start one unique action. Returns its directory; never replays an old one."""
    state_root = Path(state_root)
    with _lock(state_root):
        for path in state_root.glob("*/request.json"):
            prior = status(path.parent)
            if prior.get("cleanup_confirmed") is not True:
                raise RuntimeUnavailable(f"unresolved runtime action {path.parent.name}; inspect/clean it before a new launch")
        qualification = preflight(spec)
        request = make_request(spec, action_id=uuid.uuid4().hex, state_root=state_root,
                               workspace=workspace, public_data=public_data, argv=argv, timeout=timeout,
                               cwd=cwd, environment=environment)
        directory = state_root / request["action_id"]
        directory.mkdir(mode=0o700)
        _write(directory / "request.json", request)
        _write(directory / "preflight.json", qualification)
        command = _supervisor_command(request, directory)
        _write(directory / "dispatch.json", {"argv": command, "attempted_at": time.time()})
        try:
            _host(command, timeout=20)
        except RuntimeUnavailable as exc:
            # Dispatch may have reached systemd despite a lost response. Keep
            # the exact action pending; neither cleanup nor replay is inferred.
            _write(directory / "dispatch_error.json", {"error": str(exc), "state": "uncertain"})
            raise RuntimeUnavailable(f"dispatch uncertain; inspect {directory}") from exc
        return directory


def status(directory):
    directory = Path(directory)
    request = _load_action(directory)
    if (directory / "result.json").is_file():
        result = _read(directory / "result.json")
        if result.get("request_sha256") != request["request_sha256"]:
            raise RuntimeUnavailable("runtime result does not bind this request")
        unsigned = {key: value for key, value in result.items() if key != "result_sha256"}
        if result.get("result_sha256") != _digest(unsigned):
            raise RuntimeUnavailable("runtime result integrity failed")
        references = result.get("evidence_refs")
        present = {name for name in EVIDENCE_FILES if (directory / name).is_file()}
        if not isinstance(references, dict) or set(references) != present or "cleanup_observation.json" not in references:
            raise RuntimeUnavailable("runtime result evidence references are invalid")
        for name, expected in references.items():
            path = directory / name
            if not path.is_file() or path.is_symlink() or hashlib.sha256(path.read_bytes()).hexdigest() != expected:
                raise RuntimeUnavailable("runtime result evidence binding failed")
        expected_result = _project_result(directory, request)
        if {k: v for k, v in unsigned.items() if k != "evidence_refs"} != expected_result:
            raise RuntimeUnavailable("runtime result contradicts controller-owned execution evidence")
        for name in ("stdout", "stderr"):
            if name + ".log" in references and result.get(name) != (directory / (name + ".log")).read_bytes().decode(errors="replace"):
                raise RuntimeUnavailable("runtime output differs from captured stream evidence")
        if result.get("status") in {"candidate_exited", "candidate_timeout"}:
            if not {"worker.json", "binding.json", "final_container.json", "container.cid"} <= set(references):
                raise RuntimeUnavailable("candidate result lacks complete runtime evidence")
            worker_record = _read(directory / "worker.json")
            final = _read(directory / "final_container.json")
            if (result.get("returncode") != final["State"]["ExitCode"]
                    or final["State"].get("Running") or final["State"].get("Paused")
                    or result.get("timed_out") != worker_record.get("timed_out")
                    or worker_record.get("harness_error") or not worker_record.get("execution_started")
                    or final.get("Id") != (directory / "container.cid").read_text().strip()
                    or final.get("Id") != _read(directory / "binding.json").get("id")):
                raise RuntimeUnavailable("candidate result contradicts controller-owned execution evidence")
        return result
    return {"status": "pending", "request_sha256": request["request_sha256"],
            "cleanup_confirmed": False, "record_dir": str(directory)}


def cancel(directory):
    """Stop the exact action service. ExecStopPost performs owned cleanup."""
    directory = Path(directory)
    request = _load_action(directory)
    with _lock(directory):
        existing = status(directory)
        if existing["status"] != "pending":
            return existing
        _write(directory / "cancel.json", {"requested_at": time.time()})
    _host(["/usr/bin/systemctl", "--user", "stop", request["unit"]], timeout=60)
    return status(directory)


def run(spec, **kwargs):
    directory = launch(spec, **kwargs)
    deadline = _load_action(directory)["deadline"] + 100
    try:
        while time.time() < deadline:
            result = status(directory)
            if result["status"] != "pending":
                return result
            time.sleep(0.1)
    except BaseException:
        # A Python-level cancellation can stop promptly. SIGKILL cannot enter
        # here; the independent unit still enforces the original deadline.
        try:
            cancel(directory)
        except RuntimeUnavailable:
            pass
        raise
    raise RuntimeUnavailable(f"supervisor completion is unresolved; do not replay {directory}")


def _capture(process, directory, deadline):
    streams = selectors.DefaultSelector()
    tails, sizes = {"stdout": b"", "stderr": b""}, {"stdout": 0, "stderr": 0}
    for name in tails:
        streams.register(getattr(process, name), selectors.EVENT_READ, name)
    timed_out = False
    try:
        while streams.get_map():
            if time.time() >= deadline:
                timed_out = True
                break
            for key, _ in streams.select(min(0.2, max(0.001, deadline - time.time()))):
                chunk = os.read(key.fileobj.fileno(), 65536)
                if not chunk:
                    streams.unregister(key.fileobj)
                    continue
                name = key.data
                sizes[name] += len(chunk)
                tails[name] = (tails[name] + chunk)[-MAX_LOG_BYTES:]
    finally:
        streams.close()
        for name, content in tails.items():
            (directory / (name + ".log")).write_bytes(content)
    return timed_out, sizes


def worker(directory):
    directory = Path(directory)
    request = _load_action(directory)
    spec = ContainerSpec.from_dict(request["spec"])
    outcome = {"execution_started": False, "timed_out": False, "harness_error": None}
    process = None
    try:
        if hashlib.sha256(Path(__file__).read_bytes()).hexdigest() != request["driver_sha256"]:
            raise RuntimeUnavailable("runtime driver source changed after dispatch")
        if time.time() >= request["deadline"]:
            raise RuntimeUnavailable("action deadline elapsed before execution")
        _verify_mounts(request)
        if inspect_owned(request) is not None:
            raise RuntimeUnavailable("container already exists; action replay is forbidden")
        command = create_command(request)
        _write(directory / "create.json", {"argv": command})
        _host(command, timeout=min(20, max(0.1, request["deadline"] - time.time())))
        row = inspect_owned(request)
        if row is None:
            raise RuntimeUnavailable("created candidate container is missing")
        bound = verify_container(request, row)
        _write(directory / "binding.json", bound)
        _verify_mounts(request)
        process = subprocess.Popen(_control_argv([*_docker(spec), "start", "--attach", row["Id"]]),
                                   stdout=subprocess.PIPE, stderr=subprocess.PIPE, env=_host_env())
        outcome["timed_out"], outcome["stream_bytes"] = _capture(process, directory, request["deadline"])
        if not outcome["timed_out"]:
            # A candidate may close both streams while continuing to compute.
            # EOF is not termination and does not shorten its allotted runtime.
            try:
                process.wait(timeout=max(0.001, request["deadline"] - time.time()))
            except subprocess.TimeoutExpired:
                outcome["timed_out"] = True
        # Exit codes and StartedAt are read from the daemon, not from candidate
        # text or docker CLI conventions (a program may legitimately exit 125).
        observed = inspect_owned(request)
        if observed is None:
            raise RuntimeUnavailable("candidate container disappeared during execution")
        outcome["observed_state"] = observed["State"]
        started = observed["State"].get("StartedAt", "")
        outcome["execution_started"] = bool(started and not started.startswith("0001-"))
        if observed["State"].get("Error") or not outcome["execution_started"]:
            raise RuntimeUnavailable("Docker did not start the candidate process")
        if observed["State"].get("Running") and not outcome["timed_out"]:
            raise RuntimeUnavailable("Docker attachment ended while the candidate was still running")
    except BaseException as exc:
        outcome["harness_error"] = f"{type(exc).__name__}: {exc}"[-3000:]
    finally:
        _write(directory / "worker.json", outcome)
        if process is not None and process.poll() is None:
            # This only detaches the CLI. ExecStopPost, not this kill, owns
            # daemon-side termination and the final cleanup receipt.
            process.kill()
            process.wait(timeout=5)


def _project_result(directory, request):
    """All terminal statuses are projections of the same raw observations."""
    observed = _read(directory / "cleanup_observation.json")
    if (observed.get("request_sha256") != request["request_sha256"]
            or observed.get("observation") not in {"stopped", "absent", "unresolved"}):
        raise RuntimeUnavailable("invalid cleanup observation")
    outcome = _read(directory / "worker.json") if (directory / "worker.json").is_file() else {
        "execution_started": False, "timed_out": False, "harness_error": "worker_interrupted_before_receipt"}
    result = {**outcome, "schema_version": "evolvcoder-container-result-v1",
              "request_sha256": request["request_sha256"], "record_dir": str(directory),
              "status": "infrastructure_error", "cleanup_confirmed": observed["observation"] != "unresolved",
              "returncode": None, "oom_killed": False}
    if observed["observation"] == "stopped":
        final = _read(directory / "final_container.json")
        cid = (directory / "container.cid").read_text().strip()
        if (final.get("Id") != cid or final["State"].get("Running") or final["State"].get("Paused")
                or final.get("Config", {}).get("Labels", {}).get(LABEL) != request["action_id"]):
            raise RuntimeUnavailable("cleanup observation contradicts original container identity/state")
        result["container_state"] = final["State"]
        result["oom_killed"] = final["State"].get("OOMKilled") is True
        if not result.get("harness_error"):
            result["returncode"] = final["State"]["ExitCode"]
            result["status"] = "candidate_timeout" if result["timed_out"] else "candidate_exited"
    elif observed["observation"] == "absent" and not result.get("harness_error"):
        result["harness_error"] = "candidate container is missing"
    if observed["observation"] == "unresolved":
        result["harness_error"] = observed["error"]
    if (directory / "cancel.json").exists():
        result["status"] = "cancelled" if result["cleanup_confirmed"] else "infrastructure_error"
    for name in ("stdout", "stderr"):
        path = directory / (name + ".log")
        result[name] = path.read_bytes().decode(errors="replace") if path.is_file() else ""
    result["finished_at"] = observed["finished_at"]
    result["wall_seconds"] = observed["finished_at"] - request["created_at"]
    return result


def cleanup(directory):
    # Stop-post must wait for cancel's short receipt-write critical section.
    # A competing launch still fails immediately; cancel releases this lock
    # before waiting for systemctl, so cleanup never waits on its own waiter.
    with _lock(Path(directory), blocking=True):
        _cleanup(Path(directory))


def _cleanup(directory):
    directory = Path(directory)
    request = _load_action(directory)
    result_path = directory / "result.json"
    if result_path.exists() and status(directory).get("cleanup_confirmed") is True:
        return
    spec = ContainerSpec.from_dict(request["spec"])
    observation = {"request_sha256": request["request_sha256"], "observation": "unresolved", "error": None}
    try:
        row = inspect_owned(request)
        if row is not None:
            if row["State"].get("Running") or row["State"].get("Paused"):
                _host([*_docker(spec), "kill", row["Id"]], timeout=10)
            row = inspect_owned(request)
            if row is None or row["State"].get("Running") or row["State"].get("Paused"):
                raise RuntimeUnavailable("candidate termination is not confirmed")
            _write(directory / "final_container.json", row)
            # Keep the stopped, read-only container until explicit artifact
            # retention cleanup. No force-delete and no attribution by stderr.
            observation["observation"] = "stopped"
        else:
            observation["observation"] = "absent"
    except RuntimeUnavailable as exc:
        observation["error"] = str(exc)
    observation["finished_at"] = time.time()
    _write(directory / "cleanup_observation.json", observation)
    result = _project_result(directory, request)
    result["evidence_refs"] = {name: hashlib.sha256((directory / name).read_bytes()).hexdigest()
                               for name in sorted(EVIDENCE_FILES)
                               if (directory / name).is_file()}
    result["result_sha256"] = _digest(result)
    _write(result_path, result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("operation", choices=("worker", "cleanup", "status", "cancel"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.operation == "worker":
        worker(args.directory)
    elif args.operation == "cleanup":
        cleanup(args.directory)
    else:
        print(json.dumps(globals()[args.operation](args.directory), sort_keys=True))


if __name__ == "__main__":
    main()
