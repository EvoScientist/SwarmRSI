"""Installed full NatureBench session with explicit inputs and launch ownership.

This intentionally consumes the existing NatureBench profile, not RunConfig.
It does not pretend to implement the not-yet-integrated generic SDK or RSI loop.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import re
import shutil
import sys
import time
from urllib.parse import urlsplit

from . import naturebench_session as session
from .learning_evidence import read_run_evidence
from .session_limits import SessionLimits
from .model_settings import request_controls


OPERATIONAL_ENVIRONMENT_KEYS = frozenset({
    "PATH", "HOME", "USER", "LOGNAME", "LANG", "LC_ALL", "LC_CTYPE", "TZ",
    "TMPDIR", "TMP", "TEMP", "LD_LIBRARY_PATH", "LIBRARY_PATH",
    "CUDA_VISIBLE_DEVICES", "CUDA_HOME", "CUDA_PATH", "OMP_NUM_THREADS",
    "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS",
    "SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE",
    # An operator-provided egress proxy is deployment configuration in the same
    # sense as the CA bundle beside it: some hosts have no direct route out, so
    # dropping these turns every model call into "Network is unreachable". They
    # name a route, never a credential; a credential-env may still not use them
    # (see `validate_credential_env`), and the candidate sandbox keeps its own
    # allowlist, so a candidate does not inherit egress from the controller.
    "HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY",
    "http_proxy", "https_proxy", "no_proxy",
})

# These names are consumed by the interpreter, loader or runtime before our
# controller can isolate its environment. Never inject a credential into them,
# even when they are deliberately absent from the candidate allowlist.
STARTUP_ENVIRONMENT_PREFIXES = (
    "PYTHON", "LD_", "DYLD_", "SLURM_", "SWARMRSI_", "CUDA_", "TORCH_",
    "OMP_", "MKL_", "OPENBLAS_", "NUMEXPR_", "MALLOC_", "LC_",
)
STARTUP_ENVIRONMENT_KEYS = frozenset({
    "ENV", "BASH_ENV", "SHELLOPTS", "BASHOPTS", "IFS", "CDPATH",
    "GCONV_PATH", "LOCPATH", "NLSPATH", "HOSTALIASES", "GETCONF_DIR",
    "GLIBC_TUNABLES", "MPLBACKEND", "MPLCONFIGDIR", "VIRTUAL_ENV",
    "__PYVENV_LAUNCHER__", "SSLKEYLOGFILE",
})


def add_arguments(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--profile", type=Path, required=True,
                        help="Existing NatureBench JSON profile; relative starters resolve beside this file")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--runtime-python", required=True,
                        help="Candidate/evaluator Python on this host; venv launcher symlink is preserved")
    credentials = parser.add_mutually_exclusive_group(required=True)
    credentials.add_argument("--credential-file", type=Path,
                             help="Private JSON file with api_key, OPENROUTER_API_KEY, or key; never mounted in candidate")
    credentials.add_argument("--credential-env", help="Controller-only environment variable NAME; never its value")
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--model-cache", type=Path)
    parser.add_argument("--base-url", default="https://openrouter.ai/api/v1")
    parser.add_argument("--version", choices=("v1", "v2", "v3", "v4"), default="v1")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--preflight", action="store_true",
                        help="Check inputs and original runtime prerequisites; no model or scientific evaluation")


def executable_path(value: str | Path) -> Path:
    value = os.fspath(value)
    if not value or "\x00" in value:
        raise ValueError("runtime-python must be an executable name or path")
    candidate = shutil.which(value) if "/" not in value else os.path.abspath(os.path.expanduser(value))
    if candidate is None or not Path(candidate).is_file() or not os.access(candidate, os.X_OK):
        raise ValueError("runtime-python is not executable on this host")
    # Do NOT resolve: /venv/bin/python may point at the system interpreter.
    return Path(os.path.abspath(candidate))


def _overlap(a: Path, b: Path) -> bool:
    a, b = a.resolve(), b.resolve()
    return a.is_relative_to(b) or b.is_relative_to(a)


def validate_credential_env(name: str) -> None:
    if not isinstance(name, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", name) is None:
        raise ValueError("credential-env must be an environment variable name")
    if (name in OPERATIONAL_ENVIRONMENT_KEYS or name in STARTUP_ENVIRONMENT_KEYS
            or name.startswith(STARTUP_ENVIRONMENT_PREFIXES)):
        raise ValueError("credential-env cannot name an operational environment variable")


def _positive(value, name, *, integer=False):
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be a finite positive {'integer' if integer else 'number'} in the NatureBench profile")


def configure(args: argparse.Namespace) -> tuple[dict, dict]:
    """Resolve explicit inputs without reading a provider secret or starting it."""
    if args.task_id in {"", ".", ".."} or Path(args.task_id).name != args.task_id:
        raise ValueError("task-id must name one direct child of tasks-root")
    if args.seed < 0:
        raise ValueError("seed must be nonnegative")
    url = urlsplit(args.base_url)
    if url.scheme not in {"http", "https"} or not url.hostname or url.username or url.password or url.query or url.fragment:
        raise ValueError("base-url must be HTTP(S) without credentials, query, or fragment")
    args.profile = args.profile.resolve()
    args.tasks_root = args.tasks_root.resolve()
    args.run_dir = args.run_dir.resolve()
    args.credential_env = getattr(args, "credential_env", None)
    if (args.credential_file is None) == (args.credential_env is None):
        raise ValueError("configure exactly one credential-file or credential-env")
    if args.credential_env is not None:
        validate_credential_env(args.credential_env)
    args.credential_file = args.credential_file.resolve() if args.credential_file is not None else None
    args.model_cache = args.model_cache.resolve() if args.model_cache is not None else None
    args.runtime_python = executable_path(args.runtime_python)
    protocol = json.loads(args.profile.read_text())
    model = protocol.get("model")
    if not isinstance(model, str) or not model.strip():
        raise ValueError("profile must explicitly select model")
    task = protocol["tasks"][args.task_id]
    task_root = (args.tasks_root / args.task_id).resolve()
    starter = Path(task["starter"])
    starter = (starter if starter.is_absolute() else args.profile.parent / starter).resolve()
    required = (starter / "run.py", task_root / "metadata.json", task_root / "evaluation/evaluator.py")
    if not all(p.is_file() for p in required) or not (task_root / "problem/data").is_dir():
        raise ValueError("NatureBench task needs starter/run.py, metadata, public data and official evaluator")
    if args.model_cache is not None and not args.model_cache.is_dir():
        raise ValueError("explicit model-cache is absent")
    for asset in (starter, task_root, session.PACKAGE_ROOT, args.runtime_python.parent.parent):
        if _overlap(args.run_dir, asset):
            raise ValueError("run-dir must not overlap task, starter, package, or runtime assets")
    public_mounts = [starter, task_root / "problem/data", args.run_dir,
                     args.runtime_python.parent.parent, Path("/usr"), Path("/etc"), Path("/bin"), Path("/lib"), Path("/lib64")]
    if args.model_cache is not None:
        public_mounts.append(args.model_cache)
    private_assets = (task_root / "evaluation", task_root / "metadata.json")
    if any(_overlap(mount, private) for mount in public_mounts for private in private_assets):
        raise ValueError("candidate-visible mounts must not expose private evaluator or metadata assets")
    if args.credential_file is not None and any(args.credential_file.is_relative_to(p.resolve()) for p in public_mounts):
        raise ValueError("credential-file must be outside candidate-visible mounts and run artifacts")
    settings = {**protocol.get("screening_execution_limits", {}), **task.get("screening_execution_limits", {})}
    request_controls(protocol, settings, role="coder")
    request_controls(protocol, settings, role="planner")
    budget = SessionLimits.from_legacy(task["limits"], enforce_context=settings.get("enforce_context_read_token_limit", True))
    budget.validate_action_limits(settings)
    from .task_metrics import target
    target(task)
    session.TASKS_ROOT, session.ACTIVE_TASK_ID = args.tasks_root, args.task_id
    session.RUNTIME, session.SECRET = args.runtime_python, args.credential_file
    session.CANDIDATE_RUNTIME = protocol.get("candidate_runtime")
    if session.CANDIDATE_RUNTIME is not None:
        from .candidate_execution import validate_binding
        _, runtime_records = validate_binding(session.CANDIDATE_RUNTIME)
        if args.model_cache is not None:
            raise ValueError("explicit model-cache mounts are not yet supported by this container backend")
        if any(_overlap(runtime_records, p) for p in (starter, task_root, args.runtime_python.parent.parent, args.run_dir)):
            raise ValueError("candidate runtime records overlap candidate or protected assets")
    session.SECRET_ENV = args.credential_env
    session.MODEL_CACHE, session.BASE_URL = args.model_cache, args.base_url
    args.root_evidence = None  # No implicit historical-score or root reuse.
    search = session.strategy(args.version, args.seed, protocol) if protocol.get("search") is not None else None
    context = {"schema_version": "swarmrsi-installed-naturebench-session-v1",
               "task_id": args.task_id, "seed": args.seed, "version": args.version,
               "profile": str(args.profile), "tasks_root": str(args.tasks_root),
               "runtime_python": str(args.runtime_python), "runtime_executable_sha256": session.sha(args.runtime_python),
               "orchestrator_python": sys.executable,
               "credential_file": str(args.credential_file) if args.credential_file is not None else None, "base_url": args.base_url,
               "model_cache": str(args.model_cache) if args.model_cache is not None else None,
               "source_hashes": (session.cell_source_hashes(starter, args.profile, search)
                                 if search is not None else session.cell_source_hashes(starter, args.profile))}
    if args.credential_env is not None:
        context["credential_env"] = args.credential_env
    if session.CANDIDATE_RUNTIME is not None:
        context["candidate_runtime"] = session.CANDIDATE_RUNTIME
    return protocol, context


@contextmanager
def run_lock(directory: Path):
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / ".session.lock").open("a+") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise ValueError("this run-dir is already owned by another session") from exc
        try:
            yield
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@contextmanager
def isolated_environment(model_cache: Path | None):
    """Provider credentials remain controller-only, never inherited by candidates.

    Keep an explicit operational allowlist, not a blacklist of secret spellings.
    CLI calls restore their caller environment even after failure.
    """
    before = dict(os.environ)
    previous_controller_environment = session._CONTROLLER_ENVIRONMENT
    # Nested isolation must not discard the outer controller's credential
    # source. This mapping is not passed to subprocesses or serialized.
    session._CONTROLLER_ENVIRONMENT = previous_controller_environment if previous_controller_environment is not None else before
    clean = {k: v for k, v in before.items() if k in OPERATIONAL_ENVIRONMENT_KEYS or k.startswith("SLURM_")}
    clean.update(PYTHONDONTWRITEBYTECODE="1", PYTHONUNBUFFERED="1", MPLBACKEND="Agg")
    if model_cache is not None:
        clean.update(SWARMRSI_MODEL_CACHE=str(model_cache), TORCH_HOME=str(model_cache))
    os.environ.clear()
    os.environ.update(clean)
    try:
        yield
    finally:
        os.environ.clear()
        os.environ.update(before)
        session._CONTROLLER_ENVIRONMENT = previous_controller_environment


def verify_terminal(run_dir: Path, protocol: dict, args: argparse.Namespace) -> dict:
    manifest = json.loads((run_dir / "manifest.json").read_text())
    expected = (args.task_id, args.seed, protocol["model"], session.strategy(args.version, args.seed, protocol).name)
    actual = tuple(manifest.get(k) for k in ("task_id", "seed", "model", "version"))
    if actual != expected:
        raise ValueError("terminal belongs to a different requested task/seed/model/strategy")
    evidence = read_run_evidence(run_dir, session.ROOT, source_role="development")
    final = json.loads((run_dir / "final.json").read_text())
    return {"status": "verified_terminal", "task_id": args.task_id,
            "best_native_metric": final["best_native_metric"],
            "best_evolved_native_metric": evidence["best_evolved_native_metric"],
            "artifact_valid": evidence["artifact_valid"], "stop_reason": final["stop_reason"],
            "final_sha256": session.sha(run_dir / "final.json"),
            "usage": final["accounting"], "planner_transport": evidence.get("planner_transport")}


def execute(args: argparse.Namespace) -> int:
    protocol, context = configure(args)
    if args.preflight:
        with isolated_environment(args.model_cache):
            return session.run_session(args)
    with run_lock(args.run_dir):
        context_path = args.run_dir / "session_inputs.json"
        state_path = args.run_dir / "session_state.json"
        if context_path.exists():
            if json.loads(context_path.read_text()) != context:
                raise ValueError("session inputs changed; use the original inputs to inspect this run")
        else:
            existing = {p.name for p in args.run_dir.iterdir()} - {".session.lock"}
            if existing:
                raise ValueError("existing run has no installed-session input identity; refusing takeover")
            with context_path.open("x") as stream:
                json.dump(context, stream, sort_keys=True, indent=2)
                stream.write("\n")
        # Final first: even post-final failures never cause another provider call.
        if (args.run_dir / "final.json").exists():
            result = verify_terminal(args.run_dir, protocol, args)
            print(json.dumps({**result, "reused_terminal": True}))
            return 1 if result["stop_reason"] in {"search_proposal_rejected_without_incumbent",
                                                  "session_deadline_reached_without_incumbent",
                                                  "planner_transport_failed_without_incumbent"} else 0
        if state_path.exists():
            raise ValueError("unfinished prior launch; preserve its pending calls and artifacts, do not automatically replay")
        orphaned = {p.name for p in args.run_dir.iterdir()} - {".session.lock", "session_inputs.json"}
        if orphaned:
            raise ValueError("session state is missing but prior artifacts exist; refusing orphan replay")
        session.atomic_json(state_path, {"status": "running", "pid": os.getpid(), "started_at_unix": time.time()})
        try:
            with isolated_environment(args.model_cache):
                returncode = session.run_session(args)
            if returncode != 0:
                raise RuntimeError(f"session returned {returncode}")
            result = verify_terminal(args.run_dir, protocol, args)
            session.atomic_json(state_path, {"status": "completed", "finished_at_unix": time.time(),
                                            "final_sha256": result["final_sha256"]})
            print(json.dumps({**result, "reused_terminal": False}))
            return 0
        except BaseException as exc:
            # Preserve the primary exception if writing the status itself fails.
            try:
                session.atomic_json(state_path, {"status": "interrupted", "error_type": type(exc).__name__,
                                                "finished_at_unix": time.time()})
            except OSError:
                pass
            raise


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    add_arguments(parser)
    try:
        return execute(parser.parse_args(argv))
    except (ValueError, KeyError, OSError) as exc:
        print(f"swarmrsi run-naturebench: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
