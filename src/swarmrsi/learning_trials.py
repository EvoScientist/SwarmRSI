"""Real subprocess validation for the RSI loop, with final-first reuse.

This adapter calls the same V-search entry point as ordinary experiments. It
does not manufacture trial scores or resume an ambiguous paid launch. A failed
or interrupted trial is retained and reported; only terminal source-bound
evidence can become a cached measurement.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import os
from pathlib import Path
import signal
import socket
import subprocess
import time

from .contracts import canonical_sha256
from .learning_evidence import read_run_evidence, source_ref, verify_ref, verify_run_evidence
from .retained_results import read_json, require, write_immutable
from .rsi import ValidationPending
from .tree_policy import PolicyTreeStrategy, TreePolicy
from .search_policy import SearchPolicy, compiled_trial_profile, search_spec_of
from .validation_failure import ValidationUnavailable, unavailable_record


RUNTIME_ENVIRONMENT_KEYS = frozenset({
    "SWARMRSI_ASSET_ROOT", "SWARMRSI_NATUREBENCH_TASKS_ROOT", "SWARMRSI_RUNTIME_PYTHON",
    "SWARMRSI_OPENROUTER_SECRET", "TORCH_HOME", "HF_HOME", "TRANSFORMERS_CACHE",
    "SWARMRSI_MODEL_CACHE",
    "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS",
    "PYTHONPATH", "PYTHONHOME", "VIRTUAL_ENV", "PATH",
})

# Shared across both Wisdom arms in this process, including a secondary
# failure while persisting the uncertainty receipt. A new process also checks
# unresolved original launch/exit records before it can start a fresh trial.
_UNCERTAIN_EXECUTION_SCOPES: set[Path] = set()


# The preflight builds the task context, which probes the candidate runtime (each probe
# may take a cold-filesystem retry of up to 600 s, see naturebench_session) and profiles
# the public data. At 90 s both image tasks with ~30k files on GPFS (siim-c1d-seed0,
# ranzcr-c1d-seed0) died here before their first model call. It makes no model calls, so
# a generous limit costs nothing but wall time.
PREFLIGHT_TIMEOUT_SECONDS = 1800


class SubprocessValidationRunner:
    def __init__(self, root: Path, *, source_root: Path, profile_path: Path, runtime_python: Path,
                 cells: list[dict], environment: dict[str, str] | None = None,
                 allow_execution: bool = False, timeout_seconds: int = 14400):
        require(type(timeout_seconds) is int and timeout_seconds > 0, "validation timeout must be a positive integer")
        require(bool(cells) and all(set(c) == {"task_id", "seed"} and type(c["seed"]) is int for c in cells),
                "validation cells must be explicit")
        require(len({(c["task_id"], c["seed"]) for c in cells}) == len(cells), "duplicate validation cells")
        overrides = dict(environment or {})
        require(set(overrides).issubset(RUNTIME_ENVIRONMENT_KEYS), "unsupported runtime environment override")
        require(all(isinstance(v, str) for v in overrides.values()), "environment values must be strings")
        self.source_root, self.root = source_root.resolve(), root.resolve()
        self.root.mkdir(parents=True, exist_ok=True)
        self.profile = read_json(profile_path)
        self.profile.pop("rsi_policy", None)
        require(all(c["task_id"] in self.profile["tasks"] for c in cells), "validation task missing from base profile")
        PolicyTreeStrategy(policy=TreePolicy(), base_config=self.profile.get("tree_search"))
        self.entrypoint = self.source_root / "scripts/run_swarmrsi_v_search.py"
        source_files = [*sorted((self.source_root / "src/swarmrsi").glob("*.py")), self.entrypoint,
                        self.source_root / "scripts/evaluate_naturebench_local_workspace.py"]
        for task_id in dict.fromkeys(c["task_id"] for c in cells):
            source_files.append(self.source_root / self.profile["tasks"][task_id]["starter"] / "run.py")
        # Do not dereference a venv's python symlink for execution: doing so
        # selects the base interpreter and loses the intended dependencies.
        self.runtime = runtime_python.absolute()
        require(self.runtime.is_file(), "validation Python runtime missing")
        self.environment = {key: overrides.get(key, os.environ.get(key)) for key in sorted(RUNTIME_ENVIRONMENT_KEYS)}
        self.environment = {key: value for key, value in self.environment.items() if value is not None}
        # Secret *paths*, never their bytes, belong to the execution context.
        self.config = {"schema_version": "swarmrsi-rsi-validation-runner-v1", "base_profile": self.profile,
                       "source_root": str(self.source_root),
                       "base_profile_source": source_ref(profile_path), "cells": cells,
                       "source_refs": [source_ref(p) for p in dict.fromkeys(source_files)],
                       "runtime_python": str(self.runtime), "environment": self.environment,
                       "runtime_executable_resolved": str(self.runtime.resolve()),
                       "timeout_seconds": timeout_seconds}
        write_immutable(self.root / "runner.json", self.config)
        self.allow_execution = allow_execution

    @classmethod
    def resume(cls, root: Path, *, allow_execution: bool = False):
        """Reuse the original execution environment, not the observer's shell."""
        instance = cls.__new__(cls)
        instance.root = root.resolve()
        instance.config = read_json(instance.root / "runner.json")
        require(instance.config["schema_version"] == "swarmrsi-rsi-validation-runner-v1", "unknown validation runner")
        instance.source_root = Path(instance.config["source_root"])
        instance.entrypoint = instance.source_root / "scripts/run_swarmrsi_v_search.py"
        instance.profile = instance.config["base_profile"]
        instance.runtime = Path(instance.config["runtime_python"])
        instance.environment = instance.config["environment"]
        require(str(instance.runtime.resolve()) == instance.config["runtime_executable_resolved"], "runtime executable changed")
        verify_ref(instance.config["base_profile_source"])
        for ref in instance.config["source_refs"]:
            verify_ref(ref)
        instance.allow_execution = allow_execution
        return instance

    @contextmanager
    def _exclusive(self, directory):
        with (directory / ".execution.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValidationPending("another process owns this validation") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _prepare(self, policy, task_id: str, seed: int):
        require({"task_id": task_id, "seed": seed} in self.config["cells"], "unplanned validation cell")
        executed = search_spec_of(policy)
        if executed is not None:
            from .search_plugins import default_search_registry
            default_search_registry().create(executed, seed=seed)
        else:
            PolicyTreeStrategy(policy=policy, seed=seed, base_config=self.profile.get("tree_search"))
        profile = compiled_trial_profile(self.profile, policy)
        verify_ref(self.config["base_profile_source"])
        for ref in self.config["source_refs"]:
            verify_ref(ref)
        identity = {"runner_sha256": canonical_sha256(self.config), "policy": policy.to_dict(),
                    "task_id": task_id, "seed": seed}
        root = self.root / "trials" / canonical_sha256(identity)
        root.mkdir(parents=True, exist_ok=True)
        write_immutable(root / "identity.json", identity)
        write_immutable(root / "profile.json", profile)
        command = [str(self.runtime), str(self.entrypoint), "--version", "v1", "--task-id", task_id,
                   "--seed", str(seed), "--profile", str(root / "profile.json"), "--run-dir", str(root / "run")]
        return root, command

    def _environment(self):
        inherited = {k: v for k, v in os.environ.items() if k not in RUNTIME_ENVIRONMENT_KEYS}
        return {**inherited, **self.environment, "PYTHONDONTWRITEBYTECODE": "1"}

    def _preflight_environment(self):
        return self._environment()

    def preflight(self, *, policy: TreePolicy, task_id: str, seed: int) -> dict:
        root, command = self._prepare(policy, task_id, seed)
        result = subprocess.run([*command, "--preflight"], cwd=self.source_root, env=self._preflight_environment(),
                                text=True, capture_output=True, timeout=PREFLIGHT_TIMEOUT_SECONDS, check=False)
        return {"ready": result.returncode == 0, "returncode": result.returncode,
                "stdout": result.stdout, "stderr": result.stderr, "zero_model_calls": True,
                "trial_id": root.name, "command": command}

    def _completed(self, root: Path, policy: TreePolicy, task_id: str, seed: int) -> dict:
        receipt = root / "evidence.json"
        if receipt.exists():
            evidence = read_json(receipt)
            verify_run_evidence(evidence)
        else:
            evidence = read_run_evidence(root / "run", self.source_root, source_role="selection")
            write_immutable(receipt, evidence)
        require(evidence["task_id"] == task_id and evidence["seed"] == seed
                and evidence["policy"] == policy.to_dict() and evidence["source_role"] == "selection",
                "completed validation identity mismatch")
        require(Path(evidence["run_dir"]).resolve() == (root / "run").resolve()
                and Path(evidence["source_root"]).resolve() == self.source_root,
                "cached evidence belongs to a different validation directory")
        return evidence

    @staticmethod
    def _process_identity(pid: int) -> str | None:
        # Start time guards against PID reuse; return None when no such process.
        result = subprocess.run(["ps", "-p", str(pid), "-o", "lstart="], capture_output=True,
                                text=True, timeout=10, check=False)
        return result.stdout.strip() or None

    def _execute_process(self, root: Path, command: list[str]) -> tuple[int | None, str | None]:
        """R33 process-group ownership, including normal exits and receipt errors."""
        with (root / "execution.log").open("xb") as log:
            started_at = time.time()
            environment = self._environment()
            environment["SWARMRSI_TRIAL_STARTED_AT_UNIX"] = str(started_at)
            process = subprocess.Popen(command, cwd=self.source_root, env=environment,
                                       stdout=log, stderr=subprocess.STDOUT, start_new_session=True)
            failure = None
            try:
                write_immutable(root / "process.json", {"pid": process.pid,
                    "start_identity": self._process_identity(process.pid), "host": socket.gethostname(),
                    "started_at_unix": started_at, "watchdog_seconds": self.config["timeout_seconds"]})
                try:
                    process.wait(timeout=self.config["timeout_seconds"])
                except subprocess.TimeoutExpired:
                    failure = "validation_wall_timeout"
            finally:
                # A normally exited leader can leave scientific workers alive.
                # The same cleanup runs if writing process.json raised.
                try:
                    os.killpg(process.pid, signal.SIGTERM)
                except ProcessLookupError:
                    pass
                if process.poll() is None:
                    try:
                        process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        pass
                try:
                    os.killpg(process.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
                returncode = process.wait(timeout=10)
        return returncode, failure

    def _control_record(self) -> Path:
        return Path(self.config.get("execution_control_root", str(self.root))) / "execution_uncertain.json"

    def _unsettled_launch(self) -> bool:
        scope = self._control_record().parent
        launches = [*scope.glob("trials/*/launch.json"), *scope.glob("*/trials/*/launch.json")]
        return any(not (path.parent / "exit.json").is_file() for path in launches)

    def evaluate(self, *, policy: TreePolicy, task_id: str, seed: int) -> dict:
        root, command = self._prepare(policy, task_id, seed)
        with self._exclusive(root):
            # Never call the inner entry point again on a terminal result: its
            # own main() is resumable but does not have a final-first shortcut.
            if os.path.lexists(root / "run/final.json"):
                evidence = self._completed(root, policy, task_id, seed)
                visits = len(list(root.glob("cache_hit_*.json")))
                write_immutable(root / f"cache_hit_{visits:06d}.json", {"run_id": evidence["run_id"],
                    "evidence_sha256": evidence["evidence_sha256"], "additional_model_calls": 0})
                return evidence
            if self._control_record().exists() or self._control_record().parent in _UNCERTAIN_EXECUTION_SCOPES:
                raise ValidationPending("prior execution control or cleanup is uncertain; no additional trial launched")
            launch_path, exit_path = root / "launch.json", root / "exit.json"
            if launch_path.exists():
                if exit_path.exists():
                    if type(read_json(exit_path).get("returncode")) is not int:
                        raise ValidationPending("prior validation exit does not establish process termination")
                    raise ValidationUnavailable(unavailable_record(root))
                launch = read_json(launch_path)
                pid_path = root / "process.json"
                if launch["host"] != socket.gethostname():
                    raise ValidationPending("validation belongs to another host; do not infer termination")
                if pid_path.exists():
                    process = read_json(pid_path)
                    if process["start_identity"] is not None and self._process_identity(process["pid"]) == process["start_identity"]:
                        raise ValidationPending("existing validation process still active")
                raise ValidationPending("indeterminate/interrupted validation; artifacts retained, automatic relaunch forbidden")
            if self._unsettled_launch():
                raise ValidationPending("another trial has an unsettled launch without an exit; no new trial launched")
            require(self.allow_execution, "paid validation execution is disabled; use preflight or explicit launch authorization")
            preflight = self.preflight(policy=policy, task_id=task_id, seed=seed)
            write_immutable(root / "preflight.json", preflight)
            require(preflight["ready"], "validation zero-call preflight failed")
            write_immutable(launch_path, {"command": command, "host": socket.gethostname(),
                                          "runner_sha256": canonical_sha256(self.config)})
            returncode, failure, control_error = None, None, None
            try:
                returncode, failure = self._execute_process(root, command)
                write_immutable(exit_path, {"returncode": returncode, "failure": failure,
                                           "cost_usd": None, "cost_status": "use_run_accounting_if_available"})
            except Exception as exc:
                control_error = {"trial_id": root.name, "error_type": type(exc).__name__,
                                 "scope": "execution_or_receipt_or_cleanup_uncertain"}
                _UNCERTAIN_EXECUTION_SCOPES.add(self._control_record().parent)
                try:
                    write_immutable(self._control_record(), control_error)
                except Exception as secondary:
                    # Do not let a second receipt error hide an already valid
                    # final. Original launch/exit evidence remains unsettled.
                    control_error["secondary_receipt_error"] = type(secondary).__name__
            if os.path.lexists(root / "run/final.json"):
                # Valid measured work remains available even when post-final
                # cleanup/reporting failed. The next fresh trial stays blocked.
                return self._completed(root, policy, task_id, seed)
            if control_error is not None:
                raise ValidationPending("execution control uncertain; artifacts retained, no automatic relaunch")
            if type(returncode) is not int:
                raise ValidationPending("validation process termination is unknown; no automatic relaunch")
            raise ValidationUnavailable(unavailable_record(root))

    def accounting(self) -> dict:
        rows = []
        for root in sorted((self.root / "trials").glob("*")):
            if not root.is_dir():
                continue
            receipt = root / "evidence.json"
            run = read_json(receipt) if receipt.exists() else None
            if run is not None:
                verify_run_evidence(run)
            rows.append({"trial_id": root.name, "launched_once": (root / "launch.json").exists(),
                         "run_id": run["run_id"] if run else None, "cost_usd": run["cost_usd"] if run else None,
                         "accounting": run["accounting"] if run else None,
                         "cache_hits": len(list(root.glob("cache_hit_*.json")))})
        known = [r["cost_usd"] for r in rows if r["launched_once"] and r["cost_usd"] is not None]
        return {"unique_launches": sum(r["launched_once"] for r in rows), "known_cost_usd": sum(known),
                "unknown_cost_launches": sum(r["launched_once"] and r["cost_usd"] is None for r in rows), "trials": rows}
