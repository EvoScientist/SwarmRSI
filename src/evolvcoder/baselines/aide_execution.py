"""Original AIDE execution with a gold-isolated NatureBench scoring bridge."""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import time
from typing import Any, Callable, Mapping

from .aide_state import atomic_json, canonical, seal, unseal


class AideExecutionUnresolved(RuntimeError):
    """Host/evaluator failure, not a buggy-program label for AIDE search."""


def file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_public_tree(root: Path) -> None:
    """Upstream preview follows directory links on the host, before bwrap."""
    root = root.resolve()
    def visit(path: Path, ancestors: frozenset[Path]) -> None:
        real = path.resolve(strict=True)
        if real != root and root not in real.parents:
            raise ValueError("public input escapes the public data root")
        if real.is_dir():
            if real in ancestors:
                raise ValueError("public input contains a directory cycle")
            for child in path.iterdir():
                visit(child, ancestors | {real})
        elif not real.is_file():
            raise ValueError("public input contains an unsupported filesystem object")
    visit(root, frozenset())


def workspace_inventory(root: Path, *, skip_input: bool = False) -> dict[str, Any]:
    """Record links as links; never read a candidate symlink's host target."""
    rows: dict[str, Any] = {}
    for base, directories, filenames in os.walk(root, followlinks=False):
        parent = Path(base)
        if skip_input and parent == root:
            directories[:] = [name for name in directories if name != "input"]
        for name in sorted(directories + filenames):
            path = parent / name
            relative = path.relative_to(root).as_posix()
            if path.is_symlink():
                rows[relative] = {"kind": "symlink", "target": os.readlink(path)}
            elif path.is_file():
                rows[relative] = {"kind": "file", "sha256": file_sha(path), "bytes": path.stat().st_size}
            elif path.is_dir():
                rows[relative] = {"kind": "directory"}
            else:
                raise AideExecutionUnresolved(f"unsupported candidate filesystem object: {relative}")
    return rows


class AideNatureExecution:
    def __init__(self, *, workspace: Path, artifacts: Path, task_root: Path,
                 runtime_python: Path, upstream_root: Path, dependency_root: Path,
                 helper_path: Path, interpreter_sha256: str, bridge: Any,
                 evaluator_script: Path, timeout_seconds: int = 1800,
                 operation_budget: Any | None = None) -> None:
        self.workspace, self.artifacts = workspace.resolve(), artifacts.resolve()
        self.task_root, self.runtime_python = task_root.resolve(), runtime_python.absolute()
        self.upstream_root, self.dependency_root = upstream_root.resolve(), dependency_root.resolve()
        self.helper_path, self.evaluator_script = helper_path.resolve(), evaluator_script.resolve()
        self.interpreter_sha256, self.bridge = interpreter_sha256, bridge
        self.timeout_seconds = timeout_seconds
        self.operation_budget = operation_budget
        if timeout_seconds <= 0:
            raise ValueError("positive AIDE execution timeout required")
        if self.artifacts == self.workspace or self.workspace in self.artifacts.parents:
            raise ValueError("AIDE artifacts/receipts must be outside candidate cwd")
        if file_sha(self.upstream_root / "aide/interpreter.py") != interpreter_sha256:
            raise ValueError("AIDE Interpreter hash mismatch")

    def prepare(self) -> None:
        self.artifacts.mkdir(parents=True, exist_ok=True)
        public = self.task_root / "problem/data"
        if not public.is_dir():
            raise ValueError("NatureBench public input absent")
        validate_public_tree(public)
        ownership_path = self.artifacts / "workspace_owner.json"
        identity = {"workspace": str(self.workspace), "public_data": str(public.resolve()),
                    "public_input_sha256": canonical(workspace_inventory(public)),
                    "interpreter_sha256": self.interpreter_sha256}
        initial = not ownership_path.exists()
        if not initial:
            if unseal(json.loads(ownership_path.read_text())) != identity:
                raise ValueError("AIDE workspace ownership or frozen public input changed")
        elif self.workspace.exists() and any(self.workspace.iterdir()):
            raise ValueError("refusing to initialize an unmanaged nonempty AIDE workspace")
        self.workspace.mkdir(parents=True, exist_ok=True)
        input_dir = self.workspace / "input"
        if input_dir.is_symlink():
            raise ValueError("candidate input mapping directory is a symlink")
        input_dir.mkdir(exist_ok=True)
        if initial:
            for name in ("working", "output"):
                (self.workspace / name).mkdir()
        # Host-side upstream data_preview sees the same public input. The entire
        # physical input directory is over-mounted read-only inside bubblewrap.
        sources = sorted(public.iterdir())
        if set(path.name for path in (self.workspace / "input").iterdir()) - {p.name for p in sources}:
            raise ValueError("unexpected entry in AIDE public input mapping")
        for source in sources:
            target = self.workspace / "input" / source.name
            if target.is_symlink():
                if target.resolve() != source.resolve():
                    raise ValueError("AIDE input mapping changed")
            elif target.exists():
                raise ValueError("unexpected object in AIDE input mapping")
            else:
                target.symlink_to(source, target_is_directory=source.is_dir())
        if initial:
            atomic_json(ownership_path, seal(identity))

    def _command(self, timeout_seconds: int | None = None) -> tuple[list[str], Path, dict[str, str]]:
        # Deliberately do not inherit API keys, service settings, or private paths.
        environment = {
            "PATH": "/runtime/bin:/usr/bin:/bin", "LANG": "C.UTF-8",
            "PYTHONPATH": "/aide-deps:/aide-source", "PYTHONNOUSERSITE": "1",
            "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1",
            "MPLBACKEND": "Agg",
        }
        for name in ("CUDA_VISIBLE_DEVICES", "OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
            if name in os.environ:
                environment[name] = os.environ[name]
        command, cwd, env, sandboxed = self.bridge._candidate_command(
            workspace=self.workspace, data_dir=self.task_root / "problem/data",
            runtime_python=self.runtime_python, require_sandbox=True, environment=environment,
        )
        if not sandboxed:
            raise AideExecutionUnresolved("AIDE candidate sandbox is required")
        boundary = command.index("--")
        command = command[:boundary] + [
            "--ro-bind", str(self.task_root / "problem/data"), "/workspace/input",
            "--ro-bind", str(self.upstream_root / "aide"), "/aide-source/aide",
            "--ro-bind", str(self.dependency_root), "/aide-deps",
            "--ro-bind", str(self.helper_path), "/aide-execute.py",
            "--tmpfs", "/dev/shm",
            "--", "/runtime/bin/python", "/aide-execute.py",
            "--upstream-root", "/aide-source",
            "--interpreter-sha256", self.interpreter_sha256,
            "--timeout-seconds", str(self.timeout_seconds if timeout_seconds is None else timeout_seconds),
        ]
        return command, cwd, env

    def _snapshot(self, destination: Path) -> dict[str, Any]:
        before = workspace_inventory(self.workspace, skip_input=True)
        shutil.copytree(self.workspace, destination, symlinks=True,
                        ignore=lambda directory, names: ["input"] if Path(directory) == self.workspace else [])
        if workspace_inventory(destination) != before:
            raise AideExecutionUnresolved("AIDE mutable workspace changed while snapshotting")
        return {"workspace_sha256": canonical(workspace_inventory(destination)),
                "workspace_hash_scope": "immutable raw workspace excluding public input mount",
                "pre_snapshot_workspace_sha256": canonical(before)}

    def execute(self, *, node_id: str, code: str,
                phase_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
                execution_only: bool = False) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise ValueError("unsafe AIDE node identifier")
        if not isinstance(execution_only, bool):
            raise ValueError("execution_only must be an explicit boolean")
        inner_timeout, process_timeout = self._operation_timeout("execute", execution_only=execution_only)
        node_root = self.artifacts / node_id
        node_root.mkdir(exist_ok=False)
        (node_root / "program.py").write_text(code, encoding="utf-8")
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        atomic_json(node_root / "execution_started.json", seal({
            "schema_version": "aide-execution-start-v2", "node_id": node_id,
            "code_sha256": code_sha, "execution_only": execution_only,
        }))
        output = self.workspace / "output"
        if output.is_symlink():
            output.unlink()
        elif output.is_file():
            output.unlink()
        elif output.exists():
            shutil.rmtree(output)
        output.mkdir()
        callback = phase_callback or (lambda phase, payload: None)
        callback("executing", {"node_id": node_id, "code_sha256": code_sha})
        command, cwd, environment = self._command(inner_timeout)
        started = time.monotonic()
        failure: str | None = None
        try:
            process = self._run_external(
                command, input=code, cwd=cwd, env=environment, capture_output=True,
                text=True, timeout=process_timeout, check=False,
            )
            stdout, stderr, returncode = process.stdout, process.stderr, process.returncode
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            returncode, failure = None, "original_interpreter_helper_timeout"
        atomic_json(node_root / "helper_capture.json", seal({
            "stdout": stdout, "stderr": stderr, "returncode": returncode,
            "wall_seconds": time.monotonic() - started,
        }))
        raw_snapshot = node_root / "workspace"
        snapshot_manifest = self._snapshot(raw_snapshot)
        atomic_json(node_root / "workspace_snapshot.json", seal(snapshot_manifest))
        # Scoring metadata never shares a directory namespace with arbitrary
        # candidate files. In particular, raw .evolvcoder and raw run.py remain
        # untouched evidence and cannot redirect host writes or replace source.
        snapshot = node_root / "artifact"
        snapshot.mkdir(exist_ok=False)
        (snapshot / "run.py").write_text(code, encoding="utf-8")
        raw_output = raw_snapshot / "output"
        if raw_output.is_dir() and not raw_output.is_symlink():
            shutil.copytree(raw_output, snapshot / "output", symlinks=True)
        if failure or returncode != 0:
            raise AideExecutionUnresolved(failure or f"original_interpreter_helper_exit_{returncode}")
        try:
            receipt = json.loads(stdout)
            if (receipt["schema_version"] != "aide-original-interpreter-receipt-v1"
                    or receipt["code_sha256"] != code_sha
                    or receipt["interpreter_sha256"] != self.interpreter_sha256
                    or receipt["reset_session"] is not True):
                raise ValueError("receipt identity mismatch")
            result = receipt["execution_result"]
            if not isinstance(result["term_out"], list) or not all(isinstance(s, str) for s in result["term_out"]):
                raise ValueError("invalid original Interpreter output")
        except (ValueError, KeyError, TypeError) as exc:
            raise AideExecutionUnresolved("original_interpreter_receipt_invalid") from exc
        atomic_json(node_root / "execution_receipt.json", seal(receipt))
        valid = result["exc_type"] is None
        try:
            if raw_output.is_symlink():
                raise ValueError("candidate output root is a symlink")
            attestation = self.bridge._attest(snapshot / "output") if valid else None
        except ValueError:
            valid, attestation = False, None
        execution = {
            "schema_version": "candidate-execution-result-v1", "valid": valid,
            "execution_valid": valid, "score_bearing_evaluation_attempted": False,
            "candidate_saw_hidden_labels": False, "official_gold_access": False,
            "details": {"candidate_executable_sha256": code_sha,
                        "candidate_sandboxed": True, "submission_output_attestation": attestation,
                        "evaluation_mode": "public_execution_only"},
        }
        if not valid:
            execution["public_diagnostic"] = {
                "category": "candidate_execution" if result["exc_type"] else "candidate_output",
                "summary": str(result["exc_type"] or "missing/invalid output artifact"),
                "candidate_only": True, "hidden_data_included": False,
            }
        atomic_json(snapshot / ".evolvcoder/execution_validation.json", execution)
        callback("execution_complete", {"node_id": node_id, "execution_valid": valid,
                                        "receipt_sha256": canonical(receipt)})
        evaluation = execution
        if valid and not execution_only:
            callback("evaluating", {"node_id": node_id, "score_attempt_authorized": True})
            evaluation = self.score(node_id=node_id)
        bundle = {"execution_result": result, "evaluation": evaluation,
                  "node_id": node_id, "artifact": str(snapshot),
                  "snapshot_manifest": snapshot_manifest}
        atomic_json(node_root / "execution_bundle.json", seal(bundle))
        callback("executed", {"node_id": node_id, "bundle_sha256": canonical(bundle)})
        return bundle

    def reconcile(self, *, node_id: str, code: str,
                  phase_callback: Callable[[str, Mapping[str, Any]], None] | None = None,
                  allow_first_score: bool = True) -> dict[str, Any]:
        """Recover verified completed work; never repeat a started operation.

If execution completed but scoring provably never started, the first score
may proceed. A receipt missing after execution_started, or a score_started
without a completed scorer capture, is unresolved instead of a blind retry.
        """
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise ValueError("unsafe AIDE node identifier")
        node_root = self.artifacts / node_id
        required = ["program.py", "execution_started.json", "execution_receipt.json",
                    "workspace_snapshot.json", "artifact/run.py",
                    "artifact/.evolvcoder/execution_validation.json"]
        if any(not (node_root / name).is_file() for name in required):
            raise AideExecutionUnresolved("execution has no complete recoverable receipt; do not rerun")
        code_sha = hashlib.sha256(code.encode()).hexdigest()
        started = unseal(json.loads((node_root / "execution_started.json").read_text()))
        if (started.get("schema_version") != "aide-execution-start-v2"
                or not isinstance(started.get("execution_only"), bool)):
            raise AideExecutionUnresolved("original execution mode is ambiguous; cannot authorize scoring")
        if (started["node_id"] != node_id or started["code_sha256"] != code_sha
                or file_sha(node_root / "program.py") != code_sha
                or file_sha(node_root / "artifact/run.py") != code_sha):
            raise ValueError("AIDE recovered candidate code changed")
        if started["execution_only"]:
            raise AideExecutionUnresolved("execution-only qualification cannot start scoring through recovery")
        receipt = unseal(json.loads((node_root / "execution_receipt.json").read_text()))
        if (receipt.get("schema_version") != "aide-original-interpreter-receipt-v1"
                or receipt.get("code_sha256") != code_sha
                or receipt.get("interpreter_sha256") != self.interpreter_sha256
                or receipt.get("reset_session") is not True):
            raise ValueError("AIDE recovered original execution identity changed")
        snapshot_manifest = unseal(json.loads((node_root / "workspace_snapshot.json").read_text()))
        if canonical(workspace_inventory(node_root / "workspace")) != snapshot_manifest["workspace_sha256"]:
            raise ValueError("AIDE immutable raw workspace changed")
        snapshot = node_root / "artifact"
        evidence_path = snapshot / ".evolvcoder/execution_validation.json"
        execution = json.loads(evidence_path.read_text())
        if (execution.get("details", {}).get("candidate_executable_sha256") != code_sha
                or execution.get("details", {}).get("candidate_sandboxed") is not True
                or execution.get("score_bearing_evaluation_attempted") is not False):
            raise ValueError("AIDE execution evidence no longer matches the candidate")
        bundle_path = node_root / "execution_bundle.json"
        saved_bundle = unseal(json.loads(bundle_path.read_text())) if bundle_path.exists() else None
        if (saved_bundle is not None and saved_bundle["evaluation"].get("valid") is True
                and saved_bundle["evaluation"].get("score_bearing_evaluation_attempted") is False):
            raise AideExecutionUnresolved("execution-only qualification cannot start scoring through recovery")
        if execution.get("valid") is True:
            if receipt["execution_result"]["exc_type"] is not None:
                raise ValueError("failed original Interpreter receipt cannot attest valid execution")
            if self.bridge._attest(snapshot / "output") != execution["details"]["submission_output_attestation"]:
                raise ValueError("AIDE recovered submission output changed")
            score_started = node_root / "score_started.json"
            target, capture_path = node_root / "official_evaluation.json", node_root / "scorer_capture.json"
            if score_started.exists():
                expected = {
                    "node_id": node_id, "candidate_sha256": code_sha,
                    "execution_evidence_sha256": file_sha(evidence_path),
                    "output_attestation": self.bridge._attest(snapshot / "output"),
                    "task_root": str(self.task_root), "evaluator_bridge_sha256": file_sha(self.evaluator_script),
                }
                if unseal(json.loads(score_started.read_text())) != expected:
                    raise ValueError("AIDE scorer attempt identity changed")
                if not target.is_file() or not capture_path.is_file():
                    raise AideExecutionUnresolved("scorer attempt incomplete; do not repeat it")
                capture = unseal(json.loads(capture_path.read_text()))
                if capture.get("returncode") != 0 or capture.get("state") == "unresolved":
                    raise AideExecutionUnresolved("scorer completion is unresolved")
                evaluation_bytes = target.read_bytes()
                if capture.get("evaluation_sha256") != hashlib.sha256(evaluation_bytes).hexdigest():
                    raise AideExecutionUnresolved("scorer completion lacks a matching result hash")
                evaluation = json.loads(evaluation_bytes)
                if evaluation.get("valid") is not True:
                    raise AideExecutionUnresolved("official_evaluator_invalid_not_candidate_bug")
            elif target.exists() or capture_path.exists():
                raise AideExecutionUnresolved("scorer artifacts have no bound attempt")
            else:
                if not allow_first_score:
                    raise AideExecutionUnresolved("read-only history verification cannot start scoring")
                callback = phase_callback or (lambda phase, payload: None)
                callback("evaluating", {"node_id": node_id, "score_attempt_authorized": True})
                evaluation = self.score(node_id=node_id)
        else:
            evaluation = execution
        bundle = {"execution_result": receipt["execution_result"], "evaluation": evaluation,
                  "node_id": node_id, "artifact": str(snapshot), "snapshot_manifest": snapshot_manifest}
        if saved_bundle is not None:
            if saved_bundle != bundle:
                raise ValueError("AIDE execution bundle changed after completion")
        else:
            atomic_json(bundle_path, seal(bundle))
        return bundle

    def verify_completed(self, *, node_id: str, code: str) -> dict[str, Any]:
        """Read-only history audit: requires a completed bundle before recovery."""
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise ValueError("unsafe AIDE node identifier")
        if not (self.artifacts / node_id / "execution_bundle.json").is_file():
            raise AideExecutionUnresolved("completed node lost its execution bundle")
        return self.reconcile(node_id=node_id, code=code, allow_first_score=False)

    def score(self, *, node_id: str) -> dict[str, Any]:
        if not re.fullmatch(r"[A-Za-z0-9_-]+", node_id):
            raise ValueError("unsafe AIDE node identifier")
        node_root = self.artifacts / node_id
        target = node_root / "official_evaluation.json"
        started_path = node_root / "score_started.json"
        if target.exists() or started_path.exists():
            raise AideExecutionUnresolved("official evaluation already exists; reconcile it, do not overwrite")
        inner_timeout, process_timeout = self._operation_timeout("score")
        snapshot = node_root / "artifact"
        atomic_json(started_path, seal({
            "node_id": node_id, "candidate_sha256": file_sha(snapshot / "run.py"),
            "execution_evidence_sha256": file_sha(snapshot / ".evolvcoder/execution_validation.json"),
            "output_attestation": self.bridge._attest(snapshot / "output"),
            "task_root": str(self.task_root), "evaluator_bridge_sha256": file_sha(self.evaluator_script),
        }))
        try:
            result = self._run_external([
                str(self.runtime_python), str(self.evaluator_script),
                "--workspace", str(snapshot), "--result", str(target),
                "--task-root", str(self.task_root), "--runtime-python", str(self.runtime_python),
                "--timeout-seconds", str(inner_timeout),
                "--require-candidate-sandbox", "--score-existing-output",
            ], capture_output=True, text=True, timeout=process_timeout, check=False)
        except (subprocess.TimeoutExpired, OSError) as exc:
            atomic_json(node_root / "scorer_capture.json", seal({
                "state": "unresolved", "error_type": type(exc).__name__,
                "score_attempt_may_have_started": True,
            }))
            raise AideExecutionUnresolved("official_evaluator_attempt_unresolved") from None
        evaluation_bytes = target.read_bytes() if target.is_file() else None
        atomic_json(node_root / "scorer_capture.json", seal({
            "stdout": result.stdout, "stderr": result.stderr, "returncode": result.returncode,
            "evaluation_sha256": None if evaluation_bytes is None else hashlib.sha256(evaluation_bytes).hexdigest(),
        }))
        if result.returncode != 0 or evaluation_bytes is None:
            raise AideExecutionUnresolved("official_evaluator_transport_failed")
        evaluation = json.loads(evaluation_bytes)
        if evaluation.get("valid") is not True:
            # The evaluator itself says the candidate's output caused the invalid result
            # (e.g. malformed predictions made the official scorer error). That is a buggy
            # program, as it is for BRS, not an unresolved technical failure.
            if (evaluation.get("public_diagnostic") or {}).get("candidate_only") is True:
                return evaluation
            raise AideExecutionUnresolved("official_evaluator_invalid_not_candidate_bug")
        return evaluation

    def _operation_timeout(self, kind: str, *, execution_only: bool = False) -> tuple[int, float]:
        if self.operation_budget is None:
            return self.timeout_seconds, self.timeout_seconds + 30
        return self.operation_budget.external_timeout(
            kind=kind, configured_seconds=self.timeout_seconds, execution_only=execution_only,
        )

    def _run_external(self, *args: Any, **kwargs: Any) -> Any:
        if self.operation_budget is None:
            return subprocess.run(*args, **kwargs)
        from .aide_resources import run_bounded_process
        # Hashing, output cleanup and durable phase writes can be slow. Read
        # the same persistent deadline again immediately before dispatch. The
        # score slot was already reserved under the run-wide owner lock; do not
        # count that operation's own score_started receipt against it twice.
        inner, total = self.operation_budget.wall_timeout(configured_seconds=self.timeout_seconds)
        command = list(args[0])
        index = command.index("--timeout-seconds") + 1
        command[index] = str(min(int(command[index]), inner))
        kwargs["timeout"] = min(float(kwargs["timeout"]), total)
        return run_bounded_process(command, *args[1:], **kwargs)
