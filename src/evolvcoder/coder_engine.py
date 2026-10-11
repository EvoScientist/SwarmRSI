from __future__ import annotations

import ast
import fnmatch
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict
from pathlib import Path, PurePosixPath
from typing import Any, Protocol

from .coder_models import (
    CoderConfig,
    CoderExitStatus,
    CoderOutcome,
    CoderVersion,
    coder_sha256,
    selected_scientific_plan,
)
from .runtime_artifacts import RuntimeArtifactCollector
from .command_feedback import CommandFailureJournal
from .file_observation import read_text_page


class CoderModel(Protocol):
    def respond(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]: ...


def _file_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()



def _budget_remaining(*, model_calls: int, tool_calls: int, input_tokens: int,
                      output_tokens: int, wall_seconds: float) -> dict[str, int]:
    """What the model is told it has left, with only finite numbers in it."""
    budget = {"model_calls": model_calls, "tool_calls": tool_calls,
              "input_tokens": input_tokens, "output_tokens": output_tokens}
    if math.isfinite(wall_seconds):
        budget["wall_seconds"] = max(1, int(wall_seconds))
    return budget


def apply_search_replace(text: str, patch: str) -> tuple[str | None, list[str]]:
    """Apply SEARCH/REPLACE blocks in order; every SEARCH must occur exactly once."""
    import re
    blocks = re.findall(r"<<<<<<< SEARCH\n(.*?)\n?=======\n(.*?)\n?>>>>>>> REPLACE", patch, flags=re.S)
    if not blocks:
        return None, ["no complete <<<<<<< SEARCH / ======= / >>>>>>> REPLACE block found"]
    problems = []
    for index, (search, replace) in enumerate(blocks, 1):
        count = text.count(search) if search else 0
        if count != 1:
            problems.append(f"block {index}: SEARCH text found {count} times (must be exactly once)")
            continue
        text = text.replace(search, replace, 1)
    return (None, problems) if problems else (text, [])


class ContractBoundedCoder:
    """Persistent tool-loop whose scientific and filesystem policy is controller-owned."""

    def __init__(self, config: CoderConfig, model: CoderModel) -> None:
        self.config = config
        self.model = model
        self._runtime_artifacts: RuntimeArtifactCollector | None = None

    def _ignored(self, relative: str) -> bool:
        path = PurePosixPath(relative)
        if path.suffix == ".pyc" or any(
            part in {
                ".git", ".evolvcoder", "output", "__pycache__",
                ".pytest_cache", ".mypy_cache", ".ruff_cache",
            }
            for part in path.parts
        ):
            return True
        for pattern in self.config.ignored_workspace_patterns:
            if path.match(pattern):
                return True
            # PurePath.match('directory/**') does not cover arbitrary depth on
            # every supported Python version. Directory ignores are recursive.
            if pattern.endswith("/**") and relative.startswith(pattern[:-3].rstrip("/") + "/"):
                return True
            # A single-segment directory glob such as "*smoke*/**" names scratch output
            # directories whatever the coder called them (smoke_out, _smoke_out, ...).
            prefix = pattern[:-3].rstrip("/") if pattern.endswith("/**") else None
            if (prefix and "/" not in prefix and len(path.parts) > 1
                    and fnmatch.fnmatchcase(path.parts[0], prefix)):
                return True
        return False

    _TRANSIENT_TRANSPORT_MARKERS = (
        "IncompleteRead", "RemoteDisconnected", "Connection reset", "ConnectionResetError",
        "Connection aborted", "timed out", "TimeoutError", "total response deadline exceeded",
        "Temporary failure in name resolution", "EOF occurred", "HTTP Error 502", "HTTP Error 503",
        "HTTP Error 504", "HTTP Error 429",
        # The cluster reaches OpenRouter through a local cntlm proxy whose upstream
        # drops out for seconds at a time. urllib reports that as a tunnel failure,
        # not an HTTP status, and it ended an MLE-Bench run at step 3 with nothing
        # returned -- the same shape as every other marker here.
        "Tunnel connection failed", "Parent proxy unreach", "Proxy CONNECT",
    )

    @classmethod
    def _transient_transport_error(cls, exc: BaseException) -> bool:
        """Only failures where nothing was returned; never auth, schema or quota refusals."""
        text = f"{type(exc).__name__}: {exc}"
        return any(marker in text for marker in cls._TRANSIENT_TRANSPORT_MARKERS)

    def _respond_with_transport_retry(self, payload: dict[str, Any], events: list, started: float,
                                      exploration_deadline: float | None = None):
        """Re-send the identical request a bounded number of times after a transport failure.

        The cross-modal Biology-5 cell lost a whole action to one IncompleteRead.
        Returned usage accounts for successful responses; no response does NOT
        establish that a provider charged nothing. Retries retain their events.
        """
        if exploration_deadline is None and hasattr(self.model, "request_total_timeout_seconds"):
            # The normal action deadline applies even when the experimental
            # finalization reserve is off. Otherwise its last second can start
            # a full 300-second built-in transport request. Legacy in-process
            # adapters without a deadline control retain their old interface.
            exploration_deadline = started + self.config.max_wall_seconds
        session_deadline = getattr(self, "session_deadline_monotonic", None)
        if exploration_deadline is not None and session_deadline is not None:
            exploration_deadline = min(exploration_deadline, session_deadline)
        attempts = 0
        while True:
            try:
                if exploration_deadline is None:
                    return self.model.respond(payload)
                remaining = exploration_deadline - time.monotonic()
                if remaining <= 0:
                    raise TimeoutError("coder exploration deadline reached")
                # The built-in transport owns a killable request process. Scope
                # its deadline to exploration and restore the shared setting.
                if not hasattr(self.model, "request_total_timeout_seconds"):
                    raise RuntimeError("reserved finalization requires a deadline-aware model")
                previous = self.model.request_total_timeout_seconds
                self.model.request_total_timeout_seconds = (
                    remaining if previous is None else min(previous, remaining)
                )
                try:
                    return self.model.respond(payload)
                finally:
                    self.model.request_total_timeout_seconds = previous
            except Exception as exc:
                from .naturebench_session import transport_backoff_seconds
                budget = self.config.max_transport_retries or 0
                remaining = self.config.max_wall_seconds - (time.monotonic() - started)
                if exploration_deadline is not None:
                    remaining = min(remaining, exploration_deadline - time.monotonic())
                if not (self._transient_transport_error(exc) and attempts < budget and remaining > 30):
                    raise
                attempts += 1
                # One shared schedule with the planner, instead of the same constants
                # written twice: the ceiling moved once and only one copy followed.
                # Bounded by what is left of the action's wall.
                wait = min(transport_backoff_seconds(attempts), max(0.0, remaining - 30.0))
                self._append_event(events, "coder.model_response_retried", attempt=attempts,
                                   waited_seconds=round(wait, 1),
                                   error=f"{type(exc).__name__}:{str(exc)[:300]}")
                if wait > 0:
                    time.sleep(wait)

    @staticmethod
    def _normalize_model_action(response: dict[str, Any]) -> dict[str, Any]:
        """Accept flat or OpenAI-style {tool, arguments} without ambiguity."""
        arguments = response.get("arguments")
        if arguments is None:
            normalized = dict(response)
        else:
            if not isinstance(arguments, dict):
                raise ValueError("model_arguments_not_object")
            overlap = (set(response) - {"arguments"}) & set(arguments)
            if overlap:
                raise ValueError("ambiguous_model_arguments:" + ",".join(sorted(overlap)))
            normalized = {key: value for key, value in response.items() if key != "arguments"} | arguments
        aliases = {"file_path": "path", "old_string": "old", "new_string": "new"}
        for source, target in aliases.items():
            if source in normalized:
                if target in normalized:
                    raise ValueError(f"ambiguous_model_alias:{source}:{target}")
                normalized[target] = normalized.pop(source)
        return normalized

    def _manifest(self, root: Path) -> dict[str, str]:
        manifest: dict[str, str] = {}
        for path in sorted(root.rglob("*")):
            if path.is_symlink():
                relative = path.relative_to(root).as_posix()
                if not self._ignored(relative):
                    manifest[relative] = "symlink-forbidden"
            elif path.is_file():
                relative = path.relative_to(root).as_posix()
                if not self._ignored(relative):
                    manifest[relative] = _file_sha(path)
        return manifest

    @staticmethod
    def _workspace_sha(manifest: dict[str, str]) -> str:
        return coder_sha256(manifest)

    def _copy_manifest_workspace(
        self, source: Path, destination: Path, manifest: dict[str, str]
    ) -> None:
        """Copy exactly the files committed by a workspace manifest.

        Ignored outputs and caches are deliberately not copied: their bytes are
        outside the workspace hash and therefore cannot be trusted as repair
        input.  Each source file is checked immediately before copying so the
        destination is bound to the manifest even if the source changes.
        """
        if destination.exists():
            raise FileExistsError(destination)
        destination.mkdir(parents=True)
        for relative, expected_sha256 in manifest.items():
            if expected_sha256 == "symlink-forbidden":
                raise ValueError(f"workspace contains forbidden symlink:{relative}")
            source_path = self._safe_path(source, relative)
            if not source_path.is_file() or source_path.is_symlink():
                raise ValueError(f"manifest source file missing:{relative}")
            if _file_sha(source_path) != expected_sha256:
                raise ValueError(f"manifest source file hash mismatch:{relative}")
            destination_path = destination.joinpath(*PurePosixPath(relative).parts)
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)
        if self._manifest(destination) != manifest:
            raise ValueError("copied workspace does not match manifest")

    @staticmethod
    def _append_event(
        events: list[dict[str, Any]], kind: str, **payload: Any
    ) -> dict[str, Any]:
        event = {
            "sequence": len(events),
            "kind": kind,
            "payload": payload,
            "previous_event_sha256": events[-1]["event_sha256"] if events else None,
        }
        event["event_sha256"] = coder_sha256(event)
        events.append(event)
        return event

    def _safe_path(self, root: Path, relative: str, *, write: bool = False) -> Path:
        value = PurePosixPath(relative)
        if not relative or value.is_absolute() or ".." in value.parts or "\x00" in relative:
            raise ValueError("unsafe_relative_path")
        root_resolved = root.resolve()
        target = root_resolved.joinpath(*value.parts)
        resolved_parent = target.parent.resolve()
        if resolved_parent != root_resolved and root_resolved not in resolved_parent.parents:
            raise ValueError("path_escape")
        cursor = root_resolved
        for part in value.parts:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("symlink_path_forbidden")
        if write and self.config.version == CoderVersion.C3:
            if not self._write_allowed(value.as_posix()):
                raise ValueError(f"path_outside_contract:{value.as_posix()}")
        return target

    def _write_allowed(self, relative: str) -> bool:
        path = PurePosixPath(relative)
        return any(
            path.match(pattern)
            or (pattern.endswith("/**") and relative.startswith(pattern[:-3].rstrip("/") + "/"))
            for pattern in self.config.action.allowed_write_paths
        )

    def _read_allowed(self, relative: str) -> bool:
        path = PurePosixPath(relative)
        return any(path.match(pattern) for pattern in self.config.action.readable_paths)

    def _resolve_read_path(self, root: Path, relative: str) -> tuple[Path, str, bool]:
        """Resolve a workspace path or a controller-owned read-only mount."""
        value = PurePosixPath(relative)
        if not relative or value.is_absolute() or ".." in value.parts or "\x00" in relative:
            raise ValueError("unsafe_relative_path")
        mounts = dict(self.config.read_only_mounts)
        alias = value.parts[0]
        if alias not in mounts:
            return self._safe_path(root, relative), value.as_posix(), False
        mount_root = Path(mounts[alias]).resolve()
        if not mount_root.is_dir():
            raise ValueError(f"read_only_mount_missing:{alias}")
        target = mount_root.joinpath(*value.parts[1:])
        resolved = target.resolve()
        if resolved != mount_root and mount_root not in resolved.parents:
            raise ValueError("read_only_mount_escape")
        cursor = mount_root
        for part in value.parts[1:]:
            cursor = cursor / part
            if cursor.is_symlink():
                raise ValueError("symlink_path_forbidden")
        return target, value.as_posix(), True

    def _read_text(self, root: Path, relative: str) -> str:
        if self.config.version == CoderVersion.C3 and not self._read_allowed(relative):
            raise ValueError(f"path_not_readable:{relative}")
        path, _, _ = self._resolve_read_path(root, relative)
        if not path.is_file():
            raise ValueError(f"not_a_file:{relative}")
        if path.stat().st_size > 2_000_000:
            raise ValueError(f"file_too_large:{relative}")
        return path.read_text(encoding="utf-8", errors="replace")

    def _command_allowed(self, argv: tuple[str, ...]) -> bool:
        approved = [command.argv for command in self.config.validation]
        approved.extend(self.config.allowed_command_prefixes)
        return any(len(argv) >= len(prefix) and argv[:len(prefix)] == prefix for prefix in approved)

    def _run_command(
        self, root: Path, argv: tuple[str, ...], timeout: float, cwd: str = ".",
        *, sandboxed: bool = False,
    ) -> dict[str, Any]:
        started = time.monotonic()
        if not argv or any("\x00" in item for item in argv):
            return {"returncode": None, "harness_error": "invalid_argv", "timed_out": False,
                    "stdout": "", "stderr": "invalid argv", "wall_seconds": 0.0}
        if not self._command_allowed(argv):
            return {"returncode": None, "harness_error": "command_not_allowlisted",
                    "timed_out": False, "stdout": "", "stderr": "command not allowlisted",
                    "argv": list(argv), "wall_seconds": 0.0}
        before_runtime_artifacts = None
        container_binding = None
        try:
            cwd_path = root if cwd == "." else self._safe_path(root, cwd)
            if not cwd_path.is_dir():
                raise ValueError("command_cwd_not_directory")
            expanded_argv = tuple(
                item.replace("{workspace}", str(root.resolve())) for item in argv
            )
            runtime_environment = {
                key: value.replace("{workspace}", str(root.resolve()))
                for key, value in self.config.runtime_environment
            }
            if sandboxed:
                if not self.config.tool_command_wrapper and self.config.candidate_runtime is None:
                    return {
                        "argv": list(expanded_argv), "cwd": cwd, "returncode": None,
                        "stdout": "", "stderr": "sandbox required but not configured",
                        "timed_out": False, "harness_error": "sandbox_not_configured",
                        "wall_seconds": time.monotonic() - started,
                    }
                rewrites = self.config.tool_command_path_rewrites
                rewritten_argv = tuple(
                    next(
                        (destination + item[len(source):] for source, destination in rewrites
                         if item == source or item.startswith(source.rstrip("/") + "/")),
                        item,
                    )
                    for item in expanded_argv
                )
                sandbox_cwd = "/workspace" if cwd == "." else "/workspace/" + cwd.strip("/")
                wrapper = tuple(
                    item.replace("{workspace}", str(root.resolve())).replace(
                        "{sandbox_cwd}", sandbox_cwd
                    )
                    for item in self.config.tool_command_wrapper
                )
                if self.config.candidate_runtime is not None:
                    container_binding = json.loads(self.config.candidate_runtime)
                    expanded_argv = rewritten_argv
                else:
                    expanded_argv = wrapper + rewritten_argv
                runtime_environment.update(dict(self.config.tool_sandbox_environment))
            if self._runtime_artifacts is not None:
                before_runtime_artifacts = self._runtime_artifacts.inventory(root)
            if container_binding is not None:
                from .candidate_execution import execute
                public_data = dict(self.config.read_only_mounts).get("public_data")
                if public_data is None:
                    raise ValueError("container commands require an explicit public_data mount")
                record = execute(container_binding, workspace=root, public_data=public_data,
                                 argv=expanded_argv, timeout=timeout, cwd=sandbox_cwd)
                observation = {
                    "argv": list(expanded_argv), "cwd": cwd,
                    "execution_started": record.get("execution_started", False),
                    "returncode": record.get("returncode"),
                    "stdout": record.get("stdout", "")[-self.config.max_observation_chars:],
                    "stderr": record.get("stderr", "")[-self.config.max_observation_chars:],
                    "timed_out": record.get("timed_out", False),
                    "wall_seconds": time.monotonic() - started,
                    "runtime_receipt": {k: record[k] for k in ("record_dir", "request_sha256", "result_sha256")},
                }
                if record.get("harness_error") or record.get("status") in {"infrastructure_error", "cancelled"}:
                    observation.update(returncode=None, harness_error=record.get("harness_error") or record["status"])
            else:
                result = subprocess.run(
                    expanded_argv,
                    cwd=cwd_path,
                    text=True,
                    capture_output=True,
                    timeout=timeout,
                    check=False,
                    env={
                        **os.environ,
                        "PYTHONDONTWRITEBYTECODE": "1",
                        **runtime_environment,
                    },
                )
                observation = {
                    "argv": list(expanded_argv),
                    "execution_started": True,
                    "cwd": cwd,
                    "returncode": result.returncode,
                    "stdout": result.stdout[-self.config.max_observation_chars:],
                    "stderr": result.stderr[-self.config.max_observation_chars:],
                    "timed_out": False,
                    "wall_seconds": time.monotonic() - started,
                }
        except subprocess.TimeoutExpired as exc:
            stdout = exc.stdout.decode(errors="replace") if isinstance(exc.stdout, bytes) else (exc.stdout or "")
            stderr = exc.stderr.decode(errors="replace") if isinstance(exc.stderr, bytes) else (exc.stderr or "")
            observation = {
                "argv": list(argv), "cwd": cwd, "returncode": None, "execution_started": True,
                "stdout": stdout[-self.config.max_observation_chars:],
                "stderr": stderr[-self.config.max_observation_chars:],
                "timed_out": True, "wall_seconds": time.monotonic() - started,
            }
        except UnicodeDecodeError as exc:
            # communicate() decoded a completed child stream. This is not a
            # command that was never started, nor proof the command succeeded.
            observation = {
                "argv": list(argv), "cwd": cwd, "returncode": None,
                "execution_started": True, "stdout": "", "stderr": str(exc),
                "timed_out": False, "harness_error": "command_output_decode_error",
                "undecoded_stream_tail_hex": exc.object[-self.config.max_observation_chars:].hex(),
                "wall_seconds": time.monotonic() - started,
            }
        except (OSError, ValueError, RuntimeError) as exc:
            observation = {
                "argv": list(argv), "cwd": cwd, "returncode": None, "execution_started": False,
                "stdout": "", "stderr": str(exc)[-self.config.max_observation_chars:],
                "timed_out": False, "harness_error": type(exc).__name__,
                "wall_seconds": time.monotonic() - started,
            }
        if self._runtime_artifacts is not None and before_runtime_artifacts is not None:
            try:
                receipt = self._runtime_artifacts.collect(root, before_runtime_artifacts)
                if receipt["files"]:
                    observation["runtime_artifacts"] = receipt
            except (OSError, ValueError) as exc:
                observation.update({
                    "returncode": None,
                    "harness_error": f"runtime_artifact_archival_failed:{type(exc).__name__}",
                })
                if self._runtime_artifacts.last_receipt is not None:
                    observation["runtime_artifacts"] = self._runtime_artifacts.last_receipt
        return observation

    def _validate(
        self, root: Path, events: list[dict[str, Any]], *, deadline: float | None = None
    ) -> list[dict[str, Any]]:
        results: list[dict[str, Any]] = []
        for command in self.config.validation:
            result: dict[str, Any] = {}
            for attempt in range(1, command.max_harness_retries + 2):
                timeout = command.timeout_seconds
                if deadline is not None:
                    timeout = min(timeout, max(0.001, deadline - time.monotonic()))
                evidence_path = None
                if command.result_json_path is not None:
                    evidence_path = self._safe_path(root, command.result_json_path)
                    if evidence_path.is_file() or evidence_path.is_symlink():
                        evidence_path.unlink()
                result = self._run_command(
                    root, command.argv, timeout, command.cwd,
                    sandboxed=command.sandboxed,
                )
                result["tier"] = command.tier
                result["attempt"] = attempt
                result["blocking"] = command.blocking
                result["accepted"] = result.get("returncode") == 0
                if command.result_json_path is not None:
                    try:
                        if evidence_path is None:
                            raise ValueError("missing result evidence path")
                        evidence = json.loads(evidence_path.read_text(encoding="utf-8"))
                        if not isinstance(evidence, dict):
                            raise TypeError("result evidence is not an object")
                        checks: dict[str, bool] = {}
                        for dotted in command.required_truthy_json_paths:
                            value: Any = evidence
                            found = True
                            for part in dotted.split("."):
                                if not isinstance(value, dict) or part not in value:
                                    found = False
                                    break
                                value = value[part]
                            checks[dotted] = found and value is True
                        result["evidence_checks"] = checks
                        result["accepted"] = (
                            result.get("returncode") == 0 and all(checks.values())
                        )
                        if not result["accepted"]:
                            result["validation_contract_failed"] = True
                            diagnostic = evidence.get("public_diagnostic", {})
                            if isinstance(diagnostic, dict):
                                result["failure_category"] = diagnostic.get("category")
                                result["failure_summary"] = str(
                                    diagnostic.get("summary", "")
                                )[-1200:]
                    except (OSError, TypeError, ValueError, json.JSONDecodeError) as exc:
                        result["accepted"] = False
                        if result.get("timed_out") is True:
                            result["validation_contract_failed"] = True
                            result["failure_category"] = (
                                "candidate_timeout"
                                if command.tier in {"smoke", "full"}
                                else "validation_timeout"
                            )
                            result["failure_summary"] = (
                                "validation command timed out before writing result evidence"
                            )
                        else:
                            result["harness_error"] = (
                                f"result_evidence_invalid:{type(exc).__name__}"
                            )
                self._append_event(
                    events,
                    "coder.validation_finished",
                    tier=command.tier,
                    attempt=attempt,
                    returncode=result.get("returncode"),
                    accepted=result.get("accepted"),
                    blocking=command.blocking,
                    timed_out=result.get("timed_out", False),
                    harness_error=result.get("harness_error"),
                    observation_sha256=coder_sha256(result),
                    runtime_artifacts=result.get("runtime_artifacts"),
                )
                retryable = bool(result.get("harness_error"))
                time_remaining = deadline is None or time.monotonic() < deadline
                if retryable and attempt <= command.max_harness_retries and time_remaining:
                    self._append_event(
                        events,
                        "coder.validation_harness_retry",
                        tier=command.tier,
                        failed_attempt=attempt,
                        harness_error=result.get("harness_error"),
                        candidate_workspace_sha256=self._workspace_sha(self._manifest(root)),
                    )
                    continue
                break
            results.append(result)
            if not result["accepted"] and command.blocking:
                break
        return results

    @staticmethod
    def _blocking_validation_failures(
        validations: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        return [
            item for item in validations
            if item.get("accepted") is not True and item.get("blocking", True)
        ]

    @staticmethod
    def _ast_facts(source: str) -> tuple[set[str], set[str], set[str]]:
        tree = ast.parse(source)
        symbols: set[str] = set()
        calls: set[str] = set()
        imports: set[str] = set()
        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
                symbols.add(node.name)
            elif isinstance(node, ast.Name):
                symbols.add(node.id)
            elif isinstance(node, ast.Attribute):
                symbols.add(node.attr)
            elif isinstance(node, ast.Call):
                if isinstance(node.func, ast.Name):
                    calls.add(node.func.id)
                elif isinstance(node.func, ast.Attribute):
                    calls.add(node.func.attr)
            elif isinstance(node, ast.Import):
                imports.update(alias.name.split(".", 1)[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".", 1)[0])
        return symbols, calls, imports

    def _contract_violations(
        self,
        root: Path,
        before_manifest: dict[str, str],
        after_manifest: dict[str, str],
        parent_before: bytes,
    ) -> tuple[list[str], list[str]]:
        violations: list[str] = []
        changed = sorted(
            path for path in set(before_manifest) | set(after_manifest)
            if before_manifest.get(path) != after_manifest.get(path)
        )
        parent = root / self.config.executable_parent
        try:
            source = parent.read_text(encoding="utf-8")
            symbols: set[str] = set()
            calls: set[str] = set()
            imports: set[str] = set()
            for source_path in sorted(root.rglob("*.py")):
                relative = source_path.relative_to(root).as_posix()
                if self._ignored(relative) or source_path.is_symlink():
                    continue
                own_symbols, own_calls, own_imports = self._ast_facts(
                    source_path.read_text(encoding="utf-8")
                )
                symbols.update(own_symbols)
                calls.update(own_calls)
                imports.update(own_imports)
        except (OSError, UnicodeError, SyntaxError) as exc:
            return [f"executable_parent_not_parseable:{type(exc).__name__}"], changed
        action = self.config.action
        if self.config.version == CoderVersion.C3:
            violations.extend(
                f"path_outside_contract:{path}" for path in changed
                if not self._write_allowed(path)
            )
            violations.extend(
                f"protected_path_changed:{path}" for path in action.protected_paths
                if before_manifest.get(path) != after_manifest.get(path)
            )
            violations.extend(
                f"symlink_forbidden:{path}" for path, digest in after_manifest.items()
                if digest == "symlink-forbidden"
            )
        for name in action.required_symbols:
            if name not in symbols:
                violations.append(f"required_symbol_missing:{name}")
        for name in action.forbidden_symbols:
            if name in symbols:
                violations.append(f"forbidden_symbol_present:{name}")
        for name in action.required_call_names:
            if name not in calls:
                violations.append(f"required_call_missing:{name}")
        for name in action.forbidden_call_names:
            if name in calls:
                violations.append(f"forbidden_call_present:{name}")
        for name in action.required_import_roots:
            if name not in imports:
                violations.append(f"required_import_missing:{name}")
        for name in action.forbidden_import_roots:
            if name in imports:
                violations.append(f"forbidden_import_present:{name}")
        for pattern in action.required_source_patterns:
            if not re.search(pattern, source, flags=re.MULTILINE):
                violations.append(f"required_source_pattern_missing:{pattern}")
        before_source = parent_before.decode("utf-8", errors="replace")
        for pattern in action.preserved_source_patterns:
            if re.search(pattern, before_source, flags=re.MULTILINE) and not re.search(
                pattern, source, flags=re.MULTILINE
            ):
                violations.append(f"preserved_source_pattern_missing:{pattern}")
        if self.config.require_executable_parent_change and _file_sha(parent) == hashlib.sha256(parent_before).hexdigest():
            violations.append("scientific_action_not_implemented")
        return sorted(set(violations)), changed

    def _apply_tool(
        self,
        root: Path,
        action: dict[str, Any],
        feedback: dict[str, Any],
        deadline: float | None = None,
    ) -> tuple[bool, dict[str, Any]]:
        tool = str(action.get("tool", ""))
        if tool not in self.config.allowed_tools:
            return False, {"error": f"tool_not_allowed:{tool}"}
        limit = self.config.max_mutation_chars
        if limit is not None and tool in {"edit", "repair", "write", "append"}:
            new_text = action.get("content") if "content" in action else action.get("new")
            if isinstance(new_text, str) and len(new_text) > limit:
                return False, {
                    "error": "mutation_too_large", "characters": len(new_text), "limit": limit,
                    "instruction": (
                        f"Nothing was written. Send at most {limit} characters of new code per call: "
                        "change existing code with exact edits, and build a new module or function in "
                        "parts (write its first part, then hash-preconditioned append calls)."
                    ),
                }
        try:
            if tool == "inspect":
                relative = str(action.get("path", self.config.executable_parent))
                if self.config.version == CoderVersion.C3 and not self._read_allowed(relative):
                    raise ValueError(f"path_not_readable:{relative}")
                path, _, _ = self._resolve_read_path(root, relative)
                if not path.is_file():
                    raise ValueError(f"not_a_file:{relative}")
                return True, read_text_page(
                    path, display_path=relative,
                    start_line=action.get("start_line", 1),
                    start_column=action.get("start_column", 1),
                    end_line=action.get("end_line"),
                    max_chars=self.config.max_observation_chars,
                )
            if tool == "list":
                relative = str(action.get("path", "."))
                if relative == ".":
                    base, display, mounted = root, ".", False
                else:
                    base, display, mounted = self._resolve_read_path(root, relative)
                if not base.is_dir():
                    return False, {"error": "list_path_not_directory"}
                recursive = bool(action.get("recursive", False))
                limit = min(500, max(1, int(action.get("limit", 200))))
                iterator = base.rglob("*") if recursive else base.iterdir()
                entries: list[dict[str, Any]] = []
                for path in sorted(iterator):
                    if path.is_symlink():
                        continue
                    suffix = path.relative_to(base).as_posix()
                    shown = suffix if display == "." else f"{display.rstrip('/')}/{suffix}"
                    if not mounted and self._ignored(shown):
                        continue
                    if self.config.version == CoderVersion.C3 and not self._read_allowed(shown):
                        continue
                    entries.append({
                        "path": shown,
                        "type": "directory" if path.is_dir() else "file",
                        "size": path.stat().st_size if path.is_file() else None,
                    })
                    if len(entries) >= limit:
                        return True, {"entries": entries, "truncated": True}
                return True, {"entries": entries, "truncated": False}
            if tool == "search":
                query = action.get("query")
                if not isinstance(query, str) or not query or len(query) > 500:
                    return False, {"error": "invalid_search_query"}
                use_regex = bool(action.get("regex", False))
                matcher = re.compile(query if use_regex else re.escape(query))
                path_prefix = str(action.get("path", "."))
                if path_prefix == ".":
                    base, display, mounted = root, ".", False
                else:
                    base, display, mounted = self._resolve_read_path(root, path_prefix)
                if not base.is_dir():
                    return False, {"error": "search_path_not_directory"}
                matches: list[dict[str, Any]] = []
                for path in sorted(base.rglob("*")):
                    if not path.is_file() or path.is_symlink():
                        continue
                    suffix = path.relative_to(base).as_posix()
                    relative = suffix if display == "." else f"{display.rstrip('/')}/{suffix}"
                    if (not mounted and self._ignored(relative)) or not self._read_allowed(relative):
                        continue
                    if path.stat().st_size > 2_000_000:
                        continue
                    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
                        if matcher.search(line):
                            matches.append({"path": relative, "line": number, "text": line[:500]})
                            if len(matches) >= 100:
                                return True, {"matches": matches, "truncated": True}
                return True, {"matches": matches, "truncated": False}
            if tool in {"edit", "repair"}:
                relative = str(action.get("path", self.config.executable_parent))
                path = self._safe_path(root, relative, write=True)
                if "content" in action:
                    if path.exists():
                        return False, {"error": "content_create_requires_absent_path"}
                    content = action.get("content")
                    if not isinstance(content, str):
                        return False, {"error": "invalid_create_content"}
                    path.parent.mkdir(parents=True, exist_ok=True)
                    path.write_text(content, encoding="utf-8")
                    return True, {"path": relative, "operation": "create", "sha256": _file_sha(path)}
                old, new = action.get("old"), action.get("new")
                if not isinstance(old, str) or not isinstance(new, str) or not old:
                    return False, {"error": "invalid_exact_edit"}
                text = self._read_text(root, relative)
                occurrences = text.count(old)
                if occurrences != 1:
                    return False, {"error": "edit_anchor_not_unique", "occurrences": occurrences}
                path.write_text(text.replace(old, new, 1), encoding="utf-8")
                return True, {"path": relative, "operation": "replace", "sha256": _file_sha(path)}
            if tool == "write":
                relative = str(action.get("path", ""))
                content = action.get("content")
                if not isinstance(content, str):
                    return False, {"error": "invalid_write_content"}
                path = self._safe_path(root, relative, write=True)
                path.parent.mkdir(parents=True, exist_ok=True)
                operation = "replace" if path.exists() else "create"
                path.write_text(content, encoding="utf-8")
                return True, {"path": relative, "operation": operation, "sha256": _file_sha(path)}
            if tool == "append":
                relative = str(action.get("path", ""))
                content = action.get("content")
                expected_sha256 = action.get("expected_sha256")
                if not isinstance(content, str) or not isinstance(expected_sha256, str):
                    return False, {"error": "invalid_append_content_or_hash"}
                path = self._safe_path(root, relative, write=True)
                if not path.is_file() or path.is_symlink():
                    return False, {"error": "append_requires_regular_file"}
                current_sha256 = _file_sha(path)
                if current_sha256 != expected_sha256:
                    return False, {
                        "error": "append_precondition_failed",
                        "current_sha256": current_sha256,
                    }
                with path.open("a", encoding="utf-8") as handle:
                    handle.write(content)
                return True, {"path": relative, "operation": "append", "sha256": _file_sha(path)}
            if tool == "delete":
                relative = str(action.get("path", ""))
                if relative == self.config.executable_parent:
                    return False, {"error": "cannot_delete_executable_parent"}
                path = self._safe_path(root, relative, write=True)
                if not path.is_file() or path.is_symlink():
                    return False, {"error": "delete_requires_regular_file"}
                path.unlink()
                return True, {"path": relative, "operation": "delete"}
            if tool == "run":
                argv = action.get("argv")
                if not isinstance(argv, list) or not argv or not all(isinstance(x, str) for x in argv):
                    return False, {"error": "invalid_argv"}
                # Against the wall that is LEFT, not the whole action's. The model
                # picks this number itself, and clamping it to the full allocation let
                # one probe eat the entire budget: nyc-taxi's bench.py ran 1800.1s of a
                # 2430s coder wall, and a spooky cell admitted a 1500s run at t=1681s
                # and finished 852s past its deadline, because the deadline is only
                # tested between steps.
                remaining = (self.config.max_wall_seconds if deadline is None
                             else deadline - time.monotonic())
                timeout = min(float(action.get("timeout_seconds", 60.0)), max(1.0, remaining))
                result = self._run_command(
                    root, tuple(argv), timeout, str(action.get("cwd", ".")),
                    sandboxed=True,
                )
                return result.get("returncode") == 0, result
            if tool == "diagnose":
                return True, {"controller_feedback": feedback}
        except (OSError, UnicodeError, ValueError, re.error) as exc:
            return False, {"error": f"{type(exc).__name__}:{exc}"}
        return False, {"error": f"unhandled_tool:{tool}"}

    def _payload(
        self, root: Path, feedback: dict[str, Any], calls: int, tools: int,
        input_tokens: int, output_tokens: int,
        interaction_history: list[dict[str, Any]],
        command_failures: CommandFailureJournal | None = None,
        deadline: float | None = None,
    ) -> dict[str, Any]:
        parent_content = self._read_text(root, self.config.executable_parent)
        supplied_context: dict[str, str] = {}
        for relative in ("TASK_CONTEXT.md",):
            path = root / relative
            if path.is_file() and self._read_allowed(relative):
                text = self._read_text(root, relative)
                limit = self.config.task_context_chars
                if limit is None:
                    supplied_context[relative] = text[-20_000:]
                else:
                    supplied_context[relative] = text[:limit] + (
                        f"\n[... {len(text) - limit} characters truncated; inspect the file for the rest]\n"
                        if len(text) > limit else "")
        mutation_context = self.config.mutation_context
        mutation_guidance = None
        if mutation_context is not None:
            mutation_guidance = {
                "local_diff": (
                    "Make one or more surgical exact-anchor edits to the selected parent only."
                ),
                "structural_rewrite": (
                    "You may restructure the executable substantially, but preserve its interfaces, "
                    "the fixed hypothesis, and every contract constraint."
                ),
                "alternative_realization": (
                    "Develop an implementation-distinct realization of the fixed hypothesis. Do not "
                    "merely retune constants or repeat the parent structure."
                ),
                "cross_program_inspiration": (
                    "Recombine contract-compatible structures from the parent and inspirations into "
                    "one coherent executable; do not change the fixed scientific hypothesis."
                ),
            }[mutation_context.kernel]
        retained_history: list[dict[str, Any]] = []
        retained_chars = 0
        for item in reversed(interaction_history):
            rendered = json.dumps(item, ensure_ascii=False, sort_keys=True)
            if retained_history and retained_chars + len(rendered) > self.config.max_interaction_history_chars:
                break
            if len(rendered) > self.config.max_interaction_history_chars:
                rendered = rendered[-self.config.max_interaction_history_chars:]
                item = {"truncated_history_tail": rendered}
            retained_history.append(item)
            retained_chars += len(rendered)
        retained_history.reverse()
        payload = {
            "schema_version": "evolvcoder-coder-turn-v2",
            "version": self.config.version.value,
            "seed": self.config.seed,
            "action_contract": self.config.action.to_dict(),
            "executable_parent": self.config.executable_parent,
            "executable_parent_content": parent_content[-30_000:],
            "supplied_context_files": supplied_context,
            "allowed_tools": list(self.config.allowed_tools),
            "read_only_mounts": [alias for alias, _ in self.config.read_only_mounts],
            "runtime_artifact_directories": list(self.config.runtime_artifact_directories),
            "allowed_command_prefixes": [list(item) for item in self.config.allowed_command_prefixes],
            "tool_commands_are_filesystem_sandboxed": bool(
                self.config.tool_command_wrapper
            ),
            "validation_commands": [asdict(item) for item in self.config.validation],
            # The wall belongs here as much as the counts do. Without it the model is
            # told how many calls it has and nothing about the clock, and on a campaign
            # that caps no count it reads a budget of thousands of calls while the
            # deadline is what will actually end the action -- which is how cells came
            # to spend a whole action writing probe scripts. The key is omitted rather
            # than reported as infinity when there is genuinely no bound: an infinity
            # does not serialise and means nothing to the model.
            "budget_remaining": _budget_remaining(
                model_calls=self.config.max_model_calls - calls,
                tool_calls=self.config.max_tool_calls - tools,
                input_tokens=max(0, self.config.max_input_tokens - input_tokens),
                output_tokens=max(0, self.config.max_output_tokens - output_tokens),
                wall_seconds=(self.config.max_wall_seconds if deadline is None
                              else deadline - time.monotonic()),
            ),
            # The provider must cap this individual response at the remaining
            # action envelope.  A fixed per-turn cap can otherwise overshoot
            # the campaign limit over a multi-turn coder trajectory.
            "max_output_tokens_this_turn": max(
                0, self.config.max_output_tokens - output_tokens
            ),
            "feedback": feedback,
            "interaction_history": retained_history,
            "program_mutation_context": (
                asdict(mutation_context) if mutation_context is not None else None
            ),
            "mutation_guidance": mutation_guidance,
            "instruction": (
                "Implement only the declared action. Preserve protected scientific choices. "
                "Use one tool per response. " + (
                    f"Work in small steps: each write, append or edit carries at most "
                    f"{self.config.max_mutation_chars} characters of new code, and larger ones are rejected. "
                    "Change the current executable with exact edits instead of rewriting it; build anything "
                    "new part by part (for example data loading, then the model, then training and "
                    "evaluation) and run it once a part can be exercised. The executable, TASK_CONTEXT.md "
                    "(with a public data preview) and the parent's run output are supplied, so your first "
                    "response is the first of these edits; inspect or probe the public data only to "
                    "diagnose a failure you have observed. You may create contract-allowed helper modules. "
                    if self.config.max_mutation_chars is not None else
                    "Write first: the full current executable, TASK_CONTEXT.md (with a public data "
                    "preview) and the parent's run output are supplied, so your first response must be "
                    "the edit or write that implements the whole change, and your next must run it. "
                    "Inspect or probe the public data only to diagnose a failure you have observed. "
                    "You may create contract-allowed helper modules. "
                    if self.config.write_first else
                    "You may inspect/list/search the declared read-only public-data "
                    "mounts, run bounded local experiments, and create contract-allowed helper modules. "
                ) + "The full "
                "current executable is supplied. Do not spend successive turns repeating equivalent lookups. "
                "The interaction_history is the authoritative record of earlier tool calls in this action. "
                "Prefer exact edits for small changes. For large modules, write an initial chunk and use "
                "hash-preconditioned append calls; keep every chunk below "
                f"{self.config.max_mutation_chars or 16000} characters. Finish only "
                "when the implementation is ready for execution validation."
            ),
        }
        if self.config.diff_mode:
            payload["response_mode"] = "patch"
            payload["instruction"] = (
                "Implement only the declared action. Preserve protected scientific choices. The full "
                f"current {self.config.executable_parent}, TASK_CONTEXT.md (with a public data preview) and the "
                "parent's run output are supplied. Think the change through, then return exactly one JSON "
                "object: {\"plan\": <two or three sentences>, \"patch\": <one or more blocks>}. Each block is\n"
                "<<<<<<< SEARCH\n<lines copied exactly from the current file>\n=======\n<replacement lines>\n"
                ">>>>>>> REPLACE\n"
                "Keep each SEARCH small but unique in the file; use several blocks for several places. "
                "To replace the whole file instead, return {\"plan\": ..., \"candidate_content\": <complete file>}. "
                "The controller applies the patch, runs the program once and, if anything fails, returns the "
                "failure and the current file so the next response can be a corrective patch. The "
                "interaction_history records earlier patches and their results."
            )
        if self.config.no_silent_fallback:
            payload["instruction"] += (
                " The component this action declares must produce the submitted outputs for the instances "
                "it targets. Do not add a fallback that restores the parent's outputs when the component "
                "scores worse on your own hold-out: the controller keeps the parent anyway when a child is "
                "worse, and a silent fallback hides the evidence. Print the component's own hold-out error "
                "instead, so a poor result can be diagnosed and fixed."
            )
        if self.config.diff_mode and self.config.fresh_program_for_new_method:
            payload["instruction"] += (
                " First decide whether the declared action replaces the method for any instance (a "
                "different model, operator or representation) or refines the current one (parameters, "
                "features, a bounded fix). If it replaces the method, return candidate_content: a complete "
                "new program that implements the declared method from scratch, reusing only the parent's "
                "verified data loading and output writing, not the old method's code. If it refines, "
                "return a patch."
            )
        if self.config.action.selected_plan_json is not None:
            # A single structured copy survives background/history clipping.
            # The sealed JSON remains in the contract/config checkpoint, not
            # duplicated as escaped JSON text in the provider-facing payload.
            payload["action_contract"].pop("selected_plan_json")
            payload["selected_scientific_plan"] = selected_scientific_plan(
                self.config.action.selected_plan_json
            )
        if command_failures is not None:
            projection = command_failures.project(
                self._workspace_sha(self._manifest(root)),
                max_chars=min(6000, self.config.max_interaction_history_chars),
            )
            if projection is not None:
                payload["explicit_run_feedback"] = projection
        return payload

    def run(
        self, source_root: Path, output_dir: Path, *,
        diagnostic_seed_root: Path | None = None,
        expected_diagnostic_workspace_sha256: str | None = None,
    ) -> CoderOutcome:
        started = time.monotonic()
        deadline = started + self.config.max_wall_seconds
        session_deadline = getattr(self, "session_deadline_monotonic", None)
        if session_deadline is not None:
            deadline = min(deadline, session_deadline)
        reserve = self.config.finalization_reserve_seconds
        exploration_deadline = deadline - reserve if reserve is not None else deadline
        source_root = source_root.resolve()
        output_dir = output_dir.resolve()
        parent_source = self._safe_path(source_root, self.config.executable_parent)
        if not parent_source.is_file():
            raise FileNotFoundError(parent_source)
        parent_before = parent_source.read_bytes()
        parent_sha = hashlib.sha256(parent_before).hexdigest()
        if parent_sha != self.config.expected_parent_sha256:
            raise ValueError("executable parent hash does not match frozen config")
        if (output_dir / "result.json").exists() or (output_dir / "artifact").exists():
            raise FileExistsError("refusing to overwrite a completed coder cell")
        self._runtime_artifacts = (
            RuntimeArtifactCollector(
                self.config.runtime_artifact_directories,
                output_dir / "runtime_artifacts", self._write_allowed,
                self.config.action.protected_paths,
            ) if self.config.runtime_artifact_directories else None
        )

        before_manifest = self._manifest(source_root)
        parent_workspace_sha = self._workspace_sha(before_manifest)
        events: list[dict[str, Any]] = []
        validations: list[dict[str, Any]] = []
        calls = tools = input_tokens = output_tokens = repairs = 0
        observed_failure = False
        status = CoderExitStatus.IMPLEMENTATION_FAILED
        reason = "budget_exhausted"
        violations: list[str] = []
        changed_paths: list[str] = []
        seen_observations: set[str] = set()
        diagnosed_feedback: set[str] = set()
        interaction_history: list[dict[str, Any]] = []
        reasoning_only_truncations = 0
        command_failures = CommandFailureJournal()
        consecutive_duplicate_observations = 0
        # Bounded probes once the exploration phase is over: a run that does not
        # change the executable is a diagnostic, and the peptide/SMLM cells spent
        # 14 such probes in one action without ever implementing.
        diagnostic_runs_after_checkpoint = 0
        # The checkpoint counts calls since the implementation LAST changed, not calls
        # before it first changed. Gating on `not parent_changed` disarmed the guard
        # permanently the moment run.py was written once, which is the shape 12 of the
        # 14 wall-death cells in the MLE-Bench Lite campaign had: run.py written at call
        # three to five, then probing until the clock ran out with the brake already off.
        checkpoint_anchor_sha = parent_sha
        calls_at_anchor = 0
        repair_required_manifest: dict[str, str] | None = None
        repair_required_violations: tuple[str, ...] = ()
        repair_required_failure: dict[str, Any] | None = None

        with tempfile.TemporaryDirectory(prefix="evolvcoder-c3-") as tmp:
            temp_root = Path(tmp)
            work = temp_root / "workspace"
            checkpoint = temp_root / "checkpoint"
            self._copy_manifest_workspace(source_root, work, before_manifest)
            # A repair's failed seed is an implementation starting point, not
            # the verified fallback. Keep the rollback snapshot before loading it.
            self._copy_manifest_workspace(source_root, checkpoint, before_manifest)
            if diagnostic_seed_root is not None:
                if expected_diagnostic_workspace_sha256 is None:
                    raise ValueError("diagnostic seed requires an expected workspace hash")
                diagnostic_manifest = self._manifest(diagnostic_seed_root)
                diagnostic_sha256 = self._workspace_sha(diagnostic_manifest)
                if diagnostic_sha256 != expected_diagnostic_workspace_sha256:
                    raise ValueError("diagnostic seed workspace hash mismatch")
                shutil.rmtree(work)
                self._copy_manifest_workspace(
                    diagnostic_seed_root, work, diagnostic_manifest
                )
                self._append_event(
                    events, "coder.diagnostic_seed_applied",
                    diagnostic_workspace_sha256=diagnostic_sha256,
                )
            initial_work_manifest = self._manifest(work)
            initial_work_sha256 = self._workspace_sha(initial_work_manifest)
            self._append_event(
                events, "coder.checkpoint_created", parent_sha256=parent_sha,
                parent_workspace_sha256=parent_workspace_sha,
                initial_work_workspace_sha256=initial_work_sha256,
            )
            feedback: dict[str, Any] = ({
                "repair_start": {
                    "diagnostic_workspace_sha256": initial_work_sha256,
                    "verified_parent_workspace_sha256": parent_workspace_sha,
                    "instruction": (
                        "The current workspace is the bound failed implementation, not a new "
                        "repair. Inspect or run explicit bounded diagnostics if needed, then "
                        "make a concrete change before finish. On failure the verified parent "
                        "is retained and the attempted implementation is saved separately."
                    ),
                },
            } if diagnostic_seed_root is not None else {})

            def automatic_validation_permitted(manifest: dict[str, str]) -> bool:
                if manifest == initial_work_manifest:
                    return False
                if manifest == repair_required_manifest:
                    # This is the same state that already blocks another finish.
                    # Keep its failure evidence, not another execution. Declared
                    # harness retries still run inside the original _validate.
                    self._append_event(
                        events, "coder.unchanged_failed_validation_not_repeated",
                        workspace_sha256=self._workspace_sha(manifest),
                        validation_failure_sha256=(
                            coder_sha256(repair_required_failure)
                            if repair_required_failure is not None else None
                        ),
                        contract_violations=list(repair_required_violations),
                    )
                    return False
                return True

            while calls < self.config.max_model_calls and tools < self.config.max_tool_calls:
                if (
                    repair_required_manifest is not None
                    and self._manifest(work) != repair_required_manifest
                ):
                    repair_required_manifest = None
                    repair_required_violations = ()
                    repair_required_failure = None
                if time.monotonic() >= deadline:
                    reason = "wall_budget_exhausted"
                    break
                if reserve is not None and time.monotonic() >= exploration_deadline:
                    reason = "finalization_reserve_reached"
                    self._append_event(events, "coder.finalization_started", reason=reason,
                                       remaining_wall_seconds=max(0.0, deadline - time.monotonic()))
                    break
                if input_tokens >= self.config.max_input_tokens:
                    reason = "input_token_budget_exhausted"
                    self._append_event(events, "coder.budget_exhausted", reason=reason)
                    break
                if output_tokens >= self.config.max_output_tokens:
                    reason = "output_token_budget_exhausted"
                    self._append_event(events, "coder.budget_exhausted", reason=reason)
                    break
                try:
                    request_payload = self._payload(work, feedback, calls, tools, input_tokens, output_tokens,
                                                    interaction_history, command_failures, exploration_deadline)
                    threshold = self.config.reasoning_off_after_truncations
                    if threshold is not None and reasoning_only_truncations >= threshold:
                        request_payload["reasoning_off_this_turn"] = True
                        self._append_event(events, "coder.reasoning_off_for_turn", call=calls + 1,
                                           consecutive_reasoning_only_truncations=reasoning_only_truncations)
                    response, usage = self._respond_with_transport_retry(
                        request_payload,
                        events, started,
                        **({"exploration_deadline": exploration_deadline} if reserve is not None else {}))
                    if isinstance(response, dict):
                        response = self._normalize_model_action(response)
                except Exception as exc:  # model/provider errors are preserved; only transport re-sends
                    reason = f"model_response_error:{type(exc).__name__}:{str(exc)[:500]}"
                    self._append_event(events, "coder.model_response_failed", error=reason)
                    break
                calls += 1
                input_tokens += int(usage.get("input_tokens", 0))
                output_tokens += int(usage.get("output_tokens", 0))
                if (
                    isinstance(response, dict)
                    and response.get("_provider_finish_reason") in {"length", "max_tokens"}
                    and response.get("_parse_error_category") == "empty_response"
                ):
                    reasoning_only_truncations += 1
                else:
                    reasoning_only_truncations = 0
                self._append_event(
                    events, "coder.model_response", call=calls,
                    response=response, response_sha256=coder_sha256(response),
                    input_tokens=int(usage.get("input_tokens", 0)),
                    output_tokens=int(usage.get("output_tokens", 0)),
                )
                if input_tokens > self.config.max_input_tokens or output_tokens > self.config.max_output_tokens:
                    reason = "token_budget_exhausted"
                    self._append_event(events, "coder.budget_exhausted", reason=reason)
                    break
                if not isinstance(response, dict):
                    reason = "model_response_not_object"
                    break
                if response.get("tool") == "blocked":
                    status = CoderExitStatus.PLAN_BLOCKED
                    reason = str(response.get("reason") or "model_declared_blocked")[:2000]
                    break
                if (
                    response.get("_provider_finish_reason") in {"length", "max_tokens"}
                    and response.get("tool") in {"write", "append", "edit", "repair"}
                ):
                    feedback = {
                        "ok": False,
                        "observation": {
                            "error": "truncated_mutation_rejected",
                            "instruction": (
                                "The provider ended this mutation at its output limit. No partial edit was "
                                f"applied. Retry with a chunk below {self.config.max_mutation_chars or 16000} "
                                "characters, using write followed "
                                "by hash-preconditioned append when needed."
                            ),
                        },
                    }
                    interaction_history.append({
                        "call": calls,
                        "model_action": {
                            "tool": response.get("tool"),
                            "path": response.get("path"),
                            "rejected": "provider_output_truncated",
                        },
                        "tool_result": feedback,
                    })
                    observed_failure = True
                    self._append_event(
                        events, "coder.truncated_mutation_rejected",
                        tool=response.get("tool"), call=calls,
                    )
                    continue
                if self.config.diff_mode and (
                    isinstance(response.get("patch"), str) or isinstance(response.get("candidate_content"), str)
                ):
                    target = work / self.config.executable_parent
                    current = target.read_text(encoding="utf-8") if target.exists() else ""
                    if isinstance(response.get("candidate_content"), str):
                        new_text, problems = response["candidate_content"], []
                    else:
                        new_text, problems = apply_search_replace(current, response["patch"])
                    tools += 1
                    if new_text is None:
                        feedback = {"ok": False, "observation": {
                            "error": "patch_not_applied", "problems": problems[:10],
                            "instruction": "Nothing was changed. Return a patch whose SEARCH text is copied "
                                           "exactly from the current file supplied in this payload."}}
                        interaction_history.append({"call": calls, "model_action": {"tool": "patch", "applied": False},
                                                    "tool_result": feedback})
                        self._append_event(events, "coder.tool_finished", tool="patch", ok=False)
                        observed_failure = True
                        continue
                    target.write_text(new_text, encoding="utf-8")
                    self._append_event(events, "coder.tool_finished", tool="patch", ok=True)
                    validations = self._validate(work, events, deadline=deadline)
                    after_manifest = self._manifest(work)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, after_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status, reason = CoderExitStatus.SUCCESS, "all_gates_passed"
                        break
                    observed_failure = True
                    feedback = {"ok": False, "observation": {
                        "error": "validation_failed", "contract_violations": list(violations),
                        "validation_failure": failed[-1] if failed else None,
                        "instruction": "The patch was applied and kept. Return a corrective patch against the "
                                       "current file (supplied in this payload) that fixes this failure."}}
                    interaction_history.append({"call": calls, "model_action": {"tool": "patch", "applied": True},
                                                "tool_result": feedback})
                    continue
                if self.config.version == CoderVersion.C0:
                    candidate = response.get("candidate_content")
                    if not isinstance(candidate, str):
                        reason = "one_shot_candidate_missing"
                        break
                    (work / self.config.executable_parent).write_text(candidate, encoding="utf-8")
                    tools += 1
                    self._append_event(events, "coder.tool_finished", tool="one_shot_write", ok=True)
                    if diagnostic_seed_root is not None and self._manifest(work) == initial_work_manifest:
                        reason = "diagnostic_seed_requires_repair"
                        self._append_event(events, "coder.repair_seed_unchanged",
                                           requested_tool="one_shot_write",
                                           initial_work_workspace_sha256=initial_work_sha256)
                        break
                    validations = self._validate(work, events, deadline=deadline)
                    after_manifest = self._manifest(work)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, after_manifest, parent_before
                    )
                    if not self._blocking_validation_failures(validations) and not violations:
                        status, reason = CoderExitStatus.SUCCESS, "all_gates_passed"
                    else:
                        observed_failure = True
                        repair_required_manifest = after_manifest
                        repair_required_violations = tuple(violations)
                        failed = self._blocking_validation_failures(validations)
                        repair_required_failure = failed[-1] if failed else None
                        reason = "one_shot_failed"
                    break
                if (
                    response.get("tool") == "finish"
                    and repair_required_manifest is not None
                ):
                    feedback = {
                        "ok": False,
                        "observation": {
                            "error": "repair_mutation_required",
                            "instruction": (
                                "Controller validation already failed and the workspace has not changed. "
                                "Make a concrete edit, repair, write, append, or delete that addresses the "
                                "bound failure evidence before asking to validate again."
                            ),
                            "contract_violations": list(repair_required_violations),
                            "validation_failure": repair_required_failure,
                        },
                    }
                    interaction_history.append({
                        "call": calls,
                        "model_action": {"tool": "finish", "rejected": "repair_not_applied"},
                        "tool_result": feedback,
                    })
                    self._append_event(
                        events, "coder.repair_mutation_required",
                        requested_tool="finish",
                        contract_violations=list(repair_required_violations),
                    )
                    continue
                if (
                    response.get("tool") == "finish"
                    and diagnostic_seed_root is not None
                    and self._manifest(work) == initial_work_manifest
                ):
                    feedback = {
                        "ok": False,
                        "observation": {
                            "error": "diagnostic_seed_requires_repair",
                            "initial_work_workspace_sha256": initial_work_sha256,
                            "instruction": (
                                "No repair has changed the failed starting implementation. "
                                "Make a concrete code change, run an explicit bounded diagnostic, "
                                "or report blocked. Finish does not revalidate an unchanged failed seed."
                            ),
                        },
                    }
                    interaction_history.append({
                        "call": calls, "model_action": {"tool": "finish"},
                        "tool_result": feedback,
                    })
                    self._append_event(events, "coder.repair_seed_unchanged",
                                       requested_tool="finish",
                                       initial_work_workspace_sha256=initial_work_sha256)
                    continue
                if response.get("tool") == "finish":
                    validations = self._validate(work, events, deadline=deadline)
                    after_manifest = self._manifest(work)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, after_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status, reason = CoderExitStatus.SUCCESS, "all_gates_passed"
                        break
                    observed_failure = True
                    feedback = {
                        "validation_failure": failed[-1] if failed else None,
                        "contract_violations": violations,
                    }
                    repair_required_manifest = after_manifest
                    repair_required_violations = tuple(violations)
                    repair_required_failure = failed[-1] if failed else None
                    if repairs >= self.config.max_repairs:
                        reason = "repair_budget_exhausted"
                        break
                    repairs += 1
                    self._append_event(events, "coder.repair_requested", repair=repairs,
                                       feedback_sha256=coder_sha256(feedback))
                    continue
                feedback_before_tool_sha = coder_sha256(feedback)
                requested_tool = str(response.get("tool", ""))
                current_parent_sha = _file_sha(work / self.config.executable_parent)
                if current_parent_sha != checkpoint_anchor_sha:
                    checkpoint_anchor_sha = current_parent_sha
                    calls_at_anchor = calls
                implementation_checkpoint_due = (
                    self.config.version == CoderVersion.C3
                    and calls - calls_at_anchor >= max(2, self.config.max_model_calls // 2)
                )
                repeated_run_failure = (
                    command_failures.unchanged_failed_command(
                        response, self._workspace_sha(self._manifest(work))
                    ) if implementation_checkpoint_due and requested_tool == "run" else None
                )
                implementation_checkpoint = (
                    implementation_checkpoint_due
                    and (requested_tool in {"list", "inspect", "search", "diagnose"}
                         or (requested_tool == "run"
                             and self.config.max_diagnostic_runs_after_checkpoint is not None
                             and diagnostic_runs_after_checkpoint >= self.config.max_diagnostic_runs_after_checkpoint))
                )
                if implementation_checkpoint_due and requested_tool == "run" and not implementation_checkpoint:
                    diagnostic_runs_after_checkpoint += 1
                elif not implementation_checkpoint_due:
                    diagnostic_runs_after_checkpoint = 0
                executable_integration_required = (
                    self.config.version == CoderVersion.C3
                    and "scientific_action_not_implemented"
                    in (
                        set(feedback.get("contract_violations", []))
                        | set(repair_required_violations)
                    )
                    and requested_tool in {"write", "append", "edit", "repair"}
                    and str(response.get("path", "")) != self.config.executable_parent
                )
                repair_mutation_required = (
                    repair_required_manifest is not None
                    and requested_tool in {"list", "inspect", "search", "run", "diagnose"}
                )
                if response.get("_invalid_json") is True:
                    ok = False
                    category = response.get("_transport_error_category", "invalid_json_response")
                    instructions = {
                        "multiple_native_tool_calls": (
                            "No tool ran: this controller accepts one tool per response, not a batch. "
                            "Send only the first needed tool call and wait for its result."
                        ),
                        "response_truncated": (
                            "No tool ran: the provider stopped at its output limit. Send one shorter call. "
                            "For long code use bounded write/append chunks; avoid embedding a large Python "
                            "program inside run argv. Write a permitted helper script and run its path."
                        ),
                        "native_tool_arguments_not_object": (
                            "No tool ran: arguments must be a JSON object, not an array, string or scalar."
                        ),
                        "empty_response": "No tool ran: return exactly one nonempty allowed tool call.",
                    }
                    observation = {
                        "error": "invalid_model_action",
                        "transport_error_category": category,
                        "parse_error": str(response.get("_parse_error", "unknown"))[:1000],
                        "instruction": instructions.get(category, (
                            "The requested tool arguments were not valid JSON and no tool ran. "
                            "Issue one simpler valid tool call; do not repeat the malformed payload."
                        )),
                    }
                elif repeated_run_failure is not None:
                    ok = False
                    observation = {
                        "error": "unchanged_failed_command_at_implementation_checkpoint",
                        "execution_started": False,
                        "failure_evidence": repeated_run_failure,
                        "instruction": (
                            "This exact command already failed against the unchanged workspace. "
                            "Correct the diagnostic command or relevant code, or report plan_blocked. "
                            "A new or corrected bounded command remains available; this is not a "
                            "scientific verdict or a requirement to change the hypothesis."
                        ),
                    }
                elif implementation_checkpoint:
                    ok = False
                    observation = {
                        "error": "implementation_checkpoint_required",
                        "instruction": (
                            "The bounded exploration phase is over and the executable parent is still "
                            "unchanged. Make a concrete contract-preserving mutation to the executable "
                            "or report plan_blocked; further read-only inspection and diagnostic runs "
                            "are not available until implementation begins."
                        ),
                        "diagnostic_runs_after_checkpoint": diagnostic_runs_after_checkpoint,
                        "executable_parent": self.config.executable_parent,
                    }
                elif executable_integration_required:
                    ok = False
                    observation = {
                        "error": "executable_integration_required",
                        "instruction": (
                            "Controller validation found that helper changes are not connected to the "
                            "published executable. First edit or repair the executable parent so the "
                            "fixed scientific action is actually invoked."
                        ),
                        "executable_parent": self.config.executable_parent,
                    }
                elif repair_mutation_required:
                    ok = False
                    observation = {
                        "error": "repair_mutation_required",
                        "instruction": (
                            "Controller validation already failed and the workspace has not changed. "
                            "Use the supplied parent, interaction history, and bound failure evidence to "
                            "make a concrete mutation before further inspection or execution."
                        ),
                        "contract_violations": list(repair_required_violations),
                        "validation_failure": repair_required_failure,
                    }
                else:
                    run_before_sha = (
                        self._workspace_sha(self._manifest(work))
                        if response.get("tool") == "run" else None
                    )
                    ok, observation = self._apply_tool(work, response, feedback, exploration_deadline)
                    if run_before_sha is not None:
                        receipt = command_failures.observe(
                            response, observation, before_sha256=run_before_sha,
                            after_sha256=self._workspace_sha(self._manifest(work)),
                            tool_call=tools + 1,
                        )
                        if receipt is not None:
                            self._append_event(events, "coder.explicit_run_feedback", **receipt)
                observation_sha = coder_sha256(observation)
                if (
                    response.get("tool") == "diagnose"
                    and feedback_before_tool_sha in diagnosed_feedback
                ):
                    ok = False
                    observation = {
                        "error": "duplicate_diagnosis",
                        "instruction": (
                            "There is no new controller evidence to diagnose. Make a concrete edit, "
                            "run a frozen validation command, finish for controller validation, or "
                            "report blocked."
                        ),
                        "feedback_sha256": feedback_before_tool_sha,
                    }
                elif response.get("tool") == "diagnose":
                    diagnosed_feedback.add(feedback_before_tool_sha)
                elif response.get("tool") in {"list", "inspect", "search"} and observation_sha in seen_observations:
                    ok = False
                    observation = {
                        "error": "duplicate_observation",
                        "instruction": (
                            "This local lookup produced no new information. Use the supplied code to "
                            "edit, run a specific allowed check, or report blocked."
                        ),
                        "duplicate_observation_sha256": observation_sha,
                    }
                elif response.get("tool") in {"list", "inspect", "search"}:
                    seen_observations.add(observation_sha)
                tools += 1
                self._append_event(
                    events, "coder.tool_finished", tool=response.get("tool"), ok=ok,
                    observation=observation,
                    observation_sha256=coder_sha256(observation),
                )
                # Diagnose is a read of existing controller evidence, not new
                # evidence. Keeping the underlying feedback stable makes a
                # repeated diagnosis detectable instead of recursively nesting
                # the same observation until the token budget is exhausted.
                if response.get("tool") != "diagnose" or not ok:
                    feedback = {"ok": ok, "observation": observation}
                interaction_history.append({
                    "call": calls,
                    "model_action": {
                        key: value for key, value in response.items()
                        if key != "_provider_finish_reason"
                    },
                    "tool_result": {"ok": ok, "observation": observation},
                })
                if observation.get("error") in {"duplicate_observation", "duplicate_diagnosis"}:
                    consecutive_duplicate_observations += 1
                else:
                    consecutive_duplicate_observations = 0
                if consecutive_duplicate_observations >= 3:
                    non_progress_manifest = self._manifest(work)
                    if automatic_validation_permitted(non_progress_manifest):
                        validations = self._validate(work, events, deadline=deadline)
                        violations, changed_paths = self._contract_violations(
                            work, before_manifest, non_progress_manifest, parent_before
                        )
                        failed = self._blocking_validation_failures(validations)
                        if not failed and not violations:
                            status = CoderExitStatus.SUCCESS
                            reason = "controller_validation_after_non_progress_loop"
                        else:
                            status = CoderExitStatus.IMPLEMENTATION_FAILED
                            reason = "non_progress_loop_after_failed_validation"
                            observed_failure = True
                    elif non_progress_manifest == repair_required_manifest:
                        status = CoderExitStatus.IMPLEMENTATION_FAILED
                        reason = "non_progress_loop_after_failed_validation"
                        observed_failure = True
                    else:
                        status = CoderExitStatus.PLAN_BLOCKED
                        reason = "non_progress_loop_after_repeated_duplicate_observations"
                    self._append_event(
                        events, "coder.non_progress_loop_stopped",
                        consecutive_duplicates=consecutive_duplicate_observations,
                        workspace_changed=non_progress_manifest != before_manifest,
                        changed_from_initial_work=non_progress_manifest != initial_work_manifest,
                        final_status=status.value,
                    )
                    break
                if not ok:
                    observed_failure = True
                if (
                    self.config.version == CoderVersion.C3
                    and calls == (
                        self.config.max_model_calls - 2
                        if self.config.max_model_calls >= 4
                        else self.config.max_model_calls - 1
                    )
                    and automatic_validation_permitted(self._manifest(work))
                ):
                    # Reserve the final two model turns for an evidence-backed
                    # repair and executable integration. Without this check, a coder can spend its last
                    # turns on superficial checks and learn about a missing import
                    # or full-evaluator failure only after its call budget is gone.
                    validations = self._validate(work, events, deadline=deadline)
                    after_manifest = self._manifest(work)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, after_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status, reason = CoderExitStatus.SUCCESS, "penultimate_validation_passed"
                        break
                    observed_failure = True
                    feedback = {
                        "validation_failure": failed[-1] if failed else None,
                        "contract_violations": violations,
                        "instruction": (
                            "Use the remaining model turns for a concrete repair. If "
                            "scientific_action_not_implemented is present, connect the implementation "
                            "through the executable parent before any further helper refinement."
                        ),
                    }
                    repair_required_manifest = after_manifest
                    repair_required_violations = tuple(violations)
                    repair_required_failure = failed[-1] if failed else None
                    if repairs >= self.config.max_repairs:
                        reason = "repair_budget_exhausted"
                        break
                    repairs += 1
                    self._append_event(
                        events, "coder.repair_requested", repair=repairs,
                        feedback_sha256=coder_sha256(feedback),
                    )
                    continue
                if (
                    ok and self.config.version == CoderVersion.C1
                    and response.get("tool") in {"edit", "repair"}
                    and automatic_validation_permitted(self._manifest(work))
                ):
                    validations = self._validate(work, events, deadline=deadline)
                    after_manifest = self._manifest(work)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, after_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status, reason = CoderExitStatus.SUCCESS, "all_gates_passed"
                        break
                    observed_failure = True
                    repair_required_manifest = after_manifest
                    repair_required_violations = tuple(violations)
                    repair_required_failure = failed[-1] if failed else None
                    repairs += 1
                    feedback = {
                        "validation_failure": failed[-1] if failed else None,
                        "contract_violations": violations,
                    }
            else:
                # A model should not need to spend its final turn merely saying
                # "finish". If it made a real change before the call/tool cap,
                # the controller deterministically runs the same frozen gates.
                budget_manifest = self._manifest(work)
                if automatic_validation_permitted(budget_manifest):
                    validations = self._validate(work, events, deadline=deadline)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, budget_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status = CoderExitStatus.SUCCESS
                        reason = "controller_final_validation_at_budget"
                    else:
                        observed_failure = True
                        reason = "call_or_tool_budget_exhausted"
                else:
                    reason = "call_or_tool_budget_exhausted"

            # A provider/budget stop describes the conversation, not the
            # implementation already materialized in the workspace.  The
            # model may have completed and even exercised the program before
            # its next response was truncated or the remaining token budget
            # reached zero.  Apply the same authoritative controller gates as
            # natural call/tool exhaustion before rolling that work back.
            early_termination_reasons = {
                "finalization_reserve_reached",
                "budget_exhausted",
                "input_token_budget_exhausted",
                "output_token_budget_exhausted",
                "token_budget_exhausted",
                "model_response_not_object",
                "repair_budget_exhausted",
            }
            should_validate_early_termination = (
                status != CoderExitStatus.SUCCESS
                and (
                    reason in early_termination_reasons
                    or reason.startswith("model_response_error:")
                )
            )
            if should_validate_early_termination:
                terminal_manifest = self._manifest(work)
                workspace_changed = automatic_validation_permitted(terminal_manifest)
                if workspace_changed:
                    validations = self._validate(work, events, deadline=deadline)
                    violations, changed_paths = self._contract_violations(
                        work, before_manifest, terminal_manifest, parent_before
                    )
                    failed = self._blocking_validation_failures(validations)
                    if not failed and not violations:
                        status = CoderExitStatus.SUCCESS
                        reason = "controller_validation_after_early_termination"
                    else:
                        observed_failure = True
                    self._append_event(
                        events,
                        "coder.early_termination_validation_finished",
                        workspace_changed=True,
                        validation_passed=(not failed and not violations),
                        final_status=status.value,
                    )

            if (
                diagnostic_seed_root is not None
                and status != CoderExitStatus.SUCCESS
                and self._manifest(work) == initial_work_manifest
            ):
                self._append_event(
                    events, "coder.repair_finished_without_mutation",
                    initial_work_workspace_sha256=initial_work_sha256,
                    verified_parent_workspace_sha256=parent_workspace_sha,
                    validation_performed=bool(validations),
                )

            after_manifest = self._manifest(work)
            violations, changed_paths = self._contract_violations(
                work, before_manifest, after_manifest, parent_before
            )
            evaluator_invalid_categories = {"official_evaluation", "artifact_attestation", "grader_environment"}
            if status != CoderExitStatus.SUCCESS and any(
                item.get("harness_error")
                or item.get("failure_category") in evaluator_invalid_categories
                for item in validations
            ):
                status, reason = CoderExitStatus.EVALUATION_INVALID, "validation_harness_error"
            if status == CoderExitStatus.SUCCESS and not validations:
                status, reason = CoderExitStatus.EVALUATION_INVALID, "no_validation_evidence"
            diagnostic_workspace_sha = None
            diagnostic_artifact_relpath = None
            if status != CoderExitStatus.SUCCESS:
                # Preserve the attempted implementation before rollback.  The
                # search graph can therefore audit and learn from failed
                # branches without ever executing them as future parents.
                attempted_manifest = self._manifest(work)
                if attempted_manifest != before_manifest:
                    output_dir.mkdir(parents=True, exist_ok=True)
                    failed_tmp = output_dir / "failed_artifact.tmp"
                    failed_artifact = output_dir / "failed_artifact"
                    self._copy_manifest_workspace(work, failed_tmp, attempted_manifest)
                    failed_tmp.replace(failed_artifact)
                    diagnostic_workspace_sha = self._workspace_sha(attempted_manifest)
                    diagnostic_artifact_relpath = "failed_artifact"
                    self._append_event(
                        events,
                        "coder.failed_branch_preserved",
                        diagnostic_workspace_sha256=diagnostic_workspace_sha,
                        diagnostic_artifact_relpath=diagnostic_artifact_relpath,
                        changed_paths=changed_paths,
                    )
                shutil.rmtree(work)
                shutil.copytree(checkpoint, work)
                self._append_event(events, "coder.rollback_completed", parent_sha256=parent_sha)
            final_manifest = self._manifest(work)
            final_workspace_sha = self._workspace_sha(final_manifest)
            final_path = work / self.config.executable_parent
            final_sha = _file_sha(final_path) if final_path.is_file() else None
            parent_preserved = (
                status == CoderExitStatus.SUCCESS
                or (final_sha == parent_sha and final_workspace_sha == parent_workspace_sha)
            )
            self._append_event(
                events, "coder.finished", status=status.value, reason=reason,
                final_sha256=final_sha, final_workspace_sha256=final_workspace_sha,
            )
            output_dir.mkdir(parents=True, exist_ok=True)
            artifact_tmp = output_dir / "artifact.tmp"
            artifact = output_dir / "artifact"
            shutil.copytree(work, artifact_tmp)
            artifact_tmp.replace(artifact)

        trajectory_sha = coder_sha256([event["event_sha256"] for event in events])
        outcome = CoderOutcome(
            status=status,
            version=self.config.version,
            config_sha256=self.config.config_sha256,
            action_id=self.config.action.action_id,
            parent_sha256=parent_sha,
            parent_workspace_sha256=parent_workspace_sha,
            final_sha256=final_sha,
            final_workspace_sha256=final_workspace_sha,
            model_calls=calls,
            tool_calls=tools,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            wall_seconds=time.monotonic() - started,
            validation=validations,
            events=events,
            changed_paths=changed_paths,
            implementation_real=(status == CoderExitStatus.SUCCESS and not violations),
            contract_violations=violations,
            parent_preserved=parent_preserved,
            repaired_after_failure=(status == CoderExitStatus.SUCCESS and observed_failure),
            reason=reason,
            diagnostic_workspace_sha256=diagnostic_workspace_sha,
            diagnostic_artifact_relpath=diagnostic_artifact_relpath,
            trajectory_sha256=trajectory_sha,
        )
        outcome.seal()
        trajectory_tmp = output_dir / "trajectory.jsonl.tmp"
        with trajectory_tmp.open("w", encoding="utf-8") as handle:
            for event in events:
                handle.write(json.dumps(event, ensure_ascii=False, sort_keys=True) + "\n")
        trajectory_tmp.replace(output_dir / "trajectory.jsonl")
        result_tmp = output_dir / "result.json.tmp"
        result_tmp.write_text(
            json.dumps(asdict(outcome), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        result_tmp.replace(output_dir / "result.json")
        return outcome
