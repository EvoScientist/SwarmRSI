"""Explicit-run provenance; never a candidate or scientific verdict.

The raw tool receipts stay in the Coder event chain. This journal is a bounded
prompt projection of their provenance and recurrence, not an execution cache.
An exact, untruncated query supports the existing half-action tool checkpoint.
"""
from __future__ import annotations

import re
import json
import ast
from copy import deepcopy
from pathlib import PurePosixPath
from typing import Any

from .coder_models import coder_sha256


def _failure_scope(action: dict[str, Any], observation: dict[str, Any]) -> dict[str, str]:
    stderr = str(observation.get("stderr", ""))
    if observation.get("harness_error"):
        return {"kind": "runner_error", "scope": "execution_chain", "next_action":
                "Inspect the runner error. It is not evidence against the scientific method."}
    if observation.get("timed_out"):
        return {"kind": "timeout", "scope": "unknown", "next_action":
                "Locate the slow stage with a changed bounded probe or stage logs before rerunning. "
                "A longer limit alone does not establish a repair."}
    # An exception line is an observed diagnostic clue, not a proven root cause.
    exceptions = re.findall(r"(?m)^([A-Za-z_]\w*(?:Error|Exception))(?::[^\n]*)?$", stderr)
    locations = re.findall(r'File "([^"\n]+)", line (\d+)', stderr)
    if exceptions:
        exception = exceptions[-1]
        location = locations[-1][0] if locations else "unknown"
        argv = action.get("argv", [])
        inline = False
        if (argv and re.fullmatch(r"python(?:\d+(?:\.\d+)*)?", PurePosixPath(argv[0]).name)
                and "-c" in argv and argv.index("-c") + 1 < len(argv)
                and location == "<string>"
                and exception in {"SyntaxError", "IndentationError", "TabError"}):
            try:
                ast.parse(argv[argv.index("-c") + 1])
            except SyntaxError:
                inline = True
            except (ValueError, RecursionError):
                pass  # Unknown syntax/host-parser limitations are not evidence.
        if exception in {"SyntaxError", "IndentationError", "TabError"} and inline:
            advice = ("The inline diagnostic did not parse. Correct that command (use actual "
                      "newlines or a permitted helper script), then rerun the diagnostic. "
                      "This does not show that the candidate program has a syntax error.")
        else:
            advice = ("Use the observed traceback to locate the failing command or file; make a "
                      "targeted diagnostic or implementation correction. Do not infer a scientific "
                      "failure from this exception alone.")
        return {"kind": "observed_exception", "scope": "inline_command" if inline else "unknown",
                "exception": exception, "location": location, "next_action": advice}
    return {"kind": "nonzero_exit", "scope": "unknown", "next_action":
            "The exit code alone does not identify the cause (including 137). Obtain different "
            "evidence such as stage logs, input shapes or a bounded component probe; do not infer "
            "OOM, timeout or scientific invalidity without supporting evidence."}


def _command_context(action: dict[str, Any]) -> dict[str, Any]:
    """Exact requested command context, shared by recording and lookup."""
    return {"argv": list(action["argv"]), "cwd": action.get("cwd", "."),
            "timeout_seconds": action.get("timeout_seconds", 60.0)}


class CommandFailureJournal:
    """Record completed explicit commands; execution decisions remain in Coder."""

    def __init__(self) -> None:
        self._entries: dict[str, dict[str, Any]] = {}
        self._family_counts: dict[str, int] = {}

    def observe(
        self, action: dict[str, Any], observation: dict[str, Any], *,
        before_sha256: str, after_sha256: str, tool_call: int,
    ) -> dict[str, Any] | None:
        if action.get("tool") != "run":
            return None
        # Rejected tools and live-process polls are not terminal executions.
        if observation.get("process_status") == "running":
            return None
        executed = observation.get("execution_started")
        if executed is None:  # Legacy raw receipts used by offline replay.
            executed = (type(observation.get("returncode")) is int
                        or observation.get("timed_out") is True)
        if not executed:
            return None
        argv = action.get("argv")
        if not isinstance(argv, list) or not all(isinstance(x, str) for x in argv):
            return None
        command = _command_context(action)
        command_sha = coder_sha256(command)
        success = (observation.get("returncode") == 0
                   and not observation.get("timed_out") and not observation.get("harness_error"))
        raw_sha = coder_sha256(observation)
        if success:
            resolved = []
            for key, entry in self._entries.items():
                if (entry["command_sha256"] == command_sha
                        and entry["source_after_sha256"] == before_sha256 == after_sha256
                        and not entry.get("resolution")):
                    entry["resolution"] = {"tool_call": tool_call, "observation_sha256": raw_sha,
                                           "meaning": "same_command_succeeded_not_candidate_validated"}
                    resolved.append(key)
            return {"resolved_failure_ids": resolved, "observation_sha256": raw_sha,
                    "source_before_sha256": before_sha256, "source_after_sha256": after_sha256}
        stable = {key: value for key, value in observation.items() if key != "wall_seconds"}
        stable_sha = coder_sha256(stable)
        identity = {"command_sha256": command_sha, "source_before_sha256": before_sha256,
                    "source_after_sha256": after_sha256, "stable_observation_sha256": stable_sha}
        key = coder_sha256(identity)
        scope = _failure_scope(action, observation)
        family_sha = coder_sha256({
            "source_after_sha256": after_sha256,
            "kind": scope["kind"], "scope": scope["scope"],
            "exception": scope.get("exception"), "location": scope.get("location"),
            "returncode": observation.get("returncode"),
        })
        self._family_counts[family_sha] = self._family_counts.get(family_sha, 0) + 1
        if key not in self._entries:
            self._entries[key] = {
                **identity, "failure_id": key, "command": command, "failure_family_sha256": family_sha,
                "first_tool_call": tool_call, "repeat_count": 0, "classification": scope,
                "stderr_tail": str(observation.get("stderr", ""))[-800:],
                "stdout_tail": str(observation.get("stdout", ""))[-400:],
                "returncode": observation.get("returncode"),
                "timed_out": bool(observation.get("timed_out")),
            }
        entry = self._entries[key]
        entry.update({"last_tool_call": tool_call, "last_observation_sha256": raw_sha,
                      "execution_confirmed": observation.get("execution_started") is True,
                      "terminal_confirmed": type(observation.get("returncode")) is int
                      or observation.get("timed_out") is True,
                      "repeat_count": entry["repeat_count"] + 1})
        entry.pop("resolution", None)
        return deepcopy({**identity, "failure_id": key, "repeat_count": entry["repeat_count"],
                "observation_sha256": raw_sha, "failure_family_sha256": family_sha,
                "classification": scope})

    def unchanged_failed_command(
        self, action: dict[str, Any], current_sha256: str,
    ) -> dict[str, Any] | None:
        """Return exact current failure evidence, not a semantic failure class.

        Unlike project(), this query reads the complete journal. Legacy receipts
        lacking explicit execution confirmation remain advisory only. A command
        that mutated A to B is not evidence about starting it again from B.
        """
        argv = action.get("argv")
        if (action.get("tool") != "run" or not isinstance(argv, list)
                or not argv or not all(isinstance(item, str) for item in argv)):
            return None
        command_sha = coder_sha256(_command_context(action))
        matching = [entry for entry in self._entries.values()
                    if entry["command_sha256"] == command_sha
                    and entry["source_after_sha256"] == current_sha256]
        if not matching:
            return None
        latest = max(matching, key=lambda entry: max(
            entry["last_tool_call"], entry.get("resolution", {}).get("tool_call", 0)
        ))
        if (latest.get("resolution") or not latest.get("execution_confirmed")
                or not latest.get("terminal_confirmed")
                or latest["source_before_sha256"] != current_sha256
                or latest["classification"]["kind"] == "runner_error"):
            return None
        return deepcopy({key: latest[key] for key in (
            "failure_id", "command_sha256", "source_before_sha256",
            "source_after_sha256", "last_tool_call", "last_observation_sha256",
        )})

    def project(self, current_sha256: str, *, max_chars: int = 6000) -> dict[str, Any] | None:
        if type(max_chars) is not int or max_chars < 1000:
            raise ValueError("command feedback projection requires at least 1000 characters")
        if not self._entries:
            return None
        # Full entries are not copied into the model request; raw evidence is in
        # the event chain. The latest unresolved current-version failures win.
        result: dict[str, Any] = {
            "schema_version": "coder-explicit-run-feedback-v1",
            "authority": "diagnostic_only_not_candidate_or_scientific_verdict",
            "current_source_workspace_sha256": current_sha256,
            "total_unique_failures": len(self._entries), "omitted_failures": len(self._entries),
            "entries": [],
        }
        entries = sorted(self._entries.values(), key=lambda x: (
            x["source_after_sha256"] == current_sha256 and not x.get("resolution"),
            x["last_tool_call"],
        ), reverse=True)
        for entry in entries:
            if len(result["entries"]) >= 3:
                break
            current = entry["source_after_sha256"] == current_sha256
            command_text = json.dumps(entry["command"], ensure_ascii=False)
            row = {key: entry[key] for key in (
                "failure_id", "command_sha256", "source_before_sha256", "source_after_sha256",
                "stable_observation_sha256", "last_observation_sha256", "first_tool_call",
                "last_tool_call", "repeat_count", "returncode", "timed_out", "classification",
                "stderr_tail", "stdout_tail",
            )}
            row.update({"state": "stale" if not current else (
                "resolved_command_only" if entry.get("resolution") else "unresolved"),
                "command_excerpt": command_text[:600],
                "command_omitted_chars": max(0, len(command_text) - 600),
                "observed_family_failure_count": self._family_counts[entry["failure_family_sha256"]],
                "family_count_is_not_root_cause_equivalence": True,
            })
            if entry.get("resolution"):
                row["resolution"] = entry["resolution"]
            result["entries"].append(deepcopy(row))
            result["omitted_failures"] -= 1
            if len(json.dumps(result, ensure_ascii=False, sort_keys=True)) > max_chars:
                result["entries"].pop()
                result["omitted_failures"] += 1
                if not result["entries"]:
                    # Even the smallest supported budget should show the
                    # actionable failure, not only a count of omitted rows.
                    minimal = {"failure_id": row["failure_id"], "state": row["state"],
                               "repeat_count": row["repeat_count"], "returncode": row["returncode"],
                               "kind": row["classification"]["kind"],
                               "scope": row["classification"]["scope"],
                               "next_action_excerpt": row["classification"]["next_action"][:100],
                               "detail_fields_omitted": True}
                    result["entries"].append(minimal)
                    result["omitted_failures"] -= 1
                    if len(json.dumps(result, ensure_ascii=False, sort_keys=True)) > max_chars:
                        result["entries"].pop()
                        result["omitted_failures"] += 1
                break
        return result
