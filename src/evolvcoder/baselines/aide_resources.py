"""Whole-run resource boundaries; no AIDE search or node selection here.

Callers hold ``aide_run_lock`` throughout initialization, search and export.
Budget checks belong at new external operations, never in front of pure replay.
"""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import json
import math
import os
from pathlib import Path
import signal
import subprocess
from typing import Any, Iterator

from .aide_state import atomic_json, seal, unseal
from .aide_transport import AideCallLedger, AideResourceStop, AideTransportUnresolved


@contextmanager
def aide_run_lock(root: Path) -> Iterator[Path]:
    """Reject a second owner before it can initialize ledger/session/workspace."""
    root = root.resolve()
    root.mkdir(parents=True, exist_ok=True)
    flags = os.O_CREAT | os.O_RDWR | getattr(os, "O_NOFOLLOW", 0)
    descriptor = os.open(root / "controller.lock", flags, 0o600)
    try:
        try:
            fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError:
            raise RuntimeError("AIDE run already has an active controller") from None
        yield root
    finally:
        # Never unlink the lock: another owner must lock the same inode.
        os.close(descriptor)


def _signal_group(pid: int, sig: signal.Signals) -> None:
    if pid <= 0:
        raise ValueError("a process group must be an owned positive child PID")
    try:
        os.killpg(pid, sig)
    except ProcessLookupError:
        pass


def run_bounded_process(
    args: list[str] | tuple[str, ...], *, timeout: float,
    input: str | None = None, cwd: Path | str | None = None,
    env: dict[str, str] | None = None, capture_output: bool = True,
    text: bool = True, check: bool = False, teardown_seconds: float = 1.0,
) -> subprocess.CompletedProcess:
    """Run one owned session, terminate its process group even on normal exit.

``timeout`` includes teardown allowance. On expiry raise TimeoutExpired, so
existing execution/transport code preserves an unresolved receipt. This bounds
external processes, not time spent in unrelated Python/filesystem operations.
    """
    if (isinstance(timeout, bool) or not isinstance(timeout, (int, float))
            or not math.isfinite(timeout) or timeout <= 0):
        raise ValueError("finite positive process timeout required")
    if not 0 < teardown_seconds < timeout:
        raise ValueError("process timeout must include positive teardown headroom")
    if not capture_output or not text:
        raise ValueError("AIDE process receipts require captured text output")
    process = subprocess.Popen(
        args, cwd=cwd, env=env, stdin=subprocess.PIPE if input is not None else subprocess.DEVNULL,
        stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True,
        start_new_session=True,
    )
    try:
        try:
            stdout, stderr = process.communicate(input, timeout=timeout - teardown_seconds)
        except subprocess.TimeoutExpired:
            _signal_group(process.pid, signal.SIGTERM)
            try:
                stdout, stderr = process.communicate(timeout=teardown_seconds / 2)
            except subprocess.TimeoutExpired:
                _signal_group(process.pid, signal.SIGKILL)
                stdout, stderr = process.communicate(timeout=teardown_seconds / 2)
            raise subprocess.TimeoutExpired(args, timeout, output=stdout, stderr=stderr) from None
        result = subprocess.CompletedProcess(args, process.returncode, stdout, stderr)
        if check:
            result.check_returncode()
        return result
    finally:
        # Also reap a child that outlives a normally completed parent or ignores
        # SIGTERM. These are only descendants in this invocation's new group.
        _signal_group(process.pid, signal.SIGKILL)
        if process.poll() is None:
            process.wait(timeout=teardown_seconds)


class AideOperationBudget:
    """Share the durable provider deadline with execution and scorer attempts."""

    def __init__(self, *, ledger: AideCallLedger, artifacts: Path,
                 max_score_attempts: int) -> None:
        if (isinstance(max_score_attempts, bool) or not isinstance(max_score_attempts, int)
                or max_score_attempts < 0):
            raise ValueError("nonnegative integer score-attempt cap required")
        self.ledger, self.artifacts = ledger, artifacts.resolve()
        self.max_score_attempts = max_score_attempts
        self.path = ledger.root / "operation_budget.json"
        self.identity = {"schema_version": "aide-operation-budget-v1",
                         "ledger_identity": ledger.identity,
                         "artifacts": str(self.artifacts),
                         "max_score_attempts": max_score_attempts}
        if self.path.exists():
            if unseal(json.loads(self.path.read_text())) != self.identity:
                raise ValueError("AIDE operation budget changed on resume")
        else:
            atomic_json(self.path, seal(self.identity))

    def score_attempts(self) -> int:
        rows = list(self.artifacts.glob("*/score_started.json"))
        for path in rows:
            if path.is_symlink() or path.parent.is_symlink():
                raise ValueError("scorer attempt receipts cannot be symlinks")
            row = unseal(json.loads(path.read_text()))
            if row.get("node_id") != path.parent.name:
                raise ValueError("scorer attempt belongs to a different node")
        # A started-but-unresolved scorer still consumes an attempt. Never
        # reclaim it merely because no completed scientific score is available.
        return len(rows)

    def remaining_wall(self) -> float:
        state = self.ledger.read()
        return float(self.ledger.identity["wall_seconds"] -
                     (self.ledger.clock() - state["started_at"]))

    def before_new_search(self) -> None:
        """Only for a ready session; deterministic pending replay bypasses this."""
        state = self.ledger.read()
        if any(row["state"] not in ("completed", "failed") for row in state["attempts"]):
            raise AideTransportUnresolved("prior provider operation remains unresolved")
        if self.remaining_wall() <= 0:
            raise AideResourceStop("wall_budget_exhausted")
        if self.score_attempts() >= self.max_score_attempts:
            raise AideResourceStop("score_attempt_budget_exhausted")
        if len(state["attempts"]) >= self.ledger.identity["max_calls"]:
            raise AideResourceStop("model_call_budget_exhausted")
        if sum(row["output_tokens"] for row in state["attempts"]) >= self.ledger.identity["max_output_tokens"]:
            raise AideResourceStop("output_token_budget_exhausted")

    def external_timeout(self, *, kind: str, configured_seconds: int,
                         execution_only: bool = False) -> tuple[int, float]:
        if kind not in {"execute", "score"}:
            raise ValueError("unknown AIDE external operation")
        if (isinstance(configured_seconds, bool) or not isinstance(configured_seconds, int)
                or configured_seconds <= 0):
            raise ValueError("positive integer configured execution timeout required")
        if not execution_only and self.score_attempts() >= self.max_score_attempts:
            raise AideResourceStop("score_attempt_budget_exhausted")
        return self.wall_timeout(configured_seconds=configured_seconds)

    def wall_timeout(self, *, configured_seconds: int) -> tuple[int, float]:
        """Refresh at actual process dispatch, after the score-slot reservation."""
        if (isinstance(configured_seconds, bool) or not isinstance(configured_seconds, int)
                or configured_seconds <= 0):
            raise ValueError("positive integer configured execution timeout required")
        # Preserve the original Interpreter timeout where it fits. Existing
        # helper/scorer overhead is bounded inside, not added to, remaining wall.
        remaining = self.remaining_wall()
        inner = min(configured_seconds, math.floor(remaining - 30.0))
        if inner < 1:
            raise AideResourceStop("insufficient_remaining_execution_wall")
        return inner, min(float(inner + 30), remaining)
