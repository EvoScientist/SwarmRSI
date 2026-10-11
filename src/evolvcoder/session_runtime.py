"""Controller-owned elapsed clock and auditable planner transport side ledger.

These are execution controls, not changes to scientific selection or to the
historical completed-response/accounted-execution-wall counters.
"""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re
import time

from .contracts import canonical_sha256


def _nonnegative(value, name):
    if type(value) not in (int, float) or not math.isfinite(value) or value < 0:
        raise ValueError(f"invalid session {name}")
    return value


def runtime_clock_controls(declared_wall, trial_watchdog):
    if _nonnegative(trial_watchdog, "trial watchdog") <= 0:
        raise ValueError("invalid trial watchdog")
    reserve = min(30.0, .05 * trial_watchdog)
    inner = math.inf if declared_wall is None else _nonnegative(declared_wall, "declared wall")
    execution = min(inner, trial_watchdog) - reserve
    if execution <= 0:
        raise ValueError("session finalization reserve does not fit declared wall")
    return {"declared_inner_wall_seconds": declared_wall,
            "trial_watchdog_seconds": trial_watchdog,
            "session_finalize_reserve_seconds": reserve,
            "session_execution_wall_seconds": execution}


class SessionClock:
    def __init__(self, *, started_unix, started_monotonic, wall_limit, identity,
                 runtime_controls=None, trial_started_unix=None):
        _nonnegative(started_unix, "start time")
        if wall_limit is not None and _nonnegative(wall_limit, "wall limit") <= 0:
            raise ValueError("invalid session wall limit")
        self.origin = {"schema_version": "evolvcoder-session-clock-v1",
                       "started_at_unix": started_unix, "wall_limit_seconds": wall_limit,
                       "cell_identity_sha256": canonical_sha256(identity)}
        execution_limit = wall_limit
        if runtime_controls is not None:
            expected = runtime_clock_controls(wall_limit, runtime_controls.get("trial_watchdog_seconds"))
            if runtime_controls != expected:
                raise ValueError("session runtime clock controls mismatch")
            execution_limit = expected["session_execution_wall_seconds"]
            if trial_started_unix is not None:
                _nonnegative(trial_started_unix, "trial start time")
                if trial_started_unix > started_unix:
                    raise ValueError("trial start cannot follow session start")
                # Subtract interpreter/import/startup time as well. The trusted
                # controller timestamp is not passed into candidate environments.
                absolute_remaining = (trial_started_unix + expected["trial_watchdog_seconds"]
                                      - expected["session_finalize_reserve_seconds"] - started_unix)
                execution_limit = max(0.0, min(execution_limit, absolute_remaining))
            self.origin.update(runtime_controls=expected, trial_started_at_unix=trial_started_unix,
                               execution_wall_seconds=execution_limit)
        self._started_monotonic = started_monotonic
        self._prior_elapsed = 0.0

    @classmethod
    def restore(cls, *, origin, snapshot, wall_limit, identity, runtime_controls=None):
        if not isinstance(origin, dict) or not isinstance(snapshot, dict):
            raise ValueError("session clock missing; refusing ambiguous resume")
        obj = cls(started_unix=origin.get("started_at_unix"),
                  started_monotonic=time.monotonic(), wall_limit=wall_limit, identity=identity,
                  runtime_controls=runtime_controls, trial_started_unix=origin.get("trial_started_at_unix"))
        if origin != obj.origin or snapshot.get("origin") != origin:
            raise ValueError("session clock identity mismatch")
        recorded = _nonnegative(snapshot.get("recorded_at_unix"), "recorded time")
        elapsed = _nonnegative(snapshot.get("elapsed_seconds"), "elapsed time")
        now = time.time()
        if recorded < origin["started_at_unix"] or now < recorded:
            raise ValueError("session clock moved backwards; refusing ambiguous resume")
        obj._prior_elapsed = elapsed + now - recorded  # Downtime is not free budget.
        return obj

    def elapsed(self):
        return self._prior_elapsed + max(0.0, time.monotonic() - self._started_monotonic)

    @property
    def deadline_monotonic(self):
        limit = self.origin.get("execution_wall_seconds", self.origin["wall_limit_seconds"])
        return None if limit is None else self._started_monotonic + limit - self._prior_elapsed

    def remaining(self):
        limit = self.origin.get("execution_wall_seconds", self.origin["wall_limit_seconds"])
        return math.inf if limit is None else max(0.0, limit - self.elapsed())

    def residual(self, budget, accounting):
        result = budget.residual(accounting)
        result["remaining_wall"] = min(result["remaining_wall"], self.remaining())
        return result

    def snapshot(self):
        now = time.time()
        if now < self.origin["started_at_unix"]:
            raise ValueError("session clock moved backwards")
        return {"origin": self.origin, "recorded_at_unix": now,
                "elapsed_seconds": self.elapsed()}


def safe_transport_usage(usage):
    return {key: value for key, value in usage.items()
            if key in {"input_tokens", "output_tokens", "cost_usd", "cost"}
            and type(value) in (int, float) and math.isfinite(value) and value >= 0}


def transport_snapshot(audit_root: Path, completed_batches: int) -> dict:
    """Derive an exact hash-bound receipt, including pending requests and attempts.

    A started-only record is legitimate unknown usage, never a completed call.
    Uncheckpointed batches remain ineligible for automatic model reissue.
    """
    artifacts, groups = [], {}
    if audit_root.is_symlink() or (audit_root / "transport_attempts").is_symlink():
        raise ValueError("planner transport root cannot be a symlink")

    def read(path):
        if path.is_symlink() or not path.is_file():
            raise ValueError("planner transport artifact must be a regular file")
        raw = path.read_bytes()
        artifacts.append({"path": path.relative_to(audit_root).as_posix(),
                          "sha256": hashlib.sha256(raw).hexdigest()})
        return json.loads(raw)

    for path in sorted((audit_root / "transport_attempts").glob("*")):
        match = re.fullmatch(r"batch_(\d{6})\.attempt_(\d{3})\.(started|outcome)\.json", path.name)
        if not match:
            raise ValueError("unexpected planner transport artifact")
        batch, attempt, kind = int(match[1]), int(match[2]), match[3]
        row = read(path)
        if (batch > completed_batches or row.get("batch") != batch or row.get("attempt") != attempt
                or row.get("schema_version") != "evolvcoder-planner-transport-attempt-v1"):
            raise ValueError("planner transport identity mismatch")
        _nonnegative(row.get("timestamp_unix"), "transport timestamp")
        groups.setdefault(batch, {}).setdefault(attempt, {})[kind] = row

    if not set(range(completed_batches)).issubset(groups):
        raise ValueError("completed planner batch lacks transport ledger")
    expected_names = {f"batch_{batch:06d}.{kind}.json" for batch in range(completed_batches)
                      for kind in ("request", "response", "record")}
    expected_names.add(f"batch_{completed_batches:06d}.request.json")
    if any(path.name not in expected_names for path in audit_root.glob("batch_*.json")):
        raise ValueError("unexpected planner batch artifact")

    pending, request_cache = [], {}
    for batch in sorted(set(groups) | {completed_batches}):
        request_path = audit_root / f"batch_{batch:06d}.request.json"
        if request_path.exists():
            request_cache[batch] = read(request_path)
            if batch == completed_batches:
                if any((audit_root / f"batch_{batch:06d}.{kind}.json").exists()
                       for kind in ("response", "record")):
                    raise ValueError("uncheckpointed planner response; refusing terminal shortcut")
                pending.append(batch)
        elif batch in groups:
            raise ValueError("planner transport request missing")

    attempts_total = unresolved = not_sent = responses = 0
    known_costs = []
    for batch, attempts in groups.items():
        if sorted(attempts) != list(range(len(attempts))):
            raise ValueError("planner transport attempt sequence mismatch")
        request_sha = canonical_sha256(request_cache[batch])
        response_count = 0
        for number, pair in sorted(attempts.items()):
            started, outcome = pair.get("started"), pair.get("outcome")
            if started is None or started.get("provider_cost_status") != "unknown":
                raise ValueError("planner transport start record missing or invalid")
            if any(row.get("request_sha256") != request_sha for row in pair.values()):
                raise ValueError("planner transport request binding mismatch")
            attempts_total += 1
            if outcome is None:
                unresolved += 1
                if number != len(attempts) - 1:
                    raise ValueError("planner transport retried unresolved in-flight attempt")
                continue
            status, cost_status = outcome.get("status"), outcome.get("provider_cost_status")
            if status == "request_not_sent" and cost_status == "not_sent":
                not_sent += 1
                if number != len(attempts) - 1:
                    raise ValueError("planner transport retried after session deadline")
            elif status == "transport_failed" and cost_status == "unknown":
                unresolved += 1
            elif status == "response_received":
                usage = outcome.get("usage", {})
                if usage != safe_transport_usage(usage):
                    raise ValueError("invalid planner transport usage")
                expected = "reported" if any(k in usage for k in ("cost", "cost_usd")) else "unknown"
                if cost_status != expected:
                    raise ValueError("planner transport cost status mismatch")
                response_count += 1
                responses += 1
                if "cost_usd" in usage:
                    known_costs.append(usage["cost_usd"])
                else:
                    unresolved += 1
                if number != len(attempts) - 1:
                    raise ValueError("planner transport retried a completed response")
                response_path = audit_root / f"batch_{batch:06d}.response.json"
                response = read(response_path)
                if usage != safe_transport_usage(response["usage"]):
                    raise ValueError("planner transport response usage mismatch")
            else:
                raise ValueError("invalid planner transport outcome")
        if batch < completed_batches and response_count != 1:
            raise ValueError("completed planner batch lacks received transport response")
    return {"schema_version": "evolvcoder-planner-transport-receipt-v1",
            "artifacts": sorted(artifacts, key=lambda row: row["path"]),
            "pending_request_batches": pending,
            "transport_attempts": attempts_total, "requests_not_sent": not_sent,
            "responses_received": responses, "unresolved_cost_attempts": unresolved,
            "known_cost_usd_subtotal": math.fsum(known_costs),
            "cost_scope": "planner only; known USD subtotal excludes unresolved attempts; not total session cost"}


def verify_transport_snapshot(audit_root, completed_batches, receipt):
    if receipt != transport_snapshot(audit_root, completed_batches):
        raise ValueError("planner transport receipt mismatch")
