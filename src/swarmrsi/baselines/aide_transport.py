"""Durable accounting at each actual provider attempt, before AIDE parsing."""
from __future__ import annotations

import fcntl
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping

from .aide_state import atomic_json, canonical, seal, unseal


class AideResourceStop(RuntimeError):
    pass


class AideTransportRetryable(RuntimeError):
    """A provider call failed outright; the attempt is recorded and the same request may be retried."""


class AideTransportUnresolved(RuntimeError):
    """Not an SDK exception: upstream's unbounded backoff must not retry it."""


def _dump(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return value.model_dump(mode="json")
    if isinstance(value, dict):
        return {key: _dump(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_dump(item) for item in value]
    if hasattr(value, "__dict__"):
        return _dump(vars(value))
    return value


def _tokens(value: Any) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError("missing or invalid provider token usage")
    return value


class AideCallLedger:
    def __init__(self, root: Path, *, model: str, max_calls: int,
                 max_output_tokens: int, wall_seconds: float,
                 per_call_output_tokens: int, create: Callable[..., Any],
                 clock: Callable[[], float] = time.time,
                 retry_failed_attempts: bool = False) -> None:
        if any(isinstance(n, bool) or not isinstance(n, int) or n <= 0
               for n in (max_calls, max_output_tokens, per_call_output_tokens)):
            raise ValueError("AIDE call/token limits must be positive integers")
        if not math.isfinite(wall_seconds) or wall_seconds <= 0:
            raise ValueError("AIDE wall limit must be positive and finite")
        self.root, self.clock, self._create = root, clock, create
        # Off by default: an uncertain provider outcome may already be accepted and billed, so it
        # stays unresolved and stops the run. A caller may opt in to recording it as a failed
        # attempt (counted against the call budget) and retrying, as BRS retries planner transport.
        self.retry_failed_attempts = retry_failed_attempts
        self.path = root / "ledger.json"
        self.identity = {
            "model": model, "max_calls": max_calls,
            "max_output_tokens": max_output_tokens, "wall_seconds": wall_seconds,
            "per_call_output_tokens": per_call_output_tokens,
        }
        root.mkdir(parents=True, exist_ok=True)
        with (root / "ledger.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            if not self.path.exists():
                self._save({"schema_version": "aide-provider-ledger-v1",
                            "identity": self.identity, "started_at": clock(),
                            "attempts": []})
            self.read()

    def _save(self, value: Mapping[str, Any]) -> None:
        atomic_json(self.path, seal(value))

    def read(self) -> dict[str, Any]:
        value = unseal(json.loads(self.path.read_text(encoding="utf-8")))
        if value.get("schema_version") != "aide-provider-ledger-v1" or value["identity"] != self.identity:
            raise ValueError("AIDE transport identity changed on resume")
        return value

    def create(self, **kwargs: Any) -> Any:
        with (self.root / "ledger.lock").open("a+") as lock:
            fcntl.flock(lock, fcntl.LOCK_EX)
            ledger = self.read()
            attempts = ledger["attempts"]
            if any(row["state"] not in ("completed", "failed") for row in attempts):
                raise AideTransportUnresolved("prior provider attempt requires reconciliation")
            if kwargs.get("model") != self.identity["model"]:
                raise ValueError("AIDE attempted an unapproved backbone")
            elapsed = self.clock() - ledger["started_at"]
            remaining_time = self.identity["wall_seconds"] - elapsed
            remaining_output = self.identity["max_output_tokens"] - sum(
                row["output_tokens"] or 0 for row in attempts
            )
            if len(attempts) >= self.identity["max_calls"]:
                raise AideResourceStop("model_call_budget_exhausted")
            if remaining_time <= 0:
                raise AideResourceStop("wall_budget_exhausted")
            if remaining_output <= 0:
                raise AideResourceStop("output_token_budget_exhausted")
            logical_request = _dump(kwargs)
            request = dict(kwargs)
            requested_max = request.get("max_tokens")
            if requested_max is not None and (
                isinstance(requested_max, bool) or not isinstance(requested_max, int)
                or requested_max <= 0
            ):
                raise ValueError("invalid AIDE max_tokens")
            request["max_tokens"] = min(
                requested_max or self.identity["per_call_output_tokens"], remaining_output,
                self.identity["per_call_output_tokens"],
            )
            requested_timeout = request.get("timeout", remaining_time)
            if not isinstance(requested_timeout, (int, float)) or not math.isfinite(requested_timeout):
                raise ValueError("AIDE transport requires a finite numeric timeout")
            request["timeout"] = min(float(requested_timeout), remaining_time)
            if request["timeout"] <= 0:
                raise AideResourceStop("request_deadline_exhausted")
            index = len(attempts)
            request_path = self.root / f"{index:06d}.request.json"
            response_path = self.root / f"{index:06d}.response.json"
            request_record = {"logical_request": logical_request, "request": _dump(request)}
            atomic_json(request_path, seal(request_record))
            row = {"index": index, "state": "started", "started_at": self.clock(),
                   "request_sha256": canonical(request_record),
                   "input_tokens": None, "output_tokens": None}
            attempts.append(row)
            self._save(ledger)
            try:
                response = self._create(**request)
            except Exception as exc:
                if not self.retry_failed_attempts:
                    row.update(state="unresolved", error_type=type(exc).__name__,
                               elapsed_seconds=self.clock() - row["started_at"])
                    self._save(ledger)
                    raise AideTransportUnresolved(
                        f"provider attempt {index} has unknown acceptance/usage ({type(exc).__name__})"
                    ) from None
                row.update(state="failed", error_type=type(exc).__name__,
                           elapsed_seconds=self.clock() - row["started_at"])
                self._save(ledger)
                raise AideTransportRetryable(
                    f"provider attempt {index} failed ({type(exc).__name__})"
                ) from None
            except BaseException as exc:
                row.update(state="unresolved", error_type=type(exc).__name__,
                           elapsed_seconds=self.clock() - row["started_at"])
                self._save(ledger)
                # No exception body: SDK messages may contain request secrets.
                raise AideTransportUnresolved(
                    f"provider attempt {index} has unknown acceptance/usage ({type(exc).__name__})"
                ) from None
            try:
                raw = _dump(response)
                atomic_json(response_path, seal({"response": raw}))
                row["response_sha256"] = canonical({"response": raw})
                usage = raw.get("usage") or {}
                row["input_tokens"] = _tokens(usage.get("prompt_tokens"))
                row["output_tokens"] = _tokens(usage.get("completion_tokens"))
                row["reported_model"] = raw.get("model")
                row["state"] = "completed"
                if row["output_tokens"] > request["max_tokens"]:
                    row["state"] = "usage_exceeds_reservation"
                if raw.get("model") != self.identity["model"]:
                    row["state"] = "reported_model_drift"
            except Exception as exc:
                row.update(state="usage_unresolved", error_type=type(exc).__name__)
                self._save(ledger)
                raise AideTransportUnresolved(f"provider attempt {index} lacks reliable saved usage") from None
            row["elapsed_seconds"] = self.clock() - row["started_at"]
            self._save(ledger)
            if row["state"] != "completed":
                raise AideTransportUnresolved(f"provider response requires reconciliation: {row['state']}")
            # Even malformed tool JSON is now durable and charged before parsing.
            return response

    def replay(self, index: int, *, logical_request: Mapping[str, Any],
               response_factory: Callable[[dict[str, Any]], Any]) -> Any:
        """Reconsume one recorded response, never create a second provider call."""
        row = self.read()["attempts"][index]
        if row["state"] != "completed":
            raise AideTransportUnresolved("cannot replay an unresolved provider response")
        request = unseal(json.loads((self.root / f"{index:06d}.request.json").read_text()))
        response = unseal(json.loads((self.root / f"{index:06d}.response.json").read_text()))
        if canonical(request) != row["request_sha256"] or canonical(response) != row["response_sha256"]:
            raise ValueError("AIDE call record does not match ledger")
        if request["logical_request"] != _dump(dict(logical_request)):
            raise ValueError("AIDE replay request differs from original request")
        return response_factory(response["response"])
