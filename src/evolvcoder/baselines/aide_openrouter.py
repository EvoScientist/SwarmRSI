from __future__ import annotations

import hashlib
import json
import math
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Protocol


class CallRecorder(Protocol):
    def __call__(self, event: Mapping[str, Any]) -> None: ...


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def assert_upstream_hashes(root: Path, expected: Mapping[str, str]) -> None:
    """Fail closed if the provenance-locked AIDE sources have changed."""
    mismatches: dict[str, dict[str, str | None]] = {}
    for relative, digest in expected.items():
        path = root / relative
        actual = sha256_file(path) if path.is_file() else None
        if actual != digest:
            mismatches[relative] = {"expected": digest, "actual": actual}
    if mismatches:
        raise RuntimeError(
            "AIDE upstream source attestation mismatch: "
            + json.dumps(mismatches, sort_keys=True)
        )


# Upstream AIDE's OpenRouter backend pins Fireworks. On deepseek/deepseek-v4.1-flash that endpoint caps a
# request at 8192 total tokens, so a review that reasons for a few thousand tokens ends with
# finish_reason=length and no tool call, and the run stops. Callers pass provider=None to use
# OpenRouter's default routing, as every other method in the comparison does.
UPSTREAM_PROVIDER_PREFERENCE = {"order": ["Fireworks"], "ignore": ["Together", "DeepInfra", "Hyperbolic"]}


def make_openrouter_query(
    *,
    client: Any,
    backoff_create: Callable[..., Any],
    timeout_exceptions: tuple[type[BaseException], ...],
    recorder: CallRecorder | None = None,
    provider: Mapping[str, Any] | None = UPSTREAM_PROVIDER_PREFERENCE,
    reasoning: Mapping[str, Any] | None = None,
    missing_tool_call_retries: int = 0,
    reasoning_param: str = "openrouter",
    tool_choice: str = "named",
    request_timeout: float | None = None,
) -> Callable[..., tuple[Any, float, int, int, dict[str, Any]]]:
    """Add Chat Completions tool parsing without changing AIDE search policy.

    Message construction and the Fireworks provider preference intentionally
    match the locked upstream OpenRouter backend.  The sole behavioral addition
    is the same OpenAI-compatible tool-call path already used by AIDE's OpenAI
    backend.

    reasoning_param "reasoning_effort" and tool_choice "auto" serve an endpoint such as
    api.deepseek.com, which takes OpenAI's reasoning_effort instead of OpenRouter's
    reasoning object and, while thinking, rejects a named tool_choice (measured
    2026-09-19). The defaults keep the OpenRouter request unchanged.
    """
    if reasoning_param not in ("openrouter", "reasoning_effort"):
        raise ValueError("reasoning_param must be 'openrouter' or 'reasoning_effort'")
    if tool_choice not in ("named", "auto"):
        raise ValueError("tool_choice must be 'named' or 'auto'")

    def query(
        system_message: str | None,
        user_message: str | None,
        func_spec: Any | None = None,
        **model_kwargs: Any,
    ) -> tuple[Any, float, int, int, dict[str, Any]]:
        filtered_kwargs = {
            key: value for key, value in model_kwargs.items() if value is not None
        }
        messages = [
            {"role": "user", "content": message}
            for message in (system_message, user_message)
            if message
        ]
        # A per-request ceiling: without one the transport uses the whole remaining wall, and a
        # request the endpoint never answers ends the run (DeepSeek official API, 2026-09-28/29).
        if request_timeout is not None:
            filtered_kwargs.setdefault("timeout", float(request_timeout))
        if func_spec is not None:
            filtered_kwargs["tools"] = [func_spec.as_openai_tool_dict]
            filtered_kwargs["tool_choice"] = (
                func_spec.openai_tool_choice_dict if tool_choice == "named" else "auto")

        extra_body: dict[str, Any] = {}
        if provider:
            extra_body["provider"] = dict(provider)
        if reasoning and reasoning_param == "reasoning_effort":
            extra_body["reasoning_effort"] = str(reasoning.get("effort", "medium"))
        elif reasoning:
            extra_body["reasoning"] = dict(reasoning)
        started = time.monotonic()
        input_tokens = output_tokens = 0
        # A reasoning model can spend its whole output budget thinking and return no tool
        # call (or, for code generation, empty content); upstream then fails parsing it and
        # the run stops.
        # The same request is retried a bounded number of times, every attempt charged.
        for _attempt in range(1 + missing_tool_call_retries):
            try:
                completion = backoff_create(
                    client.chat.completions.create,
                    timeout_exceptions,
                    messages=messages,
                    **({"extra_body": extra_body} if extra_body else {}),
                    **filtered_kwargs,
                )
            except Exception as exc:
                # Only a recorded, failed provider attempt is retried; anything unresolved still stops.
                if type(exc).__name__ != "AideTransportRetryable" or _attempt == missing_tool_call_retries:
                    raise
                continue
            input_tokens += int(completion.usage.prompt_tokens)
            output_tokens += int(completion.usage.completion_tokens)
            message = completion.choices[0].message
            output: Any = message.content
            tool_calls = getattr(message, "tool_calls", None)
            if func_spec is not None and tool_calls:
                call = tool_calls[0]
                if call.function.name == func_spec.name:
                    output = json.loads(call.function.arguments)
            if func_spec is not None and isinstance(output, dict):
                break
            if func_spec is None and isinstance(output, str) and output.strip():
                break
        elapsed = time.monotonic() - started
        info = {
            "system_fingerprint": getattr(completion, "system_fingerprint", None),
            "model": completion.model,
            "created": getattr(completion, "created", None),
        }
        if recorder is not None:
            recorder({
                "schema_version": "aide-openrouter-call-v1",
                "model": completion.model,
                "request_sha256": hashlib.sha256(
                    json.dumps(
                        {
                            "messages": messages,
                            "model_kwargs": filtered_kwargs,
                            "function": getattr(func_spec, "name", None),
                        },
                        sort_keys=True,
                        default=str,
                    ).encode()
                ).hexdigest(),
                "function": getattr(func_spec, "name", None),
                "tool_call_received": bool(tool_calls),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "wall_seconds": elapsed,
                "provider_policy": "upstream_fireworks_preference" if provider == UPSTREAM_PROVIDER_PREFERENCE
                else ("none" if not provider else "profile"),
                "reasoning_param": reasoning_param, "tool_choice": tool_choice,
            })
        return output, elapsed, input_tokens, output_tokens, info

    return query


def install_openrouter_query(aide_backend: Any, query: Callable[..., Any]) -> None:
    """Install only the backend adapter; leave Agent and Journal untouched."""
    aide_backend.backend_openrouter.query = query
    aide_backend.provider_to_query_func["openrouter"] = query


def apply_official_evaluation(
    *,
    node: Any,
    evaluation: Mapping[str, Any],
    metric_value_factory: Callable[..., Any],
    worst_metric_factory: Callable[[], Any],
) -> dict[str, Any]:
    """Make the frozen benchmark evaluator authoritative for AIDE search.

    AIDE's LLM review remains available verbatim in the returned sidecar, but
    its self-reported metric and bug bit cannot overrule executable benchmark
    evidence.  No Agent.search_policy or Journal implementation is replaced.
    """
    llm_review = {
        "analysis": node.analysis,
        "is_buggy": bool(node.is_buggy),
        "metric": getattr(node.metric, "value", None),
        "maximize": getattr(node.metric, "maximize", None),
    }
    metric = evaluation.get("metrics", {}).get("aggregate_improvement")
    valid = (
        evaluation.get("valid") is True
        and evaluation.get("execution_valid") is True
        and evaluation.get("score_bearing_evaluation_attempted") is True
        and isinstance(metric, (int, float))
        and not isinstance(metric, bool)
        and math.isfinite(float(metric))
    )
    node.is_buggy = not valid
    node.metric = (
        metric_value_factory(float(metric), maximize=True)
        if valid else worst_metric_factory()
    )
    evidence_sha256 = hashlib.sha256(
        json.dumps(evaluation, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    return {
        "schema_version": "aide-official-evaluation-binding-v1",
        "node_id": node.id,
        "parent_id": node.parent.id if node.parent is not None else None,
        "code_sha256": hashlib.sha256(node.code.encode()).hexdigest(),
        "llm_review": llm_review,
        "llm_analysis_preserved": node.analysis == llm_review["analysis"],
        "official_evaluation_valid": valid,
        "official_native_metric": float(metric) if valid else None,
        "official_evaluation_sha256": evidence_sha256,
        "effective_is_buggy": node.is_buggy,
        "effective_metric": getattr(node.metric, "value", None),
        "execution": {
            "exec_time": getattr(node, "exec_time", None),
            "exc_type": getattr(node, "exc_type", None),
            "term_out_sha256": hashlib.sha256(
                str(getattr(node, "term_out", "")).encode()
            ).hexdigest(),
        },
    }
