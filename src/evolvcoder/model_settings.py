"""Shared request controls for full inner sessions and outer policy proposals.

These are per-request controls, not run budgets or runtime/evaluator watchdogs.
Credentials and endpoint selection stay with the calling installed entrypoint.
"""
from __future__ import annotations

from copy import deepcopy
import math

from .configuration import ConfigurationError, ModelConfig, _positive


def request_controls(profile: dict, settings: dict, *, role: str,
                     role_output_cap: int | None = None) -> dict:
    if role not in {"coder", "planner", "rsi"}:
        raise ConfigurationError("unknown model request role")
    explicit = profile.get("model_settings", {})
    allowed = {"temperature", "timeout_seconds", "socket_timeout_seconds", "max_output_tokens"}
    if not isinstance(explicit, dict) or set(explicit) - allowed:
        raise ConfigurationError("model_settings has unknown fields or is not an object")
    temperature = explicit.get("temperature", 0)
    if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
        raise ConfigurationError("model_settings.temperature must be finite and in [0, 2]")
    # Preserve the old role-specific defaults when the new fields are absent.
    socket_timeout = explicit.get("socket_timeout_seconds", 180.0 if role == "rsi" else 300)
    total_timeout = explicit.get("timeout_seconds", 300 if role == "rsi" else settings.get("request_total_timeout_seconds", 300))
    _positive(socket_timeout, "model_settings.socket_timeout_seconds", optional=False)
    _positive(total_timeout, "model_settings.timeout_seconds", optional=False)
    shared_cap = explicit.get("max_output_tokens")
    if "max_output_tokens" in explicit:
        _positive(shared_cap, "model_settings.max_output_tokens", integer=True, optional=False)
    if role_output_cap is None:
        role_output_cap = (settings.get("coder_output_tokens_per_turn", 12000) if role == "coder"
                           else settings.get("planner_output_tokens_per_batch", 16000) if role == "planner"
                           else 8192)
    _positive(role_output_cap, "model role output cap", integer=True, optional=False)
    return {"temperature": temperature, "request_timeout_seconds": socket_timeout,
            "request_total_timeout_seconds": float(total_timeout),
            "max_tokens_per_turn": min(shared_cap, role_output_cap) if shared_cap is not None else role_output_cap}


def profile_with_model_controls(profile: dict, model: ModelConfig) -> dict:
    """Compile non-secret ModelConfig request controls without I/O or mutation.

The caller must also pass model.base_url and the selected credential reference
to its entrypoint. This function never persists secret values or opens a key.
"""
    if not isinstance(model, ModelConfig):
        raise ConfigurationError("model must be ModelConfig")
    if model.requests_per_minute is not None or model.tokens_per_minute is not None or model.concurrency != 1:
        raise ConfigurationError("model RPM/TPM and concurrency != 1 are not implemented by this execution bridge")
    compiled = deepcopy(profile)
    compiled["model"] = model.model
    compiled["model_settings"] = {"temperature": model.temperature,
                                  "max_output_tokens": model.max_output_tokens,
                                  "timeout_seconds": model.timeout_seconds,
                                  "socket_timeout_seconds": model.timeout_seconds}
    return compiled
