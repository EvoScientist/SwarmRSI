from __future__ import annotations

import hashlib
import json
import math
import multiprocessing
import urllib.error
import urllib.request
from typing import Any

from .planner_protocol import constrained_proposal_items


def _fetch_json_in_child(
    url: str, data: bytes, headers: dict[str, str], socket_timeout: float, sender: Any
) -> None:
    """Perform one HTTP request in a disposable process so the parent owns the deadline."""
    try:
        request = urllib.request.Request(url, data=data, method="POST", headers=headers)
        with urllib.request.urlopen(request, timeout=socket_timeout) as response:
            sender.send(("ok", json.loads(response.read())))
    except BaseException as exc:  # child must serialize failures back to the controller
        detail = f"{type(exc).__name__}: {exc}"
        if isinstance(exc, urllib.error.HTTPError):
            # Keep the provider's error code with the status: a 403 key-limit refusal and a
            # 403 moderation refusal look identical otherwise. Numbers only, never the body.
            try:
                code = (json.loads(exc.read() or b"{}").get("error") or {}).get("code")
            except Exception:  # noqa: BLE001
                code = None
            if isinstance(code, (int, str)) and str(code).isdigit():
                detail += f" [provider_error_code={code}]"
        sender.send(("error", detail))
    finally:
        sender.close()


# Response keys filled from the provider's usage/routing data rather than the message;
# a check that re-decodes the message must leave them out.
TRANSPORT_USAGE_KEYS = frozenset({"_provider", "_reasoning_tokens", "_cost_usd", "_cached_input_tokens"})


def http_failure_codes(exc: BaseException) -> dict[str, int]:
    """HTTP status and provider error code named in a transport failure, digits only."""
    import re
    text = str(exc)
    out: dict[str, int] = {}
    status = re.search(r"HTTP Error (\d{3})", text)
    if status:
        out["http_status"] = int(status.group(1))
    code = re.search(r"provider_error_code=(\d{1,4})", text)
    if code:
        out["provider_error_code"] = int(code.group(1))
    return out


class OpenRouterCoderModel:
    """OpenAI-compatible JSON client with no implicit retries or fallback route."""

    def __init__(
        self, *, api_key: str, model: str, base_url: str,
        max_tokens_per_turn: int, request_timeout_seconds: float = 180.0,
        request_total_timeout_seconds: float | None = None,
        provider: dict[str, Any] | None = None,
        reasoning: dict[str, Any] | None = None,
        native_tools: bool = False,
        parallel_tool_calls: bool | None = None,
        temperature: float = 0,
        tool_choice: str | None = None,
        reasoning_can_disable: bool = True,
    ) -> None:
        if not api_key:
            raise ValueError("missing API key")
        self._api_key = api_key
        self.model = model
        self.url = base_url.rstrip("/") + "/chat/completions"
        # DeepSeek's own API is OpenAI-compatible but does NOT take OpenRouter's
        # normalized `reasoning` object; it takes OpenAI's `reasoning_effort`.
        self.native_reasoning_param = "api.deepseek.com" in base_url
        # Per-model request capabilities come from the profile, not from model names.
        # Measured via OpenRouter 2026-09-28: Opus 5.5 answers 400 to a forced
        # tool_choice ("required") and to reasoning {"enabled": false}; its profile
        # sets tool_choice "auto" and reasoning_can_disable false.
        if tool_choice not in (None, "auto", "required"):
            raise ValueError("tool_choice must be 'auto', 'required' or None")
        if not isinstance(reasoning_can_disable, bool):
            raise ValueError("reasoning_can_disable must be boolean")
        self.tool_choice = tool_choice
        self.reasoning_can_disable = reasoning_can_disable
        self.max_tokens_per_turn = max_tokens_per_turn
        self.request_timeout_seconds = request_timeout_seconds
        self.request_total_timeout_seconds = request_total_timeout_seconds
        self.provider = provider
        self.reasoning = reasoning
        self.native_tools = native_tools
        if parallel_tool_calls is not None and not isinstance(parallel_tool_calls, bool):
            raise ValueError("parallel_tool_calls must be boolean or None")
        self.parallel_tool_calls = parallel_tool_calls
        if type(temperature) not in (int, float) or not math.isfinite(temperature) or not 0 <= temperature <= 2:
            raise ValueError("temperature must be finite and in [0, 2]")
        self.temperature = temperature

    @staticmethod
    def _tool_definitions() -> list[dict[str, Any]]:
        def tool(name: str, description: str, properties: dict[str, Any], required=()):
            return {
                "type": "function",
                "function": {
                    "name": name,
                    "description": description,
                    "parameters": {
                        "type": "object",
                        "properties": properties,
                        "required": list(required),
                        "additionalProperties": False,
                    },
                },
            }

        string = {"type": "string"}
        integer = {"type": "integer"}
        return [
            tool("list", "List a bounded workspace or declared read-only mount directory.", {
                "path": string, "recursive": {"type": "boolean"}, "limit": integer,
            }),
            tool("inspect", "Read forward from a one-based line/Unicode-character column. "
                 "If truncated, use the returned next_read action to continue without losing text.",
                 {"path": string, "start_line": integer, "start_column": integer, "end_line": integer}, ("path",)),
            tool("search", "Search readable workspace or declared read-only mount text.",
                 {"query": string, "path": string, "regex": {"type": "boolean"}}, ("query",)),
            tool("edit", "Replace one exact unique text anchor in an existing file.",
                 {"path": string, "old": string, "new": string}, ("path", "old", "new")),
            tool("repair", "Repair one exact unique text anchor after validation feedback.",
                 {"path": string, "old": string, "new": string}, ("path", "old", "new")),
            tool("write", "Create or deliberately replace a contract-allowed text file.",
                 {"path": string, "content": string}, ("path", "content")),
            tool("append", "Append the next bounded chunk to a contract-allowed text file. Use after write when a complete module is too large for one response.",
                 {"path": string, "content": string, "expected_sha256": string},
                 ("path", "content", "expected_sha256")),
            tool("delete", "Delete one contract-allowed helper file; run.py cannot be deleted.",
                 {"path": string}, ("path",)),
            tool("run", "Run one controller-allowed command.", {
                "argv": {"type": "array", "items": string}, "cwd": string,
                "timeout_seconds": {"type": "number"},
            }, ("argv",)),
            tool("diagnose", "Read the latest controller validation feedback.", {}),
            tool("finish", "Finish editing and ask the controller to run all validation gates.", {}),
            tool("blocked", "Report that the scientific action cannot be implemented as stated.",
                 {"reason": string}, ("reason",)),
        ]

    @staticmethod
    def _planner_tool_definition(slots: list[dict] | None = None) -> dict[str, Any]:
        scalar = {"type": "number", "minimum": 0, "maximum": 1}
        proposal = {
            "type": "object",
            "properties": {
                "hypothesis": {"type": "string"},
                "description": {"type": "string"},
                "operator": {"type": "string"},
                "diagnostic_source_action_id": {"type": "string"},
                "family": {
                    "type": "object",
                    "properties": {
                        "input_transform": {"type": "string"},
                        "primary_operator": {"type": "string"},
                        "output_construction": {"type": "string"},
                        "objective_or_test": {"type": "string"},
                        "capacity_controls": {
                            "type": "array", "items": {"type": "string"},
                        },
                    },
                    "required": [
                        "input_transform", "primary_operator",
                        "output_construction", "objective_or_test",
                        "capacity_controls",
                    ],
                    "additionalProperties": False,
                },
                "predictions": {
                    "type": "object",
                    "properties": {
                        "quality": scalar,
                        "information_gain": scalar,
                        "reuse": scalar,
                        "validity": scalar,
                        "cost": scalar,
                    },
                    "required": [
                        "quality", "information_gain", "reuse", "validity", "cost",
                    ],
                    "additionalProperties": False,
                },
            },
            "required": ["hypothesis", "description", "operator", "family", "predictions"],
            "additionalProperties": False,
        }
        proposal["properties"]["cost_bound"] = {
            "type": "object",
            "properties": {
                "dominant_operation": {"type": "string"},
                "complexity_class": {"type": "string", "enum": [
                    "linear", "n_log_n", "bounded_neighbourhood", "quadratic_full", "cubic_or_worse"]},
                "estimated_seconds": {"type": "number", "minimum": 0},
            },
            "required": ["dominant_operation", "complexity_class", "estimated_seconds"],
            "additionalProperties": False,
        }
        array_schema = {"type": "array", "items": proposal}
        if slots is not None:
            array_schema.update(items=constrained_proposal_items(proposal, slots),
                                minItems=len(slots), maxItems=len(slots))
        return {
            "type": "function",
            "function": {
                "name": "submit_proposals",
                "description": (
                    "Submit the complete ordered scientific proposal batch requested by "
                    "planner_payload.proposal_slots."
                ),
                "parameters": {
                    "type": "object",
                    "properties": {
                        "proposals": array_schema,
                    },
                    "required": ["proposals"],
                    "additionalProperties": False,
                },
            },
        }

    @staticmethod
    def _json_object(text: str) -> dict[str, Any]:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        value = json.loads(text)
        if not isinstance(value, dict):
            raise ValueError("coder response is not a JSON object")
        return value

    def _decode_message(self, message: dict[str, Any], version: str,
                        finish_reason: str | None) -> dict[str, Any]:
        """Reject incompatible calls without guessing or executing partial arguments."""
        message_snapshot = json.dumps(message, ensure_ascii=False, sort_keys=True)
        content = message.get("content")
        tool_calls = message.get("tool_calls")
        category = "invalid_json_response"
        surplus_tool_calls = 0
        try:
            if self.native_tools and tool_calls:
                category = "malformed_native_tool_calls"
                if not isinstance(tool_calls, list):
                    raise ValueError("native_tool_calls_not_array")
                category = "multiple_native_tool_calls"
                if len(tool_calls) != 1:
                    # A planner batch is complete in one call: submit_proposals carries
                    # exactly as many proposals as there are slots, so several calls are
                    # the same batch resubmitted, not a batch split across calls. Some
                    # backbones do this routinely -- deepseek-v4.1-flash returns three --
                    # and the provider cannot be told not to: OpenRouter has no flash
                    # provider that accepts parallel_tool_calls, so sending it with
                    # require_parameters routes to nothing and returns 404. The first
                    # complete batch is taken and the surplus is counted in the
                    # diagnostic; the whole response stays journalled either way.
                    #
                    # A coder response is not merged this way. There the calls name
                    # different tools, so which one to run is a real ambiguity.
                    if version != "PLANNER" or not tool_calls:
                        raise ValueError("expected_exactly_one_native_tool_call")
                    surplus_tool_calls = len(tool_calls) - 1
                    tool_calls = tool_calls[:1]
                category = "malformed_native_tool_call"
                if not isinstance(tool_calls[0], dict):
                    raise ValueError("native_tool_call_not_object")
                function = tool_calls[0].get("function")
                if not isinstance(function, dict):
                    raise ValueError("native_tool_function_not_object")
                arguments = function.get("arguments", {})
                category = "invalid_native_tool_arguments_json"
                if isinstance(arguments, str):
                    arguments = json.loads(arguments)
                category = "native_tool_arguments_not_object"
                if not isinstance(arguments, dict):
                    raise ValueError("native_tool_arguments_not_object")
                tool_name = str(function.get("name", ""))
                if version == "PLANNER":
                    category = "unexpected_planner_tool"
                    if tool_name != "submit_proposals":
                        raise ValueError("unexpected_planner_tool")
                    parsed = arguments
                else:
                    parsed = {"tool": tool_name, "arguments": arguments}
            else:
                category = "empty_response"
                if not isinstance(content, str) or not content.strip():
                    raise ValueError("empty_or_non_string_message_content")
                category = "invalid_json_response"
                parsed = self._json_object(content)
        except ValueError as exc:
            preserved = (
                json.dumps({"content": content, "tool_calls": tool_calls}, ensure_ascii=False, sort_keys=True)
                if tool_calls is not None else content if isinstance(content, str) else ""
            )
            parsed = {
                "_invalid_json": True,
                "_parse_error": f"{type(exc).__name__}:{exc}",
                "_parse_error_category": category,
                "_transport_error_category": (
                    "response_truncated" if finish_reason in {"length", "max_tokens"} else category
                ),
                "_native_tool_count": len(tool_calls) if isinstance(tool_calls, list) else None,
                "_raw_sha256": hashlib.sha256(preserved.encode("utf-8")).hexdigest(),
                "_raw_content": preserved,
            }
        else:
            # Metadata is controller-owned, never a model-provided assertion.
            # Copy arguments too: native arguments may already be a dict that
            # aliases message, and must remain unchanged in the raw snapshot.
            parsed = {key: value for key, value in parsed.items() if not key.startswith("_")}
        if finish_reason is not None:
            parsed["_provider_finish_reason"] = str(finish_reason)
        if surplus_tool_calls:
            # Recorded so a run states plainly that the backbone resubmitted its batch
            # and how many copies were dropped, rather than hiding it behind a success.
            parsed["_surplus_native_tool_calls"] = surplus_tool_calls
        if version in {"PLANNER", "RSI_POLICY"}:
            # Persist valid JSON as well as malformed replies. Never include
            # request headers or credentials in provider-message evidence.
            parsed["_raw_content"] = message_snapshot
            parsed["_raw_sha256"] = hashlib.sha256(message_snapshot.encode("utf-8")).hexdigest()
        return parsed

    def respond(self, payload: dict[str, Any]) -> tuple[dict[str, Any], dict[str, int]]:
        version = str(payload["version"])
        if version == "PLANNER":
            format_instruction = (
                "You are the scientific planner, not the coding agent. Follow planner_payload. "
                + (
                    "Call submit_proposals exactly once with the complete ordered proposal batch."
                    if self.native_tools else
                    "Return exactly one JSON object with a proposals array and no tool request."
                )
            )
        elif version == "RSI_POLICY":
            format_instruction = (
                "You propose a bounded search policy, not program edits. Follow policy_learning_request. "
                "Return exactly one JSON object with parameters, rationale, and evidence_run_ids, "
                "or no_candidate=true and reason. Never request coding tools or invent trial measurements."
            )
        elif payload.get("response_mode") == "patch":
            format_instruction = (
                "Return exactly one JSON object with plan and patch (SEARCH/REPLACE blocks) or plan and "
                "candidate_content, as the payload instruction describes. Do not request tools."
            )
        elif version == "C0":
            format_instruction = (
                "Return exactly one JSON object with candidate_content containing the complete "
                "replacement executable parent. Do not request tools."
            )
        elif self.native_tools:
            format_instruction = (
                "Call exactly one supplied function. The current executable and latest tool "
                "observation plus the bounded interaction history are already in the user payload. "
                "Never repeat an identical list, inspect, "
                "or search call. Read-only mounts expose public task data for bounded inspection and "
                "local experiments. For a mutation, reason from the supplied executable and evidence, "
                "then edit or write only contract-allowed files. Keep each write/append chunk below "
                "16000 characters so tool arguments cannot be cut off. Never change the scientific action."
            )
        else:
            format_instruction = (
                "Return exactly one JSON object. Request exactly one allowed tool, or return "
                "{\"tool\":\"finish\"}, or {\"tool\":\"blocked\",\"reason\":...}. "
                "For edit/repair use one exact old/new anchor; use write to create or deliberately "
                "replace an explicitly allowed file. Never change the scientific action."
            )
        turn_output_limit = payload.get("max_output_tokens_this_turn")
        if turn_output_limit is None:
            max_tokens = self.max_tokens_per_turn
        else:
            if (
                not isinstance(turn_output_limit, int)
                or isinstance(turn_output_limit, bool)
                or turn_output_limit <= 0
            ):
                raise ValueError("max_output_tokens_this_turn must be a positive integer")
            max_tokens = min(self.max_tokens_per_turn, turn_output_limit)
        body: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "seed": int(payload.get("seed", 0)),
            "max_tokens": max_tokens,
            "messages": [
                {"role": "system", "content": ("" if version == "RSI_POLICY" else "You are a repository coding agent. ") + format_instruction},
                {"role": "user", "content": json.dumps(
                    {key: value for key, value in payload.items() if key != "reasoning_off_this_turn"},
                    ensure_ascii=False, sort_keys=True)},
            ],
        }
        if self.reasoning is not None:
            # Off for this request: configured off ({"enabled": false}), or one request
            # after responses that spent the whole output limit reasoning (the
            # controller restores the configured reasoning for the call after).
            reasoning_off = (self.reasoning.get("enabled") is False or bool(payload.get("reasoning_off_this_turn"))
                             or getattr(self, "reasoning_off_next", False))
            if self.native_reasoning_param:
                # DeepSeek's own API takes OpenAI's reasoning_effort and honours every
                # level our profiles use, so the configured effort carries over intact.
                # Measured 2026-09-23: thinking {"type": "disabled"} gives zero reasoning
                # and still returns the tool call.
                if reasoning_off:
                    body["thinking"] = {"type": "disabled"}
                else:
                    body["reasoning_effort"] = str(self.reasoning.get("effort", "medium"))
            elif reasoning_off and not self.reasoning_can_disable:
                # Cannot be switched off: the closest request is the lowest effort.
                body["reasoning"] = {"effort": "low"}
            elif reasoning_off:
                body["reasoning"] = {"enabled": False}
            else:
                body["reasoning"] = self.reasoning
        if self.native_tools and version != "RSI_POLICY" and payload.get("response_mode") != "patch":
            if version == "PLANNER":
                body["tools"] = [self._planner_tool_definition(
                    payload.get("planner_payload", {}).get("proposal_slots")
                )]
            else:
                exposed = set(payload.get("allowed_tools", ())) | {"finish", "blocked"}
                body["tools"] = [
                    item for item in self._tool_definitions()
                    if item["function"]["name"] in exposed
                ]
            # Measured against api.deepseek.com on 2026-09-19: while thinking is on,
            # both deepseek-flash and deepseek-v4-pro reject a "required" tool_choice
            # and a named function with 400 "Thinking mode does not support this
            # tool_choice", so keeping reasoning means asking rather than compelling.
            # "auto" returned exactly one submit_proposals call in every trial on both
            # lines at a realistic output budget, at every effort level; an earlier
            # reading that v4-pro returns none was an artefact of a 200-token cap that
            # its reasoning consumed before it could answer. A batch that does come back
            # without a call is already handled as an invalid batch.
            body["tool_choice"] = self.tool_choice or ("auto" if self.native_reasoning_param else "required")
            # Only send a provider capability when the profile explicitly
            # enables it. None is omission, not JSON null or implicit false.
            if self.parallel_tool_calls is not None:
                body["parallel_tool_calls"] = self.parallel_tool_calls
        else:
            body["response_format"] = {"type": "json_object"}
        if self.provider is not None:
            body["provider"] = self.provider
        data = json.dumps(body).encode("utf-8")
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        if self.request_total_timeout_seconds is None:
            request = urllib.request.Request(
                self.url, data=data, method="POST", headers=headers
            )
            with urllib.request.urlopen(
                request, timeout=self.request_timeout_seconds
            ) as response:
                raw = json.loads(response.read())
        else:
            context = multiprocessing.get_context("spawn")
            receiver, sender = context.Pipe(duplex=False)
            process = context.Process(
                target=_fetch_json_in_child,
                args=(self.url, data, headers, self.request_timeout_seconds, sender),
                daemon=True,
            )
            process.start()
            sender.close()
            try:
                if not receiver.poll(float(self.request_total_timeout_seconds)):
                    process.terminate()
                    process.join(timeout=5)
                    if process.is_alive():
                        process.kill()
                        process.join(timeout=5)
                    raise TimeoutError("OpenRouter total response deadline exceeded")
                status, result = receiver.recv()
                process.join(timeout=5)
                if status != "ok":
                    raise RuntimeError(f"OpenRouter request failed: {result}")
                raw = result
            finally:
                receiver.close()
                if process.is_alive():
                    process.terminate()
                    process.join(timeout=5)
        usage = raw.get("usage", {})
        accounting = {
            "input_tokens": int(usage.get("prompt_tokens", 0)),
            "output_tokens": int(usage.get("completion_tokens", 0)),
        }
        message = raw["choices"][0]["message"]
        finish_reason = raw["choices"][0].get("finish_reason")
        parsed = self._decode_message(message, version, finish_reason)
        # Which upstream served the request, and how much of the output was reasoning:
        # OpenRouter routes one model to several providers that behave differently.
        if isinstance(raw.get("provider"), str):
            parsed["_provider"] = raw["provider"]
        details = usage.get("completion_tokens_details")
        if isinstance(details, dict) and isinstance(details.get("reasoning_tokens"), int):
            parsed["_reasoning_tokens"] = details["reasoning_tokens"]
        # Billing evidence for cost comparisons: OpenRouter's charged cost and the
        # cached share of the prompt (DeepSeek's own API reports prompt_cache_hit_tokens).
        if isinstance(usage.get("cost"), (int, float)) and not isinstance(usage.get("cost"), bool):
            parsed["_cost_usd"] = float(usage["cost"])
        prompt_details = usage.get("prompt_tokens_details")
        if isinstance(prompt_details, dict) and isinstance(prompt_details.get("cached_tokens"), int):
            parsed["_cached_input_tokens"] = prompt_details["cached_tokens"]
        elif isinstance(usage.get("prompt_cache_hit_tokens"), int):
            parsed["_cached_input_tokens"] = usage["prompt_cache_hit_tokens"]
        if version in {"PLANNER", "RSI_POLICY"}:
            # Actual outbound JSON bytes, not headers. No authorization secret
            # is present in this body; model/schema/output-cap are now auditable.
            parsed["_transport_request_body_json"] = data.decode("utf-8")
            parsed["_transport_request_body_sha256"] = hashlib.sha256(data).hexdigest()
        if version == "RSI_POLICY":
            accounting["cost_usd"] = usage.get("cost")
        return parsed, accounting
