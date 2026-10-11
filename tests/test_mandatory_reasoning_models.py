"""Profile-set request capabilities (tool_choice, reasoning_can_disable), e.g. for Opus 5.5.

Measured 2026-09-28: a forced tool_choice ("required") and reasoning {"enabled": false}
each return 400 on anthropic/claude-opus-5.5, while "auto" returns the tool call.
Other models keep their exact request bodies.
"""
import json
from unittest.mock import patch

from evolvcoder.coder_openrouter import OpenRouterCoderModel

PAYLOAD = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
           "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
               "function": {"name": "inspect", "arguments": json.dumps({"path": "run.py"})}}]}}]}


class Response:
    def __enter__(self):
        return self

    def __exit__(self, *_):
        return False

    def read(self):
        return json.dumps(PAYLOAD).encode()


OPUS55 = {"tool_choice": "auto", "reasoning_can_disable": False}  # what its profile sets


def body_for(model_id, reasoning, controls=None, **request):
    model = OpenRouterCoderModel(api_key="k", model=model_id, base_url="https://openrouter.ai/api/v1",
                                 max_tokens_per_turn=100, native_tools=True, reasoning=reasoning, **(controls or {}))
    with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
        model.respond({"version": "C3", "seed": 0, **request})
    return json.loads(urlopen.call_args.args[0].data)


def test_opus_55_asks_for_the_tool_call_instead_of_forcing_it():
    assert body_for("anthropic/claude-opus-5.5", {"effort": "low"}, OPUS55)["tool_choice"] == "auto"


def test_opus_55_reasoning_off_becomes_lowest_effort():
    body = body_for("anthropic/claude-opus-5.5", {"effort": "medium"}, OPUS55, reasoning_off_this_turn=True)
    assert body["reasoning"] == {"effort": "low"}


def test_opus_55_keeps_the_configured_effort_otherwise():
    assert body_for("anthropic/claude-opus-5.5", {"effort": "medium"}, OPUS55)["reasoning"] == {"effort": "medium"}


def test_other_openrouter_models_are_unchanged():
    body = body_for("deepseek/deepseek-v4.1-flash", {"effort": "low"}, reasoning_off_this_turn=True)
    assert body["tool_choice"] == "required" and body["reasoning"] == {"enabled": False}


def test_without_profile_controls_opus_55_gets_the_old_defaults():
    body = body_for("anthropic/claude-opus-5.5", {"effort": "low"}, reasoning_off_this_turn=True)
    assert body["tool_choice"] == "required" and body["reasoning"] == {"enabled": False}
