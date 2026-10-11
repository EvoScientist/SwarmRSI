"""DeepSeek's own API needs OpenAI's reasoning_effort and an asking tool_choice.

Measured against api.deepseek.com: with thinking on, deepseek-flash and deepseek-v4-pro
reject a "required" tool_choice with 400 "Thinking mode does not support this
tool_choice", but "auto" returned exactly one submit_proposals call in every trial on
both lines, at every effort level, at a realistic output budget. So reasoning is kept
and the tool call is asked for rather than compelled -- parity with the OpenRouter route
rather than a downgrade.
"""
from evolvcoder.coder_openrouter import OpenRouterCoderModel


def build(base_url, effort="high"):
    return OpenRouterCoderModel(api_key="k", model="deepseek-flash", base_url=base_url,
                                max_tokens_per_turn=12000, native_tools=True,
                                reasoning={"effort": effort, "exclude": True})


def test_openrouter_route_is_unchanged():
    model = build("https://openrouter.ai/api/v1")
    assert model.native_reasoning_param is False


def test_deepseek_direct_is_detected_with_or_without_the_version_suffix():
    assert build("https://api.deepseek.com").native_reasoning_param is True
    assert build("https://api.deepseek.com/v1").native_reasoning_param is True


def test_the_configured_effort_is_carried_over_not_discarded():
    for effort in ("high", "medium", "low"):
        assert build("https://api.deepseek.com", effort).reasoning["effort"] == effort


def test_reasoning_off_uses_thinking_disabled_on_deepseek_and_enabled_false_on_openrouter():
    import json
    from unittest.mock import patch

    payload = {"usage": {"prompt_tokens": 1, "completion_tokens": 1},
               "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                   "function": {"name": "inspect", "arguments": json.dumps({"path": "run.py"})}}]}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(payload).encode()

    def body_for(base_url, reasoning, **request):
        model = OpenRouterCoderModel(api_key="k", model="deepseek-flash", base_url=base_url,
                                     max_tokens_per_turn=100, native_tools=True, reasoning=reasoning)
        with patch("urllib.request.urlopen", return_value=Response()) as urlopen:
            model.respond({"version": "C3", "seed": 0, **request})
        return json.loads(urlopen.call_args.args[0].data)

    direct = body_for("https://api.deepseek.com", {"enabled": False})
    assert direct["thinking"] == {"type": "disabled"} and "reasoning_effort" not in direct
    direct_turn = body_for("https://api.deepseek.com", {"effort": "low"}, reasoning_off_this_turn=True)
    assert direct_turn["thinking"] == {"type": "disabled"}
    assert body_for("https://api.deepseek.com", {"effort": "low"})["reasoning_effort"] == "low"
    assert body_for("https://openrouter.ai/api/v1", {"enabled": False})["reasoning"] == {"enabled": False}
