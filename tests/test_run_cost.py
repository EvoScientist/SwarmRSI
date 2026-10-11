import json

from evolvcoder.run_cost import summarize_run_cost


def _event(payload):
    return json.dumps({"kind": "coder.model_response", "payload": payload})


def test_run_cost_sums_coder_and_planner_calls_and_flags_missing_fields(tmp_path):
    cell = tmp_path / "cells" / "000_a"; cell.mkdir(parents=True)
    cell.joinpath("trajectory.jsonl").write_text("\n".join([
        json.dumps({"kind": "coder.checkpoint_created", "payload": {}}),
        _event({"input_tokens": 100, "output_tokens": 10, "response": {
            "tool": "edit", "_cost_usd": 0.002, "_cached_input_tokens": 80, "_reasoning_tokens": 6}}),
        _event({"input_tokens": 50, "output_tokens": 5, "response": {"tool": "finish"}}),
    ]) + "\n")
    batches = tmp_path / "planner_batches"; batches.mkdir()
    batches.joinpath("batch_000000.record.json").write_text(json.dumps({
        "usage": {"input_tokens": 1000, "output_tokens": 200},
        "response": {"_cost_usd": 0.01, "_cached_input_tokens": 0, "_reasoning_tokens": 150}}))
    summary = summarize_run_cost(tmp_path)
    total = summary["total"]
    assert total["calls"] == 3 and total["input_tokens"] == 1150 and total["output_tokens"] == 215
    assert total["cost_usd"] == 0.012 and total["cached_input_tokens"] == 80 and total["reasoning_tokens"] == 156
    assert total["calls_without_cost_usd"] == 1 and summary["complete"] is False
    assert summary["coder"]["calls"] == 2 and summary["planner"]["calls"] == 1


def test_openrouter_records_cost_and_cached_tokens():
    from unittest.mock import patch
    from evolvcoder.coder_openrouter import OpenRouterCoderModel

    payload = {"usage": {"prompt_tokens": 7, "completion_tokens": 3, "cost": 0.0001,
                         "prompt_tokens_details": {"cached_tokens": 5}},
               "choices": [{"finish_reason": "tool_calls", "message": {"content": None, "tool_calls": [{
                   "function": {"name": "inspect", "arguments": json.dumps({"path": "run.py"})}}]}}]}

    class Response:
        def __enter__(self):
            return self

        def __exit__(self, *_):
            return False

        def read(self):
            return json.dumps(payload).encode()

    model = OpenRouterCoderModel(api_key="t", model="m", base_url="https://example.invalid",
                                 max_tokens_per_turn=100, native_tools=True)
    with patch("urllib.request.urlopen", return_value=Response()):
        action, usage = model.respond({"version": "C3", "seed": 0})
    assert action["_cost_usd"] == 0.0001 and action["_cached_input_tokens"] == 5
    assert usage == {"input_tokens": 7, "output_tokens": 3}
