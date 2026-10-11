"""A backbone that resubmits its proposal batch must not stall the campaign.

deepseek-v4.1-flash returns submit_proposals several times in one response. Each call
carries the complete batch -- the schema fixes the proposal count to the slot count --
so the copies are redundant, not a batch split across calls. The provider cannot be
asked to stop: OpenRouter has no flash provider accepting parallel_tool_calls, and
sending it alongside require_parameters routes to no provider and returns 404.
"""
from evolvcoder.coder_openrouter import OpenRouterCoderModel

BATCH = {"type": "function",
         "function": {"name": "submit_proposals", "arguments": '{"proposals": [{"family": "f"}]}'}}


def decode(tool_calls, version="PLANNER"):
    model = OpenRouterCoderModel.__new__(OpenRouterCoderModel)
    model.native_tools = True
    return OpenRouterCoderModel._decode_message(
        model, {"content": None, "tool_calls": tool_calls}, version, "tool_calls")


def test_single_batch_is_unchanged_and_unannotated():
    decoded = decode([BATCH])
    assert decoded["proposals"] == [{"family": "f"}]
    assert "_surplus_native_tool_calls" not in decoded


def test_duplicate_batches_take_the_first_and_record_the_surplus():
    decoded = decode([BATCH, BATCH, BATCH])
    assert decoded["proposals"] == [{"family": "f"}]
    assert decoded["_surplus_native_tool_calls"] == 2
    assert "_invalid_json" not in decoded


def test_a_coder_reply_with_several_calls_is_still_rejected():
    # Coder calls name different tools, so which one to run is a real ambiguity.
    decoded = decode([BATCH, BATCH], version="CODER")
    assert decoded["_invalid_json"] is True
    assert "expected_exactly_one_native_tool_call" in decoded["_parse_error"]
