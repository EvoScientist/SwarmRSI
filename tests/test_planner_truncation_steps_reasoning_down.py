"""A planner batch cut off at the output cap lowers reasoning effort instead of ending the run."""
from types import SimpleNamespace

from evolvcoder.naturebench_session import REASONING_EFFORTS, step_reasoning_down


def test_effort_steps_down_one_level_at_a_time_and_stops_at_the_floor():
    model = SimpleNamespace(reasoning={"effort": "high", "exclude": True})
    assert step_reasoning_down(model) is True and model.reasoning == {"effort": "medium", "exclude": True}
    assert step_reasoning_down(model) is True and model.reasoning == {"effort": "low", "exclude": True}
    assert step_reasoning_down(model) is False and model.reasoning["effort"] == "low"
    assert REASONING_EFFORTS == ("high", "medium", "low")


def test_a_model_without_a_reasoning_setting_is_left_alone():
    model = SimpleNamespace(reasoning=None)
    assert step_reasoning_down(model) is False and model.reasoning is None
    other = SimpleNamespace(reasoning={"exclude": True})
    assert step_reasoning_down(other) is False and other.reasoning == {"exclude": True}
