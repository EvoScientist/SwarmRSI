"""The action-admission gate must measure this run rather than trust a tuned constant."""
from types import SimpleNamespace

from swarmrsi.naturebench_session import action_start_envelope, observed_action_cost

LIMITS = {
    "model_calls_per_action": 16, "tool_calls_per_action": 32,
    "full_evaluation_timeout_seconds": 1800,
    "minimum_model_calls_to_start_action": 4, "minimum_output_tokens_to_start_action": 8000,
    "minimum_tool_calls_to_start_action": 4, "minimum_wall_seconds_to_start_action": 1860,
    "enforce_context_read_token_limit": False,
}
ROOM = dict(remaining_calls=20, remaining_evals=5, remaining_input=1e6,
            remaining_output=20000, remaining_tools=40, remaining_wall=5000,
            screening_limits=LIMITS)


def accounting(actions, **totals):
    return SimpleNamespace(attempted_actions={f"a{i}" for i in range(actions)}, **totals)


def test_a_doomed_action_is_refused_once_this_run_has_measured_itself():
    # The MLE-Bench cell: six actions cost 147 model calls, 24.5 each, against a
    # configured floor of 4. With twenty calls left the floor admits an action that
    # cannot finish; the measured mean refuses it.
    observed = observed_action_cost(accounting(6, model_calls=147, tool_calls=136,
                                               output_tokens=34644, input_tokens=2_346_561,
                                               wall_seconds=884))
    assert round(observed["model_calls"], 1) == 24.5
    assert action_start_envelope(**ROOM)[0] is True
    ready, shortfalls = action_start_envelope(**ROOM, observed=observed)
    assert ready is False
    assert shortfalls == {"model_calls": {"required": 24.5, "available": 20.0}}


def test_the_configured_floor_still_applies_below_it():
    # A backbone that measures cheaper than the floor does not get to lower it: the
    # floor is a safety bound, and two samples are a thin basis for raising the limit.
    observed = observed_action_cost(accounting(2, model_calls=4, tool_calls=4,
                                               output_tokens=1000, input_tokens=1, wall_seconds=10))
    assert observed["model_calls"] == 2.0
    assert action_start_envelope(**ROOM, observed=observed)[0] is True
    thin = dict(ROOM, remaining_calls=3)
    assert action_start_envelope(**thin, observed=observed)[0] is False


def test_one_action_is_not_enough_to_generalise_from():
    assert observed_action_cost(accounting(1, model_calls=999, tool_calls=999,
                                           output_tokens=999, input_tokens=9, wall_seconds=9)) == {}


def test_opt_in_wall_floor_follows_measured_actions():
    # Leaf e91: seven actions in 2276 s, then a stop with 1753 s left against 1860.
    observed = observed_action_cost(accounting(7, model_calls=128, tool_calls=100,
                                               output_tokens=273720, input_tokens=3_613_913,
                                               wall_seconds=2276))
    room = {**ROOM, "remaining_calls": 200, "remaining_output": 10**6, "remaining_wall": 1752.8}
    assert action_start_envelope(**room, observed=observed)[0] is False
    opted = {**room, "screening_limits": {**LIMITS, "wall_floor_from_observed": True}}
    assert action_start_envelope(**opted, observed=observed)[0] is True
    # Never below 600 s, never above the configured floor.
    assert action_start_envelope(**{**opted, "remaining_wall": 599.0}, observed=observed)[0] is False
    slow = observed_action_cost(accounting(2, wall_seconds=6000))
    ready, shortfalls = action_start_envelope(**{**opted, "remaining_wall": 1800.0}, observed=slow)
    assert shortfalls["wall_seconds"]["required"] == 1860.0
