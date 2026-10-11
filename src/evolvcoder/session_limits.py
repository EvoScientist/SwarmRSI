"""Executable bridge between portable Limits and the complete search session.

None is persisted as null. Infinity is a transient comparison value only; it
never becomes a configured ceiling or a serialized model/graph/checkpoint field.
Finite legacy profiles retain their existing accounting and action allocations.
This layer does not provide transport-level monetary reservation or repair the
legacy distinction between accounted execution wall time and total elapsed time.
"""
from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from typing import Any

from .configuration import ConfigurationError, Limits

# A single action's ceiling when the campaign does not cap that count at all.
# High enough never to be the binding constraint, finite because the coder
# allocation must be.


def remaining(cap: int | float | None, used: int | float) -> int | float:
    if type(used) not in (int, float) or not math.isfinite(used) or used < 0:
        raise ValueError('budget usage must be known, finite and nonnegative')
    return math.inf if cap is None else cap - used


def exceeded(cap: int | float | None, used: int | float) -> bool:
    return remaining(cap, used) < 0


def positive_cap(value: Any, name: str, *, integer: bool = True, nullable: bool = True) -> None:
    if value is None and nullable:
        return
    if type(value) not in ((int,) if integer else (int, float)) or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(name + ' must be positive and finite' + (' or null' if nullable else ''))


@dataclass(frozen=True)
class SessionLimits:
    limits: Limits
    # Only imported, already-existing finite legacy profiles use their old
    # disabled-context sentinel. New null configurations never manufacture it.
    legacy_disabled_input: bool = False

    def __post_init__(self) -> None:
        if not isinstance(self.limits, Limits):
            raise TypeError('SessionLimits requires portable Limits')
        if self.limits.cost_usd is not None:
            raise ConfigurationError('session does not enforce cost_usd; transport-level reservation is required')

    @classmethod
    def from_legacy(cls, limits: dict, *, enforce_context: bool = True) -> SessionLimits:
        if type(enforce_context) is not bool:
            raise ConfigurationError('enforce_context_read_token_limit must be boolean')
        if not isinstance(limits, dict) or not isinstance(limits.get('tokens'), dict):
            raise ConfigurationError('session limits and tokens must be objects')
        required = {'backbone_message_ids', 'tool_calls', 'score_bearing_evaluations', 'wall_time_seconds'}
        if not required.issubset(limits) or not {'input', 'output'}.issubset(limits['tokens']):
            raise ConfigurationError('session limits require explicit ceilings; use null for unlimited')
        for name in ('cost_usd', 'dollar_cap'):
            if limits.get(name) is not None:
                raise ConfigurationError('session does not enforce ' + name)
        tokens = limits['tokens']
        for name in ('input', 'output', 'context_read'):
            if name in tokens:
                positive_cap(tokens[name], 'tokens.' + name)
        input_cap = tokens.get('context_read', tokens['input'])
        # Validate all configured values even when the legacy context switch
        # deliberately disables its historical input ceiling.
        checked = Limits(model_calls=limits['backbone_message_ids'], input_tokens=input_cap,
                         output_tokens=tokens['output'], tool_calls=limits['tool_calls'],
                         evaluations=limits['score_bearing_evaluations'], wall_seconds=limits['wall_time_seconds'])
        disabled = not enforce_context and input_cap is not None
        if not enforce_context:
            checked = Limits(**{**checked.__dict__, 'input_tokens': None})
        return cls(checked, legacy_disabled_input=disabled)

    @property
    def combined_tokens(self) -> int | None:
        if self.legacy_disabled_input and self.limits.output_tokens is not None:
            return 10**18  # Imported compatibility value, not a new unlimited cap.
        if self.limits.input_tokens is None or self.limits.output_tokens is None:
            return None
        return self.limits.input_tokens + self.limits.output_tokens

    def to_legacy(self) -> dict:
        if self.legacy_disabled_input:
            raise ConfigurationError('do not export a disabled-input legacy view as generic limits')
        c = self.limits
        return {'backbone_message_ids': c.model_calls, 'tool_calls': c.tool_calls,
                'score_bearing_evaluations': c.evaluations, 'wall_time_seconds': c.wall_seconds,
                'tokens': {'input': c.input_tokens, 'context_read': c.input_tokens, 'output': c.output_tokens}}

    def residual(self, accounting: Any) -> dict:
        c = self.limits
        return {'remaining_calls': remaining(c.model_calls, accounting.model_calls),
                'remaining_evals': remaining(c.evaluations, accounting.score_evaluations),
                'remaining_input': remaining(c.input_tokens, accounting.input_tokens),
                'remaining_output': remaining(c.output_tokens, accounting.output_tokens),
                'remaining_tools': remaining(c.tool_calls, accounting.tool_calls),
                'remaining_wall': remaining(c.wall_seconds, accounting.wall_seconds)}

    def validate_action_limits(self, settings: dict) -> None:
        for name, default in (('model_calls_per_action', 8), ('tool_calls_per_action', 24)):
            positive_cap(settings.get(name, default), name, nullable=False)
        controls = (
            ('max_input_tokens_per_action_safety', self.limits.input_tokens is None and not self.legacy_disabled_input, True),
            ('max_output_tokens_per_action', self.limits.output_tokens is None, True),
            ('wall_time_seconds_per_action', self.limits.wall_seconds is None, False),
        )
        for name, required, integer in controls:
            if required and name not in settings:
                raise ConfigurationError('unlimited campaign requires explicit finite ' + name)
            if name in settings:
                positive_cap(settings[name], name, integer=integer, nullable=False)

    def coder_allocation(self, accounting: Any, settings: dict) -> dict:
        """What one action may spend.

        This used to leave a count unbounded here whenever the campaign left it
        unbounded, on the reasoning that a per-action share of an unbounded budget
        decides nothing. That was wrong, and it was expensive. The per-action count
        is not a share of anything: it is the thing that makes an action stop
        exploring and hand back work.

        Measured on s43588-024-00698-1, same task, same per-action settings, only
        the campaign ceiling differing. With the ceiling, three actions ended on
        call_or_tool_budget_exhausted and four actions succeeded, for -0.0036.
        Without it, model_calls_per_action of 16 was never read, the allocation
        became 10000, no action ever ended on its call budget, and instead three
        died on wall and three on output tokens, for -0.178 with a single
        successful action that changed no file. The coder wrote fifteen probe
        scripts across two actions while run.py stayed byte-identical.

        It compounds with what the coder is told: budget_remaining carries the
        counts and not the clock, so an action handed 10000 calls cannot see the
        deadline that will actually kill it, and exploring further is the rational
        move right up until it is killed.
        """
        self.validate_action_limits(settings)
        residual = self.residual(accounting)
        result = {}
        for name, control, default in (('calls', 'model_calls_per_action', 8),
                                       ('tools', 'tool_calls_per_action', 24)):
            # The configured ceiling applies whether or not the campaign caps this
            # count. It is validated as a positive finite number either way, and it
            # is what makes an action stop and hand back work.
            result[name] = max(1, min(settings.get(control, default), residual['remaining_' + name]))
        result['input'] = max(1, residual['remaining_input']) if self.limits.input_tokens is not None else settings.get('max_input_tokens_per_action_safety', 2_000_000)
        output = residual['remaining_output']
        if 'max_output_tokens_per_action' in settings:
            output = min(output, settings['max_output_tokens_per_action'])
        result['output'] = max(1, output)
        # The configured ceiling stands, and `observed` is not used to raise it.
        # Letting a run's measured action cost set its own allocation was built on
        # the assumption that an action cut short would have succeeded with more
        # room. This benchmark's cells say otherwise: of 138 completed, those
        # spending 12 or fewer calls per action land 90% of them against 67% for 13
        # to 18, and calls per action correlates -0.23 with the result. A longer
        # action is a worse action here, so a rule that grows the allocation as
        # actions get more expensive would compound the wrong thing.
        result['wall'] = max(1., min(
            settings.get('wall_time_seconds_per_action', math.inf), residual['remaining_wall']))
        if any(not math.isfinite(value) for value in result.values()):
            raise ConfigurationError('Coder action allocations must all be finite')
        return result


def profile_with_limits(profile: dict, task_id: str, limits: Limits) -> dict:
    """Compile the portable object for the existing full runner, without I/O.

    Explicit generic input tokens always apply, even if the imported benchmark
    profile used to disable context accounting. No model, metric, task, strategy
    or hidden target changes here. The caller persists the returned profile so
    normal session source identities bind exactly the limits actually used.
    """
    result = deepcopy(profile)
    task = result['tasks'][task_id]
    bridge = SessionLimits(limits)
    settings = {**result.get('screening_execution_limits', {}), **task.get('screening_execution_limits', {}),
                'enforce_context_read_token_limit': True}
    bridge.validate_action_limits(settings)
    task['limits'] = bridge.to_legacy()
    task['screening_execution_limits'] = settings
    return result
