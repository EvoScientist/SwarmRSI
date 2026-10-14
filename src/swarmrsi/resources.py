from __future__ import annotations

from dataclasses import asdict, dataclass


class ResourceLimitReached(RuntimeError):
    """Raised before an operation would exceed the matched Claude envelope."""


@dataclass(frozen=True, slots=True)
class ResourceEnvelope:
    agent_turns: int
    backbone_messages: int
    tool_calls: int
    score_evaluations: int
    input_tokens: int
    cache_read_input_tokens: int
    output_tokens: int
    wall_seconds: float

    def __post_init__(self) -> None:
        values = asdict(self)
        if any(isinstance(value, bool) or value <= 0 for value in values.values()):
            raise ValueError("every resource ceiling must be positive")


@dataclass(frozen=True, slots=True)
class ResourceUsage:
    agent_turns: int = 0
    backbone_messages: int = 0
    tool_calls: int = 0
    score_evaluations: int = 0
    input_tokens: int = 0
    cache_read_input_tokens: int = 0
    output_tokens: int = 0
    wall_seconds: float = 0.0
    cost_usd: float = 0.0

    def __post_init__(self) -> None:
        if any(
            isinstance(value, bool) or value < 0
            for value in asdict(self).values()
        ):
            raise ValueError("resource usage cannot be negative")

    def plus(self, other: ResourceUsage) -> ResourceUsage:
        return ResourceUsage(**{
            key: getattr(self, key) + getattr(other, key)
            for key in asdict(self)
        })


class ResourceLedger:
    """One simple rule: refuse an operation that would cross any Claude cap."""

    _LIMIT_FIELDS = (
        "agent_turns", "backbone_messages", "tool_calls",
        "score_evaluations", "input_tokens", "cache_read_input_tokens",
        "output_tokens", "wall_seconds",
    )

    def __init__(self, envelope: ResourceEnvelope) -> None:
        self.envelope = envelope
        self.usage = ResourceUsage()

    def can_consume(self, delta: ResourceUsage) -> bool:
        proposed = self.usage.plus(delta)
        return all(
            getattr(proposed, field) <= getattr(self.envelope, field)
            for field in self._LIMIT_FIELDS
        )

    def consume(self, delta: ResourceUsage) -> ResourceUsage:
        proposed = self.usage.plus(delta)
        exceeded = [
            field for field in self._LIMIT_FIELDS
            if getattr(proposed, field) > getattr(self.envelope, field)
        ]
        if exceeded:
            raise ResourceLimitReached(
                "Claude-matched stop condition reached: " + ", ".join(exceeded)
            )
        self.usage = proposed
        return self.usage

    def remaining(self) -> dict[str, float | int]:
        return {
            field: getattr(self.envelope, field) - getattr(self.usage, field)
            for field in self._LIMIT_FIELDS
        }


CLAUDE_CODE_DSV4PRO_S43588 = ResourceEnvelope(
    agent_turns=117,
    backbone_messages=112,
    tool_calls=116,
    score_evaluations=23,
    input_tokens=46_720,
    cache_read_input_tokens=10_917_504,
    output_tokens=105_376,
    wall_seconds=10_190.187601566315,
)
