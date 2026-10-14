"""Bidirectional Research Search records.

Controller-owned bookkeeping for hypotheses, program representatives, sparse
H-P pairs, research episodes and cross-frontier pressure. These records carry
no scientific verdict: a requirement is an unmet implementation need, an
episode reward is a scheduling signal, and a capability tag is a declared or
observed property of executed code, never a proof that a mechanism works.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, fields, is_dataclass
from typing import Any

from .contracts import canonical_sha256

ARMS = ("H", "P")
HYPOTHESIS_STATUSES = frozenset({"active", "deferred", "archived"})
HYPOTHESIS_ORIGINS = frozenset({"top_down", "program_observation"})
REQUIREMENT_KINDS = frozenset({"implementation", "efficiency", "realization", "capability", "rebase"})
REQUIREMENT_STATUSES = frozenset({"open", "pending", "resolved", "deferred"})
EPISODE_STATUSES = frozenset({"pending", "settled", "censored", "plan_blocked"})
EPISODE_MODES = frozenset({"new_hypothesis", "activate_hypothesis", "gap_adaptation", "instance_recombination", "tree_local", "tree_global",
                           "tree_final", "tree_recovery", "tree_repair", "repair", "incumbent_sweep",
                           "evidence_consolidation"})
PRESSURE_STATUSES = frozenset({"queued", "pending", "resolved", "deferred"})


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


@dataclass(frozen=True, slots=True)
class Requirement:
    """One unmet need of a hypothesis; ``next_step`` empty means not actionable."""
    key: str
    kind: str
    description: str
    next_step: str = ""
    status: str = "open"
    source_action_id: str | None = None
    failed_program_id: str | None = None
    created_step: int = 0
    resolved_by_action_id: str | None = None
    # A "rebase" requirement names the program base the hypothesis has not been
    # realized on yet; it resolves only by an observed attempt on that base.
    target_program_id: str | None = None

    def __post_init__(self) -> None:
        _check(bool(self.key) and self.kind in REQUIREMENT_KINDS, "requirement needs a key and known kind")
        _check(self.status in REQUIREMENT_STATUSES, "unknown requirement status")


@dataclass(frozen=True, slots=True)
class HypothesisRecord:
    id: str
    statement: str
    family: dict[str, Any]
    family_sha256: str
    origin: str = "top_down"
    status: str = "active"
    version: int = 1
    created_step: int = 0
    created_action_id: str | None = None
    created_arm: str | None = None
    motivation_refs: tuple[str, ...] = ()
    requirements: tuple[Requirement, ...] = ()
    attempt_action_ids: tuple[str, ...] = ()
    program_ids: tuple[str, ...] = ()
    effective_test_count: int = 0
    unknown_test_count: int = 0
    best_metric: float | None = None
    last_attempt_step: int | None = None
    status_reason: str = ""
    duplicate_status: str = "distinct"

    def __post_init__(self) -> None:
        _check(bool(self.id) and self.status in HYPOTHESIS_STATUSES, "hypothesis needs an id and known status")
        _check(self.origin in HYPOTHESIS_ORIGINS, "unknown hypothesis origin")
        _check(len(self.family_sha256) == 64 and self.version >= 1, "hypothesis family hash/version invalid")
        _check(len({r.key for r in self.requirements}) == len(self.requirements), "duplicate requirement keys")

    def open_requirements(self) -> tuple[Requirement, ...]:
        return tuple(r for r in self.requirements if r.status == "open" and r.next_step)


@dataclass(frozen=True, slots=True)
class ProgramRepresentative:
    node_id: str
    executable_sha256: str
    native_metric: float
    family_sha256: str | None
    hypothesis_id: str
    capabilities: tuple[str, ...]
    evidence_level: str
    reason: str
    active: bool = True


@dataclass(frozen=True, slots=True)
class PairProposal:
    key: str
    hypothesis_id: str
    primary_program_id: str
    primary_executable_sha256: str
    source_program_ids: tuple[str, ...]
    requirement_keys: tuple[str, ...]
    coverage: float
    coverage_status: str
    closes_blocker: bool
    rank_vector: tuple[float, ...]
    status: str = "proposed"
    created_step: int = 0

    def __post_init__(self) -> None:
        _check(self.coverage_status in {"covered", "partial", "unknown"}, "unknown coverage status")
        _check(self.status in {"proposed", "selected", "attempted", "superseded"}, "unknown pair status")


@dataclass(frozen=True, slots=True)
class EpisodeRecord:
    id: str
    arm: str
    mode: str
    primary_program_id: str
    started_step: int
    action_id: str | None = None
    action_sha256: str | None = None
    hypothesis_id: str | None = None
    source_program_ids: tuple[str, ...] = ()
    incumbent_pre: float | None = None
    incumbent_pre_node_id: str | None = None
    planning_tokens: int = 0
    coder_tokens: int = 0
    status: str = "pending"
    reward: float | None = None
    cost_units: float | None = None
    native_metric: float | None = None
    terminal_kind: str | None = None
    settled_step: int | None = None
    origin_episode_id: str | None = None
    pair_key: str | None = None
    allocator: dict[str, Any] | None = None
    config_sha256: str = ""

    def __post_init__(self) -> None:
        _check(self.arm in ARMS and self.mode in EPISODE_MODES, "episode arm/mode invalid")
        _check(self.status in EPISODE_STATUSES, "unknown episode status")


@dataclass(frozen=True, slots=True)
class PressureItem:
    key: str
    side: str
    ref: str
    description: str
    next_step: str
    status: str = "queued"
    created_step: int = 0
    evidence_refs: tuple[str, ...] = ()
    consumed_by_episode_id: str | None = None

    def __post_init__(self) -> None:
        _check(self.side in ARMS and self.status in PRESSURE_STATUSES and bool(self.next_step),
               "pressure item needs a side, status and an actionable next step")


def record_to_dict(value: Any) -> Any:
    if is_dataclass(value):
        return {f.name: record_to_dict(getattr(value, f.name)) for f in fields(value)}
    if isinstance(value, dict):
        return {str(k): record_to_dict(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [record_to_dict(v) for v in value]
    return value


def _coerce(cls: type, payload: dict[str, Any]) -> Any:
    kwargs = {}
    for f in fields(cls):
        if f.name not in payload:
            continue
        value = payload[f.name]
        origin = str(f.type)
        if f.name == "requirements":
            value = tuple(Requirement(**item) if not isinstance(item, Requirement) else item for item in value)
        elif origin.startswith("tuple") and isinstance(value, list):
            value = tuple(value)
        kwargs[f.name] = value
    return cls(**kwargs)


def hypothesis_from_dict(payload: dict[str, Any]) -> HypothesisRecord:
    return _coerce(HypothesisRecord, payload)


def representative_from_dict(payload: dict[str, Any]) -> ProgramRepresentative:
    return _coerce(ProgramRepresentative, payload)


def pair_from_dict(payload: dict[str, Any]) -> PairProposal:
    return _coerce(PairProposal, payload)


def episode_from_dict(payload: dict[str, Any]) -> EpisodeRecord:
    return _coerce(EpisodeRecord, payload)


def pressure_from_dict(payload: dict[str, Any]) -> PressureItem:
    return _coerce(PressureItem, payload)


def logical_pair_key(hypothesis_id: str, version: int, primary_executable_sha256: str,
                     source_program_ids: tuple[str, ...]) -> str:
    return canonical_sha256({"hypothesis": hypothesis_id, "version": version,
                             "primary": primary_executable_sha256, "sources": sorted(source_program_ids)})


__all__ = [name for name in dir() if not name.startswith("_") and name not in {"asdict", "dataclass", "fields", "is_dataclass", "Any"}]
