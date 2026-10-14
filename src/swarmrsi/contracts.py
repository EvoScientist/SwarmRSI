from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field
from enum import Enum
from typing import Any, Mapping


def canonical_sha256(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


class ProposalRejected(ValueError):
    """A semantic proposal refusal before execution, not an integrity bypass.

    Sessions may seal an existing incumbent on this narrow failure. Parent,
    artifact, access, and checkpoint identity violations must not use this type.
    """

    def __init__(self, reason: str, *, code: str, action_id: str) -> None:
        super().__init__(reason)
        self.record = {"error_type": type(self).__name__, "reason": reason,
                       "code": code, "action_id": action_id}


class ActionOperator(str, Enum):
    PROPOSE_HYPOTHESIS = "propose_hypothesis"
    REFINE_HYPOTHESIS = "refine_hypothesis"
    BRANCH_PROGRAM = "branch_program"
    IMPROVE_PROGRAM = "improve_program"
    REPAIR_PROGRAM = "repair_program"
    FALSIFY_HYPOTHESIS = "falsify_hypothesis"
    COMBINE_PROGRAMS = "combine_programs"
    REUSE_INTERVENTION = "reuse_intervention"
    REPLICATE_EVIDENCE = "replicate_evidence"
    REVIVE_STEPPING_STONE = "revive_stepping_stone"


DEPENDENCY_KINDS = frozenset({
    "code_parent", "intervention_source", "hypothesis_dependency",
    "evidence_motivation", "obligation_target", "diagnostic_source",
})


@dataclass(frozen=True, slots=True)
class ResearchDependency:
    source_id: str
    kind: str
    source_sha256: str

    def __post_init__(self) -> None:
        if self.kind not in DEPENDENCY_KINDS:
            raise ValueError(f"unsupported dependency kind: {self.kind}")
        if not self.source_id or len(self.source_sha256) != 64:
            raise ValueError("dependency must bind a source id and sha256")


@dataclass(frozen=True, slots=True)
class ScientificFamily:
    input_transform: str
    primary_operator: str
    output_construction: str
    objective_or_test: str
    capacity_controls: tuple[str, ...] = ()

    @property
    def core(self) -> tuple[str, str, str, str]:
        return (
            self.input_transform, self.primary_operator,
            self.output_construction, self.objective_or_test,
        )

    @property
    def sha256(self) -> str:
        # Capacity changes do not manufacture a scientific family.
        return canonical_sha256(self.core)


@dataclass(frozen=True, slots=True)
class ActionCandidate:
    id: str
    hypothesis_id: str
    operator: ActionOperator
    dependencies: tuple[ResearchDependency, ...]
    family: ScientificFamily
    predicted_program_quality: float
    predicted_information_gain: float
    predicted_reuse_value: float
    predicted_validity: float
    expected_cost: float
    hypothesis_statement: str = ""
    island_id: str | None = None
    action_sha256: str = ""

    def __post_init__(self) -> None:
        numeric = (
            self.predicted_program_quality, self.predicted_information_gain,
            self.predicted_reuse_value, self.predicted_validity,
            self.expected_cost,
        )
        if not self.id or not self.hypothesis_id:
            raise ValueError("action id and hypothesis id must be non-empty")
        if self.island_id is not None and not self.island_id.strip():
            raise ValueError("island_id must be non-empty when present")
        if not all(math.isfinite(value) for value in numeric):
            raise ValueError("action predictions must be finite")
        if self.expected_cost <= 0:
            raise ValueError("expected_cost must be positive")
        code_parents = [d for d in self.dependencies if d.kind == "code_parent"]
        if len(code_parents) != 1:
            raise ValueError("every executable action must have one code_parent")
        if len({(d.source_id, d.kind) for d in self.dependencies}) != len(
            self.dependencies
        ):
            raise ValueError("duplicate typed dependencies are forbidden")
        payload = {
            "id": self.id,
            "hypothesis_id": self.hypothesis_id,
            "operator": self.operator.value,
            "dependencies": [asdict(item) for item in self.dependencies],
            "family": asdict(self.family),
            "predictions": numeric,
            "hypothesis_statement": self.hypothesis_statement,
            "island_id": self.island_id,
        }
        digest = canonical_sha256(payload)
        if self.action_sha256 and self.action_sha256 != digest:
            raise ValueError("action_sha256 does not match canonical content")
        object.__setattr__(self, "action_sha256", digest)

    @property
    def intervention_parent_count(self) -> int:
        return sum(d.kind == "intervention_source" for d in self.dependencies)


@dataclass(frozen=True, slots=True)
class SearchBudget:
    remaining_model_calls: int
    remaining_score_evaluations: int
    remaining_tokens: int
    remaining_cost_usd: float | None
    cost_policy: str = "unbounded"
    # Tokens spent so far. Strategies that charge planning to an episode measure the
    # difference of this between two snapshots; measuring it from `remaining_tokens`
    # breaks the moment a campaign has no token ceiling, because the remainder is
    # then infinite and cannot be checkpointed.
    used_tokens: int = 0

    def __post_init__(self) -> None:
        if min(
            self.remaining_model_calls,
            self.remaining_score_evaluations,
            self.remaining_tokens,
        ) < 0:
            raise ValueError("remaining budgets cannot be negative")
        if self.remaining_cost_usd is not None and self.remaining_cost_usd < 0:
            raise ValueError("remaining_cost_usd cannot be negative")
        if type(self.used_tokens) is not int or self.used_tokens < 0:
            raise ValueError("used_tokens must be a nonnegative integer")
        if self.cost_policy not in {"capped", "unbounded"}:
            raise ValueError("cost_policy must be capped or unbounded")
        if self.cost_policy == "capped" and self.remaining_cost_usd is None:
            raise ValueError("capped cost policy requires a remaining amount")

    @property
    def paid_launch_authorized(self) -> bool:
        return self.cost_policy == "unbounded" or self.remaining_cost_usd is not None


@dataclass(frozen=True, slots=True)
class SearchSnapshot:
    graph_sha256: str
    actions: tuple[ActionCandidate, ...]
    realized_family_sha256s: frozenset[str]
    attempted_action_sha256s: frozenset[str]
    node_visits: Mapping[str, int]
    budget: SearchBudget
    step: int
    node_metrics: Mapping[str, float | None] = field(default_factory=dict)
    node_parent_ids: Mapping[str, str | None] = field(default_factory=dict)
    node_family_sha256s: Mapping[str, str | None] = field(default_factory=dict)
    node_island_ids: Mapping[str, str | None] = field(default_factory=dict)
    node_artifact_validity: Mapping[str, bool] = field(default_factory=dict)
    node_executable_sha256s: Mapping[str, str] = field(default_factory=dict)
    node_evaluation_summaries: Mapping[str, Mapping[str, Any] | None] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.graph_sha256) != 64:
            raise ValueError("snapshot must bind a graph sha256")
        if len({action.id for action in self.actions}) != len(self.actions):
            raise ValueError("action ids must be unique")
        if self.step < 0:
            raise ValueError("step cannot be negative")


@dataclass(frozen=True, slots=True)
class SearchDecision:
    strategy: str
    action_id: str
    graph_sha256: str
    score_vector: tuple[tuple[str, float], ...]
    rationale_codes: tuple[str, ...]
    island_id: str | None = None
    policy_sha256: str | None = None


@dataclass(frozen=True, slots=True)
class SearchObservation:
    action_id: str
    action_sha256: str
    graph_before_sha256: str
    graph_after_sha256: str
    executable: bool
    score_bearing: bool
    artifact_valid: bool
    scientific_disposition: str
    native_metric: float | None
    realized_family_sha256: str | None
    model_calls: int
    tokens: int
    cost_usd: float
    code_parent_id: str | None = None
    program_node_id: str | None = None
    island_id: str | None = None
    evaluation_evidence_sha256: str | None = None
    evaluation_summary: Mapping[str, Any] | None = None
    failure_summary: Mapping[str, Any] | None = None
    intervention_source_ids: tuple[str, ...] = ()
    recovery_route: str | None = None
    implementation_observation: Mapping[str, Any] | None = None
    program_sha256: str | None = None
    internal_validation: Mapping[str, Any] | None = None

    def __post_init__(self) -> None:
        if self.scientific_disposition not in {
            "supports", "refutes", "inconclusive"
        }:
            raise ValueError("scientific disposition must be three-valued")
        if self.native_metric is not None and not math.isfinite(self.native_metric):
            raise ValueError("native metric must be finite when present")
        if self.evaluation_evidence_sha256 is not None and len(
            self.evaluation_evidence_sha256
        ) != 64:
            raise ValueError("evaluation evidence must be sha256-bound")


@dataclass(frozen=True, slots=True)
class PolicyTrial:
    candidate_policy_sha256: str
    frozen_policy_sha256: str
    distinct_tasks: int
    distinct_seeds: int
    matched_budget: bool
    valid_cells: int
    candidate_utility: float
    frozen_utility: float
    false_promotion_delta: float
    artifact_valid_rate_delta: float


@dataclass(frozen=True, slots=True)
class PolicyGenome:
    weights: tuple[tuple[str, float], ...]
    exploration_floor: float
    diversity_reserve: int
    max_intervention_parents: int
    policy_sha256: str = ""

    def __post_init__(self) -> None:
        if not 0 <= self.exploration_floor <= 1:
            raise ValueError("exploration_floor must be in [0, 1]")
        if self.diversity_reserve < 0 or self.max_intervention_parents < 0:
            raise ValueError("policy counts cannot be negative")
        if len({key for key, _ in self.weights}) != len(self.weights):
            raise ValueError("policy weight keys must be unique")
        digest = canonical_sha256({
            "weights": self.weights,
            "exploration_floor": self.exploration_floor,
            "diversity_reserve": self.diversity_reserve,
            "max_intervention_parents": self.max_intervention_parents,
        })
        if self.policy_sha256 and self.policy_sha256 != digest:
            raise ValueError("policy_sha256 does not match canonical content")
        object.__setattr__(self, "policy_sha256", digest)

    def weight_map(self) -> dict[str, float]:
        return dict(self.weights)
