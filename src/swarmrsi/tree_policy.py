"""Explicit, bounded learning interfaces around the existing continuation Tree.

The baseline implementation is not edited. theta0 reproduces its choices;
policy learning cannot change repair routing, evaluator authority, or hard caps.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from .contracts import SearchSnapshot, canonical_sha256
from .strategies import ContinuationTreeStrategy, _normalized_node_quality


@dataclass(frozen=True)
class TreePolicy:
    parent_quality_weight: float = 1.0
    parent_exploration_weight: float = 0.65
    parent_branch_pressure_weight: float = 0.25
    max_scored_continuations: int = 2
    stagnation_patience: int = 1

    def __post_init__(self):
        weights = (self.parent_quality_weight, self.parent_exploration_weight, self.parent_branch_pressure_weight)
        if any(type(v) not in (int, float) or not math.isfinite(v) or not 0 <= v <= 4 for v in weights):
            raise ValueError("Tree policy weights must be finite numbers in [0, 4]")
        if not any(weights):
            raise ValueError("Tree parent weights cannot all be zero")
        # Integer/float spellings of the same policy share one identity.
        for name in ("parent_quality_weight", "parent_exploration_weight", "parent_branch_pressure_weight"):
            object.__setattr__(self, name, float(getattr(self, name)))
        for name in ("max_scored_continuations", "stagnation_patience"):
            value = getattr(self, name)
            if type(value) is not int or not 1 <= value <= 16:
                raise ValueError("Tree continuation controls must be integers in [1, 16]")

    @property
    def sha256(self) -> str:
        return canonical_sha256(asdict(self))

    def to_dict(self) -> dict:
        return {"schema_version": "tree-policy-v1", "parameters": asdict(self), "policy_sha256": self.sha256}

    @classmethod
    def from_dict(cls, record: dict) -> TreePolicy:
        if set(record) != {"schema_version", "parameters", "policy_sha256"} or record["schema_version"] != "tree-policy-v1":
            raise ValueError("unknown Tree policy fields/schema")
        if set(record["parameters"]) != set(cls.__dataclass_fields__):
            raise ValueError("policy may modify only the declared Tree interfaces")
        policy = cls(**record["parameters"])
        if policy.sha256 != record["policy_sha256"]:
            raise ValueError("Tree policy hash mismatch")
        return policy

    def diff(self, incumbent: TreePolicy) -> dict:
        return {name: {"before": getattr(incumbent, name), "after": value}
                for name, value in asdict(self).items() if value != getattr(incumbent, name)}


class PolicyTreeStrategy(ContinuationTreeStrategy):
    """Frozen policy per inner run; the outer RSI loop alone replaces policies."""

    def __init__(self, *, policy: TreePolicy, seed: int = 0, base_config: dict | None = None):
        settings = dict(base_config or {})
        if settings.pop("policy", "continuation_tree") != "continuation_tree":
            raise ValueError("RSI requires the continuation Tree, not legacy v4/v5")
        settings.pop("seed", None)
        settings.update(max_scored_continuations=policy.max_scored_continuations,
                        stagnation_patience=policy.stagnation_patience)
        super().__init__(seed=seed, **settings)
        if policy.max_scored_continuations > self.config["max_local_attempts"]:
            raise ValueError("learned continuation count exceeds the unchanged local-attempt bound")
        self.learned_policy = policy
        self.config["learned_policy"] = policy.to_dict()

    def _components(self, snapshot: SearchSnapshot, node_id: str) -> tuple[tuple[str, float], ...]:
        visits = int(snapshot.node_visits.get(node_id, 0))
        total = max(1, sum(snapshot.node_visits.values()))
        children = sum(parent == node_id for parent in snapshot.node_parent_ids.values())
        return (("policy_parent_quality", self.learned_policy.parent_quality_weight * _normalized_node_quality(snapshot, node_id)),
                ("policy_parent_exploration", self.learned_policy.parent_exploration_weight * math.sqrt(math.log(total + 2) / (visits + 1))),
                ("policy_parent_branch_pressure", self.learned_policy.parent_branch_pressure_weight * (1.0 / math.sqrt(children + 1))))

    def _choice(self, snapshot: SearchSnapshot) -> tuple[str, str]:
        parent, mode = super()._choice(snapshot)
        if mode != "global" or self.learned_policy == TreePolicy():
            return parent, mode
        ranked_snapshot = self._selection_snapshot(snapshot)
        # Match the original ranker's scored-valid pool. An invalid frozen
        # root is a bootstrap fallback only when there are no valid parents.
        eligible = {node for node in self._eligible(snapshot) if snapshot.node_artifact_validity.get(node)}
        if not eligible:
            eligible = {parent}
        return max(eligible, key=lambda node: (self._score_parent(ranked_snapshot, node), node)), mode

    def _score_parent(self, snapshot: SearchSnapshot, node_id: str) -> float:
        quality, exploration, pressure = (value for _, value in self._components(snapshot, node_id))
        # Preserve the original left-associative floating-point operation;
        # Python's compensated sum() can resolve near-ties differently.
        return quality + exploration + pressure

    def select(self, snapshot: SearchSnapshot):
        decision = super().select(snapshot)
        action = next(a for a in snapshot.actions if a.id == decision.action_id)
        parent = next(d.source_id for d in action.dependencies if d.kind == "code_parent")
        return replace(decision, score_vector=decision.score_vector + self._components(self._selection_snapshot(snapshot), parent),
                       rationale_codes=decision.rationale_codes + ("frozen_tree_policy_consumed",))


def policy_from_profile(profile: dict[str, Any]) -> TreePolicy | None:
    value = profile.get("rsi_policy")
    return TreePolicy.from_dict(value) if value is not None else None
