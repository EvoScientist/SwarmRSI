from __future__ import annotations

import math
import random
from dataclasses import asdict, dataclass, replace
from typing import Any, Protocol

from .contracts import (
    ActionCandidate, PolicyGenome, PolicyTrial, SearchDecision,
    SearchObservation, SearchSnapshot, canonical_sha256,
)


class SearchStrategy(Protocol):
    name: str

    def select(self, snapshot: SearchSnapshot) -> SearchDecision: ...
    def observe(self, observation: SearchObservation) -> None: ...


@dataclass(frozen=True, slots=True)
class ProposalDirective:
    """Controller-owned slot: the planner fills the science, not the lineage."""

    code_parent_id: str
    island_id: str | None = None
    scientific_role: str | None = None
    intervention_source_ids: tuple[str, ...] = ()
    rationale_codes: tuple[str, ...] = ()


def _parent_id(action: ActionCandidate) -> str:
    return next(
        dependency.source_id for dependency in action.dependencies
        if dependency.kind == "code_parent"
    )


def _normalized_node_quality(snapshot: SearchSnapshot, node_id: str) -> float:
    values = sorted({
        float(value) for value in snapshot.node_metrics.values()
        if value is not None and math.isfinite(float(value))
    })
    metric = snapshot.node_metrics.get(node_id)
    if metric is None or not values:
        return 0.0
    if len(values) == 1:
        return 0.5
    return values.index(float(metric)) / (len(values) - 1)


def _uct_node_score(snapshot: SearchSnapshot, node_id: str) -> float:
    total = max(1, sum(snapshot.node_visits.values()))
    visits = int(snapshot.node_visits.get(node_id, 0))
    quality = _normalized_node_quality(snapshot, node_id)
    exploration = math.sqrt(math.log(total + 2) / (visits + 1))
    children = sum(parent == node_id for parent in snapshot.node_parent_ids.values())
    branch_pressure = 1.0 / math.sqrt(children + 1)
    return quality + 0.65 * exploration + 0.25 * branch_pressure


def _rank_parent_ids(snapshot: SearchSnapshot, eligible: set[str] | None = None) -> list[str]:
    nodes = [
        node_id for node_id, valid in snapshot.node_artifact_validity.items()
        if valid and (eligible is None or node_id in eligible)
    ]
    if not nodes and "root" in snapshot.node_metrics:
        nodes = ["root"]
    return sorted(nodes, key=lambda node_id: (_uct_node_score(snapshot, node_id), node_id), reverse=True)


def _available(snapshot: SearchSnapshot) -> list[ActionCandidate]:
    if snapshot.budget.remaining_model_calls <= 0:
        raise RuntimeError("no model-call budget remains")
    if snapshot.budget.remaining_score_evaluations <= 0:
        raise RuntimeError("no score-bearing evaluation budget remains")
    actions = [
        action for action in snapshot.actions
        if action.action_sha256 not in snapshot.attempted_action_sha256s
    ]
    if not actions:
        raise RuntimeError("no unattempted actions remain")
    return actions


def _pending_repairs(actions: list[ActionCandidate]) -> list[ActionCandidate]:
    """Implementation recovery precedes spending budget on fresh science."""
    return [action for action in actions if action.operator.value == "repair_program"]


def _pareto_front(actions: list[ActionCandidate]) -> list[ActionCandidate]:
    def vector(action: ActionCandidate) -> tuple[float, ...]:
        return (
            action.predicted_program_quality,
            action.predicted_information_gain,
            action.predicted_reuse_value,
            action.predicted_validity,
            -action.expected_cost,
        )

    front = []
    for candidate in actions:
        own = vector(candidate)
        if any(
            all(left >= right for left, right in zip(vector(other), own))
            and any(left > right for left, right in zip(vector(other), own))
            for other in actions if other is not candidate
        ):
            continue
        front.append(candidate)
    return front


class MultiParentGraphStrategy:
    """Progressive scalar graph-search baseline inspired by MCGS."""

    name = "v2_multi_parent_graph"

    def __init__(self, *, seed: int = 0, allow_multi_parent: bool = True) -> None:
        self.random = random.Random(seed)
        self.allow_multi_parent = allow_multi_parent
        self.observations: list[SearchObservation] = []

    def _score(self, action: ActionCandidate, snapshot: SearchSnapshot):
        parent_id = _parent_id(action)
        visits = int(snapshot.node_visits.get(parent_id, 0))
        total = max(1, sum(snapshot.node_visits.values()) + 1)
        exploration = math.sqrt(math.log(total + 1) / (visits + 1))
        progress = len(snapshot.attempted_action_sha256s) / max(
            1, len(snapshot.actions)
        )
        explore_weight = max(0.15, 1.0 - progress)
        parts = (
            ("parent_quality", _normalized_node_quality(snapshot, parent_id)),
            ("planner_quality", 0.15 * action.predicted_program_quality),
            ("exploration", explore_weight * exploration),
            ("reuse", 0.20 * action.predicted_reuse_value),
            ("validity", 0.25 * action.predicted_validity),
            ("cost", -0.05 * action.expected_cost),
        )
        return sum(value for _, value in parts), parts

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        candidates = _available(snapshot)
        if not self.allow_multi_parent:
            candidates = [a for a in candidates if not a.intervention_parent_count]
        if not candidates:
            raise RuntimeError("tree policy has no legal single-parent actions")
        repairs = _pending_repairs(candidates)
        if repairs:
            candidates = repairs
        ranked = [(self._score(action, snapshot), action) for action in candidates]
        ranked.sort(key=lambda row: (row[0][0], row[1].id), reverse=True)
        (_, vector), selected = ranked[0]
        return SearchDecision(
            strategy=self.name,
            action_id=selected.id,
            graph_sha256=snapshot.graph_sha256,
            score_vector=vector,
            rationale_codes=(
                "progressive_exploration_exploitation",
                "controller_routed_repair_first" if repairs else "fresh_scientific_action",
                "multi_parent_allowed" if self.allow_multi_parent else "single_parent_only",
            ),
        )

    def observe(self, observation: SearchObservation) -> None:
        self.observations.append(observation)

    def proposal_directives(
        self, snapshot: SearchSnapshot, count: int
    ) -> tuple[ProposalDirective, ...]:
        parents = _rank_parent_ids(snapshot)
        if not parents:
            return ()
        directives = []
        for index in range(count):
            parent = parents[index % len(parents)]
            sources = (
                tuple(node_id for node_id in parents if node_id != parent)[:2]
                if self.allow_multi_parent else ()
            )
            directives.append(ProposalDirective(
                code_parent_id=parent,
                intervention_source_ids=sources,
                rationale_codes=(
                    "controller_selected_graph_parent",
                    "typed_intervention_sources" if sources else "single_parent_slot",
                ),
            ))
        return tuple(directives)

    def checkpoint_state(self) -> dict[str, Any]:
        return {"observations": [asdict(item) for item in self.observations]}

    def restore_state(self, payload: dict[str, Any]) -> None:
        self.observations = [SearchObservation(**{
            **row, "intervention_source_ids": tuple(row.get("intervention_source_ids", ())),
        }) for row in payload.get("observations", [])]


class HypothesisProgramTreeStrategy(MultiParentGraphStrategy):
    name = "v1_hypothesis_program_tree"

    def __init__(self, *, seed: int = 0) -> None:
        super().__init__(seed=seed, allow_multi_parent=False)

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("config") != getattr(self, "config", None):
            raise ValueError("tree checkpoint configuration changed")
        super().restore_state(payload)

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        candidates = [
            action for action in _available(snapshot)
            if not action.intervention_parent_count and action.island_id is None
        ]
        if not candidates:
            raise RuntimeError("tree policy has no legal single-parent actions")
        repairs = _pending_repairs(candidates)
        if repairs:
            candidates = repairs
        coverage_needed = len(snapshot.realized_family_sha256s) < min(
            3, snapshot.budget.remaining_score_evaluations
        )
        if coverage_needed and not repairs:
            uncovered = [
                action for action in candidates
                if action.family.sha256 not in snapshot.realized_family_sha256s
            ]
            if uncovered:
                candidates = uncovered
        ranked = [(self._score(action, snapshot), action) for action in candidates]
        ranked.sort(key=lambda row: (row[0][0], row[1].id), reverse=True)
        (_, vector), selected = ranked[0]
        return SearchDecision(
            strategy=self.name,
            action_id=selected.id,
            graph_sha256=snapshot.graph_sha256,
            score_vector=vector,
            rationale_codes=(
                "empirical_parent_uct",
                "controller_owned_tree_parent",
                "controller_routed_repair_first" if repairs else "fresh_scientific_action",
                "uncovered_family_bootstrap" if coverage_needed else "best_branch_exploitation",
                "single_code_parent",
            ),
        )


class ContinuationTreeStrategy(HypothesisProgramTreeStrategy):
    """Bounded local-best refinement interleaved with empirical parent search.

    This is not full MCTS: global quality is the parent's measured score, not
    a backed-up return. Local continuation is program lineage, not a claim that
    all descendants share one scientific hypothesis. Configuration is explicit
    so historical v1 checkpoints retain their original selector.
    """

    def __init__(self, *, seed: int = 0, max_scored_continuations: int = 2,
                 stagnation_patience: int = 1, max_local_attempts: int = 4,
                 reserve_last_score_for_continuation: bool = True) -> None:
        super().__init__(seed=seed)
        for value in (max_scored_continuations, stagnation_patience, max_local_attempts):
            if type(value) is not int or value < 1:
                raise ValueError("continuation limits must be positive integers")
        self.config = {
            "policy": "continuation_tree", "seed": seed,
            "max_scored_continuations": max_scored_continuations,
            "stagnation_patience": stagnation_patience,
            "max_local_attempts": max_local_attempts,
            "reserve_last_score_for_continuation": reserve_last_score_for_continuation,
        }
        self.burst: dict[str, Any] | None = None
        self.action_bindings: dict[str, dict[str, Any]] = {}
        self.parent_attempts: dict[str, int] = {}

    def _eligible(self, snapshot: SearchSnapshot) -> set[str]:
        return {
            node_id for node_id, valid in snapshot.node_artifact_validity.items()
            if node_id == "root" or (
                valid and snapshot.node_metrics.get(node_id) is not None
                and math.isfinite(float(snapshot.node_metrics[node_id]))
            )
        }

    def _selection_snapshot(self, snapshot: SearchSnapshot) -> SearchSnapshot:
        # A frozen, unscored starter is a bootstrap source, not a scientific
        # incumbent. Failed nonroot artifacts never become executable parents.
        return replace(snapshot, node_visits=self.parent_attempts, node_metrics={
            key: metric if snapshot.node_artifact_validity.get(key) else None
            for key, metric in snapshot.node_metrics.items()
        })

    def _choice(self, snapshot: SearchSnapshot) -> tuple[str, str]:
        eligible = self._eligible(snapshot)
        if not eligible:
            raise RuntimeError("continuation tree has no valid executable parent")
        scored_children = eligible - {"root"}
        if (self.config["reserve_last_score_for_continuation"]
                and snapshot.budget.remaining_score_evaluations == 1 and scored_children):
            return max(scored_children, key=lambda node_id: (
                snapshot.node_metrics[node_id], node_id,
            )), "final"
        burst = self.burst
        if (burst and not burst["closed"] and burst["best_id"] in eligible
                and burst["scored"] < self.config["max_scored_continuations"]
                and burst["stagnation"] < self.config["stagnation_patience"]
                and burst["attempts"] < self.config["max_local_attempts"]):
            return burst["best_id"], "local"
        # Resource visits exclude classified evaluator faults. Invalid programs
        # may consume attempts, but never contribute a zero scientific score.
        ranking_snapshot = self._selection_snapshot(snapshot)
        # _rank_parent_ids already permits the historical frozen-root fallback.
        ranked = _rank_parent_ids(ranking_snapshot, eligible)
        return ranked[0], "global"

    def proposal_directives(self, snapshot: SearchSnapshot, count: int) -> tuple[ProposalDirective, ...]:
        if count != 1:
            raise ValueError("continuation tree requires incremental proposal_batch_size=1")
        parent, mode = self._choice(snapshot)
        return (ProposalDirective(parent, rationale_codes=(
            "controller_selected_tree_parent", f"tree_{mode}",
            "measured_program_lineage_not_hypothesis_verdict",
        )),)

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        candidates = [a for a in _available(snapshot)
                      if not a.intervention_parent_count and a.island_id is None]
        repairs = _pending_repairs(candidates)
        candidates = repairs or candidates
        if len(candidates) != 1:
            raise ValueError("continuation tree requires one fresh action or one bounded repair")
        action = candidates[0]
        parent = _parent_id(action)
        if parent not in self._eligible(snapshot):
            raise ValueError("tree action does not bind a valid executable parent")
        is_repair = action.operator.value == "repair_program"
        repair_sources = ([d.source_id for d in action.dependencies if d.kind == "diagnostic_source"]
                          if is_repair else [])
        if is_repair and not repair_sources:
            raise ValueError("tree repair requires a bound diagnostic origin")
        # A diagnostic edge is evidence, not a repair instruction. Fresh
        # science can cite failures from other parents without inheriting their
        # failed code, scheduling context, or controller-repair allowance.
        # A planner-generated diagnostic repair also preserves the original
        # scheduling context; action-id naming is deliberately irrelevant.
        if repair_sources:
            if len(repair_sources) != 1:
                raise ValueError("tree repair requires exactly one bound origin")
            origin = self.action_bindings.get(repair_sources[0])
            source = next((a for a in snapshot.actions if a.id == repair_sources[0]), None)
            if (origin is None or source is None or origin["action_sha256"] != source.action_sha256
                    or origin["code_parent_id"] != parent):
                raise ValueError("tree repair origin or parent binding mismatch")
            last = self.observations[-1] if self.observations else None
            pending_repair = (
                last is not None and last.action_id == source.id and not origin["is_repair"]
                and last.recovery_route in {
                    "repair_implementation", "optimize_efficiency", "contract_preserving_rewrite",
                }
            )
            if pending_repair:
                mode, origin_id = origin["mode"], origin["origin_action_id"]
                if (mode == "local" and self.burst is not None
                        and self.burst["attempts"] >= self.config["max_local_attempts"]):
                    # The coordinator's one-shot technical recovery remains
                    # allowed, but cannot spend another LOCAL authorization.
                    mode = "recovery"
                    self.burst["closed"] = True
            else:
                # Revisiting older failed evidence is a NEW scheduled attempt,
                # not a license to resurrect a closed historical local burst.
                expected_parent, mode = self._choice(snapshot)
                if parent != expected_parent:
                    raise ValueError("historical repair no longer matches the selected parent")
                origin_id = action.id
                if mode == "global":
                    self.burst = None
        else:
            expected_parent, mode = self._choice(snapshot)
            if parent != expected_parent:
                raise ValueError("tree action no longer matches the selected parent")
            origin_id = action.id
            if mode == "global":
                self.burst = None
        self.action_bindings[action.id] = {
            "action_sha256": action.action_sha256, "code_parent_id": parent,
            "operator": action.operator.value, "mode": mode,
            "origin_action_id": origin_id, "is_repair": is_repair,
        }
        return SearchDecision(
            strategy=self.name, action_id=action.id, graph_sha256=snapshot.graph_sha256,
            score_vector=(("parent_quality", _normalized_node_quality(self._selection_snapshot(snapshot), parent)),
                          ("parent_attempts", float(self.parent_attempts.get(parent, 0)))),
            rationale_codes=(f"tree_{mode}", "single_code_parent",
                             "bound_origin_repair" if repair_sources else "fresh_scientific_action"),
            policy_sha256=canonical_sha256(self.config),
        )

    def observe(self, observation: SearchObservation) -> None:
        binding = self.action_bindings.get(observation.action_id)
        if (binding is None or binding["action_sha256"] != observation.action_sha256
                or binding["code_parent_id"] != observation.code_parent_id):
            raise ValueError("tree observation is not bound to its selected action")
        if any(item.action_id == observation.action_id for item in self.observations):
            raise ValueError("tree observation already consumed")
        self.observations.append(observation)
        if observation.recovery_route == "retry_evaluator_same_candidate":
            return
        parent, mode = binding["code_parent_id"], binding["mode"]
        self.parent_attempts[parent] = self.parent_attempts.get(parent, 0) + 1
        scored = (observation.executable and observation.score_bearing and observation.artifact_valid
                  and observation.program_node_id is not None and observation.native_metric is not None)
        if mode == "global" and scored:
            self.burst = {
                "origin_action_id": binding["origin_action_id"],
                "best_id": observation.program_node_id, "best_metric": observation.native_metric,
                "scored": 0, "stagnation": 0, "attempts": 0, "closed": False,
            }
        elif mode == "local" and self.burst is not None:
            self.burst["attempts"] += 1
            if scored:
                self.burst["scored"] += 1
                if observation.native_metric > self.burst["best_metric"]:
                    self.burst["best_id"] = observation.program_node_id
                    self.burst["best_metric"] = observation.native_metric
                    self.burst["stagnation"] = 0
                else:
                    self.burst["stagnation"] += 1
            elif binding["is_repair"] or observation.recovery_route == "return_to_planner":
                # Technical end of a local attempt; no scientific refutation.
                self.burst["closed"] = True
        # FINAL never opens/resets a local burst, even after a valid improvement.

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            **super().checkpoint_state(), "config": dict(self.config),
            "burst": None if self.burst is None else dict(self.burst),
            "action_bindings": {key: dict(value) for key, value in self.action_bindings.items()},
            "parent_attempts": dict(self.parent_attempts),
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("config") != self.config:
            raise ValueError("continuation tree checkpoint configuration changed")
        super().restore_state(payload)
        self.burst = None if payload.get("burst") is None else dict(payload["burst"])
        self.action_bindings = {key: dict(value) for key, value in payload["action_bindings"].items()}
        self.parent_attempts = dict(payload["parent_attempts"])


def configured_tree_strategy(*, seed: int = 0, config: dict[str, Any] | None = None):
    """One normalization path for execution and artifact audit."""
    settings = dict(config or {})
    policy = settings.pop("policy", "legacy_empirical_tree")
    if policy == "continuation_tree":
        return ContinuationTreeStrategy(seed=seed, **settings)
    if policy == "legacy_empirical_tree" and not settings:
        return HypothesisProgramTreeStrategy(seed=seed)
    raise ValueError(f"unknown or incompatible tree search configuration: {policy}")


class HomogeneousIslandStrategy:
    """Matched islands sharing the graph and identical policy class."""

    name = "v3_homogeneous_islands"

    def __init__(self, *, island_count: int = 4, seed: int = 0) -> None:
        if island_count < 2:
            raise ValueError("homogeneous treatment requires two or more islands")
        self.islands = tuple(
            MultiParentGraphStrategy(seed=seed + index)
            for index in range(island_count)
        )
        self.cursor = 0

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        index = self.cursor % len(self.islands)
        self.cursor += 1
        decision = self.islands[index].select(snapshot)
        return replace(
            decision, strategy=self.name, island_id=f"homogeneous-{index}"
        )

    def observe(self, observation: SearchObservation) -> None:
        for island in self.islands:
            island.observe(observation)


HETEROGENEOUS_ISLAND_ROLES: tuple[tuple[str, str], ...] = (
    (
        "representation",
        "Search input representations, normalization, feature construction, dimensionality, and multiscale views.",
    ),
    (
        "inductive_bias",
        "Search model families and structural inductive biases: geometry, topology, density, graphs, or probabilistic operators.",
    ),
    (
        "robustness",
        "Search stability-aware, noise-aware, validation-aware, consensus, or uncertainty-robust scientific hypotheses.",
    ),
    (
        "hybridization",
        "Search coherent hybrids or ensembles that combine complementary realized mechanisms without merely averaging duplicates.",
    ),
)


class HeterogeneousMultiIslandStrategy:
    """Independent heterogeneous populations with evidence-only ring migration.

    Each island owns its parent population and selection history. A migrant is
    offered only as an intervention source; it never enters the destination
    population or transfers its score until a destination-island action is run
    and independently evaluated.
    """

    name = "v3_heterogeneous_multi_island"

    def __init__(
        self, *, seed: int = 0, population_size: int = 4,
        migration_interval: int = 1,
        roles: tuple[tuple[str, str], ...] = HETEROGENEOUS_ISLAND_ROLES,
        migration_enabled: bool = True,
        reserve_last_score_for_continuation: bool = False,
    ) -> None:
        if len(roles) < 2 or population_size < 2 or migration_interval < 1:
            raise ValueError("heterogeneous islands require >=2 roles/population and positive migration interval")
        if len({name for name, _ in roles}) != len(roles) or any(
            not name.strip() or not description.strip() for name, description in roles
        ):
            raise ValueError("heterogeneous roles require unique nonempty ids and descriptions")
        self.random = random.Random(seed)
        self.roles = tuple(roles)
        self.population_size = population_size
        self.migration_interval = migration_interval
        self.migration_enabled = migration_enabled
        self.reserve_last_score_for_continuation = reserve_last_score_for_continuation
        self.populations: dict[str, list[str]] = {name: ["root"] for name, _ in roles}
        self.node_metrics: dict[str, float | None] = {}
        self.node_families: dict[str, str | None] = {}
        self.node_islands: dict[str, str | None] = {"root": None}
        self.completed_by_island: dict[str, int] = {name: 0 for name, _ in roles}
        self.reward_sum_by_island: dict[str, float] = {name: 0.0 for name, _ in roles}
        self.last_migrated_round = 0
        self.migration_inbox: dict[str, list[dict[str, Any]]] = {
            name: [] for name, _ in roles
        }
        self.migration_history: list[dict[str, Any]] = []
        self.observations: list[SearchObservation] = []
        self.cursor = 0

    @property
    def role_map(self) -> dict[str, str]:
        return dict(self.roles)

    def _sync(self, snapshot: SearchSnapshot) -> None:
        self.node_metrics.update(snapshot.node_metrics)
        self.node_families.update(snapshot.node_family_sha256s)
        self.node_islands.update(snapshot.node_island_ids)

    def _local_parents(self, snapshot: SearchSnapshot, island_id: str) -> list[str]:
        legal = {
            node_id for node_id in self.populations[island_id]
            if snapshot.node_artifact_validity.get(node_id, node_id == "root")
        }
        ranked = _rank_parent_ids(snapshot, legal)
        return ranked or ["root"]

    def _island_ucb_score(
        self, island_id: str, *, virtual_visits: int = 0,
        total_virtual_visits: int = 0,
    ) -> float:
        visits = self.completed_by_island[island_id] + virtual_visits
        if visits <= 0:
            return math.inf
        mean_reward = self.reward_sum_by_island[island_id] / self.completed_by_island[
            island_id
        ] if self.completed_by_island[island_id] else 0.0
        total = max(1, sum(self.completed_by_island.values()) + total_virtual_visits)
        return mean_reward + math.sqrt(2.0 * math.log(total + 1) / visits)

    def _rank_islands(
        self, eligible: set[str], *, virtual_visits: dict[str, int] | None = None,
    ) -> list[str]:
        role_names = [name for name, _ in self.roles]
        virtual_visits = virtual_visits or {}
        missing = [
            name for name in role_names
            if name in eligible
            and self.completed_by_island[name] + virtual_visits.get(name, 0) == 0
        ]
        if missing:
            rotated = role_names[self.cursor:] + role_names[:self.cursor]
            return [name for name in rotated if name in missing]
        return sorted(
            eligible,
            key=lambda name: (
                self._island_ucb_score(
                    name,
                    virtual_visits=virtual_visits.get(name, 0),
                    total_virtual_visits=sum(virtual_visits.values()),
                ),
                -role_names.index(name),
            ),
            reverse=True,
        )

    def proposal_directives(
        self, snapshot: SearchSnapshot, count: int
    ) -> tuple[ProposalDirective, ...]:
        self._sync(snapshot)
        directives: list[ProposalDirective] = []
        if self.reserve_last_score_for_continuation and count != 1:
            raise ValueError("last-score continuation requires incremental proposal_batch_size=1")
        virtual_visits = {name: 0 for name, _ in self.roles}
        reserved_migration_packets: set[str] = set()
        reserved_migrant_sources: set[str] = set()
        for offset in range(count):
            # Coverage is not a reason to consume the final score slot when a
            # scored continuation is available. This is search allocation,
            # not a hypothesis-validity judgment or inherited migrant score.
            continuation = []
            if (
                self.reserve_last_score_for_continuation
                and snapshot.budget.remaining_score_evaluations - offset == 1
            ):
                continuation = [
                    (float(snapshot.node_metrics[node_id]), node_id, island)
                    for island, members in self.populations.items()
                    for node_id in members
                    if node_id != "root"
                    and snapshot.node_artifact_validity.get(node_id, False)
                    and snapshot.node_metrics.get(node_id) is not None
                    and math.isfinite(float(snapshot.node_metrics[node_id]))
                ]
            if continuation:
                _, parent, island_id = max(continuation)
            else:
                island_id = self._rank_islands(
                    set(self.populations), virtual_visits=virtual_visits
                )[0]
                parents = self._local_parents(snapshot, island_id)
                parent = parents[offset % len(parents)]
            virtual_visits[island_id] += 1
            inbox = self.migration_inbox[island_id]
            migrant_packets = tuple(
                packet for packet in reversed(inbox)
                if self.migration_enabled and not continuation
                if not packet.get("consumed", False)
                if packet["migration_packet_sha256"] not in reserved_migration_packets
                if packet["source_node_id"] not in reserved_migrant_sources
                if packet["source_node_id"] != parent
            )[:1]
            migrant_ids = tuple(
                packet["source_node_id"] for packet in migrant_packets
            )
            reserved_migration_packets.update(
                packet["migration_packet_sha256"] for packet in migrant_packets
            )
            reserved_migrant_sources.update(migrant_ids)
            directives.append(ProposalDirective(
                code_parent_id=parent,
                island_id=island_id,
                scientific_role=self.role_map[island_id],
                intervention_source_ids=migrant_ids,
                rationale_codes=(
                    "independent_island_population",
                    "heterogeneous_scientific_role",
                    "last_score_verified_continuation" if continuation
                    else ("bootstrap_island_coverage"
                          if self.completed_by_island[island_id] == 0
                          else "hierarchical_ucb_island_allocation"),
                    "evidence_only_migration" if migrant_ids else "island_local_search",
                ),
            ))
        return tuple(directives)

    def _action_score(self, action: ActionCandidate, snapshot: SearchSnapshot) -> tuple[float, tuple[tuple[str, float], ...]]:
        parent = _parent_id(action)
        novel_local = float(
            action.family.sha256 not in {
                self.node_families.get(node_id)
                for node_id in self.populations.get(action.island_id or "", [])
            }
        )
        parts = (
            ("local_parent_quality", _normalized_node_quality(snapshot, parent)),
            ("local_parent_uct", 0.65 * math.sqrt(
                math.log(max(2, sum(snapshot.node_visits.values()) + 2))
                / (snapshot.node_visits.get(parent, 0) + 1)
            )),
            ("island_family_novelty", 0.45 * novel_local),
            ("planner_information", 0.20 * action.predicted_information_gain),
            ("predicted_validity", 0.20 * action.predicted_validity),
            ("cost", -0.05 * action.expected_cost),
        )
        return sum(value for _, value in parts), parts

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        self._sync(snapshot)
        available = [
            action for action in _available(snapshot)
            if action.island_id in self.populations
            and _parent_id(action) in self.populations[action.island_id]
        ]
        if not available:
            raise RuntimeError("heterogeneous islands have no legal local-parent actions")
        repairs = _pending_repairs(available)
        if repairs:
            ranked = [(self._action_score(action, snapshot), action) for action in repairs]
            ranked.sort(key=lambda row: (row[0][0], row[1].id), reverse=True)
            (_, vector), selected = ranked[0]
            return SearchDecision(
                strategy=self.name,
                action_id=selected.id,
                graph_sha256=snapshot.graph_sha256,
                score_vector=vector,
                rationale_codes=(
                    "controller_routed_repair_first",
                    "independent_island_population",
                    "same_hypothesis_repair",
                ),
                island_id=selected.island_id,
            )
        role_names = [name for name, _ in self.roles]
        eligible_islands = {str(action.island_id) for action in available}
        selected_island = self._rank_islands(eligible_islands)[0]
        self.cursor = (role_names.index(selected_island) + 1) % len(role_names)
        local = [action for action in available if action.island_id == selected_island]
        ranked = [(self._action_score(action, snapshot), action) for action in local]
        ranked.sort(key=lambda row: (row[0][0], row[1].id), reverse=True)
        (_, vector), selected = ranked[0]
        return SearchDecision(
            strategy=self.name,
            action_id=selected.id,
            graph_sha256=snapshot.graph_sha256,
            score_vector=vector,
            rationale_codes=(
                "independent_island_population",
                "bootstrap_island_coverage"
                if self.completed_by_island[selected_island] == 0
                else "hierarchical_ucb_island_allocation",
                "local_empirical_uct",
                "migrant_requires_destination_reevaluation",
            ),
            island_id=selected_island,
        )

    def _truncate_population(self, island_id: str) -> None:
        members = list(dict.fromkeys(self.populations[island_id]))
        if len(members) <= self.population_size:
            self.populations[island_id] = members
            return
        ranked = sorted(
            members,
            key=lambda node_id: (
                self.node_metrics.get(node_id) is not None,
                self.node_metrics.get(node_id) if self.node_metrics.get(node_id) is not None else -math.inf,
                node_id,
            ),
            reverse=True,
        )
        retained: list[str] = []
        seen_families: set[str | None] = set()
        for node_id in ranked:
            family = self.node_families.get(node_id)
            if family not in seen_families:
                retained.append(node_id)
                seen_families.add(family)
            if len(retained) == self.population_size:
                break
        for node_id in ranked:
            if len(retained) == self.population_size:
                break
            if node_id not in retained:
                retained.append(node_id)
        self.populations[island_id] = retained

    def _maybe_migrate(self) -> None:
        if not self.migration_enabled:
            return
        completed_round = min(self.completed_by_island.values())
        if completed_round < self.last_migrated_round + self.migration_interval:
            return
        role_names = [name for name, _ in self.roles]
        for index, source_island in enumerate(role_names):
            destination = role_names[(index + 1) % len(role_names)]
            candidates = [
                node_id for node_id in self.populations[source_island]
                if node_id != "root" and self.node_metrics.get(node_id) is not None
            ]
            if not candidates:
                continue
            elite = max(candidates, key=lambda node_id: (self.node_metrics[node_id], node_id))
            packet = {
                "round": completed_round,
                "source_island": source_island,
                "destination_island": destination,
                "source_node_id": elite,
                "source_family_sha256": self.node_families.get(elite),
                "score_inherited": False,
                "requires_destination_reevaluation": True,
                "consumed": False,
                "consumed_by_action_id": None,
            }
            packet["migration_packet_sha256"] = canonical_sha256(packet)
            self.migration_inbox[destination].append(packet)
            self.migration_history.append(packet)
        self.last_migrated_round = completed_round

    def observe(self, observation: SearchObservation) -> None:
        self.observations.append(observation)
        island_id = observation.island_id
        if island_id not in self.populations:
            raise ValueError("heterogeneous observation lacks a known island")
        parent_metric = self.node_metrics.get(observation.code_parent_id or "")
        reward = 0.0
        if (
            observation.artifact_valid
            and observation.score_bearing
            and observation.native_metric is not None
            and parent_metric is not None
        ):
            if observation.native_metric > parent_metric:
                reward = 1.0
            elif observation.native_metric == parent_metric:
                reward = 0.5
        self.reward_sum_by_island[island_id] += reward
        self.completed_by_island[island_id] += 1
        # A migration is evidence-only until the receiving island has produced
        # an independently score-bearing realization.  Implementation or
        # evaluator failures must leave the packet available to the bounded
        # same-hypothesis repair lane instead of falsely recording a completed
        # cross-island transfer.
        migration_reevaluated = (
            observation.artifact_valid
            and observation.score_bearing
            and observation.native_metric is not None
            and observation.evaluation_evidence_sha256 is not None
        )
        if migration_reevaluated:
            for source_id in observation.intervention_source_ids:
                for packet in reversed(self.migration_inbox[island_id]):
                    if packet["source_node_id"] == source_id and not packet.get("consumed", False):
                        packet["consumed"] = True
                        packet["consumed_by_action_id"] = observation.action_id
                        break
        if observation.program_node_id is not None and observation.artifact_valid:
            node_id = observation.program_node_id
            self.node_metrics[node_id] = observation.native_metric
            self.node_families[node_id] = observation.realized_family_sha256
            self.node_islands[node_id] = island_id
            self.populations[island_id].append(node_id)
            self._truncate_population(island_id)
        self._maybe_migrate()

    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "schema_version": "heterogeneous-island-state-v1",
            "roles": [list(item) for item in self.roles],
            "population_size": self.population_size,
            "migration_interval": self.migration_interval,
            "migration_enabled": self.migration_enabled,
            "reserve_last_score_for_continuation": self.reserve_last_score_for_continuation,
            "populations": self.populations,
            "node_metrics": self.node_metrics,
            "node_families": self.node_families,
            "node_islands": self.node_islands,
            "completed_by_island": self.completed_by_island,
            "reward_sum_by_island": self.reward_sum_by_island,
            "last_migrated_round": self.last_migrated_round,
            "migration_inbox": self.migration_inbox,
            "migration_history": self.migration_history,
            "observations": [asdict(item) for item in self.observations],
            "cursor": self.cursor,
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("schema_version") != "heterogeneous-island-state-v1":
            raise ValueError("unsupported heterogeneous island state")
        if tuple(tuple(item) for item in payload["roles"]) != self.roles:
            raise ValueError("heterogeneous island roles changed across resume")
        if (
            payload["population_size"] != self.population_size
            or payload["migration_interval"] != self.migration_interval
            or payload.get("migration_enabled", True) != self.migration_enabled
            or payload.get("reserve_last_score_for_continuation", False)
            != self.reserve_last_score_for_continuation
        ):
            raise ValueError("heterogeneous island configuration changed across resume")
        for packet in payload.get("migration_history", []):
            issuance = {
                key: value for key, value in packet.items()
                if key != "migration_packet_sha256"
            }
            issuance["consumed"] = False
            issuance["consumed_by_action_id"] = None
            if canonical_sha256(issuance) != packet.get("migration_packet_sha256"):
                raise ValueError("heterogeneous migration packet hash mismatch")
        self.populations = {key: list(value) for key, value in payload["populations"].items()}
        self.node_metrics = dict(payload["node_metrics"])
        self.node_families = dict(payload["node_families"])
        self.node_islands = dict(payload["node_islands"])
        self.completed_by_island = {
            key: int(value) for key, value in payload["completed_by_island"].items()
        }
        self.reward_sum_by_island = {
            key: float(value) for key, value in payload.get(
                "reward_sum_by_island", {name: 0.0 for name, _ in self.roles}
            ).items()
        }
        self.last_migrated_round = int(payload["last_migrated_round"])
        self.migration_history = list(payload["migration_history"])
        packets_by_sha = {
            packet["migration_packet_sha256"]: packet
            for packet in self.migration_history
        }
        self.migration_inbox = {
            key: [packets_by_sha[item["migration_packet_sha256"]] for item in value]
            for key, value in payload["migration_inbox"].items()
        }
        self.observations = [SearchObservation(**row) for row in payload.get("observations", [])]
        self.cursor = int(payload["cursor"])


DEFAULT_POLICY = PolicyGenome(
    weights=(
        ("quality", 1.0), ("information_gain", 0.9),
        ("diversity", 0.8), ("reuse", 0.5),
        ("validity", 0.7), ("cost", 0.15),
    ),
    exploration_floor=0.20,
    diversity_reserve=3,
    max_intervention_parents=2,
)


class EvidenceCoupledStrategy:
    """Joint hypothesis-program action search over a typed Research DAG."""

    name = "v4_evidence_coupled"

    def __init__(self, *, policy: PolicyGenome = DEFAULT_POLICY) -> None:
        self.policy = policy
        self.observations: list[SearchObservation] = []

    def _vector(self, action: ActionCandidate, snapshot: SearchSnapshot):
        weights = self.policy.weight_map()
        uncovered = float(action.family.sha256 not in snapshot.realized_family_sha256s)
        return (
            ("quality", weights["quality"] * action.predicted_program_quality),
            ("information_gain", weights["information_gain"] * action.predicted_information_gain),
            ("diversity", weights["diversity"] * uncovered),
            ("reuse", weights["reuse"] * action.predicted_reuse_value),
            ("validity", weights["validity"] * action.predicted_validity),
            ("cost", -weights["cost"] * action.expected_cost),
        )

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        candidates = [
            action for action in _available(snapshot)
            if action.intervention_parent_count <= self.policy.max_intervention_parents
        ]
        coverage_needed = (
            len(snapshot.realized_family_sha256s) < self.policy.diversity_reserve
        )
        uncovered = [
            action for action in candidates
            if action.family.sha256 not in snapshot.realized_family_sha256s
        ]
        if coverage_needed and uncovered:
            candidates = uncovered
        ranked = []
        for action in _pareto_front(candidates):
            vector = self._vector(action, snapshot)
            # Scalar sum only breaks ties within the non-dominated set.
            ranked.append((sum(value for _, value in vector), action, vector))
        ranked.sort(key=lambda row: (row[0], row[1].id), reverse=True)
        _, selected, vector = ranked[0]
        return SearchDecision(
            strategy=self.name,
            action_id=selected.id,
            graph_sha256=snapshot.graph_sha256,
            score_vector=vector,
            rationale_codes=(
                "joint_hypothesis_program_action", "pareto_front",
                "uncovered_family_reserved" if coverage_needed else "coverage_satisfied",
                "typed_multi_parent_dependencies",
            ),
            policy_sha256=self.policy.policy_sha256,
        )

    def observe(self, observation: SearchObservation) -> None:
        self.observations.append(observation)

    def proposal_directives(
        self, snapshot: SearchSnapshot, count: int
    ) -> tuple[ProposalDirective, ...]:
        parents = _rank_parent_ids(snapshot)
        if not parents:
            return ()
        directives = []
        for index in range(count):
            parent = parents[index % len(parents)]
            sources = tuple(
                node_id for node_id in parents if node_id != parent
            )[:self.policy.max_intervention_parents]
            directives.append(ProposalDirective(
                code_parent_id=parent,
                intervention_source_ids=sources,
                rationale_codes=(
                    "controller_selected_evidence_parent",
                    "typed_intervention_sources",
                ),
            ))
        return tuple(directives)


class BoundedRSIStrategy(EvidenceCoupledStrategy):
    """Open policy archive with fail-closed matched promotion."""

    name = "v5_bounded_rsi"

    def __init__(self, *, frozen_policy: PolicyGenome = DEFAULT_POLICY) -> None:
        super().__init__(policy=frozen_policy)
        self.frozen_policy = frozen_policy
        self.archive = {frozen_policy.policy_sha256: frozen_policy}
        self.trials: list[PolicyTrial] = []

    def register_candidate(self, candidate: PolicyGenome) -> None:
        self.archive[candidate.policy_sha256] = candidate

    def consider_promotion(self, trial: PolicyTrial) -> bool:
        self.trials.append(trial)
        eligible = (
            trial.candidate_policy_sha256 in self.archive
            and trial.frozen_policy_sha256 == self.frozen_policy.policy_sha256
            and trial.distinct_tasks >= 2 and trial.distinct_seeds >= 3
            and trial.matched_budget
            and trial.valid_cells >= trial.distinct_tasks * trial.distinct_seeds
            and trial.candidate_utility > trial.frozen_utility
            and trial.false_promotion_delta <= 0
            and trial.artifact_valid_rate_delta >= 0
        )
        if eligible:
            self.policy = self.archive[trial.candidate_policy_sha256]
        return eligible

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        decision = super().select(snapshot)
        return replace(
            decision,
            strategy=self.name,
            rationale_codes=decision.rationale_codes + (
                "frozen_validator", "policy_archive",
            ),
        )
