from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import TYPE_CHECKING, Any, Iterable

from .contracts import ActionCandidate, SearchDecision, SearchObservation, canonical_sha256

if TYPE_CHECKING:
    from .coder_models import CoderOutcome


@dataclass(frozen=True, slots=True)
class SearchEvent:
    sequence: int
    event_type: str
    payload: dict[str, Any]
    previous_event_sha256: str | None
    event_sha256: str = ""

    def __post_init__(self) -> None:
        if self.sequence < 0:
            raise ValueError("event sequence cannot be negative")
        digest = canonical_sha256({
            "sequence": self.sequence,
            "event_type": self.event_type,
            "payload": self.payload,
            "previous_event_sha256": self.previous_event_sha256,
        })
        if self.event_sha256 and self.event_sha256 != digest:
            raise ValueError("event hash mismatch")
        object.__setattr__(self, "event_sha256", digest)


class DynamicResearchView:
    """Append-only display projection for search; never an evidence writer."""

    def __init__(self) -> None:
        self._events: list[SearchEvent] = []
        self._actions: dict[str, ActionCandidate] = {}

    @property
    def events(self) -> tuple[SearchEvent, ...]:
        return tuple(self._events)

    @property
    def sha256(self) -> str:
        return canonical_sha256([event.event_sha256 for event in self._events])

    def _append(self, event_type: str, payload: dict[str, Any]) -> SearchEvent:
        event = SearchEvent(
            sequence=len(self._events),
            event_type=event_type,
            payload=dict(payload),
            previous_event_sha256=(
                self._events[-1].event_sha256 if self._events else None
            ),
        )
        self._events.append(event)
        return event

    def register_action(self, action: ActionCandidate) -> SearchEvent:
        if action.id in self._actions:
            raise ValueError(f"action already registered: {action.id}")
        self._actions[action.id] = action
        return self._append("search.action_registered", {
            "action_id": action.id,
            "action_sha256": action.action_sha256,
            "hypothesis_id": action.hypothesis_id,
            "operator": action.operator.value,
            "hypothesis_statement": action.hypothesis_statement,
            "scientific_family": asdict(action.family),
            "family_sha256": action.family.sha256,
            "island_id": action.island_id,
            "dependencies": [asdict(item) for item in action.dependencies],
        })

    def record_strategy_event(self, event: dict[str, Any]) -> SearchEvent:
        """Append a search-strategy frontier event; never a scientific verdict."""
        kind, payload = event.get("event_type"), event.get("payload")
        if not isinstance(kind, str) or not kind.startswith("search.frontier.") or not isinstance(payload, dict):
            raise ValueError("strategy events must be search.frontier.* with an object payload")
        return self._append(kind, payload)

    def record_proposal_rejection(self, record: dict[str, Any]) -> SearchEvent:
        """Journal a semantic refusal; the refused proposal is never executed."""
        return self._append("search.proposal_rejected", dict(record))

    def record_planner_batch(self, diagnostic: dict[str, Any]) -> SearchEvent:
        required = {
            "requested_proposals", "returned_proposals", "accepted_proposals",
            "response_sha256", "provider_finish_reason", "invalid_response",
        }
        if not required.issubset(diagnostic):
            raise ValueError("planner batch diagnostic is incomplete")
        return self._append("planner.proposal_batch_finished", dict(diagnostic))

    def record_proposal_selection(
        self, *, graph_before_sha256: str, strategy: str,
        directives: list[dict[str, Any]], parent_evidence: dict[str, Any],
        policy_sha256: str | None = None,
    ) -> SearchEvent:
        if graph_before_sha256 != self.sha256:
            raise ValueError("proposal selection must bind the current graph")
        for directive in directives:
            parent = parent_evidence.get(directive["code_parent_id"])
            frozen_bootstrap = (
                directive["code_parent_id"] == "root" and parent is not None
                and parent.get("bootstrap_seed") is True and parent.get("native_metric") is None
            )
            if parent is None or not (parent["artifact_valid"] or frozen_bootstrap):
                raise ValueError("proposal selection lacks a valid parent binding")
        return self._append("search.proposal_parents_selected", {
            "graph_before_sha256": graph_before_sha256,
            "strategy": strategy,
            "policy_sha256": policy_sha256,
            "directives": directives,
            "parent_evidence": parent_evidence,
        })

    def record_migration(self, packet: dict[str, Any]) -> SearchEvent:
        required = {
            "source_island", "destination_island", "source_node_id",
            "score_inherited", "requires_destination_reevaluation",
        }
        if not required.issubset(packet):
            raise ValueError("migration packet is incomplete")
        if packet["score_inherited"] is not False or packet[
            "requires_destination_reevaluation"
        ] is not True:
            raise ValueError("migration must be evidence-only and re-evaluated")
        return self._append("search.island_migration_created", dict(packet))

    def record_migration_consumed(
        self, *, migration_packet_sha256: str, action_id: str,
    ) -> SearchEvent:
        if len(migration_packet_sha256) != 64 or action_id not in self._actions:
            raise ValueError("migration consumption must bind a packet and registered action")
        return self._append("search.island_migration_consumed", {
            "migration_packet_sha256": migration_packet_sha256,
            "action_id": action_id,
            "action_sha256": self._actions[action_id].action_sha256,
        })

    def record_decision(self, decision: SearchDecision) -> SearchEvent:
        if decision.action_id not in self._actions:
            raise ValueError("decision references an unknown action")
        return self._append("search.action_selected", asdict(decision))

    def record_coder_started(
        self,
        *,
        action_id: str,
        executable_parent_sha256: str,
        coder_config_sha256: str,
    ) -> SearchEvent:
        action = self._actions.get(action_id)
        if action is None:
            raise ValueError("coder start references an unknown action")
        code_parents = [
            item for item in action.dependencies if item.kind == "code_parent"
        ]
        if len(code_parents) != 1:
            raise ValueError("coder start requires exactly one code parent")
        if code_parents[0].source_sha256 != executable_parent_sha256:
            raise ValueError("coder parent hash differs from selected action")
        return self._append("coder.execution_started", {
            "action_id": action_id,
            "action_sha256": action.action_sha256,
            "executable_parent_sha256": executable_parent_sha256,
            "coder_config_sha256": coder_config_sha256,
        })

    def record_coder_outcome(
        self, *, action_id: str, outcome: "CoderOutcome"
    ) -> SearchEvent:
        action = self._actions.get(action_id)
        if action is None:
            raise ValueError("coder outcome references an unknown action")
        if outcome.action_id != action_id:
            raise ValueError("coder outcome action id mismatch")
        return self._append("coder.execution_finished", {
            "action_id": action_id,
            "action_sha256": action.action_sha256,
            "status": outcome.status.value,
            "implementation_real": outcome.implementation_real,
            "parent_preserved": outcome.parent_preserved,
            "repaired_after_failure": outcome.repaired_after_failure,
            "parent_sha256": outcome.parent_sha256,
            "final_sha256": outcome.final_sha256,
            "changed_paths": list(outcome.changed_paths),
            "contract_violations": list(outcome.contract_violations),
            "model_calls": outcome.model_calls,
            "tool_calls": outcome.tool_calls,
            "input_tokens": outcome.input_tokens,
            "output_tokens": outcome.output_tokens,
            "trajectory_sha256": outcome.trajectory_sha256,
            "outcome_sha256": outcome.outcome_sha256,
            "diagnostic_workspace_sha256": outcome.diagnostic_workspace_sha256,
            "diagnostic_artifact_relpath": outcome.diagnostic_artifact_relpath,
        })

    def record_scientific_evaluation(
        self, *, action_id: str, evidence_sha256: str,
        evaluation_summary: dict[str, Any], score_evaluations: int,
    ) -> SearchEvent:
        if action_id not in self._actions or len(evidence_sha256) != 64:
            raise ValueError("scientific evaluation must bind a registered action and evidence")
        if score_evaluations not in {0, 1}:
            raise ValueError("one published candidate may invoke the scientific evaluator at most once")
        return self._append("evaluator.scientific_evaluation_finished", {
            "action_id": action_id,
            "action_sha256": self._actions[action_id].action_sha256,
            "evidence_sha256": evidence_sha256,
            "score_evaluations": score_evaluations,
            "evaluation_summary": evaluation_summary,
        })

    def record_observation(self, observation: SearchObservation) -> SearchEvent:
        action = self._actions.get(observation.action_id)
        if action is None or action.action_sha256 != observation.action_sha256:
            raise ValueError("observation does not bind the registered action")
        return self._append("search.action_observed", asdict(observation))

    def record_implementation_observation(self, *, action_id: str, parent_program_sha256: str,
                                         candidate_program_sha256: str, evaluation_evidence_sha256: str | None,
                                         observation: dict[str, Any]) -> SearchEvent:
        action = self._actions.get(action_id)
        if action is None or not any(d.kind == "code_parent" and d.source_sha256 == parent_program_sha256
                                     for d in action.dependencies):
            raise ValueError("implementation observation does not bind its registered parent")
        if (not isinstance(candidate_program_sha256, str) or len(candidate_program_sha256) != 64
                or observation.get("hypothesis_disposition") != "inconclusive"):
            raise ValueError("program observation cannot establish a scientific verdict")
        return self._append("program.implementation_observed", {
            "action_id": action_id, "action_sha256": action.action_sha256,
            "parent_program_sha256": parent_program_sha256, "candidate_program_sha256": candidate_program_sha256,
            "evaluation_evidence_sha256": evaluation_evidence_sha256, "observation": observation,
            "authority": "controller_observed_source_and_output_only"})

    def record_recovery_route(
        self, *, action_id: str, route: str, reason_codes: tuple[str, ...],
        route_sha256: str, source_outcome_sha256: str,
    ) -> SearchEvent:
        if action_id not in self._actions:
            raise ValueError("recovery route references an unknown action")
        return self._append("search.recovery_routed", {
            "action_id": action_id,
            "route": route,
            "reason_codes": list(reason_codes),
            "route_sha256": route_sha256,
            "source_outcome_sha256": source_outcome_sha256,
        })

    def projection(self) -> dict[str, Any]:
        """Return one stable payload usable by CLI, WebSocket, or a web UI."""
        latest = {}
        for event in self._events:
            action_id = event.payload.get("action_id")
            if action_id:
                latest[str(action_id)] = event.event_type
        dependency_edges = [
            {
                "source_id": dependency.source_id,
                "target_id": action.id,
                "kind": dependency.kind,
                "source_sha256": dependency.source_sha256,
            }
            for action in sorted(self._actions.values(), key=lambda item: item.id)
            for dependency in action.dependencies
        ]
        consumed_migrations = {
            event.payload["migration_packet_sha256"]: event.payload["action_id"]
            for event in self._events
            if event.event_type == "search.island_migration_consumed"
        }
        migration_edges = [
            {
                "source_id": event.payload["source_node_id"],
                "target_island": event.payload["destination_island"],
                "source_island": event.payload["source_island"],
                "migration_packet_sha256": event.payload.get("migration_packet_sha256"),
                "consumed": event.payload.get("migration_packet_sha256")
                in consumed_migrations,
                "target_action_id": consumed_migrations.get(
                    event.payload.get("migration_packet_sha256")
                ),
                "score_inherited": event.payload["score_inherited"],
                "requires_destination_reevaluation": event.payload[
                    "requires_destination_reevaluation"
                ],
            }
            for event in self._events
            if event.event_type == "search.island_migration_created"
        ]
        islands: dict[str, list[str]] = {}
        for action in self._actions.values():
            if action.island_id is not None:
                islands.setdefault(action.island_id, []).append(action.id)
        return {
            "schema_version": "swarmrsi-dynamic-research-view-v1",
            "stream_sha256": self.sha256,
            "event_count": len(self._events),
            "actions": [
                {
                    "id": action.id,
                    "hypothesis_id": action.hypothesis_id,
                    "hypothesis_statement": action.hypothesis_statement,
                    "operator": action.operator.value,
                    "scientific_family": asdict(action.family),
                    "family_sha256": action.family.sha256,
                    "island_id": action.island_id,
                    "dependency_count": len(action.dependencies),
                    "intervention_parent_count": action.intervention_parent_count,
                    "latest_event": latest.get(action.id),
                }
                for action in sorted(self._actions.values(), key=lambda item: item.id)
            ],
            "dependency_edges": dependency_edges,
            "migration_edges": migration_edges,
            "islands": {
                island: sorted(action_ids) for island, action_ids in sorted(islands.items())
            },
            "events": [asdict(event) for event in self._events],
        }


def dependency_adjacency(actions: Iterable[ActionCandidate]) -> dict[str, set[str]]:
    action_ids = {action.id for action in actions}
    return {
        action.id: {
            dep.source_id for dep in action.dependencies if dep.source_id in action_ids
        }
        for action in actions
    }


def assert_acyclic_actions(actions: Iterable[ActionCandidate]) -> None:
    adjacency = dependency_adjacency(actions)
    visiting: set[str] = set()
    visited: set[str] = set()

    def walk(node: str) -> None:
        if node in visiting:
            raise ValueError("research-action dependencies contain a cycle")
        if node in visited:
            return
        visiting.add(node)
        for parent in adjacency.get(node, set()):
            walk(parent)
        visiting.remove(node)
        visited.add(node)

    for node in sorted(adjacency):
        walk(node)
