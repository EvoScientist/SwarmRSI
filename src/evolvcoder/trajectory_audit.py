from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

from .contracts import canonical_sha256


AUDIT_LABELS = frozenset({"correct", "uncertain", "incorrect"})
PROMOTION_LABELS = frozenset({"supported", "uncertain", "unsupported"})


@dataclass(frozen=True, slots=True)
class HumanTrajectoryReview:
    system: str
    task_id: str
    action_id: str
    reviewer_id: str
    implementation_alignment: str
    evidence_attribution: str
    routing_correctness: str
    promotion_verdict: str
    notes: str = ""

    def __post_init__(self) -> None:
        if not all((self.system, self.task_id, self.action_id, self.reviewer_id)):
            raise ValueError("human review identity fields are required")
        for value in (
            self.implementation_alignment,
            self.evidence_attribution,
            self.routing_correctness,
        ):
            if value not in AUDIT_LABELS:
                raise ValueError(f"unsupported audit label: {value}")
        if self.promotion_verdict not in PROMOTION_LABELS:
            raise ValueError("unsupported promotion verdict")


def canonical_evolvcoder_fragments(
    checkpoint: Mapping[str, Any], *, task_id: str, system: str = "EvoCoder"
) -> list[dict[str, Any]]:
    """Convert one V-search checkpoint into reviewer-ready decision fragments."""

    actions = {
        row["candidate"]["id"]: row for row in checkpoint.get("actions", [])
    }
    nodes = {row["id"]: row for row in checkpoint.get("nodes", [])}
    events = checkpoint.get("events", [])
    by_action: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        action_id = event.get("payload", {}).get("action_id")
        if action_id:
            by_action.setdefault(str(action_id), []).append(event)

    fragments: list[dict[str, Any]] = []
    for step in checkpoint.get("steps", []):
        action_id = str(step["action_id"])
        action_row = actions.get(action_id, {})
        candidate = action_row.get("candidate", {})
        contract = action_row.get("contract", {})
        action_events = by_action.get(action_id, [])
        event_by_type = {event["event_type"]: event for event in action_events}
        observed = event_by_type.get("search.action_observed", {}).get("payload", {})
        finished = event_by_type.get("coder.execution_finished", {}).get("payload", {})
        routed = event_by_type.get("search.recovery_routed", {}).get("payload", {})
        dependencies = candidate.get("dependencies", [])
        code_parent = next(
            (item for item in dependencies if item.get("kind") == "code_parent"), {}
        )
        parent = nodes.get(str(code_parent.get("source_id", "")), {})
        native_metric = observed.get("native_metric")
        parent_metric = parent.get("native_metric")
        gain = None
        if isinstance(native_metric, (int, float)) and isinstance(parent_metric, (int, float)):
            gain = float(native_metric) - float(parent_metric)
        evidence_fields = {
            "hypothesis": bool(candidate.get("hypothesis_id") and contract.get("description")),
            "program_parent": bool(
                code_parent.get("source_id") and code_parent.get("source_sha256")
            ),
            "scientific_family": bool(candidate.get("family") and candidate.get("action_sha256")),
            "implementation": bool(
                finished.get("outcome_sha256") and finished.get("trajectory_sha256")
            ),
            "evaluation": bool(
                "artifact_valid" in observed and "score_bearing" in observed
            ),
            "decision": bool(routed.get("route") and routed.get("route_sha256")),
        }
        fragment = {
            "schema_version": "evolvcoder-human-trajectory-fragment-v1",
            "system": system,
            "task_id": task_id,
            "action_id": action_id,
            "hypothesis_id": candidate.get("hypothesis_id"),
            "hypothesis_claim": contract.get("description"),
            "operator": candidate.get("operator"),
            "scientific_family": candidate.get("family"),
            "dependencies": dependencies,
            "code_parent_id": code_parent.get("source_id"),
            "code_parent_sha256": code_parent.get("source_sha256"),
            "parent_native_metric": parent_metric,
            "native_metric": native_metric,
            "metric_gain_over_parent": gain,
            "artifact_valid": observed.get("artifact_valid"),
            "score_bearing": observed.get("score_bearing"),
            "coder_status": finished.get("status"),
            "final_program_sha256": finished.get("final_sha256"),
            "failed_program_workspace_sha256": finished.get(
                "diagnostic_workspace_sha256"
            ),
            "coder_trajectory_sha256": finished.get("trajectory_sha256"),
            "coder_outcome_sha256": finished.get("outcome_sha256"),
            "recovery_route": routed.get("route"),
            "route_reason_codes": routed.get("reason_codes", []),
            "provisionally_promoted": routed.get("route") == "exploit_verified_program",
            "evidence_fields": evidence_fields,
            "attributable_evidence_complete": all(evidence_fields.values()),
            "event_sha256s": [event.get("event_sha256") for event in action_events],
        }
        fragment["fragment_sha256"] = canonical_sha256(fragment)
        fragments.append(fragment)
    return fragments


def summarize_trajectory_audit(
    fragments: Iterable[Mapping[str, Any]],
    reviews: Iterable[HumanTrajectoryReview] = (),
) -> dict[str, Any]:
    rows = list(fragments)
    review_rows = list(reviews)
    promoted = [row for row in rows if row.get("provisionally_promoted") is True]
    reviewed_keys = {(row.system, row.task_id, row.action_id) for row in review_rows}
    promoted_keys = {
        (str(row.get("system")), str(row.get("task_id")), str(row.get("action_id")))
        for row in promoted
    }
    audited_promoted = promoted_keys & reviewed_keys
    false_promotions = {
        (row.system, row.task_id, row.action_id)
        for row in review_rows
        if row.promotion_verdict == "unsupported"
        and (row.system, row.task_id, row.action_id) in promoted_keys
    }
    attributable_gain = 0.0
    fragment_by_key = {
        (str(row.get("system")), str(row.get("task_id")), str(row.get("action_id"))): row
        for row in rows
    }
    for review in review_rows:
        key = (review.system, review.task_id, review.action_id)
        fragment = fragment_by_key.get(key)
        if (
            fragment is not None
            and review.implementation_alignment == "correct"
            and review.evidence_attribution == "correct"
        ):
            attributable_gain += max(0.0, float(fragment.get("metric_gain_over_parent") or 0.0))
    denominator = len(rows)
    audit_denominator = len(audited_promoted)
    return {
        "schema_version": "evolvcoder-trajectory-audit-summary-v1",
        "decision_fragment_count": denominator,
        "attributable_evidence_coverage": (
            sum(row.get("attributable_evidence_complete") is True for row in rows)
            / denominator if denominator else None
        ),
        "provisional_promotion_count": len(promoted_keys),
        "promotion_audit_coverage": (
            len(audited_promoted) / len(promoted_keys) if promoted_keys else None
        ),
        "audited_false_promotion_rate": (
            len(false_promotions) / audit_denominator if audit_denominator else None
        ),
        "audited_false_promotion_count": len(false_promotions),
        "audited_promotion_count": audit_denominator,
        "human_attributable_cumulative_gain": attributable_gain,
    }
