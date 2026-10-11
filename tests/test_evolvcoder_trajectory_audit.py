from evolvcoder.trajectory_audit import (
    HumanTrajectoryReview,
    canonical_evolvcoder_fragments,
    summarize_trajectory_audit,
)


SHA = "a" * 64


def checkpoint():
    events = []
    for i, (kind, payload) in enumerate((
        ("coder.execution_finished", {
            "action_id": "a1", "status": "success", "final_sha256": SHA,
            "trajectory_sha256": SHA, "outcome_sha256": SHA,
            "diagnostic_workspace_sha256": None,
        }),
        ("search.action_observed", {
            "action_id": "a1", "artifact_valid": True, "score_bearing": True,
            "native_metric": 0.3,
        }),
        ("search.recovery_routed", {
            "action_id": "a1", "route": "exploit_verified_program",
            "route_sha256": SHA, "reason_codes": ["metric_improved_over_parent"],
        }),
    )):
        events.append({"event_type": kind, "event_sha256": str(i) * 64, "payload": payload})
    return {
        "actions": [{"candidate": {
            "id": "a1", "hypothesis_id": "h1", "operator": "improve_program",
            "action_sha256": SHA, "family": {"primary_operator": "graph"},
            "dependencies": [{"kind": "code_parent", "source_id": "root", "source_sha256": SHA}],
        }, "contract": {"description": "test graph smoothing"}}],
        "nodes": [{"id": "root", "native_metric": 0.1}],
        "steps": [{"action_id": "a1"}],
        "events": events,
    }


def test_export_and_human_audit_metrics_are_separate():
    fragments = canonical_evolvcoder_fragments(checkpoint(), task_id="task")
    assert fragments[0]["attributable_evidence_complete"] is True
    assert fragments[0]["metric_gain_over_parent"] == 0.19999999999999998
    review = HumanTrajectoryReview(
        system="EvoCoder", task_id="task", action_id="a1", reviewer_id="r1",
        implementation_alignment="correct", evidence_attribution="correct",
        routing_correctness="incorrect", promotion_verdict="unsupported",
    )
    summary = summarize_trajectory_audit(fragments, [review])
    assert summary["attributable_evidence_coverage"] == 1.0
    assert summary["promotion_audit_coverage"] == 1.0
    assert summary["audited_false_promotion_rate"] == 1.0
    assert summary["human_attributable_cumulative_gain"] > 0.19
