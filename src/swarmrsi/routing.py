from __future__ import annotations

from dataclasses import dataclass

from .coder_models import CoderExitStatus, CoderOutcome
from .contracts import canonical_sha256


@dataclass(frozen=True, slots=True)
class RecoveryRoute:
    """Controller-owned next-action classification, not an LLM suggestion."""

    route: str
    reason_codes: tuple[str, ...]
    source_outcome_sha256: str
    route_sha256: str = ""

    def __post_init__(self) -> None:
        allowed = {
            "repair_implementation",
            "optimize_efficiency",
            "contract_preserving_rewrite",
            "new_hypothesis_or_alternative_realization",
            "novel_family_exploration",
            "retry_evaluator_same_candidate",
            "return_to_planner",
            "exploit_verified_program",
            "discard_underperforming_candidate",
        }
        if self.route not in allowed:
            raise ValueError(f"unsupported recovery route: {self.route}")
        payload = {
            "route": self.route,
            "reason_codes": self.reason_codes,
            "source_outcome_sha256": self.source_outcome_sha256,
        }
        digest = canonical_sha256(payload)
        if self.route_sha256 and self.route_sha256 != digest:
            raise ValueError("route_sha256 mismatch")
        object.__setattr__(self, "route_sha256", digest)


def classify_recovery_route(
    outcome: CoderOutcome,
    *,
    artifact_valid: bool,
    native_metric: float | None,
    parent_metric: float | None,
    duplicate_scientific_family: bool = False,
    evaluation_invalid: bool = False,
    internal_validation_rejected: bool = False,
) -> RecoveryRoute:
    """Map observed execution evidence to one of the frozen recovery lanes.

    Ordering is deliberate: provider/evaluator invalidity must not alter
    scientific weight; plan infeasibility belongs to the planner; contract and
    resource failures outrank generic implementation repair.
    """

    if outcome.status == CoderExitStatus.EVALUATION_INVALID or evaluation_invalid:
        return RecoveryRoute(
            "retry_evaluator_same_candidate",
            ("evaluation_invalid", "scientific_weight_unchanged"),
            outcome.outcome_sha256,
        )
    if outcome.status == CoderExitStatus.PLAN_BLOCKED:
        return RecoveryRoute(
            "return_to_planner", ("plan_blocked",), outcome.outcome_sha256
        )
    if internal_validation_rejected and outcome.status == CoderExitStatus.SUCCESS:
        # The candidate ran and measured itself on public labels below its
        # parent; no official evaluation was spent and nothing is broken, so
        # neither a repair nor a scientific refutation follows.
        return RecoveryRoute(
            "discard_underperforming_candidate",
            ("internal_validation_below_parent", "no_official_evaluation_spent"),
            outcome.outcome_sha256,
        )
    if outcome.contract_violations:
        return RecoveryRoute(
            "contract_preserving_rewrite",
            ("contract_violation",),
            outcome.outcome_sha256,
        )
    failure_text = " ".join(
        str(value)
        for item in outcome.validation
        if item.get("accepted") is not True
        for value in (
            item.get("failure_category", ""),
            item.get("failure_summary", ""),
            item.get("stderr", ""),
        )
    ).lower()
    resource_failed = (
        any(item.get("timed_out") is True for item in outcome.validation)
        or any(
            token in (outcome.reason + " " + failure_text).lower()
            for token in ("wall_budget_exhausted", "timeout", "timed out", "oom", "out of memory")
        )
    )
    if resource_failed:
        return RecoveryRoute(
            "optimize_efficiency", ("resource_failure",), outcome.outcome_sha256
        )
    if outcome.status != CoderExitStatus.SUCCESS or not artifact_valid:
        return RecoveryRoute(
            "repair_implementation",
            ("implementation_or_validation_failure",),
            outcome.outcome_sha256,
        )
    if native_metric is not None and parent_metric is not None and native_metric > parent_metric:
        return RecoveryRoute(
            "exploit_verified_program",
            ("artifact_valid", "metric_improved_over_parent"),
            outcome.outcome_sha256,
        )
    if duplicate_scientific_family:
        return RecoveryRoute(
            "novel_family_exploration",
            ("repeated_scientific_family_without_parent_improvement",),
            outcome.outcome_sha256,
        )
    reason = (
        ("artifact_valid", "unscored_parent_requires_comparative_evidence")
        if parent_metric is None
        else ("artifact_valid", "no_metric_improvement_over_parent")
    )
    return RecoveryRoute(
        "new_hypothesis_or_alternative_realization",
        reason,
        outcome.outcome_sha256,
    )
