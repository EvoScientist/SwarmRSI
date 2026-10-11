from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Protocol

from .contracts import ActionCandidate, SearchObservation


class EvaluationRecord(Protocol):
    """The benchmark evaluator owns these fields, not the search model."""

    passes_declared_execution: bool
    passes_hard_gates: bool
    native_metric: float | None


class EvidenceDisposition(Protocol):
    relation: str


@dataclass(frozen=True, slots=True)
class EvaluatorObservationAdapter:
    """Project an external evaluator's decision without inventing a verdict.

    The embedding application supplies its fixed evidence classifier explicitly.
    There is deliberately no default classifier, legacy-package import, or
    score-to-hypothesis conversion. ``authority_id`` identifies the configured
    evaluator contract; it is not a claim that arbitrary user code is trusted.
    """

    authority_id: str
    classify_evidence: Callable[[EvaluationRecord], EvidenceDisposition]

    def __post_init__(self) -> None:
        if not isinstance(self.authority_id, str) or not self.authority_id.strip():
            raise ValueError("an explicit evaluator authority_id is required")
        if not callable(self.classify_evidence):
            raise TypeError("classify_evidence must be the configured evaluator classifier")

    def observation(
        self,
        *,
        action: ActionCandidate,
        evaluation: EvaluationRecord,
        graph_before_sha256: str,
        graph_after_sha256: str,
        realized_family_sha256: str | None,
        model_calls: int,
        tokens: int,
        cost_usd: float,
    ) -> SearchObservation:
        disposition = self.classify_evidence(evaluation)
        if disposition.relation not in {"supports", "refutes", "inconclusive"}:
            raise ValueError("unknown scientific disposition from evaluator authority")
        return SearchObservation(
            action_id=action.id,
            action_sha256=action.action_sha256,
            graph_before_sha256=graph_before_sha256,
            graph_after_sha256=graph_after_sha256,
            executable=evaluation.passes_declared_execution,
            score_bearing=evaluation.native_metric is not None,
            artifact_valid=evaluation.passes_hard_gates,
            scientific_disposition=disposition.relation,
            native_metric=evaluation.native_metric,
            realized_family_sha256=realized_family_sha256,
            model_calls=model_calls,
            tokens=tokens,
            cost_usd=cost_usd,
        )


# Import-name compatibility only. Construction now requires explicit authority;
# silently selecting the old checkout's classifier would defeat independence.
FrozenEvoCoderValidator = EvaluatorObservationAdapter
