from __future__ import annotations

import json
from pathlib import Path

import pytest

from evolvcoder import (
    MetricDirection,
    MutationKernel,
    MutationResult,
    ProgramBudget,
    ProgramEvidence,
    ProgramParticle,
    SMC,
    ResearchContract,
    SMCConfig,
    write_smc_graph_view,
)


def digest(number: int) -> str:
    return f"{number:064x}"


def evidence(metric: float | None, *, valid: bool = True, fidelity: float = 1.0):
    return ProgramEvidence(
        primary_metric=metric,
        artifact_valid=valid,
        contract_fidelity=fidelity,
        required_tests_passed=valid,
        score_bearing=metric is not None,
        model_calls=1,
        score_evaluations=int(metric is not None),
        wall_seconds=1.0,
        behavior_signature=f"behavior-{metric}",
    )


def contract() -> ResearchContract:
    return ResearchContract(
        hypothesis_id="h1",
        claim="the fixed intervention improves the frozen primary metric",
        baseline_artifact_sha256=digest(1),
        required_tests=("syntax", "official_evaluator"),
        constraints=("do not change the metric",),
        primary_metric="aggregate_improvement",
        metric_direction=MetricDirection.MAXIMIZE,
        reward_lower_bound=-1.0,
        reward_upper_bound=1.0,
        budget=ProgramBudget(100, 100, 1000.0),
    )


def initial_particles(count: int = 2) -> tuple[ProgramParticle, ...]:
    return tuple(
        ProgramParticle(
            particle_id=f"p{index}", program_sha256=digest(10 + index),
            artifact_ref=f"artifact/{index}", evidence=evidence(0.1 * index),
        )
        for index in range(count)
    )


def test_smc_rejects_invalid_and_duplicate_mutations():
    search = SMC(
        contract=contract(), config=SMCConfig(particle_count=2, seed=4),
        initial_particles=initial_particles(),
    )
    first = search.next_request()
    assert first is not None and first.contract_sha256 == search.contract.contract_sha256
    assert not search.record_result(MutationResult(
        first.request_id, digest(30), "candidate/invalid", evidence(None, valid=False)
    ))
    second = search.next_request()
    assert second is not None
    assert not search.record_result(MutationResult(
        second.request_id, initial_particles()[0].program_sha256,
        "candidate/duplicate", evidence(0.9),
    ))
    observed = [e for e in search.events if e["event_type"] == "smc.mutation_observed"]
    assert observed[0]["payload"]["eligible"] is False
    assert observed[1]["payload"]["duplicate"] is True


def test_smc_improvement_is_accepted_without_mutation_kernel_reward():
    search = SMC(
        contract=contract(), config=SMCConfig(particle_count=2, seed=2),
        initial_particles=initial_particles(),
    )
    request = search.next_request()
    assert request is not None
    accepted = search.record_result(MutationResult(
        request.request_id, digest(40), "candidate/better", evidence(0.9)
    ))
    assert accepted
    assert search.projection()["kernel_mixture"] == {
        "local_diff": 0.35, "structural_rewrite": 0.25,
        "alternative_realization": 0.20, "cross_program_inspiration": 0.20,
    }


def test_smc_ess_schedule_converges_and_stops_automatically():
    config = SMCConfig(
        particle_count=2, proposals_per_particle=1, min_stages=3,
        max_stages=5, ess_fraction=0.8, seed=8,
    )
    search = SMC(
        contract=contract(), config=config, initial_particles=initial_particles(),
    )
    counter = 100
    while (request := search.next_request()) is not None:
        counter += 1
        search.record_result(MutationResult(
            request.request_id, digest(counter), f"candidate/{counter}", evidence(0.8)
        ))
    assert search.stage == 3
    assert search.annealing_lambda == pytest.approx(1.0)
    assert search.stop_reason == "annealing_target_reached"
    assert len([e for e in search.events if e["event_type"] == "smc.stage_started"]) == 3


def test_smc_checkpoint_roundtrip_preserves_outstanding_request_and_rng():
    search = SMC(
        contract=contract(), config=SMCConfig(particle_count=2, seed=12),
        initial_particles=initial_particles(),
    )
    request = search.next_request()
    payload = json.loads(json.dumps(search.checkpoint()))
    restored = SMC.restore(payload)
    assert restored.checkpoint() == payload
    with pytest.raises(RuntimeError, match="outstanding"):
        restored.next_request()
    assert restored._outstanding == request


def test_cross_program_request_exposes_typed_inspiration_edges():
    search = SMC(
        contract=contract(), config=SMCConfig(
            particle_count=2, seed=1, kernel_probabilities=(0.0, 0.0, 0.0, 1.0)
        ),
        initial_particles=initial_particles(),
    )
    request = search.next_request()
    assert request is not None and request.kernel == MutationKernel.CROSS_PROGRAM_INSPIRATION
    assert request.inspiration_particle_ids
    search.record_result(MutationResult(
        request.request_id, digest(55), "candidate/cross", evidence(0.9)
    ))
    projection = search.projection()
    assert any(edge["kind"] == "program_inspiration" for edge in projection["dependency_edges"])


def test_smc_graph_view_exposes_stage_and_typed_edges(tmp_path: Path):
    target = tmp_path / "research_graph.html"
    write_smc_graph_view(target)
    text = target.read_text()
    assert "program_inspiration" in text
    assert 'id="stage"' in text
    assert "research_graph.json" in text
