"""SMC particle lifecycle through the real coordinator; synthetic scores only.

Every score here is a scheduling fixture for the controller, never a benchmark
result. No model is called and no candidate is executed.
"""
import json
from pathlib import Path

import pytest

from evolvcoder import naturebench_session as session
from evolvcoder.campaign import VSearchCoordinator
from evolvcoder.research_smc import MetricDirection
from evolvcoder.search_plugins import default_search_registry
from evolvcoder.smc_search import SMCSearchStrategy
from test_evolvcoder_incremental_feedback import campaign, RecordingProposer, ScriptedExecutor

PARAMS = {"particle_count": 3, "min_stages": 2, "max_stages": 4}


def smc_campaign(tmp_path, scores, *, parameters=None, evaluations=60):
    loop = campaign(tmp_path, scores=scores)
    loop.max_score_evaluations = evaluations
    loop.strategy = session.strategy("v1", 0, {"search": {"name": "research_smc",
                                                          "parameters": parameters or PARAMS}})
    loop.version = loop.strategy.name
    return loop


def runner_step(loop):
    """The production scheduling rule: expand only with no unattempted action."""
    if not any(item.candidate.action_sha256 not in loop.accounting.attempted_actions
               for item in loop.actions.values()):
        if loop.expand() == 0:
            return None
    return loop.step()


def drive(loop, limit):
    rows = []
    for _ in range(limit):
        row = runner_step(loop)
        if row is None:
            break
        rows.append(row)
    return rows


def algorithm(loop):
    return loop.strategy._algorithm


def test_population_bootstraps_then_anneals_through_real_stages(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, .45, .6, .7, .55, .65, .5, .62, .58])
    rows = drive(loop, 12)
    smc = algorithm(loop).smc
    assert smc is not None and len(algorithm(loop).bootstrap) == 3
    # The first three actions build the population; only then can a stage run.
    assert [row.code_parent_id for row in rows[:3]] == ["root", rows[0].action_id, rows[1].action_id]
    assert smc.stage >= 2 and smc.annealing_lambda == pytest.approx(1.0)
    assert smc.stop_reason == "annealing_target_reached" and algorithm(loop).stopped == smc.stop_reason
    assert len(smc.population) == 3 and all(p.accepted for p in smc.population)
    assert {p.evidence.primary_metric for p in smc.population} <= {row.native_metric for row in rows}
    assert runner_step(loop) is None  # a stopped population proposes nothing further


def test_only_eligible_evidence_enters_the_population(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, None, None, .45, .6, .7, .55, .65, .5, .62, .58])
    drive(loop, 10)
    state = algorithm(loop)
    # An initialization attempt that never produced a valid scored program is
    # charged as initialization evidence, never as a particle, and never as a
    # scientific refutation.
    assert len(state.initialization_failures) >= 1 and len(state.bootstrap) == 3
    failure = state.initialization_failures[0]
    assert failure["primary_metric"] is None and failure["score_bearing"] is False
    assert failure["artifact_valid"] is False and failure["contract_fidelity"] == 0.0
    assert state.smc is not None and all(p.evidence.score_bearing for p in state.smc.population)


def test_bounded_repair_completes_the_same_request_and_carries_its_cost(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, .45, None, .6, .7, .55, .65, .5, .62, .58])
    rows = drive(loop, 8)
    state = algorithm(loop)
    failed = next(row for row in rows if row.native_metric is None)
    repair = next(row for row in rows if row.action_id.startswith("repair_"))
    assert repair.code_parent_id == failed.code_parent_id and not state.deferred
    binding = state.action_bindings[repair.action_id]
    assert binding["repair_of"] == failed.action_id
    assert binding["request_id"] == state.action_bindings[failed.action_id]["request_id"]
    observed = [e["payload"] for e in state.smc.events if e["event_type"] == "smc.mutation_observed"]
    recorded = next(e for e in observed if e["request_id"] == binding["request_id"])
    # One request, one recorded outcome: the repair's, carrying both costs.
    assert sum(e["request_id"] == binding["request_id"] for e in observed) == 1
    assert recorded["eligible"] is True
    particle = next(p for p in state.smc.history if p.particle_id == recorded["candidate_particle_id"])
    assert particle.evidence.model_calls == 2 and particle.evidence.score_evaluations == 1


def test_unrepaired_failure_settles_its_request_at_the_next_request(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, .45, None, .6, .7, .55, .65, .5])
    drive(loop, 4)
    state = algorithm(loop)
    assert state.deferred is not None and state.smc._outstanding is not None
    # Drop the queued repair to model a controller that scheduled none.
    loop.actions = {k: v for k, v in loop.actions.items() if not k.startswith("repair_")}
    directives = state.proposal_directives(loop._snapshot(), 1)
    assert state.deferred is None and len(directives) == 1
    observed = [e["payload"] for e in state.smc.events if e["event_type"] == "smc.mutation_observed"]
    assert observed[-1]["eligible"] is False and observed[-1]["accepted"] is False


def test_mutations_bind_typed_inspirations_the_tree_would_refuse(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, .45, .6, .7, .55, .65, .5, .62],
                        parameters={**PARAMS, "max_inspirations": 2})
    drive(loop, 9)
    sources = [tuple(binding["sources"]) for binding in algorithm(loop).action_bindings.values()]
    assert any(len(item) > 0 for item in sources), "no cross-program mutation was proposed"
    multi = [action for action in loop.actions.values() if action.candidate.intervention_parent_count]
    assert multi and all(len(set(d.source_id for d in a.candidate.dependencies)) ==
                         len(a.candidate.dependencies) for a in multi)
    from evolvcoder.strategies import ContinuationTreeStrategy
    with pytest.raises((ValueError, RuntimeError)):
        ContinuationTreeStrategy().select(loop._snapshot())


def test_restart_reproduces_the_next_request_and_population(tmp_path):
    loop = smc_campaign(tmp_path, [.4, .5, .45, .6, .7, .55, .65])
    drive(loop, 5)
    saved = json.loads(json.dumps(loop.checkpoint()))
    fresh = default_search_registry().create({"name": "research_smc", "parameters": PARAMS})
    restored = VSearchCoordinator.restore(strategy=fresh, proposer=RecordingProposer(),
                                          executor=ScriptedExecutor(tmp_path / "again", [.9]), payload=saved)
    assert restored.strategy.checkpoint_state() == loop.strategy.checkpoint_state()
    assert (restored.strategy.proposal_directives(restored._snapshot(), 1)
            == loop.strategy.proposal_directives(loop._snapshot(), 1))
    left, right = algorithm(restored).smc, algorithm(loop).smc
    assert [p.particle_id for p in left.population] == [p.particle_id for p in right.population]
    assert left.annealing_lambda == right.annealing_lambda and left.stage == right.stage


@pytest.mark.parametrize("parameters", [
    {"particle_count": 0}, {"ess_fraction": 0}, {"ess_fraction": 1.5}, {"min_stages": 3, "max_stages": 2},
    {"target_beta": 0}, {"metric_direction": "sideways"}, {"reward_lower_bound": 1.0, "reward_upper_bound": 1.0},
    {"unknown_parameter": 1},
])
def test_invalid_configuration_is_refused_before_any_call(parameters):
    with pytest.raises((ValueError, TypeError)):
        default_search_registry().create({"name": "research_smc", "parameters": parameters})


def test_smc_is_not_the_tree_under_another_name(tmp_path):
    smc = default_search_registry().create({"name": "research_smc", "parameters": PARAMS})
    tree = default_search_registry().create({"name": "continuation_tree"})
    assert smc.name == "research_smc" != tree.name
    assert type(smc._algorithm) is SMCSearchStrategy
    with pytest.raises(ValueError):
        tree.restore_state(smc.checkpoint_state())
    with pytest.raises(ValueError):
        smc.restore_state(tree.checkpoint_state())
    assert smc.config["algorithm_config"]["metric_direction"] == MetricDirection.MAXIMIZE.value
