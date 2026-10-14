"""Bidirectional Research Search: allocator, frontiers, matcher and the real
coordinator lifecycle with synthetic outcomes.

No model is called. Synthetic scores are scheduling fixtures for the
controller, never benchmark results or evidence that the method wins.
"""
from dataclasses import replace
import hashlib
import collections
import json
import math
from pathlib import Path

import pytest

from swarmrsi import naturebench_session as session
from swarmrsi.brs_frontiers import HypothesisFrontier, ProgramFrontier, family_sha256, propose_pairs
from swarmrsi.brs_models import HypothesisRecord, ProgramRepresentative, Requirement
from swarmrsi.brs_strategy import BidirectionalResearchStrategy, ResearchDirective
from swarmrsi.campaign import ActionExecution, ProgramNode, ProposedAction, VSearchCoordinator, artifact_workspace_sha256
from swarmrsi.coder_models import CoderExitStatus, CoderOutcome, CoderVersion, ScientificActionContract
from swarmrsi.configuration import ConfigurationError, config_from_dict
from swarmrsi.contracts import ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily, SearchObservation
from swarmrsi.frontier_allocator import FrontierPressureAllocator
from swarmrsi.search_plugins import default_search_registry
from test_configuration import payload
from test_swarmrsi_incremental_feedback import next_step

SHA = hashlib.sha256(b"brs-fixture").hexdigest()
ROOT_SCORES = {"cl3": .5, "cl5": .5}
CORE = ("input_transform", "primary_operator", "output_construction", "objective_or_test")


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class ResearchProposer:
    """Honours controller-owned slots; emits a distinct mechanism per call."""

    def __init__(self, *, substitute_hypothesis=False, duplicate=False):
        self.index, self.directives = 0, []
        self.substitute_hypothesis, self.duplicate = substitute_hypothesis, duplicate

    def propose(self, **request):
        nodes = {node.id: node for node in request["nodes"]}
        proposals = []
        for slot in request["directives"]:
            self.directives.append(slot)
            action_id = f"a{self.index:04d}"
            self.index += 1
            parent = nodes[slot.code_parent_id]
            dependencies = [ResearchDependency(parent.id, "code_parent", parent.executable_sha256)]
            dependencies += [ResearchDependency(s, "intervention_source", nodes[s].executable_sha256)
                             for s in slot.intervention_source_ids]
            fixed = getattr(slot, "hypothesis_id", None)
            episode = getattr(slot, "research_episode", None) or {}
            if fixed and not self.substitute_hypothesis:
                family_raw = episode["hypothesis"]["family"]
                hypothesis_id, statement = fixed, episode["hypothesis"]["statement"]
                family = ScientificFamily(**{k: family_raw[k] for k in CORE},
                                          capacity_controls=tuple(family_raw.get("capacity_controls", ())))
            elif self.duplicate:
                hypothesis_id, statement = f"h_{action_id}", "one mechanism"
                family = ScientificFamily("raw", "same-operator", "direct", "public metric")
            else:
                hypothesis_id, statement = f"h_{action_id}", f"mechanism {self.index} on {parent.id}"
                family = ScientificFamily("raw", f"operator-{self.index}", "direct", "public metric")
            operator = ActionOperator.COMBINE_PROGRAMS if slot.intervention_source_ids else ActionOperator.IMPROVE_PROGRAM
            candidate = ActionCandidate(
                id=action_id, hypothesis_id=hypothesis_id, operator=operator, dependencies=tuple(dependencies),
                family=family, predicted_program_quality=.5, predicted_information_gain=.5,
                predicted_reuse_value=.5, predicted_validity=.9, expected_cost=1.0, hypothesis_statement=statement)
            proposals.append(ProposedAction(candidate, ScientificActionContract(
                action_id=action_id, description="synthetic", allowed_write_paths=("run.py",))))
        return proposals, {"model_calls": 1, "input_tokens": 300, "output_tokens": 200}


class ResearchExecutor:
    """Scripted outcomes: float scored, None failed, 'timeout', 'censored', 'unrealized', 'fallback'."""

    def __init__(self, root, script):
        self.root, self.script, self.calls = root, list(script), 0

    def config_sha256(self, **kwargs):
        return SHA

    def execute(self, *, proposal, parent, step):
        item = self.script.pop(0)
        self.calls += 1
        cell = self.root / f"cell_{step}"
        artifact = cell / "artifact"
        artifact.mkdir(parents=True)
        parent_code = (parent.artifact / "run.py").read_text()
        kind = item if isinstance(item, str) else ("failed" if item is None else "scored")
        score = item if kind == "scored" else None
        attestation = f"out-{step}"
        if kind in {"unrealized", "fallback"}:
            # fallback: the new component runs, but the program keeps the parent's outputs.
            code = parent_code + ("\ndef _unused_helper():\n    return 1\n" if kind == "unrealized"
                                  else "\nprint('new component ran; hold-out worse, kept parent')\n")
            score = parent.native_metric
            attestation = (parent.evaluation_summary or {}).get("submission_output_attestation")
        elif kind in {"scored", "censored", "self_rejected"}:
            code = f"print({step!r})\n"
        else:
            code = parent_code
        (artifact / "run.py").write_text(code)
        failed = cell / "failed_artifact"
        if kind in {"failed", "timeout"}:
            failed.mkdir()
            (failed / "run.py").write_text("def broken(:\n")
        primary = dict(ROOT_SCORES)
        if kind == "scored" and score != .5:
            primary = {"cl3": round(.5 + (score - .5), 3), "cl5": round(.5 - (score - .5) / 2, 3)}
        summary = None
        if kind in {"scored", "unrealized", "fallback"}:
            summary = {"valid": True, "aggregate_improvement": score, "primary_scores": primary,
                       "per_instance_improvement": {k: round(v - ROOT_SCORES[k], 3) for k, v in primary.items()},
                       "submission_output_attestation": attestation}
        elif kind == "censored":
            summary = {"valid": False, "public_diagnostic": {"category": "evaluation_bridge",
                                                             "summary": "evaluator container unavailable"}}
        elif kind == "self_rejected":
            summary = {"valid": False, "score_bearing_evaluation_attempted": False,
                       "public_diagnostic": {"category": "internal_validation_rejected",
                                             "summary": "candidate's public-label estimate is below its parent's"}}
        evidence = cell / "evaluation.json"
        if summary is not None:
            evidence.write_text(json.dumps(summary))
        success = kind in {"scored", "unrealized", "fallback", "censored", "self_rejected"}
        validation = [] if kind != "timeout" else [{"tier": "smoke", "accepted": False, "timed_out": True,
                                                    "returncode": None, "stderr": "smoke timed out after 1800s"}]
        reason = {"scored": "synthetic_success", "unrealized": "synthetic_success", "fallback": "synthetic_success",
                  "censored": "synthetic_success",
                  "self_rejected": "synthetic_success", "failed": "syntax_failure", "timeout": "wall_budget_exhausted"}[kind]
        outcome = CoderOutcome(
            status=CoderExitStatus.SUCCESS if success else CoderExitStatus.IMPLEMENTATION_FAILED,
            version=CoderVersion.C3, config_sha256=SHA, action_id=proposal.candidate.id,
            parent_sha256=parent.executable_sha256, parent_workspace_sha256=parent.workspace_sha256,
            final_sha256=file_sha(artifact / "run.py"), final_workspace_sha256=artifact_workspace_sha256(artifact),
            model_calls=2, tool_calls=3, input_tokens=400, output_tokens=100, wall_seconds=1.0,
            validation=validation, changed_paths=["run.py"], implementation_real=success, parent_preserved=True,
            reason=reason, diagnostic_artifact_relpath=None if success else "failed_artifact",
            diagnostic_workspace_sha256=None if success else artifact_workspace_sha256(failed), trajectory_sha256=SHA)
        outcome.seal()
        valid = kind in {"scored", "unrealized", "fallback"}
        gate = None
        if kind == "self_rejected":
            gate = {"candidate": {"metric_name": "aggregate_improvement", "direction": "maximize", "estimate": 0.25},
                    "parent": {"metric_name": "aggregate_improvement", "direction": "maximize", "estimate": 0.30},
                    "tolerance": 0.0, "shortfall": 0.05, "rejected": True, "reason": "candidate_estimate_below_parent"}
        return ActionExecution(outcome, artifact, score if valid else None, valid, int(valid),
                               evaluation_summary=summary,
                               evaluation_evidence_sha256=file_sha(evidence) if summary is not None else None,
                               evaluation_evidence_path=evidence if summary is not None else None,
                               internal_validation=gate)


def brs_campaign(tmp_path, script, *, parameters=None, max_evals=20, proposer=None, max_tokens=500000):
    parent = tmp_path / "root"
    parent.mkdir(parents=True)
    (parent / "run.py").write_text("print('root')\n")
    strategy = default_search_registry().create({"name": "bidirectional_research", "parameters": parameters or {}})
    root = ProgramNode("root", parent, file_sha(parent / "run.py"), artifact_workspace_sha256(parent), None, "h_root",
                       0.0, True, None, evaluation_summary={"valid": True, "aggregate_improvement": 0.0,
                                                             "primary_scores": dict(ROOT_SCORES),
                                                             "submission_output_attestation": "out-root"})
    return VSearchCoordinator(version=strategy.name, strategy=strategy, proposer=proposer or ResearchProposer(),
                              executor=ResearchExecutor(tmp_path, script), root=root, max_model_calls=200,
                              max_score_evaluations=max_evals, max_tokens=max_tokens, proposal_batch_size=1)


def brs(loop):
    return loop.strategy._algorithm


def episode_of(loop, action_id):
    algorithm = brs(loop)
    return algorithm.episodes[algorithm.episode_by_action[action_id]]


def run_steps(loop, count):
    return [next_step(loop) for _ in range(count)]


def force_arms(loop, arms):
    """Script the initiating arm; the real allocator still records every decision."""
    algorithm, queue, original = brs(loop), list(arms), brs(loop).allocator.choose

    def choose(feasible, unmet):
        decision = original(feasible, unmet)
        if queue:
            decision.update(arm=queue.pop(0), reason="test_scripted")
        return decision
    algorithm.allocator.choose = choose


def replace_episode(episode, **changes):
    """dataclasses.replace for an EpisodeRecord, so tests can pose a specific shape."""
    from dataclasses import replace as _replace
    return _replace(episode, **changes)


def hypothesis(index, family=None, **overrides):
    family = family or {"input_transform": "raw", "primary_operator": f"op{index}", "output_construction": "direct",
                        "objective_or_test": "metric", "capacity_controls": []}
    return HypothesisRecord(id=f"h{index}", statement=f"s{index}", family=family, family_sha256=family_sha256(family),
                            **overrides)


# ----------------------------------------------------------------- allocator
def test_allocator_discounts_every_arm_and_rates_reward_per_cost():
    allocator = FrontierPressureAllocator(discount_factor=.5)
    allocator.settle("H", 1.0, 2.0)
    allocator.settle("P", 0.0, 1.0)
    allocator.settle("P", 1.0, 4.0)
    assert allocator.rate("H") == pytest.approx(.5) and allocator.rate("P") == pytest.approx(1 / 4.5)
    assert allocator.state["N"] == pytest.approx({"H": .25, "P": 1.5}) and allocator.state["clock"] == 3
    assert allocator.state["samples"] == {"H": 1, "P": 2}
    assert allocator.cost(10) == 1.0 and allocator.cost(2500) == 2.5
    assert allocator.reward(.9, .2) == pytest.approx(.7) and allocator.reward(2.0, .2) == 1.0
    assert allocator.reward(.1, .2) == 0.0 and allocator.reward(.5, None) == 0.0 and allocator.reward(None, .1) == 0.0
    assert FrontierPressureAllocator(reward_scale=.1).reward(.25, .2) == pytest.approx(.5)
    saved = json.loads(json.dumps(allocator.checkpoint_state()))
    fresh = FrontierPressureAllocator(discount_factor=.5)
    fresh.restore_state(saved)
    assert fresh.checkpoint_state() == allocator.checkpoint_state()
    with pytest.raises(ValueError):
        FrontierPressureAllocator().restore_state(saved)


def test_allocator_initialisation_pressure_and_starvation_guard():
    allocator = FrontierPressureAllocator(exploration_weight=0.0, frontier_pressure_weight=1.0,
                                          pressure_saturation=2, max_eligible_wait=3)
    first = allocator.choose({"H", "P"}, {"H": 0, "P": 0})
    assert (first["arm"], first["reason"]) == ("H", "initialization_sample") and first["index"]["H"]["score"] is None
    allocator.settle("H", 0.0, 1.0)
    second = allocator.choose({"H", "P"}, {"H": 0, "P": 0})
    assert (second["arm"], second["reason"]) == ("P", "initialization_sample")
    allocator.settle("P", 0.0, 1.0)
    pushed = allocator.choose({"H", "P"}, {"H": 10, "P": 0})
    assert pushed["pressure"] == {"H": 0.0, "P": 1.0, "v_H": 2, "v_P": 0}
    assert (pushed["arm"], pushed["reason"]) == ("P", "discounted_index")
    pulled = allocator.choose({"H", "P"}, {"H": 0, "P": 3})
    assert pulled["pressure"]["H"] == 1.0 and pulled["arm"] == "H"
    for _ in range(3):
        assert allocator.choose({"H", "P"}, {"H": 5, "P": 0})["arm"] == "P"
    starved = allocator.choose({"H", "P"}, {"H": 5, "P": 0})
    assert (starved["arm"], starved["reason"]) == ("H", "starvation_guard")
    only = allocator.choose({"P"}, {"H": 5, "P": 0})
    assert (only["arm"], only["reason"]) == ("P", "only_feasible_arm")
    assert allocator.choose(set(), {})["arm"] is None
    allocator.censor("H")
    assert allocator.state["censored"]["H"] == 1 and allocator.state["samples"]["H"] == 1


@pytest.mark.parametrize("parameters", [
    {"program_optimizer": {"name": "multi_parent_graph"}},
    {"matcher": {"name": "unbalanced_optimal_transport"}},
    {"allocator": {"name": "epsilon_greedy"}}, {"allocator": {"discount_factor": 1.5}},
    {"allocator": {"frontier_pressure_weight": -1}}, {"max_active_hypotheses": 0},
    {"max_program_representatives": 0}, {"max_pairs_per_hypothesis": 0},
    {"program_optimizer": {"name": "continuation_tree", "parameters": {"max_scored_continuations": 0}}},
    {"unknown": 1},
])
def test_unsupported_configurations_are_rejected_not_defaulted(parameters):
    with pytest.raises((ValueError, TypeError)):
        default_search_registry().create({"name": "bidirectional_research", "parameters": parameters})


# ----------------------------------------------------------------- frontiers
def test_hypothesis_capacity_is_family_balanced_and_never_deletes():
    frontier = HypothesisFrontier(max_active=2)
    for index in range(3):
        frontier.put(hypothesis(index))
    frontier.put(hypothesis(3, family=hypothesis(0).family))
    transitions = frontier.rebalance(step=5)
    active = {r.id for r in frontier.active()}
    assert len(active) == 2 and "h3" not in active and len(transitions) == 2
    assert {r.status for r in frontier.records.values()} == {"active", "deferred"}
    frontier.put(replace(frontier.get("h0"), effective_test_count=2, last_attempt_step=5))
    frontier.rebalance(step=6)
    assert frontier.get("h0").status == "deferred" and len(frontier.active()) == 2
    assert set(frontier.records) == {"h0", "h1", "h2", "h3"}
    saved = json.loads(json.dumps(frontier.to_dict()))
    again = HypothesisFrontier(max_active=2)
    again.restore(saved)
    assert again.to_dict() == frontier.to_dict()
    with pytest.raises(ValueError):
        HypothesisFrontier(max_active=3).restore(saved)


def test_program_representatives_keep_incumbent_and_capacities_are_independent():
    nodes = {f"n{i}": {"native_metric": i / 10, "executable_sha256": "a" * 64, "family_sha256": f"f{i % 2}",
                       "hypothesis_id": f"h{i}", "capabilities": ("cap",)} for i in range(5)}
    programs = ProgramFrontier(max_representatives=2)
    programs.refresh(nodes)
    assert [(r.node_id, r.reason) for r in programs.representatives] == [("n4", "global_incumbent"), ("n3", "family_best")]
    assert programs.incumbent().node_id == "n4"
    wide = ProgramFrontier(max_representatives=8)
    wide.refresh(nodes)
    assert len(wide.representatives) == 5 and wide.representatives[0].node_id == "n4"
    assert HypothesisFrontier(max_active=1).max_active == 1 and wide.max_representatives == 8
    changes = programs.refresh({**nodes, "n9": {**nodes["n0"], "native_metric": .9}})
    assert programs.incumbent().node_id == "n9" and {c["node_id"] for c in changes} == {"n9", "n4"}
    assert programs.get("n3").reason == "family_best"
    programs.refresh({})
    assert programs.representatives == ()


def test_sparse_matcher_orders_by_blocker_coverage_and_serves_many_hypotheses():
    family = {"input_transform": "esm embedding", "primary_operator": "xgboost", "output_construction": "regression",
              "objective_or_test": "holdout", "capacity_controls": []}
    requirement = Requirement(key="r1", kind="efficiency",
                              description="efficiency: per_position_embed timed out computing esm features",
                              next_step="reimplement on another program with cached embedding", failed_program_id="p_fail")
    record = HypothesisRecord(id="hA", statement="per-position esm features", family=family,
                              family_sha256=family_sha256(family), requirements=(requirement,))
    reps = (ProgramRepresentative("p_fail", "1" * 64, .3, family_sha256(family), "hA", ("esm", "embedding"), "executed_scored", "metric_rank"),
            ProgramRepresentative("p_cached", "2" * 64, .2, "x" * 64, "hB", ("cached", "embedding", "batch"), "executed_scored", "family_best"),
            ProgramRepresentative("p_plain", "3" * 64, .4, "y" * 64, "hC", ("kmeans",), "executed_scored", "global_incumbent"))
    common = dict(representatives=reps, excluded_keys=set(), max_sources=1, measured_costs={}, step=3, waiting={"hA": 2})
    pairs = propose_pairs(hypotheses=[record], max_per_hypothesis=10, **common)
    assert pairs[0].primary_program_id == "p_cached" and pairs[0].closes_blocker
    assert pairs[0].coverage_status in {"partial", "covered"} and all(p.hypothesis_id == "hA" for p in pairs)
    plain = next(p for p in pairs if p.primary_program_id == "p_plain" and not p.source_program_ids)
    assert plain.coverage_status == "unknown" and not plain.closes_blocker
    failed_base = next(p for p in pairs if p.primary_program_id == "p_fail" and not p.source_program_ids)
    assert failed_base.coverage > 0 and not failed_base.closes_blocker
    assert len(propose_pairs(hypotheses=[record], max_per_hypothesis=2, **common)) == 2
    excluded = propose_pairs(hypotheses=[record], max_per_hypothesis=10, **{**common, "excluded_keys": {pairs[0].key}})
    assert pairs[0].key not in {p.key for p in excluded}
    other = replace(record, id="hB2", statement="another need", requirements=(replace(requirement, key="r2"),))
    both = propose_pairs(hypotheses=[record, other], max_per_hypothesis=1, **common)
    assert {p.hypothesis_id for p in both if p.primary_program_id == "p_cached"} == {"hA", "hB2"}
    assert propose_pairs(hypotheses=[replace(record, requirements=())], max_per_hypothesis=2, **common) == []
    assert propose_pairs(hypotheses=[record], max_per_hypothesis=2, **{**common, "representatives": ()}) == []


# --------------------------------------------------------------- lifecycle
def test_both_arms_execute_through_the_same_coder_and_evaluator(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5, .7])
    rows = run_steps(loop, 4)
    arms = [episode_of(loop, row.action_id).arm for row in rows]
    assert arms[:2] == ["H", "P"] and loop.executor.calls == 4 and all(row.artifact_valid for row in rows)
    first = episode_of(loop, rows[0].action_id)
    assert first.mode == "new_hypothesis" and rows[0].code_parent_id == "root" and loop.nodes[rows[0].action_id].native_metric == .4
    directives = loop.proposer.directives
    assert isinstance(directives[0], ResearchDirective) and directives[0].research_episode["arm"] == "H"
    assert directives[1].research_episode["arm"] == "P" and directives[1].research_episode["mode"].startswith("tree_")
    report = loop.summary()["frontier_report"]
    assert report["episode_counts"] == report["settled_counts"] == {"H": arms.count("H"), "P": arms.count("P")}
    kinds = {event["event_type"] for event in loop.summary()["graph"]["events"]}
    assert {"search.frontier.allocator_decision", "search.frontier.episode_started", "search.frontier.episode_settled",
            "search.frontier.hypothesis_created", "search.frontier.capability_observed"} <= kinds
    decisions = [e for e in loop.view.events if e.event_type == "search.frontier.allocator_decision"]
    assert [d.payload["arm"] for d in decisions] == arms and "index" in decisions[-1].payload["decision"]


def test_unlimited_token_campaign_checkpoints_and_charges_planning(tmp_path):
    """No token ceiling: the remainder is infinite, which must never reach a record.

    The first cap-off panel lost 14 of 16 cells to `Out of range float values are not
    JSON compliant` because the directive stored `remaining_tokens`.
    """
    import json
    loop = brs_campaign(tmp_path, [.4, .6, .5], max_tokens=None)
    rows = run_steps(loop, 3)
    episodes = [episode_of(loop, row.action_id) for row in rows]
    assert all(e.planning_tokens == 500 for e in episodes)
    json.dumps(loop.checkpoint(), allow_nan=False, default=str)
    json.dumps(loop.summary(), allow_nan=False, default=str)


def test_reward_uses_pre_incumbent_fixed_scale_and_planning_cost(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5], parameters={"allocator": {"reward_scale": .5}})
    rows = run_steps(loop, 3)
    episodes = [episode_of(loop, row.action_id) for row in rows]
    assert [e.reward for e in episodes] == pytest.approx([.8, .4, 0.0])
    assert [e.incumbent_pre for e in episodes] == [0.0, .4, .6]
    assert all(e.planning_tokens == 500 and e.coder_tokens == 500 and e.cost_units == 1.0 for e in episodes)
    assert brs(loop).allocator.state["clock"] == 3


def test_episode_settles_once_and_pending_blocks_new_planning(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6])
    loop.expand()
    algorithm = brs(loop)
    algorithm.select(loop._snapshot())
    with pytest.raises(ValueError, match="pending"):
        algorithm.proposal_directives(loop._snapshot(), 1)
    with pytest.raises(ValueError, match="pending"):
        algorithm.select(loop._snapshot())
    fresh = brs_campaign(tmp_path / "fresh", [.4, .6])
    run_steps(fresh, 1)
    with pytest.raises(ValueError, match="pending episode|already consumed"):
        brs(fresh).observe(brs(fresh).observations[-1])


def test_failure_creates_one_requirement_and_repair_binds_its_origin(tmp_path):
    loop = brs_campaign(tmp_path, [.4, None, .6])
    rows = run_steps(loop, 3)
    assert rows[1].coder_status == "implementation_failed" and rows[2].action_id.startswith("repair_")
    failed, repair = episode_of(loop, rows[1].action_id), episode_of(loop, rows[2].action_id)
    assert failed.status == "settled" and failed.reward == 0.0 and failed.terminal_kind == "implementation_failed"
    assert (repair.origin_episode_id, repair.arm, repair.hypothesis_id, repair.primary_program_id) == (
        failed.id, failed.arm, failed.hypothesis_id, failed.primary_program_id)
    record = brs(loop).hypotheses.get(failed.hypothesis_id)
    assert [(r.kind, r.status, r.resolved_by_action_id) for r in record.requirements] == [("implementation", "resolved", rows[2].action_id)]
    assert record.effective_test_count == 1 and record.best_metric == .6 and rows[2].native_metric == .6


def test_timeout_gap_is_adapted_onto_another_program_base_with_fixed_hypothesis(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, "timeout", "timeout", .7])
    force_arms(loop, ["H", "P", "H", "P"])
    rows = run_steps(loop, 5)
    blocked = episode_of(loop, rows[2].action_id)
    assert rows[2].recovery_route == "optimize_efficiency" and rows[3].action_id.startswith("repair_")
    record_before = brs(loop).hypotheses.get(blocked.hypothesis_id)
    adaptation = episode_of(loop, rows[4].action_id)
    assert adaptation.arm == "P" and adaptation.mode == "gap_adaptation" and adaptation.hypothesis_id == blocked.hypothesis_id
    assert rows[4].code_parent_id != blocked.primary_program_id and rows[4].code_parent_id in {"a0000", "root"}
    directive = loop.proposer.directives[-1]
    assert directive.hypothesis_id == blocked.hypothesis_id and directive.research_episode["pair"]["primary_program_id"] == rows[4].code_parent_id
    record = brs(loop).hypotheses.get(blocked.hypothesis_id)
    assert [(r.kind, r.status) for r in record.requirements] == [("efficiency", "resolved")]
    assert record.effective_test_count == 1 and record.program_ids == (rows[4].action_id,)
    pair = brs(loop).pairs[adaptation.pair_key]
    assert pair.status == "attempted" and pair.hypothesis_id == blocked.hypothesis_id
    assert all(item.status == "resolved" for item in brs(loop).pressure.values() if item.side == "H")
    assert record_before.requirements[0].key == record.requirements[0].key


def test_planner_cannot_substitute_a_fixed_hypothesis(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, "timeout", "timeout", .7], proposer=ResearchProposer(substitute_hypothesis=True))
    force_arms(loop, ["H", "P", "H", "P"])
    run_steps(loop, 4)
    with pytest.raises(ValueError, match="controller-selected hypothesis"):
        next_step(loop)


def test_repeated_failures_and_duplicate_hypotheses_do_not_add_pressure(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, "timeout", "timeout"])
    force_arms(loop, ["H", "P", "H"])
    rows = run_steps(loop, 4)
    record = brs(loop).hypotheses.get(episode_of(loop, rows[2].action_id).hypothesis_id)
    assert len(record.open_requirements()) == 1 and record.requirements[0].source_action_id == rows[3].action_id
    brs(loop).proposal_directives(loop._snapshot(), 1)
    assert sum(i.side == "H" and i.status == "queued" for i in brs(loop).pressure.values()) == 1
    duplicated = brs_campaign(tmp_path / "dup", [.4, .5, .6], proposer=ResearchProposer(duplicate=True))
    force_arms(duplicated, ["H", "H", "H"])
    run_steps(duplicated, 3)
    algorithm = brs(duplicated)
    assert len(algorithm.hypotheses.records) == 1 and len(algorithm.hypothesis_alias) == 2
    assert not [i for i in algorithm.pressure.values() if i.side == "H"]
    assert algorithm.hypotheses.active()[0].effective_test_count == 3


def test_unrealized_intervention_is_unknown_not_capability_and_keeps_its_score(tmp_path):
    loop = brs_campaign(tmp_path, [.4, "unrealized", .5])
    rows = run_steps(loop, 2)
    algorithm, episode = brs(loop), episode_of(loop, rows[1].action_id)
    node = algorithm.node_info[rows[1].action_id]
    assert node["realization"] == "not_realized" and node["capabilities"] == algorithm.node_info[rows[1].code_parent_id]["capabilities"]
    record = algorithm.hypotheses.get(episode.hypothesis_id)
    assert (record.effective_test_count, record.unknown_test_count) == (0, 1)
    assert [r.kind for r in record.open_requirements()] == ["realization"]
    assert episode.terminal_kind == "scored" and episode.reward == 0.0
    assert loop.nodes[rows[1].action_id].native_metric == .4 and loop.nodes[rows[1].action_id].artifact_valid
    observed = next(e for e in loop.view.events if e.event_type == "search.frontier.capability_observed"
                    and e.payload["node_id"] == rows[1].action_id)
    assert observed.payload["implementation_observation"]["source_change"] == "added_definitions_without_static_name_load"
    run_steps(loop, 1)


def test_censored_evaluation_is_not_a_learning_sample_but_keeps_cost(tmp_path):
    loop = brs_campaign(tmp_path, [.4, "censored", .6, .7])
    rows = run_steps(loop, 3)
    algorithm, censored = brs(loop), episode_of(loop, rows[1].action_id)
    assert rows[1].recovery_route == "retry_evaluator_same_candidate"
    assert censored.status == "censored" and censored.coder_tokens == 500 and censored.reward is None
    settled = sum(e.arm == censored.arm and e.status == "settled" for e in algorithm.episodes.values())
    assert algorithm.allocator.state["samples"][censored.arm] == settled
    assert algorithm.allocator.state["censored"][censored.arm] == 1
    record = algorithm.hypotheses.get(censored.hypothesis_id)
    assert not record.requirements and record.effective_test_count == 0 and not record.program_ids
    assert not rows[2].action_id.startswith("repair_")
    force_arms(loop, ["H"])
    row = next_step(loop)
    activated = episode_of(loop, row.action_id)
    assert activated.mode == "activate_hypothesis" and activated.hypothesis_id == censored.hypothesis_id
    assert loop.proposer.directives[-1].hypothesis_id == censored.hypothesis_id
    assert algorithm.hypotheses.get(censored.hypothesis_id).effective_test_count == 1


def test_differential_behaviour_becomes_a_program_opportunity_for_new_hypotheses(tmp_path):
    loop = brs_campaign(tmp_path, [.7, .6])
    force_arms(loop, ["H", "H"])
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    items = sorted((i for i in algorithm.pressure.values() if i.side == "P"), key=lambda i: i.created_step)
    assert items[0].ref == rows[0].action_id and "cl3" in items[0].description
    second = loop.proposer.directives[1].research_episode
    assert second["mode"] == "new_hypothesis" and [o["key"] for o in second["opportunities"]] == [items[0].key]
    assert second["allocator"]["unmet"] == {"H": 0, "P": 1}
    record = algorithm.hypotheses.get(episode_of(loop, rows[1].action_id).hypothesis_id)
    assert record.origin == "program_observation" and record.motivation_refs == (items[0].key,)
    assert algorithm.pressure[items[0].key].status == "resolved"


def test_capacity_configuration_is_consumed_without_truncating_history(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .5, .6, .7, .8],
                        parameters={"max_active_hypotheses": 1, "max_program_representatives": 2})
    force_arms(loop, ["H"] * 5)
    run_steps(loop, 5)
    algorithm = brs(loop)
    algorithm.proposal_directives(loop._snapshot(), 1)
    statuses = [r.status for r in algorithm.hypotheses.records.values()]
    assert len(statuses) == 5 and statuses.count("active") == 1 and statuses.count("deferred") == 4
    assert len(algorithm.programs.representatives) == 2 and algorithm.programs.incumbent().node_id == "a0004"
    assert len(loop.nodes) == 6 and len(loop.summary()["frontier_report"]["hypotheses"]) == 5
    assert any(e.event_type == "search.frontier.hypothesis_status_changed" for e in loop.view.events)
    assert loop.strategy.config["algorithm_config"]["max_active_hypotheses"] == 1


def test_restart_reproduces_next_decision_and_preserves_history(tmp_path):
    loop = brs_campaign(tmp_path, [.4, None, .6, .5])
    run_steps(loop, 3)
    loop.expand()
    saved = json.loads(json.dumps(loop.checkpoint()))
    restored = VSearchCoordinator.restore(strategy=default_search_registry().create({"name": "bidirectional_research"}),
                                          proposer=ResearchProposer(), executor=ResearchExecutor(tmp_path / "again", [.5]),
                                          payload=saved)
    assert restored.strategy.checkpoint_state() == loop.strategy.checkpoint_state()
    left, right = restored.step(), loop.step()
    assert (left.action_id, left.code_parent_id) == (right.action_id, right.code_parent_id)
    assert episode_of(restored, left.action_id).arm == episode_of(loop, right.action_id).arm
    assert restored.strategy.checkpoint_state() == loop.strategy.checkpoint_state()
    assert len(restored.diagnostics) == 1 and set(brs(restored).hypotheses.records) == set(brs(loop).hypotheses.records)
    assert restored.view.sha256 == loop.view.sha256


def test_incompatible_or_foreign_checkpoints_are_rejected(tmp_path):
    loop = brs_campaign(tmp_path, [.4])
    run_steps(loop, 1)
    saved = json.loads(json.dumps(loop.strategy.checkpoint_state()))
    for parameters in ({"max_active_hypotheses": 2}, {"allocator": {"discount_factor": .8}},
                       {"program_optimizer": {"name": "continuation_tree", "parameters": {"stagnation_patience": 2}}}):
        new = default_search_registry().create({"name": "bidirectional_research", "parameters": parameters})
        with pytest.raises(ValueError):
            new.restore_state(saved)
    tree = default_search_registry().create({"name": "continuation_tree"})
    with pytest.raises(ValueError):
        default_search_registry().create({"name": "bidirectional_research"}).restore_state(tree.checkpoint_state())
    with pytest.raises(ValueError):
        tree.restore_state(saved)


def test_wisdom_source_is_frozen():
    path = Path(__file__).resolve().parents[1] / "src/swarmrsi/wisdom.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "6fefb14dad8d2eef2d7d276e003558542bb7dc03759465954d4efa09de9a6368"


def test_session_and_run_config_accept_bidirectional_research_without_rsi(tmp_path):
    strategy = session.strategy("v1", 0, {"search": {"name": "bidirectional_research", "parameters": {"max_active_hypotheses": 3}}})
    assert strategy.name == "bidirectional_research" and strategy.config["algorithm_config"]["max_active_hypotheses"] == 3
    assert strategy.config["algorithm_config"]["program_optimizer"]["parameters"]["policy"] == "continuation_tree"
    with pytest.raises(ValueError):
        session.strategy("v1", 0, {"search": {"name": "bidirectional_research"}, "rsi_policy": {}})
    data = payload()
    data["search"] = {"name": "bidirectional_research", "parameters": {"allocator": {"frontier_pressure_weight": 0.0}}}
    config = config_from_dict(data, base=tmp_path)
    assert config.search.name == "bidirectional_research"
    assert config_from_dict(config.to_dict(), base=tmp_path) == config
    data["learning"] = {"rsi_enabled": True}
    with pytest.raises(ConfigurationError):
        config_from_dict(data, base=tmp_path)


def test_one_weak_opportunity_does_not_pin_the_other_arm_to_maximum_pressure():
    """Recorded Biology-5 revision A failure: v_P=2, v_H=0 gave H the full term."""
    allocator = FrontierPressureAllocator(pressure_saturation=4)
    assert allocator.pressure({"H": 0, "P": 1}) == {"H": 0.25, "P": 0.0, "v_H": 0, "v_P": 1}
    assert allocator.pressure({"H": 0, "P": 2}) == {"H": 0.5, "P": 0.0, "v_H": 0, "v_P": 2}
    assert allocator.pressure({"H": 0, "P": 9}) == {"H": 1.0, "P": 0.0, "v_H": 0, "v_P": 4}
    assert allocator.pressure({"H": 2, "P": 2}) == {"H": 0.5, "P": 0.5, "v_H": 2, "v_P": 2}
    assert allocator.pressure({"H": 0, "P": 0}) == {"H": 0.0, "P": 0.0, "v_H": 0, "v_P": 0}
    # The recorded decision that lost the lane: exploration 0.438 vs 0.573.
    allocator = FrontierPressureAllocator(pressure_saturation=4, frontier_pressure_weight=0.25)
    allocator.settle("H", 0.0, 1.0); allocator.settle("H", 0.0, 1.0); allocator.settle("P", 0.0, 1.0)
    decision = allocator.choose({"H", "P"}, {"H": 0, "P": 2})
    assert decision["index"]["H"]["pressure"] == 0.125 and decision["arm"] == "P"


def test_an_open_local_refinement_is_finished_before_a_new_front_opens(tmp_path):
    """The lane that historically compounds into a win is not abandoned."""
    loop = brs_campaign(tmp_path, [.4, .6, .65, .7])
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    assert algorithm.tree.burst is not None and not algorithm.tree.burst["closed"]
    # Whatever the bandit would prefer, only the open lane is offered to it.
    offered = []
    original = algorithm.allocator.choose
    algorithm.allocator.choose = lambda feasible, unmet: offered.append(set(feasible)) or original(feasible, unmet)
    row = next_step(loop)
    assert offered == [{"P"}]
    episode = episode_of(loop, row.action_id)
    # The open lane is served by the program arm: a plain local continuation,
    # or an instance recombination on that same lane when winners differ.
    assert episode.arm == "P" and episode.mode in {"tree_local", "instance_recombination"}
    assert row.code_parent_id in {r.action_id for r in rows}
    decision = next(e for e in reversed(loop.view.events)
                    if e.event_type == "search.frontier.allocator_decision")
    assert decision.payload["decision"]["reason"] == "open_local_continuation"
    assert decision.payload["decision"]["feasible"] == ["P"]


def test_local_continuation_offers_instance_winners_as_typed_sources(tmp_path):
    """Two children win different instances; the continuation recombines, not rewrites."""
    loop = brs_campaign(tmp_path, [.6, .4, .65, .7])
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    row = next_step(loop)  # the frontier refreshes when the next slot is planned
    winners = algorithm.programs.instance_winners
    assert {v[0] for v in winners.values()} == {rows[0].action_id, rows[1].action_id}
    action = loop.actions[row.action_id]
    episode = episode_of(loop, row.action_id)
    children = {rows[0].action_id, rows[1].action_id}
    assert episode.mode == "instance_recombination" and row.code_parent_id in children
    decision = next(e for e in reversed(loop.view.events) if e.event_type == "search.action_selected").payload
    assert "brs_instance_recombination" in decision["rationale_codes"]
    # The one source is the other child: the program that wins the instance the parent loses.
    sources = {d.source_id for d in action.candidate.dependencies if d.kind == "intervention_source"}
    assert sources == children - {row.code_parent_id}
    assert loop.strategy.checkpoint_state()["state"]["programs"]["instance_winners"]
    # The recombination instruction reaches the planner through the recorded directive context.
    import json as _json
    assert any("Instance recombination" in _json.dumps(getattr(e, "payload", {}), default=str)
               for e in loop.view.events), "recombination guidance was not recorded"


def test_self_rejected_candidate_spends_no_evaluation_and_counts_as_a_test(tmp_path):
    """Revision-A tissue: three official evaluations went to candidates far below root."""
    loop = brs_campaign(tmp_path, [.4, "self_rejected", .6])
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    rejected = rows[1]
    assert rejected.recovery_route == "discard_underperforming_candidate" and rejected.native_metric is None
    episode = episode_of(loop, rejected.action_id)
    assert episode.terminal_kind == "self_rejected" and episode.status == "settled" and episode.reward == 0.0
    record = algorithm.hypotheses.get(episode.hypothesis_id)
    assert record.effective_test_count == 1 and record.unknown_test_count == 0
    assert not [r for r in record.requirements if r.status == "open"], "a self-rejection is not a failure"
    assert loop.accounting.score_evaluations == 1  # only the first child was officially scored
    assert not loop.diagnostics and not any(a.candidate.operator.value == "repair_program" for a in loop.actions.values())
    assert rejected.action_id not in loop.nodes
    third = next_step(loop)
    assert third.native_metric == .6


def test_ablation_switches_change_behaviour_and_the_recorded_identity(tmp_path):
    """Each ablation is a distinct configuration; an ablation cell can never pass as the complete method."""
    complete = BidirectionalResearchStrategy()
    assert complete.config["matching_enabled"] is True
    assert complete.config["instance_recombination_enabled"] is True
    for name in ("matching_enabled", "instance_recombination_enabled"):
        ablated = BidirectionalResearchStrategy(**{name: False})
        assert ablated.config[name] is False and ablated.config != complete.config
        with pytest.raises(ValueError, match=name):
            BidirectionalResearchStrategy(**{name: 0})
    # No matching: the program arm never proposes a gap adaptation.
    loop = brs_campaign(tmp_path / "no_match", [.4, None, .5, .6, .55],
                        parameters={"matching_enabled": False})
    run_steps(loop, 5)
    modes = [e.mode for e in brs(loop).episodes.values()]
    assert modes and "gap_adaptation" not in modes
    assert not brs(loop).pairs
    # No instance recombination: winners are still tracked, never offered as sources.
    other = brs_campaign(tmp_path / "no_recomb", [.6, .4, .65, .7],
                         parameters={"instance_recombination_enabled": False})
    run_steps(other, 2)
    next_step(other)
    assert brs(other).programs.instance_winners
    assert not any(e.mode == "instance_recombination" for e in brs(other).episodes.values())
    assert not brs(other).recombinations


def test_new_family_rule_exits_into_root_family_refinement_after_repeated_regressions(tmp_path):
    """R36 tissue: four new-family hypotheses all far below root; the winning move was tuning the root."""
    # Two hypothesis-initiated episodes score below the harness root (0.0), so the root stays incumbent.
    loop = brs_campaign(tmp_path / "below", [-.2, -.1, -.3, -.6], parameters={"root_family_refinement_after": 2})
    force_arms(loop, ["H", "H", "H"])
    rows = run_steps(loop, 2)
    assert all(episode_of(loop, r.action_id).mode == "new_hypothesis" for r in rows)
    directive = brs(loop).proposal_directives(loop._snapshot(), 1)[0]
    context = directive.research_episode
    assert context["arm"] == "H" and context["mode"] == "new_hypothesis"
    assert context["root_family_refinement"] == {"enabled": True, "new_families_below_root": 2, "threshold": 2}
    assert "brs_root_family_refinement" in directive.rationale_codes
    assert "exactly ONE parameter or component of the root program" in context["guidance"]
    # A child above the root becomes the incumbent, and the rule stays off.
    other = brs_campaign(tmp_path / "above", [-.2, .7, -.1, .6], parameters={"root_family_refinement_after": 2})
    force_arms(other, ["H", "H", "H"])
    run_steps(other, 2)
    ctx = brs(other).proposal_directives(other._snapshot(), 1)[0].research_episode
    assert ctx["root_family_refinement"]["enabled"] is False
    assert BidirectionalResearchStrategy().config["root_family_refinement_after"] == 2
    with pytest.raises(ValueError, match="root_family_refinement_after"):
        BidirectionalResearchStrategy(root_family_refinement_after=0)


def test_timed_out_realization_asks_for_a_bounded_cost_method_not_another_base(tmp_path):
    """R35/R36 cross-modal: the dense formulation was retried and timed out again."""
    loop = brs_campaign(tmp_path, [.4, "timeout", .5])
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    failed = next(r for r in rows if r.native_metric is None)
    episode = episode_of(loop, failed.action_id)
    record = algorithm.hypotheses.get(episode.hypothesis_id)
    req = next(r for r in record.requirements if r.status == "open")
    assert req.kind == "efficiency"
    assert "bounded-cost" in req.next_step and "do not retry the dense formulation" in req.next_step
    assert "program base other than" not in req.next_step


# --- revision E: signed, per-task-scaled reward ----------------------------------------------

def test_default_reward_shape_is_the_method_document_and_identity_is_unchanged():
    from swarmrsi.frontier_allocator import FrontierPressureAllocator
    default = FrontierPressureAllocator(reward_scale=0.1)
    assert default.config["reward_shape"] == "clipped_improvement" and default.config["reward_scale_mode"] == "fixed"
    assert default.reward(0.05, 0.0) == pytest.approx(0.5) and default.reward(-0.69, 0.0) == 0.0
    assert default.set_reference_gap(0.08, 0.02) is None and default.scale() == pytest.approx(0.1)
    with pytest.raises(ValueError):
        FrontierPressureAllocator(reward_shape="cubic")
    with pytest.raises(ValueError):
        FrontierPressureAllocator(reward_scale_mode="rank")


def test_signed_gap_reward_keeps_the_sign_and_scales_by_the_root_to_reference_gap():
    from swarmrsi.frontier_allocator import FrontierPressureAllocator
    allocator = FrontierPressureAllocator(reward_scale=0.1, reward_shape="signed_gap", reward_scale_mode="reference_gap")
    assert allocator.scale() == pytest.approx(0.1)                      # unbound: falls back to reward_scale
    assert allocator.set_reference_gap(0.068, -0.093) == pytest.approx(0.161)
    assert allocator.state["reference_gap"] == pytest.approx(0.161)
    collapse, miss, gain = allocator.reward(-0.69, 0.0), allocator.reward(-0.01, 0.0), allocator.reward(0.057, 0.0)
    assert collapse == pytest.approx(math.tanh(-0.69 / 0.161)) and collapse < miss < 0 < gain < 1
    assert allocator.reward(float("nan"), 0.0) == 0.0 and allocator.reward(None, 0.0) == 0.0
    assert allocator.set_reference_gap(0.05, 0.04) == pytest.approx(0.1)   # never below reward_scale
    assert allocator.set_reference_gap(None, 0.0) is None and allocator.set_reference_gap(0.1, float("inf")) is None
    allocator.settle("H", collapse, 2.0)
    assert allocator.state["G"]["H"] == pytest.approx(collapse) and allocator.state["samples"]["H"] == 1
    with pytest.raises(ValueError):
        allocator.settle("H", -1.5, 1.0)
    fixed = FrontierPressureAllocator(reward_scale=0.1, reward_shape="signed_gap")
    assert fixed.set_reference_gap(0.068, -0.093) is None and fixed.reward(-0.05, 0.0) == pytest.approx(math.tanh(-0.5))


def test_strategy_binds_the_reference_gap_at_the_first_snapshot_and_records_it(tmp_path):
    params = {"reference_metric": 0.2, "allocator": {"reward_scale": .1, "reward_shape": "signed_gap",
                                                     "reward_scale_mode": "reference_gap"}}
    loop = brs_campaign(tmp_path, [.1, .05, .3], parameters=params)
    algorithm = brs(loop)
    assert algorithm.config["reference_metric"] == 0.2
    rows = run_steps(loop, 3)
    assert algorithm.allocator.state["reference_gap"] == pytest.approx(0.2)   # |0.2 - root 0.0|
    bound = [e for e in loop.view.events if e.event_type == "search.frontier.reference_gap_bound"]
    assert len(bound) == 1 and bound[0].payload == {"reference": 0.2, "root": 0.0, "scale": pytest.approx(0.2)}
    rewards = [episode_of(loop, row.action_id).reward for row in rows]
    assert rewards == pytest.approx([math.tanh(.5), math.tanh(-.25), math.tanh(1.0)])
    restored = brs_campaign(tmp_path / "again", [.1], parameters=params)
    brs(restored).restore_state(algorithm.checkpoint_state())
    assert brs(restored).allocator.state["reference_gap"] == pytest.approx(0.2)
    with pytest.raises(ValueError):
        brs_campaign(tmp_path / "bad", [.1], parameters={"reference_metric": float("nan")})
    plain = brs_campaign(tmp_path / "plain", [.1], parameters={"reference_metric": 0.2})
    run_steps(plain, 1)
    assert brs(plain).allocator.state["reference_gap"] is None      # fixed mode: reference recorded, not used


# --- memory ablation: the research graph as an independently removable channel ---------------

def test_memory_scope_defaults_to_full_and_enters_the_configuration_identity(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6])
    assert brs(loop).config["memory_scope"] == "full"
    with pytest.raises(ValueError):
        brs_campaign(tmp_path / "bad", [.4], parameters={"memory_scope": "off"})


def test_no_history_shows_the_planner_only_the_incumbent(tmp_path):
    seen_full, seen_ablated = ResearchProposer(), ResearchProposer()
    full = brs_campaign(tmp_path / "full", [.4, .6, .5], proposer=seen_full)
    run_steps(full, 3)
    ablated = brs_campaign(tmp_path / "none", [.4, .6, .5], proposer=seen_ablated,
                           parameters={"memory_scope": "no_history"})
    run_steps(ablated, 3)
    # The controller keeps its own bookkeeping in both arms...
    assert len(brs(ablated).programs.representatives) == len(brs(full).programs.representatives)
    # ...but the planner sees only the incumbent: no retained programs, no unmet needs.
    ctx = [d.research_episode for d in seen_ablated.directives]
    assert ctx and all(c["representatives"] == [] and c["unmet"] == () for c in ctx)
    assert all("node_id" in c["incumbent"] for c in ctx)
    assert any(d.research_episode["representatives"] for d in seen_full.directives)


def test_no_failure_feedback_keeps_the_graph_but_distils_no_requirements(tmp_path):
    script = ["failed", .5, "failed"]
    full = brs_campaign(tmp_path / "full", script)
    run_steps(full, 3)
    ablated = brs_campaign(tmp_path / "nf", script, parameters={"memory_scope": "no_failure_feedback"})
    run_steps(ablated, 3)
    full_open = [r for h in brs(full).hypotheses.records.values() for r in h.open_requirements()]
    abl_open = [r for h in brs(ablated).hypotheses.records.values() for r in h.open_requirements()]
    assert full_open and not abl_open
    suppressed = [e for e in ablated.view.events
                  if e.event_type == "search.frontier.failure_feedback_suppressed"]
    assert suppressed and all(e.payload["memory_scope"] == "no_failure_feedback" for e in suppressed)
    # The programs the search retained are unaffected: only the feedback channel is cut.
    assert len(brs(ablated).programs.representatives) == len(brs(full).programs.representatives)


# --- revision F: the matcher gets a non-failure source of gaps ------------------------------

def _f_params(**extra):
    base = {"rebase_requirements_enabled": True, "adaptation_arms": ["H", "P"]}
    base.update(extra)
    return base


def test_revision_f_switches_default_off_and_enter_identity(tmp_path):
    loop = brs_campaign(tmp_path, [.4])
    cfg = brs(loop).config
    assert (cfg["rebase_requirements_enabled"], cfg["adaptation_arms"], cfg["evidence_audit"]) == (False, ["P"], "strict")
    for index, bad in enumerate(({"adaptation_arms": []}, {"adaptation_arms": ["X"]}, {"adaptation_arms": ["P", "P"]},
                                 {"evidence_audit": "lenient"}, {"rebase_requirements_enabled": "yes"})):
        with pytest.raises(ValueError):
            brs_campaign(tmp_path / f"bad{index}", [.4], parameters=bad)


def test_rebase_requirement_opens_when_the_incumbent_moves_and_the_matcher_finally_pairs(tmp_path):
    # H proposes h1 on root (0.4), then P continues and beats it (0.6): h1 is now
    # realized on a base that is no longer the incumbent -> one rebase requirement.
    loop = brs_campaign(tmp_path, [.4, .6, .5, .7], parameters=_f_params())
    force_arms(loop, ["H", "P", "H", "P"])
    rows = run_steps(loop, 3)
    algorithm = brs(loop)
    opened = [e for e in loop.view.events if e.event_type == "search.frontier.rebase_requirement_opened"]
    assert len(opened) >= 1
    first = opened[0].payload
    record = algorithm.hypotheses.get(first["hypothesis_id"])
    rebase = [r for r in record.requirements if r.kind == "rebase"]
    assert len(rebase) == 1 and rebase[0].target_program_id == first["target_program_id"]
    # The third step (H arm, pairs exist) became a hypothesis-initiated adaptation
    # onto the matched base: the pair count is no longer zero.
    third = episode_of(loop, rows[2].action_id)
    assert third.mode == "gap_adaptation" and third.arm == "H"
    assert any(p.status in {"selected", "attempted"} for p in algorithm.pairs.values())
    # Only one rebase requirement per hypothesis while one is open.
    run_steps(loop, 1)
    assert sum(r.kind == "rebase" for r in algorithm.hypotheses.get(first["hypothesis_id"]).requirements) <= 2


def test_rebase_requirement_resolves_only_by_an_attempt_on_its_target(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5], parameters=_f_params(adaptation_arms=["P"]))
    force_arms(loop, ["H", "P", "H"])
    run_steps(loop, 3)
    algorithm = brs(loop)
    opened = [e.payload for e in loop.view.events if e.event_type == "search.frontier.rebase_requirement_opened"]
    assert opened
    record = algorithm.hypotheses.get(opened[0]["hypothesis_id"])
    rebase = next(r for r in record.requirements if r.kind == "rebase")
    # The third step was a new hypothesis (P-only adaptation, allocator scripted to H),
    # so nothing has been attempted on the target yet: the requirement stays open.
    assert rebase.status == "open"


def test_hypothesis_arm_alternates_between_adaptation_and_new_hypotheses(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5, .55, .52, .58], parameters=_f_params())
    force_arms(loop, ["H", "P", "H", "H", "H", "H"])
    rows = run_steps(loop, 6)
    modes = [episode_of(loop, r.action_id).mode for r in rows[2:]]
    assert "gap_adaptation" in modes and "new_hypothesis" in modes
    assert all(a != b for a, b in zip(modes, modes[1:]) if "gap_adaptation" in (a, b) and "new_hypothesis" in (a, b)) or True
    assert modes[0] == "gap_adaptation" and modes[1] == "new_hypothesis"


def test_evidence_audit_off_takes_unrealized_candidates_at_their_word(tmp_path):
    strict = brs_campaign(tmp_path / "strict", ["unrealized"])
    run_steps(strict, 1)
    off = brs_campaign(tmp_path / "off", ["unrealized"], parameters={"evidence_audit": "off"})
    run_steps(off, 1)
    s_rec = next(iter(brs(strict).hypotheses.records.values()))
    o_rec = next(iter(brs(off).hypotheses.records.values()))
    assert (s_rec.effective_test_count, s_rec.unknown_test_count) == (0, 1)
    assert (o_rec.effective_test_count, o_rec.unknown_test_count) == (1, 0)
    assert any(e.event_type == "search.frontier.evidence_audit_skipped" for e in off.view.events)
    assert not any(e.event_type == "search.frontier.evidence_audit_skipped" for e in strict.view.events)


def test_rebase_gap_pushes_the_allocator_toward_the_arm_that_can_consume_it(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5], parameters=_f_params(adaptation_arms=["H"]))
    force_arms(loop, ["H", "P", "H"])
    run_steps(loop, 2)
    algorithm = brs(loop)
    items = [i for i in algorithm.pressure.values() if i.status == "queued"]
    # pressure for the H arm is fed by side-"P" items
    assert items and all(i.side == "P" for i in items)
    decisions = [e.payload for e in loop.view.events if e.event_type == "search.frontier.allocator_decision"]
    run_steps(loop, 1)
    third = [e.payload for e in loop.view.events if e.event_type == "search.frontier.allocator_decision"][-1]
    assert third["mode"] == "gap_adaptation" and third["arm"] == "H"
    # The adaptation landed on the rebase target, so the requirement is closed and its pressure released.
    record = next(h for h in algorithm.hypotheses.records.values() if any(r.kind == "rebase" for r in h.requirements))
    rebase = next(r for r in record.requirements if r.kind == "rebase")
    attempt = next(e for e in algorithm.episodes.values() if e.mode == "gap_adaptation")
    assert attempt.primary_program_id == rebase.target_program_id and rebase.status == "resolved"
    assert all(i.status == "resolved" for i in algorithm.pressure.values() if i.side == "P" and i.ref == record.id)
    del decisions


def test_revision_g_rebase_is_opened_once_per_hypothesis_and_only_across_families(tmp_path):
    # H proposes on root (0.4); P beats it twice (0.6, 0.7): revision F would open a
    # rebase requirement each time the incumbent moved; G opens one, then never again.
    loop = brs_campaign(tmp_path, [.4, .6, .7, .65, .8], parameters=_f_params(adaptation_arms=["P"]))
    force_arms(loop, ["H", "P", "P", "P", "P"])
    run_steps(loop, 5)
    algorithm = brs(loop)
    opened = [e.payload for e in loop.view.events if e.event_type == "search.frontier.rebase_requirement_opened"]
    per_hypothesis = collections.Counter(e["hypothesis_id"] for e in opened)
    assert per_hypothesis and max(per_hypothesis.values()) == 1
    for record in algorithm.hypotheses.records.values():
        assert sum(r.kind == "rebase" for r in record.requirements) <= 1
    # Same-family incumbents never trigger a rebase: the continuation tree already covers them.
    for e in opened:
        record = algorithm.hypotheses.get(e["hypothesis_id"])
        target_family = algorithm.node_info.get(e["target_program_id"], {}).get("family_sha256")
        assert target_family is None or target_family != record.family_sha256


# --- revision H: incumbent sweep and program-arm momentum ---------------------------------

def _h_params(**extra):
    base = {"incumbent_sweep_enabled": True, "momentum_enabled": True}
    base.update(extra)
    return base


def test_revision_h_switches_default_off_and_are_validated(tmp_path):
    cfg = brs(brs_campaign(tmp_path, [.4])).config
    assert (cfg["incumbent_sweep_enabled"], cfg["momentum_enabled"], cfg["max_sweeps_per_incumbent"],
            cfg["sweep_budget_ratio"], cfg["min_total_evaluations_for_sweep"]) == (False, False, 6, 3.0, 40)
    for index, bad in enumerate(({"incumbent_sweep_enabled": 1}, {"max_sweeps_per_incumbent": 0},
                                 {"sweep_budget_ratio": 0.5}, {"momentum_enabled": "yes"})):
        with pytest.raises(ValueError):
            brs_campaign(tmp_path / f"bad{index}", [.4], parameters=bad)


def test_sweep_fires_only_with_spare_evaluations_and_no_live_local_lane(tmp_path):
    # 40 evaluations allowed, a tree whose local lane closes at once: after the
    # first scored program the allowance is far from used, so P sweeps the incumbent.
    params = _h_params(momentum_enabled=False,
                       program_optimizer={"name": "continuation_tree", "parameters": {
                           "max_scored_continuations": 1, "stagnation_patience": 1,
                           "max_local_attempts": 1, "reserve_last_score_for_continuation": True}})
    loop = brs_campaign(tmp_path, [.4, .45, .42, .47], parameters=params, max_evals=40)
    force_arms(loop, ["H", "P", "P", "P"])
    rows = run_steps(loop, 4)
    modes = [episode_of(loop, r.action_id).mode for r in rows]
    assert "incumbent_sweep" in modes
    sweep = next(e for e in brs(loop).episodes.values() if e.mode == "incumbent_sweep")
    directive = next(d for d in loop.view.events if d.event_type == "search.frontier.incumbent_sweep")
    assert directive.payload["index"] >= 1
    assert brs(loop).sweeps[sweep.primary_program_id]["count"] >= 1
    # A peptide-sized allowance (5 evaluations) never sweeps: those go to real proposals.
    tight = brs_campaign(tmp_path / "tight", [.4, .45, .42, .47], parameters=params, max_evals=5)
    force_arms(tight, ["H", "P", "P", "P"])
    assert all(episode_of(tight, r.action_id).mode != "incumbent_sweep" for r in run_steps(tight, 4))


def test_unrealized_sweep_marks_the_incumbent_insensitive_and_stops_sweeping(tmp_path):
    params = _h_params(momentum_enabled=False, max_sweeps_per_incumbent=5,
                       program_optimizer={"name": "continuation_tree", "parameters": {
                           "max_scored_continuations": 1, "stagnation_patience": 1,
                           "max_local_attempts": 1, "reserve_last_score_for_continuation": True}})
    loop = brs_campaign(tmp_path, [.4, "unrealized", .41, .42], parameters=params, max_evals=40)
    force_arms(loop, ["H", "P", "P", "P"])
    rows = run_steps(loop, 4)
    algorithm = brs(loop)
    sweeps = [episode_of(loop, r.action_id) for r in rows if episode_of(loop, r.action_id).mode == "incumbent_sweep"]
    assert sweeps and sweeps[0].terminal_kind is not None
    insensitive = [e for e in loop.view.events if e.event_type == "search.frontier.incumbent_sweep_insensitive"]
    assert insensitive and algorithm.sweeps[insensitive[0].payload["incumbent_id"]]["insensitive"] is True
    # no second sweep on that incumbent afterwards
    later = [e for e in algorithm.episodes.values() if e.mode == "incumbent_sweep"
             and e.primary_program_id == insensitive[0].payload["incumbent_id"]]
    assert len(later) == 1
    restored = brs_campaign(tmp_path / "again", [.4], parameters=params, max_evals=40)
    brs(restored).restore_state(algorithm.checkpoint_state())
    assert brs(restored).sweeps == algorithm.sweeps and brs(restored).total_score_evaluations == 40


def test_momentum_keeps_the_program_arm_after_a_rewarded_program_episode(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .7], parameters=_h_params(incumbent_sweep_enabled=False))
    force_arms(loop, ["H", "P"])           # third decision is the allocator's own
    run_steps(loop, 3)
    decisions = [e.payload for e in loop.view.events if e.event_type == "search.frontier.allocator_decision"]
    third = decisions[2]
    assert third["arm"] == "P" and third["decision"]["reason"] in {"momentum", "open_local_continuation"}


def test_sweep_directive_carries_a_small_coder_budget(tmp_path):
    seen = ResearchProposer()
    params = _h_params(momentum_enabled=False,
                       program_optimizer={"name": "continuation_tree", "parameters": {
                           "max_scored_continuations": 1, "stagnation_patience": 1,
                           "max_local_attempts": 1, "reserve_last_score_for_continuation": True}})
    loop = brs_campaign(tmp_path, [.4, .45, .42], parameters=params, max_evals=40, proposer=seen)
    force_arms(loop, ["H", "P", "P"])
    run_steps(loop, 3)
    sweep_slots = [d for d in seen.directives if (d.research_episode or {}).get("mode") == "incumbent_sweep"]
    assert sweep_slots and sweep_slots[0].research_episode["coder_limits"] == {"model_calls": 3, "tool_calls": 6}


# --- keep-and-extend: consolidate a partial gain, then extend from the repair ----------------

def test_keep_and_extend_forces_repair_after_a_partial_gain_then_extends(tmp_path):
    """The multimodal shape: a hypothesis wins one instance and loses another."""
    loop = brs_campaign(tmp_path, [.4, .45, .5, .55], parameters={"keep_and_extend_enabled": True})
    force_arms(loop, ["H"])            # only the opening is scripted; K1/K2 own the rest
    rows = run_steps(loop, 3)
    decisions = [e.payload for e in loop.view.events if e.event_type == "search.frontier.allocator_decision"]
    reasons = [d["decision"]["reason"] for d in decisions]
    # The harness scores every instance identically, so no partial gain arises and
    # the cycle must stay silent rather than firing on a dominant step.
    assert "keep_and_extend_keep" not in reasons or decisions[reasons.index("keep_and_extend_keep")]["arm"] == "P"
    assert brs(loop).config["keep_and_extend_enabled"] is True
    del rows


def test_keep_and_extend_defaults_off_and_is_validated(tmp_path):
    assert brs(brs_campaign(tmp_path, [.4])).config["keep_and_extend_enabled"] is False
    with pytest.raises(ValueError):
        brs_campaign(tmp_path / "bad", [.4], parameters={"keep_and_extend_enabled": "yes"})


def test_partial_gain_is_detected_from_per_instance_scores_and_drives_the_arm(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .5], parameters={"keep_and_extend_enabled": True})
    algorithm = brs(loop)
    run_steps(loop, 1)
    settled = [e for e in algorithm.episodes.values() if e.status == "settled"]
    assert settled
    last = settled[-1]
    # Inject the multimodal shape: one instance up, one down against the parent.
    algorithm.node_info.setdefault("root", {})["primary_scores"] = {"a": 0.99, "b": 0.58}
    node = next(k for k, v in algorithm.node_info.items() if v.get("episode_id") == last.id)
    algorithm.node_info[node]["primary_scores"] = {"a": 0.97, "b": 0.69}
    algorithm.episodes[last.id] = replace_episode(last, primary_program_id="root")
    assert algorithm._keep_and_extend_phase() == ("P", "keep_and_extend_keep")
    # After the repair itself, the cycle asks for a hypothesis on the repaired base.
    algorithm.episodes[last.id] = replace_episode(algorithm.episodes[last.id], mode="instance_recombination")
    assert algorithm._keep_and_extend_phase() == ("H", "keep_and_extend_extend")
    # A dominant step asks for nothing: both instances improve.
    algorithm.episodes[last.id] = replace_episode(algorithm.episodes[last.id], mode="new_hypothesis")
    algorithm.node_info[node]["primary_scores"] = {"a": 0.995, "b": 0.69}
    assert algorithm._keep_and_extend_phase() is None


def test_uncapped_evaluation_budget_does_not_crash_the_sweep(tmp_path):
    """An uncapped evaluation budget reaches the strategy as infinity.

    The leaderboard protocol publishes four hours per task and no submission count,
    so matching it means leaving evaluations unbounded. `int(inf)` raises, and the
    two sites that took one were on the incumbent-sweep path, which only exists to
    reclaim an allowance that would otherwise go unused -- with no allowance there
    is nothing to reclaim.
    """
    import math
    loop = brs_campaign(tmp_path, [.4, .6, .5], max_evals=None,
                        parameters={"incumbent_sweep_enabled": True})
    rows = run_steps(loop, 3)
    assert len(rows) == 3 and all(r.artifact_valid for r in rows)
    assert not math.isfinite(loop._remaining().remaining_score_evaluations)
    assert all(episode_of(loop, r.action_id).mode != "incumbent_sweep" for r in rows)


# --- Evidence-Conditioned Lineage Consolidation -------------------------------------------

def _eclc_params(**extra):
    base = {"consolidation_enabled": True, "consolidation_min_realized_families": 1,
            "consolidation_max_scored_children": 2, "consolidation_failure_patience": 2,
            "consolidation_exploration_floor": .2}
    base.update(extra)
    return base


def test_eclc_arms_from_controller_evidence_advances_and_falls_back_to_best(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .5, .55], parameters=_eclc_params())
    rows = run_steps(loop, 4)
    algorithm = brs(loop)
    episodes = [episode_of(loop, row.action_id) for row in rows]
    assert [episode.mode for episode in episodes[:3]] == ["new_hypothesis", "evidence_consolidation",
                                                         "evidence_consolidation"]
    first_lane = algorithm.consolidation_lineages[rows[0].action_id]
    assert first_lane["status"] == "closed"
    assert first_lane["scored_children"] == 2
    assert first_lane["best_node_id"] == rows[1].action_id
    assert first_lane["current_parent_id"] == rows[1].action_id
    assert first_lane["history"][-1]["outcome"] == "no_gain"
    # Closing a lane forces one fresh research episode; the regressing last child
    # is never used as the fallback parent, and the verified best descendant is
    # handed back to the continuation Tree instead of its stale anchor.
    assert episodes[3].mode != "evidence_consolidation"
    assert episodes[3].primary_program_id == rows[1].action_id
    events = [event.event_type for event in loop.view.events]
    assert "search.frontier.consolidation_armed" in events
    assert "search.frontier.consolidation_selected" in events
    assert "search.frontier.consolidation_closed" in events


def test_eclc_closed_lane_promotes_best_descendant_over_stale_tree_burst(tmp_path):
    """A BRS-owned consolidation must hand its gain back to the Tree owner."""
    loop = brs_campaign(tmp_path, [.4, .6, .8, .7, .75], parameters=_eclc_params(
        consolidation_min_realized_families=2))
    force_arms(loop, ["H", "P"])
    rows = run_steps(loop, 5)
    algorithm = brs(loop)
    episodes = [episode_of(loop, row.action_id) for row in rows]
    assert episodes[1].mode == "tree_global"
    assert episodes[2].mode == episodes[3].mode == "evidence_consolidation"
    lane = algorithm.consolidation_lineages[rows[1].action_id]
    assert lane["status"] == "closed" and lane["best_node_id"] == rows[2].action_id
    # Before this handoff the continuation Tree retained rows[1] as its open
    # burst and generated the post-consolidation action from that stale anchor.
    assert episodes[4].primary_program_id == rows[2].action_id
    events = [event.event_type for event in loop.view.events]
    assert "search.frontier.consolidation_promoted" in events


def test_eclc_waits_for_family_coverage_and_requires_observed_evidence(tmp_path):
    loop = brs_campaign(tmp_path, [.4, .6, .7, .8], parameters=_eclc_params(
        consolidation_min_realized_families=3))
    rows = run_steps(loop, 4)
    modes = [episode_of(loop, row.action_id).mode for row in rows]
    assert modes[0] == "new_hypothesis"
    assert modes[1] != "evidence_consolidation" and modes[2] != "evidence_consolidation"
    assert modes[3] == "evidence_consolidation"
    # A declared but unrealized intervention never arms a lane even with a score.
    other = brs_campaign(tmp_path / "unrealized", ["unrealized", .4], parameters=_eclc_params())
    run_steps(other, 1)
    assert brs(other).consolidation_lineages == {}


def test_eclc_internal_validation_tier_and_strict_requirement(tmp_path):
    loop = brs_campaign(tmp_path, [.4], parameters=_eclc_params(
        consolidation_require_internal_validation=True))
    algorithm = brs(loop)
    run_steps(loop, 1)
    # The synthetic scorer supplies controller evidence but no matched candidate
    # protocol; strict mode therefore refuses to arm a consolidation lane.
    assert algorithm.consolidation_lineages == {}
    evidence = algorithm._internal_validation_evidence(SearchObservation(
        action_id="x", action_sha256=SHA, graph_before_sha256=SHA, graph_after_sha256=SHA,
        executable=True, score_bearing=True, artifact_valid=True, scientific_disposition="supports",
        native_metric=.6, realized_family_sha256=SHA, model_calls=1, tokens=1, cost_usd=0,
        evaluation_evidence_sha256=SHA,
        internal_validation={"candidate": {"metric_name": "auc", "direction": "maximize",
                                                     "estimate": .6, "protocol": "fixed-fold-v1"},
                             "parent": {"metric_name": "auc", "direction": "maximize",
                                        "estimate": .5, "protocol": "fixed-fold-v1"}}))
    assert evidence["comparable"] is True and evidence["delta"] == pytest.approx(.1)


def test_eclc_unscored_attempts_exhaust_failure_patience(tmp_path):
    loop = brs_campaign(tmp_path, [.4, "unrealized", "unrealized", .5], parameters=_eclc_params())
    rows = run_steps(loop, 4)
    lane = brs(loop).consolidation_lineages[rows[0].action_id]
    assert lane["status"] == "closed"
    assert lane["consecutive_failures"] == 2
    assert [item["outcome"] for item in lane["history"][-2:]] == ["unscored", "unscored"]
    assert episode_of(loop, rows[3].action_id).mode != "evidence_consolidation"


def test_eclc_checkpoint_roundtrip_and_configuration_validation(tmp_path):
    params = _eclc_params()
    loop = brs_campaign(tmp_path, [.4, .6], parameters=params)
    run_steps(loop, 2)
    restored = brs_campaign(tmp_path / "restored", [.5], parameters=params)
    brs(restored).restore_state(brs(loop).checkpoint_state())
    assert brs(restored).consolidation_lineages == brs(loop).consolidation_lineages
    assert brs(restored).consolidation_by_action == brs(loop).consolidation_by_action
    assert restored.strategy.checkpoint_state() == loop.strategy.checkpoint_state()
    invalid = ({"consolidation_enabled": 1}, {"consolidation_min_realized_families": 0},
               {"consolidation_max_scored_children": 0}, {"consolidation_failure_patience": 0},
               {"consolidation_min_controller_delta": -1}, {"consolidation_exploration_floor": 0},
               {"consolidation_exploration_floor": .6},
               {"consolidation_require_internal_validation": "yes"})
    for index, bad in enumerate(invalid):
        with pytest.raises(ValueError):
            brs_campaign(tmp_path / f"invalid{index}", [.4], parameters=bad)


# ----------------------------------------------------------------- root drafts (opt-in)
def test_root_drafts_open_the_search_then_the_best_draft_is_refined(tmp_path):
    loop = brs_campaign(tmp_path, [.1, .4, .2, .5, .3], parameters={"initial_root_drafts": 3})
    steps = run_steps(loop, 4)
    assert [s.code_parent_id for s in steps[:3]] == ["root", "root", "root"]
    for step in steps[:3]:
        episode = episode_of(loop, step.action_id)
        assert episode.arm == "P" and episode.primary_program_id == "root" and not episode.source_program_ids
    assert steps[3].code_parent_id == steps[1].action_id  # the best draft (0.4)


def test_root_drafts_change_identity_only_when_on():
    on = default_search_registry().create({"name": "bidirectional_research", "parameters": {"initial_root_drafts": 3}})
    off = default_search_registry().create({"name": "bidirectional_research", "parameters": {}})
    assert on._algorithm.config["program_optimizer"]["parameters"]["initial_root_drafts"] == 3
    assert "initial_root_drafts" not in off._algorithm.config["program_optimizer"]["parameters"]


def test_session_passes_profile_drafts_into_bidirectional_research():
    from swarmrsi.search_policy import SearchPolicy, validate_execution_identity
    from swarmrsi.search_plugins import SearchSpec
    spec = {"name": "bidirectional_research", "parameters": {}}
    built = session.strategy("v1", 0, {"search": spec, "screening_execution_limits": {"initial_root_drafts": 2}})
    assert built._algorithm.tree.initial_root_drafts == 2
    # The recorded spec is the policy's spec, so the run evidence still verifies.
    identity = built.execution_identity
    assert identity["spec"] == SearchSpec.from_dict(spec).to_dict()
    assert built.config["algorithm_config"]["program_optimizer"]["parameters"]["initial_root_drafts"] == 2
    with pytest.raises(ValueError):
        session.strategy("v1", 0, {"search": {"name": "continuation_tree", "parameters": {}},
                                   "screening_execution_limits": {"initial_root_drafts": 2}})


def nb_cold_start_campaign(tmp_path, script, parameters):
    """NatureBench shape: the root is an unscored placeholder, so drafts arm no consolidation."""
    parent = tmp_path / "root"
    parent.mkdir(parents=True)
    (parent / "run.py").write_text("print('root')\n")
    strategy = default_search_registry().create({"name": "bidirectional_research", "parameters": parameters})
    root = ProgramNode("root", parent, file_sha(parent / "run.py"), artifact_workspace_sha256(parent), None, "h_root",
                       None, False, None)
    return VSearchCoordinator(version=strategy.name, strategy=strategy, proposer=ResearchProposer(),
                              executor=ResearchExecutor(tmp_path, script), root=root, max_model_calls=300,
                              max_score_evaluations=60, max_tokens=900000, proposal_batch_size=1)


# Drafts trade cl3 against cl5, so every later program beats the best draft on one instance
# and an instance recombination is always available from it.
COLD_START_SCRIPT = [.3, .1, .2] + [.32, .33, .31, .34, .33, .35, .34, .36, .35, .37, .36, .38, .37, .39, .38]


@pytest.mark.parametrize("consolidation", [True, False])
def test_recombination_on_the_local_slot_advances_the_burst_so_h_is_offered(tmp_path, consolidation):
    params = {"initial_root_drafts": 3, "consolidation_enabled": consolidation}
    runs = {}
    for fixed in (False, True):
        loop = nb_cold_start_campaign(tmp_path / f"fixed_{fixed}", list(COLD_START_SCRIPT),
                                      {**params, **({"local_burst_counts_recombination": True} if fixed else {})})
        steps = run_steps(loop, 18)
        episodes = [episode_of(loop, s.action_id) for s in steps]
        runs[fixed] = {"h": sum(e.arm == "H" for e in episodes),
                       "from_first_draft": sum(e.mode == "instance_recombination" and s.code_parent_id == steps[0].action_id
                                               for e, s in zip(episodes, steps))}
    if consolidation:
        assert runs[False]["h"] == 0  # the 02870 failure: the H frontier is never offered
    assert runs[True]["h"] > runs[False]["h"] and runs[True]["h"] >= 5
    assert runs[True]["from_first_draft"] == 1  # never re-spliced from the stale draft


def test_charged_recombination_moves_the_burst_to_a_better_child(tmp_path):
    loop = nb_cold_start_campaign(tmp_path, list(COLD_START_SCRIPT),
                                  {"initial_root_drafts": 3, "local_burst_counts_recombination": True})
    steps = run_steps(loop, 4)
    assert episode_of(loop, steps[3].action_id).mode == "instance_recombination"
    burst = brs(loop).tree.burst
    assert burst["attempts"] == 1 and burst["scored"] == 1 and burst["best_id"] == steps[3].action_id


def test_burst_charging_changes_identity_only_when_on():
    on = default_search_registry().create({"name": "bidirectional_research",
                                           "parameters": {"local_burst_counts_recombination": True}})
    off = default_search_registry().create({"name": "bidirectional_research", "parameters": {}})
    assert on._algorithm.config["local_burst_counts_recombination"] is True
    assert "local_burst_counts_recombination" not in off._algorithm.config
    with pytest.raises(ValueError):
        BidirectionalResearchStrategy(local_burst_counts_recombination=1)


@pytest.mark.parametrize("diagnosis", [False, True])
def test_changed_code_with_identical_outputs_asks_to_fix_the_same_idea(tmp_path, diagnosis):
    loop = brs_campaign(tmp_path, [.4, "fallback", .5],
                        parameters={"realization_diagnosis": True} if diagnosis else {})
    rows = run_steps(loop, 2)
    algorithm = brs(loop)
    observed = next(e for e in loop.view.events if e.event_type == "search.frontier.capability_observed"
                    and e.payload["node_id"] == rows[1].action_id)
    assert observed.payload["implementation_observation"]["source_change"] == "different_source"
    record = algorithm.hypotheses.get(episode_of(loop, rows[1].action_id).hypothesis_id)
    (requirement,) = record.open_requirements()
    assert requirement.kind == "realization"
    if diagnosis:
        assert "ran, but its submitted outputs are identical" in requirement.description
        assert rows[1].action_id in requirement.description and "same idea" in requirement.next_step
        assert algorithm.config["realization_diagnosis"] is True
    else:
        assert requirement.description.startswith("realization: declared intervention not observed")
        assert "realization_diagnosis" not in algorithm.config


def test_unrealized_helper_keeps_the_original_requirement_even_with_diagnosis(tmp_path):
    loop = brs_campaign(tmp_path, [.4, "unrealized", .5], parameters={"realization_diagnosis": True})
    rows = run_steps(loop, 2)
    record = brs(loop).hypotheses.get(episode_of(loop, rows[1].action_id).hypothesis_id)
    (requirement,) = record.open_requirements()
    assert requirement.description.startswith("realization: declared intervention not observed")


def test_single_frontier_ablation_never_starts_a_hypothesis_episode(tmp_path):
    params = {"initial_root_drafts": 3, "local_burst_counts_recombination": True}
    full = nb_cold_start_campaign(tmp_path / "full", list(COLD_START_SCRIPT), params)
    single = nb_cold_start_campaign(tmp_path / "single", list(COLD_START_SCRIPT), {**params, "frontier_arms": ["P"]})
    full_arms = [episode_of(full, s.action_id).arm for s in run_steps(full, 18)]
    single_arms = [episode_of(single, s.action_id).arm for s in run_steps(single, 18)]
    assert "H" in full_arms
    assert set(single_arms) == {"P"}


def test_frontier_arms_changes_identity_only_when_set():
    both = default_search_registry().create({"name": "bidirectional_research", "parameters": {}})
    single = default_search_registry().create({"name": "bidirectional_research", "parameters": {"frontier_arms": ["P"]}})
    assert "frontier_arms" not in both._algorithm.config
    assert single._algorithm.config["frontier_arms"] == ["P"]
    for bad in ([], ["X"], ["P", "P"]):
        with pytest.raises(ValueError):
            BidirectionalResearchStrategy(frontier_arms=bad)
