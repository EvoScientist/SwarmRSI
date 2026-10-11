"""Behavioral replay, not benchmark scores or proof of future superiority."""
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import random

import pytest

from evolvcoder.strategies import ContinuationTreeStrategy, _uct_node_score
from evolvcoder.tree_policy import PolicyTreeStrategy, TreePolicy
from test_evolvcoder_incremental_feedback import campaign, next_step


def test_baseline_source_is_preserved_byte_for_byte():
    path = Path(__file__).resolve().parents[1] / "src/evolvcoder/strategies.py"
    assert hashlib.sha256(path.read_bytes()).hexdigest() == "ac6c33fc5ad679dfe80495d36098bcf35574beffaffab66368923f5bbf0ad036"


@pytest.mark.parametrize("scores", [
    [.4, .6, .9, 1.], [.4, .1, -.2, .7], [.4, .4, .4, .8],
    [-.4, -.3, -.2, -.1], [None, .4, .6, .9], [.4, None, .6, .8],
    [.4, None, None, .8], [None, None, .4, .5],
])
@pytest.mark.parametrize("final_reserve", [False, True])
def test_theta0_replays_original_parent_local_repair_and_incumbent(tmp_path, scores, final_reserve):
    loop = campaign(tmp_path, scores=scores)
    loop.strategy = ContinuationTreeStrategy(reserve_last_score_for_continuation=final_reserve)
    loop.max_score_evaluations = 4 if final_reserve else 20
    shadow = PolicyTreeStrategy(policy=TreePolicy(), base_config={
        "reserve_last_score_for_continuation": final_reserve})
    best = 0.0
    for _ in scores:
        assert shadow.proposal_directives(loop._snapshot(), 1) == loop.strategy.proposal_directives(loop._snapshot(), 1)
        if not any(a.candidate.action_sha256 not in loop.accounting.attempted_actions for a in loop.actions.values()):
            loop.expand()
        snapshot = loop._snapshot()
        old, new = loop.strategy.select(snapshot), shadow.select(snapshot)
        assert old.action_id == new.action_id
        assert new.score_vector[:len(old.score_vector)] == old.score_vector
        assert new.rationale_codes[:len(old.rationale_codes)] == old.rationale_codes
        step = loop.step()
        shadow.observe(loop.strategy.observations[-1])
        assert shadow.burst == loop.strategy.burst
        assert shadow.parent_attempts == loop.strategy.parent_attempts
        assert shadow.action_bindings == loop.strategy.action_bindings
        assert loop.summary()["best_native_metric"] >= best
        best = loop.summary()["best_native_metric"]
        if step.artifact_valid:
            assert step.action_id in loop.nodes  # Regressions remain inspectable.


def test_theta0_global_score_and_ranking_are_exact_across_many_states(tmp_path):
    loop = campaign(tmp_path, scores=[])
    original, wrapped = ContinuationTreeStrategy(), PolicyTreeStrategy(policy=TreePolicy())
    rng = random.Random(1847)
    for _ in range(300):
        ids = ["root", *(f"n{i}" for i in range(rng.randint(1, 15)))]
        metrics = {n: rng.choice([-.9, -.01, 0., .01, .9, None]) for n in ids}
        validity = {n: metrics[n] is not None for n in ids}
        visits = {n: rng.randint(0, 10000) for n in ids}
        parents = {n: rng.choice(ids[:index]) if index else None for index, n in enumerate(ids)}
        snapshot = replace(loop._snapshot(), node_metrics=metrics, node_artifact_validity=validity,
                           node_parent_ids=parents, node_visits=visits)
        original.parent_attempts = wrapped.parent_attempts = visits
        selected = original._selection_snapshot(snapshot)
        for node in ids:
            assert wrapped._score_parent(selected, node) == _uct_node_score(selected, node)
        assert wrapped._choice(snapshot) == original._choice(snapshot)


@pytest.mark.parametrize("policy,scores,mode", [
    (TreePolicy(max_scored_continuations=1), [.4, .6], "global"),
    (TreePolicy(max_scored_continuations=3), [.4, .6, .8], "local"),
    (TreePolicy(stagnation_patience=2), [.4, .1], "local"),
])
def test_local_policy_knobs_are_consumed(tmp_path, policy, scores, mode):
    loop = campaign(tmp_path, scores=scores)
    loop.max_score_evaluations = 20
    loop.strategy = PolicyTreeStrategy(policy=policy)
    for _ in scores:
        next_step(loop)
    assert loop.strategy._choice(loop._snapshot())[1] == mode


def test_parent_weights_change_actual_choice(tmp_path):
    loop = campaign(tmp_path, scores=[])
    snapshot = replace(loop._snapshot(), node_metrics={"root": 0., "best": 1., "unvisited": -.5},
                       node_artifact_validity=dict.fromkeys(["root", "best", "unvisited"], True),
                       node_parent_ids={"root": None, "best": "root", "unvisited": "root"})
    quality = PolicyTreeStrategy(policy=TreePolicy(parent_exploration_weight=0, parent_branch_pressure_weight=0))
    explore = PolicyTreeStrategy(policy=TreePolicy(parent_quality_weight=0, parent_branch_pressure_weight=0))
    quality.parent_attempts = explore.parent_attempts = {"root": 1000, "best": 1000, "unvisited": 0}
    assert quality._choice(snapshot) == ("best", "global")
    assert explore._choice(snapshot) == ("unvisited", "global")


@pytest.mark.parametrize("policy", [TreePolicy(), TreePolicy(parent_exploration_weight=4)])
def test_invalid_root_cannot_displace_a_valid_child(tmp_path, policy):
    loop = campaign(tmp_path, scores=[])
    snapshot = replace(loop._snapshot(), node_metrics={"root": None, "child": .1},
                       node_artifact_validity={"root": False, "child": True},
                       node_parent_ids={"root": None, "child": "root"})
    wrapped = PolicyTreeStrategy(policy=policy)
    wrapped.parent_attempts = {"child": 10}
    assert wrapped._choice(snapshot) == ("child", "global")
    snapshot = replace(snapshot, node_artifact_validity={"root": False, "child": False})
    assert wrapped._choice(snapshot) == ("root", "global")


def test_frozen_policy_resume_and_drift_rejection(tmp_path):
    loop = campaign(tmp_path, scores=[.4, None])
    policy = TreePolicy(stagnation_patience=2)
    loop.strategy = PolicyTreeStrategy(policy=policy)
    for _ in range(2):
        next_step(loop)
    frozen = json.loads(json.dumps(loop.strategy.checkpoint_state()))
    restored = PolicyTreeStrategy(policy=policy)
    restored.restore_state(frozen)
    assert json.loads(json.dumps(restored.checkpoint_state())) == frozen
    assert restored.select(loop._snapshot()) == loop.strategy.select(loop._snapshot())
    for other in (PolicyTreeStrategy(policy=TreePolicy()), ContinuationTreeStrategy()):
        with pytest.raises(ValueError, match="configuration changed"):
            other.restore_state(frozen)


@pytest.mark.parametrize("change", [{"parent_quality_weight": float("nan")}, {"parent_quality_weight": True},
    {"parent_quality_weight": -1}, {"parent_exploration_weight": 5}, {"max_scored_continuations": 0},
    {"stagnation_patience": 1.1}, {"max_scored_continuations": True}])
def test_invalid_policy_rejected(change):
    with pytest.raises(ValueError):
        TreePolicy(**change)


def test_policy_cannot_override_hard_attempt_cap_or_claim_different_identity():
    assert TreePolicy(parent_quality_weight=1).sha256 == TreePolicy(parent_quality_weight=1.0).sha256
    with pytest.raises(ValueError, match="unchanged local-attempt"):
        PolicyTreeStrategy(policy=TreePolicy(max_scored_continuations=5))
    with pytest.raises(ValueError, match="unknown Tree"):
        TreePolicy.from_dict({**TreePolicy().to_dict(), "score": 9})
