import json

import pytest

from swarmrsi import EvidenceCoupledStrategy
from swarmrsi.campaign import VSearchCoordinator
from swarmrsi.naturebench_session import strategy
from swarmrsi.root_drafts import RootDraftingEvidenceStrategy
from test_swarmrsi_incremental_feedback import campaign, next_step, RecordingProposer, ScriptedExecutor


def _loop(tmp_path, drafts, scores):
    loop = campaign(tmp_path, scores=scores)
    loop.strategy = RootDraftingEvidenceStrategy(initial_root_drafts=drafts) if drafts else EvidenceCoupledStrategy()
    loop.version = loop.strategy.name
    loop.max_score_evaluations = 10
    return loop


def test_first_actions_are_independent_root_drafts_then_best_draft_is_refined(tmp_path):
    loop = _loop(tmp_path, 3, [.1, .4, .2, .5])
    steps = [next_step(loop) for _ in range(4)]
    assert [s.code_parent_id for s in steps[:3]] == ["root", "root", "root"]
    # Drafts do not borrow from one another.
    assert all(not d.intervention_source_ids for r in loop.proposer.requests[:3] for d in r["directives"])
    assert "initial_root_draft" in loop.proposer.requests[0]["directives"][0].rationale_codes
    # After the drafts, the ordinary v4 ranking picks the best draft (0.4).
    assert steps[3].code_parent_id == steps[1].action_id


def test_failed_draft_still_counts_so_drafting_cannot_loop(tmp_path):
    loop = _loop(tmp_path, 2, [None, .3, .5])
    steps = [next_step(loop) for _ in range(3)]
    assert steps[0].code_parent_id == "root"
    assert steps[2].code_parent_id != "root"


def test_without_drafts_v4_is_unchanged(tmp_path):
    loop = _loop(tmp_path, 0, [.4, .1])
    first, second = next_step(loop), next_step(loop)
    assert first.code_parent_id == "root" and second.code_parent_id == first.action_id


def test_drafting_resumes_from_checkpoint(tmp_path):
    loop = _loop(tmp_path, 3, [.1])
    next_step(loop)
    saved = json.loads(json.dumps(loop.checkpoint(), allow_nan=False))
    restored = VSearchCoordinator.restore(
        strategy=RootDraftingEvidenceStrategy(initial_root_drafts=3), proposer=RecordingProposer(),
        executor=ScriptedExecutor(tmp_path / "restored", [.2]), payload=saved)
    directive, = restored.strategy.proposal_directives(restored._snapshot(), 1)
    assert directive.code_parent_id == "root" and "initial_root_draft" in directive.rationale_codes
    assert directive == loop.strategy.proposal_directives(loop._snapshot(), 1)[0]


def test_profile_wiring_and_validation():
    drafting = strategy("v4", 0, {"screening_execution_limits": {"initial_root_drafts": 3}})
    assert isinstance(drafting, RootDraftingEvidenceStrategy) and drafting.initial_root_drafts == 3
    assert drafting.name == EvidenceCoupledStrategy.name
    assert type(strategy("v4", 0, {})) is EvidenceCoupledStrategy
    assert type(strategy("v4", 0, {"screening_execution_limits": {"initial_root_drafts": 0}})) is EvidenceCoupledStrategy
    with pytest.raises(ValueError):
        RootDraftingEvidenceStrategy(initial_root_drafts=0)


def _tree_loop(tmp_path, drafts, scores):
    from swarmrsi.root_drafts import RootDraftingContinuationTree
    from swarmrsi.strategies import ContinuationTreeStrategy
    loop = campaign(tmp_path, scores=scores)
    loop.strategy = (RootDraftingContinuationTree(initial_root_drafts=drafts) if drafts
                     else ContinuationTreeStrategy())
    loop.version = loop.strategy.name
    loop.max_score_evaluations = 10
    return loop


def test_v1_continuation_tree_drafts_from_root_then_refines_best(tmp_path):
    loop = _tree_loop(tmp_path, 3, [.1, .4, .2, .5])
    steps = [next_step(loop) for _ in range(4)]
    assert [s.code_parent_id for s in steps[:3]] == ["root", "root", "root"]
    assert "initial_root_draft" in loop.proposer.requests[0]["directives"][0].rationale_codes
    assert steps[3].code_parent_id == steps[1].action_id  # the best draft (0.4), not the last one


@pytest.mark.parametrize("scores, best", [([.4, .1, .2, .5], 0), ([.1, .2, .4, .5], 2), ([None, .3, .2, .5], 1)])
def test_v1_refines_the_best_draft_wherever_it_came(tmp_path, scores, best):
    loop = _tree_loop(tmp_path, 3, scores)
    steps = [next_step(loop) for _ in range(4)]
    assert steps[3].code_parent_id == steps[best].action_id


def test_v1_best_draft_burst_survives_resume(tmp_path):
    from swarmrsi.root_drafts import RootDraftingContinuationTree
    loop = _tree_loop(tmp_path, 3, [.1, .4, .2])
    for _ in range(3):
        next_step(loop)
    saved = json.loads(json.dumps(loop.checkpoint(), allow_nan=False))
    restored = VSearchCoordinator.restore(
        strategy=RootDraftingContinuationTree(initial_root_drafts=3), proposer=RecordingProposer(),
        executor=ScriptedExecutor(tmp_path / "restored", [.5]), payload=saved)
    assert restored.strategy.proposal_directives(restored._snapshot(), 1)[0].code_parent_id == \
        loop.strategy.proposal_directives(loop._snapshot(), 1)[0].code_parent_id == "proposal_0001"


def test_v1_without_drafts_is_unchanged(tmp_path):
    loop = _tree_loop(tmp_path, 0, [.4, .1])
    first, second = next_step(loop), next_step(loop)
    assert first.code_parent_id == "root" and second.code_parent_id == first.action_id


def test_v1_profile_wiring():
    from swarmrsi.root_drafts import RootDraftingContinuationTree
    from swarmrsi.strategies import ContinuationTreeStrategy
    tree = {"policy": "continuation_tree", "max_scored_continuations": 3}
    drafting = strategy("v1", 0, {"tree_search": tree, "screening_execution_limits": {"initial_root_drafts": 3}})
    assert isinstance(drafting, RootDraftingContinuationTree) and drafting.initial_root_drafts == 3
    assert drafting.config["max_scored_continuations"] == 3 and drafting.config["initial_root_drafts"] == 3
    assert type(strategy("v1", 0, {"tree_search": tree})) is ContinuationTreeStrategy
    with pytest.raises(ValueError):
        strategy("v1", 0, {"screening_execution_limits": {"initial_root_drafts": 3}})
