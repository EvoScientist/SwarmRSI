"""Zero-model outer-loop replay; synthetic scores are not benchmark results."""
from __future__ import annotations

import hashlib
import json

import pytest

from swarmrsi.campaign import (
    ActionExecution, ProgramNode, ProposedAction, VSearchCoordinator,
    artifact_workspace_sha256,
)
from swarmrsi.coder_models import CoderExitStatus, CoderOutcome, CoderVersion, ScientificActionContract
from swarmrsi.contracts import ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily
from swarmrsi.strategies import HeterogeneousMultiIslandStrategy, HypothesisProgramTreeStrategy


SHA = hashlib.sha256(b"offline-feedback-fixture").hexdigest()


def file_sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class RecordingProposer:
    def __init__(self):
        self.requests = []
        self.index = 0

    def propose(self, **request):
        self.requests.append(request)
        nodes = {node.id: node for node in request["nodes"]}
        proposals = []
        for slot in request["directives"]:
            action_id = f"proposal_{self.index:04d}"
            self.index += 1
            parent = nodes[slot.code_parent_id]
            dependencies = [ResearchDependency(parent.id, "code_parent", parent.executable_sha256)]
            dependencies.extend(ResearchDependency(source, "intervention_source", nodes[source].executable_sha256)
                                for source in slot.intervention_source_ids)
            action = ActionCandidate(
                id=action_id, hypothesis_id=getattr(slot, "hypothesis_id", None) or f"h-{action_id}",
                operator=ActionOperator.IMPROVE_PROGRAM,
                dependencies=tuple(dependencies),
                # Multi-slot fixtures respect the production planner's
                # within-batch distinct-family rule. Across one-slot batches,
                # repeated families remain possible and observable.
                family=ScientificFamily(
                    "raw", "linear" if request["count"] == 1 else f"linear-{self.index}",
                    "direct", "public metric",
                ),
                predicted_program_quality=0.5, predicted_information_gain=0.5,
                predicted_reuse_value=0.5, predicted_validity=0.9, expected_cost=1.0,
                island_id=slot.island_id,
            )
            proposals.append(ProposedAction(action, ScientificActionContract(
                action_id=action_id, description="Synthetic replay only",
                allowed_write_paths=("run.py",),
            )))
        return proposals, {"model_calls": 1, "input_tokens": 7, "output_tokens": 11}


class ScriptedExecutor:
    def __init__(self, root, scores):
        self.root = root
        self.scores = iter(scores)

    def config_sha256(self, **kwargs):
        return SHA

    def execute(self, *, proposal, parent, step):
        score = next(self.scores)
        valid = score is not None
        cell = self.root / f"cell_{step}"
        artifact = cell / "artifact"
        artifact.mkdir(parents=True)
        code = f"print({step!r})\n" if valid else (parent.artifact / "run.py").read_text()
        (artifact / "run.py").write_text(code)
        failed = cell / "failed_artifact"
        if not valid:
            failed.mkdir()
            (failed / "run.py").write_text("def broken(:\n")
        evidence = cell / "evaluation.json"
        summary = {"metric": score, "per_instance": {"easy": score, "weak": -0.8}}
        if valid:
            evidence.write_text(json.dumps(summary))
        outcome = CoderOutcome(
            status=CoderExitStatus.SUCCESS if valid else CoderExitStatus.IMPLEMENTATION_FAILED,
            version=CoderVersion.C3, config_sha256=SHA, action_id=proposal.candidate.id,
            parent_sha256=parent.executable_sha256, parent_workspace_sha256=parent.workspace_sha256,
            final_sha256=file_sha(artifact / "run.py"),
            final_workspace_sha256=artifact_workspace_sha256(artifact),
            model_calls=1, tool_calls=1, input_tokens=3, output_tokens=5, wall_seconds=0,
            changed_paths=["run.py"], implementation_real=valid, parent_preserved=True,
            reason="synthetic_success" if valid else "syntax_failure",
            diagnostic_artifact_relpath=None if valid else "failed_artifact",
            diagnostic_workspace_sha256=None if valid else artifact_workspace_sha256(failed),
            trajectory_sha256=SHA,
        )
        outcome.seal()
        return ActionExecution(
            outcome, artifact, score, valid, int(valid),
            evaluation_summary=summary if valid else None,
            evaluation_evidence_sha256=file_sha(evidence) if valid else None,
            evaluation_evidence_path=evidence if valid else None,
        )


def campaign(tmp_path, *, batch=1, islands=False, scores=(0.4, 0.1)):
    parent = tmp_path / "root"
    parent.mkdir(parents=True)
    (parent / "run.py").write_text("print('root')\n")
    policy = HeterogeneousMultiIslandStrategy() if islands else HypothesisProgramTreeStrategy()
    return VSearchCoordinator(
        version=policy.name, strategy=policy, proposer=RecordingProposer(),
        executor=ScriptedExecutor(tmp_path, scores),
        root=ProgramNode("root", parent, file_sha(parent / "run.py"),
                         artifact_workspace_sha256(parent), None, "h-root", 0.0, True, None),
        max_model_calls=50, max_score_evaluations=5, max_tokens=10000,
        proposal_batch_size=batch,
    )


def next_step(loop):
    # This is the production runner's available-queue/expand scheduling rule.
    if not any(item.candidate.action_sha256 not in loop.accounting.attempted_actions
               for item in loop.actions.values()):
        loop.expand()
    return loop.step()


def test_incremental_tree_uses_new_parent_and_component_feedback(tmp_path):
    loop = campaign(tmp_path)
    first = next_step(loop)
    second = next_step(loop)
    assert first.code_parent_id == "root"
    assert second.code_parent_id == first.action_id
    request = loop.proposer.requests[1]
    node = next(node for node in request["nodes"] if node.id == first.action_id)
    assert node.evaluation_summary["per_instance"] == {"easy": 0.4, "weak": -0.8}
    assert node.evaluation_evidence_sha256 == file_sha(node.evaluation_evidence_path)
    assert request["view"]["events"]
    assert loop.summary()["best_node_id"] == first.action_id
    assert second.action_id in loop.nodes  # A valid regression remains inspectable.
    assert len(loop.accounting.realized_families) == 1  # Not fabricated diversity.
    assert loop.accounting.model_calls == 4  # Two planner plus two coder calls.
    assert loop.accounting.output_tokens == 32


def test_four_slot_queue_keeps_root_actions_despite_new_score(tmp_path):
    loop = campaign(tmp_path, batch=4)
    first, second = next_step(loop), next_step(loop)
    assert first.code_parent_id == second.code_parent_id == "root"
    assert len(loop.proposer.requests) == 1
    assert loop.accounting.model_calls == 3


@pytest.mark.parametrize("islands", [False, True])
def test_resume_preserves_next_selection_and_accounting(tmp_path, islands):
    loop = campaign(tmp_path, islands=islands)
    next_step(loop)
    saved = json.loads(json.dumps(loop.checkpoint()))
    restored = VSearchCoordinator.restore(
        strategy=type(loop.strategy)(), proposer=RecordingProposer(),
        executor=ScriptedExecutor(tmp_path / "restored", [0.1]), payload=saved,
    )
    assert loop.strategy.proposal_directives(loop._snapshot(), 1) == restored.strategy.proposal_directives(restored._snapshot(), 1)
    assert loop.accounting == restored.accounting
    assert loop.view.sha256 == restored.view.sha256


def test_incremental_islands_still_spend_four_slots_on_coverage(tmp_path):
    loop = campaign(tmp_path, islands=True, scores=[0.4, 0.9, 0.2, 0.3])
    loop.accounting.score_evaluations = 1  # The same root-inclusive five-score ceiling.
    steps = [next_step(loop) for _ in range(4)]
    assert all(row.code_parent_id == "root" for row in steps)
    assert len({loop.actions[row.action_id].candidate.island_id for row in steps}) == 4
    assert loop._remaining().remaining_score_evaluations == 0
    assert len(loop.strategy.migration_history) == 4
    assert all(not packet["consumed"] for packet in loop.strategy.migration_history)
    assert len(loop.proposer.requests[2]["nodes"]) == 3  # New evidence IS visible.
    assert loop.strategy.reward_sum_by_island == dict.fromkeys(loop.strategy.populations, 1.0)
    assert len(loop.accounting.realized_families) == 1  # Four roles are not four families.


@pytest.mark.parametrize("islands", [False, True])
def test_incremental_failure_gets_one_bounded_repair_not_new_planning(tmp_path, islands):
    loop = campaign(tmp_path, islands=islands, scores=[None, None])
    first, repair = next_step(loop), next_step(loop)
    assert first.recovery_route == "repair_implementation"
    assert loop.actions[repair.action_id].candidate.operator == ActionOperator.REPAIR_PROGRAM
    assert repair.code_parent_id == first.code_parent_id == "root"
    assert len(loop.proposer.requests) == 1
    assert len(loop.actions) == 2  # No repair-of-repair recursion.
    assert len(loop.nodes) == 1
    assert len(loop.diagnostics) == 2
    assert loop.accounting.model_calls == 3
    assert loop.accounting.score_evaluations == 0
