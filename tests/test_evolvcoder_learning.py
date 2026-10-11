"""Zero-call file-bound replay. All scores below are explicitly synthetic."""
from dataclasses import asdict
import copy
import json
from pathlib import Path

import pytest

from evolvcoder.campaign import artifact_workspace_sha256
from evolvcoder.contracts import ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily, canonical_sha256
from evolvcoder.learning_evidence import (FeedbackPool, learning_winner_skips, read_run_evidence,
                                         retain_learning_winners, verify_run_evidence)
from evolvcoder.retained_results import file_sha, read_json, write_immutable
from evolvcoder.rsi import RSIChain, ValidationPending, compare_paired
from evolvcoder.tree_policy import TreePolicy
from evolvcoder.wisdom import WisdomContext, WisdomLibrary, verify_snapshot
from test_evolvcoder_retained_results import run as archived_fixture, put, seal


def make_run(directory, policy=TreePolicy(), score=.2, *, task_id="won", seed=0, role="selection", root_score=.05):
    directory.mkdir(parents=True)
    spec, profile_path = archived_fixture.__wrapped__(directory)
    root, source = Path(spec["archive"]), Path(spec["source_archive"])
    profile = read_json(profile_path)
    profile["rsi_policy"] = policy.to_dict()
    put(profile_path, profile)
    c = read_json(root / "checkpoint.json")
    c["strategy_state"] = {"config": {"learned_policy": policy.to_dict()}}
    for node, value in zip(c["nodes"], [root_score, score]):
        evidence = root / Path(node["evaluation_evidence_path"]).relative_to("/original/run")
        ev = read_json(evidence)
        ev["metrics"]["aggregate_improvement"] = value
        put(evidence, ev)
        node["native_metric"] = value
        node["evaluation_evidence_sha256"] = file_sha(evidence)
    action = ActionCandidate(id="child", hypothesis_id="h-child", operator=ActionOperator.IMPROVE_PROGRAM,
        dependencies=(ResearchDependency("root", "code_parent", c["nodes"][0]["executable_sha256"]),),
        family=ScientificFamily("raw", "linear", "predictions", "public metric"),
        predicted_program_quality=.5, predicted_information_gain=.5, predicted_reuse_value=.5,
        predicted_validity=.5, expected_cost=1.)
    c["actions"] = [{"candidate": asdict(action)}]
    c["steps"][0].update(code_parent_id="root", coder_status="success", recovery_route="exploit",
                         native_metric=score)
    put(root / "checkpoint.json", seal(c, "checkpoint_sha256"))
    f = read_json(root / "final.json")
    f.update(steps=c["steps"], search_strategy_state=c["strategy_state"],
             best_evolved_native_metric=score, best_native_metric=max(score, root_score),
             best_node_id="child" if score >= root_score else "root", root_native_metric=root_score)
    f["summary_sha256"] = canonical_sha256({"version": f["version"], "steps": c["steps"],
                                           "graph_sha256": canonical_sha256([])})
    put(root / "final.json", f)
    m = read_json(root / "manifest.json")
    m.update(task_id=task_id, seed=seed, learning_context={"policy": policy.to_dict(), "profile_source_key": "profile.json"})
    m["source_hashes"]["profile.json"] = file_sha(profile_path)
    put(root / "manifest.json", m)
    return read_run_evidence(root, source, original_run_root="/original/run", source_role=role)


class Proposer:
    def __init__(self, policies):
        self.policies = iter(policies)
        self.requests = []

    def propose(self, request):
        self.requests.append(request)
        return {"parameters": next(self.policies).to_dict()["parameters"], "rationale": "Synthetic allocation test.",
                "evidence_run_ids": [request["feedback"]["runs"][0]["run_id"]]}, {"model_calls": 1, "cost_usd": None}


class FileValidator:
    def __init__(self, root, score_by_policy, *, root_score=.05):
        self.root = root
        self.scores = score_by_policy
        self.root_score = root_score
        self.runs, self.requests = {}, []

    def evaluate(self, *, policy, task_id, seed):
        self.requests.append((policy.sha256, task_id, seed))
        key = (policy.sha256, task_id, seed)
        if key not in self.runs:
            self.runs[key] = make_run(self.root / str(len(self.runs)), policy, self.scores[policy.sha256],
                                     task_id=task_id, seed=seed, root_score=self.root_score)
        return self.runs[key]


def setup_chain(tmp_path, *, rounds=3, cells=None, **kwargs):
    pool = FeedbackPool(tmp_path / "pool")
    pool.add(make_run(tmp_path / "history", role="development"))
    cells = cells or [{"task_id": "won", "seed": 0}]
    chain = RSIChain(tmp_path / "chain", theta0=TreePolicy(), cells=cells, rounds=rounds,
                     stop_solved_tasks=False, **kwargs)
    return chain, pool


def test_real_source_bound_feedback_and_wisdom(tmp_path):
    run = make_run(tmp_path / "run", role="development")
    verify_run_evidence(run)
    assert run["utility"] == .2 and run["best_evolved_native_metric"] == .2
    assert run["observations"][0]["implementation_observation"]["hypothesis_disposition"] == "inconclusive"
    pool = FeedbackPool(tmp_path / "pool")
    pool.add(run); pool.add(run)
    assert len(pool.snapshot()["records"]) == 1
    library = WisdomLibrary(tmp_path / "wisdom")
    assert library.ingest(run) == library.ingest(run)
    frozen_path = tmp_path / "W0.json"
    snapshot = library.freeze(frozen_path)
    verify_snapshot(snapshot)
    context = WisdomContext(task_id="different-task", mode="readonly", snapshot_path=frozen_path)
    retrieval = context.retrieve("linear predictions public metric")
    assert retrieval["selected_count"] == 1
    assert retrieval["entries"][0]["authority"] == "advisory_only"
    assert WisdomContext(task_id="won", mode="readonly", snapshot_path=frozen_path).retrieve("linear")["selected_count"] == 0
    # Off must not touch even a nonexistent/malformed library path.
    assert WisdomContext(task_id="different-task", snapshot_path=tmp_path / "absent").retrieve("linear")["selected_count"] == 0
    frozen_path.write_text(frozen_path.read_text() + "\n")
    with pytest.raises(ValueError, match="source reference mismatch"):
        context.retrieve("linear")


@pytest.mark.parametrize("max_entries,max_characters", [(1, 16000), (3, 100)])
def test_multiple_wisdom_entries_report_numeric_truncation(tmp_path, max_entries, max_characters):
    library = WisdomLibrary(tmp_path / "wisdom")
    for index, score in enumerate([.2, .3]):
        library.ingest(make_run(tmp_path / f"source_{index}", score=score, role="development"))
    snapshot = tmp_path / "W0.json"
    library.freeze(snapshot)
    context = WisdomContext(task_id="new", mode="readonly", snapshot_path=snapshot,
                            max_entries=max_entries, max_characters=max_characters)
    retrieved = context.retrieve("linear predictions public metric")
    assert type(retrieved["omitted_count"]) is int and retrieved["omitted_count"] >= 1
    assert retrieved["projected_characters"] <= max_characters


def test_new_learning_winner_is_preserved_and_future_queue_skips_only_original_result(tmp_path):
    run = make_run(tmp_path / "run", score=.2)
    profile = run["comparison_context"]["fixed_profile"]
    winners = retain_learning_winners([run], profile, tmp_path / "retained")
    assert winners[0]["winner_metric"] == .2
    file = tmp_path / "retained" / (run["run_id"] + ".json")
    skip = learning_winner_skips([file], profile)
    assert list(skip) == ["won"] and skip["won"]["new_version_score"] is None
    changed = copy.deepcopy(profile); changed["model"] = "different"
    assert learning_winner_skips([file], changed) == {}
    program = Path(run["run_dir"]) / "cells/0_child/artifact/run.py"
    program.write_text(program.read_text() + "# changed\n")
    with pytest.raises(ValueError):
        learning_winner_skips([file], profile)


@pytest.mark.parametrize("relative", ["cells/0_child/artifact/run.py", "cells/0_child/artifact/output/pred.csv",
    "cells/0_child/evaluation.json", "cells/0_child/result.json", "cells/0_child/trajectory.jsonl", "final.json"])
def test_changed_run_bytes_never_become_learning_evidence(tmp_path, relative):
    run = make_run(tmp_path / "run")
    path = Path(run["run_dir"]) / relative
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError):
        verify_run_evidence(run)


def test_full_profile_affects_comparison_not_only_summary_limits(tmp_path):
    a = make_run(tmp_path / "a", TreePolicy(), .2)
    b = make_run(tmp_path / "b", TreePolicy(parent_exploration_weight=1), .3)
    assert a["compatibility_sha256"] == b["compatibility_sha256"]
    p = Path(b["source_root"]) / "profile.json"
    profile = read_json(p); profile["target_stop_scope"] = "different"; put(p, profile)
    mpath = Path(b["run_dir"]) / "manifest.json"
    m = read_json(mpath); m["source_hashes"]["profile.json"] = file_sha(p); put(mpath, m)
    b = read_run_evidence(Path(b["run_dir"]), Path(b["source_root"]), original_run_root="/original/run", source_role="selection")
    assert a["compatibility_sha256"] != b["compatibility_sha256"]


def test_rsi_accept_reject_resume_current_incumbent_and_export(tmp_path):
    theta0 = TreePolicy()
    better = TreePolicy(parent_exploration_weight=1.)
    worse_than_current = TreePolicy(parent_exploration_weight=2.)
    tie = TreePolicy(parent_exploration_weight=3.)
    chain, pool = setup_chain(tmp_path)
    proposer = Proposer([better, worse_than_current, tie])
    validator = FileValidator(tmp_path / "trials", {theta0.sha256: .1, better.sha256: .3,
                              worse_than_current.sha256: .2, tie.sha256: .3})
    result = chain.run(proposer, validator, pool)
    assert result["finished"] and result["current_policy"] == better.to_dict()
    assert [r["status"] for r in result["round_records"]] == ["accepted", "rejected", "rejected"]
    assert proposer.requests[1]["incumbent"] == better.to_dict()
    assert proposer.requests[2]["incumbent"] == better.to_dict()
    before = len(proposer.requests), len(validator.requests)
    resumed = RSIChain(chain.root, theta0=theta0, cells=chain.config["cells"], rounds=3, stop_solved_tasks=False)
    assert resumed.run(proposer, validator, pool) == result
    assert (len(proposer.requests), len(validator.requests)) == before
    assert resumed.export_policy(tmp_path / "policy.json")["policy"] == better.to_dict()


@pytest.mark.parametrize("margin", [-.1, float("nan"), float("inf"), True])
def test_negative_or_invalid_margin_never_permits_regression(tmp_path, margin):
    with pytest.raises(ValueError, match="finite and nonnegative"):
        RSIChain(tmp_path, theta0=TreePolicy(), cells=[{"task_id": "won", "seed": 0}],
                 stop_solved_tasks=False, minimum_mean_delta=margin)
    with pytest.raises(ValueError, match="finite and nonnegative"):
        compare_paired([], [], cells=[], candidate_policy=TreePolicy(), incumbent_policy=TreePolicy(),
                       minimum_mean_delta=margin)


@pytest.mark.parametrize("journal", ["feedback.json", "proposal_request.json", "proposal_response.json",
                                     "candidate.json", "validation_request.json"])
def test_resume_verifies_original_journals(tmp_path, journal):
    chain, pool = setup_chain(tmp_path, rounds=1)
    better = TreePolicy(parent_exploration_weight=1.)
    validator = FileValidator(tmp_path / "trials", {TreePolicy().sha256: .1, better.sha256: .2})
    chain.run(Proposer([better]), validator, pool)
    path = chain.root / "round_000" / journal
    path.write_text(path.read_text() + "\n")
    with pytest.raises(ValueError, match="reference mismatch"):
        chain.state()


def test_invalid_validator_return_stops_chain_without_trusted_score(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=3)
    class InvalidValidator:
        def evaluate(self, **kwargs):
            return {"invented_score": 999}
    proposer = Proposer([TreePolicy(parent_exploration_weight=1)])
    for _ in range(2):
        with pytest.raises((ValueError, KeyError)):
            chain.run(proposer, InvalidValidator(), pool)
    assert len(proposer.requests) == 1
    assert not chain.state()["finished"]
    assert chain.state()["current_policy"] == TreePolicy().to_dict()
    assert not (chain.root / "round_000/result.json").exists()
    assert not (chain.root / "round_001").exists()
    assert (chain.root / "round_000/received_incumbent_000.json").exists()
    assert len(pool.snapshot()["records"]) == 1


def test_preflight_rejects_impossible_continuation_before_any_trial(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=1)
    validator = FileValidator(tmp_path / "trials", {})
    state = chain.run(Proposer([TreePolicy(max_scored_continuations=5)]), validator, pool)
    assert state["round_records"][0]["status"] == "invalid_proposal"
    assert not validator.requests


def test_process_lock_prevents_second_proposal(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=1)
    proposer = Proposer([])
    with chain._exclusive(), pytest.raises(ValidationPending, match="another process"):
        chain.run_round(proposer, FileValidator(tmp_path / "trials", {}), pool)
    assert not proposer.requests


def test_root_best_alone_does_not_solve_or_remove_task(tmp_path):
    pool = FeedbackPool(tmp_path / "pool")
    pool.add(make_run(tmp_path / "history", role="development"))
    chain = RSIChain(tmp_path / "chain", theta0=TreePolicy(), rounds=1,
                     cells=[{"task_id": "won", "seed": 0}], targets={"won": .5})
    different = TreePolicy(parent_exploration_weight=1.)
    validator = FileValidator(tmp_path / "trials", {TreePolicy().sha256: .2, different.sha256: .3}, root_score=.8)
    state = chain.run(Proposer([different]), validator, pool)
    assert state["solved_tasks"] == {} and state["active_cells"] == chain.config["cells"]
    assert state["round_records"][0]["trial"]["mean_delta"] == 0


def test_pending_validation_reuses_saved_proposal(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=1)
    better = TreePolicy(parent_exploration_weight=1.)
    proposer = Proposer([better])
    class PendingValidator:
        def evaluate(self, **kwargs):
            raise ValidationPending("synthetic external job still running")
    pending = chain.run(proposer, PendingValidator(), pool)
    assert pending["status"] == "validation_pending"
    validator = FileValidator(tmp_path / "trials", {TreePolicy().sha256: .1, better.sha256: .2})
    assert chain.run(proposer, validator, pool)["finished"]
    assert len(proposer.requests) == 1


@pytest.mark.parametrize("mismatch", ["heldout", "wrong_task", "wrong_seed", "wrong_policy"])
def test_unrequested_real_evidence_never_solves_or_enters_trusted_runs(tmp_path, mismatch):
    pool = FeedbackPool(tmp_path / "pool")
    pool.add(make_run(tmp_path / "history", role="development"))
    chain = RSIChain(tmp_path / "chain", theta0=TreePolicy(), rounds=3,
                     cells=[{"task_id": "won", "seed": 0}], targets={"won": .5})
    class WrongValidator:
        record = None

        def evaluate(self, *, policy, task_id, seed):
            if self.record is None:
                self.record = make_run(tmp_path / "wrong", TreePolicy(parent_exploration_weight=4) if mismatch == "wrong_policy" else policy,
                                       .8, task_id="different" if mismatch == "wrong_task" else task_id,
                                       seed=1 if mismatch == "wrong_seed" else seed,
                                       role="heldout" if mismatch == "heldout" else "selection")
            return self.record
    proposer = Proposer([TreePolicy(parent_exploration_weight=1)])
    validator = WrongValidator()
    for _ in range(2):
        with pytest.raises(ValueError, match="response role/task/seed/policy mismatch"):
            chain.run(proposer, validator, pool)
    assert len(proposer.requests) == 1
    assert not chain.state()["finished"] and chain.state()["solved_tasks"] == {}
    assert not (chain.root / "round_000/result.json").exists()
    assert not (chain.root / "round_001").exists()
    assert (chain.root / "round_000/received_incumbent_000.json").exists()
    assert len(pool.snapshot()["records"]) == 1


def test_orphaned_pending_response_cannot_recreate_original_request(tmp_path):
    chain, pool = setup_chain(tmp_path, rounds=1)
    proposer = Proposer([TreePolicy(parent_exploration_weight=1)])
    class Pending:
        def evaluate(self, **kwargs):
            raise ValidationPending("still live")
    chain.run_round(proposer, Pending(), pool)
    (chain.root / "round_000/proposal_request.json").unlink()
    with pytest.raises(ValueError, match="orphaned RSI journal"):
        chain.run_round(proposer, Pending(), pool)
    assert len(proposer.requests) == 1
