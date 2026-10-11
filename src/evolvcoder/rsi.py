"""Persistent bounded Tree-policy learning with real, paired run evidence."""
from __future__ import annotations

from contextlib import contextmanager
import fcntl
import math
from pathlib import Path
from typing import Protocol

from .contracts import canonical_sha256
from .learning_evidence import FeedbackPool, source_ref, verify_ref, verify_run_evidence
from .learning_policies import codec_for
from .retained_results import read_json, require, verify_seal, write_immutable
from .tree_policy import PolicyTreeStrategy, TreePolicy
from .validation_failure import ValidationUnavailable, verify_unavailable


def validate_margin(value: float) -> None:
    require(type(value) in (int, float) and math.isfinite(value) and value >= 0,
            "minimum_mean_delta must be finite and nonnegative")


class PolicyProposer(Protocol):
    def propose(self, request: dict) -> tuple[dict, dict]: ...


class ValidationRunner(Protocol):
    def evaluate(self, *, policy: TreePolicy, task_id: str, seed: int) -> dict: ...


class ValidationPending(RuntimeError):
    """An actual launched run is not terminal; do not relaunch or reject it."""


def compare_paired(candidate: list[dict], incumbent: list[dict], *, cells: list[dict],
                   candidate_policy, incumbent_policy,
                   minimum_mean_delta: float = 0.0) -> dict:
    """Arithmetic mean of paired final.best_native_metric differences.

    The planned cell set is fixed for this comparison. No missing-cell mean,
    fabricated zero, hypothesis verdict, or old-theta0 comparison is allowed.
    """
    validate_margin(minimum_mean_delta)
    keys = [(c["task_id"], c["seed"]) for c in cells]
    require(bool(keys) and len(set(keys)) == len(keys), "empty or duplicate validation cells")
    require(len(candidate) == len(incumbent) == len(keys), "incomplete paired validation")
    deltas, pairs = [], []
    for expected, left, right in zip(keys, candidate, incumbent):
        verify_run_evidence(left); verify_run_evidence(right)
        require((left["task_id"], left["seed"]) == (right["task_id"], right["seed"]) == expected,
                "validation task/seed mismatch")
        require(left["source_role"] == right["source_role"] == "selection", "wrong validation role")
        require(left["policy"] == candidate_policy.to_dict() and right["policy"] == incumbent_policy.to_dict(),
                "validation does not bind the proposed/current policies")
        require(left["run_id"] != right["run_id"], "candidate and incumbent cannot be the same run")
        require(left["compatibility_sha256"] == right["compatibility_sha256"], "unmatched validation conditions")
        require(left["utility_field"] == right["utility_field"] == "final.best_native_metric", "utility field changed")
        require(left["artifact_valid"] is True and right["artifact_valid"] is True,
                "invalid validation delivery; no numeric substitute")
        delta = left["utility"] - right["utility"]
        deltas.append(delta)
        pairs.append({"task_id": expected[0], "seed": expected[1], "candidate_run_id": left["run_id"],
                      "incumbent_run_id": right["run_id"], "candidate_evidence_sha256": left["evidence_sha256"],
                      "incumbent_evidence_sha256": right["evidence_sha256"], "candidate_utility": left["utility"],
                      "incumbent_utility": right["utility"], "delta": delta})
    mean = sum(deltas) / len(deltas)
    return {"utility_field": "final.best_native_metric", "aggregation": "mean_of_fixed_paired_deltas",
            "missing_rule": "invalid_trial_no_score_imputation", "pairs": pairs, "mean_delta": mean,
            "minimum_mean_delta": minimum_mean_delta, "accepted": mean > minimum_mean_delta,
            "reason": "strict_improvement_over_current_incumbent" if mean > minimum_mean_delta else "no_strict_improvement"}


class OpenRouterPolicyProposer:
    def __init__(self, model):
        self.model = model

    def propose(self, request: dict) -> tuple[dict, dict]:
        response, usage = self.model.respond({"version": "RSI_POLICY", "seed": request["proposal_seed"],
                                              "policy_learning_request": request})
        return response, {"model_calls": 1, **usage, "cost_usd": usage.get("cost_usd")}


def feedback_projection(snapshot: dict, *, maximum_runs: int = 20, maximum_observations: int = 6) -> dict:
    """Bounded observations; full evidence remains in the frozen feedback file."""
    records = snapshot["records"][-maximum_runs:]
    projected = []
    for run in records:
        observations = []
        for original in run["observations"][-maximum_observations:]:
            row = {key: original.get(key) for key in ("action_id", "parent_id", "declared_family", "coder_status",
                   "recovery_route", "native_metric", "parent_metric", "implementation_observation", "failure_reason",
                   "failure_stage", "failure_summary", "attempted_program", "evaluation_evidence")}
            row["code_excerpt"] = original.get("code_excerpt", "")[:2000]
            row["omitted_code_characters"] = max(0, len(original.get("code_excerpt", "")) - 2000) + original.get("omitted_code_characters", 0)
            row["program_diff_excerpt"] = original.get("program_diff_excerpt", "")[:3000]
            row["omitted_diff_characters"] = max(0, len(original.get("program_diff_excerpt", "")) - 3000) + original.get("omitted_diff_characters", 0)
            observations.append(row)
        projected.append({"run_id": run["run_id"], "task_id": run["task_id"], "policy": run["policy"],
                          "utility": run["utility"], "accounting": run["accounting"], "observations": observations,
                          "omitted_observations": max(0, len(run["observations"]) - len(observations))})
    return {"feedback_sha256": snapshot["feedback_sha256"], "runs": projected,
            "omitted_runs": max(0, len(snapshot["records"]) - len(records))}


class RSIChain:
    """Disk-backed finite state machine. Records, not mutable scores, own state.

    A response absent after a recorded outbound attempt consumes its round as
    indeterminate; it is never silently reissued. A live validation stays
    pending. Completed validation is verified and reused by the concrete runner.
    """
    def __init__(self, root: Path, *, theta0: TreePolicy, cells: list[dict], rounds: int = 3,
                 proposal_seed: int = 0, minimum_mean_delta: float = 0.0,
                 stop_solved_tasks: bool = True, targets: dict[str, float] | None = None,
                 base_tree_config: dict | None = None, read_only: bool = False):
        validate_margin(minimum_mean_delta)
        require(type(rounds) is int and rounds > 0, "rounds must be a positive integer")
        require(bool(cells) and len({(c["task_id"], c["seed"]) for c in cells}) == len(cells), "invalid validation cell set")
        require(all(set(c) == {"task_id", "seed"} and type(c["seed"]) is int for c in cells), "invalid validation cell schema")
        self.root = root.resolve()
        if not read_only:
            self.root.mkdir(parents=True, exist_ok=True)
        if isinstance(theta0, TreePolicy):
            base = PolicyTreeStrategy(policy=theta0, base_config=base_tree_config).config
            base_tree_config = {k: v for k, v in base.items() if k not in {"seed", "learned_policy"}}
        self.codec = codec_for(theta0, base_tree_config=base_tree_config)
        self.config = {"schema_version": "evolvcoder-rsi-chain-v1", "theta0": theta0.to_dict(), "cells": cells,
                       "rounds": rounds, "proposal_seed": proposal_seed, "minimum_mean_delta": minimum_mean_delta,
                       "stop_solved_tasks": stop_solved_tasks, "targets": targets or {},
                       "utility_field": "final.best_native_metric", "aggregation": "mean_of_fixed_paired_deltas",
                       **self.codec.chain_context()}
        require(not stop_solved_tasks or set(c["task_id"] for c in cells).issubset(self.config["targets"]),
                "solved-task stopping requires each task's explicit target")
        require(all(type(t) in (int, float) and math.isfinite(t) for t in self.config["targets"].values()),
                "nonfinite solved-task target")
        if read_only:
            require(read_json(self.root / "chain.json") == self.config, "RSI chain configuration changed")
        else:
            write_immutable(self.root / "chain.json", self.config)
        self.theta0 = theta0

    @contextmanager
    def _exclusive(self):
        # Advisory OS locks release on crash. A persistent PID file alone
        # cannot make the read-check-write/provider-call boundary exclusive.
        with (self.root / ".execution.lock").open("a+") as stream:
            try:
                fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError as exc:
                raise ValidationPending("another process owns this RSI chain") from exc
            try:
                yield
            finally:
                fcntl.flock(stream, fcntl.LOCK_UN)

    def _solved(self, records: list[dict]) -> dict:
        solved = {}
        for row in records:
            for run in row["candidate_runs"] + row["incumbent_runs"]:
                target = self.config["targets"].get(run["task_id"])
                metric = run["best_evolved_native_metric"]
                if target is not None and run["artifact_valid"] and metric is not None and metric > target:
                    previous = solved.get(run["task_id"])
                    if previous is None or previous["best_evolved_native_metric"] < metric:
                        solved[run["task_id"]] = {"run_id": run["run_id"], "best_evolved_native_metric": metric,
                                                  "evidence_sha256": run["evidence_sha256"]}
        return solved

    def _active_cells(self, records: list[dict]) -> list[dict]:
        solved = self._solved(records)
        return [c for c in self.config["cells"] if not self.config["stop_solved_tasks"] or c["task_id"] not in solved]

    def _request(self, index, incumbent, cells, snapshot, previous):
        return {"schema_version": "evolvcoder-rsi-proposal-v1", "round": index,
                "proposal_seed": self.config["proposal_seed"] + index, "incumbent": incumbent.to_dict(),
                "theta0": self.theta0.to_dict(), "validation_cells": cells,
                **self.codec.chain_context(),
                "feedback": feedback_projection(snapshot),
                "previous_trials": [{"round": r["round"], "round_sha256": r["round_sha256"],
                                     "status": r["status"], "trial": r["trial"]} for r in previous],
                "instruction": self.codec.instruction()}

    def _finished(self) -> list[dict]:
        records = []
        for index in range(self.config["rounds"]):
            path = self.root / f"round_{index:03d}" / "result.json"
            if not path.exists():
                require(not any((self.root / f"round_{j:03d}" / "result.json").exists()
                                for j in range(index + 1, self.config["rounds"])), "RSI round history has a gap")
                break
            row = read_json(path)
            require(not records or records[-1]["status"] != "validation_unavailable",
                    "RSI has a new round after required validation became unavailable")
            verify_seal(row, "round_sha256")
            require(row["round"] == index and row["chain_sha256"] == canonical_sha256(self.config), "RSI round identity mismatch")
            expected_incumbent = records[-1]["retained_policy"] if records else self.theta0.to_dict()
            require(row["incumbent_policy"] == expected_incumbent, "RSI incumbent chain regressed or drifted")
            require(row["cells"] == self._active_cells(records), "historical paired cell set changed")
            refs = row["journal_refs"]
            required = {"feedback.json", "proposal_request.json", "proposal_response.json"}
            if row["candidate_policy"] is not None:
                required |= {"candidate.json", "validation_request.json"}
            required |= {p.name for p in path.parent.glob("received_*.json")}
            require(set(refs) == required, "incomplete RSI proposal journal")
            for name, ref in refs.items():
                require(verify_ref(ref).resolve() == (path.parent / name).resolve(), "RSI journal path mismatch")
            snapshot = read_json(path.parent / "feedback.json")
            verify_seal(snapshot, "feedback_sha256")
            require(snapshot["feedback_sha256"] == row["feedback_sha256"], "round feedback mismatch")
            for record in snapshot["records"]:
                verify_run_evidence(record)
                require(record["source_role"] != "heldout", "heldout feedback forbidden")
            request = self._request(index, self.codec.from_dict(expected_incumbent), row["cells"], snapshot, records)
            require(read_json(path.parent / "proposal_request.json") == request, "proposal request changed")
            response = read_json(path.parent / "proposal_response.json")
            require(response["usage"] == row["proposal_usage"], "proposal accounting changed")
            if row["candidate_policy"] is not None:
                candidate = self.codec.from_dict(row["candidate_policy"])
                self.codec.recheck(candidate)
                proposed = read_json(path.parent / "candidate.json")
                require(proposed["policy"] == candidate.to_dict()
                        and proposed["diff"] == candidate.diff(self.codec.from_dict(expected_incumbent))
                        and response["response"]["parameters"] == candidate.to_dict()["parameters"]
                        and proposed["rationale"] == response["response"]["rationale"]
                        and proposed["evidence_run_ids"] == response["response"]["evidence_run_ids"], "policy proposal provenance mismatch")
                require(read_json(path.parent / "validation_request.json") == {
                    "cells": row["cells"], "candidate": candidate.to_dict(),
                    "incumbent": expected_incumbent, "utility_field": self.config["utility_field"]}, "validation request changed")
            for run in row.get("candidate_runs", []) + row.get("incumbent_runs", []):
                verify_run_evidence(run)
                require(run["source_role"] == "selection", "non-selection validation record")
            for label in ("candidate", "incumbent"):
                for i, run in enumerate(row[f"{label}_runs"]):
                    require(i < len(row["cells"]) and {"task_id": run["task_id"], "seed": run["seed"]} == row["cells"][i]
                            and run["policy"] == row[f"{label}_policy"], "validation request identity mismatch")
                    require(read_json(path.parent / f"received_{label}_{i:03d}.json") == run,
                            "validation return changed")
            if row["trial"] is not None:
                derived = compare_paired(row["candidate_runs"], row["incumbent_runs"], cells=row["cells"],
                                        candidate_policy=self.codec.from_dict(row["candidate_policy"]),
                                        incumbent_policy=self.codec.from_dict(row["incumbent_policy"]),
                                        minimum_mean_delta=self.config["minimum_mean_delta"])
                require(derived == row["trial"], "RSI decision not derived from its real trials")
            expected_retained = row["candidate_policy"] if row["trial"] and row["trial"]["accepted"] else expected_incumbent
            require(row["retained_policy"] == expected_retained, "unverified RSI policy promotion")
            if row["trial"] is not None:
                require(row["status"] == ("accepted" if row["trial"]["accepted"] else "rejected"), "trial status mismatch")
            if row["status"] == "validation_unavailable":
                failure = row["validation_failure"]
                identity = verify_unavailable(failure)
                require(row["trial"] is None and row["retained_policy"] == expected_incumbent,
                        "unavailable validation cannot promote a policy")
                # Exact next required member of the fixed incumbent/candidate
                # sequence, not an unrelated trial supplied by a validator.
                received = len(row["incumbent_runs"]) + len(row["candidate_runs"])
                require(len(row["incumbent_runs"]) == len(row["candidate_runs"]) + received % 2
                        and received < 2 * len(row["cells"]), "invalid unavailable validation prefix")
                cell = row["cells"][received // 2]
                label = "incumbent" if received % 2 == 0 else "candidate"
                require(identity["policy"] == row[f"{label}_policy"]
                        and {"task_id": identity["task_id"], "seed": identity["seed"]} == cell,
                        "unavailable validation belongs to a different required trial")
            else:
                require("validation_failure" not in row, "unexpected unavailable-validation record")
            records.append(row)
        return records

    def state(self) -> dict:
        rounds = self._finished()
        retained = rounds[-1]["retained_policy"] if rounds else self.theta0.to_dict()
        solved = self._solved(rounds)
        active = self._active_cells(rounds)
        body = {"schema_version": "evolvcoder-rsi-state-v1", "chain_sha256": canonical_sha256(self.config),
                "completed_rounds": len(rounds), "current_policy": retained, "solved_tasks": solved,
                "active_cells": active, "finished": len(rounds) == self.config["rounds"] or not active,
                "round_records": [{"round": r["round"], "round_sha256": r["round_sha256"],
                                   "status": r["status"], "trial": r["trial"]} for r in rounds]}
        if rounds and rounds[-1]["status"] == "validation_unavailable":
            body.update(finished=False, halted=True, status="validation_unavailable",
                        active_cells=rounds[-1]["cells"],
                        validation_failure=rounds[-1]["validation_failure"])
        return {**body, "state_sha256": canonical_sha256(body)}

    def run_round(self, proposer: PolicyProposer, validator: ValidationRunner, pool: FeedbackPool) -> dict:
        with self._exclusive():
            return self._run_round_locked(proposer, validator, pool)

    def _run_round_locked(self, proposer: PolicyProposer, validator: ValidationRunner, pool: FeedbackPool) -> dict:
        state = self.state()
        if state["finished"] or state.get("halted"):
            return state
        index = state["completed_rounds"]
        root = self.root / f"round_{index:03d}"
        root.mkdir(parents=True, exist_ok=True)
        # A crash may leave a prefix of this journal, never a valid suffix
        # with its original request removed. Do not reconstruct lost custody.
        order = ["feedback.json", "proposal_request.json", "proposal_response.json", "candidate.json", "validation_request.json"]
        existing = [i for i, name in enumerate(order) if (root / name).exists()]
        if existing:
            require(existing == list(range(max(existing) + 1)), "orphaned RSI journal; original source missing")
        if any(root.glob("received_*.json")):
            require(len(existing) == len(order), "received validation without original request")
        incumbent = self.codec.from_dict(state["current_policy"])
        snapshot_path = root / "feedback.json"
        if not snapshot_path.exists():
            write_immutable(snapshot_path, pool.snapshot())
        snapshot = read_json(snapshot_path)
        verify_seal(snapshot, "feedback_sha256")
        for record in snapshot["records"]:
            verify_run_evidence(record)
            require(record["source_role"] != "heldout", "heldout feedback forbidden")
        cells = state["active_cells"]
        request = self._request(index, incumbent, cells, snapshot, self._finished())
        request_path, response_path = root / "proposal_request.json", root / "proposal_response.json"
        response_exists = response_path.exists()
        request_exists = request_path.exists()
        write_immutable(request_path, request)
        if not response_exists:
            if request_exists:
                response, usage = {"error": "indeterminate_provider_result_no_reissue"}, {"model_calls": None, "cost_usd": None}
            else:
                try:
                    response, usage = proposer.propose(request)
                except Exception as exc:
                    response, usage = {"error": "provider_failure", "exception_type": type(exc).__name__}, {"model_calls": 1, "cost_usd": None}
            write_immutable(response_path, {"response": response, "usage": usage})
        response_record = read_json(response_path)
        response, usage = response_record["response"], response_record["usage"]
        candidate, trial, candidate_runs, incumbent_runs = None, None, [], []
        status, error = "invalid_proposal", None
        validation_failure = None
        # History verification is controller work, not model-proposal validation.
        old_candidates = {r["candidate_policy"]["policy_sha256"] for r in self._finished() if r["candidate_policy"]}
        try:
            require(isinstance(response, dict), "policy proposal must be an object")
            if response.get("error") or response.get("no_candidate"):
                raise ValueError(response.get("error", "no_candidate"))
            clean = {k: v for k, v in response.items() if not k.startswith("_")}
            require(set(clean) == {"parameters", "rationale", "evidence_run_ids"}, "policy proposal fields")
            require(isinstance(clean["parameters"], dict), "policy parameters must be an object")
            require(isinstance(clean["rationale"], str) and bool(clean["rationale"].strip()), "missing policy rationale")
            known = {r["run_id"] for r in snapshot["records"]}
            require(bool(clean["evidence_run_ids"]) and set(clean["evidence_run_ids"]).issubset(known), "policy cites unavailable feedback")
            proposed = self.codec.build(clean["parameters"])
            require(bool(proposed.diff(incumbent)), "candidate duplicates incumbent")
            require(proposed.sha256 not in old_candidates, "candidate duplicates a previous trial")
            candidate = proposed
        except (ValueError, KeyError, TypeError) as exc:
            error = {"exception_type": type(exc).__name__, "reason": str(exc)[:1000]}
        if candidate is not None:
            # Only malformed model proposals consume an invalid-proposal round.
            # Persistence and required-evidence errors must escape the chain:
            # they are neither measured scientific rejection nor unavailable data.
            write_immutable(root / "candidate.json", {"policy": candidate.to_dict(), "diff": candidate.diff(incumbent),
                                                       "rationale": clean["rationale"], "evidence_run_ids": clean["evidence_run_ids"]})
            # The entire paired set is fixed before either policy is run. A
            # win may remove a task NEXT round, never half of this comparison.
            write_immutable(root / "validation_request.json", {"cells": cells, "candidate": candidate.to_dict(),
                                                               "incumbent": incumbent.to_dict(), "utility_field": self.config["utility_field"]})
            try:
                for cell in cells:
                    for label, policy, destination in (("incumbent", incumbent, incumbent_runs), ("candidate", candidate, candidate_runs)):
                        returned = validator.evaluate(policy=policy, **cell)
                        # Preserve a malformed return for diagnosis but never put
                        # it in the trusted run list or feedback pool.
                        received_path = root / f"received_{label}_{len(destination):03d}.json"
                        write_immutable(received_path, returned)
                        verify_run_evidence(returned)
                        require(returned["source_role"] == "selection" and returned["policy"] == policy.to_dict()
                                and returned["task_id"] == cell["task_id"] and returned["seed"] == cell["seed"],
                                "validation response role/task/seed/policy mismatch")
                        destination.append(returned)
                        pool.add(returned)
                trial = compare_paired(candidate_runs, incumbent_runs, cells=cells, candidate_policy=candidate,
                                       incumbent_policy=incumbent, minimum_mean_delta=self.config["minimum_mean_delta"])
                status = "accepted" if trial["accepted"] else "rejected"
            except ValidationPending:
                return {**state, "pending_round": index, "status": "validation_pending"}
            except ValidationUnavailable as exc:
                validation_failure = exc.evidence
                status = "validation_unavailable"
                error = {"exception_type": type(exc).__name__, "reason": str(exc)}
        retained = candidate if trial and trial["accepted"] else incumbent
        journal_names = ["feedback.json", "proposal_request.json", "proposal_response.json"]
        if candidate is not None:
            journal_names += ["candidate.json", "validation_request.json"]
        journal_names += [p.name for p in sorted(root.glob("received_*.json"))]
        body = {"schema_version": "evolvcoder-rsi-round-v1", "chain_sha256": canonical_sha256(self.config),
                "round": index, "cells": cells, "feedback_sha256": snapshot["feedback_sha256"],
                "incumbent_policy": incumbent.to_dict(), "candidate_policy": candidate.to_dict() if candidate else None,
                "retained_policy": retained.to_dict(), "status": status, "error": error, "trial": trial,
                "journal_refs": {name: source_ref(root / name) for name in journal_names},
                "candidate_runs": candidate_runs, "incumbent_runs": incumbent_runs,
                "proposal_usage": usage, "validation_accounting": {"unique_runs": len({r["run_id"] for r in candidate_runs + incumbent_runs}),
                "reported_run_costs_usd": {r["run_id"]: r["cost_usd"] for r in candidate_runs + incumbent_runs},
                "actual_spend_authority": "validation runner launch/cache ledger; logical reuse is not charged twice"}}
        if validation_failure is not None:
            body["validation_failure"] = validation_failure
        write_immutable(root / "result.json", {**body, "round_sha256": canonical_sha256(body)})
        return self.state()

    def run(self, proposer: PolicyProposer, validator: ValidationRunner, pool: FeedbackPool) -> dict:
        while not self.state()["finished"]:
            result = self.run_round(proposer, validator, pool)
            if result.get("status") == "validation_pending" or result.get("halted"):
                return result
        return self.state()

    def export_policy(self, path: Path) -> dict:
        state = self.state()
        require(not state.get("halted"), "cannot export a learned policy from unavailable validation; original incumbent remains preserved")
        require(state["finished"], "cannot freeze a policy while its learning chain is unfinished")
        body = {"schema_version": "evolvcoder-frozen-rsi-policy-v1", "policy": state["current_policy"],
                "theta0": self.theta0.to_dict(), "chain_sha256": state["chain_sha256"],
                "state_sha256": state["state_sha256"], "completed_rounds": state["completed_rounds"],
                "round_records": state["round_records"], "scope": "development adaptation; not unmeasured transfer"}
        write_immutable(path, {**body, "export_sha256": canonical_sha256(body)})
        return body
