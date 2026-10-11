"""Sequential Monte Carlo program search bound to the search-plugin lifecycle.

The particle machinery is the existing SMC: an annealed stage schedule, ESS
resampling, one outstanding mutation at a time, and weights that only executable,
score-bearing, contract-faithful evidence can move. This module supplies the
controller bindings that machinery needs and nothing else. It is not the
continuation Tree under another name, and it makes no scientific judgement: a
failed attempt is recorded as an ineligible candidate, never as a refutation.

Two bindings are deliberately conservative and stated here rather than hidden:
contract fidelity is the binary "executed and produced a valid artifact", because
the controller observation carries no graded fidelity; and SMC's own wall bound
is inert, because the session already enforces the real wall envelope.
"""
from __future__ import annotations

from dataclasses import asdict, replace
import math
from typing import Any

from .contracts import SearchDecision, SearchObservation, SearchSnapshot, canonical_sha256
from .research_smc import (MetricDirection, MutationResult, ProgramBudget, ProgramEvidence, ProgramParticle,
                           ResearchContract, SMC, SMCConfig)
from .strategies import ProposalDirective, _available, _parent_id, _rank_parent_ids

INERT_WALL_BOUND = 1e9


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


class SMCSearchStrategy:
    name = "research_smc"

    def __init__(self, *, seed: int = 0, particle_count: int = 4, proposals_per_particle: int = 1,
                 target_beta: float = 8.0, ess_fraction: float = 0.8, min_stages: int = 3, max_stages: int = 12,
                 max_inspirations: int = 3, reward_lower_bound: float = -1.0, reward_upper_bound: float = 1.0,
                 metric_direction: str = "maximize", primary_metric: str = "native_metric",
                 claim: str = "sequential Monte Carlo program search over the configured task") -> None:
        self.config = {
            "policy": self.name, "seed": seed, "particle_count": particle_count,
            "proposals_per_particle": proposals_per_particle, "target_beta": target_beta,
            "ess_fraction": ess_fraction, "min_stages": min_stages, "max_stages": max_stages,
            "max_inspirations": max_inspirations, "reward_lower_bound": reward_lower_bound,
            "reward_upper_bound": reward_upper_bound, "metric_direction": metric_direction,
            "primary_metric": primary_metric, "claim": claim,
        }
        # Constructing these now surfaces an invalid configuration before any
        # provider call, instead of at the first observation.
        self._smc_config = SMCConfig(particle_count=particle_count, proposals_per_particle=proposals_per_particle,
                                     target_beta=target_beta, ess_fraction=ess_fraction, min_stages=min_stages,
                                     max_stages=max_stages, max_inspirations=max_inspirations, seed=seed)
        self._direction = MetricDirection(metric_direction)
        if reward_lower_bound >= reward_upper_bound:
            raise ValueError("reward bounds must have positive width")
        self.observations: list[SearchObservation] = []
        self.smc: SMC | None = None
        self.particle_nodes: dict[str, str] = {}
        self.bootstrap: list[dict[str, Any]] = []
        self.initialization_failures: list[dict[str, Any]] = []
        self.action_bindings: dict[str, dict[str, Any]] = {}
        self.pending: dict[str, Any] | None = None
        self.deferred: dict[str, Any] | None = None
        self.stopped: str | None = None

    # ------------------------------------------------------------- evidence
    @staticmethod
    def _evidence(observation: SearchObservation) -> ProgramEvidence:
        scored = observation.score_bearing and observation.artifact_valid and _finite(observation.native_metric)
        realized = bool(observation.executable and observation.artifact_valid)
        return ProgramEvidence(
            primary_metric=float(observation.native_metric) if scored else None,
            artifact_valid=bool(observation.artifact_valid),
            contract_fidelity=1.0 if realized else 0.0,
            required_tests_passed=bool(observation.executable),
            score_bearing=bool(scored),
            model_calls=int(observation.model_calls),
            score_evaluations=1 if scored else 0,
            wall_seconds=0.0,
            cost_usd=float(observation.cost_usd),
            behavior_signature=observation.realized_family_sha256,
        )

    @staticmethod
    def _candidate_sha(observation: SearchObservation) -> str | None:
        """The candidate's own program hash, or the retained failed workspace."""
        for value in (observation.program_sha256, (observation.failure_summary or {}).get("diagnostic_workspace_sha256")):
            if isinstance(value, str) and len(value) == 64:
                return value
        return None

    # ------------------------------------------------------------ bootstrap
    def _valid_nodes(self, snapshot: SearchSnapshot) -> list[str]:
        return sorted((node for node, valid in snapshot.node_artifact_validity.items()
                       if valid and _finite(snapshot.node_metrics.get(node))),
                      key=lambda node: (snapshot.node_metrics[node], node), reverse=True)

    def _start_smc(self, snapshot: SearchSnapshot) -> None:
        budget = ProgramBudget(
            max_model_calls=max(1, snapshot.budget.remaining_model_calls),
            max_score_evaluations=max(1, snapshot.budget.remaining_score_evaluations),
            max_wall_seconds=INERT_WALL_BOUND,
            max_cost_usd=snapshot.budget.remaining_cost_usd)
        root_sha = snapshot.node_executable_sha256s.get("root") or self.bootstrap[0]["program_sha256"]
        contract = ResearchContract(
            hypothesis_id="smc_population", claim=self.config["claim"], baseline_artifact_sha256=root_sha,
            required_tests=("controller_execution_and_official_evaluation",),
            constraints=("controller owns the evaluator, data split and budget",),
            primary_metric=self.config["primary_metric"], metric_direction=self._direction,
            reward_lower_bound=self.config["reward_lower_bound"],
            reward_upper_bound=self.config["reward_upper_bound"], budget=budget)
        particles = tuple(ProgramParticle(particle_id=row["node_id"], program_sha256=row["program_sha256"],
                                          artifact_ref=row["node_id"],
                                          evidence=ProgramEvidence(**row["evidence"]))
                          for row in self.bootstrap[:self._smc_config.particle_count])
        self.smc = SMC(contract=contract, config=self._smc_config, initial_particles=particles,
                       initialization_failures=tuple(ProgramEvidence(**row) for row in self.initialization_failures))
        self.particle_nodes = {row["node_id"]: row["node_id"] for row in self.bootstrap}

    def _settle_deferred(self) -> None:
        """No repair followed, so the failed attempt is the request's outcome."""
        deferred, self.deferred = self.deferred, None
        if deferred is None:
            return
        if deferred["request_id"] is None:
            # A failed initialization attempt is charged, never a particle.
            self.initialization_failures.append(deferred["evidence"])
            return
        if self.smc is None:
            raise ValueError("a deferred mutation outlived its population")
        self.smc.record_result(MutationResult(
            request_id=deferred["request_id"], candidate_program_sha256=deferred["program_sha256"],
            artifact_ref=deferred["artifact_ref"], evidence=ProgramEvidence(**deferred["evidence"])))

    # ----------------------------------------------------------- directives
    def proposal_directives(self, snapshot: SearchSnapshot, count: int) -> tuple[ProposalDirective, ...]:
        if count != 1:
            raise ValueError("SMC requires incremental proposal_batch_size=1")
        if self.pending is not None:
            raise ValueError("an SMC mutation is already outstanding; record it before requesting another")
        self._settle_deferred()
        if self.stopped is not None:
            return ()
        if self.smc is None and len(self.bootstrap) >= self._smc_config.particle_count:
            self._start_smc(snapshot)
        if self.smc is None:
            ranked = self._valid_nodes(snapshot) or _rank_parent_ids(snapshot)
            if not ranked:
                raise RuntimeError("SMC has no valid executable parent to initialize from")
            parent = ranked[0]
            self.pending = {"parent": parent, "sources": (), "request_id": None}
            return (ProposalDirective(parent, rationale_codes=(
                "smc_population_initialization",
                f"particle_{len(self.bootstrap)}_of_{self._smc_config.particle_count}",
                "measured_program_lineage_not_hypothesis_verdict")),)
        request = self.smc.next_request()
        if request is None:
            self.stopped = self.smc.stop_reason or "smc_exhausted"
            return ()
        parent = self.particle_nodes[request.parent_particle_id]
        sources = tuple(self.particle_nodes[item] for item in request.inspiration_particle_ids
                        if item in self.particle_nodes)
        self.pending = {"parent": parent, "sources": sources, "request_id": request.request_id}
        return (ProposalDirective(parent, intervention_source_ids=sources, rationale_codes=(
            "smc_mutation", f"smc_stage_{request.stage}", f"smc_kernel_{request.kernel.value}",
            f"smc_lambda_{request.annealing_lambda:.4f}")),)

    # -------------------------------------------------------------- select
    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        candidates = _available(snapshot)
        if len(candidates) != 1:
            raise ValueError("SMC schedules one action at a time")
        action = candidates[0]
        parent = _parent_id(action)
        pending = self.pending
        if pending is None:
            deferred = self.deferred
            if (deferred is None or action.operator.value != "repair_program"
                    or deferred["code_parent_id"] != parent):
                raise ValueError("no SMC directive is bound for this action")
            # The controller's bounded repair re-attempts the same mutation, so
            # it inherits that request instead of opening a new one.
            pending = {"parent": parent, "sources": deferred["sources"], "request_id": deferred["request_id"],
                       "repair_of": deferred["action_id"]}
            self.pending = pending
        sources = tuple(d.source_id for d in action.dependencies if d.kind == "intervention_source")
        if parent != pending["parent"] or sources != tuple(pending["sources"]):
            raise ValueError("action does not match the outstanding SMC request")
        self.action_bindings[action.id] = {"action_sha256": action.action_sha256, "code_parent_id": parent,
                                           "request_id": pending["request_id"], "sources": list(sources),
                                           "repair_of": pending.get("repair_of")}
        stage = self.smc.stage + 1 if self.smc is not None else 0
        return SearchDecision(
            strategy=self.name, action_id=action.id, graph_sha256=snapshot.graph_sha256,
            score_vector=(("smc_stage", float(stage)),
                          ("annealing_lambda", float(self.smc.annealing_lambda) if self.smc else 0.0)),
            rationale_codes=("smc_mutation" if pending["request_id"] else "smc_population_initialization",
                             "multi_parent_allowed" if sources else "single_code_parent",
                             *(("bounded_repair_of_same_request",) if pending.get("repair_of") else ())),
            policy_sha256=canonical_sha256(self.config))

    # ------------------------------------------------------------- observe
    def observe(self, observation: SearchObservation) -> None:
        binding = self.action_bindings.get(observation.action_id)
        if (binding is None or binding["action_sha256"] != observation.action_sha256
                or binding["code_parent_id"] != observation.code_parent_id):
            raise ValueError("SMC observation is not bound to its selected action")
        if any(item.action_id == observation.action_id for item in self.observations):
            raise ValueError("SMC observation already consumed")
        self.observations.append(observation)
        if observation.recovery_route == "retry_evaluator_same_candidate":
            # An evaluator retry is not a mutation outcome; the request stays open.
            return
        evidence = self._evidence(observation)
        digest = self._candidate_sha(observation)
        carried = self.deferred if binding.get("repair_of") else None
        if carried is not None:
            # The failed attempt's cost belongs to the request it left open.
            prior = ProgramEvidence(**carried["evidence"])
            evidence = replace(evidence, model_calls=evidence.model_calls + prior.model_calls,
                               score_evaluations=evidence.score_evaluations + prior.score_evaluations,
                               cost_usd=evidence.cost_usd + prior.cost_usd, evidence_sha256="")
            self.deferred = None
        realized = bool(evidence.score_bearing and evidence.artifact_valid and observation.program_node_id)
        if carried is None and not realized:
            # Hold one failed attempt in either phase: the controller may schedule
            # a bounded repair that completes this same initialization or mutation.
            self.deferred = {"request_id": binding["request_id"], "program_sha256": digest,
                             "artifact_ref": observation.program_node_id or observation.action_id,
                             "evidence": asdict(evidence), "code_parent_id": binding["code_parent_id"],
                             "sources": list(binding["sources"]), "action_id": observation.action_id}
            self.pending = None
            return
        if binding["request_id"] is None:
            # A repaired attempt that still failed settles its initialization
            # slot as charged evidence; the controller does not repair a repair.
            if realized:
                self.bootstrap.append({"node_id": observation.program_node_id, "program_sha256": digest,
                                       "evidence": asdict(evidence)})
            else:
                self.initialization_failures.append(asdict(evidence))
            self.pending = None
            return
        if self.smc is None:
            raise ValueError("an SMC mutation was recorded before its population existed")
        if digest is None:
            raise ValueError("SMC needs the candidate program hash or its retained failed workspace hash")
        self.smc.record_result(MutationResult(
            request_id=binding["request_id"], candidate_program_sha256=digest,
            artifact_ref=observation.program_node_id or observation.action_id, evidence=evidence))
        newest = self.smc.history[-1]
        if observation.program_node_id is not None:
            # A particle can later be a parent or an inspiration, and either way
            # the controller must be able to compile an action against its node.
            self.particle_nodes[newest.particle_id] = observation.program_node_id
        self.pending = None

    def population_projection(self) -> dict[str, Any] | None:
        return None if self.smc is None else self.smc.projection()

    # --------------------------------------------------------- persistence
    def checkpoint_state(self) -> dict[str, Any]:
        return {"config": dict(self.config), "bootstrap": [dict(row) for row in self.bootstrap],
                "initialization_failures": [dict(row) for row in self.initialization_failures],
                "particle_nodes": dict(self.particle_nodes), "pending": self.pending,
                "action_bindings": {key: dict(value) for key, value in self.action_bindings.items()},
                "stopped": self.stopped, "deferred": self.deferred,
                "smc": None if self.smc is None else self.smc.checkpoint(),
                "observations": [asdict(item) for item in self.observations]}

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("config") != self.config:
            raise ValueError("SMC checkpoint configuration changed")
        self.bootstrap = [dict(row) for row in payload["bootstrap"]]
        self.initialization_failures = [dict(row) for row in payload["initialization_failures"]]
        self.particle_nodes = dict(payload["particle_nodes"])
        self.pending = payload["pending"]
        self.action_bindings = {key: dict(value) for key, value in payload["action_bindings"].items()}
        self.stopped = payload["stopped"]
        self.deferred = payload["deferred"]
        self.smc = None if payload["smc"] is None else SMC.restore(payload["smc"])
        self.observations = [SearchObservation(**{
            **row, "intervention_source_ids": tuple(row.get("intervention_source_ids", ()))})
            for row in payload.get("observations", [])]
