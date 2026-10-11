from __future__ import annotations

import math
import random
import json
from dataclasses import asdict, dataclass, replace
from enum import StrEnum
from typing import Any

from .contracts import canonical_sha256


class MetricDirection(StrEnum):
    MAXIMIZE = "maximize"
    MINIMIZE = "minimize"


class MutationKernel(StrEnum):
    LOCAL_DIFF = "local_diff"
    STRUCTURAL_REWRITE = "structural_rewrite"
    ALTERNATIVE_REALIZATION = "alternative_realization"
    CROSS_PROGRAM_INSPIRATION = "cross_program_inspiration"

    @property
    def uses_inspiration(self) -> bool:
        return self == self.CROSS_PROGRAM_INSPIRATION


@dataclass(frozen=True, slots=True)
class ProgramBudget:
    max_model_calls: int
    max_score_evaluations: int
    max_wall_seconds: float
    max_cost_usd: float | None = None

    def __post_init__(self) -> None:
        if min(self.max_model_calls, self.max_score_evaluations) <= 0:
            raise ValueError("program budget counts must be positive")
        if self.max_wall_seconds <= 0:
            raise ValueError("program wall budget must be positive")
        if self.max_cost_usd is not None and self.max_cost_usd < 0:
            raise ValueError("program cost budget cannot be negative")


@dataclass(frozen=True, slots=True)
class ResearchContract:
    hypothesis_id: str
    claim: str
    baseline_artifact_sha256: str
    required_tests: tuple[str, ...]
    constraints: tuple[str, ...]
    primary_metric: str
    metric_direction: MetricDirection
    reward_lower_bound: float
    reward_upper_bound: float
    budget: ProgramBudget
    contract_sha256: str = ""

    def __post_init__(self) -> None:
        if not self.hypothesis_id.strip() or not self.claim.strip():
            raise ValueError("research contract needs a hypothesis and claim")
        if len(self.baseline_artifact_sha256) != 64:
            raise ValueError("research contract must bind the baseline artifact")
        if not self.required_tests or not self.primary_metric.strip():
            raise ValueError("research contract needs tests and a primary metric")
        if not all(math.isfinite(x) for x in (self.reward_lower_bound, self.reward_upper_bound)):
            raise ValueError("reward bounds must be finite")
        if self.reward_lower_bound >= self.reward_upper_bound:
            raise ValueError("reward bounds must have positive width")
        payload = asdict(self)
        payload.pop("contract_sha256", None)
        digest = canonical_sha256(payload)
        if self.contract_sha256 and self.contract_sha256 != digest:
            raise ValueError("research contract hash mismatch")
        object.__setattr__(self, "contract_sha256", digest)

    def normalize_reward(self, value: float) -> float:
        clipped = min(self.reward_upper_bound, max(self.reward_lower_bound, value))
        normalized = (clipped - self.reward_lower_bound) / (
            self.reward_upper_bound - self.reward_lower_bound
        )
        return normalized if self.metric_direction == MetricDirection.MAXIMIZE else 1.0 - normalized


@dataclass(frozen=True, slots=True)
class ProgramEvidence:
    primary_metric: float | None
    artifact_valid: bool
    contract_fidelity: float
    required_tests_passed: bool
    score_bearing: bool
    model_calls: int
    score_evaluations: int
    wall_seconds: float
    cost_usd: float = 0.0
    behavior_signature: str | None = None
    evidence_sha256: str = ""

    def __post_init__(self) -> None:
        values = (self.contract_fidelity, self.wall_seconds, self.cost_usd)
        if not all(math.isfinite(x) for x in values) or not 0 <= self.contract_fidelity <= 1:
            raise ValueError("invalid program evidence values")
        if min(self.model_calls, self.score_evaluations) < 0 or self.wall_seconds < 0 or self.cost_usd < 0:
            raise ValueError("program evidence resources cannot be negative")
        if self.primary_metric is not None and not math.isfinite(self.primary_metric):
            raise ValueError("primary metric must be finite")
        if self.score_bearing != (self.primary_metric is not None):
            raise ValueError("score-bearing evidence must carry exactly one primary metric")
        payload = asdict(self)
        payload.pop("evidence_sha256", None)
        digest = canonical_sha256(payload)
        if self.evidence_sha256 and self.evidence_sha256 != digest:
            raise ValueError("program evidence hash mismatch")
        object.__setattr__(self, "evidence_sha256", digest)


@dataclass(frozen=True, slots=True)
class ProgramParticle:
    particle_id: str
    program_sha256: str
    artifact_ref: str
    evidence: ProgramEvidence
    parent_particle_id: str | None = None
    kernel: MutationKernel | None = None
    stage: int = 0
    accepted: bool = True

    def __post_init__(self) -> None:
        if not self.particle_id or len(self.program_sha256) != 64 or not self.artifact_ref:
            raise ValueError("particle must bind an id, program hash, and artifact")
        if self.stage < 0:
            raise ValueError("particle stage cannot be negative")


@dataclass(frozen=True, slots=True)
class SMCConfig:
    particle_count: int = 4
    proposals_per_particle: int = 1
    target_beta: float = 8.0
    ess_fraction: float = 0.8
    min_stages: int = 3
    max_stages: int = 12
    minimum_contract_fidelity: float = 1.0
    fidelity_weight: float = 0.25
    cost_weight: float = 0.05
    max_inspirations: int = 3
    kernel_probabilities: tuple[float, float, float, float] = (0.35, 0.25, 0.20, 0.20)
    seed: int = 0

    def __post_init__(self) -> None:
        if min(self.particle_count, self.proposals_per_particle, self.min_stages) <= 0:
            raise ValueError("SMC population, chain, and stage counts must be positive")
        if self.max_stages < self.min_stages or self.target_beta <= 0:
            raise ValueError("invalid SMC stage or temperature configuration")
        if not 0 < self.ess_fraction <= 1:
            raise ValueError("ESS fraction must be in (0, 1]")
        if not 0 <= self.minimum_contract_fidelity <= 1:
            raise ValueError("minimum contract fidelity must be in [0, 1]")
        if min(self.fidelity_weight, self.cost_weight, self.max_inspirations) < 0:
            raise ValueError("SMC utility weights and inspiration count cannot be negative")
        if len(self.kernel_probabilities) != len(MutationKernel):
            raise ValueError("kernel probabilities must cover every mutation kernel")
        if any(item < 0 or not math.isfinite(item) for item in self.kernel_probabilities):
            raise ValueError("kernel probabilities must be finite and non-negative")
        if not math.isclose(sum(self.kernel_probabilities), 1.0, abs_tol=1e-9):
            raise ValueError("kernel probabilities must sum to one")


@dataclass(frozen=True, slots=True)
class MutationRequest:
    request_id: str
    contract_sha256: str
    stage: int
    chain_index: int
    proposal_index: int
    annealing_lambda: float
    kernel: MutationKernel
    parent_particle_id: str
    parent_program_sha256: str
    inspiration_particle_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class MutationResult:
    request_id: str
    candidate_program_sha256: str
    artifact_ref: str
    evidence: ProgramEvidence

    def __post_init__(self) -> None:
        if len(self.candidate_program_sha256) != 64 or not self.artifact_ref:
            raise ValueError("mutation result must bind a candidate artifact")


@dataclass(slots=True)
class _Chain:
    index: int
    current: ProgramParticle
    proposal_index: int = 0


class SMC:
    """Contract-frozen SMC budget allocator for executable program lineages.

    Scientific validation is deliberately outside this class. Only executable,
    score-bearing, contract-faithful evidence influences particle weights.
    """

    schema_version = "evolvcoder-smc-v1"

    def __init__(
        self,
        *,
        contract: ResearchContract,
        config: SMCConfig,
        initial_particles: tuple[ProgramParticle, ...],
        initialization_failures: tuple[ProgramEvidence, ...] = (),
    ) -> None:
        if len(initial_particles) != config.particle_count:
            raise ValueError("initial population must equal configured particle count")
        if any(not self._eligible_static(p.evidence, config) for p in initial_particles):
            raise ValueError("initial particles must be valid, tested, score-bearing programs")
        if len({p.program_sha256 for p in initial_particles}) != len(initial_particles):
            raise ValueError("initial particles must be program-distinct")
        self.contract = contract
        self.config = config
        self.random = random.Random(config.seed)
        self.population = list(initial_particles)
        self.history = list(initial_particles)
        self.stage = 0
        self.annealing_lambda = 0.0
        self.stop_reason: str | None = None
        self.events: list[dict[str, Any]] = []
        self._chains: list[_Chain] | None = None
        self._outstanding: MutationRequest | None = None
        charged = [p.evidence for p in initial_particles] + list(initialization_failures)
        self.model_calls = sum(item.model_calls for item in charged)
        self.score_evaluations = sum(item.score_evaluations for item in charged)
        self.wall_seconds = sum(item.wall_seconds for item in charged)
        self.cost_usd = sum(item.cost_usd for item in charged)
        self._append("smc.initialized", {
            "contract_sha256": contract.contract_sha256,
            "particle_ids": [p.particle_id for p in initial_particles],
            "failed_initialization_evidence_sha256s": [
                item.evidence_sha256 for item in initialization_failures
            ],
        })

    @staticmethod
    def _eligible_static(evidence: ProgramEvidence, config: SMCConfig) -> bool:
        return (
            evidence.artifact_valid and evidence.required_tests_passed
            and evidence.score_bearing
            and evidence.contract_fidelity >= config.minimum_contract_fidelity
        )

    def _eligible(self, evidence: ProgramEvidence) -> bool:
        return self._eligible_static(evidence, self.config)

    def _append(self, event_type: str, payload: dict[str, Any]) -> None:
        previous = self.events[-1]["event_sha256"] if self.events else None
        row = {
            "sequence": len(self.events), "event_type": event_type,
            "payload": payload, "previous_event_sha256": previous,
        }
        row["event_sha256"] = canonical_sha256(row)
        self.events.append(row)

    def stop(self, reason: str) -> None:
        """Stop at a controller-owned target or resource boundary."""
        if not reason.strip():
            raise ValueError("SMC stop reason cannot be empty")
        if self._outstanding is not None:
            raise RuntimeError("cannot stop with an outstanding mutation")
        if self.stop_reason is None:
            self.stop_reason = reason
            self._append("smc.stopped", {"reason": reason, "stage": self.stage})

    def utility(self, particle: ProgramParticle) -> float:
        evidence = particle.evidence
        if not self._eligible(evidence) or evidence.primary_metric is None:
            return -math.inf
        reward = self.contract.normalize_reward(evidence.primary_metric)
        cost_fraction = min(1.0, evidence.wall_seconds / self.contract.budget.max_wall_seconds)
        if self.contract.budget.max_cost_usd:
            cost_fraction = max(
                cost_fraction,
                min(1.0, evidence.cost_usd / self.contract.budget.max_cost_usd),
            )
        return (
            reward + self.config.fidelity_weight * evidence.contract_fidelity
            - self.config.cost_weight * cost_fraction
        )

    @staticmethod
    def effective_sample_size(weights: list[float]) -> float:
        total = sum(weights)
        squares = sum(x * x for x in weights)
        return 0.0 if total <= 0 or squares <= 0 else total * total / squares

    def _incremental_weights(self, candidate_lambda: float) -> list[float]:
        delta_beta = (candidate_lambda - self.annealing_lambda) * self.config.target_beta
        utilities = [self.utility(p) for p in self.population]
        finite = [u for u in utilities if math.isfinite(u)]
        if not finite:
            raise RuntimeError("SMC population contains no eligible particle")
        anchor = max(finite)
        raw = [math.exp(delta_beta * (u - anchor)) if math.isfinite(u) else 0.0 for u in utilities]
        total = sum(raw)
        return [x / total for x in raw]

    def _next_lambda(self) -> float:
        upper = min(1.0, self.annealing_lambda + 1.0 / self.config.min_stages)
        target = self.config.ess_fraction * len(self.population)
        if self.effective_sample_size(self._incremental_weights(upper)) >= target:
            return upper
        low, high = self.annealing_lambda, upper
        for _ in range(60):
            middle = (low + high) / 2
            if self.effective_sample_size(self._incremental_weights(middle)) >= target:
                low = middle
            else:
                high = middle
        return low

    def _systematic_resample(self, weights: list[float]) -> list[int]:
        count = len(weights)
        start = self.random.random() / count
        points = [start + index / count for index in range(count)]
        indices: list[int] = []
        cumulative = weights[0]
        cursor = 0
        for point in points:
            while point > cumulative and cursor < count - 1:
                cursor += 1
                cumulative += weights[cursor]
            indices.append(cursor)
        return indices

    def _budget_stop(self) -> str | None:
        budget = self.contract.budget
        if self.model_calls >= budget.max_model_calls:
            return "model_call_budget_exhausted"
        if self.score_evaluations >= budget.max_score_evaluations:
            return "score_evaluation_budget_exhausted"
        if self.wall_seconds >= budget.max_wall_seconds:
            return "wall_budget_exhausted"
        if budget.max_cost_usd is not None and self.cost_usd >= budget.max_cost_usd:
            return "cost_budget_exhausted"
        return None

    def _start_stage(self) -> None:
        if self.stop_reason is not None:
            return
        budget_stop = self._budget_stop()
        if budget_stop:
            self.stop_reason = budget_stop
            return
        if self.stage >= self.config.max_stages:
            self.stop_reason = "max_stages_reached"
            return
        next_lambda = self._next_lambda()
        weights = self._incremental_weights(next_lambda)
        ancestors = self._systematic_resample(weights)
        self.annealing_lambda = next_lambda
        self._chains = [
            _Chain(index=index, current=self.population[parent])
            for index, parent in enumerate(ancestors)
        ]
        self._append("smc.stage_started", {
            "stage": self.stage + 1,
            "annealing_lambda": self.annealing_lambda,
            "ess": self.effective_sample_size(weights),
            "weights": weights,
            "ancestor_particle_ids": [self.population[i].particle_id for i in ancestors],
        })

    def _select_kernel(self) -> MutationKernel:
        return self.random.choices(
            list(MutationKernel), weights=self.config.kernel_probabilities, k=1
        )[0]

    def _inspirations(self, parent: ProgramParticle) -> tuple[str, ...]:
        eligible = [
            p for p in self.history
            if p.program_sha256 != parent.program_sha256 and self._eligible(p.evidence)
        ]
        eligible.sort(key=lambda p: (self.utility(p), p.particle_id), reverse=True)
        selected: list[ProgramParticle] = []
        seen_behaviors: set[str | None] = set()
        for particle in eligible:
            signature = particle.evidence.behavior_signature
            if signature not in seen_behaviors or not selected:
                selected.append(particle)
                seen_behaviors.add(signature)
            if len(selected) >= self.config.max_inspirations:
                break
        return tuple(p.particle_id for p in selected)

    def next_request(self) -> MutationRequest | None:
        if self._outstanding is not None:
            raise RuntimeError("record the outstanding mutation before requesting another")
        if self.stop_reason is not None:
            return None
        if self._chains is None:
            self._start_stage()
            if self.stop_reason is not None:
                return None
        assert self._chains is not None
        chain = next(
            (c for c in self._chains if c.proposal_index < self.config.proposals_per_particle),
            None,
        )
        if chain is None:
            self._finish_stage()
            return self.next_request()
        kernel = self._select_kernel()
        inspiration_ids = self._inspirations(chain.current) if kernel.uses_inspiration else ()
        request = MutationRequest(
            request_id=f"smc-s{self.stage + 1}-c{chain.index}-k{chain.proposal_index}",
            contract_sha256=self.contract.contract_sha256,
            stage=self.stage + 1,
            chain_index=chain.index,
            proposal_index=chain.proposal_index,
            annealing_lambda=self.annealing_lambda,
            kernel=kernel,
            parent_particle_id=chain.current.particle_id,
            parent_program_sha256=chain.current.program_sha256,
            inspiration_particle_ids=inspiration_ids,
        )
        self._outstanding = request
        self._append("smc.mutation_requested", asdict(request))
        return request

    def record_result(self, result: MutationResult) -> bool:
        request = self._outstanding
        if request is None or result.request_id != request.request_id:
            raise ValueError("mutation result does not bind the outstanding request")
        assert self._chains is not None
        chain = self._chains[request.chain_index]
        if chain.current.particle_id != request.parent_particle_id:
            raise ValueError("mutation result parent is stale")
        self.model_calls += result.evidence.model_calls
        self.score_evaluations += result.evidence.score_evaluations
        self.wall_seconds += result.evidence.wall_seconds
        self.cost_usd += result.evidence.cost_usd
        duplicate = any(p.program_sha256 == result.candidate_program_sha256 for p in self.history)
        candidate = ProgramParticle(
            particle_id=f"{request.request_id}-{result.candidate_program_sha256[:10]}",
            program_sha256=result.candidate_program_sha256,
            artifact_ref=result.artifact_ref,
            evidence=result.evidence,
            parent_particle_id=chain.current.particle_id,
            kernel=request.kernel,
            stage=request.stage,
            accepted=False,
        )
        eligible = self._eligible(result.evidence) and not duplicate
        acceptance_probability = 0.0
        accepted = False
        if eligible:
            delta = self.utility(candidate) - self.utility(chain.current)
            acceptance_probability = min(
                1.0, math.exp(self.annealing_lambda * self.config.target_beta * min(0.0, delta))
            )
            accepted = self.random.random() <= acceptance_probability
        if accepted:
            candidate = replace(candidate, accepted=True)
            chain.current = candidate
        self.history.append(candidate)
        chain.proposal_index += 1
        self._outstanding = None
        self._append("smc.mutation_observed", {
            "request_id": request.request_id,
            "candidate_particle_id": candidate.particle_id,
            "candidate_program_sha256": candidate.program_sha256,
            "parent_particle_id": request.parent_particle_id,
            "inspiration_particle_ids": list(request.inspiration_particle_ids),
            "kernel": request.kernel.value,
            "eligible": eligible,
            "duplicate": duplicate,
            "accepted": accepted,
            "acceptance_probability": acceptance_probability,
            "evidence_sha256": result.evidence.evidence_sha256,
        })
        return accepted

    def _finish_stage(self) -> None:
        assert self._chains is not None
        self.population = [chain.current for chain in self._chains]
        self.stage += 1
        self._chains = None
        self._append("smc.stage_finished", {
            "stage": self.stage,
            "annealing_lambda": self.annealing_lambda,
            "particle_ids": [p.particle_id for p in self.population],
            "unique_programs": len({p.program_sha256 for p in self.population}),
        })
        budget_stop = self._budget_stop()
        if budget_stop:
            self.stop_reason = budget_stop
        elif self.stage >= self.config.min_stages and math.isclose(self.annealing_lambda, 1.0):
            self.stop_reason = "annealing_target_reached"
        elif self.stage >= self.config.max_stages:
            self.stop_reason = "max_stages_reached"
        if self.stop_reason:
            self._append("smc.stopped", {"reason": self.stop_reason, "stage": self.stage})

    def projection(self) -> dict[str, Any]:
        accepted = {p.particle_id for p in self.population}
        return {
            "schema_version": self.schema_version,
            "contract": {
                "hypothesis_id": self.contract.hypothesis_id,
                "claim": self.contract.claim,
                "contract_sha256": self.contract.contract_sha256,
                "primary_metric": self.contract.primary_metric,
            },
            "stage": self.stage,
            "annealing_lambda": self.annealing_lambda,
            "stop_reason": self.stop_reason,
            "population_particle_ids": [p.particle_id for p in self.population],
            "particles": [
                {
                    "particle_id": p.particle_id,
                    "program_sha256": p.program_sha256,
                    "artifact_ref": p.artifact_ref,
                    "parent_particle_id": p.parent_particle_id,
                    "kernel": p.kernel.value if p.kernel else None,
                    "stage": p.stage,
                    "accepted_in_current_population": p.particle_id in accepted,
                    "artifact_valid": p.evidence.artifact_valid,
                    "score_bearing": p.evidence.score_bearing,
                    "primary_metric": p.evidence.primary_metric,
                    "contract_fidelity": p.evidence.contract_fidelity,
                    "behavior_signature": p.evidence.behavior_signature,
                    "evidence_sha256": p.evidence.evidence_sha256,
                    "utility": self.utility(p) if self._eligible(p.evidence) else None,
                }
                for p in self.history
            ],
            "dependency_edges": [
                {"source": p.parent_particle_id, "target": p.particle_id,
                 "kind": "program_parent"}
                for p in self.history if p.parent_particle_id is not None
            ] + [
                {"source": inspiration_id,
                 "target": event["payload"]["candidate_particle_id"],
                 "kind": "program_inspiration"}
                for event in self.events if event["event_type"] == "smc.mutation_observed"
                for inspiration_id in event["payload"]["inspiration_particle_ids"]
            ],
            "kernel_mixture": {
                kernel.value: probability
                for kernel, probability in zip(MutationKernel, self.config.kernel_probabilities)
            },
            "accounting": {
                "model_calls": self.model_calls,
                "score_evaluations": self.score_evaluations,
                "wall_seconds": self.wall_seconds,
                "cost_usd": self.cost_usd,
            },
            "events": list(self.events),
        }

    @staticmethod
    def _particle_payload(particle: ProgramParticle) -> dict[str, Any]:
        payload = asdict(particle)
        payload["kernel"] = particle.kernel.value if particle.kernel else None
        return payload

    @staticmethod
    def _particle_from_payload(payload: dict[str, Any]) -> ProgramParticle:
        raw = dict(payload)
        raw["evidence"] = ProgramEvidence(**raw["evidence"])
        raw["kernel"] = MutationKernel(raw["kernel"]) if raw.get("kernel") else None
        return ProgramParticle(**raw)

    def checkpoint(self) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "schema_version": self.schema_version,
            "contract": asdict(self.contract),
            "config": asdict(self.config),
            "population_particle_ids": [p.particle_id for p in self.population],
            "history": [self._particle_payload(p) for p in self.history],
            "stage": self.stage,
            "annealing_lambda": self.annealing_lambda,
            "stop_reason": self.stop_reason,
            "events": self.events,
            "chains": None if self._chains is None else [
                {"index": chain.index, "current_particle_id": chain.current.particle_id,
                 "proposal_index": chain.proposal_index}
                for chain in self._chains
            ],
            "outstanding": asdict(self._outstanding) if self._outstanding else None,
            "resources": {
                "model_calls": self.model_calls,
                "score_evaluations": self.score_evaluations,
                "wall_seconds": self.wall_seconds,
                "cost_usd": self.cost_usd,
            },
            "random_state": self.random.getstate(),
        }
        payload["checkpoint_sha256"] = canonical_sha256(payload)
        return json.loads(json.dumps(payload, ensure_ascii=False, sort_keys=True))

    @classmethod
    def restore(cls, payload: dict[str, Any]) -> "SMC":
        raw = dict(payload)
        expected = raw.pop("checkpoint_sha256", None)
        if payload.get("schema_version") != cls.schema_version or expected != canonical_sha256(raw):
            raise ValueError("SMC checkpoint schema or hash mismatch")
        contract_raw = dict(raw["contract"])
        contract_raw["metric_direction"] = MetricDirection(contract_raw["metric_direction"])
        contract_raw["required_tests"] = tuple(contract_raw["required_tests"])
        contract_raw["constraints"] = tuple(contract_raw["constraints"])
        contract_raw["budget"] = ProgramBudget(**contract_raw["budget"])
        contract = ResearchContract(**contract_raw)
        config_raw = dict(raw["config"])
        config_raw["kernel_probabilities"] = tuple(config_raw["kernel_probabilities"])
        config = SMCConfig(**config_raw)
        value = cls.__new__(cls)
        value.contract = contract
        value.config = config
        value.random = random.Random()

        def tuple_tree(item: Any) -> Any:
            return tuple(tuple_tree(x) for x in item) if isinstance(item, list) else item

        value.random.setstate(tuple_tree(raw["random_state"]))
        value.history = [cls._particle_from_payload(item) for item in raw["history"]]
        by_id = {particle.particle_id: particle for particle in value.history}
        if len(by_id) != len(value.history):
            raise ValueError("SMC checkpoint contains duplicate particle ids")
        value.population = [by_id[item] for item in raw["population_particle_ids"]]
        value.stage = int(raw["stage"])
        value.annealing_lambda = float(raw["annealing_lambda"])
        value.stop_reason = raw["stop_reason"]
        value.events = list(raw["events"])
        value._chains = None if raw["chains"] is None else [
            _Chain(index=int(item["index"]), current=by_id[item["current_particle_id"]],
                   proposal_index=int(item["proposal_index"]))
            for item in raw["chains"]
        ]
        outstanding = raw["outstanding"]
        if outstanding is None:
            value._outstanding = None
        else:
            outstanding = dict(outstanding)
            outstanding["kernel"] = MutationKernel(outstanding["kernel"])
            outstanding["inspiration_particle_ids"] = tuple(
                outstanding["inspiration_particle_ids"]
            )
            value._outstanding = MutationRequest(**outstanding)
        resources = raw["resources"]
        value.model_calls = int(resources["model_calls"])
        value.score_evaluations = int(resources["score_evaluations"])
        value.wall_seconds = float(resources["wall_seconds"])
        value.cost_usd = float(resources["evidence_cost_usd"] if "evidence_cost_usd" in resources else resources["cost_usd"])
        return value
