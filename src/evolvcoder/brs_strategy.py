"""Bidirectional Research Search: hypothesis-program coevolution over the Tree.

One research episode is one bounded candidate attempt through the existing
Planner, Coder and evaluator. The allocator only chooses which frontier
initiates the next episode; the matcher only proposes which hypothesis is
implemented on which program with which typed sources. The continuation Tree
remains the program optimizer for program-initiated episodes and keeps its own
selector state; hypothesis-initiated episodes are validated here and never
enter the Tree selector.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, replace
import math
from typing import Any

from .brs_frontiers import (HypothesisFrontier, MATCHER_NAME, ProgramFrontier, capability_tags, family_sha256,
                            propose_pairs, tokens)
from .brs_models import (ARMS, EpisodeRecord, HypothesisRecord, PairProposal, PressureItem, Requirement,
                         episode_from_dict, pair_from_dict, pressure_from_dict, record_to_dict)
from .contracts import (ActionCandidate, ActionOperator, ProposalRejected, SearchDecision, SearchObservation, SearchSnapshot,
                        canonical_sha256)
from .frontier_allocator import FrontierPressureAllocator
from .strategies import ContinuationTreeStrategy, ProposalDirective, _available, _parent_id, _pending_repairs

EVALUATION_INFRASTRUCTURE_CATEGORIES = frozenset({"evaluation_bridge"})
SUPPORTED_OPTIMIZERS = ("continuation_tree",)


@dataclass(frozen=True, slots=True)
class ResearchDirective(ProposalDirective):
    """Tree directive plus a fixed hypothesis identity and the episode context.

    The base Tree directive keeps its byte-identical identity; consumers read
    these two fields through getattr so every other strategy is unaffected.
    """
    hypothesis_id: str | None = None
    research_episode: dict[str, Any] | None = None


def _finite(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(value)


MEMORY_SCOPES = frozenset({"full", "no_history", "no_failure_feedback"})
EVIDENCE_AUDITS = frozenset({"strict", "off"})


class BidirectionalResearchStrategy:
    name = "bidirectional_research"
    requires_checkpoint = True

    def __init__(self, *, seed: int = 0, program_optimizer: dict[str, Any] | None = None,
                 max_active_hypotheses: int = 4, max_program_representatives: int = 8,
                 max_pairs_per_hypothesis: int = 2, max_adaptation_sources: int = 1,
                 allocator: dict[str, Any] | None = None, matcher: dict[str, Any] | None = None,
                 matching_enabled: bool = True, instance_recombination_enabled: bool = True,
                 root_family_refinement_after: int = 2, reference_metric: float | None = None,
                 memory_scope: str = "full", rebase_requirements_enabled: bool = False,
                 adaptation_arms: tuple[str, ...] | list[str] = ("P",), evidence_audit: str = "strict",
                 incumbent_sweep_enabled: bool = False, max_sweeps_per_incumbent: int = 6,
                 sweep_budget_ratio: float = 3.0, min_total_evaluations_for_sweep: int = 40,
                 momentum_enabled: bool = False, keep_and_extend_enabled: bool = False,
                 consolidation_enabled: bool = False, consolidation_min_realized_families: int = 3,
                 consolidation_max_scored_children: int = 2, consolidation_failure_patience: int = 2,
                 consolidation_min_controller_delta: float = 0.0,
                 consolidation_exploration_floor: float = 0.2,
                 consolidation_require_internal_validation: bool = False,
                 initial_root_drafts: int = 0,
                 local_burst_counts_recombination: bool = False,
                 realization_diagnosis: bool = False,
                 frontier_arms: tuple[str, ...] | list[str] = ("H", "P")) -> None:
        if type(seed) is not int or seed < 0:
            raise ValueError("seed must be a nonnegative integer")
        optimizer = dict(program_optimizer or {"name": "continuation_tree", "parameters": {}})
        if set(optimizer) - {"name", "parameters"} or optimizer.get("name") not in SUPPORTED_OPTIMIZERS:
            raise ValueError(f"unsupported program optimizer; supported: {SUPPORTED_OPTIMIZERS}")
        parameters = dict(optimizer.get("parameters") or {})
        if initial_root_drafts:
            # Opt-in AIDE-style drafts: the first K episodes each write an independent
            # program from root before any lineage is refined. The drafting tree owns
            # that choice (its _choice is what select/observe re-check), and records
            # the draft count in its config, so the strategy identity changes only
            # when drafts are on.
            from .root_drafts import RootDraftingContinuationTree
            self.tree = RootDraftingContinuationTree(initial_root_drafts=initial_root_drafts, seed=seed, **parameters)
        else:
            self.tree = ContinuationTreeStrategy(seed=seed, **parameters)
        for flag, label in ((matching_enabled, "matching_enabled"),
                            (instance_recombination_enabled, "instance_recombination_enabled")):
            if type(flag) is not bool:
                raise ValueError(f"{label} must be a boolean")
        # Ablation switches. They belong to the recorded configuration identity so
        # an ablation cell can never be mistaken for the complete method.
        self.matching_enabled = matching_enabled
        self.instance_recombination_enabled = instance_recombination_enabled
        if type(root_family_refinement_after) is not int or root_family_refinement_after < 1:
            raise ValueError("root_family_refinement_after must be a positive integer")
        # Exit condition for the family-novelty rule (POET's minimal criterion has an
        # upper bound; ours had none): once this many new-family hypotheses have all
        # scored below the root while the root is still the incumbent, the next
        # hypothesis-initiated episode may refine inside the root's own family.
        self.root_family_refinement_after = root_family_refinement_after
        # Memory ablation, symmetric with the hypothesis-tree ablations in the
        # literature. The dynamic research graph is read by the planner as
        # context in every episode, independently of the matcher and of
        # instance recombination, so it can be removed on its own:
        #   full                  - the whole frontier: retained programs, the
        #                           hypothesis record, and the open requirements
        #                           distilled from past failures.
        #   no_history            - only the current incumbent is shown; the
        #                           controller keeps its bookkeeping but the
        #                           planner cannot read the accumulated graph
        #                           (the analogue of removing the tree).
        #   no_failure_feedback   - the graph is shown but failures are no
        #                           longer distilled into requirements, so past
        #                           failures stop steering later proposals
        #                           (the analogue of removing insight feedback).
        if memory_scope not in MEMORY_SCOPES:
            raise ValueError(f"memory_scope must be one of {sorted(MEMORY_SCOPES)}")
        self.memory_scope = memory_scope
        # Revision F. The paper defines a gap as scientific feasibility: a
        # hypothesis not yet testable on the programs at hand. Revisions A-D
        # only ever created requirements from failures, so on a run that goes
        # well the matcher had nothing to match (R36: zero pairs on all five
        # tasks; three tasks never produced a requirement). With rebase
        # requirements on, a hypothesis whose best realization sits on a base
        # that is no longer the incumbent gets one open requirement: realize it
        # on the incumbent. adaptation_arms says which arm may consume a pair;
        # revisions A-D consulted pairs only from the program arm, so the one
        # task that did hold a requirement (cross-modal) never saw it because
        # the allocator picked the hypothesis arm both times.
        if type(rebase_requirements_enabled) is not bool:
            raise ValueError("rebase_requirements_enabled must be a boolean")
        arms = tuple(adaptation_arms)
        if not arms or set(arms) - set(ARMS) or len(set(arms)) != len(arms):
            raise ValueError("adaptation_arms must be a non-empty subset of H/P without repeats")
        if evidence_audit not in EVIDENCE_AUDITS:
            raise ValueError(f"evidence_audit must be one of {sorted(EVIDENCE_AUDITS)}")
        self.rebase_requirements_enabled = rebase_requirements_enabled
        self.adaptation_arms = arms
        # Evidence ablation. "strict" is the E1 rule: a scored candidate whose
        # declared intervention never entered the executed path (unchanged
        # source, identical outputs) is not a test of its hypothesis and earns
        # no capability tags. "off" takes every scored candidate at its word.
        self.evidence_audit = evidence_audit
        # Revision H. Two rules about where the next scored evaluation goes.
        #
        # incumbent_sweep: the tissue task allows 109 scored evaluations and
        # R36 used 9, at 9.2 model calls each; the Claude reference used all
        # 109 at about 1 call each, most of them seed sweeps of one model, and
        # its own trace shows macro-F1 moving 0.49-0.62 across seeds. When the
        # remaining evaluation allowance is far larger than what has been used
        # and no local refinement lane is open, the program arm may spend one
        # evaluation on the incumbent with only its random seed or one numeric
        # hyperparameter changed, under a small coder budget. A sweep whose
        # outputs are identical to the parent proves the metric insensitive
        # and stops further sweeps on that incumbent.
        #
        # momentum: every win so far came from compounding on the program
        # arm. When the last settled program episode raised the incumbent and
        # the hypothesis arm holds nothing untried on the new incumbent, the
        # next decision stays on P instead of letting the pressure term pull
        # budget away (revision A's failure on peptide).
        for flag, label in ((incumbent_sweep_enabled, "incumbent_sweep_enabled"), (momentum_enabled, "momentum_enabled")):
            if type(flag) is not bool:
                raise ValueError(f"{label} must be a boolean")
        if type(max_sweeps_per_incumbent) is not int or max_sweeps_per_incumbent < 1:
            raise ValueError("max_sweeps_per_incumbent must be a positive integer")
        if not _finite(sweep_budget_ratio) or sweep_budget_ratio < 1:
            raise ValueError("sweep_budget_ratio must be a finite number >= 1")
        if type(min_total_evaluations_for_sweep) is not int or min_total_evaluations_for_sweep < 1:
            raise ValueError("min_total_evaluations_for_sweep must be a positive integer")
        # Sweeping only makes sense on tasks whose matched allowance is generous.
        # An offline replay of R36's sequences with the threshold at 12 showed
        # sweeps displacing the instance recombinations that won multimodal
        # (allowance 23), so the default admits only tissue-sized allowances
        # (109); peptide 5, SMLM 7, cross-modal 10 and multimodal 23 keep every
        # evaluation for real proposals.
        self.min_total_evaluations_for_sweep = min_total_evaluations_for_sweep
        self.incumbent_sweep_enabled = incumbent_sweep_enabled
        self.max_sweeps_per_incumbent = max_sweeps_per_incumbent
        self.sweep_budget_ratio = float(sweep_budget_ratio)
        self.momentum_enabled = momentum_enabled
        # Keep-and-extend. Every win recorded so far has the same shape: a new
        # hypothesis improves some instances and damages others (a PARTIAL GAIN);
        # a program-arm recombination restores the damaged instances from
        # whoever holds them while keeping the gain; the next hypothesis is then
        # proposed on that repaired base and jumps well past the reference.
        # Multimodal seed 0 is the clean case: 0.0425 partial gain, 0.0643 after
        # the repair, 0.2237 from the hypothesis built on it.
        #
        # Until now the repair only happened if the allocator happened to pick P
        # while sources existed, so it was frequently missed. With this rule the
        # cycle is mandatory and deterministic:
        #   K1 keep    partial gain      -> force P, recombine the damaged instances
        #   K2 extend  repair succeeded  -> force H on the repaired base
        # and the matching ablation removes exactly K1, so a partial gain can
        # never consolidate and later hypotheses build on a damaged base.
        if type(keep_and_extend_enabled) is not bool:
            raise ValueError("keep_and_extend_enabled must be a boolean")
        self.keep_and_extend_enabled = keep_and_extend_enabled
        # Evidence-Conditioned Lineage Consolidation (ECLC).  This is not a
        # scalar Top-K shortcut: a lane is armed only by an exact parent-child
        # gain with controller-owned score evidence and an observed executed
        # intervention.  It then receives a bounded number of same-hypothesis,
        # single-parent refinements before a mandatory exploration cooldown.
        for flag, label in ((consolidation_enabled, "consolidation_enabled"),
                            (consolidation_require_internal_validation,
                             "consolidation_require_internal_validation")):
            if type(flag) is not bool:
                raise ValueError(f"{label} must be a boolean")
        for value, label in ((consolidation_min_realized_families, "consolidation_min_realized_families"),
                             (consolidation_max_scored_children, "consolidation_max_scored_children"),
                             (consolidation_failure_patience, "consolidation_failure_patience")):
            if type(value) is not int or value < 1:
                raise ValueError(f"{label} must be a positive integer")
        if not _finite(consolidation_min_controller_delta) or consolidation_min_controller_delta < 0:
            raise ValueError("consolidation_min_controller_delta must be a finite number >= 0")
        if (not _finite(consolidation_exploration_floor)
                or not 0 < consolidation_exploration_floor <= .5):
            raise ValueError("consolidation_exploration_floor must be in (0, 0.5]")
        self.consolidation_enabled = consolidation_enabled
        self.consolidation_min_realized_families = consolidation_min_realized_families
        self.consolidation_max_scored_children = consolidation_max_scored_children
        self.consolidation_failure_patience = consolidation_failure_patience
        self.consolidation_min_controller_delta = float(consolidation_min_controller_delta)
        self.consolidation_exploration_floor = float(consolidation_exploration_floor)
        self.consolidation_require_internal_validation = consolidation_require_internal_validation
        # An instance recombination issued on an open tree_local slot is not
        # tree-owned, so the tree never observed it and its local burst never
        # advanced: the burst stayed open on the same program forever, every later
        # local slot became another recombination from that stale program, and the
        # allocator never received a decision. 02870 (drafts + consolidation, MLE
        # profile): 19 of 19 decisions forced to P, the H frontier never offered.
        # When on, such a recombination is charged to the burst exactly like a
        # tree_local attempt, so the burst ends on its own limits.
        if type(local_burst_counts_recombination) is not bool:
            raise ValueError("local_burst_counts_recombination must be a boolean")
        self.local_burst_counts_recombination = local_burst_counts_recombination
        # A scored child whose changed code produced byte-identical outputs gets a
        # requirement that says what that usually means and asks for the same idea to
        # be fixed, instead of the generic "not in the execution path".
        if type(realization_diagnosis) is not bool:
            raise ValueError("realization_diagnosis must be a boolean")
        self.realization_diagnosis = realization_diagnosis
        # Ablation switch for the dual frontier. ("P",) is the single-frontier
        # arm: the allocator may only start program-initiated episodes, so no
        # hypothesis-initiated episode (new or activated hypothesis, H-side
        # adaptation) runs; drafts, local continuations and program-side pairing
        # are unchanged. The default keeps both frontiers.
        frontiers = tuple(frontier_arms)
        if not frontiers or set(frontiers) - set(ARMS) or len(set(frontiers)) != len(frontiers):
            raise ValueError("frontier_arms must be a non-empty subset of H/P without repeats")
        self.frontier_arms = frontiers
        self.consolidation_lineages: dict[str, dict[str, Any]] = {}
        self.consolidation_by_action: dict[str, str] = {}
        self.consolidation_consecutive = 0
        self.consolidation_cooldown_remaining = 0
        self.total_score_evaluations: int | None = None
        self.sweeps: dict[str, dict[str, Any]] = {}   # incumbent id -> {"count": n, "insensitive": bool}
        if reference_metric is not None and (type(reference_metric) not in (int, float) or not math.isfinite(reference_metric)):
            raise ValueError("reference_metric must be a finite number or None")
        # The public target the run stops at; with reward_scale_mode="reference_gap" the
        # allocator scores each step by the fraction of the root-to-reference gap it closes.
        self.reference_metric = None if reference_metric is None else float(reference_metric)
        matcher = dict(matcher or {"name": MATCHER_NAME})
        if matcher != {"name": MATCHER_NAME}:
            raise ValueError(f"unsupported matcher {matcher!r}; only {MATCHER_NAME} is implemented "
                             "(unbalanced optimal transport is not a supported configuration)")
        for value, label in ((max_pairs_per_hypothesis, "max_pairs_per_hypothesis"),
                             (max_adaptation_sources, "max_adaptation_sources")):
            if type(value) is not int or value < 0 or (label == "max_pairs_per_hypothesis" and value < 1):
                raise ValueError(f"{label} must be a nonnegative integer")
        self.allocator = FrontierPressureAllocator(**dict(allocator or {}))
        self.hypotheses = HypothesisFrontier(max_active=max_active_hypotheses)
        self.programs = ProgramFrontier(max_representatives=max_program_representatives)
        # parent|sources keys already offered as an instance recombination; each
        # combination is proposed once so the lane stays bounded without the
        # continuation tree having to observe these episodes.
        self.recombinations: set[str] = set()
        self.max_pairs_per_hypothesis = max_pairs_per_hypothesis
        self.max_adaptation_sources = max_adaptation_sources
        self.config = {
            "policy": self.name, "seed": seed,
            "program_optimizer": {"name": optimizer["name"], "parameters": dict(self.tree.config)},
            "max_active_hypotheses": max_active_hypotheses,
            "max_program_representatives": max_program_representatives,
            "max_pairs_per_hypothesis": max_pairs_per_hypothesis,
            "max_adaptation_sources": max_adaptation_sources,
            "allocator": dict(self.allocator.config), "matcher": {"name": MATCHER_NAME},
            "matching_enabled": matching_enabled,
            "instance_recombination_enabled": instance_recombination_enabled,
            "root_family_refinement_after": root_family_refinement_after,
            "reference_metric": self.reference_metric,
            "memory_scope": memory_scope,
            "rebase_requirements_enabled": rebase_requirements_enabled,
            "adaptation_arms": list(arms),
            "evidence_audit": evidence_audit,
            "incumbent_sweep_enabled": incumbent_sweep_enabled,
            "max_sweeps_per_incumbent": max_sweeps_per_incumbent,
            "sweep_budget_ratio": float(sweep_budget_ratio),
            "min_total_evaluations_for_sweep": min_total_evaluations_for_sweep,
            "momentum_enabled": momentum_enabled,
            "keep_and_extend_enabled": keep_and_extend_enabled,
            "consolidation_enabled": consolidation_enabled,
            "consolidation_min_realized_families": consolidation_min_realized_families,
            "consolidation_max_scored_children": consolidation_max_scored_children,
            "consolidation_failure_patience": consolidation_failure_patience,
            "consolidation_min_controller_delta": float(consolidation_min_controller_delta),
            "consolidation_exploration_floor": float(consolidation_exploration_floor),
            "consolidation_require_internal_validation": consolidation_require_internal_validation,
        }
        # Recorded only when on, so existing configurations keep their identity.
        if local_burst_counts_recombination:
            self.config["local_burst_counts_recombination"] = True
        if realization_diagnosis:
            self.config["realization_diagnosis"] = True
        if set(self.frontier_arms) != set(ARMS):
            self.config["frontier_arms"] = list(self.frontier_arms)
        self.observations: list[SearchObservation] = []
        self.node_info: dict[str, dict[str, Any]] = {}
        self.episodes: dict[str, EpisodeRecord] = {}
        self.episode_by_action: dict[str, str] = {}
        self.pairs: dict[str, PairProposal] = {}
        self.pressure: dict[str, PressureItem] = {}
        self.pending_episode_id: str | None = None
        self.next_directive: dict[str, Any] | None = None
        self.hypothesis_alias: dict[str, str] = {}
        self._events: list[dict[str, Any]] = []

    @staticmethod
    def learnable_parameters():
        """The only construction values an outer learner may change.

        Both live in the allocator: how fast old outcomes are discounted, and
        how strongly an unmet frontier can move the next initiating arm. Reward,
        metric, scale, failure classification, capacities, matcher, data,
        Wisdom and every hard budget stay fixed.
        """
        from .search_parameters import ParameterSpec
        return (ParameterSpec("discount_factor", ("allocator", "discount_factor"), "number", 0.5, 1.0),
                ParameterSpec("frontier_pressure_weight", ("allocator", "frontier_pressure_weight"), "number", 0.0, 2.0))

    # ------------------------------------------------------------------ events
    def _emit(self, kind: str, payload: dict[str, Any]) -> None:
        self._events.append({"event_type": "search.frontier." + kind, "payload": record_to_dict(payload)})

    def drain_events(self) -> list[dict[str, Any]]:
        events, self._events = self._events, []
        return events

    # ----------------------------------------------------------- node/incumbent
    def _sync_nodes(self, snapshot: SearchSnapshot) -> None:
        for node_id, valid in snapshot.node_artifact_validity.items():
            if node_id in self.node_info:
                continue
            if node_id != "root":
                raise ValueError(f"program node {node_id} is not bound to a BRS episode")
            self.node_info[node_id] = {
                "hypothesis_id": "h_root", "statement": "Frozen starter program", "family": None,
                "family_sha256": None, "capabilities": ["frozen_starter"], "realization": "starter",
                "parent_id": None, "episode_id": None, "arm": None,
                "primary_scores": self._primary_scores(snapshot.node_evaluation_summaries.get(node_id)),
                "native_metric": snapshot.node_metrics.get(node_id),
                "executable_sha256": snapshot.node_executable_sha256s.get(node_id),
            }
        for node_id, info in self.node_info.items():
            if info.get("executable_sha256") is None and node_id in snapshot.node_executable_sha256s:
                info["executable_sha256"] = snapshot.node_executable_sha256s[node_id]

    def _valid_nodes(self, snapshot: SearchSnapshot) -> dict[str, dict[str, Any]]:
        valid: dict[str, dict[str, Any]] = {}
        for node_id, ok in snapshot.node_artifact_validity.items():
            metric = snapshot.node_metrics.get(node_id)
            info = self.node_info.get(node_id)
            if ok and _finite(metric) and info is not None and info.get("executable_sha256"):
                summary = (snapshot.node_evaluation_summaries or {}).get(node_id) or {}
                per_instance = summary.get("per_instance_improvement") if isinstance(summary, dict) else None
                valid[node_id] = {"native_metric": float(metric), "executable_sha256": info["executable_sha256"],
                                  "family_sha256": info.get("family_sha256"), "hypothesis_id": info["hypothesis_id"],
                                  "capabilities": tuple(info.get("capabilities", ())),
                                  "per_instance": dict(per_instance) if isinstance(per_instance, dict) else {}}
        return valid

    @staticmethod
    def _incumbent(snapshot: SearchSnapshot) -> tuple[float | None, str | None]:
        best: tuple[float, str] | None = None
        for node_id, metric in snapshot.node_metrics.items():
            if snapshot.node_artifact_validity.get(node_id) and _finite(metric):
                candidate = (float(metric), node_id)
                if best is None or candidate[0] > best[0] or (candidate[0] == best[0] and node_id < best[1]):
                    best = candidate
        return (None, None) if best is None else best

    # -------------------------------------------------------------- pressure
    def _refresh_pressure(self, step: int) -> dict[str, int]:
        representatives = self.programs.representatives
        for record in self.hypotheses.records.values():
            if record.status == "archived":
                continue
            for requirement in record.open_requirements():
                # A rebase gap is consumed by whichever arm may adapt; the
                # pressure it exerts must point at that arm. Side "H" pushes the
                # allocator toward P, side "P" toward H.
                side = "P" if (requirement.kind == "rebase" and "P" not in self.adaptation_arms) else "H"
                key = canonical_sha256({"side": side, "hypothesis": record.id, "requirement": requirement.key})[:24]
                if key not in self.pressure and representatives:
                    self.pressure[key] = PressureItem(key=key, side=side, ref=record.id, description=requirement.description,
                                                      next_step=requirement.next_step, created_step=step,
                                                      evidence_refs=tuple(x for x in (requirement.source_action_id,) if x))
                    self._emit("pressure_queued", {"item": self.pressure[key]})
        counts = {arm: 0 for arm in ARMS}
        for item in self.pressure.values():
            if item.status == "queued":
                counts[item.side] += 1
        return counts

    def _set_pressure(self, key: str, status: str, episode_id: str | None = None) -> None:
        item = self.pressure.get(key)
        if item is not None and item.status != status:
            self.pressure[key] = replace(item, status=status, consumed_by_episode_id=episode_id or item.consumed_by_episode_id)
            self._emit("pressure_" + status, {"key": key, "side": item.side, "ref": item.ref, "episode_id": episode_id})

    def _pressure_keys(self, side: str, ref: str, *, statuses: tuple[str, ...] = ("queued", "pending")) -> list[str]:
        return [k for k, v in self.pressure.items() if v.side == side and v.ref == ref and v.status in statuses]

    # ------------------------------------------------------------- directives
    @staticmethod
    def _tree_owned(mode: str) -> bool:
        return mode.startswith("tree_")

    def _gap_pairs(self, snapshot: SearchSnapshot) -> list[PairProposal]:
        if not self.matching_enabled:
            # Matching ablation: the program arm keeps its own lane and never
            # adapts a hypothesis onto a different program base.
            return []
        excluded = {k for k, p in self.pairs.items() if p.status in {"selected", "attempted"}}
        measured = {e.primary_program_id: float(e.planning_tokens + e.coder_tokens)
                    for e in self.episodes.values() if e.status == "settled"}
        waiting = {r.id: self.hypotheses.waiting(r, snapshot.step) for r in self.hypotheses.records.values()}
        proposals = propose_pairs(hypotheses=self.hypotheses.active(), representatives=self.programs.representatives,
                                  excluded_keys=excluded, max_per_hypothesis=self.max_pairs_per_hypothesis,
                                  max_sources=self.max_adaptation_sources, measured_costs=measured,
                                  step=snapshot.step, waiting=waiting)
        for proposal in proposals:
            if proposal.key not in self.pairs:
                self.pairs[proposal.key] = proposal
                self._emit("pair_proposed", {"pair": proposal})
        return proposals

    def _keep_and_extend_phase(self) -> tuple[str, str] | None:
        """K1/K2: consolidate a partial gain, then extend from the repaired base.

        Returns the arm the cycle demands, or None when neither rule applies
        (the last episode dominated, was dominated, or produced no score).
        """
        settled = [e for e in self.episodes.values() if e.status == "settled" and e.native_metric is not None]
        if not settled:
            return None
        last = max(settled, key=lambda e: (e.settled_step or -1, e.started_step))
        if last.mode == "instance_recombination":
            # K2: the repair has been executed; extend from whatever it produced.
            return ("H", "keep_and_extend_extend")
        child = (self.node_info.get(last.action_id or "") or {}).get("primary_scores")
        if child is None:
            child = next((info.get("primary_scores") for info in self.node_info.values()
                          if info.get("episode_id") == last.id), None)
        parent = (self.node_info.get(last.primary_program_id) or {}).get("primary_scores")
        improved, degraded = self._differential(child, parent)
        if improved and degraded:
            # K1: a partial gain must be consolidated before anything else.
            return ("P", "keep_and_extend_keep")
        return None

    def _program_momentum(self, incumbent_id: str | None) -> bool:
        """The last settled program episode raised the incumbent and H has nothing untried on it."""
        settled = [e for e in self.episodes.values() if e.status == "settled"]
        if not settled or incumbent_id is None:
            return False
        last = max(settled, key=lambda e: e.settled_step or -1)
        if last.arm != "P" or not last.reward or last.reward <= 0:
            return False
        untried = [r for r in self.hypotheses.active()
                   if incumbent_id not in r.program_ids
                   and not any(e.hypothesis_id == r.id and e.primary_program_id == incumbent_id
                               for e in self.episodes.values())]
        return not untried

    @staticmethod
    def _primary_scores(summary: dict[str, Any] | None) -> dict[str, Any] | None:
        """Read component scores from both native and external-domain evidence."""
        if not isinstance(summary, dict):
            return None
        scores = summary.get("primary_scores")
        if isinstance(scores, dict):
            return scores
        details = summary.get("details")
        scores = details.get("primary_scores") if isinstance(details, dict) else None
        return scores if isinstance(scores, dict) else None

    @staticmethod
    def _internal_validation_evidence(observation: SearchObservation) -> dict[str, Any]:
        """Classify candidate-owned validation without treating self-report as score truth."""
        gate = observation.internal_validation
        if not isinstance(gate, dict):
            return {"comparable": False, "reason": "missing"}
        if gate.get("evidence_authority") == "candidate_unattested":
            return {"comparable": False, "reason": "unattested_candidate_estimate"}
        candidate, parent = gate.get("candidate"), gate.get("parent")
        if not isinstance(candidate, dict) or not isinstance(parent, dict):
            return {"comparable": False, "reason": "missing_parent"}
        direction = candidate.get("direction")
        same = (candidate.get("metric_name") == parent.get("metric_name")
                and direction in {"maximize", "minimize"}
                and direction == parent.get("direction")
                and bool(candidate.get("protocol"))
                and candidate.get("protocol") == parent.get("protocol"))
        left, right = candidate.get("estimate"), parent.get("estimate")
        if not same or not _finite(left) or not _finite(right):
            return {"comparable": False, "reason": "protocol_mismatch"}
        sign = 1.0 if direction == "maximize" else -1.0
        delta = sign * (float(left) - float(right))
        return {"comparable": True, "reason": "matched_protocol", "delta": delta,
                "metric_name": candidate.get("metric_name"), "protocol": candidate.get("protocol")}

    def _max_consecutive_consolidations(self) -> int:
        # At floor f, at least one of every ceil(1/f) decisions is exploration.
        return max(1, math.ceil(1.0 / self.consolidation_exploration_floor) - 1)

    def _eligible_consolidation(self, snapshot: SearchSnapshot) -> tuple[str, dict[str, Any]] | None:
        if not self.consolidation_enabled:
            return None
        if len(snapshot.realized_family_sha256s) < self.consolidation_min_realized_families:
            return None
        if self.consolidation_cooldown_remaining > 0:
            self.consolidation_cooldown_remaining -= 1
            self.consolidation_consecutive = 0
            self._emit("consolidation_exploration", {"step": snapshot.step, "reason": "cooldown"})
            return None
        if self.consolidation_consecutive >= self._max_consecutive_consolidations():
            self.consolidation_cooldown_remaining = 0
            self.consolidation_consecutive = 0
            self._emit("consolidation_exploration", {"step": snapshot.step, "reason": "exploration_floor"})
            return None
        eligible = []
        for key, lane in self.consolidation_lineages.items():
            if lane.get("status") not in {"armed", "active"}:
                continue
            if lane.get("scored_children", 0) >= self.consolidation_max_scored_children:
                continue
            if lane.get("consecutive_failures", 0) >= self.consolidation_failure_patience:
                continue
            parent = lane.get("current_parent_id")
            if not parent or not snapshot.node_artifact_validity.get(parent):
                continue
            eligible.append((key, lane))
        if not eligible:
            return None
        return max(eligible, key=lambda item: (
            int(item[1].get("evidence_rank", 0)), float(item[1].get("best_metric", -math.inf)),
            int(item[1].get("last_step", -1)), item[0]))

    def _lineage_history(self, lane: dict[str, Any]) -> list[dict[str, Any]]:
        return list(lane.get("history", ()))[-3:]

    def _promote_closed_consolidation(self, key: str, lane: dict[str, Any], *, step: int) -> None:
        """Release a stale Tree burst onto the best verified lineage descendant.

        Consolidation episodes are deliberately owned by BRS rather than by the
        continuation Tree.  Without an explicit handoff, however, the Tree keeps
        its pre-consolidation local burst open and immediately proposes the next
        action from the old anchor.  That silently discards the very improvement
        consolidation just established.  Close only the burst that belongs to
        this lineage and replace its retained best with the controller-scored
        descendant; the Tree's next global choice can then rank that descendant
        from the authoritative snapshot.
        """
        burst = getattr(self.tree, "burst", None)
        if not isinstance(burst, dict):
            return
        lineage_nodes = {
            str(node_id) for node_id in (
                lane.get("anchor_node_id"), lane.get("best_node_id"),
                lane.get("current_parent_id"),
                *(item.get("node_id") for item in lane.get("history", ())
                  if isinstance(item, dict)),
            ) if node_id
        }
        if burst.get("best_id") not in lineage_nodes:
            return
        previous = burst.get("best_id")
        burst["best_id"] = lane["best_node_id"]
        burst["best_metric"] = lane["best_metric"]
        burst["closed"] = True
        self._emit("consolidation_promoted", {
            "lineage_id": key, "previous_tree_parent_id": previous,
            "promoted_parent_id": lane["best_node_id"], "step": step,
        })

    def _retire_rejected_consolidation(self, key: str, lane: dict[str, Any],
                                       action: ActionCandidate, step: int) -> None:
        """Close a lane whose planner refused its bound hypothesis/family.

        Only the lane is touched: the refused action is never bound to it and
        no hypothesis or episode records it.  Its verified best is handed back
        to the Tree exactly as a normal close would.
        """
        lane.update(status="retired_rejected", rejected_action_id=action.id, last_step=step)
        self.next_directive = None
        self._promote_closed_consolidation(key, lane, step=step)
        self._emit("consolidation_retired", {"lineage_id": key, "rejected_action_id": action.id,
                                              "best_node_id": lane.get("best_node_id"), "step": step})

    def _consolidation_directive(self, key: str, lane: dict[str, Any],
                                 base_context: dict[str, Any]) -> ResearchDirective:
        parent_id = str(lane["current_parent_id"])
        parent_info = self.node_info.get(parent_id, {})
        history = self._lineage_history(lane)
        episode = {
            **base_context, "arm": "P", "mode": "evidence_consolidation", "lineage_id": key,
            "hypothesis_id": lane["hypothesis_id"],
            "hypothesis": {"id": lane["hypothesis_id"], "statement": parent_info.get("statement"),
                           "family": parent_info.get("family")},
            "parent_hypothesis": {"id": lane["hypothesis_id"], "statement": parent_info.get("statement"),
                                  "family": parent_info.get("family")},
            "consolidation": {
                "anchor_node_id": lane["anchor_node_id"], "best_node_id": lane["best_node_id"],
                "scored_children": lane.get("scored_children", 0),
                "max_scored_children": self.consolidation_max_scored_children,
                "consecutive_failures": lane.get("consecutive_failures", 0),
                "failure_patience": self.consolidation_failure_patience,
                "evidence_tier": lane.get("evidence_tier"),
                "controller_delta": lane.get("controller_delta"),
                "trajectory": history,
            },
            "guidance": (
                "Evidence-conditioned lineage consolidation: continue the verified parent under the SAME hypothesis "
                "and scientific family. Make exactly ONE localized capacity/control or implementation refinement; "
                "do not replace the primary operator, input transform, output construction, objective, data split, "
                "metric, or random seed. Preserve the verified components listed by the trajectory, explicitly name "
                "the one changed component and the observed weakness it addresses, and leave every other component "
                "unchanged. A regressing child is diagnostic only and never replaces the best ancestor. "
                "Keep the hypothesis under 50 words and description under 180 words."
            ),
        }
        self._emit("consolidation_selected", {"lineage_id": key, "parent_id": parent_id,
                                               "history": history, "step": base_context.get("step")})
        return ResearchDirective(code_parent_id=parent_id, hypothesis_id=str(lane["hypothesis_id"]),
                                 research_episode=episode,
                                 rationale_codes=("brs_program_initiated", "brs_evidence_consolidation",
                                                  "single_component_refinement"))

    def _arm_consolidation(self, observation: SearchObservation, episode: EpisodeRecord,
                           *, realization: str, step: int) -> None:
        """Create or update a bounded lane from an exact, evidence-bound gain."""
        if not self.consolidation_enabled or not observation.program_node_id:
            return
        parent_metric = next((e.native_metric for e in self.episodes.values()
                              if e.action_id == episode.primary_program_id and e.native_metric is not None), None)
        if parent_metric is None:
            # Program ids are action ids in the coordinator, but root is not an episode.
            parent_metric = (self.node_info.get(episode.primary_program_id) or {}).get("native_metric")
        if not _finite(parent_metric):
            parent_metric = episode.incumbent_pre if episode.incumbent_pre_node_id == episode.primary_program_id else None
        metric = observation.native_metric
        controller_delta = (float(metric) - float(parent_metric)) if _finite(metric) and _finite(parent_metric) else None
        internal = self._internal_validation_evidence(observation)
        evidence_ok = (observation.score_bearing and observation.artifact_valid and realization == "observed"
                       and observation.evaluation_evidence_sha256 is not None
                       and _finite(controller_delta)
                       and controller_delta > self.consolidation_min_controller_delta)
        if self.consolidation_require_internal_validation:
            evidence_ok = evidence_ok and internal.get("comparable") and internal.get("delta", 0) > 0
        if not evidence_ok:
            return
        existing_key = self.consolidation_by_action.get(observation.action_id)
        if existing_key and existing_key in self.consolidation_lineages:
            return
        key = observation.program_node_id
        tier = "controller_plus_internal" if internal.get("comparable") and internal.get("delta", 0) > 0 else "controller_paired"
        self.consolidation_lineages[key] = {
            "status": "armed", "anchor_node_id": key, "current_parent_id": key, "best_node_id": key,
            "best_metric": float(metric), "hypothesis_id": episode.hypothesis_id,
            "family_sha256": observation.realized_family_sha256, "scored_children": 0,
            "consecutive_failures": 0, "implementation_failures": 0, "created_step": step,
            "last_step": step, "evidence_tier": tier, "evidence_rank": 2 if tier == "controller_plus_internal" else 1,
            "controller_delta": controller_delta, "internal_validation": internal,
            "history": [{"node_id": key, "parent_id": episode.primary_program_id, "outcome": "gain",
                         "metric": float(metric), "controller_delta": controller_delta,
                         "evidence_sha256": observation.evaluation_evidence_sha256}],
        }
        self._emit("consolidation_armed", {"lineage_id": key, "evidence_tier": tier,
                                           "controller_delta": controller_delta, "step": step})

    def _settle_consolidation(self, observation: SearchObservation, episode: EpisodeRecord,
                              *, realization: str, step: int) -> None:
        key = self.consolidation_by_action.get(observation.action_id)
        if key is None and episode.origin_episode_id:
            origin = self.episodes.get(episode.origin_episode_id)
            key = self.consolidation_by_action.get(origin.action_id or "") if origin else None
        lane = self.consolidation_lineages.get(key or "")
        if lane is None:
            return
        lane = dict(lane)
        lane["last_step"] = step
        history = list(lane.get("history", ()))
        if (observation.score_bearing and observation.artifact_valid and realization == "observed"
                and observation.program_node_id and _finite(observation.native_metric)):
            parent_metric = lane.get("best_metric") if episode.primary_program_id == lane.get("best_node_id") else None
            if parent_metric is None:
                parent_metric = next((e.native_metric for e in self.episodes.values()
                                      if e.action_id == episode.primary_program_id), None)
            delta = float(observation.native_metric) - float(parent_metric) if _finite(parent_metric) else -math.inf
            lane["scored_children"] = int(lane.get("scored_children", 0)) + 1
            if delta > self.consolidation_min_controller_delta:
                lane.update(current_parent_id=observation.program_node_id, best_node_id=observation.program_node_id,
                            best_metric=float(observation.native_metric), consecutive_failures=0, status="active")
                outcome = "gain"
            else:
                lane["consecutive_failures"] = int(lane.get("consecutive_failures", 0)) + 1
                outcome = "no_gain"
            history.append({"node_id": observation.program_node_id, "parent_id": episode.primary_program_id,
                            "outcome": outcome, "metric": float(observation.native_metric),
                            "controller_delta": delta, "evidence_sha256": observation.evaluation_evidence_sha256})
        else:
            lane["consecutive_failures"] = int(lane.get("consecutive_failures", 0)) + 1
            if not observation.executable or not observation.artifact_valid:
                lane["implementation_failures"] = int(lane.get("implementation_failures", 0)) + 1
            history.append({"node_id": None, "parent_id": episode.primary_program_id,
                            "outcome": ("implementation_failed" if not observation.executable or not observation.artifact_valid
                                        else "unscored"), "metric": None,
                            "controller_delta": None, "evidence_sha256": observation.evaluation_evidence_sha256})
        lane["history"] = history[-6:]
        close = (int(lane.get("scored_children", 0)) >= self.consolidation_max_scored_children
                 or int(lane.get("consecutive_failures", 0)) >= self.consolidation_failure_patience)
        if close:
            lane["status"] = "closed"
            self.consolidation_cooldown_remaining = 1
            self.consolidation_consecutive = 0
            self._promote_closed_consolidation(str(key), lane, step=step)
            self._emit("consolidation_closed", {"lineage_id": key, "best_node_id": lane["best_node_id"],
                                                 "scored_children": lane["scored_children"],
                                                 "consecutive_failures": lane["consecutive_failures"], "step": step})
        else:
            lane["status"] = "active"
        self.consolidation_lineages[str(key)] = lane

    def _sweep_admissible(self, snapshot: SearchSnapshot, incumbent_id: str | None) -> bool:
        if incumbent_id is None or self.total_score_evaluations is None:
            return False
        if self.total_score_evaluations < self.min_total_evaluations_for_sweep:
            return False
        # An uncapped evaluation budget arrives here as infinity, and int() of that
        # raises. The sweep exists to spend an allowance that would otherwise go
        # unused; with no allowance there is nothing to reclaim, so it does not apply.
        if not math.isfinite(snapshot.budget.remaining_score_evaluations):
            return False
        remaining = int(snapshot.budget.remaining_score_evaluations)
        used = max(1, self.total_score_evaluations - remaining)
        if remaining < 4 or remaining < self.sweep_budget_ratio * used:
            return False
        state = self.sweeps.get(incumbent_id, {"count": 0, "insensitive": False})
        if state["insensitive"] or state["count"] >= self.max_sweeps_per_incumbent:
            return False
        burst = getattr(self.tree, "burst", None)
        if burst and not burst.get("closed") and burst.get("stagnation", 0) < self.tree.config["stagnation_patience"]:
            return False   # a local refinement lane is still live; it goes first
        return True

    def _sweep_directive(self, incumbent_id: str, incumbent_metric: float | None,
                         base_context: dict[str, Any]) -> ResearchDirective:
        state = self.sweeps.setdefault(incumbent_id, {"count": 0, "insensitive": False})
        state["count"] += 1
        parent_info = self.node_info.get(incumbent_id, {})
        episode = {**base_context, "arm": "P", "mode": "incumbent_sweep",
                   "parent_hypothesis": {"id": parent_info.get("hypothesis_id"), "statement": parent_info.get("statement"),
                                         "family": parent_info.get("family")},
                   "sweep": {"index": state["count"], "max": self.max_sweeps_per_incumbent,
                             "incumbent_metric": incumbent_metric},
                   "coder_limits": {"model_calls": 3, "tool_calls": 6},
                   "guidance": ("Program-initiated SWEEP episode on the incumbent: the scored evaluation "
                                "allowance is far from exhausted and no local refinement lane is open. Propose "
                                "operator improve_program with the SAME hypothesis and family as the parent, and "
                                "change EXACTLY ONE thing: the random seed constant(s), or one numeric "
                                "hyperparameter (cluster count, neighbourhood size, regularisation, component "
                                "count). No other edits; no new components; no data or metric change. The coder "
                                "budget for this episode is tiny: state the single change in one sentence. "
                                "Keep the proposal compact: hypothesis under 30 words, description under 60 words.")}
        self._emit("incumbent_sweep", {"incumbent_id": incumbent_id, "index": state["count"]})
        return ResearchDirective(code_parent_id=incumbent_id, research_episode=episode,
                                 rationale_codes=("brs_program_initiated", "brs_incumbent_sweep", "tree_local"))

    def _pair_directive(self, proposal: PairProposal, base_context: dict[str, Any], arm: str) -> tuple[ResearchDirective, str]:
        pair = replace(proposal, status="selected")
        self.pairs[pair.key] = pair
        record = self.hypotheses.get(pair.hypothesis_id)
        assert record is not None
        initiated = "Program-initiated" if arm == "P" else "Hypothesis-initiated"
        episode = {**base_context, "arm": arm, "mode": "gap_adaptation", "hypothesis_id": record.id,
                   "hypothesis": self._hypothesis_context(record), "pair": record_to_dict(pair),
                   "sources": [record_to_dict(self.programs.get(s)) for s in pair.source_program_ids if self.programs.get(s)],
                   "guidance": (initiated + " episode closing a recorded requirement: the hypothesis is fixed "
                                "(hypothesis_id, statement, family). Implement it on the selected code parent, whose "
                                "executed components were matched to the open requirement; listed source programs are "
                                "typed inspiration only. State which requirement the plan closes. Coverage labels are "
                                "controller bookkeeping, not verdicts. " + "Keep the proposal compact: hypothesis under 50 words, description under 180 words; planning output is charged to this episode's budget.")}
        origin = "brs_program_initiated" if arm == "P" else "brs_hypothesis_initiated"
        directive = ResearchDirective(code_parent_id=pair.primary_program_id, hypothesis_id=record.id,
                                      intervention_source_ids=pair.source_program_ids, research_episode=episode,
                                      rationale_codes=(origin, "brs_gap_adaptation", "sparse_gap_pair",
                                                       "coverage_" + pair.coverage_status))
        return directive, pair.key

    def _last_h_mode(self) -> str | None:
        settled = [e for e in self.episodes.values() if e.arm == "H" and e.status != "pending"]
        if not settled:
            return None
        return max(settled, key=lambda e: e.started_step).mode

    def _refresh_rebase_requirements(self, step: int, incumbent_id: str | None) -> None:
        """Open one "rebase" requirement per active hypothesis not yet realized on the incumbent.

        A hypothesis that produced a valid, observed realization on some base,
        while the incumbent is now a different program, is exactly the paper's
        gap: an experiment that has become feasible but has not been run.
        """
        if not self.rebase_requirements_enabled or not incumbent_id or incumbent_id == "root":
            return
        incumbent_family = self.node_info.get(incumbent_id, {}).get("family_sha256")
        for record in self.hypotheses.active():
            if record.effective_test_count == 0 or incumbent_id in record.program_ids:
                continue
            if any(e.hypothesis_id == record.id and e.primary_program_id == incumbent_id for e in self.episodes.values()):
                continue
            # Revision G rate limit. Revision F opened a new rebase requirement
            # every time the incumbent moved, which fed pressure back into the
            # lane that moves it (multimodal ran 17 recombinations and starved).
            # Now a hypothesis gets at most ONE rebase requirement per run, and
            # only when the incumbent belongs to a different family: re-realizing
            # a hypothesis on a base of its own family is what the continuation
            # tree already does.
            if any(r.kind == "rebase" for r in record.requirements):
                continue
            if incumbent_family is not None and incumbent_family == record.family_sha256:
                continue
            key = canonical_sha256({"hypothesis": record.id, "kind": "rebase", "target": incumbent_id})[:16]
            requirement = Requirement(
                key=key, kind="rebase", status="open", target_program_id=incumbent_id, created_step=step,
                description=f"rebase: hypothesis realized on {', '.join(record.program_ids)} but not on the current incumbent {incumbent_id}",
                next_step=f"realize this hypothesis on program {incumbent_id}, keeping that program's verified components")
            self.hypotheses.put(replace(record, requirements=(*record.requirements, requirement)))
            self._emit("rebase_requirement_opened", {"hypothesis_id": record.id, "target_program_id": incumbent_id, "key": key})

    def _untested_hypotheses(self) -> list[HypothesisRecord]:
        """Hypotheses whose attempts never produced a learnable outcome (e.g. censored)."""
        return [r for r in self.hypotheses.active()
                if r.effective_test_count == 0 and r.unknown_test_count == 0 and not r.open_requirements()
                and not r.program_ids and r.attempt_action_ids
                and all(self.episodes[self.episode_by_action[a]].status == "censored"
                        for a in r.attempt_action_ids if a in self.episode_by_action)]

    def _hypothesis_context(self, record: HypothesisRecord) -> dict[str, Any]:
        if self.memory_scope == "no_history":
            return {"statement": record.statement, "family": record.family, "version": record.version}
        if self.memory_scope == "no_failure_feedback":
            return {"statement": record.statement, "family": record.family, "version": record.version,
                    "attempts": len(record.attempt_action_ids), "best_metric": record.best_metric,
                    "effective_test_count": record.effective_test_count,
                    "unknown_test_count": record.unknown_test_count, "open_requirements": []}
        return {"statement": record.statement, "family": record.family, "version": record.version,
                "attempts": len(record.attempt_action_ids), "best_metric": record.best_metric,
                "effective_test_count": record.effective_test_count, "unknown_test_count": record.unknown_test_count,
                "open_requirements": [record_to_dict(r) for r in record.open_requirements()]}

    def _bind_reference_gap(self, snapshot: SearchSnapshot) -> None:
        if self.allocator.state.get("reference_gap") is None and self.reference_metric is not None:
            root = snapshot.node_metrics.get("root")
            gap = self.allocator.set_reference_gap(self.reference_metric, root)
            if gap is not None:
                self._emit("reference_gap_bound", {"reference": self.reference_metric, "root": root, "scale": gap})

    def proposal_directives(self, snapshot: SearchSnapshot, count: int) -> tuple[ProposalDirective, ...]:
        self._bind_reference_gap(snapshot)
        if count != 1:
            raise ValueError("bidirectional research requires incremental proposal_batch_size=1")
        if self.pending_episode_id is not None:
            raise ValueError("a research episode is still pending; resume it before planning a new one")
        self._sync_nodes(snapshot)
        for change in self.programs.refresh(self._valid_nodes(snapshot)):
            self._emit("program_representative_changed", change)
        for change in self.hypotheses.rebalance(snapshot.step):
            self._emit("hypothesis_status_changed", change)
        incumbent_metric, incumbent_id = self._incumbent(snapshot)
        self._refresh_rebase_requirements(snapshot.step, incumbent_id)
        unmet = self._refresh_pressure(snapshot.step)
        # Enumerate feasible pairings before the arm is chosen, so a pairing can
        # be consumed from whichever arm the allocator picks.
        pairs = self._gap_pairs(snapshot) if "H" in self.adaptation_arms else None
        try:
            tree_directive: ProposalDirective | None = self.tree.proposal_directives(snapshot, 1)[0]
        except RuntimeError:
            tree_directive = None
        if tree_directive is None:
            raise RuntimeError("bidirectional research has no valid executable parent")
        # A bounded local refinement the program optimizer has already opened is
        # finished before a new front is opened. The sub-strategy owns that
        # judgement: it reports tree_local only while the lane is authorized,
        # and the burst closes on its own stagnation and attempt limits. The
        # first Biology-5 revision left such a lane open and unused for a whole
        # cell because the pressure term kept selecting the other arm.
        open_local = any(code == "tree_local" for code in tree_directive.rationale_codes)
        # A root draft is a program-arm episode on root and nothing else: no
        # consolidation, sweep, pairing or recombination until the drafts are done.
        drafting = "initial_root_draft" in tree_directive.rationale_codes
        decision = self.allocator.choose({"P"} if open_local or drafting else set(self.frontier_arms), unmet)
        consolidation = None if drafting else self._eligible_consolidation(snapshot)
        if consolidation is not None:
            decision["arm"], decision["reason"] = "P", "evidence_lineage_consolidation"
        elif drafting:
            decision["reason"] = "initial_root_draft"
        elif open_local:
            decision["reason"] = "open_local_continuation"
        elif self.keep_and_extend_enabled and (phase := self._keep_and_extend_phase()) is not None:
            decision["arm"], decision["reason"] = phase
        elif self.momentum_enabled and decision["arm"] == "H" and self._program_momentum(incumbent_id):
            decision["arm"], decision["reason"] = "P", "momentum"
        if decision["arm"] not in self.frontier_arms:
            # Single-frontier ablation: an override above may not reopen a disabled frontier.
            decision["arm"], decision["reason"] = self.frontier_arms[0], "single_frontier"
        arm = decision["arm"]
        if self.total_score_evaluations is None:
            initial = snapshot.budget.remaining_score_evaluations
            self.total_score_evaluations = int(initial) if math.isfinite(initial) else 0
        sweep_ready = (consolidation is None and arm == "P" and not open_local and not drafting and self.incumbent_sweep_enabled
                       and self._sweep_admissible(snapshot, incumbent_id))
        base_context = {"allocator": decision, "unmet": unmet,
                        "step": snapshot.step,
                        "incumbent": {"node_id": incumbent_id, "native_metric": incumbent_metric},
                        "representatives": [record_to_dict(r) for r in self.programs.representatives]}
        if self.memory_scope == "no_history":
            # Only the incumbent survives: no retained programs, no accumulated
            # unmet needs. The planner proposes from the current best alone.
            base_context = {"allocator": decision, "unmet": (),
                            "incumbent": base_context["incumbent"], "representatives": []}
        pair_key = None
        h_adapts = (arm == "H" and "H" in self.adaptation_arms and bool(pairs) and self._last_h_mode() != "gap_adaptation")
        if consolidation is not None:
            directive, pair_key = self._consolidation_directive(*consolidation, base_context), None
            self.consolidation_consecutive += 1
        elif sweep_ready:
            directive, pair_key = self._sweep_directive(incumbent_id, incumbent_metric, base_context), None
        elif arm == "P":
            # The program arm consumes a pair only when it is an adaptation arm.
            # Otherwise its lane (continuation + instance recombination, the
            # source of three of R36's four best programs) is left untouched.
            if "P" in self.adaptation_arms and not drafting:
                pairs = self._gap_pairs(snapshot) if pairs is None else pairs
            else:
                pairs = []
            if pairs:
                directive, pair_key = self._pair_directive(pairs[0], base_context, "P")
            else:
                mode = next((c for c in tree_directive.rationale_codes if c.startswith("tree_")), "tree_global")
                parent_info = self.node_info.get(tree_directive.code_parent_id, {})
                episode = {**base_context, "arm": "P", "mode": mode,
                           "parent_hypothesis": {"id": parent_info.get("hypothesis_id"), "statement": parent_info.get("statement"),
                                                 "family": parent_info.get("family")},
                           "guidance": ("Program-initiated episode: continue this program lineage. tree_local improves the same "
                                        "realization or refines its hypothesis with one concrete change; tree_global/tree_final may "
                                        "test a scientifically distinct hypothesis on the selected parent. For a local continuation, "
                                        "read the parent's per_instance_improvement: keep the components of the instances that "
                                        "improved and change one thing for the weakest instance, so the strong part is not rewritten. "
                                        + "Keep the proposal compact: hypothesis under 50 words, description under 180 words; planning output is charged to this episode's budget.")}
                if drafting:
                    episode["guidance"] = ("Independent draft from the starter program: write a complete solution "
                                           "whose approach is clearly distinct from any earlier draft in the research "
                                           "graph; do not build on those drafts. "
                                           + "Keep the proposal compact: hypothesis under 50 words, description under 180 words; planning output is charged to this episode's budget.")
                directive = ResearchDirective(**asdict(tree_directive), research_episode=episode)
                codes = ("brs_program_initiated", *tree_directive.rationale_codes)
                # Instance-level recombination: the Claude reference and the R29b
                # win both kept the component that already won an instance and
                # changed only the weak one. Offer the per-instance winners as
                # typed sources instead of asking for a rewrite of both.
                parent_summary = (snapshot.node_evaluation_summaries or {}).get(tree_directive.code_parent_id) or {}
                sources = self.programs.recombination_sources(
                    tree_directive.code_parent_id, parent_summary.get("per_instance_improvement")
                    if isinstance(parent_summary, dict) else None) if self.instance_recombination_enabled else []
                key = tree_directive.code_parent_id + "|" + ",".join(sorted(r["node_id"] for r in sources))
                if (self.instance_recombination_enabled and sources and not drafting
                        and not tree_directive.intervention_source_ids and key not in self.recombinations):
                    self.recombinations.add(key)
                    # Not a tree-owned mode: the tree refuses multi-parent actions,
                    # so this episode is bound by the research strategy itself.
                    episode["mode"] = "instance_recombination"
                    episode["instance_recombination"] = sources
                    episode["guidance"] += (" Instance recombination: each listed source program beats this parent on the "
                                            "named instance(s). Adopt that source's component for exactly those instances "
                                            "and keep the parent's components elsewhere; assign each instance to one "
                                            "pipeline explicitly, no validation-based selection.")
                    directive = replace(directive, intervention_source_ids=tuple(r["node_id"] for r in sources))
                    codes = (*codes, "brs_instance_recombination")
                directive = replace(directive, rationale_codes=codes)
        elif h_adapts:
            # Hypothesis-initiated adaptation: the hypothesis side claims a
            # matched program base. Alternates with new-hypothesis episodes so
            # the hypothesis arm never becomes an adaptation-only lane.
            directive, pair_key = self._pair_directive(pairs[0], base_context, "H")
        else:
            untested = self._untested_hypotheses()
            parent = incumbent_id or "root"
            if untested:
                record = sorted(untested, key=lambda r: (-self.hypotheses.waiting(r, snapshot.step), r.id))[0]
                episode = {**base_context, "arm": "H", "mode": "activate_hypothesis", "hypothesis_id": record.id,
                           "hypothesis": self._hypothesis_context(record),
                           "guidance": ("Hypothesis-initiated episode: this hypothesis never received a learnable outcome. "
                                        "It is fixed (hypothesis_id, statement, family); implement it on the selected code parent. "
                                        + "Keep the proposal compact: hypothesis under 50 words, description under 180 words; planning output is charged to this episode's budget.")}
                directive = ResearchDirective(code_parent_id=parent, hypothesis_id=record.id, research_episode=episode,
                                              rationale_codes=("brs_hypothesis_initiated", "brs_activate_hypothesis"))
            else:
                opportunities = [record_to_dict(i) for i in sorted(self.pressure.values(), key=lambda i: (i.created_step, i.key))
                                 if i.side == "P" and i.status == "queued"][:3]
                covered = [{"hypothesis_id": r.id, "family": r.family, "status": r.status, "best_metric": r.best_metric,
                            "effective_test_count": r.effective_test_count, "unknown_test_count": r.unknown_test_count}
                           for r in sorted(self.hypotheses.records.values(), key=lambda r: r.id) if r.status != "archived"]
                root_metric = snapshot.node_metrics.get("root")
                below_root = [e for e in self.episodes.values()
                              if e.mode == "new_hypothesis" and e.status == "settled" and _finite(e.native_metric)
                              and _finite(root_metric) and e.native_metric < root_metric]
                refine_root_family = (parent == "root" and len(below_root) >= self.root_family_refinement_after)
                episode = {**base_context, "arm": "H", "mode": "new_hypothesis", "covered_families": covered,
                           "opportunities": opportunities,
                           "root_family_refinement": {"enabled": refine_root_family, "new_families_below_root": len(below_root),
                                                      "threshold": self.root_family_refinement_after},
                           "guidance": (
                               ("Hypothesis-initiated episode inside the root program's own family: every new mechanism "
                                "family tried so far scored below the root, so the root's method is the strongest known "
                                "one here. Propose a hypothesis that changes exactly ONE parameter or component of the "
                                "root program (for example its neighbourhood or smoothing scale, cluster count, "
                                "normalisation, or model capacity) and keeps everything else; the family core may equal "
                                "the root's. State which single change is tested and why it should move the metric. "
                                if refine_root_family else
                                "Hypothesis-initiated episode on a verified incumbent: propose ONE new mechanism whose "
                                "scientific family core differs from every covered_families entry, realized by replacing "
                                "or adding a single component of the selected code parent while keeping its other verified "
                                "components and its input/output contract unchanged. Do not rewrite the whole pipeline: "
                                "one action budget realizes one mechanism change. "
                                if parent != "root" else
                                "Hypothesis-initiated episode from the root program: propose a hypothesis whose scientific "
                                "family core differs from every covered_families entry, implemented on the selected code "
                                "parent. Prefer a mechanism realizable within one action budget: a single model family, "
                                "no multi-stage ensembles or validation-based routing. ")
                               + "Use listed opportunities (observed differential program behaviour) as motivation when "
                                 "relevant; cite them in the description. A new mechanism family, not a capacity change. "
                               + "Keep the proposal compact: hypothesis under 50 words, description under 180 words; planning output is charged to this episode's budget.")}
                directive = ResearchDirective(code_parent_id=parent, research_episode=episode,
                                              rationale_codes=("brs_hypothesis_initiated", "brs_new_hypothesis",
                                                               *(("brs_root_family_refinement",) if refine_root_family else ()),
                                                               "family_balanced_novelty"))
        self.next_directive = {"graph_sha256": snapshot.graph_sha256, "directive": asdict(directive),
                               "used_tokens_before": snapshot.budget.used_tokens, "step": snapshot.step,
                               "pair_key": pair_key}
        self._emit("allocator_decision", {"decision": decision, "arm": arm, "mode": directive.research_episode["mode"],
                                          "code_parent_id": directive.code_parent_id, "hypothesis_id": directive.hypothesis_id,
                                          "intervention_source_ids": directive.intervention_source_ids, "step": snapshot.step})
        return (directive,)

    # ---------------------------------------------------------------- select
    def _register_hypothesis(self, action: ActionCandidate, episode: dict[str, Any], step: int) -> str:
        family = record_to_dict(asdict(action.family))
        family_sha = action.family.sha256
        existing = self.hypotheses.get(action.hypothesis_id)
        if existing is not None:
            return existing.id
        duplicate = self.hypotheses.exact_duplicate(family_sha, action.hypothesis_statement)
        if duplicate is not None:
            self.hypothesis_alias[action.hypothesis_id] = duplicate
            self._emit("hypothesis_duplicate", {"hypothesis_id": action.hypothesis_id, "duplicate_of": duplicate,
                                                "duplicate_status": "exact"})
            return duplicate
        motivation = tuple(o["key"] for o in episode.get("opportunities", ()) if isinstance(o, dict) and "key" in o)
        origin = "program_observation" if episode.get("mode") == "new_hypothesis" and motivation else "top_down"
        possible = [r.id for r in self.hypotheses.records.values() if r.family_sha256 == family_sha]
        record = HypothesisRecord(id=action.hypothesis_id, statement=action.hypothesis_statement[:4000], family=family,
                                  family_sha256=family_sha, origin=origin, created_step=step, created_action_id=action.id,
                                  created_arm=episode["arm"], motivation_refs=motivation,
                                  duplicate_status="possible" if possible else "distinct")
        self.hypotheses.put(record)
        self._emit("hypothesis_created", {"hypothesis": record, "possible_duplicates": possible})
        return record.id

    def select(self, snapshot: SearchSnapshot) -> SearchDecision:
        if self.pending_episode_id is not None:
            raise ValueError("a research episode is still pending")
        self._sync_nodes(snapshot)
        candidates = _available(snapshot)
        repairs = _pending_repairs(candidates)
        incumbent_metric, incumbent_id = self._incumbent(snapshot)
        config_sha = canonical_sha256(self.config)
        if repairs:
            if len(repairs) != 1:
                raise ValueError("bidirectional research schedules one bounded repair at a time")
            action = repairs[0]
            origin_ids = [d.source_id for d in action.dependencies if d.kind == "diagnostic_source"]
            origin = self.episodes.get(self.episode_by_action.get(origin_ids[0], "")) if len(origin_ids) == 1 else None
            if origin is None:
                raise ValueError("repair origin is not a bound research episode")
            parent = _parent_id(action)
            resolved = self.hypothesis_alias.get(action.hypothesis_id, action.hypothesis_id)
            if parent != origin.primary_program_id or resolved != origin.hypothesis_id:
                raise ValueError("repair changed the origin episode parent or hypothesis")
            origin_lineage = self.consolidation_by_action.get(origin.action_id or "")
            if origin_lineage is not None:
                self.consolidation_by_action[action.id] = origin_lineage
            if self._tree_owned(origin.mode):
                decision = self.tree.select(snapshot)
                if decision.action_id != action.id:
                    raise ValueError("tree selected a different repair action")
            else:
                decision = SearchDecision(strategy=self.name, action_id=action.id, graph_sha256=snapshot.graph_sha256,
                                          score_vector=(("repair_origin_reward", float(origin.reward or 0.0)),),
                                          rationale_codes=(f"brs_{origin.arm.lower()}_initiated", "bound_origin_repair"))
            episode = EpisodeRecord(id=f"ep{len(self.episodes):04d}", arm=origin.arm,
                                    mode="tree_repair" if self._tree_owned(origin.mode) else "repair",
                                    primary_program_id=parent, started_step=snapshot.step, action_id=action.id,
                                    action_sha256=action.action_sha256, hypothesis_id=resolved,
                                    source_program_ids=tuple(d.source_id for d in action.dependencies if d.kind == "intervention_source"),
                                    incumbent_pre=incumbent_metric, incumbent_pre_node_id=incumbent_id,
                                    origin_episode_id=origin.id, pair_key=origin.pair_key, config_sha256=config_sha)
            rationale = decision.rationale_codes
        else:
            fresh = [a for a in candidates if a.operator != ActionOperator.REPAIR_PROGRAM]
            if len(fresh) != 1:
                raise ValueError("bidirectional research schedules one fresh action at a time")
            action = fresh[0]
            pending = self.next_directive
            if pending is None:
                raise ValueError("no controller directive is bound for this action")
            directive = pending["directive"]
            parent = _parent_id(action)
            sources = tuple(d.source_id for d in action.dependencies if d.kind == "intervention_source")
            if (parent != directive["code_parent_id"] or sources != tuple(directive["intervention_source_ids"])
                    or (directive.get("hypothesis_id") is not None and action.hypothesis_id != directive["hypothesis_id"])):
                raise ValueError("action does not match the controller-selected research episode")
            episode_context = directive["research_episode"]
            arm, mode = episode_context["arm"], episode_context["mode"]
            if mode == "evidence_consolidation":
                lineage_id = str(episode_context.get("lineage_id") or "")
                lane = self.consolidation_lineages.get(lineage_id)
                if lane is None:
                    raise ValueError("consolidation action is not bound to an active lineage")
                # Do not reinterpret an off-family proposal as tree_local: it would
                # inherit the old hypothesis evidence and may bind a stale Tree parent.
                # Refuse without binding the action; the lane is retired so the next
                # directive routes elsewhere instead of asking for the same lineage again.
                if action.operator != ActionOperator.IMPROVE_PROGRAM or sources:
                    self._retire_rejected_consolidation(lineage_id, lane, action, snapshot.step)
                    raise ProposalRejected("consolidation must be a single-parent improvement",
                                           code="consolidation_shape_mismatch", action_id=action.id)
                if (action.hypothesis_id != lane["hypothesis_id"]
                        or action.family.sha256 != lane["family_sha256"]):
                    self._retire_rejected_consolidation(lineage_id, lane, action, snapshot.step)
                    raise ProposalRejected("consolidation changed the bound hypothesis family",
                                           code="consolidation_family_mismatch", action_id=action.id)
                self.consolidation_by_action[action.id] = lineage_id
                lane["status"] = "active"
            if self._tree_owned(mode):
                decision = self.tree.select(snapshot)
                if decision.action_id != action.id:
                    raise ValueError("tree selected a different action than the research episode")
            else:
                index = episode_context["allocator"]["index"].get(arm) or {}
                decision = SearchDecision(strategy=self.name, action_id=action.id, graph_sha256=snapshot.graph_sha256,
                                          score_vector=tuple((k, float(v)) for k, v in sorted(index.items()) if _finite(v)),
                                          rationale_codes=(f"brs_{arm.lower()}_initiated", "brs_" + mode))
            hypothesis_id = self._register_hypothesis(action, episode_context, snapshot.step)
            planning = max(0, int(snapshot.budget.used_tokens) - int(pending["used_tokens_before"]))
            episode = EpisodeRecord(id=f"ep{len(self.episodes):04d}", arm=arm, mode=mode, primary_program_id=parent,
                                    started_step=snapshot.step, action_id=action.id, action_sha256=action.action_sha256,
                                    hypothesis_id=hypothesis_id, source_program_ids=sources, incumbent_pre=incumbent_metric,
                                    incumbent_pre_node_id=incumbent_id, planning_tokens=planning,
                                    pair_key=pending.get("pair_key"), allocator=episode_context["allocator"],
                                    config_sha256=config_sha)
            if episode.pair_key:
                self.pairs[episode.pair_key] = replace(self.pairs[episode.pair_key], status="attempted")
            for side in ARMS:
                for key in self._pressure_keys(side, hypothesis_id, statuses=("queued",)):
                    self._set_pressure(key, "pending", episode.id)
            for opportunity in episode_context.get("opportunities", ()):
                if isinstance(opportunity, dict) and opportunity.get("key") in self.pressure:
                    self._set_pressure(opportunity["key"], "pending", episode.id)
            self.next_directive = None
            rationale = (f"brs_{arm.lower()}_initiated", *decision.rationale_codes)
        record = self.hypotheses.get(episode.hypothesis_id or "")
        if record is not None:
            self.hypotheses.put(replace(record, attempt_action_ids=(*record.attempt_action_ids, action.id),
                                        last_attempt_step=snapshot.step,
                                        requirements=tuple(replace(r, status="pending") if r.status == "open" else r
                                                           for r in record.requirements)))
        self.episodes[episode.id] = episode
        self.episode_by_action[action.id] = episode.id
        self.pending_episode_id = episode.id
        self._emit("episode_started", {"episode": episode})
        return replace(decision, strategy=self.name, rationale_codes=tuple(dict.fromkeys(rationale)),
                       policy_sha256=config_sha)

    # --------------------------------------------------------------- observe
    @staticmethod
    def _requirement_from_failure(observation: SearchObservation, action_id: str, parent: str, step: int) -> Requirement:
        summary = dict(observation.failure_summary or {})
        reason = str(summary.get("reason") or observation.recovery_route or "implementation_failed")
        failed = [v for v in (summary.get("validation") or []) if isinstance(v, dict) and v.get("accepted") is False]
        tiers = sorted({str(v.get("tier")) for v in failed})
        timed_out = any(v.get("timed_out") for v in failed) or "timed_out" in reason or "wall" in reason
        kind = "efficiency" if timed_out or observation.recovery_route == "optimize_efficiency" else "implementation"
        stderr = " ".join(str(v.get("stderr") or "")[-300:] for v in failed[-2:])
        description = f"{kind}: {reason}; failed checks {tiers or ['none']}; " + " ".join(str(v) for v in (summary.get("contract_violations") or [])[:3]) + " " + stderr
        if kind == "efficiency":
            # Cross-modal R35/R36: the same dense formulation (Sinkhorn on ~1e5 x 1e5 cells)
            # was retried on another base and timed out again. The unmet need is a
            # bounded-cost realization, not a different base.
            next_step = ("realize the same mechanism with a bounded-cost method that finishes within the execution "
                         "timeout on the largest public instance: operate at cluster or landmark level, use sparse "
                         "kNN or minibatch formulations, or subsample; do not retry the dense formulation on any base")
        else:
            next_step = (f"reimplement on a program base other than {parent} whose executed components cover the failing "
                         f"{', '.join(tiers) or 'validation'} path, or repair on {parent} with the retained failed source")
        key = canonical_sha256({"action": action_id, "kind": kind})[:16]
        return Requirement(key=key, kind=kind, description=description[:1500], next_step=next_step, status="open",
                           source_action_id=action_id, failed_program_id=parent, created_step=step)

    @staticmethod
    def _differential(child: dict[str, Any] | None, parent: dict[str, Any] | None) -> tuple[list[str], list[str]]:
        if not isinstance(child, dict) or not isinstance(parent, dict):
            return [], []
        improved, degraded = [], []
        for key, value in child.items():
            other = parent.get(key)
            if _finite(value) and _finite(other):
                if value > other:
                    improved.append(str(key))
                elif value < other:
                    degraded.append(str(key))
        return sorted(improved), sorted(degraded)

    def _charge_local_burst(self, observation: SearchObservation, episode: EpisodeRecord) -> None:
        """Count a recombination taken from the open local burst as one local attempt.

        Mirrors ContinuationTreeStrategy.observe for mode "local": attempts always,
        scored/stagnation for a scored child, and a better child becomes the burst's
        best so the next local step continues from it rather than from the stale parent.
        """
        burst = getattr(self.tree, "burst", None)
        if (not isinstance(burst, dict) or burst.get("closed")
                or burst.get("best_id") != episode.primary_program_id):
            return
        burst["attempts"] += 1
        scored = (observation.executable and observation.score_bearing and observation.artifact_valid
                  and observation.program_node_id is not None and _finite(observation.native_metric))
        if scored:
            burst["scored"] += 1
            if observation.native_metric > burst["best_metric"]:
                burst["best_id"] = observation.program_node_id
                burst["best_metric"] = observation.native_metric
                burst["stagnation"] = 0
            else:
                burst["stagnation"] += 1
        self._emit("local_burst_charged", {"action_id": observation.action_id, "burst": dict(burst)})

    def observe(self, observation: SearchObservation) -> None:
        episode_id = self.pending_episode_id
        episode = self.episodes.get(episode_id or "")
        if episode is None or episode.action_id != observation.action_id or episode.action_sha256 != observation.action_sha256:
            raise ValueError("research observation is not bound to the pending episode")
        if any(item.action_id == observation.action_id for item in self.observations):
            raise ValueError("research observation already consumed")
        if self._tree_owned(episode.mode):
            self.tree.observe(observation)
        elif self.local_burst_counts_recombination and episode.mode == "instance_recombination":
            self._charge_local_burst(observation, episode)
        self.observations.append(observation)
        step = len(self.observations)
        scored = (observation.executable and observation.score_bearing and observation.artifact_valid
                  and observation.program_node_id is not None and _finite(observation.native_metric))
        category = ((observation.evaluation_summary or {}).get("public_diagnostic") or {}).get("category") \
            if isinstance(observation.evaluation_summary, dict) else None
        censored = (observation.recovery_route == "retry_evaluator_same_candidate"
                    or category in EVALUATION_INFRASTRUCTURE_CATEGORIES)
        # The candidate ran and measured itself below its parent on public
        # labels; no official evaluation was spent. That is a learnable outcome
        # of the realized hypothesis, not a failure and not a refutation.
        self_rejected = observation.recovery_route == "discard_underperforming_candidate"
        realization = "unknown"
        implementation = dict(observation.implementation_observation or {})
        if scored:
            not_realized = (implementation.get("source_change") in {"added_definitions_without_static_name_load", "ast_unchanged", "identical_source"}
                            or implementation.get("output_relation") == "identical_observed_outputs")
            if self.evidence_audit == "off" and not_realized:
                self._emit("evidence_audit_skipped", {"action_id": observation.action_id, "would_be": "not_realized"})
                not_realized = False
            realization = "not_realized" if not_realized else "observed"
            if episode.mode == "incumbent_sweep" and not_realized:
                state = self.sweeps.setdefault(episode.primary_program_id, {"count": 0, "insensitive": False})
                state["insensitive"] = True
                self._emit("incumbent_sweep_insensitive", {"incumbent_id": episode.primary_program_id})
        record = self.hypotheses.get(episode.hypothesis_id or "")
        requirements_resolved: list[str] = []
        if observation.program_node_id is not None and observation.artifact_valid:
            parent_info = self.node_info.get(episode.primary_program_id, {})
            family = (record.family if record is not None else None)
            capabilities = (capability_tags(family, record.statement if record else "") if realization == "observed"
                            else tuple(parent_info.get("capabilities", ())))
            summary = observation.evaluation_summary if isinstance(observation.evaluation_summary, dict) else {}
            self.node_info[observation.program_node_id] = {
                "hypothesis_id": episode.hypothesis_id, "statement": record.statement if record else "",
                "family": family, "family_sha256": observation.realized_family_sha256 if realization == "observed" else parent_info.get("family_sha256"),
                "capabilities": list(capabilities), "realization": realization, "parent_id": episode.primary_program_id,
                "episode_id": episode.id, "arm": episode.arm, "primary_scores": self._primary_scores(summary),
                "native_metric": observation.native_metric,
                "evaluation_evidence_sha256": observation.evaluation_evidence_sha256,
                "implementation_observation": implementation,
                "executable_sha256": None,
            }
            self._emit("capability_observed", {"node_id": observation.program_node_id, "realization": realization,
                                               "capabilities": list(capabilities), "implementation_observation": implementation})
            improved, degraded = self._differential(self._primary_scores(summary), parent_info.get("primary_scores"))
            if improved and degraded and scored:
                key = canonical_sha256({"side": "P", "node": observation.program_node_id})[:24]
                if key not in self.pressure:
                    item = PressureItem(key=key, side="P", ref=observation.program_node_id,
                                        description=(f"program {observation.program_node_id} improved instances {improved} but degraded "
                                                     f"{degraded} relative to parent {episode.primary_program_id}"),
                                        next_step="propose a hypothesis that explains the instance-dependent behaviour and predicts when it holds",
                                        created_step=step, evidence_refs=tuple(x for x in (observation.evaluation_evidence_sha256,) if x))
                    self.pressure[key] = item
                    self._emit("pressure_queued", {"item": item})
        if record is not None:
            requirements = list(record.requirements)
            effective, unknown = record.effective_test_count, record.unknown_test_count
            best, programs = record.best_metric, record.program_ids
            if scored and realization == "observed":
                effective += 1
                requirements = [replace(r, status="resolved", resolved_by_action_id=observation.action_id)
                                if r.status in {"open", "pending"} and (r.kind != "rebase" or r.target_program_id == episode.primary_program_id)
                                else (replace(r, status="open") if r.status == "pending" else r) for r in requirements]
                requirements_resolved = [r.key for r in requirements if r.resolved_by_action_id == observation.action_id]
            elif scored:
                unknown += 1
                key = canonical_sha256({"action": observation.action_id, "kind": "realization"})[:16]
                requirements = [replace(r, status="open") if r.status == "pending" else r for r in requirements]
                if self.memory_scope in ("no_history", "no_failure_feedback"):
                    self._emit("failure_feedback_suppressed",
                               {"action_id": observation.action_id, "memory_scope": self.memory_scope})
                elif all(r.key != key for r in requirements):
                    source_change = implementation.get("source_change")
                    ran_but_unchanged = (self.realization_diagnosis
                                         and implementation.get("output_relation") == "identical_observed_outputs"
                                         and source_change not in {"added_definitions_without_static_name_load",
                                                                   "ast_unchanged", "identical_source"})
                    if ran_but_unchanged:
                        # The old wording ("not observed in the execution path") sent 00732's
                        # planner to new ideas although the new component had run and been
                        # rejected by the program's own fallback (hold-out 1.05 vs 0.114).
                        description = (f"realization: the changed code of {observation.action_id} ran, but its submitted "
                                       f"outputs are identical to the parent's ({source_change}, identical_observed_outputs). "
                                       "Usually the program's own fallback kept the parent's outputs because the new "
                                       "component scored worse on its internal hold-out, or the component's result never "
                                       f"reached the written output; {observation.action_id}'s execution log shows which")
                        next_step = ("fix this component on the same idea, using the numbers in that log, and make its "
                                     "outputs the submitted ones for the instances it targets, without a fallback to the "
                                     "parent; do not switch to a different idea")
                    else:
                        description = f"realization: declared intervention not observed in the execution path ({source_change}, {implementation.get('output_relation')})"
                        next_step = "wire the declared component into the executed path on this or another program base and re-run"
                    requirements.append(Requirement(
                        key=key, kind="realization", status="open", description=description, next_step=next_step,
                        source_action_id=observation.action_id, failed_program_id=episode.primary_program_id, created_step=step))
            elif censored:
                requirements = [replace(r, status="open") if r.status == "pending" else r for r in requirements]
            elif self_rejected:
                effective += 1
                requirements = [replace(r, status="open") if r.status == "pending" else r for r in requirements]
            elif not observation.executable or not observation.artifact_valid:
                requirements = [replace(r, status="open") if r.status == "pending" else r for r in requirements]
                if self.memory_scope in ("no_history", "no_failure_feedback"):
                    # The failure is still recorded as an outcome and still ends
                    # the episode; it is simply not distilled into a requirement
                    # that steers later proposals.
                    self._emit("failure_feedback_suppressed",
                               {"action_id": observation.action_id, "memory_scope": self.memory_scope})
                else:
                    new = self._requirement_from_failure(observation, observation.action_id,
                                                         episode.primary_program_id, step)
                    same = next((r for r in requirements if r.status == "open" and r.kind == new.kind
                                 and r.failed_program_id == new.failed_program_id), None)
                    if same is not None:
                        # A repeated failure on the same base refreshes the recorded
                        # evidence; it is one unmet need, not additional pressure.
                        requirements = [replace(r, description=new.description, source_action_id=new.source_action_id)
                                        if r.key == same.key else r for r in requirements]
                    elif all(r.key != new.key for r in requirements):
                        requirements.append(new)
            else:
                requirements = [replace(r, status="open") if r.status == "pending" else r for r in requirements]
            if scored:
                metric = float(observation.native_metric)
                best = metric if best is None or metric > best else best
                programs = (*programs, observation.program_node_id)
            self.hypotheses.put(replace(record, requirements=tuple(requirements), effective_test_count=effective,
                                        unknown_test_count=unknown, best_metric=best, program_ids=programs,
                                        last_attempt_step=step))
            for side in ARMS:
                for key in self._pressure_keys(side, record.id):
                    self._set_pressure(key, "resolved" if requirements_resolved else "queued", episode.id)
            for key in [k for k, v in self.pressure.items() if v.side == "P" and v.consumed_by_episode_id == episode.id and v.status == "pending"]:
                self._set_pressure(key, "resolved", episode.id)
        if scored:
            terminal = "scored"
        elif censored:
            terminal = "censored_infrastructure"
        elif self_rejected:
            terminal = "self_rejected"
        elif observation.executable and observation.artifact_valid:
            terminal = "executed_unscored"
        elif not observation.executable:
            terminal = "implementation_failed"
        else:
            terminal = "invalid_artifact"
        coder_tokens = int(observation.tokens)
        if censored:
            self.allocator.censor(episode.arm)
            settled = replace(episode, status="censored", terminal_kind=terminal, coder_tokens=coder_tokens,
                              native_metric=observation.native_metric if scored else None, settled_step=step)
        else:
            reward = self.allocator.reward(observation.native_metric if scored else None, episode.incumbent_pre)
            cost = self.allocator.cost(episode.planning_tokens + coder_tokens)
            self.allocator.settle(episode.arm, reward, cost)
            settled = replace(episode, status="settled", terminal_kind=terminal, coder_tokens=coder_tokens, reward=reward,
                              cost_units=cost, native_metric=observation.native_metric if scored else None, settled_step=step)
        self.episodes[episode.id] = settled
        if observation.action_id in self.consolidation_by_action:
            self._settle_consolidation(observation, settled, realization=realization, step=step)
        else:
            self._arm_consolidation(observation, settled, realization=realization, step=step)
        self.pending_episode_id = None
        self._emit("episode_settled", {"episode": settled, "realization": realization, "requirements_resolved": requirements_resolved,
                                       "allocator_state": self.allocator.checkpoint_state()["state"]})

    # ------------------------------------------------------------ persistence
    def checkpoint_state(self) -> dict[str, Any]:
        return {
            "config": record_to_dict(self.config),
            "tree": self.tree.checkpoint_state(),
            "allocator": self.allocator.checkpoint_state(),
            "hypotheses": self.hypotheses.to_dict(),
            "programs": self.programs.to_dict(),
            "recombinations": sorted(self.recombinations),
            "sweeps": {k: dict(v) for k, v in self.sweeps.items()},
            "consolidation_lineages": record_to_dict(self.consolidation_lineages),
            "consolidation_by_action": dict(self.consolidation_by_action),
            "consolidation_consecutive": self.consolidation_consecutive,
            "consolidation_cooldown_remaining": self.consolidation_cooldown_remaining,
            "total_score_evaluations": self.total_score_evaluations,
            "node_info": record_to_dict(self.node_info),
            "episodes": {k: record_to_dict(v) for k, v in self.episodes.items()},
            "episode_by_action": dict(self.episode_by_action),
            "pairs": {k: record_to_dict(v) for k, v in self.pairs.items()},
            "pressure": {k: record_to_dict(v) for k, v in self.pressure.items()},
            "pending_episode_id": self.pending_episode_id,
            "next_directive": record_to_dict(self.next_directive),
            "hypothesis_alias": dict(self.hypothesis_alias),
            "observations": [asdict(item) for item in self.observations],
            "undrained_events": list(self._events),
        }

    def restore_state(self, payload: dict[str, Any]) -> None:
        if payload.get("config") != record_to_dict(self.config):
            raise ValueError("bidirectional research checkpoint configuration changed")
        self.tree.restore_state(payload["tree"])
        self.allocator.restore_state(payload["allocator"])
        self.hypotheses.restore(payload["hypotheses"])
        self.programs.restore(payload["programs"])
        self.recombinations = set(payload.get("recombinations", ()))
        self.sweeps = {k: {"count": int(v.get("count", 0)), "insensitive": bool(v.get("insensitive", False))}
                       for k, v in (payload.get("sweeps") or {}).items()}
        self.consolidation_lineages = {k: dict(v) for k, v in (payload.get("consolidation_lineages") or {}).items()}
        self.consolidation_by_action = dict(payload.get("consolidation_by_action") or {})
        self.consolidation_consecutive = int(payload.get("consolidation_consecutive", 0))
        self.consolidation_cooldown_remaining = int(payload.get("consolidation_cooldown_remaining", 0))
        total = payload.get("total_score_evaluations")
        self.total_score_evaluations = int(total) if total is not None else None
        self.node_info = {k: dict(v) for k, v in payload["node_info"].items()}
        self.episodes = {k: episode_from_dict(v) for k, v in payload["episodes"].items()}
        self.episode_by_action = dict(payload["episode_by_action"])
        self.pairs = {k: pair_from_dict(v) for k, v in payload["pairs"].items()}
        self.pressure = {k: pressure_from_dict(v) for k, v in payload["pressure"].items()}
        self.pending_episode_id = payload.get("pending_episode_id")
        self.next_directive = payload.get("next_directive")
        self.hypothesis_alias = dict(payload.get("hypothesis_alias", {}))
        self.observations = [SearchObservation(**{**row, "intervention_source_ids": tuple(row.get("intervention_source_ids", ()))})
                             for row in payload.get("observations", [])]
        self._events = list(payload.get("undrained_events", []))

    # ---------------------------------------------------------------- report
    def frontier_report(self) -> dict[str, Any]:
        episodes = list(self.episodes.values())
        return {
            "hypotheses": {k: record_to_dict(v) for k, v in sorted(self.hypotheses.records.items())},
            "representatives": [record_to_dict(r) for r in self.programs.representatives],
            "episodes": [record_to_dict(e) for e in episodes],
            "episode_counts": {arm: sum(e.arm == arm for e in episodes) for arm in ARMS},
            "settled_counts": {arm: sum(e.arm == arm and e.status == "settled" for e in episodes) for arm in ARMS},
            "censored_counts": {arm: sum(e.arm == arm and e.status == "censored" for e in episodes) for arm in ARMS},
            "pairs": {k: record_to_dict(v) for k, v in sorted(self.pairs.items())},
            "pressure": {k: record_to_dict(v) for k, v in sorted(self.pressure.items())},
            "consolidation": {
                "lineages": record_to_dict(self.consolidation_lineages),
                "by_action": dict(self.consolidation_by_action),
                "consecutive": self.consolidation_consecutive,
                "cooldown_remaining": self.consolidation_cooldown_remaining,
            },
            "allocator": self.allocator.checkpoint_state()["state"],
        }
