from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
import hashlib
import json
from pathlib import Path
from typing import Any, Protocol

from .coder_models import (
    CoderOutcome, MutationInspiration, ProgramMutationContext,
    ScientificActionContract, repair_selected_scientific_plan, selected_scientific_plan,
)
from .contracts import (
    ActionCandidate, ActionOperator, ProposalRejected, ResearchDependency, ScientificFamily,
    SearchBudget,
    SearchObservation,
    SearchSnapshot,
    canonical_sha256,
)
from .state import SearchEvent
from .state import DynamicResearchView
from .strategies import SearchStrategy
from .strategies import ProposalDirective
from .routing import classify_recovery_route
from .diagnostics import diagnostic_artifact_fields, is_repair_seed
from .session_limits import exceeded, positive_cap, remaining


def controller_repair_directive(route: str) -> str:
    """Return route-specific, hypothesis-preserving implementation guidance."""
    if route == "optimize_efficiency":
        # Revision G. The seed-1 SMLM cell timed out at the smoke tier, and its
        # repair re-ran the same formulation and timed out again, spending the
        # rest of the cell. The Claude reference on the cross-modal task never
        # touched a quadratic method at all: it rejected them by cost before
        # writing code. A repair after a resource failure must therefore change
        # the algorithmic class of the dominant operation, not tune it.
        return (
            "\n\nEFFICIENCY REPAIR REQUIREMENTS. The previous realization of this hypothesis "
            "exhausted its wall-clock or memory budget. Treat that as an algorithmic-complexity defect: "
            "the dominant operation does not fit the public data scale. Do not merely reduce restarts, "
            "iterations, or batch sizes by a constant factor; one invocation of the dominant primitive "
            "can itself exhaust the budget. "
            "MANDATORY: identify the dominant operation of the failed run and replace it with one "
            "of a strictly lower complexity class in the number of samples n: a full pairwise "
            "matrix (n x n distances, kernels, transport plans, attention over all pairs) becomes "
            "a bounded-neighbourhood or batched top-k computation; an exact assignment or dense "
            "solver becomes a greedy, sketched, or minibatch one; an exhaustive search becomes a "
            "sampled one with a fixed sample size. State the old and the new complexity class in "
            "the trajectory. Do NOT resubmit the same formulation with smaller constants: the "
            "controller records that as the same failure. Bind every data-dependent loop, restart, "
            "sample size, or solver iteration to the observed input scale. Aim for a measured "
            "public-data runtime below 50% of the recorded validation timeout, run a representative "
            "public-data benchmark before finishing, and report its timing in the trajectory. "
            "Preserve the exact scientific hypothesis, metric, data split, and output contract: the "
            "mechanism stays, only its realization changes class."
        )
    if route == "contract_preserving_rewrite":
        return (
            "\n\nCONTRACT-PRESERVING REWRITE REQUIREMENTS. Repair only the implementation "
            "while preserving the exact method, metric, data split, output schema, and hypothesis. "
            "Re-run every deterministic contract gate before finishing."
        )
    return (
        "\n\nIMPLEMENTATION REPAIR REQUIREMENTS. Resolve the recorded dependency, import, "
        "syntax, parameter, or runtime defect without changing the scientific method, metric, "
        "data split, output schema, or hypothesis. Re-run the failing gate and all deterministic "
        "contract gates before finishing."
    )


def artifact_workspace_sha256(root: Path, *, exclude_names: tuple[str, ...] = ()) -> str:
    """Content digest of a program workspace; ``exclude_names`` drops top-level files by name."""
    manifest: dict[str, str] = {}
    ignored_parts = {
        ".git", ".evolvcoder", "output", "__pycache__", ".pytest_cache",
        ".mypy_cache", ".ruff_cache",
    }
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if any(part in ignored_parts for part in path.relative_to(root).parts):
            continue
        if path.suffix == ".pyc":
            continue
        if path.parent == root and path.name in exclude_names:
            continue
        if path.is_symlink():
            manifest[relative] = "symlink-forbidden"
        elif path.is_file():
            manifest[relative] = hashlib.sha256(path.read_bytes()).hexdigest()
    return canonical_sha256(manifest)


@dataclass(frozen=True, slots=True)
class ProposedAction:
    candidate: ActionCandidate
    contract: ScientificActionContract
    mutation_context: ProgramMutationContext | None = None
    proposal_slot_index: int | None = None

    def __post_init__(self) -> None:
        if self.proposal_slot_index is not None and (
            type(self.proposal_slot_index) is not int or self.proposal_slot_index < 0
        ):
            raise ValueError("proposal slot index must be a nonnegative integer")
        if self.contract.action_id != self.candidate.id:
            raise ValueError("action candidate and coder contract ids must match")
        if self.contract.selected_plan_json is not None:
            execution = selected_scientific_plan(self.contract.selected_plan_json)["execution_action"]
            if execution["action_sha256"] != self.candidate.action_sha256:
                raise ValueError("selected plan and proposed execution candidate differ")
        if self.mutation_context is not None:
            if self.contract.research_contract_sha256 != self.mutation_context.research_contract_sha256:
                raise ValueError("proposal and mutation context bind different research contracts")


@dataclass(frozen=True, slots=True)
class ProgramNode:
    id: str
    artifact: Path
    executable_sha256: str
    workspace_sha256: str
    parent_id: str | None
    hypothesis_id: str
    native_metric: float | None
    artifact_valid: bool
    realized_family_sha256: str | None
    hypothesis_statement: str = ""
    scientific_family: dict[str, Any] | None = None
    island_id: str | None = None
    evaluation_summary: dict[str, Any] | None = None
    evaluation_evidence_sha256: str | None = None
    evaluation_evidence_path: Path | None = None
    changed_paths: tuple[str, ...] = ()
    executable_relpath: str = "run.py"
    implementation_observation: dict[str, Any] | None = None


@dataclass(slots=True)
class CampaignAccounting:
    model_calls: int = 0
    tool_calls: int = 0
    input_tokens: int = 0
    output_tokens: int = 0
    score_evaluations: int = 0
    wall_seconds: float = 0.0
    attempted_actions: set[str] = field(default_factory=set)
    realized_families: set[str] = field(default_factory=set)
    node_visits: dict[str, int] = field(default_factory=dict)


@dataclass(frozen=True, slots=True)
class ActionExecution:
    outcome: CoderOutcome
    artifact: Path
    native_metric: float | None
    artifact_valid: bool
    score_evaluations: int
    evaluation_summary: dict[str, Any] | None = None
    evaluation_evidence_sha256: str | None = None
    evaluation_evidence_path: Path | None = None
    evaluation_wall_seconds: float = 0.0
    # Controller-side comparison of the candidate's own public-label estimate
    # with its parent's; when it rejects, no official evaluation was spent.
    internal_validation: dict[str, Any] | None = None


class ActionProposer(Protocol):
    def propose(
        self,
        *,
        version: str,
        nodes: tuple[ProgramNode, ...],
        view: dict,
        count: int,
        directives: tuple[ProposalDirective, ...] = (),
        diagnostics: tuple[dict[str, Any], ...] = (),
    ) -> tuple[list[ProposedAction], dict[str, Any]]: ...


class ActionExecutor(Protocol):
    def config_sha256(
        self, *, proposal: ProposedAction, parent: ProgramNode, step: int
    ) -> str: ...

    def execute(
        self, *, proposal: ProposedAction, parent: ProgramNode, step: int
    ) -> ActionExecution: ...


@dataclass(frozen=True, slots=True)
class CampaignStep:
    step: int
    action_id: str
    code_parent_id: str
    coder_status: str
    native_metric: float | None
    artifact_valid: bool
    graph_sha256: str
    recovery_route: str | None = None


class VSearchCoordinator:
    """Outer Planner-C3-Evaluator loop shared by V1-V4 search policies."""

    def __init__(
        self,
        *,
        version: str,
        strategy: SearchStrategy,
        proposer: ActionProposer,
        executor: ActionExecutor,
        root: ProgramNode,
        max_model_calls: int | None,
        max_score_evaluations: int | None,
        max_tokens: int | None,
        proposal_batch_size: int = 4,
    ) -> None:
        for name, value in (("max_model_calls", max_model_calls),
                            ("max_score_evaluations", max_score_evaluations), ("max_tokens", max_tokens)):
            positive_cap(value, name)
        positive_cap(proposal_batch_size, "proposal_batch_size", nullable=False)
        self.version = version
        self.strategy = strategy
        self.proposer = proposer
        self.executor = executor
        self.nodes = {root.id: root}
        self.actions: dict[str, ProposedAction] = {}
        self.view = DynamicResearchView()
        self.accounting = CampaignAccounting()
        self.steps: list[CampaignStep] = []
        self.max_model_calls = max_model_calls
        self.max_score_evaluations = max_score_evaluations
        self.max_tokens = max_tokens
        self.proposal_batch_size = proposal_batch_size
        self.diagnostics: list[dict[str, Any]] = []
        self.consecutive_empty_expansions = 0
        self.consecutive_truncated_expansions = 0
        self.last_planner_diagnostic: dict | None = None
        # Semantic refusals (ProposalRejected) are journaled, and a refused
        # registered action is retired so it is never selected or executed.
        self.proposal_rejections: list[dict[str, Any]] = []
        self.rejected_actions: set[str] = set()
        self.consecutive_proposal_rejections = 0

    def _schedule_controller_repair(
        self, *, proposal: ProposedAction, route: str
    ) -> str | None:
        """Queue one same-hypothesis repair from immutable failed evidence.

        Failed workspaces never become code parents.  The verified code parent
        remains authoritative, while the retained failed program is supplied as
        diagnostic evidence so the coder can repair rather than rediscover it.
        A repair action is never recursively auto-repaired; this bounds the lane.
        """
        repair_routes = {
            "repair_implementation",
            "optimize_efficiency",
            "contract_preserving_rewrite",
        }
        action = proposal.candidate
        if route not in repair_routes or action.operator == ActionOperator.REPAIR_PROGRAM:
            return None
        if not self.diagnostics or self.diagnostics[-1].get("action_id") != action.id:
            return None
        diagnostic = self.diagnostics[-1]
        if not is_repair_seed(diagnostic):
            return None
        diagnostic_sha256 = diagnostic.get("diagnostic_workspace_sha256")
        if not isinstance(diagnostic_sha256, str) or len(diagnostic_sha256) != 64:
            return None
        repair_id = (
            f"repair_{len(self.steps):04d}_"
            f"{canonical_sha256({'source': action.id, 'route': route})[:10]}"
        )
        failed_code = ""
        diagnostic_root = Path(str(diagnostic.get("artifact", "")))
        failed_executable = diagnostic_root / "run.py"
        if failed_executable.is_file():
            failed_code = failed_executable.read_text(errors="replace")[-16_000:]
        repair_evidence = {
            "source_action_id": action.id,
            "recovery_route": route,
            "reason": diagnostic.get("reason"),
            "contract_violations": diagnostic.get("contract_violations"),
            "validation": diagnostic.get("validation"),
            "diagnostic_workspace_sha256": diagnostic_sha256,
        }
        # Old diagnostics stay on the source action, reachable through this
        # repair's origin edge. Do not relabel them into another dependency kind:
        # that can collide with a distinct evidence_motivation edge. Preserve all
        # other typed edges unchanged; only the current failure is a repair seed.
        dependencies = tuple(d for d in action.dependencies if d.kind != "diagnostic_source") + (
            ResearchDependency(action.id, "diagnostic_source", diagnostic_sha256),
        )
        candidate = ActionCandidate(
            id=repair_id,
            hypothesis_id=action.hypothesis_id,
            operator=ActionOperator.REPAIR_PROGRAM,
            dependencies=dependencies,
            family=action.family,
            predicted_program_quality=action.predicted_program_quality,
            predicted_information_gain=action.predicted_information_gain,
            predicted_reuse_value=action.predicted_reuse_value,
            predicted_validity=max(0.8, action.predicted_validity),
            expected_cost=action.expected_cost,
            hypothesis_statement=action.hypothesis_statement,
            island_id=action.island_id,
        )
        contract = replace(
            proposal.contract,
            action_id=repair_id,
            selected_plan_json=repair_selected_scientific_plan(
                proposal.contract.selected_plan_json, candidate
            ),
            description=(
                proposal.contract.description
                + "\n\nCONTROLLER-ROUTED SAME-HYPOTHESIS REPAIR. Do not propose or implement a new "
                "scientific method. Preserve the verified code_parent and the exact hypothesis/family. "
                "Use the retained failed branch only as diagnostic implementation evidence.\n"
                + controller_repair_directive(route)
                + "\n\nBound failure evidence:\n"
                + json.dumps(repair_evidence, ensure_ascii=False, sort_keys=True)[:10_000]
                + ("\n\nRetained failed run.py tail:\n```python\n" + failed_code + "\n```" if failed_code else "")
            )[-30_000:],
        )
        # A failed migrant realization does not consume the migration packet.
        # Preserve its hash-bound multi-program context across the bounded
        # same-hypothesis repair so the destination island repairs the actual
        # cross-island realization instead of silently degrading to a
        # single-parent rewrite.
        self.actions[repair_id] = ProposedAction(
            candidate, contract, proposal.mutation_context
        )
        self.view.register_action(candidate)
        diagnostic["controller_repair_action_id"] = repair_id
        diagnostic["controller_repair_route"] = route
        return repair_id

    def unavailable_action_sha256s(self) -> frozenset[str]:
        return frozenset(self.accounting.attempted_actions | self.rejected_actions)

    def record_proposal_rejection(self, exc: ProposalRejected, *, stage: str) -> dict[str, Any]:
        """Retire a refused proposal without executing or rebinding it."""
        proposal = self.actions.get(str(exc.record.get("action_id")))
        action_sha256 = proposal.candidate.action_sha256 if proposal is not None else None
        if action_sha256 is not None:
            self.rejected_actions.add(action_sha256)
        record = {**exc.record, "stage": stage, "step": len(self.steps),
                  "action_sha256": action_sha256}
        self.proposal_rejections.append(record)
        self.consecutive_proposal_rejections += 1
        self.view.record_proposal_rejection(record)
        return record

    def _remaining(self) -> SearchBudget:
        used_tokens = self.accounting.input_tokens + self.accounting.output_tokens
        return SearchBudget(
            remaining_model_calls=max(0, remaining(self.max_model_calls, self.accounting.model_calls)),
            remaining_score_evaluations=max(
                0, remaining(self.max_score_evaluations, self.accounting.score_evaluations)
            ),
            remaining_tokens=max(0, remaining(self.max_tokens, used_tokens)),
            remaining_cost_usd=None,
            used_tokens=int(used_tokens),
        )

    def _snapshot(self) -> SearchSnapshot:
        return SearchSnapshot(
            graph_sha256=self.view.sha256,
            actions=tuple(item.candidate for item in self.actions.values()),
            realized_family_sha256s=frozenset(self.accounting.realized_families),
            attempted_action_sha256s=self.unavailable_action_sha256s(),
            node_visits=dict(self.accounting.node_visits),
            budget=self._remaining(),
            step=len(self.steps),
            node_metrics={node.id: node.native_metric for node in self.nodes.values()},
            node_parent_ids={node.id: node.parent_id for node in self.nodes.values()},
            node_family_sha256s={
                node.id: node.realized_family_sha256 for node in self.nodes.values()
            },
            node_island_ids={node.id: node.island_id for node in self.nodes.values()},
            node_artifact_validity={
                node.id: node.artifact_valid for node in self.nodes.values()
            },
            node_executable_sha256s={node.id: node.executable_sha256 for node in self.nodes.values()},
            node_evaluation_summaries={
                node.id: (dict(node.evaluation_summary) if node.evaluation_summary is not None else None)
                for node in self.nodes.values()
            },
        )

    def _drain_strategy_events(self) -> None:
        drain = getattr(self.strategy, "drain_events", None)
        if callable(drain):
            for event in drain():
                self.view.record_strategy_event(event)

    def expand(self) -> int:
        remaining = self._remaining()
        if remaining.remaining_model_calls <= 0 or remaining.remaining_tokens <= 0:
            return 0
        snapshot = self._snapshot()
        directive_builder = getattr(self.strategy, "proposal_directives", None)
        if directive_builder is None:
            parents = sorted(self.nodes)
            directives = tuple(
                ProposalDirective(parents[index % len(parents)])
                for index in range(self.proposal_batch_size)
            )
        else:
            directives = tuple(directive_builder(snapshot, self.proposal_batch_size))
        if not directives:
            self._drain_strategy_events()
            return 0
        selected_ids = {
            source for directive in directives
            for source in (directive.code_parent_id, *directive.intervention_source_ids)
        }
        self.view.record_proposal_selection(
            graph_before_sha256=snapshot.graph_sha256,
            strategy=self.strategy.name,
            policy_sha256=(
                canonical_sha256(self.strategy.config)
                if hasattr(self.strategy, "config") else None
            ),
            directives=[asdict(item) for item in directives],
            parent_evidence={
                node_id: {
                    "executable_sha256": self.nodes[node_id].executable_sha256,
                    "workspace_sha256": self.nodes[node_id].workspace_sha256,
                    "artifact_valid": self.nodes[node_id].artifact_valid,
                    "bootstrap_seed": node_id == "root" and not self.nodes[node_id].artifact_valid,
                    "native_metric": self.nodes[node_id].native_metric,
                    "evaluation_evidence_sha256": self.nodes[node_id].evaluation_evidence_sha256,
                    "parent_visits": snapshot.node_visits.get(node_id, 0),
                    "execution_parent_visits": snapshot.node_visits.get(node_id, 0),
                    "selector_parent_attempts": (
                        self.strategy.parent_attempts.get(node_id, 0)
                        if hasattr(self.strategy, "parent_attempts") else None
                    ),
                }
                for node_id in sorted(selected_ids)
            },
        )
        # Frontier events belong after the selection event so that event binds
        # the pre-selection graph exactly as every other strategy's does.
        self._drain_strategy_events()
        proposals, usage = self.proposer.propose(
            version=self.version,
            nodes=tuple(self.nodes.values()),
            view=self.view.projection(),
            count=len(directives),
            directives=directives,
            diagnostics=tuple(self.diagnostics[-8:]),
        )
        self.accounting.model_calls += int(usage.get("model_calls", 1))
        self.accounting.input_tokens += int(usage.get("input_tokens", 0))
        self.accounting.output_tokens += int(usage.get("output_tokens", 0))
        planner_diagnostic = usage.get("planner_diagnostic")
        self.last_planner_diagnostic = planner_diagnostic if isinstance(planner_diagnostic, dict) else None
        if isinstance(planner_diagnostic, dict):
            self.view.record_planner_batch(planner_diagnostic)
        if (
            exceeded(self.max_model_calls, self.accounting.model_calls)
            or exceeded(self.max_tokens, self.accounting.input_tokens + self.accounting.output_tokens)
        ):
            raise RuntimeError("proposer exceeded the frozen campaign budget")
        staged: list[ProposedAction] = []
        occupied_slots: set[int] = set()
        for index, proposal in enumerate(proposals):
            slot_index = proposal.proposal_slot_index
            if slot_index is None:  # Legacy/test proposers return a complete ordered batch.
                slot_index = index
            if slot_index >= len(directives) or slot_index in occupied_slots:
                raise ValueError("planner returned invalid or duplicate proposal slot")
            occupied_slots.add(slot_index)
            action = proposal.candidate
            if action.id in self.actions or action.id in {item.candidate.id for item in staged} or action.action_sha256 in {
                item.candidate.action_sha256 for item in (*self.actions.values(), *staged)
            }:
                continue
            code_parent = next(
                item for item in action.dependencies if item.kind == "code_parent"
            )
            parent = self.nodes.get(code_parent.source_id)
            if parent is None or parent.executable_sha256 != code_parent.source_sha256:
                raise ValueError("proposal does not bind an existing executable parent")
            directive = directives[slot_index]
            if code_parent.source_id != directive.code_parent_id:
                raise ValueError("planner changed the controller-selected code parent")
            fixed_hypothesis = getattr(directive, "hypothesis_id", None)
            if fixed_hypothesis is not None and action.hypothesis_id != fixed_hypothesis:
                raise ProposalRejected("planner changed the controller-selected hypothesis",
                                       code="controller_hypothesis_mismatch", action_id=action.id)
            if action.island_id != directive.island_id:
                raise ValueError("planner changed the controller-selected island")
            intervention_ids = tuple(
                dependency.source_id for dependency in action.dependencies
                if dependency.kind == "intervention_source"
            )
            if intervention_ids != directive.intervention_source_ids:
                raise ValueError("planner changed controller-owned migration sources")
            diagnostics_by_id = {
                str(item.get("action_id")): item for item in self.diagnostics
            }
            for dependency in action.dependencies:
                if dependency.kind != "diagnostic_source":
                    continue
                diagnostic = diagnostics_by_id.get(dependency.source_id)
                if diagnostic is None or diagnostic.get(
                    "diagnostic_workspace_sha256"
                ) != dependency.source_sha256:
                    raise ValueError("proposal cites an unknown or hash-mismatched diagnostic source")
            if action.operator == ActionOperator.REPAIR_PROGRAM:
                origins = [d for d in action.dependencies if d.kind == "diagnostic_source"]
                if len(origins) != 1 or origins[0].source_id not in self.actions:
                    raise ValueError("repair requires exactly one known diagnostic origin")
                if not is_repair_seed(diagnostics_by_id[origins[0].source_id]):
                    raise ValueError("diagnostic is evidence-only, not a retained failed repair seed")
                source = self.actions[origins[0].source_id].candidate
                source_parent = next(d for d in source.dependencies if d.kind == "code_parent")
                if source_parent != code_parent:
                    raise ValueError("repair changed the diagnostic origin code parent")
            staged.append(proposal)
        # No action enters state until every returned proposal passes binding
        # checks. A later invalid slot must not leave a half-registered batch.
        for proposal in staged:
            self.actions[proposal.candidate.id] = proposal
            self.view.register_action(proposal.candidate)
        added = len(staged)
        # A batch the provider cut off at the output cap is a budget problem, not an
        # invalid proposal: the planner was still reasoning when the response ended.
        # It is counted on its own so the session can lower the reasoning effort
        # and try again, instead of ending the run after two of them as if the
        # model had twice returned nonsense.
        truncated = (added == 0 and isinstance(self.last_planner_diagnostic, dict)
                     and self.last_planner_diagnostic.get("provider_finish_reason") in {"length", "max_tokens"})
        self.consecutive_truncated_expansions = self.consecutive_truncated_expansions + 1 if truncated else 0
        self.consecutive_empty_expansions = (
            self.consecutive_empty_expansions + 1 if added == 0 and not truncated else 0
        )
        return added

    def step(self) -> CampaignStep:
        snapshot = self._snapshot()
        decision = self.strategy.select(snapshot)
        proposal = self.actions[decision.action_id]
        action = proposal.candidate
        code_parent = next(item for item in action.dependencies if item.kind == "code_parent")
        parent = self.nodes[code_parent.source_id]
        self.view.record_decision(decision)
        config_sha256 = self.executor.config_sha256(
            proposal=proposal, parent=parent, step=len(self.steps)
        )
        self.view.record_coder_started(
            action_id=action.id,
            executable_parent_sha256=parent.executable_sha256,
            coder_config_sha256=config_sha256,
        )
        execution = self.executor.execute(
            proposal=proposal, parent=parent, step=len(self.steps)
        )
        outcome = execution.outcome
        if outcome.config_sha256 != config_sha256:
            raise ValueError("executed coder config differs from pre-execution binding")
        native_metric = execution.native_metric
        artifact_valid = execution.artifact_valid
        score_evaluations = execution.score_evaluations
        self.view.record_coder_outcome(action_id=action.id, outcome=outcome)
        if execution.evaluation_evidence_sha256 is not None:
            self.view.record_scientific_evaluation(
                action_id=action.id,
                evidence_sha256=execution.evaluation_evidence_sha256,
                evaluation_summary=execution.evaluation_summary or {},
                score_evaluations=score_evaluations,
            )
        self.accounting.model_calls += outcome.model_calls
        self.accounting.tool_calls += outcome.tool_calls
        self.accounting.input_tokens += outcome.input_tokens
        self.accounting.output_tokens += outcome.output_tokens
        self.accounting.wall_seconds += outcome.wall_seconds + execution.evaluation_wall_seconds
        self.accounting.score_evaluations += score_evaluations
        self.accounting.attempted_actions.add(action.action_sha256)
        self.accounting.node_visits[parent.id] = self.accounting.node_visits.get(parent.id, 0) + 1
        if exceeded(self.max_model_calls, self.accounting.model_calls):
            raise RuntimeError("coder exceeded the frozen campaign model-call budget")
        if exceeded(self.max_tokens, self.accounting.input_tokens + self.accounting.output_tokens):
            raise RuntimeError("coder exceeded the frozen campaign token budget")
        if exceeded(self.max_score_evaluations, self.accounting.score_evaluations):
            raise RuntimeError("coder exceeded the frozen score-evaluation budget")
        family_previously_realized = action.family.sha256 in self.accounting.realized_families
        program_node_id = None
        observed_implementation = None
        if outcome.implementation_real and artifact_valid:
            from .learning_evidence import program_observation
            observed_implementation = program_observation(
                (parent.artifact / parent.executable_relpath).read_text(errors="replace"),
                (execution.artifact / parent.executable_relpath).read_text(errors="replace"),
                parent.evaluation_summary or {}, execution.evaluation_summary or {})
            self.view.record_implementation_observation(
                action_id=action.id, parent_program_sha256=parent.executable_sha256,
                candidate_program_sha256=outcome.final_sha256,
                evaluation_evidence_sha256=execution.evaluation_evidence_sha256,
                observation=observed_implementation)
            family_sha = action.family.sha256
            self.accounting.realized_families.add(family_sha)
            node = ProgramNode(
                id=action.id,
                artifact=execution.artifact,
                executable_sha256=outcome.final_sha256 or "",
                workspace_sha256=outcome.final_workspace_sha256,
                parent_id=parent.id,
                hypothesis_id=action.hypothesis_id,
                native_metric=native_metric,
                artifact_valid=True,
                realized_family_sha256=family_sha,
                hypothesis_statement=action.hypothesis_statement,
                scientific_family=asdict(action.family),
                island_id=action.island_id,
                evaluation_summary=execution.evaluation_summary,
                evaluation_evidence_sha256=execution.evaluation_evidence_sha256,
                evaluation_evidence_path=execution.evaluation_evidence_path,
                changed_paths=tuple(outcome.changed_paths),
                executable_relpath=parent.executable_relpath,
                implementation_observation=observed_implementation,
            )
            self.nodes[node.id] = node
            program_node_id = node.id
        else:
            artifact_fields = diagnostic_artifact_fields(
                relpath=outcome.diagnostic_artifact_relpath,
                digest=outcome.diagnostic_workspace_sha256,
            )
            diagnostic_path = (
                execution.artifact.parent / outcome.diagnostic_artifact_relpath
                if outcome.diagnostic_artifact_relpath is not None
                else execution.artifact
            )
            if not bool((execution.internal_validation or {}).get("rejected")):
                self.diagnostics.append({
                    "action_id": action.id,
                    "code_parent_id": parent.id,
                    "hypothesis_id": action.hypothesis_id,
                    "hypothesis_statement": action.hypothesis_statement,
                    "scientific_family": asdict(action.family),
                    "island_id": action.island_id,
                    "artifact": str(diagnostic_path),
                    **artifact_fields,
                    "diagnostic_workspace_sha256": (
                        outcome.diagnostic_workspace_sha256
                        or outcome.final_workspace_sha256
                    ),
                    "changed_paths": list(outcome.changed_paths),
                    "status": outcome.status.value,
                    "reason": outcome.reason,
                    "contract_violations": list(outcome.contract_violations),
                    "validation": outcome.validation,
                    "evaluation_summary": execution.evaluation_summary,
                    "evaluation_evidence_sha256": execution.evaluation_evidence_sha256,
                    "trajectory_sha256": outcome.trajectory_sha256,
                })
        self_rejected = bool((execution.internal_validation or {}).get("rejected"))
        route = classify_recovery_route(
            outcome,
            artifact_valid=artifact_valid,
            native_metric=native_metric,
            parent_metric=parent.native_metric,
            duplicate_scientific_family=family_previously_realized,
            internal_validation_rejected=self_rejected,
            evaluation_invalid=(
                isinstance((execution.evaluation_summary or {}).get("public_diagnostic"), dict)
                and (execution.evaluation_summary or {})["public_diagnostic"].get("category")
                in {"evaluation_bridge", "official_evaluation", "artifact_attestation", "grader_environment"}
            ),
        )
        observation = SearchObservation(
            action_id=action.id,
            action_sha256=action.action_sha256,
            graph_before_sha256=decision.graph_sha256,
            graph_after_sha256=self.view.sha256,
            executable=outcome.status.value == "success",
            score_bearing=native_metric is not None,
            artifact_valid=artifact_valid,
            scientific_disposition="inconclusive",
            native_metric=native_metric,
            realized_family_sha256=(action.family.sha256 if outcome.implementation_real else None),
            model_calls=outcome.model_calls,
            tokens=outcome.input_tokens + outcome.output_tokens,
            cost_usd=0.0,
            code_parent_id=parent.id,
            program_node_id=program_node_id,
            island_id=action.island_id,
            evaluation_evidence_sha256=execution.evaluation_evidence_sha256,
            evaluation_summary=execution.evaluation_summary,
            failure_summary=(self.diagnostics[-1] if program_node_id is None and self.diagnostics else None),
            intervention_source_ids=tuple(
                dependency.source_id for dependency in action.dependencies
                if dependency.kind == "intervention_source"
            ),
            recovery_route=route.route,
            implementation_observation=observed_implementation,
            program_sha256=outcome.final_sha256,
            internal_validation=execution.internal_validation,
        )
        migration_history = getattr(self.strategy, "migration_history", ())
        migration_count = len(migration_history)
        unconsumed_before = {
            packet.get("migration_packet_sha256")
            for packet in migration_history
            if not packet.get("consumed", False)
        }
        self.strategy.observe(observation)
        self._drain_strategy_events()
        for packet in getattr(self.strategy, "migration_history", ())[migration_count:]:
            self.view.record_migration(packet)
        for packet in getattr(self.strategy, "migration_history", ()):
            if (
                packet.get("migration_packet_sha256") in unconsumed_before
                and packet.get("consumed") is True
                and packet.get("consumed_by_action_id") == action.id
            ):
                self.view.record_migration_consumed(
                    migration_packet_sha256=packet["migration_packet_sha256"],
                    action_id=action.id,
                )
        self.view.record_observation(observation)
        self.view.record_recovery_route(
            action_id=action.id,
            route=route.route,
            reason_codes=route.reason_codes,
            route_sha256=route.route_sha256,
            source_outcome_sha256=route.source_outcome_sha256,
        )
        if self.diagnostics and self.diagnostics[-1].get("action_id") == action.id:
            self.diagnostics[-1]["recovery_route"] = route.route
            self.diagnostics[-1]["recovery_reason_codes"] = list(route.reason_codes)
        self._schedule_controller_repair(proposal=proposal, route=route.route)
        row = CampaignStep(
            step=len(self.steps),
            action_id=action.id,
            code_parent_id=parent.id,
            coder_status=outcome.status.value,
            native_metric=native_metric,
            artifact_valid=artifact_valid,
            graph_sha256=self.view.sha256,
            recovery_route=route.route,
        )
        # Read by the session to notice a scorer that cannot run; not checkpointed.
        self.last_evaluation_summary = execution.evaluation_summary
        self.steps.append(row)
        self.consecutive_proposal_rejections = 0
        return row

    def summary(self) -> dict:
        valid_nodes = [
            node for node in self.nodes.values()
            if node.artifact_valid and node.native_metric is not None
        ]
        best_node = max(valid_nodes, key=lambda node: node.native_metric) if valid_nodes else None
        evolved_nodes = [node for node in valid_nodes if node.id != "root"]
        best_evolved = (
            max(evolved_nodes, key=lambda node: node.native_metric)
            if evolved_nodes else None
        )
        return {
            "schema_version": "evolvcoder-v-search-campaign-v2",
            "version": self.version,
            "steps": [asdict(item) for item in self.steps],
            "accounting": {
                **asdict(self.accounting),
                "attempted_actions": sorted(self.accounting.attempted_actions),
                "realized_families": sorted(self.accounting.realized_families),
            },
            # The verified starter is a real incumbent.  Search may fail to
            # improve it, but it must never disappear from best-so-far.
            "root_native_metric": self.nodes["root"].native_metric,
            "best_native_metric": best_node.native_metric if best_node else None,
            "best_node_id": best_node.id if best_node else None,
            "best_evolved_native_metric": (
                best_evolved.native_metric if best_evolved else None
            ),
            "best_evolved_node_id": best_evolved.id if best_evolved else None,
            "graph": self.view.projection(),
            "diagnostics": list(self.diagnostics),
            "consecutive_empty_expansions": self.consecutive_empty_expansions,
            "consecutive_truncated_expansions": self.consecutive_truncated_expansions,
            "proposal_rejections": list(self.proposal_rejections),
            "rejected_actions": sorted(self.rejected_actions),
            "consecutive_proposal_rejections": self.consecutive_proposal_rejections,
            "search_strategy_state": (
                self.strategy.checkpoint_state()
                if hasattr(self.strategy, "checkpoint_state") else None
            ),
            "frontier_report": (
                self.strategy.frontier_report()
                if callable(getattr(self.strategy, "frontier_report", None)) else None
            ),
            "summary_sha256": canonical_sha256({
                "version": self.version,
                "steps": [asdict(item) for item in self.steps],
                "graph_sha256": self.view.sha256,
            }),
        }

    def checkpoint(self) -> dict[str, Any]:
        """Complete JSON-safe state. Artifacts remain immutable on disk."""
        payload = {
            "schema_version": "evolvcoder-v-search-checkpoint-v2",
            "version": self.version,
            "limits": {
                "max_model_calls": self.max_model_calls,
                "max_score_evaluations": self.max_score_evaluations,
                "max_tokens": self.max_tokens,
                "proposal_batch_size": self.proposal_batch_size,
            },
            "nodes": [
                {
                    **asdict(node),
                    "artifact": str(node.artifact),
                    "evaluation_evidence_path": (
                        str(node.evaluation_evidence_path)
                        if node.evaluation_evidence_path is not None else None
                    ),
                }
                for node in self.nodes.values()
            ],
            "actions": [
                {
                    "candidate": asdict(item.candidate),
                    "contract": item.contract.to_dict(),
                    "proposal_slot_index": item.proposal_slot_index,
                    "mutation_context": (
                        asdict(item.mutation_context)
                        if item.mutation_context is not None else None
                    ),
                }
                for item in self.actions.values()
            ],
            "accounting": {
                **asdict(self.accounting),
                "attempted_actions": sorted(self.accounting.attempted_actions),
                "realized_families": sorted(self.accounting.realized_families),
            },
            "steps": [asdict(item) for item in self.steps],
            "events": [asdict(item) for item in self.view.events],
            "diagnostics": list(self.diagnostics),
            "consecutive_empty_expansions": self.consecutive_empty_expansions,
            "consecutive_truncated_expansions": self.consecutive_truncated_expansions,
            "proposal_rejections": list(self.proposal_rejections),
            "rejected_actions": sorted(self.rejected_actions),
            "consecutive_proposal_rejections": self.consecutive_proposal_rejections,
            "proposer_state": (
                self.proposer.checkpoint_state()
                if hasattr(self.proposer, "checkpoint_state") else None
            ),
            "strategy_state": (
                self.strategy.checkpoint_state()
                if hasattr(self.strategy, "checkpoint_state") else None
            ),
        }
        payload["checkpoint_sha256"] = canonical_sha256(payload)
        return payload

    @classmethod
    def restore(
        cls, *, strategy: SearchStrategy, proposer: ActionProposer,
        executor: ActionExecutor, payload: dict[str, Any],
    ) -> "VSearchCoordinator":
        payload = dict(payload)
        checkpoint_sha256 = payload.pop("checkpoint_sha256", None)
        if checkpoint_sha256 is not None and checkpoint_sha256 != canonical_sha256(payload):
            raise ValueError("campaign checkpoint hash mismatch")
        if (
            payload.get("schema_version") == "evolvcoder-v-search-checkpoint-v2"
            and checkpoint_sha256 is None
        ):
            raise ValueError("v2 campaign checkpoint lacks its canonical hash")
        if payload.get("schema_version") not in {
            "evolvcoder-v-search-checkpoint-v1", "evolvcoder-v-search-checkpoint-v2"
        }:
            raise ValueError("unsupported campaign checkpoint")
        nodes = [ProgramNode(**{
            **row,
            "artifact": Path(row["artifact"]),
            "evaluation_evidence_path": (
                Path(row["evaluation_evidence_path"])
                if row.get("evaluation_evidence_path") else None
            ),
            "changed_paths": tuple(row.get("changed_paths", ())),
        }) for row in payload["nodes"]]
        if not nodes:
            raise ValueError("checkpoint has no root")
        for node in nodes:
            executable = node.artifact / node.executable_relpath
            if not executable.is_file() or hashlib.sha256(
                executable.read_bytes()
            ).hexdigest() != node.executable_sha256:
                raise ValueError(f"checkpoint node artifact hash mismatch: {node.id}")
            if artifact_workspace_sha256(node.artifact) != node.workspace_sha256:
                raise ValueError(f"checkpoint node workspace hash mismatch: {node.id}")
            if node.evaluation_evidence_sha256 is not None:
                evidence_path = node.evaluation_evidence_path
                if evidence_path is None or not evidence_path.is_file() or hashlib.sha256(
                    evidence_path.read_bytes()
                ).hexdigest() != node.evaluation_evidence_sha256:
                    raise ValueError(f"checkpoint node evidence hash mismatch: {node.id}")
        limits = payload["limits"]
        value = cls(
            version=payload["version"], strategy=strategy, proposer=proposer,
            executor=executor, root=nodes[0], **limits,
        )
        value.nodes = {node.id: node for node in nodes}
        value.actions = {}
        for row in payload["actions"]:
            raw = row["candidate"]
            family = ScientificFamily(**{
                **raw["family"],
                "capacity_controls": tuple(raw["family"].get("capacity_controls", ())),
            })
            candidate = ActionCandidate(
                id=raw["id"], hypothesis_id=raw["hypothesis_id"],
                operator=ActionOperator(raw["operator"]),
                dependencies=tuple(ResearchDependency(**item) for item in raw["dependencies"]),
                family=family,
                predicted_program_quality=raw["predicted_program_quality"],
                predicted_information_gain=raw["predicted_information_gain"],
                predicted_reuse_value=raw["predicted_reuse_value"],
                predicted_validity=raw["predicted_validity"],
                expected_cost=raw["expected_cost"],
                hypothesis_statement=raw.get("hypothesis_statement", ""),
                island_id=raw.get("island_id"),
                action_sha256=raw["action_sha256"],
            )
            contract = ScientificActionContract(**{
                key: (tuple(item) if isinstance(item, list) else item)
                for key, item in row["contract"].items()
            })
            mutation_raw = row.get("mutation_context")
            mutation_context = None
            if mutation_raw is not None:
                mutation_context = ProgramMutationContext(
                    request_id=mutation_raw["request_id"],
                    research_contract_sha256=mutation_raw["research_contract_sha256"],
                    kernel=mutation_raw["kernel"],
                    stage=int(mutation_raw["stage"]),
                    annealing_lambda=float(mutation_raw["annealing_lambda"]),
                    parent_particle_id=mutation_raw["parent_particle_id"],
                    parent_program_sha256=mutation_raw["parent_program_sha256"],
                    inspirations=tuple(
                        MutationInspiration(**item)
                        for item in mutation_raw.get("inspirations", [])
                    ),
                    lineage_evidence=tuple(mutation_raw.get("lineage_evidence", [])),
                )
            value.actions[candidate.id] = ProposedAction(
                candidate, contract, mutation_context, row.get("proposal_slot_index")
            )
        accounting = dict(payload["accounting"])
        accounting["attempted_actions"] = set(accounting["attempted_actions"])
        accounting["realized_families"] = set(accounting["realized_families"])
        value.accounting = CampaignAccounting(**accounting)
        value.steps = [CampaignStep(**row) for row in payload["steps"]]
        value.diagnostics = list(payload.get("diagnostics", []))
        value.consecutive_empty_expansions = int(
            payload.get("consecutive_empty_expansions", 0)
        )
        value.consecutive_truncated_expansions = int(
            payload.get("consecutive_truncated_expansions", 0)
        )
        value.proposal_rejections = [dict(item) for item in payload.get("proposal_rejections", [])]
        value.rejected_actions = set(payload.get("rejected_actions", []))
        value.consecutive_proposal_rejections = int(
            payload.get("consecutive_proposal_rejections", 0)
        )
        view = DynamicResearchView()
        view._actions = {key: item.candidate for key, item in value.actions.items()}
        prior = None
        for raw in payload["events"]:
            event = SearchEvent(**raw)
            if event.sequence != len(view._events) or event.previous_event_sha256 != prior:
                raise ValueError("checkpoint event chain is broken")
            view._events.append(event)
            prior = event.event_sha256
        value.view = view
        strategy_state = payload.get("strategy_state")
        if (isinstance(strategy_state, dict)
                and strategy_state.get("schema_version") == "evolvcoder-search-checkpoint-v1"
                and not getattr(strategy, "requires_checkpoint", False)):
            raise ValueError("search plugin checkpoint requires a compatible plugin restorer")
        if strategy_state is None and getattr(strategy, "requires_checkpoint", False):
            raise ValueError("search plugin checkpoint state is required for resume")
        if strategy_state is not None and hasattr(strategy, "restore_state"):
            strategy.restore_state(strategy_state)
        elif hasattr(strategy, "cursor"):
            strategy.cursor = len(value.steps)
        proposer_state = payload.get("proposer_state")
        if proposer_state is not None:
            if not hasattr(proposer, "restore_state"):
                raise ValueError("checkpoint proposer state has no compatible restorer")
            proposer.restore_state(proposer_state)
        elif hasattr(proposer, "restore_state"):
            raise ValueError("checkpoint lacks required proposer state; refusing ambiguous resume")
        return value
