from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict, dataclass, field, fields
from enum import StrEnum
from pathlib import PurePosixPath
from typing import Any

from .contracts import ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily


def coder_sha256(value: Any) -> str:
    payload = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


# Kept as canonical JSON in the frozen contract: nested provider-facing objects
# are fresh projections, never mutable aliases of controller-owned provenance.
_SELECTED_PLAN_SCHEMA = "swarmrsi-selected-scientific-plan-v1"


def _selected_plan_action(raw: Any) -> ActionCandidate:
    if not isinstance(raw, dict) or set(raw) != {item.name for item in fields(ActionCandidate)}:
        raise ValueError("selected plan requires a complete action record")
    for key in ("id", "hypothesis_id", "hypothesis_statement", "action_sha256"):
        if not isinstance(raw[key], str):
            raise ValueError("selected plan action text must be strings")
    if len(raw["action_sha256"]) != 64:
        raise ValueError("selected plan requires an explicit action digest")
    if raw["island_id"] is not None and not isinstance(raw["island_id"], str):
        raise ValueError("selected plan island id must be a string or null")
    for key in ("predicted_program_quality", "predicted_information_gain",
                "predicted_reuse_value", "predicted_validity", "expected_cost"):
        if type(raw[key]) not in (int, float):
            raise ValueError("selected plan predictions must be numeric")
    family = raw["family"]
    if not isinstance(family, dict) or set(family) != {item.name for item in fields(ScientificFamily)}:
        raise ValueError("selected plan requires a complete scientific family")
    if any(not isinstance(family[key], str) for key in family if key != "capacity_controls"):
        raise ValueError("selected plan family fields must be strings")
    controls = family["capacity_controls"]
    if isinstance(controls, str):  # one control written bare; never written back into `family`
        controls = [controls]
    if not isinstance(controls, list) or any(not isinstance(item, str) for item in controls):
        raise ValueError("selected plan capacity controls must be a list of strings")
    dependencies = raw["dependencies"]
    if not isinstance(dependencies, list):
        raise ValueError("selected plan dependencies must be a list")
    for dependency in dependencies:
        if (not isinstance(dependency, dict)
                or set(dependency) != {item.name for item in fields(ResearchDependency)}
                or any(not isinstance(item, str) for item in dependency.values())):
            raise ValueError("selected plan requires complete typed dependencies")
    return ActionCandidate(**{
        **raw,
        "operator": ActionOperator(raw["operator"]),
        "dependencies": tuple(ResearchDependency(**item) for item in dependencies),
        "family": ScientificFamily(**{**family, "capacity_controls": tuple(controls)}),
    })


def selected_scientific_plan(plan_json: str) -> dict[str, Any]:
    """Decode and verify a fresh lossless projection, including its full-text seal."""
    if not isinstance(plan_json, str):
        raise ValueError("selected plan must be immutable canonical JSON text")
    try:
        record = json.loads(plan_json)
        canonical = json.dumps(record, ensure_ascii=False, sort_keys=True,
                               separators=(",", ":"), allow_nan=False)
    except (ValueError, TypeError) as exc:
        raise ValueError("selected plan must be finite canonical JSON") from exc
    if canonical != plan_json:
        raise ValueError("selected plan JSON is not canonical")
    if not isinstance(record, dict) or set(record) != {
        "schema_version", "research_action", "execution_action",
        "implementation_description", "plan_sha256",
    }:
        raise ValueError("selected plan record is incomplete")
    if record["schema_version"] != _SELECTED_PLAN_SCHEMA:
        raise ValueError("unsupported selected plan schema")
    if not isinstance(record["implementation_description"], str):
        raise ValueError("selected plan implementation description must be text")
    body = {key: value for key, value in record.items() if key != "plan_sha256"}
    if record["plan_sha256"] != coder_sha256(body):
        raise ValueError("selected plan digest does not match complete content")
    research = _selected_plan_action(record["research_action"])
    execution = _selected_plan_action(record["execution_action"])
    if execution.action_sha256 != research.action_sha256:
        # Only a controller repair may have a new execution identity while
        # retaining the original fixed scientific plan and migration binding.
        original_edges = tuple(d for d in research.dependencies if d.kind != "diagnostic_source")
        retained_edges = tuple(d for d in execution.dependencies if d.kind != "diagnostic_source")
        diagnostics = tuple(d for d in execution.dependencies if d.kind == "diagnostic_source")
        if (execution.operator != ActionOperator.REPAIR_PROGRAM
                or research.operator == ActionOperator.REPAIR_PROGRAM
                or execution.id == research.id
                or execution.hypothesis_id != research.hypothesis_id
                or execution.hypothesis_statement != research.hypothesis_statement
                or execution.family != research.family
                or execution.island_id != research.island_id
                or retained_edges != original_edges
                or len(diagnostics) != 1
                or diagnostics[0].source_id != research.id):
            raise ValueError("selected plan repair must preserve its scientific origin and typed edges")
    return record


def make_selected_scientific_plan(
    candidate: ActionCandidate, implementation_description: str, *,
    research_action: ActionCandidate | None = None,
) -> str:
    body = {
        "schema_version": _SELECTED_PLAN_SCHEMA,
        "research_action": asdict(research_action or candidate),
        "execution_action": asdict(candidate),
        "implementation_description": implementation_description,
    }
    result = json.dumps({**body, "plan_sha256": coder_sha256(body)},
                        ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)
    selected_scientific_plan(result)
    return result


def repair_selected_scientific_plan(plan_json: str | None, candidate: ActionCandidate) -> str | None:
    if plan_json is None:
        return None
    record = selected_scientific_plan(plan_json)
    return make_selected_scientific_plan(
        candidate, record["implementation_description"],
        research_action=_selected_plan_action(record["research_action"]),
    )


class CoderVersion(StrEnum):
    C0 = "C0"
    C1 = "C1"
    C2 = "C2"
    C3 = "C3"


class CoderExitStatus(StrEnum):
    SUCCESS = "success"
    IMPLEMENTATION_FAILED = "implementation_failed"
    PLAN_BLOCKED = "plan_blocked"
    EVALUATION_INVALID = "evaluation_invalid"


@dataclass(frozen=True, slots=True)
class ValidationCommand:
    tier: str
    argv: tuple[str, ...]
    timeout_seconds: float = 60.0
    cwd: str = "."
    result_json_path: str | None = None
    required_truthy_json_paths: tuple[str, ...] = ()
    sandboxed: bool = False
    # Advisory checks are recorded for audit but cannot block an executable
    # candidate from reaching the scientific evaluator.
    blocking: bool = True
    # Infrastructure/evidence-read failures retry this exact command against
    # the unchanged candidate. Candidate failures are never retried here.
    max_harness_retries: int = 0

    def __post_init__(self) -> None:
        if self.tier not in {"syntax", "static", "import", "unit", "smoke", "full"}:
            raise ValueError(f"unknown validation tier: {self.tier}")
        if not self.argv or self.timeout_seconds <= 0:
            raise ValueError("validation command must be non-empty and bounded")
        _validate_relative(self.cwd, field_name="validation cwd", allow_dot=True)
        if self.result_json_path is not None:
            _validate_relative(self.result_json_path, field_name="validation result path")
        if self.required_truthy_json_paths and self.result_json_path is None:
            raise ValueError("JSON evidence checks require result_json_path")
        if not isinstance(self.max_harness_retries, int) or not 0 <= self.max_harness_retries <= 5:
            raise ValueError("validation harness retries must be an integer in [0, 5]")


def _validate_relative(value: str, *, field_name: str, allow_dot: bool = False) -> None:
    path = PurePosixPath(value)
    if (not value or path.is_absolute() or ".." in path.parts or "\x00" in value):
        raise ValueError(f"{field_name} must be a safe relative path")
    if not allow_dot and value in {".", ""}:
        raise ValueError(f"{field_name} must name a file or directory")


@dataclass(frozen=True, slots=True)
class ScientificActionContract:
    """Controller-owned boundary between scientific choice and implementation."""

    action_id: str
    description: str
    allowed_write_paths: tuple[str, ...]
    readable_paths: tuple[str, ...] = ("*", "**/*")
    protected_paths: tuple[str, ...] = ()
    required_symbols: tuple[str, ...] = ()
    forbidden_symbols: tuple[str, ...] = ()
    required_call_names: tuple[str, ...] = ()
    forbidden_call_names: tuple[str, ...] = ()
    required_import_roots: tuple[str, ...] = ()
    forbidden_import_roots: tuple[str, ...] = ()
    required_source_patterns: tuple[str, ...] = ()
    preserved_source_patterns: tuple[str, ...] = ()
    hypothesis_id: str | None = None
    research_contract_sha256: str | None = None
    selected_plan_json: str | None = None
    # Optional per-action ceiling on the coder's model and tool calls. A sweep
    # episode changes one constant of the incumbent and must not be allowed to
    # spend a full implementation budget on it. None means the session limits.
    coder_limits: dict[str, int] | None = None

    def __post_init__(self) -> None:
        if not self.action_id.strip() or not self.description.strip():
            raise ValueError("scientific action contract is incomplete")
        if self.coder_limits is not None:
            if (not isinstance(self.coder_limits, dict) or set(self.coder_limits) - {"model_calls", "tool_calls"}
                    or any(type(v) is not int or v < 1 for v in self.coder_limits.values())):
                raise ValueError("coder_limits must map model_calls/tool_calls to positive integers")
        if not self.allowed_write_paths:
            raise ValueError("at least one writable path is required")
        for value in (*self.allowed_write_paths, *self.protected_paths):
            _validate_relative(value, field_name="contract path")
        if set(self.allowed_write_paths) & set(self.protected_paths):
            raise ValueError("a path cannot be both writable and protected")
        for pattern in (*self.required_source_patterns, *self.preserved_source_patterns):
            if not pattern:
                raise ValueError("source patterns cannot be empty")
        if (self.hypothesis_id is None) != (self.research_contract_sha256 is None):
            raise ValueError("hypothesis and research contract bindings must appear together")
        if self.research_contract_sha256 is not None and len(self.research_contract_sha256) != 64:
            raise ValueError("research contract binding must be a sha256")
        if self.selected_plan_json is not None:
            record = selected_scientific_plan(self.selected_plan_json)
            research, execution = record["research_action"], record["execution_action"]
            if (execution["id"] != self.action_id
                    or research["hypothesis_id"] != self.hypothesis_id
                    or research["action_sha256"] != self.research_contract_sha256):
                raise ValueError("selected plan and coder contract bindings differ")

    def to_dict(self) -> dict[str, Any]:
        """Preserve legacy serialized identities when the optional plan is absent."""
        value = asdict(self)
        if self.selected_plan_json is None:
            value.pop("selected_plan_json")
        if self.coder_limits is None:
            # Same rule as selected_plan_json: an absent optional field must not
            # change the identity of contracts recorded before it existed.
            value.pop("coder_limits", None)
        return value


@dataclass(frozen=True, slots=True)
class MutationInspiration:
    particle_id: str
    program_sha256: str
    primary_metric: float
    code_excerpt: str

    def __post_init__(self) -> None:
        if not self.particle_id or len(self.program_sha256) != 64:
            raise ValueError("mutation inspiration must bind a particle and program")
        if not math.isfinite(self.primary_metric) or not self.code_excerpt:
            raise ValueError("mutation inspiration must contain finite evidence and code")


@dataclass(frozen=True, slots=True)
class ProgramMutationContext:
    request_id: str
    research_contract_sha256: str
    kernel: str
    stage: int
    annealing_lambda: float
    parent_particle_id: str
    parent_program_sha256: str
    inspirations: tuple[MutationInspiration, ...] = ()
    lineage_evidence: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        kernels = {
            "local_diff", "structural_rewrite", "alternative_realization",
            "cross_program_inspiration",
        }
        if not self.request_id or self.kernel not in kernels or self.stage <= 0:
            raise ValueError("invalid program mutation request")
        if not 0 <= self.annealing_lambda <= 1:
            raise ValueError("annealing lambda must be in [0, 1]")
        if len(self.research_contract_sha256) != 64 or len(self.parent_program_sha256) != 64:
            raise ValueError("mutation context hashes must be sha256")
        if not self.parent_particle_id:
            raise ValueError("mutation context must bind a parent particle")
        uses_inspiration = self.kernel == "cross_program_inspiration"
        if uses_inspiration != bool(self.inspirations):
            raise ValueError(
                "cross-program kernels require inspirations and single-parent kernels forbid them"
            )


@dataclass(frozen=True, slots=True)
class CoderConfig:
    version: CoderVersion
    seed: int
    model: str
    route: str
    executable_parent: str
    expected_parent_sha256: str
    action: ScientificActionContract
    validation: tuple[ValidationCommand, ...]
    max_model_calls: int
    max_tool_calls: int
    max_input_tokens: int
    max_output_tokens: int
    max_wall_seconds: float
    max_repairs: int
    # Once the bounded exploration phase is over and the executable is still
    # unchanged, only this many further diagnostic runs are accepted before an
    # implementation step is required. None keeps the pre-R34 unbounded
    # behaviour and the pre-R34 config identity; the session sets it explicitly.
    max_diagnostic_runs_after_checkpoint: int | None = None
    # Bounded re-sends of the same request after a transient transport failure
    # (connection reset, incomplete read, gateway timeout). A retry is not a
    # model call: nothing was returned. None keeps the pre-R34 behaviour of
    # failing the action on the first transport error, and the pre-R34 identity.
    max_transport_retries: int | None = None
    # Opt-in: stop exploration before the hard action deadline and validate the
    # current implementation inside the reserved time. None preserves identity.
    finalization_reserve_seconds: float | None = None
    mutation_context: ProgramMutationContext | None = None
    allowed_command_prefixes: tuple[tuple[str, ...], ...] = ()
    allowed_tools: tuple[str, ...] = (
        "list", "inspect", "search", "edit", "write", "append", "delete",
        "run", "diagnose", "repair"
    )
    # alias -> absolute controller-owned directory.  Mounts are readable by
    # tools but are never copied into, or writable from, the coder workspace.
    read_only_mounts: tuple[tuple[str, str], ...] = ()
    tool_command_wrapper: tuple[str, ...] = ()
    tool_command_path_rewrites: tuple[tuple[str, str], ...] = ()
    tool_sandbox_environment: tuple[tuple[str, str], ...] = ()
    runtime_environment: tuple[tuple[str, str], ...] = ()
    # Exact top-level namespaces for command-generated, non-source logs.
    # They are archived separately; they are not extra model write permissions.
    runtime_artifact_directories: tuple[str, ...] = ()
    ignored_workspace_patterns: tuple[str, ...] = (
        ".git/**", ".swarmrsi/**", "**/__pycache__/**", ".pytest_cache/**",
        ".mypy_cache/**", ".ruff_cache/**", "*.pyc",
    )
    max_observation_chars: int = 12_000
    # Tool-loop context is replayed explicitly because the OpenRouter chat
    # requests are stateless.  Without this record the model can only see the
    # latest observation and may repeat the same inspection until exhaustion.
    max_interaction_history_chars: int = 48_000
    # Implement first, inspect only to diagnose. The public data preview and the
    # parent's run output are supplied; probing before the first edit spent the
    # per-action call budget on 12% of NB actions. Off keeps older configs' hashes.
    write_first: bool = False
    # Characters of TASK_CONTEXT.md supplied with every request, kept from the start.
    # None keeps the historical last-20,000 slice, which on 17 of 45 NB runs dropped the
    # coder interface, dependency inventory and README task/metric/output sections.
    task_context_chars: int | None = None
    # After this many consecutive responses whose reasoning filled the output limit
    # before any tool call, send the next request with reasoning off, then restore it.
    # 18% of all NB coder calls ended that way; one action lost all 32 calls to it.
    reasoning_off_after_truncations: int | None = None
    # Largest new text one write, append or edit may carry. Writing a whole implementation
    # in one response is what overflowed the output limit (with reasoning on top); Claude
    # Code's NB messages stay under 2,600 tokens at p99 and MLEvolve edits by diff.
    max_mutation_chars: int | None = None
    # MLEvolve-style change: each response is one plan plus SEARCH/REPLACE blocks against the
    # executable (no tool calls, so reasoning never competes with a tool call); the controller
    # applies them, validates once, and returns the failure log for the next patch.
    diff_mode: bool = False
    # The declared component's outputs are submitted, not hidden behind a fallback that
    # restores the parent's outputs when the component loses on the program's own hold-out.
    # The controller already keeps the parent when a child scores worse; the fallback only
    # erased the evidence. NB 00732 (BRS): a new rollout operator with hold-out 1.05 vs the
    # parent's 0.114 fell back silently, and four actions produced identical outputs.
    no_silent_fallback: bool = False
    # A diff-mode action that replaces the method (a different model or operator for some
    # instances) is written as a complete new program, keeping only the parent's verified
    # data loading and output writing, instead of being grafted onto the old method's code.
    # 00732: the same idea scored +0.351 as a fresh draft (v4) and failed as a patch on a
    # POD-ridge program (BRS step 4, hold-out 1.05 vs 0.114).
    fresh_program_for_new_method: bool = False
    require_executable_parent_change: bool = True
    stop_on_supass_or_match_sota: bool = True
    schema_version: str = "swarmrsi-coder-config-v2"
    config_sha256: str = ""
    # Canonical, immutable JSON for an explicit candidate backend. Absent keeps
    # the original Coder identity and local execution behavior byte-compatible.
    candidate_runtime: str | None = None

    def canonical(self) -> dict[str, Any]:
        value = asdict(self)
        value.pop("config_sha256", None)
        value["action"] = self.action.to_dict()
        if self.candidate_runtime is None:
            value.pop("candidate_runtime", None)
        if self.max_diagnostic_runs_after_checkpoint is None:
            value.pop("max_diagnostic_runs_after_checkpoint", None)
        if self.max_transport_retries is None:
            value.pop("max_transport_retries", None)
        if self.finalization_reserve_seconds is None:
            value.pop("finalization_reserve_seconds", None)
        if not self.write_first:
            value.pop("write_first", None)
        if self.task_context_chars is None:
            value.pop("task_context_chars", None)
        if self.reasoning_off_after_truncations is None:
            value.pop("reasoning_off_after_truncations", None)
        if self.max_mutation_chars is None:
            value.pop("max_mutation_chars", None)
        if not self.diff_mode:
            value.pop("diff_mode", None)
        if not self.no_silent_fallback:
            value.pop("no_silent_fallback", None)
        if not self.fresh_program_for_new_method:
            value.pop("fresh_program_for_new_method", None)
        return value

    def __post_init__(self) -> None:
        if self.candidate_runtime is not None:
            from .candidate_execution import validate_binding
            import json
            if not isinstance(self.candidate_runtime, str):
                raise ValueError("candidate runtime must be an immutable JSON binding")
            validate_binding(json.loads(self.candidate_runtime))
        _validate_relative(self.executable_parent, field_name="executable parent")
        if not self.executable_parent.endswith(".py"):
            raise ValueError("executable parent must be a Python file")
        if len(self.expected_parent_sha256) != 64:
            raise ValueError("expected_parent_sha256 must bind one executable parent")
        if min(
            self.max_model_calls, self.max_tool_calls, self.max_input_tokens,
            self.max_output_tokens, self.max_repairs,
        ) < 0:
            raise ValueError("coder budgets cannot be negative")
        if self.max_model_calls == 0 or self.max_wall_seconds <= 0:
            raise ValueError("coder requires positive call and wall budgets")
        reserve = self.finalization_reserve_seconds
        if reserve is not None and (
            type(reserve) not in (int, float) or not math.isfinite(reserve)
            or not 0 < reserve < self.max_wall_seconds
        ):
            raise ValueError("finalization reserve must be finite, positive and below action wall")
        bound = self.max_diagnostic_runs_after_checkpoint
        if bound is not None and (type(bound) is not int or bound < 1):
            raise ValueError("max_diagnostic_runs_after_checkpoint must be a positive integer")
        retries = self.max_transport_retries
        if retries is not None and (type(retries) is not int or not 0 <= retries <= 12):
            raise ValueError("max_transport_retries must be an integer in [0, 12]")
        if self.max_observation_chars < 256:
            raise ValueError("observation budget is too small")
        if self.max_interaction_history_chars < 1_000:
            raise ValueError("interaction history budget is too small")
        if not self.validation:
            raise ValueError("at least one validation command is required")
        tiers = [item.tier for item in self.validation]
        order = {name: index for index, name in enumerate(
            ("syntax", "static", "import", "unit", "smoke", "full")
        )}
        if tiers != sorted(tiers, key=order.__getitem__):
            raise ValueError("validation tiers must be ordered from cheap to expensive")
        for prefix in self.allowed_command_prefixes:
            if not prefix or any(not item or "\x00" in item for item in prefix):
                raise ValueError("command prefixes must be non-empty argv tuples")
        if any(not item or "\x00" in item for item in self.tool_command_wrapper):
            raise ValueError("tool command wrapper contains an invalid argv item")
        for source, destination in self.tool_command_path_rewrites:
            if not source or not destination or "\x00" in source + destination:
                raise ValueError("invalid tool command path rewrite")
        if any(
            not key or "\x00" in key + value
            for key, value in self.tool_sandbox_environment
        ):
            raise ValueError("invalid tool sandbox environment")
        aliases: set[str] = set()
        for alias, source in self.read_only_mounts:
            _validate_relative(alias, field_name="read-only mount alias")
            if "/" in alias or alias in aliases:
                raise ValueError("read-only mount aliases must be unique top-level names")
            aliases.add(alias)
            source_path = PurePosixPath(source)
            if not source_path.is_absolute() or "\x00" in source:
                raise ValueError("read-only mount sources must be absolute paths")
        if any(not key or "\x00" in key + value for key, value in self.runtime_environment):
            raise ValueError("invalid runtime environment")
        for directory in self.runtime_artifact_directories:
            _validate_relative(directory, field_name="runtime artifact directory")
            if (
                "/" in directory or directory in {".", "output", "public_data"}
                or directory.startswith(".") or any(c in directory for c in "*?[]")
                or directory == self.executable_parent or directory in aliases
            ):
                raise ValueError("runtime artifact directories must be explicit non-reserved top-level names")
        if self.version == CoderVersion.C0 and self.max_model_calls != 1:
            raise ValueError("C0 must use exactly one model call")
        if self.action.selected_plan_json is not None:
            execution = selected_scientific_plan(self.action.selected_plan_json)["execution_action"]
            parent = next(d for d in execution["dependencies"] if d["kind"] == "code_parent")
            if parent["source_sha256"] != self.expected_parent_sha256:
                raise ValueError("selected plan and coder config bind different executable parents")
        if self.mutation_context is not None:
            context = self.mutation_context
            if self.action.research_contract_sha256 != context.research_contract_sha256:
                raise ValueError("coder action and mutation context bind different research contracts")
            if self.expected_parent_sha256 != context.parent_program_sha256:
                raise ValueError("coder executable and mutation context bind different parents")
        expected = coder_sha256(self.canonical())
        if self.config_sha256 and self.config_sha256 != expected:
            raise ValueError("config_sha256 mismatch")
        object.__setattr__(self, "config_sha256", expected)


@dataclass(slots=True)
class CoderOutcome:
    status: CoderExitStatus
    version: CoderVersion
    config_sha256: str
    action_id: str
    parent_sha256: str
    parent_workspace_sha256: str
    final_sha256: str | None
    final_workspace_sha256: str
    model_calls: int
    tool_calls: int
    input_tokens: int
    output_tokens: int
    wall_seconds: float
    validation: list[dict[str, Any]] = field(default_factory=list)
    events: list[dict[str, Any]] = field(default_factory=list)
    changed_paths: list[str] = field(default_factory=list)
    implementation_real: bool = False
    contract_violations: list[str] = field(default_factory=list)
    parent_preserved: bool = False
    repaired_after_failure: bool = False
    reason: str = ""
    # A failed implementation is retained as diagnostic evidence before the
    # executable workspace is rolled back to its verified parent.  It is never
    # eligible to become a program parent.
    diagnostic_workspace_sha256: str | None = None
    diagnostic_artifact_relpath: str | None = None
    trajectory_sha256: str = ""
    outcome_sha256: str = ""

    def seal(self) -> None:
        value = asdict(self)
        value.pop("outcome_sha256", None)
        self.outcome_sha256 = coder_sha256(value)
