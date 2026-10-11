"""Public plugin contracts shared by all methods and benchmarks.

Concrete adapters are registered explicitly. A missing adapter is never replaced
by a different algorithm or a synthetic result.
"""
from __future__ import annotations

from dataclasses import dataclass, field
import math
from pathlib import Path
from typing import Generic, Protocol, TypeVar

from .configuration import RunConfig


@dataclass(frozen=True)
class TaskSpec:
    benchmark: str
    task_id: str
    public_description: str
    starter: Path
    public_data: Path
    evaluator_id: str
    metric_name: str
    metric_direction: str
    # Opaque references for controller use, never serialized into model context.
    evaluator_config: Path

    def __post_init__(self) -> None:
        for name in ("benchmark", "task_id", "public_description", "evaluator_id", "metric_name"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ValueError(f"TaskSpec.{name} is required")
        if self.metric_direction not in {"maximize", "minimize"}:
            raise ValueError("metric_direction must be maximize or minimize")
        for name in ("starter", "public_data", "evaluator_config"):
            if not isinstance(getattr(self, name), Path) or not getattr(self, name).is_absolute():
                raise ValueError(f"TaskSpec.{name} must be an absolute Path")

    def public_context(self) -> dict:
        return {"benchmark": self.benchmark, "task_id": self.task_id,
                "description": self.public_description, "metric_name": self.metric_name,
                "metric_direction": self.metric_direction}


@dataclass(frozen=True)
class RunContext:
    run_id: str
    output_dir: Path
    config: RunConfig

    def __post_init__(self) -> None:
        if not isinstance(self.run_id, str) or not self.run_id.strip():
            raise ValueError("RunContext.run_id is required")
        if not isinstance(self.output_dir, Path) or not self.output_dir.is_absolute():
            raise ValueError("RunContext.output_dir must be an absolute Path")
        if not isinstance(self.config, RunConfig):
            raise TypeError("RunContext.config must be a RunConfig")


@dataclass(frozen=True)
class RunResult:
    run_id: str
    task_id: str
    method: str
    status: str
    artifact: Path | None = None
    score: float | None = None
    evaluator_id: str | None = None
    evidence: Path | None = None
    reason: str | None = None
    usage: dict = field(default_factory=dict)
    details: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        for name in ("run_id", "task_id", "method"):
            if not isinstance(getattr(self, name), str) or not getattr(self, name).strip():
                raise ValueError(f"RunResult.{name} is required")
        for name in ("artifact", "evidence"):
            value = getattr(self, name)
            if value is not None and (not isinstance(value, Path) or not value.is_absolute()):
                raise ValueError(f"RunResult.{name} must be an absolute Path")
        if self.status not in {"completed", "failed", "evaluation_invalid", "budget_exhausted", "retained", "pending"}:
            raise ValueError("unknown RunResult status")
        if self.status == "completed" and (self.artifact is None or self.evidence is None):
            raise ValueError("completed delivery requires artifact and evidence references")
        if self.score is not None:
            if type(self.score) not in (int, float) or not math.isfinite(self.score):
                raise ValueError("score must be finite or None")
            if self.artifact is None or self.evidence is None or not self.evaluator_id:
                raise ValueError("a score requires artifact, evidence and evaluator identity")
            if self.status in {"failed", "evaluation_invalid", "pending"}:
                raise ValueError("failed, invalid or pending results cannot claim a final score")
        if self.status == "retained" and (self.score is None or not self.reason):
            raise ValueError("retained results require source-bound score and reuse reason")


class MethodAdapter(Protocol):
    def run(self, task: TaskSpec, context: RunContext) -> RunResult: ...


class BenchmarkAdapter(Protocol):
    def task(self, task_id: str, config: RunConfig) -> TaskSpec: ...


T = TypeVar("T")


class AdapterRegistry(Generic[T]):
    def __init__(self, kind: str):
        if kind not in {"method", "benchmark"}:
            raise ValueError("registry kind must be method or benchmark")
        self.kind = kind
        self._adapters: dict[str, T] = {}

    def register(self, name: str, adapter: T) -> None:
        if not isinstance(name, str) or not name.strip() or name != name.strip():
            raise ValueError("adapter name must be nonempty without surrounding whitespace")
        if name in self._adapters:
            raise ValueError(f"{self.kind} adapter already registered: {name}")
        required = "run" if self.kind == "method" else "task"
        if not callable(getattr(adapter, required, None)):
            raise TypeError(f"{self.kind} adapter must implement {required}")
        self._adapters[name] = adapter

    def get(self, name: str) -> T:
        try:
            return self._adapters[name]
        except KeyError as exc:
            raise ValueError(f"unknown {self.kind} adapter {name!r}; registered: {', '.join(self.names()) or '(none)'}") from exc

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._adapters))


class Application:
    """SDK dispatch using the exact configured method and benchmark adapters."""

    def __init__(self, *, register_defaults: bool = True) -> None:
        self.methods: AdapterRegistry[MethodAdapter] = AdapterRegistry("method")
        self.benchmarks: AdapterRegistry[BenchmarkAdapter] = AdapterRegistry("benchmark")
        if register_defaults:
            from .unified_run import EvolvcoderAdapter, NatureBenchAdapter
            self.methods.register("evolvcoder", EvolvcoderAdapter())
            self.benchmarks.register("naturebench", NatureBenchAdapter())
            from .mlebench_run import MLEBenchAdapter
            self.benchmarks.register("mlebench", MLEBenchAdapter())
            from .openproblems_run import OpenProblemsAdapter
            self.benchmarks.register("openproblems", OpenProblemsAdapter())
            from .terminalbench_run import TerminalBenchAdapter
            self.benchmarks.register("terminalbench", TerminalBenchAdapter())

    def _operation(self, name: str, context: RunContext) -> dict:
        config = context.config
        method = self.methods.get(config.method)
        benchmark = self.benchmarks.get(config.benchmark)
        operation = getattr(method, name, None)
        if not callable(operation):
            raise ValueError(f"method {config.method!r} does not implement {name}")
        task = benchmark.task(config.task, config)
        if (task.benchmark, task.task_id) != (config.benchmark, config.task):
            raise ValueError("benchmark adapter returned a different task")
        return operation(task, context)

    def preflight(self, context: RunContext) -> dict:
        return self._operation("preflight", context)

    def status(self, context: RunContext) -> dict:
        return self._operation("status", context)

    def run(self, context: RunContext) -> RunResult:
        config = context.config
        method = self.methods.get(config.method)
        benchmark = self.benchmarks.get(config.benchmark)
        task = benchmark.task(config.task, config)
        if task.benchmark != config.benchmark or task.task_id != config.task:
            raise ValueError("benchmark adapter returned a different task")
        result = method.run(task, context)
        if not isinstance(result, RunResult) or (result.run_id, result.task_id, result.method) != (context.run_id, config.task, config.method):
            raise ValueError("method adapter returned a mismatched result")
        if result.score is not None and result.evaluator_id != task.evaluator_id:
            raise ValueError("result score belongs to a different evaluator")
        return result
