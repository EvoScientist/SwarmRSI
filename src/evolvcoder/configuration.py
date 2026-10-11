"""Portable user configuration, independent of benchmark and server adapters.

Paths are relative to the configuration file, never to a developer checkout.
Secrets are references only. Validation performs no network or model calls.
Unknown fields are errors so a misspelled limit cannot silently disappear.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
import json
import math
import os
from pathlib import Path
import re
import tomllib
from typing import Any, TypeVar
from urllib.parse import urlsplit
from .search_plugins import SearchSpec


class ConfigurationError(ValueError):
    pass


def _text(value: Any, name: str) -> None:
    if not isinstance(value, str) or not value.strip() or "\x00" in value:
        raise ConfigurationError(f"{name} must be nonempty text")


def _positive(value: Any, name: str, *, integer: bool = False, optional: bool = True) -> None:
    if value is None and optional:
        return
    types = (int,) if integer else (int, float)
    if type(value) not in types or not math.isfinite(value) or value <= 0:
        raise ConfigurationError(f"{name} must be a positive {'integer' if integer else 'number'}"
                                 + (" or null" if optional else ""))


def _boolean(value: Any, name: str) -> None:
    if type(value) is not bool:
        raise ConfigurationError(f"{name} must be true or false")


def _path(value: Any, name: str) -> None:
    if not isinstance(value, Path) or not value.is_absolute():
        raise ConfigurationError(f"{name} must be an absolute resolved Path")


@dataclass(frozen=True)
class ModelConfig:
    model: str
    base_url: str = "https://openrouter.ai/api/v1"
    api_key_env: str | None = "OPENROUTER_API_KEY"
    api_key_file: Path | None = None
    temperature: float = 0.0
    max_output_tokens: int = 16000
    timeout_seconds: float = 180.0
    requests_per_minute: int | None = None
    tokens_per_minute: int | None = None
    concurrency: int = 1

    def __post_init__(self) -> None:
        _text(self.model, "model.model")
        _text(self.base_url, "model.base_url")
        parsed = urlsplit(self.base_url)
        if parsed.scheme not in {"http", "https"} or not parsed.hostname or parsed.username or parsed.password or parsed.query or parsed.fragment:
            raise ConfigurationError("model.base_url must be an HTTP(S) endpoint without credentials, query, or fragment")
        if (self.api_key_env is None) == (self.api_key_file is None):
            raise ConfigurationError("configure exactly one of api_key_env or api_key_file")
        if self.api_key_env is not None and (not isinstance(self.api_key_env, str) or re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", self.api_key_env) is None):
            raise ConfigurationError("api_key_env must be an environment variable name, not its value")
        if self.api_key_file is not None:
            _path(self.api_key_file, "model.api_key_file")
        if type(self.temperature) not in (int, float) or not math.isfinite(self.temperature) or not 0 <= self.temperature <= 2:
            raise ConfigurationError("model.temperature must be finite and in [0, 2]")
        for name in ("max_output_tokens", "concurrency"):
            _positive(getattr(self, name), "model." + name, integer=True, optional=False)
        for name in ("requests_per_minute", "tokens_per_minute"):
            _positive(getattr(self, name), "model." + name, integer=True)
        _positive(self.timeout_seconds, "model.timeout_seconds", optional=False)


@dataclass(frozen=True)
class Limits:
    """None means explicitly unlimited, not missing usage and not a zero cap."""

    model_calls: int | None = None
    input_tokens: int | None = None
    output_tokens: int | None = None
    tool_calls: int | None = None
    evaluations: int | None = None
    wall_seconds: float | None = None
    cost_usd: float | None = None

    def __post_init__(self) -> None:
        for name in ("model_calls", "input_tokens", "output_tokens", "tool_calls", "evaluations"):
            _positive(getattr(self, name), "limits." + name, integer=True)
        for name in ("wall_seconds", "cost_usd"):
            _positive(getattr(self, name), "limits." + name)


@dataclass(frozen=True)
class ContainerOptions:
    """Candidate-only settings; RuntimeConfig.python remains the host Python."""
    python: str
    uid: int
    gid: int
    pids: int = 256
    tmp_mb: int = 512
    shm_mb: int = 64
    file_mb: int = 4096

    def __post_init__(self):
        _text(self.python, "runtime.container.python")
        if not self.python.startswith("/") or ".." in Path(self.python).parts:
            raise ConfigurationError("container.python must be an absolute image path")
        for name in ("uid", "gid", "pids", "tmp_mb", "shm_mb", "file_mb"):
            _positive(getattr(self, name), "runtime.container." + name, integer=True, optional=False)


@dataclass(frozen=True)
class RuntimeConfig:
    kind: str = "local"
    python: str = "python3"
    host: str | None = None
    image: str | None = None
    account: str | None = None
    partition: str | None = None
    cpus: int = 1
    memory_mb: int | None = None
    gpus: int = 0
    work_root: Path | None = None
    container: ContainerOptions | None = None

    def __post_init__(self) -> None:
        if self.kind not in {"local", "container", "slurm"}:
            raise ConfigurationError("runtime.kind must be local, container, or slurm")
        _text(self.python, "runtime.python")
        if ("/" in self.python or self.python.startswith("~")) and not Path(self.python).is_absolute():
            raise ConfigurationError("runtime.python paths must be resolved by config_from_dict or load_config")
        _positive(self.cpus, "runtime.cpus", integer=True, optional=False)
        _positive(self.memory_mb, "runtime.memory_mb", integer=True)
        if type(self.gpus) is not int or self.gpus < 0:
            raise ConfigurationError("runtime.gpus must be a nonnegative integer")
        for name in ("host", "image", "account", "partition"):
            if getattr(self, name) is not None:
                _text(getattr(self, name), "runtime." + name)
        if self.kind == "container" and self.image is None:
            raise ConfigurationError("container runtime requires image")
        if self.kind == "slurm" and (self.account is None or self.partition is None):
            raise ConfigurationError("slurm runtime requires account and partition")
        if self.kind == "local" and any(getattr(self, n) is not None for n in ("host", "image", "account", "partition")):
            raise ConfigurationError("local runtime cannot silently ignore remote/container settings")
        if self.kind == "container" and any(getattr(self, n) is not None for n in ("account", "partition")):
            raise ConfigurationError("container runtime cannot silently ignore Slurm settings")
        if self.work_root is not None:
            _path(self.work_root, "runtime.work_root")
        if self.container is not None:
            if self.kind != "container" or not isinstance(self.container, ContainerOptions):
                raise ConfigurationError("runtime.container options require container runtime")


@dataclass(frozen=True)
class LearningConfig:
    wisdom_enabled: bool = False
    wisdom_snapshot: Path | None = None
    rsi_enabled: bool = False
    rsi_rounds: int = 3
    feedback_store: Path | None = None
    policy: Path | None = None

    def __post_init__(self) -> None:
        _boolean(self.wisdom_enabled, "learning.wisdom_enabled")
        _boolean(self.rsi_enabled, "learning.rsi_enabled")
        _positive(self.rsi_rounds, "learning.rsi_rounds", integer=True, optional=False)
        for name in ("wisdom_snapshot", "feedback_store", "policy"):
            if getattr(self, name) is not None:
                _path(getattr(self, name), "learning." + name)
        if self.wisdom_enabled and self.wisdom_snapshot is None:
            raise ConfigurationError("Wisdom requires a snapshot; explicitly disable it for a first run")
        if self.rsi_enabled and self.feedback_store is None:
            raise ConfigurationError("RSI requires a persistent feedback_store")


@dataclass(frozen=True)
class RunConfig:
    benchmark: str
    task: str
    model: ModelConfig
    method: str = "evolvcoder"
    runtime: RuntimeConfig = field(default_factory=RuntimeConfig)
    limits: Limits = field(default_factory=Limits)
    learning: LearningConfig = field(default_factory=LearningConfig)
    seed: int = 0
    output_root: Path = field(default_factory=lambda: Path("runs").resolve())
    benchmark_config: Path | None = None
    skip_retained_winners: bool = True
    winner_registry: Path | None = None
    schema_version: str = "evolvcoder-run-config-v1"
    search: SearchSpec | None = None

    def __post_init__(self) -> None:
        if self.search is not None:
            if not isinstance(self.search, SearchSpec):
                raise ConfigurationError("search must be a SearchSpec")
            if self.method != "evolvcoder":
                raise ConfigurationError("search configuration belongs to evolvcoder, not a baseline")
            # Outer learning over a search plugin is decided by that algorithm's
            # own declared parameters, which only the constructed algorithm can
            # report. preflight refuses an algorithm that declares none, by name.
        for name in ("benchmark", "task", "method"):
            _text(getattr(self, name), name)
        if self.schema_version != "evolvcoder-run-config-v1":
            raise ConfigurationError("unsupported configuration schema_version")
        for name, cls in (("model", ModelConfig), ("runtime", RuntimeConfig), ("limits", Limits), ("learning", LearningConfig)):
            if not isinstance(getattr(self, name), cls):
                raise ConfigurationError(f"{name} must be a {cls.__name__}")
        if type(self.seed) is not int or self.seed < 0:
            raise ConfigurationError("seed must be a nonnegative integer")
        _path(self.output_root, "output_root")
        for name in ("benchmark_config", "winner_registry"):
            if getattr(self, name) is not None:
                _path(getattr(self, name), name)
        _boolean(self.skip_retained_winners, "skip_retained_winners")

    def to_dict(self) -> dict:
        def convert(value):
            if isinstance(value, Path):
                return str(value)
            if isinstance(value, dict):
                return {k: convert(v) for k, v in value.items()}
            return value
        result = convert(asdict(self))
        if self.runtime.container is None:
            result["runtime"].pop("container")
        if self.search is None:
            result.pop("search")
        return result


T = TypeVar("T")


def _construct(cls: type[T], value: Any, where: str, base: Path, paths: tuple[str, ...] = ()) -> T:
    if not isinstance(value, dict):
        raise ConfigurationError(f"{where} must be an object")
    if any(not isinstance(key, str) for key in value):
        raise ConfigurationError(f"{where} field names must be strings")
    unknown = set(value) - {f.name for f in fields(cls)}
    if unknown:
        raise ConfigurationError(f"unknown {where} fields: {', '.join(sorted(unknown))}")
    data = dict(value)
    for name in paths:
        if data.get(name) is not None:
            _text(data[name], f"{where}.{name}")
            path = Path(data[name]).expanduser()
            data[name] = (path if path.is_absolute() else base / path).resolve()
    try:
        return cls(**data)
    except TypeError as exc:
        raise ConfigurationError(f"invalid {where}: {exc}") from exc


def config_from_dict(data: dict, *, base: Path) -> RunConfig:
    if not isinstance(data, dict):
        raise ConfigurationError("configuration must be an object")
    base = base.resolve()
    payload = dict(data)
    payload["model"] = _construct(ModelConfig, payload.get("model"), "model", base, ("api_key_file",))
    runtime = payload.get("runtime", {})
    if not isinstance(runtime, dict):
        raise ConfigurationError("runtime must be an object")
    runtime = dict(runtime)
    if runtime.get("container") is not None:
        runtime["container"] = _construct(ContainerOptions, runtime["container"], "runtime.container", base)
    remote = runtime.get("kind", "local") != "local"
    if remote and runtime.get("work_root") is not None:
        _text(runtime["work_root"], "runtime.work_root")
        if not Path(runtime["work_root"]).is_absolute():
            raise ConfigurationError("remote/container work_root must be an explicit absolute target path")
        # This path belongs to the target filesystem, not the configuration
        # reader's host. Never dereference it through local symlinks.
        runtime["work_root"] = Path(os.path.normpath(runtime["work_root"]))
    executable = runtime.get("python", "python3")
    _text(executable, "runtime.python")
    if "/" in executable or executable.startswith("~"):
        if remote and executable.startswith("~"):
            raise ConfigurationError("remote runtime.python must use an absolute path or explicit work_root, not ~")
        path = Path(executable).expanduser() if not remote else Path(executable)
        if not path.is_absolute():
            executable_base = runtime.get("work_root") if remote else base
            if executable_base is None:
                raise ConfigurationError("relative remote runtime.python requires an explicit absolute work_root")
            path = executable_base / path
        # absolute()/normpath preserve venv launcher symlinks; resolve() can
        # accidentally select its base Python and discard the intended venv.
        runtime["python"] = os.path.normpath(str(path))
    payload["runtime"] = _construct(RuntimeConfig, runtime, "runtime", base, () if remote else ("work_root",))
    payload["limits"] = _construct(Limits, payload.get("limits", {}), "limits", base)
    payload["learning"] = _construct(LearningConfig, payload.get("learning", {}), "learning", base,
                                     ("wisdom_snapshot", "feedback_store", "policy"))
    payload.setdefault("output_root", "runs")
    if payload.get("search") is not None:
        payload["search"] = SearchSpec.from_dict(payload["search"])
    return _construct(RunConfig, payload, "run", base, ("output_root", "benchmark_config", "winner_registry"))


def load_config(path: Path) -> RunConfig:
    path = path.resolve()
    try:
        raw = path.read_bytes()
        if path.suffix.lower() == ".toml":
            data = tomllib.loads(raw.decode("utf-8"))
        elif path.suffix.lower() == ".json":
            def pairs(items):
                result = {}
                for key, value in items:
                    if key in result:
                        raise ConfigurationError(f"duplicate configuration field: {key}")
                    result[key] = value
                return result
            data = json.loads(raw, object_pairs_hook=pairs,
                              parse_constant=lambda value: (_ for _ in ()).throw(ConfigurationError(f"nonfinite JSON value: {value}")))
        else:
            raise ConfigurationError("configuration must be .json or .toml")
    except (OSError, UnicodeError, json.JSONDecodeError, tomllib.TOMLDecodeError) as exc:
        raise ConfigurationError(f"cannot read configuration {path.name}: {exc}") from exc
    return config_from_dict(data, base=path.parent)
