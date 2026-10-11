"""Declared learnable search parameters: the only surface an outer learner may change.

A search algorithm decides which of its own construction parameters are safe to
learn and states their bounds in code. Nothing here can reach the evaluator, the
data split, the budget or the scientific verdict, and an algorithm that declares
nothing is never silently replaced by another algorithm's policy.
"""
from __future__ import annotations

from dataclasses import dataclass, replace
import math
from typing import Any, Mapping

from .contracts import canonical_sha256
from .search_plugins import SearchSpec

SCHEMA = "evolvcoder-search-parameter-policy-v1"
KINDS = ("number", "integer")


@dataclass(frozen=True)
class ParameterSpec:
    """One learnable construction parameter, located inside SearchSpec.parameters."""

    name: str
    path: tuple[str, ...]
    kind: str = "number"
    minimum: float = 0.0
    maximum: float = 1.0

    def __post_init__(self) -> None:
        if not isinstance(self.name, str) or not self.name.strip() or self.name != self.name.strip():
            raise ValueError("learnable parameter needs a name")
        if (not isinstance(self.path, tuple) or not self.path
                or any(not isinstance(part, str) or not part.strip() for part in self.path)):
            raise ValueError("learnable parameter needs a nonempty configuration path")
        if self.kind not in KINDS:
            raise ValueError(f"learnable parameter kind must be one of {KINDS}")
        for bound in (self.minimum, self.maximum):
            if type(bound) not in (int, float) or isinstance(bound, bool) or not math.isfinite(bound):
                raise ValueError("learnable parameter bounds must be finite numbers")
        if self.minimum > self.maximum:
            raise ValueError("learnable parameter minimum exceeds its maximum")
        if self.kind == "integer" and (self.minimum != int(self.minimum) or self.maximum != int(self.maximum)):
            raise ValueError("integer parameter bounds must be whole numbers")

    def coerce(self, value: Any) -> float | int:
        if isinstance(value, bool) or type(value) not in (int, float) or not math.isfinite(value):
            raise ValueError(f"{self.name} must be a finite number")
        if self.kind == "integer":
            if type(value) is not int:
                raise ValueError(f"{self.name} must be an integer")
        else:
            value = float(value)
        if not self.minimum <= value <= self.maximum:
            raise ValueError(f"{self.name} must lie in [{self.minimum}, {self.maximum}]")
        return value

    def to_dict(self) -> dict:
        return {"name": self.name, "path": list(self.path), "kind": self.kind,
                "minimum": self.minimum, "maximum": self.maximum}

    @classmethod
    def from_dict(cls, value: Any) -> "ParameterSpec":
        if not isinstance(value, dict) or set(value) != {"name", "path", "kind", "minimum", "maximum"}:
            raise ValueError("unknown or missing learnable parameter fields")
        return cls(name=value["name"], path=tuple(value["path"]), kind=value["kind"],
                   minimum=value["minimum"], maximum=value["maximum"])


def declaration_of(algorithm: Any) -> tuple[ParameterSpec, ...]:
    """Read an algorithm's own declaration; absent means it cannot be learned."""
    declare = getattr(algorithm, "learnable_parameters", None)
    if declare is None:
        return ()
    declared = tuple(declare() if callable(declare) else declare)
    if any(not isinstance(item, ParameterSpec) for item in declared):
        raise TypeError("learnable_parameters must return ParameterSpec entries")
    names = [item.name for item in declared]
    paths = [item.path for item in declared]
    if len(set(names)) != len(names) or len(set(paths)) != len(paths):
        raise ValueError("learnable parameter names and paths must be unique")
    return tuple(sorted(declared, key=lambda item: item.name))


def _at(parameters: Mapping[str, Any], path: tuple[str, ...]) -> Any:
    cursor: Any = parameters
    for part in path:
        if not isinstance(cursor, Mapping) or part not in cursor:
            raise ValueError(f"declared learnable parameter {'.'.join(path)} is absent from the configuration")
        cursor = cursor[part]
    return cursor


def _with(parameters: Mapping[str, Any], path: tuple[str, ...], value: Any) -> dict:
    head, rest = path[0], path[1:]
    if not isinstance(parameters, Mapping) or head not in parameters:
        raise ValueError(f"declared learnable parameter {'.'.join(path)} is absent from the configuration")
    updated = dict(parameters)
    updated[head] = value if not rest else _with(parameters[head], rest, value)
    return updated


@dataclass(frozen=True)
class SearchParameterPolicy:
    """Learned values for the parameters one algorithm declares learnable.

    The record carries its declaration so read-only journal analysis can verify
    it without importing a plugin; ``verify_declaration`` re-checks it against
    the live algorithm wherever the algorithm is actually constructed, so the
    bounds always come from code and never from a stored record.
    """

    search: SearchSpec
    declaration: tuple[ParameterSpec, ...]
    values: Mapping[str, float]

    def __post_init__(self) -> None:
        if not isinstance(self.search, SearchSpec):
            raise ValueError("search parameter policy requires a SearchSpec")
        declaration = tuple(self.declaration)
        if not declaration:
            raise ValueError("this search algorithm declares no learnable parameters")
        if any(not isinstance(item, ParameterSpec) for item in declaration):
            raise TypeError("declaration must contain ParameterSpec entries")
        if [item.name for item in declaration] != sorted(item.name for item in declaration):
            raise ValueError("declaration must be ordered by parameter name")
        if not isinstance(self.values, Mapping) or set(self.values) != {item.name for item in declaration}:
            raise ValueError("policy must set exactly the declared learnable parameters")
        object.__setattr__(self, "declaration", declaration)
        object.__setattr__(self, "values", {item.name: item.coerce(self.values[item.name]) for item in declaration})
        # A learned value must be a value the algorithm can actually receive.
        self.effective_search()

    @classmethod
    def from_algorithm(cls, search: SearchSpec, algorithm: Any, values: Mapping[str, float] | None = None):
        declaration = declaration_of(algorithm)
        if not declaration:
            raise ValueError(f"search algorithm {search.name!r} declares no learnable parameters; "
                             "outer learning is not supported for it and no other policy is substituted")
        current = {item.name: _at(search.parameters, item.path) for item in declaration}
        return cls(search, declaration, dict(values) if values is not None else current)

    def effective_search(self) -> SearchSpec:
        parameters = self.search.parameters
        for item in self.declaration:
            parameters = _with(parameters, item.path, self.values[item.name])
        return replace(self.search, parameters=parameters)

    def verify_declaration(self, algorithm: Any) -> None:
        if declaration_of(algorithm) != self.declaration:
            raise ValueError("search algorithm's declared learnable parameters changed")

    @property
    def sha256(self) -> str:
        return canonical_sha256({"search": self.search.to_dict(),
                                 "declaration": [item.to_dict() for item in self.declaration],
                                 "parameters": dict(self.values)})

    def to_dict(self) -> dict:
        return {"schema_version": SCHEMA, "search": self.search.to_dict(),
                "declaration": [item.to_dict() for item in self.declaration],
                "parameters": dict(self.values), "policy_sha256": self.sha256}

    @classmethod
    def from_dict(cls, record: Any) -> "SearchParameterPolicy":
        if (not isinstance(record, dict)
                or set(record) != {"schema_version", "search", "declaration", "parameters", "policy_sha256"}
                or record["schema_version"] != SCHEMA):
            raise ValueError("unknown search parameter policy fields/schema")
        if not isinstance(record["declaration"], list) or not isinstance(record["parameters"], dict):
            raise ValueError("search parameter policy declaration/parameters must be a list and an object")
        policy = cls(SearchSpec.from_dict(record["search"]),
                     tuple(ParameterSpec.from_dict(item) for item in record["declaration"]),
                     record["parameters"])
        if policy.to_dict() != record:
            raise ValueError("search parameter policy descriptor or hash mismatch")
        return policy

    def diff(self, incumbent: "SearchParameterPolicy") -> dict:
        if incumbent.search.to_dict() != self.search.to_dict() or incumbent.declaration != self.declaration:
            raise ValueError("cannot diff policies of different algorithms or declarations")
        return {name: {"before": incumbent.values[name], "after": value}
                for name, value in self.values.items() if value != incumbent.values[name]}
