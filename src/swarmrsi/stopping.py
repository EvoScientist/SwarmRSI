from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping
import json
import math


COMPARISONS = frozenset({"gt", "gte", "lt", "lte"})


@dataclass(frozen=True, slots=True)
class ScoreStopCondition:
    name: str
    metric: str
    comparison: str
    threshold: float
    task_id: str | None = None
    requires_score_bearing: bool = True
    requires_artifact_valid: bool = True

    def __post_init__(self) -> None:
        if not self.name or not self.metric:
            raise ValueError("stop-condition name and metric must be non-empty")
        if self.comparison not in COMPARISONS:
            raise ValueError(f"comparison must be one of {sorted(COMPARISONS)}")
        if isinstance(self.threshold, bool) or not math.isfinite(self.threshold):
            raise ValueError("stop-condition threshold must be finite")

    def matches(
        self,
        *,
        task_id: str,
        metrics: Mapping[str, float | int | None],
        score_bearing: bool,
        artifact_valid: bool,
    ) -> bool:
        if self.task_id is not None and task_id != self.task_id:
            return False
        if self.requires_score_bearing and not score_bearing:
            return False
        if self.requires_artifact_valid and not artifact_valid:
            return False
        raw = metrics.get(self.metric)
        if isinstance(raw, bool) or not isinstance(raw, (int, float)):
            return False
        value = float(raw)
        if not math.isfinite(value):
            return False
        return {
            "gt": value > self.threshold,
            "gte": value >= self.threshold,
            "lt": value < self.threshold,
            "lte": value <= self.threshold,
        }[self.comparison]


@dataclass(frozen=True, slots=True)
class StopDecision:
    stop: bool
    stage: str
    matched_condition: str | None


@dataclass(frozen=True, slots=True)
class ConfigurableStopPolicy:
    stages: Mapping[str, tuple[ScoreStopCondition, ...]]
    active_stage: str

    def __post_init__(self) -> None:
        if not self.stages:
            raise ValueError("stop policy must define at least one stage")
        if self.active_stage not in self.stages:
            raise ValueError("active_stage is absent from stop-policy stages")
        if any(not rules for rules in self.stages.values()):
            raise ValueError("every stop-policy stage must contain a rule")

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> ConfigurableStopPolicy:
        if raw.get("schema_version") != "swarmrsi-stop-policy-v1":
            raise ValueError("unsupported stop-policy schema")
        raw_stages = raw.get("stages")
        if not isinstance(raw_stages, dict):
            raise ValueError("stop-policy stages must be an object")
        stages: dict[str, tuple[ScoreStopCondition, ...]] = {}
        for stage, stage_payload in raw_stages.items():
            if not isinstance(stage, str) or not stage:
                raise ValueError("stop-policy stage names must be non-empty")
            if not isinstance(stage_payload, dict):
                raise ValueError(f"stage {stage!r} must be an object")
            rules = stage_payload.get("rules")
            if not isinstance(rules, list):
                raise ValueError(f"stage {stage!r} rules must be a list")
            stages[stage] = tuple(
                ScoreStopCondition(**rule) for rule in rules
                if isinstance(rule, dict)
            )
            if len(stages[stage]) != len(rules):
                raise ValueError(f"stage {stage!r} contains a malformed rule")
        active = raw.get("active_stage")
        if not isinstance(active, str):
            raise ValueError("active_stage must be a string")
        return cls(stages=stages, active_stage=active)

    @classmethod
    def from_json(cls, path: Path) -> ConfigurableStopPolicy:
        payload = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("stop-policy JSON must contain an object")
        if "early_stop" in payload:
            payload = payload["early_stop"]
        if not isinstance(payload, dict):
            raise ValueError("early_stop must contain an object")
        return cls.from_dict(payload)

    def evaluate(
        self,
        *,
        task_id: str,
        metrics: Mapping[str, float | int | None],
        score_bearing: bool,
        artifact_valid: bool,
        stage: str | None = None,
    ) -> StopDecision:
        selected_stage = stage or self.active_stage
        if selected_stage not in self.stages:
            raise ValueError(f"unknown stop-policy stage: {selected_stage}")
        for condition in self.stages[selected_stage]:
            if condition.matches(
                task_id=task_id,
                metrics=metrics,
                score_bearing=score_bearing,
                artifact_valid=artifact_valid,
            ):
                return StopDecision(True, selected_stage, condition.name)
        return StopDecision(False, selected_stage, None)
