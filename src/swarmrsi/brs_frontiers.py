"""Hypothesis frontier, program frontier and the sparse gap-driven matcher.

Coverage between a hypothesis requirement and a program capability is token
overlap over controller-recorded text. It orders pair proposals; it is neither
a scientific judgement nor a proof of compatibility, and ``unknown`` coverage
remains eligible for budget-limited exploration.
"""
from __future__ import annotations
import math

from dataclasses import replace
import re
from typing import Any, Iterable

from .brs_models import (HypothesisRecord, PairProposal, ProgramRepresentative, hypothesis_from_dict,
                         logical_pair_key, record_to_dict, representative_from_dict)
from .contracts import canonical_sha256

_TOKEN = re.compile(r"[a-z][a-z0-9_]{2,}")
STOPWORDS = frozenset((
    "the and for with from that this into over under using use used via then than are was were will would "
    "should could may might can not but have has had per each any all its their there here when where which "
    "while whose also such more most less onto within without between among both either neither same other "
    "another only very just every some none program programs code function functions file files run main "
    "output outputs data input inputs feature features model models method methods approach approaches step "
    "steps result results score scores metric metrics task tasks hypothesis hypotheses implementation "
    "implement implemented implementing change changes changed keep keeps keeping preserve preserved existing "
    "current new baseline root starter unchanged unspecified default defaults true false none null value "
    "values based instead rather still because after before during"
).split())
FAMILY_CORE = ("input_transform", "primary_operator", "output_construction", "objective_or_test")


def tokens(*texts: Any) -> frozenset[str]:
    found: set[str] = set()
    for text in texts:
        if text is None:
            continue
        if isinstance(text, dict):
            text = " ".join(str(v) for v in text.values())
        elif isinstance(text, (list, tuple, set, frozenset)):
            text = " ".join(str(t) for t in text)
        for match in _TOKEN.findall(str(text).lower()):
            if match not in STOPWORDS:
                found.add(match)
    return frozenset(found)


def family_core(family: dict[str, Any] | None) -> dict[str, str]:
    family = family or {}
    return {key: str(family.get(key, "unspecified")) for key in FAMILY_CORE}


def family_sha256(family: dict[str, Any] | None) -> str:
    core = family_core(family)
    return canonical_sha256([core[key] for key in FAMILY_CORE])


def capability_tags(family: dict[str, Any] | None, statement: str = "", limit: int = 48) -> tuple[str, ...]:
    core = family_core(family)
    return tuple(sorted(tokens(core, family.get("capacity_controls") if family else None, statement[:400]))[:limit])


class HypothesisFrontier:
    """Bounded, family-balanced active set over a complete hypothesis archive."""

    def __init__(self, *, max_active: int) -> None:
        if type(max_active) is not int or max_active < 1:
            raise ValueError("max_active_hypotheses must be a positive integer")
        self.max_active = max_active
        self.records: dict[str, HypothesisRecord] = {}

    def get(self, hypothesis_id: str) -> HypothesisRecord | None:
        return self.records.get(hypothesis_id)

    def put(self, record: HypothesisRecord) -> None:
        self.records[record.id] = record

    def exact_duplicate(self, family_sha: str, statement: str) -> str | None:
        normalized = " ".join(statement.lower().split())
        for record in self.records.values():
            if record.family_sha256 == family_sha and " ".join(record.statement.lower().split()) == normalized:
                return record.id
        return None

    def waiting(self, record: HypothesisRecord, step: int) -> int:
        anchor = record.last_attempt_step if record.last_attempt_step is not None else record.created_step
        return max(0, step - anchor)

    def rebalance(self, step: int) -> list[dict[str, Any]]:
        live = [r for r in self.records.values() if r.status != "archived"]
        by_family: dict[str, list[HypothesisRecord]] = {}
        for record in live:
            by_family.setdefault(record.family_sha256, []).append(record)
        for group in by_family.values():
            group.sort(key=lambda r: (r.effective_test_count, -self.waiting(r, step), r.id))
        families = sorted(by_family, key=lambda f: (min(r.effective_test_count for r in by_family[f]),
                                                    -max(self.waiting(r, step) for r in by_family[f]), f))
        ordered: list[HypothesisRecord] = []
        layer = 0
        while len(ordered) < len(live):
            for family in families:
                if layer < len(by_family[family]):
                    ordered.append(by_family[family][layer])
            layer += 1
        transitions = []
        for rank, record in enumerate(ordered):
            status = "active" if rank < self.max_active else "deferred"
            if record.status != status:
                transitions.append({"hypothesis_id": record.id, "from": record.status, "to": status,
                                    "reason": "family_balanced_capacity", "rank": rank, "step": step})
                self.records[record.id] = replace(record, status=status, status_reason="family_balanced_capacity")
        return transitions

    def active(self) -> tuple[HypothesisRecord, ...]:
        return tuple(sorted((r for r in self.records.values() if r.status == "active"), key=lambda r: r.id))

    def to_dict(self) -> dict[str, Any]:
        return {"max_active": self.max_active, "records": {k: record_to_dict(v) for k, v in sorted(self.records.items())}}

    def restore(self, payload: dict[str, Any]) -> None:
        if payload.get("max_active") != self.max_active:
            raise ValueError("hypothesis frontier capacity changed")
        self.records = {k: hypothesis_from_dict(v) for k, v in payload["records"].items()}


class ProgramFrontier:
    """Capability representative index over valid scored programs; never deletes history."""

    def __init__(self, *, max_representatives: int) -> None:
        if type(max_representatives) is not int or max_representatives < 1:
            raise ValueError("max_program_representatives must be a positive integer")
        self.max_representatives = max_representatives
        self.representatives: tuple[ProgramRepresentative, ...] = ()
        # instance -> (node_id, improvement): the best executed program per
        # public instance, so a continuation can adopt a component from a
        # program that beats its parent on one instance without rewriting the
        # instances the parent already wins.
        self.instance_winners: dict[str, tuple[str, float]] = {}

    def refresh(self, valid_nodes: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
        before = {r.node_id: r.reason for r in self.representatives}
        if not valid_nodes:
            self.representatives = ()
            self.instance_winners = {}
            return [{"node_id": n, "from": r, "to": None} for n, r in sorted(before.items())]
        ordered = sorted(valid_nodes.items(), key=lambda kv: (kv[1]["native_metric"], kv[0]), reverse=True)
        winners: dict[str, tuple[str, float]] = {}
        for node_id, info in ordered:
            for instance, value in sorted((info.get("per_instance") or {}).items()):
                if isinstance(value, (int, float)) and math.isfinite(value):
                    current = winners.get(str(instance))
                    if current is None or value > current[1]:
                        winners[str(instance)] = (node_id, float(value))
        self.instance_winners = winners
        chosen: dict[str, str] = {ordered[0][0]: "global_incumbent"}
        families = {ordered[0][1].get("family_sha256")}
        for node_id, info in ordered:
            if len(chosen) >= self.max_representatives:
                break
            family = info.get("family_sha256")
            if node_id not in chosen and family not in families:
                chosen[node_id] = "family_best"
                families.add(family)
        for node_id, _ in ordered:
            if len(chosen) >= self.max_representatives:
                break
            chosen.setdefault(node_id, "metric_rank")
        self.representatives = tuple(
            ProgramRepresentative(node_id=node_id, executable_sha256=info["executable_sha256"],
                                  native_metric=float(info["native_metric"]), family_sha256=info.get("family_sha256"),
                                  hypothesis_id=info["hypothesis_id"], capabilities=tuple(info.get("capabilities", ())),
                                  evidence_level="executed_scored", reason=chosen[node_id])
            for node_id, info in ordered if node_id in chosen)
        after = {r.node_id: r.reason for r in self.representatives}
        return [{"node_id": n, "from": before.get(n), "to": after.get(n)}
                for n in sorted(set(before) | set(after)) if before.get(n) != after.get(n)]

    def incumbent(self) -> ProgramRepresentative | None:
        return next((r for r in self.representatives if r.reason == "global_incumbent"), None)

    def get(self, node_id: str) -> ProgramRepresentative | None:
        return next((r for r in self.representatives if r.node_id == node_id), None)

    def recombination_sources(self, parent_id: str, parent_per_instance: dict[str, Any] | None,
                              *, limit: int = 2) -> list[dict[str, Any]]:
        """Programs that beat ``parent_id`` on an instance, largest margin first."""
        parent_scores = parent_per_instance or {}
        rows: dict[str, dict[str, Any]] = {}
        for instance, (node_id, value) in sorted(self.instance_winners.items()):
            own = parent_scores.get(instance)
            if node_id == parent_id or not isinstance(own, (int, float)) or not math.isfinite(own) or value <= own:
                continue
            row = rows.setdefault(node_id, {"node_id": node_id, "instances": [], "margin": 0.0})
            row["instances"].append({"instance": instance, "winner": value, "parent": float(own)})
            row["margin"] = max(row["margin"], value - float(own))
        return sorted(rows.values(), key=lambda r: (-r["margin"], r["node_id"]))[:limit]

    def to_dict(self) -> dict[str, Any]:
        return {"max_representatives": self.max_representatives,
                "representatives": [record_to_dict(r) for r in self.representatives],
                "instance_winners": {k: [v[0], v[1]] for k, v in sorted(self.instance_winners.items())}}

    def restore(self, payload: dict[str, Any]) -> None:
        if payload.get("max_representatives") != self.max_representatives:
            raise ValueError("program frontier capacity changed")
        self.representatives = tuple(representative_from_dict(r) for r in payload["representatives"])
        self.instance_winners = {str(k): (str(v[0]), float(v[1])) for k, v in (payload.get("instance_winners") or {}).items()}


MATCHER_NAME = "sparse_gap_v1"


def propose_pairs(*, hypotheses: Iterable[HypothesisRecord], representatives: tuple[ProgramRepresentative, ...],
                  excluded_keys: set[str] | frozenset[str], max_per_hypothesis: int, max_sources: int,
                  measured_costs: dict[str, float], step: int, waiting: dict[str, int]) -> list[PairProposal]:
    """Sparse H-P pair proposals ordered by the explicit first-version heuristic.

    Order: closes a recorded blocker > requirement coverage > a base other than
    the one that failed > local adaptation > measured producing cost > fewer
    sources > program score. Returns at most ``max_per_hypothesis`` proposals
    per hypothesis; ``unknown`` coverage stays eligible but ranks below any
    overlap. A program without a measured cost is ranked at the mean measured
    cost (neutral), never as if it were free.
    """
    proposals: list[tuple[tuple[Any, ...], PairProposal]] = []
    measured_values = [float(v) for v in measured_costs.values()]
    neutral_cost = sum(measured_values) / len(measured_values) if measured_values else 0.0
    for record in hypotheses:
        requirements = record.open_requirements()
        if not requirements or not representatives:
            continue
        need = tokens(*(r.description + " " + r.next_step for r in requirements))
        failed = {r.failed_program_id for r in requirements if r.failed_program_id}
        # A requirement that names its base (a rebase gap) is closed only on
        # that base; token overlap must not redirect it to another program.
        targets = {r.target_program_id for r in requirements if getattr(r, "target_program_id", None)}
        local: list[PairProposal] = []
        for primary in representatives:
            if targets and primary.node_id not in targets:
                continue
            primary_cover = need & set(primary.capabilities)
            source_options: list[tuple[str, ...]] = [()]
            if max_sources >= 1:
                for source in representatives:
                    if source.node_id == primary.node_id:
                        continue
                    extra = (need & set(source.capabilities)) - primary_cover
                    if extra:
                        source_options.append((source.node_id,))
            for sources in source_options[: 1 + max(0, max_sources) * max(1, len(representatives))]:
                cap = set(primary.capabilities)
                for source_id in sources:
                    source = next(r for r in representatives if r.node_id == source_id)
                    cap |= set(source.capabilities)
                covered = need & cap
                coverage = (len(covered) / len(need)) if need else 0.0
                status = "unknown" if not covered else ("covered" if coverage >= 0.5 else "partial")
                key = logical_pair_key(record.id, record.version, primary.executable_sha256, sources)
                if key in excluded_keys:
                    continue
                closes = bool(covered) and primary.node_id not in failed
                cost = measured_costs.get(primary.node_id)
                rank = (1.0 if closes else 0.0, coverage, 0.0 if primary.node_id in failed else 1.0,
                        1.0 if primary.family_sha256 == record.family_sha256 else 0.0,
                        -(float(cost) if cost is not None else neutral_cost), -float(len(sources)), primary.native_metric)
                local.append(PairProposal(key=key, hypothesis_id=record.id, primary_program_id=primary.node_id,
                                          primary_executable_sha256=primary.executable_sha256,
                                          source_program_ids=tuple(sources),
                                          requirement_keys=tuple(r.key for r in requirements), coverage=coverage,
                                          coverage_status=status, closes_blocker=closes, rank_vector=rank,
                                          created_step=step))
        local.sort(key=lambda p: (p.rank_vector, p.primary_program_id, p.source_program_ids), reverse=True)
        for proposal in local[:max_per_hypothesis]:
            proposals.append(((waiting.get(record.id, 0), proposal.rank_vector, record.id), proposal))
    proposals.sort(key=lambda item: item[0], reverse=True)
    return [proposal for _, proposal in proposals]
