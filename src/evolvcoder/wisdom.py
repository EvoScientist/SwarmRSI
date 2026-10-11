"""Source-bound cross-task experience: advisory, immutable per inner run."""
from __future__ import annotations

from copy import deepcopy
import json
import math
from pathlib import Path
import re

from .contracts import canonical_sha256
from .learning_evidence import verify_run_evidence, verify_ref, source_ref
from .retained_results import read_json, require, verify_seal, write_immutable


def words(text: str) -> set[str]:
    ignored = {"the", "and", "for", "with", "from", "this", "that", "true", "false", "none", "return"}
    return set(re.findall(r"[a-z][a-z0-9_]{2,}", text.lower())) - ignored


def experience_entries(run: dict, *, runtime_tags: tuple[str, ...] = ("python",)) -> list[dict]:
    verify_run_evidence(run)
    # Selection is a public development policy trial, not held-out evaluation.
    # It may accumulate for FUTURE snapshots; the active run's W0 never changes.
    require(run["source_role"] in {"development", "selection"}, "heldout results cannot enter Wisdom")
    require(bool(runtime_tags) and all(isinstance(x, str) and x for x in runtime_tags), "missing compatibility tags")
    entries = []
    for observation in run["observations"]:
        measured = observation.get("native_metric")
        parent = observation.get("parent_metric")
        output_relation = observation.get("implementation_observation", {}).get("output_relation")
        if observation.get("recovery_route") == "retry_evaluator_same_candidate" or observation.get("coder_status") == "evaluation_invalid":
            kind = "evaluation_chain_failure"
            guidance = "Retry or repair the evaluation chain on the same candidate; this measurement failure does not discredit its scientific content."
        elif observation.get("coder_status") == "plan_blocked":
            kind = "implementation_plan_blocked"
            guidance = "Reconsider this implementation plan while preserving the scientific task; infeasibility of this realization is not hypothesis refutation."
        elif not observation["artifact_valid"]:
            kind = "implementation_failure"
            guidance = "Use the bound failure to avoid or repair this implementation defect; it is not a scientific refutation."
        elif output_relation == "identical_observed_outputs":
            kind = "unchanged_observed_outputs"
            guidance = "Check whether the intended intervention reached the executed path; identical observed outputs alone do not refute the hypothesis."
        elif measured is not None and parent is not None and measured > parent:
            kind = "measured_program_improvement"
            guidance = "Consider this actual code change as an implementation precedent, then re-implement and re-evaluate on the new task."
        else:
            kind = "measured_non_improvement"
            guidance = "The scored program did not improve this parent on these inputs; inspect applicability before reusing its implementation."
        body = {"schema_version": "evolvcoder-wisdom-entry-v1", "source_task_id": run["task_id"],
                "source_run_id": run["run_id"], "source_action_id": observation["action_id"],
                "source_evidence_sha256": run["evidence_sha256"], "kind": kind,
                "required_runtime_tags": sorted(set(runtime_tags)),
                "observation": deepcopy(observation), "guidance": guidance,
                "guidance_provenance": "controller_template_advisory_not_scientific_judgment",
                "source_role": run["source_role"], "authority": "advisory_only",
                "limitations": ["Declared hypotheses and families are not independently verified realizations.",
                                "Source scores are not transferable to the destination program.",
                                "No hidden labels or independent human judgments are supplied by this record."]}
        entries.append({**body, "entry_sha256": canonical_sha256(body)})
    return entries


class WisdomLibrary:
    """Append-only entries plus their complete run-evidence records."""
    def __init__(self, root: Path):
        self.root = root.resolve()
        (self.root / "sources").mkdir(parents=True, exist_ok=True)
        (self.root / "entries").mkdir(parents=True, exist_ok=True)

    def ingest(self, run: dict, *, runtime_tags: tuple[str, ...] = ("python",)) -> tuple[str, ...]:
        entries = experience_entries(run, runtime_tags=runtime_tags)
        write_immutable(self.root / "sources" / (run["evidence_sha256"] + ".json"), run)
        for entry in entries:
            write_immutable(self.root / "entries" / (entry["entry_sha256"] + ".json"), entry)
        return tuple(entry["entry_sha256"] for entry in entries)

    def freeze(self, path: Path) -> dict:
        entries = [read_json(p) for p in sorted((self.root / "entries").glob("*.json"))]
        sources = {e["source_evidence_sha256"]: read_json(self.root / "sources" / (e["source_evidence_sha256"] + ".json"))
                   for e in entries}
        body = {"schema_version": "evolvcoder-wisdom-snapshot-v1", "entries": entries, "sources": sources}
        snapshot = {**body, "snapshot_sha256": canonical_sha256(body)}
        verify_snapshot(snapshot)
        write_immutable(path, snapshot)
        return snapshot


def verify_snapshot(snapshot: dict) -> None:
    verify_seal(snapshot, "snapshot_sha256")
    require(snapshot["schema_version"] == "evolvcoder-wisdom-snapshot-v1", "unknown Wisdom snapshot")
    expected = {}
    for digest, source in snapshot["sources"].items():
        require(source["evidence_sha256"] == digest, "Wisdom source identity mismatch")
        verify_run_evidence(source)
        require(source["source_role"] in {"development", "selection"}, "non-development Wisdom source")
    for entry in snapshot["entries"]:
        verify_seal(entry, "entry_sha256")
        source = snapshot["sources"].get(entry["source_evidence_sha256"])
        require(source is not None, "Wisdom source missing")
        key = (source["evidence_sha256"], tuple(entry["required_runtime_tags"]))
        if key not in expected:
            expected[key] = {e["entry_sha256"]: e for e in experience_entries(source, runtime_tags=key[1])}
        require(expected[key].get(entry["entry_sha256"]) == entry, "Wisdom entry not derived from original observation")
    require(len({e["entry_sha256"] for e in snapshot["entries"]}) == len(snapshot["entries"]), "duplicate Wisdom entries")


class WisdomContext:
    """A frozen run-local retriever. Off mode never opens a library/snapshot."""
    def __init__(self, *, task_id: str, mode: str = "off", snapshot_path: Path | None = None,
                 runtime_tags: tuple[str, ...] = ("python",), max_entries: int = 3, max_characters: int = 16000):
        require(mode in {"off", "readonly"}, "Wisdom inner runs must use off or readonly snapshots")
        require(type(max_entries) is int and max_entries > 0 and type(max_characters) is int and max_characters > 0,
                "Wisdom projection bounds must be positive integers")
        self.task_id, self.mode = task_id, mode
        self.runtime_tags = frozenset(runtime_tags)
        self.max_entries, self.max_characters = max_entries, max_characters
        self.snapshot = None
        self.snapshot_ref = None
        if mode == "readonly":
            require(snapshot_path is not None, "readonly Wisdom requires a frozen snapshot")
            self.snapshot = read_json(snapshot_path)
            verify_snapshot(self.snapshot)
            self.snapshot_ref = source_ref(snapshot_path)
        self.config = {"mode": mode, "task_id": task_id, "runtime_tags": sorted(self.runtime_tags),
                       "max_entries": max_entries, "max_characters": max_characters,
                       "snapshot_sha256": self.snapshot["snapshot_sha256"] if self.snapshot else None}

    def retrieve(self, query: str) -> dict:
        if self.mode == "off":
            selected, considered, incompatible, same_task, omitted, used = [], 0, 0, 0, 0, 0
        else:
            # The original sources were verified when the snapshot was loaded;
            # the actual injected objects are this immutable in-memory copy.
            verify_ref(self.snapshot_ref)
            query_words = words(query)
            selected, candidates = [], []
            incompatible = same_task = omitted = used = 0
            for entry in self.snapshot["entries"]:
                if entry["source_task_id"] == self.task_id:
                    same_task += 1
                    continue
                if not set(entry["required_runtime_tags"]).issubset(self.runtime_tags):
                    incompatible += 1
                    continue
                observation = entry["observation"]
                text = json.dumps({k: observation.get(k) for k in ("declared_family", "declared_hypothesis",
                                  "failure_summary", "program_diff_excerpt")}, ensure_ascii=False)
                tokens = words(text)
                overlap = len(tokens & query_words)
                score = overlap / math.sqrt(max(1, len(tokens) * len(query_words)))
                if score > 0:
                    candidates.append((score, entry["entry_sha256"], entry))
            considered = len(candidates)
            seen_observations = set()
            for score, _, entry in sorted(candidates, key=lambda row: (-row[0], row[1])):
                # Same source/action with different compatibility tags is one precedent.
                identity = (entry["source_run_id"], entry["source_action_id"])
                if identity in seen_observations:
                    omitted += 1
                    continue
                # Full validation argv/stdout stays in the bound source entry;
                # it must not crowd actual errors and changes out of the prompt.
                observation = {k: deepcopy(v) for k, v in entry["observation"].items()
                               if k != "failure_validation"}
                observation["omitted_full_validation_rows"] = len(entry["observation"].get("failure_validation", []))
                for field, omitted_field, limit in (("code_excerpt", "omitted_code_characters", 1500),
                                               ("program_diff_excerpt", "omitted_diff_characters", 3000)):
                    value = observation.get(field, "")
                    observation[field] = value[:limit]
                    observation[omitted_field] = observation.get(omitted_field, 0) + max(0, len(value) - limit)
                projected = {"entry_sha256": entry["entry_sha256"], "source_task_id": entry["source_task_id"],
                             "source_run_id": entry["source_run_id"], "source_action_id": entry["source_action_id"],
                             "kind": entry["kind"], "observation": observation,
                             "guidance": entry["guidance"], "authority": "advisory_only",
                             "limitations": entry["limitations"], "retrieval_score": score}
                size = len(json.dumps(projected, ensure_ascii=False))
                if len(selected) >= self.max_entries or used + size > self.max_characters:
                    omitted += 1
                    continue
                selected.append(deepcopy(projected)); used += size; seen_observations.add(identity)
        body = {"schema_version": "evolvcoder-wisdom-retrieval-v1", "mode": self.mode,
                "snapshot_sha256": self.config["snapshot_sha256"], "query_sha256": canonical_sha256(query),
                "entries": selected, "selected_count": len(selected), "considered_count": considered,
                "same_task_excluded": same_task, "incompatible_excluded": incompatible,
                "omitted_count": omitted, "projected_characters": used,
                "injection_status": "ready_for_planner" if selected else "no_cross_task_context",
                "effect_on_decision": "not_inferred_from_injection"}
        return {**body, "retrieval_sha256": canonical_sha256(body)}
