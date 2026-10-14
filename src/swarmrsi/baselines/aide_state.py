"""Lossless, flat persistence for the original AIDE Node/Journal objects.

This module does not implement a search policy. Factories come from the pinned
upstream checkout; pending nodes retain their real parent.children edge without
being inserted into Journal.nodes before upstream review has completed.
"""
from __future__ import annotations

import hashlib
import json
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping


def canonical(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def atomic_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def seal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    payload["sha256"] = canonical(payload)
    return payload


def unseal(value: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(value)
    expected = payload.pop("sha256", None)
    if expected != canonical(payload):
        raise ValueError("AIDE persisted content hash mismatch")
    return payload


NODE_FIELDS = (
    "code", "plan", "step", "id", "ctime", "_term_out", "exec_time",
    "exc_type", "exc_info", "exc_stack", "analysis", "is_buggy",
)


def encode_node(node: Any) -> dict[str, Any]:
    result = {name: getattr(node, name) for name in NODE_FIELDS}
    result.update({
        "parent_id": None if node.parent is None else node.parent.id,
        "children_ids": sorted(child.id for child in node.children),
        "metric": None if node.metric is None else {
            "value": node.metric.value,
            "maximize": node.metric.maximize,
            "is_worst": node.metric.is_worst,
        },
    })
    return result


def _tuples(value: Any) -> Any:
    return tuple(_tuples(item) for item in value) if isinstance(value, list) else value


@dataclass
class RestoredJournal:
    journal: Any
    pending: Any | None
    preview: str | None
    rng_state: tuple[Any, ...]
    phase: str
    metadata: dict[str, Any]


class AideJournalStore:
    def __init__(self, path: Path, *, identity: Mapping[str, Any]) -> None:
        self.path = path
        self.identity = dict(identity)

    def save(self, *, journal: Any, pending: Any | None, preview: str | None,
             phase: str, metadata: Mapping[str, Any] | None = None) -> None:
        nodes = [encode_node(node) for node in journal.nodes]
        if pending is not None and any(row["id"] == pending.id for row in nodes):
            raise ValueError("pending node is already in Journal")
        if pending is not None and pending.step is not None:
            raise ValueError("unappended pending node has a Journal step")
        payload = {
            "schema_version": "aide-journal-checkpoint-v1",
            "identity": self.identity,
            "nodes": nodes,
            "pending": None if pending is None else encode_node(pending),
            "preview": preview,
            "preview_sha256": canonical(preview),
            "rng_state": random.getstate(),
            "phase": phase,
            "metadata": dict(metadata or {}),
        }
        atomic_json(self.path, seal(payload))

    def restore(self, *, node_factory: Callable[..., Any],
                journal_factory: Callable[..., Any], metric_factory: Callable[..., Any],
                worst_metric_factory: Callable[..., Any]) -> RestoredJournal:
        payload = unseal(json.loads(self.path.read_text(encoding="utf-8")))
        if payload.get("schema_version") != "aide-journal-checkpoint-v1":
            raise ValueError("unsupported AIDE checkpoint schema")
        if payload["identity"] != self.identity:
            raise ValueError("AIDE source/config/task identity changed on resume")
        if payload["preview_sha256"] != canonical(payload["preview"]):
            raise ValueError("AIDE data preview changed on resume")
        rows = list(payload["nodes"])
        pending_row = payload["pending"]
        if pending_row is not None:
            if pending_row["step"] is not None:
                raise ValueError("pending node must not have an appended step")
            rows.append(pending_row)
        nodes: dict[str, Any] = {}
        for row in rows:
            identifier, parent_id = row["id"], row["parent_id"]
            if identifier in nodes or (parent_id is not None and parent_id not in nodes):
                raise ValueError("duplicate node or non-ancestral parent in AIDE checkpoint")
            metric = row["metric"]
            if metric is not None and metric["is_worst"] != (metric["value"] is None):
                raise ValueError("inconsistent upstream AIDE metric")
            restored_metric = None
            if metric is not None:
                restored_metric = (
                    worst_metric_factory() if metric["is_worst"] else
                    metric_factory(metric["value"], maximize=metric["maximize"])
                )
            nodes[identifier] = node_factory(
                **{name: row[name] for name in NODE_FIELDS},
                parent=nodes.get(parent_id), metric=restored_metric,
            )
        for row in rows:
            if sorted(child.id for child in nodes[row["id"]].children) != row["children_ids"]:
                raise ValueError("AIDE pending/completed child edges do not match")
        journal = journal_factory()
        for index, row in enumerate(payload["nodes"]):
            if row["step"] != index:
                raise ValueError("AIDE Journal insertion order changed")
            journal.append(nodes[row["id"]])
        rng_state = _tuples(payload["rng_state"])
        # Validate with an isolated Random instance, not the caller's generator.
        random.Random().setstate(rng_state)
        return RestoredJournal(
            journal=journal,
            pending=None if pending_row is None else nodes[pending_row["id"]],
            preview=payload["preview"], rng_state=rng_state,
            phase=payload["phase"], metadata=payload["metadata"],
        )
