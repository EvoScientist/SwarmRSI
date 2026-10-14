"""Complete lifecycle around original AIDE search, with no new search policy.

The portable export is an index into the WHOLE run directory, not a stand-alone
run.py. Keep the referenced artifacts, raw workspaces, caches and call receipts.
Public data and the qualified runtime remain separately bound prerequisites.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Callable, Mapping

from .aide_execution import workspace_inventory
from .aide_resources import AideOperationBudget, aide_run_lock
from .aide_session import AideSession
from .aide_state import atomic_json, canonical, encode_node, seal, unseal
from .aide_transport import AideResourceStop


@dataclass
class AideRuntime:
    session: AideSession
    budget: AideOperationBudget
    restore_factories: dict[str, Callable[..., Any]]
    report: Callable[[Any], str] | None = None


def _read(path: Path) -> dict[str, Any]:
    return unseal(json.loads(path.read_text(encoding="utf-8")))


def _once(path: Path, value: Mapping[str, Any]) -> None:
    if path.exists():
        if _read(path) != dict(value):
            raise ValueError(f"immutable AIDE record changed: {path.name}")
    else:
        atomic_json(path, seal(value))


def _file(root: Path, path: Path) -> dict[str, Any]:
    if path.is_symlink() or not path.is_file():
        raise ValueError("controller record must be a regular file")
    relative = path.absolute().relative_to(root).as_posix()
    data = path.read_bytes()
    return {"path": relative, "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data)}


def _usage(runtime: AideRuntime) -> dict[str, Any]:
    ledger = runtime.session.calls.ledger
    state = ledger.read()
    rows = state["attempts"]
    return {
        "reserved_provider_attempts": len(rows),
        "completed_provider_attempts": sum(row["state"] == "completed" for row in rows),
        "unresolved_provider_reservations": sum(row["state"] != "completed" for row in rows),
        "known_input_tokens": sum(row.get("input_tokens") or 0 for row in rows),
        "known_output_tokens": sum(row.get("output_tokens") or 0 for row in rows),
        "score_attempts": runtime.budget.score_attempts(),
        "elapsed_wall_seconds": max(0.0, ledger.clock() - state["started_at"]),
        "caveat": "A reserved pre-dispatch rejection is not proof a request was sent; unknown usage is not zero usage.",
    }


def _call_evidence(root: Path, calls_root: Path, ledger: Mapping[str, Any]) -> tuple[list, list]:
    """Validate original ledger bindings before freezing a new file index."""
    files, states = [], []
    for index, row in enumerate(ledger["attempts"]):
        if row["index"] != index:
            raise ValueError("AIDE call ledger index sequence changed")
        prefix = calls_root / f"{index:06d}"
        request = Path(str(prefix) + ".request.json")
        files.append(_file(root, request))
        if canonical(_read(request)) != row["request_sha256"]:
            raise ValueError("AIDE request differs from its ledger binding")
        response = Path(str(prefix) + ".response.json")
        response_present = response.exists() or response.is_symlink()
        response_bound = row.get("response_sha256") is not None
        if row["state"] == "completed" and not (response_present and response_bound):
            raise ValueError("AIDE completed call lost its response binding")
        if response_bound and not response_present:
            raise ValueError("AIDE bound response is missing")
        if response_present:
            files.append(_file(root, response))
            digest = canonical(_read(response))
            if response_bound and digest != row["response_sha256"]:
                raise ValueError("AIDE response differs from its ledger binding")
        states.append({"index": index, "ledger_state": row["state"],
                       "response_present": response_present,
                       "response_binding": "verified" if response_bound else "unresolved"})
    return files, states


def export_snapshot(root: Path, runtime: AideRuntime, identity: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    """Read durable state, verify completed evidence, index all preserved files."""
    session, executor = runtime.session, runtime.session.executor
    view = session.store.restore(**runtime.restore_factories)
    anchors = view.metadata.get("completed_evidence", {})
    if set(anchors) != {node.id for node in view.journal.nodes}:
        raise ValueError("export lost completed evidence anchors")
    for node in view.journal.nodes:
        bundle = executor.verify_completed(node_id=node.id, code=node.code)
        if canonical(bundle) != anchors[node.id]["bundle_sha256"]:
            raise ValueError("export execution bundle differs from checkpoint")
        binding = _read(executor.artifacts / node.id / "review_binding.json")
        session._validate_binding(node, bundle, binding, anchors[node.id]["review_binding_sha256"])
    # Selection is the original Journal operation, including negative/tied scores.
    best = view.journal.get_best_node(only_good=True)
    pending_evidence: dict[str, Any] | None = None
    if view.pending is not None:
        pending_evidence = {"node_id": view.pending.id, "selected_as_best": False, "state": "unreviewed"}
        if (executor.artifacts / view.pending.id / "execution_bundle.json").is_file():
            try:
                bundle = executor.verify_completed(node_id=view.pending.id, code=view.pending.code)
                pending_evidence.update(state="execution_bundle_available", evaluation=bundle["evaluation"],
                                        bundle_sha256=canonical(bundle))
            except Exception as exc:
                pending_evidence.update(state="unresolved", error_type=type(exc).__name__)
    ledger = session.calls.ledger.read()
    call_records, call_states = _call_evidence(root, session.calls.ledger.root, ledger)
    snapshot = {
        "schema_version": "aide-complete-run-export-v1", "identity_sha256": canonical(identity),
        "checkpoint": _file(root, session.store.path), "phase": view.phase,
        "journal": [encode_node(node) for node in view.journal.nodes],
        "pending": None if view.pending is None else encode_node(view.pending),
        "pending_evidence": pending_evidence,
        "best": None if best is None else {
            "node_id": best.id, "native_metric": best.metric.value,
            "artifact": (executor.artifacts / best.id / "artifact").relative_to(root).as_posix(),
            "raw_workspace": (executor.artifacts / best.id / "workspace").relative_to(root).as_posix(),
            "program": (executor.artifacts / best.id / "program.py").relative_to(root).as_posix(),
            "evidence_anchor": anchors[best.id],
        },
        "artifacts": {"path": executor.artifacts.relative_to(root).as_posix(),
                      "inventory": workspace_inventory(executor.artifacts)},
        "workspace": {"path": executor.workspace.relative_to(root).as_posix(),
                      "inventory": workspace_inventory(executor.workspace, skip_input=True)},
        # The report appends to the live ledger. Freeze the search prefix/value,
        # NEVER a hash of the mutable calls/ledger.json as a search dependency.
        "search_call_end": len(ledger["attempts"]), "search_ledger_snapshot": ledger,
        "call_records": call_records, "call_evidence_states": call_states,
        "preservation": "Retain this entire run directory. Raw symlinks are indexed, not followed. Public inputs/runtime remain external prerequisites from the frozen identity.",
    }
    path = root / "exports" / f"{canonical(snapshot)}.json"
    _once(path, snapshot)
    return {"path": path.relative_to(root).as_posix(), "sha256": canonical(snapshot)}, view.journal


def verify_snapshot(root: Path, reference: Mapping[str, Any], runtime: AideRuntime,
                    identity: Mapping[str, Any]) -> tuple[dict[str, Any], Any]:
    path = root / reference["path"]
    if path.resolve().parent != (root / "exports").resolve():
        raise ValueError("export reference escapes the run directory")
    snapshot = _read(path)
    if canonical(snapshot) != reference["sha256"] or snapshot["identity_sha256"] != canonical(identity):
        raise ValueError("AIDE export identity/hash mismatch")
    for row in [snapshot["checkpoint"], *snapshot["call_records"]]:
        target = root / row["path"]
        if root not in target.resolve().parents or _file(root, target) != row:
            raise ValueError("AIDE frozen file changed")
    for key in ("artifacts", "workspace"):
        target = root / snapshot[key]["path"]
        if root not in target.resolve().parents:
            raise ValueError("AIDE export directory escapes the run")
        if workspace_inventory(target, skip_input=key == "workspace") != snapshot[key]["inventory"]:
            raise ValueError("AIDE frozen artifact/workspace changed")
    ledger = runtime.session.calls.ledger.read()
    prefix = {**ledger, "attempts": ledger["attempts"][:snapshot["search_call_end"]]}
    if prefix != snapshot["search_ledger_snapshot"]:
        raise ValueError("AIDE search call prefix changed after terminal publication")
    files, states = _call_evidence(root, runtime.session.calls.ledger.root, prefix)
    if files != snapshot["call_records"] or states != snapshot["call_evidence_states"]:
        raise ValueError("AIDE search call evidence changed")
    view = runtime.session.store.restore(**runtime.restore_factories)
    if ([encode_node(node) for node in view.journal.nodes] != snapshot["journal"]
            or (None if view.pending is None else encode_node(view.pending)) != snapshot["pending"]):
        raise ValueError("AIDE terminal Journal changed")
    return snapshot, view.journal


def _report(root: Path, runtime: AideRuntime, search: Mapping[str, Any], journal: Any) -> dict[str, Any]:
    session = runtime.session
    cfg = session.agent.cfg.report
    binding = {"search_final_sha256": canonical(search), "task_desc": session.agent.task_desc,
               "journal_sha256": canonical([encode_node(node) for node in journal.nodes]),
               "config": {"model": cfg.model, "temp": cfg.temp}}
    path = root / "report_state.json"
    if path.exists():
        state = _read(path)
        if state["input"] != binding:
            raise ValueError("AIDE report inputs changed on resume")
        if state["state"] == "completed":
            if _file(root, root / "report.md") != state["output"]:
                raise ValueError("AIDE completed report changed")
            return state
    else:
        state = {"schema_version": "aide-report-phase-v1", "state": "started",
                 "input": binding, "call_start": session.calls.count}
        atomic_json(path, seal(state))
    if runtime.report is None:
        raise ValueError("enabled original report generator is missing")
    try:
        with session.calls.phase(state["call_start"]):
            report = runtime.report(journal)
        if not isinstance(report, str):
            raise ValueError("original AIDE report did not return text")
        temporary = root / "report.md.tmp"
        with temporary.open("w", encoding="utf-8") as stream:
            stream.write(report)
            stream.flush()
            os.fsync(stream.fileno())
        temporary.replace(root / "report.md")
        state.update(state="completed", output=_file(root, root / "report.md"))
        state.pop("error_type", None)
    except Exception as exc:
        state.update(state="resource_stopped" if isinstance(exc, AideResourceStop) else "interrupted",
                     error_type=type(exc).__name__)
    atomic_json(path, seal(state))
    return state


def _require_empty_precheckpoint_initialization(root: Path) -> None:
    """A lost active checkpoint is not permission to create a second search."""
    if any((root / name).exists() for name in
           ("search_final.json", "interrupted.json", "final.json", "exports", "report_state.json")):
        raise ValueError("AIDE checkpoint missing after recorded activity")
    calls = root / "calls"
    if calls.exists():
        if any(p.name not in {"ledger.json", "ledger.lock"} for p in calls.iterdir()):
            raise ValueError("AIDE checkpoint missing after provider activity")
        if (calls / "ledger.json").exists() and _read(calls / "ledger.json")["attempts"]:
            raise ValueError("AIDE checkpoint missing after provider activity")
    artifacts = root / "artifacts"
    if artifacts.exists() and any(p.name != "workspace_owner.json" for p in artifacts.iterdir()):
        raise ValueError("AIDE checkpoint missing after execution activity")


def run_controller(*, root: Path, identity: Mapping[str, Any],
                   initialize: Callable[[Path], AideRuntime], resume: bool = False,
                   generate_report: bool = True) -> dict[str, Any]:
    """Own initialization through final publication; resume is explicit only."""
    with aide_run_lock(root) as root:
        identity = {**identity, "generate_report": generate_report}
        identity_path = root / "identity.json"
        if identity_path.exists():
            if not resume:
                raise ValueError("existing AIDE run requires explicit resume")
            if _read(identity_path) != identity:
                raise ValueError("AIDE controller identity changed on resume")
        else:
            if resume or set(p.name for p in root.iterdir()) != {"controller.lock"}:
                raise ValueError("refusing to initialize an unmanaged AIDE run")
            _once(identity_path, identity)
        search_path = root / "search_final.json"
        if search_path.exists() and not (root / "checkpoint.json").is_file():
            raise ValueError("terminal AIDE search lost its checkpoint; do not initialize")
        if resume and not (root / "checkpoint.json").is_file():
            _require_empty_precheckpoint_initialization(root)
        runtime = initialize(root)
        if runtime.session.store.path != root / "checkpoint.json":
            raise ValueError("controller requires its canonical checkpoint path")
        if runtime.budget.ledger is not runtime.session.calls.ledger:
            raise ValueError("AIDE controller resource ledgers differ")
        steps = runtime.session.agent.cfg.agent.steps
        if isinstance(steps, bool) or not isinstance(steps, int) or steps <= 0:
            raise ValueError("invalid original AIDE step count")
        if search_path.exists():
            # Terminal search is immutable, even if a pending node remains.
            # Report resume must NEVER try to complete that pending search node.
            search = _read(search_path)
            if search["identity_sha256"] != canonical(identity):
                raise ValueError("terminal AIDE identity changed")
            snapshot, journal = verify_snapshot(root, search["export"], runtime, identity)
        else:
            error_type = None
            try:
                while len(runtime.session.agent.journal.nodes) < steps:
                    if runtime.session.phase == "ready":
                        runtime.budget.before_new_search()
                    # Pending replay precedes fresh-operation budget checks.
                    runtime.session.step()
                reason, terminal = "original_steps_completed", True
            except AideResourceStop as exc:
                reason, terminal, error_type = str(exc), True, type(exc).__name__
            except Exception as exc:
                reason, terminal, error_type = "technical_interruption", False, type(exc).__name__
            reference, journal = export_snapshot(root, runtime, identity)
            snapshot = _read(root / reference["path"])
            search = {"schema_version": "aide-search-result-v1", "identity_sha256": canonical(identity),
                      "terminal": terminal, "stop_reason": reason, "error_type": error_type,
                      "completed_steps": len(journal.nodes), "configured_steps": steps,
                      "best": snapshot["best"], "pending": snapshot["pending_evidence"],
                      "export": reference, "search_usage": _usage(runtime)}
            if terminal:
                _once(search_path, search)
            else:
                # An interrupted snapshot is not permission to restart search.
                atomic_json(root / "interrupted.json", seal(search))
        report = {"state": "disabled" if not generate_report else "not_started_search_interrupted"}
        if search["terminal"] and generate_report:
            report = _report(root, runtime, search, journal)
        result = {"schema_version": "aide-controller-result-v1", "search": search,
                  "report": report, "total_usage": _usage(runtime),
                  "whole_run_directory_required": True}
        atomic_json(root / "final.json", seal(result))
        return result
