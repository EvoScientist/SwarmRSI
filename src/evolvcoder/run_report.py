"""One delivery summary shape for every method, derived from verified evidence.

A row reports what its run actually recorded. A field a method does not produce
stays absent instead of receiving a default, an unfinished run is reported as
unfinished rather than as a zero, and hypothesis/program structure appears only
for runs that recorded it. Nothing here re-runs, re-scores or promotes anything.
"""
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable

from .contracts import canonical_sha256
from .learning_evidence import source_ref, verify_ref, verify_run_evidence
from .retained_results import read_json, require, verify_seal

SCHEMA = "evolvcoder-run-report-v1"
WIN_RULE = ("a valid new (non-root) program whose native score is strictly greater than the task's recorded "
            "reference; a retained root or fallback never counts as a new-program win")
REFERENCE_KIND = ("historical same-model trace recorded in the benchmark profile; not a same-route rerun "
                  "of the comparator")


def _live_progress(output_dir: Path) -> dict | None:
    """Unsealed in-flight view; never mixed into a delivered result."""
    summaries = sorted((output_dir / "validation/trials").glob("*/run/live_summary.json"))
    if not summaries:
        return None
    live = read_json(summaries[-1])
    accounting = live.get("accounting") or {}
    return {"steps": len(live.get("steps", [])), "root_native_metric": live.get("root_native_metric"),
            "best_native_metric": live.get("best_native_metric"),
            "best_evolved_native_metric": live.get("best_evolved_native_metric"),
            "stop_reason": live.get("stop_reason"),
            "model_calls": accounting.get("model_calls"), "tool_calls": accounting.get("tool_calls"),
            "input_tokens": accounting.get("input_tokens"), "output_tokens": accounting.get("output_tokens"),
            "score_evaluations": accounting.get("score_evaluations"),
            "wall_seconds": accounting.get("wall_seconds"),
            "frontier_report": live.get("frontier_report")}


def _reference(profile: dict, task_id: str) -> float | None:
    task = (profile.get("tasks") or {}).get(task_id) or {}
    value = task.get("claude_code_score")
    return value if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def run_summary(output_dir: Path) -> dict:
    """Summarize one run directory produced by any method."""
    output_dir = Path(output_dir).resolve()
    row: dict[str, Any] = {"run_dir": str(output_dir), "run_id": output_dir.name, "evidence": "absent",
                           "status": "not_started"}
    inputs = output_dir / "inputs.json"
    if inputs.is_file():
        config = read_json(inputs).get("config") or {}
        row.update(benchmark=config.get("benchmark"), task_id=config.get("task"), method=config.get("method"),
                   seed=config.get("seed"), model=(config.get("model") or {}).get("model"),
                   search=config.get("search"))
        row["status"] = "unfinished"
    delivery_path = output_dir / "result.json"
    if not delivery_path.is_file():
        progress = _live_progress(output_dir)
        if progress is not None:
            row.update(evidence="live_progress", progress=progress)
        return row
    delivery = read_json(delivery_path)
    if "delivery_sha256" not in delivery:
        # Another method's terminal file; report it as unreadable rather than
        # guessing at its fields.
        return {**row, "evidence": "unreadable_delivery",
                "reason": "result.json is not an evolvcoder sealed delivery record"}
    verify_seal(delivery, "delivery_sha256")
    for ref in delivery["refs"]:
        verify_ref(ref)
    result = delivery["result"]
    row.update(evidence="sealed_delivery", status=result["status"], reason=result.get("reason"),
               method=result.get("method") or row.get("method"), task_id=result.get("task_id") or row.get("task_id"),
               run_id=result.get("run_id") or row["run_id"], score=result.get("score"),
               evaluator_id=result.get("evaluator_id"), artifact=result.get("artifact"),
               evidence_path=result.get("evidence"), usage=result.get("usage") or {},
               details=result.get("details") or {})
    profile_path = output_dir / "profile.json"
    if profile_path.is_file():
        row["reference"] = _reference(read_json(profile_path), row.get("task_id") or "")
        row["reference_kind"] = REFERENCE_KIND if row["reference"] is not None else None
    record = delivery.get("record")
    if record is not None:
        verify_run_evidence(record)
        final = read_json(Path(record["run_dir"]) / "final.json")
        row.update(terminal_status=record["terminal_status"], accounting=record["accounting"],
                   cost_usd=record.get("cost_usd"), root_native_metric=final.get("root_native_metric"),
                   best_native_metric=record["utility"],
                   best_evolved_native_metric=record["best_evolved_native_metric"],
                   best_evolved_node_id=record["best_evolved_node_id"],
                   artifact_valid=record["artifact_valid"], policy=record["policy"])
        if final.get("frontier_report") is not None:
            row["frontier_report"] = final["frontier_report"]
    terminal = output_dir / "benchmark_result.json"
    if terminal.is_file():
        official = read_json(terminal)
        verify_seal(official, "receipt_sha256")
        row["official_terminal"] = {key: official.get(key) for key in
                                    ("status", "official_score", "reason", "new_scored_candidates",
                                     "terminal_evaluations", "total_elapsed_seconds")}
    if delivery.get("failure") is not None:
        row["validation_failure"] = delivery["failure"]
    if delivery.get("rsi_state") is not None:
        state = delivery["rsi_state"]
        row["rsi"] = {key: state.get(key) for key in ("completed_rounds", "finished", "halted", "status")}
    best_new = row.get("best_evolved_native_metric")
    reference = row.get("reference")
    row["new_program_beats_reference"] = (
        None if reference is None or best_new is None or not row.get("artifact_valid")
        else bool(best_new > reference))
    return row


def campaign_report(output_dirs: Iterable[Path], *, plan_id: str | None = None) -> dict:
    """A fixed-denominator table: every requested run keeps its row."""
    rows = [run_summary(Path(directory)) for directory in output_dirs]
    require(bool(rows), "a campaign report needs at least one run directory")
    delivered = [row for row in rows if row["evidence"] == "sealed_delivery"]
    wins = [row for row in rows if row.get("new_program_beats_reference") is True]
    body = {"schema_version": SCHEMA, "plan_id": plan_id, "rows": rows,
            "denominator": len(rows), "delivered": len(delivered),
            "completed": sum(row["status"] == "completed" for row in rows),
            "new_program_wins": len(wins), "winning_tasks": sorted(row["task_id"] for row in wins if row.get("task_id")),
            "unfinished_tasks": sorted(row.get("task_id") or row["run_id"] for row in rows
                                       if row["evidence"] != "sealed_delivery"),
            "win_rule": WIN_RULE, "reference_kind": REFERENCE_KIND,
            "claim_scope": ("one recorded delivery per requested run; unfinished runs stay in the denominator and are "
                            "never counted as losses with a fabricated score, and no result here is a stability claim")}
    return {**body, "report_sha256": canonical_sha256(body)}


def write_campaign_report(path: Path, output_dirs: Iterable[Path], *, plan_id: str | None = None) -> dict:
    from .retained_results import write_immutable
    report = campaign_report(output_dirs, plan_id=plan_id)
    write_immutable(Path(path), report)
    return {**report, "report_ref": source_ref(Path(path))}
