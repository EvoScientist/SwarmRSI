"""Build the Claude Code reference thresholds from the official trace index.

The thresholds were first taken by regex out of the leaderboard page's bundled
JavaScript. That scrape searched backwards from the entry id through a fixed
window, and `.*?` anchored on the first `"id"` it met, which belonged to a
different entry; every threshold it produced was Codex + GPT-5.4 rather than
Claude Code + DeepSeek-V4-Pro, and no task in it matched the intended
configuration. A panel was run and reported against those numbers.

`metadata/index.parquet` in the NatureBench-traces dataset carries one row per
(harness, model, case) with the judged score, so there is a keyed lookup and no
parsing. Each value is cross-checked here against that case's own
submissions.jsonl when the trace directory is available.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def reference_rows(index_path: Path, harness: str, model: str) -> dict:
    import pandas as pd

    frame = pd.read_parquet(index_path)
    for column in ("harness", "model", "case_id", "effective_improvement", "judge_valid"):
        if column not in frame.columns:
            raise SystemExit(f"index is missing the {column} column")
    rows = frame[(frame.harness == harness) & (frame.model == model)]
    if rows.empty:
        raise SystemExit(f"index has no rows for {harness} + {model}")
    out = {}
    for row in rows.itertuples():
        value = None if pd.isna(row.effective_improvement) else float(row.effective_improvement)
        out[row.case_id] = {"effective_improvement": value, "judge_valid": bool(row.judge_valid)}
    return out


def crosscheck(reference: dict, traces_root: Path) -> list[str]:
    """Each threshold against the best scored submission in that case's own trace."""
    problems = []
    for case, entry in reference.items():
        submissions = traces_root / case / "submissions.jsonl"
        if not submissions.is_file():
            continue
        scored = []
        for line in submissions.read_text().splitlines():
            if not line.strip():
                continue
            value = json.loads(line).get("aggregate_improvement")
            if value is not None:
                scored.append(float(value))
        if not scored or entry["effective_improvement"] is None:
            continue
        if abs(max(scored) - entry["effective_improvement"]) > 1e-6:
            problems.append(f"{case}: index {entry['effective_improvement']} vs trace best {max(scored)}")
    return problems


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--index", type=Path, required=True)
    parser.add_argument("--harness", default="claude-code")
    parser.add_argument("--model", default="deepseek-v4-pro")
    parser.add_argument("--traces", type=Path, default=None, help="directory of per-case trace folders, to cross-check")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()

    reference = reference_rows(args.index, args.harness, args.model)
    problems = crosscheck(reference, args.traces) if args.traces else []
    if problems:
        raise SystemExit("index and traces disagree:\n  " + "\n  ".join(problems[:10]))
    args.out.write_text(json.dumps({
        "schema_version": "naturebench-reference-from-index-v1",
        "harness": args.harness, "model": args.model,
        "metric": "effective_improvement (aggregate improvement after the validity judgement)",
        "source": f"{args.index.name} of FrontisAI/NatureBench-traces",
        "crosschecked_against_traces": bool(args.traces),
        "tasks": reference,
    }, indent=1) + "\n")
    valid = sum(1 for v in reference.values() if v["judge_valid"])
    print(json.dumps({"cases": len(reference), "judge_valid": valid, "crosschecked": bool(args.traces)}))


main()
