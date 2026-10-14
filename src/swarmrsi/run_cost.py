"""Token and cost totals of one NatureBench-style run, from its own call records.

Every coder call is a ``coder.model_response`` event in ``cells/*/trajectory.jsonl``
and every planner call a ``planner_batches/*.record.json``; since the transport
records the provider's usage per response (``_cost_usd``, ``_cached_input_tokens``,
``_reasoning_tokens``) the run's totals can be summed without the provider's
dashboard. Calls without those fields (older runs, or a route that does not
report them) are counted in ``calls_without_*`` so a partial total is never
presented as a complete one. Transport retries that returned no response are not
recorded anywhere and are therefore not included.
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path
from typing import Any

FIELDS = ("cost_usd", "cached_input_tokens", "reasoning_tokens")


def _empty() -> dict[str, Any]:
    return {"calls": 0, "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
            "cached_input_tokens": 0, "reasoning_tokens": 0,
            **{f"calls_without_{field}": 0 for field in FIELDS}}


def _add(total: dict[str, Any], usage: dict[str, Any], response: dict[str, Any]) -> None:
    total["calls"] += 1
    total["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
    total["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
    for field in FIELDS:
        value = response.get("_" + field)
        if isinstance(value, (int, float)) and not isinstance(value, bool):
            total[field] += value
        else:
            total[f"calls_without_{field}"] += 1


def summarize_run_cost(run_dir: Path) -> dict[str, Any]:
    run_dir = Path(run_dir)
    coder, planner = _empty(), _empty()
    for path in sorted(glob.glob(str(run_dir / "cells" / "*" / "trajectory.jsonl"))):
        with open(path, encoding="utf-8") as handle:
            for line in handle:
                event = json.loads(line)
                if event.get("kind") == "coder.model_response":
                    payload = event.get("payload") or {}
                    response = payload.get("response") if isinstance(payload.get("response"), dict) else {}
                    _add(coder, payload, response)
    for path in sorted(glob.glob(str(run_dir / "planner_batches" / "*.record.json"))):
        record = json.loads(Path(path).read_text(encoding="utf-8"))
        response = record.get("response") if isinstance(record.get("response"), dict) else {}
        _add(planner, record.get("usage") or {}, response)
    total = _empty()
    for part in (coder, planner):
        for key in total:
            total[key] += part[key]
    for part in (coder, planner, total):
        part["cost_usd"] = round(part["cost_usd"], 6)
    return {"schema_version": "swarmrsi-run-cost-v1", "run_dir": str(run_dir),
            "coder": coder, "planner": planner, "total": total,
            "complete": all(total[f"calls_without_{field}"] == 0 for field in FIELDS)}


def main(argv: list[str]) -> int:
    for run_dir in argv or ["."]:
        summary = summarize_run_cost(Path(run_dir))
        (Path(run_dir) / "cost_summary.json").write_text(json.dumps(summary, indent=1) + "\n")
        print(json.dumps({"run_dir": run_dir, **summary["total"], "complete": summary["complete"]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
