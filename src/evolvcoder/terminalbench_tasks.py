"""Terminal-Bench task preparation, custody and official scoring.

Terminal-Bench is not a metric benchmark. A task ships an instruction, a pinned
environment image, a pinned verifier image and a hidden pytest suite; a trial
passes only when every test passes. Two consequences shape this module:

* **The official score is binary.** ``reward.txt`` is 1 or 0. The per-test CTRF
  report the verifier also writes is a dense signal, but it is derived from the
  hidden suite, which the reference agents (Claude Code, Codex, Terminus) never
  see. Using it to steer a search would give this method privileged information
  and make the comparison meaningless, so it is *not* the search metric. It is
  read only for diagnostics, and ``metadata.json`` records which signal a task
  package was built for so a diagnostic package can never be mistaken for a
  leaderboard-comparable one.
* **The search signal is the candidate's own validation.** Exactly as on
  NatureBench, the candidate writes ``internal_validation.json`` beside its
  solution: an estimate it computed itself, from evidence it is allowed to have.
  That is what the frontier ranks on. The official verifier runs once, at the
  terminal, on the selected solution.

Nothing here executes a trial. Execution is the official ``harbor`` harness, so
the environment, the verifier and the reward stay upstream's; this module only
materializes the local task directory harbor consumes and reads what it wrote.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tomllib
from pathlib import Path
from typing import Any

SCHEMA = "evolvcoder-terminalbench-custody-v1"
BENCHMARK = "terminalbench"

# The search metric is the candidate's self-reported estimate, not the hidden
# suite. `direction` is maximize because a candidate reports confidence that its
# solution satisfies the instruction.
METRIC = {"name": "self_validation", "direction": "maximize", "role": "public_validation"}

# What the official verifier writes, relative to the trial's logs directory.
REWARD_NAME = "reward.txt"
CTRF_NAME = "ctrf.json"

SOLUTION_NAME = "solve.sh"
INTERNAL_VALIDATION_NAME = "internal_validation.json"

STARTER = """#!/bin/bash
# Solution entrypoint. It runs as root inside the task's own environment image,
# with the working directory set by that image (usually /app).
#
# Write the artifacts the instruction asks for, then write your own honest
# estimate of whether you satisfied it to internal_validation.json beside this
# script. That estimate is the only signal the search ranks you on; the official
# test suite is hidden and is run exactly once, on the finally selected solution.
set -euo pipefail

echo "no solution yet" >&2
exit 1
"""

README = """# {task_id}

{instruction}

## How this task is run

Your solution is `{solution}`, a shell script executed as the task's entrypoint
inside the pinned environment image `{image}`. It runs with the resources the
task declares: {cpus} CPUs, {memory_mb} MiB of memory, {storage_mb} MiB of disk,
{gpus} GPU(s), and a wall limit of {agent_timeout_sec:.0f} seconds.

Network is available exactly as it is for every other agent on this benchmark;
installing what you need is part of the task.

## What you are scored on

The official result is binary: a hidden pytest suite runs in a separate verifier
container after your script exits, and the task counts as solved only when every
test passes. You cannot see that suite and neither could the reference agents.

During the search you are ranked on your own estimate. Before finishing, write
`{internal_validation}` beside your solution:

    {{"metric_name": "self_validation", "direction": "maximize",
     "estimate": <0.0-1.0>, "protocol": "<how you checked>"}}

Compute it from checks you ran yourself. Overstating it wastes the one official
evaluation this task gets, so an honest low estimate is worth more than a
confident wrong one.
"""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(payload: Any) -> str:
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True,
                                     separators=(",", ":")).encode()).hexdigest()


def directory_sha256(root: Path) -> str:
    """Content digest of a directory tree, by relative path."""
    root = Path(root)
    manifest = {}
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        if path.is_symlink():
            manifest[relative] = "symlink-forbidden"
        elif path.is_file():
            manifest[relative] = file_sha256(path)
    return canonical(manifest)


def read_task_config(task_dir: Path) -> dict:
    """Parse an upstream Terminal-Bench ``task.toml``."""
    path = Path(task_dir) / "task.toml"
    if not path.is_file():
        raise FileNotFoundError(f"{task_dir} is not a Terminal-Bench task: no task.toml")
    return tomllib.loads(path.read_text(encoding="utf-8"))


def _pinned_image(section: dict, what: str, task_dir: Path) -> str:
    image = section.get("docker_image")
    if not isinstance(image, str) or "@sha256:" not in image:
        raise ValueError(
            f"{Path(task_dir).name}: the {what} image is not pinned to a digest. "
            "Use a published dataset snapshot; an unpinned task would have to be built locally "
            "and could not be reproduced."
        )
    return image


def task_facts(task_dir: Path) -> dict:
    """Everything the controller needs about a task, without running it."""
    task_dir = Path(task_dir).resolve()
    config = read_task_config(task_dir)
    environment = config.get("environment") or {}
    verifier = config.get("verifier") or {}
    verifier_environment = verifier.get("environment") or environment
    agent = config.get("agent") or {}
    metadata = config.get("metadata") or {}
    name = (config.get("task") or {}).get("name") or task_dir.name
    task_id = name.split("/")[-1]
    mode = verifier.get("environment_mode", "separate")
    if mode != "separate":
        raise ValueError(f"{task_id}: unsupported verifier environment_mode {mode!r}")
    return {
        "task_id": task_id,
        "upstream_name": name,
        "schema_version": str(config.get("schema_version", "")),
        "category": metadata.get("category"),
        "subcategory": metadata.get("subcategory"),
        "expert_time_estimate_hours": metadata.get("expert_time_estimate_hours"),
        "artifacts": list(config.get("artifacts") or []),
        "environment_image": _pinned_image(environment, "environment", task_dir),
        "verifier_image": _pinned_image(verifier_environment, "verifier", task_dir),
        "cpus": int(environment.get("cpus", 1)),
        "memory_mb": int(environment.get("memory_mb", 1024)),
        "storage_mb": int(environment.get("storage_mb", 1024)),
        "gpus": int(environment.get("gpus", 0) or 0),
        "verifier_cpus": int(verifier_environment.get("cpus", environment.get("cpus", 1))),
        "verifier_memory_mb": int(verifier_environment.get("memory_mb", environment.get("memory_mb", 1024))),
        "agent_timeout_sec": float(agent.get("timeout_sec", 0.0)),
        "verifier_timeout_sec": float(verifier.get("timeout_sec", 0.0)),
        "build_timeout_sec": float(environment.get("build_timeout_sec", 0.0)),
        "mcp_servers": list(environment.get("mcp_servers") or []),
        "upstream_task_sha256": directory_sha256(task_dir),
    }


def prepare(*, source: Path, destination: Path, diagnostic_signal: bool = False) -> dict:
    """Materialize one sealed task package from an upstream task directory.

    The package deliberately carries no environment assets and no tests: both are
    already inside the digest-pinned images the task declares, so copying them
    would duplicate gigabytes and create a second source of truth. What is sealed
    here is the instruction the candidate sees, the starter, and the references
    that bind the upstream task and both images.
    """
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if destination.exists():
        raise FileExistsError("destination must be a new directory")
    facts = task_facts(source)
    instruction_path = source / "instruction.md"
    if not instruction_path.is_file():
        raise FileNotFoundError(f"{facts['task_id']}: no instruction.md")
    instruction = instruction_path.read_text(encoding="utf-8")

    (destination / "problem").mkdir(parents=True)
    (destination / "starter").mkdir()
    (destination / "evaluation").mkdir()
    (destination / "problem/instruction.md").write_text(instruction)
    (destination / "problem/README.md").write_text(README.format(
        task_id=facts["task_id"], instruction=instruction.strip(), solution=SOLUTION_NAME,
        image=facts["environment_image"].split("@")[0], cpus=facts["cpus"],
        memory_mb=facts["memory_mb"], storage_mb=facts["storage_mb"], gpus=facts["gpus"],
        agent_timeout_sec=facts["agent_timeout_sec"], internal_validation=INTERNAL_VALIDATION_NAME))
    (destination / "starter" / SOLUTION_NAME).write_text(STARTER)
    (destination / "evaluation/task_ref.json").write_text(json.dumps({
        "upstream_task_dir": str(source), "upstream_task_sha256": facts["upstream_task_sha256"],
        "environment_image": facts["environment_image"], "verifier_image": facts["verifier_image"],
        "verifier_timeout_sec": facts["verifier_timeout_sec"],
        "reward_file": REWARD_NAME, "ctrf_file": CTRF_NAME}, indent=1, sort_keys=True) + "\n")

    search_signal = "official_hidden_suite_fraction" if diagnostic_signal else "candidate_self_validation"
    (destination / "metadata.json").write_text(json.dumps({
        "task_id": facts["task_id"], "benchmark": BENCHMARK, "search_metric": METRIC,
        "official_metric": {"name": "resolved", "direction": "maximize", "role": "official_binary"},
        "search_signal": search_signal,
        "leaderboard_comparable": not diagnostic_signal,
        "hidden_suite_visible_to_candidate": False,
        "resources": {k: facts[k] for k in ("cpus", "memory_mb", "storage_mb", "gpus",
                                            "verifier_cpus", "verifier_memory_mb")},
        "timeouts": {k: facts[k] for k in ("agent_timeout_sec", "verifier_timeout_sec", "build_timeout_sec")},
        "upstream": {k: facts[k] for k in ("upstream_name", "schema_version", "category", "subcategory",
                                           "expert_time_estimate_hours", "artifacts", "mcp_servers")},
    }, indent=1, sort_keys=True) + "\n")

    visible = ["problem/instruction.md", "problem/README.md", f"starter/{SOLUTION_NAME}"]
    hidden = ["evaluation/task_ref.json"]
    files = {}
    for relative in visible + hidden + ["metadata.json"]:
        path = destination / relative
        files[relative] = {"bytes": path.stat().st_size, "sha256": file_sha256(path)}
    body = {"schema_version": SCHEMA, "task_id": facts["task_id"], "metric": dict(METRIC),
            "benchmark": BENCHMARK, "search_signal": search_signal,
            "environment_image": facts["environment_image"], "verifier_image": facts["verifier_image"],
            "upstream_task_sha256": facts["upstream_task_sha256"],
            "candidate_visible": sorted(visible), "controller_only": sorted(hidden), "files": files}
    record = {**body, "custody_sha256": canonical(body)}
    (destination / "custody.json").write_text(json.dumps(record, indent=1, sort_keys=True) + "\n")
    return record


def verify_assets(custody_path: Path) -> dict:
    """Re-check the seal and every bound file before the task is used for anything."""
    custody_path = Path(custody_path)
    record = json.loads(custody_path.read_text())
    sealed = dict(record)
    digest = sealed.pop("custody_sha256", None)
    if record.get("schema_version") != SCHEMA or canonical(sealed) != digest:
        raise ValueError("Terminal-Bench custody schema or seal mismatch")
    if record.get("metric") != METRIC:
        raise ValueError("Terminal-Bench custody does not carry the declared search metric")
    root = custody_path.parent
    for relative, expected in record["files"].items():
        path = root / relative
        if not path.is_file() or path.is_symlink() or file_sha256(path) != expected["sha256"]:
            raise ValueError(f"Terminal-Bench asset changed: {relative}")
    return record


def materialize_trial(*, package: Path, candidate: Path, destination: Path) -> dict:
    """Build the local task directory harbor runs, with the candidate as the solution.

    The environment and the verifier stay upstream's, referenced by their pinned
    digests, so the candidate is the only thing that differs from an oracle run.
    """
    package, candidate, destination = Path(package).resolve(), Path(candidate).resolve(), Path(destination).resolve()
    record = verify_assets(package / "custody.json")
    reference = json.loads((package / "evaluation/task_ref.json").read_text())
    upstream = Path(reference["upstream_task_dir"])
    if directory_sha256(upstream) != reference["upstream_task_sha256"]:
        raise ValueError("upstream Terminal-Bench task directory changed since the package was sealed")
    entry = candidate / SOLUTION_NAME
    if not entry.is_file() or entry.is_symlink():
        raise ValueError(f"candidate has no regular {SOLUTION_NAME}")
    if destination.exists():
        raise FileExistsError("trial destination must be a new directory")
    shutil.copytree(upstream, destination, symlinks=False,
                    ignore=shutil.ignore_patterns("solution", "__pycache__", "*.pyc"))
    solution = destination / "solution"
    solution.mkdir()
    for path in sorted(candidate.rglob("*")):
        if path.is_symlink():
            raise ValueError("candidate solutions may not contain symlinks")
        if path.is_file():
            target = solution / path.relative_to(candidate)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(path, target)
    (solution / SOLUTION_NAME).chmod(0o755)
    return {"task_id": record["task_id"], "trial_dir": str(destination),
            "solution_sha256": directory_sha256(solution),
            "environment_image": record["environment_image"], "verifier_image": record["verifier_image"]}


def read_reward(logs_dir: Path) -> int:
    """The official binary result the verifier wrote. Absent means the trial did not finish."""
    path = Path(logs_dir) / REWARD_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {REWARD_NAME} under {logs_dir}: the verifier did not complete")
    text = path.read_text().strip()
    if text not in {"0", "1"}:
        raise ValueError(f"{REWARD_NAME} is {text!r}; Terminal-Bench rewards are 0 or 1")
    return int(text)


def read_ctrf(logs_dir: Path) -> dict:
    """Per-test results, for diagnostics only.

    This is derived from the hidden suite. It never becomes a search signal in a
    leaderboard-comparable package, and callers that surface it must say so.
    """
    path = Path(logs_dir) / CTRF_NAME
    if not path.is_file():
        raise FileNotFoundError(f"no {CTRF_NAME} under {logs_dir}")
    report = json.loads(path.read_text())
    summary = ((report.get("results") or {}).get("summary")) or {}
    tests = ((report.get("results") or {}).get("tests")) or []
    passed = int(summary.get("passed", sum(1 for t in tests if t.get("status") == "passed")))
    total = int(summary.get("tests", len(tests)))
    return {"passed": passed, "total": total,
            "fraction": (passed / total) if total else 0.0,
            "failed_names": [t.get("name") for t in tests if t.get("status") not in {"passed", "skipped"}],
            "is_diagnostic_only": True}


def read_internal_validation(candidate: Path) -> dict | None:
    """The candidate's own estimate, or None when absent or malformed."""
    import math
    path = Path(candidate) / INTERNAL_VALIDATION_NAME
    if not path.is_file():
        return None
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    estimate = raw.get("estimate") if isinstance(raw, dict) else None
    if (not isinstance(raw, dict) or raw.get("direction") not in {"maximize", "minimize"}
            or type(estimate) not in (int, float) or not math.isfinite(estimate)):
        return None
    return {"metric_name": str(raw.get("metric_name", METRIC["name"])), "direction": raw["direction"],
            "estimate": float(estimate), "protocol": str(raw.get("protocol", ""))[:500]}


def campaign_resources(task_dirs) -> dict:
    """Aggregate what a set of tasks needs, so a plan is arithmetic rather than hope."""
    facts = [task_facts(Path(d)) for d in task_dirs]
    images = {f["environment_image"] for f in facts} | {f["verifier_image"] for f in facts}
    return {
        "tasks": len(facts),
        "distinct_images": len(images),
        "cpus_if_all_concurrent": sum(f["cpus"] for f in facts),
        "memory_gib_if_all_concurrent": round(sum(f["memory_mb"] for f in facts) / 1024, 1),
        "declared_disk_gib_if_all_concurrent": round(sum(f["storage_mb"] for f in facts) / 1024, 1),
        "gpu_tasks": sorted(f["task_id"] for f in facts if f["gpus"]),
        "max_task_cpus": max((f["cpus"] for f in facts), default=0),
        "max_task_memory_gib": round(max((f["memory_mb"] for f in facts), default=0) / 1024, 1),
        "max_declared_disk_gib": round(max((f["storage_mb"] for f in facts), default=0) / 1024, 1),
        "agent_timeout_hours": sorted({round(f["agent_timeout_sec"] / 3600, 2) for f in facts}),
        "per_task": {f["task_id"]: {k: f[k] for k in ("cpus", "memory_mb", "storage_mb", "gpus",
                                                      "agent_timeout_sec", "category")} for f in facts},
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    build = commands.add_parser("prepare", help="seal one task package from an upstream task dir")
    build.add_argument("--source", type=Path, required=True)
    build.add_argument("--destination", type=Path, required=True)
    build.add_argument("--diagnostic-signal", action="store_true",
                       help="mark the package as using the hidden suite for search; NOT leaderboard-comparable")
    check = commands.add_parser("verify", help="re-check a prepared package against its custody")
    check.add_argument("--custody", type=Path, required=True)
    res = commands.add_parser("resources", help="aggregate the declared resources of upstream tasks")
    res.add_argument("--tasks-root", type=Path, required=True)
    res.add_argument("--result", type=Path)
    trial = commands.add_parser("materialize", help="build the harbor local task dir for one candidate")
    trial.add_argument("--package", type=Path, required=True)
    trial.add_argument("--candidate", type=Path, required=True)
    trial.add_argument("--destination", type=Path, required=True)
    grade = commands.add_parser("read-result", help="read a finished trial's official reward")
    grade.add_argument("--logs-dir", type=Path, required=True)
    grade.add_argument("--result", type=Path)
    grade.add_argument("--with-diagnostics", action="store_true")
    args = parser.parse_args(argv)

    if args.command == "prepare":
        record = prepare(source=args.source, destination=args.destination,
                         diagnostic_signal=args.diagnostic_signal)
        print(json.dumps({"task_id": record["task_id"], "custody_sha256": record["custody_sha256"],
                          "search_signal": record["search_signal"], "model_calls": 0}))
        return 0
    if args.command == "verify":
        record = verify_assets(args.custody)
        print(json.dumps({"task_id": record["task_id"], "verified_files": len(record["files"])}))
        return 0
    if args.command == "resources":
        roots = sorted(p for p in Path(args.tasks_root).iterdir() if (p / "task.toml").is_file())
        summary = campaign_resources(roots)
        text = json.dumps(summary, indent=1, sort_keys=True) + "\n"
        if args.result:
            args.result.write_text(text)
        print(json.dumps({k: v for k, v in summary.items() if k != "per_task"}, indent=1, sort_keys=True))
        return 0
    if args.command == "materialize":
        print(json.dumps(materialize_trial(package=args.package, candidate=args.candidate,
                                           destination=args.destination), sort_keys=True))
        return 0
    payload: dict[str, Any] = {"resolved": read_reward(args.logs_dir)}
    if args.with_diagnostics:
        payload["diagnostics"] = read_ctrf(args.logs_dir)
    text = json.dumps(payload, indent=1, sort_keys=True) + "\n"
    if args.result:
        args.result.write_text(text)
    print(text.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
