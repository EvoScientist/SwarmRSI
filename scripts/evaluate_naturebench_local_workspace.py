#!/usr/bin/env python3
"""Gold-isolated local bridge for complete frozen NatureBench task packages."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any


def _write(path: Path, payload: dict[str, Any]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)
    return 0


def _invalid(
    path: Path, category: str, summary: str, *, score_bearing_attempted: bool = False,
    execution_valid: bool = False,
) -> int:
    return _write(path, {
        "schema_version": "external-domain-result-v1",
        "candidate_saw_hidden_labels": False,
        "official_gold_access": False,
        "score_bearing_evaluation_attempted": score_bearing_attempted,
        "execution_valid": execution_valid,
        "valid": False,
        "public_diagnostic": {
            "category": category,
            "summary": summary.replace("\n", " ")[-2000:],
            "candidate_only": True,
            "hidden_data_included": False,
        },
    })


def _attest(root: Path) -> dict[str, Any]:
    paths = tuple(root.rglob("*")) if root.is_dir() else ()
    if any(path.is_symlink() for path in paths):
        raise ValueError("output tree contains a symlink")
    files = sorted(path for path in paths if path.is_file())
    if not files:
        raise ValueError("output tree contains no files")
    digest = hashlib.sha256()
    total = 0
    for path in files:
        content = path.read_bytes()
        total += len(content)
        digest.update(path.relative_to(root).as_posix().encode())
        digest.update(b"\0")
        digest.update(content)
        digest.update(b"\0")
    return {
        "schema_version": "submission-output-attestation-v1",
        "root": "output", "artifact_sha256": digest.hexdigest(),
        "file_count": len(files), "total_bytes": total,
    }


def _number(value: Any) -> float:
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        match = re.search(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?", str(value))
        if match is None:
            raise ValueError(f"cannot parse numeric reference from {value!r}")
        result = float(match.group())
    if not math.isfinite(result):
        raise ValueError("reference is non-finite")
    return result


def _references(metadata: dict[str, Any]) -> dict[str, tuple[str, str, float]]:
    output: dict[str, tuple[str, str, float]] = {}
    for entry in metadata.get("performance_entries", []):
        primary = next(metric for metric in entry.get("metrics", []) if metric.get("is_primary"))
        scores = [_number(row["value"]) for row in primary.get("sota_score") or []]
        if not scores:
            raise ValueError(f"{entry.get('dataset_name')}: no primary SOTA reference")
        direction = str(primary["metric_direction"])
        reference = max(scores) if direction == "higher_is_better" else min(scores)
        output[str(entry["dataset_name"])] = (str(primary["name"]), direction, reference)
    if not output:
        raise ValueError("metadata has no performance entries")
    return output


def _candidate_command(
    *, workspace: Path, data_dir: Path, runtime_python: Path,
    require_sandbox: bool, environment: dict[str, str],
) -> tuple[list[str], Path, dict[str, str], bool]:
    bwrap = shutil.which("bwrap")
    if bwrap is None:
        if require_sandbox:
            raise RuntimeError("bubblewrap is required but unavailable")
        return [str(runtime_python.absolute()), "run.py"], workspace, environment, False
    command = [
        bwrap, "--die-with-parent", "--unshare-net", "--unshare-pid", "--new-session",
    ]
    for source in (Path("/usr"), Path("/bin"), Path("/lib"), Path("/lib64"), Path("/etc")):
        if source.exists():
            command.extend(("--ro-bind", str(source), str(source)))
    if Path("/sys").is_dir():
        command.extend(("--ro-bind", "/sys", "/sys"))
    runtime_root = runtime_python.parent.parent.absolute()
    command.extend((
        "--ro-bind", str(runtime_root), "/runtime",
        "--ro-bind", str(data_dir), "/public_data",
        "--bind", str(workspace), "/workspace",
        "--proc", "/proc", "--dev", "/dev", "--tmpfs", "/tmp",
    ))
    model_cache_value = os.environ.get("SWARMRSI_MODEL_CACHE")
    model_cache = Path(model_cache_value).resolve() if model_cache_value else None
    if model_cache is not None:
        if not model_cache.is_dir():
            raise RuntimeError("SWARMRSI_MODEL_CACHE is not a directory")
        command.extend(("--ro-bind", str(model_cache), "/model_cache"))
    if Path("/proc/driver/nvidia").is_dir():
        command.extend((
            "--ro-bind", "/proc/driver/nvidia", "/proc/driver/nvidia",
        ))
    for device in sorted(Path("/dev").glob("nvidia*")):
        command.extend(("--dev-bind", str(device), str(device)))
    command.extend((
        "--chdir", "/workspace", "--", "/runtime/bin/python", "run.py",
    ))
    sandbox_environment = dict(environment)
    sandbox_environment.update({
        "DATA_DIR": "/public_data", "OUTPUT_DIR": "/workspace/output",
        "TMPDIR": "/tmp",
    })
    if model_cache is not None:
        sandbox_environment["TORCH_HOME"] = "/model_cache"
    return command, workspace, sandbox_environment, True


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument(
        "--execution-only", action="store_true",
        help="Run and attest the candidate output without invoking the official scientific evaluator.",
    )
    parser.add_argument(
        "--score-existing-output", action="store_true",
        help=(
            "Score output retained from a prior execution-only gate after verifying its "
            "sandbox, executable hash, and output attestation; do not execute run.py again."
        ),
    )
    parser.add_argument(
        "--require-candidate-sandbox", action="store_true",
        help="Fail closed unless the candidate runs in a filesystem- and network-isolated bubblewrap sandbox.",
    )
    args = parser.parse_args()
    if args.execution_only and args.score_existing_output:
        raise SystemExit("--execution-only and --score-existing-output are mutually exclusive")

    workspace, task_root = args.workspace.resolve(), args.task_root.resolve()
    run_py = workspace / "run.py"
    data_dir = task_root / "problem/data"
    evaluator = task_root / "evaluation/evaluator.py"
    if not run_py.is_file():
        return _invalid(args.result, "candidate_artifact", "run.py is absent")
    if not data_dir.is_dir() or not evaluator.is_file():
        raise SystemExit("frozen task package lacks public data or official evaluator")
    output_dir = workspace / "output"
    environment = {
        key: value for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and not key.endswith("_TOKEN")
        and key not in {"EVAL_SERVICE_URL"}
    }
    environment.update({
        "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(output_dir),
        "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
    })
    reused_execution_evidence_sha256 = None
    if args.score_existing_output:
        execution_evidence_path = workspace / ".swarmrsi/execution_validation.json"
        try:
            evidence_bytes = execution_evidence_path.read_bytes()
            execution_evidence = json.loads(evidence_bytes)
            details = execution_evidence["details"]
            if (
                execution_evidence.get("schema_version") != "candidate-execution-result-v1"
                or execution_evidence.get("valid") is not True
                or execution_evidence.get("execution_valid") is not True
                or execution_evidence.get("score_bearing_evaluation_attempted") is not False
            ):
                raise ValueError("prior execution evidence is not an accepted execution-only result")
            sandboxed = details.get("candidate_sandboxed") is True
            if args.require_candidate_sandbox and not sandboxed:
                raise ValueError("prior candidate execution was not sandboxed")
            executable_sha256 = hashlib.sha256(run_py.read_bytes()).hexdigest()
            if details.get("candidate_executable_sha256") != executable_sha256:
                raise ValueError("run.py does not match prior execution evidence")
            before = _attest(output_dir)
            if details.get("submission_output_attestation") != before:
                raise ValueError("output does not match prior execution evidence")
            reused_execution_evidence_sha256 = hashlib.sha256(evidence_bytes).hexdigest()
        except Exception as exc:
            return _invalid(args.result, "execution_evidence", str(exc))
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            candidate_argv, candidate_cwd, candidate_environment, sandboxed = _candidate_command(
                workspace=workspace, data_dir=data_dir,
                runtime_python=args.runtime_python,
                require_sandbox=args.require_candidate_sandbox,
                environment=environment,
            )
        except RuntimeError as exc:
            return _invalid(args.result, "candidate_sandbox", str(exc))
        try:
            executed = subprocess.run(
                # Keep the venv entrypoint path intact. Resolving its interpreter
                # symlink discards pyvenv.cfg discovery and can silently run the
                # base Python without the benchmark dependencies.
                candidate_argv, cwd=candidate_cwd,
                env=candidate_environment, capture_output=True, text=True,
                timeout=args.timeout_seconds, check=False,
            )
        except subprocess.TimeoutExpired:
            return _invalid(args.result, "candidate_timeout", "run.py timed out")
        if executed.returncode != 0:
            return _invalid(
                args.result, "candidate_execution",
                executed.stderr or executed.stdout or f"returncode {executed.returncode}",
            )
        try:
            before = _attest(output_dir)
        except Exception as exc:
            return _invalid(args.result, "candidate_output", str(exc))
    if args.execution_only:
        return _write(args.result, {
            "schema_version": "candidate-execution-result-v1",
            "candidate_saw_hidden_labels": False,
            "official_gold_access": False,
            "score_bearing_evaluation_attempted": False,
            "valid": True,
            "execution_valid": True,
            "details": {
                "submission_output_attestation": before,
                "candidate_executable_sha256": hashlib.sha256(run_py.read_bytes()).hexdigest(),
                "evaluation_mode": "public_execution_only",
                "candidate_sandboxed": sandboxed,
            },
        })
    official_started = False
    try:
        with tempfile.TemporaryDirectory(prefix="naturebench-local-eval-") as temporary:
            official_started = True
            official = subprocess.run(
                [str(args.runtime_python.absolute()), str(evaluator)], cwd=temporary,
                env=environment, capture_output=True, text=True,
                timeout=args.timeout_seconds, check=False,
            )
            if official.returncode != 0:
                return _invalid(
                    args.result, "official_evaluation",
                    official.stderr or official.stdout or f"returncode {official.returncode}",
                    score_bearing_attempted=True,
                    execution_valid=True,
                )
            raw = json.loads((Path(temporary) / "score.json").read_text())
        if _attest(output_dir) != before:
            raise ValueError("output changed during official evaluation")
        references = _references(json.loads((task_root / "metadata.json").read_text()))
        improvements: dict[str, float] = {}
        primary_scores: dict[str, float] = {}
        for instance, (metric, direction, reference) in references.items():
            row = raw.get(instance)
            if not isinstance(row, dict) or row.get("error"):
                raise ValueError(f"{instance}: official evaluator returned an error")
            score = _number(row.get(metric))
            sign = 1.0 if direction == "higher_is_better" else -1.0
            improvements[instance] = sign * (score - reference) / max(abs(reference), 1e-12)
            primary_scores[instance] = score
        aggregate = sum(improvements.values()) / len(improvements)
    except Exception as exc:
        return _invalid(
            args.result, "official_evaluation", str(exc),
            score_bearing_attempted=official_started,
            execution_valid=True,
        )
    return _write(args.result, {
        "schema_version": "external-domain-result-v1",
        "candidate_saw_hidden_labels": False, "official_gold_access": False,
        "score_bearing_evaluation_attempted": True,
        "execution_valid": True,
        "valid": True, "metrics": {"aggregate_improvement": aggregate},
        "details": {
            "primary_scores": primary_scores, "raw_scores": raw,
            "per_instance_improvement": improvements,
            "submission_output_attestation": before,
            "candidate_executable_sha256": hashlib.sha256(run_py.read_bytes()).hexdigest(),
            "candidate_execution_reused": args.score_existing_output,
            "reused_execution_evidence_sha256": reused_execution_evidence_sha256,
            "evaluation_mode": (
                "official_evaluator_reusing_attested_execution"
                if args.score_existing_output
                else "official_evaluator_local_development"
            ),
            "candidate_sandboxed": sandboxed,
        },
    })


if __name__ == "__main__":
    raise SystemExit(main())
