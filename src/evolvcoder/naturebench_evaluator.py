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
import time
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
    execution_valid: bool = False, candidate_only: bool = True,
    details: dict[str, Any] | None = None,
) -> int:
    return _write(path, {
        **({"details": details} if details is not None else {}),
        "schema_version": "external-domain-result-v1",
        "candidate_saw_hidden_labels": False,
        "official_gold_access": False,
        "score_bearing_evaluation_attempted": score_bearing_attempted,
        "execution_valid": execution_valid,
        "valid": False,
        "public_diagnostic": {
            "category": category,
            "summary": summary.replace("\n", " ")[-2000:],
            "candidate_only": candidate_only,
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


# An official evaluator that cannot run in our runtime says so in these words. A
# candidate's output never produces them: its CSV or JSON is data, not code. They
# mean every candidate will be judged invalid no matter what it writes, so the
# failure is the controller's, is never retried, and must not be charged to the
# candidate. s43588-025-00878-7 lost a session to "No module named 'pyarrow'"
# (the ground-truth pickles need it) and s42256-024-00795-w to np.trapz, which
# numpy 2.4 removed.
_GRADER_ENVIRONMENT_FAILURE = re.compile(
    r"No module named|ModuleNotFoundError|cannot import name|ImportError"
    r"|module '(?:numpy|pandas|scipy|sklearn|torch)[\w.]*' has no attribute"
    r"|was removed in the NumPy|DLL load failed|undefined symbol|GLIBC_[0-9]"
)


_EVALUATOR_NOISE = re.compile(r"OpenBLAS warning|precompiled NUM_THREADS|OPENBLAS_NUM_THREADS|"
                              r"FutureWarning|DeprecationWarning|UserWarning|warnings\.warn|^\s*$")


def _evaluator_failure_text(completed: subprocess.CompletedProcess) -> str:
    """What a failed official evaluator actually said.

    Many evaluators print their error to stdout and exit non-zero, while stderr
    carries only library warnings. s41592-023-01955-3's whole diagnostic was one
    OpenBLAS thread warning, and the real error was never shown.
    """
    parts = []
    for label, text in (("stdout", completed.stdout), ("stderr", completed.stderr)):
        lines = [line for line in (text or "").splitlines() if not _EVALUATOR_NOISE.search(line)]
        if lines:
            parts.append(label + ": " + "\n".join(lines[-40:]))
    return " | ".join(parts) or f"returncode {completed.returncode}"


EXECUTION_LOG_HEAD = 2500
EXECUTION_LOG_TAIL = 2500


def _trim_log(stdout: str | None, stderr: str | None) -> str | None:
    """The candidate's own full-run output, head and tail, warnings removed.

    This is what lets the next step diagnose before it edits: per-epoch validation,
    per-instance error, timings. MLEvolve feeds exactly this (first and last 2,500
    characters) to every improve and debug step; our runs discarded it on success.
    It is candidate-produced from public data only.
    """
    parts = []
    for label, text in (("", stdout), ("[stderr]\n", stderr)):
        lines = [line for line in (text or "").splitlines() if not _EVALUATOR_NOISE.search(line)]
        if lines:
            parts.append(label + "\n".join(lines))
    text = "\n".join(parts)
    if not text:
        return None
    if len(text) <= EXECUTION_LOG_HEAD + EXECUTION_LOG_TAIL + 100:
        return text
    omitted = len(text) - EXECUTION_LOG_HEAD - EXECUTION_LOG_TAIL
    return text[:EXECUTION_LOG_HEAD] + f"\n... [{omitted} characters omitted] ...\n" + text[-EXECUTION_LOG_TAIL:]


def _grader_environment_failure(text: str) -> bool:
    return bool(_GRADER_ENVIRONMENT_FAILURE.search(text or ""))


def _row_errors(row: dict[str, Any]) -> list[str]:
    """Errors an official evaluator recorded inside one instance's row.

    Several graders keep going after a sub-part fails and record the reason
    under keys such as ``small_error`` while the primary metric becomes None.
    Reporting only "cannot parse numeric reference from None" hid the actual
    cause from both the planner and the operator.
    """
    # Only text counts. Metric names can end in "_error" too (relative_l2_error,
    # mean_absolute_error), and their values are numbers, not failures.
    return [f"{key}: {value}" for key, value in sorted(row.items())
            if (key == "error" or key.endswith("_error")) and isinstance(value, str) and value.strip()]


_GRADER_READINESS_SCRIPT = r"""
import ast, importlib.util, json, pathlib, re, sys
evaluation = pathlib.Path(sys.argv[1])
max_bytes = int(sys.argv[2])
sources = sorted(evaluation.rglob("*.py"))
local = {p.stem for p in sources}
missing_modules, missing_attributes, unreadable = set(), set(), []
conditional_missing = set()

def guarded(node, parents):
    return any(isinstance(parent, ast.Try) for parent in parents)

def is_main_check(test):
    return (isinstance(test, ast.Compare) and isinstance(test.left, ast.Name)
            and test.left.id == "__name__")

def conditional(parents):
    # Under an ordinary `if` the import runs only for some tasks or inputs (MLE's
    # shared evaluator imports per-task helpers that way); `if __name__ == "__main__"`
    # always runs when the grader is executed, so it stays a hard requirement.
    return any(isinstance(parent, ast.If) and not is_main_check(parent.test) for parent in parents)

for path in sources:
    text = path.read_text(errors="replace")
    try:
        tree = ast.parse(text)
    except SyntaxError:
        continue
    stack = [(tree, ())]
    while stack:
        node, parents = stack.pop()
        if isinstance(node, (ast.Import, ast.ImportFrom)) and not guarded(node, parents):
            names = [a.name for a in node.names] if isinstance(node, ast.Import) else (
                [node.module] if node.module and not node.level else [])
            for name in names:
                root = name.split(".")[0]
                if root not in local and importlib.util.find_spec(root) is None:
                    (conditional_missing if conditional(parents) else missing_modules).add(root)
        for child in ast.iter_child_nodes(node):
            stack.append((child, parents + (node,)))
    for alias, module in (("np", "numpy"), ("numpy", "numpy"), ("pd", "pandas"), ("pandas", "pandas")):
        if module in missing_modules:
            continue
        names = set(re.findall(r"(?<![\w.])" + alias + r"\.([A-Za-z_]\w*)", text))
        if not names:
            continue
        try:
            imported = __import__(module)
        except Exception:
            continue
        for name in names:
            if not hasattr(imported, name):
                missing_attributes.add(module + "." + name)

readers = {".pkl": "read_pickle", ".pickle": "read_pickle", ".parquet": "read_parquet", ".feather": "read_feather"}

def renamed_module_load(path):
    # Libraries renamed public modules to private ones (sklearn 0.22: sklearn.svm.classes ->
    # sklearn.svm._classes); official evaluators load such ground truth with a compat unpickler
    # (NB 00916). Only a module that is missing now and exists under the "_" name is remapped.
    import pickle, importlib.util
    class Compat(pickle.Unpickler):
        def find_class(self, module, name):
            head, _, tail = module.rpartition(".")
            if head and importlib.util.find_spec(module) is None:
                alt = head + "._" + tail
                if importlib.util.find_spec(alt) is not None:
                    module = alt
            return super().find_class(module, name)
    with open(path, "rb") as f:
        return Compat(f).load()

data = [p for p in evaluation.rglob("*") if p.is_file() and p.suffix.lower() in readers]
if data:
    try:
        import pandas
    except Exception as exc:
        unreadable.append({"path": "*", "error": "pandas: " + type(exc).__name__ + ": " + str(exc)[:200]})
        data = []
for path in sorted(data):
    if path.stat().st_size > max_bytes:
        continue
    try:
        getattr(pandas, readers[path.suffix.lower()])(path)
    except Exception as exc:
        try:
            if not (isinstance(exc, ModuleNotFoundError) and readers[path.suffix.lower()] == "read_pickle"):
                raise exc
            renamed_module_load(path)
        except Exception:
            unreadable.append({"path": str(path.relative_to(evaluation)),
                               "error": type(exc).__name__ + ": " + str(exc)[:200]})
print(json.dumps({"missing_modules": sorted(missing_modules),
                  "conditional_missing_modules": sorted(conditional_missing - missing_modules),
                  "missing_attributes": sorted(missing_attributes),
                  "unreadable_ground_truth": unreadable}))
"""


def grader_readiness(task_root: Path, runtime_python: Path, *, timeout_seconds: int = 900,
                     max_file_bytes: int = 2 << 30) -> dict[str, Any]:
    """Can the official evaluator run at all in the runtime that will score this task?

    Zero model calls and no candidate. It checks, inside the scoring runtime, that
    the evaluator's unguarded imports resolve (an import under an ordinary ``if``
    is only a warning), that every ``np.x``/``pd.x`` it
    names exists in the installed versions, and that its tabular ground-truth
    files can be read. Each of these failed on a paid session on 2026-09-22 and
    made every candidate of that session invalid regardless of what it wrote.
    It reports only module names, attribute names, file paths and exception
    types; no ground-truth content leaves the subprocess.
    """
    evaluation = Path(task_root) / "evaluation"
    if not evaluation.is_dir():
        return {"ready": False, "problems": ["evaluation directory is missing"]}
    if not (evaluation / "evaluator.py").is_file():
        # A partial task tree can carry an empty ground_truth/ and no evaluator at
        # all; with nothing to import, every other check would pass vacuously.
        return {"ready": False, "problems": ["evaluation/evaluator.py is missing"]}
    completed = subprocess.run(
        (str(runtime_python), "-c", _GRADER_READINESS_SCRIPT, str(evaluation), str(max_file_bytes)),
        capture_output=True, text=True, timeout=timeout_seconds, check=False,
    )
    if completed.returncode != 0:
        return {"ready": False, "problems": ["readiness probe failed: " + completed.stderr[-500:]]}
    report = json.loads(completed.stdout.strip().splitlines()[-1])
    problems = (
        [f"missing module {m}" for m in report["missing_modules"]]
        + [f"missing attribute {a}" for a in report["missing_attributes"]]
        + [f"unreadable ground truth {u['path']}: {u['error']}" for u in report["unreadable_ground_truth"]]
    )
    warnings = [f"missing module {m} (imported only under a condition)"
                for m in report.get("conditional_missing_modules", [])]
    return {"ready": not problems, "problems": problems, "warnings": warnings, **report}


_UNCERTAINTY = re.compile(r"\([^)]*\)|(?:±|\+/-|\+-)\s*[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?\s*%?")
_NUMBER = re.compile(r"[-+]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][-+]?\d+)?")


def _single_reference(value: Any) -> float | None:
    """One paper value, or None when the entry does not state one number.

    s42256-024-00833-7 gives null for one instance and "G14: 3042, G15: 3030, ..."
    (one value per graph) for another. The official pipeline aggregates only the
    instance with a single stated value: Claude Code's recorded submission there
    has per_instance_improvement for ndc_hypergraph_maxcut alone. Reading the
    first number of the per-graph string would have used 14 as the reference.
    Uncertainty such as "3.1 (0.13)" or "28649 ± 31" is not a second value.
    """
    if value is None:
        return None
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        return _number(value)
    stripped = _UNCERTAINTY.sub(" ", str(value))
    if len(_NUMBER.findall(stripped)) != 1:
        return None
    return _number(stripped)


def _references(metadata: dict[str, Any]) -> dict[str, tuple[str, str, float]]:
    output: dict[str, tuple[str, str, float]] = {}
    for entry in metadata.get("performance_entries", []):
        primary = next(metric for metric in entry.get("metrics", []) if metric.get("is_primary"))
        scores = [v for v in (_single_reference(row.get("value")) for row in primary.get("sota_score") or [])
                  if v is not None]
        if not scores:
            continue  # the official aggregate leaves such an instance out
        direction = str(primary["metric_direction"])
        reference = max(scores) if direction == "higher_is_better" else min(scores)
        output[str(entry["dataset_name"])] = (str(primary["name"]), direction, reference)
    if not output:
        raise ValueError("metadata has no instance with a single stated SOTA reference")
    return output


# What a sandbox with a bound /model_cache must see. Search (naturebench_session)
# and terminal share it so they load the same weights. Without the transformers
# variables, importing torchvision (which imports transformers) tries to migrate
# /model_cache/transformers and dies on the read-only bind; without the offline
# flags, every load first waits on network retries in a sandbox with no network.
MODEL_CACHE_ENVIRONMENT = (
    ("TORCH_HOME", "/model_cache"),
    ("HF_HOME", "/model_cache/transformers"),
    ("TRANSFORMERS_CACHE", "/model_cache/transformers"),
    ("HF_HUB_OFFLINE", "1"),
    ("TRANSFORMERS_OFFLINE", "1"),
    ("HF_DATASETS_OFFLINE", "1"),
)


def _candidate_command(
    *, workspace: Path, data_dir: Path, runtime_python: Path,
    require_sandbox: bool, environment: dict[str, str],
    inherit_model_cache: bool = True,
    model_cache: Path | None = None,
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
    # An explicit cache wins over the ambient variable: the MLE terminal runs after
    # the search session has restored the caller's environment, so the variable
    # the search sandbox saw is gone by then.
    if model_cache is not None:
        model_cache = Path(model_cache).resolve()
    else:
        model_cache_value = os.environ.get("EVOLVCODER_MODEL_CACHE") if inherit_model_cache else None
        model_cache = Path(model_cache_value).resolve() if model_cache_value else None
    if model_cache is not None:
        if not model_cache.is_dir():
            raise RuntimeError("EVOLVCODER_MODEL_CACHE is not a directory")
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
        # torchvision reads TORCH_HOME; transformers does not, so its own two
        # variables are set as well. Without them a candidate that
        # asks for distilbert or deberta tries to download inside a sandbox that
        # has no network, and falls back to a bag-of-words model instead.
        sandbox_environment.update(MODEL_CACHE_ENVIRONMENT)
    return command, workspace, sandbox_environment, True


FOLDS_DIRECTORY = "evaluation/folds"


def fold_evaluation_root(task_root: Path, fold: int, destination: Path) -> Path:
    """A view of the task package whose evaluation/answers.csv is fold ``fold``'s.

    The MLE task evaluator finds its answers relative to its own file
    (``Path(__file__).resolve().parent.parent``), so scoring another carve with the
    SAME evaluator means running a copy of it beside that fold's answers. Only the
    top-level evaluation files (the evaluator and the helper modules it imports) and
    metadata.json are copied; the sealed package is never modified.
    """
    task_root = Path(task_root)
    answers = task_root / FOLDS_DIRECTORY / str(fold) / "answers.csv"
    if not answers.is_file():
        raise ValueError(f"repeated carve fold {fold} has no answers.csv")
    evaluation = Path(destination) / "evaluation"
    evaluation.mkdir(parents=True)
    for source in sorted((task_root / "evaluation").iterdir()):
        if source.is_file() and not source.is_symlink() and source.name != "answers.csv":
            shutil.copy2(source, evaluation / source.name)
    shutil.copy2(answers, evaluation / "answers.csv")
    shutil.copy2(task_root / "metadata.json", Path(destination) / "metadata.json")
    return evaluation / "evaluator.py"


def repeated_carve_folds(task_root: Path) -> int:
    """How many carve folds the package holds (1 when it has no evaluation/folds)."""
    folds = Path(task_root) / FOLDS_DIRECTORY
    if not folds.is_dir():
        return 1
    count = 1
    while (folds / str(count) / "answers.csv").is_file() and (folds / str(count) / "data").is_dir():
        count += 1
    return count


def _exception_name(text: str | None) -> str | None:
    """The exception class a failed fold raised, and nothing else it printed.

    Fold k's fit rows include fold 0's held-out labels, so free text from a fold
    execution never reaches the planner; the class name is enough to tell a crash
    on an unseen category from a timeout or a memory error.
    """
    for line in reversed((text or "").splitlines()):
        match = re.match(r"\s*([A-Za-z_][\w.]*(?:Error|Exception|Interrupt|Exit))\b", line)
        if match:
            return match.group(1)[:120]
    return None


def _copy_fold_workspace(workspace: Path, destination: Path) -> None:
    """The candidate as scored, minus fold 0's output, in a scratch workspace."""
    if destination.exists():
        shutil.rmtree(destination)
    shutil.copytree(workspace, destination, symlinks=True,
                    ignore=lambda directory, names: ["output"] if Path(directory) == workspace else [])
    (destination / "output").mkdir()


def _run_extra_fold(
    *, fold: int, folds: int, workspace: Path, task_root: Path, runtime_python: Path,
    environment: dict[str, str], require_sandbox: bool, container_binding: dict[str, Any] | None,
    fold_timeout: float, grader_timeout: float, deadline: float, metadata: dict[str, Any],
) -> dict[str, Any]:
    """Execute and score the same program on one extra carve.

    Returns {"ok": True, ...fold record} or {"ok": False, "category", "summary",
    "candidate_only", "score_bearing", "execution_valid"} mirroring fold 0's failure
    semantics.
    """
    label = f"repeated carve fold {fold} of {folds}"
    data_dir = task_root / FOLDS_DIRECTORY / str(fold) / "data"
    scratch = workspace.parent / f".repeated_carve_{workspace.name}" / str(fold)
    started = time.monotonic()
    keep_scratch = False
    try:
        _copy_fold_workspace(workspace, scratch)
        output_dir = scratch / "output"
        fold_environment = {**environment, "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(output_dir)}
        receipt = None
        # Same timeout as fold 0, cut to what is left of the evaluation deadline; a
        # timeout caused by that cut is an incomplete evaluation, not a slow candidate.
        timeout = max(1, int(min(fold_timeout, deadline - time.monotonic())))
        truncated = timeout < fold_timeout
        try:
            if container_binding is not None:
                from evolvcoder.candidate_execution import execute, validate_binding
                spec, _ = validate_binding(container_binding)
                receipt = execute(container_binding, workspace=scratch, public_data=data_dir,
                                  argv=(spec.python, "run.py"), timeout=timeout)
                if receipt.get("harness_error") or receipt.get("status") in {"infrastructure_error", "cancelled"}:
                    keep_scratch = True
                    return {"ok": False, "category": "evaluation_bridge", "candidate_only": False,
                            "summary": f"{label}: candidate backend failed: " + str(receipt.get("record_dir")),
                            "score_bearing": True, "execution_valid": False}
                if receipt.get("timed_out"):
                    raise subprocess.TimeoutExpired("run.py", timeout)
                executed = subprocess.CompletedProcess([spec.python, "run.py"], receipt["returncode"],
                                                       receipt["stdout"], receipt["stderr"])
            else:
                argv, cwd, candidate_environment, _ = _candidate_command(
                    workspace=scratch, data_dir=data_dir, runtime_python=runtime_python,
                    require_sandbox=require_sandbox, environment=fold_environment,
                )
                executed = subprocess.run(argv, cwd=cwd, env=candidate_environment, capture_output=True,
                                          text=True, timeout=timeout, check=False)
        except RuntimeError as exc:
            return {"ok": False, "category": "evaluation_bridge" if container_binding is not None else "candidate_sandbox",
                    "summary": f"{label}: {exc}", "candidate_only": container_binding is None,
                    "score_bearing": True, "execution_valid": False}
        except subprocess.TimeoutExpired:
            if truncated:
                return {"ok": False, "category": "repeated_carve_incomplete", "candidate_only": False,
                        "summary": f"{label} ran out of the evaluation deadline", "incomplete": True,
                        "score_bearing": True, "execution_valid": True}
            return {"ok": False, "category": "candidate_timeout", "candidate_only": True,
                    "summary": f"{label}: run.py timed out", "score_bearing": True, "execution_valid": False}
        execution_seconds = time.monotonic() - started
        if executed.returncode != 0:
            name = _exception_name(executed.stderr) or _exception_name(executed.stdout)
            return {"ok": False, "category": "candidate_execution", "candidate_only": True,
                    "summary": (f"{label}: run.py exited with returncode {executed.returncode}"
                                + (f" ({name})" if name else "")
                                + " after fold 0 succeeded; the program does not generalise to "
                                  "another split of the same public train. Fold output is withheld."),
                    "score_bearing": True, "execution_valid": False}
        log = _trim_log(executed.stdout, executed.stderr)
        try:
            attestation = _attest(output_dir)
        except Exception as exc:
            return {"ok": False, "category": "candidate_output", "candidate_only": True,
                    "summary": f"{label}: {exc}", "score_bearing": True, "execution_valid": False}
        with tempfile.TemporaryDirectory(prefix="naturebench-fold-eval-") as temporary:
            evaluator = fold_evaluation_root(task_root, fold, Path(temporary) / "view")
            work = Path(temporary) / "work"
            work.mkdir()
            left = deadline - time.monotonic()
            if left < 1:
                return {"ok": False, "category": "repeated_carve_incomplete", "candidate_only": False,
                        "summary": f"{label} executed but no evaluation time is left to grade it",
                        "incomplete": True, "score_bearing": True, "execution_valid": True}
            grading_timeout = max(1, int(min(grader_timeout, left)))
            truncated = grading_timeout < grader_timeout
            try:
                official = subprocess.run(
                    [str(runtime_python.absolute()), str(evaluator)], cwd=work,
                    env=fold_environment, capture_output=True, text=True,
                    timeout=grading_timeout, check=False,
                )
            except subprocess.TimeoutExpired:
                if truncated:
                    return {"ok": False, "category": "repeated_carve_incomplete", "candidate_only": False,
                            "summary": f"{label} ran out of the evaluation deadline while grading",
                            "incomplete": True, "score_bearing": True, "execution_valid": True}
                return {"ok": False, "category": "official_evaluation", "candidate_only": True,
                        "summary": f"{label}: official evaluator timed out",
                        "score_bearing": True, "execution_valid": True}
            if official.returncode != 0:
                message = _evaluator_failure_text(official)
                grader = _grader_environment_failure(message)
                return {"ok": False, "category": "grader_environment" if grader else "official_evaluation",
                        "summary": f"{label}: {message}", "candidate_only": not grader,
                        "score_bearing": True, "execution_valid": True}
            raw = json.loads((work / "score.json").read_text())
        if _attest(output_dir) != attestation:
            raise ValueError(f"{label}: output changed during official evaluation")
        from evolvcoder.task_metrics import public_metric
        native, utility = public_metric(raw, metadata)
        return {"ok": True, "fold": fold, "native": native, "utility": utility, "raw_scores": raw,
                "wall_seconds": time.monotonic() - started, "execution_seconds": execution_seconds,
                "submission_output_attestation": attestation,
                "execution_log_sha256": hashlib.sha256((log or "").encode()).hexdigest(),
                "execution_log_chars": len(log or ""),
                **({"candidate_runtime_receipt": receipt} if receipt is not None else {})}
    finally:
        if not keep_scratch:
            shutil.rmtree(scratch, ignore_errors=True)
            try:
                scratch.parent.rmdir()
            except OSError:
                pass


def _repeated_carve(
    *, args: argparse.Namespace, folds: int, workspace: Path, task_root: Path,
    environment: dict[str, str], container_binding: dict[str, Any] | None,
    metadata: dict[str, Any], fold0: tuple[float, float, dict[str, Any], dict[str, Any]],
    started: float,
) -> dict[str, Any]:
    """Folds 1..K-1 after fold 0 was executed (or reused) and scored.

    Every fold must complete: a failure is the candidate's failure, and a deadline
    that cannot fit the next fold ends the evaluation as repeated_carve_incomplete
    rather than averaging fewer folds.

    With --score-existing-output, fold 0 keeps reusing the Coder's attested
    execution (run.py hash and output tree verified above), and only folds 1..K-1
    execute here. Re-running fold 0 instead would spend one more execution and
    could score a different (nondeterministic) output than the one the Coder
    validated and the terminal step will re-fit; the reused fold 0 is exactly the
    single-carve score, so K=1 and fold 0 of K>1 always agree.
    """
    native0, utility0, raw0, attestation0 = fold0
    fold_timeout = args.carve_fold_timeout_seconds or args.timeout_seconds
    deadline = (started + args.carve_deadline_seconds
                if args.carve_deadline_seconds is not None else math.inf)
    records: list[dict[str, Any]] = [{
        "fold": 0, "native": native0, "utility": utility0, "raw_scores": raw0,
        "submission_output_attestation": attestation0,
        "execution_reused": bool(args.score_existing_output),
        "wall_seconds": time.monotonic() - started,
    }]

    def details(completed: int) -> dict[str, Any]:
        return {"folds": folds, "completed_folds": completed,
                "fold_execution_timeout_seconds": fold_timeout,
                "deadline_seconds": args.carve_deadline_seconds,
                "fold0_execution_reused": bool(args.score_existing_output),
                "fold_records": records}

    for fold in range(1, folds):
        remaining = deadline - time.monotonic()
        # The slowest fold so far estimates the next one. Fold 0's own execution
        # time is unknown when it was reused, so only fresh folds inform it.
        observed = max((r["wall_seconds"] for r in records[1:]), default=0.0)
        if remaining < 1 or remaining < observed:
            return {"ok": False, "category": "repeated_carve_incomplete", "candidate_only": False,
                    "execution_valid": True,
                    "summary": f"repeated_carve_incomplete: {len(records)}/{folds} folds completed; "
                               f"{max(0.0, remaining):.0f}s left of the evaluation deadline cannot fit the next fold",
                    "details": details(len(records))}
        result = _run_extra_fold(
            fold=fold, folds=folds, workspace=workspace, task_root=task_root,
            runtime_python=args.runtime_python, environment=environment,
            require_sandbox=args.require_candidate_sandbox, container_binding=container_binding,
            fold_timeout=float(fold_timeout), grader_timeout=float(args.timeout_seconds),
            deadline=deadline, metadata=metadata,
        )
        if not result.pop("ok"):
            summary = result["summary"]
            if result.get("incomplete"):
                summary = f"repeated_carve_incomplete: {len(records)}/{folds} folds completed; {summary}"
            return {**result, "ok": False, "summary": summary, "details": details(len(records))}
        records.append(result)
    natives = [r["native"] for r in records]
    utilities = [r["utility"] for r in records]
    mean_native = sum(natives) / folds
    mean_utility = sum(utilities) / folds
    std = math.sqrt(sum((v - mean_native) ** 2 for v in natives) / folds)
    return {"ok": True, "native": mean_native, "utility": mean_utility,
            "details": {**details(folds), "aggregation": "mean_over_folds",
                        "fold_native_scores": natives, "fold_utilities": utilities,
                        "native_mean": mean_native, "native_std": std}}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--task-root", type=Path, required=True)
    parser.add_argument("--runtime-python", type=Path, required=True)
    parser.add_argument("--timeout-seconds", type=int, default=1800)
    parser.add_argument("--candidate-runtime-json", help="Explicit controller-owned candidate backend binding; no credentials")
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
    parser.add_argument(
        "--carve-folds", type=int, default=None,
        help=(
            "Repeated carve (MLE packages with evaluation/folds only): after fold 0, re-execute "
            "the same program on folds 1..K-1 and report the mean public metric."
        ),
    )
    parser.add_argument("--carve-fold-timeout-seconds", type=int, default=None,
                        help="Execution timeout of each extra fold; defaults to --timeout-seconds")
    parser.add_argument("--carve-deadline-seconds", type=float, default=None,
                        help="Wall budget of this whole bridge run, fold 0 included")
    args = parser.parse_args()
    bridge_started = time.monotonic()
    container_binding = json.loads(args.candidate_runtime_json) if args.candidate_runtime_json else None
    runtime_receipt = None
    if container_binding is not None:
        from evolvcoder.candidate_execution import validate_binding
        validate_binding(container_binding)
    if args.execution_only and args.score_existing_output:
        raise SystemExit("--execution-only and --score-existing-output are mutually exclusive")
    carve_folds = args.carve_folds if args.carve_folds is not None and args.carve_folds > 1 else None
    if carve_folds is not None and args.execution_only:
        raise SystemExit("--carve-folds scores a candidate; it cannot be combined with --execution-only")

    workspace, task_root = args.workspace.resolve(), args.task_root.resolve()
    run_py = workspace / "run.py"
    data_dir = task_root / "problem/data"
    evaluator = task_root / "evaluation/evaluator.py"
    if not run_py.is_file():
        return _invalid(args.result, "candidate_artifact", "run.py is absent")
    if not data_dir.is_dir() or not evaluator.is_file():
        raise SystemExit("frozen task package lacks public data or official evaluator")
    if carve_folds is not None:
        available = repeated_carve_folds(task_root)
        carve_metadata = json.loads((task_root / "metadata.json").read_text())
        if available < carve_folds or "search_metric" not in carve_metadata:
            return _invalid(args.result, "repeated_carve_configuration",
                            f"--carve-folds {carve_folds} but the task package holds {available} "
                            "carve fold(s)" + ("" if "search_metric" in carve_metadata
                                               else " and no public search metric"),
                            candidate_only=False)
    output_dir = workspace / "output"
    # The candidate never inherits the controller's egress route. bwrap already
    # unshares the network, so this is the second of two independent reasons a
    # candidate cannot reach the internet, and it keeps that true on the
    # unsandboxed fallback path as well.
    proxy_names = {"HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "ALL_PROXY",
                   "http_proxy", "https_proxy", "no_proxy", "all_proxy"}
    environment = {
        key: value for key, value in os.environ.items()
        if not key.endswith("_API_KEY") and not key.endswith("_TOKEN")
        and key not in {"EVAL_SERVICE_URL"} and key not in proxy_names
    }
    environment.update({
        "DATA_DIR": str(data_dir), "OUTPUT_DIR": str(output_dir),
        "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
    })
    reused_execution_evidence_sha256 = None
    candidate_execution_log = None
    if args.score_existing_output:
        execution_evidence_path = workspace / ".evolvcoder/execution_validation.json"
        try:
            evidence_bytes = execution_evidence_path.read_bytes()
            execution_evidence = json.loads(evidence_bytes)
            details = execution_evidence["details"]
            if details.get("candidate_runtime_binding") != container_binding:
                raise ValueError("prior execution used a different candidate backend")
            if container_binding is not None:
                from evolvcoder.candidate_execution import validate_binding, verify_receipt
                spec, _ = validate_binding(container_binding)
                # The scored artifact is a retained copy of the Coder's temporary
                # workspace: bind the command and public data here; run.py and
                # the output tree are bound by content below, not by host path.
                runtime_receipt = verify_receipt(container_binding, details.get("candidate_runtime_receipt"),
                                                 argv=[spec.python, "run.py"], public_data=data_dir)
                if runtime_receipt.get("returncode") != 0 or runtime_receipt.get("status") != "candidate_exited":
                    raise ValueError("prior container execution did not complete successfully")
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
            candidate_execution_log = details.get("candidate_execution_log")
        except Exception as exc:
            return _invalid(args.result, "execution_evidence", str(exc))
    else:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        output_dir.mkdir(parents=True)
        try:
            if container_binding is not None:
                from evolvcoder.candidate_execution import execute, validate_binding
                spec, _ = validate_binding(container_binding)
                runtime_receipt = execute(container_binding, workspace=workspace, public_data=data_dir,
                                          argv=(spec.python, "run.py"), timeout=args.timeout_seconds)
                sandboxed = True
                if runtime_receipt.get("harness_error") or runtime_receipt.get("status") in {"infrastructure_error", "cancelled"}:
                    return _invalid(args.result, "evaluation_bridge", "candidate backend failed: " + runtime_receipt["record_dir"], candidate_only=False)
                if runtime_receipt.get("timed_out"):
                    return _invalid(args.result, "candidate_timeout", "run.py timed out")
                executed = subprocess.CompletedProcess([spec.python, "run.py"], runtime_receipt["returncode"], runtime_receipt["stdout"], runtime_receipt["stderr"])
            else:
                candidate_argv, candidate_cwd, candidate_environment, sandboxed = _candidate_command(
                    workspace=workspace, data_dir=data_dir,
                    runtime_python=args.runtime_python,
                    require_sandbox=args.require_candidate_sandbox,
                    environment=environment,
                )
        except RuntimeError as exc:
            return _invalid(args.result, "evaluation_bridge" if container_binding is not None else "candidate_sandbox",
                            str(exc), candidate_only=container_binding is None)
        try:
            if container_binding is None:
                executed = subprocess.run(
                    # Preserve the venv launcher; resolving it loses pyvenv.cfg.
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
        candidate_execution_log = _trim_log(executed.stdout, executed.stderr)
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
                "candidate_execution_log": candidate_execution_log,
                "candidate_sandboxed": sandboxed,
                **({"candidate_runtime_binding": container_binding, "candidate_runtime_receipt": runtime_receipt}
                   if container_binding is not None else {}),
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
                message = _evaluator_failure_text(official)
                environment = _grader_environment_failure(message)
                return _invalid(
                    args.result, "grader_environment" if environment else "official_evaluation",
                    message,
                    score_bearing_attempted=True,
                    execution_valid=True,
                    candidate_only=not environment,
                )
            raw = json.loads((Path(temporary) / "score.json").read_text())
        if _attest(output_dir) != before:
            raise ValueError("output changed during official evaluation")
        metadata = json.loads((task_root / "metadata.json").read_text())
        if "search_metric" in metadata:
            from evolvcoder.task_metrics import public_metric
            native, aggregate = public_metric(raw, metadata)
            carve_details: dict[str, Any] | None = None
            if carve_folds is not None:
                outcome = _repeated_carve(
                    args=args, folds=carve_folds, workspace=workspace, task_root=task_root,
                    environment=environment, container_binding=container_binding,
                    metadata=metadata, fold0=(native, aggregate, raw, before),
                    started=bridge_started,
                )
                if not outcome["ok"]:
                    return _invalid(
                        args.result, outcome["category"], outcome["summary"],
                        score_bearing_attempted=True, execution_valid=outcome["execution_valid"],
                        candidate_only=outcome["candidate_only"],
                        details={"repeated_carve": outcome["details"],
                                 "candidate_execution_log": candidate_execution_log},
                    )
                carve_details = outcome["details"]
                native, aggregate = outcome["native"], outcome["utility"]
            return _write(args.result, {
                "schema_version": "external-domain-result-v1", "valid": True,
                "execution_valid": True, "score_bearing_evaluation_attempted": True,
                "candidate_saw_hidden_labels": False, "official_gold_access": False,
                "metrics": {"aggregate_improvement": aggregate, "search_utility": aggregate,
                            metadata['search_metric']['name']: native},
                "details": {"metric": metadata['search_metric'], "native_score": native,
                    "aggregate_improvement_semantics": "internal_maximization_utility_alias_not_SOTA_improvement",
                    "evaluation_mode": "public_validation", "primary_scores": {"public_validation": native},
                    "raw_scores": raw, "submission_output_attestation": before,
                    "candidate_executable_sha256": hashlib.sha256(run_py.read_bytes()).hexdigest(),
                    "candidate_execution_reused": args.score_existing_output,
                    "reused_execution_evidence_sha256": reused_execution_evidence_sha256,
                    "candidate_execution_log": candidate_execution_log,
                    "candidate_sandboxed": sandboxed,
                    **({"repeated_carve": carve_details} if carve_details is not None else {}),
                    **({"candidate_runtime_binding": container_binding, "candidate_runtime_receipt": runtime_receipt}
                       if container_binding is not None else {})},
            })
        references = _references(metadata)
        improvements: dict[str, float] = {}
        primary_scores: dict[str, float] = {}
        for instance, (metric, direction, reference) in references.items():
            row = raw.get(instance)
            if not isinstance(row, dict):
                raise ValueError(f"{instance}: official evaluator returned no result row")
            errors = _row_errors(row)
            if row.get(metric) is None:
                # One level of nesting with a single item, as 00833-7 reports its one
                # NDC hypergraph: {"NDC": {"cut_size": 28987}}. The official aggregate
                # scored exactly that value; several items stay ambiguous and refused.
                nested = [v for v in row.values() if isinstance(v, dict) and metric in v]
                if len(nested) == 1:
                    row = {**row, metric: nested[0][metric]}
            if errors or row.get(metric) is None:
                detail = "; ".join(errors) if errors else f"primary metric {metric!r} is missing"
                raise ValueError(f"{instance}: official evaluator returned an error: {detail}"[:1500])
            value = row.get(metric)
            if isinstance(value, float) and not math.isfinite(value):
                # The candidate's score, not the paper reference. Constant
                # predictions make a rank correlation or R-squared NaN, and the
                # old message blamed the reference.
                raise ValueError(f"{instance}: primary metric {metric!r} is {value}; "
                                 "constant or degenerate predictions produce this")
            score = _number(value)
            sign = 1.0 if direction == "higher_is_better" else -1.0
            improvements[instance] = sign * (score - reference) / max(abs(reference), 1e-12)
            primary_scores[instance] = score
        aggregate = sum(improvements.values()) / len(improvements)
    except Exception as exc:
        environment = _grader_environment_failure(str(exc))
        return _invalid(
            args.result, "grader_environment" if environment else "official_evaluation", str(exc),
            score_bearing_attempted=official_started,
            execution_valid=True,
            candidate_only=not environment,
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
            "candidate_execution_log": candidate_execution_log,
            "evaluation_mode": (
                "official_evaluator_reusing_attested_execution"
                if args.score_existing_output
                else "official_evaluator_local_development"
            ),
            "candidate_sandboxed": sandboxed,
            **({"candidate_runtime_binding": container_binding, "candidate_runtime_receipt": runtime_receipt}
               if container_binding is not None else {}),
        },
    })


if __name__ == "__main__":
    raise SystemExit(main())
