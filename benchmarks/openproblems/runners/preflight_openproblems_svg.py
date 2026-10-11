#!/usr/bin/env python3
"""Preflight and replay the TusoAI OpenProblems SVG panel.

This adapter deliberately distinguishes the evaluator released with TusoAI
from the current OpenProblems SVG leaderboard metric.  It makes no model API
calls.  Candidate code is trusted code and should be sandboxed by the caller
for formal experiments.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import random
import subprocess
import sys
import urllib.request
from importlib.metadata import version
from pathlib import Path
from typing import Any, Callable


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MANIFEST = (
    ROOT
    / "config"
    / "benchmark_manifests"
    / "openproblems_svg_tusoai_replay_v1.json"
)


class SVGContractError(ValueError):
    """Raised when data or a candidate violates the frozen SVG contract."""


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_manifest(path: Path = DEFAULT_MANIFEST) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _git_head(path: Path) -> str | None:
    if not (path / ".git").exists():
        return None
    result = subprocess.run(
        ["git", "-C", str(path), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        text=True,
    )
    return result.stdout.strip() if result.returncode == 0 else None


def _require_scientific_stack():
    try:
        import anndata as ad
        import numpy as np
        import pandas as pd
        import sklearn
    except ImportError as exc:  # pragma: no cover - environment dependent
        raise SVGContractError(
            "SVG replay requires the singlecell environment "
            "(anndata, numpy, pandas, scikit-learn)"
        ) from exc
    return ad, np, pd, sklearn


def validate_simulated_dataset(adata: Any) -> list[str]:
    """Validate the exact TusoAI-visible historical data shape."""
    errors: list[str] = []
    if "counts" not in adata.layers:
        errors.append("missing layers['counts']")
    if "spatial" not in adata.obsm:
        errors.append("missing obsm['spatial']")
    elif tuple(adata.obsm["spatial"].shape) != (adata.n_obs, 2):
        errors.append("obsm['spatial'] must have shape (n_obs, 2)")
    if "annotation" not in adata.obs:
        errors.append("missing obs['annotation'] required by released TusoAI function")
    if "true_spatial_var_score" not in adata.var:
        errors.append("missing var['true_spatial_var_score'] for sealed evaluation")
    if "orig_feature_name" not in adata.var:
        errors.append("missing var['orig_feature_name'] for OpenProblems correlation")
    if not adata.var_names.is_unique:
        errors.append("var_names are not unique")
    return errors


def make_candidate_view(adata: Any, protocol: str = "sealed") -> Any:
    """Return the candidate-visible object while retaining truth externally."""
    if protocol not in {"sealed", "released"}:
        raise SVGContractError(f"unknown protocol: {protocol}")
    view = adata.copy()
    view.X = view.layers["counts"].copy()
    if protocol == "sealed":
        del view.var["true_spatial_var_score"]
    return view


def _coerce_released_labels(values: Any, name: str):
    """Reproduce the cast-before-validation behavior in TusoAI's evaluator."""
    _, np, pd, _ = _require_scientific_stack()
    series = values if isinstance(values, pd.Series) else pd.Series(values)
    if series.isna().any():
        raise SVGContractError(f"{name} contains NaN")
    labels = series.astype(int).to_numpy()
    if not set(np.unique(labels)).issubset({0, 1}):
        raise SVGContractError(f"{name} is not binary after the released integer cast")
    return labels


def evaluate_released_tusoai(
    true_spatial_var_score: Any,
    pred_spatial_var_score: Any,
) -> dict[str, float]:
    """Exactly reproduce ``examples/svg/svg_initial.py:evaluate``."""
    _, np, pd, _ = _require_scientific_stack()
    from sklearn.metrics import f1_score, roc_auc_score

    truth = (
        true_spatial_var_score
        if isinstance(true_spatial_var_score, pd.Series)
        else pd.Series(true_spatial_var_score)
    )
    pred = (
        pred_spatial_var_score
        if isinstance(pred_spatial_var_score, pd.Series)
        else pd.Series(pred_spatial_var_score)
    )
    if len(truth) != len(pred):
        raise SVGContractError("truth/prediction length mismatch")
    mask = truth.notna() & pred.notna()
    if not bool(mask.any()):
        raise SVGContractError("no valid rows after dropping NaNs")
    y_true = _coerce_released_labels(truth[mask], "truth")
    y_pred = _coerce_released_labels(pred[mask], "prediction")

    f1_macro = float(f1_score(y_true, y_pred, average="macro"))
    f1_micro = float(f1_score(y_true, y_pred, average="micro"))
    auc_macro = float("nan")
    auc_micro = float("nan")
    try:
        y_true_oh = np.column_stack((1 - y_true, y_true))
        y_pred_oh = np.column_stack((1 - y_pred, y_pred))
        auc_macro = float(roc_auc_score(y_true_oh, y_pred_oh, average="macro"))
        auc_micro = float(roc_auc_score(y_true_oh, y_pred_oh, average="micro"))
    except ValueError:
        pass
    components = np.asarray(
        [auc_macro, auc_micro, f1_macro, f1_micro], dtype=float
    )
    return {
        "auc_macro": auc_macro,
        "auc_micro": auc_micro,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "score": float(np.nanmean(components)),
    }


def validate_prediction_artifact(prediction: Any, expected_length: int) -> list[str]:
    """Apply a stricter artifact gate without altering the released score."""
    _, np, _, _ = _require_scientific_stack()
    values = np.asarray(prediction)
    errors: list[str] = []
    if values.ndim != 1:
        errors.append("prediction must be one-dimensional")
        return errors
    if len(values) != expected_length:
        errors.append(
            f"prediction length mismatch: {len(values)} != {expected_length}"
        )
    if not np.isfinite(values).all():
        errors.append("prediction contains non-finite values")
    if not set(np.unique(values)).issubset({0, 1}):
        errors.append("prediction values must be exactly binary before evaluator casting")
    return errors


def evaluate_current_openproblems_correlation(
    orig_feature_name: Any,
    true_spatial_var_score: Any,
    pred_spatial_var_score: Any,
) -> float:
    """Reproduce the current OpenProblems mean per-gene Kendall correlation."""
    _, _, pd, _ = _require_scientific_stack()
    frame = pd.DataFrame(
        {
            "orig_feature_name": list(orig_feature_name),
            "true_spatial_var_score": list(true_spatial_var_score),
            "pred_spatial_var_score": list(pred_spatial_var_score),
        }
    )
    correlations = frame.groupby(
        "orig_feature_name", observed=True
    ).apply(
        lambda group: group["pred_spatial_var_score"].corr(
            group["true_spatial_var_score"], method="kendall"
        ),
        include_groups=False,
    )
    return float(correlations.mean())


def load_trusted_candidate(path: Path) -> Callable[[Any], Any]:
    spec = importlib.util.spec_from_file_location(
        f"evocoder_svg_candidate_{file_sha256(path)[:12]}", path
    )
    if spec is None or spec.loader is None:
        raise SVGContractError(f"cannot import candidate: {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise SVGContractError("candidate must define callable tuso_model(adata)")
    return candidate


def replay_dataset(
    dataset_path: Path,
    candidate_path: Path,
    protocol: str = "sealed",
    seed: int = 42,
) -> dict[str, Any]:
    ad, np, _, sklearn = _require_scientific_stack()
    adata = ad.read_h5ad(dataset_path)
    errors = validate_simulated_dataset(adata)
    if errors:
        raise SVGContractError("; ".join(errors))
    truth = adata.var["true_spatial_var_score"].copy()
    original_feature = adata.var["orig_feature_name"].copy()
    expected_index = adata.var_names.copy()
    candidate_view = make_candidate_view(adata, protocol=protocol)

    random.seed(seed)
    np.random.seed(seed)
    result = load_trusted_candidate(candidate_path)(candidate_view)
    result = candidate_view if result is None else result
    if "pred_spatial_var_score" not in result.var:
        raise SVGContractError("candidate omitted var['pred_spatial_var_score']")
    if not result.var_names.equals(expected_index):
        raise SVGContractError("candidate changed or reordered var_names")
    prediction = result.var["pred_spatial_var_score"]
    artifact_errors = validate_prediction_artifact(prediction, len(truth))
    metric = evaluate_released_tusoai(truth, prediction)
    current_metric = evaluate_current_openproblems_correlation(
        original_feature, truth, prediction
    )
    return {
        "dataset": str(dataset_path.resolve()),
        "dataset_sha256": file_sha256(dataset_path),
        "candidate": str(candidate_path.resolve()),
        "candidate_sha256": file_sha256(candidate_path),
        "protocol": protocol,
        "seed": seed,
        "released_tusoai_metric": metric,
        "current_openproblems_metric": {
            "metric_id": "mean_gene_level_kendall_rank_correlation",
            "score": current_metric,
        },
        "artifact_valid": not artifact_errors,
        "artifact_errors": artifact_errors,
        "prediction_positive_rate": float(np.asarray(prediction).mean()),
        "runtime": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
            "anndata": version("anndata"),
            "networkx": version("networkx"),
        },
        "main_table_eligible": False,
    }


def fetch_dataset(asset: dict[str, Any], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + ".partial")
    urllib.request.urlretrieve(asset["url"], temporary)
    if temporary.stat().st_size != asset["bytes"]:
        raise SVGContractError(f"downloaded byte count mismatch for {asset['dataset_id']}")
    if file_sha256(temporary) != asset["sha256"]:
        raise SVGContractError(f"downloaded SHA-256 mismatch for {asset['dataset_id']}")
    temporary.replace(destination)


def preflight(
    manifest_path: Path,
    data_dir: Path,
    dataset_ids: set[str] | None = None,
    fetch: bool = False,
    require_runtime: bool = False,
) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    blockers: list[str] = []
    warnings: list[str] = [
        "TusoAI-released SVG scoring is not the current OpenProblems "
        "Kendall-correlation metric.",
        "The released input exposes true_spatial_var_score; formal EvoCoder "
        "runs must use protocol=sealed.",
        "The released evaluator integer-casts continuous alpha ground truth; "
        "report its score only as a matched TusoAI replay.",
    ]

    dependency_status: dict[str, bool] = {}
    import_names = {
        "anndata": "anndata",
        "numpy": "numpy",
        "pandas": "pandas",
        "scikit-learn": "sklearn",
        "networkx": "networkx",
    }
    for package in manifest["required_python_packages"]:
        available = importlib.util.find_spec(import_names[package]) is not None
        dependency_status[package] = available
        if not available:
            blockers.append(f"missing Python package: {package}")

    repository_status: dict[str, Any] = {}
    for name, repository in manifest["repositories"].items():
        path = ROOT / repository["path"]
        actual = _git_head(path)
        ok = actual == repository["commit"]
        repository_status[name] = {
            "path": str(path),
            "expected_commit": repository["commit"],
            "actual_commit": actual,
            "ok": ok,
        }
        if not ok:
            blockers.append(f"repository mismatch: {name}")

    source_status: dict[str, Any] = {}
    for name, source in manifest["frozen_sources"].items():
        path = ROOT / source["path"]
        actual = file_sha256(path) if path.is_file() else None
        ok = actual == source["sha256"]
        source_status[name] = {
            "path": str(path),
            "expected_sha256": source["sha256"],
            "actual_sha256": actual,
            "ok": ok,
        }
        if not ok:
            blockers.append(f"frozen source mismatch: {name}")

    adapter = manifest.get("evocoder_development_adapter", {})
    adapter_asset_status: dict[str, Any] = {}
    for name in (
        "starter",
        "task",
        "phase_contract",
        "domain_evaluator",
        "runner_wrapper",
    ):
        asset = adapter.get(name, {})
        path = ROOT / str(asset.get("path", ""))
        expected = str(asset.get("sha256", ""))
        actual = file_sha256(path) if path.is_file() else None
        ok = len(expected) == 64 and actual == expected
        adapter_asset_status[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": ok,
        }
        if not ok:
            blockers.append(f"EvoCoder adapter asset mismatch: {name}")

    terminal = manifest.get("evocoder_terminal_deployment_adapter", {})
    terminal_asset_status: dict[str, Any] = {}
    for name, path_key, digest_key in (
        ("contract", "path", "contract_sha256"),
        ("adapter", "adapter_program", "adapter_sha256"),
    ):
        path = ROOT / str(terminal.get(path_key, ""))
        expected = str(terminal.get(digest_key, ""))
        actual = file_sha256(path) if path.is_file() else None
        ok = len(expected) == 64 and actual == expected
        terminal_asset_status[name] = {
            "path": str(path),
            "expected_sha256": expected,
            "actual_sha256": actual,
            "ok": ok,
        }
        if not ok:
            blockers.append(f"EvoCoder terminal adapter asset mismatch: {name}")
    expected_dependency = str(terminal.get("search_evaluator_sha256", ""))
    search_adapter_path = ROOT / str(
        adapter.get("domain_evaluator", {}).get("path", "")
    )
    search_contract = (
        json.loads(search_adapter_path.read_text(encoding="utf-8"))
        if search_adapter_path.is_file()
        else {}
    )
    search_program = ROOT / str(search_contract.get("adapter_program", ""))
    actual_dependency = (
        file_sha256(search_program) if search_program.is_file() else None
    )
    dependency_ok = (
        len(expected_dependency) == 64
        and actual_dependency == expected_dependency
    )
    terminal_asset_status["search_evaluator_dependency"] = {
        "path": str(search_program),
        "expected_sha256": expected_dependency,
        "actual_sha256": actual_dependency,
        "ok": dependency_ok,
    }
    if not dependency_ok:
        blockers.append(
            "EvoCoder terminal adapter transitive search-evaluator mismatch"
        )

    dataset_roles = {
        str(item["dataset_id"]): str(item["role"])
        for item in manifest["datasets"]
    }
    feedback_ids = set(adapter.get("search_feedback_datasets", []))
    deployment_ids = set(adapter.get("deployment_only_datasets", []))
    isolation_ok = (
        bool(feedback_ids)
        and feedback_ids.isdisjoint(deployment_ids)
        and all(dataset_roles.get(item) == "optimization" for item in feedback_ids)
        and all(dataset_roles.get(item) == "deployment" for item in deployment_ids)
        and feedback_ids | deployment_ids == set(dataset_roles)
    )
    if not isolation_ok:
        blockers.append("EvoCoder SVG search/deployment dataset isolation is invalid")

    runtime_status: dict[str, Any] = {"required": require_runtime, "ok": None}
    if require_runtime and adapter_asset_status.get("domain_evaluator", {}).get("ok"):
        domain_contract = json.loads(
            (ROOT / adapter["domain_evaluator"]["path"]).read_text(encoding="utf-8")
        )
        arguments = domain_contract["arguments"]
        image = arguments[arguments.index("--image") + 1]
        expected_image_id = arguments[arguments.index("--image-id") + 1]
        inspected = subprocess.run(
            ["docker", "image", "inspect", image, "--format", "{{.Id}}"],
            capture_output=True,
            text=True,
            check=False,
        )
        actual_image_id = inspected.stdout.strip() if inspected.returncode == 0 else None
        runtime_imports = sorted(set(import_names.values()))
        import_probe = subprocess.run(
            [
                "docker",
                "run",
                "--rm",
                "--network",
                "none",
                image,
                "python",
                "-c",
                "; ".join(f"import {name}" for name in runtime_imports),
            ],
            capture_output=True,
            text=True,
            check=False,
        ) if actual_image_id == expected_image_id else None
        imports_ok = import_probe is not None and import_probe.returncode == 0
        runtime_status = {
            "required": True,
            "image": image,
            "expected_image_id": expected_image_id,
            "actual_image_id": actual_image_id,
            "required_imports": runtime_imports,
            "imports_ok": imports_ok,
            "import_stderr_tail": (
                import_probe.stderr[-1000:] if import_probe is not None else None
            ),
            "ok": actual_image_id == expected_image_id and imports_ok,
        }
        if runtime_status["ok"] is not True:
            blockers.append(
                "pinned SVG Docker runtime is unavailable, changed, or missing "
                "a required candidate dependency"
            )

    selected = [
        asset
        for asset in manifest["datasets"]
        if dataset_ids is None or asset["dataset_id"] in dataset_ids
    ]
    if dataset_ids is not None:
        unknown = dataset_ids - {asset["dataset_id"] for asset in selected}
        blockers.extend(f"unknown dataset: {dataset_id}" for dataset_id in sorted(unknown))

    dataset_status: dict[str, Any] = {}
    for asset in selected:
        path = data_dir / f"{asset['dataset_id']}.h5ad"
        if fetch and not path.exists():
            fetch_dataset(asset, path)
        actual_bytes = path.stat().st_size if path.is_file() else None
        actual_sha = file_sha256(path) if path.is_file() else None
        integrity_ok = (
            actual_bytes == asset["bytes"] and actual_sha == asset["sha256"]
        )
        contract_errors: list[str] = []
        if integrity_ok:
            try:
                ad, _, _, _ = _require_scientific_stack()
                contract_errors = validate_simulated_dataset(ad.read_h5ad(path))
            except Exception as exc:  # pragma: no cover - dependency/corrupt data
                contract_errors = [f"cannot load dataset: {type(exc).__name__}: {exc}"]
        ok = integrity_ok and not contract_errors
        dataset_status[asset["dataset_id"]] = {
            "path": str(path),
            "expected_bytes": asset["bytes"],
            "actual_bytes": actual_bytes,
            "expected_sha256": asset["sha256"],
            "actual_sha256": actual_sha,
            "integrity_ok": integrity_ok,
            "contract_errors": contract_errors,
            "ok": ok,
        }
        if not ok:
            blockers.append(f"dataset not ready: {asset['dataset_id']}")

    return {
        "schema_version": "evocoder-openproblems-svg-preflight-v1",
        "manifest": str(manifest_path.resolve()),
        "manifest_sha256": file_sha256(manifest_path),
        "python_dependencies": dependency_status,
        "repositories": repository_status,
        "frozen_sources": source_status,
        "evocoder_adapter_assets": adapter_asset_status,
        "evocoder_terminal_adapter_assets": terminal_asset_status,
        "search_deployment_isolation_ok": isolation_ok,
        "runtime": runtime_status,
        "datasets": dataset_status,
        "warnings": warnings,
        "blockers": blockers,
        "launch_ready": not blockers,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--dataset-id", action="append")
    parser.add_argument("--fetch", action="store_true")
    parser.add_argument("--require-runtime", action="store_true")
    parser.add_argument("--candidate", type=Path)
    parser.add_argument("--protocol", choices=("sealed", "released"), default="sealed")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    selected = set(args.dataset_id) if args.dataset_id else None
    payload = preflight(
        args.manifest,
        args.data_dir,
        dataset_ids=selected,
        fetch=args.fetch,
        require_runtime=args.require_runtime,
    )
    if args.candidate and payload["launch_ready"]:
        ids = sorted(payload["datasets"])
        payload["replays"] = [
            replay_dataset(
                args.data_dir / f"{dataset_id}.h5ad",
                args.candidate,
                protocol=args.protocol,
                seed=args.seed,
            )
            for dataset_id in ids
        ]
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    payload["result_sha256"] = hashlib.sha256(canonical.encode()).hexdigest()
    rendered = json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered, encoding="utf-8")
    print(rendered, end="")
    return 0 if payload["launch_ready"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
