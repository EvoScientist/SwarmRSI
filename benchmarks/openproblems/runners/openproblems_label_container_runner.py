#!/usr/bin/env python3
"""Run one label-projection candidate without exposing test labels."""

from __future__ import annotations

import argparse
import importlib.util
import random
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd


class LabelContractError(ValueError):
    pass


def _load_solution(path: Path):
    spec = importlib.util.spec_from_file_location("evocoder_label_solution", path)
    if spec is None or spec.loader is None:
        raise LabelContractError("cannot import solution.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    candidate = getattr(module, "tuso_model", None)
    if not callable(candidate):
        raise LabelContractError(
            "solution.py must define tuso_model(train_adata, test_adata)"
        )
    return candidate


def extract_prediction(result, expected_names) -> np.ndarray:
    if isinstance(result, ad.AnnData):
        if not result.obs_names.equals(expected_names):
            raise LabelContractError("candidate changed or reordered test obs_names")
        if "label_pred" not in result.obs:
            raise LabelContractError("candidate omitted obs['label_pred']")
        values = result.obs["label_pred"].to_numpy()
    else:
        values = np.asarray(result)
    if values.ndim != 1:
        raise LabelContractError("label prediction must be one-dimensional")
    if len(values) != len(expected_names):
        raise LabelContractError(
            f"label prediction length mismatch: {len(values)} != {len(expected_names)}"
        )
    if bool(pd.isna(values).any()):
        raise LabelContractError("label prediction contains missing values")
    strings = values.astype(str)
    if np.any(np.char.str_len(strings) == 0):
        raise LabelContractError("label prediction contains empty labels")
    return strings


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--solution", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    parser.add_argument("--test", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    train = ad.read_h5ad(args.train)
    test = ad.read_h5ad(args.test)
    if "label" not in train.obs:
        raise LabelContractError("training input is missing obs['label']")
    if "label" in test.obs:
        raise LabelContractError("candidate-visible test input contains obs['label']")
    expected_names = test.obs_names.copy()
    random.seed(args.seed)
    np.random.seed(args.seed)
    result = _load_solution(args.solution)(train, test)
    prediction = extract_prediction(result, expected_names)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        prediction=prediction,
        obs_names=np.asarray(expected_names.astype(str), dtype=str),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
