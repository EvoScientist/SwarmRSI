"""SIIM carves by patient: no patient on both sides of the public split."""
import importlib.util
from pathlib import Path

import pandas as pd

MODULE = Path(__file__).resolve().parents[1] / 'deploy/mle/make_task_root.py'


def builder():
    spec = importlib.util.spec_from_file_location('subject_builder', MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def tables(public, n=60):
    public.mkdir(parents=True)
    pd.DataFrame({'image_name': [f'ISIC_{i:04d}' for i in range(n)], 'patient_id': [f'P{i // 3}' for i in range(n)],
                  'target': [i % 2 for i in range(n)]}).to_csv(public / 'train.csv', index=False)
    pd.DataFrame({'image_name': ['ISIC_9000', 'ISIC_9001'], 'patient_id': ['Q1', 'Q2']}).to_csv(public / 'test.csv', index=False)
    pd.DataFrame({'image_name': ['ISIC_9000', 'ISIC_9001'], 'target': [0, 0]}).to_csv(public / 'sample_submission.csv', index=False)


def test_siim_splits_by_patient(tmp_path):
    public = tmp_path / 'siim-isic-melanoma-classification/prepared/public'
    tables(public)
    result = builder().carve_split(public.parent, tmp_path / 'out', seed=0, holdout=.2)
    assert result['grouped_on'] == 'patient_id'
    fit = pd.read_csv(tmp_path / 'out/problem/data/train.csv')
    held = pd.read_csv(public / 'train.csv').set_index('image_name').loc[
        pd.read_csv(tmp_path / 'out/problem/data/test.csv').image_name]
    assert not set(fit.patient_id) & set(held.patient_id)


def test_other_competitions_keep_the_example_split(tmp_path):
    public = tmp_path / 'some-other-competition/prepared/public'
    tables(public)
    assert builder().carve_split(public.parent, tmp_path / 'out', seed=0, holdout=.2)['grouped_on'] is None
