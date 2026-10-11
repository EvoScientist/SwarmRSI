"""Log-loss tasks tell the coder about tail risk; other metrics do not."""
import json

import pytest

from evolvcoder import naturebench_session as session


def task(tmp_path, name):
    root = tmp_path / name.replace(' ', '_')
    (root / 'problem/data').mkdir(parents=True)
    (root / 'problem/data/train.csv').write_text('id,y\n1,0\n')
    (root / 'metadata.json').write_text(json.dumps({'search_metric': {'name': name, 'direction': 'minimize'}}))
    return root


@pytest.mark.parametrize('name', ['multi_class_log_loss', 'log_loss', 'Log Loss', 'binary cross-entropy'])
def test_log_loss_metrics_are_detected(tmp_path, name):
    assert session.log_loss_metric_name(task(tmp_path, name)) == name


@pytest.mark.parametrize('name', ['auc_roc', 'rmse', 'quadratic_weighted_kappa'])
def test_other_metrics_are_not(tmp_path, name):
    assert session.log_loss_metric_name(task(tmp_path, name)) is None


def test_missing_metadata_is_not(tmp_path):
    assert session.log_loss_metric_name(tmp_path / 'missing') is None


def test_task_context_carries_the_section_only_for_log_loss(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'runtime_dependency_inventory', lambda: (['numpy'], []))
    monkeypatch.setattr(session, 'accelerator_inventory', lambda: {})
    with_loss = session.public_task_context(task(tmp_path, 'multi_class_log_loss'))
    without = session.public_task_context(task(tmp_path, 'auc_roc'))
    assert '## Log-loss tail risk' in with_loss and '1e-5' in with_loss
    assert '## Log-loss tail risk' not in without
