"""Opt-in data-file suffixes are ignored like output/; code and internal_validation.json never are."""
from types import SimpleNamespace

import pytest

from evolvcoder import coder_engine, naturebench_session as session

ENGINE = next(v for v in vars(coder_engine).values() if isinstance(v, type) and hasattr(v, '_ignored'))
BASE = (".git/**", ".evolvcoder/**", "output/**", "**/__pycache__/**", ".pytest_cache/**", "*.pyc")


def ignored(patterns, relative):
    return ENGINE._ignored(SimpleNamespace(config=SimpleNamespace(ignored_workspace_patterns=patterns)), relative)


def test_default_adds_nothing():
    assert session.scratch_suffix_patterns({}) == ()
    assert not ignored(BASE, 'feats.npz')


def test_opted_in_suffixes_ignore_data_files_at_any_depth():
    patterns = BASE + session.scratch_suffix_patterns({"scratch_output_suffixes": [".npz", ".pkl"]})
    for relative in ('feats.npz', 'cache/emb.npz', 'model.pkl'):
        assert ignored(patterns, relative)
    for relative in ('run.py', 'src/model.py', 'internal_validation.json', 'features.csv'):
        assert not ignored(patterns, relative)


@pytest.mark.parametrize('bad', ['.py', '.PY', '.json', 'npz', '*.npz', 'a/.npz', 3])
def test_code_and_malformed_suffixes_are_refused(bad):
    with pytest.raises(ValueError):
        session.scratch_suffix_patterns({"scratch_output_suffixes": [bad]})


def _task(tmp_path):
    root = tmp_path / 'task'
    (root / 'problem/data').mkdir(parents=True, exist_ok=True)
    (root / 'problem/data/train.csv').write_text('id,y\n1,0\n')
    return root


def test_task_context_tells_the_coder_scratch_files_are_regenerated(tmp_path, monkeypatch):
    monkeypatch.setattr(session, 'runtime_dependency_inventory', lambda: (['numpy'], []))
    monkeypatch.setattr(session, 'accelerator_inventory', lambda: {})
    plain = session.public_task_context(_task(tmp_path))
    noted = session.public_task_context(_task(tmp_path), scratch_suffixes=('.npz', '.pkl'))
    assert '## Scratch files' not in plain
    assert '## Scratch files' in noted and '.npz, .pkl' in noted and 'regenerate' in noted
