import ast
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
import pandas as pd
from PIL import Image
import pytest
from sklearn.model_selection import train_test_split

MODULE = Path(__file__).parents[1] / 'deploy/mle/image_projection.py'
spec = importlib.util.spec_from_file_location('image_projection', MODULE)
images = importlib.util.module_from_spec(spec)
spec.loader.exec_module(images)


def fixture(tmp_path, classes=2, image_size=None):
    prepared = tmp_path / 'prepared'
    public = prepared / 'public'
    for name in ('train', 'test'):
        (public / name).mkdir(parents=True)
    (prepared / 'private').mkdir()
    (public / 'description.md').write_text('Synthetic TIFF competition')
    train = pd.DataFrame({'id': [f'img_{i:03d}' for i in range(30)],
                          'label': [i % classes for i in range(30)]})
    train.to_csv(public / 'train_labels.csv', index=False)
    sample = pd.DataFrame({'id': ['official_000', 'official_001']})
    targets = ['label'] if classes == 2 else [f'class_{c}' for c in range(classes)]
    sample[targets] = .5 if classes == 2 else 1/classes
    sample.to_csv(public / 'sample_submission.csv', index=False)
    for index, identifier in enumerate([*train.id, *sample.id]):
        width, height = image_size or (7, 5 + index % 2)
        array = np.full((height, width, 3), index, dtype=np.uint8)
        Image.fromarray(array).save(public / ('train' if index < 30 else 'test') / (identifier + '.tif'))
    return prepared


def digest_tree(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


@pytest.mark.parametrize('classes', [2, 3])
def test_projection_bytes_ids_visibility_and_determinism(tmp_path, classes):
    source = fixture(tmp_path, classes)
    public = source / 'public'
    # Archives and auxiliary labels are never whitelisted into candidate data.
    (public / 'train.zip').write_bytes(b'full original images and labels')
    (public / 'all_labels.csv').write_text('held secret labels')
    before = digest_tree(source)
    kwargs = {} if classes == 2 else {'classes': (0, 1, 2), 'targets': ('class_0', 'class_1', 'class_2')}
    out = tmp_path / 'out'
    split = images.carve_images(source, out, seed=0, holdout=.2, **kwargs)
    again = tmp_path / 'again'
    assert images.carve_images(source, again, seed=0, holdout=.2, **kwargs) == split
    assert digest_tree(out) == digest_tree(again)
    train = pd.read_csv(public / 'train_labels.csv')
    fit, held = train_test_split(train, test_size=.2, random_state=0, stratify=train.label)
    data = out / 'problem/data'
    assert set(p.name for p in data.iterdir()) == {'train', 'test', 'train_labels.csv', 'sample_submission.csv'}
    assert pd.read_csv(data / 'train_labels.csv').equals(fit.sort_index().reset_index(drop=True))
    assert set(p.stem for p in (data / 'train').iterdir()) == set(fit.id)
    assert set(p.stem for p in (data / 'test').iterdir()) == set(held.id)
    for frame, folder in ((fit, 'train'), (held, 'test')):
        for identifier in frame.id:
            assert (data / folder / (identifier + '.tif')).read_bytes() == (public / 'train' / (identifier + '.tif')).read_bytes()
            with Image.open(data / folder / (identifier + '.tif')) as image:
                image.load()
                assert image.format == 'TIFF'
    answers = pd.read_csv(out / 'evaluation/answers.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    assert sample.id.tolist() == held.sort_index().id.tolist()
    assert np.allclose(sample.iloc[:, 1:], .5 if classes == 2 else 1/classes)
    aligned = images.validate_probabilities(answers.iloc[::-1], answers)
    assert aligned.equals(answers)
    manifest = [json.loads(line) for line in (out / 'evaluation/image_manifest.jsonl').read_text().splitlines()]
    assert len(manifest) == 30 and len({r['decoded_sha256'] for r in manifest}) == 30
    assert not any('sha256' in p.name or 'manifest' in p.name or 'split' in p.name for p in data.rglob('*'))
    assert before == digest_tree(source)
    assert split['fit_rows'] == 24 and split['holdout_rows'] == 6


@pytest.mark.parametrize('bad', ['missing', 'extra', 'duplicate_id', 'unsafe_id', 'symlink_file',
    'symlink_dir', 'wrong_suffix', 'corrupt', 'wrong_format', 'multiframe', 'wrong_table',
    'missing_label', 'wrong_label', 'official_overlap', 'wrong_mode'])
def test_malformed_sources_fail_closed(tmp_path, bad):
    source = fixture(tmp_path)
    public = source / 'public'
    path = public / 'train/img_000.tif'
    train = pd.read_csv(public / 'train_labels.csv')
    if bad == 'missing': path.unlink()
    elif bad == 'extra': (public / 'train/extra.tif').write_bytes(path.read_bytes())
    elif bad == 'duplicate_id': train.loc[1, 'id'] = train.loc[0, 'id']
    elif bad == 'unsafe_id': train.loc[0, 'id'] = '../outside'
    elif bad == 'symlink_file':
        path.unlink(); path.symlink_to(public / 'train/img_001.tif')
    elif bad == 'symlink_dir':
        (public / 'train').rename(public / 'other'); (public / 'train').symlink_to(public / 'other')
    elif bad == 'wrong_suffix': path.rename(path.with_suffix('.png'))
    elif bad == 'corrupt': path.write_bytes(b'not a TIFF')
    elif bad == 'wrong_format': Image.new('RGB', (7, 5)).save(path, format='PNG')
    elif bad == 'wrong_mode': Image.new('L', (7, 5)).save(path)
    elif bad == 'multiframe': Image.new('RGB', (7,5)).save(path, save_all=True, append_images=[Image.new('RGB', (7,5))])
    elif bad == 'wrong_table': train['extra_target'] = train.label
    elif bad == 'missing_label': train.loc[0, 'label'] = np.nan
    elif bad == 'wrong_label': train.loc[0, 'label'] = 4
    elif bad == 'official_overlap':
        sample = pd.read_csv(public / 'sample_submission.csv'); sample.loc[0, 'id'] = train.id.iloc[0]
        sample.to_csv(public / 'sample_submission.csv', index=False)
    train.to_csv(public / 'train_labels.csv', index=False)
    with pytest.raises((ValueError, OSError)):
        images.carve_images(source, tmp_path / 'out', seed=0, holdout=.2)
    assert not (tmp_path / 'out/problem/data').exists()


@pytest.mark.parametrize('same_label', [True, False])
def test_cross_fold_equal_pixels_different_tiff_encoding_fail_closed(tmp_path, same_label):
    source = fixture(tmp_path)
    public = source / 'public'
    train = pd.read_csv(public / 'train_labels.csv')
    fit, held = train_test_split(train, test_size=.2, random_state=0, stratify=train.label)
    a = fit.iloc[0]
    b = held[held.label == a.label].iloc[0] if same_label else held[held.label != a.label].iloc[0]
    with Image.open(public / 'train' / (a.id + '.tif')) as image:
        image.save(public / 'train' / (b.id + '.tif'), compression='tiff_lzw')
    assert (public / 'train' / (a.id + '.tif')).read_bytes() != (public / 'train' / (b.id + '.tif')).read_bytes()
    with pytest.raises(ValueError, match='duplicate decoded'):
        images.carve_images(source, tmp_path / 'out', seed=0, holdout=.2)
    assert not (tmp_path / 'out/problem/data').exists()


@pytest.mark.parametrize('bad', ['missing', 'extra', 'duplicate', 'nan', 'inf', 'range', 'columns'])
def test_prediction_contract(tmp_path, bad):
    expected = pd.DataFrame({'id': ['first', 'second'], 'label': [0, 1]})
    submission = expected.copy()
    submission['label'] = submission.label.astype(float)
    if bad == 'missing': submission = submission.iloc[:1]
    elif bad == 'extra': submission.loc[2] = ['third', .1]
    elif bad == 'duplicate': submission.loc[1, 'id'] = 'first'
    elif bad in ('nan', 'inf', 'range'): submission.loc[0, 'label'] = {'nan': np.nan, 'inf': np.inf, 'range': 1.1}[bad]
    else: submission['unrequested'] = 0
    with pytest.raises(ValueError): images.validate_probabilities(submission, expected)


def test_identical_starter_public_terminal_and_missing_test_rejected(tmp_path):
    assert any(isinstance(node, ast.FunctionDef) and node.name == 'main' for node in ast.parse(images.STARTER).body)
    source = fixture(tmp_path)
    out = tmp_path / 'out'
    images.carve_images(source, out, seed=0, holdout=.2)
    for index, data in enumerate((out / 'problem/data', source / 'public')):
        target = tmp_path / f'predictions{index}'
        env = {**os.environ, 'DATA_DIR': str(data), 'OUTPUT_DIR': str(target),
               'EVOLVCODER_SUBMISSION_FIT': 'full' if index else 'validation'}
        subprocess.run([sys.executable, '-c', images.STARTER], check=True, env=env)
        sample = pd.read_csv(data / 'sample_submission.csv')
        submission = pd.read_csv(target / 'submission.csv')
        assert images.validate_probabilities(submission.iloc[::-1], sample).id.tolist() == sample.id.tolist()
        assert (submission.label == pd.read_csv(data / 'train_labels.csv').label.mean()).all()
    (source / 'public/test/official_000.tif').unlink()
    assert subprocess.run([sys.executable, '-c', images.STARTER], env=env, capture_output=True).returncode != 0


def test_projection_reads_each_source_image_once(tmp_path, monkeypatch):
    source = fixture(tmp_path)
    original = Path.read_bytes
    counts = {}
    def counted(path):
        if path.parent == source / 'public/train':
            counts[path.name] = counts.get(path.name, 0) + 1
        return original(path)
    monkeypatch.setattr(Path, 'read_bytes', counted)
    images.carve_images(source, tmp_path / 'out', seed=0, holdout=.2)
    assert len(counts) == 30 and set(counts.values()) == {1}


def test_exact_string_ids_not_coerced_to_row_numbers(tmp_path):
    source = fixture(tmp_path)
    public = source / 'public'
    train = pd.read_csv(public / 'train_labels.csv')
    for index, identifier in enumerate(train.id):
        replacement = f'{index:040d}'
        (public / 'train' / (identifier + '.tif')).rename(public / 'train' / (replacement + '.tif'))
        train.loc[index, 'id'] = replacement
    train.to_csv(public / 'train_labels.csv', index=False)
    images.carve_images(source, tmp_path / 'out', seed=0, holdout=.2)
    fit = pd.read_csv(tmp_path / 'out/problem/data/train_labels.csv', dtype={'id': str})
    assert all(len(identifier) == 40 for identifier in fit.id)
    assert set(fit.id) <= set(train.id)


def test_builder_actual_main_and_official_style_auc_join(tmp_path, monkeypatch, capsys):
    prepared = fixture(tmp_path, image_size=(96, 96))
    official = tmp_path / 'official'
    (official / 'mlebench').mkdir(parents=True)
    (official / 'UPSTREAM_REVISION').write_text('synthetic-official-revision')
    (official / 'mlebench/__init__.py').write_text('')
    (official / 'mlebench/registry.py').write_text('''from sklearn.metrics import roc_auc_score
class Grader:
    def __call__(self, submission, answers):
        assert set(submission.id) == set(answers.id)
        return roc_auc_score(answers.sort_values('id').label, submission.sort_values('id').label)
class Registry:
    def set_data_dir(self, path): return self
    def get_competition(self, name):
        class Competition: grader = Grader()
        return Competition()
registry = Registry()
''')
    grader = types.SimpleNamespace(name='auc-roc', is_lower_better=lambda _: False,
        rank_score=lambda *_: {'gold_threshold': .9})
    registry = types.SimpleNamespace(set_data_dir=lambda _: types.SimpleNamespace(
        get_competition=lambda _: types.SimpleNamespace(grader=grader)))
    monkeypatch.setitem(sys.modules, 'mlebench', types.ModuleType('mlebench'))
    monkeypatch.setitem(sys.modules, 'mlebench.grade', types.SimpleNamespace(get_leaderboard=lambda _: [1,2]))
    monkeypatch.setitem(sys.modules, 'mlebench.registry', types.SimpleNamespace(registry=registry))
    monkeypatch.setitem(sys.modules, 'image_projection', images)
    builder_spec = importlib.util.spec_from_file_location('image_builder', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    # Real starter/evaluator subprocesses; only pinned external registry is synthetic.
    monkeypatch.setenv('PYTHONPATH', str(official) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    tasks = tmp_path / 'tasks'
    monkeypatch.setattr(sys, 'argv', ['make_task_root.py', 'histopathologic-cancer-detection',
        '--prepared', str(prepared), '--tasks-root', str(tasks), '--cache', str(tmp_path / 'cache'),
        '--official-source', str(official), '--python', sys.executable])
    assert builder.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['split'] == {'fit_rows': 24, 'holdout_rows': 6, 'stratified_on': 'label'}
    assert summary['starter_baseline'] == .5
    root = tasks / 'histopathologic-cancer-detection'
    assert builder.verify_assets(root / 'custody.json')['task'] == 'histopathologic-cancer-detection'
    assert (root / 'evaluation/image_projection.py').is_file()
    assert 'TIFF bytes' in (root / 'problem/README.md').read_text()


def test_explicit_histopath_dimensions_reject_wrong_shape(tmp_path):
    source = fixture(tmp_path)
    with pytest.raises(ValueError, match='dimensions'):
        images.carve_images(source, tmp_path / 'out', seed=0, holdout=.2, expected_size=(96,96))
    assert not (tmp_path / 'out/problem/data').exists()


def test_existing_tabular_projection_unchanged(tmp_path):
    public = tmp_path / 'prepared/public'
    public.mkdir(parents=True)
    pd.DataFrame({'id': range(30), 'text': ['sample'] * 30, 'label': [0, 1] * 15}).to_csv(public / 'train.csv', index=False)
    pd.DataFrame({'id': [100,101], 'text': ['new', 'other']}).to_csv(public / 'test.csv', index=False)
    pd.DataFrame({'id': [100,101], 'label': [0,0]}).to_csv(public / 'sample_submission.csv', index=False)
    builder_spec = importlib.util.spec_from_file_location('tabular_builder', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    result = builder.carve_split(public.parent, tmp_path / 'out', seed=0, holdout=.2)
    assert result['fit_rows'] == 24 and result['holdout_rows'] == 6
    assert not result.get('image_projection')
    assert list(pd.read_csv(tmp_path / 'out/problem/data/test.csv').columns) == ['id', 'text']
