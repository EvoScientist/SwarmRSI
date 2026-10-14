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

MODULE = Path(__file__).parents[1] / 'deploy/mle/aptos_images.py'
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location('aptos_images', MODULE)
aptos = importlib.util.module_from_spec(spec)
spec.loader.exec_module(aptos)


def fixture(tmp_path, duplicate=True, groups_per_class=10):
    prepared = tmp_path / 'prepared'
    public = prepared / 'public'
    for directory in ('train_images', 'test_images'):
        (public / directory).mkdir(parents=True)
    (prepared / 'private').mkdir()
    (public / 'description.md').write_text('Synthetic native ordinal PNG task')
    rows = []
    for index in range(groups_per_class * 5):
        identifier = f'{index:012d}'
        rows.append((identifier, index % 5))
        image = Image.fromarray(np.full((5 + index % 2, 9, 3), index, dtype=np.uint8))
        image.save(public / 'train_images' / (identifier + '.png'), compress_level=0)
    if duplicate:
        rows.append(('duplicate000', 0))
        with Image.open(public / 'train_images/000000000000.png') as image:
            image.save(public / 'train_images/duplicate000.png', compress_level=9)
    train = pd.DataFrame(rows, columns=['id_code', 'diagnosis'])
    train.to_csv(public / 'train.csv', index=False)
    test = pd.DataFrame({'id_code': ['new000', 'new001']})
    test.to_csv(public / 'test.csv', index=False)
    sample = test.copy()
    sample['diagnosis'] = 0
    sample.to_csv(public / 'sample_submission.csv', index=False)
    for index, identifier in enumerate(test.id_code):
        Image.new('RGB', (8, 6), (200 + index, 1, 2)).save(public / 'test_images' / (identifier + '.png'))
    return prepared


def tree_hash(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def test_group_split_all_rows_bytes_and_no_leaks(tmp_path):
    source = fixture(tmp_path)
    public = source / 'public'
    (public / 'full_labels.csv').write_text('original labels excluded')
    (public / 'images.zip').write_bytes(b'original complete dataset')
    before = tree_hash(source)
    out, repeat = tmp_path / 'out', tmp_path / 'repeat'
    split = aptos.carve_aptos(source, out, seed=0, holdout=.2)
    assert aptos.carve_aptos(source, repeat, seed=0, holdout=.2) == split
    assert tree_hash(out) == tree_hash(repeat)
    manifest = json.loads((out / 'evaluation/image_manifest.json').read_text())
    groups = {row['id_code']: row['group'] for row in manifest}
    assert groups['000000000000'] == groups['duplicate000']
    assert (public / 'train_images/000000000000.png').read_bytes() != (public / 'train_images/duplicate000.png').read_bytes()
    original = pd.read_csv(public / 'train.csv', dtype={'id_code': str})
    unique = list(dict.fromkeys(groups.values()))
    labels = dict(zip(original.id_code.map(groups), original.diagnosis))
    expected_fit, expected_held = train_test_split(unique, test_size=.2, random_state=0,
        stratify=[labels[group] for group in unique])
    receipt = json.loads((out / 'evaluation/image_split.json').read_text())
    assert receipt['fit_groups'] == expected_fit and receipt['held_groups'] == expected_held
    data = out / 'problem/data'
    assert {p.name for p in data.iterdir()} == {'train_images', 'test_images', 'train.csv', 'test.csv', 'sample_submission.csv'}
    fit = pd.read_csv(data / 'train.csv', dtype={'id_code': str})
    held = pd.read_csv(out / 'evaluation/answers.csv', dtype={'id_code': str})
    assert set(fit.diagnosis) == set(held.diagnosis) == aptos.CLASSES
    assert not set(fit.id_code) & set(held.id_code)
    assert set(fit.id_code) | set(held.id_code) == set(original.id_code)
    assert len(fit) + len(held) == 51 and split['fit_group_count'] + split['holdout_group_count'] == 50
    assert not set(fit.id_code.map(groups)) & set(held.id_code.map(groups))
    for frame, folder in ((fit, 'train_images'), (held, 'test_images')):
        assert frame.id_code.tolist() == original[original.id_code.isin(frame.id_code)].id_code.tolist()
        for identifier in frame.id_code:
            assert (data / folder / (identifier + '.png')).read_bytes() == (public / 'train_images' / (identifier + '.png')).read_bytes()
    assert list(pd.read_csv(data / 'test.csv').columns) == ['id_code']
    assert (pd.read_csv(data / 'sample_submission.csv').diagnosis == 0).all()
    assert before == tree_hash(source)


def test_conflicting_duplicate_keeps_published_labels_on_one_side(tmp_path):
    # APTOS publishes 27 pixel-identical groups whose rows disagree on diagnosis.
    source = fixture(tmp_path)
    public = source / 'public'
    train = pd.read_csv(public / 'train.csv', dtype={'id_code': str})
    train.loc[train.id_code == 'duplicate000', 'diagnosis'] = 1
    train.to_csv(public / 'train.csv', index=False)
    split = aptos.carve_aptos(source, tmp_path / 'out', seed=0, holdout=.2)
    assert split['conflicting_label_groups'] == 1
    fit = pd.read_csv(tmp_path / 'out/problem/data/train.csv', dtype={'id_code': str})
    held = pd.read_csv(tmp_path / 'out/evaluation/answers.csv', dtype={'id_code': str})
    both = pd.concat([fit, held]).set_index('id_code').diagnosis
    assert both.loc['duplicate000'] == 1 and both.loc['000000000000'] == train.set_index('id_code').diagnosis.loc['000000000000']
    assert ({'duplicate000', '000000000000'} <= set(fit.id_code)) or ({'duplicate000', '000000000000'} <= set(held.id_code))


@pytest.mark.parametrize('bad', ['duplicate_id', 'unsafe_id', 'missing_image',
    'extra_image', 'symlink_image', 'symlink_dir', 'corrupt', 'wrong_format', 'test_join',
    'train_test_overlap', 'held_label_in_test', 'label_nan', 'label_inf', 'label_fraction', 'label_range', 'palette'])
def test_bad_source_fails_closed(tmp_path, bad):
    source = fixture(tmp_path)
    public = source / 'public'
    train = pd.read_csv(public / 'train.csv', dtype={'id_code': str})
    path = public / 'train_images/000000000000.png'
    if bad == 'duplicate_id': train.loc[1, 'id_code'] = train.id_code.iloc[0]
    elif bad == 'unsafe_id': train.loc[0, 'id_code'] = '../escape'
    elif bad == 'missing_image': path.unlink()
    elif bad == 'extra_image': (public / 'train_images/extra.png').write_bytes(path.read_bytes())
    elif bad == 'symlink_image': path.unlink(); path.symlink_to(public / 'train_images/000000000001.png')
    elif bad == 'symlink_dir':
        (public / 'train_images').rename(public / 'elsewhere'); (public / 'train_images').symlink_to(public / 'elsewhere')
    elif bad == 'corrupt': path.write_bytes(b'corrupt')
    elif bad == 'wrong_format': Image.new('RGB', (9,5)).save(path, format='TIFF')
    elif bad == 'palette': Image.new('P', (9,5)).save(path)
    elif bad in ('test_join', 'train_test_overlap', 'held_label_in_test'):
        test = pd.read_csv(public / 'test.csv')
        if bad == 'test_join': test.loc[0, 'id_code'] = 'missing'
        elif bad == 'train_test_overlap': test.loc[0, 'id_code'] = train.id_code.iloc[0]
        else: test['diagnosis'] = 0
        test.to_csv(public / 'test.csv', index=False)
    else:
        train['diagnosis'] = train.diagnosis.astype(float)
        train.loc[0, 'diagnosis'] = {'label_nan': np.nan, 'label_inf': np.inf, 'label_fraction': .5, 'label_range': 5}[bad]
    train.to_csv(public / 'train.csv', index=False)
    with pytest.raises((ValueError, OSError)):
        aptos.carve_aptos(source, tmp_path / 'out', seed=0, holdout=.2)
    assert not (tmp_path / 'out/problem/data').exists()


def test_insufficient_groups_stops_instead_of_resplitting(tmp_path):
    source = fixture(tmp_path, groups_per_class=1)
    with pytest.raises(ValueError, match='insufficient'):
        aptos.carve_aptos(source, tmp_path / 'out', seed=0, holdout=.2)


def test_small_held_group_count_does_not_fall_back_to_row_split(tmp_path):
    source = fixture(tmp_path, groups_per_class=2)
    with pytest.raises(ValueError):
        aptos.carve_aptos(source, tmp_path / 'out', seed=0, holdout=.2)
    assert not (tmp_path / 'out/problem/data').exists()


@pytest.mark.parametrize('bad', ['missing', 'extra', 'duplicate', 'fraction', 'nan', 'inf', 'low', 'high', 'columns'])
def test_prediction_contract(tmp_path, bad):
    answers = pd.DataFrame({'id_code': ['a', 'b'], 'diagnosis': [0, 4]})
    submission = answers.copy()
    submission['diagnosis'] = submission.diagnosis.astype(float)
    if bad == 'missing': submission = submission.iloc[:1]
    elif bad == 'extra': submission.loc[2] = ['c', 2]
    elif bad == 'duplicate': submission.loc[1, 'id_code'] = 'a'
    elif bad == 'columns': submission['extra'] = 0
    else: submission.loc[0, 'diagnosis'] = {'fraction': .5, 'nan': np.nan, 'inf': np.inf, 'low': -1, 'high': 5}[bad]
    with pytest.raises(ValueError): aptos.validate_diagnosis(submission, answers)


def test_main_and_identical_native_public_terminal_contract(tmp_path):
    assert any(isinstance(node, ast.FunctionDef) and node.name == 'main' for node in ast.parse(aptos.STARTER).body)
    source = fixture(tmp_path)
    out = tmp_path / 'out'
    aptos.carve_aptos(source, out, seed=0, holdout=.2)
    for index, data in enumerate((out / 'problem/data', source / 'public')):
        output = tmp_path / f'output{index}'
        env = {**os.environ, 'DATA_DIR': str(data), 'OUTPUT_DIR': str(output),
            'SWARMRSI_SUBMISSION_FIT': 'full' if index else 'validation'}
        subprocess.run([sys.executable, '-c', aptos.STARTER], check=True, env=env)
        prediction = pd.read_csv(output / 'submission.csv', dtype={'id_code': str})
        sample = pd.read_csv(data / 'sample_submission.csv', dtype={'id_code': str})
        assert aptos.validate_diagnosis(prediction.iloc[::-1], sample).id_code.tolist() == sample.id_code.tolist()
        assert set(prediction.diagnosis) == {int(pd.read_csv(data / 'train.csv').diagnosis.mode().iloc[0])}


def test_real_builder_and_unchanged_official_quadratic_kappa(tmp_path, monkeypatch, capsys):
    source = fixture(tmp_path)
    official = tmp_path / 'official'
    (official / 'mlebench').mkdir(parents=True)
    (official / 'UPSTREAM_REVISION').write_text('synthetic-official')
    (official / 'mlebench/__init__.py').write_text('')
    (official / 'mlebench/registry.py').write_text('''from sklearn.metrics import cohen_kappa_score
class Grader:
    def __call__(self, submission, answers):
        assert set(submission.id_code) == set(answers.id_code)
        return cohen_kappa_score(submission.sort_values('id_code').diagnosis,
            answers.sort_values('id_code').diagnosis, weights='quadratic')
class Registry:
    def set_data_dir(self, path): return self
    def get_competition(self, name):
        class Competition: grader = Grader()
        return Competition()
registry = Registry()
''')
    grader = types.SimpleNamespace(name='quadratic-kappa', is_lower_better=lambda _: False,
        rank_score=lambda *_: {'gold_threshold': .9})
    registry = types.SimpleNamespace(set_data_dir=lambda _: types.SimpleNamespace(
        get_competition=lambda _: types.SimpleNamespace(grader=grader)))
    monkeypatch.setitem(sys.modules, 'mlebench', types.ModuleType('mlebench'))
    monkeypatch.setitem(sys.modules, 'mlebench.grade', types.SimpleNamespace(get_leaderboard=lambda _: [1,2]))
    monkeypatch.setitem(sys.modules, 'mlebench.registry', types.SimpleNamespace(registry=registry))
    monkeypatch.setitem(sys.modules, 'aptos_images', aptos)
    builder_spec = importlib.util.spec_from_file_location('aptos_builder', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    monkeypatch.setenv('PYTHONPATH', str(official) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    tasks = tmp_path / 'tasks'
    monkeypatch.setattr(sys, 'argv', ['make_task_root.py', 'aptos2019-blindness-detection', '--prepared', str(source),
        '--tasks-root', str(tasks), '--official-source', str(official), '--python', sys.executable])
    assert builder.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['starter_baseline'] == pytest.approx(0.0)
    root = tasks / 'aptos2019-blindness-detection'
    assert builder.verify_assets(root / 'custody.json')['task'] == 'aptos2019-blindness-detection'
    assert 'new content-group' in (root / 'problem/README.md').read_text()
    output = tmp_path / 'permuted-output'
    output.mkdir()
    pd.read_csv(root / 'evaluation/answers.csv', dtype={'id_code': str}).iloc[::-1].to_csv(output / 'submission.csv', index=False)
    subprocess.run([sys.executable, str(root / 'evaluation/evaluator.py')], cwd=output,
        env={**os.environ, 'OUTPUT_DIR': str(output)}, check=True)
    assert json.loads((output / 'score.json').read_text())['quadratic_kappa'] == 1.0
