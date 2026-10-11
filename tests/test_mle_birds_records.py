import csv
import hashlib
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import types
import wave

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

MODULE = Path(__file__).parents[1] / 'deploy/mle/bird_records.py'
spec = importlib.util.spec_from_file_location('bird_records', MODULE)
birds = importlib.util.module_from_spec(spec)
spec.loader.exec_module(birds)


def write_table(path, header, rows):
    with path.open('w', newline='') as stream:
        writer = csv.writer(stream)
        writer.writerow(header)
        writer.writerows(rows)


def fixture(tmp_path, monkeypatch, known=10, unknown=3):
    prepared = tmp_path / 'prepared'
    public = prepared / 'public'
    essential, supplemental = public / 'essential_data', public / 'supplemental_data'
    (essential / 'src_wavs').mkdir(parents=True)
    supplemental.mkdir()
    (prepared / 'private').mkdir()
    (public / 'description.md').write_text('Synthetic native bird task')
    ids = list(range(known + unknown))
    write_table(essential / 'CVfolds_2.txt', ['rec_id', 'fold'], [[rec, int(rec >= known)] for rec in ids])
    write_table(essential / 'rec_id2filename.txt', ['rec_id', 'filename'], [[rec, f'a_{rec}'] for rec in ids])
    write_table(essential / 'rec_labels_test_hidden.txt', ['rec_id', 'labels'],
        [[rec, '?'] if rec >= known else ([rec] if rec % 3 == 0 else [rec, rec % 19]) for rec in ids])
    write_table(essential / 'species_list.txt', ['species_id', 'species'], [[species, f'Species{species}'] for species in range(19)])
    pd.DataFrame([(rec * 100 + species, 0.0) for rec in ids if rec >= known for species in range(19)],
        columns=['Id', 'Probability']).to_csv(public / 'sample_submission.csv', index=False)
    for rec in ids:
        with wave.open(str(essential / 'src_wavs' / f'a_{rec}.wav'), 'wb') as stream:
            stream.setparams((1, 2, 16000, 0, 'NONE', 'not compressed'))
            stream.writeframes(np.full(160000, rec, dtype='<i2').tobytes())
    for directory in ('spectrograms', 'filtered_spectrograms', 'supervised_segmentation', 'segmentation_examples'):
        (supplemental / directory).mkdir()
        for rec in ids:
            (supplemental / directory / f'a_{rec}.bmp').write_bytes(b'BM' + str(rec).encode())
    hashes = {}
    for name in birds.GLOBAL_BITMAP_SHA256:
        content = b'BM synthetic unlabeled visualization ' + name.encode()
        (supplemental / name).write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    # Test fixture assets explicitly attested only inside this test process.
    monkeypatch.setattr(birds, 'GLOBAL_BITMAP_SHA256', hashes)
    write_table(supplemental / 'histogram_of_segments.txt', ['rec_id', '[histogram of segment features]'],
        [[rec] + ([0.0] * 100 if rec % 2 == 0 else [1.0] + [0.0] * 99) for rec in ids])
    write_table(supplemental / 'segment_features.txt', ['rec_id', '[histogram of segment features]'],
        [[rec, 0] + [0.2] * 38 for rec in ids if rec % 2])
    write_table(supplemental / 'segment_rectangles.txt', ['rec_id', '[histogram of segment features]'],
        [[rec, 0, 1, 2, 3, 4, ''] for rec in ids if rec % 2])
    return prepared


def test_ragged_known_empty_and_unknown():
    assert birds.label_set([]) == set()
    assert birds.label_set(['?']) is None
    assert birds.label_set(['0', '18']) == {0, 18}
    for fields in [[''], ['?', '0'], ['19'], ['-1'], ['0', '0'], ['1.0']]:
        with pytest.raises(ValueError): birds.label_set(fields)


def test_frozen_206_52_split_and_visibility(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch, known=258, unknown=64)
    before = {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}
    out = tmp_path / 'out'
    split = birds.carve_records(source, out, seed=0, holdout=0.2)
    assert (split['fit_rows'], split['holdout_rows']) == (206, 52)
    manifest = json.loads((out / 'evaluation/record_split.json').read_text())
    fit, held = train_test_split(list(range(258)), test_size=.2, random_state=0, shuffle=True)
    assert manifest['fit_rec_ids'] == sorted(fit) and manifest['held_rec_ids'] == sorted(held)
    data = out / 'problem/data'
    folds, names, labels = birds.load_records(data)
    assert len(folds) == 258 and {r for r, y in labels.items() if y is None} == set(held)
    assert all(labels[r] == (set() if r % 3 == 0 else {r % 19}) for r in fit)
    assert len(pd.read_csv(out / 'evaluation/answers.csv')) == 988
    for rec in range(258):
        name = f'a_{rec}.wav'
        assert (data / 'essential_data/src_wavs' / name).read_bytes() == (source / 'public/essential_data/src_wavs' / name).read_bytes()
    for directory in ('spectrograms', 'filtered_spectrograms', 'supervised_segmentation'):
        assert len(list((data / 'supplemental_data' / directory).iterdir())) == 258
        for rec in range(258):
            assert (data / 'supplemental_data' / directory / f'a_{rec}.bmp').read_bytes() == (source / 'public/supplemental_data' / directory / f'a_{rec}.bmp').read_bytes()
    assert {p.stem for p in (data / 'supplemental_data/segmentation_examples').iterdir()} == {f'a_{r}' for r in fit}
    for table in ('histogram_of_segments.txt', 'segment_features.txt', 'segment_rectangles.txt'):
        _, original = birds.rows(source / 'public/supplemental_data' / table)
        _, projected = birds.rows(data / 'supplemental_data' / table)
        assert projected == [row for row in original if int(row[0]) < 258]
    assert len(birds.rows(data / 'supplemental_data/histogram_of_segments.txt')[1]) == 258
    assert not any('answers' in p.name or p.suffix in ('.zip', '.tar') for p in data.rglob('*'))
    assert before == {p.relative_to(source): hashlib.sha256(p.read_bytes()).hexdigest() for p in source.rglob('*') if p.is_file()}


@pytest.mark.parametrize('bad', ['duplicate_rec', 'bad_join', 'traversal', 'unknown_label', 'archive', 'symlink', 'feature_duplicate', 'feature_missing', 'rectangle_key', 'hist_nan', 'bad_bitmap', 'audio_duplicate'])
def test_invalid_source_fails_closed(tmp_path, monkeypatch, bad):
    source = fixture(tmp_path, monkeypatch)
    public = source / 'public'
    essential, supplemental = public / 'essential_data', public / 'supplemental_data'
    if bad in ('duplicate_rec', 'bad_join', 'traversal'):
        path = essential / 'rec_id2filename.txt'
        header, body = birds.rows(path)
        if bad == 'duplicate_rec': body.append(body[0])
        elif bad == 'bad_join': body.pop()
        else: body[0][1] = '../private'
        write_table(path, header, body)
    elif bad == 'unknown_label':
        path = essential / 'rec_labels_test_hidden.txt'
        header, body = birds.rows(path)
        body[0] = ['0', '?']
        write_table(path, header, body)
    elif bad == 'archive': (public / 'backup.zip').write_bytes(b'held labels')
    elif bad == 'symlink':
        path = essential / 'src_wavs/a_0.wav'
        path.unlink()
        path.symlink_to(essential / 'src_wavs/a_1.wav')
    elif bad == 'bad_bitmap': (supplemental / 'segment_mosaic.bmp').write_bytes(b'unknown')
    elif bad == 'audio_duplicate': (essential / 'src_wavs/a_0.wav').write_bytes((essential / 'src_wavs/a_1.wav').read_bytes())
    else:
        name = 'segment_rectangles.txt' if bad == 'rectangle_key' else 'histogram_of_segments.txt'
        path = supplemental / name
        header, body = birds.rows(path)
        if bad == 'feature_duplicate': body.append(body[0])
        elif bad == 'feature_missing': body.pop()
        elif bad == 'rectangle_key': body[0][1] = '1'
        elif bad == 'hist_nan': body[0][1] = 'nan'
        write_table(path, header, body)
    with pytest.raises(ValueError): birds.carve_records(source, tmp_path / 'out', seed=0, holdout=.2)
    assert not (tmp_path / 'out/problem/data').exists()


@pytest.mark.parametrize('bad', ['duplicate', 'missing', 'extra', 'fractional', 'negative_id', 'nan_id', 'nan', 'inf', 'negative', 'large', 'columns', 'missing_species'])
def test_prediction_rejections(bad):
    expected = pd.DataFrame({'Id': list(range(100, 119)), 'Probability': 0.0})
    submitted = expected.copy()
    if bad == 'duplicate': submitted.loc[1, 'Id'] = 100
    elif bad == 'missing': submitted = submitted.iloc[:-1]
    elif bad == 'extra': submitted.loc[0, 'Id'] = 200
    elif bad in ('fractional', 'negative_id', 'nan_id'):
        submitted = submitted.astype({'Id': float})
        submitted.loc[0, 'Id'] = {'fractional': 100.5, 'negative_id': -1, 'nan_id': np.nan}[bad]
    elif bad == 'columns': submitted.columns = ['id', 'Probability']
    elif bad == 'missing_species': submitted, expected = submitted.iloc[:-1], expected.iloc[:-1]
    else: submitted.loc[0, 'Probability'] = {'nan':np.nan, 'inf':np.inf, 'negative':-.1, 'large':1.1}[bad]
    with pytest.raises(ValueError): birds.validate_probabilities(submitted, expected)


def test_identical_starter_native_public_and_terminal(tmp_path, monkeypatch):
    source = fixture(tmp_path, monkeypatch)
    out = tmp_path / 'out'
    birds.carve_records(source, out, seed=0, holdout=.2)
    for index, data in enumerate((out / 'problem/data', source / 'public')):
        target = tmp_path / f'predictions{index}'
        subprocess.run([sys.executable, '-c', birds.STARTER], check=True,
            env={**os.environ, 'DATA_DIR': str(data), 'OUTPUT_DIR': str(target)})
        submission = pd.read_csv(target / 'submission.csv')
        sample = pd.read_csv(data / 'sample_submission.csv')
        assert birds.validate_probabilities(submission.iloc[::-1], sample).Id.tolist() == sample.Id.tolist()
        folds, _, labels = birds.load_records(data)
        fit = [y for r, y in labels.items() if folds[r] == 0]
        for row in submission.itertuples():
            assert row.Probability == pytest.approx((sum(row.Id % 100 in y for y in fit) + 1) / (len(fit) + 2))


def test_single_flatten_auc_is_not_macro_auc():
    labels = np.array([[0, 0], [1, 0], [1, 0], [1, 1]])
    priors = np.tile([.75, .25], (4, 1))
    assert roc_auc_score(labels.ravel(), priors.ravel()) == .75
    assert roc_auc_score(labels, priors, average='macro') == .5


def test_builder_seals_and_evaluator_uses_official_flattened_metric(tmp_path, monkeypatch, capsys):
    prepared = fixture(tmp_path, monkeypatch)
    official = tmp_path / 'official'
    (official / 'mlebench').mkdir(parents=True)
    (official / 'UPSTREAM_REVISION').write_text('synthetic-official-revision')
    (official / 'mlebench/__init__.py').write_text('')
    # Simulate the pinned official grade.py's one flattened target-column AUC.
    (official / 'mlebench/registry.py').write_text('''from sklearn.metrics import roc_auc_score
class Grader:
    def __call__(self, submission, answers):
        return roc_auc_score(answers.sort_values('Id').Probability, submission.sort_values('Id').Probability)
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
    monkeypatch.setitem(sys.modules, 'bird_records', birds)
    builder_spec = importlib.util.spec_from_file_location('bird_builder', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    def starter_score(python, root, metric):
        output = tmp_path / 'builder-predictions'
        env = {**os.environ, 'DATA_DIR': str(root / 'problem/data'), 'OUTPUT_DIR': str(output),
            'PYTHONPATH': str(official)}
        subprocess.run([python, str(root / 'starter/run.py')], check=True, env=env)
        subprocess.run([python, str(root / 'evaluation/evaluator.py')], check=True, env=env, cwd=output)
        result = json.loads((output / 'score.json').read_text())[metric]
        answers = pd.read_csv(root / 'evaluation/answers.csv')
        submission = pd.read_csv(output / 'submission.csv')
        assert result == roc_auc_score(answers.Probability, submission.Probability)
        return result
    monkeypatch.setattr(builder, 'run_starter', starter_score)
    tasks = tmp_path / 'tasks'
    monkeypatch.setattr(sys, 'argv', ['make_task_root.py', 'mlsp-2013-birds', '--prepared', str(prepared),
        '--tasks-root', str(tasks), '--cache', str(tmp_path / 'cache'), '--official-source', str(official),
        '--python', sys.executable, '--seed', '0', '--holdout', '0.2'])
    assert builder.main() == 0
    summary = json.loads(capsys.readouterr().out)
    assert summary['split'] == {'fit_rows':8, 'holdout_rows':2, 'stratified_on':None}
    root = tasks / 'mlsp-2013-birds'
    assert builder.verify_assets(root / 'custody.json')['task'] == 'mlsp-2013-birds'
    metadata = json.loads((root / 'metadata.json').read_text())
    assert metadata['official_test_unlabeled_derived_assets_visible'] is True
    assert 'not absence of all official-test-derived input information' in metadata['official_test_visibility_scope']
    assert (root / 'evaluation/bird_records.py').is_file()
    assert 'ONE flattened AUC' in (root / 'problem/README.md').read_text()
