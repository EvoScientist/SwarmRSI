"""RANZCR carves like the official prepare.py and grades exactly the official nine classes."""
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

MODULE = Path(__file__).parents[1] / 'deploy/mle/ranzcr_images.py'
sys.path.insert(0, str(MODULE.parent))
spec = importlib.util.spec_from_file_location('ranzcr_images', MODULE)
ranzcr = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ranzcr)

ID, PATIENT, CLASSES = ranzcr.ID, ranzcr.PATIENT, ranzcr.CLASSES
COMPETITION = 'ranzcr-clip-catheter-line-classification'
OFFICIAL_CLASSES = [  # the pinned official classes.py, restated independently
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal', 'NGT - Abnormal', 'NGT - Borderline',
    'NGT - Incompletely Imaged', 'NGT - Normal', 'CVC - Abnormal', 'CVC - Borderline']


def uid(index):
    return f'1.2.826.0.1.3680043.8.498.{10**20 + index}'


def jpeg(path, shade):
    Image.new('L', (6, 5), shade).save(path, format='JPEG')


def fixture(tmp_path, patients=60, per_patient=3):
    prepared = tmp_path / 'prepared'
    public = prepared / 'public'
    for folder in ('train', 'test'):
        (public / folder).mkdir(parents=True)
    (prepared / 'private').mkdir()
    (public / 'description.md').write_text('Synthetic RANZCR description')
    random = np.random.RandomState(0)
    rows = []
    for patient in range(patients):
        for repeat in range(per_patient):
            index = patient * per_patient + repeat
            labels = (random.rand(11) < .4).astype(int).tolist()
            rows.append([uid(index), *labels, f'{patient:07x}ab'])
    train = pd.DataFrame(rows, columns=ranzcr.TRAIN_COLUMNS)
    train.to_csv(public / 'train.csv', index=False)
    for index, identifier in enumerate(train[ID]):
        jpeg(public / 'train' / (identifier + '.jpg'), index)
    annotations = pd.DataFrame([[identifier, 'CVC - Normal', f'[[{i}, 7], [13, {i + 1}]]']
                                for i, identifier in enumerate(train[ID]) if i % 3 == 0]
                               + [[train[ID].iloc[1], 'ETT - Normal', '[[1, 2]]']],
                               columns=ranzcr.ANNOTATION_COLUMNS)
    annotations.to_csv(public / 'train_annotations.csv', index=False)
    test_ids = [uid(9000 + i) for i in range(3)]
    sample = pd.DataFrame({ID: test_ids, **{c: 0 for c in CLASSES}})
    sample.to_csv(public / 'sample_submission.csv', index=False)
    for index, identifier in enumerate(test_ids):
        jpeg(public / 'test' / (identifier + '.jpg'), 200 + index)
    private = pd.DataFrame([[i, *([0] * 11), 'zz'] for i in test_ids], columns=ranzcr.TRAIN_COLUMNS)
    private.to_csv(prepared / 'private/test.csv', index=False)
    return prepared


def tree_hash(root):
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in root.rglob('*') if p.is_file()}


def read(path):
    return pd.read_csv(path, dtype={ID: str, PATIENT: str})


def test_scored_classes_are_the_official_nine():
    assert CLASSES == OFFICIAL_CLASSES
    assert ranzcr.UNSCORED == ['CVC - Normal', 'Swan Ganz Catheter Present']


def test_official_row_carve_projects_tables_images_and_annotations(tmp_path):
    source = fixture(tmp_path)
    public = source / 'public'
    before = tree_hash(source)
    out = tmp_path / 'out'
    split = ranzcr.carve_ranzcr(source, out, seed=0)
    ranzcr.carve_ranzcr(source, tmp_path / 'repeat', seed=0)
    assert tree_hash(out) == tree_hash(tmp_path / 'repeat')
    assert split['grouped_on'] is None and split['holdout'] == ranzcr.OFFICIAL_HOLDOUT == .1
    data = out / 'problem/data'
    assert {p.name for p in data.iterdir()} == {'train', 'test', 'train.csv',
                                                'train_annotations.csv', 'sample_submission.csv'}
    original = read(public / 'train.csv')
    fit = read(data / 'train.csv')
    answers = read(out / 'evaluation/answers.csv')
    held = original.set_index(ID).loc[answers[ID]].reset_index()
    # The same rows the official prepare.py call would hold out; patients may straddle, as officially.
    _, expected = train_test_split(original, test_size=.1, random_state=0)
    assert answers[ID].tolist() == expected.sort_index()[ID].tolist()
    assert split['holdout_rows'] == 18 and split['shared_patients'] == len(set(fit[PATIENT]) & set(held[PATIENT])) > 0
    assert set(fit[ID]) | set(held[ID]) == set(original[ID]) and not set(fit[ID]) & set(held[ID])
    assert split['fit_rows'] + split['holdout_rows'] == len(original)
    # The fit table keeps every public column and the original rows verbatim.
    assert list(fit.columns) == ranzcr.TRAIN_COLUMNS
    pd.testing.assert_frame_equal(fit, original[original[ID].isin(fit[ID])].reset_index(drop=True))
    # Answers and the carved sample carry exactly the officially scored columns.
    assert list(answers.columns) == [ID, *OFFICIAL_CLASSES]
    sample = read(data / 'sample_submission.csv')
    assert list(sample.columns) == list(answers.columns) == list(read(public / 'sample_submission.csv').columns)
    assert sample[ID].tolist() == answers[ID].tolist() and (sample[CLASSES] == 0).all().all()
    pd.testing.assert_frame_equal(answers[CLASSES], held[CLASSES])
    # Annotations restricted to fit images, rows verbatim.
    annotations = pd.read_csv(public / 'train_annotations.csv', dtype=str)
    kept = pd.read_csv(data / 'train_annotations.csv', dtype=str)
    pd.testing.assert_frame_equal(kept, annotations[annotations[ID].isin(fit[ID])].reset_index(drop=True))
    assert not set(kept[ID]) & set(held[ID]) and split['annotation_rows_withheld'] > 0
    # Fit images under train/, held images under test/, bytes unchanged; no official test images.
    for folder, frame in (('train', fit), ('test', held)):
        assert {p.name for p in (data / folder).iterdir()} == {i + '.jpg' for i in frame[ID]}
        for identifier in frame[ID]:
            assert (data / folder / (identifier + '.jpg')).read_bytes() == (public / 'train' / (identifier + '.jpg')).read_bytes()
    official_test = {p.name for p in (public / 'test').iterdir()}
    assert not official_test & {p.name for p in data.rglob('*.jpg')}
    # Held ids appear candidate-side only as the sample's rows and the test images, never with labels.
    for table in ('train.csv', 'train_annotations.csv'):
        text = (data / table).read_text()
        assert not any(identifier in text for identifier in held[ID])
    assert before == tree_hash(source)


def test_repeated_carve_stays_refused(tmp_path):
    builder_spec = importlib.util.spec_from_file_location('ranzcr_folds', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    assert builder.ADAPTER_COMPETITIONS[COMPETITION] == 'ranzcr_images'
    with pytest.raises(SystemExit, match='ranzcr_images'):
        builder.check_carve_folds(COMPETITION, fixture(tmp_path), 3)
    builder.check_carve_folds(COMPETITION, tmp_path / 'prepared', 1)


@pytest.mark.parametrize('bad', ['schema', 'missing_image', 'extra_image', 'not_jpeg', 'duplicate_id',
                                 'unsafe_id', 'label_range', 'no_patient', 'annotation_orphan',
                                 'train_test_overlap', 'sample_schema', 'held_class_absent'])
def test_bad_source_fails_closed(tmp_path, bad):
    source = fixture(tmp_path)
    public = source / 'public'
    train = pd.read_csv(public / 'train.csv', dtype=str)
    first = public / 'train' / (train[ID].iloc[0] + '.jpg')
    if bad == 'schema': train = train.drop(columns=['Swan Ganz Catheter Present'])
    elif bad == 'missing_image': first.unlink()
    elif bad == 'extra_image': jpeg(public / 'train' / (uid(7777) + '.jpg'), 3)
    elif bad == 'not_jpeg': Image.new('L', (6, 5)).save(first, format='PNG')
    elif bad == 'duplicate_id': train.loc[1, ID] = train[ID].iloc[0]
    elif bad == 'unsafe_id': train.loc[0, ID] = '../escape'
    elif bad == 'label_range': train.loc[0, 'ETT - Normal'] = '2'
    elif bad == 'no_patient': train.loc[0, PATIENT] = ''
    elif bad == 'annotation_orphan':
        with (public / 'train_annotations.csv').open('a') as stream:
            stream.write(f'{uid(8888)},CVC - Normal,"[[1, 2]]"\n')
    elif bad == 'train_test_overlap':
        sample = pd.read_csv(public / 'sample_submission.csv', dtype=str)
        sample.loc[0, ID] = train[ID].iloc[0]
        sample.to_csv(public / 'sample_submission.csv', index=False)
    elif bad == 'sample_schema':
        sample = pd.read_csv(public / 'sample_submission.csv', dtype=str)
        sample['CVC - Normal'] = 0
        sample.to_csv(public / 'sample_submission.csv', index=False)
    else: train['ETT - Abnormal'] = '0'
    train.to_csv(public / 'train.csv', index=False)
    with pytest.raises((ValueError, OSError)):
        ranzcr.carve_ranzcr(source, tmp_path / 'out', seed=0)
    assert not (tmp_path / 'out/problem/data').exists()


@pytest.mark.parametrize('bad', ['missing_row', 'extra_row', 'duplicate', 'missing_class',
                                 'high', 'negative', 'nan', 'text'])
def test_prediction_contract(bad):
    answers = pd.DataFrame({ID: [uid(1), uid(2)], **{c: [0, 1] for c in CLASSES}})
    submission = answers.copy().astype({c: float for c in CLASSES})
    if bad == 'missing_row': submission = submission.iloc[:1]
    elif bad == 'extra_row': submission.loc[2] = [uid(3), *([.5] * 9)]
    elif bad == 'duplicate': submission.loc[1, ID] = uid(1)
    elif bad == 'missing_class': submission = submission.drop(columns=['CVC - Borderline'])
    elif bad == 'high': submission.loc[0, 'ETT - Normal'] = 1.5
    elif bad == 'negative': submission.loc[0, 'ETT - Normal'] = -.1
    elif bad == 'nan': submission.loc[0, 'ETT - Normal'] = np.nan
    else: submission['ETT - Normal'] = submission['ETT - Normal'].astype(object); submission.loc[0, 'ETT - Normal'] = 'x'
    with pytest.raises(ValueError):
        ranzcr.validate_probabilities(submission, answers)


def test_extra_unscored_columns_are_dropped_like_the_official_grader():
    answers = pd.DataFrame({ID: [uid(1), uid(2)], **{c: [0, 1] for c in CLASSES}})
    submission = answers.iloc[::-1].copy()
    submission['CVC - Normal'] = .3
    submission['Swan Ganz Catheter Present'] = .1
    checked = ranzcr.validate_probabilities(submission, answers)
    assert list(checked.columns) == [ID, *CLASSES] and checked[ID].tolist() == answers[ID].tolist()


def test_starter_runs_on_carved_and_official_public_tree(tmp_path):
    source = fixture(tmp_path)
    out = tmp_path / 'out'
    ranzcr.carve_ranzcr(source, out, seed=0)
    for index, data in enumerate((out / 'problem/data', source / 'public')):
        output = tmp_path / f'output{index}'
        subprocess.run([sys.executable, '-c', ranzcr.STARTER], check=True,
                       env={**os.environ, 'DATA_DIR': str(data), 'OUTPUT_DIR': str(output)})
        prediction = read(output / 'submission.csv')
        sample = read(data / 'sample_submission.csv')
        assert list(prediction.columns) == [ID, *CLASSES]
        assert ranzcr.validate_probabilities(prediction, sample)[ID].tolist() == sample[ID].tolist()


OFFICIAL_REGISTRY = '''import numpy as np
from sklearn.metrics import roc_auc_score
CLASSES = %r
class Grader:
    def __call__(self, submission, answers):
        # Mirrors the official grade.py: per-class AUC after sorting on StudyInstanceUID.
        for name in CLASSES:
            assert name in answers.columns and name in submission.columns
        submission = submission.sort_values("StudyInstanceUID")
        answers = answers.sort_values("StudyInstanceUID")
        assert (submission.StudyInstanceUID.values == answers.StudyInstanceUID.values).all()
        return float(np.average([roc_auc_score(answers[c], submission[c]) for c in CLASSES]))
class Registry:
    def set_data_dir(self, path): return self
    def get_competition(self, name):
        class Competition: grader = Grader()
        return Competition()
registry = Registry()
''' % (OFFICIAL_CLASSES,)


def test_real_builder_seals_package_and_evaluator_grades_official_columns(tmp_path, monkeypatch, capsys):
    source = fixture(tmp_path)
    official = tmp_path / 'official'
    (official / 'mlebench').mkdir(parents=True)
    (official / 'UPSTREAM_REVISION').write_text('synthetic-official')
    (official / 'mlebench/__init__.py').write_text('')
    (official / 'mlebench/registry.py').write_text(OFFICIAL_REGISTRY)
    grader = types.SimpleNamespace(name='auc-roc', is_lower_better=lambda _: False,
                                   rank_score=lambda *_: {'gold_threshold': .97})
    registry = types.SimpleNamespace(set_data_dir=lambda _: types.SimpleNamespace(
        get_competition=lambda _: types.SimpleNamespace(grader=grader)))
    monkeypatch.setitem(sys.modules, 'mlebench', types.ModuleType('mlebench'))
    monkeypatch.setitem(sys.modules, 'mlebench.grade', types.SimpleNamespace(get_leaderboard=lambda _: [1, 2]))
    monkeypatch.setitem(sys.modules, 'mlebench.registry', types.SimpleNamespace(registry=registry))
    monkeypatch.setitem(sys.modules, 'ranzcr_images', ranzcr)
    builder_spec = importlib.util.spec_from_file_location('ranzcr_builder', MODULE.with_name('make_task_root.py'))
    builder = importlib.util.module_from_spec(builder_spec)
    builder_spec.loader.exec_module(builder)
    monkeypatch.setenv('PYTHONPATH', str(official) + os.pathsep + os.environ.get('PYTHONPATH', ''))
    tasks = tmp_path / 'tasks'
    monkeypatch.setattr(sys, 'argv', ['make_task_root.py', COMPETITION, '--prepared', str(source),
                                      '--tasks-root', str(tasks), '--official-source', str(official),
                                      '--python', sys.executable])
    assert builder.main() == 0
    result = json.loads(capsys.readouterr().out)
    assert result['starter_baseline'] == pytest.approx(.5)
    root = tasks / COMPETITION
    assert builder.verify_assets(root / 'custody.json')['task'] == COMPETITION
    metadata = json.loads((root / 'metadata.json').read_text())
    split = metadata['public_split']
    assert split['ranzcr_images'] and split['grouped_on'] is None and split['algorithm'] == 'train_test_split'
    assert split['holdout'] == .1 and split['requested_holdout'] == .2
    readme = (root / 'problem/README.md').read_text()
    assert 'not grouped by patient' in readme and 'nine scored' in readme and 'ROC AUC' in readme
    assert (root / 'private/private_test.csv').exists()
    assert not (root / 'problem/data/test.csv').exists()

    def grade(frame):
        output = tmp_path / 'graded'
        shutil_rmtree(output)
        output.mkdir()
        frame.to_csv(output / 'submission.csv', index=False)
        subprocess.run([sys.executable, str(root / 'evaluation/evaluator.py')], cwd=output,
                       env={**os.environ, 'OUTPUT_DIR': str(output)}, check=True)
        return json.loads((output / 'score.json').read_text())['auc_roc']

    answers = read(root / 'evaluation/answers.csv')
    perfect = answers.iloc[::-1].copy()
    perfect['CVC - Normal'] = .5  # unscored extras are ignored, as by the official grader
    assert grade(perfect) == 1.0
    inverted = answers.copy()
    inverted[CLASSES] = 1 - inverted[CLASSES]
    assert grade(inverted) == 0.0


def shutil_rmtree(path):
    import shutil
    shutil.rmtree(path, ignore_errors=True)
