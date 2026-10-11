"""Explicit labelled-image projection; no search policy or private-test feedback.

Only Histopath is routed here by the builder. Other schemas must be declared,
not guessed from filenames. Bytes are read once for decode, hashing and copying.
"""
import hashlib
import io
import json
from pathlib import Path
import re
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

CONTRACT = '''
Execution data contract: read DATA_DIR/train_labels.csv (id,label) and exact
96x96 RGB TIFF bytes at DATA_DIR/train/<id>.tif. Only fit labels and fit images are there.
Predict DATA_DIR/test/<id>.tif for precisely the IDs in sample_submission.csv;
these are held images during search, with labels accessible only to the
controller. IDs retain their original values and are not row positions.
Write OUTPUT_DIR/submission.csv with unique id,label rows; label is a finite
probability in [0,1]. The official metric is ROC AUC. Submission order is free
but the complete ID set must match. The identical program/path contract is
used for terminal refit on the unchanged full official public train/test tree.
Official test images are excluded from the search package. No archive, full
training-label table, held answers or split/content manifest is candidate-visible.
'''


def ordinary(path, *, directory=False):
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'ordinary {"directory" if directory else "file"} required: {path.name}')


def identifiers(frame):
    if frame.empty or 'id' not in frame or frame.id.isna().any():
        raise ValueError('nonempty image IDs required')
    values = frame.id.tolist()
    if any(not isinstance(v, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', v) for v in values):
        raise ValueError('unsafe image identifier')
    if len(set(values)) != len(values):
        raise ValueError('duplicate image IDs')
    return values


def validate_probabilities(submission, answers):
    """Validate the full table before the unchanged official grader is called."""
    if list(submission.columns) != list(answers.columns) or list(answers.columns)[0] != 'id':
        raise ValueError('submission columns differ from answer contract')
    left, right = identifiers(submission), identifiers(answers)
    if set(left) != set(right):
        raise ValueError('submission ID set mismatch')
    for frame in (submission, answers):
        values = frame.iloc[:, 1:].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
        if values.shape[1] == 0 or not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
            raise ValueError('probabilities must be finite in [0,1]')
        if values.shape[1] > 1 and not np.allclose(values.sum(axis=1), 1.0, atol=1e-6, rtol=0):
            raise ValueError('multiclass probabilities must sum to one')
    return submission.set_index('id').loc[right].reset_index()


def inventory(directory, ids):
    ordinary(directory, directory=True)
    expected = {identifier + '.tif' for identifier in ids}
    if {p.name for p in directory.iterdir()} != expected:
        raise ValueError('TIFF inventory differs from image IDs')
    for path in directory.iterdir():
        ordinary(path)


def carve_images(prepared, destination, *, seed, holdout, classes=(0, 1), targets=('label',), expected_size=None):
    """Stratified row split with exact TIFF projection and fail-closed duplicates.

    Multiclass parameters exist for contract tests, not an implicit task route.
    All original public training rows survive; no deduplication or resplitting.
    """
    public = prepared / 'public'
    ordinary(prepared, directory=True)
    ordinary(public, directory=True)
    for filename in ('train_labels.csv', 'sample_submission.csv'):
        ordinary(public / filename)
    train = pd.read_csv(public / 'train_labels.csv', dtype={'id': str})
    sample = pd.read_csv(public / 'sample_submission.csv', dtype={'id': str})
    if list(train.columns) != ['id', 'label'] or list(sample.columns) != ['id', *targets]:
        raise ValueError('explicit image label/sample schema required')
    train_ids, test_ids = identifiers(train), identifiers(sample)
    if set(train_ids) & set(test_ids):
        raise ValueError('official public train/test ID overlap')
    if (train.label.isna().any() or set(train.label) != set(classes)
            or (len(targets) != 1 and len(targets) != len(classes))):
        raise ValueError('training labels do not match declared classes')
    if len(targets) == 1 and tuple(classes) != (0, 1):
        raise ValueError('single probability column requires binary labels')
    validate_probabilities(sample, sample)
    inventory(public / 'train', train_ids)
    inventory(public / 'test', test_ids)
    fit, held = train_test_split(train, test_size=holdout, random_state=seed,
                                shuffle=True, stratify=train.label)
    fit, held = fit.sort_index(), held.sort_index()
    fit_ids = set(fit.id)
    data, evaluation = destination / 'problem/data', destination / 'evaluation'
    if data.exists():
        raise ValueError('refusing to overwrite an existing image projection')
    evaluation.mkdir(parents=True, exist_ok=True)
    data.parent.mkdir(parents=True, exist_ok=True)
    # Publish only a complete validated projection; partial writes are never DATA_DIR.
    with tempfile.TemporaryDirectory(prefix='.image-projection-', dir=destination) as temporary:
        staging = Path(temporary)
        staged_data = staging / 'data'
        for name in ('train', 'test'):
            (staged_data / name).mkdir(parents=True)
        seen = {}
        with (staging / 'image_manifest.jsonl').open('w') as manifest:
            for identifier in train_ids:
                blob = (public / 'train' / (identifier + '.tif')).read_bytes()
                with Image.open(io.BytesIO(blob)) as image:
                    if image.format != 'TIFF' or image.mode != 'RGB' or getattr(image, 'n_frames', 1) != 1:
                        raise ValueError('single-frame RGB TIFF required')
                    image.load()
                    mode, size = image.mode, image.size
                    if expected_size is not None and size != tuple(expected_size):
                        raise ValueError('image dimensions differ from declared task contract')
                    decoded = hashlib.sha256(json.dumps([mode, size]).encode() + image.tobytes()).hexdigest()
                if decoded in seen:
                    raise ValueError('duplicate decoded image content; no automatic resplit or deduplication')
                seen[decoded] = identifier
                fold = 'train' if identifier in fit_ids else 'test'
                (staged_data / fold / (identifier + '.tif')).write_bytes(blob)
                manifest.write(json.dumps({'id': identifier, 'fold': fold,
                    'file_sha256': hashlib.sha256(blob).hexdigest(), 'decoded_sha256': decoded,
                    'mode': mode, 'size': size}, sort_keys=True) + '\n')
        fit.to_csv(staged_data / 'train_labels.csv', index=False)
        answers = pd.DataFrame({'id': held.id.to_numpy()})
        for index, target in enumerate(targets):
            answers[target] = held.label.to_numpy() if len(targets) == 1 else (held.label == classes[index]).astype(int).to_numpy()
        neutral = answers.copy()
        neutral[list(targets)] = 0.5 if len(targets) == 1 else 1.0 / len(targets)
        validate_probabilities(neutral, answers)
        neutral.to_csv(staged_data / 'sample_submission.csv', index=False)
        answers.to_csv(evaluation / 'answers.csv', index=False)
        (evaluation / 'image_split.json').write_text(json.dumps({'fit_ids': fit.id.tolist(),
            'held_ids': held.id.tolist(), 'seed': seed, 'holdout': holdout}, sort_keys=True))
        (staging / 'image_manifest.jsonl').replace(evaluation / 'image_manifest.jsonl')
        staged_data.replace(data)
    return {'image_projection': True, 'algorithm': 'train_test_split', 'seed': seed,
        'holdout': holdout, 'stratified_on': 'label', 'grouped_on': None,
        'fit_rows': len(fit), 'holdout_rows': len(held), 'label_column': 'label',
        'id_column': 'id', 'target_columns': list(targets), 'image_size': expected_size,
        'filenames': {'train': 'train_labels.csv', 'test': None, 'sample': 'sample_submission.csv'}}


STARTER = '''import os
from pathlib import Path
import pandas as pd
from PIL import Image
def main():
    data = Path(os.environ['DATA_DIR'])
    train = pd.read_csv(data / 'train_labels.csv', dtype={'id': str})
    sample = pd.read_csv(data / 'sample_submission.csv', dtype={'id': str})
    # Exercise the actual image contract, not just the label and sample tables.
    for folder, frame in [('train', train), ('test', sample)]:
        for identifier in frame.id:
            with Image.open(data / folder / (identifier + '.tif')) as image:
                image.load()
    sample['label'] = float(train.label.mean())
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / 'submission.csv', index=False)
if __name__ == '__main__':
    main()
'''
