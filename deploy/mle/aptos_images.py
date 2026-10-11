"""Native APTOS ordinal image contract; no private-test feedback or search logic."""
import hashlib
import io
import json
from pathlib import Path
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

from image_projection import ordinary, identifiers

CLASSES = {0, 1, 2, 3, 4}
CONTRACT = '''
Execution data contract: DATA_DIR/train.csv has id_code,diagnosis and training
images are train_images/<id_code>.png. DATA_DIR/test.csv has id_code only;
predict test_images/<id_code>.png, using all IDs in sample_submission.csv.
Write OUTPUT_DIR/submission.csv with exactly id_code,diagnosis: one unique row
per requested ID, diagnosis an INTEGER class 0..4, not a probability. IDs and
PNG bytes retain their original values. No held diagnosis is candidate-visible.
The metric is the unchanged official quadratic weighted Cohen kappa, maximize.
Internal validation groups exact decoded mode/shape/pixels, stratifies unique
groups by diagnosis, and keeps all duplicate rows within the same side. Thus
row counts can differ from a naive 80/20 split. This is a new content-group
validation protocol, not an equal-protocol comparison with historical scores.
The split is frozen before scoring. The published training table has content
groups whose rows carry different diagnoses; every row keeps its published
label, the group stays on one side, and stratification uses the group's first
label. No samples are dropped or relabelled. The same loader and
program run on the original official public tree for terminal refit/prediction.
Official test images, hidden targets and group/hash manifests are not exposed
in the search data root. Never infer unknown test diagnosis as zero from sample
placeholders, and never treat rows with identical pixels as independent folds.
There is no network. ImageNet weights available offline through TORCH_HOME are
exactly torchvision resnet18 (IMAGENET1K_V1), resnet50 (IMAGENET1K_V2) and
efficientnet_b0 (IMAGENET1K_V1); any other weights would need a download and fail.
'''


def image_ids(frame):
    return identifiers(frame.rename(columns={'id_code': 'id'}))


def integer_labels(frame):
    values = pd.to_numeric(frame.diagnosis, errors='raise').to_numpy(dtype=float)
    if (not np.isfinite(values).all() or not (values == np.floor(values)).all()
            or not set(values) <= CLASSES):
        raise ValueError('diagnosis must be a finite integer class0..4')
    return values.astype(int)


def validate_diagnosis(submission, answers):
    for frame in (submission, answers):
        if list(frame.columns) != ['id_code', 'diagnosis']:
            raise ValueError('expected id_code,diagnosis schema')
        image_ids(frame)
        integer_labels(frame)
    if set(submission.id_code) != set(answers.id_code):
        raise ValueError('submission ID set mismatch')
    result = submission.set_index('id_code').loc[answers.id_code].reset_index()
    result['diagnosis'] = integer_labels(result)
    return result


def inventory(directory, ids):
    ordinary(directory, directory=True)
    if {p.name for p in directory.iterdir()} != {identifier + '.png' for identifier in ids}:
        raise ValueError('PNG inventory differs from declared IDs')
    for path in directory.iterdir():
        ordinary(path)


def carve_aptos(prepared, destination, *, seed, holdout):
    public = prepared / 'public'
    ordinary(prepared, directory=True)
    ordinary(public, directory=True)
    frames = {}
    for filename, columns in [('train.csv', ['id_code', 'diagnosis']),
                              ('test.csv', ['id_code']),
                              ('sample_submission.csv', ['id_code', 'diagnosis'])]:
        ordinary(public / filename)
        frame = pd.read_csv(public / filename, dtype={'id_code': str})
        if list(frame.columns) != columns:
            raise ValueError('explicit native APTOS table schema required')
        image_ids(frame)
        frames[filename] = frame
    train, test, sample = (frames[name] for name in ('train.csv', 'test.csv', 'sample_submission.csv'))
    train['diagnosis'] = integer_labels(train)
    integer_labels(sample)
    if set(train.diagnosis) != CLASSES:
        raise ValueError('all five classes required in source training labels')
    if set(train.id_code) & set(test.id_code) or set(test.id_code) != set(sample.id_code):
        raise ValueError('public train/test/sample ID join mismatch')
    inventory(public / 'train_images', train.id_code)
    inventory(public / 'test_images', test.id_code)
    data, evaluation = destination / 'problem/data', destination / 'evaluation'
    if data.exists():
        raise ValueError('refusing to overwrite existing APTOS projection')
    data.parent.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    # Decode/hash once and spool exact bytes locally before groups determine folds.
    with tempfile.TemporaryDirectory(prefix='.aptos-projection-', dir=destination) as temporary:
        staging = Path(temporary)
        spool = staging / 'spool'
        spool.mkdir()
        manifest, groups, labels, conflicts = [], {}, {}, set()
        for row in train.itertuples(index=False):
            blob = (public / 'train_images' / (row.id_code + '.png')).read_bytes()
            with Image.open(io.BytesIO(blob)) as image:
                if image.format != 'PNG' or getattr(image, 'n_frames', 1) != 1:
                    raise ValueError('single-frame PNG required')
                if image.mode == 'P':
                    raise ValueError('palette PNG requires an explicitly audited decoded-pixel contract')
                image.load()
                mode, size = image.mode, image.size
                digest = hashlib.sha256(json.dumps([mode, size]).encode() + image.tobytes()).hexdigest()
            # Stratify on the first label seen; a later row with a different label
            # keeps its own label and follows its group (27 such groups in APTOS).
            if digest in labels and labels[digest] != row.diagnosis:
                conflicts.add(digest)
            labels.setdefault(digest, row.diagnosis)
            groups[row.id_code] = digest
            (spool / (row.id_code + '.png')).write_bytes(blob)
            manifest.append({'id_code': row.id_code, 'group': digest,
                'byte_sha256': hashlib.sha256(blob).hexdigest(), 'mode': mode, 'size': size})
        # Python dict insertion order preserves first occurrence in the original table.
        unique = list(labels)
        counts = pd.Series(list(labels.values())).value_counts()
        if set(counts.index) != CLASSES or counts.min() < 2:
            raise ValueError('insufficient distinct content groups per class')
        fit_groups, held_groups = train_test_split(unique, test_size=holdout,
            random_state=seed, shuffle=True, stratify=[labels[group] for group in unique])
        if {labels[group] for group in fit_groups} != CLASSES or {labels[group] for group in held_groups} != CLASSES:
            raise ValueError('every class must occur in both content-group folds')
        held_set = set(held_groups)
        mask = train.id_code.map(groups).isin(held_set)
        fit, held = train.loc[~mask], train.loc[mask]
        staged_data = staging / 'data'
        for folder, frame in [('train_images', fit), ('test_images', held)]:
            (staged_data / folder).mkdir(parents=True)
            for identifier in frame.id_code:
                (spool / (identifier + '.png')).replace(staged_data / folder / (identifier + '.png'))
        fit.to_csv(staged_data / 'train.csv', index=False)
        held[['id_code']].to_csv(staged_data / 'test.csv', index=False)
        neutral = held.copy()
        neutral['diagnosis'] = 0
        neutral.to_csv(staged_data / 'sample_submission.csv', index=False)
        held.to_csv(evaluation / 'answers.csv', index=False)
        (evaluation / 'image_manifest.json').write_text(json.dumps(manifest, sort_keys=True))
        (evaluation / 'image_split.json').write_text(json.dumps({'seed': seed, 'holdout': holdout,
            'fit_ids': fit.id_code.tolist(), 'held_ids': held.id_code.tolist(),
            'fit_groups': fit_groups, 'held_groups': held_groups}, sort_keys=True))
        staged_data.replace(data)
    return {'aptos_images': True, 'algorithm': 'stratified_content_group_train_test_split',
        'seed': seed, 'holdout': holdout, 'fraction_unit': 'decoded_content_groups',
        'stratified_on': 'content_group_diagnosis', 'grouped_on': 'decoded_mode_shape_pixels',
        'fit_rows': len(fit), 'holdout_rows': len(held), 'fit_group_count': len(fit_groups),
        'holdout_group_count': len(held_groups), 'conflicting_label_groups': len(conflicts), 'label_column': 'diagnosis', 'id_column': 'id_code',
        'target_columns': ['diagnosis'],
        'filenames': {'train': 'train.csv', 'test': 'test.csv', 'sample': 'sample_submission.csv'}}


STARTER = '''import os
from pathlib import Path
import pandas as pd
from PIL import Image
def main():
    data = Path(os.environ['DATA_DIR'])
    train = pd.read_csv(data / 'train.csv', dtype={'id_code': str})
    test = pd.read_csv(data / 'test.csv', dtype={'id_code': str})
    sample = pd.read_csv(data / 'sample_submission.csv', dtype={'id_code': str})
    if set(test.id_code) != set(sample.id_code):
        raise ValueError('test/sample ID mismatch')
    for folder, frame in [('train_images', train), ('test_images', test)]:
        for identifier in frame.id_code:
            with Image.open(data / folder / (identifier + '.png')) as image:
                image.load()
    sample['diagnosis'] = int(train.diagnosis.mode().iloc[0])
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / 'submission.csv', index=False)
if __name__ == '__main__':
    main()
'''
