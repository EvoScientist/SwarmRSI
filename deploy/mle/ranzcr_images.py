"""Native RANZCR CLiP image contract: official row-split carve, unchanged official grade.

The official prepare.py keeps the Kaggle layout: public/train.csv (StudyInstanceUID,
eleven binary label columns, PatientID), public/train_annotations.csv (segmentation
polylines for some training images), public/sample_submission.csv (StudyInstanceUID
and the nine CLASSES of the pinned classes.py), and public/train/ and public/test/
holding <StudyInstanceUID>.jpg. There is no public test table. The official grader
scores the mean column-wise ROC AUC over those nine CLASSES only.

The generic carve cannot build this package: train_annotations.csv also matches the
"train" table pattern, the whole train/ directory (held images included) would be
copied under train/, and held images would never reach test/. This adapter projects
the images, annotations and tables explicitly.

The carve copies the official prepare.py split: a random row split with test_size 0.1
and the given seed, NOT grouped by patient. The hold-out exists to estimate the
official score, and the official test shares ~96% of its patients with the official
train (1,419 of 1,472 on the prepared data); a patient-grouped carve would select
candidates by a criterion the official grade does not reward.
"""
import json
from pathlib import Path
import re
import shutil
import tempfile

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import train_test_split

ID = 'StudyInstanceUID'
PATIENT = 'PatientID'
# Exactly the pinned official classes.py; the grader scores these and nothing else.
CLASSES = [
    'ETT - Abnormal', 'ETT - Borderline', 'ETT - Normal',
    'NGT - Abnormal', 'NGT - Borderline', 'NGT - Incompletely Imaged', 'NGT - Normal',
    'CVC - Abnormal', 'CVC - Borderline',
]
# Published in train.csv but not scored by the official grader.
UNSCORED = ['CVC - Normal', 'Swan Ganz Catheter Present']
TRAIN_COLUMNS = [ID, *CLASSES, *UNSCORED, PATIENT]
ANNOTATION_COLUMNS = [ID, 'label', 'data']
# prepare.py: train_test_split(old_train, test_size=0.1, random_state=0).
OFFICIAL_HOLDOUT = 0.1

CONTRACT = '''
Execution data contract: DATA_DIR/train.csv has StudyInstanceUID, eleven binary
label columns (ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal,
NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal,
CVC - Borderline, CVC - Normal, Swan Ganz Catheter Present) and PatientID.
Training chest x-rays are DATA_DIR/train/<StudyInstanceUID>.jpg.
DATA_DIR/train_annotations.csv (StudyInstanceUID,label,data) holds segmentation
polylines for some training images only; it is optional auxiliary input.
There is no test table: predict DATA_DIR/test/<StudyInstanceUID>.jpg for exactly
the IDs in DATA_DIR/sample_submission.csv. Write OUTPUT_DIR/submission.csv with one
unique row per requested ID and the columns StudyInstanceUID plus the nine scored
classes ETT - Abnormal, ETT - Borderline, ETT - Normal, NGT - Abnormal,
NGT - Borderline, NGT - Incompletely Imaged, NGT - Normal, CVC - Abnormal,
CVC - Borderline, each a finite probability in [0,1]. CVC - Normal and Swan Ganz
Catheter Present are NOT scored (the competition description mentions eleven; the
official MLE-Bench grader scores nine); extra columns are ignored. The metric is the
unchanged official one: ROC AUC per scored class, averaged, maximize.
Search validation mirrors the official MLE-Bench split: a random 10% of the
official public train rows is held out, not grouped by patient, exactly as the
official test was cut, so a patient can have images on both sides, as in the
official test. Held rows' labels, annotations and PatientIDs are controller-only.
The split is frozen before scoring. The same program and path
contract run on the unchanged full official public tree for terminal refit and
prediction, where test/ holds the official test images. Official test images are
not in the search data root.
'''


def ordinary(path, *, directory=False):
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'ordinary {"directory" if directory else "file"} required: {path.name}')


def identifiers(frame):
    if frame.empty or ID not in frame or frame[ID].isna().any():
        raise ValueError('nonempty StudyInstanceUIDs required')
    values = frame[ID].tolist()
    if any(not isinstance(v, str) or not re.fullmatch(r'[0-9][0-9.]*[0-9]', v) for v in values):
        raise ValueError('unsafe StudyInstanceUID')
    if len(set(values)) != len(values):
        raise ValueError('duplicate StudyInstanceUIDs')
    return values


def _read(path, columns):
    ordinary(path)
    # Everything as text: ids, patient ids and polylines survive byte-for-byte.
    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    if list(frame.columns) != columns:
        raise ValueError(f'explicit native RANZCR schema required for {path.name}')
    return frame


def binary(frame, columns):
    values = frame[columns].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or not np.isin(values, (0, 1)).all():
        raise ValueError('labels must be binary 0/1')
    return values


def probabilities(frame):
    values = frame[CLASSES].apply(pd.to_numeric, errors='raise').to_numpy(dtype=float)
    if not np.isfinite(values).all() or not ((values >= 0) & (values <= 1)).all():
        raise ValueError('probabilities must be finite in [0,1]')
    return values


def validate_probabilities(submission, answers):
    """Checked before the unchanged official grader; extra columns are dropped as it ignores them."""
    if list(answers.columns) != [ID, *CLASSES]:
        raise ValueError('answers must be StudyInstanceUID plus the scored classes')
    missing = [c for c in [ID, *CLASSES] if c not in submission.columns]
    if missing:
        raise ValueError(f'submission lacks columns {missing}')
    submission = submission[[ID, *CLASSES]].copy()
    submission[ID] = submission[ID].astype(str)
    if set(identifiers(submission)) != set(identifiers(answers)):
        raise ValueError('submission ID set mismatch')
    probabilities(submission)
    submission[CLASSES] = submission[CLASSES].apply(pd.to_numeric)
    return submission.set_index(ID).loc[answers[ID]].reset_index()


def inventory(directory, ids):
    ordinary(directory, directory=True)
    if {p.name for p in directory.iterdir()} != {identifier + '.jpg' for identifier in ids}:
        raise ValueError('JPEG inventory differs from declared IDs')


def _copy_image(source, target):
    ordinary(source)
    # Header only: decoding 27k full-size radiographs would dominate the build.
    with Image.open(source) as image:
        if image.format != 'JPEG':
            raise ValueError('JPEG required')
    shutil.copyfile(source, target)


def carve_ranzcr(prepared, destination, *, seed, holdout=OFFICIAL_HOLDOUT):
    public = prepared / 'public'
    ordinary(prepared, directory=True)
    ordinary(public, directory=True)
    train = _read(public / 'train.csv', TRAIN_COLUMNS)
    annotations = _read(public / 'train_annotations.csv', ANNOTATION_COLUMNS)
    sample = _read(public / 'sample_submission.csv', [ID, *CLASSES])
    train_ids, test_ids = identifiers(train), identifiers(sample)
    binary(train, CLASSES + UNSCORED)
    if (train[PATIENT] == '').any():
        raise ValueError('every training row needs a PatientID')
    if set(train_ids) & set(test_ids):
        raise ValueError('official public train/test ID overlap')
    if not set(annotations[ID]) <= set(train_ids):
        raise ValueError('annotations reference images outside the public train')
    inventory(public / 'train', train_ids)
    inventory(public / 'test', test_ids)

    # Same call as the official prepare.py (unstratified, shuffled rows).
    fit, held = train_test_split(train, test_size=holdout, random_state=seed)
    fit, held = fit.sort_index(), held.sort_index()
    # Every scored AUC needs both classes on the held side, or the grade is undefined.
    for frame, side in ((fit, 'fit'), (held, 'held')):
        values = binary(frame, CLASSES)
        if not ((values.max(axis=0) == 1) & (values.min(axis=0) == 0)).all():
            raise ValueError(f'every scored class needs positives and negatives on the {side} side')

    data, evaluation = destination / 'problem/data', destination / 'evaluation'
    if data.exists():
        raise ValueError('refusing to overwrite an existing RANZCR projection')
    data.parent.mkdir(parents=True, exist_ok=True)
    evaluation.mkdir(parents=True, exist_ok=True)
    fit_ids = set(fit[ID])
    with tempfile.TemporaryDirectory(prefix='.ranzcr-projection-', dir=destination) as temporary:
        staged = Path(temporary) / 'data'
        for folder, frame in (('train', fit), ('test', held)):
            (staged / folder).mkdir(parents=True)
            for identifier in frame[ID]:
                _copy_image(public / 'train' / (identifier + '.jpg'), staged / folder / (identifier + '.jpg'))
        fit.to_csv(staged / 'train.csv', index=False)
        kept = annotations[annotations[ID].isin(fit_ids)]
        kept.to_csv(staged / 'train_annotations.csv', index=False)
        answers = held[[ID, *CLASSES]]
        neutral = answers.copy()
        neutral[CLASSES] = 0  # as the official prepare.py writes its sample
        neutral.to_csv(staged / 'sample_submission.csv', index=False)
        answers.to_csv(evaluation / 'answers.csv', index=False)
        (evaluation / 'image_split.json').write_text(json.dumps({
            'seed': seed, 'holdout': holdout, 'grouped_on': None,
            'fit_ids': fit[ID].tolist(), 'held_ids': held[ID].tolist()}, sort_keys=True))
        staged.replace(data)
    return {'ranzcr_images': True, 'algorithm': 'train_test_split', 'seed': seed,
            'holdout': holdout, 'fraction_unit': 'rows', 'grouped_on': None,
            'stratified_on': None, 'fit_rows': len(fit), 'holdout_rows': len(held),
            'shared_patients': len(set(fit[PATIENT]) & set(held[PATIENT])),
            'annotation_rows_kept': len(kept), 'annotation_rows_withheld': len(annotations) - len(kept),
            'label_column': None, 'id_column': ID, 'target_columns': list(CLASSES),
            'unscored_label_columns': list(UNSCORED),
            'filenames': {'train': 'train.csv', 'test': None, 'sample': 'sample_submission.csv'}}


STARTER = '''"""Label-frequency prior for the nine scored RANZCR classes; a floor, not a method."""
import os
from pathlib import Path
import pandas as pd
CLASSES = __CLASSES__
def main():
    data = Path(os.environ['DATA_DIR'])
    train = pd.read_csv(data / 'train.csv', dtype={'StudyInstanceUID': str, 'PatientID': str})
    sample = pd.read_csv(data / 'sample_submission.csv', dtype={'StudyInstanceUID': str})
    # Check the image contract without decoding every radiograph.
    for folder, frame in (('train', train), ('test', sample)):
        present = {p.name for p in (data / folder).iterdir()}
        missing = [i for i in frame.StudyInstanceUID if i + '.jpg' not in present]
        if missing:
            raise FileNotFoundError(f'{len(missing)} images missing under {folder}/')
    submission = sample[['StudyInstanceUID']].copy()
    for column in CLASSES:
        submission[column] = float(train[column].mean())
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / 'submission.csv', index=False)
if __name__ == '__main__':
    main()
'''.replace('__CLASSES__', repr(CLASSES))
