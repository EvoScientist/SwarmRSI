"""Native MLSP record projection. No search policy or private-test feedback."""
import csv
import hashlib
import json
from pathlib import Path
import re
import shutil
import wave

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

SPECIES = set(range(19))
# Public-only source/format/visual audit, 2026-09-22. These contain segment
# mosaics, not species annotations. Unknown replacements require a new audit.
GLOBAL_BITMAP_SHA256 = {
    'segment_clusters.bmp': '3a48f1ccd30d58c6180704bdf99d57e1314b4937f130f7c28211be47552210df',
    'segment_mosaic.bmp': 'ce13367006ceef1acce1b7d2cb6b03ccedef39af28f09baf4aea3d4c6b04ea1b',
}
CONTRACT = '''
Execution data contract: DATA_DIR uses the official native essential_data and
supplemental_data layout. CVfolds_2.txt fold 0 identifies fit recordings; fold 1
identifies prediction recordings. rec_labels_test_hidden.txt is ragged: a row
containing only rec_id means KNOWN all-negative; '?' means UNKNOWN, never a
negative training example. Only fit species labels are visible. The split unit
is an entire recording, not a segment. Internal IDs preserve the upstream IDs
and may have gaps. All 19 species (0..18) must remain in the output, including
species without fit positives. Write OUTPUT_DIR/submission.csv with columns
Id,Probability and exactly the IDs from DATA_DIR/sample_submission.csv;
Id = rec_id*100 + species_id. Probabilities must be finite and in [0,1]. The
official metric is ONE flattened AUC over all record/species entries, NOT macro
per-species AUC. The identical native loader must work at terminal refit on the
unchanged official public tree. Held species answers are controller-only.
The supplied histogram and segment files have a generic upstream header: parse
their BODY positionally (100 histogram features; segment_id plus 38 segment
features; segment_id plus 4 rectangle coordinates, possibly trailing comma).
Empty segment bags are valid records: retain each record, represent missing
aggregates with count=0 and fixed zero features, and never discard its labels.
Raw audio remains available for all fit and prediction records.
Feature provenance: supplied automatic segmentation uses upstream auxiliary
bird-versus-noise pixel supervision, NOT species labels. The fixed 100-cluster
codebook used original train+test UNLABELED segments and is transductive; these
are not strictly inductive raw features. Global visualizations contain the
original unlabeled segments. Direct manual annotation images are projected to
fit records only here; official terminal assets remain unchanged. Never fit a
representation using hidden species targets. Clip-level validation does not
claim unseen-site/session generalization. Frequency priors can score above .5
under flattened AUC without acoustic discrimination.
'''


def integer(value):
    if not re.fullmatch(r'0|[1-9][0-9]*', str(value)):
        raise ValueError('expected nonnegative integer ID')
    return int(value)


def ordinary(path, directory=False):
    if path.is_symlink() or not (path.is_dir() if directory else path.is_file()):
        raise ValueError(f'ordinary {"directory" if directory else "file"} required: {path.name}')


def rows(path):
    ordinary(path)
    with path.open(newline='') as stream:
        content = list(csv.reader(stream))
    if not content or not content[0] or any(not row for row in content[1:]):
        raise ValueError('missing header or blank table row')
    return content[0], content[1:]


def keyed(body, parse):
    result = {}
    for row in body:
        rec = integer(row[0])
        if rec in result:
            raise ValueError('duplicate record ID')
        result[rec] = parse(row[1:])
    return result


def label_set(fields):
    if fields == ['?']:
        return None
    # Official empty sets have no second field. Reject silent empty tokens.
    values = [integer(value) for value in fields]
    if len(set(values)) != len(values) or not set(values) <= SPECIES:
        raise ValueError('duplicate or invalid species')
    return set(values)


def load_records(data):
    essential = data / 'essential_data'
    ordinary(data, True)
    ordinary(essential, True)
    _, body = rows(essential / 'CVfolds_2.txt')
    def fold(fields):
        if len(fields) != 1 or fields[0] not in ('0', '1'):
            raise ValueError('invalid fold')
        return int(fields[0])
    folds = keyed(body, fold)
    _, body = rows(essential / 'rec_id2filename.txt')
    def filename(fields):
        if len(fields) != 1 or not re.fullmatch(r'[A-Za-z0-9_-]+', fields[0]):
            raise ValueError('unsafe filename')
        return fields[0]
    names = keyed(body, filename)
    _, body = rows(essential / 'rec_labels_test_hidden.txt')
    labels = keyed(body, label_set)
    _, species = rows(essential / 'species_list.txt')
    if len(species) != 19 or {integer(row[0]) for row in species} != SPECIES:
        raise ValueError('expected exactly 19 species')
    if not folds or set(folds) != set(names) or set(folds) != set(labels):
        raise ValueError('record table ID join mismatch')
    if len(set(names.values())) != len(names):
        raise ValueError('duplicate filename')
    if any((labels[rec] is None) != (folds[rec] == 1) for rec in folds):
        raise ValueError('fold/unknown-label mismatch')
    ordinary(data / 'sample_submission.csv')
    sample = pd.read_csv(data / 'sample_submission.csv')
    expected = pd.DataFrame({'Id': [rec * 100 + species for rec in sorted(folds)
        if folds[rec] == 1 for species in range(19)], 'Probability': 0.0})
    validate_probabilities(sample, expected)
    return folds, names, labels


def validate_probabilities(submission, answers):
    normalized = []
    for frame in (submission, answers):
        if list(frame.columns) != ['Id', 'Probability'] or frame.empty:
            raise ValueError('expected nonempty Id,Probability table')
        ids = pd.to_numeric(frame.Id, errors='raise').to_numpy(dtype=float)
        probabilities = pd.to_numeric(frame.Probability, errors='raise').to_numpy(dtype=float)
        if (not np.isfinite(ids).all() or (ids < 0).any() or (ids > 2**53).any()
                or not (ids == np.floor(ids)).all()):
            raise ValueError('invalid integer ID')
        ids = ids.astype(np.int64)
        if len(set(ids)) != len(ids) or not set(ids % 100) <= SPECIES:
            raise ValueError('duplicate ID or invalid species')
        if not np.isfinite(probabilities).all() or not ((probabilities >= 0) & (probabilities <= 1)).all():
            raise ValueError('probabilities must be finite in [0,1]')
        normalized.append(pd.DataFrame({'Id': ids, 'Probability': probabilities}))
    left, right = normalized
    if set(left.Id) != set(right.Id):
        raise ValueError('submission ID set mismatch')
    for rec in set(right.Id // 100):
        if set(right.Id[right.Id // 100 == rec] % 100) != SPECIES:
            raise ValueError('incomplete 19-species record')
    return left.set_index('Id').loc[right.Id].reset_index()


def inventory(directory, expected, *, subset=False):
    ordinary(directory, True)
    actual = {path.name for path in directory.iterdir()}
    mismatch = not actual <= set(expected) if subset else actual != set(expected)
    if mismatch:
        raise ValueError(f'unknown or missing asset in {directory.name}')
    for path in directory.iterdir():
        ordinary(path)


def carve_records(prepared, destination, *, seed, holdout):
    public = prepared / 'public'
    folds, names, labels = load_records(public)
    ordinary(public / 'description.md')
    essential, supplemental = public / 'essential_data', public / 'supplemental_data'
    ordinary(supplemental, True)
    if {p.name for p in public.iterdir()} != {'essential_data', 'supplemental_data', 'sample_submission.csv', 'description.md'}:
        raise ValueError('unknown top-level asset')
    if {p.name for p in essential.iterdir()} != {'CVfolds_2.txt', 'rec_id2filename.txt', 'rec_labels_test_hidden.txt', 'species_list.txt', 'src_wavs'}:
        raise ValueError('unknown essential asset')
    tables = {'histogram_of_segments.txt': 101, 'segment_features.txt': 40, 'segment_rectangles.txt': 6}
    image_dirs = ('spectrograms', 'filtered_spectrograms', 'supervised_segmentation', 'segmentation_examples')
    if {p.name for p in supplemental.iterdir()} != set(tables) | set(image_dirs) | set(GLOBAL_BITMAP_SHA256):
        raise ValueError('unknown or missing supplemental asset')
    for filename, digest in GLOBAL_BITMAP_SHA256.items():
        path = supplemental / filename
        ordinary(path)
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError('global bitmap lacks source/format attestation')
    inventory(essential / 'src_wavs', [name + '.wav' for name in names.values()])
    decoded = set()
    for name in names.values():
        with wave.open(str(essential / 'src_wavs' / (name + '.wav')), 'rb') as stream:
            if (stream.getnchannels(), stream.getsampwidth(), stream.getframerate(), stream.getnframes()) != (1, 2, 16000, 160000):
                raise ValueError('unexpected audio format')
            content = stream.readframes(stream.getnframes())
            if len(content) != 320000:
                raise ValueError('truncated audio')
            digest = hashlib.sha256(content).hexdigest()
            if digest in decoded:
                raise ValueError('duplicate decoded audio requires a grouped protocol')
            decoded.add(digest)
    for directory in image_dirs:
        inventory(supplemental / directory, [name + '.bmp' for name in names.values()], subset=directory == 'segmentation_examples')
    parsed, keys = {}, {}
    for filename, width in tables.items():
        header, body = rows(supplemental / filename)
        seen = set()
        for row in body:
            fields = row[:-1] if filename == 'segment_rectangles.txt' and row[-1] == '' else row
            if len(fields) != width:
                raise ValueError('invalid feature width')
            rec = integer(fields[0])
            key = rec if width == 101 else (rec, integer(fields[1]))
            if rec not in folds or key in seen:
                raise ValueError('unknown or duplicate feature key')
            seen.add(key)
            values = np.asarray(fields[1:] if width == 101 else fields[2:], dtype=float)
            if not np.isfinite(values).all():
                raise ValueError('nonfinite feature')
            if width == 101 and ((values < 0).any() or (values.sum() != 0 and abs(values.sum() - 1) > 1e-4)):
                raise ValueError('invalid histogram')
        parsed[filename] = (header, body)
        keys[filename] = seen
    if keys['histogram_of_segments.txt'] != set(folds) or keys['segment_features.txt'] != keys['segment_rectangles.txt']:
        raise ValueError('feature key join mismatch')
    population = sorted(rec for rec in folds if folds[rec] == 0)
    fit, held = train_test_split(population, test_size=holdout, random_state=seed, shuffle=True)
    fit, held = sorted(fit), sorted(held)
    keep, fit_set = set(population), set(fit)
    data, evaluation = destination / 'problem/data', destination / 'evaluation'
    (data / 'essential_data/src_wavs').mkdir(parents=True, exist_ok=False)
    (data / 'supplemental_data').mkdir()
    evaluation.mkdir(exist_ok=False)
    def write_table(path, header, body):
        with path.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(header)
            writer.writerows(body)
    for filename, body in {
        'CVfolds_2.txt': [[rec, 0 if rec in fit_set else 1] for rec in population],
        'rec_id2filename.txt': [[rec, names[rec]] for rec in population],
        'rec_labels_test_hidden.txt': [[rec, *sorted(labels[rec])] if rec in fit_set else [rec, '?'] for rec in population],
    }.items():
        write_table(data / 'essential_data' / filename, rows(essential / filename)[0], body)
    shutil.copyfile(essential / 'species_list.txt', data / 'essential_data/species_list.txt')
    for rec in population:
        filename = names[rec] + '.wav'
        shutil.copyfile(essential / 'src_wavs' / filename, data / 'essential_data/src_wavs' / filename)
    for filename, (header, body) in parsed.items():
        write_table(data / 'supplemental_data' / filename, header, [row for row in body if integer(row[0]) in keep])
    for directory in image_dirs:
        target = data / 'supplemental_data' / directory
        target.mkdir()
        allowed = {names[rec] for rec in (fit if directory == 'segmentation_examples' else population)}
        for path in (supplemental / directory).iterdir():
            if path.stem in allowed:
                shutil.copyfile(path, target / path.name)
    for filename in GLOBAL_BITMAP_SHA256:
        shutil.copyfile(supplemental / filename, data / 'supplemental_data' / filename)
    answers = pd.DataFrame([(rec * 100 + species, int(species in labels[rec])) for rec in held for species in range(19)], columns=['Id', 'Probability'])
    sample = answers.copy()
    sample['Probability'] = 0.0
    validate_probabilities(sample, answers)
    sample.to_csv(data / 'sample_submission.csv', index=False)
    answers.to_csv(evaluation / 'answers.csv', index=False)
    split = {'algorithm': 'train_test_split', 'seed': seed, 'holdout': holdout,
        'fraction_unit': 'record', 'stratified_on': None, 'fit_rows': len(fit), 'holdout_rows': len(held),
        'record_multilabel': True, 'species_count': 19, 'id_column': 'Id', 'target_columns': ['Probability'],
        'metric_aggregation': 'single_flattened_auc', 'feature_provenance': 'fixed_auxiliary_segmentation_and_transductive_unlabeled_codebook',
        'filenames': {'train': None, 'test': None, 'sample': 'sample_submission.csv'}}
    (evaluation / 'record_split.json').write_text(json.dumps({**split, 'fit_rec_ids': fit, 'held_rec_ids': held}, sort_keys=True))
    return split


STARTER = '''import csv
import os
from pathlib import Path
import pandas as pd

def main():
    data = Path(os.environ['DATA_DIR'])
    with (data / 'essential_data/CVfolds_2.txt').open(newline='') as stream:
        rows = list(csv.reader(stream))[1:]
    fit = {int(rec) for rec, fold in rows if fold == '0'}
    positives = [0] * 19
    count = 0
    with (data / 'essential_data/rec_labels_test_hidden.txt').open(newline='') as stream:
        for fields in list(csv.reader(stream))[1:]:
            if int(fields[0]) not in fit or fields[1:] == ['?']:
                continue
            labels = [int(value) for value in fields[1:]]
            if len(set(labels)) != len(labels) or not set(labels) <= set(range(19)):
                raise ValueError('invalid species labels')
            count += 1
            for species in labels:
                positives[species] += 1
    if count == 0:
        raise ValueError('no known fit records')
    sample = pd.read_csv(data / 'sample_submission.csv')
    sample['Probability'] = [(positives[int(identifier) % 100] + 1) / (count + 2) for identifier in sample.Id]
    output = Path(os.environ['OUTPUT_DIR'])
    output.mkdir(parents=True, exist_ok=True)
    sample.to_csv(output / 'submission.csv', index=False)

if __name__ == '__main__':
    main()
'''
