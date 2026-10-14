"""MLE-Bench task packages: public holdout search, one untouched official grade.

The first concrete task is deliberately explicit. Unsupported competition
schemas raise; they are never coerced to an insults-style task.
Preparation requires pandas and sklearn, not a provider credential.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from .contracts import canonical_sha256
from .retained_results import require, read_json, write_immutable, verify_seal
from .learning_evidence import source_ref, verify_ref

TASK = 'detecting-insults-in-social-commentary'
REVISION = '507f92e1138bb6e40dac5c6ee7a6758e6424bf97'
METRIC = {'name': 'auc', 'direction': 'maximize', 'role': 'public_validation'}

STARTER = '''import os
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def main():
    data = Path(os.environ['DATA_DIR'])
    output = Path(os.environ['OUTPUT_DIR']); output.mkdir(parents=True, exist_ok=True)
    train = pd.read_csv(data / 'train.csv')
    test = pd.read_csv(data / 'test.csv')
    vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True)
    x = vectorizer.fit_transform(train.Comment.fillna(''))
    model = LogisticRegression(C=2, max_iter=1000, random_state=0)
    model.fit(x, train.Insult)
    result = test.copy()
    result['Insult'] = model.predict_proba(vectorizer.transform(test.Comment.fillna('')))[:, 1]
    result.to_csv(output / 'submission.csv', index=False)

if __name__ == '__main__':
    main()
'''


def verify_assets(path):
    record = read_json(path)
    verify_seal(record, 'assets_sha256')
    for ref in record['refs']:
        verify_ref(ref)
    return record


def prepare(*, official_source: Path, prepared: Path, destination: Path, seed=0):
    import pandas as pd
    import yaml
    from sklearn.model_selection import GroupShuffleSplit
    official_source, prepared, destination = (p.resolve() for p in (official_source, prepared, destination))
    require(not destination.exists(), 'prepare destination must be new; no overwrite/re-split')
    source_task = official_source / 'mlebench/competitions' / TASK
    # Pin bytes rather than trust a directory/remote name. A source archive may
    # carry this revision marker instead of a .git checkout.
    if (official_source / '.git').exists():
        revision = subprocess.check_output(['git', '-C', str(official_source), 'rev-parse', 'HEAD'], text=True).strip()
        require(not subprocess.check_output(['git', '-C', str(official_source), 'status', '--porcelain'], text=True).strip(), 'official source is dirty')
    else:
        revision = (official_source / 'UPSTREAM_REVISION').read_text().strip()
    require(revision == REVISION, 'unsupported upstream revision; requalify explicitly')
    checksums = yaml.safe_load((source_task / 'checksums.yaml').read_text())
    for scope in ('public', 'private'):
        for name, digest in checksums[scope].items():
            asset = prepared / scope / name
            require(hashlib.md5(asset.read_bytes()).hexdigest() == digest, 'official prepared checksum mismatch: ' + str(asset))
    train = pd.read_csv(prepared / 'public/train.csv')
    fit, holdout = next(GroupShuffleSplit(n_splits=1, test_size=.2, random_state=seed).split(train, groups=train.Comment))
    require(not set(train.iloc[fit].Comment) & set(train.iloc[holdout].Comment), 'group split leaked comments')
    require(all(train.iloc[ids].Insult.nunique() == 2 for ids in (fit, holdout)), 'split must contain both labels')
    task = destination / 'tasks' / TASK
    data, evaluation = task / 'problem/data', task / 'evaluation'
    data.mkdir(parents=True); evaluation.mkdir()
    train.iloc[fit].to_csv(data / 'train.csv', index=False)
    train.iloc[holdout].drop(columns=['Insult']).to_csv(data / 'test.csv', index=False)
    train.iloc[holdout].to_csv(evaluation / 'answers.csv', index=False)
    starter = destination / 'starter'; starter.mkdir()
    (starter / 'run.py').write_text(STARTER)
    (task / 'problem/README.md').write_text(
        'MLE-Bench: detecting insults in social commentary. Predict probability of Insult from Comment.\n'
        'Implement main() in run.py. Read DATA_DIR/train.csv and DATA_DIR/test.csv, write '
        'OUTPUT_DIR/submission.csv retaining every test row and Comment, with finite Insult probabilities [0,1]. '
        'Optimize validation ROC AUC. Training and validation are grouped by exact Comment. '
        'Do not change the split, metric, row multiplicity or labels. The final unchanged code will fit '
        'the full original training set and predict the official test set using this same interface.\n')
    write_immutable(evaluation / 'split.json', {'seed': seed, 'algorithm': 'GroupShuffleSplit',
        'group': 'Comment', 'test_size': .2, 'fit_row_ids': fit.tolist(), 'holdout_row_ids': holdout.tolist(),
        'original_train': source_ref(prepared / 'public/train.csv')})
    # This entrypoint lives only on the controller. Neither it nor its answers
    # or official source path is mounted into a candidate.
    (evaluation / 'evaluator.py').write_text(
        'from swarmrsi.mlebench_tasks import score_public\n'
        'from pathlib import Path\n'
        f'score_public(Path({str(evaluation / "answers.csv")!r}), Path({str(official_source)!r}))\n')
    write_immutable(task / 'metadata.json', {'benchmark': 'mlebench', 'task_id': TASK, 'search_metric': METRIC,
                                          'custody': str(destination / 'custody.json')})
    refs = [source_ref(p) for p in sorted(prepared.rglob('*')) if p.is_file()]
    refs += [source_ref(p) for p in sorted((official_source / 'mlebench').rglob('*'))
             if p.is_file() and '__pycache__' not in p.parts and p.suffix != '.pyc']
    refs += [source_ref(p) for p in sorted(destination.rglob('*')) if p.is_file()]
    body = {'schema_version': 'swarmrsi-mlebench-task-assets-v1', 'task': TASK,
            'official_source': str(official_source), 'official_revision': revision,
            'prepared': str(prepared), 'search_metric': METRIC, 'refs': refs}
    write_immutable(destination / 'custody.json', {**body, 'assets_sha256': canonical_sha256(body)})
    return destination / 'custody.json'


def load_official(source, task=TASK):
    import sys
    sys.path.insert(0, str(source))
    import importlib
    module = importlib.import_module('mlebench.competitions.' + task + '.grade')
    require(Path(module.__file__).resolve().is_relative_to(source.resolve()), 'different official grader was imported')
    return module.grade


def validate_submission(submission, answers):
    """Check what the official grader does not: that no value is silently unusable.

    Row identity is the grader's own job -- mlebench rejects a submission whose ids do
    not match the answers -- and duplicating it here meant guessing which column is the
    identifier, which differs per competition and got insults wrong twice. The answers
    frame also carries bookkeeping the submission never has, such as Kaggle's Usage
    column marking the public/private split, so requiring its columns was simply wrong.

    What is checked is finiteness, because a NaN passes some graders as a neutral score
    rather than failing; an empty or wrong-length submission; and that there is a column
    beside the identifier at all, since a submission of nothing but ids carries no
    prediction and says so without having to know which column is the target.

    It does NOT require a numeric column. text-normalization submits id,after where
    `after` is the normalized token and the metric is exact-match accuracy over strings;
    demanding a number there rejected a valid 8.9M-row submission after the terminal
    program had already produced it.
    """
    import numpy as np
    require(len(submission) > 0, 'submission has no rows')
    require(len(submission) == len(answers), 'submission row count differs from the answers')
    require(len(submission.columns) > 1, 'submission carries no column beside the identifier')
    for column in submission.columns:
        values = submission[column]
        if values.dtype.kind in 'fciu':
            require(np.isfinite(values.to_numpy(dtype=float)).all(),
                    'non-finite values in column ' + str(column))


def score_public(answers_path, source, task=TASK, metric=METRIC['name']):
    import os
    import pandas as pd
    answers = pd.read_csv(answers_path)
    submission = pd.read_csv(Path(os.environ['OUTPUT_DIR']) / 'submission.csv')
    validate_submission(submission, answers)
    value = float(load_official(source, task)(submission, answers))
    Path('score.json').write_text(json.dumps({metric: value}, allow_nan=False))


def grade_terminal(source, prepared, submission, result, task=TASK):
    import pandas as pd
    load_official(source, task)
    from mlebench.registry import registry
    from mlebench.grade import grade_csv
    # Validation does not alter rows, order, scores, or the official grader.
    # The private answer table is not called the same thing in every competition: the
    # carve writes answers.csv, some trees carry test.csv, and text-normalization has
    # only answers.csv -- where this read used to raise FileNotFoundError AFTER the
    # terminal program had already produced a valid 8.9M-row submission, losing the
    # grade to a hard-coded name. The row count is all this pre-check needs, so the
    # table is located rather than named.
    private = prepared / 'private'
    answers = next((private / name for name in ('answers.csv', 'test.csv', 'private_test.csv',
                                                'private_answers.csv')
                    if (private / name).is_file()), None)
    if answers is None:
        answers = next((p for p in sorted(private.glob('*.csv'))
                        if 'sample' not in p.name.lower()), None)
    require(answers is not None, 'no private answer table to validate the submission against')
    validate_submission(pd.read_csv(submission), pd.read_csv(answers))
    competition = registry.set_data_dir(prepared.parent.parent).get_competition(task)
    report = grade_csv(submission, competition).to_dict()
    Path(result).write_text(json.dumps(report, indent=2, allow_nan=False, default=str))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest='operation', required=True)
    prep = sub.add_parser('prepare')
    for name in ('official-source', 'prepared', 'destination'):
        prep.add_argument('--' + name, type=Path, required=True)
    prep.add_argument('--seed', type=int, default=0)
    grade = sub.add_parser('grade')
    for name in ('source', 'prepared', 'submission', 'result'):
        grade.add_argument('--' + name, type=Path, required=True)
    grade.add_argument('--task', default=TASK)
    args = vars(parser.parse_args()); operation = args.pop('operation')
    if operation == 'prepare':
        print(prepare(**args))
    else:
        grade_terminal(**args)


if __name__ == '__main__':
    main()
