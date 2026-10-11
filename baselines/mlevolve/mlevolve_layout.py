"""Lay out our MLE-Bench task the way upstream MLEvolve expects to find it.

MLEvolve imports the real `mlebench` package (`registry`, `grade.validate_submission`)
and reads `<dataset_dir>/<competition>/prepared/{public,private}`. Our task ships the
same three files under different names, and the columns already match what this
competition's `prepare.py` asserts (train: Insult,Date,Comment; test: Date,Comment;
gold: Insult,Date,Comment), so this only renames and places them -- no data changes.

Scoring stays ours: MLEvolve's grader computes ROC AUC over (Comment, Insult), which
is the same metric our evaluator uses, against the same held-out answers.
"""
import argparse
import shutil
from pathlib import Path

import pandas as pd

parser = argparse.ArgumentParser()
parser.add_argument("--task-root", required=True, help="our task directory")
parser.add_argument("--dataset-dir", required=True, help="MLE-bench style dataset root to create")
parser.add_argument("--competition", default="detecting-insults-in-social-commentary")
parser.add_argument("--description", default=None, help="competition description.md from the mlebench package")
args = parser.parse_args()

task, out = Path(args.task_root), Path(args.dataset_dir) / args.competition / "prepared"
public, private = out / "public", out / "private"
public.mkdir(parents=True, exist_ok=True)
private.mkdir(parents=True, exist_ok=True)

train = pd.read_csv(task / "problem/data/train.csv")
test = pd.read_csv(task / "problem/data/test.csv")
gold = pd.read_csv(task / "evaluation/answers.csv")

assert list(train.columns) == ["Insult", "Date", "Comment"], list(train.columns)
assert list(test.columns) == ["Date", "Comment"], list(test.columns)
assert list(gold.columns) == ["Insult", "Date", "Comment"], list(gold.columns)
assert len(test) == len(gold), f"test {len(test)} != gold {len(gold)}"

train.to_csv(public / "train.csv", index=False)
test.to_csv(public / "test.csv", index=False)
sample = gold.copy()
sample["Insult"] = 0
sample.to_csv(public / "sample_submission_null.csv", index=False)
sample.to_csv(public / "sample_submission.csv", index=False)
# The registry points `answers` at private/test.csv while prepare.py writes
# gold_submission.csv; both names are produced so either lookup finds the labels.
gold.to_csv(private / "test.csv", index=False)
gold.to_csv(private / "gold_submission.csv", index=False)

desc = Path(args.description) if args.description else None
if desc and desc.is_file():
    shutil.copy2(desc, public / "description.md")
else:
    (public / "description.md").write_text((task / "problem/README.md").read_text())

print(f"train {len(train)} | test {len(test)} | gold {len(gold)} -> {out}")
