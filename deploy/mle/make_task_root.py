"""Build one MLE-Bench task package from a prepared competition.

The existing builder in `mlebench_tasks.py` hard-codes the insults competition: its
task id, its metric name, its split column, its starter. This is the same design
generalised over the Lite panel, with three properties kept exactly as they were.

The search never sees the official test. A candidate is scored on a slice carved out
of the official *public* training set; the official public test and its private
answers go to `private/`, which no candidate mount reaches. The controller uses them
only for the final, once-per-run official grade.

The metric is the competition's own. Rather than restate a formula per task, the
evaluator calls the grade function mlebench ships for that competition, so a
generated package cannot drift from the official scoring. The direction comes from
the leaderboard, not from a guess: spooky-author is log loss, lower-is-better, while
insults is AUC, higher-is-better.

The starter is deliberately weak -- the uniform prior over training-label frequencies
-- because the harness measures improvement against the root, and a strong root would
flatter nothing. Its score is measured once and pinned in the package, so
regenerating the package is checked against it.
"""
from __future__ import annotations

import argparse
import itertools
import json
import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
from swarmrsi.contracts import canonical_sha256  # noqa: E402
from swarmrsi.learning_evidence import source_ref  # noqa: E402
from swarmrsi.mlebench_tasks import verify_assets  # noqa: E402
from swarmrsi.retained_results import write_immutable  # noqa: E402

STARTER = '''"""Uniform prior baseline: predict the training-set label frequencies for every row.

A floor to improve on, not a method. It reads this competition's own training table and
sample submission from DATA_DIR and writes OUTPUT_DIR/submission.csv with the same
rows, in the same order, as that sample.

It deliberately does not read a test table. An image competition has none -- its test
set is a directory of pictures and the sample submission is what enumerates the test
rows -- and the same code has to run against the official tree at the end.
"""
import os
from pathlib import Path

import pandas as pd


def _read(path):
    """Read a competition table by whatever format it ships in."""
    if path.suffix.lower() == ".json" or path.name.lower().endswith(".json.zip"):
        return pd.read_json(path)
    return pd.read_csv(path)


def main():
    data = Path(os.environ["DATA_DIR"])
    sample = _read(data / "__SAMPLE__")
    train = _read(data / "__TRAIN__")
    submission = sample.copy()
    targets = __TARGETS__
    for column in targets:
        if column in train.columns:
            values = train[column]
            # A competition graded on labels wants a label, not an average of them.
            # Integer and categorical targets get the training mode; only a genuinely
            # continuous target, or a probability, gets the mean. Predicting 4.7 where
            # the grader expects a class is what it rejected as an invalid submission.
            if values.dtype.kind in "fc":
                submission[column] = float(values.mean())
            else:
                submission[column] = values.mode().iloc[0]
        else:
            # One-hot targets: the column name is a class label of the label column.
            label = next((c for c in train.columns if set(map(str, train[c].unique())) >= set(map(str, targets))), None)
            submission[column] = float((train[label].astype(str) == column).mean()) if label else 1.0 / len(targets)
    output = Path(os.environ["OUTPUT_DIR"])
    output.mkdir(parents=True, exist_ok=True)
    submission.to_csv(output / "submission.csv", index=False)


if __name__ == "__main__":
    main()
'''

EVALUATOR = '''"""Public validation scorer: the controller-held slice only, never the official test."""
import json
import os
from pathlib import Path

import pandas as pd

from mlebench.registry import registry

TASK_ROOT = Path(__file__).resolve().parent.parent
metadata = json.loads((TASK_ROOT / "metadata.json").read_text())
competition = registry.set_data_dir(Path(metadata["mlebench_data_dir"])).get_competition(metadata["task_id"])
id_dtype = {"id": str} if metadata.get("public_split", {}).get("image_projection") else None
if metadata.get("public_split", {}).get("aptos_images"):
    id_dtype = {"id_code": str}
if metadata.get("public_split", {}).get("ranzcr_images"):
    id_dtype = {"StudyInstanceUID": str}
answers = pd.read_csv(TASK_ROOT / "evaluation/answers.csv", dtype=id_dtype)
submission = pd.read_csv(Path(os.environ["OUTPUT_DIR"]) / "submission.csv", dtype=id_dtype)
if metadata.get("public_split", {}).get("paired_pages"):
    from paired_pages import validate_pixels
    submission = validate_pixels(submission, answers)
if metadata.get("public_split", {}).get("record_multilabel"):
    from bird_records import validate_probabilities
    submission = validate_probabilities(submission, answers)
if metadata.get("public_split", {}).get("image_projection"):
    from image_projection import validate_probabilities
    submission = validate_probabilities(submission, answers)
if metadata.get("public_split", {}).get("aptos_images"):
    from aptos_images import validate_diagnosis
    submission = validate_diagnosis(submission, answers)
if metadata.get("public_split", {}).get("ranzcr_images"):
    from ranzcr_images import validate_probabilities
    submission = validate_probabilities(submission, answers)
if len(submission) != len(answers):
    raise ValueError(f"submission has {len(submission)} rows against {len(answers)} held-out rows")
score = competition.grader(submission, answers)
if score is None:
    raise ValueError("the official grader rejected this submission")
Path("score.json").write_text(json.dumps({metadata["search_metric"]["name"]: float(score)}))
'''


def _is_table_archive(path: Path) -> bool:
    """True when a .zip is just a compressed copy of one table.

    An archive holding a single CSV or JSON is the table again and must not be copied
    into the package, since it would carry the unsplit training set. An archive holding
    pictures is the competition's payload and must be.
    """
    import zipfile

    try:
        with zipfile.ZipFile(path) as archive:
            entries = [name for name in archive.namelist() if not name.endswith("/")]
    except (zipfile.BadZipFile, OSError):
        return False
    return len(entries) == 1 and Path(entries[0]).suffix.lower() in (".csv", ".json", ".tsv")


def _table_path(directory: Path, kind: str):
    """Find and read this competition's train or sample-submission table.

    MLE-Bench passes each competition's own files through untouched, so the names are
    whatever Kaggle used: train.csv, train.json, en_train.csv.zip; sample_submission.csv,
    sampleSubmission.csv, sample_submission_null.csv. Matching a fixed name covered one
    competition. The file is located by kind and read by suffix instead, and an
    ambiguous match is an error rather than an arbitrary pick.
    """
    import pandas as pd

    # "labels" is the training table for a competition whose examples are image files:
    # dog-breed ships labels.csv beside train/ and test/, with no file called train.
    patterns = {"train": ("train", "labels"), "test": ("test",),
                "sample": ("sample", "samplesubmission")}[kind]
    # "sample_submission" contains neither "train" nor "test", but a competition whose
    # sample is named sampleSubmission would still be picked up by the test pattern, so
    # both kinds exclude it explicitly.
    excluded = {"train": ("test", "sample"), "test": ("sample", "labels"), "sample": ()}[kind]
    matches = [p for p in sorted(directory.iterdir())
               if p.is_file()
               and any(word in p.name.lower().replace("_", "") for word in patterns)
               and not any(word in p.name.lower() for word in excluded)
               and p.suffix.lower() in (".csv", ".json", ".zip")]
    if not matches:
        return None
    if len(matches) > 1:
        # A competition often ships both train.json and train.zip. The uncompressed
        # copy is the same data and cheaper to read, so it wins; only a genuine tie
        # between two uncompressed tables is ambiguous.
        plain = [p for p in matches if p.suffix.lower() != ".zip"]
        matches = plain or matches
    # An archive of pictures is payload, never a table, however it is named. Dropping
    # these here means test.zip is not mistaken for aerial-cactus's test table and is
    # copied into the package as the competition's own test images, so the package
    # mirrors the official tree the terminal step will hand the same program.
    matches = [p for p in matches if p.suffix.lower() != ".zip" or _is_table_archive(p)]
    if not matches:
        return None
    if len(matches) > 1:
        raise SystemExit(f"ambiguous {kind} table in {directory}: {[p.name for p in matches]}")
    return matches[0]


def _read_table(directory: Path, kind: str, optional: bool = False):
    """Read a competition table, or None when it has none of that kind."""
    import pandas as pd

    path = _table_path(directory, kind)
    if path is None:
        if optional:
            return None
        raise SystemExit(f"no {kind} table in {directory}")
    try:
        if path.suffix.lower() == ".json" or path.name.lower().endswith(".json.zip"):
            return pd.read_json(path)
        return pd.read_csv(path)
    except ValueError:
        # An image competition ships its test set as an archive of pictures, not as a
        # table. pandas reads a zip holding one CSV and refuses one holding thousands
        # of JPEGs, and that refusal is the signal that this competition has no test
        # table to read the submission schema from.
        if optional:
            return None
        raise


def _write_table(frame, path: Path):
    """Write a carved table back under the competition's own filename."""
    if path.suffix.lower() == ".json" or path.name.lower().endswith(".json.zip"):
        frame.to_json(path, orient="records")
    else:
        frame.to_csv(path, index=False)


def _composite_identifier(sample, official_test, given: set):
    """A submission id built by joining columns the official test does supply.

    text-normalization submits `id` = "<sentence_id>_<token_id>", and no column of its
    official test carries that string. Settling identifiers by membership alone finds
    none, files `id` among the columns to PREDICT, and then keys the answers frame to a
    range index -- every held-out row labelled with the wrong token. Recovering the
    identifier is what lets a competition shaped this way be carved at all.

    Only an exact, row-for-row reconstruction counts: a separator that merely looks
    plausible would key the holdout to ids the competition's grader cannot match. The
    search runs on a prefix because the frames have millions of rows and the candidate
    orderings are dozens, and whatever the prefix proposes is then confirmed in full.
    """
    # Element-wise str: insults' partly-missing Date column survives astype(str) as float
    # NaN under the image's pandas, and the join then raises.
    columns = [c for c in official_test.columns if c in given]
    probe = min(len(official_test), 2000)
    for column in (c for c in sample.columns if c not in given):
        values = sample[column].astype(str).to_numpy()
        if len(values) != len(official_test):
            continue
        # An identifier is unique per row. Without this the target column of a wide
        # table (tabular-dec's Cover_Type against 55 test columns) was searched over
        # every 3-column permutation and the build ran for hours.
        if len(set(values)) != len(values):
            continue
        for size in (2, 3):
            for parts in itertools.permutations(columns, size):
                for separator in ("_", "-", "/", ".", "|"):
                    head = official_test[list(parts)].head(probe).map(str).agg(separator.join, axis=1)
                    if not (head.to_numpy() == values[:probe]).all():
                        continue
                    full = official_test[list(parts)].map(str).agg(separator.join, axis=1)
                    if (full.to_numpy() == values).all():
                        return {"column": column, "parts": list(parts), "separator": separator}
    return None


FILENAME_LABELS = {
    # competition -> (image directory, how a file name yields its label, the label column)
    "dogs-vs-cats-redux-kernels-edition": ("train", lambda stem: stem.split(".")[0], "label"),
}


def synthesize_train_table(prepared: Path, competition: str):
    """Write the training table a competition keeps in its file names, once.

    dogs-vs-cats ships 22,500 pictures called cat.0.jpg and dog.10000.jpg and no train
    table at all, so the carve had nothing to split and the package was refused. The
    label is right there in the name; reading it into a real table is a transcription,
    not a modelling choice. Written beside the pictures under the name the sample
    submission's id column implies, so the starter, the carve and the terminal
    re-execution all see the same file.
    """
    spec = FILENAME_LABELS.get(competition)
    if spec is None:
        return None
    directory, label_of, column = spec
    public = prepared / "public"
    target = public / "train_labels.csv"
    if target.exists():
        return target
    images = sorted(p for p in (public / directory).iterdir()
                    if p.is_file() and p.suffix.lower() in (".jpg", ".jpeg", ".png"))
    if not images:
        raise SystemExit(f"{competition}: no images under {directory}/ to read labels from")
    classes = sorted({label_of(p.stem) for p in images})
    if len(classes) != 2:
        raise SystemExit(f"{competition}: expected two classes in the file names, saw {classes}")
    positive = classes[1]
    rows = ["id,%s" % column]
    rows += ["%s,%d" % (p.stem, int(label_of(p.stem) == positive)) for p in images]
    target.write_text("\n".join(rows) + "\n")
    print(f"synthesized {target.name}: {len(images)} rows, 1 = {positive}", file=sys.stderr)
    return target


def _payload_entries(prepared: Path) -> list[Path]:
    """The public entries copied into a carved data directory besides the three tables."""
    tables = {path.name for path in
              (_table_path(prepared / "public", kind) for kind in ("train", "test", "sample"))
              if path is not None}
    entries = []
    for entry in sorted((prepared / "public").iterdir()):
        if entry.name in tables or entry.name == "description.md":
            continue
        if entry.suffix.lower() == ".zip" and _is_table_archive(entry):
            # train.zip beside train.json is the same data compressed, and copying it
            # would hand the candidate the unsplit training set.
            #
            # Matching on the stem alone was wrong and silently gutted a whole class of
            # competition: aerial-cactus ships its pictures AS train.zip and test.zip
            # beside a train.csv of labels, so both archives were dropped and the
            # candidate got filenames with no pixels. Its search never beat the uniform
            # prior on any of six candidates, which is the signature of that.
            continue
        entries.append(entry)
    return entries


def _payload_bytes(prepared: Path) -> int:
    total = 0
    for entry in _payload_entries(prepared):
        paths = entry.rglob("*") if entry.is_dir() else (entry,)
        total += sum(p.stat().st_size for p in paths if p.is_file())
    return total


# Competitions routed to their own carve adapters. Repeated carve is defined for
# carve_split's row/group split only; each adapter has its own grouping rules and
# would need its own repeated-carve implementation.
ADAPTER_COMPETITIONS = {
    "aptos2019-blindness-detection": "aptos_images",
    "histopathologic-cancer-detection": "image_projection",
    "mlsp-2013-birds": "bird_records",
    "denoising-dirty-documents": "paired_pages",
    "ranzcr-clip-catheter-line-classification": "ranzcr_images",
}
CARVE_FOLDS_MAX_PAYLOAD_MB = 200


def check_carve_folds(competition: str, prepared: Path, folds: int,
                      max_payload_mb: float = CARVE_FOLDS_MAX_PAYLOAD_MB) -> None:
    """Refuse a repeated carve this builder cannot produce faithfully or cheaply."""
    if folds < 1:
        raise SystemExit("--carve-folds must be at least 1")
    if folds == 1:
        return
    if competition in ADAPTER_COMPETITIONS:
        raise SystemExit(
            f"--carve-folds {folds} is only supported for table competitions carved by "
            f"carve_split; {competition} is routed to the {ADAPTER_COMPETITIONS[competition]} "
            "adapter. Use --carve-folds 1.")
    payload = _payload_bytes(prepared)
    if payload > max_payload_mb * 1024 * 1024:
        raise SystemExit(
            f"--carve-folds {folds} would copy the {payload / 2**20:.0f} MB payload of "
            f"{competition} {folds - 1} more times, over the {max_payload_mb:g} MB repeated-carve "
            "cap. Repeated carve is meant for small table competitions; use --carve-folds 1.")


def carve_extra_folds(prepared: Path, root: Path, *, seed: int, holdout: float, folds: int) -> list[dict]:
    """Folds 1..K-1: the same carve with seeds seed+1..seed+K-1, controller-only.

    Each lands in evaluation/folds/<k>/ with problem/data's layout under data/ and the
    held-out labels in answers.csv. evaluation/ is never mounted for a candidate, so
    these splits -- whose fit rows include fold 0's held-out rows -- only ever reach a
    candidate program through a controller-run, sandboxed scoring execution.
    """
    records = []
    for k in range(1, folds):
        fold = root / "evaluation/folds" / str(k)
        split = carve_split(prepared, root, seed=seed + k, holdout=holdout,
                            data_dir=fold / "data", answers_path=fold / "answers.csv")
        records.append({"fold": k, "seed": seed + k, "fit_rows": split["fit_rows"],
                        "holdout_rows": split["holdout_rows"], "algorithm": split["algorithm"],
                        "grouped_on": split["grouped_on"], "stratified_on": split["stratified_on"],
                        "filenames": split["filenames"]})
    return records


def score_starter_on_fold(python: str, root: Path, metric: str, fold: int) -> float:
    """The starter scored on one extra fold through the same fold view the search uses."""
    sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
    from swarmrsi.naturebench_evaluator import fold_evaluation_root

    work = Path(tempfile.mkdtemp(prefix=f"starter_{root.name}_fold{fold}_"))
    try:
        output = work / "output"
        output.mkdir()
        evaluator = fold_evaluation_root(root, fold, work / "view")
        env = {**os.environ, "DATA_DIR": str(root / "evaluation/folds" / str(fold) / "data"),
               "OUTPUT_DIR": str(output), "PYTHONDONTWRITEBYTECODE": "1"}
        subprocess.run([python, str(root / "starter/run.py")], cwd=work, env=env, check=True)
        subprocess.run([python, str(evaluator)], cwd=work, env=env, check=True)
        return float(json.loads((work / "score.json").read_text())[metric])
    finally:
        shutil.rmtree(work, ignore_errors=True)


SUBJECT_GROUPS = {"siim-isic-melanoma-classification": "patient_id"}


def carve_split(prepared: Path, destination: Path, *, seed: int, holdout: float,
                data_dir: Path | None = None, answers_path: Path | None = None):
    """Hold out a slice of the official public train, keeping the sample's column shape.

    ``data_dir`` and ``answers_path`` default to the candidate-visible package
    (problem/data, evaluation/answers.csv). A repeated-carve fold points them at
    evaluation/folds/<k>/ instead and is otherwise the same carve with another seed.
    """
    import pandas as pd
    from sklearn.model_selection import train_test_split

    train = _read_table(prepared / "public", "train")
    sample = _read_table(prepared / "public", "sample")
    # Which submission columns are predictions is settled by the official test, not by
    # position. Taking the first column as the identifier works for spooky (id, EAP,
    # HPL, MWS) and fails for insults, whose submission is Insult, Date, Comment -- the
    # target comes first there, and the carve then treated it as a row id. A column the
    # official test already supplies is given; anything else is to be predicted.
    official_test = _read_table(prepared / "public", "test", optional=True)
    if official_test is not None:
        given = set(official_test.columns)
    else:
        # No test table to compare against, which is the image competitions. Fall back
        # to what the training table says: a submission column that is a key in train
        # -- present and unique for every row -- is the row identifier, and anything
        # else is a prediction. That reads aerial-cactus (id, has_cactus) correctly.
        given = {c for c in sample.columns
                 if c in train.columns and train[c].is_unique and not train[c].isna().any()}
    # A submission id the test does not carry as a column may still be built from ones
    # it does; that reconstruction is an identifier, not something to predict.
    composite = _composite_identifier(sample, official_test, given) if official_test is not None else None
    composite_column = composite["column"] if composite else None
    identifiers = [c for c in sample.columns if c in given or c == composite_column]
    target_columns = [c for c in sample.columns if c not in given and c != composite_column]
    if not target_columns:
        raise SystemExit("every submission column is present in the official test; no target to predict")
    id_column = identifiers[0] if identifiers else sample.columns[0]
    # The second test asks whether a column's VALUES are the target names, which is how
    # a one-hot submission like spooky's (id, EAP, HPL, MWS) finds its label column. It
    # is only ever true of a low-cardinality column, so the cardinality is checked first:
    # without that, text-normalization built a set of every distinct token in ten million
    # rows for each column it ruled out.
    def _is_label(column) -> bool:
        if column in target_columns:
            return True
        if train[column].nunique() > max(1000, len(target_columns)):
            return False
        return set(map(str, train[column].unique())) >= set(map(str, target_columns))

    label = next((c for c in train.columns if _is_label(c)), None)
    # A repeated identifier means the same example appears more than once, and a plain
    # row split would put copies of it on both sides -- the carved holdout would then
    # score a memorised answer. Those competitions are split by group instead, which
    # rules out stratifying.
    # Group on the given column that best identifies an example: the one with the most
    # distinct values and no gaps. Insults is given both Date and Comment, and Date --
    # first in the submission, and partly missing -- is neither the row's identity nor
    # a legal grouping key, which is what sklearn rejected.
    # A composite id is not itself a column of train, so grouping has to reach through it
    # to its parts. text-normalization keys on sentence_id: a plain row split would put
    # tokens of the same sentence on both sides, and the holdout would score a sentence
    # the search had already read.
    groupable = list(identifiers) + (composite["parts"] if composite else [])
    candidates = [c for c in groupable if c in train.columns and not train[c].isna().any()]
    group_column = max(candidates, key=lambda c: train[c].nunique(), default=None)
    # Some competitions repeat the subject, not the example: SIIM gives one image per
    # row but several images per patient, and the official test was cut patient-wise.
    # A row split would let the carve score a patient the search had already seen.
    subject = SUBJECT_GROUPS.get(prepared.parent.name)
    if subject is not None and subject in train.columns and not train[subject].isna().any():
        group_column = subject
    grouped = group_column is not None and train[group_column].duplicated().any()
    if grouped:
        from sklearn.model_selection import GroupShuffleSplit
        fit_ids, held_ids = next(GroupShuffleSplit(n_splits=1, test_size=holdout, random_state=seed)
                                 .split(train, groups=train[group_column]))
        fit, held = train.iloc[fit_ids], train.iloc[held_ids]
        if set(fit[group_column]) & set(held[group_column]):
            raise SystemExit("group split leaked identifiers across the carve")
        stratify = None
    else:
        # Stratifying needs at least two rows of every class. Some competitions carry a
        # class that appears once, and asking for a stratified split then fails outright
        # rather than falling back, so the check is made here.
        stratify = None
        if label is not None and train[label].nunique() <= 50:
            counts = train[label].value_counts()
            stratify = train[label] if counts.min() >= 2 else None
        fit, held = train_test_split(train, test_size=holdout, random_state=seed, shuffle=True, stratify=stratify)
    fit, held = fit.sort_index(), held.sort_index()

    data = destination / "problem/data" if data_dir is None else data_dir
    answers_path = destination / "evaluation/answers.csv" if answers_path is None else answers_path
    for directory in (data, answers_path.parent):
        directory.mkdir(parents=True, exist_ok=True)
    # Whatever is not one of the three tables is the competition's own payload: the
    # image directories, and the odd auxiliary file. The candidate's DATA_DIR is bound
    # into the sandbox as one resolved path, so a symlink pointing at the archive would
    # not resolve there and the payload has to be copied in.
    #
    # The whole payload is copied, official test images included. Those are inputs, not
    # answers -- MLE-Bench's own agents see them too -- and the labels, which are what
    # must not leak, live in the tables this function splits.
    # The tables to withhold are the three files actually READ as tables, not every
    # file whose name happens to contain "train" or "test". Matching by substring put
    # aerial-cactus's train.zip and test.zip -- which hold its pictures -- in this set,
    # so they were dropped before the archive inspection below ever ran.
    for entry in _payload_entries(prepared):
        target = data / entry.name
        if entry.is_dir():
            shutil.copytree(entry, target, dirs_exist_ok=True)
        else:
            shutil.copy2(entry, target)

    # The carved tables keep the competition's own filenames. They used to be
    # normalised to train.csv/test.csv/sample_submission.csv, and the terminal step
    # then broke every task whose real names differ: the frozen program re-fits on the
    # OFFICIAL public tree, where the file is sampleSubmission.csv or train.json, and
    # died on FileNotFoundError before it could produce a submission to grade.
    names = {kind: (_table_path(prepared / "public", kind) or Path(default)).name
             for kind, default in (("train", "train.csv"), ("test", "test.csv"),
                                   ("sample", "sample_submission.csv"))}
    if names["test"] == names["train"] or official_test is None:
        # The competition has no readable test table -- its test set is an archive of
        # images, and `test.zip` is a payload file, not a name the starter may read.
        # The carved test gets a name of its own, in the training table's format.
        names["test"] = "test" + "".join(Path(names["train"]).suffixes)
    _write_table(fit, data / names["train"])
    # The carved test shows exactly what the official test would: the given columns,
    # never a target. Dropping only the label column leaked the other targets on
    # competitions that predict several at once.
    feature_columns = [c for c in train.columns if c not in target_columns and c != label]
    if official_test is not None:
        # Exactly the given columns, and no more. train may carry a column the official
        # test withholds -- text-normalization ships `class`, the token's category, which
        # is most of the answer and which the real test does not provide -- and handing
        # it to the candidate would score a run on information the official grade never
        # gives it.
        given_features = [c for c in feature_columns if c in given]
        _write_table(held[given_features or feature_columns or list(train.columns)], data / names["test"])
    else:
        names["test"] = None

    # The answers carry exactly the sample's columns, because the competition's own
    # grader is handed this frame and will reject anything shaped differently.
    answers = pd.DataFrame({c: held[c].values for c in identifiers if c in held.columns})
    if composite is not None:
        answers[composite["column"]] = (held[composite["parts"]].astype(str)
                                        .agg(composite["separator"].join, axis=1).values)
        answers = answers[[c for c in sample.columns if c in answers.columns]]
    if answers.empty:
        answers = pd.DataFrame({id_column: held[id_column] if id_column in held.columns else range(len(held))})
    for column in target_columns:
        if column in held.columns:
            answers[column] = held[column].values
        elif label is not None:
            answers[column] = (held[label].astype(str) == column).astype(int).values
        else:
            raise SystemExit(f"cannot build answers for target column {column!r}")
    answers.to_csv(answers_path, index=False)
    sample_held = answers.copy()
    for column in target_columns:
        sample_held[column] = 0
    _write_table(sample_held, data / names["sample"])
    return {"filenames": names,
            "algorithm": "GroupShuffleSplit" if grouped else "train_test_split",
            "grouped_on": group_column if grouped else None, "seed": seed, "holdout": holdout,
            "stratified_on": label if stratify is not None else None,
            "fit_rows": len(fit), "holdout_rows": len(held),
            "label_column": label, "id_column": id_column, "target_columns": target_columns,
            "composite_id": composite}


def run_starter(python: str, root: Path, metric: str) -> float:
    work = Path(tempfile.mkdtemp(prefix=f"starter_{root.name}_"))
    output = work / "output"
    output.mkdir()
    env = {**os.environ, "DATA_DIR": str(root / "problem/data"), "OUTPUT_DIR": str(output),
           "PYTHONDONTWRITEBYTECODE": "1"}
    for script in ("starter/run.py", "evaluation/evaluator.py"):
        subprocess.run([python, str(root / script)], cwd=work, env=env, check=True)
    score = float(json.loads((work / "score.json").read_text())[metric])
    shutil.rmtree(work, ignore_errors=True)
    return score


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("competition")
    parser.add_argument("--cache", type=Path, default=Path("/cache"),
                        help="mlebench data dir; only its registry, leaderboard and grader are read")
    parser.add_argument("--prepared", type=Path, default=None,
                        help="the competition's prepared tree; defaults to <cache>/<competition>/prepared. "
                             "The archive of record lives on the NAS under a flat prepared/<competition>, "
                             "which is not the nested layout the registry would infer.")
    parser.add_argument("--tasks-root", type=Path, required=True)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--holdout", type=float, default=0.2)
    parser.add_argument("--baseline", type=float, default=None,
                        help="pinned starter score; omit to measure and print it")
    parser.add_argument("--official-source", type=Path, required=True,
                        help="the mle-bench checkout whose graders and revision are pinned into custody")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--carve-folds", type=int, default=1,
                        help="repeated hold-out carve: K>1 adds K-1 controller-only splits of the same "
                             "public train (seeds seed+1..seed+K-1) under evaluation/folds/<k>/; "
                             "table competitions only")
    parser.add_argument("--carve-folds-max-payload-mb", type=float, default=CARVE_FOLDS_MAX_PAYLOAD_MB)
    args = parser.parse_args()

    from mlebench.grade import get_leaderboard
    from mlebench.registry import registry

    # The revision is pinned the way mlebench_tasks.prepare pins it, so a package
    # states which upstream produced its grades rather than trusting a directory name.
    if (args.official_source / ".git").exists():
        revision = subprocess.check_output(
            ["git", "-C", str(args.official_source), "rev-parse", "HEAD"], text=True).strip()
        if subprocess.check_output(
                ["git", "-C", str(args.official_source), "status", "--porcelain"], text=True).strip():
            raise SystemExit("official source is dirty")
    else:
        revision = (args.official_source / "UPSTREAM_REVISION").read_text().strip()

    competition = registry.set_data_dir(args.cache).get_competition(args.competition)
    leaderboard = get_leaderboard(competition)
    lower_is_better = competition.grader.is_lower_better(leaderboard)
    thresholds = competition.grader.rank_score(None, leaderboard)
    metric = {"name": competition.grader.name.replace("-", "_"),
              "direction": "minimize" if lower_is_better else "maximize",
              "role": "public_validation"}

    prepared = args.prepared or (args.cache / args.competition / "prepared")
    check_carve_folds(args.competition, prepared, args.carve_folds, args.carve_folds_max_payload_mb)
    root = args.tasks_root / args.competition
    if root.exists():
        if not args.force:
            raise SystemExit(f"{root} exists; pass --force")
        shutil.rmtree(root)
    (root / "problem").mkdir(parents=True)
    (root / "starter").mkdir()
    (root / "private").mkdir()

    paired_pages = args.competition == 'denoising-dirty-documents'
    bird_records = args.competition == 'mlsp-2013-birds'
    image_projection = args.competition == 'histopathologic-cancer-detection'
    aptos_images = args.competition == 'aptos2019-blindness-detection'
    ranzcr_images = args.competition == 'ranzcr-clip-catheter-line-classification'
    if ranzcr_images:
        from ranzcr_images import OFFICIAL_HOLDOUT, carve_ranzcr
        # The carve estimates the official score, so it takes the official prepare.py
        # split fraction; run_matched always passes --holdout (default 0.2), so the
        # requested value is recorded but cannot override it.
        split = carve_ranzcr(prepared, root, seed=args.seed, holdout=OFFICIAL_HOLDOUT)
        split['requested_holdout'] = args.holdout
    elif aptos_images:
        from aptos_images import carve_aptos
        split = carve_aptos(prepared, root, seed=args.seed, holdout=args.holdout)
    elif image_projection:
        from image_projection import carve_images
        split = carve_images(prepared, root, seed=args.seed, holdout=args.holdout, expected_size=(96, 96))
    elif bird_records:
        from bird_records import carve_records
        split = carve_records(prepared, root, seed=args.seed, holdout=args.holdout)
    elif paired_pages:
        from paired_pages import carve_pages
        split = carve_pages(prepared, root, seed=args.seed, holdout=args.holdout)
    else:
        synthesize_train_table(prepared, args.competition)
        # The payload check ran before the train table may have been synthesized; that
        # table is a table, not payload, so the check stands.
        split = carve_split(prepared, root, seed=args.seed, holdout=args.holdout)
        if args.carve_folds > 1:
            extra = carve_extra_folds(prepared, root, seed=args.seed, holdout=args.holdout,
                                      folds=args.carve_folds)
            split = {**split, "carve_folds": args.carve_folds,
                     "folds": [{"fold": 0, "seed": args.seed, "fit_rows": split["fit_rows"],
                                "holdout_rows": split["holdout_rows"]}]
                     + [{k: v for k, v in record.items() if k in ("fold", "seed", "fit_rows", "holdout_rows")}
                        for record in extra]}
            for record in extra:
                if record["filenames"] != split["filenames"] or record["algorithm"] != split["algorithm"]:
                    raise SystemExit(f"fold {record['fold']} carved a different layout than fold 0")
    # The starter names this competition's actual files. The names are substituted
    # rather than discovered at run time so that the frozen program keeps working when
    # the terminal step points DATA_DIR at the official public tree, which carries
    # exactly these names.
    starter_source = STARTER
    if ranzcr_images:
        from ranzcr_images import STARTER as RANZCR_STARTER
        starter_source = RANZCR_STARTER
        shutil.copy2(Path(__file__).with_name('ranzcr_images.py'), root / 'evaluation/ranzcr_images.py')
    elif aptos_images:
        from aptos_images import STARTER as APTOS_STARTER
        starter_source = APTOS_STARTER
        for module in ('aptos_images.py', 'image_projection.py'):
            shutil.copy2(Path(__file__).with_name(module), root / 'evaluation' / module)
    elif image_projection:
        from image_projection import STARTER as IMAGE_STARTER
        starter_source = IMAGE_STARTER
        shutil.copy2(Path(__file__).with_name('image_projection.py'), root / 'evaluation/image_projection.py')
    elif bird_records:
        from bird_records import STARTER as BIRD_STARTER
        starter_source = BIRD_STARTER
        shutil.copy2(Path(__file__).with_name('bird_records.py'), root / 'evaluation/bird_records.py')
    elif paired_pages:
        from paired_pages import STARTER as PAGE_STARTER
        starter_source = PAGE_STARTER
        shutil.copy2(Path(__file__).with_name('paired_pages.py'), root / 'evaluation/paired_pages.py')
    else:
        for placeholder, kind in (("__TRAIN__", "train"), ("__SAMPLE__", "sample")):
            starter_source = starter_source.replace(placeholder, split["filenames"][kind])
        starter_source = starter_source.replace("__TARGETS__", repr(split["target_columns"]))
    (root / "starter/run.py").write_text(starter_source)
    (root / "evaluation/evaluator.py").write_text(EVALUATOR)
    shutil.copy2(prepared / "public/description.md", root / "problem/README.md")
    if ranzcr_images:
        from ranzcr_images import CONTRACT
        with (root / 'problem/README.md').open('a') as stream:
            stream.write(CONTRACT)
    if aptos_images:
        from aptos_images import CONTRACT
        with (root / 'problem/README.md').open('a') as stream:
            stream.write(CONTRACT)
    if image_projection:
        from image_projection import CONTRACT
        with (root / 'problem/README.md').open('a') as stream:
            stream.write(CONTRACT)
    if bird_records:
        from bird_records import CONTRACT
        with (root / 'problem/README.md').open('a') as stream:
            stream.write(CONTRACT)
    if paired_pages:
        with (root / 'problem/README.md').open('a') as stream:
            stream.write('\n\nExecution data contract: DATA_DIR/train contains noisy fit PNG pages; '
                'DATA_DIR/train_cleaned contains their paired clean pages. DATA_DIR/test contains '
                'unlabelled noisy pages; their cleaned targets are controller-only and inaccessible. '
                'Write OUTPUT_DIR/submission.csv with exactly the id,value rows listed in '
                'DATA_DIR/sampleSubmission.csv. IDs are page_stem_1based_row_1based_column; '
                'value is grayscale L intensity divided by 255, finite and in [0,1]. '
                'The same program and directory contract are used on the full public tree '
                'for terminal refit and prediction. Never read hidden clean targets. '
                'Internal validation holds out unseen clean templates; this is stricter and '
                'distribution-different from the official random noisy-file split. The validation '
                'protocol is frozen before scoring and must not be changed after seeing scores.\n')
    # The official test and its answers are kept beside the package for the terminal
    # grade and never mounted for a candidate. Their names vary by competition, so the
    # whole of private/ is taken and the public test is matched by kind.
    for source in sorted((prepared / "private").iterdir()):
        if source.is_file():
            shutil.copy2(source, root / "private" / ("private_" + source.name))
    for source in sorted((prepared / "public").iterdir()):
        if source.is_file() and "test" in source.name.lower():
            shutil.copy2(source, root / "private" / ("public_" + source.name))

    metadata = {"benchmark": "mlebench", "custody": str(root / "custody.json"),
                "task_id": args.competition, "search_metric": metric, "search_target": None,
                "official_test_visible_to_candidate": False,
                "mlebench_data_dir": str(args.cache), "public_split": split,
                "official_thresholds": {k: thresholds.get(k) for k in
                                        ("gold_threshold", "silver_threshold", "bronze_threshold", "median_threshold")},
                "leaderboard_teams": len(leaderboard)}
    if bird_records:
        metadata.update({
            "official_test_visibility_scope": "Official test record tables, audio and targets are excluded; fixed upstream unlabeled derived visualizations are retained. This is not absence of all official-test-derived input information.",
            "official_test_unlabeled_derived_assets_visible": True,
        })
    (root / "metadata.json").write_text(json.dumps(metadata, indent=1, sort_keys=True, default=str) + "\n")

    measured = run_starter(args.python, root, metric["name"])
    if args.baseline is not None and measured != args.baseline:
        raise SystemExit(f"starter scored {measured!r}, spec pins {args.baseline!r}; package NOT sealed")
    metadata["starter_baseline"] = measured
    if args.carve_folds > 1:
        # fold 0 is the pinned baseline above; the extra folds are measured so a package
        # whose fold view cannot be graded is never sealed.
        metadata["public_split"]["starter_fold_scores"] = [measured] + [
            score_starter_on_fold(args.python, root, metric["name"], k) for k in range(1, args.carve_folds)]
    (root / "metadata.json").write_text(json.dumps(metadata, indent=1, sort_keys=True, default=str) + "\n")

    # The custody record is the MLE shape the adapter verifies (refs, prepared,
    # official_source, assets_sha256), not the panel shape: mlebench_run.py checks the
    # task root against exactly these refs before it will mount anything.
    refs = [source_ref(p) for p in sorted(prepared.rglob("*")) if p.is_file()]
    refs += [source_ref(p) for p in sorted((args.official_source / "mlebench").rglob("*"))
             if p.is_file() and "__pycache__" not in p.parts and p.suffix != ".pyc"]
    refs += [source_ref(p) for p in sorted(root.rglob("*")) if p.is_file()]
    body = {"schema_version": "swarmrsi-mlebench-task-assets-v1", "task": args.competition,
            "official_source": str(args.official_source), "official_revision": revision,
            "prepared": str(prepared), "search_metric": metric, "starter_baseline": measured,
            "refs": refs}
    write_immutable(root / "custody.json", {**body, "assets_sha256": canonical_sha256(body)})
    record = {"files": refs}
    verify_assets(root / "custody.json")
    print(json.dumps({"competition": args.competition, "root": str(root), "metric": metric,
                      "starter_baseline": measured, "thresholds": metadata["official_thresholds"],
                      "split": {k: split[k] for k in ("fit_rows", "holdout_rows", "stratified_on")},
                      "files_sealed": len(record["files"])}, indent=1, default=str))
    return 0


if __name__ == "__main__":
    sys.exit(main())
