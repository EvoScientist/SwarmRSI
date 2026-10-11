"""Repeated carve: score an MLE candidate on K public hold-out splits and select on the mean.

The package builder adds folds 1..K-1 under evaluation/folds (controller-only), and the
bridge re-executes the same program on each and averages. With K=1, or without the
flag, or without evaluation/folds, nothing changes: same package bytes, same result.
"""
import hashlib
import importlib.util
import json
import os
import shutil
import subprocess
import sys
import types
from pathlib import Path

import pandas as pd
import pytest

REPO = Path(__file__).resolve().parents[1]
MODULE = REPO / "deploy/mle/make_task_root.py"
COMPETITION = "toy-table"

from evolvcoder import naturebench_evaluator as bridge  # noqa: E402
from evolvcoder import naturebench_session as session  # noqa: E402


def load(path, name):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def prepared_tree(tmp_path, n=100, payload_bytes=0):
    prepared = tmp_path / "cache" / COMPETITION / "prepared"
    public = prepared / "public"
    public.mkdir(parents=True)
    (prepared / "private").mkdir()
    (public / "description.md").write_text("Toy table competition\n")
    pd.DataFrame({"id": [f"r{i:03d}" for i in range(n)], "feature": [(i * 7) % 13 for i in range(n)],
                  "label": [int((i * 7) % 13 > 5) ^ int(i % 9 == 0) for i in range(n)]}
                 ).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"id": ["t0", "t1"], "feature": [1, 9]}).to_csv(public / "test.csv", index=False)
    pd.DataFrame({"id": ["t0", "t1"], "label": [0, 0]}).to_csv(public / "sample_submission.csv", index=False)
    pd.DataFrame({"id": ["t0", "t1"], "label": [0, 1]}).to_csv(prepared / "private/answers.csv", index=False)
    if payload_bytes:
        (public / "extra").mkdir()
        (public / "extra/blob.bin").write_bytes(b"x" * payload_bytes)
    return prepared


def official_tree(tmp_path):
    official = tmp_path / "official"
    (official / "mlebench").mkdir(parents=True)
    (official / "UPSTREAM_REVISION").write_text("synthetic-official-revision")
    (official / "mlebench/__init__.py").write_text("")
    (official / "mlebench/registry.py").write_text('''
class Grader:
    def __call__(self, submission, answers):
        joined = answers.merge(submission, on="id", suffixes=("", "_p"))
        if len(joined) != len(answers):
            raise ValueError("ids differ")
        return float((joined["label"] == joined["label_p"]).mean())
class Registry:
    def set_data_dir(self, path): return self
    def get_competition(self, name):
        class Competition: grader = Grader()
        return Competition()
registry = Registry()
''')
    return official


def build(tmp_path, monkeypatch, capsys, *, folds=1, name="tasks", extra=()):
    prepared = tmp_path / "cache" / COMPETITION / "prepared"
    if not prepared.exists():
        prepared_tree(tmp_path)
    official = tmp_path / "official"
    if not official.exists():
        official_tree(tmp_path)
    grader = types.SimpleNamespace(name="accuracy", is_lower_better=lambda _: False,
                                   rank_score=lambda *_: {"gold_threshold": .9})
    registry = types.SimpleNamespace(set_data_dir=lambda _: types.SimpleNamespace(
        get_competition=lambda _: types.SimpleNamespace(grader=grader)))
    monkeypatch.setitem(sys.modules, "mlebench", types.ModuleType("mlebench"))
    monkeypatch.setitem(sys.modules, "mlebench.grade", types.SimpleNamespace(get_leaderboard=lambda _: [1, 2]))
    monkeypatch.setitem(sys.modules, "mlebench.registry", types.SimpleNamespace(registry=registry))
    monkeypatch.setenv("PYTHONPATH", str(official) + os.pathsep + os.environ.get("PYTHONPATH", ""))
    builder = load(MODULE, "carve_builder")
    tasks = tmp_path / name
    monkeypatch.setattr(sys, "argv", ["make_task_root.py", COMPETITION, "--prepared", str(prepared),
                                      "--tasks-root", str(tasks), "--cache", str(tmp_path / "cache"),
                                      "--official-source", str(official), "--python", sys.executable,
                                      *(["--carve-folds", str(folds)] if folds != 1 else []), *extra])
    assert builder.main() == 0
    capsys.readouterr()
    return builder, tasks / COMPETITION


def tree(root):
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- package builder


def test_three_folds_are_controller_only_repeated_carves(tmp_path, monkeypatch, capsys):
    builder, root = build(tmp_path, monkeypatch, capsys, folds=3)
    split = json.loads((root / "metadata.json").read_text())["public_split"]
    assert split["carve_folds"] == 3
    assert [f["fold"] for f in split["folds"]] == [0, 1, 2]
    assert [f["seed"] for f in split["folds"]] == [0, 1, 2]
    assert {(f["fit_rows"], f["holdout_rows"]) for f in split["folds"]} == {(80, 20)}
    assert len(split["starter_fold_scores"]) == 3
    train = pd.read_csv(tmp_path / "cache" / COMPETITION / "prepared/public/train.csv").set_index("id")
    held_sets = [frozenset(pd.read_csv(root / "evaluation/answers.csv").id)]
    for k in (1, 2):
        fold = root / "evaluation/folds" / str(k)
        assert sorted(p.name for p in (fold / "data").iterdir()) == sorted(
            p.name for p in (root / "problem/data").iterdir())
        answers = pd.read_csv(fold / "answers.csv")
        assert list(answers.columns) == list(pd.read_csv(root / "evaluation/answers.csv").columns)
        fit = pd.read_csv(fold / "data/train.csv")
        test = pd.read_csv(fold / "data/test.csv")
        # The answers are the held rows' true labels, the test is those rows without them.
        assert list(test.id) == list(answers.id) and "label" not in test.columns
        assert (train.loc[answers.id, "label"].values == answers.label.values).all()
        assert not set(fit.id) & set(answers.id) and len(fit) + len(answers) == len(train)
        # Same stratification rule as fold 0.
        assert abs(answers.label.mean() - train.label.mean()) < 0.06
        held_sets.append(frozenset(answers.id))
    assert len(set(held_sets)) == 3
    # Custody seals every fold file, so tampering is detected.
    custody = json.loads((root / "custody.json").read_text())
    sealed = {ref["path"] for ref in custody["refs"]}
    for path in (root / "evaluation/folds").rglob("*"):
        if path.is_file():
            assert str(path.resolve()) in sealed
    builder.verify_assets(root / "custody.json")
    (root / "evaluation/folds/2/answers.csv").write_text("id,label\nr000,1\n")
    with pytest.raises(Exception):
        builder.verify_assets(root / "custody.json")


def test_one_fold_package_is_byte_identical_and_candidate_view_unchanged(tmp_path, monkeypatch, capsys):
    _, single = build(tmp_path, monkeypatch, capsys, folds=1, name="single")
    _, again = build(tmp_path, monkeypatch, capsys, folds=1, name="again")
    _, repeated = build(tmp_path, monkeypatch, capsys, folds=3, name="repeated")
    assert not (single / "evaluation/folds").exists()
    single_tree = tree(single)
    # metadata.json and custody.json name the package's own path; everything else is
    # content only.
    located = ("metadata.json", "custody.json")
    assert {k: v for k, v in single_tree.items() if k not in located} == {
        k: v for k, v in tree(again).items() if k not in located}
    repeated_tree = tree(repeated)
    for name, digest in single_tree.items():
        if name not in located:
            assert repeated_tree[name] == digest, name

    def unlocated(root):
        metadata = json.loads((root / "metadata.json").read_text())
        metadata.pop("custody")
        return metadata

    assert unlocated(single) == unlocated(again)
    single_split, repeated_split = unlocated(single)["public_split"], unlocated(repeated)["public_split"]
    assert "carve_folds" not in single_split and "folds" not in single_split
    assert {k: v for k, v in repeated_split.items()
            if k not in ("carve_folds", "folds", "starter_fold_scores")} == single_split


def test_grouped_folds_keep_the_group_rule(tmp_path):
    builder = load(MODULE, "carve_builder_groups")
    public = tmp_path / "siim-isic-melanoma-classification/prepared/public"
    public.mkdir(parents=True)
    pd.DataFrame({"image_name": [f"I{i:03d}" for i in range(90)], "patient_id": [f"P{i // 3}" for i in range(90)],
                  "target": [i % 2 for i in range(90)]}).to_csv(public / "train.csv", index=False)
    pd.DataFrame({"image_name": ["X1"], "patient_id": ["Q"]}).to_csv(public / "test.csv", index=False)
    pd.DataFrame({"image_name": ["X1"], "target": [0]}).to_csv(public / "sample_submission.csv", index=False)
    out = tmp_path / "out"
    fold0 = builder.carve_split(public.parent, out, seed=4, holdout=.2)
    records = builder.carve_extra_folds(public.parent, out, seed=4, holdout=.2, folds=3)
    assert [r["seed"] for r in records] == [5, 6]
    for record in records:
        assert record["grouped_on"] == fold0["grouped_on"] == "patient_id"
        data = out / "evaluation/folds" / str(record["fold"]) / "data"
        held = pd.read_csv(public / "train.csv").set_index("image_name").loc[pd.read_csv(data / "test.csv").image_name]
        assert not set(pd.read_csv(data / "train.csv").patient_id) & set(held.patient_id)


@pytest.mark.parametrize("competition", ["aptos2019-blindness-detection", "histopathologic-cancer-detection",
                                         "mlsp-2013-birds", "denoising-dirty-documents"])
def test_adapter_competitions_refuse_repeated_carve(tmp_path, competition):
    builder = load(MODULE, "carve_builder_refuse")
    with pytest.raises(SystemExit, match="only supported for table competitions"):
        builder.check_carve_folds(competition, tmp_path, 3)
    builder.check_carve_folds(competition, tmp_path, 1)


def test_adapter_refusal_happens_before_anything_is_built(tmp_path, monkeypatch):
    builder = load(MODULE, "carve_builder_main_refuse")
    monkeypatch.setitem(sys.modules, "mlebench", types.ModuleType("mlebench"))
    monkeypatch.setitem(sys.modules, "mlebench.grade", types.SimpleNamespace(get_leaderboard=lambda _: []))
    grader = types.SimpleNamespace(name="qwk", is_lower_better=lambda _: False, rank_score=lambda *_: {})
    monkeypatch.setitem(sys.modules, "mlebench.registry", types.SimpleNamespace(registry=types.SimpleNamespace(
        set_data_dir=lambda _: types.SimpleNamespace(get_competition=lambda _: types.SimpleNamespace(grader=grader)))))
    official = official_tree(tmp_path)
    monkeypatch.setattr(sys, "argv", ["make_task_root.py", "aptos2019-blindness-detection", "--prepared",
                                      str(tmp_path / "nowhere"), "--tasks-root", str(tmp_path / "tasks"),
                                      "--official-source", str(official), "--carve-folds", "3"])
    with pytest.raises(SystemExit, match="aptos_images"):
        builder.main()
    assert not (tmp_path / "tasks").exists()


def test_large_payload_refuses_repeated_carve(tmp_path):
    builder = load(MODULE, "carve_builder_payload")
    prepared = prepared_tree(tmp_path, payload_bytes=4096)
    with pytest.raises(SystemExit, match="repeated-carve cap"):
        builder.check_carve_folds(COMPETITION, prepared, 3, max_payload_mb=0.001)
    builder.check_carve_folds(COMPETITION, prepared, 3, max_payload_mb=1)


def test_payload_is_copied_into_every_fold(tmp_path):
    builder = load(MODULE, "carve_builder_payload_copy")
    prepared = prepared_tree(tmp_path, payload_bytes=16)
    out = tmp_path / "out"
    builder.carve_split(prepared, out, seed=0, holdout=.2)
    builder.carve_extra_folds(prepared, out, seed=0, holdout=.2, folds=3)
    for k in (1, 2):
        assert (out / "evaluation/folds" / str(k) / "data/extra/blob.bin").read_bytes() == b"x" * 16


def test_run_matched_passes_folds_and_writes_the_profile_limit(tmp_path, monkeypatch):
    runner = load(REPO / "deploy/mle/run_matched.py", "carve_run_matched")
    calls = []

    def fake_run(argv, check):
        calls.append(list(argv))
        package = tmp_path / "root/tasks" / COMPETITION
        package.mkdir(parents=True, exist_ok=True)
        (package / "metadata.json").write_text(json.dumps({
            "official_thresholds": {"gold_threshold": 1.0}, "search_metric": {"direction": "maximize"},
            "public_split": {"carve_folds": 3}}))

    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setattr(runner, "build_profile", lambda *a, **k: {"screening_execution_limits": {}})
    monkeypatch.setattr(sys, "argv", ["run_matched.py", COMPETITION, "--root", str(tmp_path / "root"),
                                      "--official-source", str(tmp_path), "--envelope", str(tmp_path / "none.json"),
                                      "--carve-folds", "3"])
    assert runner.main() == 0
    assert calls[0][-2:] == ["--carve-folds", "3"]
    profile = json.loads((tmp_path / "root/profiles" / f"{COMPETITION}.json").read_text())
    assert profile["screening_execution_limits"]["carve_folds"] == 3
    # A package built with another fold count is refused rather than silently reused.
    monkeypatch.setattr(sys, "argv", ["run_matched.py", COMPETITION, "--root", str(tmp_path / "root"),
                                      "--official-source", str(tmp_path), "--envelope", str(tmp_path / "none.json")])
    with pytest.raises(SystemExit, match="built with 3 carve fold"):
        runner.main()


# ---------------------------------------------------------------- evaluator bridge

# Under bwrap every fold is bound to the same /public_data path, so the candidate tells
# folds apart by content: the ids of the test table it was handed.
CANDIDATE = """import os, pathlib, time
import pandas as pd
data = pathlib.Path(os.environ["DATA_DIR"])
test = pd.read_csv(data / "test.csv")
fold = __FOLDS__.get(",".join(sorted(map(str, test.id))), 0)
extra_fold = fold > 0
print("fitting on", len(pd.read_csv(data / "train.csv")), "rows", "FOLD" if extra_fold else "PUBLIC")
__EXTRA__
out = pathlib.Path(os.environ["OUTPUT_DIR"]); out.mkdir(parents=True, exist_ok=True)
pd.DataFrame({"id": test.id, "label": (test.feature > 5).astype(int)}).to_csv(out / "submission.csv", index=False)
"""


def fold_keys(root):
    """Map each extra fold's sorted test ids to its index."""
    keys = {}
    for data in sorted((root / "evaluation/folds").glob("*/data")) if root is not None else ():
        ids = pd.read_csv(data / "test.csv").id
        keys[",".join(sorted(map(str, ids)))] = int(data.parent.name)
    if root is not None:
        fold0 = ",".join(sorted(map(str, pd.read_csv(root / "problem/data/test.csv").id)))
        assert fold0 not in keys and len(keys) == len(list((root / "evaluation/folds").glob("*/data")))
    return keys


def candidate(tmp_path, extra="", name="workspace", root=None):
    workspace = tmp_path / name
    workspace.mkdir()
    (workspace / "run.py").write_text(
        CANDIDATE.replace("__FOLDS__", repr(fold_keys(root))).replace("__EXTRA__", extra))
    return workspace


def evaluate(tmp_path, root, workspace, *extra):
    result = tmp_path / ("result%d.json" % len(list(tmp_path.glob("result*.json"))))
    completed = subprocess.run(
        [sys.executable, "-m", "evolvcoder.naturebench_evaluator", "--workspace", str(workspace),
         "--result", str(result), "--task-root", str(root), "--runtime-python", sys.executable,
         "--timeout-seconds", "120", *extra],
        capture_output=True, text=True, env={**os.environ, "PYTHONPATH": os.pathsep.join(
            [str(REPO / "src"), str(tmp_path / "official"), os.environ.get("PYTHONPATH", "")])})
    assert result.is_file(), completed.stderr
    return json.loads(result.read_text())


def expected_accuracy(answers, data):
    test = pd.read_csv(data / "test.csv")
    predicted = (test.feature > 5).astype(int).values
    return float((pd.read_csv(answers).label.values == predicted).mean())


def test_three_folds_score_the_mean_and_record_each_fold(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    workspace = candidate(tmp_path, root=root)
    result = evaluate(tmp_path, root, workspace, "--carve-folds", "3")
    assert result["valid"] is True
    expected = [expected_accuracy(root / "evaluation/answers.csv", root / "problem/data")] + [
        expected_accuracy(root / f"evaluation/folds/{k}/answers.csv", root / f"evaluation/folds/{k}/data")
        for k in (1, 2)]
    assert len(set(expected)) > 1
    carve = result["details"]["repeated_carve"]
    assert carve["folds"] == carve["completed_folds"] == 3
    assert carve["fold_native_scores"] == pytest.approx(expected)
    mean = sum(expected) / 3
    assert result["metrics"]["accuracy"] == pytest.approx(mean)
    assert result["metrics"]["search_utility"] == result["metrics"]["aggregate_improvement"] == pytest.approx(mean)
    assert result["details"]["native_score"] == pytest.approx(mean)
    assert result["details"]["primary_scores"] == {"public_validation": pytest.approx(mean)}
    std = (sum((v - mean) ** 2 for v in expected) / 3) ** 0.5
    assert carve["native_std"] == pytest.approx(std)
    # Only fold 0's run log reaches the planner; extra folds are recorded by digest.
    log = result["details"]["candidate_execution_log"]
    assert "PUBLIC" in log and "FOLD" not in log
    assert all("execution_log_sha256" in r for r in carve["fold_records"][1:])
    # The scored workspace keeps fold 0's output and gains no scratch.
    assert sorted(p.name for p in workspace.iterdir()) == ["output", "run.py"]
    assert result["details"]["submission_output_attestation"] == bridge._attest(workspace / "output")
    assert not list(tmp_path.glob(".repeated_carve_*"))


def test_score_existing_output_reuses_fold_zero_and_runs_the_others(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    workspace = candidate(tmp_path)
    execution = evaluate(tmp_path, root, workspace, "--execution-only")
    (workspace / ".evolvcoder").mkdir()
    (workspace / ".evolvcoder/execution_validation.json").write_text(json.dumps(execution))
    fresh = evaluate(tmp_path, root, candidate(tmp_path, name="fresh"), "--carve-folds", "3")
    scored = evaluate(tmp_path, root, workspace, "--score-existing-output", "--carve-folds", "3")
    assert scored["valid"] is True and scored["details"]["candidate_execution_reused"] is True
    carve = scored["details"]["repeated_carve"]
    assert carve["fold0_execution_reused"] is True and carve["fold_records"][0]["execution_reused"] is True
    assert carve["fold_native_scores"] == fresh["details"]["repeated_carve"]["fold_native_scores"]
    assert scored["details"]["candidate_execution_log"] == execution["details"]["candidate_execution_log"]
    # --execution-only never runs folds.
    result = tmp_path / "combo.json"
    completed = subprocess.run(
        [sys.executable, "-m", "evolvcoder.naturebench_evaluator", "--workspace", str(workspace),
         "--result", str(result), "--task-root", str(root), "--runtime-python", sys.executable,
         "--execution-only", "--carve-folds", "3"], capture_output=True, text=True,
        env={**os.environ, "PYTHONPATH": str(REPO / "src")})
    assert completed.returncode != 0 and "execution-only" in completed.stderr


def test_a_candidate_failing_on_one_fold_is_invalid(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    secret = "r_SECRET_LABEL_TEXT"
    workspace = candidate(tmp_path, root=root, extra=(
        "if fold == 2:\n"
        f"    raise KeyError('{secret}')\n"))
    result = evaluate(tmp_path, root, workspace, "--carve-folds", "3")
    assert result["valid"] is False and "metrics" not in result
    diagnostic = result["public_diagnostic"]
    assert diagnostic["category"] == "candidate_execution"
    assert "fold 2 of 3" in diagnostic["summary"] and "KeyError" in diagnostic["summary"]
    assert secret not in json.dumps(result)
    assert result["details"]["repeated_carve"]["completed_folds"] == 2


def test_a_deadline_that_cannot_fit_every_fold_is_incomplete_not_averaged(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    workspace = candidate(tmp_path, root=root, extra="if extra_fold:\n    time.sleep(30)\n")
    result = evaluate(tmp_path, root, workspace, "--carve-folds", "3", "--carve-deadline-seconds", "6")
    assert result["valid"] is False and "metrics" not in result
    assert result["public_diagnostic"]["category"] == "repeated_carve_incomplete"
    assert result["public_diagnostic"]["summary"].startswith("repeated_carve_incomplete: 1/3 folds")
    assert result["public_diagnostic"]["candidate_only"] is False
    assert result["details"]["repeated_carve"]["completed_folds"] == 1
    assert not session.evaluation_evidence_retryable(result)


def test_without_the_flag_or_without_folds_the_result_is_unchanged(tmp_path, monkeypatch, capsys):
    _, single = build(tmp_path, monkeypatch, capsys, folds=1, name="single")
    _, repeated = build(tmp_path, monkeypatch, capsys, folds=3, name="repeated")
    before = evaluate(tmp_path, single, candidate(tmp_path, name="a"))
    after = evaluate(tmp_path, repeated, candidate(tmp_path, name="b"))
    assert "repeated_carve" not in after["details"]
    assert before == after
    # Asking for folds a package does not hold is a configuration failure, never a
    # silent single-fold score.
    refused = evaluate(tmp_path, single, candidate(tmp_path, name="c"), "--carve-folds", "3")
    assert refused["valid"] is False
    assert refused["public_diagnostic"]["category"] == "repeated_carve_configuration"
    assert not session.evaluation_evidence_retryable(refused)


def test_session_enables_folds_only_with_both_the_limit_and_the_folds(tmp_path):
    task = tmp_path / "task"
    (task / "evaluation").mkdir(parents=True)
    assert session.repeated_carve_folds_enabled(task, {}) is None
    assert session.repeated_carve_folds_enabled(task, {"carve_folds": 3}) is None
    assert session.repeated_carve_arguments(None, fold_timeout=10, deadline=10) == ()
    for k in (1, 2):
        (task / f"evaluation/folds/{k}/data").mkdir(parents=True)
        (task / f"evaluation/folds/{k}/answers.csv").write_text("id,label\n")
    assert session.repeated_carve_folds_enabled(task, {}) is None
    assert session.repeated_carve_folds_enabled(task, {"carve_folds": 1}) is None
    assert session.repeated_carve_folds_enabled(task, {"carve_folds": 3}) == 3
    with pytest.raises(ValueError, match="holds 3"):
        session.repeated_carve_folds_enabled(task, {"carve_folds": 4})
    with pytest.raises(ValueError):
        session.repeated_carve_folds_enabled(task, {"carve_folds": "3"})
    assert session.repeated_carve_arguments(3, fold_timeout=600, deadline=1900.5) == (
        "--carve-folds", "3", "--carve-fold-timeout-seconds", "600", "--carve-deadline-seconds", "1900")


# ---------------------------------------------------------------- session wiring


@pytest.mark.skipif(shutil.which("bwrap") is None,
                    reason="evaluate_root always requires the bubblewrap candidate sandbox")
def test_root_is_scored_on_the_same_folds_end_to_end(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    monkeypatch.setattr(session, "RUNTIME", Path(sys.executable))
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    monkeypatch.setenv("PYTHONPATH", os.pathsep.join([str(REPO / "src"), str(tmp_path / "official")]))
    run_dir = tmp_path / "run"; run_dir.mkdir()
    metric, valid, wall, summary, _ = session.evaluate_root(
        starter=root / "starter", task_root=root, run_dir=run_dir, timeout_seconds=120,
        carve_folds=3, carve_deadline_seconds=360)
    scores = json.loads((root / "metadata.json").read_text())["public_split"]["starter_fold_scores"]
    assert valid and metric == pytest.approx(sum(scores) / 3)
    evidence = json.loads((run_dir / "root_evaluation.json").read_text())
    assert evidence["details"]["repeated_carve"]["fold_native_scores"] == pytest.approx(scores)
    # Reusing this root evidence in a single-carve run is refused.
    other = tmp_path / "other"; other.mkdir()
    (run_dir / "manifest.json").write_text(json.dumps({"task_id": root.name, "source_hashes": {
        "starter": session.sha(root / "starter/run.py"), "evaluator": session.sha(session.EVALUATOR)}}))
    with pytest.raises(ValueError, match="carve folds"):
        session.evaluate_root(starter=root / "starter", task_root=root, run_dir=other, timeout_seconds=120,
                              reuse_evidence_path=run_dir / "root_evaluation.json")


def test_root_without_folds_makes_the_historical_call(tmp_path, monkeypatch):
    starter = tmp_path / "starter"; starter.mkdir(); (starter / "run.py").write_text("print(1)\n")
    task_root = tmp_path / "task"; (task_root / "problem/data").mkdir(parents=True)
    run_dir = tmp_path / "run"; run_dir.mkdir()
    seen = []

    def fake_run(argv, **kwargs):
        seen.append((list(argv), kwargs["timeout"]))
        result = Path(argv[list(argv).index("--result") + 1])
        result.write_text(json.dumps({"schema_version": "external-domain-result-v1", "valid": True,
                                      "execution_valid": True, "score_bearing_evaluation_attempted": True,
                                      "metrics": {"aggregate_improvement": 0.5}}))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(session, "RUNTIME", Path(sys.executable))
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    monkeypatch.setattr(session, "subprocess", types.SimpleNamespace(run=fake_run, CompletedProcess=subprocess.CompletedProcess))
    session.evaluate_root(starter=starter, task_root=task_root, run_dir=run_dir, timeout_seconds=50)
    argv, timeout = seen[0]
    assert "--carve-folds" not in argv and argv[-1] == "--require-candidate-sandbox" and timeout == 80
    (run_dir / "root_evaluation.json").unlink()
    session.evaluate_root(starter=starter, task_root=task_root, run_dir=run_dir, timeout_seconds=50,
                          carve_folds=3, carve_deadline_seconds=150)
    argv, timeout = seen[1]
    assert argv[-6:] == ["--carve-folds", "3", "--carve-fold-timeout-seconds", "50",
                         "--carve-deadline-seconds", "150"] and timeout == 180


def _scoring_argv(tmp_path, monkeypatch, limits, with_folds):
    task_root = tmp_path / "task"
    (task_root / "problem/data").mkdir(parents=True, exist_ok=True)
    if with_folds:
        for k in (1, 2):
            (task_root / f"evaluation/folds/{k}/data").mkdir(parents=True, exist_ok=True)
            (task_root / f"evaluation/folds/{k}/answers.csv").write_text("id,label\n")
    run_dir = tmp_path / ("run%d" % len(list(tmp_path.glob("run*"))))
    run_dir.mkdir()
    executor = session.Executor.__new__(session.Executor)
    executor.model, executor.run_dir, executor.seed = None, run_dir, 0
    executor.task_root, executor.required_symbols = task_root, ("main",)
    executor.screening_limits = {"internal_validation_enabled": False, **limits}
    executor.carve_folds = session.repeated_carve_folds_enabled(task_root, executor.screening_limits)
    monkeypatch.setattr(executor, "_config", lambda *a: None, raising=False)
    monkeypatch.setattr(executor, "_diagnostic_seed", lambda *a: (None, None), raising=False)

    class Coder:
        def __init__(self, *a): pass
        def run(self, artifact, cell, **_):
            (cell / "artifact").mkdir(parents=True)
            return types.SimpleNamespace(status=types.SimpleNamespace(value="success"), implementation_real=True)

    seen = []

    def fake_run(argv, **kwargs):
        seen.append((list(argv), kwargs["timeout"]))
        result = Path(argv[list(argv).index("--result") + 1])
        result.write_text(json.dumps({"schema_version": "external-domain-result-v1", "valid": True,
                                      "execution_valid": True, "score_bearing_evaluation_attempted": True,
                                      "metrics": {"aggregate_improvement": 0.5}}))
        return subprocess.CompletedProcess(argv, 0, "", "")

    monkeypatch.setattr(session, "ContractBoundedCoder", Coder)
    monkeypatch.setattr(session, "RUNTIME", Path(sys.executable))
    monkeypatch.setattr(session, "CANDIDATE_RUNTIME", None)
    monkeypatch.setattr(session, "subprocess", types.SimpleNamespace(
        run=fake_run, CompletedProcess=subprocess.CompletedProcess, TimeoutExpired=subprocess.TimeoutExpired))
    proposal = types.SimpleNamespace(candidate=types.SimpleNamespace(id="a1"))
    parent = types.SimpleNamespace(artifact=tmp_path / "parent")
    execution = executor.execute(proposal=proposal, parent=parent, step=1)
    assert execution.native_metric == 0.5
    return seen[0]


def test_search_scoring_passes_the_folds_and_sizes_the_bridge(tmp_path, monkeypatch):
    limits = {"full_evaluation_timeout_seconds": 100, "execution_validation_timeout_seconds": 60}
    plain, plain_timeout = _scoring_argv(tmp_path, monkeypatch, limits, with_folds=False)
    assert "--carve-folds" not in plain and plain_timeout == 130
    also_plain, _ = _scoring_argv(tmp_path, monkeypatch, {**limits, "carve_folds": 3}, with_folds=False)
    assert "--carve-folds" not in also_plain
    unrequested, _ = _scoring_argv(tmp_path, monkeypatch, limits, with_folds=True)
    assert "--carve-folds" not in unrequested
    carved, timeout = _scoring_argv(tmp_path, monkeypatch, {**limits, "carve_folds": 3}, with_folds=True)
    # fold 0 grading (100) + two extra folds of execution (60) and grading (100) each.
    assert carved[-6:] == ["--carve-folds", "3", "--carve-fold-timeout-seconds", "60",
                           "--carve-deadline-seconds", "420"] and timeout == 450
    assert "--score-existing-output" in carved


def test_bridge_fold_scores_match_the_builder_measured_starter_scores(tmp_path, monkeypatch, capsys):
    _, root = build(tmp_path, monkeypatch, capsys, folds=3)
    workspace = tmp_path / "starter_copy"
    shutil.copytree(root / "starter", workspace)
    result = evaluate(tmp_path, root, workspace, "--carve-folds", "3")
    scores = json.loads((root / "metadata.json").read_text())["public_split"]["starter_fold_scores"]
    assert result["details"]["repeated_carve"]["fold_native_scores"] == pytest.approx(scores)
