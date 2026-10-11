from dataclasses import replace
from types import SimpleNamespace

import pytest

from evolvcoder.api import Application, RunContext, RunResult, TaskSpec
from evolvcoder.configuration import config_from_dict
from evolvcoder.validation import EvaluatorObservationAdapter, FrozenEvoCoderValidator
from test_naturebench_entrypoint import inputs  # noqa: F401
from test_unified_run import setup  # noqa: F401


def task(tmp_path):
    return TaskSpec("benchmark", "task", "public task", tmp_path / "starter", tmp_path / "data",
                    "official-v1", "accuracy", "maximize", tmp_path / "private-evaluator.json")


def context(tmp_path):
    config = config_from_dict({"benchmark": "benchmark", "task": "task", "method": "method",
                               "model": {"model": "provider/model"}}, base=tmp_path)
    return RunContext("run-1", tmp_path / "out", config)


def test_public_context_never_exposes_controller_paths(tmp_path):
    public = task(tmp_path).public_context()
    assert "evaluator_config" not in public
    assert "starter" not in public and "public_data" not in public
    assert public["description"] == "public task"


def test_sdk_dispatches_requested_adapters_and_rejects_mismatches(tmp_path):
    app = Application(); ctx = context(tmp_path); called = []
    class Benchmark:
        def task(self, task_id, config):
            called.append(("benchmark", task_id))
            return task(tmp_path)
    class Method:
        def run(self, spec, context):
            called.append(("method", spec.task_id))
            return RunResult(context.run_id, spec.task_id, "method", "completed",
                             tmp_path / "artifact", evidence=tmp_path / "evidence.json")
    app.benchmarks.register("benchmark", Benchmark()); app.methods.register("method", Method())
    assert app.run(ctx).status == "completed"
    assert called == [("benchmark", "task"), ("method", "task")]
    with pytest.raises(ValueError, match="already registered"):
        app.methods.register("method", Method())
    with pytest.raises(ValueError, match="unknown method"):
        app.methods.get("unconfigured")
    with pytest.raises(TypeError):
        app.methods.register("broken", object())
    with pytest.raises(ValueError, match="different task"):
        app.run(replace(ctx, config=replace(ctx.config, task="other")))


@pytest.mark.parametrize("mismatch", ["run_id", "task_id", "method", "evaluator_id"])
def test_sdk_does_not_accept_result_from_other_run_or_evaluator(tmp_path, mismatch):
    app = Application(); ctx = context(tmp_path)
    result = RunResult("run-1", "task", "method", "completed", tmp_path / "artifact", .5,
                       "official-v1", tmp_path / "evidence.json")
    result = replace(result, **{mismatch: "wrong"})
    app.benchmarks.register("benchmark", SimpleNamespace(task=lambda task_id, config: task(tmp_path)))
    app.methods.register("method", SimpleNamespace(run=lambda spec, context: result))
    with pytest.raises(ValueError):
        app.run(ctx)


@pytest.mark.parametrize("changes", [
    {"score": float("nan")}, {"score": True}, {"artifact": None}, {"evidence": None},
    {"evaluator_id": None}, {"status": "pending"}, {"status": "evaluation_invalid"}, {"status": "failed"},
])
def test_scored_result_requires_complete_identity_fields(tmp_path, changes):
    values = dict(run_id="r", task_id="t", method="m", status="completed", artifact=tmp_path / "artifact",
                  score=.5, evidence=tmp_path / "evidence.json", evaluator_id="e")
    values.update(changes)
    with pytest.raises(ValueError):
        RunResult(**values)


def test_missing_score_is_not_fabricated_zero():
    result = RunResult("r", "t", "m", "evaluation_invalid", reason="grader unavailable")
    assert result.score is None


def test_completed_is_not_used_for_missing_delivery():
    with pytest.raises(ValueError, match="completed delivery"):
        RunResult("r", "t", "m", "completed")


def test_validator_has_no_default_or_legacy_import():
    with pytest.raises(TypeError):
        FrozenEvoCoderValidator()
    with pytest.raises(ValueError):
        EvaluatorObservationAdapter("", lambda value: value)
    with pytest.raises(TypeError):
        EvaluatorObservationAdapter("official-v1", None)


@pytest.mark.parametrize("relation", ["supports", "refutes", "inconclusive"])
def test_validator_preserves_external_three_valued_authority(relation):
    seen = []
    evaluation = SimpleNamespace(passes_declared_execution=True, passes_hard_gates=True, native_metric=.8)
    def classify(value):
        seen.append(value)
        return SimpleNamespace(relation=relation)
    adapter = EvaluatorObservationAdapter("official-v1", classify)
    result = adapter.observation(action=SimpleNamespace(id="a", action_sha256="a" * 64), evaluation=evaluation,
        graph_before_sha256="b" * 64, graph_after_sha256="c" * 64, realized_family_sha256=None,
        model_calls=1, tokens=10, cost_usd=0.)
    assert seen == [evaluation]
    assert result.scientific_disposition == relation
    assert result.native_metric == .8


def test_classifier_failure_propagates_without_scientific_verdict():
    def broken(value):
        raise RuntimeError("evaluation chain failed")
    adapter = EvaluatorObservationAdapter("official-v1", broken)
    with pytest.raises(RuntimeError, match="evaluation chain"):
        adapter.observation(action=SimpleNamespace(id="a", action_sha256="a" * 64), evaluation=object(),
            graph_before_sha256="b" * 64, graph_after_sha256="c" * 64, realized_family_sha256=None,
            model_calls=1, tokens=10, cost_usd=0.)


def test_uniform_report_is_evidence_bound_and_keeps_a_fixed_denominator(tmp_path, monkeypatch, capsys):
    """One row shape for every method; unfinished runs are never scored as losses."""
    import json as _json
    from evolvcoder import cli
    from evolvcoder.run_report import campaign_report, run_summary
    from evolvcoder.retained_results import read_json

    absent = tmp_path / "never_started"
    row = run_summary(absent)
    assert row["evidence"] == "absent" and row["status"] == "not_started" and "score" not in row

    foreign = tmp_path / "other_method"
    (foreign / "validation").mkdir(parents=True)
    (foreign / "inputs.json").write_text(_json.dumps({"config": {"benchmark": "naturebench", "task": "t",
                                                                 "method": "aide", "seed": 0}}))
    (foreign / "result.json").write_text(_json.dumps({"score": 1.0}))
    row = run_summary(foreign)
    assert row["evidence"] == "unreadable_delivery" and row["method"] == "aide" and "score" not in row

    report = campaign_report([absent, foreign])
    assert report["denominator"] == 2 and report["delivered"] == 0 and report["new_program_wins"] == 0
    assert sorted(report["unfinished_tasks"]) == ["never_started", "t"]
    assert campaign_report([absent, foreign])["report_sha256"] == report["report_sha256"]

    out = tmp_path / "report.json"
    assert cli.main(["report", "--run", str(absent), "--run", str(foreign), "--output", str(out)]) == 0
    assert read_json(out)["denominator"] == 2 and _json.loads(capsys.readouterr().out)["new_program_wins"] == 0


def test_report_summarizes_a_real_delivery_and_its_reference(setup, tmp_path):
    from evolvcoder import unified_run
    from evolvcoder.api import Application
    from evolvcoder.run_report import campaign_report, run_summary
    fixture, _, config, _, _ = setup
    ctx = unified_run.context_for(config, "reported")
    result = Application().run(ctx)
    assert result.status == "completed"
    row = run_summary(ctx.output_dir)
    assert row["evidence"] == "sealed_delivery" and row["status"] == "completed"
    assert row["benchmark"] == "naturebench" and row["method"] == "evolvcoder" and row["task_id"] == "fixture"
    assert row["score"] == result.score and row["best_native_metric"] == result.score
    assert row["accounting"]["model_calls"] >= 0 and row["terminal_status"]
    # The fixture profile records a -1.0 reference, and the synthetic root is the
    # only valid program, so there is no new-program win to claim.
    assert row["reference"] == -1.0 and row["best_evolved_native_metric"] is None
    assert row["new_program_beats_reference"] is None
    report = campaign_report([ctx.output_dir])
    assert report["denominator"] == 1 and report["completed"] == 1 and report["new_program_wins"] == 0
    assert "strictly greater" in report["win_rule"]
