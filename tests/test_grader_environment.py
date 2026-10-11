"""A scorer that cannot run must be named as such, before and during a session.

On 2026-09-22 s43588-025-00878-7 scored every candidate invalid because the
scoring runtime lacked pyarrow for the ground-truth pickles, and
s42256-024-00795-w because numpy 2.4 removed np.trapz. Both were reported to the
planner as a candidate problem ("cannot parse numeric reference from None").
"""
import json
import pickle
import subprocess
import sys
from pathlib import Path

import pytest

from evolvcoder import naturebench_evaluator as bridge


@pytest.mark.parametrize("text", [
    "small: No module named 'pyarrow'",
    "ModuleNotFoundError: No module named 'tensorflow'",
    "AttributeError: module 'numpy' has no attribute 'trapz'",
    "`np.trapz` was removed in the NumPy 2.4 release",
    "ImportError: cannot import name 'x' from 'sklearn.metrics'",
    "OSError: /lib64/libc.so.6: version `GLIBC_2.34' not found",
])
def test_environment_signatures_are_recognised(text):
    assert bridge._grader_environment_failure(text)


@pytest.mark.parametrize("text", [
    "Missing predictions for 12/551 samples.",
    "predicted_probability must be in [0.0, 1.0]",
    "Output file not found: /out/gtex_v8/small/predictions.csv",
    "",
])
def test_candidate_output_errors_are_not_environment_failures(text):
    assert not bridge._grader_environment_failure(text)


def test_metric_names_ending_in_error_are_not_errors():
    row = {"relative_l2_error": 1.0, "mean_absolute_error": 0.0, "laplace_error": None, "flag_error": False}
    assert bridge._row_errors(row) == []


def test_numeric_error_metric_scores_normally(tmp_path):
    grader = "import json\njson.dump({'inst': {'AUC': 0.9, 'relative_l2_error': 1.0}}, open('score.json', 'w'))\n"
    task, workspace = _task(tmp_path, grader)
    assert _score(tmp_path, task, workspace)["valid"] is True


def test_row_errors_collects_sub_part_errors_only():
    row = {"AUC": None, "small_error": "No module named 'pyarrow'", "large_error": None,
           "error": "", "avg_AUC": None}
    assert bridge._row_errors(row) == ["small_error: No module named 'pyarrow'"]


def _task(tmp_path, grader_body, *, sota="~0.822"):
    task = tmp_path / "task"
    (task / "evaluation").mkdir(parents=True)
    (task / "problem/data").mkdir(parents=True)
    (task / "metadata.json").write_text(json.dumps({"performance_entries": [{
        "dataset_name": "inst", "metrics": [{"name": "AUC", "is_primary": True,
        "metric_direction": "higher_is_better", "sota_score": [{"value": sota}]}]}]}))
    (task / "evaluation/evaluator.py").write_text(grader_body)
    workspace = tmp_path / "ws"
    workspace.mkdir()
    (workspace / "run.py").write_text(
        "import os, pathlib\n"
        "def main():\n"
        "    out = pathlib.Path(os.environ['OUTPUT_DIR']); out.mkdir(parents=True, exist_ok=True)\n"
        "    (out / 'p.csv').write_text('a\\n1\\n')\n"
        "if __name__ == '__main__':\n    main()\n")
    return task, workspace


def _score(tmp_path, task, workspace):
    result = tmp_path / "result.json"
    rc = subprocess.run([sys.executable, "-m", "evolvcoder.naturebench_evaluator",
                         "--workspace", str(workspace), "--result", str(result),
                         "--task-root", str(task), "--runtime-python", sys.executable,
                         "--timeout-seconds", "120"], capture_output=True, text=True)
    assert result.is_file(), rc.stderr[-2000:]
    return json.loads(result.read_text())


def test_sub_part_environment_error_is_named_and_not_charged_to_candidate(tmp_path):
    grader = ("import json\n"
              "json.dump({'inst': {'AUC': None, 'small_error': \"No module named 'pyarrow'\"}},"
              " open('score.json', 'w'))\n")
    task, workspace = _task(tmp_path, grader)
    evidence = _score(tmp_path, task, workspace)
    assert evidence["valid"] is False
    diagnostic = evidence["public_diagnostic"]
    assert diagnostic["category"] == "grader_environment"
    assert diagnostic["candidate_only"] is False
    assert "pyarrow" in diagnostic["summary"]


def test_sub_part_candidate_error_reaches_the_planner(tmp_path):
    grader = ("import json\n"
              "json.dump({'inst': {'AUC': None, 'small_error': 'Missing predictions for 12/551 samples.'}},"
              " open('score.json', 'w'))\n")
    task, workspace = _task(tmp_path, grader)
    diagnostic = _score(tmp_path, task, workspace)["public_diagnostic"]
    assert diagnostic["category"] == "official_evaluation" and diagnostic["candidate_only"] is True
    assert "Missing predictions for 12/551" in diagnostic["summary"]


def test_valid_score_is_unchanged(tmp_path):
    grader = "import json\njson.dump({'inst': {'AUC': 0.8856}}, open('score.json', 'w'))\n"
    task, workspace = _task(tmp_path, grader)
    evidence = _score(tmp_path, task, workspace)
    assert evidence["valid"] is True
    assert evidence["metrics"]["aggregate_improvement"] == pytest.approx((0.8856 - 0.822) / 0.822)


def test_grader_crash_on_missing_module_is_environment(tmp_path):
    task, workspace = _task(tmp_path, "import module_that_is_not_installed_anywhere\n")
    diagnostic = _score(tmp_path, task, workspace)["public_diagnostic"]
    assert diagnostic["category"] == "grader_environment"


def test_readiness_passes_a_runnable_grader(tmp_path):
    pd = pytest.importorskip("pandas")
    task, _ = _task(tmp_path, "import json\nimport numpy as np\ntry:\n    import optional_thing\n"
                              "except ImportError:\n    pass\nx = np.trapezoid([1, 2])\n")
    pd.DataFrame({"a": [1, 2]}).to_pickle(task / "evaluation/y_ref.pkl")
    report = bridge.grader_readiness(task, Path(sys.executable))
    assert report["ready"] is True, report


def test_readiness_names_every_problem_before_any_call(tmp_path):
    pytest.importorskip("pandas")
    task, _ = _task(tmp_path, "import json\nimport numpy as np\nimport module_that_is_not_installed_anywhere\n"
                              "x = np.attribute_that_numpy_does_not_have(1)\n")
    (task / "evaluation/y_ref.pkl").write_bytes(b"not a pickle")
    report = bridge.grader_readiness(task, Path(sys.executable))
    assert report["ready"] is False
    assert report["missing_modules"] == ["module_that_is_not_installed_anywhere"]
    assert report["missing_attributes"] == ["numpy.attribute_that_numpy_does_not_have"]
    assert [u["path"] for u in report["unreadable_ground_truth"]] == ["y_ref.pkl"]


def test_readiness_ignores_modules_local_to_the_evaluation_directory(tmp_path):
    task, _ = _task(tmp_path, "import metrics\n")
    (task / "evaluation/metrics.py").write_text("VALUE = 1\n")
    assert bridge.grader_readiness(task, Path(sys.executable))["ready"] is True



def test_readiness_warns_but_passes_on_imports_under_an_ordinary_if(tmp_path):
    task, _ = _task(tmp_path, "import json\nmeta = {}\nif meta.get('images'):\n"
                              "    from per_task_helper_not_here import check\n"
                              "if __name__ == '__main__':\n    import json\n")
    report = bridge.grader_readiness(task, Path(sys.executable))
    assert report["ready"] is True, report
    assert report["conditional_missing_modules"] == ["per_task_helper_not_here"]
    assert report["warnings"] and "per_task_helper_not_here" in report["warnings"][0]


def test_readiness_still_requires_imports_under_the_main_guard(tmp_path):
    task, _ = _task(tmp_path, "if __name__ == '__main__':\n    import module_that_is_not_installed_anywhere\n")
    report = bridge.grader_readiness(task, Path(sys.executable))
    assert report["ready"] is False
    assert report["missing_modules"] == ["module_that_is_not_installed_anywhere"]

# --- session level: stop paying once the scorer is known not to run ----------

from types import SimpleNamespace  # noqa: E402

from evolvcoder import naturebench_cli as cli, naturebench_session as session  # noqa: E402
from evolvcoder.campaign import CampaignStep  # noqa: E402
from evolvcoder.contracts import ProposalRejected  # noqa: E402
from test_naturebench_entrypoint import inputs, synthetic_root  # noqa: E402,F401

ENVIRONMENT = {"public_diagnostic": {"category": "grader_environment", "candidate_only": False,
                                     "summary": "inst: official evaluator returned an error: small_error: "
                                                "No module named 'pyarrow'"}}
SCORED = {"public_diagnostic": None}


def _session(inputs, monkeypatch, outcomes):
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    inputs.profile.write_text(json.dumps(raw))
    cli.configure(inputs)
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic context")
    monkeypatch.setattr(session, "build_search_models", lambda *a: (
        SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    monkeypatch.setattr(session.VSearchCoordinator, "expand", lambda _: 1)
    calls = []

    def step(loop):
        if len(calls) == len(outcomes):
            raise ProposalRejected("fixture end", code="fixture_end", action_id="fixture")
        summary, valid = outcomes[len(calls)]
        calls.append(summary)
        loop.last_evaluation_summary = summary
        return CampaignStep(step=len(calls), action_id=f"a{len(calls)}", code_parent_id="root",
                            coder_status="success", native_metric=(0.5 if valid else None),
                            artifact_valid=valid, graph_sha256="synthetic")

    monkeypatch.setattr(session.VSearchCoordinator, "step", step)
    assert cli.execute(inputs) == 0
    return calls, json.loads((inputs.run_dir / "final.json").read_text())


def test_two_scorer_environment_failures_stop_the_session(inputs, monkeypatch):
    calls, final = _session(inputs, monkeypatch, [(ENVIRONMENT, False)] * 5)
    assert len(calls) == 2
    assert final["stop_reason"] == "grader_environment_failure"
    assert final["stop_details"]["consecutive_grader_environment_failures"] == 2
    assert "pyarrow" in final["stop_details"]["public_diagnostic"]["summary"]


def test_a_scored_candidate_between_failures_resets_the_count(inputs, monkeypatch):
    calls, final = _session(inputs, monkeypatch, [(ENVIRONMENT, False), (SCORED, True), (ENVIRONMENT, False)])
    assert len(calls) == 3
    assert final["stop_reason"] == "search_proposal_rejected"


def test_readiness_gate_stops_before_the_first_model_call(inputs, monkeypatch):
    task_evaluator = inputs.tasks_root / "fixture/evaluation/evaluator.py"
    task_evaluator.write_text("import module_that_is_not_installed_anywhere\n")
    raw = json.loads(inputs.profile.read_text())
    raw["tasks"]["fixture"]["claude_code_score"] = 10.0
    inputs.profile.write_text(json.dumps(raw))
    cli.configure(inputs)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic context")

    class NoCalls:
        native_tools = True

        def respond(self, request):
            raise AssertionError("a model call was made before the scorer was checked")

    monkeypatch.setattr(session, "build_search_models", lambda *a: (NoCalls(), NoCalls()))
    monkeypatch.setattr(session, "evaluate_root", lambda **kw: pytest.fail("root evaluated before gate"))
    with pytest.raises(SystemExit, match="module_that_is_not_installed_anywhere"):
        session.run_session(inputs)
    report = json.loads((inputs.run_dir / "grader_readiness.json").read_text())
    assert report["ready"] is False


def test_readiness_refuses_a_task_tree_without_its_evaluator(tmp_path):
    task, _ = _task(tmp_path, "VALUE = 1\n")
    (task / "evaluation/evaluator.py").unlink()
    (task / "evaluation/ground_truth").mkdir()
    report = bridge.grader_readiness(task, Path(sys.executable))
    assert report == {"ready": False, "problems": ["evaluation/evaluator.py is missing"]}


def test_non_finite_candidate_score_is_named_as_the_candidates(tmp_path):
    grader = "import json\njson.dump({'inst': {'AUC': float('nan')}}, open('score.json', 'w'))\n"
    task, workspace = _task(tmp_path, grader)
    diagnostic = _score(tmp_path, task, workspace)["public_diagnostic"]
    assert diagnostic["category"] == "official_evaluation"
    assert "degenerate predictions" in diagnostic["summary"] and "reference" not in diagnostic["summary"]


def test_evaluator_error_on_stdout_is_not_hidden_by_stderr_warnings(tmp_path):
    grader = ("import sys\n"
              "print('OpenBLAS warning: precompiled NUM_THREADS exceeded', file=sys.stderr)\n"
              "print('ERROR: ground truth for haber_intestinal not found')\n"
              "sys.exit(1)\n")
    task, workspace = _task(tmp_path, grader)
    summary = _score(tmp_path, task, workspace)["public_diagnostic"]["summary"]
    assert "haber_intestinal not found" in summary and "OpenBLAS" not in summary


def test_reference_parsing_matches_the_official_aggregate_for_00833():
    metadata = {"performance_entries": [
        {"dataset_name": "gset_graph_maxcut", "metrics": [{"name": "cut_size", "is_primary": True,
         "metric_direction": "higher_is_better",
         "sota_score": [{"value": "G14: 3042, G15: 3030, G22: 13185, G49: 5999, G50: 5879, G55: 10150"}]}]},
        {"dataset_name": "ndc_hypergraph_maxcut", "metrics": [{"name": "cut_size", "is_primary": True,
         "metric_direction": "higher_is_better", "sota_score": [{"value": "28649 ± 31"}]}]},
        {"dataset_name": "synthetic_hypergraph_maxcut", "metrics": [{"name": "cut_size", "is_primary": True,
         "metric_direction": "higher_is_better", "sota_score": [{"value": None}]}]},
    ]}
    refs = bridge._references(metadata)
    assert refs == {"ndc_hypergraph_maxcut": ("cut_size", "higher_is_better", 28649.0)}
    # Claude Code's recorded submission: NDC cut 28987 -> per_instance_improvement 0.01179796851548047
    assert (28987 - refs["ndc_hypergraph_maxcut"][2]) / 28649.0 == pytest.approx(0.01179796851548047)


@pytest.mark.parametrize("value,expected", [("3.1 (0.13)", 3.1), ("13.8 (1.6%)", 13.8), ("~0.822", 0.822),
                                            ("0.76 ± 0.05", 0.76), ("~3.2e-2", 0.032), (0.951, 0.951)])
def test_single_references_still_parse(value, expected):
    assert bridge._single_reference(value) == pytest.approx(expected)


def test_single_nested_item_is_scored_like_the_official_aggregate(tmp_path):
    task, workspace = _task(tmp_path, "import json\njson.dump({'inst': {'NDC': {'AUC': 0.8856}}}, open('score.json', 'w'))\n")
    evidence = _score(tmp_path, task, workspace)
    assert evidence["valid"] is True
    assert evidence["metrics"]["aggregate_improvement"] == pytest.approx((0.8856 - 0.822) / 0.822)


def test_several_nested_items_remain_ambiguous(tmp_path):
    grader = "import json\njson.dump({'inst': {'a': {'AUC': 0.9}, 'b': {'AUC': 0.8}}}, open('score.json', 'w'))\n"
    task, workspace = _task(tmp_path, grader)
    assert _score(tmp_path, task, workspace)["valid"] is False
