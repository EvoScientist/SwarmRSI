"""Original installed journals: synthetic observations, never scientific gains."""
from pathlib import Path
from unittest.mock import patch

import pytest

from evolvcoder import cli, naturebench_session as session
from evolvcoder.contracts import canonical_sha256
from evolvcoder.installed_learning import InstalledValidationRunner, portable_profile
from evolvcoder.learning_analysis import analyze_learning, export_learning_analysis
from evolvcoder.learning_evidence import source_ref
from evolvcoder.retained_results import read_json
from evolvcoder.tree_policy import TreePolicy
from evolvcoder.validation_failure import unavailable_record, ValidationUnavailable
from test_evolvcoder_learning import setup_chain, Proposer
from test_evolvcoder_learning_analysis import inventory
from test_evolvcoder_retained_results import put
from test_installed_learning import runner_for
from test_naturebench_entrypoint import inputs  # noqa: F401 -- shared pytest fixture


def prepare(inputs, credential_mode):
    if credential_mode == "file":
        runner = runner_for(inputs)
    else:
        profile = portable_profile(read_json(inputs.profile), inputs.profile)
        path = inputs.profile.parent / "fixed.json"
        put(path, profile)
        runner = InstalledValidationRunner(inputs.profile.parent / "validation", profile_path=path,
            runtime_python=Path(inputs.runtime_python), tasks_root=inputs.tasks_root,
            credential_file=None, credential_env="SYNTHETIC_ANALYSIS_AUTH", base_url=inputs.base_url,
            cells=[{"task_id": "fixture", "seed": 0}])
    trial, command = runner._prepare(TreePolicy(), "fixture", 0)
    put(trial / "launch.json", {"command": command, "host": "synthetic",
                               "runner_sha256": canonical_sha256(runner.config)})
    return runner, trial, command


@pytest.mark.parametrize("credential_mode", ["file", "env"])
def test_original_installed_argv_and_public_export_without_execution(inputs, tmp_path, monkeypatch, credential_mode):
    runner, trial, command = prepare(inputs, credential_mode)
    chain, _ = setup_chain(tmp_path / "chain")
    # Analysis must not read credentials or import/resume the old installation.
    inputs.credential_file.unlink()
    monkeypatch.delenv("SYNTHETIC_ANALYSIS_AUTH", raising=False)
    monkeypatch.setattr(session, "ROOT", tmp_path / "different_observer")
    before = inventory(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("readonly analysis attempted execution or resume")
    with patch.object(InstalledValidationRunner, "resume", forbidden), \
         patch.object(InstalledValidationRunner, "_prepare", forbidden), \
         patch.object(InstalledValidationRunner, "preflight", forbidden), \
         patch("subprocess.Popen", forbidden), patch("urllib.request.urlopen", forbidden):
        bundle = analyze_learning(chain_root=chain.root, runner_roots=[runner.root])
        assert inventory(tmp_path) == before
        assert cli.main(["analyze-learning", "--chain-root", str(chain.root),
                         "--runner-root", str(runner.root), "--output", str(tmp_path / "export")]) == 0
    row = bundle["validation_ledger"][0]
    assert row["status"] == "launch_intent_process_unknown"
    assert row["process_liveness"] == "not_checked" and row["reported_cost_usd"] is None
    assert row["wisdom_usage_batches"] is None
    assert command[1:3] == ["-m", "evolvcoder.naturebench_cli"]
    assert {k: v for k, v in inventory(tmp_path).items() if not k.startswith("export/")} == before


@pytest.mark.parametrize("damage", ["argv", "identity", "source", "executable", "profile", "membership"])
def test_installed_drift_rejected_without_export(inputs, tmp_path, damage):
    runner, trial, command = prepare(inputs, "file")
    chain, _ = setup_chain(tmp_path / "chain")
    if damage == "argv":
        row = read_json(trial / "launch.json"); row["command"] = command + ["--changed"]
        put(trial / "launch.json", row)
    elif damage == "identity":
        row = read_json(runner.root / "runner_identity.json"); row["source"]["sha256"] = "0" * 64
        put(runner.root / "runner_identity.json", row)
    elif damage == "profile":
        row = read_json(trial / "profile.json"); row["model"] = "changed"
        put(trial / "profile.json", row)
    else:
        # Never touch real product sources or interpreter. Damage only synthetic
        # refs/config and preserve a self-consistent outer identity to test inner checks.
        cfg = read_json(runner.root / "runner.json")
        if damage == "source":
            cfg["source_refs"][0]["sha256"] = "0" * 64
        elif damage == "executable":
            cfg["candidate_executable_ref"]["sha256"] = "0" * 64
        else:
            cfg["source_refs"] = [r for r in cfg["source_refs"] if not r["path"].endswith("naturebench_cli.py")]
        put(runner.root / "runner.json", cfg)
        put(runner.root / "runner_identity.json", {"source": source_ref(runner.root / "runner.json")})
    before = inventory(tmp_path)
    with pytest.raises(ValueError):
        export_learning_analysis(tmp_path / "export", chain_root=chain.root, runner_roots=[runner.root])
    assert inventory(tmp_path) == before and not (tmp_path / "export").exists()


@pytest.mark.parametrize("credential_mode", ["file", "env"])
def test_settled_unavailable_retains_failure_not_score_or_zero_usage(inputs, tmp_path, credential_mode):
    runner, trial, _ = prepare(inputs, credential_mode)
    chain, _ = setup_chain(tmp_path / "chain")
    put(trial / "exit.json", {"returncode": 17, "cost_usd": None})
    failure = unavailable_record(trial)
    before = inventory(tmp_path)
    bundle = analyze_learning(chain_root=chain.root, runner_roots=[runner.root])
    row = bundle["validation_ledger"][0]
    assert row["status"] == "validation_unavailable" and row["validation_failure"] == failure
    assert row["run_id"] is row["accounting"] is row["reported_cost_usd"] is None
    assert row["wisdom_usage_batches"] is None
    assert not bundle["validation_runs"] and inventory(tmp_path) == before


@pytest.mark.parametrize("target", ["trial", "source"])
def test_output_cannot_enter_installed_inputs(inputs, tmp_path, target):
    runner, trial, _ = prepare(inputs, "file")
    chain, _ = setup_chain(tmp_path / "chain")
    destination = (trial if target == "trial" else runner.source_root) / "analysis_should_not_exist"
    with pytest.raises(ValueError, match="input"):
        export_learning_analysis(destination, chain_root=chain.root, runner_roots=[runner.root])
    assert not destination.exists()


@pytest.mark.parametrize("credential_mode", ["file", "env"])
def test_chain_only_unavailable_protects_implicit_runner_and_source(inputs, tmp_path, credential_mode):
    runner, trial, _ = prepare(inputs, credential_mode)
    chain, pool = setup_chain(tmp_path / "chain", cells=[{"task_id": "fixture", "seed": 0}])
    put(trial / "exit.json", {"returncode": 17})
    failure = unavailable_record(trial)
    class Unavailable:
        def evaluate(self, **kwargs):
            raise ValidationUnavailable(failure)
    chain.run_round(Proposer([TreePolicy(parent_exploration_weight=1.)]), Unavailable(), pool)
    before = inventory(tmp_path)
    bundle = analyze_learning(chain_root=chain.root)
    row = bundle["policy_trials"][0]
    assert row["status"] == "validation_unavailable" and row["terminal"]
    assert row["trial"] is None and not row["comparison_complete"]
    assert row["retained_policy"] == TreePolicy().to_dict()
    for destination in (trial / "export", runner.source_root / "analysis_should_not_exist"):
        with pytest.raises(ValueError, match="referenced input"):
            export_learning_analysis(destination, chain_root=chain.root)
        assert not destination.exists()
    assert inventory(tmp_path) == before


def test_appearing_destination_keeps_other_writers_output(inputs, tmp_path, monkeypatch):
    import evolvcoder.learning_analysis as module
    runner, _, _ = prepare(inputs, "file")
    chain, _ = setup_chain(tmp_path / "chain")
    output = tmp_path / "export"
    original = module.analyze_learning
    def appeared(**kwargs):
        result = original(**kwargs)
        output.mkdir()
        (output / "other-writer.txt").write_text("preserve")
        return result
    monkeypatch.setattr(module, "analyze_learning", appeared)
    with pytest.raises(ValueError, match="output appeared"):
        module.export_learning_analysis(output, chain_root=chain.root, runner_roots=[runner.root])
    assert (output / "other-writer.txt").read_text() == "preserve"
    assert sorted(p.name for p in output.iterdir()) == ["other-writer.txt"]
    assert not list(tmp_path.glob(".learning-analysis-*"))
