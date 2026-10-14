"""Public config reaches the full inner lifecycle; all measurements synthetic.

The candidate process boundary is substituted in local tests, not the runner,
RSI, checkpoint, evidence verifier or Application dispatch. Installed Linux
qualification is separate; these tests never call a real provider or grader.
"""
import argparse
from contextlib import redirect_stdout
from dataclasses import replace
import io
import json
import os
from pathlib import Path
from types import SimpleNamespace
import threading

import pytest

from swarmrsi.api import Application
from swarmrsi.configuration import config_from_dict
from swarmrsi import cli, unified_run as unified, naturebench_cli as inner, naturebench_session as session
from swarmrsi.installed_learning import InstalledValidationRunner
from swarmrsi.retained_results import read_json
from swarmrsi.tree_policy import TreePolicy
from swarmrsi.rsi import ValidationPending
from test_naturebench_entrypoint import inputs, synthetic_root


@pytest.fixture
def setup(inputs, monkeypatch):
    profile = read_json(inputs.profile)
    profile["screening_execution_limits"] = {"max_input_tokens_per_action_safety": 100000,
        "max_output_tokens_per_action": 30000, "wall_time_seconds_per_action": 6000}
    inputs.profile.write_text(json.dumps(profile))
    manifest = inputs.profile.parent / "assets.json"
    manifest.write_text(json.dumps({"schema_version": "swarmrsi-naturebench-assets-v1", "profile": "profile.json",
                                    "tasks_root": "tasks", "trial_timeout_seconds": 100}))
    raw = {"benchmark": "naturebench", "task": "fixture", "model": {"model": "synthetic/model", "api_key_env": None,
        "api_key_file": str(inputs.credential_file), "base_url": "https://example.invalid"},
        "runtime": {"python": inputs.runtime_python}, "benchmark_config": str(manifest), "output_root": "outputs",
        "limits": {"model_calls": 20, "tool_calls": 20, "evaluations": 4,
                   "input_tokens": 100000, "output_tokens": 30000, "wall_seconds": 6000}}
    config = config_from_dict(raw, base=inputs.profile.parent)
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "Synthetic public context")
    models = []
    monkeypatch.setattr(session, "build_search_models", lambda *a: models.append(a) or
                        (SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    executions = []
    def preflight(runner, **cell):
        _, command = runner._prepare(**cell)
        parser = argparse.ArgumentParser(); inner.add_arguments(parser)
        args = parser.parse_args(command[3:])
        inner.configure(args)
        return {"ready": True, "zero_model_calls": True}
    def execute(runner, root, command):
        executions.append(command)
        parser = argparse.ArgumentParser(); inner.add_arguments(parser)
        args = parser.parse_args(command[3:])
        with redirect_stdout(io.StringIO()):
            code = inner.execute(args)
        return code, None
    monkeypatch.setattr(InstalledValidationRunner, "preflight", preflight)
    monkeypatch.setattr(InstalledValidationRunner, "_execute_process", execute)
    return inputs, raw, config, models, executions


def test_full_single_sdk_and_cli_reentry(setup, monkeypatch, capsys):
    inputs, raw, config, models, executions = setup
    context = unified.context_for(config, "first")
    before = {n: getattr(session, n) for n in unified._GLOBALS}
    environment = dict(os.environ)
    app = Application()
    assert app.status(context)["status"] == "not_started"
    assert not context.output_dir.exists()
    assert app.preflight(context)["ready"] and not models and not executions
    result = app.run(context)
    assert result.status == "completed" and result.score == .1
    assert result.details["best_evolved_native_metric"] is None and result.details["best_node_id"] == "root"
    assert len(models) == len(executions) == 1
    assert all(getattr(session, n) == v for n, v in before.items())
    assert dict(os.environ) == environment
    profile = read_json(context.output_dir / "profile.json")
    assert profile["tasks"]["fixture"]["limits"]["wall_time_seconds"] == 6000
    assert profile["session_runtime_controls"] == {
        "declared_inner_wall_seconds": 6000, "trial_watchdog_seconds": 100,
        "session_finalize_reserve_seconds": 5.0, "session_execution_wall_seconds": 95.0}
    assert profile["tasks"]["fixture"]["starter"] == str(inputs.profile.parent / "starter")
    assert profile["model_settings"]["timeout_seconds"] == config.model.timeout_seconds
    runner = read_json(context.output_dir / "validation/runner.json")
    assert runner["execution"]["base_url"] == config.model.base_url
    assert runner["environment"]["CUDA_VISIBLE_DEVICES"] == ""
    assert result == app.run(context)
    assert app.status(context)["result"]["score"] == .1
    inputs.credential_file.unlink()
    path = inputs.profile.parent / "config.json"; path.write_text(json.dumps(raw))
    assert cli.main(["run", "--config", str(path), "--run-id", "first"]) == 0
    assert json.loads(capsys.readouterr().out)["score"] == .1
    assert len(models) == len(executions) == 1
    assert len(list((context.output_dir / "feedback").glob("*.json"))) == 1


def test_identity_change_and_corrupt_terminal_fail_closed(setup):
    _, _, config, _, executions = setup
    ctx = unified.context_for(config, "same")
    app = Application(); result = app.run(ctx)
    with pytest.raises(ValueError, match="identity"):
        app.run(replace(ctx, config=replace(config, seed=1)))
    result.evidence.write_text(result.evidence.read_text() + " ")
    with pytest.raises(ValueError):
        app.run(ctx)
    assert len(executions) == 1


def test_imported_policy_and_none_limits_reach_full_runner(setup):
    inputs, _, config, _, _ = setup
    policy = TreePolicy(parent_exploration_weight=1.2)
    path = inputs.profile.parent / "policy.json"; path.write_text(json.dumps(policy.to_dict()))
    config = replace(config, learning=replace(config.learning, policy=path),
                     limits=replace(config.limits, model_calls=None, input_tokens=None))
    ctx = unified.context_for(config)
    result = Application().run(ctx)
    assert result.status == "completed"
    trial = next((ctx.output_dir / "validation/trials").iterdir())
    profile = read_json(trial / "profile.json")
    assert profile["rsi_policy"] == policy.to_dict()
    assert profile["tasks"]["fixture"]["limits"]["backbone_message_ids"] is None
    assert profile["tasks"]["fixture"]["limits"]["tokens"]["input"] is None
    assert read_json(trial / "run/checkpoint.json")["strategy_state"]


def test_unsafe_env_and_unsupported_options_no_outputs(setup):
    _, _, config, models, executions = setup
    for model in (replace(config.model, requests_per_minute=3),
                  replace(config.model, api_key_file=None, api_key_env="PYTHONHOME")):
        ctx = unified.context_for(replace(config, model=model))
        with pytest.raises(ValueError):
            Application().run(ctx)
        assert not ctx.output_dir.exists()
    assert not models and not executions


def test_unsettled_launch_is_pending_not_reissued(setup, monkeypatch):
    _, _, config, _, executions = setup
    ctx = unified.context_for(config)
    app = Application(); app.preflight(ctx)
    trial = next((ctx.output_dir / "validation/trials").iterdir())
    (trial / "launch.json").write_text(json.dumps({"host": "other-host", "runner_sha256": read_json(trial / "identity.json")["runner_sha256"]}))
    monkeypatch.setattr(unified, "proposer", lambda *a: pytest.fail("no proposal"))
    assert app.run(ctx).status == "pending"
    assert app.run(ctx).status == "pending"
    assert not executions


def test_settled_failed_trial_preserved_no_next_proposal(setup, monkeypatch):
    inputs, _, config, _, executions = setup
    config = replace(config, learning=replace(config.learning, rsi_enabled=True,
                                            feedback_store=inputs.profile.parent / "pool"))
    ctx = unified.context_for(config)
    app = Application(); app.preflight(ctx)
    trial = next((ctx.output_dir / "validation/trials").iterdir())
    (trial / "launch.json").write_text(json.dumps({"host": "other-host", "runner_sha256": read_json(trial / "identity.json")["runner_sha256"]}))
    (trial / "exit.json").write_text(json.dumps({"returncode": 2, "failure": "synthetic"}))
    (trial / "execution.log").write_text("synthetic failed code retained")
    monkeypatch.setattr(unified, "proposer", lambda *a: pytest.fail("no proposal"))
    result = app.run(ctx)
    assert result.status == "failed" and result.score is None
    assert app.run(ctx) == result and not executions
    assert (trial / "execution.log").read_text() == "synthetic failed code retained"


def test_rsi_complete_pair_and_fixed_wisdom_accumulate(setup, monkeypatch):
    from swarmrsi.wisdom import WisdomLibrary
    inputs, _, config, models, executions = setup
    w0 = inputs.profile.parent / "W0.json"
    WisdomLibrary(inputs.profile.parent / "seed-wisdom").freeze(w0)
    config = replace(config, learning=replace(config.learning, rsi_enabled=True, rsi_rounds=1,
        wisdom_enabled=True, wisdom_snapshot=w0, feedback_store=inputs.profile.parent / "pool"))
    ctx = unified.context_for(config)
    requests = []
    candidate = TreePolicy(parent_exploration_weight=1.1)
    class Proposer:
        def propose(self, request):
            requests.append(request)
            return {"parameters": candidate.to_dict()["parameters"], "rationale": "Synthetic parameter test only",
                    "evidence_run_ids": [request["feedback"]["runs"][0]["run_id"]]}, {"model_calls": 1, "cost_usd": None}
    monkeypatch.setattr(unified, "proposer", lambda *a: Proposer())
    result = Application().run(ctx)
    assert result.status == "completed" and len(executions) == 2 and len(requests) == 1
    assert result.details["rsi"]["round_records"][0]["status"] == "rejected"
    assert result.details["rsi"]["current_policy"] == TreePolicy().to_dict()
    assert len(list(config.learning.feedback_store.glob("*.json"))) == 2
    profiles = [read_json(p) for p in (ctx.output_dir / "validation/trials").glob("*/profile.json")]
    assert all(p["wisdom"] == {"mode": "readonly", "snapshot_path": str(w0)} for p in profiles)
    assert len(models) == 2
    config.model.api_key_file.unlink()
    monkeypatch.setattr(unified, "proposer", lambda *a: pytest.fail("terminal must not open provider"))
    assert Application().run(ctx) == result


def test_alternating_sdk_configs_restore_and_concurrent_rejected(setup):
    _, _, config, _, executions = setup
    first = unified.context_for(config, "one")
    second = unified.context_for(replace(config, model=replace(config.model, base_url="https://second.invalid")), "two")
    before = {n: getattr(session, n) for n in unified._GLOBALS}
    assert Application().run(first).status == "completed"
    assert Application().run(second).status == "completed"
    assert all(getattr(session, n) == value for n, value in before.items())
    errors = []
    with unified.session_scope(None):
        def attempt():
            try:
                Application().run(first)
            except ValidationPending as exc:
                errors.append(str(exc))
        thread = threading.Thread(target=attempt); thread.start(); thread.join(timeout=5)
        assert not thread.is_alive()
    assert errors and len(executions) == 2


def test_public_context_does_not_include_evaluator_paths(setup):
    _, _, config, _, _ = setup
    task = unified.NatureBenchAdapter().task(config.task, config)
    assert str(task.evaluator_config) not in json.dumps(task.public_context())
    assert task.metric_direction == "maximize"


def test_work_root_and_unknown_manifest_fields_reject(setup):
    inputs, _, config, _, executions = setup
    bad = replace(config, runtime=replace(config.runtime, work_root=inputs.profile.parent / "scratch"))
    with pytest.raises(ValueError, match="work_root"):
        Application().run(unified.context_for(bad))
    manifest = read_json(config.benchmark_config); manifest["secret"] = "forbidden"
    config.benchmark_config.write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="unknown fields"):
        Application().run(unified.context_for(config))
    assert not executions


def test_two_rsi_rounds_compare_current_incumbent_and_reuse(setup, monkeypatch):
    inputs, _, config, _, executions = setup
    first = TreePolicy(parent_exploration_weight=1.1)
    second = TreePolicy(parent_exploration_weight=1.2)
    policies = iter([first, second]); requests = []
    scores = {TreePolicy().sha256: .1, first.sha256: .2, second.sha256: .15}
    def root_measurement(**kwargs):
        synthetic_root(**kwargs)
        run_dir = kwargs["run_dir"]
        policy = TreePolicy.from_dict(read_json(run_dir.parent / "profile.json")["rsi_policy"])
        score = scores[policy.sha256]
        path = run_dir / "root_evaluation.json"; record = read_json(path)
        record["metrics"]["aggregate_improvement"] = score
        session.atomic_json(path, record)
        return score, True, .01, session.compact_evaluation_summary(record), session.sha(path)
    monkeypatch.setattr(session, "evaluate_root", root_measurement)
    class Proposer:
        def propose(self, request):
            requests.append(request)
            return {"parameters": next(policies).to_dict()["parameters"], "rationale": "Synthetic paired utility test",
                    "evidence_run_ids": [request["feedback"]["runs"][0]["run_id"]]}, {"model_calls": 1, "cost_usd": None}
    monkeypatch.setattr(unified, "proposer", lambda *a: Proposer())
    config = replace(config, learning=replace(config.learning, rsi_enabled=True, rsi_rounds=2,
                                            feedback_store=inputs.profile.parent / "pool"))
    ctx = unified.context_for(config)
    result = Application().run(ctx)
    assert result.score == .2 and len(executions) == 3
    assert requests[1]["incumbent"] == first.to_dict()
    assert [r["status"] for r in result.details["rsi"]["round_records"]] == ["accepted", "rejected"]
    assert result.details["validation_accounting"]["unique_launches"] == 3
    assert read_json(ctx.output_dir / "learned_policy.json")["policy"] == first.to_dict()
    # Corrupt a historical paired journal, while leaving the best run intact.
    history = ctx.output_dir / "rsi/round_000/proposal_request.json"
    history.write_text(history.read_text() + " ")
    with pytest.raises(ValueError):
        Application().status(ctx)
    assert len(executions) == 3


@pytest.mark.parametrize("registry_type", ["historical", "learning", "collection"])
def test_retained_winners_skip_before_key_or_runner(setup, monkeypatch, registry_type):
    from test_swarmrsi_retained_results import run as archive, put
    from test_swarmrsi_learning import make_run
    from swarmrsi.retained_results import build_registry
    from swarmrsi.learning_evidence import retain_learning_winners, source_ref
    from swarmrsi.contracts import canonical_sha256
    inputs, _, config, _, executions = setup
    historical = inputs.profile.parent / "history"
    registry_path = inputs.profile.parent / "registry.json"
    if registry_type == "historical":
        historical.mkdir()
        spec, path = archive.__wrapped__(historical)
        registry = build_registry([spec], path)
    else:
        evidence = make_run(historical)
        winners = retain_learning_winners([evidence], evidence["comparison_context"]["fixed_profile"], inputs.profile.parent / "retained")
        registry = winners[0]
        if registry_type == "collection":
            path = inputs.profile.parent / "retained" / (evidence["run_id"] + ".json")
            body = {"schema_version": "swarmrsi-retained-learning-collection-v1", "winner_refs": [source_ref(path)]}
            registry = {**body, "collection_sha256": canonical_sha256(body)}
    put(registry_path, registry)
    profile = read_json(inputs.profile)
    task = profile["tasks"].pop("fixture")
    task.update(claude_code_score=.1, transcript_sha256="c" * 64)
    profile["tasks"]["won"] = task
    put(inputs.profile, profile)
    (inputs.tasks_root / "fixture").rename(inputs.tasks_root / "won")
    config = replace(config, task="won", model=replace(config.model, model="same-model"), winner_registry=registry_path)
    ctx = unified.context_for(config)
    inputs.credential_file.unlink()
    monkeypatch.setattr(unified, "local_admission", lambda *a: pytest.fail("retained must not need a runtime launch"))
    result = Application().run(ctx)
    assert result.status == "retained" and result.score == .2 and result.details["new_version_score"] is None
    assert not executions and not (ctx.output_dir / "validation").exists()
    assert Application().status(ctx)["status"] == "retained"


def test_env_secret_private_through_sdk_and_exception_restoration(setup, monkeypatch):
    _, _, config, _, executions = setup
    name, secret = "UNIFIED_TEST_MODEL_AUTH", "synthetic-unified-key-never-log"
    monkeypatch.setenv(name, secret)
    config = replace(config, model=replace(config.model, api_key_file=None, api_key_env=name))
    seen = []
    def models(*args):
        seen.append(session.secret_key())
        assert name not in os.environ
        return SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)
    monkeypatch.setattr(session, "build_search_models", models)
    ctx = unified.context_for(config)
    assert Application().run(ctx).status == "completed"
    assert seen == [secret] and len(executions) == 1 and os.environ[name] == secret
    for path in ctx.output_dir.rglob("*"):
        if path.is_file():
            assert secret.encode() not in path.read_bytes()
    before = {n: getattr(session, n) for n in unified._GLOBALS}
    with pytest.raises(RuntimeError):
        with unified.session_scope(None):
            session.SECRET_ENV = "temporary"
            raise RuntimeError("synthetic")
    assert all(getattr(session, n) == value for n, value in before.items())


def test_next_wisdom_preserves_previous_source_entries(setup):
    from test_swarmrsi_learning import make_run
    from swarmrsi.wisdom import WisdomLibrary
    inputs, _, config, _, _ = setup
    prior = make_run(inputs.profile.parent / "history", role="development")
    library = WisdomLibrary(inputs.profile.parent / "seed-library"); library.ingest(prior)
    w0 = inputs.profile.parent / "W0.json"; previous = library.freeze(w0)
    config = replace(config, learning=replace(config.learning, wisdom_enabled=True, wisdom_snapshot=w0))
    ctx = unified.context_for(config)
    assert Application().run(ctx).status == "completed"
    after = read_json(ctx.output_dir / "wisdom_next.json")
    assert set(previous["sources"]).issubset(after["sources"])
    assert previous["entries"][0] in after["entries"]
    assert read_json(w0) == previous


def container_runtime_config(inputs):
    import os
    from swarmrsi.configuration import ContainerOptions, RuntimeConfig
    from test_candidate_execution import IMAGE, PYTHON
    return RuntimeConfig(kind="container", python=inputs.runtime_python, image=IMAGE, cpus=1, memory_mb=1024,
                         container=ContainerOptions(python=PYTHON, uid=max(1, os.getuid()), gid=max(1, os.getgid()), tmp_mb=128))


def test_container_runtime_binding_reaches_inner_session_and_is_restored(setup, monkeypatch):
    import subprocess
    from swarmrsi.candidate_execution import for_context
    from test_candidate_execution import PYTHON
    inputs, raw, config, models, executions = setup
    config = replace(config, runtime=container_runtime_config(inputs))
    # Host admission reads /proc; the backend itself is never launched here.
    monkeypatch.setattr(unified, "local_admission", lambda config: ({"CUDA_VISIBLE_DEVICES": ""}, {}))
    probes = []
    def probe(script):
        probes.append(script)
        return subprocess.CompletedProcess([PYTHON, "-I", "-c", script], 0,
                                           json.dumps({n: True for n in session.COMMON_SCIENTIFIC_IMPORT_ROOTS}), "")
    monkeypatch.setattr(session, "runtime_probe", probe)
    seen, original = [], InstalledValidationRunner._execute_process
    def execute(runner, root, command):
        result = original(runner, root, command)
        seen.append(session.CANDIDATE_RUNTIME)
        return result
    monkeypatch.setattr(InstalledValidationRunner, "_execute_process", execute)
    context = unified.context_for(config, "container")
    binding = for_context(context)
    assert binding["state_root"] == str(context.output_dir / "candidate_runtime")
    result = Application().run(context)
    assert result.status == "completed" and result.score == .1
    assert read_json(context.output_dir / "profile.json")["candidate_runtime"] == binding
    trial = next((context.output_dir / "validation/trials").iterdir())
    assert read_json(trial / "profile.json")["candidate_runtime"] == binding
    assert seen == [binding] and session.CANDIDATE_RUNTIME is None and "import pyflakes" in probes
    assert len(executions) == 1 and Application().run(context) == result and len(executions) == 1


def test_container_runtime_rejects_model_cache_and_profile_supplied_binding(setup, monkeypatch):
    inputs, raw, config, models, executions = setup
    config = replace(config, runtime=container_runtime_config(inputs))
    monkeypatch.setattr(unified, "local_admission", lambda config: ({}, {}))
    manifest = inputs.profile.parent / "assets.json"
    (inputs.profile.parent / "cache").mkdir()
    manifest.write_text(json.dumps({**read_json(manifest), "model_cache": "cache"}))
    with pytest.raises(ValueError, match="model-cache"):
        Application().run(unified.context_for(config, "cache"))
    manifest.write_text(json.dumps({k: v for k, v in read_json(manifest).items() if k != "model_cache"}))
    profile = read_json(inputs.profile)
    profile["candidate_runtime"] = {"kind": "docker"}
    inputs.profile.write_text(json.dumps(profile))
    with pytest.raises(ValueError, match="RunConfig"):
        Application().run(unified.context_for(config, "profile"))
    assert not models and not executions
