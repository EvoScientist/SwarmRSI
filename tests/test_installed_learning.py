"""Installed outer/inner dispatch tests. Every score and provider is synthetic."""
import argparse
from copy import deepcopy
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from swarmrsi import cli as public_cli, learning_cli as learning, naturebench_cli as inner
from swarmrsi import naturebench_session as session
from swarmrsi.installed_learning import InstalledValidationRunner, portable_profile
from swarmrsi.learning_evidence import read_run_evidence, verify_run_evidence
from swarmrsi.retained_results import read_json, write_immutable
from swarmrsi.rsi import ValidationPending
from swarmrsi.tree_policy import TreePolicy
from test_swarmrsi_learning import make_run, FileValidator, Proposer
from test_swarmrsi_learning_runtime import validation_runner
from test_naturebench_entrypoint import inputs, synthetic_root, forbid


def config_for(inputs):
    base = inputs.profile.parent
    history = make_run(base / "history", role="development")
    profile = read_json(inputs.profile)
    profile["tasks"]["fixture"]["claude_code_score"] = .5
    inputs.profile.write_text(json.dumps(profile))
    config = {"schema_version": "swarmrsi-installed-learning-v1", "root": "learning",
              "profile": "profile.json", "tasks_root": "tasks", "runtime_python": sys.executable,
              "credential_file": "credential.json", "base_url": "https://example.invalid",
              "rounds": 2, "wisdom": {"mode": "readonly"}, "feedback_sources": [{
                  "run_dir": history["run_dir"], "source_root": history["source_root"],
                  "original_run_root": "/original/run"}]}
    path = base / "learning.json"; path.write_text(json.dumps(config))
    return path, config


def runner_for(inputs, *, execute=False):
    profile = portable_profile(read_json(inputs.profile), inputs.profile)
    path = inputs.profile.parent / "fixed.json"; path.write_text(json.dumps(profile))
    return InstalledValidationRunner(inputs.profile.parent / "validation", profile_path=path,
        runtime_python=Path(inputs.runtime_python), tasks_root=inputs.tasks_root,
        credential_file=inputs.credential_file, base_url=inputs.base_url,
        cells=[{"task_id": "fixture", "seed": 0}], allow_execution=execute)


def test_installed_prepare_resolves_original_starter_and_is_idempotent(inputs, capsys):
    path, config = config_for(inputs)
    assert public_cli.main(["learn", "prepare", "--config", str(path)]) == 0
    first = learning.prepare(path)
    assert learning.prepare(path) == first
    profile = read_json(Path(first["profile"]))
    assert profile["tasks"]["fixture"]["starter"] == str(inputs.profile.parent / "starter")
    _, runner, pool = learning.components(first)
    assert isinstance(runner, InstalledValidationRunner)
    assert len(pool.snapshot()["records"]) == 1
    root, command = runner._prepare(TreePolicy(parent_exploration_weight=1), "fixture", 0)
    assert read_json(root / "profile.json")["tasks"] == profile["tasks"]
    assert command[1:3] == ["-m", "swarmrsi.naturebench_cli"]
    assert not any("scripts/run_swarmrsi_v_search.py" in part for part in command)
    assert command[command.index("--base-url") + 1] == inputs.base_url
    assert "synthetic-never-sent" not in json.dumps(runner.config) + capsys.readouterr().out


def test_interpreter_roles_and_complete_present_absent_environment(inputs, monkeypatch):
    candidate = inputs.profile.parent / "candidate/bin/python"; candidate.parent.mkdir(parents=True)
    candidate.symlink_to(sys.executable)
    inputs.runtime_python = candidate
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "2")
    monkeypatch.setenv("LD_LIBRARY_PATH", "/synthetic/original")
    monkeypatch.delenv("CUDA_HOME", raising=False)
    monkeypatch.setenv("ARBITRARY_PRIVATE_TOKEN", "never-inherit")
    monkeypatch.setenv("PYTHONPATH", "/old/checkout")
    runner = runner_for(inputs)
    _, command = runner._prepare(TreePolicy(), "fixture", 0)
    assert command[0] == sys.executable
    assert command[command.index("--runtime-python") + 1] == str(candidate)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "9")
    monkeypatch.setenv("CUDA_HOME", "/later")
    resumed = InstalledValidationRunner.resume(runner.root)
    assert resumed._environment() == runner._environment()
    assert resumed._environment()["CUDA_VISIBLE_DEVICES"] == "2"
    assert resumed._environment()["LD_LIBRARY_PATH"] == "/synthetic/original"
    assert not ({"CUDA_HOME", "PYTHONPATH", "ARBITRARY_PRIVATE_TOKEN"} & resumed._environment().keys())


@pytest.mark.parametrize("field", ["root", "tasks_root", "credential_file"])
def test_resource_path_aliases_remain_resumable(inputs, field):
    path, config = config_for(inputs)
    alias = path.parent / "alias"; alias.symlink_to(path.parent, target_is_directory=True)
    config[field] = "alias/" + config[field]; path.write_text(json.dumps(config))
    record = learning.prepare(path)
    _, first, _ = learning.components(record)
    _, second, _ = learning.components(record)
    assert first.config == second.config
    assert learning.prepare(path) == record


@pytest.mark.parametrize("change", ["model", "starter", "limits"])
def test_generated_profile_drift_rejected_before_any_trial(inputs, change):
    path, _ = config_for(inputs); record = learning.prepare(path)
    generated = Path(record["profile"]); value = read_json(generated)
    if change == "model": value["model"] = "different"
    else: value["tasks"]["fixture"][change] = "changed"
    generated.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="generated profile"):
        learning.prepare(path)
    assert not (Path(record["root"]) / "validation/runner.json").exists()


@pytest.mark.parametrize("target", ["tasks", "starter", "history", "."])
def test_prepare_never_overlaps_inputs_or_prior_evidence(inputs, target):
    path, config = config_for(inputs)
    config["root"] = target; path.write_text(json.dumps(config))
    before = inputs.profile.read_bytes()
    with pytest.raises(ValueError, match="overlaps"):
        learning.prepare(path)
    assert not (inputs.profile.parent / target / "campaign.json").exists()
    assert inputs.profile.read_bytes() == before


def test_private_cache_rejected_before_campaign_is_written(inputs):
    path, config = config_for(inputs)
    config["model_cache"] = "tasks/fixture/evaluation"; path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="private evaluator"):
        learning.prepare(path)
    assert not (inputs.profile.parent / "learning").exists()


def test_campaign_cannot_be_created_inside_its_candidate_model_cache(inputs):
    path, config = config_for(inputs)
    cache = path.parent / "cache"; cache.mkdir()
    config.update(root="cache/campaign", model_cache="cache"); path.write_text(json.dumps(config))
    with pytest.raises(ValueError, match="overlaps"):
        learning.prepare(path)
    assert not (cache / "campaign").exists()


@pytest.mark.parametrize("field", ["execution", "environment", "timeout_seconds"])
def test_captured_runner_cannot_drift_before_first_trial(inputs, field):
    path, _ = config_for(inputs); record = learning.prepare(path)
    learning.components(record)
    runner_file = Path(record["root"]) / "validation/runner.json"
    value = read_json(runner_file)
    if field == "execution": value[field]["runtime_python"] = "/different/python"
    elif field == "environment": value[field]["CUDA_VISIBLE_DEVICES"] = "9"
    else: value[field] = 999
    runner_file.write_text(json.dumps(value))
    with pytest.raises(ValueError, match="source reference mismatch"):
        learning.components(record)


def test_real_installed_outer_help_prepare_status_and_zero_call_preflight(inputs):
    path, _ = config_for(inputs)
    env = dict(os.environ); env.pop("PYTHONPATH", None)
    base = [sys.executable, "-m", "swarmrsi.cli", "learn"]
    def invoke(stage):
        return subprocess.run([*base, stage, "--config", str(path)], cwd=path.parent,
                              env=env, text=True, capture_output=True, timeout=60)
    with patch("urllib.request.urlopen", side_effect=AssertionError("no parent network")):
        result = invoke("prepare")
        assert result.returncode == 0, result.stderr
        result = invoke("status")
        assert result.returncode == 0 and "prepared_not_started" in result.stdout
        result = invoke("preflight")
        assert result.returncode in (0, 1), result.stderr
        assert json.loads(result.stdout)["zero_model_calls"] is True
        assert not list((path.parent / "learning").rglob("launch.json"))


def test_failed_preflight_prevents_outer_proposal(inputs, monkeypatch):
    path, _ = config_for(inputs)
    monkeypatch.setattr(InstalledValidationRunner, "preflight", lambda *a, **k: {"ready": False})
    monkeypatch.setattr(learning, "proposer", forbid)
    with pytest.raises(ValueError, match="no proposal requested"):
        learning.main(["run", "--config", str(path), "--execute"])


def test_wisdom_off_arm_clones_actual_captured_environment(inputs, monkeypatch):
    path, _ = config_for(inputs)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "3")
    record = learning.prepare(path); _, runner, _ = learning.components(record)
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "7")
    monkeypatch.setenv("CUDA_PATH", "/later")
    seen = []
    def invalid(self, **kwargs):
        seen.append(self._environment())
        raise RuntimeError("synthetic invalid, not a measured score")
    monkeypatch.setattr(InstalledValidationRunner, "evaluate", invalid)
    result = learning.wisdom_ablation(record, runner, execute=True)
    assert seen[0] == seen[1] == runner._environment()
    assert seen[0]["CUDA_VISIBLE_DEVICES"] == "3" and "CUDA_PATH" not in seen[0]
    assert result["comparison"]["mean_paired_delta"] is None
    assert len(result["comparison"]["outcomes"]) == 2
    assert read_json(Path(record["root"]) / "W0.json")["snapshot_sha256"] == record["wisdom_snapshot_sha256"]


def test_control_uncertainty_is_shared_across_wisdom_arms(inputs, monkeypatch):
    path, _ = config_for(inputs); record = learning.prepare(path)
    _, on, _ = learning.components(record)
    def stop(self, **kwargs):
        raise ValidationPending("fixture does not launch")
    monkeypatch.setattr(InstalledValidationRunner, "evaluate", stop)
    learning.wisdom_ablation(record, on, execute=True)
    off = InstalledValidationRunner.resume(Path(record["root"]) / "validation_no_wisdom")
    assert on._control_record() == off._control_record() == Path(record["root"]) / "execution_uncertain.json"
    write_immutable(off._control_record(), {"synthetic": "prior off-arm cleanup uncertain"})
    monkeypatch.undo()
    with patch("subprocess.Popen", side_effect=AssertionError("no new arm on uncertain GPU")):
        with pytest.raises(ValidationPending, match="cleanup is uncertain"):
            on.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)


def test_proposer_consumes_explicit_endpoint_and_never_uses_coding_tools(inputs, monkeypatch):
    path, _ = config_for(inputs); record = learning.prepare(path)
    _, runner, _ = learning.components(record)
    seen = []
    monkeypatch.setattr(learning, "OpenRouterCoderModel", lambda **kwargs: seen.append(kwargs))
    learning.proposer(record, runner)
    assert seen[0]["base_url"] == "https://example.invalid"
    assert seen[0]["model"] == "synthetic/model" and not seen[0]["native_tools"]


def test_installed_terminal_is_reverified_without_spawn_secret_or_root_score(inputs, monkeypatch, capsys):
    runner = runner_for(inputs)
    root, command = runner._prepare(TreePolicy(), "fixture", 0)
    parser = argparse.ArgumentParser(); inner.add_arguments(parser)
    args = parser.parse_args(command[3:])
    monkeypatch.setattr(session, "evaluate_root", synthetic_root)
    monkeypatch.setattr(session, "coder_sandbox_config", lambda _: {"available": True})
    monkeypatch.setattr(session, "public_task_context", lambda *a, **k: "synthetic")
    monkeypatch.setattr(session, "build_search_models", lambda *a: (SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    assert inner.execute(args) == 0
    inputs.credential_file.unlink()
    monkeypatch.setattr(session, "evaluate_root", forbid)
    with patch("subprocess.Popen", side_effect=AssertionError("must not spawn")):
        evidence = runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)
        verify_run_evidence(evidence)
        assert evidence["utility"] == .1 and evidence["best_evolved_native_metric"] is None
        assert runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0) == evidence
    assert runner.accounting()["unique_launches"] == 0
    final = root / "run/final.json"; final.write_text(final.read_text() + " ")
    with pytest.raises(ValueError):
        runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)


@pytest.mark.parametrize("timeout", [False, True])
def test_child_process_group_cleanup_on_normal_exit_and_timeout(tmp_path, timeout):
    runner = validation_runner(tmp_path)
    root = tmp_path / "group"; root.mkdir()
    # The child confirms its handler before the leader can exit. It survives
    # TERM, so the KILL assertion exercises a live owned descendant rather than
    # racing a fast zombie reaper during the second signal on macOS.
    child = "import signal,time; signal.signal(signal.SIGTERM,signal.SIG_IGN); print('ready',flush=True); time.sleep(300)"
    code = ("import subprocess,sys,time; p=subprocess.Popen([sys.executable,'-c'," + repr(child)
            + "],stdout=subprocess.PIPE,text=True); assert p.stdout.readline().strip()=='ready'; print(p.pid,flush=True)")
    if timeout: code += ";time.sleep(300)"
    runner.config["timeout_seconds"] = .2 if timeout else 10
    rc, failure = runner._execute_process(root, [sys.executable, "-c", code])
    assert failure == ("validation_wall_timeout" if timeout else None)
    if not timeout: assert rc == 0
    pid = int((root / "execution.log").read_text().strip())
    # ps may briefly report a dead child as a zombie; it cannot be a worker.
    state = subprocess.run(["ps", "-p", str(pid), "-o", "stat="], text=True, capture_output=True).stdout.strip()
    assert not state or state.startswith("Z")


@pytest.mark.parametrize("startup_delay", [0, .25])
def test_real_child_clock_finishes_before_watchdog_including_startup(tmp_path, startup_delay):
    """Real process/timer smoke, not a benchmark score or a scientific final."""
    runner = validation_runner(tmp_path)
    root = tmp_path / "clock-child"
    root.mkdir()
    runner.config["timeout_seconds"] = 1.5
    code = """
import json,os,sys,time
from pathlib import Path
from swarmrsi.session_runtime import SessionClock, runtime_clock_controls
time.sleep(float(sys.argv[2]))
origin=float(os.environ['SWARMRSI_TRIAL_STARTED_AT_UNIX'])
clock=SessionClock(started_unix=time.time(), started_monotonic=time.monotonic(),
    wall_limit=1.5, identity={'synthetic_timer_only':True},
    runtime_controls=runtime_clock_controls(1.5,1.5), trial_started_unix=origin)
initial=clock.remaining()
while clock.remaining()>0:
    time.sleep(min(.005,clock.remaining()))
Path(sys.argv[1],'timer_final.json').write_text(json.dumps({
    'origin':clock.origin,'remaining_at_entry':initial,'finished_at_unix':time.time()}))
"""
    rc, failure = runner._execute_process(root, [sys.executable, "-I", "-c", code, str(root), str(startup_delay)])
    assert rc == 0 and failure is None
    receipt = read_json(root / "process.json")
    final = read_json(root / "timer_final.json")
    assert final["origin"]["trial_started_at_unix"] == receipt["started_at_unix"]
    assert final["remaining_at_entry"] <= 1.425 - startup_delay
    assert final["finished_at_unix"] < receipt["started_at_unix"] + 1.5


def test_receipt_failure_still_cleans_actual_started_process(tmp_path, monkeypatch):
    import swarmrsi.learning_trials as module
    runner = validation_runner(tmp_path)
    root = tmp_path / "receipt"; root.mkdir()
    original = module.write_immutable
    def fail_receipt(path, value):
        if path.name == "process.json":
            raise OSError("synthetic process receipt failure")
        return original(path, value)
    monkeypatch.setattr(module, "write_immutable", fail_receipt)
    with patch.object(module.os, "killpg", wraps=os.killpg) as kill:
        with pytest.raises(OSError, match="receipt failure"):
            runner._execute_process(root, [sys.executable, "-c", "import time; time.sleep(300)"])
        assert [c.args[1] for c in kill.call_args_list] == [signal.SIGTERM, signal.SIGKILL]


def test_postfinal_control_error_preserves_evidence_and_blocks_next_trial(tmp_path, monkeypatch):
    runner = validation_runner(tmp_path, execute=True)
    original = make_run(tmp_path / "result", task_id="fixture")
    calls = []
    def failed(root, command):
        calls.append(command)
        write_immutable(root / "run/final.json", {"dispatch_sentinel": "not used as a score"})
        raise OSError("synthetic cleanup uncertainty")
    monkeypatch.setattr(runner, "_execute_process", failed)
    monkeypatch.setattr(runner, "_completed", lambda *a: original)
    assert runner.evaluate(policy=TreePolicy(), task_id="fixture", seed=0) == original
    with pytest.raises(ValidationPending, match="cleanup is uncertain"):
        runner.evaluate(policy=TreePolicy(parent_exploration_weight=1), task_id="fixture", seed=0)
    assert len(calls) == 1


def test_secondary_control_receipt_failure_keeps_final_and_blocks_another_arm(inputs, monkeypatch):
    import swarmrsi.learning_trials as module
    path, _ = config_for(inputs); record = learning.prepare(path)
    _, on, _ = learning.components(record, execute=True)
    off_root = Path(record["root"]) / "validation_no_wisdom"
    InstalledValidationRunner.write_config(off_root, deepcopy(on.config))
    off = InstalledValidationRunner.resume(off_root, allow_execution=True)
    sentinel = {"run_id": "dispatch-only", "evidence_sha256": "a" * 64}
    checked = []
    monkeypatch.setattr(on, "preflight", lambda **k: {"ready": True})
    def error(root, command):
        write_immutable(root / "run/final.json", {"synthetic": "ordering only"})
        raise OSError("synthetic cleanup")
    monkeypatch.setattr(on, "_execute_process", error)
    monkeypatch.setattr(on, "_completed", lambda *a: checked.append(True) or sentinel)
    original = module.write_immutable
    def unavailable(path, value):
        if path == on._control_record(): raise OSError("secondary receipt unavailable")
        return original(path, value)
    monkeypatch.setattr(module, "write_immutable", unavailable)
    assert on.evaluate(policy=TreePolicy(), task_id="fixture", seed=0) == sentinel
    assert checked == [True] and not on._control_record().exists()
    with patch("subprocess.Popen", side_effect=AssertionError("cannot launch another arm")):
        with pytest.raises(ValidationPending, match="cleanup is uncertain"):
            off.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)
        # Simulate a new observer process: the original unclosed launch still
        # prevents respend even when the transient in-memory flag is absent.
        module._UNCERTAIN_EXECUTION_SCOPES.discard(on._control_record().parent)
        resumed = InstalledValidationRunner.resume(off_root, allow_execution=True)
        with pytest.raises(ValidationPending, match="unsettled launch"):
            resumed.evaluate(policy=TreePolicy(), task_id="fixture", seed=0)


def test_complete_installed_campaign_dispatch_accumulates_two_policy_rounds(inputs, monkeypatch, capsys):
    path, config = config_for(inputs)
    # Reuse the original file-bound policy-trial fixture. It is deliberately
    # synthetic, while comparison and FeedbackPool re-verification are real.
    p1 = TreePolicy(parent_exploration_weight=1); p2 = TreePolicy(parent_exploration_weight=2)
    fixture = FileValidator(inputs.profile.parent / "trials", {TreePolicy().sha256: .2, p1.sha256: .3, p2.sha256: .25})
    proposal = Proposer([p1, p2])
    monkeypatch.setattr(learning, "proposer", lambda *a: proposal)
    monkeypatch.setattr(InstalledValidationRunner, "preflight", lambda *a, **k: {"ready": True})
    monkeypatch.setattr(InstalledValidationRunner, "evaluate", lambda self, **kwargs: fixture.evaluate(**kwargs))
    # Use the fixture's actual matched model/baseline so accumulation validates
    # it too; never bypass the actual evidence verifier or compare_paired.
    example = make_run(inputs.profile.parent / "example", task_id="fixture")
    source_profile = example["comparison_context"]["fixed_profile"]
    profile = read_json(inputs.profile); profile["model"] = example["model"]
    profile["tasks"]["fixture"].update({k: source_profile["tasks"]["won"].get(k) for k in ("claude_code_score", "transcript_sha256")})
    # Fixture make_run's profile uses won, so align it at the fixture boundary.
    def measured(self, **kwargs):
        ev = fixture.evaluate(**kwargs)
        pp = Path(ev["source_root"]) / "profile.json"
        old = read_json(pp)
        if "fixture" not in old["tasks"]:
            old["tasks"]["fixture"] = deepcopy(old["tasks"]["won"]); pp.write_text(json.dumps(old))
            manifest = Path(ev["run_dir"]) / "manifest.json"; m = read_json(manifest)
            from swarmrsi.retained_results import file_sha
            m["source_hashes"]["profile.json"] = file_sha(pp); manifest.write_text(json.dumps(m))
            ev = read_run_evidence(Path(ev["run_dir"]), Path(ev["source_root"]), original_run_root="/original/run", source_role="selection")
            fixture.runs[(kwargs["policy"].sha256, kwargs["task_id"], kwargs["seed"])] = ev
        return ev
    monkeypatch.setattr(InstalledValidationRunner, "evaluate", measured)
    inputs.profile.write_text(json.dumps(profile))
    config["stop_solved_tasks"] = False; path.write_text(json.dumps(config))
    assert learning.main(["run", "--config", str(path), "--execute"]) == 0
    output = json.loads(capsys.readouterr().out.splitlines()[-1])
    assert output["state"]["completed_rounds"] == 2
    assert [r["status"] for r in output["state"]["round_records"]] == ["accepted", "rejected"]
    assert output["state"]["current_policy"] == p1.to_dict()
    assert len(proposal.requests[1]["feedback"]["runs"]) > len(proposal.requests[0]["feedback"]["runs"])
    assert (path.parent / "learning/frozen_policy.json").exists()
    inputs.credential_file.unlink()
    monkeypatch.setattr(learning, "proposer", forbid)
    assert learning.main(["run", "--config", str(path), "--execute"]) == 0
