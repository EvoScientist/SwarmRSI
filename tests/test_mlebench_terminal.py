"""Private-terminal custody regressions; synthetic execution and grade outputs."""
from dataclasses import replace
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace
import pytest
from swarmrsi import mlebench_run as bridge, unified_run
from swarmrsi import container_runtime
from swarmrsi.configuration import ContainerOptions, RuntimeConfig
from test_candidate_execution import IMAGE, PYTHON, completed_action
from swarmrsi.api import RunResult
from swarmrsi.contracts import canonical_sha256
from swarmrsi.learning_evidence import source_ref


def write(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value))


@pytest.fixture
def workflow(tmp_path, monkeypatch):
    output = tmp_path / 'run'; output.mkdir()
    write(output / 'inputs.json', {'synthetic': True})
    write(output / 'result.json', {'original_public_delivery': True})
    original = tmp_path / 'public_program'; original.mkdir()
    (original / 'run.py').write_text('def main():\n    return 1\n')
    final = tmp_path / 'public_evidence/final.json'
    write(final, {'best_node_id': 'child'})
    write(final.parent / 'checkpoint.json', {'nodes': [{'id': 'root', 'artifact_valid': True}, {'id': 'child', 'artifact_valid': True}]})
    prepared = tmp_path / 'data/prepared'; (prepared / 'public').mkdir(parents=True); (prepared / 'private').mkdir()
    custody = {'prepared': str(prepared), 'official_source': str(tmp_path / 'official')}
    assets = {'custody': tmp_path / 'custody.json', 'terminal_timeout_seconds': 600}
    config = SimpleNamespace(task='synthetic', method='swarmrsi', runtime=RuntimeConfig(python=sys.executable),
                             limits=SimpleNamespace(wall_seconds=3600), to_dict=lambda: {'synthetic': True})
    context = SimpleNamespace(run_id='synthetic', output_dir=output, config=config)
    public = RunResult('synthetic', 'synthetic', 'swarmrsi', 'completed', original, .6,
                       'synthetic-authority', final, 'public_search_done', {'evaluations': 3}, {})
    monkeypatch.setattr(bridge, 'manifest', lambda config: assets)
    monkeypatch.setattr(bridge, 'verify_assets', lambda path: custody)
    monkeypatch.setattr(unified_run, 'local_admission', lambda config: ({}, {}))
    monkeypatch.setattr(bridge.shutil, 'which', lambda name: '/synthetic/bwrap')
    calls = []
    def process(argv, *, cwd, env, deadline, log):
        calls.append(argv); log.write_text('synthetic execution\n')
        if '--submission' in argv:
            write(output / 'terminal/official_report.json', {'competition_id': 'synthetic', 'valid_submission': True,
                'score': .7, 'submission_path': argv[argv.index('--submission') + 1]})
        else:
            (output / 'terminal/program/output/submission.csv').write_text('Insult,Comment\n.3,synthetic\n')
        return 0
    monkeypatch.setattr(bridge, '_process', process)
    return SimpleNamespace(context=context, public=public, adapter=SimpleNamespace(_operate=lambda *a, **k: public),
                           task=SimpleNamespace(task_id='synthetic'), prepared=prepared, calls=calls, output=output)


def test_terminal_replay_and_environment_isolation(workflow, monkeypatch):
    import os
    w = workflow; monkeypatch.setenv('SWARMRSI_MODEL_CACHE', str(w.prepared / 'private'))
    before = dict(os.environ)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.score == .7 and result.details['public_auc'] == .6
    assert result.details['new_scored_candidates'] == 1 and len(w.calls) == 2
    assert bridge.workflow_run(w.adapter, w.task, w.context) == result and len(w.calls) == 2
    assert str(w.prepared / 'private') not in w.calls[0] and dict(os.environ) == before


def test_terminal_binds_manifest_model_cache(workflow, monkeypatch, tmp_path):
    # Search sees the cache through its session environment; the terminal runs after
    # that environment is restored, so it must bind the manifest cache itself.
    w = workflow; cache = tmp_path / 'model_cache'; cache.mkdir()
    bridge.manifest(w.context.config)['model_cache'] = cache
    monkeypatch.delenv('SWARMRSI_MODEL_CACHE', raising=False)
    bridge.workflow_run(w.adapter, w.task, w.context)
    argv = w.calls[0]; bind = argv.index(str(cache.resolve()))
    assert argv[bind - 1] == '--ro-bind' and argv[bind + 1] == '/model_cache'
    launch = json.loads((w.output / 'terminal/launch.json').read_text())
    assert launch['environment']['TORCH_HOME'] == '/model_cache'


def reseal(w):
    path = w.output / 'benchmark_result.json'; record = json.loads(path.read_text()); record.pop('receipt_sha256')
    record['refs'] = [source_ref(p) for p in sorted((w.output / 'terminal').glob('*')) if p.is_file()]
    write(path, {**record, 'receipt_sha256': canonical_sha256(record)})


@pytest.mark.parametrize('name', ['selection.json', 'launch.json', 'execution.log', 'grade_launch.json', 'grader.log', 'official_report.json'])
def test_missing_terminal_evidence_rejected(workflow, name):
    w = workflow; bridge.workflow_run(w.adapter, w.task, w.context)
    (w.output / 'terminal' / name).unlink(); reseal(w)
    with pytest.raises((ValueError, FileNotFoundError)): bridge._receipt(w.context, w.public)


@pytest.mark.parametrize('fault', ['task', 'submission', 'grade', 'selection', 'source', 'score', 'output'])
def test_changed_terminal_evidence_rejected(workflow, fault):
    w = workflow; bridge.workflow_run(w.adapter, w.task, w.context)
    terminal = w.output / 'terminal'
    if fault in {'task', 'submission', 'score'}:
        path = terminal / 'official_report.json'; value = json.loads(path.read_text())
        value[{'task': 'competition_id', 'submission': 'submission_path', 'score': 'score'}[fault]] = .99 if fault == 'score' else 'foreign'
        write(path, value)
    elif fault == 'grade':
        path = terminal / 'grade_launch.json'; value = json.loads(path.read_text())
        value['argv'][value['argv'].index('--submission') + 1] = 'foreign.csv'; write(path, value)
    elif fault == 'selection':
        path = terminal / 'launch.json'; value = json.loads(path.read_text()); value['selection'] = {}; write(path, value)
    else:
        path = terminal / ('program/run.py' if fault == 'source' else 'program/output/submission.csv')
        path.write_text('changed')
    reseal(w)
    with pytest.raises(ValueError): bridge._receipt(w.context, w.public)


def test_internal_validation_rewrite_is_output_not_source_change(workflow, monkeypatch):
    """r7 regression: run.py obeys the coder instruction and rewrites internal_validation.json."""
    w = workflow
    (w.public.artifact / 'internal_validation.json').write_text('{"estimate": 0.5}')
    inner = bridge._process
    def process(argv, *, cwd, env, deadline, log):
        if '--submission' not in argv:
            (w.output / 'terminal/program/internal_validation.json').write_text('{"estimate": 0.51}')
        return inner(argv, cwd=cwd, env=env, deadline=deadline, log=log)
    monkeypatch.setattr(bridge, '_process', process)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed' and result.score == .7
    record = json.loads((w.output / 'benchmark_result.json').read_text())
    assert record['source_sha256'] != record['workspace_sha256']
    assert bridge._receipt(w.context, w.public).status == 'completed'
    (w.output / 'terminal/program/helper.py').write_text('planted'); reseal(w)
    with pytest.raises(ValueError, match='terminal source changed'): bridge._receipt(w.context, w.public)


def test_uncertain_terminal_not_reexecuted(workflow):
    w = workflow; (w.output / 'terminal').mkdir()
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'pending' and not w.calls


@pytest.fixture
def container(workflow, monkeypatch):
    """Same custody fixture with an explicit container backend; no Docker daemon.

    execute_candidate installs a genuine controller-owned action record per call
    so _receipt's verify_receipt runs the production projection.
    """
    w = workflow
    w.context.config.runtime = RuntimeConfig(kind='container', python=sys.executable, image=IMAGE, cpus=2, memory_mb=1024,
        container=ContainerOptions(python=PYTHON, uid=max(1, os.getuid()), gid=max(1, os.getgid()), tmp_mb=128))
    w.executions, w.fault = [], {}
    def execute(binding, *, workspace, public_data, argv, timeout, cwd='/workspace', environment=None):
        w.executions.append(list(argv))
        if 'raise' in w.fault:
            raise w.fault['raise']
        _, state = bridge.validate_binding(binding)
        state.mkdir(mode=0o700, parents=True, exist_ok=True)
        (Path(workspace) / 'output/submission.csv').write_text('Insult,Comment\n.3,synthetic\n')
        _, receipt, _ = completed_action(state.parent, monkeypatch, exit_code=w.fault.get('exit_full' if environment else 'exit', 0), argv=argv, state=state,
                                         workspace=Path(workspace).resolve(), public=Path(public_data).resolve())
        return receipt
    monkeypatch.setattr(bridge, 'execute_candidate', execute)
    w.binding = bridge.for_context(w.context)
    return w


def test_container_terminal_runs_frozen_program_in_bound_backend_and_replays(container):
    w = container
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed' and result.score == .7 and result.details['public_auc'] == .6
    terminal = w.output / 'terminal'
    launch = json.loads((terminal / 'launch.json').read_text())
    assert launch['candidate_runtime'] == w.binding and launch['argv'] == [PYTHON, 'run.py'] and launch['sandboxed'] is True
    assert w.executions == [[PYTHON, 'run.py']] and len(w.calls) == 1 and '--submission' in w.calls[0]
    receipt = json.loads((terminal / 'candidate_runtime.json').read_text())
    assert receipt['status'] == 'candidate_exited' and receipt['cleanup_confirmed'] is True
    assert (terminal / 'execution.log').read_text() == receipt['stdout'] + receipt['stderr']
    assert Path(receipt['record_dir']).parent == w.output / 'candidate_runtime'
    assert bridge.workflow_run(w.adapter, w.task, w.context) == result and len(w.executions) == 1


@pytest.mark.parametrize('fault', ['receipt', 'backend_to_local', 'backend_spec', 'foreign_action'])
def test_container_terminal_receipt_and_backend_binding_verified(container, monkeypatch, fault):
    w = container; bridge.workflow_run(w.adapter, w.task, w.context)
    path = w.output / 'terminal/candidate_runtime.json'
    if fault == 'receipt':
        value = json.loads(path.read_text()); value['returncode'] = 1; write(path, value); reseal(w)
    elif fault == 'backend_to_local':
        w.context.config.runtime = RuntimeConfig(python=sys.executable)
    elif fault == 'backend_spec':
        w.context.config.runtime = replace(w.context.config.runtime, cpus=3)
    else:
        # A real sibling record for another command must not stand in for the terminal program.
        _, receipt, _ = completed_action(w.output, monkeypatch, argv=[PYTHON, '-c', 'pass'], state=w.output / 'candidate_runtime',
                                         workspace=w.output / 'terminal/program', public=w.prepared / 'public')
        write(path, receipt); reseal(w)
    with pytest.raises(ValueError):
        bridge._receipt(w.context, w.public)


def test_container_backend_failure_is_recorded_once_without_retry(container):
    w = container
    w.fault['raise'] = container_runtime.RuntimeUnavailable('dispatch uncertain; inspect the record')
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'evaluation_invalid' and 'dispatch uncertain' in result.reason and not w.calls
    w.fault.clear()
    assert bridge.workflow_run(w.adapter, w.task, w.context) == result and len(w.executions) == 1


def test_container_candidate_failure_is_invalid_not_graded(container):
    w = container; w.fault['exit'] = 3
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'evaluation_invalid' and 'terminal container execution failed' in result.reason and not w.calls
    assert json.loads((w.output / 'terminal/candidate_runtime.json').read_text())['returncode'] == 3
    assert bridge.workflow_run(w.adapter, w.task, w.context) == result and len(w.executions) == 1


def test_container_full_refit_crash_falls_back_and_receipt_verifies(container, monkeypatch):
    w = container
    _with_profile(w, monkeypatch, {'submission_fit_policy': 'full_refit'})
    w.fault['exit_full'] = 1
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed' and len(w.executions) == 2
    assert json.loads((w.output / 'terminal/candidate_runtime.json').read_text())['returncode'] == 1
    assert json.loads((w.output / 'terminal/candidate_runtime_search_fit.json').read_text())['returncode'] == 0
    assert bridge._receipt(w.context, w.public).status == 'completed'


def test_container_preflight_qualifies_backend_instead_of_bwrap(container, monkeypatch):
    w = container
    qualified = []
    monkeypatch.setattr(bridge.shutil, 'which', lambda name: None)
    monkeypatch.setattr(container_runtime, 'preflight', lambda spec: qualified.append(spec))
    monkeypatch.setattr(bridge.subprocess, 'run', lambda *a, **k: SimpleNamespace(returncode=0, stderr=''))
    adapter = SimpleNamespace(_operate=lambda task, context, execute: {'ready': True})
    ready = bridge.workflow_preflight(adapter, w.task, w.context)
    assert ready['ready'] and qualified == [bridge.validate_binding(w.binding)[0]]
    w.context.config.runtime = RuntimeConfig(python=sys.executable)
    with pytest.raises(ValueError, match='bubblewrap'):
        bridge.workflow_preflight(adapter, w.task, w.context)


def _with_profile(w, monkeypatch, screening):
    """Give the fixture's manifest a profile carrying the given screening limits."""
    assets = dict(bridge.manifest(w.context.config))
    profile = w.output.parent / 'profile.json'
    write(profile, {'tasks': {'synthetic': {}}, 'screening_execution_limits': screening})
    monkeypatch.setattr(bridge, 'manifest', lambda config: {**assets, 'profile': profile})


def _capture_env(w, monkeypatch, on_candidate=None):
    seen = []
    inner = bridge._process
    def process(argv, *, cwd, env, deadline, log):
        seen.append((argv, dict(env)))
        if '--submission' not in argv and on_candidate is not None:
            on_candidate()
        return inner(argv, cwd=cwd, env=env, deadline=deadline, log=log)
    monkeypatch.setattr(bridge, '_process', process)
    return seen


def test_submission_fit_flag_set_only_under_policy_and_report_recorded(workflow, monkeypatch):
    """full_refit: the terminal execution is flagged, and what the program says it fit on is kept."""
    w = workflow
    _with_profile(w, monkeypatch, {'submission_fit_policy': 'full_refit'})
    def program_reports():
        (w.output / 'terminal/program/internal_validation.json').write_text(json.dumps({
            'metric_name': 'auc', 'direction': 'maximize', 'estimate': .61,
            'submission_fit': {'mode': 'full', 'rows_available': 1000, 'rows_fit': 1000}}))
    seen = _capture_env(w, monkeypatch, program_reports)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed' and result.score == .7
    candidate_env = next(env for argv, env in seen if '--submission' not in argv)
    assert candidate_env[bridge.SUBMISSION_FIT_ENV] == 'full'
    record = json.loads((w.output / 'benchmark_result.json').read_text())
    assert record['submission_fit'] == {'policy': 'full_refit', 'reported': True, 'mode': 'full',
                                        'rows_available': 1000, 'rows_fit': 1000}
    assert result.details['submission_fit'] == record['submission_fit']
    # the rewrite is candidate output, not a source change: the receipt still verifies
    assert bridge._receipt(w.context, w.public).status == 'completed'


def test_submission_fit_report_absent_is_recorded_not_fatal(workflow, monkeypatch):
    w = workflow
    _with_profile(w, monkeypatch, {'submission_fit_policy': 'full_refit'})
    _capture_env(w, monkeypatch)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed'
    assert result.details['submission_fit'] == {'policy': 'full_refit', 'reported': False}


def test_full_refit_crash_falls_back_to_the_search_fit_program(workflow, monkeypatch):
    """The all-rows branch the search never ran crashes: the searched path runs once more."""
    w = workflow
    _with_profile(w, monkeypatch, {'submission_fit_policy': 'full_refit'})
    seen = []
    inner = bridge._process
    def process(argv, *, cwd, env, deadline, log):
        seen.append(dict(env))
        if '--submission' not in argv and env.get(bridge.SUBMISSION_FIT_ENV) == 'full':
            log.write_text('IndexError in the refit branch')
            return 1
        return inner(argv, cwd=cwd, env=env, deadline=deadline, log=log)
    monkeypatch.setattr(bridge, '_process', process)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'completed' and result.score == .7
    candidates = seen[:2]  # the two candidate executions; grading comes after
    assert candidates[0].get(bridge.SUBMISSION_FIT_ENV) == 'full'
    assert bridge.SUBMISSION_FIT_ENV not in candidates[1]
    record = json.loads((w.output / 'benchmark_result.json').read_text())
    assert record['submission_fit']['fallback'] == 'search_fit_after_full_refit_failure'
    assert record['terminal_evaluations'] == 2
    assert (w.output / 'terminal/execution_search_fit.log').exists()
    assert bridge._receipt(w.context, w.public).status == 'completed'


def test_legacy_terminal_failure_does_not_retry(workflow, monkeypatch):
    w = workflow
    calls = []
    def process(argv, *, cwd, env, deadline, log):
        calls.append(argv)
        log.write_text('crash')
        return 1
    monkeypatch.setattr(bridge, '_process', process)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert result.status == 'evaluation_invalid' and 'terminal execution failed' in result.reason
    assert len(calls) == 1


def test_legacy_terminal_sets_no_submission_fit_flag(workflow, monkeypatch):
    """No profile, or a profile without the policy: nothing about the terminal run changes."""
    w = workflow
    seen = _capture_env(w, monkeypatch)
    result = bridge.workflow_run(w.adapter, w.task, w.context)
    assert all(bridge.SUBMISSION_FIT_ENV not in env for _, env in seen)
    assert result.details['submission_fit'] == {'policy': 'legacy'}
