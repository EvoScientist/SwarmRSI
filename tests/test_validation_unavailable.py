"""Infrastructure failure fixtures: no real model or scientific evaluation."""
import json
from pathlib import Path
import sys
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from evolvcoder import learning_cli
from evolvcoder.learning_trials import SubprocessValidationRunner
from evolvcoder.learning_evidence import FeedbackPool
from evolvcoder.retained_results import read_json, write_immutable
from evolvcoder.rsi import RSIChain, ValidationPending
from evolvcoder.tree_policy import TreePolicy
from evolvcoder.validation_failure import ValidationUnavailable, verify_unavailable
from test_evolvcoder_learning import make_run, FileValidator, Proposer


def runner_fixture(root, cells):
    source = root / 'source'
    for directory in ('src/evolvcoder', 'scripts', 'starter'):
        (source / directory).mkdir(parents=True, exist_ok=True)
    (source / 'src/evolvcoder/fixture.py').write_text('# synthetic\n')
    (source / 'starter/run.py').write_text('# no scientific code executes\n')
    (source / 'scripts/evaluate_naturebench_local_workspace.py').write_text('# never called\n')
    (source / 'scripts/run_evolvcoder_v_search.py').write_text(
        'import sys\nraise SystemExit(0 if "--preflight" in sys.argv else 17)\n')
    profile = source / 'profile.json'
    write_immutable(profile, {'tree_search': {'policy': 'continuation_tree'},
                             'tasks': {c['task_id']: {'starter': 'starter'} for c in cells}})
    return SubprocessValidationRunner(root / 'validator', source_root=source, profile_path=profile,
        runtime_python=Path(sys.executable), cells=cells, allow_execution=True)


def paired_fixture(tmp_path, failure_position):
    cells = [{'task_id': 'first', 'seed': 0}, {'task_id': 'second', 'seed': 2}]
    pool = FeedbackPool(tmp_path / 'feedback')
    pool.add(make_run(tmp_path / 'history', role='development'))
    chain = RSIChain(tmp_path / 'chain', theta0=TreePolicy(), cells=cells, rounds=3, stop_solved_tasks=False)
    proposed = TreePolicy(parent_exploration_weight=1.)
    proposals = Proposer([proposed, TreePolicy(parent_exploration_weight=2.), TreePolicy(parent_exploration_weight=3.)])
    real_runner = runner_fixture(tmp_path / 'real_child', cells)
    good = FileValidator(tmp_path / 'valid', {TreePolicy().sha256: .1, proposed.sha256: .2})
    class Mixed:
        def evaluate(self, **kwargs):
            index = 2 * cells.index({'task_id': kwargs['task_id'], 'seed': kwargs['seed']})
            index += kwargs['policy'] != TreePolicy()
            return real_runner.evaluate(**kwargs) if index == failure_position else good.evaluate(**kwargs)
    return chain, pool, proposals, Mixed(), real_runner


@pytest.mark.parametrize('position', [0, 1, 2, 3])
def test_real_failure_halts_exact_required_pair_and_reentry(tmp_path, position):
    chain, pool, proposer, mixed, runner = paired_fixture(tmp_path, position)
    state = chain.run(proposer, mixed, pool)
    assert state['halted'] and not state['finished'] and state['status'] == 'validation_unavailable'
    assert state['active_cells'] == chain.config['cells']
    assert state['current_policy'] == TreePolicy().to_dict() and state['completed_rounds'] == 1
    assert len(proposer.requests) == 1 and runner.accounting()['unique_launches'] == 1
    row = read_json(chain.root / 'round_000/result.json')
    assert row['trial'] is None and len(row['incumbent_runs']) + len(row['candidate_runs']) == position
    assert row['validation_failure']['cost_usd'] is None and row['validation_failure']['accounting'] is None
    identity = verify_unavailable(row['validation_failure'])
    assert {'task_id': identity['task_id'], 'seed': identity['seed']} == chain.config['cells'][position // 2]
    assert len(pool.snapshot()['records']) == 1 + position
    before = {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}
    forbidden = Mock(side_effect=AssertionError('no new provider, child or credential activity'))
    assert chain.run(forbidden, forbidden, pool) == state
    assert chain.run_round(forbidden, forbidden, pool) == state
    with pytest.raises(ValueError, match='cannot export'):
        chain.export_policy(tmp_path / 'new_policy.json')
    assert not (chain.root / 'round_001').exists()
    assert before == {str(p): p.read_bytes() for p in tmp_path.rglob('*') if p.is_file()}


@pytest.mark.parametrize('filename', ['identity.json', 'launch.json', 'exit.json', 'execution.log'])
@pytest.mark.parametrize('damage', ['edit', 'delete'])
def test_halted_state_rechecks_original_failure_refs(tmp_path, filename, damage):
    chain, pool, proposer, mixed, runner = paired_fixture(tmp_path, 2)
    state = chain.run(proposer, mixed, pool)
    path = Path(state['validation_failure']['trial_root']) / filename
    if damage == 'delete':
        path.unlink()
    else:
        path.write_bytes(path.read_bytes() + b'\n')
    with pytest.raises(ValueError, match='reference mismatch'):
        chain.state()


def test_failed_child_progress_is_referenced_not_fabricated_final(tmp_path):
    cells = [{'task_id': 'fixture', 'seed': 0}]
    runner = runner_fixture(tmp_path, cells)
    root, _ = runner._prepare(TreePolicy(), **cells[0])
    progress = root / 'run/cells/0_child'
    progress.mkdir(parents=True)
    write_immutable(root / 'run/checkpoint.json', {'test': 'partial synthetic, not a final score'})
    write_immutable(progress / 'result.json', {'test': 'retained failure diagnostic'})
    with pytest.raises(ValidationUnavailable) as error:
        runner.evaluate(policy=TreePolicy(), **cells[0])
    refs = error.value.evidence['source_refs']
    assert 'run/checkpoint.json' in refs and 'run/cells/0_child/result.json' in refs
    assert not (root / 'run/final.json').exists()
    (progress / 'result.json').write_text('{}')
    with pytest.raises(ValueError, match='reference mismatch'):
        verify_unavailable(error.value.evidence)


@pytest.mark.parametrize('returncode', [None, True])
def test_unknown_termination_remains_pending(tmp_path, returncode):
    runner = runner_fixture(tmp_path, [{'task_id': 'fixture', 'seed': 0}])
    root, command = runner._prepare(TreePolicy(), 'fixture', 0)
    write_immutable(root / 'launch.json', {'synthetic': True})
    write_immutable(root / 'exit.json', {'returncode': returncode})
    with pytest.raises(ValidationPending):
        runner.evaluate(policy=TreePolicy(), task_id='fixture', seed=0)
    assert not (root / 'unavailable.json').exists()


def test_absent_exit_never_means_a_settled_failure(tmp_path, monkeypatch):
    import socket
    runner = runner_fixture(tmp_path, [{'task_id': 'fixture', 'seed': 0}])
    root, _ = runner._prepare(TreePolicy(), 'fixture', 0)
    write_immutable(root / 'launch.json', {'host': socket.gethostname()})
    monkeypatch.setattr(runner, '_process_identity', lambda _: None)
    with pytest.raises(ValidationPending, match='indeterminate'):
        runner.evaluate(policy=TreePolicy(), task_id='fixture', seed=0)
    assert not (root / 'unavailable.json').exists()


@pytest.mark.parametrize('post_final_exit', [0, 17])
def test_present_corrupted_final_is_not_reclassified_as_unavailable(tmp_path, post_final_exit):
    runner = runner_fixture(tmp_path, [{'task_id': 'fixture', 'seed': 0}])
    root, _ = runner._prepare(TreePolicy(), 'fixture', 0)
    (root / 'run').mkdir()
    (root / 'run/final.json').write_text('corrupted synthetic final')
    write_immutable(root / 'launch.json', {'test': True})
    write_immutable(root / 'exit.json', {'returncode': post_final_exit})
    with pytest.raises((ValueError, FileNotFoundError)) as error:
        runner.evaluate(policy=TreePolicy(), task_id='fixture', seed=0)
    assert not isinstance(error.value, ValidationUnavailable)
    assert not (root / 'unavailable.json').exists()


def test_crash_before_round_seal_does_not_read_credentials_or_reissue(tmp_path, monkeypatch):
    import evolvcoder.rsi as module
    chain, pool, proposer, mixed, runner = paired_fixture(tmp_path, 2)
    original_write = module.write_immutable
    def fail_result(path, value):
        if path.name == 'result.json':
            raise OSError('synthetic persistence interruption before round seal')
        return original_write(path, value)
    monkeypatch.setattr(module, 'write_immutable', fail_result)
    with pytest.raises(OSError, match='persistence interruption'):
        chain.run(proposer, mixed, pool)
    assert len(proposer.requests) == 1 and runner.accounting()['unique_launches'] == 1
    monkeypatch.setattr(module, 'write_immutable', original_write)
    constructor = Mock(side_effect=AssertionError('do not read credential or construct model'))
    state = chain.run(learning_cli._LazyPolicyProposer(constructor), mixed, pool)
    assert state['halted'] and runner.accounting()['unique_launches'] == 1
    constructor.assert_not_called()


def test_actual_cli_halted_reentry_does_not_create_proposer_or_preflight(tmp_path, monkeypatch, capsys):
    chain, pool, proposer, mixed, runner = paired_fixture(tmp_path, 0)
    chain.run(proposer, mixed, pool)
    record = {'root': str(tmp_path), 'cells': chain.config['cells']}
    monkeypatch.setattr(learning_cli, 'prepare', lambda *a, **k: record)
    monkeypatch.setattr(learning_cli, 'rsi_entry', lambda *a, **k: None)
    monkeypatch.setattr(learning_cli, 'components', lambda *a, **k: (chain, runner, pool))
    monkeypatch.setattr(learning_cli, 'accumulate', lambda *a, **k: {})
    forbidden = Mock(side_effect=AssertionError('no preflight or credential read after halt'))
    monkeypatch.setattr(learning_cli, 'proposer', forbidden)
    monkeypatch.setattr(runner, 'preflight', forbidden)
    args = SimpleNamespace(stage='run', execute=True, config=tmp_path / 'synthetic.json')
    assert learning_cli.execute(args) == 1
    report = json.loads(capsys.readouterr().out)
    assert report['state']['halted'] and report['state']['validation_failure']['accounting'] is None
    forbidden.assert_not_called()
