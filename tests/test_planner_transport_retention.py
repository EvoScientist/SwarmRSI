"""Provider-failure terminal custody: synthetic requests only, no network."""
import json

import pytest

from swarmrsi import naturebench_cli as cli, naturebench_session as ns
from test_naturebench_entrypoint import inputs, forbid  # noqa: F401
from test_session_elapsed_deadline import prepare


def failing_provider(inputs, monkeypatch, *, valid=True, earlier_response=False,
                     exception_type=RuntimeError):
    _, received = prepare(inputs, monkeypatch, valid=valid, two_batches=earlier_response)
    profile = json.loads(inputs.profile.read_text())
    profile['screening_execution_limits']['max_transport_retries'] = 0
    inputs.profile.write_text(json.dumps(profile))
    model = ns.build_search_models(None, None, None)[1]
    original = model.respond

    def respond(request):
        if earlier_response and not received:
            return original(request)
        received.append(model.request_total_timeout_seconds)
        raise exception_type('OpenRouter request failed: Remote end closed connection; secret=DO_NOT_PERSIST')

    monkeypatch.setattr(model, 'respond', respond)
    return received


@pytest.mark.parametrize('valid', [True, False])
@pytest.mark.parametrize('earlier_response', [True, False])
@pytest.mark.parametrize('exception_type', [RuntimeError, ConnectionError, TimeoutError])
def test_provider_failure_seals_retained_result_without_replay(
        inputs, monkeypatch, valid, earlier_response, exception_type):
    received = failing_provider(inputs, monkeypatch, valid=valid,
                                earlier_response=earlier_response, exception_type=exception_type)
    if valid:
        assert cli.execute(inputs) == 0
    else:
        with pytest.raises(RuntimeError, match='session returned 1'):
            cli.execute(inputs)
    final = json.loads((inputs.run_dir/'final.json').read_text())
    checkpoint = json.loads((inputs.run_dir/'checkpoint.json').read_text())
    assert final['stop_reason'] == ('planner_transport_failed' if valid else 'planner_transport_failed_without_incumbent')
    assert final['best_node_id'] == ('root' if valid else None)
    assert final['stop_details']['incumbent_retained'] is valid
    assert final['stop_details']['exception_type'] == exception_type.__name__
    assert final['stop_details']['resends'] == 0
    assert checkpoint['proposer_state']['terminal_intent']['stop_reason'] == final['stop_reason']
    assert final['accounting']['model_calls'] == int(earlier_response)
    assert final['planner_transport']['unresolved_cost_attempts'] == 1
    assert final['planner_transport']['pending_request_batches'] == [int(earlier_response)]
    assert len(received) == 1 + int(earlier_response)
    assert not any('DO_NOT_PERSIST' in p.read_text() for p in inputs.run_dir.rglob('*.json'))
    monkeypatch.setattr(ns, 'run_session', forbid)
    monkeypatch.setattr(ns, 'build_search_models', forbid)
    inputs.credential_file.unlink()
    assert cli.execute(inputs) == (0 if valid else 1)
    (inputs.run_dir/'root/run.py').write_text('# tampered\n')
    with pytest.raises(ValueError, match='program artifact mismatch'):
        cli.execute(inputs)


def test_provider_failure_terminal_checkpoint_cannot_replay_after_final_write_crash(inputs, monkeypatch):
    received = failing_provider(inputs, monkeypatch)
    original = ns.atomic_json

    def write(path, value):
        if path.name == 'final.json':
            raise OSError('synthetic final write failure')
        original(path, value)

    monkeypatch.setattr(ns, 'atomic_json', write)
    with pytest.raises(OSError, match='final write failure'):
        cli.execute(inputs)
    monkeypatch.setattr(ns, 'atomic_json', original)
    with pytest.raises(ValueError, match='unfinished prior launch'):
        cli.execute(inputs)
    with pytest.raises(ValueError, match='terminal checkpoint'):
        ns.run_session(inputs)
    assert len(received) == 1


def test_model_programming_error_is_not_reclassified_as_provider_stop(inputs, monkeypatch):
    failing_provider(inputs, monkeypatch, exception_type=ValueError)
    with pytest.raises(ValueError):
        cli.execute(inputs)
    assert not (inputs.run_dir/'final.json').exists()


def test_exhausted_existing_retry_policy_seals_all_unknown_attempts(inputs, monkeypatch):
    received = failing_provider(inputs, monkeypatch, exception_type=TimeoutError)
    profile = json.loads(inputs.profile.read_text())
    profile['screening_execution_limits']['max_transport_retries'] = 1
    inputs.profile.write_text(json.dumps(profile))
    monkeypatch.setattr(ns.time, 'sleep', lambda _: None)
    assert cli.execute(inputs) == 0
    final = json.loads((inputs.run_dir/'final.json').read_text())
    assert len(received) == 2
    assert final['stop_details']['resends'] == 1
    assert final['planner_transport']['unresolved_cost_attempts'] == 2
    assert final['accounting']['model_calls'] == 0


def test_step_boundary_retains_provider_failure(inputs, monkeypatch):
    received = failing_provider(inputs, monkeypatch)
    expand = ns.VSearchCoordinator.expand
    monkeypatch.setattr(ns.VSearchCoordinator, 'expand', lambda _: 1)
    monkeypatch.setattr(ns.VSearchCoordinator, 'step', expand)
    assert cli.execute(inputs) == 0
    final = json.loads((inputs.run_dir/'final.json').read_text())
    assert final['stop_reason'] == 'planner_transport_failed'
    assert final['stop_details']['incumbent_retained']
    assert len(received) == 1


def test_pending_transport_requires_matching_sealed_terminal_intent(inputs, monkeypatch):
    from swarmrsi.contracts import canonical_sha256
    failing_provider(inputs, monkeypatch)
    assert cli.execute(inputs) == 0
    path = inputs.run_dir/'checkpoint.json'
    checkpoint = json.loads(path.read_text())
    checkpoint['proposer_state'].pop('terminal_intent')
    checkpoint.pop('checkpoint_sha256')
    checkpoint['checkpoint_sha256'] = canonical_sha256(checkpoint)
    path.write_text(json.dumps(checkpoint))
    with pytest.raises(ValueError, match='terminal stop intent'):
        cli.execute(inputs)
