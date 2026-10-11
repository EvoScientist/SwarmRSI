"""No paid transport: capture the original HTTP adapter with controlled I/O."""
from copy import deepcopy
import json
from types import SimpleNamespace
from unittest.mock import Mock, patch

import pytest

from evolvcoder.configuration import ConfigurationError, Limits, ModelConfig
from evolvcoder.model_settings import request_controls, profile_with_model_controls
from evolvcoder import learning_cli, naturebench_cli, naturebench_session as session
from evolvcoder.coder_openrouter import OpenRouterCoderModel
from evolvcoder.session_limits import SessionLimits
from test_session_limits import ACTION, usage
import test_naturebench_entrypoint as entry_tests

inputs = entry_tests.inputs


def captured_response(model, payload):
    raw = {'usage': {'prompt_tokens': 3, 'completion_tokens': 5},
           'choices': [{'message': {'content': '{"proposals": [], "tool": "finish"}'}}]}
    receiver, sender = Mock(), Mock()
    receiver.poll.return_value = True
    receiver.recv.return_value = ('ok', raw)
    process = Mock()
    process.is_alive.return_value = False
    context = Mock()
    context.Pipe.return_value = receiver, sender
    context.Process.return_value = process
    with patch('multiprocessing.get_context', return_value=context):
        model.respond(payload)
    receiver.poll.assert_called_once_with(float(model.request_total_timeout_seconds))
    args = context.Process.call_args.kwargs['args']
    assert args[3] == model.request_timeout_seconds
    return json.loads(args[1])


@pytest.mark.parametrize('role', ['coder', 'planner', 'rsi'])
@pytest.mark.parametrize('shared,role_cap,remaining,expected', [(100, 900, 400, 100), (900, 100, 400, 100), (900, 400, 100, 100)])
def test_real_http_body_uses_three_way_cap_and_explicit_time_temperature(role, shared, role_cap, remaining, expected):
    profile = {'model_settings': {'temperature': .7, 'timeout_seconds': 777,
                                 'socket_timeout_seconds': 51, 'max_output_tokens': shared}}
    controls = request_controls(profile, {}, role=role, role_output_cap=role_cap)
    model = OpenRouterCoderModel(api_key='synthetic-key', model='synthetic', base_url='https://example.invalid', **controls)
    version = {'coder': 'C3', 'planner': 'PLANNER', 'rsi': 'RSI_POLICY'}[role]
    body = captured_response(model, {'version': version, 'max_output_tokens_this_turn': remaining})
    assert body['max_tokens'] == expected and body['temperature'] == .7
    assert model.request_timeout_seconds == 51 and model.request_total_timeout_seconds == 777
    assert 'synthetic-key' not in json.dumps(body)
    # A smaller response reservation must not permanently shrink the ceiling.
    later = captured_response(model, {'version': version, 'max_output_tokens_this_turn': 1000})
    assert later['max_tokens'] == min(shared, role_cap)


@pytest.mark.parametrize('role,socket,total,cap', [('coder', 300, 300, 12000), ('planner', 300, 300, 16000), ('rsi', 180., 300, 8192)])
def test_absent_settings_keep_exact_legacy_defaults(role, socket, total, cap):
    assert request_controls({}, {}, role=role) == {'temperature': 0, 'request_timeout_seconds': socket,
        'request_total_timeout_seconds': total, 'max_tokens_per_turn': cap}


@pytest.mark.parametrize('field', ['temperature', 'timeout_seconds', 'socket_timeout_seconds', 'max_output_tokens'])
@pytest.mark.parametrize('invalid', [None, True, False, -1, float('nan'), float('inf'), '7'])
def test_invalid_settings_are_not_ignored(field, invalid):
    with pytest.raises(ConfigurationError):
        request_controls({'model_settings': {field: invalid}}, {}, role='coder')


@pytest.mark.parametrize('invalid', [None, [], {'request_timeout': 300}, {'api_key': 'must-not-be-accepted'}])
def test_unknown_settings_are_rejected(invalid):
    with pytest.raises(ConfigurationError):
        request_controls({'model_settings': invalid}, {}, role='rsi')


@pytest.mark.parametrize('changes', [{'requests_per_minute': 2}, {'tokens_per_minute': 3}, {'concurrency': 2}])
def test_portable_unsupported_controls_fail_explicitly(changes):
    with pytest.raises(ConfigurationError, match='not implemented'):
        profile_with_model_controls({}, ModelConfig(model='synthetic', **changes))


def test_portable_bridge_is_nonmutating_and_has_no_credential_or_endpoint_io():
    original = {'tasks': {'t': {'metric': 'unchanged'}}, 'provider': {'order': ['unchanged']}}
    before = deepcopy(original)
    model = ModelConfig(model='chosen', timeout_seconds=901, max_output_tokens=99, temperature=.4)
    compiled = profile_with_model_controls(original, model)
    assert original == before and compiled['model'] == 'chosen'
    assert compiled['model_settings'] == {'timeout_seconds': 901, 'socket_timeout_seconds': 901,
                                          'max_output_tokens': 99, 'temperature': .4}
    assert compiled['tasks'] == original['tasks'] and compiled['provider'] == original['provider']
    assert not {'api_key_env', 'api_key_file', 'base_url'} & compiled.keys()


def test_inner_constructors_and_executor_cannot_enlarge_shared_cap(tmp_path):
    profile = {'model': 'synthetic', 'model_settings': {'max_output_tokens': 123, 'timeout_seconds': 900}}
    coder, planner = session.build_search_models('synthetic-key', profile, {})
    assert coder.max_tokens_per_turn == planner.max_tokens_per_turn == 123
    executor = session.Executor(coder, tmp_path, 0, tmp_path, ('main',), ACTION)
    limits = SessionLimits(Limits(output_tokens=1000)).to_legacy()
    executor.set_remaining(usage(output_tokens=930), limits)
    assert coder.max_tokens_per_turn == 70
    executor.set_remaining(usage(output_tokens=100), limits)
    assert coder.max_tokens_per_turn == 123
    assert captured_response(coder, {'version': 'C3'})['max_tokens'] == 123


def test_explicit_larger_role_cap_is_configurable():
    coder, planner = session.build_search_models('synthetic-key', {'model': 'synthetic', 'model_settings': {'max_output_tokens': 30000}},
                                                 {'coder_output_tokens_per_turn': 22000, 'planner_output_tokens_per_batch': 25000})
    assert (coder.max_tokens_per_turn, planner.max_tokens_per_turn) == (22000, 25000)


def test_outer_proposer_consumes_same_controls_and_invalid_precedes_secret(tmp_path, monkeypatch):
    profile = {'model': 'synthetic', 'model_settings': {'max_output_tokens': 71, 'timeout_seconds': 333,
                                                     'socket_timeout_seconds': 55, 'temperature': .2}}
    record = {'requested_config': {'proposal_output_tokens': 999},
              'installed_execution': {'credential_file': str(tmp_path / 'never-read.json'), 'base_url': 'https://example.invalid'}}
    validator = SimpleNamespace(profile=profile, environment={})
    reader = Mock(return_value={'api_key': 'synthetic-key'})
    monkeypatch.setattr(learning_cli, 'read_json', reader)
    proposal = learning_cli.proposer(record, validator)
    body = captured_response(proposal.model, {'version': 'RSI_POLICY'})
    assert body['max_tokens'] == 71 and body['temperature'] == .2
    assert proposal.model.request_total_timeout_seconds == 333 and proposal.model.request_timeout_seconds == 55
    reader.reset_mock()
    profile['model_settings']['timeout_seconds'] = None
    with pytest.raises(ConfigurationError):
        learning_cli.proposer(record, validator)
    reader.assert_not_called()


def test_invalid_profile_rejected_in_actual_installed_preflight(inputs, monkeypatch):
    profile = json.loads(inputs.profile.read_text())
    profile['model_settings'] = {'timeout_seconds': None}
    inputs.profile.write_text(json.dumps(profile))
    secret = Mock(side_effect=AssertionError('credential read forbidden'))
    monkeypatch.setattr(session, 'secret_key', secret)
    with pytest.raises(ConfigurationError):
        naturebench_cli.execute(inputs)
    secret.assert_not_called()
    assert not inputs.run_dir.exists()


def test_full_session_per_step_assignments_preserve_shared_http_cap(inputs, monkeypatch):
    from test_evolvcoder_incremental_feedback import ScriptedExecutor
    from test_session_limits import SessionRecordingProposer
    from test_naturebench_entrypoint import synthetic_root
    profile = json.loads(inputs.profile.read_text())
    profile.update(model_settings={'max_output_tokens': 71, 'temperature': .6, 'timeout_seconds': 901},
                   screening_max_completed_actions=2, screening_execution_limits=ACTION)
    profile['tasks']['fixture']['claude_code_score'] = 10.
    inputs.profile.write_text(json.dumps(profile))
    naturebench_cli.configure(inputs)
    monkeypatch.setattr(session, 'evaluate_root', synthetic_root)
    monkeypatch.setattr(session, 'coder_sandbox_config', lambda _: {'available': True})
    monkeypatch.setattr(session, 'public_task_context', lambda *a, **k: 'synthetic context')
    bodies = []
    class ProbePlanner(SessionRecordingProposer):
        def __init__(self, model, *args, **kwargs):
            super().__init__()
            self.model = model

        def propose(self, **kwargs):
            bodies.append(captured_response(self.model, {'version': 'PLANNER', 'planner_payload': {'synthetic': True}}))
            return super().propose(**kwargs)
    original_executor = session.Executor
    class ProbeExecutor(ScriptedExecutor):
        def __init__(self, model, run_dir, seed, task_root, symbols, screening):
            super().__init__(run_dir, [.4, .2])
            original_executor.__init__(self, model, run_dir, seed, task_root, symbols, screening)

        set_remaining = original_executor.set_remaining

        def execute(self, **kwargs):
            bodies.append(captured_response(self.model, {'version': 'C3'}))
            return super().execute(**kwargs)
    monkeypatch.setattr(session, 'Planner', ProbePlanner)
    monkeypatch.setattr(session, 'Executor', ProbeExecutor)
    assert session.run_session(inputs) == 0
    assert len(bodies) == 4
    assert all(body['max_tokens'] == 71 and body['temperature'] == .6 for body in bodies)
    final = json.loads((inputs.run_dir / 'final.json').read_text())
    assert len(final['steps']) == 2 and final['stop_reason'] == 'screening_action_cap_reached'
    assert json.loads(inputs.profile.read_text()) == profile
