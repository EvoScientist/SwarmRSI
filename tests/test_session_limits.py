"""Portable limits through the real complete loop; scores/models are synthetic."""
from copy import deepcopy
import json
import math
from types import SimpleNamespace

import pytest

from swarmrsi.configuration import ConfigurationError, Limits
from swarmrsi.session_limits import SessionLimits, profile_with_limits, remaining
from swarmrsi.campaign import VSearchCoordinator
from swarmrsi import naturebench_cli as cli, naturebench_session as session
from swarmrsi.strategies import (HypothesisProgramTreeStrategy, ContinuationTreeStrategy,
    MultiParentGraphStrategy, HeterogeneousMultiIslandStrategy, EvidenceCoupledStrategy)
from swarmrsi.tree_policy import PolicyTreeStrategy, TreePolicy
from test_swarmrsi_incremental_feedback import campaign, next_step, RecordingProposer, ScriptedExecutor
import test_naturebench_entrypoint as entry_tests

inputs = entry_tests.inputs
synthetic_root = entry_tests.synthetic_root


ACTION = {'model_calls_per_action': 8, 'tool_calls_per_action': 24,
          'max_input_tokens_per_action_safety': 2000000,
          'max_output_tokens_per_action': 96000, 'wall_time_seconds_per_action': 3600}


class SessionRecordingProposer(RecordingProposer):
    """Synthetic proposals, but preserve the real session clock state contract."""
    def checkpoint_state(self):
        clock = getattr(self, "session_clock", None)
        return {"schema_version": "synthetic-proposer-state",
                "session_clock": clock.snapshot() if clock is not None else None}


def usage(**kwargs):
    return SimpleNamespace(**({'model_calls': 2, 'tool_calls': 3, 'score_evaluations': 1,
                              'input_tokens': 13, 'output_tokens': 17, 'wall_seconds': 19.} | kwargs))


@pytest.mark.parametrize('field', ['model_calls', 'input_tokens', 'output_tokens', 'tool_calls', 'evaluations', 'wall_seconds'])
def test_each_finite_dimension_survives_other_unbounded_caps(field):
    bridge = SessionLimits(Limits(**{field: 100}))
    assert bridge.combined_tokens is None
    residual = bridge.residual(usage())
    assert sum(math.isfinite(v) for v in residual.values()) == 1
    allocation = bridge.coder_allocation(usage(), ACTION)
    assert all(math.isfinite(value) and value > 0 for value in allocation.values())
    json.dumps(bridge.to_legacy(), allow_nan=False)
    assert SessionLimits.from_legacy(bridge.to_legacy()) == bridge


@pytest.mark.parametrize('name', ['max_input_tokens_per_action_safety', 'max_output_tokens_per_action', 'wall_time_seconds_per_action'])
def test_unlimited_campaign_requires_explicit_finite_action_bound(name):
    bridge = SessionLimits(Limits())
    controls = {k: v for k, v in ACTION.items() if k != name}
    with pytest.raises(ConfigurationError, match=name):
        bridge.validate_action_limits(controls)
    for invalid in (None, True, 0, -1, math.inf, math.nan, '100'):
        with pytest.raises(ConfigurationError):
            bridge.validate_action_limits(controls | {name: invalid})


@pytest.mark.parametrize('used', [None, True, -1, math.inf, math.nan, '1'])
def test_unlimited_does_not_mean_unknown_usage(used):
    with pytest.raises(ValueError):
        remaining(None, used)


def test_finite_import_and_disabled_input_legacy_compatibility():
    raw = {'backbone_message_ids': 40, 'tool_calls': 100, 'score_bearing_evaluations': 10,
           'wall_time_seconds': 5000, 'tokens': {'input': 1000, 'context_read': 10000, 'output': 30000}}
    finite = SessionLimits.from_legacy(raw)
    assert finite.combined_tokens == 40000
    assert finite.coder_allocation(usage(), {}) == {'calls': 8, 'tools': 24, 'input': 9987, 'output': 29983, 'wall': 4981.}
    legacy = SessionLimits.from_legacy(raw, enforce_context=False)
    assert legacy.combined_tokens == 10**18
    assert legacy.coder_allocation(usage(), {})['input'] == 2_000_000
    with pytest.raises(ConfigurationError):
        legacy.to_legacy()


def test_portable_override_cannot_ignore_input_cap_or_mutate_source():
    original = {'model': 'synthetic', 'screening_execution_limits': {'enforce_context_read_token_limit': False},
                'tasks': {'task': {'starter': '/starter', 'claude_code_score': .1,
                                   'screening_execution_limits': ACTION, 'limits': {'old': 'unchanged'}}}}
    before = deepcopy(original)
    output = profile_with_limits(original, 'task', Limits(input_tokens=200))
    assert original == before
    assert output['tasks']['task']['screening_execution_limits']['enforce_context_read_token_limit'] is True
    bridge = SessionLimits.from_legacy(output['tasks']['task']['limits'])
    assert bridge.residual(usage(input_tokens=201))['remaining_input'] == -1
    assert bridge.coder_allocation(usage(), ACTION)['input'] == 187
    assert output['model'] == original['model'] and output['tasks']['task']['claude_code_score'] == .1


def test_unsupported_money_caps_fail():
    with pytest.raises(ConfigurationError, match='cost_usd'):
        SessionLimits(Limits(cost_usd=10))


@pytest.mark.parametrize('factory', [HypothesisProgramTreeStrategy, ContinuationTreeStrategy, MultiParentGraphStrategy,
                                    HeterogeneousMultiIslandStrategy, EvidenceCoupledStrategy,
                                    lambda: PolicyTreeStrategy(policy=TreePolicy())])
def test_all_existing_policies_execute_with_unbounded_campaign_and_restore(tmp_path, factory):
    loop = campaign(tmp_path, scores=[.4, .2])
    loop.strategy = factory()
    loop.max_model_calls = loop.max_score_evaluations = loop.max_tokens = None
    loop.version = loop.strategy.name
    next_step(loop)
    saved = json.loads(json.dumps(loop.checkpoint(), allow_nan=False))
    assert saved['limits'] == {'max_model_calls': None, 'max_score_evaluations': None,
                               'max_tokens': None, 'proposal_batch_size': 1}
    restored = VSearchCoordinator.restore(strategy=factory(), proposer=RecordingProposer(),
                                         executor=ScriptedExecutor(tmp_path / 'restored', [.2]), payload=saved)
    assert json.loads(json.dumps(restored.checkpoint(), allow_nan=False)) == saved
    assert math.isinf(restored._remaining().remaining_model_calls)
    assert loop.strategy.proposal_directives(loop._snapshot(), 1) == restored.strategy.proposal_directives(restored._snapshot(), 1)
    json.dumps(loop.summary(), allow_nan=False)
    json.dumps(loop.proposer.requests, default=str, allow_nan=False)


@pytest.mark.parametrize('field', ['max_model_calls', 'max_score_evaluations', 'max_tokens'])
@pytest.mark.parametrize('cap', [True, 0, -1, math.inf, math.nan, '10'])
def test_coordinator_rejects_invalid_configured_caps(tmp_path, field, cap):
    loop = campaign(tmp_path, scores=[])
    limits = {'max_model_calls': None, 'max_score_evaluations': None, 'max_tokens': None, field: cap}
    with pytest.raises(ValueError):
        VSearchCoordinator(version=loop.version, strategy=loop.strategy, proposer=loop.proposer,
                           executor=loop.executor, root=loop.nodes['root'], **limits)


@pytest.mark.parametrize('finite_field', [None, 'input_tokens', 'output_tokens'])
def test_full_session_consumes_portable_limits_and_serializes_cleanly(inputs, monkeypatch, finite_field):
    raw = json.loads(inputs.profile.read_text())
    raw['tasks']['fixture']['claude_code_score'] = 10.0
    raw['screening_max_completed_actions'] = 2  # finite synthetic test, not a production default
    raw['screening_execution_limits'] = ACTION
    config = Limits(**({finite_field: 100000} if finite_field else {}))
    inputs.profile.write_text(json.dumps(profile_with_limits(raw, 'fixture', config), allow_nan=False))
    cli.configure(inputs)
    monkeypatch.setattr(session, 'evaluate_root', synthetic_root)
    monkeypatch.setattr(session, 'coder_sandbox_config', lambda _: {'available': True})
    monkeypatch.setattr(session, 'public_task_context', lambda *a, **k: 'synthetic context')
    monkeypatch.setattr(session, 'build_search_models', lambda *a: (SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    proposer = SessionRecordingProposer()
    monkeypatch.setattr(session, 'Planner', lambda *a, **k: proposer)
    allocations = []
    original_executor = session.Executor

    class ReplayExecutor(ScriptedExecutor):
        def __init__(self, model, run_dir, seed, task_root, required_symbols, screening_limits):
            super().__init__(run_dir, [.4, .2])
            original_executor.__init__(self, model, run_dir, seed, task_root, required_symbols, screening_limits)

        def set_remaining(self, accounting, limits):
            original_executor.set_remaining(self, accounting, limits)
            allocation = {name: getattr(self, 'remaining_' + name) for name in ('calls', 'input', 'output', 'tools', 'wall')}
            allocations.append(allocation)

    monkeypatch.setattr(session, 'Executor', ReplayExecutor)
    assert session.run_session(inputs) == 0
    assert len(allocations) == 2
    assert all(math.isfinite(v) and v > 0 for a in allocations for v in a.values())
    for filename in ('manifest.json', 'checkpoint.json', 'final.json', 'live_summary.json'):
        value = json.loads((inputs.run_dir / filename).read_text())
        json.dumps(value, allow_nan=False)
    final = json.loads((inputs.run_dir / 'final.json').read_text())
    assert final['stop_reason'] == 'screening_action_cap_reached'
    assert len(final['steps']) == 2
    assert final['best_evolved_native_metric'] == .4
    assert final['accounting']['model_calls'] == 4
    json.dumps(proposer.requests, default=str, allow_nan=False)


def test_missing_per_action_controls_fail_in_real_entry_before_provider(inputs, monkeypatch):
    raw = json.loads(inputs.profile.read_text())
    raw['tasks']['fixture']['limits'] = SessionLimits(Limits()).to_legacy()
    inputs.profile.write_text(json.dumps(raw))
    monkeypatch.setattr(session, 'run_session', lambda _: pytest.fail('must stop before execution'))
    with pytest.raises(ConfigurationError, match='explicit finite'):
        cli.execute(inputs)
    assert not inputs.run_dir.exists()


@pytest.mark.parametrize('limits', [Limits(), Limits(input_tokens=100000), Limits(output_tokens=100000)])
def test_actual_coder_config_and_model_payload_never_receive_infinity(inputs, monkeypatch, limits):
    from swarmrsi.coder_engine import ContractBoundedCoder
    cli.configure(inputs)
    loop = campaign(inputs.profile.parent / 'payload-case', scores=[])
    loop.expand()
    proposal = next(iter(loop.actions.values()))
    parent = loop.nodes['root']
    monkeypatch.setattr(session, 'coder_sandbox_config', lambda _: {
        'available': True, 'wrapper': ('/synthetic/bwrap',), 'rewrites': (), 'environment': ()})
    executor = session.Executor(SimpleNamespace(model='synthetic'), inputs.run_dir, 0,
                                inputs.tasks_root / inputs.task_id, ('main',), ACTION)
    executor.set_remaining(usage(), SessionLimits(limits).to_legacy())
    cfg = executor._config(proposal, parent, 0)
    json.dumps(cfg.canonical(), allow_nan=False)
    coder = ContractBoundedCoder(cfg, SimpleNamespace())
    payload = coder._payload(parent.artifact, {}, 0, 0, 0, 0, [])
    json.dumps(payload, allow_nan=False)
    assert all(math.isfinite(v) and v > 0 for v in payload['budget_remaining'].values())
    assert math.isfinite(cfg.max_wall_seconds)


@pytest.mark.parametrize('dimension', ['input_tokens', 'output_tokens', 'model_calls', 'tool_calls', 'evaluations', 'wall_seconds'])
def test_only_finite_dimension_stops_complete_session(inputs, monkeypatch, dimension):
    raw = json.loads(inputs.profile.read_text())
    raw['tasks']['fixture']['claude_code_score'] = 10.
    raw['screening_execution_limits'] = ACTION
    limits = Limits(**{dimension: 1})
    inputs.profile.write_text(json.dumps(profile_with_limits(raw, 'fixture', limits)))
    cli.configure(inputs)
    monkeypatch.setattr(session, 'evaluate_root', synthetic_root)
    monkeypatch.setattr(session, 'coder_sandbox_config', lambda _: {'available': True})
    monkeypatch.setattr(session, 'public_task_context', lambda *a, **k: 'synthetic context')
    monkeypatch.setattr(session, 'build_search_models', lambda *a: (SimpleNamespace(native_tools=True), SimpleNamespace(native_tools=True)))
    class NoCalls(SessionRecordingProposer):
        def propose(self, **kwargs):
            pytest.fail('insufficient finite dimension must stop the whole session')
    monkeypatch.setattr(session, 'Planner', lambda *a, **k: NoCalls())
    assert session.run_session(inputs) == 0
    final = json.loads((inputs.run_dir / 'final.json').read_text())
    json.dumps(final, allow_nan=False)
    assert not final['steps'] and final['accounting']['model_calls'] == 0
    assert final['stop_reason'] in {'matched_resource_envelope_exhausted', 'insufficient_remaining_planner_and_action_envelope'}


def _budget(**caps):
    from swarmrsi.configuration import Limits
    from swarmrsi.session_limits import SessionLimits
    base = dict(model_calls=None, input_tokens=None, output_tokens=None,
                tool_calls=None, evaluations=None, wall_seconds=14400.0)
    return SessionLimits(Limits(**{**base, **caps}))


class _Acct:
    model_calls = tool_calls = input_tokens = output_tokens = score_evaluations = 0
    wall_seconds = 0.0
    attempted_actions = ()


SETTINGS = {'model_calls_per_action': 16, 'tool_calls_per_action': 32,
            'wall_time_seconds_per_action': 3600.0,
            'max_output_tokens_per_action': 192000,
            'max_input_tokens_per_action_safety': 2_000_000}


def test_the_configured_action_wall_is_not_raised_by_what_a_run_measures():
    """Growing the allocation as actions get more expensive compounds the wrong thing.

    An adaptive version of this shipped on the premise that an action cut short
    would have succeeded with more room. This benchmark's cells say the opposite:
    of 138 completed, those spending 12 or fewer calls per action land 90% of them
    against 67% for 13 to 18, and calls per action correlates -0.23 with the result.
    """
    acct = _Acct()
    acct.wall_seconds = 1000.0
    settings = {**SETTINGS, 'wall_time_seconds_per_action': 1800.0}
    assert _budget().coder_allocation(acct, settings)['wall'] == 1800.0


def test_an_action_is_never_allocated_more_wall_than_remains():
    acct = _Acct()
    acct.wall_seconds = 14000.0
    assert _budget().coder_allocation(acct, SETTINGS)['wall'] <= 400.0
