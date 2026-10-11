"""Public runner dispatch and result replay; no paid/provider measurements."""
from dataclasses import replace
import json

import pytest

from evolvcoder.api import Application
from evolvcoder.configuration import config_from_dict, ConfigurationError
from evolvcoder import unified_run
from evolvcoder.retained_results import read_json
from evolvcoder.search_plugins import SearchSpec
from evolvcoder.search_policy import SearchPolicy, policy_from_record
from test_unified_run import setup  # noqa: F401
from test_naturebench_entrypoint import inputs  # noqa: F401


@pytest.mark.parametrize('name', ['continuation_tree', 'legacy_tree', 'multi_parent_graph', 'heterogeneous_islands'])
def test_public_selected_search_actual_runner_terminal_and_replay(setup, name):
    fixture, raw, _, models, executions = setup
    raw['search'] = {'name': name}
    config = config_from_dict(raw, base=fixture.profile.parent)
    assert config_from_dict(config.to_dict(), base=fixture.profile.parent) == config
    ctx = unified_run.context_for(config)
    app = Application()
    assert app.preflight(ctx)['ready'] and not executions and not models
    result = app.run(ctx)
    assert result.status == 'completed'
    trial = next((ctx.output_dir / 'validation/trials').iterdir())
    profile = read_json(trial / 'profile.json')
    assert profile['search'] == config.search.to_dict() and 'rsi_policy' not in profile
    state = read_json(trial / 'run/checkpoint.json')['strategy_state']
    assert state['identity']['spec'] == config.search.to_dict()
    assert state['identity']['seed'] == config.seed
    assert read_json(trial / 'evidence.json')['policy'] == SearchPolicy(config.search).to_dict()
    assert app.run(ctx) == result and app.status(ctx)['status'] == 'completed'
    assert len(executions) == len(models) == 1


def test_no_explicit_search_retains_old_serialized_shape(setup):
    _, _, config, _, _ = setup
    assert 'search' not in config.to_dict()


def test_search_for_a_baseline_method_is_refused_by_configuration(setup):
    _, _, config, models, executions = setup
    with pytest.raises(ConfigurationError):
        replace(config, search=SearchSpec('multi_parent_graph'), method='aide')
    assert not executions and not models


@pytest.mark.parametrize('conflict', ['rsi', 'policy'])
def test_outer_learning_over_an_undeclaring_algorithm_is_refused_by_name(setup, conflict):
    """No algorithm without declared learnable parameters gets a substituted policy."""
    fixture, _, config, models, executions = setup
    learning = (replace(config.learning, rsi_enabled=True, feedback_store=fixture.profile.parent / 'pool')
                if conflict == 'rsi' else replace(config.learning, policy=fixture.profile.parent / 'policy.json'))
    bound = replace(config, search=SearchSpec('multi_parent_graph'), learning=learning)
    ctx = unified_run.context_for(bound)
    for operation in (Application().preflight, Application().run):
        with pytest.raises(ValueError, match="declares no learnable parameters"):
            operation(ctx)
    assert not ctx.output_dir.exists() and not executions and not models


def test_declared_parameters_are_the_only_learnable_surface():
    from evolvcoder.learning_policies import codec_for, learnable_policy
    from evolvcoder.search_parameters import ParameterSpec, SearchParameterPolicy
    from evolvcoder.search_plugins import default_search_registry
    spec = SearchSpec('bidirectional_research', {'allocator': {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}})
    algorithm = default_search_registry().create(spec)
    theta0 = learnable_policy(spec, algorithm)
    assert theta0.values == {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}
    codec = codec_for(theta0)
    learned = codec.build({'discount_factor': 0.8, 'frontier_pressure_weight': 0.0})
    assert learned.effective_search().parameters['allocator']['discount_factor'] == 0.8
    assert learned.effective_search().parameters['allocator']['frontier_pressure_weight'] == 0.0
    assert learned.diff(theta0) == {'discount_factor': {'before': 0.9, 'after': 0.8},
                                    'frontier_pressure_weight': {'before': 0.25, 'after': 0.0}}
    assert policy_from_record(learned.to_dict()) == learned and codec.from_dict(learned.to_dict()) == learned
    for illegal in ({'discount_factor': 0.4, 'frontier_pressure_weight': 0.1},
                    {'discount_factor': 0.9},
                    {'discount_factor': 0.9, 'frontier_pressure_weight': 0.1, 'exploration_weight': 0.5},
                    {'discount_factor': 0.9, 'frontier_pressure_weight': True}):
        with pytest.raises((ValueError, TypeError)):
            codec.build(illegal)
    # A stored record cannot widen the bounds its algorithm declared.
    widened = json.loads(json.dumps(learned.to_dict()))
    widened['declaration'][0]['minimum'] = 0.0
    with pytest.raises(ValueError, match='hash mismatch'):
        SearchParameterPolicy.from_dict(widened)
    resealed = SearchParameterPolicy(learned.search,
                                     (ParameterSpec.from_dict(widened['declaration'][0]), learned.declaration[1]),
                                     dict(learned.values))
    with pytest.raises(ValueError, match='declaration differs'):
        codec.from_dict(resealed.to_dict())
    other = SearchSpec('bidirectional_research', {'max_active_hypotheses': 2,
                       'allocator': {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}})
    with pytest.raises(ValueError, match='different search algorithm'):
        codec.recheck(learnable_policy(other, default_search_registry().create(other)))
    # A declared parameter the configuration does not contain is an error, not a default.
    bare = SearchSpec('bidirectional_research', {'max_active_hypotheses': 2})
    with pytest.raises(ValueError, match='absent from the configuration'):
        learnable_policy(bare, default_search_registry().create(bare))


def test_policy_parser_checks_complete_descriptor_without_plugin_load(monkeypatch):
    from evolvcoder import search_plugins
    monkeypatch.setattr(search_plugins.metadata, 'entry_points', lambda **kwargs: pytest.fail('read-only parser loaded plugin'))
    value = SearchPolicy(SearchSpec('external', distribution='not-installed')).to_dict()
    assert policy_from_record(value).to_dict() == value
    value = json.loads(json.dumps(value)); value['search']['name'] = 'different'
    with pytest.raises(ValueError): policy_from_record(value)


@pytest.mark.parametrize('settled', [False, True])
def test_plugin_pending_and_failed_trials_never_relaunch(setup, settled):
    _, _, config, models, executions = setup
    config = replace(config, search=SearchSpec('multi_parent_graph'))
    ctx = unified_run.context_for(config)
    app = Application(); app.preflight(ctx)
    trial = next((ctx.output_dir / 'validation/trials').iterdir())
    (trial / 'launch.json').write_text(json.dumps({'host': 'other-host',
        'runner_sha256': read_json(trial / 'identity.json')['runner_sha256']}))
    if settled:
        (trial / 'exit.json').write_text(json.dumps({'returncode': 2, 'failure': 'synthetic'}))
        (trial / 'execution.log').write_text('synthetic preserved failure')
    result = app.run(ctx)
    assert result.status == ('failed' if settled else 'pending') and result.score is None
    assert app.run(ctx) == result and not executions and not models


def test_plugin_run_analysis_does_not_load_plugin(setup, monkeypatch):
    from evolvcoder import search_plugins
    from evolvcoder.learning_analysis import analyze_learning
    _, _, config, _, _ = setup
    config = replace(config, search=SearchSpec('multi_parent_graph'))
    ctx = unified_run.context_for(config)
    Application().run(ctx)
    monkeypatch.setattr(search_plugins, 'default_search_registry', lambda: pytest.fail('observer constructed a search'))
    report = analyze_learning(runner_roots=[ctx.output_dir / 'validation'])
    assert report['manifest']['chain_state'] is None
    assert len(report['validation_runs']) == 1 and report['policy_trials'] == []
    from evolvcoder.learning_analysis import export_learning_analysis
    exported = export_learning_analysis(ctx.output_dir.parent / 'analysis', runner_roots=[ctx.output_dir / 'validation'])
    assert exported['chain_state'] is None and exported['verified_physical_terminal_runs'] == 1


def test_nonroot_public_graph_search_with_real_planner_journal(setup, monkeypatch):
    from dataclasses import asdict
    from types import SimpleNamespace
    from unittest.mock import patch
    from evolvcoder import naturebench_session as session
    from evolvcoder.campaign import ActionExecution, artifact_workspace_sha256
    from evolvcoder.coder_models import CoderOutcome, CoderExitStatus, CoderVersion
    from evolvcoder.coder_openrouter import OpenRouterCoderModel
    from evolvcoder.contracts import canonical_sha256
    from evolvcoder.retained_results import output_attestation
    from learning_test_helpers import raw
    fixture, _, config, _, executions = setup
    profile = read_json(fixture.profile)
    profile['tasks']['fixture']['claude_code_score'] = .3
    fixture.profile.write_text(json.dumps(profile))
    # This fixture exercises a complete scored action, not an early root stop.
    # Its old 100-second watchdog cannot fit the declared 1260-second action
    # start envelope now that the session actually consumes that watchdog.
    assets = read_json(config.benchmark_config)
    assets['trial_timeout_seconds'] = 6000
    config.benchmark_config.write_text(json.dumps(assets))
    config = replace(config, search=SearchSpec('multi_parent_graph'))
    class PlannerTransport:
        native_tools = True
        def respond(self, request):
            slots = request['planner_payload']['proposal_slots']
            rows = [raw(index=i, family=f'linear-{i}') for i in range(len(slots))]
            response = {'choices': [{'message': {'content': None, 'tool_calls': [{'function': {
                'name': 'submit_proposals', 'arguments': json.dumps({'proposals': rows})}}]}, 'finish_reason': 'stop'}],
                'usage': {'prompt_tokens': 3, 'completion_tokens': 5}}
            model = OpenRouterCoderModel(api_key='synthetic-only', model=config.model.model,
                base_url='https://example.invalid', max_tokens_per_turn=1000, native_tools=True)
            with patch('urllib.request.urlopen') as http:
                http.return_value.__enter__.return_value.read.return_value = json.dumps(response).encode()
                return model.respond(request)
    monkeypatch.setattr(session, 'build_search_models', lambda *a: (SimpleNamespace(native_tools=True), PlannerTransport()))
    monkeypatch.setattr(session.Executor, 'config_sha256', lambda *a, **k: 'a' * 64)
    def execute(executor, *, proposal, parent, step):
        cell = executor.run_dir / 'cells' / f'{step:03d}_{proposal.candidate.id}'
        artifact = cell / 'artifact'; (artifact / 'output').mkdir(parents=True)
        (artifact / 'run.py').write_text('def main():\n    return 1\n')
        (artifact / 'output/pred.csv').write_text('prediction\n0.4\n')
        outcome = CoderOutcome(status=CoderExitStatus.SUCCESS, version=CoderVersion.C3,
            config_sha256='a' * 64, action_id=proposal.candidate.id,
            parent_sha256=parent.executable_sha256, parent_workspace_sha256=parent.workspace_sha256,
            final_sha256=session.sha(artifact / 'run.py'), final_workspace_sha256=artifact_workspace_sha256(artifact),
            model_calls=1, tool_calls=1, input_tokens=3, output_tokens=5, wall_seconds=0,
            changed_paths=['run.py'], implementation_real=True, parent_preserved=True,
            reason='synthetic_provider_boundary', trajectory_sha256=canonical_sha256([]))
        outcome.seal(); session.atomic_json(cell / 'result.json', asdict(outcome))
        (cell / 'trajectory.jsonl').write_text('')
        evidence = {'valid': True, 'execution_valid': True, 'score_bearing_evaluation_attempted': True,
            'metrics': {'aggregate_improvement': .4}, 'details': {'candidate_executable_sha256': outcome.final_sha256,
                'submission_output_attestation': output_attestation(artifact / 'output')}}
        session.atomic_json(cell / 'evaluation.json', evidence)
        return ActionExecution(outcome, artifact, .4, True, 1,
            evaluation_summary=session.compact_evaluation_summary(evidence),
            evaluation_evidence_sha256=session.sha(cell / 'evaluation.json'), evaluation_evidence_path=cell / 'evaluation.json')
    monkeypatch.setattr(session.Executor, 'execute', execute)
    ctx = unified_run.context_for(config)
    result = Application().run(ctx)
    assert result.status == 'completed' and result.details['best_evolved_native_metric'] == .4
    final = read_json(result.evidence)
    assert len(final['steps']) == 1
    assert final['search_strategy_state']['identity']['spec']['name'] == 'multi_parent_graph'
    assert Application().run(ctx) == result and len(executions) == 1


def learned_policy_record(spec, **values):
    from evolvcoder.learning_policies import learnable_policy
    from evolvcoder.search_plugins import default_search_registry
    algorithm = default_search_registry().create(spec)
    return learnable_policy(spec, algorithm, values).to_dict()


def test_imported_learned_search_policy_reaches_the_actual_runner(setup):
    """An imported policy changes the executed algorithm configuration, not just a file."""
    fixture, raw, _, models, executions = setup
    spec = SearchSpec('bidirectional_research', {'allocator': {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}})
    record = learned_policy_record(spec, discount_factor=0.75, frontier_pressure_weight=0.0)
    path = fixture.profile.parent / 'learned_search.json'
    path.write_text(json.dumps(record))
    raw['search'] = spec.to_dict()
    raw['learning'] = {'wisdom_enabled': False, 'rsi_enabled': False, 'policy': str(path)}
    config = config_from_dict(raw, base=fixture.profile.parent)
    ctx = unified_run.context_for(config)
    assert Application().preflight(ctx)['ready']
    result = Application().run(ctx)
    assert result.status == 'completed'
    effective = {'name': 'bidirectional_research', 'distribution': None,
                 'parameters': {'allocator': {'discount_factor': 0.75, 'frontier_pressure_weight': 0.0}}}
    trial = next((ctx.output_dir / 'validation/trials').iterdir())
    assert read_json(trial / 'profile.json')['search'] == effective
    assert read_json(trial / 'run/checkpoint.json')['strategy_state']['identity']['spec'] == effective
    evidence = read_json(trial / 'evidence.json')
    assert evidence['policy'] == record
    # What the policy varies is bound as the trial's policy, not as a matched condition,
    # so a learned candidate and its incumbent can still be compared as a pair.
    fixed = evidence['comparison_context']['fixed_profile']
    assert 'search' not in fixed and 'search_policy' not in fixed and fixed['model'] == config.model.model
    assert len(executions) == len(models) == 1


def test_imported_policy_must_belong_to_the_configured_search(setup):
    fixture, raw, _, _, _ = setup
    other = SearchSpec('bidirectional_research', {'max_active_hypotheses': 2,
                       'allocator': {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}})
    path = fixture.profile.parent / 'foreign.json'
    path.write_text(json.dumps(learned_policy_record(other, discount_factor=0.8, frontier_pressure_weight=0.1)))
    raw['search'] = {'name': 'bidirectional_research',
                     'parameters': {'allocator': {'discount_factor': 0.9, 'frontier_pressure_weight': 0.25}}}
    raw['learning'] = {'wisdom_enabled': False, 'rsi_enabled': False, 'policy': str(path)}
    ctx = unified_run.context_for(config_from_dict(raw, base=fixture.profile.parent))
    with pytest.raises(ValueError, match='different configured search|belongs to a different'):
        Application().preflight(ctx)
    assert not ctx.output_dir.exists()
