"""Nullable target and explicit existing native protocol regression coverage."""
import json
from dataclasses import replace
import pytest
from swarmrsi.api import Application
from swarmrsi.unified_run import context_for
from swarmrsi import naturebench_session as session
from swarmrsi.retained_results import read_json
from swarmrsi.planner_protocol import proposal_structure_error
from swarmrsi.mlebench_tasks import METRIC
from test_unified_run import setup  # noqa: F401
from test_naturebench_entrypoint import inputs  # noqa: F401


@pytest.mark.parametrize('threshold', [None, -.1, .9])
def test_full_budget_terminal_without_claude_comparison(setup, threshold):
    fixture, _, config, _, executions = setup
    profile = read_json(fixture.profile)
    task = profile['tasks']['fixture']; task.pop('claude_code_score')
    task.update(search_metric=METRIC, search_target=threshold)
    fixture.profile.write_text(json.dumps(profile))
    config = replace(config, limits=replace(config.limits, evaluations=1))
    context = context_for(config)
    result = Application().run(context)
    assert result.status == 'completed'
    final = read_json(result.evidence)
    assert final['target_threshold'] == threshold
    assert 'overall_beats_claude_code_same_backbone' not in final
    assert 'evolved_beats_claude_code_same_backbone' not in final
    assert Application().run(context) == result and len(executions) == 1


@pytest.mark.parametrize('controls', [{'estimated_runtime_seconds': 60}, 42, ['ok', 3]])
def test_invalid_capacity_controls_still_rejected(controls):
    value = {'slot_index': 0, 'hypothesis': 'test', 'description': 'test', 'operator': 'improve_program',
             'family': {'input_transform': 'text', 'primary_operator': 'linear', 'output_construction': 'probability',
                        'objective_or_test': 'auc', 'capacity_controls': controls},
             'predictions': {'quality': .8, 'information_gain': .5, 'reuse': .8, 'validity': .9, 'cost': .2}}
    assert proposal_structure_error(value) == 'proposal_capacity_controls_type'
    value['family']['capacity_controls'] = ['max_iter=1000', 'cv=3']
    assert proposal_structure_error(value) is None
    # A single control written as one string is the same declaration as a
    # one-element list; the parser already coerced it, the validator now agrees.
    value['family']['capacity_controls'] = 'max_iter=1000, cv=3'
    assert proposal_structure_error(value) is None
    # The validator must not write the coerced list back: the provider message is
    # already journalled, and mutating it breaks the learning-evidence binding audit.
    assert value['family']['capacity_controls'] == 'max_iter=1000, cv=3'


def test_native_existing_protocol_enabled_explicitly(monkeypatch):
    monkeypatch.setattr(session, 'BASE_URL', 'https://example.invalid')
    coder, planner = session.build_search_models('synthetic', {'model': 'synthetic',
        'native_coder_tools': True, 'native_planner_tools': True}, {})
    assert coder.native_tools and planner.native_tools
    tool = planner._planner_tool_definition()
    schema = tool['function']['parameters']['properties']['proposals']['items']
    assert schema['properties']['family']['properties']['capacity_controls'] == {'type': 'array', 'items': {'type': 'string'}}
    assert schema['properties']['predictions']['properties']['quality']['type'] == 'number'


def test_native_minimize_target_and_utility_are_aligned():
    from swarmrsi.task_metrics import public_metric, target, terminal_targets, planner_metric_instruction
    task = {'search_metric': {'name': 'mse', 'direction': 'minimize', 'role': 'public_validation'}, 'search_target': .2}
    assert target(task) == -.2
    native, utility = public_metric({'mse': .1}, task)
    assert (native, utility) == (.1, -.1)
    assert terminal_targets(task, utility, utility)['evolved_public_target_reached'] is True
    assert terminal_targets(task, -.3, -.3)['evolved_public_target_reached'] is False
    assert 'public validation mse (minimize)' in planner_metric_instruction(task['search_metric'])
    assert planner_metric_instruction() == 'Scores are higher-is-better aggregate improvement. '
    assert terminal_targets({'claude_code_score': .2}, .3, None) == {
        'target_threshold': .2, 'overall_beats_claude_code_same_backbone': True,
        'evolved_beats_claude_code_same_backbone': False}
