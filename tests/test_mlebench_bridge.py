"""Synthetic unit/negative controls, not benchmark performance evidence."""
import json
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
import pytest

from evolvcoder.task_metrics import target, public_metric
from evolvcoder.mlebench_tasks import validate_submission, METRIC
from evolvcoder import mlebench_run as bridge
from evolvcoder.configuration import Limits
from evolvcoder.unified_run import compiled_profile


@pytest.mark.parametrize('value', [None, .8, 1.])
def test_public_target_is_optional_and_not_a_claude_reference(value):
    assert target({'search_metric': METRIC, 'search_target': value}) == value


@pytest.mark.parametrize('value', [float('nan'), float('inf'), True, '0.8'])
def test_invalid_target_rejected(value):
    with pytest.raises(ValueError): target({'search_metric': METRIC, 'search_target': value})


@pytest.mark.parametrize('direction,utility', [('maximize', .8), ('minimize', -.8)])
def test_native_and_utility_separate(direction, utility):
    assert public_metric({'auc': .8}, {'search_metric': {**METRIC, 'direction': direction}}) == (.8, utility)


def frames():
    pd = pytest.importorskip('pandas')
    return pd.DataFrame({'Comment': ['a', 'a', 'b'], 'Insult': [0, 1, 1]}), pd.DataFrame({'Comment': ['a', 'a', 'b'], 'Insult': [.2, .2, .8]})


def test_valid_duplicate_rows_retained():
    answers, submission = frames()
    validate_submission(submission, answers)
    assert len(submission) == 3


@pytest.mark.parametrize('fault', ['missing', 'extra', 'nan', 'inf', 'column'])
def test_invalid_submission_rejected(fault):
    answers, submission = frames()
    if fault == 'missing': submission = submission.iloc[:-1]
    elif fault == 'extra': submission.loc[3] = ['b', .1]
    elif fault == 'column': submission = submission.drop(columns='Insult')
    else: submission.loc[0, 'Insult'] = {'nan': float('nan'), 'inf': float('inf')}[fault]
    with pytest.raises(ValueError): validate_submission(submission, answers)


@pytest.mark.parametrize('fault', ['multiplicity', 'wrong_id', 'negative', 'above_one'])
def test_identity_and_range_are_left_to_the_official_grader(fault):
    """These pass this gate on purpose; the competition's own grader is what rejects them.

    Checking them here meant guessing which column is the identifier, which differs per
    competition and got insults wrong twice, and asserting a [0, 1] range that is correct
    for an AUC competition and wrong for every regression one on the Lite panel -- taxi
    fares and formation energies are neither probabilities nor bounded.
    """
    answers, submission = frames()
    if fault == 'multiplicity': submission.loc[1, 'Comment'] = 'b'
    elif fault == 'wrong_id': submission.loc[1, 'Comment'] = 'c'
    else: submission.loc[0, 'Insult'] = {'negative': -.1, 'above_one': 1.1}[fault]
    validate_submission(submission, answers)


def test_terminal_budget_reserved_inside_total(tmp_path):
    profile = tmp_path / 'profile.json'
    profile.write_text(json.dumps({'model': 'synthetic', 'tasks': {'x': {'starter': str(tmp_path / 'starter'), 'search_metric': METRIC}}}))
    from evolvcoder.configuration import ModelConfig, LearningConfig
    config = SimpleNamespace(task='x', benchmark='mlebench', model=ModelConfig('synthetic'),
        limits=Limits(model_calls=40, input_tokens=1000000, output_tokens=100000, tool_calls=80, evaluations=4, wall_seconds=3600),
        search=None, learning=LearningConfig())
    assets = {'profile': profile, 'terminal_timeout_seconds': 600, 'trial_timeout_seconds': 3000}
    result = compiled_profile(config, assets)
    assert result['tasks']['x']['limits']['score_bearing_evaluations'] == 3
    assert result['tasks']['x']['limits']['wall_time_seconds'] == 3000
    assert result['session_runtime_controls'] == {
        'declared_inner_wall_seconds': 3000, 'trial_watchdog_seconds': 3000,
        'session_finalize_reserve_seconds': 30.0, 'session_execution_wall_seconds': 2970.0}
    assert assets['terminal_timeout_seconds'] == 600  # Separate purpose, not deducted twice.
    assert 'claude_code_score' not in result['tasks']['x']
    config.limits = replace(config.limits, evaluations=1)
    with pytest.raises(ValueError): compiled_profile(config, assets)


def test_unknown_manifest_fields_rejected(tmp_path):
    path = tmp_path / 'manifest.json'; path.write_text('{}')
    with pytest.raises(ValueError): bridge.manifest(SimpleNamespace(benchmark_config=path))


def test_generic_cli_target_accepted_without_legacy_threshold(inputs):
    from evolvcoder import naturebench_cli as cli
    profile = json.loads(inputs.profile.read_text())
    entry = profile['tasks']['fixture']; entry.pop('claude_code_score'); entry['search_metric'] = METRIC
    inputs.profile.write_text(json.dumps(profile))
    result, _ = cli.configure(inputs)
    assert 'claude_code_score' not in result['tasks']['fixture']


from test_naturebench_entrypoint import inputs  # noqa: F401,E402


def test_text_submission_without_a_numeric_column_is_accepted():
    """text-normalization submits id,after where the prediction is a string."""
    pd = pytest.importorskip('pandas')
    answers = pd.DataFrame({'id': ['0_0', '0_1'], 'after': ['two', 'Rocky']})
    submission = pd.DataFrame({'id': ['0_0', '0_1'], 'after': ['two', 'Rocky']})
    validate_submission(submission, answers)
    with pytest.raises(ValueError):
        validate_submission(submission[['id']], answers)
