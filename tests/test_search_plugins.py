"""Real coordinator lifecycle with synthetic candidate results, no model calls."""
from dataclasses import replace
import json

import pytest

from evolvcoder.campaign import VSearchCoordinator
from evolvcoder.contracts import canonical_sha256
from evolvcoder import naturebench_session as session
from evolvcoder.search_plugins import SearchRegistry, SearchSpec, default_search_registry
from evolvcoder.strategies import ContinuationTreeStrategy, MultiParentGraphStrategy
from test_evolvcoder_incremental_feedback import campaign, next_step, RecordingProposer, ScriptedExecutor


def custom_factory(*, seed, parameters):
    return ContinuationTreeStrategy(seed=seed, **parameters)


class CheckpointProposer(RecordingProposer):
    def checkpoint_state(self):
        return {'index': self.index}

    def restore_state(self, state):
        self.index = state['index']


@pytest.mark.parametrize('name', ['continuation_tree', 'legacy_tree', 'multi_parent_graph', 'heterogeneous_islands', 'bidirectional_research'])
def test_real_coordinator_plugin_lifecycle_and_resume(tmp_path, name):
    registry = default_search_registry()
    loop = campaign(tmp_path, scores=[.4, .6, -.1])
    loop.proposer = CheckpointProposer()
    loop.strategy = session.strategy('v1', 0, {'search': {'name': name}})
    loop.version = loop.strategy.name
    next_step(loop)
    saved = json.loads(json.dumps(loop.checkpoint()))
    new = registry.create({'name': name})
    restored = VSearchCoordinator.restore(strategy=new, proposer=CheckpointProposer(),
        executor=ScriptedExecutor(tmp_path / 'restored', [.6]), payload=saved)
    assert new is not loop.strategy
    assert restored.strategy.checkpoint_state() == loop.strategy.checkpoint_state()
    assert restored.strategy.proposal_directives(restored._snapshot(), 1) == loop.strategy.proposal_directives(loop._snapshot(), 1)
    assert restored.view.sha256 == loop.view.sha256
    assert restored.accounting == loop.accounting
    next_step(restored)
    assert len(restored.strategy.observations) == 2


def test_tree_behavior_same_through_actual_campaign(tmp_path):
    loop = campaign(tmp_path, scores=[.4, .6, -.1, .9])
    loop.strategy = default_search_registry().create({'name': 'continuation_tree'})
    shadow = ContinuationTreeStrategy()
    for _ in range(4):
        assert shadow.proposal_directives(loop._snapshot(), 1) == loop.strategy.proposal_directives(loop._snapshot(), 1)
        loop.expand()
        snapshot = loop._snapshot()
        expected = shadow.select(snapshot)
        observed = loop.strategy.select(snapshot)
        assert expected.action_id == observed.action_id and expected.score_vector == observed.score_vector
        step = loop.step()
        shadow.observe(loop.strategy.observations[-1])
        assert shadow.burst == loop.strategy.burst
        assert shadow.parent_attempts == loop.strategy.parent_attempts
        assert step.artifact_valid


@pytest.mark.parametrize('change', ['seed', 'name', 'parameters', 'state', 'missing_state'])
def test_incompatible_checkpoint_rejected(tmp_path, change):
    loop = campaign(tmp_path, scores=[.4])
    loop.strategy = default_search_registry().create({'name': 'continuation_tree'})
    next_step(loop)
    saved = loop.strategy.checkpoint_state()
    spec = {'name': 'continuation_tree'}
    seed = 0
    if change == 'seed': seed = 1
    if change == 'name': spec['name'] = 'legacy_tree'
    if change == 'parameters': spec['parameters'] = {'stagnation_patience': 2}
    if change == 'state': saved['state']['burst']['best_metric'] = 999
    if change == 'missing_state': saved.pop('state')
    new = default_search_registry().create(spec, seed=seed)
    with pytest.raises(ValueError): new.restore_state(saved)


def test_campaign_cannot_silently_drop_plugin_state(tmp_path):
    loop = campaign(tmp_path, scores=[.4])
    loop.strategy = default_search_registry().create({'name': 'continuation_tree'})
    next_step(loop)
    saved = loop.checkpoint()
    saved['strategy_state'] = None
    body = {k: v for k, v in saved.items() if k != 'checkpoint_sha256'}
    saved['checkpoint_sha256'] = canonical_sha256(body)
    with pytest.raises(ValueError, match='checkpoint state is required'):
        VSearchCoordinator.restore(strategy=default_search_registry().create({'name': 'continuation_tree'}),
            proposer=RecordingProposer(), executor=ScriptedExecutor(tmp_path / 'restored', []), payload=saved)


def test_explicit_registered_custom_factory_and_lazy_construction():
    registry = SearchRegistry()
    registry.register('custom', custom_factory)
    registry.register('never', lambda **kwargs: (_ for _ in ()).throw(AssertionError('unselected factory ran')))
    assert isinstance(registry.create(SearchSpec('custom'))._algorithm, ContinuationTreeStrategy)
    with pytest.raises(ValueError, match='unique'): registry.register('custom', custom_factory)
    with pytest.raises(ValueError, match='unknown search'): registry.create({'name': 'missing'})


def test_reused_factory_instance_rejected():
    registry = SearchRegistry()
    instance = MultiParentGraphStrategy()
    registry.register('bad', lambda **kwargs: instance)
    first = registry.create({'name': 'bad'})
    with pytest.raises(ValueError, match='reused'): registry.create({'name': 'bad'})
    assert first._algorithm is instance


def test_missing_lifecycle_rejected():
    registry = SearchRegistry()
    registry.register('bad', lambda **kwargs: object())
    with pytest.raises(TypeError, match='lifecycle'): registry.create({'name': 'bad'})


@pytest.mark.parametrize('value', [ {'name': ''}, {'name': 'tree', 'typo': 1},
    {'name': 'tree', 'parameters': {'seed': 1}}, {'name': 'tree', 'parameters': {'x': float('nan')}},
    {'name': 'tree', 'parameters': {1: 2}}, {'name': 'tree', 'parameters': {'x': (1, 2)}} ])
def test_bad_spec_rejected(value):
    with pytest.raises(ValueError): SearchSpec.from_dict(value)


def test_legacy_entry_unchanged_and_no_silent_learning_substitution():
    assert isinstance(session.strategy('v1', 0, {'tree_search': {'policy': 'continuation_tree'}}), ContinuationTreeStrategy)
    with pytest.raises(ValueError, match='RSI policy adapter'):
        session.strategy('v1', 0, {'search': {'name': 'multi_parent_graph'}, 'rsi_policy': {}})
    with pytest.raises(ValueError, match='legacy version'):
        session.strategy('v3', 0, {'search': {'name': 'continuation_tree'}})


def test_invalid_decision_cannot_change_graph(tmp_path, monkeypatch):
    loop = campaign(tmp_path)
    loop.strategy = default_search_registry().create({'name': 'continuation_tree'})
    loop.expand()
    original = loop.strategy._algorithm.select
    monkeypatch.setattr(loop.strategy._algorithm, 'select', lambda snapshot: replace(original(snapshot), graph_sha256='0' * 64))
    with pytest.raises(ValueError, match='current graph'): loop.strategy.select(loop._snapshot())
