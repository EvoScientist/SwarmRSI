"""Search lifecycle plugins; trusted controller code, never an evaluator.

New plugins implement the same campaign lifecycle. Loading a plugin is explicit
and is not a sandbox for malicious Python. Legacy callers retain their original
algorithm objects and checkpoint formats until they opt into this interface.
"""
from __future__ import annotations

from dataclasses import dataclass, field, replace
import hashlib
from importlib import metadata
import inspect
import json
from pathlib import Path
from typing import Any, Callable
import weakref
from threading import RLock

from .contracts import SearchDecision, SearchObservation, canonical_sha256
from .strategies import ProposalDirective

_INSTANCE_LOCK = RLock()
_LIVE_INSTANCES = []


def _json(value, *, tuples=False):
    """Detach JSON configuration and reject lossy/nonfinite encodings."""
    def check(item):
        if isinstance(item, dict):
            if any(not isinstance(k, str) for k in item):
                raise ValueError('search configuration keys must be strings')
            for v in item.values():
                check(v)
        elif isinstance(item, list) or tuples and isinstance(item, tuple):
            for v in item:
                check(v)
        elif item is not None and type(item) not in (str, bool, int, float):
            raise ValueError('search configuration must contain JSON values')
    check(value)
    return json.loads(json.dumps(value, allow_nan=False, sort_keys=True))


@dataclass(frozen=True)
class SearchSpec:
    name: str
    parameters: dict = field(default_factory=dict)
    distribution: str | None = None

    def __post_init__(self):
        if not isinstance(self.name, str) or not self.name or self.name.strip() != self.name:
            raise ValueError('search name must be nonempty without surrounding whitespace')
        if not isinstance(self.parameters, dict) or 'seed' in self.parameters:
            raise ValueError('search parameters must be an object; seed belongs to the run')
        _json(self.parameters)
        if self.distribution is not None and (not isinstance(self.distribution, str)
                                             or not self.distribution.strip()):
            raise ValueError('search distribution must be an installed distribution name')

    def to_dict(self):
        return _json({'name': self.name, 'parameters': self.parameters, 'distribution': self.distribution})

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or 'name' not in value or set(value) - {'name', 'parameters', 'distribution'}:
            raise ValueError('unknown or missing search configuration fields')
        return cls(**_json(value))


def _source_files(factory, algorithm):
    # Bind the complete Python source package of both factory and algorithm.
    # Runtime environment dependencies are separately bound by the runner.
    paths = set()
    for obj in (factory, type(algorithm)):
        filename = inspect.getsourcefile(obj)
        if filename is None:
            raise ValueError('search plugin needs inspectable Python implementation sources')
        path = Path(filename).resolve()
        root = path.parent
        if (root / '__init__.py').is_file():
            while (root.parent / '__init__.py').is_file():
                root = root.parent
            paths.update(root.rglob('*.py'))
        else:
            paths.add(path)
    return {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(paths)}


def _lifecycle(algorithm):
    for method in ('select', 'observe', 'proposal_directives', 'checkpoint_state', 'restore_state'):
        if not callable(getattr(algorithm, method, None)):
            raise TypeError(f'search plugin lacks required lifecycle method {method}')
    if not isinstance(getattr(algorithm, 'name', None), str) or not algorithm.name:
        raise TypeError('search plugin needs an algorithm name')


class BoundSearch:
    """Delegate real decisions while binding implementation and resumable state."""
    requires_checkpoint = True
    def __init__(self, algorithm, factory, spec, seed, distribution_version):
        _lifecycle(algorithm)
        self._algorithm = algorithm
        self._factory = factory
        self._spec = spec.to_dict()
        self._seed = seed
        self._algorithm_config = _json(getattr(algorithm, 'config', {}), tuples=True)
        self._sources = _source_files(factory, algorithm)
        self._identity = {'schema_version': 'evolvcoder-search-plugin-v1',
                          'spec': self._spec, 'seed': seed,
                          'factory': factory.__module__ + ':' + factory.__qualname__,
                          'algorithm': type(algorithm).__module__ + ':' + type(algorithm).__qualname__,
                          'distribution_version': distribution_version, 'source_hashes': self._sources}
        self.name = algorithm.name

    def __getattr__(self, name):
        return getattr(self._algorithm, name)

    @property
    def execution_identity(self):
        return _json(self._identity)

    @property
    def config(self):
        if _json(getattr(self._algorithm, 'config', {}), tuples=True) != self._algorithm_config:
            raise ValueError('search configuration changed within a session; adaptive state belongs in the checkpoint')
        return {'search_plugin': _json(self._identity),
                'algorithm_config': _json(self._algorithm_config)}

    def learnable_parameters(self):
        """Algorithm-declared learnable construction parameters; () means none."""
        from .search_parameters import declaration_of
        return declaration_of(self._algorithm)

    def source_hashes(self):
        current = _source_files(self._factory, self._algorithm)
        if current != self._sources:
            raise ValueError('search plugin implementation changed')
        return dict(current)

    def proposal_directives(self, snapshot, count):
        directives = tuple(self._algorithm.proposal_directives(snapshot, count))
        if len(directives) > count or any(not isinstance(d, ProposalDirective) for d in directives):
            raise ValueError('search plugin returned invalid proposal directives')
        return directives

    def select(self, snapshot):
        decision = self._algorithm.select(snapshot)
        if not isinstance(decision, SearchDecision) or decision.graph_sha256 != snapshot.graph_sha256:
            raise ValueError('search plugin decision is not bound to the current graph')
        action = next((a for a in snapshot.actions if a.id == decision.action_id), None)
        if action is None:
            raise ValueError('search plugin selected an unknown action')
        if action.action_sha256 in snapshot.attempted_action_sha256s:
            raise ValueError('search plugin selected an already attempted action')
        return replace(decision, policy_sha256=canonical_sha256(self.config),
                       rationale_codes=decision.rationale_codes + ('search_plugin_bound',))

    def observe(self, observation):
        if not isinstance(observation, SearchObservation):
            raise TypeError('search feedback must be a SearchObservation')
        self._algorithm.observe(observation)

    def checkpoint_state(self):
        self.source_hashes()
        state = _json(self._algorithm.checkpoint_state(), tuples=True)
        if not isinstance(state, dict):
            raise TypeError('search checkpoint must be an object')
        body = {'schema_version': 'evolvcoder-search-checkpoint-v1',
                'identity': _json(self._identity), 'state': state}
        return {**body, 'checkpoint_sha256': canonical_sha256(body)}

    def restore_state(self, value):
        self.source_hashes()
        if not isinstance(value, dict) or set(value) != {'schema_version', 'identity', 'state', 'checkpoint_sha256'}:
            raise ValueError('missing or unknown search checkpoint fields')
        body = {k: v for k, v in value.items() if k != 'checkpoint_sha256'}
        if (body['schema_version'] != 'evolvcoder-search-checkpoint-v1'
                or body['identity'] != self._identity or not isinstance(body['state'], dict)
                or value['checkpoint_sha256'] != canonical_sha256(body)):
            raise ValueError('search checkpoint implementation, configuration, seed or state mismatch')
        self._algorithm.restore_state(_json(body['state']))
        if _json(self._algorithm.checkpoint_state(), tuples=True) != body['state']:
            raise ValueError('search plugin did not restore the complete checkpoint state')


Factory = Callable[..., Any]


class SearchRegistry:
    def __init__(self):
        self._factories: dict[str, tuple[Factory, str | None]] = {}

    def register(self, name, factory, *, distribution_version=None):
        SearchSpec(name)
        if name in self._factories or not callable(factory):
            raise ValueError('search name must be unique and factory callable')
        self._factories[name] = (factory, distribution_version)

    def names(self):
        return tuple(sorted(self._factories))

    def create(self, spec, *, seed=0, session_parameters=None):
        """Build a bound search.

        session_parameters are construction options owned by the session profile
        (e.g. screening_execution_limits.initial_root_drafts), not by the search
        spec. They reach the factory but not the recorded spec, so the executed
        spec still equals the policy's spec when the run evidence is verified;
        the algorithm config, which is bound in the checkpoint, records them.
        """
        if not isinstance(spec, SearchSpec):
            spec = SearchSpec.from_dict(spec)
        if type(seed) is not int or seed < 0:
            raise ValueError('search seed must be a nonnegative integer')
        # External plugins are opt-in and tied to an installed distribution.
        if spec.distribution is not None:
            if spec.name in self._factories:
                raise ValueError('external search cannot shadow a registered search')
            matches = [ep for ep in metadata.entry_points(group='evolvcoder.search')
                       if ep.name == spec.name and ep.dist.name == spec.distribution]
            if len(matches) != 1:
                raise ValueError('search entry point must resolve to exactly one declared distribution')
            factory, version = matches[0].load(), matches[0].dist.version
        else:
            try:
                factory, version = self._factories[spec.name]
            except KeyError as exc:
                raise ValueError(f'unknown search {spec.name!r}; available: {self.names()}') from exc
        parameters = _json(spec.parameters)
        extra = _json(dict(session_parameters or {}))
        if set(extra) & set(parameters):
            raise ValueError('session parameters cannot override the search spec')
        construction = {**parameters, **extra}
        algorithm = factory(seed=seed, parameters=construction)
        if parameters != spec.parameters or construction != {**parameters, **extra}:
            raise ValueError('search factory mutated its configuration')
        with _INSTANCE_LOCK:
            _LIVE_INSTANCES[:] = [ref for ref in _LIVE_INSTANCES if ref() is not None]
            if any(ref() is algorithm for ref in _LIVE_INSTANCES):
                raise ValueError('search factory reused an existing session instance')
            bound = BoundSearch(algorithm, factory, spec, seed, version)
            _LIVE_INSTANCES.append(weakref.ref(algorithm))
        return bound


def _continuation(*, seed, parameters):
    from .strategies import ContinuationTreeStrategy
    return ContinuationTreeStrategy(seed=seed, **parameters)


def _legacy(*, seed, parameters):
    from .strategies import HypothesisProgramTreeStrategy
    return HypothesisProgramTreeStrategy(seed=seed, **parameters)


def _graph(*, seed, parameters):
    from .strategies import MultiParentGraphStrategy
    return MultiParentGraphStrategy(seed=seed, **parameters)


def _islands(*, seed, parameters):
    from .strategies import HeterogeneousMultiIslandStrategy
    if 'roles' in parameters:
        parameters = {**parameters, 'roles': tuple(tuple(r) for r in parameters['roles'])}
    return HeterogeneousMultiIslandStrategy(seed=seed, **parameters)


def _bidirectional(*, seed, parameters):
    from .brs_strategy import BidirectionalResearchStrategy
    return BidirectionalResearchStrategy(seed=seed, **parameters)


def default_search_registry():
    registry = SearchRegistry()
    for name, factory in [('continuation_tree', _continuation), ('legacy_tree', _legacy),
                          ('multi_parent_graph', _graph), ('heterogeneous_islands', _islands),
                          ('bidirectional_research', _bidirectional)]:
        registry.register(name, factory)
    return registry
