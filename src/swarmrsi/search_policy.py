"""Serializable run selection, separate from Tree-only learned RSI policies."""
from copy import deepcopy
from dataclasses import dataclass

from .contracts import canonical_sha256
from .search_plugins import SearchSpec
from .tree_policy import TreePolicy

SEARCH_PARAMETER_SCHEMA = "swarmrsi-search-parameter-policy-v1"


@dataclass(frozen=True)
class SearchPolicy:
    search: SearchSpec

    def __post_init__(self):
        if not isinstance(self.search, SearchSpec):
            raise ValueError("search policy requires SearchSpec")

    def to_dict(self):
        body = {"schema_version": "swarmrsi-search-policy-v1", "search": self.search.to_dict()}
        return {**body, "policy_sha256": canonical_sha256(body)}

    @classmethod
    def from_dict(cls, value):
        if not isinstance(value, dict) or set(value) != {"schema_version", "search", "policy_sha256"}:
            raise ValueError("invalid search policy descriptor")
        result = cls(SearchSpec.from_dict(value["search"]))
        if result.to_dict() != value:
            raise ValueError("search policy descriptor or hash mismatch")
        return result


def policy_from_record(value):
    if isinstance(value, dict) and value.get("schema_version") == "swarmrsi-search-policy-v1":
        return SearchPolicy.from_dict(value)
    if isinstance(value, dict) and value.get("schema_version") == SEARCH_PARAMETER_SCHEMA:
        from .search_parameters import SearchParameterPolicy
        return SearchParameterPolicy.from_dict(value)
    return TreePolicy.from_dict(value)


def search_spec_of(policy):
    """The SearchSpec a policy actually executes, or None for a Tree policy."""
    if isinstance(policy, SearchPolicy):
        return policy.search
    if type(policy).__name__ == "SearchParameterPolicy":
        return policy.effective_search()
    return None


def validate_execution_identity(identity, policy, seed):
    """Check an original descriptor without importing or invoking its plugin."""
    keys = {'schema_version', 'spec', 'seed', 'factory', 'algorithm',
            'distribution_version', 'source_hashes'}
    executed = search_spec_of(policy)
    if (not isinstance(identity, dict) or set(identity) != keys
            or identity['schema_version'] != 'swarmrsi-search-plugin-v1'
            or executed is None or identity['spec'] != executed.to_dict()
            or type(identity['seed']) is not int or identity['seed'] != seed):
        raise ValueError('incomplete or mismatched search execution identity')
    if any(not isinstance(identity[k], str) or ':' not in identity[k] for k in ('factory', 'algorithm')):
        raise ValueError('search execution identity lacks implementation symbols')
    sources = identity['source_hashes']
    if not isinstance(sources, dict) or not sources:
        raise ValueError('search execution identity lacks source hashes')
    if any(not isinstance(p, str) or not isinstance(digest, str) or len(digest) != 64
           or any(c not in '0123456789abcdef' for c in digest) for p, digest in sources.items()):
        raise ValueError('invalid search source identity')
    return sources


def matched_profile(profile, policy):
    """The part of a compiled profile that must be identical across paired trials.

    Whatever the policy itself varies is bound separately as that trial's
    policy; keeping it here would make every pair look like a condition change.
    """
    if search_spec_of(policy) is None:
        varying = {"rsi_policy"}
    elif isinstance(policy, SearchPolicy):
        varying = set()
    else:
        varying = {"search", "search_policy"}
    return {key: value for key, value in profile.items() if key not in varying}


def compiled_trial_profile(profile, policy):
    """Pure preparation, also used for read-only historical journal checks."""
    result = deepcopy(profile)
    executed = search_spec_of(policy)
    if executed is not None:
        if profile.get("rsi_policy") is not None:
            raise ValueError("search plugin cannot silently replace an RSI policy")
        spec = executed.to_dict()
        base = policy.search.to_dict()
        if profile.get("search") is not None and SearchSpec.from_dict(profile["search"]).to_dict() not in (spec, base):
            raise ValueError("search policy differs from runner profile")
        result["search"] = spec
        if not isinstance(policy, SearchPolicy):
            # The inner session only receives a profile; record which learned
            # policy produced this configuration so its evidence can bind it.
            result["search_policy"] = policy.to_dict()
    else:
        if profile.get("search") is not None:
            raise ValueError("Tree policy cannot silently replace a configured search")
        result["rsi_policy"] = policy.to_dict()
    return result
