"""One interface between the outer learner and whatever algorithm it tunes.

The RSI chain owns evidence, pairing and promotion; it must not know how any
one search algorithm spells its parameters. Each codec supplies exactly that:
how a policy is decoded, what the proposer may return, and how a returned
candidate is checked against the algorithm's own declaration. An algorithm
that declares nothing gets no codec, and no other algorithm's policy is
substituted for it.
"""
from __future__ import annotations

from typing import Any, Protocol

from .search_parameters import SearchParameterPolicy, declaration_of
from .search_policy import SearchPolicy
from .tree_policy import PolicyTreeStrategy, TreePolicy

TREE_INSTRUCTION = (
    "Return parameters (all and only the five incumbent parameter names), rationale, and evidence_run_ids. "
    "Propose one evidence-motivated Tree search-policy change. Parent weights must be finite in [0,4]; "
    "continuation count and patience are positive integers in [1,16], count must fit the unchanged "
    "max_local_attempts in base_tree_config. Do not change the scientific task, evaluator, model, hard budget, "
    "or verdict. Planned methods and failed implementations are not scientifically proven hypotheses. "
    "You may return no_candidate with a reason instead.")


class PolicyCodec(Protocol):
    kind: str

    def from_dict(self, record: dict) -> Any: ...
    def recheck(self, policy: Any) -> None: ...
    def build(self, parameters: dict) -> Any: ...
    def chain_context(self) -> dict: ...
    def instruction(self) -> str: ...


class TreePolicyCodec:
    """The original bounded Tree-parameter learner; unchanged behaviour."""

    kind = "tree_policy"

    def __init__(self, *, base_tree_config: dict) -> None:
        self.base_tree_config = base_tree_config

    def from_dict(self, record):
        return TreePolicy.from_dict(record)

    def recheck(self, policy):
        PolicyTreeStrategy(policy=policy, base_config=self.base_tree_config)

    def build(self, parameters):
        if set(parameters) != set(TreePolicy.__dataclass_fields__):
            raise ValueError("policy interface whitelist")
        policy = TreePolicy(**parameters)
        self.recheck(policy)
        return policy

    def chain_context(self):
        return {"base_tree_config": self.base_tree_config}

    def instruction(self):
        return TREE_INSTRUCTION


class SearchParameterCodec:
    """Learns only the parameters a search algorithm declares for itself."""

    kind = "search_parameters"

    def __init__(self, theta0: SearchParameterPolicy) -> None:
        if not isinstance(theta0, SearchParameterPolicy):
            raise TypeError("search parameter codec requires a SearchParameterPolicy")
        self.theta0 = theta0

    def from_dict(self, record):
        policy = SearchParameterPolicy.from_dict(record)
        self.recheck(policy)
        return policy

    def recheck(self, policy):
        # Pure check: a stored record can never widen its own bounds, because
        # the declaration must equal the one bound when the chain was created.
        if policy.search.to_dict() != self.theta0.search.to_dict():
            raise ValueError("policy belongs to a different search algorithm or configuration")
        if policy.declaration != self.theta0.declaration:
            raise ValueError("policy declaration differs from the algorithm bound to this chain")

    def build(self, parameters):
        if set(parameters) != {item.name for item in self.theta0.declaration}:
            raise ValueError("policy interface whitelist")
        return SearchParameterPolicy(self.theta0.search, self.theta0.declaration, parameters)

    def chain_context(self):
        return {"search_policy_context": {"search": self.theta0.search.to_dict(),
                                          "declaration": [item.to_dict() for item in self.theta0.declaration]}}

    def instruction(self):
        bounds = "; ".join(
            f"{item.name}: {item.kind} in [{item.minimum}, {item.maximum}] (currently {self.theta0.values[item.name]})"
            for item in self.theta0.declaration)
        return (f"Return parameters (all and only these declared names: "
                f"{', '.join(item.name for item in self.theta0.declaration)}), rationale, and evidence_run_ids. "
                f"Propose one evidence-motivated change to the search algorithm's declared learnable parameters. "
                f"Bounds: {bounds}. These are the only values you may change: the scientific task, evaluator, data, "
                "model, reward, metric, failure classification, hard budgets and every other algorithm parameter stay "
                "fixed. Planned methods and failed implementations are not scientifically proven hypotheses. "
                "You may return no_candidate with a reason instead.")


def codec_for(theta0: Any, *, base_tree_config: dict | None = None) -> PolicyCodec:
    """Select the codec an outer learner must use for this initial policy."""
    if isinstance(theta0, TreePolicy):
        return TreePolicyCodec(base_tree_config=base_tree_config or {})
    if isinstance(theta0, SearchParameterPolicy):
        if base_tree_config:
            raise ValueError("a learned search policy does not use a Tree base configuration")
        return SearchParameterCodec(theta0)
    if isinstance(theta0, SearchPolicy):
        raise ValueError(f"search algorithm {theta0.search.name!r} is configured without learnable parameters; "
                         "outer learning needs a policy bound to its declared parameters")
    raise TypeError("unsupported initial policy for outer learning")


def learnable_policy(search_spec, algorithm, values=None) -> SearchParameterPolicy:
    """Bind a configured search algorithm to its own declaration, or refuse."""
    if not declaration_of(algorithm):
        raise ValueError(f"search algorithm {search_spec.name!r} declares no learnable parameters; outer learning "
                         "is not supported for it and no other algorithm's policy is substituted")
    return SearchParameterPolicy.from_algorithm(search_spec, algorithm, values)
