from __future__ import annotations

from .contracts import SearchSnapshot
from .strategies import DEFAULT_POLICY, ContinuationTreeStrategy, EvidenceCoupledStrategy, ProposalDirective


def _drafting(snapshot: SearchSnapshot, drafts: int) -> bool:
    return "root" in snapshot.node_artifact_validity and int(snapshot.node_visits.get("root", 0)) < drafts


def _draft_directives(count: int) -> tuple[ProposalDirective, ...]:
    # Drafts are independent: no intervention sources from earlier drafts.
    return tuple(ProposalDirective(
        code_parent_id="root",
        rationale_codes=("initial_root_draft", "independent_draft_distinct_from_earlier_drafts"),
    ) for _ in range(count))


def _check_drafts(initial_root_drafts: int) -> None:
    if type(initial_root_drafts) is not int or initial_root_drafts < 1:
        raise ValueError("initial_root_drafts must be a positive integer")


class RootDraftingEvidenceStrategy(EvidenceCoupledStrategy):
    """v4 with AIDE-style independent drafts before any refinement (opt-in).

    The first K actions all start from root, so the run chooses its lineage
    among K independent programs instead of committing to whichever direction
    the first draft happened to take. Two 00732 runs of the same core diverged
    exactly there: +0.093 after a good first direction, -1.86 after a poor one
    that eleven later actions only refined. After the drafts, parent selection
    is the unchanged v4 ranking.

    Kept outside strategies.py, whose bytes are pinned as the frozen baseline;
    a profile without initial_root_drafts still gets EvidenceCoupledStrategy.
    Counted from node_visits (every executed action from root, failed ones
    included), so drafting cannot loop and needs no state of its own across a
    resume.
    """

    def __init__(self, *, initial_root_drafts: int, policy=DEFAULT_POLICY) -> None:
        _check_drafts(initial_root_drafts)
        super().__init__(policy=policy)
        self.initial_root_drafts = initial_root_drafts

    def proposal_directives(self, snapshot: SearchSnapshot, count: int) -> tuple[ProposalDirective, ...]:
        if _drafting(snapshot, self.initial_root_drafts):
            return _draft_directives(count)
        return super().proposal_directives(snapshot, count)


class RootDraftingContinuationTree(ContinuationTreeStrategy):
    """The same opt-in drafts for the v1 continuation tree, which MLE-bench runs.

    MLE sessions are launched with --version v1, so the v4-only drafts never
    applied there. Once the last draft is observed, the local burst is opened on
    the best scored draft; otherwise the tree would keep refining whichever
    draft came last, because every scored global action opens a burst on
    itself. From then on the unchanged continuation tree picks parents. Its
    config (and so its checkpoint identity) records the draft count.
    """

    def __init__(self, *, initial_root_drafts: int, **settings) -> None:
        _check_drafts(initial_root_drafts)
        super().__init__(**settings)
        self.initial_root_drafts = initial_root_drafts
        self.config = {**self.config, "initial_root_drafts": initial_root_drafts}

    def _choice(self, snapshot: SearchSnapshot) -> tuple[str, str]:
        # select() re-checks every action against _choice, so the draft parent is
        # decided here rather than only in the proposal directive. "global" closes
        # any local burst, as a fresh global parent does.
        if _drafting(snapshot, self.initial_root_drafts):
            return "root", "global"
        return super()._choice(snapshot)

    def observe(self, observation) -> None:
        super().observe(observation)
        # Same count as _drafting: every executed action from root (node_visits).
        from_root = [o for o in self.observations
                     if self.action_bindings.get(o.action_id, {}).get("code_parent_id") == "root"]
        if len(from_root) != self.initial_root_drafts or from_root[-1] is not observation:
            return
        scored = [o for o in from_root
                  if o.executable and o.score_bearing and o.artifact_valid
                  and o.program_node_id is not None and o.native_metric is not None]
        if not scored:
            self.burst = None
            return
        best = max(scored, key=lambda o: o.native_metric)
        self.burst = {
            "origin_action_id": self.action_bindings[best.action_id]["origin_action_id"],
            "best_id": best.program_node_id, "best_metric": best.native_metric,
            "scored": 0, "stagnation": 0, "attempts": 0, "closed": False,
        }

    def proposal_directives(self, snapshot: SearchSnapshot, count: int) -> tuple[ProposalDirective, ...]:
        if _drafting(snapshot, self.initial_root_drafts):
            if count != 1:
                raise ValueError("continuation tree requires incremental proposal_batch_size=1")
            return _draft_directives(count)
        return super().proposal_directives(snapshot, count)
