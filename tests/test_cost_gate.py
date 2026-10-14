"""Revision G cost gate: a proposal declares the complexity of its dominant operation
and the controller refuses what cannot fit the public scale or the timeout."""
import pytest

from swarmrsi.planner_protocol import (COMPLEXITY_CLASSES, compile_proposal_slots, constrained_proposal_items,
                                         cost_bound_structure_error, cost_gate_error, proposal_legality_error,
                                         proposal_structure_error)


def proposal(**bound):
    raw = {"slot_index": 0, "hypothesis": "h", "description": "d", "operator": "improve_program",
           "family": {"input_transform": "raw", "primary_operator": "op", "output_construction": "direct",
                      "objective_or_test": "metric", "capacity_controls": []},
           "predictions": {"quality": .5, "information_gain": .5, "reuse": .5, "validity": .5, "cost": .5}}
    if bound:
        raw["cost_bound"] = bound
    return raw


def test_cost_bound_is_optional_in_structure_but_checked_when_present():
    assert proposal_structure_error(proposal()) is None
    good = proposal(dominant_operation="batched top-k", complexity_class="bounded_neighbourhood", estimated_seconds=120)
    assert proposal_structure_error(good) is None
    assert cost_bound_structure_error(proposal(dominant_operation="x", complexity_class="quadratic_full")) == "proposal_cost_bound_fields"
    assert cost_bound_structure_error(proposal(dominant_operation="", complexity_class="linear", estimated_seconds=1)) == "proposal_cost_bound_operation_type"
    assert cost_bound_structure_error(proposal(dominant_operation="x", complexity_class="exponential", estimated_seconds=1)) == "proposal_cost_bound_class"
    assert cost_bound_structure_error(proposal(dominant_operation="x", complexity_class="linear", estimated_seconds=-1)) == "proposal_cost_bound_seconds_type"
    assert "quadratic_full" in COMPLEXITY_CLASSES and "bounded_neighbourhood" in COMPLEXITY_CLASSES


def test_gate_refuses_full_pairwise_on_large_instances_and_estimates_without_headroom():
    dense = proposal(dominant_operation="n x n Sinkhorn transport plan", complexity_class="quadratic_full", estimated_seconds=300)
    # The cross-modal instances have ~1e5 cells: refused regardless of the estimate.
    assert cost_gate_error(dense, max_rows=120_000, timeout_seconds=1800) == "proposal_cost_bound_infeasible_class"
    # The same method on a small instance is allowed.
    assert cost_gate_error(dense, max_rows=3_000, timeout_seconds=1800) is None
    bounded = proposal(dominant_operation="batched top-k", complexity_class="bounded_neighbourhood", estimated_seconds=300)
    assert cost_gate_error(bounded, max_rows=120_000, timeout_seconds=1800) is None
    slow = proposal(dominant_operation="batched top-k", complexity_class="bounded_neighbourhood", estimated_seconds=1200)
    assert cost_gate_error(slow, max_rows=120_000, timeout_seconds=1800) == "proposal_cost_bound_over_timeout"
    assert cost_gate_error(proposal(), max_rows=10, timeout_seconds=10) == "proposal_cost_bound_missing"
    # Unknown scale or timeout only disables the corresponding rule, never the declaration.
    assert cost_gate_error(dense, max_rows=None, timeout_seconds=None) is None


def test_response_schema_requires_cost_bound_only_when_the_slot_carries_the_gate():
    base = {"type": "object", "properties": {
        "hypothesis": {"type": "string"}, "description": {"type": "string"}, "operator": {"type": "string"},
        "family": {"type": "object", "properties": {k: {"type": "string"} for k in
                   ("input_transform", "primary_operator", "output_construction", "objective_or_test")}},
        "predictions": {"type": "object"}}, "required": ["hypothesis", "description", "operator", "family", "predictions"]}
    slots = compile_proposal_slots([{"code_parent_id": "root"}], [])
    off = constrained_proposal_items(base, slots)
    assert "cost_bound" not in off["oneOf"][0]["required"] if "oneOf" in off else "cost_bound" not in off["required"]
    for slot in slots:
        slot["cost_gate"] = True
    on = constrained_proposal_items(base, slots)
    variants = on["oneOf"] if "oneOf" in on else [on]
    assert all("cost_bound" in v["required"] for v in variants)


def test_a_single_capacity_control_string_is_accepted_as_a_one_element_list():
    raw = proposal()
    raw["family"]["capacity_controls"] = "number of components (e.g., 20)"
    assert proposal_structure_error(raw) is None
    # The validator coerces locally and leaves the provider message untouched: it is
    # already journalled, and mutating it breaks the learning-evidence binding audit
    # (an OpenProblems cell died on exactly that after two batches).
    assert raw["family"]["capacity_controls"] == "number of components (e.g., 20)"
    raw["family"]["capacity_controls"] = 20
    assert proposal_structure_error(raw) == "proposal_capacity_controls_type"


def test_evidence_consolidation_freezes_operator_and_family_core_in_protocol_and_schema():
    core = {"input_transform": "frozen input", "primary_operator": "frozen operator",
            "output_construction": "frozen output", "objective_or_test": "frozen objective"}
    directive = {
        "code_parent_id": "parent", "hypothesis_id": "h1", "intervention_source_ids": [],
        "research_episode": {"mode": "evidence_consolidation",
                             "hypothesis": {"id": "h1", "family": {**core, "capacity_controls": ["old"]}}},
    }
    slot = compile_proposal_slots([directive], [])[0]
    assert slot["allowed_operators"] == ["improve_program"]
    assert slot["frozen_family_core"] == core

    raw = proposal()
    raw["family"].update(core)
    assert proposal_legality_error(raw, slot) is None
    raw["family"]["primary_operator"] = "silently replaced operator"
    assert proposal_legality_error(raw, slot) == "consolidation_changed_frozen_family_core"

    base = {"type": "object", "properties": {
        "hypothesis": {"type": "string"}, "description": {"type": "string"},
        "operator": {"type": "string"}, "diagnostic_source_action_id": {"type": "string"},
        "family": {"type": "object", "properties": {
            **{key: {"type": "string"} for key in core},
            "capacity_controls": {"type": "array", "items": {"type": "string"}},
        }},
        "predictions": {"type": "object"},
    }, "required": ["hypothesis", "description", "operator", "family", "predictions"]}
    schema = constrained_proposal_items(base, [slot])
    assert schema["properties"]["operator"]["enum"] == ["improve_program"]
    for key, value in core.items():
        assert schema["properties"]["family"]["properties"][key] == {"type": "string", "const": value}
