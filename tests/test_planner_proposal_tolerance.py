"""s41592-023-02035-2 stopped with four hours left because two proposals were refused
for form, not substance: one omitted slot_index, one added a note key to family."""
from swarmrsi.planner_protocol import proposal_legality_error

SLOT = {"slot_index": 0, "allowed_operators": ["combine_programs"], "repair_diagnostic_ids": [],
        "evidence_diagnostic_ids": ["a0004_6e26811c05"]}
FAMILY = {"input_transform": "x", "primary_operator": "y", "output_construction": "z",
          "objective_or_test": "w", "capacity_controls": ["k"]}


def proposal(**change):
    raw = {"slot_index": 0, "hypothesis": "h", "description": "d", "operator": "combine_programs",
           "family": dict(FAMILY), "diagnostic_source_action_id": "a0004_6e26811c05",
           "predictions": {"quality": .5, "information_gain": .5, "reuse": .5, "validity": .5, "cost": .5}}
    raw.update(change)
    return raw


def test_missing_slot_index_takes_the_array_position():
    raw = proposal(); del raw["slot_index"]
    assert proposal_legality_error(raw, SLOT) is None


def test_extra_annotation_key_in_family_is_ignored():
    assert proposal_legality_error(proposal(family={**FAMILY, "objective_or_test_note": "n"}), SLOT) is None


def test_real_errors_are_still_refused():
    assert proposal_legality_error(proposal(slot_index=1), SLOT) == "proposal_slot_index_mismatch"
    missing = dict(FAMILY); del missing["primary_operator"]
    assert proposal_legality_error(proposal(family=missing), SLOT) == "proposal_family_fields"
    assert proposal_legality_error(proposal(operator="improve_program"), SLOT) == "operator_not_available_for_slot"
    raw = proposal(); del raw["hypothesis"]
    assert proposal_legality_error(raw, SLOT) == "proposal_required_or_extra_fields"
    assert proposal_legality_error(proposal(slot_index="0"), SLOT) == "proposal_slot_index_mismatch"


from swarmrsi.planner_protocol import normalize_proposal  # noqa: E402


def test_predictions_nested_in_family_are_lifted_on_a_copy():
    raw = proposal(); preds = raw.pop("predictions")
    raw["family"] = {**FAMILY, "predictions": preds}
    fixed = normalize_proposal(raw)
    assert proposal_legality_error(raw, SLOT) == "proposal_required_or_extra_fields"
    assert proposal_legality_error(fixed, SLOT) is None
    assert fixed["predictions"] == preds and "predictions" not in fixed["family"]
    assert "predictions" in raw["family"] and "predictions" not in raw  # journalled message untouched


def test_normalize_leaves_well_formed_proposals_alone():
    raw = proposal()
    assert normalize_proposal(raw) is raw
