"""Controller-owned planner action spaces; no scientific judgments or model calls."""
from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

from .contracts import ActionOperator
from .diagnostics import is_repair_seed

PROTOCOL_VERSION = "swarmrsi-planner-action-space-v2"


COMPLEXITY_CLASSES = ("linear", "n_log_n", "bounded_neighbourhood", "quadratic_full", "cubic_or_worse")


def cost_bound_structure_error(raw: dict) -> str | None:
    """Shape check for the optional cost_bound declaration (revision G cost gate)."""
    bound = raw.get("cost_bound")
    if bound is None:
        return None
    if not isinstance(bound, dict) or set(bound) != {"dominant_operation", "complexity_class", "estimated_seconds"}:
        return "proposal_cost_bound_fields"
    if not isinstance(bound["dominant_operation"], str) or not bound["dominant_operation"].strip():
        return "proposal_cost_bound_operation_type"
    if bound["complexity_class"] not in COMPLEXITY_CLASSES:
        return "proposal_cost_bound_class"
    seconds = bound["estimated_seconds"]
    if type(seconds) not in {int, float} or not math.isfinite(seconds) or seconds < 0:
        return "proposal_cost_bound_seconds_type"
    return None


def cost_gate_error(raw: dict, *, max_rows: int | None, timeout_seconds: int | None,
                    quadratic_row_limit: int = 20_000, headroom: float = 0.5) -> str | None:
    """Controller-side feasibility rule. Zero model calls; a declaration is required and checked.

    The Claude reference on the cross-modal task rejected every quadratic method
    by cost before writing code, and won. Our runs proposed a dense n x n
    transport three times on ~1e5 cells, timed out each time, and produced no
    program. This gate makes that decision a rule: a full-pairwise or worse
    method on a large public instance is refused at proposal time, and any
    method whose own estimate does not leave headroom under the execution
    timeout is refused as well.
    """
    bound = raw.get("cost_bound")
    if bound is None:
        return "proposal_cost_bound_missing"
    if max_rows is not None and max_rows >= quadratic_row_limit and \
            bound["complexity_class"] in {"quadratic_full", "cubic_or_worse"}:
        return "proposal_cost_bound_infeasible_class"
    if timeout_seconds is not None and bound["estimated_seconds"] > headroom * timeout_seconds:
        return "proposal_cost_bound_over_timeout"
    return None


def normalize_proposal(raw: dict) -> dict:
    """Undo one misplacement the planner makes, on a copy.

    s42256-023-00619-3's last batch nested the whole predictions object inside
    family; everything else was correct. A copy is returned so the journalled
    provider message is never mutated (the learning-evidence audit binds it).
    """
    if not isinstance(raw, dict) or "predictions" in raw:
        return raw
    family = raw.get("family")
    if isinstance(family, dict) and isinstance(family.get("predictions"), dict):
        fixed = dict(raw)
        fixed["family"] = {k: v for k, v in family.items() if k != "predictions"}
        fixed["predictions"] = family["predictions"]
        return fixed
    return raw


def proposal_structure_error(raw: dict) -> str | None:
    # slot_index is optional: array position is authoritative (proposal_legality_error
    # already refuses a present-but-different index). s41592-023-02035-2 lost a correct
    # proposal to its absence and, two empty batches later, four hours of budget.
    required = {"hypothesis", "description", "operator", "family", "predictions"}
    if not required.issubset(raw) or set(raw) - required - {"slot_index", "diagnostic_source_action_id",
                                                           "wisdom_entry_ids", "cost_bound"}:
        return "proposal_required_or_extra_fields"
    bound_error = cost_bound_structure_error(raw)
    if bound_error:
        return bound_error
    if "wisdom_entry_ids" in raw and (not isinstance(raw["wisdom_entry_ids"], list)
            or any(not isinstance(value, str) for value in raw["wisdom_entry_ids"])):
        return "proposal_wisdom_ids_type"
    if "slot_index" in raw and type(raw["slot_index"]) is not int:
        return "proposal_slot_index_type"
    if any(not isinstance(raw[key], str) or not raw[key].strip()
           for key in ("hypothesis", "description", "operator")):
        return "proposal_text_field_type_or_empty"
    family = raw["family"]
    family_keys = {"input_transform", "primary_operator", "output_construction",
                   "objective_or_test", "capacity_controls"}
    # All five keys are required; an extra annotation key (02035 added
    # "objective_or_test_note") is ignored, since the session reads only these five.
    if not isinstance(family, dict) or not family_keys.issubset(family):
        return "proposal_family_fields"
    if any(not isinstance(family[key], str) or not family[key].strip()
           for key in family_keys - {"capacity_controls"}):
        return "proposal_family_text_type_or_empty"
    controls = family["capacity_controls"]
    if isinstance(controls, str):
        # A single control written as a string ("number of components (e.g., 20)")
        # is the same declaration as a one-element list; the session's parser
        # already coerces it, so the validator must not refuse what the parser
        # accepts. An OpenProblems cell died on exactly this after two batches.
        # The coercion stays local: writing it back into `family` would mutate the
        # provider message that was already journalled, and the learning-evidence
        # audit would then reject the batch as a request/response binding mismatch.
        controls = [controls]
    if not isinstance(controls, list) or any(not isinstance(value, str) for value in controls):
        return "proposal_capacity_controls_type"
    predictions = raw["predictions"]
    if not isinstance(predictions, dict) or set(predictions) != {"quality", "information_gain", "reuse", "validity", "cost"}:
        return "proposal_prediction_fields"
    if any(type(value) not in {int, float} or not 0 <= value <= 1 or not math.isfinite(value)
           for value in predictions.values()):
        return "proposal_prediction_type_or_range"
    return None


def compile_proposal_slots(directives: list[dict], diagnostics: list[dict]) -> list[dict]:
    sources = {}
    for row in diagnostics:
        identifier, digest = row.get("action_id"), row.get("diagnostic_workspace_sha256")
        if not isinstance(identifier, str) or not identifier or not isinstance(digest, str):
            continue
        if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
            continue
        if identifier in sources:
            raise ValueError("ambiguous planner diagnostic identity")
        sources[identifier] = row
    slots = []
    for index, directive in enumerate(directives):
        episode = directive.get("research_episode") or {}
        consolidation = episode.get("mode") == "evidence_consolidation"
        repair_ids = sorted(key for key, row in sources.items()
                            if row.get("code_parent_id") == directive["code_parent_id"] and is_repair_seed(row))
        if directive.get("intervention_source_ids"):
            operators = [ActionOperator.COMBINE_PROGRAMS.value]
            repair_ids = []
        elif consolidation:
            # ECLC is a controller-owned continuation, not a fresh scientific
            # proposal.  Freeze its operator and scientific core in the tool
            # schema instead of relying on the planner to copy long family
            # strings byte-for-byte.  Capacity controls remain the only family
            # field the planner may refine.
            operators = [ActionOperator.IMPROVE_PROGRAM.value]
            repair_ids = []
        else:
            operators = [op.value for op in ActionOperator
                         if op not in {ActionOperator.REPAIR_PROGRAM, ActionOperator.COMBINE_PROGRAMS}]
            if repair_ids:
                operators.append(ActionOperator.REPAIR_PROGRAM.value)
        slot = {**directive, "slot_index": index, "allowed_operators": operators,
                      "evidence_diagnostic_ids": sorted(sources),
                      "repair_diagnostic_ids": repair_ids}
        if consolidation:
            family = (episode.get("hypothesis") or {}).get("family")
            core_keys = ("input_transform", "primary_operator", "output_construction", "objective_or_test")
            if not isinstance(family, dict) or any(not isinstance(family.get(key), str) or not family[key]
                                                   for key in core_keys):
                raise ValueError("evidence consolidation requires a frozen scientific family core")
            slot["frozen_family_core"] = {key: family[key] for key in core_keys}
        slots.append(slot)
    return slots


def proposal_legality_error(raw: dict, slot: dict) -> str | None:
    # Array position is authoritative. A model-supplied slot may confirm it,
    # but can never redirect the proposal to another controller slot.
    if "slot_index" in raw and (type(raw["slot_index"]) is not int
                                 or raw["slot_index"] != slot["slot_index"]):
        return "proposal_slot_index_mismatch"
    operator = raw.get("operator")
    diagnostic = raw.get("diagnostic_source_action_id", "")
    if operator == ActionOperator.REPAIR_PROGRAM.value and (
        diagnostic not in slot["repair_diagnostic_ids"]
    ):
        return "repair_requires_original_parent_and_diagnostic"
    if operator not in slot["allowed_operators"]:
        return "operator_not_available_for_slot"
    if diagnostic not in ["", *slot["evidence_diagnostic_ids"]]:
        return "unknown_diagnostic_source"
    if isinstance(raw.get("wisdom_entry_ids"), list) and any(
        entry not in slot.get("wisdom_entry_ids", []) for entry in raw["wisdom_entry_ids"]
    ):
        return "unknown_wisdom_source"
    structure_error = proposal_structure_error(raw)
    if structure_error:
        return structure_error
    frozen = slot.get("frozen_family_core")
    if isinstance(frozen, dict) and any(raw["family"].get(key) != value for key, value in frozen.items()):
        return "consolidation_changed_frozen_family_core"
    return None


def constrained_proposal_items(base: dict, slots: list[dict]) -> dict[str, Any]:
    variants = []
    for slot in slots:
        normal = [op for op in slot["allowed_operators"] if op != "repair_program"]
        for operators, diagnostic_ids, requires_diagnostic in (
            (normal, ["", *slot["evidence_diagnostic_ids"]], False),
            (["repair_program"] if slot["repair_diagnostic_ids"] else [],
             slot["repair_diagnostic_ids"], True),
        ):
            if not operators:
                continue
            row = deepcopy(base)
            for key in ("hypothesis", "description"):
                row["properties"][key].update(minLength=1, pattern=r"\S")
            for key in ("input_transform", "primary_operator", "output_construction", "objective_or_test"):
                row["properties"]["family"]["properties"][key].update(minLength=1, pattern=r"\S")
            for key, value in (slot.get("frozen_family_core") or {}).items():
                row["properties"]["family"]["properties"][key] = {"type": "string", "const": value}
            row["properties"]["slot_index"] = {"type": "integer", "enum": [slot["slot_index"]]}
            row["properties"]["operator"] = {"type": "string", "enum": operators}
            row["properties"]["diagnostic_source_action_id"] = {
                "type": "string", "enum": diagnostic_ids,
            }
            row["required"].append("slot_index")
            if slot.get("wisdom_entry_ids"):
                row["properties"]["wisdom_entry_ids"] = {
                    "type": "array", "items": {"type": "string", "enum": slot["wisdom_entry_ids"]},
                    "uniqueItems": True,
                }
            if requires_diagnostic:
                row["required"].append("diagnostic_source_action_id")
            if slot.get("cost_gate"):
                row["required"].append("cost_bound")
            variants.append(row)
    if not variants:
        raise ValueError("planner requires at least one legal proposal slot")
    return variants[0] if len(variants) == 1 else {"anyOf": variants}
