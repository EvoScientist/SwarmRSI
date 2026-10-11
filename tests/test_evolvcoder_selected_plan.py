"""R34 zero-provider handoff regressions; synthetic executions are not science scores."""
from dataclasses import asdict, replace
import json
from pathlib import Path
from unittest.mock import patch

import pytest

from evolvcoder.campaign import ProposedAction, VSearchCoordinator
from evolvcoder.coder_engine import ContractBoundedCoder
from evolvcoder.coder_models import (
    CoderConfig, CoderVersion, ScientificActionContract, ValidationCommand,
    coder_sha256, make_selected_scientific_plan, repair_selected_scientific_plan,
    selected_scientific_plan,
)
from evolvcoder.contracts import (
    ActionCandidate, ActionOperator, ResearchDependency, ScientificFamily,
)
from evolvcoder.strategies import ProposalDirective
from test_evolvcoder_incremental_feedback import campaign, next_step, ScriptedExecutor

try:
    from evolvcoder.naturebench_session import Planner
except ModuleNotFoundError as exc:
    if exc.name != "evolvcoder.naturebench_session":
        raise
    from scripts.run_evolvcoder_v_search import Planner


def canonical(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False)


def reseal(record):
    record["plan_sha256"] = coder_sha256({k: v for k, v in record.items() if k != "plan_sha256"})
    return canonical(record)


def action():
    return ActionCandidate(
        id="selected", hypothesis_id="hypothesis", operator=ActionOperator.IMPROVE_PROGRAM,
        dependencies=(ResearchDependency("root", "code_parent", "a" * 64),),
        family=ScientificFamily("rank transform", "KMeans", "matching", "fixed metric",
                                ("PCA components = 50", "n_init = 5")),
        predicted_program_quality=.5, predicted_information_gain=.3,
        predicted_reuse_value=.2, predicted_validity=.8, expected_cost=.3,
        hypothesis_statement="Preserve cross-modal geometry.",
    )


def contract(candidate=None, description="完整 implementation description"):
    candidate = candidate or action()
    return ScientificActionContract(
        candidate.id, "Keep the fixed method.", ("run.py",),
        hypothesis_id=candidate.hypothesis_id, research_contract_sha256=candidate.action_sha256,
        selected_plan_json=make_selected_scientific_plan(candidate, description),
    )


def coder_config(selected=None):
    parent_sha = "a" * 64
    if selected is not None and selected.selected_plan_json is not None:
        execution = selected_scientific_plan(selected.selected_plan_json)["execution_action"]
        parent_sha = next(d["source_sha256"] for d in execution["dependencies"] if d["kind"] == "code_parent")
    return CoderConfig(
        version=CoderVersion.C3, seed=7, model="synthetic/never-called", route="offline",
        executable_parent="run.py", expected_parent_sha256=parent_sha,
        action=selected or ScientificActionContract("legacy", "Keep the fixed method.", ("run.py",)),
        validation=(ValidationCommand("syntax", ("python", "-m", "py_compile", "run.py")),),
        max_model_calls=4, max_tool_calls=4, max_input_tokens=10000, max_output_tokens=1000,
        max_wall_seconds=30, max_repairs=1,
    )


def payload(cfg):
    coder = ContractBoundedCoder(cfg, None)
    with patch.object(coder, "_read_text", return_value="print(0)\n"):
        return coder._payload(Path("/nonexistent-offline-fixture"), {}, 0, 0, 0, 0, [])


def raw(short=False, long=False):
    return {
        "slot_index": 0, "operator": "improve_program", "diagnostic_source_action_id": "",
        "hypothesis": "5" if short else "A fixed matching hypothesis with public features.",
        "description": "I" if short else ("full description 开始\n" + "detail;" * (4000 if long else 3) + "END"),
        "family": {
            "input_transform": "1" if short else "rank Gaussian features",
            "primary_operator": "4" if short else "MiniBatchKMeans and linear_sum_assignment",
            "output_construction": "3" if short else "one-to-one matches",
            "objective_or_test": "2" if short else "fixed benchmark metric",
            "capacity_controls": [f"complete-control-{i}: " + "detail;" * 25 for i in range(8)],
        },
        "predictions": dict.fromkeys(("quality", "information_gain", "reuse", "validity", "cost"), .5),
    }


def configure_planner(loop, row):
    planner = Planner.__new__(Planner)
    class Model:
        def respond(self, request):
            return {"proposals": [row]}, {"input_tokens": 0, "output_tokens": 0}
    planner.model, planner.seed, planner.batch = Model(), 7, 0
    planner.required_symbols = ("main",)
    planner.preserved_source_patterns = ()
    planner.task_context = "PUBLIC BACKGROUND " * 3000
    planner.available_import_roots, planner.unavailable_import_roots = ("numpy",), ()
    planner.audit_root = None
    loop.proposer = planner
    return planner


@pytest.mark.parametrize("short,long", [(True, False), (False, False), (False, True)])
def test_real_full_planner_handoff_survives_clipped_context(tmp_path, short, long):
    loop = campaign(tmp_path)
    row = raw(short, long)
    configure_planner(loop, row)
    assert loop.expand() == 1
    proposal = next(iter(loop.actions.values()))
    cfg = coder_config(proposal.contract)
    projected = payload(cfg)
    selected = projected["selected_scientific_plan"]
    assert selected["research_action"] == json.loads(canonical(asdict(proposal.candidate)))
    assert selected["execution_action"] == selected["research_action"]
    assert selected["implementation_description"] == row["description"]
    assert selected["research_action"]["family"] == row["family"]
    assert "selected_plan_json" not in projected["action_contract"]
    assert projected["budget_remaining"]["input_tokens"] == 10000
    # Every turn gets the plan independently of a long executable/context/history.
    coder = ContractBoundedCoder(cfg, None)
    (tmp_path / "run.py").write_text("# public executable\n" * 4000)
    (tmp_path / "TASK_CONTEXT.md").write_text("public background\n" * 4000)
    history = [{"response": "history" * 20000}]
    later = coder._payload(tmp_path, {"error": "synthetic"}, 2, 2, 3, 4, history)
    assert later["selected_scientific_plan"] == selected
    assert len(later["executable_parent_content"]) == 30000
    assert len(later["supplied_context_files"]["TASK_CONTEXT.md"]) == 20000
    projected["selected_scientific_plan"]["research_action"]["family"]["capacity_controls"].clear()
    assert payload(cfg)["selected_scientific_plan"] == selected_scientific_plan(proposal.contract.selected_plan_json)
    assert cfg.config_sha256 == coder_sha256(cfg.canonical())


def test_legacy_exact_payload_and_config_hash_from_pre_r34_installed_package():
    # The CONFIG hash is independently captured from the unmodified noneditable P4b
    # local wheel and is unchanged. It is the one a run records and re-verifies
    # (campaign.py compares outcome.config_sha256 against the config it asked for), so
    # keeping it fixed is what makes a run resumable and an old cell still checkable.
    cfg = coder_config()
    assert cfg.config_sha256 == "8173a7e82565211b0ff049f0d06462af2faf43f7e83cf4a38f5118737be9909b"
    # The PAYLOAD hash changed once, deliberately, from
    # 421af6d606649534b204e52cfeea0a85a50f6647dd39c3a5388c03fa6fa68507, when
    # budget_remaining gained wall_seconds. Nothing at runtime hashes the payload; this
    # pin exists so the prompt cannot drift unnoticed, and the drift here was the point.
    # The model was being handed its call and token budgets and no clock at all, which
    # on a campaign that caps no count reads as thousands of calls remaining while the
    # deadline is what ends the action -- the mechanism behind cells that spent a whole
    # action writing probe scripts and never an implementation.
    assert coder_sha256(payload(cfg)) == "e92bdfab8ce26d68131310c4405c58fa4e08b95f49d3a86e8f0836371a48f6b3"
    assert "selected_plan_json" not in cfg.canonical()["action"]
    assert "selected_scientific_plan" not in payload(cfg)
    assert replace(cfg, config_sha256=cfg.config_sha256) == cfg
    restored_action = ScientificActionContract(**cfg.action.to_dict())
    assert replace(cfg, action=restored_action) == cfg


def test_description_has_its_own_complete_record_and_config_binding():
    left, right = contract(description="prefix" * 9000 + "LEFT"), contract(description="prefix" * 9000 + "RIGHT")
    assert left.research_contract_sha256 == right.research_contract_sha256
    assert selected_scientific_plan(left.selected_plan_json)["plan_sha256"] != selected_scientific_plan(right.selected_plan_json)["plan_sha256"]
    assert coder_config(left).config_sha256 != coder_config(right).config_sha256
    raw_record = json.loads(left.selected_plan_json)
    raw_record["implementation_description"] = "tampered"
    with pytest.raises(ValueError, match="digest"):
        replace(left, selected_plan_json=canonical(raw_record))


def test_selected_plan_cannot_bind_a_different_executable_parent():
    with pytest.raises(ValueError, match="different executable parents"):
        replace(coder_config(contract()), expected_parent_sha256="b" * 64, config_sha256="")
    # Absence of the new record retains legacy behavior and canonical identity.
    assert replace(coder_config(), expected_parent_sha256="b" * 64, config_sha256="").action.selected_plan_json is None


@pytest.mark.parametrize("has_plan", [False, True])
def test_config_json_roundtrip_preserves_complete_record_and_hash(has_plan):
    cfg = coder_config(contract() if has_plan else None)
    raw_config = json.loads(canonical({**cfg.canonical(), "config_sha256": cfg.config_sha256}))
    raw_config["version"] = CoderVersion(raw_config["version"])
    raw_config["action"] = ScientificActionContract(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in raw_config["action"].items()
    })
    raw_config["validation"] = tuple(ValidationCommand(**{
        key: tuple(value) if isinstance(value, list) else value
        for key, value in row.items()
    }) for row in raw_config["validation"])
    restored = CoderConfig(**raw_config)
    assert restored.config_sha256 == cfg.config_sha256
    assert payload(restored)["action_contract"] == payload(cfg)["action_contract"]
    assert restored.action.selected_plan_json == cfg.action.selected_plan_json


@pytest.mark.parametrize("corruption", [
    "hash", "content", "missing_hash", "empty_hash", "unknown_field", "missing_family",
    "bad_controls", "bad_dependencies", "boolean_prediction", "wrong_operator",
])
def test_inner_candidate_is_reconstructed_not_trusted(corruption):
    record = selected_scientific_plan(contract().selected_plan_json)
    candidate = record["execution_action"]
    if corruption == "hash":
        candidate["action_sha256"] = "b" * 64
    elif corruption == "content":
        candidate["family"]["capacity_controls"].append("unbound change")
    elif corruption == "missing_hash":
        candidate.pop("action_sha256")
    elif corruption == "empty_hash":
        candidate["action_sha256"] = ""
    elif corruption == "unknown_field":
        candidate["invented"] = "field"
    elif corruption == "missing_family":
        candidate["family"].pop("primary_operator")
    elif corruption == "bad_controls":
        candidate["family"]["capacity_controls"] = 42  # a bare string is a one-element list
    elif corruption == "bad_dependencies":
        candidate["dependencies"][0]["kind"] = "invented"
    elif corruption == "boolean_prediction":
        candidate["predicted_validity"] = True
    else:
        candidate["operator"] = "not-an-operator"
    with pytest.raises(ValueError):
        selected_scientific_plan(reseal(record))


@pytest.mark.parametrize("text", ["{}", "[]", "null", "{", '{"x":NaN}', '{"x":Infinity}'])
def test_malformed_records_rejected(text):
    with pytest.raises(ValueError):
        selected_scientific_plan(text)


def test_noncanonical_duplicate_keys_and_null_description_rejected():
    text = contract().selected_plan_json
    for changed in (" " + text, text.replace('"schema_version":', '"duplicate":0,"duplicate":1,"schema_version":')):
        with pytest.raises(ValueError, match="canonical"):
            selected_scientific_plan(changed)
    record = json.loads(text)
    record["implementation_description"] = None
    with pytest.raises(ValueError, match="text"):
        selected_scientific_plan(reseal(record))


@pytest.mark.parametrize("change", [
    {"action_id": "different"}, {"hypothesis_id": "different"},
    {"research_contract_sha256": "b" * 64},
])
def test_contract_rejects_mismatched_identity(change):
    with pytest.raises(ValueError, match="bindings"):
        replace(contract(), **change)


def test_proposal_rejects_same_id_different_execution_candidate():
    candidate = replace(action(), predicted_validity=.9, action_sha256="")
    with pytest.raises(ValueError, match="execution candidate"):
        ProposedAction(candidate, contract())


@pytest.mark.parametrize("has_plan", [False, True])
@pytest.mark.parametrize("migration", [False, True])
def test_failed_action_repair_roundtrip_preserves_both_identities(tmp_path, has_plan, migration):
    loop = campaign(tmp_path, scores=[None, .4], islands=migration)
    row = raw()
    if migration:
        loop.nodes["source"] = replace(loop.nodes["root"], id="source", native_metric=.7)
        loop.strategy.proposal_directives = lambda snapshot, count: (
            ProposalDirective("root", island_id="hybridization", intervention_source_ids=("source",)),
        )
        row["operator"] = "combine_programs"
    configure_planner(loop, row)
    assert loop.expand() == 1
    origin = next(iter(loop.actions.values()))
    if not has_plan:
        origin = replace(origin, contract=replace(origin.contract, selected_plan_json=None))
        loop.actions[origin.candidate.id] = origin
    first = loop.step()
    assert first.recovery_route == "repair_implementation"
    repair = next(p for p in loop.actions.values() if p.candidate.operator == ActionOperator.REPAIR_PROGRAM)
    assert repair.contract.research_contract_sha256 == origin.candidate.action_sha256
    assert repair.candidate.action_sha256 != origin.candidate.action_sha256
    assert repair.mutation_context is origin.mutation_context
    if migration:
        assert repair.mutation_context.kernel == "cross_program_inspiration"
        assert repair.mutation_context.research_contract_sha256 == origin.candidate.action_sha256
        assert repair.mutation_context.inspirations[0].particle_id == "source"
    if has_plan:
        plan = selected_scientific_plan(repair.contract.selected_plan_json)
        cfg = coder_config(repair.contract)
        if migration:
            cfg = replace(cfg, mutation_context=repair.mutation_context, config_sha256="")
        assert cfg.expected_parent_sha256 == loop.nodes["root"].executable_sha256
        with pytest.raises(ValueError, match="different executable parents"):
            replace(cfg, expected_parent_sha256="e" * 64, config_sha256="")
        assert plan["research_action"] == json.loads(canonical(asdict(origin.candidate)))
        assert plan["execution_action"] == json.loads(canonical(asdict(repair.candidate)))
        assert plan["implementation_description"] == row["description"]
        assert plan["execution_action"]["dependencies"][-1]["kind"] == "diagnostic_source"
        assert plan["execution_action"]["dependencies"][-1]["source_id"] == origin.candidate.id
    else:
        assert repair.contract.selected_plan_json is None
        assert "selected_plan_json" not in repair.contract.to_dict()
    checkpoint = json.loads(json.dumps(loop.checkpoint()))
    restored = VSearchCoordinator.restore(
        strategy=type(loop.strategy)(), proposer=loop.proposer,
        executor=ScriptedExecutor(tmp_path / "restored", [.4]), payload=checkpoint,
    )
    assert restored.actions[repair.candidate.id] == repair
    assert restored.checkpoint()["checkpoint_sha256"] == checkpoint["checkpoint_sha256"]
    # The original bounded repair executes, not an extra planner call or new hypothesis.
    completed = next_step(restored)
    assert completed.action_id == repair.candidate.id
    assert completed.code_parent_id == first.code_parent_id == "root"
    assert restored.accounting.score_evaluations == 1
    assert len(restored.actions) == 2


@pytest.mark.parametrize("change", ["hypothesis", "family", "parent", "diagnostic", "operator"])
def test_repair_cannot_change_scientific_origin(change):
    original = action()
    repaired = replace(original, id="repair", operator=ActionOperator.REPAIR_PROGRAM,
                       dependencies=(*original.dependencies, ResearchDependency(original.id, "diagnostic_source", "c" * 64)),
                       action_sha256="")
    if change == "hypothesis":
        repaired = replace(repaired, hypothesis_id="other", action_sha256="")
    elif change == "family":
        repaired = replace(repaired, family=replace(repaired.family, primary_operator="Leiden"), action_sha256="")
    elif change == "parent":
        repaired = replace(repaired, dependencies=(ResearchDependency("other", "code_parent", "d" * 64),
                                                   repaired.dependencies[-1]), action_sha256="")
    elif change == "diagnostic":
        repaired = replace(repaired, dependencies=original.dependencies, action_sha256="")
    else:
        repaired = replace(repaired, operator=ActionOperator.BRANCH_PROGRAM, action_sha256="")
    with pytest.raises(ValueError, match="repair must preserve"):
        repair_selected_scientific_plan(contract().selected_plan_json, repaired)
