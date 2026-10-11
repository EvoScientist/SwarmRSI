"""Synthetic contract fixtures ONLY; these records are not human research evidence."""
import copy
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from evolvcoder.project_audit import SCHEMA, export_project_audit, summarize_project_audit


def fixture(tmp_path, *, systems=("Evolvcoder",), requirements=2):
    source = tmp_path / "synthetic-evidence.txt"
    source.write_text("SYNTHETIC TEST FIXTURE: not a scientific or human judgment.\n")
    digest = hashlib.sha256(source.read_bytes()).hexdigest()
    return {
        "schema_version": SCHEMA, "rubric_version": "synthetic-rubric-v1",
        "rubric_source_refs": ["synthetic-source"],
        "evidence": [{"evidence_id": "synthetic-source", "path": source.name, "sha256": digest}],
        "runs": [{"run_id": f"r{i}", "system": system, "benchmark": "synthetic-bench",
                  "task_id": "synthetic-task", "terminal": True, "delivery_status": "delivered",
                  "source_refs": ["synthetic-source"], "artifact_refs": ["synthetic-source"],
                  "promotion_declaration": None} for i, system in enumerate(systems)],
        "requirements": [{"benchmark": "synthetic-bench", "task_id": "synthetic-task",
                          "requirement_id": f"q{i}", "rubric_version": "synthetic-rubric-v1",
                          "scope": "public", "applicable": True, "critical": True,
                          "criterion": f"synthetic requirement {i}", "source_refs": ["synthetic-source"]}
                         for i in range(requirements)],
        "audit_protocols": [{"audit_id": "human", "provenance_type": "expert_adjudicated",
                             "rubric_version": "synthetic-rubric-v1", "source_refs": ["synthetic-source"]}],
        "human_reviews": [], "adjudications": [], "automated_audits": [],
    }


def assessment(bundle, label, *, requirement_id=None, run_id="r0", audit_id="human"):
    run = next(x for x in bundle["runs"] if x["run_id"] == run_id)
    return {"audit_id": audit_id, "run_id": run_id,
            "unit_type": "project" if requirement_id is None else "requirement",
            "requirement_id": requirement_id, "rubric_version": bundle["rubric_version"],
            "label": label, "reason": "Synthetic test reason; not human evidence.",
            "evidence_refs": ["synthetic-source"], "artifact_refs": list(run["artifact_refs"])}


def adjudicate(bundle, label, **kwargs):
    common = assessment(bundle, label, **kwargs)
    prefix = f"{common['audit_id']}:{common['run_id']}:{common['requirement_id']}"
    ids = []
    for reviewer_id in ("synthetic-reviewer-1", "synthetic-reviewer-2"):
        review_id = f"{prefix}:{reviewer_id}"
        ids.append(review_id)
        bundle["human_reviews"].append({**common, "review_id": review_id,
            "reviewer_id": reviewer_id, "reviewer_role": "independent_expert",
            "reviewer_identity_refs": ["synthetic-source"], "provenance_type": "human_review"})
    bundle["adjudications"].append({**common, "adjudication_id": prefix,
        "review_ids": ids, "provenance_type": "expert_adjudicated"})


def promote(bundle, tmp_path, index=0):
    run = bundle["runs"][index]
    evidence = {row["evidence_id"]: row for row in bundle["evidence"]}
    declaration = {"schema_version": "evolvcoder-terminal-promotion-declaration-v1", "run_id": run["run_id"],
        "terminal": True, "delivery_status": "delivered", "explicitly_promoted": True,
        "artifacts": sorted([{"path": evidence[x]["path"], "sha256": evidence[x]["sha256"]}
                             for x in run["artifact_refs"]], key=lambda item: item["path"])}
    path = tmp_path / f"synthetic-promotion-{index}.json"
    path.write_text(json.dumps(declaration))
    evidence_id = f"synthetic-promotion-{index}"
    bundle["evidence"].append({"evidence_id": evidence_id, "path": path.name,
                               "sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
    run["promotion_declaration"] = {"run_id": run["run_id"], "source_refs": [evidence_id],
        "artifact_refs": list(run["artifact_refs"]), "declaration_evidence_id": evidence_id}


def summarize(bundle, tmp_path):
    return summarize_project_audit(bundle, evidence_root=tmp_path)


@pytest.mark.parametrize("system", ["Evolvcoder", "Claude Code", "AIDE"])
def test_complete_fields_without_reviews_are_not_scientific_support(tmp_path, system):
    bundle = fixture(tmp_path, systems=(system,))
    summary = summarize(bundle, tmp_path)
    project = summary["projects"][0]
    assert project["public_aec"]["value"] == 0
    assert project["public_requirement_counts"]["missing"] == 2
    assert project["aec_status"] == "incomplete_labels"
    audit = summary["summaries"][0]["delivery_audit"]
    assert audit["attempted_count"] == audit["conclusive_count"] == 0
    assert audit["audited_deliverable_error_fraction"]["value"] is None


def test_uncertain_only_is_attempted_not_conclusive(tmp_path):
    bundle = fixture(tmp_path)
    promote(bundle, tmp_path)
    adjudicate(bundle, "inconclusive")
    adjudicate(bundle, "insufficient", requirement_id="q0")
    adjudicate(bundle, "supported", requirement_id="q1")
    result = summarize(bundle, tmp_path)
    assert result["projects"][0]["public_aec"]["value"] == .5
    audit = result["summaries"][0]["promotion_audit"]
    assert audit["attempted_coverage"]["value"] == 1
    assert audit["conclusive_coverage"]["value"] == 0
    assert audit["audited_fpr"]["value"] is None
    assert audit["inconclusive_count"] == 1


def test_two_original_reviewers_count_once_after_adjudication(tmp_path):
    bundle = fixture(tmp_path)
    for req in ("q0", "q1"):
        adjudicate(bundle, "supported", requirement_id=req)
    adjudicate(bundle, "valid")
    result = summarize(bundle, tmp_path)
    assert result["raw_human_review_count"] == 6
    assert result["adjudication_count"] == 3
    assert result["summaries"][0]["delivery_audit"]["conclusive_count"] == 1
    assert result["projects"][0]["public_aec"]["value"] == 1
    assert "human_attributable_cumulative_gain" not in result


def test_conflicting_raw_reviews_do_not_vote_automatically(tmp_path):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "inconclusive")
    bundle["human_reviews"][0]["label"] = "valid"
    bundle["human_reviews"][1]["label"] = "invalid"
    bundle["adjudications"] = []
    audit = summarize(bundle, tmp_path)["summaries"][0]["delivery_audit"]
    assert audit["attempted_count"] == 1
    assert audit["conclusive_count"] == 0
    assert audit["missing_verdict_count"] == 1


def test_invalid_requires_critical_evidence_and_is_distinct_from_non_delivery(tmp_path):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "invalid")
    with pytest.raises(ValueError, match="critical defect"):
        summarize(bundle, tmp_path)
    adjudicate(bundle, "contradicted", requirement_id="q0")
    result = summarize(bundle, tmp_path)
    assert result["summaries"][0]["delivery_audit"]["audited_deliverable_error_fraction"]["value"] == 1
    assert result["summaries"][0]["promotion_audit"]["audited_fpr"]["value"] is None


def test_public_denominator_does_not_depend_on_extra_claims_or_missing_labels(tmp_path):
    bundle = fixture(tmp_path)
    extra = {**bundle["requirements"][0], "requirement_id": "extra", "scope": "additional_claim",
             "applies_to_run_ids": ["r0"]}
    bundle["requirements"].append(extra)
    adjudicate(bundle, "supported", requirement_id="q0")
    adjudicate(bundle, "supported", requirement_id="extra")
    project = summarize(bundle, tmp_path)["projects"][0]
    assert project["public_aec"] == {"value": .5, "numerator": 1, "denominator": 2, "null_reason": None}
    assert project["requirement_label_coverage"]["value"] == .5
    assert project["additional_claim_requirement_count"] == 1


def test_three_systems_same_task_have_separate_units_and_same_denominator(tmp_path):
    bundle = fixture(tmp_path, systems=("Evolvcoder", "Claude Code", "AIDE"))
    adjudicate(bundle, "supported", requirement_id="q0", run_id="r0")
    adjudicate(bundle, "contradicted", requirement_id="q0", run_id="r1")
    rows = summarize(bundle, tmp_path)["projects"]
    assert [r["public_aec"]["value"] for r in rows] == [.5, 0, 0]
    assert all(r["public_aec"]["denominator"] == 2 for r in rows)


def test_non_delivery_is_zero_support_but_not_false_promotion(tmp_path):
    bundle = fixture(tmp_path)
    bundle["runs"][0].update(delivery_status="not_delivered", artifact_refs=[])
    result = summarize(bundle, tmp_path)
    assert result["projects"][0]["public_aec"]["value"] == 0
    assert result["projects"][0]["aec_status"] == "not_delivered"
    group = result["summaries"][0]
    assert group["not_delivered_count"] == 1
    assert group["delivery_rate"]["value"] == 0
    assert group["delivery_audit"]["cohort_size"] == 0
    assert group["delivery_audit"]["conclusive_coverage"]["value"] is None


def test_running_project_is_pending_not_non_delivery(tmp_path):
    bundle = fixture(tmp_path)
    bundle["runs"][0].update(terminal=False, delivery_status="pending", artifact_refs=[])
    result = summarize(bundle, tmp_path)
    assert result["projects"][0]["public_aec"]["value"] is None
    assert result["projects"][0]["public_aec"]["null_reason"] == "run_not_terminal"
    assert result["summaries"][0]["pending_count"] == 1
    assert result["summaries"][0]["not_delivered_count"] == 0


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["runs"].append(copy.deepcopy(b["runs"][0])), "duplicate identity"),
    (lambda b: b["requirements"].clear(), "zero applicable"),
    (lambda b: b["requirements"].append(copy.deepcopy(b["requirements"][0])), "duplicate identity"),
    (lambda b: b["requirements"][0].update(rubric_version="drifted"), "rubric mismatch"),
    (lambda b: b["runs"][0].update(terminal=False), "pending"),
    (lambda b: b["runs"][0].update(artifact_refs=[]), "source evidence"),
    (lambda b: b["runs"][0].update(source_refs=["unknown"]), "unknown evidence"),
    (lambda b: b["evidence"][0].update(sha256="a" * 64), "hash mismatch"),
])
def test_invalid_identity_and_evidence_rejected(tmp_path, mutation, match):
    bundle = fixture(tmp_path)
    mutation(bundle)
    with pytest.raises(ValueError, match=match):
        summarize(bundle, tmp_path)


@pytest.mark.parametrize("mutation,match", [
    (lambda b: b["adjudications"].append({**b["adjudications"][0], "adjudication_id": "second"}), "duplicate final"),
    (lambda b: b["adjudications"][0].update(review_ids=[b["human_reviews"][0]["review_id"]]), "two independently"),
    (lambda b: b["human_reviews"][1].update(reviewer_id="synthetic-reviewer-1"), "duplicate original"),
    (lambda b: b["human_reviews"][1].update(reviewer_role="author"), "two independently"),
    (lambda b: b["adjudications"][0].update(provenance_type="llm_assessed"), "provenance mismatch"),
    (lambda b: b["adjudications"][0].update(review_ids=["missing"]), "unknown original"),
    (lambda b: b["adjudications"][0].update(artifact_refs=[]), "source evidence"),
])
def test_adjudication_source_and_independence_rejections(tmp_path, mutation, match):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "inconclusive")
    mutation(bundle)
    with pytest.raises(ValueError, match=match):
        summarize(bundle, tmp_path)


def add_llm_protocol(bundle):
    bundle["audit_protocols"].append({"audit_id": "llm", "provenance_type": "llm_assessed",
        "rubric_version": bundle["rubric_version"], "source_refs": ["synthetic-source"],
        "judge_model": "synthetic-model", "judge_route": "synthetic-route", "prompt_sha256": "a" * 64})


def test_llm_and_human_never_merge_and_failed_call_is_not_a_verdict(tmp_path):
    bundle = fixture(tmp_path)
    add_llm_protocol(bundle)
    bundle["automated_audits"].append({**assessment(bundle, None, audit_id="llm"),
        "assessment_id": "synthetic-llm-project", "provenance_type": "llm_assessed", "call_status": "failed"})
    bundle["automated_audits"].append({**assessment(bundle, "supported", requirement_id="q0", audit_id="llm"),
        "assessment_id": "synthetic-llm-q0", "provenance_type": "llm_assessed", "call_status": "completed"})
    rows = summarize(bundle, tmp_path)["projects"]
    assert rows[0]["audit_id"] == "human" and rows[0]["public_aec"]["value"] == 0
    assert rows[1]["audit_id"] == "llm" and rows[1]["public_aec"]["value"] == .5
    assert rows[1]["project_audit_attempted"] is True
    assert rows[1]["project_audit_conclusive"] is False
    bundle["automated_audits"][0]["label"] = "valid"
    with pytest.raises(ValueError, match="failed LLM call"):
        summarize(bundle, tmp_path)


def test_promotion_never_inferred_from_exploit_or_score(tmp_path):
    bundle = fixture(tmp_path)
    bundle["runs"][0].update(recovery_route="exploit_verified_program", native_metric=100)
    assert not summarize(bundle, tmp_path)["summaries"][0]["promotion_audit"]["applicable"]
    promote(bundle, tmp_path)
    assert summarize(bundle, tmp_path)["summaries"][0]["promotion_audit"]["cohort_size"] == 1
    bundle["runs"][0]["promotion_declaration"]["run_id"] = "wrong-run"
    with pytest.raises(ValueError, match="promotion run mismatch"):
        summarize(bundle, tmp_path)


def test_evidence_byte_changes_fail_and_prior_export_stays_intact(tmp_path):
    bundle = fixture(tmp_path)
    path = tmp_path / "bundle.json"
    output = tmp_path / "result.json"
    path.write_text(json.dumps(bundle))
    export_project_audit(path, evidence_root=tmp_path, output=output)
    before = output.read_bytes()
    (tmp_path / "synthetic-evidence.txt").write_text("tampered synthetic bytes")
    with pytest.raises(ValueError, match="hash mismatch"):
        export_project_audit(path, evidence_root=tmp_path, output=output)
    assert output.read_bytes() == before
    assert not list(tmp_path.glob("*.tmp"))


def test_export_is_deterministic_and_preserves_original_labels(tmp_path):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "inconclusive")
    path = tmp_path / "bundle.json"
    output = tmp_path / "result.json"
    path.write_text(json.dumps(bundle))
    first = export_project_audit(path, evidence_root=tmp_path, output=output)
    before = output.read_bytes()
    second = export_project_audit(path, evidence_root=tmp_path, output=output)
    assert before == output.read_bytes() and first == second
    assert json.loads(output.read_text())["bundle"]["human_reviews"] == bundle["human_reviews"]
    with pytest.raises(ValueError, match="overwrite input"):
        export_project_audit(path, evidence_root=tmp_path, output=path)
    with pytest.raises(ValueError, match="overwrite evidence"):
        export_project_audit(path, evidence_root=tmp_path, output=tmp_path / "synthetic-evidence.txt")


def test_cli_has_same_semantics_and_zero_model_calls(tmp_path):
    bundle = fixture(tmp_path)
    path = tmp_path / "bundle.json"
    output = tmp_path / "export.json"
    path.write_text(json.dumps(bundle))
    cli = Path(__file__).resolve().parents[1] / "scripts/export_evolvcoder_project_audit.py"
    process = subprocess.run([sys.executable, str(cli), "--bundle", str(path),
        "--evidence-root", str(tmp_path), "--output", str(output)], capture_output=True, text=True,
        # A hang guard, not a performance bound: this takes ~2 s locally and timed
        # out at 30 s on a Snellius node where the whole suite took 22-62 minutes.
        timeout=300)
    assert process.returncode == 0, process.stderr
    assert json.loads(output.read_text())["summary"]["projects"][0]["aec_status"] == "incomplete_labels"


def test_empty_panel_is_explicit_not_perfect(tmp_path):
    bundle = fixture(tmp_path)
    bundle["runs"] = []
    result = summarize(bundle, tmp_path)
    assert result["projects"] == result["summaries"] == []
    assert result["status"] == "empty_panel"
    path = tmp_path / "empty.json"
    path.write_text(json.dumps(bundle))
    output = tmp_path / "previous.json"
    output.write_text("preserved previous export")
    with pytest.raises(ValueError, match="empty panel"):
        export_project_audit(path, evidence_root=tmp_path, output=output)
    assert output.read_text() == "preserved previous export"


def test_extra_claim_does_not_impose_obligation_on_another_system(tmp_path):
    bundle = fixture(tmp_path, systems=("Evolvcoder", "AIDE"))
    bundle["requirements"].append({**bundle["requirements"][0], "requirement_id": "extra",
                                   "scope": "additional_claim", "applies_to_run_ids": ["r0"]})
    for req in ("q0", "q1"):
        adjudicate(bundle, "supported", run_id="r1", requirement_id=req)
    adjudicate(bundle, "valid", run_id="r1")
    result = summarize(bundle, tmp_path)
    assert result["projects"][1]["project_verdict"] == "valid"
    assert result["projects"][1]["additional_claim_requirement_count"] == 0
    adjudicate(bundle, "supported", run_id="r1", requirement_id="extra")
    with pytest.raises(ValueError, match="does not belong"):
        summarize(bundle, tmp_path)


def test_source_paths_cannot_escape_evidence_root(tmp_path):
    bundle = fixture(tmp_path)
    bundle["evidence"][0]["path"] = str((tmp_path / "synthetic-evidence.txt").resolve())
    with pytest.raises(ValueError, match="root-relative"):
        summarize(bundle, tmp_path)


def test_promotion_artifact_binding_cannot_use_different_evidence(tmp_path):
    bundle = fixture(tmp_path)
    bundle["evidence"].append({**bundle["evidence"][0], "evidence_id": "different-reference"})
    promote(bundle, tmp_path)
    bundle["runs"][0]["promotion_declaration"]["artifact_refs"] = ["different-reference"]
    with pytest.raises(ValueError, match="exact terminal artifacts"):
        summarize(bundle, tmp_path)


def test_valid_cannot_be_derived_from_partial_critical_labels(tmp_path):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "supported", requirement_id="q0")
    adjudicate(bundle, "valid")
    with pytest.raises(ValueError, match="supported critical"):
        summarize(bundle, tmp_path)


def test_author_adjudication_is_allowed_but_never_reported_as_expert(tmp_path):
    bundle = fixture(tmp_path)
    bundle["audit_protocols"][0]["provenance_type"] = "author_adjudicated"
    adjudicate(bundle, "inconclusive")
    bundle["adjudications"][0]["provenance_type"] = "author_adjudicated"
    for row in bundle["human_reviews"]:
        row["reviewer_role"] = "author"
    result = summarize(bundle, tmp_path)
    assert result["projects"][0]["provenance_type"] == "author_adjudicated"


def test_non_delivery_cannot_be_labeled_scientifically_invalid(tmp_path):
    bundle = fixture(tmp_path)
    bundle["runs"][0].update(delivery_status="not_delivered", artifact_refs=[])
    adjudicate(bundle, "invalid")
    with pytest.raises(ValueError, match="non-delivery"):
        summarize(bundle, tmp_path)


def test_requirement_only_audit_counts_as_attempted_not_conclusive(tmp_path):
    bundle = fixture(tmp_path)
    adjudicate(bundle, "supported", requirement_id="q0")
    cohort = summarize(bundle, tmp_path)["summaries"][0]["delivery_audit"]
    assert cohort["attempted_count"] == 1 and cohort["conclusive_count"] == 0


def test_bundle_only_promotion_flag_cannot_change_source_declared_cohort(tmp_path):
    bundle = fixture(tmp_path)
    run = bundle["runs"][0]
    run["promotion_declaration"] = {"run_id": "r0", "source_refs": ["synthetic-source"],
                                    "artifact_refs": list(run["artifact_refs"])}
    with pytest.raises(ValueError, match="source-bound terminal declaration"):
        summarize(bundle, tmp_path)


@pytest.mark.parametrize("change", [
    {"explicitly_promoted": False}, {"run_id": "another-run"}, {"terminal": False},
    {"delivery_status": "pending"}, {"artifacts": []},
])
def test_source_declaration_rejects_false_promotion_and_artifact_mismatch(tmp_path, change):
    bundle = fixture(tmp_path)
    promote(bundle, tmp_path)
    evidence = bundle["evidence"][-1]
    path = tmp_path / evidence["path"]
    declaration = json.loads(path.read_text())
    declaration.update(change)
    path.write_text(json.dumps(declaration))
    evidence["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    with pytest.raises(ValueError, match="source"):
        summarize(bundle, tmp_path)


def test_declaration_parses_exact_hashed_bytes_even_if_changed_after_first_read(tmp_path, monkeypatch):
    bundle = fixture(tmp_path)
    promote(bundle, tmp_path)
    evidence = bundle["evidence"][-1]
    path = tmp_path / evidence["path"]
    original = json.loads(path.read_text())
    original["explicitly_promoted"] = False
    path.write_text(json.dumps(original))
    evidence["sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
    changed = {**original, "explicitly_promoted": True}
    actual_read_bytes = Path.read_bytes

    def changed_read_bytes(self):
        if self == path:
            return json.dumps(changed).encode()
        return actual_read_bytes(self)

    monkeypatch.setattr(Path, "read_bytes", changed_read_bytes)
    with pytest.raises(ValueError, match="parsed declaration bytes"):
        summarize(bundle, tmp_path)
