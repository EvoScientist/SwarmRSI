"""Post-hoc project audit accounting; no search, grader or verdict-generation authority.

This version implements ARCHITECTURE.md sections 5.2--5.6. It deliberately does
not reinterpret the legacy trajectory fragment exporter as scientific auditing.
"""
from __future__ import annotations

import hashlib
import json
import os
import tempfile
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Mapping


SCHEMA = "evolvcoder-project-audit-bundle-v2"
REQUIREMENT_LABELS = {"supported", "contradicted", "insufficient"}
PROJECT_LABELS = {"valid", "invalid", "inconclusive", "not_delivered"}
PROVENANCES = {"expert_adjudicated", "author_adjudicated", "llm_assessed"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise ValueError(message)


def _text(value: Any) -> bool:
    return isinstance(value, str) and bool(value.strip())


def _digest(value: Any) -> str:
    return hashlib.sha256(json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":"),
        allow_nan=False,
    ).encode()).hexdigest()


def _index(rows: list[dict[str, Any]], fields: tuple[str, ...]) -> dict[tuple, dict]:
    result = {}
    for row in rows:
        _require(isinstance(row, dict), "records must be objects")
        key = tuple(row.get(field) for field in fields)
        _require(all(_text(x) for x in key), f"missing identity: {fields}")
        _require(key not in result, f"duplicate identity: {fields}={key}")
        result[key] = row
    return result


def _ratio(numerator: int, denominator: int, reason: str) -> dict[str, Any]:
    return {"value": numerator / denominator if denominator else None,
            "numerator": numerator, "denominator": denominator,
            "null_reason": None if denominator else reason}


def summarize_project_audit(bundle: Mapping[str, Any], *, evidence_root: Path) -> dict[str, Any]:
    """Verify source bytes and aggregate *supplied* labels, keeping sources separate.

    Hash verification authenticates the supplied bytes, not scientific truth or
    reviewer expertise. There are no fabricated reviews or automatic verdicts.
    Running projects remain pending; terminal non-deliveries remain in the panel.
    """
    _require(bundle.get("schema_version") == SCHEMA, "unsupported bundle schema")
    _require(_text(bundle.get("rubric_version")), "rubric_version required")
    rubric = bundle["rubric_version"]
    data = {}
    for name in ("evidence", "runs", "requirements", "audit_protocols",
                 "human_reviews", "adjudications", "automated_audits"):
        _require(isinstance(bundle.get(name), list), f"{name} must be an explicit list")
        data[name] = bundle[name]
    evidence = _index(data["evidence"], ("evidence_id",))
    root = Path(evidence_root).resolve(strict=True)
    _require(root.is_dir(), "evidence_root must be a directory")
    for row in evidence.values():
        _require(_text(row.get("path")), "evidence path required")
        relative = Path(row["path"])
        _require(not relative.is_absolute(), "evidence paths must be root-relative")
        path = (root / relative).resolve(strict=True)
        _require(path.is_relative_to(root) and path.is_file(), "evidence escapes root or is not a file")
        with path.open("rb") as handle:
            actual = hashlib.file_digest(handle, "sha256").hexdigest()
        _require(actual == row.get("sha256"), f"evidence hash mismatch: {row['evidence_id']}")

    def refs(value: Any, *, required: bool = True) -> None:
        _require(isinstance(value, list), "evidence references must be a list")
        _require(not required or bool(value), "source evidence reference required")
        _require(all(_text(x) and (x,) in evidence for x in value), "unknown evidence reference")
        _require(len(value) == len(set(value)), "duplicate evidence reference")

    refs(bundle.get("rubric_source_refs"))
    runs = _index(data["runs"], ("run_id",))
    requirements = _index(data["requirements"], ("benchmark", "task_id", "requirement_id"))
    protocols = _index(data["audit_protocols"], ("audit_id",))
    _require(bool(protocols), "declare at least one audit protocol, even when labels are missing")
    for row in protocols.values():
        _require(row.get("provenance_type") in PROVENANCES, "unsupported audit provenance")
        _require(row.get("rubric_version") == rubric, "audit protocol rubric mismatch")
        refs(row.get("source_refs"))
        if row["provenance_type"] == "llm_assessed":
            for name in ("judge_model", "judge_route", "prompt_sha256"):
                _require(_text(row.get(name)), f"LLM audit {name} required")
            _require(len(row["prompt_sha256"]) == 64 and
                     all(x in "0123456789abcdef" for x in row["prompt_sha256"]), "invalid prompt hash")
    for req in requirements.values():
        _require(req.get("rubric_version") == rubric, "requirement rubric mismatch")
        _require(req.get("scope") in {"public", "additional_claim"}, "requirement scope required")
        _require(type(req.get("applicable")) is bool and type(req.get("critical")) is bool,
                 "requirement applicability/criticality must be frozen booleans")
        _require(_text(req.get("criterion")), "requirement criterion required")
        refs(req.get("source_refs"))
        if req["scope"] == "additional_claim":
            applicable_runs = req.get("applies_to_run_ids")
            _require(isinstance(applicable_runs, list) and bool(applicable_runs)
                     and all(_text(x) and (x,) in runs for x in applicable_runs),
                     "additional claims must name their declaring runs")
            _require(len(applicable_runs) == len(set(applicable_runs)), "duplicate additional-claim run")
            _require(all((runs[(x,)].get("benchmark"), runs[(x,)].get("task_id")) ==
                         (req["benchmark"], req["task_id"]) for x in applicable_runs),
                     "additional claim run belongs to another task")
        else:
            _require("applies_to_run_ids" not in req,
                     "public requirements cannot vary by system or run")

    def task_requirements(run: Mapping[str, Any]) -> list[dict]:
        return [req for key, req in requirements.items()
                if key[:2] == (run["benchmark"], run["task_id"]) and req["applicable"]
                and (req["scope"] == "public" or run["run_id"] in req["applies_to_run_ids"])]

    for run in runs.values():
        for field in ("system", "benchmark", "task_id"):
            _require(_text(run.get(field)), f"run {field} required")
        _require(type(run.get("terminal")) is bool, "terminal must be explicit boolean")
        _require(run.get("delivery_status") in {"delivered", "not_delivered", "pending"}, "invalid delivery status")
        _require(run["terminal"] == (run["delivery_status"] != "pending"), "running projects must remain pending")
        refs(run.get("source_refs"))
        refs(run.get("artifact_refs"), required=run["delivery_status"] == "delivered")
        _require(bool([q for q in task_requirements(run) if q["scope"] == "public"]),
                 "zero applicable public requirements: rubric configuration error")
        _require("promotion_declaration" in run, "promotion declaration must be explicit or null")
        promotion = run["promotion_declaration"]
        if promotion is not None:
            _require(isinstance(promotion, dict), "promotion declaration must be an object")
            _require(run["terminal"] and run["delivery_status"] == "delivered", "promotion cohort requires terminal delivery")
            _require(promotion.get("run_id") == run["run_id"], "promotion run mismatch")
            refs(promotion.get("source_refs"))
            refs(promotion.get("artifact_refs"))
            _require(set(promotion["artifact_refs"]) == set(run["artifact_refs"]), "promotion must bind exact terminal artifacts")
            declaration_id = promotion.get("declaration_evidence_id")
            _require(_text(declaration_id) and declaration_id in promotion["source_refs"],
                     "promotion requires a source-bound terminal declaration")
            declaration_evidence = evidence[(declaration_id,)]
            declaration_path = (root / declaration_evidence["path"]).resolve(strict=True)
            _require(declaration_path.is_relative_to(root) and declaration_path.is_file(),
                     "declaration escapes evidence root or is not a file")
            declaration_bytes = declaration_path.read_bytes()
            _require(hashlib.sha256(declaration_bytes).hexdigest() == declaration_evidence["sha256"],
                     "parsed declaration bytes no longer match verified hash")
            declaration = json.loads(declaration_bytes)
            _require(isinstance(declaration, dict) and
                     declaration.get("schema_version") == "evolvcoder-terminal-promotion-declaration-v1",
                     "unsupported terminal promotion declaration")
            _require(declaration.get("run_id") == run["run_id"] and declaration.get("terminal") is True
                     and declaration.get("delivery_status") == "delivered"
                     and declaration.get("explicitly_promoted") is True,
                     "source does not declare this terminal delivery promoted")
            bound_artifacts = sorted(
                [{"path": evidence[(x,)]["path"], "sha256": evidence[(x,)]["sha256"]}
                 for x in run["artifact_refs"]], key=lambda item: item["path"])
            _require(declaration.get("artifacts") == bound_artifacts,
                     "source promotion declaration artifact mismatch")

    def unit(row: Mapping[str, Any]) -> tuple[str, str, str | None]:
        _require(row.get("unit_type") in {"project", "requirement"}, "unsupported unit type")
        requirement_id = row.get("requirement_id")
        _require((row["unit_type"] == "requirement") == _text(requirement_id), "requirement identity mismatch")
        _require(requirement_id is None or _text(requirement_id), "invalid requirement id")
        return row["run_id"], row["unit_type"], requirement_id

    def validate_assessment(row: Mapping[str, Any], *, failed: bool = False) -> tuple:
        _require((row.get("audit_id"),) in protocols, "unknown audit protocol")
        _require((row.get("run_id"),) in runs, "unknown reviewed run")
        run = runs[(row["run_id"],)]
        _require(run["terminal"], "final-project labels cannot be applied to running projects")
        key = unit(row)
        _require(row.get("rubric_version") == rubric, "assessment rubric mismatch")
        _require(_text(row.get("reason")), "assessment reason required")
        refs(row.get("evidence_refs"))
        refs(row.get("artifact_refs"), required=run["delivery_status"] == "delivered")
        _require(set(row["artifact_refs"]) == set(run["artifact_refs"]), "assessment terminal artifact mismatch")
        if key[1] == "requirement":
            req = requirements.get((run["benchmark"], run["task_id"], key[2]))
            _require(req is not None and req["applicable"], "unknown or inapplicable requirement")
            _require(req["scope"] == "public" or run["run_id"] in req["applies_to_run_ids"],
                     "additional claim does not belong to this run")
        if failed:
            _require(row.get("label") is None, "failed LLM call cannot contain a verdict")
        else:
            labels = PROJECT_LABELS if key[1] == "project" else REQUIREMENT_LABELS
            _require(row.get("label") in labels, "unsupported assessment label")
            if key[1] == "project":
                _require((row["label"] == "not_delivered") == (run["delivery_status"] == "not_delivered"),
                         "non-delivery is separate from scientific invalidity")
            elif run["delivery_status"] == "not_delivered":
                _require(row["label"] == "insufficient", "non-delivery cannot support or refute a scientific requirement")
        return (row["audit_id"], *key)

    reviews = _index(data["human_reviews"], ("review_id",))
    seen_reviews = set()
    for review in reviews.values():
        key = validate_assessment(review)
        _require(review.get("provenance_type") == "human_review", "raw human provenance required")
        _require(protocols[(key[0],)]["provenance_type"] != "llm_assessed", "human/LLM protocol mixing")
        _require(_text(review.get("reviewer_id")), "reviewer identity required")
        _require(review.get("reviewer_role") in {"independent_expert", "author", "other"}, "reviewer role required")
        refs(review.get("reviewer_identity_refs"))
        identity = (*key, review["reviewer_id"])
        _require(identity not in seen_reviews, "duplicate original review for reviewer and unit")
        seen_reviews.add(identity)

    effective: dict[tuple, dict] = {}
    _index(data["adjudications"], ("adjudication_id",))
    for row in data["adjudications"]:
        key = validate_assessment(row)
        protocol = protocols[(key[0],)]
        _require(protocol["provenance_type"] != "llm_assessed" and
                 row.get("provenance_type") == protocol["provenance_type"], "adjudication provenance mismatch")
        _require(key not in effective, "duplicate final adjudication for unit")
        ids = row.get("review_ids")
        _require(isinstance(ids, list) and bool(ids) and all(_text(x) for x in ids), "original review references required")
        _require(len(ids) == len(set(ids)), "duplicate original review reference")
        originals = []
        for review_id in ids:
            _require((review_id,) in reviews, "unknown original review")
            original = reviews[(review_id,)]
            _require((original["audit_id"], *unit(original)) == key, "review belongs to a different audit unit")
            originals.append(original)
        if protocol["provenance_type"] == "expert_adjudicated":
            _require(len({r["reviewer_id"] for r in originals}) >= 2 and
                     all(r["reviewer_role"] == "independent_expert" for r in originals),
                     "expert adjudication requires two independently identified expert originals")
        effective[key] = row

    _index(data["automated_audits"], ("assessment_id",))
    for row in data["automated_audits"]:
        _require(row.get("call_status") in {"completed", "failed"}, "LLM call status required")
        key = validate_assessment(row, failed=row["call_status"] == "failed")
        _require(row.get("provenance_type") == "llm_assessed" and
                 protocols[(key[0],)]["provenance_type"] == "llm_assessed", "LLM provenance mismatch")
        _require(key not in effective, "duplicate effective LLM assessment; keep retry logs as evidence")
        effective[key] = row

    projects = []
    for (audit_id,), protocol in sorted(protocols.items()):
        for (run_id,), run in sorted(runs.items()):
            reqs = task_requirements(run)
            public = [q for q in reqs if q["scope"] == "public"]
            assessments = {q["requirement_id"]: effective.get((audit_id, run_id, "requirement", q["requirement_id"])) for q in reqs}
            counts = Counter()
            for req in public:
                assessment = assessments[req["requirement_id"]]
                label = assessment.get("label") if assessment else None
                counts[label if label is not None else "missing"] += 1
            decision = effective.get((audit_id, run_id, "project", None))
            verdict = decision.get("label") if decision else None
            critical_labels = [assessments[q["requirement_id"]].get("label")
                               if assessments[q["requirement_id"]] else None for q in reqs if q["critical"]]
            if verdict == "valid":
                _require(bool(critical_labels) and all(x == "supported" for x in critical_labels),
                         "valid project verdict lacks supported critical requirements")
            if verdict == "invalid":
                _require("contradicted" in critical_labels, "invalid verdict lacks an evidenced critical defect")
            pending = not run["terminal"]
            supported = counts["supported"] if run["delivery_status"] == "delivered" else 0
            aec = (_ratio(supported, len(public), "zero_applicable_requirements") if not pending else
                   {"value": None, "numerator": None, "denominator": len(public), "null_reason": "run_not_terminal"})
            raw_attempt = any(k[:2] == (audit_id, run_id) for k in seen_reviews)
            effective_attempt = any(k[:2] == (audit_id, run_id) for k in effective)
            projects.append({
                "audit_id": audit_id, "provenance_type": protocol["provenance_type"],
                "run_id": run_id, "system": run["system"], "benchmark": run["benchmark"],
                "task_id": run["task_id"], "terminal": run["terminal"],
                "delivery_status": run["delivery_status"],
                "explicit_terminal_promotion": run["promotion_declaration"] is not None,
                "project_verdict": verdict, "verdict_missing_reason": None if verdict else
                ("run_not_terminal" if pending else "no_effective_project_verdict"),
                "project_audit_attempted": effective_attempt or raw_attempt,
                "project_audit_conclusive": verdict in {"valid", "invalid"},
                "public_aec": aec,
                "public_requirement_counts": {x: counts[x] for x in (*sorted(REQUIREMENT_LABELS), "missing")},
                "requirement_label_coverage": _ratio(len(public) - counts["missing"], len(public), "zero_applicable_requirements"),
                "aec_status": "pending" if pending else "not_delivered" if run["delivery_status"] == "not_delivered"
                else "incomplete_labels" if counts["missing"] else "complete_labels",
                "additional_claim_requirement_count": len(reqs) - len(public),
            })

    def cohort(rows: list[dict], *, promotion: bool = False) -> dict:
        conclusive = sum(r["project_audit_conclusive"] for r in rows)
        invalid = sum(r["project_verdict"] == "invalid" for r in rows)
        attempted = sum(r["project_audit_attempted"] for r in rows)
        return {
            "cohort_size": len(rows), "attempted_count": attempted, "conclusive_count": conclusive,
            "invalid_count": invalid,
            "inconclusive_count": sum(r["project_verdict"] == "inconclusive" for r in rows),
            "missing_verdict_count": sum(r["project_verdict"] is None for r in rows),
            "attempted_coverage": _ratio(attempted, len(rows), "empty_cohort"),
            "conclusive_coverage": _ratio(conclusive, len(rows), "empty_cohort"),
            "audited_fpr" if promotion else "audited_deliverable_error_fraction":
                _ratio(invalid, conclusive, "no_conclusive_audits"),
            "applicable": bool(rows),
            "scope": "explicit_terminal_delivered_promotion" if promotion else "terminal_delivered_projects",
        }

    groups = defaultdict(list)
    for project in projects:
        groups[(project["audit_id"], project["system"], project["benchmark"])].append(project)
    summaries = []
    for (audit_id, system, benchmark), rows in sorted(groups.items()):
        terminal = [r for r in rows if r["terminal"]]
        delivered = [r for r in terminal if r["delivery_status"] == "delivered"]
        summaries.append({
            "audit_id": audit_id, "provenance_type": protocols[(audit_id,)]["provenance_type"],
            "system": system, "benchmark": benchmark, "run_count": len(rows),
            "terminal_count": len(terminal), "pending_count": len(rows) - len(terminal),
            "not_delivered_count": sum(r["delivery_status"] == "not_delivered" for r in terminal),
            "delivery_rate": _ratio(len(delivered), len(terminal), "no_terminal_runs"),
            "delivery_audit": cohort(delivered),
            "promotion_audit": cohort([r for r in delivered if r["explicit_terminal_promotion"]], promotion=True),
        })
    result = {
        "schema_version": "evolvcoder-project-audit-summary-v2", "input_sha256": _digest(bundle),
        "status": "analyzed" if runs else "empty_panel",
        "rubric_version": rubric, "rubric_source_refs": bundle["rubric_source_refs"],
        "verified_evidence_count": len(evidence), "projects": projects, "summaries": summaries,
        "raw_human_review_count": len(reviews), "adjudication_count": len(data["adjudications"]),
        "automated_assessment_count": len(data["automated_audits"]),
        "limitations": ["Supplied identity attestations do not independently prove reviewer expertise.",
                        "AEC with missing labels is observed support coverage, not a completed audit or refutation.",
                        "No score gain is summed across reviewers, tasks or metrics.",
                        "This post-hoc output has no runtime promotion authority."],
    }
    result["summary_sha256"] = _digest(result)
    return result


def export_project_audit(bundle_path: Path, *, evidence_root: Path, output: Path) -> dict:
    """Validate everything before atomically publishing; retain the previous result on failure."""
    bundle_path, output = Path(bundle_path).resolve(), Path(output).resolve()
    _require(bundle_path != output, "output must not overwrite input bundle")
    bundle = json.loads(bundle_path.read_text(encoding="utf-8"))
    for row in bundle.get("evidence", []):
        _require(output != (Path(evidence_root) / row["path"]).resolve(), "output must not overwrite evidence")
    summary = summarize_project_audit(bundle, evidence_root=evidence_root)
    _require(summary["status"] != "empty_panel", "empty panel cannot replace a published audit")
    # Preserve original independent reviews, adjudications and LLM records verbatim.
    exported = {"schema_version": "evolvcoder-project-audit-export-v2", "bundle": bundle, "summary": summary}
    serialized = json.dumps(exported, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n"
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=output.parent,
                                         prefix=f".{output.name}.", suffix=".tmp", delete=False) as handle:
            temporary = Path(handle.name)
            handle.write(serialized)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return summary
