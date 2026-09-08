from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort, load_swing_research_cohort


def _cohort_inputs() -> tuple[dict[str, Any], dict[str, Any]]:
    original = [f"sec:{i:04}" for i in range(631)]
    inherited = original[:27]
    parent = {
        "modeled_security_count": 631,
        "modeled_security_ids_sha256": json_sha256(original),
        "excluded_security_ids": inherited,
        "excluded_security_ids_sha256": json_sha256(inherited),
        "excluded_security_count": 27,
        "retained_security_count": 604,
    }
    payload = {
        "schema_version": "market_predictor.swing_research_cohort",
        "scope": "retrospective_development_restriction",
        "price_basis_status": "not_certified_by_cohort",
        "combined_daily_inputs_sha256": json_sha256(parent),
        "original_security_ids": original,
        "inherited_excluded_security_ids": inherited,
        "warmup_only_security_ids": ["warmup:only"],
        "exclusions": [{"security_id": item, "tickers": ["AAA"], "reason": "unresolved_holding_identity"}
                       for item in original[27:45]],
        "maximum_exclusion_bps": 500,
        "cap_approval_reference": "test-only-explicit-policy",
        "source_files": {"parent.json": "a" * 64},
    }
    return payload, parent


def test_cumulative_exclusion_cap_and_retrospective_status() -> None:
    payload, _ = _cohort_inputs()
    blocked = SwingResearchCohort.model_validate_json(json.dumps(payload))
    assert not blocked.within_cap
    assert blocked.summary()["total_excluded_securities"] == 45
    assert blocked.summary()["retained_securities"] == 586
    assert blocked.summary()["excluded_fraction"] == pytest.approx(45 / 631)
    payload["maximum_exclusion_bps"] = 800
    accepted = SwingResearchCohort.model_validate_json(json.dumps(payload))
    assert accepted.within_cap
    assert accepted.summary()["accounting_eligible"] is False
    assert accepted.summary()["promotion_eligible"] is False
    assert accepted.sha256() != blocked.sha256()


def test_inherited_and_new_overlap_is_counted_once() -> None:
    payload, _ = _cohort_inputs()
    payload["exclusions"][0]["security_id"] = "sec:0000"
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    assert len(cohort.excluded_security_ids) == 44
    assert cohort.summary()["additional_excluded_securities"] == 17


@pytest.mark.parametrize("excluded_count,accepted", [(63, True), (64, False)])
def test_ten_percent_cap_counts_whole_securities_without_rounding(excluded_count: int, accepted: bool) -> None:
    payload, _ = _cohort_inputs()
    payload["maximum_exclusion_bps"] = 1000
    payload["exclusions"] = [
        {"security_id": item, "tickers": ["AAA"], "reason": "unresolved_holding_identity"}
        for item in payload["original_security_ids"][27:excluded_count]
    ]
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    assert cohort.within_cap is accepted
    assert cohort.summary()["total_excluded_securities"] == excluded_count
    assert cohort.summary()["accounting_eligible"] is False
    assert cohort.summary()["promotion_eligible"] is False


@pytest.mark.parametrize("mutation", ["unknown", "duplicate", "warmup", "reordered", "nonnumeric_cap", "extra"])
def test_invalid_cohort_contract_rejected(mutation: str) -> None:
    payload, _ = _cohort_inputs()
    if mutation == "unknown":
        payload["exclusions"][-1]["security_id"] = "unknown"
    elif mutation == "duplicate":
        payload["exclusions"].append(payload["exclusions"][-1])
    elif mutation == "warmup":
        payload["warmup_only_security_ids"] = ["sec:0001"]
    elif mutation == "reordered":
        payload["original_security_ids"].reverse()
    elif mutation == "nonnumeric_cap":
        payload["maximum_exclusion_bps"] = "800"
    else:
        payload["accounting_eligible"] = True
    with pytest.raises(ValidationError):
        SwingResearchCohort.model_validate_json(json.dumps(payload))


@pytest.mark.parametrize("change", ["reset_denominator", "drop_inherited", "parent_changed", "warmup_changed"])
def test_source_population_cannot_be_rebased(change: str) -> None:
    payload, parent = _cohort_inputs()
    if change == "reset_denominator":
        payload["original_security_ids"] = payload["original_security_ids"][27:]
        payload["inherited_excluded_security_ids"] = []
    elif change == "drop_inherited":
        payload["inherited_excluded_security_ids"] = []
    elif change == "parent_changed":
        parent["new_source"] = True
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    warmup = ("different",) if change == "warmup_changed" else ("warmup:only",)
    with pytest.raises(DataReadinessError, match="parent population"):
        cohort.assert_source_matches(parent, tuple(f"sec:{i:04}" for i in range(27, 631)), warmup)


def test_restriction_removes_all_rows_of_security_not_a_bad_outcome() -> None:
    payload, parent = _cohort_inputs()
    rows = pd.DataFrame({"security_id": [f"sec:{i:04}" for i in range(27, 631)] * 2})
    rows["return"] = [1.0] * 604 + [-1.0] * 604
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    with pytest.raises(DataReadinessError, match="exceed approved cap"):
        cohort.restrict_memberships(rows, parent)
    payload["maximum_exclusion_bps"] = 800
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    selected = cohort.restrict_memberships(rows, parent)
    assert len(selected) == 586 * 2
    assert selected["return"].sum() == 0
    assert set(selected.security_id) == set(cohort.retained_security_ids)
    assert len(rows) == 604 * 2


def test_loader_rejects_tampered_summary_and_cohort(tmp_path: Path) -> None:
    payload, _ = _cohort_inputs()
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    report = {"cohort": payload, "cohort_sha256": cohort.sha256(), "summary": cohort.summary(), "coverage": []}
    report["audit_sha256"] = json_sha256(report)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    assert load_swing_research_cohort(path) == cohort
    report["summary"]["accounting_eligible"] = True
    report["audit_sha256"] = json_sha256({key: value for key, value in report.items() if key != "audit_sha256"})
    path.write_text(json.dumps(report))
    with pytest.raises(DataReadinessError, match="identity or summary"):
        load_swing_research_cohort(path)
    report["summary"] = cohort.summary()
    report["cohort"]["maximum_exclusion_bps"] = 800
    report["audit_sha256"] = json_sha256({key: value for key, value in report.items() if key != "audit_sha256"})
    path.write_text(json.dumps(report))
    with pytest.raises(DataReadinessError, match="identity or summary"):
        load_swing_research_cohort(path)


def test_loader_binds_coverage_and_checks_source_files(tmp_path: Path) -> None:
    payload, _ = _cohort_inputs()
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    report = {"cohort": payload, "cohort_sha256": cohort.sha256(), "summary": cohort.summary(), "coverage": []}
    report["audit_sha256"] = json_sha256(report)
    path = tmp_path / "audit.json"
    path.write_text(json.dumps(report))
    (tmp_path / "parent.json").write_text("tampered")
    with pytest.raises(DataReadinessError, match="source hash differs"):
        load_swing_research_cohort(path, source_root=tmp_path)
    report["coverage"] = [{"removed": 0}]
    path.write_text(json.dumps(report))
    with pytest.raises(DataReadinessError, match="audit hash differs"):
        load_swing_research_cohort(path)
