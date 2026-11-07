"""Observed entitlement and split semantics without simulated payment."""
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.corporate_actions import InvestmentCorporateActions
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.corrected_outcome_admission import reconcile_actions


def build(tmp_path: Path, records: dict[str, list[dict[str, Any]]], *, refresh_rate: float | None = None) -> InvestmentCorporateActions:
    def write(name: str, data: dict[str, Any]) -> str:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return file_sha256(path)
    body_name = "archive/tickers/TEST/attempt/000.bin"
    metadata_name = "archive/tickers/TEST/attempt/000.json"
    receipt_name = "archive/tickers/TEST/attempt/receipt.json"
    body_hash = write(body_name, {"corporate_actions": records})
    metadata_hash = write(metadata_name, {"sha256": body_hash, "body_length": (tmp_path / body_name).stat().st_size,
        "retrieved_at_utc": "2026-11-07T00:00:00Z"})
    request_hash = "a" * 64
    receipt_hash = write(receipt_name, {"request_sha256": request_hash, "state": "acquired", "ticker": "TEST",
        "pages": [{"body_path": "000.bin", "metadata_path": "000.json", "body_sha256": body_hash,
            "metadata_sha256": metadata_hash}]})
    request_digest = write("archive/_request.json", {"request_sha256": request_hash,
        "policy": {"process_start": "2019-01-01", "process_end": "2026-09-11"}})
    policy_hash = write("policy.json", {"nominal_unpaid_claim_valuation": True})
    source_files = {body_name: body_hash, metadata_name: metadata_hash, receipt_name: receipt_hash,
        "archive/_request.json": request_digest}
    projected = {"TEST": records}
    projection = json_sha256({"records": projected, "unavailable": [], "source_files": source_files, "replay": {}})
    evidence = CorporateActionEvidence("b" * 64, request_hash, projected, (), source_files, {}, projection)
    policy = EvidenceReference(reference="policy.json", artifact_sha256=policy_hash, record_locator="approved nominal unpaid claims",
        interpretation_policy_sha256="c" * 64, retrieved_at=datetime(2026, 11, 7, tzinfo=UTC),
        available_at=datetime(2026, 11, 7, tzinfo=UTC))
    refreshed: dict[str, tuple[dict[str, Any], EvidenceReference]] = {}
    if refresh_rate is not None:
        original = records["cash_dividends"][0]
        record = {**original, "currency": "USD", "rate": refresh_rate}
        refresh_hash = write("refreshed.json", {"corporate_actions": {"cash_dividends": [record]}})
        ref = EvidenceReference(reference="refreshed.json", artifact_sha256=refresh_hash,
            record_locator="exact refreshed USD provider record", interpretation_policy_sha256="c" * 64,
            retrieved_at=datetime(2026, 11, 7, tzinfo=UTC), available_at=datetime(2026, 11, 7, tzinfo=UTC))
        refreshed[original["id"]] = ({**record, "original_record_sha256": json_sha256(original)}, ref)
    return InvestmentCorporateActions(tmp_path, evidence, reconcile_actions(evidence, {"TEST"}), (), policy,
        date(2024, 5, 28), currency_evidence=refreshed)


def dividend(identity: str = "div", ex: str = "2024-05-23") -> dict[str, Any]:
    return {"id": identity, "symbol": "TEST", "rate": 2.0, "currency": "USD", "special": False,
        "foreign": False, "ex_date": ex, "payable_date": "2024-05-24"}


SESSIONS = (date(2024, 5, 22), date(2024, 5, 23), date(2024, 5, 24), date(2024, 5, 28))


def test_dividend_remains_nominal_unpaid_claim_after_payable_date(tmp_path: Path) -> None:
    action = build(tmp_path, {"cash_dividends": [dividend()]})
    result = action.validate_sources("issuer:test", "TEST", SESSIONS, "owned")
    assert not result.missing_reasons
    assert [event.kind for event in result.events] == ["corporate_action"]
    assert len(result.non_share_marks) == 3
    assert result.events[0].legs[0].kind == "unpaid_proceeds"
    assert all(mark.value_per_unit == 1.0 for mark in result.non_share_marks)
    assert all(ref.available_at == datetime(2026, 11, 7, tzinfo=UTC) for ref in result.evidence)
    entered_ex = action.validate_sources("issuer:test", "TEST", SESSIONS[1:], "owned")
    assert not entered_ex.events and not entered_ex.missing_reasons


def test_split_changes_share_marks_and_preserves_earlier_claim(tmp_path: Path) -> None:
    action = build(tmp_path, {"cash_dividends": [dividend(ex="2024-05-23")], "forward_splits": [
        {"id": "split", "symbol": "TEST", "new_rate": 2, "old_rate": 1, "ex_date": "2024-05-24"}]})
    result = action.validate_sources("issuer:test", "TEST", SESSIONS, "owned")
    assert not result.missing_reasons
    assert result.share_position_ids[:2] == ("owned", "owned")
    assert result.share_position_ids[2] != "owned"
    assert {mark.position_id for mark in result.non_share_marks} == {"owned:action:div:unpaid"}
    assert result.events[1].legs[0].units_per_owned_unit == 2.0


@pytest.mark.parametrize("change", [{"currency": "EUR"}, {"currency": None}, {"special": True}, {"cancelled": True}])
def test_unsupported_currency_special_or_cancelled_stays_unresolved(tmp_path: Path, change: dict[str, Any]) -> None:
    record = {**dividend(), **change}
    result = build(tmp_path, {"cash_dividends": [record]}).validate_sources("issuer:test", "TEST", SESSIONS, "owned")
    assert result.missing_reasons


def test_simultaneous_split_and_dividend_requires_basis(tmp_path: Path) -> None:
    result = build(tmp_path, {"cash_dividends": [dividend()], "forward_splits": [
        {"id": "split", "symbol": "TEST", "new_rate": 2, "old_rate": 1, "ex_date": "2024-05-23"}]}).validate_sources(
            "issuer:test", "TEST", SESSIONS, "owned")
    assert any("simultaneous_split" in reason for reason in result.missing_reasons)


def test_action_coverage_cannot_extend_numeric_boundary(tmp_path: Path) -> None:
    result = build(tmp_path, {}).validate_sources("issuer:test", "TEST", (date(2024, 5, 29),), "owned")
    assert "corporate_action_query_interval_unproven" in result.missing_reasons


def test_explicit_usd_refresh_binds_original_record_and_capture(tmp_path: Path) -> None:
    record = dividend()
    del record["currency"]
    action = build(tmp_path, {"cash_dividends": [record]}, refresh_rate=2.0)
    result = action.validate_sources("issuer:test", "TEST", SESSIONS, "owned")
    assert not result.missing_reasons
    assert any(ref.reference == "refreshed.json" for ref in result.events[0].evidence)
    assert "refreshed.json" in action.source_files


def test_currency_refresh_cannot_change_amount(tmp_path: Path) -> None:
    record = dividend()
    del record["currency"]
    result = build(tmp_path, {"cash_dividends": [record]}, refresh_rate=3.0).validate_sources(
        "issuer:test", "TEST", SESSIONS, "owned")
    assert result.missing_reasons
