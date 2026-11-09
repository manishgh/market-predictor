"""Observed entitlement and split semantics without simulated payment."""
import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.corporate_actions import InvestmentCorporateActions
from market_predictor.investment.dividend_currencies import CurrencyCorroboration, IssuerDividendAmountCorrection
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.action_windows import ActionWindow
from market_predictor.swing.datasets.corrected_outcome_admission import reconcile_actions
from tests.test_investment_dividend_currencies import captured, special_cash_fixture, workbook
from tests.test_investment_dividend_currencies import fixture as currency_fixture
from tests.test_investment_dividend_currencies import model as currency_model


def build(tmp_path: Path, records: dict[str, list[dict[str, Any]]], *, refresh_rate: float | None = None,
          symbol: str = "TEST", corroborations: dict[str, CurrencyCorroboration] | None = None,
          amount_corrections: dict[str, IssuerDividendAmountCorrection] | None = None) -> InvestmentCorporateActions:
    def write(name: str, data: dict[str, Any]) -> str:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
        return file_sha256(path)
    body_name = f"archive/tickers/{symbol}/attempt/000.bin"
    metadata_name = f"archive/tickers/{symbol}/attempt/000.json"
    receipt_name = f"archive/tickers/{symbol}/attempt/receipt.json"
    body_hash = write(body_name, {"corporate_actions": records})
    metadata_hash = write(metadata_name, {"sha256": body_hash, "body_length": (tmp_path / body_name).stat().st_size,
        "retrieved_at_utc": "2026-11-07T00:00:00Z"})
    request_hash = "a" * 64
    receipt_hash = write(receipt_name, {"request_sha256": request_hash, "state": "acquired", "ticker": symbol,
        "pages": [{"body_path": "000.bin", "metadata_path": "000.json", "body_sha256": body_hash,
            "metadata_sha256": metadata_hash}]})
    request_digest = write("archive/_request.json", {"request_sha256": request_hash,
        "policy": {"process_start": "2019-01-01", "process_end": "2026-09-11"}})
    policy_hash = write("policy.json", {"nominal_unpaid_claim_valuation": True})
    source_files = {body_name: body_hash, metadata_name: metadata_hash, receipt_name: receipt_hash,
        "archive/_request.json": request_digest}
    projected = {symbol: records}
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
    return InvestmentCorporateActions(tmp_path, evidence, reconcile_actions(evidence, {symbol}), (), policy,
        date(2024, 5, 28), currency_evidence=refreshed, currency_corroborations=corroborations,
        dividend_amount_corrections=amount_corrections)


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



def test_independent_currency_sources_reach_unpaid_holding_terms(tmp_path: Path) -> None:
    original, item = currency_fixture(tmp_path, bridge=True)
    currency = currency_model(item)
    actions = build(tmp_path, {"cash_dividends": [original]}, symbol="SPY", corroborations={original["id"]: currency})
    sessions = (date(2019, 9, 19), date(2019, 9, 20), date(2019, 9, 23))
    result = actions.validate_sources("benchmark:SPY", "SPY", sessions, "owned")
    assert not result.missing_reasons
    assert len(result.events) == 1
    leg = result.events[0].legs[0]
    assert leg.kind == "unpaid_proceeds" and leg.units_per_owned_unit == original["rate"]
    assert "currency" not in original
    assert set(ref.reference for ref in currency.evidence).issubset(actions.source_files)
    assert any(ref.available_at == datetime(2026, 11, 8, 20, 5, 42, tzinfo=UTC) for ref in result.events[0].evidence)
    Path(item["tax"]["source"]["body"]["path"]).write_text("{}")
    with pytest.raises(DataReadinessError, match="source changed"):
        actions.recheck()


def test_independent_currency_cannot_contradict_explicit_provider_currency(tmp_path: Path) -> None:
    original, item = currency_fixture(tmp_path)
    original["currency"] = "EUR"
    with pytest.raises(DataReadinessError, match="contradicts provider currency"):
        build(tmp_path, {"cash_dividends": [original]}, symbol="SPY", corroborations={original["id"]: currency_model(item)})


def test_unused_independent_currency_action_cannot_be_silently_ignored(tmp_path: Path) -> None:
    original, item = currency_fixture(tmp_path)
    with pytest.raises(DataReadinessError, match="unused or unsupported"):
        build(tmp_path, {}, symbol="SPY", corroborations={original["id"]: currency_model(item)})



def corrected_dividend_fixture(tmp_path: Path) -> tuple[dict[str, Any], CurrencyCorroboration, IssuerDividendAmountCorrection]:
    original, item = currency_fixture(tmp_path, bridge=True)
    path = tmp_path / "corrected-issuer.xlsx"
    source = captured(path, workbook(path, rate="1.383620"), authority="issuer")
    item["issuer_rate"] = {"source": source, "row": 2}
    item["tax_resolved_value_difference"] = "-0.000620"
    correction = IssuerDividendAmountCorrection.model_validate_json(json.dumps({
        "method": "issuer_gross_amount_correction", "action_id": original["id"],
        "original_record_sha256": json_sha256(original), "symbol": "SPY", "ex_date": original["ex_date"],
        "cusip": "78462F103", "isin": "US78462F1030", "original_rate": "1.383619",
        "corrected_gross_amount": "1.383620", "issuer_rate": {"source": source, "row": 2}}))
    return original, currency_model(item), correction


def test_issuer_correction_changes_only_evidenced_entitlement_amount(tmp_path: Path) -> None:
    original, currency, correction = corrected_dividend_fixture(tmp_path)
    original_hash = json_sha256(original)
    actions = build(tmp_path, {"cash_dividends": [original]}, symbol="SPY",
        corroborations={original["id"]: currency}, amount_corrections={original["id"]: correction})
    result = actions.validate_sources("benchmark:SPY", "SPY", (date(2019, 9, 19), date(2019, 9, 20)), "owned")
    assert not result.missing_reasons
    assert result.events[0].legs[0].units_per_owned_unit == 1.383620
    assert result.events[0].legs[0].kind == "unpaid_proceeds"
    assert original_hash == json_sha256(original) and original["rate"] == 1.383619
    assert correction.issuer_rate.source.body.path in actions.source_files
    Path(correction.issuer_rate.source.body.path).write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="source changed"):
        actions.recheck()


def test_amount_correction_does_not_supply_missing_currency(tmp_path: Path) -> None:
    original, _, correction = corrected_dividend_fixture(tmp_path)
    actions = build(tmp_path, {"cash_dividends": [original]}, symbol="SPY", amount_corrections={original["id"]: correction})
    result = actions.validate_sources("benchmark:SPY", "SPY", (date(2019, 9, 19), date(2019, 9, 20)), "owned")
    assert any("dividend_usd_currency_unproven" in reason for reason in result.missing_reasons)
    assert not result.events


def test_unused_amount_correction_refuses(tmp_path: Path) -> None:
    original, _, correction = corrected_dividend_fixture(tmp_path)
    with pytest.raises(DataReadinessError, match="unused or unsupported"):
        build(tmp_path, {}, symbol="SPY", amount_corrections={original["id"]: correction})


def test_reviewed_special_cash_is_unpaid_with_ex_date_ownership_cutoff(tmp_path: Path) -> None:
    original, item = special_cash_fixture(tmp_path)
    saved = dict(original)
    actions = build(tmp_path, {"cash_dividends": [original]}, symbol="SPY",
        corroborations={original["id"]: currency_model(item)})
    sessions = (date(2019, 9, 19), date(2019, 9, 20), date(2019, 9, 23), date(2019, 10, 3))
    result = actions.validate_sources("benchmark:SPY", "SPY", sessions, "owned")
    assert not result.missing_reasons and len(result.events) == 1
    assert result.events[0].legs[0].kind == "unpaid_proceeds"
    assert result.events[0].legs[0].units_per_owned_unit == original["rate"]
    assert all(mark.value_per_unit == 1 for mark in result.non_share_marks)
    assert original == saved
    for holding in (sessions[1:], sessions[2:]):
        entered_ex_or_later = actions.validate_sources("benchmark:SPY", "SPY", holding, "new")
        assert not entered_ex_or_later.events and not entered_ex_or_later.missing_reasons
    # The shared swing index is unchanged and remains conservative.
    assert actions.actions["SPY"][0].first_relevant_date is None


@pytest.mark.parametrize("reason", ["invalid_action_participant", "unsupported_action_participant_field",
    "conflicting_action_record_identity", "invalid_action_economic_date"])
def test_special_cash_override_preserves_reconciliation_failures(tmp_path: Path, reason: str) -> None:
    original, item = special_cash_fixture(tmp_path)
    actions = build(tmp_path, {"cash_dividends": [original]}, symbol="SPY",
        corroborations={original["id"]: currency_model(item)})
    actions.actions["SPY"] = (*actions.actions["SPY"], ActionWindow(original["id"], "cash_dividends", None, None, reason))
    result = actions.validate_sources("benchmark:SPY", "SPY", (date(2019, 9, 19), date(2019, 9, 20)), "owned")
    assert any(reason in missing for missing in result.missing_reasons)


def test_special_cash_with_simultaneous_split_still_requires_distribution_basis(tmp_path: Path) -> None:
    original, item = special_cash_fixture(tmp_path)
    actions = build(tmp_path, {"cash_dividends": [original], "forward_splits": [{
        "id": "split", "symbol": "SPY", "new_rate": 2, "old_rate": 1, "ex_date": original["ex_date"]}]},
        symbol="SPY", corroborations={original["id"]: currency_model(item)})
    result = actions.validate_sources("benchmark:SPY", "SPY", (date(2019, 9, 19), date(2019, 9, 20)), "owned")
    assert any("simultaneous_split_distribution_basis_unproven" in reason for reason in result.missing_reasons)
