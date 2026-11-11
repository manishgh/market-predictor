"""Retained completed distribution and pre-event existing recipient proof checks."""
from __future__ import annotations

import json
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.investment import action_relevance, completed_cash_distributions
from market_predictor.investment.action_relevance import ExistingRecipientStockAuthority, InvestmentActionRelevance, RelevanceRecord
from market_predictor.investment.completed_cash_distributions import (
    ReviewedCompletedCashDistribution,
    ReviewedCompletedCashDistributions,
    VerifiedCompletedCashDistribution,
    new_completed_cash_ids,
)
from market_predictor.investment.dividend_currencies import TaxCurrencyEvent, replay_tax_currency_event
from market_predictor.investment.sec_document_evidence import RetainedSecDocument, SecArchivedDocument, read_retained_sec_document
from market_predictor.investment.stock_identity import SecClassAnchor, StockIdentityProof
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_investment_dividend_currencies import POLICY, captured, pin, replace_tax
from tests.test_investment_dividend_currencies import fixture as tax_fixture
from tests.test_investment_label_materialization import reference
from tests.test_investment_remaining_outcomes import SPIN
from tests.test_investment_stock_identity import _real_class_proof

EX = date(2019, 9, 20)
RECORD = date(2019, 9, 23)
PAY = date(2019, 10, 31)


@pytest.mark.parametrize("poison", [None, "structured_date", "structured_payment", "structured_record", "gross_payment", "other_table"])
def test_completed_tax_uses_exact_payment_and_same_unique_brutto_row(tmp_path: Path, poison: str | None) -> None:
    _, item = tax_fixture(tmp_path)
    payment = int(datetime(2019, 10, 31, tzinfo=UTC).timestamp() * 1000)
    record = int(datetime(2019, 9, 23, tzinfo=UTC).timestamp() * 1000)

    def update(payload: dict[str, Any]) -> None:
        entry = payload["data"]["calculationDetails"]["entries"][0]
        entry.update(date=payment, paymentDate=payment, recordDate=record)
        if poison == "structured_date":
            entry["date"] += 86_400_000
        if poison == "structured_payment":
            entry["paymentDate"] += 86_400_000
        if poison == "structured_record":
            entry["recordDate"] += 86_400_000

    replace_tax(item, update)
    gross = Path(item["tax"]["gross_context"]["body"]["path"])
    payload = json.loads(gross.read_text())
    html = payload["data"]["html"].replace("<td>20.09.</td><td></td>", "<td>20.09.</td><td>31.10.</td>")
    if poison == "gross_payment":
        html = html.replace("31.10.", "30.10.")
    if poison == "other_table":
        html = html.replace("<th>Brutto</th>", "<th>Other</th>") + "<table><tr><th>Brutto</th></tr></table>"
    payload["data"]["html"] = html
    item["tax"]["gross_context"] = captured(gross, json.dumps(payload).encode())
    tax = TaxCurrencyEvent.model_validate(item["tax"])
    if poison:
        with pytest.raises(DataReadinessError, match="payment or record date"):
            replay_tax_currency_event(tmp_path, tax, EX, Decimal("1.383619"), Decimal(0), POLICY,
                payable_date=PAY, record_date=RECORD)
    else:
        amount, refs = replay_tax_currency_event(tmp_path, tax, EX, Decimal("1.383619"), Decimal(0), POLICY,
            payable_date=PAY, record_date=RECORD)
        assert amount == Decimal("1.383619")
        assert any("payment-date=2019-10-31" in ref.record_locator for ref in refs)


def fact_payload(tmp_path: Path) -> dict[str, Any]:
    _, tax = tax_fixture(tmp_path)
    tax["tax"]["lookup"] = tax["tax"]["source"]
    p = {"path": str(tmp_path / "evidence.json"), "sha256": "b" * 64}
    document = {"collection_manifest": p, "receipt_artifact": p, "filing_metadata": p,
        "index_receipt_artifact": p, "unit_id": "archived/2", "body_sha256": "a" * 64}
    declared = ("Board of Directors has approved a special dividend of $1.383619 per share to be distributed "
        "on October 31, 2019, to shareholders of record on September 23, 2019.")
    completed = "The special dividend was paid on Oct. 31 in the amount of $1.383619 per share."
    return {"action_id": "test-cash", "retained_record_sha256": "c" * 64, "security_id": "stable:TEST",
        "ticker": "TEST", "sec_cik": "0001000697", "cusip": "78462F103", "ex_date": EX.isoformat(),
        "record_date": RECORD.isoformat(), "payable_date": PAY.isoformat(), "gross_amount": "1.383619", "currency": "USD",
        "stock_class_proof": p, "tax": tax["tax"], "declaration": {"kind": "amount_record_and_payable",
            "statement": {"document": {**document, "unit_id": "declared/2"}, "quote": declared}},
        "completion": {"kind": "paid_date", "statement": {"document": document, "quote": completed},
            "upper_bound": PAY.isoformat()}}


@pytest.mark.parametrize("poison", [
    None, "event_context", "scheduled", "amount", "missing_quote", "wrong_declaration", "truncated_conditional", "truncated_negation",
])
def test_exact_issuer_quote_must_prove_actual_payment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, poison: str | None) -> None:
    fact = fact_payload(tmp_path)
    if poison == "event_context":
        fact["declaration"].update(kind="event_context", event_context="divestiture closing")
        fact["declaration"]["statement"]["quote"] = (
            "Board of Directors declares a $100 million special dividend in conjunction with divestiture closing")
        fact["completion"]["statement"]["quote"] = (
            "In conjunction with the divestiture closing, Company paid a $100 million special dividend to shareholders. "
            + fact["completion"]["statement"]["quote"])
    if poison == "scheduled":
        fact["completion"]["statement"]["quote"] = fact["completion"]["statement"]["quote"].replace("was paid", "will be paid")
    if poison == "amount":
        fact["completion"]["statement"]["quote"] = fact["completion"]["statement"]["quote"].replace("1.383619", "1.37")
    if poison == "wrong_declaration":
        fact["declaration"]["statement"]["quote"] = fact["declaration"]["statement"]["quote"].replace("September 23", "September 25")
    parsed = ReviewedCompletedCashDistribution.model_validate_json(json.dumps(fact))
    capture = pd.Timestamp("2026-10-01T10:00:00Z")
    path = tmp_path / "proof.zip"
    pins = {str(path): "c" * 64}

    def read(_root: Path, source: SecArchivedDocument, *_args: Any, **_kwargs: Any) -> RetainedSecDocument:
        declared = source.unit_id == "declared/2"
        quote = parsed.declaration.statement.quote if declared else parsed.completion.statement.quote
        if not declared:
            if poison == "truncated_conditional":
                quote = "It is unclear whether " + quote
            elif poison == "truncated_negation":
                quote = "The following assertion is false: " + quote
            elif poison == "missing_quote":
                quote = "unrelated"
        body = ("<p>" + quote + "</p>").encode()
        return RetainedSecDocument(body, path, source.unit_id, path, "index-member", "d" * 64, "EX-99.1",
            pd.Timestamp("2019-09-18T18:00:00Z" if declared else "2019-11-05T18:00:00Z"), capture, capture)

    monkeypatch.setattr(completed_cash_distributions, "read_retained_sec_document", read)
    if poison not in (None, "event_context"):
        with pytest.raises(DataReadinessError, match="issuer"):
            completed_cash_distributions._verified_issuer_cash(tmp_path, parsed, POLICY, pins, {}, lambda: None)
    else:
        refs = completed_cash_distributions._verified_issuer_cash(tmp_path, parsed, POLICY, pins, {}, lambda: None)
        assert len(refs) == 4 and all(ref.available_at == capture for ref in refs)
        verified = VerifiedCompletedCashDistribution(parsed, refs)
        assert verified.references("stable:TEST", "TEST", date(2019, 11, 1)) == refs
        assert not verified.references("stable:OTHER", "TEST", date(2019, 11, 1))
        assert not verified.references("stable:TEST", "TEST", PAY)


@pytest.mark.parametrize("same_quarter", [True, False])
def test_reported_completion_needs_full_slide_bullet_and_same_quarter_period(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, same_quarter: bool,
) -> None:
    fact = fact_payload(tmp_path)
    document = fact["completion"]["statement"]["document"]
    completion_quote = "$1.383619 per share special dividend distributed in the fourth quarter"
    period_quote = "For the three months ended December 31, 2019, the company reported the following results:"
    fact["completion"] = {"kind": "reported_period", "upper_bound": "2019-12-31",
        "statement": {"document": document, "quote": completion_quote},
        "reporting_period": {"document": {**document, "unit_id": "period/2"}, "quote": period_quote}}
    parsed = ReviewedCompletedCashDistribution.model_validate_json(json.dumps(fact))
    path = tmp_path / "proof.zip"
    capture = pd.Timestamp("2026-10-01T10:00:00Z")

    def read(_root: Path, source: SecArchivedDocument, *_args: Any, **_kwargs: Any) -> RetainedSecDocument:
        if source.unit_id == "declared/2":
            text, accepted = "<p>" + parsed.declaration.statement.quote + "</p>", "2019-09-18T18:00:00Z"
        elif source.unit_id == "period/2":
            quarter = "FOURTH" if same_quarter else "THIRD"
            text = f"<div>COMPANY REPORTS {quarter} QUARTER 2019 RESULTS</div><p>{period_quote}</p>"
            accepted = "2020-02-15T18:00:00Z"
        else:
            text = ("<div><img title='slide8'><div>Capital allocation \u2022 " + completion_quote
                + " \u2022 Other facts</div></div><p>Appendix Fourth Quarter 2019 Performance</p>")
            accepted = "2020-02-18T18:00:00Z"
        return RetainedSecDocument(text.encode(), path, source.unit_id, path, "index", "d" * 64, "EX-99.1",
            pd.Timestamp(accepted), capture, capture)

    monkeypatch.setattr(completed_cash_distributions, "read_retained_sec_document", read)
    if same_quarter:
        assert len(completed_cash_distributions._verified_issuer_cash(
            tmp_path, parsed, POLICY, {str(path): "c" * 64}, {}, lambda: None)) == 6
    else:
        with pytest.raises(DataReadinessError, match="exact closed reporting period"):
            completed_cash_distributions._verified_issuer_cash(tmp_path, parsed, POLICY, {str(path): "c" * 64}, {}, lambda: None)


def test_completed_facts_are_append_only_and_preserve_nested_source_pins(tmp_path: Path) -> None:
    original = ReviewedCompletedCashDistribution.model_validate_json(json.dumps(fact_payload(tmp_path)))
    prior = {original.action_id: original}
    other = original.model_copy(update={"action_id": "other"})
    assert new_completed_cash_ids(prior, {**prior, other.action_id: other}) == {"other"}
    changed = original.model_copy(update={"stock_class_proof": SourcePin(path=original.stock_class_proof.path, sha256="e" * 64)})
    with pytest.raises(DataReadinessError, match="prior exact facts"):
        new_completed_cash_ids(prior, {original.action_id: changed})
    with pytest.raises(ValueError, match="duplicate action or class/date"):
        ReviewedCompletedCashDistributions(schema_version="market_predictor.reviewed_completed_cash_distributions",
            scope="retrospective_post_distribution_relevance_only", facts=(original, other))
    assert completed_cash_distributions._issuer_name_key("DEVON ENERGY CORP/DE", "DE") == (
        completed_cash_distributions._issuer_name_key("Devon Energy Corporation"))
    assert completed_cash_distributions._issuer_name_key("DEVON ENERGY CORP/DE", "TX") != "devon energy corporation"
    assert completed_cash_distributions._issuer_name_key("DEVON ENERGY CORP/DE") != "devon energy corporation"
    assert completed_cash_distributions._issuer_name_key("Devon Energy/DE Advisors") != "devon energy advisors"


@pytest.mark.parametrize("poison", [
    None, "no_authority", "different_version", "one_query", "boundary", "late_class", "coverage", "when_issued", "exchange",
])
def test_existing_recipient_requires_exact_two_queries_and_raw_interval_class(monkeypatch: pytest.MonkeyPatch,
                                                                            tmp_path: Path, poison: str | None) -> None:
    pin_ref = SourcePin(path="class.json", sha256="a" * 64)
    proof = StockIdentityProof(security_id="stable:DAUGHTER", ticker="DAUGHTER", sec_cik="0001000697",
        security_title="Common Stock, par value $0.01 per share" if poison != "when_issued" else "Common Stock when issued",
        identity_authority=pin_ref, membership_authority=pin_ref, anchor=SecClassAnchor(
            collection_manifest=pin_ref, receipt_artifact=pin_ref, unit_id="unit", body_sha256="b" * 64,
            filing_metadata=pin_ref))
    subject = RelevanceRecord("spin_offs", SPIN, (reference("daughter.bin"), reference("daughter.transport")))
    parent_record = {**SPIN, "new_rate": .5} if poison == "different_version" else SPIN
    parent = RelevanceRecord("spin_offs", parent_record, (reference("parent.bin"), reference("parent.transport")))
    records = {("DAUGHTER", "spin"): subject}
    if poison != "one_query":
        records[("TEST", "spin")] = parent
    authority = None if poison == "no_authority" else ExistingRecipientStockAuthority(pin_ref, {("stable:DAUGHTER", "DAUGHTER"): proof})
    engine = InvestmentActionRelevance(tmp_path, records, None, None, reference("policy"), memory_guard=lambda: None,
        recipients=authority)
    sessions = (date(2024, 7, 18), date(2024, 7, 20) if poison == "boundary" else date(2024, 7, 19))
    accepted = datetime(2024, 7, 20 if poison == "late_class" else 10, tzinfo=UTC)
    checked = datetime(2026, 11, 11, tzinfo=UTC)

    def identity(*_args: Any, **kwargs: Any) -> Any:
        assert kwargs["sessions"] == sessions
        return SimpleNamespace(proven_sessions=sessions[:1] if poison == "coverage" else sessions, missing_reasons=(),
            competing_class_tickers=(), exchange="Unsupported Exchange" if poison == "exchange" else "NYSE",
            evidence=(reference("sec.zip"),),
            anchor_accepted_at_utc=accepted, verified_at_utc=checked)

    monkeypatch.setattr(action_relevance, "replay_stock_identity", identity)
    refs = engine.references("DAUGHTER", "spin_offs", "spin", *sessions, security_id="stable:DAUGHTER", sessions=sessions)
    if poison:
        assert not refs
    else:
        assert {ref.reference for ref in refs} == {
            "daughter.bin", "daughter.transport", "parent.bin", "parent.transport", "class.json", "sec.zip",
        }
        assert "existing_recipient_common_stock_holding_ends_before_distribution" in refs[0].record_locator
        assert refs[-2].available_at == checked
        assert not engine.references("DAUGHTER", "spin_offs", "spin", *sessions, security_id="stable:OTHER", sessions=sessions)


def test_stock_cover_optional_incorporation_requires_one_same_issuer_context(tmp_path: Path) -> None:
    import hashlib

    from market_predictor.universe.security_class_evidence import extract_security_class_and_jurisdiction
    from tests.test_security_class_evidence import _context, _document, _issuer, _row

    for index, (context, code, expected) in enumerate((("base", "DE", "DE"), ("other", "DE", None), ("base", "Delaware", None))):
        destination = tmp_path / str(index)
        destination.mkdir()
        incorporation = f'<ix:nonNumeric name="dei:EntityIncorporationStateCountryCode" contextRef="{context}">{code}</ix:nonNumeric>'
        document = _document(_context("base"), _issuer("base") + incorporation, _row("base", ticker="TEST"))
        path = destination / "class.html"
        body = document.encode("utf-8")
        path.write_bytes(body)
        _, found = extract_security_class_and_jurisdiction(path, hashlib.sha256(body).hexdigest(), "0001040971", "TEST")
        assert found == expected


def test_sec_exhibit_requires_exact_archived_index_and_keeps_primary_gate(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    proof = _real_class_proof(tmp_path, monkeypatch)
    folder = Path(proof.anchor.receipt_artifact.path).parent
    receipts = [(path, row) for path in folder.glob("*.parquet") for row in pd.read_parquet(path).to_dict("records")
        if row["sec_cik"] == proof.sec_cik and row["state"] == "archived"]
    document_path, document = next((path, row) for path, row in receipts if row["document_type"] == "EX-99.1")
    index_path = next(path for path, row in receipts if row["phase"] == "index" and row["body_sha256"] == document["index_body_sha256"])
    anchor = SecArchivedDocument(collection_manifest=proof.anchor.collection_manifest,
        receipt_artifact=SourcePin.model_validate(pin(document_path)),
        filing_metadata=proof.anchor.filing_metadata, index_receipt_artifact=SourcePin.model_validate(pin(index_path)),
        unit_id=document["unit_id"], body_sha256=document["body_sha256"])
    result = read_retained_sec_document(tmp_path, anchor, proof.sec_cik, {}, {}, primary_only=False, memory_guard=lambda: None)
    assert result.document_type == "EX-99.1" and result.body_member == document["member"]
    assert result.accepted_at <= result.index_captured_at <= result.captured_at
    with pytest.raises(DataReadinessError, match="document binding differs"):
        read_retained_sec_document(tmp_path, anchor, proof.sec_cik, {}, {}, primary_only=True, memory_guard=lambda: None)
