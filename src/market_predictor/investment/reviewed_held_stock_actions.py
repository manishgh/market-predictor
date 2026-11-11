"""Reviewed whole-window split and ordinary USD dividend terms for fixed holdings."""
from __future__ import annotations

import os
import re
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated, Any, Literal, Self

from bs4 import BeautifulSoup
from pydantic import Field, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.investment.completed_cash_distributions import _date_pattern, _issuer_name_key, _official_blocks
from market_predictor.investment.dividend_currencies import (
    TaxCurrencyEvent,
    _decimal,
    _isin,
    replay_stock_tax_class,
    replay_tax_currency_event,
    valid_cusip,
)
from market_predictor.investment.sec_document_evidence import SecArchivedDocument, pin_retained_source, read_retained_sec_document
from market_predictor.investment.stock_identity import StockIdentityProof, replay_stock_identity
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Identifier, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.reviewed_action_sources import verify_reviewed_action_sources

HELD_STOCK_ACTION_LOCATOR = "reviewed held-stock action"
REVIEWED_SPLIT_UNITS = 3.0
_PFIZER_HISTORY = "https://investors.pfizer.com/Investors/stock-dividend/dividend-split-history/default.aspx"
_COMMON_FIELDS = {"id", "symbol", "cusip", "ex_date", "record_date", "payable_date", "process_date"}


class _ReviewedStockAction(HoldingContract):
    action_id: Identifier
    retained_record_sha256: Sha256
    security_id: Identifier
    ticker: Identifier
    cusip: Identifier
    ex_date: date
    record_date: date
    payable_date: date
    stock_class_proof: SourcePin

    @model_validator(mode="after")
    def stable_class(self) -> Self:
        if not valid_cusip(self.cusip) or re.fullmatch(r"cik:[0-9]{10}", self.security_id) is None:
            raise ValueError("held-stock action requires exact stable issuer and CUSIP")
        return self


class ReviewedHoldingSplit(_ReviewedStockAction):
    kind: Literal["held_stock_split"]
    due_bill_redemption_date: date
    issuer_declaration: SecArchivedDocument
    declaration_quote: Annotated[Identifier, Field(max_length=1024)]

    @model_validator(mode="after")
    def complete_split_dates(self) -> Self:
        if (not self.record_date < self.payable_date < self.ex_date < self.due_bill_redemption_date
                or self.issuer_declaration.index_receipt_artifact is None
                or self.declaration_quote != " ".join(self.declaration_quote.split())):
            raise ValueError("reviewed stock-dividend split requires all dates and indexed issuer declaration")
        return self


class IssuerRegularCashHistory(HoldingContract):
    receipt: SourcePin
    request: SourcePin
    transport: SourcePin
    body: SourcePin


class ReviewedRegularCash(_ReviewedStockAction):
    kind: Literal["regular_cash"]
    gross_amount: Identifier
    issuer_history: IssuerRegularCashHistory
    tax: TaxCurrencyEvent

    @model_validator(mode="after")
    def regular_cash_terms(self) -> Self:
        if (not self.ex_date < self.record_date < self.payable_date or _decimal(self.gross_amount) <= 0
                or self.tax.isin != _isin(self.cusip) or self.tax.lookup is None
                or _decimal(self.tax.gross_amount) != _decimal(self.gross_amount)):
            raise ValueError("reviewed regular cash terms and stock tax source differ")
        return self


ReviewedHeldStockAction = ReviewedHoldingSplit | ReviewedRegularCash


class ReviewedHeldStockActions(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_held_stock_actions"]
    scope: Literal["retrospective_fixed_holding_terms_only"]
    facts: Annotated[tuple[Annotated[ReviewedHeldStockAction, Field(discriminator="kind")], ...],
                     Field(min_length=1, max_length=64)]

    @model_validator(mode="after")
    def distinct_terms(self) -> Self:
        if (len({fact.action_id for fact in self.facts}) != len(self.facts)
                or len({(fact.security_id, fact.ticker, fact.ex_date) for fact in self.facts}) != len(self.facts)):
            raise ValueError("reviewed held-stock actions duplicate provider ID or class/date")
        return self


@dataclass(frozen=True)
class VerifiedHeldStockAction:
    fact: ReviewedHeldStockAction
    value: float
    binding: EvidenceReference
    evidence: tuple[EvidenceReference, ...]

    def covers(self, security_id: str, ticker: str, first: date, last: date) -> bool:
        return held_stock_action_covers(self.fact, security_id, ticker, first, last)


def held_stock_action_covers(
    fact: ReviewedHeldStockAction, security_id: str, ticker: str, first: date, last: date,
) -> bool:
    if (security_id, ticker) != (fact.security_id, fact.ticker):
        return False
    if isinstance(fact, ReviewedHoldingSplit):
        return first < fact.record_date and last > fact.due_bill_redemption_date
    return first < fact.ex_date <= last


def read_reviewed_held_stock_actions(
    root: Path, pin: SourcePin | None, files: MutableMapping[str, str],
) -> dict[str, ReviewedHeldStockAction]:
    if pin is None:
        return {}
    path = pin_retained_source(root, pin, files)
    if path.stat().st_size > 1024**2:
        raise DataReadinessError("reviewed held-stock facts exceed bounded size")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed held-stock action facts")
    manifest = ReviewedHeldStockActions.model_validate_json(raw)
    declared = {os.path.normcase(str(inside(root, name))): digest for name, digest in files.items()}
    for fact in manifest.facts:
        sources = [fact.stock_class_proof]
        if isinstance(fact, ReviewedHoldingSplit):
            doc = fact.issuer_declaration
            sources.extend((doc.collection_manifest, doc.filing_metadata, doc.receipt_artifact))
            if doc.index_receipt_artifact is not None:
                sources.append(doc.index_receipt_artifact)
        else:
            history = fact.issuer_history
            sources.extend((history.body, history.receipt, history.request, history.transport))
            for captured in (fact.tax.source, fact.tax.gross_context, fact.tax.lookup):
                if captured is not None:
                    sources.extend(SourcePin(path=source.path, sha256=source.sha256)
                                   for source in (captured.body, captured.receipt))
        for source in sources:
            name = os.path.normcase(str(inside(root, source.path)))
            if declared.setdefault(name, source.sha256) != source.sha256:
                raise DataReadinessError("reviewed held-stock facts declare conflicting exact source pins")
    return {item.action_id: item for item in manifest.facts}


def _checked_provider_record(fact: ReviewedHeldStockAction, original: dict[str, Any]) -> float:
    expected: dict[str, Any] = {
        "id": fact.action_id, "symbol": fact.ticker, "cusip": fact.cusip,
        "ex_date": fact.ex_date.isoformat(), "record_date": fact.record_date.isoformat(),
        "payable_date": fact.payable_date.isoformat(),
    }
    if isinstance(fact, ReviewedHoldingSplit):
        fields = _COMMON_FIELDS | {"old_rate", "new_rate", "due_bill_redemption_date"}
        expected.update(process_date=fact.ex_date.isoformat(),
                        due_bill_redemption_date=fact.due_bill_redemption_date.isoformat())
        valid_terms = (type(original.get("old_rate")) in (int, float) and original["old_rate"] == 1
                       and type(original.get("new_rate")) in (int, float) and original["new_rate"] == REVIEWED_SPLIT_UNITS)
        value = REVIEWED_SPLIT_UNITS
    else:
        fields = _COMMON_FIELDS | {"rate", "foreign", "special"}
        expected["process_date"] = fact.payable_date.isoformat()
        amount = _decimal(fact.gross_amount)
        valid_terms = (original.get("foreign") is True and original.get("special") is False
                       and type(original.get("rate")) in (int, float) and _decimal(original["rate"]) == amount)
        value = float(amount)
    if (set(original) != fields or not valid_terms or any(original.get(k) != v for k, v in expected.items())
            or json_sha256(original) != fact.retained_record_sha256):
        raise DataReadinessError("reviewed held-stock action differs from its exact retained provider terms")
    return value


def _reference(pin: SourcePin, policy: str, locator: str, captured: datetime) -> EvidenceReference:
    return EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256, record_locator=locator,
                             interpretation_policy_sha256=policy, retrieved_at=captured, available_at=captured)


def _split_issuer(
    root: Path, fact: ReviewedHoldingSplit, policy: str, files: MutableMapping[str, str], cache: dict[str, Any],
    *, memory_guard: Callable[[], None],
) -> tuple[EvidenceReference, ...]:
    source = fact.issuer_declaration
    document = read_retained_sec_document(root, source, fact.security_id.removeprefix("cik:"), files, cache,
                                         primary_only=False, memory_guard=memory_guard)
    _, blocks = _official_blocks(document)
    quote = fact.declaration_quote
    record = _date_pattern(fact.record_date, year_required=True)
    payable = _date_pattern(fact.payable_date, year_required=True)
    ex = _date_pattern(fact.ex_date, year_required=True)
    terms = (rf"Each stockholder of record on {record} will receive a dividend of two additional shares of common stock "
             rf"for each then-held share, to be distributed after close of trading on {payable}\. Trading will begin "
             rf"on a stock split-adjusted basis on {ex}\.")
    if (document.document_type != "EX-99.1" or document.accepted_at.date() >= fact.record_date or quote not in blocks
            or re.search(r"\bBoard of Directors has approved and declared a three-for-one split of [^.!?]+ common stock "
                         r"in the form of a stock dividend\b", quote, re.I) is None
            or re.search(terms + "$", quote, re.I) is None):
        raise DataReadinessError("issuer split declaration lacks exact three-for-one stock-dividend terms")
    body = SourcePin(path=document.body_path.relative_to(root).as_posix(), sha256=files[str(document.body_path)])
    index = SourcePin(path=document.index_path.relative_to(root).as_posix(), sha256=files[str(document.index_path)])
    return (
        _reference(body, policy, f"issuer split declaration;action_id={fact.action_id};member={document.body_member};"
                                f"body_sha256={source.body_sha256}", document.captured_at.to_pydatetime()),
        _reference(index, policy, f"issuer split declaration index;action_id={fact.action_id};member={document.index_member};"
                                 f"body_sha256={document.index_body_sha256}", document.index_captured_at.to_pydatetime()),
    )


def _regular_cash_issuer(
    root: Path, fact: ReviewedRegularCash, manifest: SourcePin, policy: str, files: MutableMapping[str, str],
    *, memory_guard: Callable[[], None],
) -> tuple[EvidenceReference, ...]:
    history = fact.issuer_history
    sources = (history.receipt, history.request, history.transport, history.body)
    if fact.security_id != "cik:0000078003" or len({item.path for item in sources}) != 4:
        raise DataReadinessError("reviewed regular cash history lacks supported exact issuer binding")
    pins = verify_reviewed_action_sources(root, manifest, {item.path: item.sha256 for item in sources},
                                         (history.receipt,), memory_guard=memory_guard)
    receipt_path = inside(root, history.receipt.path)
    receipt = parse_strict_json_object(receipt_path.read_bytes(), label="reviewed cash issuer receipt")
    body_path = inside(root, history.body.path)
    if (receipt.get("url") != _PFIZER_HISTORY
            or inside(receipt_path.parent, str(receipt.get("body_path"))) != body_path
            or receipt_path.parent / "_request.json" != inside(root, history.request.path)
            or receipt_path.parent / "transport.json" != inside(root, history.transport.path)):
        raise DataReadinessError("reviewed cash issuer history receipt does not bind its exact sources")
    transport = parse_strict_json_object(inside(root, history.transport.path).read_bytes(), label="reviewed cash transport")
    raw_clocks = receipt.get("started_at_utc"), transport.get("retrieved_at_utc"), receipt.get("completed_at_utc")
    if any(not isinstance(clock, str) for clock in raw_clocks):
        raise DataReadinessError("reviewed cash issuer source is missing a capture clock")
    start, captured, end = (datetime.fromisoformat(str(clock)) for clock in raw_clocks)
    if (any(clock.utcoffset() is None for clock in (start, captured, end))
            or not start <= captured <= end <= datetime.now(UTC) or captured.date() <= fact.payable_date):
        raise DataReadinessError("reviewed cash issuer capture chronology differs")
    soup = BeautifulSoup(body_path.read_bytes(), "html.parser")
    ex = fact.ex_date.strftime("%m/%d/%Y")
    expected = [ex, fact.record_date.strftime("%m/%d/%Y"), fact.payable_date.strftime("%m/%d/%Y"),
                fact.gross_amount, "Regular Cash"]
    headings = ["Ex Date:", "Record Date:", "Payable Date:", "Amount:", "Type"]
    matching = [cells for row in soup.find_all("tr")
                if (cells := row.find_all("td", recursive=False)) and cells[0].get_text(strip=True) == ex]
    if (len(matching) != 1 or [cell.get("data-heading") for cell in matching[0]] != headings
            or [cell.get_text(strip=True) for cell in matching[0]] != expected):
        raise DataReadinessError("reviewed cash issuer history lacks one exact regular cash schedule")
    for path, digest in pins.items():
        pin_retained_source(root, SourcePin(path=path, sha256=digest), files)
    locator = f"issuer regular cash schedule;action_id={fact.action_id};ex_date={fact.ex_date};gross_amount={fact.gross_amount}"
    return tuple(_reference(item, policy, locator, captured) for item in sources)


def verify_held_stock_action(
    root: Path, fact: ReviewedHeldStockAction, original: dict[str, Any], manifest: SourcePin, policy: str,
    files: MutableMapping[str, str], cache: dict[str, Any], *, memory_guard: Callable[[], None],
) -> VerifiedHeldStockAction:
    value = _checked_provider_record(fact, original)
    proof_path = pin_retained_source(root, fact.stock_class_proof, files)
    if proof_path.stat().st_size > 1024**2:
        raise DataReadinessError("reviewed held-stock class proof exceeds size bound")
    raw = proof_path.read_bytes()
    parse_strict_json_object(raw, label="reviewed held-stock class proof")
    proof = StockIdentityProof.model_validate_json(raw)
    if ((proof.security_id, proof.ticker, proof.sec_cik) !=
            (fact.security_id, fact.ticker, fact.security_id.removeprefix("cik:"))
            or re.match(r"^common (?:stock|shares)\b", proof.security_title, re.I) is None):
        raise DataReadinessError("reviewed held-stock action differs from exact common share class")
    sessions: tuple[date, ...] = (fact.ex_date, fact.record_date)
    if isinstance(fact, ReviewedHoldingSplit):
        sessions = (fact.record_date, fact.payable_date, fact.ex_date, fact.due_bill_redemption_date)
    identity = replay_stock_identity(root, proof, sessions=sessions, source_files=files,
                                     interpretation_sha256=policy, cache=cache)
    if (identity.proven_sessions != sessions or identity.missing_reasons or identity.competing_class_tickers
            or not identity.evidence or identity.anchor_accepted_at_utc.date() >= sessions[0]):
        raise DataReadinessError("reviewed held-stock action lacks unique dated common share ownership")
    if isinstance(fact, ReviewedHoldingSplit):
        source_refs = _split_issuer(root, fact, policy, files, cache, memory_guard=memory_guard)
    else:
        issuer_refs = _regular_cash_issuer(root, fact, manifest, policy, files, memory_guard=memory_guard)

        def tax_class(calculation: Mapping[str, Any]) -> tuple[EvidenceReference, ...]:
            refs, observed = replay_stock_tax_class(calculation, fact.tax.lookup, root, policy,
                                                    issuer_name=identity.issuer_name, security_title=identity.security_title)
            issuer = _issuer_name_key(identity.issuer_name)
            if not issuer or _issuer_name_key(observed["tax_institution"]) != issuer:
                raise DataReadinessError("reviewed regular cash tax issuer differs from dated share class")
            return refs

        _, tax_refs = replay_tax_currency_event(root, fact.tax, fact.ex_date, _decimal(fact.gross_amount), Decimal(0), policy,
            stock_tax_validation=tax_class, payable_date=fact.payable_date, record_date=fact.record_date)
        source_refs = (*issuer_refs, *tax_refs)
    locator = (f"{HELD_STOCK_ACTION_LOCATOR};kind={fact.kind};action_id={fact.action_id};security_id={fact.security_id};"
               f"ticker={fact.ticker};retained_record_sha256={fact.retained_record_sha256}")
    binding = _reference(manifest, policy, locator, datetime.now(UTC))
    proof_ref = _reference(fact.stock_class_proof, policy, f"reviewed held-stock class;action_id={fact.action_id}",
                           identity.verified_at_utc)
    evidence = (binding, proof_ref, *identity.evidence, *source_refs)
    for ref in evidence:
        pin_retained_source(root, SourcePin(path=ref.reference, sha256=ref.artifact_sha256), files)
    return VerifiedHeldStockAction(fact, value, binding, evidence)
