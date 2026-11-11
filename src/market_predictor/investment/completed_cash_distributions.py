"""Exact completed special distributions used only to rule out later lot entitlement."""
from __future__ import annotations

import calendar
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
from market_predictor.investment.dividend_currencies import (
    CurrencyArtifactPin,
    TaxCurrencyEvent,
    _decimal,
    _isin,
    replay_stock_tax_class,
    replay_tax_currency_event,
    valid_cusip,
)
from market_predictor.investment.sec_document_evidence import (
    RetainedSecDocument,
    SecArchivedDocument,
    pin_retained_source,
    read_retained_sec_document,
)
from market_predictor.investment.stock_identity import StockIdentityProof, replay_stock_identity
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Identifier, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin

_CASH_FIELDS = {"id", "symbol", "cusip", "ex_date", "record_date", "payable_date", "process_date", "rate", "foreign", "special"}
COMPLETED_CASH_RELEVANCE = "fresh_holding_after_verified_completed_distribution"
_US_JURISDICTIONS = frozenset("AL AK AZ AR CA CO CT DE DC FL GA HI ID IL IN IA KS KY LA ME MD MA MI MN MS MO MT NE NV "
    "NH NJ NM NY NC ND OH OK OR PA RI SC SD TN TX UT VT VA WA WV WI WY".split())


class SecCashStatement(HoldingContract):
    document: SecArchivedDocument
    quote: Annotated[Identifier, Field(min_length=30, max_length=1024)]

    @model_validator(mode="after")
    def require_exact_index(self) -> Self:
        if self.document.index_receipt_artifact is None or self.quote != " ".join(self.quote.split()):
            raise ValueError("issuer cash statements require a pinned index and exact normalized passage")
        return self


class CashDeclaration(HoldingContract):
    kind: Literal["amount_record_and_payable", "event_context"]
    statement: SecCashStatement
    event_context: Annotated[Identifier, Field(min_length=4, max_length=80)] | None = None

    @model_validator(mode="after")
    def event_context_required(self) -> Self:
        if (self.kind == "event_context") != (self.event_context is not None):
            raise ValueError("cash declaration context differs from its typed source")
        return self


class CashCompletion(HoldingContract):
    kind: Literal["paid_date", "reported_period"]
    statement: SecCashStatement
    upper_bound: date
    reporting_period: SecCashStatement | None = None

    @model_validator(mode="after")
    def period_required(self) -> Self:
        if (self.kind == "reported_period") != (self.reporting_period is not None):
            raise ValueError("cash completion period differs from its typed source")
        return self


class ReviewedCompletedCashDistribution(HoldingContract):
    action_id: Identifier
    retained_record_sha256: Sha256
    security_id: Identifier
    ticker: Identifier
    sec_cik: Annotated[str, Field(pattern=r"^[0-9]{10}$")]
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{8}[0-9]$")]
    ex_date: date
    record_date: date
    payable_date: date
    gross_amount: Identifier
    currency: Literal["USD"]
    stock_class_proof: SourcePin
    tax: TaxCurrencyEvent
    declaration: CashDeclaration
    completion: CashCompletion

    @model_validator(mode="after")
    def completed_terms(self) -> Self:
        if (not self.ex_date <= self.record_date <= self.payable_date <= self.completion.upper_bound
                or (self.completion.kind == "paid_date" and self.completion.upper_bound != self.payable_date)
                or not valid_cusip(self.cusip) or _decimal(self.gross_amount) <= 0 or self.tax.lookup is None
                or self.tax.isin != _isin(self.cusip) or _decimal(self.tax.gross_amount) != _decimal(self.gross_amount)):
            raise ValueError("completed cash dates, gross amount or class tax sources differ")
        return self


class ReviewedCompletedCashDistributions(HoldingContract):
    schema_version: Literal["market_predictor.reviewed_completed_cash_distributions"]
    scope: Literal["retrospective_post_distribution_relevance_only"]
    facts: Annotated[tuple[ReviewedCompletedCashDistribution, ...], Field(min_length=1, max_length=256)]

    @model_validator(mode="after")
    def unique_facts(self) -> Self:
        if (len({item.action_id for item in self.facts}) != len(self.facts)
                or len({(item.security_id, item.ticker, item.ex_date) for item in self.facts}) != len(self.facts)):
            raise ValueError("completed distributions contain duplicate action or class/date facts")
        return self


@dataclass(frozen=True)
class VerifiedCompletedCashDistribution:
    fact: ReviewedCompletedCashDistribution
    evidence: tuple[EvidenceReference, ...]

    def references(self, security_id: str, ticker: str, first: date) -> tuple[EvidenceReference, ...]:
        if (security_id, ticker) != (self.fact.security_id, self.fact.ticker) or first <= self.fact.completion.upper_bound:
            return ()
        return self.evidence


def _declared_pins(fact: ReviewedCompletedCashDistribution) -> list[SourcePin | CurrencyArtifactPin]:
    statements = [fact.declaration.statement, fact.completion.statement]
    if fact.completion.reporting_period is not None:
        statements.append(fact.completion.reporting_period)
    result: list[SourcePin | CurrencyArtifactPin] = [fact.stock_class_proof]
    for statement in statements:
        document = statement.document
        result.extend((document.collection_manifest, document.receipt_artifact, document.filing_metadata))
        if document.index_receipt_artifact is not None:
            result.append(document.index_receipt_artifact)
    for source in (fact.tax.source, fact.tax.gross_context, fact.tax.lookup):
        if source is not None:
            result.extend((source.body, source.receipt))
    return result


def read_completed_cash_facts(
    root: Path, pin: SourcePin | None, files: MutableMapping[str, str],
) -> dict[str, ReviewedCompletedCashDistribution]:
    if pin is None:
        return {}
    path = pin_retained_source(root, pin, files)
    if path.stat().st_size > 2 * 1024**2:
        raise DataReadinessError("completed cash facts exceed bounded size")
    raw = path.read_bytes()
    parse_strict_json_object(raw, label="reviewed completed cash facts")
    manifest = ReviewedCompletedCashDistributions.model_validate_json(raw)
    declared = {str(path): pin.sha256}
    for fact in manifest.facts:
        for source in _declared_pins(fact):
            name = str(inside(root, source.path))
            if (name in declared and declared[name] != source.sha256) or (name in files and files[name] != source.sha256):
                raise DataReadinessError("completed cash facts declare conflicting exact source pins")
            declared[name] = source.sha256
    return {item.action_id: item for item in manifest.facts}


def new_completed_cash_ids(
    prior: Mapping[str, ReviewedCompletedCashDistribution], supplied: Mapping[str, ReviewedCompletedCashDistribution],
) -> frozenset[str]:
    if any(supplied.get(action_id) != fact for action_id, fact in prior.items()):
        raise DataReadinessError("completed cash facts must preserve all prior exact facts and source pins")
    return frozenset(supplied.keys() - prior.keys())


def _official_blocks(document: RetainedSecDocument) -> tuple[str, set[str]]:
    soup = BeautifulSoup(document.body, "html.parser")
    for node in soup(["head", "script", "style", "template", "ix:hidden"]):
        node.decompose()
    block_tags = ("p", "div", "li", "td")
    blocks: set[str] = set()
    for node in soup.find_all(block_tags):
        if node.find(block_tags) is not None:
            continue
        block = " ".join(" ".join(node.stripped_strings).split())
        previous = node.find_previous_sibling()
        # The retained issuer slide pairs an image with a text transcript.
        # Its repeated U+2022 markers delimit actual slide bullets.
        if (node.name == "div" and previous is not None and previous.name == "img"
                and re.fullmatch(r"slide[0-9]+", str(previous.get("title", "")), re.I)
                and block.count("\u2022") >= 2):
            blocks.update(chunk.strip() for chunk in block.split("\u2022") if chunk.strip())
        else:
            blocks.add(block.lstrip("\u2022 ").strip())
    return " ".join(" ".join(soup.stripped_strings).split()), blocks


def _date_pattern(day: date, *, year_required: bool) -> str:
    month = calendar.month_name[day.month]
    short = calendar.month_abbr[day.month]
    year = rf",? {day.year}" if year_required else rf"(?:,? {day.year})?"
    return rf"(?:{month}|{short}\.?) {day.day}{year}(?![0-9])"


def _amount_pattern(amount: Decimal) -> str:
    return re.escape(format(amount, "f")) + r"(?![0-9.])"


def _issuer_name_key(value: object, incorporation_code: str | None = None) -> str:
    if not isinstance(value, str):
        return ""
    base, slash, suffix = value.rpartition("/")
    # EDGAR may append a terminal jurisdiction to the registrant's legal name.
    if slash and incorporation_code in _US_JURISDICTIONS and suffix.strip().upper() == incorporation_code:
        value = base
    words = re.findall(r"[a-z0-9]+", value.casefold())
    if words and words[0] == "the":
        words.pop(0)
    if words and words[-1] == "the":
        words.pop()
    aliases = {"co": "company", "corp": "corporation"}
    return " ".join(aliases.get(word, word) for word in words)


def _verified_issuer_cash(
    root: Path, fact: ReviewedCompletedCashDistribution, policy: str, files: MutableMapping[str, str],
    sec_cache: dict[str, Any], memory_guard: Callable[[], None],
) -> tuple[EvidenceReference, ...]:
    def source(statement: SecCashStatement, role: str) -> tuple[RetainedSecDocument, set[str], tuple[EvidenceReference, ...]]:
        document = read_retained_sec_document(root, statement.document, fact.sec_cik, files, sec_cache,
            primary_only=False, memory_guard=memory_guard)
        text, blocks = _official_blocks(document)
        if text.count(statement.quote) != 1 or statement.quote not in blocks:
            raise DataReadinessError("issuer cash source lacks one exact complete contextual statement")
        capture = document.captured_at.to_pydatetime()
        locator = (f"issuer completed cash {role};action_id={fact.action_id};unit={statement.document.unit_id};"
            f"member={document.body_member};body_sha256={statement.document.body_sha256}")
        references = (
            EvidenceReference(reference=str(document.body_path), artifact_sha256=files[str(document.body_path)],
                interpretation_policy_sha256=policy, record_locator=locator, retrieved_at=capture, available_at=capture),
            EvidenceReference(reference=str(document.index_path), artifact_sha256=files[str(document.index_path)],
                interpretation_policy_sha256=policy,
                record_locator=f"issuer completed cash index;action_id={fact.action_id};member={document.index_member};"
                               f"body_sha256={document.index_body_sha256}",
                retrieved_at=document.index_captured_at.to_pydatetime(), available_at=document.index_captured_at.to_pydatetime()),
        )
        return document, blocks, references

    declaration, _, declared_refs = source(fact.declaration.statement, "declaration")
    completion, completion_blocks, completed_refs = source(fact.completion.statement, "completion")
    if not declaration.accepted_at.date() < fact.ex_date <= fact.completion.upper_bound <= completion.accepted_at.date():
        raise DataReadinessError("issuer declaration and completed distribution source chronology differs")
    declared, completed = fact.declaration.statement.quote, fact.completion.statement.quote
    if not re.search(r"\bboard(?: of directors)? (?:has approved|approved|declares|declared)\b", declared, re.I):
        raise DataReadinessError("issuer cash declaration does not affirm a board decision")
    amount = _amount_pattern(_decimal(fact.gross_amount))
    if fact.declaration.kind == "amount_record_and_payable":
        record = _date_pattern(fact.record_date, year_required=True)
        payable = _date_pattern(fact.payable_date, year_required=True)
        if (not re.search(rf"special (?:cash )?dividend of \${amount} per share\b", declared, re.I)
                or not re.search(rf"shareholders of record on {record}", declared, re.I)
                or not re.search(rf"to be distributed on {payable}", declared, re.I)):
            raise DataReadinessError("issuer declaration amount, record date or announced distribution differs")
    else:
        context = fact.declaration.event_context
        special_total = re.compile(r"\$[0-9,.]+ (?:million|billion) special (?:cash )?dividend", re.I)
        if (context is None or context not in declared or context not in completed
                or len(special_total.findall(declared)) != 1
                or special_total.findall(declared) != special_total.findall(completed)):
            raise DataReadinessError("issuer declaration and completion do not identify the same special distribution")

    if fact.completion.kind == "paid_date":
        paid = _date_pattern(fact.payable_date, year_required=False)
        statement = rf"The special dividend was paid on {paid} in the amount of \${amount} per share\."
        if fact.declaration.event_context is not None:
            context = re.escape(fact.declaration.event_context)
            statement = (rf"In conjunction with the {context}, [A-Za-z][A-Za-z &.'-]{{0,80}} paid a "
                rf"\$[0-9,.]+ (?:million|billion) special (?:cash )?dividend to shareholders\. " + statement)
        if (completion.accepted_at.year != fact.payable_date.year
                or re.search(r"\b(?:not|never|unclear|whether|would|will|could|should|expected|pending)\b", completed, re.I)
                or re.fullmatch(statement, completed, re.I) is None):
            raise DataReadinessError("issuer source does not affirm the exact completed payment date and amount")
        return (*declared_refs, *completed_refs)

    period_statement = fact.completion.reporting_period
    assert period_statement is not None
    period, period_blocks, period_refs = source(period_statement, "reporting_period")
    quarter = re.fullmatch(rf"\${amount} per share special dividend distributed in the (first|second|third|fourth) quarter\.?",
        completed, re.I)
    period_end = _date_pattern(fact.completion.upper_bound, year_required=True)
    year = fact.completion.upper_bound.year
    if (quarter is None or not any(re.fullmatch(rf"Appendix {quarter[1]} Quarter {year} Performance(?: [0-9]+)?", block, re.I)
                or re.search(rf"\b{quarter[1]} quarter {year} performance\b", block, re.I) for block in completion_blocks)
            or not any(re.fullmatch(rf".+ reports {quarter[1]} quarter {year} results", block, re.I) for block in period_blocks)
            or re.fullmatch(rf"For the three months ended {period_end}, the company reported .+[:.]", period_statement.quote, re.I) is None
            or not fact.completion.upper_bound <= period.accepted_at.date() <= completion.accepted_at.date()):
        raise DataReadinessError("issuer source does not affirm the completed distribution in the exact closed reporting period")
    return (*declared_refs, *completed_refs, *period_refs)


def verify_completed_cash_fact(
    root: Path, fact: ReviewedCompletedCashDistribution, original: dict[str, Any], manifest: SourcePin,
    policy: str, files: MutableMapping[str, str], stock_cache: dict[str, Any], sec_cache: dict[str, Any],
    *, memory_guard: Callable[[], None],
) -> VerifiedCompletedCashDistribution:
    process = original.get("process_date")
    if (not _CASH_FIELDS.issubset(original) or set(original) - (_CASH_FIELDS | {"currency"})
            or json_sha256(original) != fact.retained_record_sha256 or original["id"] != fact.action_id
            or original["symbol"] != fact.ticker or original["cusip"] != fact.cusip
            or original["special"] is not True or original["foreign"] is not False
            or original.get("currency") not in (None, "", "USD")
            or original["ex_date"] != fact.ex_date.isoformat() or original["record_date"] != fact.record_date.isoformat()
            or original["payable_date"] != fact.payable_date.isoformat()
            or _decimal(original["rate"]) != _decimal(fact.gross_amount)
            or not isinstance(process, str) or date.fromisoformat(process).isoformat() != process):
        raise DataReadinessError("completed cash proof differs from the exact original special provider action")

    proof_path = pin_retained_source(root, fact.stock_class_proof, files)
    if proof_path.stat().st_size > 1024**2:
        raise DataReadinessError("completed cash stock proof exceeds bound")
    raw = proof_path.read_bytes()
    parse_strict_json_object(raw, label="completed cash stock class proof")
    proof = StockIdentityProof.model_validate_json(raw)
    if (proof.security_id, proof.ticker, proof.sec_cik) != (fact.security_id, fact.ticker, fact.sec_cik):
        raise DataReadinessError("completed cash proof names a different stable security, ticker or issuer")
    sessions = tuple(sorted({fact.ex_date, fact.record_date}))
    identity = replay_stock_identity(root, proof, sessions=sessions, source_files=files, interpretation_sha256=policy, cache=stock_cache)
    if (identity.proven_sessions != sessions or identity.missing_reasons or identity.competing_class_tickers
            or not identity.evidence):
        raise DataReadinessError("completed cash proof lacks the unique dated stock class")

    def tax_class(calculation: Mapping[str, Any]) -> tuple[EvidenceReference, ...]:
        refs, observation = replay_stock_tax_class(calculation, fact.tax.lookup, root, policy,
            issuer_name=identity.issuer_name, security_title=identity.security_title)
        expected = _issuer_name_key(identity.issuer_name, identity.issuer_incorporation_code)
        if not expected or _issuer_name_key(observation["tax_institution"]) != expected:
            raise DataReadinessError("completed tax distribution issuer does not match the dated stock class")
        return refs

    _, tax_refs = replay_tax_currency_event(root, fact.tax, fact.ex_date, _decimal(fact.gross_amount), Decimal(0), policy,
        stock_tax_validation=tax_class, payable_date=fact.payable_date, record_date=fact.record_date)
    official_refs = _verified_issuer_cash(root, fact, policy, files, sec_cache, memory_guard)
    checked = datetime.now(UTC)
    fact_ref = EvidenceReference(reference=manifest.path, artifact_sha256=manifest.sha256,
        interpretation_policy_sha256=policy,
        record_locator=f"completed special cash fact;action_id={fact.action_id};security_id={fact.security_id};"
                       f"completion_upper_bound={fact.completion.upper_bound};relevance={COMPLETED_CASH_RELEVANCE}",
        retrieved_at=checked, available_at=checked)
    proof_ref = EvidenceReference(reference=fact.stock_class_proof.path, artifact_sha256=fact.stock_class_proof.sha256,
        interpretation_policy_sha256=policy, record_locator=f"completed special cash class proof;action_id={fact.action_id};"
                       f"security_id={fact.security_id};ticker={fact.ticker}",
        retrieved_at=identity.verified_at_utc, available_at=identity.verified_at_utc)
    evidence = (*identity.evidence, proof_ref, *tax_refs, *official_refs, fact_ref)
    for reference in evidence:
        pin_retained_source(root, SourcePin(path=reference.reference, sha256=reference.artifact_sha256), files)
    return VerifiedCompletedCashDistribution(fact, evidence)
