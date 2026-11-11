"""Source-bound relevance for selected investment holdings, without event valuation."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping, MutableMapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.investment.completed_cash_distributions import (
    COMPLETED_CASH_RELEVANCE,
    VerifiedCompletedCashDistribution,
)
from market_predictor.investment.dividend_currencies import valid_cusip
from market_predictor.investment.stock_identity import StockIdentityProof, StockIdentityUnavailable, replay_stock_identity
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_windows import holding_ends_before_distribution, observed_economic_dates
from market_predictor.swing.datasets.reviewed_action_dates import ReviewedActionDate, load_reviewed_action_dates
from market_predictor.swing.datasets.reviewed_parent_units import ReviewedParentSpinUnits, load_reviewed_parent_spin_units

_NAME_FIELDS = {"id", "old_cusip", "new_cusip", "old_symbol", "new_symbol", "process_date"}
_SPIN_FIELDS = {"id", "source_cusip", "new_cusip", "source_symbol", "new_symbol", "source_rate", "new_rate", "process_date"}
_SPIN_DATES = {"ex_date", "record_date", "payable_date", "due_bill_redemption_date", "effective_date"}
SUPPORTED_REASONS = {
    "missing_action_economic_date", "unbounded_entitlement_or_due_bill_dates", "corporate_terms_require_replay",
}


def _symbol(value: object) -> bool:
    return isinstance(value, str) and re.fullmatch(r"[A-Z0-9][A-Z0-9.-]*", value) is not None


RECIPIENT_PRE_DISTRIBUTION_RELEVANCE = "existing_recipient_common_stock_holding_ends_before_distribution"
_REGULAR_STOCK_EXCHANGES = frozenset({
    "new york stock exchange", "nyse", "nyse american", "nyse american llc",
    "nasdaq", "the nasdaq stock market llc", "nasdaq global select market", "the nasdaq global select market",
    "nasdaq global market", "the nasdaq global market", "nasdaq capital market", "the nasdaq capital market",
    "cboe bzx exchange", "cboe bzx exchange, inc.",
})


def _spin_terms_supported(record: Mapping[str, Any]) -> bool:
    source, daughter = record.get("source_symbol"), record.get("new_symbol")
    source_cusip, daughter_cusip = record.get("source_cusip"), record.get("new_cusip")
    source_rate, daughter_rate = record.get("source_rate"), record.get("new_rate")
    return (_SPIN_FIELDS.issubset(record) and not set(record) - (_SPIN_FIELDS | _SPIN_DATES)
        and _symbol(source) and _symbol(daughter) and source != daughter
        and valid_cusip(source_cusip) and valid_cusip(daughter_cusip) and source_cusip != daughter_cusip
        and type(source_rate) in (int, float) and source_rate == 1
        and isinstance(daughter_rate, (int, float)) and not isinstance(daughter_rate, bool)
        and math.isfinite(daughter_rate) and daughter_rate > 0)


def action_relevance_reason(
    family: str, record: Mapping[str, Any], symbol: str, first: date, last: date,
    reviewed_date: ReviewedActionDate | None, reviewed_parent: ReviewedParentSpinUnits | None,
) -> str | None:
    """Prove that an exact action cannot change this newly opened parent holding."""
    if first > last:
        raise DataReadinessError("investment action relevance interval is reversed")
    try:
        if not observed_economic_dates(record, ("process_date",)):
            return None
        dates = observed_economic_dates(record)
    except ValueError:
        return None
    if family == "name_changes":
        old, new = record.get("old_symbol"), record.get("new_symbol")
        old_cusip, new_cusip = record.get("old_cusip"), record.get("new_cusip")
        if (not _NAME_FIELDS.issubset(record) or set(record) - (_NAME_FIELDS | {"effective_date"})
                or not (_symbol(old) and _symbol(new) and valid_cusip(old_cusip) and valid_cusip(new_cusip))
                or symbol not in (old, new)):
            return None
        if set(record) == _NAME_FIELDS and old_cusip == new_cusip:
            return "same_cusip_name_change_preserves_owned_units"
        if reviewed_date is not None and reviewed_date.family == family and reviewed_date.action_id == record.get("id"):
            effective = reviewed_date.economic_dates["effective_date"]
            if last < effective and symbol == old:
                return "holding_ends_before_official_identity_change"
            if first > effective and old == new == symbol:
                return "new_same_symbol_raw_window_after_official_identity_change"
        return None
    if family != "spin_offs":
        return None
    if record.get("source_symbol") != symbol or not _spin_terms_supported(record):
        return None
    if "record_date" in dates and "payable_date" in dates and dates["record_date"] > dates["payable_date"]:
        return None
    if (reviewed_parent is not None and reviewed_parent.source_symbol == symbol
            and reviewed_parent.action_id == record.get("id")):
        ex_date = reviewed_parent.ex_date
        if last < ex_date:
            return "parent_holding_marked_before_distribution_ex_date"
        if first >= ex_date:
            return "new_regular_way_parent_purchase_on_or_after_ex_date"
    if holding_ends_before_distribution(dates, last):
        return "holding_ends_before_distribution_economic_dates"
    return None


@dataclass(frozen=True)
class RelevanceRecord:
    family: str
    record: dict[str, Any]
    evidence: tuple[EvidenceReference, ...]


@dataclass(frozen=True)
class ExistingRecipientStockAuthority:
    manifest: SourcePin
    proofs: Mapping[tuple[str, str], StockIdentityProof]


@dataclass(frozen=True)
class _RecipientCandidate:
    recipient: RelevanceRecord
    source: RelevanceRecord
    proof: StockIdentityProof
    first_distribution_date: date


class InvestmentActionRelevance:
    """Bind a scoped provider record and reviewed official fact to each omitted event."""

    def __init__(
        self, root: Path, records: dict[tuple[str, str], RelevanceRecord],
        dates_pin: SourcePin | None, parent_pin: SourcePin | None, policy: EvidenceReference,
        *, memory_guard: Callable[[], None],
        completed_cash: Mapping[str, VerifiedCompletedCashDistribution] | None = None,
        recipients: ExistingRecipientStockAuthority | None = None,
        stock_cache: dict[str, Any] | None = None,
        recipient_source_files: MutableMapping[str, str] | None = None,
    ) -> None:
        self.records = records
        self.root, self.policy = root, policy
        self.completed_cash = {} if completed_cash is None else completed_cash
        self.recipients = recipients
        self.stock_cache = {} if stock_cache is None else stock_cache
        self.recipient_source_files = {} if recipient_source_files is None else recipient_source_files
        by_action = {key[1]: (item.family, item.record) for key, item in records.items()}
        self.dates, date_pins = load_reviewed_action_dates(root, dates_pin, by_action, memory_guard=memory_guard)
        self.parent_units, unit_pins = load_reviewed_parent_spin_units(
            root, parent_pin, by_action, self.dates, memory_guard=memory_guard,
        )
        if any(name in date_pins and date_pins[name] != digest for name, digest in unit_pins.items()):
            raise DataReadinessError("investment action relevance source pins conflict")
        self.source_files = {**date_pins, **unit_pins}
        self.official: dict[str, tuple[EvidenceReference, ...]] = {}
        for facts, pin in ((self.dates, dates_pin), (self.parent_units, parent_pin)):
            if pin is None:
                continue
            for action, fact in facts.items():
                locator = f"reviewed action relevance;action_id={action};record_sha256={fact.retained_record_sha256}"
                evidence = [EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256,
                    interpretation_policy_sha256=policy.interpretation_policy_sha256, record_locator=locator,
                    retrieved_at=policy.retrieved_at, available_at=policy.available_at)]
                for receipt_pin in fact.official_receipts:
                    receipt_path = inside(root, receipt_pin.path)
                    transport = receipt_path.parent / "transport.json"
                    value = parse_strict_json_object(transport.read_bytes(), label="reviewed official action transport")
                    raw_clock = value.get("retrieved_at_utc")
                    if not isinstance(raw_clock, str):
                        raise DataReadinessError("reviewed official action transport lacks a capture clock")
                    captured = datetime.fromisoformat(raw_clock)
                    if captured.utcoffset() is None or captured > policy.retrieved_at:
                        raise DataReadinessError("reviewed action capture is unavailable at current verification")
                    evidence.append(EvidenceReference(reference=receipt_pin.path, artifact_sha256=receipt_pin.sha256,
                        interpretation_policy_sha256=policy.interpretation_policy_sha256, record_locator=locator,
                        retrieved_at=captured, available_at=captured))
                self.official[action] = (*self.official.get(action, ()), *evidence)

    def _recipient_candidate(self, security_id: str, symbol: str, action_id: str, last: date) -> _RecipientCandidate | None:
        if self.recipients is None:
            return None
        recipient = self.records.get((symbol, action_id))
        proof = self.recipients.proofs.get((security_id, symbol))
        if (recipient is None or recipient.family != "spin_offs" or proof is None
                or (proof.security_id, proof.ticker) != (security_id, symbol)
                or not re.match(r"^common (?:stock|shares)\b", proof.security_title, re.I)
                or re.search(r"\bwhen[- ](?:issued|distributed)\b", proof.security_title, re.I)):
            return None
        record = recipient.record
        source_symbol = record.get("source_symbol")
        if not isinstance(source_symbol, str):
            return None
        parent = self.records.get((source_symbol, action_id))
        if (record.get("new_symbol") != symbol or not _spin_terms_supported(record)
                or parent is None or parent.family != "spin_offs" or json_sha256(parent.record) != json_sha256(record)):
            return None
        try:
            dates = observed_economic_dates(record)
            if (not observed_economic_dates(record, ("process_date",))
                    or not holding_ends_before_distribution(dates, last)
                    or dates["record_date"] > dates["payable_date"]):
                return None
        except ValueError:
            return None
        return _RecipientCandidate(recipient, parent, proof, min(dates.values()))

    def is_new_distribution_candidate(
        self, security_id: str, symbol: str, family: str, action_id: str, first: date, last: date,
        *, new_cash_ids: frozenset[str], recipient_source_added: bool,
    ) -> bool:
        if family == "cash_dividends" and action_id in new_cash_ids:
            fact = self.completed_cash.get(action_id)
            return fact is not None and bool(fact.references(security_id, symbol, first))
        return (family == "spin_offs" and recipient_source_added
            and self._recipient_candidate(security_id, symbol, action_id, last) is not None)

    def _recipient_references(
        self, security_id: str, symbol: str, action_id: str, sessions: tuple[date, ...],
    ) -> tuple[EvidenceReference, ...]:
        candidate = self._recipient_candidate(security_id, symbol, action_id, sessions[-1])
        if candidate is None or self.recipients is None:
            return ()
        try:
            identity = replay_stock_identity(self.root, candidate.proof, sessions=sessions, cache=self.stock_cache,
                source_files=self.recipient_source_files, interpretation_sha256=self.policy.interpretation_policy_sha256)
        except StockIdentityUnavailable:
            return ()
        if (identity.proven_sessions != sessions or identity.missing_reasons or identity.competing_class_tickers
                or identity.exchange.strip().casefold() not in _REGULAR_STOCK_EXCHANGES or not identity.evidence
                or identity.anchor_accepted_at_utc.date() >= candidate.first_distribution_date):
            return ()
        manifest = self.recipients.manifest
        class_ref = EvidenceReference(reference=manifest.path, artifact_sha256=manifest.sha256,
            interpretation_policy_sha256=self.policy.interpretation_policy_sha256,
            record_locator=f"existing regular-way recipient class;action_id={action_id};security_id={security_id};ticker={symbol}",
            retrieved_at=identity.verified_at_utc, available_at=identity.verified_at_utc)
        record, *transport = candidate.recipient.evidence
        binding = record.model_copy(update={"record_locator": f"{record.record_locator};relevance={RECIPIENT_PRE_DISTRIBUTION_RELEVANCE}"})
        return (binding, *transport, *candidate.source.evidence, class_ref, *identity.evidence)

    def references(
        self, symbol: str, family: str, action_id: str, first: date, last: date,
        *, security_id: str = "", sessions: tuple[date, ...] = (),
    ) -> tuple[EvidenceReference, ...]:
        item = self.records.get((symbol, action_id))
        if item is None or item.family != family:
            return ()
        if family == "cash_dividends":
            fact = self.completed_cash.get(action_id)
            if fact is not None and (cash_refs := fact.references(security_id, symbol, first)):
                record, *transport = item.evidence
                binding = record.model_copy(update={"record_locator": f"{record.record_locator};relevance={COMPLETED_CASH_RELEVANCE}"})
                return (binding, *transport, *cash_refs)
            return ()
        if family == "spin_offs" and sessions and item.record.get("new_symbol") == symbol:
            return self._recipient_references(security_id, symbol, action_id, sessions)
        reason = action_relevance_reason(
            family, item.record, symbol, first, last, self.dates.get(action_id), self.parent_units.get(action_id),
        )
        if reason is None:
            return ()
        record, *transport = item.evidence
        binding = record.model_copy(update={"record_locator": f"{record.record_locator};relevance={reason}"})
        return (binding, *transport, *self.official.get(action_id, ()))

