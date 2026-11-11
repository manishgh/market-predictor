"""Source-bound relevance for selected investment holdings, without event valuation."""
from __future__ import annotations

import math
import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.io import inside
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


def _cusip(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9]{8}[0-9]", value):
        return False
    total = 0
    for index, character in enumerate(value[:8]):
        weighted = int(character, 36) * (1 + index % 2)
        total += weighted // 10 + weighted % 10
    return (total + int(value[-1])) % 10 == 0


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
                or not (_symbol(old) and _symbol(new) and _cusip(old_cusip) and _cusip(new_cusip))
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
    source, daughter = record.get("source_symbol"), record.get("new_symbol")
    source_cusip, daughter_cusip = record.get("source_cusip"), record.get("new_cusip")
    source_rate, daughter_rate = record.get("source_rate"), record.get("new_rate")
    if (not _SPIN_FIELDS.issubset(record) or set(record) - (_SPIN_FIELDS | _SPIN_DATES)
            or source != symbol or not (_symbol(source) and _symbol(daughter)) or source == daughter
            or not (_cusip(source_cusip) and _cusip(daughter_cusip)) or source_cusip == daughter_cusip
            or type(source_rate) not in (int, float) or source_rate != 1
            or not isinstance(daughter_rate, (int, float)) or isinstance(daughter_rate, bool)
            or not math.isfinite(daughter_rate) or daughter_rate <= 0):
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


class InvestmentActionRelevance:
    """Bind a scoped provider record and reviewed official fact to each omitted event."""

    def __init__(
        self, root: Path, records: dict[tuple[str, str], RelevanceRecord],
        dates_pin: SourcePin | None, parent_pin: SourcePin | None, policy: EvidenceReference,
        *, memory_guard: Callable[[], None],
    ) -> None:
        self.records = records
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

    def references(
        self, symbol: str, family: str, action_id: str, first: date, last: date,
    ) -> tuple[EvidenceReference, ...]:
        item = self.records.get((symbol, action_id))
        if item is None or item.family != family:
            return ()
        reason = action_relevance_reason(
            family, item.record, symbol, first, last, self.dates.get(action_id), self.parent_units.get(action_id),
        )
        if reason is None:
            return ()
        record, *transport = item.evidence
        binding = record.model_copy(update={"record_locator": f"{record.record_locator};relevance={reason}"})
        return (binding, *transport, *self.official.get(action_id, ()))

