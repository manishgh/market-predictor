"""Provider-bounded action relevance, never universal absence or payment evidence."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date
from typing import Any

from market_predictor.core.errors import DataReadinessError

_EX_DATE_FAMILIES = frozenset(("cash_dividends", "stock_dividends", "forward_splits", "reverse_splits",
    "unit_splits", "spin_offs", "rights_distributions", "capital_gains_distributions"))
_EFFECTIVE_DATE_FAMILIES = frozenset(("cash_mergers", "stock_mergers", "stock_and_cash_mergers",
    "redemptions", "worthless_removals", "partial_calls", "reorganizations", "name_changes"))
_ENTITLEMENT_DATES = ("record_date", "payable_date", "due_bill_redemption_date", "effective_date", "ex_date")


@dataclass(frozen=True, slots=True)
class ActionWindow:
    action_id: str
    family: str
    first_relevant_date: date | None
    last_relevant_date: date | None
    unavailable_reason: str
    requires_ownership_before_first_date: bool = False

    def intersects(self, first: date, last: date) -> bool:
        if first > last:
            raise DataReadinessError("action comparison has a reversed holding interval")
        if self.first_relevant_date is not None and self.first_relevant_date > last:
            return False
        if self.last_relevant_date is not None and self.last_relevant_date < first:
            return False
        if self.requires_ownership_before_first_date:
            if self.first_relevant_date is None:
                return True
            return first < self.first_relevant_date <= last
        return True


def action_window(family: str, record: Mapping[str, Any]) -> ActionWindow:
    """Conservatively bound entitlement exposure from economic, not process, dates.

    This only identifies windows requiring corporate-action interpretation. Even a
    fully dated record cannot itself establish a claim's valuation or spendable cash.
    Unknown families or missing/invalid clocks stay unbounded, never action-free.
    """
    identity = record.get("id")
    if not isinstance(identity, str) or not identity.strip():
        identity = "unidentified_action"
    if family not in _EX_DATE_FAMILIES | _EFFECTIVE_DATE_FAMILIES:
        return ActionWindow(identity, family, None, None, "unclassified_action_family")
    required = "ex_date" if family in _EX_DATE_FAMILIES else "effective_date"
    parsed: dict[str, date] = {}
    ordinary_cash = family == "cash_dividends" and record.get("special") is False
    # Ancillary errors remain in the raw record for claim interpretation. They
    # cannot give a newly opened ex-date holding an earlier owner's entitlement.
    fields = ("ex_date",) if ordinary_cash else _ENTITLEMENT_DATES
    for key in fields:
        value = record.get(key)
        if value is None or value == "":
            continue
        try:
            if not isinstance(value, str):
                raise ValueError("date must be ISO text")
            parsed[key] = date.fromisoformat(value)
            if parsed[key].isoformat() != value:
                raise ValueError("date must be canonical ISO text")
        except ValueError:
            return ActionWindow(identity, family, None, None, "invalid_action_economic_date")
    if required not in parsed:
        return ActionWindow(identity, family, None, None, "missing_action_economic_date")
    if (family in {"spin_offs", "rights_distributions", "stock_dividends"}
            or family == "cash_dividends" and record.get("special") is not False):
        if not {"record_date", "payable_date", "due_bill_redemption_date"}.issubset(parsed):
            return ActionWindow(identity, family, None, None, "unbounded_entitlement_or_due_bill_dates")
    if ordinary_cash:
        # Entry at the ex-date open does not earn this ordinary dividend. A later
        # payment to a prior owner is not a claim of the newly opened holding.
        return ActionWindow(identity, family, parsed["ex_date"], parsed["ex_date"],
            "corporate_terms_require_replay", requires_ownership_before_first_date=True)
    return ActionWindow(identity, family, min(parsed.values()), max(parsed.values()), "corporate_terms_require_replay")


def relevant_action_windows(
    families: Mapping[str, list[dict[str, Any]]], *, first: date, last: date,
) -> tuple[ActionWindow, ...]:
    """Report unresolved action exposure without silently discarding unfamiliar rows."""
    return tuple(window for family, records in sorted(families.items()) for record in records
        if (window := action_window(family, record)).intersects(first, last))
