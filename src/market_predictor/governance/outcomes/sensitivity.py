"""Diagnostic fills for unresolvable outcomes, so that a report shows what leaving them out hides.

A mean over matured outcomes silently gives every unresolvable one the mean. Each unresolvable
outcome that was entered gets two fills, net of the execution cost like a matured net return,
with the sector ETF from the entry open to the fill session's close:
- `last_usable_close`: sold at the last usable close before the gap;
- a stress fill from the evidence: a cash merger's cash per share; a stock merger's acquirer
  shares at the acquirer's close on the effective session, plus the cash of a stock-and-cash
  merger (`merger_at_last_close` when the acquirer has no bar, such as a foreign listing); a
  reorganization, which preserves value, at the last usable close; nothing for a worthless
  removal; for any other removal the Nasdaq delisting return, -55% from the last usable close
  (Shumway and Warther 1999), because the listing exchange is not recorded yet; and for a
  halt, the managed path continued past it.
A fill session after the horizon's last session is capped there, and a fill needs the sector
ETF's bar on its session. The acquirer's close carries the adjustments made between the
effective date and its retrieval, so a dividend in between slightly understates the shares'
nominal value. The fills never gate.
"""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd

from market_predictor.execution_policy import DEFAULT_EXECUTION_POLICY, executable_fill_price, round_trip_cost_bps
from market_predictor.governance.outcomes.contracts import OPERATOR_VERIFIED, PredictionMaturationIntent, SensitivityFill
from market_predictor.governance.outcomes.evidence import PathEvidence, usable_rows
from market_predictor.governance.outcomes.sessions import session_on_or_before

DELISTING_STRESS_RETURN = -0.55
_STOCK_MERGERS = ("stock_merger", "stock_and_cash_merger")


def sensitivity_fills(
    intent: PredictionMaturationIntent,
    *,
    evidence: PathEvidence,
    reason: str,
    cessation_record: Mapping[str, Any] | None,
    acquirer_close: float | None,
    acquirer_collected: bool,
) -> tuple[tuple[SensitivityFill, ...], bool]:
    """The fills of an unresolvable outcome, and whether its position was entered at all.

    A never-entered decision has no position and so no return to fill. A stock merger's stress
    fill waits until the acquirer's bars were collected (`acquirer_collected`).
    """
    stock = usable_rows(evidence.bars, intent.ticker)
    sector = usable_rows(evidence.bars, intent.primary_benchmark)
    decision, entry_session = intent.decision_session_et, evidence.path_sessions[0]
    if decision not in stock or entry_session not in stock:
        return (), False
    if entry_session not in sector:
        return (), True
    entry = float(stock[entry_session]["open"])
    decision_close = float(stock[decision]["close"])
    cost = round_trip_cost_bps(
        price=entry,
        atr_pct=intent.decision_atr_fraction * decision_close / entry,
        participation=0.0,
        policy=DEFAULT_EXECUTION_POLICY,
    ) / 10_000.0
    last_session = evidence.path_sessions[-1]

    def fill(basis: str, value: float, session: date) -> SensitivityFill | None:
        capped = min(session, last_session)
        if capped not in sector:
            return None
        net = value / entry - 1.0 - cost
        sector_return = float(sector[capped]["close"]) / float(sector[entry_session]["open"]) - 1.0
        return SensitivityFill.model_validate(
            {
                "basis": basis,
                "fill_session": capped,
                "net_return": net,
                "sector_return": sector_return,
                "excess_return_vs_sector": net - sector_return,
            }
        )

    gap = evidence.first_gap or last_session
    traded = [session for session in evidence.path_sessions if session < gap and session in stock]
    if not traded:
        return (), False
    last_traded = max(traded)
    last_close = float(stock[last_traded]["close"])
    fills: list[SensitivityFill | None] = [fill("last_usable_close", last_close, last_traded)]
    effective = _effective_session(cessation_record)
    if reason == "cash_merger" and cessation_record is not None and effective is not None:
        fills.append(fill("cash_merger", _number(cessation_record, "rate"), effective))
    elif reason in _STOCK_MERGERS and cessation_record is not None and effective is not None:
        if acquirer_close is not None:
            cash = _number(cessation_record, "cash_rate") if reason == "stock_and_cash_merger" else 0.0
            acquiree_shares = _number(cessation_record, "acquiree_rate")
            if acquiree_shares == 0:
                raise ValueError("a stock merger's acquiree_rate is zero")
            shares = _number(cessation_record, "acquirer_rate") / acquiree_shares
            fills.append(fill(reason, cash + shares * acquirer_close, effective))
        elif acquirer_collected:
            # The acquirer has no bar to value the shares; the deal is taken at the last close.
            fills.append(fill("merger_at_last_close", last_close, effective))
    elif reason == "reorganization":
        fills.append(fill("reorganization", last_close, effective or gap))
    elif reason == "worthless_removal":
        fills.append(fill("worthless_removal", 0.0, gap))
    elif reason == "interior_gap" or (reason == OPERATOR_VERIFIED and evidence.trades_after(gap)):
        # The bars after a halt are observed, so the managed path continues past it.
        fills.append(_halt_crossing(intent, stock, evidence.path_sessions, entry, decision_close, fill))
    else:
        # A membership removal or an operator's finding on a tail gap: a delisting.
        fills.append(fill("delisting_stress", last_close * (1.0 + DELISTING_STRESS_RETURN), gap))
    return tuple(item for item in fills if item is not None), True


def acquirer_terms(reason: str, record: Mapping[str, Any] | None) -> tuple[str, date] | None:
    """The acquirer symbol and effective session a stock merger's stress fill needs bars for."""
    effective = _effective_session(record)
    if reason not in _STOCK_MERGERS or record is None or effective is None or not record.get("acquirer_symbol"):
        return None
    return str(record["acquirer_symbol"]).strip().upper(), effective


def _halt_crossing(
    intent: PredictionMaturationIntent,
    stock: Mapping[date, pd.Series],
    path_sessions: tuple[date, ...],
    entry: float,
    decision_close: float,
    fill: Any,
) -> SensitivityFill | None:
    """The managed path continued over the usable bars past the halt, with the pinned barrier's levels.

    A stop crossed during the halt fills at the lower of the stop and the first open after it.
    """
    policy = intent.label_policy
    atr = intent.decision_atr_fraction * decision_close
    target = entry + float(str(policy["target_atr_multiple"])) * atr
    stop = entry - float(str(policy["stop_atr_multiple"])) * atr
    usable = [session for session in path_sessions if session in stock]
    for session in usable:
        row = stock[session]
        low, high = float(row["low"]), float(row["high"])
        if low <= stop or high >= target:
            outcome = "stop_first" if low <= stop else "target_first"
            price = executable_fill_price(
                outcome=outcome, target_price=target, stop_price=stop,
                trigger_open=float(row["open"]), final_price=float(row["close"]),
            )
            result: SensitivityFill | None = fill("halt_crossing", price, session)
            return result
    last = usable[-1]
    timeout: SensitivityFill | None = fill("halt_crossing", float(stock[last]["close"]), last)
    return timeout


def _effective_session(record: Mapping[str, Any] | None) -> date | None:
    if record is None:
        return None
    try:
        return session_on_or_before(date.fromisoformat(str(record.get("effective_date"))))
    except ValueError:
        return None


def _number(record: Mapping[str, Any], key: str) -> float:
    value = record.get(key)
    try:
        number = float(str(value))
    except ValueError as exc:
        raise ValueError(f"a corporate action's {key} is not a number: {value}") from exc
    if not number >= 0:
        raise ValueError(f"a corporate action's {key} is negative or missing: {value}")
    return number
