"""Canonical cash-funded swing evaluation; no training-orchestrator dependencies."""
from __future__ import annotations

import math
from collections.abc import Mapping
from datetime import date
from typing import Any, Protocol

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.features.panel import MANAGED_PATH_NET_RETURN_COLUMNS, MANAGED_PATH_SESSION_ORDINAL_COLUMNS


class SwingLedgerConfig(Protocol):
    """Read-only capital inputs shared by training and accounting contracts."""

    @property
    def horizon_sessions(self) -> int: ...

    @property
    def expected_round_trip_cost_bps(self) -> float: ...

    @property
    def maximum_trades_per_decision(self) -> int: ...


def swing_valuation_sessions(session_calendar: tuple[str, ...], horizon: int) -> tuple[str, ...]:
    """Freeze valuation dates independently of selected trades, including the tail."""
    try:
        if not session_calendar:
            raise ValueError("empty decision calendar")
        parsed = tuple(date.fromisoformat(value) for value in session_calendar)
        if tuple(value.isoformat() for value in parsed) != session_calendar:
            raise ValueError("non-canonical decision dates")
        calendar = xcals.get_calendar("XNYS")
        expected = tuple(value.date().isoformat() for value in calendar.sessions_in_range(parsed[0], parsed[-1]))
        if expected != session_calendar:
            raise ValueError("decision calendar must contain every XNYS session exactly once in order")
        first = calendar.session_offset(parsed[0], 1)
        last = calendar.session_offset(parsed[-1], horizon)
        return tuple(value.date().isoformat() for value in calendar.sessions_in_range(first, last))
    except (TypeError, ValueError, IndexError) as exc:
        raise DataReadinessError(f"invalid swing ledger calendar: {exc}") from exc


def _ledger_number(row: Mapping[str, Any], column: str) -> float:
    value = row[column]
    if isinstance(value, (bool, np.bool_, str)):
        raise DataReadinessError(f"ledger {column} must be a finite number, not a boolean or string")
    try:
        number = float(value)
        if not math.isfinite(number):
            raise DataReadinessError(f"ledger {column} must be finite")
        return number
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataReadinessError(f"ledger {column} must be finite") from exc


def build_funded_swing_ledger(
    selected: pd.DataFrame,
    config: SwingLedgerConfig,
    *,
    session_calendar: tuple[str, ...],
    additional_round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    """Replay one funded account in price-ratio units; never infer raw shares or dividends."""
    if (
        not isinstance(config.horizon_sessions, int)
        or isinstance(config.horizon_sessions, bool)
        or config.horizon_sessions != len(MANAGED_PATH_NET_RETURN_COLUMNS)
        or not isinstance(config.maximum_trades_per_decision, int)
        or isinstance(config.maximum_trades_per_decision, bool)
        or not 1 <= config.maximum_trades_per_decision <= 50
        or isinstance(config.expected_round_trip_cost_bps, (bool, np.bool_, str))
        or not math.isfinite(config.expected_round_trip_cost_bps)
        or config.expected_round_trip_cost_bps < 0
    ):
        raise DataReadinessError("ledger requires a ten-session horizon, bounded trade cap and finite nonnegative costs")
    session_dates = swing_valuation_sessions(session_calendar, config.horizon_sessions)
    ordinals = tuple(date.fromisoformat(value).toordinal() for value in session_dates)
    decision_ordinals = {date.fromisoformat(value).toordinal() for value in session_calendar}
    position_by_ordinal = {ordinal: index for index, ordinal in enumerate(ordinals)}
    if (
        isinstance(additional_round_trip_cost, (bool, np.bool_))
        or not math.isfinite(additional_round_trip_cost) or additional_round_trip_cost < 0.0
    ):
        raise DataReadinessError("additional ledger cost must be finite and non-negative")
    required = {
        "decision_id", "decision_group_id", "security_id", "sector", "session_date_et",
        "barrier_holding_sessions", "barrier_exit_session_date_et", "barrier_cost",
        "barrier_net_return", "barrier_gross_return",
        *MANAGED_PATH_SESSION_ORDINAL_COLUMNS, *MANAGED_PATH_NET_RETURN_COLUMNS,
    }
    if not selected.columns.is_unique:
        raise DataReadinessError("ledger input requires unique column names")
    if not required.issubset(selected.columns):
        raise DataReadinessError(f"ledger input is missing columns: {sorted(required.difference(selected.columns))}")
    for name in ("decision_id", "decision_group_id", "security_id", "sector", "session_date_et"):
        if not selected[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise DataReadinessError(f"ledger identity {name} must be present")
    if selected["decision_id"].duplicated().any() or selected.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("ledger requires unique decisions and one row per security/session")
    for group, other in (("decision_group_id", "session_date_et"), ("session_date_et", "decision_group_id")):
        if selected.groupby(group, observed=True)[other].nunique().gt(1).any():
            raise DataReadinessError("ledger requires one decision cohort per session")
    entries: dict[int, list[dict[str, Any]]] = {}
    original_cost = config.expected_round_trip_cost_bps / 10_000.0
    ordered = selected.sort_values(["session_date_et", "security_id", "decision_id"], kind="stable")
    for _, group in ordered.groupby("decision_group_id", observed=True, sort=False):
        if len(group) > config.maximum_trades_per_decision:
            raise DataReadinessError("ledger cohort exceeds the frozen trade cap")
        for row in group.to_dict(orient="records"):
            try:
                decision = date.fromisoformat(row["session_date_et"])
            except ValueError as exc:
                raise DataReadinessError("ledger decision date is invalid") from exc
            if decision.isoformat() != row["session_date_et"] or decision.toordinal() not in decision_ordinals:
                raise DataReadinessError("selected decision lies outside the frozen calendar")
            holding_value = _ledger_number(row, "barrier_holding_sessions")
            if not holding_value.is_integer() or not 1 <= holding_value <= config.horizon_sessions:
                raise DataReadinessError("ledger holding duration is invalid")
            holding = int(holding_value)
            cost = _ledger_number(row, "barrier_cost")
            if not math.isclose(cost, original_cost, rel_tol=0.0, abs_tol=1e-7):
                raise DataReadinessError("ledger label cost differs from the frozen original cost")
            path_ordinals = tuple(_ledger_number(row, name) for name in MANAGED_PATH_SESSION_ORDINAL_COLUMNS[:holding])
            if any(not value.is_integer() for value in path_ordinals):
                raise DataReadinessError("ledger path ordinals must be integral")
            entry = int(path_ordinals[0])
            entry_index = position_by_ordinal.get(entry)
            expected_entry_index = position_by_ordinal.get(decision.toordinal(), -1) + 1
            expected_path = ordinals[expected_entry_index:expected_entry_index + holding]
            if entry_index != expected_entry_index or tuple(map(int, path_ordinals)) != expected_path:
                raise DataReadinessError("ledger path has a gap, duplicate, wrong entry or missing tail")
            if str(row["barrier_exit_session_date_et"]) != date.fromordinal(int(path_ordinals[-1])).isoformat():
                raise DataReadinessError("ledger exit date differs from its terminal path")
            net_path = tuple(_ledger_number(row, name) for name in MANAGED_PATH_NET_RETURN_COLUMNS[:holding])
            net = _ledger_number(row, "barrier_net_return")
            gross = _ledger_number(row, "barrier_gross_return")
            if (
                not math.isclose(net_path[-1], net, rel_tol=1e-6, abs_tol=1e-7)
                or not math.isclose(gross - cost, net, abs_tol=1e-7)
            ):
                raise DataReadinessError("ledger terminal path/gross/net/cost do not reconcile")
            gross_path = tuple(value + cost for value in net_path)
            if any(value < -1.0 - 1e-7 for value in gross_path):
                raise DataReadinessError("ledger gross holding value cannot be negative")
            entries.setdefault(entry, []).append({
                "id": row["decision_id"], "security": row["security_id"], "sector": row["sector"],
                "weight": 1.0 / config.horizon_sessions / len(group),
                "ordinals": tuple(map(int, path_ordinals)), "gross_path": gross_path,
                "cost_rate": cost + additional_round_trip_cost,
            })
    cash = equity = peak = 1.0
    holdings = 0.0
    cumulative_cost = realized_gross = realized_cost = 0.0
    max_drawdown = turnover_sum = maximum_sector = maximum_gross = 0.0
    active: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    funded_trades = 0
    for ordinal, session in zip(ordinals, session_dates, strict=True):
        before = equity
        templates = entries.get(ordinal, [])
        requested = math.fsum(before * trade["weight"] * (1.0 + trade["cost_rate"]) for trade in templates)
        scale = min(1.0, cash / requested) if requested > 0 else 0.0
        entry_notional = daily_cost = exit_value = 0.0
        sector_pnl: dict[str, float] = {}
        for template in templates:
            notional = before * template["weight"] * scale
            if notional <= 0:
                continue
            trade = {**template, "notional": notional, "value": notional, "step": 0}
            fee = notional * trade["cost_rate"]
            trade["paid_cost"] = fee
            entry_notional += notional
            daily_cost += fee
            sector = trade["sector"]
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) - fee
            active.append(trade)
            funded_trades += 1
        debit = entry_notional + daily_cost
        if debit > cash + 1e-12:
            raise DataReadinessError("ledger attempted an unfunded purchase")
        cash = max(0.0, cash - debit)
        cumulative_cost += daily_cost
        sector_values: dict[str, float] = {}
        security_values: dict[str, float] = {}
        remaining: list[dict[str, Any]] = []
        for trade in active:
            step = trade["step"]
            if trade["ordinals"][step] != ordinal:
                raise DataReadinessError("active holding lacks an exact daily mark")
            value = trade["notional"] * max(0.0, 1.0 + trade["gross_path"][step])
            sector = trade["sector"]
            security = trade["security"]
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + value - trade["value"]
            sector_values[sector] = sector_values.get(sector, 0.0) + value
            security_values[security] = security_values.get(security, 0.0) + value
            trade["value"] = value
            trade["step"] = step + 1
            if trade["step"] == len(trade["ordinals"]):
                exit_value += value
                realized_gross += value - trade["notional"]
                realized_cost += trade["paid_cost"]
            else:
                remaining.append(trade)
        # Exit proceeds cannot fund this session's earlier entries.
        cash += exit_value
        active = remaining
        holdings = math.fsum(trade["value"] for trade in active)
        unrealized_gross = math.fsum(trade["value"] - trade["notional"] for trade in active)
        equity = cash + holdings
        if not math.isfinite(equity) or equity <= 0 or cash < 0:
            raise DataReadinessError("funded ledger produced invalid cash/equity")
        if not math.isclose(equity, 1.0 + realized_gross + unrealized_gross - cumulative_cost, abs_tol=1e-10):
            raise DataReadinessError("cash, holdings and cumulative P&L fail reconciliation")
        if not math.isclose(equity - before, math.fsum(sector_pnl.values()), abs_tol=1e-10):
            raise DataReadinessError("sector contributions fail daily NAV reconciliation")
        exposure = math.fsum(sector_values.values()) / equity
        if exposure > 1.0 + 1e-10:
            raise DataReadinessError("funded ledger exceeded unlevered gross exposure")
        maximum_gross = max(maximum_gross, exposure)
        maximum_sector = max(maximum_sector, max(sector_values.values(), default=0.0) / equity)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        turnover_sum += (entry_notional + exit_value) / before
        daily_records.append({
            "session_date_et": session, "cash": cash, "holdings": holdings, "equity": equity,
            "net_return": equity / before - 1.0, "gross_exposure": exposure,
            "entry_notional": entry_notional, "exit_value": exit_value, "cost": daily_cost,
            "requested_purchase_and_cost": requested, "funding_scale": scale,
            "realized_gross_pnl": realized_gross, "unrealized_gross_pnl": unrealized_gross,
            "realized_net_pnl": realized_gross - realized_cost,
            "unrealized_net_pnl": unrealized_gross - (cumulative_cost - realized_cost),
            "cumulative_cost": cumulative_cost, "sector_pnl": sector_pnl,
            "marked_sector_values_before_exits": sector_values,
            "marked_security_values_before_exits": security_values,
        })
    if active or holdings != 0.0:
        raise DataReadinessError("fixed maturation tail did not close every selected lot")
    return {
        "sessions": len(session_dates), "session_dates": list(session_dates),
        "compounded_return": equity - 1.0, "max_drawdown": max_drawdown,
        "average_daily_turnover": turnover_sum / len(session_dates),
        "maximum_sector_weight": maximum_sector, "maximum_gross_exposure": maximum_gross,
        "daily_returns": [record["net_return"] for record in daily_records],
        "daily_records": daily_records, "final_cash": cash, "final_holdings": holdings,
        "total_cost": cumulative_cost, "selected_trades": len(selected), "funded_trades": funded_trades,
        "units": "adjusted_price_ratio_units_not_raw_executable_shares",
        "distribution_policy": "no_separate_cash_distribution_credit",
        "sector_attribution_basis": "sector_known_at_each_lot_decision",
    }
