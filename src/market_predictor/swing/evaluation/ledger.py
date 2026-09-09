"""Canonical cash-funded swing evaluation; no training-orchestrator dependencies."""
from __future__ import annotations

import math
from collections.abc import Iterable, Mapping
from datetime import date
from typing import Any, Literal, Protocol

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_accounting import ExecutionEvent, HoldingSpecification, PaymentEvent
from market_predictor.swing.contracts.trade_simulation import TradeSimulationContext
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales, simulation_metadata, simulation_replay_metadata
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
    """Explicitly unadmitted price-ratio diagnostic using the shared funding loop."""
    _validate_ledger_config(config, additional_round_trip_cost)
    session_dates = swing_valuation_sessions(session_calendar, config.horizon_sessions)
    ordinals = tuple(date.fromisoformat(value).toordinal() for value in session_dates)
    decision_ordinals = {date.fromisoformat(value).toordinal() for value in session_calendar}
    position_by_ordinal = {ordinal: index for index, ordinal in enumerate(ordinals)}
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
                "ordinals": tuple(map(int, path_ordinals)),
                "steps": tuple({
                    "tradable": max(0.0, value + 1.0) if index < holding - 1 else 0.0,
                    "unpaid": 0.0, "contingent": 0.0,
                    "cash_before_open": 0.0,
                    "released_cash": max(0.0, value + 1.0) if index == holding - 1 else 0.0,
                    "sale_proceeds": max(0.0, value + 1.0) if index == holding - 1 else 0.0,
                    "sale_cash_at_exit": max(0.0, value + 1.0) if index == holding - 1 else 0.0,
                    "fully_settled": index == holding - 1, "gaps": (),
                } for index, value in enumerate(gross_path)),
                "cost_rate": cost + additional_round_trip_cost,
            })
    return _replay_funded_entries(
        entries, session_dates=session_dates, selected_trades=len(selected),
        units="adjusted_price_ratio_units_not_raw_executable_shares",
        distribution_policy="no_separate_cash_distribution_credit",
    )


def _validate_ledger_config(config: SwingLedgerConfig, additional_round_trip_cost: float) -> None:
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
    if (
        isinstance(additional_round_trip_cost, (bool, np.bool_))
        or not math.isfinite(additional_round_trip_cost) or additional_round_trip_cost < 0.0
    ):
        raise DataReadinessError("additional ledger cost must be finite and non-negative")


def build_event_aware_funded_swing_ledger(
    selected: pd.DataFrame, holdings: Iterable[HoldingSpecification], config: SwingLedgerConfig, *,
    session_calendar: tuple[str, ...], research_contract_sha256: str,
    execution_policy: Literal["fixed_horizon", "managed"], additional_round_trip_cost: float = 0.0,
    simulation: TradeSimulationContext | None = None,
) -> dict[str, Any]:
    """Replay exact selected lots through the shared holding kernel and funding loop.

    This is arithmetic verification, not source or promotion admission. Missing
    component marks stop NAV-dependent allocations instead of silently dropping lots.
    """
    _validate_ledger_config(config, additional_round_trip_cost)
    if execution_policy not in {"fixed_horizon", "managed"}:
        raise DataReadinessError("unsupported event-aware execution policy")
    names = ("decision_id", "security_id", "sector", "session_date_et", "decision_group_id")
    if not selected.columns.is_unique or not set(names).issubset(selected.columns):
        raise DataReadinessError("event ledger requires unique selection identity columns")
    for name in names:
        if not selected[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all():
            raise DataReadinessError("event ledger selection identities must be complete")
    if selected.decision_id.duplicated().any() or selected.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("event ledger contains duplicate selections")
    for group, other in (("decision_group_id", "session_date_et"), ("session_date_et", "decision_group_id")):
        if selected.groupby(group, observed=True)[other].nunique().gt(1).any():
            raise DataReadinessError("event ledger requires one cohort per session")
    sizes = selected.groupby("session_date_et", observed=True).size().to_dict()
    if any(size > config.maximum_trades_per_decision for size in sizes.values()):
        raise DataReadinessError("event ledger exceeds the frozen trade cap")
    selected_by_id = {row["decision_id"]: row for row in selected.loc[:, list(names)].to_dict(orient="records")}
    simulation_replays: dict[str, dict[str, object]] = {}
    dates = swing_valuation_sessions(session_calendar, config.horizon_sessions)
    calendar = xcals.get_calendar("XNYS")
    closes = tuple(calendar.session_close(session).to_pydatetime() for session in dates)
    opens = tuple(calendar.session_open(session).to_pydatetime() for session in dates)
    ordinals = tuple(date.fromisoformat(session).toordinal() for session in dates)
    entries: dict[int, list[dict[str, Any]]] = {}
    seen: set[str] = set()
    for spec in holdings:
        assert_memory_budget(stage="event-aware lot replay", hard_budget_gib=5.0, headroom_gib=0.75)
        # Revalidate even instances built through unchecked model construction.
        spec = HoldingSpecification.model_validate_json(spec.model_dump_json())
        row = selected_by_id.get(spec.decision_id)
        if (row is None or spec.decision_id in seen or spec.security_id != row["security_id"]
                or spec.sector != row["sector"] or spec.policy != execution_policy
                or spec.research_contract_sha256 != research_contract_sha256
                or spec.cost_prepaid_fraction != config.expected_round_trip_cost_bps / 10_000):
            raise DataReadinessError("holding and selected decision/policy identities differ")
        seen.add(spec.decision_id)
        if row["session_date_et"] not in session_calendar:
            raise DataReadinessError("event selection lies outside the decision calendar")
        entry_date = calendar.session_offset(row["session_date_et"], 1).date().isoformat()
        offset = dates.index(entry_date)
        if spec.initial_entry_timestamp != opens[offset]:
            raise DataReadinessError("event ledger entry is not next-session open")
        ends = spec.session_end_timestamps
        if ends != closes[offset:offset + len(ends)]:
            raise DataReadinessError("event ledger snapshots differ from exact portfolio sessions")
        generated_ids: set[str] = set()
        if simulation is not None:
            simulated = simulate_ordinary_sales(spec, simulation)
            spec, outcome = simulated.specification, simulated.outcome
            simulation_replays[spec.decision_id] = simulation_replay_metadata(simulated)
            generated_ids = set(simulated.generated_event_ids)
        else:
            outcome = replay_holding(spec)
        sale_by_claim = {event.proceeds_id: event.event_id for event in spec.events if isinstance(event, ExecutionEvent)}
        sale_by_payment = {event.event_id: sale_by_claim.get(event.claim_id)
            for event in spec.events if isinstance(event, PaymentEvent)}
        sale_times = {item.source_event_id: item.executed_at for item in outcome.executed_sales}
        previous_cash = 0.0
        previous_close = spec.initial_entry_timestamp
        steps = []
        for index, snapshot in enumerate(outcome.snapshots):
            end = snapshot.session_end_timestamp
            releases = tuple(item for item in outcome.cash_releases if previous_close < item.available_at <= end)
            released = math.fsum(item.amount for item in releases)
            if not math.isclose(snapshot.cumulative_cash_released - previous_cash, released, abs_tol=1e-10):
                raise DataReadinessError("cash release receipts do not reconcile with holding snapshots")
            before_open = math.fsum(item.amount for item in releases
                if (item.available_at < opens[offset + index] or (
                    item.available_at == opens[offset + index] and item.availability_event_id in generated_ids
                    and item.basis == "research_assumption" and item.source_kind == "sale_proceeds"
                )) and (
                    item.source_kind == "corporate_payment"
                    or sale_times.get(sale_by_payment.get(item.payment_event_id) or "", end) <= previous_close
                ))
            sale_proceeds = math.fsum(item.normalized_proceeds for item in outcome.executed_sales
                if previous_close < item.executed_at <= end)
            sale_ids = {item.source_event_id for item in outcome.executed_sales
                if previous_close < item.executed_at <= end}
            sale_cash = math.fsum(item.amount for item in releases
                if item.source_kind == "sale_proceeds" and sale_by_payment.get(item.payment_event_id) in sale_ids)
            steps.append({
                "tradable": snapshot.tradable_value, "unpaid": snapshot.unpaid_proceeds_value,
                "contingent": snapshot.contingent_right_value, "cash_before_open": before_open,
                "released_cash": released, "sale_proceeds": sale_proceeds,
                "sale_cash_at_exit": sale_cash,
                "fully_settled": snapshot.fully_settled,
                "gaps": tuple(item.model_dump(mode="json") for item in snapshot.gaps),
            })
            previous_cash = snapshot.cumulative_cash_released
            previous_close = end
        if not outcome.fully_settled and len(ends) != len(closes) - offset:
            raise DataReadinessError("residual claims require observations through the portfolio endpoint")
        entries.setdefault(ordinals[offset], []).append({
            "id": spec.decision_id, "security": spec.security_id, "sector": spec.sector,
            "weight": 1.0 / config.horizon_sessions / sizes[row["session_date_et"]],
            "ordinals": ordinals[offset:offset + len(steps)], "steps": tuple(steps),
            "cost_rate": spec.cost_prepaid_fraction + additional_round_trip_cost,
        })
    if seen != set(selected_by_id):
        raise DataReadinessError("selected decisions are missing event-aware holding specifications")
    for entry_group in entries.values():
        entry_group.sort(key=lambda item: (item["security"], item["id"]))
    result = _replay_funded_entries(
        entries, session_dates=dates, selected_trades=len(selected),
        units="raw_share_entitlements_per_entry_notional",
        distribution_policy="cash_only_on_evidenced_availability_residual_claims_retained",
    )
    result["trade_simulation"] = simulation_metadata(simulation) if simulation else None
    result["simulation_replays"] = simulation_replays
    return result


def _replay_funded_entries(
    entries: dict[int, list[dict[str, Any]]], *, session_dates: tuple[str, ...],
    selected_trades: int, units: str, distribution_policy: str,
) -> dict[str, Any]:
    """One funding loop for explicit event paths and retained price-ratio diagnostics."""
    ordinals = tuple(date.fromisoformat(value).toordinal() for value in session_dates)
    cash = equity = peak = 1.0
    holdings = 0.0
    cumulative_cost = realized_gross = realized_cost = 0.0
    max_drawdown = turnover_sum = maximum_sector = maximum_gross = 0.0
    active: list[dict[str, Any]] = []
    daily_records: list[dict[str, Any]] = []
    funded_trades = 0
    for ordinal, session in zip(ordinals, session_dates, strict=True):
        before = equity
        # Only independently timed corporate releases may fund this open.
        cash += math.fsum(
            trade["notional"] * trade["steps"][trade["step"]]["cash_before_open"] for trade in active
        )
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
        component_values = {"tradable": 0.0, "unpaid": 0.0, "contingent": 0.0}
        unavailable: list[dict[str, Any]] = []
        for trade in active:
            step = trade["step"]
            if trade["ordinals"][step] != ordinal:
                raise DataReadinessError("active holding lacks an exact daily mark")
            observation = trade["steps"][step]
            released = trade["notional"] * observation["released_cash"]
            cash += released - trade["notional"] * observation["cash_before_open"]
            exit_value += trade["notional"] * observation["sale_proceeds"]
            if observation["gaps"] or any(observation[name] is None for name in component_values):
                unavailable.append({"decision_id": trade["id"], "gaps": observation["gaps"],
                    "component_marks_per_entry_unit": {name: observation[name] for name in component_values}})
                continue
            components = {name: trade["notional"] * observation[name] for name in component_values}
            for name, amount in components.items():
                if not math.isfinite(amount) or amount < 0:
                    raise DataReadinessError("holding component is negative or non-finite")
                component_values[name] += amount
            value = math.fsum(components.values())
            sector = trade["sector"]
            security = trade["security"]
            sector_pnl[sector] = sector_pnl.get(sector, 0.0) + value + released - trade["value"]
            before_exit_value = value + trade["notional"] * observation["sale_cash_at_exit"]
            sector_values[sector] = sector_values.get(sector, 0.0) + before_exit_value
            security_values[security] = security_values.get(security, 0.0) + before_exit_value
            trade["value"] = value
            trade["released"] = trade.get("released", 0.0) + released
            trade["step"] = step + 1
            if observation["fully_settled"]:
                if value != 0.0:
                    raise DataReadinessError("fully settled lot still contains noncash value")
                realized_gross += trade["released"] - trade["notional"]
                realized_cost += trade["paid_cost"]
            else:
                if trade["step"] == len(trade["ordinals"]) and ordinal != ordinals[-1]:
                    raise DataReadinessError("residual claims lack marks through the portfolio endpoint")
                remaining.append(trade)
        if unavailable:
            return {
                "status": "valuation_unavailable", "accounting_eligible": False,
                "blocked_session": session, "gaps": unavailable,
                "sessions": len(session_dates), "session_dates": list(session_dates),
                "daily_records": daily_records, "daily_returns": None,
                "compounded_return": None, "max_drawdown": None,
                "final_cash": None, "final_holdings": None, "known_cash_at_block": cash,
                "total_cost": cumulative_cost, "selected_trades": selected_trades,
                "funded_trades": funded_trades, "fully_settled": False,
                "units": units, "distribution_policy": distribution_policy,
            }
        active = remaining
        holdings = math.fsum(trade["value"] for trade in active)
        unrealized_gross = math.fsum(trade["value"] + trade.get("released", 0.0) - trade["notional"] for trade in active)
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
            "tradable_holdings": component_values["tradable"],
            "unpaid_proceeds": component_values["unpaid"],
            "contingent_rights": component_values["contingent"],
            "component_exposure_at_close": {name: value / equity for name, value in component_values.items()},
        })
    return {
        "status": "computed", "accounting_eligible": False, "fully_settled": not active,
        "sessions": len(session_dates), "session_dates": list(session_dates),
        "compounded_return": equity - 1.0, "max_drawdown": max_drawdown,
        "average_daily_turnover": turnover_sum / len(session_dates),
        "maximum_sector_weight": maximum_sector, "maximum_gross_exposure": maximum_gross,
        "daily_returns": [record["net_return"] for record in daily_records],
        "daily_records": daily_records, "final_cash": cash, "final_holdings": holdings,
        "total_cost": cumulative_cost, "selected_trades": selected_trades, "funded_trades": funded_trades,
        "final_tradable_holdings": component_values["tradable"],
        "final_unpaid_proceeds": component_values["unpaid"],
        "final_contingent_rights": component_values["contingent"],
        "residual_decision_ids": [trade["id"] for trade in active],
        "units": units,
        "distribution_policy": distribution_policy,
        "sector_attribution_basis": "sector_known_at_each_lot_decision",
    }
