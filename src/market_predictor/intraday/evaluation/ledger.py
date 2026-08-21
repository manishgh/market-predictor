from __future__ import annotations

import math
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.training.config import IntradayDevelopmentConfig


def _position_ledger(
    scored: pd.DataFrame,
    threshold_bps: float,
    maximum_stop_probability: float,
    cost_bps: float,
    config: IntradayDevelopmentConfig,
) -> dict[str, Any]:
    threshold = threshold_bps / 10_000.0
    equity = 1.0
    positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    equity_marks: list[float] = [equity]
    for session, session_frame in scored.groupby("session_date_et", sort=True, observed=True):
        start_equity = equity
        cash = equity
        open_positions: list[dict[str, Any]] = []
        cooldown: dict[str, pd.Timestamp] = {}

        for _, group in session_frame.groupby("decision_group_id", sort=True, observed=True):
            entry_times = pd.to_datetime(group["entry_time_utc"], utc=True).unique()
            if len(entry_times) != 1:
                raise DataReadinessError("decision group must have one executable entry time")
            entry_time = pd.Timestamp(entry_times[0])
            cash, equity = _close_due_positions(
                open_positions,
                cutoff=entry_time,
                cash=cash,
                cost_bps=cost_bps,
                cooldown_minutes=config.per_security_cooldown_minutes,
                cooldown=cooldown,
                completed=positions,
                equity_marks=equity_marks,
            )
            candidates = group.loc[
                group["predicted_net_return"].ge(threshold) & group["predicted_stop_probability"].le(maximum_stop_probability)
            ].sort_values(["predicted_net_return", "security_id"], ascending=[False, True], kind="stable")
            candidates = candidates.head(config.maximum_candidates_per_decision)
            for row in candidates.itertuples(index=False):
                security_id = str(row.security_id)
                if len(open_positions) >= config.maximum_concurrent_positions:
                    break
                if any(str(item["security_id"]) == security_id for item in open_positions):
                    continue
                if cooldown.get(security_id, pd.Timestamp.min.tz_localize("UTC")) > entry_time:
                    continue
                notional = min(config.position_weight * equity, cash)
                if notional <= 1e-12:
                    break
                cash -= notional
                open_positions.append(
                    {
                        "dataset_row_id": str(row.dataset_row_id),
                        "ticker": str(row.ticker),
                        "security_id": security_id,
                        "session_date_et": str(session),
                        "decision_group_id": str(row.decision_group_id),
                        "entry_time_utc": entry_time,
                        "exit_time_utc": pd.Timestamp(row.exit_bar_end_utc),
                        "predicted_net_return": float(row.predicted_net_return),
                        "predicted_stop_probability": float(row.predicted_stop_probability),
                        "gross_return": float(row.gross_return),
                        "spy_return": float(row.spy_return),
                        "qqq_return": float(row.qqq_return),
                        "sector_return": float(row.sector_return),
                        "entry_price": float(row.entry_price),
                        "stop_price": float(row.stop_price),
                        "fold": int(row.fold),
                        "notional": notional,
                        "entry_weight": notional / equity,
                        "round_trip_cost_bps": cost_bps,
                    }
                )
        cash, equity = _close_due_positions(
            open_positions,
            cutoff=None,
            cash=cash,
            cost_bps=cost_bps,
            cooldown_minutes=config.per_security_cooldown_minutes,
            cooldown=cooldown,
            completed=positions,
            equity_marks=equity_marks,
        )
        if open_positions:
            raise AssertionError("session ended with open intraday positions")
        end_equity = cash
        equity = end_equity
        session_positions = [row for row in positions if row["session_date_et"] == str(session)]
        daily_rows.append(
            {
                "session_date_et": str(session),
                "fold": int(session_frame["fold"].iloc[0]),
                "starting_equity": start_equity,
                "ending_equity": end_equity,
                "daily_return": end_equity / start_equity - 1.0,
                "spy_excess_return": sum(float(row["notional"]) * float(row["realized_spy_excess_return"]) for row in session_positions)
                / start_equity,
                "qqq_excess_return": sum(float(row["notional"]) * float(row["realized_qqq_excess_return"]) for row in session_positions)
                / start_equity,
                "sector_excess_return": sum(
                    float(row["notional"]) * float(row["realized_sector_excess_return"]) for row in session_positions
                )
                / start_equity,
                "entries": len(session_positions),
                "entry_notional": sum(float(row["notional"]) for row in session_positions),
            }
        )
        equity_marks.append(equity)
    daily_returns = np.asarray([float(row["daily_return"]) for row in daily_rows], dtype="float64")
    return {
        "position_records": positions,
        "daily_records": daily_rows,
        "positions": len(positions),
        "daily_rows": len(daily_rows),
        "daily_returns": daily_returns,
        "equity_marks": np.asarray(equity_marks, dtype="float64"),
    }


def _close_due_positions(
    open_positions: list[dict[str, Any]],
    *,
    cutoff: pd.Timestamp | None,
    cash: float,
    cost_bps: float,
    cooldown_minutes: int,
    cooldown: dict[str, pd.Timestamp],
    completed: list[dict[str, Any]],
    equity_marks: list[float],
) -> tuple[float, float]:
    due = [item for item in open_positions if cutoff is None or item["exit_time_utc"] <= cutoff]
    equity = cash + sum(_conservative_open_value(item, cost_bps) for item in open_positions)
    if open_positions:
        equity_marks.append(equity)
    exit_times = sorted({pd.Timestamp(item["exit_time_utc"]) for item in due})
    for exit_time in exit_times:
        batch = sorted(
            (item for item in due if pd.Timestamp(item["exit_time_utc"]) == exit_time),
            key=lambda value: str(value["security_id"]),
        )
        realized: list[tuple[dict[str, Any], float, float]] = []
        for item in batch:
            realized_return = float(item["gross_return"]) - cost_bps / 10_000.0
            pnl = float(item["notional"]) * realized_return
            cash += float(item["notional"]) + pnl
            open_positions.remove(item)
            cooldown[str(item["security_id"])] = exit_time + pd.Timedelta(minutes=cooldown_minutes)
            realized.append((item, realized_return, pnl))
        equity = cash + sum(_conservative_open_value(opened, cost_bps) for opened in open_positions)
        for item, realized_return, pnl in realized:
            item["realized_net_return"] = realized_return
            item["realized_spy_excess_return"] = realized_return - float(item["spy_return"])
            item["realized_qqq_excess_return"] = realized_return - float(item["qqq_return"])
            item["realized_sector_excess_return"] = realized_return - float(item["sector_return"])
            item["pnl"] = pnl
            item["equity_after_exit"] = equity
            completed.append(item)
        equity_marks.append(equity)
    return cash, equity


def _conservative_open_value(position: Mapping[str, Any], cost_bps: float) -> float:
    entry = float(position["entry_price"])
    stop = float(position["stop_price"])
    stop_return = stop / entry - 1.0 - cost_bps / 10_000.0
    return float(position["notional"]) * (1.0 + min(0.0, stop_return))


def _ledger_metrics(ledger: Mapping[str, Any]) -> dict[str, Any]:
    positions = list(ledger["position_records"])
    daily = list(ledger["daily_records"])
    returns = np.asarray(ledger["daily_returns"], dtype="float64")
    marks = np.asarray(ledger["equity_marks"], dtype="float64")
    notionals = np.asarray([float(row["notional"]) for row in positions], dtype="float64")
    pnls = np.asarray([float(row["pnl"]) for row in positions], dtype="float64")
    average_equity = float(np.mean([float(row["starting_equity"]) for row in daily])) if daily else 1.0
    profit = float(pnls[pnls > 0.0].sum()) if len(pnls) else 0.0
    loss = float(pnls[pnls < 0.0].sum()) if len(pnls) else 0.0
    running_peak = np.maximum.accumulate(marks)
    drawdown = 1.0 - marks / running_peak
    concurrency = _maximum_observed_concurrency(positions)
    sessions = len(daily)
    round_trip_turnover = float(2.0 * notionals.sum() / average_equity) if average_equity > 0.0 else 0.0
    compounded = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    fold_means = pd.DataFrame(daily).groupby("fold", observed=True)["daily_return"].mean() if daily else pd.Series(dtype="float64")
    return {
        "trade_count": len(positions),
        "sessions": sessions,
        "sessions_with_trades": sum(int(row["entries"]) > 0 for row in daily),
        "average_trade_net_return": float(pnls.sum() / notionals.sum()) if notionals.sum() > 0.0 else 0.0,
        "average_daily_net_return": float(returns.mean()) if len(returns) else 0.0,
        "compounded_net_return": compounded,
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "profit_factor": profit / abs(loss) if loss < 0.0 else (1_000_000.0 if profit > 0.0 else 0.0),
        "win_rate": float((pnls > 0.0).mean()) if len(pnls) else 0.0,
        "negative_session_rate": float((returns < 0.0).mean()) if len(returns) else 1.0,
        "return_to_drawdown": compounded / float(drawdown.max()) if len(drawdown) and drawdown.max() > 0.0 else 0.0,
        "average_spy_excess_return": _notional_weighted_position_mean(positions, notionals, "realized_spy_excess_return"),
        "average_qqq_excess_return": _notional_weighted_position_mean(positions, notionals, "realized_qqq_excess_return"),
        "average_sector_excess_return": _notional_weighted_position_mean(positions, notionals, "realized_sector_excess_return"),
        "one_way_turnover": float(notionals.sum() / average_equity) if average_equity > 0.0 else 0.0,
        "round_trip_turnover": round_trip_turnover,
        "average_daily_round_trip_turnover": round_trip_turnover / sessions if sessions else math.inf,
        "profitable_fold_fraction": float(fold_means.gt(0.0).mean()) if len(fold_means) else 0.0,
        "maximum_concurrent_positions_observed": concurrency,
        "maximum_entries_per_session_observed": max((int(row["entries"]) for row in daily), default=0),
        "maximum_entries_per_decision_observed": _maximum_entries_per_decision(positions),
        "maximum_entry_weight_observed": max(
            (float(row["entry_weight"]) for row in positions),
            default=0.0,
        ),
        "maximum_concurrent_positions_enforced": True,
        "capital_weights_enforced": True,
        "security_cooldown_enforced": True,
    }


def _maximum_entries_per_decision(positions: Sequence[Mapping[str, Any]]) -> int:
    counts: dict[tuple[str, str], int] = {}
    for position in positions:
        key = (str(position["session_date_et"]), str(position["decision_group_id"]))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def _notional_weighted_position_mean(
    positions: Sequence[Mapping[str, Any]],
    notionals: np.ndarray,
    column: str,
) -> float:
    if not positions or notionals.sum() <= 0.0:
        return 0.0
    values = np.asarray([float(row[column]) for row in positions], dtype="float64")
    return float(np.average(values, weights=notionals))


def _maximum_observed_concurrency(positions: Sequence[Mapping[str, Any]]) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    for position in positions:
        events.append((pd.Timestamp(position["entry_time_utc"]), 1))
        events.append((pd.Timestamp(position["exit_time_utc"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum
