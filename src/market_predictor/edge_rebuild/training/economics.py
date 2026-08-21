from __future__ import annotations

import hashlib
import math
from collections.abc import Mapping
from datetime import date
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
)

if TYPE_CHECKING:
    from market_predictor.edge_rebuild.training.swing_types import SwingTrainingConfig
from market_predictor.edge_rebuild.training.utils import _finite, _mapping
from market_predictor.v3.errors import DataReadinessError


def _session_economic_blocks(
    selected: pd.DataFrame,
    *,
    session_calendar: tuple[str, ...],
) -> list[dict[str, Any]]:
    columns = [
        "barrier_net_return",
        "approx_managed_exit_session_close_excess_vs_spy",
        "approx_managed_exit_session_close_excess_vs_qqq",
        "approx_managed_exit_session_close_excess_vs_sector",
    ]
    grouped = (
        selected.groupby("session_date_et", as_index=False, sort=True, observed=True)[columns]
        .mean()
    )
    calendar = pd.DataFrame({"session_date_et": list(session_calendar)})
    complete = calendar.merge(
        grouped,
        on="session_date_et",
        how="left",
        validate="one_to_one",
    )
    complete[columns] = complete[columns].fillna(0.0)
    return cast(list[dict[str, Any]], complete.to_dict(orient="records"))


def _economic_gate(
    metrics: Mapping[str, Any],
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    bootstrap = _mapping(metrics.get("moving_block_bootstrap_95_ci"), "bootstrap")
    calendar_net_ci = _mapping(
        bootstrap.get("calendar_average_managed_net_return"),
        "calendar net CI",
    )
    portfolio_ci = _mapping(
        bootstrap.get("portfolio_daily_return"),
        "portfolio daily CI",
    )
    stress_portfolio_ci = _mapping(
        bootstrap.get("double_cost_portfolio_daily_return"),
        "double-cost portfolio daily CI",
    )
    excess_lows = [
        _finite(
            _mapping(
                bootstrap.get(
                    f"calendar_average_managed_exit_session_close_{name}_excess"
                ),
                f"managed {name} CI",
            ),
            "low",
        )
        for name in ("spy", "qqq", "sector")
    ]
    minimum_edge = strategy_contract.swing.minimum_expected_net_edge_bps / 10_000.0
    stress_multiplier = strategy_contract.stress.cost_multiplier
    checks = {
        "conditional_trade_mean_net_return_at_least_minimum_edge": (
            _finite(metrics, "selected_average_managed_net_return") >= minimum_edge
        ),
        "calendar_entry_cohort_net_ci_low_positive": (
            _finite(calendar_net_ci, "low") > 0.0
        ),
        "portfolio_daily_return_ci_low_positive": _finite(portfolio_ci, "low") > 0.0,
        "worst_holding_aligned_benchmark_ci_low_positive": min(excess_lows) > 0.0,
        "double_cost_portfolio_daily_ci_low_positive": (
            _finite(stress_portfolio_ci, "low") > 0.0
        ),
        "active_portfolio_sector_weight_at_or_below_hard_maximum": (
            _finite(metrics, "maximum_observed_sector_weight")
            <= strategy_contract.swing.hard_maximum_sector_weight + 1e-12
        ),
    }
    return {
        "passed": all(checks.values()),
        "checks": checks,
        "minimum_expected_net_edge_bps": strategy_contract.swing.minimum_expected_net_edge_bps,
        "stress_cost_multiplier": stress_multiplier,
        "stress_portfolio_daily_return": _finite(
            metrics,
            "double_cost_portfolio_daily_average_return",
        ),
        "stress_portfolio_daily_return_ci_low": _finite(
            stress_portfolio_ci,
            "low",
        ),
    }


def _daily_position_ledger(
    selected: pd.DataFrame,
    config: SwingTrainingConfig,
    *,
    session_calendar: tuple[str, ...],
    additional_round_trip_cost: float = 0.0,
) -> dict[str, Any]:
    trades: list[dict[str, Any]] = []
    for _, group in selected.groupby("decision_group_id", sort=False, observed=True):
        cohort_weight = 1.0 / config.horizon_sessions
        trade_weight = cohort_weight / len(group)
        for _, row in group.iterrows():
            holding = int(row["barrier_holding_sessions"])
            trades.append({
                "entry": int(row[MANAGED_PATH_SESSION_ORDINAL_COLUMNS[0]]),
                "holding": holding,
                "weight": trade_weight,
                "sector": str(row["sector"]),
                "ordinals": tuple(int(row[column]) for column in MANAGED_PATH_SESSION_ORDINAL_COLUMNS[:holding]),
                "path": tuple(
                    float(row[column]) - additional_round_trip_cost
                    for column in MANAGED_PATH_NET_RETURN_COLUMNS[:holding]
                ),
            })
    if not trades:
        raise DataReadinessError("daily ledger requires selected trades")
    entry_groups: dict[int, list[dict[str, Any]]] = {}
    for trade in trades:
        entry_groups.setdefault(int(trade["entry"]), []).append(trade)
    calendar_ordinals = {
        date.fromisoformat(value).toordinal() for value in session_calendar
    }
    sessions = sorted(
        calendar_ordinals.union(
            {ordinal for trade in trades for ordinal in trade["ordinals"]}
        )
    )
    equity = 1.0
    peak = 1.0
    max_drawdown = 0.0
    turnover_sum = 0.0
    max_sector_weight = 0.0
    active: list[dict[str, Any]] = []
    daily_returns: list[float] = []
    for ordinal in sessions:
        equity_before = equity
        for template in entry_groups.get(ordinal, []):
            trade = dict(template)
            trade["notional"] = equity_before * float(trade["weight"])
            trade["previous"] = 0.0
            trade["step"] = 0
            active.append(trade)
            turnover_sum += float(trade["notional"]) / max(equity_before, 1e-12)
        pnl = 0.0
        exits: list[dict[str, Any]] = []
        sector_values: dict[str, float] = {}
        for trade in active:
            step = int(trade["step"])
            ordinals = cast(tuple[int, ...], trade["ordinals"])
            if step >= len(ordinals) or ordinals[step] != ordinal:
                continue
            path = cast(tuple[float, ...], trade["path"])
            current = path[step]
            pnl += float(trade["notional"]) * (current - float(trade["previous"]))
            trade["previous"] = current
            trade["step"] = step + 1
            value = float(trade["notional"]) * (1.0 + current)
            sector = str(trade["sector"])
            sector_values[sector] = sector_values.get(sector, 0.0) + value
            if step + 1 == int(trade["holding"]):
                turnover_sum += value / max(equity_before, 1e-12)
                exits.append(trade)
        equity += pnl
        if equity <= 0 or not math.isfinite(equity):
            raise DataReadinessError("daily portfolio ledger produced invalid equity")
        active = [trade for trade in active if trade not in exits]
        daily_returns.append((equity - equity_before) / equity_before)
        peak = max(peak, equity)
        max_drawdown = max(max_drawdown, 1.0 - equity / peak)
        if sector_values:
            max_sector_weight = max(
                max_sector_weight,
                max(sector_values.values()) / equity,
            )
    return {
        "sessions": len(sessions),
        "compounded_return": equity - 1.0,
        "max_drawdown": max_drawdown,
        "average_daily_turnover": turnover_sum / len(sessions),
        "maximum_sector_weight": max_sector_weight,
        "daily_returns": daily_returns,
    }


def _stability_breakdown(selected: pd.DataFrame, column: str) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for value, group in selected.groupby(column, sort=True, observed=True):
        records.append(
            {
                "value": str(value),
                "sessions": int(group["session_date_et"].nunique()),
                "trades": len(group),
                "average_managed_net_return": float(group["barrier_net_return"].mean()),
                "win_rate_after_costs": float(group["barrier_net_return"].gt(0).mean()),
                "average_exact_10_session_spy_excess": float(group["future_excess_return_10d_vs_spy"].mean()),
                "average_exact_10_session_qqq_excess": float(group["future_excess_return_10d_vs_qqq"].mean()),
                "average_exact_10_session_sector_excess": float(group["future_excess_return_10d_vs_sector"].mean()),
                "diagnostic_approx_managed_exit_session_close_spy_excess": float(
                    group["approx_managed_exit_session_close_excess_vs_spy"].mean()
                ),
                "diagnostic_approx_managed_exit_session_close_qqq_excess": float(
                    group["approx_managed_exit_session_close_excess_vs_qqq"].mean()
                ),
                "diagnostic_approx_managed_exit_session_close_sector_excess": float(
                    group["approx_managed_exit_session_close_excess_vs_sector"].mean()
                ),
            }
        )
    return records


def _year_breakdown(selected: pd.DataFrame) -> list[dict[str, Any]]:
    data = selected.copy()
    data["__year"] = data["session_date_et"].astype(str).str[:4]
    return _stability_breakdown(data, "__year")


def _stability_summary(records: object) -> dict[str, float | int | None]:
    if not isinstance(records, list) or not records:
        return {"scopes": 0, "positive_scope_fraction": None, "worst_average_net_return": None}
    values = np.asarray(
        [float(_mapping(record, "stability record")["average_managed_net_return"]) for record in records],
        dtype="float64",
    )
    return {
        "scopes": len(values),
        "positive_scope_fraction": float((values > 0).mean()),
        "worst_average_net_return": float(values.min()),
    }


def _session_bootstrap(
    selected: pd.DataFrame,
    config: SwingTrainingConfig,
    *,
    session_calendar: tuple[str, ...],
) -> dict[str, dict[str, float | int]]:
    blocks = pd.DataFrame.from_records(
        _session_economic_blocks(
            selected,
            session_calendar=session_calendar,
        )
    ).rename(
        columns={
            "barrier_net_return": "calendar_average_managed_net_return",
            "approx_managed_exit_session_close_excess_vs_spy": "calendar_average_managed_exit_session_close_spy_excess",
            "approx_managed_exit_session_close_excess_vs_qqq": "calendar_average_managed_exit_session_close_qqq_excess",
            "approx_managed_exit_session_close_excess_vs_sector": "calendar_average_managed_exit_session_close_sector_excess",
        }
    )
    output: dict[str, dict[str, float | int]] = {}
    for column in blocks.columns:
        if column == "session_date_et":
            continue
        seed = config.random_seed + int(hashlib.sha256(column.encode()).hexdigest()[:8], 16)
        output[column] = _moving_block_bootstrap_mean_interval(
            blocks[column].to_numpy(dtype="float64"),
            config.bootstrap_samples,
            config.bootstrap_block_sessions,
            seed,
        )
    return output


def _moving_block_bootstrap_mean_interval(
    values: np.ndarray,
    samples: int,
    block_sessions: int,
    seed: int,
) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if len(finite) < block_sessions:
        raise DataReadinessError("moving-block bootstrap has fewer sessions than one block")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    maximum_start = len(finite) - block_sessions
    block_count = math.ceil(len(finite) / block_sessions)
    for index in range(samples):
        starts = rng.integers(0, maximum_start + 1, size=block_count)
        sampled = np.concatenate(
            [finite[start : start + block_sessions] for start in starts]
        )[: len(finite)]
        means[index] = float(sampled.mean())
    return {
        "estimate": float(finite.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "sessions": len(finite),
        "bootstrap_samples": samples,
        "block_sessions": block_sessions,
    }




