from __future__ import annotations

import hashlib
from collections.abc import Mapping
from typing import TYPE_CHECKING, Any, cast

import numpy as np
import pandas as pd

from market_predictor.modeling.resampling import moving_block_mean_interval
from market_predictor.modeling.strategy_contract import StrategyContract

if TYPE_CHECKING:
    from market_predictor.edge_rebuild.training.swing_types import SwingTrainingConfig
from market_predictor.edge_rebuild.training.utils import _mapping


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
    # Price-basis admission is not supplied by caller-authored metric flags. The
    # retained all-adjusted source has no independently reconciled TR authority.
    accounting = metrics.get("funded_spy_accounting")
    return {
        "passed": False,
        "status": "price_basis_pending",
        "reason": "scope-matched stock and SPY total-return source reconciliation is unavailable",
        "checks": {
            "funded_daily_spy_comparison_available": isinstance(accounting, Mapping),
            "stock_and_spy_price_basis_reconciled": False,
        },
        "minimum_expected_net_edge_bps": strategy_contract.swing.minimum_expected_net_edge_bps,
        "stress_cost_multiplier": strategy_contract.stress.cost_multiplier,
        "managed_exit_session_close_benchmarks": "approximate_diagnostic_only",
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
        output[column] = moving_block_mean_interval(
            blocks[column].to_numpy(dtype="float64"),
            config.bootstrap_samples,
            config.bootstrap_block_sessions,
            seed,
        )
    return output




