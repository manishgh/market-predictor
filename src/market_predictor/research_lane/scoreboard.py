"""Model-free scoreboard: does a signed predictor order stocks by later research-label return?

All returns are fractions (0.01 = 1 percent). Rank correlations are Spearman
coefficients in -1..+1. A date is scored only when at least ``MINIMUM_STOCKS``
eligible stocks have both a predictor value and a known label.
"""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane.labels import MISSING_REASONS, HorizonLabels

MINIMUM_STOCKS: Final = 50
TOP_FRACTION: Final = 0.10
COST_PER_TURNOVER: Final = 0.002  # 0.20% of traded value per unit of top-tenth turnover
SESSIONS_PER_YEAR: Final = 252

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]


def rank_rows(values: FloatArray, include: BoolArray) -> FloatArray:
    """Average ranks (1..n) per row over included finite cells; other cells NaN."""
    masked = np.where(include & np.isfinite(values), values, np.nan)
    ranks: FloatArray = pd.DataFrame(masked).rank(axis=1, method="average").to_numpy(dtype=np.float64)
    return ranks


def row_pearson(x: FloatArray, y: FloatArray, include: BoolArray) -> FloatArray:
    """Per-row Pearson correlation over included cells (NaN when fewer than two cells or zero variance)."""
    count = include.sum(axis=1).astype(np.float64)
    xs, ys = np.where(include, x, 0.0), np.where(include, y, 0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_x = xs.sum(axis=1) / count
        mean_y = ys.sum(axis=1) / count
        dx = np.where(include, x - mean_x[:, None], 0.0)
        dy = np.where(include, y - mean_y[:, None], 0.0)
        denominator = np.sqrt((dx * dx).sum(axis=1) * (dy * dy).sum(axis=1))
        result = (dx * dy).sum(axis=1) / denominator
    return np.where((count >= 2) & (denominator > 0), result, np.nan)


def scored_cells(predictor: FloatArray, forward: FloatArray, eligible: BoolArray, decision_rows: BoolArray,
                 minimum: int = MINIMUM_STOCKS) -> BoolArray:
    cells = eligible & np.isfinite(predictor) & np.isfinite(forward) & decision_rows[:, None]
    return cells & (cells.sum(axis=1) >= minimum)[:, None]


def daily_rank_correlation(predictor: FloatArray, forward: FloatArray, cells: BoolArray) -> FloatArray:
    """Spearman correlation per date (NaN for dates without scored cells)."""
    return row_pearson(rank_rows(predictor, cells), rank_rows(forward, cells), cells)


def _sector_demeaned(values: FloatArray, sector: npt.NDArray[np.integer], cells: BoolArray) -> FloatArray:
    out = np.full(values.shape, np.nan, dtype=np.float64)
    for code in np.unique(sector[cells]):
        members = cells & (sector == code)
        count = members.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(members, values, 0.0).sum(axis=1) / count
        out = np.where(members, values - mean[:, None], out)
    return out


def sector_neutral_rank_correlation(predictor: FloatArray, forward: FloatArray, sector: npt.NDArray[np.integer],
                                    cells: BoolArray) -> FloatArray:
    """Spearman correlation after subtracting the same-date sector mean of predictor rank and of forward return."""
    cells = cells & (sector >= 0)
    rank_residual = _sector_demeaned(rank_rows(predictor, cells), sector, cells)
    return_residual = _sector_demeaned(forward, sector, cells)
    return daily_rank_correlation(rank_residual, return_residual, cells)


def newey_west(series: Sequence[float] | FloatArray, lag: int) -> dict[str, float | int | None]:
    """Mean, Bartlett-weighted Newey-West standard error of the mean, and t-statistic."""
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count < 2:
        return {"mean": float(values.mean()) if count else None, "standard_error": None, "t": None, "count": count}
    centered = values - values.mean()
    long_run = float(centered @ centered) / count
    for distance in range(1, min(lag, count - 1) + 1):
        weight = 1.0 - distance / (lag + 1.0)
        long_run += 2.0 * weight * float(centered[distance:] @ centered[:-distance]) / count
    error = float(np.sqrt(long_run / count)) if long_run > 0 else None
    mean = float(values.mean())
    return {"mean": mean, "standard_error": error, "t": mean / error if error else None, "count": count}


def simple_t(series: Sequence[float] | FloatArray) -> dict[str, float | int | None]:
    values = np.asarray(series, dtype=np.float64)
    values = values[np.isfinite(values)]
    count = int(values.size)
    if count < 2:
        return {"mean": float(values.mean()) if count else None, "standard_error": None, "t": None, "count": count}
    error = float(values.std(ddof=1) / np.sqrt(count))
    mean = float(values.mean())
    return {"mean": mean, "standard_error": error, "t": mean / error if error > 0 else None, "count": count}


def rebalance_rows(first_row: int, total_rows: int, horizon: int) -> npt.NDArray[np.int64]:
    return np.arange(first_row, total_rows, horizon, dtype=np.int64)


@dataclass(frozen=True)
class TopTenth:
    spreads: FloatArray
    turnovers: FloatArray
    top_counts: npt.NDArray[np.int64]
    scored_counts: npt.NDArray[np.int64]
    rows: npt.NDArray[np.int64]


def top_tenth_spread(predictor: FloatArray, forward: FloatArray, cells: BoolArray,
                     rows: npt.NDArray[np.int64]) -> TopTenth:
    """Top-tenth minus all-scored mean forward return on each rebalance row with enough scored stocks.

    The top tenth holds floor(n / 10) stocks (at least one) with the highest signed predictor;
    ties are broken by column order. Turnover is the fraction of the current top-tenth names
    absent from the previous rebalance's top tenth (NaN when the previous rebalance was not scored).
    """
    spreads: list[float] = []
    turnovers: list[float] = []
    top_counts: list[int] = []
    scored_counts: list[int] = []
    kept_rows: list[int] = []
    previous: set[int] | None = None
    for row in rows:
        selected = np.flatnonzero(cells[row])
        if selected.size == 0:
            previous = None
            continue
        count = max(1, int(selected.size * TOP_FRACTION))
        order = np.argsort(-predictor[row, selected], kind="stable")
        top = selected[order[:count]]
        spreads.append(float(forward[row, top].mean() - forward[row, selected].mean()))
        current = {int(column) for column in top}
        turnovers.append(np.nan if previous is None else len(current - previous) / len(current))
        previous = current
        top_counts.append(count)
        scored_counts.append(int(selected.size))
        kept_rows.append(int(row))
    return TopTenth(np.asarray(spreads, dtype=np.float64), np.asarray(turnovers, dtype=np.float64),
                    np.asarray(top_counts, dtype=np.int64), np.asarray(scored_counts, dtype=np.int64),
                    np.asarray(kept_rows, dtype=np.int64))


def _by_year(values: FloatArray, years: npt.NDArray[np.int64]) -> dict[str, dict[str, float | int]]:
    finite = np.isfinite(values)
    result: dict[str, dict[str, float | int]] = {}
    for year in np.unique(years[finite]):
        chosen = values[finite & (years == year)]
        result[str(int(year))] = {"mean": float(chosen.mean()), "dates": int(chosen.size)}
    return result


def _mean_or_none(values: FloatArray) -> float | None:
    finite = values[np.isfinite(values)]
    return float(finite.mean()) if finite.size else None


def score_predictor(predictor: FloatArray, labels: HorizonLabels, *, eligible: BoolArray, decision_rows: BoolArray,
                    sector: npt.NDArray[np.integer], years: npt.NDArray[np.int64], first_row: int) -> dict[str, Any]:
    """Scoreboard entry for one signed predictor at one horizon."""
    horizon = labels.horizon
    cells = scored_cells(predictor, labels.stock, eligible, decision_rows)
    correlation = daily_rank_correlation(predictor, labels.stock, cells)
    rows = rebalance_rows(first_row, len(decision_rows), horizon)
    by_year = _by_year(correlation, years)
    positive_years = sum(1 for item in by_year.values() if item["mean"] > 0)
    neutral = sector_neutral_rank_correlation(predictor, labels.stock, sector, cells)
    top = top_tenth_spread(predictor, labels.stock, cells, rows)
    spread = simple_t(top.spreads)
    mean_spread = spread["mean"]
    mean_turnover = _mean_or_none(top.turnovers)
    net = None if mean_spread is None or mean_turnover is None else float(mean_spread) - COST_PER_TURNOVER * mean_turnover
    periods_per_year = SESSIONS_PER_YEAR / horizon
    return {
        "rank_correlation": {
            **newey_west(correlation, horizon), "newey_west_lag_sessions": horizon,
            "non_overlapping": simple_t(correlation[rows]),
        },
        "rank_correlation_by_year": by_year,
        "years_scored": len(by_year),
        "share_of_years_positive": positive_years / len(by_year) if by_year else None,
        "sector_neutral_rank_correlation": {**newey_west(neutral, horizon), "newey_west_lag_sessions": horizon},
        "top_tenth": {
            "rebalance_periods": int(top.spreads.size),
            "mean_spread_per_period_fraction": mean_spread,
            "annualized_spread_fraction": None if mean_spread is None else float(mean_spread) * periods_per_year,
            "spread_t": spread["t"],
            "mean_turnover_fraction": mean_turnover,
            "turnover_pairs": int(np.isfinite(top.turnovers).sum()),
            "net_mean_spread_per_period_fraction": net,
            "net_annualized_spread_fraction": None if net is None else net * periods_per_year,
            "mean_top_count_stocks": _mean_or_none(top.top_counts.astype(np.float64)),
            "mean_scored_count_stocks": _mean_or_none(top.scored_counts.astype(np.float64)),
        },
    }


def universe_facts(labels: HorizonLabels, *, eligible: BoolArray, decision_rows: BoolArray,
                   years: npt.NDArray[np.int64]) -> dict[str, Any]:
    """Label coverage and the equal-weighted eligible member's research-label return versus SPY and QQQ."""
    decisions = eligible & decision_rows[:, None]
    known = decisions & np.isfinite(labels.stock)
    missing = {name: int((decisions & (labels.reason == code)).sum()) for code, name in MISSING_REASONS.items()}
    count = known.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        average = np.where(count >= MINIMUM_STOCKS, np.where(known, labels.stock, 0.0).sum(axis=1) / count, np.nan)
    comparison: dict[str, Any] = {}
    for name, benchmark in (("spy", labels.spy), ("qqq", labels.qqq)):
        difference = average - benchmark
        comparison[f"average_stock_minus_{name}"] = {
            **newey_west(difference, labels.horizon), "newey_west_lag_sessions": labels.horizon,
            "by_year": _by_year(difference, years),
        }
    return {
        "eligible_decisions_stock_sessions": int(decisions.sum()),
        "known_label_stock_sessions": int(known.sum()),
        "known_label_share": float(known.sum() / decisions.sum()) if decisions.sum() else None,
        "missing_label_stock_sessions_by_reason": missing,
        "decision_dates_with_average": int(np.isfinite(average).sum()),
        **comparison,
    }


def build_scoreboard(signed_predictors: Mapping[str, FloatArray], labels: Sequence[HorizonLabels], *,
                     eligible: BoolArray, sessions: npt.NDArray[np.datetime64], sector: npt.NDArray[np.integer],
                     first_row: int) -> dict[str, Any]:
    decision_rows = np.arange(len(sessions)) >= first_row
    years = sessions.astype("datetime64[Y]").astype(np.int64) + 1970
    results: dict[str, Any] = {name: {} for name in signed_predictors}
    universe: dict[str, Any] = {}
    for horizon_labels in labels:
        key = str(horizon_labels.horizon)
        universe[key] = universe_facts(horizon_labels, eligible=eligible, decision_rows=decision_rows, years=years)
        for name, values in signed_predictors.items():
            results[name][key] = score_predictor(values, horizon_labels, eligible=eligible, decision_rows=decision_rows,
                                                 sector=sector, years=years, first_row=first_row)
    return {"predictors": results, "universe": universe}
