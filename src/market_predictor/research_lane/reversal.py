"""Very-short-term reversal: the frozen step 2c signals and their pre-registered reading rule.

Every input and threshold here was fixed before any 1-3 session result was seen. The
lead itself came from inspecting the same development history, so a pass only earns
one test on the sealed period; it is not confirmation.
"""
from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from market_predictor.research_lane.predictors import centered_rank

FloatArray = npt.NDArray[np.float64]

REVERSAL_INPUTS: Final = ("return_1d", "return_3d", "intraday_today", "streak_length")
REVERSAL_SIGN: Final = -1
REVERSAL_COMPOSITE: Final = "reversal_short"
REVERSAL_MINIMUM_PRESENT: Final = 3
REVERSAL_HORIZONS: Final = (1, 2, 3, 5)
REQUIRED_T: Final = 3.0
REQUIRED_POSITIVE_YEARS: Final = 5
FROZEN_COST: Final = 0.002
INFORMATION_COST: Final = 0.001
READING_RULE: Final = (
    "The lead earns one test on the sealed period only if reversal_short, at one of the four horizons, has a "
    "Newey-West t-statistic of at least 3, is positive in at least 5 of 6 calendar years, and has a positive "
    "top-tenth spread after the 0.20% cost. Otherwise the lead is closed."
)


def reversal_signals(indicators: Mapping[str, FloatArray], eligible: npt.NDArray[np.bool_]) -> dict[str, FloatArray]:
    """Signed inputs (sign -1) plus the composite: mean signed centered rank, at least 3 of 4 present."""
    signed = {name: REVERSAL_SIGN * np.asarray(indicators[name], dtype=np.float64) for name in REVERSAL_INPUTS}
    ranks = np.stack([centered_rank(signed[name], eligible) for name in REVERSAL_INPUTS])
    finite = np.isfinite(ranks)
    present = finite.sum(axis=0)
    total = np.where(finite, ranks, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = total / np.maximum(present, 1)
    signed[REVERSAL_COMPOSITE] = np.where(present >= REVERSAL_MINIMUM_PRESENT, mean, np.nan)
    return signed


def _net(top: Mapping[str, Any], cost: float) -> float | None:
    spread, turnover = top.get("mean_spread_per_period_fraction"), top.get("mean_turnover_fraction")
    if spread is None or turnover is None:
        return None
    return float(spread) - cost * float(turnover)


def reading(scoreboard: Mapping[str, Any], horizons: tuple[int, ...] = REVERSAL_HORIZONS) -> dict[str, Any]:
    """Apply the frozen reading rule to the composite at each horizon."""
    per_horizon: dict[str, Any] = {}
    for horizon in (str(h) for h in horizons):
        entry = scoreboard["predictors"][REVERSAL_COMPOSITE][horizon]
        t = entry["rank_correlation"].get("t")
        years = entry["rank_correlation_by_year"]
        positive_years = sum(1 for item in years.values() if item["mean"] > 0)
        top = entry["top_tenth"]
        net_frozen, net_information = _net(top, FROZEN_COST), _net(top, INFORMATION_COST)
        passes = (t is not None and t >= REQUIRED_T and positive_years >= REQUIRED_POSITIVE_YEARS
                  and net_frozen is not None and net_frozen > 0)
        per_horizon[horizon] = {
            "mean_rank_correlation": entry["rank_correlation"].get("mean"), "newey_west_t": t,
            "years_positive": positive_years, "years_scored": len(years),
            "gross_spread_per_period_fraction": top.get("mean_spread_per_period_fraction"),
            "mean_turnover_fraction": top.get("mean_turnover_fraction"),
            "net_spread_per_period_at_0_20pct_fraction": net_frozen,
            "net_spread_per_period_at_0_10pct_fraction": net_information,
            "net_annualized_at_0_20pct_fraction": None if net_frozen is None else net_frozen * 252 / int(horizon),
            "passes": passes,
        }
    return {"rule": READING_RULE, "horizons": per_horizon,
            "earns_sealed_period_test": any(item["passes"] for item in per_horizon.values())}


def reading_markdown(result: Mapping[str, Any]) -> list[str]:
    def pct(value: Any) -> str:
        return "n/a" if value is None else f"{100 * float(value):.3f}%"

    lines = ["## Pre-registered reading rule", "", result["rule"], "",
             "| Horizon | Mean rank corr. [NW t] | Years positive | Gross spread | Turnover | Net at 0.20% | "
             "Net at 0.10% | Net per year at 0.20% | Passes |", "|---|---|---|---|---|---|---|---|---|"]
    for horizon, item in result["horizons"].items():
        mean, t = item["mean_rank_correlation"], item["newey_west_t"]
        lines.append(
            f"| {horizon} | {'n/a' if mean is None else f'{mean:.4f}'} [{'n/a' if t is None else f'{t:.2f}'}] | "
            f"{item['years_positive']}/{item['years_scored']} | {pct(item['gross_spread_per_period_fraction'])} | "
            f"{pct(item['mean_turnover_fraction'])} | {pct(item['net_spread_per_period_at_0_20pct_fraction'])} | "
            f"{pct(item['net_spread_per_period_at_0_10pct_fraction'])} | "
            f"{pct(item['net_annualized_at_0_20pct_fraction'])} | {'yes' if item['passes'] else 'no'} |")
    verdict = "earns one sealed-period test" if result["earns_sealed_period_test"] else "lead closed"
    return [*lines, "", f"Verdict: **{verdict}**.", ""]
