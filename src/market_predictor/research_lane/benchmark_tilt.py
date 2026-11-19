"""Option C (Research Lane Step 4): cap-weighted benchmark tilt and turnover-buffered reversal & quality.

Evaluates whether combining the step 2c short-term pullback signal (`reversal_short`) with the
step 2a accounting-quality signal (`composite_quality`), under buy/hold rank hysteresis and
point-in-time market-cap benchmark tilts bounded within +-2.0% per stock and sector neutrality,
overcomes turnover friction and equal-weight index-composition drag against SPY.

All returns are research labels built on provider split- and dividend-adjusted SIP daily bars
over continuous rebalance windows `[open[r_k + 1], open[r_{k+1} + 1]]`. Market values for
benchmark weights use unadjusted daily closes times point-in-time SEC shares outstanding.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane import fundamentals, reversal
from market_predictor.research_lane.company_facts import (
    COMPONENTS,
    INCOMPLETE_HISTORY,
    ISSUER_SHARED_BY_SECURITIES,
    NO_COMPANY_FACTS,
    NO_DAY,
    NONPOSITIVE_DENOMINATOR,
    PRESENT,
    IssuerFacts,
    evaluate_issuer,
)
from market_predictor.research_lane.indicators import compute_indicators
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.predictors import centered_rank, minimum_present
from market_predictor.research_lane.scoreboard import (
    MINIMUM_STOCKS,
    SESSIONS_PER_YEAR,
    TOP_FRACTION,
    newey_west,
    simple_t,
)

FloatArray = npt.NDArray[np.float64]
BoolArray = npt.NDArray[np.bool_]
IntArray = npt.NDArray[np.int64]
ReasonArray = npt.NDArray[np.int8]

EXTRA_COMPANY_FACTS_DIRECTORY: Final = Path("data/raw/investment_company_facts")

TILT_HORIZONS: Final = (3, 5, 10, 21)
SIGNAL_NAMES: Final = (
    "reversal_short",
    "quality_slow",
    "quality_plus_reversal",
    "quality_gated_reversal",
)
POLICY_NAMES: Final = (
    "skill_top10_unbuffered",
    "skill_top10_hold33",
    "product_cap_mult_buffered",
    "product_sector_active_slice",
    "product_core_spy_80_sat_20",
)

QUALITY_MEMBERS: Final = fundamentals.COMPOSITE_GROUPS["composite_quality"]
BUY_TOP_FRACTION: Final = TOP_FRACTION  # 0.10 (enter at >= 90th percentile)
HOLD_TOP_FRACTION: Final = 1.0 / 3.0  # keep held while in top third (>= 66.67th percentile)

MULT_ENTER_TOP_PCT: Final = 0.80  # enter +1x overweight in top fifth
MULT_EXIT_TOP_PCT: Final = 0.50  # keep +1x overweight while at or above median
MULT_ENTER_BOT_PCT: Final = 0.20  # enter -1x underweight in bottom fifth
MULT_EXIT_BOT_PCT: Final = 0.50  # keep -1x underweight while at or below median
MULT_SCALE: Final = 1.0  # w_raw = w_cap * (1 + MULT_SCALE * state), state in {-1, 0, +1}

SECTOR_ACTIVE_SLICE_SHARE: Final = 0.30  # 30% within-sector active slice on buffered top-10%/hold-33%
CORE_SPY_WEIGHT: Final = 0.80  # 80% passive SPY + 20% buffered satellite
STOCK_ACTIVE_LIMIT: Final = 0.02  # +-2.0 percentage points relative to benchmark weight
SECTOR_ACTIVE_LIMIT: Final = 0.05  # +-5.0 percentage points (within-sector projection achieves 0.0%)
TRADE_DEADBAND: Final = 0.0010  # 10 bps unchanged-state weight deadband

FROZEN_COST_PER_TURNOVER: Final = 0.0020  # 20 bps per unit of one-way replaced weight (10 bps buy + 10 bps sell)
INFO_COST_PER_TURNOVER: Final = 0.0010  # 10 bps informational cost

REQUIRED_T_PRIMARY: Final = 2.50
REQUIRED_T_STRICT: Final = 3.00
REQUIRED_POSITIVE_YEARS: Final = 5

STEP4_READING_RULE: Final = (
    "A product specification (product_cap_mult_buffered, product_sector_active_slice, or "
    "product_core_spy_80_sat_20) earns one test on the sealed period only if, at one of the "
    "four pre-registered horizons (3, 5, 10, 21 sessions), it has positive mean net excess "
    "return against SPY after the frozen 0.20% turnover cost, achieves a Newey-West t-statistic "
    "of at least 2.50 on per-period net excess return against its matched benchmark (reporting "
    "t >= 3.00 alongside), is net positive against SPY in at least 5 of 6 calendar years "
    "(2019-2024), and its paired buffered skill basket (skill_top10_hold33) is also net positive "
    "at 0.20% with Newey-West t >= 2.50. Otherwise Option C does not earn a sealed test."
)


@dataclass(frozen=True)
class QualityAndCapResult:
    """Point-in-time `quality_slow` composite and unadjusted-price `market_value` matrix."""

    quality_slow: FloatArray  # sessions x securities, centered-rank mean (-0.5..+0.5)
    market_value: FloatArray  # sessions x securities, USD (unadjusted_close * shares), NaN when unavailable
    market_value_reason: ReasonArray
    split_blanked: BoolArray
    report: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class ContinuousWindows:
    """Non-overlapping open-to-open holding windows `[open[r_k + 1], open[r_k + h + 1]]`."""

    horizon: int
    decision_rows: IntArray  # r_k (length K)
    entry_rows: IntArray  # r_k + 1 (length K)
    exit_rows: IntArray  # r_k + h + 1 (length K)
    years: IntArray  # calendar year of decision session r_k (length K)
    tradable: BoolArray  # K x C: eligible[r_k, i] & finite positive open[r_k + 1, i]
    stock_returns: FloatArray  # K x C: gross return over [open[r_k + 1], open[r_k + h + 1]]
    spy_returns: FloatArray  # K: gross SPY return over [spy_open[r_k + 1], spy_open[r_k + h + 1]]
    unfilled_eligible_entries: int
    missing_exit_fallbacks: int


@dataclass(frozen=True)
class PathMetrics:
    """Realized portfolio return, drift-adjusted turnover, and active-bound diagnostics."""

    weights: FloatArray  # K x A (sum == 1 on active periods, NaN when abstaining)
    gross_returns: FloatArray  # K
    one_way_turnovers: FloatArray  # K (0.5 initial entry on first active period, post-drift thereafter)
    steady_turnovers: FloatArray  # K (NaN on first active period, post-drift thereafter)
    net_returns_20bps: FloatArray  # K
    net_returns_10bps: FloatArray  # K
    active_shares: FloatArray  # K
    max_stock_active_abs: float
    max_sector_active_abs: float


def map_security_ciks(
    security_ids: tuple[str, ...],
    recorded_ciks: Mapping[str, tuple[str, ...]],
    has_facts: Callable[[str], bool],
) -> tuple[dict[str, str], dict[str, Any]]:
    """Map panel `security_id` to 10-digit CIK when a `company_facts.json` file exists."""
    mapping: dict[str, str] = {}
    counts: Counter[str] = Counter()
    for security in security_ids:
        parts = security.split(":")
        kind = parts[0]
        if kind == "cik" and len(parts) >= 2:
            cik, source = parts[1], "cik_security_id"
        else:
            ciks = recorded_ciks.get(security, ())
            if len(ciks) != 1:
                counts[f"{kind}:no_single_recorded_cik"] += 1
                continue
            cik, source = ciks[0], "filing_inventory_recorded_cik"
        if not has_facts(cik):
            counts[f"{kind}:{source}:no_company_facts_file"] += 1
            continue
        counts[f"{kind}:{source}:mapped"] += 1
        mapping[security] = cik
    return mapping, {
        "panel_securities": len(security_ids),
        "mapped_securities": len(mapping),
        "by_id_kind_and_source": dict(sorted(counts.items())),
    }


def _ratio_checked(
    numerator: FloatArray, denominator: FloatArray, reason: ReasonArray,
) -> tuple[FloatArray, ReasonArray]:
    positive = np.isfinite(denominator) & (denominator > 0.0)
    out_reason = np.where((reason == PRESENT) & ~positive, NONPOSITIVE_DENOMINATOR, reason).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(out_reason == PRESENT, numerator / np.where(positive, denominator, 1.0), np.nan)
    finite = np.isfinite(value)
    if bool(np.any(finite != (out_reason == PRESENT))):
        value = np.where(out_reason == PRESENT, value, np.nan)
        out_reason = np.where((out_reason == PRESENT) & ~np.isfinite(value), INCOMPLETE_HISTORY, out_reason).astype(
            np.int8
        )
    return value, out_reason


def build_quality_and_market_cap(
    panel: ResearchPanel,
    unadjusted_close: FloatArray,
    security_issuer: Mapping[str, str],
    load_issuer: Callable[[str], IssuerFacts | None],
) -> QualityAndCapResult:
    """Compute Step 2a's exact `composite_quality` and split-checked `market_value` in one pass."""
    rows, columns = len(panel.sessions), len(panel.security_ids)
    shape = (rows, columns)
    if unadjusted_close.shape != shape:
        raise ValueError("unadjusted_close shape differs from panel shape")

    column_index = pd.Index(panel.security_ids)
    needed_components = (
        "gross_profit_ttm",
        "operating_income_ttm",
        "net_income_ttm",
        "operating_cash_flow_ttm",
        "total_assets",
        "stockholders_equity",
        "asset_growth",
        "share_issuance",
        "shares",
    )
    comp = {name: np.full(shape, np.nan, dtype=np.float64) for name in needed_components}
    comp_reason = {name: np.full(shape, NO_COMPANY_FACTS, dtype=np.int8) for name in needed_components}
    share_as_of = np.full(shape, NO_DAY, dtype=np.int64)

    issuers: dict[str, list[int]] = {}
    for security, cik in security_issuer.items():
        pos = int(column_index.get_indexer([security])[0])
        if pos >= 0:
            issuers.setdefault(cik, []).append(pos)

    issuers_without_facts: list[str] = []
    for cik in sorted(issuers):
        facts = load_issuer(cik)
        if facts is None:
            issuers_without_facts.append(cik)
            continue
        series = evaluate_issuer(facts, panel.sessions)
        del facts
        for pos in issuers[cik]:
            for name in COMPONENTS:
                if name in comp:
                    comp[name][:, pos] = series.values[name]
                    comp_reason[name][:, pos] = series.reasons[name]
            share_as_of[:, pos] = series.share_as_of

    shared = np.zeros(columns, dtype=bool)
    for positions in issuers.values():
        if len(positions) > 1:
            shared[positions] = True
    shared_reason = np.where(np.broadcast_to(shared, shape), ISSUER_SHARED_BY_SECURITIES, PRESENT).astype(np.int8)
    share_reason = fundamentals.first_reason(comp_reason["shares"], shared_reason)

    adjusted_close = panel.close.astype(np.float64)
    mv, mv_reason, blanked, split_ev = fundamentals.market_value(
        unadjusted_close, adjusted_close, comp["shares"], share_reason, share_as_of, panel.sessions,
    )

    equity_reason = fundamentals.first_reason(
        comp_reason["stockholders_equity"],
        np.where(
            (comp_reason["stockholders_equity"] == PRESENT) & ~(comp["stockholders_equity"] > 0.0),
            NONPOSITIVE_DENOMINATOR,
            PRESENT,
        ).astype(np.int8),
    )

    quality_raw: dict[str, FloatArray] = {}
    quality_raw["gross_profit_to_assets"], _ = _ratio_checked(
        comp["gross_profit_ttm"],
        comp["total_assets"],
        fundamentals.first_reason(comp_reason["gross_profit_ttm"], comp_reason["total_assets"]),
    )
    quality_raw["operating_profitability"], _ = _ratio_checked(
        comp["operating_income_ttm"],
        comp["stockholders_equity"],
        fundamentals.first_reason(comp_reason["operating_income_ttm"], equity_reason),
    )
    quality_raw["accruals"], _ = _ratio_checked(
        comp["net_income_ttm"] - comp["operating_cash_flow_ttm"],
        comp["total_assets"],
        fundamentals.first_reason(
            comp_reason["net_income_ttm"], comp_reason["operating_cash_flow_ttm"], comp_reason["total_assets"],
        ),
    )
    ag_ok = (comp_reason["asset_growth"] == PRESENT) & np.isfinite(comp["asset_growth"])
    quality_raw["asset_growth"] = np.where(ag_ok, comp["asset_growth"], np.nan)

    issuance_reason = fundamentals.first_reason(comp_reason["share_issuance"], shared_reason)
    iss_ok = (issuance_reason == PRESENT) & np.isfinite(comp["share_issuance"])
    quality_raw["net_share_issuance"] = np.where(iss_ok, comp["share_issuance"], np.nan)

    signed_ranks = np.stack(
        [fundamentals.EXPECTED_SIGNS[name] * centered_rank(quality_raw[name], panel.eligible) for name in QUALITY_MEMBERS]
    )
    present = np.isfinite(signed_ranks).sum(axis=0)
    total = np.where(np.isfinite(signed_ranks), signed_ranks, 0.0).sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean_rank = total / np.maximum(present, 1)
    quality_slow = np.where(present >= minimum_present(len(QUALITY_MEMBERS)), mean_rank, np.nan)

    report = {
        "issuers_mapped": len(issuers),
        "issuers_without_facts": issuers_without_facts,
        "panel_securities_sharing_an_issuer": int(shared.sum()),
        "split_events_detected": int(split_ev.sum()),
        "split_blanked_cells": int(blanked.sum()),
        "securities_with_any_market_value": int(np.isfinite(mv).any(axis=0).sum()),
        "securities_with_any_quality_slow": int(np.isfinite(quality_slow).any(axis=0).sum()),
    }
    return QualityAndCapResult(
        quality_slow=quality_slow,
        market_value=mv,
        market_value_reason=mv_reason,
        split_blanked=blanked,
        report=report,
    )


def build_option_c_signals(
    panel: ResearchPanel, quality_slow: FloatArray,
) -> dict[str, FloatArray]:
    """Build the four pre-registered Option C signal matrices (`sessions x securities`, `-0.5..+0.5`)."""
    indicators = compute_indicators(panel)
    rev_short = reversal.reversal_signals(indicators, panel.eligible)[reversal.REVERSAL_COMPOSITE]
    del indicators

    q_rank = centered_rank(quality_slow, panel.eligible)
    r_rank = centered_rank(rev_short, panel.eligible)
    both_ok = panel.eligible & np.isfinite(q_rank) & np.isfinite(r_rank)

    blend_raw = np.where(both_ok, 0.5 * q_rank + 0.5 * r_rank, np.nan)
    quality_plus_reversal = centered_rank(blend_raw, panel.eligible)

    gated_raw = np.where(both_ok, np.where(q_rank >= 0.0, 1.0 + r_rank, -1.0 + r_rank), np.nan)
    quality_gated_reversal = centered_rank(gated_raw, panel.eligible)

    return {
        "reversal_short": rev_short,
        "quality_slow": quality_slow,
        "quality_plus_reversal": quality_plus_reversal,
        "quality_gated_reversal": quality_gated_reversal,
    }


def continuous_rebalance_windows(
    panel: ResearchPanel, first_row: int, horizon: int,
) -> ContinuousWindows:
    """Build contiguous open-to-open holding intervals `[open[r_k + 1], open[r_k + h + 1]]`."""
    if horizon < 1:
        raise ValueError("horizon must be >= 1")
    n_sessions, n_sec = len(panel.sessions), len(panel.security_ids)
    # Require r_k + horizon + 1 < n_sessions so both open[r_k + 1] and open[r_k + h + 1] exist in the panel.
    max_decision = n_sessions - horizon - 1
    if max_decision <= first_row:
        raise ValueError(f"insufficient sessions ({n_sessions}) after first_row ({first_row}) for horizon {horizon}")

    decision_rows = np.arange(first_row, max_decision, horizon, dtype=np.int64)
    entry_rows = decision_rows + 1
    exit_rows = decision_rows + horizon + 1
    years = panel.sessions[decision_rows].astype("datetime64[Y]").astype(np.int64) + 1970

    opens = panel.open.astype(np.float64)
    closes = panel.close.astype(np.float64)
    spy_opens = panel.spy_open.astype(np.float64)

    k_len = len(decision_rows)
    tradable = np.zeros((k_len, n_sec), dtype=bool)
    stock_returns = np.zeros((k_len, n_sec), dtype=np.float64)
    spy_returns = np.full(k_len, np.nan, dtype=np.float64)
    unfilled_entries = 0
    fallback_exits = 0

    for idx in range(k_len):
        r_row = int(decision_rows[idx])
        e_row = int(entry_rows[idx])
        x_row = int(exit_rows[idx])

        elig = panel.eligible[r_row]
        entry_px = opens[e_row]
        entry_ok = np.isfinite(entry_px) & (entry_px > 0.0)
        unfilled_entries += int((elig & ~entry_ok).sum())
        trad = elig & entry_ok
        tradable[idx] = trad

        exit_px = opens[x_row]
        exit_ok = np.isfinite(exit_px) & (exit_px > 0.0)
        normal = trad & exit_ok
        with np.errstate(invalid="ignore", divide="ignore"):
            stock_returns[idx, normal] = exit_px[normal] / entry_px[normal] - 1.0

        missing_exit_cols = np.flatnonzero(trad & ~exit_ok)
        for col in missing_exit_cols:
            c = int(col)
            last_val = entry_px[c]
            for bar_row in range(x_row - 1, e_row - 1, -1):
                cand_close = closes[bar_row, c]
                if np.isfinite(cand_close) and cand_close > 0.0:
                    last_val = cand_close
                    break
                cand_open = opens[bar_row, c]
                if np.isfinite(cand_open) and cand_open > 0.0:
                    last_val = cand_open
                    break
            stock_returns[idx, c] = last_val / entry_px[c] - 1.0
            fallback_exits += 1

        if np.isfinite(spy_opens[e_row]) and spy_opens[e_row] > 0.0 and np.isfinite(spy_opens[x_row]) and spy_opens[x_row] > 0.0:
            spy_returns[idx] = float(spy_opens[x_row] / spy_opens[e_row] - 1.0)

    return ContinuousWindows(
        horizon=horizon,
        decision_rows=decision_rows,
        entry_rows=entry_rows,
        exit_rows=exit_rows,
        years=years,
        tradable=tradable,
        stock_returns=stock_returns,
        spy_returns=spy_returns,
        unfilled_eligible_entries=unfilled_entries,
        missing_exit_fallbacks=fallback_exits,
    )


def project_sector_bounded_weights(
    desired: FloatArray,
    bench: FloatArray,
    sector: npt.NDArray[np.integer[Any]],
    *,
    stock_active_limit: float = STOCK_ACTIVE_LIMIT,
) -> FloatArray:
    """Project `desired` onto `{w >= 0, |w - bench| <= stock_active_limit, sum_{i in g} w_i = sum_{i in g} bench_i}`.

    Exact within-sector clipped water-filling guarantees:
    1. Long-only (`w_i >= 0`),
    2. Per-stock active deviation `|w_i - bench_i| <= stock_active_limit`,
    3. Exact sector neutrality (`sum_{i in g} (w_i - bench_i) == 0.0`, strictly inside +-5%),
    4. Full investment (`sum_i w_i == sum_i bench_i == 1.0`).
    """
    out = np.zeros_like(bench, dtype=np.float64)
    active_mask = bench > 0.0
    if not bool(active_mask.any()):
        return out

    for sec_code in np.unique(sector[active_mask]):
        idx = np.flatnonzero(active_mask & (sector == sec_code))
        wb = bench[idx]
        target_sum = float(wb.sum())
        low = np.maximum(0.0, wb - stock_active_limit)
        high = np.minimum(1.0, wb + stock_active_limit)
        d_sec = desired[idx]
        clipped = np.clip(d_sec, low, high)

        current_sum = float(clipped.sum())
        if abs(current_sum - target_sum) <= 1e-15:
            out[idx] = clipped
            continue
        if target_sum <= float(low.sum()) + 1e-15:
            out[idx] = low
            continue
        if target_sum >= float(high.sum()) - 1e-15:
            out[idx] = high
            continue

        lo_shift = float(np.min(low - d_sec)) - 1e-12
        hi_shift = float(np.max(high - d_sec)) + 1e-12
        for _ in range(60):
            mid = 0.5 * (lo_shift + hi_shift)
            val = float(np.clip(d_sec + mid, low, high).sum())
            if val < target_sum:
                lo_shift = mid
            else:
                hi_shift = mid
        out[idx] = np.clip(d_sec + 0.5 * (lo_shift + hi_shift), low, high)

    total = float(out.sum())
    if total > 0.0 and abs(total - 1.0) > 1e-13:
        out /= total
    return out


def _percentiles_among_selected(scores: FloatArray, indices: IntArray) -> FloatArray:
    """Midpoint percentile in `(0, 1)` among `indices`, breaking ties stably by column order."""
    n = indices.size
    order = np.argsort(scores[indices], kind="stable")
    pct = np.empty(n, dtype=np.float64)
    pct[order] = (np.arange(n, dtype=np.float64) + 0.5) / float(n)
    return pct


def simulate_portfolio_path(
    weights: FloatArray,
    asset_returns: FloatArray,
    bench_weights: FloatArray | None = None,
    sector_matrix: npt.NDArray[np.integer[Any]] | None = None,
) -> PathMetrics:
    """Compute exact gross/net returns and post-drift one-way turnover for weight schedule `weights` (`K x A`)."""
    k_len, n_assets = weights.shape
    gross = np.full(k_len, np.nan, dtype=np.float64)
    turnovers = np.full(k_len, np.nan, dtype=np.float64)
    steady_turnovers = np.full(k_len, np.nan, dtype=np.float64)
    net_20 = np.full(k_len, np.nan, dtype=np.float64)
    net_10 = np.full(k_len, np.nan, dtype=np.float64)
    active_shares = np.full(k_len, np.nan, dtype=np.float64)

    max_stock_active = 0.0
    max_sector_active = 0.0
    drifted_prev: FloatArray | None = None

    for k in range(k_len):
        w = weights[k]
        if not bool(np.isfinite(w).all()) or float(w.sum()) <= 0.0:
            drifted_prev = None
            continue
        r = asset_returns[k]
        g_ret = float(np.dot(w, r))
        gross[k] = g_ret

        if drifted_prev is None:
            # Initial entry from cash: 0.5 one-way replaced weight (= 10 bps at 20 bps round-trip cost).
            tau = 0.5
            steady_turnovers[k] = np.nan
        else:
            tau = 0.5 * float(np.abs(w - drifted_prev).sum())
            steady_turnovers[k] = tau

        turnovers[k] = tau
        net_20[k] = g_ret - FROZEN_COST_PER_TURNOVER * tau
        net_10[k] = g_ret - INFO_COST_PER_TURNOVER * tau

        denom = 1.0 + g_ret
        if denom > 1e-12:
            drifted_prev = w * (1.0 + r) / denom
        else:
            drifted_prev = None

        if bench_weights is not None and bool(np.isfinite(bench_weights[k]).all()) and float(bench_weights[k].sum()) > 0.0:
            diff = w - bench_weights[k]
            active_shares[k] = 0.5 * float(np.abs(diff).sum())
            max_stock_active = max(max_stock_active, float(np.max(np.abs(diff))))
            if sector_matrix is not None:
                sec_row = sector_matrix[k]
                for code in np.unique(sec_row[(w > 0.0) | (bench_weights[k] > 0.0)]):
                    sec_diff = abs(float(diff[sec_row == code].sum()))
                    max_sector_active = max(max_sector_active, sec_diff)

    return PathMetrics(
        weights=weights,
        gross_returns=gross,
        one_way_turnovers=turnovers,
        steady_turnovers=steady_turnovers,
        net_returns_20bps=net_20,
        net_returns_10bps=net_10,
        active_shares=active_shares,
        max_stock_active_abs=max_stock_active,
        max_sector_active_abs=max_sector_active,
    )


def build_skill_weights(
    panel: ResearchPanel,
    signal: FloatArray,
    decision_rows: IntArray,
    *,
    buffered: bool,
    minimum_stocks: int = MINIMUM_STOCKS,
) -> tuple[FloatArray, FloatArray]:
    """Build equal-weighted top-10% decision weights at `r_k` close using strictly data `<= r_k`."""
    k_len = len(decision_rows)
    n_sec = len(panel.security_ids)
    weights = np.full((k_len, n_sec), np.nan, dtype=np.float64)
    bench = np.full((k_len, n_sec), np.nan, dtype=np.float64)
    held: set[int] = set()

    for k in range(k_len):
        r_row = int(decision_rows[k])
        scores = signal[r_row]
        valid = np.flatnonzero(panel.eligible[r_row] & np.isfinite(scores))
        if valid.size < minimum_stocks:
            held.clear()
            continue

        b_row = np.zeros(n_sec, dtype=np.float64)
        b_row[valid] = 1.0 / float(valid.size)
        bench[k] = b_row

        target_m = max(1, int(valid.size * BUY_TOP_FRACTION))
        desc_order = valid[np.argsort(-scores[valid], kind="stable")]

        if not buffered or not held:
            chosen = desc_order[:target_m]
        else:
            pct = _percentiles_among_selected(scores, valid)
            keep_threshold = 1.0 - HOLD_TOP_FRACTION
            surviving = [
                int(valid[pos])
                for pos in range(valid.size)
                if int(valid[pos]) in held and pct[pos] >= keep_threshold
            ]
            if len(surviving) > target_m:
                surviving.sort(key=lambda col: (-float(scores[col]), col))
                surviving = surviving[:target_m]
            surviving_set = set(surviving)
            needed = target_m - len(surviving)
            additions = [int(col) for col in desc_order if int(col) not in surviving_set][:needed]
            chosen = np.asarray(surviving + additions, dtype=np.int64)

        held = {int(c) for c in chosen}
        w_row = np.zeros(n_sec, dtype=np.float64)
        w_row[chosen] = 1.0 / float(chosen.size)
        weights[k] = w_row

    return weights, bench


def build_cap_tilt_weights(
    panel: ResearchPanel,
    signal: FloatArray,
    market_value: FloatArray,
    decision_rows: IntArray,
    *,
    mode: str,
    minimum_stocks: int = MINIMUM_STOCKS,
) -> tuple[FloatArray, FloatArray]:
    """Build cap-tilt decision weights at `r_k` close using strictly data `<= r_k`."""
    if mode not in ("mult_buffered", "sector_active_slice"):
        raise ValueError(f"unsupported cap tilt mode: {mode}")

    k_len = len(decision_rows)
    n_sec = len(panel.security_ids)
    weights = np.full((k_len, n_sec), np.nan, dtype=np.float64)
    cap_bench = np.full((k_len, n_sec), np.nan, dtype=np.float64)

    opens = panel.open.astype(np.float64)
    closes = panel.close.astype(np.float64)
    prev_state = np.zeros(n_sec, dtype=np.float64)
    prev_weights: FloatArray | None = None
    prev_decision_row: int | None = None

    for k in range(k_len):
        r_row = int(decision_rows[k])
        mv_row = market_value[r_row]
        cap_ok = panel.eligible[r_row] & np.isfinite(mv_row) & (mv_row > 0.0)
        cap_cols = np.flatnonzero(cap_ok)
        if cap_cols.size < minimum_stocks:
            prev_state.fill(0.0)
            prev_weights = None
            prev_decision_row = None
            continue

        w_cap = np.zeros(n_sec, dtype=np.float64)
        w_cap[cap_cols] = mv_row[cap_cols] / float(mv_row[cap_cols].sum())
        cap_bench[k] = w_cap

        # Observe post-drift weights at close[r_k] from prior entry open[r_{k-1} + 1] (both <= r_row).
        drifted_at_close: FloatArray | None = None
        if prev_weights is not None and prev_decision_row is not None and prev_decision_row + 1 <= r_row:
            e_prev = prev_decision_row + 1
            entry_px = opens[e_prev]
            close_px = closes[r_row]
            px_ok = np.isfinite(entry_px) & (entry_px > 0.0) & np.isfinite(close_px) & (close_px > 0.0)
            obs_ret = np.zeros(n_sec, dtype=np.float64)
            with np.errstate(invalid="ignore", divide="ignore"):
                obs_ret[px_ok] = close_px[px_ok] / entry_px[px_ok] - 1.0
            denom = 1.0 + float(np.dot(prev_weights, obs_ret))
            if denom > 1e-12:
                drifted_at_close = prev_weights * (1.0 + obs_ret) / denom

        scores = signal[r_row]
        scored_cap = cap_cols[np.isfinite(scores[cap_cols])]
        new_state = np.zeros(n_sec, dtype=np.float64)

        if scored_cap.size >= minimum_stocks:
            pct = _percentiles_among_selected(scores, scored_cap)
            if mode == "mult_buffered":
                for pos, col_id in enumerate(scored_cap):
                    c = int(col_id)
                    p = float(pct[pos])
                    ps = float(prev_state[c])
                    if p >= MULT_ENTER_TOP_PCT or (ps > 0.0 and p >= MULT_EXIT_TOP_PCT):
                        new_state[c] = 1.0
                    elif p <= MULT_ENTER_BOT_PCT or (ps < 0.0 and p <= MULT_EXIT_BOT_PCT):
                        new_state[c] = -1.0
                    else:
                        new_state[c] = 0.0
                desired = w_cap * (1.0 + MULT_SCALE * new_state)
            else:
                keep_threshold = 1.0 - HOLD_TOP_FRACTION
                enter_threshold = 1.0 - BUY_TOP_FRACTION
                for pos, col_id in enumerate(scored_cap):
                    c = int(col_id)
                    p = float(pct[pos])
                    if p >= enter_threshold or (prev_state[c] > 0.0 and p >= keep_threshold):
                        new_state[c] = 1.0
                desired = w_cap.copy()
                sec_row = panel.sector[r_row]
                for sec_code in np.unique(sec_row[cap_cols]):
                    sec_cols = cap_cols[sec_row[cap_cols] == sec_code]
                    sec_w = float(w_cap[sec_cols].sum())
                    winners = sec_cols[new_state[sec_cols] > 0.0]
                    if winners.size > 0 and sec_w > 0.0:
                        desired[sec_cols] = (1.0 - SECTOR_ACTIVE_SLICE_SHARE) * w_cap[sec_cols]
                        desired[winners] += (SECTOR_ACTIVE_SLICE_SHARE * sec_w) / float(winners.size)
        else:
            desired = w_cap.copy()

        sec_row = panel.sector[r_row]
        ideal_w = project_sector_bounded_weights(desired, w_cap, sec_row)

        if drifted_at_close is not None:
            unchanged = cap_ok & (new_state == prev_state) & (np.abs(ideal_w - drifted_at_close) <= TRADE_DEADBAND)
            if bool(unchanged.any()):
                blended = np.where(unchanged, drifted_at_close, ideal_w)
                ideal_w = project_sector_bounded_weights(blended, w_cap, sec_row)

        weights[k] = ideal_w
        prev_state = new_state
        prev_weights = ideal_w
        prev_decision_row = r_row

    return weights, cap_bench


def _apply_open_fills(decision_weights: FloatArray, tradable: BoolArray) -> FloatArray:
    """Zero any target weight whose `open[r_k + 1]` bar is missing/halted and renormalize filled weights."""
    filled = decision_weights.copy()
    for k in range(filled.shape[0]):
        row = filled[k]
        if not bool(np.isfinite(row).all()):
            continue
        row = np.where(tradable[k], row, 0.0)
        s = float(row.sum())
        if s > 0.0:
            filled[k] = row / s
        else:
            filled[k] = np.nan
    return filled


def _compounded_return(series: FloatArray) -> float | None:
    valid = series[np.isfinite(series)]
    if valid.size == 0:
        return None
    return float(np.prod(1.0 + valid) - 1.0)


def _summarize_excess(
    net_20: FloatArray,
    net_10: FloatArray,
    gross: FloatArray,
    bench: FloatArray,
    years: IntArray,
    horizon: int,
) -> dict[str, Any]:
    valid = np.isfinite(net_20) & np.isfinite(bench)
    if not bool(valid.any()):
        return {
            "periods": 0,
            "gross_excess_per_period": None,
            "net_excess_20bps_per_period": None,
            "net_excess_10bps_per_period": None,
            "net_excess_20bps_annualized": None,
            "net_excess_10bps_annualized": None,
            "newey_west_20bps": {"mean": None, "standard_error": None, "t": None, "count": 0},
            "simple_t_20bps": {"mean": None, "standard_error": None, "t": None, "count": 0},
            "by_year_net_20bps": {},
            "positive_years_20bps": 0,
            "years_scored": 0,
            "compounded_net_20bps": None,
            "compounded_benchmark": None,
            "compounded_net_excess_20bps": None,
        }

    ex_gross = gross[valid] - bench[valid]
    ex_20 = net_20[valid] - bench[valid]
    ex_10 = net_10[valid] - bench[valid]
    yrs = years[valid]
    lag = max(1, 21 // horizon)
    nw = newey_west(ex_20, lag)
    st = simple_t(ex_20)
    periods_per_year = SESSIONS_PER_YEAR / float(horizon)

    by_year: dict[str, dict[str, float | int]] = {}
    for yr in np.unique(yrs):
        y_vals = ex_20[yrs == yr]
        by_year[str(int(yr))] = {"mean": float(y_vals.mean()), "periods": int(y_vals.size)}
    pos_years = sum(1 for item in by_year.values() if float(item["mean"]) > 0.0)

    comp_port = _compounded_return(net_20[valid])
    comp_bench = _compounded_return(bench[valid])
    return {
        "periods": int(valid.sum()),
        "gross_excess_per_period": float(ex_gross.mean()),
        "net_excess_20bps_per_period": float(ex_20.mean()),
        "net_excess_10bps_per_period": float(ex_10.mean()),
        "net_excess_20bps_annualized": float(ex_20.mean()) * periods_per_year,
        "net_excess_10bps_annualized": float(ex_10.mean()) * periods_per_year,
        "newey_west_20bps": {**nw, "lag_periods": lag},
        "simple_t_20bps": st,
        "by_year_net_20bps": by_year,
        "positive_years_20bps": pos_years,
        "years_scored": len(by_year),
        "compounded_net_20bps": comp_port,
        "compounded_benchmark": comp_bench,
        "compounded_net_excess_20bps": (
            None if comp_port is None or comp_bench is None else comp_port - comp_bench
        ),
    }


def _pack_policy_summary(
    path: PathMetrics,
    matched_bench: FloatArray,
    matched_name: str,
    windows: ContinuousWindows,
    horizon: int,
) -> dict[str, Any]:
    steady = path.steady_turnovers[np.isfinite(path.steady_turnovers)]
    act = path.active_shares[np.isfinite(path.active_shares)]
    vs_matched = _summarize_excess(
        path.net_returns_20bps,
        path.net_returns_10bps,
        path.gross_returns,
        matched_bench,
        windows.years,
        horizon,
    )
    vs_spy = _summarize_excess(
        path.net_returns_20bps,
        path.net_returns_10bps,
        path.gross_returns,
        windows.spy_returns,
        windows.years,
        horizon,
    )
    return {
        "matched_benchmark_name": matched_name,
        "mean_steady_one_way_turnover": float(steady.mean()) if steady.size else None,
        "mean_active_share": float(act.mean()) if act.size else None,
        "max_stock_active_abs": path.max_stock_active_abs,
        "max_sector_active_abs": path.max_sector_active_abs,
        "compounded_gross": _compounded_return(path.gross_returns),
        "compounded_net_20bps": _compounded_return(path.net_returns_20bps),
        "compounded_net_10bps": _compounded_return(path.net_returns_10bps),
        "vs_matched_benchmark": vs_matched,
        "vs_spy": vs_spy,
    }


def evaluate_option_c(
    panel: ResearchPanel,
    signals: Mapping[str, FloatArray],
    market_value: FloatArray,
    *,
    first_row: int,
    horizons: tuple[int, ...] = TILT_HORIZONS,
) -> dict[str, Any]:
    """Run all five pre-registered Option C policies across `signals x horizons` and apply the reading rule."""
    per_horizon_universe: dict[str, Any] = {}
    cells: dict[str, dict[str, dict[str, Any]]] = {sig: {} for sig in signals}
    passing_specs: list[dict[str, Any]] = []
    passing_strict_t3: list[dict[str, Any]] = []

    for horizon in horizons:
        h_key = str(horizon)
        windows = continuous_rebalance_windows(panel, first_row, horizon)
        sec_at_decisions = panel.sector[windows.decision_rows]

        # Passive market-cap benchmark path across cap-covered stocks (independent of signal).
        _, cap_bench_dec = build_cap_tilt_weights(
            panel, next(iter(signals.values())), market_value, windows.decision_rows, mode="mult_buffered",
        )
        cap_bench_w = _apply_open_fills(cap_bench_dec, windows.tradable)
        cap_bench_path = simulate_portfolio_path(cap_bench_w, windows.stock_returns)
        ew_all_w = np.where(
            windows.tradable,
            1.0 / np.maximum(windows.tradable.sum(axis=1, keepdims=True), 1),
            0.0,
        )
        ew_active = windows.tradable.sum(axis=1) >= MINIMUM_STOCKS
        ew_all_w[~ew_active] = np.nan
        ew_all_path = simulate_portfolio_path(ew_all_w, windows.stock_returns)

        per_horizon_universe[h_key] = {
            "rebalance_periods": int(len(windows.decision_rows)),
            "first_entry_session": str(panel.sessions[int(windows.entry_rows[0])]),
            "last_exit_session": str(panel.sessions[int(windows.exit_rows[-1])]),
            "mean_tradable_eligible_stocks": float(windows.tradable.sum(axis=1).mean()),
            "mean_cap_covered_stocks": float((cap_bench_w > 0.0).sum(axis=1)[np.isfinite(cap_bench_w).all(axis=1)].mean())
            if bool(np.isfinite(cap_bench_w).all(axis=1).any())
            else 0.0,
            "unfilled_eligible_entries": windows.unfilled_eligible_entries,
            "missing_exit_fallbacks": windows.missing_exit_fallbacks,
            "spy_compounded_return": _compounded_return(windows.spy_returns),
            "cap_subset_compounded_gross_return": _compounded_return(cap_bench_path.gross_returns),
            "cap_subset_mean_steady_turnover": (
                float(cap_bench_path.steady_turnovers[np.isfinite(cap_bench_path.steady_turnovers)].mean())
                if bool(np.isfinite(cap_bench_path.steady_turnovers).any())
                else None
            ),
            "cap_subset_minus_spy": _summarize_excess(
                cap_bench_path.gross_returns,
                cap_bench_path.gross_returns,
                cap_bench_path.gross_returns,
                windows.spy_returns,
                windows.years,
                horizon,
            ),
            "equal_weight_eligible_minus_spy": _summarize_excess(
                ew_all_path.gross_returns,
                ew_all_path.gross_returns,
                ew_all_path.gross_returns,
                windows.spy_returns,
                windows.years,
                horizon,
            ),
        }

        for sig_name, sig_matrix in signals.items():
            # 1. Skill unbuffered (S0) & Skill buffered hold-33% (S1)
            w_s0_dec, b_s0_dec = build_skill_weights(
                panel, sig_matrix, windows.decision_rows, buffered=False,
            )
            w_s0 = _apply_open_fills(w_s0_dec, windows.tradable)
            b_s0 = _apply_open_fills(b_s0_dec, windows.tradable)
            p_s0 = simulate_portfolio_path(w_s0, windows.stock_returns, b_s0)
            b_s0_gross = np.einsum("ki,ki->k", np.where(np.isfinite(b_s0), b_s0, 0.0), windows.stock_returns)
            b_s0_gross[~np.isfinite(b_s0).all(axis=1)] = np.nan

            w_s1_dec, b_s1_dec = build_skill_weights(
                panel, sig_matrix, windows.decision_rows, buffered=True,
            )
            w_s1 = _apply_open_fills(w_s1_dec, windows.tradable)
            b_s1 = _apply_open_fills(b_s1_dec, windows.tradable)
            p_s1 = simulate_portfolio_path(w_s1, windows.stock_returns, b_s1)
            b_s1_gross = np.einsum("ki,ki->k", np.where(np.isfinite(b_s1), b_s1, 0.0), windows.stock_returns)
            b_s1_gross[~np.isfinite(b_s1).all(axis=1)] = np.nan

            # 2. Cap-weighted multiplicative buffered tilt (P1) & Sector active slice tilt (P2)
            w_p1_dec, b_cap1_dec = build_cap_tilt_weights(
                panel, sig_matrix, market_value, windows.decision_rows, mode="mult_buffered",
            )
            w_p1 = _apply_open_fills(w_p1_dec, windows.tradable)
            b_cap1 = _apply_open_fills(b_cap1_dec, windows.tradable)
            p_p1 = simulate_portfolio_path(w_p1, windows.stock_returns, b_cap1, sec_at_decisions)

            w_p2_dec, b_cap2_dec = build_cap_tilt_weights(
                panel, sig_matrix, market_value, windows.decision_rows, mode="sector_active_slice",
            )
            w_p2 = _apply_open_fills(w_p2_dec, windows.tradable)
            b_cap2 = _apply_open_fills(b_cap2_dec, windows.tradable)
            p_p2 = simulate_portfolio_path(w_p2, windows.stock_returns, b_cap2, sec_at_decisions)

            # 3. Core 80% SPY + 20% Buffered Top-10% Satellite (P3)
            k_len, n_sec = windows.stock_returns.shape
            w_p3 = np.full((k_len, n_sec + 1), np.nan, dtype=np.float64)
            s1_active = np.isfinite(w_s1).all(axis=1) & np.isfinite(windows.spy_returns)
            w_p3[s1_active, :n_sec] = (1.0 - CORE_SPY_WEIGHT) * w_s1[s1_active]
            w_p3[s1_active, n_sec] = CORE_SPY_WEIGHT
            b_p3 = np.full((k_len, n_sec + 1), np.nan, dtype=np.float64)
            b_p3[s1_active, :n_sec] = 0.0
            b_p3[s1_active, n_sec] = 1.0
            ret_with_spy = np.column_stack([windows.stock_returns, windows.spy_returns])
            p_p3 = simulate_portfolio_path(w_p3, ret_with_spy, b_p3)

            s0_summary = _pack_policy_summary(p_s0, b_s0_gross, "equal_weight_scored_stocks", windows, horizon)
            s1_summary = _pack_policy_summary(p_s1, b_s1_gross, "equal_weight_scored_stocks", windows, horizon)
            p1_summary = _pack_policy_summary(
                p_p1, cap_bench_path.gross_returns, "own_cap_weighted_constituents", windows, horizon,
            )
            p2_summary = _pack_policy_summary(
                p_p2, cap_bench_path.gross_returns, "own_cap_weighted_constituents", windows, horizon,
            )
            p3_summary = _pack_policy_summary(p_p3, windows.spy_returns, "spy", windows, horizon)

            s1_t = s1_summary["vs_matched_benchmark"]["newey_west_20bps"]["t"]
            s1_mean = s1_summary["vs_matched_benchmark"]["net_excess_20bps_per_period"]
            s1_ok = s1_mean is not None and s1_mean > 0.0 and s1_t is not None and float(s1_t) >= REQUIRED_T_PRIMARY

            for pol_name, pol_summary in (
                ("product_cap_mult_buffered", p1_summary),
                ("product_sector_active_slice", p2_summary),
                ("product_core_spy_80_sat_20", p3_summary),
            ):
                spy_ex = pol_summary["vs_spy"]["net_excess_20bps_per_period"]
                spy_pos_yrs = int(pol_summary["vs_spy"]["positive_years_20bps"])
                matched_t = pol_summary["vs_matched_benchmark"]["newey_west_20bps"]["t"]
                matched_ex = pol_summary["vs_matched_benchmark"]["net_excess_20bps_per_period"]

                passes = bool(
                    spy_ex is not None
                    and spy_ex > 0.0
                    and matched_ex is not None
                    and matched_ex > 0.0
                    and matched_t is not None
                    and float(matched_t) >= REQUIRED_T_PRIMARY
                    and spy_pos_yrs >= REQUIRED_POSITIVE_YEARS
                    and s1_ok
                )
                passes_t3 = bool(passes and matched_t is not None and float(matched_t) >= REQUIRED_T_STRICT)
                pol_summary["passes_reading_rule"] = passes
                pol_summary["passes_strict_t_3_0"] = passes_t3
                if passes:
                    entry_id = {"signal": sig_name, "horizon_sessions": horizon, "policy": pol_name}
                    passing_specs.append(entry_id)
                    if passes_t3:
                        passing_strict_t3.append(entry_id)

            cells[sig_name][h_key] = {
                "skill_top10_unbuffered": s0_summary,
                "skill_top10_hold33": s1_summary,
                "product_cap_mult_buffered": p1_summary,
                "product_sector_active_slice": p2_summary,
                "product_core_spy_80_sat_20": p3_summary,
            }

    return {
        "universe_by_horizon": per_horizon_universe,
        "cells": cells,
        "reading_summary": {
            "rule": STEP4_READING_RULE,
            "passing_specifications": passing_specs,
            "passing_strict_t_3_0_specifications": passing_strict_t3,
            "earns_sealed_period_test": bool(passing_specs),
        },
    }
