"""Paired funded-account diagnostics; adjusted-price basis is not yet admitted."""

from __future__ import annotations

import math
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling import resampling
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.contracts.research import SwingResearchContract
from market_predictor.swing.evaluation import ledger as funded_ledger
from market_predictor.swing.evaluation.ledger import SwingLedgerConfig


def _valuation_calendar(decisions: tuple[str, ...], horizon: int) -> tuple[list[str], float]:
    sessions = list(funded_ledger.swing_valuation_sessions(decisions, horizon))
    try:
        calendar = xcals.get_calendar("XNYS")
        elapsed = calendar.session_close(sessions[-1]) - calendar.session_open(sessions[0])
        years = elapsed.total_seconds() / (365.2425 * 24 * 60 * 60)
    except (TypeError, ValueError, IndexError, KeyError) as exc:
        raise DataReadinessError(f"invalid accounting calendar: {exc}") from exc
    return sessions, float(years)


def _cagr(final_equity: float, years: float) -> float:
    try:
        result = math.expm1(math.log(final_equity) / years)
    except (ValueError, OverflowError, ZeroDivisionError) as exc:
        raise DataReadinessError("accounting CAGR is not finite") from exc
    if not math.isfinite(result):
        raise DataReadinessError("accounting CAGR is not finite")
    return result


def _benchmark_curves(
    bars: pd.DataFrame, selected: pd.DataFrame, sessions: list[str], years: float,
) -> dict[str, dict[str, Any]]:
    required = {"ticker", "session_date_et", "open", "close"}
    if not bars.columns.is_unique or not required.issubset(bars.columns):
        raise DataReadinessError("benchmark bars require unique ticker/session/open/close columns")
    tickers = {"SPY", "QQQ"}
    if not selected.empty:
        if "primary_benchmark" not in selected or selected["primary_benchmark"].isna().any():
            raise DataReadinessError("selected rows require point-in-time sector benchmarks")
        sectors = selected["primary_benchmark"].astype(str)
        if sectors.str.strip().eq("").any() or not sectors.eq(sectors.str.strip().str.upper()).all():
            raise DataReadinessError("selected sector benchmark identity is invalid")
        tickers.update(sectors)
    if bars[["ticker", "session_date_et"]].isna().any().any():
        raise DataReadinessError("benchmark identities must be complete")
    frame = bars.loc[:, sorted(required)].copy()
    frame["session_date_et"] = frame["session_date_et"].astype(str)
    frame["ticker"] = frame["ticker"].astype(str)
    if frame.duplicated(["ticker", "session_date_et"]).any():
        raise DataReadinessError("duplicate benchmark session")
    curves: dict[str, dict[str, Any]] = {}
    for ticker in sorted(tickers):
        rows = frame.loc[frame["ticker"].eq(ticker) & frame["session_date_et"].isin(sessions)]
        if set(rows["session_date_et"]) != set(sessions):
            raise DataReadinessError(f"{ticker} benchmark lacks exact valuation calendar coverage")
        rows = rows.set_index("session_date_et").loc[sessions]
        if rows[["open", "close"]].map(lambda value: isinstance(value, (str, bool, np.bool_))).any().any():
            raise DataReadinessError(f"{ticker} benchmark prices cannot be strings or booleans")
        prices = rows[["open", "close"]].apply(pd.to_numeric, errors="coerce").to_numpy(dtype="float64")
        if not np.isfinite(prices).all() or (prices <= 0).any():
            raise DataReadinessError(f"{ticker} benchmark prices must be positive and finite")
        equity = prices[:, 1] / prices[0, 0]
        daily = equity / np.concatenate(([1.0], equity[:-1])) - 1.0
        if not np.isfinite(equity).all() or not np.isfinite(daily).all():
            raise DataReadinessError("benchmark curve is non-finite")
        curves[ticker] = {
            "session_dates": sessions, "equity": equity.tolist(), "daily_returns": daily.tolist(),
            "compounded_return": float(equity[-1] - 1), "cagr": _cagr(float(equity[-1]), years),
            "entry_reference": "first_valuation_session_open", "transaction_cost_bps": 0.0,
            "price_basis": "price_ratio_diagnostics", "separate_distributions_added": False,
        }
    return curves


def _active_intervals(values: np.ndarray, research: SwingResearchContract) -> dict[str, dict[str, Any]]:
    intervals: dict[str, dict[str, Any]] = {}
    for block in (research.bootstrap_block_sessions, research.sensitivity_block_sessions):
        if len(values) < block:
            intervals[str(block)] = {
                "status": "insufficient_sessions", "estimate": float(values.mean()), "low": None, "high": None,
                "sessions": len(values), "block_sessions": block, "bootstrap_samples": 0,
                "lower_tail_probability": research.familywise_lower_tail_probability,
            }
        else:
            intervals[str(block)] = {
                **resampling.moving_block_mean_interval(
                    values, research.bootstrap_samples, block, research.random_seed,
                    lower_tail_probability=research.familywise_lower_tail_probability,
                ),
                "status": "computed", "lower_tail_probability": research.familywise_lower_tail_probability,
            }
    return intervals


def _fixed_horizon_diagnostics(selected: pd.DataFrame) -> dict[str, dict[str, Any]]:
    diagnostics: dict[str, dict[str, Any]] = {}
    for benchmark in ("spy", "qqq", "sector"):
        column = f"future_excess_return_10d_vs_{benchmark}"
        values = pd.to_numeric(selected[column], errors="coerce") if column in selected else pd.Series(dtype="float64")
        valid = np.isfinite(values.to_numpy(dtype="float64"))
        complete = len(values) > 0 and bool(valid.all())
        diagnostics[benchmark] = {
            "status": "computed" if complete else "unavailable_or_incomplete",
            "mean_selected_excess": float(values.mean()) if complete else None,
            "selected_rows": len(selected), "available_rows": int(valid.sum()),
            "role": "fixed_horizon_label_diagnostic_not_daily_benchmark_or_gate",
        }
    return diagnostics


def _attribution_diagnostics(
    ledger: dict[str, Any], daily: np.ndarray, spy_daily: np.ndarray,
) -> dict[str, Any]:
    """Descriptive price-ratio attribution, never causal timing or gate evidence."""
    centered_spy = spy_daily - spy_daily.mean()
    variance_sum = float(centered_spy @ centered_spy)
    beta = None
    if len(spy_daily) > 1 and np.ptp(spy_daily) > 0 and variance_sum > 0:
        beta = float((daily - daily.mean()) @ centered_spy / variance_sum)
        if not math.isfinite(beta):
            beta = None
    records = ledger["daily_records"]
    timing_limitation = (
        "Ledger gross exposure uses same-session marks before exits, not start-period weights. "
        "Lagged closing holdings omit next-open purchases and intraday exits; daily paths do not "
        "identify their separate overnight/intraday returns. No exposure-matched return or cash "
        "opportunity-cost decomposition is inferred."
    )
    return {
        "status": "price_ratio_diagnostics", "role": "descriptive_only_not_an_eligibility_gate",
        "realized_beta_vs_spy": {
            "status": "computed" if beta is not None else "unavailable",
            "value": beta, "definition": "covariance(portfolio_net_daily_return, SPY_daily_return)/variance(SPY_daily_return)",
            "unavailable_reason": None if beta is not None else "insufficient_or_zero_variance_or_nonfinite_result",
        },
        "mean_marked_gross_exposure_before_exits": float(np.mean([r["gross_exposure"] for r in records])),
        "mean_close_holdings_weight": float(np.mean([r["holdings"] / r["equity"] for r in records])),
        "mean_close_cash_weight": float(np.mean([r["cash"] / r["equity"] for r in records])),
        "mean_close_cash_unit_equity": float(np.mean([r["cash"] for r in records])),
        "cash_yield_assumed": 0.0,
        "exposure_matched_spy": {"status": "unavailable", "reason": timing_limitation},
        "cash_drag": {"status": "unavailable", "reason": timing_limitation},
        "limitations": (
            "Ex-post beta includes cash and transaction costs, not causal stock-selection alpha. "
            "The first return is first-open to close; subsequent observations are daily NAV and "
            "SPY close-to-close returns including idle/tail sessions. Means are equally session-weighted. "
            "Closing cash weight is descriptive, not a measured return drag. "
            "Sector net PnL, prepaid costs and two-sided turnover remain in the canonical ledger. "
            "Independent total-return source reconciliation remains unavailable."
        ),
    }


def evaluate_funded_swing_accounting(
    selected: pd.DataFrame, benchmark_bars: pd.DataFrame, *, config: SwingLedgerConfig,
    strategy_contract: StrategyContract, research_contract: SwingResearchContract,
    session_calendar: tuple[str, ...],
) -> dict[str, Any]:
    """Return base/stress ledgers, benchmark curves, comparisons and compact summary.

    `summary` includes net_cagr_difference_vs_spy, active_return_ci (base/stress,
    then block-length strings), condition_checks, economic_conditions_passed,
    eligible=False, price_basis_status and research_contract_sha256. Extra input
    metadata is never evidence admission. No raw fills or total returns are claimed.
    Each comparison's attribution contains ex-post beta and session-mean exposure/
    cash diagnostics; timing-dependent exposure matching and cash drag are unavailable.
    """
    research_contract.assert_strategy_matches(strategy_contract)
    if (config.horizon_sessions != research_contract.horizon_sessions
            or config.expected_round_trip_cost_bps != research_contract.base_round_trip_cost_bps):
        raise DataReadinessError("accounting config cost/horizon differs from research contract")
    sessions, years = _valuation_calendar(session_calendar, config.horizon_sessions)
    benchmarks = _benchmark_curves(benchmark_bars, selected, sessions, years)
    base = funded_ledger.build_funded_swing_ledger(selected, config, session_calendar=session_calendar)
    stress = funded_ledger.build_funded_swing_ledger(
        selected, config, session_calendar=session_calendar,
        additional_round_trip_cost=(research_contract.stress_cost_multiplier - 1)
        * research_contract.base_round_trip_cost_bps / 10_000,
    )
    spy_daily = np.asarray(benchmarks["SPY"]["daily_returns"], dtype="float64")
    comparisons: dict[str, dict[str, Any]] = {}
    checks: dict[str, bool] = {}
    for name, ledger in (("base", base), ("stress", stress)):
        if ledger["session_dates"] != sessions:
            raise DataReadinessError("ledger and benchmark valuation calendars differ")
        daily = np.asarray(ledger["daily_returns"], dtype="float64")
        if daily.shape != spy_daily.shape or not np.isfinite(daily).all() or (daily <= -1).any():
            raise DataReadinessError("ledger daily returns must be finite and calendar-aligned")
        final = float(ledger["final_cash"]) + float(ledger["final_holdings"])
        if not math.isclose(float(np.prod(1 + daily)), final, rel_tol=1e-10, abs_tol=1e-10):
            raise DataReadinessError("ledger daily returns do not reconcile to final equity")
        active = daily - spy_daily
        intervals = _active_intervals(active, research_contract)
        cagr = _cagr(final, years)
        excess_cagr = cagr - float(benchmarks["SPY"]["cagr"])
        comparisons[name] = {
            "cagr": cagr, "net_cagr_difference_vs_spy": excess_cagr,
            "mean_daily_active_return": float(active.mean()), "daily_active_returns": active.tolist(),
            "active_return_ci": intervals,
            "attribution": _attribution_diagnostics(ledger, daily, spy_daily),
        }
        checks[f"{name}_net_cagr_exceeds_spy"] = excess_cagr > 0
        checks[f"{name}_mean_daily_active_return_positive"] = float(active.mean()) > 0
        for block, interval in intervals.items():
            checks[f"{name}_{block}_session_active_ci_low_positive"] = (
                interval["status"] == "computed" and float(interval["low"]) > 0
            )
        for metric, ceiling in (
            ("max_drawdown", research_contract.maximum_drawdown),
            ("maximum_sector_weight", strategy_contract.swing.hard_maximum_sector_weight),
            ("maximum_gross_exposure", research_contract.maximum_gross_exposure),
        ):
            value = float(ledger[metric])
            if not math.isfinite(value) or value < 0:
                raise DataReadinessError(f"ledger {metric} is invalid")
            checks[f"{name}_{metric}_within_limit"] = value <= ceiling + 1e-12
    summary = {
        "net_cagr_difference_vs_spy": comparisons["base"]["net_cagr_difference_vs_spy"],
        "stress_net_cagr_difference_vs_spy": comparisons["stress"]["net_cagr_difference_vs_spy"],
        "active_return_ci": {name: item["active_return_ci"] for name, item in comparisons.items()},
        "condition_checks": checks, "economic_conditions_passed": all(checks.values()),
        "eligible": False, "price_basis_status": "price_basis_pending",
        "research_contract_sha256": research_contract.sha256(),
    }
    return {
        "schema_version": "market_predictor.swing_funded_accounting.v1",
        "status": "price_ratio_diagnostics", "eligible": False,
        "price_basis_status": "price_basis_pending", "eligibility_blockers": ["price_basis_pending"],
        "basis_explanation": "Price ratios lack independently bound total-return reconciliation; metadata flags cannot admit proof.",
        "base_ledger": base, "stress_ledger": stress, "benchmarks": benchmarks,
        "comparisons": comparisons, "summary": summary,
        "fixed_horizon_selected_excess": _fixed_horizon_diagnostics(selected),
        "session_dates": sessions, "elapsed_years": years,
        "annualization": "XNYS_first_open_to_last_close_elapsed_seconds_over_365.2425_days",
        "benchmark_cost_policy": "frictionless_buy_and_hold_price_ratio_comparator",
        "managed_benchmark_role": "approximate_diagnostic_only_not_used",
        "strategy_contract_sha256": strategy_contract.sha256(),
        "research_contract_sha256": research_contract.sha256(),
    }
