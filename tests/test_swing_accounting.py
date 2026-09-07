from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.training.swing_types import SwingTrainingConfig
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.evaluation import accounting
from market_predictor.swing.features.panel import MANAGED_PATH_NET_RETURN_COLUMNS, MANAGED_PATH_SESSION_ORDINAL_COLUMNS

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture
def inputs(monkeypatch: pytest.MonkeyPatch) -> tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]:
    calendar = xcals.get_calendar("XNYS", start="2024-01-01", end="2024-06-30")
    decisions = tuple(value.date().isoformat() for value in calendar.sessions[:50])
    valuation = [value.date().isoformat() for value in calendar.sessions[1:60]]
    bars = pd.DataFrame([
        {"ticker": ticker, "session_date_et": session, "open": 100.0, "close": 100.0}
        for ticker in ("SPY", "QQQ", "XLK") for session in valuation
    ])

    def interval(
        values: np.ndarray, samples: int, block_sessions: int, seed: int, *, lower_tail_probability: float,
    ) -> dict[str, float | int]:
        # Test the paired API contract without four 20,000-resample runs per case.
        assert samples == 20_000
        assert block_sessions in (20, 40)
        assert seed == 42
        assert lower_tail_probability == pytest.approx(0.05 / 12)
        assert np.isfinite(values).all()
        mean = float(values.mean())
        return {"estimate": mean, "low": mean, "high": mean, "sessions": len(values),
                "bootstrap_samples": samples, "block_sessions": block_sessions}

    monkeypatch.setattr(accounting.resampling, "moving_block_mean_interval", interval)
    empty = pd.DataFrame(columns=[
        "decision_id", "decision_group_id", "security_id", "sector", "session_date_et", "primary_benchmark",
        "barrier_holding_sessions", "barrier_exit_session_date_et", "barrier_cost",
        "barrier_net_return", "barrier_gross_return",
        *MANAGED_PATH_SESSION_ORDINAL_COLUMNS, *MANAGED_PATH_NET_RETURN_COLUMNS,
    ])
    return empty, bars, decisions


def _evaluate(selected: pd.DataFrame, bars: pd.DataFrame, decisions: tuple[str, ...]) -> dict[str, Any]:
    return accounting.evaluate_funded_swing_accounting(
        selected, bars, config=SwingTrainingConfig(),
        strategy_contract=load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"),
        research_contract=load_swing_research_contract(ROOT / "configs/swing_research.toml"),
        session_calendar=decisions,
    )


def test_cash_and_flat_benchmark_zero_identity(inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]) -> None:
    selected, bars, decisions = inputs
    report = _evaluate(selected, bars, decisions)
    assert report["summary"]["net_cagr_difference_vs_spy"] == pytest.approx(0)
    assert report["comparisons"]["base"]["mean_daily_active_return"] == pytest.approx(0)
    assert report["base_ledger"]["final_cash"] == pytest.approx(1)
    assert report["base_ledger"]["final_holdings"] == pytest.approx(0)
    assert report["base_ledger"]["total_cost"] == pytest.approx(0)
    assert report["session_dates"] == bars.loc[bars.ticker.eq("SPY"), "session_date_et"].tolist()
    assert len(report["session_dates"]) == len(decisions) - 1 + 10
    assert report["eligible"] is False
    assert report["summary"]["economic_conditions_passed"] is False
    attribution = report["comparisons"]["base"]["attribution"]
    assert attribution["realized_beta_vs_spy"]["value"] is None
    assert attribution["realized_beta_vs_spy"]["status"] == "unavailable"
    assert attribution["mean_marked_gross_exposure_before_exits"] == 0
    assert attribution["mean_close_holdings_weight"] == 0
    assert attribution["mean_close_cash_weight"] == 1
    assert attribution["mean_close_cash_unit_equity"] == 1
    assert attribution["exposure_matched_spy"]["status"] == "unavailable"
    assert attribution["cash_drag"]["status"] == "unavailable"


def test_positive_economics_and_metadata_cannot_admit_basis(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]],
) -> None:
    selected, bars, decisions = inputs
    mask = bars.ticker.eq("SPY")
    length = int(mask.sum())
    bars.loc[mask, "open"] = 100 * 0.999 ** np.arange(length)
    bars.loc[mask, "close"] = 100 * 0.999 ** np.arange(1, length + 1)
    before = _evaluate(selected, bars, decisions)
    assert before["summary"]["economic_conditions_passed"] is True
    assert before["eligible"] is False
    bars["passed"] = True
    bars["price_basis_verified"] = True
    bars.attrs.update(passed=True, price_basis_verified=True, adjustment="all")
    selected.attrs["passed"] = True
    after = _evaluate(selected, bars, decisions)
    assert after == before
    assert after["summary"]["eligible"] is False
    assert after["eligibility_blockers"] == ["price_basis_pending"]


@pytest.mark.parametrize("mutation", ["missing_spy", "missing_qqq", "duplicate", "nan", "zero", "infinite", "boolean", "string"])
def test_invalid_benchmark_evidence_fails_closed(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]], mutation: str,
) -> None:
    selected, bars, decisions = inputs
    if mutation.startswith("missing_"):
        ticker = mutation.removeprefix("missing_").upper()
        bars = bars.drop(bars.index[bars.ticker.eq(ticker)][-1])
    elif mutation == "duplicate":
        bars = pd.concat([bars, bars.iloc[[0]]], ignore_index=True)
    else:
        bars["close"] = bars["close"].astype(object)
        bars.loc[0, "close"] = {"nan": np.nan, "zero": 0.0, "infinite": np.inf, "boolean": True, "string": "100"}[mutation]
    with pytest.raises(DataReadinessError):
        _evaluate(selected, bars, decisions)


def test_selected_sector_requires_complete_bars(inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]) -> None:
    _, bars, decisions = inputs
    selected = pd.DataFrame({"primary_benchmark": ["XLF"]})
    with pytest.raises(DataReadinessError, match="XLF"):
        _evaluate(selected, bars, decisions)


def test_decision_gap_rejected(inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]]) -> None:
    selected, bars, decisions = inputs
    with pytest.raises(DataReadinessError, match="calendar"):
        _evaluate(selected, bars, decisions[:2] + decisions[3:])


def test_short_calendar_keeps_diagnostics_but_not_inferential_pass(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]],
) -> None:
    selected, bars, decisions = inputs
    report = _evaluate(selected, bars, decisions[:1])
    assert len(report["session_dates"]) == 10
    assert report["comparisons"]["base"]["active_return_ci"]["40"]["low"] is None
    assert report["summary"]["economic_conditions_passed"] is False


def test_cagr_uses_actual_exchange_open_to_close_elapsed_time(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]],
) -> None:
    selected, bars, decisions = inputs
    mask = bars.ticker.eq("SPY")
    bars.loc[mask, "close"] = np.linspace(100, 110, int(mask.sum()))
    report = _evaluate(selected, bars, decisions)
    calendar = xcals.get_calendar("XNYS")
    elapsed = calendar.session_close(report["session_dates"][-1]) - calendar.session_open(report["session_dates"][0])
    years = elapsed.total_seconds() / (365.2425 * 86400)
    assert report["elapsed_years"] == pytest.approx(years)
    assert report["benchmarks"]["SPY"]["cagr"] == pytest.approx(1.1 ** (1 / years) - 1)


def test_fixed_horizon_labels_do_not_supply_daily_benchmark(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]],
) -> None:
    _, bars, decisions = inputs
    selected = pd.DataFrame({"primary_benchmark": ["XLK"], "future_excess_return_10d_vs_spy": [1.0]})
    with pytest.raises(DataReadinessError, match="SPY"):
        _evaluate(selected, bars.loc[~bars.ticker.eq("SPY")], decisions)


def test_flat_stock_cost_is_paid_once_and_stress_uses_same_calendar(
    inputs: tuple[pd.DataFrame, pd.DataFrame, tuple[str, ...]],
) -> None:
    _, bars, decisions = inputs
    dates = bars.loc[bars.ticker.eq("SPY"), "session_date_et"].tolist()[:10]
    row: dict[str, Any] = {
        "decision_id": "one", "decision_group_id": decisions[0], "security_id": "one",
        "sector": "Technology", "session_date_et": decisions[0], "primary_benchmark": "XLK",
        "barrier_holding_sessions": 10, "barrier_exit_session_date_et": dates[-1],
        "barrier_cost": 0.002, "barrier_net_return": -0.002, "barrier_gross_return": 0.0,
        "future_excess_return_10d_vs_spy": -0.002,
    }
    for ordinal, net, session in zip(MANAGED_PATH_SESSION_ORDINAL_COLUMNS, MANAGED_PATH_NET_RETURN_COLUMNS, dates, strict=True):
        row[ordinal] = date.fromisoformat(session).toordinal()
        row[net] = -0.002
    report = _evaluate(pd.DataFrame([row]), bars, decisions)
    assert report["base_ledger"]["total_cost"] == pytest.approx(0.0002)
    assert report["stress_ledger"]["total_cost"] == pytest.approx(0.0004)
    assert report["base_ledger"]["final_cash"] == pytest.approx(0.9998)
    assert report["stress_ledger"]["final_cash"] == pytest.approx(0.9996)
    assert report["base_ledger"]["session_dates"] == report["stress_ledger"]["session_dates"]
    assert report["fixed_horizon_selected_excess"]["spy"]["mean_selected_excess"] == pytest.approx(-0.002)
    for name in ("base", "stress"):
        attribution = report["comparisons"][name]["attribution"]
        records = report[f"{name}_ledger"]["daily_records"]
        assert attribution["mean_close_cash_weight"] + attribution["mean_close_holdings_weight"] == pytest.approx(1)
        assert attribution["mean_marked_gross_exposure_before_exits"] == pytest.approx(
            np.mean([r["gross_exposure"] for r in records])
        )
        assert attribution["mean_close_cash_unit_equity"] == pytest.approx(np.mean([r["cash"] for r in records]))
        assert attribution["mean_marked_gross_exposure_before_exits"] > attribution["mean_close_holdings_weight"]
        assert attribution["exposure_matched_spy"]["status"] == "unavailable"


@pytest.mark.parametrize("slope", [0.0, 2.0, -0.5])
def test_realized_beta_is_centered_covariance_ratio(slope: float) -> None:
    spy = np.array([0.01, -0.02, 0.03, -0.01])
    daily = slope * spy + 0.001
    ledger = {"daily_records": [
        {"cash": 0.75, "holdings": 0.25, "equity": 1.0, "gross_exposure": 0.4}
        for _ in spy
    ]}
    attribution = accounting._attribution_diagnostics(ledger, daily, spy)
    assert attribution["realized_beta_vs_spy"]["status"] == "computed"
    assert attribution["realized_beta_vs_spy"]["value"] == pytest.approx(slope)
    assert attribution["mean_close_cash_weight"] == pytest.approx(0.75)
    assert attribution["mean_close_holdings_weight"] == pytest.approx(0.25)
    assert attribution["mean_marked_gross_exposure_before_exits"] == pytest.approx(0.4)
    assert attribution["exposure_matched_spy"]["status"] == "unavailable"
    assert "start-period" in attribution["exposure_matched_spy"]["reason"]


def test_nonzero_constant_spy_return_has_no_beta() -> None:
    spy = np.full(59, 0.01)
    ledger = {"daily_records": [
        {"cash": 1.0, "holdings": 0.0, "equity": 1.0, "gross_exposure": 0.0}
        for _ in spy
    ]}
    attribution = accounting._attribution_diagnostics(ledger, np.linspace(-0.01, 0.02, len(spy)), spy)
    assert attribution["realized_beta_vs_spy"]["status"] == "unavailable"
    assert attribution["realized_beta_vs_spy"]["value"] is None


def test_stock_identical_to_spy_has_zero_gross_label_excess_but_funded_cash_drag() -> None:
    decisions = ("2024-01-02",)
    sessions = list(accounting.funded_ledger.swing_valuation_sessions(decisions, 10))
    prices = np.linspace(101.0, 110.0, 10)
    bars = pd.DataFrame([
        {"ticker": ticker, "session_date_et": session, "open": 100.0, "close": close}
        for ticker in ("SPY", "QQQ", "XLK") for session, close in zip(sessions, prices, strict=True)
    ])
    row: dict[str, Any] = {
        "decision_id": "same-path", "decision_group_id": decisions[0], "security_id": "stock",
        "sector": "Technology", "session_date_et": decisions[0], "primary_benchmark": "XLK",
        "barrier_holding_sessions": 10, "barrier_exit_session_date_et": sessions[-1],
        "barrier_cost": 0.002, "barrier_net_return": 0.098, "barrier_gross_return": 0.1,
        "future_excess_return_10d_vs_spy": -0.002,
    }
    for ordinal, net, session, close in zip(
        MANAGED_PATH_SESSION_ORDINAL_COLUMNS, MANAGED_PATH_NET_RETURN_COLUMNS, sessions, prices, strict=True,
    ):
        row[ordinal] = date.fromisoformat(session).toordinal()
        row[net] = close / 100.0 - 1.0 - 0.002
    report = _evaluate(pd.DataFrame([row]), bars, decisions)
    spy_gross = report["benchmarks"]["SPY"]["compounded_return"]
    assert row["barrier_gross_return"] - spy_gross == pytest.approx(0)
    assert report["fixed_horizon_selected_excess"]["spy"]["mean_selected_excess"] == pytest.approx(-0.002)
    assert report["base_ledger"]["compounded_return"] == pytest.approx(0.1 * (spy_gross - 0.002))
    assert report["base_ledger"]["compounded_return"] < spy_gross
