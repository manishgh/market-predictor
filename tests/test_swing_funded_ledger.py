from __future__ import annotations

from datetime import date
from types import SimpleNamespace

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.training.swing_types import SwingTrainingConfig
from market_predictor.modeling.resampling import moving_block_mean_interval
from market_predictor.swing.evaluation.ledger import (
    build_funded_swing_ledger,
    swing_valuation_sessions,
)
from market_predictor.swing.features.panel import (
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
)
from market_predictor.swing.labels.barrier_and_rank import BarrierSpec, apply_triple_barrier


def _fixture(
    count: int = 12, *, path: tuple[float, ...] = (0.0,) * 10,
) -> tuple[pd.DataFrame, tuple[str, ...]]:
    dates = tuple(value.date() for value in xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-06-28"))
    rows: list[dict[str, object]] = []
    for index, decision in enumerate(dates[:count]):
        row: dict[str, object] = {
            "decision_id": f"{decision}|issuer", "decision_group_id": str(decision),
            "security_id": "issuer", "sector": "technology", "session_date_et": str(decision),
            "barrier_holding_sessions": len(path), "barrier_exit_session_date_et": str(dates[index + len(path)]),
            "barrier_cost": 0.002, "barrier_net_return": path[-1] - 0.002, "barrier_gross_return": path[-1],
        }
        for offset in range(10):
            row[MANAGED_PATH_SESSION_ORDINAL_COLUMNS[offset]] = dates[index + offset + 1].toordinal()
            row[MANAGED_PATH_NET_RETURN_COLUMNS[offset]] = path[min(offset, len(path) - 1)] - 0.002
        rows.append(row)
    return pd.DataFrame(rows), tuple(map(str, dates[:count]))


def test_single_lot_charges_cost_once_and_reconciles_gross_net_cash() -> None:
    rows, calendar = _fixture(count=1, path=(0.02, 0.10))
    ledger = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    assert ledger["total_cost"] == pytest.approx(0.1 * 0.002)
    assert ledger["compounded_return"] == pytest.approx(0.1 * (0.10 - 0.002))
    assert ledger["final_cash"] == pytest.approx(1.0098)
    assert ledger["final_holdings"] == 0.0
    assert ledger["sessions"] == 10
    assert ledger["daily_records"][0]["cash"] == pytest.approx(0.8998)
    assert ledger["daily_records"][0]["unrealized_net_pnl"] == pytest.approx(0.0018)
    assert ledger["daily_records"][-1]["realized_net_pnl"] == pytest.approx(0.0098)


@pytest.mark.parametrize("bar,expected_price", [
    ((96.0, 97.0, 95.0, 96.5), 96.0),
    ((106.0, 107.0, 105.0, 106.5), 103.0),
    ((100.0, 110.0, 90.0, 100.0), 98.5),
])
def test_shared_barrier_gap_and_collision_fills_reconcile_to_funded_cash(
    bar: tuple[float, float, float, float], expected_price: float,
) -> None:
    dates = tuple(str(value.date()) for value in xcals.get_calendar("XNYS").sessions_in_range("2024-01-02", "2024-01-31"))
    bars = pd.DataFrame([(session, 100.0, 100.0, 100.0, 100.0) for session in dates],
                        columns=["session", "open", "high", "low", "close"])
    bars.loc[2, ["open", "high", "low", "close"]] = bar
    outcome = apply_triple_barrier(
        bars, pd.DataFrame({"session": [dates[0]], "atr": [1.0]}),
        spec=BarrierSpec(target_atr_multiple=3.0, stop_atr_multiple=1.5, horizon_sessions=10),
    ).iloc[0]
    assert outcome["exit_price"] == pytest.approx(expected_price)
    assert outcome["holding_sessions"] == 2
    rows, calendar = _fixture(count=1, path=(0.0, float(outcome["exit_price"]) / 100.0 - 1.0))
    ledger = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    assert ledger["final_cash"] == pytest.approx(1.0 + 0.1 * (expected_price / 100.0 - 1.0 - 0.002))
    assert ledger["daily_records"][1]["holdings"] == 0.0


def test_overlapping_repeated_lots_are_cash_capped_before_same_day_exit_proceeds() -> None:
    rows, calendar = _fixture()
    ledger = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    daily = ledger["daily_records"]
    assert min(row["cash"] for row in daily) >= 0.0
    assert ledger["maximum_gross_exposure"] <= 1.0 + 1e-12
    assert daily[9]["funding_scale"] < 1.0
    assert daily[9]["exit_value"] > 0.0
    assert daily[9]["cash"] == pytest.approx(daily[9]["exit_value"])
    assert ledger["maximum_sector_weight"] > 0.9
    assert max(row["marked_security_values_before_exits"].get("issuer", 0.0) for row in daily) > 0.9
    for row in daily:
        assert row["equity"] == pytest.approx(row["cash"] + row["holdings"])
        assert row["equity"] - 1.0 == pytest.approx(row["realized_net_pnl"] + row["unrealized_net_pnl"])
        assert row["equity"] - 1.0 == pytest.approx(
            row["realized_gross_pnl"] + row["unrealized_gross_pnl"] - row["cumulative_cost"]
        )


def test_double_cost_changes_cash_once_not_path_returns_or_exit_fee() -> None:
    rows, calendar = _fixture(count=1)
    base = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    stress = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar, additional_round_trip_cost=0.002)
    assert stress["total_cost"] == pytest.approx(2 * base["total_cost"])
    assert stress["final_cash"] == pytest.approx(0.9996)
    assert sum(row["cost"] > 0 for row in stress["daily_records"]) == 1


def test_full_cash_calendar_and_maturation_tail_do_not_depend_on_selection() -> None:
    rows, calendar = _fixture()
    cash = build_funded_swing_ledger(rows.iloc[:0], SwingTrainingConfig(), session_calendar=calendar)
    partial = build_funded_swing_ledger(rows.iloc[:1], SwingTrainingConfig(), session_calendar=calendar)
    assert cash["session_dates"] == partial["session_dates"] == list(swing_valuation_sessions(calendar, 10))
    assert cash["sessions"] == len(calendar) + 9
    assert cash["daily_returns"] == [0.0] * cash["sessions"]
    assert cash["final_cash"] == 1.0


def test_mark_to_market_drawdown_is_not_only_terminal_trade_pnl() -> None:
    rows, calendar = _fixture(count=1, path=(0.20, -0.10, 0.10))
    ledger = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    assert ledger["compounded_return"] > 0
    assert ledger["max_drawdown"] == pytest.approx(1 - 0.9898 / 1.0198)


def test_split_adjusted_units_do_not_create_free_distribution_credit() -> None:
    rows, calendar = _fixture(count=1)
    before = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    rows["cash_dividend"] = 100.0
    rows["split_ratio"] = 10.0
    after = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    assert before == after
    assert after["final_cash"] < 1
    assert after["distribution_policy"] == "no_separate_cash_distribution_credit"


def test_unused_marks_after_managed_exit_do_not_extend_the_holding() -> None:
    rows, calendar = _fixture(count=1, path=(0.02, 0.10))
    before = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    rows[MANAGED_PATH_NET_RETURN_COLUMNS[-1]] = 1_000_000.0
    assert before == build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)


@pytest.mark.parametrize("column,value", [
    ("barrier_cost", 0.004), ("barrier_cost", float("nan")),
    ("barrier_holding_sessions", True), ("barrier_holding_sessions", 1.5),
    ("barrier_holding_sessions", 11), ("barrier_net_return", 0.30),
    (MANAGED_PATH_NET_RETURN_COLUMNS[0], float("nan")),
    (MANAGED_PATH_NET_RETURN_COLUMNS[0], -2.0),
    (MANAGED_PATH_SESSION_ORDINAL_COLUMNS[0], date(2024, 1, 4).toordinal()),
    (MANAGED_PATH_SESSION_ORDINAL_COLUMNS[1], date(2024, 1, 3).toordinal()),
    ("barrier_exit_session_date_et", "2024-01-06"), ("session_date_et", "2024-01-01"),
])
def test_malformed_or_mismatched_paths_fail_closed(column: str, value: object) -> None:
    rows, calendar = _fixture(count=1)
    rows[column] = value
    with pytest.raises(DataReadinessError):
        build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)


def test_duplicate_rows_and_multiple_cohorts_cannot_multiply_capital() -> None:
    rows, calendar = _fixture(count=1)
    with pytest.raises(DataReadinessError, match="unique decisions"):
        build_funded_swing_ledger(pd.concat([rows, rows]), SwingTrainingConfig(), session_calendar=calendar)
    second = rows.copy()
    second["decision_id"] = "another"
    second["security_id"] = "another"
    second["decision_group_id"] = "another"
    with pytest.raises(DataReadinessError, match="one decision cohort"):
        build_funded_swing_ledger(pd.concat([rows, second]), SwingTrainingConfig(), session_calendar=calendar)


@pytest.mark.parametrize("column", ["decision_group_id", "session_date_et", "security_id"])
def test_missing_group_identity_cannot_silently_drop_selected_trades(column: str) -> None:
    rows, calendar = _fixture(count=1)
    rows[column] = None
    with pytest.raises(DataReadinessError, match="identity"):
        build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)


def test_duplicate_columns_cannot_change_ledger_selection() -> None:
    rows, calendar = _fixture(count=1)
    duplicate = pd.concat([rows, rows[["decision_id"]]], axis=1)
    with pytest.raises(DataReadinessError, match="unique column"):
        build_funded_swing_ledger(duplicate, SwingTrainingConfig(), session_calendar=calendar)


@pytest.mark.parametrize("changed", [
    {"horizon_sessions": 9}, {"maximum_trades_per_decision": 0},
    {"maximum_trades_per_decision": True}, {"expected_round_trip_cost_bps": -20.0},
    {"expected_round_trip_cost_bps": float("nan")}, {"expected_round_trip_cost_bps": True},
])
def test_structural_ledger_config_cannot_waive_capital_rules(changed: dict[str, object]) -> None:
    rows, calendar = _fixture(count=1)
    config = SimpleNamespace(**{
        "horizon_sessions": 10, "maximum_trades_per_decision": 25,
        "expected_round_trip_cost_bps": 20.0, **changed,
    })
    with pytest.raises(DataReadinessError, match="ledger requires"):
        build_funded_swing_ledger(rows, config, session_calendar=calendar)


@pytest.mark.parametrize("calendar", [(), ("2024-01-06",), ("20240102",), ("2024-01-02", "2024-01-04")])
def test_non_exchange_or_incomplete_decision_calendars_fail(calendar: tuple[str, ...]) -> None:
    with pytest.raises(DataReadinessError):
        swing_valuation_sessions(calendar, 10)


def test_shuffled_input_replays_identically() -> None:
    rows, calendar = _fixture()
    original = build_funded_swing_ledger(rows, SwingTrainingConfig(), session_calendar=calendar)
    assert original == build_funded_swing_ledger(rows.sample(frac=1, random_state=7), SwingTrainingConfig(), session_calendar=calendar)


def test_bootstrap_cannot_silently_remove_missing_dates() -> None:
    values = np.zeros(60)
    values[3] = np.nan
    with pytest.raises(DataReadinessError, match="complete finite"):
        moving_block_mean_interval(values, 2000, 20, 42)


def test_bootstrap_applies_frozen_lower_tail_probability() -> None:
    values = np.linspace(-0.02, 0.03, 80)
    ordinary = moving_block_mean_interval(values, 2000, 20, 42)
    corrected = moving_block_mean_interval(values, 2000, 20, 42, lower_tail_probability=0.05 / 12)
    assert corrected["low"] < ordinary["low"]
    assert corrected["estimate"] == ordinary["estimate"]
    assert corrected["lower_tail_probability"] == pytest.approx(0.05 / 12)
