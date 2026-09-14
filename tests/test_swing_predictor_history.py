import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.joins import decisions_from_completed_bars
from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import StrategyContract
from market_predictor.swing.features.panel import TECHNICAL_RANKING_FEATURES
from market_predictor.swing.features.predictors import build_swing_predictor_history
from tests.test_swing_features import _canonical_bars
from tests.test_swing_features import contract as contract


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    import exchange_calendars as xcals

    sessions = xcals.get_calendar("XNYS").sessions_in_range("2022-01-03", "2023-04-28")[:300]
    stock = decisions_from_completed_bars(_canonical_bars("AAA", sessions, drift=0.001), mode="swing-nightly")
    stock["security_id"] = "issuer-a"
    stock["sector"] = "Technology"
    stock["primary_benchmark"] = "XLK"
    stock["feature_profile"] = "technical_market"
    stock["membership_available_at_utc"] = pd.Timestamp("2021-12-31T00:00:00Z")
    stock["membership_effective_from_utc"] = pd.Timestamp("2021-12-31T00:00:00Z")
    stock["membership_effective_to_utc"] = pd.Series(pd.NaT, index=stock.index, dtype="datetime64[ns, UTC]")
    stock["universe_snapshot_id"] = "synthetic-membership"
    benchmarks = pd.concat([_canonical_bars(symbol, sessions, drift=drift)
        for symbol, drift in (("SPY", 0.0004), ("QQQ", 0.0005), ("XLK", 0.0006))], ignore_index=True)
    return stock, benchmarks


def test_history_builds_all_base_predictors_without_labels(contract: StrategyContract) -> None:
    stock, benchmarks = _inputs()
    rows = build_swing_predictor_history(stock, benchmarks, contract=contract)
    assert set(TECHNICAL_RANKING_FEATURES).issubset(rows)
    assert rows.daily_bar_count.max() == 300
    assert not any(name.startswith(("future_", "managed_", "barrier_", "forward_")) for name in rows)
    assert rows.kaufman_efficiency_ratio.notna().any()


def test_later_prices_cannot_change_earlier_predictors(contract: StrategyContract) -> None:
    stock, benchmarks = _inputs()
    original = build_swing_predictor_history(stock, benchmarks, contract=contract)
    cutoff = stock.decision_time_utc.iloc[-15]
    later = stock.decision_time_utc.gt(cutoff)
    stock.loc[later, ["open", "high", "low", "close"]] *= 10
    poisoned = build_swing_predictor_history(stock, benchmarks, contract=contract)
    columns = list(TECHNICAL_RANKING_FEATURES)
    left = original.loc[original.decision_time_utc.le(cutoff), columns].to_numpy(float)
    right = poisoned.loc[poisoned.decision_time_utc.le(cutoff), columns].to_numpy(float)
    np.testing.assert_allclose(left, right, equal_nan=True)


def test_history_refuses_future_target_inputs(contract: StrategyContract) -> None:
    stock, benchmarks = _inputs()
    stock["future_net_return_10d"] = 0.2
    with pytest.raises(DataReadinessError, match="outcome columns"):
        build_swing_predictor_history(stock, benchmarks, contract=contract)
