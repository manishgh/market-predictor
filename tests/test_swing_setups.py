"""Tests for the ER3 deterministic swing setup population.

No corpus is read. Every panel is generated in the test from an exchange calendar
and a closed-form price path, so a failure here is a defect in the setup rule or
in its causality, never in collected data.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.edge_rebuild.setup_economics import (
    UNSEEN_TICKER_SCOPE,
    WALK_FORWARD_SCOPE,
    evaluate_setup_economics,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_pipeline_steps import SetupComponentsStep
from market_predictor.edge_rebuild.swing_setups import (
    SWING_SETUP_COLUMNS,
    SWING_SETUP_ECONOMICS_CONFIG,
    build_swing_setup_candidates,
    drop_placeholder_bars,
    finalize_swing_setup_population,
    swing_dataset_config,
    swing_setup_mask,
)
from market_predictor.core.errors import DataReadinessError

CALENDAR = xcals.get_calendar("XNYS")
SESSION_COUNT = 620
BENCHMARK = "SPY"
SECTOR_BENCHMARK = "XLK"
INGESTED = pd.Timestamp("2026-01-01T00:00:00Z")

# Six securities with a rising trend and a 21-session wiggle, so price repeatedly
# pulls back below its ten-session EMA and reclaims it on an up bar.
STOCKS = (
    ("AAA", "sec:aaa", 0.0013, 0.035, 0),
    ("BBB", "sec:bbb", 0.0015, 0.040, 4),
    ("CCC", "sec:ccc", 0.0012, 0.030, 8),
    ("DDD", "sec:ddd", 0.0016, 0.038, 12),
    ("EEE", "sec:eee", 0.0014, 0.033, 16),
    ("FFF", "sec:fff", 0.0011, 0.036, 2),
)


# --------------------------------------------------------------------------- #
# Synthetic corpus
# --------------------------------------------------------------------------- #
def _sessions() -> list[date]:
    sessions = CALENDAR.sessions_in_range("2019-01-02", "2022-12-30")
    return [pd.Timestamp(session).date() for session in sessions][:SESSION_COUNT]


def _closes(drift: float, amplitude: float, phase: int, count: int) -> np.ndarray:
    steps = np.arange(count, dtype=float)
    trend = 100.0 * np.exp(drift * steps)
    return trend * (1.0 + amplitude * np.sin(2.0 * np.pi * (steps + phase) / 21.0))


def _bars(ticker: str, sessions: Sequence[date], closes: np.ndarray) -> pd.DataFrame:
    opens = np.concatenate(([closes[0] * 0.999], closes[:-1]))
    frame = pd.DataFrame(
        {
            "date": list(sessions),
            "open": opens,
            "high": np.maximum(opens, closes) * 1.004,
            "low": np.minimum(opens, closes) * 0.996,
            "close": closes,
            "volume": np.where(closes > opens, 1_500_000, 900_000).astype(float),
        }
    )
    return canonicalize_bars(
        frame,
        timeframe="1d",
        ticker=ticker,
        price_feed="sip",
        adjustment="all",
        ingested_at_utc=INGESTED,
    )


def _benchmark_bars(sessions: Sequence[date]) -> pd.DataFrame:
    count = len(sessions)
    return pd.concat(
        [
            _bars(BENCHMARK, sessions, _closes(0.0004, 0.004, 0, count)),
            _bars("QQQ", sessions, _closes(0.0005, 0.005, 3, count)),
            _bars(SECTOR_BENCHMARK, sessions, _closes(0.0005, 0.006, 6, count)),
        ],
        ignore_index=True,
    )


def _stock_bars(sessions: Sequence[date]) -> pd.DataFrame:
    count = len(sessions)
    return pd.concat(
        [
            _bars(ticker, sessions, _closes(drift, amplitude, phase, count))
            for ticker, _, drift, amplitude, phase in STOCKS
        ],
        ignore_index=True,
    )


def _memberships(sessions: Sequence[date]) -> pd.DataFrame:
    start = pd.Timestamp(sessions[0]).tz_localize("UTC")
    return pd.DataFrame(
        [
            {
                "ticker": ticker,
                "security_id": security_id,
                "effective_from_utc": start,
                "effective_to_utc": pd.NaT,
                "available_at_utc": start,
                "sector": "Information Technology",
                "industry": "Semiconductors",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": SECTOR_BENCHMARK,
                "universe_snapshot_id": "test-snapshot",
                "source": "test",
            }
            for ticker, security_id, _, _, _ in STOCKS
        ]
    )


@pytest.fixture(scope="module")
def contract() -> StrategyContract:
    from pathlib import Path

    from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract

    root = Path(__file__).resolve().parents[1]
    return load_strategy_contract(root / "configs" / "edge_rebuild_strategy_contract.toml")


@pytest.fixture(scope="module")
def sessions() -> list[date]:
    return _sessions()


@pytest.fixture(scope="module")
def candidates(contract: StrategyContract, sessions: list[date]) -> pd.DataFrame:
    return build_swing_setup_candidates(
        _stock_bars(sessions),
        _benchmark_bars(sessions),
        _memberships(sessions),
        contract=contract,
    )


@pytest.fixture(scope="module")
def population(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> pd.DataFrame:
    return finalize_swing_setup_population(
        candidates,
        contract=contract,
        sessions=[pd.Timestamp(session) for session in sessions],
        universe_tickers=pd.Series([ticker for ticker, _, _, _, _ in STOCKS]),
    )


# --------------------------------------------------------------------------- #
# Point-in-time causality
# --------------------------------------------------------------------------- #
def test_future_bars_never_change_a_decision(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> None:
    """Poison every bar after one qualifying decision; that decision must not move.

    The cut is the session of a real qualifying setup, so the poison lands inside
    that row's own label window. Its components and its qualification must be
    bit-identical, while its label must change: if the label did not change the
    test would have no power, and if a component changed a future bar reached a
    feature.
    """

    cut = pd.Timestamp(candidates.iloc[len(candidates) // 2]["session"])
    poisoned_stock = _poison(_stock_bars(sessions), cut)
    poisoned_benchmark = _poison(_benchmark_bars(sessions), cut)
    poisoned = build_swing_setup_candidates(
        poisoned_stock,
        poisoned_benchmark,
        _memberships(sessions),
        contract=contract,
    )

    key = ["security_id", "session"]
    components = [
        column
        for column in SWING_SETUP_COLUMNS
        if column not in {"scope", "phase"} and column.startswith(
            ("residual_", "dist_", "prior_", "intraday_", "volume_", "dollar_", "daily_")
        )
    ]
    before = candidates.loc[candidates["session"].le(cut)].set_index(key).sort_index()
    after = poisoned.loc[poisoned["session"].le(cut)].set_index(key).sort_index()

    assert list(before.index) == list(after.index), "a future bar changed which setups fired"
    pd.testing.assert_frame_equal(
        before.loc[:, components],
        after.loc[:, components],
        check_exact=True,
    )
    assert before.loc[:, ["market_regime", "sector"]].equals(after.loc[:, ["market_regime", "sector"]])

    # Power: the poisoned future must actually reach the labels of the cut session.
    cut_rows = before.xs(cut, level="session", drop_level=False)
    assert not cut_rows.empty
    changed = ~np.isclose(
        cut_rows["gross_return"].to_numpy(dtype=float),
        after.loc[cut_rows.index, "gross_return"].to_numpy(dtype=float),
    )
    assert changed.any(), "poisoning the future did not move any label; the test has no power"


def _poison(bars: pd.DataFrame, cut: pd.Timestamp) -> pd.DataFrame:
    """Rewrite every bar after ``cut`` with a session-dependent factor.

    The factor has to grow with distance from the cut. Scaling the whole future by
    one constant would leave every future return ratio unchanged, and the test
    would silently lose its power to detect a leaked label.
    """

    session = bars["bar_start_utc"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    future = session.gt(cut)
    offset = session.loc[future].rank(method="dense").astype(float)
    factor = 1.0 + 0.05 * offset
    poisoned = bars.copy()
    for column in ("open", "high", "low", "close"):
        poisoned.loc[future, column] = poisoned.loc[future, column] * factor
    poisoned.loc[future, "volume"] = poisoned.loc[future, "volume"] * 10.0
    return poisoned


def test_truncating_the_future_leaves_earlier_decisions_identical(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> None:
    """A decision computed with less future data must be the same decision."""

    kept = sessions[: SESSION_COUNT - 40]
    cut = pd.Timestamp(kept[-1]) - pd.Timedelta(days=40)
    truncated = build_swing_setup_candidates(
        _stock_bars(kept),
        _benchmark_bars(kept),
        _memberships(kept),
        contract=contract,
    )
    key = ["security_id", "session"]
    before = candidates.loc[candidates["session"].le(cut)].set_index(key).sort_index()
    after = truncated.loc[truncated["session"].le(cut)].set_index(key).sort_index()
    assert not before.empty
    pd.testing.assert_frame_equal(before, after, check_exact=True)


def test_zero_volume_placeholders_do_not_reach_features(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> None:
    """Placeholder bars are not observations: adding them changes nothing."""

    stock = _stock_bars(sessions)
    placeholders = stock.loc[stock["ticker"].eq("AAA")].tail(30).copy()
    placeholders["volume"] = 0.0
    placeholders["ticker"] = "GGG"
    for column in ("open", "high", "low", "close"):
        placeholders[column] = 50.0
    contaminated = pd.concat([stock, placeholders], ignore_index=True)

    traded, dropped = drop_placeholder_bars(contaminated)
    assert dropped == len(placeholders)
    assert traded["volume"].gt(0).all()

    rebuilt = build_swing_setup_candidates(
        contaminated,
        _benchmark_bars(sessions),
        _memberships(sessions),
        contract=contract,
    )
    pd.testing.assert_frame_equal(rebuilt, candidates, check_exact=True)


# --------------------------------------------------------------------------- #
# The setup rule
# --------------------------------------------------------------------------- #
def _qualifying_components(contract: StrategyContract) -> pd.DataFrame:
    horizon = contract.swing.horizon_sessions
    return pd.DataFrame(
        {
            "residual_return_20d_vs_spy": [0.05],
            "residual_return_20d_vs_sector": [0.04],
            "residual_return_60d_vs_spy": [0.09],
            "residual_return_60d_vs_sector": [0.08],
            "dist_sma_200": [0.12],
            "sma_200_slope_20d": [0.03],
            "prior_dist_ema_10": [-0.01],
            "prior_dist_sma_200": [0.11],
            "dist_ema_10": [0.008],
            "intraday_return": [0.012],
            "volume_ratio_20": [1.4],
            "dollar_volume": [5.0e8],
            "daily_bar_count": [contract.swing.minimum_warmup_sessions],
            "label_window_expected": [True],
            "label_path_exact": [True],
            f"future_gross_return_{horizon}d": [0.03],
            f"future_net_return_{horizon}d": [0.028],
            f"future_excess_return_{horizon}d_vs_spy": [0.02],
            f"future_excess_return_{horizon}d_vs_sector": [0.01],
        }
    )


def test_every_component_is_necessary(contract: StrategyContract) -> None:
    baseline = _qualifying_components(contract)
    assert bool(swing_setup_mask(baseline, contract=contract).iloc[0])

    breakages = {
        "residual_return_20d_vs_spy": 0.0,
        "residual_return_20d_vs_sector": -0.01,
        "residual_return_60d_vs_spy": 0.0,
        "residual_return_60d_vs_sector": -0.02,
        "dist_sma_200": -0.01,
        "sma_200_slope_20d": 0.0,
        "prior_dist_ema_10": 0.005,
        "prior_dist_sma_200": -0.001,
        "dist_ema_10": 0.0,
        "intraday_return": -0.001,
        "volume_ratio_20": 1.0,
        "daily_bar_count": contract.swing.minimum_warmup_sessions - 1,
        "label_path_exact": False,
        "label_window_expected": False,
    }
    for column, value in breakages.items():
        broken = baseline.copy()
        broken[column] = value
        assert not bool(swing_setup_mask(broken, contract=contract).iloc[0]), column


def test_missing_evidence_never_qualifies(contract: StrategyContract) -> None:
    for column in ("dist_ema_10", "residual_return_60d_vs_sector", "volume_ratio_20"):
        broken = _qualifying_components(contract)
        broken[column] = np.nan
        assert not bool(swing_setup_mask(broken, contract=contract).iloc[0]), column


def test_reclaim_confirmation_is_one_shot(candidates: pd.DataFrame) -> None:
    """A standing condition repeats daily; a crossing cannot.

    Every qualifying row must sit below its ten-session EMA on the prior bar and
    above it on the decision bar, so the same pullback leg can only fire once.
    """

    assert not candidates.empty
    assert candidates["prior_dist_ema_10"].lt(0.0).all()
    assert candidates["dist_ema_10"].gt(0.0).all()
    for _, rows in candidates.groupby("security_id"):
        gaps = rows.sort_values("session")["session"].diff().dropna()
        assert gaps.dt.days.min() > 1


def test_warmup_below_the_contract_produces_nothing(
    contract: StrategyContract,
    sessions: list[date],
) -> None:
    short = sessions[: contract.swing.minimum_warmup_sessions - 5]
    built = build_swing_setup_candidates(
        _stock_bars(short),
        _benchmark_bars(short),
        _memberships(short),
        contract=contract,
    )
    assert built.empty


def test_residuals_remove_both_benchmarks(
    contract: StrategyContract,
    sessions: list[date],
) -> None:
    """A stock that trails its sector never qualifies, however strong its own trend."""

    count = len(sessions)
    strong_sector = pd.concat(
        [
            _bars(BENCHMARK, sessions, _closes(0.0004, 0.004, 0, count)),
            _bars("QQQ", sessions, _closes(0.0005, 0.005, 3, count)),
            _bars(SECTOR_BENCHMARK, sessions, _closes(0.0030, 0.006, 6, count)),
        ],
        ignore_index=True,
    )
    built = build_swing_setup_candidates(
        _stock_bars(sessions),
        strong_sector,
        _memberships(sessions),
        contract=contract,
    )
    assert built.empty


# --------------------------------------------------------------------------- #
# Labels and economics
# --------------------------------------------------------------------------- #
def test_cost_is_applied_exactly_once(
    contract: StrategyContract,
    candidates: pd.DataFrame,
) -> None:
    expected = contract.swing.round_trip_cost_bps / 10_000.0
    assert bool(np.isclose(candidates["cost"].to_numpy(dtype=float), expected).all())
    residual = candidates["net_return"] - (candidates["gross_return"] - candidates["cost"])
    assert residual.abs().max() < 1e-12


def test_benchmark_excess_is_already_net_of_that_single_cost(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> None:
    """Excess is ``net - benchmark`` over the identical interval, never re-deducted."""

    benchmarks = _benchmark_bars(sessions)
    benchmarks["session"] = (
        benchmarks["bar_start_utc"].dt.tz_convert("America/New_York").dt.normalize().dt.tz_localize(None)
    )
    lookup = benchmarks.set_index(["ticker", "session"])
    sample = candidates.head(20)
    for row in sample.itertuples(index=False):
        entry = pd.Timestamp(row.entry_session_date_et)
        exit_ = pd.Timestamp(row.exit_session_date_et)
        for ticker, column in ((BENCHMARK, "spy_excess_return"), (SECTOR_BENCHMARK, "sector_excess_return")):
            benchmark_return = (
                lookup.loc[(ticker, exit_), "close"] / lookup.loc[(ticker, entry), "open"] - 1.0
            )
            assert getattr(row, column) == pytest.approx(row.net_return - benchmark_return, abs=1e-12)


def test_exit_is_the_horizon_close_and_entry_the_next_open(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
) -> None:
    horizon = contract.swing.horizon_sessions
    ordinals = {pd.Timestamp(session): index for index, session in enumerate(sessions)}
    for row in candidates.head(50).itertuples(index=False):
        decision = ordinals[pd.Timestamp(row.session)]
        assert ordinals[pd.Timestamp(row.entry_session_date_et)] == decision + 1
        assert ordinals[pd.Timestamp(row.exit_session_date_et)] == decision + horizon


def test_labels_reject_a_broken_cost_identity(candidates: pd.DataFrame) -> None:
    from market_predictor.edge_rebuild.swing_setups import _verify_label_economics

    tampered = candidates.copy()
    tampered.loc[tampered.index[0], "net_return"] = tampered.loc[tampered.index[0], "gross_return"]
    with pytest.raises(DataReadinessError):
        _verify_label_economics(tampered)


# --------------------------------------------------------------------------- #
# Population shape: scope, phase, cap
# --------------------------------------------------------------------------- #
def test_population_matches_the_setup_economics_input_contract(
    population: pd.DataFrame,
) -> None:
    assert list(population.columns) == list(SWING_SETUP_COLUMNS)
    assert set(population["scope"]).issubset({WALK_FORWARD_SCOPE, UNSEEN_TICKER_SCOPE})
    assert not population.duplicated(subset=["security_id", "decision_time_utc"]).any()
    for column in SWING_SETUP_ECONOMICS_CONFIG.required_columns:
        assert column in population.columns


def test_phase_separates_overlapping_labels(
    contract: StrategyContract,
    sessions: list[date],
    population: pd.DataFrame,
) -> None:
    """Within one phase, consecutive decision sessions are a full horizon apart."""

    horizon = contract.swing.horizon_sessions
    ordinals = {pd.Timestamp(session): index for index, session in enumerate(sessions)}
    assert population["phase"].between(0, horizon - 1).all()
    for phase, rows in population.groupby("phase"):
        ordered = sorted({ordinals[pd.Timestamp(session)] for session in rows["session"]})
        assert all(ordinal % horizon == phase for ordinal in ordered)
        gaps = np.diff(np.array(ordered))
        assert gaps.size == 0 or gaps.min() >= horizon


def test_trade_cap_is_applied_per_scope_and_session(
    contract: StrategyContract,
    population: pd.DataFrame,
) -> None:
    counts = population.groupby(["scope", "session"]).size()
    assert counts.max() <= contract.swing.maximum_trades_per_decision


def test_a_security_is_never_split_across_scopes(population: pd.DataFrame) -> None:
    assert population.groupby("security_id")["scope"].nunique().max() == 1
    assert population.groupby("ticker")["scope"].nunique().max() == 1


def test_scope_assignment_is_deterministic(
    contract: StrategyContract,
    sessions: list[date],
    candidates: pd.DataFrame,
    population: pd.DataFrame,
) -> None:
    repeated = finalize_swing_setup_population(
        candidates,
        contract=contract,
        sessions=[pd.Timestamp(session) for session in sessions],
        universe_tickers=pd.Series([ticker for ticker, _, _, _, _ in STOCKS]),
    )
    pd.testing.assert_frame_equal(repeated, population, check_exact=True)


def test_evaluator_accepts_the_emitted_population_schema(
    contract: StrategyContract,
    population: pd.DataFrame,
) -> None:
    """The frame is valid evidence; this small fixture may fail sample readiness."""

    report = evaluate_setup_economics(
        population,
        strategy_id=contract.swing.strategy_id,
        config=SWING_SETUP_ECONOMICS_CONFIG,
    )
    assert report.strategy_id == contract.swing.strategy_id
    assert {scope.scope for scope in report.scopes} == {WALK_FORWARD_SCOPE, UNSEEN_TICKER_SCOPE}
    assert all(gate.gate for scope in report.scopes for gate in scope.gates)
    assert report.ready_for_modeling is False
    assert report.readiness_failure_reasons


def test_an_empty_candidate_frame_fails_closed(
    contract: StrategyContract,
    sessions: list[date],
) -> None:
    with pytest.raises(DataReadinessError):
        finalize_swing_setup_population(
            pd.DataFrame(columns=list(SWING_SETUP_COLUMNS)),
            contract=contract,
            sessions=[pd.Timestamp(session) for session in sessions],
            universe_tickers=pd.Series([ticker for ticker, _, _, _, _ in STOCKS]),
        )


def test_residual_components_require_spy_benchmark_features(
    contract: StrategyContract,
    sessions: list[date],
) -> None:
    config = swing_dataset_config(contract)
    assert config.horizon_sessions == contract.swing.horizon_sessions
    assert config.min_daily_bars == contract.swing.minimum_warmup_sessions
    benchmarks = pd.DataFrame({"ticker": ["XLK"], "session_date_et": [sessions[0]], "return_60d": [0.1]})
    with pytest.raises(DataReadinessError, match="require SPY benchmark features"):
        step = SetupComponentsStep(benchmarks)
        step.transform(pd.DataFrame({"session_date_et": [sessions[0]]}))
