from __future__ import annotations

from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.edge_rebuild.cross_sectional import RANK_SUFFIX
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.swing_features import (
    CATALYST_RANKING_FEATURES,
    TECHNICAL_RANKING_FEATURES,
    build_swing_feature_rows,
    finalize_swing_feature_panel,
    swing_model_feature_columns,
)
from market_predictor.v3.errors import DataReadinessError


@pytest.fixture(scope="module")
def contract() -> StrategyContract:
    return load_strategy_contract(
        Path(__file__).resolve().parents[1]
        / "configs"
        / "edge_rebuild_strategy_contract.toml"
    )


def _panel(*, sessions: int = 2, securities: int = 60) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for session_number in range(sessions):
        session = pd.Timestamp("2024-01-02") + pd.offsets.BDay(session_number)
        for security_number in range(securities):
            base = float(security_number + 1 + session_number)
            row: dict[str, object] = {
                "security_id": f"sec:{security_number:03d}",
                "ticker": f"T{security_number:03d}",
                "session_date_et": session.date(),
                "sector": "Technology",
                "feature_profile": "technical_market",
                "feature_eligible": True,
                "cross_section_eligible": False,
                "daily_bar_count": 300,
                "forward_return": (base - 30.0) / 1_000.0,
                "barrier_label": 1 if base > 30.0 else -1,
            }
            for index, feature in enumerate(TECHNICAL_RANKING_FEATURES):
                row[feature] = base * float(index + 1)
            rows.append(row)
    return pd.DataFrame.from_records(rows)


def test_complete_panel_has_one_row_and_both_labels(
    contract: StrategyContract,
) -> None:
    output = finalize_swing_feature_panel(_panel(), contract=contract)

    assert not output.duplicated(["security_id", "session_date_et"]).any()
    assert output.columns.tolist().count("cross_section_eligible") == 1
    assert output["cross_section_eligible"].all()
    assert output["barrier_label"].notna().all()
    assert output["rank_label"].notna().all()
    assert "forward_return" in output
    assert any(
        column.endswith(RANK_SUFFIX)
        for column in output.columns
    )
    for session_rows in output.groupby("session_date_et", sort=False):
        labels = session_rows[1]["rank_label"]
        assert int(labels.eq(1).sum()) == 12
        assert int(labels.eq(-1).sum()) == 12


def test_later_session_cannot_change_earlier_cross_section(
    contract: StrategyContract,
) -> None:
    source = _panel()
    before = finalize_swing_feature_panel(
        source.loc[source["session_date_et"].eq(source["session_date_et"].min())],
        contract=contract,
    )
    poisoned = source.copy()
    later = poisoned["session_date_et"].eq(poisoned["session_date_et"].max())
    poisoned.loc[later, list(TECHNICAL_RANKING_FEATURES)] *= -10_000.0
    after = finalize_swing_feature_panel(poisoned, contract=contract)
    after = after.loc[
        after["session_date_et"].eq(after["session_date_et"].min())
    ].reset_index(drop=True)

    columns = list(swing_model_feature_columns(contract=contract, catalyst=False))
    pd.testing.assert_frame_equal(
        before.loc[:, columns],
        after.loc[:, columns],
        check_exact=True,
    )
    pd.testing.assert_series_equal(
        before["rank_label"],
        after["rank_label"],
        check_exact=True,
    )


def test_under_warm_rows_do_not_enter_peer_scaling(
    contract: StrategyContract,
) -> None:
    source = _panel(securities=61)
    cold = source["security_id"].eq("sec:060")
    source.loc[cold, "daily_bar_count"] = 249
    source.loc[cold, list(TECHNICAL_RANKING_FEATURES)] = 1e12

    output = finalize_swing_feature_panel(source, contract=contract)
    rank_columns = [
        column
        for column in output.columns
        if column.endswith(("_xs_z", "_xs_rank", "_sector_z"))
    ]

    assert output.loc[cold, rank_columns].isna().all(axis=None)
    assert (
        output.loc[~cold, "return_5d_xs_rank"].max()
        == pytest.approx(1.0)
    )


def test_all_under_warm_partition_retains_null_transforms(
    contract: StrategyContract,
) -> None:
    source = _panel()
    source["daily_bar_count"] = contract.swing.minimum_warmup_sessions - 1

    output = finalize_swing_feature_panel(source, contract=contract)

    rank_columns = [
        column
        for column in output.columns
        if column.endswith(("_xs_z", "_xs_rank", "_sector_z"))
    ]
    assert len(output) == len(source)
    assert output[rank_columns].isna().all(axis=None)
    assert output["rank_label"].isna().all()
    assert not output["cross_section_eligible"].any()


def test_catalyst_raw_counts_never_enter_estimator_schema(
    contract: StrategyContract,
) -> None:
    columns = swing_model_feature_columns(contract=contract, catalyst=True)

    assert set(CATALYST_RANKING_FEATURES).isdisjoint(columns)
    assert "event_count_1d_xs_rank" in columns
    assert "source_count_reddit_3d_sector_z" in columns


def test_incomplete_population_is_refused(
    contract: StrategyContract,
) -> None:
    with pytest.raises(DataReadinessError, match="missing expected securities"):
        finalize_swing_feature_panel(
            _panel(securities=59),
            contract=contract,
            expected_security_ids=[f"sec:{index:03d}" for index in range(60)],
        )


def test_duplicate_security_session_is_refused(
    contract: StrategyContract,
) -> None:
    source = _panel()
    duplicated = pd.concat([source, source.iloc[[0]]], ignore_index=True)

    with pytest.raises(DataReadinessError, match="one row"):
        finalize_swing_feature_panel(duplicated, contract=contract)


def _canonical_bars(
    ticker: str,
    sessions: pd.DatetimeIndex,
    *,
    drift: float,
) -> pd.DataFrame:
    steps = np.arange(len(sessions), dtype=float)
    closes = 100.0 * np.exp(drift * steps) * (
        1.0 + 0.015 * np.sin(steps / 7.0)
    )
    opens = np.concatenate(([closes[0] * 0.999], closes[:-1]))
    raw = pd.DataFrame(
        {
            "date": [pd.Timestamp(session).date() for session in sessions],
            "open": opens,
            "high": np.maximum(opens, closes) * 1.006,
            "low": np.minimum(opens, closes) * 0.994,
            "close": closes,
            "volume": np.where(closes >= opens, 1_500_000, 900_000),
        }
    )
    return canonicalize_bars(
        raw,
        timeframe="1d",
        ticker=ticker,
        price_feed="sip",
        adjustment="all",
        ingested_at_utc=pd.Timestamp("2025-06-01T00:00:00Z"),
    )


def test_row_builder_uses_shared_relationships_and_barrier_labels(
    contract: StrategyContract,
) -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2023-01-03", "2024-06-28")[:300]
    stock = _canonical_bars("AAA", sessions, drift=0.001)
    benchmarks = pd.concat(
        [
            _canonical_bars("SPY", sessions, drift=0.0004),
            _canonical_bars("QQQ", sessions, drift=0.0005),
            _canonical_bars("XLK", sessions, drift=0.0006),
        ],
        ignore_index=True,
    )
    effective_from = pd.Timestamp(sessions[0]).tz_localize("UTC")
    memberships = pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security_id": ["sec:aaa"],
            "effective_from_utc": [effective_from],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [effective_from],
            "sector": ["Technology"],
            "industry": ["Software"],
            "market_cap_bucket": ["large_cap_sp500"],
            "liquidity_bucket": ["sp500_constituent"],
            "primary_benchmark": ["XLK"],
            "universe_snapshot_id": ["test"],
            "source": ["test"],
        }
    )

    rows = build_swing_feature_rows(
        stock,
        benchmarks,
        memberships,
        contract=contract,
    )
    resolved = rows.loc[rows["barrier_label"].notna()]

    assert not rows.duplicated(["security_id", "session_date_et"]).any()
    assert not resolved.empty
    assert rows["feature_eligible"].any()
    assert rows["cross_section_size"].isna().all()
    assert rows["kaufman_efficiency_ratio"].notna().any()
    assert rows["price_obv_confirmation"].notna().any()
    assert np.allclose(
        resolved["forward_return"],
        resolved["barrier_net_return"],
    )
    expected_gross = (
        resolved["barrier_exit_price"] / resolved["entry_price"] - 1.0
    )
    assert np.allclose(resolved["barrier_gross_return"], expected_gross)
    assert (
        resolved["barrier_label_available_at_utc"]
        .ge(resolved["decision_time_utc"])
        .all()
    )
    assert (
        resolved["barrier_exit_session_date_et"]
        .le(resolved["exit_session_date_et"])
        .all()
    )
    assert set(resolved["barrier_label"].astype(int)).issubset({-1, 0, 1})
