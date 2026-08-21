from __future__ import annotations

from pathlib import Path
from typing import cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.edge_rebuild import swing_features as swing_feature_module
from market_predictor.edge_rebuild import swing_catalyst_features as swing_catalyst_module
from market_predictor.edge_rebuild.catalyst_authority import (
    COVERAGE_FLAG_COLUMNS,
    CatalystDecisionAuthority,
)
from market_predictor.edge_rebuild.cross_sectional import RANK_SUFFIX
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.edge_rebuild.swing_features import (
    CATALYST_AUDIT_FEATURES,
    CATALYST_RANKING_FEATURES,
    TECHNICAL_RANKING_FEATURES,
    apply_sparse_session_gap_abstentions,
    build_swing_ablation_rows,
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


def test_sector_ranking_uses_target_fifty_and_hard_floor_thirty(
    contract: StrategyContract,
) -> None:
    below_target = finalize_swing_feature_panel(
        _panel(securities=40),
        contract=contract,
    )

    assert below_target["sector_peer_count"].eq(40).all()
    assert below_target["sector_rank_eligible"].all()
    assert not below_target["sector_rank_target_met"].any()
    assert below_target["ranking_group_size"].eq(40).all()
    assert below_target["ranking_reliability_weight"].eq(0.8).all()
    assert below_target["rank_label"].notna().all()

    below_floor = finalize_swing_feature_panel(
        _panel(securities=29),
        contract=contract,
    )
    assert not below_floor["sector_rank_eligible"].any()
    assert below_floor["rank_label"].isna().all()
    assert below_floor["ranking_reliability_weight"].isna().all()


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
    assert "source_count_alpaca_3d_sector_z" in columns
    assert "source_count_sec_3d_sector_z" not in columns
    assert CATALYST_RANKING_FEATURES == (
        "event_count_1d",
        "event_count_3d",
        "sentiment_mean_1d",
        "sentiment_mean_3d",
        "sentiment_coverage_1d",
        "sentiment_coverage_3d",
        "event_relevance_mean_1d",
        "event_relevance_mean_3d",
        "source_count_alpaca_1d",
        "source_count_alpaca_3d",
    )


def test_optional_sources_cannot_change_ablation_model_columns(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _panel(securities=60)
    source["label_eligible"] = True
    source["decision_id"] = [f"decision-{index}" for index in range(len(source))]
    source["decision_time_utc"] = pd.Timestamp("2024-01-02T21:01:00Z")
    values = np.arange(1, len(source) + 1, dtype=float)
    for column in CATALYST_AUDIT_FEATURES:
        source[column] = values
    source["event_count_1d"] = values
    source["event_count_3d"] = values
    source["source_count_alpaca_1d"] = values
    source["source_count_alpaca_3d"] = values
    for family in ("sec", "finviz"):
        for window in ("1d", "3d"):
            source[f"source_count_{family}_{window}"] = 0.0
    for column in COVERAGE_FLAG_COLUMNS:
        source[column] = True
    source["catalyst_source_complete_1d"] = True
    source["catalyst_source_complete_3d"] = True
    monkeypatch.setattr(
        swing_catalyst_module,
        "attach_catalyst_decision_features",
        lambda _rows, _authority: source.copy(),
    )
    panels = build_swing_ablation_rows(
        source,
        cast(CatalystDecisionAuthority, object()),
    )
    technical_source = panels["technical_market"]
    catalyst_source = panels["catalyst_full"]
    technical_before = finalize_swing_feature_panel(
        technical_source,
        contract=contract,
    )
    catalyst_before = finalize_swing_feature_panel(
        catalyst_source,
        contract=contract,
    )

    technical_poisoned = technical_source.copy()
    catalyst_poisoned = catalyst_source.copy()
    audit_only_columns = {
        column
        for column in catalyst_source.columns
        if any(
            column.startswith(f"source_count_{family}_")
            or column.startswith(f"source_coverage_known_{family}_")
            for family in ("sec", "finviz")
        )
    }
    for column in audit_only_columns:
        technical_poisoned[column] = values[::-1] * 1_000_000.0
        catalyst_poisoned[column] = values[::-1] * 1_000_000.0
    technical_after = finalize_swing_feature_panel(
        technical_poisoned,
        contract=contract,
    )
    catalyst_after = finalize_swing_feature_panel(
        catalyst_poisoned,
        contract=contract,
    )

    technical_columns = list(
        swing_model_feature_columns(contract=contract, catalyst=False)
    )
    catalyst_columns = list(
        swing_model_feature_columns(contract=contract, catalyst=True)
    )
    pd.testing.assert_frame_equal(
        technical_before.loc[:, technical_columns],
        technical_after.loc[:, technical_columns],
        check_exact=True,
    )
    pd.testing.assert_frame_equal(
        catalyst_before.loc[:, catalyst_columns],
        catalyst_after.loc[:, catalyst_columns],
        check_exact=True,
    )

    sentiment_changed = catalyst_source.copy()
    sentiment_changed["sentiment_mean_1d"] = values[::-1]
    changed_vector = finalize_swing_feature_panel(
        sentiment_changed,
        contract=contract,
    )
    assert not catalyst_before["sentiment_mean_1d_xs_z"].equals(
        changed_vector["sentiment_mean_1d_xs_z"]
    )


def test_ablation_rows_share_population_and_preserve_optional_missingness(
    contract: StrategyContract,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    decision = pd.Timestamp("2024-01-02T21:01:00Z")
    source = pd.DataFrame(
        {
            "decision_id": ["one", "two"],
            "security_id": ["sec:one", "sec:two"],
            "ticker": ["ONE", "TWO"],
            "decision_time_utc": [decision, decision],
            "feature_profile": ["technical_market", "technical_market"],
            "feature_eligible": [True, False],
            "label_eligible": [True, False],
            "forward_return": [0.02, -0.01],
        }
    )

    def attach(
        rows: pd.DataFrame,
        _authority: CatalystDecisionAuthority,
    ) -> pd.DataFrame:
        attached = rows.copy()
        for column in CATALYST_AUDIT_FEATURES:
            attached[column] = 1.0
        for column in CATALYST_RANKING_FEATURES:
            attached[column] = 1.0
        for family in ("sec", "finviz"):
            for window in ("1d", "3d"):
                attached[f"source_count_{family}_{window}"] = np.nan
        for column in COVERAGE_FLAG_COLUMNS:
            attached[column] = column.startswith("source_coverage_known_alpaca_")
        attached["catalyst_source_complete_1d"] = True
        attached["catalyst_source_complete_3d"] = True
        return attached

    monkeypatch.setattr(
        swing_catalyst_module,
        "attach_catalyst_decision_features",
        attach,
    )
    panels = build_swing_ablation_rows(
        source,
        cast(CatalystDecisionAuthority, object()),
    )

    technical = panels["technical_market"]
    catalyst = panels["catalyst_full"]
    assert technical["decision_id"].tolist() == catalyst["decision_id"].tolist()
    assert technical["forward_return"].tolist() == catalyst["forward_return"].tolist()
    assert technical["feature_eligible"].tolist() == [True, False]
    assert catalyst["feature_eligible"].tolist() == [True, False]
    assert catalyst["source_count_sec_3d"].isna().all()
    assert catalyst["source_count_alpaca_3d"].eq(1.0).all()


def test_sparse_gap_invalidates_feature_and_label_windows(
    contract: StrategyContract,
) -> None:
    calendar = xcals.get_calendar("XNYS")
    sessions = [
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range("2023-01-03", "2025-01-31")
    ]
    gap = sessions[100]
    selected_sessions = [sessions[95], sessions[101], sessions[350]]
    rows = pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "session_date_et": selected_sessions,
            "feature_eligible": [True] * 3,
            "label_eligible": [True] * 3,
            "forward_return": [0.01, 0.02, 0.03],
            "future_net_return_10d": [0.01, 0.02, 0.03],
            "barrier_label": [1, 1, 1],
        }
    )
    benchmarks = pd.DataFrame(
        {
            "ticker": ["SPY"] * len(sessions),
            "session_date_et": sessions,
        }
    )

    output = apply_sparse_session_gap_abstentions(
        rows,
        benchmark_bars=benchmarks,
        sparse_missing_sessions_by_ticker={"AAA": (gap,)},
        contract=contract,
    )

    assert output["sparse_gap_label_eligible"].tolist() == [False, True, True]
    assert output["sparse_gap_feature_eligible"].tolist() == [True, False, True]
    assert output["feature_eligible"].tolist() == [True, False, True]
    assert output["label_eligible"].tolist() == [False, False, True]
    assert pd.isna(output.loc[0, "forward_return"])
    assert pd.isna(output.loc[0, "barrier_label"])
    assert output.loc[2, "forward_return"] == pytest.approx(0.03)


def test_unverified_global_inputs_are_refused(
    contract: StrategyContract,
) -> None:
    with pytest.raises(DataReadinessError, match="verified global"):
        build_swing_feature_rows(
            pd.DataFrame(),
            pd.DataFrame(),
            pd.DataFrame(),
            contract=contract,
            global_events=pd.DataFrame({"event_id": ["unbound"]}),
        )


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


def test_unavailable_sector_benchmark_makes_rows_abstain() -> None:
    rows = pd.DataFrame(
        {
            "feature_eligible": [True, True, True],
            "label_eligible": [True, True, True],
            "sector_available_at_utc": [
                pd.NaT,
                pd.Timestamp("2024-01-02T21:00:00Z"),
                pd.Timestamp("2024-01-02T21:00:00Z"),
            ],
            "sector_return_5d": [np.nan, 0.01, 0.01],
            "sector_return_20d": [np.nan, 0.02, 0.02],
            "sector_return_60d": [np.nan, 0.03, 0.03],
            "future_sector_return_5d": [np.nan, np.nan, 0.04],
        }
    )

    eligible = swing_feature_module._apply_sector_benchmark_eligibility(
        rows,
        horizon_sessions=5,
    )

    assert eligible["feature_eligible"].tolist() == [False, True, True]
    assert eligible["label_eligible"].tolist() == [False, False, True]
    assert eligible["sector_benchmark_abstention_reason"].tolist() == [
        "sector_benchmark_feature_unavailable",
        "sector_benchmark_label_window_unavailable",
        "",
    ]

    for column in (
        "barrier_label",
        "barrier_exit_session_date_et",
        "barrier_exit_price",
        "barrier_holding_sessions",
        "barrier_target_price",
        "barrier_stop_price",
        "barrier_label_available_at_utc",
        "barrier_gross_return",
        "barrier_cost",
        "barrier_net_return",
        "forward_return",
        *swing_feature_module.MANAGED_BENCHMARK_RETURN_COLUMNS,
        *swing_feature_module.MANAGED_EXCESS_RETURN_COLUMNS,
        *swing_feature_module.MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
        *swing_feature_module.MANAGED_PATH_NET_RETURN_COLUMNS,
    ):
        eligible[column] = [1, 1, 1]
    eligible["managed_path_eligible"] = [True, True, True]

    masked = swing_feature_module._mask_sector_benchmark_ineligible_outcomes(
        eligible
    )

    assert masked.loc[:1, "barrier_label"].isna().all()
    assert masked.loc[:1, "forward_return"].isna().all()
    assert masked.loc[2, "barrier_label"] == 1
    assert masked.loc[2, "forward_return"] == 1


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
    sparse_gap = pd.Timestamp(sessions[20]).date()
    stock_sessions = (
        pd.to_datetime(stock["bar_start_utc"], utc=True)
        .dt.tz_convert("America/New_York")
        .dt.date
    )
    stock = stock.loc[stock_sessions.ne(sparse_gap)].copy()
    stock = pd.concat(
        [stock, _canonical_bars("BBB", sessions, drift=0.0008)],
        ignore_index=True,
    )
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
            "ticker": ["AAA", "BBB"],
            "security_id": ["sec:aaa", "sec:bbb"],
            "effective_from_utc": [effective_from, effective_from],
            "effective_to_utc": [pd.NaT, pd.NaT],
            "available_at_utc": [effective_from, effective_from],
            "sector": ["Technology", "Technology"],
            "industry": ["Software", "Hardware"],
            "market_cap_bucket": ["large_cap_sp500"] * 2,
            "liquidity_bucket": ["sp500_constituent"] * 2,
            "primary_benchmark": ["XLK", "XLK"],
            "universe_snapshot_id": ["test", "test"],
            "source": ["test", "test"],
        }
    )

    rows = build_swing_feature_rows(
        stock,
        benchmarks,
        memberships,
        contract=contract,
        sparse_missing_sessions_by_ticker={"AAA": (sparse_gap,)},
    )
    resolved = rows.loc[rows["barrier_label"].notna()]

    assert not rows.duplicated(["security_id", "session_date_et"]).any()
    assert rows["decision_id"].str.fullmatch(r"[0-9a-f]{64}").all()
    assert not rows["decision_id"].duplicated().any()
    assert not resolved.empty
    assert rows["feature_eligible"].any()
    assert rows["cross_section_size"].isna().all()
    assert rows["kaufman_efficiency_ratio"].notna().any()
    assert rows["price_obv_confirmation"].notna().any()
    assert rows["sparse_gap_abstention_reason"].ne("").any()
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
    managed = rows.loc[rows["managed_path_eligible"].fillna(False)]
    assert not managed.empty
    assert managed.groupby("security_id").size().gt(50).all()
    assert np.allclose(
        managed[swing_feature_module.MANAGED_PATH_NET_RETURN_COLUMNS[-1]],
        managed["barrier_net_return"],
    )
    assert np.allclose(
        managed["approx_managed_exit_session_close_excess_vs_spy"],
        managed["barrier_net_return"]
        - managed["approx_managed_exit_session_close_spy_return"],
    )
    ordinals = managed.loc[
        :, list(swing_feature_module.MANAGED_PATH_SESSION_ORDINAL_COLUMNS)
    ].to_numpy(dtype="int64")
    assert (np.diff(ordinals, axis=1) > 0).all()


def test_session_ordinals_accept_second_resolution_timestamps() -> None:
    values = pd.Series(
        np.asarray(
            ["2024-01-02T00:00:00", "2024-01-03T00:00:00"],
            dtype="datetime64[s]",
        )
    )

    ordinals = swing_feature_module._session_ordinal_values(values)

    assert ordinals.dtype == np.dtype("int32")
    assert ordinals.tolist() == [
        pd.Timestamp("2024-01-02").date().toordinal(),
        pd.Timestamp("2024-01-03").date().toordinal(),
    ]
