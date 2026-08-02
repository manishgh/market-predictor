"""Causal, population-wide swing feature rows for cross-sectional ranking.

The builder has two explicit stages because their memory and information
boundaries differ:

* :func:`build_swing_feature_rows` may run on bounded security batches. It
  computes only within-security history, exact future outcomes, and shared
  technical relationships.
* :func:`finalize_swing_feature_panel` must receive the complete tradable
  population. It compares securities only with peers from the same decision
  session and then assigns the sector-relative rank label.

Splitting the second stage by security would produce different z-scores and
ranks for the same row depending on batch membership. That is refused rather
than approximated.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from datetime import date
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.canonical.joins import (
    decisions_from_completed_bars,
    join_universe_membership,
)
from market_predictor.canonical.reconciliation import stamp_canonical_decision_ids
from market_predictor.edge_rebuild.catalyst_authority import (
    COVERAGE_FLAG_COLUMNS,
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
    CatalystDecisionAuthority,
    attach_catalyst_decision_features,
)
from market_predictor.edge_rebuild.cross_sectional import (
    CrossSectionSpec,
    add_cross_sectional_features,
    cross_sectional_feature_names,
)
from market_predictor.edge_rebuild.labeling import (
    BarrierSpec,
    apply_cross_sectional_rank,
    apply_triple_barrier,
    forward_return_from_barrier,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.technical_relationships import (
    add_technical_relationship_features,
    relationship_spec_from_contract,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.dataset import build_swing_feature_history
from market_predictor.swing.labels import add_exact_swing_labels
from market_predictor.v3.errors import DataReadinessError

SWING_FEATURE_PANEL_SCHEMA: Final = "edge_rebuild.swing_feature_panel.v9"
SWING_FEATURE_PROFILE: Final = "technical_market"
SWING_CATALYST_FEATURE_PROFILE: Final = "catalyst_full"
SWING_ABLATION_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
    SWING_CATALYST_FEATURE_PROFILE,
)

MOMENTUM_FEATURES: Final = (
    "return_5d",
    "return_20d",
    "return_60d",
    "rel_return_5d_vs_spy",
    "rel_return_20d_vs_spy",
    "rel_return_5d_vs_sector",
    "rel_return_20d_vs_sector",
    "residual_return_20d_vs_spy",
    "residual_return_20d_vs_sector",
    "residual_return_60d_vs_spy",
    "residual_return_60d_vs_sector",
)
TREND_FEATURES: Final = (
    "dist_ema_20",
    "dist_ema_50",
    "dist_sma_200",
    "sma_200_slope_20d",
    "macd_signal_diff_pct",
    "rsi_trend_alignment",
    "kaufman_efficiency_ratio",
)
PULLBACK_FEATURES: Final = (
    "return_1d",
    "dist_ema_10",
    "prior_dist_ema_10",
    "gap_return",
    "intraday_return",
    "close_location",
    "rsi_range_position",
    "rsi_bearish_divergence_strength",
    "rsi_bearish_divergence_confirmation_age_bars",
    "rsi_bullish_divergence_strength",
    "rsi_bullish_divergence_confirmation_age_bars",
)
VOLUME_FEATURES: Final = (
    "volume_z20",
    "volume_ratio_20",
    "dollar_volume_log",
    "obv_directional_change_ratio",
    "price_obv_confirmation",
)
CATALYST_AUDIT_FEATURES: Final = (
    "event_count_1d",
    "event_count_3d",
    "sentiment_mean_1d",
    "sentiment_mean_3d",
    "sentiment_coverage_1d",
    "sentiment_coverage_3d",
    "event_relevance_mean_1d",
    "event_relevance_mean_3d",
    "low_relevance_event_fraction_1d",
    "low_relevance_event_fraction_3d",
    *(
        f"source_count_{family}_{window}"
        for family in TRACKED_SOURCE_FAMILIES
        for window in ("1d", "3d")
    ),
)
CATALYST_RANKING_FEATURES: Final = tuple(
    (
        "event_count_1d",
        "event_count_3d",
        "sentiment_mean_1d",
        "sentiment_mean_3d",
        "sentiment_coverage_1d",
        "sentiment_coverage_3d",
        "event_relevance_mean_1d",
        "event_relevance_mean_3d",
        *(
            f"source_count_{family}_{window}"
            for family in REQUIRED_MODEL_SOURCE_FAMILIES
            for window in ("1d", "3d")
        ),
    )
)
TECHNICAL_RANKING_FEATURES: Final = tuple(
    dict.fromkeys(
        (
            *MOMENTUM_FEATURES,
            *TREND_FEATURES,
            *PULLBACK_FEATURES,
            *VOLUME_FEATURES,
        )
    )
)

_BARRIER_RENAMES: Final = {
    "exit_session": "barrier_exit_session_date_et",
    "exit_price": "barrier_exit_price",
    "holding_sessions": "barrier_holding_sessions",
    "target_price": "barrier_target_price",
    "stop_price": "barrier_stop_price",
}
MANAGED_PATH_SESSION_ORDINAL_COLUMNS: Final = tuple(
    f"managed_path_session_ordinal_d{offset}" for offset in range(1, 11)
)
MANAGED_PATH_NET_RETURN_COLUMNS: Final = tuple(
    f"managed_cumulative_net_return_d{offset}" for offset in range(1, 11)
)
MANAGED_BENCHMARK_RETURN_COLUMNS: Final = (
    "approx_managed_exit_session_close_spy_return",
    "approx_managed_exit_session_close_qqq_return",
    "approx_managed_exit_session_close_sector_return",
)
MANAGED_EXCESS_RETURN_COLUMNS: Final = (
    "approx_managed_exit_session_close_excess_vs_spy",
    "approx_managed_exit_session_close_excess_vs_qqq",
    "approx_managed_exit_session_close_excess_vs_sector",
)
MANAGED_PATH_COST_POLICY: Final = "full_round_trip_cost_applied_from_first_daily_mark"


def swing_dataset_config(
    contract: StrategyContract,
    *,
    feature_profile: str = SWING_FEATURE_PROFILE,
    required_ticker_sources: tuple[str, ...] = (),
    required_global_sources: tuple[str, ...] = (),
) -> SwingDatasetConfig:
    """Bind the canonical feature history to the frozen swing contract."""

    if feature_profile not in contract.features.profiles:
        raise ValueError(f"undeclared swing feature profile: {feature_profile}")
    return SwingDatasetConfig(
        feature_profile=feature_profile,
        horizon_sessions=contract.swing.horizon_sessions,
        round_trip_cost_bps=contract.swing.round_trip_cost_bps,
        min_daily_bars=contract.swing.minimum_warmup_sessions,
        required_ticker_sources=required_ticker_sources,
        required_global_sources=required_global_sources,
        minimum_cross_section=contract.labels.minimum_cross_section_for_ranking,
    )


def attach_setup_components(
    features: pd.DataFrame,
    benchmark_features: pd.DataFrame,
) -> pd.DataFrame:
    """Attach residual momentum, prior-bar pullback state, and dollar volume."""

    horizon_returns = benchmark_features.loc[
        :, ["ticker", "session_date_et", "return_60d"]
    ]
    spy_rows = horizon_returns.loc[
        horizon_returns["ticker"].astype(str).str.upper().eq("SPY")
    ]
    if spy_rows.empty:
        raise DataReadinessError(
            "swing residual features require SPY benchmark features"
        )
    spy = spy_rows.rename(
        columns={"return_60d": "spy_return_60d"}
    ).drop(columns="ticker")
    sector = horizon_returns.rename(
        columns={
            "ticker": "primary_benchmark",
            "return_60d": "sector_return_60d",
        }
    )
    data = features.merge(
        spy,
        on="session_date_et",
        how="left",
        validate="many_to_one",
    )
    data = data.merge(
        sector,
        on=["primary_benchmark", "session_date_et"],
        how="left",
        validate="many_to_one",
    )
    for window in (20, 60):
        stock = pd.to_numeric(data[f"return_{window}d"], errors="coerce")
        data[f"residual_return_{window}d_vs_spy"] = (
            stock - data[f"spy_return_{window}d"]
        )
        data[f"residual_return_{window}d_vs_sector"] = (
            stock - data[f"sector_return_{window}d"]
        )

    data = data.sort_values(
        ["security_id", "session_date_et"],
        kind="stable",
    )
    grouped = data.groupby("security_id", sort=False)
    data["prior_dist_ema_10"] = grouped["dist_ema_10"].shift(1)
    data["prior_dist_sma_200"] = grouped["dist_sma_200"].shift(1)
    data["dollar_volume"] = data["close"] * data["volume"]
    return data


def build_swing_feature_rows(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
    config: SwingDatasetConfig | None = None,
    sparse_missing_sessions_by_ticker: Mapping[str, Sequence[date]] | None = None,
    global_events: pd.DataFrame | None = None,
    global_source_collections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal rows for one or more complete security histories.

    This stage is security-batch safe. It does not emit population-relative
    values or rank labels.
    """

    if global_events is not None or global_source_collections is not None:
        raise DataReadinessError(
            "edge-rebuild swing rows do not accept unverified global inputs; "
            "a separate hash-bound global authority is required"
        )
    effective = config or swing_dataset_config(contract)
    _validate_config(effective, contract)
    if effective.feature_profile != SWING_FEATURE_PROFILE:
        raise DataReadinessError(
            "edge-rebuild catalyst rows must be attached from a verified "
            "CatalystDecisionAuthority"
        )
    traded = stock_bars.loc[
        pd.to_numeric(stock_bars["volume"], errors="coerce").gt(0)
    ].copy()
    decisions = decisions_from_completed_bars(traded, mode="swing-nightly")
    decisions = join_universe_membership(decisions, memberships)
    decisions = stamp_canonical_decision_ids(decisions)
    decisions["feature_profile"] = effective.feature_profile
    features, benchmark_features = build_swing_feature_history(
        decisions,
        benchmark_bars,
        global_events=global_events,
        global_source_collections=global_source_collections,
        config=effective,
        defer_cross_sectional=True,
    )
    labelled = add_exact_swing_labels(
        features,
        benchmark_features,
        effective,
        inplace=True,
    )
    rows = attach_setup_components(labelled, benchmark_features)
    rows = _apply_sector_benchmark_eligibility(
        rows,
        horizon_sessions=effective.horizon_sessions,
    )
    rows = add_technical_relationship_features(
        rows,
        spec=relationship_spec_from_contract(
            contract,
            group_columns=("security_id",),
            time_column="session_date_et",
        ),
    )
    rows = _add_barrier_outcomes(
        rows,
        benchmark_bars=benchmark_features,
        contract=contract,
    )
    rows = _mask_sector_benchmark_ineligible_outcomes(rows)
    rows = apply_sparse_session_gap_abstentions(
        rows,
        benchmark_bars=benchmark_features,
        sparse_missing_sessions_by_ticker=(
            sparse_missing_sessions_by_ticker or {}
        ),
        contract=contract,
    )
    rows["swing_feature_panel_schema"] = SWING_FEATURE_PANEL_SCHEMA
    rows["strategy_contract_sha256"] = contract.sha256()
    return rows.sort_values(
        ["session_date_et", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_swing_ablation_rows(
    technical_rows: pd.DataFrame,
    catalyst_authority: CatalystDecisionAuthority,
) -> dict[str, pd.DataFrame]:
    """Create matched technical and catalyst populations from one row authority."""

    attached = attach_catalyst_decision_features(
        technical_rows,
        catalyst_authority,
    )
    required_flags = {
        f"source_coverage_known_{family}_{window}"
        for family in REQUIRED_MODEL_SOURCE_FAMILIES
        for window in ("1d", "3d")
    }
    required = {
        "feature_eligible",
        "label_eligible",
        "catalyst_source_complete_1d",
        "catalyst_source_complete_3d",
        *COVERAGE_FLAG_COLUMNS,
        *CATALYST_AUDIT_FEATURES,
        *CATALYST_RANKING_FEATURES,
        *required_flags,
    }
    missing = sorted(required.difference(attached.columns))
    if missing:
        raise DataReadinessError(
            f"catalyst ablation rows are missing authority columns: {missing}"
        )

    required_complete = pd.Series(True, index=attached.index)
    for window in ("1d", "3d"):
        declared = (
            attached[f"catalyst_source_complete_{window}"]
            .fillna(False)
            .astype(bool)
        )
        observed = pd.concat(
            [
                attached[f"source_coverage_known_{family}_{window}"]
                .fillna(False)
                .astype(bool)
                for family in REQUIRED_MODEL_SOURCE_FAMILIES
            ],
            axis=1,
        ).all(axis=1)
        if bool(declared.ne(observed).any()):
            raise DataReadinessError(
                f"catalyst required-source completeness conflicts in {window}"
            )
        required_complete &= observed
    attached = _scope_catalyst_aggregates_to_required_sources(attached)
    comparable = (
        attached["feature_eligible"].fillna(False).astype(bool)
        & required_complete
    )
    base = attached.copy()
    base["pre_catalyst_feature_eligible"] = (
        base["feature_eligible"].fillna(False).astype(bool)
    )
    base["catalyst_required_source_complete"] = required_complete
    base["ablation_population_eligible"] = comparable
    base["feature_eligible"] = comparable
    base["label_eligible"] = (
        base["label_eligible"].fillna(False).astype(bool) & comparable
    )

    technical = base.copy()
    technical["feature_profile"] = SWING_FEATURE_PROFILE
    catalyst = base.copy()
    catalyst["feature_profile"] = SWING_CATALYST_FEATURE_PROFILE
    return {
        SWING_FEATURE_PROFILE: technical,
        SWING_CATALYST_FEATURE_PROFILE: catalyst,
    }


def _scope_catalyst_aggregates_to_required_sources(
    rows: pd.DataFrame,
) -> pd.DataFrame:
    """Expose aggregate model fields only when counts prove Alpaca-only evidence."""

    if REQUIRED_MODEL_SOURCE_FAMILIES != ("alpaca",):
        raise DataReadinessError(
            "swing aggregate scoping currently requires Alpaca as the sole "
            "historical model source"
        )
    output = rows.copy()
    aggregate_bases = (
        "sentiment_mean",
        "sentiment_coverage",
        "event_relevance_mean",
        "low_relevance_event_fraction",
    )
    for window in ("1d", "3d"):
        event_column = f"event_count_{window}"
        alpaca_column = f"source_count_alpaca_{window}"
        event_count = pd.to_numeric(output[event_column], errors="coerce")
        alpaca_count = pd.to_numeric(output[alpaca_column], errors="coerce")
        optional_positive = pd.Series(False, index=output.index)
        for family in TRACKED_SOURCE_FAMILIES:
            if family in REQUIRED_MODEL_SOURCE_FAMILIES:
                continue
            optional_count = pd.to_numeric(
                output[f"source_count_{family}_{window}"],
                errors="coerce",
            )
            optional_positive |= optional_count.gt(0).fillna(False)
        if bool((event_count < alpaca_count).fillna(False).any()):
            raise DataReadinessError(
                f"catalyst event count is below Alpaca evidence in {window}"
            )
        alpaca_only = (
            event_count.notna()
            & alpaca_count.notna()
            & np.isclose(event_count, alpaca_count)
            & ~optional_positive
        )
        output[event_column] = alpaca_count
        for base in aggregate_bases:
            column = f"{base}_{window}"
            numeric = pd.to_numeric(output[column], errors="coerce")
            output[column] = numeric.where(alpaca_only)
        output[f"alpaca_aggregate_complete_{window}"] = alpaca_only
    return output


def apply_sparse_session_gap_abstentions(
    rows: pd.DataFrame,
    *,
    benchmark_bars: pd.DataFrame,
    sparse_missing_sessions_by_ticker: Mapping[str, Sequence[date]],
    contract: StrategyContract,
) -> pd.DataFrame:
    """Invalidate feature and label windows that cross retained sparse gaps."""

    data = rows.copy()
    data["sparse_gap_feature_eligible"] = True
    data["sparse_gap_label_eligible"] = True
    data["sparse_gap_abstention_reason"] = ""
    if not sparse_missing_sessions_by_ticker:
        return data

    required = {
        "ticker",
        "session_date_et",
        "feature_eligible",
        "label_eligible",
    }
    missing_columns = sorted(required.difference(data.columns))
    if missing_columns:
        raise DataReadinessError(
            f"sparse-gap abstention rows are missing columns: {missing_columns}"
        )
    market_ticker = contract.labels.benchmark_market.upper()
    market_session_values = (
        benchmark_bars.loc[
            benchmark_bars["ticker"].astype(str).str.upper().eq(market_ticker),
            "session_date_et",
        ]
        .drop_duplicates()
        .sort_values()
    )
    market_sessions = pd.to_datetime(
        market_session_values,
        errors="coerce",
    )
    if market_sessions.empty or bool(market_sessions.isna().any()):
        raise DataReadinessError(
            f"sparse-gap abstention requires {market_ticker} sessions"
        )
    ordinal = {
        value: index
        for index, value in enumerate(market_sessions.dt.date.tolist())
    }
    row_sessions = pd.to_datetime(data["session_date_et"], errors="coerce")
    row_ordinal = row_sessions.dt.date.map(ordinal)
    if bool(row_sessions.isna().any() or row_ordinal.isna().any()):
        raise DataReadinessError(
            "sparse-gap abstention found decisions outside benchmark sessions"
        )

    feature_cross = pd.Series(False, index=data.index)
    label_cross = pd.Series(False, index=data.index)
    normalized = {
        str(ticker).strip().upper(): tuple(sorted(set(sessions)))
        for ticker, sessions in sparse_missing_sessions_by_ticker.items()
    }
    for ticker, sessions in normalized.items():
        selected = data["ticker"].astype(str).str.upper().eq(ticker)
        if not bool(selected.any()):
            continue
        unknown = sorted(set(sessions).difference(ordinal))
        if unknown:
            raise DataReadinessError(
                f"sparse-gap sessions are absent from {market_ticker}: "
                f"{ticker} {unknown[:5]}"
            )
        positions = row_ordinal.loc[selected].astype(int)
        for missing_session in sessions:
            gap_ordinal = ordinal[missing_session]
            distance = positions - gap_ordinal
            feature_cross.loc[selected] |= (
                distance.ge(0)
                & distance.lt(contract.swing.minimum_warmup_sessions)
            ).to_numpy()
            label_cross.loc[selected] |= (
                distance.lt(0)
                & distance.ge(-contract.swing.horizon_sessions)
            ).to_numpy()

    data["sparse_gap_feature_eligible"] = ~feature_cross
    data["sparse_gap_label_eligible"] = ~label_cross
    data["sparse_gap_abstention_reason"] = np.select(
        [feature_cross & label_cross, feature_cross, label_cross],
        [
            "feature_and_label_window_crosses_missing_session",
            "feature_warmup_crosses_missing_session",
            "label_window_crosses_missing_session",
        ],
        default="",
    )
    data["feature_eligible"] = (
        data["feature_eligible"].fillna(False).astype(bool) & ~feature_cross
    )
    data["label_eligible"] = (
        data["label_eligible"].fillna(False).astype(bool)
        & ~feature_cross
        & ~label_cross
    )
    label_columns = [
        column
        for column in data.columns
        if column.startswith("future_")
        or column.startswith("target_net_positive_")
        or column.startswith("barrier_")
        or column in {"forward_return", "target_excess_rank"}
    ]
    if label_columns:
        data.loc[label_cross, label_columns] = pd.NA
    return data


def finalize_swing_feature_panel(
    rows: pd.DataFrame,
    *,
    contract: StrategyContract,
    expected_security_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add same-session transforms and the sector-relative managed-return label."""

    required = {
        "security_id",
        "session_date_et",
        "sector",
        "feature_eligible",
        "daily_bar_count",
        "forward_return",
        *TECHNICAL_RANKING_FEATURES,
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise DataReadinessError(
            f"swing feature rows are missing required columns: {missing}"
        )
    if rows.empty:
        raise DataReadinessError("swing feature panel cannot be empty")
    identity = ["security_id", "session_date_et"]
    if bool(rows.duplicated(identity).any()):
        raise DataReadinessError(
            "swing feature panel requires one row per security and session"
        )
    if expected_security_ids is not None:
        expected = {str(value) for value in expected_security_ids}
        observed = set(rows["security_id"].astype(str))
        missing_identities = sorted(expected.difference(observed))
        unexpected_identities = sorted(observed.difference(expected))
        if missing_identities:
            raise DataReadinessError(
                "population-wide swing scaling is missing expected securities: "
                f"{missing_identities[:10]}"
            )
        if unexpected_identities:
            raise DataReadinessError(
                "population-wide swing scaling contains unexpected securities: "
                f"{unexpected_identities[:10]}"
            )

    data = rows.copy()
    profiles = set(data["feature_profile"].astype(str))
    if len(profiles) != 1:
        raise DataReadinessError(
            f"swing feature panel mixes feature profiles: {sorted(profiles)}"
        )
    ranking_inputs = list(TECHNICAL_RANKING_FEATURES)
    if profiles == {"catalyst_full"}:
        missing_catalyst = sorted(
            set(CATALYST_RANKING_FEATURES).difference(data.columns)
        )
        if missing_catalyst:
            raise DataReadinessError(
                "catalyst swing rows are missing ranking features: "
                f"{missing_catalyst}"
            )
        ranking_inputs.extend(CATALYST_RANKING_FEATURES)
    spec = _cross_section_spec(contract)
    transformed_names = cross_sectional_feature_names(
        ranking_inputs,
        spec=spec,
    )
    eligible = (
        data["feature_eligible"].fillna(False).astype(bool)
        & data["daily_bar_count"].ge(contract.swing.minimum_warmup_sessions)
    )
    transformed = add_cross_sectional_features(
        data.loc[eligible],
        ranking_inputs,
        spec=spec,
        timestamp_column="session_date_et",
        sector_column="sector",
    )
    transformed_block = pd.DataFrame(
        np.nan,
        index=data.index,
        columns=transformed_names,
        dtype="float32",
    )
    if not transformed.empty:
        transformed_block.loc[eligible, :] = transformed.loc[
            :, transformed_names
        ].to_numpy(dtype=np.float32)
    data = pd.concat([data, transformed_block], axis=1)

    sector_peer_count = pd.Series(0, index=data.index, dtype="int32")
    if bool(eligible.any()):
        eligible_peers = data.loc[eligible, ["session_date_et", "sector"]]
        sector_peer_count.loc[eligible] = (
            eligible_peers.groupby(
                ["session_date_et", "sector"],
                sort=False,
            )["sector"]
            .transform("size")
            .to_numpy(dtype="int32")
        )
    sector_rank_eligible = sector_peer_count.ge(
        contract.labels.minimum_cross_section_for_ranking
    )
    data["sector_peer_count"] = sector_peer_count
    data["sector_rank_eligible"] = sector_rank_eligible
    data["sector_rank_target_met"] = sector_peer_count.ge(
        contract.labels.swing_target_cross_section_for_ranking
    )

    rank_eligible = (
        eligible
        & sector_rank_eligible
        & data["barrier_label"].notna()
        & data["forward_return"].notna()
    )
    ranked = apply_cross_sectional_rank(
        data.loc[
            rank_eligible,
            ["session_date_et", "sector", "forward_return"],
        ].rename(columns={"session_date_et": "session"}),
        top_quantile=contract.labels.rank_top_quantile,
        bottom_quantile=contract.labels.rank_bottom_quantile,
        within_sector=contract.labels.rank_within_sector,
        minimum_cross_section=contract.labels.minimum_cross_section_for_ranking,
    )
    rank_label = pd.Series(pd.NA, index=data.index, dtype="Int64")
    rank_label.loc[rank_eligible] = ranked["rank_label"].to_numpy()
    rank_percentile = pd.Series(np.nan, index=data.index, dtype="float32")
    rank_percentile.loc[rank_eligible] = ranked["rank_percentile"].to_numpy(
        dtype=np.float32
    )
    ranking_group_size = pd.Series(pd.NA, index=data.index, dtype="Int32")
    ranking_group_size.loc[rank_eligible] = ranked[
        "ranking_group_size"
    ].to_numpy(dtype="int32")
    ranking_reliability_weight = pd.Series(
        np.nan,
        index=data.index,
        dtype="float32",
    )
    ranking_reliability_weight.loc[sector_rank_eligible] = np.minimum(
        sector_peer_count.loc[sector_rank_eligible].to_numpy(dtype="float32")
        / float(contract.labels.swing_target_cross_section_for_ranking),
        1.0,
    )
    data["rank_label"] = rank_label
    data["rank_percentile"] = rank_percentile
    data["ranking_group_size"] = ranking_group_size
    data["ranking_reliability_weight"] = ranking_reliability_weight
    data["cross_section_eligible"] = sector_rank_eligible
    return data.sort_values(
        ["session_date_et", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def build_swing_feature_panel(
    stock_bars: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
    config: SwingDatasetConfig | None = None,
    global_events: pd.DataFrame | None = None,
    global_source_collections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build and finalize a complete in-memory swing population."""

    rows = build_swing_feature_rows(
        stock_bars,
        benchmark_bars,
        memberships,
        contract=contract,
        config=config,
        global_events=global_events,
        global_source_collections=global_source_collections,
    )
    return finalize_swing_feature_panel(
        rows,
        contract=contract,
        expected_security_ids=memberships["security_id"].astype(str).unique(),
    )


def swing_model_feature_columns(
    *,
    contract: StrategyContract,
    catalyst: bool,
) -> tuple[str, ...]:
    """Return the normalized estimator schema; raw levels are not included."""

    inputs = [
        *TECHNICAL_RANKING_FEATURES,
        *(CATALYST_RANKING_FEATURES if catalyst else ()),
    ]
    columns = tuple(
        cross_sectional_feature_names(
            inputs,
            spec=_cross_section_spec(contract),
        )
    )
    if contract.features.raw_news_counts_prohibited:
        raw_counts = {
            name
            for name in CATALYST_RANKING_FEATURES
            if "count_" in name or name.startswith("event_count_")
        }
        leaked = sorted(raw_counts.intersection(columns))
        if leaked:
            raise DataReadinessError(
                f"raw news counts entered the estimator schema: {leaked}"
            )
    return columns


def _add_barrier_outcomes(
    rows: pd.DataFrame,
    *,
    benchmark_bars: pd.DataFrame,
    contract: StrategyContract,
) -> pd.DataFrame:
    spec = BarrierSpec(
        target_atr_multiple=contract.swing.target_atr_multiple,
        stop_atr_multiple=contract.swing.stop_atr_multiple,
        horizon_sessions=contract.swing.horizon_sessions,
        same_bar_resolution=contract.swing.same_bar_barrier_resolution,
    )
    parts: list[pd.DataFrame] = []
    for security_id, security_rows in rows.groupby(
        "security_id",
        sort=False,
    ):
        ordered = security_rows.sort_values(
            "session_date_et", kind="stable"
        ).reset_index(drop=True)
        bars = ordered.loc[
            :, ["session_date_et", "open", "high", "low", "close"]
        ].rename(columns={"session_date_et": "session"})
        entries = pd.DataFrame(
            {
                "session": ordered["session_date_et"],
                "atr": (
                    pd.to_numeric(ordered["atr_pct_14"], errors="coerce")
                    * pd.to_numeric(ordered["close"], errors="coerce")
                ),
            }
        )
        outcomes = apply_triple_barrier(bars, entries, spec=spec)
        availability = ordered.set_index("session_date_et")[
            "available_at_utc"
        ]
        outcomes["barrier_label_available_at_utc"] = outcomes[
            "exit_session"
        ].map(availability)
        entry = pd.to_numeric(ordered["open"].shift(-1), errors="coerce")
        holding = pd.to_numeric(outcomes["holding_sessions"], errors="coerce")
        exit_price = pd.to_numeric(outcomes["exit_price"], errors="coerce")
        cost = contract.swing.round_trip_cost_bps / 10_000.0
        for offset, (session_column, return_column) in enumerate(
            zip(
                MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
                MANAGED_PATH_NET_RETURN_COLUMNS,
                strict=True,
            ),
            start=1,
        ):
            path_session = pd.to_datetime(
                ordered["session_date_et"].shift(-offset), errors="coerce"
            )
            path_close = pd.to_numeric(
                ordered["close"].shift(-offset), errors="coerce"
            )
            mark = path_close.where(holding.gt(offset), exit_price)
            valid = (
                holding.notna()
                & entry.gt(0)
                & mark.notna()
                & path_session.notna()
            )
            ordinals = pd.Series(pd.NA, index=ordered.index, dtype="Int32")
            # Avoid pandas interpreting mapped ordinals as datetime64 when the
            # source happens to use second-resolution timestamps.
            ordinals.loc[valid] = _session_ordinal_values(
                path_session.loc[valid]
            )
            cumulative_net = pd.Series(np.nan, index=ordered.index, dtype="float64")
            cumulative_net.loc[valid] = (
                mark.loc[valid] / entry.loc[valid] - 1.0 - cost
            )
            outcomes[session_column] = ordinals.to_numpy()
            outcomes[return_column] = cumulative_net.to_numpy()
        outcomes["security_id"] = str(security_id)
        parts.append(outcomes)
    barriers = pd.concat(parts, ignore_index=True).rename(
        columns={"session": "session_date_et", **_BARRIER_RENAMES}
    )
    data = rows.merge(
        barriers,
        on=["security_id", "session_date_et"],
        how="left",
        validate="one_to_one",
    )
    barrier_gross = forward_return_from_barrier(
        pd.DataFrame({"exit_price": data["barrier_exit_price"]}),
        data["entry_price"],
    )
    cost = contract.swing.round_trip_cost_bps / 10_000.0
    data["barrier_gross_return"] = barrier_gross
    data["barrier_cost"] = cost
    data["barrier_net_return"] = barrier_gross - cost
    data = _attach_managed_benchmark_returns(data, benchmark_bars)
    # The rank target uses the managed position outcome. Since the cost is
    # constant across the same decision cross-section, gross and net ordering
    # are identical; net is retained so the label reflects tradable economics.
    data["forward_return"] = data["barrier_net_return"]
    return data


def _session_ordinal_values(values: pd.Series) -> np.ndarray:
    """Return date ordinals without pandas datetime-unit inference."""

    return np.fromiter(
        (pd.Timestamp(value).date().toordinal() for value in values),
        dtype=np.int32,
        count=len(values),
    )


def _attach_managed_benchmark_returns(
    rows: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
) -> pd.DataFrame:
    required = {"ticker", "session_date_et", "open", "close"}
    missing = sorted(required.difference(benchmark_bars.columns))
    if missing:
        raise DataReadinessError(
            f"managed benchmark paths are missing columns: {missing}"
        )
    benchmarks = benchmark_bars.loc[:, sorted(required)].copy()
    benchmarks["ticker"] = benchmarks["ticker"].astype(str).str.upper().str.strip()
    if benchmarks.duplicated(["ticker", "session_date_et"]).any():
        raise DataReadinessError("managed benchmark paths contain duplicate sessions")
    lookup = benchmarks.set_index(["ticker", "session_date_et"])
    data = rows.copy()
    for name, ticker in (
        ("spy", pd.Series("SPY", index=data.index)),
        ("qqq", pd.Series("QQQ", index=data.index)),
        ("sector", data["primary_benchmark"]),
    ):
        benchmark_return = _managed_benchmark_return(
            data,
            lookup,
            ticker,
        )
        data[f"approx_managed_exit_session_close_{name}_return"] = benchmark_return
        data[f"approx_managed_exit_session_close_excess_vs_{name}"] = (
            data["barrier_net_return"] - benchmark_return
        )
    managed_required = [
        *MANAGED_BENCHMARK_RETURN_COLUMNS,
        *MANAGED_EXCESS_RETURN_COLUMNS,
        *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
        *MANAGED_PATH_NET_RETURN_COLUMNS,
    ]
    complete = data[managed_required].notna().all(axis=1)
    data["managed_path_eligible"] = complete
    data["label_eligible"] = (
        data["label_eligible"].fillna(False).astype(bool) & complete
    )
    return data


def _managed_benchmark_return(
    decisions: pd.DataFrame,
    lookup: pd.DataFrame,
    benchmark_tickers: pd.Series,
) -> pd.Series:
    tickers = benchmark_tickers.astype(str).str.upper().str.strip()
    entry_index = pd.MultiIndex.from_arrays(
        [tickers, decisions["entry_session_date_et"]],
        names=lookup.index.names,
    )
    exit_index = pd.MultiIndex.from_arrays(
        [tickers, decisions["barrier_exit_session_date_et"]],
        names=lookup.index.names,
    )
    entry_open = pd.to_numeric(
        lookup["open"].reindex(entry_index), errors="coerce"
    ).to_numpy(dtype="float64")
    exit_close = pd.to_numeric(
        lookup["close"].reindex(exit_index), errors="coerce"
    ).to_numpy(dtype="float64")
    values = np.divide(
        exit_close,
        entry_open,
        out=np.full(len(decisions), np.nan, dtype="float64"),
        where=np.isfinite(entry_open) & np.isfinite(exit_close) & (entry_open > 0),
    )
    return pd.Series(values - 1.0, index=decisions.index, dtype="float64")


def _apply_sector_benchmark_eligibility(
    rows: pd.DataFrame,
    *,
    horizon_sessions: int,
) -> pd.DataFrame:
    required = {
        "feature_eligible",
        "label_eligible",
        "sector_available_at_utc",
        "sector_return_5d",
        "sector_return_20d",
        "sector_return_60d",
        f"future_sector_return_{horizon_sessions}d",
    }
    missing = sorted(required.difference(rows.columns))
    if missing:
        raise DataReadinessError(
            f"sector benchmark eligibility inputs are missing: {missing}"
        )
    data = rows.copy()
    feature_ready = data[
        [
            "sector_available_at_utc",
            "sector_return_5d",
            "sector_return_20d",
            "sector_return_60d",
        ]
    ].notna().all(axis=1)
    label_ready = pd.to_numeric(
        data[f"future_sector_return_{horizon_sessions}d"],
        errors="coerce",
    ).notna()
    data["sector_benchmark_feature_eligible"] = feature_ready
    data["sector_benchmark_label_eligible"] = feature_ready & label_ready
    data["sector_benchmark_abstention_reason"] = np.select(
        [~feature_ready, feature_ready & ~label_ready],
        [
            "sector_benchmark_feature_unavailable",
            "sector_benchmark_label_window_unavailable",
        ],
        default="",
    )
    data["feature_eligible"] = (
        data["feature_eligible"].fillna(False).astype(bool) & feature_ready
    )
    data["label_eligible"] = (
        data["label_eligible"].fillna(False).astype(bool)
        & data["sector_benchmark_label_eligible"]
    )
    return data


def _mask_sector_benchmark_ineligible_outcomes(rows: pd.DataFrame) -> pd.DataFrame:
    if "sector_benchmark_label_eligible" not in rows.columns:
        raise DataReadinessError(
            "managed swing outcomes require sector benchmark label eligibility"
        )
    data = rows.copy()
    abstain = ~data["sector_benchmark_label_eligible"].fillna(False).astype(bool)
    outcome_columns = [
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
        *MANAGED_BENCHMARK_RETURN_COLUMNS,
        *MANAGED_EXCESS_RETURN_COLUMNS,
        *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
        *MANAGED_PATH_NET_RETURN_COLUMNS,
    ]
    missing = sorted(set(outcome_columns).difference(data.columns))
    if missing:
        raise DataReadinessError(
            f"managed swing outcome columns are missing: {missing}"
        )
    data.loc[abstain, outcome_columns] = pd.NA
    if "managed_path_eligible" not in data.columns:
        raise DataReadinessError("managed swing outcome column is missing: managed_path_eligible")
    data.loc[abstain, "managed_path_eligible"] = False
    return data


def _cross_section_spec(contract: StrategyContract) -> CrossSectionSpec:
    features = contract.features
    return CrossSectionSpec(
        minimum_cross_section=contract.labels.minimum_cross_section_for_ranking,
        winsorize_quantile=features.cross_sectional_winsorize_quantile,
        emit_zscore=features.cross_sectional_emit_zscore,
        emit_rank=features.cross_sectional_emit_rank,
        emit_sector_relative=features.cross_sectional_emit_sector_relative,
    )


def _validate_config(
    config: SwingDatasetConfig,
    contract: StrategyContract,
) -> None:
    expected = (
        config.horizon_sessions == contract.swing.horizon_sessions
        and config.round_trip_cost_bps == contract.swing.round_trip_cost_bps
        and config.min_daily_bars == contract.swing.minimum_warmup_sessions
    )
    if not expected:
        raise DataReadinessError(
            "swing dataset config does not match the frozen strategy contract"
        )
