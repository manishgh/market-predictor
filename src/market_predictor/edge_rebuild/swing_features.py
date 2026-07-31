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

from collections.abc import Sequence
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.canonical.joins import (
    decisions_from_completed_bars,
    join_universe_membership,
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

SWING_FEATURE_PANEL_SCHEMA: Final = "edge_rebuild.swing_feature_panel.v2"
SWING_FEATURE_PROFILE: Final = "technical_market"

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
CATALYST_RANKING_FEATURES: Final = (
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
    "source_count_alpaca_3d",
    "source_count_reddit_3d",
    "source_count_seeking_alpha_3d",
    "source_count_sec_3d",
    "source_count_finviz_3d",
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
    global_events: pd.DataFrame | None = None,
    global_source_collections: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Build causal rows for one or more complete security histories.

    This stage is security-batch safe. It does not emit population-relative
    values or rank labels.
    """

    effective = config or swing_dataset_config(contract)
    _validate_config(effective, contract)
    traded = stock_bars.loc[
        pd.to_numeric(stock_bars["volume"], errors="coerce").gt(0)
    ].copy()
    decisions = decisions_from_completed_bars(traded, mode="swing-nightly")
    decisions = join_universe_membership(decisions, memberships)
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
    rows = add_technical_relationship_features(
        rows,
        spec=relationship_spec_from_contract(
            contract,
            group_columns=("security_id",),
            time_column="session_date_et",
        ),
    )
    rows = _add_barrier_outcomes(rows, contract=contract)
    rows["swing_feature_panel_schema"] = SWING_FEATURE_PANEL_SCHEMA
    rows["strategy_contract_sha256"] = contract.sha256()
    return rows.sort_values(
        ["session_date_et", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


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
    transformed_block.loc[eligible, :] = transformed.loc[
        :, transformed_names
    ].to_numpy(dtype=np.float32)
    data = pd.concat([data, transformed_block], axis=1)

    rank_eligible = (
        eligible
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
    eligible_count = (
        eligible.astype("int16")
        .groupby(data["session_date_et"], sort=False)
        .transform("sum")
    )
    data["rank_label"] = rank_label
    data["rank_percentile"] = rank_percentile
    data["cross_section_eligible"] = eligible_count.ge(
        contract.labels.minimum_cross_section_for_ranking
    )
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
        ordered = security_rows.sort_values("session_date_et", kind="stable")
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
    # The rank target uses the managed position outcome. Since the cost is
    # constant across the same decision cross-section, gross and net ordering
    # are identical; net is retained so the label reflects tradable economics.
    data["forward_return"] = data["barrier_net_return"]
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
