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
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.cross_sectional import (
    CrossSectionSpec,
    cross_sectional_feature_names,
)
from market_predictor.edge_rebuild.labeling import (
    BarrierSpec,
    apply_triple_barrier,
    forward_return_from_barrier,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_catalyst_features import (
    _scope_catalyst_aggregates_to_required_sources as _scope_catalyst_aggregates_to_required_sources,
)
from market_predictor.edge_rebuild.swing_catalyst_features import (
    build_swing_ablation_rows as build_swing_ablation_rows,
)
from market_predictor.edge_rebuild.swing_filters import (
    MANAGED_BENCHMARK_RETURN_COLUMNS as MANAGED_BENCHMARK_RETURN_COLUMNS,
)
from market_predictor.edge_rebuild.swing_filters import (
    MANAGED_EXCESS_RETURN_COLUMNS as MANAGED_EXCESS_RETURN_COLUMNS,
)
from market_predictor.edge_rebuild.swing_filters import (
    MANAGED_PATH_COST_POLICY as MANAGED_PATH_COST_POLICY,
)
from market_predictor.edge_rebuild.swing_filters import (
    MANAGED_PATH_NET_RETURN_COLUMNS as MANAGED_PATH_NET_RETURN_COLUMNS,
)
from market_predictor.edge_rebuild.swing_filters import (
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS as MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
)
from market_predictor.edge_rebuild.swing_filters import (
    _apply_sector_benchmark_eligibility as _apply_sector_benchmark_eligibility,
)
from market_predictor.edge_rebuild.swing_filters import (
    _mask_sector_benchmark_ineligible_outcomes as _mask_sector_benchmark_ineligible_outcomes,
)
from market_predictor.edge_rebuild.swing_filters import (
    apply_sparse_session_gap_abstentions as apply_sparse_session_gap_abstentions,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.dataset import build_swing_feature_history
from market_predictor.swing.features.catalyst_decision_authority import (
    REQUIRED_MODEL_SOURCE_FAMILIES,
    TRACKED_SOURCE_FAMILIES,
)
from market_predictor.swing.labels import add_exact_swing_labels

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
    "rsi_14",
    # `macd`, `macd_signal` and `macd_hist` were raw dollar differences of two
    # EMAs, so a $500 stock's MACD ran ~10x a $50 stock's for the same
    # percentage move (measured spearman(|macd|, close) = 0.635). Cross-sectional
    # z-scoring cannot repair that -- the rank stays a price proxy. The
    # scale-free form is `macd_signal_diff_pct` in TREND_FEATURES.
    "realized_vol_20d",
)
TREND_FEATURES: Final = (
    "dist_ema_20",
    "dist_ema_50",
    "dist_sma_50",
    "dist_sma_200",
    "sma_200_slope_20d",
    "macd_signal_diff_pct",
    "rsi_trend_alignment",
    "kaufman_efficiency_ratio",
    "bb_upper_dist",
    "bb_lower_dist",
    "bb_pb",
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
SWING_BASELINE_ABLATION_INPUTS: Final = {
    "momentum_volatility": MOMENTUM_FEATURES,
    "trend_confirmation": tuple(
        dict.fromkeys((*MOMENTUM_FEATURES, *TREND_FEATURES))
    ),
    "pullback_timing": tuple(
        dict.fromkeys((*MOMENTUM_FEATURES, *TREND_FEATURES, *PULLBACK_FEATURES))
    ),
    "volume_liquidity": TECHNICAL_RANKING_FEATURES,
}
SWING_BASELINE_ABLATION_ORDER: Final = tuple(SWING_BASELINE_ABLATION_INPUTS)

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
    from market_predictor.edge_rebuild.pipeline import FeaturePipeline
    from market_predictor.edge_rebuild.swing_pipeline_steps import (
        SetupComponentsStep,
        TechnicalRelationshipsStep,
    )

    # Indicators are computed inside `build_swing_feature_history`, which sees the
    # full warm-up history. Nothing here may recompute them: an AdvancedIndicators
    # step used to run at this point with min_periods=1, after warm-up rows had
    # been dropped, and overwrote the correct values for 88,999 eligible rows
    # (11%) with a "200-day average" built from as little as one bar.
    pipeline = FeaturePipeline([
        SetupComponentsStep(benchmark_features),
        TechnicalRelationshipsStep(contract),
    ])
    rows = pipeline.transform(labelled)

    rows = _apply_sector_benchmark_eligibility(
        rows,
        horizon_sessions=effective.horizon_sessions,
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


def finalize_swing_feature_panel(
    rows: pd.DataFrame,
    *,
    contract: StrategyContract,
    expected_security_ids: Sequence[str] | None = None,
) -> pd.DataFrame:
    """Add same-session transforms and the sector-relative managed-return label."""

    from market_predictor.edge_rebuild.pipeline import FeaturePipeline
    from market_predictor.edge_rebuild.swing_pipeline_steps import (
        CrossSectionalRankStep,
        CrossSectionalValidationStep,
        SectorRelativeScalingStep,
    )

    pipeline = FeaturePipeline([
        CrossSectionalValidationStep(expected_security_ids=expected_security_ids),
        SectorRelativeScalingStep(contract=contract),
        CrossSectionalRankStep(contract=contract),
    ])
    return pipeline.transform(rows)

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


def swing_baseline_feature_columns(
    feature_group: str,
    *,
    contract: StrategyContract,
) -> tuple[str, ...]:
    """Return one preregistered nested swing-baseline feature contract."""

    inputs = SWING_BASELINE_ABLATION_INPUTS.get(feature_group)
    if inputs is None:
        raise DataReadinessError(
            f"unsupported swing baseline feature group: {feature_group}"
        )
    return tuple(
        cross_sectional_feature_names(
            inputs,
            spec=_cross_section_spec(contract),
        )
    )


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
