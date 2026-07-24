from __future__ import annotations

import hashlib
import json
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

INTRADAY_FEATURE_SCHEMA_VERSION = "intraday.features.v1"
INTRADAY_MODEL_SCHEMA_VERSION = "intraday.model.v1"
INTRADAY_MODEL_TYPE = "canonical_intraday"
INTRADAY_VALIDATION_SPLIT = "session_purged_walk_forward_and_ticker_holdout"

SECTOR_BENCHMARKS = (
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)

FIVE_MINUTE_FEATURES = (
    "return_1bar_5m",
    "return_3bar_5m",
    "return_6bar_5m",
    "return_12bar_5m",
    "realized_vol_6bar_5m",
    "realized_vol_12bar_5m",
    "dist_ema_10_5m",
    "dist_ema_20_5m",
    "dist_ema_50_5m",
    "ema_10_slope_3bar_5m",
    "ema_20_slope_3bar_5m",
    "macd_signal_diff_pct_5m",
    "rsi_14_5m",
    "atr_pct_14_5m",
    "dist_session_vwap_5m",
    "session_vwap_slope_3bar_5m",
    "opening_range_width_pct_5m",
    "dist_opening_range_high_5m",
    "dist_opening_range_low_5m",
    "overnight_gap",
    "relative_volume_same_slot_20d_5m",
    "volume_burst_20bar_5m",
    "dollar_volume_log_5m",
    "dist_recent_20bar_high_atr_5m",
    "dist_recent_20bar_low_atr_5m",
    "range_pct_5m",
    "close_location_5m",
    "session_progress",
    "session_minute_sin",
    "session_minute_cos",
)

ONE_MINUTE_FEATURES = (
    "return_1bar_1m",
    "return_3bar_1m",
    "return_5bar_1m",
    "realized_vol_5bar_1m",
    "realized_vol_20bar_1m",
    "dist_ema_5_1m",
    "dist_ema_20_1m",
    "macd_signal_diff_pct_1m",
    "rsi_14_1m",
    "atr_pct_14_1m",
    "dist_session_vwap_1m",
    "volume_burst_20bar_1m",
    "relative_volume_same_slot_20d_1m",
    "range_pct_1m",
    "close_location_1m",
)

BENCHMARK_FEATURES = (
    "spy_return_1bar_5m",
    "spy_return_3bar_5m",
    "spy_return_6bar_5m",
    "qqq_return_1bar_5m",
    "qqq_return_3bar_5m",
    "qqq_return_6bar_5m",
    "sector_return_1bar_5m",
    "sector_return_3bar_5m",
    "sector_return_6bar_5m",
    "rel_return_1bar_vs_qqq_5m",
    "rel_return_3bar_vs_qqq_5m",
    "rel_return_6bar_vs_qqq_5m",
    "rel_return_1bar_vs_sector_5m",
    "rel_return_3bar_vs_sector_5m",
    "rel_return_6bar_vs_sector_5m",
    "spy_dist_session_vwap_5m",
    "qqq_dist_session_vwap_5m",
    "sector_dist_session_vwap_5m",
    "eligible_breadth_positive_1bar",
    "eligible_breadth_above_vwap",
    "regime_risk_on",
    "regime_risk_off",
    "regime_high_volatility",
)

CROSS_SECTIONAL_FEATURES = (
    "xs_rank_return_3bar_5m",
    "xs_rank_rel_return_3bar_vs_qqq_5m",
    "xs_rank_rel_return_3bar_vs_sector_5m",
    "xs_rank_relative_volume_same_slot_20d_5m",
    "xs_rank_volume_burst_20bar_5m",
    "xs_rank_atr_pct_14_5m",
    "xs_rank_dist_session_vwap_5m",
    "xs_rank_return_3bar_1m",
)

MEMBERSHIP_FEATURES = (
    "market_cap_micro",
    "market_cap_small",
    "market_cap_mid",
    "market_cap_large",
    "market_cap_mega",
    "liquidity_low",
    "liquidity_medium",
    "liquidity_high",
    *(f"sector_benchmark_{ticker.lower()}" for ticker in SECTOR_BENCHMARKS),
)

INTRADAY_MODEL_FEATURES = tuple(
    dict.fromkeys(
        (
            *FIVE_MINUTE_FEATURES,
            *ONE_MINUTE_FEATURES,
            *BENCHMARK_FEATURES,
            *CROSS_SECTIONAL_FEATURES,
            *MEMBERSHIP_FEATURES,
        )
    )
)

CATALYST_AUDIT_FEATURES = (
    "event_count_2h",
    "event_count_1d",
    "sentiment_mean_2h",
    "sentiment_mean_1d",
    "sentiment_coverage_2h",
    "sentiment_coverage_1d",
    "event_relevance_mean_2h",
    "event_relevance_mean_1d",
    "low_relevance_event_fraction_2h",
    "low_relevance_event_fraction_1d",
    "source_count_alpaca_2h",
    "source_count_reddit_2h",
    "source_count_seeking_alpha_2h",
    "source_count_sec_2h",
    "source_count_finviz_2h",
    "global_event_count_2h",
    "global_event_count_1d",
    "global_sentiment_mean_2h",
    "global_sentiment_mean_1d",
    "global_sentiment_coverage_2h",
    "global_sentiment_coverage_1d",
)


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntradayDatasetConfig(FrozenConfig):
    horizon_minutes: int = Field(default=60, ge=5, le=240)
    decision_bar_minutes: Literal[5] = 5
    execution_bar_minutes: Literal[1] = 1
    decision_stride_bars: int = Field(default=3, ge=1, le=12)
    target_atr: float = Field(default=1.0, gt=0, le=10)
    stop_atr: float = Field(default=0.75, gt=0, le=10)
    round_trip_cost_bps: float = Field(default=10.0, ge=0, le=500)
    min_five_minute_bars: int = Field(default=130, ge=50, le=2_000)
    min_one_minute_bars: int = Field(default=130, ge=50, le=10_000)
    minimum_cross_section: int = Field(default=20, ge=2)
    first_decision_minute_et: int = Field(default=9 * 60 + 45, ge=9 * 60 + 30, le=15 * 60 + 55)
    last_decision_minute_et: int = Field(default=14 * 60 + 55, ge=9 * 60 + 30, le=15 * 60 + 55)
    required_price_feed: str = "sip"
    required_adjustment: str = "all"
    broad_benchmark: str = "SPY"
    growth_benchmark: str = "QQQ"
    required_catalyst_sources: tuple[str, ...] = ("alpaca",)
    required_global_sources: tuple[str, ...] = ("alpaca", "gdelt")
    source_coverage_max_age_minutes: int = Field(default=60, ge=0, le=1_440)
    ambiguous_barrier_policy: Literal["stop"] = "stop"
    max_build_memory_gb: float = Field(default=4.0, ge=1.0, le=64)
    memory_guard_headroom_gb: float = Field(default=0.25, ge=0.1, le=2.0)
    schema_version: str = INTRADAY_FEATURE_SCHEMA_VERSION

    def label_policy(self) -> dict[str, object]:
        """Complete reproducible intraday path and cost semantics."""

        return {
            "policy": "intraday_label.v2",
            "horizon_minutes": self.horizon_minutes,
            "decision_bar_minutes": self.decision_bar_minutes,
            "execution_bar_minutes": self.execution_bar_minutes,
            "decision_stride_bars": self.decision_stride_bars,
            "target_atr": self.target_atr,
            "stop_atr": self.stop_atr,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "ambiguous_barrier_policy": self.ambiguous_barrier_policy,
            "entry_rule": "exact_bar_start_at_decision_time",
            "stop_fill_rule": "worse_of_stop_or_trigger_open",
            "target_fill_rule": "target_price",
            "timeout_fill_rule": "final_horizon_bar_close",
            "broad_benchmark": self.broad_benchmark.upper(),
            "growth_benchmark": self.growth_benchmark.upper(),
        }

    def label_config_sha256(self) -> str:
        """Content hash of the complete intraday label/cost semantics."""

        payload = self.label_policy()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()

    @model_validator(mode="after")
    def validate_dataset(self) -> Self:
        if self.horizon_minutes % self.execution_bar_minutes:
            raise ValueError("horizon_minutes must be divisible by execution_bar_minutes")
        if self.first_decision_minute_et > self.last_decision_minute_et:
            raise ValueError("first_decision_minute_et must not exceed last_decision_minute_et")
        if self.memory_guard_headroom_gb >= self.max_build_memory_gb:
            raise ValueError("memory guard headroom must be below the hard budget")
        return self


class IntradayTrainingConfig(FrozenConfig):
    family: Literal["logistic", "hist_gradient_boosting"] = "hist_gradient_boosting"
    n_splits: int = Field(default=4, ge=2, le=8)
    embargo_sessions: int = Field(default=1, ge=1, le=10)
    min_train_sessions: int = Field(default=60, ge=20)
    min_train_rows: int = Field(default=10_000, ge=100)
    min_training_tickers: int = Field(default=100, ge=2)
    min_features: int = Field(default=25, ge=5)
    min_feature_non_null_rate: float = Field(default=0.05, ge=0, le=1)
    ticker_holdout_fraction: float = Field(default=0.2, gt=0, lt=1)
    top_k: int = Field(default=10, ge=1, le=100)
    max_downside_probability: float = Field(default=0.45, ge=0, le=1)
    max_trades_per_session: int = Field(default=10, ge=1, le=100)
    max_iter: int = Field(default=250, ge=25, le=2_000)
    learning_rate: float = Field(default=0.04, gt=0, le=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    random_seed: int = 42
    max_training_memory_gb: float = Field(default=4.0, ge=1.0, le=64)
    memory_guard_headroom_gb: float = Field(default=0.25, ge=0.1, le=2.0)
    schema_version: str = INTRADAY_MODEL_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_training(self) -> Self:
        if self.memory_guard_headroom_gb >= self.max_training_memory_gb:
            raise ValueError("memory guard headroom must be below the hard budget")
        if self.top_k > self.min_train_rows:
            raise ValueError("top_k cannot exceed min_train_rows")
        return self


class IntradayPromotionConfig(FrozenConfig):
    min_opportunity_roc_auc: float = Field(default=0.58, ge=0.5, le=1)
    min_opportunity_holdout_roc_auc: float = Field(default=0.54, ge=0.5, le=1)
    min_opportunity_top_decile_lift: float = Field(default=1.10, ge=1)
    min_opportunity_holdout_lift: float = Field(default=1.03, ge=1)
    min_opportunity_group_lift_at_k: float = Field(default=1.05, ge=0)
    min_opportunity_holdout_group_lift_at_k: float = Field(default=1.02, ge=0)
    min_downside_roc_auc: float = Field(default=0.55, ge=0.5, le=1)
    min_downside_holdout_roc_auc: float = Field(default=0.52, ge=0.5, le=1)
    max_opportunity_brier: float = Field(default=0.25, ge=0, le=1)
    max_downside_brier: float = Field(default=0.25, ge=0, le=1)
    max_calibration_error: float = Field(default=0.10, ge=0, le=1)
    min_validated_rows: int = Field(default=20_000, ge=100)
    min_tickers: int = Field(default=200, ge=2)
    min_decision_groups: int = Field(default=250, ge=1)
    min_independent_sessions: int = Field(default=60, ge=1)
    min_validation_folds: int = Field(default=4, ge=1)
    min_effective_sample_size: float = Field(default=200.0, ge=0)
    min_stress_avg_trade_return: float = 0.0
    min_stress_avg_excess_return_vs_spy: float = 0.0
    min_worst_regime_avg_excess_return_vs_spy: float = -0.01
    max_worst_regime_drawdown: float = Field(default=0.30, gt=0, le=1)
    max_worst_regime_calibration_error: float = Field(default=0.15, ge=0, le=1)
    min_capacity_avg_net_return: float = -0.02
    min_selected_trades: int = Field(default=200, ge=1)
    min_avg_trade_return: float = 0.0
    min_avg_excess_return_vs_spy: float = 0.0
    min_avg_excess_return_vs_qqq: float = 0.0
    min_avg_excess_return_vs_sector: float = 0.0
    min_profit_factor: float = Field(default=1.05, ge=0)
    max_drawdown: float = Field(default=0.15, gt=0, le=1)
    min_return_drawdown_ratio: float = Field(default=0.5, ge=0)
    max_negative_session_rate: float = Field(default=0.55, ge=0, le=1)
    max_average_turnover: float = Field(default=1.0, ge=0, le=2)
    min_regimes: int = Field(default=3, ge=1)
    max_single_regime_share: float = Field(default=0.85, gt=0, le=1)
    min_catalyst_coverage_rate: float = Field(default=0.05, ge=0, le=1)
    max_low_relevance_event_rate: float = Field(default=0.25, ge=0, le=1)
    max_alignment_errors: int = Field(default=0, ge=0)
    max_peak_working_set_gib: float = Field(default=4.0, ge=1.0)


def opportunity_target_column(horizon_minutes: int) -> str:
    return f"target_before_stop_{horizon_minutes}m"


def downside_target_column(horizon_minutes: int) -> str:
    return f"stop_before_target_{horizon_minutes}m"


def net_return_column(horizon_minutes: int) -> str:
    return f"path_realized_return_net_{horizon_minutes}m"


def excess_return_column(horizon_minutes: int, benchmark: str) -> str:
    return f"path_excess_return_{horizon_minutes}m_vs_{benchmark.lower()}"
