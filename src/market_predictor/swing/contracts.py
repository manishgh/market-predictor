from __future__ import annotations

from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

SWING_FEATURE_SCHEMA_VERSION = "swing.features.v1"
SWING_MODEL_SCHEMA_VERSION = "swing.model.v1"
SWING_MODEL_TYPE = "canonical_swing"
SWING_VALIDATION_SPLIT = "session_purged_walk_forward_and_ticker_holdout"

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

TECHNICAL_FEATURES = (
    "return_1d",
    "return_5d",
    "return_10d",
    "return_20d",
    "return_60d",
    "realized_vol_10d",
    "realized_vol_20d",
    "realized_vol_60d",
    "atr_pct_14",
    "rsi_14",
    "macd_signal_diff_pct",
    "dist_ema_10",
    "dist_ema_20",
    "dist_ema_50",
    "dist_sma_20",
    "dist_sma_50",
    "dist_sma_200",
    "sma_200_slope_20d",
    "gap_return",
    "intraday_return",
    "range_pct",
    "close_location",
    "volume_z20",
    "volume_ratio_20",
    "dollar_volume_log",
)

BENCHMARK_FEATURES = (
    "spy_return_1d",
    "spy_return_5d",
    "spy_return_20d",
    "spy_realized_vol_20d",
    "spy_dist_sma_200",
    "qqq_return_1d",
    "qqq_return_5d",
    "qqq_return_20d",
    "qqq_realized_vol_20d",
    "qqq_dist_sma_200",
    "sector_return_1d",
    "sector_return_5d",
    "sector_return_20d",
    "sector_realized_vol_20d",
    "sector_dist_sma_200",
    "rel_return_1d_vs_spy",
    "rel_return_5d_vs_spy",
    "rel_return_20d_vs_spy",
    "rel_return_1d_vs_sector",
    "rel_return_5d_vs_sector",
    "rel_return_20d_vs_sector",
    "regime_risk_on",
    "regime_risk_off",
)

CATALYST_FEATURES = (
    "event_count_2h",
    "event_count_1d",
    "event_count_3d",
    "sentiment_mean_2h",
    "sentiment_mean_1d",
    "sentiment_mean_3d",
    "sentiment_coverage_2h",
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
    "global_event_count_1d",
    "global_event_count_3d",
    "global_sentiment_mean_1d",
    "global_sentiment_mean_3d",
    "global_sentiment_coverage_1d",
    "global_sentiment_coverage_3d",
)

FUNDAMENTAL_FEATURES = (
    "fundamental_revenue",
    "fundamental_net_income",
    "fundamental_eps_diluted",
    "fundamental_operating_cash_flow",
    "fundamental_revenue_present",
    "fundamental_net_income_present",
    "fundamental_eps_diluted_present",
    "fundamental_operating_cash_flow_present",
)

CROSS_SECTIONAL_FEATURES = (
    "xs_rank_return_5d",
    "xs_rank_return_20d",
    "xs_rank_volume_z20",
    "xs_rank_rel_return_20d_vs_spy",
    "xs_rank_rel_return_20d_vs_sector",
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

SWING_FEATURES = tuple(
    dict.fromkeys(
        (
            *TECHNICAL_FEATURES,
            *BENCHMARK_FEATURES,
            *CATALYST_FEATURES,
            *FUNDAMENTAL_FEATURES,
            *CROSS_SECTIONAL_FEATURES,
            *MEMBERSHIP_FEATURES,
        )
    )
)


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingDatasetConfig(FrozenConfig):
    horizon_sessions: int = Field(default=5, ge=1, le=20)
    round_trip_cost_bps: float = Field(default=10.0, ge=0, le=500)
    min_daily_bars: int = Field(default=250, ge=220, le=1_000)
    required_price_feed: str = "sip"
    required_adjustment: str = "all"
    broad_benchmark: str = "SPY"
    growth_benchmark: str = "QQQ"
    required_global_sources: tuple[str, ...] = ("alpaca", "gdelt")
    source_coverage_max_age_minutes: int = Field(default=60, ge=0, le=1_440)
    minimum_cross_section: int = Field(default=20, ge=2)
    schema_version: str = SWING_FEATURE_SCHEMA_VERSION


class SwingTrainingConfig(FrozenConfig):
    family: str = "hist_gradient_boosting"
    n_splits: int = Field(default=4, ge=2, le=8)
    min_train_sessions: int = Field(default=120, ge=20)
    min_train_rows: int = Field(default=5_000, ge=100)
    min_training_tickers: int = Field(default=100, ge=2)
    min_features: int = Field(default=25, ge=5)
    min_feature_non_null_rate: float = Field(default=0.05, ge=0, le=1)
    ticker_holdout_fraction: float = Field(default=0.2, gt=0, lt=1)
    top_k: int = Field(default=10, ge=1, le=100)
    max_iter: int = Field(default=250, ge=25, le=2_000)
    learning_rate: float = Field(default=0.04, gt=0, le=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    random_seed: int = 42
    max_training_memory_gb: float = Field(default=4.0, ge=1.0, le=64)
    memory_guard_headroom_gb: float = Field(default=0.25, ge=0.1, le=2.0)
    schema_version: str = SWING_MODEL_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_training(self) -> Self:
        if self.family not in {"logistic", "hist_gradient_boosting"}:
            raise ValueError("family must be logistic or hist_gradient_boosting")
        if self.memory_guard_headroom_gb >= self.max_training_memory_gb:
            raise ValueError("memory guard headroom must be below the hard budget")
        if self.top_k > self.min_train_rows:
            raise ValueError("top_k cannot exceed min_train_rows")
        return self


class SwingPromotionConfig(FrozenConfig):
    min_roc_auc: float = Field(default=0.60, ge=0.5, le=1)
    min_ticker_holdout_roc_auc: float = Field(default=0.55, ge=0.5, le=1)
    min_top_decile_lift: float = Field(default=1.15, ge=1)
    min_ticker_holdout_lift: float = Field(default=1.05, ge=1)
    min_validated_rows: int = Field(default=20_000, ge=100)
    min_tickers: int = Field(default=200, ge=2)
    min_selected_trades: int = Field(default=100, ge=1)
    min_avg_trade_return: float = 0.0
    min_avg_excess_return_vs_spy: float = 0.0
    min_avg_excess_return_vs_qqq: float = 0.0
    min_avg_excess_return_vs_sector: float = 0.0
    min_profit_factor: float = Field(default=1.05, ge=0)
    max_drawdown: float = Field(default=0.20, gt=0, le=1)
    min_return_drawdown_ratio: float = Field(default=0.5, ge=0)
    max_negative_period_rate: float = Field(default=0.55, ge=0, le=1)
    min_regimes: int = Field(default=3, ge=1)
    max_single_regime_share: float = Field(default=0.85, gt=0, le=1)
    min_catalyst_row_rate: float = Field(default=0.05, ge=0, le=1)
    max_low_relevance_event_rate: float = Field(default=0.25, ge=0, le=1)
    max_alignment_errors: int = Field(default=0, ge=0)
    max_peak_working_set_gib: float = Field(default=4.0, ge=1.0)


def swing_target_column(horizon_sessions: int) -> str:
    return f"target_net_positive_{horizon_sessions}d"


def swing_net_return_column(horizon_sessions: int) -> str:
    return f"future_net_return_{horizon_sessions}d"


def swing_excess_column(horizon_sessions: int, benchmark: str) -> str:
    return f"future_excess_return_{horizon_sessions}d_vs_{benchmark.lower()}"
