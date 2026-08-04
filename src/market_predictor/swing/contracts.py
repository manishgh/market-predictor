from __future__ import annotations

import hashlib
import json
from datetime import date
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

MINIMUM_SWING_DECISION_DATE = date(2019, 7, 9)
SWING_FEATURE_SCHEMA_VERSION = "swing.features.v3"
SWING_MODEL_SCHEMA_VERSION = "swing.model.v1"
SWING_MODEL_TYPE = "canonical_swing"
SWING_VALIDATION_SPLIT = "session_purged_walk_forward_and_ticker_holdout"
SWING_REQUIRED_MARKET_REGIMES = ("risk_on", "neutral", "risk_off")
SwingFeatureProfile = Literal["technical_market", "catalyst_full"]

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
    "source_count_alpaca_1d",
    "source_count_alpaca_3d",
    "source_count_sec_1d",
    "source_count_sec_3d",
    "source_count_finviz_1d",
    "source_count_finviz_3d",
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

TECHNICAL_MARKET_FEATURES = tuple(
    dict.fromkeys(
        (
            *TECHNICAL_FEATURES,
            *BENCHMARK_FEATURES,
            *CROSS_SECTIONAL_FEATURES,
        )
    )
)

SWING_FEATURE_PROFILES: dict[SwingFeatureProfile, tuple[str, ...]] = {
    "technical_market": TECHNICAL_MARKET_FEATURES,
    "catalyst_full": SWING_FEATURES,
}


def swing_features_for_profile(profile: SwingFeatureProfile) -> tuple[str, ...]:
    return SWING_FEATURE_PROFILES[profile]


class FrozenConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingDatasetConfig(FrozenConfig):
    feature_profile: SwingFeatureProfile = "catalyst_full"
    decision_start_date: date = MINIMUM_SWING_DECISION_DATE
    decision_end_date: date | None = None
    horizon_sessions: int = Field(default=5, ge=1, le=20)
    round_trip_cost_bps: float = Field(default=10.0, ge=0, le=500)
    min_daily_bars: int = Field(default=250, ge=220, le=1_000)
    required_price_feed: str = "sip"
    required_adjustment: str = "all"
    broad_benchmark: str = "SPY"
    growth_benchmark: str = "QQQ"
    required_ticker_sources: tuple[str, ...] = ("alpaca",)
    required_global_sources: tuple[str, ...] = ()
    source_coverage_max_age_minutes: int = Field(default=60, ge=0, le=1_440)
    minimum_cross_section: int = Field(default=20, ge=2)
    min_exact_label_coverage: float = Field(default=0.995, ge=0.95, le=1.0)
    max_build_memory_gb: float = Field(default=4.0, ge=1.0, le=5.0)
    memory_guard_headroom_gb: float = Field(default=0.75, ge=0.5, le=2.0)
    schema_version: str = SWING_FEATURE_SCHEMA_VERSION

    @model_validator(mode="after")
    def validate_profile(self) -> Self:
        if self.decision_start_date < MINIMUM_SWING_DECISION_DATE:
            raise ValueError(
                "decision_start_date must be on or after "
                f"{MINIMUM_SWING_DECISION_DATE.isoformat()}"
            )
        if (
            self.decision_end_date is not None
            and self.decision_start_date > self.decision_end_date
        ):
            raise ValueError("decision_start_date must not follow decision_end_date")
        if self.memory_guard_headroom_gb >= self.max_build_memory_gb:
            raise ValueError("memory guard headroom must be below the hard budget")
        if self.feature_profile == "technical_market" and (
            self.required_ticker_sources or self.required_global_sources
        ):
            raise ValueError(
                "technical_market requires required_ticker_sources=[] and "
                "required_global_sources=[]; "
                "news coverage is not part of this baseline"
            )
        if self.feature_profile == "catalyst_full" and not self.required_ticker_sources:
            raise ValueError("catalyst_full requires at least one ticker source")
        return self

    def label_policy(self) -> dict[str, object]:
        """Complete reproducible swing outcome semantics."""

        return {
            "policy": "swing_label.v2",
            "horizon_sessions": self.horizon_sessions,
            "round_trip_cost_bps": self.round_trip_cost_bps,
            "entry_rule": "next_exact_exchange_session_open",
            "exit_rule": "decision_plus_horizon_session_close",
            "path_rule": "all_exchange_sessions_required",
            "broad_benchmark": self.broad_benchmark.upper(),
            "growth_benchmark": self.growth_benchmark.upper(),
        }

    def label_config_sha256(self) -> str:
        """Content hash of the complete swing label/cost semantics."""

        payload = self.label_policy()
        return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


class SwingTrainingConfig(FrozenConfig):
    feature_profile: SwingFeatureProfile = "catalyst_full"
    family: str = "hist_gradient_boosting"
    n_splits: int = Field(default=4, ge=2, le=8)
    min_train_sessions: int = Field(default=120, ge=20)
    min_train_rows: int = Field(default=5_000, ge=100)
    min_training_tickers: int = Field(default=100, ge=2)
    min_features: int = Field(default=25, ge=5)
    min_feature_non_null_rate: float = Field(default=0.05, ge=0, le=1)
    ticker_holdout_fraction: float = Field(default=0.2, gt=0, lt=1)
    top_k: int = Field(default=10, ge=1, le=100)
    min_regime_sessions: int = Field(default=5, ge=2)
    min_regime_trades: int = Field(default=20, ge=1)
    max_iter: int = Field(default=250, ge=25, le=2_000)
    learning_rate: float = Field(default=0.04, gt=0, le=1)
    l2_regularization: float = Field(default=1.0, ge=0)
    random_seed: int = 42
    max_training_memory_gb: float = Field(default=4.0, ge=1.0, le=5.0)
    memory_guard_headroom_gb: float = Field(default=0.75, ge=0.5, le=2.0)
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
    min_group_lift_at_k: float = Field(default=1.10, ge=0)
    min_ticker_holdout_group_lift_at_k: float = Field(default=1.03, ge=0)
    min_validated_rows: int = Field(default=20_000, ge=100)
    min_tickers: int = Field(default=200, ge=2)
    min_decision_groups: int = Field(default=250, ge=1)
    min_independent_sessions: int = Field(default=120, ge=1)
    min_validation_folds: int = Field(default=4, ge=1)
    min_stress_avg_trade_return: float = 0.0
    min_stress_avg_excess_return_vs_spy: float = 0.0
    min_worst_regime_avg_excess_return_vs_spy: float = -0.01
    min_worst_regime_avg_trade_return_ci_low: float = 0.0
    min_worst_regime_avg_excess_return_vs_spy_ci_low: float = 0.0
    min_required_regime_sessions: int = Field(default=5, ge=2)
    min_required_regime_trades: int = Field(default=20, ge=1)
    max_worst_regime_drawdown: float = Field(default=0.35, gt=0, le=1)
    max_worst_regime_calibration_error: float = Field(default=0.15, ge=0, le=1)
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
    max_calibration_error: float = Field(default=0.10, ge=0, le=1)
    max_ticker_holdout_calibration_error: float = Field(default=0.12, ge=0, le=1)
    max_calibration_bias: float = Field(default=0.05, ge=0, le=1)
    min_calibration_slope: float = Field(default=0.70, ge=0)
    max_calibration_slope: float = Field(default=1.40, ge=0)
    max_abs_calibration_intercept: float = Field(default=0.10, ge=0, le=1)
    max_alignment_errors: int = Field(default=0, ge=0)
    max_peak_working_set_gib: float = Field(default=4.0, ge=1.0)


def swing_target_column(horizon_sessions: int) -> str:
    return f"target_net_positive_{horizon_sessions}d"


def swing_net_return_column(horizon_sessions: int) -> str:
    return f"future_net_return_{horizon_sessions}d"


def swing_excess_column(horizon_sessions: int, benchmark: str) -> str:
    return f"future_excess_return_{horizon_sessions}d_vs_{benchmark.lower()}"
