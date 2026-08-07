"""Frozen research contracts for KS4 intraday specialists."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.v3.errors import DataReadinessError

INTRADAY_SPECIALIST_RESEARCH_SCHEMA = "intraday.specialist_research.v1"

INTRADAY_SPECIALIST_IDS = (
    "INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1",
    "INTRADAY.GAP_CONTINUATION.60M.V1",
    "INTRADAY.GAP_FADE.60M.V1",
    "INTRADAY.VWAP_CONTINUATION.60M.V1",
    "INTRADAY.VWAP_REVERSION.30M.V1",
    "INTRADAY.MOMENTUM_CONTINUATION.60M.V1",
    "INTRADAY.SHORT_HORIZON_REVERSAL.30M.V1",
)

INTRADAY_SPECIALIST_SOURCE_FEATURES = frozenset(
    {
        "return_1bar",
        "return_3bar",
        "return_6bar",
        "return_12bar",
        "volatility_3bar",
        "volatility_6bar",
        "volatility_12bar",
        "ema_10_slope_3bar",
        "ema_20_slope_3bar",
        "macd_histogram",
        "rsi_14",
        "atr_pct",
        "dist_session_vwap",
        "session_vwap_slope_3bar",
        "opening_range_width_pct",
        "dist_opening_range_high",
        "dist_opening_range_low",
        "overnight_gap",
        "volume_burst_20bar",
        "relative_volume_same_minute_20d",
        "dollar_volume",
        "dist_recent_20bar_high_atr",
        "dist_recent_20bar_low_atr",
        "qqq_return_1bar",
        "qqq_return_3bar",
        "qqq_return_6bar",
        "spy_return_1bar",
        "spy_return_3bar",
        "spy_return_6bar",
        "sector_return_1bar",
        "sector_return_3bar",
        "sector_return_6bar",
        "rel_return_1bar_vs_qqq",
        "rel_return_1bar_vs_sector",
        "rel_return_3bar_vs_qqq",
        "rel_return_3bar_vs_sector",
        "rel_return_6bar_vs_qqq",
        "rel_return_6bar_vs_sector",
        "eligible_breadth_positive_1bar",
        "eligible_breadth_above_vwap",
        "regime_risk_on",
        "regime_risk_off",
        "regime_high_volatility",
        "xs_rank_return_1bar",
        "xs_rank_return_3bar",
        "xs_rank_return_6bar",
        "xs_rank_relative_volume_same_minute_20d",
        "xs_rank_dollar_volume",
        "xs_rank_atr_pct",
        "xs_rank_dist_session_vwap",
        "xs_rank_dist_opening_range_high",
        "xs_rank_rel_return_3bar_vs_qqq",
        "xs_rank_rel_return_3bar_vs_sector",
        "xs_rank_overnight_gap",
        "return_3bar_atr_units",
        "dist_session_vwap_atr_units",
        "close_location_5m",
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
        "range_pct_1m",
        "close_location_1m",
    }
)

INTRADAY_CATALYST_OVERLAY_FEATURES = frozenset(
    {
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
        "source_family_count_2h",
        "source_family_count_1d",
        "catalyst_source_complete_1d",
    }
)

EstimatorFamily = Literal[
    "deterministic_baseline",
    "logistic",
    "hist_gradient_boosting",
    "direct_ranker",
]
SelectionPolicy = Literal[
    "model_score_only",
    "catalyst_confirmation_overlay",
]
SessionSegment = Literal[
    "opening",
    "late_opening",
    "midday",
    "late_session",
]
DeterministicScore = Literal[
    "opening_breakout_confirmation",
    "gap_continuation_confirmation",
    "gap_fade_confirmation",
    "vwap_continuation_confirmation",
    "vwap_reversion_confirmation",
    "cross_sectional_momentum",
    "shock_reversal_confirmation",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntradaySetupRule(FrozenModel):
    feature: str = Field(min_length=1)
    minimum: float | None = None
    maximum: float | None = None

    @model_validator(mode="after")
    def validate_rule(self) -> Self:
        if self.feature not in INTRADAY_SPECIALIST_SOURCE_FEATURES:
            raise ValueError(f"unsupported KS4 setup feature: {self.feature}")
        if self.minimum is None and self.maximum is None:
            raise ValueError("setup rule needs a minimum or maximum")
        if (
            self.minimum is not None
            and self.maximum is not None
            and self.minimum > self.maximum
        ):
            raise ValueError("setup rule minimum must not exceed maximum")
        return self


class IntradaySpecialistStrategyConfig(FrozenModel):
    direction: Literal["long"] = "long"
    horizon_minutes: Literal[30, 60]
    session_segment: SessionSegment
    first_decision_minute_et: int = Field(ge=570, le=959)
    last_decision_minute_et_exclusive: int = Field(ge=571, le=960)
    minimum_setup_spacing_minutes: int = Field(ge=30, le=240)
    deterministic_score: DeterministicScore
    estimator_families: tuple[EstimatorFamily, ...]
    selection_policies: tuple[SelectionPolicy, ...]
    setup_rules: tuple[IntradaySetupRule, ...]

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        if self.first_decision_minute_et >= self.last_decision_minute_et_exclusive:
            raise ValueError("intraday specialist session interval is empty")
        if (
            self.last_decision_minute_et_exclusive - 1
            + self.horizon_minutes
            > 960
        ):
            raise ValueError("intraday specialist horizon extends beyond session close")
        if self.minimum_setup_spacing_minutes < self.horizon_minutes:
            raise ValueError("setup spacing must cover the complete label horizon")
        if (
            not self.estimator_families
            or len(set(self.estimator_families)) != len(self.estimator_families)
        ):
            raise ValueError("estimator families must be non-empty and unique")
        if "deterministic_baseline" not in self.estimator_families:
            raise ValueError("every KS4 strategy needs a deterministic baseline")
        if (
            not self.selection_policies
            or len(set(self.selection_policies)) != len(self.selection_policies)
        ):
            raise ValueError("selection policies must be non-empty and unique")
        if self.selection_policies != (
            "model_score_only",
            "catalyst_confirmation_overlay",
        ):
            raise ValueError("KS4 selection policies are frozen in comparison order")
        if not self.setup_rules:
            raise ValueError("intraday specialist setup rules cannot be empty")
        rule_features = [rule.feature for rule in self.setup_rules]
        if len(set(rule_features)) != len(rule_features):
            raise ValueError("intraday specialist setup rule features must be unique")
        return self


class IntradaySpecialistResearchConfig(FrozenModel):
    schema_version: str = INTRADAY_SPECIALIST_RESEARCH_SCHEMA
    required_technical_manifest_schema: str
    required_price_feed: Literal["sip"] = "sip"
    required_adjustment: Literal["all"] = "all"
    intraday_finalization_delay_seconds: Literal[30] = 30
    entry_latency_minutes: Literal[1] = 1
    cross_section_batch_sessions: Literal[5] = 5
    alpaca_unit_max_expected_rows: int = Field(ge=1_000, le=9_500)
    alpaca_unit_max_symbols: int = Field(ge=1, le=50)
    alpaca_collection_workers: int = Field(ge=1, le=4)
    alpaca_collection_retries: int = Field(ge=1, le=8)
    alpaca_request_timeout_seconds: int = Field(ge=10, le=120)
    alpaca_max_pages_per_unit: int = Field(ge=1, le=128)
    minimum_one_minute_warmup_bars: int = Field(ge=20, le=10_000)
    target_atr: float = Field(gt=0, le=10)
    stop_atr: float = Field(gt=0, le=10)
    minimum_round_trip_cost_bps: float = Field(ge=10, le=500)
    n_splits: int = Field(ge=2, le=8)
    embargo_sessions: int = Field(ge=1, le=10)
    min_train_sessions: int = Field(ge=20)
    min_train_rows: int = Field(ge=100)
    min_training_tickers: int = Field(ge=2)
    ticker_holdout_fraction: float = Field(gt=0, lt=1)
    top_k: int = Field(ge=1, le=100)
    max_trades_per_session: int = Field(ge=1, le=100)
    random_seed: int
    logistic_max_iter: int = Field(ge=25, le=2_000)
    hgb_max_iter: int = Field(ge=25, le=2_000)
    hgb_learning_rate: float = Field(gt=0, le=1)
    hgb_l2_regularization: float = Field(ge=0)
    ranker_max_iter: int = Field(ge=25, le=2_000)
    ranker_learning_rate: float = Field(gt=0, le=1)
    ranker_max_depth: int = Field(ge=1, le=12)
    ranker_max_bin: int = Field(ge=16, le=1_024)
    ranker_n_jobs: int = Field(ge=1, le=4)
    minimum_feature_non_null_rate: float = Field(ge=0, le=1)
    minimum_selected_trades: int = Field(ge=1)
    minimum_avg_net_return: float
    minimum_avg_excess_return_vs_spy: float
    minimum_avg_excess_return_vs_sector: float
    minimum_avg_net_return_ci_low: float
    minimum_avg_excess_return_vs_spy_ci_low: float
    minimum_profit_factor: float = Field(ge=0)
    maximum_drawdown: float = Field(gt=0, le=1)
    maximum_negative_session_rate: float = Field(ge=0, le=1)
    required_market_regimes: tuple[str, ...]
    minimum_regime_selected_trades: int = Field(ge=1)
    minimum_regime_avg_net_return: float
    minimum_regime_avg_excess_return_vs_spy: float
    catalyst_overlay_minimum_relevance: float = Field(ge=0, le=1)
    catalyst_overlay_maximum_low_relevance_fraction: float = Field(ge=0, le=1)
    catalyst_overlay_minimum_net_return_improvement: float
    catalyst_overlay_minimum_spy_excess_improvement: float
    catalyst_overlay_maximum_drawdown_increase: float = Field(ge=0, le=1)
    maximum_process_memory_gib: float = Field(ge=1, le=5)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    technical_features: tuple[str, ...]
    catalyst_overlay_features: tuple[str, ...]
    strategies: dict[str, IntradaySpecialistStrategyConfig]

    @model_validator(mode="after")
    def validate_research(self) -> Self:
        if self.schema_version != INTRADAY_SPECIALIST_RESEARCH_SCHEMA:
            raise ValueError(
                "KS4 research schema must be "
                f"{INTRADAY_SPECIALIST_RESEARCH_SCHEMA}"
            )
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard limit")
        if (
            self.intraday_finalization_delay_seconds
            >= self.entry_latency_minutes * 60
        ):
            raise ValueError(
                "entry latency must exceed the intraday finalization delay"
            )
        if (
            self.alpaca_unit_max_expected_rows
            > 10_000 - 500
        ):
            raise ValueError(
                "Alpaca unit rows must retain at least 500 rows of page headroom"
            )
        if tuple(self.strategies) != INTRADAY_SPECIALIST_IDS:
            raise ValueError("KS4 strategy order and identity must match the catalog")
        if (
            not self.technical_features
            or len(set(self.technical_features)) != len(self.technical_features)
        ):
            raise ValueError("technical features must be non-empty and unique")
        unsupported = set(self.technical_features).difference(
            INTRADAY_SPECIALIST_SOURCE_FEATURES
        )
        if unsupported:
            raise ValueError(
                "unsupported KS4 technical features: "
                + ", ".join(sorted(unsupported))
            )
        if set(self.catalyst_overlay_features) != INTRADAY_CATALYST_OVERLAY_FEATURES:
            raise ValueError("KS4 catalyst overlay feature inventory is incomplete")
        for strategy_id, strategy in self.strategies.items():
            has_ranker = "direct_ranker" in strategy.estimator_families
            if has_ranker != (
                strategy_id
                == "INTRADAY.MOMENTUM_CONTINUATION.60M.V1"
            ):
                raise ValueError(
                    "direct ranker is allowed only for intraday momentum"
                )
        return self

    def policy_sha256(self) -> str:
        payload = self.model_dump(mode="json")
        return hashlib.sha256(
            json.dumps(
                payload,
                sort_keys=True,
                separators=(",", ":"),
            ).encode("utf-8")
        ).hexdigest()


def load_intraday_specialist_research_config(
    path: Path,
) -> IntradaySpecialistResearchConfig:
    if not path.is_file():
        raise DataReadinessError(
            f"missing KS4 research policy: {path}"
        )
    with path.open("rb") as handle:
        payload = tomllib.load(handle)
    try:
        return IntradaySpecialistResearchConfig.model_validate(payload)
    except ValueError as exc:
        raise DataReadinessError(
            f"invalid KS4 research policy {path}: {exc}"
        ) from exc


def intraday_specialist_policy_identity(path: Path) -> dict[str, str]:
    config = load_intraday_specialist_research_config(path)
    return {
        "path": path.as_posix(),
        "file_sha256": file_sha256(path),
        "policy_sha256": config.policy_sha256(),
        "schema_version": config.schema_version,
    }
