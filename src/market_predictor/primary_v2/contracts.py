"""Frozen contracts for the primary V2 distributional strategies."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.v3.errors import DataReadinessError

PRIMARY_V2_RESEARCH_SCHEMA = "primary_strategy_v2.research.v1"
SWING_V2_ID = "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V2"
INTRADAY_V2_ID = "INTRADAY.VWAP_REVERSION.30M.V2"
PRIMARY_V2_STRATEGY_IDS = frozenset({SWING_V2_ID, INTRADAY_V2_ID})

Timeframe = Literal["swing", "intraday"]
Direction = Literal["long"]
HorizonUnit = Literal["exchange_sessions", "regular_session_minutes"]
CandidateFamily = Literal[
    "deterministic_v1_baseline",
    "multinomial_v1_baseline",
    "hgb_mean_return",
    "hgb_quantile_return",
    "hgb_competing_risks",
]
SelectionPolicy = Literal[
    "expected_net_top_10",
    "positive_lower_bound_then_median_top_10",
    "no_veto_expected_net_top_10",
    "distributional_safety_top_10",
]


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class CompetingRiskTargets(FrozenModel):
    target_first: str
    stop_first: str
    timeout: str
    outcome: str
    time_to_resolution: str


class PrimaryV2StrategyConfig(FrozenModel):
    display_name: str = Field(min_length=20)
    source_strategy_id: str
    timeframe: Timeframe
    direction: Direction
    horizon_value: int = Field(ge=1)
    horizon_unit: HorizonUnit
    decision_rule: str
    entry_rule: str
    timeout_exit_rule: str
    source_target: str
    spy_excess_target: str
    sector_excess_target: str
    mfe_target: str
    mae_target: str
    period_column: str
    row_id_column: str
    eligibility_column: str
    candidate_families: tuple[CandidateFamily, ...]
    selection_policies: tuple[SelectionPolicy, ...]
    top_k: int = Field(ge=1, le=10)
    minimum_train_sessions: int = Field(ge=20)
    minimum_train_rows: int = Field(ge=100)
    minimum_training_tickers: int = Field(ge=20)
    minimum_round_trip_cost_bps: float = Field(ge=10)
    competing_risk_targets: CompetingRiskTargets | None = None

    @model_validator(mode="after")
    def validate_semantics(self) -> Self:
        if len(set(self.candidate_families)) != len(self.candidate_families):
            raise ValueError("candidate families must be unique")
        if len(set(self.selection_policies)) != len(self.selection_policies):
            raise ValueError("selection policies must be unique")
        if not self.candidate_families or not self.selection_policies:
            raise ValueError("candidate families and selection policies cannot be empty")
        experiment_count = len(self.candidate_families) * len(self.selection_policies)
        if experiment_count > 12:
            raise ValueError("strategy exceeds the frozen 12-experiment budget")
        if self.timeframe == "swing":
            if self.horizon_unit != "exchange_sessions" or self.horizon_value != 5:
                raise ValueError("swing V2 must use five exchange sessions")
            if self.competing_risk_targets is not None:
                raise ValueError("swing V2 cannot define competing-risk targets")
            required = {"deterministic_v1_baseline", "hgb_mean_return", "hgb_quantile_return"}
            if set(self.candidate_families) != required:
                raise ValueError("swing V2 candidate catalog mismatch")
        else:
            if self.horizon_unit != "regular_session_minutes" or self.horizon_value != 30:
                raise ValueError("intraday V2 must use 30 regular-session minutes")
            if self.competing_risk_targets is None:
                raise ValueError("intraday V2 requires competing-risk targets")
            required = {"multinomial_v1_baseline", "hgb_competing_risks", "hgb_quantile_return"}
            if set(self.candidate_families) != required:
                raise ValueError("intraday V2 candidate catalog mismatch")
        return self

    @property
    def required_source_columns(self) -> frozenset[str]:
        columns = {
            "ticker",
            "decision_time_utc",
            "entry_time_utc",
            "exit_time_utc",
            "label_available_at_utc",
            "primary_benchmark",
            self.period_column,
            self.row_id_column,
            self.eligibility_column,
            self.source_target,
            self.spy_excess_target,
            self.sector_excess_target,
            self.mfe_target,
            self.mae_target,
        }
        if self.competing_risk_targets is not None:
            columns.update(self.competing_risk_targets.model_dump().values())
            columns.update({"price_feed", "adjustment"})
        return frozenset(columns)


class PrimaryV2ResearchConfig(FrozenModel):
    schema_version: str = PRIMARY_V2_RESEARCH_SCHEMA
    random_seed: int
    n_splits: int = Field(ge=2, le=8)
    ticker_holdout_fraction: float = Field(gt=0, lt=1)
    minimum_feature_non_null_rate: float = Field(ge=0, le=1)
    minimum_selected_trades: int = Field(ge=100)
    minimum_average_net_return: float
    minimum_average_excess_return_vs_spy: float
    minimum_average_excess_return_vs_sector: float
    minimum_average_net_return_ci_low: float
    minimum_average_excess_return_vs_spy_ci_low: float
    minimum_incremental_net_return_ci_low: float
    minimum_incremental_spy_excess_ci_low: float
    minimum_profit_factor: float = Field(ge=1)
    maximum_drawdown: float = Field(gt=0, le=0.20)
    maximum_negative_period_rate: float = Field(ge=0, le=1)
    required_market_regimes: tuple[str, ...]
    minimum_regime_selected_trades: int = Field(ge=1)
    minimum_regime_average_net_return: float
    minimum_regime_average_excess_return_vs_spy: float
    maximum_process_memory_gib: float = Field(ge=1, le=4)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    hgb_max_iter: int = Field(ge=25, le=2_000)
    hgb_learning_rate: float = Field(gt=0, le=1)
    hgb_l2_regularization: float = Field(ge=0)
    quantiles: tuple[float, ...]
    minimum_calibration_rows: int = Field(ge=100)
    maximum_quantile_calibration_error: float = Field(gt=0, le=0.20)
    minimum_q10_q90_interval_coverage: float = Field(ge=0.50, le=0.80)
    maximum_q10_q90_interval_coverage: float = Field(ge=0.80, le=1)
    maximum_raw_quantile_crossing_rate: float = Field(ge=0, le=0.25)
    maximum_event_log_loss: float = Field(gt=0)
    maximum_event_brier_score: float = Field(gt=0, le=1)
    strategies: dict[str, PrimaryV2StrategyConfig]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != PRIMARY_V2_RESEARCH_SCHEMA:
            raise ValueError("unsupported primary V2 research schema")
        if set(self.strategies) != PRIMARY_V2_STRATEGY_IDS:
            raise ValueError("primary V2 strategy catalog mismatch")
        if self.quantiles != (0.10, 0.50, 0.90):
            raise ValueError("primary V2 quantiles must be exactly 0.10, 0.50, 0.90")
        if (
            self.minimum_q10_q90_interval_coverage
            >= self.maximum_q10_q90_interval_coverage
        ):
            raise ValueError("primary V2 interval coverage bounds are invalid")
        if len(set(self.required_market_regimes)) != len(self.required_market_regimes):
            raise ValueError("required market regimes must be unique")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        swing = self.strategies[SWING_V2_ID]
        intraday = self.strategies[INTRADAY_V2_ID]
        if swing.source_strategy_id != "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1":
            raise ValueError("swing V2 source strategy mismatch")
        if intraday.source_strategy_id != "INTRADAY.VWAP_REVERSION.30M.V1":
            raise ValueError("intraday V2 source strategy mismatch")
        return self

    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_primary_v2_research_config(path: Path) -> PrimaryV2ResearchConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"primary V2 policy is unreadable: {path}") from exc
    return PrimaryV2ResearchConfig.model_validate(raw)
