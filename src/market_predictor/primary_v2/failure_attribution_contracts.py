"""Frozen contracts for primary V2 failure attribution."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.primary_v2.contracts import INTRADAY_V2_ID, SWING_V2_ID
from market_predictor.v3.errors import DataReadinessError

FAILURE_ATTRIBUTION_SCHEMA = "primary_v2.failure_attribution.v1"
VALIDATION_SCOPES = ("walk_forward", "ticker_holdout")
SWING_DIMENSIONS = (
    "overall",
    "market_regime",
    "sector",
    "volatility_bucket",
)
INTRADAY_DIMENSIONS = (
    "overall",
    "market_regime",
    "sector",
    "volatility_bucket",
    "market_cap_bucket",
    "liquidity_bucket",
    "time_of_day",
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class FailureAttributionStrategyConfig(FrozenModel):
    strategy_id: str
    baseline_candidate_id: str = Field(min_length=10)
    dimensions: tuple[str, ...]
    gross_return_column: str
    net_return_column: str
    spy_excess_return_column: str
    mfe_column: str
    mae_column: str
    period_column: str
    row_id_column: str
    volatility_column: str
    volatility_low_upper_bound: float = Field(gt=0)
    volatility_medium_upper_bound: float = Field(gt=0)
    opening_segment_end_et: str | None = None
    midday_segment_end_et: str | None = None

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        if self.volatility_low_upper_bound >= self.volatility_medium_upper_bound:
            raise ValueError("volatility bucket bounds must increase")
        expected_dimensions = (
            SWING_DIMENSIONS
            if self.strategy_id == SWING_V2_ID
            else INTRADAY_DIMENSIONS
            if self.strategy_id == INTRADAY_V2_ID
            else ()
        )
        if self.dimensions != expected_dimensions:
            raise ValueError("failure-attribution dimensions are not frozen")
        if self.strategy_id == SWING_V2_ID:
            if (
                self.opening_segment_end_et is not None
                or self.midday_segment_end_et is not None
            ):
                raise ValueError("swing attribution cannot define intraday segments")
        elif (
            self.opening_segment_end_et != "11:00"
            or self.midday_segment_end_et != "14:00"
        ):
            raise ValueError("intraday time-of-day boundaries must be 11:00 and 14:00 ET")
        return self


class FailureAttributionConfig(FrozenModel):
    schema_version: str = FAILURE_ATTRIBUTION_SCHEMA
    validation_scopes: tuple[str, ...]
    bootstrap_iterations: int = Field(ge=1_000, le=10_000)
    bootstrap_seed: int
    minimum_rows_per_scope: int = Field(ge=200)
    minimum_sessions_per_scope: int = Field(ge=60)
    minimum_average_net_return: float = Field(ge=0)
    minimum_average_net_return_ci_low: float = Field(ge=0)
    minimum_average_excess_return_vs_spy: float = Field(ge=0)
    minimum_average_excess_return_vs_spy_ci_low: float = Field(ge=0)
    minimum_profit_factor: float = Field(ge=1.05)
    maximum_drawdown: float = Field(gt=0, le=0.20)
    minimum_stamped_round_trip_cost_bps: float = Field(ge=10)
    maximum_process_memory_gib: float = Field(ge=1, le=4)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    swing: FailureAttributionStrategyConfig
    intraday: FailureAttributionStrategyConfig

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != FAILURE_ATTRIBUTION_SCHEMA:
            raise ValueError("unsupported failure-attribution schema")
        if self.validation_scopes != VALIDATION_SCOPES:
            raise ValueError("failure-attribution validation scopes are not frozen")
        if self.swing.strategy_id != SWING_V2_ID:
            raise ValueError("swing failure-attribution strategy ID differs")
        if self.intraday.strategy_id != INTRADAY_V2_ID:
            raise ValueError("intraday failure-attribution strategy ID differs")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        return self

    def strategy(self, strategy_id: str) -> FailureAttributionStrategyConfig:
        if strategy_id == SWING_V2_ID:
            return self.swing
        if strategy_id == INTRADAY_V2_ID:
            return self.intraday
        raise DataReadinessError(
            f"unknown primary V2 failure-attribution strategy: {strategy_id}"
        )

    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_failure_attribution_config(path: Path) -> FailureAttributionConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"failure-attribution policy is unreadable: {path}"
        ) from exc
    return FailureAttributionConfig.model_validate(raw)
