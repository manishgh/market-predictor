"""Frozen contracts for the ER1 edge-rebuild readiness audit."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.v3.errors import DataReadinessError

READINESS_SCHEMA = "edge_rebuild.readiness.v2"
SWING_STRATEGY_ID = "SWING.SECTOR_RESIDUAL_MOMENTUM.10D.V1"
INTRADAY_STRATEGY_ID = "INTRADAY.VWAP_EXHAUSTION_REVERSAL.30M.V1"
INTRADAY_PROXY_STRATEGY_ID = "INTRADAY.VWAP_REVERSION.30M.V1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingReadinessConfig(FrozenModel):
    strategy_id: str
    proposed_horizon_sessions: int = Field(ge=2, le=20)
    non_overlapping_phases: int = Field(ge=2, le=20)
    minimum_valid_sessions: int = Field(ge=1_000)
    minimum_sessions_per_phase: int = Field(ge=60)
    minimum_daily_warmup_bars: int = Field(ge=250)

    @model_validator(mode="after")
    def validate_swing(self) -> Self:
        if self.strategy_id != SWING_STRATEGY_ID:
            raise ValueError("swing readiness strategy ID is not frozen")
        if self.proposed_horizon_sessions != 10:
            raise ValueError("swing readiness horizon must be ten sessions")
        if self.non_overlapping_phases != self.proposed_horizon_sessions:
            raise ValueError("swing phase count must equal its overlapping horizon")
        return self


class IntradayReadinessConfig(FrozenModel):
    strategy_id: str
    proxy_strategy_id: str
    proposed_horizon_minutes: int = Field(ge=5, le=120)
    minimum_causal_sessions: int = Field(ge=750)
    required_purged_folds: int = Field(ge=4, le=10)
    minimum_test_sessions_per_fold: int = Field(ge=60)
    required_timeframe: str

    @model_validator(mode="after")
    def validate_intraday(self) -> Self:
        if self.strategy_id != INTRADAY_STRATEGY_ID:
            raise ValueError("intraday readiness strategy ID is not frozen")
        if self.proxy_strategy_id != INTRADAY_PROXY_STRATEGY_ID:
            raise ValueError("intraday readiness proxy strategy ID is not frozen")
        if self.proposed_horizon_minutes != 30:
            raise ValueError("intraday readiness horizon must be thirty minutes")
        if self.required_timeframe != "1Min":
            raise ValueError("intraday exact-path source must use one-minute bars")
        if (
            self.required_purged_folds
            * self.minimum_test_sessions_per_fold
            > self.minimum_causal_sessions
        ):
            raise ValueError("intraday fold capacity exceeds minimum session history")
        return self


class CatalystReadinessConfig(FrozenModel):
    required_source_families: tuple[str, ...]
    required_relation_channels: tuple[str, ...]
    required_fields: tuple[str, ...]
    research_availability_policies: tuple[str, ...]
    promotion_availability_policies: tuple[str, ...]

    @model_validator(mode="after")
    def validate_catalyst(self) -> Self:
        for name, values in (
            ("required_source_families", self.required_source_families),
            ("required_relation_channels", self.required_relation_channels),
            ("required_fields", self.required_fields),
            ("research_availability_policies", self.research_availability_policies),
            ("promotion_availability_policies", self.promotion_availability_policies),
        ):
            normalized = tuple(value.strip().lower() for value in values)
            if not normalized or any(not value for value in normalized):
                raise ValueError(f"{name} must contain non-empty values")
            if len(normalized) != len(set(normalized)):
                raise ValueError(f"{name} must contain unique values")
        if "first_observed_at_utc" not in self.required_fields:
            raise ValueError("catalyst readiness must require first-observed evidence")
        return self


class EdgeRebuildReadinessConfig(FrozenModel):
    schema_version: str
    required_price_feed: str
    required_adjustment: str
    target_history_sessions: int = Field(ge=1_000)
    maximum_process_memory_gib: float = Field(ge=1, le=4)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    swing: SwingReadinessConfig
    intraday: IntradayReadinessConfig
    catalyst: CatalystReadinessConfig

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != READINESS_SCHEMA:
            raise ValueError("unsupported edge-rebuild readiness schema")
        if self.required_price_feed.strip().lower() != "sip":
            raise ValueError("volume-dependent readiness requires SIP")
        if self.required_adjustment.strip().lower() != "all":
            raise ValueError("readiness adjustment identity must be all")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        return self

    def sha256(self) -> str:
        encoded = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_edge_rebuild_readiness_config(
    path: Path,
) -> EdgeRebuildReadinessConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"edge-rebuild readiness policy is unreadable: {path}"
        ) from exc
    try:
        return EdgeRebuildReadinessConfig.model_validate(raw)
    except ValueError as exc:
        raise DataReadinessError(
            f"edge-rebuild readiness policy is invalid: {path}"
        ) from exc
