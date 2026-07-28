"""Frozen ER1A historical intraday acquisition contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.v3.errors import DataReadinessError

INTRADAY_HISTORY_SCHEMA = "edge_rebuild.intraday_history.v1"
INTRADAY_HISTORY_PLAN_SCHEMA = "edge_rebuild.intraday_history_plan.v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntradayHistoryConfig(FrozenModel):
    schema_version: str
    provider: str
    calendar: str
    required_price_feed: str
    required_adjustment: str
    feature_timeframe: str
    exact_path_timeframe: str
    target_usable_sessions: int = Field(ge=1_000)
    minimum_usable_sessions: int = Field(ge=750)
    feature_warmup_sessions: int = Field(ge=20, le=100)
    minimum_session_cross_section: int = Field(ge=300)
    maximum_expected_rows_per_unit: int = Field(ge=1_000, le=10_000)
    maximum_symbols_per_unit: int = Field(ge=1, le=500)
    collection_workers: int = Field(ge=1, le=4)
    collection_retries: int = Field(ge=1, le=10)
    request_timeout_seconds: float = Field(ge=10, le=300)
    maximum_process_memory_gib: float = Field(ge=1, le=4)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    benchmark_tickers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != INTRADAY_HISTORY_SCHEMA:
            raise ValueError("unsupported ER1A intraday-history schema")
        if self.provider.strip().lower() != "alpaca":
            raise ValueError("ER1A provider must be Alpaca")
        if self.calendar != "XNYS":
            raise ValueError("ER1A calendar must be XNYS")
        if self.required_price_feed.strip().lower() != "sip":
            raise ValueError("ER1A volume features require SIP")
        if self.required_adjustment.strip().lower() != "all":
            raise ValueError("ER1A adjustment identity must be all")
        if self.feature_timeframe != "5Min":
            raise ValueError("ER1A feature discovery must use five-minute bars")
        if self.exact_path_timeframe != "1Min":
            raise ValueError("ER1A exact labels must use one-minute bars")
        if self.minimum_usable_sessions > self.target_usable_sessions:
            raise ValueError("minimum sessions cannot exceed target sessions")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard budget")
        normalized = tuple(
            ticker.strip().upper() for ticker in self.benchmark_tickers
        )
        if (
            "SPY" not in normalized
            or "QQQ" not in normalized
            or len(normalized) != len(set(normalized))
            or any(not ticker for ticker in normalized)
        ):
            raise ValueError(
                "benchmark tickers must be unique and include SPY and QQQ"
            )
        return self

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_intraday_history_config(path: Path) -> IntradayHistoryConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"ER1A intraday-history policy is unreadable: {path}"
        ) from exc
    try:
        return IntradayHistoryConfig.model_validate(raw)
    except ValueError as exc:
        raise DataReadinessError(
            f"ER1A intraday-history policy is invalid: {path}"
        ) from exc

