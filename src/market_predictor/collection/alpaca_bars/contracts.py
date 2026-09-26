"""Frozen Alpaca SIP bar collection contracts.

Field names, nesting, defaults and types are part of every recorded policy hash
(`sha256()` over `model_dump(mode="json")`), and the schema strings are written
into existing plans and collections; neither may change.
"""
from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.core.errors import DataReadinessError

REGULAR_BAR_HISTORY_SCHEMA = "edge_rebuild.intraday_history.v1"
REGULAR_BAR_HISTORY_PLAN_SCHEMA = "edge_rebuild.intraday_history_plan.v1"
SESSION_BENCHMARK_SCHEMA = "edge_rebuild.selected_session_benchmark_one_minute.v1"
SESSION_BENCHMARK_PLAN_SCHEMA = "edge_rebuild.selected_session_benchmark_one_minute_plan.v1"
REGULAR_SEGMENT = "regular"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class AlpacaTransportConfig(FrozenModel):
    """Provider identity and transport limits shared by every Alpaca bar collection."""

    schema_version: str
    provider: str
    calendar: str
    required_price_feed: str
    required_adjustment: str
    maximum_expected_rows_per_unit: int = Field(ge=1_000, le=10_000)
    maximum_symbols_per_unit: int = Field(ge=1, le=50)
    collection_workers: int = Field(ge=1, le=4)
    collection_retries: int = Field(ge=1, le=10)
    request_timeout_seconds: float = Field(ge=10, le=300)
    maximum_pages_per_unit: int = Field(ge=1, le=10)
    maximum_failures_before_stop: int = Field(ge=1, le=20)
    intraday_finalization_delay_seconds: int = Field(ge=0, le=300)
    maximum_process_memory_gib: float = Field(ge=1, le=5)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)

    @model_validator(mode="after")
    def validate_transport_contract(self) -> Self:
        if self.provider.strip().lower() != "alpaca":
            raise ValueError("ER1 provider must be Alpaca")
        if self.calendar != "XNYS":
            raise ValueError("ER1 calendar must be XNYS")
        if self.required_price_feed.strip().lower() != "sip":
            raise ValueError("ER1 volume features require SIP")
        if self.required_adjustment.strip().lower() != "all":
            raise ValueError("ER1 adjustment identity must be all")
        if self.maximum_pages_per_unit != 4:
            raise ValueError("ER1 page budget must remain four")
        if self.maximum_symbols_per_unit != 50:
            raise ValueError("ER1 Alpaca units must remain capped at 50 symbols")
        if self.maximum_failures_before_stop != 5:
            raise ValueError("ER1 failure circuit must remain five")
        if self.intraday_finalization_delay_seconds != 60:
            raise ValueError("ER1 finalization delay must remain 60 seconds")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard budget")
        return self

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"),
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
        return hashlib.sha256(payload).hexdigest()


class PointInTimeUniverseConfig(AlpacaTransportConfig):
    """Transport plus the cross-section a whole-universe plan enumerates.

    A plan that walks the point-in-time universe on every session needs a floor
    on how many securities that cross-section may contain, and needs benchmarks
    added to each request. A plan that requests an already-selected handful of
    stock-sessions has neither obligation, so both live here rather than on the
    shared transport contract.
    """

    minimum_session_cross_section: int = Field(ge=300)
    benchmark_tickers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_universe_contract(self) -> Self:
        normalized = self.normalized_benchmarks()
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

    def normalized_benchmarks(self) -> tuple[str, ...]:
        return tuple(
            ticker.strip().upper() for ticker in self.benchmark_tickers
        )


class RegularBarHistoryConfig(PointInTimeUniverseConfig):
    """The frozen regular-session five-minute bar history contract."""

    feature_timeframe: str
    exact_path_timeframe: str
    target_usable_sessions: int = Field(ge=1_000)
    minimum_usable_sessions: int = Field(ge=750)
    feature_warmup_sessions: int = Field(ge=20, le=100)

    @model_validator(mode="after")
    def validate_history_contract(self) -> Self:
        if self.schema_version != REGULAR_BAR_HISTORY_SCHEMA:
            raise ValueError("unsupported ER1A intraday-history schema")
        if self.feature_timeframe != "5Min":
            raise ValueError("ER1A feature discovery must use five-minute bars")
        if self.exact_path_timeframe != "1Min":
            raise ValueError("ER1A exact labels must use one-minute bars")
        if self.minimum_usable_sessions > self.target_usable_sessions:
            raise ValueError("minimum sessions cannot exceed target sessions")
        return self


class SessionBenchmarkConfig(AlpacaTransportConfig):
    """One-minute market and sector paths for selected decision sessions."""

    history_timeframe: str
    session_segments: tuple[str, ...]
    benchmark_tickers: tuple[str, ...]

    @model_validator(mode="after")
    def validate_benchmark_contract(self) -> Self:
        if self.schema_version != SESSION_BENCHMARK_SCHEMA:
            raise ValueError("unsupported selected-session benchmark schema")
        if self.history_timeframe != "1Min":
            raise ValueError("selected-session benchmarks require one-minute bars")
        if tuple(self.session_segments) != (REGULAR_SEGMENT,):
            raise ValueError(
                "selected-session benchmarks cover exactly the regular session"
            )
        normalized = self.normalized_benchmarks()
        required = {
            "SPY",
            "QQQ",
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
        }
        if set(normalized) != required or len(normalized) != len(set(normalized)):
            raise ValueError(
                "benchmark tickers must contain SPY, QQQ, and all eleven "
                "Select Sector SPDR funds exactly once"
            )
        return self

    def normalized_benchmarks(self) -> tuple[str, ...]:
        return tuple(ticker.strip().upper() for ticker in self.benchmark_tickers)


ConfigT = TypeVar("ConfigT", bound=AlpacaTransportConfig)


def load_transport_config(path: Path, model: type[ConfigT], label: str) -> ConfigT:
    """Parse one TOML collection policy strictly into its frozen model."""
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"{label} policy is unreadable: {path}"
        ) from exc
    try:
        return model.model_validate(raw)
    except ValueError as exc:
        raise DataReadinessError(f"{label} policy is invalid: {path}") from exc


def load_regular_bar_history_config(path: Path) -> RegularBarHistoryConfig:
    return load_transport_config(path, RegularBarHistoryConfig, "ER1A intraday-history")


def load_session_benchmark_config(path: Path) -> SessionBenchmarkConfig:
    return load_transport_config(path, SessionBenchmarkConfig, "selected-session benchmark one-minute history")
