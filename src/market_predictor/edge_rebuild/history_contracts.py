"""Frozen ER1A/ER1B historical intraday acquisition contracts."""

from __future__ import annotations

import hashlib
import json
import tomllib
from collections.abc import Callable
from datetime import time
from pathlib import Path
from typing import Self, TypeVar

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.v3.errors import DataReadinessError

INTRADAY_HISTORY_SCHEMA = "edge_rebuild.intraday_history.v1"
INTRADAY_HISTORY_PLAN_SCHEMA = "edge_rebuild.intraday_history_plan.v1"
EXTENDED_CONTEXT_SCHEMA = "edge_rebuild.extended_session_context.v1"
EXTENDED_CONTEXT_PLAN_SCHEMA = "edge_rebuild.extended_session_context_plan.v1"
SELECTED_SESSION_HISTORY_SCHEMA = "edge_rebuild.selected_session_history.v1"
SELECTED_SESSION_PLAN_SCHEMA = "edge_rebuild.selected_session_history_plan.v1"
SELECTED_SESSION_ONE_MINUTE_SCHEMA = (
    "edge_rebuild.selected_session_one_minute.v1"
)
SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA = (
    "edge_rebuild.selected_session_one_minute_plan.v1"
)
REGULAR_SEGMENT = "regular"
PREMARKET_SEGMENT = "premarket"
POSTMARKET_SEGMENT = "postmarket"
EXTENDED_SEGMENTS = (PREMARKET_SEGMENT, POSTMARKET_SEGMENT)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class IntradayTransportConfig(FrozenModel):
    """Provider identity and transport limits shared by every ER1 collection."""

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
    maximum_process_memory_gib: float = Field(ge=1, le=4)
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


class PointInTimeUniverseConfig(IntradayTransportConfig):
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


class IntradayHistoryConfig(PointInTimeUniverseConfig):
    """The frozen ER1A regular-session history contract."""

    feature_timeframe: str
    exact_path_timeframe: str
    target_usable_sessions: int = Field(ge=1_000)
    minimum_usable_sessions: int = Field(ge=750)
    feature_warmup_sessions: int = Field(ge=20, le=100)

    @model_validator(mode="after")
    def validate_history_contract(self) -> Self:
        if self.schema_version != INTRADAY_HISTORY_SCHEMA:
            raise ValueError("unsupported ER1A intraday-history schema")
        if self.feature_timeframe != "5Min":
            raise ValueError("ER1A feature discovery must use five-minute bars")
        if self.exact_path_timeframe != "1Min":
            raise ValueError("ER1A exact labels must use one-minute bars")
        if self.minimum_usable_sessions > self.target_usable_sessions:
            raise ValueError("minimum sessions cannot exceed target sessions")
        return self


class ExtendedSessionContextConfig(PointInTimeUniverseConfig):
    """The frozen ER1B extended-session context contract.

    Extended-hours bars are a separate context layer. They are never merged
    into the regular-session store, so thin pre/post-market volume cannot
    reach a regular-session VWAP, EMA, ATR, or relative-volume denominator.
    """

    context_timeframe: str
    premarket_start_et: str
    postmarket_end_et: str
    session_segments: tuple[str, ...]

    @model_validator(mode="after")
    def validate_context_contract(self) -> Self:
        if self.schema_version != EXTENDED_CONTEXT_SCHEMA:
            raise ValueError("unsupported ER1B extended-context schema")
        if self.context_timeframe != "5Min":
            raise ValueError("ER1B context must use five-minute bars")
        if tuple(self.session_segments) != EXTENDED_SEGMENTS:
            raise ValueError(
                "ER1B segments must remain exactly premarket and postmarket"
            )
        if self.premarket_start_time() >= self.postmarket_end_time():
            raise ValueError("premarket start must precede postmarket end")
        return self

    def premarket_start_time(self) -> time:
        return _exchange_clock_time(self.premarket_start_et, "premarket_start_et")

    def postmarket_end_time(self) -> time:
        return _exchange_clock_time(self.postmarket_end_et, "postmarket_end_et")


class SelectedSessionHistoryConfig(IntradayTransportConfig):
    """The frozen contract for bars covering only selected stock-sessions.

    The two whole-universe layers request every point-in-time member on every
    session. This one requests only the stock-sessions that already passed the
    frozen two-layer screen, which is roughly one request unit per session
    rather than eleven. It therefore carries no cross-section floor.

    It also adds no benchmarks. Benchmark and sector-ETF bars already span the
    whole research window in the regular-session corpus, so re-requesting them
    here would spend units on bars that exist and would collide with them on
    materialization.
    """

    history_timeframe: str
    session_segments: tuple[str, ...]

    @model_validator(mode="after")
    def validate_selected_session_contract(self) -> Self:
        if self.schema_version != SELECTED_SESSION_HISTORY_SCHEMA:
            raise ValueError("unsupported selected-session history schema")
        if self.history_timeframe != "5Min":
            raise ValueError("selected-session history must use five-minute bars")
        if tuple(self.session_segments) != (REGULAR_SEGMENT,):
            raise ValueError(
                "selected-session history covers exactly the regular session"
            )
        return self


class SelectedSessionOneMinuteConfig(IntradayTransportConfig):
    """Exact-path and volume-bar input for the screened stock-sessions."""

    history_timeframe: str
    session_segments: tuple[str, ...]

    @model_validator(mode="after")
    def validate_selected_session_contract(self) -> Self:
        if self.schema_version != SELECTED_SESSION_ONE_MINUTE_SCHEMA:
            raise ValueError("unsupported selected-session one-minute schema")
        if self.history_timeframe != "1Min":
            raise ValueError("selected-session exact paths require one-minute bars")
        if tuple(self.session_segments) != (REGULAR_SEGMENT,):
            raise ValueError(
                "selected-session one-minute history covers exactly the regular session"
            )
        return self


ConfigT = TypeVar("ConfigT", bound=IntradayTransportConfig)


def _exchange_clock_time(value: str, field: str) -> time:
    try:
        hour, minute = (int(part) for part in value.strip().split(":"))
        return time(hour=hour, minute=minute)
    except ValueError as exc:
        raise ValueError(f"{field} must be an exact HH:MM exchange time") from exc


def _load_config(path: Path, model: type[ConfigT], label: str) -> ConfigT:
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


def load_intraday_history_config(path: Path) -> IntradayHistoryConfig:
    return _load_config(path, IntradayHistoryConfig, "ER1A intraday-history")


def load_extended_session_context_config(
    path: Path,
) -> ExtendedSessionContextConfig:
    return _load_config(
        path,
        ExtendedSessionContextConfig,
        "ER1B extended-session context",
    )


def load_selected_session_history_config(
    path: Path,
) -> SelectedSessionHistoryConfig:
    return _load_config(
        path,
        SelectedSessionHistoryConfig,
        "selected-session history",
    )


def load_selected_session_one_minute_config(
    path: Path,
) -> SelectedSessionOneMinuteConfig:
    return _load_config(
        path,
        SelectedSessionOneMinuteConfig,
        "selected-session one-minute history",
    )


_TRANSPORT_CONFIG_LOADERS: dict[str, Callable[[Path], IntradayTransportConfig]] = {
    INTRADAY_HISTORY_SCHEMA: load_intraday_history_config,
    EXTENDED_CONTEXT_SCHEMA: load_extended_session_context_config,
    SELECTED_SESSION_HISTORY_SCHEMA: load_selected_session_history_config,
    SELECTED_SESSION_ONE_MINUTE_SCHEMA: load_selected_session_one_minute_config,
}


def load_collection_transport_config(path: Path) -> IntradayTransportConfig:
    """Load whichever collection policy the file itself declares.

    The collector is generic across plan layers, so the layer must be named
    once. Taking it from the policy's own `schema_version` removes the second
    place an operator could name it, and with it the chance of collecting one
    layer under another layer's transport identity.
    """

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"collection policy is unreadable: {path}") from exc
    loader = _TRANSPORT_CONFIG_LOADERS.get(str(raw.get("schema_version", "")))
    if loader is None:
        raise DataReadinessError(
            f"collection policy declares no known schema_version: {path}"
        )
    return loader(path)
