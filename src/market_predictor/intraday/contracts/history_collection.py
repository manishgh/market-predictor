"""Intraday-research acquisition contracts, retired with the intraday package.

The retained Alpaca bar contracts live in `market_predictor.collection.alpaca_bars.contracts`.
"""
from __future__ import annotations

import tomllib
from collections.abc import Callable
from datetime import date, time
from pathlib import Path
from typing import Self

from pydantic import model_validator

from market_predictor.collection.alpaca_bars.contracts import (
    REGULAR_BAR_HISTORY_SCHEMA,
    REGULAR_SEGMENT,
    SESSION_BENCHMARK_SCHEMA,
    AlpacaTransportConfig,
    PointInTimeUniverseConfig,
    load_regular_bar_history_config,
    load_session_benchmark_config,
    load_transport_config,
)
from market_predictor.core.errors import DataReadinessError

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
BROAD_INTRADAY_HISTORY_SCHEMA = "edge_rebuild.broad_intraday_history.v1"
BROAD_INTRADAY_HISTORY_PLAN_SCHEMA = (
    "edge_rebuild.broad_intraday_history_plan.v1"
)
PREMARKET_SEGMENT = "premarket"
POSTMARKET_SEGMENT = "postmarket"
EXTENDED_SEGMENTS = (PREMARKET_SEGMENT, POSTMARKET_SEGMENT)


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


class SelectedSessionHistoryConfig(AlpacaTransportConfig):
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


class SelectedSessionOneMinuteConfig(AlpacaTransportConfig):
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


class BroadIntradayHistoryConfig(AlpacaTransportConfig):
    """Bounded five-minute acquisition policy for the broad research universe."""

    history_timeframe: str
    session_segments: tuple[str, ...]
    first_session: date
    last_session: date
    explicit_fund_exclusions: tuple[str, ...]

    @model_validator(mode="after")
    def validate_broad_history_contract(self) -> Self:
        if self.schema_version != BROAD_INTRADAY_HISTORY_SCHEMA:
            raise ValueError("unsupported broad intraday-history schema")
        if self.history_timeframe != "5Min":
            raise ValueError("broad intraday history must use five-minute bars")
        if tuple(self.session_segments) != (REGULAR_SEGMENT,):
            raise ValueError("broad intraday history is regular-session only")
        if self.first_session > self.last_session:
            raise ValueError("broad intraday history window is reversed")
        exclusions = self.normalized_fund_exclusions()
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
        if not required.issubset(exclusions) or len(exclusions) != len(set(exclusions)):
            raise ValueError(
                "fund exclusions must be unique and include broad and sector ETFs"
            )
        return self

    def normalized_fund_exclusions(self) -> tuple[str, ...]:
        return tuple(value.strip().upper() for value in self.explicit_fund_exclusions)


def _exchange_clock_time(value: str, field: str) -> time:
    try:
        hour, minute = (int(part) for part in value.strip().split(":"))
        return time(hour=hour, minute=minute)
    except ValueError as exc:
        raise ValueError(f"{field} must be an exact HH:MM exchange time") from exc


def load_extended_session_context_config(
    path: Path,
) -> ExtendedSessionContextConfig:
    return load_transport_config(
        path,
        ExtendedSessionContextConfig,
        "ER1B extended-session context",
    )


def load_selected_session_history_config(
    path: Path,
) -> SelectedSessionHistoryConfig:
    return load_transport_config(
        path,
        SelectedSessionHistoryConfig,
        "selected-session history",
    )


def load_selected_session_one_minute_config(
    path: Path,
) -> SelectedSessionOneMinuteConfig:
    return load_transport_config(
        path,
        SelectedSessionOneMinuteConfig,
        "selected-session one-minute history",
    )


def load_broad_intraday_history_config(path: Path) -> BroadIntradayHistoryConfig:
    return load_transport_config(
        path,
        BroadIntradayHistoryConfig,
        "broad intraday history",
    )


_TRANSPORT_CONFIG_LOADERS: dict[str, Callable[[Path], AlpacaTransportConfig]] = {
    REGULAR_BAR_HISTORY_SCHEMA: load_regular_bar_history_config,
    EXTENDED_CONTEXT_SCHEMA: load_extended_session_context_config,
    SELECTED_SESSION_HISTORY_SCHEMA: load_selected_session_history_config,
    SELECTED_SESSION_ONE_MINUTE_SCHEMA: load_selected_session_one_minute_config,
    SESSION_BENCHMARK_SCHEMA: load_session_benchmark_config,
    BROAD_INTRADAY_HISTORY_SCHEMA: load_broad_intraday_history_config,
}


def load_collection_transport_config(path: Path) -> AlpacaTransportConfig:
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
