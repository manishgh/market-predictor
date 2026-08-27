from __future__ import annotations

from datetime import date, datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from market_predictor.core.schema import CROSS_SECTIONAL_SCHEMA_VERSION, FrozenContract
from market_predictor.core.symbols import normalized_ticker
from market_predictor.core.time import utc_datetime


class DecisionRowIdentity(FrozenContract):
    ticker: str
    decision_time_utc: datetime
    feature_available_at_utc: datetime
    entry_time_utc: datetime
    session_date_et: date
    decision_group_id: str = Field(min_length=1)
    universe_snapshot_id: str = Field(min_length=1)
    price_feed: str = Field(min_length=1)
    feature_schema_version: str = CROSS_SECTIONAL_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("decision_time_utc", "feature_available_at_utc", "entry_time_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime) -> datetime:
        return utc_datetime(value)

    @field_validator("price_feed")
    @classmethod
    def normalize_feed(cls, value: str) -> str:
        feed = value.strip().lower()
        if feed not in {"sip", "iex", "unknown"}:
            raise ValueError("price_feed must be sip, iex, or unknown")
        return feed

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.feature_available_at_utc > self.decision_time_utc:
            raise ValueError("features cannot become available after the decision")
        if self.entry_time_utc <= self.decision_time_utc:
            raise ValueError("entry must be after the decision")
        return self
