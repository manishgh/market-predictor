from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract


def utc_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


def normalized_ticker(value: str) -> str:
    ticker = value.strip().upper().replace("/", ".")
    if not ticker or len(ticker) > 16 or not all(character.isalnum() or character in ".-" for character in ticker):
        raise ValueError("ticker must be a valid normalized US-listed symbol")
    return ticker


class UniverseMembership(FrozenContract):
    ticker: str
    effective_from_utc: datetime
    effective_to_utc: datetime | None = None
    sector: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    market_cap_bucket: str = Field(min_length=1)
    liquidity_bucket: str = Field(min_length=1)
    primary_benchmark: str
    universe_snapshot_id: str = Field(min_length=1)
    schema_version: str = ML_V3_SCHEMA_VERSION

    @field_validator("ticker", "primary_benchmark")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("effective_from_utc", "effective_to_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else utc_datetime(value)

    @model_validator(mode="after")
    def validate_effective_window(self) -> Self:
        if self.effective_to_utc is not None and self.effective_to_utc <= self.effective_from_utc:
            raise ValueError("effective_to_utc must be later than effective_from_utc")
        return self

    def contains(self, timestamp: datetime) -> bool:
        moment = utc_datetime(timestamp)
        return self.effective_from_utc <= moment and (self.effective_to_utc is None or moment < self.effective_to_utc)


class DecisionRowIdentity(FrozenContract):
    ticker: str
    decision_time_utc: datetime
    feature_available_at_utc: datetime
    entry_time_utc: datetime
    session_date_et: date
    decision_group_id: str = Field(min_length=1)
    universe_snapshot_id: str = Field(min_length=1)
    price_feed: str = Field(min_length=1)
    feature_schema_version: str = ML_V3_SCHEMA_VERSION

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


class SourceAvailability(FrozenContract):
    ticker: str
    source_family: str = Field(min_length=1)
    available: bool
    row_count: int = Field(ge=0)
    first_available_at_utc: datetime | None = None
    last_available_at_utc: datetime | None = None
    collected_at_utc: datetime
    schema_version: str = ML_V3_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("first_available_at_utc", "last_available_at_utc", "collected_at_utc")
    @classmethod
    def validate_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else utc_datetime(value)

    @model_validator(mode="after")
    def validate_coverage(self) -> Self:
        if self.available != (self.row_count > 0):
            raise ValueError("available must agree with row_count")
        if self.row_count == 0 and (self.first_available_at_utc is not None or self.last_available_at_utc is not None):
            raise ValueError("empty sources cannot declare coverage timestamps")
        if self.row_count > 0 and (self.first_available_at_utc is None or self.last_available_at_utc is None):
            raise ValueError("available sources require first and last timestamps")
        if self.first_available_at_utc and self.last_available_at_utc and self.first_available_at_utc > self.last_available_at_utc:
            raise ValueError("source coverage timestamps are reversed")
        return self
