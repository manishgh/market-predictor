from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.v3.contracts import normalized_ticker

CANONICAL_SCHEMA_VERSION = "market_data.v1"
AvailabilityPolicy = Literal["observed", "market_interval_close", "provider_publication_proxy"]
SourceCollectionStatus = Literal["observed", "observed_empty", "partial", "failed", "disabled", "not_collected"]


class CanonicalContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None or value.utcoffset() is None:
        raise ValueError("timestamp must include a timezone")
    return value.astimezone(UTC)


class CanonicalBar(CanonicalContract):
    ticker: str
    timeframe: str = Field(min_length=1)
    bar_start_utc: datetime
    bar_end_utc: datetime
    available_at_utc: datetime
    ingested_at_utc: datetime
    open: float = Field(gt=0)
    high: float = Field(gt=0)
    low: float = Field(gt=0)
    close: float = Field(gt=0)
    volume: float = Field(ge=0)
    source: str = Field(min_length=1)
    price_feed: Literal["sip", "iex", "unknown"]
    adjustment: str = Field(min_length=1)
    availability_policy: AvailabilityPolicy
    schema_version: str = CANONICAL_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("timeframe")
    @classmethod
    def normalize_timeframe(cls, value: str) -> str:
        normalized = value.strip().lower()
        if normalized not in {"1m", "5m", "1h", "1d"}:
            raise ValueError("timeframe must be 1m, 5m, 1h, or 1d")
        return normalized

    @field_validator("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)

    @field_validator("source", "adjustment")
    @classmethod
    def normalize_nonempty(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def validate_timing_and_prices(self) -> Self:
        if self.bar_end_utc <= self.bar_start_utc:
            raise ValueError("bar_end_utc must be later than bar_start_utc")
        if self.available_at_utc < self.bar_end_utc:
            raise ValueError("a bar cannot be available before its interval ends")
        if self.high < max(self.open, self.close, self.low):
            raise ValueError("bar high is inconsistent with OHLC")
        if self.low > min(self.open, self.close, self.high):
            raise ValueError("bar low is inconsistent with OHLC")
        return self


class CanonicalEvent(CanonicalContract):
    event_id: str = Field(min_length=16)
    ticker: str
    source_family: str = Field(min_length=1)
    source: str = Field(min_length=1)
    published_at_utc: datetime
    provider_updated_at_utc: datetime | None = None
    first_seen_at_utc: datetime
    available_at_utc: datetime
    sentiment_scored_at_utc: datetime | None = None
    feature_available_at_utc: datetime
    title: str = Field(min_length=1)
    url: str = ""
    summary: str = ""
    text: str = ""
    sentiment_numeric: float | None = Field(default=None, ge=-1, le=1)
    relevance: float | None = Field(default=None, ge=0)
    availability_policy: AvailabilityPolicy
    raw_sha256: str = Field(min_length=64, max_length=64)
    schema_version: str = CANONICAL_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator(
        "published_at_utc",
        "provider_updated_at_utc",
        "first_seen_at_utc",
        "available_at_utc",
        "sentiment_scored_at_utc",
        "feature_available_at_utc",
    )
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @field_validator("source_family", "source")
    @classmethod
    def normalize_source(cls, value: str) -> str:
        return value.strip().lower()

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        content_timestamp = max(
            self.published_at_utc,
            self.provider_updated_at_utc or self.published_at_utc,
        )
        if self.available_at_utc < content_timestamp:
            raise ValueError("event availability precedes provider content timestamp")
        if self.availability_policy == "observed" and self.available_at_utc < self.first_seen_at_utc:
            raise ValueError("observed event availability precedes first_seen_at_utc")
        if self.sentiment_numeric is not None and self.sentiment_scored_at_utc is None:
            raise ValueError("scored sentiment requires sentiment_scored_at_utc")
        if self.sentiment_scored_at_utc is not None and self.sentiment_scored_at_utc < self.available_at_utc:
            raise ValueError("sentiment cannot be scored before event availability")
        expected_feature_time = self.sentiment_scored_at_utc or self.available_at_utc
        if self.feature_available_at_utc != expected_feature_time:
            raise ValueError("feature_available_at_utc must include sentiment scoring latency")
        return self


class CanonicalFundamentalFact(CanonicalContract):
    fact_id: str = Field(min_length=16)
    ticker: str
    metric: str = Field(min_length=1)
    value: float
    unit: str = Field(min_length=1)
    fiscal_period_end: date
    filed_at_utc: datetime
    first_seen_at_utc: datetime
    available_at_utc: datetime
    accession_number: str = Field(min_length=1)
    form: str = Field(min_length=1)
    source: str = "sec"
    availability_policy: AvailabilityPolicy
    schema_version: str = CANONICAL_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("filed_at_utc", "first_seen_at_utc", "available_at_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_availability(self) -> Self:
        if self.available_at_utc < self.filed_at_utc:
            raise ValueError("fundamental fact availability precedes filing")
        if self.availability_policy == "observed" and self.available_at_utc < self.first_seen_at_utc:
            raise ValueError("observed fact availability precedes first_seen_at_utc")
        return self


class CanonicalUniverseMembership(CanonicalContract):
    ticker: str
    effective_from_utc: datetime
    effective_to_utc: datetime | None = None
    available_at_utc: datetime
    sector: str = Field(min_length=1)
    industry: str = Field(min_length=1)
    market_cap_bucket: str = Field(min_length=1)
    liquidity_bucket: str = Field(min_length=1)
    primary_benchmark: str
    universe_snapshot_id: str = Field(min_length=1)
    source: str = Field(min_length=1)
    availability_policy: AvailabilityPolicy
    schema_version: str = CANONICAL_SCHEMA_VERSION

    @field_validator("ticker", "primary_benchmark")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("effective_from_utc", "effective_to_utc", "available_at_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime | None) -> datetime | None:
        return None if value is None else _utc(value)

    @field_validator(
        "sector",
        "industry",
        "market_cap_bucket",
        "liquidity_bucket",
        "universe_snapshot_id",
        "source",
    )
    @classmethod
    def normalize_text(cls, value: str) -> str:
        return value.strip()

    @model_validator(mode="after")
    def validate_membership(self) -> Self:
        if self.effective_to_utc is not None and self.effective_to_utc <= self.effective_from_utc:
            raise ValueError("effective_to_utc must be later than effective_from_utc")
        return self


class SourceCollection(CanonicalContract):
    collection_id: str = Field(min_length=8)
    ticker: str
    source_family: str = Field(min_length=1)
    requested_start_utc: datetime
    requested_end_utc: datetime
    started_at_utc: datetime
    completed_at_utc: datetime
    status: SourceCollectionStatus
    row_count: int = Field(ge=0)
    error_type: str | None = None
    schema_version: str = CANONICAL_SCHEMA_VERSION

    @field_validator("ticker")
    @classmethod
    def validate_ticker(cls, value: str) -> str:
        return normalized_ticker(value)

    @field_validator("source_family")
    @classmethod
    def normalize_source_family(cls, value: str) -> str:
        return value.strip().lower()

    @field_validator("requested_start_utc", "requested_end_utc", "started_at_utc", "completed_at_utc")
    @classmethod
    def normalize_timestamp(cls, value: datetime) -> datetime:
        return _utc(value)

    @model_validator(mode="after")
    def validate_collection(self) -> Self:
        if self.requested_end_utc < self.requested_start_utc:
            raise ValueError("source collection request window is reversed")
        if self.completed_at_utc < self.started_at_utc:
            raise ValueError("source collection completion precedes start")
        if self.status in {"observed", "partial"} and self.row_count == 0:
            raise ValueError("observed or partial source collection requires rows")
        if self.status not in {"observed", "partial"} and self.row_count != 0:
            raise ValueError("only observed or partial source collections can have rows")
        if self.status in {"failed", "partial"} and not self.error_type:
            raise ValueError("failed or partial source collection requires error_type")
        if self.status not in {"failed", "partial"} and self.error_type:
            raise ValueError("error_type is only valid for failed or partial source collections")
        return self
