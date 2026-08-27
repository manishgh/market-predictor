from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from market_predictor.core.schema import CROSS_SECTIONAL_SCHEMA_VERSION, FrozenContract
from market_predictor.core.symbols import normalized_ticker
from market_predictor.core.time import utc_datetime


class SourceAvailability(FrozenContract):
    ticker: str
    source_family: str = Field(min_length=1)
    available: bool
    row_count: int = Field(ge=0)
    first_available_at_utc: datetime | None = None
    last_available_at_utc: datetime | None = None
    collected_at_utc: datetime
    schema_version: str = CROSS_SECTIONAL_SCHEMA_VERSION

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
        if self.last_available_at_utc and self.last_available_at_utc > self.collected_at_utc:
            raise ValueError("source coverage cannot extend beyond collection time")
        return self
