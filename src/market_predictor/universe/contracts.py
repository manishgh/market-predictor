from __future__ import annotations

from datetime import datetime
from typing import Self

from pydantic import Field, field_validator, model_validator

from market_predictor.core.schema import CROSS_SECTIONAL_SCHEMA_VERSION, FrozenContract
from market_predictor.core.symbols import normalized_ticker
from market_predictor.core.time import utc_datetime


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
    schema_version: str = CROSS_SECTIONAL_SCHEMA_VERSION

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
