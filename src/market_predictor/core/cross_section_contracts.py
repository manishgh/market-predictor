"""Point-in-time membership evidence retained even when a model cannot score it."""
from __future__ import annotations

from datetime import UTC, datetime
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator


class CrossSectionMember(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    security_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1, max_length=16)
    sector: str = Field(min_length=1)
    membership_available_at_utc: datetime
    abstention_reason: Literal["live_inputs_incomplete", "sector_peer_floor"] | None = None

    @field_validator("membership_available_at_utc")
    @classmethod
    def aware_timestamp(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("membership availability must be timezone-aware")
        return value.astimezone(UTC)
