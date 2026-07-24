from __future__ import annotations

import hashlib
import json
import re
from datetime import UTC, date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.label_policy import policy_sha256

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PredictionMaturationIntentV1(FrozenContract):
    contract_version: Literal["market_predictor.maturation_intent.v1"] = (
        "market_predictor.maturation_intent.v1"
    )
    maturation_key: str = Field(pattern=SHA256_PATTERN)
    semantic_prediction_id: str = Field(pattern=SHA256_PATTERN)
    snapshot_id: str = Field(pattern=SHA256_PATTERN)
    ticker: str = Field(min_length=1, max_length=16)
    canonical_security_id: str = Field(min_length=1, max_length=128)
    view: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d)$")
    decision_time_utc: datetime
    decision_session_et: date
    decision_group_id: str = Field(min_length=1, max_length=256)
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    serving_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy: dict[str, object]
    primary_benchmark: str = Field(min_length=1, max_length=16)
    market_regime: str = Field(min_length=1, max_length=64)
    sector: str = Field(min_length=1, max_length=128)
    market_cap_bucket: str = Field(min_length=1, max_length=64)
    liquidity_bucket: str = Field(min_length=1, max_length=64)
    price_feed: str = Field(min_length=1, max_length=32)
    probability: float = Field(ge=0.0, le=1.0)
    downside_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_bin: int = Field(ge=0, le=9)
    signal: str = Field(min_length=1, max_length=128)
    actionable: bool
    catalyst_status: str = Field(min_length=1, max_length=32)
    decision_atr: float | None = Field(default=None, gt=0)

    @field_validator("decision_time_utc")
    @classmethod
    def aware_decision_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("decision_time_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("ticker", "primary_benchmark", "price_feed")
    @classmethod
    def normalize_upper(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_identity(self) -> Self:
        if policy_sha256(self.label_policy) != self.label_policy_sha256:
            raise ValueError("label policy hash does not match its payload")
        semantic = semantic_prediction_sha256(self.model_dump(exclude={"maturation_key"}))
        if semantic != self.semantic_prediction_id:
            raise ValueError("semantic prediction identity is invalid")
        if maturation_key_sha256(self.snapshot_id, semantic) != self.maturation_key:
            raise ValueError("maturation key is invalid")
        if self.view == "intraday" and self.decision_atr is None:
            raise ValueError("intraday maturation requires decision ATR")
        return self


class MaturationAttemptV1(FrozenContract):
    contract_version: Literal["market_predictor.maturation_attempt.v1"] = (
        "market_predictor.maturation_attempt.v1"
    )
    attempt_id: str = Field(pattern=SHA256_PATTERN)
    maturation_key: str = Field(pattern=SHA256_PATTERN)
    semantic_prediction_id: str = Field(pattern=SHA256_PATTERN)
    observed_as_of_utc: datetime
    status: Literal["pending", "blocked"]
    reasons: tuple[str, ...] = ()
    missing_intervals: tuple[str, ...] = ()

    @field_validator("observed_as_of_utc")
    @classmethod
    def aware_observation_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("observed_as_of_utc must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_attempt_identity(self) -> Self:
        content = self.model_dump(mode="json", exclude={"attempt_id"})
        if content_sha256(content) != self.attempt_id:
            raise ValueError("maturation attempt identity is invalid")
        return self


class MaturedOutcomeV1(FrozenContract):
    contract_version: Literal["market_predictor.matured_outcome.v1"] = (
        "market_predictor.matured_outcome.v1"
    )
    outcome_id: str = Field(pattern=SHA256_PATTERN)
    maturation_key: str = Field(pattern=SHA256_PATTERN)
    semantic_prediction_id: str = Field(pattern=SHA256_PATTERN)
    snapshot_id: str = Field(pattern=SHA256_PATTERN)
    ticker: str
    view: Literal["swing", "intraday"]
    horizon: str
    entry_time_utc: datetime
    exit_time_utc: datetime
    label_available_at_utc: datetime
    matured_at_utc: datetime
    entry_price: float = Field(gt=0)
    exit_price: float = Field(gt=0)
    gross_return: float
    net_return: float
    mfe: float
    mae: float
    path_outcome: Literal["positive", "negative", "target_first", "stop_first", "timeout"]
    opportunity_target: int = Field(ge=0, le=1)
    downside_target: int | None = Field(default=None, ge=0, le=1)
    spy_return: float
    qqq_return: float
    sector_return: float
    excess_return_vs_spy: float
    excess_return_vs_qqq: float
    excess_return_vs_sector: float
    evidence_sha256: str = Field(pattern=SHA256_PATTERN)

    @field_validator(
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "matured_at_utc",
    )
    @classmethod
    def aware_outcome_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("outcome timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_times(self) -> Self:
        if self.exit_time_utc <= self.entry_time_utc:
            raise ValueError("outcome exit must follow entry")
        if self.matured_at_utc != self.label_available_at_utc:
            raise ValueError("matured_at_utc must equal deterministic label availability")
        content = self.model_dump(mode="json", exclude={"outcome_id"})
        if content_sha256(content) != self.outcome_id:
            raise ValueError("matured outcome identity is invalid")
        return self


def semantic_prediction_sha256(intent_without_key: dict[str, object]) -> str:
    content = dict(intent_without_key)
    content.pop("semantic_prediction_id", None)
    content.pop("snapshot_id", None)
    canonical = json.dumps(
        content,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def maturation_key_sha256(snapshot_id: str, semantic_prediction_id: str) -> str:
    if not re.fullmatch(SHA256_PATTERN, snapshot_id) or not re.fullmatch(
        SHA256_PATTERN,
        semantic_prediction_id,
    ):
        raise ValueError("maturation identity inputs must be SHA-256 values")
    return hashlib.sha256(
        f"{snapshot_id}:{semantic_prediction_id}".encode("ascii")
    ).hexdigest()


def content_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")
