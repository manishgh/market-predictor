from __future__ import annotations

import hashlib
import json
import math
import re
from datetime import UTC, date, datetime
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    EXECUTION_POLICY_SHA256,
    round_trip_cost_bps,
)
from market_predictor.label_policy import policy_sha256
from market_predictor.modeling.prediction_selection import (
    parse_prediction_policy,
    parse_swing_prediction_policy,
)

SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FrozenContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)


class PredictionMonitoringObservationV1(FrozenContract):
    contract_version: Literal["market_predictor.prediction_observation.v1"] = (
        "market_predictor.prediction_observation.v1"
    )
    observation_id: str = Field(pattern=SHA256_PATTERN)
    semantic_prediction_id: str = Field(pattern=SHA256_PATTERN)
    snapshot_id: str = Field(pattern=SHA256_PATTERN)
    ticker: str = Field(min_length=1, max_length=16)
    view: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d|b)$")
    decision_time_utc: datetime
    decision_session_et: date
    decision_group_id: str = Field(min_length=1, max_length=256)
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    market_regime: str = Field(min_length=1, max_length=64)
    sector: str = Field(min_length=1, max_length=128)
    market_cap_bucket: str = Field(min_length=1, max_length=64)
    liquidity_bucket: str = Field(min_length=1, max_length=64)
    probability: float | None = Field(default=None, ge=0.0, le=1.0)
    downside_probability: float | None = Field(default=None, ge=0.0, le=1.0)
    calibration_bin: int | None = Field(default=None, ge=0, le=9)
    signal: str = Field(min_length=1, max_length=128)
    rank: int | None = Field(default=None, ge=1)
    selection_eligible: bool
    selected_for_policy: bool
    actionable: bool
    readiness_status: Literal["valid", "warn", "invalid"]
    catalyst_status: str = Field(min_length=1, max_length=32)
    maturation_key: str | None = Field(default=None, pattern=SHA256_PATTERN)

    @field_validator("decision_time_utc")
    @classmethod
    def aware_observation_time(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("decision_time_utc must be timezone-aware")
        return value.astimezone(UTC)

    @field_validator("ticker")
    @classmethod
    def normalize_observation_ticker(cls, value: str) -> str:
        return value.strip().upper()

    @model_validator(mode="after")
    def validate_observation(self) -> Self:
        if self.selected_for_policy and not self.selection_eligible:
            raise ValueError("selected observation must be eligible")
        if self.actionable != (
            self.readiness_status == "valid"
            and self.selected_for_policy
            and self.signal != "not_ready"
        ):
            raise ValueError("observation actionability is inconsistent")
        if self.actionable and self.probability is None:
            raise ValueError("actionable observation requires probability")
        if self.view == "intraday" and self.probability is not None and self.downside_probability is None:
            raise ValueError("scored intraday observation requires downside probability")
        if self.probability is None and self.calibration_bin is not None:
            raise ValueError("unscored observation cannot have a calibration bin")
        if self.probability is not None and self.calibration_bin != min(int(self.probability * 10), 9):
            raise ValueError("observation calibration bin is inconsistent")
        if self.maturation_key is not None:
            if (
                maturation_key_sha256(self.snapshot_id, self.semantic_prediction_id)
                != self.maturation_key
            ):
                raise ValueError("observation maturation identity is invalid")
        elif (
            monitoring_semantic_sha256(
                self.model_dump(
                    mode="python",
                    exclude={"observation_id", "semantic_prediction_id"},
                )
            )
            != self.semantic_prediction_id
        ):
            raise ValueError("unmatured observation semantic identity is invalid")
        content = self.model_dump(mode="python", exclude={"observation_id"})
        if content_sha256(content) != self.observation_id:
            raise ValueError("prediction observation identity is invalid")
        return self


class PredictionMaturationIntentV2(FrozenContract):
    contract_version: Literal["market_predictor.maturation_intent.v2"] = (
        "market_predictor.maturation_intent.v2"
    )
    maturation_key: str = Field(pattern=SHA256_PATTERN)
    semantic_prediction_id: str = Field(pattern=SHA256_PATTERN)
    snapshot_id: str = Field(pattern=SHA256_PATTERN)
    ticker: str = Field(min_length=1, max_length=16)
    canonical_security_id: str = Field(min_length=1, max_length=128)
    view: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d|b)$")
    decision_time_utc: datetime
    decision_session_et: date
    decision_group_id: str = Field(min_length=1, max_length=256)
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    feature_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy: dict[str, object]
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
    rank: int | None = Field(default=None, ge=1)
    selection_eligible: bool
    selected_for_policy: bool
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
        if self.view == "swing":
            if (
                self.prediction_policy.get("contract_version")
                != "market_predictor.swing_prediction_policy.v1"
            ):
                raise ValueError("swing intent requires the swing prediction policy")
            prediction_policy = parse_swing_prediction_policy(
                self.prediction_policy,
                expected_sha256=self.prediction_policy_sha256,
            )
            horizon = _horizon_amount(self.horizon, unit="b", view="swing")
            if (
                self.label_policy.get("policy")
                != "market_predictor.swing_outcome_policy.v1"
                or self.label_policy.get("horizon_sessions") != horizon
                or prediction_policy.horizon_sessions != horizon
            ):
                raise ValueError("swing intent policy horizons are inconsistent")
            if self.downside_probability is not None:
                raise ValueError("swing intent cannot contain downside probability")
        else:
            if (
                self.prediction_policy.get("contract_version")
                != "market_predictor.prediction_policy.v2"
            ):
                raise ValueError("intraday intent requires the intraday prediction policy")
            parse_prediction_policy(
                self.prediction_policy,
                expected_sha256=self.prediction_policy_sha256,
            )
            horizon = _horizon_amount(self.horizon, unit="m", view="intraday")
            if (
                self.label_policy.get("policy") != "intraday_label.v2"
                or self.label_policy.get("horizon_minutes") != horizon
            ):
                raise ValueError("intraday intent policy horizon is inconsistent")
            if self.downside_probability is None:
                raise ValueError("intraday intent requires downside probability")
        if policy_sha256(self.label_policy) != self.label_policy_sha256:
            raise ValueError("label policy hash does not match its payload")
        if content_sha256(self.prediction_policy) != self.prediction_policy_sha256:
            raise ValueError("prediction policy hash does not match its payload")
        if self.execution_policy_sha256 != EXECUTION_POLICY_SHA256:
            raise ValueError("maturation intent uses an unsupported execution policy")
        if self.selected_for_policy and not self.selection_eligible:
            raise ValueError("selected maturation intent must be eligible")
        if self.actionable != (
            self.selected_for_policy and self.signal != "not_ready"
        ):
            raise ValueError("maturation actionability does not match selection")
        semantic = semantic_prediction_sha256(self.model_dump(exclude={"maturation_key"}))
        if semantic != self.semantic_prediction_id:
            raise ValueError("semantic prediction identity is invalid")
        if maturation_key_sha256(self.snapshot_id, semantic) != self.maturation_key:
            raise ValueError("maturation key is invalid")
        if self.decision_atr is None:
            raise ValueError(f"{self.view} maturation requires decision ATR")
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


class MaturedOutcomeV2(FrozenContract):
    contract_version: Literal["market_predictor.matured_outcome.v2"] = (
        "market_predictor.matured_outcome.v2"
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
    label_round_trip_cost_bps: float = Field(ge=0, le=500)
    label_net_return: float
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    decision_atr: float = Field(gt=0)
    execution_participation_fraction: float = Field(ge=0, le=1)
    execution_cost_bps: float = Field(ge=0)
    net_return: float
    mfe: float
    mae: float
    path_outcome: Literal["positive", "negative", "target_first", "stop_first", "timeout"]
    opportunity_target: int | None = Field(default=None, ge=0, le=1)
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
        if self.label_available_at_utc < self.exit_time_utc:
            raise ValueError("outcome cannot be available before its exit")
        if self.view == "swing":
            if self.path_outcome not in {"target_first", "stop_first", "timeout"}:
                raise ValueError("swing outcome must use managed barrier semantics")
            if self.opportunity_target is not None or self.downside_target is not None:
                raise ValueError("swing outcome cannot contain intraday calibration targets")
        else:
            if self.path_outcome not in {"target_first", "stop_first", "timeout"}:
                raise ValueError("intraday outcome must use managed barrier semantics")
            if self.opportunity_target is None or self.downside_target is None:
                raise ValueError("intraday outcome requires both calibration targets")
        expected_gross = self.exit_price / self.entry_price - 1.0
        expected_label_net = (
            expected_gross - self.label_round_trip_cost_bps / 10_000.0
        )
        if self.execution_policy_sha256 != EXECUTION_POLICY_SHA256:
            raise ValueError("matured outcome uses an unsupported execution policy")
        expected_execution_cost_bps = round_trip_cost_bps(
            price=self.entry_price,
            atr_pct=self.decision_atr / self.entry_price,
            participation=self.execution_participation_fraction,
            policy=DEFAULT_EXECUTION_POLICY,
        )
        expected_net = expected_gross - expected_execution_cost_bps / 10_000.0
        if not math.isclose(self.gross_return, expected_gross, abs_tol=1e-12):
            raise ValueError("matured outcome gross return is inconsistent")
        if not math.isclose(
            self.label_net_return,
            expected_label_net,
            abs_tol=1e-12,
        ):
            raise ValueError("matured outcome label net return is inconsistent")
        if not math.isclose(
            self.execution_cost_bps,
            expected_execution_cost_bps,
            abs_tol=1e-12,
        ):
            raise ValueError("matured outcome execution cost is inconsistent")
        if not math.isclose(self.net_return, expected_net, abs_tol=1e-12):
            raise ValueError("matured outcome net return is inconsistent")
        benchmark_excess = (
            (self.spy_return, self.excess_return_vs_spy),
            (self.qqq_return, self.excess_return_vs_qqq),
            (self.sector_return, self.excess_return_vs_sector),
        )
        if any(
            not math.isclose(self.net_return - benchmark, excess, abs_tol=1e-12)
            for benchmark, excess in benchmark_excess
        ):
            raise ValueError("matured outcome benchmark excess return is inconsistent")
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


def _horizon_amount(horizon: str, *, unit: str, view: str) -> int:
    match = re.fullmatch(rf"([1-9]\d*){re.escape(unit)}", horizon)
    if match is None:
        raise ValueError(f"{view} intent horizon has the wrong unit")
    return int(match.group(1))


def maturation_key_sha256(snapshot_id: str, semantic_prediction_id: str) -> str:
    if not re.fullmatch(SHA256_PATTERN, snapshot_id) or not re.fullmatch(
        SHA256_PATTERN,
        semantic_prediction_id,
    ):
        raise ValueError("maturation identity inputs must be SHA-256 values")
    return hashlib.sha256(
        f"{snapshot_id}:{semantic_prediction_id}".encode("ascii")
    ).hexdigest()


def monitoring_semantic_sha256(observation_without_ids: dict[str, object]) -> str:
    content = dict(observation_without_ids)
    content.pop("snapshot_id", None)
    content.pop("maturation_key", None)
    return content_sha256(content)


def content_sha256(value: object) -> str:
    canonical = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=_json_default,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def monitoring_observation_from_intent(
    intent: PredictionMaturationIntentV2,
) -> PredictionMonitoringObservationV1:
    content: dict[str, object] = {
        "contract_version": "market_predictor.prediction_observation.v1",
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "ticker": intent.ticker,
        "view": intent.view,
        "horizon": intent.horizon,
        "decision_time_utc": intent.decision_time_utc,
        "decision_session_et": intent.decision_session_et,
        "decision_group_id": intent.decision_group_id,
        "model_release_id": intent.model_release_id,
        "model_artifact_sha256": intent.model_artifact_sha256,
        "feature_artifact_sha256": intent.feature_artifact_sha256,
        "prediction_policy_sha256": intent.prediction_policy_sha256,
        "label_policy_sha256": intent.label_policy_sha256,
        "execution_policy_sha256": intent.execution_policy_sha256,
        "market_regime": intent.market_regime,
        "sector": intent.sector,
        "market_cap_bucket": intent.market_cap_bucket,
        "liquidity_bucket": intent.liquidity_bucket,
        "probability": intent.probability,
        "downside_probability": intent.downside_probability,
        "calibration_bin": intent.calibration_bin,
        "signal": intent.signal,
        "rank": intent.rank,
        "selection_eligible": intent.selection_eligible,
        "selected_for_policy": intent.selected_for_policy,
        "actionable": intent.actionable,
        "readiness_status": "valid",
        "catalyst_status": intent.catalyst_status,
        "maturation_key": intent.maturation_key,
    }
    return PredictionMonitoringObservationV1.model_validate(
        {**content, "observation_id": content_sha256(content)}
    )


def _json_default(value: object) -> str:
    if isinstance(value, datetime):
        return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
    if isinstance(value, date):
        return value.isoformat()
    raise TypeError(f"unsupported canonical JSON value: {type(value).__name__}")
