from __future__ import annotations

import json
import math
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal
from uuid import uuid4

from pydantic import (
    BaseModel,
    ConfigDict,
    Field,
    ValidationError,
    field_validator,
    model_validator,
)

from market_predictor.locking import file_lock
from market_predictor.outcome_contracts import content_sha256
from market_predictor.performance_monitoring import validate_performance_report
from market_predictor.prediction_contracts import PredictionConflictError
from market_predictor.v3.errors import DataReadinessError

DRIFT_POLICY_VERSION = "market_predictor.drift_policy.v1"
DRIFT_ASSESSMENT_VERSION = "market_predictor.drift_assessment.v1"


class DriftPolicyV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.drift_policy.v1"] = (
        "market_predictor.drift_policy.v1"
    )
    minimum_matured_samples: int = Field(default=30, ge=1)
    maximum_report_age_minutes: int = Field(default=1_440, ge=1)
    warning_brier_score: float = Field(default=0.25, ge=0, le=1)
    severe_brier_score: float = Field(default=0.35, ge=0, le=1)
    warning_min_excess_return: float = -0.001
    severe_min_excess_return: float = -0.005
    warning_max_drawdown: float = Field(default=0.15, ge=0, le=1)
    severe_max_drawdown: float = Field(default=0.25, ge=0, le=1)
    feature_drift_required: bool = True

    @model_validator(mode="after")
    def ordered_thresholds(self) -> DriftPolicyV1:
        if self.warning_brier_score > self.severe_brier_score:
            raise ValueError("warning Brier threshold cannot exceed severe threshold")
        if self.warning_min_excess_return < self.severe_min_excess_return:
            raise ValueError(
                "warning excess-return threshold cannot be below severe threshold"
            )
        if self.warning_max_drawdown > self.severe_max_drawdown:
            raise ValueError("warning drawdown threshold cannot exceed severe threshold")
        return self

    def sha256(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


class DriftAssessmentV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.drift_assessment.v1"] = (
        "market_predictor.drift_assessment.v1"
    )
    assessment_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    mode: Literal["swing", "intraday"]
    horizon: str
    model_release_id: str = Field(pattern=r"^[0-9a-f]{64}$")
    policy_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    performance_report_id: str | None = Field(
        default=None,
        pattern=r"^[0-9a-f]{64}$",
    )
    evaluated_at_utc: datetime
    state: Literal["stable", "warning", "warming", "severe", "stale", "unavailable"]
    actionability: Literal["actionable", "rank_only", "not_ready"]
    reasons: tuple[str, ...] = ()
    feature_drift_status: str
    matured_samples: int = Field(ge=0)

    @field_validator("evaluated_at_utc")
    @classmethod
    def aware_evaluation(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("drift evaluation must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_assessment_identity(self) -> DriftAssessmentV1:
        content = self.model_dump(mode="json", exclude={"assessment_id"})
        if content_sha256(content) != self.assessment_id:
            raise ValueError("drift assessment identity is invalid")
        return self


def evaluate_drift(
    *,
    mode: str,
    horizon: str,
    model_release_id: str,
    feature_drift: dict[str, object] | None,
    performance_report: dict[str, object] | None,
    policy: DriftPolicyV1,
    evaluated_at: datetime | None = None,
) -> DriftAssessmentV1:
    now = (evaluated_at or datetime.now(UTC)).astimezone(UTC)
    reasons: list[str] = []
    feature_status = str((feature_drift or {}).get("status", "unavailable"))
    if feature_status not in {"stable", "warning", "severe", "stale", "unavailable"}:
        feature_status = "unavailable"
        reasons.append("feature_drift_status_invalid")
    validated_report = (
        validate_performance_report(performance_report)
        if performance_report is not None
        else None
    )
    if policy.feature_drift_required and feature_status in {"unavailable", "stale"}:
        state = "unavailable" if feature_status == "unavailable" else "stale"
        actionability = "not_ready"
        reasons.append(f"feature_drift_{feature_status}")
        row = None
    elif feature_status == "severe":
        state = "severe"
        actionability = "not_ready"
        reasons.append("feature_drift_severe")
        row = None
    else:
        row = _route_row(
            validated_report,
            mode=mode,
            horizon=horizon,
            model_release_id=model_release_id,
        )
        state, actionability = _performance_state(
            row,
            performance_report=validated_report,
            policy=policy,
            now=now,
            reasons=reasons,
        )
        if feature_status == "warning" and state == "stable":
            state = "warning"
            reasons.append("feature_drift_warning")
    report_id = (
        str(validated_report.get("report_id"))
        if validated_report is not None and validated_report.get("report_id")
        else None
    )
    samples = _as_int(row.get("samples", 0), "samples") if row is not None else 0
    content = {
        "contract_version": DRIFT_ASSESSMENT_VERSION,
        "mode": mode,
        "horizon": horizon,
        "model_release_id": model_release_id,
        "policy_sha256": policy.sha256(),
        "performance_report_id": report_id,
        "evaluated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "state": state,
        "actionability": actionability,
        "reasons": tuple(sorted(set(reasons))),
        "feature_drift_status": feature_status,
        "matured_samples": samples,
    }
    return DriftAssessmentV1.model_validate(
        {**content, "assessment_id": content_sha256(content)}
    )


class DriftStateStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def publish(self, assessment: DriftAssessmentV1) -> DriftAssessmentV1:
        path = self._path(
            assessment.mode,
            assessment.horizon,
            assessment.model_release_id,
        )
        with file_lock(path):
            _write_json_atomic(path, assessment.model_dump(mode="json"))
        return assessment

    def load(
        self,
        mode: str,
        horizon: str,
        model_release_id: str,
    ) -> DriftAssessmentV1:
        path = self._path(mode, horizon, model_release_id)
        if not path.exists():
            raise DataReadinessError("route drift assessment is unavailable")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        try:
            assessment = DriftAssessmentV1.model_validate(loaded)
        except ValidationError as exc:
            raise PredictionConflictError from exc
        content = assessment.model_dump(mode="json", exclude={"assessment_id"})
        if content_sha256(content) != assessment.assessment_id:
            raise PredictionConflictError
        return assessment

    def _path(self, mode: str, horizon: str, release_id: str) -> Path:
        if mode not in {"swing", "intraday"}:
            raise ValueError("drift state mode is invalid")
        if not re.fullmatch(r"[1-9]\d*(?:m|d)", horizon):
            raise ValueError("drift state horizon is invalid")
        if not re.fullmatch(r"[0-9a-f]{64}", release_id):
            raise ValueError("drift state release identity is invalid")
        return self.root / mode / horizon / f"{release_id}.json"


def _route_row(
    report: dict[str, object] | None,
    *,
    mode: str,
    horizon: str,
    model_release_id: str,
) -> dict[str, object] | None:
    if report is None:
        return None
    rows = report.get("rows")
    if not isinstance(rows, list):
        return None
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and row.get("cohort_type") == "all"
        and row.get("view") == mode
        and row.get("horizon") == horizon
        and row.get("model_release_id") == model_release_id
    ]
    return matches[0] if len(matches) == 1 else None


def _performance_state(
    row: dict[str, object] | None,
    *,
    performance_report: dict[str, object] | None,
    policy: DriftPolicyV1,
    now: datetime,
    reasons: list[str],
) -> tuple[str, str]:
    if performance_report is None or row is None:
        reasons.append("matured_performance_unavailable")
        return "warming", "rank_only"
    generated_raw = performance_report.get("generated_at_utc")
    try:
        generated = datetime.fromisoformat(str(generated_raw)).astimezone(UTC)
    except (TypeError, ValueError):
        reasons.append("performance_report_timestamp_invalid")
        return "stale", "not_ready"
    if now - generated > timedelta(minutes=policy.maximum_report_age_minutes):
        reasons.append("performance_report_stale")
        return "stale", "not_ready"
    samples = _as_int(row.get("samples", 0), "samples")
    if samples < policy.minimum_matured_samples:
        reasons.append("matured_sample_count_below_policy")
        return "warming", "rank_only"
    brier = _as_float(row.get("brier_score"), "brier_score")
    excess = _as_float(
        row.get("average_excess_return_vs_spy"),
        "average_excess_return_vs_spy",
    )
    drawdown = _as_float(row.get("max_drawdown"), "max_drawdown")
    severe = (
        brier >= policy.severe_brier_score
        or excess <= policy.severe_min_excess_return
        or drawdown >= policy.severe_max_drawdown
    )
    if severe:
        reasons.append("matured_performance_severe")
        return "severe", "not_ready"
    warning = (
        brier >= policy.warning_brier_score
        or excess <= policy.warning_min_excess_return
        or drawdown >= policy.warning_max_drawdown
    )
    if warning:
        reasons.append("matured_performance_warning")
        return "warning", "actionable"
    return "stable", "actionable"


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(f"performance report {name} is invalid")
    return value


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"performance report {name} is invalid")
    result = float(value)
    if not math.isfinite(result):
        raise DataReadinessError(f"performance report {name} is invalid")
    return result


def _write_json_atomic(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, ensure_ascii=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
