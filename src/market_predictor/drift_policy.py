from __future__ import annotations

import json
import math
import os
import re
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self
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

DRIFT_POLICY_VERSION = "market_predictor.drift_policy.v2"
DRIFT_ASSESSMENT_VERSION = "market_predictor.drift_assessment.v2"
SHA256_PATTERN = r"^[0-9a-f]{64}$"


class DriftPolicyV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.drift_policy.v2"] = (
        "market_predictor.drift_policy.v2"
    )
    minimum_matured_samples: int = Field(default=30, ge=1)
    minimum_independent_decision_groups: int = Field(default=10, ge=1)
    maximum_report_age_minutes: int = Field(default=1_440, ge=1)
    maximum_last_matured_age_minutes: int = Field(default=10_080, ge=1)
    warning_opportunity_brier_score: float = Field(default=0.25, ge=0, le=1)
    severe_opportunity_brier_score: float = Field(default=0.35, ge=0, le=1)
    warning_downside_brier_score: float = Field(default=0.25, ge=0, le=1)
    severe_downside_brier_score: float = Field(default=0.35, ge=0, le=1)
    warning_calibration_error: float = Field(default=0.12, ge=0, le=1)
    severe_calibration_error: float = Field(default=0.20, ge=0, le=1)
    warning_min_excess_return: float = -0.001
    severe_min_excess_return: float = -0.005
    warning_max_drawdown: float = Field(default=0.15, ge=0, le=1)
    severe_max_drawdown: float = Field(default=0.25, ge=0, le=1)
    feature_drift_required: bool = True

    @model_validator(mode="after")
    def ordered_thresholds(self) -> Self:
        pairs = (
            (
                self.warning_opportunity_brier_score,
                self.severe_opportunity_brier_score,
                "opportunity Brier",
            ),
            (
                self.warning_downside_brier_score,
                self.severe_downside_brier_score,
                "downside Brier",
            ),
            (
                self.warning_calibration_error,
                self.severe_calibration_error,
                "calibration error",
            ),
            (
                self.warning_max_drawdown,
                self.severe_max_drawdown,
                "drawdown",
            ),
        )
        for warning, severe, name in pairs:
            if warning > severe:
                raise ValueError(
                    f"warning {name} threshold cannot exceed severe threshold"
                )
        if self.warning_min_excess_return < self.severe_min_excess_return:
            raise ValueError(
                "warning excess-return threshold cannot be below severe threshold"
            )
        return self

    def sha256(self) -> str:
        return content_sha256(self.model_dump(mode="json"))


class DriftAssessmentV2(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.drift_assessment.v2"] = (
        "market_predictor.drift_assessment.v2"
    )
    assessment_id: str = Field(pattern=SHA256_PATTERN)
    mode: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d)$")
    model_release_id: str = Field(pattern=SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=SHA256_PATTERN)
    prediction_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    label_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    execution_policy_sha256: str = Field(pattern=SHA256_PATTERN)
    policy_sha256: str = Field(pattern=SHA256_PATTERN)
    performance_report_id: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    performance_cohort_id: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    feature_artifact_set_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    evaluated_at_utc: datetime
    state: Literal[
        "stable",
        "warning",
        "warming",
        "severe",
        "stale",
        "unavailable",
    ]
    actionability: Literal["actionable", "rank_only", "not_ready"]
    reasons: tuple[str, ...] = ()
    feature_drift_status: str
    total_predictions: int = Field(ge=0)
    selected_predictions: int = Field(ge=0)
    matured_samples: int = Field(ge=0)
    independent_decision_groups: int = Field(ge=0)
    last_matured_outcome_utc: datetime | None = None

    @field_validator("evaluated_at_utc", "last_matured_outcome_utc")
    @classmethod
    def aware_times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        if value.utcoffset() is None:
            raise ValueError("drift assessment timestamp must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_assessment_identity(self) -> Self:
        if self.selected_predictions > self.total_predictions:
            raise ValueError("drift assessment selection counts are invalid")
        content = self.model_dump(mode="json", exclude={"assessment_id"})
        if content_sha256(content) != self.assessment_id:
            raise ValueError("drift assessment identity is invalid")
        return self


def evaluate_drift(
    *,
    mode: str,
    horizon: str,
    model_release_id: str,
    model_artifact_sha256: str,
    prediction_policy_sha256: str,
    label_policy_sha256: str,
    execution_policy_sha256: str,
    feature_drift: dict[str, object] | None,
    performance_report: dict[str, object] | None,
    policy: DriftPolicyV2,
    evaluated_at: datetime | None = None,
) -> DriftAssessmentV2:
    now = _utc(evaluated_at or datetime.now(UTC))
    route_identity = {
        "model_release_id": model_release_id,
        "model_artifact_sha256": model_artifact_sha256,
        "prediction_policy_sha256": prediction_policy_sha256,
        "label_policy_sha256": label_policy_sha256,
        "execution_policy_sha256": execution_policy_sha256,
    }
    for name, value in route_identity.items():
        if re.fullmatch(SHA256_PATTERN, value) is None:
            raise ValueError(f"drift route {name} is invalid")
    reasons: list[str] = []
    feature_status = str((feature_drift or {}).get("status", "unavailable"))
    if feature_status not in {
        "stable",
        "warning",
        "severe",
        "stale",
        "unavailable",
    }:
        feature_status = "unavailable"
        reasons.append("feature_drift_status_invalid")
    validated_report = (
        validate_performance_report(performance_report)
        if performance_report is not None
        else None
    )
    row = _route_row(
        validated_report,
        mode=mode,
        horizon=horizon,
        **route_identity,
    )
    if policy.feature_drift_required and feature_status in {
        "unavailable",
        "stale",
    }:
        state = "unavailable" if feature_status == "unavailable" else "stale"
        actionability = "not_ready"
        reasons.append(f"feature_drift_{feature_status}")
    elif feature_status == "severe":
        state = "severe"
        actionability = "not_ready"
        reasons.append("feature_drift_severe")
    else:
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
        if validated_report is not None
        and validated_report.get("report_id")
        else None
    )
    content = {
        "contract_version": DRIFT_ASSESSMENT_VERSION,
        "mode": mode,
        "horizon": horizon,
        **route_identity,
        "policy_sha256": policy.sha256(),
        "performance_report_id": report_id,
        "performance_cohort_id": (
            str(row["cohort_id"]) if row is not None else None
        ),
        "feature_artifact_set_sha256": (
            str(row["feature_artifact_set_sha256"])
            if row is not None
            else None
        ),
        "evaluated_at_utc": now.isoformat().replace("+00:00", "Z"),
        "state": state,
        "actionability": actionability,
        "reasons": tuple(sorted(set(reasons))),
        "feature_drift_status": feature_status,
        "total_predictions": (
            _as_int(row["total_predictions"], "total_predictions")
            if row is not None
            else 0
        ),
        "selected_predictions": (
            _as_int(row["selected_predictions"], "selected_predictions")
            if row is not None
            else 0
        ),
        "matured_samples": (
            _as_int(row["matured_selected_samples"], "matured_selected_samples")
            if row is not None
            else 0
        ),
        "independent_decision_groups": (
            _as_int(
                row["independent_decision_groups"],
                "independent_decision_groups",
            )
            if row is not None
            else 0
        ),
        "last_matured_outcome_utc": (
            row.get("last_matured_outcome_utc")
            if row is not None
            else None
        ),
    }
    return DriftAssessmentV2.model_validate(
        {**content, "assessment_id": content_sha256(content)}
    )


class DriftStateStore:
    def __init__(self, root: Path) -> None:
        self.root = root.resolve()

    def publish(self, assessment: DriftAssessmentV2) -> DriftAssessmentV2:
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
    ) -> DriftAssessmentV2:
        path = self._path(mode, horizon, model_release_id)
        if not path.exists():
            raise DataReadinessError("route drift assessment is unavailable")
        loaded = json.loads(path.read_text(encoding="utf-8"))
        try:
            assessment = DriftAssessmentV2.model_validate(loaded)
        except ValidationError as exc:
            raise PredictionConflictError from exc
        return assessment

    def _path(self, mode: str, horizon: str, release_id: str) -> Path:
        if mode not in {"swing", "intraday"}:
            raise ValueError("drift state mode is invalid")
        if not re.fullmatch(r"[1-9]\d*(?:m|d)", horizon):
            raise ValueError("drift state horizon is invalid")
        if not re.fullmatch(SHA256_PATTERN, release_id):
            raise ValueError("drift state release identity is invalid")
        return self.root / mode / horizon / f"{release_id}.json"


def _route_row(
    report: dict[str, object] | None,
    *,
    mode: str,
    horizon: str,
    model_release_id: str,
    model_artifact_sha256: str,
    prediction_policy_sha256: str,
    label_policy_sha256: str,
    execution_policy_sha256: str,
) -> dict[str, object] | None:
    if report is None:
        return None
    rows = report.get("rows")
    if not isinstance(rows, list):
        return None
    expected = {
        "cohort_type": "all",
        "view": mode,
        "horizon": horizon,
        "model_release_id": model_release_id,
        "model_artifact_sha256": model_artifact_sha256,
        "prediction_policy_sha256": prediction_policy_sha256,
        "label_policy_sha256": label_policy_sha256,
        "execution_policy_sha256": execution_policy_sha256,
    }
    matches = [
        row
        for row in rows
        if isinstance(row, dict)
        and all(row.get(name) == value for name, value in expected.items())
    ]
    return matches[0] if len(matches) == 1 else None


def _performance_state(
    row: dict[str, object] | None,
    *,
    performance_report: dict[str, object] | None,
    policy: DriftPolicyV2,
    now: datetime,
    reasons: list[str],
) -> tuple[str, str]:
    if performance_report is None:
        reasons.append("selected_policy_performance_unavailable")
        return "warming", "rank_only"
    if row is None:
        reasons.append("selected_policy_identity_mismatch")
        return "unavailable", "not_ready"
    generated = _timestamp(
        performance_report.get("generated_at_utc"),
        "generated_at_utc",
    )
    if generated > now + timedelta(minutes=5):
        reasons.append("performance_report_from_future")
        return "stale", "not_ready"
    if now - generated > timedelta(minutes=policy.maximum_report_age_minutes):
        reasons.append("performance_report_stale")
        return "stale", "not_ready"
    samples = _as_int(
        row.get("matured_selected_samples"),
        "matured_selected_samples",
    )
    groups = _as_int(
        row.get("independent_decision_groups"),
        "independent_decision_groups",
    )
    if (
        samples < policy.minimum_matured_samples
        or groups < policy.minimum_independent_decision_groups
        or row.get("evidence_status") != "sufficient"
    ):
        reasons.append("selected_policy_evidence_insufficient")
        return "warming", "rank_only"
    last_matured = _timestamp(
        row.get("last_matured_outcome_utc"),
        "last_matured_outcome_utc",
    )
    if last_matured > now + timedelta(minutes=5):
        reasons.append("last_matured_outcome_from_future")
        return "stale", "not_ready"
    if now - last_matured > timedelta(
        minutes=policy.maximum_last_matured_age_minutes
    ):
        reasons.append("last_matured_outcome_stale")
        return "stale", "not_ready"
    opportunity_brier = _as_float(
        row.get("opportunity_brier_score"),
        "opportunity_brier_score",
    )
    opportunity_calibration = _as_float(
        row.get("opportunity_calibration_error"),
        "opportunity_calibration_error",
    )
    downside_brier = (
        _as_float(row.get("downside_brier_score"), "downside_brier_score")
        if row.get("view") == "intraday"
        else 0.0
    )
    downside_calibration = (
        _as_float(
            row.get("downside_calibration_error"),
            "downside_calibration_error",
        )
        if row.get("view") == "intraday"
        else 0.0
    )
    excess = _as_float(
        row.get("average_excess_return_vs_spy"),
        "average_excess_return_vs_spy",
    )
    drawdown = _as_float(row.get("max_drawdown"), "max_drawdown")
    severe = (
        opportunity_brier >= policy.severe_opportunity_brier_score
        or downside_brier >= policy.severe_downside_brier_score
        or opportunity_calibration >= policy.severe_calibration_error
        or downside_calibration >= policy.severe_calibration_error
        or excess <= policy.severe_min_excess_return
        or drawdown >= policy.severe_max_drawdown
    )
    if severe:
        reasons.append("selected_policy_performance_severe")
        return "severe", "not_ready"
    warning = (
        opportunity_brier >= policy.warning_opportunity_brier_score
        or downside_brier >= policy.warning_downside_brier_score
        or opportunity_calibration >= policy.warning_calibration_error
        or downside_calibration >= policy.warning_calibration_error
        or excess <= policy.warning_min_excess_return
        or drawdown >= policy.warning_max_drawdown
    )
    if warning:
        reasons.append("selected_policy_performance_warning")
        return "warning", "actionable"
    return "stable", "actionable"


def _timestamp(value: object, name: str) -> datetime:
    try:
        return _utc(datetime.fromisoformat(str(value)))
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(
            f"selected-policy performance {name} is invalid"
        ) from exc


def _as_int(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise DataReadinessError(
            f"selected-policy performance {name} is invalid"
        )
    return value


def _as_float(value: object, name: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(
            f"selected-policy performance {name} is invalid"
        )
    result = float(value)
    if not math.isfinite(result):
        raise DataReadinessError(
            f"selected-policy performance {name} is invalid"
        )
    return result


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("drift timestamp must be timezone-aware")
    return value.astimezone(UTC)


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
