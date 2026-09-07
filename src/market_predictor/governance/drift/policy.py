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

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.drift.features import validate_feature_drift_report
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.governance.outcomes.performance import validate_performance_report
from market_predictor.locking import file_lock

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
    maximum_pending_age_minutes_swing: int = Field(default=30_240, ge=1)
    maximum_pending_age_minutes_intraday: int = Field(default=1_440, ge=1)
    minimum_feature_drift_live_rows: int = Field(default=30, ge=1)
    standardized_shift_warning: float = Field(default=2.0, gt=0)
    standardized_shift_severe: float = Field(default=4.0, gt=0)
    missing_rate_delta_warning: float = Field(default=0.20, gt=0, le=1)
    missing_rate_delta_severe: float = Field(default=0.50, gt=0, le=1)
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
            (
                self.standardized_shift_warning,
                self.standardized_shift_severe,
                "feature standardized shift",
            ),
            (
                self.missing_rate_delta_warning,
                self.missing_rate_delta_severe,
                "feature missing-rate delta",
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
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d|b)$")
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
    feature_drift_report_id: str | None = Field(default=None, pattern=SHA256_PATTERN)
    feature_reference_profile_sha256: str | None = Field(
        default=None,
        pattern=SHA256_PATTERN,
    )
    feature_reference_names_sha256: str | None = Field(
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
        if self.feature_drift_status in {"stable", "warning", "severe"} and (
            self.feature_drift_report_id is None
            or self.feature_reference_profile_sha256 is None
            or self.feature_reference_names_sha256 is None
        ):
            raise ValueError("available feature drift lacks bound evidence identity")
        expected_actionability = {
            "stable": "actionable",
            "warning": "actionable",
            "warming": "rank_only",
            "severe": "not_ready",
            "stale": "not_ready",
            "unavailable": "not_ready",
        }[self.state]
        if self.actionability != expected_actionability:
            raise ValueError("drift state and actionability are inconsistent")
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
    feature_reference_profile_sha256: str,
    feature_reference_names_sha256: str,
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
    if re.fullmatch(SHA256_PATTERN, feature_reference_profile_sha256) is None:
        raise ValueError("drift feature reference identity is invalid")
    if re.fullmatch(SHA256_PATTERN, feature_reference_names_sha256) is None:
        raise ValueError("drift feature-name identity is invalid")
    reasons: list[str] = []
    validated_feature = (
        validate_feature_drift_report(feature_drift)
        if feature_drift is not None
        else None
    )
    feature_status = "unavailable"
    if validated_feature is not None:
        feature_status = str(validated_feature["status"])
        feature_identity = {
            "mode": mode,
            "horizon": horizon,
            "model_release_id": model_release_id,
            "model_artifact_sha256": model_artifact_sha256,
        }
        if any(
            validated_feature.get(name) != value
            for name, value in feature_identity.items()
        ):
            feature_status = "unavailable"
            reasons.append("feature_drift_identity_mismatch")
        elif (
            validated_feature.get("reference_profile_sha256")
            != feature_reference_profile_sha256
        ):
            feature_status = "unavailable"
            reasons.append("feature_drift_reference_identity_mismatch")
        elif (
            validated_feature.get("feature_names_sha256")
            != feature_reference_names_sha256
        ):
            feature_status = "unavailable"
            reasons.append("feature_drift_feature_names_identity_mismatch")
        elif any(
            not math.isclose(
                _as_float(validated_feature.get(name), name),
                float(getattr(policy, name)),
                abs_tol=1e-12,
            )
            for name in (
                "standardized_shift_warning",
                "standardized_shift_severe",
                "missing_rate_delta_warning",
                "missing_rate_delta_severe",
            )
        ):
            feature_status = "unavailable"
            reasons.append("feature_drift_threshold_identity_mismatch")
        elif _as_int(validated_feature.get("live_rows"), "live_rows") < (
            policy.minimum_feature_drift_live_rows
        ):
            feature_status = "unavailable"
            reasons.append("feature_drift_sample_insufficient")
        else:
            feature_generated = _timestamp(
                validated_feature.get("generated_at_utc"),
                "feature_drift.generated_at_utc",
            )
            feature_window_end = _timestamp(
                validated_feature.get("window_end_utc"),
                "feature_drift.window_end_utc",
            )
            if feature_generated > now:
                feature_status = "stale"
                reasons.append("feature_drift_from_future")
            elif feature_window_end > now:
                feature_status = "stale"
                reasons.append("feature_drift_window_from_future")
            elif now - feature_window_end > timedelta(
                minutes=policy.maximum_report_age_minutes
            ):
                feature_status = "stale"
                reasons.append("feature_drift_observations_stale")
            elif now - feature_generated > timedelta(
                minutes=policy.maximum_report_age_minutes
            ):
                feature_status = "stale"
                reasons.append("feature_drift_stale")
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
    if validated_report is not None and row is None:
        reasons.append("selected_policy_identity_mismatch")
    if (
        row is not None
        and validated_feature is not None
        and validated_feature.get("feature_artifact_set_sha256")
        != row.get("feature_artifact_set_sha256")
    ):
        feature_status = "unavailable"
        reasons.append("feature_drift_feature_set_mismatch")
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
            else str(validated_feature["feature_artifact_set_sha256"])
            if validated_feature is not None
            else None
        ),
        "feature_drift_report_id": (
            str(validated_feature["report_id"])
            if validated_feature is not None
            else None
        ),
        "feature_reference_profile_sha256": (
            feature_reference_profile_sha256
        ),
        "feature_reference_names_sha256": feature_reference_names_sha256,
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
            if path.exists():
                existing = self._load_path(path)
                if existing.evaluated_at_utc > assessment.evaluated_at_utc:
                    raise PredictionConflictError
                if existing.evaluated_at_utc == assessment.evaluated_at_utc:
                    if existing != assessment:
                        raise PredictionConflictError
                    return existing
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
        return self._load_path(path)

    @staticmethod
    def _load_path(path: Path) -> DriftAssessmentV2:
        try:
            loaded = parse_strict_json_object(
                path.read_bytes(),
                label="route drift assessment",
            )
            assessment = DriftAssessmentV2.model_validate(loaded)
        except (OSError, ValueError, ValidationError) as exc:
            raise PredictionConflictError from exc
        return assessment

    def _path(self, mode: str, horizon: str, release_id: str) -> Path:
        if mode not in {"swing", "intraday"}:
            raise ValueError("drift state mode is invalid")
        if not re.fullmatch(r"[1-9]\d*(?:m|d|b)", horizon):
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
    if generated > now:
        reasons.append("performance_report_from_future")
        return "stale", "not_ready"
    if now - generated > timedelta(minutes=policy.maximum_report_age_minutes):
        reasons.append("performance_report_stale")
        return "stale", "not_ready"
    pending = _as_int(
        row.get("pending_selected_samples"),
        "pending_selected_samples",
    )
    if pending > 0:
        oldest_pending = _timestamp(
            row.get("oldest_pending_decision_time_utc"),
            "oldest_pending_decision_time_utc",
        )
        maximum_pending_age = (
            policy.maximum_pending_age_minutes_intraday
            if row.get("view") == "intraday"
            else policy.maximum_pending_age_minutes_swing
        )
        if now - oldest_pending > timedelta(minutes=maximum_pending_age):
            reasons.append("selected_policy_outcomes_overdue")
            return "unavailable", "not_ready"
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
    if last_matured > now:
        reasons.append("last_matured_outcome_from_future")
        return "stale", "not_ready"
    if now - last_matured > timedelta(
        minutes=policy.maximum_last_matured_age_minutes
    ):
        reasons.append("last_matured_outcome_stale")
        return "stale", "not_ready"
    opportunity_brier = (
        _as_float(
            row.get("opportunity_brier_score"),
            "opportunity_brier_score",
        )
        if row.get("view") == "intraday"
        else 0.0
    )
    opportunity_calibration = (
        _as_float(
            row.get("opportunity_calibration_error"),
            "opportunity_calibration_error",
        )
        if row.get("view") == "intraday"
        else 0.0
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
    excess_by_benchmark = {
        "spy": _as_float(
            row.get("average_excess_return_vs_spy"),
            "average_excess_return_vs_spy",
        ),
        "qqq": _as_float(
            row.get("average_excess_return_vs_qqq"),
            "average_excess_return_vs_qqq",
        ),
        "sector": _as_float(
            row.get("average_excess_return_vs_sector"),
            "average_excess_return_vs_sector",
        ),
    }
    weakest_excess = min(excess_by_benchmark.values())
    drawdown = _as_float(row.get("max_drawdown"), "max_drawdown")
    severe = (
        opportunity_brier >= policy.severe_opportunity_brier_score
        or downside_brier >= policy.severe_downside_brier_score
        or opportunity_calibration >= policy.severe_calibration_error
        or downside_calibration >= policy.severe_calibration_error
        or weakest_excess <= policy.severe_min_excess_return
        or drawdown >= policy.severe_max_drawdown
    )
    if severe:
        reasons.extend(
            f"selected_policy_{benchmark}_excess_return_severe"
            for benchmark, value in excess_by_benchmark.items()
            if value <= policy.severe_min_excess_return
        )
        reasons.append("selected_policy_performance_severe")
        return "severe", "not_ready"
    warning = (
        opportunity_brier >= policy.warning_opportunity_brier_score
        or downside_brier >= policy.warning_downside_brier_score
        or opportunity_calibration >= policy.warning_calibration_error
        or downside_calibration >= policy.warning_calibration_error
        or weakest_excess <= policy.warning_min_excess_return
        or drawdown >= policy.warning_max_drawdown
    )
    if warning:
        reasons.extend(
            f"selected_policy_{benchmark}_excess_return_warning"
            for benchmark, value in excess_by_benchmark.items()
            if value <= policy.warning_min_excess_return
        )
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
    encoded = json.dumps(
        value,
        indent=2,
        sort_keys=True,
        ensure_ascii=True,
        allow_nan=False,
    ).encode("utf-8")
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        with temporary.open("xb") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
        _fsync_directory(path.parent)
    finally:
        temporary.unlink(missing_ok=True)


def _fsync_directory(path: Path) -> None:
    if os.name == "nt":
        return
    descriptor = os.open(path, os.O_RDONLY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
