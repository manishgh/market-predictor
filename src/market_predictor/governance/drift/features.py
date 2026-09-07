"""Identity-bound feature-distribution drift evidence."""

from __future__ import annotations

from datetime import UTC, datetime
from typing import Any, Literal, Self

import numpy as np
import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.modeling.feature_reference import feature_reference_names_sha256

FEATURE_DRIFT_REPORT_VERSION = "market_predictor.feature_drift_report.v1"
_SHA256_PATTERN = r"^[0-9a-f]{64}$"


class FeatureDriftRow(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    feature: str = Field(min_length=1)
    standardized_mean_shift: float | None = Field(default=None, ge=0)
    missing_rate_delta: float | None = Field(default=None, ge=0, le=1)


class FeatureDriftReportV1(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, allow_inf_nan=False)

    contract_version: Literal["market_predictor.feature_drift_report.v1"] = (
        "market_predictor.feature_drift_report.v1"
    )
    report_id: str = Field(pattern=_SHA256_PATTERN)
    mode: Literal["swing", "intraday"]
    horizon: str = Field(pattern=r"^[1-9]\d*(?:m|d|b)$")
    model_release_id: str = Field(pattern=_SHA256_PATTERN)
    model_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    feature_artifact_set_sha256: str = Field(pattern=_SHA256_PATTERN)
    reference_profile_sha256: str = Field(pattern=_SHA256_PATTERN)
    feature_names_sha256: str = Field(pattern=_SHA256_PATTERN)
    window_start_utc: datetime
    window_end_utc: datetime
    generated_at_utc: datetime
    source_artifact_sha256: str = Field(pattern=_SHA256_PATTERN)
    live_rows: int = Field(ge=0)
    status: Literal["stable", "warning", "severe", "unavailable"]
    features_compared: int = Field(ge=0)
    warning_feature_count: int = Field(ge=0)
    severe_feature_count: int = Field(ge=0)
    max_standardized_mean_shift: float = Field(ge=0)
    max_missing_rate_delta: float = Field(ge=0, le=1)
    standardized_shift_warning: float = Field(gt=0)
    standardized_shift_severe: float = Field(gt=0)
    missing_rate_delta_warning: float = Field(gt=0, le=1)
    missing_rate_delta_severe: float = Field(gt=0, le=1)
    feature_rows: tuple[FeatureDriftRow, ...]
    reason: str | None = None

    @field_validator("window_start_utc", "window_end_utc", "generated_at_utc")
    @classmethod
    def aware_times(cls, value: datetime) -> datetime:
        if value.utcoffset() is None:
            raise ValueError("feature-drift timestamps must be timezone-aware")
        return value.astimezone(UTC)

    @model_validator(mode="after")
    def validate_report(self) -> Self:
        if not self.window_start_utc <= self.window_end_utc <= self.generated_at_utc:
            raise ValueError("feature-drift observation window is invalid")
        if self.standardized_shift_warning > self.standardized_shift_severe:
            raise ValueError("feature-drift standardized thresholds are inverted")
        if self.missing_rate_delta_warning > self.missing_rate_delta_severe:
            raise ValueError("feature-drift missing-rate thresholds are inverted")
        if self.status == "unavailable":
            if (
                self.features_compared != 0
                or self.feature_rows
                or not self.reason
                or self.warning_feature_count != 0
                or self.severe_feature_count != 0
                or self.max_standardized_mean_shift != 0.0
                or self.max_missing_rate_delta != 0.0
            ):
                raise ValueError("unavailable feature drift requires a reason and no rows")
        else:
            if (
                self.features_compared == 0
                or self.features_compared != len(self.feature_rows)
                or self.reason is not None
            ):
                raise ValueError("available feature drift requires every compared row")
            feature_names = [row.feature for row in self.feature_rows]
            if len(feature_names) != len(set(feature_names)):
                raise ValueError("feature-drift rows contain duplicate feature names")
            if feature_reference_names_sha256(feature_names) != self.feature_names_sha256:
                raise ValueError("feature-drift feature-name identity is invalid")
            severe = sum(
                _row_exceeds(
                    row,
                    standardized=self.standardized_shift_severe,
                    missing=self.missing_rate_delta_severe,
                )
                for row in self.feature_rows
            )
            warnings = sum(
                _row_exceeds(
                    row,
                    standardized=self.standardized_shift_warning,
                    missing=self.missing_rate_delta_warning,
                )
                for row in self.feature_rows
            )
            expected_status = "severe" if severe else "warning" if warnings else "stable"
            shifts = [
                row.standardized_mean_shift
                for row in self.feature_rows
                if row.standardized_mean_shift is not None
            ]
            missing_deltas = [
                row.missing_rate_delta
                for row in self.feature_rows
                if row.missing_rate_delta is not None
            ]
            if (
                self.warning_feature_count != warnings
                or self.severe_feature_count != severe
                or self.status != expected_status
                or not np.isclose(
                    self.max_standardized_mean_shift,
                    max(shifts, default=0.0),
                )
                or not np.isclose(
                    self.max_missing_rate_delta,
                    max(missing_deltas, default=0.0),
                )
            ):
                raise ValueError("feature-drift summary does not match feature rows")
        content = self.model_dump(mode="json", exclude={"report_id"})
        if content_sha256(content) != self.report_id:
            raise ValueError("feature-drift report identity is invalid")
        return self


def audit_feature_drift(
    frame: pd.DataFrame,
    reference: dict[str, Any] | None,
    *,
    mode: Literal["swing", "intraday"],
    horizon: str,
    model_release_id: str,
    model_artifact_sha256: str,
    feature_artifact_set_sha256: str,
    source_artifact_sha256: str,
    window_start: datetime,
    window_end: datetime,
    generated_at: datetime,
    standardized_shift_warning: float = 2.0,
    standardized_shift_severe: float = 4.0,
    missing_rate_delta_warning: float = 0.20,
    missing_rate_delta_severe: float = 0.50,
) -> dict[str, object]:
    """Compare a live cohort with one exact model reference profile."""

    identity: dict[str, object] = {
        "contract_version": FEATURE_DRIFT_REPORT_VERSION,
        "mode": mode,
        "horizon": horizon,
        "model_release_id": model_release_id,
        "model_artifact_sha256": model_artifact_sha256,
        "feature_artifact_set_sha256": feature_artifact_set_sha256,
        "reference_profile_sha256": content_sha256(reference or {}),
        "feature_names_sha256": (
            feature_reference_names_sha256(reference)
            if reference
            else content_sha256([])
        ),
        "window_start_utc": _utc_text(window_start),
        "window_end_utc": _utc_text(window_end),
        "generated_at_utc": _utc_text(generated_at),
        "source_artifact_sha256": source_artifact_sha256,
        "live_rows": len(frame),
        "standardized_shift_warning": standardized_shift_warning,
        "standardized_shift_severe": standardized_shift_severe,
        "missing_rate_delta_warning": missing_rate_delta_warning,
        "missing_rate_delta_severe": missing_rate_delta_severe,
    }
    rows: list[dict[str, object]] = []
    incomplete: list[str] = []
    if reference:
        for feature, raw in reference.items():
            if not isinstance(raw, dict):
                incomplete.append(f"{feature}:invalid_reference")
                continue
            if feature not in frame:
                incomplete.append(f"{feature}:missing_live_feature")
                continue
            values = pd.to_numeric(frame[feature], errors="coerce").replace(
                [np.inf, -np.inf], np.nan
            )
            live_mean = float(values.mean()) if values.notna().any() else None
            live_missing = float(values.isna().mean())
            reference_mean = _finite(raw.get("mean"))
            reference_std = _finite(raw.get("std"))
            reference_missing = _finite(raw.get("missing_rate"))
            shift = None
            if live_mean is not None and reference_mean is not None:
                denominator = max(abs(reference_std or 0.0), 1e-6)
                shift = abs(live_mean - reference_mean) / denominator
            missing_delta = (
                abs(live_missing - reference_missing)
                if reference_missing is not None
                else None
            )
            rows.append(
                {
                    "feature": str(feature),
                    "standardized_mean_shift": shift,
                    "missing_rate_delta": missing_delta,
                }
            )
    if not rows or incomplete:
        content: dict[str, object] = {
            **identity,
            "status": "unavailable",
            "features_compared": 0,
            "warning_feature_count": 0,
            "severe_feature_count": 0,
            "max_standardized_mean_shift": 0.0,
            "max_missing_rate_delta": 0.0,
            "feature_rows": (),
            "reason": (
                "model manifest has no feature reference profile"
                if not reference
                else "incomplete feature coverage: " + ", ".join(sorted(incomplete))
                if incomplete
                else "live frame has no features from the model reference profile"
            ),
        }
        return _validated_report(content)
    shifts = [
        value
        for row in rows
        if (value := _finite(row["standardized_mean_shift"])) is not None
    ]
    missing_deltas = [
        value
        for row in rows
        if (value := _finite(row["missing_rate_delta"])) is not None
    ]
    severe = sum(
        bool(
            (_finite(row["standardized_mean_shift"]) or 0.0)
            >= standardized_shift_severe
            or (_finite(row["missing_rate_delta"]) or 0.0)
            >= missing_rate_delta_severe
        )
        for row in rows
    )
    warnings = sum(
        bool(
            (_finite(row["standardized_mean_shift"]) or 0.0)
            >= standardized_shift_warning
            or (_finite(row["missing_rate_delta"]) or 0.0)
            >= missing_rate_delta_warning
        )
        for row in rows
    )
    ordered_rows = sorted(
        rows,
        key=lambda row: max(
            _finite(row["standardized_mean_shift"]) or 0.0,
            _finite(row["missing_rate_delta"]) or 0.0,
        ),
        reverse=True,
    )
    content = {
        **identity,
        "status": "severe" if severe else "warning" if warnings else "stable",
        "features_compared": len(rows),
        "warning_feature_count": warnings,
        "severe_feature_count": severe,
        "max_standardized_mean_shift": max(shifts, default=0.0),
        "max_missing_rate_delta": max(missing_deltas, default=0.0),
        "feature_rows": ordered_rows,
        "reason": None,
    }
    return _validated_report(content)


def validate_feature_drift_report(value: object) -> dict[str, object]:
    return FeatureDriftReportV1.model_validate(value).model_dump(mode="json")


def _validated_report(content: dict[str, object]) -> dict[str, object]:
    report = FeatureDriftReportV1.model_validate(
        {**content, "report_id": content_sha256(content)}
    )
    return report.model_dump(mode="json")


def _finite(value: object) -> float | None:
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None


def _row_exceeds(
    row: FeatureDriftRow,
    *,
    standardized: float,
    missing: float,
) -> int:
    return int(
        (row.standardized_mean_shift or 0.0) >= standardized
        or (row.missing_rate_delta or 0.0) >= missing
    )


def _utc_text(value: datetime) -> str:
    if value.utcoffset() is None:
        raise ValueError("feature-drift timestamps must be timezone-aware")
    return value.astimezone(UTC).isoformat().replace("+00:00", "Z")
