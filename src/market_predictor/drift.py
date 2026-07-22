from __future__ import annotations

from collections.abc import Sequence
from typing import Any, TypedDict

import numpy as np
import pandas as pd


class DriftRow(TypedDict):
    feature: str
    standardized_mean_shift: float | None
    missing_rate_delta: float | None


def build_feature_reference_profile(
    frame: pd.DataFrame,
    features: Sequence[str],
) -> dict[str, dict[str, float | int | None]]:
    """Build a compact numeric training reference without retaining raw rows."""

    profile: dict[str, dict[str, float | int | None]] = {}
    for feature in features:
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
        valid = values.dropna()
        profile[feature] = {
            "rows": int(len(values)),
            "observed": int(len(valid)),
            "missing_rate": float(values.isna().mean()),
            "mean": float(valid.mean()) if not valid.empty else None,
            "std": float(valid.std(ddof=0)) if not valid.empty else None,
        }
    return profile


def audit_feature_drift(
    frame: pd.DataFrame,
    reference: dict[str, Any] | None,
    *,
    standardized_shift_warning: float = 2.0,
    standardized_shift_severe: float = 4.0,
    missing_rate_delta_warning: float = 0.20,
    missing_rate_delta_severe: float = 0.50,
) -> dict[str, object]:
    """Compare a live cross section with training means and missingness."""

    if not reference:
        return {
            "status": "unavailable",
            "features_compared": 0,
            "reason": "model manifest has no feature reference profile",
        }
    rows: list[DriftRow] = []
    for feature, raw in reference.items():
        if feature not in frame or not isinstance(raw, dict):
            continue
        values = pd.to_numeric(frame[feature], errors="coerce").replace([np.inf, -np.inf], np.nan)
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
    if not rows:
        return {
            "status": "unavailable",
            "features_compared": 0,
            "reason": "live frame has no features from the model reference profile",
        }
    shifts = [float(row["standardized_mean_shift"]) for row in rows if row["standardized_mean_shift"] is not None]
    missing_deltas = [float(row["missing_rate_delta"]) for row in rows if row["missing_rate_delta"] is not None]
    severe = sum(
        bool(
            float(row["standardized_mean_shift"] or 0.0) >= standardized_shift_severe
            or float(row["missing_rate_delta"] or 0.0) >= missing_rate_delta_severe
        )
        for row in rows
    )
    warnings = sum(
        bool(
            float(row["standardized_mean_shift"] or 0.0) >= standardized_shift_warning
            or float(row["missing_rate_delta"] or 0.0) >= missing_rate_delta_warning
        )
        for row in rows
    )
    status = "severe" if severe else "warning" if warnings else "stable"
    largest = sorted(
        rows,
        key=lambda row: max(
            float(row["standardized_mean_shift"] or 0.0),
            float(row["missing_rate_delta"] or 0.0),
        ),
        reverse=True,
    )[:10]
    return {
        "status": status,
        "features_compared": len(rows),
        "warning_feature_count": warnings,
        "severe_feature_count": severe,
        "max_standardized_mean_shift": max(shifts, default=0.0),
        "max_missing_rate_delta": max(missing_deltas, default=0.0),
        "largest_shifts": largest,
    }


def _finite(value: object) -> float | None:
    try:
        converted = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    return converted if np.isfinite(converted) else None
