from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import field_validator

from market_predictor.v3.contracts import utc_datetime
from market_predictor.v3.errors import ArtifactIntegrityError, LeakageAuditError
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract

DEFAULT_DEVELOPMENT_CUTOFF_UTC = datetime(2026, 7, 8, 23, 59, 59, tzinfo=UTC)


class DevelopmentShadowPolicy(FrozenContract):
    timestamp_column: str = "decision_time_utc"
    development_cutoff_utc: datetime = DEFAULT_DEVELOPMENT_CUTOFF_UTC
    schema_version: str = ML_V3_SCHEMA_VERSION

    @field_validator("development_cutoff_utc")
    @classmethod
    def validate_cutoff(cls, value: datetime) -> datetime:
        return utc_datetime(value)


def partition_development_shadow(
    frame: pd.DataFrame,
    *,
    policy: DevelopmentShadowPolicy = DevelopmentShadowPolicy(),
) -> tuple[pd.DataFrame, pd.DataFrame]:
    timestamp = _timestamps(frame, policy.timestamp_column)
    development_mask = timestamp <= policy.development_cutoff_utc
    return frame.loc[development_mask].copy(), frame.loc[~development_mask].copy()


def assert_development_only(frame: pd.DataFrame, *, policy: DevelopmentShadowPolicy = DevelopmentShadowPolicy()) -> None:
    timestamp = _timestamps(frame, policy.timestamp_column)
    violating = int((timestamp > policy.development_cutoff_utc).sum())
    if violating:
        raise LeakageAuditError(f"development input contains {violating} immutable shadow rows")


def write_shadow_partition(
    frame: pd.DataFrame,
    output_path: Path,
    *,
    policy: DevelopmentShadowPolicy = DevelopmentShadowPolicy(),
) -> dict[str, Any]:
    if frame.empty:
        raise ArtifactIntegrityError("shadow partition cannot be empty")
    if output_path.suffix.lower() != ".parquet":
        raise ArtifactIntegrityError("shadow partition path must end in .parquet")
    manifest_path = output_path.with_suffix(".manifest.json")
    if output_path.exists() or manifest_path.exists():
        raise ArtifactIntegrityError(f"shadow partition is immutable and already exists: {output_path}")
    timestamp = _timestamps(frame, policy.timestamp_column)
    if bool((timestamp <= policy.development_cutoff_utc).any()):
        raise LeakageAuditError("shadow partition contains development rows")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fingerprint = _frame_fingerprint(frame)
    manifest: dict[str, Any] = {
        "schema_version": policy.schema_version,
        "timestamp_column": policy.timestamp_column,
        "development_cutoff_utc": policy.development_cutoff_utc.isoformat(),
        "rows": len(frame),
        "first_timestamp_utc": timestamp.min().isoformat() if len(timestamp) else None,
        "last_timestamp_utc": timestamp.max().isoformat() if len(timestamp) else None,
        "sha256": fingerprint,
    }
    temporary = output_path.with_suffix(".tmp.parquet")
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(output_path)
        manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    except Exception:
        temporary.unlink(missing_ok=True)
        output_path.unlink(missing_ok=True)
        manifest_path.unlink(missing_ok=True)
        raise
    return manifest


def _timestamps(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        raise LeakageAuditError(f"missing partition timestamp column: {column}")
    parsed = frame[column].map(_aware_utc_timestamp)
    if bool(parsed.isna().any()):
        raise LeakageAuditError(f"invalid timestamps in partition column: {column}")
    return pd.to_datetime(parsed, utc=True)


def _aware_utc_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")


def _frame_fingerprint(frame: pd.DataFrame) -> str:
    digest = hashlib.sha256()
    digest.update("|".join(f"{column}:{frame[column].dtype}" for column in frame.columns).encode())
    digest.update(pd.util.hash_pandas_object(frame, index=True).to_numpy().tobytes())
    return digest.hexdigest()
