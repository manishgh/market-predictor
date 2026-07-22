from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import cast
from uuid import uuid4

import pandas as pd

from market_predictor.live_features import (
    LIVE_ARTIFACT_TYPES,
    LIVE_SCHEMA_VERSIONS,
    LiveMode,
    forbidden_live_columns,
    live_feature_columns,
)

FeatureMode = LiveMode
LIVE_FEATURE_SCHEMA = "market_predictor.live_feature_snapshot.v1"


@dataclass(frozen=True)
class LiveFeatureStoreConfig:
    swing_path: Path = Path("data/live/features/swing.parquet")
    intraday_path: Path = Path("data/live/features/intraday.parquet")
    swing_max_age: timedelta = timedelta(hours=36)
    intraday_max_age: timedelta = timedelta(minutes=20)
    swing_feature_max_age: timedelta = timedelta(days=4)
    intraday_feature_max_age: timedelta = timedelta(minutes=20)


class LiveFeatureStore:
    """Registered, integrity-checked feature snapshots for low-latency serving."""

    def __init__(self, root: Path | str, config: LiveFeatureStoreConfig | None = None) -> None:
        self.root = Path(root)
        self.config = config or LiveFeatureStoreConfig()

    def load(
        self,
        mode: FeatureMode,
        *,
        as_of: datetime | None = None,
    ) -> pd.DataFrame:
        path = self._path(mode)
        manifest = self.validate(mode, as_of=as_of)
        frame = pd.read_parquet(path)
        if "ticker" not in frame.columns or "date" not in frame.columns:
            raise ValueError(f"live {mode} feature snapshot must contain ticker and date")
        expected_rows_raw = manifest.get("rows")
        if not isinstance(expected_rows_raw, int):
            raise ValueError(f"live {mode} feature manifest has an invalid row count")
        expected_rows = expected_rows_raw
        if expected_rows != len(frame):
            raise ValueError(f"live {mode} feature snapshot row count does not match its manifest")
        expected_columns = manifest.get("columns")
        if not isinstance(expected_columns, list) or list(frame.columns) != expected_columns:
            raise ValueError(f"live {mode} feature snapshot columns do not match its manifest")
        if "price_feed" not in frame.columns:
            frame["price_feed"] = str(manifest.get("price_feed", "unknown"))
        frame["stale_cache"] = False
        return frame

    def validate(
        self,
        mode: FeatureMode,
        *,
        as_of: datetime | None = None,
    ) -> dict[str, object]:
        """Validate a registered snapshot without loading its feature rows."""

        path = self._path(mode)
        manifest_path = _manifest_path(path)
        if not path.exists() or not manifest_path.exists():
            raise FileNotFoundError(f"registered live {mode} feature snapshot is unavailable")
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
        if not isinstance(loaded, dict):
            raise ValueError(f"live {mode} feature manifest must contain a JSON object")
        manifest = {str(key): value for key, value in loaded.items()}
        self._validate_manifest(manifest, mode=mode, path=path, as_of=as_of)
        return manifest

    def publish(
        self,
        mode: FeatureMode,
        frame: pd.DataFrame,
        *,
        price_feed: str,
        feature_schema_version: str,
        source_artifact_sha256: str,
        source_artifact_type: str,
        source_watermarks: dict[str, str] | None = None,
        generated_at: datetime | None = None,
    ) -> dict[str, object]:
        if frame.empty:
            raise ValueError("cannot publish an empty live feature snapshot")
        required = {"ticker", "date", "decision_time_utc", "feature_available_at_utc", "price_feed"}
        required.update(live_feature_columns(mode))
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise ValueError(f"live feature snapshot missing columns: {', '.join(missing)}")
        forbidden = forbidden_live_columns(frame.columns)
        if forbidden:
            raise ValueError(f"live feature snapshot contains labels or future paths: {', '.join(forbidden)}")
        expected_schema = LIVE_SCHEMA_VERSIONS[mode]
        if feature_schema_version != expected_schema:
            raise ValueError(f"live {mode} feature schema {feature_schema_version} does not match {expected_schema}")
        if not _is_sha256(source_artifact_sha256):
            raise ValueError("source_artifact_sha256 must be a SHA-256 digest")
        if source_artifact_type != LIVE_ARTIFACT_TYPES[mode]:
            raise ValueError(f"source artifact type is not valid for live {mode} features")
        feeds = set(frame["price_feed"].astype(str).str.lower().str.strip().unique())
        normalized_feed = price_feed.strip().lower()
        if feeds != {normalized_feed}:
            raise ValueError("live feature rows and manifest price feed must match exactly")
        if normalized_feed not in {"sip", "consolidated"}:
            raise ValueError("live feature snapshot requires SIP or consolidated price-feed provenance")
        decision_times = pd.to_datetime(frame["decision_time_utc"], errors="coerce", utc=True)
        if bool(decision_times.isna().any()) or decision_times.nunique() != 1:
            raise ValueError("live feature snapshot must contain one coherent decision time")
        tickers = frame["ticker"].astype(str).str.upper().str.strip()
        if bool(tickers.eq("").any()) or bool(tickers.duplicated().any()):
            raise ValueError("live feature snapshot contains invalid or duplicate tickers")
        generated = _utc(generated_at or datetime.now(UTC))
        availability = pd.to_datetime(frame["feature_available_at_utc"], errors="coerce", utc=True)
        if bool(availability.isna().any()):
            raise ValueError("live feature snapshot contains invalid feature availability")
        if bool((availability > decision_times).any()):
            raise ValueError("live feature snapshot contains future feature values")
        latest_feature = cast(datetime, availability.max().to_pydatetime())
        if latest_feature > generated + timedelta(minutes=1):
            raise ValueError("live feature snapshot contains rows generated in the future")
        feature_max_age = self.config.swing_feature_max_age if mode == "swing" else self.config.intraday_feature_max_age
        if bool(((generated - availability) > feature_max_age).any()):
            raise ValueError(f"live {mode} feature snapshot contains stale rows at publication")
        path = self._path(mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        manifest_path = _manifest_path(path)
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
        try:
            frame.to_parquet(temporary, index=False)
            digest = _file_sha256(temporary)
            manifest = {
                "schema": LIVE_FEATURE_SCHEMA,
                "mode": mode,
                "generated_at_utc": generated.isoformat(),
                "artifact_sha256": digest,
                "feature_schema_version": feature_schema_version,
                "source_artifact_sha256": source_artifact_sha256,
                "source_artifact_type": source_artifact_type,
                "rows": int(len(frame)),
                "tickers": int(tickers.nunique()),
                "columns": list(frame.columns),
                "decision_time_utc": decision_times.iloc[0].isoformat(),
                "first_feature_time": availability.min().isoformat(),
                "last_feature_time": latest_feature.isoformat(),
                "price_feed": normalized_feed,
                "source_watermarks": source_watermarks or {},
            }
            manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
            os.replace(temporary, path)
            os.replace(manifest_tmp, manifest_path)
        finally:
            temporary.unlink(missing_ok=True)
            manifest_tmp.unlink(missing_ok=True)
        return manifest

    def _validate_manifest(
        self,
        manifest: dict[str, object],
        *,
        mode: FeatureMode,
        path: Path,
        as_of: datetime | None,
    ) -> None:
        if manifest.get("schema") != LIVE_FEATURE_SCHEMA or manifest.get("mode") != mode:
            raise ValueError(f"invalid live {mode} feature manifest schema")
        expected_feature_schema = LIVE_SCHEMA_VERSIONS[mode]
        if manifest.get("feature_schema_version") != expected_feature_schema:
            raise ValueError(f"live {mode} feature schema does not match the serving contract")
        if not _is_sha256(str(manifest.get("source_artifact_sha256", ""))):
            raise ValueError(f"live {mode} feature manifest is missing canonical source identity")
        if manifest.get("source_artifact_type") != LIVE_ARTIFACT_TYPES[mode]:
            raise ValueError(f"live {mode} feature manifest has an invalid canonical source type")
        columns = manifest.get("columns")
        required_columns = {"ticker", "date", "decision_time_utc", "feature_available_at_utc", "price_feed"}
        required_columns.update(live_feature_columns(mode))
        if not isinstance(columns, list) or not required_columns.issubset(set(columns)):
            raise ValueError(f"live {mode} feature manifest has an invalid column contract")
        forbidden = forbidden_live_columns(str(column) for column in columns)
        if forbidden:
            raise ValueError(f"live {mode} feature manifest contains labels or future paths")
        if str(manifest.get("price_feed", "")).lower() not in {"sip", "consolidated"}:
            raise ValueError(f"live {mode} feature manifest has invalid price-feed provenance")
        expected_hash = str(manifest.get("artifact_sha256", ""))
        if not expected_hash or _file_sha256(path) != expected_hash:
            raise ValueError(f"live {mode} feature snapshot integrity check failed")
        generated_raw = manifest.get("generated_at_utc")
        if not generated_raw:
            raise ValueError(f"live {mode} feature manifest is missing generated_at_utc")
        generated = _parse_utc(str(generated_raw))
        cutoff = _utc(as_of or datetime.now(UTC))
        if generated > cutoff + timedelta(minutes=1):
            raise ValueError(f"live {mode} feature snapshot was generated after the requested as_of")
        max_age = self.config.swing_max_age if mode == "swing" else self.config.intraday_max_age
        if cutoff - generated > max_age:
            raise ValueError(f"live {mode} feature snapshot is stale")
        last_feature_raw = manifest.get("last_feature_time")
        first_feature_raw = manifest.get("first_feature_time")
        if not first_feature_raw or not last_feature_raw:
            raise ValueError(f"live {mode} feature manifest is missing feature timestamps")
        first_feature = _parse_utc(str(first_feature_raw))
        last_feature = _parse_utc(str(last_feature_raw))
        if last_feature > cutoff + timedelta(minutes=1):
            raise ValueError(f"live {mode} feature snapshot contains future feature rows")
        feature_max_age = self.config.swing_feature_max_age if mode == "swing" else self.config.intraday_feature_max_age
        if cutoff - first_feature > feature_max_age:
            raise ValueError(f"live {mode} feature rows are stale")

    def _path(self, mode: FeatureMode) -> Path:
        configured = self.config.swing_path if mode == "swing" else self.config.intraday_path
        return configured if configured.is_absolute() else self.root / configured

    def paths(self, mode: FeatureMode) -> tuple[Path, Path]:
        path = self._path(mode)
        return path, _manifest_path(path)


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(character in "0123456789abcdef" for character in value.lower())


def _parse_utc(value: str) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.tz_localize("UTC")
    else:
        timestamp = timestamp.tz_convert("UTC")
    return cast(datetime, timestamp.to_pydatetime())


def _utc(value: datetime) -> datetime:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise ValueError("feature snapshot timestamps must be timezone-aware")
    return cast(datetime, timestamp.tz_convert("UTC").to_pydatetime())
