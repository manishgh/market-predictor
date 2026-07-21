from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, cast
from uuid import uuid4

import pandas as pd

FeatureMode = Literal["swing", "intraday"]
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
        source_watermarks: dict[str, str] | None = None,
        generated_at: datetime | None = None,
    ) -> dict[str, object]:
        if frame.empty:
            raise ValueError("cannot publish an empty live feature snapshot")
        if "ticker" not in frame.columns or "date" not in frame.columns:
            raise ValueError("live feature snapshot must contain ticker and date")
        generated = _utc(generated_at or datetime.now(UTC))
        path = self._path(mode)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
        frame.to_parquet(temporary, index=False)
        digest = _file_sha256(temporary)
        dates = pd.to_datetime(frame["date"], errors="coerce", utc=True)
        manifest = {
            "schema": LIVE_FEATURE_SCHEMA,
            "mode": mode,
            "generated_at_utc": generated.isoformat(),
            "artifact_sha256": digest,
            "rows": int(len(frame)),
            "tickers": int(frame["ticker"].astype(str).str.upper().nunique()),
            "first_feature_time": dates.min().isoformat() if dates.notna().any() else None,
            "last_feature_time": dates.max().isoformat() if dates.notna().any() else None,
            "price_feed": price_feed.strip().lower() or "unknown",
            "source_watermarks": source_watermarks or {},
        }
        manifest_path = _manifest_path(path)
        manifest_tmp = manifest_path.with_name(f".{manifest_path.name}.{uuid4().hex}.tmp")
        manifest_tmp.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
        os.replace(manifest_tmp, manifest_path)
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
        if not last_feature_raw:
            raise ValueError(f"live {mode} feature manifest is missing last_feature_time")
        last_feature = _parse_utc(str(last_feature_raw))
        if last_feature > cutoff + timedelta(minutes=1):
            raise ValueError(f"live {mode} feature snapshot contains future feature rows")
        feature_max_age = (
            self.config.swing_feature_max_age
            if mode == "swing"
            else self.config.intraday_feature_max_age
        )
        if cutoff - last_feature > feature_max_age:
            raise ValueError(f"live {mode} feature rows are stale")

    def _path(self, mode: FeatureMode) -> Path:
        configured = self.config.swing_path if mode == "swing" else self.config.intraday_path
        return configured if configured.is_absolute() else self.root / configured


def _manifest_path(path: Path) -> Path:
    return path.with_suffix(path.suffix + ".manifest.json")


def _file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


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
