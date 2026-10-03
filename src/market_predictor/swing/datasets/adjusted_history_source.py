"""Normalize independently pinned adjusted history; never admit features or training."""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from types import MappingProxyType
from typing import Any, cast

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import history_archive
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS


@dataclass(frozen=True)
class AdjustedHistorySource:
    root: Path
    directory: Path
    records: Mapping[str, Mapping[str, Any]]
    source_files: Mapping[str, str]
    training_ready: bool = field(default=False, init=False)
    promotion_ready: bool = field(default=False, init=False)


@dataclass(frozen=True)
class AdjustedHistoryUnit:
    bars: pd.DataFrame
    missing_sessions: tuple[date, ...]
    invalid_sessions: tuple[date, ...]
    record: Mapping[str, Any]

    @property
    def security_id(self) -> str:
        return str(self.record["security_id"])

    @property
    def ticker(self) -> str:
        return str(self.record["ticker"])

    @property
    def provider_symbol(self) -> str:
        return str(self.record["provider_symbol"])


def _check_hash(path: Path, expected: str) -> None:
    if not path.is_file() or file_sha256(path) != expected:
        raise DataReadinessError(f"adjusted history source hash differs: {path}")


def load_adjusted_history_source(
    *, root: Path, plan_authority: SourcePin, archive_authority: SourcePin,
) -> AdjustedHistorySource:
    """Replay the strict collector before exposing an immutable unit inventory.

    Authority pins are supplied independently by the caller. Source lineage and
    retrospective retrieval clocks remain in the unchanged archive; canonical bar
    availability is only the established historical research proxy.
    """
    root = root.resolve()
    plan_path, archive_path = (inside(root, pin.path) for pin in (plan_authority, archive_authority))
    if plan_path.name != "_authority.json" or archive_path.name != "_authority.json":
        raise DataReadinessError("adjusted history pins must identify authority files")
    _check_hash(plan_path, plan_authority.sha256)
    _check_hash(archive_path, archive_authority.sha256)
    plan_directory, directory = plan_path.parent, archive_path.parent
    metadata = [plan_path, archive_path, plan_directory / "_request.json", plan_directory / "_manifest.json",
        plan_directory / "daily_bar_units.csv", directory / "_request.json", directory / "_manifest.json"]
    source_files = {path.relative_to(root).as_posix(): file_sha256(path) for path in metadata}
    request = parse_strict_json_object((plan_directory / "_request.json").read_bytes(), label="adjusted history plan")
    archive_request = parse_strict_json_object((directory / "_request.json").read_bytes(), label="adjusted history archive")
    if request.get("scope") != "initial_fit_adjusted_history":
        raise DataReadinessError("adjusted history source requires the initial-fit adjusted plan scope")
    if Path(str(request.get("source_root", ""))).resolve() != root:
        raise DataReadinessError("adjusted history source root differs from the plan")
    if archive_request.get("transport_receipts_required") is not True:
        raise DataReadinessError("adjusted history source requires exact transport receipts")
    manifest = history_archive.load_complete_swing_history_collection(
        directory, plan_directory=plan_directory, expected_adjustment="all",
        expected_plan_authority_sha256=plan_authority.sha256,
    )
    records = {str(record["unit_id"]): MappingProxyType(dict(record)) for record in manifest["unit_artifacts"]}
    if len(records) != len(manifest["unit_artifacts"]):
        raise DataReadinessError("adjusted history source repeats a unit identity")

    def retain(path: Path, digest: str) -> None:
        relative = path.relative_to(root).as_posix()
        if relative in source_files and source_files[relative] != digest:
            raise DataReadinessError("adjusted history source has conflicting artifact pins")
        source_files[relative] = digest

    for name, digest in cast(Mapping[str, str], request["source_files"]).items():
        retain(inside(root, name), digest)
    for record in records.values():
        for path_key, hash_key in (("bars_path", "bars_sha256"), ("unit_manifest_path", "unit_manifest_sha256")):
            path = resolve_inside_authority(directory, record[path_key])
            retain(path, str(record[hash_key]))
        unit_path = resolve_inside_authority(directory, record["unit_manifest_path"])
        _check_hash(unit_path, str(record["unit_manifest_sha256"]))
        unit = parse_strict_json_object(unit_path.read_bytes(), label="adjusted history source unit")
        for page in cast(list[dict[str, Any]], unit["pages"]):
            retain(resolve_inside_authority(directory, page["raw_path"]), page["raw_sha256"])
            transport = page["transport"]
            retain(resolve_inside_authority(directory, transport["body_path"]), transport["metadata"]["sha256"])
    for relative, digest in source_files.items():
        _check_hash(inside(root, relative), digest)
    _check_hash(plan_path, plan_authority.sha256)
    _check_hash(archive_path, archive_authority.sha256)
    return AdjustedHistorySource(root, directory, MappingProxyType(records), MappingProxyType(source_files))


def _timestamps(values: pd.Series) -> pd.Series:
    try:
        if any(pd.isna(stamp := pd.Timestamp(value)) or stamp.tzinfo is None for value in values):
            raise ValueError("missing or timezone-naive timestamp")
        return pd.to_datetime(values, utc=True, errors="raise", format="mixed")
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("adjusted history source has invalid observation clocks") from exc


def read_adjusted_history_unit(context: AdjustedHistorySource, unit_id: str) -> AdjustedHistoryUnit:
    """Read one complete query unit without filling candles or excluding identities.

    Missing sessions describe absence from the query's exchange calendar, not
    historical membership. Invalid stock candles remain explicit session facts;
    downstream feature code must abstain on affected histories.
    """
    if unit_id not in context.records:
        raise DataReadinessError(f"adjusted history unit is absent: {unit_id}")
    record = context.records[unit_id]
    path = resolve_inside_authority(context.directory, record["bars_path"])
    unit_manifest = resolve_inside_authority(context.directory, record["unit_manifest_path"])
    _check_hash(unit_manifest, str(record["unit_manifest_sha256"]))
    _check_hash(path, str(record["bars_sha256"]))
    first, last = (date.fromisoformat(str(record[key])) for key in ("start_date", "end_date"))
    calendar = tuple(xcals.get_calendar("XNYS").sessions_in_range(first, last).date)
    frame = pd.read_parquet(path, columns=RAW_COLUMNS,
        filters=[("session_date", ">=", str(first)), ("session_date", "<=", str(last))])
    _check_hash(path, str(record["bars_sha256"]))
    _check_hash(unit_manifest, str(record["unit_manifest_sha256"]))
    if len(frame) != record["rows"]:
        raise DataReadinessError("adjusted history bounded read differs from the unit inventory")
    for column, expected in (("security_id", record["security_id"]), ("ticker", record["ticker"]),
        ("source", "alpaca"), ("price_feed", "sip"), ("adjustment", "all"), ("timeframe", "1Day")):
        if not frame[column].eq(expected).all():
            raise DataReadinessError(f"adjusted history source identity/feed/basis differs: {column}")
    try:
        dates = frame.session_date.map(date.fromisoformat)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("adjusted history session date is invalid") from exc
    if dates.duplicated().any() or not set(dates).issubset(calendar):
        raise DataReadinessError("adjusted history has duplicate or non-calendar sessions")
    clock = _timestamps(frame.bar_start_utc).dt.tz_convert("America/New_York")
    _timestamps(frame.ingested_at_utc)  # Validate even clocks on unusable candles.
    if not clock.dt.date.eq(dates).all() or not clock.eq(clock.dt.normalize()).all():
        raise DataReadinessError("adjusted history provider midnight differs from its session")
    values = frame.loc[:, ["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid = (~np.isfinite(values.to_numpy(dtype=float)).all(axis=1) | values.le(0).any(axis=1)
        | values.high.lt(values[["open", "close", "low"]].max(axis=1))
        | values.low.gt(values[["open", "close", "high"]].min(axis=1)))
    missing_sessions = tuple(sorted(set(calendar) - set(dates)))
    invalid_sessions = tuple(sorted(dates.loc[invalid]))
    if record["role"] == "benchmark" and (missing_sessions or invalid_sessions):
        raise DataReadinessError("adjusted benchmark has missing or invalid sessions")
    bars = canonicalize_bars(frame.loc[~invalid].copy(), timeframe="1d", availability_policy="market_interval_close")
    return AdjustedHistoryUnit(bars, missing_sessions, invalid_sessions, record)
