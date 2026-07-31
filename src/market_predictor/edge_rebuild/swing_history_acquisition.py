"""Outcome-blind, authority-bound acquisition planning for swing history."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from datetime import date, timedelta
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.sp500_memberships import (
    MEMBERSHIP_REQUEST_SCHEMA,
    require_sp500_membership_authority,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError

PLAN_SCHEMA = "edge_rebuild.swing_history_acquisition_plan.v2"
AUTHORITY_SCHEMA = "edge_rebuild.swing_history_acquisition_plan_authority.v2"
TEMPORAL_SCHEMA = "edge_rebuild.temporal_manifest.v1"
TEMPORAL_AUTHORITY_SCHEMA = "edge_rebuild.temporal_manifest_authority.v1"
DAILY_REQUEST_SCHEMA = "swing.daily_history_collection.v1"
DAILY_MANIFEST_SCHEMA = "swing.daily_history_manifest.v1"
DAILY_BAR_UNITS_FILE = "daily_bar_units.csv"
MAX_MEMORY_GIB = 4.0
MEMORY_HEADROOM_GIB = 0.75
ANNOUNCEMENT_LEAD_DAYS = 45
EASTERN = ZoneInfo("America/New_York")


def publish_swing_history_acquisition_plan(
    *,
    repository_root: Path,
    temporal_manifest_directory: Path,
    membership_authority_directory: Path,
    raw_archive_directory: Path,
    event_authority_directory: Path,
    transition_authority_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    current_daily_collection_directory: Path,
    output_directory: Path,
    security_exclusions_path: Path | None = None,
) -> dict[str, Any]:
    """Publish exact missing-history units from a fully verified PIT universe."""

    root = repository_root.resolve()
    temporal_dir = _bound_directory(root, temporal_manifest_directory)
    membership_dir = _bound_directory(root, membership_authority_directory)
    raw_archive_dir = _bound_directory(root, raw_archive_directory)
    event_authority_dir = _bound_directory(root, event_authority_directory)
    transition_authority_dir = _bound_directory(root, transition_authority_directory)
    reviewed_transitions = _bound_file(root, reviewed_transitions_path)
    anchor = _bound_file(root, anchor_path)
    security_exclusions = _bound_file(root, security_exclusions_path) if security_exclusions_path is not None else None
    daily_dir = _bound_directory(root, current_daily_collection_directory)
    output = _bound_output(root, output_directory)
    if output.exists():
        raise DataReadinessError(f"swing acquisition-plan output must be new: {output}")
    _guard("swing acquisition-plan start")

    temporal, temporal_hashes = _load_temporal(temporal_dir)
    missing_ranges = _missing_ranges(temporal)
    missing_start = date.fromisoformat(str(missing_ranges[0]["first_session"]))
    missing_end = date.fromisoformat(str(missing_ranges[-1]["last_session"]))
    memberships, membership, membership_hashes = _load_membership_authority(
        membership_directory=membership_dir,
        raw_archive_directory=raw_archive_dir,
        event_authority_directory=event_authority_dir,
        transition_authority_directory=transition_authority_dir,
        reviewed_transitions_path=reviewed_transitions,
        anchor_path=anchor,
        security_exclusions_path=security_exclusions,
    )
    daily, daily_hashes = _load_daily_collection(root, daily_dir)
    _validate_coverage(
        memberships=memberships,
        authority_start=membership["authority_start"],
        authority_cutoff=membership["authority_cutoff"],
        daily=daily,
        missing_start=missing_start,
        missing_end=missing_end,
    )
    units = _build_daily_bar_units(memberships, missing_ranges)
    _guard("swing acquisition-plan evidence")

    request = {
        "schema": PLAN_SCHEMA,
        "temporal_manifest_sha256": temporal_hashes["manifest"],
        "temporal_authority_sha256": temporal_hashes["authority"],
        "membership_authority": membership_hashes,
        **daily_hashes,
    }
    stock_units = int(units["role"].eq("stock").sum())
    benchmark_units = int(units["role"].eq("benchmark").sum())
    manifest: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": "ready_for_daily_history_collection",
        "outcomes_read": False,
        "missing_session_ranges": missing_ranges,
        "membership": {
            "authority_start": membership["authority_start"].isoformat(),
            "authority_cutoff": membership["authority_cutoff"].isoformat(),
            "current_membership_start": _membership_start(memberships).isoformat(),
            "membership_dates_cover_required_window": True,
            "required_start": missing_start.isoformat(),
            "required_end": missing_end.isoformat(),
            "official_announcement_discovery_start": (missing_start - timedelta(days=ANNOUNCEMENT_LEAD_DAYS)).isoformat(),
            "official_announcement_discovery_end": missing_end.isoformat(),
            "security_count": membership["security_count"],
            "excluded_security_count": membership["excluded_security_count"],
            "excluded_security_fraction": membership["excluded_security_fraction"],
            "benchmark_session_exclusions": 0,
            "universe_sha256": membership["universe_sha256"],
            "parent_lineage": membership["parent_lineage"],
        },
        "daily_bars": {
            "status": "ready",
            "planned_units": len(units),
            "stock_units": stock_units,
            "benchmark_units": benchmark_units,
            "source": "alpaca",
            "timeframe": "1Day",
            "price_feed": "sip",
            "adjustment": "all",
            "unit_policy": (
                "verified security/ticker membership interval intersected with "
                "each missing session range; benchmarks cover every missing range"
            ),
            "current_collection_total_rows": daily["total_rows"],
            "current_collection_reused": str(daily_dir.relative_to(root)).replace("\\", "/"),
        },
        "next_operations": [
            "collect exactly the published daily-bar units from Alpaca SIP",
            "validate whole-security stock coverage and complete benchmark coverage",
            "materialize the extended causal swing panel",
        ],
    }

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True)
        units_path = staging / DAILY_BAR_UNITS_FILE
        units.to_csv(units_path, index=False, lineterminator="\n")
        manifest["daily_bars"]["units_artifact"] = {
            "path": DAILY_BAR_UNITS_FILE,
            "bytes": units_path.stat().st_size,
            "sha256": file_sha256(units_path),
        }
        _write_json(staging / "_request.json", request)
        manifest["request_sha256"] = file_sha256(staging / "_request.json")
        assert_peak_memory_budget(
            hard_budget_gib=MAX_MEMORY_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
            stage="swing acquisition-plan publication",
        )
        manifest["resources"] = memory_audit(
            hard_budget_gib=MAX_MEMORY_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
        ).to_record()
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": manifest["request_sha256"],
                "units_sha256": manifest["daily_bars"]["units_artifact"]["sha256"],
                "universe_sha256": membership["universe_sha256"],
            },
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _load_temporal(directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    authority_path = directory / "_authority.json"
    manifest_path = directory / "_manifest.json"
    authority = _read_json(authority_path)
    if (
        authority.get("schema") != TEMPORAL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("temporal acquisition input lacks valid authority")
    manifest = _read_json(manifest_path)
    coverage = manifest.get("coverage")
    if (
        manifest.get("schema") != TEMPORAL_SCHEMA
        or manifest.get("status") != "insufficient_history"
        or not isinstance(coverage, dict)
        or coverage.get("outcomes_read") is not False
    ):
        raise DataReadinessError("temporal acquisition input is not an outcome-blind gap")
    return manifest, {
        "manifest": file_sha256(manifest_path),
        "authority": file_sha256(authority_path),
    }


def _missing_ranges(manifest: dict[str, Any]) -> list[dict[str, Any]]:
    coverage = manifest["coverage"]
    raw = coverage.get("missing_ranges")
    if not isinstance(raw, list) or not raw:
        raise DataReadinessError("temporal acquisition input has no missing ranges")
    ranges: list[dict[str, Any]] = []
    total = 0
    previous_end: date | None = None
    for record in raw:
        if not isinstance(record, dict):
            raise DataReadinessError("temporal missing-range record is invalid")
        try:
            start = date.fromisoformat(str(record["first_session"]))
            end = date.fromisoformat(str(record["last_session"]))
            sessions = int(record["sessions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataReadinessError("temporal missing-range record is invalid") from exc
        if start > end or sessions < 1 or (previous_end is not None and start <= previous_end):
            raise DataReadinessError("temporal missing ranges are unordered or empty")
        ranges.append(
            {
                "first_session": start.isoformat(),
                "last_session": end.isoformat(),
                "sessions": sessions,
            }
        )
        total += sessions
        previous_end = end
    if total != int(coverage.get("target_sessions_missing", -1)):
        raise DataReadinessError("temporal missing-range count is inconsistent")
    return ranges


def _load_membership_authority(
    *,
    membership_directory: Path,
    raw_archive_directory: Path,
    event_authority_directory: Path,
    transition_authority_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    security_exclusions_path: Path | None,
) -> tuple[pd.DataFrame, dict[str, Any], dict[str, Any]]:
    request_path = membership_directory / "_request.json"
    manifest_path = membership_directory / "_manifest.json"
    authority_path = membership_directory / "_authority.json"
    request = _read_json(request_path)
    if request.get("schema") != MEMBERSHIP_REQUEST_SCHEMA:
        raise DataReadinessError("membership acquisition input is not a supported authority")
    try:
        start_date = date.fromisoformat(str(request["start_date"]))
        cutoff_date = date.fromisoformat(str(request["cutoff_date"]))
        maximum_exclusion_fraction = float(request["maximum_security_exclusion_fraction"])
    except (KeyError, TypeError, ValueError) as exc:
        raise DataReadinessError("membership authority request window is invalid") from exc
    frame = require_sp500_membership_authority(
        membership_directory,
        archive_directory=raw_archive_directory,
        event_directory=event_authority_directory,
        transition_directory=transition_authority_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        start_date=start_date,
        cutoff_date=cutoff_date,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_exclusion_fraction,
    )
    manifest = _read_json(manifest_path)
    parent_lineage = manifest.get("parent_lineage")
    membership_artifact = manifest.get("membership_artifact")
    if not isinstance(parent_lineage, dict) or not isinstance(membership_artifact, dict):
        raise DataReadinessError("membership authority lineage inventory is invalid")
    if int(manifest.get("benchmark_session_exclusions", -1)) != 0:
        raise DataReadinessError("membership authority excludes benchmark sessions")
    metadata: dict[str, Any] = {
        "authority_start": start_date,
        "authority_cutoff": cutoff_date,
        "security_count": int(manifest.get("security_count", -1)),
        "excluded_security_count": int(manifest.get("excluded_security_count", -1)),
        "excluded_security_fraction": float(manifest.get("excluded_security_fraction", -1.0)),
        "universe_sha256": str(manifest.get("universe_sha256", "")),
        "parent_lineage": parent_lineage,
    }
    if metadata["security_count"] < 1 or len(metadata["universe_sha256"]) != 64:
        raise DataReadinessError("membership authority semantic identity is invalid")
    hashes: dict[str, Any] = {
        "request_sha256": file_sha256(request_path),
        "manifest_sha256": file_sha256(manifest_path),
        "authority_sha256": file_sha256(authority_path),
        "membership_artifact_sha256": str(membership_artifact.get("sha256", "")),
        "universe_sha256": metadata["universe_sha256"],
        "parent_lineage": parent_lineage,
    }
    return frame, metadata, hashes


def _build_daily_bar_units(
    memberships: pd.DataFrame,
    missing_ranges: list[dict[str, Any]],
) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for missing in missing_ranges:
        range_start = date.fromisoformat(str(missing["first_session"]))
        range_end = date.fromisoformat(str(missing["last_session"]))
        for row in memberships.to_dict(orient="records"):
            membership_start = _eastern_date(row["effective_from_utc"], field="effective_from_utc")
            raw_end = row.get("effective_to_utc")
            membership_end = range_end if pd.isna(raw_end) else _eastern_date(raw_end, field="effective_to_utc") - timedelta(days=1)
            unit_start = max(range_start, membership_start)
            unit_end = min(range_end, membership_end)
            if unit_start <= unit_end:
                records.append(
                    {
                        "security_id": str(row["security_id"]),
                        "ticker": str(row["ticker"]),
                        "start_date": unit_start.isoformat(),
                        "end_date": unit_end.isoformat(),
                        "role": "stock",
                    }
                )
        benchmarks = {
            "SPY",
            "QQQ",
            *memberships["primary_benchmark"].astype(str).str.strip(),
        }
        for ticker in sorted(benchmarks):
            if not ticker:
                raise DataReadinessError("membership authority has an empty benchmark")
            records.append(
                {
                    "security_id": f"benchmark:{ticker}",
                    "ticker": ticker,
                    "start_date": range_start.isoformat(),
                    "end_date": range_end.isoformat(),
                    "role": "benchmark",
                }
            )
    units = pd.DataFrame(
        records,
        columns=["security_id", "ticker", "start_date", "end_date", "role"],
    ).drop_duplicates()
    if units.empty or not bool(units["role"].eq("stock").any()):
        raise DataReadinessError("membership authority produces no stock acquisition units")
    return units.sort_values(
        ["role", "ticker", "start_date", "security_id"],
        kind="stable",
    ).reset_index(drop=True)


def _load_daily_collection(root: Path, directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    request_path = directory / "_request.json"
    status_path = directory / "_status.json"
    manifest_path = directory / "_manifest.json"
    request = _read_json(request_path)
    status = _read_json(status_path)
    manifest = _read_json(manifest_path)
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request_hash = hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
    if (
        request.get("schema") != DAILY_REQUEST_SCHEMA
        or request.get("request_sha256") != request_hash
        or request.get("source") != "alpaca"
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
        or request.get("timeframe") != "1d"
        or status.get("schema") != DAILY_MANIFEST_SCHEMA
        or status.get("status") not in {"complete", "complete_with_gaps"}
        or status.get("request_sha256") != request_hash
        or manifest.get("schema") != DAILY_MANIFEST_SCHEMA
        or manifest.get("request_sha256") != request_hash
    ):
        raise DataReadinessError("current daily collection is incomplete or unsupported")
    ledger = _bound_file(root, Path(str(status.get("source_collections_path", ""))))
    if file_sha256(ledger) != str(status.get("source_collections_sha256", "")):
        raise DataReadinessError("current daily source ledger hash mismatch")
    return {
        "start_date": str(request.get("start_date")),
        "end_date": str(request.get("end_date")),
        "total_rows": int(manifest.get("total_rows", -1)),
    }, {
        "daily_request_identity_sha256": request_hash,
        "daily_request_file_sha256": file_sha256(request_path),
        "daily_status_sha256": file_sha256(status_path),
        "daily_manifest_sha256": file_sha256(manifest_path),
    }


def _validate_coverage(
    *,
    memberships: pd.DataFrame,
    authority_start: date,
    authority_cutoff: date,
    daily: dict[str, Any],
    missing_start: date,
    missing_end: date,
) -> None:
    membership_start = _membership_start(memberships)
    daily_start = date.fromisoformat(str(daily["start_date"]))
    if authority_start > missing_start or membership_start > missing_start:
        raise DataReadinessError("membership authority does not cover the missing-history start")
    if authority_cutoff < missing_end:
        raise DataReadinessError("membership authority does not cover the missing-history end")
    if missing_end >= daily_start:
        raise DataReadinessError("missing temporal range overlaps reusable daily history")
    if int(daily["total_rows"]) < 1:
        raise DataReadinessError("current daily collection has no reusable rows")


def _membership_start(frame: pd.DataFrame) -> date:
    values = pd.to_datetime(frame["effective_from_utc"], utc=True, errors="coerce")
    if bool(values.isna().any()):
        raise DataReadinessError("membership effective start is invalid")
    return _eastern_date(values.min(), field="effective_from_utc")


def _eastern_date(value: object, *, field: str) -> date:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"membership {field} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise DataReadinessError(f"membership {field} is invalid")
    eastern = timestamp.tz_convert(EASTERN)
    return date(eastern.year, eastern.month, eastern.day)


def _bound_file(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise DataReadinessError(f"input escapes repository root: {path}")
    if not resolved.is_file():
        raise DataReadinessError(f"required input is missing: {resolved}")
    return resolved


def _bound_directory(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root != resolved and root not in resolved.parents:
        raise DataReadinessError(f"input escapes repository root: {path}")
    if not resolved.is_dir():
        raise DataReadinessError(f"required directory is missing: {resolved}")
    return resolved


def _bound_output(root: Path, path: Path) -> Path:
    resolved = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root not in resolved.parents:
        raise DataReadinessError(f"output escapes repository root: {path}")
    return resolved


def _read_json(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"JSON artifact must be an object: {path}")
    return payload


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def _guard(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MAX_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )
