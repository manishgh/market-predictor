"""Outcome-blind acquisition planning for missing swing history."""

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

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.swing.market_history import DEFAULT_BENCHMARKS
from market_predictor.v3.errors import DataReadinessError

PLAN_SCHEMA = "edge_rebuild.swing_history_acquisition_plan.v1"
AUTHORITY_SCHEMA = "edge_rebuild.swing_history_acquisition_plan_authority.v1"
TEMPORAL_SCHEMA = "edge_rebuild.temporal_manifest.v1"
TEMPORAL_AUTHORITY_SCHEMA = "edge_rebuild.temporal_manifest_authority.v1"
UNIVERSE_AUDIT_SCHEMA = "ml_v3.sp500_point_in_time_universe.v1"
SOURCE_MANIFEST_SCHEMA = "ml_v3.sp500_change_sources.v1"
DAILY_REQUEST_SCHEMA = "swing.daily_history_collection.v1"
DAILY_MANIFEST_SCHEMA = "swing.daily_history_manifest.v1"
MAX_MEMORY_GIB = 4.0
MEMORY_HEADROOM_GIB = 0.75
ANNOUNCEMENT_LEAD_DAYS = 45
EASTERN = ZoneInfo("America/New_York")


def publish_swing_history_acquisition_plan(
    *,
    repository_root: Path,
    temporal_manifest_directory: Path,
    memberships_path: Path,
    universe_audit_path: Path,
    current_daily_collection_directory: Path,
    output_directory: Path,
) -> dict[str, Any]:
    """Publish membership-first acquisition scope without reading outcomes."""

    root = repository_root.resolve()
    temporal_dir = _bound_directory(root, temporal_manifest_directory)
    memberships = _bound_file(root, memberships_path)
    universe_audit = _bound_file(root, universe_audit_path)
    daily_dir = _bound_directory(root, current_daily_collection_directory)
    output = _bound_output(root, output_directory)
    if output.exists():
        raise DataReadinessError(f"swing acquisition-plan output must be new: {output}")
    _guard("swing acquisition-plan start")

    temporal, temporal_hashes = _load_temporal(temporal_dir)
    missing_ranges = _missing_ranges(temporal)
    missing_start = date.fromisoformat(str(missing_ranges[0]["first_session"]))
    missing_end = date.fromisoformat(str(missing_ranges[-1]["last_session"]))
    membership_frame, membership_hashes = _load_memberships(memberships)
    universe, universe_hashes = _load_universe_audit(root, universe_audit)
    daily, daily_hashes = _load_daily_collection(root, daily_dir)
    _validate_current_coverage(membership_frame, universe, daily, missing_end)
    _guard("swing acquisition-plan evidence")

    membership_start = _membership_start(membership_frame)
    universe_start = date.fromisoformat(str(universe["start_date"]))
    source_ready = int(universe["invalid_source_count"]) == 0
    membership_dates_ready = (
        membership_start <= missing_start and universe_start <= missing_start
    )
    membership_ready = source_ready and membership_dates_ready
    units = (
        _build_daily_units(membership_frame, missing_start, missing_end)
        if membership_ready
        else pd.DataFrame(
            columns=["security_id", "ticker", "start_date", "end_date", "role"]
        )
    )
    status = (
        "official_source_reacquisition_required"
        if not source_ready
        else "ready_for_daily_bar_collection"
        if membership_ready
        else "membership_evidence_required"
    )
    request = {
        "schema": PLAN_SCHEMA,
        "temporal_manifest_sha256": temporal_hashes["manifest"],
        "temporal_authority_sha256": temporal_hashes["authority"],
        **membership_hashes,
        **universe_hashes,
        **daily_hashes,
    }
    manifest: dict[str, Any] = {
        "schema": PLAN_SCHEMA,
        "status": status,
        "outcomes_read": False,
        "missing_session_ranges": missing_ranges,
        "membership": {
            "current_membership_start": membership_start.isoformat(),
            "current_universe_audit_start": universe_start.isoformat(),
            "required_start": missing_start.isoformat(),
            "required_end": missing_end.isoformat(),
            "official_announcement_discovery_start": (
                missing_start - timedelta(days=ANNOUNCEMENT_LEAD_DAYS)
            ).isoformat(),
            "official_announcement_discovery_end": missing_end.isoformat(),
            "rebuild_cutoff": str(universe["cutoff_date"]),
            "reusable_official_sources": universe["source_count"],
            "total_official_sources": universe["total_source_count"],
            "reusable_official_source_bytes": universe["source_bytes"],
            "invalid_official_sources": universe["invalid_source_count"],
            "invalid_official_source_records": universe["invalid_sources"],
            "required_evidence": [
                "official S&P Dow Jones Indices constituent-change releases",
                "Alpaca corporate-action symbol transitions",
                "reviewed primary-source security transitions",
            ],
        },
        "daily_bars": {
            "status": (
                "blocked_until_source_reacquisition"
                if not source_ready
                else "ready"
                if membership_ready
                else "blocked_until_membership_authority"
            ),
            "planned_units": len(units),
            "source": "alpaca",
            "timeframe": "1Day",
            "price_feed": "sip",
            "adjustment": "all",
            "unit_policy": (
                "verified security/ticker membership interval intersected with "
                "the missing session range"
            ),
            "current_collection_total_rows": daily["total_rows"],
            "current_collection_reused": str(daily_dir.relative_to(root)).replace(
                "\\", "/"
            ),
            "refusal_reason": (
                None
                if membership_ready
                else "official source files fail their declared SHA-256 identities"
                if not source_ready
                else "historical ticker/date ownership is not yet authoritative"
            ),
        },
        "next_operations": (
            [
                "reacquire all invalid official S&P releases into an immutable archive",
                "rebuild point-in-time membership from hash-valid source evidence",
                "rerun this planner before any market-data request",
            ]
            if not source_ready
            else
            [
                "collect only missing official membership releases and transitions",
                "rebuild and verify point-in-time membership",
                "rerun this planner to publish immutable daily-bar units",
            ]
            if not membership_ready
            else [
                "collect the published Alpaca SIP/all units sequentially",
                "verify whole-security exclusions remain at or below 5%",
                "rebuild the swing panel and rerun temporal coverage",
            ]
        ),
    }

    staging = output.with_name(f".{output.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=True)
        _write_json(staging / "_request.json", request)
        manifest["request_sha256"] = file_sha256(staging / "_request.json")
        if membership_ready:
            units_path = staging / "daily_bar_units.csv"
            units.to_csv(units_path, index=False, lineterminator="\n")
            manifest["daily_bars"]["units_sha256"] = file_sha256(units_path)
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


def _load_memberships(path: Path) -> tuple[pd.DataFrame, dict[str, str]]:
    sidecar = manifest_path_for(path)
    manifest = _read_json(sidecar)
    if (
        manifest.get("schema") != "market_data.artifact_manifest.v1"
        or manifest.get("artifact_type") != "memberships"
        or manifest.get("artifact_sha256") != file_sha256(path)
    ):
        raise DataReadinessError("membership acquisition input is not authoritative")
    columns = [
        "security_id",
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
        "primary_benchmark",
    ]
    frame = pd.read_parquet(path, columns=columns)
    if frame.empty or len(frame) != int(manifest.get("rows", -1)):
        raise DataReadinessError("membership acquisition input has invalid rows")
    for column in ("security_id", "ticker"):
        values = frame[column].astype("string").str.strip()
        if bool(values.isna().any()) or bool(values.eq("").any()):
            raise DataReadinessError(f"membership input has empty {column}")
    return frame, {
        "memberships_sha256": file_sha256(path),
        "membership_manifest_sha256": file_sha256(sidecar),
    }


def _load_universe_audit(root: Path, path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    audit = _read_json(path)
    source_manifest = audit.get("source_manifest")
    if (
        audit.get("schema") != UNIVERSE_AUDIT_SCHEMA
        or not isinstance(source_manifest, dict)
        or source_manifest.get("schema") != SOURCE_MANIFEST_SCHEMA
    ):
        raise DataReadinessError("universe acquisition audit is unsupported")
    sources = source_manifest.get("sources")
    if not isinstance(sources, list) or not sources:
        raise DataReadinessError("universe acquisition audit has no official sources")
    source_bytes = 0
    archive_records: list[dict[str, str]] = []
    invalid_sources: list[dict[str, str]] = []
    for source in sources:
        if not isinstance(source, dict):
            raise DataReadinessError("official S&P source record is invalid")
        raw_path = _bound_file(root, Path(str(source.get("raw_path", ""))))
        expected = str(source.get("sha256", ""))
        actual = file_sha256(raw_path)
        record = {
            "path": str(raw_path.relative_to(root)).replace("\\", "/"),
            "expected_sha256": expected,
            "actual_sha256": actual,
        }
        archive_records.append(record)
        if actual == expected:
            source_bytes += raw_path.stat().st_size
        else:
            invalid_sources.append(record)
    anchor = _bound_file(root, Path(str(audit.get("anchor_source", ""))))
    transitions = audit.get("security_transition_evidence")
    if not isinstance(transitions, dict):
        raise DataReadinessError("universe acquisition audit lacks transitions")
    for path_key, hash_key in (
        ("provider_path", "provider_sha256"),
        ("reviewed_path", "reviewed_sha256"),
    ):
        evidence_path = _bound_file(root, Path(str(transitions.get(path_key, ""))))
        if file_sha256(evidence_path) != str(transitions.get(hash_key, "")):
            raise DataReadinessError(f"security transition hash mismatch: {evidence_path}")
    return {
        "start_date": str(audit.get("start_date")),
        "cutoff_date": str(audit.get("cutoff_date")),
        "source_count": len(sources) - len(invalid_sources),
        "total_source_count": len(sources),
        "source_bytes": source_bytes,
        "invalid_source_count": len(invalid_sources),
        "invalid_sources": invalid_sources,
    }, {
        "universe_audit_sha256": file_sha256(path),
        "anchor_file_sha256": file_sha256(anchor),
        "anchor_semantic_snapshot_sha256": str(audit.get("snapshot_sha256", "")),
        "official_archive_fingerprint": _json_sha256(archive_records),
    }


def _load_daily_collection(root: Path, directory: Path) -> tuple[dict[str, Any], dict[str, str]]:
    request_path = directory / "_request.json"
    status_path = directory / "_status.json"
    manifest_path = directory / "_manifest.json"
    request = _read_json(request_path)
    status = _read_json(status_path)
    manifest = _read_json(manifest_path)
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request_hash = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
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


def _validate_current_coverage(
    memberships: pd.DataFrame,
    universe: dict[str, Any],
    daily: dict[str, Any],
    missing_end: date,
) -> None:
    membership_start = _membership_start(memberships)
    universe_start = date.fromisoformat(str(universe["start_date"]))
    daily_start = date.fromisoformat(str(daily["start_date"]))
    if membership_start != universe_start:
        raise DataReadinessError("membership and universe-audit starts disagree")
    if missing_end >= daily_start:
        raise DataReadinessError("missing temporal range overlaps reusable daily history")
    if int(daily["total_rows"]) < 1:
        raise DataReadinessError("current daily collection has no reusable rows")


def _membership_start(frame: pd.DataFrame) -> date:
    values = pd.to_datetime(frame["effective_from_utc"], utc=True, errors="coerce")
    if bool(values.isna().any()):
        raise DataReadinessError("membership effective start is invalid")
    minimum = pd.Timestamp(values.min()).tz_convert(EASTERN)
    return date(int(minimum.year), int(minimum.month), int(minimum.day))


def _build_daily_units(frame: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    records: list[dict[str, str]] = []
    for row in frame.itertuples(index=False):
        effective_start = pd.Timestamp(row.effective_from_utc).tz_convert(EASTERN).date()
        effective_end = (
            None
            if pd.isna(row.effective_to_utc)
            else pd.Timestamp(row.effective_to_utc).tz_convert(EASTERN).date()
        )
        unit_start = max(start, effective_start)
        unit_end = min(end, effective_end - timedelta(days=1) if effective_end else end)
        if unit_start <= unit_end:
            records.append(
                {
                    "security_id": str(row.security_id),
                    "ticker": str(row.ticker),
                    "start_date": unit_start.isoformat(),
                    "end_date": unit_end.isoformat(),
                    "role": "stock",
                }
            )
    benchmarks = set(DEFAULT_BENCHMARKS) | set(
        frame["primary_benchmark"].dropna().astype("string").str.strip()
    )
    records.extend(
        {
            "security_id": f"benchmark:{ticker}",
            "ticker": ticker,
            "start_date": start.isoformat(),
            "end_date": end.isoformat(),
            "role": "benchmark",
        }
        for ticker in sorted(benchmarks - {""})
    )
    units = pd.DataFrame.from_records(records)
    if units.empty:
        raise DataReadinessError("extended membership produced no daily-bar units")
    duplicates = units.duplicated(
        subset=["security_id", "ticker", "start_date", "end_date", "role"]
    )
    if bool(duplicates.any()):
        raise DataReadinessError("daily-bar acquisition units are duplicated")
    return units.sort_values(
        ["role", "ticker", "start_date", "security_id"], kind="stable"
    ).reset_index(drop=True)


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


def _json_sha256(payload: object) -> str:
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def _guard(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MAX_MEMORY_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )
