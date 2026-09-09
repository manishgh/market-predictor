"""Select immutable daily source segments without merging issuers or inventing rows."""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any
from uuid import uuid4

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.datasets.symbol_corrections import (
    ArchiveLoader,
    SymbolCorrectionPolicy,
    inside,
    pinned_object,
)
from market_predictor.swing.labels.holding_paths import holding_calendar, validate_outcome_observations


def _segment(archive: Path, artifact: dict[str, Any], start: date, end: date, *, replacement: bool) -> dict[str, Any]:
    path = resolve_inside_authority(archive, artifact["bars_path"])
    if file_sha256(path) != artifact["bars_sha256"]:
        raise DataReadinessError("selected source bar file differs from replayed artifact")
    frame = pd.read_parquet(path)
    frame = frame.loc[frame.session_date.between(str(start), str(end))].copy()
    dates = frame.session_date.astype(str).tolist()
    expected = [day.isoformat() for day in holding_calendar(start, end)]
    if len(dates) != len(set(dates)) or dates != sorted(dates) or set(dates).difference(expected):
        raise DataReadinessError("selected source has duplicate, unsorted or unexpected exchange sessions")
    missing = sorted(set(expected).difference(dates))
    if replacement and missing:
        raise DataReadinessError("corrected source is missing sessions; parent fallback is prohibited")
    # Calendar clocks are validation coordinates only, not an assertion of execution
    # or historical first-observation time. Provider midnight remains in the raw file.
    if not frame.empty:
        schedule = xcals.get_calendar("XNYS").schedule.reindex(pd.DatetimeIndex(dates))
        validation = frame.rename(columns={"session_date": "session_date_et"}).copy()
        validation["bar_start_utc"] = schedule.open.array
        validation["bar_end_utc"] = schedule.close.array
        validation["available_at_utc"] = frame.ingested_at_utc
        validated = validate_outcome_observations(validation)
        invalid = int((~validated.outcome_observation_valid).sum())
    else:
        invalid = 0
    if replacement and invalid:
        raise DataReadinessError("corrected source contains unusable OHLCV observations")
    if file_sha256(path) != artifact["bars_sha256"]:
        raise DataReadinessError("selected source changed during observation validation")
    return {"archive": str(archive), "artifact": artifact, "first_session": str(start), "last_session": str(end),
        "selection": "corrected" if replacement else "parent", "rows": len(dates),
        "sessions_sha256": json_sha256(dates), "required_sessions": len(expected), "missing_sessions": missing,
        "invalid_observations": invalid}


def reconstruct_symbol_corrected_sources(*, root: Path, correction_plan: Path, correction_plan_sha256: str,
    correction_archive: Path, correction_archive_sha256: str, loader: ArchiveLoader) -> dict[str, Any]:
    plan, archive = inside(root, correction_plan), inside(root, correction_archive)
    plan_authority = pinned_object(plan / "_authority.json", correction_plan_sha256)
    pinned_object(archive / "_authority.json", correction_archive_sha256)
    corrected = loader(archive, plan_directory=plan, expected_adjustment="raw",
        expected_plan_authority_sha256=correction_plan_sha256)
    request = pinned_object(plan / "_request.json", plan_authority["request_sha256"])
    policy = SymbolCorrectionPolicy.model_validate(request["policy"])
    parent_root = inside(root, policy.parent_archive)
    pinned_object(parent_root / "_authority.json", policy.parent_archive_sha256)
    parent = pinned_object(parent_root / "_manifest.json", request["parent_archive_manifest_sha256"])
    corrections = {c.parent_unit_id: c for c in policy.corrections}
    replacements = {item["unit_id"]: item for item in corrected["unit_artifacts"]}
    corrected_ids = {item["parent_unit_id"]: item["unit_id"] for item in request["replacements"]}
    segments: list[dict[str, Any]] = []
    discarded_rows = 0
    for item in parent["unit_artifacts"]:
        assert_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="symbol corrected source selection")
        start, end = date.fromisoformat(item["start_date"]), date.fromisoformat(item["end_date"])
        correction = corrections.get(item["unit_id"])
        if correction is None:
            segments.append(_segment(parent_root, item, start, end, replacement=False))
            continue
        kept = 0
        for lower, upper in ((start, correction.start_date - timedelta(days=1)),
            (correction.end_date + timedelta(days=1), end)):
            if lower <= upper:
                segment = _segment(parent_root, item, lower, upper, replacement=False)
                kept += segment["rows"]
                segments.append(segment)
        discarded_rows += int(item["rows"]) - kept
        replacement = replacements[corrected_ids[item["unit_id"]]]
        segments.append(_segment(archive, replacement, correction.start_date, correction.end_date, replacement=True))
    pinned_object(plan / "_authority.json", correction_plan_sha256)
    pinned_object(plan / "_request.json", plan_authority["request_sha256"])
    pinned_object(archive / "_authority.json", correction_archive_sha256)
    result = {"schema": "market_predictor.swing_symbol_corrected_sources", "status": "source_selection_verified",
        "correction_plan": str(plan), "correction_plan_sha256": correction_plan_sha256,
        "correction_archive": str(archive), "correction_archive_sha256": correction_archive_sha256,
        "parent_plan_sha256": policy.parent_plan_sha256, "parent_archive_sha256": policy.parent_archive_sha256,
        "policy_sha256": request["policy_sha256"], "segments": segments, "discarded_parent_rows": discarded_rows,
        "selected_rows": sum(s["rows"] for s in segments),
        "missing_sessions": sum(len(s["missing_sessions"]) for s in segments),
        "invalid_observations": sum(s["invalid_observations"] for s in segments),
        "exclusions_added": [], "accounting_eligible": False, "label_eligible": False, "promotion_eligible": False,
        "historical_availability_proven": False,
        "derived_artifacts": "rebuild_labels_features_news_joins_before_use_no_parent_derived_fallback"}
    return {**result, "audit_sha256": json_sha256(result)}


def publish_or_verify_symbol_corrected_sources(output: Path, result: dict[str, Any],
    *, expected_sha256: str | None = None) -> None:
    """Publish once atomically, or compare a full reconstruction to an external pin."""
    output.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(output, timeout=0.0):
        if output.exists():
            if expected_sha256 is None or pinned_object(output, expected_sha256) != result:
                raise DataReadinessError("corrected source selection requires identical replay and independent file pin")
            return
        if expected_sha256 is not None:
            raise DataReadinessError("cannot replace a missing pinned source selection")
        staging = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
        try:
            write_json_object(staging, result)
            staging.rename(output)
        finally:
            staging.unlink(missing_ok=True)
