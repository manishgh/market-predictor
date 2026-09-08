"""Bounded identity inventory for a retrospective whole-security research restriction.

Old panels supply identity coverage only, never admitted features or outcomes.
The supplied restriction is frozen independently of numeric performance.
"""
from __future__ import annotations

import hashlib
import json
import os
import tomllib
from collections import Counter
from pathlib import Path
from typing import Any, Literal
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget, release_process_memory
from market_predictor.swing.contracts.research_cohort import ResearchSecurityExclusion, Sha256, SwingResearchCohort


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_research_cohort_request"] = Field(alias="schema")
    parent_request_path: str = Field(min_length=1)
    parent_request_sha256: Sha256
    membership_path: str = Field(min_length=1)
    membership_sha256: Sha256
    parent_manifest_path: str = Field(min_length=1)
    parent_manifest_sha256: Sha256
    control_report_path: str = Field(min_length=1)
    control_report_sha256: Sha256
    maximum_exclusion_bps: int = Field(ge=0, le=10000)
    cap_approval_reference: str = Field(min_length=1)
    exclusions: tuple[ResearchSecurityExclusion, ...] = Field(min_length=1)


def _guard(stage: str) -> None:
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage=stage)
    assert_peak_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage=stage)


def _inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise DataReadinessError(f"cohort path escapes repository: {path}")
    return resolved


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("cohort metadata exceeds bounded size")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _checked(root: Path, path: str, expected: str, bound: dict[str, str]) -> Path:
    resolved = resolve_inside_authority(root, path)
    if file_sha256(resolved) != expected:
        raise DataReadinessError(f"cohort source hash differs: {path}")
    relative = resolved.relative_to(root).as_posix()
    if relative in bound and bound[relative] != expected:
        raise DataReadinessError("cohort source has conflicting bindings")
    bound[relative] = expected
    return resolved


def _ids(payload: dict[str, Any], prefix: str) -> tuple[str, ...]:
    values = payload.get(f"{prefix}_security_ids")
    if (not isinstance(values, list) or any(not isinstance(x, str) or not x or x.strip() != x for x in values)
            or values != sorted(set(values))
            or type(payload.get(f"{prefix}_security_count")) is not int
            or payload[f"{prefix}_security_count"] != len(values)
            or payload.get(f"{prefix}_security_ids_sha256") != json_sha256(values)):
        raise DataReadinessError(f"cohort parent {prefix} identity count/hash differs")
    return tuple(values)


def _sources(root: Path, request: _Request, bound: dict[str, str]) -> tuple[dict[str, Any], dict[str, Any], Path]:
    # Verify all pins before opening even the membership identity projection.
    paths = {
        name: _checked(root, getattr(request, f"{name}_path"), getattr(request, f"{name}_sha256"), bound)
        for name in ("parent_request", "membership", "parent_manifest", "control_report")
    }
    parent, manifest, control = (_object(paths[name]) for name in ("parent_request", "parent_manifest", "control_report"))
    if parent.get("request_sha256") != json_sha256({k: v for k, v in parent.items() if k != "request_sha256"}):
        raise DataReadinessError("cohort parent request canonical hash differs")
    if manifest.get("request_sha256") != parent["request_sha256"]:
        raise DataReadinessError("cohort parent manifest request differs")
    if (control.get("audit_sha256") != json_sha256({k: v for k, v in control.items() if k != "audit_sha256"})
            or control.get("scope") != "initial_fit_deterministic_control_not_out_of_sample"
            or control.get("validation_or_test_outcomes_read") is not False):
        raise DataReadinessError("cohort control evidence identity or development scope differs")
    control_files = control.get("bound_files")
    if not isinstance(control_files, dict) or any(
        control_files.get(paths[name].relative_to(root).as_posix()) != getattr(request, f"{name}_sha256")
        for name in ("parent_request", "parent_manifest")
    ):
        raise DataReadinessError("cohort control does not bind the parent panel")
    combined = parent.get("combined_daily_inputs")
    if (not isinstance(combined, dict) or not isinstance(combined.get("membership_authority"), dict)
            or combined["membership_authority"].get("membership_artifact_sha256") != request.membership_sha256):
        raise DataReadinessError("cohort membership is not bound to the parent request")
    return parent, manifest, paths["membership"]


def _cohort(request: _Request, parent: dict[str, Any], membership_path: Path, bound: dict[str, str]) -> SwingResearchCohort:
    combined = parent["combined_daily_inputs"]
    warmup = _ids(parent, "warmup_only")
    if warmup != _ids(combined, "warmup_only"):
        raise DataReadinessError("cohort parent warm-up identities disagree")
    inherited = _ids(combined, "excluded")
    _guard("cohort membership projection")
    memberships = pd.read_parquet(membership_path, columns=["security_id", "ticker"])
    try:
        if (memberships.empty or memberships.isna().any().any()
                or any(not memberships[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                       for name in ("security_id", "ticker"))):
            raise DataReadinessError("cohort membership identity is missing")
        all_ids = set(memberships["security_id"])
        if not set(warmup).issubset(all_ids):
            raise DataReadinessError("cohort warm-up identity absent from memberships")
        original = tuple(sorted(all_ids.difference(warmup)))
        for entry in request.exclusions:
            tickers = set(memberships.loc[memberships["security_id"].eq(entry.security_id), "ticker"])
            if not tickers or not set(entry.tickers).issubset(tickers):
                raise DataReadinessError(f"cohort exclusion unknown identity or ticker display: {entry.security_id}")
        cohort = SwingResearchCohort(
            schema_version="market_predictor.swing_research_cohort", scope="retrospective_development_restriction",
            price_basis_status="not_certified_by_cohort", combined_daily_inputs_sha256=json_sha256(combined),
            original_security_ids=original, inherited_excluded_security_ids=inherited,
            warmup_only_security_ids=warmup, exclusions=request.exclusions,
            maximum_exclusion_bps=request.maximum_exclusion_bps, cap_approval_reference=request.cap_approval_reference,
            source_files=dict(bound),
        )
        retained = tuple(sorted(set(original).difference(inherited)))
        cohort.assert_source_matches(combined, retained, warmup)
        if (parent.get("modeled_security_count") != len(retained) or parent.get("security_count") != len(retained)
                or parent.get("modeled_security_ids_sha256") != json_sha256(list(retained))
                or parent.get("security_ids_sha256") != json_sha256(list(retained))):
            raise DataReadinessError("cohort parent retained identity count/hash differs")
        return cohort
    finally:
        del memberships
        release_process_memory()


def _partition_counts(path: Path, record: dict[str, Any], parent_ids: set[str], excluded: set[str]) -> tuple[
    Counter[tuple[int, str]], Counter[tuple[int, str]], set[str],
]:
    _guard("cohort monthly identity projection")
    # ParquetFile avoids implicit Hive columns and can only decode this explicit projection.
    arrow: Any = pq
    with arrow.ParquetFile(path) as source:
        if source.metadata.num_rows > 1_000_000:
            raise DataReadinessError("cohort monthly identity partition exceeds bounded row limit")
        frame = source.read(columns=["security_id", "session_date_et", "sector"], use_threads=False).to_pandas()
    try:
        _guard("cohort projected monthly identities")
        if (frame.empty or frame.isna().any().any() or type(record.get("rows")) is not int
                or len(frame) != record["rows"]
                or any(not frame[name].map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                       for name in ("security_id", "sector"))):
            raise DataReadinessError("cohort partition rows or identities differ")
        sessions = frame["session_date_et"].astype(str)
        parsed = pd.to_datetime(sessions, format="%Y-%m-%d", errors="coerce")
        observed = set(frame["security_id"])
        if (parsed.isna().any() or not parsed.dt.strftime("%Y-%m-%d").eq(sessions).all()
                or not sessions.str.slice(0, 7).eq(record.get("partition_month")).all()
                or not observed.issubset(parent_ids)
                or frame.duplicated(["security_id", "session_date_et"]).any()
                or sessions.min() != record.get("first_session") or sessions.max() != record.get("last_session")
                or sessions.nunique() != record.get("sessions") or len(observed) != record.get("securities")):
            raise DataReadinessError("cohort monthly session/identity inventory differs")
        original: Counter[tuple[int, str]] = Counter()
        removed: Counter[tuple[int, str]] = Counter()
        for security, year, sector in zip(frame["security_id"], parsed.dt.year, frame["sector"], strict=True):
            key = (int(year), str(sector))
            original[key] += 1
            if security in excluded:
                removed[key] += 1
        return original, removed, observed
    finally:
        del frame
        release_process_memory()


def _coverage(root: Path, request: _Request, manifest: dict[str, Any], cohort: SwingResearchCohort,
              bound: dict[str, str]) -> dict[str, Any]:
    records = manifest.get("files")
    if (not isinstance(records, list) or not records or len(records) > 240
            or manifest.get("feature_profiles") != ["technical_market"]):
        raise DataReadinessError("cohort requires a bounded single-profile monthly parent inventory")
    parent_ids = set(cohort.original_security_ids).difference(cohort.inherited_excluded_security_ids)
    original: Counter[tuple[int, str]] = Counter()
    removed: Counter[tuple[int, str]] = Counter()
    observed: set[str] = set()
    months: set[str] = set()
    paths: set[Path] = set()
    for record in records:
        if (not isinstance(record, dict) or record.get("feature_profile") != "technical_market"
                or not isinstance(record.get("partition_month"), str) or record["partition_month"] in months
                or not isinstance(record.get("path"), str) or not isinstance(record.get("sha256"), str)):
            raise DataReadinessError("cohort duplicate or malformed monthly partition")
        path = _checked(root, str(Path(request.parent_manifest_path).parent / record["path"]), record["sha256"], bound)
        if path in paths:
            raise DataReadinessError("cohort partition path is duplicated")
        paths.add(path)
        months.add(record["partition_month"])
        counts, excluded_counts, ids = _partition_counts(path, record, parent_ids, set(cohort.excluded_security_ids))
        original.update(counts)
        removed.update(excluded_counts)
        observed.update(ids)
    if (observed != parent_ids or sum(original.values()) != manifest.get("rows")
            or manifest.get("securities") != len(parent_ids) or manifest.get("modeled_security_count") != len(parent_ids)
            or manifest.get("modeled_security_ids_sha256") != json_sha256(sorted(parent_ids))):
        raise DataReadinessError("cohort manifest rows or observed retained identities differ")
    return {
        "scope": "parent_identity_inventory_only_not_training_authority",
        "restriction_basis": "supplied_whole_security_research_restriction_not_performance_filter",
        "outcome_columns_read": False, "numeric_features_read": False, "heldout_outcomes_read": False,
        "original_rows": sum(original.values()), "removed_rows": sum(removed.values()),
        "retained_rows": sum(original.values()) - sum(removed.values()),
        "row_denominator": "observed_parent_rows_after_inherited_coverage_exclusions",
        "observed_parent_securities": len(observed), "partitions": len(records),
        "by_calendar_year_and_sector": [
            {"calendar_year": year, "sector": sector, "original_rows": count,
             "removed_rows": removed[(year, sector)], "retained_rows": count - removed[(year, sector)]}
            for (year, sector), count in sorted(original.items())
        ],
    }


def run_swing_research_cohort_audit(root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    """Publish a frozen, bounded cohort audit; cap failure is a report, not admission."""
    root = root.resolve()
    config_path, output_path = _inside(root, config_path), _inside(root, output_path)
    runtime = _inside(root, heavy_job_runtime_dir())
    with heavy_job_lease("audit-swing-research-cohort", runtime_dir=runtime):
        try:
            _guard("cohort input loading")
            if config_path.stat().st_size > 1024**2:
                raise DataReadinessError("cohort configuration exceeds bounded size")
            config_bytes = config_path.read_bytes()
            request = _Request.model_validate_json(json.dumps(tomllib.loads(config_bytes.decode("utf-8"))))
            bound = {config_path.relative_to(root).as_posix(): hashlib.sha256(config_bytes).hexdigest()}
            parent, manifest, membership_path = _sources(root, request, bound)
            cohort = _cohort(request, parent, membership_path, bound)
            coverage = _coverage(root, request, manifest, cohort, bound)
            cohort = cohort.model_copy(update={"source_files": dict(bound)})
            envelope = {"cohort": cohort.model_dump(mode="json"), "cohort_sha256": cohort.sha256(),
                        "summary": cohort.summary(), "coverage": coverage}
            envelope["audit_sha256"] = json_sha256(envelope)
            if output_path.relative_to(root).as_posix() in bound:
                raise DataReadinessError("cohort output cannot replace an input")
            for relative, digest in tuple(bound.items()):
                _checked(root, relative, digest, bound)
            _guard("cohort publication")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(output_path, timeout=0.0):
                if output_path.exists():
                    if _object(output_path) != envelope:
                        raise DataReadinessError("immutable cohort output conflicts with this request")
                    return envelope
                staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
                try:
                    write_json_object(staging, envelope)
                    os.rename(staging, output_path)
                finally:
                    staging.unlink(missing_ok=True)
            return envelope
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DataReadinessError(f"invalid swing research cohort audit: {exc}") from exc
