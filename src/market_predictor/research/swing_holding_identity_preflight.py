"""Bounded, immutable membership-identity preflight; no prices or model outcomes."""
from __future__ import annotations

import json
import os
import tomllib
from collections import Counter
from datetime import date
from pathlib import Path
from typing import Any, Literal, Self
from uuid import uuid4

import pandas as pd
import pyarrow.parquet as pq
from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.joins import MEMBERSHIP_VALUE_COLUMNS
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.locking import file_lock
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget, release_process_memory
from market_predictor.swing.contracts.research_cohort import Sha256, load_swing_research_cohort
from market_predictor.swing.labels.holding_identity import inspect_holding_membership_windows
from market_predictor.swing.labels.holding_paths import holding_calendar

IDENTITY_COLUMNS = (
    "decision_id", "security_id", "ticker", "sector", "primary_benchmark", "session_date_et", "decision_time_utc",
)


class _Request(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)
    schema_version: Literal["market_predictor.swing_holding_identity_preflight_request"] = Field(alias="schema")
    cohort_path: str
    cohort_sha256: Sha256
    parent_manifest_path: str
    membership_path: str
    decision_start: date
    snapshot_end: date
    initial_fit_end: date
    horizon_sessions: Literal[10]

    @model_validator(mode="after")
    def validate_dates(self) -> Self:
        if not self.decision_start <= self.initial_fit_end <= self.snapshot_end:
            raise ValueError("holding identity snapshot/development dates are inconsistent")
        return self


def _guard(stage: str) -> None:
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage=stage)
    assert_peak_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage=stage)


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("holding identity metadata exceeds bounded size")
    return parse_strict_json_object(path.read_bytes(), label=str(path))


def _inside(root: Path, path: Path) -> Path:
    resolved = (root / path).resolve()
    if not resolved.is_relative_to(root):
        raise DataReadinessError(f"holding identity path escapes repository: {path}")
    return resolved


def _projection(path: Path, columns: tuple[str, ...]) -> pd.DataFrame:
    arrow: Any = pq
    with arrow.ParquetFile(path) as source:
        if source.metadata.num_rows > 1_000_000:
            raise DataReadinessError("holding identity projection exceeds bounded row limit")
        return source.read(columns=list(columns), use_threads=False).to_pandas()


def _counts(frame: pd.DataFrame) -> dict[str, int]:
    return {
        "decisions": len(frame),
        "terminal_immature": int(frame.terminal_immature.sum()),
        "covered_matured": int(frame.membership_holding_window_covered.sum()),
        "uncovered_matured": int(frame.uncovered_holding_sessions.gt(0).sum()),
        "uncovered_session_instances": int(frame.uncovered_holding_sessions.sum()),
    }


def _collect(root: Path, request: _Request, bound: dict[str, str]) -> dict[str, Any]:
    cohort_path = resolve_inside_authority(root, request.cohort_path)
    cohort = load_swing_research_cohort(cohort_path, source_root=root)
    if cohort.sha256() != request.cohort_sha256 or not cohort.within_cap:
        raise DataReadinessError("holding identity requires the pinned, accepted research cohort")
    bound.update(cohort.source_files)
    bound[cohort_path.relative_to(root).as_posix()] = file_sha256(cohort_path)

    def source(relative: str) -> Path:
        path = resolve_inside_authority(root, relative)
        key = path.relative_to(root).as_posix()
        if key not in cohort.source_files or file_sha256(path) != cohort.source_files[key]:
            raise DataReadinessError(f"holding identity source is not bound to cohort: {relative}")
        return path

    manifest_path = source(request.parent_manifest_path)
    manifest = _object(manifest_path)
    records = manifest.get("files")
    if (not isinstance(records, list) or not records or len(records) > 240
            or manifest.get("feature_profiles") != ["technical_market"]):
        raise DataReadinessError("holding identity requires bounded monthly technical identity inventory")
    memberships = _projection(source(request.membership_path), tuple(dict.fromkeys((
        "ticker", *MEMBERSHIP_VALUE_COLUMNS, "effective_from_utc", "effective_to_utc", "available_at_utc",
    ))))
    if set(memberships.security_id) != set(cohort.original_security_ids).union(cohort.warmup_only_security_ids):
        raise DataReadinessError("holding identity requires the full, unfiltered membership authority")
    sessions = holding_calendar(request.decision_start, request.snapshot_end)
    totals: Counter[str] = Counter()
    development: Counter[str] = Counter()
    affected: dict[str, dict[str, Any]] = {}
    observed: set[str] = set()
    decision_ids: set[str] = set()
    months: set[str] = set()
    partitions: list[dict[str, Any]] = []
    parent_rows = 0
    for record in records:
        if (not isinstance(record, dict) or not isinstance(record.get("path"), str)
                or not isinstance(record.get("partition_month"), str) or record["partition_month"] in months
                or record.get("feature_profile") != "technical_market"):
            raise DataReadinessError("holding identity duplicate or invalid monthly inventory")
        months.add(record["partition_month"])
        path = source((manifest_path.parent / record["path"]).relative_to(root).as_posix())
        if record.get("sha256") != file_sha256(path):
            raise DataReadinessError("holding identity manifest partition hash differs")
        _guard("holding identity monthly projection")
        frame = _projection(path, IDENTITY_COLUMNS)
        parent_rows += len(frame)
        if (not frame.decision_id.map(lambda value: isinstance(value, str) and bool(value.strip())).all()
                or frame.decision_id.duplicated().any() or decision_ids.intersection(frame.decision_id)):
            raise DataReadinessError("holding identity duplicate or invalid decision IDs across partitions")
        decision_ids.update(frame.decision_id)
        dates = pd.to_datetime(frame.session_date_et, errors="coerce")
        parent_ids = set(cohort.original_security_ids).difference(cohort.inherited_excluded_security_ids)
        if (frame.empty or frame.isna().any().any() or len(frame) != record.get("rows")
                or not set(frame.security_id).issubset(parent_ids) or dates.isna().any()
                or not dates.dt.strftime("%Y-%m").eq(record["partition_month"]).all()
                or str(dates.min().date()) != record.get("first_session")
                or str(dates.max().date()) != record.get("last_session")
                or dates.nunique() != record.get("sessions")
                or frame.security_id.nunique() != record.get("securities")
                or dates.dt.date.lt(request.decision_start).any() or dates.dt.date.gt(request.snapshot_end).any()):
            raise DataReadinessError("holding identity partition identities, rows or dates differ")
        observed.update(frame.security_id)
        frame = frame.loc[frame.security_id.isin(cohort.retained_security_ids)].reset_index(drop=True)
        if frame.empty:
            raise DataReadinessError("holding identity partition has no retained decisions")
        checked = inspect_holding_membership_windows(
            frame, memberships, sessions=sessions, horizon_sessions=request.horizon_sessions,
            initial_fit_end=request.initial_fit_end,
        )
        counts = _counts(checked)
        development_counts = _counts(checked.loc[checked.development_matured])
        totals.update(counts)
        development.update(development_counts)
        partitions.append({"month": record["partition_month"], "all_history": counts, "initial_fit": development_counts})
        failures = checked.loc[checked.uncovered_holding_sessions.gt(0)]
        for identity, group in failures.groupby("security_id", sort=True):
            key = str(identity)
            entry = affected.setdefault(key, {"security_id": key, "tickers": [], "decisions": 0,
                "initial_fit_decisions": 0, "first_decision": str(group.session_date_et.min()),
                "last_decision": str(group.session_date_et.max()), "uncovered_session_instances": 0})
            entry["tickers"] = sorted(set(entry["tickers"]).union(group.ticker))
            entry["decisions"] += len(group)
            entry["initial_fit_decisions"] += int(group.development_matured.sum())
            entry["uncovered_session_instances"] += int(group.uncovered_holding_sessions.sum())
            entry["first_decision"] = min(entry["first_decision"], str(group.session_date_et.min()))
            entry["last_decision"] = max(entry["last_decision"], str(group.session_date_et.max()))
        del frame, checked, failures
        release_process_memory()
        _guard("holding identity completed partition")
    if (parent_rows != manifest.get("rows")
            or observed != set(cohort.original_security_ids).difference(cohort.inherited_excluded_security_ids)):
        raise DataReadinessError("holding identity complete parent inventory differs")
    return {
        "scope": "membership_identity_only_not_bar_or_return_admission",
        "status": "uncovered_holding_identity" if totals["uncovered_matured"] else "membership_identity_covered",
        "cohort_sha256": cohort.sha256(), "all_history": dict(totals), "initial_fit": dict(development),
        "affected_securities": [affected[key] for key in sorted(affected)], "partitions": partitions,
        "numeric_features_read": False, "outcome_columns_read": False, "heldout_outcomes_read": False,
        "bar_coverage_verified": False, "benchmark_paths_verified": False,
        "accounting_eligible": False, "promotion_eligible": False,
    }


def run_swing_holding_identity_preflight(root: Path, config_path: Path, output_path: Path) -> dict[str, Any]:
    """Inspect identity coverage only; unresolved identity is a report, never a fill."""
    root = root.resolve()
    config_path = resolve_inside_authority(root, str(config_path))
    output_path = _inside(root, output_path)
    runtime = _inside(root, heavy_job_runtime_dir())
    with heavy_job_lease("audit-swing-holding-identity", runtime_dir=runtime):
        try:
            _guard("holding identity input loading")
            if config_path.stat().st_size > 1024**2:
                raise DataReadinessError("holding identity configuration exceeds bounded size")
            request = _Request.model_validate_json(json.dumps(tomllib.loads(config_path.read_text(encoding="utf-8"))))
            bound = {config_path.relative_to(root).as_posix(): file_sha256(config_path)}
            report = _collect(root, request, bound)
            report["request"] = request.model_dump(mode="json", by_alias=True)
            report["source_files"] = bound
            report["audit_sha256"] = json_sha256(report)
            if output_path.relative_to(root).as_posix() in bound:
                raise DataReadinessError("holding identity output cannot overwrite input")
            for relative, digest in bound.items():
                if file_sha256(resolve_inside_authority(root, relative)) != digest:
                    raise DataReadinessError("holding identity source changed during audit")
            output_path.parent.mkdir(parents=True, exist_ok=True)
            with file_lock(output_path, timeout=0.0):
                if output_path.exists():
                    if _object(output_path) != report:
                        raise DataReadinessError("immutable holding identity report conflicts")
                    return report
                staging = output_path.with_name(f".{output_path.name}.{uuid4().hex}.tmp")
                try:
                    write_json_object(staging, report)
                    os.rename(staging, output_path)
                finally:
                    staging.unlink(missing_ok=True)
            return report
        except (OSError, ValueError, TypeError, KeyError) as exc:
            raise DataReadinessError(f"invalid swing holding identity preflight: {exc}") from exc
