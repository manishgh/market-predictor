"""Verified, resumable combination of the two swing daily-history generations."""
from __future__ import annotations



import hashlib
import json
import uuid
from bisect import bisect_left
from collections import defaultdict
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport, audit_canonical_bars
from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.edge_rebuild.sp500_memberships import (
    MEMBERSHIP_REQUEST_SCHEMA,
    require_sp500_membership_authority,
)
from market_predictor.edge_rebuild.swing_history_collection import (
    load_complete_swing_history_collection,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

COMBINED_REQUEST_SCHEMA: Final = "edge_rebuild.swing_combined_daily_request.v5"
COMBINED_MANIFEST_SCHEMA: Final = "edge_rebuild.swing_combined_daily_manifest.v5"
COMBINED_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_combined_daily_authority.v5"
COMBINED_TICKER_SCHEMA: Final = "edge_rebuild.swing_combined_daily_ticker.v5"
COVERAGE_AUDIT_SCHEMA: Final = "edge_rebuild.swing_combined_daily_coverage.v4"
SESSION_GAP_AUDIT_SCHEMA: Final = "edge_rebuild.swing_session_gap_audit.v1"
POST_REQUEST_SCHEMA: Final = "swing.daily_history_collection.v1"
POST_MANIFEST_SCHEMA: Final = "swing.daily_history_manifest.v1"
START_DATE: Final = date(2018, 5, 29)
PRE_END_DATE: Final = date(2019, 7, 8)
POST_START_DATE: Final = date(2019, 7, 9)
CUTOFF_DATE: Final = date(2026, 7, 8)
MAXIMUM_EXCLUSION_FRACTION: Final = 0.05
# A sparse source defect must be both rare over the security's membership lifetime
# and short in exchange-session time. Larger gaps remove the whole security.
MAXIMUM_SPARSE_MISSING_FRACTION: Final = 0.005
MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS: Final = 5
FULL_COVERAGE_BENCHMARKS: Final = frozenset({"SPY", "QQQ"})
EASTERN: Final = ZoneInfo("America/New_York")


@dataclass(frozen=True, slots=True)
class VerifiedCombinedInputs:
    memberships: pd.DataFrame
    request_payload: dict[str, Any]
    pre_records: tuple[dict[str, Any], ...]
    post_records: dict[str, dict[str, Any]]
    excluded_security_ids: tuple[str, ...]
    benchmark_tickers: tuple[str, ...]
    coverage_audit: dict[str, Any]
    warmup_only_security_ids: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class _CoveragePreflight:
    audit: dict[str, Any]
    excluded_security_ids: tuple[str, ...]
    exclusion_records: tuple[dict[str, Any], ...]


@dataclass(frozen=True, slots=True)
class CombinedDailyStore:
    memberships: pd.DataFrame
    artifacts: dict[str, tuple[Path, str]]
    manifest: dict[str, Any]


def verify_combined_swing_inputs(
    *,
    pre_plan_directory: Path,
    pre_collection_directory: Path,
    post_collection_directory: Path,
    membership_directory: Path,
    raw_archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    security_exclusions_path: Path | None = None,
    model_decision_start: date | None = None,
    model_decision_cutoff: date | None = None,
) -> VerifiedCombinedInputs:
    """Verify both collection generations and the complete membership lineage."""

    membership_request = _load_json(membership_directory / "_request.json")
    if membership_request.get("schema") != MEMBERSHIP_REQUEST_SCHEMA:
        raise DataReadinessError("unsupported S&P membership request")
    try:
        membership_start = date.fromisoformat(str(membership_request["start_date"]))
        membership_cutoff = date.fromisoformat(str(membership_request["cutoff_date"]))
        maximum_exclusion = float(
            membership_request["maximum_security_exclusion_fraction"]
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise DataReadinessError("S&P membership request window is invalid") from exc
    if membership_start != START_DATE or membership_cutoff != CUTOFF_DATE:
        raise DataReadinessError(
            "S&P membership authority does not cover 2018-05-29 through 2026-07-08"
        )
    all_memberships = require_sp500_membership_authority(
        membership_directory,
        archive_directory=raw_archive_directory,
        event_directory=event_directory,
        transition_directory=transition_directory,
        reviewed_transitions_path=reviewed_transitions_path,
        anchor_path=anchor_path,
        start_date=membership_start,
        cutoff_date=membership_cutoff,
        security_exclusions_path=security_exclusions_path,
        maximum_security_exclusion_fraction=maximum_exclusion,
    )
    if (model_decision_start is None) != (model_decision_cutoff is None):
        raise ValueError(
            "model_decision_start and model_decision_cutoff must be provided together"
        )
    if model_decision_start is None or model_decision_cutoff is None:
        memberships = all_memberships
        modeled_security_ids = tuple(
            sorted(memberships["security_id"].astype(str).unique())
        )
        warmup_only_security_ids: tuple[str, ...] = ()
    else:
        memberships, modeled_security_ids, warmup_only_security_ids = (
            _modeled_security_population(
                all_memberships,
                decision_start=model_decision_start,
                decision_cutoff=model_decision_cutoff,
            )
        )
    membership_hashes = _membership_hashes(membership_directory)

    pre_manifest = load_complete_swing_history_collection(
        pre_collection_directory,
        plan_directory=pre_plan_directory,
    )
    pre_request = _load_json(pre_collection_directory / "_request.json")
    pre_authority = _load_json(pre_collection_directory / "_authority.json")
    plan_request = _load_json(pre_plan_directory / "_request.json")
    plan_membership = plan_request.get("membership_authority")
    if not isinstance(plan_membership, Mapping) or dict(plan_membership) != membership_hashes:
        raise DataReadinessError(
            "swing history plan and membership authority lineage differ"
        )
    pre_records = _verified_pre_records(pre_collection_directory, pre_manifest)
    post_records, post_unavailable, post_hashes = _verify_post_collection(
        post_collection_directory,
        expected_hashes=plan_request,
    )
    _validate_source_windows(pre_records, post_records)

    modeled_security_id_set = set(modeled_security_ids)
    pre_unavailable = _pre_unavailable_security_ids(pre_manifest).intersection(
        modeled_security_id_set
    )
    benchmark_tickers = tuple(
        sorted(
            {
                "SPY",
                "QQQ",
                *memberships["primary_benchmark"].astype(str).str.strip().str.upper(),
            }
        )
    )
    if set(post_unavailable).intersection(benchmark_tickers):
        raise DataReadinessError("a benchmark is unavailable in post-2019 history")
    post_unavailable_ids = _security_ids_for_unavailable_tickers(
        all_memberships,
        post_unavailable,
        start=POST_START_DATE,
        end=CUTOFF_DATE,
    ).intersection(modeled_security_id_set)
    all_security_ids = set(memberships["security_id"].astype(str))
    initially_excluded = pre_unavailable.union(post_unavailable_ids)
    unknown = sorted(initially_excluded.difference(all_security_ids))
    if unknown:
        raise DataReadinessError(f"unavailable security identities are absent: {unknown}")
    initial_reasons: dict[str, set[str]] = defaultdict(set)
    for security_id in pre_unavailable:
        initial_reasons[security_id].add("pre_collection_unavailable")
    for security_id in post_unavailable_ids:
        initial_reasons[security_id].add("post_collection_unavailable")
    coverage = _preflight_exact_coverage(
        memberships=memberships,
        pre_records=pre_records,
        post_records=post_records,
        benchmark_tickers=benchmark_tickers,
        initial_reasons=initial_reasons,
    )
    excluded = coverage.excluded_security_ids
    session_gap_audit = coverage.audit["session_gap_audit"]
    fraction = len(excluded) / len(all_security_ids)
    retained = memberships.loc[
        ~memberships["security_id"].astype(str).isin(excluded)
    ].copy()
    if retained.empty:
        raise DataReadinessError("combined daily history excludes every security")

    request_payload: dict[str, Any] = {
        "schema": COMBINED_REQUEST_SCHEMA,
        "start_date": START_DATE.isoformat(),
        "pre_end_date": PRE_END_DATE.isoformat(),
        "post_start_date": POST_START_DATE.isoformat(),
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "source": "alpaca",
        "timeframe": "1Day",
        "price_feed": "sip",
        "adjustment": "all",
        "pre_plan": {
            "directory": str(pre_plan_directory.resolve()),
            "request_sha256": file_sha256(pre_plan_directory / "_request.json"),
            "manifest_sha256": file_sha256(pre_plan_directory / "_manifest.json"),
            "authority_sha256": file_sha256(pre_plan_directory / "_authority.json"),
        },
        "pre_collection": {
            "directory": str(pre_collection_directory.resolve()),
            "request_sha256": file_sha256(pre_collection_directory / "_request.json"),
            "manifest_sha256": file_sha256(pre_collection_directory / "_manifest.json"),
            "authority_sha256": file_sha256(pre_collection_directory / "_authority.json"),
            "unit_set_sha256": str(pre_manifest["unit_set_sha256"]),
            "universe_sha256": str(pre_manifest["universe_sha256"]),
        },
        "post_collection": {
            "directory": str(post_collection_directory.resolve()),
            **post_hashes,
        },
        "membership_authority": membership_hashes,
        "pre_unavailable_security_count": len(pre_unavailable),
        "post_unavailable_security_count": len(post_unavailable_ids),
        "excluded_security_count": len(excluded),
        "excluded_security_fraction": fraction,
        "excluded_security_ids": list(excluded),
        "excluded_security_ids_sha256": _json_sha256(list(excluded)),
        "security_exclusions": list(coverage.exclusion_records),
        "coverage_audit_schema": COVERAGE_AUDIT_SCHEMA,
        "coverage_audit_sha256": _json_sha256(coverage.audit),
        "session_gap_audit_schema": SESSION_GAP_AUDIT_SCHEMA,
        "session_gap_audit_sha256": _json_sha256(session_gap_audit),
        "session_gap_abstention_count": int(
            session_gap_audit["missing_session_count"]
        ),
        "benchmark_coverage": coverage.audit["benchmark_audit"],
        "retained_security_count": int(retained["security_id"].nunique()),
        "modeled_security_count": len(modeled_security_ids),
        "modeled_security_ids_sha256": _json_sha256(
            list(modeled_security_ids)
        ),
        "warmup_only_security_count": len(warmup_only_security_ids),
        "warmup_only_security_ids": list(warmup_only_security_ids),
        "warmup_only_security_ids_sha256": _json_sha256(
            list(warmup_only_security_ids)
        ),
        "model_decision_start": (
            model_decision_start.isoformat()
            if model_decision_start is not None
            else None
        ),
        "model_decision_cutoff": (
            model_decision_cutoff.isoformat()
            if model_decision_cutoff is not None
            else None
        ),
        "benchmark_tickers": list(benchmark_tickers),
        "pre_request_identity_sha256": str(pre_request["request_sha256"]),
        "pre_authority_unit_set_sha256": str(pre_authority["unit_set_sha256"]),
    }
    return VerifiedCombinedInputs(
        memberships=retained,
        request_payload=request_payload,
        pre_records=pre_records,
        post_records=post_records,
        excluded_security_ids=excluded,
        benchmark_tickers=benchmark_tickers,
        coverage_audit=coverage.audit,
        warmup_only_security_ids=warmup_only_security_ids,
    )


def _modeled_security_population(
    memberships: pd.DataFrame,
    *,
    decision_start: date,
    decision_cutoff: date,
) -> tuple[pd.DataFrame, tuple[str, ...], tuple[str, ...]]:
    """Return modeled identities while preserving their complete warm-up history."""

    if decision_start > decision_cutoff:
        raise ValueError("modeled decision start is after its cutoff")
    required = {
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(
            f"swing memberships lack modeled-population fields: {missing}"
        )
    frame = memberships.copy()
    security_ids = frame["security_id"].astype("string").str.strip()
    effective_from = _strict_utc_membership_series(
        frame["effective_from_utc"], field="effective_from_utc"
    )
    effective_to = _strict_utc_membership_series(
        frame["effective_to_utc"], field="effective_to_utc", optional=True
    )
    available_at = _strict_utc_membership_series(
        frame["available_at_utc"], field="available_at_utc"
    )
    if (
        security_ids.isna().any()
        or security_ids.eq("").any()
        or bool((effective_to.notna() & effective_to.le(effective_from)).any())
    ):
        raise DataReadinessError("swing memberships have invalid identity intervals")

    calendar = xcals.get_calendar("XNYS")
    sessions = pd.Series(
        [
            pd.Timestamp(value).date()
            for value in calendar.sessions_in_range(
                decision_start, decision_cutoff
            )
        ]
    )
    if sessions.empty:
        raise DataReadinessError("modeled decision window has no XNYS sessions")
    cutoffs = list(swing_prediction_cutoffs(sessions).sort_values())
    modeled: set[str] = set()
    for index, security_id in enumerate(security_ids.astype(str)):
        active_from = max(effective_from.iloc[index], available_at.iloc[index])
        first = bisect_left(cutoffs, active_from)
        if first < len(cutoffs) and (
            pd.isna(effective_to.iloc[index])
            or cutoffs[first] < effective_to.iloc[index]
        ):
            modeled.add(security_id)
    all_ids = set(security_ids.astype(str))
    modeled_ids = tuple(sorted(modeled))
    warmup_only_ids = tuple(sorted(all_ids.difference(modeled)))
    retained = frame.loc[security_ids.astype(str).isin(modeled_ids)].copy()
    if not modeled_ids or sorted(
        retained["security_id"].astype(str).unique()
    ) != list(modeled_ids):
        raise DataReadinessError("swing modeled-population filtering is inconsistent")
    return retained.reset_index(drop=True), modeled_ids, warmup_only_ids


def _strict_utc_membership_series(
    values: pd.Series,
    *,
    field: str,
    optional: bool = False,
) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        if optional and pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    parsed = pd.to_datetime(values.map(parse), utc=True)
    invalid = parsed.isna() & (~values.isna() if optional else pd.Series(True, index=values.index))
    if bool(invalid.any()) or (not optional and bool(parsed.isna().any())):
        raise DataReadinessError(f"swing memberships contain invalid {field}")
    return parsed


def prepare_combined_daily_store(
    *,
    verified: VerifiedCombinedInputs,
    output_directory: Path,
    parent_request_sha256: str,
    memory_budget_gib: float,
    memory_headroom_gib: float,
) -> CombinedDailyStore:
    """Publish or verify one canonical combined bar artifact per ticker."""

    coverage_audit_sha256 = _json_sha256(verified.coverage_audit)
    session_gap_audit = _require_session_gap_audit(verified.coverage_audit)
    session_gap_audit_sha256 = _json_sha256(session_gap_audit)
    if (
        verified.coverage_audit.get("schema") != COVERAGE_AUDIT_SCHEMA
        or verified.request_payload.get("coverage_audit_sha256")
        != coverage_audit_sha256
        or verified.request_payload.get("benchmark_coverage")
        != verified.coverage_audit.get("benchmark_audit")
        or verified.request_payload.get("session_gap_audit_sha256")
        != session_gap_audit_sha256
        or verified.request_payload.get("session_gap_audit_schema")
        != SESSION_GAP_AUDIT_SCHEMA
        or verified.request_payload.get("session_gap_abstention_count")
        != session_gap_audit.get("missing_session_count")
    ):
        raise DataReadinessError("combined daily coverage audit identity is invalid")
    request = {
        **verified.request_payload,
        "parent_materialization_request_sha256": parent_request_sha256,
    }
    request_sha256 = _json_sha256(request)
    bound_request = {**request, "request_sha256": request_sha256}
    request_path = output_directory / "_request.json"
    if request_path.exists():
        if _load_json(request_path) != bound_request:
            raise DataReadinessError("combined daily resume request differs")
    else:
        output_directory.mkdir(parents=True, exist_ok=True)
        _write_json_atomic(request_path, bound_request)
    coverage_audit_path = output_directory / "_coverage_audit.json"
    if coverage_audit_path.exists():
        if _load_json(coverage_audit_path) != verified.coverage_audit:
            raise DataReadinessError("combined daily coverage audit differs on resume")
    else:
        _write_json_atomic(coverage_audit_path, verified.coverage_audit)
    session_gap_audit_path = output_directory / "_session_gap_audit.json"
    if session_gap_audit_path.exists():
        if _load_json(session_gap_audit_path) != session_gap_audit:
            raise DataReadinessError(
                "combined daily session-gap audit differs on resume"
            )
    else:
        _write_json_atomic(session_gap_audit_path, session_gap_audit)

    if (output_directory / "_authority.json").exists():
        return _load_complete_combined_store(
            output_directory,
            memberships=verified.memberships,
            request_sha256=request_sha256,
        )

    calendar = xcals.get_calendar("XNYS")
    all_sessions = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range(START_DATE, CUTOFF_DATE)
    )
    pre_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in verified.pre_records:
        pre_by_ticker[str(record["ticker"])].append(record)
    tickers = sorted(
        {
            *verified.memberships["ticker"].astype(str).str.upper(),
            *verified.benchmark_tickers,
        }
    )
    benchmark_starts = _benchmark_start_sessions(
        verified.coverage_audit,
        expected_tickers=verified.benchmark_tickers,
    )
    session_abstentions = _session_abstentions_by_ticker(session_gap_audit)
    unknown_gap_tickers = sorted(set(session_abstentions).difference(tickers))
    if unknown_gap_tickers:
        raise DataReadinessError(
            "session-gap audit contains tickers outside retained membership: "
            f"{unknown_gap_tickers}"
        )
    records: list[dict[str, Any]] = []
    for ticker in tickers:
        abstained_sessions = session_abstentions.get(ticker, set())
        expected_sessions = _expected_ticker_sessions(
            ticker,
            memberships=verified.memberships,
            benchmark_tickers=verified.benchmark_tickers,
            benchmark_start_sessions=benchmark_starts,
            all_sessions=all_sessions,
            session_abstentions=abstained_sessions,
        )
        if not expected_sessions:
            raise DataReadinessError(f"combined daily ticker has no expected sessions: {ticker}")
        token = hashlib.sha256(ticker.encode()).hexdigest()[:16]
        path = output_directory / "bars" / f"{token}.parquet"
        record_path = output_directory / "bars" / f"{token}.json"
        existing = _load_ticker_record(
            path,
            record_path,
            request_sha256=request_sha256,
            ticker=ticker,
            expected_sessions=expected_sessions,
            abstained_sessions=abstained_sessions,
            session_gap_audit_sha256=session_gap_audit_sha256,
        )
        if existing is None:
            bars = _combine_ticker(
                ticker,
                pre_records=pre_by_ticker.get(ticker, []),
                post_record=verified.post_records.get(ticker),
                expected_sessions=expected_sessions,
                abstained_sessions=abstained_sessions,
                calendar=calendar,
            )
            inputs = {
                "combined_request_sha256": request_sha256,
                "pre_collection_manifest_sha256": str(
                    verified.request_payload["pre_collection"]["manifest_sha256"]
                ),
                "post_collection_manifest_sha256": str(
                    verified.request_payload["post_collection"]["manifest_sha256"]
                ),
            }
            audit = CanonicalAuditReport(
                checks=audit_canonical_bars(bars, require_sip=True)
            )
            manifest = write_canonical_artifact(
                bars,
                path,
                artifact_type="bars",
                audit=audit,
                inputs=inputs,
            )
            existing = {
                "schema": COMBINED_TICKER_SCHEMA,
                "request_sha256": request_sha256,
                "ticker": ticker,
                "path": str(path.relative_to(output_directory)).replace("\\", "/"),
                "sha256": str(manifest["artifact_sha256"]),
                "canonical_manifest_sha256": file_sha256(manifest_path_for(path)),
                "rows": len(bars),
                "first_session": min(expected_sessions).isoformat(),
                "last_session": max(expected_sessions).isoformat(),
                "coverage_policy": "exact_observed_membership_sessions",
                "expected_sessions_sha256": _session_set_sha256(expected_sessions),
                "session_abstention_count": len(abstained_sessions),
                "session_abstentions_sha256": _session_set_sha256(
                    abstained_sessions
                ),
                "session_gap_audit_sha256": session_gap_audit_sha256,
            }
            _write_json_atomic(record_path, existing)
            del bars
            release_process_memory()
        records.append(existing)
        _guard(memory_budget_gib, memory_headroom_gib, f"combined daily {ticker}")

    manifest = {
        "schema": COMBINED_MANIFEST_SCHEMA,
        "status": "complete",
        "request_sha256": request_sha256,
        "start_date": START_DATE.isoformat(),
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "ticker_count": len(records),
        "rows": sum(int(record["rows"]) for record in records),
        "excluded_security_ids": list(verified.excluded_security_ids),
        "excluded_security_count": len(verified.excluded_security_ids),
        "security_exclusions": verified.request_payload["security_exclusions"],
        "benchmark_coverage": verified.request_payload["benchmark_coverage"],
        "coverage_audit": {
            "path": "_coverage_audit.json",
            "sha256": file_sha256(coverage_audit_path),
            "semantic_sha256": coverage_audit_sha256,
        },
        "session_gap_audit": {
            "path": "_session_gap_audit.json",
            "sha256": file_sha256(session_gap_audit_path),
            "semantic_sha256": session_gap_audit_sha256,
            "gap_count": int(session_gap_audit["gap_count"]),
            "missing_session_count": int(
                session_gap_audit["missing_session_count"]
            ),
        },
        "source_lineage": {
            "pre_collection": verified.request_payload["pre_collection"],
            "post_collection": verified.request_payload["post_collection"],
            "membership_authority": verified.request_payload["membership_authority"],
        },
        "memory": memory_audit(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
        "artifacts": records,
    }
    _write_json_atomic(output_directory / "_manifest.json", manifest)
    _write_json_atomic(
        output_directory / "_authority.json",
        {
            "schema": COMBINED_AUTHORITY_SCHEMA,
            "state": "complete",
            "request_sha256": request_sha256,
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(output_directory / "_manifest.json"),
            "pre_collection_authority_sha256": verified.request_payload[
                "pre_collection"
            ]["authority_sha256"],
            "post_collection_manifest_sha256": verified.request_payload[
                "post_collection"
            ]["manifest_sha256"],
            "coverage_audit_sha256": file_sha256(coverage_audit_path),
            "session_gap_audit_sha256": file_sha256(session_gap_audit_path),
        },
    )
    return _load_complete_combined_store(
        output_directory,
        memberships=verified.memberships,
        request_sha256=request_sha256,
    )


def _membership_hashes(directory: Path) -> dict[str, Any]:
    manifest = _load_json(directory / "_manifest.json")
    membership = manifest.get("membership_artifact")
    parent = manifest.get("parent_lineage")
    if not isinstance(membership, Mapping) or not isinstance(parent, Mapping):
        raise DataReadinessError("membership authority lineage inventory is invalid")
    return {
        "request_sha256": file_sha256(directory / "_request.json"),
        "manifest_sha256": file_sha256(directory / "_manifest.json"),
        "authority_sha256": file_sha256(directory / "_authority.json"),
        "membership_artifact_sha256": str(membership.get("sha256", "")),
        "universe_sha256": str(manifest.get("universe_sha256", "")),
        "parent_lineage": dict(parent),
    }


def _verified_pre_records(
    directory: Path,
    manifest: Mapping[str, Any],
) -> tuple[dict[str, Any], ...]:
    raw = manifest.get("unit_artifacts")
    if not isinstance(raw, list):
        raise DataReadinessError("pre-2019 collection has no unit artifacts")
    records: list[dict[str, Any]] = []
    for item in raw:
        if not isinstance(item, Mapping) or item.get("status") != "observed":
            continue
        record = {str(key): value for key, value in item.items()}
        path = _resolve_inside(directory, str(record.get("bars_path", "")))
        if file_sha256(path) != record.get("bars_sha256"):
            raise DataReadinessError(f"pre-2019 partition hash mismatch: {path}")
        record["resolved_path"] = str(path)
        records.append(record)
    if not records:
        raise DataReadinessError("pre-2019 collection has no observed units")
    return tuple(records)


def _verify_post_collection(
    directory: Path,
    *,
    expected_hashes: Mapping[str, Any],
) -> tuple[dict[str, dict[str, Any]], tuple[str, ...], dict[str, Any]]:
    request_path = directory / "_request.json"
    status_path = directory / "_status.json"
    manifest_path = directory / "_manifest.json"
    ledger_path = directory / "_source_collections.parquet"
    request = _load_json(request_path)
    status = _load_json(status_path)
    manifest = _load_json(manifest_path)
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    identity = hashlib.sha256(
        json.dumps(payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    expected = {
        "daily_request_file_sha256": file_sha256(request_path),
        "daily_status_sha256": file_sha256(status_path),
        "daily_manifest_sha256": file_sha256(manifest_path),
        "daily_request_identity_sha256": identity,
    }
    for key, value in expected.items():
        if expected_hashes.get(key) != value:
            raise DataReadinessError(f"post-2019 collection lineage mismatch: {key}")
    if (
        request.get("schema") != POST_REQUEST_SCHEMA
        or request.get("request_sha256") != identity
        or request.get("source") != "alpaca"
        or request.get("timeframe") != "1d"
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
        or request.get("start_date") != POST_START_DATE.isoformat()
        or request.get("end_date") != CUTOFF_DATE.isoformat()
    ):
        raise DataReadinessError("post-2019 collection request is unsupported")
    for terminal in (status, manifest):
        if (
            terminal.get("schema") != POST_MANIFEST_SCHEMA
            or terminal.get("status") not in {"complete", "complete_with_gaps"}
            or terminal.get("request_sha256") != identity
            or terminal.get("source_collections_sha256") != file_sha256(ledger_path)
        ):
            raise DataReadinessError("post-2019 terminal manifest is invalid")
    shared_terminal = (
        "status",
        "request_sha256",
        "requested_symbols",
        "observed_symbols",
        "unavailable_symbols",
        "failed_symbols",
        "skipped_symbols",
        "source_collections_sha256",
    )
    if (
        any(status.get(key) != manifest.get(key) for key in shared_terminal)
        or manifest.get("failed_symbols") != {}
    ):
        raise DataReadinessError("post-2019 status and manifest disagree")
    if file_sha256(ledger_path) != manifest.get("source_collections_sha256"):
        raise DataReadinessError("post-2019 source ledger hash mismatch")
    try:
        ledger = pd.read_parquet(ledger_path)
    except (OSError, ValueError) as exc:
        raise DataReadinessError("post-2019 source ledger is unreadable") from exc
    required_ledger = {
        "collection_id",
        "ticker",
        "source_family",
        "requested_start_utc",
        "requested_end_utc",
        "status",
        "row_count",
    }
    if (
        not required_ledger.issubset(ledger.columns)
        or ledger.empty
        or bool(ledger["ticker"].astype(str).str.upper().duplicated().any())
        or set(ledger["source_family"].astype(str)) != {"alpaca_daily_bars"}
    ):
        raise DataReadinessError("post-2019 source ledger identity is invalid")
    requested_starts = pd.to_datetime(
        ledger["requested_start_utc"], utc=True, errors="coerce"
    )
    requested_ends = pd.to_datetime(
        ledger["requested_end_utc"], utc=True, errors="coerce"
    )
    collection_ids = ledger["collection_id"].astype(str)
    fresh = (
        requested_starts.dt.date.eq(POST_START_DATE)
        & requested_ends.dt.date.eq(CUTOFF_DATE)
        & collection_ids.str.startswith("alpaca-")
    )
    resumed = (
        requested_starts.eq(requested_ends)
        & requested_starts.dt.date.gt(CUTOFF_DATE)
        & collection_ids.str.startswith("resumed-")
    )
    if (
        requested_starts.isna().any()
        or requested_ends.isna().any()
        or not bool((fresh | resumed).all())
    ):
        raise DataReadinessError("post-2019 source ledger window is invalid")
    raw_artifacts = manifest.get("artifacts")
    if (
        not isinstance(raw_artifacts, list)
        or len(raw_artifacts) != int(manifest.get("artifact_count", -1))
        or len(raw_artifacts) != int(manifest.get("observed_symbols", -1))
    ):
        raise DataReadinessError("post-2019 partition inventory is invalid")
    records: dict[str, dict[str, Any]] = {}
    row_total = 0
    for raw in raw_artifacts:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("post-2019 partition record is malformed")
        record = {str(key): value for key, value in raw.items()}
        ticker = str(record.get("ticker", "")).strip().upper()
        if not ticker or ticker in records:
            raise DataReadinessError("post-2019 partition ticker identity is invalid")
        path = _resolve_collection_artifact(directory, str(record.get("path", "")))
        sidecar = _resolve_collection_artifact(
            directory,
            str(record.get("manifest_path", "")),
        )
        if (
            file_sha256(path) != record.get("sha256")
            or sidecar != manifest_path_for(path)
            or not sidecar.is_file()
        ):
            raise DataReadinessError(f"post-2019 partition hash mismatch: {ticker}")
        bars, canonical = load_canonical_artifact(path, expected_type="bars")
        if (
            canonical.get("artifact_sha256") != record.get("sha256")
            or set(bars["ticker"].astype(str).str.upper()) != {ticker}
            or set(bars["timeframe"].astype(str)) != {"1d"}
            or set(bars["source"].astype(str)) != {"alpaca"}
            or set(bars["price_feed"].astype(str)) != {"sip"}
            or set(bars["adjustment"].astype(str)) != {"all"}
            or len(bars) != int(record.get("rows", -1))
        ):
            raise DataReadinessError(f"post-2019 partition identity mismatch: {ticker}")
        _validate_ohlcv(bars, ticker=ticker)
        _validate_post_timestamps(bars, ticker=ticker)
        starts = pd.to_datetime(bars["bar_start_utc"], utc=True)
        if (
            record.get("first_bar_start_utc") != starts.min().isoformat()
            or record.get("last_bar_start_utc") != starts.max().isoformat()
            or record.get("price_feed") != "sip"
            or record.get("adjustment") != "all"
        ):
            raise DataReadinessError(
                f"post-2019 partition metadata mismatch: {ticker}"
            )
        record["resolved_path"] = str(path)
        records[ticker] = record
        row_total += len(bars)
        del bars
        release_process_memory()
    if row_total != int(manifest.get("total_rows", -1)):
        raise DataReadinessError("post-2019 partition row total is invalid")
    unavailable_raw = manifest.get("unavailable_symbols", [])
    if not isinstance(unavailable_raw, list):
        raise DataReadinessError("post-2019 unavailable-symbol inventory is invalid")
    unavailable = tuple(sorted(str(value).strip().upper() for value in unavailable_raw))
    if len(records) + len(unavailable) != int(manifest.get("requested_symbols", -1)):
        raise DataReadinessError("post-2019 terminal symbol counts are inconsistent")
    ledger_rows = {
        str(row.ticker).strip().upper(): row
        for row in ledger.itertuples(index=False)
    }
    if set(ledger_rows) != set(records).union(unavailable):
        raise DataReadinessError("post-2019 source ledger symbol set is inconsistent")
    for ticker, record in records.items():
        row = ledger_rows[ticker]
        if str(row.status) != "observed" or int(row.row_count) != int(record["rows"]):
            raise DataReadinessError(
                f"post-2019 source ledger observed row is invalid: {ticker}"
            )
    for ticker in unavailable:
        row = ledger_rows[ticker]
        if str(row.status) != "observed_empty" or int(row.row_count) != 0:
            raise DataReadinessError(
                f"post-2019 source ledger unavailable row is invalid: {ticker}"
            )
    return records, unavailable, {
        "request_file_sha256": expected["daily_request_file_sha256"],
        "request_identity_sha256": identity,
        "status_sha256": expected["daily_status_sha256"],
        "manifest_sha256": expected["daily_manifest_sha256"],
        "source_collections_sha256": file_sha256(ledger_path),
    }


def _validate_source_windows(
    pre_records: tuple[dict[str, Any], ...],
    post_records: Mapping[str, dict[str, Any]],
) -> None:
    pre_starts = [date.fromisoformat(str(record["start_date"])) for record in pre_records]
    pre_ends = [date.fromisoformat(str(record["end_date"])) for record in pre_records]
    if min(pre_starts) != START_DATE or max(pre_ends) != PRE_END_DATE:
        raise DataReadinessError("pre-2019 collection does not span its exact source window")
    calendar = xcals.get_calendar("XNYS")
    next_session = pd.Timestamp(calendar.next_session(PRE_END_DATE)).date()
    if next_session != POST_START_DATE:
        raise DataReadinessError("daily source windows are not adjacent XNYS sessions")
    if not post_records:
        raise DataReadinessError("post-2019 collection has no observed partitions")


def _pre_unavailable_security_ids(manifest: Mapping[str, Any]) -> set[str]:
    raw = manifest.get("unavailable_units")
    if not isinstance(raw, list):
        raise DataReadinessError("pre-2019 unavailable-unit inventory is invalid")
    security_ids: set[str] = set()
    for item in raw:
        if not isinstance(item, Mapping) or item.get("role") != "stock":
            raise DataReadinessError("a benchmark cannot be an unavailable pre-2019 unit")
        security_id = str(item.get("security_id", ""))
        if not security_id or not bool(item.get("allowed")):
            raise DataReadinessError("pre-2019 unavailable security is not allowed")
        security_ids.add(security_id)
    if len(security_ids) != int(manifest.get("unavailable_security_count", -1)):
        raise DataReadinessError("pre-2019 unavailable-security count is inconsistent")
    return security_ids


def _security_ids_for_unavailable_tickers(
    memberships: pd.DataFrame,
    tickers: tuple[str, ...],
    *,
    start: date,
    end: date,
) -> set[str]:
    result: set[str] = set()
    for ticker in tickers:
        rows = memberships.loc[memberships["ticker"].astype(str).str.upper().eq(ticker)]
        matched = False
        for row in rows.itertuples(index=False):
            interval_start = _eastern_date(row.effective_from_utc)
            interval_end = (
                end
                if pd.isna(row.effective_to_utc)
                else _eastern_date(row.effective_to_utc) - timedelta(days=1)
            )
            if max(start, interval_start) <= min(end, interval_end):
                result.add(str(row.security_id))
                matched = True
        if not matched:
            raise DataReadinessError(
                f"post-2019 unavailable ticker has no membership identity: {ticker}"
            )
    return result


def _preflight_exact_coverage(
    *,
    memberships: pd.DataFrame,
    pre_records: tuple[dict[str, Any], ...],
    post_records: Mapping[str, dict[str, Any]],
    benchmark_tickers: tuple[str, ...],
    initial_reasons: Mapping[str, set[str]],
) -> _CoveragePreflight:
    """Classify stock gaps without imputing or weakening whole-security limits."""

    calendar = xcals.get_calendar("XNYS")
    all_sessions = tuple(
        pd.Timestamp(value).date()
        for value in calendar.sessions_in_range(START_DATE, CUTOFF_DATE)
    )
    pre_by_ticker: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in pre_records:
        pre_by_ticker[str(record["ticker"]).strip().upper()].append(record)

    security_ids = sorted(memberships["security_id"].astype(str).unique())
    if not security_ids:
        raise DataReadinessError("coverage preflight has no stock securities")
    expected_by_security: dict[str, set[date]] = {
        security_id: set() for security_id in security_ids
    }
    observed_by_security: dict[str, set[date]] = {
        security_id: set() for security_id in security_ids
    }
    tickers_by_security: dict[str, set[str]] = {
        security_id: set() for security_id in security_ids
    }
    ticker_by_security_session: dict[str, dict[date, str]] = {
        security_id: {} for security_id in security_ids
    }
    benchmark_audit: list[dict[str, Any]] = []
    stock_tickers = set(memberships["ticker"].astype(str).str.strip().str.upper())
    for ticker in sorted(stock_tickers.union(benchmark_tickers)):
        observed = _load_observed_ticker_sessions(
            ticker,
            pre_records=pre_by_ticker.get(ticker, []),
            post_record=post_records.get(ticker),
        )
        if ticker in benchmark_tickers:
            benchmark_audit.append(
                _benchmark_coverage_record(
                    ticker,
                    observed_sessions=observed,
                    market_sessions=all_sessions,
                )
            )
        rows = memberships.loc[
            memberships["ticker"].astype(str).str.strip().str.upper().eq(ticker)
        ]
        session_owner: dict[date, str] = {}
        for row in rows.itertuples(index=False):
            security_id = str(row.security_id)
            start = max(START_DATE, _eastern_date(row.effective_from_utc))
            end = (
                CUTOFF_DATE
                if pd.isna(row.effective_to_utc)
                else min(
                    CUTOFF_DATE,
                    _eastern_date(row.effective_to_utc) - timedelta(days=1),
                )
            )
            expected = {value for value in all_sessions if start <= value <= end}
            for session in expected:
                prior = session_owner.setdefault(session, security_id)
                if prior != security_id:
                    raise DataReadinessError(
                        f"ticker {ticker} maps to multiple securities on {session}"
                    )
            overlap = expected_by_security[security_id].intersection(expected)
            if overlap:
                first = min(overlap)
                raise DataReadinessError(
                    f"security {security_id} has overlapping ticker identities on {first}"
                )
            expected_by_security[security_id].update(expected)
            observed_by_security[security_id].update(expected.intersection(observed))
            tickers_by_security[security_id].add(ticker)
            for session in expected:
                ticker_by_security_session[security_id][session] = ticker
        release_process_memory()
        _guard(4.0, 0.75, f"combined daily coverage preflight {ticker}")

    security_audit: list[dict[str, Any]] = []
    exclusion_records: list[dict[str, Any]] = []
    session_gap_records: list[dict[str, Any]] = []
    for security_id in security_ids:
        expected = expected_by_security[security_id]
        if not expected:
            raise DataReadinessError(
                f"coverage preflight has no membership sessions for {security_id}"
            )
        observed = observed_by_security[security_id]
        missing = sorted(expected.difference(observed))
        reasons = set(initial_reasons.get(security_id, set()))
        missing_fraction = len(missing) / len(expected)
        missing_blocks = _contiguous_session_blocks(missing, expected)
        maximum_contiguous_gap = max(
            (len(block) for block in missing_blocks),
            default=0,
        )
        sparse = bool(missing) and not reasons and (
            missing_fraction <= MAXIMUM_SPARSE_MISSING_FRACTION
            and maximum_contiguous_gap <= MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
        )
        if missing and not sparse:
            reasons.add("membership_session_gap")
        if sparse:
            missing_by_ticker: dict[str, list[date]] = defaultdict(list)
            for session in missing:
                missing_by_ticker[
                    ticker_by_security_session[security_id][session]
                ].append(session)
            ordered_expected = sorted(expected)
            expected_rank = {
                session: index for index, session in enumerate(ordered_expected)
            }
            for ticker, ticker_missing in sorted(missing_by_ticker.items()):
                for block in _contiguous_session_blocks(ticker_missing, expected):
                    first_rank = expected_rank[block[0]]
                    last_rank = expected_rank[block[-1]]
                    session_gap_records.append(
                        {
                            "security_id": security_id,
                            "ticker": ticker,
                            "missing_sessions": [
                                session.isoformat() for session in block
                            ],
                            "missing_session_count": len(block),
                            "first_missing_session": block[0].isoformat(),
                            "last_missing_session": block[-1].isoformat(),
                            "previous_membership_session": (
                                ordered_expected[first_rank - 1].isoformat()
                                if first_rank > 0
                                else None
                            ),
                            "next_membership_session": (
                                ordered_expected[last_rank + 1].isoformat()
                                if last_rank + 1 < len(ordered_expected)
                                else None
                            ),
                            "required_downstream_action": (
                                "abstain_if_feature_or_label_window_crosses_gap"
                            ),
                        }
                    )
        action = (
            "exclude_security"
            if reasons
            else "retain_with_session_abstentions"
            if sparse
            else "retain"
        )
        record = {
            "security_id": security_id,
            "tickers": sorted(tickers_by_security[security_id]),
            "expected_session_count": len(expected),
            "observed_session_count": len(observed),
            "missing_session_count": len(missing),
            "missing_fraction": missing_fraction,
            "maximum_contiguous_missing_sessions": maximum_contiguous_gap,
            "first_missing_session": missing[0].isoformat() if missing else None,
            "reasons": (
                ["verified_sparse_membership_session_gap"]
                if sparse
                else sorted(reasons)
            ),
            "action": action,
        }
        security_audit.append(record)
        if action == "exclude_security":
            exclusion_records.append(record)
    excluded = tuple(
        sorted(str(record["security_id"]) for record in exclusion_records)
    )
    fraction = len(excluded) / len(security_ids)
    if fraction > MAXIMUM_EXCLUSION_FRACTION:
        raise DataReadinessError(
            "combined daily whole-security exclusions exceed 5% after exact "
            f"coverage preflight: {len(excluded)}/{len(security_ids)}"
        )
    audit: dict[str, Any] = {
        "schema": COVERAGE_AUDIT_SCHEMA,
        "calendar": "XNYS",
        "start_date": START_DATE.isoformat(),
        "cutoff_date": CUTOFF_DATE.isoformat(),
        "source": "alpaca",
        "timeframe": "1Day",
        "price_feed": "sip",
        "adjustment": "all",
        "maximum_security_exclusion_fraction": MAXIMUM_EXCLUSION_FRACTION,
        "session_gap_audit": {
            "schema": SESSION_GAP_AUDIT_SCHEMA,
            "classification_policy": {
                "maximum_missing_fraction": MAXIMUM_SPARSE_MISSING_FRACTION,
                "maximum_contiguous_missing_sessions": (
                    MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
                ),
                "unavailable_source_action": "exclude_security",
                "substantial_gap_action": "exclude_security",
                "sparse_gap_action": "abstain",
                "imputation": "prohibited",
                "downstream_rule": (
                    "abstain from every feature or label row whose required "
                    "membership-session window intersects a missing session"
                ),
            },
            "gap_count": len(session_gap_records),
            "security_count": len(
                {record["security_id"] for record in session_gap_records}
            ),
            "missing_session_count": sum(
                int(record["missing_session_count"])
                for record in session_gap_records
            ),
            "gaps": session_gap_records,
        },
        "security_count": len(security_ids),
        "excluded_security_count": len(excluded),
        "excluded_security_fraction": fraction,
        "retained_security_count": len(security_ids) - len(excluded),
        "benchmark_audit": benchmark_audit,
        "security_audit": security_audit,
    }
    return _CoveragePreflight(
        audit=audit,
        excluded_security_ids=excluded,
        exclusion_records=tuple(exclusion_records),
    )


def _contiguous_session_blocks(
    missing_sessions: list[date],
    expected_sessions: set[date],
) -> list[list[date]]:
    """Group gaps by adjacent expected XNYS sessions, not calendar days."""

    if not missing_sessions:
        return []
    rank = {
        session: index for index, session in enumerate(sorted(expected_sessions))
    }
    blocks: list[list[date]] = []
    for session in sorted(missing_sessions):
        if session not in rank:
            raise DataReadinessError(
                f"missing session is outside membership coverage: {session}"
            )
        if not blocks or rank[session] != rank[blocks[-1][-1]] + 1:
            blocks.append([session])
        else:
            blocks[-1].append(session)
    return blocks


def _benchmark_coverage_record(
    ticker: str,
    *,
    observed_sessions: set[date],
    market_sessions: tuple[date, ...],
) -> dict[str, Any]:
    expected = set(market_sessions)
    observed = expected.intersection(observed_sessions)
    missing = sorted(expected.difference(observed))
    first_observed = min(observed) if observed else None
    full_coverage_required = ticker in FULL_COVERAGE_BENCHMARKS
    if first_observed is None:
        raise DataReadinessError(
            f"benchmark daily history has no observed XNYS sessions: {ticker}"
        )
    pre_inception = tuple(
        session for session in market_sessions if session < first_observed
    )
    internal_or_later = [
        session for session in missing if session >= first_observed
    ]
    if full_coverage_required and missing:
        raise DataReadinessError(
            "SPY/QQQ benchmark history requires exact full-window coverage: "
            f"{ticker}; missing={len(missing)}; first={missing[0]}"
        )
    if internal_or_later:
        raise DataReadinessError(
            "benchmark daily history has an internal or post-inception XNYS gap: "
            f"{ticker}; missing={len(internal_or_later)}; "
            f"first={internal_or_later[0]}"
        )
    if set(missing) != set(pre_inception):
        raise DataReadinessError(
            f"benchmark pre-inception gap is not one contiguous prefix: {ticker}"
        )
    return {
        "ticker": ticker,
        "coverage_policy": (
            "exact_full_window"
            if full_coverage_required
            else "contiguous_pre_inception_prefix_allowed"
        ),
        "requested_first_session": market_sessions[0].isoformat(),
        "requested_last_session": market_sessions[-1].isoformat(),
        "first_observed_session": first_observed.isoformat(),
        "expected_session_count": len(market_sessions),
        "observed_session_count": len(observed),
        "missing_session_count": len(missing),
        "pre_inception_missing_session_count": len(pre_inception),
        "first_missing_session": missing[0].isoformat() if missing else None,
        "action": "retain",
    }


def _load_observed_ticker_sessions(
    ticker: str,
    *,
    pre_records: list[dict[str, Any]],
    post_record: dict[str, Any] | None,
) -> set[date]:
    sessions: list[date] = []
    for record in pre_records:
        frame = pd.read_parquet(Path(str(record["resolved_path"])))
        _validate_pre_unit(frame, record=record)
        sessions.extend(_session_dates(frame["bar_start_utc"]))
        del frame
    if post_record is not None:
        frame, _ = load_canonical_artifact(
            Path(str(post_record["resolved_path"])),
            expected_type="bars",
        )
        _validate_ohlcv(frame, ticker=ticker)
        _validate_post_timestamps(frame, ticker=ticker)
        sessions.extend(_session_dates(frame["bar_start_utc"]))
        del frame
    if len(sessions) != len(set(sessions)):
        raise DataReadinessError(
            f"combined daily source sessions overlap for {ticker}"
        )
    return set(sessions)


def _expected_ticker_sessions(
    ticker: str,
    *,
    memberships: pd.DataFrame,
    benchmark_tickers: tuple[str, ...],
    benchmark_start_sessions: Mapping[str, date],
    all_sessions: tuple[date, ...],
    session_abstentions: set[date] | None = None,
) -> set[date]:
    abstentions = session_abstentions or set()
    if ticker in benchmark_tickers:
        if abstentions:
            raise DataReadinessError(
                f"benchmark cannot have stock session abstentions: {ticker}"
            )
        start = benchmark_start_sessions.get(ticker)
        if start is None:
            raise DataReadinessError(
                f"benchmark coverage audit is absent for {ticker}"
            )
        return {session for session in all_sessions if session >= start}
    rows = memberships.loc[memberships["ticker"].astype(str).str.upper().eq(ticker)]
    expected: dict[date, str] = {}
    for row in rows.itertuples(index=False):
        start = max(START_DATE, _eastern_date(row.effective_from_utc))
        end = (
            CUTOFF_DATE
            if pd.isna(row.effective_to_utc)
            else min(CUTOFF_DATE, _eastern_date(row.effective_to_utc) - timedelta(days=1))
        )
        for session in all_sessions:
            if start <= session <= end:
                prior = expected.setdefault(session, str(row.security_id))
                if prior != str(row.security_id):
                    raise DataReadinessError(
                        f"ticker {ticker} maps to multiple securities on {session}"
                    )
    unknown = abstentions.difference(expected)
    if unknown:
        raise DataReadinessError(
            f"session abstention is outside ticker membership: {ticker}; "
            f"first={min(unknown)}"
        )
    return set(expected).difference(abstentions)


def _benchmark_start_sessions(
    coverage_audit: Mapping[str, Any],
    *,
    expected_tickers: tuple[str, ...],
) -> dict[str, date]:
    raw = coverage_audit.get("benchmark_audit")
    if not isinstance(raw, list):
        raise DataReadinessError("combined daily benchmark coverage audit is invalid")
    starts: dict[str, date] = {}
    for item in raw:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily benchmark coverage record is invalid")
        ticker = str(item.get("ticker", "")).strip().upper()
        try:
            start = date.fromisoformat(str(item["first_observed_session"]))
            pre_inception = int(item["pre_inception_missing_session_count"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataReadinessError(
                f"combined daily benchmark coverage identity is invalid: {ticker}"
            ) from exc
        if (
            not ticker
            or ticker in starts
            or item.get("action") != "retain"
            or pre_inception < 0
        ):
            raise DataReadinessError(
                f"combined daily benchmark coverage record is invalid: {ticker}"
            )
        starts[ticker] = start
    if set(starts) != set(expected_tickers):
        raise DataReadinessError(
            "combined daily benchmark coverage population differs from its request"
        )
    return starts


def _require_session_gap_audit(
    coverage_audit: Mapping[str, Any],
) -> dict[str, Any]:
    raw = coverage_audit.get("session_gap_audit")
    if not isinstance(raw, Mapping):
        raise DataReadinessError("combined daily session-gap audit is absent")
    policy = raw.get("classification_policy")
    gaps = raw.get("gaps")
    security_audit = coverage_audit.get("security_audit")
    expected_policy = {
        "maximum_missing_fraction": MAXIMUM_SPARSE_MISSING_FRACTION,
        "maximum_contiguous_missing_sessions": (
            MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
        ),
        "unavailable_source_action": "exclude_security",
        "substantial_gap_action": "exclude_security",
        "sparse_gap_action": "abstain",
        "imputation": "prohibited",
        "downstream_rule": (
            "abstain from every feature or label row whose required "
            "membership-session window intersects a missing session"
        ),
    }
    if (
        raw.get("schema") != SESSION_GAP_AUDIT_SCHEMA
        or policy != expected_policy
        or not isinstance(gaps, list)
        or not isinstance(security_audit, list)
    ):
        raise DataReadinessError("combined daily session-gap audit is invalid")
    seen: set[tuple[str, date]] = set()
    security_ids: set[str] = set()
    missing_count = 0
    for item in gaps:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily session-gap record is invalid")
        security_id = str(item.get("security_id", "")).strip()
        ticker = str(item.get("ticker", "")).strip().upper()
        raw_sessions = item.get("missing_sessions")
        if not security_id or not ticker or not isinstance(raw_sessions, list):
            raise DataReadinessError("combined daily session-gap record is invalid")
        try:
            sessions = [date.fromisoformat(str(value)) for value in raw_sessions]
        except ValueError as exc:
            raise DataReadinessError(
                "combined daily session-gap record has an invalid session"
            ) from exc
        if (
            not sessions
            or sessions != sorted(set(sessions))
            or len(sessions) > MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
            or int(item.get("missing_session_count", -1)) != len(sessions)
            or item.get("first_missing_session") != sessions[0].isoformat()
            or item.get("last_missing_session") != sessions[-1].isoformat()
            or item.get("required_downstream_action")
            != "abstain_if_feature_or_label_window_crosses_gap"
            or any(session < START_DATE or session > CUTOFF_DATE for session in sessions)
        ):
            raise DataReadinessError("combined daily session-gap record is invalid")
        for session in sessions:
            identity = (ticker, session)
            if identity in seen:
                raise DataReadinessError(
                    "combined daily session-gap audit contains duplicate sessions"
                )
            seen.add(identity)
        security_ids.add(security_id)
        missing_count += len(sessions)
    if (
        int(raw.get("gap_count", -1)) != len(gaps)
        or int(raw.get("security_count", -1)) != len(security_ids)
        or int(raw.get("missing_session_count", -1)) != missing_count
    ):
        raise DataReadinessError("combined daily session-gap totals are invalid")
    sparse_security_counts: dict[str, int] = {}
    for item in security_audit:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily security coverage is invalid")
        if item.get("action") != "retain_with_session_abstentions":
            continue
        security_id = str(item.get("security_id", "")).strip()
        try:
            item_missing_count = int(item["missing_session_count"])
            item_missing_fraction = float(item["missing_fraction"])
            item_maximum_gap = int(item["maximum_contiguous_missing_sessions"])
        except (KeyError, TypeError, ValueError) as exc:
            raise DataReadinessError(
                "combined daily sparse security coverage is invalid"
            ) from exc
        if (
            not security_id
            or item_missing_count <= 0
            or item_missing_fraction > MAXIMUM_SPARSE_MISSING_FRACTION
            or item_maximum_gap > MAXIMUM_SPARSE_CONTIGUOUS_SESSIONS
            or item.get("reasons")
            != ["verified_sparse_membership_session_gap"]
        ):
            raise DataReadinessError(
                "combined daily sparse security coverage is invalid"
            )
        sparse_security_counts[security_id] = item_missing_count
    gap_security_counts: dict[str, int] = defaultdict(int)
    for item in gaps:
        gap_security_counts[str(item["security_id"])] += int(
            item["missing_session_count"]
        )
    if sparse_security_counts != dict(gap_security_counts):
        raise DataReadinessError(
            "combined daily sparse coverage and session-gap audit differ"
        )
    return dict(raw)


def _session_abstentions_by_ticker(
    session_gap_audit: Mapping[str, Any],
) -> dict[str, set[date]]:
    result: dict[str, set[date]] = defaultdict(set)
    gaps = session_gap_audit.get("gaps")
    if not isinstance(gaps, list):
        raise DataReadinessError("combined daily session-gap records are absent")
    for item in gaps:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily session-gap record is invalid")
        ticker = str(item["ticker"]).strip().upper()
        result[ticker].update(
            date.fromisoformat(str(value)) for value in item["missing_sessions"]
        )
    return dict(result)


def _combine_ticker(
    ticker: str,
    *,
    pre_records: list[dict[str, Any]],
    post_record: dict[str, Any] | None,
    expected_sessions: set[date],
    calendar: Any,
    abstained_sessions: set[date] | None = None,
) -> pd.DataFrame:
    abstentions = abstained_sessions or set()
    parts: list[pd.DataFrame] = []
    for record in pre_records:
        frame = pd.read_parquet(Path(str(record["resolved_path"])))
        _validate_pre_unit(frame, record=record)
        parts.append(_canonicalize(frame, ticker=ticker, calendar=calendar))
    if post_record is not None:
        frame, _ = load_canonical_artifact(
            Path(str(post_record["resolved_path"])),
            expected_type="bars",
        )
        parts.append(_canonicalize(frame, ticker=ticker, calendar=calendar))
    if not parts:
        raise DataReadinessError(f"daily history is absent for retained ticker {ticker}")
    combined = pd.concat(parts, ignore_index=True)
    sessions = _session_dates(combined["bar_start_utc"])
    combined = combined.assign(_session=sessions)
    if bool(combined.duplicated(["ticker", "_session"]).any()):
        raise DataReadinessError(f"combined daily history overlaps for {ticker}")
    observed = set(cast(pd.Series, combined["_session"]).tolist())
    stale_abstentions = sorted(observed.intersection(abstentions))
    if stale_abstentions:
        raise DataReadinessError(
            "combined daily session-gap audit is stale for "
            f"{ticker}; first={stale_abstentions[0]}"
        )
    missing = sorted(expected_sessions.difference(observed))
    if missing:
        raise DataReadinessError(
            f"combined daily history has {len(missing)} membership gaps for {ticker}; first={missing[0]}"
        )
    selected = combined.loc[combined["_session"].isin(expected_sessions)].drop(
        columns="_session"
    )
    if len(selected) != len(expected_sessions):
        raise DataReadinessError(f"combined daily identity coverage is invalid for {ticker}")
    return selected.sort_values("bar_start_utc", kind="stable").reset_index(drop=True)


def _validate_pre_unit(frame: pd.DataFrame, *, record: Mapping[str, Any]) -> None:
    ticker = str(record["ticker"])
    security_id = str(record["security_id"])
    if (
        frame.empty
        or set(frame["ticker"].astype(str).str.upper()) != {ticker}
        or set(frame["security_id"].astype(str)) != {security_id}
        or set(frame["role"].astype(str)) != {str(record["role"])}
        or set(frame["source"].astype(str)) != {"alpaca"}
        or set(frame["timeframe"].astype(str)) != {"1Day"}
        or set(frame["price_feed"].astype(str)) != {"sip"}
        or set(frame["adjustment"].astype(str)) != {"all"}
        or len(frame) != int(record["rows"])
    ):
        raise DataReadinessError(f"pre-2019 unit identity mismatch: {ticker}")
    _validate_ohlcv(frame, ticker=ticker)
    timestamps = pd.to_datetime(frame["bar_start_utc"], utc=True, errors="coerce")
    eastern = timestamps.dt.tz_convert(EASTERN)
    sessions = pd.to_datetime(frame["session_date"], errors="coerce").dt.date
    if (
        timestamps.isna().any()
        or not bool((eastern.dt.hour.eq(0) & eastern.dt.minute.eq(0)).all())
        or not bool(pd.Series(eastern.dt.date).reset_index(drop=True).eq(sessions.reset_index(drop=True)).all())
        or min(sessions) < date.fromisoformat(str(record["start_date"]))
        or max(sessions) > date.fromisoformat(str(record["end_date"]))
        or bool(pd.Series(sessions).duplicated().any())
    ):
        raise DataReadinessError(f"pre-2019 unit timestamps are invalid: {ticker}")


def _validate_post_timestamps(frame: pd.DataFrame, *, ticker: str) -> None:
    calendar = xcals.get_calendar("XNYS")
    starts = pd.to_datetime(frame["bar_start_utc"], utc=True, errors="coerce")
    ends = pd.to_datetime(frame["bar_end_utc"], utc=True, errors="coerce")
    if starts.isna().any() or ends.isna().any() or bool(starts.duplicated().any()):
        raise DataReadinessError(f"post-2019 timestamps are invalid: {ticker}")
    sessions = _session_dates(starts)
    if min(sessions) < POST_START_DATE or max(sessions) > CUTOFF_DATE:
        raise DataReadinessError(f"post-2019 timestamps leave the source window: {ticker}")
    for index, session in enumerate(sessions):
        if (
            starts.iloc[index] != pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
            or ends.iloc[index] != pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
        ):
            raise DataReadinessError(f"post-2019 bar is not aligned to XNYS: {ticker}")


def _canonicalize(frame: pd.DataFrame, *, ticker: str, calendar: Any) -> pd.DataFrame:
    sessions = _session_dates(pd.to_datetime(frame["bar_start_utc"], utc=True))
    opens = [pd.Timestamp(calendar.session_open(value)).tz_convert("UTC") for value in sessions]
    closes = [pd.Timestamp(calendar.session_close(value)).tz_convert("UTC") for value in sessions]
    ingested = pd.to_datetime(
        frame["ingested_at_utc"], utc=True, errors="coerce"
    ).reset_index(drop=True)
    if ingested.isna().any():
        raise DataReadinessError(f"daily history has invalid ingestion timestamps: {ticker}")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1d",
            "bar_start_utc": opens,
            "bar_end_utc": closes,
            "available_at_utc": [value + pd.Timedelta(minutes=15) for value in closes],
            "ingested_at_utc": ingested,
            "open": pd.to_numeric(frame["open"]),
            "high": pd.to_numeric(frame["high"]),
            "low": pd.to_numeric(frame["low"]),
            "close": pd.to_numeric(frame["close"]),
            "volume": pd.to_numeric(frame["volume"]),
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
            "availability_policy": "market_interval_close",
            "schema_version": "market_data.v1",
        }
    )


def _validate_ohlcv(frame: pd.DataFrame, *, ticker: str) -> None:
    numeric = frame[["open", "high", "low", "close", "volume"]].apply(
        pd.to_numeric,
        errors="coerce",
    )
    invalid = (
        numeric.isna().any(axis=1)
        | numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
        | numeric["volume"].lt(0)
        | numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    if bool(invalid.any()):
        raise DataReadinessError(f"daily history has invalid OHLCV: {ticker}")


def _load_ticker_record(
    path: Path,
    record_path: Path,
    *,
    request_sha256: str,
    ticker: str,
    expected_sessions: set[date],
    abstained_sessions: set[date] | None = None,
    session_gap_audit_sha256: str | None = None,
) -> dict[str, Any] | None:
    abstentions = abstained_sessions or set()
    if not path.exists() and not record_path.exists() and not manifest_path_for(path).exists():
        return None
    if not path.is_file() or not record_path.is_file() or not manifest_path_for(path).is_file():
        raise DataReadinessError(f"combined daily ticker publication is incomplete: {ticker}")
    record = _load_json(record_path)
    bars, manifest = load_canonical_artifact(path, expected_type="bars")
    sessions = set(_session_dates(bars["bar_start_utc"]))
    if (
        record.get("schema") != COMBINED_TICKER_SCHEMA
        or record.get("request_sha256") != request_sha256
        or record.get("ticker") != ticker
        or record.get("sha256") != file_sha256(path)
        or record.get("sha256") != manifest.get("artifact_sha256")
        or record.get("canonical_manifest_sha256") != file_sha256(manifest_path_for(path))
        or int(record.get("rows", -1)) != len(bars)
        or record.get("coverage_policy")
        != "exact_observed_membership_sessions"
        or record.get("expected_sessions_sha256") != _session_set_sha256(expected_sessions)
        or int(record.get("session_abstention_count", -1)) != len(abstentions)
        or record.get("session_abstentions_sha256")
        != _session_set_sha256(abstentions)
        or record.get("session_gap_audit_sha256")
        != session_gap_audit_sha256
        or sessions != expected_sessions
        or set(bars["ticker"].astype(str)) != {ticker}
    ):
        raise DataReadinessError(f"combined daily ticker does not verify: {ticker}")
    return record


def _load_complete_combined_store(
    directory: Path,
    *,
    memberships: pd.DataFrame,
    request_sha256: str,
) -> CombinedDailyStore:
    manifest_path = directory / "_manifest.json"
    request = _load_json(directory / "_request.json")
    manifest = _load_json(manifest_path)
    authority = _load_json(directory / "_authority.json")
    raw = manifest.get("artifacts")
    coverage_record = manifest.get("coverage_audit")
    session_gap_record = manifest.get("session_gap_audit")
    coverage_path = directory / "_coverage_audit.json"
    session_gap_path = directory / "_session_gap_audit.json"
    coverage_audit = _load_json(coverage_path) if coverage_path.is_file() else None
    session_gap_audit = (
        _load_json(session_gap_path) if session_gap_path.is_file() else None
    )
    if (
        manifest.get("schema") != COMBINED_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or not isinstance(raw, list)
        or len(raw) != int(manifest.get("ticker_count", -1))
        or authority.get("schema") != COMBINED_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or not isinstance(coverage_record, Mapping)
        or coverage_record.get("path") != "_coverage_audit.json"
        or not coverage_path.is_file()
        or coverage_record.get("sha256") != file_sha256(coverage_path)
        or authority.get("coverage_audit_sha256") != file_sha256(coverage_path)
        or coverage_record.get("semantic_sha256")
        != request.get("coverage_audit_sha256")
        or not isinstance(coverage_audit, Mapping)
        or _json_sha256(coverage_audit)
        != request.get("coverage_audit_sha256")
        or coverage_audit.get("benchmark_audit")
        != request.get("benchmark_coverage")
        or not isinstance(session_gap_record, Mapping)
        or session_gap_record.get("path") != "_session_gap_audit.json"
        or not session_gap_path.is_file()
        or session_gap_record.get("sha256") != file_sha256(session_gap_path)
        or authority.get("session_gap_audit_sha256")
        != file_sha256(session_gap_path)
        or not isinstance(session_gap_audit, Mapping)
        or _json_sha256(session_gap_audit)
        != request.get("session_gap_audit_sha256")
        or session_gap_record.get("semantic_sha256")
        != request.get("session_gap_audit_sha256")
        or coverage_audit.get("session_gap_audit") != session_gap_audit
        or manifest.get("security_exclusions")
        != request.get("security_exclusions")
        or manifest.get("benchmark_coverage")
        != request.get("benchmark_coverage")
    ):
        raise DataReadinessError("combined daily authority is invalid")
    if _require_session_gap_audit(coverage_audit) != session_gap_audit:
        raise DataReadinessError("combined daily session-gap authority is invalid")
    artifacts: dict[str, tuple[Path, str]] = {}
    rows = 0
    for item in raw:
        if not isinstance(item, Mapping):
            raise DataReadinessError("combined daily artifact record is malformed")
        ticker = str(item.get("ticker", ""))
        path = _resolve_inside(directory, str(item.get("path", "")))
        if (
            ticker in artifacts
            or file_sha256(path) != item.get("sha256")
            or item.get("session_gap_audit_sha256")
            != request.get("session_gap_audit_sha256")
        ):
            raise DataReadinessError(f"combined daily artifact is invalid: {ticker}")
        if file_sha256(manifest_path_for(path)) != item.get("canonical_manifest_sha256"):
            raise DataReadinessError(f"combined daily sidecar is invalid: {ticker}")
        artifacts[ticker] = (path, str(item["sha256"]))
        rows += int(item.get("rows", -1))
    if rows != int(manifest.get("rows", -1)):
        raise DataReadinessError("combined daily row count does not add up")
    return CombinedDailyStore(
        memberships=memberships,
        artifacts=artifacts,
        manifest=manifest,
    )


def _resolve_collection_artifact(directory: Path, raw: str) -> Path:
    normalized = Path(raw.replace("\\", "/"))
    candidates = [normalized] if normalized.is_absolute() else [directory / normalized]
    if not normalized.is_absolute() and len(directory.resolve().parents) >= 3:
        candidates.append(directory.resolve().parents[2] / normalized)
    for candidate in candidates:
        if candidate.is_file():
            resolved = candidate.resolve()
            try:
                resolved.relative_to(directory.resolve())
            except ValueError as exc:
                raise DataReadinessError(
                    f"collection artifact escapes its directory: {resolved}"
                ) from exc
            return resolved
    raise DataReadinessError(f"collection artifact is missing: {raw}")


def _resolve_inside(directory: Path, raw: str) -> Path:
    path = (directory / Path(raw.replace("\\", "/"))).resolve()
    try:
        path.relative_to(directory.resolve())
    except ValueError as exc:
        raise DataReadinessError(f"artifact escapes its authority directory: {raw}") from exc
    if not path.is_file():
        raise DataReadinessError(f"authority artifact is missing: {path}")
    return path


def _session_dates(values: pd.Series | pd.DatetimeIndex) -> list[date]:
    series = pd.Series(pd.to_datetime(values, utc=True, errors="coerce"))
    if series.isna().any():
        raise DataReadinessError("daily history contains invalid timestamps")
    return list(series.dt.tz_convert(EASTERN).dt.date)


def _eastern_date(value: object) -> date:
    timestamp = pd.Timestamp(value)
    if timestamp.tzinfo is None:
        raise DataReadinessError("membership timestamp is not timezone-aware")
    return cast(date, timestamp.tz_convert(EASTERN).date())


def _session_set_sha256(sessions: set[date]) -> str:
    return _json_sha256([value.isoformat() for value in sorted(sessions)])


def _guard(hard: float, headroom: float, stage: str) -> None:
    assert_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)
    assert_peak_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid or missing JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): item for key, item in value.items()}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
