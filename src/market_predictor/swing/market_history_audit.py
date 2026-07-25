from __future__ import annotations

import json
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.swing.market_history import DEFAULT_BENCHMARKS
from market_predictor.v3.errors import DataReadinessError

MARKET_HISTORY_AUDIT_SCHEMA = "swing.daily_history_coverage.v1"
_NEW_YORK = ZoneInfo("America/New_York")
_BLOCKING_GAPS = {"initial_nontrading_gap", "interior_gap", "no_member_session_overlap"}


def audit_swing_daily_history(
    *,
    memberships_path: Path,
    collection_dir: Path,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    manifest_path = collection_dir / "_manifest.json"
    request_path = collection_dir / "_request.json"
    ledger_path = collection_dir / "_source_collections.parquet"
    for path in (memberships_path, manifest_path, request_path, ledger_path):
        if not path.exists():
            raise FileNotFoundError(path)
    request = _load_json_object(request_path)
    manifest = _load_json_object(manifest_path)
    if manifest.get("status") not in {"complete", "complete_with_gaps"}:
        raise DataReadinessError("daily-history collection has not reached a terminal complete state")
    membership_sha = file_sha256(memberships_path)
    if request.get("memberships_sha256") != membership_sha:
        raise DataReadinessError("daily-history request does not match the supplied membership artifact")
    if manifest.get("request_sha256") != request.get("request_sha256"):
        raise DataReadinessError("daily-history manifest and request identities do not match")
    if manifest.get("source_collections_sha256") != file_sha256(ledger_path):
        raise DataReadinessError("daily-history source-collection ledger hash does not match")

    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, list) or len(artifacts) != manifest.get("artifact_count"):
        raise DataReadinessError("daily-history manifest artifact inventory is invalid")
    session_dates: dict[str, set[date]] = {}
    artifact_identity: dict[str, dict[str, Any]] = {}
    for raw_record in artifacts:
        if not isinstance(raw_record, dict):
            raise DataReadinessError("daily-history artifact record is invalid")
        ticker = str(raw_record.get("ticker") or "").strip().upper()
        path = Path(str(raw_record.get("path") or ""))
        if not path.exists() and ticker:
            path = collection_dir / "bars" / "1d" / f"{ticker}.parquet"
        if not ticker or not path.exists():
            raise DataReadinessError(f"daily-history artifact path is missing for {ticker or 'unknown'}")
        if file_sha256(path) != raw_record.get("sha256"):
            raise DataReadinessError(f"daily-history artifact hash does not match for {ticker}")
        bars, artifact_manifest = load_canonical_artifact(path, expected_type="bars")
        if str(artifact_manifest.get("artifact_sha256")) != raw_record.get("sha256"):
            raise DataReadinessError(f"daily-history canonical manifest does not match for {ticker}")
        observed_tickers = set(bars["ticker"].astype(str).str.upper())
        if observed_tickers != {ticker}:
            raise DataReadinessError(f"daily-history ticker identity does not match for {ticker}")
        session_dates[ticker] = set(pd.to_datetime(bars["bar_start_utc"], utc=True).dt.tz_convert(_NEW_YORK).dt.date)
        artifact_identity[ticker] = {
            "artifact_path": str(path),
            "artifact_sha256": str(raw_record["sha256"]),
            "first_bar_date": min(session_dates[ticker]).isoformat(),
            "last_bar_date": max(session_dates[ticker]).isoformat(),
            "bar_count": len(session_dates[ticker]),
        }
        assert_memory_budget(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage=f"daily history audit {ticker}",
        )
        del bars
        release_process_memory()

    market_sessions = session_dates.get("SPY")
    if not market_sessions:
        raise DataReadinessError("SPY is required as the observed SIP market-session calendar")
    start_date = date.fromisoformat(str(request["start_date"]))
    end_date = date.fromisoformat(str(request["end_date"]))
    expected_market_sessions = {value for value in market_sessions if start_date <= value <= end_date}
    unavailable = {str(value).strip().upper() for value in manifest.get("unavailable_symbols", [])}
    benchmark_audit = _audit_benchmarks(
        session_dates,
        expected_market_sessions,
        unavailable,
        benchmarks,
    )

    memberships = pd.read_parquet(memberships_path) if memberships_path.suffix.lower() == ".parquet" else pd.read_csv(memberships_path)
    required = {"ticker", "security_id", "effective_from_utc", "effective_to_utc"}
    missing_columns = sorted(required.difference(memberships.columns))
    if missing_columns:
        raise DataReadinessError(f"point-in-time memberships are missing columns: {missing_columns}")
    rows: list[dict[str, Any]] = []
    for record in memberships.itertuples(index=False):
        ticker = str(record.ticker).strip().upper()
        interval_start = pd.Timestamp(record.effective_from_utc).tz_convert(_NEW_YORK).date()
        interval_end = (
            pd.Timestamp(record.effective_to_utc).tz_convert(_NEW_YORK).date()
            if pd.notna(record.effective_to_utc)
            else date.fromordinal(end_date.toordinal() + 1)
        )
        expected = sorted(value for value in expected_market_sessions if interval_start <= value < interval_end)
        all_observed = session_dates.get(ticker, set())
        observed = sorted(set(expected).intersection(all_observed))
        missing = sorted(set(expected).difference(all_observed))
        gap_class = _gap_classification(
            expected=expected,
            observed=observed,
            missing=missing,
            all_observed=all_observed,
            unavailable=ticker in unavailable,
        )
        training_action = (
            "block"
            if gap_class in _BLOCKING_GAPS
            else "exclude_interval"
            if gap_class in {"source_observed_empty", "ticker_reuse_no_matching_history"}
            else "use_observed_sessions"
        )
        identity = artifact_identity.get(
            ticker,
            {
                "artifact_path": "",
                "artifact_sha256": "",
                "first_bar_date": "",
                "last_bar_date": "",
                "bar_count": 0,
            },
        )
        rows.append(
            {
                "schema_version": MARKET_HISTORY_AUDIT_SCHEMA,
                "ticker": ticker,
                "security_id": str(record.security_id),
                "effective_from_utc": pd.Timestamp(record.effective_from_utc).isoformat(),
                "effective_to_utc": (pd.Timestamp(record.effective_to_utc).isoformat() if pd.notna(record.effective_to_utc) else ""),
                "expected_member_sessions": len(expected),
                "observed_member_sessions": len(observed),
                "missing_member_sessions": len(missing),
                "coverage_rate": len(observed) / len(expected) if expected else 1.0,
                "first_missing_session": missing[0].isoformat() if missing else "",
                "last_missing_session": missing[-1].isoformat() if missing else "",
                "gap_class": gap_class,
                "training_action": training_action,
                **identity,
            }
        )
    report = pd.DataFrame(rows).sort_values(["effective_from_utc", "ticker"], kind="stable").reset_index(drop=True)
    blocking = report["training_action"].eq("block")
    expected_total = int(report["expected_member_sessions"].sum())
    observed_total = int(report["observed_member_sessions"].sum())
    summary = {
        "schema_version": MARKET_HISTORY_AUDIT_SCHEMA,
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "inputs": {
            "memberships_path": str(memberships_path.resolve()),
            "memberships_sha256": membership_sha,
            "collection_manifest_path": str(manifest_path.resolve()),
            "collection_manifest_sha256": file_sha256(manifest_path),
            "collection_request_sha256": str(request["request_sha256"]),
            "source_collections_sha256": file_sha256(ledger_path),
        },
        "interval_count": len(report),
        "expected_member_sessions": expected_total,
        "observed_member_sessions": observed_total,
        "missing_member_sessions": expected_total - observed_total,
        "coverage_rate": observed_total / expected_total if expected_total else 1.0,
        "gap_classes": {str(key): int(value) for key, value in report["gap_class"].value_counts(dropna=False).sort_index().items()},
        "blocking_interval_count": int(blocking.sum()),
        "excluded_interval_count": int(report["training_action"].eq("exclude_interval").sum()),
        "benchmark_audit": benchmark_audit,
        "training_ready": not bool(blocking.any()) and not bool(benchmark_audit["blocking_symbols"]),
        "memory": memory_audit(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    return report, summary


def _audit_benchmarks(
    session_dates: dict[str, set[date]],
    expected_market_sessions: set[date],
    unavailable: set[str],
    benchmarks: tuple[str, ...],
) -> dict[str, Any]:
    rows: list[dict[str, Any]] = []
    for ticker in benchmarks:
        observed = session_dates.get(ticker, set())
        missing = expected_market_sessions.difference(observed)
        rows.append(
            {
                "ticker": ticker,
                "expected_sessions": len(expected_market_sessions),
                "observed_sessions": len(expected_market_sessions.intersection(observed)),
                "missing_sessions": len(missing),
                "source_observed_empty": ticker in unavailable,
            }
        )
    return {
        "symbols": rows,
        "blocking_symbols": [str(row["ticker"]) for row in rows if int(row["missing_sessions"]) > 0 or bool(row["source_observed_empty"])],
    }


def _gap_classification(
    *,
    expected: list[date],
    observed: list[date],
    missing: list[date],
    all_observed: set[date],
    unavailable: bool,
) -> str:
    if not missing:
        return "complete"
    if unavailable:
        return "source_observed_empty"
    if not observed:
        if all_observed and expected and (max(all_observed) < min(expected) or min(all_observed) > max(expected)):
            return "ticker_reuse_no_matching_history"
        return "no_member_session_overlap"
    if all(value > max(observed) for value in missing):
        return "terminal_nontrading_gap"
    if all(value < min(observed) for value in missing):
        return "initial_nontrading_gap"
    return "interior_gap"


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"expected a JSON object: {path}")
    return payload
