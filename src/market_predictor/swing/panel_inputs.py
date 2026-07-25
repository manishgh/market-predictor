from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_canonical_bars,
    audit_universe_memberships,
)
from market_predictor.canonical.normalize import canonicalize_universe_memberships
from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.swing.market_history import DEFAULT_BENCHMARKS
from market_predictor.swing.market_history_audit import MARKET_HISTORY_AUDIT_SCHEMA
from market_predictor.v3.errors import DataReadinessError

SWING_MARKET_PANEL_INPUT_SCHEMA = "swing.market_panel_inputs.v1"
_TERMINAL_COLLECTION_STATES = {"complete", "complete_with_gaps"}
_ALLOWED_TRAINING_ACTIONS = {"use_observed_sessions", "exclude_interval", "block"}


def build_swing_market_panel_inputs(
    *,
    memberships_path: Path,
    collection_dir: Path,
    coverage_report_path: Path,
    coverage_summary_path: Path,
    benchmarks: tuple[str, ...] = DEFAULT_BENCHMARKS,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    """Build hash-bound PIT stock, benchmark, and membership inputs without filling gaps."""

    collection_manifest_path = collection_dir / "_manifest.json"
    source_ledger_path = collection_dir / "_source_collections.parquet"
    required_paths = (
        memberships_path,
        collection_manifest_path,
        source_ledger_path,
        coverage_report_path,
        coverage_summary_path,
    )
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(path)

    summary = _load_json_object(coverage_summary_path)
    collection_manifest = _load_json_object(collection_manifest_path)
    _validate_bound_inputs(
        summary=summary,
        collection_manifest=collection_manifest,
        memberships_path=memberships_path,
        collection_manifest_path=collection_manifest_path,
        source_ledger_path=source_ledger_path,
        coverage_report_path=coverage_report_path,
    )
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="swing market panel input validation",
    )

    memberships = _read_frame(memberships_path)
    coverage = _read_frame(coverage_report_path)
    _validate_coverage_totals(coverage, summary)
    approved_memberships, exclusions = _approved_memberships(memberships, coverage)
    artifacts = _artifact_inventory(collection_manifest, collection_dir)
    normalized_benchmarks = tuple(dict.fromkeys(value.strip().upper() for value in benchmarks))

    stock_parts: list[pd.DataFrame] = []
    for ticker, intervals in approved_memberships.groupby("ticker", sort=True):
        bars = _load_verified_bars(
            ticker=str(ticker),
            artifacts=artifacts,
            require_sip=True,
        )
        selected = _filter_to_membership_intervals(bars, intervals)
        if not selected.empty:
            stock_parts.append(selected)
        assert_memory_budget(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
            stage=f"swing market panel stock filtering {ticker}",
        )

    if not stock_parts:
        raise DataReadinessError("no stock bars remain after point-in-time membership filtering")
    stock_bars = pd.concat(stock_parts, ignore_index=True).sort_values(
        ["bar_start_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)
    del stock_parts

    benchmark_parts = [
        _load_verified_bars(ticker=ticker, artifacts=artifacts, require_sip=True)
        for ticker in normalized_benchmarks
    ]
    benchmark_bars = pd.concat(benchmark_parts, ignore_index=True).sort_values(
        ["bar_start_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)
    del benchmark_parts

    canonical_memberships = _canonical_memberships(approved_memberships)
    stock_audit = CanonicalAuditReport(checks=audit_canonical_bars(stock_bars, require_sip=True))
    benchmark_audit = CanonicalAuditReport(checks=audit_canonical_bars(benchmark_bars, require_sip=True))
    membership_audit = CanonicalAuditReport(
        checks=audit_universe_memberships(canonical_memberships, require_observed=False)
    )
    stock_audit.raise_for_failure()
    benchmark_audit.raise_for_failure()
    membership_audit.raise_for_failure()

    expected_stock_rows = int(
        pd.to_numeric(
            approved_memberships["observed_member_sessions"],
            errors="raise",
        ).sum()
    )
    if len(stock_bars) != expected_stock_rows:
        raise DataReadinessError(
            "point-in-time stock row count does not reconcile with coverage evidence: "
            f"expected={expected_stock_rows}, observed={len(stock_bars)}"
        )
    expected_benchmark_rows = sum(
        int(record["observed_sessions"])
        for record in _benchmark_records(summary)
        if str(record["ticker"]).strip().upper() in normalized_benchmarks
    )
    if len(benchmark_bars) != expected_benchmark_rows:
        raise DataReadinessError(
            "benchmark row count does not reconcile with coverage evidence: "
            f"expected={expected_benchmark_rows}, observed={len(benchmark_bars)}"
        )

    audit = {
        "schema_version": SWING_MARKET_PANEL_INPUT_SCHEMA,
        "training_ready": True,
        "inputs": {
            "memberships_path": str(memberships_path.resolve()),
            "memberships_sha256": file_sha256(memberships_path),
            "collection_manifest_path": str(collection_manifest_path.resolve()),
            "collection_manifest_sha256": file_sha256(collection_manifest_path),
            "source_collections_path": str(source_ledger_path.resolve()),
            "source_collections_sha256": file_sha256(source_ledger_path),
            "coverage_report_path": str(coverage_report_path.resolve()),
            "coverage_report_sha256": file_sha256(coverage_report_path),
            "coverage_summary_path": str(coverage_summary_path.resolve()),
            "coverage_summary_sha256": file_sha256(coverage_summary_path),
        },
        "stock_rows": len(stock_bars),
        "stock_tickers": int(stock_bars["ticker"].nunique()),
        "benchmark_rows": len(benchmark_bars),
        "benchmark_tickers": sorted(benchmark_bars["ticker"].astype(str).unique()),
        "membership_intervals": len(canonical_memberships),
        "security_identities": int(canonical_memberships["security_id"].nunique()),
        "excluded_intervals": exclusions,
        "gap_policy": "preserve_observed_sessions_without_fill_or_interpolation",
        "membership_availability_policy": "provider_publication_proxy_research_only",
        "production_ready": {
            "stock_bars": True,
            "benchmark_bars": True,
            "memberships": False,
            "bundle": False,
        },
        "canonical_audits": {
            "stock_bars": [check.model_dump() for check in stock_audit.checks],
            "benchmark_bars": [check.model_dump() for check in benchmark_audit.checks],
            "memberships": [check.model_dump() for check in membership_audit.checks],
        },
        "memory": memory_audit(
            hard_budget_gib=memory_budget_gib,
            headroom_gib=memory_headroom_gib,
        ).to_record(),
    }
    return stock_bars, benchmark_bars, canonical_memberships, audit


def _validate_bound_inputs(
    *,
    summary: dict[str, Any],
    collection_manifest: dict[str, Any],
    memberships_path: Path,
    collection_manifest_path: Path,
    source_ledger_path: Path,
    coverage_report_path: Path,
) -> None:
    if summary.get("schema_version") != MARKET_HISTORY_AUDIT_SCHEMA:
        raise DataReadinessError("unsupported swing daily-history coverage summary schema")
    if summary.get("training_ready") is not True:
        raise DataReadinessError("daily-history coverage has not passed the training-readiness gate")
    if int(summary.get("blocking_interval_count", -1)) != 0:
        raise DataReadinessError("daily-history coverage contains blocking membership intervals")
    inputs = summary.get("inputs")
    report = summary.get("report")
    if not isinstance(inputs, dict) or not isinstance(report, dict):
        raise DataReadinessError("daily-history coverage summary is missing hash-bound inputs")
    expected_hashes = (
        ("memberships_sha256", memberships_path),
        ("collection_manifest_sha256", collection_manifest_path),
        ("source_collections_sha256", source_ledger_path),
    )
    for key, path in expected_hashes:
        if inputs.get(key) != file_sha256(path):
            raise DataReadinessError(f"daily-history coverage input hash does not match: {key}")
    if report.get("sha256") != file_sha256(coverage_report_path):
        raise DataReadinessError("daily-history coverage report hash does not match its summary")
    if collection_manifest.get("status") not in _TERMINAL_COLLECTION_STATES:
        raise DataReadinessError("daily-history collection is not terminal")
    if collection_manifest.get("request_sha256") != inputs.get("collection_request_sha256"):
        raise DataReadinessError("daily-history collection request identity does not match coverage")
    if collection_manifest.get("source_collections_sha256") != inputs.get("source_collections_sha256"):
        raise DataReadinessError("daily-history source ledger identity does not match coverage")


def _approved_memberships(
    memberships: pd.DataFrame,
    coverage: pd.DataFrame,
) -> tuple[pd.DataFrame, list[dict[str, str]]]:
    membership_required = {
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
        "sector",
        "industry",
        "market_cap_bucket",
        "liquidity_bucket",
        "primary_benchmark",
        "universe_snapshot_id",
    }
    coverage_required = {
        "schema_version",
        "ticker",
        "security_id",
        "effective_from_utc",
        "effective_to_utc",
        "training_action",
        "gap_class",
        "expected_member_sessions",
        "observed_member_sessions",
        "missing_member_sessions",
    }
    missing_memberships = sorted(membership_required.difference(memberships.columns))
    missing_coverage = sorted(coverage_required.difference(coverage.columns))
    if missing_memberships or missing_coverage:
        raise DataReadinessError(
            f"panel membership inputs are incomplete: memberships={missing_memberships}, coverage={missing_coverage}"
        )
    actions = set(coverage["training_action"].astype(str))
    if not actions.issubset(_ALLOWED_TRAINING_ACTIONS):
        raise DataReadinessError(f"coverage report contains unsupported training actions: {sorted(actions)}")
    if bool(coverage["training_action"].eq("block").any()):
        raise DataReadinessError("coverage report contains blocking membership intervals")
    if len(coverage) != len(memberships):
        raise DataReadinessError("coverage report does not contain exactly one row per membership interval")
    use_gap_classes = set(
        coverage.loc[
            coverage["training_action"].eq("use_observed_sessions"),
            "gap_class",
        ].astype(str)
    )
    if not use_gap_classes.issubset({"complete", "terminal_nontrading_gap"}):
        raise DataReadinessError(
            f"coverage report uses unsupported trainable gap classes: {sorted(use_gap_classes)}"
        )
    excluded = coverage.loc[coverage["training_action"].eq("exclude_interval")].copy()
    excluded_gap_classes = set(excluded["gap_class"].astype(str))
    if not excluded_gap_classes.issubset(
        {"source_observed_empty", "ticker_reuse_no_matching_history"}
    ):
        raise DataReadinessError(
            f"coverage report uses unsupported exclusion gap classes: {sorted(excluded_gap_classes)}"
        )
    excluded_observed = pd.to_numeric(
        excluded["observed_member_sessions"],
        errors="coerce",
    )
    if bool(excluded_observed.isna().any()) or bool(excluded_observed.ne(0).any()):
        raise DataReadinessError("excluded membership intervals must have zero observed member sessions")

    left = memberships.copy()
    right = coverage.copy()
    for frame in (left, right):
        frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
        frame["security_id"] = frame["security_id"].astype(str).str.strip()
        frame["effective_from_utc"] = pd.to_datetime(frame["effective_from_utc"], utc=True, errors="coerce")
        raw_end = frame["effective_to_utc"]
        frame["effective_to_utc"] = pd.to_datetime(raw_end, utc=True, errors="coerce")
        invalid_end = (
            raw_end.notna()
            & raw_end.astype(str).str.strip().ne("")
            & frame["effective_to_utc"].isna()
        )
        if bool(invalid_end.any()):
            raise DataReadinessError("membership or coverage interval ends are invalid")
    if bool(left["effective_from_utc"].isna().any()) or bool(right["effective_from_utc"].isna().any()):
        raise DataReadinessError("membership or coverage interval starts are invalid")
    right = right.loc[
        :,
        [
            "ticker",
            "security_id",
            "effective_from_utc",
            "effective_to_utc",
            "training_action",
            "gap_class",
            "observed_member_sessions",
        ],
    ]
    merged = left.merge(
        right,
        on=["ticker", "security_id", "effective_from_utc", "effective_to_utc"],
        how="left",
        validate="one_to_one",
    )
    if bool(merged["training_action"].isna().any()):
        raise DataReadinessError("coverage report membership identity does not match the membership artifact")
    excluded = merged.loc[merged["training_action"].eq("exclude_interval")]
    exclusions = [
        {
            "ticker": str(record.ticker),
            "security_id": str(record.security_id),
            "effective_from_utc": pd.Timestamp(record.effective_from_utc).isoformat(),
            "gap_class": str(record.gap_class),
        }
        for record in excluded.itertuples(index=False)
    ]
    approved = merged.loc[merged["training_action"].eq("use_observed_sessions")].copy()
    if approved.empty:
        raise DataReadinessError("all point-in-time membership intervals were excluded")
    return approved, exclusions


def _validate_coverage_totals(
    coverage: pd.DataFrame,
    summary: dict[str, Any],
) -> None:
    if "schema_version" not in coverage.columns:
        raise DataReadinessError("coverage report is missing schema_version")
    schemas = set(coverage["schema_version"].astype(str))
    if schemas != {MARKET_HISTORY_AUDIT_SCHEMA}:
        raise DataReadinessError(f"coverage report schema identity is invalid: {sorted(schemas)}")
    if len(coverage) != int(summary.get("interval_count", -1)):
        raise DataReadinessError("coverage report interval count does not match summary")
    metric_columns = (
        "expected_member_sessions",
        "observed_member_sessions",
        "missing_member_sessions",
    )
    for column in metric_columns:
        if column not in coverage.columns:
            raise DataReadinessError(f"coverage report is missing metric: {column}")
        values = pd.to_numeric(coverage[column], errors="coerce")
        if bool(values.isna().any()) or bool(values.lt(0).any()):
            raise DataReadinessError(f"coverage report metric is invalid: {column}")
        if int(values.sum()) != int(summary.get(column, -1)):
            raise DataReadinessError(f"coverage report metric does not match summary: {column}")
    excluded = int(coverage["training_action"].astype(str).eq("exclude_interval").sum())
    if excluded != int(summary.get("excluded_interval_count", -1)):
        raise DataReadinessError("coverage report exclusion count does not match summary")


def _artifact_inventory(
    collection_manifest: dict[str, Any],
    collection_dir: Path,
) -> dict[str, tuple[Path, str]]:
    records = collection_manifest.get("artifacts")
    if not isinstance(records, list) or len(records) != int(collection_manifest.get("artifact_count", -1)):
        raise DataReadinessError("daily-history artifact inventory is invalid")
    inventory: dict[str, tuple[Path, str]] = {}
    for value in records:
        if not isinstance(value, dict):
            raise DataReadinessError("daily-history artifact inventory contains an invalid record")
        ticker = str(value.get("ticker") or "").strip().upper()
        expected_hash = str(value.get("sha256") or "")
        path = Path(str(value.get("path") or ""))
        if not path.exists() and ticker:
            path = collection_dir / "bars" / "1d" / f"{ticker}.parquet"
        if not ticker or not expected_hash or ticker in inventory:
            raise DataReadinessError(f"daily-history artifact identity is invalid: {ticker or 'unknown'}")
        inventory[ticker] = (path, expected_hash)
    return inventory


def _load_verified_bars(
    *,
    ticker: str,
    artifacts: dict[str, tuple[Path, str]],
    require_sip: bool,
) -> pd.DataFrame:
    normalized = ticker.strip().upper()
    if normalized not in artifacts:
        raise DataReadinessError(f"daily-history artifact is absent for approved ticker: {normalized}")
    path, expected_hash = artifacts[normalized]
    if not path.exists() or file_sha256(path) != expected_hash:
        raise DataReadinessError(f"daily-history artifact hash does not match for {normalized}")
    bars, manifest = load_canonical_artifact(path, expected_type="bars")
    if manifest.get("artifact_sha256") != expected_hash:
        raise DataReadinessError(f"canonical daily-history manifest does not match for {normalized}")
    if set(bars["ticker"].astype(str).str.upper()) != {normalized}:
        raise DataReadinessError(f"daily-history ticker identity does not match for {normalized}")
    audit = CanonicalAuditReport(checks=audit_canonical_bars(bars, require_sip=require_sip))
    audit.raise_for_failure()
    return bars


def _filter_to_membership_intervals(
    bars: pd.DataFrame,
    intervals: pd.DataFrame,
) -> pd.DataFrame:
    starts = pd.to_datetime(bars["bar_start_utc"], utc=True, errors="coerce")
    selected = pd.Series(False, index=bars.index)
    for interval in intervals.itertuples(index=False):
        effective_from = pd.Timestamp(interval.effective_from_utc)
        effective_to = (
            pd.Timestamp(interval.effective_to_utc)
            if pd.notna(interval.effective_to_utc)
            else None
        )
        in_interval = starts.ge(effective_from)
        if effective_to is not None:
            in_interval &= starts.lt(effective_to)
        selected |= in_interval
    return bars.loc[selected].copy()


def _canonical_memberships(approved: pd.DataFrame) -> pd.DataFrame:
    frame = approved.copy()
    frame["available_at_utc"] = pd.to_datetime(frame["effective_from_utc"], utc=True)
    frame["source"] = "sp_global_primary_evidence"
    frame["availability_policy"] = "provider_publication_proxy"
    return canonicalize_universe_memberships(frame)


def _benchmark_records(summary: dict[str, Any]) -> Iterable[dict[str, Any]]:
    benchmark_audit = summary.get("benchmark_audit")
    if not isinstance(benchmark_audit, dict):
        raise DataReadinessError("coverage summary is missing benchmark audit evidence")
    records = benchmark_audit.get("symbols")
    if not isinstance(records, list):
        raise DataReadinessError("coverage summary benchmark audit is invalid")
    for record in records:
        if not isinstance(record, dict):
            raise DataReadinessError("coverage summary benchmark record is invalid")
        yield record


def _read_frame(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        return pd.read_parquet(path)
    if path.suffix.lower() == ".csv":
        return pd.read_csv(path)
    raise ValueError(f"unsupported table format: {path.suffix}")


def _load_json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"expected a JSON object: {path}")
    return payload
