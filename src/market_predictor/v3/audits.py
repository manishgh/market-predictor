from __future__ import annotations

from datetime import timedelta
from typing import Literal

import pandas as pd
from pydantic import Field

from market_predictor.v3.contracts import UniverseMembership
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract


class AuditCheck(FrozenContract):
    name: str
    status: Literal["pass", "fail", "warning", "not_run"]
    failures: int = Field(ge=0)
    rows_checked: int = Field(ge=0)
    detail: str = ""


class DataAuditReport(FrozenContract):
    schema_version: str = ML_V3_SCHEMA_VERSION
    checks: tuple[AuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status not in {"fail", "not_run"} for check in self.checks)

    def raise_for_failure(self) -> None:
        failed = [check.name for check in self.checks if check.status in {"fail", "not_run"}]
        if failed:
            raise DataReadinessError(f"V3 data audit failed: {', '.join(failed)}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.model_dump() for check in self.checks])


def audit_bars(frame: pd.DataFrame, *, interval: timedelta, require_sip: bool = True) -> tuple[AuditCheck, ...]:
    required = {"ticker", "timestamp", "open", "high", "low", "close", "volume", "price_feed"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_failed("bars_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    data = frame.copy()
    timestamp = _aware_utc_series(data["timestamp"])
    invalid_timestamp = int(timestamp.isna().sum())
    duplicate_count = int(data.assign(_timestamp=timestamp).duplicated(["ticker", "_timestamp"]).sum())
    numeric = data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid_numeric = numeric.isna().any(axis=1)
    invalid_ohlc = (
        (numeric[["open", "high", "low", "close"]] <= 0).any(axis=1)
        | (numeric["volume"] < 0)
        | (numeric["high"] < numeric[["open", "close", "low"]].max(axis=1))
        | (numeric["low"] > numeric[["open", "close", "high"]].min(axis=1))
    )
    gap_count = _intraday_gap_count(data.assign(_timestamp=timestamp), interval=interval)
    feeds = data["price_feed"].fillna("unknown").astype(str).str.lower().str.strip()
    wrong_feed = int(feeds.ne("sip").sum()) if require_sip else 0
    return (
        _check("bars_schema", 0, len(data), "required columns present"),
        _check("bars_timestamp", invalid_timestamp, len(data), "timestamps parse as UTC"),
        _check("bars_duplicates", duplicate_count, len(data), "unique ticker/timestamp"),
        _check("bars_gaps", gap_count, len(data), f"no within-session gaps over {interval}"),
        _check("bars_ohlcv", int((invalid_numeric | invalid_ohlc).sum()), len(data), "positive coherent OHLC and non-negative volume"),
        _check("bars_sip_feed", wrong_feed, len(data), "full SIP provenance required"),
    )


def audit_events(frame: pd.DataFrame) -> tuple[AuditCheck, ...]:
    required = {"event_id", "ticker", "published_at_utc", "ingested_at_utc", "source_family"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_failed("events_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    published = _aware_utc_series(frame["published_at_utc"])
    ingested = _aware_utc_series(frame["ingested_at_utc"])
    invalid = int((published.isna() | ingested.isna()).sum())
    ordering = int(((published > ingested) & published.notna() & ingested.notna()).sum())
    duplicates = int(frame["event_id"].astype(str).duplicated().sum())
    identity = frame[["event_id", "ticker", "source_family"]].fillna("").astype(str)
    missing_identity = int(identity.apply(lambda column: column.str.strip().eq("")).any(axis=1).sum())
    return (
        _check("events_schema", 0, len(frame), "required columns present"),
        _check("events_timestamp", invalid, len(frame), "publication and ingestion timestamps parse as UTC"),
        _check("events_ordering", ordering, len(frame), "publication is not after ingestion"),
        _check("events_duplicates", duplicates, len(frame), "event_id is unique"),
        _check("events_identity", missing_identity, len(frame), "event identity fields are populated"),
    )


def audit_universe(decisions: pd.DataFrame, memberships: pd.DataFrame) -> tuple[AuditCheck, ...]:
    decision_required = {"ticker", "decision_time_utc", "universe_snapshot_id"}
    membership_required = {name for name, field in UniverseMembership.model_fields.items() if field.is_required()}
    missing_decision = sorted(decision_required.difference(decisions.columns))
    missing_membership = sorted(membership_required.difference(memberships.columns))
    if missing_decision or missing_membership:
        detail = f"decision missing={missing_decision}; membership missing={missing_membership}"
        return (_failed("universe_schema", len(decisions), detail),)
    parsed: list[UniverseMembership] = []
    invalid_memberships = 0
    for record in memberships.to_dict(orient="records"):
        try:
            if pd.isna(record.get("effective_to_utc")):
                record["effective_to_utc"] = None
            parsed.append(UniverseMembership.model_validate(record))
        except ValueError:
            invalid_memberships += 1
    uncovered = 0
    for record in decisions.to_dict(orient="records"):
        try:
            moment = pd.Timestamp(record["decision_time_utc"])
            if moment.tzinfo is None:
                raise ValueError
            timestamp = moment.to_pydatetime()
            ticker = str(record["ticker"]).strip().upper()
            snapshot = str(record["universe_snapshot_id"])
            covered = any(item.ticker == ticker and item.universe_snapshot_id == snapshot and item.contains(timestamp) for item in parsed)
            uncovered += int(not covered)
        except (TypeError, ValueError):
            uncovered += 1
    return (
        _check("universe_memberships", invalid_memberships, len(memberships), "membership windows are valid"),
        _check("universe_point_in_time", uncovered, len(decisions), "every decision has effective membership"),
    )


def audit_benchmarks(decisions: pd.DataFrame) -> tuple[AuditCheck, ...]:
    required = {"ticker", "decision_time_utc", "primary_benchmark", "benchmark_close"}
    missing = sorted(required.difference(decisions.columns))
    if missing:
        return (_failed("benchmark_schema", len(decisions), f"missing columns: {', '.join(missing)}"),)
    missing_value = int(
        decisions[["primary_benchmark", "benchmark_close"]]
        .replace(r"^\s*$", pd.NA, regex=True)
        .isna()
        .any(axis=1)
        .sum()
    )
    return (_check("benchmark_coverage", missing_value, len(decisions), "benchmark mapping and price are populated"),)


def build_data_audit(
    *,
    bars: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    decisions: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    interval: timedelta = timedelta(minutes=5),
    require_sip: bool = True,
) -> DataAuditReport:
    checks: list[AuditCheck] = []
    checks.extend(audit_bars(bars, interval=interval, require_sip=require_sip) if bars is not None else (_not_run("bars"),))
    checks.extend(audit_events(events) if events is not None else (_not_run("events"),))
    if decisions is not None and memberships is not None:
        checks.extend(audit_universe(decisions, memberships))
    else:
        checks.append(_not_run("universe"))
    checks.extend(audit_benchmarks(decisions) if decisions is not None else (_not_run("benchmarks"),))
    return DataAuditReport(checks=tuple(checks))


def _intraday_gap_count(data: pd.DataFrame, *, interval: timedelta) -> int:
    valid = data.dropna(subset=["_timestamp"]).copy()
    if valid.empty:
        return 0
    valid["_session"] = valid["_timestamp"].dt.tz_convert("America/New_York").dt.date
    valid = valid.sort_values(["ticker", "_timestamp"])
    gaps = valid.groupby(["ticker", "_session"], sort=False)["_timestamp"].diff()
    return int((gaps > interval).sum())


def _aware_utc_series(values: pd.Series) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    return values.map(parse)


def _check(name: str, failures: int, rows: int, detail: str) -> AuditCheck:
    return AuditCheck(name=name, status="pass" if failures == 0 else "fail", failures=failures, rows_checked=rows, detail=detail)


def _failed(name: str, rows: int, detail: str) -> AuditCheck:
    return AuditCheck(name=name, status="fail", failures=1, rows_checked=rows, detail=detail)


def _not_run(name: str) -> AuditCheck:
    return AuditCheck(name=name, status="not_run", failures=0, rows_checked=0, detail="input not supplied")
