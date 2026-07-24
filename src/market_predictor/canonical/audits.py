from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Literal

import pandas as pd
from pydantic import Field

from market_predictor.canonical.contracts import (
    CANONICAL_SCHEMA_VERSION,
    CanonicalBar,
    CanonicalContract,
    CanonicalEvent,
    CanonicalFundamentalFact,
    CanonicalUniverseMembership,
    SourceCollection,
)
from market_predictor.v3.errors import DataReadinessError


class CanonicalAuditCheck(CanonicalContract):
    name: str
    status: Literal["pass", "fail", "not_run"]
    failures: int = Field(ge=0)
    rows_checked: int = Field(ge=0)
    detail: str


class CanonicalAuditReport(CanonicalContract):
    schema_version: str = "market_data.audit.v1"
    checks: tuple[CanonicalAuditCheck, ...]

    @property
    def passed(self) -> bool:
        return all(check.status == "pass" for check in self.checks)

    def raise_for_failure(self) -> None:
        failures = [check.name for check in self.checks if check.status != "pass"]
        if failures:
            raise DataReadinessError(f"canonical data audit failed: {', '.join(failures)}")

    def to_frame(self) -> pd.DataFrame:
        return pd.DataFrame([check.model_dump() for check in self.checks])


def audit_canonical_bars(frame: pd.DataFrame, *, require_sip: bool = True) -> tuple[CanonicalAuditCheck, ...]:
    required = set(CanonicalBar.model_fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_fail("bar_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    data = frame.copy()
    start = _utc_series(data["bar_start_utc"])
    end = _utc_series(data["bar_end_utc"])
    available = _utc_series(data["available_at_utc"])
    ingested = _utc_series(data["ingested_at_utc"])
    timestamp_failures = int(pd.concat([start, end, available, ingested], axis=1).isna().any(axis=1).sum())
    policies = data["availability_policy"].astype(str)
    observed = policies.eq("observed")
    ordering_failures = int(((end <= start) | (available < end) | (observed & (available < ingested))).fillna(True).sum())
    duplicates = int(data.assign(_start=start).duplicated(["ticker", "timeframe", "_start"]).sum())
    numeric = data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    ohlcv_failures = int(
        (
            numeric.isna().any(axis=1)
            | numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
            | numeric["volume"].lt(0)
            | numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
            | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
        ).sum()
    )
    feeds = data["price_feed"].fillna("unknown").astype(str).str.lower().str.strip()
    feed_failures = int(feeds.ne("sip").sum()) if require_sip else int((~feeds.isin({"sip", "iex", "unknown"})).sum())
    policy_failures = int((~policies.isin({"observed", "market_interval_close"})).sum())
    schemas = data["schema_version"].astype(str)
    schema_failures = int(schemas.ne(CANONICAL_SCHEMA_VERSION).sum())
    return (
        _check("bar_schema", 0, len(data), "canonical columns present"),
        _check("bar_rows", int(data.empty), len(data), "production bar artifact is not empty"),
        _check("bar_timestamps", timestamp_failures, len(data), "timestamps are timezone-aware UTC"),
        _check("bar_availability_order", ordering_failures, len(data), "bar_start < bar_end <= available"),
        _check("bar_identity", duplicates, len(data), "ticker/timeframe/bar_start is unique"),
        _check("bar_ohlcv", ohlcv_failures, len(data), "OHLCV is numeric and coherent"),
        _check("bar_price_feed", feed_failures, len(data), "SIP provenance is required" if require_sip else "feed is known"),
        _check("bar_availability_policy", policy_failures, len(data), "bar availability uses an approved policy"),
        _check("bar_schema_version", schema_failures, len(data), "schema version matches"),
    )


def audit_canonical_events(
    frame: pd.DataFrame,
    *,
    require_observed: bool = True,
) -> tuple[CanonicalAuditCheck, ...]:
    required = set(CanonicalEvent.model_fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_fail("event_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    data = frame.copy()
    published = _utc_series(data["published_at_utc"])
    updated = _utc_series(data["provider_updated_at_utc"], allow_null=True)
    first_seen = _utc_series(data["first_seen_at_utc"])
    available = _utc_series(data["available_at_utc"])
    scored = _utc_series(data["sentiment_scored_at_utc"], allow_null=True)
    feature_available = _utc_series(data["feature_available_at_utc"])
    timestamp_failures = int(pd.concat([published, first_seen, available, feature_available], axis=1).isna().any(axis=1).sum())
    observed = data["availability_policy"].astype(str).eq("observed")
    ordering = (
        (available < published)
        | (updated.notna() & available.lt(updated))
        | (observed & (available < first_seen))
        | (feature_available < available)
    )
    has_sentiment = pd.to_numeric(data.get("sentiment_numeric"), errors="coerce").notna()
    expected_feature = scored.where(scored.notna(), available)
    ordering |= (has_sentiment & scored.isna()) | feature_available.ne(expected_feature)
    ordering_failures = int(ordering.fillna(True).sum())
    proxy_failures = int((~observed).sum()) if require_observed else 0
    duplicates = int(data["event_id"].astype(str).duplicated().sum())
    identity = data[["event_id", "ticker", "source_family"]].fillna("").astype(str)
    identity_failures = int(identity.apply(lambda column: column.str.strip().eq("")).any(axis=1).sum())
    schema_failures = int(data["schema_version"].astype(str).ne(CANONICAL_SCHEMA_VERSION).sum())
    return (
        _check("event_schema", 0, len(data), "canonical columns present"),
        _check("event_timestamps", timestamp_failures, len(data), "required timestamps are timezone-aware UTC"),
        _check("event_availability_order", ordering_failures, len(data), "publication/observation/scoring precede features"),
        _check("event_observed_history", proxy_failures, len(data), "production requires observed first-seen history"),
        _check("event_identity", identity_failures + duplicates, len(data), "event identity is populated and unique"),
        _check("event_schema_version", schema_failures, len(data), "schema version matches"),
    )


def event_reconciliation_checks(summary: Mapping[str, int]) -> tuple[CanonicalAuditCheck, ...]:
    """Build the event-to-feature reconciliation audit check.

    The mandatory invariant is that no accepted event is left unexplained; the
    per-status counts (matched / duplicate / wrong_ticker / unavailable_future /
    unknown_relevance / irrelevant / outside_window) are recorded in the detail.
    """

    total = int(summary.get("total_events", 0))
    unexplained = int(summary.get("unexplained_events", 0))
    detail = ", ".join(f"{key}={summary[key]}" for key in sorted(summary))
    return (
        _check(
            "event_reconciliation",
            unexplained,
            total,
            f"every accepted event resolves to exactly one status ({detail})",
        ),
    )


def event_assignment_checks(
    assignment_summary: Mapping[str, int],
    aggregate_summary: Mapping[str, int],
) -> tuple[CanonicalAuditCheck, ...]:
    """Fail closed on altered assignments or unreproducible feature aggregates."""

    assignment_errors = int(
        assignment_summary.get("assignment_integrity_errors", 0)
    )
    aggregate_errors = int(
        aggregate_summary.get("aggregate_reconciliation_errors", 0)
    )
    assignment_detail = ", ".join(
        f"{key}={assignment_summary[key]}" for key in sorted(assignment_summary)
    )
    aggregate_detail = ", ".join(
        f"{key}={aggregate_summary[key]}" for key in sorted(aggregate_summary)
    )
    return (
        _check(
            "event_assignment_integrity",
            assignment_errors,
            int(assignment_summary.get("expected_assignment_rows", 0)),
            f"persisted assignments equal deterministic rebuild ({assignment_detail})",
        ),
        _check(
            "event_aggregate_reproduction",
            aggregate_errors,
            int(aggregate_summary.get("aggregate_cells_checked", 0)),
            f"decision aggregates reproduce from assignments ({aggregate_detail})",
        ),
    )


def audit_universe_memberships(
    frame: pd.DataFrame,
    *,
    decisions: pd.DataFrame | None = None,
    require_observed: bool = True,
) -> tuple[CanonicalAuditCheck, ...]:
    required = set(CanonicalUniverseMembership.model_fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_fail("universe_membership_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    records: list[CanonicalUniverseMembership] = []
    invalid = 0
    for row in frame.loc[:, list(CanonicalUniverseMembership.model_fields)].to_dict(orient="records"):
        try:
            if pd.isna(row.get("effective_to_utc")):
                row["effective_to_utc"] = None
            records.append(CanonicalUniverseMembership.model_validate(row))
        except ValueError:
            invalid += 1
    duplicate_columns = ["ticker", "effective_from_utc", "effective_to_utc", "universe_snapshot_id"]
    duplicates = int(frame.duplicated(duplicate_columns).sum())
    proxy = sum(record.availability_policy != "observed" for record in records) if require_observed else 0
    overlap = 0
    by_ticker: dict[str, list[CanonicalUniverseMembership]] = {}
    for record in records:
        by_ticker.setdefault(record.ticker, []).append(record)
    for ticker_records in by_ticker.values():
        prior_end: pd.Timestamp | None = None
        open_ended = False
        for record in sorted(ticker_records, key=lambda item: item.effective_from_utc):
            start = pd.Timestamp(record.effective_from_utc)
            if open_ended or (prior_end is not None and start < prior_end):
                overlap += 1
            open_ended = record.effective_to_utc is None
            prior_end = None if open_ended else pd.Timestamp(record.effective_to_utc)
    checks = [
        _check("universe_membership_rows", int(frame.empty), len(frame), "membership artifact is not empty"),
        _check(
            "universe_membership_schema",
            invalid + duplicates,
            len(frame),
            "membership rows are typed, versioned, and unique",
        ),
        _check("universe_membership_windows", overlap, len(frame), "effective windows do not overlap"),
        _check("universe_membership_observed", proxy, len(frame), "production membership is observed"),
    ]
    if decisions is not None:
        required_decisions = {"ticker", "decision_time_utc"}
        missing_decisions = sorted(required_decisions.difference(decisions.columns))
        if missing_decisions:
            checks.append(
                _fail("universe_membership_coverage", len(decisions), f"missing decision columns: {missing_decisions}")
            )
        else:
            uncovered = 0
            ambiguous = 0
            for decision in decisions.loc[:, ["ticker", "decision_time_utc"]].to_dict(orient="records"):
                try:
                    moment = pd.Timestamp(decision["decision_time_utc"])
                    if moment.tzinfo is None:
                        raise ValueError
                    moment = moment.tz_convert("UTC")
                except (TypeError, ValueError):
                    uncovered += 1
                    continue
                ticker = str(decision["ticker"]).strip().upper().replace("/", ".")
                matches = sum(
                    record.effective_from_utc <= moment
                    and (record.effective_to_utc is None or moment < record.effective_to_utc)
                    and record.available_at_utc <= moment
                    for record in by_ticker.get(ticker, [])
                )
                uncovered += int(matches == 0)
                ambiguous += int(matches > 1)
            checks.append(
                _check(
                    "universe_membership_coverage",
                    uncovered + ambiguous,
                    len(decisions),
                    f"every decision has exactly one known membership; uncovered={uncovered}, ambiguous={ambiguous}",
                )
            )
    return tuple(checks)


def audit_source_collections(
    frame: pd.DataFrame,
    *,
    required_tickers: Iterable[str] = (),
    required_sources: Iterable[str] = (),
    require_success: bool = True,
) -> tuple[CanonicalAuditCheck, ...]:
    required_columns = set(SourceCollection.model_fields)
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        return (_fail("source_collection_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    invalid = 0
    records: list[SourceCollection] = []
    for row in frame.to_dict(orient="records"):
        try:
            records.append(SourceCollection.model_validate(row))
        except ValueError:
            invalid += 1
    failures = sum(record.status in {"failed", "partial"} for record in records) if require_success else 0
    observed_pairs = {(record.ticker, record.source_family) for record in records if record.status in {"observed", "observed_empty"}}
    expected = {
        (str(ticker).strip().upper().replace("/", "."), str(source).strip().lower())
        for ticker in required_tickers
        for source in required_sources
    }
    missing_pairs = expected.difference(observed_pairs)
    return (
        _check("source_collection_rows", int(frame.empty), len(frame), "source collection artifact is not empty"),
        _check("source_collection_schema", invalid, len(frame), "collection records satisfy typed status semantics"),
        _check("source_collection_failures", failures, len(frame), "no required source collection failed"),
        _check("source_collection_coverage", len(missing_pairs), len(expected), "required ticker/source pairs were observed"),
    )


def audit_fundamental_facts(frame: pd.DataFrame, *, require_observed: bool = True) -> tuple[CanonicalAuditCheck, ...]:
    required = set(CanonicalFundamentalFact.model_fields)
    missing = sorted(required.difference(frame.columns))
    if missing:
        return (_fail("fundamental_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    invalid = 0
    proxy = 0
    for row in frame.to_dict(orient="records"):
        try:
            fact = CanonicalFundamentalFact.model_validate(row)
            proxy += int(require_observed and fact.availability_policy != "observed")
        except ValueError:
            invalid += 1
    duplicates = int(frame["fact_id"].astype(str).duplicated().sum())
    return (
        _check("fundamental_schema", invalid + duplicates, len(frame), "facts are typed, versioned, and unique"),
        _check("fundamental_observed_history", proxy, len(frame), "production fundamentals require observed availability"),
    )


def audit_decision_availability(
    frame: pd.DataFrame,
    *,
    feature_timestamp_columns: Iterable[str],
) -> tuple[CanonicalAuditCheck, ...]:
    if "decision_time_utc" not in frame.columns:
        return (_fail("decision_availability_schema", len(frame), "missing decision_time_utc"),)
    columns = list(feature_timestamp_columns)
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        return (_fail("decision_availability_schema", len(frame), f"missing feature timestamps: {', '.join(missing)}"),)
    decision = _utc_series(frame["decision_time_utc"])
    invalid = int(decision.isna().sum())
    future = 0
    for column in columns:
        available = _utc_series(frame[column], allow_null=True)
        future += int((available.notna() & available.gt(decision)).sum())
    return (
        _check("decision_rows", int(frame.empty), len(frame), "production decision artifact is not empty"),
        _check("decision_timestamps", invalid, len(frame), "decision timestamps are timezone-aware UTC"),
        _check("decision_no_future_features", future, len(frame), "every joined feature was available by decision time"),
    )


def audit_decision_source_coverage(
    frame: pd.DataFrame,
    *,
    required_sources: Iterable[str],
    max_coverage_age: pd.Timedelta = pd.Timedelta(minutes=60),
) -> tuple[CanonicalAuditCheck, ...]:
    sources = [str(source).strip().lower() for source in required_sources]
    required_columns = {
        column
        for source in sources
        for column in (
            f"source_status_{source}",
            f"source_status_available_at_utc_{source}",
            f"source_coverage_end_utc_{source}",
        )
    }
    missing = sorted(required_columns.difference(frame.columns))
    if missing:
        return (_fail("decision_source_coverage_schema", len(frame), f"missing columns: {', '.join(missing)}"),)
    failures = 0
    decision = _utc_series(frame["decision_time_utc"])
    for source in sources:
        statuses = frame[f"source_status_{source}"].astype(str).str.lower().str.strip()
        available = _utc_series(frame[f"source_status_available_at_utc_{source}"], allow_null=True)
        coverage = _utc_series(frame[f"source_coverage_end_utc_{source}"], allow_null=True)
        stale = (
            coverage.isna()
            | coverage.gt(available)
            | coverage.gt(decision)
            | decision.sub(coverage).gt(max_coverage_age)
        )
        failures += int((~statuses.isin({"observed", "observed_empty"}) | available.isna() | stale).sum())
    return (
        _check(
            "decision_source_coverage",
            failures,
            len(frame) * len(sources),
            "every required source was successfully observed through a fresh coverage end",
        ),
    )


def build_canonical_audit(
    *,
    bars: pd.DataFrame | None = None,
    events: pd.DataFrame | None = None,
    source_collections: pd.DataFrame | None = None,
    fundamentals: pd.DataFrame | None = None,
    memberships: pd.DataFrame | None = None,
    decisions: pd.DataFrame | None = None,
    decision_feature_timestamps: Iterable[str] = (),
    required_decision_sources: Iterable[str] = (),
    require_sip: bool = True,
    require_observed_events: bool = True,
) -> CanonicalAuditReport:
    checks: list[CanonicalAuditCheck] = []
    checks.extend(audit_canonical_bars(bars, require_sip=require_sip) if bars is not None else (_not_run("bars"),))
    checks.extend(
        audit_canonical_events(events, require_observed=require_observed_events) if events is not None else (_not_run("events"),)
    )
    checks.extend(audit_source_collections(source_collections) if source_collections is not None else (_not_run("sources"),))
    checks.extend(audit_fundamental_facts(fundamentals) if fundamentals is not None else (_not_run("fundamentals"),))
    checks.extend(
        audit_universe_memberships(memberships, decisions=decisions)
        if memberships is not None
        else (_not_run("memberships"),)
    )
    checks.extend(
        audit_decision_availability(decisions, feature_timestamp_columns=decision_feature_timestamps)
        if decisions is not None
        else (_not_run("decisions"),)
    )
    checks.extend(
        audit_decision_source_coverage(decisions, required_sources=required_decision_sources)
        if decisions is not None
        else (_not_run("decision_sources"),)
    )
    return CanonicalAuditReport(checks=tuple(checks))


def _utc_series(values: pd.Series, *, allow_null: bool = False) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        if allow_null and (value is None or pd.isna(value)):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    return pd.to_datetime(values.map(parse), utc=True)


def _check(name: str, failures: int, rows: int, detail: str) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(
        name=name,
        status="pass" if failures == 0 else "fail",
        failures=int(failures),
        rows_checked=int(rows),
        detail=detail,
    )


def _fail(name: str, rows: int, detail: str) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(name=name, status="fail", failures=1, rows_checked=int(rows), detail=detail)


def _not_run(name: str) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(name=name, status="not_run", failures=0, rows_checked=0, detail="input not supplied")
