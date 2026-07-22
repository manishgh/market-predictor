from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Literal

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF
from market_predictor.intraday.contracts import (
    CATALYST_AUDIT_FEATURES,
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_FEATURES,
)
from market_predictor.swing.contracts import (
    CATALYST_FEATURES,
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_FEATURES,
)

LiveMode = Literal["swing", "intraday"]

LIVE_ARTIFACT_TYPES: dict[LiveMode, str] = {
    "swing": "swing_inference_features",
    "intraday": "intraday_inference_features",
}
LIVE_SCHEMA_VERSIONS: dict[LiveMode, str] = {
    "swing": SWING_FEATURE_SCHEMA_VERSION,
    "intraday": INTRADAY_FEATURE_SCHEMA_VERSION,
}
_FORBIDDEN_PREFIXES = (
    "concurrent_label_",
    "entry_",
    "exit_",
    "future_",
    "label_",
    "overlap_",
    "path_",
    "target_",
)
_FORBIDDEN_COLUMNS = {
    "entry_time_utc",
    "entry_price",
    "exit_time_utc",
    "exit_price",
    "stop_price",
    "target_price",
}


def select_and_audit_live_features(
    frame: pd.DataFrame,
    *,
    mode: LiveMode,
    required_price_feed: str,
    required_adjustment: str,
    minimum_bar_count: int,
    minimum_one_minute_bar_count: int | None = None,
    minimum_cross_section: int,
    source_coverage_max_age_minutes: int,
    required_global_sources: Sequence[str],
) -> tuple[pd.DataFrame, CanonicalAuditReport]:
    """Select one complete latest decision group and audit it for live scoring."""

    model_features = SWING_FEATURES if mode == "swing" else INTRADAY_MODEL_FEATURES
    catalyst_features = CATALYST_FEATURES if mode == "swing" else CATALYST_AUDIT_FEATURES
    schema_column = "swing_feature_schema_version" if mode == "swing" else "intraday_feature_schema_version"
    bar_count_column = "daily_bar_count" if mode == "swing" else "five_minute_bar_count"
    required = {
        "ticker",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "feature_eligible",
        "cross_section_eligible",
        "price_feed",
        "adjustment",
        bar_count_column,
        schema_column,
        *model_features,
        *catalyst_features,
    }
    if mode == "intraday":
        required.update(
            {
                "bar_start_utc",
                "one_minute_available_at_utc",
                "one_minute_bar_count",
                "five_minute_history_exact",
                "one_minute_history_exact",
                "catalyst_eligible",
            }
        )
    else:
        required.update(
            {
                "session_date_et",
                "bar_available_at_utc",
                "prediction_cutoff_policy_id",
            }
        )
    missing = sorted(required.difference(frame.columns))
    if missing:
        report = CanonicalAuditReport(
            checks=(
                _check(
                    f"{mode}_live_schema",
                    1,
                    len(frame),
                    f"missing columns: {', '.join(missing)}",
                ),
            )
        )
        return frame.iloc[0:0].copy(), report

    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    cutoff = _utc(data["decision_time_utc"])
    feature = _utc(data["feature_available_at_utc"])
    latest_decision = cutoff.max()
    latest = cutoff.eq(latest_decision) & data["feature_eligible"].fillna(False).astype(bool)
    selected = data.loc[latest].copy()
    selected_cutoff = cutoff.loc[selected.index]
    selected_feature = feature.loc[selected.index]

    forbidden = forbidden_live_columns(data.columns)
    future_features = int((selected_feature > selected_cutoff).fillna(True).sum())
    availability_columns = [
        column
        for column in selected.columns
        if (column.endswith("available_at_utc") or "_available_at_utc_" in column) and not column.startswith("label_")
    ]
    for column in availability_columns:
        available = _utc(selected[column], allow_null=True)
        future_features += int((available.notna() & available.gt(selected_cutoff)).sum())

    source_failures = _source_failures(
        selected,
        selected_cutoff,
        source_coverage_max_age_minutes=source_coverage_max_age_minutes,
        required_global_sources=required_global_sources,
    )
    identity_failures = int(selected.duplicated(["ticker", "decision_time_utc"]).sum())
    decision_group_failures = max(0, selected["decision_group_id"].nunique() - 1)
    feed_failures = int(selected["price_feed"].astype(str).str.lower().ne(required_price_feed.lower()).sum())
    adjustment_failures = int(selected["adjustment"].astype(str).str.lower().ne(required_adjustment.lower()).sum())
    warm_failures = int(pd.to_numeric(selected[bar_count_column], errors="coerce").lt(minimum_bar_count).sum())
    if mode == "intraday":
        one_minute_minimum = minimum_one_minute_bar_count or minimum_bar_count
        warm_failures += int(pd.to_numeric(selected["one_minute_bar_count"], errors="coerce").lt(one_minute_minimum).sum())
        warm_failures += int(
            (~selected["five_minute_history_exact"].fillna(False).astype(bool)).sum()
            + (~selected["one_minute_history_exact"].fillna(False).astype(bool)).sum()
        )
    cross_section_failures = int(len(selected) < minimum_cross_section) + int(
        (~selected["cross_section_eligible"].fillna(False).astype(bool)).sum()
    )
    catalyst_failures = int((~selected["catalyst_eligible"].fillna(False).astype(bool)).sum()) if mode == "intraday" else 0
    schema_failures = int(selected[schema_column].astype(str).ne(LIVE_SCHEMA_VERSIONS[mode]).sum())
    timestamp_failures = int(selected_cutoff.isna().sum() + selected_feature.isna().sum())
    cutoff_contract_failures = 0
    if mode == "swing":
        bar_available = _utc(selected["bar_available_at_utc"])
        cutoff_contract_failures = int(
            bar_available.isna().sum()
            + bar_available.gt(selected_cutoff).sum()
            + selected["prediction_cutoff_policy_id"].astype(str).ne(SWING_NIGHTLY_CUTOFF.policy_id).sum()
        )
    checks = (
        _check(f"{mode}_live_schema", 0, len(selected), "frozen live feature columns are present"),
        _check(f"{mode}_live_rows", int(selected.empty), len(selected), "latest eligible decision group is not empty"),
        _check(f"{mode}_live_identity", identity_failures, len(selected), "ticker/decision identity is unique"),
        _check(
            f"{mode}_live_decision_group",
            decision_group_failures,
            len(selected),
            "all rows belong to one latest decision group",
        ),
        _check(f"{mode}_live_timestamps", timestamp_failures, len(selected), "timestamps are valid UTC"),
        _check(
            f"{mode}_live_cutoff_contract",
            cutoff_contract_failures,
            len(selected),
            "bar availability is separate and the frozen swing cutoff policy is preserved",
        ),
        _check(f"{mode}_live_no_future_features", future_features, len(selected), "all inputs were available by prediction cutoff"),
        _check(f"{mode}_live_no_labels", len(forbidden), len(data.columns), "live input contains no targets or future paths"),
        _check(f"{mode}_live_sources", source_failures, len(selected), "source coverage is observed and fresh"),
        _check(f"{mode}_live_feed", feed_failures, len(selected), "volume features use the required full feed"),
        _check(f"{mode}_live_adjustment", adjustment_failures, len(selected), "prices use the required adjustment"),
        _check(f"{mode}_live_warmup", warm_failures, len(selected), "technical histories are warm and exact"),
        _check(
            f"{mode}_live_cross_section",
            cross_section_failures,
            len(selected),
            f"latest group contains at least {minimum_cross_section} eligible symbols",
        ),
        _check(f"{mode}_live_catalyst", catalyst_failures, len(selected), "required catalyst sources are current"),
        _check(f"{mode}_live_schema_version", schema_failures, len(selected), "feature schema matches the frozen contract"),
    )
    if mode == "swing":
        selected["date"] = selected["session_date_et"]
    else:
        selected["date"] = selected["bar_start_utc"]
    return selected.sort_values("ticker", kind="stable").reset_index(drop=True), CanonicalAuditReport(checks=checks)


def live_feature_columns(mode: LiveMode) -> tuple[str, ...]:
    model_features = SWING_FEATURES if mode == "swing" else INTRADAY_MODEL_FEATURES
    catalyst_features = CATALYST_FEATURES if mode == "swing" else CATALYST_AUDIT_FEATURES
    return tuple(dict.fromkeys((*model_features, *catalyst_features)))


def forbidden_live_columns(columns: Iterable[str]) -> list[str]:
    return sorted(column for column in columns if column in _FORBIDDEN_COLUMNS or column.startswith(_FORBIDDEN_PREFIXES))


def _source_failures(
    frame: pd.DataFrame,
    decision: pd.Series,
    *,
    source_coverage_max_age_minutes: int,
    required_global_sources: Sequence[str],
) -> int:
    failures = 0
    max_age = pd.Timedelta(minutes=source_coverage_max_age_minutes)
    ticker_statuses = [
        column
        for column in frame.columns
        if column.startswith("source_status_") and not column.startswith("source_status_available_at_utc_")
    ]
    if not ticker_statuses:
        failures += max(1, len(frame))
    for status_column in ticker_statuses:
        source = status_column.removeprefix("source_status_")
        failures += _one_source_failures(
            frame,
            decision,
            status_column=status_column,
            available_column=f"source_status_available_at_utc_{source}",
            coverage_column=f"source_coverage_end_utc_{source}",
            max_age=max_age,
        )
    for source in required_global_sources:
        normalized = source.strip().lower()
        failures += _one_source_failures(
            frame,
            decision,
            status_column=f"global_source_status_{normalized}",
            available_column=f"global_source_status_available_at_utc_{normalized}",
            coverage_column=f"global_source_coverage_end_utc_{normalized}",
            max_age=max_age,
        )
    return failures


def _one_source_failures(
    frame: pd.DataFrame,
    decision: pd.Series,
    *,
    status_column: str,
    available_column: str,
    coverage_column: str,
    max_age: pd.Timedelta,
) -> int:
    if any(column not in frame.columns for column in (status_column, available_column, coverage_column)):
        return max(1, len(frame))
    status = frame[status_column].astype(str).str.lower().str.strip()
    available = _utc(frame[available_column], allow_null=True)
    coverage = _utc(frame[coverage_column], allow_null=True)
    stale = (
        available.isna()
        | coverage.isna()
        | available.gt(decision)
        | coverage.gt(available)
        | coverage.gt(decision)
        | decision.sub(coverage).gt(max_age)
    )
    return int((~status.isin({"observed", "observed_empty"}) | stale).sum())


def _utc(values: pd.Series, *, allow_null: bool = False) -> pd.Series:
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
