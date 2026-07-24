from __future__ import annotations

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.label_reconciliation import (
    LABEL_IDENTITY_COLUMNS,
    replay_mismatch_count,
    stamped_material_hash_is_valid,
    swing_label_material_columns,
)
from market_predictor.swing.contracts import SWING_FEATURES, SwingDatasetConfig, swing_target_column
from market_predictor.swing.labels import add_exact_swing_labels


def audit_swing_dataset(
    frame: pd.DataFrame,
    config: SwingDatasetConfig,
    *,
    source_frame: pd.DataFrame | None = None,
    benchmark_bars: pd.DataFrame | None = None,
) -> CanonicalAuditReport:
    horizon = config.horizon_sessions
    required = {
        "ticker",
        "decision_time_utc",
        "feature_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "decision_group_id",
        "session_date_et",
        "daily_bar_count",
        "feature_eligible",
        "cross_section_eligible",
        "label_window_expected",
        "label_path_exact",
        "label_eligible",
        "horizon_sessions",
        "round_trip_cost_bps",
        "minimum_daily_bars",
        "price_feed",
        "adjustment",
        "spy_available_at_utc",
        "qqq_available_at_utc",
        "sector_available_at_utc",
        "swing_feature_schema_version",
        "label_material_sha256",
        "label_source_reconciliation_sha256",
        "label_source_reconciliation_errors",
        swing_target_column(horizon),
        f"future_net_return_{horizon}d",
        f"future_excess_return_{horizon}d_vs_spy",
        f"future_excess_return_{horizon}d_vs_qqq",
        f"future_excess_return_{horizon}d_vs_sector",
        *SWING_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        return CanonicalAuditReport(checks=(_check("swing_schema", 1, len(frame), f"missing columns: {', '.join(missing)}"),))

    data = frame.copy()
    decision = _utc(data["decision_time_utc"])
    feature_available = _utc(data["feature_available_at_utc"])
    entry = _utc(data["entry_time_utc"], allow_null=True)
    exit_time = _utc(data["exit_time_utc"], allow_null=True)
    label_available = _utc(data["label_available_at_utc"], allow_null=True)
    feature_eligible = data["feature_eligible"].fillna(False).astype(bool)
    warm_candidate = data["daily_bar_count"].ge(config.min_daily_bars)
    label_expected = data["label_window_expected"].fillna(False).astype(bool)
    label_exact = data["label_path_exact"].fillna(False).astype(bool)
    label_eligible = data["label_eligible"].fillna(False).astype(bool)

    timestamp_failures = int(decision.isna().sum() + feature_available.isna().sum())
    future_features = int((feature_available > decision).fillna(True).sum())
    feature_timestamp_columns = [
        "membership_available_at_utc",
        "latest_event_feature_available_at_utc",
        "global_event_feature_available_at_utc",
        "spy_available_at_utc",
        "qqq_available_at_utc",
        "sector_available_at_utc",
        *[column for column in data if column.startswith("fundamental_available_at_utc_")],
        *[column for column in data if column.startswith("source_status_available_at_utc_")],
        *[column for column in data if column.startswith("global_source_status_available_at_utc_")],
    ]
    for column in dict.fromkeys(feature_timestamp_columns):
        if column not in data.columns:
            continue
        available = _utc(data[column], allow_null=True)
        future_features += int((available.notna() & available.gt(decision)).sum())

    benchmark_missing = int(
        data.loc[
            warm_candidate,
            ["spy_available_at_utc", "qqq_available_at_utc", "sector_available_at_utc"],
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    source_failures = 0
    max_source_age = pd.Timedelta(minutes=config.source_coverage_max_age_minutes)
    status_columns = [column for column in data if column.startswith("source_status_") and not column.startswith("source_status_available")]
    for column in status_columns:
        source = column.removeprefix("source_status_")
        statuses = data.loc[feature_eligible, column].astype(str).str.lower().str.strip()
        coverage_column = f"source_coverage_end_utc_{source}"
        available_column = f"source_status_available_at_utc_{source}"
        if coverage_column not in data.columns or available_column not in data.columns:
            source_failures += max(1, int(feature_eligible.sum()))
            continue
        coverage = _utc(data.loc[feature_eligible, coverage_column], allow_null=True)
        available = _utc(data.loc[feature_eligible, available_column], allow_null=True)
        source_decision = decision.loc[feature_eligible]
        stale = (
            coverage.isna()
            | available.isna()
            | coverage.gt(available)
            | coverage.gt(source_decision)
            | source_decision.sub(coverage).gt(max_source_age)
        )
        source_failures += int((~statuses.isin({"observed", "observed_empty"}) | stale).sum())
    global_source_failures = 0
    for source in config.required_global_sources:
        normalized = source.strip().lower()
        status_column = f"global_source_status_{normalized}"
        available_column = f"global_source_status_available_at_utc_{normalized}"
        coverage_column = f"global_source_coverage_end_utc_{normalized}"
        if status_column not in data.columns or available_column not in data.columns or coverage_column not in data.columns:
            global_source_failures += max(1, int(feature_eligible.sum()))
            continue
        statuses = data.loc[feature_eligible, status_column].astype(str).str.lower().str.strip()
        available = _utc(data.loc[feature_eligible, available_column], allow_null=True)
        coverage = _utc(data.loc[feature_eligible, coverage_column], allow_null=True)
        source_decision = decision.loc[feature_eligible]
        stale = coverage.isna() | coverage.gt(available) | coverage.gt(source_decision) | source_decision.sub(coverage).gt(max_source_age)
        global_source_failures += int((~statuses.isin({"observed", "observed_empty"}) | available.isna() | stale).sum())

    feed_failures = int(data.loc[feature_eligible, "price_feed"].astype(str).str.lower().ne(config.required_price_feed).sum())
    adjustment_failures = int(data.loc[feature_eligible, "adjustment"].astype(str).str.lower().ne(config.required_adjustment).sum())
    identity_failures = int(data.duplicated(["ticker", "decision_time_utc"]).sum())
    warm_rows = int(feature_eligible.sum())
    cross_section_failures = int((feature_eligible & ~data["cross_section_eligible"].fillna(False).astype(bool)).sum())
    internal_path_failures = int((feature_eligible & label_expected & ~label_exact).sum())
    label_order = label_eligible & (
        entry.isna() | exit_time.isna() | label_available.isna() | entry.le(decision) | exit_time.le(entry) | label_available.lt(exit_time)
    )
    label_order_failures = int(label_order.sum())
    target = pd.to_numeric(data[swing_target_column(horizon)], errors="coerce")
    label_value_failures = int((label_eligible & (target.isna() | ~target.isin({0, 1}))).sum())
    schema_failures = int(data["swing_feature_schema_version"].astype(str).ne(config.schema_version).sum())
    material_columns = swing_label_material_columns(horizon)
    lineage_failures = int(
        not stamped_material_hash_is_valid(
            data,
            identity_columns=LABEL_IDENTITY_COLUMNS,
            material_columns=material_columns,
        )
    )
    stamped_errors = pd.to_numeric(
        data["label_source_reconciliation_errors"],
        errors="coerce",
    )
    lineage_failures += int(
        stamped_errors.isna().any() or stamped_errors.nunique(dropna=False) != 1 or stamped_errors.fillna(1).iloc[0] != 0
    )
    reconciliation_hashes = data["label_source_reconciliation_sha256"].fillna("").astype(str).unique()
    lineage_failures += int(len(reconciliation_hashes) != 1 or len(reconciliation_hashes[0]) != 64)
    if (source_frame is None) != (benchmark_bars is None):
        lineage_failures += 1
    elif source_frame is not None and benchmark_bars is not None:
        reproduced = add_exact_swing_labels(
            source_frame,
            benchmark_bars,
            config,
        )
        lineage_failures += replay_mismatch_count(
            data,
            reproduced,
            identity_columns=LABEL_IDENTITY_COLUMNS,
            material_columns=material_columns,
        )
    leakage_named_features = [
        feature for feature in SWING_FEATURES if feature.startswith(("future_", "target_", "entry_", "exit_", "label_"))
    ]

    checks = (
        _check("swing_schema", 0, len(data), "frozen swing feature and label columns are present"),
        _check("swing_rows", int(data.empty), len(data), "swing dataset is not empty"),
        _check("swing_identity", identity_failures, len(data), "ticker/decision identity is unique"),
        _check("swing_timestamps", timestamp_failures, len(data), "decision and feature timestamps are valid UTC"),
        _check("swing_no_future_features", future_features, len(data), "all features were available by decision time"),
        _check(
            "swing_benchmark_coverage",
            benchmark_missing,
            int(warm_candidate.sum()),
            "SPY, QQQ, and sector bars cover every warm candidate row",
        ),
        _check("swing_source_coverage", source_failures, warm_rows, "required event sources were observed"),
        _check(
            "swing_global_source_coverage",
            global_source_failures,
            warm_rows,
            "required global context sources were observed",
        ),
        _check("swing_price_feed", feed_failures, warm_rows, "eligible volume features use SIP"),
        _check("swing_adjustment", adjustment_failures, warm_rows, "eligible prices use adjusted bars"),
        _check("swing_warmup", int(warm_rows == 0), len(data), f"eligible rows have at least {config.min_daily_bars} bars"),
        _check(
            "swing_cross_section",
            cross_section_failures,
            warm_rows,
            f"eligible groups contain at least {config.minimum_cross_section} symbols",
        ),
        _check("swing_exact_label_path", internal_path_failures, int(label_expected.sum()), "future sessions are consecutive"),
        _check("swing_label_order", label_order_failures, int(label_eligible.sum()), "entry and exit follow decision"),
        _check("swing_label_values", label_value_failures, int(label_eligible.sum()), "eligible targets are binary"),
        _check(
            "swing_label_source_reconciliation",
            lineage_failures,
            len(data),
            "material labels reproduce from immutable daily stock and benchmark paths",
        ),
        _check("swing_feature_names", len(leakage_named_features), len(SWING_FEATURES), "feature names exclude labels"),
        _check("swing_schema_version", schema_failures, len(data), "swing schema version matches"),
    )
    return CanonicalAuditReport(checks=checks)


def audit_feature_availability_columns(frame: pd.DataFrame) -> list[str]:
    return [column for column in frame.columns if column.endswith("available_at_utc") and column not in {"label_available_at_utc"}]


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
