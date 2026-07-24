from __future__ import annotations

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.intraday.contracts import (
    INTRADAY_MODEL_FEATURES,
    IntradayDatasetConfig,
    downside_target_column,
    excess_return_column,
    net_return_column,
    opportunity_target_column,
)


def audit_intraday_dataset(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> CanonicalAuditReport:
    horizon = config.horizon_minutes
    required = {
        "ticker",
        "decision_time_utc",
        "ticker_decision_time_utc",
        "cross_section_cutoff_utc",
        "feature_available_at_utc",
        "one_minute_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "label_window_end_utc",
        "decision_group_id",
        "nominal_decision_group_id",
        "session_date_et",
        "five_minute_bar_count",
        "one_minute_bar_count",
        "five_minute_history_exact",
        "one_minute_history_exact",
        "feature_eligible",
        "cross_section_eligible",
        "catalyst_eligible",
        "label_window_expected",
        "label_path_exact",
        "label_eligible",
        "independent_event_id",
        "concurrent_label_count",
        "overlap_weight",
        "price_feed",
        "adjustment",
        "spy_available_at_utc",
        "qqq_available_at_utc",
        "sector_available_at_utc",
        "intraday_feature_schema_version",
        opportunity_target_column(horizon),
        downside_target_column(horizon),
        f"path_timeout_{horizon}m",
        net_return_column(horizon),
        excess_return_column(horizon, "spy"),
        excess_return_column(horizon, "qqq"),
        excess_return_column(horizon, "sector"),
        *INTRADAY_MODEL_FEATURES,
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        return CanonicalAuditReport(
            checks=(
                _check(
                    "intraday_schema",
                    1,
                    len(frame),
                    f"missing columns: {', '.join(missing)}",
                ),
            )
        )

    data = frame.copy()
    decision = _utc(data["decision_time_utc"])
    ticker_decision = _utc(data["ticker_decision_time_utc"])
    cross_section_cutoff = _utc(data["cross_section_cutoff_utc"])
    feature = _utc(data["feature_available_at_utc"])
    one_minute_available = _utc(data["one_minute_available_at_utc"], allow_null=True)
    entry = _utc(data["entry_time_utc"], allow_null=True)
    exit_time = _utc(data["exit_time_utc"], allow_null=True)
    label_available = _utc(data["label_available_at_utc"], allow_null=True)
    label_window_end = _utc(data["label_window_end_utc"], allow_null=True)
    feature_eligible = data["feature_eligible"].fillna(False).astype(bool)
    label_expected = data["label_window_expected"].fillna(False).astype(bool)
    label_exact = data["label_path_exact"].fillna(False).astype(bool)
    label_eligible = data["label_eligible"].fillna(False).astype(bool)

    timestamp_failures = int(decision.isna().sum() + feature.isna().sum())
    future_features = int((feature > decision).fillna(True).sum())
    peer_cutoff_failures = int(
        (
            cross_section_cutoff.ne(decision)
            | ticker_decision.gt(cross_section_cutoff)
        )
        .fillna(True)
        .sum()
    )
    expected_cutoff = feature.groupby(data["nominal_decision_group_id"]).transform(
        "max"
    )
    peer_cutoff_failures += int(
        expected_cutoff.ne(cross_section_cutoff).fillna(True).sum()
    )
    availability_columns = [
        "membership_available_at_utc",
        "one_minute_available_at_utc",
        "latest_event_feature_available_at_utc",
        "global_event_feature_available_at_utc",
        "spy_available_at_utc",
        "qqq_available_at_utc",
        "sector_available_at_utc",
        *[column for column in data if column.startswith("source_status_available_at_utc_")],
        *[column for column in data if column.startswith("global_source_status_available_at_utc_")],
    ]
    for column in dict.fromkeys(availability_columns):
        if column not in data:
            continue
        available = _utc(data[column], allow_null=True)
        future_features += int((available.notna() & available.gt(decision)).sum())

    identity_failures = int(data.duplicated(["ticker", "decision_time_utc"]).sum())
    group_session_failures = int(data.groupby("decision_group_id", sort=False)["session_date_et"].nunique().gt(1).sum())
    benchmark_missing = int(
        data.loc[
            feature_eligible,
            ["spy_available_at_utc", "qqq_available_at_utc", "sector_available_at_utc"],
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    feed_failures = int(data.loc[feature_eligible, "price_feed"].astype(str).str.lower().ne(config.required_price_feed).sum())
    adjustment_failures = int(data.loc[feature_eligible, "adjustment"].astype(str).str.lower().ne(config.required_adjustment).sum())
    warm_failures = int(
        (
            feature_eligible
            & (
                data["five_minute_bar_count"].lt(config.min_five_minute_bars)
                | data["one_minute_bar_count"].lt(config.min_one_minute_bars)
                | ~data["five_minute_history_exact"].fillna(False).astype(bool)
                | ~data["one_minute_history_exact"].fillna(False).astype(bool)
            )
        ).sum()
    )
    cross_section_failures = int((feature_eligible & ~data["cross_section_eligible"].fillna(False).astype(bool)).sum())
    path_failures = int((feature_eligible & label_expected & ~label_exact).sum())
    label_order = label_eligible & (
        entry.isna()
        | exit_time.isna()
        | label_available.isna()
        | label_window_end.isna()
        | entry.lt(decision)
        | exit_time.le(entry)
        | label_available.lt(exit_time)
        | label_window_end.lt(exit_time)
    )
    label_order_failures = int(label_order.sum())
    opportunity = pd.to_numeric(data[opportunity_target_column(horizon)], errors="coerce")
    downside = pd.to_numeric(data[downside_target_column(horizon)], errors="coerce")
    timeout = pd.to_numeric(data[f"path_timeout_{horizon}m"], errors="coerce")
    label_value_failures = int(
        (
            label_eligible
            & (
                opportunity.isna()
                | downside.isna()
                | timeout.isna()
                | ~opportunity.isin({0, 1})
                | ~downside.isin({0, 1})
                | ~timeout.isin({0, 1})
                | opportunity.add(downside).add(timeout).ne(1)
            )
        ).sum()
    )
    gross = pd.to_numeric(
        data[f"path_realized_return_gross_{horizon}m"],
        errors="coerce",
    )
    net = pd.to_numeric(data[net_return_column(horizon)], errors="coerce")
    expected_net = gross - config.round_trip_cost_bps / 10_000.0
    cost_failures = int((label_eligible & ~np.isclose(net, expected_net, equal_nan=False, atol=1e-12)).sum())
    benchmark_label_scope = feature_eligible & label_exact
    benchmark_label_failures = int(
        data.loc[
            benchmark_label_scope,
            [
                excess_return_column(horizon, "spy"),
                excess_return_column(horizon, "qqq"),
                excess_return_column(horizon, "sector"),
            ],
        ]
        .isna()
        .any(axis=1)
        .sum()
    )
    overlap_failures = int(
        (
            label_exact
            & (
                pd.to_numeric(data["concurrent_label_count"], errors="coerce").lt(1)
                | pd.to_numeric(data["overlap_weight"], errors="coerce").le(0)
            )
        ).sum()
    )
    catalyst_future = int((data["catalyst_eligible"].fillna(False).astype(bool) & one_minute_available.gt(decision)).sum())
    schema_failures = int(data["intraday_feature_schema_version"].astype(str).ne(config.schema_version).sum())
    leakage_named_features = [
        feature for feature in INTRADAY_MODEL_FEATURES if feature.startswith(("target_", "path_", "entry_", "exit_", "label_", "future_"))
    ]

    eligible_rows = int(feature_eligible.sum())
    checks = (
        _check("intraday_schema", 0, len(data), "frozen intraday feature and label columns are present"),
        _check("intraday_rows", int(data.empty), len(data), "intraday dataset is not empty"),
        _check("intraday_identity", identity_failures, len(data), "ticker/decision identity is unique"),
        _check(
            "intraday_group_sessions",
            group_session_failures,
            data["decision_group_id"].nunique(),
            "decision groups never span sessions",
        ),
        _check("intraday_timestamps", timestamp_failures, len(data), "decision timestamps are valid UTC"),
        _check(
            "intraday_no_future_features",
            future_features,
            len(data),
            "all technical, benchmark, catalyst, and membership features were available",
        ),
        _check(
            "intraday_cross_section_availability",
            peer_cutoff_failures,
            len(data),
            "every peer row uses one cutoff at or after all contributing availability",
        ),
        _check(
            "intraday_benchmark_coverage",
            benchmark_missing,
            eligible_rows,
            "SPY, QQQ, and sector bars exactly cover eligible decisions",
        ),
        _check("intraday_price_feed", feed_failures, eligible_rows, "eligible volume features use SIP"),
        _check(
            "intraday_adjustment",
            adjustment_failures,
            eligible_rows,
            "eligible prices use all-event adjustment",
        ),
        _check(
            "intraday_warmup",
            warm_failures + int(eligible_rows == 0),
            len(data),
            "eligible rows have exact 5m and 1m warm-up history",
        ),
        _check(
            "intraday_cross_section",
            cross_section_failures,
            eligible_rows,
            f"eligible groups contain at least {config.minimum_cross_section} symbols",
        ),
        _check(
            "intraday_exact_label_path",
            path_failures,
            int((feature_eligible & label_expected).sum()),
            "expected execution paths contain consecutive one-minute bars",
        ),
        _check(
            "intraday_label_order",
            label_order_failures,
            int(label_eligible.sum()),
            "entry, exit, and label availability are causal",
        ),
        _check(
            "intraday_label_values",
            label_value_failures,
            int(label_eligible.sum()),
            "target, stop, and timeout outcomes are mutually exclusive",
        ),
        _check(
            "intraday_label_costs",
            cost_failures,
            int(label_eligible.sum()),
            "net path returns include the frozen round-trip cost",
        ),
        _check(
            "intraday_label_benchmarks",
            benchmark_label_failures,
            int(benchmark_label_scope.sum()),
            "SPY, QQQ, and sector returns use the exact trade interval",
        ),
        _check(
            "intraday_overlap_metadata",
            overlap_failures,
            int(label_exact.sum()),
            "overlapping outcomes carry weights and independent event identities",
        ),
        _check(
            "intraday_catalyst_timing",
            catalyst_future,
            int(data["catalyst_eligible"].sum()),
            "catalyst eligibility never uses future source state",
        ),
        _check(
            "intraday_feature_names",
            len(leakage_named_features),
            len(INTRADAY_MODEL_FEATURES),
            "model feature names exclude labels and future outcomes",
        ),
        _check(
            "intraday_schema_version",
            schema_failures,
            len(data),
            "intraday schema version matches",
        ),
    )
    return CanonicalAuditReport(checks=checks)


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
