"""Exact event-to-decision assignment and aggregate reconciliation.

The assignment artifact is the evidence behind every canonical event feature.
Assigned rows identify the exact decision and lookback window that consumed an
event. Events that are not assigned receive one deterministic exclusion status.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

DEFAULT_EVENT_WINDOWS: Mapping[str, pd.Timedelta] = {
    "2h": pd.Timedelta(hours=2),
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
}
ASSIGNMENT_SCHEMA_VERSION = "event_assignment.v3"
ASSIGNMENT_STATUSES: tuple[str, ...] = (
    "assigned",
    "duplicate_event_id",
    "security_not_in_decisions",
    "invalid_availability",
    "no_future_decision",
    "outside_all_windows",
)
ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "event_id",
    "ticker",
    "security_id",
    "source_family",
    "feature_available_at_utc",
    "decision_id",
    "decision_time_utc",
    "window_name",
    "window_seconds",
    "status",
    "sentiment_numeric",
    "relevance",
    "schema_version",
)
ASSIGNMENT_FEATURE_COLUMNS: tuple[str, ...] = (
    "source_family",
    "feature_available_at_utc",
    "decision_id",
    "window_name",
    "status",
    "sentiment_numeric",
    "relevance",
)


def stamp_canonical_decision_ids(
    decisions: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    """Validate decision identity inputs and attach their canonical hashes."""

    return _prepare_decisions(decisions, inplace=inplace)


def build_event_assignments(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
) -> pd.DataFrame:
    """Build immutable assignment evidence for all input event rows."""

    _require_columns(
        decisions,
        {"ticker", "security_id", "decision_time_utc"},
        "decisions",
    )
    _require_columns(
        events,
        {
            "event_id",
            "ticker",
            "security_id",
            "source_family",
            "feature_available_at_utc",
        },
        "events",
    )
    prepared_decisions = _prepare_decisions(decisions)
    prepared_events = events.copy()
    prepared_events["ticker"] = _ticker(prepared_events["ticker"])
    prepared_events["security_id"] = (
        prepared_events["security_id"].fillna("").astype(str).str.strip()
    )
    prepared_events["source_family"] = (
        prepared_events["source_family"].fillna("").astype(str).str.lower().str.strip()
    )
    prepared_events["feature_available_at_utc"] = pd.to_datetime(
        prepared_events["feature_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    prepared_events["sentiment_numeric"] = pd.to_numeric(
        prepared_events.get("sentiment_numeric"),
        errors="coerce",
    )
    prepared_events["relevance"] = pd.to_numeric(
        prepared_events.get("relevance"),
        errors="coerce",
    )

    ordered_decisions = prepared_decisions.sort_values(
        ["security_id", "decision_time_utc", "decision_id"],
        kind="stable",
    ).reset_index(drop=True)
    decision_times_ns = (
        pd.DatetimeIndex(ordered_decisions["decision_time_utc"])
        .as_unit("ns")
        .asi8
    )
    decision_ranges: dict[str, tuple[int, int]] = {}
    for security_id, indices in ordered_decisions.groupby(
        "security_id",
        sort=False,
    ).indices.items():
        positions = np.asarray(indices, dtype=np.int64)
        decision_ranges[str(security_id)] = (
            int(positions[0]),
            int(positions[-1]) + 1,
        )
    decision_security_ids = set(decision_ranges)
    records: list[dict[str, Any]] = []
    seen: set[str] = set()
    ordered_windows = tuple(
        sorted(
            ((str(name), pd.Timedelta(window)) for name, window in windows.items()),
            key=lambda item: (item[1], item[0]),
        )
    )
    max_window = max(window for _, window in ordered_windows)
    for event in prepared_events.to_dict(orient="records"):
        event_id = str(event["event_id"])
        security_id = str(event["security_id"])
        if event_id in seen:
            records.append(_exclusion_record(event, "duplicate_event_id"))
            continue
        seen.add(event_id)
        if security_id not in decision_security_ids:
            records.append(_exclusion_record(event, "security_not_in_decisions"))
            continue
        available = event["feature_available_at_utc"]
        if pd.isna(available):
            records.append(_exclusion_record(event, "invalid_availability"))
            continue
        range_start, range_end = decision_ranges[security_id]
        security_times = decision_times_ns[range_start:range_end]
        available_ns = pd.Timestamp(available).value
        first_future_offset = int(
            np.searchsorted(
                security_times,
                available_ns,
                side="left",
            )
        )
        if first_future_offset >= len(security_times):
            records.append(_exclusion_record(event, "no_future_decision"))
            continue
        last_window_offset = int(
            np.searchsorted(
                security_times,
                pd.Timestamp(available + max_window).value,
                side="right",
            )
        )
        if last_window_offset <= first_future_offset:
            records.append(_exclusion_record(event, "outside_all_windows"))
            continue
        security_decisions = ordered_decisions.iloc[
            range_start + first_future_offset : range_start + last_window_offset
        ]
        assigned = False
        for decision in security_decisions.to_dict(orient="records"):
            age = decision["decision_time_utc"] - available
            for window_name, window in ordered_windows:
                if age <= window:
                    assigned = True
                    records.append(
                        _assignment_record(
                            event,
                            decision,
                            window_name=window_name,
                            window=window,
                        )
                    )
        if not assigned:
            records.append(_exclusion_record(event, "outside_all_windows"))
    artifact = pd.DataFrame.from_records(records, columns=ASSIGNMENT_COLUMNS)
    if artifact.empty:
        return artifact
    return artifact.sort_values(
        ["event_id", "status", "decision_time_utc", "window_seconds"],
        na_position="last",
    ).reset_index(drop=True)


def reproduce_event_features(
    decisions: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
    source_families: list[str] | tuple[str, ...] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Rebuild all canonical event aggregates only from assignment evidence."""

    aggregates = aggregate_event_assignments(
        assignments,
        windows=windows,
        source_families=source_families,
    )
    return apply_event_assignment_features(
        decisions,
        aggregates,
        windows=windows,
        source_families=source_families,
        inplace=inplace,
    )


def aggregate_event_assignments(
    assignments: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
    source_families: list[str] | tuple[str, ...] | None = None,
) -> pd.DataFrame:
    """Reduce assignment evidence to decision-level sufficient statistics."""

    _require_columns(
        assignments,
        set(ASSIGNMENT_FEATURE_COLUMNS),
        "event assignments",
    )
    assigned_mask = assignments["status"].astype(str).eq("assigned")
    assignments_are_all_assigned = bool(assigned_mask.all())
    assigned = (
        assignments
        if assignments_are_all_assigned
        else assignments.loc[assigned_mask].copy()
    )
    families = (
        sorted(
            assignments["source_family"].fillna("").astype(str).str.lower().unique()
        )
        if source_families is None
        else sorted({str(value).lower().strip() for value in source_families})
    )
    if assigned.empty:
        return pd.DataFrame(
            columns=[
                "decision_id",
                *event_feature_columns(
                    windows,
                    source_families=families,
                ),
            ]
        )

    assigned["sentiment_numeric"] = pd.to_numeric(
        assigned["sentiment_numeric"],
        errors="coerce",
    )
    assigned["relevance"] = pd.to_numeric(
        assigned["relevance"],
        errors="coerce",
    )
    valid_windows = set(windows)
    invalid_windows = sorted(
        set(assigned["window_name"].astype(str)).difference(valid_windows)
    )
    if invalid_windows:
        raise DataReadinessError(
            "event assignments contain unsupported windows: "
            + ", ".join(invalid_windows)
        )

    temporary_columns = (
        "_relevance_known",
        "_sentiment_present",
        "_sentiment_weight",
        "_weighted_sentiment",
        "_relevance_sum",
        "_low_relevance",
        "_source_family",
    )
    try:
        assigned["_relevance_known"] = assigned["relevance"].notna()
        assigned["_sentiment_present"] = assigned["sentiment_numeric"].notna()
        weighted = (
            assigned["_relevance_known"]
            & assigned["_sentiment_present"]
        )
        assigned["_sentiment_weight"] = assigned["relevance"].where(
            weighted,
            0.0,
        )
        assigned["_weighted_sentiment"] = (
            assigned["sentiment_numeric"].where(weighted, 0.0)
            * assigned["_sentiment_weight"]
        )
        assigned["_relevance_sum"] = assigned["relevance"].fillna(0.0)
        assigned["_low_relevance"] = (
            assigned["relevance"].isna()
            | assigned["relevance"].lt(0.5)
        )
        assigned["_source_family"] = (
            assigned["source_family"]
            .fillna("")
            .astype(str)
            .str.lower()
            .str.strip()
        )
        grouped = (
            assigned.groupby(
                ["decision_id", "window_name"],
                sort=False,
                observed=True,
            )
            .agg(
                event_count=("decision_id", "size"),
                unknown_relevance=(
                    "_relevance_known",
                    lambda values: (~values).sum(),
                ),
                sentiment_present=("_sentiment_present", "sum"),
                weighted_sentiment=("_weighted_sentiment", "sum"),
                sentiment_weight=("_sentiment_weight", "sum"),
                relevance_sum=("_relevance_sum", "sum"),
                low_relevance=("_low_relevance", "sum"),
                source_family_count=(
                    "_source_family",
                    lambda values: values[values.ne("")].nunique(),
                ),
            )
            .reset_index()
        )
        output = pd.DataFrame(
            {
                "decision_id": grouped["decision_id"]
                .astype(str)
                .drop_duplicates()
                .reset_index(drop=True)
            }
        )
        output_index = output["decision_id"]
        for name in windows:
            part = grouped.loc[
                grouped["window_name"].astype(str).eq(name)
            ].set_index("decision_id")
            count = pd.to_numeric(
                output_index.map(part["event_count"]),
                errors="coerce",
            ).fillna(0.0)
            output[f"event_count_{name}"] = count.astype("int64")
            for target, source in (
                ("unknown_relevance_event_fraction", "unknown_relevance"),
                ("sentiment_coverage", "sentiment_present"),
                ("event_relevance_mean", "relevance_sum"),
                ("low_relevance_event_fraction", "low_relevance"),
            ):
                numerator = pd.to_numeric(
                    output_index.map(part[source]),
                    errors="coerce",
                ).fillna(0.0)
                output[f"{target}_{name}"] = np.divide(
                    numerator,
                    count,
                    out=np.zeros(len(output), dtype=float),
                    where=count.gt(0),
                )
            weighted_sentiment = pd.to_numeric(
                output_index.map(part["weighted_sentiment"]),
                errors="coerce",
            ).fillna(0.0)
            sentiment_weight = pd.to_numeric(
                output_index.map(part["sentiment_weight"]),
                errors="coerce",
            ).fillna(0.0)
            output[f"sentiment_mean_{name}"] = np.divide(
                weighted_sentiment,
                sentiment_weight,
                out=np.zeros(len(output), dtype=float),
                where=sentiment_weight.gt(0),
            )
            output[f"source_family_count_{name}"] = (
                pd.to_numeric(
                    output_index.map(part["source_family_count"]),
                    errors="coerce",
                )
                .fillna(0)
                .astype("int64")
            )
            family_counts = (
                assigned.loc[
                    assigned["window_name"].astype(str).eq(name)
                    & assigned["_source_family"].ne("")
                ]
                .groupby(
                    ["decision_id", "_source_family"],
                    sort=False,
                    observed=True,
                )
                .size()
            )
            for family in families:
                family_part = (
                    family_counts.xs(
                        family,
                        level="_source_family",
                        drop_level=True,
                    )
                    if family in family_counts.index.get_level_values(
                        "_source_family"
                    )
                    else pd.Series(dtype=float)
                )
                output[f"source_count_{family}_{name}"] = (
                    pd.to_numeric(
                        output_index.map(family_part),
                        errors="coerce",
                    )
                    .fillna(0.0)
                    .astype(float)
                )
        latest = assigned.groupby("decision_id")[
            "feature_available_at_utc"
        ].max()
        output["latest_event_feature_available_at_utc"] = output[
            "decision_id"
        ].map(latest)
        output["latest_event_feature_available_at_utc"] = pd.to_datetime(
            output["latest_event_feature_available_at_utc"],
            utc=True,
        )
    finally:
        if assignments_are_all_assigned:
            assignments.drop(
                columns=list(temporary_columns),
                inplace=True,
                errors="ignore",
            )
    return output


def apply_event_assignment_features(
    decisions: pd.DataFrame,
    aggregates: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
    source_families: list[str] | tuple[str, ...] | None = None,
    inplace: bool = False,
) -> pd.DataFrame:
    """Apply pre-aggregated assignment evidence to canonical decisions."""

    output = _prepare_decisions(decisions, inplace=inplace)
    families = (
        sorted(
            {
                column.removeprefix("source_count_").rsplit("_", 1)[0]
                for column in aggregates
                if column.startswith("source_count_")
                and any(column.endswith(f"_{name}") for name in windows)
            }
        )
        if source_families is None
        else sorted({str(value).lower().strip() for value in source_families})
    )
    feature_columns = event_feature_columns(
        windows,
        source_families=families,
    )
    _require_columns(
        aggregates,
        {"decision_id", *feature_columns},
        "event assignment aggregates",
    )
    if bool(aggregates["decision_id"].astype(str).duplicated().any()):
        raise DataReadinessError(
            "event assignment aggregates contain duplicate decisions"
        )
    decision_ids = set(output["decision_id"].astype(str))
    unknown_decisions = sorted(
        set(aggregates["decision_id"].astype(str)).difference(decision_ids)
    )
    if unknown_decisions:
        raise DataReadinessError(
            "assigned events reference decisions outside the feature population"
        )

    indexed = aggregates.set_index("decision_id")
    output_index = output["decision_id"].astype(str)
    for column in feature_columns:
        mapped = output_index.map(indexed[column])
        if column == "latest_event_feature_available_at_utc":
            output[column] = pd.to_datetime(
                mapped,
                utc=True,
                errors="coerce",
            )
        elif column.startswith(("event_count_", "source_family_count_")):
            output[column] = (
                pd.to_numeric(mapped, errors="coerce")
                .fillna(0)
                .astype("int64")
            )
        else:
            output[column] = (
                pd.to_numeric(mapped, errors="coerce")
                .fillna(0.0)
                .astype(float)
            )
    return output


def assignment_integrity_summary(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
) -> dict[str, int]:
    """Compare persisted assignments with a clean deterministic rebuild."""

    expected = build_event_assignments(decisions, events, windows=windows)
    expected_rows = _assignment_row_counts(expected)
    actual_rows = _assignment_row_counts(assignments)
    deleted = sum(max(expected_rows.get(key, 0) - actual_rows.get(key, 0), 0) for key in expected_rows)
    unexpected = sum(max(actual_rows.get(key, 0) - expected_rows.get(key, 0), 0) for key in actual_rows)
    duplicate_rows = int(assignments.duplicated(list(ASSIGNMENT_COLUMNS)).sum())
    invalid_status = int(
        (~assignments["status"].astype(str).isin(ASSIGNMENT_STATUSES)).sum()
    )
    return {
        "expected_assignment_rows": int(len(expected)),
        "actual_assignment_rows": int(len(assignments)),
        "deleted_assignment_rows": int(deleted),
        "unexpected_assignment_rows": int(unexpected),
        "duplicate_assignment_rows": duplicate_rows,
        "invalid_assignment_status_rows": invalid_status,
        "assignment_integrity_errors": int(
            deleted + unexpected + duplicate_rows + invalid_status
        ),
    }


def aggregate_reconciliation_summary(
    decisions_with_features: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
) -> dict[str, int]:
    """Independently reproduce event aggregates and count discrepant cells."""

    reproduced = reproduce_event_features(
        decisions_with_features,
        assignments,
        windows=windows,
    )
    feature_columns = event_feature_columns(
        windows,
        source_families=sorted(
            assignments["source_family"].fillna("").astype(str).str.lower().unique()
        ),
    )
    missing_columns = [
        column for column in feature_columns if column not in decisions_with_features
    ]
    mismatches = 0
    checked = 0
    for column in feature_columns:
        if column in missing_columns:
            continue
        left = decisions_with_features[column]
        right = reproduced[column]
        checked += len(left)
        if column == "latest_event_feature_available_at_utc":
            left_time = pd.to_datetime(left, utc=True, errors="coerce")
            right_time = pd.to_datetime(right, utc=True, errors="coerce")
            mismatches += int(
                (~((left_time == right_time) | (left_time.isna() & right_time.isna()))).sum()
            )
        else:
            left_numeric = pd.to_numeric(left, errors="coerce").to_numpy(dtype=float)
            right_numeric = pd.to_numeric(right, errors="coerce").to_numpy(dtype=float)
            mismatches += int(
                (~np.isclose(left_numeric, right_numeric, rtol=1e-10, atol=1e-12, equal_nan=True)).sum()
            )
    return {
        "aggregate_cells_checked": int(checked),
        "missing_aggregate_columns": int(len(missing_columns)),
        "aggregate_value_mismatches": int(mismatches),
        "aggregate_reconciliation_errors": int(len(missing_columns) + mismatches),
    }


def event_feature_columns(
    windows: Mapping[str, pd.Timedelta],
    *,
    source_families: list[str] | tuple[str, ...],
) -> list[str]:
    columns: list[str] = []
    for name in windows:
        columns.extend(
            [
                f"event_count_{name}",
                f"unknown_relevance_event_fraction_{name}",
                f"sentiment_mean_{name}",
                f"sentiment_coverage_{name}",
                f"event_relevance_mean_{name}",
                f"low_relevance_event_fraction_{name}",
                f"source_family_count_{name}",
            ]
        )
        columns.extend(
            f"source_count_{family}_{name}"
            for family in source_families
            if family
        )
    columns.append("latest_event_feature_available_at_utc")
    return columns


def reconciliation_summary(artifact: pd.DataFrame) -> dict[str, int]:
    counts = artifact["status"].astype(str).value_counts().to_dict()
    summary = {
        status: int(counts.get(status, 0)) for status in ASSIGNMENT_STATUSES
    }
    summary["total_assignment_rows"] = int(len(artifact))
    summary["total_events"] = int(artifact["event_id"].astype(str).nunique())
    summary["unexplained_events"] = int(
        (
            ~artifact["status"].astype(str).isin(ASSIGNMENT_STATUSES)
        ).sum()
    )
    summary["lineage_error_events"] = int(
        artifact["status"].astype(str).isin(
            {"duplicate_event_id", "invalid_availability"}
        ).sum()
    )
    return summary


def stamped_scalar(frame: pd.DataFrame, column: str, *, default: int = 0) -> int:
    """Read a per-dataset integer stamped as a constant column (or the default)."""

    if column in frame.columns and len(frame):
        return int(pd.to_numeric(frame[column], errors="coerce").fillna(default).iloc[0])
    return default


def stamped_hash(frame: pd.DataFrame, column: str) -> str:
    """Read a per-dataset hash stamped as a constant column (or empty string)."""

    if column in frame.columns and len(frame):
        value = str(frame[column].iloc[0])
        return value if value and value.lower() != "nan" else ""
    return ""


def reconciliation_sha256(artifact: pd.DataFrame) -> str:
    columns = [column for column in ASSIGNMENT_COLUMNS if column in artifact]
    payload = _normalized_records(artifact.loc[:, columns], columns)
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def event_aggregate_sha256(
    decisions_with_features: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
) -> str:
    """Hash the exact decision/event aggregate cells used by training."""

    _require_columns(
        decisions_with_features,
        {"decision_id"},
        "decisions with event features",
    )
    source_families = sorted(
        {
            column.removeprefix("source_count_").rsplit("_", 1)[0]
            for column in decisions_with_features
            if column.startswith("source_count_")
            and any(column.endswith(f"_{name}") for name in windows)
        }
    )
    columns = [
        "decision_id",
        *event_feature_columns(
            windows,
            source_families=source_families,
        ),
    ]
    _require_columns(
        decisions_with_features,
        set(columns),
        "decisions with event features",
    )
    payload = _normalized_records(
        decisions_with_features.loc[:, columns],
        columns,
    )
    return hashlib.sha256(
        json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()


def _prepare_decisions(
    decisions: pd.DataFrame,
    *,
    inplace: bool = False,
) -> pd.DataFrame:
    output = decisions if inplace else decisions.copy()
    normalized_ticker = _ticker(output["ticker"])
    if bool(output["ticker"].astype(str).ne(normalized_ticker).any()):
        output["ticker"] = normalized_ticker
    normalized_security = (
        output["security_id"].fillna("").astype(str).str.strip()
    )
    if bool(
        output["security_id"]
        .fillna("")
        .astype(str)
        .ne(normalized_security)
        .any()
    ):
        output["security_id"] = normalized_security
    if bool(output["security_id"].eq("").any()):
        raise DataReadinessError("decision rows contain empty security_id values")
    decision_time = pd.to_datetime(
        output["decision_time_utc"],
        utc=True,
        errors="coerce",
    )
    if bool(decision_time.isna().any()):
        raise DataReadinessError(
            "decision rows contain invalid event-assignment timestamps"
        )
    if not decision_time.equals(output["decision_time_utc"]):
        output["decision_time_utc"] = decision_time
    identities = _decision_identities(output)
    if "decision_id" in output and bool(
        output["decision_id"].astype(str).ne(identities).any()
    ):
        raise DataReadinessError("decision_id does not match canonical identity")
    output["decision_id"] = identities
    if bool(output["decision_id"].duplicated().any()):
        raise DataReadinessError("canonical decision identity is not unique")
    return output


def _decision_identities(
    frame: pd.DataFrame,
    *,
    chunk_rows: int = 50_000,
) -> pd.Series:
    identities = np.empty(len(frame), dtype=object)
    cutoff = (
        frame["prediction_cutoff_policy_id"]
        if "prediction_cutoff_policy_id" in frame
        else pd.Series("", index=frame.index)
    )
    timeframe = (
        frame["timeframe"]
        if "timeframe" in frame
        else pd.Series("", index=frame.index)
    )
    bar_start = (
        frame["bar_start_utc"]
        if "bar_start_utc" in frame
        else pd.Series("", index=frame.index)
    )
    for start in range(0, len(frame), chunk_rows):
        end = min(start + chunk_rows, len(frame))
        identities[start:end] = [
            hashlib.sha256(
                "|".join(
                    (
                        str(security_id).strip(),
                        str(ticker).strip().upper(),
                        pd.Timestamp(decision).isoformat(),
                        str(cutoff_id),
                        str(row_timeframe),
                        str(row_bar_start),
                    )
                ).encode("utf-8")
            ).hexdigest()
            for (
                security_id,
                ticker,
                decision,
                cutoff_id,
                row_timeframe,
                row_bar_start,
            ) in zip(
                frame["security_id"].iloc[start:end],
                frame["ticker"].iloc[start:end],
                frame["decision_time_utc"].iloc[start:end],
                cutoff.iloc[start:end],
                timeframe.iloc[start:end],
                bar_start.iloc[start:end],
                strict=True,
            )
        ]
    return pd.Series(identities, index=frame.index, dtype="string")


def _assignment_record(
    event: Mapping[str, Any],
    decision: Mapping[str, Any],
    *,
    window_name: str,
    window: pd.Timedelta,
) -> dict[str, Any]:
    assignment_id = hashlib.sha256(
        "|".join(
            (
                str(event["event_id"]),
                str(decision["decision_id"]),
                window_name,
            )
        ).encode("utf-8")
    ).hexdigest()
    return {
        "assignment_id": assignment_id,
        "event_id": str(event["event_id"]),
        "ticker": str(decision["ticker"]),
        "security_id": str(decision["security_id"]),
        "source_family": str(event["source_family"]),
        "feature_available_at_utc": event["feature_available_at_utc"],
        "decision_id": str(decision["decision_id"]),
        "decision_time_utc": decision["decision_time_utc"],
        "window_name": window_name,
        "window_seconds": int(window.total_seconds()),
        "status": "assigned",
        "sentiment_numeric": event["sentiment_numeric"],
        "relevance": event["relevance"],
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
    }


def _exclusion_record(
    event: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    assignment_id = hashlib.sha256(
        "|".join((str(event["event_id"]), status, "excluded")).encode("utf-8")
    ).hexdigest()
    return {
        "assignment_id": assignment_id,
        "event_id": str(event["event_id"]),
        "ticker": str(event["ticker"]),
        "security_id": str(event["security_id"]),
        "source_family": str(event["source_family"]),
        "feature_available_at_utc": event["feature_available_at_utc"],
        "decision_id": "",
        "decision_time_utc": pd.NaT,
        "window_name": "",
        "window_seconds": 0,
        "status": status,
        "sentiment_numeric": event["sentiment_numeric"],
        "relevance": event["relevance"],
        "schema_version": ASSIGNMENT_SCHEMA_VERSION,
    }


def _assignment_row_counts(frame: pd.DataFrame) -> dict[str, int]:
    columns = list(ASSIGNMENT_COLUMNS)
    _require_columns(frame, set(columns), "event assignments")
    counts: dict[str, int] = {}
    for record in _normalized_records(frame.loc[:, columns], columns):
        key = json.dumps(record, sort_keys=True, separators=(",", ":"))
        counts[key] = counts.get(key, 0) + 1
    return counts


def _normalized_records(
    frame: pd.DataFrame,
    columns: list[str],
) -> list[dict[str, str]]:
    normalized = frame.copy()
    for column in columns:
        if column.endswith("_utc"):
            values = pd.to_datetime(normalized[column], utc=True, errors="coerce")
            normalized[column] = values.map(
                lambda value: "" if pd.isna(value) else value.isoformat()
            )
        elif column in {"sentiment_numeric", "relevance"}:
            values = pd.to_numeric(normalized[column], errors="coerce")
            normalized[column] = values.map(
                lambda value: "" if pd.isna(value) else format(float(value), ".17g")
            )
        else:
            normalized[column] = normalized[column].fillna("").astype(str)
    records = normalized.sort_values(columns).to_dict(orient="records")
    return [
        {str(key): str(value) for key, value in record.items()}
        for record in records
    ]


def _require_columns(
    frame: pd.DataFrame,
    required: set[str],
    name: str,
) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"{name} missing columns: {', '.join(missing)}")


def _ticker(values: pd.Series) -> pd.Series:
    return values.astype(str).str.upper().str.strip().str.replace("/", ".", regex=False)
