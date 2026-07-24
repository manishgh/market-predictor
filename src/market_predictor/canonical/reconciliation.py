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
ASSIGNMENT_SCHEMA_VERSION = "event_assignment.v1"
ASSIGNMENT_STATUSES: tuple[str, ...] = (
    "assigned",
    "duplicate_event_id",
    "ticker_not_in_decisions",
    "invalid_availability",
    "no_future_decision",
    "outside_all_windows",
)
ASSIGNMENT_COLUMNS: tuple[str, ...] = (
    "assignment_id",
    "event_id",
    "ticker",
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


def build_event_assignments(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
) -> pd.DataFrame:
    """Build immutable assignment evidence for all input event rows."""

    _require_columns(decisions, {"ticker", "decision_time_utc"}, "decisions")
    _require_columns(
        events,
        {"event_id", "ticker", "source_family", "feature_available_at_utc"},
        "events",
    )
    prepared_decisions = _prepare_decisions(decisions)
    prepared_events = events.copy()
    prepared_events["ticker"] = _ticker(prepared_events["ticker"])
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

    decision_tickers = set(prepared_decisions["ticker"])
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
        ticker = str(event["ticker"])
        if event_id in seen:
            records.append(_exclusion_record(event, "duplicate_event_id"))
            continue
        seen.add(event_id)
        if ticker not in decision_tickers:
            records.append(_exclusion_record(event, "ticker_not_in_decisions"))
            continue
        available = event["feature_available_at_utc"]
        if pd.isna(available):
            records.append(_exclusion_record(event, "invalid_availability"))
            continue
        ticker_decisions = prepared_decisions.loc[
            prepared_decisions["ticker"].eq(ticker)
            & prepared_decisions["decision_time_utc"].ge(available)
        ]
        if ticker_decisions.empty:
            records.append(_exclusion_record(event, "no_future_decision"))
            continue
        ticker_decisions = ticker_decisions.loc[
            ticker_decisions["decision_time_utc"].le(available + max_window)
        ]
        if ticker_decisions.empty:
            records.append(_exclusion_record(event, "outside_all_windows"))
            continue
        assigned = False
        for decision in ticker_decisions.to_dict(orient="records"):
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
) -> pd.DataFrame:
    """Rebuild all canonical event aggregates only from assignment evidence."""

    output = _prepare_decisions(decisions)
    _require_columns(assignments, set(ASSIGNMENT_COLUMNS), "event assignments")
    assigned = assignments.loc[assignments["status"].astype(str).eq("assigned")].copy()
    families = (
        sorted(
            assignments["source_family"].fillna("").astype(str).str.lower().unique()
        )
        if source_families is None
        else sorted({str(value).lower().strip() for value in source_families})
    )
    for name in windows:
        output[f"event_count_{name}"] = 0
        output[f"unknown_relevance_event_fraction_{name}"] = 0.0
        output[f"sentiment_mean_{name}"] = 0.0
        output[f"sentiment_coverage_{name}"] = 0.0
        output[f"event_relevance_mean_{name}"] = 0.0
        output[f"low_relevance_event_fraction_{name}"] = 0.0
        output[f"source_family_count_{name}"] = 0
        for family in families:
            output[f"source_count_{family}_{name}"] = 0.0
    output["latest_event_feature_available_at_utc"] = pd.Series(
        pd.NaT,
        index=output.index,
        dtype="datetime64[ns, UTC]",
    )
    if assigned.empty:
        return output

    assigned["sentiment_numeric"] = pd.to_numeric(
        assigned["sentiment_numeric"],
        errors="coerce",
    )
    assigned["relevance"] = pd.to_numeric(
        assigned["relevance"],
        errors="coerce",
    )
    for (decision_id, window_name), part in assigned.groupby(
        ["decision_id", "window_name"],
        sort=False,
    ):
        indices = output.index[output["decision_id"].eq(str(decision_id))]
        if len(indices) != 1 or window_name not in windows:
            continue
        index = indices[0]
        count = len(part)
        relevance = part["relevance"]
        relevance_known = relevance.notna()
        sentiment_present = part["sentiment_numeric"].notna()
        weighted_rows = relevance_known & sentiment_present
        weight = relevance.loc[weighted_rows].sum()
        weighted_sentiment = (
            part.loc[weighted_rows, "sentiment_numeric"]
            * relevance.loc[weighted_rows]
        ).sum()
        output.at[index, f"event_count_{window_name}"] = count
        output.at[index, f"unknown_relevance_event_fraction_{window_name}"] = (
            float((~relevance_known).sum()) / count
        )
        output.at[index, f"sentiment_mean_{window_name}"] = (
            float(weighted_sentiment / weight) if weight > 0 else 0.0
        )
        output.at[index, f"sentiment_coverage_{window_name}"] = (
            float(sentiment_present.sum()) / count
        )
        output.at[index, f"event_relevance_mean_{window_name}"] = (
            float(relevance.fillna(0.0).sum()) / count
        )
        output.at[index, f"low_relevance_event_fraction_{window_name}"] = (
            float((relevance.isna() | relevance.lt(0.5)).sum()) / count
        )
        observed_families = set(
            part["source_family"].fillna("").astype(str).str.lower()
        )
        observed_families.discard("")
        output.at[index, f"source_family_count_{window_name}"] = len(
            observed_families
        )
        for family in families:
            output.at[index, f"source_count_{family}_{window_name}"] = float(
                part["source_family"].astype(str).str.lower().eq(family).sum()
            )
    latest = (
        assigned.drop_duplicates(["decision_id", "event_id"])
        .groupby("decision_id")["feature_available_at_utc"]
        .max()
    )
    output["latest_event_feature_available_at_utc"] = output["decision_id"].map(
        latest
    )
    output["latest_event_feature_available_at_utc"] = pd.to_datetime(
        output["latest_event_feature_available_at_utc"],
        utc=True,
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


def _prepare_decisions(decisions: pd.DataFrame) -> pd.DataFrame:
    output = decisions.copy()
    output["ticker"] = _ticker(output["ticker"])
    output["decision_time_utc"] = pd.to_datetime(
        output["decision_time_utc"],
        utc=True,
        errors="coerce",
    )
    if bool(output["decision_time_utc"].isna().any()):
        raise DataReadinessError(
            "decision rows contain invalid event-assignment timestamps"
        )
    identities = output.apply(_decision_id, axis=1)
    if "decision_id" in output and bool(
        output["decision_id"].astype(str).ne(identities).any()
    ):
        raise DataReadinessError("decision_id does not match canonical identity")
    output["decision_id"] = identities
    if bool(output["decision_id"].duplicated().any()):
        raise DataReadinessError("canonical decision identity is not unique")
    return output


def _decision_id(row: pd.Series) -> str:
    fields = (
        str(row["ticker"]).strip().upper(),
        pd.Timestamp(row["decision_time_utc"]).isoformat(),
        str(row.get("prediction_cutoff_policy_id", "")),
        str(row.get("timeframe", "")),
        str(row.get("bar_start_utc", "")),
    )
    return hashlib.sha256("|".join(fields).encode("utf-8")).hexdigest()


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
        "ticker": str(event["ticker"]),
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
