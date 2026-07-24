from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Literal

import pandas as pd

from market_predictor.canonical.contracts import CANONICAL_SCHEMA_VERSION
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF, swing_prediction_cutoffs
from market_predictor.canonical.reconciliation import (
    DEFAULT_EVENT_WINDOWS,
    build_event_assignments,
    reproduce_event_features,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

MEMBERSHIP_VALUE_COLUMNS = (
    "sector",
    "industry",
    "market_cap_bucket",
    "liquidity_bucket",
    "primary_benchmark",
    "universe_snapshot_id",
    "source",
)
DecisionMode = Literal["swing-nightly", "intraday-bar-availability", "research-bar-availability"]
INTRADAY_BAR_AVAILABILITY_POLICY_ID = "intraday_bar_available_at_v1"
RESEARCH_BAR_AVAILABILITY_POLICY_ID = "research_bar_available_at_v1"


def decisions_from_completed_bars(bars: pd.DataFrame, *, mode: DecisionMode) -> pd.DataFrame:
    """Create explicit decision identities without conflating bar and prediction availability."""

    required = {
        "ticker",
        "timeframe",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "price_feed",
        "schema_version",
    }
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise SchemaMismatchError(f"canonical bars missing decision columns: {', '.join(missing)}")
    output = bars.copy()
    output["bar_available_at_utc"] = _utc_series(output["available_at_utc"])
    if bool(output["bar_available_at_utc"].isna().any()):
        raise DataReadinessError("canonical bars contain invalid availability timestamps")
    timeframes = output["timeframe"].astype(str).str.lower().str.strip()
    output["session_date_et"] = _utc_series(output["bar_start_utc"]).dt.tz_convert("America/New_York").dt.date
    if mode == "swing-nightly":
        invalid_timeframes = sorted(timeframes[timeframes.ne("1d")].unique())
        if invalid_timeframes:
            raise DataReadinessError(f"swing-nightly decisions require only 1d bars; received timeframes={invalid_timeframes}")
        output["decision_time_utc"] = swing_prediction_cutoffs(output["session_date_et"])
        output["prediction_cutoff_policy_id"] = SWING_NIGHTLY_CUTOFF.policy_id
        after_cutoff = output["bar_available_at_utc"].gt(output["decision_time_utc"])
        if bool(after_cutoff.any()):
            raise DataReadinessError(f"daily bars available after frozen swing cutoff: rows={int(after_cutoff.sum())}")
    elif mode == "intraday-bar-availability":
        if bool(timeframes.eq("1d").any()):
            raise DataReadinessError("intraday-bar-availability decisions reject daily bars")
        output["decision_time_utc"] = output["bar_available_at_utc"]
        output["prediction_cutoff_policy_id"] = INTRADAY_BAR_AVAILABILITY_POLICY_ID
    elif mode == "research-bar-availability":
        output["decision_time_utc"] = output["bar_available_at_utc"]
        output["prediction_cutoff_policy_id"] = RESEARCH_BAR_AVAILABILITY_POLICY_ID
    else:
        raise ValueError(f"unsupported decision mode: {mode}")
    output["feature_available_at_utc"] = output["bar_available_at_utc"]
    output["decision_group_id"] = output["decision_time_utc"].map(lambda value: value.isoformat())
    output["canonical_schema_version"] = CANONICAL_SCHEMA_VERSION
    return output.sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)


def join_universe_membership(decisions: pd.DataFrame, memberships: pd.DataFrame) -> pd.DataFrame:
    """Attach the single effective membership snapshot known at each decision."""

    decision_required = {"ticker", "decision_time_utc"}
    membership_required = {
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
        *MEMBERSHIP_VALUE_COLUMNS,
    }
    missing_decisions = sorted(decision_required.difference(decisions.columns))
    missing_memberships = sorted(membership_required.difference(memberships.columns))
    if missing_decisions or missing_memberships:
        raise SchemaMismatchError(f"membership join missing decisions={missing_decisions}, memberships={missing_memberships}")
    output = decisions.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["decision_time_utc"] = _utc_series(output["decision_time_utc"])
    if bool(output["decision_time_utc"].isna().any()):
        raise DataReadinessError("decision rows contain invalid timestamps")
    prepared = memberships.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper().str.strip()
    prepared["effective_from_utc"] = _utc_series(prepared["effective_from_utc"])
    prepared["effective_to_utc"] = _utc_series(prepared["effective_to_utc"])
    prepared["available_at_utc"] = _utc_series(prepared["available_at_utc"])
    invalid_required = prepared[["effective_from_utc", "available_at_utc"]].isna().any(axis=1)
    invalid_end = memberships["effective_to_utc"].notna() & prepared["effective_to_utc"].isna()
    if bool((invalid_required | invalid_end).any()):
        raise DataReadinessError("universe memberships contain invalid or timezone-naive timestamps")

    matches = pd.Series(0, index=output.index, dtype="int64")
    output["membership_effective_from_utc"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["membership_effective_to_utc"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    output["membership_available_at_utc"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    for column in MEMBERSHIP_VALUE_COLUMNS:
        output[column] = pd.NA

    for ticker, membership_part in prepared.groupby("ticker", sort=False):
        decision_indices = output.index[output["ticker"].eq(ticker)]
        if decision_indices.empty:
            continue
        decision_times = output.loc[decision_indices, "decision_time_utc"]
        for membership in membership_part.to_dict(orient="records"):
            eligible = decision_times.ge(membership["effective_from_utc"]) & decision_times.ge(membership["available_at_utc"])
            if pd.notna(membership["effective_to_utc"]):
                eligible &= decision_times.lt(membership["effective_to_utc"])
            selected = decision_indices[eligible.to_numpy()]
            if selected.empty:
                continue
            matches.loc[selected] += 1
            output.loc[selected, "membership_effective_from_utc"] = membership["effective_from_utc"]
            output.loc[selected, "membership_effective_to_utc"] = membership["effective_to_utc"]
            output.loc[selected, "membership_available_at_utc"] = membership["available_at_utc"]
            for column in MEMBERSHIP_VALUE_COLUMNS:
                output.loc[selected, column] = membership[column]

    uncovered = int(matches.eq(0).sum())
    ambiguous = int(matches.gt(1).sum())
    if uncovered or ambiguous:
        raise DataReadinessError(f"point-in-time universe membership failed: uncovered={uncovered}, ambiguous={ambiguous}")
    return output.sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)


def aggregate_event_features(
    decisions: pd.DataFrame,
    events: pd.DataFrame,
    *,
    windows: Mapping[str, pd.Timedelta] = DEFAULT_EVENT_WINDOWS,
    require_observed: bool = True,
) -> pd.DataFrame:
    """Join event counts and sentiment using feature availability, not publication time."""

    decision_required = {"ticker", "decision_time_utc"}
    event_required = {
        "ticker",
        "source_family",
        "feature_available_at_utc",
        "availability_policy",
        "event_id",
    }
    missing_decisions = sorted(decision_required.difference(decisions.columns))
    missing_events = sorted(event_required.difference(events.columns))
    if missing_decisions or missing_events:
        raise SchemaMismatchError(f"event join missing decisions={missing_decisions}, events={missing_events}")
    if require_observed and bool(events["availability_policy"].astype(str).ne("observed").any()):
        raise DataReadinessError("production event features reject provider publication proxy history")
    output = decisions.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["decision_time_utc"] = _utc_series(output["decision_time_utc"])
    if bool(output["decision_time_utc"].isna().any()):
        raise DataReadinessError("decision rows contain invalid or timezone-naive timestamps")
    clean = events.copy()
    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    clean["feature_available_at_utc"] = _utc_series(clean["feature_available_at_utc"])
    if bool(clean["feature_available_at_utc"].isna().any()):
        raise DataReadinessError("events contain invalid feature availability timestamps")
    clean = clean.sort_values(["ticker", "feature_available_at_utc"])
    clean["sentiment_numeric"] = pd.to_numeric(clean.get("sentiment_numeric"), errors="coerce")
    # Unknown relevance stays NaN (never coerced to fully-relevant); it is excluded
    # from relevance-weighted features and counted against quality downstream.
    clean["relevance"] = pd.to_numeric(clean.get("relevance"), errors="coerce").clip(lower=0)
    source_families = sorted(clean["source_family"].astype(str).str.lower().unique())
    assignments = build_event_assignments(output, clean, windows=windows)
    return reproduce_event_features(
        output,
        assignments,
        windows=windows,
        source_families=source_families,
    ).sort_values(["decision_time_utc", "ticker"]).reset_index(drop=True)


def join_source_collection_status(
    decisions: pd.DataFrame,
    collections: pd.DataFrame,
    *,
    source_families: Sequence[str],
) -> pd.DataFrame:
    """Attach the latest completed source attempt known at each decision time."""

    required = {
        "ticker",
        "source_family",
        "requested_end_utc",
        "completed_at_utc",
        "status",
        "row_count",
    }
    missing = sorted(required.difference(collections.columns))
    if missing:
        raise SchemaMismatchError(f"source collections missing columns: {', '.join(missing)}")
    output = decisions.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["decision_time_utc"] = _utc_series(output["decision_time_utc"])
    attempts = collections.copy()
    attempts["ticker"] = attempts["ticker"].astype(str).str.upper().str.strip()
    attempts["source_family"] = attempts["source_family"].astype(str).str.lower().str.strip()
    attempts["completed_at_utc"] = _utc_series(attempts["completed_at_utc"])
    attempts["requested_end_utc"] = _utc_series(attempts["requested_end_utc"])
    if bool(attempts[["completed_at_utc", "requested_end_utc"]].isna().any(axis=1).any()):
        raise DataReadinessError("source collections contain invalid coverage or completion timestamps")
    output["source_state_available_at_utc"] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
    for family in source_families:
        normalized_family = family.strip().lower()
        status_column = f"source_status_{normalized_family}"
        rows_column = f"source_observed_rows_{normalized_family}"
        available_column = f"source_status_available_at_utc_{normalized_family}"
        coverage_column = f"source_coverage_end_utc_{normalized_family}"
        output[status_column] = "not_collected"
        output[rows_column] = 0
        output[available_column] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
        output[coverage_column] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
        family_attempts = attempts[attempts["source_family"] == normalized_family]
        for ticker, indices in output.groupby("ticker", sort=False).groups.items():
            ticker_attempts = family_attempts[family_attempts["ticker"] == ticker].sort_values("completed_at_utc")
            if ticker_attempts.empty:
                continue
            decision_part = output.loc[indices].sort_values("decision_time_utc")
            joined = pd.merge_asof(
                decision_part[["decision_time_utc"]],
                ticker_attempts[["completed_at_utc", "requested_end_utc", "status", "row_count"]],
                left_on="decision_time_utc",
                right_on="completed_at_utc",
                direction="backward",
                allow_exact_matches=True,
            )
            output.loc[decision_part.index, status_column] = joined["status"].fillna("not_collected").to_numpy()
            output.loc[decision_part.index, rows_column] = joined["row_count"].fillna(0).astype(int).to_numpy()
            output.loc[decision_part.index, available_column] = joined["completed_at_utc"].array
            output.loc[decision_part.index, coverage_column] = joined["requested_end_utc"].array
    available_columns = [f"source_status_available_at_utc_{family.strip().lower()}" for family in source_families]
    output["source_state_available_at_utc"] = output[available_columns].max(axis=1)
    return output


def join_fundamentals_asof(
    decisions: pd.DataFrame,
    facts: pd.DataFrame,
    *,
    metrics: Sequence[str],
    require_observed: bool = True,
) -> pd.DataFrame:
    """Join the latest filed fact version known at each decision without snapshot backfill."""

    required = {"ticker", "metric", "value", "available_at_utc", "availability_policy", "fact_id"}
    missing = sorted(required.difference(facts.columns))
    if missing:
        raise SchemaMismatchError(f"fundamental facts missing columns: {', '.join(missing)}")
    if require_observed and bool(facts["availability_policy"].astype(str).ne("observed").any()):
        raise DataReadinessError("production fundamental features reject proxy availability")
    output = decisions.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["decision_time_utc"] = _utc_series(output["decision_time_utc"])
    prepared = facts.copy()
    prepared["ticker"] = prepared["ticker"].astype(str).str.upper().str.strip()
    prepared["metric"] = prepared["metric"].astype(str).str.lower().str.strip()
    prepared["available_at_utc"] = _utc_series(prepared["available_at_utc"])
    prepared = prepared.sort_values("available_at_utc").drop_duplicates("fact_id", keep="last")
    for metric in metrics:
        normalized_metric = metric.strip().lower()
        value_column = f"fundamental_{normalized_metric}"
        available_column = f"fundamental_available_at_utc_{normalized_metric}"
        output[value_column] = float("nan")
        output[available_column] = pd.Series(pd.NaT, index=output.index, dtype="datetime64[ns, UTC]")
        selected = prepared[prepared["metric"] == normalized_metric]
        for ticker, indices in output.groupby("ticker", sort=False).groups.items():
            ticker_facts = selected[selected["ticker"] == ticker].sort_values("available_at_utc")
            if ticker_facts.empty:
                continue
            decision_part = output.loc[indices].sort_values("decision_time_utc")
            joined = pd.merge_asof(
                decision_part[["decision_time_utc"]],
                ticker_facts[["available_at_utc", "value"]],
                left_on="decision_time_utc",
                right_on="available_at_utc",
                direction="backward",
                allow_exact_matches=True,
            )
            output.loc[decision_part.index, value_column] = joined["value"].to_numpy()
            output.loc[decision_part.index, available_column] = joined["available_at_utc"].array
    return output


def _utc_series(values: pd.Series) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    return pd.to_datetime(values.map(parse), utc=True)
