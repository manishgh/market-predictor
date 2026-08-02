"""Fail-closed live construction for the frozen causal intraday feature vector."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.intraday_features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
    build_causal_intraday_features,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError

INTRADAY_LIVE_SCHEMA_VERSION: Final = "edge_rebuild.intraday_live.v1"
INTRADAY_LIVE_AUDIT_COLUMNS: Final = (
    "decision_id",
    "ticker",
    "security_id",
    "session_date_et",
    "volume_bar_number",
    "bar_start_utc",
    "bar_end_utc",
    "last_source_minute_utc",
    "feature_context_minute_utc",
    "available_at_utc",
    "feature_available_at_utc",
    "primary_benchmark",
    "universe_snapshot_id",
    "strategy_contract_sha256",
    "feature_schema_version",
    "source",
    "price_feed",
    "adjustment",
    "staleness_minutes",
    "maximum_staleness_minutes",
    "as_of_utc",
    "live_schema_version",
)


@dataclass(frozen=True)
class IntradayLiveFeatureBatch:
    """Model input and its row-aligned identity, kept as separate contracts."""

    model_features: pd.DataFrame
    audit_identity: pd.DataFrame


def build_live_intraday_features(
    completed_volume_bars: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    point_in_time_memberships: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
    as_of_utc: object,
) -> IntradayLiveFeatureBatch:
    """Build the newest exact eligible decision per ticker at ``as_of_utc``.

    All feature math is delegated to ``build_causal_intraday_features``. This
    boundary only enforces the live cutoff and freezes selection/output shape.
    It never falls back to an older eligible row when the newest decision lacks
    exact benchmark or other causal evidence.
    """

    cutoff = _utc_cutoff(as_of_utc)
    _reject_future_evidence(
        completed_volume_bars,
        label="completed volume bars",
        timestamp_columns=(
            "bar_start_utc",
            "bar_end_utc",
            "available_at_utc",
            "first_source_minute_utc",
            "last_source_minute_utc",
            "activation_time_utc",
        ),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        stock_one_minute_bars,
        label="stock one-minute bars",
        timestamp_columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        benchmark_one_minute_bars,
        label="benchmark one-minute bars",
        timestamp_columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        point_in_time_memberships,
        label="point-in-time memberships",
        timestamp_columns=("session_open_utc", "effective_from_utc"),
        cutoff=cutoff,
    )

    built = build_causal_intraday_features(
        completed_volume_bars,
        stock_one_minute_bars,
        benchmark_one_minute_bars,
        point_in_time_memberships,
        contract=contract,
        strategy_contract_sha256=strategy_contract_sha256,
    )
    available = built.loc[built["feature_available_at_utc"].le(cutoff)].copy()
    if available.empty:
        raise DataReadinessError("no completed intraday decision is available at the live cutoff")

    selected = (
        available.sort_values(
            ["ticker", "feature_available_at_utc", "session_date_et", "volume_bar_number"],
            kind="stable",
        )
        .groupby("ticker", sort=True, observed=True)
        .tail(1)
        .sort_values("ticker", kind="stable")
        .reset_index(drop=True)
    )
    rejected = selected.loc[~selected["feature_eligible"].astype(bool)]
    if not rejected.empty:
        details = ", ".join(
            f"{row.ticker}:{row.feature_ineligible_reason}"
            for row in rejected[["ticker", "feature_ineligible_reason"]]
            .head(10)
            .itertuples(index=False)
        )
        raise DataReadinessError(
            "newest live intraday decision is not exactly eligible; no stale fallback is allowed: "
            f"{details}"
        )

    maximum_staleness_minutes = (
        3.0 * 390.0 / float(contract.intraday.volume_bars_per_session_target)
    )
    staleness_minutes = (
        cutoff - selected["feature_available_at_utc"]
    ) / pd.Timedelta(minutes=1)
    if bool(staleness_minutes.gt(maximum_staleness_minutes).any()):
        stale = selected.loc[
            staleness_minutes.gt(maximum_staleness_minutes),
            ["ticker", "feature_available_at_utc"],
        ]
        details = ", ".join(
            f"{row.ticker}:{pd.Timestamp(row.feature_available_at_utc).isoformat()}"
            for row in stale.head(10).itertuples(index=False)
        )
        raise DataReadinessError(
            "newest live intraday decision is stale for as_of_utc: " + details
        )

    model_features = selected.loc[:, CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS].copy()
    if tuple(model_features.columns) != CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS:
        raise DataReadinessError("live intraday model feature order does not match the frozen contract")
    if not bool(np.isfinite(model_features.to_numpy(dtype="float64")).all()):
        raise DataReadinessError("live intraday model features must all be finite")
    for column in CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS:
        model_features[column] = model_features[column].astype("float32")

    audit = selected.loc[:, INTRADAY_LIVE_AUDIT_COLUMNS[1:-4]].copy()
    audit.insert(0, "decision_id", _decision_ids(audit))
    audit["staleness_minutes"] = staleness_minutes.to_numpy(dtype="float64")
    audit["maximum_staleness_minutes"] = maximum_staleness_minutes
    audit["as_of_utc"] = cutoff
    audit["live_schema_version"] = INTRADAY_LIVE_SCHEMA_VERSION
    audit = audit.loc[:, INTRADAY_LIVE_AUDIT_COLUMNS]
    if len(audit) != len(model_features):
        raise DataReadinessError("live intraday features and audit identity are not row-aligned")
    return IntradayLiveFeatureBatch(
        model_features=model_features.reset_index(drop=True),
        audit_identity=audit.reset_index(drop=True),
    )


def _utc_cutoff(value: object) -> pd.Timestamp:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("as_of_utc must be a valid timezone-aware timestamp") from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise DataReadinessError("as_of_utc must be a valid timezone-aware timestamp")
    return cutoff.tz_convert("UTC")


def _reject_future_evidence(
    frame: pd.DataFrame,
    *,
    label: str,
    timestamp_columns: tuple[str, ...],
    cutoff: pd.Timestamp,
) -> None:
    missing = sorted(set(timestamp_columns).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} are missing cutoff columns: {missing}")
    for column in timestamp_columns:
        try:
            timestamps = pd.to_datetime(frame[column], utc=True, errors="raise")
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(f"{label} contain an invalid {column}") from exc
        if bool(timestamps.isna().any()):
            raise DataReadinessError(f"{label} contain a missing {column}")
        if bool(timestamps.gt(cutoff).any()):
            raise DataReadinessError(
                f"{label} contain {column} evidence available after as_of_utc"
            )


def _decision_ids(audit: pd.DataFrame) -> pd.Series:
    identities = (
        audit["security_id"].astype(str)
        + "|"
        + audit["session_date_et"].astype(str)
        + "|"
        + audit["volume_bar_number"].astype("int64").astype(str)
        + "|"
        + audit["feature_available_at_utc"].map(lambda value: pd.Timestamp(value).isoformat())
        + "|"
        + audit["strategy_contract_sha256"].astype(str)
    )
    return identities.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest()).astype("string")
