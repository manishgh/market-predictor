"""Fail-closed live adapter for the shared fixed-cohort bar transformation."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Final

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.features.bar_features import (
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
    build_causal_intraday_bar_features,
)
from market_predictor.modeling.strategy_contract import StrategyContract

INTRADAY_BAR_LIVE_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_live.v1"
INTRADAY_BAR_LIVE_AUDIT_COLUMNS: Final = (
    "decision_id",
    "decision_cohort_id",
    "ticker",
    "security_id",
    "session_date_et",
    "decision_time_utc",
    "source_feature_available_at_utc",
    "feature_available_at_utc",
    "primary_benchmark",
    "universe_snapshot_id",
    "strategy_contract_sha256",
    "feature_schema_version",
    "ordered_feature_sha256",
    "as_of_utc",
    "live_schema_version",
)
INTRADAY_BAR_LIVE_ABSTENTION_COLUMNS: Final = (
    *INTRADAY_BAR_LIVE_AUDIT_COLUMNS[:-2],
    "feature_ineligible_reason",
    "as_of_utc",
    "live_schema_version",
)


@dataclass(frozen=True, slots=True)
class IntradayBarLiveFeatureBatch:
    model_features: pd.DataFrame
    audit_identity: pd.DataFrame
    abstention_identity: pd.DataFrame


def build_live_intraday_bar_features(
    completed_volume_bars: pd.DataFrame,
    selected_five_minute_bars: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    point_in_time_memberships: pd.DataFrame,
    activation_rows: pd.DataFrame,
    *,
    contract: StrategyContract,
    as_of_utc: object,
) -> IntradayBarLiveFeatureBatch:
    """Build the exact scheduled cohort at ``as_of_utc`` with no stale fallback."""

    cutoff = _utc_cutoff(as_of_utc)
    _require_fixed_cohort(cutoff, contract)
    _reject_future_evidence(
        completed_volume_bars,
        label="completed volume bars",
        columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        selected_five_minute_bars,
        label="selected five-minute bars",
        columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        stock_one_minute_bars,
        label="stock one-minute bars",
        columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        benchmark_one_minute_bars,
        label="benchmark one-minute bars",
        columns=("bar_start_utc", "bar_end_utc", "available_at_utc"),
        cutoff=cutoff,
    )
    _reject_future_evidence(
        activation_rows,
        label="activation rows",
        columns=("activation_time_utc",),
        cutoff=cutoff,
    )
    built = build_causal_intraday_bar_features(
        completed_volume_bars,
        selected_five_minute_bars,
        stock_one_minute_bars,
        benchmark_one_minute_bars,
        point_in_time_memberships,
        activation_rows,
        contract=contract,
    )
    selected = built.loc[
        pd.to_datetime(built["decision_time_utc"], utc=True, errors="raise").eq(
            cutoff
        )
    ].copy()
    if selected.empty:
        raise DataReadinessError(
            "no scheduled intraday bar decision exists at as_of_utc"
        )
    selected = selected.sort_values("ticker", kind="stable").reset_index(drop=True)
    if not selected["ordered_feature_sha256"].astype(str).eq(
        INTRADAY_BAR_MODEL_FEATURES_SHA256
    ).all():
        raise DataReadinessError("live intraday bar feature hash differs")
    eligible_mask = selected["feature_eligible"].astype(bool)
    eligible = selected.loc[eligible_mask].copy()
    rejected = selected.loc[~eligible_mask].copy()
    if not rejected.empty and bool(
        rejected["feature_ineligible_reason"].isna().any()
        or rejected["feature_ineligible_reason"].astype(str).str.strip().eq("").any()
    ):
        raise DataReadinessError("ineligible live intraday bar rows lack a reason")
    if eligible.empty:
        details = ", ".join(
            f"{row.ticker}:{row.feature_ineligible_reason}"
            for row in rejected[
                ["ticker", "feature_ineligible_reason"]
            ].head(10).itertuples(index=False)
        )
        raise DataReadinessError(
            "scheduled live intraday bar cohort has no eligible rows: " + details
        )
    model_features = eligible.loc[:, INTRADAY_BAR_MODEL_FEATURE_COLUMNS].copy()
    if tuple(model_features.columns) != INTRADAY_BAR_MODEL_FEATURE_COLUMNS:
        raise DataReadinessError("live intraday bar feature order differs")
    if not np.isfinite(model_features.to_numpy(dtype="float64")).all():
        raise DataReadinessError("live intraday bar features must be finite")
    for column in INTRADAY_BAR_MODEL_FEATURE_COLUMNS:
        model_features[column] = model_features[column].astype("float32")
    audit = eligible.loc[:, INTRADAY_BAR_LIVE_AUDIT_COLUMNS[:-2]].copy()
    audit["as_of_utc"] = cutoff
    audit["live_schema_version"] = INTRADAY_BAR_LIVE_SCHEMA_VERSION
    audit = audit.loc[:, INTRADAY_BAR_LIVE_AUDIT_COLUMNS]
    abstentions = rejected.loc[
        :,
        INTRADAY_BAR_LIVE_ABSTENTION_COLUMNS[:-2],
    ].copy()
    abstentions["as_of_utc"] = cutoff
    abstentions["live_schema_version"] = INTRADAY_BAR_LIVE_SCHEMA_VERSION
    abstentions = abstentions.loc[:, INTRADAY_BAR_LIVE_ABSTENTION_COLUMNS]
    return IntradayBarLiveFeatureBatch(
        model_features=model_features.reset_index(drop=True),
        audit_identity=audit.reset_index(drop=True),
        abstention_identity=abstentions.reset_index(drop=True),
    )


def _utc_cutoff(value: object) -> pd.Timestamp:
    try:
        cutoff = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(
            "as_of_utc must be a valid timezone-aware timestamp"
        ) from exc
    if pd.isna(cutoff) or cutoff.tzinfo is None:
        raise DataReadinessError(
            "as_of_utc must be a valid timezone-aware timestamp"
        )
    return cutoff.tz_convert("UTC")


def _require_fixed_cohort(
    cutoff: pd.Timestamp,
    contract: StrategyContract,
) -> None:
    local = cutoff.tz_convert("America/New_York")
    minutes_after_open = local.hour * 60 + local.minute - (9 * 60 + 30)
    expected_remainder = contract.intraday.decision_finalization_seconds // 60
    if (
        cutoff.second != 0
        or cutoff.microsecond != 0
        or minutes_after_open <= expected_remainder
        or (minutes_after_open - expected_remainder) % 5 != 0
    ):
        raise DataReadinessError("as_of_utc is not a fixed five-minute cohort cutoff")


def _reject_future_evidence(
    frame: pd.DataFrame,
    *,
    label: str,
    columns: tuple[str, ...],
    cutoff: pd.Timestamp,
) -> None:
    missing = sorted(set(columns).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} omit cutoff columns: {missing}")
    for column in columns:
        values = pd.to_datetime(frame[column], utc=True, errors="raise")
        if bool(values.isna().any()) or bool(values.gt(cutoff).any()):
            raise DataReadinessError(
                f"{label} contain {column} evidence after as_of_utc"
            )
