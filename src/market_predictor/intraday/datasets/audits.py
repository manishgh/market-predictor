"""Atomic, lineage-bound publisher for the causal intraday training dataset."""
from __future__ import annotations



from collections.abc import Mapping
from datetime import date
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError


def _row_abstentions(frame: pd.DataFrame) -> list[dict[str, Any]]:
    rows = frame.loc[~frame["dataset_eligible"].astype(bool)]
    output: list[dict[str, Any]] = []
    for row in rows.itertuples(index=False):
        reason = str(row.dataset_ineligible_reason)
        output.append(
            {
                "dataset_row_id": row.dataset_row_id,
                "ticker": row.ticker,
                "session_date_et": row.session_date_et,
                "volume_bar_number": int(row.volume_bar_number),
                "feature_available_at_utc": row.feature_available_at_utc,
                "stage": reason.split(":", 1)[0],
                "reason": reason.split(":", 1)[1],
            }
        )
    return output

def _record_excluded_pairs(
    selection: pd.DataFrame,
    excluded: frozenset[str],
    *,
    pair_audits: list[dict[str, Any]],
    abstentions: list[dict[str, Any]],
    stage: str,
    reason: str,
) -> None:
    for row in selection[selection["ticker"].isin(excluded)].itertuples(index=False):
        pair_audits.append(
            _pair_audit(
                str(row.ticker),
                str(row.session_date_et),
                status="excluded",
                reason=reason,
            )
        )
        abstentions.append(
            _pair_abstention(
                str(row.ticker),
                str(row.session_date_et),
                stage,
                reason,
            )
        )

def _pair_abstention(ticker: str, session_date: str, stage: str, reason: str) -> dict[str, Any]:
    return {
        "dataset_row_id": pd.NA,
        "ticker": ticker,
        "session_date_et": session_date,
        "volume_bar_number": pd.NA,
        "feature_available_at_utc": pd.NaT,
        "stage": stage,
        "reason": reason,
    }

def _pair_audit(
    ticker: str,
    session_date: str,
    *,
    status: str,
    reason: str | None,
    source_rows: int = 0,
    completed_volume_bars: int = 0,
    feature_rows: int = 0,
    feature_eligible_rows: int = 0,
    label_eligible_rows: int = 0,
    dataset_eligible_rows: int = 0,
    abstention_rows: int = 1,
) -> dict[str, Any]:
    return {
        "ticker": ticker,
        "session_date_et": session_date,
        "status": status,
        "reason": reason,
        "source_rows": source_rows,
        "completed_volume_bars": completed_volume_bars,
        "feature_rows": feature_rows,
        "feature_eligible_rows": feature_eligible_rows,
        "label_eligible_rows": label_eligible_rows,
        "dataset_eligible_rows": dataset_eligible_rows,
        "abstention_rows": abstention_rows,
    }

def _activation_abstention_reason(
    activation: Mapping[str, object],
    membership: pd.DataFrame,
    *,
    maximum_delay_seconds: int,
) -> str | None:
    activated_at = pd.Timestamp(activation["activation_time_utc"])
    open_at = pd.Timestamp(membership.iloc[0]["session_open_utc"])
    close_at = pd.Timestamp(membership.iloc[0]["session_close_utc"])
    identity = (str(activation["session_date_et"]), str(activation["ticker"]))
    if activated_at < open_at:
        raise DataReadinessError(
            f"selection activation precedes the exchange open for {identity}"
        )
    if activated_at >= close_at:
        latest_valid_availability = close_at + pd.Timedelta(
            seconds=maximum_delay_seconds
        )
        if activated_at > latest_valid_availability:
            raise DataReadinessError(
                f"selection activation exceeds the frozen close delay for {identity}"
            )
        return "activation_not_executable_before_session_close"
    return None

def _monthly_stock_session_counts(frame: pd.DataFrame) -> dict[str, int]:
    if frame.empty:
        return {}
    months = pd.to_datetime(frame["session_date_et"], errors="raise").dt.strftime(
        "%Y-%m"
    )
    counts = frame.assign(_session_month_et=months).groupby(
        "_session_month_et", sort=True, observed=True
    ).size()
    return {str(month): int(count) for month, count in counts.items()}

def _expected_monthly_counts(raw: object, *, label: str) -> dict[str, int]:
    if not isinstance(raw, Mapping) or not raw:
        raise DataReadinessError(
            f"intraday dataset has no {label} monthly coverage contract"
        )
    output: dict[str, int] = {}
    for raw_month, raw_count in raw.items():
        month = str(raw_month)
        try:
            canonical = date.fromisoformat(f"{month}-01").strftime("%Y-%m")
            count = int(raw_count)
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(
                f"intraday dataset has invalid {label} monthly coverage"
            ) from exc
        if canonical != month or count < 1:
            raise DataReadinessError(
                f"intraday dataset has invalid {label} monthly coverage"
            )
        output[month] = count
    return dict(sorted(output.items()))