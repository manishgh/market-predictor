"""Horizon-neutral contracts and helpers for causal outcome maturation."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import date, datetime
from typing import Protocol, cast

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError


class MaturationIntent(Protocol):
    @property
    def ticker(self) -> str: ...

    @property
    def decision_time_utc(self) -> datetime: ...

    @property
    def decision_session_et(self) -> date: ...

    @property
    def decision_atr(self) -> float | None: ...

    @property
    def label_policy(self) -> dict[str, object]: ...

    @property
    def primary_benchmark(self) -> str: ...


@dataclass(frozen=True)
class PendingPath:
    reasons: tuple[str, ...]
    missing_intervals: tuple[str, ...] = ()


@dataclass(frozen=True)
class MaturedPath:
    entry_time: datetime
    exit_time: datetime
    label_available: datetime
    entry_price: float
    exit_price: float
    gross_return: float
    label_round_trip_cost_bps: float
    label_net_return: float
    mfe: float
    mae: float
    path_outcome: str
    opportunity_target: int | None
    downside_target: int | None
    spy_return: float
    qqq_return: float
    sector_return: float
    evidence_rows: list[dict[str, object]]


PathEvaluation = PendingPath | MaturedPath


def daily_path(
    bars: pd.DataFrame,
    *,
    ticker: str,
    sessions: list[object],
) -> tuple[pd.DataFrame, list[str]]:
    ticker_rows = bars[bars["ticker"].eq(ticker)]
    counts = ticker_rows.groupby("session_date_et").size()
    missing = [
        f"{ticker}:{session}"
        for session in sessions
        if int(counts.get(session, 0)) != 1
    ]
    if missing:
        return pd.DataFrame(), missing
    rows = ticker_rows.set_index("session_date_et")
    return rows.loc[sessions].reset_index(), []


def intraday_path(
    bars: pd.DataFrame,
    *,
    ticker: str,
    expected_starts: list[datetime],
    session: object,
    bar_minutes: int,
) -> tuple[pd.DataFrame, list[str]]:
    ticker_rows = bars[bars["ticker"].eq(ticker)].set_index("bar_start_utc")
    missing = [
        f"{ticker}:{start.isoformat()}"
        for start in expected_starts
        if pd.Timestamp(start) not in ticker_rows.index
    ]
    if missing:
        return pd.DataFrame(), missing
    selected = ticker_rows.loc[pd.DatetimeIndex(expected_starts)].copy()
    selected.index.name = "bar_start_utc"
    selected = selected.reset_index()
    durations = selected["bar_end_utc"] - selected["bar_start_utc"]
    expected_duration = pd.Timedelta(minutes=bar_minutes)
    invalid_duration = durations.ne(expected_duration)
    if bool(invalid_duration.any()):
        missing.extend(
            f"{ticker}:{start.isoformat()}:wrong_bar_duration"
            for start in pd.to_datetime(
                selected.loc[invalid_duration, "bar_start_utc"],
                utc=True,
            )
        )
        return pd.DataFrame(), missing
    wrong_session = selected["session_date_et"].ne(session)
    if bool(wrong_session.any()):
        missing.extend(
            f"{ticker}:{start.isoformat()}:cross_session"
            for start in pd.to_datetime(
                selected.loc[wrong_session, "bar_start_utc"],
                utc=True,
            )
        )
        return pd.DataFrame(), missing
    return selected, []


def one_daily_row(
    bars: pd.DataFrame,
    *,
    ticker: str,
    session: object,
) -> pd.Series | None:
    rows = bars[bars["ticker"].eq(ticker) & bars["session_date_et"].eq(session)]
    return rows.iloc[0] if len(rows) == 1 else None


def one_intraday_row(
    bars: pd.DataFrame,
    *,
    ticker: str,
    start: datetime,
    bar_minutes: int,
) -> pd.Series | None:
    rows = bars[
        bars["ticker"].eq(ticker)
        & bars["bar_start_utc"].eq(pd.Timestamp(start))
    ]
    if len(rows) != 1:
        return None
    row = rows.iloc[0]
    if row["bar_end_utc"] - row["bar_start_utc"] != pd.Timedelta(
        minutes=bar_minutes
    ):
        return None
    return row


def pair_return(pair: tuple[pd.Series, pd.Series]) -> float:
    entry, exit_row = pair
    entry_price = float(entry["open"])
    exit_price = float(exit_row["close"])
    require_positive_prices(entry_price, exit_price)
    return exit_price / entry_price - 1.0


def evidence_rows(frames: list[pd.DataFrame]) -> list[dict[str, object]]:
    combined = pd.concat(frames, ignore_index=True).drop_duplicates(
        ["ticker", "bar_start_utc"]
    )
    combined = combined.sort_values(["ticker", "bar_start_utc"], kind="stable")
    columns = [
        "ticker",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "session_date_et",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "price_feed",
        "adjustment",
        "source_artifact_sha256",
    ]
    records: list[dict[str, object]] = []
    for record in combined[columns].to_dict(orient="records"):
        records.append(
            {
                key: (
                    value.isoformat()
                    if isinstance(value, (datetime, date, pd.Timestamp))
                    else value
                )
                for key, value in record.items()
            }
        )
    return records


def max_available(frames: list[pd.DataFrame]) -> datetime:
    return max(timestamp(frame["available_at_utc"].max()) for frame in frames)


def require_policy(policy: dict[str, object], field: str, expected: object) -> None:
    if policy.get(field) != expected:
        raise DataReadinessError(
            f"unsupported maturation label policy: {policy.get(field)!r}"
        )


def policy_int(policy: dict[str, object], field: str) -> int:
    value = policy.get(field)
    if not isinstance(value, int) or isinstance(value, bool):
        raise DataReadinessError(f"label policy field {field} must be an integer")
    return value


def policy_float(policy: dict[str, object], field: str) -> float:
    value = policy.get(field)
    if not isinstance(value, (int, float)) or isinstance(value, bool):
        raise DataReadinessError(f"label policy field {field} must be numeric")
    result = float(value)
    if not np.isfinite(result):
        raise DataReadinessError(f"label policy field {field} must be finite")
    return result


def require_positive_prices(*values: float) -> None:
    if any(not np.isfinite(value) or value <= 0 for value in values):
        raise DataReadinessError("maturation price evidence is invalid")


def timestamp(value: object) -> datetime:
    parsed = pd.Timestamp(value)
    if parsed.tzinfo is None:
        raise DataReadinessError("maturation timestamp is timezone-naive")
    return cast(datetime, parsed.tz_convert("UTC").to_pydatetime())
