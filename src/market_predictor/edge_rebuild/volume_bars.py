"""Causal, session-scoped volume bars for the active intraday design."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.resources import assert_memory_budget, memory_audit
from market_predictor.v3.errors import DataReadinessError

EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
MEMORY_HARD_BUDGET_GIB = 4.0
MEMORY_HEADROOM_GIB = 0.75

_BAR_COLUMNS = frozenset(
    {
        "ticker",
        "timeframe",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "price_feed",
        "adjustment",
    }
)
_ACTIVATION_COLUMNS = frozenset(
    {
        "ticker",
        "session_date_et",
        "activation_time_utc",
        "median_volume_prior_sessions",
        "relative_volume_at_activation",
    }
)
VOLUME_BAR_COLUMNS = (
    "ticker",
    "session_date_et",
    "volume_bar_number",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "first_source_minute_utc",
    "last_source_minute_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source_row_count",
    "volume_threshold",
    "volume_overshoot",
    "relative_volume_at_activation",
    "activation_time_utc",
    "model_eligible",
    "source",
    "price_feed",
    "adjustment",
    "source_timeframe",
    "strategy_contract_sha256",
)
AUDIT_COLUMNS = (
    "ticker",
    "session_date_et",
    "activation_time_utc",
    "median_volume_prior_sessions",
    "relative_volume_at_activation",
    "volume_bars_per_session_target",
    "volume_threshold",
    "source_rows",
    "source_volume",
    "completed_volume_bars",
    "model_eligible_volume_bars",
    "incomplete_remainder_source_rows",
    "incomplete_remainder_volume",
    "incomplete_remainder_first_source_minute_utc",
    "incomplete_remainder_last_source_minute_utc",
    "strategy_contract_sha256",
)


@dataclass(frozen=True, slots=True)
class VolumeBarBuildResult:
    """Completed feature bars and discarded-remainder evidence."""

    bars: pd.DataFrame
    audit: pd.DataFrame
    memory: dict[str, float | None]


def build_causal_volume_bars(
    one_minute_bars: pd.DataFrame,
    activations: pd.DataFrame,
    *,
    contract: StrategyContract,
    strategy_contract_sha256: str,
) -> VolumeBarBuildResult:
    """Build fixed-threshold volume bars independently per activated session.

    Completed bars before activation are retained as same-session warm-up. A
    completed bar becomes model-eligible only when its source data is available
    at or after the causal activation timestamp.
    """

    _validate_contract(contract, strategy_contract_sha256)
    assert_memory_budget(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="volume-bar input validation",
    )
    source = _validate_one_minute_bars(one_minute_bars)
    selected = _validate_activations(
        activations,
        minimum_relative_volume=contract.intraday_universe.minimum_relative_volume,
    )
    source_group_indices = source.groupby(["ticker", "session_date_et"], sort=False, observed=True).indices
    output_rows: list[dict[str, Any]] = []
    audit_rows: list[dict[str, Any]] = []
    target = contract.intraday.volume_bars_per_session_target

    for activation in selected.itertuples(index=False):
        key = (str(activation.ticker), activation.session_date_et)
        positions = source_group_indices.get(key)
        if positions is None or len(positions) == 0:
            raise DataReadinessError(f"one-minute input has no rows for activated stock-session {key}")
        session = source.iloc[positions]
        ordered = session.sort_values("bar_start_utc", kind="stable")
        threshold = float(activation.median_volume_prior_sessions) / target
        completed, remainder = _build_one_session(
            ordered,
            threshold=threshold,
            activation_time=pd.Timestamp(activation.activation_time_utc),
            relative_volume_at_activation=float(
                activation.relative_volume_at_activation
            ),
            minimum_warmup_bars=contract.intraday.minimum_warmup_bars,
            strategy_contract_sha256=strategy_contract_sha256,
        )
        output_rows.extend(completed)
        eligible_count = sum(bool(row["model_eligible"]) for row in completed)
        audit_rows.append(
            {
                "ticker": key[0],
                "session_date_et": key[1],
                "activation_time_utc": activation.activation_time_utc,
                "median_volume_prior_sessions": float(activation.median_volume_prior_sessions),
                "relative_volume_at_activation": float(
                    activation.relative_volume_at_activation
                ),
                "volume_bars_per_session_target": target,
                "volume_threshold": threshold,
                "source_rows": int(len(ordered)),
                "source_volume": float(ordered["volume"].sum()),
                "completed_volume_bars": len(completed),
                "model_eligible_volume_bars": eligible_count,
                "incomplete_remainder_source_rows": len(remainder),
                "incomplete_remainder_volume": float(sum(float(row.volume) for row in remainder)),
                "incomplete_remainder_first_source_minute_utc": (remainder[0].bar_start_utc if remainder else pd.NaT),
                "incomplete_remainder_last_source_minute_utc": (remainder[-1].bar_start_utc if remainder else pd.NaT),
                "strategy_contract_sha256": strategy_contract_sha256,
            }
        )
        assert_memory_budget(
            hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
            stage=f"volume bars {key[0]} {key[1]}",
        )

    bars = pd.DataFrame(output_rows, columns=list(VOLUME_BAR_COLUMNS))
    audit = pd.DataFrame(audit_rows, columns=list(AUDIT_COLUMNS))
    if not bars.empty:
        bars = bars.sort_values(["session_date_et", "bar_end_utc", "ticker"], kind="stable").reset_index(drop=True)
    audit = audit.sort_values(["session_date_et", "ticker"], kind="stable").reset_index(drop=True)
    return VolumeBarBuildResult(
        bars=bars,
        audit=audit,
        memory=memory_audit(
            hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
        ).to_record(),
    )


def _build_one_session(
    rows: pd.DataFrame,
    *,
    threshold: float,
    activation_time: pd.Timestamp,
    relative_volume_at_activation: float,
    minimum_warmup_bars: int,
    strategy_contract_sha256: str,
) -> tuple[list[dict[str, Any]], list[Any]]:
    completed: list[dict[str, Any]] = []
    pending: list[Any] = []
    pending_volume = 0.0
    ticker = str(rows.iloc[0]["ticker"])
    session_date = rows.iloc[0]["session_date_et"]

    for row in rows.itertuples(index=False):
        pending.append(row)
        pending_volume += float(row.volume)
        if pending_volume < threshold:
            continue
        first = pending[0]
        last = pending[-1]
        available_at = max(pd.Timestamp(item.available_at_utc) for item in pending)
        completed.append(
            {
                "ticker": ticker,
                "session_date_et": session_date,
                "volume_bar_number": len(completed) + 1,
                "bar_start_utc": first.bar_start_utc,
                "bar_end_utc": last.bar_end_utc,
                "available_at_utc": available_at,
                "first_source_minute_utc": first.bar_start_utc,
                "last_source_minute_utc": last.bar_start_utc,
                "open": float(first.open),
                "high": max(float(item.high) for item in pending),
                "low": min(float(item.low) for item in pending),
                "close": float(last.close),
                "volume": pending_volume,
                "source_row_count": len(pending),
                "volume_threshold": threshold,
                "volume_overshoot": pending_volume - threshold,
                "relative_volume_at_activation": relative_volume_at_activation,
                "activation_time_utc": activation_time,
                "model_eligible": (
                    available_at >= activation_time
                    and len(completed) + 1 >= minimum_warmup_bars
                ),
                "source": "alpaca",
                "price_feed": "sip",
                "adjustment": "all",
                "source_timeframe": "1m",
                "strategy_contract_sha256": strategy_contract_sha256,
            }
        )
        pending = []
        pending_volume = 0.0
    return completed, pending


def _validate_contract(contract: StrategyContract, strategy_contract_sha256: str) -> None:
    actual = contract.sha256()
    if not strategy_contract_sha256 or strategy_contract_sha256 != actual:
        raise DataReadinessError("volume-bar strategy contract hash does not match")
    if contract.intraday.volume_bar_source_timeframe != "1Min":
        raise DataReadinessError("volume bars require a 1Min strategy source")
    if contract.intraday.volume_bars_per_session_target != 78:
        raise DataReadinessError("volume bars require the frozen target of 78 per session")


def _validate_one_minute_bars(bars: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(_BAR_COLUMNS.difference(bars.columns))
    if missing:
        raise DataReadinessError(f"one-minute bars are missing columns: {missing}")
    if bars.empty:
        raise DataReadinessError("one-minute bars are empty")
    frame = bars.loc[:, sorted(_BAR_COLUMNS)].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    identity_ok = (
        frame["timeframe"].astype(str).str.lower().str.strip().eq("1m").all()
        and frame["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and frame["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and frame["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
    )
    if not bool(identity_ok):
        raise DataReadinessError("volume bars require canonical Alpaca SIP/all 1m rows")
    numeric = frame.loc[:, ["open", "high", "low", "close", "volume"]]
    if (
        bool(frame["ticker"].eq("").any())
        or bool(numeric.isna().any().any())
        or not bool(np.isfinite(numeric.to_numpy(dtype="float64")).all())
        or bool(numeric.le(0).any().any())
        or bool(frame["high"].lt(frame[["open", "close"]].max(axis=1)).any())
        or bool(frame["low"].gt(frame[["open", "close"]].min(axis=1)).any())
        or bool(frame["high"].lt(frame["low"]).any())
    ):
        raise DataReadinessError("one-minute bars contain invalid OHLCV values")
    starts = frame["bar_start_utc"]
    if (
        bool(starts.dt.second.ne(0).any())
        or bool(starts.dt.microsecond.ne(0).any())
        or bool((frame["bar_end_utc"] - starts).ne(pd.Timedelta(minutes=1)).any())
        or bool(frame["available_at_utc"].dt.second.ne(0).any())
        or bool(frame["available_at_utc"].dt.microsecond.ne(0).any())
        or bool(frame["available_at_utc"].lt(frame["bar_end_utc"]).any())
        or bool(frame.duplicated(["ticker", "bar_start_utc"]).any())
    ):
        raise DataReadinessError("one-minute bars require unique exact-minute timestamps and causal availability")
    local = starts.dt.tz_convert(EXCHANGE_TIMEZONE)
    frame["session_date_et"] = local.dt.date
    return frame


def _validate_activations(
    activations: pd.DataFrame,
    *,
    minimum_relative_volume: float,
) -> pd.DataFrame:
    missing = sorted(_ACTIVATION_COLUMNS.difference(activations.columns))
    if missing:
        raise DataReadinessError(f"activation rows are missing columns: {missing}")
    if activations.empty:
        raise DataReadinessError("activation rows are empty")
    frame = activations.loc[:, sorted(_ACTIVATION_COLUMNS)].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["session_date_et"] = pd.to_datetime(frame["session_date_et"], errors="raise").dt.date
    frame["activation_time_utc"] = pd.to_datetime(frame["activation_time_utc"], utc=True, errors="raise")
    frame["median_volume_prior_sessions"] = pd.to_numeric(frame["median_volume_prior_sessions"], errors="coerce")
    frame["relative_volume_at_activation"] = pd.to_numeric(
        frame["relative_volume_at_activation"], errors="coerce"
    )
    medians = frame["median_volume_prior_sessions"].to_numpy(dtype="float64")
    relative_volume = frame["relative_volume_at_activation"].to_numpy(
        dtype="float64"
    )
    activation_local_date = frame["activation_time_utc"].dt.tz_convert(EXCHANGE_TIMEZONE).dt.date
    if (
        bool(frame["ticker"].eq("").any())
        or bool(frame.duplicated(["ticker", "session_date_et"]).any())
        or not bool(np.isfinite(medians).all())
        or bool((medians <= 0).any())
        or not bool(np.isfinite(relative_volume).all())
        or bool((relative_volume < minimum_relative_volume).any())
        or bool(activation_local_date.ne(frame["session_date_et"]).any())
        or bool(frame["activation_time_utc"].dt.second.ne(0).any())
        or bool(frame["activation_time_utc"].dt.microsecond.ne(0).any())
    ):
        raise DataReadinessError("activation rows contain invalid identity, time, or median volume")
    return frame.sort_values(["session_date_et", "activation_time_utc", "ticker"], kind="stable").reset_index(drop=True)
