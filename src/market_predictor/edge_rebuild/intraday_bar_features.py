"""Causal hybrid intraday features on fixed five-minute decision cohorts."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable
from datetime import date
from typing import Final
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.v3.errors import DataReadinessError

INTRADAY_BAR_FEATURE_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_features.v1"
EXCHANGE_TIMEZONE: Final = ZoneInfo("America/New_York")
_HORIZONS: Final = (1, 5, 20)

# MACD is intentionally absent: its 26-bar base period conflicts with the frozen
# 20-volume-bar warm-up. ATR is a five-minute clock-bar feature, never volume-bar state.
INTRADAY_BAR_MODEL_FEATURE_COLUMNS: Final = (
    "volume_return_1_bar",
    "volume_return_3_bars",
    "volume_return_5_bars",
    "volume_rsi_14",
    "volume_ema_10_distance",
    "volume_ema_20_distance",
    "volume_sma_10_distance",
    "volume_sma_20_distance",
    "volume_realized_volatility_5",
    "volume_realized_volatility_20",
    "volume_granville_obv_confirmation",
    "volume_kaufman_efficiency_ratio",
    "five_minute_atr_14_fraction_of_close",
    "session_vwap_distance_five_minute_atr",
    "opening_range_high_distance_five_minute_atr",
    "opening_range_low_distance_five_minute_atr",
    "volume_bar_progress",
    "normalized_volume_overshoot",
    "volume_bar_duration_minutes",
    "relative_volume_at_activation",
    "cumulative_volume_fraction_of_prior_session_median",
    "minutes_since_activation",
    "regular_session_progress",
    "stock_return_1m",
    "stock_return_5m",
    "stock_return_20m",
    "spy_return_1m",
    "spy_return_5m",
    "spy_return_20m",
    "qqq_return_1m",
    "qqq_return_5m",
    "qqq_return_20m",
    "sector_return_1m",
    "sector_return_5m",
    "sector_return_20m",
    "spy_residual_return_1m",
    "spy_residual_return_5m",
    "spy_residual_return_20m",
    "qqq_residual_return_1m",
    "qqq_residual_return_5m",
    "qqq_residual_return_20m",
    "sector_residual_return_1m",
    "sector_residual_return_5m",
    "sector_residual_return_20m",
)
INTRADAY_BAR_MODEL_FEATURES_JSON: Final = json.dumps(
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    separators=(",", ":"),
)
INTRADAY_BAR_MODEL_FEATURES_SHA256: Final = hashlib.sha256(
    INTRADAY_BAR_MODEL_FEATURES_JSON.encode("utf-8")
).hexdigest()

_CLOCK_BAR_COLUMNS: Final = frozenset(
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
_VOLUME_BAR_COLUMNS: Final = frozenset(
    {
        "ticker",
        "session_date_et",
        "volume_bar_number",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_threshold",
        "volume_overshoot",
        "relative_volume_at_activation",
        "activation_time_utc",
        "source",
        "price_feed",
        "adjustment",
        "source_timeframe",
        "strategy_contract_sha256",
    }
)
_ACTIVATION_COLUMNS: Final = frozenset(
    {
        "ticker",
        "session_date_et",
        "activation_time_utc",
        "median_volume_prior_sessions",
        "relative_volume_at_activation",
    }
)
_MEMBERSHIP_COLUMNS: Final = frozenset(
    {
        "ticker",
        "security_id",
        "sector",
        "primary_benchmark",
        "universe_snapshot_id",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
    }
)


def build_causal_intraday_bar_features(
    completed_volume_bars: pd.DataFrame,
    selected_five_minute_bars: pd.DataFrame,
    stock_one_minute_bars: pd.DataFrame,
    benchmark_one_minute_bars: pd.DataFrame,
    point_in_time_memberships: pd.DataFrame,
    activation_rows: pd.DataFrame,
    *,
    contract: StrategyContract,
) -> pd.DataFrame:
    """Build fixed five-minute cohorts with causal volume-bar state as of each cutoff."""

    _validate_contract(contract)
    contract_sha256 = contract.sha256()
    volume = _validate_volume_bars(completed_volume_bars, contract_sha256)
    five = _validate_clock_bars(selected_five_minute_bars, timeframe="5m", label="selected five-minute")
    stocks = _validate_clock_bars(stock_one_minute_bars, timeframe="1m", label="stock one-minute")
    benchmarks = _validate_clock_bars(benchmark_one_minute_bars, timeframe="1m", label="benchmark one-minute")
    memberships = _validate_memberships(point_in_time_memberships)
    activations = _validate_activations(activation_rows)

    five_groups = {
        key: group.reset_index(drop=True)
        for key, group in five.groupby(["ticker", "session_date_et"], sort=False, observed=True)
    }
    volume_states = {
        key: _volume_states(group, contract)
        for key, group in volume.groupby(["ticker", "session_date_et"], sort=False, observed=True)
    }
    stock_context = _minute_context(stocks)
    benchmark_context = _minute_context(benchmarks)
    stock_contexts = {
        key: group.reset_index(drop=True)
        for key, group in stock_context.groupby(["ticker", "session_date_et"], sort=False, observed=True)
    }
    benchmark_contexts = {
        key: group.reset_index(drop=True)
        for key, group in benchmark_context.groupby(["ticker", "session_date_et"], sort=False, observed=True)
    }

    outputs: list[pd.DataFrame] = []
    calendar = xcals.get_calendar("XNYS")
    for activation in activations.itertuples(index=False):
        ticker = str(activation.ticker)
        session_date = activation.session_date_et
        key = (ticker, session_date)
        membership, session_open, session_close = _membership_for_session(
            memberships,
            ticker=ticker,
            session_date=session_date,
            calendar=calendar,
        )
        fixed = _five_minute_states(
            five_groups.get(key),
            ticker=ticker,
            session_date=session_date,
            session_open=session_open,
            session_close=session_close,
            contract=contract,
        )
        fixed = fixed.loc[
            fixed["decision_time_utc"].ge(pd.Timestamp(activation.activation_time_utc))
        ].copy()
        if fixed.empty:
            continue
        fixed["activation_time_utc"] = pd.Timestamp(activation.activation_time_utc)
        fixed["median_volume_prior_sessions"] = float(activation.median_volume_prior_sessions)
        fixed["relative_volume_at_activation"] = float(activation.relative_volume_at_activation)
        fixed = _attach_latest_volume_state(fixed, volume_states.get(key))
        target_minutes = fixed["five_minute_bar_end_utc"] - pd.Timedelta(minutes=1)
        fixed["context_minute_utc"] = target_minutes
        fixed = _attach_context(
            fixed,
            stock_contexts.get(key),
            prefix="stock",
        )
        fixed = _attach_context(
            fixed,
            benchmark_contexts.get(("SPY", session_date)),
            prefix="spy",
        )
        fixed = _attach_context(
            fixed,
            benchmark_contexts.get(("QQQ", session_date)),
            prefix="qqq",
        )
        fixed = _attach_context(
            fixed,
            benchmark_contexts.get((str(membership.primary_benchmark), session_date)),
            prefix="sector",
        )
        outputs.append(
            _finalize_rows(
                fixed,
                membership=membership,
                session_open=session_open,
                session_close=session_close,
                contract=contract,
                contract_sha256=contract_sha256,
            )
        )

    if not outputs:
        raise DataReadinessError("hybrid intraday feature build produced no fixed cohorts")
    result = pd.concat(outputs, ignore_index=True)
    if bool(result.duplicated(["ticker", "decision_time_utc"]).any()):
        raise DataReadinessError("hybrid intraday features repeat a ticker within a fixed cohort")
    return result.sort_values(["decision_time_utc", "ticker"], kind="stable").reset_index(drop=True)


def _five_minute_states(
    frame: pd.DataFrame | None,
    *,
    ticker: str,
    session_date: date,
    session_open: pd.Timestamp,
    session_close: pd.Timestamp,
    contract: StrategyContract,
) -> pd.DataFrame:
    starts = pd.date_range(
        session_open,
        session_close - pd.Timedelta(minutes=5),
        freq="5min",
    )
    grid = pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": session_date,
            "five_minute_bar_number": np.arange(1, len(starts) + 1, dtype="int64"),
            "five_minute_bar_start_utc": starts,
            "five_minute_bar_end_utc": starts + pd.Timedelta(minutes=5),
        }
    )
    grid["decision_time_utc"] = grid["five_minute_bar_end_utc"] + pd.Timedelta(
        seconds=contract.intraday.decision_finalization_seconds
    )
    grid = grid.loc[grid["decision_time_utc"].lt(session_close)].reset_index(drop=True)
    if frame is None or frame.empty:
        observed = pd.DataFrame(
            columns=[
                "five_minute_bar_start_utc",
                "five_minute_source_available_at_utc",
                "high",
                "low",
                "five_minute_close",
            ]
        )
    else:
        ordered = frame.sort_values("bar_start_utc", kind="stable").reset_index(drop=True)
        observed = ordered.rename(
            columns={
                "bar_start_utc": "five_minute_bar_start_utc",
                "available_at_utc": "five_minute_source_available_at_utc",
                "close": "five_minute_close",
            }
        ).loc[
            :,
            [
                "five_minute_bar_start_utc",
                "five_minute_source_available_at_utc",
                "high",
                "low",
                "five_minute_close",
            ],
        ]
    joined = grid.merge(
        observed,
        on="five_minute_bar_start_utc",
        how="left",
        validate="one_to_one",
    )
    joined["five_minute_source_available_at_utc"] = pd.to_datetime(
        joined["five_minute_source_available_at_utc"],
        utc=True,
        errors="coerce",
    ).cummax()
    joined["five_minute_bar_observed"] = joined["five_minute_close"].notna()
    joined["five_minute_prefix_complete"] = joined["five_minute_bar_observed"].cummin()
    high = pd.to_numeric(joined["high"], errors="coerce").to_numpy(dtype="float64")
    low = pd.to_numeric(joined["low"], errors="coerce").to_numpy(dtype="float64")
    close = pd.to_numeric(joined["five_minute_close"], errors="coerce").to_numpy(dtype="float64")
    previous = np.concatenate(([np.nan], close[:-1]))
    true_range = np.maximum.reduce([high - low, np.abs(high - previous), np.abs(low - previous)])
    if len(true_range):
        true_range[0] = high[0] - low[0]
    joined["atr_14_5m"] = pd.Series(true_range).ewm(
        alpha=1.0 / contract.intraday.atr_lookback_bars,
        adjust=False,
        min_periods=contract.intraday.atr_lookback_bars,
    ).mean().to_numpy(dtype="float64")
    joined.loc[~joined["five_minute_prefix_complete"], "atr_14_5m"] = np.nan
    return joined.loc[
        :,
        [
            "ticker",
            "session_date_et",
            "five_minute_bar_number",
            "five_minute_bar_start_utc",
            "five_minute_bar_end_utc",
            "decision_time_utc",
            "five_minute_source_available_at_utc",
            "five_minute_bar_observed",
            "five_minute_prefix_complete",
            "atr_14_5m",
            "five_minute_close",
        ],
    ]


def _volume_states(frame: pd.DataFrame, contract: StrategyContract) -> pd.DataFrame:
    ordered = frame.sort_values("volume_bar_number", kind="stable").reset_index(drop=True)
    close = pd.Series(ordered["close"].to_numpy(dtype="float64"))
    volume = ordered["volume"].to_numpy(dtype="float64")
    for horizon in (1, 3, 5):
        suffix = "bar" if horizon == 1 else "bars"
        ordered[f"volume_return_{horizon}_{suffix}"] = close.pct_change(
            periods=horizon,
            fill_method=None,
        ).to_numpy(dtype="float64")
    movement = close.diff()
    gain = movement.clip(lower=0.0).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    loss = (-movement.clip(upper=0.0)).ewm(alpha=1.0 / 14.0, adjust=False, min_periods=14).mean()
    relative_strength = gain / loss.replace(0.0, np.nan)
    rsi = 100.0 - 100.0 / (1.0 + relative_strength)
    rsi = rsi.mask((loss == 0.0) & (gain > 0.0), 100.0).mask((loss == 0.0) & (gain == 0.0), 50.0)
    ordered["volume_rsi_14"] = rsi
    ema_10 = close.ewm(span=10, adjust=False, min_periods=10).mean()
    ema_20 = close.ewm(span=20, adjust=False, min_periods=20).mean()
    ordered["volume_ema_10_distance"] = close / ema_10 - 1.0
    ordered["volume_ema_20_distance"] = close / ema_20 - 1.0
    ordered["volume_sma_10_distance"] = close / close.rolling(10, min_periods=10).mean() - 1.0
    ordered["volume_sma_20_distance"] = close / close.rolling(20, min_periods=20).mean() - 1.0
    log_return = np.log(close / close.shift(1))
    log_return.iloc[0] = np.log(float(ordered.iloc[0]["close"]) / float(ordered.iloc[0]["open"]))
    for horizon in (5, 20):
        ordered[f"volume_realized_volatility_{horizon}"] = np.sqrt(
            log_return.pow(2.0).rolling(horizon, min_periods=horizon).sum()
        )
    obv, efficiency = _volume_relationships(close.to_numpy(dtype="float64"), volume, lookback=20)
    ordered["volume_granville_obv_confirmation"] = obv
    ordered["volume_kaufman_efficiency_ratio"] = efficiency
    ordered["volume_bar_progress"] = ordered["volume_bar_number"] / float(
        contract.intraday.volume_bars_per_session_target
    )
    ordered["volume_state_cumulative_volume"] = ordered["volume"].cumsum()
    ordered["normalized_volume_overshoot"] = ordered["volume_overshoot"] / ordered["volume_threshold"]
    ordered["volume_bar_duration_minutes"] = (
        ordered["bar_end_utc"] - ordered["bar_start_utc"]
    ) / pd.Timedelta(minutes=1)
    ordered["volume_state_close"] = ordered["close"]
    ordered["volume_state_available_at_utc"] = ordered["available_at_utc"].cummax()
    columns = [
        "volume_bar_number",
        "volume_state_close",
        "volume_state_available_at_utc",
        "volume_state_cumulative_volume",
        *INTRADAY_BAR_MODEL_FEATURE_COLUMNS[:12],
        "volume_bar_progress",
        "normalized_volume_overshoot",
        "volume_bar_duration_minutes",
    ]
    return ordered.loc[:, list(dict.fromkeys(columns))]


def _minute_context(frame: pd.DataFrame) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for _, group in frame.groupby(["ticker", "session_date_et"], sort=False, observed=True):
        ordered = group.sort_values("bar_start_utc", kind="stable").copy()
        opens = ordered["open"].to_numpy(dtype="float64")
        closes = ordered["close"].to_numpy(dtype="float64")
        typical = (
            ordered["high"].to_numpy(dtype="float64")
            + ordered["low"].to_numpy(dtype="float64")
            + closes
        ) / 3.0
        volumes = ordered["volume"].to_numpy(dtype="float64")
        cumulative_volume = np.cumsum(volumes)
        ordered["session_vwap"] = np.cumsum(typical * volumes) / cumulative_volume
        starts = pd.DatetimeIndex(ordered["bar_start_utc"])
        open_by_minute = pd.Series(opens, index=starts)
        for horizon in _HORIZONS:
            required_start = starts - pd.Timedelta(minutes=horizon - 1)
            start_open = open_by_minute.reindex(required_start).to_numpy(dtype="float64")
            ordered[f"return_{horizon}m"] = closes / start_open - 1.0
        local = ordered["bar_start_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
        expected_prefix_rows = local.dt.hour * 60 + local.dt.minute - (9 * 60 + 30) + 1
        ordered["context_complete"] = np.arange(1, len(ordered) + 1) == expected_prefix_rows.to_numpy()
        opening = (local.dt.hour * 60 + local.dt.minute).between(570, 584)
        opening_high = float(ordered.loc[opening, "high"].max()) if int(opening.sum()) == 15 else np.nan
        opening_low = float(ordered.loc[opening, "low"].min()) if int(opening.sum()) == 15 else np.nan
        opening_known = (local.dt.hour * 60 + local.dt.minute).ge(584)
        ordered["opening_range_high"] = np.where(opening_known, opening_high, np.nan)
        ordered["opening_range_low"] = np.where(opening_known, opening_low, np.nan)
        ordered["context_source_available_at_utc"] = ordered["available_at_utc"].cummax()
        ordered["context_minute_utc"] = ordered["bar_start_utc"]
        parts.append(
            ordered.loc[
                :,
                [
                    "ticker",
                    "session_date_et",
                    "context_minute_utc",
                    "context_source_available_at_utc",
                    "context_complete",
                    "close",
                    "session_vwap",
                    "opening_range_high",
                    "opening_range_low",
                    *[f"return_{horizon}m" for horizon in _HORIZONS],
                ],
            ]
        )
    if not parts:
        raise DataReadinessError("one-minute context cannot be empty")
    return pd.concat(parts, ignore_index=True)


def _attach_latest_volume_state(fixed: pd.DataFrame, state: pd.DataFrame | None) -> pd.DataFrame:
    if state is None or state.empty:
        output = fixed.copy()
        for column in _volume_state_output_columns():
            output[column] = pd.NA
        return output
    left = fixed.copy()
    right = state.copy()
    left["decision_time_utc"] = pd.DatetimeIndex(left["decision_time_utc"]).as_unit("ns")
    right["volume_state_available_at_utc"] = pd.DatetimeIndex(
        right["volume_state_available_at_utc"]
    ).as_unit("ns")
    return pd.merge_asof(
        left.sort_values("decision_time_utc", kind="stable"),
        right.sort_values("volume_state_available_at_utc", kind="stable"),
        left_on="decision_time_utc",
        right_on="volume_state_available_at_utc",
        direction="backward",
        allow_exact_matches=True,
    )


def _attach_context(fixed: pd.DataFrame, context: pd.DataFrame | None, *, prefix: str) -> pd.DataFrame:
    names = {
        "context_source_available_at_utc": f"{prefix}_source_available_at_utc",
        "context_complete": f"{prefix}_context_complete",
        "close": f"{prefix}_context_close",
        "session_vwap": f"{prefix}_session_vwap",
        "opening_range_high": f"{prefix}_opening_range_high",
        "opening_range_low": f"{prefix}_opening_range_low",
        **{f"return_{horizon}m": f"{prefix}_return_{horizon}m" for horizon in _HORIZONS},
    }
    if context is None or context.empty:
        output = fixed.copy()
        for column in names.values():
            output[column] = pd.NA
        return output
    selected = context.loc[:, ["context_minute_utc", *names]].rename(columns=names)
    return fixed.merge(selected, on="context_minute_utc", how="left", validate="many_to_one")


def _finalize_rows(
    frame: pd.DataFrame,
    *,
    membership: pd.Series,
    session_open: pd.Timestamp,
    session_close: pd.Timestamp,
    contract: StrategyContract,
    contract_sha256: str,
) -> pd.DataFrame:
    output = frame.copy()
    output["ticker"] = str(membership["ticker"])
    output["security_id"] = str(membership["security_id"])
    output["sector"] = str(membership["sector"])
    output["primary_benchmark"] = str(membership["primary_benchmark"])
    output["universe_snapshot_id"] = str(membership["universe_snapshot_id"])
    output["membership_available_at_utc"] = pd.Timestamp(membership["available_at_utc"])
    output["session_open_utc"] = session_open
    output["session_close_utc"] = session_close

    atr = pd.to_numeric(output["atr_14_5m"], errors="coerce")
    stock_close = pd.to_numeric(output["stock_context_close"], errors="coerce")
    output["five_minute_atr_14_fraction_of_close"] = atr / stock_close
    output["session_vwap_distance_five_minute_atr"] = (
        pd.to_numeric(output["volume_state_close"], errors="coerce")
        - pd.to_numeric(output["stock_session_vwap"], errors="coerce")
    ) / atr
    output["opening_range_high_distance_five_minute_atr"] = (
        stock_close - pd.to_numeric(output["stock_opening_range_high"], errors="coerce")
    ) / atr
    output["opening_range_low_distance_five_minute_atr"] = (
        stock_close - pd.to_numeric(output["stock_opening_range_low"], errors="coerce")
    ) / atr
    output["cumulative_volume_fraction_of_prior_session_median"] = (
        pd.to_numeric(output["volume_state_cumulative_volume"], errors="coerce")
        / pd.to_numeric(output["median_volume_prior_sessions"], errors="coerce")
    )
    # The frozen activation carries the causal relative-volume state. The separate
    # progress feature describes the latest available volume-bar state at the cutoff.
    output["minutes_since_activation"] = (
        output["decision_time_utc"] - output["activation_time_utc"]
    ) / pd.Timedelta(minutes=1)
    output["regular_session_progress"] = (
        (output["decision_time_utc"] - session_open)
        / (session_close - session_open)
    ).clip(lower=0.0, upper=1.0)
    for prefix in ("spy", "qqq", "sector"):
        for horizon in _HORIZONS:
            output[f"{prefix}_residual_return_{horizon}m"] = (
                pd.to_numeric(output[f"stock_return_{horizon}m"], errors="coerce")
                - pd.to_numeric(output[f"{prefix}_return_{horizon}m"], errors="coerce")
            )

    source_columns = [
        "five_minute_source_available_at_utc",
        "volume_state_available_at_utc",
        "stock_source_available_at_utc",
        "spy_source_available_at_utc",
        "qqq_source_available_at_utc",
        "sector_source_available_at_utc",
        "membership_available_at_utc",
        "activation_time_utc",
    ]
    for column in source_columns:
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    output["source_feature_available_at_utc"] = output[source_columns].max(axis=1)
    output["feature_available_at_utc"] = output["decision_time_utc"]
    output["feature_schema_version"] = INTRADAY_BAR_FEATURE_SCHEMA_VERSION
    output["ordered_feature_names_json"] = INTRADAY_BAR_MODEL_FEATURES_JSON
    output["ordered_feature_sha256"] = INTRADAY_BAR_MODEL_FEATURES_SHA256
    output["strategy_contract_sha256"] = contract_sha256
    output["decision_cohort_id"] = output["decision_time_utc"].map(
        lambda value: _identity_hash(str(output["session_date_et"].iloc[0]), pd.Timestamp(value).isoformat(), contract_sha256)
    )
    output["decision_id"] = [
        _identity_hash(str(security), pd.Timestamp(decision).isoformat(), contract_sha256)
        for security, decision in zip(output["security_id"], output["decision_time_utc"], strict=True)
    ]

    reasons = pd.Series(pd.NA, index=output.index, dtype="string")
    _set_reason(
        reasons,
        ~output["five_minute_bar_observed"].fillna(False).astype(bool),
        "missing_exact_five_minute_bar",
    )
    _set_reason(
        reasons,
        ~output["five_minute_prefix_complete"].fillna(False).astype(bool),
        "incomplete_five_minute_session_prefix",
    )
    _set_reason(reasons, output["five_minute_source_available_at_utc"].gt(output["decision_time_utc"]), "five_minute_evidence_late")
    _set_reason(reasons, output["volume_state_available_at_utc"].isna(), "no_completed_volume_bar_available_at_cutoff")
    _set_reason(
        reasons,
        pd.to_numeric(output["volume_bar_number"], errors="coerce").lt(contract.intraday.minimum_warmup_bars),
        "insufficient_completed_volume_bars",
    )
    for prefix in ("stock", "spy", "qqq", "sector"):
        missing = output[f"{prefix}_source_available_at_utc"].isna()
        late = output[f"{prefix}_source_available_at_utc"].gt(output["decision_time_utc"])
        _set_reason(reasons, missing, f"missing_exact_{prefix}_one_minute_context")
        _set_reason(reasons, late, f"{prefix}_one_minute_context_late")
        _set_reason(
            reasons,
            ~output[f"{prefix}_context_complete"].fillna(False).astype(bool),
            f"incomplete_{prefix}_one_minute_context",
        )
    _set_reason(
        reasons,
        output["membership_available_at_utc"].gt(session_open),
        "membership_not_available_at_session_open",
    )
    numeric = output.loc[:, INTRADAY_BAR_MODEL_FEATURE_COLUMNS].apply(pd.to_numeric, errors="coerce")
    finite = np.isfinite(numeric.to_numpy(dtype="float64")).all(axis=1)
    _set_reason(reasons, pd.Series(~finite, index=output.index), "incomplete_required_feature_history")
    output["feature_eligible"] = reasons.isna()
    output["feature_ineligible_reason"] = reasons
    for column in INTRADAY_BAR_MODEL_FEATURE_COLUMNS:
        output[column] = pd.to_numeric(output[column], errors="coerce").astype("float32")
    return output


def _validate_contract(contract: StrategyContract) -> None:
    intraday = contract.intraday
    if (
        intraday.decision_finalization_seconds != 60
        or intraday.atr_timeframe != "5Min"
        or intraday.atr_lookback_bars != 14
        or intraday.minimum_warmup_bars != 20
        or contract.intraday_universe.activation_delay_seconds != 60
        or contract.labels.benchmark_market != "SPY"
    ):
        raise DataReadinessError("hybrid intraday features require the frozen causal timing and ATR contract")


def _validate_clock_bars(frame: pd.DataFrame, *, timeframe: str, label: str) -> pd.DataFrame:
    _require_columns(frame, _CLOCK_BAR_COLUMNS, label)
    if frame.empty:
        raise DataReadinessError(f"{label} bars are empty")
    output = frame.loc[:, sorted(_CLOCK_BAR_COLUMNS)].copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
    for column in ("open", "high", "low", "close", "volume"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    duration = pd.Timedelta(minutes=5 if timeframe == "5m" else 1)
    numeric = output[["open", "high", "low", "close", "volume"]]
    identity = (
        output["timeframe"].astype(str).str.lower().str.strip().eq(timeframe).all()
        and output["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and output["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and output["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
    )
    if (
        not bool(identity)
        or bool(output["ticker"].eq("").any())
        or not bool(np.isfinite(numeric.to_numpy(dtype="float64")).all())
        or bool(numeric[["open", "high", "low", "close"]].le(0.0).any().any())
        or bool(numeric["volume"].lt(0.0).any())
        or bool(output["high"].lt(output[["open", "close"]].max(axis=1)).any())
        or bool(output["low"].gt(output[["open", "close"]].min(axis=1)).any())
        or bool((output["bar_end_utc"] - output["bar_start_utc"]).ne(duration).any())
        or bool(output.duplicated(["ticker", "bar_start_utc"]).any())
    ):
        raise DataReadinessError(f"{label} bars violate canonical Alpaca SIP/all {timeframe} identity")
    local = output["bar_start_utc"].dt.tz_convert(EXCHANGE_TIMEZONE)
    minutes = local.dt.hour * 60 + local.dt.minute
    expected_modulus = 5 if timeframe == "5m" else 1
    aligned = (
        local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
        & minutes.sub(9 * 60 + 30).mod(expected_modulus).eq(0)
    )
    if not bool(aligned.all()):
        raise DataReadinessError(f"{label} bars are not aligned to the regular-session clock")
    output["session_date_et"] = local.dt.date
    return output.sort_values(["ticker", "bar_start_utc"], kind="stable").reset_index(drop=True)


def _validate_volume_bars(frame: pd.DataFrame, contract_sha256: str) -> pd.DataFrame:
    _require_columns(frame, _VOLUME_BAR_COLUMNS, "completed volume")
    if frame.empty:
        raise DataReadinessError("completed volume bars are empty")
    output = frame.loc[:, sorted(_VOLUME_BAR_COLUMNS)].copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["session_date_et"] = pd.to_datetime(output["session_date_et"], errors="raise").dt.date
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc", "activation_time_utc"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="raise")
    numeric_columns = (
        "open",
        "high",
        "low",
        "close",
        "volume",
        "volume_threshold",
        "volume_overshoot",
        "relative_volume_at_activation",
        "volume_bar_number",
    )
    for column in numeric_columns:
        output[column] = pd.to_numeric(output[column], errors="coerce")
    identity = (
        output["source"].astype(str).str.lower().str.strip().eq("alpaca").all()
        and output["price_feed"].astype(str).str.lower().str.strip().eq("sip").all()
        and output["adjustment"].astype(str).str.lower().str.strip().eq("all").all()
        and output["source_timeframe"].astype(str).str.lower().str.strip().eq("1m").all()
        and output["strategy_contract_sha256"].astype(str).eq(contract_sha256).all()
    )
    numeric = output[list(numeric_columns)]
    if (
        not bool(identity)
        or not bool(np.isfinite(numeric.to_numpy(dtype="float64")).all())
        or bool(
            numeric[
                ["open", "high", "low", "close", "volume", "volume_threshold", "volume_bar_number"]
            ]
            .le(0.0)
            .any()
            .any()
        )
        or bool(output["volume_overshoot"].lt(0.0).any())
        or bool(output.duplicated(["ticker", "session_date_et", "volume_bar_number"]).any())
    ):
        raise DataReadinessError("completed volume bars violate causal archive identity")
    for _, group in output.groupby(["ticker", "session_date_et"], sort=False, observed=True):
        ordered = group.sort_values("volume_bar_number", kind="stable")
        if bool(ordered["available_at_utc"].diff().dropna().lt(pd.Timedelta(0)).any()):
            raise DataReadinessError("completed volume-bar availability must be nondecreasing")
    return output


def _validate_activations(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _ACTIVATION_COLUMNS, "activation")
    if frame.empty:
        raise DataReadinessError("activation rows are empty")
    output = frame.loc[:, sorted(_ACTIVATION_COLUMNS)].copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["session_date_et"] = pd.to_datetime(output["session_date_et"], errors="raise").dt.date
    output["activation_time_utc"] = pd.to_datetime(output["activation_time_utc"], utc=True, errors="raise")
    for column in ("median_volume_prior_sessions", "relative_volume_at_activation"):
        output[column] = pd.to_numeric(output[column], errors="coerce")
    if (
        bool(output.duplicated(["ticker", "session_date_et"]).any())
        or bool(output["ticker"].eq("").any())
        or bool(output[["median_volume_prior_sessions", "relative_volume_at_activation"]].le(0.0).any().any())
    ):
        raise DataReadinessError("activation rows contain invalid causal identity")
    return output.sort_values(["session_date_et", "activation_time_utc", "ticker"], kind="stable").reset_index(drop=True)


def _validate_memberships(frame: pd.DataFrame) -> pd.DataFrame:
    _require_columns(frame, _MEMBERSHIP_COLUMNS, "point-in-time membership")
    if frame.empty:
        raise DataReadinessError("point-in-time memberships are empty")
    output = frame.copy()
    output["ticker"] = output["ticker"].astype(str).str.upper().str.strip()
    output["primary_benchmark"] = output["primary_benchmark"].astype(str).str.upper().str.strip()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        output[column] = pd.to_datetime(output[column], utc=True, errors="coerce")
    if bool(
        output[["effective_from_utc", "available_at_utc"]]
        .isna()
        .any()
        .any()
    ):
        raise DataReadinessError(
            "point-in-time membership causal timestamps are incomplete"
        )
    if bool(output[["ticker", "security_id", "sector", "primary_benchmark", "universe_snapshot_id"]].isna().any().any()):
        raise DataReadinessError("point-in-time membership identity is incomplete")
    return output


def _membership_for_session(
    frame: pd.DataFrame,
    *,
    ticker: str,
    session_date: date,
    calendar: object,
) -> tuple[pd.Series, pd.Timestamp, pd.Timestamp]:
    session = pd.Timestamp(session_date)
    session_open = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")  # type: ignore[attr-defined]
    session_close = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")  # type: ignore[attr-defined]
    candidates = frame.loc[frame["ticker"].eq(ticker)].copy()
    active = candidates["effective_from_utc"].le(session_open) & (
        candidates["effective_to_utc"].isna() | candidates["effective_to_utc"].gt(session_open)
    )
    candidates = candidates.loc[active]
    if len(candidates) != 1:
        raise DataReadinessError(f"PIT membership is not unique for {(ticker, session_date)}")
    return candidates.iloc[0], session_open, session_close


def _volume_relationships(close: np.ndarray, volume: np.ndarray, *, lookback: int) -> tuple[np.ndarray, np.ndarray]:
    obv = np.full(len(close), np.nan, dtype="float64")
    efficiency = np.full(len(close), np.nan, dtype="float64")
    movement = np.concatenate(([0.0], np.diff(close)))
    signed_volume = np.sign(movement) * volume
    for index in range(lookback - 1, len(close)):
        start = index - lookback + 1
        total_volume = float(volume[start : index + 1].sum())
        direction = float(np.sign(close[index] - close[start]))
        if total_volume > 0.0:
            obv[index] = direction * float(signed_volume[start : index + 1].sum()) / total_volume
        path = float(np.abs(np.diff(close[start : index + 1])).sum())
        efficiency[index] = abs(close[index] - close[start]) / path if path > 0.0 else 0.0
    return obv, efficiency


def _volume_state_output_columns() -> tuple[str, ...]:
    return (
        "volume_bar_number",
        "volume_state_close",
        "volume_state_available_at_utc",
        "volume_state_cumulative_volume",
        *INTRADAY_BAR_MODEL_FEATURE_COLUMNS[:12],
        "volume_bar_progress",
        "normalized_volume_overshoot",
        "volume_bar_duration_minutes",
    )


def _set_reason(reasons: pd.Series, mask: pd.Series, reason: str) -> None:
    selected = reasons.isna() & mask.fillna(False)
    reasons.loc[selected] = reason


def _identity_hash(*parts: str) -> str:
    return hashlib.sha256("|".join(parts).encode("utf-8")).hexdigest()


def _require_columns(frame: pd.DataFrame, required: Iterable[str], label: str) -> None:
    missing = sorted(set(required).difference(frame.columns))
    if missing:
        raise DataReadinessError(f"{label} rows are missing columns: {missing}")
