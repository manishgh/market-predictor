from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.canonical.joins import join_source_collection_status
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.intraday.audits import audit_intraday_dataset
from market_predictor.intraday.contracts import (
    CATALYST_AUDIT_FEATURES,
    INTRADAY_FEATURE_SCHEMA_VERSION,
    SECTOR_BENCHMARKS,
    IntradayDatasetConfig,
)
from market_predictor.intraday.labels import add_exact_one_minute_labels, add_overlap_metadata
from market_predictor.live_features import select_and_audit_live_features
from market_predictor.resources import assert_memory_budget
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

DECISION_REQUIRED_COLUMNS = {
    "ticker",
    "timeframe",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "decision_time_utc",
    "feature_available_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
    "primary_benchmark",
    "universe_snapshot_id",
    "market_cap_bucket",
    "liquidity_bucket",
    "membership_available_at_utc",
}
BAR_REQUIRED_COLUMNS = {
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
    "price_feed",
    "adjustment",
}
GLOBAL_EVENT_WINDOWS = {
    "2h": pd.Timedelta(hours=2),
    "1d": pd.Timedelta(days=1),
}


def build_intraday_dataset(
    five_minute_decisions: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    benchmark_five_minute_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: IntradayDatasetConfig | None = None,
) -> tuple[pd.DataFrame, CanonicalAuditReport]:
    """Build completed-5m decisions and exact subsequent-1m path labels."""

    config = config or IntradayDatasetConfig()
    decisions, one_minute = _build_intraday_feature_history(
        five_minute_decisions,
        one_minute_bars,
        benchmark_five_minute_bars,
        global_events=global_events,
        global_source_collections=global_source_collections,
        config=config,
    )
    decisions = add_exact_one_minute_labels(decisions, one_minute, config)
    decisions = add_overlap_metadata(decisions)
    _guard_memory(config, "intraday exact path labeling")
    decisions["horizon_minutes"] = config.horizon_minutes
    decisions["decision_bar_minutes"] = config.decision_bar_minutes
    decisions["execution_bar_minutes"] = config.execution_bar_minutes
    decisions["decision_stride_bars"] = config.decision_stride_bars
    decisions["target_atr_multiple"] = config.target_atr
    decisions["stop_atr_multiple"] = config.stop_atr
    decisions["round_trip_cost_bps"] = config.round_trip_cost_bps
    decisions["minimum_five_minute_bars"] = config.min_five_minute_bars
    decisions["minimum_one_minute_bars"] = config.min_one_minute_bars
    decisions["dataset_label_config_sha256"] = config.label_config_sha256()
    decisions["execution_policy_sha256"] = EXECUTION_POLICY_SHA256
    decisions = decisions.replace([np.inf, -np.inf], np.nan)
    audit = audit_intraday_dataset(decisions, config)
    return (
        decisions.sort_values(["decision_time_utc", "ticker"], kind="stable").reset_index(drop=True),
        audit,
    )


def build_intraday_inference_features(
    five_minute_decisions: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    benchmark_five_minute_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: IntradayDatasetConfig | None = None,
) -> tuple[pd.DataFrame, CanonicalAuditReport]:
    """Build one audited latest completed-5m group without future path labels."""

    config = config or IntradayDatasetConfig()
    decisions, _ = _build_intraday_feature_history(
        five_minute_decisions,
        one_minute_bars,
        benchmark_five_minute_bars,
        global_events=global_events,
        global_source_collections=global_source_collections,
        config=config,
    )
    return select_and_audit_live_features(
        decisions,
        mode="intraday",
        required_price_feed=config.required_price_feed,
        required_adjustment=config.required_adjustment,
        minimum_bar_count=config.min_five_minute_bars,
        minimum_one_minute_bar_count=config.min_one_minute_bars,
        minimum_cross_section=config.minimum_cross_section,
        source_coverage_max_age_minutes=config.source_coverage_max_age_minutes,
        required_global_sources=config.required_global_sources,
    )


def _build_intraday_feature_history(
    five_minute_decisions: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    benchmark_five_minute_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(five_minute_decisions, DECISION_REQUIRED_COLUMNS, "canonical five-minute decisions")
    _require_columns(one_minute_bars, BAR_REQUIRED_COLUMNS, "canonical one-minute bars")
    _require_columns(
        benchmark_five_minute_bars,
        BAR_REQUIRED_COLUMNS,
        "canonical five-minute benchmark bars",
    )
    decisions = _prepare_bars(
        five_minute_decisions,
        timeframe="5m",
        name="canonical five-minute decisions",
        require_membership=True,
    )
    one_minute = _prepare_bars(
        one_minute_bars,
        timeframe="1m",
        name="canonical one-minute bars",
        require_membership=False,
    )
    benchmarks = _prepare_bars(
        benchmark_five_minute_bars,
        timeframe="5m",
        name="canonical five-minute benchmark bars",
        require_membership=False,
    )
    _guard_memory(config, "intraday input normalization")
    if decisions.empty or one_minute.empty or benchmarks.empty:
        raise DataReadinessError("intraday dataset requires non-empty 5m decisions, 1m bars, and benchmarks")
    if bool(decisions.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError("five-minute decisions contain duplicate ticker/bar rows")
    if bool(one_minute.duplicated(["ticker", "bar_start_utc"]).any()):
        raise DataReadinessError("one-minute bars contain duplicate ticker/bar rows")

    decisions = _add_five_minute_features(decisions, config)
    benchmark_features = _add_five_minute_features(benchmarks, config)
    decisions = _select_decision_rows(decisions, config)
    decisions = _join_one_minute_features(decisions, one_minute, config)
    decisions = _join_benchmark_features(decisions, benchmark_features, config)
    _guard_memory(config, "intraday technical feature construction")
    decisions = _add_breadth_regime_and_relative_features(decisions)
    decisions = _add_global_event_features(decisions, global_events)
    decisions = _add_global_source_status(
        decisions,
        global_source_collections,
        config.required_global_sources,
    )
    decisions = _add_catalyst_features_and_eligibility(decisions, config)
    decisions = _add_membership_features(decisions)
    decisions = _add_cross_sectional_features(decisions, config)
    _guard_memory(config, "intraday context feature construction")
    decisions["intraday_feature_schema_version"] = INTRADAY_FEATURE_SCHEMA_VERSION
    decisions = decisions.replace([np.inf, -np.inf], np.nan)
    return decisions, one_minute


def _guard_memory(config: IntradayDatasetConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.max_build_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage=stage,
    )


def _prepare_bars(
    frame: pd.DataFrame,
    *,
    timeframe: str,
    name: str,
    require_membership: bool,
) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["timeframe"] = data["timeframe"].astype(str).str.lower().str.strip()
    data = data[data["timeframe"].eq(timeframe)].copy()
    timestamp_columns = ["bar_start_utc", "bar_end_utc", "available_at_utc"]
    if require_membership:
        timestamp_columns.extend(["decision_time_utc", "feature_available_at_utc", "membership_available_at_utc"])
    for column in timestamp_columns:
        data[column] = _strict_utc(data[column], f"{name}.{column}")
    numeric = ["open", "high", "low", "close", "volume"]
    data[numeric] = data[numeric].apply(pd.to_numeric, errors="coerce")
    invalid_ohlcv = data[numeric].isna().any(axis=1) | data[["open", "high", "low", "close"]].le(0).any(axis=1) | data["volume"].lt(0)
    if bool(invalid_ohlcv.any()):
        raise DataReadinessError(f"{name} contains invalid OHLCV")
    eastern = data["bar_start_utc"].dt.tz_convert("America/New_York")
    minute = eastern.dt.hour * 60 + eastern.dt.minute
    expected_minutes = 5 if timeframe == "5m" else 1
    aligned = minute.sub(9 * 60 + 30).mod(expected_minutes).eq(0)
    regular = minute.between(9 * 60 + 30, 15 * 60 + 59) & aligned
    data = data[regular].copy()
    data["session_date_et"] = eastern.loc[data.index].dt.date
    data["session_minute_et"] = minute.loc[data.index].astype("int16")
    data["session_slot"] = ((data["session_minute_et"] - (9 * 60 + 30)) // expected_minutes).astype("int16")
    expected_duration = pd.Timedelta(minutes=expected_minutes)
    if bool(data["bar_end_utc"].sub(data["bar_start_utc"]).ne(expected_duration).any()):
        raise DataReadinessError(f"{name} contains bars with an unexpected interval")
    return data.sort_values(["ticker", "bar_start_utc"], kind="stable").reset_index(drop=True)


def _add_five_minute_features(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    parts = [_five_minute_ticker_features(part, config) for _, part in frame.groupby("ticker", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else frame.copy()


def _five_minute_ticker_features(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.sort_values("bar_start_utc", kind="stable").copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    session = data.groupby("session_date_et", sort=False)
    data["five_minute_bar_count"] = np.arange(1, len(data) + 1, dtype=np.int32)
    transition = _grid_transition_valid(data, bars_per_session=78)
    data["five_minute_history_exact"] = (
        transition.rolling(config.min_five_minute_bars, min_periods=config.min_five_minute_bars).min().fillna(0).astype(bool)
    )
    one_return = session["close"].pct_change(fill_method=None)
    for window in (1, 3, 6, 12):
        data[f"return_{window}bar_5m"] = session["close"].pct_change(window, fill_method=None)
    for window in (6, 12):
        data[f"realized_vol_{window}bar_5m"] = one_return.groupby(data["session_date_et"], sort=False).transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std()
        )
    for span in (10, 20, 50):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        data[f"dist_ema_{span}_5m"] = close / ema - 1.0
        if span in {10, 20}:
            data[f"ema_{span}_slope_3bar_5m"] = ema.pct_change(3, fill_method=None)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    data["macd_signal_diff_pct_5m"] = (macd - macd_signal) / close
    data["rsi_14_5m"] = _rsi(close, 14)
    prior_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_14_price_5m"] = atr
    data["atr_pct_14_5m"] = atr / close
    typical = data[["high", "low", "close"]].mean(axis=1)
    cumulative_dollar = (typical * volume).groupby(data["session_date_et"], sort=False).cumsum()
    cumulative_volume = volume.groupby(data["session_date_et"], sort=False).cumsum()
    data["session_vwap_5m"] = cumulative_dollar / cumulative_volume.replace(0, np.nan)
    data["dist_session_vwap_5m"] = close / data["session_vwap_5m"] - 1.0
    data["session_vwap_slope_3bar_5m"] = session["session_vwap_5m"].pct_change(3, fill_method=None)
    opening = data["session_slot"].between(0, 5)
    opening_high = high.where(opening).groupby(data["session_date_et"], sort=False).cummax()
    opening_low = low.where(opening).groupby(data["session_date_et"], sort=False).cummin()
    opening_high = opening_high.groupby(data["session_date_et"], sort=False).ffill()
    opening_low = opening_low.groupby(data["session_date_et"], sort=False).ffill()
    data["opening_range_width_pct_5m"] = (opening_high - opening_low) / close
    data["dist_opening_range_high_5m"] = close / opening_high - 1.0
    data["dist_opening_range_low_5m"] = close / opening_low - 1.0
    data["overnight_gap"] = _overnight_gap(data)
    slot_baseline = volume.groupby(data["session_slot"], sort=False).transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    data["relative_volume_same_slot_20d_5m"] = volume / slot_baseline.replace(0, np.nan)
    prior_volume = volume.shift(1).rolling(20, min_periods=5).median()
    data["volume_burst_20bar_5m"] = volume / prior_volume.replace(0, np.nan)
    data["dollar_volume_log_5m"] = np.log1p(close * volume)
    recent_high = high.shift(1).rolling(20, min_periods=5).max()
    recent_low = low.shift(1).rolling(20, min_periods=5).min()
    data["dist_recent_20bar_high_atr_5m"] = (close - recent_high) / atr.replace(0, np.nan)
    data["dist_recent_20bar_low_atr_5m"] = (close - recent_low) / atr.replace(0, np.nan)
    prior = prior_close.replace(0, np.nan)
    data["range_pct_5m"] = (high - low) / prior
    spread = (high - low).replace(0, np.nan)
    data["close_location_5m"] = (close - low) / spread
    data["session_progress"] = data["session_slot"] / 77.0
    radians = 2 * np.pi * data["session_progress"]
    data["session_minute_sin"] = np.sin(radians)
    data["session_minute_cos"] = np.cos(radians)
    return data


def _select_decision_rows(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.copy()
    selected = data["session_minute_et"].between(
        config.first_decision_minute_et,
        config.last_decision_minute_et,
    ) & data["session_slot"].mod(config.decision_stride_bars).eq(0)
    data = data[selected].copy()
    data["decision_group_id"] = data["bar_end_utc"].map(lambda value: value.isoformat())
    return data.reset_index(drop=True)


def _join_one_minute_features(
    decisions: pd.DataFrame,
    one_minute_bars: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    tickers = set(decisions["ticker"].astype(str).unique())
    bars = one_minute_bars[one_minute_bars["ticker"].isin(tickers)].copy()
    parts = [_one_minute_ticker_features(part, config) for _, part in bars.groupby("ticker", sort=False)]
    features = pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()
    if features.empty:
        raise DataReadinessError("one-minute bars do not cover any decision ticker")
    feature_columns = [
        "available_at_utc",
        "bar_start_utc",
        "one_minute_bar_count",
        "one_minute_history_exact",
        *[column for column in features if column.endswith("_1m")],
    ]
    output_parts: list[pd.DataFrame] = []
    for ticker, left in decisions.groupby("ticker", sort=False):
        right = features[features["ticker"].eq(ticker)].sort_values("available_at_utc")
        joined = pd.merge_asof(
            left.sort_values("decision_time_utc"),
            right[feature_columns],
            left_on="decision_time_utc",
            right_on="available_at_utc",
            direction="backward",
            allow_exact_matches=True,
            suffixes=("", "_latest_1m"),
        )
        joined = joined.rename(
            columns={
                "available_at_utc_latest_1m": "one_minute_available_at_utc",
                "bar_start_utc_latest_1m": "one_minute_bar_start_utc",
            }
        )
        output_parts.append(joined)
    output = pd.concat(output_parts, ignore_index=True)
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        ["feature_available_at_utc", "one_minute_available_at_utc"],
    )
    return output


def _one_minute_ticker_features(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.sort_values("bar_start_utc", kind="stable").copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    volume = data["volume"].astype(float)
    session = data.groupby("session_date_et", sort=False)
    data["one_minute_bar_count"] = np.arange(1, len(data) + 1, dtype=np.int32)
    transition = _grid_transition_valid(data, bars_per_session=390)
    data["one_minute_history_exact"] = (
        transition.rolling(config.min_one_minute_bars, min_periods=config.min_one_minute_bars).min().fillna(0).astype(bool)
    )
    one_return = session["close"].pct_change(fill_method=None)
    for window in (1, 3, 5):
        data[f"return_{window}bar_1m"] = session["close"].pct_change(window, fill_method=None)
    for window in (5, 20):
        data[f"realized_vol_{window}bar_1m"] = one_return.groupby(data["session_date_et"], sort=False).transform(
            lambda values, size=window: values.rolling(size, min_periods=size).std()
        )
    for span in (5, 20):
        ema = close.ewm(span=span, adjust=False, min_periods=span).mean()
        data[f"dist_ema_{span}_1m"] = close / ema - 1.0
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    data["macd_signal_diff_pct_1m"] = (macd - signal) / close
    data["rsi_14_1m"] = _rsi(close, 14)
    prior_close = close.shift(1)
    true_range = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    atr = true_range.ewm(alpha=1 / 14, adjust=False, min_periods=14).mean()
    data["atr_pct_14_1m"] = atr / close
    typical = data[["high", "low", "close"]].mean(axis=1)
    cumulative_dollar = (typical * volume).groupby(data["session_date_et"], sort=False).cumsum()
    cumulative_volume = volume.groupby(data["session_date_et"], sort=False).cumsum()
    vwap = cumulative_dollar / cumulative_volume.replace(0, np.nan)
    data["dist_session_vwap_1m"] = close / vwap - 1.0
    prior_volume = volume.shift(1).rolling(20, min_periods=5).median()
    data["volume_burst_20bar_1m"] = volume / prior_volume.replace(0, np.nan)
    slot_baseline = volume.groupby(data["session_slot"], sort=False).transform(
        lambda values: values.shift(1).rolling(20, min_periods=5).median()
    )
    data["relative_volume_same_slot_20d_1m"] = volume / slot_baseline.replace(0, np.nan)
    data["range_pct_1m"] = (high - low) / prior_close.replace(0, np.nan)
    spread = (high - low).replace(0, np.nan)
    data["close_location_1m"] = (close - low) / spread
    return data


def _join_benchmark_features(
    decisions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    output = decisions.copy()
    lookup = benchmarks.set_index(["ticker", "bar_end_utc"])
    for name, tickers in (
        ("spy", pd.Series(config.broad_benchmark.upper(), index=output.index)),
        ("qqq", pd.Series(config.growth_benchmark.upper(), index=output.index)),
        ("sector", output["primary_benchmark"].astype(str).str.upper()),
    ):
        keys = pd.MultiIndex.from_arrays([tickers, output["bar_end_utc"]])
        for window in (1, 3, 6):
            source = f"return_{window}bar_5m"
            output[f"{name}_return_{window}bar_5m"] = _lookup_values(lookup, keys, source)
        output[f"{name}_dist_session_vwap_5m"] = _lookup_values(
            lookup,
            keys,
            "dist_session_vwap_5m",
        )
        output[f"{name}_available_at_utc"] = pd.to_datetime(
            _lookup_values(lookup, keys, "available_at_utc", numeric=False),
            utc=True,
            errors="coerce",
        )
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        [
            "feature_available_at_utc",
            "spy_available_at_utc",
            "qqq_available_at_utc",
            "sector_available_at_utc",
        ],
    )
    return output


def _add_breadth_regime_and_relative_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for window in (1, 3, 6):
        data[f"rel_return_{window}bar_vs_qqq_5m"] = data[f"return_{window}bar_5m"] - data[f"qqq_return_{window}bar_5m"]
        data[f"rel_return_{window}bar_vs_sector_5m"] = data[f"return_{window}bar_5m"] - data[f"sector_return_{window}bar_5m"]
    grouped = data.groupby("decision_group_id", sort=False)
    data["eligible_breadth_positive_1bar"] = grouped["return_1bar_5m"].transform(lambda values: values.gt(0).mean())
    data["eligible_breadth_above_vwap"] = grouped["dist_session_vwap_5m"].transform(lambda values: values.gt(0).mean())
    qqq_positive = data["qqq_return_6bar_5m"].gt(0)
    breadth_positive = data["eligible_breadth_positive_1bar"].ge(0.55)
    data["regime_risk_on"] = (qqq_positive & breadth_positive).astype("int8")
    data["regime_risk_off"] = (data["qqq_return_6bar_5m"].lt(0) & data["eligible_breadth_positive_1bar"].le(0.45)).astype("int8")
    volatility = data.groupby("session_date_et", sort=False)["qqq_return_1bar_5m"].transform(
        lambda values: values.expanding(min_periods=6).std()
    )
    threshold = volatility.groupby(data["session_date_et"], sort=False).transform(lambda values: values.expanding(min_periods=6).median())
    data["regime_high_volatility"] = volatility.gt(threshold).fillna(False).astype("int8")
    data["market_regime"] = np.select(
        [
            data["regime_risk_off"].eq(1),
            data["regime_risk_on"].eq(1),
            data["regime_high_volatility"].eq(1),
        ],
        ["risk_off", "risk_on", "high_volatility"],
        default="neutral",
    )
    return data


def _add_global_event_features(decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "ticker",
        "event_id",
        "feature_available_at_utc",
        "availability_policy",
        "sentiment_numeric",
        "relevance",
    }
    _require_columns(events, required, "canonical global events")
    event_frame = events[events["ticker"].astype(str).str.upper().eq("MARKET")].copy()
    if not event_frame.empty and bool(event_frame["availability_policy"].astype(str).ne("observed").any()):
        raise DataReadinessError("intraday production features reject proxy global events")
    if not event_frame.empty:
        event_frame["feature_available_at_utc"] = _strict_utc(
            event_frame["feature_available_at_utc"],
            "global events.feature_available_at_utc",
        )
        event_frame = event_frame.sort_values("feature_available_at_utc").drop_duplicates(
            "event_id",
            keep="first",
        )
    output = decisions.copy()
    unique = output[["decision_time_utc"]].drop_duplicates().sort_values("decision_time_utc")
    decision_ns = pd.DatetimeIndex(unique["decision_time_utc"]).as_unit("ns").asi8
    event_ns = (
        pd.DatetimeIndex(event_frame["feature_available_at_utc"]).as_unit("ns").asi8
        if not event_frame.empty
        else np.array([], dtype=np.int64)
    )
    sentiment = pd.to_numeric(event_frame.get("sentiment_numeric"), errors="coerce").fillna(0.0).to_numpy(float)
    sentiment_present = pd.to_numeric(event_frame.get("sentiment_numeric"), errors="coerce").notna().to_numpy(float)
    # Unknown relevance carries zero weight in global sentiment (excluded, not fully relevant).
    relevance = pd.to_numeric(event_frame.get("relevance"), errors="coerce").fillna(0.0).clip(lower=0).to_numpy(float)
    end = np.searchsorted(event_ns, decision_ns, side="right")
    for name, window in GLOBAL_EVENT_WINDOWS.items():
        start = np.searchsorted(event_ns, decision_ns - int(window.value), side="left")
        counts = end - start
        weights = _window_sum(relevance * sentiment_present, start, end)
        unique[f"global_event_count_{name}"] = counts
        unique[f"global_sentiment_mean_{name}"] = np.divide(
            _window_sum(sentiment * relevance, start, end),
            weights,
            out=np.zeros(len(unique)),
            where=weights > 0,
        )
        unique[f"global_sentiment_coverage_{name}"] = np.divide(
            _window_sum(sentiment_present, start, end),
            counts,
            out=np.zeros(len(unique)),
            where=counts > 0,
        )
    latest = end - 1
    values = np.full(len(unique), np.datetime64("NaT"), dtype="datetime64[ns]")
    present = latest >= 0
    if present.any():
        values[present] = event_frame["feature_available_at_utc"].to_numpy(dtype="datetime64[ns]")[latest[present]]
    unique["global_event_feature_available_at_utc"] = pd.to_datetime(values, utc=True)
    output = output.merge(unique, on="decision_time_utc", how="left", validate="many_to_one")
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        ["feature_available_at_utc", "global_event_feature_available_at_utc"],
    )
    return output


def _add_global_source_status(
    decisions: pd.DataFrame,
    collections: pd.DataFrame,
    required_sources: Sequence[str],
) -> pd.DataFrame:
    unique = decisions[["decision_time_utc"]].drop_duplicates().sort_values("decision_time_utc").copy()
    unique["ticker"] = "MARKET"
    joined = join_source_collection_status(unique, collections, source_families=required_sources)
    rename = {column: f"global_{column}" for column in joined if column.startswith("source_")}
    joined = joined.rename(columns=rename).drop(columns="ticker")
    output = decisions.merge(joined, on="decision_time_utc", how="left", validate="many_to_one")
    availability = [f"global_source_status_available_at_utc_{source.strip().lower()}" for source in required_sources]
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        ["feature_available_at_utc", *availability],
    )
    return output


def _add_catalyst_features_and_eligibility(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.copy()
    for feature in CATALYST_AUDIT_FEATURES:
        if feature not in data.columns:
            data[feature] = 0.0
        data[feature] = pd.to_numeric(data[feature], errors="coerce").fillna(0.0)
    eligible = pd.Series(True, index=data.index)
    max_age = pd.Timedelta(minutes=config.source_coverage_max_age_minutes)
    for prefix, sources in (
        ("", config.required_catalyst_sources),
        ("global_", config.required_global_sources),
    ):
        for source in sources:
            normalized = source.strip().lower()
            status_column = f"{prefix}source_status_{normalized}"
            available_column = f"{prefix}source_status_available_at_utc_{normalized}"
            coverage_column = f"{prefix}source_coverage_end_utc_{normalized}"
            _require_columns(
                data,
                {status_column, available_column, coverage_column},
                f"{prefix or 'ticker_'}catalyst source state",
            )
            status = data[status_column].astype(str).str.lower().str.strip()
            available = _nullable_utc(data[available_column])
            coverage = _nullable_utc(data[coverage_column])
            fresh = (
                available.notna()
                & coverage.notna()
                & coverage.le(available)
                & coverage.le(data["decision_time_utc"])
                & data["decision_time_utc"].sub(coverage).le(max_age)
            )
            eligible &= status.isin({"observed", "observed_empty"}) & fresh
    data["catalyst_eligible"] = eligible.astype(bool)
    return data


def _add_membership_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    cap = data["market_cap_bucket"].fillna("").astype(str).str.lower()
    liquidity = data["liquidity_bucket"].fillna("").astype(str).str.lower()
    for name in ("micro", "small", "mid", "large", "mega"):
        data[f"market_cap_{name}"] = cap.str.contains(name, regex=False).astype("int8")
    for name in ("low", "medium", "high"):
        data[f"liquidity_{name}"] = liquidity.str.contains(name, regex=False).astype("int8")
    benchmark = data["primary_benchmark"].fillna("").astype(str).str.upper()
    for ticker in SECTOR_BENCHMARKS:
        data[f"sector_benchmark_{ticker.lower()}"] = benchmark.eq(ticker).astype("int8")
    return data


def _add_cross_sectional_features(
    frame: pd.DataFrame,
    config: IntradayDatasetConfig,
) -> pd.DataFrame:
    data = frame.copy()
    rank_inputs = {
        "return_3bar_5m": "xs_rank_return_3bar_5m",
        "rel_return_3bar_vs_qqq_5m": "xs_rank_rel_return_3bar_vs_qqq_5m",
        "rel_return_3bar_vs_sector_5m": "xs_rank_rel_return_3bar_vs_sector_5m",
        "relative_volume_same_slot_20d_5m": "xs_rank_relative_volume_same_slot_20d_5m",
        "volume_burst_20bar_5m": "xs_rank_volume_burst_20bar_5m",
        "atr_pct_14_5m": "xs_rank_atr_pct_14_5m",
        "dist_session_vwap_5m": "xs_rank_dist_session_vwap_5m",
        "return_3bar_1m": "xs_rank_return_3bar_1m",
    }
    grouped = data.groupby("decision_group_id", sort=False)
    for source, target in rank_inputs.items():
        data[target] = grouped[source].rank(method="average", pct=True)
    core_ready = (
        data[
            [
                "dist_ema_50_5m",
                "macd_signal_diff_pct_5m",
                "atr_pct_14_5m",
                "return_3bar_1m",
                "rsi_14_1m",
            ]
        ]
        .notna()
        .all(axis=1)
    )
    benchmark_ready = data[["spy_available_at_utc", "qqq_available_at_utc", "sector_available_at_utc"]].notna().all(axis=1)
    data["feature_eligible"] = (
        data["five_minute_bar_count"].ge(config.min_five_minute_bars)
        & data["one_minute_bar_count"].ge(config.min_one_minute_bars)
        & data["five_minute_history_exact"].fillna(False).astype(bool)
        & data["one_minute_history_exact"].fillna(False).astype(bool)
        & core_ready
        & benchmark_ready
        & data["price_feed"].astype(str).str.lower().eq(config.required_price_feed)
        & data["adjustment"].astype(str).str.lower().eq(config.required_adjustment)
    )
    count = data["feature_eligible"].groupby(data["decision_group_id"]).transform("sum")
    data["cross_section_size"] = count.astype("int32")
    data["cross_section_eligible"] = count.ge(config.minimum_cross_section)
    data["feature_eligible"] &= data["cross_section_eligible"]
    return data


def _grid_transition_valid(data: pd.DataFrame, *, bars_per_session: int) -> pd.Series:
    previous_session = data["session_date_et"].shift(1)
    previous_slot = data["session_slot"].shift(1)
    same_session = data["session_date_et"].eq(previous_session)
    transition = (same_session & data["session_slot"].eq(previous_slot + 1)) | (
        ~same_session & data["session_slot"].eq(0) & previous_slot.eq(bars_per_session - 1)
    )
    if not transition.empty:
        transition.iloc[0] = True
    return transition.astype("int8")


def _overnight_gap(data: pd.DataFrame) -> pd.Series:
    sessions = data.groupby("session_date_et", sort=False).agg(
        session_open=("open", "first"),
        session_close=("close", "last"),
    )
    sessions["prior_close"] = sessions["session_close"].shift(1)
    sessions["gap"] = sessions["session_open"] / sessions["prior_close"] - 1.0
    return data["session_date_et"].map(sessions["gap"])


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).ewm(alpha=1 / window, adjust=False, min_periods=window).mean()
    relative = gain / loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    result = result.where(~(loss.eq(0) & gain.gt(0)), 100.0)
    return result.where(~(loss.eq(0) & gain.eq(0)), 50.0)


def _lookup_values(
    lookup: pd.DataFrame,
    keys: pd.MultiIndex,
    column: str,
    *,
    numeric: bool = True,
) -> np.ndarray:
    if column not in lookup.columns:
        return np.full(len(keys), np.nan if numeric else None)
    values = lookup[column].reindex(keys)
    if numeric:
        return np.asarray(pd.to_numeric(values, errors="coerce").to_numpy(float), dtype=float)
    return np.asarray(values.to_numpy(), dtype=object)


def _window_sum(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return np.asarray(cumulative[end] - cumulative[start], dtype=float)


def _row_timestamp_max(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    parsed = pd.concat(
        [pd.to_datetime(frame[column], utc=True, errors="coerce") for column in available],
        axis=1,
    )
    return pd.to_datetime(parsed.max(axis=1), utc=True)


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = _nullable_utc(values)
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid or timezone-naive timestamps")
    return parsed


def _nullable_utc(values: pd.Series) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        if value is None or pd.isna(value):
            return pd.NaT
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    return pd.to_datetime(values.map(parse), utc=True)


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"{name} missing columns: {', '.join(missing)}")
