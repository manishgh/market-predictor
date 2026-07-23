from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.canonical.joins import join_source_collection_status
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.live_features import select_and_audit_live_features
from market_predictor.swing.audits import audit_swing_dataset
from market_predictor.swing.contracts import (
    CATALYST_FEATURES,
    FUNDAMENTAL_FEATURES,
    SECTOR_BENCHMARKS,
    SWING_FEATURE_SCHEMA_VERSION,
    SwingDatasetConfig,
    swing_excess_column,
    swing_net_return_column,
    swing_target_column,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

DECISION_REQUIRED_COLUMNS = {
    "ticker",
    "timeframe",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "bar_available_at_utc",
    "decision_time_utc",
    "feature_available_at_utc",
    "prediction_cutoff_policy_id",
    "decision_group_id",
    "session_date_et",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
    "primary_benchmark",
    "market_cap_bucket",
    "liquidity_bucket",
    "membership_available_at_utc",
}
BENCHMARK_REQUIRED_COLUMNS = {
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


def build_swing_dataset(
    decisions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: SwingDatasetConfig | None = None,
) -> tuple[pd.DataFrame, CanonicalAuditReport]:
    """Build post-close swing features and next-open labels from canonical inputs."""

    config = config or SwingDatasetConfig()
    data, benchmark_features = _build_swing_feature_history(
        decisions,
        benchmark_bars,
        global_events=global_events,
        global_source_collections=global_source_collections,
        config=config,
    )
    data = _add_exact_labels(data, benchmark_features, config)
    audit = audit_swing_dataset(data, config)
    return data.sort_values(["decision_time_utc", "ticker"], kind="stable").reset_index(drop=True), audit


def build_swing_inference_features(
    decisions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: SwingDatasetConfig | None = None,
) -> tuple[pd.DataFrame, CanonicalAuditReport]:
    """Build one audited latest post-close feature group without future labels."""

    config = config or SwingDatasetConfig()
    data, _ = _build_swing_feature_history(
        decisions,
        benchmark_bars,
        global_events=global_events,
        global_source_collections=global_source_collections,
        config=config,
    )
    return select_and_audit_live_features(
        data,
        mode="swing",
        required_price_feed=config.required_price_feed,
        required_adjustment=config.required_adjustment,
        minimum_bar_count=config.min_daily_bars,
        minimum_cross_section=config.minimum_cross_section,
        source_coverage_max_age_minutes=config.source_coverage_max_age_minutes,
        required_global_sources=config.required_global_sources,
    )


def _build_swing_feature_history(
    decisions: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    global_events: pd.DataFrame,
    global_source_collections: pd.DataFrame,
    config: SwingDatasetConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    _require_columns(decisions, DECISION_REQUIRED_COLUMNS, "canonical decisions")
    _require_columns(benchmark_bars, BENCHMARK_REQUIRED_COLUMNS, "canonical benchmark bars")
    data = _prepare_daily_rows(decisions, name="canonical decisions")
    benchmarks = _prepare_daily_rows(benchmark_bars, name="canonical benchmark bars")
    if data.empty or benchmarks.empty:
        raise DataReadinessError("swing dataset requires non-empty daily decisions and benchmark bars")
    if bool(data.duplicated(["ticker", "session_date_et"]).any()):
        raise DataReadinessError("canonical decisions contain duplicate ticker/session rows")
    if bool(benchmarks.duplicated(["ticker", "session_date_et"]).any()):
        raise DataReadinessError("benchmark bars contain duplicate ticker/session rows")

    data = _add_technical_features(data)
    benchmark_features = _add_technical_features(benchmarks)
    data = _join_benchmark_features(data, benchmark_features, config)
    data = _add_relative_and_regime_features(data)
    data = _add_global_event_features(data, global_events)
    data = _add_global_source_status(data, global_source_collections, config.required_global_sources)
    data = _add_canonical_optional_features(data)
    data = _add_membership_features(data)
    data = _add_cross_sectional_features(data, config)
    data["horizon_sessions"] = config.horizon_sessions
    data["round_trip_cost_bps"] = config.round_trip_cost_bps
    data["minimum_daily_bars"] = config.min_daily_bars
    data["swing_feature_schema_version"] = SWING_FEATURE_SCHEMA_VERSION
    data["dataset_label_config_sha256"] = config.label_config_sha256()
    data["execution_policy_sha256"] = EXECUTION_POLICY_SHA256
    data = data.replace([np.inf, -np.inf], np.nan)
    return data, benchmark_features


def _prepare_daily_rows(frame: pd.DataFrame, *, name: str) -> pd.DataFrame:
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["timeframe"] = data["timeframe"].astype(str).str.lower().str.strip()
    data = data[data["timeframe"].eq("1d")].copy()
    for column in [
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "bar_available_at_utc",
        "decision_time_utc",
        "feature_available_at_utc",
    ]:
        if column in data.columns:
            data[column] = _strict_utc(data[column], f"{name}.{column}")
    data["session_date_et"] = pd.to_datetime(data["bar_start_utc"], utc=True).dt.tz_convert("America/New_York").dt.date
    for column in ["open", "high", "low", "close", "volume"]:
        data[column] = pd.to_numeric(data[column], errors="coerce")
    if bool(data[["open", "high", "low", "close", "volume"]].isna().any(axis=1).any()):
        raise DataReadinessError(f"{name} contains invalid OHLCV values")
    return data.sort_values(["ticker", "session_date_et"], kind="stable").reset_index(drop=True)


def _add_technical_features(frame: pd.DataFrame) -> pd.DataFrame:
    parts = [_technical_ticker(part) for _, part in frame.groupby("ticker", sort=False)]
    return pd.concat(parts, ignore_index=True) if parts else frame.copy()


def _technical_ticker(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.sort_values("session_date_et", kind="stable").copy()
    close = data["close"].astype(float)
    high = data["high"].astype(float)
    low = data["low"].astype(float)
    open_price = data["open"].astype(float)
    volume = data["volume"].astype(float)
    prior_close = close.shift(1)
    data["daily_bar_count"] = np.arange(1, len(data) + 1, dtype=np.int32)
    for window in (1, 5, 10, 20, 60):
        data[f"return_{window}d"] = close.pct_change(window, fill_method=None)
    daily_return = close.pct_change(fill_method=None)
    for window in (10, 20, 60):
        data[f"realized_vol_{window}d"] = daily_return.rolling(window, min_periods=window).std()
    true_range = pd.concat(
        [high - low, (high - prior_close).abs(), (low - prior_close).abs()],
        axis=1,
    ).max(axis=1)
    data["atr_pct_14"] = true_range.rolling(14, min_periods=14).mean() / close
    data["rsi_14"] = _rsi(close, 14)
    ema_12 = close.ewm(span=12, adjust=False, min_periods=12).mean()
    ema_26 = close.ewm(span=26, adjust=False, min_periods=26).mean()
    macd = ema_12 - ema_26
    macd_signal = macd.ewm(span=9, adjust=False, min_periods=9).mean()
    data["macd_signal_diff_pct"] = (macd - macd_signal) / close
    for window in (10, 20, 50):
        ema = close.ewm(span=window, adjust=False, min_periods=window).mean()
        data[f"dist_ema_{window}"] = close / ema - 1.0
    for window in (20, 50, 200):
        sma = close.rolling(window, min_periods=window).mean()
        data[f"dist_sma_{window}"] = close / sma - 1.0
        if window == 200:
            data["sma_200_slope_20d"] = sma / sma.shift(20) - 1.0
    data["gap_return"] = open_price / prior_close - 1.0
    data["intraday_return"] = close / open_price - 1.0
    data["range_pct"] = (high - low) / prior_close
    spread = (high - low).replace(0, np.nan)
    data["close_location"] = (close - low) / spread
    volume_mean = volume.rolling(20, min_periods=20).mean()
    volume_std = volume.rolling(20, min_periods=20).std().replace(0, np.nan)
    data["volume_z20"] = (volume - volume_mean) / volume_std
    data["volume_ratio_20"] = volume / volume_mean
    data["dollar_volume_log"] = np.log1p(close * volume)
    return data


def _join_benchmark_features(
    decisions: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: SwingDatasetConfig,
) -> pd.DataFrame:
    output = decisions.copy()
    selected = [
        "ticker",
        "session_date_et",
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "open",
        "high",
        "low",
        "close",
        "return_1d",
        "return_5d",
        "return_20d",
        "realized_vol_20d",
        "dist_sma_200",
    ]
    for ticker, prefix in (
        (config.broad_benchmark.upper(), "spy"),
        (config.growth_benchmark.upper(), "qqq"),
    ):
        part = benchmarks.loc[benchmarks["ticker"].eq(ticker), selected].copy()
        renamed = {column: f"{prefix}_{column}" for column in selected if column not in {"ticker", "session_date_et"}}
        part = part.rename(columns=renamed).drop(columns="ticker")
        part = part.rename(columns={f"{prefix}_available_at_utc": f"{prefix}_available_at_utc"})
        output = output.merge(part, on="session_date_et", how="left", validate="many_to_one")

    sector = benchmarks.loc[:, selected].rename(columns={"ticker": "primary_benchmark"})
    sector = sector.rename(columns={column: f"sector_{column}" for column in selected if column not in {"ticker", "session_date_et"}})
    output["primary_benchmark"] = output["primary_benchmark"].astype(str).str.upper().str.strip()
    output = output.merge(
        sector,
        on=["primary_benchmark", "session_date_et"],
        how="left",
        validate="many_to_one",
    )
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        ["feature_available_at_utc", "spy_available_at_utc", "qqq_available_at_utc", "sector_available_at_utc"],
    )
    return output


def _add_relative_and_regime_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for window in (1, 5, 20):
        data[f"rel_return_{window}d_vs_spy"] = data[f"return_{window}d"] - data[f"spy_return_{window}d"]
        data[f"rel_return_{window}d_vs_sector"] = data[f"return_{window}d"] - data[f"sector_return_{window}d"]
    risk_on = data["spy_dist_sma_200"].gt(0) & data["spy_return_20d"].gt(0)
    risk_off = data["spy_dist_sma_200"].lt(0) & data["spy_return_20d"].lt(0)
    data["regime_risk_on"] = risk_on.astype("int8")
    data["regime_risk_off"] = risk_off.astype("int8")
    data["market_regime"] = np.select([risk_on, risk_off], ["risk_on", "risk_off"], default="neutral")
    return data


def _add_global_event_features(decisions: pd.DataFrame, events: pd.DataFrame) -> pd.DataFrame:
    required = {"event_id", "feature_available_at_utc", "availability_policy", "sentiment_numeric", "relevance"}
    _require_columns(events, required, "canonical global events")
    if bool(events["availability_policy"].astype(str).ne("observed").any()):
        raise DataReadinessError("production swing global context requires observed events")
    output = decisions.copy()
    event_frame = events.copy()
    event_frame["feature_available_at_utc"] = _strict_utc(
        event_frame["feature_available_at_utc"],
        "global_events.feature_available_at_utc",
    )
    event_frame = event_frame.sort_values("feature_available_at_utc").drop_duplicates("event_id", keep="first")
    event_times = pd.DatetimeIndex(event_frame["feature_available_at_utc"]).as_unit("ns").asi8
    sentiment = pd.to_numeric(event_frame["sentiment_numeric"], errors="coerce")
    sentiment_values = sentiment.fillna(0.0).to_numpy(dtype=float)
    sentiment_present = sentiment.notna().to_numpy(dtype=float)
    # Unknown relevance carries zero weight in global sentiment (excluded, not fully relevant).
    relevance = pd.to_numeric(event_frame["relevance"], errors="coerce").fillna(0.0).clip(lower=0).to_numpy(dtype=float)
    unique = output[["decision_time_utc"]].drop_duplicates().sort_values("decision_time_utc").copy()
    decision_times = pd.DatetimeIndex(unique["decision_time_utc"]).as_unit("ns").asi8
    end = np.searchsorted(event_times, decision_times, side="right")
    for name, duration in (("1d", pd.Timedelta(days=1)), ("3d", pd.Timedelta(days=3))):
        start = np.searchsorted(event_times, decision_times - int(duration.value), side="left")
        counts = end - start
        weighted = _window_sum(sentiment_values * relevance, start, end)
        weights = _window_sum(sentiment_present * relevance, start, end)
        unique[f"global_event_count_{name}"] = counts
        unique[f"global_sentiment_mean_{name}"] = np.divide(
            weighted,
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
    rename = {column: f"global_{column}" for column in joined.columns if column.startswith("source_")}
    joined = joined.rename(columns=rename).drop(columns="ticker")
    output = decisions.merge(joined, on="decision_time_utc", how="left", validate="many_to_one")
    availability_columns = [f"global_source_status_available_at_utc_{source.strip().lower()}" for source in required_sources]
    output["feature_available_at_utc"] = _row_timestamp_max(
        output,
        ["feature_available_at_utc", *availability_columns],
    )
    return output


def _add_canonical_optional_features(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.copy()
    for feature in CATALYST_FEATURES:
        if feature not in data.columns:
            data[feature] = 0.0
        data[feature] = pd.to_numeric(data[feature], errors="coerce").fillna(0.0)
    for metric in ("revenue", "net_income", "eps_diluted", "operating_cash_flow"):
        feature = f"fundamental_{metric}"
        if feature not in data.columns:
            data[feature] = np.nan
        data[feature] = pd.to_numeric(data[feature], errors="coerce")
        data[f"{feature}_present"] = data[feature].notna().astype("int8")
    for feature in FUNDAMENTAL_FEATURES:
        if feature not in data.columns:
            data[feature] = 0.0 if feature.endswith("_present") else np.nan
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


def _add_cross_sectional_features(frame: pd.DataFrame, config: SwingDatasetConfig) -> pd.DataFrame:
    data = frame.copy()
    rank_inputs = {
        "return_5d": "xs_rank_return_5d",
        "return_20d": "xs_rank_return_20d",
        "volume_z20": "xs_rank_volume_z20",
        "rel_return_20d_vs_spy": "xs_rank_rel_return_20d_vs_spy",
        "rel_return_20d_vs_sector": "xs_rank_rel_return_20d_vs_sector",
    }
    grouped = data.groupby("decision_group_id", sort=False)
    for source, target in rank_inputs.items():
        data[target] = grouped[source].rank(method="average", pct=True)
    core_ready = data[["dist_sma_200", "sma_200_slope_20d", "return_60d", "spy_dist_sma_200"]].notna().all(axis=1)
    benchmark_ready = data[["spy_available_at_utc", "qqq_available_at_utc", "sector_available_at_utc"]].notna().all(axis=1)
    data["feature_eligible"] = (
        data["daily_bar_count"].ge(config.min_daily_bars)
        & core_ready
        & benchmark_ready
        & data["price_feed"].astype(str).str.lower().eq(config.required_price_feed)
        & data["adjustment"].astype(str).str.lower().eq(config.required_adjustment)
    )
    eligible_count = data["feature_eligible"].groupby(data["decision_group_id"]).transform("sum")
    data["cross_section_size"] = eligible_count.astype("int32")
    data["cross_section_eligible"] = eligible_count.ge(config.minimum_cross_section)
    data["feature_eligible"] &= data["cross_section_eligible"]
    return data


def _add_exact_labels(
    frame: pd.DataFrame,
    benchmarks: pd.DataFrame,
    config: SwingDatasetConfig,
) -> pd.DataFrame:
    horizon = config.horizon_sessions
    data = frame.sort_values(["ticker", "session_date_et"], kind="stable").copy()
    spy = benchmarks[benchmarks["ticker"].eq(config.broad_benchmark.upper())].sort_values("session_date_et")
    if spy.empty:
        raise DataReadinessError(f"benchmark bars do not contain {config.broad_benchmark}")
    ordered_sessions = list(spy["session_date_et"])
    session_ordinal = {session: idx for idx, session in enumerate(ordered_sessions)}
    data["_session_ordinal"] = data["session_date_et"].map(session_ordinal)
    if bool(data["_session_ordinal"].isna().any()):
        raise DataReadinessError("equity decisions contain sessions absent from SPY")

    grouped = data.groupby("ticker", sort=False)
    data["entry_time_utc"] = grouped["bar_start_utc"].shift(-1)
    data["exit_time_utc"] = grouped["bar_end_utc"].shift(-horizon)
    data["label_available_at_utc"] = grouped["available_at_utc"].shift(-horizon)
    data["entry_session_date_et"] = grouped["session_date_et"].shift(-1)
    data["exit_session_date_et"] = grouped["session_date_et"].shift(-horizon)
    data["entry_price"] = grouped["open"].shift(-1)
    data["exit_price"] = grouped["close"].shift(-horizon)
    expected_entry = data["_session_ordinal"] + 1
    expected_exit = data["_session_ordinal"] + horizon
    actual_entry = data["entry_session_date_et"].map(session_ordinal)
    actual_exit = data["exit_session_date_et"].map(session_ordinal)
    data["label_window_expected"] = expected_exit.lt(len(ordered_sessions))
    data["label_path_exact"] = actual_entry.eq(expected_entry) & actual_exit.eq(expected_exit)

    future_highs = pd.concat([grouped["high"].shift(-offset) for offset in range(1, horizon + 1)], axis=1)
    future_lows = pd.concat([grouped["low"].shift(-offset) for offset in range(1, horizon + 1)], axis=1)
    data[f"future_mfe_{horizon}d"] = future_highs.max(axis=1) / data["entry_price"] - 1.0
    data[f"future_mae_{horizon}d"] = future_lows.min(axis=1) / data["entry_price"] - 1.0
    gross = data["exit_price"] / data["entry_price"] - 1.0
    net = gross - config.round_trip_cost_bps / 10_000.0
    data[f"future_gross_return_{horizon}d"] = gross
    data[swing_net_return_column(horizon)] = net

    benchmark_lookup = benchmarks.set_index(["ticker", "session_date_et"])
    for benchmark_name, benchmark_ticker in (
        ("spy", config.broad_benchmark.upper()),
        ("qqq", config.growth_benchmark.upper()),
    ):
        benchmark_return = _benchmark_label_return(
            data,
            benchmark_lookup,
            pd.Series(benchmark_ticker, index=data.index),
        )
        data[f"future_{benchmark_name}_return_{horizon}d"] = benchmark_return
        data[swing_excess_column(horizon, benchmark_name)] = net - benchmark_return
    sector_return = _benchmark_label_return(data, benchmark_lookup, data["primary_benchmark"])
    data[f"future_sector_return_{horizon}d"] = sector_return
    data[swing_excess_column(horizon, "sector")] = net - sector_return
    data[swing_target_column(horizon)] = (net > 0).astype("Int64")
    invalid_label = ~data["label_path_exact"] | net.isna() | sector_return.isna()
    label_columns = [
        f"future_gross_return_{horizon}d",
        swing_net_return_column(horizon),
        f"future_spy_return_{horizon}d",
        f"future_qqq_return_{horizon}d",
        f"future_sector_return_{horizon}d",
        swing_excess_column(horizon, "spy"),
        swing_excess_column(horizon, "qqq"),
        swing_excess_column(horizon, "sector"),
        f"future_mfe_{horizon}d",
        f"future_mae_{horizon}d",
    ]
    data.loc[invalid_label, label_columns] = np.nan
    data.loc[invalid_label, swing_target_column(horizon)] = pd.NA
    data["target_excess_rank"] = data.groupby("decision_group_id")[swing_excess_column(horizon, "spy")].rank(method="average", pct=True)
    data["label_eligible"] = data["feature_eligible"] & data["label_path_exact"] & data[swing_target_column(horizon)].notna()
    return data.drop(columns="_session_ordinal")


def _benchmark_label_return(
    decisions: pd.DataFrame,
    lookup: pd.DataFrame,
    benchmark_tickers: pd.Series,
) -> pd.Series:
    values = np.full(len(decisions), np.nan, dtype=float)
    for position, (ticker, entry_date, exit_date) in enumerate(
        zip(
            benchmark_tickers.astype(str).str.upper(),
            decisions["entry_session_date_et"],
            decisions["exit_session_date_et"],
            strict=True,
        )
    ):
        if pd.isna(entry_date) or pd.isna(exit_date):
            continue
        try:
            entry_open = float(lookup.loc[(ticker, entry_date), "open"])
            exit_close = float(lookup.loc[(ticker, exit_date), "close"])
        except (KeyError, TypeError, ValueError):
            continue
        values[position] = exit_close / entry_open - 1.0
    return pd.Series(values, index=decisions.index, dtype="float64")


def _rsi(close: pd.Series, window: int) -> pd.Series:
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(window, min_periods=window).mean()
    loss = (-delta.clip(upper=0)).rolling(window, min_periods=window).mean()
    relative = gain / loss.replace(0, np.nan)
    result = 100.0 - 100.0 / (1.0 + relative)
    return result.where(loss.ne(0), 100.0)


def _window_sum(values: np.ndarray, start: np.ndarray, end: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate(([0.0], np.cumsum(values, dtype=float)))
    return np.asarray(cumulative[end] - cumulative[start], dtype=float)


def _row_timestamp_max(frame: pd.DataFrame, columns: Sequence[str]) -> pd.Series:
    available = [column for column in columns if column in frame.columns]
    if not available:
        return pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
    parsed = pd.concat([pd.to_datetime(frame[column], utc=True, errors="coerce") for column in available], axis=1)
    return pd.to_datetime(parsed.max(axis=1), utc=True)


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    def parse(value: object) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            return pd.NaT
        return timestamp.tz_convert("UTC")

    parsed = pd.to_datetime(values.map(parse), utc=True)
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid or timezone-naive timestamps")
    return parsed


def _require_columns(frame: pd.DataFrame, required: set[str], name: str) -> None:
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"{name} missing columns: {', '.join(missing)}")
