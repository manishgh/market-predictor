from __future__ import annotations

import re
from bisect import bisect_left
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Protocol, cast
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd

from market_predictor.config import Settings
from market_predictor.price import fetch_daily_prices, fetch_hourly_prices
from market_predictor.sources.sec import SecSource
from market_predictor.sources.seeking_alpha import SeekingAlphaQuantCsvSource

NY_TZ = ZoneInfo("America/New_York")
MARKET_CLOSE = time(16, 0)

EVENT_KEYWORDS = {
    "earnings": ["earnings", "eps", "revenue", "quarter", "q1", "q2", "q3", "q4", "guidance"],
    "analyst": ["upgrade", "downgrade", "initiated", "price target", "analyst", "rating"],
    "guidance": ["guidance", "raises outlook", "cuts outlook", "lowers outlook", "forecast"],
    "ma": ["acquire", "acquisition", "merger", "takeover", "buyout", "strategic alternatives"],
    "fda": ["fda", "phase 1", "phase 2", "phase 3", "trial", "approval", "complete response letter", "pdufa"],
    "contract": ["contract", "award", "partnership", "deal", "supplier", "order"],
    "sec": ["sec filing", "sec 8-k", "sec 10-q", "sec 10-k", "form 4"],
    "offering": ["offering", "shelf", "atm", "at-the-market", "s-3", "s-1", "424b5", "424b3", "prospectus"],
    "insider": ["form 4", "insider", "beneficial ownership", "13d", "13g"],
}


class SentimentScorer(Protocol):
    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame: ...


def events_to_frame(events: list[dict[str, object]]) -> pd.DataFrame:
    if not events:
        return pd.DataFrame(columns=["ticker", "timestamp", "source", "title", "url", "summary", "text"])
    frame = pd.DataFrame(events)
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].map(feature_date_for_timestamp)
    frame["text"] = frame["text"].fillna(frame["summary"]).fillna(frame["title"]).fillna("")
    for col in ["engagement_score", "engagement_comments", "engagement_upvote_ratio"]:
        if col not in frame.columns:
            frame[col] = 0.0
        frame[col] = frame[col].fillna(0.0)
    return frame


def source_family_for_source(value: object) -> str:
    raw = str(value or "").strip().lower()
    if raw.startswith("seeking_alpha"):
        return "seeking_alpha"
    if raw.startswith("alpaca"):
        return "alpaca"
    if raw.startswith("reddit"):
        return "reddit"
    if raw.startswith("sec"):
        return "sec"
    if raw.startswith("finviz"):
        return "finviz"
    return raw.split(":", 1)[0] if raw else "unknown"


def feature_date_for_timestamp(timestamp: pd.Timestamp | datetime) -> date:
    """Assign a public event to the trading feature date that could use it.

    Events published after the regular-session close roll to the next weekday.
    This is a conservative approximation until a full exchange calendar is added.
    """
    ts = pd.Timestamp(timestamp)
    if ts.tzinfo is None:
        ts = ts.tz_localize(UTC)
    local = ts.tz_convert(NY_TZ)
    assigned = cast(date, local.date())
    if local.time() > MARKET_CLOSE:
        assigned = _next_weekday(assigned)
    elif local.weekday() >= 5:
        assigned = _next_weekday(assigned)
    return assigned


def _next_weekday(value: date) -> date:
    value = value + timedelta(days=1)
    while value.weekday() >= 5:
        value = value + timedelta(days=1)
    return value


def align_events_to_trading_dates(events: pd.DataFrame, trading_dates: pd.Series | list[date]) -> pd.DataFrame:
    """Assign each event to the first actual price candle date it could affect."""
    if events.empty:
        return events
    dates: list[date] = sorted({cast(date, pd.Timestamp(value).date()) for value in trading_dates if pd.notna(value)})
    if not dates:
        output = events.copy()
        output["date"] = pd.NaT
        return output.iloc[0:0].copy()
    output = events.copy()
    output["_candidate_date"] = output["timestamp"].map(feature_date_for_timestamp)

    def next_trading_date(candidate: date) -> date | None:
        idx = bisect_left(dates, candidate)
        if idx >= len(dates):
            return None
        return dates[idx]

    output["date"] = output["_candidate_date"].map(next_trading_date)
    output = output.drop(columns=["_candidate_date"])
    return output.dropna(subset=["date"]).reset_index(drop=True)


def add_finbert(events: pd.DataFrame, model_name: str) -> pd.DataFrame:
    if events.empty:
        return events
    from market_predictor.sentiment import FinbertScorer

    scorer = FinbertScorer(model_name)
    return add_finbert_with_scorer(events, scorer)


def add_finbert_with_scorer(
    events: pd.DataFrame,
    scorer: SentimentScorer,
    *,
    batch_size: int = 16,
    text_column: str = "text",
) -> pd.DataFrame:
    if events.empty:
        return events
    if text_column not in events.columns:
        raise ValueError(f"FinBERT input column {text_column!r} is missing.")
    scores = scorer.score_texts(events[text_column].astype(str).tolist(), batch_size=batch_size)
    existing_score_cols = [col for col in scores.columns if col in events.columns]
    events = events.drop(columns=existing_score_cols, errors="ignore")
    return pd.concat([events.reset_index(drop=True), scores.reset_index(drop=True)], axis=1)


def build_daily_dataset(
    ticker: str,
    events_path: Path,
    settings: Settings,
    *,
    horizon_days: int = 5,
    seeking_alpha_path: Path | None = None,
    market_context_path: Path | None = None,
    end: datetime | None = None,
) -> pd.DataFrame:
    events = pd.read_parquet(events_path)
    if events.empty:
        raise ValueError(f"No events found in {events_path}. Collect a wider window or check API credentials.")
    events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
    if "sentiment_numeric" not in events.columns:
        events["sentiment_numeric"] = 0.0
    for col in ["title", "summary", "text", "source"]:
        if col not in events.columns:
            events[col] = ""

    start = events["timestamp"].min().to_pydatetime()
    price_start = start - timedelta(days=420)
    end = end or datetime.now(UTC)
    prices = fetch_daily_prices(ticker, price_start, end, settings)
    if prices.empty:
        raise ValueError(f"No price data returned for {ticker}.")

    prices = prices.sort_values("date").copy()
    prices = add_price_features(prices)
    prices = add_benchmark_features(ticker, prices, settings, price_start, end)
    prices = add_forward_labels(prices, horizon_days=horizon_days)
    events = align_events_to_trading_dates(events, prices["date"])
    if events.empty:
        raise ValueError(f"No events in {events_path} matched an available trading candle for {ticker}.")

    events["source_family"] = events["source"].map(source_family_for_source)
    events = add_event_taxonomy(events)
    text_daily = events.groupby("date").agg(
        news_count=("title", "count"),
        sentiment_mean=("sentiment_numeric", "mean"),
        sentiment_min=("sentiment_numeric", "min"),
        sentiment_max=("sentiment_numeric", "max"),
        sentiment_pos_frac=("sentiment_numeric", lambda values: float((values > 0.15).mean())),
        sentiment_neg_frac=("sentiment_numeric", lambda values: float((values < -0.15).mean())),
    )
    family_counts = events.pivot_table(
        index="date",
        columns="source_family",
        values="title",
        aggfunc="count",
        fill_value=0,
    ).add_prefix("source_count_")
    reddit_events = events[events["source_family"] == "reddit"]
    if reddit_events.empty:
        reddit_daily = pd.DataFrame(index=text_daily.index)
    else:
        reddit_daily = reddit_events.groupby("date").agg(
            reddit_mentions=("title", "count"),
            reddit_sentiment_mean=("sentiment_numeric", "mean"),
            reddit_score_sum=("engagement_score", "sum"),
            reddit_comments_sum=("engagement_comments", "sum"),
            reddit_upvote_ratio_mean=("engagement_upvote_ratio", "mean"),
        )
    event_daily = _event_daily_features(events)
    text_daily = text_daily.join(family_counts, how="left").join(reddit_daily, how="left").join(event_daily, how="left")
    text_daily = _add_text_rolling_features(text_daily.reset_index())
    reaction_daily = build_event_reaction_features(events, prices, ticker, start, end, settings)
    if not reaction_daily.empty:
        text_daily = text_daily.merge(reaction_daily, on="date", how="left")

    dataset = prices.merge(text_daily, on="date", how="left")
    market_context = load_market_context_daily(market_context_path)
    if not market_context.empty:
        dataset = dataset.merge(market_context, on="date", how="left")
    fill_zero_cols = [
        "news_count",
        "sentiment_mean",
        "sentiment_min",
        "sentiment_max",
        "sentiment_pos_frac",
        "sentiment_neg_frac",
        "news_count_z30",
        "sentiment_momentum_5d",
        "source_count_alpaca",
        "source_count_reddit",
        "source_count_seeking_alpha",
        "source_count_sec",
        "source_count_finviz",
        "reddit_mentions",
        "reddit_velocity_7d",
        "reddit_newly_trending",
        "reddit_sentiment_mean",
        "reddit_score_sum",
        "reddit_comments_sum",
        "reddit_upvote_ratio_mean",
        "event_count",
        "event_reaction_2h_mean",
        "event_reaction_2h_abs_max",
        "event_reaction_volume_sum",
        "premarket_gap_mean",
        "premarket_day_return_mean",
        "intraday_reaction_2h_mean",
        "intraday_to_close_mean",
        "afterhours_next_open_gap_mean",
        "afterhours_next_day_return_mean",
        "market_context_news_count",
        "market_context_sentiment_mean",
        "market_context_sentiment_min",
        "market_context_sentiment_max",
        "market_context_sentiment_neg_frac",
        "market_context_sentiment_pos_frac",
        "market_context_news_count_z30",
        "market_context_sentiment_momentum_5d",
    ]
    fill_zero_cols.extend([f"event_{name}_count" for name in EVENT_KEYWORDS])
    for col in fill_zero_cols:
        if col not in dataset.columns:
            dataset[col] = 0.0
        dataset[col] = dataset[col].fillna(0.0)
    dataset["has_news"] = (dataset["news_count"] > 0).astype(int)

    if seeking_alpha_path:
        sa = SeekingAlphaQuantCsvSource(seeking_alpha_path).load(ticker)
        if not sa.empty:
            sa["date"] = sa["timestamp"].dt.date
            sa_daily = sa.sort_values("timestamp").drop_duplicates("date", keep="last").drop(columns=["timestamp"])
            dataset = dataset.merge(sa_daily, on=["date"], how="left")
            quant_cols = [
                "quant_rating",
                "valuation",
                "growth",
                "profitability",
                "momentum",
                "eps_revision",
                "eps_actual",
                "eps_estimate",
                "revenue_actual",
                "revenue_estimate",
                "earnings_date",
                "fiscal_period",
            ]
            for col in quant_cols:
                if col not in dataset.columns:
                    dataset[col] = np.nan
            dataset[quant_cols] = dataset[quant_cols].ffill()
            for col in [
                "quant_rating",
                "valuation",
                "growth",
                "profitability",
                "momentum",
                "eps_revision",
            ]:
                dataset[f"{col}_score"] = dataset[col].map(_rating_to_numeric)
            dataset["eps_surprise"] = pd.to_numeric(dataset["eps_actual"], errors="coerce") - pd.to_numeric(
                dataset["eps_estimate"],
                errors="coerce",
            )
            dataset["revenue_surprise"] = pd.to_numeric(dataset["revenue_actual"], errors="coerce") - pd.to_numeric(
                dataset["revenue_estimate"],
                errors="coerce",
            )
            earnings_dates = pd.to_datetime(dataset["earnings_date"], errors="coerce", utc=True).dt.date
            dataset["days_to_earnings"] = [
                (earnings_date - current_date).days if pd.notna(earnings_date) else np.nan
                for earnings_date, current_date in zip(earnings_dates, dataset["date"], strict=False)
            ]

    try:
        sec = SecSource(settings).latest_company_facts(ticker)
        dataset["sec_eps_diluted_recent"] = sec.eps_diluted_recent
        dataset["sec_eps_basic_recent"] = sec.eps_basic_recent
        dataset["sec_revenue_recent"] = sec.revenue_recent
        dataset["sec_net_income_recent"] = sec.net_income_recent
    except Exception:
        dataset["sec_eps_diluted_recent"] = np.nan
        dataset["sec_eps_basic_recent"] = np.nan
        dataset["sec_revenue_recent"] = np.nan
        dataset["sec_net_income_recent"] = np.nan

    dataset = add_interaction_features(dataset)
    dataset = dataset.replace([np.inf, -np.inf], np.nan)
    return dataset.dropna(subset=["close"]).reset_index(drop=True)


def load_market_context_daily(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    frame = pd.read_parquet(path)
    if frame.empty:
        return pd.DataFrame()
    frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
    frame["date"] = frame["timestamp"].map(feature_date_for_timestamp)
    if "sentiment_numeric" not in frame.columns:
        frame["sentiment_numeric"] = 0.0
    daily = frame.groupby("date").agg(
        market_context_news_count=("title", "count"),
        market_context_sentiment_mean=("sentiment_numeric", "mean"),
        market_context_sentiment_min=("sentiment_numeric", "min"),
        market_context_sentiment_max=("sentiment_numeric", "max"),
        market_context_sentiment_neg_frac=("sentiment_numeric", lambda values: float((values < -0.15).mean())),
        market_context_sentiment_pos_frac=("sentiment_numeric", lambda values: float((values > 0.15).mean())),
    )
    daily = daily.sort_index().reset_index()
    rolling_mean = daily["market_context_news_count"].rolling(30, min_periods=5).mean()
    rolling_std = daily["market_context_news_count"].rolling(30, min_periods=5).std()
    daily["market_context_news_count_z30"] = (
        (daily["market_context_news_count"] - rolling_mean) / rolling_std.replace(0, np.nan)
    ).fillna(0.0)
    daily["market_context_sentiment_momentum_5d"] = (
        daily["market_context_sentiment_mean"] - daily["market_context_sentiment_mean"].rolling(5, min_periods=1).mean()
    ).fillna(0.0)
    return daily


def add_price_features(prices: pd.DataFrame) -> pd.DataFrame:
    frame = prices.copy()
    for col in ["open", "high", "low", "close", "volume"]:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    frame["return_1d"] = frame["close"].pct_change()
    for window in [5, 10, 20]:
        frame[f"return_{window}d_past"] = frame["close"].pct_change(window)
    daily_return = frame["close"].pct_change()
    for window in [10, 20, 60]:
        frame[f"realized_vol_{window}d"] = daily_return.rolling(window).std()
    prev_close = frame["close"].shift(1)
    true_range = pd.concat(
        [
            frame["high"] - frame["low"],
            (frame["high"] - prev_close).abs(),
            (frame["low"] - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    frame["atr_14"] = true_range.rolling(14).mean()
    frame["atr_pct_14"] = frame["atr_14"] / frame["close"]
    frame["rsi_14"] = _rsi(frame["close"], 14)
    ema12 = frame["close"].ewm(span=12, adjust=False).mean()
    ema26 = frame["close"].ewm(span=26, adjust=False).mean()
    frame["macd"] = ema12 - ema26
    frame["macd_signal"] = frame["macd"].ewm(span=9, adjust=False).mean()
    frame["macd_signal_diff"] = frame["macd"] - frame["macd_signal"]
    for window in [10, 20, 50]:
        ema = frame["close"].ewm(span=window, adjust=False).mean()
        frame[f"ema_{window}"] = ema
        frame[f"dist_ema_{window}"] = frame["close"] / ema - 1.0
    for window in [20, 50]:
        sma = frame["close"].rolling(window).mean()
        frame[f"sma_{window}"] = sma
        frame[f"dist_sma_{window}"] = frame["close"] / sma - 1.0
    frame["sma20_gt_sma50"] = (frame["sma_20"] > frame["sma_50"]).astype(int)
    frame["volume_z20"] = (frame["volume"] - frame["volume"].rolling(20).mean()) / frame["volume"].rolling(20).std()
    frame["gap_pct"] = frame["open"] / prev_close - 1.0
    rolling_high = frame["close"].rolling(252, min_periods=20).max()
    rolling_low = frame["close"].rolling(252, min_periods=20).min()
    frame["pct_from_52w_high"] = frame["close"] / rolling_high - 1.0
    frame["pct_from_52w_low"] = frame["close"] / rolling_low - 1.0
    return frame


def add_benchmark_features(
    ticker: str,
    prices: pd.DataFrame,
    settings: Settings,
    start: datetime,
    end: datetime,
) -> pd.DataFrame:
    frame = prices.copy()
    market_ticker = settings.market_benchmark_ticker
    sector_ticker = settings.sector_benchmark_for_ticker(ticker)
    frame["sector_name"] = settings.sector_for_ticker(ticker) or "unknown"
    frame["market_benchmark"] = market_ticker
    frame["sector_benchmark"] = sector_ticker

    frame = _merge_benchmark(frame, market_ticker, start, end, settings, "spy")
    if sector_ticker == market_ticker:
        for col in [
            "return_1d",
            "return_5d_past",
            "return_10d_past",
            "return_20d_past",
            "volume_z20",
            "gap_pct",
        ]:
            frame[f"sector_{col}"] = frame.get(f"spy_{col}", 0.0)
    else:
        frame = _merge_benchmark(frame, sector_ticker, start, end, settings, "sector")

    frame["rel_return_1d_vs_spy"] = _numeric(frame, "return_1d") - _numeric(frame, "spy_return_1d")
    frame["rel_return_5d_vs_spy"] = _numeric(frame, "return_5d_past") - _numeric(frame, "spy_return_5d_past")
    frame["rel_return_10d_vs_spy"] = _numeric(frame, "return_10d_past") - _numeric(frame, "spy_return_10d_past")
    frame["rel_return_20d_vs_spy"] = _numeric(frame, "return_20d_past") - _numeric(frame, "spy_return_20d_past")
    frame["rel_return_1d_vs_sector"] = _numeric(frame, "return_1d") - _numeric(frame, "sector_return_1d")
    frame["rel_return_5d_vs_sector"] = _numeric(frame, "return_5d_past") - _numeric(frame, "sector_return_5d_past")
    frame["rel_return_10d_vs_sector"] = _numeric(frame, "return_10d_past") - _numeric(frame, "sector_return_10d_past")
    frame["rel_return_20d_vs_sector"] = _numeric(frame, "return_20d_past") - _numeric(frame, "sector_return_20d_past")
    return frame


def _merge_benchmark(
    frame: pd.DataFrame,
    benchmark_ticker: str,
    start: datetime,
    end: datetime,
    settings: Settings,
    prefix: str,
) -> pd.DataFrame:
    cols = [
        "return_1d",
        "return_5d_past",
        "return_10d_past",
        "return_20d_past",
        "realized_vol_20d",
        "volume_z20",
        "gap_pct",
    ]
    try:
        benchmark = fetch_daily_prices(benchmark_ticker, start, end, settings)
        if benchmark.empty:
            raise ValueError(f"No benchmark prices for {benchmark_ticker}")
        benchmark = add_price_features(benchmark.sort_values("date"))
        keep = ["date", *[col for col in cols if col in benchmark.columns]]
        benchmark = benchmark[keep].rename(columns={col: f"{prefix}_{col}" for col in keep if col != "date"})
        merged = frame.merge(benchmark, on="date", how="left")
    except Exception:
        merged = frame.copy()
        for col in cols:
            merged[f"{prefix}_{col}"] = 0.0
    for col in cols:
        name = f"{prefix}_{col}"
        if name not in merged.columns:
            merged[name] = 0.0
        merged[name] = pd.to_numeric(merged[name], errors="coerce").fillna(0.0)
    return merged


def _numeric(frame: pd.DataFrame, column: str) -> pd.Series:
    if column not in frame.columns:
        return pd.Series(0.0, index=frame.index, dtype="float")
    return pd.to_numeric(frame[column], errors="coerce").fillna(0.0)


def _rsi(series: pd.Series, window: int) -> pd.Series:
    delta = series.diff()
    gain = delta.clip(lower=0).rolling(window).mean()
    loss = (-delta.clip(upper=0)).rolling(window).mean()
    rs = gain / loss.replace(0, np.nan)
    return 100.0 - (100.0 / (1.0 + rs))


def add_forward_labels(
    prices: pd.DataFrame,
    *,
    horizon_days: int,
    atr_threshold_mult: float = 0.5,
    take_profit_atr: float = 1.0,
    stop_loss_atr: float = 1.0,
) -> pd.DataFrame:
    frame = prices.copy()
    entry = frame["open"].shift(-1)
    exit_close = frame["close"].shift(-horizon_days)
    forward_return = exit_close / entry - 1.0
    threshold = frame["atr_pct_14"].fillna(frame["return_1d"].rolling(20).std()) * atr_threshold_mult
    frame[f"entry_next_open_{horizon_days}d"] = entry
    frame[f"future_return_{horizon_days}d"] = forward_return
    frame[f"target_up_{horizon_days}d"] = (forward_return > 0).astype("float")
    frame[f"target_bucket_{horizon_days}d"] = np.select(
        [forward_return > threshold, forward_return < -threshold],
        [1, -1],
        default=0,
    )
    frame[f"target_favorable_before_stop_{horizon_days}d"] = _favorable_before_stop(
        frame,
        horizon_days=horizon_days,
        take_profit_atr=take_profit_atr,
        stop_loss_atr=stop_loss_atr,
    )
    label_tail = frame.index[-horizon_days:]
    frame.loc[label_tail, [f"target_up_{horizon_days}d", f"target_bucket_{horizon_days}d"]] = np.nan
    return frame


def _favorable_before_stop(
    frame: pd.DataFrame,
    *,
    horizon_days: int,
    take_profit_atr: float,
    stop_loss_atr: float,
) -> pd.Series:
    outcomes: list[float] = []
    for idx in range(len(frame)):
        entry_idx = idx + 1
        end_idx = min(idx + horizon_days, len(frame) - 1)
        if entry_idx > end_idx:
            outcomes.append(np.nan)
            continue
        entry = frame.iloc[entry_idx]["open"]
        atr = frame.iloc[idx]["atr_14"]
        if pd.isna(entry) or pd.isna(atr) or atr <= 0:
            outcomes.append(np.nan)
            continue
        target = entry + take_profit_atr * atr
        stop = entry - stop_loss_atr * atr
        result = 0.0
        for row_idx in range(entry_idx, end_idx + 1):
            row = frame.iloc[row_idx]
            hit_stop = row["low"] <= stop
            hit_target = row["high"] >= target
            if hit_stop and hit_target:
                result = 0.0
                break
            if hit_target:
                result = 1.0
                break
            if hit_stop:
                result = 0.0
                break
        outcomes.append(result)
    return pd.Series(outcomes, index=frame.index, dtype="float")


def add_event_taxonomy(events: pd.DataFrame) -> pd.DataFrame:
    frame = events.copy()
    text = (frame["title"].fillna("") + " " + frame["summary"].fillna("") + " " + frame["text"].fillna("")).str.lower()
    for name, keywords in EVENT_KEYWORDS.items():
        pattern = "|".join(re.escape(keyword) for keyword in keywords)
        frame[f"event_{name}"] = text.str.contains(pattern, regex=True, na=False).astype(int)
    local_times = frame["timestamp"].dt.tz_convert(NY_TZ)
    frame["event_time_bucket"] = np.select(
        [
            local_times.dt.time < time(9, 30),
            (local_times.dt.time >= time(9, 30)) & (local_times.dt.time <= MARKET_CLOSE),
        ],
        ["pre_market", "intraday"],
        default="after_hours",
    )
    return frame


def _event_daily_features(events: pd.DataFrame) -> pd.DataFrame:
    rows: dict[str, tuple[str, str]] = {"event_count": ("title", "count")}
    for name in EVENT_KEYWORDS:
        rows[f"event_{name}_count"] = (f"event_{name}", "sum")
    event_daily = events.groupby("date").agg(**rows)
    for bucket in ["pre_market", "intraday", "after_hours"]:
        event_daily[f"event_time_{bucket}_count"] = (
            events.assign(_bucket=(events["event_time_bucket"] == bucket).astype(int)).groupby("date")["_bucket"].sum()
        )
    return event_daily


def _add_text_rolling_features(text_daily: pd.DataFrame) -> pd.DataFrame:
    frame = text_daily.sort_values("date").copy()
    frame["news_count_z30"] = (
        frame["news_count"] - frame["news_count"].rolling(30, min_periods=5).mean()
    ) / frame["news_count"].rolling(30, min_periods=5).std()
    frame["sentiment_momentum_5d"] = frame["sentiment_mean"] - frame["sentiment_mean"].rolling(5, min_periods=2).mean()
    if "reddit_mentions" not in frame.columns:
        frame["reddit_mentions"] = 0.0
    baseline = frame["reddit_mentions"].rolling(7, min_periods=2).mean().shift(1)
    frame["reddit_velocity_7d"] = frame["reddit_mentions"] / baseline.replace(0, np.nan)
    frame["reddit_newly_trending"] = ((frame["reddit_mentions"] >= 3) & (baseline.fillna(0) < 1)).astype(int)
    return frame


def add_interaction_features(dataset: pd.DataFrame) -> pd.DataFrame:
    frame = dataset.copy()
    reddit_velocity = _feature_series(frame, "reddit_velocity_7d")
    volume_z = _feature_series(frame, "volume_z20")
    sentiment = _feature_series(frame, "sentiment_mean")
    news_count_z = _feature_series(frame, "news_count_z30")
    event_count = _feature_series(frame, "event_count")
    earnings_count = _feature_series(frame, "event_earnings_count")
    eps_surprise = _feature_series(frame, "eps_surprise")
    reaction_2h = _feature_series(frame, "event_reaction_2h_mean")
    premarket_gap = _feature_series(frame, "premarket_gap_mean")
    eps_revision = _feature_series(frame, "eps_revision_score")
    days_to_earnings = _feature_series(frame, "days_to_earnings", default=np.nan)
    frame["buzz_spike_x_volume_z"] = reddit_velocity * volume_z.clip(lower=0)
    frame["sentiment_x_news_attention"] = sentiment * news_count_z
    frame["earnings_x_eps_surprise"] = earnings_count * eps_surprise
    frame["catalyst_x_volume_z"] = event_count * volume_z.clip(lower=0)
    frame["reaction_x_sentiment"] = reaction_2h * sentiment
    frame["premarket_gap_x_sentiment"] = premarket_gap * sentiment
    frame["revision_x_days_to_earnings"] = eps_revision * (1.0 / (1.0 + days_to_earnings.abs()))
    return frame


def _feature_series(frame: pd.DataFrame, column: str, default: float = 0.0) -> pd.Series:
    if column in frame.columns:
        return pd.to_numeric(frame[column], errors="coerce").fillna(default)
    return pd.Series(default, index=frame.index, dtype="float")


def build_event_reaction_features(
    events: pd.DataFrame,
    daily_prices: pd.DataFrame,
    ticker: str,
    start: datetime,
    end: datetime,
    settings: Settings,
) -> pd.DataFrame:
    if events.empty:
        return pd.DataFrame()
    hourly = fetch_hourly_prices(ticker, start, end, settings)
    if not hourly.empty:
        hourly = hourly.sort_values("timestamp").copy()
        for col in ["open", "high", "low", "close", "volume"]:
            hourly[col] = pd.to_numeric(hourly[col], errors="coerce")
    daily = daily_prices.sort_values("date").copy()
    daily["prev_close"] = daily["close"].shift(1)
    daily_by_date = daily.set_index("date")
    rows: list[dict[str, object]] = []
    for event in events.sort_values("timestamp").itertuples(index=False):
        event_date = event.date
        bucket = getattr(event, "event_time_bucket", "")
        day = daily_by_date.loc[event_date] if event_date in daily_by_date.index else None
        row: dict[str, object] = {
            "date": event_date,
            "event_reaction_2h": np.nan,
            "event_reaction_abs_2h": np.nan,
            "event_reaction_volume": np.nan,
            "premarket_gap": np.nan,
            "premarket_day_return": np.nan,
            "intraday_reaction_2h": np.nan,
            "intraday_to_close": np.nan,
            "afterhours_next_open_gap": np.nan,
            "afterhours_next_day_return": np.nan,
        }
        if day is not None and pd.notna(day.get("open")) and pd.notna(day.get("close")):
            open_price = float(day["open"])
            close_price = float(day["close"])
            prev_close = float(day["prev_close"]) if pd.notna(day.get("prev_close")) else np.nan
            if bucket == "pre_market":
                row["premarket_gap"] = open_price / prev_close - 1.0 if prev_close and not pd.isna(prev_close) else np.nan
                row["premarket_day_return"] = close_price / open_price - 1.0
            elif bucket == "after_hours":
                row["afterhours_next_open_gap"] = (
                    open_price / prev_close - 1.0 if prev_close and not pd.isna(prev_close) else np.nan
                )
                row["afterhours_next_day_return"] = close_price / open_price - 1.0
        if not hourly.empty:
            event_ts = pd.Timestamp(event.timestamp)
            if event_ts.tzinfo is None:
                event_ts = event_ts.tz_localize(UTC)
            window = hourly[(hourly["timestamp"] >= event_ts) & (hourly["timestamp"] <= event_ts + pd.Timedelta(hours=2))]
            if not window.empty:
                reaction = float(window.iloc[-1]["close"]) / float(window.iloc[0]["open"]) - 1.0
                row["event_reaction_2h"] = reaction
                row["event_reaction_abs_2h"] = abs(reaction)
                row["event_reaction_volume"] = float(window["volume"].sum())
                if bucket == "intraday":
                    row["intraday_reaction_2h"] = reaction
                    if day is not None and pd.notna(day.get("close")):
                        row["intraday_to_close"] = float(day["close"]) / float(window.iloc[0]["open"]) - 1.0
        rows.append(row)
    frame = pd.DataFrame(rows)
    if frame.empty:
        return frame
    return frame.groupby("date", as_index=False).agg(
        event_reaction_2h_mean=("event_reaction_2h", "mean"),
        event_reaction_2h_abs_max=("event_reaction_abs_2h", "max"),
        event_reaction_volume_sum=("event_reaction_volume", "sum"),
        premarket_gap_mean=("premarket_gap", "mean"),
        premarket_day_return_mean=("premarket_day_return", "mean"),
        intraday_reaction_2h_mean=("intraday_reaction_2h", "mean"),
        intraday_to_close_mean=("intraday_to_close", "mean"),
        afterhours_next_open_gap_mean=("afterhours_next_open_gap", "mean"),
        afterhours_next_day_return_mean=("afterhours_next_day_return", "mean"),
    )


def _rating_to_numeric(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    text = str(value).strip().lower()
    if not text:
        return None
    try:
        return float(text)
    except ValueError:
        pass
    named = {
        "strong buy": 5.0,
        "buy": 4.0,
        "hold": 3.0,
        "sell": 2.0,
        "strong sell": 1.0,
        "very bullish": 5.0,
        "bullish": 4.0,
        "neutral": 3.0,
        "bearish": 2.0,
        "very bearish": 1.0,
    }
    if text in named:
        return named[text]
    letter = text.upper()
    letter_scores = {
        "A+": 5.0,
        "A": 4.8,
        "A-": 4.6,
        "B+": 4.2,
        "B": 4.0,
        "B-": 3.8,
        "C+": 3.2,
        "C": 3.0,
        "C-": 2.8,
        "D+": 2.2,
        "D": 2.0,
        "D-": 1.8,
        "F": 1.0,
    }
    return letter_scores.get(letter)
