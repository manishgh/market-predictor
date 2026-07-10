from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, time, timedelta, timezone
import json
from pathlib import Path
import re
import time as time_module
from urllib.parse import urlsplit, urlunsplit

import pandas as pd
import requests
import typer
from rich.console import Console

from market_predictor.alerts import AlertConfig, backtest_indicator_alerts, generate_indicator_alerts
from market_predictor.config import get_settings
from market_predictor.data_quality import sanitize_events_frame
from market_predictor.entry_exit import (
    EntryExitLabelConfig,
    build_entry_exit_dataset,
    score_entry_exit_frame,
    train_entry_exit_model,
)
from market_predictor.features import (
    add_event_taxonomy,
    add_finbert,
    add_finbert_with_scorer,
    add_price_features,
    align_events_to_trading_dates,
    build_daily_dataset,
    build_event_swing_dataset,
    events_to_frame,
    feature_date_for_timestamp,
    source_family_for_source,
)
from market_predictor.global_context import build_sector_theme_monitor, score_flashpoints
from market_predictor.intraday_confirmation import build_intraday_decision_report
from market_predictor.intraday_enrichment import build_enriched_intraday_dataset
from market_predictor.intraday_universe import build_intraday_candidate_universe
from market_predictor.model import (
    DEFAULT_FEATURES,
    heuristic_watch_score,
    predict_event_swing_frame,
    predict_latest,
    train_direction_model,
    train_event_swing_model,
    train_event_swing_model_with_metrics,
)
from market_predictor.azure_store import AzureBlobStore
from market_predictor.price import fetch_daily_prices
from market_predictor.price import fetch_intraday_prices
from market_predictor.promotion_audit import (
    ProfitabilityAuditConfig,
    build_catalyst_news_audit,
    build_market_regime_audit,
    build_walk_forward_profitability_audit,
    read_audit_record,
)
from market_predictor.registry import promote_model_manifest
from market_predictor.sources import AlpacaSource, FinvizSource, GdeltSource, RedditSource, SeekingAlphaRapidApiSource
from market_predictor.sources.gdelt import DEFAULT_GDELT_CONTEXT_QUERIES
from market_predictor.sources.sec import SecSource
from market_predictor.volatile import (
    VolatileLabelConfig,
    build_volatile_dataset,
    load_volatile_universe,
    score_volatile_frame,
    train_volatile_model,
)

app = typer.Typer(help="Collect news, build features, and train next-week market direction models.")
console = Console()
DEFAULT_MARKET_CONTEXT_PATH = Path("data/external/market_context/market_context_events_scored.parquet")


@app.command("serve-api")
def serve_api(
    host: str = typer.Option("127.0.0.1", help="API bind host."),
    port: int = typer.Option(8000, help="API bind port."),
    reload: bool = typer.Option(False, help="Enable uvicorn reload for local development."),
) -> None:
    """Serve the typed prediction API for swing, intraday, and unified views."""
    try:
        import uvicorn
    except ImportError as exc:
        raise typer.BadParameter("uvicorn is not installed. Run `python -m pip install -e .` first.") from exc
    uvicorn.run("market_predictor.api:app", host=host, port=port, reload=reload)


def _daily_training_columns(frame: pd.DataFrame, horizon_days: int) -> list[str]:
    keep = {
        "date",
        "ticker",
        "open",
        "high",
        "low",
        "close",
        "volume",
        f"entry_next_open_{horizon_days}d",
        f"future_return_{horizon_days}d",
        f"target_up_{horizon_days}d",
        f"target_bucket_{horizon_days}d",
        f"target_favorable_before_stop_{horizon_days}d",
    }
    keep.update(DEFAULT_FEATURES)
    return [column for column in frame.columns if column in keep]


def _parse_tickers(tickers: str | None, fallback: list[str]) -> list[str]:
    if tickers:
        values = [item.strip().upper() for item in tickers.replace(";", ",").split(",")]
        return [item for item in dict.fromkeys(values) if item]
    return fallback


def _parse_path_list(value: str | None) -> list[Path]:
    if not value:
        return []
    return [Path(item.strip()) for item in value.replace(";", ",").split(",") if item.strip()]


def _parse_end_date(value: str | None) -> datetime | None:
    if not value:
        return None
    parsed = datetime.combine(date.fromisoformat(value), time(23, 59, 59), tzinfo=timezone.utc)
    return parsed


def _filter_events_until(frame: pd.DataFrame, end: datetime | None) -> pd.DataFrame:
    if end is None or frame.empty or "timestamp" not in frame.columns:
        return frame
    output = frame.copy()
    output["timestamp"] = pd.to_datetime(output["timestamp"], utc=True)
    return output[output["timestamp"] <= pd.Timestamp(end)].reset_index(drop=True)


def collect_events_for_ticker(
    ticker: str,
    days: int,
    *,
    end: datetime | None = None,
    no_reddit: bool = False,
    no_finviz: bool = False,
    no_seeking_alpha: bool = False,
    no_sec: bool = False,
    score: bool = True,
) -> tuple[pd.DataFrame, list[str]]:
    settings = get_settings()
    end = end or datetime.now(timezone.utc)
    start = end - timedelta(days=days)
    events = []
    errors: list[str] = []

    if settings.has_alpaca:
        try:
            console.print(f"{ticker}: collecting Alpaca premium news...")
            events.extend(event.to_record() for event in AlpacaSource(settings).fetch_news(ticker, start, end=end, limit=50))
        except Exception as exc:
            errors.append(f"alpaca:{exc}")
            console.print(f"[yellow]{ticker}: Alpaca collection failed: {exc}[/yellow]")
    else:
        console.print(f"[yellow]{ticker}: skipping Alpaca because keys are not configured.[/yellow]")

    if not no_reddit and settings.has_reddit:
        try:
            console.print(f"{ticker}: collecting Reddit mentions and comments...")
            events.extend(event.to_record() for event in RedditSource(settings).fetch_mentions(ticker, start))
        except Exception as exc:
            errors.append(f"reddit:{exc}")
            console.print(f"[yellow]{ticker}: Reddit collection failed: {exc}[/yellow]")
    elif not no_reddit:
        console.print(f"[yellow]{ticker}: skipping Reddit because credentials are not configured.[/yellow]")

    if not no_finviz:
        try:
            console.print(f"{ticker}: collecting Finviz ticker news...")
            events.extend(event.to_record() for event in FinvizSource().fetch_news(ticker, start, end=end, limit=100))
        except Exception as exc:
            errors.append(f"finviz:{exc}")
            console.print(f"[yellow]{ticker}: Finviz news collection failed: {exc}[/yellow]")

    if not no_seeking_alpha and settings.has_seeking_alpha_rapidapi:
        try:
            console.print(f"{ticker}: collecting Seeking Alpha news/analysis via RapidAPI...")
            sa_events, sa_errors = SeekingAlphaRapidApiSource(settings).fetch_events_with_errors(ticker, start)
            events.extend(event.to_record() for event in sa_events)
            errors.extend(f"seeking_alpha:{error}" for error in sa_errors)
        except Exception as exc:
            errors.append(f"seeking_alpha:{exc}")
            console.print(f"[yellow]{ticker}: Seeking Alpha collection failed: {exc}[/yellow]")

    if not no_sec:
        try:
            console.print(f"{ticker}: collecting SEC filing events...")
            sec_forms = {
                "8-K",
                "10-Q",
                "10-K",
                "S-1",
                "S-3",
                "424B5",
                "424B3",
                "FWP",
                "DEF 14A",
                "SC 13G",
                "SC 13D",
                "4",
            }
            events.extend(event.to_record() for event in SecSource(settings).fetch_filings(ticker, start, end=end, forms=sec_forms))
        except Exception as exc:
            errors.append(f"sec:{exc}")
            console.print(f"[yellow]{ticker}: SEC filing collection failed: {exc}[/yellow]")

    frame = events_to_frame(events)
    frame = _filter_events_until(frame, end)
    frame, report = sanitize_events_frame(frame)
    if score and not frame.empty:
        try:
            console.print(f"{ticker}: scoring {len(frame)} events with FinBERT...")
            frame = add_finbert(frame, settings.finbert_model)
        except Exception as exc:
            errors.append(f"finbert:{exc}")
            console.print(f"[yellow]{ticker}: FinBERT scoring failed; raw events kept: {exc}[/yellow]")
    if report.missing_required_rows_removed:
        errors.append(f"sanitize:removed_missing_required={report.missing_required_rows_removed}")
    if report.duplicate_rows_removed:
        errors.append(f"sanitize:removed_duplicates={report.duplicate_rows_removed}")
    if report.future_timestamp_rows:
        errors.append(f"sanitize:removed_future_timestamps={report.future_timestamp_rows}")
    return frame, errors


def collect_events_frame(
    ticker: str,
    days: int,
    *,
    end: datetime | None = None,
    no_reddit: bool = False,
    score: bool = True,
) -> pd.DataFrame:
    frame, _ = collect_events_for_ticker(ticker, days, end=end, no_reddit=no_reddit, score=score)
    return frame


def _recent_events_for_behavior(
    ticker: str,
    days: int,
    *,
    raw_dir: Path,
    refresh: bool,
    no_reddit: bool,
    no_seeking_alpha: bool,
) -> tuple[pd.DataFrame, list[str]]:
    now = datetime.now(timezone.utc)
    cutoff = now - timedelta(days=days)
    errors: list[str] = []
    frame = pd.DataFrame()
    path = raw_dir / f"{ticker}_events.parquet"
    if path.exists() and not refresh:
        try:
            frame = pd.read_parquet(path)
        except Exception as exc:
            errors.append(f"cache_read:{exc}")
    if frame.empty or refresh:
        frame, collect_errors = collect_events_for_ticker(
            ticker,
            days,
            no_reddit=no_reddit,
            no_seeking_alpha=no_seeking_alpha,
            score=False,
        )
        errors.extend(collect_errors)
    frame, verify = sanitize_events_frame(frame)
    if verify.rows_out:
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame = frame[frame["timestamp"] >= cutoff].copy()
    if verify.duplicate_rows_removed:
        errors.append(f"sanitize:removed_duplicates={verify.duplicate_rows_removed}")
    if verify.future_timestamp_rows:
        errors.append(f"sanitize:removed_future_timestamps={verify.future_timestamp_rows}")
    return frame, errors


def _reaction_behavior_label(row: pd.Series) -> str:
    news_count = float(row.get("news_count", 0) or 0)
    sentiment = float(row.get("sentiment_mean", 0) or 0)
    return_1d = float(row.get("return_1d", 0) or 0)
    volume_z = float(row.get("volume_z20", 0) or 0)
    intraday_reaction = float(row.get("intraday_reaction_2h_mean", 0) or 0)
    premarket_gap = float(row.get("premarket_gap_mean", 0) or 0)
    afterhours_gap = float(row.get("afterhours_next_open_gap_mean", 0) or 0)
    reaction = max(intraday_reaction, premarket_gap, afterhours_gap, key=abs)

    if news_count <= 0:
        if abs(return_1d) >= 0.04 and volume_z > 1.0:
            return "price-led move; news catalyst not found"
        return "quiet/no recent catalyst"
    if sentiment >= 0.15 and (return_1d > 0.01 or reaction > 0.005):
        return "positive news confirmed by price"
    if sentiment >= 0.15 and return_1d < -0.01:
        return "positive news faded or sold"
    if sentiment <= -0.15 and return_1d >= 0:
        return "negative news absorbed"
    if sentiment <= -0.15 and return_1d < -0.01:
        return "negative news confirmed by price"
    if volume_z > 1.5 and abs(return_1d) >= 0.03:
        return "high-volume reaction, sentiment mixed"
    return "mixed/needs confirmation"


def _recent_headlines(events: pd.DataFrame, limit: int = 3) -> str:
    if events.empty or "title" not in events.columns:
        return ""
    values = (
        events.sort_values("timestamp", ascending=False)["title"]
        .dropna()
        .astype(str)
        .map(lambda value: value.strip())
    )
    values = [value for value in values if value]
    return " || ".join(values[:limit])


def _model_path(preferred: Path, fallback: Path | None = None) -> Path | None:
    if preferred.exists():
        return preferred
    if fallback and fallback.exists():
        return fallback
    return None


def _market_cap_bucket(market_cap: float | None) -> str:
    if market_cap is None or pd.isna(market_cap) or market_cap <= 0:
        return "unknown"
    if market_cap < 300_000_000:
        return "micro_cap"
    if market_cap < 2_000_000_000:
        return "small_cap"
    if market_cap < 10_000_000_000:
        return "mid_cap"
    if market_cap < 200_000_000_000:
        return "large_cap"
    return "mega_cap"


def _lookup_market_profile(ticker: str) -> dict[str, object]:
    try:
        import yfinance as yf

        info = yf.Ticker(ticker).fast_info
        market_cap = info.get("marketCap") if hasattr(info, "get") else None
        market_cap_value = float(market_cap) if market_cap not in (None, "") else None
        return {
            "market_cap": market_cap_value,
            "cap_bucket": _market_cap_bucket(market_cap_value),
        }
    except Exception as exc:
        return {"market_cap": None, "cap_bucket": "unknown", "profile_error": str(exc)}


def _signal_from_probability(probability: float | None) -> str:
    if probability is None or pd.isna(probability):
        return "no_model_signal"
    if probability >= 0.62:
        return "bullish_watch"
    if probability >= 0.55:
        return "lean_bullish"
    if probability <= 0.38:
        return "bearish_or_fade_watch"
    if probability <= 0.45:
        return "lean_bearish"
    return "neutral"


def _latest_price_snapshot(ticker: str, settings: object) -> dict[str, object]:
    try:
        prices = fetch_daily_prices(ticker, datetime.now(timezone.utc) - timedelta(days=45), datetime.now(timezone.utc), settings)
        if prices.empty:
            return {}
        prices = prices.sort_values("date")
        latest = prices.iloc[-1]
        previous_close = float(prices.iloc[-2]["close"]) if len(prices) > 1 else None
        close = float(latest["close"])
        return {
            "latest_bar_date": latest["date"],
            "latest_open": float(latest["open"]),
            "latest_close": close,
            "latest_volume": float(latest["volume"]),
            "latest_return_1d": close / previous_close - 1.0 if previous_close else None,
        }
    except Exception as exc:
        return {"price_error": str(exc)}


WATCHLIST_COLUMN_TITLES = {
    "ticker": "Ticker",
    "signal": "Overall Signal",
    "combined_probability_up": "Combined Probability Up",
    "daily_model_1d_probability_up": "Daily Model: Next-Day Up Probability",
    "daily_model_1d_prediction": "Daily Model: Next-Day Direction",
    "daily_model_5d_probability_up": "Daily Model: Next-5-Trading-Days Up Probability",
    "daily_model_5d_prediction": "Daily Model: Next-5-Trading-Days Direction",
    "event_model_1d_latest_probability_up": "Latest Event Model: Next-Day Up Probability",
    "event_model_1d_max_probability_up": "Strongest Recent Event: Next-Day Up Probability",
    "event_model_1d_mean_probability_up": "Average Recent Event: Next-Day Up Probability",
    "event_model_5d_latest_probability_up": "Latest Event Model: Next-5-Trading-Days Up Probability",
    "event_model_5d_max_probability_up": "Strongest Recent Event: Next-5-Trading-Days Up Probability",
    "event_model_5d_mean_probability_up": "Average Recent Event: Next-5-Trading-Days Up Probability",
    "recent_event_count": "Recent Catalyst Count",
    "alpaca_event_count": "Alpaca News Count",
    "reddit_event_count": "Reddit Chatter Count",
    "seeking_alpha_event_count": "Seeking Alpha Event Count",
    "sec_event_count": "SEC Filing Count",
    "latest_bar_date": "Latest Price Bar Date",
    "latest_open": "Latest Open",
    "latest_close": "Latest Close",
    "latest_return_1d": "Latest 1-Day Return",
    "latest_volume": "Latest Volume",
    "daily_return_1d": "Daily Feature 1-Day Return",
    "volume_z20": "20-Day Volume Z-Score",
    "cap_bucket": "Market-Cap Bucket",
    "market_cap": "Market Cap",
    "sector": "Configured Sector",
    "sector_benchmark": "Sector Benchmark ETF",
    "market_benchmark": "Market Benchmark ETF",
    "sentiment_mean_recent": "Average Recent News Sentiment",
    "watch_score": "Heuristic Watch Score",
    "recent_headlines": "Recent Headlines Used",
    "errors": "Collection or Scoring Notes",
    "lookback_days": "News Lookback Days",
    "status": "Status",
}


WATCHLIST_FIELD_DEFINITIONS = {
    "Combined Probability Up": "Average of available clean daily and event model probabilities. This is a ranking score, not a guaranteed trading probability.",
    "Daily Model: Next-Day Up Probability": "Probability from the daily ticker model that the stock closes up over the next trading day, using recent aggregated news, daily bars, SPY, and sector ETF context.",
    "Daily Model: Next-5-Trading-Days Up Probability": "Probability from the daily ticker model that the stock is up over the next 5 trading days.",
    "Latest Event Model: Next-Day Up Probability": "Probability from the event-level model using the most recent catalyst/news item. It asks: based on this event, prior price pattern, sector/SPY context, and event metadata, did similar historical events lead to a positive next-day move?",
    "Strongest Recent Event: Next-Day Up Probability": "Highest next-day event-model probability among all events found in the lookback window.",
    "Average Recent Event: Next-Day Up Probability": "Average next-day event-model probability across all recent events in the lookback window.",
    "Latest Event Model: Next-5-Trading-Days Up Probability": "Same as latest event probability, but target is the next 5 trading days.",
    "Overall Signal": "Bucketed interpretation of Combined Probability Up: bullish_watch, lean_bullish, neutral, lean_bearish, or bearish_or_fade_watch.",
    "Recent Catalyst Count": "Number of recent news/SEC/Reddit/Seeking Alpha events collected for the ticker.",
    "20-Day Volume Z-Score": "How unusual current volume is versus the recent 20-day baseline. Positive means above normal.",
    "Average Recent News Sentiment": "FinBERT sentiment average across recent collected event text: positive above 0, negative below 0.",
    "Heuristic Watch Score": "Rule-based score using news count, sentiment, volume, and price reaction. It is separate from ML probabilities.",
}


def _humanize_watchlist_report(report: pd.DataFrame) -> pd.DataFrame:
    preferred = [
        "ticker",
        "signal",
        "combined_probability_up",
        "daily_model_1d_probability_up",
        "daily_model_5d_probability_up",
        "event_model_1d_latest_probability_up",
        "event_model_5d_latest_probability_up",
        "event_model_1d_max_probability_up",
        "event_model_5d_max_probability_up",
        "recent_event_count",
        "alpaca_event_count",
        "reddit_event_count",
        "seeking_alpha_event_count",
        "sec_event_count",
        "latest_bar_date",
        "latest_open",
        "latest_close",
        "latest_return_1d",
        "latest_volume",
        "cap_bucket",
        "market_cap",
        "sector",
        "sector_benchmark",
        "sentiment_mean_recent",
        "watch_score",
        "recent_headlines",
        "errors",
    ]
    ordered = [col for col in preferred if col in report.columns]
    ordered.extend([col for col in report.columns if col not in ordered])
    human = report[ordered].rename(columns=WATCHLIST_COLUMN_TITLES)
    return human


def _write_watchlist_dictionary(path: Path) -> None:
    rows = [
        {"Field": field, "Meaning": meaning}
        for field, meaning in WATCHLIST_FIELD_DEFINITIONS.items()
    ]
    pd.DataFrame(rows).to_csv(path, index=False)


def _event_identity_columns(frame: pd.DataFrame) -> list[str]:
    return [col for col in ["ticker", "timestamp", "source", "title", "url"] if col in frame.columns]


def _upsert_events(existing_path: Path, new_events: pd.DataFrame) -> tuple[pd.DataFrame, int]:
    if existing_path.exists():
        existing = pd.read_parquet(existing_path)
    else:
        existing = pd.DataFrame()
    before = len(existing)
    combined = pd.concat([existing, new_events], ignore_index=True) if not existing.empty else new_events.copy()
    combined, _ = sanitize_events_frame(combined)
    identity = _event_identity_columns(combined)
    if identity:
        combined = combined.sort_values("timestamp").drop_duplicates(identity, keep="last")
    combined = combined.sort_values("timestamp").reset_index(drop=True)
    added = max(0, len(combined) - before)
    existing_path.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(existing_path, index=False)
    return combined, added


def _score_unscored_events(frame: pd.DataFrame, scorer: object, settings: object) -> pd.DataFrame:
    if frame.empty:
        return frame
    if "sentiment_numeric" in frame.columns and frame["sentiment_numeric"].notna().all():
        return frame
    scored = frame.copy()
    needs_score = (
        scored["sentiment_numeric"].isna()
        if "sentiment_numeric" in scored.columns
        else pd.Series(True, index=scored.index)
    )
    if not needs_score.any():
        return scored
    for col in ["title", "summary", "text"]:
        if col not in scored.columns:
            scored[col] = ""
    scored["text"] = scored["text"].fillna(scored["summary"]).fillna(scored["title"]).fillna("")
    score_input = scored.loc[needs_score].drop(
        columns=["sentiment_label", "sentiment_score", "sentiment_numeric"],
        errors="ignore",
    )
    score_output = add_finbert_with_scorer(score_input, scorer, batch_size=settings.finbert_batch_size)
    for col in ["sentiment_label", "sentiment_score", "sentiment_numeric"]:
        if col not in scored.columns:
            scored[col] = pd.NA
        scored.loc[needs_score, col] = score_output[col].to_numpy()
    return scored


def _score_live_ticker(
    symbol: str,
    events_path: Path,
    feature_dir: Path,
    predictions_dir: Path,
    settings: object,
    *,
    run_id: str,
    lookback_days: int,
    event_model_1d: Path | None,
    event_model_5d: Path | None,
    daily_model_1d: Path | None,
    daily_model_5d: Path | None,
) -> dict[str, object]:
    events = pd.read_parquet(events_path)
    event_features = build_event_swing_dataset(
        symbol,
        events_path,
        settings,
        horizon_days=5,
        market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
    )
    feature_path = feature_dir / f"{symbol}_event_swing.parquet"
    feature_path.parent.mkdir(parents=True, exist_ok=True)
    event_features.to_parquet(feature_path, index=False)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=lookback_days)
    recent_features = event_features[pd.to_datetime(event_features["event_timestamp"], utc=True) >= cutoff].copy()
    if recent_features.empty and not event_features.empty:
        recent_features = event_features.sort_values("event_timestamp").tail(3).copy()

    record: dict[str, object] = {
        "run_id": run_id,
        "ticker": symbol,
        "event_store_rows": len(events),
        "event_feature_rows": len(event_features),
        "recent_event_rows": len(recent_features),
        "feature_path": str(feature_path),
        "latest_headlines": _recent_headlines(events, limit=5),
        "sector": settings.sector_for_ticker(symbol) or "unknown",
        "sector_benchmark": settings.sector_benchmark_for_ticker(symbol),
        **_latest_price_snapshot(symbol, settings),
    }

    if not recent_features.empty and event_model_1d:
        scored = predict_event_swing_frame(recent_features, event_model_1d)
        scored_path = predictions_dir / f"{symbol}_recent_event_scores_1d.csv"
        scored.to_csv(scored_path, index=False)
        record["event_1d_latest_probability_up"] = float(scored.sort_values("event_timestamp").iloc[-1]["model_probability_up"])
        record["event_1d_max_probability_up"] = float(scored["model_probability_up"].max())
        record["event_1d_mean_probability_up"] = float(scored["model_probability_up"].mean())
    if not recent_features.empty and event_model_5d:
        scored = predict_event_swing_frame(recent_features, event_model_5d)
        scored_path = predictions_dir / f"{symbol}_recent_event_scores_5d.csv"
        scored.to_csv(scored_path, index=False)
        record["event_5d_latest_probability_up"] = float(scored.sort_values("event_timestamp").iloc[-1]["model_probability_up"])
        record["event_5d_max_probability_up"] = float(scored["model_probability_up"].max())
        record["event_5d_mean_probability_up"] = float(scored["model_probability_up"].mean())

    try:
        daily_1d = build_daily_dataset(
            symbol,
            events_path,
            settings,
            horizon_days=1,
            market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
        )
        record["watch_score"] = heuristic_watch_score(daily_1d, weights=settings.watch_score_weights).get("watch_score")
        if daily_model_1d:
            pred = predict_latest(daily_1d, daily_model_1d)
            record["daily_1d_probability_up"] = pred["probability_up"]
        if daily_model_5d:
            daily_5d = build_daily_dataset(
                symbol,
                events_path,
                settings,
                horizon_days=5,
                market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
            )
            pred = predict_latest(daily_5d, daily_model_5d)
            record["daily_5d_probability_up"] = pred["probability_up"]
    except Exception as exc:
        record["daily_error"] = str(exc)

    probs = [
        record.get("event_1d_latest_probability_up"),
        record.get("event_5d_latest_probability_up"),
        record.get("daily_1d_probability_up"),
        record.get("daily_5d_probability_up"),
    ]
    usable = [float(value) for value in probs if value is not None and not pd.isna(value)]
    record["combined_probability_up"] = sum(usable) / len(usable) if usable else None
    record["signal"] = _signal_from_probability(record["combined_probability_up"])
    return record


def _curate_live_training_set(feature_dir: Path, out: Path) -> dict[str, object]:
    frames = []
    for path in feature_dir.glob("*_event_swing.parquet"):
        frame = pd.read_parquet(path)
        if frame.empty:
            continue
        labeled = frame[
            frame["target_next_1d_up"].notna() | frame["target_next_5d_up"].notna()
        ].copy()
        if not labeled.empty:
            frames.append(labeled)
    out.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        empty = pd.DataFrame()
        empty.to_parquet(out, index=False)
        return {"curated_rows": 0, "curated_tickers": 0, "path": str(out)}
    combined = pd.concat(frames, ignore_index=True)
    identity = [col for col in ["ticker", "event_timestamp", "source", "title"] if col in combined.columns]
    if identity:
        combined = combined.drop_duplicates(identity, keep="last")
    combined = combined.sort_values(["event_timestamp", "ticker"]).reset_index(drop=True)
    combined.to_parquet(out, index=False)
    return {
        "curated_rows": len(combined),
        "curated_tickers": int(combined["ticker"].nunique()) if "ticker" in combined.columns else 0,
        "path": str(out),
    }


def _normalize_ohlcv(ticker: str, frame: pd.DataFrame, timeframe: str) -> pd.DataFrame:
    normalized = frame.copy()
    if timeframe == "1d":
        normalized["timestamp"] = pd.to_datetime(normalized["date"], utc=True)
    else:
        normalized["timestamp"] = pd.to_datetime(normalized["timestamp"], utc=True)
    normalized["symbol"] = ticker.upper()
    normalized["timeframe"] = timeframe
    normalized["source"] = "alpaca"
    normalized["adjustment"] = "all"
    normalized["ingested_at_utc"] = pd.Timestamp.now(tz="UTC")
    columns = [
        "symbol",
        "timeframe",
        "timestamp",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "source",
        "adjustment",
        "ingested_at_utc",
    ]
    for col in ["open", "high", "low", "close", "volume"]:
        normalized[col] = pd.to_numeric(normalized[col], errors="coerce")
    return normalized[columns].dropna(subset=["timestamp", "open", "high", "low", "close"]).sort_values("timestamp")


def _write_artifact_manifest(path: Path, payload: dict[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_seeking_alpha_snapshot(ticker: str, out: Path) -> Path:
    settings = get_settings()
    snapshot = SeekingAlphaRapidApiSource(settings).fetch_quant_snapshot(ticker)
    out.parent.mkdir(parents=True, exist_ok=True)
    row = pd.DataFrame([snapshot])
    if out.exists():
        existing = pd.read_csv(out)
        frame = pd.concat([existing, row], ignore_index=True)
    else:
        frame = row
    frame.to_csv(out, index=False)
    return out


def _worker_count(requested: int | None, configured: int, total: int) -> int:
    return max(1, min(int(requested or configured), max(1, total)))


@app.command("download-model")
def download_model_command() -> None:
    """Download the configured FinBERT model into the local Hugging Face cache."""
    from market_predictor.sentiment import download_model

    settings = get_settings()
    download_model(settings.finbert_model)
    console.print(f"Downloaded model cache for {settings.finbert_model}")


@app.command("alpaca-tickers")
def alpaca_tickers(
    out: Path = typer.Option(Path("data/universe/alpaca_tickers.csv"), help="Output ticker universe CSV."),
) -> None:
    """Fetch active/tradable US equity tickers from Alpaca assets."""
    settings = get_settings()
    frame = AlpacaSource(settings).fetch_ticker_universe()
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote {len(frame)} Alpaca tickers to {out}")


@app.command()
def collect(
    ticker: str,
    days: int = typer.Option(90, help="Lookback window in calendar days."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD. Defaults to today/now."),
    out: Path = typer.Option(Path("data/raw/events.parquet"), help="Output parquet path."),
    no_reddit: bool = typer.Option(False, help="Disable Reddit enrichment."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha enrichment."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
) -> None:
    """Collect raw events for a ticker and score them with FinBERT."""
    end = _parse_end_date(end_date)
    frame, _ = collect_events_for_ticker(
        ticker,
        days,
        end=end,
        no_reddit=no_reddit,
        no_seeking_alpha=no_seeking_alpha,
        no_sec=no_sec,
        score=True,
    )
    if frame.empty:
        raise typer.BadParameter("No events collected. Configure Alpaca/Reddit credentials or widen the date range.")
    frame, report = sanitize_events_frame(frame)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    console.print({"verification": report.to_record()})
    console.print(f"Wrote {len(frame)} events to {out}")


@app.command("swing-universe")
def swing_universe(
    out: Path = typer.Option(Path("data/universe/swing_candidates.csv"), help="Output CSV for configured swing symbols."),
    tickers: str | None = typer.Option(None, help="Optional comma-separated symbols to use instead of config."),
) -> None:
    """Write the configured swing watch universe."""
    settings = get_settings()
    values = _parse_tickers(tickers, settings.swing_candidate_tickers)
    frame = pd.DataFrame({"ticker": values, "is_seed": [ticker in settings.swing_seed_tickers for ticker in values]})
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(f"Wrote {len(frame)} swing tickers to {out}")


def _finviz_candidates_from_values(values: list[str], settings: object) -> pd.DataFrame:
    cleaned = []
    for value in values:
        symbol = str(value).strip().upper()
        if re.fullmatch(r"[A-Z][A-Z0-9.]{0,9}", symbol):
            cleaned.append(symbol)
    current = set(settings.swing_candidate_tickers)
    sector_map = settings.ticker_sector_map
    rows = [
        {
            "ticker": symbol,
            "already_in_universe": symbol in current,
            "sector": sector_map.get(symbol, ""),
            "sector_benchmark": settings.sector_benchmark_for_ticker(symbol),
            "market_benchmark": settings.market_benchmark_ticker,
        }
        for symbol in dict.fromkeys(cleaned)
    ]
    return pd.DataFrame(rows)


FINVIZ_DEFAULT_SECTORS = {
    "technology": "sec_technology",
    "healthcare": "sec_healthcare",
    "financial": "sec_financial",
    "industrial": "sec_industrials",
    "consumer_cyclical": "sec_consumercyclical",
    "energy": "sec_energy",
    "communication": "sec_communicationservices",
    "materials": "sec_basicmaterials",
}

FINVIZ_DEFAULT_CAPS = {
    "mega": "cap_mega",
    "large": "cap_large",
    "mid": "cap_mid",
    "small": "cap_small",
    "micro": "cap_micro",
}


def _redact_url_query(url: str) -> str:
    parts = urlsplit(url)
    return urlunsplit((parts.scheme, parts.netloc, parts.path, "<redacted>", parts.fragment))


def _redact_finviz_auth_text(value: object) -> str:
    return re.sub(r"auth=[^&\s]+", "auth=<redacted>", str(value))


def _finviz_export_url(base_url: str, filters: list[str], auth: str) -> str:
    request = requests.Request(
        "GET",
        base_url,
        params={"v": "111", "f": ",".join(filters), "auth": auth},
    ).prepare()
    if not request.url:
        raise ValueError("Could not build Finviz export URL.")
    return request.url


@app.command("import-finviz")
def import_finviz(
    tickers: str | None = typer.Option(None, help="Pasted symbols from Finviz, separated by commas/spaces/newlines."),
    csv: Path | None = typer.Option(None, help="Optional Finviz CSV export path."),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name for Finviz CSV exports."),
    out: Path = typer.Option(Path("data/universe/finviz_candidates.csv"), help="Cleaned output CSV."),
) -> None:
    """Clean Finviz Elite symbols and compare them with the configured universe."""
    settings = get_settings()
    values: list[str] = []
    if csv:
        frame = pd.read_csv(csv)
        column = symbol_column if symbol_column in frame.columns else frame.columns[0]
        values.extend(frame[column].dropna().astype(str).tolist())
    if tickers:
        values.extend(re.split(r"[\s,;|]+", tickers))
    result = _finviz_candidates_from_values(values, settings)
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    console.print(result.head(50))
    console.print(f"Wrote {len(result)} cleaned Finviz candidates to {out}")


@app.command("download-finviz")
def download_finviz(
    url: str = typer.Option(..., help="Finviz Elite export URL. The auth query is not saved."),
    raw_out: Path = typer.Option(Path("data/external/finviz/finviz_export.csv"), help="Raw CSV output path."),
    candidates_out: Path = typer.Option(
        Path("data/universe/finviz_candidates.csv"),
        help="Cleaned ticker candidate output CSV.",
    ),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name in the Finviz export."),
) -> None:
    """Download a Finviz Elite export and extract candidate symbols."""
    response = requests.get(
        url,
        headers={"User-Agent": "market-predictor/0.1"},
        timeout=60,
    )
    response.raise_for_status()
    raw_out.parent.mkdir(parents=True, exist_ok=True)
    raw_out.write_bytes(response.content)
    frame = pd.read_csv(raw_out)
    if frame.empty:
        raise typer.BadParameter("Finviz export returned no rows.")
    column = symbol_column if symbol_column in frame.columns else frame.columns[0]
    settings = get_settings()
    candidates = _finviz_candidates_from_values(frame[column].dropna().astype(str).tolist(), settings)
    candidates_out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(candidates_out, index=False)
    console.print(
        {
            "downloaded": str(raw_out),
            "source": _redact_url_query(url),
            "rows": len(frame),
            "candidate_symbols": len(candidates),
            "candidates": str(candidates_out),
        }
    )
    console.print(candidates.head(50))


@app.command("download-finviz-screeners")
def download_finviz_screeners(
    auth: str | None = typer.Option(None, help="Finviz Elite auth token. Defaults to FINVIZ_ELITE_AUTH."),
    base_url: str = typer.Option("https://elite.finviz.com/export", help="Finviz Elite export endpoint."),
    sectors: str | None = typer.Option(None, help="Comma-separated sector keys. Defaults to broad sector set."),
    caps: str | None = typer.Option(None, help="Comma-separated cap keys. Defaults to mega,large,mid,small,micro."),
    extra_filters: str = typer.Option("sh_price_o5,sh_avgvol_o500", help="Comma-separated Finviz filters added to each screen."),
    max_per_bucket: int = typer.Option(20, help="Maximum symbols to keep from each sector/cap bucket."),
    sleep_seconds: float = typer.Option(1.5, help="Delay between Finviz export requests to avoid throttling."),
    raw_dir: Path = typer.Option(Path("data/external/finviz/screeners"), help="Raw per-screen CSV directory."),
    out: Path = typer.Option(Path("data/universe/finviz_sector_cap_candidates.csv"), help="Combined candidate CSV."),
    tickers_out: Path = typer.Option(Path("data/universe/finviz_sector_cap_tickers.txt"), help="Comma-separated ticker output."),
    symbol_column: str = typer.Option("Ticker", help="Symbol column name in Finviz exports."),
) -> None:
    """Download multiple Finviz Elite screener exports across sector and cap buckets."""
    settings = get_settings()
    token = auth or settings.finviz_elite_auth
    if not token:
        raise typer.BadParameter("Provide --auth or set FINVIZ_ELITE_AUTH in .env.")
    sector_keys = [key.strip() for key in (sectors.split(",") if sectors else FINVIZ_DEFAULT_SECTORS.keys())]
    cap_keys = [key.strip() for key in (caps.split(",") if caps else FINVIZ_DEFAULT_CAPS.keys())]
    selected_sectors = {key: FINVIZ_DEFAULT_SECTORS[key] for key in sector_keys if key in FINVIZ_DEFAULT_SECTORS}
    selected_caps = {key: FINVIZ_DEFAULT_CAPS[key] for key in cap_keys if key in FINVIZ_DEFAULT_CAPS}
    extras = [item.strip() for item in extra_filters.split(",") if item.strip()]
    raw_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    summary = []
    for sector_name, sector_filter in selected_sectors.items():
        for cap_name, cap_filter in selected_caps.items():
            filters = [sector_filter, cap_filter, *extras]
            url = _finviz_export_url(base_url, filters, token)
            raw_path = raw_dir / f"{sector_name}_{cap_name}.csv"
            try:
                response = requests.get(url, headers={"User-Agent": "market-predictor/0.1"}, timeout=60)
                response.raise_for_status()
                raw_path.write_bytes(response.content)
                frame = pd.read_csv(raw_path)
                if frame.empty:
                    summary.append({"sector": sector_name, "cap": cap_name, "rows": 0, "kept": 0})
                    continue
                column = symbol_column if symbol_column in frame.columns else frame.columns[0]
                frame = frame.head(max_per_bucket).copy()
                frame["finviz_sector_bucket"] = sector_name
                frame["finviz_cap_bucket"] = cap_name
                frame["finviz_filters"] = ",".join(filters)
                rows.append(frame)
                summary.append({"sector": sector_name, "cap": cap_name, "rows": len(pd.read_csv(raw_path)), "kept": len(frame)})
            except Exception as exc:
                summary.append(
                    {
                        "sector": sector_name,
                        "cap": cap_name,
                        "rows": 0,
                        "kept": 0,
                        "error": _redact_finviz_auth_text(exc),
                    }
                )
            if sleep_seconds > 0:
                time_module.sleep(sleep_seconds)
    if not rows:
        pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
        raise typer.BadParameter("No Finviz rows downloaded from the requested screens.")
    combined = pd.concat(rows, ignore_index=True)
    column = symbol_column if symbol_column in combined.columns else combined.columns[0]
    candidates = _finviz_candidates_from_values(combined[column].dropna().astype(str).tolist(), settings)
    metadata_cols = [column, "finviz_sector_bucket", "finviz_cap_bucket", "Sector", "Industry", "Market Cap", "Price", "Volume"]
    metadata = combined[[col for col in metadata_cols if col in combined.columns]].copy()
    metadata = metadata.rename(columns={column: "ticker"}).drop_duplicates("ticker")
    metadata["ticker"] = metadata["ticker"].astype(str).str.upper()
    result = candidates.merge(metadata, on="ticker", how="left").drop_duplicates("ticker")
    result = result.sort_values(["already_in_universe", "finviz_sector_bucket", "finviz_cap_bucket", "ticker"])
    out.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(out, index=False)
    tickers = result.loc[~result["already_in_universe"], "ticker"].dropna().astype(str).tolist()
    tickers_out.parent.mkdir(parents=True, exist_ok=True)
    tickers_out.write_text(",".join(tickers), encoding="utf-8")
    pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
    console.print({"screens": len(summary), "unique_candidates": len(result), "new_tickers": len(tickers), "out": str(out)})
    console.print(result.head(80))


@app.command("build-intraday-universe")
def build_intraday_universe_command(
    raw: Path = typer.Option(
        Path("data/external/finviz/nasdaq200/nasdaq_liquid_raw_20260707.csv"),
        help="Raw Finviz export CSV.",
    ),
    out: Path = typer.Option(
        Path("data/universe/intraday_nasdaq_activity_latest.csv"),
        help="Ranked intraday candidate CSV.",
    ),
    tickers_out: Path = typer.Option(
        Path("data/universe/intraday_nasdaq_activity_latest_tickers.txt"),
        help="Comma-separated ticker output.",
    ),
    top_n: int = typer.Option(200, help="Number of candidates to keep."),
    min_price: float = typer.Option(2.0, help="Minimum stock price."),
    min_volume: int = typer.Option(500_000, help="Minimum current volume."),
    min_abs_change_pct: float = typer.Option(0.5, help="Minimum absolute day change percent."),
    min_market_cap_m: float = typer.Option(100.0, help="Minimum market cap in millions."),
) -> None:
    """Rank NASDAQ Finviz rows for volatile/high-volume intraday candidates."""
    if not raw.exists():
        raise typer.BadParameter(f"Missing raw Finviz CSV: {raw}")
    frame = pd.read_csv(raw)
    candidates = build_intraday_candidate_universe(
        frame,
        top_n=top_n,
        min_price=min_price,
        min_volume=min_volume,
        min_abs_change_pct=min_abs_change_pct,
        min_market_cap_m=min_market_cap_m,
    )
    if candidates.empty:
        raise typer.BadParameter("No intraday candidates matched the requested filters.")
    out.parent.mkdir(parents=True, exist_ok=True)
    tickers_out.parent.mkdir(parents=True, exist_ok=True)
    candidates.to_csv(out, index=False)
    tickers_out.write_text(",".join(candidates["ticker"].astype(str)), encoding="utf-8")
    console.print({"raw_rows": len(frame), "candidates": len(candidates), "out": str(out)})
    console.print(candidates.head(50))


@app.command("collect-swing")
def collect_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(120, help="Lookback window in calendar days."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD. Defaults to today/now."),
    out_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory for per-ticker event parquet files."),
    no_reddit: bool = typer.Option(False, help="Disable Reddit enrichment."),
    no_finviz: bool = typer.Option(False, help="Disable Finviz ticker-news enrichment."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha enrichment."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing enrichment."),
    score: bool = typer.Option(False, help="Run FinBERT during collection. Default false keeps API download separate."),
    workers: int | None = typer.Option(None, help="Parallel API download workers. Defaults to config performance.max_workers."),
) -> None:
    """Bulk collect Alpaca news, Reddit chatter, and Seeking Alpha events for swing candidates."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    end = _parse_end_date(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Collecting {len(symbols)} tickers with {max_workers} worker(s).")

    def run_symbol(symbol: str) -> dict:
        try:
            frame, errors = collect_events_for_ticker(
                symbol,
                days,
                end=end,
                no_reddit=no_reddit,
                no_finviz=no_finviz,
                no_seeking_alpha=no_seeking_alpha,
                no_sec=no_sec,
                score=score,
            )
            path = out_dir / f"{symbol}_events.parquet"
            frame.to_parquet(path, index=False)
            _, verify = sanitize_events_frame(frame)
            return {
                "ticker": symbol,
                "events": len(frame),
                "path": str(path),
                "errors": " | ".join(errors),
                "sources": verify.sources,
            }
        except Exception as exc:
            return {"ticker": symbol, "events": 0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            record = future.result()
            summary.append(record)
            if record.get("error"):
                console.print(f"[red]{symbol}: collection failed: {record['error']}[/red]")
            else:
                console.print(f"{symbol}: wrote {record['events']} events to {record['path']}")
    pd.DataFrame(summary).to_csv(out_dir / "_collection_summary.csv", index=False)


@app.command("verify-events")
def verify_events(
    events: Path = typer.Option(..., help="Input events parquet."),
    rewrite: bool = typer.Option(False, help="Rewrite the file with sanitized rows."),
) -> None:
    """Sanitize and verify an event parquet file without calling APIs or ML."""
    frame = pd.read_parquet(events)
    clean, report = sanitize_events_frame(frame)
    if rewrite:
        clean.to_parquet(events, index=False)
    console.print(report.to_record())


@app.command("verify-swing")
def verify_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing per-ticker event parquet files."),
    rewrite: bool = typer.Option(False, help="Rewrite each file with sanitized rows."),
    out: Path = typer.Option(Path("data/raw/swing/_verification_summary.csv"), help="Output verification summary CSV."),
) -> None:
    """Bulk verify swing event files with per-ticker isolation."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows = []
    for symbol in symbols:
        path = raw_dir / f"{symbol}_events.parquet"
        if not path.exists():
            rows.append({"ticker": symbol, "rows_out": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            clean, report = sanitize_events_frame(frame)
            if rewrite:
                clean.to_parquet(path, index=False)
            record = report.to_record()
            record["ticker"] = symbol
            rows.append(record)
        except Exception as exc:
            rows.append({"ticker": symbol, "rows_out": 0, "error": str(exc)})
            console.print(f"[red]{symbol}: verification failed: {exc}[/red]")
    summary = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary.to_csv(out, index=False)
    console.print(summary[["ticker", "rows_out", "duplicate_rows_removed", "missing_required_rows_removed"]].head(40))
    console.print(f"Wrote verification summary to {out}")


@app.command("score-swing-events")
def score_swing_events(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Directory containing raw per-ticker event parquet files.",
    ),
    out_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean_scored"),
        help="Directory for FinBERT-scored per-ticker event parquet files.",
    ),
    force: bool = typer.Option(False, help="Rescore even if an output file already exists."),
) -> None:
    """Run FinBERT on existing downloaded events without calling news APIs."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    out_dir.mkdir(parents=True, exist_ok=True)
    scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
    summary = []
    for symbol in symbols:
        source = raw_dir / f"{symbol}_events.parquet"
        target = out_dir / f"{symbol}_events.parquet"
        if target.exists() and not force:
            frame = pd.read_parquet(target)
            summary.append({"ticker": symbol, "events": len(frame), "path": str(target), "skipped": True})
            console.print(f"{symbol}: scored file exists, skipped")
            continue
        if not source.exists():
            summary.append({"ticker": symbol, "events": 0, "error": f"missing {source}"})
            console.print(f"[red]{symbol}: missing {source}[/red]")
            continue
        try:
            frame = pd.read_parquet(source)
            frame, report = sanitize_events_frame(frame)
            for col in ["title", "summary", "text"]:
                if col not in frame.columns:
                    frame[col] = ""
            frame["text"] = frame["text"].fillna(frame["summary"]).fillna(frame["title"]).fillna("")
            frame = frame.drop(
                columns=["sentiment_label", "sentiment_score", "sentiment_numeric"],
                errors="ignore",
            )
            frame = add_finbert_with_scorer(frame, scorer, batch_size=settings.finbert_batch_size)
            frame.to_parquet(target, index=False)
            summary.append(
                {
                    "ticker": symbol,
                    "events": len(frame),
                    "path": str(target),
                    "missing_required_rows_removed": report.missing_required_rows_removed,
                    "duplicate_rows_removed": report.duplicate_rows_removed,
                }
            )
            console.print(f"{symbol}: scored {len(frame)} events to {target}")
        except Exception as exc:
            summary.append({"ticker": symbol, "events": 0, "error": str(exc)})
            console.print(f"[red]{symbol}: sentiment scoring failed: {exc}[/red]")
    pd.DataFrame(summary).to_csv(out_dir / "_sentiment_summary.csv", index=False)


@app.command("audit-swing-alignment")
def audit_swing_alignment(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing per-ticker event parquet files."),
    feature_dir: Path = typer.Option(Path("data/features/swing"), help="Directory containing per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Feature dataset horizon to audit."),
    out: Path = typer.Option(Path("data/reports/swing_alignment_audit.csv"), help="Output audit CSV."),
) -> None:
    """Audit news timing assignment and daily/hourly candle feature matching."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows = []
    for symbol in symbols:
        events_path = raw_dir / f"{symbol}_events.parquet"
        features_path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not events_path.exists() or not features_path.exists():
            rows.append(
                {
                    "ticker": symbol,
                    "error": f"missing events={events_path.exists()} features={features_path.exists()}",
                }
            )
            continue
        try:
            events, verify = sanitize_events_frame(pd.read_parquet(events_path))
            dataset = pd.read_parquet(features_path)
            if events.empty or dataset.empty:
                rows.append({"ticker": symbol, "events": len(events), "feature_rows": len(dataset), "error": "empty"})
                continue
            events["timestamp"] = pd.to_datetime(events["timestamp"], utc=True)
            dataset_dates = set(pd.to_datetime(dataset["date"]).dt.date)
            events = align_events_to_trading_dates(events, list(dataset_dates))
            events = add_event_taxonomy(events)
            latest_feature_date = max(dataset_dates)
            events["has_feature_row"] = events["date"].isin(dataset_dates)
            events["pending_after_latest_feature_date"] = events["date"] > latest_feature_date
            events["missing_historical_feature_row"] = (~events["has_feature_row"]) & (
                ~events["pending_after_latest_feature_date"]
            )
            source_counts = events["source"].map(source_family_for_source).value_counts().to_dict()
            grouped_events = events.groupby("date").size().rename("event_count_raw")
            grouped_dataset = dataset.copy()
            grouped_dataset["date"] = pd.to_datetime(grouped_dataset["date"]).dt.date
            grouped_dataset = grouped_dataset.set_index("date")
            joined = grouped_events.to_frame().join(grouped_dataset[["news_count"]], how="left")
            joined["news_count_diff"] = joined["event_count_raw"] - joined["news_count"].fillna(0)
            event_dates = events.groupby("event_time_bucket")["date"].nunique().to_dict()

            def nonzero_days(column: str, bucket: str | None = None) -> int:
                dates = set(events["date"] if bucket is None else events.loc[events["event_time_bucket"] == bucket, "date"])
                if column not in grouped_dataset.columns or not dates:
                    return 0
                series = pd.to_numeric(grouped_dataset.loc[grouped_dataset.index.intersection(dates), column], errors="coerce")
                return int((series.fillna(0).abs() > 1e-12).sum())

            rows.append(
                {
                    "ticker": symbol,
                    "events": len(events),
                    "feature_rows": len(dataset),
                    "alpaca_events": int(source_counts.get("alpaca", 0)),
                    "seeking_alpha_events": int(source_counts.get("seeking_alpha", 0)),
                    "reddit_events": int(source_counts.get("reddit", 0)),
                    "finviz_events": int(source_counts.get("finviz", 0)),
                    "events_without_feature_row": int((~events["has_feature_row"]).sum()),
                    "pending_after_latest_feature_date": int(events["pending_after_latest_feature_date"].sum()),
                    "missing_historical_feature_rows": int(events["missing_historical_feature_row"].sum()),
                    "dates_with_news_count_mismatch": int((joined["news_count_diff"].abs() > 0).sum()),
                    "max_abs_news_count_diff": float(joined["news_count_diff"].abs().max() or 0),
                    "pre_market_event_dates": int(event_dates.get("pre_market", 0)),
                    "intraday_event_dates": int(event_dates.get("intraday", 0)),
                    "after_hours_event_dates": int(event_dates.get("after_hours", 0)),
                    "premarket_gap_matched_days": nonzero_days("premarket_gap_mean", "pre_market"),
                    "intraday_2h_reaction_matched_days": nonzero_days("intraday_reaction_2h_mean", "intraday"),
                    "afterhours_gap_matched_days": nonzero_days("afterhours_next_open_gap_mean", "after_hours"),
                    "sanitized_rows_out": verify.rows_out,
                }
            )
        except Exception as exc:
            rows.append({"ticker": symbol, "error": str(exc)})
            console.print(f"[red]{symbol}: alignment audit failed: {exc}[/red]")
    frame = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(frame.head(60))
    console.print(f"Wrote alignment audit to {out}")


@app.command("score-events")
def score_events(
    events: Path = typer.Option(..., help="Input raw events parquet."),
    out: Path | None = typer.Option(None, help="Output scored parquet. Defaults to overwriting input."),
) -> None:
    """Run FinBERT scoring for one raw event file."""
    settings = get_settings()
    frame = pd.read_parquet(events)
    frame, verify = sanitize_events_frame(frame)
    if frame.empty:
        raise typer.BadParameter(f"No events found in {events}")
    scored = add_finbert(frame, settings.finbert_model)
    scored, _ = sanitize_events_frame(scored)
    target = out or events
    target.parent.mkdir(parents=True, exist_ok=True)
    scored.to_parquet(target, index=False)
    console.print({"input_verification": verify.to_record()})
    console.print(f"Scored {len(scored)} events into {target}")


@app.command("score-swing")
def score_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing per-ticker event parquet files."),
    out_dir: Path | None = typer.Option(None, help="Output directory. Defaults to overwriting raw_dir files."),
    batch_size: int | None = typer.Option(None, help="FinBERT batch size. Defaults to config performance.finbert_batch_size."),
) -> None:
    """Bulk FinBERT scoring with one GPU-aware model load and per-ticker writes."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    target_dir = out_dir or raw_dir
    target_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    frames = []
    for symbol in symbols:
        source = raw_dir / f"{symbol}_events.parquet"
        if not source.exists():
            summary.append({"ticker": symbol, "events": 0, "error": f"missing {source}"})
            console.print(f"[yellow]{symbol}: missing {source}; skipping scoring.[/yellow]")
            continue
        try:
            frame = pd.read_parquet(source)
            frame, verify = sanitize_events_frame(frame)
            if frame.empty:
                summary.append({"ticker": symbol, "events": 0, "error": "empty events"})
                continue
            frame["_score_symbol"] = symbol
            frame["_score_row"] = range(len(frame))
            frames.append(frame)
            summary.append({"ticker": symbol, "events": len(frame), "sources": verify.sources})
        except Exception as exc:
            summary.append({"ticker": symbol, "events": 0, "error": str(exc)})
            console.print(f"[red]{symbol}: staging failed: {exc}[/red]")

    if frames:
        combined = pd.concat(frames, ignore_index=True)
        effective_batch = int(batch_size or settings.finbert_batch_size)
        console.print(
            f"Scoring {len(combined)} events from {combined['_score_symbol'].nunique()} ticker(s) "
            f"with one FinBERT model, batch_size={effective_batch}."
        )
        scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
        scored_all = add_finbert_with_scorer(combined, scorer, batch_size=effective_batch)
        for symbol, scored in scored_all.groupby("_score_symbol", sort=False):
            target = target_dir / f"{symbol}_events.parquet"
            scored = scored.drop(columns=["_score_symbol", "_score_row"], errors="ignore")
            scored, _ = sanitize_events_frame(scored)
            scored.to_parquet(target, index=False)
            console.print(f"{symbol}: scored {len(scored)} events into {target}")
    pd.DataFrame(summary).to_csv(target_dir / "_scoring_summary.csv", index=False)


@app.command("build-swing-datasets")
def build_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(Path("data/raw/swing"), help="Directory containing *_events.parquet files."),
    out_dir: Path = typer.Option(Path("data/features/swing"), help="Directory for per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Forward trading-day target horizon."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD for price/features."),
    with_seeking_alpha: bool = typer.Option(True, help="Fetch/cache Seeking Alpha quant snapshots when configured."),
    workers: int | None = typer.Option(None, help="Parallel dataset build workers. Defaults to config performance.max_workers."),
) -> None:
    """Build daily feature/label datasets for the swing universe."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    end = _parse_end_date(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = []
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Building {len(symbols)} datasets with {max_workers} worker(s).")

    def run_symbol(symbol: str) -> dict:
        events_path = raw_dir / f"{symbol}_events.parquet"
        if not events_path.exists():
            return {"ticker": symbol, "rows": 0, "error": f"missing {events_path}"}
        try:
            sa_path = None
            if with_seeking_alpha and settings.has_seeking_alpha_rapidapi:
                try:
                    sa_path = write_seeking_alpha_snapshot(symbol, out_dir / f"{symbol}_seeking_alpha_quant.csv")
                except Exception as exc:
                    console.print(f"[yellow]{symbol}: Seeking Alpha snapshot failed; building without quant: {exc}[/yellow]")
            dataset = build_daily_dataset(
                symbol,
                events_path,
                settings,
                horizon_days=horizon_days,
                seeking_alpha_path=sa_path,
                market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                end=end,
            )
            out = out_dir / f"{symbol}_daily_{horizon_days}d.parquet"
            dataset.to_parquet(out, index=False)
            return {"ticker": symbol, "rows": len(dataset), "path": str(out)}
        except Exception as exc:
            return {"ticker": symbol, "rows": 0, "error": str(exc)}

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            record = future.result()
            summary.append(record)
            if record.get("error"):
                console.print(f"[red]{symbol}: dataset build failed: {record['error']}[/red]")
            else:
                console.print(f"{symbol}: wrote {record['rows']} rows to {record['path']}")
    pd.DataFrame(summary).to_csv(out_dir / "_dataset_summary.csv", index=False)


@app.command("rank-swing")
def rank_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(Path("data/features/swing"), help="Directory containing per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Dataset horizon to rank."),
    model: Path | None = typer.Option(None, help="Optional trained model to add probability_up."),
    out: Path = typer.Option(Path("data/reports/swing_watch_rank.csv"), help="Output rank CSV."),
) -> None:
    """Rank latest watch scores across the swing universe."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not path.exists():
            continue
        dataset = pd.read_parquet(path)
        score = heuristic_watch_score(dataset, weights=settings.watch_score_weights)
        if model:
            try:
                prediction = predict_latest(dataset, model)
                score["model_probability_up"] = prediction["probability_up"]
                score["model_prediction"] = prediction["prediction"]
            except Exception as exc:
                score["model_error"] = str(exc)
        score["ticker"] = symbol
        rows.append(score)
    if not rows:
        raise typer.BadParameter("No feature datasets found to rank.")
    frame = pd.DataFrame(rows).sort_values(["watch_score", "event_count", "news_count"], ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(frame.head(30))
    console.print(f"Wrote ranked swing watchlist to {out}")


@app.command("negative-reaction")
def negative_reaction(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(
        Path("data/features/uslisted_6sector_2y_clean"),
        help="Directory containing per-ticker datasets.",
    ),
    horizon_days: int = typer.Option(1, help="Dataset horizon to scan."),
    lookback_rows: int = typer.Option(5, help="Recent trading rows to inspect."),
    out: Path = typer.Option(Path("data/reports/negative_reaction_candidates.csv"), help="Output CSV."),
) -> None:
    """Find recent candidates where news/catalyst attention was met with weak price reaction."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    rows = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not path.exists():
            continue
        try:
            dataset = pd.read_parquet(path).sort_values("date").tail(lookback_rows).copy()
            if dataset.empty:
                continue
            for row in dataset.itertuples(index=False):
                record = row._asdict()
                news_count = float(record.get("news_count", 0) or 0)
                event_count = float(record.get("event_count", 0) or 0)
                sentiment = float(record.get("sentiment_mean", 0) or 0)
                rel_spy = float(record.get("rel_return_1d_vs_spy", record.get("return_1d", 0)) or 0)
                rel_sector = float(record.get("rel_return_1d_vs_sector", record.get("return_1d", 0)) or 0)
                reaction = float(record.get("event_reaction_2h_mean", 0) or 0)
                volume_z = float(record.get("volume_z20", 0) or 0)
                if news_count + event_count <= 0:
                    continue
                if rel_spy >= 0 and rel_sector >= 0 and reaction >= 0:
                    continue
                attention = min(news_count + event_count, 20)
                mismatch = max(-rel_spy, 0) + max(-rel_sector, 0) + max(-reaction, 0)
                score = attention * (0.5 + max(sentiment, 0)) + 10.0 * mismatch + max(volume_z, 0) * 0.25
                rows.append(
                    {
                        "ticker": symbol,
                        "date": record.get("date"),
                        "negative_reaction_score": round(float(score), 4),
                        "news_count": news_count,
                        "event_count": event_count,
                        "sentiment_mean": sentiment,
                        "return_1d": float(record.get("return_1d", 0) or 0),
                        "rel_return_1d_vs_spy": rel_spy,
                        "rel_return_1d_vs_sector": rel_sector,
                        "event_reaction_2h_mean": reaction,
                        "volume_z20": volume_z,
                        "sector": record.get("sector_name", settings.sector_for_ticker(symbol) or ""),
                        "sector_benchmark": record.get("sector_benchmark", settings.sector_benchmark_for_ticker(symbol)),
                    }
                )
        except Exception as exc:
            rows.append({"ticker": symbol, "error": str(exc)})
    if not rows:
        raise typer.BadParameter("No negative reaction candidates found.")
    frame = pd.DataFrame(rows).sort_values("negative_reaction_score", ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    console.print(frame.head(40))
    console.print(f"Wrote negative reaction candidates to {out}")


@app.command("combine-swing-datasets")
def combine_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(Path("data/features/swing"), help="Directory containing per-ticker datasets."),
    horizon_days: int = typer.Option(1, help="Dataset horizon to combine."),
    out: Path = typer.Option(Path("data/features/swing_combined_1d.parquet"), help="Combined output parquet."),
) -> None:
    """Combine per-ticker swing datasets for cross-sectional model training."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    frames = []
    summary = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_daily_{horizon_days}d.parquet"
        if not path.exists():
            summary.append({"ticker": symbol, "rows": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            frame["ticker"] = symbol
            frame = frame[_daily_training_columns(frame, horizon_days)]
            frames.append(frame)
            summary.append({"ticker": symbol, "rows": len(frame), "path": str(path)})
        except Exception as exc:
            summary.append({"ticker": symbol, "rows": 0, "error": str(exc)})
    if not frames:
        raise typer.BadParameter("No datasets found to combine.")
    combined = pd.concat(frames, ignore_index=True).sort_values(["date", "ticker"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
    console.print(f"Wrote {len(combined)} rows across {len(frames)} tickers to {out}")


@app.command("build-event-swing-datasets")
def build_event_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Directory containing per-ticker event parquet files.",
    ),
    out_dir: Path = typer.Option(
        Path("data/features/event_swing_2y_clean"),
        help="Directory for per-ticker event-level datasets.",
    ),
    horizon_days: int = typer.Option(5, help="Forward trading-day swing horizon."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD for price/features."),
    workers: int | None = typer.Option(None, help="Parallel feature workers. Defaults to config performance.max_workers."),
) -> None:
    """Build one row per news/filing/chatter event with bar reaction features."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    end = _parse_end_date(end_date)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Building {len(symbols)} event-swing datasets with {max_workers} worker(s).")

    def run_symbol(symbol: str) -> dict:
        path = raw_dir / f"{symbol}_events.parquet"
        if not path.exists():
            return {"ticker": symbol, "rows": 0, "error": f"missing {path}"}
        try:
            dataset = build_event_swing_dataset(
                symbol,
                path,
                settings,
                horizon_days=horizon_days,
                market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                end=end,
            )
            out = out_dir / f"{symbol}_event_swing.parquet"
            dataset.to_parquet(out, index=False)
            return {
                "ticker": symbol,
                "rows": len(dataset),
                "path": str(out),
                "label_1d_rows": int(dataset["target_next_1d_up"].notna().sum()) if "target_next_1d_up" in dataset else 0,
                f"label_{horizon_days}d_rows": int(dataset[f"target_next_{horizon_days}d_up"].notna().sum())
                if f"target_next_{horizon_days}d_up" in dataset
                else 0,
            }
        except Exception as exc:
            return {"ticker": symbol, "rows": 0, "error": str(exc)}

    summary = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            record = future.result()
            summary.append(record)
            if record.get("error"):
                console.print(f"[red]{symbol}: event dataset build failed: {record['error']}[/red]")
            else:
                console.print(f"{symbol}: wrote {record['rows']} event rows to {record['path']}")
    pd.DataFrame(summary).to_csv(out_dir / "_event_dataset_summary.csv", index=False)


@app.command("combine-event-swing-datasets")
def combine_event_swing_datasets(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(
        Path("data/features/event_swing_2y_clean"),
        help="Directory containing per-ticker event-level datasets.",
    ),
    out: Path = typer.Option(
        Path("data/features/event_swing_combined_2y_clean.parquet"),
        help="Combined output parquet.",
    ),
) -> None:
    """Combine per-ticker event-swing datasets for event-level model training."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    frames = []
    summary = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_event_swing.parquet"
        if not path.exists():
            summary.append({"ticker": symbol, "rows": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            frames.append(frame)
            summary.append({"ticker": symbol, "rows": len(frame), "path": str(path)})
        except Exception as exc:
            summary.append({"ticker": symbol, "rows": 0, "error": str(exc)})
    if not frames:
        raise typer.BadParameter("No event-swing datasets found to combine.")
    combined = pd.concat(frames, ignore_index=True).sort_values(["event_timestamp", "ticker"]).reset_index(drop=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    pd.DataFrame(summary).to_csv(out.with_suffix(".summary.csv"), index=False)
    console.print(f"Wrote {len(combined)} event rows across {len(frames)} tickers to {out}")


@app.command("train-event-swing")
def train_event_swing(
    dataset: Path = typer.Option(..., help="Combined event-swing dataset parquet."),
    model_out: Path = typer.Option(Path("models/event_swing.joblib"), help="Output model path."),
    target_col: str = typer.Option("target_next_1d_up", help="Target column to train."),
    include_reaction_features: bool = typer.Option(
        True,
        help="Use post-event 1h/2h/4h reaction features. Disable for pre-reaction prediction.",
    ),
    max_iter: int = typer.Option(250, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.04, help="Boosting learning rate."),
) -> None:
    """Train an event-level continuation/fade model."""
    frame = pd.read_parquet(dataset)
    report = train_event_swing_model(
        frame,
        model_out,
        target_col=target_col,
        include_reaction_features=include_reaction_features,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )
    console.print(report)
    console.print(f"Wrote event-swing model to {model_out}")


@app.command("score-event-swing")
def score_event_swing(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    feature_dir: Path = typer.Option(
        Path("data/features/event_swing_2y_clean"),
        help="Directory containing per-ticker event-level datasets.",
    ),
    model: Path = typer.Option(
        Path("models/event_swing_2y_market_context_1d_prereaction_max.joblib"),
        help="Event-swing model path.",
    ),
    days: int = typer.Option(14, help="Recent calendar-day event lookback."),
    min_probability: float = typer.Option(0.0, help="Optional minimum probability_up filter."),
    out: Path = typer.Option(Path("data/reports/event_swing_scores_latest.csv"), help="Output scored event CSV."),
) -> None:
    """Score recent event/news rows for continuation/fade watch decisions."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    cutoff = pd.Timestamp.now(tz="UTC") - pd.Timedelta(days=days)
    rows = []
    for symbol in symbols:
        path = feature_dir / f"{symbol}_event_swing.parquet"
        if not path.exists():
            continue
        try:
            dataset = pd.read_parquet(path)
            dataset["event_timestamp"] = pd.to_datetime(dataset["event_timestamp"], utc=True)
            recent = dataset[dataset["event_timestamp"] >= cutoff].copy()
            if recent.empty:
                recent = dataset.sort_values("event_timestamp").tail(3).copy()
            scored = predict_event_swing_frame(recent, model)
            rows.append(scored)
        except Exception as exc:
            rows.append(pd.DataFrame([{"ticker": symbol, "error": str(exc)}]))
    if not rows:
        raise typer.BadParameter("No event-swing datasets found to score.")
    frame = pd.concat(rows, ignore_index=True)
    if "model_probability_up" in frame.columns and min_probability > 0:
        frame = frame[frame["model_probability_up"] >= min_probability].copy()
    sort_cols = [col for col in ["model_probability_up", "event_timestamp"] if col in frame.columns]
    if sort_cols:
        frame = frame.sort_values(sort_cols, ascending=[False, False][: len(sort_cols)])
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(out, index=False)
    display_cols = [
        col
        for col in [
            "ticker",
            "event_timestamp",
            "source",
            "event_time_bucket",
            "model_probability_up",
            "model_prediction",
            "sentiment_numeric",
            "reaction_2h",
            "next_1d_return",
            "title",
            "error",
        ]
        if col in frame.columns
    ]
    console.print(frame[display_cols].head(40))
    console.print(f"Wrote scored event-swing rows to {out}")


@app.command("predict-watchlist")
def predict_watchlist(
    tickers: str = typer.Option(..., help="Comma-separated symbols to analyze."),
    days: int = typer.Option(3, help="Fresh news/chatter lookback window in calendar days."),
    out: Path = typer.Option(Path("data/reports/watchlist_predictions_latest.csv"), help="Output prediction CSV."),
    daily_model_1d: Path = typer.Option(
        Path("models/daily_swing_2y_market_context_1d_max.joblib"),
        help="Market-context daily 1-day model path.",
    ),
    daily_model_5d: Path = typer.Option(
        Path("models/daily_swing_2y_market_context_5d_max.joblib"),
        help="Market-context daily 5-day model path.",
    ),
    event_model_1d: Path = typer.Option(
        Path("models/event_swing_2y_market_context_1d_prereaction_max.joblib"),
        help="Clean event-level 1-day model path.",
    ),
    event_model_5d: Path = typer.Option(
        Path("models/event_swing_2y_market_context_5d_prereaction_max.joblib"),
        help="Clean event-level 5-day model path.",
    ),
    no_reddit: bool = typer.Option(False, help="Disable Reddit collection."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha collection."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing collection."),
    no_profile: bool = typer.Option(False, help="Skip yfinance market-cap lookup."),
) -> None:
    """Fetch latest catalysts for a ticker list and score clean swing models."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    symbols = _parse_tickers(tickers, [])
    if not symbols:
        raise typer.BadParameter("Provide at least one ticker.")

    resolved_daily_1d = _model_path(daily_model_1d, Path("models/uslisted_6sector_direction_2y_clean_1d.joblib"))
    resolved_daily_5d = _model_path(daily_model_5d, Path("models/uslisted_6sector_direction_2y_clean_5d.joblib"))
    resolved_event_1d = _model_path(event_model_1d, Path("models/event_swing_2y_clean_1d_prereaction.joblib"))
    resolved_event_5d = _model_path(event_model_5d, Path("models/event_swing_2y_clean_5d_prereaction.joblib"))

    staged: list[pd.DataFrame] = []
    errors_by_symbol: dict[str, list[str]] = {}
    for symbol in symbols:
        frame, errors = collect_events_for_ticker(
            symbol,
            days,
            no_reddit=no_reddit,
            no_seeking_alpha=no_seeking_alpha,
            no_sec=no_sec,
            score=False,
        )
        if not frame.empty:
            frame["_watch_symbol"] = symbol
            staged.append(frame)
        errors_by_symbol[symbol] = errors

    scored_by_symbol: dict[str, pd.DataFrame] = {}
    if staged:
        combined = pd.concat(staged, ignore_index=True)
        scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
        combined = add_finbert_with_scorer(combined, scorer, batch_size=settings.finbert_batch_size)
        for symbol, frame in combined.groupby("_watch_symbol", sort=False):
            scored_by_symbol[symbol] = frame.drop(columns=["_watch_symbol"], errors="ignore")

    tmp_dir = Path("data/tmp/watchlist")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        events = scored_by_symbol.get(symbol, pd.DataFrame())
        profile = {} if no_profile else _lookup_market_profile(symbol)
        price = _latest_price_snapshot(symbol, settings)
        sector = settings.sector_for_ticker(symbol) or "unknown"
        sector_benchmark = settings.sector_benchmark_for_ticker(symbol)
        record: dict[str, object] = {
            "ticker": symbol,
            "lookback_days": days,
            "sector": sector,
            "sector_benchmark": sector_benchmark,
            "market_benchmark": settings.market_benchmark_ticker,
            "recent_event_count": len(events),
            "recent_headlines": _recent_headlines(events, limit=5),
            "errors": " | ".join(errors_by_symbol.get(symbol, [])),
            **profile,
            **price,
        }
        if not events.empty:
            events["source_family"] = events["source"].astype(str).str.split(":").str[0]
            source_counts = events["source_family"].value_counts()
            for source_name in ["alpaca", "reddit", "seeking_alpha", "sec"]:
                record[f"{source_name}_event_count"] = int(source_counts.get(source_name, 0))
            record["sentiment_mean_recent"] = float(pd.to_numeric(events["sentiment_numeric"], errors="coerce").mean())
            events_path = tmp_dir / f"{symbol}_events.parquet"
            events.to_parquet(events_path, index=False)
            try:
                daily_1d = build_daily_dataset(
                    symbol,
                    events_path,
                    settings,
                    horizon_days=1,
                    market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                )
                latest_daily = daily_1d.sort_values("date").iloc[-1]
                record["watch_score"] = heuristic_watch_score(daily_1d, weights=settings.watch_score_weights).get("watch_score")
                record["daily_return_1d"] = float(latest_daily.get("return_1d", 0) or 0)
                record["volume_z20"] = float(latest_daily.get("volume_z20", 0) or 0)
                if resolved_daily_1d:
                    pred = predict_latest(daily_1d, resolved_daily_1d)
                    record["daily_model_1d_probability_up"] = pred["probability_up"]
                    record["daily_model_1d_prediction"] = pred["prediction"]
                if resolved_daily_5d:
                    daily_5d = build_daily_dataset(
                        symbol,
                        events_path,
                        settings,
                        horizon_days=5,
                        market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                    )
                    pred = predict_latest(daily_5d, resolved_daily_5d)
                    record["daily_model_5d_probability_up"] = pred["probability_up"]
                    record["daily_model_5d_prediction"] = pred["prediction"]
            except Exception as exc:
                record["daily_model_error"] = str(exc)
            try:
                event_dataset = build_event_swing_dataset(
                    symbol,
                    events_path,
                    settings,
                    horizon_days=5,
                    market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                )
                if not event_dataset.empty and resolved_event_1d:
                    scored = predict_event_swing_frame(event_dataset, resolved_event_1d)
                    record["event_model_1d_latest_probability_up"] = float(
                        scored.sort_values("event_timestamp").iloc[-1]["model_probability_up"]
                    )
                    record["event_model_1d_max_probability_up"] = float(scored["model_probability_up"].max())
                    record["event_model_1d_mean_probability_up"] = float(scored["model_probability_up"].mean())
                if not event_dataset.empty and resolved_event_5d:
                    scored = predict_event_swing_frame(event_dataset, resolved_event_5d)
                    record["event_model_5d_latest_probability_up"] = float(
                        scored.sort_values("event_timestamp").iloc[-1]["model_probability_up"]
                    )
                    record["event_model_5d_max_probability_up"] = float(scored["model_probability_up"].max())
                    record["event_model_5d_mean_probability_up"] = float(scored["model_probability_up"].mean())
            except Exception as exc:
                record["event_model_error"] = str(exc)
        else:
            record["status"] = "no_recent_events"

        probability_inputs = [
            record.get("daily_model_1d_probability_up"),
            record.get("daily_model_5d_probability_up"),
            record.get("event_model_1d_latest_probability_up"),
            record.get("event_model_5d_latest_probability_up"),
        ]
        usable = [float(value) for value in probability_inputs if value is not None and not pd.isna(value)]
        record["combined_probability_up"] = sum(usable) / len(usable) if usable else None
        record["signal"] = _signal_from_probability(record["combined_probability_up"])
        rows.append(record)

    report = pd.DataFrame(rows).sort_values(
        ["combined_probability_up", "recent_event_count"],
        ascending=[False, False],
        na_position="last",
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    raw_out = out.with_name(f"{out.stem}_raw{out.suffix}")
    dictionary_out = out.with_name(f"{out.stem}_fields{out.suffix}")
    report.to_csv(raw_out, index=False)
    human_report = _humanize_watchlist_report(report)
    human_report.to_csv(out, index=False)
    _write_watchlist_dictionary(dictionary_out)
    display_cols = [
        col
        for col in [
            "Ticker",
            "Overall Signal",
            "Combined Probability Up",
            "Daily Model: Next-Day Up Probability",
            "Daily Model: Next-5-Trading-Days Up Probability",
            "Latest Event Model: Next-Day Up Probability",
            "Latest Event Model: Next-5-Trading-Days Up Probability",
            "Recent Catalyst Count",
            "Reddit Chatter Count",
            "Latest Close",
            "Latest 1-Day Return",
            "Market-Cap Bucket",
            "Configured Sector",
            "Recent Headlines Used",
            "Collection or Scoring Notes",
        ]
        if col in human_report.columns
    ]
    console.print(human_report[display_cols])
    console.print(f"Wrote readable watchlist predictions to {out}")
    console.print(f"Wrote raw watchlist predictions to {raw_out}")
    console.print(f"Wrote field definitions to {dictionary_out}")


@app.command("export-ohlcv-artifacts")
def export_ohlcv_artifacts(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(730, help="Calendar days of bars to export."),
    timeframes: str = typer.Option("1d,1h", help="Comma-separated timeframes: 1d,1h,5m,1m."),
    out_dir: Path = typer.Option(Path("data/artifacts/ohlcv"), help="Local OHLCV artifact output root."),
    workers: int | None = typer.Option(None, help="Parallel export workers."),
) -> None:
    """Export project-owned OHLCV parquet artifacts for this ML pipeline."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    requested = {item.strip().lower() for item in timeframes.split(",") if item.strip()}
    valid = {"1d", "1h", "5m", "1m"}
    unknown = requested - valid
    if unknown:
        raise typer.BadParameter(f"Unsupported timeframes: {sorted(unknown)}")
    start = datetime.now(timezone.utc) - timedelta(days=days)
    end = datetime.now(timezone.utc)
    out_dir.mkdir(parents=True, exist_ok=True)
    max_workers = _worker_count(workers, settings.max_workers, len(symbols))

    def run_symbol(symbol: str) -> list[dict[str, object]]:
        rows = []
        if "1d" in requested:
            daily = fetch_daily_prices(symbol, start, end, settings)
            normalized = _normalize_ohlcv(symbol, daily, "1d")
            path = out_dir / "1d" / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(path, index=False)
            rows.append({"ticker": symbol, "timeframe": "1d", "rows": len(normalized), "path": str(path)})
        for timeframe in ["1h", "5m", "1m"]:
            if timeframe not in requested:
                continue
            intraday = fetch_intraday_prices(symbol, start, end, settings, timeframe=timeframe)
            normalized = _normalize_ohlcv(symbol, intraday, timeframe)
            path = out_dir / timeframe / f"{symbol}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            normalized.to_parquet(path, index=False)
            rows.append({"ticker": symbol, "timeframe": timeframe, "rows": len(normalized), "path": str(path)})
        return rows

    summary = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(run_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            symbol = futures[future]
            try:
                rows = future.result()
                summary.extend(rows)
                console.print(f"{symbol}: exported {sum(int(row['rows']) for row in rows)} OHLCV rows")
            except Exception as exc:
                summary.append({"ticker": symbol, "error": str(exc)})
                console.print(f"[red]{symbol}: OHLCV export failed: {exc}[/red]")
    summary_frame = pd.DataFrame(summary)
    summary_path = out_dir / "_ohlcv_manifest.csv"
    summary_frame.to_csv(summary_path, index=False)
    contract = {
        "schema_version": "ohlcv.v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "columns": ["symbol", "timeframe", "timestamp", "open", "high", "low", "close", "volume", "source", "adjustment", "ingested_at_utc"],
        "timeframes": sorted(requested),
        "source": "alpaca",
        "adjustment": "all",
        "manifest": str(summary_path),
    }
    _write_artifact_manifest(out_dir / "_schema.json", contract)
    console.print(f"Wrote OHLCV manifest to {summary_path}")


@app.command("azure-upload-artifacts")
def azure_upload_artifacts(
    root: Path = typer.Option(Path("data/artifacts"), help="Local artifact root to upload."),
    blob_prefix: str = typer.Option("", help="Blob prefix under AZURE_BLOB_PREFIX. Defaults to local root name."),
    patterns: str = typer.Option("*.parquet,*.csv,*.json,*.joblib", help="Comma-separated glob patterns."),
) -> None:
    """Upload project artifacts to the configured Azure Blob container."""
    settings = get_settings()
    store = AzureBlobStore(settings)
    prefix = blob_prefix.strip("/") or root.name
    pattern_values = [item.strip() for item in patterns.split(",") if item.strip()]
    uploaded = store.upload_tree(root, blob_prefix=prefix, patterns=pattern_values)
    console.print(f"Uploaded {len(uploaded)} files to container {settings.azure_storage_container}/{settings.azure_prefix}/{prefix}")
    console.print(pd.DataFrame(uploaded).tail(20) if uploaded else "No files uploaded.")


@app.command("azure-publish-models")
def azure_publish_models(
    models_dir: Path = typer.Option(Path("models"), help="Local active models directory."),
    blob_prefix: str = typer.Option("models/active", help="Blob prefix for active models."),
) -> None:
    """Publish active clean model artifacts and a manifest to Azure Blob."""
    settings = get_settings()
    manifest_rows = []
    for path in sorted(models_dir.glob("*.joblib")):
        manifest_rows.append(
            {
                "name": path.name,
                "relative_blob": f"{blob_prefix}/{path.name}",
                "bytes": path.stat().st_size,
                "modified_utc": datetime.fromtimestamp(path.stat().st_mtime, tz=timezone.utc).isoformat(),
            }
        )
    if not manifest_rows:
        raise typer.BadParameter(f"No .joblib models found in {models_dir}")
    manifest = pd.DataFrame(manifest_rows)
    manifest_path = models_dir / "_active_models_manifest.csv"
    manifest.to_csv(manifest_path, index=False)
    store = AzureBlobStore(settings)
    uploaded = []
    for row in manifest_rows:
        uploaded.append(store.upload_file(models_dir / str(row["name"]), str(row["relative_blob"])))
    uploaded.append(store.upload_file(manifest_path, f"{blob_prefix}/_active_models_manifest.csv"))
    console.print({"uploaded_models": len(manifest_rows), "manifest": f"{settings.azure_prefix}/{blob_prefix}/_active_models_manifest.csv"})


def _alert_config(
    min_score: float,
    volume_confirm_z: float,
    strong_volume_z: float,
    rsi_oversold: float,
    rsi_overbought: float,
) -> AlertConfig:
    return AlertConfig(
        min_score=min_score,
        volume_confirm_z=volume_confirm_z,
        strong_volume_z=strong_volume_z,
        rsi_oversold=rsi_oversold,
        rsi_overbought=rsi_overbought,
    )


def _collect_latest_indicator_alerts(
    symbols: list[str],
    *,
    days: int,
    settings: object,
    config: AlertConfig,
) -> pd.DataFrame:
    start = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    for symbol in symbols:
        prices = fetch_daily_prices(symbol, start, None, settings)
        if prices.empty:
            rows.append(
                {
                    "ticker": symbol,
                    "date": None,
                    "alert_type": "no_price_data",
                    "direction": "none",
                    "severity": "low",
                    "score": 0.0,
                    "close": None,
                    "volume_z20": None,
                    "rsi_14": None,
                    "macd_signal_diff": None,
                    "details": "No daily price bars available for alert evaluation.",
                }
            )
            continue
        features = add_price_features(prices)
        features["ticker"] = symbol
        alerts = generate_indicator_alerts(features, config=config, latest_only=True)
        if alerts.empty:
            latest = features.sort_values("date").iloc[-1]
            rows.append(
                {
                    "ticker": symbol,
                    "date": latest.get("date"),
                    "alert_type": "no_active_alert",
                    "direction": "none",
                    "severity": "low",
                    "score": 0.0,
                    "close": latest.get("close"),
                    "volume_z20": latest.get("volume_z20"),
                    "rsi_14": latest.get("rsi_14"),
                    "macd_signal_diff": latest.get("macd_signal_diff"),
                    "details": "No configured indicator alert fired on the latest daily bar.",
                }
            )
        else:
            rows.extend(alerts.to_dict("records"))
    return pd.DataFrame(rows)


@app.command("monitor-alerts")
def monitor_alerts(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(180, help="Daily bar lookback for indicator warm-up."),
    poll_seconds: int = typer.Option(0, help="Repeat every N seconds. Use 0 to run once."),
    out: Path = typer.Option(Path("data/live/alerts/latest_alerts.csv"), help="Latest alert CSV."),
    history_out: Path = typer.Option(Path("data/live/alerts/alert_history.csv"), help="Append-only alert history CSV."),
    min_score: float = typer.Option(2.0, help="Minimum alert score to emit."),
    volume_confirm_z: float = typer.Option(0.75, help="Volume z-score that confirms a move."),
    strong_volume_z: float = typer.Option(1.5, help="Volume z-score for high-conviction confirmation."),
    rsi_oversold: float = typer.Option(30.0, help="RSI oversold threshold."),
    rsi_overbought: float = typer.Option(70.0, help="RSI overbought threshold."),
) -> None:
    """Monitor a watchlist for MACD/EMA/RSI/volume technical alerts."""
    if poll_seconds and poll_seconds < 60:
        raise typer.BadParameter("poll_seconds must be 0 or at least 60 to avoid API abuse.")
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    if not symbols:
        raise typer.BadParameter("No tickers configured or supplied.")
    config = _alert_config(min_score, volume_confirm_z, strong_volume_z, rsi_oversold, rsi_overbought)

    while True:
        run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
        alerts = _collect_latest_indicator_alerts(symbols, days=days, settings=settings, config=config)
        alerts.insert(0, "run_id", run_id)
        alerts.insert(1, "checked_at_utc", datetime.now(timezone.utc).isoformat())
        out.parent.mkdir(parents=True, exist_ok=True)
        history_out.parent.mkdir(parents=True, exist_ok=True)
        alerts.to_csv(out, index=False)
        append_history = history_out.exists()
        alerts.to_csv(history_out, mode="a" if append_history else "w", header=not append_history, index=False)
        display_cols = [
            col
            for col in ["ticker", "date", "alert_type", "direction", "severity", "score", "close", "volume_z20", "details"]
            if col in alerts.columns
        ]
        console.print(alerts[display_cols].sort_values(["severity", "score"], ascending=[True, False]).head(80))
        console.print(f"Wrote latest alerts to {out}")
        console.print(f"Appended alert history to {history_out}")
        if not poll_seconds:
            return
        time_module.sleep(poll_seconds)


@app.command("backtest-alerts")
def backtest_alerts(
    dataset: Path = typer.Option(
        Path("data/features/largecap_50b_news_volume_combined_2y_20260630_1d.parquet"),
        help="Historical feature parquet with ticker/date and future-return labels.",
    ),
    horizon_days: int = typer.Option(1, help="Forward return horizon to evaluate."),
    tickers: str | None = typer.Option(None, help="Optional comma-separated ticker subset."),
    out: Path = typer.Option(Path("data/reports/indicator_alert_backtest_alerts.csv"), help="Per-alert backtest rows."),
    summary_out: Path = typer.Option(Path("data/reports/indicator_alert_backtest_summary.csv"), help="Grouped backtest summary."),
    min_score: float = typer.Option(2.0, help="Minimum alert score to include."),
    volume_confirm_z: float = typer.Option(0.75, help="Volume z-score that confirms a move."),
    strong_volume_z: float = typer.Option(1.5, help="Volume z-score for high-conviction confirmation."),
    rsi_oversold: float = typer.Option(30.0, help="RSI oversold threshold."),
    rsi_overbought: float = typer.Option(70.0, help="RSI overbought threshold."),
) -> None:
    """Backtest MACD/EMA/RSI/volume alerts against historical forward returns."""
    if horizon_days == 5 and str(dataset).endswith("_1d.parquet"):
        fallback = Path(str(dataset).replace("_1d.parquet", "_5d.parquet"))
        if fallback.exists():
            dataset = fallback
    if not dataset.exists():
        raise typer.BadParameter(f"Missing dataset: {dataset}")
    frame = pd.read_parquet(dataset)
    symbols = _parse_tickers(tickers, []) if tickers else []
    if symbols:
        frame = frame[frame["ticker"].astype(str).str.upper().isin(symbols)].copy()
    future_col = f"future_return_{horizon_days}d"
    target_col = f"target_up_{horizon_days}d"
    missing = [col for col in ["ticker", "date", "close", future_col, target_col] if col not in frame.columns]
    if missing:
        raise typer.BadParameter(f"Dataset is missing required columns: {', '.join(missing)}")
    config = _alert_config(min_score, volume_confirm_z, strong_volume_z, rsi_oversold, rsi_overbought)
    alerts, summary = backtest_indicator_alerts(frame, horizon_days=horizon_days, config=config)
    out.parent.mkdir(parents=True, exist_ok=True)
    summary_out.parent.mkdir(parents=True, exist_ok=True)
    alerts.to_csv(out, index=False)
    summary.to_csv(summary_out, index=False)
    console.print(summary.head(30))
    console.print(f"Wrote alert backtest rows to {out}")
    console.print(f"Wrote alert backtest summary to {summary_out}")


@app.command("build-volatile-dataset")
def build_volatile_dataset_command(
    daily_1d: Path = typer.Option(
        Path("data/features/largecap_50b_news_volume_combined_2y_20260630_1d.parquet"),
        help="Daily feature parquet containing 1-day forward labels.",
    ),
    daily_5d: Path = typer.Option(
        Path("data/features/largecap_50b_news_volume_combined_2y_20260630_5d.parquet"),
        help="Daily feature parquet containing 5-day forward labels.",
    ),
    universe: Path = typer.Option(
        Path("data/universe/volatile_mover_research_universe_20260704.csv"),
        help="Volatile mover universe CSV with ticker/theme metadata.",
    ),
    out: Path = typer.Option(
        Path("data/features/volatile_mover_daily_20260704.parquet"),
        help="Output volatile mover training dataset.",
    ),
    audit_out: Path = typer.Option(
        Path("data/reports/volatile_mover_dataset_audit_20260704.csv"),
        help="Per-ticker readiness/audit CSV.",
    ),
    min_rows_per_ticker: int = typer.Option(120, help="Minimum historical daily rows per ticker."),
    min_news_rows_per_ticker: int = typer.Option(3, help="Minimum rows with ticker-linked news/catalysts per ticker."),
    next_day_big_move_threshold: float = typer.Option(0.03, help="Absolute 1-day return threshold for big-move labels."),
    next_week_big_move_threshold: float = typer.Option(0.08, help="Absolute 5-day return threshold for big-move labels."),
) -> None:
    """Build a news+volume+technical volatile mover dataset with auditable labels."""
    if not daily_1d.exists():
        raise typer.BadParameter(f"Missing 1-day dataset: {daily_1d}")
    one = pd.read_parquet(daily_1d)
    five = pd.read_parquet(daily_5d) if daily_5d.exists() else None
    universe_frame = load_volatile_universe(universe)
    config = VolatileLabelConfig(
        next_day_big_move_threshold=next_day_big_move_threshold,
        next_week_big_move_threshold=next_week_big_move_threshold,
        min_rows_per_ticker=min_rows_per_ticker,
        min_news_rows_per_ticker=min_news_rows_per_ticker,
    )
    dataset, audit = build_volatile_dataset(one, daily_5d=five, universe=universe_frame, config=config)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    audit.to_csv(audit_out, index=False)
    target_cols = [col for col in dataset.columns if col.startswith("target_next_")]
    summary = {
        "schema": "volatile_mover.v1",
        "rows": len(dataset),
        "tickers": int(dataset["ticker"].nunique()) if not dataset.empty else 0,
        "first_date": str(dataset["date"].min()) if not dataset.empty else None,
        "last_date": str(dataset["date"].max()) if not dataset.empty else None,
        "target_columns": target_cols,
        "out": str(out),
        "audit": str(audit_out),
    }
    console.print(summary)
    console.print(audit.sort_values(["model_eligible", "rows"], ascending=[True, False]).head(30))


@app.command("train-volatile-model")
def train_volatile_model_command(
    dataset: Path = typer.Option(
        Path("data/features/volatile_mover_daily_20260704.parquet"),
        help="Volatile mover dataset produced by build-volatile-dataset.",
    ),
    target_col: str = typer.Option("target_next_week_big_up", help="Target column to train."),
    model_out: Path = typer.Option(
        Path("models/volatile_mover_next_week_big_up_20260704_candidate.joblib"),
        help="Output model artifact.",
    ),
    predictions_out: Path = typer.Option(
        Path("data/reports/volatile_mover_next_week_big_up_oos_predictions_20260704.csv"),
        help="Out-of-sample prediction audit CSV.",
    ),
    metrics_out: Path = typer.Option(
        Path("data/reports/volatile_mover_next_week_big_up_metrics_20260704.csv"),
        help="Model metrics/model-card CSV.",
    ),
    max_iter: int = typer.Option(400, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.035, help="Boosting learning rate."),
    embargo_rows: int = typer.Option(5, help="Purged walk-forward embargo rows."),
) -> None:
    """Train a production-audited volatile mover classifier."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing volatile dataset: {dataset}")
    frame = pd.read_parquet(dataset)
    report, metrics, _ = train_volatile_model(
        frame,
        target_col=target_col,
        model_out=model_out,
        predictions_out=predictions_out,
        metrics_out=metrics_out,
        max_iter=max_iter,
        learning_rate=learning_rate,
        embargo_rows=embargo_rows,
    )
    console.print(report)
    console.print(metrics)


@app.command("audit-promotion-readiness")
def audit_promotion_readiness(
    dataset: Path = typer.Option(..., help="Feature dataset used by the candidate model."),
    predictions: Path = typer.Option(..., help="Out-of-sample predictions CSV from training."),
    target_col: str | None = typer.Option(None, help="Target column. Defaults to target_entry_success_* when available."),
    alignment_audit: Path | None = typer.Option(None, help="Optional existing news/candle alignment audit CSV."),
    out_prefix: Path = typer.Option(Path("data/reports/model_promotion_audit"), help="Output prefix for audit files."),
    probability_col: str = typer.Option("oos_probability", help="OOS probability column."),
    top_fraction: float = typer.Option(0.10, help="Top probability fraction to simulate as trades."),
    min_probability: float | None = typer.Option(None, help="Optional minimum probability floor for selected trades."),
    max_trades_per_period: int | None = typer.Option(
        None,
        help="Optional cap on selected trades per session/day for drawdown-aware selection.",
    ),
) -> None:
    """Build promotion audits for catalyst alignment, regime coverage, and OOS trade economics."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing dataset: {dataset}")
    if not predictions.exists():
        raise typer.BadParameter(f"Missing predictions CSV: {predictions}")
    frame = pd.read_parquet(dataset)
    prediction_frame = pd.read_csv(predictions)
    alignment_frame = pd.read_csv(alignment_audit) if alignment_audit is not None and alignment_audit.exists() else None
    summary, trades, regime_profit = build_walk_forward_profitability_audit(
        dataset=frame,
        predictions=prediction_frame,
        target_col=target_col,
        config=ProfitabilityAuditConfig(
            probability_col=probability_col,
            top_fraction=top_fraction,
            min_probability=min_probability,
            max_trades_per_period=max_trades_per_period,
        ),
    )
    regime = build_market_regime_audit(
        dataset=frame,
        predictions=prediction_frame,
        probability_col=probability_col,
        top_fraction=top_fraction,
    )
    catalyst = build_catalyst_news_audit(dataset=frame, alignment_audit=alignment_frame)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    profitability_out = out_prefix.with_name(out_prefix.name + "_profitability.csv")
    trades_out = out_prefix.with_name(out_prefix.name + "_selected_trades.csv")
    regime_out = out_prefix.with_name(out_prefix.name + "_regime.csv")
    regime_profit_out = out_prefix.with_name(out_prefix.name + "_regime_profitability.csv")
    catalyst_out = out_prefix.with_name(out_prefix.name + "_catalyst.csv")
    summary.to_csv(profitability_out, index=False)
    trades.to_csv(trades_out, index=False)
    regime.to_csv(regime_out, index=False)
    regime_profit.to_csv(regime_profit_out, index=False)
    catalyst.to_csv(catalyst_out, index=False)
    console.print({"profitability": str(profitability_out), "selected_trades": str(trades_out), "regime": str(regime_out), "catalyst": str(catalyst_out)})
    console.print(summary.iloc[0].to_dict())
    console.print(regime.iloc[0].to_dict())
    console.print(catalyst.iloc[0].to_dict())


@app.command("promote-model")
def promote_model_command(
    model: Path = typer.Option(..., help="Model artifact to promote."),
    metrics: Path = typer.Option(..., help="Metrics CSV produced by the model training command."),
    alignment_audit: Path | None = typer.Option(
        None,
        help="Alignment audit CSV from audit-swing-alignment. Required by default.",
    ),
    profitability_audit: Path | None = typer.Option(None, help="Profitability audit CSV from audit-promotion-readiness."),
    regime_audit: Path | None = typer.Option(None, help="Market-regime audit CSV from audit-promotion-readiness."),
    catalyst_audit: Path | None = typer.Option(None, help="Catalyst/news audit CSV from audit-promotion-readiness."),
    report_out: Path | None = typer.Option(
        None,
        help="Promotion/rejection JSON report. Defaults beside the model manifest.",
    ),
    min_roc_auc: float = typer.Option(0.65, help="Minimum out-of-sample ROC AUC."),
    min_top_decile_lift: float = typer.Option(2.0, help="Minimum top-decile lift."),
    min_validated_rows: int = typer.Option(20_000, help="Minimum out-of-sample validated rows."),
    min_tickers: int = typer.Option(200, help="Minimum distinct tickers in the training set."),
    max_alignment_errors: int = typer.Option(0, help="Maximum allowed alignment audit errors/mismatches."),
    require_alignment_audit: bool = typer.Option(True, help="Reject if alignment audit is not supplied."),
    min_selected_trades: int = typer.Option(100, help="Minimum selected OOS trades in profitability audit."),
    min_avg_trade_return: float = typer.Option(0.0, help="Minimum average return of selected OOS trades."),
    min_profit_factor: float = typer.Option(1.05, help="Minimum selected-trade profit factor."),
    max_strategy_drawdown: float = typer.Option(0.25, help="Maximum selected-trade cumulative drawdown."),
    min_return_drawdown_ratio: float = typer.Option(0.5, help="Minimum selected-trade cumulative return to drawdown ratio."),
    max_negative_period_rate: float = typer.Option(0.55, help="Maximum share of selected periods with negative average return."),
    require_profitability_audit: bool = typer.Option(True, help="Reject if profitability audit is not supplied."),
    min_regime_count: int = typer.Option(3, help="Minimum number of market regimes represented."),
    max_single_regime_share: float = typer.Option(0.85, help="Maximum training rows allowed in one market regime."),
    require_regime_audit: bool = typer.Option(True, help="Reject if regime audit is not supplied."),
    max_low_relevance_event_rate: float = typer.Option(0.25, help="Maximum low-relevance event rate in catalyst audit."),
    require_catalyst_audit: bool = typer.Option(True, help="Reject if catalyst/news audit is not supplied."),
) -> None:
    """Promote a candidate model only after production audit gates pass."""
    if not model.exists():
        raise typer.BadParameter(f"Missing model artifact: {model}")
    if not metrics.exists():
        raise typer.BadParameter(f"Missing metrics CSV: {metrics}")
    metric_frame = pd.read_csv(metrics)
    if metric_frame.empty:
        raise typer.BadParameter(f"Metrics CSV has no rows: {metrics}")
    metric_record = metric_frame.iloc[0].to_dict()
    audit_frame = None
    if alignment_audit is not None:
        if not alignment_audit.exists():
            raise typer.BadParameter(f"Missing alignment audit CSV: {alignment_audit}")
        audit_frame = pd.read_csv(alignment_audit)
    profitability_frame = read_audit_record(profitability_audit)
    regime_frame = read_audit_record(regime_audit)
    catalyst_frame = read_audit_record(catalyst_audit)
    if report_out is None:
        report_out = model.with_suffix(model.suffix + ".promotion_report.json")
    result = promote_model_manifest(
        model_path=model,
        metrics=metric_record,
        alignment_audit=audit_frame,
        profitability_audit=profitability_frame,
        regime_audit=regime_frame,
        catalyst_audit=catalyst_frame,
        min_roc_auc=min_roc_auc,
        min_top_decile_lift=min_top_decile_lift,
        min_validated_rows=min_validated_rows,
        min_tickers=min_tickers,
        max_alignment_errors=max_alignment_errors,
        require_alignment_audit=require_alignment_audit,
        min_selected_trades=min_selected_trades,
        min_avg_trade_return=min_avg_trade_return,
        min_profit_factor=min_profit_factor,
        max_strategy_drawdown=max_strategy_drawdown,
        min_return_drawdown_ratio=min_return_drawdown_ratio,
        max_negative_period_rate=max_negative_period_rate,
        require_profitability_audit=require_profitability_audit,
        min_regime_count=min_regime_count,
        max_single_regime_share=max_single_regime_share,
        require_regime_audit=require_regime_audit,
        max_low_relevance_event_rate=max_low_relevance_event_rate,
        require_catalyst_audit=require_catalyst_audit,
        report_path=report_out,
    )
    if not result["passed"]:
        console.print("[red]Model promotion rejected[/red]")
        console.print(result)
        raise typer.Exit(code=1)
    console.print("[green]Model promoted[/green]")
    console.print(result)


@app.command("score-volatile-latest")
def score_volatile_latest(
    dataset: Path = typer.Option(
        Path("data/features/volatile_mover_daily_20260704.parquet"),
        help="Volatile mover feature dataset.",
    ),
    model: Path = typer.Option(
        Path("models/volatile_mover_next_week_big_up_20260704_candidate.joblib"),
        help="Volatile mover model artifact.",
    ),
    tickers: str | None = typer.Option(None, help="Optional comma-separated ticker subset."),
    out: Path = typer.Option(
        Path("data/reports/volatile_mover_latest_scores_20260704.csv"),
        help="Latest per-ticker volatile model scores.",
    ),
) -> None:
    """Score the latest row per ticker with a volatile mover model."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing volatile dataset: {dataset}")
    if not model.exists():
        raise typer.BadParameter(f"Missing volatile model: {model}")
    frame = pd.read_parquet(dataset)
    symbols = _parse_tickers(tickers, []) if tickers else []
    if symbols:
        frame = frame[frame["ticker"].astype(str).str.upper().isin(symbols)].copy()
    latest = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    scored = score_volatile_frame(latest, model)
    scored = scored.sort_values("volatile_model_probability", ascending=False)
    keep = [
        col
        for col in [
            "ticker",
            "date",
            "theme_bucket",
            "close",
            "return_1d",
            "volume_z20",
            "news_count",
            "event_count",
            "sentiment_mean",
            "volatile_setup_score",
            "volatile_model_probability",
            "volatile_model_prediction",
            "volatile_model_target",
            "volatile_model_schema",
        ]
        if col in scored.columns
    ]
    out.parent.mkdir(parents=True, exist_ok=True)
    scored[keep].to_csv(out, index=False)
    console.print(scored[keep].head(50))
    console.print(f"Wrote volatile latest scores to {out}")


@app.command("score-flashpoints")
def score_flashpoints_command(
    events: Path = typer.Option(
        DEFAULT_MARKET_CONTEXT_PATH,
        help="Global/market-context events parquet or CSV.",
    ),
    out: Path = typer.Option(
        Path("data/reports/global_flashpoints_latest.csv"),
        help="Output flashpoint score CSV.",
    ),
    lookback_hours: int = typer.Option(48, help="Lookback window for flashpoint scoring."),
) -> None:
    """Score global flashpoint and commodity-channel risk from market-context news."""
    if not events.exists():
        raise typer.BadParameter(f"Missing events file: {events}")
    frame = pd.read_parquet(events) if events.suffix.lower() == ".parquet" else pd.read_csv(events)
    scored = score_flashpoints(frame, lookback_hours=lookback_hours)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    console.print(scored.head(30))
    console.print(f"Wrote flashpoint scores to {out}")


@app.command("monitor-sector-themes")
def monitor_sector_themes(
    dataset: Path = typer.Option(
        Path("data/features/sp500_6m_volatile_daily_20260708.parquet"),
        help="Volatile feature dataset with latest ticker rows.",
    ),
    universe: Path = typer.Option(
        Path("data/universe/sp500_current_20260708.csv"),
        help="Universe CSV with ticker, sector, industry, and company columns.",
    ),
    model: Path = typer.Option(
        Path("models/sp500_6m_next_week_big_up_v2_20260708_candidate.joblib"),
        help="Promoted volatile model artifact.",
    ),
    flashpoints: Path | None = typer.Option(
        None,
        help="Optional flashpoint CSV from score-flashpoints.",
    ),
    sector_out: Path = typer.Option(
        Path("data/reports/sector_theme_monitor_latest.csv"),
        help="Output sector/theme monitor CSV.",
    ),
    ticker_out: Path = typer.Option(
        Path("data/reports/sector_theme_monitor_tickers_latest.csv"),
        help="Output ticker-level monitor CSV.",
    ),
    allow_candidate_model: bool = typer.Option(False, help="Allow non-promoted model for research only."),
) -> None:
    """Monitor sectors/themes using the promoted model plus global flashpoint context."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing dataset: {dataset}")
    if not universe.exists():
        raise typer.BadParameter(f"Missing universe: {universe}")
    if not model.exists():
        raise typer.BadParameter(f"Missing model: {model}")
    feature_frame = pd.read_parquet(dataset) if dataset.suffix.lower() == ".parquet" else pd.read_csv(dataset)
    universe_frame = pd.read_csv(universe)
    flashpoint_frame = None
    if flashpoints is not None:
        if not flashpoints.exists():
            raise typer.BadParameter(f"Missing flashpoint CSV: {flashpoints}")
        flashpoint_frame = pd.read_csv(flashpoints)
    sector_report, ticker_report = build_sector_theme_monitor(
        dataset=feature_frame,
        universe=universe_frame,
        model_path=model,
        flashpoints=flashpoint_frame,
        require_promoted=not allow_candidate_model,
    )
    sector_out.parent.mkdir(parents=True, exist_ok=True)
    ticker_out.parent.mkdir(parents=True, exist_ok=True)
    sector_report.to_csv(sector_out, index=False)
    ticker_keep = [
        col
        for col in [
            "ticker",
            "date",
            "monitor_theme",
            "monitor_signal",
            "monitor_score",
            "volatile_model_probability",
            "global_net_impact",
            "global_positive_impact",
            "global_negative_impact",
            "volume_z20",
            "news_count",
            "event_count",
            "return_1d",
            "sector_return_1d",
            "rel_return_1d_vs_sector",
        ]
        if col in ticker_report.columns
    ]
    ticker_report[ticker_keep].to_csv(ticker_out, index=False)
    console.print(sector_report.head(30))
    console.print(ticker_report[ticker_keep].head(50))
    console.print(f"Wrote sector/theme monitor to {sector_out}")
    console.print(f"Wrote ticker monitor to {ticker_out}")


@app.command("build-entry-exit-dataset")
def build_entry_exit_dataset_command(
    input_path: Path = typer.Option(
        Path("data/features/volatile_mover_daily_20260704.parquet"),
        "--input",
        help="Input feature parquet/CSV with ticker, date, OHLCV, and optional news/model features.",
    ),
    out: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Output entry/exit labeled dataset.",
    ),
    audit_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_5b_audit_20260704.csv"),
        help="Per-ticker entry/exit readiness audit CSV.",
    ),
    horizon_bars: int = typer.Option(5, help="Number of bars after next open to evaluate target/stop path."),
    take_profit_atr: float = typer.Option(1.5, help="ATR multiple for target from next open."),
    stop_loss_atr: float = typer.Option(1.0, help="ATR multiple for stop from next open."),
    min_rows_per_ticker: int = typer.Option(120, help="Minimum rows per ticker."),
    min_labeled_rows_per_ticker: int = typer.Option(40, help="Minimum non-null path labels per ticker."),
    bar_kind: str = typer.Option("swing", help="Human label for bar type: swing, daily, hourly, 5min, etc."),
    allow_overnight: bool = typer.Option(False, help="Allow intraday path labels to cross session boundaries."),
) -> None:
    """Build leak-safe entry/exit path labels from OHLCV feature rows."""
    if not input_path.exists():
        raise typer.BadParameter(f"Missing input dataset: {input_path}")
    if input_path.is_dir():
        files = sorted(path for path in input_path.rglob("*.parquet") if not path.name.startswith("_"))
        if not files:
            raise typer.BadParameter(f"No parquet files found under input directory: {input_path}")
        frame = pd.concat((pd.read_parquet(path) for path in files), ignore_index=True)
    else:
        frame = pd.read_parquet(input_path) if input_path.suffix.lower() == ".parquet" else pd.read_csv(input_path)
    config = EntryExitLabelConfig(
        horizon_bars=horizon_bars,
        take_profit_atr=take_profit_atr,
        stop_loss_atr=stop_loss_atr,
        min_rows_per_ticker=min_rows_per_ticker,
        min_labeled_rows_per_ticker=min_labeled_rows_per_ticker,
        bar_kind=bar_kind,
        allow_overnight=allow_overnight,
    )
    dataset, audit = build_entry_exit_dataset(frame, config=config)
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    audit.to_csv(audit_out, index=False)
    suffix = f"{horizon_bars}b"
    summary = {
        "schema": "entry_exit.v1",
        "bar_kind": bar_kind,
        "allow_overnight": allow_overnight,
        "rows": len(dataset),
        "tickers": int(dataset["ticker"].nunique()) if not dataset.empty else 0,
        "first_date": str(dataset["date"].min()) if not dataset.empty else None,
        "last_date": str(dataset["date"].max()) if not dataset.empty else None,
        "target_columns": [
            f"target_entry_success_{suffix}",
            f"target_exit_risk_{suffix}",
            f"target_timeout_positive_{suffix}",
        ],
        "out": str(out),
        "audit": str(audit_out),
    }
    console.print(summary)
    console.print(audit.sort_values(["model_eligible", "rows"], ascending=[True, False]).head(30))


@app.command("train-entry-exit-model")
def train_entry_exit_model_command(
    dataset: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Entry/exit dataset produced by build-entry-exit-dataset.",
    ),
    target_col: str = typer.Option("target_entry_success_5b", help="Target column to train."),
    model_out: Path = typer.Option(
        Path("models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib"),
        help="Output model artifact.",
    ),
    predictions_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_entry_success_5b_oos_predictions_20260704.csv"),
        help="Out-of-sample prediction audit CSV.",
    ),
    metrics_out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_entry_success_5b_metrics_20260704.csv"),
        help="Model metrics/model-card CSV.",
    ),
    max_iter: int = typer.Option(350, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.04, help="Boosting learning rate."),
    embargo_rows: int | None = typer.Option(None, help="Purged walk-forward embargo rows. Defaults from target horizon."),
    feature_set: str = typer.Option("all", help="Feature ablation set: all, technical, or catalyst."),
) -> None:
    """Train an entry/exit path classifier."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing entry/exit dataset: {dataset}")
    frame = pd.read_parquet(dataset)
    report, metrics, _ = train_entry_exit_model(
        frame,
        target_col=target_col,
        model_out=model_out,
        predictions_out=predictions_out,
        metrics_out=metrics_out,
        max_iter=max_iter,
        learning_rate=learning_rate,
        embargo_rows=embargo_rows,
        feature_set=feature_set,
    )
    console.print(report)
    console.print(metrics)


@app.command("build-intraday-enriched-dataset")
def build_intraday_enriched_dataset_command(
    input_path: Path = typer.Option(..., "--input", help="5m entry/exit dataset parquet."),
    out: Path = typer.Option(..., help="Output enriched training parquet."),
    audit_out: Path = typer.Option(..., help="Output enrichment audit CSV."),
    candidates: Path | None = typer.Option(None, help="Optional Finviz intraday candidate CSV."),
    one_minute_dir: Path | None = typer.Option(None, help="Optional 1m OHLCV parquet directory."),
    benchmark_dir: Path | None = typer.Option(None, help="Optional 5m benchmark OHLCV directory containing QQQ/SPY."),
    event_dirs: str | None = typer.Option(None, help="Comma-separated event directories containing SYMBOL_events.parquet files."),
    market_context: Path | None = typer.Option(
        DEFAULT_MARKET_CONTEXT_PATH,
        help="Optional global market-context events parquet for intraday catalyst features.",
    ),
    setup_only: bool = typer.Option(True, help="Keep only rows passing setup-candidate filters."),
    min_setup_score: float = typer.Option(2.0, help="Minimum setup-candidate score when setup-only is true."),
) -> None:
    """Create setup-filtered, market-relative, 1m-confirmed intraday training rows."""
    if not input_path.exists():
        raise typer.BadParameter(f"Missing input dataset: {input_path}")
    frame = pd.read_parquet(input_path)
    candidate_frame = pd.read_csv(candidates) if candidates is not None and candidates.exists() else None
    enriched, audit = build_enriched_intraday_dataset(
        frame,
        candidates=candidate_frame,
        one_minute_dir=one_minute_dir,
        benchmark_dir=benchmark_dir,
        event_dirs=_parse_path_list(event_dirs),
        market_context_path=market_context,
        setup_only=setup_only,
        min_setup_score=min_setup_score,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    audit_out.parent.mkdir(parents=True, exist_ok=True)
    enriched.to_parquet(out, index=False)
    audit.to_csv(audit_out, index=False)
    summary = {
        "input_rows": len(frame),
        "input_tickers": int(frame["ticker"].nunique()) if "ticker" in frame.columns else 0,
        "output_rows": len(enriched),
        "output_tickers": int(enriched["ticker"].nunique()) if not enriched.empty else 0,
        "setup_only": setup_only,
        "min_setup_score": min_setup_score,
        "event_dirs": event_dirs,
        "market_context": str(market_context) if market_context else None,
        "target_entry_success_rate": float(pd.to_numeric(enriched.get("target_entry_success_12b"), errors="coerce").mean())
        if not enriched.empty and "target_entry_success_12b" in enriched.columns
        else None,
        "out": str(out),
        "audit": str(audit_out),
    }
    console.print(summary)
    console.print(audit.sort_values("rows", ascending=False).head(30))


@app.command("score-entry-exit-latest")
def score_entry_exit_latest(
    dataset: Path = typer.Option(
        Path("data/features/entry_exit_swing_5b_20260704.parquet"),
        help="Entry/exit feature dataset.",
    ),
    model: Path = typer.Option(
        Path("models/entry_exit_swing_entry_success_5b_20260704_candidate.joblib"),
        help="Entry/exit model artifact.",
    ),
    tickers: str | None = typer.Option(None, help="Optional comma-separated ticker subset."),
    out: Path = typer.Option(
        Path("data/reports/entry_exit_swing_latest_scores_20260704.csv"),
        help="Latest per-ticker entry/exit model scores.",
    ),
) -> None:
    """Score latest rows with an entry/exit model."""
    if not dataset.exists():
        raise typer.BadParameter(f"Missing entry/exit dataset: {dataset}")
    if not model.exists():
        raise typer.BadParameter(f"Missing entry/exit model: {model}")
    frame = pd.read_parquet(dataset)
    symbols = _parse_tickers(tickers, []) if tickers else []
    if symbols:
        frame = frame[frame["ticker"].astype(str).str.upper().isin(symbols)].copy()
    latest = frame.sort_values(["ticker", "date"]).groupby("ticker", as_index=False).tail(1)
    scored = score_entry_exit_frame(latest, model)
    probability_cols = [col for col in scored.columns if col.endswith("_probability")]
    sort_col = probability_cols[-1] if probability_cols else "ticker"
    scored = scored.sort_values(sort_col, ascending=False)
    out.parent.mkdir(parents=True, exist_ok=True)
    scored.to_csv(out, index=False)
    keep = [
        col
        for col in [
            "ticker",
            "date",
            "close",
            "return_1d",
            "volume_z20",
            "news_count",
            "event_count",
            "rsi_14",
            "macd_signal_diff",
            "entry_stop_pct",
            "entry_target_pct",
            sort_col,
            sort_col.replace("probability", "prediction"),
            "entry_exit_model_target",
            "entry_exit_model_schema",
        ]
        if col in scored.columns
    ]
    console.print(scored[keep].head(50))
    console.print(f"Wrote entry/exit latest scores to {out}")


@app.command("build-intraday-decision-report")
def build_intraday_decision_report_command(
    scores: Path = typer.Option(..., help="Latest 5m entry/exit score CSV."),
    one_minute_dir: Path = typer.Option(..., help="Directory containing 1m OHLCV parquet files."),
    candidates: Path | None = typer.Option(None, help="Optional Finviz intraday candidate CSV."),
    out: Path = typer.Option(Path("data/reports/intraday_decision_latest.csv"), help="Output decision report CSV."),
) -> None:
    """Merge 5m entry model scores with latest 1m confirmation features."""
    if not scores.exists():
        raise typer.BadParameter(f"Missing scores CSV: {scores}")
    if not one_minute_dir.exists():
        raise typer.BadParameter(f"Missing 1m directory: {one_minute_dir}")
    score_frame = pd.read_csv(scores)
    candidate_frame = pd.read_csv(candidates) if candidates is not None and candidates.exists() else None
    report = build_intraday_decision_report(
        scores=score_frame,
        one_minute_dir=one_minute_dir,
        candidates=candidate_frame,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    display_cols = [
        col
        for col in [
            "ticker",
            "intraday_decision",
            "entry_model_probability",
            "entry_model_rank",
            "one_minute_confirmation_signal",
            "one_minute_dist_vwap",
            "one_minute_return_15m",
            "one_minute_volume_burst_15m",
            "above_opening_range",
            "intraday_theme",
            "intraday_candidate_score",
        ]
        if col in report.columns
    ]
    console.print(report[display_cols].head(80))
    console.print(f"Wrote intraday decision report to {out}")


@app.command("live-once")
def live_once(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    live_dir: Path = typer.Option(Path("data/live"), help="Managed live-pipeline state directory."),
    lookback_days: int = typer.Option(3, help="News/chatter lookback window for each collection cycle."),
    workers: int | None = typer.Option(None, help="Parallel API collection workers."),
    score_sentiment: bool = typer.Option(True, help="Run FinBERT on newly collected/unscored events."),
    no_reddit: bool = typer.Option(False, help="Disable Reddit collection."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha collection."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing collection."),
    curate_training: bool = typer.Option(True, help="Refresh the labeled live training parquet after scoring."),
    out: Path | None = typer.Option(None, help="Optional run prediction summary CSV."),
) -> None:
    """Run one managed live-data cycle: collect, validate, curate, and score."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    if not symbols:
        raise typer.BadParameter("No tickers configured or supplied.")
    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    event_dir = live_dir / "events_scored"
    feature_dir = live_dir / "features"
    prediction_dir = live_dir / "predictions" / run_id
    curated_out = live_dir / "curated" / "event_swing_labeled.parquet"
    state_path = live_dir / "state.json"
    for directory in [event_dir, feature_dir, prediction_dir, curated_out.parent]:
        directory.mkdir(parents=True, exist_ok=True)

    max_workers = _worker_count(workers, settings.max_workers, len(symbols))
    console.print(f"Live cycle {run_id}: collecting {len(symbols)} tickers with {max_workers} worker(s).")

    def collect_symbol(symbol: str) -> dict[str, object]:
        try:
            frame, errors = collect_events_for_ticker(
                symbol,
                lookback_days,
                no_reddit=no_reddit,
                no_seeking_alpha=no_seeking_alpha,
                no_sec=no_sec,
                score=False,
            )
            return {"ticker": symbol, "frame": frame, "errors": errors}
        except Exception as exc:
            return {"ticker": symbol, "frame": pd.DataFrame(), "errors": [str(exc)]}

    collected = []
    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        futures = {pool.submit(collect_symbol, symbol): symbol for symbol in symbols}
        for future in as_completed(futures):
            record = future.result()
            collected.append(record)
            console.print(
                f"{record['ticker']}: collected {len(record['frame'])} events"
                + (f" ({' | '.join(record['errors'])})" if record["errors"] else "")
            )

    scorer = None
    if score_sentiment and any(len(record["frame"]) for record in collected):
        scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)

    event_model_1d = _model_path(Path("models/event_swing_2y_market_context_1d_prereaction_max.joblib"))
    event_model_5d = _model_path(Path("models/event_swing_2y_market_context_5d_prereaction_max.joblib"))
    daily_model_1d = _model_path(Path("models/daily_swing_2y_market_context_1d_max.joblib"))
    daily_model_5d = _model_path(Path("models/daily_swing_2y_market_context_5d_max.joblib"))
    rows = []
    collection_summary = []
    for record in collected:
        symbol = str(record["ticker"])
        frame = record["frame"]
        errors = list(record["errors"])
        event_path = event_dir / f"{symbol}_events.parquet"
        try:
            if not frame.empty:
                frame, verify = sanitize_events_frame(frame)
                if verify.rows_out and scorer:
                    frame = _score_unscored_events(frame, scorer, settings)
                store, added = _upsert_events(event_path, frame)
            elif event_path.exists():
                store = pd.read_parquet(event_path)
                if scorer:
                    store = _score_unscored_events(store, scorer, settings)
                    store.to_parquet(event_path, index=False)
                added = 0
            else:
                store = pd.DataFrame()
                added = 0
            collection_summary.append(
                {
                    "run_id": run_id,
                    "ticker": symbol,
                    "new_events_collected": len(frame),
                    "new_events_added_to_store": added,
                    "event_store_rows": len(store),
                    "errors": " | ".join(errors),
                }
            )
            if store.empty:
                rows.append(
                    {
                        "run_id": run_id,
                        "ticker": symbol,
                        "status": "no_events_in_store",
                        "signal": "quiet/no_recent_catalyst",
                        "sector": settings.sector_for_ticker(symbol) or "unknown",
                        "sector_benchmark": settings.sector_benchmark_for_ticker(symbol),
                        **_latest_price_snapshot(symbol, settings),
                        "errors": " | ".join(errors),
                    }
                )
                continue
            score = _score_live_ticker(
                symbol,
                event_path,
                feature_dir,
                prediction_dir,
                settings,
                run_id=run_id,
                lookback_days=lookback_days,
                event_model_1d=event_model_1d,
                event_model_5d=event_model_5d,
                daily_model_1d=daily_model_1d,
                daily_model_5d=daily_model_5d,
            )
            score["errors"] = " | ".join(errors)
            rows.append(score)
        except Exception as exc:
            rows.append({"run_id": run_id, "ticker": symbol, "status": "failed", "errors": " | ".join([*errors, str(exc)])})

    predictions = pd.DataFrame(rows).sort_values(
        ["combined_probability_up", "recent_event_rows"],
        ascending=[False, False],
        na_position="last",
    )
    prediction_out = out or (prediction_dir / "summary.csv")
    prediction_out.parent.mkdir(parents=True, exist_ok=True)
    predictions.to_csv(prediction_out, index=False)
    pd.DataFrame(collection_summary).to_csv(prediction_dir / "collection_summary.csv", index=False)

    curated = _curate_live_training_set(feature_dir, curated_out) if curate_training else {}
    state = {
        "last_run_id": run_id,
        "last_run_utc": datetime.now(timezone.utc).isoformat(),
        "tickers": symbols,
        "lookback_days": lookback_days,
        "prediction_summary": str(prediction_out),
        "collection_summary": str(prediction_dir / "collection_summary.csv"),
        "curated_training": curated,
        "models": {
            "event_1d": str(event_model_1d) if event_model_1d else None,
            "event_5d": str(event_model_5d) if event_model_5d else None,
            "daily_1d": str(daily_model_1d) if daily_model_1d else None,
            "daily_5d": str(daily_model_5d) if daily_model_5d else None,
        },
    }
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
    display_cols = [
        col
        for col in [
            "ticker",
            "signal",
            "combined_probability_up",
            "daily_1d_probability_up",
            "daily_5d_probability_up",
            "event_1d_latest_probability_up",
            "event_5d_latest_probability_up",
            "recent_event_rows",
            "latest_close",
            "latest_return_1d",
            "sector",
            "latest_headlines",
            "errors",
        ]
        if col in predictions.columns
    ]
    console.print(predictions[display_cols].head(50))
    console.print(f"Wrote live prediction summary to {prediction_out}")
    if curated:
        console.print({"curated_training": curated})
    console.print(f"Wrote live state to {state_path}")


@app.command("live-run")
def live_run(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    live_dir: Path = typer.Option(Path("data/live"), help="Managed live-pipeline state directory."),
    lookback_days: int = typer.Option(3, help="News/chatter lookback window per cycle."),
    poll_seconds: int = typer.Option(1800, help="Seconds to sleep between live cycles."),
    workers: int | None = typer.Option(None, help="Parallel API collection workers."),
    no_reddit: bool = typer.Option(False, help="Disable Reddit collection."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha collection."),
    no_sec: bool = typer.Option(False, help="Disable SEC filing collection."),
) -> None:
    """Keep collecting live data in a foreground process until stopped."""
    if poll_seconds < 60:
        raise typer.BadParameter("poll_seconds must be at least 60 to avoid API abuse.")
    console.print(f"Starting live pipeline. Poll interval: {poll_seconds}s. Stop with Ctrl+C.")
    while True:
        try:
            live_once(
                tickers=tickers,
                live_dir=live_dir,
                lookback_days=lookback_days,
                workers=workers,
                score_sentiment=True,
                no_reddit=no_reddit,
                no_seeking_alpha=no_seeking_alpha,
                no_sec=no_sec,
                curate_training=True,
                out=None,
            )
        except KeyboardInterrupt:
            console.print("Live pipeline stopped.")
            raise typer.Exit(0)
        except Exception as exc:
            console.print(f"[red]Live cycle failed: {exc}[/red]")
        time_module.sleep(poll_seconds)


@app.command("live-train-event")
def live_train_event(
    live_dir: Path = typer.Option(Path("data/live"), help="Managed live-pipeline state directory."),
    base_dataset: Path = typer.Option(
        Path("data/features/event_swing_combined_2y_clean.parquet"),
        help="Base historical event-swing training parquet.",
    ),
    min_live_rows: int = typer.Option(1000, help="Minimum matured live rows required before retraining."),
    min_accuracy: float = typer.Option(0.505, help="Minimum walk-forward accuracy required for promotion."),
    candidate_dir: Path | None = typer.Option(None, help="Optional candidate model output directory."),
    promote: bool = typer.Option(True, help="Promote candidates to active model names if validation passes."),
    max_iter: int = typer.Option(900, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.025, help="Boosting learning rate."),
) -> None:
    """Retrain event models from historical data plus matured live labels, with promotion gates."""
    curated = live_dir / "curated" / "event_swing_labeled.parquet"
    if not base_dataset.exists():
        raise typer.BadParameter(f"Missing base dataset: {base_dataset}")
    if not curated.exists():
        raise typer.BadParameter(f"Missing curated live dataset: {curated}. Run live-once first.")

    base = pd.read_parquet(base_dataset)
    live = pd.read_parquet(curated)
    if live.empty or len(live) < min_live_rows:
        state = {
            "status": "skipped",
            "reason": "not_enough_matured_live_rows",
            "live_rows": int(len(live)),
            "min_live_rows": min_live_rows,
            "base_rows": int(len(base)),
            "checked_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        out = live_dir / "training" / "last_train_state.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")
        console.print(state)
        return

    combined = pd.concat([base, live], ignore_index=True, sort=False)
    identity = [col for col in ["ticker", "event_timestamp", "source", "title"] if col in combined.columns]
    if identity:
        combined = combined.drop_duplicates(identity, keep="last")
    combined = combined.sort_values(["event_timestamp", "ticker"]).reset_index(drop=True)

    run_id = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    candidate_root = candidate_dir or (live_dir / "training" / run_id)
    candidate_root.mkdir(parents=True, exist_ok=True)
    combined_path = candidate_root / "combined_training.parquet"
    combined.to_parquet(combined_path, index=False)

    jobs = [
        {
            "name": "event_swing_2y_clean_1d_prereaction_max.joblib",
            "target_col": "target_next_1d_up",
            "include_reaction_features": False,
        },
        {
            "name": "event_swing_2y_clean_5d_prereaction_max.joblib",
            "target_col": "target_next_5d_up",
            "include_reaction_features": False,
        },
        {
            "name": "event_swing_2y_clean_1d_reaction_max.joblib",
            "target_col": "target_next_1d_up",
            "include_reaction_features": True,
        },
        {
            "name": "event_swing_2y_clean_5d_reaction_max.joblib",
            "target_col": "target_next_5d_up",
            "include_reaction_features": True,
        },
    ]
    metrics_rows = []
    reports = {}
    for job in jobs:
        model_out = candidate_root / str(job["name"])
        report, metrics = train_event_swing_model_with_metrics(
            combined,
            model_out,
            target_col=str(job["target_col"]),
            include_reaction_features=bool(job["include_reaction_features"]),
            max_iter=max_iter,
            learning_rate=learning_rate,
        )
        metrics["name"] = job["name"]
        metrics["passed"] = bool(float(metrics["accuracy"]) >= min_accuracy)
        metrics_rows.append(metrics)
        reports[str(job["name"])] = report
        (candidate_root / f"{Path(str(job['name'])).stem}_report.txt").write_text(report, encoding="utf-8")

    metrics_frame = pd.DataFrame(metrics_rows)
    metrics_path = candidate_root / "metrics.csv"
    metrics_frame.to_csv(metrics_path, index=False)
    all_passed = bool(metrics_frame["passed"].all())
    promoted = []
    if promote and all_passed:
        models_dir = Path("models")
        models_dir.mkdir(parents=True, exist_ok=True)
        for job in jobs:
            src = candidate_root / str(job["name"])
            dst = models_dir / str(job["name"])
            dst.write_bytes(src.read_bytes())
            promoted.append(str(dst))

    train_state = {
        "status": "promoted" if promoted else "trained_not_promoted",
        "run_id": run_id,
        "base_rows": int(len(base)),
        "live_rows": int(len(live)),
        "combined_rows": int(len(combined)),
        "min_live_rows": min_live_rows,
        "min_accuracy": min_accuracy,
        "candidate_dir": str(candidate_root),
        "combined_training": str(combined_path),
        "metrics": str(metrics_path),
        "all_passed": all_passed,
        "promoted_models": promoted,
        "checked_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    state_path = live_dir / "training" / "last_train_state.json"
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(train_state, indent=2, sort_keys=True), encoding="utf-8")
    console.print(metrics_frame[["name", "target_col", "include_reaction_features", "accuracy", "passed"]])
    console.print(train_state)


@app.command("build-dataset")
def build_dataset(
    ticker: str,
    events: Path = typer.Option(..., help="Input events parquet from collect."),
    out: Path = typer.Option(Path("data/features/daily.parquet"), help="Output dataset parquet."),
    horizon_days: int = typer.Option(5, help="Forward trading-day target horizon."),
    seeking_alpha: Path | None = typer.Option(None, help="Optional Seeking Alpha quant CSV export."),
    market_context: Path | None = typer.Option(DEFAULT_MARKET_CONTEXT_PATH, help="Optional global market-news context parquet."),
    end_date: str | None = typer.Option(None, help="Inclusive UTC end date as YYYY-MM-DD for price/features."),
) -> None:
    """Join events, prices, SEC facts, optional quant data, and labels."""
    settings = get_settings()
    end = _parse_end_date(end_date)
    dataset = build_daily_dataset(
        ticker,
        events,
        settings,
        horizon_days=horizon_days,
        seeking_alpha_path=seeking_alpha,
        market_context_path=market_context,
        end=end,
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_parquet(out, index=False)
    console.print(f"Wrote {len(dataset)} daily rows to {out}")


@app.command()
def train(
    dataset: Path = typer.Option(..., help="Daily dataset parquet."),
    model_out: Path = typer.Option(Path("models/direction.joblib"), help="Output model path."),
    horizon_days: int | None = typer.Option(None, help="Horizon used when building the dataset."),
    max_iter: int = typer.Option(200, help="Maximum boosting iterations."),
    learning_rate: float = typer.Option(0.05, help="Boosting learning rate."),
) -> None:
    """Train a baseline next-week direction classifier."""
    frame = pd.read_parquet(dataset)
    report = train_direction_model(
        frame,
        model_out,
        horizon_days=horizon_days,
        max_iter=max_iter,
        learning_rate=learning_rate,
    )
    console.print(report)
    console.print(f"Wrote model to {model_out}")


@app.command()
def predict(
    ticker: str,
    model: Path = typer.Option(..., help="Trained model path."),
    days: int = typer.Option(30, help="Recent collection window for latest feature row."),
) -> None:
    """Collect current data, rebuild recent features, and predict next-week direction."""
    settings = get_settings()
    tmp_events = Path("data/tmp") / f"{ticker.upper()}_events.parquet"
    tmp_dataset = Path("data/tmp") / f"{ticker.upper()}_daily.parquet"
    tmp_sa = Path("data/tmp") / f"{ticker.upper()}_seeking_alpha_quant.csv"
    sa_path = write_seeking_alpha_snapshot(ticker, tmp_sa) if settings.has_seeking_alpha_rapidapi else None
    collect(ticker, days=days, out=tmp_events, no_reddit=False)
    build_dataset(ticker, events=tmp_events, out=tmp_dataset, horizon_days=5, seeking_alpha=sa_path)
    result = predict_latest(pd.read_parquet(tmp_dataset), model)
    console.print(result)


@app.command("collect-seeking-alpha")
def collect_seeking_alpha(
    ticker: str,
    out: Path = typer.Option(
        Path("data/external/seeking_alpha_quant.csv"),
        help="Output CSV used by build-dataset --seeking-alpha.",
    ),
) -> None:
    """Collect a Seeking Alpha quant, earnings, and rating snapshot via RapidAPI."""
    write_seeking_alpha_snapshot(ticker, out)
    console.print(f"Wrote Seeking Alpha quant snapshot for {ticker.upper()} to {out}")


@app.command("collect-seeking-alpha-universe")
def collect_seeking_alpha_universe(
    tickers: str | None = typer.Option(None, help="Comma-separated symbols. Defaults to configured swing universe."),
    days: int = typer.Option(730, help="Event lookback window in calendar days."),
    out_dir: Path = typer.Option(Path("data/external/seeking_alpha_full"), help="Per-ticker output directory."),
    include_events: bool = typer.Option(True, help="Collect configured Seeking Alpha event feeds."),
    include_snapshots: bool = typer.Option(True, help="Collect configured Seeking Alpha snapshot feeds."),
) -> None:
    """Collect all configured Seeking Alpha event/snapshot feeds with per-ticker isolation."""
    settings = get_settings()
    symbols = _parse_tickers(tickers, settings.swing_candidate_tickers)
    if not symbols:
        raise typer.BadParameter("No tickers configured or supplied.")
    out_dir.mkdir(parents=True, exist_ok=True)
    start = datetime.now(timezone.utc) - timedelta(days=days)
    source = SeekingAlphaRapidApiSource(settings)
    summary_rows = []
    snapshot_rows = []
    for index, symbol in enumerate(symbols, start=1):
        row: dict[str, object] = {"ticker": symbol, "status": "ok", "events": 0, "snapshot_fields": 0}
        try:
            if include_events:
                events, event_errors = source.fetch_events_with_errors(symbol, start)
                event_frame = pd.DataFrame([event.to_record() for event in events])
                if "raw" in event_frame.columns:
                    event_frame["raw"] = event_frame["raw"].map(
                        lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, dict) else value
                    )
                event_path = out_dir / f"{symbol}_events.parquet"
                event_frame.to_parquet(event_path, index=False)
                row["events"] = len(event_frame)
                row["events_path"] = str(event_path)
                row["event_errors"] = " | ".join(event_errors)
            if include_snapshots:
                snapshot = source.fetch_quant_snapshot(symbol)
                snapshot_frame = pd.DataFrame([snapshot])
                snapshot_path = out_dir / f"{symbol}_snapshot.csv"
                snapshot_frame.to_csv(snapshot_path, index=False)
                snapshot_rows.append(snapshot)
                row["snapshot_fields"] = len(snapshot)
                row["snapshot_path"] = str(snapshot_path)
                row["snapshot_errors"] = " | ".join(
                    f"{key}={value}" for key, value in snapshot.items() if str(key).endswith("_error")
                )
                row["snapshot_skips"] = " | ".join(
                    f"{key}={value}" for key, value in snapshot.items() if str(key).endswith("_skipped")
                )
        except Exception as exc:
            row["status"] = "failed"
            row["error"] = str(exc)
        summary_rows.append(row)
        console.print(f"{index}/{len(symbols)} {symbol}: {row['status']} events={row.get('events')} fields={row.get('snapshot_fields')}")
    summary = pd.DataFrame(summary_rows)
    summary_path = out_dir / "_summary.csv"
    summary.to_csv(summary_path, index=False)
    if snapshot_rows:
        combined_path = out_dir / "_snapshots_combined.csv"
        pd.DataFrame(snapshot_rows).to_csv(combined_path, index=False)
        console.print(f"Wrote combined snapshots to {combined_path}")
    console.print(f"Wrote Seeking Alpha collection summary to {summary_path}")


@app.command("collect-market-context")
def collect_market_context(
    days: int = typer.Option(730, help="Market/global news lookback window in calendar days."),
    out: Path = typer.Option(
        Path("data/external/market_context/market_context_events.parquet"),
        help="Output market context events parquet.",
    ),
    score_sentiment: bool = typer.Option(True, help="Run FinBERT on global market/news context rows."),
    include_gdelt: bool = typer.Option(True, help="Include GDELT global flashpoint/news context."),
    gdelt_max_records_per_query: int = typer.Option(75, help="Maximum GDELT articles per configured query."),
) -> None:
    """Collect broad market/global news that can affect all tickers without being ticker-specific."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    start = datetime.now(timezone.utc) - timedelta(days=days)
    rows = []
    errors = []
    if settings.has_seeking_alpha_rapidapi:
        try:
            rows.extend([event.to_record() for event in SeekingAlphaRapidApiSource(settings).fetch_market_context_events(start)])
        except Exception as exc:
            errors.append(f"seeking_alpha_market_context:{exc}")
    if include_gdelt:
        try:
            gdelt_events, gdelt_errors = GdeltSource().fetch_context_events_with_errors(
                start,
                max_records_per_query=gdelt_max_records_per_query,
            )
            rows.extend([event.to_record() for event in gdelt_events])
            errors.extend(f"gdelt_context:{error}" for error in gdelt_errors)
        except Exception as exc:
            errors.append(f"gdelt_context:{exc}")
    frame = pd.DataFrame(rows)
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "timestamp", "source", "title", "url", "summary", "text", "raw"])
    else:
        frame = sanitize_events_frame(frame)[0]
        if score_sentiment:
            scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
            frame = add_finbert_with_scorer(frame, scorer, batch_size=settings.finbert_batch_size)
        if "raw" in frame.columns:
            frame["raw"] = frame["raw"].map(
                lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, dict) else value
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    summary = {
        "rows": len(frame),
        "sources": frame["source"].value_counts().to_dict() if "source" in frame else {},
        "errors": errors,
        "out": str(out),
    }
    (out.parent / "market_context_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    console.print(summary)


@app.command("collect-gdelt-context")
def collect_gdelt_context(
    days: int = typer.Option(2, help="Global flashpoint lookback window in calendar days."),
    out: Path = typer.Option(
        Path("data/external/market_context/gdelt_context_events.parquet"),
        help="Output GDELT context events parquet.",
    ),
    max_records_per_query: int = typer.Option(75, help="Maximum GDELT articles per configured query."),
    query: str | None = typer.Option(None, help="Optional single GDELT DOC query for one flashpoint family."),
    request_pause_seconds: float = typer.Option(5.5, help="Pause/retry backoff for GDELT public rate limits."),
    request_retries: int = typer.Option(2, help="HTTP retries per GDELT query."),
    score_sentiment: bool = typer.Option(False, help="Run FinBERT on GDELT context rows."),
) -> None:
    """Collect real global flashpoint news from GDELT into the market-context schema."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    start = datetime.now(timezone.utc) - timedelta(days=days)
    queries = (query,) if query else None
    source = GdeltSource(request_pause_seconds=request_pause_seconds, request_retries=request_retries)
    events, errors = source.fetch_context_events_with_errors(
        start,
        queries=queries or DEFAULT_GDELT_CONTEXT_QUERIES,
        max_records_per_query=max_records_per_query,
    )
    frame = pd.DataFrame([event.to_record() for event in events])
    if frame.empty:
        frame = pd.DataFrame(columns=["ticker", "timestamp", "source", "title", "url", "summary", "text", "raw"])
    else:
        frame = sanitize_events_frame(frame)[0]
        if score_sentiment:
            scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
            frame = add_finbert_with_scorer(frame, scorer, batch_size=settings.finbert_batch_size)
        if "raw" in frame.columns:
            frame["raw"] = frame["raw"].map(
                lambda value: json.dumps(value, ensure_ascii=True, sort_keys=True) if isinstance(value, dict) else value
            )
    out.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(out, index=False)
    summary = {
        "rows": len(frame),
        "sources": frame["source"].value_counts().to_dict() if "source" in frame else {},
        "out": str(out),
        "days": days,
        "max_records_per_query": max_records_per_query,
        "query": query,
        "errors": errors,
    }
    (out.parent / "gdelt_context_summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True), encoding="utf-8")
    console.print(summary)


@app.command("build-market-context-from-proxies")
def build_market_context_from_proxies(
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Raw event directory containing proxy ETF/index event files.",
    ),
    symbols: str = typer.Option(
        "SPY,QQQ,DIA,IWM,RSP,XLK,SMH,XBI,IBB,XAR,ITA,ARKF,ARKK,TLT,HYG,LQD,GLD,USO,UUP,KWEB,BITO",
        help="Comma-separated proxy symbols used as historical market/global context.",
    ),
    out: Path = typer.Option(
        Path("data/external/market_context/market_context_events.parquet"),
        help="Output market context parquet.",
    ),
) -> None:
    """Build historical broad-market context from proxy ETF/index event stores."""
    proxy_symbols = _parse_tickers(symbols, [])
    frames = []
    summary = []
    for symbol in proxy_symbols:
        path = raw_dir / f"{symbol}_events.parquet"
        if not path.exists():
            summary.append({"proxy": symbol, "rows": 0, "error": f"missing {path}"})
            continue
        try:
            frame = pd.read_parquet(path)
            if frame.empty:
                summary.append({"proxy": symbol, "rows": 0, "path": str(path)})
                continue
            frame = frame.copy()
            frame["market_proxy_symbol"] = symbol
            frame["ticker"] = "MARKET"
            frame["source"] = "market_proxy:" + symbol + ":" + frame["source"].astype(str)
            frames.append(frame)
            summary.append({"proxy": symbol, "rows": len(frame), "path": str(path)})
        except Exception as exc:
            summary.append({"proxy": symbol, "rows": 0, "error": str(exc)})
    if not frames:
        raise typer.BadParameter("No proxy event files found.")
    combined = pd.concat(frames, ignore_index=True)
    combined = sanitize_events_frame(combined)[0]
    out.parent.mkdir(parents=True, exist_ok=True)
    combined.to_parquet(out, index=False)
    summary_path = out.parent / "market_context_proxy_summary.csv"
    pd.DataFrame(summary).to_csv(summary_path, index=False)
    console.print({"rows": len(combined), "out": str(out), "summary": str(summary_path)})


@app.command("seeking-alpha-limits")
def seeking_alpha_limits() -> None:
    """Show local RapidAPI usage and the last rate-limit headers seen."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    status = source.quota_status()
    console.print(
        {
            "month": status.month,
            "local_used": status.used,
            "configured_monthly_limit": status.limit,
            "local_remaining": status.remaining,
            "last_rapidapi_headers": status.last_headers,
            "cache_dir": str(settings.seeking_alpha_cache_dir),
        }
    )


@app.command("seeking-alpha-token")
def seeking_alpha_token(
    refresh: bool = typer.Option(False, help="Force a new token request instead of using the local cache."),
) -> None:
    """Fetch/cache a Seeking Alpha account access token without printing the token."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    token = source.get_account_access_token(force_refresh=refresh)
    console.print(
        {
            "token_cached": bool(token),
            "cache_file": str(settings.seeking_alpha_access_token_cache_file),
        }
    )


@app.command("seeking-alpha-token-status")
def seeking_alpha_token_status() -> None:
    """Show whether Seeking Alpha account credentials/token cache are configured."""
    settings = get_settings()
    source = SeekingAlphaRapidApiSource(settings)
    console.print(source.account_token_status())


@app.command()
def watch(
    ticker: str,
    model: Path | None = typer.Option(None, help="Optional trained 1-day or 5-day model path."),
    days: int = typer.Option(45, help="Recent collection window for the watch score."),
    horizon_days: int = typer.Option(1, help="Forward trading-day horizon for the temporary feature set."),
) -> None:
    """Build a tomorrow/swing watch report for a ticker."""
    settings = get_settings()
    tmp_events = Path("data/tmp") / f"{ticker.upper()}_watch_events.parquet"
    tmp_dataset = Path("data/tmp") / f"{ticker.upper()}_watch_daily.parquet"
    tmp_sa = Path("data/tmp") / f"{ticker.upper()}_watch_seeking_alpha_quant.csv"
    sa_path = write_seeking_alpha_snapshot(ticker, tmp_sa) if settings.has_seeking_alpha_rapidapi else None
    collect(ticker, days=days, out=tmp_events, no_reddit=False)
    build_dataset(ticker, events=tmp_events, out=tmp_dataset, horizon_days=horizon_days, seeking_alpha=sa_path)
    dataset = pd.read_parquet(tmp_dataset)
    result = heuristic_watch_score(dataset, weights=settings.watch_score_weights)
    if model:
        result["model_prediction"] = predict_latest(dataset, model)
    console.print(result)


@app.command("behavior")
def behavior(
    tickers: str = typer.Option(..., help="Comma-separated symbols to analyze, for example MTNL,LUNR."),
    days: int = typer.Option(3, help="Recent news lookback window in calendar days."),
    raw_dir: Path = typer.Option(
        Path("data/raw/uslisted_6sector_2y_clean"),
        help="Cached raw event directory to use before live API calls.",
    ),
    out: Path = typer.Option(Path("data/reports/behavior_latest.csv"), help="Output behavior report CSV."),
    model_1d: Path | None = typer.Option(
        Path("models/uslisted_6sector_direction_2y_clean_1d_max.joblib"),
        help="Optional 1-day model path.",
    ),
    model_5d: Path | None = typer.Option(
        Path("models/uslisted_6sector_direction_2y_clean_5d_max.joblib"),
        help="Optional 5-day model path.",
    ),
    refresh: bool = typer.Option(False, help="Ignore cached raw files and collect only fresh recent events."),
    no_reddit: bool = typer.Option(False, help="Disable Reddit enrichment during refresh/live fallback."),
    no_seeking_alpha: bool = typer.Option(False, help="Disable Seeking Alpha during refresh/live fallback."),
) -> None:
    """Explain recent news/reaction behavior for a short ticker list."""
    from market_predictor.sentiment import FinbertScorer

    settings = get_settings()
    symbols = _parse_tickers(tickers, [])
    if not symbols:
        raise typer.BadParameter("Provide at least one ticker.")

    staged: list[pd.DataFrame] = []
    errors_by_symbol: dict[str, list[str]] = {}
    for symbol in symbols:
        frame, errors = _recent_events_for_behavior(
            symbol,
            days,
            raw_dir=raw_dir,
            refresh=refresh,
            no_reddit=no_reddit,
            no_seeking_alpha=no_seeking_alpha,
        )
        if not frame.empty:
            frame["_behavior_symbol"] = symbol
            staged.append(frame)
        errors_by_symbol[symbol] = errors

    scored_by_symbol: dict[str, pd.DataFrame] = {}
    if staged:
        combined = pd.concat(staged, ignore_index=True)
        scorer = FinbertScorer(settings.finbert_model, torch_num_threads=settings.torch_num_threads)
        combined = add_finbert_with_scorer(combined, scorer, batch_size=settings.finbert_batch_size)
        for symbol, frame in combined.groupby("_behavior_symbol", sort=False):
            scored_by_symbol[symbol] = frame.drop(columns=["_behavior_symbol"], errors="ignore")

    tmp_dir = Path("data/tmp/behavior")
    tmp_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for symbol in symbols:
        events = scored_by_symbol.get(symbol, pd.DataFrame())
        if events.empty:
            rows.append(
                {
                    "ticker": symbol,
                    "lookback_days": days,
                    "status": "no_recent_events",
                    "behavior": "quiet/no recent catalyst",
                    "errors": " | ".join(errors_by_symbol.get(symbol, [])),
                }
            )
            continue
        events_path = tmp_dir / f"{symbol}_behavior_events.parquet"
        events.to_parquet(events_path, index=False)
        try:
            dataset_1d = build_daily_dataset(
                symbol,
                events_path,
                settings,
                horizon_days=1,
                market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
            )
            score = heuristic_watch_score(dataset_1d, weights=settings.watch_score_weights)
            latest = dataset_1d.sort_values("date").iloc[-1]
            record = {
                "ticker": symbol,
                "lookback_days": days,
                "status": "ok",
                "date": score.get("date"),
                "behavior": _reaction_behavior_label(latest),
                "watch_score": score.get("watch_score"),
                "signal": score.get("signal"),
                "latest_close": score.get("latest_close"),
                "news_count_latest_day": score.get("news_count"),
                "recent_event_count": len(events),
                "sentiment_mean": score.get("sentiment_mean"),
                "return_1d": float(latest.get("return_1d", 0) or 0),
                "return_5d_past": float(latest.get("return_5d_past", 0) or 0),
                "volume_z20": float(latest.get("volume_z20", 0) or 0),
                "premarket_gap_mean": score.get("premarket_gap_mean"),
                "intraday_reaction_2h_mean": score.get("intraday_reaction_2h_mean"),
                "afterhours_next_open_gap_mean": score.get("afterhours_next_open_gap_mean"),
                "recent_headlines": _recent_headlines(events),
                "errors": " | ".join(errors_by_symbol.get(symbol, [])),
            }
            if model_1d and model_1d.exists():
                prediction = predict_latest(dataset_1d, model_1d)
                record["model_1d_probability_up"] = prediction["probability_up"]
                record["model_1d_prediction"] = prediction["prediction"]
            if model_5d and model_5d.exists():
                dataset_5d = build_daily_dataset(
                    symbol,
                    events_path,
                    settings,
                    horizon_days=5,
                    market_context_path=DEFAULT_MARKET_CONTEXT_PATH,
                )
                prediction = predict_latest(dataset_5d, model_5d)
                record["model_5d_probability_up"] = prediction["probability_up"]
                record["model_5d_prediction"] = prediction["prediction"]
            rows.append(record)
        except Exception as exc:
            rows.append(
                {
                    "ticker": symbol,
                    "lookback_days": days,
                    "status": "failed",
                    "recent_event_count": len(events),
                    "recent_headlines": _recent_headlines(events),
                    "errors": " | ".join([*errors_by_symbol.get(symbol, []), str(exc)]),
                }
            )

    report = pd.DataFrame(rows)
    out.parent.mkdir(parents=True, exist_ok=True)
    report.to_csv(out, index=False)
    display_cols = [
        col
        for col in [
            "ticker",
            "status",
            "date",
            "behavior",
            "watch_score",
            "signal",
            "model_1d_probability_up",
            "model_5d_probability_up",
            "recent_event_count",
            "news_count_latest_day",
            "sentiment_mean",
            "return_1d",
            "volume_z20",
        ]
        if col in report.columns
    ]
    console.print(report[display_cols])
    console.print(f"Wrote behavior report to {out}")
