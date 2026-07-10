from __future__ import annotations

from pathlib import Path
import re
from typing import Iterable

import numpy as np
import pandas as pd

from market_predictor.data_quality import sanitize_events_frame
from market_predictor.features import EVENT_KEYWORDS, add_event_taxonomy, source_family_for_source


CATALYST_WINDOWS = {
    "30m": pd.Timedelta(minutes=30),
    "2h": pd.Timedelta(hours=2),
    "1d": pd.Timedelta(days=1),
}

SOURCE_FAMILIES = ["alpaca", "reddit", "seeking_alpha", "sec", "finviz"]

INTRADAY_CATALYST_FEATURES = [
    "minutes_since_last_catalyst",
    "latest_catalyst_sentiment",
    "latest_catalyst_relevance",
    "has_recent_catalyst_2h",
    "catalyst_attention_score_2h",
    "market_context_minutes_since_last_news",
    "market_context_intraday_shock_score_2h",
]

for _window in CATALYST_WINDOWS:
    INTRADAY_CATALYST_FEATURES.extend(
        [
            f"news_count_{_window}",
            f"sentiment_mean_{_window}",
            f"sentiment_abs_mean_{_window}",
            f"sentiment_pos_frac_{_window}",
            f"sentiment_neg_frac_{_window}",
            f"event_relevance_mean_{_window}",
            f"generic_movers_count_{_window}",
            f"market_context_news_count_{_window}",
            f"market_context_sentiment_mean_{_window}",
            f"market_context_sentiment_abs_mean_{_window}",
            f"market_context_sentiment_neg_frac_{_window}",
        ]
    )
    for _family in SOURCE_FAMILIES:
        INTRADAY_CATALYST_FEATURES.append(f"source_count_{_family}_{_window}")
    for _event_name in EVENT_KEYWORDS:
        INTRADAY_CATALYST_FEATURES.append(f"event_{_event_name}_count_{_window}")


def add_intraday_catalyst_features(
    frame: pd.DataFrame,
    *,
    event_dirs: Iterable[Path] | None = None,
    market_context_path: Path | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Join ticker and market-context catalyst features to intraday bars.

    Features are strictly as-of: an intraday bar only sees events whose timestamp
    is less than or equal to that row timestamp.
    """
    if frame.empty:
        return frame.copy(), pd.DataFrame()
    data = frame.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["timestamp"] = pd.to_datetime(data.get("timestamp", data.get("date")), errors="coerce", utc=True)
    data = data.dropna(subset=["ticker", "timestamp"]).sort_values(["ticker", "timestamp"]).reset_index(drop=True)
    event_dirs = [Path(path) for path in event_dirs or []]
    market_events = _load_market_context_events(market_context_path)
    pieces = []
    audit_rows: list[dict[str, object]] = []
    for ticker, group in data.groupby("ticker", sort=False):
        events = _load_ticker_events(ticker, event_dirs)
        enriched = _apply_event_features(group, events, prefix="")
        enriched = _apply_event_features(enriched, market_events, prefix="market_context_")
        pieces.append(enriched)
        audit_rows.append(
            {
                "ticker": ticker,
                "rows": int(len(group)),
                "ticker_events": int(len(events)),
                "market_context_events": int(len(market_events)),
                "rows_with_2h_news": int(pd.to_numeric(enriched.get("news_count_2h", 0), errors="coerce").fillna(0).gt(0).sum()),
                "rows_with_1d_news": int(pd.to_numeric(enriched.get("news_count_1d", 0), errors="coerce").fillna(0).gt(0).sum()),
                "first_timestamp": group["timestamp"].min(),
                "last_timestamp": group["timestamp"].max(),
            }
        )
    output = pd.concat(pieces, ignore_index=True) if pieces else data
    output = _add_compatibility_columns(output)
    return output.reset_index(drop=True), pd.DataFrame(audit_rows)


def _apply_event_features(group: pd.DataFrame, events: pd.DataFrame, *, prefix: str) -> pd.DataFrame:
    out = group.sort_values("timestamp").copy()
    if events.empty:
        return _add_empty_event_columns(out, prefix)
    events = _prepare_events(events, ticker=str(out["ticker"].iloc[0]) if prefix == "" else None)
    if events.empty:
        return _add_empty_event_columns(out, prefix)
    start = out["timestamp"].min() - max(CATALYST_WINDOWS.values())
    end = out["timestamp"].max()
    events = events[(events["timestamp"] >= start) & (events["timestamp"] <= end)].sort_values("timestamp").reset_index(drop=True)
    if events.empty:
        return _add_empty_event_columns(out, prefix)
    bar_ns = out["timestamp"].astype("int64").to_numpy()
    event_ns = events["timestamp"].astype("int64").to_numpy()
    end_idx = np.searchsorted(event_ns, bar_ns, side="right")
    last_idx = end_idx - 1
    has_last = last_idx >= 0
    latest_ts = np.full(len(out), np.nan)
    latest_sentiment = np.full(len(out), np.nan)
    latest_relevance = np.full(len(out), np.nan)
    latest_ts[has_last] = event_ns[last_idx[has_last]]
    latest_sentiment[has_last] = pd.to_numeric(events["sentiment_numeric"], errors="coerce").fillna(0.0).to_numpy()[last_idx[has_last]]
    latest_relevance[has_last] = pd.to_numeric(events["event_relevance_score"], errors="coerce").fillna(0.0).to_numpy()[last_idx[has_last]]
    minutes_since = (bar_ns - latest_ts) / (1_000_000_000 * 60)
    if prefix:
        out[f"{prefix}minutes_since_last_news"] = minutes_since
    else:
        out["minutes_since_last_catalyst"] = minutes_since
        out["latest_catalyst_sentiment"] = latest_sentiment
        out["latest_catalyst_relevance"] = latest_relevance

    numeric = _numeric_event_columns(events)
    for window_name, window in CATALYST_WINDOWS.items():
        start_idx = np.searchsorted(event_ns, (out["timestamp"] - window).astype("int64").to_numpy(), side="left")
        count = end_idx - start_idx
        out[f"{prefix}news_count_{window_name}"] = count.astype(float)
        for source in SOURCE_FAMILIES:
            out[f"{prefix}source_count_{source}_{window_name}"] = _window_sum(numeric[f"source_is_{source}"], start_idx, end_idx)
        for event_name in EVENT_KEYWORDS:
            out[f"{prefix}event_{event_name}_count_{window_name}"] = _window_sum(numeric[f"event_{event_name}"], start_idx, end_idx)
        sentiment_sum = _window_sum(numeric["sentiment_numeric"], start_idx, end_idx)
        sentiment_abs_sum = _window_sum(numeric["sentiment_abs"], start_idx, end_idx)
        relevance_sum = _window_sum(numeric["event_relevance_score"], start_idx, end_idx)
        pos_count = _window_sum(numeric["sentiment_positive"], start_idx, end_idx)
        neg_count = _window_sum(numeric["sentiment_negative"], start_idx, end_idx)
        generic_count = _window_sum(numeric["generic_movers_headline"], start_idx, end_idx)
        divisor = np.where(count > 0, count, np.nan)
        out[f"{prefix}sentiment_mean_{window_name}"] = sentiment_sum / divisor
        out[f"{prefix}sentiment_abs_mean_{window_name}"] = sentiment_abs_sum / divisor
        out[f"{prefix}event_relevance_mean_{window_name}"] = relevance_sum / divisor
        out[f"{prefix}sentiment_pos_frac_{window_name}"] = pos_count / divisor
        out[f"{prefix}sentiment_neg_frac_{window_name}"] = neg_count / divisor
        out[f"{prefix}generic_movers_count_{window_name}"] = generic_count
    if prefix:
        out["market_context_intraday_shock_score_2h"] = (
            pd.to_numeric(out["market_context_news_count_2h"], errors="coerce").fillna(0).clip(upper=20) / 20.0
            + pd.to_numeric(out["market_context_sentiment_neg_frac_2h"], errors="coerce").fillna(0.0)
        ).clip(upper=1.0)
    else:
        out["has_recent_catalyst_2h"] = pd.to_numeric(out["news_count_2h"], errors="coerce").fillna(0).gt(0).astype(int)
        out["catalyst_attention_score_2h"] = (
            pd.to_numeric(out["news_count_2h"], errors="coerce").fillna(0).clip(upper=10)
            * (1.0 + pd.to_numeric(out.get("event_relevance_mean_2h", 0), errors="coerce").fillna(0.0))
            * (1.0 + pd.to_numeric(out.get("volume_z20", 0), errors="coerce").fillna(0.0).clip(lower=0, upper=5))
        )
    return out


def _add_empty_event_columns(frame: pd.DataFrame, prefix: str) -> pd.DataFrame:
    out = frame.copy()
    if prefix:
        out[f"{prefix}minutes_since_last_news"] = np.nan
        out["market_context_intraday_shock_score_2h"] = 0.0
    else:
        out["minutes_since_last_catalyst"] = np.nan
        out["latest_catalyst_sentiment"] = 0.0
        out["latest_catalyst_relevance"] = 0.0
        out["has_recent_catalyst_2h"] = 0
        out["catalyst_attention_score_2h"] = 0.0
    for window_name in CATALYST_WINDOWS:
        out[f"{prefix}news_count_{window_name}"] = 0.0
        out[f"{prefix}sentiment_mean_{window_name}"] = 0.0
        out[f"{prefix}sentiment_abs_mean_{window_name}"] = 0.0
        out[f"{prefix}sentiment_pos_frac_{window_name}"] = 0.0
        out[f"{prefix}sentiment_neg_frac_{window_name}"] = 0.0
        out[f"{prefix}event_relevance_mean_{window_name}"] = 0.0
        out[f"{prefix}generic_movers_count_{window_name}"] = 0.0
        for source in SOURCE_FAMILIES:
            out[f"{prefix}source_count_{source}_{window_name}"] = 0.0
        for event_name in EVENT_KEYWORDS:
            out[f"{prefix}event_{event_name}_count_{window_name}"] = 0.0
    return out


def _add_compatibility_columns(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    compatibility = {
        "news_count": "news_count_1d",
        "event_count": "news_count_1d",
        "sentiment_mean": "sentiment_mean_1d",
        "sentiment_pos_frac": "sentiment_pos_frac_1d",
        "sentiment_neg_frac": "sentiment_neg_frac_1d",
        "source_count_alpaca": "source_count_alpaca_1d",
        "source_count_reddit": "source_count_reddit_1d",
        "source_count_seeking_alpha": "source_count_seeking_alpha_1d",
        "source_count_sec": "source_count_sec_1d",
        "source_count_finviz": "source_count_finviz_1d",
        "market_context_news_count": "market_context_news_count_1d",
        "market_context_sentiment_mean": "market_context_sentiment_mean_1d",
        "market_context_sentiment_neg_frac": "market_context_sentiment_neg_frac_1d",
    }
    for target, source in compatibility.items():
        if source in out.columns:
            out[target] = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
    for event_name in EVENT_KEYWORDS:
        source = f"event_{event_name}_count_1d"
        if source in out.columns:
            out[f"event_{event_name}_count"] = pd.to_numeric(out[source], errors="coerce").fillna(0.0)
    if "news_count_1d" in out.columns:
        count = pd.to_numeric(out["news_count_1d"], errors="coerce").fillna(0.0)
        rolling_mean = count.rolling(390, min_periods=20).mean()
        rolling_std = count.rolling(390, min_periods=20).std()
        out["news_count_z30"] = ((count - rolling_mean) / rolling_std.replace(0, np.nan)).fillna(0.0)
    return out


def _load_ticker_events(ticker: str, event_dirs: list[Path]) -> pd.DataFrame:
    frames = []
    for directory in event_dirs:
        path = directory / f"{ticker.upper()}_events.parquet"
        if path.exists():
            frames.append(pd.read_parquet(path))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _load_market_context_events(path: Path | None) -> pd.DataFrame:
    if path is None or not path.exists():
        return pd.DataFrame()
    return pd.read_parquet(path)


def _prepare_events(events: pd.DataFrame, *, ticker: str | None) -> pd.DataFrame:
    clean, _ = sanitize_events_frame(events)
    if clean.empty:
        return clean
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], errors="coerce", utc=True)
    clean = clean.dropna(subset=["timestamp"]).copy()
    clean["source_family"] = clean["source"].map(source_family_for_source)
    clean = add_event_taxonomy(clean)
    if "sentiment_numeric" not in clean.columns:
        clean["sentiment_numeric"] = 0.0
    clean["sentiment_numeric"] = pd.to_numeric(clean["sentiment_numeric"], errors="coerce").fillna(0.0)
    text = _event_text(clean)
    if ticker:
        clean["title_has_ticker"] = clean["title"].fillna("").astype(str).map(lambda value: _contains_ticker(value, ticker))
        clean["text_has_ticker"] = text.map(lambda value: _contains_ticker(value, ticker))
    else:
        clean["title_has_ticker"] = True
        clean["text_has_ticker"] = True
    clean["generic_movers_headline"] = clean["title"].fillna("").astype(str).map(_is_generic_market_headline).astype(int)
    clean["event_relevance_score"] = clean.apply(_event_relevance_score, axis=1)
    return clean.sort_values("timestamp").reset_index(drop=True)


def _numeric_event_columns(events: pd.DataFrame) -> dict[str, np.ndarray]:
    values: dict[str, np.ndarray] = {}
    values["sentiment_numeric"] = pd.to_numeric(events["sentiment_numeric"], errors="coerce").fillna(0.0).to_numpy(dtype="float")
    values["sentiment_abs"] = np.abs(values["sentiment_numeric"])
    values["sentiment_positive"] = (values["sentiment_numeric"] > 0.15).astype(float)
    values["sentiment_negative"] = (values["sentiment_numeric"] < -0.15).astype(float)
    values["event_relevance_score"] = pd.to_numeric(events["event_relevance_score"], errors="coerce").fillna(0.0).to_numpy(dtype="float")
    values["generic_movers_headline"] = pd.to_numeric(events["generic_movers_headline"], errors="coerce").fillna(0.0).to_numpy(dtype="float")
    for source in SOURCE_FAMILIES:
        values[f"source_is_{source}"] = events["source_family"].eq(source).astype(float).to_numpy()
    for event_name in EVENT_KEYWORDS:
        values[f"event_{event_name}"] = pd.to_numeric(events.get(f"event_{event_name}", 0), errors="coerce").fillna(0.0).to_numpy(dtype="float")
    return values


def _window_sum(values: np.ndarray, start_idx: np.ndarray, end_idx: np.ndarray) -> np.ndarray:
    cumulative = np.concatenate([[0.0], np.cumsum(values)])
    return cumulative[end_idx] - cumulative[start_idx]


def _event_text(frame: pd.DataFrame) -> pd.Series:
    output = frame["title"].fillna("").astype(str)
    for column in ["summary", "text"]:
        if column in frame.columns:
            output = output + " " + frame[column].fillna("").astype(str)
    return output


def _contains_ticker(text: str, ticker: str) -> bool:
    return bool(re.search(rf"(?<![A-Z0-9])\$?{re.escape(ticker.upper())}(?![A-Z0-9])", str(text).upper()))


def _is_generic_market_headline(title: str) -> bool:
    lowered = str(title or "").lower()
    patterns = [
        "stocks moving",
        "stock moving",
        "moving higher",
        "moving lower",
        "premarket",
        "pre-market",
        "after-market",
        "biggest stock movers",
        "market summary",
        "trending stocks",
    ]
    return any(pattern in lowered for pattern in patterns)


def _event_relevance_score(row: pd.Series) -> float:
    score = 1.0
    if bool(row.get("title_has_ticker", False)):
        score += 0.75
    elif bool(row.get("text_has_ticker", False)):
        score += 0.35
    if bool(row.get("generic_movers_headline", False)):
        score -= 0.6
    if str(row.get("source_family", "")) == "sec":
        score += 0.3
    return max(score, 0.1)
