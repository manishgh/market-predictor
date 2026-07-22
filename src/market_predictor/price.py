from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource
from market_predictor.symbols import PROVIDER_ALPACA, PROVIDER_YAHOO, provider_symbol

INTRADAY_TIMEFRAMES = {
    "1m": {"alpaca": "1Min", "yfinance": "1m"},
    "5m": {"alpaca": "5Min", "yfinance": "5m"},
    "1h": {"alpaca": "1Hour", "yfinance": "60m"},
}


def _tag_feed(frame: pd.DataFrame, feed: str) -> pd.DataFrame:
    """Record bar provenance so downstream readiness gates can see it.

    ``DataFrame.attrs`` is preserved through ``copy``/``sort_values`` but not
    through merges, so callers should read the tag right after fetching.
    """
    frame.attrs["price_feed"] = feed
    return frame


def fetch_daily_prices(ticker: str, start: datetime, end: datetime | None, settings: Settings) -> pd.DataFrame:
    alpaca_symbol = provider_symbol(ticker, PROVIDER_ALPACA)
    yahoo_symbol = provider_symbol(ticker, PROVIDER_YAHOO)
    if settings.has_alpaca:
        bars = AlpacaSource(settings).fetch_daily_bars(alpaca_symbol, start, end)
        if not bars.empty:
            return _tag_feed(bars, "alpaca")

    yf_frame = yf.download(
        yahoo_symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat() if end else None,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if yf_frame.empty:
        return _tag_feed(pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"]), "none")
    yf_frame = yf_frame.reset_index()
    return _tag_feed(
        pd.DataFrame(
            {
                "date": pd.to_datetime(yf_frame["Date"]).dt.date,
                "open": yf_frame["Open"],
                "high": yf_frame["High"],
                "low": yf_frame["Low"],
                "close": yf_frame["Close"],
                "volume": yf_frame["Volume"],
            }
        ),
        "yfinance",
    )


def fetch_hourly_prices(ticker: str, start: datetime, end: datetime | None, settings: Settings) -> pd.DataFrame:
    return fetch_intraday_prices(ticker, start, end, settings, timeframe="1h")


def fetch_intraday_prices(
    ticker: str,
    start: datetime,
    end: datetime | None,
    settings: Settings,
    *,
    timeframe: str,
) -> pd.DataFrame:
    key = timeframe.strip().lower()
    if key not in INTRADAY_TIMEFRAMES:
        raise ValueError(f"Unsupported intraday timeframe: {timeframe}")
    alpaca_symbol = provider_symbol(ticker, PROVIDER_ALPACA)
    yahoo_symbol = provider_symbol(ticker, PROVIDER_YAHOO)
    if settings.has_alpaca:
        bars = AlpacaSource(settings).fetch_intraday_bars(
            alpaca_symbol,
            start,
            end,
            timeframe=INTRADAY_TIMEFRAMES[key]["alpaca"],
        )
        if not bars.empty:
            return _tag_feed(bars, "alpaca")

    yf_frame = yf.download(
        yahoo_symbol,
        start=start.date().isoformat(),
        end=end.date().isoformat() if end else None,
        interval=INTRADAY_TIMEFRAMES[key]["yfinance"],
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if yf_frame.empty:
        return _tag_feed(pd.DataFrame(columns=["timestamp", "date", "open", "high", "low", "close", "volume"]), "none")
    yf_frame = yf_frame.reset_index()
    timestamp_col = "Datetime" if "Datetime" in yf_frame.columns else "Date"
    timestamps = pd.to_datetime(yf_frame[timestamp_col], utc=True)
    return _tag_feed(
        pd.DataFrame(
            {
                "timestamp": timestamps,
                "date": timestamps.dt.date,
                "open": yf_frame["Open"],
                "high": yf_frame["High"],
                "low": yf_frame["Low"],
                "close": yf_frame["Close"],
                "volume": yf_frame["Volume"],
            }
        ),
        "yfinance",
    )
