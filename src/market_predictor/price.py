from __future__ import annotations

from datetime import datetime

import pandas as pd
import yfinance as yf

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource


def fetch_daily_prices(ticker: str, start: datetime, end: datetime | None, settings: Settings) -> pd.DataFrame:
    if settings.has_alpaca:
        bars = AlpacaSource(settings).fetch_daily_bars(ticker, start, end)
        if not bars.empty:
            return bars

    yf_frame = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat() if end else None,
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if yf_frame.empty:
        return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
    yf_frame = yf_frame.reset_index()
    return pd.DataFrame(
        {
            "date": pd.to_datetime(yf_frame["Date"]).dt.date,
            "open": yf_frame["Open"],
            "high": yf_frame["High"],
            "low": yf_frame["Low"],
            "close": yf_frame["Close"],
            "volume": yf_frame["Volume"],
        }
    )


def fetch_hourly_prices(ticker: str, start: datetime, end: datetime | None, settings: Settings) -> pd.DataFrame:
    if settings.has_alpaca:
        bars = AlpacaSource(settings).fetch_hourly_bars(ticker, start, end)
        if not bars.empty:
            return bars

    yf_frame = yf.download(
        ticker,
        start=start.date().isoformat(),
        end=end.date().isoformat() if end else None,
        interval="60m",
        auto_adjust=True,
        progress=False,
        threads=False,
    )
    if yf_frame.empty:
        return pd.DataFrame(columns=["timestamp", "date", "open", "high", "low", "close", "volume"])
    yf_frame = yf_frame.reset_index()
    timestamp_col = "Datetime" if "Datetime" in yf_frame.columns else "Date"
    timestamps = pd.to_datetime(yf_frame[timestamp_col], utc=True)
    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "date": timestamps.dt.date,
            "open": yf_frame["Open"],
            "high": yf_frame["High"],
            "low": yf_frame["Low"],
            "close": yf_frame["Close"],
            "volume": yf_frame["Volume"],
        }
    )
