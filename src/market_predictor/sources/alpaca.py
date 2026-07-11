from __future__ import annotations

from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


class AlpacaSource:
    news_url = "https://data.alpaca.markets/v1beta1/news"
    bars_url = "https://data.alpaca.markets/v2/stocks/bars"
    corporate_actions_url = "https://data.alpaca.markets/v1/corporate-actions"

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.client = HttpClient()

    @property
    def headers(self) -> dict[str, str]:
        if not self.settings.has_alpaca:
            raise ValueError("ALPACA_API_KEY_ID and ALPACA_API_SECRET_KEY are required.")
        return {
            "APCA-API-KEY-ID": self.settings.alpaca_api_key_id or "",
            "APCA-API-SECRET-KEY": self.settings.alpaca_api_secret_key or "",
        }

    @property
    def assets_url(self) -> str:
        return f"{self.settings.alpaca_trading_base_url.rstrip('/')}/v2/assets"

    def fetch_assets(self) -> pd.DataFrame:
        payload = self.client.get_json(
            self.assets_url,
            params={
                "status": self.settings.universe_status,
                "asset_class": self.settings.universe_asset_class,
            },
            headers=self.headers,
        )
        frame = pd.DataFrame(payload)
        if frame.empty:
            return pd.DataFrame(
                columns=[
                    "symbol",
                    "name",
                    "exchange",
                    "status",
                    "tradable",
                    "marginable",
                    "shortable",
                    "easy_to_borrow",
                    "fractionable",
                ]
            )
        keep_cols = [
            col
            for col in [
                "symbol",
                "name",
                "exchange",
                "status",
                "tradable",
                "marginable",
                "shortable",
                "easy_to_borrow",
                "fractionable",
            ]
            if col in frame.columns
        ]
        return frame[keep_cols].sort_values("symbol").reset_index(drop=True)

    def fetch_ticker_universe(self) -> pd.DataFrame:
        assets = self.fetch_assets()
        if assets.empty:
            return assets
        if "exchange" in assets.columns:
            assets = assets[assets["exchange"].isin(self.settings.universe_exchanges)]
        if self.settings.universe_tradable_only and "tradable" in assets.columns:
            assets = assets[assets["tradable"] == True]  # noqa: E712
        return assets.sort_values("symbol").reset_index(drop=True)

    def fetch_name_changes(self, start: date, end: date) -> pd.DataFrame:
        """Fetch point-in-time US symbol changes used to preserve security continuity."""
        params: dict[str, Any] = {
            "types": "name_change",
            "region": "us",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "sort": "asc",
        }
        rows: list[dict[str, Any]] = []
        while True:
            payload = self.client.get_json(self.corporate_actions_url, params=params, headers=self.headers)
            rows.extend(payload.get("corporate_actions", {}).get("name_changes", []))
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        columns = ["id", "process_date", "old_symbol", "new_symbol", "old_cusip", "new_cusip"]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        keep = [column for column in columns if column in frame.columns]
        return frame[keep].sort_values(["process_date", "old_symbol"], kind="stable").reset_index(drop=True)

    def fetch_news(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        *,
        include_content: bool = True,
        limit: int = 50,
    ) -> list[NewsEvent]:
        end = end or datetime.now(UTC)
        params: dict[str, Any] = {
            "symbols": ticker.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort": "asc",
            "limit": min(limit, 50),
            "include_content": str(include_content).lower(),
        }

        events: list[NewsEvent] = []
        while True:
            payload = self.client.get_json(self.news_url, params=params, headers=self.headers)
            for item in payload.get("news", []):
                timestamp = pd.to_datetime(item.get("updated_at") or item.get("created_at"), utc=True)
                events.append(
                    NewsEvent(
                        ticker=ticker.upper(),
                        timestamp=timestamp.to_pydatetime(),
                        source=f"alpaca:{item.get('source', 'unknown')}",
                        title=item.get("headline") or "",
                        url=item.get("url"),
                        summary=item.get("summary"),
                        text=item.get("content") or item.get("summary"),
                        raw=item,
                    )
                )
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        return events

    def fetch_daily_bars(self, ticker: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        end = end or datetime.now(UTC)
        params = {
            "symbols": ticker.upper(),
            "timeframe": "1Day",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": self.settings.alpaca_stock_feed,
            "limit": 10000,
            "adjustment": "all",
        }
        rows = self._fetch_bar_rows(ticker, params)
        if not rows:
            return pd.DataFrame(columns=["date", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.date
        return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

    def fetch_hourly_bars(self, ticker: str, start: datetime, end: datetime | None = None) -> pd.DataFrame:
        return self.fetch_intraday_bars(ticker, start, end, timeframe="1Hour")

    def fetch_intraday_bars(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        *,
        timeframe: str,
    ) -> pd.DataFrame:
        end = end or datetime.now(UTC)
        params = {
            "symbols": ticker.upper(),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": self.settings.alpaca_stock_feed,
            "limit": 10000,
            "adjustment": "all",
        }
        rows = self._fetch_bar_rows(ticker, params)
        if not rows:
            return pd.DataFrame(columns=["timestamp", "date", "open", "high", "low", "close", "volume"])
        frame = pd.DataFrame(rows).rename(
            columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"}
        )
        frame["timestamp"] = pd.to_datetime(frame["timestamp"], utc=True)
        frame["date"] = frame["timestamp"].dt.date
        return frame[["timestamp", "date", "open", "high", "low", "close", "volume"]].sort_values("timestamp")

    def _fetch_bar_rows(self, ticker: str, params: dict[str, Any]) -> list[dict[str, Any]]:
        request_params = dict(params)
        rows: list[dict[str, Any]] = []
        while True:
            payload = self.client.get_json(self.bars_url, params=request_params, headers=self.headers)
            rows.extend(payload.get("bars", {}).get(ticker.upper(), []))
            token = payload.get("next_page_token")
            if not token:
                break
            request_params["page_token"] = token
        return rows
