from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, date, datetime
from typing import Any

import pandas as pd

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


@dataclass(frozen=True, slots=True)
class AlpacaNewsPage:
    request_page_token: str | None
    next_page_token: str | None
    news: tuple[dict[str, Any], ...]


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
            "APCA-API-SECRET-KEY": self.settings.alpaca_api_secret_value or "",
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

    def fetch_security_transitions(self, start: date, end: date) -> pd.DataFrame:
        """Fetch symbol transitions that can carry index membership across a corporate event."""

        params: dict[str, Any] = {
            "types": "name_change,cash_merger,stock_merger,stock_and_cash_merger,reorganization",
            "region": "us",
            "start": start.isoformat(),
            "end": end.isoformat(),
            "limit": 1000,
            "sort": "asc",
        }
        rows: list[dict[str, Any]] = []
        while True:
            payload = self.client.get_json(self.corporate_actions_url, params=params, headers=self.headers)
            actions = payload.get("corporate_actions", {})
            for item in actions.get("name_changes", []):
                rows.append(
                    {
                        "id": item.get("id"),
                        "process_date": item.get("process_date"),
                        "effective_date": item.get("process_date"),
                        "old_symbol": item.get("old_symbol"),
                        "new_symbol": item.get("new_symbol"),
                        "old_cusip": item.get("old_cusip"),
                        "new_cusip": item.get("new_cusip"),
                        "transition_type": "name_change",
                        "identity_continuity": item.get("old_cusip") == item.get("new_cusip"),
                        "membership_continuity": True,
                    }
                )
            for family in ("cash_mergers", "stock_mergers", "stock_and_cash_mergers", "reorganizations"):
                for item in actions.get(family, []):
                    if not item.get("acquiree_symbol") or not item.get("acquirer_symbol"):
                        continue
                    rows.append(
                        {
                            "id": item.get("id"),
                            "process_date": item.get("process_date"),
                            "effective_date": item.get("effective_date") or item.get("process_date"),
                            "old_symbol": item.get("acquiree_symbol"),
                            "new_symbol": item.get("acquirer_symbol"),
                            "old_cusip": item.get("acquiree_cusip"),
                            "new_cusip": item.get("acquirer_cusip"),
                            "transition_type": family.removesuffix("s"),
                            "identity_continuity": False,
                            "membership_continuity": False,
                        }
                    )
            token = payload.get("next_page_token")
            if not token:
                break
            params["page_token"] = token
        columns = [
            "id",
            "process_date",
            "effective_date",
            "old_symbol",
            "new_symbol",
            "old_cusip",
            "new_cusip",
            "transition_type",
            "identity_continuity",
            "membership_continuity",
        ]
        if not rows:
            return pd.DataFrame(columns=columns)
        frame = pd.DataFrame(rows)
        frame = frame.dropna(subset=["id", "effective_date", "old_symbol", "new_symbol"])
        frame = frame.sort_values(
            ["effective_date", "old_symbol", "new_symbol", "transition_type", "id"],
            kind="stable",
        )
        frame = frame.drop_duplicates(["effective_date", "old_symbol", "new_symbol"], keep="first")
        return frame.loc[:, columns].reset_index(drop=True)

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
        events: list[NewsEvent] = []
        token: str | None = None
        seen_tokens: set[str] = set()
        while True:
            page = self.fetch_news_page(
                ticker,
                start,
                end,
                page_token=token,
                include_content=include_content,
                limit=limit,
            )
            for item in page.news:
                timestamp = pd.to_datetime(
                    item.get("created_at") or item.get("updated_at"),
                    utc=True,
                )
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
            token = page.next_page_token
            if not token:
                break
            if token in seen_tokens:
                raise RuntimeError("Alpaca news pagination repeated a page token")
            seen_tokens.add(token)
        return events

    def fetch_news_page(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        page_token: str | None = None,
        include_content: bool = True,
        limit: int = 50,
    ) -> AlpacaNewsPage:
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Alpaca news bounds must be timezone-aware")
        if start >= end:
            raise ValueError("Alpaca news start must precede end")
        if limit < 1 or limit > 50:
            raise ValueError("Alpaca news page limit must be between 1 and 50")
        params: dict[str, Any] = {
            "symbols": ticker.upper(),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "sort": "asc",
            "limit": limit,
            "include_content": str(include_content).lower(),
        }
        if page_token:
            params["page_token"] = page_token
        payload = self.client.get_json(
            self.news_url,
            params=params,
            headers=self.headers,
        )
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca news response must be an object")
        raw_news = payload.get("news", [])
        if not isinstance(raw_news, list) or any(
            not isinstance(item, dict) for item in raw_news
        ):
            raise RuntimeError("Alpaca news response has invalid news rows")
        next_token_value = payload.get("next_page_token")
        next_token = (
            str(next_token_value).strip()
            if next_token_value is not None and str(next_token_value).strip()
            else None
        )
        return AlpacaNewsPage(
            request_page_token=page_token,
            next_page_token=next_token,
            news=tuple({str(key): value for key, value in item.items()} for item in raw_news),
        )

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
        frame = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
        frame["date"] = pd.to_datetime(frame["timestamp"], utc=True).dt.date
        return frame[["date", "open", "high", "low", "close", "volume"]].sort_values("date")

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
        frame = pd.DataFrame(rows).rename(columns={"t": "timestamp", "o": "open", "h": "high", "l": "low", "c": "close", "v": "volume"})
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
