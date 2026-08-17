from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import UTC, date, datetime
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qsl, urlsplit

import pandas as pd

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


@dataclass(frozen=True, slots=True)
class AlpacaNewsPage:
    request_page_token: str | None
    next_page_token: str | None
    news: tuple[dict[str, Any], ...]
    response_headers: dict[str, str] = field(default_factory=dict)
    raw_payload: dict[str, Any] | None = None
    raw_body: bytes | None = None
    requested_url: str | None = None
    status_code: int | None = None
    retrieved_at_utc: datetime | None = None
    final_url: str | None = None
    redirect_chain: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlpacaAssetSnapshot:
    assets: pd.DataFrame
    raw_body: bytes
    response_headers: dict[str, str]
    requested_url: str
    status_code: int
    retrieved_at_utc: datetime
    final_url: str | None = None
    redirect_chain: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlpacaBarsPage:
    request_page_token: str | None
    next_page_token: str | None
    bars: dict[str, tuple[dict[str, Any], ...]]
    response_headers: dict[str, str]
    raw_payload: dict[str, Any] | None = None
    raw_body: bytes | None = None
    requested_url: str | None = None
    status_code: int | None = None
    retrieved_at_utc: datetime | None = None
    final_url: str | None = None
    redirect_chain: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class AlpacaTradesPage:
    request_page_token: str | None
    next_page_token: str | None
    trades: dict[str, tuple[dict[str, Any], ...]]
    response_headers: dict[str, str]
    raw_payload: dict[str, Any]


@dataclass(frozen=True, slots=True)
class AlpacaQuotesPage:
    request_page_token: str | None
    next_page_token: str | None
    quotes: dict[str, tuple[dict[str, Any], ...]]
    response_headers: dict[str, str]
    raw_payload: dict[str, Any]


def _verify_bars_page_request_url(
    requested_url: str,
    *,
    params: dict[str, Any],
) -> None:
    parsed = urlsplit(requested_url)
    pairs = parse_qsl(parsed.query, keep_blank_values=True)
    expected = {str(key): str(value) for key, value in params.items()}
    if (
        parsed.scheme.lower() != "https"
        or parsed.hostname != "data.alpaca.markets"
        or parsed.port not in (None, 443)
        or parsed.username is not None
        or parsed.password is not None
        or parsed.path != "/v2/stocks/bars"
        or parsed.fragment
        or len(pairs) != len(expected)
        or {key: value for key, value in pairs} != expected
    ):
        raise RuntimeError("Alpaca bars page request URL does not match the frozen query")


class AlpacaSource:
    news_url = "https://data.alpaca.markets/v1beta1/news"
    bars_url = "https://data.alpaca.markets/v2/stocks/bars"
    trades_url = "https://data.alpaca.markets/v2/stocks/trades"
    quotes_url = "https://data.alpaca.markets/v2/stocks/quotes"
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
                    "id",
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
                "id",
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

    def fetch_assets_snapshot(self) -> AlpacaAssetSnapshot:
        response = self.client.get_bytes_with_metadata(
            self.assets_url,
            params={
                "status": self.settings.universe_status,
                "asset_class": self.settings.universe_asset_class,
            },
            headers=self.headers,
        )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Alpaca asset response is not valid UTF-8 JSON") from exc
        if not isinstance(payload, list) or any(
            not isinstance(item, dict) for item in payload
        ):
            raise RuntimeError("Alpaca asset response must be an array of objects")
        frame = pd.DataFrame(payload)
        keep_cols = [
            column
            for column in ("id", "symbol", "status", "exchange", "tradable")
            if column in frame.columns
        ]
        assets = frame.loc[:, keep_cols].copy()
        return AlpacaAssetSnapshot(
            assets=assets,
            raw_body=response.body,
            response_headers=dict(response.safe_headers),
            requested_url=response.requested_url,
            status_code=response.status_code,
            retrieved_at_utc=response.retrieved_at_utc,
            final_url=response.final_url,
            redirect_chain=response.redirect_chain,
        )

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
            response_headers={},
            raw_payload={str(key): value for key, value in payload.items()},
        )

    def fetch_news_page_observed(
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
        response = self.client.get_bytes_with_metadata(
            self.news_url,
            params=params,
            headers=self.headers,
        )
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Alpaca news response is not valid UTF-8 JSON") from exc
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
            news=tuple(
                {str(key): value for key, value in item.items()}
                for item in raw_news
            ),
            response_headers=dict(response.safe_headers),
            raw_payload={str(key): value for key, value in payload.items()},
            raw_body=response.body,
            requested_url=response.requested_url,
            status_code=response.status_code,
            retrieved_at_utc=response.retrieved_at_utc,
            final_url=response.final_url,
            redirect_chain=response.redirect_chain,
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

    def fetch_bars_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        timeframe: str,
        page_token: str | None = None,
        asof: date | None = None,
        limit: int = 10_000,
        retries: int = 5,
    ) -> AlpacaBarsPage:
        """Fetch one auditable multi-symbol historical-bars page."""

        normalized = tuple(
            dict.fromkeys(
                symbol.upper().strip()
                for symbol in symbols
                if symbol.strip()
            )
        )
        if not normalized:
            raise ValueError("Alpaca bars page requires at least one symbol")
        if len(normalized) > 50:
            raise ValueError("Alpaca bars page supports at most 50 symbols")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError("Alpaca bars page bounds must be timezone-aware")
        if start >= end:
            raise ValueError("Alpaca bars page start must precede end")
        if limit < 1 or limit > 10_000:
            raise ValueError("Alpaca bars page limit must be 1..10000")
        if asof is None:
            raise ValueError("Alpaca bars page requires an explicit point-in-time asof date")
        if self.settings.alpaca_stock_feed.lower().strip() != "sip":
            raise ValueError("Alpaca bars page requires the consolidated SIP feed")
        params: dict[str, Any] = {
            "symbols": ",".join(normalized),
            "timeframe": timeframe,
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": self.settings.alpaca_stock_feed,
            "limit": limit,
            "adjustment": "all",
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        params["asof"] = asof.isoformat()
        response = self.client.get_bytes_with_metadata(
            self.bars_url,
            params=params,
            headers=self.headers,
            retries=retries,
            maximum_body_bytes=32 * 1024 * 1024,
            allow_redirects=False,
        )
        if (
            response.status_code != 200
            or response.redirect_chain
            or response.final_url != response.requested_url
        ):
            raise RuntimeError("Alpaca bars page transport must be a direct HTTP 200 response")
        if (
            response.retrieved_at_utc.tzinfo is None
            or response.retrieved_at_utc.utcoffset() is None
        ):
            raise RuntimeError("Alpaca bars page retrieval time must be timezone-aware")
        _verify_bars_page_request_url(response.requested_url, params=params)
        if (response.content_type or "").split(";", maxsplit=1)[0].strip().lower() != "application/json":
            raise RuntimeError("Alpaca bars page response must use application/json")
        if (
            response.body_length != len(response.body)
            or response.sha256 != sha256(response.body).hexdigest()
            or response.body_representation != "http_entity_encoded"
        ):
            raise RuntimeError("Alpaca bars page response body metadata is inconsistent")
        try:
            payload = json.loads(response.body.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise RuntimeError("Alpaca bars page response is not valid UTF-8 JSON") from exc
        if not isinstance(payload, dict):
            raise RuntimeError("Alpaca bars page response must be an object")
        raw_bars = payload.get("bars", {})
        if not isinstance(raw_bars, dict):
            raise RuntimeError("Alpaca bars page has invalid bars")
        unexpected = sorted(
            set(str(symbol).upper() for symbol in raw_bars).difference(
                normalized
            )
        )
        if unexpected:
            raise RuntimeError(
                "Alpaca bars page returned unexpected symbols: "
                + ", ".join(unexpected)
            )
        bars: dict[str, tuple[dict[str, Any], ...]] = {}
        for symbol, rows in raw_bars.items():
            if not isinstance(rows, list) or any(
                not isinstance(row, dict) for row in rows
            ):
                raise RuntimeError(
                    f"Alpaca bars page has invalid rows for {symbol}"
                )
            bars[str(symbol).upper()] = tuple(
                {str(key): value for key, value in row.items()}
                for row in rows
            )
        next_value = payload.get("next_page_token")
        next_token = (
            str(next_value).strip()
            if next_value is not None and str(next_value).strip()
            else None
        )
        return AlpacaBarsPage(
            request_page_token=page_token,
            next_page_token=next_token,
            bars=bars,
            response_headers={
                str(key): str(value)
                for key, value in response.safe_headers
            },
            raw_payload={
                str(key): value for key, value in payload.items()
            },
            raw_body=response.body,
            requested_url=response.requested_url,
            status_code=response.status_code,
            retrieved_at_utc=response.retrieved_at_utc,
            final_url=response.final_url,
            redirect_chain=response.redirect_chain,
        )

    def fetch_trades_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        page_token: str | None = None,
        asof: date | None = None,
        limit: int = 10_000,
        retries: int = 5,
    ) -> AlpacaTradesPage:
        """Fetch one auditable page of historical SIP trades."""

        rows, headers, raw, next_token = self._fetch_market_event_page(
            url=self.trades_url,
            response_key="trades",
            symbols=symbols,
            start=start,
            end=end,
            page_token=page_token,
            asof=asof,
            limit=limit,
            retries=retries,
        )
        return AlpacaTradesPage(
            request_page_token=page_token,
            next_page_token=next_token,
            trades=rows,
            response_headers=headers,
            raw_payload=raw,
        )

    def fetch_quotes_page(
        self,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        *,
        page_token: str | None = None,
        asof: date | None = None,
        limit: int = 10_000,
        retries: int = 5,
    ) -> AlpacaQuotesPage:
        """Fetch one auditable page of historical SIP NBBO quotes."""

        rows, headers, raw, next_token = self._fetch_market_event_page(
            url=self.quotes_url,
            response_key="quotes",
            symbols=symbols,
            start=start,
            end=end,
            page_token=page_token,
            asof=asof,
            limit=limit,
            retries=retries,
        )
        return AlpacaQuotesPage(
            request_page_token=page_token,
            next_page_token=next_token,
            quotes=rows,
            response_headers=headers,
            raw_payload=raw,
        )

    def _fetch_market_event_page(
        self,
        *,
        url: str,
        response_key: str,
        symbols: tuple[str, ...],
        start: datetime,
        end: datetime,
        page_token: str | None,
        asof: date | None,
        limit: int,
        retries: int,
    ) -> tuple[
        dict[str, tuple[dict[str, Any], ...]],
        dict[str, str],
        dict[str, Any],
        str | None,
    ]:
        normalized = tuple(
            dict.fromkeys(symbol.upper().strip() for symbol in symbols if symbol.strip())
        )
        if not normalized:
            raise ValueError(f"Alpaca {response_key} page requires at least one symbol")
        if len(normalized) > 50:
            raise ValueError(f"Alpaca {response_key} page supports at most 50 symbols")
        if start.tzinfo is None or end.tzinfo is None:
            raise ValueError(f"Alpaca {response_key} page bounds must be timezone-aware")
        if start >= end:
            raise ValueError(f"Alpaca {response_key} page start must precede end")
        if limit < 1 or limit > 10_000:
            raise ValueError(f"Alpaca {response_key} page limit must be 1..10000")
        params: dict[str, Any] = {
            "symbols": ",".join(normalized),
            "start": start.isoformat(),
            "end": end.isoformat(),
            "feed": self.settings.alpaca_stock_feed,
            "limit": limit,
            "sort": "asc",
        }
        if page_token:
            params["page_token"] = page_token
        if asof is not None:
            params["asof"] = asof.isoformat()
        payload, response_headers = self.client.get_json_with_headers(
            url,
            params=params,
            headers=self.headers,
            retries=retries,
        )
        if not isinstance(payload, dict):
            raise RuntimeError(f"Alpaca {response_key} page response must be an object")
        raw_rows = payload.get(response_key, {})
        if not isinstance(raw_rows, dict):
            raise RuntimeError(f"Alpaca {response_key} page has invalid {response_key}")
        unexpected = sorted(
            set(str(symbol).upper() for symbol in raw_rows).difference(normalized)
        )
        if unexpected:
            raise RuntimeError(
                f"Alpaca {response_key} page returned unexpected symbols: "
                + ", ".join(unexpected)
            )
        records: dict[str, tuple[dict[str, Any], ...]] = {}
        for symbol, values in raw_rows.items():
            if not isinstance(values, list) or any(
                not isinstance(value, dict) for value in values
            ):
                raise RuntimeError(
                    f"Alpaca {response_key} page has invalid rows for {symbol}"
                )
            records[str(symbol).upper()] = tuple(
                {str(key): value for key, value in value.items()} for value in values
            )
        next_value = payload.get("next_page_token")
        next_token = (
            str(next_value).strip()
            if next_value is not None and str(next_value).strip()
            else None
        )
        return (
            records,
            {str(key): str(value) for key, value in response_headers.items()},
            {str(key): value for key, value in payload.items()},
            next_token,
        )

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
