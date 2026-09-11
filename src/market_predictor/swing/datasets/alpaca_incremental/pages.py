"""Strict raw-page receipts shared by acquisition and offline replay."""
from __future__ import annotations

import base64
from dataclasses import asdict, dataclass
from datetime import UTC, date, datetime, time, timedelta
from hashlib import sha256
from typing import Any
from urllib.parse import parse_qsl, urlsplit

from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.alpaca import AlpacaBarsPage, AlpacaNewsPage, AlpacaSource
from market_predictor.swing.datasets.alpaca_incremental.config import Config

MAX_BODY = 32 * 1024**2
MAX_LINE = 48 * 1024**2
SAFE_HEADERS = frozenset(("cache-control", "content-encoding", "content-language", "content-type", "date", "etag",
    "expires", "last-modified", "retry-after", "x-ratelimit-limit", "x-ratelimit-remaining", "x-ratelimit-reset"))


@dataclass(frozen=True)
class Unit:
    day: date
    family: str
    symbols: tuple[str, ...]
    capture: str = "daily"
    cutoff: datetime | None = None

    @property
    def start(self) -> datetime:
        return datetime.combine(self.day, time(), UTC)

    @property
    def end(self) -> datetime:
        return self.cutoff or self.start + timedelta(days=1, microseconds=-1)

    def record(self) -> dict[str, Any]:
        return {"day": self.day.isoformat(), "family": self.family, "symbols": list(self.symbols),
            "capture": self.capture, "start": self.start.isoformat(), "end_inclusive": self.end.isoformat(),
            "partial": self.cutoff is not None}

    @property
    def key(self) -> str:
        return f"{self.day.isoformat()}-{self.family}-{json_sha256(self.record())}"

    def params(self, config: Config, token: str | None) -> dict[str, str]:
        params = {"symbols": ",".join(self.symbols), "start": self.start.isoformat(), "end": self.end.isoformat(), "sort": "asc"}
        if self.family == "news":
            params.update(limit=str(config.news_limit), include_content="true")
        else:
            params.update(timeframe="1Day", feed="sip", adjustment=self.family.removeprefix("bars_"),
                limit=str(config.bars_limit), asof=self.day.isoformat())
        if token is not None:
            params["page_token"] = token
        return params

    def fetch(self, source: AlpacaSource, config: Config, token: str | None) -> AlpacaBarsPage | AlpacaNewsPage:
        if self.family == "news":
            return source.fetch_news_page_observed(",".join(self.symbols), self.start, self.end,
                page_token=token, include_content=True, limit=config.news_limit)
        return source.fetch_bars_page(self.symbols, self.start, self.end, timeframe="1Day", page_token=token,
            asof=self.day, limit=config.bars_limit, adjustment=self.family.removeprefix("bars_"))


def _url(value: object, unit: Unit, config: Config, token: str | None) -> None:
    if not isinstance(value, str):
        raise ValueError("missing response URL")
    url = urlsplit(value)
    pairs = parse_qsl(url.query, keep_blank_values=True)
    endpoint = "/v1beta1/news" if unit.family == "news" else "/v2/stocks/bars"
    if (url.scheme != "https" or url.hostname != "data.alpaca.markets" or url.port not in (None, 443)
            or url.username is not None or url.password is not None or url.fragment or url.path != endpoint
            or len(pairs) != len(unit.params(config, token)) or dict(pairs) != unit.params(config, token)):
        raise ValueError("response URL does not match frozen request")


def page_record(page: AlpacaBarsPage | AlpacaNewsPage, unit: Unit, config: Config, token: str | None) -> dict[str, Any]:
    body = page.raw_body
    if body is None or len(body) > MAX_BODY:
        raise ValueError("missing or oversized exact raw body")
    payload = parse_strict_json_object(body, label="Alpaca incremental page")
    if page.raw_payload != payload or page.request_page_token != token:
        raise ValueError("adapter raw payload or request token mismatch")
    if isinstance(page, AlpacaNewsPage):
        if list(page.news) != payload.get("news"):
            raise ValueError("news adapter differs from exact raw body")
    elif {symbol: list(rows) for symbol, rows in page.bars.items()} != payload.get("bars"):
        raise ValueError("bars adapter differs from exact raw body")
    record: dict[str, Any] = {
        "request_page_token": token, "next_page_token": page.next_page_token,
        "requested_url": page.requested_url, "final_url": page.final_url,
        "redirect_chain": list(page.redirect_chain), "status_code": page.status_code,
        "retrieved_at_utc": page.retrieved_at_utc.isoformat() if page.retrieved_at_utc else None,
        "response_headers": {key: value for key, value in page.response_headers.items() if key.lower() in SAFE_HEADERS},
        "body_base64": base64.b64encode(body).decode("ascii"), "body_sha256": sha256(body).hexdigest(),
        "body_length": len(body), "body_representation": "http_entity_encoded",
    }
    if isinstance(page, AlpacaBarsPage) and page.transport_response is not None:
        metadata = asdict(page.transport_response)
        del metadata["body"]
        metadata["retrieved_at_utc"] = page.transport_response.retrieved_at_utc.isoformat()
        metadata["safe_headers"] = [[key, value] for key, value in page.transport_response.safe_headers if key.lower() in SAFE_HEADERS]
        metadata["redirect_chain"] = list(page.transport_response.redirect_chain)
        record["transport_metadata"] = metadata
    validate_record(record, unit, config, token)
    return record


def validate_record(record: dict[str, Any], unit: Unit, config: Config, token: str | None) -> tuple[dict[str, int], int, str | None]:
    if record.get("request_page_token") != token or record.get("status_code") != 200 or record.get("redirect_chain") != []:
        raise ValueError("invalid page token, status or redirect metadata")
    _url(record.get("requested_url"), unit, config, token)
    _url(record.get("final_url"), unit, config, token)
    if record["requested_url"] != record["final_url"]:
        raise ValueError("response was redirected")
    timestamp = datetime.fromisoformat(record["retrieved_at_utc"])
    if timestamp.utcoffset() != timedelta(0) or timestamp < unit.end or timestamp > datetime.now(UTC):
        raise ValueError("retrieval timestamp must be UTC and no earlier than the request cutoff")
    if unit.capture.startswith("revision-"):
        capture_start = datetime.combine(date.fromisoformat(unit.capture.removeprefix("revision-")), time(), UTC)
        if timestamp < capture_start:
            raise ValueError("revision retrieval predates its capture date")
    headers = record.get("response_headers")
    if not isinstance(headers, dict) or any(not isinstance(k, str) or k.lower() not in SAFE_HEADERS
            or not isinstance(v, str) or len(v) > 2048 for k, v in headers.items()):
        raise ValueError("unsafe response headers")
    normalized = {k.lower(): v for k, v in headers.items()}
    if normalized.get("content-type", "").split(";", 1)[0].strip().lower() != "application/json":
        raise ValueError("response must be JSON")
    if normalized.get("content-encoding", "identity").lower() not in ("", "identity"):
        raise ValueError("encoded JSON entity is unsupported")
    encoded = record.get("body_base64")
    if not isinstance(encoded, str) or len(encoded) > (MAX_BODY + 2) // 3 * 4:
        raise ValueError("raw body exceeds bounded size")
    body = base64.b64decode(encoded, validate=True)
    if (len(body) > MAX_BODY or len(body) != record.get("body_length") or sha256(body).hexdigest() != record.get("body_sha256")
            or record.get("body_representation") != "http_entity_encoded"):
        raise ValueError("raw body hash or metadata mismatch")
    transport = record.get("transport_metadata")
    if transport is not None:
        if not isinstance(transport, dict) or not isinstance(transport.get("safe_headers"), list):
            raise ValueError("invalid transport metadata")
        transport_headers = transport["safe_headers"]
        if (any(not isinstance(pair, list) or len(pair) != 2 or not all(isinstance(v, str) for v in pair)
                for pair in transport_headers) or len(transport_headers) != len(headers) or dict(transport_headers) != headers):
            raise ValueError("transport headers differ from receipt")
        expected = {"requested_url": record["requested_url"], "final_url": record["final_url"], "redirect_chain": [],
            "status_code": 200, "retrieved_at_utc": record["retrieved_at_utc"], "body_length": len(body),
            "sha256": record["body_sha256"], "body_representation": "http_entity_encoded",
            "safe_headers": transport_headers, "content_type": normalized.get("content-type"),
            "content_encoding": normalized.get("content-encoding"), "etag": normalized.get("etag"),
            "last_modified": normalized.get("last-modified")}
        if transport != expected:
            raise ValueError("transport metadata differs from receipt")
    payload = parse_strict_json_object(body, label="Alpaca incremental page")
    if set(payload).intersection({"error", "message", "code"}):
        raise ValueError("provider error payload cannot establish no-data evidence")
    next_token = payload.get("next_page_token")
    if next_token is not None and (not isinstance(next_token, str) or not next_token
            or len(next_token) > 4096 or next_token.strip() != next_token):
        raise ValueError("invalid continuation token")
    if record.get("next_page_token") != next_token:
        raise ValueError("raw continuation token differs from adapter")
    counts = dict.fromkeys(unit.symbols, 0)
    rows_total = 0
    if unit.family == "news":
        news = payload.get("news")
        if not isinstance(news, list) or len(news) > config.news_limit:
            raise ValueError("missing or oversized news rows")
        for item in news:
            if not isinstance(item, dict):
                raise ValueError("invalid news row")
            if type(item.get("id")) is not int or item["id"] <= 0:
                raise ValueError("news requires a positive provider article id")
            created = _news_timestamp(item.get("created_at"))
            updated = _news_timestamp(item.get("updated_at"))
            # Alpaca's news window includes revisions to older articles. Keep the
            # original creation clock; do not backdate this observed body to it.
            if not unit.start <= updated <= unit.end or not created <= updated <= timestamp:
                raise ValueError("news update outside requested UTC range or inconsistent creation timestamp")
            tags = item.get("symbols", [])
            if not isinstance(tags, list) or any(not isinstance(s, str) for s in tags):
                raise ValueError("invalid news symbol tags")
            for symbol in set(tags).intersection(counts):
                counts[symbol] += 1
        rows_total = len(news)
    else:
        bars = payload.get("bars")
        if not isinstance(bars, dict) or set(bars).difference(counts):
            raise ValueError("missing bars or unexpected symbols")
        for symbol, rows in bars.items():
            if not isinstance(rows, list):
                raise ValueError("invalid bars rows")
            for row in rows:
                if not isinstance(row, dict) or not isinstance(row.get("t"), str):
                    raise ValueError("bar timestamp missing")
                stamp = datetime.fromisoformat(row["t"].replace("Z", "+00:00"))
                if stamp.utcoffset() is None or not unit.start <= stamp <= unit.end:
                    raise ValueError("bar timestamp outside request")
            counts[symbol] = len(rows)
            rows_total += len(rows)
        if rows_total > config.bars_limit:
            raise ValueError("bar page exceeds configured limit")
    return counts, rows_total, next_token


def _news_timestamp(value: object) -> datetime:
    if not isinstance(value, str):
        raise ValueError("news timestamp missing")
    timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if timestamp.utcoffset() != timedelta(0):
        raise ValueError("news timestamp must be UTC")
    return timestamp
