from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, cast
from urllib.parse import parse_qsl

import pandas as pd

from market_predictor.config import Settings
from market_predictor.quota import MonthlyQuotaTracker, QuotaStatus
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


class SeekingAlphaRapidApiSource:
    TICKER_ALIASES = {
        "GOOG": {"GOOG", "GOOGL"},
        "GOOGL": {"GOOG", "GOOGL"},
        "BRK.A": {"BRK.A", "BRK.B", "BRK-A", "BRK-B"},
        "BRK.B": {"BRK.A", "BRK.B", "BRK-A", "BRK-B"},
    }

    def __init__(self, settings: Settings) -> None:
        if not settings.has_seeking_alpha_rapidapi:
            raise ValueError("RAPIDAPI_KEY and SEEKING_ALPHA_RAPIDAPI_HOST are required.")
        self.settings = settings
        self.client = HttpClient()
        self.quota = MonthlyQuotaTracker(
            settings.seeking_alpha_usage_file,
            "seeking_alpha_rapidapi",
            settings.seeking_alpha_monthly_request_limit,
        )

    @property
    def headers(self) -> dict[str, str]:
        return {
            "X-RapidAPI-Key": self.settings.rapidapi_key or "",
            "X-RapidAPI-Host": self.settings.seeking_alpha_rapidapi_host,
        }

    @property
    def base_url(self) -> str:
        return f"https://{self.settings.seeking_alpha_rapidapi_host}"

    def fetch_events(self, ticker: str, start: datetime) -> list[NewsEvent]:
        events, _errors = self.fetch_events_with_errors(ticker, start)
        return events

    def fetch_events_with_errors(self, ticker: str, start: datetime) -> tuple[list[NewsEvent], list[str]]:
        events: list[NewsEvent] = []
        errors: list[str] = []
        for feed in self.settings.seeking_alpha_event_feeds:
            name = str(feed.get("name", "event"))
            try:
                events.extend(self._fetch_event_feed(ticker, start, feed))
            except Exception as exc:
                message = f"{name}:{exc}"
                errors.append(message)
                if "status=429" in message and "MONTHLY quota" in message:
                    break
        return self._dedupe_events(events), errors

    def fetch_market_context_events(self, start: datetime) -> list[NewsEvent]:
        events: list[NewsEvent] = []
        for feed in self.settings.seeking_alpha_event_feeds:
            if not bool(feed.get("market_context", False)):
                continue
            events.extend(self._fetch_event_feed("MARKET", start, feed, require_relevance=False))
        return self._dedupe_events(events)

    def fetch_analysis(self, ticker: str, start: datetime, limit: int = 40) -> list[NewsEvent]:
        feed = {
            "name": "analysis",
            "endpoint": self.settings.seeking_alpha_analysis_endpoint,
            "params": self.settings.seeking_alpha_analysis_params,
            "cache_hours": self.settings.seeking_alpha_analysis_cache_hours,
            "limit": limit,
        }
        return self._fetch_event_feed(ticker, start, feed)

    def _fetch_event_feed(
        self,
        ticker: str,
        start: datetime,
        feed: dict[str, Any],
        *,
        require_relevance: bool = True,
    ) -> list[NewsEvent]:
        name = str(feed.get("name", "event"))
        limit = int(feed.get("limit", 40))
        params = self._template_params(
            str(feed["params"]),
            ticker,
            extra={"limit": str(limit), "size": str(limit)},
        )
        payload = self._get_json_cached(
            str(feed["endpoint"]),
            params,
            cache_group=f"events_{name}",
            cache_hours=int(feed.get("cache_hours", 24)),
        )
        events: list[NewsEvent] = []
        tag_symbols = self._tag_symbol_map(payload)
        for item in self._iter_event_items(payload):
            title = self._pick(item, ["title", "headline"])
            raw_nested = item.get("attributes")
            nested = cast(dict[str, Any], raw_nested) if isinstance(raw_nested, dict) else {}
            if not title and nested:
                title = self._pick(nested, ["title", "headline"])
            if not title:
                continue
            timestamp = self._parse_time(
                self._pick(item, ["publishOn", "publishedAt", "createdAt", "date"])
                or self._pick(nested, ["publishOn", "publishedAt", "createdAt", "date"])
            )
            if timestamp is None:
                continue
            if timestamp < start.astimezone(UTC):
                continue
            summary = self._pick(item, ["summary", "commentary"]) or self._pick(nested, ["summary", "commentary"])
            item_symbols = self._item_symbols(item, tag_symbols)
            if require_relevance and not self._is_ticker_relevant(ticker, item_symbols, str(title), str(summary) if summary else ""):
                continue
            raw = dict(item)
            raw["_matched_symbols"] = sorted(item_symbols)
            url = self._url_from_item(item)
            events.append(
                NewsEvent(
                    ticker=ticker.upper(),
                    timestamp=timestamp,
                    source=f"seeking_alpha:rapidapi_{name}",
                    title=str(title),
                    url=str(url) if url else None,
                    summary=str(summary) if summary else None,
                    text=" ".join(str(part) for part in [title, summary] if part).strip(),
                    raw=raw,
                )
            )
        return events[:limit]

    def fetch_quant_snapshot(self, ticker: str) -> dict[str, Any]:
        snapshot = {
            "timestamp": datetime.now(UTC).isoformat(),
            "ticker": ticker.upper(),
        }
        template_values: dict[str, str] = {}
        for feed in self.settings.seeking_alpha_snapshot_feeds:
            name = str(feed.get("name", "snapshot"))
            missing = [str(item) for item in feed.get("requires", []) if not template_values.get(str(item))]
            if missing:
                snapshot[f"{name}_skipped"] = f"missing required values: {','.join(missing)}"
                continue
            try:
                params = self._template_params(str(feed["params"]), ticker, extra=template_values)
                payload = self._get_json_cached(
                    str(feed["endpoint"]),
                    params,
                    cache_group=f"snapshot_{name}",
                    cache_hours=int(feed.get("cache_hours", 24)),
                )
            except Exception as exc:
                snapshot[f"{name}_error"] = str(exc)
                continue
            flat = self._flatten_snapshot_payload(payload)
            if name == "ratings":
                snapshot.update(self._extract_ratings(flat))
            elif name == "earnings":
                snapshot.update(self._extract_earnings(flat))
            else:
                snapshot.update(self._prefix_scalar_fields(name, flat))
            ticker_id = self._extract_ticker_id(payload, flat)
            if ticker_id:
                template_values.setdefault("ticker_id", ticker_id)
                snapshot.setdefault("seeking_alpha_ticker_id", ticker_id)
        return snapshot

    def get_account_access_token(self, *, force_refresh: bool = False) -> str:
        if not self.settings.has_seeking_alpha_account_credentials:
            raise ValueError(
                "SEEKING_ALPHA_ACCOUNT_EMAIL and SEEKING_ALPHA_ACCOUNT_PASSWORD are required "
                "to request a Seeking Alpha access token."
            )
        if not force_refresh:
            cached = self._read_cached_access_token()
            if cached:
                return cached
        if self.settings.seeking_alpha_fail_when_monthly_limit_reached:
            self.quota.assert_available()
        payload, headers = self.client.post_json_with_headers(
            f"{self.base_url}{self.settings.seeking_alpha_access_token_endpoint}",
            payload={
                "email": self.settings.seeking_alpha_account_email,
                "password": self.settings.seeking_alpha_account_password,
            },
            headers={**self.headers, "Content-Type": "application/json"},
        )
        self.quota.record_call(headers)
        token = self._extract_access_token(payload)
        self._write_cached_access_token(token, payload)
        return token

    def quota_status(self) -> QuotaStatus:
        return self.quota.status()

    def account_token_status(self) -> dict[str, Any]:
        token = self._read_cached_access_token()
        return {
            "credentials_configured": self.settings.has_seeking_alpha_account_credentials,
            "cached_token_available": bool(token),
            "cache_file": str(self.settings.seeking_alpha_access_token_cache_file),
        }

    def _get_json_cached(
        self,
        endpoint: str,
        params: dict[str, str],
        *,
        cache_group: str,
        cache_hours: int,
    ) -> Any:
        cache_path = self._cache_path(endpoint, params, cache_group)
        cached = self._read_cache(cache_path, cache_hours)
        if cached is not None:
            return cached
        if self.settings.seeking_alpha_fail_when_monthly_limit_reached:
            self.quota.assert_available()
        payload, headers = self.client.get_json_with_headers(
            f"{self.base_url}{endpoint}",
            params=params,
            headers=self.headers,
        )
        self.quota.record_call(headers)
        self._write_cache(cache_path, payload)
        return payload

    def _cache_path(self, endpoint: str, params: dict[str, str], cache_group: str) -> Path:
        material = json.dumps(
            {"endpoint": endpoint, "params": params},
            sort_keys=True,
            separators=(",", ":"),
        )
        digest = hashlib.sha256(material.encode("utf-8")).hexdigest()[:24]
        return self.settings.seeking_alpha_cache_dir / cache_group / f"{digest}.json"

    @staticmethod
    def _read_cache(path: Path, cache_hours: int) -> Any | None:
        if cache_hours <= 0 or not path.exists():
            return None
        modified = datetime.fromtimestamp(path.stat().st_mtime, tz=UTC)
        if datetime.now(UTC) - modified > timedelta(hours=cache_hours):
            return None
        return json.loads(path.read_text(encoding="utf-8"))

    @staticmethod
    def _write_cache(path: Path, payload: Any) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(payload, ensure_ascii=True), encoding="utf-8")

    def _read_cached_access_token(self) -> str | None:
        path = self.settings.seeking_alpha_access_token_cache_file
        cached = self._read_cache(path, self.settings.seeking_alpha_access_token_cache_hours)
        if not isinstance(cached, dict):
            return None
        token = cached.get("access_token")
        return str(token) if token else None

    def _write_cached_access_token(self, token: str, raw_payload: Any) -> None:
        self._write_cache(
            self.settings.seeking_alpha_access_token_cache_file,
            {
                "access_token": token,
                "cached_at": datetime.now(UTC).isoformat(),
                "raw_keys": sorted(raw_payload.keys()) if isinstance(raw_payload, dict) else [],
            },
        )

    @classmethod
    def _extract_access_token(cls, payload: Any) -> str:
        if isinstance(payload, dict):
            for key in ["access_token", "accessToken", "token", "jwt"]:
                value = payload.get(key)
                if value:
                    return str(value)
            for item in cls._iter_dicts(payload):
                for key in ["access_token", "accessToken", "token", "jwt"]:
                    value = item.get(key)
                    if value:
                        return str(value)
        raise ValueError("Seeking Alpha access token response did not contain a recognized token field.")

    @staticmethod
    def _dedupe_events(events: list[NewsEvent]) -> list[NewsEvent]:
        seen: set[tuple[str, str, str | None]] = set()
        deduped: list[NewsEvent] = []
        for event in sorted(events, key=lambda item: item.timestamp):
            key = (event.source, event.title, event.url)
            if key in seen:
                continue
            seen.add(key)
            deduped.append(event)
        return deduped

    @staticmethod
    def _template_params(template: str, ticker: str, extra: dict[str, str] | None = None) -> dict[str, str]:
        values = {
            "ticker": ticker.upper(),
            "ticker_lower": ticker.lower(),
            **(extra or {}),
        }
        params: dict[str, str] = {}
        for key, value in parse_qsl(template, keep_blank_values=True):
            params[key] = value.format(**values)
        return params

    @classmethod
    def _iter_dicts(cls, value: Any) -> Iterator[dict[str, Any]]:
        if isinstance(value, dict):
            yield cast(dict[str, Any], value)
            for child in value.values():
                yield from cls._iter_dicts(child)
        elif isinstance(value, list):
            for child in value:
                yield from cls._iter_dicts(child)

    @staticmethod
    def _iter_event_items(payload: Any) -> Iterator[dict[str, Any]]:
        if isinstance(payload, dict) and isinstance(payload.get("data"), list):
            for item in payload["data"]:
                if isinstance(item, dict):
                    yield cast(dict[str, Any], item)
            return
        yield from SeekingAlphaRapidApiSource._iter_dicts(payload)

    @staticmethod
    def _tag_symbol_map(payload: Any) -> dict[str, str]:
        symbols: dict[str, str] = {}
        if not isinstance(payload, dict) or not isinstance(payload.get("included"), list):
            return symbols
        for item in payload["included"]:
            if not isinstance(item, dict) or item.get("type") != "tag":
                continue
            item = cast(dict[str, Any], item)
            tag_id = item.get("id")
            raw_attributes = item.get("attributes")
            attributes = cast(dict[str, Any], raw_attributes) if isinstance(raw_attributes, dict) else {}
            symbol = (
                attributes.get("name")
                or attributes.get("slug")
                or attributes.get("symbol")
                or attributes.get("ticker")
            )
            if tag_id is not None and symbol:
                symbols[str(tag_id)] = str(symbol).upper()
        return symbols

    @classmethod
    def _item_symbols(cls, item: dict[str, Any], tag_symbols: dict[str, str]) -> set[str]:
        raw_relationships = item.get("relationships")
        relationships = cast(dict[str, Any], raw_relationships) if isinstance(raw_relationships, dict) else {}
        symbols: set[str] = set()
        for key in ["primaryTickers", "secondaryTickers"]:
            relationship = relationships.get(key)
            if not isinstance(relationship, dict):
                continue
            entries = relationship.get("data")
            if isinstance(entries, dict):
                entries = [entries]
            if not isinstance(entries, list):
                continue
            for entry in entries:
                if not isinstance(entry, dict):
                    continue
                tag_id = entry.get("id")
                symbol = tag_symbols.get(str(tag_id)) if tag_id is not None else None
                if symbol:
                    symbols.add(symbol.upper())
        return symbols

    @classmethod
    def _is_ticker_relevant(cls, ticker: str, symbols: set[str], title: str, summary: str) -> bool:
        ticker_upper = ticker.upper()
        aliases = cls.TICKER_ALIASES.get(ticker_upper, {ticker_upper})
        if symbols:
            return bool(symbols & aliases)

        # If the API payload has no tag metadata, fall back only to explicit symbol text.
        # This keeps broad SA feeds from becoming ticker-specific training examples.
        text = f"{title} {summary}".upper()
        return any(re.search(rf"(?<![A-Z0-9])\$?{re.escape(alias)}(?![A-Z0-9])", text) for alias in aliases)

    @classmethod
    def _url_from_item(cls, item: dict[str, Any]) -> Any:
        raw_nested = item.get("attributes")
        nested = cast(dict[str, Any], raw_nested) if isinstance(raw_nested, dict) else {}
        url = cls._pick(item, ["url"]) or cls._pick(nested, ["url"])
        if url:
            return url
        raw_links = item.get("links")
        links = cast(dict[str, Any], raw_links) if isinstance(raw_links, dict) else {}
        return links.get("self") or links.get("canonical") or links.get("uri")

    @staticmethod
    def _pick(item: dict[str, Any], keys: list[str]) -> Any:
        for key in keys:
            if key in item and item[key] not in (None, ""):
                return item[key]
        return None

    @staticmethod
    def _parse_time(value: Any) -> datetime | None:
        if not value:
            return None
        try:
            parsed = pd.to_datetime(value, utc=True, errors="coerce")
        except Exception:
            return None
        if pd.isna(parsed):
            return None
        return cast(datetime, parsed.to_pydatetime())

    @classmethod
    def _flatten_first_rating(cls, payload: Any) -> dict[str, Any]:
        return cls._flatten_first_interesting_payload(payload)

    @classmethod
    def _flatten_first_interesting_payload(cls, payload: Any) -> dict[str, Any]:
        best: dict[str, Any] = {}
        for item in cls._iter_dicts(payload):
            keys = {str(key).lower() for key in item}
            if keys & {
                "quantrating",
                "quant_rating",
                "valuation",
                "growth",
                "profitability",
                "momentum",
                "epsactual",
                "epsestimate",
                "eps_actual",
                "eps_estimate",
                "revenueactual",
                "revenueestimate",
                "reportdate",
                "earningsdate",
            }:
                best.update(item)
                attributes = item.get("attributes")
                if isinstance(attributes, dict):
                    best.update(attributes)
                return best
        return best

    @classmethod
    def _flatten_snapshot_payload(cls, payload: Any) -> dict[str, Any]:
        interesting = cls._flatten_first_interesting_payload(payload)
        if interesting:
            flat = dict(interesting)
            flat.update(cls._flatten_nested_scalars(payload, max_fields=80))
            return flat
        record = cls._first_payload_record(payload)
        fallback_flat: dict[str, Any] = {}
        if record:
            for key, value in record.items():
                if key == "attributes" and isinstance(value, dict):
                    fallback_flat.update(value)
                elif cls._is_scalar(value):
                    fallback_flat[str(key)] = value
            attributes = record.get("attributes")
            if isinstance(attributes, dict):
                for key, value in attributes.items():
                    if cls._is_scalar(value):
                        fallback_flat[str(key)] = value
        fallback_flat.update(
            {
                key: value
                for key, value in cls._flatten_nested_scalars(payload, max_fields=80).items()
                if key not in fallback_flat
            }
        )
        return fallback_flat

    @classmethod
    def _first_payload_record(cls, payload: Any) -> dict[str, Any]:
        if isinstance(payload, dict):
            data = payload.get("data")
            if isinstance(data, dict):
                return cast(dict[str, Any], data)
            if isinstance(data, list):
                for item in data:
                    if isinstance(item, dict):
                        return cast(dict[str, Any], item)
            return cast(dict[str, Any], payload)
        if isinstance(payload, list):
            for item in payload:
                if isinstance(item, dict):
                    return cast(dict[str, Any], item)
        for item in cls._iter_dicts(payload):
            if item:
                return item
        return {}

    @classmethod
    def _extract_ticker_id(cls, payload: Any, flat: dict[str, Any]) -> str | None:
        for key in ["ticker_id", "tickerId", "id", "tag_id", "tagId", "sa_id", "saId"]:
            value = flat.get(key)
            if value not in (None, "") and str(value).isdigit():
                return str(value)
        for item in cls._iter_dicts(payload):
            item_type = str(item.get("type", "")).lower()
            item_id = item.get("id")
            if item_id not in (None, "") and str(item_id).isdigit() and item_type in {"tag", "symbol", "ticker"}:
                return str(item_id)
            attributes = item.get("attributes")
            if isinstance(attributes, dict):
                for key in ["ticker_id", "tickerId", "id", "tag_id", "tagId"]:
                    value = attributes.get(key)
                    if value not in (None, "") and str(value).isdigit():
                        return str(value)
        return None

    @classmethod
    def _prefix_scalar_fields(cls, prefix: str, flat: dict[str, Any]) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        for key, value in flat.items():
            if cls._is_scalar(value):
                safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_")
                if safe_key:
                    fields[f"{prefix}_{safe_key}"] = value
        return fields

    @classmethod
    def _flatten_nested_scalars(
        cls,
        value: Any,
        *,
        prefix: str = "",
        depth: int = 0,
        max_depth: int = 6,
        max_fields: int = 80,
    ) -> dict[str, Any]:
        fields: dict[str, Any] = {}
        if len(fields) >= max_fields or depth > max_depth:
            return fields
        if cls._is_scalar(value):
            if prefix:
                fields[prefix] = value
            return fields
        if isinstance(value, dict):
            for key, child in value.items():
                if len(fields) >= max_fields:
                    break
                safe_key = re.sub(r"[^A-Za-z0-9_]+", "_", str(key)).strip("_")
                if not safe_key:
                    continue
                child_prefix = f"{prefix}_{safe_key}" if prefix else safe_key
                if cls._is_scalar(child):
                    fields[child_prefix] = child
                elif isinstance(child, list):
                    fields[f"{child_prefix}_count"] = len(child)
                    if child:
                        fields.update(
                            cls._flatten_nested_scalars(
                                child[0],
                                prefix=f"{child_prefix}_first",
                                depth=depth + 1,
                                max_depth=max_depth,
                                max_fields=max_fields - len(fields),
                            )
                        )
                elif isinstance(child, dict):
                    fields.update(
                        cls._flatten_nested_scalars(
                            child,
                            prefix=child_prefix,
                            depth=depth + 1,
                            max_depth=max_depth,
                            max_fields=max_fields - len(fields),
                        )
                    )
            return fields
        if isinstance(value, list):
            fields[f"{prefix}_count" if prefix else "record_count"] = len(value)
            if value:
                fields.update(
                    cls._flatten_nested_scalars(
                        value[0],
                        prefix=f"{prefix}_first" if prefix else "first",
                        depth=depth + 1,
                        max_depth=max_depth,
                        max_fields=max_fields - len(fields),
                    )
                )
        return fields

    @staticmethod
    def _field(item: dict[str, Any], names: list[str]) -> Any:
        lowered = {str(key).lower(): value for key, value in item.items()}
        for name in names:
            value = lowered.get(name.lower())
            if isinstance(value, dict):
                return value.get("value") or value.get("grade") or value.get("rating")
            if value not in (None, ""):
                return value
        return None

    def _extract_ratings(self, flat: dict[str, Any]) -> dict[str, Any]:
        return {
            "quant_rating": self._field(flat, ["quant_rating", "quantRating", "rating", "overall"]),
            "valuation": self._field(flat, ["valuation", "value"]),
            "growth": self._field(flat, ["growth"]),
            "profitability": self._field(flat, ["profitability"]),
            "momentum": self._field(flat, ["momentum"]),
            "eps_revision": self._field(flat, ["eps_revision", "epsRevisions", "revisions"]),
        }

    def _extract_earnings(self, flat: dict[str, Any]) -> dict[str, Any]:
        return {
            "eps_actual": self._field(flat, ["eps_actual", "epsActual", "actualEPS", "eps"]),
            "eps_estimate": self._field(flat, ["eps_estimate", "epsEstimate", "consensusEPS", "estimateEPS"]),
            "revenue_actual": self._field(flat, ["revenue_actual", "revenueActual", "actualRevenue", "revenue"]),
            "revenue_estimate": self._field(
                flat,
                ["revenue_estimate", "revenueEstimate", "consensusRevenue", "estimateRevenue"],
            ),
            "earnings_date": self._field(flat, ["earnings_date", "earningsDate", "reportDate", "date"]),
            "fiscal_period": self._field(flat, ["fiscal_period", "fiscalPeriod", "period", "quarter"]),
        }

    @staticmethod
    def _is_scalar(value: Any) -> bool:
        return value is None or isinstance(value, str | int | float | bool)
