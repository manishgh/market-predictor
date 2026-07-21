from __future__ import annotations

import hashlib
import json
from datetime import datetime
from typing import Any, cast

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.contracts import (
    CANONICAL_SCHEMA_VERSION,
    AvailabilityPolicy,
    CanonicalUniverseMembership,
)
from market_predictor.data_quality import sanitize_events_frame
from market_predictor.features import source_family_for_source
from market_predictor.v3.contracts import normalized_ticker
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError

BAR_DURATIONS = {
    "1m": pd.Timedelta(minutes=1),
    "5m": pd.Timedelta(minutes=5),
    "1h": pd.Timedelta(hours=1),
}
TIMEFRAME_ALIASES = {
    "1min": "1m",
    "1minute": "1m",
    "1t": "1m",
    "5min": "5m",
    "5minute": "5m",
    "5t": "5m",
    "1hour": "1h",
    "1hr": "1h",
    "1day": "1d",
    "1d": "1d",
    "1m": "1m",
    "5m": "5m",
    "1h": "1h",
}
CANONICAL_BAR_COLUMNS = (
    "ticker",
    "timeframe",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "ingested_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "price_feed",
    "adjustment",
    "availability_policy",
    "schema_version",
)
CANONICAL_EVENT_COLUMNS = (
    "event_id",
    "ticker",
    "source_family",
    "source",
    "published_at_utc",
    "provider_updated_at_utc",
    "first_seen_at_utc",
    "available_at_utc",
    "sentiment_scored_at_utc",
    "feature_available_at_utc",
    "title",
    "url",
    "summary",
    "text",
    "sentiment_numeric",
    "relevance",
    "availability_policy",
    "raw_sha256",
    "schema_version",
)
CANONICAL_MEMBERSHIP_COLUMNS = tuple(CanonicalUniverseMembership.model_fields)


def canonicalize_universe_memberships(
    frame: pd.DataFrame,
    *,
    source: str | None = None,
    availability_policy: AvailabilityPolicy | None = None,
) -> pd.DataFrame:
    """Validate observed point-in-time membership without inventing availability."""

    required = {
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
        "available_at_utc",
        "sector",
        "industry",
        "market_cap_bucket",
        "liquidity_bucket",
        "primary_benchmark",
        "universe_snapshot_id",
    }
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"universe membership input is missing columns: {', '.join(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_MEMBERSHIP_COLUMNS)
    data = frame.copy()
    if "source" not in data.columns:
        if source is None:
            raise SchemaMismatchError("universe membership input requires source")
        data["source"] = source
    if "availability_policy" not in data.columns:
        if availability_policy is None:
            raise SchemaMismatchError("universe membership input requires availability_policy")
        data["availability_policy"] = availability_policy
    data["schema_version"] = CANONICAL_SCHEMA_VERSION
    data = data.loc[:, list(CANONICAL_MEMBERSHIP_COLUMNS)]
    records: list[dict[str, object]] = []
    for raw in data.to_dict(orient="records"):
        if pd.isna(raw.get("effective_to_utc")):
            raw["effective_to_utc"] = None
        membership = CanonicalUniverseMembership.model_validate(raw)
        records.append(membership.model_dump())
    return pd.DataFrame(records, columns=CANONICAL_MEMBERSHIP_COLUMNS).sort_values(
        ["ticker", "effective_from_utc", "available_at_utc"],
        kind="stable",
    ).reset_index(drop=True)


def canonicalize_bars(
    frame: pd.DataFrame,
    *,
    timeframe: str | None = None,
    ticker: str | None = None,
    source: str = "alpaca",
    price_feed: str | None = None,
    adjustment: str = "all",
    ingested_at_utc: datetime | pd.Timestamp | None = None,
    availability_policy: AvailabilityPolicy = "market_interval_close",
    intraday_finalization_delay: pd.Timedelta = pd.Timedelta(seconds=30),
    daily_finalization_delay: pd.Timedelta = pd.Timedelta(minutes=15),
) -> pd.DataFrame:
    """Normalize left-edge provider bars into explicit interval and availability time."""

    required = {"open", "high", "low", "close", "volume"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"bar input is missing columns: {', '.join(missing)}")
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_BAR_COLUMNS)
    if availability_policy == "provider_publication_proxy":
        raise ValueError("provider_publication_proxy is not valid for market bars")

    data = frame.copy()
    data["ticker"] = _ticker_series(data, ticker)
    data["timeframe"] = _timeframe_series(data, timeframe)
    if data["timeframe"].nunique() != 1:
        raise DataReadinessError("canonicalize_bars requires one timeframe per call")
    normalized_timeframe = str(data["timeframe"].iloc[0])
    timestamp_column = "bar_start_utc" if "bar_start_utc" in data.columns else "timestamp"
    if timestamp_column not in data.columns and normalized_timeframe == "1d" and "date" in data.columns:
        timestamp_column = "date"
    if timestamp_column not in data.columns:
        raise SchemaMismatchError("bar input requires timestamp, date, or bar_start_utc")
    input_timestamp = _strict_utc_series(data[timestamp_column], allow_dates=normalized_timeframe == "1d")
    if bool(input_timestamp.isna().any()):
        raise DataReadinessError("bar input contains invalid or timezone-naive timestamps")

    if normalized_timeframe == "1d":
        bar_start, bar_end = _daily_market_intervals(input_timestamp)
        delay = daily_finalization_delay
    else:
        bar_start = input_timestamp
        bar_end = bar_start + BAR_DURATIONS[normalized_timeframe]
        delay = intraday_finalization_delay
    if delay < pd.Timedelta(0):
        raise ValueError("bar finalization delay cannot be negative")

    ingestion = _ingestion_series(data, ingested_at_utc)
    interval_available = bar_end + delay
    if availability_policy == "observed":
        available = _timestamp_max(interval_available, ingestion)
    else:
        available = interval_available

    numeric = data[["open", "high", "low", "close", "volume"]].apply(pd.to_numeric, errors="coerce")
    invalid = (
        numeric.isna().any(axis=1)
        | numeric[["open", "high", "low", "close"]].le(0).any(axis=1)
        | numeric["volume"].lt(0)
        | numeric["high"].lt(numeric[["open", "close", "low"]].max(axis=1))
        | numeric["low"].gt(numeric[["open", "close", "high"]].min(axis=1))
    )
    if bool(invalid.any()):
        raise DataReadinessError(f"bar input contains {int(invalid.sum())} invalid OHLCV rows")

    feed = _string_series(data, "price_feed", price_feed or "unknown").str.lower()
    if not set(feed).issubset({"sip", "iex", "unknown"}):
        raise DataReadinessError("bar price_feed must be sip, iex, or unknown")
    output = pd.DataFrame(
        {
            "ticker": data["ticker"],
            "timeframe": data["timeframe"],
            "bar_start_utc": bar_start,
            "bar_end_utc": bar_end,
            "available_at_utc": pd.to_datetime(available, utc=True),
            "ingested_at_utc": ingestion,
            **{column: numeric[column] for column in numeric.columns},
            "source": _string_series(data, "source", source).str.lower(),
            "price_feed": feed,
            "adjustment": _string_series(data, "adjustment", adjustment).str.lower(),
            "availability_policy": availability_policy,
            "schema_version": CANONICAL_SCHEMA_VERSION,
        }
    )
    if bool(output.duplicated(["ticker", "timeframe", "bar_start_utc"]).any()):
        raise DataReadinessError("canonical bars contain duplicate ticker/timeframe/bar_start rows")
    return output.loc[:, CANONICAL_BAR_COLUMNS].sort_values(["bar_start_utc", "ticker"]).reset_index(drop=True)


def canonicalize_events(
    frame: pd.DataFrame,
    *,
    collected_at_utc: datetime | pd.Timestamp | None = None,
    availability_policy: AvailabilityPolicy = "observed",
    sentiment_scored_at_utc: datetime | pd.Timestamp | None = None,
) -> pd.DataFrame:
    """Normalize provider events without pretending historical backfill was observed live."""

    if availability_policy == "market_interval_close":
        raise ValueError("market_interval_close is not valid for events")
    if frame.empty:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)
    collected_at = _strict_utc_timestamp(collected_at_utc) if collected_at_utc is not None else pd.NaT
    if collected_at_utc is not None and pd.isna(collected_at):
        raise ValueError("collected_at_utc must be timezone-aware")
    clean, _ = sanitize_events_frame(frame)
    if clean.empty:
        return pd.DataFrame(columns=CANONICAL_EVENT_COLUMNS)

    raw_records = clean["raw"].map(_raw_record)
    raw_created = pd.Series([record.get("created_at") for record in raw_records], index=clean.index)
    raw_updated = pd.Series([record.get("updated_at") for record in raw_records], index=clean.index)
    published_input = clean["published_at_utc"] if "published_at_utc" in clean.columns else clean["timestamp"]
    published = _coalesce_utc(raw_created, published_input)
    updated_input = clean["provider_updated_at_utc"] if "provider_updated_at_utc" in clean.columns else raw_updated
    provider_updated = _optional_utc_series(updated_input)
    invalid_updates = provider_updated.notna() & provider_updated.lt(published)
    if bool(invalid_updates.any()):
        raise DataReadinessError("event provider update timestamp precedes publication")
    first_seen_input: pd.Series | object
    if "first_seen_at_utc" in clean.columns:
        first_seen_input = clean["first_seen_at_utc"]
    elif "ingested_at_utc" in clean.columns:
        first_seen_input = clean["ingested_at_utc"]
    else:
        if pd.isna(collected_at):
            raise SchemaMismatchError("events require first_seen_at_utc, ingested_at_utc, or explicit collected_at_utc")
        first_seen_input = collected_at
    first_seen = _timestamp_series(first_seen_input, clean.index)
    if bool(published.isna().any() | first_seen.isna().any()):
        raise DataReadinessError("event input contains invalid or timezone-naive publication/first-seen timestamps")
    content_time = _timestamp_max(published, provider_updated)
    if availability_policy == "observed":
        available = _timestamp_max(content_time, first_seen)
    else:
        available = content_time

    sentiment = pd.to_numeric(clean.get("sentiment_numeric"), errors="coerce")
    has_sentiment = sentiment.notna()
    score_source: pd.Series | object | None = None
    if "sentiment_scored_at_utc" in clean.columns:
        score_source = clean["sentiment_scored_at_utc"]
    elif sentiment_scored_at_utc is not None:
        score_source = sentiment_scored_at_utc
    if bool(has_sentiment.any()) and score_source is None:
        raise DataReadinessError("sentiment values require sentiment_scored_at_utc")
    scored_at = _optional_timestamp_series(score_source, clean.index)
    if bool((has_sentiment & scored_at.isna()).any()):
        raise DataReadinessError("every sentiment value requires a valid sentiment_scored_at_utc")
    scored_at = scored_at.where(~has_sentiment, _timestamp_max(scored_at, available))
    feature_available = available.where(~has_sentiment, scored_at)

    raw_json = clean["raw"].fillna("").astype(str)
    raw_sha = raw_json.map(lambda value: hashlib.sha256(value.encode("utf-8")).hexdigest())
    source = clean["source"].astype(str).str.strip().str.lower()
    event_ids = [
        _event_id(
            ticker=str(row.ticker),
            source=str(row.source),
            published=pd.Timestamp(row.published),
            title=str(row.title),
            url=str(row.url),
            provider_id=_provider_event_id(record),
        )
        for row, record in zip(
            pd.DataFrame(
                {
                    "ticker": clean["ticker"],
                    "source": source,
                    "published": published,
                    "title": clean["title"],
                    "url": clean["url"],
                }
            ).itertuples(index=False),
            raw_records,
            strict=True,
        )
    ]
    output = pd.DataFrame(
        {
            "event_id": event_ids,
            "ticker": clean["ticker"].map(normalized_ticker),
            "source_family": source.map(source_family_for_source),
            "source": source,
            "published_at_utc": published,
            "provider_updated_at_utc": provider_updated,
            "first_seen_at_utc": first_seen,
            "available_at_utc": available,
            "sentiment_scored_at_utc": scored_at,
            "feature_available_at_utc": feature_available,
            "title": clean["title"].astype(str).str.strip(),
            "url": clean["url"].fillna("").astype(str),
            "summary": clean["summary"].fillna("").astype(str),
            "text": clean["text"].fillna("").astype(str),
            "sentiment_numeric": sentiment.clip(-1, 1),
            "relevance": pd.to_numeric(clean.get("relevance"), errors="coerce"),
            "availability_policy": availability_policy,
            "raw_sha256": raw_sha,
            "schema_version": CANONICAL_SCHEMA_VERSION,
        }
    )
    output = output.sort_values(["first_seen_at_utc", "event_id"]).drop_duplicates("event_id", keep="first")
    return output.loc[:, CANONICAL_EVENT_COLUMNS].sort_values(["feature_available_at_utc", "ticker"]).reset_index(drop=True)


def normalize_timeframe(value: str) -> str:
    key = value.strip().lower()
    try:
        return TIMEFRAME_ALIASES[key]
    except KeyError as exc:
        raise ValueError(f"unsupported timeframe: {value}") from exc


def _ticker_series(data: pd.DataFrame, supplied: str | None) -> pd.Series:
    if supplied is not None:
        values = pd.Series(supplied, index=data.index)
    elif "ticker" in data.columns:
        values = data["ticker"]
    elif "symbol" in data.columns:
        values = data["symbol"]
    else:
        raise SchemaMismatchError("bar input requires ticker/symbol or explicit ticker")
    try:
        return values.astype(str).map(normalized_ticker)
    except ValueError as exc:
        raise DataReadinessError(f"bar input contains invalid ticker: {exc}") from exc


def _timeframe_series(data: pd.DataFrame, supplied: str | None) -> pd.Series:
    if supplied is not None:
        values = pd.Series(supplied, index=data.index)
    elif "timeframe" in data.columns:
        values = data["timeframe"]
    else:
        raise SchemaMismatchError("bar input requires timeframe or explicit timeframe")
    try:
        return values.astype(str).map(normalize_timeframe)
    except ValueError as exc:
        raise DataReadinessError(str(exc)) from exc


def _daily_market_intervals(timestamp: pd.Series) -> tuple[pd.Series, pd.Series]:
    eastern_dates = timestamp.dt.tz_convert("America/New_York").dt.date
    calendar = xcals.get_calendar("XNYS")
    schedule = calendar.schedule.loc[str(min(eastern_dates)) : str(max(eastern_dates))]  # type: ignore[misc]
    intervals = {
        index.date(): (pd.Timestamp(row.open).tz_convert("UTC"), pd.Timestamp(row.close).tz_convert("UTC"))
        for index, row in schedule.iterrows()
    }
    missing = sorted(set(eastern_dates).difference(intervals))
    if missing:
        raise DataReadinessError(f"daily bars contain non-session dates: {missing[:3]}")
    starts = eastern_dates.map(lambda session: intervals[session][0])
    ends = eastern_dates.map(lambda session: intervals[session][1])
    return pd.to_datetime(starts, utc=True), pd.to_datetime(ends, utc=True)


def _ingestion_series(data: pd.DataFrame, supplied: datetime | pd.Timestamp | None) -> pd.Series:
    if "ingested_at_utc" in data.columns:
        result = _optional_utc_series(data["ingested_at_utc"])
    elif supplied is not None:
        result = _timestamp_series(supplied, data.index)
    else:
        raise SchemaMismatchError("canonical bars require ingested_at_utc")
    if bool(result.isna().any()):
        raise DataReadinessError("bar ingestion timestamps are invalid or timezone-naive")
    return result


def _string_series(data: pd.DataFrame, column: str, default: str) -> pd.Series:
    values = data[column] if column in data.columns else pd.Series(default, index=data.index)
    values = values.fillna(default).astype(str).str.strip()
    if bool(values.eq("").any()):
        raise DataReadinessError(f"{column} contains empty values")
    return values


def _strict_utc_series(values: pd.Series, *, allow_dates: bool = False) -> pd.Series:
    if allow_dates and not isinstance(values.dtype, pd.DatetimeTZDtype):
        parsed = pd.to_datetime(values, utc=True, errors="coerce")
        return cast(pd.Series, parsed)
    return values.map(_strict_utc_timestamp)


def _optional_utc_series(values: pd.Series | object) -> pd.Series:
    if isinstance(values, pd.Series):
        return pd.to_datetime(values.map(_optional_utc_timestamp), utc=True)
    raise TypeError("timestamp values must be a Series")


def _timestamp_series(values: pd.Series | object, index: pd.Index) -> pd.Series:
    if isinstance(values, pd.Series):
        return values.map(_strict_utc_timestamp)
    timestamp = _strict_utc_timestamp(values)
    return pd.Series(timestamp, index=index, dtype="datetime64[ns, UTC]")


def _optional_timestamp_series(values: pd.Series | object | None, index: pd.Index) -> pd.Series:
    if values is None:
        return pd.Series(pd.NaT, index=index, dtype="datetime64[ns, UTC]")
    if isinstance(values, pd.Series):
        return pd.to_datetime(values.map(_optional_utc_timestamp), utc=True)
    timestamp = _optional_utc_timestamp(values)
    return pd.Series(timestamp, index=index, dtype="datetime64[ns, UTC]")


def _coalesce_utc(primary: pd.Series, fallback: pd.Series) -> pd.Series:
    parsed_primary = primary.map(_optional_utc_timestamp)
    parsed_fallback = fallback.map(_strict_utc_timestamp)
    return parsed_primary.where(parsed_primary.notna(), parsed_fallback)


def _timestamp_max(left: pd.Series, right: pd.Series) -> pd.Series:
    normalized_left = pd.to_datetime(left, utc=True)
    normalized_right = pd.to_datetime(right, utc=True)
    use_right = normalized_right.notna() & (normalized_left.isna() | normalized_right.gt(normalized_left))
    return normalized_left.where(~use_right, normalized_right)


def _strict_utc_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")


def _optional_utc_timestamp(value: object) -> pd.Timestamp:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return pd.NaT
    return _strict_utc_timestamp(value)


def _raw_record(value: object) -> dict[str, Any]:
    if isinstance(value, dict):
        return {str(key): item for key, item in value.items()}
    if not isinstance(value, str) or not value.strip():
        return {}
    try:
        parsed = json.loads(value)
    except json.JSONDecodeError:
        return {}
    return {str(key): item for key, item in parsed.items()} if isinstance(parsed, dict) else {}


def _provider_event_id(record: dict[str, Any]) -> str:
    for key in ("id", "news_id", "accession_number"):
        value = str(record.get(key, "")).strip()
        if value:
            return value
    return ""


def _event_id(
    *,
    ticker: str,
    source: str,
    published: pd.Timestamp,
    title: str,
    url: str,
    provider_id: str,
) -> str:
    identity = provider_id or "|".join((published.isoformat(), title.strip(), url.strip()))
    payload = "|".join((ticker.strip().upper(), source.strip().lower(), identity))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()
