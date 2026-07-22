from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from typing import Any

import pandas as pd

EVENT_COLUMNS = [
    "ticker",
    "timestamp",
    "source",
    "title",
    "url",
    "summary",
    "text",
    "engagement_score",
    "engagement_comments",
    "engagement_upvote_ratio",
    "raw",
]


@dataclass(frozen=True)
class VerificationReport:
    rows_in: int
    rows_out: int
    duplicate_rows_removed: int
    missing_required_rows_removed: int
    future_timestamp_rows: int
    sources: dict[str, int]

    def to_record(self) -> dict[str, Any]:
        return asdict(self)


def sanitize_events_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, VerificationReport]:
    rows_in = len(frame)
    if frame.empty:
        empty = pd.DataFrame(columns=EVENT_COLUMNS)
        return empty, VerificationReport(0, 0, 0, 0, 0, {})

    clean = frame.copy()
    original_columns = list(clean.columns)
    for col in EVENT_COLUMNS:
        if col not in clean.columns:
            clean[col] = None
    extra_columns = [col for col in original_columns if col not in EVENT_COLUMNS]
    clean = clean[EVENT_COLUMNS + extra_columns]

    clean["ticker"] = clean["ticker"].astype(str).str.upper().str.strip()
    clean["source"] = clean["source"].fillna("unknown").astype(str).str.strip()
    for col in ["title", "url", "summary", "text"]:
        clean[col] = clean[col].fillna("").astype(str)
    clean["text"] = clean["text"].where(clean["text"].str.len() > 0, clean["summary"])
    clean["text"] = clean["text"].where(clean["text"].str.len() > 0, clean["title"])
    clean["timestamp"] = pd.to_datetime(clean["timestamp"], utc=True, errors="coerce")

    for col in ["engagement_score", "engagement_comments", "engagement_upvote_ratio"]:
        clean[col] = pd.to_numeric(clean[col], errors="coerce").fillna(0.0)

    clean["raw"] = clean["raw"].map(_safe_json)
    for col in extra_columns:
        if clean[col].dtype == "object":
            clean[col] = clean[col].map(_safe_extra_value)

    required = clean["ticker"].ne("") & clean["timestamp"].notna() & clean["title"].ne("")
    missing_required = int((~required).sum())
    clean = clean[required].copy()

    before_dedupe = len(clean)
    clean = clean.drop_duplicates(subset=["ticker", "timestamp", "source", "title", "url"], keep="first")
    duplicates_removed = before_dedupe - len(clean)

    now = pd.Timestamp.now(tz="UTC") + pd.Timedelta(minutes=5)
    future_rows = int((clean["timestamp"] > now).sum())
    clean = clean[clean["timestamp"] <= now].copy()

    clean = clean.sort_values(["ticker", "timestamp", "source", "title"]).reset_index(drop=True)
    sources = {str(key): int(value) for key, value in clean["source"].value_counts().sort_index().items()}
    report = VerificationReport(
        rows_in=rows_in,
        rows_out=len(clean),
        duplicate_rows_removed=int(duplicates_removed),
        missing_required_rows_removed=missing_required,
        future_timestamp_rows=future_rows,
        sources=sources,
    )
    return clean, report


def _safe_json(value: Any) -> str:
    if value in (None, ""):
        return ""
    try:
        return json.dumps(value, ensure_ascii=True, sort_keys=True, default=str)
    except TypeError:
        return json.dumps(str(value), ensure_ascii=True)


def _safe_extra_value(value: Any) -> Any:
    if value is None:
        return None
    if isinstance(value, str | int | float | bool):
        return value
    if isinstance(value, dict | list | tuple | set):
        return _safe_json(value)
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    if hasattr(value, "isoformat"):
        return value.isoformat()
    return _safe_json(value)
