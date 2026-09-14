"""Pure, bounded legacy-news conversions backed by four pinned identity intervals.

Callers verify artifact pins and keep corrected-source generations separate. Build
against complete target authorities (never a decision cohort) before removing
corrected-symbol conversions. An evidence hash binds inputs, but is not identity proof.
"""
from __future__ import annotations

import re
from bisect import bisect_right
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, replace
from datetime import datetime, timedelta
from typing import Any

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import normalized_ticker
from market_predictor.evidence.hashing import json_sha256

MAXIMUM_ROWS = 250_000
MAXIMUM_FRAME_BYTES = 128 * 1024**2
BRIDGE_COLUMNS = (
    "source_security_id", "ticker", "target_security_id", "effective_from_utc",
    "effective_to_utc", "available_at_utc", "bridge_row_sha256",
)
_AUTHORITY_COLUMNS = (
    "security_id", "ticker", "effective_from_utc", "effective_to_utc", "available_at_utc",
)
_SHA256 = re.compile(r"[0-9a-f]{64}")
_CIK = re.compile(r"cik:([0-9]{10})(?::ticker:([A-Z0-9.-]{1,16}))?")
_SCHEMA = "market_predictor.issuer_news_identity.v1"


@dataclass(frozen=True)
class _Span:
    security_id: str
    ticker: str
    start: pd.Timestamp
    end: pd.Timestamp | None
    available: pd.Timestamp
    cik: str | None = None
    source_id: str = ""
    digest: str = ""


_Index = dict[tuple[str, ...], tuple[list[pd.Timestamp], list[_Span]]]


def _frame(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    if not isinstance(frame, pd.DataFrame) or not frame.columns.is_unique:
        raise DataReadinessError("identity input requires a frame with unique columns")
    _bounded(frame)
    if missing := set(columns).difference(frame.columns):
        raise DataReadinessError(f"identity input missing columns: {sorted(missing)}")
    return frame.copy()


def _bounded(frame: pd.DataFrame) -> None:
    if len(frame) > MAXIMUM_ROWS or int(frame.memory_usage(deep=True).sum()) > MAXIMUM_FRAME_BYTES:
        raise DataReadinessError("identity transformation exceeds bounded frame size")


def _text(value: object, name: str) -> str:
    if (not isinstance(value, str) or not value or not value.isascii() or len(value) > 512
            or any(char.isspace() or not char.isprintable() for char in value)):
        raise DataReadinessError(f"{name} requires canonical strings")
    if name.endswith("ticker"):
        try:
            if normalized_ticker(value) != value:
                raise ValueError("noncanonical ticker")
        except ValueError as exc:
            raise DataReadinessError(f"{name} requires canonical ticker strings") from exc
    return value


def _sha(value: object) -> str:
    if not isinstance(value, str) or _SHA256.fullmatch(value) is None:
        raise DataReadinessError("identity evidence requires canonical SHA256")
    return value


def _utc(value: object, name: str, *, nullable: bool = False) -> pd.Timestamp | None:
    if value is None or value is pd.NaT or value is pd.NA:
        if nullable:
            return None
        raise DataReadinessError(f"{name} requires a finite aware UTC timestamp")
    if not isinstance(value, (str, datetime, pd.Timestamp)):
        raise DataReadinessError(f"{name} requires a finite aware UTC timestamp")
    try:
        stamp = pd.Timestamp(value)
        if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
            raise ValueError("not aware UTC")
        stamp = stamp.tz_convert("UTC")
        _ = stamp.value  # Reject dates outside the nanosecond frame representation.
    except (TypeError, ValueError, OverflowError) as exc:
        raise DataReadinessError(f"{name} requires a finite aware UTC timestamp") from exc
    return stamp


def _required_utc(value: object, name: str) -> pd.Timestamp:
    stamp = _utc(value, name)
    assert stamp is not None
    return stamp


def _cik(security_id: str, ticker: str) -> str | None:
    if not security_id.startswith("cik:"):
        return None
    match = _CIK.fullmatch(security_id)
    if match is None or int(match[1]) == 0 or match[2] not in (None, ticker):
        raise DataReadinessError("explicit CIK identity must match its exact ticker")
    return str(match[1])


def _authority(frame: pd.DataFrame, *, sec: bool = False) -> list[_Span]:
    columns = (*_AUTHORITY_COLUMNS, "sec_cik") if sec else _AUTHORITY_COLUMNS
    checked = _frame(frame, columns)
    spans: list[_Span] = []
    for row in checked.loc[:, columns].to_dict("records"):
        security, ticker = _text(row["security_id"], "security_id"), _text(row["ticker"], "ticker")
        cik = _cik(security, ticker)
        if sec:
            proof = _text(row["sec_cik"], "sec_cik")
            if re.fullmatch(r"[0-9]{10}", proof) is None or int(proof) == 0 or cik not in (None, proof):
                raise DataReadinessError("target SEC CIK contradicts identity or is not canonical")
            cik = proof
        start = _required_utc(row["effective_from_utc"], "effective_from_utc")
        end = _utc(row["effective_to_utc"], "effective_to_utc", nullable=True)
        if end is not None and end <= start:
            raise DataReadinessError("identity interval must be nonempty and half-open")
        spans.append(_Span(security, ticker, start, end,
            _required_utc(row["available_at_utc"], "available_at_utc"), cik))
    return spans


def _index(spans: list[_Span], key: Callable[[_Span], tuple[str, ...]]) -> _Index:
    groups: dict[tuple[str, ...], list[_Span]] = {}
    for span in spans:
        groups.setdefault(key(span), []).append(span)
    result: _Index = {}
    for identity, rows in groups.items():
        rows.sort(key=lambda row: row.start)
        for previous, current in zip(rows, rows[1:], strict=False):
            if previous.end is None or current.start < previous.end:
                raise DataReadinessError(f"duplicate or ambiguous overlapping identity intervals: {identity}")
        result[identity] = ([row.start for row in rows], rows)
    return result


def _overlaps(index: _Index, key: tuple[str, ...], start: pd.Timestamp,
              end: pd.Timestamp | None) -> Iterator[_Span]:
    entry = index.get(key)
    if entry is None:
        return
    starts, spans = entry
    position = max(0, bisect_right(starts, start) - 1)
    while position < len(spans):
        span = spans[position]
        if end is not None and span.start >= end:
            break
        if span.end is None or span.end > start:
            yield span
        position += 1


def _intersection(left: _Span, right: _Span) -> _Span:
    ends = [end for end in (left.end, right.end) if end is not None]
    return replace(left, start=max(left.start, right.start), end=min(ends) if ends else None,
        available=max(left.available, right.available))


def _append(rows: list[Any], row: Any) -> None:
    if len(rows) >= MAXIMUM_ROWS:
        raise DataReadinessError("identity output exceeds bounded row count")
    rows.append(row)


def build_issuer_news_bridge(
    source_registry: pd.DataFrame,
    source_memberships: pd.DataFrame,
    target_memberships: pd.DataFrame,
    target_sec_identities: pd.DataFrame,
    *,
    evidence_sha256: str,
) -> pd.DataFrame:
    """Convert only explicit source CIK + exact-ticker claims, never guess aliases.

    Source memberships must be the verified legacy authority, with exclusions
    already applied. Target authorities must be global, not cohort-filtered.
    """
    evidence = _sha(evidence_sha256)
    registry = _authority(source_registry)
    legacy = _authority(source_memberships)
    members = _authority(target_memberships)
    sec = _authority(target_sec_identities, sec=True)

    def pair(row: _Span) -> tuple[str, str]:
        return row.security_id, row.ticker

    _index(registry, pair)
    legacy_index = _index(legacy, pair)
    member_index = _index(members, pair)
    for authority in (members, sec):
        # Validate global claims before excluding non-conversions or missing sources.
        _index(authority, lambda row: (row.ticker,))
        _index(authority, lambda row: (row.security_id,))
    targets: list[_Span] = []
    for identity in sec:
        for member in _overlaps(member_index, pair(identity), identity.start, identity.end):
            _append(targets, _intersection(identity, member))
    target_index = _index(targets, lambda row: (str(row.cik), row.ticker))
    rows: list[dict[str, Any]] = []
    for source in registry:
        if source.cik is None:
            continue
        for membership in _overlaps(legacy_index, pair(source), source.start, source.end):
            verified = _intersection(source, membership)
            for target in _overlaps(target_index, (source.cik, source.ticker), verified.start, verified.end):
                if target.security_id == source.security_id:
                    continue
                proof = _intersection(verified, target)
                row = {
                    "source_security_id": source.security_id, "ticker": source.ticker,
                    "target_security_id": target.security_id, "effective_from_utc": proof.start,
                    "effective_to_utc": proof.end, "available_at_utc": proof.available,
                }
                row["bridge_row_sha256"] = _hash("bridge", evidence, row)
                _append(rows, row)
    result = pd.DataFrame(rows, columns=BRIDGE_COLUMNS)
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True).dt.as_unit("ns")
    _bounded(result)
    return result.sort_values(list(BRIDGE_COLUMNS[:4]), kind="stable").reset_index(drop=True)


def _hash(kind: str, original: str, values: dict[str, Any]) -> str:
    return json_sha256({"schema": _SCHEMA, "kind": kind, "original": original,
        **{key: value.isoformat() if isinstance(value, pd.Timestamp) else value for key, value in values.items()}})


def _bridge_index(bridge: pd.DataFrame) -> _Index:
    checked = _frame(bridge, BRIDGE_COLUMNS)
    authority = checked.rename(columns={"target_security_id": "security_id"})
    spans = _authority(authority)
    rows: list[_Span] = []
    for span, row in zip(spans, checked.to_dict("records"), strict=True):
        source_id = _text(row["source_security_id"], "source_security_id")
        if _cik(source_id, span.ticker) is None or source_id == span.security_id:
            raise DataReadinessError("bridge requires an explicit CIK source and a conversion")
        source_cik = _cik(source_id, span.ticker)
        if span.cik not in (None, source_cik):
            raise DataReadinessError("bridge target contradicts source CIK")
        rows.append(replace(span, source_id=source_id, digest=_sha(row["bridge_row_sha256"])))
    return _index(rows, lambda row: (row.source_id, row.ticker))


def _provenance(frame: pd.DataFrame, columns: Sequence[str]) -> pd.DataFrame:
    added = [*(f"identity_original_{column}" for column in columns),
        "identity_translation_status", "identity_bridge_row_sha256"]
    if set(added).intersection(frame.columns):
        raise DataReadinessError("identity transformation requires untranslated input")
    for column in columns:
        frame[f"identity_original_{column}"] = frame[column]
    frame["identity_translation_status"] = "unmapped"
    frame["identity_bridge_row_sha256"] = ""
    return frame


def map_news_relations(relations: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Retain every relation and its source evidence; translate targets at event time."""
    index = _bridge_index(bridge)
    columns = ("relation_id", "target_security_id", "target_ticker", "event_feature_available_at_utc",
        "identity_available_at_utc", "feature_available_at_utc")
    output = _frame(relations, columns)
    output = _provenance(output, ("relation_id", "target_security_id", "target_ticker",
        "identity_available_at_utc", "feature_available_at_utc"))
    # An all-null stored column can be timezone-naive, and strings are valid input
    # clocks. Validate before conversion so normalization cannot legitimize naive dates.
    validated_clocks = {}
    clocks_normalized = False
    for column, nullable in (("identity_available_at_utc", True), ("feature_available_at_utc", False)):
        values = [_utc(value, column, nullable=nullable) for value in output[column]]
        validated_clocks[column] = pd.Series(values, index=output.index, dtype="datetime64[ns, UTC]")
    if output.relation_id.duplicated().any():
        raise DataReadinessError("relation IDs must be unique")
    for position, row in enumerate(output.to_dict("records")):
        original = _text(row["relation_id"], "relation_id")
        key = (_text(row["target_security_id"], "target_security_id"), _text(row["target_ticker"], "target_ticker"))
        event = _required_utc(row["event_feature_available_at_utc"], "event_feature_available_at_utc")
        feature = _required_utc(row["feature_available_at_utc"], "feature_available_at_utc")
        identity = _utc(row["identity_available_at_utc"], "identity_available_at_utc", nullable=True)
        if feature < event or (identity is not None and feature < identity):
            raise DataReadinessError("relation feature availability precedes its dependencies")
        for span in _overlaps(index, key, event, None):
            if span.start > event:
                break
            if span.available > event or (identity is not None and identity > event):
                continue
            if not clocks_normalized:
                for column, normalized_values in validated_clocks.items():
                    output[column] = normalized_values
                clocks_normalized = True
            updates = {
                "target_security_id": span.security_id,
                "relation_id": _hash("relation", original, {"bridge_row_sha256": span.digest}),
                "identity_translation_status": "mapped", "identity_bridge_row_sha256": span.digest,
                "identity_available_at_utc": max(identity, span.available) if identity is not None else span.available,
                "feature_available_at_utc": max(feature, span.available),
            }
            for column, value in updates.items():
                output.iat[position, output.columns.get_loc(column)] = value
    if output.relation_id.duplicated().any():
        raise DataReadinessError("translated relation IDs collide")
    _bounded(output)
    return output


def map_news_coverage(ledger: pd.DataFrame, bridge: pd.DataFrame) -> pd.DataFrame:
    """Partition requested coverage without inventing coverage, dates or availability.

    Availability clips the start of mapped segments; earlier and unproved segments
    keep the original security ID and status. Input chunk IDs must be unique.
    """
    index = _bridge_index(bridge)
    columns = ("chunk_id", "security_id", "ticker", "requested_start_utc", "requested_end_utc")
    checked = _provenance(_frame(ledger, columns), columns)
    if checked.chunk_id.duplicated().any():
        raise DataReadinessError("coverage chunk IDs must be unique")
    if "identity_available_at_utc" in checked:
        raise DataReadinessError("coverage requires untranslated input")
    checked["identity_available_at_utc"] = pd.Series(pd.NaT, index=checked.index, dtype="datetime64[ns, UTC]")
    rows: list[dict[str, Any]] = []
    for row in checked.to_dict("records"):
        original = _text(row["chunk_id"], "chunk_id")
        key = (_text(row["security_id"], "security_id"), _text(row["ticker"], "ticker"))
        start = _required_utc(row["requested_start_utc"], "requested_start_utc")
        end = _required_utc(row["requested_end_utc"], "requested_end_utc")
        if end <= start:
            raise DataReadinessError("requested coverage must be a finite nonempty half-open interval")
        cursor = start
        segments: list[tuple[pd.Timestamp, pd.Timestamp, _Span | None]] = []
        for span in _overlaps(index, key, start, end):
            lower, upper = max(start, span.start, span.available), min(end, span.end) if span.end is not None else end
            if lower >= upper:
                continue
            if cursor < lower:
                segments.append((cursor, lower, None))
            segments.append((lower, upper, span))
            cursor = upper
        if cursor < end:
            segments.append((cursor, end, None))
        for lower, upper, proof in segments:
            segment = {**row, "requested_start_utc": lower, "requested_end_utc": upper}
            if proof is not None:
                segment.update(security_id=proof.security_id, identity_translation_status="mapped",
                    identity_bridge_row_sha256=proof.digest, identity_available_at_utc=proof.available)
            if proof is not None or len(segments) > 1:
                segment["chunk_id"] = _hash("coverage", original, {
                    "start": lower, "end": upper, "bridge_row_sha256": proof.digest if proof is not None else None,
                })
            _append(rows, segment)
    result = pd.DataFrame(rows, columns=checked.columns)
    for column in ("requested_start_utc", "requested_end_utc", "identity_available_at_utc"):
        result[column] = pd.to_datetime(result[column], utc=True)
    if result.chunk_id.duplicated().any():
        raise DataReadinessError("translated coverage chunk IDs collide")
    _bounded(result)
    return result
