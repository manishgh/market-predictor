"""Bounded original Alpaca field inventory, never content or source admission."""
from __future__ import annotations

import hashlib
import io
import json
import re
from collections.abc import Callable, Iterator
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.normalize import _event_id, _provider_event_id
from market_predictor.canonical.store import CANONICAL_MANIFEST_SCHEMA, manifest_path_for
from market_predictor.catalysts.issuer_events.news_history_contracts import (
    NEWS_HISTORY_REQUEST_SCHEMA,
    NEWS_PAGE_SCHEMA,
)
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.core.symbols import canonical_symbol, normalized_ticker
from market_predictor.data_quality import _safe_json
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside

MAX_FILE_BYTES = 64 * 1024 * 1024
MAX_RAW_PAGE_BYTES = 64 * 1024 * 1024
MAX_PARQUET_UNCOMPRESSED_BYTES = 128 * 1024 * 1024
MAX_DECODED_BATCH_BYTES = 8 * 1024 * 1024
MAX_DECODED_EVENT_BYTES = 128 * 1024 * 1024
EVENT_BATCH_ROWS = 1
MAX_EVENT_ROWS = 25_000
MAX_EMPTY_CHUNK_PAGES = 16
PROXY_POLICY = "provider_publication_proxy"
_CLOCKS = ("published_at_utc", "provider_updated_at_utc", "first_seen_at_utc", "available_at_utc")
_STRINGS = (
    "event_id", "chunk_id", "query_security_id", "query_ticker", "source_family", "source", "raw_sha256",
    "source_version_sha256", "provider_story_id", "availability_policy", "inventory_status", "content_category",
    "body_state", "summary_state", "chosen_field", "chosen_field_sha256", "raw_record_locators_json",
    "attribution_status",
)
_BOOLS = ("body_equals_title", "production_eligible", "training_eligible", "serving_eligible")
_STATUSES = ("included", "version_after_cutoff")
_CATEGORIES = ("provider_body_field", "provider_summary_field", "headline_only")
_DISCARDS = ("clock", "window", "symbol", "title")
_EVENT_COLUMNS = ("event_id", "security_id", "ticker", "source_family", "source", "raw_sha256",
                  "title", "url", "summary", "text", "availability_policy", *_CLOCKS)
MemoryCheck = Callable[[], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _bytes(path: Path) -> bytes:
    _require(path.stat().st_size <= MAX_FILE_BYTES, "content source exceeds 64 MiB")
    with path.open("rb") as handle:
        raw = handle.read(MAX_FILE_BYTES + 1)
    _require(len(raw) <= MAX_FILE_BYTES, "content source exceeds 64 MiB")
    return raw


def _read(path: Path, pin: str) -> bytes:
    _require(re.fullmatch(r"[0-9a-f]{64}", pin) is not None, "invalid source SHA256")
    raw = _bytes(path)
    _require(hashlib.sha256(raw).hexdigest() == pin, f"content source hash mismatch: {path}")
    return raw


def _clock(value: Any, label: str) -> pd.Timestamp:
    try:
        _require(isinstance(value, (str, datetime)), f"{label} requires an aware timestamp")
        stamp = pd.Timestamp(value)
        _require(not pd.isna(stamp) and stamp.tzinfo is not None, f"{label} requires an aware timestamp")
        return stamp.tz_convert("UTC").as_unit("ns", round_ok=False)
    except (ValueError, OverflowError) as exc:
        raise DataReadinessError(f"{label} requires exact UTC nanoseconds") from exc


def _optional_clock(value: Any, label: str) -> Any:
    return pd.NaT if value is None or value is pd.NaT else _clock(value, label)


def _field(item: dict[str, Any], name: str) -> tuple[str, str]:
    if name not in item:
        return "absent", ""
    value = item[name]
    if value is None:
        return "null", ""
    if not isinstance(value, str):
        return "non_text", ""
    return ("empty" if value == "" else "blank" if not value.strip() else "nonempty"), value


def _relative(root: Path, path: Path) -> str:
    return path.relative_to(root).as_posix()


def _frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    result = pd.DataFrame(rows, columns=[*_STRINGS, *_CLOCKS, *_BOOLS])
    for column in _STRINGS:
        result[column] = result[column].astype("string")
    for column in _CLOCKS:
        result[column] = pd.array(result[column], dtype="datetime64[ns, UTC]")
    for column in _BOOLS:
        result[column] = result[column].astype(bool)
    return result.sort_values(["event_id", "raw_sha256"], kind="stable").reset_index(drop=True)


def _event_batches(parquet: pq.ParquetFile, columns: list[str], memory_check: MemoryCheck) -> Iterator[list[dict[str, Any]]]:
    batches = parquet.iter_batches(  # type: ignore[no-untyped-call]
        batch_size=EVENT_BATCH_ROWS, columns=columns, use_threads=False)
    decoded = 0
    while True:
        memory_check()
        batch = next(batches, None)
        if batch is None:
            return
        decoded += batch.nbytes
        _require(batch.nbytes <= MAX_DECODED_BATCH_BYTES, "decoded Parquet batch exceeds byte limit")
        _require(decoded <= MAX_DECODED_EVENT_BYTES, "decoded Parquet cumulative bytes exceed limit")
        memory_check()
        yield batch.to_pylist()
        del batch


def _identity(chunk: Any, request: Any) -> tuple[str, str]:
    _require(isinstance(chunk, str) and re.fullmatch(r"[0-9a-f]{24}", chunk) is not None,
             "original sidecar lacks chunk identity")
    _require(isinstance(request, str) and re.fullmatch(r"[0-9a-f]{64}", request) is not None,
             "original sidecar lacks collection request identity")
    return chunk, request


def _work_unit(root: Path, collection: Path, chunk: str, request: str, memory_check: MemoryCheck) -> dict[str, Any]:
    """Bind the chunk to the one request work unit whose query produced it."""
    memory_check()
    path = collection / "_request.json"
    raw = _bytes(path)
    payload = parse_strict_json_object(raw, label="original collection request")
    body = {key: value for key, value in payload.items() if key != "request_sha256"}
    _require(payload.get("request_sha256") == request == json_sha256(body), "original collection request hash mismatch")
    # Headline-only rows mean provider-empty fields only when the query asked for article content.
    _require(payload.get("schema") == NEWS_HISTORY_REQUEST_SCHEMA and payload.get("include_content") is True
             and payload.get("availability_policy") == PROXY_POLICY, "original collection request semantics unsupported")
    units = payload.get("work_units")
    if not isinstance(units, list):
        raise DataReadinessError("original collection request lacks work units")
    found = [unit for unit in units if isinstance(unit, dict) and unit.get("chunk_id") == chunk]
    _require(len(found) == 1, "chunk is not exactly one request work unit")
    unit = found[0]
    names = ("security_id", "ticker", "provider_symbol", "start_utc", "end_exclusive_utc")
    _require(all(isinstance(unit.get(name), str) and unit[name] for name in names), "malformed request work unit")
    identity = "|".join(unit[name] for name in ("security_id", "ticker", "start_utc", "end_exclusive_utc"))
    _require(hashlib.sha256(identity.encode("utf-8")).hexdigest()[:24] == chunk,
             "chunk identity does not derive from its request work unit")
    start, end = _clock(unit["start_utc"], "work-unit start"), _clock(unit["end_exclusive_utc"], "work-unit end")
    _require(start < end, "request work unit has an empty window")
    return {"chunk_id": chunk, "collection_request_sha256": request, "security_id": unit["security_id"],
            "ticker": unit["ticker"], "provider_symbol": unit["provider_symbol"], "start": start, "end": end,
            "request_path": _relative(root, path), "request_file_sha256": hashlib.sha256(raw).hexdigest()}


def _admission(item: dict[str, Any], unit: dict[str, Any]) -> tuple[str, pd.Timestamp, Any] | str:
    """Mirror the original producer's acceptance; a string names why it discarded the item."""
    try:
        created = pd.to_datetime(item.get("created_at"), utc=True, errors="coerce")
        updated = pd.to_datetime(item.get("updated_at"), utc=True, errors="coerce")
        if pd.isna(created) or (pd.notna(updated) and updated < created):
            return "clock"
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("provider item is not processable by its original producer") from exc
    created = pd.Timestamp(created)
    # Compare before unit conversion: a clock beyond nanoseconds lies outside every query window.
    if not unit["start"] <= created < unit["end"]:
        return "window"
    symbols = item.get("symbols")
    if unit["ticker"] not in ({canonical_symbol(str(symbol)) for symbol in symbols} if isinstance(symbols, list) else set()):
        return "symbol"
    title = str(item.get("headline") or "").strip()
    if not title:
        return "title"
    story = str(item.get("id") or "").strip() or hashlib.sha256(
        "|".join((created.isoformat(), title, str(item.get("url") or ""))).encode("utf-8")).hexdigest()
    try:
        return story, created.as_unit("ns"), pd.Timestamp(updated).as_unit("ns") if pd.notna(updated) else pd.NaT
    except ValueError as exc:
        raise DataReadinessError("admitted provider update is outside exact UTC nanoseconds") from exc


def _pages(root: Path, collection: Path, inputs: dict[str, Any], unit: dict[str, Any], wanted: set[str],
           memory_check: MemoryCheck) -> tuple[dict[str, Any], dict[str, tuple[Any, str, pd.Timestamp]], dict[str, Any]]:
    chunk, request = unit["chunk_id"], unit["collection_request_sha256"]
    declared: list[tuple[Path, str]] = []
    for key, pin in inputs.items():
        if key in ("chunk_id", "collection_request_sha256"):
            continue
        _require(isinstance(pin, str), "invalid page pin")
        path = inside(root, key)
        _require(path.parent == collection / "raw_pages" / chunk
                 and re.fullmatch(r"page_[0-9]{6}\.json", path.name) is not None, "invalid declared raw-page path")
        declared.append((path, pin))
    declared.sort()
    _require(bool(declared) and len({p for p, _ in declared}) == len(declared), "missing or duplicate declared pages")
    _require(sum(path.stat().st_size for path, _ in declared) <= MAX_RAW_PAGE_BYTES,
             "combined raw pages exceed 64 MiB")
    matches: dict[str, Any] = {}
    files: dict[str, str] = {}
    # The producer keeps each story's latest revision; equal revision clocks keep the later occurrence.
    retained: dict[str, tuple[Any, str, pd.Timestamp]] = {}
    versions: set[tuple[str, str]] = set()
    discarded = dict.fromkeys(_DISCARDS, 0)
    token: str | None = None
    tokens: set[str] = set()
    records = admitted = consumed_bytes = 0
    for index, (path, pin) in enumerate(declared):
        memory_check()
        raw = _read(path, pin)
        consumed_bytes += len(raw)
        _require(consumed_bytes <= MAX_RAW_PAGE_BYTES, "combined raw pages exceed 64 MiB")
        page: dict[str, Any] = parse_strict_json_object(raw, label=str(path))
        relative = _relative(root, path)
        files[relative] = pin
        _require(path.name == f"page_{index:06d}.json" and type(page.get("page_index")) is int
                 and page["page_index"] == index, "page sequence mismatch")
        _require(page.get("schema") == NEWS_PAGE_SCHEMA and page.get("chunk_id") == chunk
                 and page.get("collection_request_sha256") == request, "page schema/request/chunk mismatch")
        _require(page.get("content_sha256") == json_sha256({k: v for k, v in page.items() if k != "content_sha256"}),
                 "page envelope hash mismatch")
        _require("request_page_token" in page and page["request_page_token"] == token, "pagination request mismatch")
        _require("next_page_token" in page, "missing pagination terminal evidence")
        next_token = page["next_page_token"]
        _require(next_token is None or isinstance(next_token, str) and bool(next_token), "invalid page token")
        _require(next_token is None or next_token not in tokens, "pagination token cycle")
        if isinstance(next_token, str):
            tokens.add(next_token)
        _require((next_token is None) == (index == len(declared) - 1), "nonterminal or early-terminal page")
        token = next_token if isinstance(next_token, str) else None
        seen = _clock(page.get("collected_at_utc"), "page collection")
        news = page.get("news")
        if not isinstance(news, list):
            raise DataReadinessError("page news must be an array")
        for position, item in enumerate(news):
            _require(isinstance(item, dict), "provider article must be an object")
            raw_hash = hashlib.sha256(_safe_json(item).encode("utf-8")).hexdigest()
            records += 1
            admission = _admission(item, unit)
            if isinstance(admission, str):
                discarded[admission] += 1
            else:
                story, created, updated = admission
                revision = created if pd.isna(updated) else updated
                admitted += 1
                versions.add((story, raw_hash))
                if story not in retained or revision >= retained[story][0]:
                    retained[story] = (revision, raw_hash, seen)
            if raw_hash in wanted:
                match = matches.setdefault(raw_hash, {"item": item, "admission": admission, "locators": []})
                _require(match["item"] == item, "conflicting article hash match")
                match["locators"].append({"page_path": relative, "page_sha256": pin, "news_index": position})
    return matches, retained, {"source_files": files, "query_chunk_scope": "whole_query_window_not_inventory_window",
        "query_chunk_pages": len(declared), "query_chunk_provider_records": records, "query_chunk_discarded_records": discarded,
        "query_chunk_admitted_records": admitted, "query_chunk_admitted_stories": len(retained),
        "query_chunk_admitted_versions": len(versions),
        "query_chunk_duplicate_version_occurrences": admitted - len(versions),
        "query_chunk_superseded_versions": len(versions) - len(retained)}


def _record(row: dict[str, Any], match: dict[str, Any], unit: dict[str, Any],
            retained: dict[str, tuple[Any, str, pd.Timestamp]]) -> dict[str, Any]:
    item, admission = match["item"], match["admission"]
    if isinstance(admission, str):
        raise DataReadinessError("canonical row maps to a provider item its producer discarded")
    story, published, updated = admission
    _require(published == row["published_at_utc"] and (pd.isna(updated) and pd.isna(row["provider_updated_at_utc"])
             or updated == row["provider_updated_at_utc"]), "canonical provider clocks mismatch")
    _require(retained[story][1:] == (row["raw_sha256"], row["first_seen_at_utc"]),
             "canonical row is not the producer-retained provider version")
    source = f"alpaca:{item.get('source') or 'unknown'}".strip().lower()
    title, url, summary = (str(item.get("headline") or "").strip(), str(item.get("url") or ""),
                           str(item.get("summary") or ""))
    text = str(item.get("content") or item.get("summary") or "") or summary or title
    _require(row["source_family"] == "alpaca" and row["source"] == source and row["title"] == title
             and row["url"] == url and row["summary"] == summary and row["text"] == text, "canonical field fallback mismatch")
    _require(_event_id(ticker=row["ticker"], security_id=row["security_id"], source=source, published=published,
                       title=title, url=url, provider_id=_provider_event_id(item)) == row["event_id"],
             "canonical event identity mismatch")
    body_state, body = _field(item, "content")
    summary_state, summary_value = _field(item, "summary")
    category, field, chosen = ("provider_body_field", "content", body) if body_state == "nonempty" else (
        ("provider_summary_field", "summary", summary_value) if summary_state == "nonempty" else (
            "headline_only", "headline", title))
    return {"event_id": row["event_id"], "chunk_id": unit["chunk_id"], "query_security_id": row["security_id"],
        "query_ticker": unit["ticker"], "source_family": "alpaca", "source": source, "raw_sha256": row["raw_sha256"],
        "source_version_sha256": row["raw_sha256"], "provider_story_id": story,
        **{name: row[name] for name in _CLOCKS}, "availability_policy": PROXY_POLICY,
        "content_category": category, "body_state": body_state, "summary_state": summary_state,
        "chosen_field": field, "chosen_field_sha256": hashlib.sha256(chosen.encode("utf-8")).hexdigest(),
        "body_equals_title": body_state == "nonempty" and body.strip() == title,
        "raw_record_locators_json": json.dumps(match["locators"], sort_keys=True, separators=(",", ":")),
        "attribution_status": "not_established_by_content_inventory",
        "production_eligible": False, "training_eligible": False, "serving_eligible": False}


def inspect_saved_alpaca_content(*, root: Path, event_artifact: SourcePin, event_manifest: SourcePin,
                                security_id: str, ticker: str, start_utc: pd.Timestamp, cutoff_utc: pd.Timestamp,
                                memory_check: MemoryCheck) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Inspect one pinned original query chunk; the caller owns its lease and memory policy.

    Provider fields prove neither complete article bodies nor event meaning. Counts
    are chunk-local, not target-cohort attribution or known-empty source coverage.
    Rows published in [start, cutoff] are recorded; versions available after the
    cutoff keep a separate status, and rows published outside it are only counted.
    """
    root = root.resolve()
    start, cutoff = _clock(start_utc, "start"), _clock(cutoff_utc, "cutoff")
    _require(start <= cutoff and bool(security_id) and security_id.strip() == security_id, "invalid query bounds/identity")
    query_ticker = canonical_symbol(normalized_ticker(ticker))
    artifact, sidecar = inside(root, event_artifact.path), inside(root, event_manifest.path)
    _require(sidecar == manifest_path_for(artifact) and artifact.parent.name == "events", "event sidecar path mismatch")
    memory_check()
    manifest: dict[str, Any] = parse_strict_json_object(_read(sidecar, event_manifest.sha256), label="original event sidecar")
    _require(manifest.get("schema") == CANONICAL_MANIFEST_SCHEMA and manifest.get("artifact_type") == "events"
             and manifest.get("artifact_sha256") == event_artifact.sha256, "original event artifact declaration mismatch")
    declared_path = manifest.get("artifact_path")
    _require(isinstance(declared_path, str) and inside(root, declared_path) == artifact, "declared artifact path mismatch")
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataReadinessError("missing original event inputs")
    chunk, request = _identity(inputs.get("chunk_id"), inputs.get("collection_request_sha256"))
    _require(artifact.name == f"{chunk}.parquet", "original chunk artifact name mismatch")
    collection = artifact.parent.parent
    unit = _work_unit(root, collection, chunk, request, memory_check)
    _require(unit["security_id"] == security_id and unit["ticker"] == query_ticker,
             "query identity differs from its request work unit")
    _require(type(manifest.get("rows")) is int and 0 <= manifest["rows"] <= MAX_EVENT_ROWS, "event row limit/declaration invalid")
    raw = _read(artifact, event_artifact.sha256)
    parquet = pq.ParquetFile(io.BytesIO(raw))  # type: ignore[no-untyped-call]
    _require(parquet.metadata.num_rows == manifest["rows"], "canonical row count mismatch")
    uncompressed = sum(parquet.metadata.row_group(group).column(column).total_uncompressed_size
                       for group in range(parquet.metadata.num_row_groups)
                       for column in range(parquet.metadata.num_columns))
    _require(0 <= uncompressed <= MAX_PARQUET_UNCOMPRESSED_BYTES, "Parquet uncompressed data exceed 128 MiB")
    _require(set(_EVENT_COLUMNS).issubset(parquet.schema_arrow.names), "canonical event columns missing")
    wanted: set[str] = set()
    for batch in _event_batches(parquet, ["raw_sha256"], memory_check):
        for raw_hash in (row["raw_sha256"] for row in batch):
            _require(isinstance(raw_hash, str) and re.fullmatch(r"[0-9a-f]{64}", raw_hash) is not None,
                     "invalid canonical raw hash")
            wanted.add(raw_hash)
    matches, retained, diagnostics = _pages(root, collection, inputs, unit, wanted, memory_check)
    rows: list[dict[str, Any]] = []
    event_ids: set[str] = set()
    stories: set[str] = set()
    outside = 0
    for batch in _event_batches(parquet, list(_EVENT_COLUMNS), memory_check):
        for row in batch:
            _require(row["event_id"] not in event_ids, "duplicate canonical event IDs")
            event_ids.add(row["event_id"])
            _require(row["security_id"] == security_id and canonical_symbol(row["ticker"]) == query_ticker,
                     "canonical query identity mismatch")
            for name in _CLOCKS:
                row[name] = _optional_clock(row[name], name) if name == "provider_updated_at_utc" else _clock(row[name], name)
            published, updated, available = row["published_at_utc"], row["provider_updated_at_utc"], row["available_at_utc"]
            _require(pd.isna(updated) or updated >= published, "canonical update precedes publication")
            # Historical backfill availability is exactly the provider publication proxy, never receipt time.
            _require(row["availability_policy"] == PROXY_POLICY
                     and available == (published if pd.isna(updated) else max(published, updated)),
                     "canonical availability differs from its provider publication proxy")
            _require(row["raw_sha256"] in matches, "canonical raw hash has no original provider object")
            record = _record(row, matches[row["raw_sha256"]], unit, retained)
            stories.add(record["provider_story_id"])
            # Exclusion is a verified provenance result, never a way to bypass original-record checks.
            if not start <= published <= cutoff:
                outside += 1
                continue
            record["inventory_status"] = "included" if available <= cutoff else "version_after_cutoff"
            rows.append(record)
    result = _frame(rows)
    diagnostics["source_files"].update({unit["request_path"]: unit["request_file_sha256"],
        _relative(root, sidecar): event_manifest.sha256, _relative(root, artifact): event_artifact.sha256})
    statuses = {status: result.loc[result.inventory_status.eq(status)] for status in _STATUSES}
    summary = {**diagnostics, "chunk_id": unit["chunk_id"], "collection_request_sha256": unit["collection_request_sha256"],
        "query_security_id": security_id, "query_ticker": query_ticker, "query_provider_symbol": unit["provider_symbol"],
        "query_window_start_utc": unit["start"].isoformat(), "query_window_end_exclusive_utc": unit["end"].isoformat(),
        "include_content_requested": True, "availability_policy": PROXY_POLICY,
        "canonical_rows": parquet.metadata.num_rows, "recorded_rows": len(result),
        "included_rows": len(statuses["included"]), "version_after_cutoff_rows": len(statuses["version_after_cutoff"]),
        "outside_publication_window_rows": outside,
        "query_chunk_admitted_stories_without_canonical_row": len(set(retained) - stories),
        "category_counts": {status: {category: int(frame.content_category.eq(category).sum()) for category in _CATEGORIES}
                            for status, frame in statuses.items()},
        "count_scope": "single_query_chunk_not_cohort_or_coverage", "known_empty_coverage": False,
        "production_eligible": False, "training_eligible": False, "serving_eligible": False,
        "attribution_status": "not_established_by_content_inventory"}
    return result, summary


def verify_saved_alpaca_empty_chunk(*, root: Path, collection: Path, chunk_id: str, collection_request_sha256: str,
                                    security_id: str, ticker: str, memory_check: MemoryCheck) -> dict[str, Any]:
    """Verify a producer-empty query chunk from its saved pages; the caller owns the lease.

    Page envelopes, request/chunk binding and pagination are verified and the producer's
    acceptance filter must admit nothing. These pages carry no external pin: they are
    bound to the pinned request, so zero news is known only for this query window.
    """
    root = root.resolve()
    chunk, request = _identity(chunk_id, collection_request_sha256)
    collection = inside(root, collection)
    unit = _work_unit(root, collection, chunk, request, memory_check)
    _require(unit["security_id"] == security_id and unit["ticker"] == canonical_symbol(normalized_ticker(ticker)),
             "query identity differs from its request work unit")
    pages = sorted((collection / "raw_pages" / chunk).glob("page_*.json"))
    _require(0 < len(pages) <= MAX_EMPTY_CHUNK_PAGES, "producer-empty chunk requires bounded saved pages")
    inputs = {_relative(root, page): hashlib.sha256(_bytes(page)).hexdigest() for page in pages}
    _, retained, diagnostics = _pages(root, collection, inputs, unit, set(), memory_check)
    _require(not retained, "producer-empty chunk contains admitted provider items")
    diagnostics["source_files"][unit["request_path"]] = unit["request_file_sha256"]
    return {**diagnostics, "chunk_id": chunk, "collection_request_sha256": request, "query_security_id": security_id,
            "query_ticker": unit["ticker"], "query_provider_symbol": unit["provider_symbol"],
            "query_window_start_utc": unit["start"].isoformat(), "query_window_end_exclusive_utc": unit["end"].isoformat(),
            "include_content_requested": True, "evidence": "saved_pages_bound_to_request_not_externally_pinned",
            "known_empty_scope": "this_query_window_only"}
