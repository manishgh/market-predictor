"""Read-only, bounded sources for initial-fit content review, never admission.

The caller owns the shared heavy-job lease, exhausts the iterators, and rechecks
their accumulated pins before publication. Historical implementation/configuration
pins remain evidence; these readers do not rewrite them or reopen collection stores.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
import zipfile
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import pandas as pd
import pyarrow.parquet as pq
from bs4 import BeautifulSoup, Tag

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.document_collection import (
    RECEIPT_COLUMNS,
    TERMINAL_STATES,
    parse_filing_header,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_materialization import SourcePin

MAX_BODY_BYTES = 16 * 1024**2
MAX_PAGE_BYTES = 64 * 1024**2
MAX_PARQUET_BYTES = 128 * 1024**2
END = pd.Timestamp("2024-05-28T22:00:00Z")
START = pd.Timestamp("2019-07-09T00:00:00Z")
MemoryCheck = Callable[[], None]


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _sha(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def _clock(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, "review source clock must be timezone-aware")
    return stamp.tz_convert("UTC")


def _optional(value: Any) -> Any:
    return None if value is None or not isinstance(value, (str, dict, list)) and pd.isna(value) else value


def _decoded_size(value: Any) -> int:
    """Bound cached JSON by actual Python object sizes, not compressed file size."""
    pending = [value]
    seen: set[int] = set()
    total = 0
    while pending:
        item = pending.pop()
        if id(item) in seen:
            continue
        seen.add(id(item))
        total += sys.getsizeof(item)
        _require(total <= MAX_PARQUET_BYTES, "decoded original page exceeds 128 MiB cache bound")
        if isinstance(item, dict):
            pending.extend(item.keys())
            pending.extend(item.values())
        elif isinstance(item, list):
            pending.extend(item)
    return total


def _pin(root: Path, path: Path, digest: str, pins: dict[str, str]) -> Path:
    path = inside(root, path)
    name = path.relative_to(root).as_posix()
    _require(isinstance(digest, str) and re.fullmatch(r"[0-9a-f]{64}", digest) is not None, "invalid source hash")
    _require(name not in pins or pins[name] == digest, "conflicting review source pins")
    _require(file_sha256(path) == digest, f"review source hash differs: {name}")
    pins[name] = digest
    return path


def recheck_source_pins(root: Path, pins: dict[str, str]) -> None:
    for name, digest in pins.items():
        _require(file_sha256(inside(root, name)) == digest, f"review source changed: {name}")


def _bytes(path: Path, maximum: int) -> bytes:
    _require(path.stat().st_size <= maximum, "review source exceeds byte bound")
    with path.open("rb") as stream:
        raw = stream.read(maximum + 1)
    _require(len(raw) <= maximum, "review source grew beyond byte bound")
    return raw


def _json(root: Path, path: Path, digest: str | None, pins: dict[str, str], maximum: int = MAX_BODY_BYTES) -> dict[str, Any]:
    path = inside(root, path)
    raw = _bytes(path, maximum)
    actual = _sha(raw)
    _require(digest is None or actual == digest, "review JSON hash differs")
    _pin(root, path, actual, pins)
    return parse_strict_json_object(raw, label=str(path))


def _rows(path: Path, memory_check: MemoryCheck, columns: list[str] | None = None) -> Iterator[dict[str, Any]]:
    _require(path.stat().st_size <= MAX_PARQUET_BYTES, "review partition exceeds byte bound")
    parquet: Any = pq.ParquetFile(path)  # type: ignore[no-untyped-call]
    _require(parquet.metadata.serialized_size <= 8 * 1024**2 and parquet.metadata.num_rows <= 500_000,
             "review partition metadata exceeds bound")
    _require(sum(parquet.metadata.row_group(i).total_byte_size for i in range(parquet.num_row_groups)) <= 512 * 1024**2,
             "review partition decoded size exceeds bound")
    for batch in parquet.iter_batches(batch_size=128, columns=columns):
        memory_check()
        _require(batch.nbytes <= MAX_PAGE_BYTES, "review batch exceeds decoded byte bound")
        for row in batch.to_pandas().to_dict("records"):
            yield {name: _optional(value) for name, value in row.items()}


@dataclass(frozen=True)
class ReviewSourceRecord:
    source_family: Literal["alpaca", "sec"]
    source_id: str
    source_version_sha256: str | None
    security_id: str | None
    ticker: str | None
    published_at_utc: pd.Timestamp
    version_available_at_utc: pd.Timestamp
    first_seen_at_utc: pd.Timestamp | None
    payload: dict[str, Any] | bytes | None
    content_kind: str
    source_locator: str
    metadata: dict[str, Any]
    unavailable_reasons: tuple[str, ...]


def iter_alpaca_review_sources(*, root: Path, inventory: SourcePin, pins: dict[str, str],
                              memory_check: MemoryCheck) -> Iterator[ReviewSourceRecord]:
    """Yield every retained inventory occurrence, including unclassified identities.

Repeated query occurrences deliberately remain present for publisher-side grouping
by original story/version. Superseded versions outside the closed inventory are
not silently asserted to have been inventoried by this reader.
"""
    root = root.resolve()
    manifest_path = inside(root, inventory.path)
    manifest = _json(root, manifest_path, inventory.sha256, pins)
    _require(manifest.get("schema") == "market_predictor.issuer_content_cohort_inventory"
             and manifest.get("status") == "complete_inventory_only", "Alpaca content inventory is incomplete")
    request = _json(root, manifest_path.parent / "_request.json", manifest["request_sha256"], pins)
    _require(_clock(request["window"]["start_utc"]) == START
             and _clock(request["window"]["cutoff_utc"]) == END and manifest["window"] == request["window"],
             "review source must use the initial-fit window")
    count = 0
    for name, part in sorted(manifest["parts"].items()):
        memory_check()
        _require(re.fullmatch(r"(?:early|later|corrected)-\d{4}", name) is not None, "invalid inventory part name")
        directory = manifest_path.parent / "parts"
        units = _json(root, directory / f"{name}.units.json", part["units_sha256"], pins)
        _require(units["request_sha256"] == manifest["request_sha256"], "inventory unit request differs")
        unit_map = {row["chunk_id"]: row for row in units["units"]}
        _require(len(unit_map) == len(units["units"]), "duplicate content inventory unit")
        path = _pin(root, directory / f"{name}.parquet", part["records_sha256"], pins)
        page_cache: dict[tuple[str, str], dict[str, Any]] = {}
        cached_chunk: str | None = None
        cached_raw_bytes = cached_decoded_bytes = 0
        for row in _rows(path, memory_check):
            count += 1
            unit = unit_map[row["chunk_id"]]
            if row["chunk_id"] != cached_chunk:
                page_cache.clear()
                cached_raw_bytes = cached_decoded_bytes = 0
                cached_chunk = row["chunk_id"]
            _require(row["query_security_id"] == unit["query_security_id"] and row["query_ticker"] == unit["query_ticker"],
                     "inventory query identity differs from its unit")
            locators = json.loads(row["raw_record_locators_json"])
            _require(isinstance(locators, list) and bool(locators), "inventory record lacks original locators")
            original: dict[str, Any] | None = None
            collected: list[pd.Timestamp] = []
            for locator in locators:
                page_path, digest = locator["page_path"], locator["page_sha256"]
                _require(unit["source_files"].get(page_path) == digest, "original page lacks closed matching pin")
                key = (page_path, digest)
                if key not in page_cache:
                    memory_check()
                    source = inside(root, page_path)
                    cached_raw_bytes += source.stat().st_size
                    _require(cached_raw_bytes <= MAX_PAGE_BYTES, "original chunk pages exceed 64 MiB cache bound")
                    page = _json(root, source, digest, pins, MAX_PAGE_BYTES)
                    cached_decoded_bytes += _decoded_size(page)
                    _require(cached_decoded_bytes <= MAX_PARQUET_BYTES, "original chunk decoded cache exceeds 128 MiB")
                    _require(page.get("chunk_id") == row["chunk_id"], "original page belongs to another query")
                    _require(page.get("content_sha256") == json_sha256({k: v for k, v in page.items() if k != "content_sha256"}),
                             "original page envelope differs")
                    page_cache[key] = page
                page = page_cache[key]
                position = locator["news_index"]
                _require(type(position) is int and 0 <= position < len(page["news"]), "invalid original record locator")
                item = page["news"][position]
                _require(isinstance(item, dict), "original article is not an object")
                digest = _sha(json.dumps(item, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8"))
                _require(digest == row["source_version_sha256"] == row["raw_sha256"], "original article version differs")
                _require(original is None or original == item, "original record locators disagree")
                original = item
                collected.append(_clock(page["collected_at_utc"]))
            assert original is not None
            published = _clock(original["created_at"])
            version = published if original.get("updated_at") is None else _clock(original["updated_at"])
            seen = _clock(row["first_seen_at_utc"])
            _require(published == _clock(row["published_at_utc"]) and version >= published
                     and seen in collected and seen >= version, "original article clocks differ")
            updated = row["provider_updated_at_utc"]
            _require(version == (published if updated is None else _clock(updated)), "inventory revision clock differs")
            field = row["chosen_field"]
            _require(field in {"content", "summary", "headline"} and isinstance(original.get(field), str), "chosen field differs")
            chosen = original[field].strip() if field == "headline" else original[field]
            _require(_sha(chosen.encode("utf-8")) == row["chosen_field_sha256"], "original selected-field hash differs")
            reasons = []
            if row["inventory_status"] != "included":
                reasons.append(str(row["inventory_status"]))
            if row["query_identity_resolution"] not in {"bridged", "identity_equal", "proven_legacy_identity"}:
                reasons.append(str(row["query_identity_resolution"]))
            if not row.get("cohort_security_id"):
                reasons.append("missing_cohort_identity")
            _require(START <= published <= END, "inventory publication escapes initial fit")
            yield ReviewSourceRecord("alpaca", str(row["provider_story_id"]), row["source_version_sha256"],
                row.get("cohort_security_id"), row["query_ticker"], published, version, seen, original,
                field, json.dumps(locators, sort_keys=True), row, tuple(reasons))
        _pin(root, path, part["records_sha256"], pins)
    _require(count == manifest["records_rows"], "content inventory row count differs")
    recheck_source_pins(root, pins)


def _sec_metadata(root: Path, archive: SourcePin, inventory: SourcePin, pins: dict[str, str]
                  ) -> tuple[Path, dict[str, Any], dict[str, Any], dict[str, Any]]:
    path = inside(root, archive.path)
    manifest = _json(root, path, archive.sha256, pins)
    _require(manifest.get("schema") == "market_predictor.sec_filing_document_collection"
             and manifest.get("status") == "complete" and manifest.get("inventory_mode") == "initial_fit"
             and manifest.get("sealed_until_rules_frozen") is False, "SEC collection is sealed or not complete initial fit")
    request = _json(root, path.parent / "_request.json", None, pins)
    semantic = json_sha256({key: value for key, value in request.items() if key != "request_sha256"})
    _require(semantic == request.get("request_sha256") == manifest["request_sha256"], "SEC request identity differs")
    _require(request.get("inventory_mode") == "initial_fit" and request.get("sealed_until_rules_frozen") is False
             and request["inventory"] == inventory.model_dump(mode="json") and request.get("pilot_accessions") == [],
             "SEC request is sealed, partial or bound to another inventory")
    parent = _json(root, inside(root, inventory.path), inventory.sha256, pins)
    _require(parent.get("schema") == "market_predictor.sec_form_inventory" and parent.get("status") == "complete"
             and parent.get("mode") == "initial_fit" and parent.get("sealed_until_rules_frozen", False) is False,
             "SEC form inventory is sealed or incomplete")
    _require(_clock(parent["window"]["start_utc"]) == START and _clock(parent["window"]["end_utc"]) == END,
             "SEC form inventory window differs")
    checkpoint = _json(root, path.parent / "_checkpoint.json", manifest["checkpoint_sha256"], pins)
    _require(checkpoint["request_sha256"] == semantic and checkpoint["shards"] == manifest["shards"],
             "SEC checkpoint and manifest differ")
    return path.parent, manifest, request, parent


def _filing_map(root: Path, inventory: SourcePin, parent: dict[str, Any], pins: dict[str, str],
                memory_check: MemoryCheck) -> tuple[dict[str, list[dict[str, Any]]], str]:
    path = inside(root, inventory.path).parent / "filings.parquet"
    _pin(root, path, parent["artifacts"]["filings.parquet"], pins)
    authority = [(name, digest) for name, digest in parent["source_files"].items()
                 if name.endswith("sec_acceptance_clock/_manifest.json")]
    _require(len(authority) == 1, "corrected SEC acceptance clock authority is missing")
    _pin(root, inside(root, authority[0][0]), authority[0][1], pins)
    relations = [(name, digest) for name, digest in parent["source_files"].items()
                 if name.endswith("/sec_identity_relations.parquet")]
    _require(len(relations) == 1, "SEC ticker identity relation pin is missing")
    relation_path = _pin(root, inside(root, relations[0][0]), relations[0][1], pins)
    identities: dict[str, list[dict[str, Any]]] = {}
    identity_columns = ["security_id", "ticker", "sec_cik", "effective_from_utc", "effective_to_utc", "available_at_utc"]
    for row in _rows(relation_path, memory_check, identity_columns):
        identities.setdefault(row["security_id"], []).append(row)
    records: dict[str, list[dict[str, Any]]] = {}
    for row in _rows(path, memory_check):
        if row["sec_form"] not in {"8-K", "8-K/A"} or not {"2.02", "7.01", "8.01"}.intersection(str(row["item_codes"]).split(",")):
            continue
        accepted = _clock(row["accepted_at_utc"])
        available = _clock(row["available_at_utc"])
        _require(accepted <= available and START <= available <= END, "SEC filing escapes initial fit")
        matching = [item for item in identities.get(row["security_id"], [])
                    if str(item["sec_cik"]).zfill(10) == str(row["sec_cik"]).zfill(10)
                    and _clock(item["effective_from_utc"]) <= available
                    and _clock(item["available_at_utc"]) <= available
                    and (item["effective_to_utc"] is None or available < _clock(item["effective_to_utc"]))]
        tickers = {item["ticker"] for item in matching}
        _require(len(tickers) <= 1, "SEC filing has competing ticker identities")
        row["ticker"] = next(iter(tickers), None)
        row["identity_available_at_utc"] = max((_clock(item["available_at_utc"]) for item in matching), default=None)
        row["identity_authority_sha256"] = relations[0][1]
        records.setdefault(row["accession_number"], []).append(row)
    return records, authority[0][1]


def _zip_body(path: Path, row: dict[str, Any]) -> bytes:
    with zipfile.ZipFile(path) as archive:
        entries = [item for item in archive.infolist() if item.filename == row["member"]]
        _require(len(entries) == 1 and not entries[0].is_dir(), "SEC body member missing or duplicated")
        info = entries[0]
        _require(0 <= info.file_size <= MAX_BODY_BYTES and info.file_size == row["body_length"], "SEC ZIP member exceeds body bound")
        with archive.open(info) as stream:
            body = stream.read(MAX_BODY_BYTES + 1)
        _require(len(body) == info.file_size and _sha(body) == row["body_sha256"], "SEC body hash/length differs")
        return body


def iter_sec_review_sources(*, root: Path, archive: SourcePin, inventory: SourcePin, pins: dict[str, str],
                           memory_check: MemoryCheck, phase: Literal["index", "document"] = "document"
                           ) -> Iterator[ReviewSourceRecord]:
    """Read selected final receipts without calling the collection store or opening sealed bodies.

Index-only traversal hashes no document-only ZIP. Prior retry reasons are retained
on each final unit. Missing, rejected, oversize and unsupported PDFs yield rows.
"""
    _require(phase in {"index", "document"}, "unsupported SEC review phase")
    root = root.resolve()
    directory, manifest, request, parent = _sec_metadata(root, archive, inventory, pins)
    filings, clock_pin = _filing_map(root, inventory, parent, pins, memory_check)
    _require(len(filings) == request["work_list_units"], "SEC inventory selected accession count differs")
    terminal: set[str] = set()
    retries: dict[str, list[dict[str, Any]]] = {}
    seen_accessions: set[str] = set()
    for name, shard in sorted(manifest["shards"].items()):
        memory_check()
        _require(re.fullmatch(r"shard-\d{5}", name) is not None, "invalid SEC shard name")
        path = _pin(root, directory / "shards" / f"{name}.parquet", shard["parquet_sha256"], pins)
        count, zip_checked = 0, False
        for row in _rows(path, memory_check):
            count += 1
            clean = {key: int(row[key]) if row[key] is not None and key in {"attempt", "status_code", "body_length"}
                     else row[key] for key in RECEIPT_COLUMNS if key != "receipt_sha256"}
            _require(row["shard"] == name and json_sha256(clean) == row["receipt_sha256"], "SEC receipt hash differs")
            if row["phase"] != phase:
                continue
            unit = str(row["unit_id"])
            _require(unit not in terminal, "SEC unit has attempts after its final outcome")
            if row["state"] not in TERMINAL_STATES:
                retries.setdefault(unit, []).append({key: row[key] for key in ("attempt", "state", "reason", "receipt_sha256")})
                continue
            terminal.add(unit)
            accession = row["accession_number"]
            _require(accession in filings, "SEC receipt is outside the selected initial-fit inventory")
            filer_ciks = {str(item["sec_cik"]).zfill(10) for item in filings[accession]}
            _require(str(row["sec_cik"]).zfill(10) in filer_ciks, "SEC receipt CIK is not a selected accession filer")
            seen_accessions.add(accession)
            reasons = [] if row["state"] == "archived" else [f"sec_{row['state']}"]
            if row["reason"]:
                reasons.append(str(row["reason"]))
            body: bytes | None = None
            if row["member"] is not None:
                if not zip_checked:
                    _pin(root, directory / "shards" / f"{name}.zip", shard["zip_sha256"], pins)
                    zip_checked = True
                body = _zip_body(directory / "shards" / f"{name}.zip", row)
                if body.startswith(b"%PDF") or "pdf" in str(row["content_type"]).lower():
                    reasons.append("pdf_requires_independent_text_extraction")
                if row["content_encoding"] not in (None, "", "identity"):
                    reasons.append("unsupported_content_encoding")
            elif row["state"] == "archived":
                raise DataReadinessError("archived SEC document has no body")
            prior_attempts = retries.pop(unit, [])
            for filing in filings[accession]:
                accepted = _clock(filing["accepted_at_utc"])
                if phase == "index" and body is not None and not reasons:
                    _require(parse_filing_header(body, accession=accession).accepted_at_utc == accepted,
                             "SEC index and corrected acceptance differ")
                extra = ["missing_ticker_identity"] if not filing["ticker"] else []
                metadata = {**row, "sec_cik": str(filing["sec_cik"]).zfill(10), "retrieval_sec_cik": row["sec_cik"],
                    "filing": filing, "accepted_at_utc": accepted.isoformat(),
                    "identity_available_at_utc": filing["identity_available_at_utc"],
                    "identity_authority_sha256": filing["identity_authority_sha256"],
                    "available_at_utc": _clock(filing["available_at_utc"]).isoformat(),
                    "acceptance_clock_authority_sha256": clock_pin, "inventory_sha256": inventory.sha256,
                    "prior_attempts": prior_attempts, "availability_semantics": "historical_proxy"}
                yield ReviewSourceRecord("sec", unit, row["body_sha256"], filing["security_id"], filing["ticker"],
                    accepted, _clock(filing["available_at_utc"]),
                    None if row["retrieved_at_utc"] is None else _clock(row["retrieved_at_utc"]),
                    body if not reasons else None, phase if phase == "index" else str(row["document_type"]),
                    f"{(directory / 'shards' / (name + '.zip')).relative_to(root).as_posix()}!{row['member']}",
                    metadata, tuple([*reasons, *extra]))
        _require(count == shard["attempts"], "SEC receipt shard count differs")
    expected = sum(count for key, count in manifest["totals"]["units_by_phase_and_state"].items() if key.startswith(phase + "/"))
    _require(len(terminal) == expected and not retries, "SEC final unit inventory differs or has unfinished attempts")
    if phase == "index":
        _require(seen_accessions == set(filings), "SEC index population is incomplete")
    recheck_source_pins(root, pins)


@dataclass(frozen=True)
class AliasProof:
    sec_cik: str
    aliases: tuple[str, ...]
    available_at_utc: pd.Timestamp
    source_sha256: str
    clock_authority_sha256: str
    source_locator: str


@dataclass(frozen=True)
class AliasExtraction:
    proofs: tuple[AliasProof, ...]
    unavailable_reasons: tuple[str, ...]


def extract_sec_alias_proofs(*, body: bytes, body_sha256: str, accession: str, expected_cik: str,
                             accepted_at_utc: pd.Timestamp, clock_authority_sha256: str,
                             locator: str) -> AliasExtraction:
    """Use only the matching companyName/CIK block of the original filing index."""
    _require(len(body) <= MAX_BODY_BYTES and _sha(body) == body_sha256, "SEC alias body hash/bound differs")
    _require(re.fullmatch(r"\d{1,10}", expected_cik) is not None
             and re.fullmatch(r"[0-9a-f]{64}", clock_authority_sha256) is not None, "invalid alias identity/clock pin")
    accepted = _clock(accepted_at_utc)
    _require(accepted <= END and parse_filing_header(body, accession=accession).accepted_at_utc == accepted,
             "SEC alias acceptance differs or escapes initial fit")
    soup = BeautifulSoup(body, "html.parser")
    names: list[tuple[str, str]] = []
    for index, block in enumerate(soup.select(".companyInfo")):
        node = block.find(class_="companyName")
        if not isinstance(node, Tag):
            continue
        text = " ".join(node.get_text(" ", strip=True).split())
        ciks = re.findall(r"\bCIK\s*:\s*(\d{1,10})\b", text, re.IGNORECASE)
        if len(ciks) != 1 or ciks[0].zfill(10) != expected_cik.zfill(10):
            continue
        match = re.match(r"(.+?)\s*\(Filer\)\s*CIK\s*:", text, re.IGNORECASE)
        if match is not None:
            names.append((match[1].strip(), f"{locator}#companyInfo[{index + 1}]/companyName"))
    if not names:
        return AliasExtraction((), ("missing_exact_filer_company_name",))
    if len({name.casefold() for name, _ in names}) != 1:
        return AliasExtraction((), ("conflicting_filer_company_names",))
    name, source = names[0]
    stripped = re.sub(r"(?:[,\s]+(?:incorporated|inc|corporation|corp|company|co|limited|ltd|plc|llc)\.?)$", "", name,
                      flags=re.IGNORECASE).strip(" ,")
    aliases = tuple(dict.fromkeys([name, *([stripped] if len(stripped) >= 2 else [])]))
    return AliasExtraction((AliasProof(expected_cik.zfill(10), aliases, accepted, body_sha256,
                                       clock_authority_sha256, source),), ())


def read_early_alias_proofs(*, root: Path, artifact: SourcePin, sidecar: SourcePin, pins: dict[str, str],
                           memory_check: MemoryCheck) -> tuple[AliasProof, ...]:
    """Reuse the independently pinned two-row filing-proven FISV/SATS authority."""
    root = root.resolve()
    path = _pin(root, inside(root, artifact.path), artifact.sha256, pins)
    manifest = _json(root, inside(root, sidecar.path), sidecar.sha256, pins)
    _require(manifest["artifact_sha256"] == artifact.sha256 and manifest["rows"] == 2
             and manifest["production_ready"] is False, "early alias authority differs")
    result = []
    for row in _rows(path, memory_check):
        _require(row["security_id"] in {"cik:0000798354", "cik:0001415404"}
                 and row["availability_policy"] == "filing_date_next_day_new_york_publication_proxy",
                 "early alias identity/availability differs")
        aliases = json.loads(row["official_aliases"])
        _require(isinstance(aliases, list) and bool(aliases) and all(isinstance(name, str) and len(name) >= 2 for name in aliases),
                 "early aliases are malformed")
        result.append(AliasProof(row["security_id"].removeprefix("cik:"), tuple(aliases), _clock(row["available_at_utc"]),
            artifact.sha256, sidecar.sha256, f"{artifact.path}#{row['identity_document_id']}:{row['record_locator']}"))
    _require(len(result) == 2 and len({item.sec_cik for item in result}) == 2, "early alias authority population differs")
    recheck_source_pins(root, pins)
    return tuple(result)
