"""Read one exact retained EDGAR document and its indexed transport evidence."""
from __future__ import annotations

import hashlib
import zipfile
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.document_collection import (
    RECEIPT_COLUMNS,
    index_rejection,
    index_url,
    parse_filing_index,
    selected_documents,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin


class SecArchivedDocument(HoldingContract):
    collection_manifest: SourcePin
    receipt_artifact: SourcePin
    unit_id: Identifier
    body_sha256: Sha256
    filing_metadata: SourcePin
    index_receipt_artifact: SourcePin | None = None


@dataclass(frozen=True)
class RetainedSecDocument:
    body: bytes
    body_path: Path
    body_member: str
    index_path: Path
    index_member: str
    index_body_sha256: str
    document_type: str
    accepted_at: pd.Timestamp
    captured_at: pd.Timestamp
    index_captured_at: pd.Timestamp


def pin_retained_source(root: Path, pin: SourcePin, files: MutableMapping[str, str], *,
    paths: EvidencePaths | None = None) -> Path:
    paths = paths or EvidencePaths(root)
    path = paths.read(pin.path)
    key = str(paths.identity(path))
    if key in files and files[key] != pin.sha256:
        raise DataReadinessError("retained SEC or identity source pins conflict")
    if key not in files and file_sha256(path) != pin.sha256:
        raise DataReadinessError("retained SEC or identity source hash differs")
    files[key] = pin.sha256
    return path


def retained_source_object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 32 * 1024**2:
        raise DataReadinessError("retained SEC or identity metadata exceeds bound")
    return parse_strict_json_object(path.read_bytes(), label="retained SEC or identity metadata")


def pin_retained_child(root: Path, parent: Path, name: str, digest: str, files: MutableMapping[str, str], *,
    paths: EvidencePaths | None = None) -> Path:
    paths = paths or EvidencePaths(root)
    path = paths.read_child(parent, name)
    return pin_retained_source(root, SourcePin(path=str(paths.identity(path)), sha256=digest), files, paths=paths)


def _verify_receipt(row: dict[str, Any]) -> None:
    cleaned = {key: None if pd.isna(row[key]) else int(row[key]) if key in {"attempt", "status_code", "body_length"}
               else row[key] for key in RECEIPT_COLUMNS if key != "receipt_sha256"}
    if json_sha256(cleaned) != row["receipt_sha256"]:
        raise DataReadinessError("retained SEC receipt hash differs")


def _zip_body(path: Path, row: dict[str, Any]) -> bytes:
    with zipfile.ZipFile(path) as archive:
        members = [item for item in archive.infolist() if item.filename == row["member"]]
        if len(members) != 1 or members[0].file_size > 16 * 1024**2:
            raise DataReadinessError("retained SEC member is absent, ambiguous or exceeds bound")
        body = archive.read(members[0])
    if len(body) != int(row["body_length"]) or hashlib.sha256(body).hexdigest() != row["body_sha256"]:
        raise DataReadinessError("retained SEC body hash or length differs")
    return body


def _capture(row: dict[str, Any], accepted: pd.Timestamp) -> pd.Timestamp:
    start, captured, completed = (pd.Timestamp(row[name])
        for name in ("started_at_utc", "retrieved_at_utc", "completed_at_utc"))
    if (any(value.tzinfo is None for value in (accepted, start, captured, completed))
            or not accepted <= start <= captured <= completed <= datetime.now(UTC)):
        raise DataReadinessError("retained SEC acceptance and transport clocks differ")
    return captured


def read_retained_sec_document(
    root: Path, anchor: SecArchivedDocument, sec_cik: str, files: MutableMapping[str, str], cache: dict[str, Any],
    *, primary_only: bool, memory_guard: Callable[[], None],
    paths: EvidencePaths | None = None,
) -> RetainedSecDocument:
    """Replay source bytes and the selected EDGAR index row without opening a mutable collection."""
    paths = paths or EvidencePaths(root)
    manifest_path = pin_retained_source(root, anchor.collection_manifest, files, paths=paths)
    manifest = retained_source_object(manifest_path)
    if (manifest.get("schema") != "market_predictor.sec_filing_document_collection"
            or manifest.get("inventory_mode") != "initial_fit" or manifest.get("status") != "complete"
            or manifest.get("sealed_until_rules_frozen") is not False):
        raise DataReadinessError("retained SEC proof requires the unsealed initial-fit collection")
    folder = manifest_path.parent / "shards"
    receipt_cache = cache.setdefault("receipt_frames", {})

    def shard_rows(pin: SourcePin) -> tuple[str, pd.DataFrame]:
        path = pin_retained_source(root, pin, files, paths=paths)
        shard = path.stem
        if (path.parent != folder or path.suffix != ".parquet" or shard not in manifest["shards"]
                or pin.sha256 != manifest["shards"][shard]["parquet_sha256"]):
            raise DataReadinessError("retained SEC receipt is outside its accepted collection")
        if str(path) not in receipt_cache:
            receipt_cache[str(path)] = pd.read_parquet(path)
        return shard, receipt_cache[str(path)]

    shard, receipts = shard_rows(anchor.receipt_artifact)
    selected = receipts.loc[receipts.unit_id.eq(anchor.unit_id) & receipts.state.eq("archived")]
    if len(selected) != 1:
        raise DataReadinessError("retained SEC proof needs one exact archived receipt")
    row = selected.iloc[0].to_dict()
    _verify_receipt(row)
    accepted_document_types = {"8-K", "8-K/A"} if primary_only else {"8-K", "8-K/A", "EX-99.1"}
    if (row["phase"] != "document" or row["document_type"] not in accepted_document_types
            or row["unit_id"] != f"{row['accession_number']}/{row['sequence']}"
            or row["member"] != f"{row['unit_id']}/{row['body_sha256']}"
            or row["sec_cik"] != sec_cik or row["body_sha256"] != anchor.body_sha256
            or row["status_code"] != 200 or row["url"] != row["final_url"] or row["redirect_chain_json"] != "[]"):
        raise DataReadinessError("retained SEC document binding differs")
    parsed_url = urlsplit(str(row["url"]))
    if (parsed_url.scheme != "https" or parsed_url.netloc != "www.sec.gov"
            or parsed_url.query or parsed_url.fragment):
        raise DataReadinessError("retained document is outside the official SEC URL")
    body_path = pin_retained_child(root, folder, shard + ".zip", manifest["shards"][shard]["zip_sha256"], files, paths=paths)
    body = _zip_body(body_path, row)
    filing_path = pin_retained_source(root, anchor.filing_metadata, files, paths=paths)
    filing_cache = cache.setdefault("filing_frames", {})
    if str(filing_path) not in filing_cache:
        filing_cache[str(filing_path)] = pd.read_parquet(filing_path, columns=["sec_cik", "accession_number", "sec_form",
            "item_codes", "report_date", "filing_date", "accepted_at_utc", "primary_document"])
    filings = filing_cache[str(filing_path)]
    metadata = filings.loc[filings.sec_cik.eq(sec_cik) & filings.accession_number.eq(row["accession_number"])].drop_duplicates()
    if len(metadata) != 1:
        raise DataReadinessError("retained SEC accession metadata is ambiguous")
    item = metadata.iloc[0].to_dict()
    accepted = pd.Timestamp(item["accepted_at_utc"])
    captured = _capture(row, accepted)
    accession, index_hash = row["accession_number"], row["index_body_sha256"]
    if anchor.index_receipt_artifact is not None:
        index_shard, index_frame = shard_rows(anchor.index_receipt_artifact)
        candidates = index_frame.loc[index_frame.phase.eq("index") & index_frame.state.eq("archived")
            & index_frame.accession_number.eq(accession) & index_frame.body_sha256.eq(index_hash)]
        index_rows = [(index_shard, candidate) for candidate in candidates.to_dict("records")]
    else:
        index_key = "index_rows:" + str(manifest_path) + ":" + anchor.collection_manifest.sha256
        if index_key not in cache:
            index_lookup: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
            for name, record in manifest["shards"].items():
                memory_guard()
                _, frame = shard_rows(SourcePin(path=str(folder / f"{name}.parquet"), sha256=record["parquet_sha256"]))
                found = frame.loc[frame.phase.eq("index") & frame.state.eq("archived")]
                for candidate in found.to_dict("records"):
                    index_lookup.setdefault((candidate["accession_number"], candidate["body_sha256"]), []).append((name, candidate))
            cache[index_key] = index_lookup
        index_rows = cache[index_key].get((accession, index_hash), [])
    if len(index_rows) != 1:
        raise DataReadinessError("retained SEC document lacks one exact detail-page receipt")
    index_shard, index_row = index_rows[0]
    _verify_receipt(index_row)
    if (index_row["unit_id"] != f"{accession}/index"
            or index_row["member"] != f"{index_row['unit_id']}/{index_row['body_sha256']}"
            or index_row["sec_cik"] != sec_cik or index_row["status_code"] != 200
            or index_row["url"] != index_url(sec_cik, str(accession))
            or index_row["final_url"] != index_row["url"] or index_row["redirect_chain_json"] != "[]"):
        raise DataReadinessError("retained SEC detail-page transport differs")
    index_captured = _capture(index_row, accepted)
    if index_captured > captured:
        raise DataReadinessError("retained SEC document was captured before its detail page")
    index_path = pin_retained_child(root, folder, index_shard + ".zip", manifest["shards"][index_shard]["zip_sha256"], files, paths=paths)
    index = parse_filing_index(_zip_body(index_path, index_row), accession=str(accession))
    comparable = {key: str(value) for key, value in item.items()}
    comparable["accepted_at_utc"] = accepted.isoformat()
    if index_rejection(index, comparable) is not None:
        raise DataReadinessError("retained SEC detail page contradicts filing metadata")
    documents = selected_documents(index, str(item["primary_document"]))
    selection = parse_strict_json_object(str(row["index_row_json"]), label="SEC selected document row")
    exact = [doc for doc in documents if doc.row() == selection and doc.document_type == row["document_type"]
             and (not primary_only or doc.filename == item["primary_document"])]
    if len(exact) != 1 or row["url"] != "https://www.sec.gov" + exact[0].path:
        raise DataReadinessError("retained SEC document differs from its exact listed URL or index row")
    return RetainedSecDocument(body, paths.identity(body_path), str(row["member"]), paths.identity(index_path), str(index_row["member"]),
        str(index_hash), str(row["document_type"]), accepted, captured, index_captured)
