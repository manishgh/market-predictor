"""Reconstructed research stock identity from exact retained SEC class evidence.

Cover facts establish a class anchor; only the accepted effective intervals own
sessions. Index exit, common CIKs and provider captures never extend those bounds.
"""
from __future__ import annotations

import tempfile
import zipfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

import exchange_calendars as xcals
import pandas as pd
from pydantic import Field

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
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.universe.security_class_evidence import (
    SecurityClassEvidence,
    SecurityClassIssuerMismatch,
    extract_security_class_evidence,
)


class SecClassAnchor(HoldingContract):
    collection_manifest: SourcePin
    receipt_artifact: SourcePin
    unit_id: str = Field(min_length=1)
    body_sha256: Sha256
    filing_metadata: SourcePin


class StockIdentityUnavailable(DataReadinessError):
    """Retained sources do not establish a required class or ownership fact."""


class StockIdentityProof(HoldingContract):
    security_id: str = Field(min_length=1)
    ticker: str = Field(min_length=1)
    sec_cik: str = Field(pattern=r"^[0-9]{10}$")
    security_title: str = Field(min_length=1)
    identity_authority: SourcePin
    membership_authority: SourcePin
    anchor: SecClassAnchor


class StockIdentityManifest(HoldingContract):
    proofs: tuple[StockIdentityProof, ...]


@dataclass(frozen=True)
class ReplayedStockIdentity:
    security_id: str
    ticker: str
    sec_cik: str
    security_title: str
    exchange: str
    issuer_name: str
    evidence: tuple[EvidenceReference, ...]
    proven_sessions: tuple[date, ...]
    missing_reasons: tuple[str, ...]
    competing_class_tickers: tuple[str, ...]
    anchor_accepted_at_utc: datetime
    anchor_retrieved_at_utc: datetime
    verified_at_utc: datetime


def _pinned(root: Path, pin: SourcePin, files: MutableMapping[str, str]) -> Path:
    path = inside(root, pin.path)
    key = str(path)
    if key in files and files[key] != pin.sha256:
        raise DataReadinessError("stock identity has conflicting source pins")
    if key not in files and file_sha256(path) != pin.sha256:
        raise DataReadinessError("stock identity source hash differs")
    files[key] = pin.sha256
    return path


def _object(path: Path) -> dict[str, Any]:
    if path.stat().st_size > 32 * 1024**2:
        raise DataReadinessError("stock identity metadata exceeds bound")
    return parse_strict_json_object(path.read_bytes(), label="stock identity metadata")


def _child(root: Path, parent: Path, name: str, digest: str, files: MutableMapping[str, str]) -> Path:
    return _pinned(root, SourcePin(path=str(inside(parent, name)), sha256=digest), files)


def _authority_frames(root: Path, proof: StockIdentityProof, files: MutableMapping[str, str]
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    authority_path = _pinned(root, proof.identity_authority, files)
    authority = _object(authority_path)
    if (authority.get("schema") != "edge_rebuild.sec_identity_authority.v2"
            or authority.get("state") != "identity_complete" or authority.get("artifact") != "_manifest.json"):
        raise DataReadinessError("stock identity requires the exact retained accepted SEC authority")
    directory = authority_path.parent
    manifest = _object(_child(root, directory, "_manifest.json", authority["artifact_sha256"], files))
    request_path = directory / "_request.json"
    request = _object(request_path)
    files[str(request_path)] = file_sha256(request_path)
    if (manifest.get("schema") != "edge_rebuild.sec_identity_manifest.v2" or manifest.get("status") != "complete"
            or request.get("schema") != "edge_rebuild.sec_identity_request.v2"
            or json_sha256({k: v for k, v in request.items() if k != "request_sha256"}) != authority["request_sha256"]
            or request.get("request_sha256") != authority["request_sha256"]
            or manifest.get("request_sha256") != authority["request_sha256"]
            or request.get("parent_lineage") != manifest.get("parent_lineage")
            or request["parent_lineage"]["membership_authority"]["authority_sha256"] != proof.membership_authority.sha256):
        raise DataReadinessError("stock SEC relation parent/request binding differs")
    relation = manifest["relation_artifact"]
    relations = pd.read_parquet(_child(root, directory, relation["path"], relation["sha256"], files))
    membership_path = _pinned(root, proof.membership_authority, files)
    membership_authority = _object(membership_path)
    if (membership_authority.get("state") != "membership_complete"
            or membership_authority.get("artifact") != "_manifest.json"):
        raise DataReadinessError("stock membership authority is incomplete")
    member_manifest = _object(_child(root, membership_path.parent, "_manifest.json",
                                    membership_authority["artifact_sha256"], files))
    member_artifact = member_manifest["membership_artifact"]
    members = pd.read_parquet(_child(root, membership_path.parent, member_artifact["path"],
                                    member_artifact["sha256"], files))
    required = {"security_id", "ticker", "effective_from_utc", "effective_to_utc"}
    for frame in (relations, members):
        if not required.issubset(frame.columns) or not frame.columns.is_unique:
            raise DataReadinessError("stock ownership intervals lack required columns")
        for field in ("effective_from_utc", "effective_to_utc"):
            if any(pd.Timestamp(value).tzinfo is None for value in frame[field].dropna()):
                raise DataReadinessError("stock ownership interval clock is naive")
            frame[field] = pd.to_datetime(frame[field], utc=True)
        if frame.effective_from_utc.isna().any() or (
                frame.effective_to_utc.notna() & frame.effective_to_utc.le(frame.effective_from_utc)).any():
            raise DataReadinessError("stock ownership interval bounds differ")
    return relations, members


def _receipt(row: dict[str, Any]) -> dict[str, Any]:
    cleaned = {key: None if pd.isna(row[key]) else int(row[key]) if key in {"attempt", "status_code", "body_length"}
               else row[key] for key in RECEIPT_COLUMNS if key != "receipt_sha256"}
    if json_sha256(cleaned) != row["receipt_sha256"]:
        raise DataReadinessError("stock SEC receipt hash differs")
    return cleaned


def _zip_body(path: Path, row: dict[str, Any]) -> bytes:
    import hashlib

    with zipfile.ZipFile(path) as archive:
        member = archive.getinfo(str(row["member"]))
        if member.file_size > 16 * 1024**2:
            raise DataReadinessError("stock class body exceeds bound")
        body = archive.read(member)
    if len(body) != int(row["body_length"]) or hashlib.sha256(body).hexdigest() != row["body_sha256"]:
        raise DataReadinessError("stock SEC body hash or length differs")
    return body


def _class_anchor(root: Path, proof: StockIdentityProof, files: MutableMapping[str, str], cache: dict[str, Any]
                  ) -> tuple[SecurityClassEvidence, pd.Timestamp, pd.Timestamp, str]:
    anchor = proof.anchor
    manifest_path = _pinned(root, anchor.collection_manifest, files)
    manifest = _object(manifest_path)
    if (manifest.get("schema") != "market_predictor.sec_filing_document_collection"
            or manifest.get("inventory_mode") != "initial_fit" or manifest.get("sealed_until_rules_frozen") is not False):
        raise DataReadinessError("stock class requires the unsealed initial-fit retained collection")
    receipt_path = _pinned(root, anchor.receipt_artifact, files)
    shard = receipt_path.stem
    if (receipt_path.parent != manifest_path.parent / "shards" or shard not in manifest["shards"]
            or anchor.receipt_artifact.sha256 != manifest["shards"][shard]["parquet_sha256"]):
        raise DataReadinessError("stock class receipt is outside its accepted collection")
    receipt_cache = cache.setdefault("receipt_frames", {})
    if str(receipt_path) not in receipt_cache:
        receipt_cache[str(receipt_path)] = pd.read_parquet(receipt_path)
    receipts = receipt_cache[str(receipt_path)]
    selected = receipts.loc[receipts.unit_id.eq(anchor.unit_id) & receipts.state.eq("archived")]
    if len(selected) != 1:
        raise DataReadinessError("stock class needs one exact archived receipt")
    row = selected.iloc[0].to_dict()
    _receipt(row)
    if (row["phase"] != "document" or row["document_type"] not in {"8-K", "8-K/A"}
            or row["sec_cik"] != proof.sec_cik or row["body_sha256"] != anchor.body_sha256
            or row["status_code"] != 200 or row["url"] != row["final_url"] or row["redirect_chain_json"] != "[]"):
        raise DataReadinessError("stock class SEC document binding differs")
    parsed_url = urlsplit(str(row["url"]))
    if (parsed_url.scheme != "https" or parsed_url.netloc != "www.sec.gov"
            or parsed_url.query or parsed_url.fragment):
        raise DataReadinessError("stock class source is not its official SEC issuer URL")
    zip_path = _child(root, receipt_path.parent, shard + ".zip", manifest["shards"][shard]["zip_sha256"], files)
    body = _zip_body(zip_path, row)
    filing_path = _pinned(root, anchor.filing_metadata, files)
    filing_cache = cache.setdefault("filing_frames", {})
    if str(filing_path) not in filing_cache:
        filing_cache[str(filing_path)] = pd.read_parquet(filing_path, columns=["sec_cik", "accession_number", "sec_form",
            "item_codes", "report_date", "filing_date", "accepted_at_utc", "primary_document"])
    filings = filing_cache[str(filing_path)]
    metadata = filings.loc[filings.sec_cik.eq(proof.sec_cik) & filings.accession_number.eq(row["accession_number"])].drop_duplicates()
    if len(metadata) != 1:
        raise DataReadinessError("stock class accession metadata is ambiguous")
    item = metadata.iloc[0].to_dict()
    accepted = pd.Timestamp(item["accepted_at_utc"])
    captured = pd.Timestamp(row["retrieved_at_utc"])
    if accepted.tzinfo is None or captured.tzinfo is None or accepted > captured:
        raise DataReadinessError("stock class acceptance/capture clock differs")
    if accepted >= pd.Timestamp("2024-05-29T00:00:00Z"):
        raise DataReadinessError("stock class anchor escapes the frozen pre-boundary scope")
    # The primary source must be named by the exact verified SEC detail page.
    index_key = "index_rows:" + str(manifest_path) + ":" + anchor.collection_manifest.sha256
    if index_key not in cache:
        index_lookup: dict[tuple[str, str], list[tuple[str, dict[str, Any]]]] = {}
        for name, record in manifest["shards"].items():
            guard("stock class index receipt")
            path = _child(root, manifest_path.parent / "shards", name + ".parquet", record["parquet_sha256"], files)
            if str(path) not in receipt_cache:
                receipt_cache[str(path)] = pd.read_parquet(path)
            part = receipt_cache[str(path)]
            found = part.loc[part.phase.eq("index") & part.state.eq("archived")]
            for candidate in found.to_dict("records"):
                index_lookup.setdefault((candidate["accession_number"], candidate["body_sha256"]), []).append((name, candidate))
        cache[index_key] = index_lookup
    index_rows = cache[index_key].get((row["accession_number"], row["index_body_sha256"]), [])
    if len(index_rows) != 1:
        raise DataReadinessError("stock class lacks one exact SEC detail-page receipt")
    index_shard, index_row = index_rows[0]
    _receipt(index_row)
    if (index_row["sec_cik"] != proof.sec_cik or index_row["status_code"] != 200
            or index_row["url"] != index_url(proof.sec_cik, str(row["accession_number"]))
            or index_row["final_url"] != index_row["url"] or index_row["redirect_chain_json"] != "[]"):
        raise DataReadinessError("stock class SEC detail-page transport differs")
    index_zip = _child(root, manifest_path.parent / "shards", index_shard + ".zip",
                       manifest["shards"][index_shard]["zip_sha256"], files)
    index = parse_filing_index(_zip_body(index_zip, index_row), accession=str(row["accession_number"]))
    comparable = {key: str(value) for key, value in item.items()}
    comparable["accepted_at_utc"] = accepted.isoformat()
    if index_rejection(index, comparable) is not None:
        raise DataReadinessError("stock class SEC detail page contradicts filing metadata")
    documents = selected_documents(index, str(item["primary_document"]))
    selected_row = parse_strict_json_object(str(row["index_row_json"]), label="SEC selected document row")
    primary = [document for document in documents if document.row() == selected_row
               and document.filename == item["primary_document"]]
    if len(primary) != 1:
        raise DataReadinessError("stock class is not the bound primary filing document")
    # EDGAR detail pages may link a joint filing's primary document under a
    # co-registrant folder. The shared parser enforces the exact accession and
    # numeric folder; its selected link, rather than folder/issuer equality,
    # determines the only permitted primary source URL.
    if row["url"] != "https://www.sec.gov" + primary[0].path:
        raise DataReadinessError("stock class source differs from the exact listed primary URL")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "body.htm"
        path.write_bytes(body)
        try:
            facts = extract_security_class_evidence(path, anchor.body_sha256, proof.sec_cik, proof.ticker)
        except SecurityClassIssuerMismatch:
            raise
        except DataReadinessError as exc:
            raise StockIdentityUnavailable("retained SEC class facts are unsupported: " + str(exc)) from exc
    if facts.security_title.value != proof.security_title:
        raise StockIdentityUnavailable("stock class title differs from the frozen proof")
    return facts, accepted, captured, str(zip_path)


def replay_stock_identity(root: Path, proof: StockIdentityProof, *, sessions: tuple[date, ...],
    source_files: MutableMapping[str, str], interpretation_sha256: str,
    cache: dict[str, Any] | None = None) -> ReplayedStockIdentity:
    """Replay exact source facts and unique open-to-close ownership, without inference.

    Caller owns the shared lease and rechecks ``source_files`` at publication.
    A caller-owned cache is bounded by the finite declared proof population.
    """
    root = root.resolve()
    key = str(root) + ":" + json_sha256(proof.model_dump(mode="json"))
    exact_cache_hit = cache is not None and cache.get("source_root") == str(root) and cache.get(key) is not None
    guard("stock identity replay", reclaim_unused_memory=not exact_cache_hit)
    if not sessions or tuple(sorted(set(sessions))) != sessions:
        raise DataReadinessError("stock identity requires exact ordered nonempty sessions")
    cache = {} if cache is None else cache
    if cache.setdefault("source_root", str(root)) != str(root):
        raise DataReadinessError("stock identity cache cannot cross source roots")
    cached = cache.get(key)
    if cached is None:
        files = cache.setdefault("source_files", {})
        frame_key = "frames:" + json_sha256([proof.identity_authority.model_dump(mode="json"),
                                            proof.membership_authority.model_dump(mode="json")])
        if frame_key not in cache:
            cache[frame_key] = _authority_frames(root, proof, files)
        relations, members = cache[frame_key]
        facts, accepted, captured, body_reference = _class_anchor(root, proof, files, cache)
        cached = relations, members, facts, accepted, captured, files, body_reference
        cache[key] = cached
    relations, members, facts, accepted, captured, files, body_reference = cached
    for name, digest in files.items():
        if name in source_files and source_files[name] != digest:
            raise DataReadinessError("stock identity cached source conflicts")
        source_files[name] = digest
    matching = relations.loc[relations.security_id.eq(proof.security_id) & relations.ticker.eq(proof.ticker)]
    if matching.empty or set(matching.sec_cik) != {proof.sec_cik}:
        raise StockIdentityUnavailable("stock class issuer differs from exact retained security/ticker relation")
    schedule = xcals.get_calendar("XNYS").schedule.reindex(pd.DatetimeIndex(sessions))
    if schedule.open.isna().any():
        raise DataReadinessError("stock identity requested a nonexchange session")
    proven = []
    for day, opening, closing in zip(sessions, schedule.open, schedule.close, strict=True):
        owned = True
        for frame in (relations,):
            same_ticker = frame.loc[frame.ticker.eq(proof.ticker)]
            overlapping = same_ticker.loc[same_ticker.effective_from_utc.le(closing)
                & (same_ticker.effective_to_utc.isna() | same_ticker.effective_to_utc.gt(opening))]
            exact = overlapping.loc[overlapping.security_id.eq(proof.security_id)
                & overlapping.effective_from_utc.le(opening) & overlapping.effective_to_utc.notna()
                & overlapping.effective_to_utc.gt(closing)]
            owned = owned and len(overlapping) == len(exact) == 1
        # Membership is a competing-owner constraint. It does not create a sale
        # or end an independently established security relation at index exit.
        competition = members.loc[members.ticker.eq(proof.ticker) & members.security_id.ne(proof.security_id)
            & members.effective_from_utc.le(closing)
            & (members.effective_to_utc.isna() | members.effective_to_utc.gt(opening))]
        owned = owned and competition.empty
        if owned:
            proven.append(day)
    competing = tuple(sorted(set(relations.loc[relations.sec_cik.eq(proof.sec_cik), "ticker"]) - {proof.ticker}))
    evidence_names = {str(inside(root, pin.path)) for pin in (proof.identity_authority, proof.membership_authority,
                      proof.anchor.collection_manifest, proof.anchor.receipt_artifact, proof.anchor.filing_metadata)}
    verified_at = datetime.now(UTC)
    references = tuple(EvidenceReference(reference=name, artifact_sha256=digest,
        interpretation_policy_sha256=interpretation_sha256,
        record_locator=f"reconstructed stock class and exact interval; security_id={proof.security_id};ticker={proof.ticker}",
        retrieved_at=verified_at, available_at=verified_at)
        for name, digest in sorted(files.items()) if name in evidence_names)
    references = (*references, EvidenceReference(reference=body_reference, artifact_sha256=files[body_reference],
        interpretation_policy_sha256=interpretation_sha256,
        record_locator=f"actual SEC primary cover;unit={proof.anchor.unit_id};body_sha256={proof.anchor.body_sha256};"
                       f"ticker={proof.ticker};class={proof.security_title}",
        retrieved_at=captured.to_pydatetime(), available_at=captured.to_pydatetime()))
    missing = tuple(f"stock_identity_session_unproven:{day}" for day in sessions if day not in proven)
    return ReplayedStockIdentity(proof.security_id, proof.ticker, proof.sec_cik, proof.security_title,
        facts.exchange.value, facts.issuer_name.value, references, tuple(proven), missing, competing,
        accepted.to_pydatetime(), captured.to_pydatetime(), verified_at)
