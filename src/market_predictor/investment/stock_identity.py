"""Reconstructed research stock identity from exact retained SEC class evidence.

Cover facts establish a class anchor; only the accepted effective intervals own
sessions. Index exit, common CIKs and provider captures never extend those bounds.
"""
from __future__ import annotations

import tempfile
from collections.abc import MutableMapping
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
from pydantic import Field

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.investment.sec_document_evidence import (
    SecArchivedDocument,
    pin_retained_child,
    pin_retained_source,
    read_retained_sec_document,
    retained_source_object,
)
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.universe.security_class_evidence import (
    SecurityClassEvidence,
    SecurityClassIssuerMismatch,
    extract_security_class_and_jurisdiction,
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
    issuer_incorporation_code: str | None = None


def _authority_frames(root: Path, proof: StockIdentityProof, files: MutableMapping[str, str]
                      ) -> tuple[pd.DataFrame, pd.DataFrame]:
    authority_path = pin_retained_source(root, proof.identity_authority, files)
    authority = retained_source_object(authority_path)
    if (authority.get("schema") != "edge_rebuild.sec_identity_authority.v2"
            or authority.get("state") != "identity_complete" or authority.get("artifact") != "_manifest.json"):
        raise DataReadinessError("stock identity requires the exact retained accepted SEC authority")
    directory = authority_path.parent
    manifest = retained_source_object(pin_retained_child(root, directory, "_manifest.json", authority["artifact_sha256"], files))
    request_path = directory / "_request.json"
    request = retained_source_object(request_path)
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
    relations = pd.read_parquet(pin_retained_child(root, directory, relation["path"], relation["sha256"], files))
    membership_path = pin_retained_source(root, proof.membership_authority, files)
    membership_authority = retained_source_object(membership_path)
    if (membership_authority.get("state") != "membership_complete"
            or membership_authority.get("artifact") != "_manifest.json"):
        raise DataReadinessError("stock membership authority is incomplete")
    member_manifest = retained_source_object(pin_retained_child(root, membership_path.parent, "_manifest.json",
                                    membership_authority["artifact_sha256"], files))
    member_artifact = member_manifest["membership_artifact"]
    members = pd.read_parquet(pin_retained_child(root, membership_path.parent, member_artifact["path"],
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


def _proof_cache_key(root: Path, proof: StockIdentityProof) -> str:
    return str(root) + ":" + json_sha256(proof.model_dump(mode="json"))


def _class_anchor(root: Path, proof: StockIdentityProof, files: MutableMapping[str, str], cache: dict[str, Any]
                  ) -> tuple[SecurityClassEvidence, pd.Timestamp, pd.Timestamp, str]:
    anchor = proof.anchor
    document = read_retained_sec_document(root, SecArchivedDocument.model_validate(anchor.model_dump()), proof.sec_cik,
        files, cache, primary_only=True, memory_guard=lambda: guard("stock class index receipt"))
    if document.accepted_at >= pd.Timestamp("2024-05-29T00:00:00Z"):
        raise DataReadinessError("stock class anchor escapes the frozen pre-boundary scope")
    with tempfile.TemporaryDirectory() as directory:
        path = Path(directory) / "body.htm"
        path.write_bytes(document.body)
        try:
            facts, incorporation = extract_security_class_and_jurisdiction(path, anchor.body_sha256, proof.sec_cik, proof.ticker)
        except SecurityClassIssuerMismatch:
            raise
        except DataReadinessError as exc:
            raise StockIdentityUnavailable("retained SEC class facts are unsupported: " + str(exc)) from exc
    if facts.security_title.value != proof.security_title:
        raise StockIdentityUnavailable("stock class title differs from the frozen proof")
    cache.setdefault("issuer_incorporation_codes", {})[_proof_cache_key(root, proof)] = incorporation
    return facts, document.accepted_at, document.captured_at, str(document.body_path)


def replay_stock_identity(root: Path, proof: StockIdentityProof, *, sessions: tuple[date, ...],
    source_files: MutableMapping[str, str], interpretation_sha256: str,
    cache: dict[str, Any] | None = None) -> ReplayedStockIdentity:
    """Replay exact source facts and unique open-to-close ownership, without inference.

    Caller owns the shared lease and rechecks ``source_files`` at publication.
    A caller-owned cache is bounded by the finite declared proof population.
    """
    root = root.resolve()
    key = _proof_cache_key(root, proof)
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
        accepted.to_pydatetime(), captured.to_pydatetime(), verified_at,
        cache.get("issuer_incorporation_codes", {}).get(key))
