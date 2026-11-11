"""Synthetic retained envelopes exercise identity bounds without price outcomes."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment import stock_identity as identity
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_security_class_evidence import _context, _document, _extract, _issuer, _row


def _json(path: Path, value: dict[str, Any]) -> SourcePin:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value), encoding="utf-8")
    return SourcePin(path=str(path), sha256=file_sha256(path))


@pytest.fixture
def scope(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    monkeypatch.setattr(identity, "guard", lambda *args, **kwargs: None)
    relation = {"security_id": "issuer:class-a", "ticker": "TEST", "sec_cik": "0001040971",
        "effective_from_utc": pd.Timestamp("2019-07-09T04:00:00Z"),
        "effective_to_utc": pd.Timestamp("2024-06-03T04:00:00Z")}
    relations = pd.DataFrame([relation])
    members = relations.drop(columns="sec_cik").copy()
    folder = tmp_path / "member"
    folder.mkdir()
    members.to_parquet(folder / "memberships.parquet", index=False)
    member_manifest = _json(folder / "_manifest.json", {"membership_artifact": {
        "path": "memberships.parquet", "sha256": file_sha256(folder / "memberships.parquet")}})
    member = _json(folder / "_authority.json", {"state": "membership_complete", "artifact": "_manifest.json",
        "artifact_sha256": member_manifest.sha256})
    sec = tmp_path / "sec"
    sec.mkdir()
    relations.to_parquet(sec / "relations.parquet", index=False)
    parent = {"membership_authority": {"authority_sha256": member.sha256}}
    request = {"schema": "edge_rebuild.sec_identity_request.v2", "parent_lineage": parent}
    request_sha = json_sha256(request)
    _json(sec / "_request.json", {**request, "request_sha256": request_sha})
    manifest = _json(sec / "_manifest.json", {"schema": "edge_rebuild.sec_identity_manifest.v2",
        "status": "complete", "request_sha256": request_sha, "parent_lineage": parent,
        "relation_artifact": {"path": "relations.parquet", "sha256": file_sha256(sec / "relations.parquet")}})
    authority = _json(sec / "_authority.json", {"schema": "edge_rebuild.sec_identity_authority.v2",
        "state": "identity_complete", "artifact": "_manifest.json", "artifact_sha256": manifest.sha256,
        "request_sha256": request_sha})
    dummy = SourcePin(path="synthetic-class-anchor.json", sha256="a" * 64)
    proof = identity.StockIdentityProof(security_id=relation["security_id"], ticker="TEST", sec_cik="0001040971",
        security_title="Common Stock, $0.01 par value", identity_authority=authority, membership_authority=member,
        anchor=identity.SecClassAnchor(collection_manifest=dummy, receipt_artifact=dummy, unit_id="class",
            body_sha256="b" * 64, filing_metadata=dummy))
    facts = _extract(tmp_path, _document(_context("base"), _issuer("base"), _row("base", ticker="TEST")), ticker="TEST")
    calls: list[str] = []

    def anchor(*args: Any):
        calls.append("parse")
        source = tmp_path / "synthetic.zip"
        source.write_bytes(b"explicit synthetic class source")
        args[2][str(source)] = file_sha256(source)
        return facts, pd.Timestamp("2024-05-01T12:00:00Z"), pd.Timestamp("2026-11-09T10:00:00Z"), str(source)

    monkeypatch.setattr(identity, "_class_anchor", anchor)
    return {"root": tmp_path, "proof": proof, "relations": relations, "members": members, "calls": calls}


def replay(scope: dict[str, Any], *days: str, **kwargs: Any):
    return identity.replay_stock_identity(scope["root"], scope["proof"],
        sessions=tuple(date.fromisoformat(day) for day in days), source_files={},
        interpretation_sha256="c" * 64, **kwargs)


def test_anchor_is_not_a_lifetime_start_and_clocks_remain_current(scope):
    result = replay(scope, "2019-07-10", "2024-05-31", "2024-06-03")
    assert result.proven_sessions == (date(2019, 7, 10), date(2024, 5, 31))
    assert result.missing_reasons == ("stock_identity_session_unproven:2024-06-03",)
    assert result.anchor_accepted_at_utc.year == 2024
    assert result.anchor_retrieved_at_utc.year == 2026
    assert result.verified_at_utc >= result.anchor_retrieved_at_utc
    assert any(ref.retrieved_at == result.anchor_retrieved_at_utc for ref in result.evidence)
    assert any(ref.retrieved_at == result.verified_at_utc for ref in result.evidence)


def test_absent_index_membership_does_not_end_positive_relation(scope, monkeypatch):
    relations = scope["relations"]
    members = scope["members"].iloc[:0]
    monkeypatch.setattr(identity, "_authority_frames", lambda *args: (relations, members))
    assert replay(scope, "2024-05-31").missing_reasons == ()


@pytest.mark.parametrize("owner_kind", ["relation", "membership"])
def test_excluded_or_other_competing_owner_cannot_disappear(scope, monkeypatch, owner_kind):
    relations, members = scope["relations"].copy(), scope["members"].copy()
    target = relations if owner_kind == "relation" else members
    other = target.iloc[0].to_dict()
    other["security_id"] = "excluded:other-class"
    target.loc[len(target)] = other
    monkeypatch.setattr(identity, "_authority_frames", lambda *args: (relations, members))
    assert not replay(scope, "2024-05-31").proven_sessions


def test_partial_session_owner_is_not_a_full_session_owner(scope, monkeypatch):
    relations = scope["relations"].copy()
    relations.loc[0, "effective_to_utc"] = pd.Timestamp("2024-05-31T16:00:00Z")
    monkeypatch.setattr(identity, "_authority_frames", lambda *args: (relations, scope["members"]))
    assert replay(scope, "2024-05-31").proven_sessions == ()


def test_cik_alone_cannot_resolve_another_share_class(scope, monkeypatch):
    relations = scope["relations"].copy()
    other = relations.iloc[0].to_dict()
    other.update(ticker="TESTB", security_id="issuer:class-b")
    relations.loc[1] = other
    monkeypatch.setattr(identity, "_authority_frames", lambda *args: (relations, scope["members"]))
    assert replay(scope, "2024-05-31").competing_class_tickers == ("TESTB",)
    scope["proof"] = scope["proof"].model_copy(update={"security_id": "issuer:class-b"})
    with pytest.raises(DataReadinessError, match="exact retained security/ticker"):
        replay(scope, "2024-05-31")


def test_caller_cache_reuses_class_parsing_but_checks_each_interval(scope):
    cache = {}
    assert not replay(scope, "2024-05-31", cache=cache).missing_reasons
    assert replay(scope, "2024-06-03", cache=cache).missing_reasons
    assert scope["calls"] == ["parse"]


def test_mutated_authority_file_is_rejected_before_source_facts(scope):
    Path(scope["proof"].identity_authority.path).write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="source hash differs"):
        replay(scope, "2024-05-31")
    assert not scope["calls"]


def _real_class_proof(tmp_path, monkeypatch, *, cofolder=False, body_cik="0000001800"):
    from market_predictor.catalysts.sec_filings import document_collection as documents
    from market_predictor.research import sec_page_collection as runner
    from tests.test_sec_filing_documents import ABT, ABT_CIK, _collect, _Edgar, _ok, _page, _scripts, _url

    monkeypatch.setattr(identity, "guard", lambda *args, **kwargs: None)
    monkeypatch.setattr(runner, "_guard", lambda: None)
    monkeypatch.setattr(documents, "RETRY_WAITS_SECONDS", (0.0, 0.0))
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)
    body = _document(_context("base", entity="1800"), _issuer("base", cik=body_cik), _row("base", ticker="ABT")).encode()
    scripts = _scripts()
    scripts[_url(ABT_CIK, ABT, "a19-12883_18k.htm")] = [_ok(body)]
    if cofolder:
        scripts[documents.index_url(ABT_CIK, ABT)] = [_ok(_page(ABT, ABT_CIK).replace(b"/data/1800/", b"/data/1801/"))]
        for filename in ("a19-12883_18k.htm", "a19-12883_1ex99d1.htm"):
            scripts[_url("0000001801", ABT, filename)] = scripts.pop(_url(ABT_CIK, ABT, filename))
    _collect(tmp_path, _Edgar(scripts), monkeypatch, name="class")
    folder = tmp_path / "data/raw/class"
    paths = sorted((folder / "shards").glob("*.parquet"))
    candidates = [(path, row) for path in paths for row in pd.read_parquet(path).to_dict("records")
                  if row["phase"] == "document" and row["accession_number"] == ABT and row["document_type"] == "8-K"]
    path, row = candidates[0]
    def pin(value):
        return SourcePin(path=str(value), sha256=file_sha256(value))
    proof = identity.StockIdentityProof(security_id="issuer:ABT", ticker="ABT", sec_cik=ABT_CIK,
        security_title="Common Stock, $0.01 par value", identity_authority=pin(folder / "_manifest.json"),
        membership_authority=pin(folder / "_manifest.json"), anchor=identity.SecClassAnchor(
            collection_manifest=pin(folder / "_manifest.json"), receipt_artifact=pin(path), unit_id=row["unit_id"],
            body_sha256=row["body_sha256"], filing_metadata=pin(tmp_path / "data/research/sec_inventory/filings.parquet")))
    return proof


@pytest.mark.parametrize("cofolder", [False, True])
def test_real_receipt_and_primary_body_chain_replays_and_rejects_body_pin(tmp_path, monkeypatch, cofolder):
    proof = _real_class_proof(tmp_path, monkeypatch, cofolder=cofolder)
    files: dict[str, str] = {}
    facts, accepted, captured, reference = identity._class_anchor(tmp_path, proof, files, {})
    assert facts.trading_symbol.value == "ABT" and accepted < captured and Path(reference).is_file()
    changed = proof.model_copy(update={"anchor": proof.anchor.model_copy(update={"body_sha256": "0" * 64})})
    with pytest.raises(DataReadinessError, match="document binding differs"):
        identity._class_anchor(tmp_path, changed, {}, {})


@pytest.mark.parametrize("poison", ["wrong_accession", "unlisted_filename", "unlisted_cofolder", "redirect", "selected_row"])
def test_primary_source_poison_is_fatal_even_with_consistent_synthetic_receipt_hashes(tmp_path, monkeypatch, poison):
    from market_predictor.catalysts.sec_filings import document_collection as documents

    proof = _real_class_proof(tmp_path, monkeypatch)
    path = Path(proof.anchor.receipt_artifact.path)
    receipts = pd.read_parquet(path)
    selected = receipts.index[receipts.unit_id.eq(proof.anchor.unit_id)][0]
    row = receipts.loc[selected].to_dict()
    if poison == "wrong_accession":
        row["url"] = row["url"].replace("000110465919040664", "000110465919040665")
        row["final_url"] = row["url"]
    elif poison == "unlisted_filename":
        row["url"] = row["url"].replace("a19-12883_18k.htm", "unlisted.htm")
        row["final_url"] = row["url"]
    elif poison == "unlisted_cofolder":
        row["url"] = row["url"].replace("/data/1800/", "/data/1802/")
        row["final_url"] = row["url"]
    elif poison == "redirect":
        row["final_url"] = row["url"] + "?redirected=1"
    else:
        selected_row = json.loads(row["index_row_json"])
        selected_row["path"] = selected_row["path"].replace("a19-12883_18k.htm", "unlisted.htm")
        row["index_row_json"] = json.dumps(selected_row, sort_keys=True)
    refreshed = documents._receipt(row)
    for field, value in refreshed.items():
        receipts.loc[selected, field] = value
    receipts.to_parquet(path, index=False)
    manifest_path = Path(proof.anchor.collection_manifest.path)
    manifest = json.loads(manifest_path.read_bytes())
    manifest["shards"][path.stem]["parquet_sha256"] = file_sha256(path)
    manifest_path.write_text(json.dumps(manifest), encoding="utf-8")
    anchor = proof.anchor.model_copy(update={"collection_manifest": SourcePin(path=str(manifest_path), sha256=file_sha256(manifest_path)),
        "receipt_artifact": SourcePin(path=str(path), sha256=file_sha256(path))})
    poisoned = proof.model_copy(update={"anchor": anchor})
    with pytest.raises(DataReadinessError) as exc:
        identity._class_anchor(tmp_path, poisoned, {}, {})
    assert not isinstance(exc.value, identity.StockIdentityUnavailable)


def test_explicit_body_issuer_mismatch_remains_fatal(tmp_path, monkeypatch):
    proof = _real_class_proof(tmp_path, monkeypatch, cofolder=True, body_cik="0000001802")
    with pytest.raises(identity.SecurityClassIssuerMismatch, match="explicit issuer CIK") as exc:
        identity._class_anchor(tmp_path, proof, {}, {})
    assert not isinstance(exc.value, identity.StockIdentityUnavailable)
