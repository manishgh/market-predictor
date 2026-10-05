"""Small immutable archive fixtures; no provider access or retained archive scans."""
from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings import document_collection as producer
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.research import issuer_content_review_sources as owner
from market_predictor.swing.contracts.holding_materialization import SourcePin

ACCEPTED = pd.Timestamp("2020-01-02T21:30:00Z")
ACCESSION = "0000000001-20-000001"


def sha(body: bytes) -> str:
    return hashlib.sha256(body).hexdigest()


def save(root: Path, name: str, value: dict[str, Any]) -> SourcePin:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, default=str), encoding="utf-8")
    return SourcePin(path=name, sha256=file_sha256(path))


def parquet(root: Path, name: str, rows: list[dict[str, Any]]) -> str:
    path = root / name
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(path, index=False)
    return file_sha256(path)


def original_hash(record: dict[str, Any]) -> str:
    return sha(json.dumps(record, ensure_ascii=True, sort_keys=True, default=str).encode())


@pytest.fixture
def alpaca_archive(tmp_path: Path) -> dict[str, Any]:
    request = save(tmp_path, "data/research/news/_request.json", {"window": {
        "start_utc": owner.START.isoformat(), "cutoff_utc": owner.END.isoformat()}})
    records = []
    for index, text in enumerate(("Acme reports Q1 2020 earnings.", "Acme reports revised Q1 2020 earnings.")):
        records.append({"id": "story", "headline": text, "created_at": "2020-01-02T21:30:00Z",
            "updated_at": f"2020-01-02T21:3{index}:00Z", "content": "", "summary": ""})
    page = {"chunk_id": "chunk", "news": records, "collected_at_utc": "2026-01-01T00:00:00Z"}
    page["content_sha256"] = json_sha256(page)
    page_pin = save(tmp_path, "data/raw/news/raw_pages/chunk/page_000000.json", page)
    rows = [{"query_security_id": "query", "query_ticker": "ACME", "chunk_id": "chunk",
        "raw_record_locators_json": json.dumps([{"page_path": page_pin.path, "page_sha256": page_pin.sha256,
                                                "news_index": index}]),
        "source_version_sha256": original_hash(record), "raw_sha256": original_hash(record),
        "published_at_utc": pd.Timestamp(record["created_at"]), "provider_updated_at_utc": pd.Timestamp(record["updated_at"]),
        "first_seen_at_utc": pd.Timestamp(page["collected_at_utc"]), "chosen_field": "headline",
        "chosen_field_sha256": sha(record["headline"].encode()), "inventory_status": "included",
        "query_identity_resolution": "identity_equal", "cohort_security_id": "cik:0000000001",
        "provider_story_id": "story"} for index, record in enumerate(records)]
    row_pin = parquet(tmp_path, "data/research/news/parts/early-0000.parquet", rows)
    units = save(tmp_path, "data/research/news/parts/early-0000.units.json", {"request_sha256": request.sha256,
        "units": [{"chunk_id": "chunk", "query_security_id": "query", "query_ticker": "ACME",
                   "source_files": {page_pin.path: page_pin.sha256}}]})
    manifest = {"schema": "market_predictor.issuer_content_cohort_inventory", "status": "complete_inventory_only",
        "request_sha256": request.sha256, "window": json.loads((tmp_path / request.path).read_text())["window"],
        "records_rows": 2, "parts": {"early-0000": {"records_sha256": row_pin, "units_sha256": units.sha256}}}
    return {"root": tmp_path, "inventory": save(tmp_path, "data/research/news/_manifest.json", manifest),
            "pins": {}, "memory_check": lambda: None}


def test_alpaca_preserves_original_versions_bytes_and_locator_pins(alpaca_archive: dict[str, Any]) -> None:
    rows = list(owner.iter_alpaca_review_sources(**alpaca_archive))
    assert len(rows) == 2 and {row.source_id for row in rows} == {"story"}
    assert len({row.source_version_sha256 for row in rows}) == 2
    assert rows[1].version_available_at_utc > rows[0].version_available_at_utc
    assert all(row.security_id == "cik:0000000001" and row.unavailable_reasons == () for row in rows)
    assert all(original_hash(row.payload) == row.source_version_sha256 for row in rows)
    assert any("raw_pages" in name for name in alpaca_archive["pins"])


def test_alpaca_changed_page_during_iteration_is_rejected(alpaca_archive: dict[str, Any]) -> None:
    iterator = owner.iter_alpaca_review_sources(**alpaca_archive)
    next(iterator)
    path = alpaca_archive["root"] / "data/raw/news/raw_pages/chunk/page_000000.json"
    path.write_bytes(path.read_bytes() + b" ")
    with pytest.raises(DataReadinessError, match="source changed"):
        list(iterator)


def test_alternating_record_locators_cache_each_page_once_per_chunk(
    alpaca_archive: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    root = alpaca_archive["root"]
    page_path = "data/raw/news/raw_pages/chunk/page_000000.json"
    second = save(root, page_path.replace("000000", "000001"), json.loads((root / page_path).read_text()))
    part_path = root / "data/research/news/parts/early-0000.parquet"
    frame = pd.read_parquet(part_path)
    frame.loc[1, "raw_record_locators_json"] = json.dumps([
        {"page_path": second.path, "page_sha256": second.sha256, "news_index": 1}])
    frame = pd.concat([frame, frame.iloc[:1]], ignore_index=True)
    frame.to_parquet(part_path, index=False)
    unit_path = "data/research/news/parts/early-0000.units.json"
    units = json.loads((root / unit_path).read_text())
    units["units"][0]["source_files"][second.path] = second.sha256
    unit_pin = save(root, unit_path, units)
    manifest = json.loads((root / alpaca_archive["inventory"].path).read_text())
    manifest["records_rows"] = 3
    manifest["parts"]["early-0000"] = {"records_sha256": file_sha256(part_path), "units_sha256": unit_pin.sha256}
    alpaca_archive["inventory"] = save(root, alpaca_archive["inventory"].path, manifest)
    original = owner._json
    reads: list[str] = []
    def counted(*args: Any, **kwargs: Any) -> dict[str, Any]:
        path = Path(args[1])
        if path.name.startswith("page_"):
            reads.append(path.name)
        return original(*args, **kwargs)
    monkeypatch.setattr(owner, "_json", counted)
    assert len(list(owner.iter_alpaca_review_sources(**alpaca_archive))) == 3
    assert reads == ["page_000000.json", "page_000001.json"]


def test_alpaca_unmapped_and_after_cutoff_rows_remain_explicit(alpaca_archive: dict[str, Any]) -> None:
    root = alpaca_archive["root"]
    path = root / "data/research/news/parts/early-0000.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "query_identity_resolution"] = "no_proven_identity"
    frame.loc[0, "cohort_security_id"] = None
    frame.loc[1, "inventory_status"] = "version_after_cutoff"
    frame.to_parquet(path, index=False)
    manifest = json.loads((root / alpaca_archive["inventory"].path).read_text())
    manifest["parts"]["early-0000"]["records_sha256"] = file_sha256(path)
    alpaca_archive["inventory"] = save(root, alpaca_archive["inventory"].path, manifest)
    rows = list(owner.iter_alpaca_review_sources(**alpaca_archive))
    assert "missing_cohort_identity" in rows[0].unavailable_reasons
    assert rows[1].unavailable_reasons == ("version_after_cutoff",)


def index_html() -> bytes:
    return f"""<div id="secNum">Accession No. {ACCESSION}</div>
    <div class="infoHead">Filing Date</div><div class="info">2020-01-02</div>
    <div class="infoHead">Accepted</div><div class="info">2020-01-02 16:30:00</div>
    <div class="companyInfo"><span class="companyName">Acme, Inc. (Filer) CIK: <a>0000000001</a></span></div>
    <div class="companyInfo"><span class="companyName">Beta Corporation (Filer) CIK: <a>0000000002</a></span></div>""".encode()


@pytest.fixture
def sec_archive(tmp_path: Path) -> dict[str, Any]:
    root = tmp_path
    clock = save(root, "data/research/sec_acceptance_clock/_manifest.json", {"scope": "corrected"})
    relation_name = "data/canonical/identities/sec_identity_relations.parquet"
    relation_sha = parquet(root, relation_name, [{"security_id": "cik:0000000001", "ticker": "ACME", "sec_cik": "0000000001",
        "effective_from_utc": owner.START, "effective_to_utc": None, "available_at_utc": owner.START}])
    filings_sha = parquet(root, "data/research/sec_inventory/filings.parquet", [{"security_id": "cik:0000000001",
        "sec_cik": "0000000001", "accession_number": ACCESSION, "sec_form": "8-K", "item_codes": "2.02",
        "accepted_at_utc": ACCEPTED, "available_at_utc": ACCEPTED}])
    inventory = save(root, "data/research/sec_inventory/_manifest.json", {
        "schema": "market_predictor.sec_form_inventory", "status": "complete", "mode": "initial_fit",
        "window": {"start_utc": owner.START.isoformat(), "end_utc": owner.END.isoformat()},
        "source_files": {clock.path: clock.sha256, relation_name: relation_sha},
        "artifacts": {"filings.parquet": filings_sha}})
    request = {"inventory_mode": "initial_fit", "sealed_until_rules_frozen": False,
        "inventory": inventory.model_dump(mode="json"), "pilot_accessions": [], "work_list_units": 1}
    request["request_sha256"] = json_sha256(request)
    save(root, "data/raw/documents/_request.json", request)
    shards = {}
    for number, items in enumerate(([('index', 'index', 'archived', index_html())],
        [('document', '1', 'archived', b'<p>Acme reports Q1 2020 earnings.</p>'),
         ('document', '2', 'archived', b'%PDF-1.7'), ('document', '3', 'oversize', None),
         ('document', '4', 'missing', b'Not found')])):
        name = f"shard-{number:05d}"
        folder = root / "data/raw/documents/shards"
        folder.mkdir(parents=True, exist_ok=True)
        rows = []
        with zipfile.ZipFile(folder / f"{name}.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for phase, sequence, state, body in items:
                values = {"unit_id": f"{ACCESSION}/{sequence}", "phase": phase, "sec_cik": "0000000001",
                    "accession_number": ACCESSION, "sequence": sequence, "document_type": "EX-99.1",
                    "attempt": 1, "state": state, "retrieved_at_utc": "2026-01-01T00:00:00Z",
                    "content_type": "application/pdf" if sequence == "2" else "text/html", "content_encoding": None}
                if body is not None:
                    member = f"{ACCESSION}/{sequence}/{sha(body)}"
                    values.update(member=member, body_sha256=sha(body), body_length=len(body))
                    archive.writestr(member, body)
                rows.append({**producer._receipt(values), "shard": name})
        digest = parquet(root, f"data/raw/documents/shards/{name}.parquet", rows)
        shards[name] = {"parquet_sha256": digest, "zip_sha256": file_sha256(folder / f"{name}.zip"), "attempts": len(rows)}
    checkpoint = save(root, "data/raw/documents/_checkpoint.json", {"request_sha256": request["request_sha256"], "shards": shards})
    manifest = {"schema": "market_predictor.sec_filing_document_collection", "status": "complete",
        "inventory_mode": "initial_fit", "sealed_until_rules_frozen": False, "request_sha256": request["request_sha256"],
        "checkpoint_sha256": checkpoint.sha256, "shards": shards,
        "totals": {"units_by_phase_and_state": {"index/archived": 1, "document/archived": 2, "document/oversize": 1,
                                               "document/missing": 1}}}
    return {"root": root, "archive": save(root, "data/raw/documents/_manifest.json", manifest),
            "inventory": inventory, "pins": {}, "memory_check": lambda: None}


def test_sec_documents_keep_all_unavailable_units_and_clock_authority(sec_archive: dict[str, Any]) -> None:
    rows = list(owner.iter_sec_review_sources(**sec_archive))
    assert len(rows) == 4
    assert isinstance(rows[0].payload, bytes) and rows[0].unavailable_reasons == ()
    assert "pdf_requires_independent_text_extraction" in rows[1].unavailable_reasons
    assert rows[2].unavailable_reasons == ("sec_oversize",)
    assert rows[3].unavailable_reasons == ("sec_missing",)
    for row in rows:
        assert row.security_id == "cik:0000000001" and row.ticker == "ACME"
        assert row.version_available_at_utc == ACCEPTED
        assert row.metadata["acceptance_clock_authority_sha256"]
        assert row.metadata["inventory_sha256"] == sec_archive["inventory"].sha256


def test_index_pass_never_reads_unused_document_zip_or_opens_mutating_store(
    sec_archive: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original = owner.file_sha256
    def guarded(path: Path) -> str:
        assert path.name != "shard-00001.zip"
        return original(path)
    monkeypatch.setattr(owner, "file_sha256", guarded)
    monkeypatch.setattr(producer.Store, "open", lambda *args, **kwargs: pytest.fail("mutating Store.open used"))
    [row] = list(owner.iter_sec_review_sources(**sec_archive, phase="index"))
    assert row.payload == index_html()


@pytest.mark.parametrize("effective_delay, known_delay, expected_ticker", [(2, 2, "ACME"), (0, 2, "ACME"), (0, 6, None)])
def test_sec_identity_resolves_at_filing_availability_without_backdating(
    sec_archive: dict[str, Any], effective_delay: int, known_delay: int, expected_ticker: str | None,
) -> None:
    root = sec_archive["root"]
    inventory = json.loads((root / sec_archive["inventory"].path).read_text())
    relation_name = "data/canonical/identities/sec_identity_relations.parquet"
    known = ACCEPTED + pd.Timedelta(minutes=known_delay)
    inventory["source_files"][relation_name] = parquet(root, relation_name, [{
        "security_id": "cik:0000000001", "ticker": "ACME", "sec_cik": "0000000001",
        "effective_from_utc": ACCEPTED + pd.Timedelta(minutes=effective_delay), "effective_to_utc": None,
        "available_at_utc": known}])
    available = ACCEPTED + pd.Timedelta(minutes=5)
    inventory["artifacts"]["filings.parquet"] = parquet(root, "data/research/sec_inventory/filings.parquet", [{
        "security_id": "cik:0000000001", "sec_cik": "0000000001", "accession_number": ACCESSION,
        "sec_form": "8-K", "item_codes": "2.02", "accepted_at_utc": ACCEPTED, "available_at_utc": available}])
    sec_archive["inventory"] = save(root, sec_archive["inventory"].path, inventory)
    request_path = "data/raw/documents/_request.json"
    request = json.loads((root / request_path).read_text())
    request["inventory"] = sec_archive["inventory"].model_dump(mode="json")
    request["request_sha256"] = json_sha256({key: value for key, value in request.items() if key != "request_sha256"})
    save(root, request_path, request)
    checkpoint_path = "data/raw/documents/_checkpoint.json"
    checkpoint = json.loads((root / checkpoint_path).read_text())
    checkpoint["request_sha256"] = request["request_sha256"]
    checkpoint_pin = save(root, checkpoint_path, checkpoint)
    manifest = json.loads((root / sec_archive["archive"].path).read_text())
    manifest.update(request_sha256=request["request_sha256"], checkpoint_sha256=checkpoint_pin.sha256)
    sec_archive["archive"] = save(root, sec_archive["archive"].path, manifest)
    rows = list(owner.iter_sec_review_sources(**sec_archive))
    assert len(rows) == 4
    for row in rows:
        assert row.ticker == expected_ticker
        assert row.published_at_utc == ACCEPTED
        assert row.version_available_at_utc == available
        assert row.metadata["identity_available_at_utc"] == (known if expected_ticker else None)
        assert row.metadata["identity_authority_sha256"] == inventory["source_files"][relation_name]
    if expected_ticker is None:
        assert "missing_ticker_identity" in rows[0].unavailable_reasons
        assert isinstance(rows[0].payload, bytes)


@pytest.mark.parametrize("target", ["archive", "inventory"])
def test_sealed_metadata_refused_before_parquet_or_zip_access(sec_archive: dict[str, Any], target: str,
                                                            monkeypatch: pytest.MonkeyPatch) -> None:
    root = sec_archive["root"]
    pin = sec_archive[target]
    value = json.loads((root / pin.path).read_text())
    value["sealed_until_rules_frozen"] = True
    if target == "inventory":
        value["mode"] = "later_sealed"
    sec_archive[target] = save(root, pin.path, value)
    monkeypatch.setattr(owner, "_rows", lambda *args, **kwargs: pytest.fail("sealed parquet read"))
    monkeypatch.setattr(owner.zipfile, "ZipFile", lambda *args, **kwargs: pytest.fail("sealed ZIP opened"))
    with pytest.raises(DataReadinessError):
        list(owner.iter_sec_review_sources(**sec_archive))


def test_sec_body_byte_limit_and_mutation_fail_before_source_publication(sec_archive: dict[str, Any],
                                                                      monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "MAX_BODY_BYTES", 5)
    with pytest.raises(DataReadinessError, match="body bound"):
        list(owner.iter_sec_review_sources(**sec_archive))


def test_sec_final_source_recheck_detects_changed_receipt_shard(sec_archive: dict[str, Any]) -> None:
    iterator = owner.iter_sec_review_sources(**sec_archive)
    next(iterator)
    path = sec_archive["root"] / "data/raw/documents/shards/shard-00001.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="source changed"):
        list(iterator)


def test_index_company_aliases_never_cross_filer_or_backdate() -> None:
    body = index_html()
    kwargs = {"body": body, "body_sha256": sha(body), "accession": ACCESSION,
        "accepted_at_utc": ACCEPTED, "clock_authority_sha256": "a" * 64, "locator": "shard.zip!index"}
    one = owner.extract_sec_alias_proofs(**kwargs, expected_cik="1")
    two = owner.extract_sec_alias_proofs(**kwargs, expected_cik="2")
    assert one.proofs[0].aliases == ("Acme, Inc.", "Acme")
    assert two.proofs[0].aliases == ("Beta Corporation", "Beta")
    assert all(proof.available_at_utc == ACCEPTED for proof in (*one.proofs, *two.proofs))
    missing = owner.extract_sec_alias_proofs(**kwargs, expected_cik="3")
    assert missing.proofs == () and missing.unavailable_reasons == ("missing_exact_filer_company_name",)
    kwargs["accepted_at_utc"] = ACCEPTED - pd.Timedelta(seconds=1)
    with pytest.raises(DataReadinessError, match="acceptance"):
        owner.extract_sec_alias_proofs(**kwargs, expected_cik="1")


def test_alias_conflicting_same_cik_names_stay_unavailable() -> None:
    body = index_html().replace(b'0000000002', b'0000000001')
    result = owner.extract_sec_alias_proofs(body=body, body_sha256=sha(body), accession=ACCESSION,
        expected_cik="1", accepted_at_utc=ACCEPTED, clock_authority_sha256="a" * 64, locator="index")
    assert result.proofs == () and result.unavailable_reasons == ("conflicting_filer_company_names",)
