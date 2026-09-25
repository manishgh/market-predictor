"""SEC form inventory over archives written by the real SEC collector and official-document collector."""
from __future__ import annotations

import dataclasses
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings.collection import (
    load_sec_filing_collection,
    normalize_sec_identity_relations,
    replay_sec_filing_collection,
)
from market_predictor.catalysts.sec_filings.form_inventory import (
    SavedDocument,
    document_status,
    issuer_inventory,
    report_timing,
    saved_document,
    session_position,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import sec_form_inventory as inventory
from market_predictor.sources.http import HttpByteResponse
from market_predictor.sources.official_documents import (
    OfficialDocument,
    OfficialDocumentInventory,
    collect_official_documents,
)
from tests.support.research_population import write_population
from tests.support.sec_archive import filing, settings, submissions, write_collection

A, B, C, D = "0000000001", "0000000002", "0000000003", "0000000004"
COHORT = ("cik:0000000001", "cik:0000000002:ticker:BBA", "cik:0000000002:ticker:BBB", "cik:0000000003",
          "sp500-historical:nosec")
FORMS = ("8-K", "8-K/A", "10-Q", "4")
SAVED_URL = "https://www.sec.gov/Archives/edgar/data/1/000000000119000001/a-earnings.htm"


def _relation(security: str, ticker: str, cik: str, start: str = "2019-07-09T04:00:00Z", end: str | None = None,
              policy: str = "latest_sec_ticker_propagated_within_stable_security_id_v1") -> dict[str, Any]:
    return {"security_id": security, "ticker": ticker, "sec_cik": cik, "effective_from_utc": pd.Timestamp(start),
            "effective_to_utc": pd.Timestamp(end) if end else pd.NaT, "available_at_utc": pd.Timestamp(start),
            "identity_policy": policy}


def _relations() -> pd.DataFrame:
    frame = pd.DataFrame([
        _relation("cik:0000000001", "AAA", A),
        _relation("cik:0000000002:ticker:BBA", "BBA", B),
        _relation("cik:0000000002:ticker:BBB", "BBB", B, policy="reviewed_official_sec_filing_override_v1"),
        _relation("cik:0000000003", "CCC", C, start="2021-01-04T05:00:00Z"),
        _relation("cik:0000000004", "DDD", D),
    ])
    frame["effective_to_utc"] = pd.to_datetime(frame.effective_to_utc, utc=True)
    return frame


def _pages() -> dict[str, dict[str, Any]]:
    older = [filing("0000000001-19-000001", "8-K", "2019-08-01T20:05:00Z", items="2.02,9.01", report="2019-08-01",
                    primary="a-earnings.htm", size=5000),
             filing("0000000001-19-000002", "8-K", "2019-08-06T13:00:00Z", items="8.01", report="2019-08-02"),
             filing("0000000001-19-000003", "4", "2019-08-10T15:00:00Z"),
             filing("0000000001-19-000004", "SC 13G", "2019-09-10T15:00:00Z")]
    recent = [filing("0000000001-20-000005", "10-Q", "2020-11-02T15:00:00Z", report="2020-09-30"),
              filing("0000000001-24-000006", "8-K", "2024-05-28T21:55:00Z", items="7.01", report="2024-05-28"),
              filing("0000000001-25-000007", "8-K", "2025-02-03T21:05:00Z", items="2.02", report="2025-02-03")]
    pages = submissions(A, "ALPHA CORP", recent, older=(("CIK0000000001-submissions-001.json", older),))
    pages |= submissions(B, "BETA HOLDINGS", [filing("0000000002-20-000001", "8-K", "2020-03-02T21:10:00Z",
                                                     items="2.02", report="2020-03-02"),
                                              filing("0000000002-20-000002", "8-K/A", "2020-03-05T14:00:00Z",
                                                     items="2.02", report="2020-03-02")])
    pages |= submissions(C, "GAMMA INC", [filing("0000000003-20-000001", "8-K", "2020-06-01T14:00:00Z", items="1.01",
                                                 report="2020-05-29"),
                                          filing("0000000003-21-000002", "8-K", "2021-03-01T14:00:00Z", items="2.02",
                                                 report="2021-03-01", primary="c.htm")])
    pages |= submissions(D, "DELTA LTD", [filing("0000000004-20-000001", "8-K", "2020-03-02T14:00:00Z", items="8.01",
                                                 report="2020-03-02")])
    return pages


def _pin(root: Path, path: Path) -> dict[str, str]:
    return {"path": path.relative_to(root).as_posix(), "sha256": file_sha256(path)}


def _official(root: Path) -> Path:
    body = b"<html>earnings release</html>"

    def fetch(document: OfficialDocument, _: int) -> HttpByteResponse:
        return HttpByteResponse(body=body, requested_url=document.url, final_url=document.url, redirect_chain=(),
                                status_code=200, retrieved_at_utc=datetime.now(UTC),
                                content_type="text/html", content_encoding=None, etag=None, last_modified=None,
                                body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
                                body_representation="http_entity_encoded", safe_headers=(("content-type", "text/html"),))

    documents = OfficialDocumentInventory(schema_version="market_predictor.official_document_inventory",
        maximum_response_bytes=1024, attempts_per_document_per_run=1,
        documents=[OfficialDocument(document_id="alpha_earnings", url=SAVED_URL, purpose="Test-only saved 8-K primary document.",
                                    expected_media="html")])
    directory = root / "data/raw/test_documents"
    collect_official_documents(inventory=documents, output_directory=directory, fetch=fetch)
    return directory / "_request.json"


def _evidence(root: Path) -> Path:
    store = root / "data/raw/evidence"
    store.mkdir(parents=True)
    document = store / "CCC_0000000003-21-000002_c.htm"
    document.write_bytes(b"<html>gamma 8-K</html>")
    rows = [{"filing_url": "https://www.sec.gov/Archives/edgar/data/3/000000000321000002/c.htm",
             "filing_path": "data/raw/evidence/CCC_0000000003-21-000002_c.htm", "filing_sha256": file_sha256(document)},
            {"filing_url": "", "filing_path": "", "filing_sha256": ""}]
    path = store / "filing_proof_inventory.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def _world(root: Path) -> dict[str, Any]:
    root.mkdir(parents=True)
    relations = _relations()
    relation_path = root / "data/canonical/sec_identity/test/sec_identity_relations.parquet"
    relation_path.parent.mkdir(parents=True)
    relations.to_parquet(relation_path, index=False)
    collection = write_collection(root / "data/external/sec", _pages(), pd.read_parquet(relation_path), FORMS)
    identity = root / "data/research/identity/_manifest.json"
    identity.parent.mkdir(parents=True)
    identity.write_text(json.dumps({"schema": "market_predictor.issuer_news_identity_alignment",
                                    "source_files": {relation_path.relative_to(root).as_posix(): file_sha256(relation_path)}}),
                        encoding="utf-8")
    config = {"schema": inventory.CONFIG_SCHEMA, "sec_collection": _pin(root, collection.directory / "_authority.json"),
              "identity_manifest": _pin(root, identity), "sec_identity_relations": _pin(root, relation_path),
              "approved_population": _pin(root, write_population(root, COHORT)),
              "official_document_collections": [_pin(root, _official(root))],
              "identity_evidence_inventories": [_pin(root, _evidence(root))]}
    path = root / "configs/sec_inventory.json"
    path.parent.mkdir(parents=True)
    path.write_text(json.dumps(config), encoding="utf-8")
    return {"root": root, "config": path, "sha256": file_sha256(path), "settings": config, "collection": collection.directory}


@pytest.fixture(scope="module")
def world(tmp_path_factory: pytest.TempPathFactory) -> dict[str, Any]:
    return _world(tmp_path_factory.mktemp("sec-inventory") / "repo")


@pytest.fixture(autouse=True)
def _policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inventory, "_guard", lambda: None)
    monkeypatch.setattr(inventory, "Settings", settings)
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)


def _run(world: dict[str, Any], name: str, mode: inventory.Mode = "initial_fit", *, config: Path | None = None,
         sha256: str | None = None) -> dict[str, Any]:
    return inventory.publish_sec_form_inventory(root=world["root"], config=config or world["config"],
        config_sha256=sha256 or world["sha256"], output=world["root"] / "data/research" / name, mode=mode)


def test_replay_reproduces_each_issuer_and_detects_divergence(world: dict[str, Any]) -> None:
    collection = load_sec_filing_collection(world["collection"])
    replayed = list(replay_sec_filing_collection(collection, settings()))
    assert [issuer.cik for issuer in replayed] == [A, B, C, D]
    assert replayed[0].history.submission_files == ("CIK0000000001.json", "CIK0000000001-submissions-001.json")
    missing_event = dataclasses.replace(collection, events=collection.events.iloc[1:])
    with pytest.raises(DataReadinessError, match="differ from canonical events"):
        list(replay_sec_filing_collection(missing_event, settings()))
    page = collection.raw_inventory.requested_url.str.endswith("submissions-001.json")
    missing_page = dataclasses.replace(collection, raw_inventory=collection.raw_inventory.loc[~page])
    with pytest.raises(DataReadinessError, match="do not replay"):
        list(replay_sec_filing_collection(missing_page, settings()))
    changed = collection.source_collections.assign(response_sha256="0" * 64)
    with pytest.raises(DataReadinessError, match="differ from the saved inventory"):
        list(replay_sec_filing_collection(dataclasses.replace(collection, source_collections=changed), settings()))


def test_inventory_attribution_timing_items_and_documents(world: dict[str, Any]) -> None:
    report = _run(world, "complete")
    root = world["root"] / "data/research/complete"
    filings = pd.read_parquet(root / "filings.parquet").set_index(["security_id", "accession_number"])
    totals = report["totals"]
    earnings = filings.loc[("cik:0000000001", "0000000001-19-000001")]
    assert (earnings.item_codes, earnings.acceptance_session_position, earnings.report_timing,
            earnings.document_status, earnings.filing_size_bytes) == ("2.02,9.01", "after_close",
                                                                      "report_session_may_precede_sec",
                                                                      "primary_document_saved", 5000)
    lagged = filings.loc[("cik:0000000001", "0000000001-19-000002")]
    assert (lagged.report_timing, lagged.acceptance_lag_days, lagged.post_report_sessions_before_availability,
            lagged.report_session_opened_before_availability, lagged.acceptance_session_position) == (
        "post_report_sessions_precede_sec", 4, 1, True, "pre_open")
    amendment = filings.loc[("cik:0000000002:ticker:BBA", "0000000002-20-000002")]
    assert (amendment.report_timing, amendment.post_report_sessions_before_availability) == (
        "post_report_sessions_precede_sec", 2)
    assert filings.loc[("cik:0000000003", "0000000003-21-000002")].report_timing == "no_post_report_session_before_sec"
    ownership = filings.loc[("cik:0000000001", "0000000001-19-000003")]
    assert (ownership.acceptance_session_position, ownership.report_timing) == ("non_session", "not_applicable")
    assert pd.isna(ownership.acceptance_lag_days)
    assert filings.loc[("cik:0000000003", "0000000003-21-000002")].document_status == "saved_without_receipt"
    shared = filings.xs("0000000002-20-000001", level="accession_number")
    assert set(shared.index) == {"cik:0000000002:ticker:BBA", "cik:0000000002:ticker:BBB"}
    assert shared.issuer_cohort_securities.eq(2).all()
    assert set(shared.identity_policy) == {"latest_sec_ticker_propagated_within_stable_security_id_v1",
                                           "reviewed_official_sec_filing_override_v1"}
    assert "0000000004-20-000001" not in set(filings.index.get_level_values("accession_number"))
    assert "0000000003-20-000001" not in set(filings.index.get_level_values("accession_number"))
    assert totals["cik_identity_outside_relation_rows"] == 1
    assert totals["accepted_in_window_available_after"] == 1
    assert totals["unrequested_form_rows_in_window"] == {"SC 13G": 1}
    assert totals["cohort_securities_without_sec_identity"] == 1 and totals["share_class_duplicate_rows"] == 4
    assert totals["accessions"] == 7 and totals["filing_rows"] == 9
    assert totals["accessions_by_document_status"] == {"primary_document_saved": 1, "other_filing_document_saved": 0,
                                                       "saved_without_receipt": 1, "not_saved": 5}
    assert totals["current_report_accessions_by_item_and_timing"]["8.01/post_report_sessions_precede_sec"] == 1
    years = pd.read_parquet(root / "security_years.parquet").set_index(["security_id", "year_new_york"])
    gamma = years.loc["cik:0000000003"]
    assert gamma.at[2020, "sec_identity_days"] == 0 and gamma.at[2020, "cik_identity_outside_relation_filings"] == 1
    assert gamma.at[2021, "unknown_identity_days"] == pytest.approx(3.0)
    assert years.loc["sp500-historical:nosec"].sec_identity_days.eq(0).all()
    counts = pd.read_parquet(root / "security_year_counts.parquet")
    alpha = counts.loc[counts.security_id.eq("cik:0000000001") & counts.year_new_york.eq(2019)]
    assert set(zip(alpha.sec_form, alpha.item_code, alpha.report_timing, alpha.filings, strict=True)) == {
        ("8-K", "2.02", "report_session_may_precede_sec", 1), ("8-K", "9.01", "report_session_may_precede_sec", 1),
        ("8-K", "8.01", "post_report_sessions_precede_sec", 1), ("4", "not_applicable", "not_applicable", 1)}
    manifest = json.loads((root / "_manifest.json").read_text())
    assert manifest["mode"] == "initial_fit" and not manifest["training_eligible"]
    assert not any(Path(path).is_absolute() or "\\" in path for path in manifest["source_files"])
    assert "data/raw/evidence/CCC_0000000003-21-000002_c.htm" in manifest["source_files"]
    with pytest.raises(FileExistsError):
        _run(world, "complete")
    _run(world, "complete-again")
    for name in ("filings.parquet", "security_years.parquet", "security_year_counts.parquet"):
        pd.testing.assert_frame_equal(pd.read_parquet(root / name), pd.read_parquet(root.parent / "complete-again" / name))


def test_sealed_mode_lists_later_filings_without_statistics(world: dict[str, Any]) -> None:
    report = _run(world, "sealed", "later_sealed")
    root = world["root"] / "data/research/sealed"
    assert sorted(path.name for path in root.iterdir()) == ["_manifest.json", "filings.parquet"]
    filings = pd.read_parquet(root / "filings.parquet")
    assert set(filings.accession_number) == {"0000000001-24-000006", "0000000001-25-000007"}
    assert list(filings.columns) == list(inventory.SEALED_COLUMNS)
    assert report["sealed_until_rules_frozen"] is True and "totals" not in report and report["accessions"] == 2
    record = {"path": "data/research/sealed/_manifest.json", "sha256": report["manifest_sha256"]}
    with pytest.raises(DataReadinessError, match="sealed"):
        inventory.load_sec_form_inventory(world["root"], record, {})
    opened, _ = inventory.load_sec_form_inventory(world["root"], record, {}, allow_sealed_collection=True)
    assert len(opened) == 2


@pytest.mark.parametrize("target", ["configs/sec_inventory.json", "data/external/sec/_authority.json",
    "data/research/identity/_manifest.json", "data/canonical/sec_identity/test/sec_identity_relations.parquet",
    "data/reports/population.json", "data/raw/test_documents/_request.json", "data/raw/evidence/filing_proof_inventory.csv",
    "data/raw/evidence/CCC_0000000003-21-000002_c.htm"])
def test_every_pin_is_verified_before_any_output(world: dict[str, Any], target: str) -> None:
    path = world["root"] / target
    original = path.read_bytes()
    try:
        path.write_bytes(original + b" ")
        with pytest.raises(DataReadinessError):
            _run(world, f"tamper-{path.name}")
    finally:
        path.write_bytes(original)
    assert not (world["root"] / "data/research" / f"tamper-{path.name}").exists()


def test_relations_must_equal_the_bridge_pin_and_the_collection_request(world: dict[str, Any], tmp_path: Path) -> None:
    root, settings_ = world["root"], world["settings"]
    other = root / "data/canonical/sec_identity/other/sec_identity_relations.parquet"
    other.parent.mkdir(parents=True, exist_ok=True)
    _relations().iloc[:-1].to_parquet(other, index=False)
    for name, changes, message in (
            ("unbound", {"sec_identity_relations": _pin(root, other)}, "identity alignment manifest's"),
            ("extra", {**settings_, "unexpected": True}, "configuration differs")):
        path = root / f"configs/{name}.json"
        path.write_text(json.dumps({**settings_, **changes}), encoding="utf-8")
        with pytest.raises(DataReadinessError, match=message):
            _run(world, f"lineage-{name}", config=path, sha256=file_sha256(path))
    identity = json.loads((root / settings_["identity_manifest"]["path"]).read_text())
    identity["source_files"][other.relative_to(root).as_posix()] = file_sha256(other)
    manifest = root / "data/research/identity-other/_manifest.json"
    manifest.parent.mkdir(parents=True, exist_ok=True)
    manifest.write_text(json.dumps(identity), encoding="utf-8")
    path = root / "configs/other-relations.json"
    path.write_text(json.dumps({**settings_, "identity_manifest": _pin(root, manifest),
                                "sec_identity_relations": _pin(root, other)}), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="SEC collection request"):
        _run(world, "lineage-request", config=path, sha256=file_sha256(path))


def test_busy_lease_prevents_any_source_read(world: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(inventory, "load_sec_filing_collection", lambda *_: pytest.fail("source read without the lease"))
    with heavy_job_lease("test-owner", runtime_dir=world["root"] / "data/runtime"), pytest.raises(HeavyJobBusyError):
        _run(world, "busy")
    with pytest.raises(DataReadinessError, match="direct child"):
        inventory.publish_sec_form_inventory(root=world["root"], config=world["config"], config_sha256=world["sha256"],
                                             output=world["root"] / "data/research/nested/out", mode="initial_fit")


def test_window_boundaries_relation_clock_and_overlap(world: dict[str, Any]) -> None:
    collection = load_sec_filing_collection(world["collection"])
    alpha = next(replay_sec_filing_collection(collection, settings()))
    relations = normalize_sec_identity_relations(_relations()).assign(identity_policy="policy")
    available = pd.to_datetime(alpha.events.available_at_utc, utc=True).sort_values()
    first, last = available.iloc[0], available.iloc[2]
    kwargs: dict[str, Any] = {"requested": frozenset(FORMS), "cohort": frozenset(COHORT), "saved": {}}
    inclusive = issuer_inventory(alpha, relations=relations, window=(first, last), **kwargs)
    assert len(inclusive.filings) == 3
    exclusive = issuer_inventory(alpha, relations=relations, window=(first, last - pd.Timedelta(1, "ns")), **kwargs)
    assert len(exclusive.filings) == 2
    late = relations.assign(available_at_utc=relations.available_at_utc.where(relations.sec_cik.ne(A),
                                                                               pd.Timestamp("2019-08-05T00:00:00Z")))
    delayed = issuer_inventory(alpha, relations=late, window=(first, last), **kwargs)
    assert len(delayed.filings) == 2 and len(delayed.outside_relation) == 1
    overlap = pd.concat([relations, relations.loc[relations.sec_cik.eq(A)].assign(ticker="AAA2")])
    with pytest.raises(DataReadinessError, match="overlap"):
        issuer_inventory(alpha, relations=overlap, window=(first, last), **kwargs)


def test_session_position_and_report_timing_follow_the_exchange_calendar() -> None:
    assert session_position(pd.Timestamp("2019-11-29T18:30:00Z")) == "after_close"  # Half day closes 13:00 ET.
    assert session_position(pd.Timestamp("2019-11-29T17:30:00Z")) == "intraday"
    assert session_position(pd.Timestamp("2019-07-04T15:00:00Z")) == "non_session"
    assert session_position(pd.Timestamp("2020-03-09T13:00:00Z")) == "pre_open"  # 09:00 EDT after the DST change.

    def timing(report: str, accepted: str, available: str) -> tuple[int | None, int | None, bool | None, str]:
        result = report_timing(report, pd.Timestamp(accepted), pd.Timestamp(available))
        return result.lag_days, result.post_report_sessions, result.report_session_opened, result.label

    # Friday report, accepted before Monday's open: only the Friday session itself may have reacted.
    assert timing("2019-08-02", "2019-08-05T12:00:00Z", "2019-08-05T12:05:00Z") == (3, 0, True, "report_session_may_precede_sec")
    # Accepted the same day after the close.
    assert timing("2019-08-01", "2019-08-01T20:05:00Z", "2019-08-01T20:10:00Z") == (0, 0, True, "report_session_may_precede_sec")
    # A weekend or holiday report accepted before the next open: no session traded first.
    assert timing("2019-08-03", "2019-08-05T12:00:00Z", "2019-08-05T12:05:00Z")[3] == "no_post_report_session_before_sec"
    assert timing("2019-07-04", "2019-07-05T12:00:00Z", "2019-07-05T12:05:00Z")[3] == "no_post_report_session_before_sec"
    # A half-day report and a later intraday acceptance: the next full session traded first.
    assert timing("2019-11-29", "2019-12-02T18:00:00Z", "2019-12-02T18:05:00Z") == (3, 1, True, "post_report_sessions_precede_sec")
    assert timing("", "2019-08-02T01:00:00Z", "2019-08-02T13:30:00Z") == (None, None, None, "unknown")
    assert timing("not-a-date", "2019-08-02T01:00:00Z", "2019-08-02T13:30:00Z")[3] == "unknown"
    assert timing("1990-01-02", "2019-08-02T14:00:00Z", "2019-08-02T14:05:00Z")[1:] == (None, None, "unknown")


def test_saved_document_parsing_and_status_order() -> None:
    parsed = saved_document(SAVED_URL, "official:x", has_receipt=True)
    assert parsed == SavedDocument("0000000001", "0000000001-19-000001", "a-earnings.htm", "official:x", True)
    assert saved_document("https://investors.example.com/release.pdf", "official:x", has_receipt=True) is None
    receipt = SavedDocument("0000000001", "0000000001-19-000001", "other.htm", "official:x", True)
    evidence = SavedDocument("0000000001", "0000000001-19-000001", "a-earnings.htm", "identity_evidence:x", False)
    assert document_status([evidence], "a-earnings.htm") == "saved_without_receipt"
    assert document_status([evidence, receipt], "a-earnings.htm") == "other_filing_document_saved"
    assert document_status([], "a-earnings.htm") == "not_saved"
