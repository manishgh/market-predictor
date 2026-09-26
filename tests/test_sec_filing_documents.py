"""SEC filing-document collection over real EDGAR detail pages and an HTTP-level fake; no network."""
from __future__ import annotations

import hashlib
import inspect
import json
import threading
import time
from collections.abc import Callable
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.catalysts.sec_filings import document_collection as documents
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import sec_filing_documents as collector
from market_predictor.research import sec_page_collection as runner
from market_predictor.research.sec_form_inventory import SCHEMA as INVENTORY_SCHEMA
from market_predictor.sources import http
from market_predictor.sources.http import HttpByteResponse
from tests.support.sec_archive import header_page, settings

FIXTURES = Path(__file__).parent / "fixtures" / "sec"
ABT, ABT_CIK = "0001104659-19-040664", "0000001800"
REAL = (FIXTURES / f"{ABT}-index.htm").read_bytes()


def _page(accession: str, cik: str, *, replace: tuple[tuple[str, str], ...] = ()) -> bytes:
    """The real Abbott detail page re-addressed to another accession and CIK, optionally edited."""
    text = REAL.decode("utf-8").replace(ABT, accession).replace(ABT.replace("-", ""), accession.replace("-", ""))
    text = text.replace("/data/1800/", f"/data/{int(cik)}/")
    for old, new in replace:
        assert old in text, old
        text = text.replace(old, new)
    return text.encode("utf-8")


def _filing(accession: str, cik: str, items: str, *, form: str = "8-K", security: str = "cik:x") -> dict[str, Any]:
    return {"security_id": security, "sec_cik": cik, "accession_number": accession, "sec_form": form, "item_codes": items,
            "report_date": "2019-07-17", "filing_date": "2019-07-17",
            "accepted_at_utc": pd.Timestamp("2019-07-17T11:37:35Z"), "primary_document": "a19-12883_18k.htm"}


FILINGS = [_filing(ABT, ABT_CIK, "2.02,9.01", security="cik:0000001800"),
           _filing("0000000002-19-000001", "0000000002", "2.02,9.01"),
           _filing("0000000003-19-000001", "0000000003", "2.02,9.01"),
           _filing("0000000004-19-000001", "0000000004", "5.02")]


def _url(cik: str, accession: str, name: str) -> str:
    return f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/{accession.replace('-', '')}/{name}"


def _response(url: str, status: int, body: bytes, content_type: str = "text/html",
              retry_after: str | None = None) -> HttpByteResponse:
    headers = (("content-type", content_type), *((("retry-after", retry_after),) if retry_after is not None else ()))
    return HttpByteResponse(body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=status,
                            retrieved_at_utc=datetime.now(UTC), content_type=content_type, content_encoding=None, etag=None,
                            last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
                            body_representation="http_entity_encoded", safe_headers=headers)


class _Edgar:
    """Scripted responses per URL, consumed in order; the last one repeats."""

    def __init__(self, scripts: dict[str, list[Callable[[str], HttpByteResponse]]]) -> None:
        self.scripts = scripts
        self.urls: list[str] = []

    def __call__(self, url: str) -> HttpByteResponse:
        self.urls.append(url)
        script = self.scripts[url]
        step = script.pop(0) if len(script) > 1 else script[0]
        return step(url)

    def close(self) -> None:
        """No session to release."""


def _ok(body: bytes, content_type: str = "text/html") -> Callable[[str], HttpByteResponse]:
    return lambda url: _response(url, 200, body, content_type)


def _status(code: int, retry_after: str | None = None) -> Callable[[str], HttpByteResponse]:
    return lambda url: _response(url, code, b"", retry_after=retry_after)


def _oversize(url: str) -> HttpByteResponse:
    raise RuntimeError(f"{documents.OVERSIZE_MESSAGE}: declared=99999999 limit=16777216")


def _scripts() -> dict[str, list[Callable[[str], HttpByteResponse]]]:
    two, three = "0000000002-19-000001", "0000000003-19-000001"
    return {
        documents.index_url(ABT_CIK, ABT): [_ok(REAL)],
        _url(ABT_CIK, ABT, "a19-12883_18k.htm"): [_ok(b"<html>8-K</html>")],
        _url(ABT_CIK, ABT, "a19-12883_1ex99d1.htm"): [_status(503), _ok(b"<html>release</html>")],
        documents.index_url("0000000002", two): [_ok(_page(two, "0000000002"))],
        _url("0000000002", two, "a19-12883_18k.htm"): [_ok(b"<html>8-K two</html>")],
        _url("0000000002", two, "a19-12883_1ex99d1.htm"): [_oversize],
        documents.index_url("0000000003", three): [_status(404)],
    }


def _inventory(root: Path, rows: list[dict[str, Any]] = FILINGS, *, sealed: bool = False) -> dict[str, str]:
    folder = root / "data/research/sec_inventory"
    folder.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(rows).to_parquet(folder / "filings.parquet", index=False)
    manifest = folder / "_manifest.json"
    seal = {"sealed_until_rules_frozen": True} if sealed else {}
    manifest.write_text(json.dumps({"schema": INVENTORY_SCHEMA, "status": "complete",
                                    "mode": "later_sealed" if sealed else "initial_fit", **seal,
                                    "artifacts": {"filings.parquet": file_sha256(folder / "filings.parquet")}}), encoding="utf-8")
    return {"path": manifest.relative_to(root).as_posix(), "sha256": file_sha256(manifest)}


@pytest.fixture(autouse=True)
def _policy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(runner, "_guard", lambda: None)
    monkeypatch.setattr(documents, "RETRY_WAITS_SECONDS", (0.0, 0.0))
    monkeypatch.delenv("MARKET_PREDICTOR_RUNTIME_DIR", raising=False)


def _collect(root: Path, edgar: _Edgar, monkeypatch: pytest.MonkeyPatch, name: str = "sec_documents", **changes: Any
             ) -> dict[str, Any]:
    monkeypatch.setattr(runner, "sec_fetch", lambda _settings, _stop: edgar)
    return collector.collect_sec_filing_documents(root=root, inventory=changes.pop("inventory", None) or _inventory(root),
                                                  output=root / "data/raw" / name, settings=settings(),
                                                  **changes)


def _after_cooldown(monkeypatch: pytest.MonkeyPatch) -> None:
    """Resume as if SEC's cooldown (at least the configured minimum) has elapsed."""
    original = documents.collect
    monkeypatch.setattr(documents, "collect",
                        lambda **kwargs: original(**kwargs, now=lambda: datetime.now(UTC) + timedelta(hours=2)))


def test_real_detail_page_parses_strictly_and_selects_primary_and_ex99() -> None:
    index = documents.parse_filing_index(REAL, accession=ABT)
    assert (index.filing_date, index.period_of_report, index.item_codes) == ("2019-07-17", "2019-07-17", ("2.02", "9.01"))
    assert index.accepted_at_utc == pd.Timestamp("2019-07-17T11:37:35Z")
    assert [(item.sequence, item.document_type, item.filename) for item in index.documents] == [
        ("1", "8-K", "a19-12883_18k.htm"), ("2", "EX-99.1", "a19-12883_1ex99d1.htm"), ("3", "GRAPHIC", "g128831mm01i001.gif")]
    selected = documents.selected_documents(index, "a19-12883_18k.htm")
    assert [item.sequence for item in selected] == ["1", "2"]
    unit = {key: str(value) for key, value in documents.work_list(pd.DataFrame(FILINGS[:1])).iloc[0].items()}
    assert documents.index_rejection(index, unit) is None
    for key, value, reason in (("accepted_at_utc", "2019-07-17T11:37:36+00:00", "acceptance"), ("filing_date", "2019-07-18", "filing date"),
                               ("report_date", "2019-07-16", "period"), ("item_codes", "2.02", "item codes"),
                               ("primary_document", "other.htm", "primary document")):
        assert reason in str(documents.index_rejection(index, {**unit, key: value}))


@pytest.mark.parametrize(("edit", "message"), [
    (('href="/Archives/edgar/data/1800/000110465919040664/a19-12883_18k.htm"',
      'href="/Archives/edgar/data/1800/000110465919099999/a19-12883_18k.htm"'), "outside its accession folder"),
    (('<div class="infoHead">Accepted</div>', '<div class="infoHead">Received</div>'), "acceptance"),
    (('summary="Document Format Files"', 'summary="Other Files"'), "exactly one document table"),
    (('<th scope="col">Size</th>', '<th scope="col">Bytes</th>'), "header differs"),
    (("a19-12883_1ex99d1.htm</a>", "a19-12883_1ex99d1.htm</a></td><td>extra"), "malformed"),
])
def test_malformed_or_foreign_detail_pages_raise(edit: tuple[str, str], message: str) -> None:
    with pytest.raises(DataReadinessError, match=message):
        documents.parse_filing_index(REAL.decode().replace(*edit).encode(), accession=ABT)
    with pytest.raises(DataReadinessError, match="another accession"):
        documents.parse_filing_index(REAL, accession="0001104659-19-000001")


def test_inline_viewer_links_resolve_to_their_document() -> None:
    page = _page(ABT, ABT_CIK, replace=(('href="/Archives/edgar/data/1800/000110465919040664/a19-12883_18k.htm"',
                                         'href="/ix?doc=/Archives/edgar/data/1800/000110465919040664/a19-12883_18k.htm"'),))
    assert documents.parse_filing_index(page, accession=ABT).documents[0].filename == "a19-12883_18k.htm"


def test_collection_classifies_outcomes_retries_and_verifies(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edgar = _Edgar(_scripts())
    report = _collect(tmp_path, edgar, monkeypatch)
    assert report["status"] == "complete"
    totals = report["totals"]
    assert totals["units_by_phase_and_state"] == {"document/archived": 3, "document/oversize": 1, "index/archived": 2,
                                                  "index/missing": 1}
    assert totals["documents_by_type_and_state"] == {"EX-99.1/archived": 1, "EX-99.1/oversize": 1, "primary/archived": 2}
    assert totals["attempts_by_state"]["retryable"] == 1 and totals["archived_content_types"] == {"text/html": 5}
    assert set(edgar.urls) == set(_scripts()) and "0000000004" not in " ".join(edgar.urls)
    output = tmp_path / "data/raw/sec_documents"
    manifest = json.loads((output / "_manifest.json").read_text())
    store = documents.Store.open(output, manifest["checkpoint_sha256"], manifest["request_sha256"])
    final, receipts = documents.outcomes(store, documents.work_list(pd.DataFrame(FILINGS)))
    store.verify(receipts)
    release = receipts.loc[receipts.unit_id.eq(f"{ABT}/2") & receipts.state.eq("archived")].iloc[0]
    assert json.loads(release.index_row_json)["document_type"] == "EX-99.1" and int(release.attempt) == 2
    with pytest.raises(DataReadinessError, match="immutable"):
        _collect(tmp_path, edgar, monkeypatch, resume_checkpoint_sha256=manifest["checkpoint_sha256"])
    opened, published = collector.open_sec_document_collection(
        tmp_path, {"path": "data/raw/sec_documents/_manifest.json", "sha256": report["manifest_sha256"]})
    assert published["status"] == "complete" and set(opened.shards) == set(store.shards)


def test_stop_on_403_then_resume_only_with_pinned_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = _scripts()
    scripts[documents.index_url("0000000002", "0000000002-19-000001")].insert(0, _status(403, retry_after="0"))
    edgar = _Edgar(scripts)
    stopped = _collect(tmp_path, edgar, monkeypatch)
    assert stopped["status"] == "stopped" and stopped["stop_status_code"] == 403
    with pytest.raises(DataReadinessError, match="checkpoint SHA256"):
        _collect(tmp_path, edgar, monkeypatch)
    with pytest.raises(DataReadinessError, match="pin differs"):
        _collect(tmp_path, edgar, monkeypatch, resume_checkpoint_sha256="0" * 64)
    shard = next((tmp_path / "data/raw/sec_documents/shards").glob("*.zip"))
    original = shard.read_bytes()
    shard.write_bytes(original + b" ")
    with pytest.raises(DataReadinessError, match="shard changed"):
        _collect(tmp_path, edgar, monkeypatch, resume_checkpoint_sha256=stopped["checkpoint_sha256"])
    shard.write_bytes(original)
    _after_cooldown(monkeypatch)
    resumed = _collect(tmp_path, edgar, monkeypatch, resume_checkpoint_sha256=stopped["checkpoint_sha256"])
    assert resumed["status"] == "complete"
    assert resumed["totals"]["attempts_by_state"]["stopped"] == 1


def test_request_binds_inventory_pilot_and_work_list(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    edgar = _Edgar(_scripts())
    pilot = _collect(tmp_path, edgar, monkeypatch, "pilot", pilot_accessions=(ABT,))
    assert pilot["totals"]["units_by_phase_and_state"] == {"document/archived": 2, "index/archived": 1}
    with pytest.raises(DataReadinessError, match="not selected filings"):
        _collect(tmp_path, edgar, monkeypatch, "pilot-bad", pilot_accessions=("0000000004-19-000001",))
    stopped_scripts = _scripts()
    stopped_scripts[documents.index_url(ABT_CIK, ABT)].insert(0, _status(429, retry_after="0"))
    first = _collect(tmp_path, _Edgar(stopped_scripts), monkeypatch, "changing")
    with pytest.raises(DataReadinessError, match="request differs"):
        _collect(tmp_path, edgar, monkeypatch, "changing", pilot_accessions=(ABT,),
                 resume_checkpoint_sha256=first["checkpoint_sha256"])
    with pytest.raises(DataReadinessError, match="direct child of data/raw"):
        collector.collect_sec_filing_documents(root=tmp_path, inventory=_inventory(tmp_path),
                                               output=tmp_path / "data/raw/nested/out")


def test_rejected_detail_page_is_kept_and_plans_no_documents(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = _scripts()
    scripts[documents.index_url(ABT_CIK, ABT)] = [_ok(REAL.replace(b"2019-07-17 07:37:35", b"2019-07-17 08:37:35"))]
    report = _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,))
    assert report["totals"]["units_by_phase_and_state"] == {"index/rejected": 1}


def test_busy_lease_prevents_any_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(collector, "load_sec_form_inventory", lambda *_, **__: pytest.fail("inventory read without the lease"))
    with heavy_job_lease("test-owner", runtime_dir=tmp_path / "data/runtime"), pytest.raises(HeavyJobBusyError):
        _collect(tmp_path, _Edgar(_scripts()), monkeypatch, inventory={"path": "x", "sha256": "0" * 64})


def test_co_registrant_links_keep_their_filer_folder() -> None:
    page = _page(ABT, ABT_CIK, replace=(('href="/Archives/edgar/data/1800/000110465919040664/a19-12883_1ex99d1.htm"',
                                         'href="/Archives/edgar/data/1415404/000110465919040664/a19-12883_1ex99d1.htm"'),))
    index = documents.parse_filing_index(page, accession=ABT)
    assert index.documents[1].path == "/Archives/edgar/data/1415404/000110465919040664/a19-12883_1ex99d1.htm"


def test_joint_filing_is_one_unit_with_every_filer() -> None:
    rows = [_filing(ABT, "0000001800", "2.02,9.01"), _filing(ABT, "0000001801", "2.02,9.01")]
    units = documents.work_list(pd.DataFrame(rows))
    assert (len(units), units.sec_cik.item(), units.filer_ciks.item()) == (1, "0000001800", "0000001800,0000001801")
    with pytest.raises(DataReadinessError, match="conflicting"):
        documents.work_list(pd.DataFrame([rows[0], {**rows[1], "report_date": "2019-07-18"}]))


def test_units_still_failing_leave_the_run_incomplete_until_a_resume_finishes_them(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = _scripts()
    exhibit = _url(ABT_CIK, ABT, "a19-12883_1ex99d1.htm")
    scripts[exhibit] = [_status(503)]
    edgar = _Edgar(scripts)
    first = _collect(tmp_path, edgar, monkeypatch, pilot_accessions=(ABT,))
    assert (first["status"], first["phase"], first["units_without_final_outcome"]) == ("incomplete", "document", 1)
    output = tmp_path / "data/raw/sec_documents"
    assert not (output / "_manifest.json").exists()
    assert edgar.urls.count(exhibit) == documents.MAXIMUM_ATTEMPTS
    scripts[exhibit] = [_ok(b"<html>release</html>")]
    resumed = _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,),
                       resume_checkpoint_sha256=first["checkpoint_sha256"])
    assert resumed["status"] == "complete"
    assert resumed["totals"]["units_by_phase_and_state"] == {"document/archived": 2, "index/archived": 1}
    receipts = pd.concat([pd.read_parquet(path) for path in sorted((output / "shards").glob("*.parquet"))])
    exhibit_attempts = receipts.loc[receipts.url.eq(exhibit)].sort_values("attempt")
    assert exhibit_attempts.attempt.astype(int).tolist() == [1, 2, 3, 4]
    assert exhibit_attempts.state.tolist() == ["retryable"] * 3 + ["archived"]


def test_later_passes_wait_before_retrying(tmp_path: Path) -> None:
    units = _header_units(2)
    calls: list[str] = []

    def fetch(url: str) -> HttpByteResponse:
        calls.append(url)
        return _response(url, 503, b"")

    waits: list[float] = []
    store = documents.Store(tmp_path, {})
    store.write_checkpoint("request")
    result = documents.collect(store=store, units=units, fetch=fetch, request_sha256="request", memory_check=lambda: None,
                               cooldowns={403: 1.0, 429: 1.0}, stop=threading.Event(), phases=("header",), workers=1,
                               retry_waits=(7.0, 11.0), sleep=waits.append)
    assert result == {"status": "incomplete", "phase": "header", "units_without_final_outcome": 2}
    assert waits == [7.0, 11.0] and len(calls) == 2 * documents.MAXIMUM_ATTEMPTS


def test_uncommitted_shard_files_are_removed_on_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = _scripts()
    scripts[documents.index_url(ABT_CIK, ABT)].insert(0, _status(429, retry_after="0"))
    stopped = _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,))
    output = tmp_path / "data/raw/sec_documents"
    for suffix in ("zip", "parquet"):  # A crash after the shard rename but before its checkpoint.
        (output / "shards" / f"shard-00099.{suffix}").write_bytes(b"uncommitted")
    (output / ".shard-crashed").mkdir()
    _after_cooldown(monkeypatch)
    resumed = _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,),
                       resume_checkpoint_sha256=stopped["checkpoint_sha256"])
    assert resumed["status"] == "complete"
    assert not (output / "shards/shard-00099.zip").exists() and not (output / ".shard-crashed").exists()


def test_rewritten_receipt_is_detected_even_with_a_rewritten_checkpoint(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    report = _collect(tmp_path, _Edgar(_scripts()), monkeypatch, pilot_accessions=(ABT,))
    output = tmp_path / "data/raw/sec_documents"
    shard = output / "shards/shard-00000.parquet"
    frame = pd.read_parquet(shard)
    frame.loc[0, "content_type"] = "text/plain"
    frame.to_parquet(shard, index=False)
    checkpoint = json.loads((output / "_checkpoint.json").read_text())
    checkpoint["shards"]["shard-00000"]["parquet_sha256"] = file_sha256(shard)
    (output / "_checkpoint.json").write_text(json.dumps(checkpoint), encoding="utf-8")
    store = documents.Store.open(output, file_sha256(output / "_checkpoint.json"), report["request_sha256"])
    with pytest.raises(DataReadinessError, match="receipt differs"):
        store.verify(store.receipts())


def test_cooldown_blocks_an_early_resume(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    scripts = _scripts()
    scripts[documents.index_url(ABT_CIK, ABT)].insert(0, _status(429, retry_after="3600"))
    stopped = _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,))
    assert stopped["status"] == "stopped" and stopped["stop_status_code"] == 429
    with pytest.raises(DataReadinessError, match="cooldown has not elapsed"):
        _collect(tmp_path, _Edgar(scripts), monkeypatch, pilot_accessions=(ABT,),
                 resume_checkpoint_sha256=stopped["checkpoint_sha256"])


def test_oversize_classification_matches_the_pinned_client() -> None:
    assert documents.OVERSIZE_MESSAGE in inspect.getsource(http._read_bounded_http_entity)


def test_sealed_collection_reports_units_only_and_stays_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    rows = [{key: value for key, value in row.items() if key != "security_id"} for row in FILINGS]
    report = _collect(tmp_path, _Edgar(_scripts()), monkeypatch, inventory=_inventory(tmp_path, rows, sealed=True))
    assert report["sealed_until_rules_frozen"] is True
    assert set(report["totals"]) == {"units_by_phase_and_state", "attempts_by_state"}
    with pytest.raises(DataReadinessError, match="sealed"):
        collector.open_sec_document_collection(
            tmp_path, {"path": "data/raw/sec_documents/_manifest.json", "sha256": report["manifest_sha256"]})


PILOT = json.loads((FIXTURES / "pilot_expected.json").read_text(encoding="utf-8"))


@pytest.mark.parametrize("accession", sorted(PILOT))
def test_real_pilot_pages_verify_against_saved_metadata(accession: str) -> None:
    """Real EDGAR pages: summer and winter clocks, inline XBRL, an 8-K/A, no EX-99, and co-registrant folders."""
    expected = PILOT[accession]
    index = documents.parse_filing_index((FIXTURES / f"{accession}-index.htm").read_bytes(), accession=accession)
    assert documents.index_rejection(index, expected["unit"]) is None
    selected = documents.selected_documents(index, expected["unit"]["primary_document"])
    assert sorted([item.sequence, item.document_type, f"{item.path.split('/')[4]}/{item.filename}"] for item in selected) == (
        expected["documents"])


def test_header_phase_keeps_any_parseable_detail_page_and_rejects_another_accession() -> None:
    unit = {"unit_id": f"{ABT}/header", "phase": "header", "sec_cik": ABT_CIK, "accession_number": ABT, "sequence": "header",
            "document_type": "filing_index", "url": documents.index_url(ABT_CIK, ABT), "index_body_sha256": None,
            "index_row_json": None, "metadata": {}}
    kept, body = documents.attempt(unit, lambda url: _response(url, 200, REAL), number=1, now=lambda: datetime.now(UTC))
    assert (kept["state"], kept["reason"], body) == ("archived", None, REAL)
    foreign, _ = documents.attempt({**unit, "accession_number": "0001104659-19-000001"},
                                   lambda url: _response(url, 200, REAL), number=1, now=lambda: datetime.now(UTC))
    assert (foreign["state"], foreign["reason"]) == ("rejected", "filing index is for another accession")
    header = documents.parse_filing_header(REAL, accession=ABT)
    assert (header.filing_date, header.accepted_at_utc) == ("2019-07-17", pd.Timestamp("2019-07-17T11:37:35Z"))


class _Session:
    def __init__(self) -> None:
        self.closed = False

    def close(self) -> None:
        self.closed = True


class _GovernedTransport:
    """SEC's client hooks around a scripted, slow response: the real governor paces and blocks every send."""

    def __init__(self, governor: Any, log: dict[str, Any]) -> None:
        self.governor = governor
        self.log = log
        self.session = _Session()
        log["sessions"].append(self.session)

    def get_bytes_with_metadata(self, url: str, **_: Any) -> HttpByteResponse:
        self.governor.acquire()
        self.log["sent"].append(time.monotonic())
        time.sleep(0.05)
        accession = url.rsplit("/", 1)[1].removesuffix("-index.htm")
        blocked = url == self.log["blocked"]
        response = _response(url, 403, b"") if blocked else _response(url, 200, header_page(accession, "2019-08-01",
                                                                                               "2019-08-01 16:05:00"))
        self.governor.observe_response(response.status_code, dict(response.safe_headers))
        if blocked:
            self.log["blocked_at"] = time.monotonic()
        return response


def _header_units(count: int) -> pd.DataFrame:
    accessions = [f"0000000001-19-{number:06d}" for number in range(count)]
    return pd.DataFrame({"sec_cik": "0000000001", "accession_number": accessions,
                         "url": [documents.index_url("0000000001", accession) for accession in accessions]})


def test_a_403_through_the_real_governor_sends_nothing_more_and_returns_promptly(
        tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    units = _header_units(4 * runner.WORKERS)
    log: dict[str, Any] = {"sent": [], "sessions": [], "blocked": units.url.iloc[3]}
    monkeypatch.setattr(runner, "SecSource", lambda _settings, *, governor: type("Source", (), {
        "client": _GovernedTransport(governor, log)})())
    store = documents.Store(tmp_path, {})
    store.write_checkpoint("request")
    stop = threading.Event()
    fetch = runner.sec_fetch(settings(), stop)
    outcome: dict[str, Any] = {}
    worker = threading.Thread(target=lambda: outcome.update(documents.collect(
        store=store, units=units, fetch=fetch, request_sha256="request", memory_check=lambda: None,
        cooldowns={403: runner.FORBIDDEN_COOLDOWN_SECONDS, 429: runner.RATE_LIMIT_COOLDOWN_SECONDS}, stop=stop,
        phases=("header",), workers=runner.WORKERS)))
    started = time.monotonic()
    worker.start()
    worker.join(timeout=15)
    assert not worker.is_alive() and time.monotonic() - started < 15
    assert outcome["status"] == "stopped" and outcome["stop_status_code"] == 403
    assert all(sent <= log["blocked_at"] for sent in log["sent"])
    receipts = store.receipts()
    assert len(receipts) == len(log["sent"]) < len(units) and receipts.state.eq("stopped").sum() == 1
    fetch.close()
    assert log["sessions"] and all(session.closed for session in log["sessions"])


def test_a_failed_attempt_keeps_every_other_finished_receipt(tmp_path: Path) -> None:
    units = _header_units(6)
    bad = units.url.iloc[2]

    def fetch(url: str) -> HttpByteResponse:
        time.sleep(0.02)
        accession = url.rsplit("/", 1)[1].removesuffix("-index.htm")
        page = _response(url, 200, header_page(accession, "2019-08-01", "2019-08-01 16:05:00"))
        return _response(units.url.iloc[0], 200, page.body) if url == bad else page

    store = documents.Store(tmp_path, {})
    store.write_checkpoint("request")
    with pytest.raises(DataReadinessError, match="does not belong"):
        documents.collect(store=store, units=units, fetch=fetch, request_sha256="request", memory_check=lambda: None,
                          cooldowns={403: 1.0, 429: 1.0}, stop=threading.Event(), phases=("header",), workers=3)
    kept = store.receipts()
    assert not kept.empty and bad not in set(kept.url) and kept.state.eq("archived").all()


def test_a_page_without_an_accession_number_is_distinguished_from_a_foreign_page() -> None:
    with pytest.raises(DataReadinessError, match="lacks its accession number"):
        documents.parse_filing_header(b"<html>EDGAR is temporarily unavailable</html>", accession=ABT)
    with pytest.raises(DataReadinessError, match="another accession"):
        documents.parse_filing_header(REAL, accession="0001104659-19-000001")


def test_a_failed_memory_check_keeps_the_batch_and_every_in_flight_receipt(tmp_path: Path,
                                                                          monkeypatch: pytest.MonkeyPatch) -> None:
    units = _header_units(8)
    monkeypatch.setattr(documents, "SHARD_ATTEMPTS", 2)

    def fetch(url: str) -> HttpByteResponse:
        time.sleep(0.02)
        accession = url.rsplit("/", 1)[1].removesuffix("-index.htm")
        return _response(url, 200, header_page(accession, "2019-08-01", "2019-08-01 16:05:00"))

    def memory_check() -> None:
        raise MemoryBudgetError("system memory above the limit")

    store = documents.Store(tmp_path, {})
    store.write_checkpoint("request")
    with pytest.raises(MemoryBudgetError):
        documents.collect(store=store, units=units, fetch=fetch, request_sha256="request", memory_check=memory_check,
                          cooldowns={403: 1.0, 429: 1.0}, stop=threading.Event(), phases=("header",), workers=3)
    kept = store.receipts()
    assert len(kept) >= documents.SHARD_ATTEMPTS and kept.state.eq("archived").all()
    assert len(kept) < len(units) and not kept.unit_id.duplicated().any()
