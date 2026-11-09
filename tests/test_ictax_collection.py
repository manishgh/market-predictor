from __future__ import annotations

import json
from datetime import UTC, datetime
from hashlib import sha256
from types import SimpleNamespace

import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.heavy_jobs import HeavyJobBusyError, heavy_job_lease
from market_predictor.research import investment_features as artifacts
from market_predictor.sources import ictax
from market_predictor.sources.http import HttpByteResponse

TOKEN = "session-secret-must-never-be-retained"
ISIN = "US78462F1030"
CAPTURE = datetime(2026, 11, 8, 20, tzinfo=UTC)


def response(url, payload, status=200):
    body = (json.dumps(payload, ensure_ascii=False) + "\r\n").encode("utf-8")
    return HttpByteResponse(body=body, requested_url=url, final_url=url, redirect_chain=(), status_code=status,
                            retrieved_at_utc=CAPTURE, content_type="application/json", content_encoding=None,
                            etag=None, last_modified=None, body_length=len(body), sha256=sha256(body).hexdigest(),
                            body_representation="http_entity_encoded", safe_headers=(("content-type", "application/json"),))


def source_payload(url, request):
    if url == ictax.SEARCH_URL:
        return {"total": 1, "count": 1, "security": [{"isin": request["isin"], "vn": 45088, "gk": 45088}]}
    if url == ictax.STRUCTURED_URL:
        return {"status": "SUCCESS", "error": None, "data": {"calculation": request,
                "calculationDetails": {"entries": [{"type": "PAYMENT", "currency": {"shortName": "USD"}}]}}}
    return {"status": "SUCCESS", "error": None, "data": {"html": "<table>Brutto USD</table>"}}


class Client:
    def __init__(self, transform=None):
        self.calls = []
        self.gets = 0
        self.transform = transform
        self.session = SimpleNamespace(close=lambda: None)

    def get_bytes_with_metadata(self, url, **options):
        assert url == ictax.SESSION_URL and options["allow_redirects"] is False
        self.gets += 1
        return response(url, {"status": "SUCCESS", "data": {"csrfToken": TOKEN}})

    def post_bytes_with_metadata(self, url, *, payload, headers, **options):
        assert headers["X-CSRF-TOKEN"] == TOKEN and options["raise_for_status"] is False
        assert options["allow_redirects"] is False and options["maximum_body_bytes"] == ictax.MAXIMUM_RESPONSE_BYTES
        self.calls.append((url, dict(payload)))
        result = response(url, source_payload(url, payload))
        return self.transform(url, payload, result) if self.transform else result


@pytest.fixture
def setup(tmp_path, monkeypatch):
    monkeypatch.setattr(ictax, "guard", lambda *args: None)
    monkeypatch.setattr(artifacts, "guard", lambda *args: None)
    # A synthetic authority root owns its producer files, as a real checkout does.
    implementation = tmp_path / "producer"
    implementation.mkdir()
    for name in ("ictax.py", "http.py"):
        (implementation / name).write_text("synthetic fixture producer\n")
    monkeypatch.setattr(ictax, "__file__", str(implementation / "ictax.py"))

    def prepare(years=(2019, 2020), funds=None, retained=None):
        payload = {"schema": "market_predictor.ictax_archive_request",
                   "funds": funds or [{"symbol": "SPY", "isin": ISIN, "years": list(years)}]}
        if retained:
            payload["retained"] = retained
        request = tmp_path / "request.json"
        request.write_text(json.dumps(payload))
        return {"root": tmp_path, "request": request, "request_sha256": file_sha256(request), "output": tmp_path / "archive"}
    return prepare


def verify(kwargs):
    path = kwargs["output"] / "final/_manifest.json"
    return ictax.verify_ictax_archive(root=kwargs["root"], manifest=path, manifest_sha256=file_sha256(path))


def test_archive_original_bodies_requests_clocks_and_lookup_once_then_verify(setup):
    kwargs = setup()
    client = Client()
    ictax.collect_ictax_archive(**kwargs, client=client)
    report = verify(kwargs)
    assert report["requested_fund_years"] == 2 and report["source_gap_fund_years"] == 0
    assert len(client.calls) == 5 and client.gets == 1
    assert sum(url == ictax.SEARCH_URL for url, _ in client.calls) == 1
    for row in report["records"]:
        assert row["status"] == "responses_archived"
        assert row["request"]["referenceDate"] == ictax.reference_date(row["year"])
        for name in ("structured", "gross_context"):
            source = row["sources"][name]
            body = (kwargs["root"] / source["body"]["path"]).read_bytes()
            receipt = json.loads((kwargs["root"] / source["receipt"]["path"]).read_text())
            assert body.endswith(b"\r\n") and sha256(body).hexdigest() == receipt["sha256"]
            assert receipt["request"] == row["request"] and receipt["retrieved_at"] == CAPTURE.isoformat()
            assert receipt["status_code"] == 200 and receipt["bytes"] == len(body)
    assert all(report[key] is False for key in artifacts.FLAGS)
    assert TOKEN not in "".join(p.read_text() for p in kwargs["output"].rglob("*") if p.is_file())


def test_missing_year_and_http_error_are_archived_and_other_years_continue(setup):
    def missing(url, request, original):
        if url == ictax.STRUCTURED_URL and request["referenceDate"] == ictax.reference_date(2019):
            return response(url, {"status": "NOT_FOUND", "error": None})
        if url == ictax.HTML_URL and request["referenceDate"] == ictax.reference_date(2019):
            return response(url, {"reason": "provider error"}, status=503)
        return original

    kwargs = setup()
    ictax.collect_ictax_archive(**kwargs, client=Client(missing))
    report = verify(kwargs)
    assert report["source_gap_fund_years"] == 1
    assert report["records"][0]["status"] == "source_gap" and report["records"][1]["status"] == "responses_archived"
    assert "gross_context:http_status_503" in report["records"][0]["reasons"]
    receipt = json.loads((kwargs["output"] / "units/SPY-2019/gross_context/receipt.json").read_text())
    assert receipt["status_code"] == 503


def test_transport_error_does_not_leak_exception_or_stop_other_endpoints(setup):
    def fail(url, request, original):
        if url == ictax.STRUCTURED_URL:
            raise RuntimeError("unsafe exception payload " + TOKEN)
        return original

    kwargs = setup(years=(2019,))
    client = Client(fail)
    ictax.collect_ictax_archive(**kwargs, client=client)
    report = verify(kwargs)
    assert len(client.calls) == 3 and report["source_gap_fund_years"] == 1
    capture = json.loads((kwargs["output"] / "units/SPY-2019/structured/_manifest.json").read_text())
    assert capture["error_type"] == "RuntimeError" and capture["files"] == {}
    assert TOKEN not in "".join(p.read_text() for p in kwargs["output"].rglob("*") if p.is_file())


def test_echoed_csrf_token_is_never_archived(setup):
    def echo(url, request, original):
        return response(url, {"secret": TOKEN}) if url == ictax.HTML_URL else original

    kwargs = setup(years=(2019,))
    ictax.collect_ictax_archive(**kwargs, client=Client(echo))
    report = verify(kwargs)
    assert report["source_gap_fund_years"] == 1
    assert not (kwargs["output"] / "units/SPY-2019/gross_context/response.bin").exists()
    assert TOKEN not in "".join(p.read_text() for p in kwargs["output"].rglob("*") if p.is_file())


def test_unicode_escaped_csrf_token_is_never_archived(setup):
    from dataclasses import replace

    escaped = "".join(f"\\u{ord(character):04x}" for character in TOKEN)

    def echo(url, request, original):
        if url == ictax.HTML_URL:
            body = ('{"nested":{"values":["' + escaped + '"]}}').encode("utf-8")
            assert TOKEN.encode("utf-8") not in body
            return replace(original, body=body, body_length=len(body), sha256=sha256(body).hexdigest())
        return original

    kwargs = setup(years=(2019,))
    ictax.collect_ictax_archive(**kwargs, client=Client(echo))
    assert verify(kwargs)["source_gap_fund_years"] == 1
    assert not (kwargs["output"] / "units/SPY-2019/gross_context/response.bin").exists()
    assert escaped not in "".join(p.read_text() for p in kwargs["output"].rglob("*") if p.is_file())


def test_identity_mismatch_is_fatal_after_retaining_original_response(setup):
    def mismatch(url, request, original):
        if url == ictax.STRUCTURED_URL:
            data = source_payload(url, {**request, "isin": "US46090E1038"})
            return response(url, data)
        return original

    kwargs = setup(years=(2019,))
    with pytest.raises(DataReadinessError, match="security/year"):
        ictax.collect_ictax_archive(**kwargs, client=Client(mismatch))
    assert (kwargs["output"] / "units/SPY-2019/structured/response.bin").exists()
    assert not (kwargs["output"] / "final").exists()


def test_ambiguous_lookup_cannot_select_first_result(setup):
    def ambiguous(url, request, original):
        if url == ictax.SEARCH_URL:
            return response(url, {"total": 2, "count": 2, "security": [{"isin": ISIN}, {"isin": ISIN}]})
        return original

    with pytest.raises(DataReadinessError, match="uniquely"):
        ictax.collect_ictax_archive(**setup(years=(2019,)), client=Client(ambiguous))


def test_actual_qqq_lookup_uses_valor_vn_when_gk_is_distinct(setup):
    actual_lookup = {"total": 1, "count": 1, "offset": 0,
                     "security": [{"id": 613167, "schema": "ICT", "gk": 28837, "vn": 41887067,
                                   "isin": "US46090E1038", "institution": "Invesco QQQ Trust Series I"}]}

    def source(url, request, original):
        return response(url, actual_lookup) if url == ictax.SEARCH_URL else original

    kwargs = setup(funds=[{"symbol": "QQQ", "isin": "US46090E1038", "years": [2021]}])
    client = Client(source)
    ictax.collect_ictax_archive(**kwargs, client=client)
    report = verify(kwargs)
    assert report["source_gap_fund_years"] == 0
    assert report["records"][0]["request"]["valorNumber"] == 41887067
    assert all(request["valorNumber"] == 41887067 for url, request in client.calls if url != ictax.SEARCH_URL)
    captured = kwargs["output"] / "lookups/US46090E1038/2021/response.bin"
    assert json.loads(captured.read_bytes()) == actual_lookup


def test_lookup_not_found_does_not_abort_other_funds(setup):
    def absent(url, request, original):
        if url == ictax.SEARCH_URL and request["isin"] == ISIN:
            return response(url, {"total": 0, "count": 0, "security": []})
        return original

    kwargs = setup(funds=[{"symbol": "SPY", "isin": ISIN, "years": [2019]},
                          {"symbol": "QQQ", "isin": "US46090E1038", "years": [2019]}])
    ictax.collect_ictax_archive(**kwargs, client=Client(absent))
    report = verify(kwargs)
    assert report["source_gap_fund_years"] == 1 and report["records"][1]["status"] == "responses_archived"


def test_missing_lookup_year_retries_next_year_and_resume_keeps_both_checkpoints(setup):
    interrupt = [True]

    def source(url, request, original):
        if url == ictax.SEARCH_URL and request["year"] == 2019:
            return response(url, {"total": 0, "count": 0, "security": []})
        if url == ictax.HTML_URL and interrupt[0]:
            raise MemoryBudgetError("fixture stop after successful 2020 lookup")
        return original

    kwargs = setup()
    client = Client(source)
    with pytest.raises(MemoryBudgetError):
        ictax.collect_ictax_archive(**kwargs, client=client)
    assert [request["year"] for url, request in client.calls if url == ictax.SEARCH_URL] == [2019, 2020]
    for year in (2019, 2020):
        assert (kwargs["output"] / "lookups" / ISIN / str(year) / "_manifest.json").exists()
    interrupt[0] = False
    resumed = Client(source)
    ictax.collect_ictax_archive(**kwargs, client=resumed)
    assert len(resumed.calls) == 1 and resumed.calls[0][0] == ictax.HTML_URL
    report = verify(kwargs)
    assert report["source_gap_fund_years"] == 1
    assert report["records"][0]["status"] == "source_gap" and report["records"][1]["status"] == "responses_archived"


def test_interrupt_resumes_only_missing_response_and_replays_before_refetch(setup):
    interrupt = [True]

    def stop(url, request, original):
        if url == ictax.HTML_URL and interrupt[0]:
            raise MemoryBudgetError("fixture stop")
        return original

    kwargs = setup(years=(2019,))
    client = Client(stop)
    with pytest.raises(MemoryBudgetError):
        ictax.collect_ictax_archive(**kwargs, client=client)
    assert not (kwargs["output"] / "final").exists()
    assert len(client.calls) == 3
    interrupt[0] = False
    ictax.collect_ictax_archive(**kwargs, client=client)
    assert len(client.calls) == 4 and client.calls[-1][0] == ictax.HTML_URL
    assert verify(kwargs)["source_gap_fund_years"] == 0


def test_corrupted_saved_response_refuses_resume_before_network(setup):
    def stop(url, request, original):
        if url == ictax.HTML_URL:
            raise MemoryBudgetError("fixture stop")
        return original

    kwargs = setup(years=(2019,))
    with pytest.raises(MemoryBudgetError):
        ictax.collect_ictax_archive(**kwargs, client=Client(stop))
    (kwargs["output"] / "units/SPY-2019/structured/response.bin").write_bytes(b"{}")
    client = Client()
    with pytest.raises(DataReadinessError, match="hash"):
        ictax.collect_ictax_archive(**kwargs, client=client)
    assert client.calls == [] and client.gets == 0


def retained_pair(tmp_path):
    detail = {"isin": ISIN, "valorNumber": 45088, "referenceDate": ictax.reference_date(2019), "language": "en", "isCantonBL": False}
    retained = {"symbol": "SPY", "year": 2019}
    for name, url in (("structured", ictax.STRUCTURED_URL), ("gross_context", ictax.HTML_URL)):
        body = response(url, source_payload(url, detail)).body
        body_path = tmp_path / f"legacy-{name}.json"
        body_path.write_bytes(body)
        receipt_path = tmp_path / f"legacy-{name}-receipt.json"
        receipt_path.write_text(json.dumps({"url": url, "method": "POST", "request": detail, "bytes": len(body),
                                            "sha256": sha256(body).hexdigest(), "retrieved_at": CAPTURE.isoformat()}))
        retained[name] = {"body": {"path": body_path.name, "sha256": file_sha256(body_path)},
                          "receipt": {"path": receipt_path.name, "sha256": file_sha256(receipt_path)}}
    return retained


def test_retained_legacy_pairs_keep_original_receipts_without_inventing_status(setup, tmp_path):
    retained = retained_pair(tmp_path)
    kwargs = setup(years=(2019,), retained=[retained])
    client = Client()
    ictax.collect_ictax_archive(**kwargs, client=client)
    report = verify(kwargs)
    assert client.calls == [] and client.gets == 0 and report["retained_fund_years"] == 1
    assert report["records"][0]["transport_status_basis"] == "original_legacy_receipts_no_http_status"
    for name in ("structured", "gross_context"):
        assert "status_code" not in json.loads((tmp_path / retained[name]["receipt"]["path"]).read_text())


def test_retained_year_mismatch_is_not_reused(setup, tmp_path):
    retained = retained_pair(tmp_path)
    path = tmp_path / retained["structured"]["receipt"]["path"]
    receipt = json.loads(path.read_text())
    receipt["request"]["referenceDate"] = ictax.reference_date(2020)
    path.write_text(json.dumps(receipt))
    retained["structured"]["receipt"]["sha256"] = file_sha256(path)
    kwargs = setup(years=(2019,), retained=[retained])
    with pytest.raises(DataReadinessError, match="year/security"):
        ictax.collect_ictax_archive(**kwargs, client=Client())


def test_shared_job_lease_refuses_before_request_read(setup):
    kwargs = setup()
    client = Client()
    with heavy_job_lease("fixture other job", runtime_dir=kwargs["root"] / "data/runtime"):
        with pytest.raises(HeavyJobBusyError):
            ictax.collect_ictax_archive(**kwargs, client=client)
    assert client.calls == []


def test_memory_refusal_precedes_request_read(setup, monkeypatch):
    kwargs = setup()
    monkeypatch.setattr(ictax, "guard", lambda *args: (_ for _ in ()).throw(MemoryBudgetError("memory refusal")))
    with pytest.raises(MemoryBudgetError):
        ictax.collect_ictax_archive(**kwargs, client=Client())
    assert not kwargs["output"].exists()


@pytest.mark.parametrize("years", [[2020, 2019], [2019, 2019]])
def test_request_population_is_explicit_sorted_and_unique(years):
    with pytest.raises(ValidationError):
        ictax.ICTaxArchiveRequest.model_validate_json(json.dumps({"schema": "market_predictor.ictax_archive_request",
                                                                 "funds": [{"symbol": "SPY", "isin": ISIN, "years": years}]}))
