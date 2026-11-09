from __future__ import annotations

import copy
import json
from contextlib import contextmanager
from pathlib import Path

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment import dividend_resolution as resolution
from market_predictor.research import investment_features
from market_predictor.swing.contracts.holding_materialization import SourcePin
from tests.test_investment_dividend_currencies import correction_fixture, fixture, write_json


def setup_resolution(tmp_path, monkeypatch, *, bridge=False, record_updates=None):
    original, item = fixture(tmp_path, bridge=bridge)
    original["cusip"] = "78462F103"
    if record_updates:
        original.update(record_updates)
    page = write_json(tmp_path / "provider.json", {"corporate_actions": {"cash_dividends": [original]}})
    seed = copy.deepcopy(item)
    seed["action_id"] = "preserved"
    seed_pin = write_json(tmp_path / "seed.json", {
        "schema": "market_predictor.investment_dividend_currency_corroborations", "corroborations": [seed]})
    separate = tmp_path / "correction-source"
    separate.mkdir()
    _, _, correction = correction_fixture(separate)
    correction_pin = write_json(tmp_path / "corrections.json", {
        "schema": "market_predictor.investment_dividend_amount_corrections", "corrections": [correction]})
    archive_pin = write_json(tmp_path / "archive.json", {"metadata": "test-only"})
    unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": item["tax"]["source"], "gross_context": item["tax"]["gross_context"]}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {"source_files": {}, "records": [unit]})
    monkeypatch.setattr(resolution, "guard", lambda *args: None)
    monkeypatch.setattr(investment_features, "guard", lambda *args: None)

    @contextmanager
    def lease(*args, **kwargs):
        yield
    monkeypatch.setattr(resolution, "heavy_job_lease", lease)
    request = {"schema": "market_predictor.investment_dividend_resolution_request", "events": [{
        "action_id": original["id"], "symbol": "SPY", "ex_date": original["ex_date"],
        "original_record_sha256": json_sha256(original), "source": page}], "identities": [{
        "symbol": "SPY", "cusip": "78462F103", "isin": "US78462F1030", "issuer_workbook": item["identity"]["source"]}],
        "tax_archives": [archive_pin], "preserved_corroborations": seed_pin,
        "preserved_amount_corrections": correction_pin}
    return request, original, item


def run_resolution(tmp_path, request):
    reference = write_json(tmp_path / "request.json", request)
    resolution.resolve_investment_dividends(root=tmp_path, request=SourcePin(**reference), output=tmp_path / "final")
    return json.loads((tmp_path / "final/_manifest.json").read_text())


@pytest.mark.parametrize("bridge", [False, True])
def test_exact_and_issuer_bridge_preserve_sources_and_prior_bytes(tmp_path, monkeypatch, bridge):
    request, original, item = setup_resolution(tmp_path, monkeypatch, bridge=bridge)
    correction_bytes = (tmp_path / "corrections.json").read_bytes()
    report = run_resolution(tmp_path, request)
    assert report["required_events"] == report["new_supported_events"] == 1
    assert report["unresolved_events"] == 0
    assert (tmp_path / "final/amount-corrections.json").read_bytes() == correction_bytes
    output = json.loads((tmp_path / "final/currency-corroborations.json").read_text())["corroborations"]
    assert output[0] == json.loads((tmp_path / "seed.json").read_text())["corroborations"][0]
    assert output[1]["original_record_sha256"] == json_sha256(original)
    assert output[1]["method"] == ("issuer_rate_tax_currency_bridge" if bridge else "exact_tax_event")
    assert all(report[key] is False for key in investment_features.FLAGS)


def test_original_hash_poison_is_fatal_before_publication(tmp_path, monkeypatch):
    request, _, _ = setup_resolution(tmp_path, monkeypatch)
    request["events"][0]["original_record_sha256"] = "a" * 64
    with pytest.raises(DataReadinessError, match="original record"):
        run_resolution(tmp_path, request)
    assert not (tmp_path / "final").exists()


@pytest.mark.parametrize("updates", [{"special": True}, {"foreign": True}, {"rate": 1.383620}])
def test_unsupported_terms_and_amount_conflict_remain_local_gaps(tmp_path, monkeypatch, updates):
    request, _, _ = setup_resolution(tmp_path, monkeypatch, record_updates=updates)
    report = run_resolution(tmp_path, request)
    assert report["unresolved_events"] == 1 and report["new_supported_events"] == 0
    if "rate" in updates:
        assert report["pending_amount_corrections"] == 1


def test_duplicate_required_action_ids_refuse_closed_request(tmp_path, monkeypatch):
    request, _, _ = setup_resolution(tmp_path, monkeypatch)
    request["events"].append(copy.deepcopy(request["events"][0]))
    with pytest.raises(ValueError, match="repeat"):
        run_resolution(tmp_path, request)


def test_conflicting_payment_versions_do_not_select_latest(tmp_path, monkeypatch):
    request, _, item = setup_resolution(tmp_path, monkeypatch)
    body = json.loads((tmp_path / "tax.json").read_text())
    body["data"]["calculationDetails"]["entries"].append({
        **body["data"]["calculationDetails"]["entries"][0], "value": 9})
    source = write_json(tmp_path / "conflicting-tax.json", body)
    unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": {"body": source, "receipt": item["tax"]["source"]["receipt"]},
            "gross_context": item["tax"]["gross_context"]}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {"source_files": {}, "records": [unit]})
    report = run_resolution(tmp_path, request)
    assert report["unresolved_events"] == 1
    row = json.loads((tmp_path / "final/event-coverage.json").read_text())["events"][0]
    assert row["reasons"] == ["tax_payment_versions_conflict"]


def test_identical_duplicate_payment_in_one_source_is_local_ambiguous_gap(tmp_path, monkeypatch):
    request, _, item = setup_resolution(tmp_path, monkeypatch)
    body = json.loads((tmp_path / "tax.json").read_text())
    body["data"]["calculationDetails"]["entries"].append(copy.deepcopy(body["data"]["calculationDetails"]["entries"][0]))
    source = write_json(tmp_path / "duplicate-tax.json", body)
    unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": {"body": source, "receipt": item["tax"]["source"]["receipt"]},
            "gross_context": item["tax"]["gross_context"]}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {"source_files": {}, "records": [unit]})
    report = run_resolution(tmp_path, request)
    assert report["unresolved_events"] == 1
    row = json.loads((tmp_path / "final/event-coverage.json").read_text())["events"][0]
    assert row["reasons"] == ["tax_payment_ex_date_ambiguous"]


@pytest.mark.parametrize("variant", ["missing_gross", "gross_ambiguous", "receipt_corruption"])
def test_valid_source_semantic_gaps_are_local_but_receipt_corruption_fatal(tmp_path, monkeypatch, variant):
    from tests.test_investment_dividend_currencies import captured

    request, _, item = setup_resolution(tmp_path, monkeypatch)
    if variant == "receipt_corruption":
        receipt = item["tax"]["source"]["receipt"]
        Path(receipt["path"]).write_text("{}")
        with pytest.raises(DataReadinessError, match="hash differs"):
            run_resolution(tmp_path, request)
        assert not (tmp_path / "final").exists()
        return
    body = json.loads((tmp_path / "gross.json").read_text())
    if variant == "missing_gross":
        body["data"]["html"] = body["data"]["html"].replace("Brutto", "Other value")
    else:
        html = body["data"]["html"]
        cells = html[html.index("<tr>"):html.index("</tr>") + 5]
        body["data"]["html"] = html.replace("</tbody>", cells + "</tbody>")
    item["tax"]["gross_context"] = captured(tmp_path / "modified-gross.json", json.dumps(body).encode())
    unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": item["tax"]["source"], "gross_context": item["tax"]["gross_context"]}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {"source_files": {}, "records": [unit]})
    report = run_resolution(tmp_path, request)
    assert report["unresolved_events"] == 1 and report["new_supported_events"] == 0


@pytest.mark.parametrize("bad_stamp,bad_value", [(None, 0), (1568937600001, 0.1)])
def test_malformed_unrelated_payment_date_is_local_and_other_security_continues(tmp_path, monkeypatch, bad_stamp, bad_value):
    from tests.test_investment_dividend_currencies import captured

    request, original, item = setup_resolution(tmp_path, monkeypatch)
    clean_tax = json.loads((tmp_path / "tax.json").read_text())
    malformed_tax = copy.deepcopy(clean_tax)
    malformed_tax["data"]["calculationDetails"]["entries"].append({
        "type": "PAYMENT", "exDate": bad_stamp, "value": bad_value,
        "date": 1689033600000, "paymentDate": 1689033600000, "valueChf": 0, "valueTotal": 0,
        "currency": {"shortName": "USD"}, "nominalQuantity": 1})
    malformed_source = captured(tmp_path / "malformed-tax.json", json.dumps(malformed_tax).encode())
    first_unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": malformed_source, "gross_context": item["tax"]["gross_context"]}}

    unaffected = dict(original, id="unaffected-qqq", symbol="QQQ", cusip="46090E103")
    page = write_json(tmp_path / "qqq-provider.json", {"corporate_actions": {"cash_dividends": [unaffected]}})
    request["events"].append({"action_id": unaffected["id"], "symbol": "QQQ", "ex_date": unaffected["ex_date"],
        "original_record_sha256": json_sha256(unaffected), "source": page})
    official = captured(tmp_path / "qqq-official.html", b"<p>Fund QQQ 46090E103</p>", authority="official")
    request["identities"].append({"symbol": "QQQ", "cusip": "46090E103", "isin": "US46090E1038", "official_document": {
        "kind": "official_document", "source": official, "cusip": "46090E103", "fund_name": "Fund", "quotes": ["Fund QQQ 46090E103"]}})
    clean_tax["data"]["calculation"].update(isin="US46090E1038", valorNumber=123)
    clean_source = captured(tmp_path / "qqq-tax.json", json.dumps(clean_tax).encode())
    gross_body = Path(item["tax"]["gross_context"]["body"]["path"]).read_bytes()
    clean_gross = captured(tmp_path / "qqq-gross.json", gross_body)
    for pair in (clean_source, clean_gross):
        receipt_path = Path(pair["receipt"]["path"])
        receipt = json.loads(receipt_path.read_text())
        receipt["request"].update(isin="US46090E1038", valorNumber=123)
        pair["receipt"] = write_json(receipt_path, receipt)
    second_unit = {"status": "responses_archived", "symbol": "QQQ", "isin": "US46090E1038", "year": 2019,
        "sources": {"structured": clean_source, "gross_context": clean_gross}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {
        "source_files": {}, "records": [first_unit, second_unit]})
    result = run_resolution(tmp_path, request)
    assert result["unresolved_events"] == result["new_supported_events"] == 1
    rows = {r["action_id"]: r for r in json.loads((tmp_path / "final/event-coverage.json").read_text())["events"]}
    assert rows[original["id"]]["reasons"] == ["tax distribution ex-date is not UTC midnight milliseconds"]
    assert rows[unaffected["id"]]["status"] == "new_assertion_source_validated"


def test_payload_with_only_unsupported_dates_reports_date_gap_not_event_absence(tmp_path, monkeypatch):
    from tests.test_investment_dividend_currencies import captured

    request, _, item = setup_resolution(tmp_path, monkeypatch)
    body = json.loads((tmp_path / "tax.json").read_text())
    body["data"]["calculationDetails"]["entries"][0].update(exDate=None, value=0)
    source = captured(tmp_path / "only-null-date.json", json.dumps(body).encode())
    unit = {"status": "responses_archived", "symbol": "SPY", "isin": "US78462F1030", "year": 2019,
        "sources": {"structured": source, "gross_context": item["tax"]["gross_context"]}}
    monkeypatch.setattr(resolution, "verify_ictax_archive", lambda **kwargs: {"source_files": {}, "records": [unit]})
    result = run_resolution(tmp_path, request)
    assert result["unresolved_events"] == 1
    row = json.loads((tmp_path / "final/event-coverage.json").read_text())["events"][0]
    assert row["reasons"] == ["tax distribution ex-date is not UTC midnight milliseconds"]


@pytest.mark.parametrize("mode", ["first", "next_file_chunk", "next_byte_chunk", "large_single_file"])
def test_bounded_publication_hash_guard_refuses_before_next_reads(tmp_path, monkeypatch, mode):
    import hashlib

    from market_predictor.core.errors import MemoryBudgetError

    sizes = {
        "first": [1], "next_file_chunk": [1] * 33,
        "next_byte_chunk": [2 * 1024**2] * 4 + [1], "large_single_file": [1, 8 * 1024**2 + 1],
    }[mode]
    pins = {}
    for number, size in enumerate(sizes):
        path = tmp_path / f"source-{number}.bin"
        raw = b"x" * size
        path.write_bytes(raw)
        pins[str(path)] = hashlib.sha256(raw).hexdigest()
    hashed = []
    original_hash = resolution.file_sha256
    def hash_source(path):
        hashed.append(path)
        return original_hash(path)
    monkeypatch.setattr(resolution, "file_sha256", hash_source)
    calls = []
    def refuse(*args):
        calls.append(args)
        if mode == "first" or len(calls) == 2:
            raise MemoryBudgetError("publication memory refusal")
    monkeypatch.setattr(resolution, "guard", refuse)
    with pytest.raises(MemoryBudgetError, match="memory refusal"):
        resolution._recheck_publication_sources(tmp_path, pins)
    assert len(hashed) == {"first": 0, "next_file_chunk": 32, "next_byte_chunk": 4, "large_single_file": 1}[mode]


def test_bounded_publication_hashes_every_source_and_tampering_remains_fatal(tmp_path, monkeypatch):
    import hashlib

    pins = {}
    for number in range(35):
        path = tmp_path / f"source-{number}.bin"
        path.write_bytes(b"original")
        pins[str(path)] = hashlib.sha256(b"original").hexdigest()
    guards = []
    monkeypatch.setattr(resolution, "guard", lambda *args: guards.append(args))
    resolution._recheck_publication_sources(tmp_path, pins)
    assert len(guards) == 2
    Path(next(iter(pins))).write_bytes(b"modified")
    with pytest.raises(DataReadinessError, match="source changed"):
        resolution._recheck_publication_sources(tmp_path, pins)
