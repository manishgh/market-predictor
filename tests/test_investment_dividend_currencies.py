"""Synthetic unit evidence for strict independent currency-source admission."""
from __future__ import annotations

import copy
import hashlib
import io
import json
import zipfile
from datetime import date, datetime, timedelta
from decimal import Decimal
from pathlib import Path
from typing import Any
from xml.etree import ElementTree as ET

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.investment.dividend_currencies import (
    HEADERS,
    CurrencyCorroboration,
    IssuerDividendAmountCorrection,
    load_currency_corroborations,
    load_dividend_amount_corrections,
)

POLICY = "a" * 64
CAPTURE = "2026-11-08T20:05:42+00:00"
NS = "http://schemas.openxmlformats.org/spreadsheetml/2006/main"


def pin(path: Path) -> dict[str, str]:
    return {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}


def write_json(path: Path, value: Any) -> dict[str, str]:
    path.write_text(json.dumps(value), encoding="utf-8")
    return pin(path)


def captured(path: Path, raw: bytes, *, authority: str = "tax") -> dict[str, Any]:
    path.write_bytes(raw)
    if authority == "tax":
        receipt = {"url": "https://www.ictax.admin.ch/extern/api/security/security.json", "retrieved_at": CAPTURE,
            "bytes": len(raw), "sha256": pin(path)["sha256"],
            "request": {"isin": "US78462F1030", "valorNumber": 45088, "referenceDate": 1577750400000}}
    elif authority == "issuer":
        receipt = {"source_url": "https://www.ssga.com/distributions.xlsx", "retrieved_at_utc": CAPTURE,
            "size_bytes": len(raw), "sha256": pin(path)["sha256"].upper()}
    else:
        receipt = {"url": "https://www.sec.gov/prospectus.htm", "captured_at_utc": CAPTURE,
            "bytes": len(raw), "sha256": pin(path)["sha256"]}
    return {"body": pin(path), "receipt": write_json(path.with_suffix(path.suffix + ".receipt.json"), receipt)}


def workbook(path: Path, *, rate: str = "1.383619", capital_gain: str = "", duplicate: bool = False,
             formula: bool = False, ex_date: str = "09/20/2019") -> bytes:
    strings = list(HEADERS) + ["Fund", "SPY", "78462F103", ex_date, "09/23/2019", "10/31/2019",
                             rate, capital_gain, "", "Quarterly"]
    shared = ET.Element(f"{{{NS}}}sst")
    for text in strings:
        node = ET.SubElement(shared, f"{{{NS}}}si")
        ET.SubElement(node, f"{{{NS}}}t").text = text
    sheet = ET.Element(f"{{{NS}}}worksheet")
    data = ET.SubElement(sheet, f"{{{NS}}}sheetData")
    for number in range(1, 4 if duplicate else 3):
        row = ET.SubElement(data, f"{{{NS}}}row", {"r": str(number)})
        for position, column in enumerate("ABCDEFGHIJ"):
            cell = ET.SubElement(row, f"{{{NS}}}c", {"r": f"{column}{number}", "t": "s"})
            ET.SubElement(cell, f"{{{NS}}}v").text = str(position if number == 1 else position + 10)
            if formula and number == 2 and column == "G":
                ET.SubElement(cell, f"{{{NS}}}f").text = "1+1"
    with zipfile.ZipFile(path, "w") as archive:
        archive.writestr("xl/workbook.xml", f'<workbook xmlns="{NS}" '
            'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships">'
            '<sheets><sheet name="dividend" r:id="rId1"/></sheets></workbook>')
        archive.writestr("xl/_rels/workbook.xml.rels", '<Relationships>'
            '<Relationship Id="rId1" Target="worksheets/sheet1.xml"/></Relationships>')
        archive.writestr("xl/sharedStrings.xml", ET.tostring(shared))
        archive.writestr("xl/worksheets/sheet1.xml", ET.tostring(sheet))
    return path.read_bytes()


def fixture(tmp_path: Path, *, bridge: bool = False) -> tuple[dict[str, Any], dict[str, Any]]:
    original = {"id": "action-one", "symbol": "SPY", "ex_date": "2019-09-20", "rate": 1.383619,
                "special": False, "foreign": False, "cusip": ""}
    amount = "1.383" if bridge else "1.383619"
    tax_body = {"status": "SUCCESS", "error": None, "data": {"calculation": {"isin": "US78462F1030",
        "referenceDate": 1577750400000, "valorNumber": 45088, "quantity": 1, "quotationType": "PIECE"},
        "calculationDetails": {"entries": [{"type": "PAYMENT", "exDate": 1568937600000,
            "value": float(amount), "currency": {"shortName": "USD"}, "nominalQuantity": 1}]}}}
    cells = [""] * 13
    cells[5], cells[8], cells[9] = "20.09.", "USD", amount
    html = '<table><thead><th>Brutto</th></thead><tbody><tr>' + ''.join(f'<td>{v}</td>' for v in cells) + '</tr></tbody></table>'
    html_body = {"status": "SUCCESS", "error": None, "data": {"html": html}}
    issuer = captured(tmp_path / "issuer.xlsx", workbook(tmp_path / "issuer.xlsx"), authority="issuer")
    item = {"method": "issuer_rate_tax_currency_bridge" if bridge else "exact_tax_event", "action_id": original["id"],
        "original_record_sha256": json_sha256(original), "symbol": "SPY", "ex_date": "2019-09-20", "currency": "USD",
        "tax": {"source": captured(tmp_path / "tax.json", json.dumps(tax_body).encode()),
                "gross_context": captured(tmp_path / "gross.json", json.dumps(html_body).encode()),
                "isin": "US78462F1030", "entry_index": 0, "gross_amount": amount},
        "identity": {"kind": "issuer_workbook", "source": issuer, "row": 2, "cusip": "78462F103"},
        "issuer_rate": {"source": issuer, "row": 2} if bridge else None,
        "tax_value_difference": str(Decimal(amount) - Decimal(str(original["rate"])))}
    return original, item


def model(item: dict[str, Any]) -> CurrencyCorroboration:
    return CurrencyCorroboration.model_validate_json(json.dumps(item))


def replace_tax(item: dict[str, Any], change: Any) -> None:
    path = Path(item["tax"]["source"]["body"]["path"])
    payload = json.loads(path.read_text())
    change(payload)
    item["tax"]["source"] = captured(path, json.dumps(payload).encode())


@pytest.mark.parametrize("bridge", [False, True])
def test_currency_only_preserves_original_terms_and_actual_capture(tmp_path: Path, bridge: bool) -> None:
    original, item = fixture(tmp_path, bridge=bridge)
    before = copy.deepcopy(original)
    evidence = model(item)
    refs = evidence.validates_against(original, tmp_path, POLICY)
    assert original == before and "currency" not in original
    assert evidence.evidence == refs and len(refs) == 6
    assert all(r.available_at == datetime.fromisoformat(CAPTURE) and r.retrieved_at == r.available_at for r in refs)
    assert all(r.available_at.date() > date.fromisoformat(original["ex_date"]) for r in refs if r.available_at is not None)
    assert evidence.tax.gross_amount == ("1.383" if bridge else "1.383619")


def test_exact_method_does_not_infer_rounding_or_truncation(tmp_path: Path) -> None:
    original, item = fixture(tmp_path, bridge=True)
    item["method"], item["issuer_rate"] = "exact_tax_event", None
    with pytest.raises(DataReadinessError, match="amount differs"):
        model(item).validates_against(original, tmp_path, POLICY)


def test_bridge_does_not_allow_issuer_provider_rate_conflict(tmp_path: Path) -> None:
    original, item = fixture(tmp_path, bridge=True)
    source = captured(tmp_path / "conflict.xlsx", workbook(tmp_path / "conflict.xlsx", rate="1.383620"), authority="issuer")
    item["issuer_rate"]["source"] = source
    with pytest.raises(DataReadinessError, match="issuer gross distribution rate differs"):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("field,value", [("id", "changed"), ("symbol", "OTHER"), ("ex_date", "2019-09-19"),
    ("rate", 1.38), ("foreign", True), ("special", True), ("currency", "EUR"), ("revision", True)])
def test_original_terms_and_ordinary_flags_cannot_be_repaired(tmp_path: Path, field: str, value: Any) -> None:
    original, item = fixture(tmp_path)
    original[field] = value
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("variant", ["currency", "quantity", "boolean_quantity", "nominal", "isin", "duplicate", "date"])
def test_tax_event_identity_currency_date_and_per_share_basis(tmp_path: Path, variant: str) -> None:
    original, item = fixture(tmp_path)

    def mutate(payload: dict[str, Any]) -> None:
        calculation = payload["data"]["calculation"]
        entry = payload["data"]["calculationDetails"]["entries"][0]
        if variant == "currency":
            entry["currency"]["shortName"] = "CHF"
        elif variant == "quantity":
            calculation["quantity"] = 2
        elif variant == "boolean_quantity":
            calculation["quantity"] = True
        elif variant == "nominal":
            entry["nominalQuantity"] = 2
        elif variant == "isin":
            calculation["isin"] = "US46090E1038"
        elif variant == "duplicate":
            payload["data"]["calculationDetails"]["entries"].append(copy.deepcopy(entry))
        else:
            entry["exDate"] += 86_400_000

    replace_tax(item, mutate)
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("variant", ["body", "receipt", "naive_clock", "future_clock", "wrong_size", "wrong_host"])
def test_source_hash_receipt_and_observed_clock_tampering(tmp_path: Path, variant: str) -> None:
    original, item = fixture(tmp_path)
    source = item["tax"]["source"]
    if variant == "body":
        Path(source["body"]["path"]).write_text("{}")
    elif variant == "receipt":
        Path(source["receipt"]["path"]).write_text("{}")
    else:
        path = Path(source["receipt"]["path"])
        receipt = json.loads(path.read_text())
        if variant == "naive_clock":
            receipt["retrieved_at"] = "2026-11-08T20:05:42"
        elif variant == "future_clock":
            receipt["retrieved_at"] = (datetime.fromisoformat(CAPTURE) + timedelta(days=36500)).isoformat()
        elif variant == "wrong_size":
            receipt["bytes"] += 1
        else:
            receipt["url"] = "https://example.com/security.json"
        source["receipt"] = write_json(path, receipt)
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("variant", ["duplicate", "formula", "gains"])
def test_workbook_ambiguous_rows_formulas_and_capital_gains_refused(tmp_path: Path, variant: str) -> None:
    original, item = fixture(tmp_path, bridge=True)
    path = tmp_path / "unsafe.xlsx"
    source = captured(path, workbook(path, duplicate=variant == "duplicate", formula=variant == "formula",
        capital_gain="0.1" if variant == "gains" else ""), authority="issuer")
    item["issuer_rate"]["source"] = source
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


def test_official_document_identity_quotes_are_source_bound(tmp_path: Path) -> None:
    original, item = fixture(tmp_path)
    quote = 'Market Symbol: Fund shares trade as “SPY.” CUSIP: 78462F103'
    source = captured(tmp_path / "prospectus.html", ('<p>' + quote + '</p>').encode(), authority="sec")
    item["identity"] = {"kind": "official_document", "source": source, "cusip": "78462F103",
        "fund_name": "Fund shares", "quotes": [quote]}
    assert model(item).validates_against(original, tmp_path, POLICY)
    item["identity"]["quotes"] = [quote.replace("SPY", "OTHER")]
    with pytest.raises(DataReadinessError, match="quoted ticker/CUSIP/fund identity"):
        model(item).validates_against(original, tmp_path, POLICY)


def test_isin_must_derive_from_source_cusip(tmp_path: Path) -> None:
    original, item = fixture(tmp_path)
    item["tax"]["isin"] = "US78462F1031"
    with pytest.raises(DataReadinessError, match="CUSIP/ISIN identity"):
        model(item).validates_against(original, tmp_path, POLICY)


def special_cash_fixture(tmp_path: Path, *, notice_replace: tuple[str, str] | None = None) -> tuple[dict[str, Any], dict[str, Any]]:
    """Synthetic alternate fund demonstrates source-driven, reusable admission."""
    from pypdf import PdfWriter
    from pypdf.generic import DecodedStreamObject, DictionaryObject, NameObject

    original, item = fixture(tmp_path)
    original.update(special=True, cusip="78462F103", record_date="2019-09-23", payable_date="2019-10-02")
    item["original_record_sha256"] = json_sha256(original)
    text = ("#12345 Fund (SPY) has announced a Special Cash Dividend of $1.383619 per SPY share. "
        "The record date is September 23, 2019; the payable date is October 1, 2019. "
        "The ex-distribution date for this distribution will be September 20, 2019. CUSIP: 78462F103")
    if notice_replace is not None:
        text = text.replace(*notice_replace)
    writer = PdfWriter()
    page = writer.add_blank_page(width=612, height=792)
    page[NameObject("/Resources")] = DictionaryObject({NameObject("/Font"): DictionaryObject({
        NameObject("/F1"): DictionaryObject({NameObject("/Type"): NameObject("/Font"),
            NameObject("/Subtype"): NameObject("/Type1"), NameObject("/BaseFont"): NameObject("/Helvetica")})})})
    stream = DecodedStreamObject()
    escaped = text.replace("\\", "\\\\").replace("(", "\\(").replace(")", "\\)")
    stream.set_data(("BT /F1 10 Tf 10 700 Td (" + escaped + ") Tj ET").encode("ascii"))
    page[NameObject("/Contents")] = writer._add_object(stream)
    output = io.BytesIO()
    writer.write(output)
    raw = output.getvalue()
    digest = hashlib.sha256(raw).hexdigest()
    path = tmp_path / f"body-{digest}.bin"
    path.write_bytes(raw)
    receipt = {"schema_version": "market_predictor.official_document_attempt", "inventory_sha256": "a" * 64,
        "document_id": "test_special_cash", "attempt_id": "b" * 32, "started_at_utc": CAPTURE,
        "completed_at_utc": CAPTURE, "state": "archived_unreviewed", "body_path": path.name,
        "error_type": None, "failure_status_code": None, "interpretation_status": "not_reviewed",
        "historical_availability_proven": False, "accounting_eligible": False,
        "response": {"requested_url": "https://infomemo.theocc.com/infomemos?number=12345",
            "final_url": "https://infomemo.theocc.com/infomemos?number=12345", "redirect_chain": [],
            "status_code": 200, "retrieved_at_utc": CAPTURE, "content_type": "application/pdf",
            "content_encoding": None, "etag": None, "last_modified": None, "body_length": len(raw),
            "sha256": digest, "body_representation": "http_entity_encoded", "safe_headers": []}}
    receipt["receipt_sha256"] = json_sha256(receipt)
    source = {"body": pin(path), "receipt": write_json(tmp_path / "_receipt.json", receipt)}
    item["special_cash_distribution"] = {"kind": "occ_special_cash_distribution", "source": source,
        "notice_number": 12345, "fund_name": "Fund", "record_date": "2019-09-23",
        "announced_payable_date": "2019-10-01", "provider_payable_date": "2019-10-02",
        "payable_date_conflict": True, "economic_treatment": "nominal_unpaid_nontradable_entitlement"}
    return original, item


def test_special_cash_source_preserves_raw_class_and_payable_conflict(tmp_path: Path) -> None:
    original, item = special_cash_fixture(tmp_path)
    before = copy.deepcopy(original)
    proof = model(item)
    refs = proof.validates_against(original, tmp_path, POLICY)
    assert original == before and original["special"] is True
    assert len(refs) == 8
    assert sum("payable_date_conflict=True" in ref.record_locator for ref in refs) == 2
    assert proof.special_cash_distribution is not None
    assert proof.special_cash_distribution.announced_payable_date != proof.special_cash_distribution.provider_payable_date


@pytest.mark.parametrize("change", [{"foreign": True}, {"special": False}, {"rate": 1.38},
    {"record_date": "2019-09-24"}, {"payable_date": "2019-10-03"}, {"revision": True},
    {"cancelled": True}, {"due_bill_redemption_date": "2019-10-03"}, {"due_bill_redemption_date": False}])
def test_special_cash_cannot_repair_unsupported_or_conflicting_terms(tmp_path: Path, change: dict[str, Any]) -> None:
    original, item = special_cash_fixture(tmp_path)
    original.update(change)
    # Rebinding a declaration cannot authorize changed economic terms.
    item["original_record_sha256"] = json_sha256(original)
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("field,value", [("notice_number", 12346), ("fund_name", "Other"),
    ("payable_date_conflict", False), ("announced_payable_date", "2019-10-03")])
def test_special_cash_assertion_fields_must_replay_source(tmp_path: Path, field: str, value: Any) -> None:
    original, item = special_cash_fixture(tmp_path)
    item["special_cash_distribution"][field] = value
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


def test_special_cash_requires_separate_exact_usd_proof_and_source_hash(tmp_path: Path) -> None:
    original, item = special_cash_fixture(tmp_path)
    replace_tax(item, lambda payload: payload["data"]["calculationDetails"]["entries"][0]["currency"].update(shortName="EUR"))
    with pytest.raises(DataReadinessError, match="tax currency"):
        model(item).validates_against(original, tmp_path, POLICY)
    Path(item["special_cash_distribution"]["source"]["body"]["path"]).write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="hash differs"):
        model(item).validates_against(original, tmp_path, POLICY)


def test_special_cash_does_not_enable_ordinary_or_issuer_bridge_fallback(tmp_path: Path) -> None:
    original, item = special_cash_fixture(tmp_path)
    admission = item.pop("special_cash_distribution")
    with pytest.raises(DataReadinessError, match="original record binding"):
        model(item).validates_against(original, tmp_path, POLICY)
    item.update(special_cash_distribution=admission, method="issuer_rate_tax_currency_bridge",
        issuer_rate={"source": item["identity"]["source"], "row": 2})
    with pytest.raises(ValidationError, match="requires exact tax"):
        model(item)


@pytest.mark.parametrize("replacement", [("$1.383619", "$1.383618"), ("CUSIP: 78462F103", "CUSIP: 78462F104"),
    ("September 20, 2019.", "September 19, 2019."), ("September 23, 2019;", "September 24, 2019;"),
    ("Special Cash Dividend", "Capital Gain Distribution"), ("CUSIP:", "Due-bill period applies. CUSIP:")])
def test_special_cash_notice_cannot_supply_conflicting_or_different_distribution(tmp_path: Path, replacement: tuple[str, str]) -> None:
    original, item = special_cash_fixture(tmp_path, notice_replace=replacement)
    with pytest.raises(DataReadinessError):
        model(item).validates_against(original, tmp_path, POLICY)


def test_net_or_taxable_column_cannot_supply_gross_currency_amount(tmp_path: Path) -> None:
    original, item = fixture(tmp_path)
    source = item["tax"]["gross_context"]
    path = Path(source["body"]["path"])
    payload = json.loads(path.read_text())
    payload["data"]["html"] = payload["data"]["html"].replace("Brutto", "Taxable income CHF")
    item["tax"]["gross_context"] = captured(path, json.dumps(payload).encode())
    with pytest.raises(DataReadinessError, match="semantic context"):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("poison", ['{"schema":1,"schema":2}', '{"amount":NaN}'])
def test_duplicate_json_and_nonfinite_values_refuse_before_typed_loading(tmp_path: Path, poison: str) -> None:
    path = tmp_path / "poison.json"
    path.write_text(poison)
    with pytest.raises(DataReadinessError):
        load_currency_corroborations(tmp_path, path, pin(path)["sha256"], policy_sha256=POLICY)


def test_manifest_is_closed_hash_bound_and_duplicate_actions_fail(tmp_path: Path) -> None:
    _, item = fixture(tmp_path)
    path = tmp_path / "manifest.json"
    content = {"schema": "market_predictor.investment_dividend_currency_corroborations", "corroborations": [item]}
    reference = write_json(path, content)
    loaded = load_currency_corroborations(tmp_path, path, reference["sha256"], policy_sha256=POLICY)
    assert set(loaded) == {"action-one"}
    with pytest.raises(DataReadinessError, match="hash differs"):
        load_currency_corroborations(tmp_path, path, "b" * 64, policy_sha256=POLICY)
    content["corroborations"].append(item)
    reference = write_json(path, content)
    with pytest.raises(DataReadinessError, match="repeats action ID"):
        load_currency_corroborations(tmp_path, path, reference["sha256"], policy_sha256=POLICY)
    content["corroborations"] = [dict(item, fake_provider_response={"currency": "USD"})]
    reference = write_json(path, content)
    with pytest.raises(ValidationError):
        load_currency_corroborations(tmp_path, path, reference["sha256"], policy_sha256=POLICY)


def correction_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any]]:
    """Synthetic unit source with the observed conflict's exact numerical values."""
    original, currency = fixture(tmp_path, bridge=True)
    original.update(id="a6587d80-e08a-4b1a-aa94-857549880c59", ex_date="2020-06-19", rate=1.3662419, cusip="78462F103")
    path = tmp_path / "corrected-issuer.xlsx"
    issuer = captured(path, workbook(path, rate="1.366242", ex_date="06/19/2020"), authority="issuer")
    currency.update(action_id=original["id"], original_record_sha256=json_sha256(original), ex_date=original["ex_date"],
        identity={"kind": "issuer_workbook", "source": issuer, "row": 2, "cusip": "78462F103"},
        issuer_rate={"source": issuer, "row": 2}, tax_value_difference="-0.0002419",
        tax_resolved_value_difference="-0.000242")
    currency["tax"]["gross_amount"] = "1.366"

    def update_event(payload: dict[str, Any]) -> None:
        entry = payload["data"]["calculationDetails"]["entries"][0]
        entry.update(exDate=1592524800000, value=1.366)

    replace_tax(currency, update_event)
    html = json.loads(Path(currency["tax"]["gross_context"]["body"]["path"]).read_text())
    html["data"]["html"] = html["data"]["html"].replace("20.09.", "19.06.").replace("1.383", "1.366")
    currency["tax"]["gross_context"] = captured(tmp_path / "corrected-gross.json", json.dumps(html).encode())
    correction = {"method": "issuer_gross_amount_correction", "action_id": original["id"],
        "original_record_sha256": json_sha256(original), "symbol": original["symbol"], "ex_date": original["ex_date"],
        "cusip": "78462F103", "isin": "US78462F1030", "original_rate": "1.3662419",
        "corrected_gross_amount": "1.366242", "issuer_rate": {"source": issuer, "row": 2}}
    return original, currency, correction


def correction_model(item: dict[str, Any]) -> IssuerDividendAmountCorrection:
    return IssuerDividendAmountCorrection.model_validate_json(json.dumps(item))


def test_explicit_issuer_correction_preserves_raw_amount_and_supplies_no_currency(tmp_path: Path) -> None:
    original, _, item = correction_fixture(tmp_path)
    saved = copy.deepcopy(original)
    resolved = correction_model(item).resolve_against(original, tmp_path, POLICY)
    assert resolved.original_rate == Decimal("1.3662419")
    assert resolved.gross_amount == Decimal("1.366242")
    assert resolved.original_record_sha256 == json_sha256(saved)
    assert original == saved and "currency" not in resolved.model_dump()
    assert all(r.available_at == datetime.fromisoformat(CAPTURE) for r in resolved.evidence)
    assert all("dividend!A2:J2" == r.record_locator for r in resolved.evidence)


def test_corrected_amount_and_usd_proof_are_separate_and_keep_both_tax_comparisons(tmp_path: Path) -> None:
    original, currency, item = correction_fixture(tmp_path)
    saved = copy.deepcopy(original)
    proof = model(currency)
    refs = proof.validates_against(original, tmp_path, POLICY, amount_correction=correction_model(item))
    assert len(refs) == 6 and original == saved
    assert proof.tax_value_difference == "-0.0002419"
    assert proof.tax_resolved_value_difference == "-0.000242"
    replace_tax(currency, lambda p: p["data"]["calculationDetails"]["entries"][0]["currency"].update(shortName="CHF"))
    with pytest.raises(DataReadinessError, match="tax currency"):
        model(currency).validates_against(original, tmp_path, POLICY, amount_correction=correction_model(item))


def test_old_currency_bridge_still_rejects_amount_conflict_without_explicit_correction(tmp_path: Path) -> None:
    original, currency, _ = correction_fixture(tmp_path)
    currency.pop("tax_resolved_value_difference")
    with pytest.raises(DataReadinessError, match="issuer gross distribution rate differs"):
        model(currency).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("variant", ["missing", "original_difference_reused", "provider_difference_changed", "unused"])
def test_resolved_and_original_tax_comparisons_cannot_be_reinterpreted(tmp_path: Path, variant: str) -> None:
    original, currency, item = correction_fixture(tmp_path)
    correction = correction_model(item)
    if variant == "missing":
        currency.pop("tax_resolved_value_difference")
    elif variant == "original_difference_reused":
        currency["tax_resolved_value_difference"] = currency["tax_value_difference"]
    elif variant == "provider_difference_changed":
        currency["tax_value_difference"] = currency["tax_resolved_value_difference"]
    with pytest.raises(DataReadinessError):
        model(currency).validates_against(original, tmp_path, POLICY,
            amount_correction=None if variant == "unused" else correction)


def test_resolved_comparison_failure_remains_fatal_when_gross_row_is_also_unavailable(tmp_path: Path) -> None:
    original, currency, item = correction_fixture(tmp_path)
    currency["tax_resolved_value_difference"] = currency["tax_value_difference"]
    path = Path(currency["tax"]["gross_context"]["body"]["path"])
    payload = json.loads(path.read_text())
    payload["data"]["html"] = payload["data"]["html"].replace("<td>USD</td>", "<td>EUR</td>")
    currency["tax"]["gross_context"] = captured(path, json.dumps(payload).encode())
    with pytest.raises(DataReadinessError, match="tax-minus-resolved"):
        model(currency).validates_against(original, tmp_path, POLICY, amount_correction=correction_model(item))


@pytest.mark.parametrize("field,value", [("original_record_sha256", "b" * 64), ("action_id", "other-action"),
    ("symbol", "OTHER"), ("ex_date", "2020-06-18"), ("original_rate", "1.3662418"),
    ("corrected_gross_amount", "1.3662419999999999999999"), ("isin", "US78462F1031"), ("cusip", "46090E103")])
def test_correction_requires_exact_original_identity_rate_and_issuer_amount(tmp_path: Path, field: str, value: str) -> None:
    original, _, item = correction_fixture(tmp_path)
    item[field] = value
    with pytest.raises(DataReadinessError):
        correction_model(item).resolve_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("field,value", [("special", True), ("foreign", True), ("revision", True),
    ("due_bill_redemption_date", "2020-06-20"), ("currency", "EUR")])
def test_even_rebound_correction_cannot_repair_unsupported_original_terms(tmp_path: Path, field: str, value: Any) -> None:
    original, _, item = correction_fixture(tmp_path)
    original[field] = value
    item["original_record_sha256"] = json_sha256(original)
    with pytest.raises(DataReadinessError):
        correction_model(item).resolve_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("value", ["1.3662419", "0", "-1", "NaN", "Infinity", 1.366242])
def test_correction_is_positive_decimal_string_and_must_change_amount(tmp_path: Path, value: Any) -> None:
    _, _, item = correction_fixture(tmp_path)
    item["corrected_gross_amount"] = value
    with pytest.raises((ValidationError, DataReadinessError)):
        correction_model(item)


@pytest.mark.parametrize("variant", ["body_tamper", "receipt_tamper", "future", "wrong_row", "gains", "duplicate"])
def test_correction_replays_pinned_issuer_source_and_actual_clock(tmp_path: Path, variant: str) -> None:
    original, _, item = correction_fixture(tmp_path)
    source = item["issuer_rate"]["source"]
    if variant == "body_tamper":
        Path(source["body"]["path"]).write_bytes(b"wrong workbook")
    elif variant == "receipt_tamper":
        Path(source["receipt"]["path"]).write_text("{}")
    elif variant == "future":
        path = Path(source["receipt"]["path"])
        receipt = json.loads(path.read_text())
        receipt["retrieved_at_utc"] = "2126-11-08T20:05:42+00:00"
        source["receipt"] = write_json(path, receipt)
    elif variant == "wrong_row":
        item["issuer_rate"]["row"] = 3
    else:
        path = tmp_path / "unsupported-issuer.xlsx"
        item["issuer_rate"]["source"] = captured(path, workbook(path, rate="1.366242", ex_date="06/19/2020",
            capital_gain="0.1" if variant == "gains" else "", duplicate=variant == "duplicate"), authority="issuer")
    with pytest.raises(DataReadinessError):
        correction_model(item).resolve_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("conflicting", [False, True])
def test_duplicate_or_conflicting_amount_corrections_for_same_action_refused(tmp_path: Path, conflicting: bool) -> None:
    _, _, item = correction_fixture(tmp_path)
    second = copy.deepcopy(item)
    if conflicting:
        second["corrected_gross_amount"] = "1.366243"
    path = tmp_path / "amount-corrections.json"
    reference = write_json(path, {"schema": "market_predictor.investment_dividend_amount_corrections",
        "corrections": [item, second]})
    with pytest.raises(DataReadinessError, match="repeats action ID"):
        load_dividend_amount_corrections(tmp_path, path, reference["sha256"], policy_sha256=POLICY)


def test_correction_manifest_uses_separate_closed_hash_bound_schema(tmp_path: Path) -> None:
    original, _, item = correction_fixture(tmp_path)
    path = tmp_path / "amount-corrections.json"
    payload = {"schema": "market_predictor.investment_dividend_amount_corrections", "corrections": [item]}
    reference = write_json(path, payload)
    loaded = load_dividend_amount_corrections(tmp_path, path, reference["sha256"], policy_sha256=POLICY)
    assert loaded[original["id"]].resolve_against(original, tmp_path, POLICY).gross_amount == Decimal("1.366242")
    with pytest.raises(DataReadinessError, match="hash differs"):
        load_dividend_amount_corrections(tmp_path, path, "b" * 64, policy_sha256=POLICY)
    with pytest.raises(ValidationError):
        load_currency_corroborations(tmp_path, path, reference["sha256"], policy_sha256=POLICY)
    payload["corrections"][0]["provider_response"] = {"currency": "USD", "rate": 1.366242}
    reference = write_json(path, payload)
    with pytest.raises(ValidationError):
        load_dividend_amount_corrections(tmp_path, path, reference["sha256"], policy_sha256=POLICY)


@pytest.mark.parametrize("variant", ["valid", "missing", "competing", "wrong_session", "wrong_ticker", "cusip",
    "tax_category", "tax_preferred", "tax_group", "tax_name", "lookup_isin", "tax_name_preferred", "tax_name_adr"])
def test_stock_composite_requires_exact_dated_class_and_provider_cusip(tmp_path, monkeypatch, variant):
    from types import SimpleNamespace

    from market_predictor.investment import stock_identity
    from market_predictor.swing.contracts.holding_accounting import EvidenceReference

    original, item = fixture(tmp_path)
    original["cusip"] = "78462F103"
    item["original_record_sha256"] = json_sha256(original)
    authority = write_json(tmp_path / "stock-authority.json", {"synthetic_test_only": True})
    proof = {"security_id": "stock:one", "ticker": "WRONG" if variant == "wrong_ticker" else "SPY",
        "sec_cik": "0000000001", "security_title": "Common stock", "identity_authority": authority,
        "membership_authority": authority, "anchor": {"collection_manifest": authority, "receipt_artifact": authority,
            "unit_id": "test-only-unit", "body_sha256": authority["sha256"], "filing_metadata": authority}}
    proof_pin = write_json(tmp_path / "proof.json", proof)
    item["identity"] = {"kind": "stock_composite", "cusip": "78462F103", "proof": proof_pin,
        "cusip_provenance": "provider_original_cash_action"}
    ref = EvidenceReference(reference=authority["path"], artifact_sha256=authority["sha256"], record_locator="test-only SEC class",
        interpretation_policy_sha256=POLICY, retrieved_at=datetime.fromisoformat(CAPTURE), available_at=datetime.fromisoformat(CAPTURE))
    result = SimpleNamespace(issuer_name="SPY Corporation", security_title="Common stock", evidence=(ref,),
        missing_reasons=("dated_relation_missing",) if variant == "missing" else (),
        competing_class_tickers=("OTHER",) if variant == "competing" else (),
        proven_sessions=() if variant == "wrong_session" else (date.fromisoformat(original["ex_date"]),))
    monkeypatch.setattr(stock_identity, "replay_stock_identity", lambda *args, **kwargs: result)
    replace_tax(item, lambda payload: payload["data"]["calculation"].update(name="SPY Corp., Namenaktien, US", instrumentGroup="SHARE"))
    item["tax"]["lookup"] = captured(tmp_path / "lookup.json", json.dumps({"total": 1, "count": 1, "security": [{
        "isin": "US78462F1030", "vn": 45088, "institution": "SPY Corp.", "name": "Common shares",
        "categoryName": "SHARE", "typeName": "SHARE.COMMON", "typeDescription": "Ordinary shares"}]}).encode())
    if variant in ("tax_group", "tax_name"):
        replace_tax(item, lambda payload: payload["data"]["calculation"].update(
            **({"instrumentGroup": "BOND"} if variant == "tax_group" else {"name": ""})))
    if variant in ("tax_name_preferred", "tax_name_adr"):
        replace_tax(item, lambda payload: payload["data"]["calculation"].update(
            name="SPY Corp. Preferred Stock" if variant == "tax_name_preferred" else "SPY Corp. ADR"))
    if variant in ("tax_category", "tax_preferred", "lookup_isin"):
        payload = json.loads(Path(item["tax"]["lookup"]["body"]["path"]).read_text())
        changes = {"tax_category": {"categoryName": "BOND"}, "tax_preferred": {"typeName": "SHARE.PREFERRED"},
            "lookup_isin": {"isin": "US46090E1038"}}
        payload["security"][0].update(changes[variant])
        item["tax"]["lookup"] = captured(tmp_path / "modified-lookup.json", json.dumps(payload).encode())
    if variant == "cusip":
        original["cusip"] = ""
        item["original_record_sha256"] = json_sha256(original)
    if variant == "valid":
        refs = model(item).validates_against(original, tmp_path, POLICY, stock_identity_cache={})
        assert any("provider-only CUSIP=78462F103" in r.record_locator for r in refs)
    else:
        candidate = model(item)
        with pytest.raises(DataReadinessError):
            candidate.validates_against(original, tmp_path, POLICY)
        if variant in ("tax_category", "tax_preferred"):
            assert candidate.identity.tax_identity_observation["tax_category_name"] == (
                "BOND" if variant == "tax_category" else "SHARE")
            assert candidate.identity.tax_identity_observation["tax_type_name"] == (
                "SHARE.PREFERRED" if variant == "tax_preferred" else "SHARE.COMMON")


def test_currency_manifest_explicit_scale_bound_is_separate_from_receipts(tmp_path, monkeypatch):
    from market_predictor.investment import dividend_currencies

    _, item = fixture(tmp_path)
    path = tmp_path / "scale-manifest.json"
    raw = (" " * (4 * 1024**2)) + json.dumps({"schema": dividend_currencies.SCHEMA, "corroborations": [item]})
    path.write_text(raw, encoding="utf-8")
    assert set(load_currency_corroborations(tmp_path, path, pin(path)["sha256"], policy_sha256=POLICY)) == {"action-one"}
    monkeypatch.setattr(dividend_currencies, "MAX_CURRENCY_MANIFEST_BYTES", 1024)
    with pytest.raises(DataReadinessError, match="byte bound"):
        load_currency_corroborations(tmp_path, path, pin(path)["sha256"], policy_sha256=POLICY)
    assert dividend_currencies.MAX_MANIFEST_BYTES == 4 * 1024**2

def purchase_offer_fixture(tmp_path: Path) -> tuple[dict[str, Any], dict[str, Any], dict[str, Any], list[str]]:
    original, item = fixture(tmp_path)
    paid = 1562803200000  # 2019-07-11T00:00:00Z
    entry = {"type": "PAYMENT", "name": "calculation:payment_title",
        "currency": {"categoryShortName": "CURREN", "shortName": "USD"}, "nominalQuantity": 1,
        "date": paid, "exDate": None, "paymentDate": paid, "variant": None, "percent": None,
        "value": 0, "valueTotal": 0, "exchangeRate": 0.882125, "valueChf": 0, "valueTotalChf": 0, "taxValueChf": None,
        "withHoldingTax": None, "paymentValueTotalChfWithHoldingTax": 0, "valuePrefix": None,
        "paymentValueTotalChfWithoutHoldingTax": 0, "remark": None, "sign": ""}
    row = ["11.07.2019 Kaufofferte der Example Holdings LLC zum Preis von USD 65.75 pro Aktie",
        "", "", "", "", "", "11.07.", "", "USD", "0.00", "", "0.00", ""]
    return original, item, entry, row


def add_purchase_offer(tmp_path: Path, item: dict[str, Any], entries: list[dict[str, Any]], rows: list[list[str]],
                       *, separate_table: bool = False) -> None:
    # The matched dividend retains its actual index in the original structured entries.
    replace_tax(item, lambda payload: payload["data"]["calculationDetails"]["entries"].__setitem__(slice(0, 0), entries))
    item["tax"]["entry_index"] += len(entries)
    gross_path = Path(item["tax"]["gross_context"]["body"]["path"])
    gross = json.loads(gross_path.read_bytes())
    markup = "".join("<tr>" + "".join(f"<td>{value}</td>" for value in row) + "</tr>" for row in rows)
    if separate_table:
        gross["data"]["html"] += f"<table><thead><th>Brutto</th></thead><tbody>{markup}</tbody></table>"
    else:
        gross["data"]["html"] = gross["data"]["html"].replace("</tbody>", markup + "</tbody>")
    item["tax"]["gross_context"] = captured(tmp_path / "gross-offer.json", json.dumps(gross).encode())


def test_exact_retained_purchase_offer_shape_preserves_target_index_and_amount(tmp_path: Path) -> None:
    original, item, entry, row = purchase_offer_fixture(tmp_path)
    add_purchase_offer(tmp_path, item, [entry], [row])
    assertion = model(item)
    refs = assertion.validates_against(original, tmp_path, POLICY)
    assert assertion.tax.entry_index == 1 and assertion.tax.gross_amount == "1.383619"
    assert any("/entries/1/currency" in ref.record_locator for ref in refs)
    assert any("zero-purchase-offer-entry-to-gross-row=0:1" in ref.record_locator for ref in refs)
    item["tax"]["entry_index"] = 0
    with pytest.raises(DataReadinessError, match="absent or ambiguous"):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("poison", [
    "missing_ex_date", "extra_field", "nonzero", "string_zero", "boolean_zero", "foreign", "quantity",
    "different_payment_date", "invalid_midnight", "outside_date_range", "undated_positive", "invalid_other_date", "duplicate_structured",
    "missing_gross", "duplicate_gross", "ambiguous_sibling", "wrong_wording", "blank_offeror",
    "wrong_full_date", "zero_offer_price", "gross_nonzero", "chf_nonzero",
])
def test_malformed_or_unpaired_purchase_offer_keeps_all_dates_strict(tmp_path: Path, poison: str) -> None:
    original, item, entry, row = purchase_offer_fixture(tmp_path)
    entries, rows = [entry], [row]
    match = "not UTC midnight"
    if poison == "missing_ex_date":
        entry.pop("exDate")
    elif poison == "extra_field":
        entry["paidAmount"] = 1
    elif poison == "nonzero":
        entry["paymentValueTotalChfWithoutHoldingTax"] = 1
    elif poison == "string_zero":
        entry["value"] = "0"
    elif poison == "boolean_zero":
        entry["value"] = False
    elif poison == "foreign":
        entry["currency"]["shortName"] = "CHF"
    elif poison == "quantity":
        entry["nominalQuantity"] = True
    elif poison == "different_payment_date":
        entry["paymentDate"] += 86_400_000
    elif poison == "invalid_midnight":
        entry["date"] += 1
        entry["paymentDate"] += 1
    elif poison == "outside_date_range":
        entry["date"] = entry["paymentDate"] = 253402300800000
    elif poison == "undated_positive":
        entries.append({"type": "PAYMENT", "exDate": None, "value": 1})
    elif poison == "invalid_other_date":
        entries.append({"type": "PAYMENT", "exDate": 1568937600001, "value": 1})
    elif poison == "duplicate_structured":
        entries.append(copy.deepcopy(entry))
    elif poison == "missing_gross":
        rows.clear()
    elif poison == "duplicate_gross":
        rows.append(list(row))
    elif poison == "ambiguous_sibling":
        rows.append(["unknown entry", "", "", "", "", "", row[6], "", "USD", "10.00", "", "9.00", ""])
    elif poison == "wrong_wording":
        row[0] += " bezahlt"
    elif poison == "blank_offeror":
        row[0] = row[0].replace("Example Holdings LLC", "")
    elif poison == "wrong_full_date":
        row[0] = row[0].replace("2019", "2018")
    elif poison == "zero_offer_price":
        row[0] = row[0].replace("65.75", "0.00")
    elif poison == "gross_nonzero":
        row[9] = "0.01"
    else:
        row[11] = "0.01"
    add_purchase_offer(tmp_path, item, entries, rows)
    with pytest.raises(DataReadinessError, match=match):
        model(item).validates_against(original, tmp_path, POLICY)


def test_purchase_offer_must_share_target_gross_table_and_request(tmp_path: Path) -> None:
    original, item, entry, row = purchase_offer_fixture(tmp_path)
    add_purchase_offer(tmp_path, item, [entry], [row], separate_table=True)
    with pytest.raises(DataReadinessError, match="outside the target gross table"):
        model(item).validates_against(original, tmp_path, POLICY)
    receipt_path = Path(item["tax"]["gross_context"]["receipt"]["path"])
    receipt = json.loads(receipt_path.read_bytes())
    receipt["request"]["valorNumber"] = 123
    item["tax"]["gross_context"]["receipt"] = write_json(receipt_path, receipt)
    with pytest.raises(DataReadinessError, match="structured/HTML request identity differs"):
        model(item).validates_against(original, tmp_path, POLICY)


@pytest.mark.parametrize("nested", ["offer_cells", "brutto_header"])
def test_nested_markup_cannot_certify_purchase_offer_table_ownership(tmp_path: Path, nested: str) -> None:
    original, item, entry, row = purchase_offer_fixture(tmp_path)
    add_purchase_offer(tmp_path, item, [entry], [row])
    gross_path = Path(item["tax"]["gross_context"]["body"]["path"])
    gross = json.loads(gross_path.read_bytes())
    html = gross["data"]["html"]
    if nested == "offer_cells":
        html = html.replace(f"<tr><td>{row[0]}</td>", f"<tr><table><td>{row[0]}</td>")
        html = html.replace("</tr></tbody>", "</table></tr></tbody>")
    else:
        html = html.replace("<th>Brutto</th>", "<th><table>Brutto</table></th>")
    gross["data"]["html"] = html
    item["tax"]["gross_context"] = captured(gross_path, json.dumps(gross).encode())
    with pytest.raises(DataReadinessError, match="outside the target gross table"):
        model(item).validates_against(original, tmp_path, POLICY)


def _exact_provider_payment(tmp_path: Path, original: dict[str, Any], item: dict[str, Any]) -> None:
    original.update(payable_date="2019-10-31", record_date="2019-09-23")
    item["original_record_sha256"] = json_sha256(original)
    paid = int(datetime.fromisoformat("2019-10-31T00:00:00+00:00").timestamp() * 1000)
    replace_tax(item, lambda payload: payload["data"]["calculationDetails"]["entries"][0].update(
        date=paid, paymentDate=paid))
    path = Path(item["tax"]["gross_context"]["body"]["path"])
    payload = json.loads(path.read_bytes())
    payload["data"]["html"] = payload["data"]["html"].replace("<td>20.09.</td><td></td>", "<td>20.09.</td><td>31.10.</td>")
    item["tax"]["gross_context"] = captured(tmp_path / "gross-exact-payment.json", json.dumps(payload).encode())


@pytest.mark.parametrize("variant", [
    "valid", "class_lookup", "class_calculation", "issuer_lookup", "issuer_calculation",
    "tax_lookup_endpoint", "capital_stock", "wrong_provider_payable", "unrelated_gross_table",
])
def test_competing_common_stock_requires_independently_matched_class_issuer_and_dates(tmp_path, monkeypatch, variant):
    from types import SimpleNamespace

    from market_predictor.investment import stock_identity
    from market_predictor.swing.contracts.holding_accounting import EvidenceReference

    original, item = fixture(tmp_path)
    original["cusip"] = "78462F103"
    _exact_provider_payment(tmp_path, original, item)
    if variant == "wrong_provider_payable":
        original["payable_date"] = "2019-11-01"
        item["original_record_sha256"] = json_sha256(original)
    if variant == "unrelated_gross_table":
        path = Path(item["tax"]["gross_context"]["body"]["path"])
        gross = json.loads(path.read_bytes())
        gross["data"]["html"] = gross["data"]["html"].replace(
            "</thead><tbody>", "</thead></table><table><tbody>")
        item["tax"]["gross_context"] = captured(path, json.dumps(gross).encode())
    authority = write_json(tmp_path / "official.json", {"synthetic_test_only": True})
    title = "Class B Common Stock, par value $0.01 per share"
    if variant == "capital_stock":
        title = "Class B Capital Stock, par value $0.01 per share"
    proof = {"security_id": "stock:one", "ticker": "SPY", "sec_cik": "0000000001", "security_title": title,
        "identity_authority": authority, "membership_authority": authority, "anchor": {"collection_manifest": authority,
            "receipt_artifact": authority, "unit_id": "test-only-unit", "body_sha256": authority["sha256"],
            "filing_metadata": authority}}
    item["identity"] = {"kind": "stock_composite", "cusip": "78462F103",
        "proof": write_json(tmp_path / "proof.json", proof), "cusip_provenance": "provider_original_cash_action"}
    ref = EvidenceReference(reference=authority["path"], artifact_sha256=authority["sha256"],
        record_locator="test-only SEC class", interpretation_policy_sha256=POLICY,
        retrieved_at=datetime.fromisoformat(CAPTURE), available_at=datetime.fromisoformat(CAPTURE))
    result = SimpleNamespace(issuer_name="EXAMPLE INC.", security_title=title, evidence=(ref,), missing_reasons=(),
        competing_class_tickers=("OTHER",), proven_sessions=(date.fromisoformat(original["ex_date"]),))
    monkeypatch.setattr(stock_identity, "replay_stock_identity", lambda *args, **kwargs: result)
    calc_label = "Namenaktien A" if variant == "class_calculation" else "Namenaktien B"
    calc_issuer = "Other Inc." if variant == "issuer_calculation" else "Example Inc."
    replace_tax(item, lambda payload: payload["data"]["calculation"].update(
        name=f"{calc_issuer}, {calc_label}, US", instrumentGroup="SHARE"))
    lookup = {"total": 1, "count": 1, "security": [{
        "isin": "US78462F1030", "vn": 45088, "institution": "Example Inc" if variant == "issuer_lookup" else "Example Inc.",
        "name": "Namenaktien A" if variant == "class_lookup" else "Namenaktien B",
        "categoryName": "SHARE", "typeName": "SHARE.NOMINAL", "typeDescription": "Registered shares"}]}
    item["tax"]["lookup"] = captured(tmp_path / "lookup.json", json.dumps(lookup).encode())
    if variant != "tax_lookup_endpoint":
        path = Path(item["tax"]["lookup"]["receipt"]["path"])
        receipt = json.loads(path.read_bytes())
        receipt["url"] = "https://www.ictax.admin.ch/lsi/api/security"
        item["tax"]["lookup"]["receipt"] = write_json(path, receipt)
    if variant != "valid":
        match = "payment or record date differs" if variant == "unrelated_gross_table" else None
        with pytest.raises(DataReadinessError, match=match):
            model(item).validates_against(original, tmp_path, POLICY)
        return
    refs = model(item).validates_against(original, tmp_path, POLICY)
    assert any("SEC common-stock class=B" in r.record_locator for r in refs)
    assert any("registered-share class=B" in r.record_locator for r in refs)
    assert any("payment-date=2019-10-31" in r.record_locator for r in refs)


@pytest.mark.parametrize(("text", "kind"), [
    ("Kaufofferte der Example LLC zum Preis von USD 1'394.75 pro Aktie", "grouped_price_purchase_offer"),
    ("Rückkauf eigener Aktien", "issuer_repurchase"),
    ("Umtauschofferte: 1 Akt. Example Inc. von USD 2.50 nom. = 12.0298 Akt. Other Inc. von je USD 0.0001 nom. (VN 125483017)",
     "exchange_offer"),
    ("Fusion/Umtausch: 1 Akt. Example Inc. von USD 1.00 nom. = 0.255 Akt. Other Inc. von USD 0.01 nom. (VN 1330331)",
     "stock_only_merger_exchange"),
    ("Kaufofferte der Example LLC zum Preis von USD 1'39.75 pro Aktie", None),
    ("Rückkauf eigener Aktien bezahlt", None),
    ("Kaufofferte der Example LLC or Cash zum Preis von USD 1'394.75 pro Aktie", None),
    ("Umtauschofferte: 1 Akt. Example Inc. Oder Barzahlung von USD 2.50 nom. = 1 Akt. Other Inc. von USD 0.01 nom. (VN 1330331)",
     None),
    ("Fusion/Umtausch: 1 Akt. Example Inc. von USD 1.00 nom. = 1 Akt. Other Inc. oder USD 15.00 in bar "
     "von USD 0.01 nom. (VN 1330331)", None),
    ("Fusion/Umtausch: 1 Akt. Example Inc. von USD 1.00 nom. = 1 Akt. Other Inc. von USD 0.01 nom. (VN 1330331) oder USD 15.00 in bar",
     None),
])
def test_exact_zero_information_annotation_keeps_cash_election_outside_currency_proof(tmp_path, text, kind):
    original, item, entry, row = purchase_offer_fixture(tmp_path)
    _exact_provider_payment(tmp_path, original, item)
    row[0] = f"11.07.2019 {text}"
    add_purchase_offer(tmp_path, item, [entry], [row])
    if kind is None:
        with pytest.raises(DataReadinessError, match="not UTC midnight"):
            model(item).validates_against(original, tmp_path, POLICY)
        return
    refs = model(item).validates_against(original, tmp_path, POLICY)
    assert any(f"zero-information-entry-to-gross-row=0:1:{kind}" in r.record_locator for r in refs)
    original["payable_date"] = "2019-11-01"
    item["original_record_sha256"] = json_sha256(original)
    with pytest.raises(DataReadinessError, match="payment or record date differs"):
        model(item).validates_against(original, tmp_path, POLICY)


def test_loaded_currency_replay_binds_manifest_action_and_current_verification(tmp_path: Path) -> None:
    original, item = fixture(tmp_path)
    manifest = tmp_path / "supplying-currency-manifest.json"
    ref = write_json(manifest, {"schema": "market_predictor.investment_dividend_currency_corroborations",
        "corroborations": [item]})
    corroboration = load_currency_corroborations(tmp_path, manifest, ref["sha256"], policy_sha256=POLICY)[original["id"]]
    assert corroboration.model_dump(mode="json") == model(item).model_dump(mode="json")
    with pytest.raises(DataReadinessError, match="supplying manifest interpretation policy differs"):
        corroboration.validates_against(original, tmp_path, "b" * 64)
    refs = corroboration.validates_against(original, tmp_path, POLICY)
    binding, = [e for e in refs if e.reference == manifest.as_posix()]
    assert binding.artifact_sha256 == ref["sha256"]
    assert binding.record_locator == (
        f"currency corroboration;action_id={original['id']};original_record_sha256={json_sha256(original)}")
    assert binding.interpretation_policy_sha256 == POLICY
    assert binding.retrieved_at == binding.available_at
    assert binding.retrieved_at > datetime.fromisoformat(CAPTURE)
