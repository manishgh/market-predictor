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
