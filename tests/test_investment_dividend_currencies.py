"""Synthetic unit evidence for strict independent currency-source admission."""
from __future__ import annotations

import copy
import hashlib
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
    load_currency_corroborations,
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
             formula: bool = False) -> bytes:
    strings = list(HEADERS) + ["Fund", "SPY", "78462F103", "09/20/2019", "09/23/2019", "10/31/2019",
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
