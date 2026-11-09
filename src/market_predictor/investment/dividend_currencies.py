"""Replay dividend currency evidence and explicit source-bound issuer corrections."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Mapping
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from pydantic import Field, PrivateAttr, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.sources.official_documents import _Receipt
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Identifier, Sha256

SCHEMA = "market_predictor.investment_dividend_currency_corroborations"
AMOUNT_CORRECTION_SCHEMA = "market_predictor.investment_dividend_amount_corrections"
MAX_SOURCE_BYTES = 16 * 1024**2
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_ZIP_EXPANDED_BYTES = 64 * 1024**2
MAX_XML_BYTES = 16 * 1024**2
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = ("FUND NAME", "TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE", "DIVIDEND ($)",
           "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)", "FREQUENCY")


class CurrencyArtifactPin(HoldingContract):
    path: Identifier
    sha256: Sha256


class CapturedCurrencySource(HoldingContract):
    body: CurrencyArtifactPin
    receipt: CurrencyArtifactPin


class TaxCurrencyEvent(HoldingContract):
    source: CapturedCurrencySource
    gross_context: CapturedCurrencySource
    isin: Annotated[str, Field(pattern=r"^US[A-Z0-9]{9}[0-9]$")]
    entry_index: Annotated[int, Field(ge=0)]
    gross_amount: Identifier


class IssuerWorkbookIdentity(HoldingContract):
    kind: Literal["issuer_workbook"]
    source: CapturedCurrencySource
    row: Annotated[int, Field(ge=2, le=100_000)]
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{9}$")]


class OfficialDocumentIdentity(HoldingContract):
    kind: Literal["official_document"]
    source: CapturedCurrencySource
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{9}$")]
    fund_name: Identifier
    quotes: Annotated[tuple[Identifier, ...], Field(min_length=1, max_length=8)]


class IssuerDistributionRate(HoldingContract):
    source: CapturedCurrencySource
    row: Annotated[int, Field(ge=2, le=100_000)]


class OccSpecialCashDistribution(HoldingContract):
    """An announced cash entitlement; neither a tax classification nor payment proof."""

    kind: Literal["occ_special_cash_distribution"]
    source: CapturedCurrencySource
    notice_number: Annotated[int, Field(gt=0)]
    fund_name: Identifier
    record_date: date
    announced_payable_date: date
    provider_payable_date: date
    payable_date_conflict: bool
    economic_treatment: Literal["nominal_unpaid_nontradable_entitlement"]

    def validates_against(self, original: Mapping[str, Any], root: Path, policy_sha256: str, *,
                          action_id: str, digest: str, symbol: str, ex_date: date,
                          cusip: str) -> tuple[EvidenceReference, ...]:
        if (json_sha256(original) != digest or original.get("id") != action_id
            or original.get("symbol") != symbol or original.get("ex_date") != ex_date.isoformat()
            or original.get("special") is not True or original.get("foreign") is not False
            or original.get("cusip") != cusip or original.get("currency") not in (None, "", "USD")
            or original.get("record_date") != self.record_date.isoformat()
            or original.get("payable_date") != self.provider_payable_date.isoformat()
            or any(original.get(key) not in (None, "", False) for key in
                ("cancelled", "canceled", "revised", "revision"))
            or original.get("due_bill_redemption_date") not in (None, "")
            or not ex_date <= self.record_date <= min(self.announced_payable_date, self.provider_payable_date)
            or self.payable_date_conflict != (self.announced_payable_date != self.provider_payable_date)):
            raise DataReadinessError("special cash distribution original terms or payable conflict differ")
        raw = _read(root, self.source.body)
        payload = _json(_read(root, self.source.receipt, MAX_MANIFEST_BYTES))
        if payload.pop("receipt_sha256", None) != json_sha256(payload):
            raise DataReadinessError("special cash distribution official receipt hash differs")
        try:
            receipt = _Receipt.model_validate(payload)
        except ValueError as exc:
            raise DataReadinessError("special cash distribution official receipt unsupported") from exc
        response = receipt.response
        urls = {f"https://infomemo.theocc.com/infomemos?number={self.notice_number}",
            f"https://www.miaxglobal.com/sites/default/files/alert-files/{symbol}_Distribution_{self.notice_number}.pdf"}
        if (response is None or receipt.state != "archived_unreviewed"
            or response.requested_url not in urls or response.final_url != response.requested_url or response.redirect_chain
            or response.status_code != 200 or response.sha256 != self.source.body.sha256
            or response.body_length != len(raw) or not raw.startswith(b"%PDF-")
            or (response.content_type or "").split(";", 1)[0].strip().lower() != "application/pdf"
            or receipt.error_type is not None or receipt.failure_status_code is not None
            or receipt.body_path != f"body-{self.source.body.sha256}.bin"
            or receipt.body_path != Path(self.source.body.path).name
            or not receipt.started_at_utc <= response.retrieved_at_utc <= receipt.completed_at_utc <= datetime.now(UTC)):
            raise DataReadinessError("special cash distribution official source binding differs")
        # Ordinary currency replay has no PDF dependency. Special admission always
        # re-extracts the pinned original document, never an operator text summary.
        from pypdf import PdfReader
        from pypdf.errors import PyPdfError

        try:
            reader = PdfReader(io.BytesIO(raw), strict=True)
            if reader.is_encrypted or not 1 <= len(reader.pages) <= 4:
                raise DataReadinessError("special cash distribution PDF exceeds supported layout")
            parts: list[str] = []
            for page in reader.pages:
                content = page.get_contents()
                if content is not None and len(content.get_data()) > 2 * 1024**2:
                    raise DataReadinessError("special cash distribution PDF content exceeds bound")
                parts.append(page.extract_text() or "")
            text = " ".join(" ".join(parts).split())
            if len(text) > 65_536:
                raise DataReadinessError("special cash distribution PDF text exceeds bound")
        except (ValueError, TypeError, KeyError, OSError, PyPdfError) as exc:
            raise DataReadinessError("special cash distribution PDF cannot be replayed") from exc
        declaration = re.compile(
            re.escape(self.fund_name) + r" \((?P<symbol>[A-Z][A-Z0-9.\-]{0,15})\) has announced a Special Cash Dividend of "
            r"\$(?P<amount>[0-9]+(?:\.[0-9]+)?) per (?P=symbol) [Ss]hare\. "
            r"The record date is (?P<record>[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4}); (?:the )?payable date is "
            r"(?P<payable>[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4})\. The ex-distribution "
            r"date for this distribution will be (?P<ex>[A-Z][a-z]+ [0-9]{1,2}, [0-9]{4})\."
        )
        matches = list(declaration.finditer(text))
        if (len(matches) != 1 or re.search(r"#" + str(self.notice_number) + r"\b", text) is None
            or re.findall(r"\bCUSIP:\s*([A-Z0-9]{9})\b", text) != [cusip]
            or re.search(r"\bdue[ -]?bills?\b|\bcapital gains?\b", text, re.IGNORECASE)):
            raise DataReadinessError("special cash distribution declaration or identity unsupported")
        match = matches[0]
        if (match["symbol"] != symbol or _decimal(match["amount"]) != _decimal(original.get("rate"))
            or _decimal(match["amount"]) <= 0
            or _notice_date(match["record"]) != self.record_date
            or _notice_date(match["payable"]) != self.announced_payable_date
            or _notice_date(match["ex"]) != ex_date):
            raise DataReadinessError("special cash distribution source economic terms differ")
        locator = (f"OCC#{self.notice_number};Special Cash Dividend;symbol={symbol};CUSIP={cusip};"
            f"ex={ex_date};record={self.record_date};amount={match['amount']};"
            f"announced_payable={self.announced_payable_date};provider_payable={self.provider_payable_date};"
            f"payable_date_conflict={self.payable_date_conflict};{self.economic_treatment}")
        return tuple(EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256, record_locator=locator,
            interpretation_policy_sha256=policy_sha256, retrieved_at=receipt.completed_at_utc,
            available_at=receipt.completed_at_utc) for pin in (self.source.body, self.source.receipt))


def _notice_date(value: str) -> date:
    months = ("January", "February", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December")
    month, day, year = value.replace(",", "").split()
    try:
        return date(int(year), months.index(month) + 1, int(day))
    except ValueError as exc:
        raise DataReadinessError("special cash distribution notice date unsupported") from exc


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.parts: list[str] = []

    def handle_data(self, data: str) -> None:
        self.parts.append(data)


def _text(html: str) -> str:
    parser = _HtmlText()
    parser.feed(html)
    return " ".join(" ".join(parser.parts).split())


class _GrossRows(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.rows: list[list[str]] = []
        self.row: list[str] | None = None
        self.cell: list[str] | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "tr":
            self.row = []
        elif tag == "td" and self.row is not None:
            self.cell = []

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)

    def handle_endtag(self, tag: str) -> None:
        if tag == "td" and self.cell is not None and self.row is not None:
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row = None


def _pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for name, value in pairs:
        if name in result:
            raise DataReadinessError("duplicate independent currency JSON field")
        result[name] = value
    return result


def _json(raw: bytes) -> dict[str, Any]:
    def refuse_constant(value: str) -> None:
        raise DataReadinessError(f"non-finite independent currency JSON value: {value}")

    value = json.loads(raw, parse_float=Decimal, parse_constant=refuse_constant, object_pairs_hook=_pairs)
    if not isinstance(value, dict):
        raise DataReadinessError("independent currency source must be a JSON object")
    return value


def _read(root: Path, pin: CurrencyArtifactPin, maximum: int = MAX_SOURCE_BYTES) -> bytes:
    path = resolve_inside_authority(root, pin.path)
    if path.stat().st_size > maximum:
        raise DataReadinessError("independent currency artifact exceeds byte bound")
    raw = path.read_bytes()
    if len(raw) > maximum or hashlib.sha256(raw).hexdigest() != pin.sha256:
        raise DataReadinessError("independent currency artifact hash differs")
    return raw


def _decimal(value: object) -> Decimal:
    if isinstance(value, bool) or not isinstance(value, (str, int, float, Decimal)):
        raise DataReadinessError("independent currency amount is not numeric")
    try:
        result = Decimal(str(value).strip())
    except InvalidOperation as exc:
        raise DataReadinessError("independent currency amount is not numeric") from exc
    if not result.is_finite():
        raise DataReadinessError("independent currency amount is not finite")
    return result


def _captured(root: Path, source: CapturedCurrencySource, policy_sha256: str,
              hosts: set[str]) -> tuple[bytes, dict[str, Any], tuple[EvidenceReference, ...]]:
    raw = _read(root, source.body)
    receipt = _json(_read(root, source.receipt, MAX_MANIFEST_BYTES))
    url = receipt.get("url", receipt.get("source_url"))
    if not isinstance(url, str) or urlparse(url).scheme != "https" or urlparse(url).hostname not in hosts:
        raise DataReadinessError("independent currency source authority differs")
    digest = receipt.get("sha256")
    size = receipt.get("bytes", receipt.get("size_bytes"))
    if not isinstance(digest, str) or digest.lower() != source.body.sha256 or size != len(raw):
        raise DataReadinessError("independent currency receipt does not bind body")
    clock = receipt.get("retrieved_at", receipt.get("retrieved_at_utc", receipt.get("captured_at_utc")))
    if not isinstance(clock, str):
        raise DataReadinessError("independent currency capture clock missing")
    captured = datetime.fromisoformat(clock.replace("Z", "+00:00"))
    if captured.utcoffset() is None or captured > datetime.now(UTC):
        raise DataReadinessError("independent currency capture clock must be aware and not in the future")
    refs = tuple(EvidenceReference(reference=pin.path, artifact_sha256=pin.sha256,
        record_locator="captured independent distribution source" if pin is source.body else "source retrieval receipt",
        interpretation_policy_sha256=policy_sha256, retrieved_at=captured, available_at=captured)
        for pin in (source.body, source.receipt))
    return raw, receipt, refs


def _isin(cusip: str) -> str:
    prefix = "US" + cusip
    for check in range(10):
        candidate = prefix + str(check)
        digits = "".join(str(ord(c) - 55) if c.isalpha() else c for c in candidate)
        total = sum((int(d) * (2 if i % 2 else 1)) // 10 + (int(d) * (2 if i % 2 else 1)) % 10
                    for i, d in enumerate(reversed(digits)))
        if total % 10 == 0:
            return candidate
    raise DataReadinessError("US security identifier checksum unresolved")


def _xml(archive: zipfile.ZipFile, name: str) -> ET.Element:
    raw = archive.read(name)
    if len(raw) > MAX_XML_BYTES or b"<!DOCTYPE" in raw or b"<!ENTITY" in raw:
        raise DataReadinessError("issuer workbook XML exceeds bound or contains entities")
    return ET.fromstring(raw)


def _workbook_rows(raw: bytes) -> dict[int, dict[str, str]]:
    with zipfile.ZipFile(io.BytesIO(raw)) as archive:
        entries = archive.infolist()
        if (len(entries) > 128 or len({item.filename for item in entries}) != len(entries)
            or sum(item.file_size for item in entries) > MAX_ZIP_EXPANDED_BYTES
            or any(item.file_size > MAX_XML_BYTES or item.flag_bits & 1 for item in entries)):
            raise DataReadinessError("issuer workbook ZIP exceeds bounded layout")
        book = _xml(archive, "xl/workbook.xml")
        sheets = book.findall("m:sheets/m:sheet", NS)
        if len(sheets) != 1 or sheets[0].get("name") != "dividend":
            raise DataReadinessError("issuer workbook sheet differs")
        rel_id = sheets[0].get("{http://schemas.openxmlformats.org/officeDocument/2006/relationships}id")
        rels = _xml(archive, "xl/_rels/workbook.xml.rels")
        targets = [r.get("Target") for r in rels if r.get("Id") == rel_id and r.get("TargetMode") != "External"]
        if targets != ["worksheets/sheet1.xml"]:
            raise DataReadinessError("issuer workbook worksheet binding differs")
        strings = ["".join(node.itertext()) for node in _xml(archive, "xl/sharedStrings.xml").findall("m:si", NS)]
        result: dict[int, dict[str, str]] = {}
        for row in _xml(archive, "xl/worksheets/sheet1.xml").findall("m:sheetData/m:row", NS):
            number = int(row.attrib["r"])
            if number in result or not 1 <= number <= 100_000:
                raise DataReadinessError("issuer workbook row identity differs")
            cells: dict[str, str] = {}
            for cell in row:
                address = cell.get("r", "")
                match = re.fullmatch(r"([A-J])([0-9]+)", address)
                if match is None or int(match[2]) != number or match[1] in cells or cell.find("m:f", NS) is not None:
                    raise DataReadinessError("issuer workbook cell or formula unsupported")
                value = cell.find("m:v", NS)
                text = "" if value is None or value.text is None else value.text
                if cell.get("t") == "s":
                    index = int(text)
                    if not 0 <= index < len(strings):
                        raise DataReadinessError("issuer workbook shared string differs")
                    text = strings[index]
                elif cell.get("t") not in (None, "n"):
                    raise DataReadinessError("issuer workbook cell type unsupported")
                cells[match[1]] = text.strip()
            result[number] = cells
        if tuple(result.get(1, {}).get(col, "") for col in "ABCDEFGHIJ") != HEADERS:
            raise DataReadinessError("issuer workbook distribution headers differ")
        return result


def _issuer_row(root: Path, source: CapturedCurrencySource, row_number: int, symbol: str, cusip: str,
                ex_date: date, policy_sha256: str) -> tuple[dict[str, str], tuple[EvidenceReference, ...]]:
    raw, _, refs = _captured(root, source, policy_sha256, {"www.ssga.com", "ssga.com"})
    rows = _workbook_rows(raw)
    matches = []
    for number, row in rows.items():
        if number == 1 or row.get("B") != symbol or row.get("C") != cusip:
            continue
        try:
            day = datetime.strptime(row.get("D", ""), "%m/%d/%Y").date()
        except ValueError as exc:
            raise DataReadinessError("issuer distribution ex-date unsupported") from exc
        if day == ex_date:
            matches.append(number)
    if matches != [row_number]:
        raise DataReadinessError("issuer distribution identity/date row is absent or ambiguous")
    return rows[row_number], tuple(ref.model_copy(update={"record_locator": f"dividend!A{row_number}:J{row_number}"}) for ref in refs)


def _date_ms(value: object) -> date:
    if isinstance(value, bool) or not isinstance(value, int) or value % 86_400_000:
        raise DataReadinessError("tax distribution ex-date is not UTC midnight milliseconds")
    return datetime.fromtimestamp(value / 1000, UTC).date()


def _original_action(original: Mapping[str, Any], action_id: str, digest: str, symbol: str, ex_date: date) -> None:
    if (json_sha256(original) != digest or original.get("id") != action_id
        or original.get("symbol") != symbol or original.get("ex_date") != ex_date.isoformat()
        or original.get("foreign") is not False or original.get("special") is not False
        or original.get("currency") not in (None, "", "USD")):
        raise DataReadinessError("independent dividend original record binding differs")
    if any(original.get(k) not in (None, "", False) for k in
           ("cancelled", "canceled", "revised", "revision", "due_bill_redemption_date")):
        raise DataReadinessError("independent dividend cannot repair unsupported action terms")


def _issuer_gross_amount(root: Path, issuer: IssuerDistributionRate, symbol: str, cusip: str, ex_date: date,
                         policy_sha256: str) -> tuple[Decimal, tuple[EvidenceReference, ...]]:
    row, refs = _issuer_row(root, issuer.source, issuer.row, symbol, cusip, ex_date, policy_sha256)
    amount = _decimal(row.get("G"))
    if amount <= 0 or any(row.get(col, "") and _decimal(row[col]) != 0 for col in ("H", "I")):
        raise DataReadinessError("issuer gross distribution amount is nonpositive or contains capital gains")
    return amount, refs


class ResolvedDividendAmount(HoldingContract):
    """An issuer amount with source clocks; this record supplies no currency proof."""
    action_id: Identifier
    original_record_sha256: Sha256
    symbol: Identifier
    ex_date: date
    original_rate: Annotated[Decimal, Field(gt=0)]
    gross_amount: Annotated[Decimal, Field(gt=0)]
    evidence: Annotated[tuple[EvidenceReference, ...], Field(min_length=1)]


class IssuerDividendAmountCorrection(HoldingContract):
    """Explicit reviewed issuer correction; never a tolerance or an altered provider record."""
    method: Literal["issuer_gross_amount_correction"]
    action_id: Identifier
    original_record_sha256: Sha256
    symbol: Identifier
    ex_date: date
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{9}$")]
    isin: Annotated[str, Field(pattern=r"^US[A-Z0-9]{9}[0-9]$")]
    original_rate: Identifier
    corrected_gross_amount: Identifier
    issuer_rate: IssuerDistributionRate

    @model_validator(mode="after")
    def distinct_positive_amounts(self) -> Self:
        original, corrected = _decimal(self.original_rate), _decimal(self.corrected_gross_amount)
        if original <= 0 or corrected <= 0 or original == corrected:
            raise ValueError("issuer correction requires distinct positive original and corrected amounts")
        return self

    def resolve_against(self, original: Mapping[str, Any], root: Path, policy_sha256: str) -> ResolvedDividendAmount:
        _original_action(original, self.action_id, self.original_record_sha256, self.symbol, self.ex_date)
        original_amount, corrected = _decimal(self.original_rate), _decimal(self.corrected_gross_amount)
        if original_amount != _decimal(original.get("rate")):
            raise DataReadinessError("issuer correction declared original rate differs from provider")
        if _isin(self.cusip) != self.isin or original.get("cusip") not in (None, "", self.cusip):
            raise DataReadinessError("issuer correction CUSIP/ISIN identity differs")
        issuer_amount, refs = _issuer_gross_amount(root, self.issuer_rate, self.symbol, self.cusip,
            self.ex_date, policy_sha256)
        if issuer_amount != corrected:
            raise DataReadinessError("issuer correction amount differs from exact issuer gross distribution")
        return ResolvedDividendAmount(action_id=self.action_id, original_record_sha256=self.original_record_sha256,
            symbol=self.symbol, ex_date=self.ex_date, original_rate=original_amount, gross_amount=corrected, evidence=refs)


class CurrencyCorroboration(HoldingContract):
    method: Literal["exact_tax_event", "issuer_rate_tax_currency_bridge"]
    action_id: Identifier
    original_record_sha256: Sha256
    symbol: Identifier
    ex_date: date
    currency: Literal["USD"]
    tax: TaxCurrencyEvent
    identity: Annotated[IssuerWorkbookIdentity | OfficialDocumentIdentity, Field(discriminator="kind")]
    issuer_rate: IssuerDistributionRate | None = None
    tax_value_difference: Identifier
    tax_resolved_value_difference: Identifier | None = None
    special_cash_distribution: OccSpecialCashDistribution | None = None
    _validated_evidence: tuple[EvidenceReference, ...] = PrivateAttr(default=())

    @model_validator(mode="after")
    def method_sources(self) -> Self:
        if (self.method == "exact_tax_event") != (self.issuer_rate is None):
            raise ValueError("currency method and issuer rate source differ")
        if self.special_cash_distribution is not None and self.method != "exact_tax_event":
            raise ValueError("special cash distribution requires exact tax event currency proof")
        return self

    @property
    def evidence(self) -> tuple[EvidenceReference, ...]:
        return self._validated_evidence

    def validates_against(self, original: Mapping[str, Any], root: Path, policy_sha256: str, *,
                          amount_correction: IssuerDividendAmountCorrection | None = None) -> tuple[EvidenceReference, ...]:
        """Replay separate currency and optional cash-class evidence; retain provider terms."""
        special_refs: tuple[EvidenceReference, ...] = ()
        if self.special_cash_distribution is None:
            _original_action(original, self.action_id, self.original_record_sha256, self.symbol, self.ex_date)
        else:
            if amount_correction is not None:
                raise DataReadinessError("special cash distribution cannot use an amount correction")
            special_refs = self.special_cash_distribution.validates_against(original, root, policy_sha256,
                action_id=self.action_id, digest=self.original_record_sha256, symbol=self.symbol,
                ex_date=self.ex_date, cusip=self.identity.cusip)
        cusip = self.identity.cusip
        if _isin(cusip) != self.tax.isin or original.get("cusip") not in (None, "", cusip):
            raise DataReadinessError("independent currency CUSIP/ISIN identity differs")
        refs: list[EvidenceReference] = list(special_refs)
        effective_amount = _decimal(original.get("rate"))
        if amount_correction is not None:
            if amount_correction.cusip != cusip or amount_correction.isin != self.tax.isin:
                raise DataReadinessError("currency and amount correction security identities differ")
            resolved = amount_correction.resolve_against(original, root, policy_sha256)
            effective_amount = resolved.gross_amount
            refs.extend(resolved.evidence)
            if self.tax_resolved_value_difference is None:
                raise DataReadinessError("corrected dividend currency requires tax-minus-resolved comparison")
        elif self.tax_resolved_value_difference is not None:
            raise DataReadinessError("tax-minus-resolved comparison requires an explicit amount correction")
        if isinstance(self.identity, IssuerWorkbookIdentity):
            _, found = _issuer_row(root, self.identity.source, self.identity.row, self.symbol, cusip, self.ex_date, policy_sha256)
        else:
            raw, _, found = _captured(root, self.identity.source, policy_sha256,
                {"www.sec.gov", "sec.gov", "www.invesco.com", "invesco.com"})
            text = _text(raw.decode("utf-8"))
            quotes = tuple(" ".join(q.split()) for q in self.identity.quotes)
            joined = " ".join(quotes)
            if (any(len(q) > 4096 or q not in text for q in quotes) or cusip not in joined
                or not re.search(r"\b" + re.escape(self.symbol) + r"\b", joined)
                or self.identity.fund_name not in joined):
                raise DataReadinessError("official source does not prove quoted ticker/CUSIP/fund identity")
        refs.extend(found)
        raw, receipt, found = _captured(root, self.tax.source, policy_sha256, {"www.ictax.admin.ch", "ictax.admin.ch"})
        tax_locator = (f"/data/calculation/isin={self.tax.isin};"
            f"/data/calculationDetails/entries/{self.tax.entry_index}/currency/shortName=USD;"
            f"/data/calculationDetails/entries/{self.tax.entry_index}/value={self.tax.gross_amount}")
        refs.extend(ref.model_copy(update={"record_locator": tax_locator}) for ref in found)
        payload = _json(raw)
        if payload.get("status") != "SUCCESS" or payload.get("error") is not None:
            raise DataReadinessError("tax distribution response unsuccessful")
        calculation = payload["data"]["calculation"]
        request = receipt.get("request", {})
        if (calculation.get("isin") != self.tax.isin or request.get("isin") != self.tax.isin
            or request.get("referenceDate") != calculation.get("referenceDate")
            or request.get("valorNumber") != calculation.get("valorNumber")
            or calculation.get("quotationType") != "PIECE" or type(calculation.get("quantity")) is not int
            or calculation.get("quantity") != 1):
            raise DataReadinessError("tax distribution request/security/per-share basis differs")
        entries = payload["data"]["calculationDetails"]["entries"]
        if not isinstance(entries, list) or len(entries) > 128:
            raise DataReadinessError("tax distribution entries exceed bounded shape")
        matches = [(i, e) for i, e in enumerate(entries) if isinstance(e, dict) and e.get("type") == "PAYMENT"
                   and _date_ms(e.get("exDate")) == self.ex_date]
        if len(matches) != 1 or matches[0][0] != self.tax.entry_index:
            raise DataReadinessError("tax distribution ex-date is absent or ambiguous")
        entry = matches[0][1]
        tax_amount = _decimal(entry.get("value"))
        provider_amount = _decimal(original.get("rate"))
        if (entry.get("currency", {}).get("shortName") != "USD" or type(entry.get("nominalQuantity")) is not int
            or entry.get("nominalQuantity") != 1
            or tax_amount <= 0 or provider_amount <= 0 or tax_amount != _decimal(self.tax.gross_amount)
            or tax_amount - provider_amount != _decimal(self.tax_value_difference)):
            raise DataReadinessError("tax currency/gross value/per-share comparison differs")
        if (amount_correction is not None
            and tax_amount - effective_amount != _decimal(self.tax_resolved_value_difference)):
            raise DataReadinessError("tax-minus-resolved distribution amount comparison differs")
        html_raw, html_receipt, found = _captured(root, self.tax.gross_context, policy_sha256,
            {"www.ictax.admin.ch", "ictax.admin.ch"})
        html_payload = _json(html_raw)
        html = html_payload.get("data", {}).get("html")
        if (html_payload.get("status") != "SUCCESS" or not isinstance(html, str)
            or html_receipt.get("request") != receipt.get("request") or "Brutto" not in _text(html)):
            raise DataReadinessError("tax gross-column semantic context differs")
        table = _GrossRows()
        table.feed(html)
        gross_rows = [index for index, row in enumerate(table.rows)
            if len(row) == 13 and row[5] == self.ex_date.strftime("%d.%m.") and row[8] == "USD"
            and _decimal(row[9]) == tax_amount]
        if len(gross_rows) != 1:
            raise DataReadinessError("tax amount/currency is absent or ambiguous in gross table column")
        gross_locator = (f"/data/html;gross-table-row={gross_rows[0]};ex-date={self.ex_date.isoformat()};"
            f"currency-column=8:USD;gross-column=9:{self.tax.gross_amount}")
        refs.extend(ref.model_copy(update={"record_locator": gross_locator}) for ref in found)
        if self.issuer_rate is None:
            if provider_amount != tax_amount:
                raise DataReadinessError("independent distribution amount differs from provider rate")
        else:
            issuer_amount, found = _issuer_gross_amount(root, self.issuer_rate,
                self.symbol, cusip, self.ex_date, policy_sha256)
            refs.extend(found)
            if issuer_amount != effective_amount:
                raise DataReadinessError("issuer gross distribution rate differs from resolved economic amount")
        unique: dict[str, EvidenceReference] = {}
        for ref in refs:
            previous = unique.get(ref.reference)
            if previous is not None and previous.artifact_sha256 != ref.artifact_sha256:
                raise DataReadinessError("currency evidence has conflicting source hashes")
            unique[ref.reference] = ref
        self._validated_evidence = tuple(unique[key] for key in sorted(unique))
        return self._validated_evidence


class CurrencyCorroborationManifest(HoldingContract):
    schema_name: Literal["market_predictor.investment_dividend_currency_corroborations"] = Field(alias="schema")
    corroborations: Annotated[tuple[CurrencyCorroboration, ...], Field(min_length=1, max_length=10_000)]


def load_currency_corroborations(root: Path, manifest: Path, sha256: str, *, policy_sha256: str) -> dict[str, CurrencyCorroboration]:
    """Load closed typed assertions; source admission occurs against each actual action."""
    # Strict JSON parsing rejects duplicate fields before Pydantic's typed JSON parser.
    raw = _read(root, CurrencyArtifactPin(path=manifest.as_posix(), sha256=sha256), MAX_MANIFEST_BYTES)
    _json(raw)
    parsed = CurrencyCorroborationManifest.model_validate_json(raw)
    result: dict[str, CurrencyCorroboration] = {}
    for item in parsed.corroborations:
        if item.action_id in result:
            raise DataReadinessError("independent currency manifest repeats action ID")
        if re.fullmatch(r"[0-9a-f]{64}", policy_sha256) is None:
            raise DataReadinessError("independent currency interpretation policy hash differs")
        result[item.action_id] = item
    return result


class DividendAmountCorrectionManifest(HoldingContract):
    schema_name: Literal["market_predictor.investment_dividend_amount_corrections"] = Field(alias="schema")
    corrections: Annotated[tuple[IssuerDividendAmountCorrection, ...], Field(min_length=1, max_length=10_000)]


def load_dividend_amount_corrections(root: Path, manifest: Path, sha256: str, *,
                                    policy_sha256: str) -> dict[str, IssuerDividendAmountCorrection]:
    """Load explicit amount corrections; each consuming action must replay its actual issuer sources."""
    raw = _read(root, CurrencyArtifactPin(path=manifest.as_posix(), sha256=sha256), MAX_MANIFEST_BYTES)
    _json(raw)
    parsed = DividendAmountCorrectionManifest.model_validate_json(raw)
    if re.fullmatch(r"[0-9a-f]{64}", policy_sha256) is None:
        raise DataReadinessError("issuer amount correction interpretation policy hash differs")
    result: dict[str, IssuerDividendAmountCorrection] = {}
    for correction in parsed.corrections:
        if correction.action_id in result:
            raise DataReadinessError("issuer amount correction manifest repeats action ID")
        result[correction.action_id] = correction
    return result
