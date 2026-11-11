"""Replay dividend currency evidence and explicit source-bound issuer corrections."""
from __future__ import annotations

import hashlib
import io
import json
import re
import zipfile
from collections.abc import Callable, Mapping
from datetime import UTC, date, datetime
from decimal import Decimal, InvalidOperation
from html.parser import HTMLParser
from pathlib import Path
from typing import Annotated, Any, Literal, Self
from urllib.parse import urlparse
from xml.etree import ElementTree as ET

from pydantic import Field, PrivateAttr, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.sources.official_documents import _Receipt
from market_predictor.swing.contracts.holding_accounting import EvidenceReference, HoldingContract, Identifier, Sha256

SCHEMA = "market_predictor.investment_dividend_currency_corroborations"
AMOUNT_CORRECTION_SCHEMA = "market_predictor.investment_dividend_amount_corrections"
MAX_SOURCE_BYTES = 16 * 1024**2
MAX_MANIFEST_BYTES = 4 * 1024**2
MAX_CURRENCY_MANIFEST_BYTES = 64 * 1024**2
MAX_ZIP_EXPANDED_BYTES = 64 * 1024**2
MAX_XML_BYTES = 16 * 1024**2
NS = {"m": "http://schemas.openxmlformats.org/spreadsheetml/2006/main"}
HEADERS = ("FUND NAME", "TICKER", "CUSIP", "EX-DATE", "RECORD DATE", "PAYABLE DATE", "DIVIDEND ($)",
           "SHORT TERM CAPITAL GAIN ($)", "LONG TERM CAPITAL GAIN ($)", "FREQUENCY")


class DividendEvidenceUnavailable(DataReadinessError):
    """Valid captured authority lacks a supported economic/class interpretation."""


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
    lookup: CapturedCurrencySource | None = None


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


class StockCompositeIdentity(HoldingContract):
    """Official stock class plus provider-asserted CUSIP and exact tax ISIN.

    The provider CUSIP is not independently corroborated by the SEC class proof.
    Missing dated ownership or another issuer share class prevents admission.
    """

    kind: Literal["stock_composite"]
    cusip: Annotated[str, Field(pattern=r"^[A-Z0-9]{9}$")]
    proof: CurrencyArtifactPin
    cusip_provenance: Literal["provider_original_cash_action"]
    _issuer_name: str = PrivateAttr(default="")
    _security_title: str = PrivateAttr(default="")
    _tax_identity_observation: dict[str, Any] = PrivateAttr(default_factory=dict)

    @property
    def tax_identity_observation(self) -> dict[str, Any]:
        return dict(self._tax_identity_observation)

    def validates_tax_identity(self, calculation: Mapping[str, Any], lookup: CapturedCurrencySource | None,
                               root: Path, policy_sha256: str) -> tuple[EvidenceReference, ...]:
        def remember(observation: dict[str, Any]) -> None:
            self._tax_identity_observation = observation

        refs, _ = replay_stock_tax_class(calculation, lookup, root, policy_sha256,
            issuer_name=self._issuer_name, security_title=self._security_title, observe=remember)
        return refs

    def validates_against(self, original: Mapping[str, Any], root: Path, policy_sha256: str,
                          cache: dict[str, Any] | None = None) -> tuple[EvidenceReference, ...]:
        from market_predictor.investment.stock_identity import StockIdentityProof, StockIdentityUnavailable, replay_stock_identity

        if (original.get("cusip") != self.cusip or original.get("foreign") is not False
            or original.get("special") is not False):
            raise DataReadinessError("stock composite original domestic class/CUSIP terms differ")
        raw = _read(root, self.proof, MAX_MANIFEST_BYTES)
        _json(raw)
        proof = StockIdentityProof.model_validate_json(raw)
        if proof.ticker != original.get("symbol"):
            raise DataReadinessError("stock composite official ticker differs from original event")
        pins = {self.proof.path: self.proof.sha256}
        try:
            replayed = replay_stock_identity(root, proof, sessions=(date.fromisoformat(original["ex_date"]),),
                source_files=pins, interpretation_sha256=policy_sha256, cache=cache)
        except StockIdentityUnavailable as exc:
            raise DividendEvidenceUnavailable(str(exc)) from exc
        if (replayed.missing_reasons or replayed.competing_class_tickers
            or replayed.proven_sessions != (date.fromisoformat(original["ex_date"]),)):
            raise DividendEvidenceUnavailable("stock composite dated class identity missing or multiple share classes unresolved")
        self._issuer_name = replayed.issuer_name
        self._security_title = replayed.security_title
        refs = list(replayed.evidence)
        if not refs:
            raise DividendEvidenceUnavailable("stock composite lacks official dated class evidence")
        # A proof's availability is the latest real authority capture, never its
        # retrospective action date or filesystem modification time.
        captures = [r.retrieved_at for r in refs]
        available = [r.available_at for r in refs if r.available_at is not None]
        refs.append(EvidenceReference(reference=self.proof.path, artifact_sha256=self.proof.sha256,
            record_locator="SEC dated stock-class proof; provider-only CUSIP=" + self.cusip,
            interpretation_policy_sha256=policy_sha256, retrieved_at=max(captures),
            available_at=max(available) if available else max(captures)))
        return tuple(refs)


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
        self.row_tables: list[int | None] = []
        self.brutto_tables: set[int] = set()
        self.table_stack: list[int] = []
        self.table_count = 0
        self.row_table: int | None = None
        self.cell_table: int | None = None
        self.header: list[str] | None = None
        self.header_table: int | None = None

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag == "table":
            # A nested table can still contribute legacy flattened cells. It
            # cannot certify same-table ownership for the new exception.
            if self.row is not None:
                self.row_table = None
            if self.header is not None:
                self.header_table = None
            self.table_count += 1
            self.table_stack.append(self.table_count)
        elif tag == "tr":
            self.row = []
            self.row_table = self.table_stack[-1] if self.table_stack else None
        elif tag == "th" and self.table_stack:
            self.header = []
            self.header_table = self.table_stack[-1]
        elif tag == "td" and self.row is not None:
            self.cell = []
            self.cell_table = self.table_stack[-1] if self.table_stack else None
            if self.cell_table != self.row_table:
                self.row_table = None

    def handle_data(self, data: str) -> None:
        if self.cell is not None:
            self.cell.append(data)
        if self.header is not None:
            self.header.append(data)

    def handle_endtag(self, tag: str) -> None:
        table = self.table_stack[-1] if self.table_stack else None
        if tag == "td" and self.cell is not None and self.row is not None:
            if self.cell_table != table:
                self.row_table = None
            self.row.append(" ".join(" ".join(self.cell).split()))
            self.cell = None
        elif tag == "tr" and self.row is not None:
            self.rows.append(self.row)
            self.row_tables.append(self.row_table if self.row_table == table else None)
            self.row = None
        elif tag == "th" and self.header is not None:
            if table is not None and self.header_table == table and " ".join(" ".join(self.header).split()) == "Brutto":
                self.brutto_tables.add(table)
            self.header = None
        elif tag == "table" and self.table_stack:
            self.table_stack.pop()


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


def valid_cusip(value: object) -> bool:
    if not isinstance(value, str) or not re.fullmatch(r"[A-Z0-9]{8}[0-9]", value):
        return False
    total = 0
    for index, character in enumerate(value[:8]):
        weighted = int(character, 36) * (1 + index % 2)
        total += weighted // 10 + weighted % 10
    return (total + int(value[-1])) % 10 == 0


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
        raise DividendEvidenceUnavailable("tax distribution ex-date is not UTC midnight milliseconds")
    return datetime.fromtimestamp(value / 1000, UTC).date()


_OFFER_ZERO_FIELDS = ("value", "valueTotal", "valueChf", "valueTotalChf",
    "paymentValueTotalChfWithHoldingTax", "paymentValueTotalChfWithoutHoldingTax")
_OFFER_NULL_FIELDS = ("exDate", "variant", "percent", "taxValueChf", "withHoldingTax", "valuePrefix", "remark")
_OFFER_FIELDS = {"type", "name", "currency", "nominalQuantity", "date", "paymentDate", "exchangeRate", "sign",
    *_OFFER_ZERO_FIELDS, *_OFFER_NULL_FIELDS}
_OFFER_TEXT = re.compile(
    r"(?P<day>[0-9]{2}\.[0-9]{2}\.[0-9]{4}) Kaufofferte der "
    r"(?P<offeror>[^<>\x00-\x1f]{1,160}) zum Preis von USD (?P<price>[0-9]{1,16}(?:\.[0-9]{1,12})?) pro Aktie"
)


def _purchase_offer_gross_row(entry: dict[str, Any], rows: list[list[str]]) -> int | None:
    """Bind the retained zero PAYMENT shape to an explicit purchase-offer disclosure."""
    if (set(entry) != _OFFER_FIELDS or entry["type"] != "PAYMENT" or entry["name"] != "calculation:payment_title"
        or entry["currency"] != {"categoryShortName": "CURREN", "shortName": "USD"}
        or type(entry["nominalQuantity"]) is not int or entry["nominalQuantity"] != 1 or entry["sign"] != ""
        or any(entry[field] is not None for field in _OFFER_NULL_FIELDS)
        or any(type(entry[field]) not in (int, Decimal) or entry[field] != 0 for field in _OFFER_ZERO_FIELDS)
        or type(entry["exchangeRate"]) not in (int, Decimal) or entry["exchangeRate"] <= 0):
        return None
    if type(entry["date"]) is not int or type(entry["paymentDate"]) is not int or entry["date"] != entry["paymentDate"]:
        return None
    try:
        paid = _date_ms(entry["paymentDate"])
    except (OverflowError, OSError, ValueError):
        return None
    # Decide uniqueness before interpreting the annotation text or any zero fields.
    candidates = [index for index, row in enumerate(rows)
        if len(row) > 6 and row[5] == "" and row[6] == paid.strftime("%d.%m.")]
    if len(candidates) != 1:
        return None
    index = candidates[0]
    row = rows[index]
    if row[1:] != ["", "", "", "", "", paid.strftime("%d.%m."), "", "USD", "0.00", "", "0.00", ""]:
        return None
    match = _OFFER_TEXT.fullmatch(row[0])
    if (match is None or match["day"] != paid.strftime("%d.%m.%Y") or not match["offeror"].strip()
        or _decimal(match["price"]) <= 0):
        return None
    return index


def _tax_payment_matches(entries: list[Any], table: _GrossRows | None, ex_date: date
                         ) -> tuple[list[tuple[int, dict[str, Any]]], dict[int, int]]:
    matches: list[tuple[int, dict[str, Any]]] = []
    offers: dict[int, int] = {}
    for index, entry in enumerate(entries):
        if not isinstance(entry, dict) or entry.get("type") != "PAYMENT":
            continue
        offer_row = _purchase_offer_gross_row(entry, table.rows) if table is not None and entry.get("exDate") is None else None
        if offer_row is not None and offer_row not in offers.values():
            offers[index] = offer_row
        elif _date_ms(entry.get("exDate")) == ex_date:
            matches.append((index, entry))
    return matches, offers


def _tax_gross_table(root: Path, source: CapturedCurrencySource, policy_sha256: str,
                     structured_receipt: dict[str, Any]) -> tuple[_GrossRows, tuple[EvidenceReference, ...]]:
    raw, receipt, refs = _captured(root, source, policy_sha256, {"www.ictax.admin.ch", "ictax.admin.ch"})
    payload = _json(raw)
    html = payload.get("data", {}).get("html")
    if receipt.get("request") != structured_receipt.get("request"):
        raise DataReadinessError("tax structured/HTML request identity differs")
    if payload.get("status") != "SUCCESS" or not isinstance(html, str) or "Brutto" not in _text(html):
        raise DividendEvidenceUnavailable("tax gross-column semantic context differs")
    table = _GrossRows()
    table.feed(html)
    return table, refs


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


def replay_stock_tax_class(
    calculation: Mapping[str, Any], lookup: CapturedCurrencySource | None,
    root: Path, policy_sha256: str, *, issuer_name: str, security_title: str,
    observe: Callable[[dict[str, Any]], None] | None = None,
) -> tuple[tuple[EvidenceReference, ...], dict[str, Any]]:
    if (calculation.get("instrumentGroup") != "SHARE" or not isinstance(calculation.get("name"), str)
        or not calculation["name"].strip()):
        raise DividendEvidenceUnavailable("stock tax instrument group/name missing or unsupported")
    if lookup is None:
        raise DividendEvidenceUnavailable("stock tax classification source missing")
    raw, receipt, refs = _captured(root, lookup, policy_sha256, {"www.ictax.admin.ch", "ictax.admin.ch"})
    payload = _json(raw)
    records = payload.get("security")
    if (payload.get("total") != 1 or payload.get("count") != 1 or not isinstance(records, list) or len(records) != 1
        or not isinstance(records[0], dict) or receipt.get("request", {}).get("isin") != calculation.get("isin")
        or records[0].get("isin") != calculation.get("isin") or records[0].get("vn") != calculation.get("valorNumber")):
        raise DataReadinessError("stock tax lookup exact ISIN/valor/request binding differs")
    record = records[0]
    observation = {"tax_security_name": calculation["name"], "tax_instrument_group": calculation["instrumentGroup"],
        "tax_institution": record.get("institution"), "tax_category_name": record.get("categoryName"),
        "tax_type_name": record.get("typeName"), "tax_type_description": record.get("typeDescription"),
        "sec_issuer_name": issuer_name, "sec_security_title": security_title,
        "cusip_provenance": "provider_original_cash_action", "lookup_source": lookup.model_dump(mode="json")}
    if observe is not None:
        observe(observation)
    category = record.get("categoryName")
    class_text = " ".join(str(record.get(field) or "") for field in ("name", "typeName", "typeDescription"))
    explicit_non_common = re.search(
        r"\b(preferred|preference|ADR|ADS|bonds?|notes?|debentures?|Vorzugsaktien)\b|deposit[ao]ry", class_text,
        re.IGNORECASE)
    calculation_non_common = re.search(
        r"\b(preferred|preference)\s+(stock|shares)\b|\b(ADR|ADS|bonds?|notes?|debentures?)\b|deposit[ao]ry",
        calculation["name"], re.IGNORECASE)
    official_common = re.search(r"\b(common|ordinary)\s+(stock|shares)\b", security_title, re.IGNORECASE)
    if category not in (None, "", "SHARE") or explicit_non_common or calculation_non_common or official_common is None:
        raise DividendEvidenceUnavailable("stock tax classification contradicts direct common/ordinary stock")
    return refs, observation


def replay_tax_currency_event(
    root: Path, tax: TaxCurrencyEvent, ex_date: date, provider_amount: Decimal,
    difference: Decimal, policy_sha256: str,
    *, stock_tax_validation: Callable[[Mapping[str, Any]], tuple[EvidenceReference, ...]] | None = None,
    payable_date: date | None = None, record_date: date | None = None,
    resolved_comparison: tuple[Decimal, Decimal] | None = None,
) -> tuple[Decimal, tuple[EvidenceReference, ...]]:
    """Replay structured and Brutto evidence, with an exact payment match when requested."""
    refs: list[EvidenceReference] = []
    raw, receipt, found = _captured(root, tax.source, policy_sha256, {"www.ictax.admin.ch", "ictax.admin.ch"})
    tax_locator = (f"/data/calculation/isin={tax.isin};"
        f"/data/calculationDetails/entries/{tax.entry_index}/currency/shortName=USD;"
        f"/data/calculationDetails/entries/{tax.entry_index}/value={tax.gross_amount}")
    refs.extend(ref.model_copy(update={"record_locator": tax_locator}) for ref in found)
    payload = _json(raw)
    if payload.get("status") != "SUCCESS" or payload.get("error") is not None:
        raise DataReadinessError("tax distribution response unsuccessful")
    calculation = payload["data"]["calculation"]
    request = receipt.get("request", {})
    if (calculation.get("isin") != tax.isin or request.get("isin") != tax.isin
        or request.get("referenceDate") != calculation.get("referenceDate")
        or request.get("valorNumber") != calculation.get("valorNumber")):
        raise DataReadinessError("tax distribution request/security identity differs")
    if (calculation.get("quotationType") != "PIECE" or type(calculation.get("quantity")) is not int
        or calculation.get("quantity") != 1):
        raise DividendEvidenceUnavailable("tax distribution per-share basis differs")
    if stock_tax_validation is not None:
        refs.extend(stock_tax_validation(calculation))
    entries = payload["data"]["calculationDetails"]["entries"]
    if not isinstance(entries, list) or len(entries) > 128:
        raise DataReadinessError("tax distribution entries exceed bounded shape")
    table: _GrossRows | None = None
    gross_refs: tuple[EvidenceReference, ...] = ()
    if any(isinstance(entry, dict) and entry.get("type") == "PAYMENT" and entry.get("exDate") is None for entry in entries):
        table, gross_refs = _tax_gross_table(root, tax.gross_context, policy_sha256, receipt)
    matches, offers = _tax_payment_matches(entries, table, ex_date)
    if len(matches) != 1 or matches[0][0] != tax.entry_index:
        raise DividendEvidenceUnavailable("tax distribution ex-date is absent or ambiguous")
    entry = matches[0][1]
    tax_amount = _decimal(entry.get("value"))
    if (entry.get("currency", {}).get("shortName") != "USD" or type(entry.get("nominalQuantity")) is not int
        or entry.get("nominalQuantity") != 1
        or tax_amount <= 0 or provider_amount <= 0 or tax_amount != _decimal(tax.gross_amount)
        or tax_amount - provider_amount != difference):
        raise DataReadinessError("tax currency/gross value/per-share comparison differs")
    if resolved_comparison is not None:
        resolved_amount, resolved_difference = resolved_comparison
        if tax_amount - resolved_amount != resolved_difference:
            raise DataReadinessError("tax-minus-resolved distribution amount comparison differs")
    if table is None:
        table, gross_refs = _tax_gross_table(root, tax.gross_context, policy_sha256, receipt)
    gross_rows = [index for index, row in enumerate(table.rows)
        if len(row) == 13 and row[5] == ex_date.strftime("%d.%m.") and row[8] == "USD"
        and _decimal(row[9]) == tax_amount]
    if len(gross_rows) != 1:
        raise DividendEvidenceUnavailable("tax amount/currency is absent or ambiguous in gross table column")
    gross_locator = (f"/data/html;gross-table-row={gross_rows[0]};ex-date={ex_date.isoformat()};"
        f"currency-column=8:USD;gross-column=9:{tax.gross_amount}")
    if payable_date is not None:
        row_index = gross_rows[0]
        if (_date_ms(entry.get("paymentDate")) != payable_date or _date_ms(entry.get("date")) != payable_date
                or table.rows[row_index][6] != payable_date.strftime("%d.%m.")
                or table.row_tables[row_index] not in table.brutto_tables
                or (entry.get("recordDate") is not None and _date_ms(entry["recordDate"]) != record_date)):
            raise DividendEvidenceUnavailable("completed tax distribution payment or record date differs")
        gross_locator += f";payment-date={payable_date.isoformat()}"
    if offers:
        target_table = table.row_tables[gross_rows[0]]
        if target_table not in table.brutto_tables or any(table.row_tables[row] != target_table for row in offers.values()):
            raise DividendEvidenceUnavailable("tax purchase-offer disclosure is outside the target gross table")
        offer_locators = ",".join(f"{entry}:{row}" for entry, row in offers.items())
        gross_locator += f";zero-purchase-offer-entry-to-gross-row={offer_locators}"
    refs.extend(ref.model_copy(update={"record_locator": gross_locator}) for ref in gross_refs)
    return tax_amount, tuple(refs)


class CurrencyCorroboration(HoldingContract):
    method: Literal["exact_tax_event", "issuer_rate_tax_currency_bridge"]
    action_id: Identifier
    original_record_sha256: Sha256
    symbol: Identifier
    ex_date: date
    currency: Literal["USD"]
    tax: TaxCurrencyEvent
    identity: Annotated[IssuerWorkbookIdentity | OfficialDocumentIdentity | StockCompositeIdentity, Field(discriminator="kind")]
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
                          amount_correction: IssuerDividendAmountCorrection | None = None,
                          stock_identity_cache: dict[str, Any] | None = None) -> tuple[EvidenceReference, ...]:
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
        elif isinstance(self.identity, StockCompositeIdentity):
            if self.method != "exact_tax_event" or amount_correction is not None:
                raise DataReadinessError("stock composite requires exact original gross amount")
            found = self.identity.validates_against(original, root, policy_sha256, stock_identity_cache)
        else:
            raw, _, found = _captured(root, self.identity.source, policy_sha256,
                {"www.sec.gov", "sec.gov", "www.invesco.com", "invesco.com"})
            text = _text(raw.decode("utf-8"))
            quotes = tuple(" ".join(q.split()) for q in self.identity.quotes)
            joined = " ".join(quotes)
            if (any(len(q) > 4096 or q not in text for q in quotes) or cusip not in joined
                or not re.search(r"\b" + re.escape(self.symbol) + r"\b", joined)
                or self.identity.fund_name not in joined):
                raise DividendEvidenceUnavailable("official source does not prove quoted ticker/CUSIP/fund identity")
        refs.extend(found)
        provider_amount = _decimal(original.get("rate"))
        identity = self.identity
        stock_validation = None
        if isinstance(identity, StockCompositeIdentity):
            def validate_tax_identity(calculation: Mapping[str, Any]) -> tuple[EvidenceReference, ...]:
                return identity.validates_tax_identity(calculation, self.tax.lookup, root, policy_sha256)

            stock_validation = validate_tax_identity
        tax_amount, tax_refs = replay_tax_currency_event(
            root, self.tax, self.ex_date, provider_amount, _decimal(self.tax_value_difference), policy_sha256,
            stock_tax_validation=stock_validation, resolved_comparison=None if amount_correction is None
                else (effective_amount, _decimal(self.tax_resolved_value_difference)),
        )
        refs.extend(tax_refs)
        if self.issuer_rate is None:
            if provider_amount != tax_amount:
                raise DividendEvidenceUnavailable("independent distribution amount differs from provider rate")
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
    guard("investment dividend currency manifest")
    # Strict JSON parsing rejects duplicate fields before Pydantic's typed JSON parser.
    raw = _read(root, CurrencyArtifactPin(path=manifest.as_posix(), sha256=sha256), MAX_CURRENCY_MANIFEST_BYTES)
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
