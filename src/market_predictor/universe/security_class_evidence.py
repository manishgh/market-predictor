"""Hash-bound SEC cover-page facts, not mapping intervals or admission decisions."""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from datetime import date
from pathlib import Path

from bs4 import BeautifulSoup, Tag

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.path_integrity import resolve_existing_file_inside

_XBRLI = "http://www.xbrl.org/2003/instance"
_XBRLDI = "http://xbrl.org/2006/xbrldi"
_INLINE = {"http://www.xbrl.org/2013/inlineXBRL", "http://www.xbrl.org/2008/inlineXBRL"}
_INCORPORATION = "EntityIncorporationStateCountryCode"
_CONCEPTS = {"TradingSymbol", "Security12bTitle", "SecurityExchangeName", "EntityRegistrantName", "EntityCentralIndexKey", _INCORPORATION}
_MAX_BODY_BYTES = 16 * 1024 * 1024


class SecurityClassIssuerMismatch(DataReadinessError):
    """An explicit source issuer contradicts the requested issuer identity."""


@dataclass(frozen=True)
class InlineSourceFact:
    """Whitespace-normalized displayed text; format is recorded, not evaluated."""

    concept: str
    value: str
    contextref: str
    format: str | None = None


@dataclass(frozen=True)
class ReportingContext:
    """Dates describe the reporting context only, never ticker validity."""

    contextref: str
    entity_identifier: str
    entity_scheme: str
    period: tuple[tuple[str, str], ...]
    dimensions: tuple[tuple[str, str], ...]


@dataclass(frozen=True)
class SecurityClassEvidence:
    body_sha256: str
    trading_symbol: InlineSourceFact
    security_title: InlineSourceFact
    exchange: InlineSourceFact
    issuer_name: InlineSourceFact
    issuer_cik: InlineSourceFact
    contexts: tuple[ReportingContext, ...]


def _attribute(tag: Tag, name: str) -> str:
    value = tag.get(name)
    return value if isinstance(value, str) else ""


def _qname(tag: Tag, value: str) -> tuple[str, str]:
    prefix, separator, local = value.partition(":")
    attribute = f"xmlns:{prefix.lower()}" if separator else "xmlns"
    for ancestor in (tag, *tag.parents):
        if isinstance(ancestor, Tag) and ancestor.has_attr(attribute):
            return _attribute(ancestor, attribute), local if separator else prefix
    raise DataReadinessError(f"unbound XML namespace: {value}")


def _expanded(tag: Tag, value: str) -> str:
    namespace, local = _qname(tag, value)
    return f"{{{namespace}}}{local}"


def _text(tag: Tag) -> str:
    if any(tag.has_attr(attr) for attr in ("continuedat", "escape")):
        raise DataReadinessError("unsupported transformed or continued source fact")
    if tag.has_attr("format"):
        namespace, transform = _qname(tag, _attribute(tag, "format"))
        if (_attribute(tag, "name").split(":")[-1] != "SecurityExchangeName" or transform != "exchnameen"
                or not re.fullmatch(r"http://www.sec.gov/inlineXBRL/transformation/[0-9-]+", namespace)):
            raise DataReadinessError("unsupported source fact transformation")
    if any(str(key).lower().endswith(":nil") for key in tag.attrs):
        raise DataReadinessError("nil source fact is unsupported")
    for child in tag.find_all(True):
        if ":" in child.name and _qname(child, child.name)[0] in _INLINE:
            raise DataReadinessError("unsupported nested inline source fact content")
    value = " ".join(tag.get_text("", strip=False).split())
    if not value:
        raise DataReadinessError("empty source fact")
    return value


def _cik(value: str) -> str:
    if not re.fullmatch(r"[0-9]{1,10}", value) or int(value) == 0:
        raise DataReadinessError("invalid issuer CIK")
    return value.zfill(10)


def _context(tag: Tag) -> ReportingContext:
    identifiers: list[Tag] = []
    periods: list[Tag] = []
    dimensions: dict[str, str] = {}
    structure = {
        "context": {"entity", "period", "scenario"}, "entity": {"identifier", "segment"},
        "period": {"startdate", "enddate", "instant"}, "segment": {"explicitmember"},
        "scenario": {"explicitmember"},
    }
    for child in tag.find_all(True):
        namespace, local = _qname(child, child.name)
        parent = child.parent
        if not isinstance(parent, Tag) or local not in structure.get(parent.name.split(":")[-1], set()):
            raise DataReadinessError("unsupported reporting context structure")
        if (namespace, local) == (_XBRLI, "identifier"):
            identifiers.append(child)
        elif (namespace, local) == (_XBRLI, "period"):
            periods.append(child)
        elif (namespace, local) == (_XBRLDI, "explicitmember"):
            axis = _expanded(child, _attribute(child, "dimension"))
            if axis in dimensions:
                raise DataReadinessError("duplicate context dimension")
            dimensions[axis] = _expanded(child, _text(child))
        elif (namespace, local) not in {
            (_XBRLI, name) for name in ("entity", "segment", "scenario", "startdate", "enddate", "instant")
        }:
            raise DataReadinessError("unsupported reporting context content")
    if len(identifiers) != 1 or len(periods) != 1:
        raise DataReadinessError("context requires one entity identifier and period")
    scheme = _attribute(identifiers[0], "scheme")
    if scheme != "http://www.sec.gov/CIK":
        raise DataReadinessError("unsupported context entity scheme")
    entity = _text(identifiers[0])
    _cik(entity)
    period = tuple((child.name.split(":")[-1], _text(child)) for child in periods[0].find_all(True, recursive=False))
    if tuple(name for name, _ in period) not in {("startdate", "enddate"), ("instant",)}:
        raise DataReadinessError("invalid reporting period")
    try:
        dates = [date.fromisoformat(value) for _, value in period]
        if any(day.isoformat() != value for day, (_, value) in zip(dates, period, strict=True)) or dates[0] > dates[-1]:
            raise ValueError("invalid reporting dates")
    except ValueError as exc:
        raise DataReadinessError("invalid reporting dates") from exc
    return ReportingContext(_attribute(tag, "id"), entity, scheme, period, tuple(sorted(dimensions.items())))


def _stock_axis(axis: str) -> bool:
    return bool(re.fullmatch(r"\{http://fasb.org/us-gaap/[0-9-]+\}StatementClassOfStockAxis", axis))


def _issuer_matches(candidate: ReportingContext, security: ReportingContext) -> bool:
    # Remove only the class axis from the security context. In particular, never
    # erase LegalEntityAxis or ConsolidatedEntitiesAxis to match a registrant.
    issuer_dimensions = tuple(pair for pair in security.dimensions if not _stock_axis(pair[0]))
    return (
        _cik(candidate.entity_identifier) == _cik(security.entity_identifier)
        and candidate.entity_scheme == security.entity_scheme
        and candidate.period == security.period
        and candidate.dimensions in (security.dimensions, issuer_dimensions)
    )


def _extract(soup: BeautifulSoup, digest: str, expected_cik: str, target_ticker: str) -> tuple[SecurityClassEvidence, str | None]:
    tags: dict[str, list[Tag]] = {concept: [] for concept in _CONCEPTS}
    context_tags: dict[str, Tag] = {}
    for tag in soup.find_all(True):
        local = tag.name.split(":")[-1]
        if local not in {"nonnumeric", "context"}:
            continue
        namespace, _ = _qname(tag, tag.name)
        if local == "context" and namespace == _XBRLI:
            contextref = _attribute(tag, "id")
            if not contextref or contextref in context_tags:
                raise DataReadinessError("missing or duplicate context ID")
            context_tags[contextref] = tag
        elif local == "nonnumeric" and namespace in _INLINE:
            concept_namespace, concept = _qname(tag, _attribute(tag, "name"))
            if concept in _CONCEPTS and re.fullmatch(r"https?://xbrl.sec.gov/dei/[0-9-]+", concept_namespace):
                tags[concept].append(tag)

    contexts: dict[str, ReportingContext] = {}

    def fact(tag: Tag) -> InlineSourceFact:
        contextref = _attribute(tag, "contextref")
        if contextref not in context_tags:
            raise DataReadinessError("source fact references missing context")
        if contextref not in contexts:
            contexts[contextref] = _context(context_tags[contextref])
        return InlineSourceFact(_attribute(tag, "name"), _text(tag), contextref, _attribute(tag, "format") or None)

    def row_facts(symbol_tag: Tag) -> tuple[InlineSourceFact, InlineSourceFact, InlineSourceFact]:
        symbol = fact(symbol_tag)
        row = symbol_tag.find_parent("tr")
        if row is None:
            raise DataReadinessError("target TradingSymbol has no table row")
        row_symbols = [tag for tag in tags["TradingSymbol"] if tag.find_parent("tr") is row]
        if len(row_symbols) != 1:
            raise DataReadinessError("ambiguous security classes in target table row")

        def row_fact(concept: str) -> InlineSourceFact:
            found = [tag for tag in tags[concept] if tag.find_parent("tr") is row]
            if len(found) != 1:
                raise DataReadinessError(f"target table row requires exactly one {concept}")
            result = fact(found[0])
            if result.contextref != symbol.contextref:
                raise DataReadinessError("security title/exchange context mismatch")
            return result

        return symbol, row_fact("Security12bTitle"), row_fact("SecurityExchangeName")

    def issuer_facts(security: ReportingContext) -> tuple[InlineSourceFact, InlineSourceFact]:
        names = [fact(tag) for tag in tags["EntityRegistrantName"]]
        ciks = [fact(tag) for tag in tags["EntityCentralIndexKey"]]
        names = [item for item in names if _issuer_matches(contexts[item.contextref], security)]
        ciks = [item for item in ciks if _issuer_matches(contexts[item.contextref], security)]
        if len(names) != 1 or len(ciks) != 1:
            raise DataReadinessError("missing or ambiguous issuer association")
        name_context, cik_context = contexts[names[0].contextref], contexts[ciks[0].contextref]
        if name_context.dimensions != cik_context.dimensions:
            raise DataReadinessError("issuer name/CIK context dimensions mismatch")
        if _cik(ciks[0].value) != expected_cik:
            raise SecurityClassIssuerMismatch("explicit issuer CIK differs from expected CIK")
        return names[0], ciks[0]

    matches = [tag for tag in tags["TradingSymbol"] if _text(tag) == target_ticker]
    if not matches:
        raise DataReadinessError("target TradingSymbol must occur exactly once")
    if len(matches) == 1:
        ticker, title, exchange = row_facts(matches[0])
    else:
        # Some registered notes use the same displayed symbol as the common stock.
        # Validate every competing row; only an explicitly typed debt class can be
        # excluded. Unknown securities and multiple equity rows remain ambiguous.
        try:
            rows = [row_facts(tag) for tag in matches]
        except DataReadinessError as exc:
            raise DataReadinessError("target TradingSymbol must occur exactly once unless all competing rows are verified") from exc
        row_contexts = [contexts[row[0].contextref] for row in rows]
        signatures = {(_cik(context.entity_identifier), context.entity_scheme, context.period, context.dimensions)
            for context in row_contexts}
        if len(signatures) != len(rows):
            raise DataReadinessError("target TradingSymbol must occur exactly once per distinct class context")
        issuers = [issuer_facts(contexts[row[0].contextref]) for row in rows]
        equities = []
        for index, (symbol, candidate_title, _) in enumerate(rows):
            debt_title = re.search(r"\b(notes?|bonds?|debentures?)\b", candidate_title.value, re.IGNORECASE)
            debt_only_title = re.fullmatch(r"[0-9]+(?:\.[0-9]+)?% Notes due [0-9]{4}", candidate_title.value, re.IGNORECASE)
            if debt_title is None:
                equities.append(index)
                continue
            members = [member for axis, member in contexts[symbol.contextref].dimensions if _stock_axis(axis)]
            if (debt_only_title is None or len(members) != 1
                or not re.fullmatch(r"\{https?://[^}]+\}Notes[A-Za-z0-9]*Due[A-Za-z0-9]+Member",
                    members[0], re.IGNORECASE)
                or re.search(r"Stock|Share|Equity|Warrant|Unit|Preferred|Common|Convertible|Mixed",
                    members[0].split("}")[-1], re.IGNORECASE)):
                raise DataReadinessError("competing debt title requires an unambiguous matching debt class member")
            title_kind = debt_title[1].lower().rstrip("s")
            if title_kind not in members[0].split("}")[-1].lower():
                raise DataReadinessError("competing debt title and class member disagree")
        if len(equities) != 1:
            raise DataReadinessError("target TradingSymbol must occur exactly once for an equity class")
        ticker, title, exchange = rows[equities[0]]
        equity_issuer = issuers[equities[0]]
        security_context = contexts[ticker.contextref]
        for name, cik in issuers:
            if (name.value != equity_issuer[0].value or _cik(cik.value) != _cik(equity_issuer[1].value)
                or not _issuer_matches(contexts[name.contextref], security_context)
                or not _issuer_matches(contexts[cik.contextref], security_context)):
                raise DataReadinessError("competing listed security issuer or reporting context differs")
    stock_type = re.search(r"\b(common|ordinary|preferred|capital)\s+(stock|shares)\b", title.value, re.IGNORECASE)
    if stock_type is None or re.search(r"\b(notes|bonds|debentures)\b", title.value, re.IGNORECASE):
        raise DataReadinessError("unsupported non-stock security title")
    title_classes = set(re.findall(r"\bclass\s+([a-z])\b", title.value, re.IGNORECASE))
    title_classes = {letter.upper() for letter in title_classes}
    title_types = {kind.lower() for kind, _ in re.findall(
        r"\b(common|ordinary|preferred|capital)\s+(stock|shares)\b", title.value, re.IGNORECASE)}
    if len(title_classes) > 1 or len(title_types) > 1:
        raise DataReadinessError("ambiguous stock classes or types in security title")
    security = contexts[ticker.contextref]
    class_members = [member for axis, member in security.dimensions if _stock_axis(axis)]
    if len(class_members) > 1:
        raise DataReadinessError("ambiguous stock-class dimensions")
    for member in class_members:
        local = member.split("}")[-1]
        generic = re.fullmatch(r"\{http://fasb.org/us-gaap/[0-9-]+\}(CommonStockMember|PreferredStockMember)", member)
        common_class = re.fullmatch(r"\{http://fasb.org/us-gaap/[0-9-]+\}CommonClass([A-Z])Member", member)
        capital_class = re.fullmatch(r"\{https?://[^}]+\}CapitalClass([A-Z])Member", member)
        if common_class or capital_class:
            explicit = common_class or capital_class
            assert explicit is not None
            expected_kind = "common" if common_class else "capital"
            if title_classes != {explicit[1]} or stock_type[1].lower() != expected_kind:
                raise DataReadinessError("stock title contradicts explicit class member")
        elif not generic:
            raise DataReadinessError("unsupported stock-class member")
        elif (local == "CommonStockMember" and stock_type[1].lower() not in {"common", "ordinary"}
                or local == "PreferredStockMember" and stock_type[1].lower() != "preferred"):
            raise DataReadinessError("stock title contradicts class dimension")
    if stock_type[1].lower() == "capital" and (len(class_members) != 1 or not re.fullmatch(
            r"\{https?://[^}]+\}CapitalClass([A-Z])Member", class_members[0])):
        raise DataReadinessError("capital stock requires an explicit matching class member")

    issuer_name, issuer_cik = issuer_facts(security)
    selected = {item.contextref for item in (ticker, title, exchange, issuer_name, issuer_cik)}
    incorporation = None
    if issuer_name.contextref == issuer_cik.contextref:
        same_context = [tag for tag in tags[_INCORPORATION] if _attribute(tag, "contextref") == issuer_name.contextref]
        if len(same_context) == 1:
            try:
                code = _text(same_context[0])
                if re.fullmatch(r"[A-Z]{2}", code):
                    incorporation = code
            except DataReadinessError:
                # Unsupported optional evidence does not affect class admission.
                pass
    facts = SecurityClassEvidence(
        digest, ticker, title, exchange, issuer_name, issuer_cik, tuple(contexts[key] for key in sorted(selected)),
    )
    return facts, incorporation



def extract_security_class_and_jurisdiction(
    path: Path, expected_sha256: str, expected_cik: str, target_ticker: str,
) -> tuple[SecurityClassEvidence, str | None]:
    """Read one bounded retained body with an optional same-context jurisdiction.

    Only bytes matching the caller's pinned digest are parsed. Unsupported contexts,
    transformed facts and ambiguous associations fail closed with DataReadinessError.
    This does not verify SEC provenance, historical availability or ticker lifetimes.
    """
    if not re.fullmatch(r"[0-9a-f]{64}", expected_sha256):
        raise DataReadinessError("expected body SHA256 must be lowercase hexadecimal")
    cik = _cik(expected_cik)
    if not target_ticker or target_ticker != target_ticker.strip():
        raise DataReadinessError("target ticker must be nonempty and exact")
    try:
        contained = resolve_existing_file_inside(path.parent, path.name, label="SEC stock-class body")
        with contained.open("rb") as stream:
            body = stream.read(_MAX_BODY_BYTES + 1)
    except OSError as exc:
        raise DataReadinessError("SEC stock-class body is unreadable") from exc
    if len(body) > _MAX_BODY_BYTES:
        raise DataReadinessError("SEC stock-class body exceeds size limit")
    digest = hashlib.sha256(body).hexdigest()
    if digest != expected_sha256:
        raise DataReadinessError("SEC stock-class body hash mismatch")
    soup = BeautifulSoup(body, "html.parser")
    try:
        return _extract(soup, digest, cik, target_ticker)
    finally:
        soup.decompose()


def extract_security_class_evidence(
    path: Path, expected_sha256: str, expected_cik: str, target_ticker: str,
) -> SecurityClassEvidence:
    """Read exact retained class and issuer facts with the original public shape."""
    facts, _ = extract_security_class_and_jurisdiction(path, expected_sha256, expected_cik, target_ticker)
    return facts
