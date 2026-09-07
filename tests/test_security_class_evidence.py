"""Explicit synthetic filings exercise class and multi-registrant isolation."""

from __future__ import annotations

import hashlib
from collections.abc import Callable
from dataclasses import asdict
from pathlib import Path

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.universe.security_class_evidence import SecurityClassEvidence, extract_security_class_evidence

_HEADER = '''<html xmlns="http://www.w3.org/1999/xhtml"
xmlns:ix="http://www.xbrl.org/2013/inlineXBRL"
xmlns:xbrli="http://www.xbrl.org/2003/instance"
xmlns:xbrldi="http://xbrl.org/2006/xbrldi"
xmlns:dei="http://xbrl.sec.gov/dei/2020-01-31"
xmlns:us-gaap="http://fasb.org/us-gaap/2020-01-31"
xmlns:srt="http://fasb.org/srt/2020-01-31"
xmlns:company="https://example.test/company"><body>'''


def _context(ref: str, *, dimensions: tuple[tuple[str, str], ...] = (), entity: str = "1040971",
             end: str = "2020-12-31") -> str:
    members = "".join(f'<xbrldi:explicitMember dimension="{axis}">{member}</xbrldi:explicitMember>'
                      for axis, member in dimensions)
    segment = f"<xbrli:segment>{members}</xbrli:segment>" if members else ""
    return (f'<xbrli:context id="{ref}"><xbrli:entity><xbrli:identifier scheme="http://www.sec.gov/CIK">'
            f'{entity}</xbrli:identifier>{segment}</xbrli:entity><xbrli:period>'
            f'<xbrli:startDate>2020-01-01</xbrli:startDate><xbrli:endDate>{end}</xbrli:endDate>'
            '</xbrli:period></xbrli:context>')


def _fact(concept: str, value: str, ref: str) -> str:
    return f'<ix:nonNumeric name="dei:{concept}" contextRef="{ref}">{value}</ix:nonNumeric>'


def _issuer(ref: str, cik: str = "0001040971", name: str = "Parent Realty Corp") -> str:
    return _fact("EntityCentralIndexKey", cik, ref) + _fact("EntityRegistrantName", name, ref)


def _row(ref: str, ticker: str = "SLG", title: str = "Common Stock, $0.01 par value") -> str:
    return ("<table><tr><td>" + _fact("Security12bTitle", title, ref) + "</td><td>"
            + _fact("TradingSymbol", ticker, ref) + "</td><td>"
            + _fact("SecurityExchangeName", "New York Stock Exchange", ref) + "</td></tr></table>")


def _document(*parts: str) -> str:
    return _HEADER + "".join(parts) + "</body></html>"


def _extract(tmp_path: Path, html: str, *, cik: str = "1040971", ticker: str = "SLG") -> SecurityClassEvidence:
    path = tmp_path / "filing.htm"
    body = html.encode("utf-8")
    path.write_bytes(body)
    return extract_security_class_evidence(path, hashlib.sha256(body).hexdigest(), cik, ticker)


def test_exact_source_facts_and_reporting_dates_only(tmp_path: Path) -> None:
    html = _document(_context("base"), _issuer("base"), _row("base"))
    result = _extract(tmp_path, html)
    assert result.body_sha256 == hashlib.sha256(html.encode()).hexdigest()
    assert result.trading_symbol.value == "SLG"
    assert result.security_title.value == "Common Stock, $0.01 par value"
    assert result.exchange.value == "New York Stock Exchange"
    assert result.issuer_name.value == "Parent Realty Corp"
    assert result.issuer_cik.value == "0001040971"
    assert result.trading_symbol.contextref == result.issuer_cik.contextref == "base"
    assert result.contexts[0].period == (("startdate", "2020-01-01"), ("enddate", "2020-12-31"))
    assert set(asdict(result)) == {"body_sha256", "trading_symbol", "security_title", "exchange",
                                   "issuer_name", "issuer_cik", "contexts"}


@pytest.mark.parametrize("axis,member,child_cik", [
    ("srt:ConsolidatedEntitiesAxis", "srt:SubsidiariesMember", "0001492869"),
    ("dei:LegalEntityAxis", "company:XeroxCorporationMember", "000108772"),
])
def test_shared_entity_identifier_does_not_merge_registrants(tmp_path: Path, axis: str, member: str, child_cik: str) -> None:
    stock = (("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonStockMember"),)
    html = _document(_context("parent"), _context("stock", dimensions=stock),
                     _context("child", dimensions=((axis, member),)),
                     _issuer("parent"), _issuer("child", child_cik, "Other registrant"), _row("stock"))
    result = _extract(tmp_path, html)
    assert result.issuer_name.value == "Parent Realty Corp"
    assert result.issuer_cik.contextref == "parent"
    assert {context.contextref for context in result.contexts} == {"parent", "stock"}
    with pytest.raises(DataReadinessError, match="expected CIK"):
        _extract(tmp_path, html, cik=child_cik)


def test_explicit_child_issuer_can_differ_from_shared_entity_identifier(tmp_path: Path) -> None:
    entity = (("srt:ConsolidatedEntitiesAxis", "srt:SubsidiariesMember"),)
    stock = entity + (("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonStockMember"),)
    result = _extract(tmp_path, _document(_context("base"), _context("child", dimensions=entity),
                      _context("stock", dimensions=stock), _issuer("base"),
                      _issuer("child", "1492869", "Operating Partnership"), _row("stock")), cik="1492869")
    assert result.issuer_cik.value == "1492869"
    assert result.contexts[0].entity_identifier == "1040971"


def test_xerox_child_name_without_cik_is_not_parent_name(tmp_path: Path) -> None:
    html = _document(_context("base"), _context("child", dimensions=(("dei:LegalEntityAxis", "company:XeroxCorporationMember"),)),
                     _issuer("base"), _fact("EntityRegistrantName", "Xerox Corporation", "child"), _row("base"))
    assert _extract(tmp_path, html).issuer_name.value == "Parent Realty Corp"


def test_dxc_exact_ticker_never_selects_bond_or_other_class(tmp_path: Path) -> None:
    html = _document(_context("base"),
                     _context("stock", dimensions=(("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonStockMember"),)),
                     _context("bond", dimensions=(("us-gaap:StatementClassOfStockAxis", "company:SeniorNotesMember"),)),
                     _issuer("base"), _row("stock", "DXC"), _row("bond", "DXC26", "1.750% Senior Notes Due 2026"))
    assert _extract(tmp_path, html, ticker="DXC").security_title.contextref == "stock"
    with pytest.raises(DataReadinessError, match="non-stock"):
        _extract(tmp_path, html, ticker="DXC26")
    with pytest.raises(DataReadinessError, match="exactly once"):
        _extract(tmp_path, html, ticker="DX")


@pytest.mark.parametrize("extra", [_row("base"), _fact("TradingSymbol", "SLG", "base")])
def test_duplicate_target_rejected_even_if_same_value(tmp_path: Path, extra: str) -> None:
    with pytest.raises(DataReadinessError, match="exactly once"):
        _extract(tmp_path, _document(_context("base"), _issuer("base"), _row("base"), extra))


@pytest.mark.parametrize("concept", ["Security12bTitle", "SecurityExchangeName"])
def test_title_and_exchange_require_same_row_and_context(tmp_path: Path, concept: str) -> None:
    html = _document(_context("base"), _context("other"), _issuer("base"), _row("base"))
    mismatch = html.replace(f'name="dei:{concept}" contextRef="base"', f'name="dei:{concept}" contextRef="other"')
    with pytest.raises(DataReadinessError, match="context mismatch"):
        _extract(tmp_path, mismatch)
    value = "Common Stock, $0.01 par value" if concept == "Security12bTitle" else "New York Stock Exchange"
    moved = html.replace(_fact(concept, value, "base"), "").replace("</body>", _row("other", "OTHER") + "</body>")
    with pytest.raises(DataReadinessError, match="table row requires"):
        _extract(tmp_path, moved)


@pytest.mark.parametrize("parts", [
    (_context("other", end="2021-12-31"), _issuer("other")),
    (_context("other", entity="1492869"), _issuer("other")),
    (_context("other", dimensions=(("dei:LegalEntityAxis", "company:OtherMember"),)), _issuer("other")),
    (_context("other"), _issuer("base"), _issuer("other", "1492869", "Conflicting registrant")),
    (_context("other"), _fact("EntityCentralIndexKey", "1040971", "base")),
])
def test_incompatible_or_ambiguous_issuer_has_no_fallback(tmp_path: Path, parts: tuple[str, ...]) -> None:
    with pytest.raises(DataReadinessError, match="issuer association"):
        _extract(tmp_path, _document(_context("base"), _row("base"), *parts))


def test_common_title_cannot_use_preferred_context(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="contradicts class"):
        _extract(tmp_path, _document(_context("base"), _issuer("base"),
                 _context("stock", dimensions=(("us-gaap:StatementClassOfStockAxis", "us-gaap:PreferredStockMember"),)), _row("stock")))


@pytest.mark.parametrize("change,match", [
    (lambda html: html.replace('contextRef="base"', 'contextRef="missing"'), "missing context"),
    (lambda html: html.replace("SLG", "slg"), "exactly once"),
    (lambda html: html.replace("</body>", _context("base") + "</body>"), "duplicate context"),
    (lambda html: html.replace('name="dei:TradingSymbol"', 'continuedAt="more" name="dei:TradingSymbol"'), "continued"),
    (lambda html: html.replace("http://xbrl.sec.gov/dei/2020-01-31", "https://example.test/impostor"), "exactly once"),
])
def test_malformed_or_unsupported_source_is_rejected(tmp_path: Path, change: Callable[[str], str], match: str) -> None:
    with pytest.raises(DataReadinessError, match=match):
        _extract(tmp_path, change(_document(_context("base"), _issuer("base"), _row("base"))))


def test_hash_tamper_rejected_before_html_parsing(tmp_path: Path) -> None:
    path = tmp_path / "filing.htm"
    body = _document(_context("base"), _issuer("base"), _row("base")).encode()
    path.write_bytes(body + b"tamper")
    with pytest.raises(DataReadinessError, match="hash mismatch"):
        extract_security_class_evidence(path, hashlib.sha256(body).hexdigest(), "1040971", "SLG")


def test_namespace_aliases_and_nested_text_preserve_facts(tmp_path: Path) -> None:
    html = _document(_context("base"), _issuer("base"), _row("base", title="<b>Common</b> Stock &amp; Shares"))
    html = html.replace("ix:", "inline:").replace("xmlns:ix=", "xmlns:inline=")
    assert _extract(tmp_path, html).security_title.value == "Common Stock & Shares"


def test_exchange_format_is_recorded_without_transforming_displayed_text(tmp_path: Path) -> None:
    html = _document(_context("base"), _issuer("base"), _row("base"))
    html = html.replace('<body>', '<body xmlns:ixt-sec="http://www.sec.gov/inlineXBRL/transformation/2015-08-31">')
    html = html.replace('name="dei:SecurityExchangeName"', 'format="ixt-sec:exchnameen" name="dei:SecurityExchangeName"')
    result = _extract(tmp_path, html)
    assert result.exchange.value == "New York Stock Exchange"
    assert result.exchange.format == "ixt-sec:exchnameen"
    with pytest.raises(DataReadinessError, match="transformation"):
        _extract(tmp_path, html.replace("exchnameen", "unknown"))


def test_unknown_class_member_does_not_override_stock_title(tmp_path: Path) -> None:
    html = _document(_context("base"), _issuer("base"),
                     _context("stock", dimensions=(("us-gaap:StatementClassOfStockAxis", "company:SeniorNotesMember"),)),
                     _row("stock"))
    with pytest.raises(DataReadinessError, match="unsupported stock-class member"):
        _extract(tmp_path, html)


def test_aliased_inline_exclusion_is_not_silently_included(tmp_path: Path) -> None:
    html = _document(_context("base"), _issuer("base"), _row("base", title="Common Stock<ix:exclude> Other</ix:exclude>"))
    html = html.replace("ix:", "inline:").replace("xmlns:ix=", "xmlns:inline=")
    with pytest.raises(DataReadinessError, match="nested inline"):
        _extract(tmp_path, html)


def test_contained_reader_rejects_reparse_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.core import path_integrity

    monkeypatch.setattr(path_integrity, "is_reparse_point", lambda path: path.name == "filing.htm")
    with pytest.raises(DataReadinessError, match="reparse point"):
        _extract(tmp_path, _document(_context("base"), _issuer("base"), _row("base")))


def test_cik_and_name_cannot_be_combined_across_class_scopes(tmp_path: Path) -> None:
    html = _document(_context("base"),
                     _context("stock", dimensions=(("us-gaap:StatementClassOfStockAxis", "us-gaap:CommonStockMember"),)),
                     _fact("EntityCentralIndexKey", "1040971", "base"),
                     _fact("EntityRegistrantName", "Parent Realty Corp", "stock"), _row("stock"))
    with pytest.raises(DataReadinessError, match="name/CIK context dimensions mismatch"):
        _extract(tmp_path, html)


def test_distinct_equivalent_contextrefs_are_resolved_structurally(tmp_path: Path) -> None:
    html = _document(_context("base"), _context("name"),
                     _fact("EntityCentralIndexKey", "1040971", "base"),
                     _fact("EntityRegistrantName", "Parent Realty Corp", "name"), _row("base"))
    assert _extract(tmp_path, html).issuer_name.contextref == "name"


def test_body_limit_and_missing_body_fail_closed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from market_predictor.universe import security_class_evidence

    monkeypatch.setattr(security_class_evidence, "_MAX_BODY_BYTES", 10)
    with pytest.raises(DataReadinessError, match="size limit"):
        _extract(tmp_path, _document(_context("base"), _issuer("base"), _row("base")))
    with pytest.raises(DataReadinessError, match="missing"):
        extract_security_class_evidence(tmp_path / "missing.htm", "0" * 64, "1040971", "SLG")
