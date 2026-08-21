from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from market_predictor.core.errors import DataReadinessError

RELEVANCE_POLICY_VERSION = "swing.event_relevance.v1"
_TOKEN = re.compile(r"[a-z0-9]+")
_GENERIC_PATTERNS = (
    "biggest stock movers",
    "biggest movers",
    "stocks to watch",
    "trending stocks",
    "key deals this week",
    "market roundup",
    "wall street breakfast",
    "most active stocks",
)
_MATERIAL_TERMS = (
    "earnings",
    "guidance",
    "revenue",
    "profit",
    "contract",
    "order",
    "acquisition",
    "merger",
    "offering",
    "fda",
    "clinical trial",
    "approval",
    "downgrade",
    "upgrade",
    "price target",
)
_LEGAL_SUFFIXES = {
    "inc",
    "incorporated",
    "corp",
    "corporation",
    "company",
    "co",
    "ltd",
    "limited",
    "plc",
    "holdings",
    "group",
    "class",
}
_THEME_KEYWORDS: tuple[tuple[tuple[str, ...], tuple[str, ...]], ...] = (
    (
        ("aerospace", "defense"),
        ("space", "satellite", "rocket", "launch", "lunar", "defense", "aerospace"),
    ),
    (
        ("semiconductor", "chip"),
        ("semiconductor", "chip", "foundry", "wafer", "gpu", "memory"),
    ),
    (
        ("biotech", "biotechnology", "pharmaceutical"),
        ("biotech", "drug", "clinical", "trial", "fda", "therapy", "pharma"),
    ),
    (
        ("oil", "gas", "energy"),
        ("oil", "gas", "crude", "lng", "refinery", "drilling", "energy"),
    ),
    (
        ("software", "information technology"),
        ("software", "cloud", "cybersecurity", "saas", "artificial intelligence"),
    ),
    (
        ("bank", "financial", "capital markets"),
        ("bank", "lending", "credit", "deposit", "interest rate", "capital markets"),
    ),
    (
        ("automotive", "automobile"),
        ("auto", "vehicle", "ev", "electric vehicle", "automotive"),
    ),
    (
        ("telecom", "communication"),
        ("telecom", "wireless", "broadband", "network", "communications"),
    ),
    (
        ("real estate", "reit"),
        ("real estate", "reit", "property", "mortgage"),
    ),
    (
        ("utility", "utilities"),
        ("utility", "electricity", "power grid", "rate case"),
    ),
    (
        ("insurance",),
        ("insurance", "premium", "underwriting", "claims"),
    ),
    (
        ("retail",),
        ("retail", "consumer spending", "same-store sales", "e-commerce"),
    ),
    (
        ("airline", "air transportation"),
        ("airline", "air travel", "passenger", "faa"),
    ),
)


@dataclass(frozen=True, slots=True)
class SecurityMetadata:
    security_id: str
    ticker: str
    company: str
    sector: str
    industry: str


def add_event_relevance(
    events: pd.DataFrame,
    metadata: SecurityMetadata,
) -> pd.DataFrame:
    """Score provider-tagged events without treating tag presence as full relevance."""

    required = {"security_id", "ticker", "title", "summary", "text"}
    missing = sorted(required.difference(events.columns))
    if missing:
        raise DataReadinessError(
            f"event relevance input is missing columns: {missing}"
        )
    if bool(events["security_id"].astype(str).ne(metadata.security_id).any()):
        raise DataReadinessError(
            "event relevance metadata does not match the financial security"
        )
    output = events.copy()
    scored = [
        _score(
            title=str(row.title or ""),
            summary=str(row.summary or ""),
            text=str(row.text or ""),
            metadata=metadata,
        )
        for row in output.loc[:, ["title", "summary", "text"]].itertuples(
            index=False
        )
    ]
    output["relevance"] = pd.Series(
        [item[0] for item in scored],
        index=output.index,
        dtype=float,
    )
    output["relevance_basis"] = [item[1] for item in scored]
    output["relevance_policy_version"] = RELEVANCE_POLICY_VERSION
    return output


def _score(
    *,
    title: str,
    summary: str,
    text: str,
    metadata: SecurityMetadata,
) -> tuple[float, str]:
    title_normalized = _normalize(title)
    summary_normalized = _normalize(f"{summary} {text}")
    company_terms = _company_terms(metadata.company)
    theme_terms = _theme_terms(metadata.sector, metadata.industry)

    title_ticker = _contains_ticker(title, metadata.ticker)
    summary_ticker = _contains_ticker(f"{summary} {text}", metadata.ticker)
    title_company = _contains_any(title_normalized, company_terms)
    summary_company = _contains_any(summary_normalized, company_terms)
    title_theme = _contains_any(title_normalized, theme_terms)
    summary_theme = _contains_any(summary_normalized, theme_terms)
    generic = any(pattern in title_normalized for pattern in _GENERIC_PATTERNS)
    material = any(term in title_normalized for term in _MATERIAL_TERMS)

    score = 0.35
    basis: list[str] = ["provider_tag"]
    if title_ticker:
        score += 0.75
        basis.append("ticker_title")
    if title_company:
        score += 0.65
        basis.append("company_title")
    if not title_ticker and summary_ticker:
        score += 0.25
        basis.append("ticker_summary")
    if not title_company and summary_company:
        score += 0.20
        basis.append("company_summary")
    if title_theme:
        score += 0.45
        basis.append("industry_theme_title")
    elif summary_theme:
        score += 0.15
        basis.append("industry_theme_summary")
    if material:
        score += 0.10
        basis.append("material_term")
    if generic and not (title_ticker or title_company or title_theme):
        score -= 0.35
        basis.append("generic_penalty")
    return max(0.1, min(score, 2.0)), "+".join(basis)


def _company_terms(company: str) -> tuple[str, ...]:
    tokens = [
        token
        for token in _tokens(company)
        if token not in _LEGAL_SUFFIXES and len(token) >= 3
    ]
    if not tokens:
        return ()
    phrases = {" ".join(tokens)}
    phrases.update(token for token in tokens if len(token) >= 5)
    return tuple(sorted(phrases, key=lambda value: (-len(value), value)))


def _theme_terms(sector: str, industry: str) -> tuple[str, ...]:
    metadata = _normalize(f"{sector} {industry}")
    terms: set[str] = {
        token
        for token in _tokens(industry)
        if len(token) >= 5 and token not in _LEGAL_SUFFIXES
    }
    for triggers, keywords in _THEME_KEYWORDS:
        if any(trigger in metadata for trigger in triggers):
            terms.update(keywords)
    return tuple(sorted(terms, key=lambda value: (-len(value), value)))


def _normalize(value: str) -> str:
    return " ".join(_tokens(value))


def _tokens(value: str) -> list[str]:
    return _TOKEN.findall(str(value).lower())


def _contains_any(text: str, phrases: tuple[str, ...]) -> bool:
    return any(_contains_phrase(text, phrase) for phrase in phrases)


def _contains_phrase(text: str, phrase: str) -> bool:
    normalized = _normalize(phrase)
    return bool(normalized) and f" {normalized} " in f" {text} "


def _contains_ticker(text: str, ticker: str) -> bool:
    canonical = ticker.strip().upper()
    if len(canonical) == 1:
        explicit = (
            rf"\${re.escape(canonical)}(?![A-Z0-9])",
            rf"\({re.escape(canonical)}\)",
            rf"\b(?:NASDAQ|NYSE|AMEX)\s*:\s*{re.escape(canonical)}\b",
            rf"\bTICKER\s*:\s*{re.escape(canonical)}\b",
        )
        return any(re.search(pattern, str(text)) is not None for pattern in explicit)
    aliases = {
        canonical,
        canonical.replace("-", "."),
    }
    return any(
        bool(alias)
        and re.search(
            rf"(?<![A-Z0-9]){re.escape(alias)}(?![A-Z0-9])",
            str(text),
        )
        is not None
        for alias in aliases
    )
