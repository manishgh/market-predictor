"""Feature-side lexical news cues, not verified events or predicted stock returns.

Input text stays unchanged. Offsets use Python Unicode code points. Publication
and version availability are caller-supplied source facts, never inferred from an
event date mentioned in text. Rule cues do not authorize training or source QA.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from itertools import chain
from typing import Literal

from market_predictor.research.news_sample_buckets import EnrichedNewsReference, NewsSampleBuckets, Sentiment

AttributionScope = Literal["company", "sector", "market", "unresolved"]
BusinessDirection = Literal["up", "down", "favorable", "adverse", "unclear"]
CueStatus = Literal["announced", "rumor", "planned", "negated", "unclear"]


@dataclass(frozen=True, slots=True)
class EnrichmentLimits:
    max_text_characters: int = 100_000
    max_cues: int = 64
    max_quote_characters: int = 320

    def __post_init__(self) -> None:
        for value in (self.max_text_characters, self.max_cues, self.max_quote_characters):
            if type(value) is not int or value < 1:
                raise ValueError("enrichment limits must be positive integers")
        if self.max_quote_characters < 80:
            raise ValueError("max_quote_characters must be at least 80")


@dataclass(frozen=True, slots=True)
class NewsCue:
    category: str
    business_direction: BusinessDirection
    status: CueStatus
    rule_id: str
    quote: str
    start: int
    end: int
    actor_role: Literal["analyst", "unspecified"]


@dataclass(frozen=True, slots=True)
class NewsFeatureEnrichment:
    document_ref: str
    version_ref: str
    source: str
    published_at_utc: datetime
    available_at_utc: datetime
    attribution_scope: AttributionScope
    categories: tuple[str, ...]
    cues: tuple[NewsCue, ...]
    existing_sentiment_score: float | None
    existing_sentiment_label: Sentiment | None
    rule_signal_state: Sentiment
    total_text_characters: int
    inspected_text_characters: int
    truncation_reasons: tuple[str, ...]
    method: Literal["rule_cues"] = "rule_cues"

    def to_qa_reference(self) -> EnrichedNewsReference:
        """QA mixedness includes opposing business cues, not a stock-return forecast.

        Existing observed sentiment remains separately available on this result.
        Absent model sentiment, heuristic directions are only QA strata; neutral
        sentiment is never invented from absence of a cue. Confidence is unscored.
        """
        sentiment = self.rule_signal_state
        if sentiment != "mixed" and self.existing_sentiment_label is not None:
            sentiment = self.existing_sentiment_label
        return EnrichedNewsReference(
            self.document_ref,
            self.version_ref,
            self.source,
            self.published_at_utc,
            self.categories,
            sentiment,
            "unavailable",
        )


@dataclass(frozen=True, slots=True)
class _Rule:
    category: str
    topic: re.Pattern[str]
    up: re.Pattern[str] | None = None
    down: re.Pattern[str] | None = None
    favorable: re.Pattern[str] | None = None
    adverse: re.Pattern[str] | None = None


def _rx(pattern: str) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + pattern + r")\b", re.IGNORECASE)


_UP = _rx(r"raises?|raised|raising|increases?|increased|increasing|grew|grows?|growth|rose|rises?|higher|beats?|beat|exceeds?|exceeded")
_DOWN = _rx(r"cuts?|cutting|lowers?|lowered|lowering|decreases?|decreased|declines?|declined|fell|falls?|lower|misses?|missed")
_RULES = (
    _Rule("earnings", _rx(r"earnings|revenue|revenues|EPS|profits?|quarterly results|financial results"), _UP, _DOWN),
    _Rule("guidance", _rx(r"guidance|outlook|(?:revenue|earnings|profit|sales) forecast"), _UP, _DOWN),
    _Rule(
        "contracts_partnerships",
        _rx(r"contracts?|partnerships?|supply agreement|commercial agreement"),
        favorable=_rx(r"wins?|won|awarded|secured|secures?|signed|signs?|renewed|renews?"),
        adverse=_rx(r"lost|loses?|terminated|terminates?|cancelled|canceled|cancels?"),
    ),
    _Rule("corporate_transactions", _rx(r"acquisitions?|acquires?|acquired|mergers?|takeover|buyout|divestiture|spin.off|asset sale")),
    _Rule(
        "analyst_rating_target",
        _rx(r"analysts?|price targets?|upgrades?|upgraded|downgrades?|downgraded|brokerage|rating"),
        _UP,
        _DOWN,
        favorable=_rx(r"upgrades?|upgraded"),
        adverse=_rx(r"downgrades?|downgraded"),
    ),
    _Rule(
        "products_clinical",
        _rx(r"products?|launch(?:es|ed)?|clinical|trial|drug|FDA|commercial milestone"),
        favorable=_rx(r"approved|approval|successful|success|met (?:its |the )?(?:primary )?endpoint"),
        adverse=_rx(r"failed|fails?|recall|recalled|rejected|rejects?|missed (?:its |the )?(?:primary )?endpoint"),
    ),
    _Rule(
        "legal_regulatory",
        _rx(r"lawsuits?|litigation|sued|court|regulatory|regulator|antitrust|investigation|fines?|penalty"),
        favorable=_rx(r"dismissed|cleared|acquitted"),
        adverse=_rx(r"sued|fined|penalty|charged|sanctioned"),
    ),
    _Rule(
        "capital_returns_financing",
        _rx(r"dividends?|buybacks?|repurchases?|financing|debt offering|equity offering|capital raise|bankruptcy"),
        _UP,
        _DOWN,
        adverse=_rx(r"bankruptcy|defaulted|defaults?"),
    ),
    _Rule(
        "management_operations_security",
        _rx(r"CEO|CFO|chief executive|resigns?|resigned|layoffs?|factory|plant|outage|cyberattack|data breach|production"),
        favorable=_rx(r"restored|reopened|resumed"),
        adverse=_rx(r"outage|cyberattack|data breach|shutdown|halted|layoffs?"),
    ),
    _Rule(
        "sector_market",
        _rx(r"sector|industry|inflation|interest rates?|central bank|Federal Reserve|tariffs?|war|oil prices?|GDP|recession"),
        _UP,
        _DOWN,
        adverse=_rx(r"war|recession"),
    ),
)
_NEGATED = _rx(r"not|never|no longer|denies?|denied|without")
_RUMOR = _rx(r"rumou?rs?|rumou?red|reportedly|unconfirmed|speculation|may|might|could")
_PLANNED = _rx(r"plans?|planned|planning|proposes?|proposed|intends?|intended|will|expects?|expected|seeks?|seeking")
_ANNOUNCED = _rx(
    r"announces?|announced|reports?|reported|completed|closed|signed|won|awarded|lost|terminated|approved|rejected|"
    r"launched|raises?|raised|cuts?|lowered|upgraded|downgraded|grew|fell|sued|fined|resigned|restored"
)
# Clause boundaries limit unrelated direction/status transfer. This remains a cue
# heuristic, not grammatical coreference or actor/company identity resolution.
_BOUNDARY = re.compile(r"[;!?\n]+|\.(?=\s|$)|\b(?:but|while|whereas|however|and)\b", re.IGNORECASE)


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} must be an aware UTC datetime")


def _status(quote: str) -> CueStatus:
    if _NEGATED.search(quote):
        return "negated"
    if _RUMOR.search(quote):
        return "rumor"
    if _PLANNED.search(quote):
        return "planned"
    if _ANNOUNCED.search(quote):
        return "announced"
    return "unclear"


def _signal(cues: list[NewsCue]) -> Sentiment:
    directions = {cue.business_direction for cue in cues if cue.status != "negated"}
    positive = bool(directions & {"up", "favorable"})
    negative = bool(directions & {"down", "adverse"})
    if positive and negative:
        return "mixed"
    if positive:
        return "positive"
    if negative:
        return "negative"
    return "unavailable"


def enrich_news_features(
    *,
    document_ref: str,
    version_ref: str,
    source: str,
    published_at_utc: datetime,
    available_at_utc: datetime,
    text: str,
    attribution_scope: AttributionScope = "unresolved",
    existing_sentiment_score: float | None = None,
    existing_sentiment_label: Sentiment | None = None,
    as_of_utc: datetime | None = None,
    limits: EnrichmentLimits = EnrichmentLimits(),
) -> NewsFeatureEnrichment:
    """Recognize bounded source-text cues without changing source or claiming facts.

    available_at_utc must describe THIS version, not the original story's timestamp.
    An optional cutoff rejects a revised version not yet available. Missing fiscal
    periods do not suppress general news; no fiscal year, EPS surprise, entity
    identity, confidence or sentiment model output is inferred by these rules.
    """
    _utc(published_at_utc, "published_at_utc")
    _utc(available_at_utc, "available_at_utc")
    if available_at_utc < published_at_utc:
        raise ValueError("available_at_utc must not precede publication")
    if as_of_utc is not None:
        _utc(as_of_utc, "as_of_utc")
        if available_at_utc > as_of_utc:
            raise ValueError("this document version is not yet available at the cutoff")
    if not isinstance(text, str):
        raise ValueError("text must be unchanged source text")
    if attribution_scope not in ("company", "sector", "market", "unresolved"):
        raise ValueError("unsupported attribution_scope")
    if existing_sentiment_score is not None and (
        type(existing_sentiment_score) not in (float, int) or not math.isfinite(existing_sentiment_score)
    ):
        raise ValueError("existing_sentiment_score must be a finite actual score or None")
    # Reuse the QA reference owner's bounds/state validation; no fake score is used.
    EnrichedNewsReference(
        document_ref,
        version_ref,
        source,
        published_at_utc,
        (),
        existing_sentiment_label if existing_sentiment_label is not None else "unavailable",
        "unavailable",
    )
    inspected = min(len(text), limits.max_text_characters)
    prefix = text[:inspected]
    reasons: set[str] = set()
    if inspected < len(text):
        reasons.add("text_character_limit")
    cues: list[NewsCue] = []
    cursor = 0
    boundaries = _BOUNDARY.finditer(prefix)
    spans = ((match.start(), match.end()) for match in boundaries)
    # Append a final sentinel without materializing all sentence boundaries.
    for boundary_start, boundary_end in chain(spans, ((len(prefix), len(prefix)),)):
        segment_start, segment_end = cursor, boundary_start
        cursor = boundary_end
        # Status belongs to the source clause, not an artificial quote window.
        status = _status(prefix[segment_start:segment_end])
        while segment_start < segment_end:
            while segment_start < segment_end and prefix[segment_start].isspace():
                segment_start += 1
            end = min(segment_end, segment_start + limits.max_quote_characters)
            window_capped = end < segment_end
            while end > segment_start and prefix[end - 1].isspace():
                end -= 1
            if end == segment_start:
                break
            if window_capped:
                reasons.add("quote_window_limit")
            quote = text[segment_start:end]
            analyst = bool(_RULES[4].topic.search(quote))
            for rule in _RULES:
                if not rule.topic.search(quote):
                    continue
                # Analyst forecast/target prose is not automatically issuer guidance.
                if rule.category == "guidance" and analyst:
                    continue
                directions: list[BusinessDirection] = []
                if status != "negated":
                    patterns: tuple[tuple[BusinessDirection, re.Pattern[str] | None], ...] = (
                        ("up", rule.up),
                        ("down", rule.down),
                        ("favorable", rule.favorable),
                        ("adverse", rule.adverse),
                    )
                    for direction, pattern in patterns:
                        if pattern is not None and pattern.search(quote):
                            directions.append(direction)
                if not directions:
                    directions.append("unclear")
                for direction in directions:
                    if len(cues) >= limits.max_cues:
                        reasons.add("cue_limit")
                        break
                    cues.append(
                        NewsCue(
                            rule.category,
                            direction,
                            status,
                            rule.category + ".lexical",
                            quote,
                            segment_start,
                            end,
                            "analyst" if analyst else "unspecified",
                        )
                    )
                if "cue_limit" in reasons:
                    break
            if "cue_limit" in reasons:
                break
            segment_start = end
        if "cue_limit" in reasons:
            # Scanning stopped at this exact source-text position, not the prefix end.
            inspected = end
            break
    categories = tuple(sorted({cue.category for cue in cues})) or ("other_unresolved",)
    return NewsFeatureEnrichment(
        document_ref,
        version_ref,
        source,
        published_at_utc,
        available_at_utc,
        attribution_scope,
        categories,
        tuple(cues),
        existing_sentiment_score,
        existing_sentiment_label,
        _signal(cues),
        len(text),
        inspected,
        tuple(sorted(reasons)),
    )


def add_enrichment_sample(result: NewsFeatureEnrichment, sampler: NewsSampleBuckets) -> None:
    """Side-branch QA collection only; the caller retains every enriched output."""
    sampler.add(result.to_qa_reference())
