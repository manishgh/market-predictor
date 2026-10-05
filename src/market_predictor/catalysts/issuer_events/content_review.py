"""Source-bound earnings/guidance review candidates, never model admission.

Both historical and live callers use these adapters and the same extraction.
The caller still verifies query/filing ownership, causal aliases and their pinned
authorities. Text matches are proposed annotations; independent precision and
recall review is required before a downstream qualification authority can exist.
"""
from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import datetime
from html.parser import HTMLParser
from typing import Any, Literal

import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256

MAX_CONTENT_BYTES = 16 * 1024 * 1024
_HASH = re.compile(r"[0-9a-f]{64}")
_BLOCKS = frozenset({"p", "div", "br", "tr", "td", "th", "li", "h1", "h2", "h3", "section", "article"})
_OMIT = frozenset({"script", "style", "head", "noscript", "template"})
_PERIOD = re.compile(
    r"\b(?:Q[1-4]\s*(?:FY\s*)?(?:20)?\d{2}|(?:FY|fiscal\s+year|full[ -]year)\s*(?:20)?\d{2}"
    r"|(?:first|second|third|fourth|1st|2nd|3rd|4th)\s+quarter(?:\s+(?:of\s+)?(?:fiscal\s+)?)?(?:20)\d{2}"
    r"|(?:quarter|year)\s+ended\s+[A-Za-z]+\s+\d{1,2},?\s+20\d{2})\b", re.IGNORECASE,
)
_PREVIEW = re.compile(r"\b(?:preview|will\s+report|expects?\s+to\s+report|scheduled\s+to\s+report|"
                      r"ahead\s+of\s+(?:earnings|results)|what\s+to\s+expect|stocks\s+to\s+watch|roundup)\b", re.IGNORECASE)
_POLICY = {
    "schema": "market_predictor.issuer_content_review_candidates",
    "families": ["earnings", "guidance"],
    "subject": "explicit_caller_bound_issuer_alias_immediately_precedes_action",
    "earnings": "reported_results_not_preview_requires_result_or_metric_text",
    "guidance": "explicit_raise_or_lower_requires_guidance_or_forecast_text",
    "period": "explicit_span_or_missing_never_infer_from_publication",
    "html": "visible_text_nodes_with_element_path_and_source_line_column",
    "availability": "max_exact_version_and_identity_clock_observed_also_requires_first_seen",
    "deduplication": "identical_normalized_sentence_family_action_once_per_content_version",
    "admission": "none_precision_and_recall_review_required",
    "alpaca_record_hash": "original_producer_json_dumps_ensure_ascii_sort_keys_default_str_default_separators",
}
EXTRACTION_POLICY_SHA256 = json_sha256(_POLICY)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _clock(value: Any, name: str) -> pd.Timestamp:
    _require(isinstance(value, (str, datetime, pd.Timestamp)), f"{name} requires an aware timestamp")
    try:
        stamp = pd.Timestamp(value)
        _require(not pd.isna(stamp) and stamp.tzinfo is not None, f"{name} requires an aware timestamp")
        return stamp.tz_convert("UTC").as_unit("ns", round_ok=False)
    except (ValueError, TypeError, OverflowError) as exc:
        raise DataReadinessError(f"{name} requires exact UTC nanoseconds") from exc


@dataclass(frozen=True)
class IssuerContentContext:
    """Caller-bound identity and observation evidence; hashes alone are not admission."""

    event_id: str
    security_id: str
    ticker: str
    issuer_aliases: tuple[str, ...]
    identity_authority_sha256: str
    identity_available_at_utc: pd.Timestamp
    first_seen_at_utc: pd.Timestamp
    availability_semantics: Literal["observed", "historical_proxy"]
    availability_policy_sha256: str

    def __post_init__(self) -> None:
        for name in ("event_id", "security_id", "ticker"):
            value = getattr(self, name)
            _require(isinstance(value, str) and bool(value) and value == value.strip(), f"invalid content {name}")
        _require(isinstance(self.issuer_aliases, tuple) and bool(self.issuer_aliases), "content requires causal issuer aliases")
        _require(all(isinstance(alias, str) and len(alias.strip()) >= 2 and alias == alias.strip()
                     for alias in self.issuer_aliases), "invalid issuer alias")
        _require(len({alias.casefold() for alias in self.issuer_aliases}) == len(self.issuer_aliases), "duplicate issuer aliases")
        for name in ("identity_authority_sha256", "availability_policy_sha256"):
            _require(_HASH.fullmatch(getattr(self, name)) is not None, f"invalid content {name}")
        _require(self.availability_semantics in ("observed", "historical_proxy"), "unknown content availability semantics")
        for name in ("identity_available_at_utc", "first_seen_at_utc"):
            object.__setattr__(self, name, _clock(getattr(self, name), name))


@dataclass(frozen=True)
class TextSegment:
    start: int
    end: int
    source_locator: str


@dataclass(frozen=True)
class IssuerContentEvidence:
    context: IssuerContentContext
    source_family: Literal["alpaca", "sec"]
    payload_sha256: str
    source_version_sha256: str
    source_locator: str
    chosen_field: str
    chosen_field_sha256: str
    is_html: bool
    text: str
    text_sha256: str
    text_segments: tuple[TextSegment, ...]
    published_at_utc: pd.Timestamp
    version_available_at_utc: pd.Timestamp
    available_at_utc: pd.Timestamp
    extraction_policy_sha256: str
    content_kind: str


class _HtmlText(HTMLParser):
    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.stack: list[tuple[str, str, bool]] = []
        self.counts: dict[str, int] = {}
        self.nodes: list[tuple[str, str]] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        parent = self.stack[-1][1] if self.stack else ""
        key = f"{parent}/{tag}"
        self.counts[key] = self.counts.get(key, 0) + 1
        path = f"{key}[{self.counts[key]}]"
        attributes = dict(attrs)
        style = re.sub(r"\s+", "", attributes.get("style") or "").lower()
        omitted = (tag in _OMIT or "hidden" in attributes or attributes.get("aria-hidden") == "true"
                   or "display:none" in style or "visibility:hidden" in style or bool(self.stack and self.stack[-1][2]))
        if tag in _BLOCKS and not omitted:
            self.nodes.append(("\n", path))
        if tag not in {"br", "hr", "img", "meta", "link", "input", "wbr", "area", "base", "embed", "source", "track", "col", "param"}:
            self.stack.append((tag, path, omitted))

    def handle_startendtag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        self.handle_starttag(tag, attrs)
        self.handle_endtag(tag)

    def handle_endtag(self, tag: str) -> None:
        for index in range(len(self.stack) - 1, -1, -1):
            if self.stack[index][0] == tag:
                omitted = self.stack[index][2]
                self.stack = self.stack[:index]
                if tag in _BLOCKS and not omitted:
                    self.nodes.append(("\n", f"/{tag}/end"))
                return

    def handle_data(self, data: str) -> None:
        if not self.stack or not self.stack[-1][2]:
            line, column = self.getpos()
            path = self.stack[-1][1] if self.stack else "/text"
            self.nodes.append((data, f"{path}:line={line}:column={column}"))


def _extract(text: str, *, html: bool, locator: str) -> tuple[str, tuple[TextSegment, ...]]:
    if html:
        parser = _HtmlText()
        parser.feed(text)
        parser.close()
        nodes = parser.nodes
    else:
        nodes = [(text, locator)]
    result: list[str] = []
    segments: list[TextSegment] = []
    position = 0
    for node, path in nodes:
        # Block boundaries stay separate from inline whitespace; this never joins
        # table rows or adjacent paragraphs into one fabricated event statement.
        if node == "\n":
            if result and result[-1] != "\n":
                result.append("\n")
                position += 1
            continue
        normalized = re.sub(r"\s+", " ", node).strip()
        if not normalized:
            continue
        if result and result[-1] != "\n":
            result.append(" ")
            position += 1
        result.append(normalized)
        segments.append(TextSegment(position, position + len(normalized), path))
        position += len(normalized)
    # A trailing boundary is not text evidence; no segment points into it.
    return "".join(result).rstrip("\n"), tuple(segments)


def extract_source_text(text: str, *, html: bool, locator: str) -> tuple[str, tuple[TextSegment, ...]]:
    """Normalize source text for blind review even when issuer identity is unknown.

    This creates no event, identity proof or feature availability timestamp.
    The source-only review population must retain readable identity failures.
    """
    _require(isinstance(text, str) and len(text.encode("utf-8")) <= MAX_CONTENT_BYTES,
             "source review text exceeds byte limit or is not text")
    _require(type(html) is bool and isinstance(locator, str) and bool(locator.strip()),
             "source review text requires its original format and locator")
    return _extract(text, html=html, locator=locator)


def _evidence(*, context: IssuerContentContext, source: Literal["alpaca", "sec"], payload_sha256: str,
              locator: str, chosen: str, html: bool, published: pd.Timestamp, version: pd.Timestamp,
              content_kind: str, purpose: Literal["historical_research", "live_construction"]) -> IssuerContentEvidence:
    _require(purpose in ("historical_research", "live_construction"), "unknown content construction purpose")
    if purpose == "live_construction":
        _require(context.availability_semantics == "observed", "live content cannot use historical proxy availability")
    _require(context.first_seen_at_utc >= version, "content observation precedes its retained version")
    text, segments = extract_source_text(chosen, html=html, locator=locator)
    clocks = [version, context.identity_available_at_utc]
    if context.availability_semantics == "observed":
        clocks.append(context.first_seen_at_utc)
    return IssuerContentEvidence(context, source, payload_sha256, payload_sha256, locator,
                                 chosen, _sha(chosen.encode("utf-8")), html, text, _sha(text.encode("utf-8")), segments,
                                 published, version, max(clocks), EXTRACTION_POLICY_SHA256, content_kind)


def alpaca_content_evidence(*, record: dict[str, Any], expected_record_sha256: str, context: IssuerContentContext,
                            purpose: Literal["historical_research", "live_construction"] = "historical_research",
                            ) -> IssuerContentEvidence:
    """Bind one original provider record; body/summary/headline provenance stays explicit."""
    _require(_HASH.fullmatch(expected_record_sha256) is not None, "invalid expected Alpaca record hash")
    # Preserve the original news producer's raw-record encoding, including spaces.
    # The compact semantic JSON hash used by manifests is a different authority.
    payload = json.dumps(record, ensure_ascii=True, sort_keys=True, default=str).encode("utf-8")
    _require(len(payload) <= MAX_CONTENT_BYTES, "Alpaca record exceeds content byte limit")
    _require(_sha(payload) == expected_record_sha256, "Alpaca record hash differs from saved version")
    published = _clock(record.get("created_at"), "Alpaca publication")
    updated = published if record.get("updated_at") is None else _clock(record["updated_at"], "Alpaca revision")
    _require(updated >= published, "Alpaca revision precedes publication")
    for field in ("content", "summary", "headline"):
        value = record.get(field)
        _require(value is None or isinstance(value, str), f"Alpaca {field} is not text")
        if isinstance(value, str) and value.strip():
            return _evidence(context=context, source="alpaca", payload_sha256=expected_record_sha256,
                             locator=f"/{field}", chosen=value,
                             html=field == "content", published=published, version=updated,
                             content_kind=field, purpose=purpose)
    raise DataReadinessError("Alpaca record has no nonblank content, summary or headline")


def sec_content_evidence(*, body: bytes, expected_body_sha256: str, accepted_at_utc: pd.Timestamp,
                         acceptance_clock_authority_sha256: str, context: IssuerContentContext,
                         document_locator: str, encoding: Literal["utf-8", "windows-1252", "ascii"] = "utf-8",
                         purpose: Literal["historical_research", "live_construction"] = "historical_research",
                         ) -> IssuerContentEvidence:
    """Bind a caller-verified SEC HTML document and corrected acceptance authority.

    PDF/image content requires its own extraction evidence and is not decoded here.
    This adapter neither opens collections nor unseals later-window documents.
    """
    _require(isinstance(body, bytes) and len(body) <= MAX_CONTENT_BYTES, "SEC content exceeds byte limit or is not bytes")
    _require(_HASH.fullmatch(expected_body_sha256) is not None and _sha(body) == expected_body_sha256,
             "SEC document hash differs from archived body")
    _require(_HASH.fullmatch(acceptance_clock_authority_sha256) is not None, "SEC content requires corrected acceptance authority")
    _require(isinstance(document_locator, str) and bool(document_locator.strip()), "SEC document locator is missing")
    _require(not body.startswith(b"%PDF"), "SEC PDF requires independently bound text extraction")
    _require(encoding in ("utf-8", "windows-1252", "ascii"), "unsupported SEC document encoding")
    try:
        chosen = body.decode(encoding, errors="strict")
    except UnicodeDecodeError as exc:
        raise DataReadinessError("SEC body cannot be decoded with its declared encoding") from exc
    accepted = _clock(accepted_at_utc, "corrected SEC acceptance")
    evidence = _evidence(context=context, source="sec", payload_sha256=expected_body_sha256,
                         locator=document_locator, chosen=chosen, html=True, published=accepted, version=accepted,
                         content_kind="filing_document", purpose=purpose)
    # Keep the separate clock authority and decoding choice in the text-policy binding.
    policy = json_sha256({"policy": EXTRACTION_POLICY_SHA256, "encoding": encoding,
                          "acceptance_clock_authority_sha256": acceptance_clock_authority_sha256})
    return IssuerContentEvidence(**{**evidence.__dict__, "extraction_policy_sha256": policy})


@dataclass(frozen=True)
class EvidenceSpan:
    role: Literal["statement", "issuer", "action", "fiscal_period", "result"]
    start: int
    end: int
    text: str
    source_locators: tuple[str, ...]


@dataclass(frozen=True)
class ReviewCandidate:
    candidate_id: str
    event_family: Literal["earnings", "guidance"]
    action: str
    rule_id: str
    fiscal_period: str | None
    unresolved_reasons: tuple[str, ...]
    spans: tuple[EvidenceSpan, ...]
    training_eligible: Literal[False] = False
    serving_eligible: Literal[False] = False


@dataclass(frozen=True)
class CandidateExtraction:
    evidence: IssuerContentEvidence
    candidates: tuple[ReviewCandidate, ...]
    disposition: Literal["review_candidates", "unclassified"]
    reasons: tuple[str, ...]
    training_eligible: Literal[False] = False
    serving_eligible: Literal[False] = False


def _statements(text: str, aliases: str) -> Iterator[tuple[int, int]]:
    start = 0
    abbreviations = {"inc", "corp", "co", "ltd", "plc", "mr", "ms", "dr", "vs"}
    actions = r"(?:reports?|reported|announces?|announced|posts?|posted|raises?|raised|lowers?|lowered|cuts?|cut|lifts?|lifted)"
    split = rf"\n+|;|\band\s+(?=(?:{aliases})(?!\w)\s+{actions}\b)|[.!?](?=\s|$)"
    for boundary in re.finditer(split, text, re.IGNORECASE):
        if boundary.group() == ".":
            word = re.search(r"(\w+)$", text[start:boundary.start()])
            if word is not None and word.group().casefold() in abbreviations:
                continue
        end = boundary.start() if boundary.group().startswith("\n") or boundary.group().lower().startswith("and") else boundary.end()
        if start < end:
            yield start, end
        start = boundary.end()
    if start < len(text):
        yield start, len(text)


def _span(content: IssuerContentEvidence, role: Literal["statement", "issuer", "action", "fiscal_period", "result"],
          start: int, end: int) -> EvidenceSpan:
    locators = tuple(segment.source_locator for segment in content.text_segments
                     if segment.start < end and segment.end > start)
    return EvidenceSpan(role, start, end, content.text[start:end], locators)


def extract_review_candidates(content: IssuerContentEvidence) -> CandidateExtraction:
    """Propose content-grounded annotations without claiming their truth or recall."""
    _require(_sha(content.text.encode("utf-8")) == content.text_sha256, "extracted content text hash differs")
    _require(_sha(content.chosen_field.encode("utf-8")) == content.chosen_field_sha256, "chosen content field hash differs")
    _require(all(0 <= segment.start < segment.end <= len(content.text) for segment in content.text_segments),
             "content text segment lies outside extracted text")
    text, segments = _extract(content.chosen_field, html=content.is_html, locator=content.source_locator)
    _require((text, segments) == (content.text, content.text_segments), "content text or source locators do not replay")
    _require(content.payload_sha256 == content.source_version_sha256, "content source version differs from payload")
    published = _clock(content.published_at_utc, "content publication")
    version = _clock(content.version_available_at_utc, "content version")
    available = _clock(content.available_at_utc, "content availability")
    _require(published <= version <= content.context.first_seen_at_utc, "content version clocks do not replay")
    clocks = [version, content.context.identity_available_at_utc]
    if content.context.availability_semantics == "observed":
        clocks.append(content.context.first_seen_at_utc)
    _require(available == max(clocks), "content availability clock does not replay")
    aliases = "|".join(re.escape(alias) for alias in sorted(content.context.issuer_aliases, key=len, reverse=True))
    subject = rf"(?<!\w)(?P<issuer>{aliases})(?!\w)(?:\s*\([^()\n]{{1,60}}\))?\s+"
    patterns: tuple[tuple[Literal["earnings", "guidance"], re.Pattern[str]], ...] = (
        ("earnings", re.compile(subject + r"(?P<action>reports?|reported|announces?|announced|posts?|posted)\s+"
                                r"[^\n]{0,240}?\b(?P<result>financial\s+results|results|earnings|EPS|revenue|profit)\b", re.IGNORECASE)),
        ("guidance", re.compile(subject + r"(?P<action>raises?|raised|lifts?|lifted|increases?|increased|"
                                r"lowers?|lowered|cuts?|cut|reduces?|reduced)\s+"
                                r"(?:(?:its|the|annual|quarterly|full[ -]year|fiscal|year|first|second|third|fourth|quarter|"
                                r"Q[1-4]|FY(?:20)?\d{2}|FY|20\d{2}|revenue|sales|earnings|profit|EPS|adjusted|net|operating|"
                                r"income|per[ -]share|and|diluted|GAAP|non[ -]GAAP|growth)\s+){0,12}"
                                r"(?P<result>guidance|forecast|outlook)\b", re.IGNORECASE)),
    )
    candidates: list[ReviewCandidate] = []
    seen: set[tuple[str, str, str]] = set()
    for statement_start, statement_end in _statements(content.text, aliases):
        text = content.text[statement_start:statement_end]
        if _PREVIEW.search(text):
            continue
        for family, pattern in patterns:
            match = pattern.search(text)
            if match is None:
                continue
            between = text[match.end("action"):match.start("result")]
            if re.search(r"\b(?:reports?|reported|announces?|announced|posts?|posted|raises?|raised|lowers?|lowered)\b", between,
                         re.IGNORECASE):
                continue
            action = match.group("action").casefold()
            if family == "earnings" and re.search(r"\b(?:conference\s+call|webcast|earnings\s+date)\b", text, re.IGNORECASE):
                continue
            if (family == "earnings" and action.startswith("announc")
                    and match.group("result").casefold() not in ("results", "financial results")
                    and not re.search(r"\b(?:results|EPS|revenue|profit)\b", text, re.IGNORECASE)):
                continue
            key = (family, action, re.sub(r"\s+", " ", text).strip().casefold())
            if key in seen:
                continue
            seen.add(key)
            periods = list(_PERIOD.finditer(text))
            distinct_periods = {re.sub(r"\s+", " ", period.group()).casefold() for period in periods}
            period = periods[0] if len(distinct_periods) == 1 else None

            spans = [_span(content, "statement", statement_start, statement_end)]
            roles: tuple[Literal["issuer", "action", "result"], ...] = ("issuer", "action", "result")
            for role in roles:
                left, right = match.span(role)
                spans.append(_span(content, role, statement_start + left, statement_start + right))
            if period is not None:
                spans.append(_span(content, "fiscal_period", statement_start + period.start(), statement_start + period.end()))
            fiscal = None if period is None else period.group()
            rule = f"{family}_explicit_issuer_action"
            candidate_id = json_sha256({"event": content.context.event_id, "security": content.context.security_id,
                                       "version": content.source_version_sha256, "text": content.text_sha256,
                                       "policy": content.extraction_policy_sha256, "rule": rule, "statement": key})
            candidates.append(ReviewCandidate(candidate_id, family, action, rule, fiscal,
                                               (("ambiguous_explicit_fiscal_periods",) if periods else
                                                ("missing_explicit_fiscal_period",)) if fiscal is None else (), tuple(spans)))
    return CandidateExtraction(content, tuple(candidates), "review_candidates" if candidates else "unclassified",
                               () if candidates else ("no_supported_explicit_issuer_event_statement",))
