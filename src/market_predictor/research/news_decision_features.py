"""Pure prediction-time statistics of source-text cues with explicit missingness.

The caller verifies source, exact-copy, company-link and coverage proofs. These
typed references do not grant source/model admission. No prices after a decision,
outcomes, inferred query-stock links, imputation or corpus sampling enter here.

Presence and shares describe the observed readable linked document pool. Category
absence in a nonempty pool is zero, not proof of provider-wide or world absence.
An empty pool with unknown coverage remains null; complete coverage can establish
zero presence only when no known story was suppressed.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Literal

Semantics = Literal["observed", "historical_proxy"]
Purpose = Literal["research_proxy", "observed_live"]
Direction = Literal["up", "down", "favorable", "adverse", "unclear"]
Status = Literal["announced", "rumor", "planned", "negated", "unclear"]
CATEGORIES = (
    "earnings",
    "guidance",
    "contracts_partnerships",
    "corporate_transactions",
    "analyst_rating_target",
    "products_clinical",
    "legal_regulatory",
    "capital_returns_financing",
    "management_operations_security",
    "sector_market",
    "other",
)
DIRECTIONS: tuple[Direction, ...] = ("up", "down", "favorable", "adverse")
STATUSES: tuple[Status, ...] = ("announced", "rumor", "planned", "negated", "unclear")
SOURCES = ("alpaca", "sec")
TECHNICAL_COLUMNS = ("rel_return_5d_vs_spy_xs_z", "return_1d_xs_z", "dist_sma_200_xs_z")
NEWS_COLUMNS = (
    *(
        f"news_direct_{category}_{suffix}"
        for category in CATEGORIES
        for suffix in ("present_1d", "present_3d", "up_share_3d", "down_share_3d", "favorable_share_3d", "adverse_share_3d")
    ),
    *(f"news_direct_{status}_share_3d" for status in STATUSES),
    *(
        f"news_direct_{name}_3d"
        for name in (
            "finbert_positive_mean",
            "finbert_negative_magnitude_mean",
            "finbert_score_coverage",
            "mixed_direction_share",
            "latest_age_hours",
            "truncated_text_share",
        )
    ),
    *(f"news_direct_source_coverage_known_{source}_{window}" for source in SOURCES for window in ("1d", "3d")),
    "news_earnings_up_times_guidance_down_3d",
    "news_mixed_times_relative_pullback_3d",
    "news_macro_mention_adverse_times_daily_return_3d",
    "news_positive_sentiment_times_long_trend_3d",
)


def _text(value: str, name: str) -> None:
    if not isinstance(value, str) or not value or value != value.strip() or len(value) > 512:
        raise ValueError(f"{name} requires nonempty bounded text")


def _sha(value: str, name: str) -> None:
    if not isinstance(value, str) or len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
        raise ValueError(f"{name} requires a lowercase SHA256")


def _utc(value: datetime, name: str) -> None:
    if not isinstance(value, datetime) or value.tzinfo is None or value.utcoffset() != timedelta(0):
        raise ValueError(f"{name} requires an aware UTC datetime")


def _semantics(value: Semantics) -> None:
    if value not in ("observed", "historical_proxy"):
        raise ValueError("unsupported availability semantics")


def _number(value: float | None, name: str) -> None:
    if value is not None and (type(value) not in (float, int) or not math.isfinite(value)):
        raise ValueError(f"{name} requires finite numerical evidence or None")


@dataclass(frozen=True, slots=True)
class TechnicalValue:
    value: float | None
    available_at_utc: datetime | None
    semantics: Semantics

    def __post_init__(self) -> None:
        _number(self.value, "technical value")
        _semantics(self.semantics)
        if self.available_at_utc is not None:
            _utc(self.available_at_utc, "technical clock")
        if self.value is not None and self.available_at_utc is None:
            raise ValueError("nonnull technical value requires its clock")


@dataclass(frozen=True, slots=True)
class NewsDecision:
    decision_id: str
    security_id: str
    decision_time_utc: datetime
    rel_return_5d_vs_spy_xs_z: TechnicalValue
    return_1d_xs_z: TechnicalValue
    dist_sma_200_xs_z: TechnicalValue

    def __post_init__(self) -> None:
        _text(self.decision_id, "decision_id")
        _text(self.security_id, "security_id")
        _utc(self.decision_time_utc, "decision cutoff")
        for name in TECHNICAL_COLUMNS:
            value = getattr(self, name)
            if not isinstance(value, TechnicalValue):
                raise ValueError("decision requires the exact typed parent technical values")
            if value.value is not None:
                assert value.available_at_utc is not None
                if value.available_at_utc > self.decision_time_utc:
                    raise ValueError("parent technical value is after the decision cutoff")


@dataclass(frozen=True, slots=True)
class DecisionCue:
    category: str
    business_direction: Direction
    status: Status

    def __post_init__(self) -> None:
        if self.category not in CATEGORIES or self.business_direction not in (*DIRECTIONS, "unclear") or self.status not in STATUSES:
            raise ValueError("unsupported category, business direction or status")


@dataclass(frozen=True, slots=True)
class NewsVersion:
    version_id: str
    source_family: Literal["alpaca", "sec"]
    source_document_id: str
    source_version_sha256: str
    source_proof_sha256: str
    published_at_utc: datetime | None
    known_at_utc: datetime
    cue_available_at_utc: datetime | None
    usable: bool
    categories: tuple[str, ...]
    cues: tuple[DecisionCue, ...]
    truncated: bool
    sentiment_score: float | None
    sentiment_available_at_utc: datetime | None
    semantics: Semantics
    copy_proof_sha256: str | None = None
    copy_proof_available_at_utc: datetime | None = None

    def __post_init__(self) -> None:
        _text(self.version_id, "version_id")
        _text(self.source_document_id, "source_document_id")
        if self.source_family not in SOURCES:
            raise ValueError("unsupported source family")
        _sha(self.source_version_sha256, "raw source version")
        _sha(self.source_proof_sha256, "source proof")
        _utc(self.known_at_utc, "version known clock")
        _semantics(self.semantics)
        for name in ("published_at_utc", "cue_available_at_utc", "sentiment_available_at_utc", "copy_proof_available_at_utc"):
            value = getattr(self, name)
            if value is not None:
                _utc(value, name)
        if self.published_at_utc is not None and self.known_at_utc < self.published_at_utc:
            raise ValueError("version cannot be known before publication")
        if type(self.usable) is not bool or type(self.truncated) is not bool:
            raise ValueError("version states require explicit booleans")
        if not isinstance(self.categories, tuple) or not isinstance(self.cues, tuple) or len(self.cues) > 64:
            raise ValueError("version cues must be a bounded immutable tuple")
        if len(self.categories) > len(CATEGORIES) or len(set(self.categories)) != len(self.categories):
            raise ValueError("version categories must be unique and bounded")
        if any(category not in CATEGORIES for category in self.categories):
            raise ValueError("unknown category; adapter must explicitly project other/unresolved to other")
        if any(not isinstance(cue, DecisionCue) or cue.category not in self.categories for cue in self.cues):
            raise ValueError("cue category differs from version categories")
        if self.usable:
            if not self.categories or self.published_at_utc is None or self.cue_available_at_utc is None:
                raise ValueError("usable version requires categories and source/cue clocks")
            if self.cue_available_at_utc < self.known_at_utc:
                raise ValueError("cue cannot precede known source version")
        elif self.categories or self.cues or self.sentiment_score is not None:
            raise ValueError("unusable version cannot supply feature values")
        _number(self.sentiment_score, "sentiment score")
        if self.sentiment_score is not None:
            if not -1 <= self.sentiment_score <= 1 or self.sentiment_available_at_utc is None:
                raise ValueError("sentiment requires its signed score and separate clock")
            if self.published_at_utc is None or self.sentiment_available_at_utc < self.published_at_utc:
                raise ValueError("sentiment cannot precede publication")
        elif self.sentiment_available_at_utc is not None:
            raise ValueError("missing sentiment has no score clock")
        if (self.copy_proof_sha256 is None) != (self.copy_proof_available_at_utc is None):
            raise ValueError("copy proof requires both identity and availability")
        if self.copy_proof_sha256 is not None:
            _sha(self.copy_proof_sha256, "exact query-copy proof")


@dataclass(frozen=True, slots=True)
class VerifiedCompanyLink:
    version_id: str
    security_id: str
    relation_available_at_utc: datetime
    identity_available_at_utc: datetime
    proof_sha256: str
    semantics: Semantics
    channel: Literal["direct_issuer"] = "direct_issuer"

    def __post_init__(self) -> None:
        _text(self.version_id, "linked version")
        _text(self.security_id, "linked company")
        _utc(self.relation_available_at_utc, "relation clock")
        _utc(self.identity_available_at_utc, "identity clock")
        _sha(self.proof_sha256, "company-link proof")
        _semantics(self.semantics)
        if self.channel != "direct_issuer":
            raise ValueError("company features require independently verified direct-issuer links")


@dataclass(frozen=True, slots=True)
class SourceCoverage:
    security_id: str
    source_family: Literal["alpaca", "sec"]
    start_utc: datetime
    end_utc: datetime
    available_at_utc: datetime
    complete: bool
    proof_sha256: str
    semantics: Semantics

    def __post_init__(self) -> None:
        _text(self.security_id, "coverage company")
        for value in (self.start_utc, self.end_utc, self.available_at_utc):
            _utc(value, "coverage clock")
        if self.source_family not in SOURCES or self.end_utc <= self.start_utc or type(self.complete) is not bool:
            raise ValueError("invalid source coverage interval")
        _sha(self.proof_sha256, "coverage proof")
        _semantics(self.semantics)


@dataclass(frozen=True, slots=True)
class AggregationLimits:
    max_decisions: int = 4096
    max_versions: int = 100000
    max_links: int = 200000
    max_coverage_intervals: int = 100000

    def __post_init__(self) -> None:
        if any(
            type(value) is not int or value < 1
            for value in (
                self.max_decisions,
                self.max_versions,
                self.max_links,
                self.max_coverage_intervals,
            )
        ):
            raise ValueError("aggregation limits must be positive integers")


@dataclass(frozen=True, slots=True)
class DecisionNewsFeatures:
    decision_id: str
    security_id: str
    decision_time_utc: datetime
    values: tuple[float | None, ...]
    available_at_utc: tuple[datetime | None, ...]
    selected_version_ids: tuple[str, ...]
    suppressed_story_count: int
    purpose: Purpose
    columns: tuple[str, ...] = NEWS_COLUMNS
    source_admission: bool = False
    training_eligible: bool = False
    serving_eligible: bool = False
    promotion_eligible: bool = False

    def as_values(self) -> dict[str, float | None]:
        return dict(zip(self.columns, self.values, strict=True))

    def as_clocks(self) -> dict[str, datetime | None]:
        return dict(zip(self.columns, self.available_at_utc, strict=True))


@dataclass(frozen=True, slots=True)
class _Selected:
    version: NewsVersion
    available_at_utc: datetime
    sentiment_score: float | None
    sentiment_available_at_utc: datetime | None


def _payload_signature(version: NewsVersion) -> tuple[object, ...]:
    return (
        version.published_at_utc,
        tuple(sorted(version.categories)),
        tuple(sorted({(cue.category, cue.business_direction, cue.status) for cue in version.cues})),
        version.truncated,
    )


def _prepare(
    decisions: Sequence[NewsDecision],
    versions: Sequence[NewsVersion],
    links: Sequence[VerifiedCompanyLink],
    coverage: Sequence[SourceCoverage],
    purpose: Purpose,
    limits: AggregationLimits,
) -> tuple[dict[tuple[str, str], dict[str, list[NewsVersion]]], dict[str, list[VerifiedCompanyLink]]]:
    if purpose not in ("research_proxy", "observed_live"):
        raise ValueError("explicit research_proxy or observed_live purpose is required")
    for values, maximum, expected in (
        (decisions, limits.max_decisions, NewsDecision),
        (versions, limits.max_versions, NewsVersion),
        (links, limits.max_links, VerifiedCompanyLink),
        (coverage, limits.max_coverage_intervals, SourceCoverage),
    ):
        if len(values) > maximum or any(not isinstance(value, expected) for value in values):
            raise ValueError("typed aggregation inputs exceed bounds or contain invalid objects")
    if len({decision.decision_id for decision in decisions}) != len(decisions):
        raise ValueError("decisions must have unique identities")
    if len({version.version_id for version in versions}) != len(versions):
        raise ValueError("version reference IDs must be unique")
    by_id = {version.version_id: version for version in versions}
    groups: dict[tuple[str, str], dict[str, list[NewsVersion]]] = {}
    for version in versions:
        groups.setdefault((version.source_family, version.source_document_id), {}).setdefault(version.source_version_sha256, []).append(
            version
        )
    for story in groups.values():
        for copies in story.values():
            if len(copies) > 1 and any(copy.copy_proof_sha256 is None for copy in copies):
                raise ValueError("duplicate source-document/version keys require exact query-copy proof")
            usable = [copy for copy in copies if copy.usable]
            if len({_payload_signature(copy) for copy in usable}) > 1:
                raise ValueError("proven exact copies have conflicting cue/content metadata")
            if len({copy.sentiment_score for copy in copies if copy.sentiment_score is not None}) > 1:
                raise ValueError("proven exact copies have conflicting sentiment")
    link_index: dict[str, list[VerifiedCompanyLink]] = {}
    for link in links:
        if link.version_id not in by_id:
            raise ValueError("company link references an absent source version")
        link_index.setdefault(link.version_id, []).append(link)
    if purpose == "observed_live":
        states = [version.semantics for version in versions]
        states.extend(link.semantics for link in links)
        states.extend(item.semantics for item in coverage)
        states.extend(getattr(decision, name).semantics for decision in decisions for name in TECHNICAL_COLUMNS)
        if any(state != "observed" for state in states):
            raise ValueError("observed/live construction rejects historical proxies")
    return groups, link_index


def _link_clock(link: VerifiedCompanyLink) -> datetime:
    return max(link.relation_available_at_utc, link.identity_available_at_utc)


def _select(
    decision: NewsDecision,
    groups: dict[tuple[str, str], dict[str, list[NewsVersion]]],
    links: dict[str, list[VerifiedCompanyLink]],
) -> tuple[list[_Selected], int]:
    cutoff = decision.decision_time_utc
    selected: list[_Selected] = []
    suppressed = 0
    for story_key in sorted(groups):
        story = groups[story_key]
        known: dict[str, list[NewsVersion]] = {
            digest: [copy for copy in copies if copy.known_at_utc <= cutoff] for digest, copies in story.items()
        }
        known = {digest: copies for digest, copies in known.items() if copies}
        if not known:
            continue
        # Earliest independently known exact copy establishes version knowledge;
        # later query copies do not manufacture a new revision of that same text.
        clocks = {digest: min(copy.known_at_utc for copy in copies) for digest, copies in known.items()}
        latest = max(clocks.values())
        latest_hashes = [digest for digest, clock in clocks.items() if clock == latest]
        historical_company_link = any(
            link.security_id == decision.security_id and _link_clock(link) <= cutoff
            for copies in known.values()
            for copy in copies
            for link in links.get(copy.version_id, ())
        )
        if len(latest_hashes) != 1:
            suppressed += historical_company_link
            continue
        digest = latest_hashes[0]
        candidates: list[tuple[datetime, NewsVersion]] = []
        for copy in known[digest]:
            if not copy.usable:
                continue
            assert copy.cue_available_at_utc is not None
            # Do not choose one global representative before company filtering:
            # each proven copy retains its own independently verified company links.
            for link in links.get(copy.version_id, ()):
                if link.security_id != decision.security_id:
                    continue
                effective = max(copy.known_at_utc, copy.cue_available_at_utc, _link_clock(link))
                if effective <= cutoff:
                    candidates.append((effective, copy))
        if not candidates:
            suppressed += historical_company_link
            continue
        effective, representative = min(candidates, key=lambda item: (item[0], item[1].version_id))
        # A later equivalence proof cannot withdraw independent source/link evidence.
        # It gates only evidence merged from another exact copy into this representative.
        scores: list[tuple[datetime, str, float]] = []
        for available, copy in candidates:
            if copy.sentiment_score is None or copy.sentiment_available_at_utc is None:
                continue
            score_available = max(available, copy.sentiment_available_at_utc)
            if copy.version_id != representative.version_id:
                assert copy.copy_proof_available_at_utc is not None
                assert representative.copy_proof_available_at_utc is not None
                score_available = max(
                    score_available,
                    copy.copy_proof_available_at_utc,
                    representative.copy_proof_available_at_utc,
                )
            if score_available <= cutoff:
                scores.append((score_available, copy.version_id, copy.sentiment_score))
        score_clock, score = None, None
        if scores:
            score_clock, _, score = min(scores)
        selected.append(_Selected(representative, effective, score, score_clock))
    return selected, suppressed


def _coverage_known(decision: NewsDecision, intervals: Sequence[SourceCoverage], source: str, days: int) -> bool:
    cutoff = decision.decision_time_utc
    lower = cutoff - timedelta(days=days)
    candidates = sorted(
        (item.start_utc, item.end_utc)
        for item in intervals
        if item.security_id == decision.security_id
        and item.source_family == source
        and item.complete
        and item.available_at_utc <= cutoff
        and item.end_utc >= lower
        and item.start_utc <= cutoff
    )
    cursor = lower
    for start, end in candidates:
        if start > cursor:
            return False
        cursor = max(cursor, end)
        if cursor >= cutoff:
            return True
    return False


def _directions(version: NewsVersion, category: str | None = None) -> set[Direction]:
    return {
        cue.business_direction
        for cue in version.cues
        if cue.status != "negated" and cue.business_direction != "unclear" and (category is None or cue.category == category)
    }


def _mixed(version: NewsVersion) -> bool:
    directions = _directions(version)
    return bool(directions & {"up", "favorable"}) and bool(directions & {"down", "adverse"})


def _build(
    decision: NewsDecision,
    selected: list[_Selected],
    suppressed: int,
    coverage: Sequence[SourceCoverage],
    purpose: Purpose,
) -> DecisionNewsFeatures:
    cutoff = decision.decision_time_utc
    windows = {days: [item for item in selected if cutoff - timedelta(days=days) <= item.available_at_utc <= cutoff] for days in (1, 3)}
    known = {(source, days): _coverage_known(decision, coverage, source, days) for source in SOURCES for days in (1, 3)}
    values: dict[str, float | None] = {}
    clocks: dict[str, datetime | None] = {}

    def put(name: str, value: float | None, clock: datetime | None) -> None:
        values[name] = value
        clocks[name] = None if value is None else clock

    def latest(items: list[_Selected]) -> datetime | None:
        return max((item.available_at_utc for item in items), default=None)

    for category in CATEGORIES:
        for days in (1, 3):
            items = windows[days]
            present = any(category in item.version.categories for item in items)
            complete_empty = not suppressed and all(known[source, days] for source in SOURCES)
            value = float(present) if items or complete_empty else None
            put(f"news_direct_{category}_present_{days}d", value, latest(items) if items else cutoff)
        bearing = [item for item in windows[3] if category in item.version.categories]
        for direction in DIRECTIONS:
            share = (sum(direction in _directions(item.version, category) for item in bearing) / len(bearing)) if bearing else None
            put(f"news_direct_{category}_{direction}_share_3d", share, latest(bearing))
    docs = windows[3]
    document_clock = latest(docs)
    for status in STATUSES:
        # One article contributes at most once to each status, irrespective of
        # sentence/category multiplicity. Different statuses can coexist.
        share = sum(status in ({cue.status for cue in item.version.cues} or {"unclear"}) for item in docs) / len(docs) if docs else None
        put(f"news_direct_{status}_share_3d", share, document_clock)
    scored = [item for item in docs if item.sentiment_score is not None]
    score_clock = max((item.sentiment_available_at_utc for item in scored if item.sentiment_available_at_utc is not None), default=None)
    positive = sum(max(item.sentiment_score or 0.0, 0.0) for item in scored) / len(scored) if scored else None
    negative = sum(max(-(item.sentiment_score or 0.0), 0.0) for item in scored) / len(scored) if scored else None
    put("news_direct_finbert_positive_mean_3d", positive, score_clock)
    put("news_direct_finbert_negative_magnitude_mean_3d", negative, score_clock)
    # Missing scores are a zero coverage fraction when readable documents exist,
    # never zero or neutral sentiment. The denominator is observed documents only.
    put("news_direct_finbert_score_coverage_3d", len(scored) / len(docs) if docs else None, cutoff)
    put("news_direct_mixed_direction_share_3d", sum(_mixed(item.version) for item in docs) / len(docs) if docs else None, document_clock)
    put("news_direct_latest_age_hours_3d", (cutoff - document_clock).total_seconds() / 3600 if document_clock else None, cutoff)
    put("news_direct_truncated_text_share_3d", sum(item.version.truncated for item in docs) / len(docs) if docs else None, document_clock)
    for source in SOURCES:
        for days in (1, 3):
            put(f"news_direct_source_coverage_known_{source}_{days}d", float(known[source, days]), cutoff)

    def multiply(name: str, left: str, right: str | TechnicalValue) -> None:
        a = values[left]
        b = values[right] if isinstance(right, str) else right.value
        clock_b = clocks[right] if isinstance(right, str) else right.available_at_utc
        if a is None or b is None:
            put(name, None, None)
        else:
            clock_a = clocks[left]
            assert clock_a is not None and clock_b is not None
            put(name, a * b, max(clock_a, clock_b))

    multiply("news_earnings_up_times_guidance_down_3d", "news_direct_earnings_up_share_3d", "news_direct_guidance_down_share_3d")
    multiply("news_mixed_times_relative_pullback_3d", "news_direct_mixed_direction_share_3d", decision.rel_return_5d_vs_spy_xs_z)
    multiply("news_macro_mention_adverse_times_daily_return_3d", "news_direct_sector_market_adverse_share_3d", decision.return_1d_xs_z)
    multiply("news_positive_sentiment_times_long_trend_3d", "news_direct_finbert_positive_mean_3d", decision.dist_sma_200_xs_z)
    return DecisionNewsFeatures(
        decision.decision_id,
        decision.security_id,
        cutoff,
        tuple(values[name] for name in NEWS_COLUMNS),
        tuple(clocks[name] for name in NEWS_COLUMNS),
        tuple(sorted(item.version.version_id for item in docs)),
        suppressed,
        purpose,
    )


def aggregate_news_decisions(
    *,
    decisions: Sequence[NewsDecision],
    versions: Sequence[NewsVersion],
    links: Sequence[VerifiedCompanyLink],
    coverage: Sequence[SourceCoverage] = (),
    purpose: Purpose,
    limits: AggregationLimits = AggregationLimits(),
) -> tuple[DecisionNewsFeatures, ...]:
    """The same pure transform serves bounded batch and single-decision callers.

    Source document IDs identify provider stories or individual SEC documents,
    never accession-wide bundles. Adapters project extractor other_unresolved into
    the feature category other explicitly. Raw query security is not an input.

    Each version-specific direct-issuer proof must be externally verified before
    constructing a link. Coverage proofs must cover the full specified source and
    issuer attribution denominator; an observed query alone is not known empty.
    Collection completeness flags do not certify category recognition precision.
    """
    groups, link_index = _prepare(decisions, versions, links, coverage, purpose, limits)
    results: list[DecisionNewsFeatures] = []
    for decision in decisions:
        selected, suppressed = _select(decision, groups, link_index)
        results.append(_build(decision, selected, suppressed, coverage, purpose))
    return tuple(results)
