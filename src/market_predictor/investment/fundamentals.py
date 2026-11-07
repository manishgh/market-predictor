"""Pure, research-only point-in-time accounting features for nonfinancial issuers."""

from __future__ import annotations

from collections.abc import Sequence
from datetime import date, datetime, timedelta
from functools import cache
from typing import Literal, Self

from pydantic import model_validator

from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import HoldingContract, Identifier, Sha256, Timestamp

Concept = Literal["revenue", "operating_income", "operating_cash_flow", "capex", "net_income", "assets", "cash", "debt"]
FEATURE_NAMES = (
    "revenue_growth_ttm",
    "operating_margin_ttm",
    "operating_margin_change",
    "fcf_margin_ttm",
    "capex_to_revenue_ttm",
    "cash_conversion_ttm",
)
FEATURE_DEFINITIONS = {
    "revenue_growth_ttm": "latest constructed as-of TTM / preceding year constructed as-of TTM - 1; positive denominator",
    "operating_margin_ttm": "operating income TTM / revenue TTM; positive denominator",
    "operating_margin_change": "latest operating margin TTM minus previous year operating margin TTM",
    "fcf_margin_ttm": "(operating cash flow TTM - positive cash capex TTM) / revenue TTM",
    "capex_to_revenue_ttm": "positive cash capex TTM / revenue TTM",
    "cash_conversion_ttm": "operating cash flow TTM / net income TTM; positive net income",
}
FEATURE_CONTRACT_SHA256 = json_sha256(
    {
        "definitions": FEATURE_DEFINITIONS,
        "scope": "nonfinancial",
        "research_only": True,
        "ttm_policy": "four contiguous quarters OR annual direct OR annual + same-filing current YTD - prior YTD; exact intervals",
        "debt": "deferred until qualified total interest-bearing debt is available",
        "bridge_assumption": "unchanged accounting basis across annual and comparative YTD filings; research reconstruction",
        "rejections": (
            "as-of joint valid/rejected versions; exact invalid masks period until later exact valid; "
            "malformed period blocks concept; rejected revenue end anchors current period"
        ),
        "version_policy": (
            "earlier as-of filing record usable only when value exactly matches latest period value; bind both evidence versions"
        ),
    }
)


class FilingFact(HoldingContract):
    security_id: Identifier
    cik: Identifier
    accession: Identifier
    form: Literal["10-K", "10-Q", "10-K/A", "10-Q/A"] = "10-Q"
    source_sha256: Sha256
    source_locator: Identifier
    taxonomy: Literal["us-gaap"] = "us-gaap"
    concept: Concept
    tag: Identifier
    unit: Literal["USD"] = "USD"
    actual_gaap_consolidated: Literal[True] = True
    period_kind: Literal["instant", "duration"]
    start: date | None = None
    end: date
    value: float
    published_at: Timestamp
    retrieved_at: Timestamp
    first_observed_at: Timestamp | None = None
    availability_basis: Literal["observed", "reconstructed_research"] = "reconstructed_research"

    @model_validator(mode="after")
    def validate_fact(self) -> Self:
        if self.published_at > self.retrieved_at:
            raise ValueError("publication cannot follow retrieval")
        if self.end > self.published_at.date():
            raise ValueError("financial period cannot end after publication")
        if self.period_kind == "duration":
            if self.start is None or self.start > self.end:
                raise ValueError("duration needs ordered start and end")
        elif self.start is not None:
            raise ValueError("instant cannot have start")
        if (self.concept in {"assets", "cash", "debt"}) != (self.period_kind == "instant"):
            raise ValueError("concept has incorrect instant/duration kind")
        if self.concept in {"capex", "assets", "cash", "debt"} and self.value < 0:
            raise ValueError("balance and cash expenditure concepts cannot be negative")
        if self.availability_basis == "observed" and self.first_observed_at is None:
            raise ValueError("observed facts require an actual observation clock")
        if self.first_observed_at is not None and not self.published_at <= self.first_observed_at <= self.retrieved_at:
            raise ValueError("observation must fall between publication and retrieval")
        return self

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))

    def available_at(self) -> datetime:
        return self.first_observed_at if self.availability_basis == "observed" and self.first_observed_at else self.published_at


class FilingFactRejection(HoldingContract):
    """A source-backed invalid filing version, never a silently absent fact."""

    security_id: Identifier
    cik: Identifier
    accession: Identifier
    form: Literal["10-K", "10-Q", "10-K/A", "10-Q/A"]
    source_sha256: Sha256
    source_locator: Identifier
    related_source_locators: tuple[str, ...] = ()
    taxonomy: Literal["us-gaap"] = "us-gaap"
    concept: Concept
    tag: Identifier
    unit: Literal["USD"] = "USD"
    actual_gaap_consolidated: Literal[True] = True
    period_scope: Literal["exact", "concept"]
    start: date | None = None
    end: date | None = None
    reason: Literal["invalid_fact", "ambiguous_alias_values"]
    detail: Identifier
    published_at: Timestamp
    retrieved_at: Timestamp
    first_observed_at: Timestamp | None = None
    availability_basis: Literal["observed", "reconstructed_research"] = "reconstructed_research"

    @model_validator(mode="after")
    def validate_rejection(self) -> Self:
        if self.published_at > self.retrieved_at:
            raise ValueError("publication cannot follow retrieval")
        if self.availability_basis == "observed" and self.first_observed_at is None:
            raise ValueError("observed rejection needs actual observation clock")
        if self.first_observed_at is not None and not self.published_at <= self.first_observed_at <= self.retrieved_at:
            raise ValueError("rejection observation clock unordered")
        if self.period_scope == "exact" and (self.end is None or (self.concept not in {"assets", "cash", "debt"} and self.start is None)):
            raise ValueError("exact rejection requires known reporting period")
        return self

    def available_at(self) -> datetime:
        return self.first_observed_at if self.availability_basis == "observed" and self.first_observed_at else self.published_at

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))


class FundamentalSnapshot(HoldingContract):
    security_id: Identifier
    cik: Identifier
    decision_time: Timestamp
    scope: Literal["nonfinancial"] = "nonfinancial"
    research_only: Literal[True] = True
    training_allowed: Literal[False] = False
    promotion_allowed: Literal[False] = False
    feature_contract_sha256: Sha256 = FEATURE_CONTRACT_SHA256
    feature_names: tuple[str, ...] = FEATURE_NAMES
    metrics: dict[str, float | None]
    missing_reasons: dict[str, str]
    consumed_fact_sha256: tuple[str, ...]
    consumed_rejection_sha256: tuple[str, ...] = ()
    available_at: Timestamp | None


def _quarter_values(facts: list[FilingFact]) -> dict[tuple[date, date], tuple[float, tuple[FilingFact, ...]]]:
    result: dict[tuple[date, date], tuple[float, tuple[FilingFact, ...]]] = {}
    ambiguous: set[tuple[date, date]] = set()

    def insert(key: tuple[date, date], value: float, evidence: tuple[FilingFact, ...]) -> None:
        if key in result and result[key][0] != value:
            ambiguous.add(key)
        elif key not in result:
            result[key] = (value, evidence)

    for fact in facts:
        assert fact.start is not None
        days = (fact.end - fact.start).days + 1
        if 70 <= days <= 105:
            insert((fact.start, fact.end), fact.value, (fact,))
        elif 150 <= days <= 380:
            # Only subtract a prefix in the SAME filing. Cross-version subtraction
            # can combine an amended YTD total with an unrevised earlier quarter.
            prefixes = [
                p
                for p in facts
                if p.start == fact.start and p.end < fact.end and p.accession == fact.accession and 70 <= (fact.end - p.end).days <= 105
            ]
            if len(prefixes) == 1:
                prefix = prefixes[0]
                value = fact.value - prefix.value
                if fact.concept != "capex" or value >= 0:
                    insert((prefix.end + timedelta(days=1), fact.end), value, (prefix, fact))
    return {key: value for key, value in result.items() if key not in ambiguous}


def build_fundamental_snapshot(
    facts: Sequence[FilingFact],
    *,
    security_id: str,
    cik: str,
    decision_time: datetime,
    scope: Literal["nonfinancial"] = "nonfinancial",
    rejections: Sequence[FilingFactRejection] = (),
) -> FundamentalSnapshot:
    """Resolve period versions, then constructed annual/YTD or contiguous quarters; no imputation."""
    if decision_time.tzinfo is None or decision_time.utcoffset() is None:
        raise ValueError("decision clock must be timezone aware")
    if scope != "nonfinancial":
        raise ValueError("fundamental definitions support only nonfinancial issuers")
    selected: dict[tuple[str, date | None, date], FilingFact] = {}
    admissible: list[FilingFact] = []
    for original in facts:
        fact = FilingFact.model_validate_json(original.model_dump_json())
        if fact.security_id != security_id or fact.cik != cik or fact.available_at() > decision_time:
            continue
        admissible.append(fact)
        key = (fact.concept, fact.start, fact.end)
        old = selected.get(key)
        if old is None or fact.published_at > old.published_at:
            selected[key] = fact
        elif fact.published_at == old.published_at and fact.sha256() != old.sha256():
            raise ValueError("ambiguous facts at the same period/version clock")
    visible_rejections = [FilingFactRejection.model_validate_json(r.model_dump_json()) for r in rejections]
    visible_rejections = [
        r for r in visible_rejections if r.security_id == security_id and r.cik == cik and r.available_at() <= decision_time
    ]
    unresolved: dict[str, FilingFactRejection] = {}
    blocked_concepts: set[str] = set()
    for rejection in visible_rejections:
        if rejection.period_scope == "concept":
            # Unknown identity cannot be cleared by an unrelated later period.
            blocked_concepts.add(rejection.concept)
            unresolved[rejection.sha256()] = rejection
            continue
        assert rejection.end is not None
        rejection_key = (rejection.concept, rejection.start, rejection.end)
        latest = selected.get(rejection_key)
        if latest is None or latest.published_at <= rejection.published_at:
            unresolved[rejection.sha256()] = rejection

    def permitted(fact: FilingFact) -> bool:
        return fact.concept not in blocked_concepts and not any(
            r.concept == fact.concept and r.start == fact.start and r.end == fact.end and r.published_at >= fact.published_at
            for r in unresolved.values()
            if r.period_scope == "exact"
        )

    selected = {key: fact for key, fact in selected.items() if permitted(fact)}
    admissible = [fact for fact in admissible if permitted(fact)]
    revenue_ends = [f.end for f in selected.values() if f.concept == "revenue"]
    revenue_ends.extend(r.end for r in unresolved.values() if r.concept == "revenue" and r.end is not None)
    endpoint = max(revenue_ends, default=None)
    consumed_rejections: dict[str, FilingFactRejection] = {}

    used: dict[str, FilingFact] = {}

    def prior_year(day: date) -> date | None:
        try:
            return day.replace(year=day.year - 1)
        except ValueError:
            return None  # No invented anniversary for February 29.

    @cache
    def aggregate(concept: str, end: date | None) -> tuple[date, float, tuple[FilingFact, ...]] | None:
        if end is None:
            return None
        concept_issues = [r for r in unresolved.values() if r.concept == concept and r.period_scope == "concept"]
        if concept_issues:
            for rejection in concept_issues:
                consumed_rejections[rejection.sha256()] = rejection
            return None
        exact_issues = [r for r in unresolved.values() if r.concept == concept and r.period_scope == "exact"]
        blocked_path_issues: dict[str, FilingFactRejection] = {}

        def rejected_interval(start: date | None, stop: date | None) -> bool:
            matches = [r for r in exact_issues if r.start == start and r.end == stop]
            for rejection in matches:
                blocked_path_issues[rejection.sha256()] = rejection
            return bool(matches)

        # Earlier filing records are usable only when the latest as-of value for
        # that EXACT period agrees. Changed values cannot resurrect an old bridge.
        rows = [f for f in admissible if f.concept == concept and f.value == selected[(f.concept, f.start, f.end)].value]
        rows.sort(key=lambda f: (f.published_at, f.accession, f.sha256()))

        def bound_evidence(*arithmetic: FilingFact) -> tuple[FilingFact, ...]:
            bound = {f.sha256(): f for f in arithmetic}
            for fact in arithmetic:
                latest = selected[(fact.concept, fact.start, fact.end)]
                bound[latest.sha256()] = latest
            return tuple(bound[key] for key in sorted(bound))

        candidates: list[tuple[date, float, tuple[FilingFact, ...]]] = []
        quarters = {
            key: value
            for key, value in _quarter_values([f for f in selected.values() if f.concept == concept]).items()
            if not rejected_interval(*key)
        }
        total, current = 0.0, end
        chain: list[FilingFact] = []
        for _ in range(4):
            matches = [(start, value, evidence) for (start, stop), (value, evidence) in quarters.items() if stop == current]
            if len(matches) != 1:
                for rejection in exact_issues:
                    if rejection.end == current:
                        blocked_path_issues[rejection.sha256()] = rejection
                break
            start, value, evidence = matches[0]
            total += value
            chain.extend(evidence)
            current = start - timedelta(days=1)
        else:
            candidates.append((current + timedelta(days=1), total, tuple(chain)))
        annuals = [f for f in rows if f.start is not None and f.form in {"10-K", "10-K/A"} and 350 <= (f.end - f.start).days + 1 <= 380]
        for annual in annuals:
            assert annual.start is not None
            if annual.end == end:
                candidates.append((annual.start, annual.value, bound_evidence(annual)))
            # Require comparative YTDs in the same filing, never unrelated versions.
            for ytd in rows:
                if (
                    ytd.start is None
                    or ytd.end != end
                    or ytd.start != annual.end + timedelta(days=1)
                    or not 70 <= (ytd.end - ytd.start).days + 1 <= 300
                    or annual.form.endswith("/A")
                    or ytd.form.endswith("/A")
                ):
                    continue
                prefixes = [
                    f
                    for f in rows
                    if f.start == annual.start
                    and f.end == prior_year(ytd.end)
                    and f.start == prior_year(ytd.start)
                    and f.accession == ytd.accession
                ]
                if len(prefixes) != 1:
                    rejected_interval(annual.start, prior_year(ytd.end))
                    continue
                prefix = prefixes[0]
                value = annual.value + ytd.value - prefix.value
                if concept == "capex" and value < 0:
                    continue
                candidates.append((prefix.end + timedelta(days=1), value, bound_evidence(annual, ytd, prefix)))
        # A rejected reported aggregate cannot revive through an older quarter
        # sum or another arithmetic path producing that same exact interval.
        candidates = [candidate for candidate in candidates if not rejected_interval(candidate[0], end)]
        # A missing annual bridge constituent can itself be a rejected version.
        for rejection in exact_issues:
            if rejection.start is None or rejection.end is None:
                continue
            for ytd in rows:
                if (
                    ytd.end == end
                    and ytd.start == rejection.end + timedelta(days=1)
                    and any(
                        prefix.accession == ytd.accession and prefix.start == rejection.start and prefix.end == prior_year(ytd.end)
                        for prefix in rows
                    )
                ):
                    blocked_path_issues[rejection.sha256()] = rejection
        # Multiple arithmetic paths must agree in both interval and amount.
        if not candidates or len({(start, value) for start, value, _ in candidates}) != 1:
            consumed_rejections.update(blocked_path_issues)
            return None
        return candidates[0]

    def ttm(concept: str, end: date | None) -> float | None:
        result, revenue_result = aggregate(concept, end), aggregate("revenue", end)
        if result is None or revenue_result is None or result[0] != revenue_result[0]:
            return None
        for fact in result[2]:
            used[fact.sha256()] = fact
        return result[1]

    previous_end = prior_year(endpoint) if endpoint else None
    rev, old_rev = ttm("revenue", endpoint), ttm("revenue", previous_end)
    op, old_op = ttm("operating_income", endpoint), ttm("operating_income", previous_end)
    ocf, capex, income = (ttm(c, endpoint) for c in ("operating_cash_flow", "capex", "net_income"))

    def ratio(a: float | None, b: float | None) -> float | None:
        return a / b if a is not None and b is not None and b > 0 else None

    margin, old_margin = ratio(op, rev), ratio(old_op, old_rev)
    growth = ratio(rev, old_rev)
    metrics = dict(
        zip(
            FEATURE_NAMES,
            (
                growth - 1 if growth is not None else None,
                margin,
                margin - old_margin if margin is not None and old_margin is not None else None,
                ratio(ocf - capex if ocf is not None and capex is not None else None, rev),
                ratio(capex, rev),
                ratio(ocf, income),
            ),
            strict=True,
        )
    )
    for value in metrics.values():
        if value is None:
            # Include rejection authority only when it limits a feature calculation.
            for rejection in unresolved.values():
                if rejection.concept == "revenue" and rejection.period_scope == "concept":
                    consumed_rejections[rejection.sha256()] = rejection
    dependencies = {
        "revenue_growth_ttm": {"revenue"},
        "operating_margin_ttm": {"revenue", "operating_income"},
        "operating_margin_change": {"revenue", "operating_income"},
        "fcf_margin_ttm": {"revenue", "operating_cash_flow", "capex"},
        "capex_to_revenue_ttm": {"revenue", "capex"},
        "cash_conversion_ttm": {"revenue", "operating_cash_flow", "net_income"},
    }
    missing_reasons = {}
    for name, value in metrics.items():
        if value is None:
            rejected = sorted({f"{r.concept}:{r.reason}" for r in consumed_rejections.values() if r.concept in dependencies[name]})
            missing_reasons[name] = (
                "rejected_fact_version:" + ",".join(rejected)
                if rejected
                else "missing_or_ambiguous_contiguous_periods_or_nonpositive_denominator"
            )
    return FundamentalSnapshot(
        security_id=security_id,
        cik=cik,
        decision_time=decision_time,
        metrics=metrics,
        missing_reasons=missing_reasons,
        consumed_fact_sha256=tuple(sorted(used)),
        consumed_rejection_sha256=tuple(sorted(consumed_rejections)),
        available_at=max(
            [f.available_at() for f in used.values()] + [r.available_at() for r in consumed_rejections.values()], default=None
        ),
    )


def build_fundamental_batch(
    facts: Sequence[FilingFact],
    decisions: Sequence[tuple[str, str, datetime]],
    *,
    rejections: Sequence[FilingFactRejection] = (),
) -> tuple[FundamentalSnapshot, ...]:
    return tuple(build_fundamental_snapshot(facts, security_id=s, cik=c, decision_time=t, rejections=rejections) for s, c, t in decisions)
