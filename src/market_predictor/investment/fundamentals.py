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
FEATURE_NAMES = ("revenue_growth_ttm", "operating_margin_ttm", "operating_margin_change", "fcf_margin_ttm",
                 "capex_to_revenue_ttm", "cash_conversion_ttm")
FEATURE_DEFINITIONS = {
    "revenue_growth_ttm": "latest constructed as-of TTM / preceding year constructed as-of TTM - 1; positive denominator",
    "operating_margin_ttm": "operating income TTM / revenue TTM; positive denominator",
    "operating_margin_change": "latest operating margin TTM minus previous year operating margin TTM",
    "fcf_margin_ttm": "(operating cash flow TTM - positive cash capex TTM) / revenue TTM",
    "capex_to_revenue_ttm": "positive cash capex TTM / revenue TTM",
    "cash_conversion_ttm": "operating cash flow TTM / net income TTM; positive net income",
}
FEATURE_CONTRACT_SHA256 = json_sha256({
    "definitions": FEATURE_DEFINITIONS, "scope": "nonfinancial", "research_only": True,
    "ttm_policy": "four contiguous quarters OR annual direct OR annual + same-filing current YTD - prior YTD; exact intervals",
    "debt": "deferred until qualified total interest-bearing debt is available",
    "bridge_assumption": "unchanged accounting basis across annual and comparative YTD filings; research reconstruction",
    "version_policy": "earlier as-of filing record usable only when value exactly matches latest period value; bind both evidence versions",
})


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
            prefixes = [p for p in facts if p.start == fact.start and p.end < fact.end
                        and p.accession == fact.accession and 70 <= (fact.end - p.end).days <= 105]
            if len(prefixes) == 1:
                prefix = prefixes[0]
                value = fact.value - prefix.value
                if fact.concept != "capex" or value >= 0:
                    insert((prefix.end + timedelta(days=1), fact.end), value, (prefix, fact))
    return {key: value for key, value in result.items() if key not in ambiguous}


def build_fundamental_snapshot(
    facts: Sequence[FilingFact], *, security_id: str, cik: str, decision_time: datetime,
    scope: Literal["nonfinancial"] = "nonfinancial",
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
    endpoint = max((f.end for f in selected.values() if f.concept == "revenue"), default=None)
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
        # Earlier filing records are usable only when the latest as-of value for
        # that EXACT period agrees. Changed values cannot resurrect an old bridge.
        rows = [f for f in admissible if f.concept == concept
                and f.value == selected[(f.concept, f.start, f.end)].value]
        rows.sort(key=lambda f: (f.published_at, f.accession, f.sha256()))

        def bound_evidence(*arithmetic: FilingFact) -> tuple[FilingFact, ...]:
            bound = {f.sha256(): f for f in arithmetic}
            for fact in arithmetic:
                latest = selected[(fact.concept, fact.start, fact.end)]
                bound[latest.sha256()] = latest
            return tuple(bound[key] for key in sorted(bound))

        candidates: list[tuple[date, float, tuple[FilingFact, ...]]] = []
        quarters = _quarter_values([f for f in selected.values() if f.concept == concept])
        total, current = 0.0, end
        chain: list[FilingFact] = []
        for _ in range(4):
            matches = [(start, value, evidence) for (start, stop), (value, evidence) in quarters.items() if stop == current]
            if len(matches) != 1:
                break
            start, value, evidence = matches[0]
            total += value
            chain.extend(evidence)
            current = start - timedelta(days=1)
        else:
            candidates.append((current + timedelta(days=1), total, tuple(chain)))
        annuals = [f for f in rows if f.start is not None and f.form in {"10-K", "10-K/A"}
                   and 350 <= (f.end - f.start).days + 1 <= 380]
        for annual in annuals:
            assert annual.start is not None
            if annual.end == end:
                candidates.append((annual.start, annual.value, bound_evidence(annual)))
            # Require comparative YTDs in the same filing, never unrelated versions.
            for ytd in rows:
                if (ytd.start is None or ytd.end != end or ytd.start != annual.end + timedelta(days=1)
                        or not 70 <= (ytd.end - ytd.start).days + 1 <= 300
                        or annual.form.endswith("/A") or ytd.form.endswith("/A")):
                    continue
                prefixes = [f for f in rows if f.start == annual.start and f.end == prior_year(ytd.end)
                            and f.start == prior_year(ytd.start) and f.accession == ytd.accession]
                if len(prefixes) != 1:
                    continue
                prefix = prefixes[0]
                value = annual.value + ytd.value - prefix.value
                if concept == "capex" and value < 0:
                    continue
                candidates.append((prefix.end + timedelta(days=1), value, bound_evidence(annual, ytd, prefix)))
        # Multiple arithmetic paths must agree in both interval and amount.
        if not candidates or len({(start, value) for start, value, _ in candidates}) != 1:
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
    metrics = dict(zip(FEATURE_NAMES, (growth - 1 if growth is not None else None, margin,
        margin - old_margin if margin is not None and old_margin is not None else None,
        ratio(ocf - capex if ocf is not None and capex is not None else None, rev),
        ratio(capex, rev), ratio(ocf, income)), strict=True))
    return FundamentalSnapshot(security_id=security_id, cik=cik, decision_time=decision_time, metrics=metrics,
        missing_reasons={name: "missing_or_ambiguous_contiguous_periods_or_nonpositive_denominator"
            for name, value in metrics.items() if value is None},
        consumed_fact_sha256=tuple(sorted(used)), available_at=max((f.available_at() for f in used.values()), default=None))


def build_fundamental_batch(
    facts: Sequence[FilingFact], decisions: Sequence[tuple[str, str, datetime]],
) -> tuple[FundamentalSnapshot, ...]:
    return tuple(build_fundamental_snapshot(facts, security_id=s, cik=c, decision_time=t) for s, c, t in decisions)
