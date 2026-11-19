"""Point-in-time SEC company facts for research lane step 2a.

Rule 2: an issuer's companyfacts JSON is parsed and every fact filed after the
development end is dropped in the same statement, before any other processing.
Rule 3: a value is usable for the decision at the close of session t only when its
filing date is an earlier calendar day than t.
Rule 4: for each period the value from the earliest filing is kept; a later
restatement never replaces it. Among several accepted tags reported in that earliest
filing, the first tag in the item's ordered list is used.

Per-session series are built by evaluating the issuer's state on every filing date
("change date") with only the periods usable on or before it, and applying that state
to the sessions after it. Values never depend on a filing made on or after the session.
"""
from __future__ import annotations

import json
import math
from bisect import bisect_left
from collections import Counter, defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt

from market_predictor.research_lane.panel import SealedPeriodError

COMPANY_FACTS_DIRECTORY: Final = Path("data/research/investment_financial_coverage_20261107")
COMPANY_FACTS_MANIFEST: Final = "final/_manifest.json"
COMPANY_FACTS_SOURCES: Final = "sources"
COMPANY_FACTS_FILE: Final = "company_facts.json"

# Ordered accepted tags per item (first tag in the earliest filing wins). USD units only.
FLOW_TAGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "revenue": ("us-gaap:Revenues", "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax",
                "us-gaap:RevenueFromContractWithCustomerIncludingAssessedTax", "us-gaap:SalesRevenueNet"),
    "net_income": ("us-gaap:NetIncomeLoss",),
    "operating_income": ("us-gaap:OperatingIncomeLoss",),
    "operating_cash_flow": ("us-gaap:NetCashProvidedByUsedInOperatingActivities",),
    "capital_expenditure": ("us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",),
    "gross_profit": ("us-gaap:GrossProfit",),
    "cost_of_revenue": ("us-gaap:CostOfRevenue", "us-gaap:CostOfGoodsAndServicesSold"),
})
INSTANT_TAGS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "total_assets": ("us-gaap:Assets",),
    "stockholders_equity": ("us-gaap:StockholdersEquity",),
})
SHARES_OUTSTANDING_TAG: Final = "dei:EntityCommonStockSharesOutstanding"
DILUTED_SHARES_TAG: Final = "us-gaap:WeightedAverageNumberOfDilutedSharesOutstanding"
SHARE_TAGS: Final = (SHARES_OUTSTANDING_TAG, DILUTED_SHARES_TAG)
NEEDED_UNITS: Final[Mapping[str, str]] = MappingProxyType({
    **{tag: "USD" for tags in FLOW_TAGS.values() for tag in tags},
    **{tag: "USD" for tags in INSTANT_TAGS.values() for tag in tags},
    SHARES_OUTSTANDING_TAG: "shares", DILUTED_SHARES_TAG: "shares",
})

# Duration bands in days (end - start). Half-year and nine-month bands are this module's choice.
QUARTER_DAYS: Final = (80, 100)
HALF_YEAR_DAYS: Final = (170, 200)
NINE_MONTH_DAYS: Final = (260, 290)
FULL_YEAR_DAYS: Final = (350, 380)
DURATION_BANDS: Final = (QUARTER_DAYS, HALF_YEAR_DAYS, NINE_MONTH_DAYS, FULL_YEAR_DAYS)
TRAILING_YEAR_DAYS: Final = (350, 380)
YEAR_EARLIER_DAYS: Final = (350, 380)
SHARE_PRIOR_FILING_DAYS: Final = (330, 400)
SURPRISE_QUARTERS: Final = 8
SURPRISE_MINIMUM_DIFFERENCES: Final = 6
SURPRISE_WINDOW_DAYS: Final = 730
STALE_AFTER_DAYS: Final = 200

# Missing-value reason codes shared by every step 2a input (0 = value present).
PRESENT: Final = 0
NO_COMPANY_FACTS: Final = 1
NOT_YET_REPORTED: Final = 2
STALE: Final = 3
INCOMPLETE_HISTORY: Final = 4
NONPOSITIVE_DENOMINATOR: Final = 5
ZERO_DISPERSION: Final = 6
NO_UNADJUSTED_CLOSE: Final = 7
SHARE_COUNT_UNAVAILABLE: Final = 8
SPLIT_AFTER_SHARE_COUNT: Final = 9
SPLIT_CHECK_NOT_POSSIBLE: Final = 10
SHARE_COUNT_AMBIGUOUS: Final = 11
SHARE_CHANGE_OVER_LIMIT: Final = 12
ISSUER_SHARED_BY_SECURITIES: Final = 13
NO_USABLE_RELEASE: Final = 14
RELEASE_EXPIRED: Final = 15
REACTION_PRICES_MISSING: Final = 16
MISSING_REASONS: Final[Mapping[int, str]] = MappingProxyType({
    NO_COMPANY_FACTS: "no_company_facts_for_security",
    NOT_YET_REPORTED: "no_usable_reported_value_yet",
    STALE: "latest_period_end_more_than_200_days_before_decision",
    INCOMPLETE_HISTORY: "incomplete_quarter_history",
    NONPOSITIVE_DENOMINATOR: "denominator_not_positive",
    ZERO_DISPERSION: "zero_standard_deviation",
    NO_UNADJUSTED_CLOSE: "no_unadjusted_close_on_decision_session",
    SHARE_COUNT_UNAVAILABLE: "no_usable_share_count",
    SPLIT_AFTER_SHARE_COUNT: "split_after_share_count_date",
    SPLIT_CHECK_NOT_POSSIBLE: "split_check_not_possible",
    SHARE_COUNT_AMBIGUOUS: "share_count_filing_reports_several_values",
    SHARE_CHANGE_OVER_LIMIT: "share_count_change_over_50_percent",
    ISSUER_SHARED_BY_SECURITIES: "issuer_shared_by_several_panel_securities",
    NO_USABLE_RELEASE: "no_usable_earnings_release",
    RELEASE_EXPIRED: "latest_release_more_than_63_sessions_old",
    REACTION_PRICES_MISSING: "reaction_window_close_missing",
})

FloatArray = npt.NDArray[np.float64]
ReasonArray = npt.NDArray[np.int8]
DayArray = npt.NDArray[np.int64]  # days since 1970-01-01; -1 = none
NO_DAY: Final = -1


@dataclass(frozen=True)
class Fact:
    tag: str
    start: date | None
    end: date
    value: float
    filed: date
    accession: str


IssuerFacts = dict[str, list[Fact]]


@dataclass(frozen=True)
class Period:
    start: date | None
    end: date
    value: float
    usable: date  # filing date (later of the two inputs for a derived quarter)
    tag: str
    accessions: frozenset[str]
    derived: bool = False
    conflict: bool = False  # the earliest filing reports several values for this period and tag


def _date(text: Any) -> date | None:
    if not isinstance(text, str) or len(text) != 10:
        return None
    try:
        return date.fromisoformat(text)
    except ValueError:
        return None


def keep_filed_by(document: Mapping[str, Any], last_filed: str) -> tuple[IssuerFacts, Counter[str]]:
    """Rule 2 first: keep needed-tag rows filed on or before ``last_filed``; everything else is discarded."""
    counts: Counter[str] = Counter()
    taxonomies = document.get("facts", {}) if isinstance(document, Mapping) else {}
    result: IssuerFacts = {}
    for key, unit in NEEDED_UNITS.items():
        taxonomy, tag = key.split(":", 1)
        rows = taxonomies.get(taxonomy, {}).get(tag, {}).get("units", {}).get(unit, [])
        kept = [row for row in rows if isinstance(row, Mapping) and isinstance(row.get("filed"), str)
                and row["filed"] <= last_filed]
        facts: list[Fact] = []
        for row in kept:
            end, filed, start = _date(row.get("end")), _date(row.get("filed")), _date(row.get("start"))
            value = row.get("val")
            if end is None or filed is None or not isinstance(value, int | float) or not math.isfinite(float(value)) \
                    or ("start" in row and start is None):
                counts["malformed_rows_skipped"] += 1
                continue
            facts.append(Fact(tag=key, start=start, end=end, value=float(value), filed=filed,
                              accession=str(row.get("accn", ""))))
        if facts:
            result[key] = facts
            counts[f"rows:{key}"] += len(facts)
    return result, counts


def read_company_facts(path: Path, development_end: date) -> tuple[IssuerFacts, Counter[str]]:
    """Parse one issuer and apply rule 2 in the same statement; the full JSON is released on return."""
    facts, counts = keep_filed_by(json.loads(path.read_bytes()), development_end.isoformat())
    if any(fact.filed > development_end for items in facts.values() for fact in items):
        raise SealedPeriodError(f"company facts filed after {development_end} reached memory from {path}")
    return facts, counts


def first_reported(facts: Iterable[Fact], tag_order: Sequence[str]) -> dict[tuple[date | None, date], Period]:
    """Rule 4: one value per (start, end) from the earliest filing; tag priority breaks same-filing ties."""
    groups: dict[tuple[date | None, date], list[Fact]] = defaultdict(list)
    for fact in facts:
        if fact.tag in tag_order:
            groups[(fact.start, fact.end)].append(fact)
    result: dict[tuple[date | None, date], Period] = {}
    for (start, end), items in groups.items():
        first = min(fact.filed for fact in items)
        earliest = [fact for fact in items if fact.filed == first]
        tag = min((fact.tag for fact in earliest), key=tag_order.index)
        chosen = [fact for fact in earliest if fact.tag == tag]
        values = {fact.value for fact in chosen}
        result[(start, end)] = Period(start=start, end=end, value=chosen[0].value if len(values) == 1 else math.nan,
                                      usable=first, tag=tag.split(":", 1)[1],
                                      accessions=frozenset(fact.accession for fact in chosen), conflict=len(values) > 1)
    return result


def _days(period: Period) -> int:
    assert period.start is not None
    return (period.end - period.start).days


def _within(value: int, band: tuple[int, int]) -> bool:
    return band[0] <= value <= band[1]


def _band(period: Period) -> int | None:
    days = _days(period)
    for index, band in enumerate(DURATION_BANDS):
        if _within(days, band):
            return index
    return None


def single_quarters(durations: Mapping[tuple[date | None, date], Period]) -> list[Period]:
    """Single-quarter values per quarter end, direct (80-100 days) or derived from year-to-date facts.

    A derived quarter is long minus short where both share a start date and the short one is the
    next-shorter band (6m-3m, 9m-6m, full year-9m); it is usable from the later of the two filing
    dates. When several candidates exist for one quarter end, the one usable first is kept
    (direct before derived on the same date), so a later filing never replaces a known quarter.
    """
    valid = [period for period in durations.values() if period.start is not None and not period.conflict]
    by_start: dict[date, list[tuple[int, Period]]] = defaultdict(list)
    candidates: dict[date, list[Period]] = defaultdict(list)
    for period in valid:
        band = _band(period)
        if band is None:
            continue
        assert period.start is not None
        by_start[period.start].append((band, period))
        if band == 0:
            candidates[period.end].append(period)
    for items in by_start.values():
        for band, long in items:
            if band == 0:
                continue
            for short_band, short in items:
                if short_band != band - 1 or short.end >= long.end:
                    continue
                start = short.end + timedelta(days=1)
                if not _within((long.end - start).days, QUARTER_DAYS):
                    continue
                tag = long.tag if long.tag == short.tag else f"{long.tag}-{short.tag}"
                candidates[long.end].append(Period(
                    start=start, end=long.end, value=long.value - short.value, usable=max(long.usable, short.usable),
                    tag=tag, accessions=long.accessions | short.accessions, derived=True))
    chosen = [min(items, key=lambda p: (p.usable, p.derived, p.start or date.min)) for items in candidates.values()]
    return sorted(chosen, key=lambda p: p.end)


def gross_profit_quarters(gross: Sequence[Period], revenue: Sequence[Period], cost: Sequence[Period]) -> list[Period]:
    """GrossProfit quarters, else revenue minus cost of revenue for the same quarter end (first usable wins)."""
    candidates: dict[date, list[tuple[int, Period]]] = defaultdict(list)
    for period in gross:
        candidates[period.end].append((0, period))
    costs = {period.end: period for period in cost}
    for period in revenue:
        other = costs.get(period.end)
        if other is None:
            continue
        candidates[period.end].append((1, Period(
            start=period.start, end=period.end, value=period.value - other.value,
            usable=max(period.usable, other.usable), tag=f"{period.tag}-minus-{other.tag}",
            accessions=period.accessions | other.accessions, derived=period.derived or other.derived)))
    chosen = [min(items, key=lambda item: (item[1].usable, item[0]))[1] for items in candidates.values()]
    return sorted(chosen, key=lambda p: p.end)


def instants(periods: Mapping[tuple[date | None, date], Period]) -> list[Period]:
    return sorted((p for p in periods.values() if p.start is None and not p.conflict), key=lambda p: p.end)


@dataclass(frozen=True)
class State:
    value: float
    reason: int
    period_end: date | None
    used: tuple[Period, ...] = ()


def _missing(reason: int, period_end: date | None = None) -> State:
    return State(math.nan, reason, period_end)


def trailing_year(quarters: Sequence[Period]) -> State:
    """Sum of the latest four usable quarters when contiguous (ends 80-100 days apart, 350-380 days covered)."""
    if not quarters:
        return _missing(NOT_YET_REPORTED)
    latest = quarters[-1]
    last = list(quarters[-4:])
    if len(last) < 4:
        return _missing(INCOMPLETE_HISTORY, latest.end)
    gaps_ok = all(_within((later.end - earlier.end).days, QUARTER_DAYS) for earlier, later in zip(last, last[1:], strict=False))
    first_start = last[0].start
    if not gaps_ok or first_start is None or not _within((latest.end - first_start).days, TRAILING_YEAR_DAYS):
        return _missing(INCOMPLETE_HISTORY, latest.end)
    return State(float(sum(q.value for q in last)), PRESENT, latest.end, tuple(last))


def _year_earlier(periods: Sequence[Period], end: date) -> Period | None:
    matches = [p for p in periods if _within((end - p.end).days, YEAR_EARLIER_DAYS)]
    return min(matches, key=lambda p: (abs((end - p.end).days - 365), -p.end.toordinal())) if matches else None


def standardized_change(quarters: Sequence[Period]) -> State:
    """(X_q - X_{q-4}) / sample standard deviation of that difference over the latest 8 quarters (>= 6 needed)."""
    if not quarters:
        return _missing(NOT_YET_REPORTED)
    latest = quarters[-1]
    window = [q for q in quarters[-SURPRISE_QUARTERS:] if (latest.end - q.end).days < SURPRISE_WINDOW_DAYS]
    differences: list[float] = []
    used: list[Period] = []
    current: float | None = None
    for quarter in window:
        prior = _year_earlier(quarters, quarter.end)
        if prior is None:
            continue
        difference = quarter.value - prior.value
        differences.append(difference)
        used.extend((quarter, prior))
        if quarter is latest:
            current = difference
    if current is None or len(differences) < SURPRISE_MINIMUM_DIFFERENCES:
        return _missing(INCOMPLETE_HISTORY, latest.end)
    spread = float(np.std(np.asarray(differences, dtype=np.float64), ddof=1))
    if not spread > 0:
        return _missing(ZERO_DISPERSION, latest.end)
    return State(current / spread, PRESENT, latest.end, tuple(used))


def latest_instant(values: Sequence[Period]) -> State:
    if not values:
        return _missing(NOT_YET_REPORTED)
    latest = values[-1]
    return State(latest.value, PRESENT, latest.end, (latest,))


def instant_growth(values: Sequence[Period]) -> State:
    """Latest instant / the instant 350-380 days earlier - 1."""
    if not values:
        return _missing(NOT_YET_REPORTED)
    latest = values[-1]
    prior = _year_earlier(values, latest.end)
    if prior is None:
        return _missing(INCOMPLETE_HISTORY, latest.end)
    if not prior.value > 0:
        return _missing(NONPOSITIVE_DENOMINATOR, latest.end)
    return State(latest.value / prior.value - 1.0, PRESENT, latest.end, (latest, prior))


@dataclass(frozen=True)
class ShareState:
    count: State
    issuance: State
    tag: str | None


def _share_prior(periods: Sequence[Period], current: Period) -> Period | None:
    low, high = SHARE_PRIOR_FILING_DAYS
    matches = [p for p in periods if not p.conflict and low <= (current.usable - p.usable).days <= high]
    return min(matches, key=lambda p: (abs((current.usable - p.usable).days - 365), -p.end.toordinal())) \
        if matches else None


def share_state(outstanding: Sequence[Period], diluted: Sequence[Period]) -> ShareState:
    """Most recent usable dei share count (by filing date); the diluted quarterly count only while no dei
    count is usable. The share date used by the split rule is the period end (as-of date)."""
    source, tag = (outstanding, "EntityCommonStockSharesOutstanding") if outstanding else \
        (diluted, "WeightedAverageNumberOfDilutedSharesOutstanding")
    if not source:
        return ShareState(_missing(NOT_YET_REPORTED), _missing(NOT_YET_REPORTED), None)
    current = max(source, key=lambda p: (p.usable, p.end))
    if current.conflict:
        return ShareState(_missing(SHARE_COUNT_AMBIGUOUS, current.end), _missing(SHARE_COUNT_AMBIGUOUS, current.end), tag)
    if not current.value > 0:
        return ShareState(_missing(NONPOSITIVE_DENOMINATOR, current.end),
                          _missing(NONPOSITIVE_DENOMINATOR, current.end), tag)
    count = State(current.value, PRESENT, current.end, (current,))
    prior = _share_prior(source, current)
    if prior is None:
        return ShareState(count, _missing(INCOMPLETE_HISTORY, current.end), tag)
    if not prior.value > 0:
        return ShareState(count, _missing(NONPOSITIVE_DENOMINATOR, current.end), tag)
    change = current.value / prior.value - 1.0
    if abs(change) > 0.5:
        return ShareState(count, _missing(SHARE_CHANGE_OVER_LIMIT, current.end), tag)
    return ShareState(count, State(change, PRESENT, current.end, (current, prior)), tag)


@dataclass(frozen=True)
class IssuerPeriods:
    quarters: Mapping[str, list[Period]]
    assets: list[Period]
    equity: list[Period]
    outstanding: list[Period]
    diluted: list[Period]
    conflicts: Counter[str]


def issuer_periods(facts: IssuerFacts) -> IssuerPeriods:
    def facts_for(tags: Sequence[str]) -> list[Fact]:
        return [fact for tag in tags for fact in facts.get(tag, [])]

    conflicts: Counter[str] = Counter()
    durations: dict[str, dict[tuple[date | None, date], Period]] = {}
    for item, tags in FLOW_TAGS.items():
        durations[item] = first_reported(facts_for(tags), tags)
        conflicts[item] = sum(1 for p in durations[item].values() if p.conflict)
    quarters = {item: single_quarters(periods) for item, periods in durations.items()}
    quarters["gross_profit"] = gross_profit_quarters(quarters["gross_profit"], quarters["revenue"],
                                                     quarters.pop("cost_of_revenue"))
    balance = {}
    for item, tags in INSTANT_TAGS.items():
        periods = first_reported(facts_for(tags), tags)
        conflicts[item] = sum(1 for p in periods.values() if p.conflict)
        balance[item] = instants(periods)
    outstanding_periods = first_reported(facts_for((SHARES_OUTSTANDING_TAG,)), (SHARES_OUTSTANDING_TAG,))
    outstanding = sorted((p for p in outstanding_periods.values() if p.start is None), key=lambda p: (p.usable, p.end))
    conflicts["shares_outstanding"] = sum(1 for p in outstanding if p.conflict)
    diluted_periods = first_reported(facts_for((DILUTED_SHARES_TAG,)), (DILUTED_SHARES_TAG,))
    diluted = sorted((p for p in diluted_periods.values() if p.start is not None and not p.conflict
                      and _within(_days(p), QUARTER_DAYS)), key=lambda p: (p.usable, p.end))
    return IssuerPeriods(quarters=quarters, assets=balance["total_assets"], equity=balance["stockholders_equity"],
                         outstanding=outstanding, diluted=diluted, conflicts=+conflicts)


TTM_COMPONENTS: Final = ("revenue", "net_income", "operating_income", "operating_cash_flow", "capital_expenditure",
                         "gross_profit")
COMPONENTS: Final = (*(f"{item}_ttm" for item in TTM_COMPONENTS), "total_assets", "stockholders_equity",
                     "asset_growth", "net_income_change", "revenue_change", "shares", "share_issuance")


@dataclass
class IssuerSeries:
    """Per-session values and reason codes (stale values already removed) for one issuer."""

    values: dict[str, FloatArray]
    reasons: dict[str, ReasonArray]
    share_as_of: DayArray
    share_tag: npt.NDArray[np.int8]  # 0 none, 1 dei outstanding, 2 diluted weighted average
    tag_usage: Counter[str] = field(default_factory=Counter)
    accessions: set[str] = field(default_factory=set)
    conflicts: Counter[str] = field(default_factory=Counter)


def _usable(periods: Sequence[Period], day: date) -> list[Period]:
    return [p for p in periods if p.usable <= day]


def _states_on(periods: IssuerPeriods, day: date) -> tuple[dict[str, State], ShareState]:
    states: dict[str, State] = {}
    for item in TTM_COMPONENTS:
        states[f"{item}_ttm"] = trailing_year(_usable(periods.quarters[item], day))
    assets = _usable(periods.assets, day)
    states["total_assets"] = latest_instant(assets)
    states["stockholders_equity"] = latest_instant(_usable(periods.equity, day))
    states["asset_growth"] = instant_growth(assets)
    states["net_income_change"] = standardized_change(_usable(periods.quarters["net_income"], day))
    states["revenue_change"] = standardized_change(_usable(periods.quarters["revenue"], day))
    shares = share_state(_usable(periods.outstanding, day), _usable(periods.diluted, day))
    states["shares"] = shares.count
    states["share_issuance"] = shares.issuance
    return states, shares


EPOCH: Final = date(1970, 1, 1)


def _day_number(day: date | None) -> int:
    return NO_DAY if day is None else (day - EPOCH).days


def day_numbers(days: npt.NDArray[np.datetime64]) -> DayArray:
    return np.asarray(days, dtype="datetime64[D]").astype(np.int64)


def evaluate_issuer(facts: IssuerFacts, sessions: npt.NDArray[np.datetime64]) -> IssuerSeries:
    """Per-session component series; a period filed on calendar day d is used from the first session after d."""
    periods = issuer_periods(facts)
    session_days = day_numbers(sessions)
    count = len(session_days)
    every = [*(p for qs in periods.quarters.values() for p in qs), *periods.assets, *periods.equity,
             *periods.outstanding, *periods.diluted]
    change_dates = sorted({p.usable for p in every})
    first_session = EPOCH + timedelta(days=int(session_days[0])) if count else date.max
    begin = max(0, bisect_left(change_dates, first_session) - 1)
    change_dates = change_dates[begin:]
    change_days = np.asarray([_day_number(d) for d in change_dates], dtype=np.int64)
    states = [_states_on(periods, day) for day in change_dates]
    index = np.searchsorted(change_days, session_days, side="left") - 1  # last change date strictly before session
    values: dict[str, FloatArray] = {}
    reasons: dict[str, ReasonArray] = {}
    tag_usage: Counter[str] = Counter()
    accessions: set[str] = set()
    used_keys: set[tuple[str, date, str]] = set()
    reached = set(int(i) for i in np.unique(index) if i >= 0)
    for name in COMPONENTS:
        value_by_state = np.asarray([s[0][name].value for s in states] + [math.nan], dtype=np.float64)
        reason_by_state = np.asarray([s[0][name].reason for s in states] + [NOT_YET_REPORTED], dtype=np.int8)
        end_by_state = np.asarray([_day_number(s[0][name].period_end) for s in states] + [NO_DAY], dtype=np.int64)
        value = value_by_state[index]  # index -1 selects the trailing "not yet reported" entry
        reason = reason_by_state[index].copy()
        period_end = end_by_state[index]
        stale = (period_end != NO_DAY) & (session_days - period_end > STALE_AFTER_DAYS)
        reason[stale] = STALE
        values[name] = np.where(reason == PRESENT, value, np.nan)
        reasons[name] = reason
        for position in reached:
            state = states[position][0][name]
            if state.reason != PRESENT:
                continue
            for period in state.used:
                key = (name, period.end, period.tag)
                if key not in used_keys:
                    used_keys.add(key)
                    tag_usage[f"{name}:{period.tag}"] += 1
                    accessions.update(period.accessions)
    share_as_of = np.asarray([_day_number(s[0]["shares"].period_end) for s in states] + [NO_DAY],
                             dtype=np.int64)[index]
    tag_code = {None: 0, "EntityCommonStockSharesOutstanding": 1, "WeightedAverageNumberOfDilutedSharesOutstanding": 2}
    share_tag = np.asarray([tag_code[s[1].tag] for s in states] + [0], dtype=np.int8)[index]
    share_tag = np.where(reasons["shares"] == PRESENT, share_tag, 0).astype(np.int8)
    return IssuerSeries(values=values, reasons=reasons, share_as_of=np.where(reasons["shares"] == PRESENT,
                                                                             share_as_of, NO_DAY),
                        share_tag=share_tag, tag_usage=tag_usage, accessions=accessions,
                        conflicts=periods.conflicts)
