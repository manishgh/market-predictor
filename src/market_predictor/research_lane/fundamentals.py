"""Earnings-release reaction and financial-statement inputs for research lane step 2a.

Expected directions are frozen in ``EXPECTED_SIGNS`` before any result is seen: +1 means a
higher value is expected to precede a higher forward return. Every value at session row t
uses only prices through the close of t, filings made on calendar days before t, and
earnings releases whose reaction window ended by t. Market value uses unadjusted closes.
"""
from __future__ import annotations

from collections import Counter
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Any, Final

import numpy as np
import numpy.typing as npt
import pandas as pd

from market_predictor.research_lane.company_facts import (
    COMPONENTS,
    INCOMPLETE_HISTORY,
    ISSUER_SHARED_BY_SECURITIES,
    MISSING_REASONS,
    NO_COMPANY_FACTS,
    NO_DAY,
    NO_UNADJUSTED_CLOSE,
    NO_USABLE_RELEASE,
    NONPOSITIVE_DENOMINATOR,
    PRESENT,
    REACTION_PRICES_MISSING,
    RELEASE_EXPIRED,
    SPLIT_AFTER_SHARE_COUNT,
    SPLIT_CHECK_NOT_POSSIBLE,
    IssuerFacts,
    day_numbers,
    evaluate_issuer,
)
from market_predictor.research_lane.panel import ResearchPanel
from market_predictor.research_lane.predictors import centered_rank, minimum_present
from market_predictor.research_lane.sec_filings import event_rows

FloatArray = npt.NDArray[np.float64]
ReasonArray = npt.NDArray[np.int8]
BoolArray = npt.NDArray[np.bool_]

EXPECTED_SIGNS: Final[Mapping[str, int]] = MappingProxyType({
    "earnings_reaction_3d": 1,
    "earnings_surprise": 1,
    "revenue_surprise": 1,
    "earnings_yield": 1,
    "free_cash_flow_yield": 1,
    "sales_to_price": 1,
    "book_to_price": 1,
    "gross_profit_to_assets": 1,
    "operating_profitability": 1,
    "accruals": -1,
    "asset_growth": -1,
    "net_share_issuance": -1,
})
PREDICTOR_NAMES: Final = tuple(EXPECTED_SIGNS)
COMPOSITE_GROUPS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType({
    "composite_earnings": PREDICTOR_NAMES[0:3],
    "composite_valuation": PREDICTOR_NAMES[3:7],
    "composite_quality": PREDICTOR_NAMES[7:12],
    "composite_fundamental_all": PREDICTOR_NAMES,
})
REACTION_EXPIRY_SESSIONS: Final = 63
SPLIT_RETURN_GAP: Final = 0.15
SHARE_TAG_NAMES: Final = {1: "EntityCommonStockSharesOutstanding", 2: "WeightedAverageNumberOfDilutedSharesOutstanding"}


@dataclass(frozen=True)
class FundamentalInputs:
    panel: ResearchPanel
    unadjusted_close: FloatArray  # sessions x securities, NaN when absent
    releases: pd.DataFrame  # security_id, accession_number, accepted_at_utc, available_at_utc
    security_issuer: Mapping[str, str]  # panel security_id -> CIK whose company facts are read
    load_issuer: Callable[[str], IssuerFacts | None]


@dataclass
class FundamentalResult:
    values: dict[str, FloatArray]
    reasons: dict[str, ReasonArray]
    split_blanked: BoolArray
    share_tag: npt.NDArray[np.int8]
    reaction_source_event: npt.NDArray[np.int64]  # event id used per cell, -1 none
    report: dict[str, Any] = field(default_factory=dict)


def first_reason(*reasons: ReasonArray) -> ReasonArray:
    """Elementwise first non-zero reason code in argument order."""
    result = np.zeros(reasons[0].shape, dtype=np.int8)
    for reason in reasons:
        result = np.where(result == PRESENT, reason, result).astype(np.int8)
    return result


def _ratio(numerator: FloatArray, denominator: FloatArray, reason: ReasonArray) -> tuple[FloatArray, ReasonArray]:
    positive = np.isfinite(denominator) & (denominator > 0)
    reason = np.where((reason == PRESENT) & ~positive, NONPOSITIVE_DENOMINATOR, reason).astype(np.int8)
    with np.errstate(invalid="ignore", divide="ignore"):
        value = np.where(reason == PRESENT, numerator / np.where(positive, denominator, 1.0), np.nan)
    return value, reason


def earnings_reaction(panel: ResearchPanel, releases: pd.DataFrame) -> tuple[FloatArray, ReasonArray,
                                                                               npt.NDArray[np.int64], dict[str, Any]]:
    """Market-relative close[e+1]/close[e-2] reaction to the latest usable release (usable t >= e+1, until e+63)."""
    rows, columns = len(panel.sessions), len(panel.security_ids)
    values = np.full((rows, columns), np.nan, dtype=np.float64)
    reasons = np.full((rows, columns), NO_USABLE_RELEASE, dtype=np.int8)
    source = np.full((rows, columns), -1, dtype=np.int64)
    column_index = pd.Index(panel.security_ids)
    report: dict[str, Any] = {"release_rows": len(releases)}
    positions = column_index.get_indexer(releases["security_id"].astype(str))
    in_panel = positions >= 0
    report["release_rows_for_panel_securities"] = int(in_panel.sum())
    chosen = releases.loc[in_panel]
    events = event_rows(panel.sessions, chosen["available_at_utc"]) if len(chosen) else np.zeros(0, dtype=np.int64)
    by_acceptance = event_rows(panel.sessions, chosen["accepted_at_utc"]) if len(chosen) else events
    inside = events < rows
    report["release_rows_mapped_to_a_panel_session"] = int(inside.sum())
    report["release_rows_whose_session_differs_if_acceptance_time_were_used"] = int(
        (inside & (by_acceptance != events)).sum())
    frame = pd.DataFrame({"column": positions[in_panel][inside], "event": events[inside]}).drop_duplicates()
    report["distinct_release_events"] = len(frame)
    report["release_rows_merged_into_same_event_session"] = int(inside.sum()) - len(frame)
    close = panel.close.astype(np.float64)
    spy = panel.spy_close.astype(np.float64)
    session_rows = np.arange(rows)
    finite_events = 0
    event_ids = 0
    for column, group in frame.groupby("column"):
        event = np.sort(group["event"].to_numpy(dtype=np.int64))
        before, after = event - 2, event + 1
        valid = (before >= 0) & (after < rows)
        reaction = np.full(event.shape, np.nan, dtype=np.float64)
        b, a = before[valid], after[valid]
        with np.errstate(invalid="ignore", divide="ignore"):
            reaction[valid] = (close[a, column] / close[b, column] - 1.0) - (spy[a] / spy[b] - 1.0)
        finite_events += int(np.isfinite(reaction).sum())
        latest = np.searchsorted(after, session_rows, side="right") - 1
        has = latest >= 0
        picked = np.where(has, latest, 0)
        expired = has & (session_rows > event[picked] + REACTION_EXPIRY_SESSIONS)
        value = reaction[picked]
        reason = np.where(~has, NO_USABLE_RELEASE, np.where(expired, RELEASE_EXPIRED, np.where(
            np.isfinite(value), PRESENT, REACTION_PRICES_MISSING))).astype(np.int8)
        values[:, int(column)] = np.where(reason == PRESENT, value, np.nan)
        reasons[:, int(column)] = reason
        source[:, int(column)] = np.where(reason == PRESENT, event_ids + picked, -1)
        event_ids += len(event)
    report["release_events_with_reaction"] = finite_events
    return values, reasons, source, report


def split_events(unadjusted: FloatArray, adjusted: FloatArray) -> BoolArray:
    """True at session b when, versus the previous session a where both closes exist, the unadjusted
    close-to-close return differs from the adjusted one by more than 15% in absolute terms."""
    comparable = np.isfinite(unadjusted) & np.isfinite(adjusted) & (unadjusted > 0) & (adjusted > 0)
    events = np.zeros(unadjusted.shape, dtype=bool)
    for column in range(unadjusted.shape[1]):
        rows = np.flatnonzero(comparable[:, column])
        if rows.size < 2:
            continue
        raw = unadjusted[rows, column]
        adj = adjusted[rows, column]
        gap = np.abs((raw[1:] / raw[:-1]) - (adj[1:] / adj[:-1]))
        events[rows[1:][gap > SPLIT_RETURN_GAP], column] = True
    return events


def market_value(unadjusted: FloatArray, adjusted: FloatArray, shares: FloatArray, share_reason: ReasonArray,
                 share_as_of: npt.NDArray[np.int64], sessions: npt.NDArray[np.datetime64]
                 ) -> tuple[FloatArray, ReasonArray, BoolArray, BoolArray]:
    """Unadjusted close x share count; missing from a detected split after the share count's as-of date
    until a share count dated after that split is usable. Adjusted prices only locate splits."""
    rows, columns = unadjusted.shape
    session_days = day_numbers(sessions)
    comparable = np.isfinite(unadjusted) & np.isfinite(adjusted) & (unadjusted > 0) & (adjusted > 0)
    events = split_events(unadjusted, adjusted)
    cumulative = np.cumsum(events, axis=0)
    has_share = share_as_of != NO_DAY
    as_of_row = np.searchsorted(session_days, np.where(has_share, share_as_of, 0).ravel(), side="right").reshape(
        rows, columns) - 1  # last session on or before the as-of date
    column_ids = np.broadcast_to(np.arange(columns), (rows, columns))
    count_at_as_of = np.where(as_of_row >= 0, cumulative[np.maximum(as_of_row, 0), column_ids], 0)
    split_after = has_share & (cumulative - count_at_as_of > 0)
    first_comparable = np.where(comparable.any(axis=0), comparable.argmax(axis=0), rows)
    checkable = has_share & (as_of_row >= first_comparable[None, :])
    raw_ok = np.isfinite(unadjusted) & (unadjusted > 0)
    reason = share_reason.copy()
    reason = np.where((reason == PRESENT) & ~raw_ok, NO_UNADJUSTED_CLOSE, reason)
    reason = np.where((reason == PRESENT) & ~(comparable & checkable), SPLIT_CHECK_NOT_POSSIBLE, reason)
    blanked = (reason == PRESENT) & split_after
    reason = np.where(blanked, SPLIT_AFTER_SHARE_COUNT, reason).astype(np.int8)
    value = np.where(reason == PRESENT, unadjusted * shares, np.nan)
    return value, reason, blanked, events


def compute_fundamental_predictors(inputs: FundamentalInputs) -> FundamentalResult:
    """All twelve raw inputs (sessions x securities, float64) with a reason code for every missing cell."""
    panel = inputs.panel
    rows, columns = len(panel.sessions), len(panel.security_ids)
    shape = (rows, columns)
    if inputs.unadjusted_close.shape != shape:
        raise ValueError("unadjusted close matrix differs from the panel shape")
    column_index = pd.Index(panel.security_ids)
    components = {name: np.full(shape, np.nan, dtype=np.float64) for name in COMPONENTS}
    component_reasons = {name: np.full(shape, NO_COMPANY_FACTS, dtype=np.int8) for name in COMPONENTS}
    share_as_of = np.full(shape, NO_DAY, dtype=np.int64)
    share_tag = np.zeros(shape, dtype=np.int8)
    issuers: dict[str, list[int]] = {}
    for security, cik in inputs.security_issuer.items():
        position = int(column_index.get_indexer([security])[0])
        if position >= 0:
            issuers.setdefault(cik, []).append(position)
    tag_usage: Counter[str] = Counter()
    conflicts: Counter[str] = Counter()
    accessions: set[str] = set()
    issuers_without_facts: list[str] = []
    for cik in sorted(issuers):
        facts = inputs.load_issuer(cik)
        if facts is None:
            issuers_without_facts.append(cik)
            continue
        series = evaluate_issuer(facts, panel.sessions)
        del facts
        tag_usage.update(series.tag_usage)
        conflicts.update(series.conflicts)
        accessions.update(series.accessions)
        for position in issuers[cik]:
            for name in COMPONENTS:
                components[name][:, position] = series.values[name]
                component_reasons[name][:, position] = series.reasons[name]
            share_as_of[:, position] = series.share_as_of
            share_tag[:, position] = series.share_tag
    shared = np.zeros(columns, dtype=bool)
    for positions in issuers.values():
        if len(positions) > 1:
            shared[positions] = True
    shared_reason = np.where(np.broadcast_to(shared, shape), ISSUER_SHARED_BY_SECURITIES, PRESENT).astype(np.int8)
    share_reason = first_reason(component_reasons["shares"], shared_reason)

    adjusted = panel.close.astype(np.float64)
    value_mv, reason_mv, blanked, events = market_value(inputs.unadjusted_close, adjusted, components["shares"],
                                                        share_reason, share_as_of, panel.sessions)
    c, r = components, component_reasons
    values: dict[str, FloatArray] = {}
    reasons: dict[str, ReasonArray] = {}
    values["earnings_reaction_3d"], reasons["earnings_reaction_3d"], source, release_report = earnings_reaction(
        panel, inputs.releases)
    values["earnings_surprise"], reasons["earnings_surprise"] = c["net_income_change"], r["net_income_change"]
    values["revenue_surprise"], reasons["revenue_surprise"] = c["revenue_change"], r["revenue_change"]
    values["earnings_yield"], reasons["earnings_yield"] = _ratio(
        c["net_income_ttm"], value_mv, first_reason(r["net_income_ttm"], reason_mv))
    values["free_cash_flow_yield"], reasons["free_cash_flow_yield"] = _ratio(
        c["operating_cash_flow_ttm"] - c["capital_expenditure_ttm"], value_mv,
        first_reason(r["operating_cash_flow_ttm"], r["capital_expenditure_ttm"], reason_mv))
    values["sales_to_price"], reasons["sales_to_price"] = _ratio(
        c["revenue_ttm"], value_mv, first_reason(r["revenue_ttm"], reason_mv))
    equity_reason = first_reason(r["stockholders_equity"], np.where(
        (r["stockholders_equity"] == PRESENT) & ~(c["stockholders_equity"] > 0), NONPOSITIVE_DENOMINATOR,
        PRESENT).astype(np.int8))
    values["book_to_price"], reasons["book_to_price"] = _ratio(
        c["stockholders_equity"], value_mv, first_reason(equity_reason, reason_mv))
    values["gross_profit_to_assets"], reasons["gross_profit_to_assets"] = _ratio(
        c["gross_profit_ttm"], c["total_assets"], first_reason(r["gross_profit_ttm"], r["total_assets"]))
    values["operating_profitability"], reasons["operating_profitability"] = _ratio(
        c["operating_income_ttm"], c["stockholders_equity"], first_reason(r["operating_income_ttm"], equity_reason))
    values["accruals"], reasons["accruals"] = _ratio(
        c["net_income_ttm"] - c["operating_cash_flow_ttm"], c["total_assets"],
        first_reason(r["net_income_ttm"], r["operating_cash_flow_ttm"], r["total_assets"]))
    values["asset_growth"], reasons["asset_growth"] = c["asset_growth"], r["asset_growth"]
    issuance_reason = first_reason(r["share_issuance"], shared_reason)
    values["net_share_issuance"] = np.where(issuance_reason == PRESENT, c["share_issuance"], np.nan)
    reasons["net_share_issuance"] = issuance_reason
    for name in PREDICTOR_NAMES:  # a present reason must carry a finite value and vice versa
        finite = np.isfinite(values[name])
        if bool(np.any(finite != (reasons[name] == PRESENT))):
            values[name] = np.where(reasons[name] == PRESENT, values[name], np.nan)
            reasons[name] = np.where((reasons[name] == PRESENT) & ~np.isfinite(values[name]), INCOMPLETE_HISTORY,
                                     reasons[name]).astype(np.int8)
    report = {
        "earnings_releases": release_report,
        "issuers_mapped": len(issuers),
        "issuers_without_company_facts_file": issuers_without_facts,
        "panel_securities_sharing_an_issuer": int(shared.sum()),
        "tag_usage_periods": dict(sorted(tag_usage.items())),
        "periods_dropped_several_values_in_first_filing": dict(sorted(conflicts.items())),
        "filings_contributing_used_periods": len(accessions),
        "split_events_detected": int(events.sum()),
        "securities_with_split_event": int(events.any(axis=0).sum()),
    }
    return FundamentalResult(values=values, reasons=reasons, split_blanked=blanked, share_tag=share_tag,
                             reaction_source_event=source, report=report)


def compute_composites(predictors: Mapping[str, FloatArray], eligible: BoolArray) -> dict[str, FloatArray]:
    """Step 1's composite rule with the step 2a sign table: mean signed centered rank among eligible stocks,
    present only when at least two thirds of the group's inputs are present."""
    signed_ranks = {name: EXPECTED_SIGNS[name] * centered_rank(predictors[name], eligible) for name in PREDICTOR_NAMES}
    composites: dict[str, FloatArray] = {}
    for composite, members in COMPOSITE_GROUPS.items():
        stack = np.stack([signed_ranks[name] for name in members])
        present = np.isfinite(stack).sum(axis=0)
        total = np.where(np.isfinite(stack), stack, 0.0).sum(axis=0)
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = total / present
        composites[composite] = np.where(present >= minimum_present(len(members)), mean, np.nan)
    return composites


def coverage(result: FundamentalResult, composites: Mapping[str, FloatArray], eligible: BoolArray,
             decision_rows: BoolArray) -> dict[str, Any]:
    """Share of eligible decision stock-sessions with a value, securities with any value, and missing reasons."""
    decisions = eligible & decision_rows[:, None]
    denominator = int(decisions.sum())
    report: dict[str, Any] = {}
    for name, values in [*result.values.items(), *composites.items()]:
        present = decisions & np.isfinite(values)
        entry: dict[str, Any] = {
            "eligible_decision_stock_sessions": denominator,
            "stock_sessions_with_value": int(present.sum()),
            "coverage_share": float(present.sum() / denominator) if denominator else None,
            "securities_with_any_value": int(present.any(axis=0).sum()),
        }
        if name in result.reasons:
            codes = result.reasons[name][decisions]
            counted = Counter(int(code) for code in codes[codes != PRESENT])
            entry["missing_stock_sessions_by_reason"] = {MISSING_REASONS[code]: counted[code]
                                                         for code in sorted(counted)}
        report[name] = entry
    share_source = result.share_tag[decisions]
    report["_market_value"] = {
        "split_blanked_cells_all_sessions": int(result.split_blanked.sum()),
        "split_blanked_eligible_decision_stock_sessions": int((result.split_blanked & decisions).sum()),
        "share_count_source_eligible_decision_stock_sessions": {
            SHARE_TAG_NAMES[code]: int((share_source == code).sum()) for code in SHARE_TAG_NAMES},
    }
    used = result.reaction_source_event[decisions & np.isfinite(result.values["earnings_reaction_3d"])]
    report["_earnings_releases_used"] = {"distinct_release_events_used_on_eligible_decisions":
                                         int(np.unique(used[used >= 0]).size) if used.size else 0}
    return report
