"""Synthetic step 2a inputs (company facts, earnings releases, unadjusted closes) for unit tests only.

Nothing here is real-data evidence. ``synthetic_fundamental_inputs`` builds a 40-security panel
whose issuers file quarterly 10-Q / annual 10-K style facts (with year-to-date cash flows,
restated prior-year comparatives, a tag switch, a missing quarterly filing, an issuer without a
dei share count and a 2-for-1 split), plus item 2.02 releases at varied times of day.
"""
from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, replace
from datetime import date, timedelta

import numpy as np
import pandas as pd

from market_predictor.research_lane.company_facts import (
    DILUTED_SHARES_TAG,
    SHARES_OUTSTANDING_TAG,
    Fact,
    IssuerFacts,
)
from market_predictor.research_lane.fundamentals import FundamentalInputs
from market_predictor.research_lane.panel import MARKET_TIMEZONE, ResearchPanel
from tests.support.research_lane_panel import synthetic_panel

SPLIT_COLUMN = 4
SPLIT_ROW = 400
SHARED_SECURITY = "sp500-historical:shared"
RELEASE_CLOCKS = ("08:00", "11:00", "16:30", "16:00", "19:45")


def cik(index: int) -> str:
    return f"{index + 1:010d}"


def _quarter_ends(first_year: int, last_year: int) -> list[date]:
    ends = []
    for year in range(first_year, last_year + 1):
        ends += [date(year, 3, 31), date(year, 6, 30), date(year, 9, 30), date(year, 12, 31)]
    return ends


def _revenue_tag(issuer: int, end: date) -> str:
    if issuer % 3 == 0:
        return "us-gaap:Revenues"
    if issuer % 3 == 1 or end >= date(2018, 1, 1):
        return "us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax"
    return "us-gaap:SalesRevenueNet"


def issuer_facts(issuer: int, split_day: date | None = None, *, seed: int = 0) -> IssuerFacts:
    """Quarterly filings 2015-2020 for one synthetic issuer."""
    rng = np.random.default_rng(1000 + issuer + seed)
    ends = _quarter_ends(2015, 2020)
    revenue = {end: 1e9 * (1 + issuer / 10) * (1.02 ** k) * (1 + rng.normal(0, 0.05)) for k, end in enumerate(ends)}
    margin = {end: 0.1 + rng.normal(0, 0.03) for end in ends}
    facts: IssuerFacts = {}

    def add(tag: str, start: date | None, end: date, value: float, filed: date, accession: str) -> None:
        facts.setdefault(tag, []).append(Fact(tag=tag, start=start, end=end, value=float(value), filed=filed,
                                              accession=accession))

    def flows(end: date) -> dict[str, float]:
        rev = revenue[end]
        return {"revenue": rev, "net_income": rev * margin[end], "operating_income": rev * (margin[end] + 0.05),
                "operating_cash_flow": rev * (margin[end] + 0.03), "capital_expenditure": rev * 0.04,
                "gross_profit": rev * 0.4, "cost_of_revenue": rev * 0.6}

    tags = {"net_income": "us-gaap:NetIncomeLoss", "operating_income": "us-gaap:OperatingIncomeLoss",
            "operating_cash_flow": "us-gaap:NetCashProvidedByUsedInOperatingActivities",
            "capital_expenditure": "us-gaap:PaymentsToAcquirePropertyPlantAndEquipment",
            "gross_profit": "us-gaap:GrossProfit", "cost_of_revenue": "us-gaap:CostOfRevenue"}
    income = ("revenue", "net_income", "operating_income", "gross_profit" if issuer % 2 == 0 else "cost_of_revenue")
    shares = 1e8 * (1 + issuer / 20)
    for k, end in enumerate(ends):
        year_start = date(end.year, 1, 1)
        quarter_number = (end.month - 1) // 3 + 1
        annual = quarter_number == 4
        filed = end + timedelta(days=(60 if annual else 35) + issuer % 9)
        if issuer % 11 == 5 and end == date(2019, 6, 30):
            continue  # missing quarterly filing: the 2019 Q2 values are first reported a year later
        accession = f"{cik(issuer)}-{end.isoformat()}"
        year_ends = [e for e in ends if e.year == end.year and e <= end]

        def tag_of(item: str, period_end: date) -> str:
            return _revenue_tag(issuer, period_end) if item == "revenue" else tags[item]

        for item in (*income, "operating_cash_flow", "capital_expenditure"):
            ytd = sum(flows(e)[item] for e in year_ends)
            add(tag_of(item, end), year_start, end, ytd, filed, accession)
            if not annual and item in income:
                quarter_start = date(end.year, end.month - 2, 1)
                add(tag_of(item, end), quarter_start, end, flows(end)[item], filed, accession)
                prior_end = date(end.year - 1, end.month, end.day)
                if prior_end in revenue:  # restated prior-year comparative (rule 4 keeps the original)
                    add(tag_of(item, prior_end), date(prior_end.year, prior_end.month - 2, 1), prior_end,
                        flows(prior_end)[item] * 1.07, filed, accession)
        assets = 5e9 * (1 + issuer / 10) * (1.01 ** k)
        add("us-gaap:Assets", None, end, assets, filed, accession)
        add("us-gaap:StockholdersEquity", None, end, assets * (0.4 if issuer % 13 != 7 else -0.1), filed, accession)
        add("us-gaap:Assets", None, date(end.year - 1, 12, 31), 1.0, filed, accession)  # later comparative
        shares *= 1.0 + rng.normal(0, 0.01)
        as_of = filed - timedelta(days=7)
        count = shares * (2 if split_day is not None and as_of >= split_day else 1)
        if issuer % 7 != 3:
            add(SHARES_OUTSTANDING_TAG, None, as_of, count, filed, accession)
        if not annual:
            add(DILUTED_SHARES_TAG, date(end.year, end.month - 2, 1), end, count * 1.01, filed, accession)
    return facts


def release_table(security_ids: list[str], ends: list[date], *, offset_days: int = 28) -> pd.DataFrame:
    rows = []
    for position, security in enumerate(security_ids):
        for k, end in enumerate(ends):
            day = end + timedelta(days=offset_days + position % 5)
            clock = RELEASE_CLOCKS[(position + k) % len(RELEASE_CLOCKS)]
            accepted = pd.Timestamp(f"{day.isoformat()} {clock}", tz=MARKET_TIMEZONE).tz_convert("UTC")
            rows.append({"security_id": security, "sec_cik": security.split(":")[-1],
                         "accession_number": f"{security}-{end.isoformat()}-8k", "accepted_at_utc": accepted,
                         "available_at_utc": accepted + pd.Timedelta(minutes=5)})
    return pd.DataFrame(rows)


@dataclass(frozen=True)
class SyntheticFundamentals:
    inputs: FundamentalInputs
    facts: Mapping[str, IssuerFacts]


def synthetic_fundamental_inputs(*, seed: int = 7) -> SyntheticFundamentals:
    base = synthetic_panel(seed=seed)
    columns = len(base.security_ids)
    ids = [f"cik:{cik(i)}" for i in range(columns - 3)] + ["sec:unmapped", "sec:no-facts", SHARED_SECURITY]
    panel = replace(base, security_ids=tuple(ids))
    adjusted = panel.close.astype(np.float64)
    unadjusted = adjusted * 1.0
    unadjusted[:SPLIT_ROW, SPLIT_COLUMN] *= 2.0  # 2-for-1 split at SPLIT_ROW: raw history was twice the adjusted
    unadjusted[:, 9] *= 3.0  # constant factor (no split): market value must still use the unadjusted close
    unadjusted[200:260, 12] = np.nan  # unadjusted bars missing for a stretch
    split_day = pd.Timestamp(panel.sessions[SPLIT_ROW]).date()
    facts = {cik(i): issuer_facts(i, split_day if i == SPLIT_COLUMN else None) for i in range(columns - 3)}
    mapping = {security: security.split(":")[1] for security in ids if security.startswith("cik:")}
    mapping["sec:no-facts"] = "9999999999"
    mapping[SHARED_SECURITY] = cik(1)
    releases = release_table(ids, _quarter_ends(2018, 2020))
    inputs = FundamentalInputs(panel=panel, unadjusted_close=unadjusted, releases=releases, security_issuer=mapping,
                               load_issuer=facts.get)
    return SyntheticFundamentals(inputs=inputs, facts=facts)


def truncated_inputs(source: SyntheticFundamentals, cut_day: date) -> FundamentalInputs:
    """What the readers would return with ``cut_day`` as development end."""
    inputs = source.inputs
    panel = inputs.panel.truncated(cut_day)
    rows = len(panel.sessions)
    facts = {key: {tag: [f for f in items if f.filed <= cut_day] for tag, items in issuer.items()}
             for key, issuer in source.facts.items()}
    cutoff = pd.Timestamp(cut_day + timedelta(days=1), tz=MARKET_TIMEZONE).tz_convert("UTC")
    releases = inputs.releases.loc[inputs.releases["accepted_at_utc"] < cutoff].reset_index(drop=True)
    return FundamentalInputs(panel=panel, unadjusted_close=inputs.unadjusted_close[:rows].copy(), releases=releases,
                             security_issuer=inputs.security_issuer, load_issuer=facts.get)


def poisoned_inputs(source: SyntheticFundamentals, cut_row: int, seed: int) -> FundamentalInputs:
    """Later prices, later filings (new periods, restatements, new tags, share counts), later releases, a split."""
    rng = np.random.default_rng(seed)
    inputs = source.inputs
    panel = inputs.panel
    cut_day = pd.Timestamp(panel.sessions[cut_row]).date()
    later = slice(cut_row + 1, None)

    def garbage(values: np.ndarray) -> np.ndarray:
        result = values.copy()
        noise = rng.uniform(1.0, 1e6, result[later].shape).astype(result.dtype)
        noise[rng.random(noise.shape) < 0.1] = np.nan
        result[later] = noise
        return result

    eligible = panel.eligible.copy()
    eligible[later] = rng.random(eligible[later].shape) > 0.5
    poisoned_panel = replace(panel, open=garbage(panel.open), high=garbage(panel.high), low=garbage(panel.low),
                             close=garbage(panel.close), volume=garbage(panel.volume),
                             spy_open=garbage(panel.spy_open), spy_close=garbage(panel.spy_close),
                             qqq_open=garbage(panel.qqq_open), qqq_close=garbage(panel.qqq_close), eligible=eligible)
    unadjusted = garbage(inputs.unadjusted_close)
    unadjusted[cut_row + 3:, :] /= 4.0  # a later 4-for-1 split on every security
    after = cut_day + timedelta(days=1)
    facts: dict[str, IssuerFacts] = {}
    for key, issuer in source.facts.items():
        poisoned: IssuerFacts = {tag: list(items) for tag, items in issuer.items()}
        for tag, items in issuer.items():
            for fact in items:  # restatement of every period, filed after the cut
                poisoned[tag].append(replace(fact, value=-3.0 * fact.value + 1e9, filed=after + timedelta(days=1),
                                             accession="late-restatement"))
            for step in range(1, 4):  # new periods filed after the cut
                end = after + timedelta(days=30 * step)
                start = None if items[0].start is None else end - timedelta(days=91)
                poisoned[tag].append(Fact(tag=tag, start=start, end=end, value=float(rng.uniform(-1e12, 1e12)),
                                          filed=after + timedelta(days=step), accession="late-new"))
        poisoned.setdefault("us-gaap:Revenues", []).extend(  # higher-priority tag for old periods, filed late
            replace(fact, tag="us-gaap:Revenues", value=7.0, filed=after)
            for tag in ("us-gaap:RevenueFromContractWithCustomerExcludingAssessedTax", "us-gaap:SalesRevenueNet")
            for fact in issuer.get(tag, []))
        poisoned.setdefault(SHARES_OUTSTANDING_TAG, []).append(Fact(
            tag=SHARES_OUTSTANDING_TAG, start=None, end=cut_day - timedelta(days=3), value=1.0, filed=after,
            accession="late-shares"))
        facts[key] = poisoned
    extra = release_table(list(panel.security_ids), [cut_day - timedelta(days=27)], offset_days=28)
    releases = pd.concat([inputs.releases, extra], ignore_index=True)
    return FundamentalInputs(panel=poisoned_panel, unadjusted_close=unadjusted, releases=releases,
                             security_issuer=inputs.security_issuer, load_issuer=facts.get)


def tiny_panel(days: list[str], columns: int = 1, close: float = 10.0) -> ResearchPanel:
    sessions = np.asarray(days, dtype="datetime64[D]")
    shape = (len(days), columns)
    prices = np.full(shape, close, dtype=np.float32)
    flat = np.full(len(days), 100.0, dtype=np.float32)
    return ResearchPanel(sessions=sessions, security_ids=tuple(f"cik:{cik(i)}" for i in range(columns)),
                         open=prices.copy(), high=prices.copy(), low=prices.copy(), close=prices.copy(),
                         volume=prices.copy(), eligible=np.ones(shape, dtype=bool),
                         sector=np.zeros(shape, dtype=np.int16), sector_names=("A",), spy_open=flat.copy(),
                         spy_close=flat.copy(), qqq_open=flat.copy(), qqq_close=flat.copy(),
                         development_end=date.fromisoformat(days[-1]))
