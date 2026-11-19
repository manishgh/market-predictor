"""Unit tests on hand-made facts: rule 2 reader filter, rule 3 boundary, rule 4, quarters, TTM, surprise, shares."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pytest

from market_predictor.research_lane.company_facts import (
    DILUTED_SHARES_TAG,
    INCOMPLETE_HISTORY,
    NOT_YET_REPORTED,
    PRESENT,
    SHARE_CHANGE_OVER_LIMIT,
    SHARE_COUNT_AMBIGUOUS,
    SHARES_OUTSTANDING_TAG,
    STALE,
    Fact,
    IssuerFacts,
    evaluate_issuer,
    first_reported,
    read_company_facts,
    single_quarters,
    standardized_change,
    trailing_year,
)
from market_predictor.research_lane.panel import SealedPeriodError

REVENUE = "us-gaap:Revenues"
NET_INCOME = "us-gaap:NetIncomeLoss"
ASSETS = "us-gaap:Assets"


def _d(text: str) -> date:
    return date.fromisoformat(text)


def fact(tag: str, start: str | None, end: str, value: float, filed: str, accession: str = "a") -> Fact:
    return Fact(tag=tag, start=None if start is None else _d(start), end=_d(end), value=value, filed=_d(filed),
                accession=accession)


def sessions(*days: str) -> np.ndarray:
    return np.asarray(days, dtype="datetime64[D]")


def _companyfacts(rows: dict[str, list[dict[str, object]]]) -> dict[str, object]:
    facts: dict[str, dict[str, object]] = {"us-gaap": {}, "dei": {}}
    for key, items in rows.items():
        taxonomy, tag = key.split(":")
        unit = "shares" if "Shares" in tag else "USD"
        facts[taxonomy][tag] = {"units": {unit: items, "EUR": [{"end": "2020-01-01", "val": 1, "filed": "2020-01-02"}]}}
    return {"cik": 1, "facts": facts}


def test_reader_drops_every_fact_filed_after_the_development_end(tmp_path: Path) -> None:
    path = tmp_path / "company_facts.json"
    path.write_text(json.dumps(_companyfacts({
        ASSETS: [{"end": "2024-03-31", "val": 10, "accn": "x", "filed": "2024-05-28", "form": "10-Q"},
                 {"end": "2024-03-31", "val": 99, "accn": "y", "filed": "2024-05-29", "form": "10-Q/A"},
                 {"end": "2024-06-30", "val": 77, "accn": "z", "filed": "2024-08-01", "form": "10-Q"}],
        "us-gaap:UnneededTag": [{"end": "2024-03-31", "val": 5, "accn": "u", "filed": "2024-04-01"}],
        SHARES_OUTSTANDING_TAG: [{"end": "2024-04-20", "val": 1000, "accn": "x", "filed": "2024-05-01"},
                                 {"end": "2024-07-20", "val": 2000, "accn": "q", "filed": "2025-01-01"}],
    })), encoding="utf-8")
    facts, counts = read_company_facts(path, date(2024, 5, 28))
    assert set(facts) == {ASSETS, SHARES_OUTSTANDING_TAG}
    assert [(f.value, f.filed) for f in facts[ASSETS]] == [(10.0, date(2024, 5, 28))]
    assert [f.value for f in facts[SHARES_OUTSTANDING_TAG]] == [1000.0]
    assert all(f.filed <= date(2024, 5, 28) for items in facts.values() for f in items)
    assert counts[f"rows:{ASSETS}"] == 1


def test_reader_rejects_late_rows_if_the_filter_were_bypassed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    path = tmp_path / "company_facts.json"
    path.write_text(json.dumps(_companyfacts({ASSETS: [{"end": "2024-03-31", "val": 1, "accn": "x",
                                                        "filed": "2024-06-01"}]})), encoding="utf-8")
    import market_predictor.research_lane.company_facts as module
    monkeypatch.setattr(module, "keep_filed_by", lambda document, last: (
        {ASSETS: [fact(ASSETS, None, "2024-03-31", 1, "2024-06-01")]}, module.Counter()))
    with pytest.raises(SealedPeriodError):
        read_company_facts(path, date(2024, 5, 28))


def test_rule_3_fact_filed_on_the_decision_date_is_used_from_the_next_session() -> None:
    facts: IssuerFacts = {ASSETS: [fact(ASSETS, None, "2023-12-31", 100.0, "2024-02-01"),
                                   fact(ASSETS, None, "2024-03-31", 200.0, "2024-05-02")]}
    days = sessions("2024-05-01", "2024-05-02", "2024-05-03")
    series = evaluate_issuer(facts, days)
    np.testing.assert_array_equal(series.values["total_assets"], [100.0, 100.0, 200.0])
    first = evaluate_issuer({ASSETS: [fact(ASSETS, None, "2024-03-31", 200.0, "2024-05-02")]}, days)
    np.testing.assert_array_equal(first.reasons["total_assets"], [NOT_YET_REPORTED, NOT_YET_REPORTED, PRESENT])


def test_rule_4_later_restatement_never_changes_the_originally_reported_value() -> None:
    original = [fact(ASSETS, None, "2023-12-31", 100.0, "2024-02-01"),
                fact(ASSETS, None, "2024-03-31", 200.0, "2024-05-02")]
    restated = original + [fact(ASSETS, None, "2023-12-31", 999.0, "2024-05-02", "later"),
                           fact(ASSETS, None, "2024-03-31", 555.0, "2024-05-10", "later")]
    days = sessions("2024-02-05", "2024-05-01", "2024-05-03", "2024-05-13")
    clean = evaluate_issuer({ASSETS: original}, days)
    dirty = evaluate_issuer({ASSETS: restated}, days)
    np.testing.assert_array_equal(dirty.values["total_assets"], clean.values["total_assets"])
    np.testing.assert_array_equal(dirty.values["total_assets"], [100.0, 100.0, 200.0, 200.0])
    periods = first_reported(restated, (ASSETS,))
    assert periods[(None, _d("2023-12-31"))].value == 100.0
    # Same filing day, two tags: the first tag in the ordered list wins; a later higher-priority tag never does.
    tags = ("us-gaap:Revenues", "us-gaap:SalesRevenueNet")
    durations = first_reported([fact(tags[1], "2024-01-01", "2024-03-31", 5.0, "2024-04-30"),
                                fact(tags[0], "2024-01-01", "2024-03-31", 6.0, "2024-05-15")], tags)
    assert durations[(_d("2024-01-01"), _d("2024-03-31"))].value == 5.0
    same_day = first_reported([fact(tags[1], "2024-01-01", "2024-03-31", 5.0, "2024-04-30"),
                               fact(tags[0], "2024-01-01", "2024-03-31", 6.0, "2024-04-30")], tags)
    assert same_day[(_d("2024-01-01"), _d("2024-03-31"))].value == 6.0


def _year(prefix: str, values: tuple[float, float, float, float], filed: tuple[str, str, str, str],
          tag: str = REVENUE) -> list[Fact]:
    """Q1 3-month, 6-month YTD, 9-month YTD and full year (no direct Q2/Q3/Q4)."""
    year = int(prefix)
    q1, q2, q3, q4 = values
    return [fact(tag, f"{year}-01-01", f"{year}-03-31", q1, filed[0]),
            fact(tag, f"{year}-01-01", f"{year}-06-30", q1 + q2, filed[1]),
            fact(tag, f"{year}-01-01", f"{year}-09-30", q1 + q2 + q3, filed[2]),
            fact(tag, f"{year}-01-01", f"{year}-12-31", q1 + q2 + q3 + q4, filed[3])]


def test_quarter_derivation_and_trailing_year_match_hand_calculation() -> None:
    facts = _year("2022", (10.0, 15.0, 20.0, 25.0), ("2022-04-30", "2022-07-30", "2022-10-30", "2023-02-28"))
    quarters = single_quarters(first_reported(facts, (REVENUE,)))
    assert [(q.start, q.end, q.value, q.usable, q.derived) for q in quarters] == [
        (_d("2022-01-01"), _d("2022-03-31"), 10.0, _d("2022-04-30"), False),
        (_d("2022-04-01"), _d("2022-06-30"), 15.0, _d("2022-07-30"), True),
        (_d("2022-07-01"), _d("2022-09-30"), 20.0, _d("2022-10-30"), True),
        (_d("2022-10-01"), _d("2022-12-31"), 25.0, _d("2023-02-28"), True),  # Q4 = full year - 9 months
    ]
    state = trailing_year(quarters)
    assert state.reason == PRESENT and state.value == pytest.approx(70.0) and state.period_end == _d("2022-12-31")
    assert trailing_year(quarters[:3]).reason == INCOMPLETE_HISTORY
    days = sessions("2023-02-28", "2023-03-01")
    series = evaluate_issuer({REVENUE: facts}, days)
    np.testing.assert_array_equal(series.reasons["revenue_ttm"], [INCOMPLETE_HISTORY, PRESENT])
    assert series.values["revenue_ttm"][1] == pytest.approx(70.0)


def test_derived_quarter_is_usable_only_from_the_later_input_filing() -> None:
    facts = [fact(REVENUE, "2022-01-01", "2022-06-30", 30.0, "2022-07-30"),
             fact(REVENUE, "2022-01-01", "2022-03-31", 12.0, "2022-08-15")]
    quarters = single_quarters(first_reported(facts, (REVENUE,)))
    derived = [q for q in quarters if q.end == _d("2022-06-30")][0]
    assert derived.value == 18.0 and derived.usable == _d("2022-08-15")


def test_trailing_year_requires_four_contiguous_quarters() -> None:
    facts = [fact(REVENUE, "2022-01-01", "2022-03-31", 1.0, "2022-04-30"),
             fact(REVENUE, "2022-04-01", "2022-06-30", 2.0, "2022-07-30"),
             fact(REVENUE, "2022-07-01", "2022-09-30", 3.0, "2022-10-30"),
             fact(REVENUE, "2023-01-01", "2023-03-31", 4.0, "2023-04-30")]  # 2022 Q4 never reported
    quarters = single_quarters(first_reported(facts, (REVENUE,)))
    assert trailing_year(quarters).reason == INCOMPLETE_HISTORY
    assert trailing_year(quarters).period_end == _d("2023-03-31")
    full = quarters[:3] + single_quarters(first_reported(
        [fact(REVENUE, "2022-10-01", "2022-12-31", 9.0, "2023-02-20")], (REVENUE,))) + quarters[3:]
    assert trailing_year(full).value == pytest.approx(2.0 + 3.0 + 9.0 + 4.0)


def test_standardized_change_matches_hand_calculation() -> None:
    values = [10.0, 11.0, 9.0, 12.0, 13.0, 12.0, 11.0, 15.0, 14.0, 16.0, 12.0, 19.0]  # 2021Q1..2023Q4
    ends = ["03-31", "06-30", "09-30", "12-31"]
    starts = ["01-01", "04-01", "07-01", "10-01"]
    facts = [fact(NET_INCOME, f"{2021 + k // 4}-{starts[k % 4]}", f"{2021 + k // 4}-{ends[k % 4]}", value,
                  f"{2021 + k // 4 + (1 if k % 4 == 3 else 0)}-{['05-01', '08-01', '11-01', '02-15'][k % 4]}")
             for k, value in enumerate(values)]
    quarters = single_quarters(first_reported(facts, (NET_INCOME,)))
    state = standardized_change(quarters)
    differences = np.array(values[4:]) - np.array(values[:8])  # 8 year-over-year differences
    assert state.reason == PRESENT
    assert state.value == pytest.approx(differences[-1] / np.std(differences, ddof=1))
    # Only five differences available -> incomplete.
    assert standardized_change(quarters[:9]).reason == INCOMPLETE_HISTORY
    assert standardized_change(quarters[:10]).reason == PRESENT


def test_values_expire_200_days_after_the_latest_period_end() -> None:
    facts: IssuerFacts = {ASSETS: [fact(ASSETS, None, "2023-12-31", 100.0, "2024-02-01")]}
    series = evaluate_issuer(facts, sessions("2024-07-18", "2024-07-19"))  # 200 and 201 days after the end
    np.testing.assert_array_equal(series.reasons["total_assets"], [PRESENT, STALE])


def test_share_count_fallback_ambiguity_and_issuance() -> None:
    days = sessions("2023-06-01", "2024-06-03")
    diluted = [fact(DILUTED_SHARES_TAG, "2023-01-01", "2023-03-31", 500.0, "2023-05-01")]
    series = evaluate_issuer({DILUTED_SHARES_TAG: diluted}, days)
    assert series.values["shares"][0] == 500.0 and series.share_tag[0] == 2
    dei = [fact(SHARES_OUTSTANDING_TAG, None, "2023-04-20", 1000.0, "2023-05-01"),
           fact(SHARES_OUTSTANDING_TAG, None, "2024-04-20", 1100.0, "2024-05-01")]
    both = evaluate_issuer({DILUTED_SHARES_TAG: diluted, SHARES_OUTSTANDING_TAG: dei}, days)
    np.testing.assert_array_equal(both.values["shares"], [1000.0, 1100.0])
    assert list(both.share_tag) == [1, 1]
    assert np.isnan(both.values["share_issuance"][0]) and both.reasons["share_issuance"][0] == INCOMPLETE_HISTORY
    assert both.values["share_issuance"][1] == pytest.approx(0.1)
    jump = [dei[0], fact(SHARES_OUTSTANDING_TAG, None, "2024-04-20", 2100.0, "2024-05-01")]
    assert evaluate_issuer({SHARES_OUTSTANDING_TAG: jump}, days).reasons["share_issuance"][1] == SHARE_CHANGE_OVER_LIMIT
    classes = dei + [fact(SHARES_OUTSTANDING_TAG, None, "2024-04-20", 300.0, "2024-05-01")]
    ambiguous = evaluate_issuer({SHARES_OUTSTANDING_TAG: classes}, days)
    assert ambiguous.reasons["shares"][1] == SHARE_COUNT_AMBIGUOUS and np.isnan(ambiguous.values["shares"][1])
