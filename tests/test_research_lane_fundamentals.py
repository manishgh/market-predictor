"""Unit tests on synthetic step 2a inputs: frozen signs, event timing, market value and causality."""
from __future__ import annotations

from dataclasses import replace
from datetime import date

import numpy as np
import pandas as pd
import pytest

from market_predictor.research_lane.company_facts import (
    NO_COMPANY_FACTS,
    NO_DAY,
    NO_USABLE_RELEASE,
    PRESENT,
    RELEASE_EXPIRED,
    SPLIT_AFTER_SHARE_COUNT,
    SPLIT_CHECK_NOT_POSSIBLE,
    day_numbers,
)
from market_predictor.research_lane.fundamentals import (
    COMPOSITE_GROUPS,
    EXPECTED_SIGNS,
    PREDICTOR_NAMES,
    FundamentalResult,
    compute_composites,
    compute_fundamental_predictors,
    coverage,
    earnings_reaction,
    market_value,
)
from market_predictor.research_lane.panel import MARKET_TIMEZONE
from market_predictor.research_lane.predictors import minimum_present
from market_predictor.research_lane.sec_filings import event_rows
from tests.support.research_lane_fundamentals import (
    SHARED_SECURITY,
    SPLIT_COLUMN,
    SPLIT_ROW,
    SyntheticFundamentals,
    poisoned_inputs,
    synthetic_fundamental_inputs,
    tiny_panel,
    truncated_inputs,
)

CUTS = (300, 371, 455, 530, 619)


def test_expected_sign_table_is_frozen() -> None:
    assert dict(EXPECTED_SIGNS) == {
        "earnings_reaction_3d": 1, "earnings_surprise": 1, "revenue_surprise": 1, "earnings_yield": 1,
        "free_cash_flow_yield": 1, "sales_to_price": 1, "book_to_price": 1, "gross_profit_to_assets": 1,
        "operating_profitability": 1, "accruals": -1, "asset_growth": -1, "net_share_issuance": -1,
    }
    assert COMPOSITE_GROUPS["composite_earnings"] == PREDICTOR_NAMES[:3]
    assert COMPOSITE_GROUPS["composite_valuation"] == PREDICTOR_NAMES[3:7]
    assert COMPOSITE_GROUPS["composite_quality"] == PREDICTOR_NAMES[7:]
    assert COMPOSITE_GROUPS["composite_fundamental_all"] == PREDICTOR_NAMES
    assert [minimum_present(len(m)) for m in COMPOSITE_GROUPS.values()] == [2, 3, 4, 8]


@pytest.fixture(scope="module")
def synthetic() -> SyntheticFundamentals:
    return synthetic_fundamental_inputs()


def _outputs(result: FundamentalResult, eligible: np.ndarray) -> dict[str, np.ndarray]:
    return {**result.values, **compute_composites(result.values, eligible)}


@pytest.fixture(scope="module")
def full(synthetic: SyntheticFundamentals) -> FundamentalResult:
    return compute_fundamental_predictors(synthetic.inputs)


def _day(synthetic: SyntheticFundamentals, row: int) -> date:
    return pd.Timestamp(synthetic.inputs.panel.sessions[row]).date()


def test_every_input_has_values_and_the_expected_missing_cases(synthetic: SyntheticFundamentals,
                                                               full: FundamentalResult) -> None:
    ids = synthetic.inputs.panel.security_ids
    late = slice(450, None)
    for name in PREDICTOR_NAMES:
        assert np.isfinite(full.values[name][late]).mean() > 0.3, name
        assert (np.isfinite(full.values[name]) == (full.reasons[name] == PRESENT)).all(), name
    for security in ("sec:unmapped", "sec:no-facts"):
        column = ids.index(security)
        assert (full.reasons["earnings_yield"][:, column] == NO_COMPANY_FACTS).all()
    shared = ids.index(SHARED_SECURITY)
    assert np.isnan(full.values["earnings_yield"][:, shared]).all()
    assert np.isnan(full.values["earnings_yield"][:, 1]).all()  # its issuer is shared with SHARED_SECURITY
    assert np.isfinite(full.values["gross_profit_to_assets"][late, shared]).any()
    assert np.isfinite(full.values["earnings_reaction_3d"][late, ids.index("sec:unmapped")]).any()


@pytest.mark.parametrize("cut", CUTS)
def test_truncated_history_gives_identical_values_for_every_input_and_composite(
        synthetic: SyntheticFundamentals, full: FundamentalResult, cut: int) -> None:
    eligible = synthetic.inputs.panel.eligible
    expected = _outputs(full, eligible)
    truncated_input = truncated_inputs(synthetic, _day(synthetic, cut))
    truncated = _outputs(compute_fundamental_predictors(truncated_input), truncated_input.panel.eligible)
    assert set(truncated) == set(PREDICTOR_NAMES) | set(COMPOSITE_GROUPS)
    for name, values in truncated.items():
        assert values.shape[0] == cut + 1
        np.testing.assert_array_equal(values, expected[name][:cut + 1], err_msg=name)
        if cut >= 371:
            assert np.isfinite(values[cut - 30:]).any(), name


@pytest.mark.parametrize("cut", (330, 470))
def test_future_poison_changes_no_input_at_or_before_cut(synthetic: SyntheticFundamentals, full: FundamentalResult,
                                                        cut: int) -> None:
    eligible = synthetic.inputs.panel.eligible
    expected = _outputs(full, eligible)
    poisoned_input = poisoned_inputs(synthetic, cut, seed=cut)
    poisoned_result = compute_fundamental_predictors(poisoned_input)
    poisoned = _outputs(poisoned_result, poisoned_input.panel.eligible)
    changed_later = []
    for name, values in poisoned.items():
        np.testing.assert_array_equal(values[:cut + 1], expected[name][:cut + 1], err_msg=name)
        if not np.array_equal(values[cut + 1:], expected[name][cut + 1:], equal_nan=True):
            changed_later.append(name)
    assert {"earnings_reaction_3d", "earnings_yield", "book_to_price", "composite_fundamental_all"} <= set(changed_later)


def test_split_rule_blanks_market_value_until_a_post_split_share_count(synthetic: SyntheticFundamentals,
                                                                        full: FundamentalResult) -> None:
    reasons = full.reasons["book_to_price"][:, SPLIT_COLUMN]
    blanked_rows = np.flatnonzero(full.split_blanked[:, SPLIT_COLUMN])
    assert blanked_rows.size > 0 and blanked_rows.min() == SPLIT_ROW
    assert (reasons[blanked_rows] == SPLIT_AFTER_SHARE_COUNT).all()
    resumed = blanked_rows.max() + 1
    assert reasons[resumed] == PRESENT and np.isfinite(full.values["book_to_price"][resumed, SPLIT_COLUMN])
    assert not full.split_blanked[:, [c for c in range(30) if c != SPLIT_COLUMN]].any()


def test_market_value_uses_unadjusted_close_times_shares() -> None:
    days = [d.date().isoformat() for d in pd.bdate_range("2024-01-02", periods=8)]
    sessions = np.asarray(days, dtype="datetime64[D]")
    adjusted = np.full((8, 1), 50.0)
    unadjusted = np.full((8, 1), 150.0)  # adjusted history divided by a later factor of 3
    shares = np.full((8, 1), 1000.0)
    as_of = np.full((8, 1), int(day_numbers(sessions[:1])[0]), dtype=np.int64)
    value, reason, blanked, _ = market_value(unadjusted, adjusted, shares, np.zeros((8, 1), dtype=np.int8), as_of,
                                             sessions)
    np.testing.assert_array_equal(value[:, 0], 150_000.0)  # never 50 x 1000
    assert (reason == PRESENT).all() and not blanked.any()


def test_split_rule_on_hand_built_series() -> None:
    days = [d.date().isoformat() for d in pd.bdate_range("2024-01-02", periods=10)]
    sessions = np.asarray(days, dtype="datetime64[D]")
    adjusted = np.full((10, 1), 50.0)
    unadjusted = np.where(np.arange(10)[:, None] < 4, 100.0, 50.0)  # 2-for-1 split effective at row 4
    numbers = day_numbers(sessions)
    shares = np.where(np.arange(10)[:, None] < 7, 1000.0, 2000.0)
    as_of = np.where(np.arange(10)[:, None] < 7, numbers[1], numbers[6]).astype(np.int64)  # post-split count from row 7
    value, reason, blanked, events = market_value(unadjusted, adjusted, shares, np.zeros((10, 1), dtype=np.int8),
                                                  as_of, sessions)
    assert np.flatnonzero(events[:, 0]).tolist() == [4]
    assert np.flatnonzero(blanked[:, 0]).tolist() == [4, 5, 6]
    np.testing.assert_array_equal(value[:4, 0], 100_000.0)
    np.testing.assert_array_equal(value[7:, 0], 100_000.0)
    no_history = market_value(unadjusted, adjusted, shares, np.zeros((10, 1), dtype=np.int8),
                              np.full((10, 1), numbers[0] - 5, dtype=np.int64), sessions)
    assert (no_history[1][:, 0] == SPLIT_CHECK_NOT_POSSIBLE).all()  # share date before the first comparable bar
    missing = market_value(unadjusted, adjusted, shares, np.zeros((10, 1), dtype=np.int8),
                           np.full((10, 1), NO_DAY, dtype=np.int64), sessions)
    assert (missing[1][:, 0] == SPLIT_CHECK_NOT_POSSIBLE).all()


def test_event_session_mapping_for_release_times() -> None:
    sessions = np.asarray(["2024-01-05", "2024-01-08", "2024-01-09", "2024-01-10"], dtype="datetime64[D]")
    local = ["2024-01-08 08:00", "2024-01-08 11:00", "2024-01-08 16:00", "2024-01-08 16:01",
             "2024-01-06 12:00", "2024-01-10 17:00"]
    times = pd.Series(pd.to_datetime(local).tz_localize(MARKET_TIMEZONE).tz_convert("UTC"))
    assert event_rows(sessions, times).tolist() == [1, 1, 1, 2, 1, 4]


def test_reaction_usable_from_the_session_after_the_window_and_expires_after_63_sessions() -> None:
    days = [d.date().isoformat() for d in pd.bdate_range("2024-01-02", periods=90)]
    panel = tiny_panel(days)
    close = panel.close.copy()
    close[11:, 0] = 11.0  # stock +10% from row 11; SPY flat
    panel = replace(panel, close=close)
    accepted = pd.Timestamp(f"{days[10]} 17:00", tz=MARKET_TIMEZONE).tz_convert("UTC")  # after close -> e = 11
    releases = pd.DataFrame({"security_id": [panel.security_ids[0]] * 2, "accession_number": ["a", "b"],
                             "accepted_at_utc": [accepted, accepted + pd.Timedelta(minutes=10)],
                             "available_at_utc": [accepted + pd.Timedelta(minutes=5)] * 2})
    values, reasons, _, report = earnings_reaction(panel, releases)
    assert report["distinct_release_events"] == 1 and report["release_rows_merged_into_same_event_session"] == 1
    column = reasons[:, 0]
    assert (column[:12] == NO_USABLE_RELEASE).all()  # e = 11, usable from e + 1 = 12
    assert values[12, 0] == pytest.approx(11.0 / 10.0 - 1.0)  # close[12] / close[9] - 1, SPY flat
    assert column[11 + 63] == PRESENT and column[11 + 64] == RELEASE_EXPIRED


def test_coverage_counts_eligible_decision_cells(synthetic: SyntheticFundamentals, full: FundamentalResult) -> None:
    eligible = synthetic.inputs.panel.eligible
    decision_rows = np.arange(len(synthetic.inputs.panel.sessions)) >= 290
    report = coverage(full, compute_composites(full.values, eligible), eligible, decision_rows)
    denominator = int((eligible & decision_rows[:, None]).sum())
    entry = report["earnings_yield"]
    assert entry["eligible_decision_stock_sessions"] == denominator
    missing = sum(entry["missing_stock_sessions_by_reason"].values())
    assert entry["stock_sessions_with_value"] + missing == denominator
    assert report["_market_value"]["split_blanked_eligible_decision_stock_sessions"] > 0
