from datetime import date

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets.action_windows import ActionWindow, action_window, relevant_action_windows


def test_processing_date_never_replaces_missing_effective_date() -> None:
    window = action_window("name_changes", {"id": "rename", "process_date": "2026-09-11"})
    assert window.first_relevant_date is None
    assert window.intersects(date(2020, 1, 2), date(2020, 1, 16))


def test_delayed_processing_does_not_hide_historical_dividend() -> None:
    window = action_window("cash_dividends", {"id": "dividend", "process_date": "2026-09-11",
        "ex_date": "2020-01-06", "payable_date": "2020-02-01", "special": False})
    assert window.intersects(date(2020, 1, 2), date(2020, 1, 16))
    assert not window.intersects(date(2019, 12, 1), date(2019, 12, 20))


def test_due_bill_period_is_not_truncated_to_ex_date() -> None:
    window = action_window("spin_offs", {"id": "spinoff", "record_date": "2022-12-27",
        "payable_date": "2023-01-13", "ex_date": "2023-01-17", "due_bill_redemption_date": "2023-01-18"})
    assert window.intersects(date(2022, 12, 28), date(2023, 1, 12))
    assert window.last_relevant_date == date(2023, 1, 18)


@pytest.mark.parametrize("family,record", [
    ("future_provider_family", {"ex_date": "2026-01-01"}),
    ("cash_dividends", {"ex_date": None}),
    ("stock_mergers", {"effective_date": "not-a-date"}),
    ("cash_dividends", {"ex_date": "2020-01-02", "payable_date": 20200120}),
])
def test_unknown_or_invalid_actions_are_not_negative_evidence(family: str, record: dict[str, object]) -> None:
    window = action_window(family, record)
    assert window.intersects(date(2019, 7, 9), date(2019, 7, 23))


def test_known_outside_window_is_not_an_intersection() -> None:
    assert not relevant_action_windows({"cash_dividends": [{"id": "dated", "ex_date": "2025-01-01", "special": False}]},
        first=date(2020, 1, 2), last=date(2020, 1, 16))


def test_reversed_interval_rejected() -> None:
    with pytest.raises(DataReadinessError):
        action_window("cash_mergers", {}).intersects(date(2020, 2, 1), date(2020, 1, 1))


def test_ex_date_alone_cannot_hide_due_bill_exposure() -> None:
    window = action_window("spin_offs", {"id": "spinoff", "ex_date": "2023-01-17"})
    assert window.intersects(date(2022, 12, 28), date(2023, 1, 12))
    assert window.unavailable_reason == "unbounded_entitlement_or_due_bill_dates"


def test_entry_on_or_after_ordinary_ex_date_does_not_earn_earlier_dividend() -> None:
    window = action_window("cash_dividends", {"id": "dividend", "ex_date": "2020-01-06",
        "record_date": "2020-01-07", "payable_date": "2020-02-01", "special": False})
    assert window.intersects(date(2020, 1, 3), date(2020, 1, 17))
    assert not window.intersects(date(2020, 1, 6), date(2020, 1, 21))
    assert not window.intersects(date(2020, 1, 7), date(2020, 2, 3))


def test_malformed_payment_date_does_not_create_unearned_ordinary_dividend() -> None:
    window = action_window("cash_dividends", {"id": "dividend", "ex_date": "2020-01-06",
        "payable_date": "invalid", "special": False})
    assert not window.intersects(date(2020, 1, 6), date(2020, 1, 21))
    assert window.intersects(date(2020, 1, 3), date(2020, 1, 17))


def test_independently_known_lower_bound_does_not_require_invented_upper_bound() -> None:
    window = ActionWindow("reviewed", "cash_dividends", date(2023, 12, 27), None, "resolution_unavailable")
    assert not window.intersects(date(2019, 7, 10), date(2019, 7, 23))
    assert not window.intersects(date(2023, 12, 1), date(2023, 12, 26))
    assert window.intersects(date(2023, 12, 26), date(2023, 12, 27))
    assert window.intersects(date(2024, 1, 17), date(2024, 1, 31))


def test_independently_known_upper_bound_respects_unknown_lower_bound() -> None:
    window = ActionWindow("reviewed", "cash_dividends", None, date(2023, 12, 27), "onset_unavailable")
    assert window.intersects(date(2019, 7, 10), date(2019, 7, 23))
    assert not window.intersects(date(2024, 1, 17), date(2024, 1, 31))
