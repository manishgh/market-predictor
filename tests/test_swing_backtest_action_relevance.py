"""Synthetic UNIT evidence for source-bound ownership relevance; no real returns."""
from __future__ import annotations

from datetime import date
from typing import Any

import pytest

from market_predictor.swing.datasets.action_windows import ActionWindow, action_window
from tests.test_swing_backtest_corporate_actions import _bridge, _cash
from tests.test_swing_funded_policy_inputs import provider as provider


def _merger() -> dict[str, Any]:
    return {"id": "merger", "acquirer_symbol": "AAA", "acquiree_symbol": "OTHER",
        "acquirer_cusip": "012345678", "acquiree_cusip": "987654321", "effective_date": "2024-01-05", "rate": 10.0}


@pytest.mark.parametrize("family", ["cash_mergers", "stock_mergers", "stock_and_cash_mergers"])
def test_acquirer_shares_are_not_acquiree_consideration(provider: Any, family: str) -> None:
    record = _merger()
    bridge = _bridge(provider, {family: [record]})
    window = action_window(family, record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17))
    assert bridge.irrelevance_reason("OTHER", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    assert not provider.lot(provider.selected_ids[0]).missing_reasons


@pytest.mark.parametrize("change", [{"acquiree_symbol": "AAA"}, {"acquiree_cusip": "012345678"},
    {"acquirer_cusip": None}, {"surviving_symbol": "AAA"}])
def test_merger_ambiguous_roles_or_classes_remain_gated(provider: Any, change: dict[str, Any]) -> None:
    record = _merger() | change
    _bridge(provider, {"cash_mergers": [record]})
    assert provider.lot(provider.selected_ids[0]).missing_reasons


@pytest.mark.parametrize("same", [True, False])
def test_name_change_uses_same_cusip_not_process_clock(provider: Any, same: bool) -> None:
    record = {"id": "name", "old_symbol": "AAA", "new_symbol": "NEW", "old_cusip": "012345678",
        "new_cusip": "012345678" if same else "987654321", "process_date": "2035-01-01"}
    _bridge(provider, {"name_changes": [record]})
    lot = provider.lot(provider.selected_ids[0])
    assert bool(lot.missing_reasons) is not same


def test_special_cash_is_raw_price_history_but_not_holding_cash(provider: Any) -> None:
    record = _cash("2024-01-05") | {"special": True}
    bridge = _bridge(provider, {"cash_dividends": [record]})
    window = action_window("cash_dividends", record)
    assert bridge.irrelevance_reason("AAA", window, date(2023, 12, 1), date(2024, 1, 2), price_history=True)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    lot = provider.lot(provider.selected_ids[0])
    assert lot.decision_atr is not None
    assert lot.specification is None and lot.missing_reasons
    assert not bridge.terms


@pytest.mark.parametrize("day", ["2020-01-06", "2025-01-06"])
@pytest.mark.parametrize("role", ["source", "distributed"])
def test_distribution_outside_new_holding_is_not_a_claim_of_that_holding(provider: Any, day: str, role: str) -> None:
    record = {"id": "spin", "source_symbol": "AAA" if role == "source" else "PARENT",
        "new_symbol": "CHILD" if role == "source" else "AAA", "ex_date": day,
        "record_date": day, "payable_date": day, "source_rate": 1, "new_rate": 0.5}
    bridge = _bridge(provider, {"spin_offs": [record]})
    assert not bridge.terms
    assert not provider.lot(provider.selected_ids[0]).missing_reasons


@pytest.mark.parametrize("change", [{"ex_date": "2024-01-05"}, {"record_date": "2023-12-20"},
    {"payable_date": "bad"}, {"ex_date": None}, {"record_date": None}, {"payable_date": None},
    {"due_bill_redemption_date": "2023-12-20"}])
def test_future_distribution_requires_all_economic_anchors_after_sale(provider: Any, change: dict[str, Any]) -> None:
    record = {"id": "spin", "source_symbol": "AAA", "new_symbol": "CHILD", "ex_date": "2025-01-06",
        "record_date": "2025-01-02", "payable_date": "2025-01-08"} | change
    _bridge(provider, {"spin_offs": [record]})
    assert provider.lot(provider.selected_ids[0]).missing_reasons


def test_past_distribution_does_not_ignore_a_known_later_payment(provider: Any) -> None:
    record = {"id": "spin", "source_symbol": "AAA", "new_symbol": "CHILD", "ex_date": "2023-12-01",
        "payable_date": "2024-01-05"}
    _bridge(provider, {"spin_offs": [record]})
    assert provider.lot(provider.selected_ids[0]).missing_reasons


@pytest.mark.parametrize("reason", ["conflicting_action_record_identity", "unsupported_action_participant_field",
    "action_query_unavailable", "invalid_action_economic_date", "unclassified_action_family"])
def test_valid_record_cannot_erase_local_or_coverage_errors(provider: Any, reason: str) -> None:
    record = _merger()
    bridge = _bridge(provider, {"cash_mergers": [record]})
    window = ActionWindow(record["id"], "cash_mergers", None, None, reason)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17)) is None


def test_conflicting_ids_cannot_authorize_irrelevance_or_events(provider: Any) -> None:
    _bridge(provider, {"cash_mergers": [_merger(), _merger() | {"rate": 11.0}]})
    assert provider.lot(provider.selected_ids[0]).missing_reasons


def test_unmatched_symbol_or_window_family_has_no_relevance_authority(provider: Any) -> None:
    record = _cash("2020-01-06") | {"special": True}
    bridge = _bridge(provider, {"cash_dividends": [record]})
    window = action_window("cash_dividends", record)
    assert bridge.irrelevance_reason("OTHER", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    assert bridge.irrelevance_reason("AAA", ActionWindow(record["id"], "stock_dividends", None, None,
        window.unavailable_reason), date(2024, 1, 3), date(2024, 1, 17)) is None


def test_irrelevance_preserves_official_identity_gaps(provider: Any) -> None:
    from market_predictor.swing.contracts.corrected_outcomes import OfficialWindow
    record = _merger()
    _bridge(provider, {"cash_mergers": [record]})
    provider.policy.official_windows = (OfficialWindow(security_id="common:A", first_session=date(2024, 1, 3),
        last_session=date(2024, 1, 17), document={"path": "document.txt", "sha256": "a" * 64},
        record_locator="identity conflict", reason="identity unresolved"),)
    assert provider.lot(provider.selected_ids[0]).missing_reasons
