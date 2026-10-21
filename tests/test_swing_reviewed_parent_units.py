"""Synthetic UNIT parent spin units; no operational evidence or daughter-share admission."""
from __future__ import annotations

import json
from datetime import date
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_windows import ActionWindow, action_window
from market_predictor.swing.datasets.backtest_corporate_actions import BacktestCorporateActions, load_backtest_corporate_action_context
from tests.test_swing_funded_policy_inputs import provider as provider
from tests.test_swing_reviewed_action_dates import _spin, _with_facts


def _units(provider: Any, *, fault: str | None = None) -> tuple[BacktestCorporateActions, dict[str, Any]]:
    record = _spin()
    old = _with_facts(provider, "spin_offs", record,
        {"ex_date": "2024-01-10", "record_date": "2024-01-08", "payable_date": "2024-01-09"})
    dates = json.loads((provider.root / "reviewed-dates.json").read_text())
    fact = {"action_id": record["id"], "retained_record_sha256": json_sha256(record), "source_symbol": "AAA",
        "ex_date": "2024-01-10", "parent_units_per_owned_unit": 1.0,
        "regular_way_before_ex_includes_distribution": True, "official_receipts": dates["records"][0]["official_receipts"]}
    if fault == "record":
        fact["retained_record_sha256"] = "e" * 64
    if fault == "date":
        fact["ex_date"] = "2024-01-11"
    if fault == "symbol":
        fact["source_symbol"] = "CHILD"
    if fault == "ratio":
        fact["parent_units_per_owned_unit"] = 2.0
    artifact = {"schema_version": "market_predictor.reviewed_parent_spin_units",
        "scope": "raw_parent_units_and_new_purchase_relevance_only", "production_eligible": False,
        "source_files": dates["source_files"], "records": [fact, fact] if fault == "duplicate" else [fact]}
    path = provider.root / "parent-units.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    policy = provider.root / "parent-units.toml"
    policy.write_text((provider.root / "reviewed-backtest.toml").read_text()
        + f'\n[reviewed_parent_units]\npath="{path.name}"\nsha256="{file_sha256(path)}"\n', encoding="utf-8")
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    bridge = BacktestCorporateActions(provider.root, provider.unit_action_evidence, context, memory_guard=lambda: None)
    assert not old.reviewed_parent_units
    return bridge, record


def test_parent_raw_unit_review_preserves_price_history_without_valuing_daughter(provider: Any) -> None:
    bridge, record = _units(provider)
    window = action_window("spin_offs", record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17), price_history=True)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    assert not bridge.terms and not bridge.split_ratios("AAA", date(2024, 1, 1), date(2024, 1, 17))
    assert record == _spin()


@pytest.mark.parametrize("first,last", [("2024-01-08", "2024-01-09"), ("2024-01-10", "2024-01-17")])
def test_regular_way_parent_sale_before_ex_or_new_ex_day_purchase_has_no_daughter_entitlement(
    provider: Any, first: str, last: str,
) -> None:
    bridge, record = _units(provider)
    assert bridge.irrelevance_reason("AAA", action_window("spin_offs", record), date.fromisoformat(first), date.fromisoformat(last))


def test_spanning_or_daughter_holding_and_invalid_gate_remain_unavailable(provider: Any) -> None:
    bridge, record = _units(provider)
    window = action_window("spin_offs", record)
    assert bridge.irrelevance_reason("AAA", window, date(2024, 1, 9), date(2024, 1, 10)) is None
    assert bridge.irrelevance_reason("CHILD", window, date(2024, 1, 3), date(2024, 1, 17), price_history=True) is None
    invalid = ActionWindow(record["id"], "spin_offs", None, None, "conflicting_action_record_identity")
    assert bridge.irrelevance_reason("AAA", invalid, date(2024, 1, 3), date(2024, 1, 17), price_history=True) is None
    disjoint = ActionWindow(record["id"], "spin_offs", date(2025, 1, 2), date(2025, 1, 3), "corporate_terms_require_replay")
    assert bridge.irrelevance_reason("AAA", disjoint, date(2024, 1, 3), date(2024, 1, 17), price_history=True) is None


@pytest.mark.parametrize("fault", ["record", "date", "symbol", "ratio", "duplicate"])
def test_changed_source_dates_units_or_duplicate_review_cannot_waive_parent_gate(provider: Any, fault: str) -> None:
    with pytest.raises((ValueError, DataReadinessError)):
        _units(provider, fault=fault)
