"""Synthetic UNIT refutation evidence; real provider records are never rewritten."""
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
from tests.test_swing_backtest_corporate_actions import _bridge
from tests.test_swing_funded_policy_inputs import provider as provider
from tests.test_swing_reviewed_action_dates import _name, _with_facts


def _disposition(provider: Any, *, fault: str | None = None) -> tuple[BacktestCorporateActions, dict[str, Any]]:
    _with_facts(provider, "name_changes", _name() | {"old_symbol": "REFERENCE", "new_symbol": "REFERENCE"},
        {"effective_date": "2024-02-01"})
    sources = json.loads((provider.root / "reviewed-dates.json").read_text())
    record = {"id": "refuted-stock-dividend", "symbol": "AAA", "cusip": "012345678", "ex_date": "2024-01-05",
        "payable_date": "2024-01-05", "record_date": "2024-01-08", "rate": 1.3}
    _bridge(provider, {"stock_dividends": [record]})
    fact = {"action_id": record["id"], "family": "stock_dividends", "symbol": "AAA",
        "retained_record_sha256": json_sha256(record), "disposition": "refuted_by_audited_issuer_share_reconciliation",
        "fiscal_first": "2023-10-01", "fiscal_last": "2024-09-30", "official_receipts": sources["records"][0]["official_receipts"]}
    if fault == "record":
        fact["retained_record_sha256"] = "e" * 64
    if fault == "period":
        fact["fiscal_first"], fact["fiscal_last"] = "2022-10-01", "2023-09-30"
    if fault == "symbol":
        fact["symbol"] = "OTHER"
    if fault == "family":
        fact["family"] = "cash_dividends"
    artifact = {"schema_version": "market_predictor.reviewed_action_record_dispositions",
        "scope": "exact_retained_record_refutation_only", "production_eligible": False,
        "source_files": sources["source_files"], "records": [fact, fact] if fault == "duplicate" else [fact]}
    path = provider.root / "dispositions.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    policy = provider.root / "dispositions.toml"
    policy.write_text((provider.root / "backtest.toml").read_text()
        + f'\n[reviewed_record_dispositions]\npath="{path.name}"\nsha256="{file_sha256(path)}"\n', encoding="utf-8")
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    bridge = BacktestCorporateActions(provider.root, provider.unit_action_evidence, context, memory_guard=lambda: None)
    provider.corporate_actions = bridge
    provider.source_files.update(bridge.source_files)
    return bridge, record


def test_exact_refuted_record_does_not_emit_shares_or_remove_selected_holding(provider: Any) -> None:
    bridge, record = _disposition(provider)
    assert bridge.irrelevance_reason("AAA", action_window("stock_dividends", record), date(2024, 1, 3), date(2024, 1, 17))
    assert not bridge.terms
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons and lot.specification is not None
    assert record["rate"] == 1.3 and len(provider.selected_ids) == 1


def test_other_symbols_action_ids_and_conflicting_gates_are_never_waived(provider: Any) -> None:
    bridge, record = _disposition(provider)
    window = action_window("stock_dividends", record)
    assert bridge.irrelevance_reason("OTHER", window, date(2024, 1, 3), date(2024, 1, 17)) is None
    for identity, reason in [("unreviewed", "unbounded_entitlement_or_due_bill_dates"),
        (record["id"], "conflicting_action_record_identity")]:
        assert bridge.irrelevance_reason("AAA", ActionWindow(identity, "stock_dividends", None, None, reason),
            date(2024, 1, 3), date(2024, 1, 17)) is None


@pytest.mark.parametrize("fault", ["record", "period", "symbol", "family", "duplicate"])
def test_disposition_must_match_the_exact_record_and_audited_period(provider: Any, fault: str) -> None:
    with pytest.raises((ValueError, DataReadinessError)):
        _disposition(provider, fault=fault)
