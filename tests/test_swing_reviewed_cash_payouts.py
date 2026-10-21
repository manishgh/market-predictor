"""Synthetic UNIT cash-entitlement groups; never operational accounting evidence."""
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
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from tests.test_swing_backtest_corporate_actions import _bridge, _cash, _prices
from tests.test_swing_funded_policy_inputs import provider as provider


def _reviewed(provider: Any, records: list[dict[str, Any]], components: list[tuple[str, float]],
    *, approve: bool = True, fault: str | None = None) -> BacktestCorporateActions:
    records = [row | {"cusip": "012345678"} for row in records]
    _bridge(provider, {"cash_dividends": records})
    directory = provider.root / "official/cash"
    directory.mkdir(parents=True)
    body = directory / "body.bin"
    body.write_bytes(b"UNIT issuer declares exact cash components, no real source data")
    request = directory / "_request.json"
    query = {"url": "https://issuer.example/cash"}
    request.write_text(json.dumps(query), encoding="utf-8")
    transport = directory / "transport.json"
    transport.write_text(json.dumps({"status_code": 200, "redirect_chain": [], "requested_url": query["url"],
        "final_url": query["url"], "sha256": file_sha256(body), "body_length": body.stat().st_size}), encoding="utf-8")
    receipt = directory / "receipt.json"
    receipt.write_text(json.dumps({"state": "archived_unreviewed", "status_code": 200,
        "url": query["url"], "request_sha256": json_sha256(query), "body_path": body.name,
        "body_sha256": file_sha256(body), "transport_sha256": file_sha256(transport)}), encoding="utf-8")
    pins = {p.relative_to(provider.root).as_posix(): file_sha256(p) for p in [body, request, transport, receipt]}
    fact = {"entitlement_id": "UNIT:one-declared-entitlement", "symbol": records[0]["symbol"],
        "ex_date": records[0]["ex_date"], "currency": "USD", "amount_basis": "issuer_declared_gross",
        "components": [{"kind": kind, "declared_rate_usd": rate} for kind, rate in components],
        "provider_records": [{"action_id": row["id"], "record_sha256": json_sha256(row)} for row in records],
        "official_receipts": [{"path": receipt.relative_to(provider.root).as_posix(), "sha256": file_sha256(receipt)}]}
    if fault == "partial_group":
        fact["provider_records"] = fact["provider_records"][:1]
    elif fault == "changed_record":
        fact["provider_records"][0]["record_sha256"] = "e" * 64
    elif fault == "currency":
        fact["currency"] = "EUR"
    artifact = {"schema_version": "market_predictor.reviewed_backtest_cash_payouts",
        "scope": "retrospective_declared_cash_only", "production_eligible": False,
        "source_files": pins, "entitlements": [fact, fact] if fault == "duplicate" else [fact]}
    path = provider.root / "reviewed-cash.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")
    policy = provider.root / "reviewed-cash.toml"
    policy.write_text((provider.root / "backtest.toml").read_text(encoding="utf-8")
        + ('\ndocumented_cash_payouts="issuer_declared_usd_at_ex_date_close"\n' if approve else "\n")
        + f'\n[reviewed_cash_payouts]\npath="{path.name}"\nsha256="{file_sha256(path)}"\n', encoding="utf-8")
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    bridge = BacktestCorporateActions(provider.root, provider.unit_action_evidence, context, memory_guard=lambda: None)
    provider.corporate_actions = bridge
    provider.source_files.update(bridge.source_files)
    return bridge


def test_distinct_regular_and_supplemental_legs_both_count_once(provider: Any) -> None:
    records = [_cash("2024-01-05", identity="regular") | {"rate": 0.15},
        _cash("2024-01-05", identity="supplemental") | {"rate": 0.65}]
    bridge = _reviewed(provider, records, [("ordinary", 0.15), ("supplemental", 0.65)])
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    outcome = lot.exit_result.simulation.outcome
    releases = [item for item in outcome.cash_releases if item.source_kind == "corporate_payment"]
    assert len(releases) == 1 and releases[0].available_at.date() == date(2024, 1, 5)
    assert outcome.snapshots[-1].net_return == pytest.approx(0.008 - 0.002)
    assert len(bridge.terms) == 1


def test_duplicate_provider_rows_make_one_issuer_entitlement(provider: Any) -> None:
    records = [_cash("2024-01-05", "QQQ", "duplicate-a") | {"payable_date": "2024-01-10", "rate": 0.5},
        _cash("2024-01-05", "QQQ", "duplicate-b") | {"payable_date": "2024-01-12", "rate": 0.5}]
    bridge = _reviewed(provider, records, [("ordinary", 0.5)])
    lot = provider.benchmark("QQQ")
    assert not lot.missing_reasons
    outcome = replay_holding(lot.specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.005)
    assert outcome.snapshots[-1].tradable_value == 1.0
    assert len(outcome.cash_releases) == 1


def test_combined_provider_total_is_not_added_to_its_components(provider: Any) -> None:
    bridge = _reviewed(provider, [_cash("2024-01-05", "QQQ") | {"rate": 0.18, "special": True}],
        [("ordinary", 0.10), ("special", 0.08)])
    lot = provider.benchmark("QQQ")
    outcome = replay_holding(lot.specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.0018)
    assert len(outcome.cash_releases) == 1


def test_foreign_flagged_net_record_uses_reviewed_gross_usd_without_invented_tax(provider: Any) -> None:
    bridge = _reviewed(provider, [_cash("2024-01-05", "QQQ") | {"rate": 0.84, "foreign": True}], [("ordinary", 1.12)])
    lot = provider.benchmark("QQQ")
    outcome = replay_holding(lot.specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.0112)
    assert len(outcome.cash_releases) == 1 and outcome.cash_releases[0].basis == "research_assumption"


def test_special_cash_requires_explicit_extended_timing_policy(provider: Any) -> None:
    with pytest.raises(DataReadinessError, match="explicit approval"):
        _reviewed(provider, [_cash("2024-01-05") | {"special": True}], [("special", 1.0)], approve=False)


def test_supplemental_with_false_provider_special_flag_still_requires_approval(provider: Any) -> None:
    with pytest.raises(DataReadinessError, match="explicit approval"):
        _reviewed(provider, [_cash("2024-01-05")], [("supplemental", 1.0)], approve=False)


def test_declared_gross_ordinary_cash_does_not_require_special_policy_extension(provider: Any) -> None:
    bridge = _reviewed(provider, [_cash("2024-01-05", "QQQ") | {"foreign": True, "rate": 0.75}],
        [("ordinary", 1.0)], approve=False)
    outcome = replay_holding(provider.benchmark("QQQ").specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.01)


@pytest.mark.parametrize("sale_day,earned", [("2024-01-04", False), ("2024-01-05", True)])
def test_reviewed_special_entitlement_survives_ex_day_sale_only_when_owned_before_ex(provider: Any, sale_day: str, earned: bool) -> None:
    _prices(provider, [], target_day=sale_day)
    _reviewed(provider, [_cash("2024-01-05") | {"special": True}], [("special", 1.0)])
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    outcome = lot.exit_result.simulation.outcome
    assert len([item for item in outcome.cash_releases if item.source_kind == "corporate_payment"]) == int(earned)


def test_purchase_at_ex_open_does_not_earn_reviewed_special(provider: Any) -> None:
    bridge = _reviewed(provider, [_cash("2024-01-03", "QQQ") | {"special": True}], [("special", 1.0)])
    lot = provider.benchmark("QQQ")
    assert not lot.missing_reasons and not lot.specification.events
    assert replay_holding(lot.specification, corporate_simulation=bridge.context).snapshots[-1].total_value == 1.0


@pytest.mark.parametrize("fault", ["partial_group", "changed_record", "duplicate", "currency"])
def test_incomplete_changed_duplicate_or_non_usd_facts_refuse_cash(provider: Any, fault: str) -> None:
    records = [_cash("2024-01-05", identity="one"), _cash("2024-01-05", identity="two")]
    with pytest.raises((ValueError, DataReadinessError)):
        _reviewed(provider, records, [("ordinary", 1.0)], fault=fault)


def test_exact_reviewed_old_scope_can_resolve_cash_but_invalid_record_gate_remains(provider: Any) -> None:
    record = _cash("2024-01-05") | {"special": True}
    bridge = _reviewed(provider, [record], [("special", 1.0)])
    assert bridge.resolves("AAA", ActionWindow(record["id"], "cash_dividends", date(2024, 1, 5), date(2024, 1, 5),
        "reviewed_cash_distribution_resolution_unavailable", True))
    assert not bridge.resolves("AAA", ActionWindow(record["id"], "cash_dividends", None, None, "conflicting_action_record_identity"))
    assert not bridge.resolves("OTHER", action_window("cash_dividends", record))
    assert not bridge.resolves("AAA", ActionWindow(record["id"], "cash_dividends", date(2025, 1, 6), date(2025, 1, 6),
        "corporate_terms_require_replay", True))
