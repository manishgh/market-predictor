"""Tiny synthetic UNIT prices/actions; never operational data or performance."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import ExecutionEvent
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.backtest_corporate_actions import BacktestCorporateActions, load_backtest_corporate_action_context
from market_predictor.swing.datasets.corrected_outcome_admission import reconcile_actions
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from tests.test_swing_funded_policy_inputs import provider as provider

REPO = Path(__file__).resolve().parents[1]


def _bridge(provider: Any, records: dict[str, list[dict[str, Any]]]) -> BacktestCorporateActions:
    policy = provider.root / "backtest.toml"
    policy.write_bytes((REPO / "configs/swing_backtest_corporate_actions.toml").read_bytes())
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    by_symbol: dict[str, dict[str, list[dict[str, Any]]]] = {}
    for family, values in records.items():
        for record in values:
            symbol = next(record[key] for key in ("symbol", "source_symbol", "old_symbol", "acquirer_symbol") if record.get(key))
            by_symbol.setdefault(symbol, {}).setdefault(family, []).append(record)
    files: dict[str, str] = {}
    for symbol, families in by_symbol.items():
        path = provider.root / f"actions/tickers/{symbol}/selected/000.bin"
        path.parent.mkdir(parents=True)
        path.write_text(json.dumps({"corporate_actions": families}), encoding="utf-8")
        files[path.relative_to(provider.root).as_posix()] = file_sha256(path)
        metadata = path.with_suffix(".json")
        metadata.write_text(json.dumps({"sha256": file_sha256(path), "body_length": path.stat().st_size,
            "retrieved_at_utc": "2026-01-01T00:00:00+00:00"}), encoding="utf-8")
        files[metadata.relative_to(provider.root).as_posix()] = file_sha256(metadata)
        receipt = path.parent / "receipt.json"
        receipt.write_text(json.dumps({"request_sha256": "b" * 64, "state": "acquired", "ticker": symbol,
            "pages": [{"body_path": path.name, "body_sha256": file_sha256(path),
                "metadata_path": metadata.name, "metadata_sha256": file_sha256(metadata)}]}), encoding="utf-8")
        files[receipt.relative_to(provider.root).as_posix()] = file_sha256(receipt)
    projection = {"records": by_symbol, "unavailable": [], "source_files": files, "replay": {}}
    evidence = CorporateActionEvidence("a" * 64, "b" * 64, by_symbol, (), files, {}, json_sha256(projection))
    provider.unit_action_evidence = evidence
    result = BacktestCorporateActions(provider.root, evidence, context, memory_guard=lambda: None)
    provider.corporate_actions = result
    provider.source_files.update(result.source_files)
    provider.sources["action_index"] = reconcile_actions(evidence, set(by_symbol))
    return result


def _cash(day: str, symbol: str = "AAA", identity: str = "dividend") -> dict[str, Any]:
    return {"id": identity, "symbol": symbol, "ex_date": day, "rate": 1.0, "special": False, "foreign": False}


def _split(day: str, ratio: float, identity: str = "split") -> dict[str, Any]:
    return {"id": identity, "symbol": "AAA", "ex_date": day, "new_rate": ratio, "old_rate": 1.0}


def _prices(provider: Any, changes: list[tuple[str, float]], *, target_day: str | None = None) -> None:
    path = provider.root / "raw/AAA.parquet"
    frame = pd.read_parquet(path)
    for day, factor in changes:
        mask = frame.session_date >= day
        for column in ("open", "high", "low", "close"):
            frame.loc[mask, column] /= factor
    if target_day:
        mask = frame.session_date.eq(target_day)
        frame.loc[mask, "high"] = frame.loc[mask, "open"] * 1.1
    frame.to_parquet(path, index=False)
    digest = file_sha256(path)
    provider.source_files[path.relative_to(provider.root).as_posix()] = digest
    for segment in provider.sources["selection"]["segments"]:
        if segment["artifact"]["provider_symbol"] == "AAA":
            segment["artifact"]["bars_sha256"] = digest


def test_benchmark_dividend_is_simulated_cash_and_keeps_original_shares(provider: Any) -> None:
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05", "SPY")]})
    lot = provider.benchmark("SPY")
    assert not lot.missing_reasons and lot.specification is not None
    outcome = replay_holding(lot.specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.01)
    assert len(outcome.cash_releases) == 1 and outcome.cash_releases[0].basis == "research_assumption"
    assert outcome.cash_releases[0].available_at.date() == date(2024, 1, 5)
    assert outcome.snapshots[-1].tradable_value == pytest.approx(1.0)
    with pytest.raises(ValueError, match="ordinary-sale assumptions"):
        replay_holding(lot.specification)


def test_purchase_at_ex_date_open_does_not_earn_that_dividend(provider: Any) -> None:
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-03", "SPY")]})
    lot = provider.benchmark("SPY")
    assert lot.specification is not None and not lot.specification.events
    assert replay_holding(lot.specification, corporate_simulation=bridge.context).snapshots[-1].total_value == 1.0


@pytest.mark.parametrize("sale_day,earned", [("2024-01-04", False), ("2024-01-05", True)])
def test_sale_before_or_on_ex_date_keeps_exact_dividend_entitlement(provider: Any, sale_day: str, earned: bool) -> None:
    _prices(provider, [], target_day=sale_day)
    _bridge(provider, {"cash_dividends": [_cash("2024-01-05")]})
    result = provider.lot(provider.selected_ids[0])
    assert not result.missing_reasons and result.exit_result is not None
    assert result.exit_result.exit_timestamp.date().isoformat() == sale_day
    outcome = result.exit_result.simulation.outcome
    corporate = [release for release in outcome.cash_releases if release.source_kind == "corporate_payment"]
    assert len(corporate) == int(earned)
    expected = 3 * result.decision_atr.value / 100 + (0.01 if earned else 0.0) - 0.002
    assert outcome.snapshots[-1].net_return == pytest.approx(expected)


@pytest.mark.parametrize("splits", [[("2024-01-05", 2.0)], [("2024-01-05", 0.5)],
    [("2024-01-05", 2.0), ("2024-01-09", 0.5)]])
def test_splits_preserve_economic_value_and_exit_current_owned_class(provider: Any, splits: list[tuple[str, float]]) -> None:
    _prices(provider, splits)
    records: dict[str, list[dict[str, Any]]] = {"forward_splits": [], "reverse_splits": []}
    for number, (day, ratio) in enumerate(splits):
        records["forward_splits" if ratio > 1 else "reverse_splits"].append(_split(day, ratio, str(number)))
    _bridge(provider, records)
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons and lot.exit_result is not None
    assert lot.exit_result.exit_reason == "timeout"
    assert lot.exit_result.simulation.outcome.snapshots[-1].net_return == pytest.approx(-0.002)
    sales = [event for event in lot.specification.events if isinstance(event, ExecutionEvent)]
    assert len(sales) == 1 and "backtest-split:" in sales[0].position_id


def test_split_between_decision_and_entry_converts_volatility_to_entry_share_units(provider: Any) -> None:
    _prices(provider, [("2024-01-03", 2.0)])
    _bridge(provider, {"forward_splits": [_split("2024-01-03", 2.0)]})
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons and lot.exit_result.exit_reason == "timeout"
    assert lot.decision_atr.value == pytest.approx(2.0)
    assert lot.exit_result.stop_price == pytest.approx(48.5)
    assert lot.exit_result.simulation.outcome.snapshots[-1].net_return == pytest.approx(-0.002)


def test_historical_split_normalizes_raw_atr_without_rewriting_source_prices(provider: Any) -> None:
    _prices(provider, [("2023-12-20", 2.0)])
    _bridge(provider, {"forward_splits": [_split("2023-12-20", 2.0)]})
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    assert lot.decision_atr.value == pytest.approx(1.0)
    assert pd.read_parquet(provider.root / "raw/AAA.parquet").iloc[0].close == 100.0


@pytest.mark.parametrize("ratio", [2.0, 0.5])
def test_split_day_profit_target_uses_new_share_price_and_quantity(provider: Any, ratio: float) -> None:
    _prices(provider, [("2024-01-05", ratio)], target_day="2024-01-05")
    family = "forward_splits" if ratio > 1 else "reverse_splits"
    _bridge(provider, {family: [_split("2024-01-05", ratio)]})
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons and lot.exit_result.exit_reason == "target"
    assert lot.exit_result.exit_timestamp.date() == date(2024, 1, 5)
    assert lot.exit_result.simulation.outcome.snapshots[-1].net_return == pytest.approx(0.058)


def test_multiple_benchmark_dividends_keep_cash_without_reinvestment(provider: Any) -> None:
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05", "SPY", "one"), _cash("2024-01-10", "SPY", "two")]})
    lot = provider.benchmark("SPY")
    outcome = replay_holding(lot.specification, corporate_simulation=bridge.context)
    assert outcome.snapshots[-1].total_value == pytest.approx(1.02)
    assert outcome.snapshots[-1].tradable_value == pytest.approx(1.0)


def test_special_dividends_and_conflicting_splits_remain_unavailable(provider: Any) -> None:
    special = _cash("2024-01-05") | {"special": True}
    bridge = _bridge(provider, {"cash_dividends": [special], "forward_splits": [
        _split("2024-01-08", 2.0), _split("2024-01-08", 3.0)]})
    assert not bridge.terms
    result = provider.lot(provider.selected_ids[0])
    assert result.specification is None and result.missing_reasons
    assert len(provider.selected_ids) == 1


def test_action_body_mutation_stops_final_source_recheck(provider: Any) -> None:
    _bridge(provider, {"cash_dividends": [_cash("2024-01-05")]})
    (provider.root / "actions/tickers/AAA/selected/000.bin").write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="source changed"):
        provider.recheck()


def test_action_free_prices_exits_and_cash_remain_identical(provider: Any) -> None:
    original = provider.lot(provider.selected_ids[0])
    original_benchmark = provider.benchmark("SPY")
    _bridge(provider, {})
    assert provider.lot(provider.selected_ids[0]) == original
    assert provider.benchmark("SPY") == original_benchmark


def test_same_day_split_and_dividend_require_evidenced_rate_basis(provider: Any) -> None:
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05")],
        "forward_splits": [_split("2024-01-05", 2.0)]})
    assert not bridge.terms
    assert provider.lot(provider.selected_ids[0]).missing_reasons


@pytest.mark.parametrize("poison", ["missing_context", "early_cash", "assumption_policy"])
def test_assumed_cash_cannot_escape_exact_research_policy_or_time(provider: Any, poison: str) -> None:
    from market_predictor.swing.contracts.holding_accounting import CashAvailabilityEvent
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05", "SPY")]})
    spec = provider.benchmark("SPY").specification
    if poison == "missing_context":
        with pytest.raises(ValueError):
            replay_holding(spec)
        return
    if poison == "early_cash":
        events = tuple(event.model_copy(update={"effective_at": spec.initial_entry_timestamp})
            if isinstance(event, CashAvailabilityEvent) else event for event in spec.events)
        spec = spec.model_copy(update={"events": events})
    else:
        bridge.context = bridge.context.model_copy(update={"policy_reference":
            bridge.context.policy_reference.model_copy(update={"interpretation_policy_sha256": "0" * 64})})
    with pytest.raises(ValueError):
        replay_holding(spec, corporate_simulation=bridge.context)


def test_funded_ledger_reconciles_dividend_cash_and_unchanged_base_stress_costs(provider: Any) -> None:
    from types import SimpleNamespace

    from market_predictor.swing.evaluation.ledger import build_event_aware_funded_swing_ledger
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05")]})
    lot = provider.lot(provider.selected_ids[0])
    row = dict(provider._decisions[provider.selected_ids[0]])
    row["decision_group_id"] = row["session_date_et"].isoformat()
    row["session_date_et"] = row["session_date_et"].isoformat()
    arguments = dict(selected=pd.DataFrame([row]), holdings=[lot.specification],
        config=SimpleNamespace(horizon_sessions=10, expected_round_trip_cost_bps=20.0, maximum_trades_per_decision=25),
        session_calendar=(row["session_date_et"],), research_contract_sha256=provider.research.sha256(),
        execution_policy="managed", simulation=provider.simulation, corporate_simulation=bridge.context)
    base = build_event_aware_funded_swing_ledger(**arguments)
    stress = build_event_aware_funded_swing_ledger(**arguments, additional_round_trip_cost=0.002)
    assert base["compounded_return"] == pytest.approx(0.1 * (0.01 - 0.002))
    assert stress["compounded_return"] == pytest.approx(0.1 * (0.01 - 0.004))


def test_failed_historical_page_cannot_create_a_dividend(provider: Any) -> None:
    from dataclasses import replace
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05", "SPY")]})
    evidence = provider.unit_action_evidence
    files = dict(evidence.source_files)
    receipt_name = next(name for name in files if name.endswith("receipt.json"))
    receipt = provider.root / receipt_name
    value = json.loads(receipt.read_text()) | {"state": "failed"}
    receipt.write_text(json.dumps(value))
    files[receipt_name] = file_sha256(receipt)
    # Historical body is fully pinned; successful projection contains no records.
    clean = replace(evidence, records_by_symbol={}, source_files=files)
    repaired = BacktestCorporateActions(provider.root, clean, bridge.context, memory_guard=lambda: None)
    assert not repaired.terms
    provider.corporate_actions = repaired
    provider.sources["action_index"] = ({}, ())
    spec = provider.benchmark("SPY").specification
    assert not spec.events


def test_successful_page_must_match_projected_record_hash(provider: Any) -> None:
    from dataclasses import replace
    bridge = _bridge(provider, {"cash_dividends": [_cash("2024-01-05", "SPY")]})
    evidence = replace(provider.unit_action_evidence, records_by_symbol={"SPY": {"cash_dividends": []}})
    with pytest.raises(DataReadinessError, match="absent from successful projection"):
        BacktestCorporateActions(provider.root, evidence, bridge.context, memory_guard=lambda: None)


@pytest.mark.parametrize("sale_context", [None, "f" * 64])
def test_dividend_assumption_cannot_authorize_a_sale_proceeds_mark(provider: Any, sale_context: str | None) -> None:
    from market_predictor.swing.contracts.holding_accounting import KnownMark, SimulationReference
    bridge = _bridge(provider, {})
    spec = provider.lot(provider.selected_ids[0]).specification
    sale = next(event for event in spec.events if isinstance(event, ExecutionEvent))
    assumption = SimulationReference(**bridge.context.policy_reference.model_dump(), basis="research_assumption")
    mark = KnownMark(position_id=sale.proceeds_id, mark_at=spec.session_end_timestamps[-1],
        value_per_unit=1.0, currency="USD", evidence=(assumption,))
    poisoned = spec.model_copy(update={"marks": (*spec.marks, mark)})
    with pytest.raises(ValueError, match="simulated marks"):
        replay_holding(poisoned, simulation_policy_sha256=sale_context, corporate_simulation=bridge.context)
