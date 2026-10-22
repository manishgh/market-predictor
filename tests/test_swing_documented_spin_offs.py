"""Synthetic UNIT documented daughters; never operational prices or performance."""
from __future__ import annotations

import json
from dataclasses import asdict
from datetime import date
from pathlib import Path
from typing import Any
from urllib.parse import urlencode

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.sources.alpaca_corporate_actions import corporate_action_parameters
from market_predictor.sources.http import HttpByteResponse
from market_predictor.swing.contracts.holding_accounting import (
    CorporateActionEvent,
    ExecutionEvent,
    HoldingSpecification,
    SimulationReference,
)
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.action_windows import ActionWindow
from market_predictor.swing.datasets.backtest_corporate_actions import BacktestCorporateActions, load_backtest_corporate_action_context
from market_predictor.swing.evaluation.holding_accounting import replay_holding
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales
from tests.test_swing_backtest_corporate_actions import _prices
from tests.test_swing_funded_policy_inputs import provider as provider
from tests.test_swing_ordinary_holding import RETRIEVED
from tests.test_swing_reviewed_action_dates import _with_facts


def _http(root: Path, name: str, url: str, params: dict[str, Any], payload: dict[str, Any], state: str) -> tuple[dict[str, str], SourcePin]:
    directory = root / name
    directory.mkdir()
    body = json.dumps(payload).encode()
    (directory / "body.bin").write_bytes(body)
    query = {"url": url, "params": params}
    (directory / "_request.json").write_text(json.dumps(query))
    import hashlib
    response = HttpByteResponse(body=body, requested_url=url + "?" + urlencode(params), final_url=url + "?" + urlencode(params),
        redirect_chain=(), status_code=200, retrieved_at_utc=RETRIEVED, content_type="application/json", content_encoding="identity",
        etag=None, last_modified=None, body_length=len(body), sha256=hashlib.sha256(body).hexdigest(),
        body_representation="http_entity_encoded", safe_headers=())
    metadata = asdict(response)
    metadata.pop("body")
    (directory / "transport.json").write_text(json.dumps(metadata, default=str))
    (directory / "receipt.json").write_text(json.dumps({"state": state, "query_sha256": json_sha256(query),
        "body_path": "body.bin", "body_sha256": response.sha256}))
    pins = {p.relative_to(root).as_posix(): file_sha256(p) for p in directory.iterdir()}
    return pins, SourcePin(path=(directory / "receipt.json").relative_to(root).as_posix(), sha256=file_sha256(directory / "receipt.json"))


def _spin(provider: Any, *, ex: str = "2024-01-04", fault: str | None = None,
    raw_response: bool = False, approve: bool = True) -> BacktestCorporateActions:
    record = {"id": "exact-spin", "source_symbol": "AAA", "source_cusip": "012345678", "source_rate": 1,
        "new_symbol": "CHILD", "new_cusip": "987654321", "new_rate": 0.25, "ex_date": ex}
    _with_facts(provider, "spin_offs", record, {"ex_date": ex, "record_date": "2024-01-02", "payable_date": ex})
    date_artifact = json.loads((provider.root / "reviewed-dates.json").read_text())
    prices = pd.read_parquet(provider.root / "raw/AAA.parquet")
    prices = prices.loc[prices.session_date.between(ex, "2024-01-17")].copy()
    prices["security_id"], prices["ticker"], prices["role"] = "cusip:987654321", "CHILD", "stock"
    for column, value in {"open": 10.0, "high": 13.0, "low": 9.0, "close": 12.0}.items():
        prices[column] = value
    raw_rows = [{"t": row.bar_start_utc.isoformat(), "o": row.open, "h": row.high, "l": row.low, "c": row.close, "v": row.volume}
        for row in prices.itertuples()]
    if fault == "missing_exit_quote":
        prices = prices.loc[prices.session_date.ne("2024-01-05")]
    if fault == "projection":
        prices.loc[prices.session_date.eq("2024-01-05"), "close"] = 11.0
    child = provider.root / "raw/CHILD.parquet"
    prices.to_parquet(child, index=False)
    selection = {"segments": [{"archive": "raw", "first_session": ex, "last_session": "2024-01-17",
        "artifact": {"unit_id": "child-unit", "security_id": "cusip:987654321", "provider_symbol": "CHILD", "ticker": "CHILD",
            "role": "stock", "bars_path": child.name, "bars_sha256": file_sha256(child), "start_date": ex, "end_date": "2024-01-17"}}]}
    selected = provider.root / "child-selection.json"
    selected.write_text(json.dumps(selection))
    pins = {**date_artifact["source_files"], child.relative_to(provider.root).as_posix(): file_sha256(child),
        selected.name: file_sha256(selected)}
    params = corporate_action_parameters(ticker="CHILD", start=date(2019, 7, 9), end=date(2024, 5, 28))
    families = {"spin_offs": [record], "name_changes": [{"id": "child-regular-way", "old_symbol": "CHILD.WI", "new_symbol": "CHILD",
        "old_cusip": "987654321", "new_cusip": "987654321", "process_date": ex}]}
    if fault == "unknown_action":
        families["unknown_action"] = [{"id": "unknown", "symbol": "CHILD"}]
    action_pins, action_receipt = _http(provider.root, "child-actions", "https://data.alpaca.markets/v1/corporate-actions",
        params, {"corporate_actions": families, "next_page_token": None}, "acquired_complete_single_page")
    pins.update(action_pins)
    price_receipt = None
    if raw_response:
        params = {"symbols": "CHILD", "timeframe": "1Day", "start": ex + "T05:00:00+00:00",
            "end": "2024-01-18T04:59:59.999999+00:00", "feed": "sip", "limit": 1000,
            "adjustment": "raw", "sort": "asc", "asof": "2024-01-17"}
        more, price_receipt = _http(provider.root, "child-prices", "https://data.alpaca.markets/v2/stocks/bars", params,
            {"bars": {"CHILD": raw_rows}, "next_page_token": None}, "acquired_exact_nine_sessions")
        pins.update(more)
    fact = {"action_id": record["id"], "retained_record_sha256": json_sha256(record), "source_symbol": "AAA",
        "source_cusip": "012345678", "daughter_symbol": "CHILD", "daughter_cusip": "987654321",
        "daughter_security_id": "cusip:987654321", "ex_date": ex, "daughter_units_per_parent": 0.25,
        "price_selection": {"path": selected.name, "sha256": file_sha256(selected)},
        "price_response_receipt": price_receipt.model_dump(mode="json") if price_receipt else None,
        "action_coverage_receipt": action_receipt.model_dump(mode="json"),
        "official_receipts": date_artifact["records"][0]["official_receipts"]}
    if fault == "ratio":
        fact["daughter_units_per_parent"] = 0.5
    if fault == "record":
        fact["retained_record_sha256"] = "e" * 64
    artifact = {"schema_version": "market_predictor.reviewed_documented_spin_offs",
        "scope": "documented_distribution_and_matched_daughter_sale_only", "production_eligible": False,
        "source_files": pins, "records": [fact, fact] if fault == "duplicate" else [fact]}
    path = provider.root / "spins.json"
    path.write_text(json.dumps(artifact))
    text = (provider.root / "reviewed-backtest.toml").read_text()
    if approve:
        text = text.replace('production_eligible = false',
            'production_eligible = false\ndocumented_spin_offs="proportional_daughter_shares_sold_with_parent"')
    text += f'\n[reviewed_spin_offs]\npath="{path.name}"\nsha256="{file_sha256(path)}"\n'
    policy = provider.root / "spins.toml"
    policy.write_text(text)
    context = load_backtest_corporate_action_context(provider.root, SourcePin(path=policy.name, sha256=file_sha256(policy)))
    bridge = BacktestCorporateActions(provider.root, provider.unit_action_evidence, context, memory_guard=lambda: None)
    provider.corporate_actions = bridge
    provider.source_files.update(bridge.source_files)
    return bridge


@pytest.mark.parametrize("raw_response", [False, True])
def test_source_bound_daughter_shares_sell_once_at_parent_close_and_cost_once(provider: Any, raw_response: bool) -> None:
    _prices(provider, [], target_day="2024-01-05")
    bridge = _spin(provider, raw_response=raw_response)
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons and lot.specification is not None
    sales = [event for event in lot.specification.events if isinstance(event, ExecutionEvent)]
    assert len(sales) == 2 and sales[0].price_per_unit == 106.0 and sales[1].price_per_unit == 12.0
    assert sales[0].effective_at == sales[1].effective_at
    replay = simulate_ordinary_sales(lot.specification, provider.simulation, corporate_simulation=bridge.context)
    assert replay.outcome.snapshots[-1].net_return == pytest.approx(0.06 + 0.25 * 12 / 100 - 0.002)
    assert len(replay.generated_event_ids) == 4 and replay.next_open_settlement.fully_settled
    assert not replay.next_open_settlement.residual_positions
    for event in replay.specification.events:
        if event.event_id.startswith("simulated-sale:"):
            assert all(ref.interpretation_policy_sha256 == provider.simulation.policy.sha256()
                for ref in event.evidence if isinstance(ref, SimulationReference))


def test_open_gap_parent_and_daughter_use_raw_opens(provider: Any) -> None:
    parent = provider.root / "raw/AAA.parquet"
    frame = pd.read_parquet(parent)
    frame.loc[frame.session_date.eq("2024-01-05"), ["open", "low", "close"]] = [95.0, 94.0, 95.0]
    frame.to_parquet(parent, index=False)
    for segment in provider.sources["selection"]["segments"]:
        if segment["artifact"]["ticker"] == "AAA":
            segment["artifact"]["bars_sha256"] = file_sha256(parent)
    _spin(provider)
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    sales = [event for event in lot.specification.events if isinstance(event, ExecutionEvent)]
    assert [sale.price_per_unit for sale in sales] == [95.0, 10.0]
    assert lot.exit_result.exit_clock_basis == "session_open_gap"


def test_pre_distribution_parent_exit_skips_daughter_entitlement(provider: Any) -> None:
    _prices(provider, [], target_day="2024-01-03")
    bridge = _spin(provider)
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    assert len([event for event in lot.specification.events if isinstance(event, ExecutionEvent)]) == 1
    replay = simulate_ordinary_sales(lot.specification, provider.simulation, corporate_simulation=bridge.context)
    assert replay.outcome.snapshots[-1].net_return == pytest.approx(0.06 - 0.002)


def test_new_purchase_on_ex_open_does_not_receive_daughter(provider: Any) -> None:
    _spin(provider, ex="2024-01-03")
    lot = provider.lot(provider.selected_ids[0])
    assert not lot.missing_reasons
    assert not any(isinstance(event, CorporateActionEvent) for event in lot.specification.events)


def test_missing_daughter_exit_quote_remains_unavailable(provider: Any) -> None:
    _prices(provider, [], target_day="2024-01-05")
    _spin(provider, fault="missing_exit_quote")
    lot = provider.lot(provider.selected_ids[0])
    assert lot.specification is None and any("daughter_raw_price_unavailable" in reason for reason in lot.missing_reasons)


@pytest.mark.parametrize("fault", ["ratio", "record", "duplicate", "unknown_action", "projection"])
def test_source_ratio_record_coverage_and_projection_poison_are_rejected(provider: Any, fault: str) -> None:
    with pytest.raises((ValueError, DataReadinessError)):
        _spin(provider, fault=fault, raw_response=True)


def test_special_stock_treatment_requires_its_explicit_policy(provider: Any) -> None:
    with pytest.raises(DataReadinessError, match="approval"):
        _spin(provider, approve=False)


def test_corrupt_daughter_sale_cannot_borrow_cash_policy_or_arbitrary_marks(provider: Any) -> None:
    _prices(provider, [], target_day="2024-01-05")
    bridge = _spin(provider)
    lot = provider.lot(provider.selected_ids[0])
    sale = next(event for event in lot.specification.events if isinstance(event, ExecutionEvent) and event.security_id.startswith("cusip:"))
    corrupt = sale.model_copy(update={"price_per_unit": sale.price_per_unit + 1.0})
    spec = HoldingSpecification.model_validate({**lot.specification.model_dump(),
        "events": tuple(corrupt if event.event_id == sale.event_id else event for event in lot.specification.events)})
    with pytest.raises(ValueError, match="bound raw"):
        replay_holding(spec, corporate_simulation=bridge.context)
    window = ActionWindow("exact-spin", "spin_offs", None, None, "conflicting_action_record_identity")
    assert not bridge.resolves("AAA", window)


def test_duplicate_documented_entitlement_is_rejected(provider: Any) -> None:
    _prices(provider, [], target_day="2024-01-05")
    bridge = _spin(provider)
    spec = provider.lot(provider.selected_ids[0]).specification
    distribution = next(event for event in spec.events if isinstance(event, CorporateActionEvent))
    leg = distribution.legs[0]
    sale = next(event for event in spec.events if isinstance(event, ExecutionEvent) and event.position_id == leg.position_id)
    second_position = leg.position_id + ":duplicate"
    clone = distribution.model_copy(update={"event_id": distribution.event_id + ":duplicate",
        "legs": (leg.model_copy(update={"position_id": second_position}),)})
    second_sale = sale.model_copy(update={"event_id": sale.event_id + ":duplicate", "position_id": second_position,
        "proceeds_id": sale.proceeds_id + ":duplicate", "order": sale.order + 1})
    marks = tuple(mark.model_copy(update={"position_id": second_position}) for mark in spec.marks if mark.position_id == leg.position_id)
    poisoned = HoldingSpecification.model_validate({**spec.model_dump(), "events": (*spec.events, clone, second_sale),
        "marks": (*spec.marks, *marks)})
    with pytest.raises(ValueError, match="entitlement duplicated"):
        replay_holding(poisoned, corporate_simulation=bridge.context)
