"""Synthetic source-bound funded inputs; no retained archive or network access."""
from __future__ import annotations

from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.holding_accounting import ExecutionEvent, PaymentEvent
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import funded_policy_inputs as owner
from market_predictor.swing.datasets.action_windows import ActionWindow
from market_predictor.swing.evaluation.trade_simulation import simulate_ordinary_sales
from market_predictor.swing.labels.frozen_exit import compile_frozen_exit
from tests.test_swing_frozen_exit import case as exit_case
from tests.test_swing_ordinary_holding import CALENDAR, RETRIEVED, inputs


@pytest.mark.parametrize("selection_fails", [False, True])
def test_lazy_selection_and_publication_remain_inside_the_single_source_lease(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, selection_fails: bool,
) -> None:
    order: list[str] = []
    active = False
    pin = SourcePin(path="policy.toml", sha256="a" * 64)
    policy = SimpleNamespace(action_audit_sha256="b" * 64, research_contract=pin, simulation_policy=pin)
    research = SimpleNamespace(assert_strategy_matches=lambda _: None, exit_policies=("target_stop_ten_session_timeout",))
    monkeypatch.setattr(owner, "load_corrected_outcome_policy", lambda *args: policy)
    monkeypatch.setattr(owner, "load_swing_research_contract", lambda *args: research)
    monkeypatch.setattr(owner, "load_trade_simulation_context", lambda *args, **kwargs: None)

    @contextmanager
    def leased_sources(*args: Any) -> Any:
        nonlocal active
        assert not active
        active = True
        order.append("source_lease_enter")
        try:
            yield {"source_files": {}}
        finally:
            order.append("source_lease_exit")
            active = False

    selected = pd.DataFrame({"decision_id": ["exact-selected-id"]})
    calendar = ("2024-01-02",)

    def selection_loader() -> tuple[pd.DataFrame, tuple[str, ...]]:
        assert active
        order.append("load_oof_payload")
        if selection_fails:
            raise DataReadinessError("synthetic selection failure")
        return selected, calendar

    def recheck() -> None:
        assert active
        order.append("source_recheck")

    def provider_factory(**kwargs: Any) -> Any:
        assert active and kwargs["selected"] is selected and kwargs["session_calendar"] == calendar
        return SimpleNamespace(recheck=recheck)

    monkeypatch.setattr(owner, "verified_corrected_research_sources", leased_sources)
    monkeypatch.setattr(owner, "FundedPolicyInputProvider", provider_factory)
    context = owner.verified_funded_policy_inputs(root=tmp_path, target_config=pin,
        evidence=SimpleNamespace(audit_sha256="b" * 64), strategy=SimpleNamespace(),
        selection_loader=selection_loader, exit_policy="target_stop_ten_session_timeout")
    if selection_fails:
        with pytest.raises(DataReadinessError, match="synthetic selection failure"), context:
            pytest.fail("selection failure must not yield a provider")
        assert order == ["source_lease_enter", "load_oof_payload", "source_lease_exit"]
    else:
        with context:
            assert active
            order.append("publish_report")
        assert order == ["source_lease_enter", "load_oof_payload", "source_recheck",
                         "publish_report", "source_recheck", "source_lease_exit"]
    assert not active


@pytest.fixture
def provider(tmp_path: Path) -> Any:
    base = inputs.__wrapped__()
    frozen = exit_case.__wrapped__()
    decision = base["decision"] | {"primary_benchmark": "XLV"}
    all_days = tuple(value.date() for value in CALENDAR.sessions_in_range("2023-12-01", "2024-02-01"))
    archive = tmp_path / "raw"
    archive.mkdir()
    segments = []
    pins = {}
    for identity, ticker, role in (("common:A", "AAA", "stock"), ("benchmark:SPY", "SPY", "benchmark"),
                                   ("benchmark:QQQ", "QQQ", "benchmark"), ("benchmark:XLV", "XLV", "benchmark")):
        path = archive / f"{ticker}.parquet"
        raw = pd.DataFrame([{"security_id": identity, "ticker": ticker, "session_date": str(day),
            "bar_start_utc": pd.Timestamp(day, tz="America/New_York").tz_convert("UTC"),
            "open": 100.0, "high": 101.0, "low": 99.0, "close": 100.0, "volume": 1000.0,
            "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": "raw",
            "ingested_at_utc": RETRIEVED} for day in all_days])
        raw.to_parquet(path, index=False)
        digest = file_sha256(path)
        pins[path.relative_to(tmp_path).as_posix()] = digest
        segments.append({"archive": "raw", "first_session": str(all_days[0]), "last_session": str(all_days[-1]),
            "artifact": {"unit_id": ticker, "security_id": identity, "ticker": ticker, "role": role,
                         "provider_symbol": ticker, "bars_sha256": digest, "bars_path": path.name}})
    policy = SimpleNamespace(source_selection=SourcePin(path="selection.json", sha256="a" * 64),
        decision_corrections=(), official_windows=(), cost_prepaid_fraction=0.002)
    memberships = pd.DataFrame([{"security_id": "common:A", "ticker": "AAA",
        "effective_from_utc": pd.Timestamp("2019-01-01T00:00:00Z"), "effective_to_utc": pd.NaT}])
    result = owner.FundedPolicyInputProvider(root=tmp_path, policy=policy,
        config=SourcePin(path="config.toml", sha256="b" * 64), evidence=SimpleNamespace(recheck=lambda _: None),
        sources={"source_files": pins, "selection": {"segments": segments}, "action_index": ({}, ()),
                 "memberships": memberships}, strategy=frozen["strategy"], research=frozen["research"],
        simulation=base["simulation"], selected=pd.DataFrame([decision]),
        session_calendar=(str(decision["session_date_et"]),), exit_policy="target_stop_ten_session_timeout")
    return result


def test_source_bound_lot_has_raw_atr_and_only_one_settlement(provider: Any) -> None:
    value = provider.lot(provider.selected_ids[0])
    assert value.missing_reasons == () and value.specification is not None
    assert value.decision_atr.value == 2.0
    assert value.decision_atr.price_basis == "raw_with_no_adjustment"
    assert value.exit_result.exit_reason == "timeout"
    assert len(value.specification.events) == 1 and isinstance(value.specification.events[0], ExecutionEvent)
    assert not any(isinstance(event, PaymentEvent) for event in value.specification.events)
    replay = simulate_ordinary_sales(value.specification, provider.simulation)
    assert len([event for event in replay.specification.events if isinstance(event, PaymentEvent)]) == 1
    assert replay.outcome.snapshots[-1].net_return == pytest.approx(-0.002)
    assert replay.next_open_settlement.available_cash == pytest.approx(1.0)
    assert provider.selected_ids == (value.decision_id,)


def test_generated_settlement_reverse_checks_exact_input_hash() -> None:
    result = compile_frozen_exit(**exit_case.__wrapped__())
    clean = owner.unsimulated_managed_specification(result)
    assert json_sha256(clean.model_dump(mode="json")) == result.simulation.input_specification_sha256
    corrupt = result.model_copy(update={"simulation": result.simulation.model_copy(update={"generated_event_ids": ()})})
    with pytest.raises(DataReadinessError, match="reconstruct"):
        owner.unsimulated_managed_specification(corrupt)


@pytest.mark.parametrize("ticker", ["SPY", "QQQ", "XLV"])
def test_benchmark_uses_full_calendar_zero_cost_and_bound_canonical_identity(provider: Any, ticker: str) -> None:
    value = provider.benchmark(ticker)
    spec = value.specification
    assert value.canonical_security_id == f"benchmark:{ticker}"
    assert value.missing_reasons == () and spec.security_id == ticker
    assert spec.cost_prepaid_fraction == 0 and spec.events == ()
    assert tuple(stamp.date().isoformat() for stamp in spec.session_end_timestamps) == provider.valuation_sessions
    assert spec.entry_evidence[0].record_locator.startswith(f"security_id=benchmark:{ticker};")


@pytest.mark.parametrize("stage", ["atr", "holding", "benchmark"])
def test_action_scope_requires_payment_and_residual_evidence_never_filters_selected(provider: Any, stage: str) -> None:
    day = date(2023, 12, 20) if stage == "atr" else date(2024, 1, 5)
    symbol = "SPY" if stage == "benchmark" else "AAA"
    provider.sources["action_index"] = ({symbol: (ActionWindow("dividend-42", "cash_dividends", day, day,
        "payment_and_entitlement_not_admitted"),)}, ())
    result = provider.benchmark("SPY") if stage == "benchmark" else provider.lot(provider.selected_ids[0])
    assert result.specification is None
    assert any("dividend-42" in reason for reason in result.missing_reasons)
    assert any("payment_and_residual_marks_not_admitted" in reason for reason in result.missing_reasons)
    assert len(provider.selected_ids) == 1


def test_original_raw_byte_mutation_cannot_yield_lot_or_pass_final_recheck(provider: Any) -> None:
    path = provider.root / "raw/AAA.parquet"
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="hash differs"):
        provider.lot(provider.selected_ids[0])
    with pytest.raises(DataReadinessError, match="funded source changed"):
        provider.recheck()


def test_terminal_scope_and_missing_selected_outcomes_are_explicit(provider: Any) -> None:
    frame, refs, reasons = provider._read("common:A", "AAA", (date(2024, 5, 29),), benchmark=False)
    assert frame.empty and refs == {} and reasons == ("raw_window_outside_initial_fit",)
    provider.sources["selection"]["segments"] = []
    result = provider.lot(provider.selected_ids[0])
    assert result.specification is None and result.decision_id == provider.selected_ids[0]
    assert result.missing_reasons == ("atr:missing_or_cross_segment_raw_binding",)


@pytest.mark.parametrize("fault", ["adjusted", "clock", "invalid"])
def test_raw_atr_refuses_adjusted_basis_future_clock_or_invalid_bar(provider: Any, fault: str) -> None:
    decision = provider._decisions[provider.selected_ids[0]]
    days = tuple(value.date() for value in CALENDAR.sessions_window(pd.Timestamp(decision["session_date_et"]), -15))
    frame, refs, gaps = provider._read("common:A", "AAA", days, benchmark=False)
    assert not gaps
    if fault == "adjusted":
        frame["adjustment"] = "all"
    elif fault == "clock":
        frame.loc[days[-1], "bar_end_utc"] += pd.Timedelta(days=1)
    else:
        frame.loc[days[-1], "volume"] = 0.0
        assert owner.raw_decision_atr(decision=decision, observations=frame, evidence=refs) is None
        return
    with pytest.raises(DataReadinessError):
        owner.raw_decision_atr(decision=decision, observations=frame, evidence=refs)
