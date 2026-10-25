"""Synthetic UNIT coverage for bounded storage, tamper refusal and ledger parity."""
from __future__ import annotations

import gc
import weakref
from copy import deepcopy
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.research.swing_holding_spool import temporary_holding_spool
from market_predictor.swing.evaluation import ledger
from tests.test_swing_frozen_exit import case as exit_case
from tests.test_swing_oof_policy_evaluation import _available_lot


def test_spool_releases_appended_objects_and_replays_exact_specs_twice(tmp_path: Path) -> None:
    original = exit_case.__wrapped__()["specification"]
    references = []
    expected = []
    calls = 0

    def guard() -> None:
        nonlocal calls
        calls += 1

    with temporary_holding_spool(tmp_path, memory_guard=guard) as spool:
        directory = spool.path.parent
        for index in range(25):
            specification = original.model_copy(update={"decision_id": f"synthetic-{index}"})
            references.append(weakref.ref(specification))
            expected.append(specification.model_dump(mode="json"))
            spool.append(specification)
        del specification
        gc.collect()
        assert all(ref() is None for ref in references)
        spool.seal()
        for _ in range(2):
            assert [spec.model_dump(mode="json") for spec in spool.read()] == expected
        assert calls >= 25 * 3
    assert not directory.exists()


def test_spool_requires_seal_and_cannot_accept_late_append(tmp_path: Path) -> None:
    spec = exit_case.__wrapped__()["specification"]
    with temporary_holding_spool(tmp_path, memory_guard=lambda: None) as spool:
        with pytest.raises(DataReadinessError, match="sealed before"):
            list(spool.read())
        spool.append(spec)
        spool.seal()
        with pytest.raises(DataReadinessError, match="already sealed"):
            spool.append(spec)


@pytest.mark.parametrize("when", ["before_first_pass", "between_passes", "during_pass"])
def test_spool_rejects_changed_bytes_at_each_replay_boundary(tmp_path: Path, when: str) -> None:
    spec = exit_case.__wrapped__()["specification"]
    with temporary_holding_spool(tmp_path, memory_guard=lambda: None) as spool:
        spool.append(spec)
        spool.seal()
        if when == "between_passes":
            assert len(list(spool.read())) == 1
        iterator = spool.read()
        if when == "during_pass":
            assert next(iterator) == spec
        with spool.path.open("ab") as stream:
            stream.write(b"changed")
        with pytest.raises((DataReadinessError, OSError), match="changed|gzip"):
            list(iterator)


def test_spool_cleanup_on_guard_failure_preserves_other_output(tmp_path: Path) -> None:
    protected = tmp_path / "_request.json"
    protected.write_text("protected request", encoding="utf-8")

    def guard() -> None:
        raise DataReadinessError("UNIT pressure refusal")

    with pytest.raises(DataReadinessError, match="pressure refusal"):
        with temporary_holding_spool(tmp_path, memory_guard=guard) as spool:
            directory = spool.path.parent
            spool.append(exit_case.__wrapped__()["specification"])
    assert not directory.exists()
    assert protected.read_text(encoding="utf-8") == "protected request"


@pytest.mark.parametrize("additional_cost", [0.0, 0.002])
def test_compact_settled_steps_equal_full_tail_funding_including_zero_days(
    monkeypatch: pytest.MonkeyPatch, additional_cost: float,
) -> None:
    case = exit_case.__wrapped__()
    decisions = tuple(day.date().isoformat() for day in ledger.xcals.get_calendar("XNYS").sessions_in_range(
        "2024-01-02", "2024-02-13"))
    valuation = ledger.swing_valuation_sessions(decisions, 10)
    decision = {"decision_id": "synthetic-memory-lot", "security_id": "synthetic-issuer", "ticker": "SYN",
        "sector": "synthetic-sector", "session_date_et": case["decision_session"]}
    lot, simulation = _available_lot(decision, valuation)
    assert lot.specification is not None
    selected = pd.DataFrame([{**decision, "session_date_et": decision["session_date_et"].isoformat(),
        "decision_group_id": "synthetic-group"}])
    replay = ledger._replay_funded_entries
    compared = []

    def compare(entries: dict[int, list[dict[str, Any]]], **kwargs: Any) -> dict[str, Any]:
        full = deepcopy(entries)
        for bucket in full.values():
            for item in bucket:
                assert item["steps"][-1]["fully_settled"]
                assert len(item["steps"]) < len(valuation) // 2
                start = valuation.index(lot.specification.initial_entry_timestamp.date().isoformat())
                item["ordinals"] = tuple(pd.Timestamp(day).date().toordinal() for day in valuation[start:])
                zero = {**item["steps"][-1], "cash_before_open": 0.0, "released_cash": 0.0,
                    "sale_proceeds": 0.0, "sale_cash_at_exit": 0.0}
                item["steps"] += tuple(deepcopy(zero) for _ in range(len(item["ordinals"]) - len(item["steps"])))
        compact_result, full_result = replay(entries, **kwargs), replay(full, **kwargs)
        assert compact_result == full_result
        compared.append(compact_result)
        return compact_result

    monkeypatch.setattr(ledger, "_replay_funded_entries", compare)
    result = ledger.build_event_aware_funded_swing_ledger(selected, [lot.specification],
        SimpleNamespace(horizon_sessions=10, expected_round_trip_cost_bps=20.0, maximum_trades_per_decision=50),
        session_calendar=decisions, research_contract_sha256=case["research"].sha256(), execution_policy="managed",
        additional_round_trip_cost=additional_cost, simulation=simulation)
    assert len(compared) == 1
    assert len(result["daily_records"]) == len(valuation)
    assert result["fully_settled"] and result["funded_trades"] == 1
    assert result["daily_records"][-1]["entry_notional"] == 0.0
