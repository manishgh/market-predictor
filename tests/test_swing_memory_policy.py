"""Percentage-only research policy without allocating the measured memory."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest
from pydantic import ValidationError

from market_predictor.core import system_memory as memory
from market_predictor.core.errors import MemoryBudgetError
from market_predictor.research import swing_training_readiness as owner
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy

GIB = 1024**3
CONFIG = Path(__file__).resolve().parents[1] / "configs/swing_training_readiness.json"


def policy(ceiling: float = 90.0) -> TrainingReadinessPolicy:
    payload = json.loads(CONFIG.read_text())
    payload["maximum_system_used_percent"] = ceiling
    return TrainingReadinessPolicy.model_validate(payload)


def test_checked_in_policy_uses_approved_ceiling() -> None:
    configured = TrainingReadinessPolicy.model_validate_json(CONFIG.read_text())
    assert configured.maximum_system_used_percent == 90.0


@pytest.mark.parametrize(("available", "allowed"), [
    (int(1.25 * GIB), True), (GIB + 1, True), (GIB, False), (GIB - 1, False),
])
def test_percentage_only_boundary(available: int, allowed: bool, monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = memory.SystemMemory(10 * GIB, available)
    monkeypatch.setattr(owner, "system_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: snapshot)
    process = Mock()
    monkeypatch.setattr(owner, "assert_memory_budget", process)
    if allowed:
        owner._guard(policy())
    else:
        with pytest.raises(MemoryBudgetError, match="system memory pressure"):
            owner._guard(policy())
    process.assert_called_once_with(stage="swing training readiness", hard_budget_gib=5.0, headroom_gib=0.75)


@pytest.mark.parametrize("readings", [
    (None, None),
    (memory.SystemMemory(10 * GIB, 2 * GIB), None),
    (memory.SystemMemory(10 * GIB, 2 * GIB), memory.SystemMemory(10 * GIB, GIB)),
    (memory.SystemMemory(10 * GIB, 2 * GIB), memory.SystemMemory(12 * GIB, 3 * GIB)),
])
def test_unstable_or_unknown_measurements_fail(readings: tuple[memory.SystemMemory | None, ...],
    monkeypatch: pytest.MonkeyPatch) -> None:
    measurements = iter(readings)
    monkeypatch.setattr(owner, "system_memory_snapshot", lambda: next(measurements))
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: next(measurements))
    monkeypatch.setattr(owner, "assert_memory_budget", Mock())
    with pytest.raises(MemoryBudgetError):
        owner._guard(policy())


def test_stricter_percentage_can_be_configured(monkeypatch: pytest.MonkeyPatch) -> None:
    snapshot = memory.SystemMemory(10 * GIB, int(1.25 * GIB))
    monkeypatch.setattr(owner, "system_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(memory, "system_memory_snapshot", lambda: snapshot)
    monkeypatch.setattr(owner, "assert_memory_budget", Mock())
    with pytest.raises(MemoryBudgetError):
        owner._guard(policy(85.0))


def test_process_guard_remains_mandatory(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(owner, "assert_memory_budget", Mock(side_effect=MemoryBudgetError("process limit")))
    probe = Mock()
    monkeypatch.setattr(owner, "system_memory_snapshot", probe)
    with pytest.raises(MemoryBudgetError, match="process limit"):
        owner._guard(policy())
    probe.assert_not_called()


@pytest.mark.parametrize("value", [None, "90", True, float("inf"), float("nan"), 0.0, -1.0, 90.1, 100.0])
def test_invalid_percentage_rejected(value: object) -> None:
    payload = json.loads(CONFIG.read_text())
    payload["maximum_system_used_percent"] = value
    with pytest.raises(ValidationError):
        TrainingReadinessPolicy.model_validate(payload)


def test_percentage_must_be_explicit() -> None:
    payload = json.loads(CONFIG.read_text())
    payload.pop("maximum_system_used_percent", None)
    with pytest.raises(ValidationError):
        TrainingReadinessPolicy.model_validate(payload)
