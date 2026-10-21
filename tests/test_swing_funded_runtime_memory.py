"""Synthetic UNIT snapshots; no allocation, operational mock, or funded result."""
from __future__ import annotations

from pathlib import Path

import pytest

from market_predictor.commands import swing_oof_policy_evaluation as command
from market_predictor.core.errors import MemoryBudgetError
from market_predictor.core.system_memory import SystemMemory
from market_predictor.research import news_runtime_memory as runtime
from market_predictor.research import swing_oof_policy_evaluation as oof


def test_funded_callback_consumers_preserve_package_boundary() -> None:
    from tests.test_package_dependency_boundaries import _forbidden_imports
    package = Path(__file__).resolve().parents[1] / "src/market_predictor/swing/datasets"
    for name in ("action_evidence.py", "action_scope.py", "corrected_outcomes.py", "funded_policy_inputs.py"):
        assert _forbidden_imports(package / name) == []


@pytest.mark.parametrize("available", [145, 129, 99, 91, 90])
@pytest.mark.parametrize("profile", [*oof.CAMPAIGN_PROFILES, "action_projection"])
def test_all_funded_consumers_use_91_percent_without_absolute_floor(
    monkeypatch: pytest.MonkeyPatch, available: int, profile: str,
) -> None:
    calls: list[dict[str, object]] = []
    monkeypatch.setattr(runtime, "assert_memory_budget", lambda **kwargs: calls.append(kwargs))
    monkeypatch.setattr(runtime, "system_memory_snapshot", lambda: SystemMemory(1000, available))
    monkeypatch.setattr(runtime, "MAXIMUM_SYSTEM_PAUSE_SECONDS", 0.0)
    def run() -> None:
        if profile == "action_projection":
            command._action_memory_guard()
        else:
            oof._profile_guard(profile)
    if available > 90:
        run()
    else:
        with pytest.raises(MemoryBudgetError, match="strictly below 91"):
            run()
    assert len(calls) == 1
    assert calls[0]["hard_budget_gib"] == 5.0 and calls[0]["headroom_gib"] == 0.75
