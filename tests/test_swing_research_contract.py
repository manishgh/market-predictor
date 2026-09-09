from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.research import (
    SwingResearchContract,
    assert_unexposed_swing_test,
    load_swing_research_contract,
)

ROOT = Path(__file__).resolve().parents[1]


def test_research_contract_is_explicit_and_preserves_historical_strategy() -> None:
    contract = load_swing_research_contract(ROOT / "configs/swing_research.toml")
    strategy = load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml")
    contract.assert_strategy_matches(strategy)
    assert contract.forecast_target == "future_excess_return_10d_vs_spy"
    assert contract.maximum_model_policy_trials == 12
    assert contract.initial_equity_units == 1.0
    assert contract.binary_auc_role == "diagnostic_only"
    assert contract.price_accounting == "raw_prices_with_explicit_entitlements"
    assert contract.residual_claims == "retain_at_horizon_without_forced_liquidation"
    assert contract.sha256() == load_swing_research_contract(ROOT / "configs/swing_research.toml").sha256()


@pytest.mark.parametrize("field,value", [
    ("horizon_sessions", True), ("horizon_sessions", 10.0),
    ("base_round_trip_cost_bps", float("nan")), ("base_round_trip_cost_bps", True),
    ("maximum_model_policy_trials", 11), ("maximum_model_policy_trials", 13),
    ("managed_benchmark_role", "selection_gate"), ("primary_benchmark", "QQQ"),
    ("maximum_gross_exposure", 1.1), ("historical_test_status", "untouched"),
    ("unrecognized", 1),
    ("schema_version", "market_predictor.swing_long_only_research.v1"),
    ("price_accounting", "verified_total_return_units_no_separate_distributions"),
    ("entitlement_valuation", "use_payout_cap"),
    ("corporate_cash_release", "payable_date"),
    ("residual_claims", "force_cash_at_timeout"),
])
def test_research_contract_rejects_semantic_or_type_drift(field: str, value: object) -> None:
    payload = load_swing_research_contract(ROOT / "configs/swing_research.toml").model_dump(mode="json")
    payload[field] = value
    with pytest.raises((ValidationError, ValueError)):
        SwingResearchContract.model_validate_json(json.dumps(payload))


def test_cost_change_changes_identity_and_requires_strategy_agreement() -> None:
    contract = load_swing_research_contract(ROOT / "configs/swing_research.toml")
    payload = contract.model_dump(mode="json")
    payload["base_round_trip_cost_bps"] = 30.0
    changed = SwingResearchContract.model_validate_json(json.dumps(payload))
    assert changed.sha256() != contract.sha256()
    with pytest.raises(DataReadinessError, match="differ"):
        changed.assert_strategy_matches(load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"))


@pytest.mark.parametrize("sessions", [
    ["2025-07-01"], ["2026-06-30"], ["2025-06-30", "2025-07-01"],
])
def test_known_exposed_outcomes_cannot_be_reopened_as_protected(sessions: list[str]) -> None:
    with pytest.raises(DataReadinessError, match="already exposed"):
        assert_unexposed_swing_test(sessions)


@pytest.mark.parametrize("sessions", [[], ["20260908"], ["2026-09-08", "2026-09-08"], ["2026-09-09", "2026-09-08"]])
def test_protected_sessions_must_be_explicit_canonical_unique_and_ordered(sessions: list[str]) -> None:
    with pytest.raises(DataReadinessError):
        assert_unexposed_swing_test(sessions)


def test_non_exposed_dates_do_not_claim_approval_or_freshness() -> None:
    assert assert_unexposed_swing_test(["2026-09-08", "2026-09-09"]) is None


def test_research_contract_rejects_duplicate_toml_fields(tmp_path: Path) -> None:
    path = tmp_path / "invalid.toml"
    path.write_text('side = "long_only"\nside = "short"\n')
    with pytest.raises(DataReadinessError):
        load_swing_research_contract(path)


def test_retained_trainer_refuses_exposed_interval_before_loading_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    from market_predictor.edge_rebuild import swing_training

    def forbidden_load(*args: object, **kwargs: object) -> None:
        pytest.fail("training must reject known exposed test dates before loading any panel")

    monkeypatch.setattr(swing_training, "load_swing_panel_binding", forbidden_load)
    monkeypatch.setattr(swing_training, "_guard", lambda *args, **kwargs: None)
    with pytest.raises(DataReadinessError, match="already exposed"):
        swing_training.train_swing_edge_candidate(
            tmp_path / "not-read", tmp_path / "not-written",
            strategy_contract=load_strategy_contract(ROOT / "configs/edge_rebuild_strategy_contract.toml"),
            config=swing_training.load_swing_training_config(ROOT / "configs/edge_rebuild_swing_training.toml"),
            temporal_policy_path=ROOT / "configs/edge_rebuild_temporal_manifest.toml",
        )
    assert not (tmp_path / "not-written").exists()


def test_statistical_procedure_is_frozen_before_results() -> None:
    contract = load_swing_research_contract(ROOT / "configs/swing_research.toml")
    assert contract.familywise_lower_tail_probability == pytest.approx(0.05 / 12)
    assert contract.bootstrap_samples == 20_000
    assert contract.prospective_minimum_sessions == 252
    assert contract.prospective_session_unit == "decision_sessions"
    assert contract.terminal_policy == "ten_session_maturation_tail_no_new_entries"
