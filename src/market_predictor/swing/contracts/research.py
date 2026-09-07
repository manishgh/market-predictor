"""Frozen scope and evidence rules for long-only, SPY-relative swing research."""

from __future__ import annotations

import json
import tomllib
from collections.abc import Sequence
from datetime import date
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.modeling.strategy_contract import StrategyContract

EXPOSED_TEST_START = date(2025, 7, 1)
EXPOSED_TEST_END = date(2026, 6, 30)
EXPOSED_TEST_EVIDENCE = "data/models/swing/technical/evaluation.json"


class SwingResearchContract(BaseModel):
    """Research policy, distinct from immutable historical feature/label lineage."""

    model_config = ConfigDict(extra="forbid", frozen=True, strict=True, allow_inf_nan=False)

    schema_version: Literal["market_predictor.swing_long_only_research.v1"]
    side: Literal["long_only"]
    primary_benchmark: Literal["SPY"]
    comparison_benchmarks: tuple[Literal["QQQ", "point_in_time_sector_etf"], ...]
    decision_start: Literal["2019-07-09"]
    horizon_sessions: int = Field(ge=10, le=10)
    entry: Literal["next_session_open"]
    forecast_target: Literal["future_excess_return_10d_vs_spy"]
    target_units: Literal["decimal_net_return_difference"]
    economic_target: Literal["funded_daily_nav_excess_vs_buy_and_hold_spy"]
    managed_benchmark_role: Literal["approximate_diagnostic_only"]
    binary_auc_role: Literal["diagnostic_only"]
    learner_families: tuple[str, ...]
    feature_profiles: tuple[str, ...]
    exit_policies: tuple[str, ...]
    maximum_learned_specifications: int = Field(ge=1, le=6)
    maximum_model_policy_trials: int = Field(ge=1, le=12)
    base_round_trip_cost_bps: float = Field(gt=0, le=100)
    stress_cost_multiplier: float = Field(ge=2, le=2)
    cost_timing: Literal["round_trip_prepaid_at_entry"]
    price_accounting: Literal["verified_total_return_units_no_separate_distributions"]
    initial_equity_units: float = Field(ge=1, le=1)
    cash_yield: float = Field(ge=0, le=0)
    maximum_gross_exposure: float = Field(ge=1, le=1)
    cohort_allocation: Literal["one_tenth_prior_close_nav_equal_weight"]
    funding_policy: Literal["pro_rata_cash_cap_including_prepaid_cost"]
    entry_exit_order: Literal["entries_before_exits_proceeds_next_session"]
    repeated_security_policy: Literal["separate_exit_lots_aggregate_exposure"]
    maximum_drawdown: float = Field(gt=0, le=1)
    capacity_claim: Literal["unavailable_without_account_size_and_execution_evidence"]
    bootstrap_samples: int = Field(ge=2000, le=100000)
    bootstrap_block_sessions: int = Field(ge=20, le=20)
    sensitivity_block_sessions: int = Field(ge=40, le=40)
    confidence_level: float = Field(ge=0.95, le=0.95)
    confidence_sidedness: Literal["one_sided_lower"]
    bootstrap_method: Literal["moving_block_percentile"]
    tested_statistic: Literal["mean_daily_portfolio_return_minus_spy_return"]
    multiple_trial_control: Literal["bonferroni_model_policy_family"]
    sensitivity_role: Literal["conjunctive_not_alternative_pass"]
    random_seed: int = Field(ge=0)
    prospective_minimum_sessions: int = Field(ge=252)
    prospective_session_unit: Literal["decision_sessions"]
    terminal_policy: Literal["ten_session_maturation_tail_no_new_entries"]
    prospective_assessments: int = Field(ge=1, le=1)
    historical_test_status: Literal["already_exposed_research_only"]
    maximum_process_memory_gib: float = Field(gt=0, le=5)

    @model_validator(mode="after")
    def validate_frozen_comparisons(self) -> Self:
        if self.comparison_benchmarks != ("QQQ", "point_in_time_sector_etf"):
            raise ValueError("QQQ and the sector benchmark remain required diagnostics")
        if self.learner_families != ("regularized_linear_return", "shallow_boosted_return"):
            raise ValueError("the research campaign has exactly two return learner families")
        if self.feature_profiles != (
            "existing_technical", "technical_relationships", "technical_relationships_issuer_reaction"
        ):
            raise ValueError("the three feature comparisons must remain explicit and ordered")
        if self.exit_policies != ("target_stop_ten_session_timeout", "stop_ten_session_timeout"):
            raise ValueError("only the control and uncapped-upside research exit policies are admitted")
        count = len(self.learner_families) * len(self.feature_profiles)
        if count > self.maximum_learned_specifications:
            raise ValueError("learned specification budget is smaller than the frozen comparisons")
        if count * len(self.exit_policies) > self.maximum_model_policy_trials:
            raise ValueError("model-policy comparisons exceed the frozen trial budget")
        return self

    def sha256(self) -> str:
        return json_sha256(self.model_dump(mode="json"))

    @property
    def familywise_lower_tail_probability(self) -> float:
        return (1.0 - self.confidence_level) / self.maximum_model_policy_trials

    def assert_strategy_matches(self, strategy: StrategyContract) -> None:
        if (
            strategy.swing.horizon_sessions != self.horizon_sessions
            or strategy.swing.round_trip_cost_bps != self.base_round_trip_cost_bps
            or strategy.stress.cost_multiplier != self.stress_cost_multiplier
            or strategy.swing.entry_reference != self.entry
        ):
            raise DataReadinessError("swing research costs, horizon or entry differ from historical strategy")


def load_swing_research_contract(path: Path) -> SwingResearchContract:
    """Parse strictly; unknown fields, non-finite values and implicit defaults fail."""
    try:
        payload = tomllib.loads(path.read_text(encoding="utf-8"))
        return SwingResearchContract.model_validate_json(json.dumps(payload, allow_nan=False))
    except (OSError, UnicodeError, ValueError, TypeError) as exc:
        raise DataReadinessError(f"invalid swing research contract: {path}: {exc}") from exc


def assert_unexposed_swing_test(sessions: Sequence[str]) -> None:
    """Forbid known exposed outcomes from being reopened as a protected final test."""
    if not sessions:
        raise DataReadinessError("protected swing evaluation requires explicit sessions")
    try:
        parsed = tuple(date.fromisoformat(value) for value in sessions)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("protected swing sessions must be ISO dates") from exc
    if tuple(value.isoformat() for value in parsed) != tuple(sessions) or tuple(sorted(set(parsed))) != parsed:
        raise DataReadinessError("protected swing sessions must be canonical, unique and ordered")
    if any(EXPOSED_TEST_START <= value <= EXPOSED_TEST_END for value in parsed):
        raise DataReadinessError(
            "July 2025-June 2026 swing outcomes were already exposed by "
            f"{EXPOSED_TEST_EVIDENCE}; they cannot be a new protected test. "
            "Use disclosed historical development evaluation and a frozen prospective holdout."
        )
