"""Frozen contracts for KS3 swing-specialist research."""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.swing.strategy_labels import STRATEGY_IDS
from market_predictor.v3.errors import DataReadinessError

SPECIALIST_RESEARCH_SCHEMA = "swing.specialist_research.v3"
SPECIALIST_DATASET_SCHEMA = "swing.specialist_dataset.v3"
SPECIALIST_DATASET_BUNDLE_SCHEMA = "swing.specialist_dataset_bundle.v3"
SPECIALIST_MODEL_SCHEMA = "swing.specialist_model.v3"
SPECIALIST_EVIDENCE_SCHEMA = "swing.specialist_evidence.v3"

FeatureProfile = Literal[
    "technical_only",
    "catalyst_only",
    "technical_plus_catalyst",
]
EstimatorFamily = Literal[
    "deterministic_baseline",
    "logistic",
    "hist_gradient_boosting",
    "direct_ranker",
]
DeterministicScore = Literal[
    "xs_rank_rel_return_20d_vs_sector",
    "trend_strength",
    "catalyst_confirmation",
    "reversal_extremity",
    "breakout_confirmation",
]

FEATURE_PROFILES = (
    "technical_only",
    "catalyst_only",
    "technical_plus_catalyst",
)
ESTIMATOR_FAMILIES = (
    "deterministic_baseline",
    "logistic",
    "hist_gradient_boosting",
    "direct_ranker",
)
RANKER_STRATEGIES = frozenset(
    {
        "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1",
        "SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1",
    }
)


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingSpecialistStrategyConfig(FrozenModel):
    feature_profiles: tuple[FeatureProfile, ...]
    estimator_families: tuple[EstimatorFamily, ...]
    deterministic_score: DeterministicScore

    @model_validator(mode="after")
    def validate_strategy(self) -> Self:
        if not self.feature_profiles or len(set(self.feature_profiles)) != len(
            self.feature_profiles
        ):
            raise ValueError("strategy feature profiles must be non-empty and unique")
        if (
            not self.estimator_families
            or len(set(self.estimator_families))
            != len(self.estimator_families)
        ):
            raise ValueError(
                "strategy estimator families must be non-empty and unique"
            )
        if "deterministic_baseline" not in self.estimator_families:
            raise ValueError("every specialist needs a deterministic baseline")
        return self

    def experiment_count(self) -> int:
        learned = sum(
            family != "deterministic_baseline"
            for family in self.estimator_families
        )
        return 1 + learned * len(self.feature_profiles)


class SwingSpecialistResearchConfig(FrozenModel):
    schema_version: str = SPECIALIST_RESEARCH_SCHEMA
    n_splits: int = Field(ge=2, le=8)
    min_train_sessions: int = Field(ge=20)
    min_train_rows: int = Field(ge=100)
    min_training_tickers: int = Field(ge=2)
    ticker_holdout_fraction: float = Field(gt=0, lt=1)
    top_k: int = Field(ge=1, le=100)
    random_seed: int
    logistic_max_iter: int = Field(ge=25, le=2_000)
    hgb_max_iter: int = Field(ge=25, le=2_000)
    hgb_learning_rate: float = Field(gt=0, le=1)
    hgb_l2_regularization: float = Field(ge=0)
    ranker_max_iter: int = Field(ge=25, le=2_000)
    ranker_learning_rate: float = Field(gt=0, le=1)
    ranker_max_depth: int = Field(ge=1, le=12)
    ranker_max_bin: int = Field(ge=16, le=1_024)
    ranker_n_jobs: int = Field(ge=1, le=4)
    minimum_feature_non_null_rate: float = Field(ge=0, le=1)
    minimum_selected_trades: int = Field(ge=1)
    minimum_avg_net_return: float
    minimum_avg_excess_return_vs_spy: float
    minimum_avg_excess_return_vs_sector: float
    minimum_avg_net_return_ci_low: float
    minimum_avg_excess_return_vs_spy_ci_low: float
    minimum_profit_factor: float = Field(ge=0)
    maximum_drawdown: float = Field(gt=0, le=1)
    maximum_negative_phase_rate: float = Field(ge=0, le=1)
    required_market_regimes: tuple[str, ...]
    minimum_regime_selected_trades: int = Field(ge=1)
    minimum_regime_avg_net_return: float
    minimum_regime_avg_excess_return_vs_spy: float
    capacity_participation_rate: float = Field(gt=0, le=0.05)
    maximum_process_memory_gib: float = Field(ge=1, le=4)
    memory_guard_headroom_gib: float = Field(ge=0.5, le=2)
    feature_profiles: dict[FeatureProfile, tuple[str, ...]]
    strategies: dict[str, SwingSpecialistStrategyConfig]

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != SPECIALIST_RESEARCH_SCHEMA:
            raise ValueError("unsupported specialist research schema")
        if self.memory_guard_headroom_gib >= self.maximum_process_memory_gib:
            raise ValueError("memory guard headroom must be below the hard budget")
        if (
            not self.required_market_regimes
            or len(set(self.required_market_regimes))
            != len(self.required_market_regimes)
        ):
            raise ValueError(
                "required market regimes must be non-empty and unique"
            )
        if set(self.feature_profiles) != set(FEATURE_PROFILES):
            raise ValueError("specialist feature-profile catalog mismatch")
        if set(self.strategies) != set(STRATEGY_IDS):
            raise ValueError("specialist strategy catalog mismatch")
        for name, features in self.feature_profiles.items():
            if not features or len(set(features)) != len(features):
                raise ValueError(
                    f"feature profile {name} must be non-empty and unique"
                )
            forbidden = sorted(
                feature
                for feature in features
                if feature.startswith(("future_", "target_", "path_"))
                or feature
                in {
                    "entry_time_utc",
                    "exit_time_utc",
                    "label_available_at_utc",
                    "strategy_target",
                    "strategy_outcome",
                }
            )
            if forbidden:
                raise ValueError(
                    f"feature profile {name} contains labels: {forbidden}"
                )
        for strategy_id, strategy in self.strategies.items():
            if strategy.experiment_count() > 12:
                raise ValueError(
                    f"{strategy_id} exceeds the 12-experiment budget"
                )
            if (
                "direct_ranker" in strategy.estimator_families
                and strategy_id not in RANKER_STRATEGIES
            ):
                raise ValueError(
                    f"direct ranker is not eligible for {strategy_id}"
                )
        return self

    def sha256(self) -> str:
        payload = self.model_dump(mode="json")
        encoded = json.dumps(
            payload,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()


def load_swing_specialist_research_config(
    path: Path,
) -> SwingSpecialistResearchConfig:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(
            f"specialist research policy is unreadable: {path}"
        ) from exc
    return SwingSpecialistResearchConfig.model_validate(raw)
