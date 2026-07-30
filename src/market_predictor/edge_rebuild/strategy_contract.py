"""The frozen strategy contract: every number chosen before any result is seen.

Thresholds are picked once, from trading rationale and a design window that is
kept separate from evaluation. After that they are immutable. Adjusting a
threshold after seeing how validation turned out converts a random result into
an apparent discovery, and that is the specific mistake behind the rejected V2
strategies.

The contract also caps how many variants may be tried. Testing enough ideas
guarantees one looks good by luck, so the budget is the control for that, not a
matter of convenience.
"""

from __future__ import annotations

import hashlib
import json
import tomllib
from pathlib import Path
from typing import Self

from pydantic import BaseModel, ConfigDict, Field, model_validator

from market_predictor.v3.errors import DataReadinessError

STRATEGY_CONTRACT_SCHEMA = "edge_rebuild.strategy_contract.v1"


class FrozenModel(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class SwingContract(FrozenModel):
    strategy_id: str
    horizon_sessions: int = Field(ge=2, le=30)
    entry_reference: str
    exit_rule: str
    decision_cutoff: str
    round_trip_cost_bps: float = Field(gt=0, le=100)
    minimum_warmup_sessions: int = Field(ge=250)
    maximum_trades_per_decision: int = Field(ge=1, le=50)
    minimum_expected_net_edge_bps: float = Field(ge=0)

    @model_validator(mode="after")
    def validate_swing(self) -> Self:
        if self.entry_reference != "next_session_open":
            raise ValueError("swing entry must be the next session open")
        # A daily-bar strategy cannot observe when an intraday stop was touched,
        # so a target/stop exit would score fills the data cannot support.
        if self.exit_rule != "horizon_close":
            raise ValueError("swing exit must be timeout-only at the horizon close")
        if self.minimum_expected_net_edge_bps >= self.round_trip_cost_bps:
            raise ValueError("required net edge cannot exceed the round trip cost")
        return self


class IntradayContract(FrozenModel):
    strategy_id: str
    horizon_minutes: int = Field(ge=5, le=390)
    entry_reference: str
    exit_rule: str
    decision_finalization_seconds: int = Field(ge=0, le=300)
    round_trip_cost_bps: float = Field(gt=0, le=100)
    target_atr_multiple: float = Field(gt=0, le=5)
    stop_atr_multiple: float = Field(ge=1.0, le=5)
    atr_timeframe: str
    atr_lookback_bars: int = Field(ge=5, le=50)
    minimum_warmup_bars: int = Field(ge=130)
    maximum_trades_per_session: int = Field(ge=1, le=50)
    minimum_expected_net_edge_bps: float = Field(ge=0)
    session_segments: tuple[str, ...]
    opening_end_et: str
    midday_end_et: str

    @model_validator(mode="after")
    def validate_intraday(self) -> Self:
        if self.entry_reference != "next_one_minute_open":
            raise ValueError("intraday entry must be the next one-minute open")
        if self.exit_rule != "target_stop_timeout":
            raise ValueError("intraday exit must resolve target, stop, or timeout")
        if self.target_atr_multiple <= self.stop_atr_multiple:
            raise ValueError(
                "target must exceed stop, otherwise the setup risks more than it seeks"
            )
        # A daily ATR applied to a thirty-minute hold makes the stop unreachable
        # and silently converts this into a timeout-only strategy.
        if self.atr_timeframe != "5Min":
            raise ValueError("ATR must be measured on the trading timeframe")
        if tuple(self.session_segments) != ("opening", "midday", "late"):
            raise ValueError("intraday segments must remain opening, midday, and late")
        if self.opening_end_et >= self.midday_end_et:
            raise ValueError("opening must end before midday ends")
        return self


class IntradayUniverseContract(FrozenModel):
    """Two-layer selection: what could be traded, then what is moving today."""

    scope: str
    index_restricted: bool
    minimum_average_volume_shares: int = Field(ge=100_000)
    average_volume_lookback_sessions: int = Field(ge=5, le=120)
    minimum_price: float = Field(gt=0)
    maximum_price: float = Field(gt=0)
    minimum_bar_continuity: float = Field(gt=0, le=1)
    minimum_relative_volume: float = Field(ge=1.0)
    relative_volume_lookback_sessions: int = Field(ge=5, le=120)
    relative_volume_excludes_current_session: bool
    maximum_candidates_per_session: int = Field(ge=1, le=200)

    @model_validator(mode="after")
    def validate_universe(self) -> Self:
        if self.index_restricted:
            raise ValueError(
                "the intraday universe must not be index-restricted; the most "
                "tradable names are frequently not index constituents"
            )
        if self.minimum_price >= self.maximum_price:
            raise ValueError("price floor must be below the price ceiling")
        if self.minimum_relative_volume < 1.5:
            raise ValueError(
                "relative volume below 1.5 does not select stocks in play"
            )
        # Selecting on the session being traded would use information the
        # decision could not have had.
        if not self.relative_volume_excludes_current_session:
            raise ValueError(
                "relative volume must be measured from prior sessions only"
            )
        return self


class MethodologyContract(FrozenModel):
    """Named published methods, so an implementation can be checked against them."""

    labeling: str
    cross_validation: str
    sampling: str
    meta_labeling_enabled: bool

    @model_validator(mode="after")
    def validate_methodology(self) -> Self:
        if self.labeling != "triple_barrier":
            raise ValueError("labels must resolve target, stop, or timeout first")
        if self.cross_validation != "purged_k_fold_with_embargo":
            raise ValueError(
                "overlapping labels require purged and embargoed validation"
            )
        if self.sampling != "event_based":
            raise ValueError(
                "fixed-clock sampling trains mostly on stocks that were not moving"
            )
        return self


class LabelContract(FrozenModel):
    retained: tuple[str, ...]
    benchmark_market: str
    benchmark_sector_source: str

    @model_validator(mode="after")
    def validate_labels(self) -> Self:
        required = {
            "gross_return",
            "cost",
            "net_return",
            "spy_excess_return",
            "sector_excess_return",
        }
        if not required.issubset(self.retained):
            raise ValueError(f"labels must retain {sorted(required)}")
        if self.benchmark_sector_source != "point_in_time_membership":
            raise ValueError("sector benchmark must come from point-in-time membership")
        return self


class ValidationContract(FrozenModel):
    purged_folds: int = Field(ge=3, le=10)
    minimum_test_sessions_per_fold: int = Field(ge=30)
    embargo_sessions: int = Field(ge=1)
    unseen_ticker_holdout_fraction: float = Field(gt=0, lt=0.5)
    unseen_ticker_assignment: str

    @model_validator(mode="after")
    def validate_validation(self) -> Self:
        if self.unseen_ticker_assignment != "deterministic_hash":
            raise ValueError("unseen-ticker assignment must be deterministic")
        return self


class ExperimentBudget(FrozenModel):
    maximum_learned_candidates: int = Field(ge=1, le=6)
    maximum_feature_profiles: int = Field(ge=1, le=2)
    maximum_selection_policies: int = Field(ge=1, le=2)
    shadow_retries: int = Field(ge=0, le=0)


class FeatureContract(FrozenModel):
    profiles: tuple[str, ...]
    promotion_eligible_profile: str
    extended_context_enabled: bool
    news_count_normalization: str
    raw_news_counts_prohibited: bool
    sentiment_decay_half_life_minutes_intraday: float = Field(gt=0, le=1_440)
    sentiment_decay_half_life_hours_swing: float = Field(gt=0, le=336)
    swing_sentiment_decay_evidence: str

    @model_validator(mode="after")
    def validate_features(self) -> Self:
        if self.promotion_eligible_profile not in self.profiles:
            raise ValueError("the promotion-eligible profile must be a declared profile")
        # Provider news coverage grew across the sample, so a raw count trends
        # upward for reasons that have nothing to do with the market.
        if not self.raw_news_counts_prohibited:
            raise ValueError("raw news counts are prohibited as estimator features")
        if self.news_count_normalization not in {
            "cross_sectional_rank",
            "trailing_baseline_ratio",
        }:
            raise ValueError("news counts must be normalized within a cross-section")
        return self


class StressContract(FrozenModel):
    cost_multiplier: float = Field(ge=1.5, le=5.0)


class RetirementContract(FrozenModel):
    rule: str

    @model_validator(mode="after")
    def validate_retirement(self) -> Self:
        if len(self.rule.strip()) < 40:
            raise ValueError("the retirement rule must be stated explicitly")
        return self


class StrategyContract(FrozenModel):
    """One immutable contract covering both strategies."""

    schema_version: str
    swing: SwingContract
    intraday: IntradayContract
    intraday_universe: IntradayUniverseContract
    methodology: MethodologyContract
    labels: LabelContract
    validation: ValidationContract
    experiment_budget: ExperimentBudget
    features: FeatureContract
    stress: StressContract
    retirement: RetirementContract

    @model_validator(mode="after")
    def validate_contract(self) -> Self:
        if self.schema_version != STRATEGY_CONTRACT_SCHEMA:
            raise ValueError("unsupported strategy contract schema")
        if self.swing.strategy_id == self.intraday.strategy_id:
            raise ValueError("strategies must have distinct identities")
        for strategy_id in (self.swing.strategy_id, self.intraday.strategy_id):
            # New semantics get new identities; a redefined setup reusing an old
            # name silently invalidates every comparison against it.
            if not strategy_id.endswith(".V1"):
                raise ValueError(f"strategy identity must be versioned: {strategy_id}")
        return self

    def sha256(self) -> str:
        payload = json.dumps(
            self.model_dump(mode="json"), sort_keys=True, separators=(",", ":")
        ).encode()
        return hashlib.sha256(payload).hexdigest()


def load_strategy_contract(path: Path) -> StrategyContract:
    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"strategy contract is unreadable: {path}") from exc
    try:
        return StrategyContract.model_validate(raw)
    except ValueError as exc:
        raise DataReadinessError(f"strategy contract is invalid: {path}") from exc
