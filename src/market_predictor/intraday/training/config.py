from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import date
from typing import Any, Mapping
import json
import hashlib

def _json_sha256(value: Any) -> str:
    encoded = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

@dataclass(frozen=True, slots=True)
class IntradayDevelopmentConfig:
    """Frozen development policy. Future observations are not an input."""

    development_end_date: str = "2026-07-08"
    future_holdout_start_date: str = "2026-07-09"
    validation_folds: int = 4
    minimum_train_sessions: int = 120
    minimum_validation_sessions: int = 40
    embargo_sessions: int = 1
    maximum_label_horizon_minutes: int = 30
    minimum_rows: int = 1_000
    minimum_securities: int = 200
    security_holdout_fraction: float = 0.20
    calibration_fraction: float = 0.20
    minimum_calibration_sessions: int = 20
    maximum_candidates_per_decision: int = 5
    maximum_concurrent_positions: int = 5
    position_weight: float = 0.10
    per_security_cooldown_minutes: int = 30
    expected_net_return_thresholds_bps: tuple[float, ...] = (0.0, 3.0)
    maximum_stop_probability_thresholds: tuple[float, ...] = (0.35,)
    ridge_alphas: tuple[float, ...] = (1.0,)
    logistic_c_values: tuple[float, ...] = (1.0,)
    hgb_learning_rates: tuple[float, ...] = (0.05,)
    hgb_max_leaf_nodes: tuple[int, ...] = (15, 31)
    hgb_max_iter: int = 150
    hgb_max_bins: int = 127
    bootstrap_samples: int = 2_000
    bootstrap_block_sessions: int = 5
    random_seed: int = 42
    minimum_validation_trades: int = 200
    minimum_validation_sessions_with_trades: int = 40
    minimum_scope_rows: int = 1_000
    minimum_scope_securities: int = 20
    minimum_positive_net_return_roc_auc: float = 0.60
    minimum_seen_positive_net_lift: float = 1.10
    minimum_unseen_positive_net_lift: float = 1.03
    minimum_seen_stop_hit_roc_auc: float = 0.55
    minimum_unseen_stop_hit_roc_auc: float = 0.52
    maximum_stop_hit_brier: float = 0.25
    maximum_stop_hit_ece: float = 0.10
    minimum_average_trade_net_return_bps: float = 3.0
    minimum_average_daily_net_return_bps: float = 0.0
    minimum_daily_return_ci_low_bps: float = 0.0
    minimum_profit_factor: float = 1.05
    minimum_economic_rank_gain_bps: float = 0.0
    minimum_average_spy_excess_bps: float = 0.0
    minimum_average_qqq_excess_bps: float = 0.0
    minimum_average_sector_excess_bps: float = 0.0
    maximum_drawdown: float = 0.15
    maximum_round_trip_turnover: float = 1.0
    minimum_profitable_fold_fraction: float = 1.0
    maximum_negative_session_rate: float = 0.55
    minimum_return_to_drawdown: float = 0.50
    stress_cost_bps: float = 20.0
    minimum_stress_average_daily_return_bps: float = 0.0
    cost_curve_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    continuation_min_volume_return_1_bar: float = 0.0
    continuation_min_stock_return_20m: float = 0.0
    continuation_min_vwap_distance_atr: float = 0.0
    reversion_max_stock_return_20m: float = 0.0
    reversion_max_vwap_distance_atr: float = -0.5
    reversion_max_volume_rsi_14: float = 45.0
    maximum_process_memory_gib: float = 4.0
    memory_guard_headroom_gib: float = 0.75
    future_access_registry_directory: str = "data/state/intraday_future_access"

    def __post_init__(self) -> None:
        development_end = date.fromisoformat(self.development_end_date)
        future_start = date.fromisoformat(self.future_holdout_start_date)
        if future_start <= development_end:
            raise ValueError("future holdout must start strictly after development")
        if self.validation_folds < 2:
            raise ValueError("validation_folds must be at least two")
        if self.minimum_train_sessions < 20 or self.minimum_validation_sessions < 5:
            raise ValueError("walk-forward session minimums are too small")
        if self.embargo_sessions < 1 or self.maximum_label_horizon_minutes != 30:
            raise ValueError("one-session embargo and 30-minute labels are required")
        if self.minimum_rows < 1 or self.minimum_securities < 2:
            raise ValueError("training population minimums are invalid")
        if not self.future_access_registry_directory.strip():
            raise ValueError("future access registry directory is required")
        if not 0.05 <= self.security_holdout_fraction <= 0.40:
            raise ValueError("security holdout fraction is invalid")
        if not 0.10 <= self.calibration_fraction <= 0.35 or self.minimum_calibration_sessions < 5:
            raise ValueError("downside calibration controls are invalid")
        if not 1 <= self.maximum_candidates_per_decision <= 30:
            raise ValueError("maximum candidates per decision must be in [1, 30]")
        if not 1 <= self.maximum_concurrent_positions <= 30:
            raise ValueError("maximum concurrent positions must be in [1, 30]")
        if not 0.0 < self.position_weight <= 1.0 / self.maximum_concurrent_positions + 1e-12:
            raise ValueError("position_weight can neither be zero nor imply leverage")
        if self.per_security_cooldown_minutes < self.maximum_label_horizon_minutes:
            raise ValueError("security cooldown must cover the complete label horizon")
        if not self.expected_net_return_thresholds_bps or any(
            value < 0.0 for value in self.expected_net_return_thresholds_bps
        ):
            raise ValueError("expected-return thresholds must be non-negative")
        if tuple(sorted(set(self.expected_net_return_thresholds_bps))) != self.expected_net_return_thresholds_bps:
            raise ValueError("expected-return thresholds must be unique and ordered")
        if (
            not self.maximum_stop_probability_thresholds
            or any(not 0.0 < value < 1.0 for value in self.maximum_stop_probability_thresholds)
            or tuple(sorted(set(self.maximum_stop_probability_thresholds)))
            != self.maximum_stop_probability_thresholds
        ):
            raise ValueError("stop probability thresholds are invalid")
        if not self.ridge_alphas or any(value <= 0.0 for value in self.ridge_alphas):
            raise ValueError("ridge alphas must be positive")
        if not self.logistic_c_values or any(value <= 0.0 for value in self.logistic_c_values):
            raise ValueError("logistic C values must be positive")
        if not self.hgb_learning_rates or any(value <= 0.0 for value in self.hgb_learning_rates):
            raise ValueError("HGB learning rates must be positive")
        if not self.hgb_max_leaf_nodes or any(value < 2 for value in self.hgb_max_leaf_nodes):
            raise ValueError("HGB leaf-node limits are invalid")
        if self.hgb_max_iter < 10 or not 2 <= self.hgb_max_bins <= 255:
            raise ValueError("HGB iteration or bin limits are invalid")
        if not 100 <= self.bootstrap_samples <= 5_000 or self.bootstrap_block_sessions < 2:
            raise ValueError("moving-block bootstrap controls are invalid")
        if self.minimum_validation_trades < 1 or self.minimum_validation_sessions_with_trades < 2:
            raise ValueError("economic sample gates are invalid")
        if self.minimum_scope_rows < 100 or self.minimum_scope_securities < 5:
            raise ValueError("validation scope row minimum is invalid")
        if not 0.5 <= self.minimum_positive_net_return_roc_auc <= 1.0:
            raise ValueError("positive-return ROC-AUC gate is invalid")
        if not 1.0 <= self.minimum_seen_positive_net_lift or not 1.0 <= self.minimum_unseen_positive_net_lift:
            raise ValueError("positive-return lift gates are invalid")
        if not 0.5 <= self.minimum_seen_stop_hit_roc_auc <= 1.0 or not 0.5 <= self.minimum_unseen_stop_hit_roc_auc <= 1.0:
            raise ValueError("stop-hit ROC-AUC gates are invalid")
        if not 0.0 < self.maximum_stop_hit_brier < 1.0 or not 0.0 < self.maximum_stop_hit_ece < 1.0:
            raise ValueError("stop-risk calibration gates are invalid")
        if self.minimum_profit_factor < 1.0 or not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("profit-factor or drawdown gate is invalid")
        if self.maximum_round_trip_turnover <= 0.0 or not 0.0 <= self.minimum_profitable_fold_fraction <= 1.0:
            raise ValueError("turnover or fold-stability gate is invalid")
        if not 0.0 <= self.maximum_negative_session_rate <= 1.0 or self.minimum_return_to_drawdown < 0.0:
            raise ValueError("loss-frequency or return/drawdown gate is invalid")
        if self.stress_cost_bps not in self.cost_curve_bps or any(value < 0.0 for value in self.cost_curve_bps):
            raise ValueError("cost curve must contain the configured stress cost")
        if tuple(sorted(set(self.cost_curve_bps))) != self.cost_curve_bps:
            raise ValueError("cost curve must be unique and ordered")
        if not 0.0 < self.maximum_process_memory_gib <= 4.0:
            raise ValueError("process memory hard limit must be in (0, 4] GiB")
        if not 0.0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard limit")

@dataclass(frozen=True, slots=True)
class BaselineProfile:
    profile_id: str
    description: str
    population_rule: Mapping[str, float]

    def sha256(self) -> str:
        return _json_sha256(asdict(self))

@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate_id: str
    family: str
    hyperparameters: Mapping[str, float | int]
