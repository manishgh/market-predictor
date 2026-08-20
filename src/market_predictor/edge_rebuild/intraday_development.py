"""Development-only, cost-aware intraday model training and evaluation."""

from __future__ import annotations

import gc
import hashlib
import json
import math
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_event_training import (
    DIRECTIONAL_EVENT_SUBTYPES,
    filter_to_research_event_cohort,
    load_intraday_research_event_cohort,
)
from market_predictor.edge_rebuild.intraday_training import (
    MODEL_FEATURE_COLUMNS,
    PublishedIntradayDataset,
    load_published_intraday_dataset,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


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
        development_end = _parse_date(self.development_end_date, "development_end_date")
        future_start = _parse_date(self.future_holdout_start_date, "future_holdout_start_date")
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
class DevelopmentTrainingResult:
    output_directory: Path
    status: str
    selected_candidate_id: str | None
    evaluation: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate_id: str
    family: str
    hyperparameters: Mapping[str, float | int]


@dataclass(frozen=True, slots=True)
class BaselineProfile:
    profile_id: str
    description: str
    population_rule: Mapping[str, float]

    def sha256(self) -> str:
        return _json_sha256(asdict(self))


@dataclass(frozen=True, slots=True)
class _FittedPair:
    opportunity_estimator: Any
    downside_estimator: Any
    downside_calibrator: LogisticRegression
    fit_sessions: tuple[str, ...]
    calibration_sessions: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class _Fold:
    fold: int
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    embargo_sessions: tuple[str, ...]


def load_intraday_development_config(path: Path) -> IntradayDevelopmentConfig:
    """Load a complete policy. Partial implicit overrides are forbidden."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"intraday development policy is unreadable: {path}") from exc
    payload = raw.get("training")
    if not isinstance(payload, Mapping):
        raise DataReadinessError("intraday development policy requires [training]")
    expected = {field.name for field in fields(IntradayDevelopmentConfig)}
    actual = {str(key) for key in payload}
    if expected != actual:
        raise DataReadinessError(
            f"intraday development policy fields differ; missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    values = dict(payload)
    for name in (
        "expected_net_return_thresholds_bps",
        "maximum_stop_probability_thresholds",
        "ridge_alphas",
        "logistic_c_values",
        "hgb_learning_rates",
        "hgb_max_leaf_nodes",
        "cost_curve_bps",
    ):
        value = values[name]
        if not isinstance(value, list):
            raise DataReadinessError(f"{name} must be an array")
        values[name] = tuple(value)
    try:
        return IntradayDevelopmentConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("intraday development policy is invalid") from exc


def baseline_profile(
    hypothesis: str,
    config: IntradayDevelopmentConfig,
) -> BaselineProfile:
    """Return one frozen, causal long-only hypothesis contract."""

    if hypothesis == "continuation":
        return BaselineProfile(
            profile_id="intraday_bar_continuation_long_v1",
            description="positive one-volume-bar and twenty-minute continuation at or above session VWAP",
            population_rule={
                "volume_return_1_bar_gt": config.continuation_min_volume_return_1_bar,
                "stock_return_20m_gt": config.continuation_min_stock_return_20m,
                "session_vwap_distance_five_minute_atr_gte": config.continuation_min_vwap_distance_atr,
            },
        )
    if hypothesis == "long-reversion":
        return BaselineProfile(
            profile_id="intraday_bar_long_reversion_v1",
            description="long reversion after negative twenty-minute return below session VWAP with low volume RSI",
            population_rule={
                "stock_return_20m_lt": config.reversion_max_stock_return_20m,
                "session_vwap_distance_five_minute_atr_lte": config.reversion_max_vwap_distance_atr,
                "volume_rsi_14_lte": config.reversion_max_volume_rsi_14,
            },
        )
    raise ValueError("hypothesis must be 'continuation' or 'long-reversion'")


def _profile_mask(
    data: pd.DataFrame,
    profile: BaselineProfile,
) -> pd.Series:
    if profile.profile_id == "intraday_bar_continuation_long_v1":
        return (
            data["volume_return_1_bar"].gt(profile.population_rule["volume_return_1_bar_gt"])
            & data["stock_return_20m"].gt(profile.population_rule["stock_return_20m_gt"])
            & data["session_vwap_distance_five_minute_atr"].ge(
                profile.population_rule["session_vwap_distance_five_minute_atr_gte"]
            )
        )
    if profile.profile_id == "intraday_bar_long_reversion_v1":
        return (
            data["stock_return_20m"].lt(profile.population_rule["stock_return_20m_lt"])
            & data["session_vwap_distance_five_minute_atr"].le(
                profile.population_rule["session_vwap_distance_five_minute_atr_lte"]
            )
            & data["volume_rsi_14"].le(profile.population_rule["volume_rsi_14_lte"])
        )
    raise DataReadinessError("intraday baseline profile identity is unsupported")


def train_intraday_development_candidate(
    dataset_authority_directory: Path,
    output_directory: Path,
    *,
    hypothesis: str,
    config: IntradayDevelopmentConfig | None = None,
    research_event_preflight_directory: Path | None = None,
    research_event_subtype: str | None = None,
) -> DevelopmentTrainingResult:
    """Train one technical or event-confirmed hypothesis without future data."""

    policy = config or IntradayDevelopmentConfig()
    profile = baseline_profile(hypothesis, policy)
    _guard_memory(policy, "intraday development start", peak=False)
    immutable_inputs = [dataset_authority_directory]
    if research_event_subtype is not None and research_event_preflight_directory is None:
        raise DataReadinessError(
            "intraday event subtype requires a historical event preflight"
        )
    if (
        research_event_subtype is not None
        and research_event_subtype not in DIRECTIONAL_EVENT_SUBTYPES
    ):
        raise DataReadinessError(
            f"unsupported intraday analyst-event subtype: {research_event_subtype}"
        )
    if research_event_preflight_directory is not None:
        immutable_inputs.append(research_event_preflight_directory)
    _require_output_isolated(output_directory, *immutable_inputs)
    event_cohort = None
    if research_event_preflight_directory is not None:
        event_cohort = load_intraday_research_event_cohort(
            research_event_preflight_directory,
            event_subtype=research_event_subtype,
        )
        release_process_memory()
    published = load_published_intraday_dataset(dataset_authority_directory)
    data = _validate_development_frame(published, policy)
    if event_cohort is not None:
        data = filter_to_research_event_cohort(data, event_cohort)
    data = data.loc[_profile_mask(data, profile)].reset_index(drop=True)
    if len(data) < policy.minimum_rows or data["security_id"].nunique() < policy.minimum_securities:
        raise DataReadinessError(
            f"{profile.profile_id} population is too small for governed training"
        )
    sessions = _ordered_sessions(data)
    folds = _walk_forward_folds(data, sessions, policy)
    security_holdout = _stable_security_holdout(data, policy.security_holdout_fraction)
    # Keep one compact feature matrix; candidate fits are sequential.
    features_full = data[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype="float32", copy=True)
    opportunity_target = data["net_return"].to_numpy(dtype="float64", copy=True)
    downside_target = data["stop_hit"].to_numpy(dtype="int8", copy=True)
    data.drop(columns=list(MODEL_FEATURE_COLUMNS), inplace=True)

    frozen_cost_bps = published.frozen_round_trip_cost_bps
    dataset_identity_val = _dataset_identity(published)
    if event_cohort is None:
        model_family = "intraday_technical"
    elif research_event_subtype is None:
        model_family = "intraday_event_confirmed_research"
    else:
        model_family = f"intraday_{research_event_subtype}_confirmed_research"
    if event_cohort is not None:
        dataset_identity_val["research_event_cohort"] = event_cohort.identity
    gc.collect()

    validation_records: list[dict[str, Any]] = []
    retained_predictions: dict[str, pd.DataFrame] = {}
    for spec in _candidate_specs(policy):
        scored, fold_records = _walk_forward_predictions(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            folds,
            security_holdout,
            policy,
        )
        record = _evaluate_spec(spec, scored, fold_records, policy, frozen_cost_bps)
        validation_records.append(record)
        selection_passed = [
            r for r in validation_records if bool(r["selection_passed"])
        ]
        current_winner = (
            max(selection_passed, key=_selection_key) if selection_passed else None
        )
        current_selected = (
            current_winner
            if current_winner is not None
            and bool(current_winner["validation_passed"])
            else None
        )
        current_audit_candidate, _, _, _ = _audit_policy_choice(
            validation_records,
            current_selected,
            preferred=current_winner,
        )

        retained_predictions[spec.candidate_id] = scored
        keys_to_keep = {current_audit_candidate}
        if current_selected is not None:
            keys_to_keep.add(str(current_selected["candidate_id"]))

        for k in list(retained_predictions.keys()):
            if k not in keys_to_keep:
                del retained_predictions[k]

        gc.collect()

        _guard_memory(policy, f"{spec.candidate_id} validation", peak=True)

    selection_passed = [
        record for record in validation_records if bool(record["selection_passed"])
    ]
    selection_winner = (
        max(selection_passed, key=_selection_key) if selection_passed else None
    )
    selected = (
        selection_winner
        if selection_winner is not None
        and bool(selection_winner["validation_passed"])
        else None
    )
    status = "candidate" if selected is not None else "no_candidate"
    selected_id = str(selected["candidate_id"]) if selected is not None else None
    config_payload = asdict(policy)
    config_hash = _json_sha256(config_payload)
    evaluation: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "model_family": model_family,
        "promotion_permitted": False,
        "selection_basis": "development_walk_forward_validation_only",
        "objective": "expected_net_return_with_calibrated_stop_risk_after_frozen_cost",
        "baseline_profile": asdict(profile),
        "baseline_profile_sha256": profile.sha256(),
        "target_hit_used_as_training_target": False,
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "raw_ndcg_reported": False,
        "future_holdout_opened": False,
        "test_access_count": 0,
        "future_holdout_start_date": policy.future_holdout_start_date,
        "development_end_date": policy.development_end_date,
        "dataset": dataset_identity_val,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "training_config": config_payload,
        "training_config_sha256": config_hash,
        "validation_candidates": validation_records,
        "selected_candidate_id": selected_id,
        "gates": _gate_contract(policy),
        "security_holdout": {
            "fraction": policy.security_holdout_fraction,
            "security_count": len(security_holdout),
            "security_set_sha256": _security_set_sha256(security_holdout),
        },
        "future_data_contract": _future_data_contract(policy),
        "memory": memory_audit(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
        ).to_record(),
    }
    audit_candidate, audit_threshold, audit_stop_threshold, audit_passed = _audit_policy_choice(
        validation_records,
        selected,
        preferred=selection_winner,
    )
    audit_ledger = _position_ledger(
        retained_predictions[audit_candidate],
        audit_threshold,
        audit_stop_threshold,
        frozen_cost_bps,
        policy,
    )
    evaluation["auditable_policy_ledger"] = {
        "candidate_id": audit_candidate,
        "threshold_bps": audit_threshold,
        "maximum_stop_probability": audit_stop_threshold,
        "validation_passed": audit_passed,
        "selection_status": "selected_candidate" if audit_passed else "best_failed_diagnostic_only",
        "position_ledger_path": _POSITION_LEDGER_NAME,
        "daily_ledger_path": _DAILY_LEDGER_NAME,
    }
    model_card: dict[str, Any] = {
        "schema_version": "edge_rebuild.intraday_bar_baseline_model_card.v1",
        "status": status,
        "model_family": model_family,
        "promotion_permitted": False,
        "candidate_id": selected_id,
        "horizon_minutes": 30,
        "baseline_profile": asdict(profile),
        "baseline_profile_sha256": profile.sha256(),
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "selection_target": "capital_weighted_net_economics",
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "development_rows": int(len(data)),
        "development_sessions": int(len(sessions)),
        "development_securities": int(data["security_id"].nunique()),
        "future_holdout_opened": False,
        "future_data_contract": _future_data_contract(policy),
        "limitations": [
            "candidate is development-only and cannot be promoted without a separately collected future holdout",
            "event-time equity marks open positions at their frozen stop until exact recorded exit",
            (
                "historical catalyst timestamps are provider-publication proxies; catalyst is a research-only confirmation filter"
                if event_cohort is not None
                else "catalyst and trade/quote microstructure are outside this technical estimator contract"
            ),
        ],
    }
    candidate: dict[str, Any] | None = None
    if selected is not None:
        spec = next(item for item in _candidate_specs(policy) if item.candidate_id == selected_id)
        fitted = _fit_pair(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            sessions,
            policy,
            excluded_securities=security_holdout,
        )
        candidate = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "candidate",
            "model_family": model_family,
            "promotion_permitted": False,
            "validation_passed": True,
            "candidate_id": selected_id,
            "baseline_profile": asdict(profile),
            "baseline_profile_sha256": profile.sha256(),
            "family": spec.family,
            "hyperparameters": dict(spec.hyperparameters),
            "expected_net_return_threshold_bps": float(selected["selected_threshold_bps"]),
            "maximum_stop_probability": float(selected["selected_maximum_stop_probability"]),
            "frozen_round_trip_cost_bps": frozen_cost_bps,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "ordered_feature_sha256": published.ordered_feature_sha256,
            "opportunity_estimator": fitted.opportunity_estimator,
            "downside_estimator": fitted.downside_estimator,
            "downside_calibrator": fitted.downside_calibrator,
            "downside_fit_sessions": list(fitted.fit_sessions),
            "downside_calibration_sessions": list(fitted.calibration_sessions),
            "dataset": dataset_identity_val,
            "training_config": config_payload,
            "training_config_sha256": config_hash,
            "future_data_contract": _future_data_contract(policy),
        }
    _guard_memory(policy, "intraday development publication", peak=True)
    _publish_development(
        output_directory,
        candidate,
        evaluation,
        model_card,
        audit_ledger,
        retained_predictions[audit_candidate],
    )
    load_complete_intraday_development_output(output_directory)
    return DevelopmentTrainingResult(output_directory, status, selected_id, evaluation)


def evaluate_future_intraday_holdout(
    candidate_authority_directory: Path,
    future_dataset_authority_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Evaluate one accepted development policy on a separate future authority.

    Candidate validation is verified before the future path is inspected. The
    future dataset is rejected unless every row starts on/after the frozen date.
    """

    candidate, manifest = _load_validation_passed_candidate(candidate_authority_directory)
    if candidate.get("model_family") != "intraday_technical":
        raise DataReadinessError(
            "research-only event-confirmed candidates cannot open the future holdout"
        )
    contract = _object(candidate.get("future_data_contract"), "future_data_contract")
    future_start = _parse_date(str(contract.get("minimum_session_date")), "minimum_session_date")
    development_end = _parse_date(str(contract.get("development_end_date")), "development_end_date")
    policy = IntradayDevelopmentConfig(
        **_tuple_config_values(_object(candidate.get("training_config"), "training_config"))
    )
    _require_output_isolated(
        output_directory,
        candidate_authority_directory,
        future_dataset_authority_directory,
    )
    if not future_dataset_authority_directory.is_dir():
        raise DataReadinessError(
            f"future holdout data does not exist; collect sessions from {future_start.isoformat()} onward"
        )
    access_lock = _consume_future_access(
        candidate_authority_directory,
        future_dataset_authority_directory,
        Path(str(contract.get("future_access_registry_directory", ""))),
    )
    published = load_published_intraday_dataset(future_dataset_authority_directory)
    development_dataset = _object(candidate.get("dataset"), "candidate dataset")
    for identity_key in (
        "transformation_sha256",
        "strategy_contract_sha256",
        "ordered_feature_sha256",
    ):
        if getattr(published, identity_key) != development_dataset.get(identity_key):
            raise DataReadinessError(
                f"future holdout {identity_key} differs from development"
            )
    expected_cost_bps = _required_finite_number(
        candidate.get("frozen_round_trip_cost_bps"),
        "frozen_round_trip_cost_bps",
    )
    if not math.isclose(
        published.frozen_round_trip_cost_bps,
        expected_cost_bps,
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise DataReadinessError("future holdout cost contract differs from development")
    data = _validate_future_frame(published, future_start, development_end, policy)
    profile_raw = _object(candidate.get("baseline_profile"), "baseline_profile")
    profile = BaselineProfile(
        profile_id=str(profile_raw.get("profile_id", "")),
        description=str(profile_raw.get("description", "")),
        population_rule={
            str(key): _required_finite_number(value, f"population rule {key}")
            for key, value in _object(
                profile_raw.get("population_rule"), "population_rule"
            ).items()
        },
    )
    if candidate.get("baseline_profile_sha256") != profile.sha256():
        raise DataReadinessError("candidate baseline profile identity differs")
    data = data.loc[_profile_mask(data, profile)].reset_index(drop=True)
    minimum_sessions = int(contract.get("minimum_sessions", 0))
    minimum_rows = int(contract.get("minimum_rows", 0))
    minimum_securities = int(contract.get("minimum_securities", 0))
    actual_sessions = int(data["session_date_et"].nunique())
    actual_rows = int(len(data))
    actual_securities = int(data["security_id"].nunique())
    if minimum_sessions < 1 or actual_sessions < minimum_sessions:
        raise DataReadinessError(
            f"future holdout has {actual_sessions} complete sessions; requires {minimum_sessions}"
        )
    if minimum_rows < 1 or actual_rows < minimum_rows:
        raise DataReadinessError(
            f"future holdout has {actual_rows} profile rows; requires {minimum_rows}"
        )
    if minimum_securities < 2 or actual_securities < minimum_securities:
        raise DataReadinessError(
            f"future holdout has {actual_securities} profile securities; requires {minimum_securities}"
        )
    opportunity = candidate.get("opportunity_estimator")
    downside = candidate.get("downside_estimator")
    calibrator = candidate.get("downside_calibrator")
    if (
        opportunity is None
        or not hasattr(opportunity, "predict")
        or downside is None
        or not hasattr(downside, "predict_proba")
        or calibrator is None
        or not hasattr(calibrator, "predict_proba")
    ):
        raise DataReadinessError("candidate paired estimators are unavailable")
    features = data.loc[:, MODEL_FEATURE_COLUMNS].to_numpy(dtype="float32", copy=False)
    opportunity_score = np.asarray(opportunity.predict(features), dtype="float64")
    raw_stop = _raw_stop_logit(downside, features)
    stop_probability = np.asarray(
        calibrator.predict_proba(raw_stop.reshape(-1, 1))[:, 1], dtype="float64"
    )
    scored = _scored_frame(data, opportunity_score, stop_probability)
    scored["fold"] = 0
    scored["validation_scope"] = "future_holdout"
    threshold = _required_finite_number(
        candidate.get("expected_net_return_threshold_bps"),
        "expected_net_return_threshold_bps",
    )
    stop_threshold = _required_finite_number(
        candidate.get("maximum_stop_probability"), "maximum_stop_probability"
    )
    metrics = _evaluate_policy(
        scored,
        threshold,
        stop_threshold,
        policy,
        published.frozen_round_trip_cost_bps,
    )
    ledger = _position_ledger(
        scored,
        threshold,
        stop_threshold,
        published.frozen_round_trip_cost_bps,
        policy,
    )
    evaluation = {
        "schema_version": FUTURE_EVALUATION_SCHEMA_VERSION,
        "status": "locked_future_evaluated",
        "promotion_permitted": False,
        "selection_changed_after_future_observation": False,
        "future_access_lock_sha256": file_sha256(access_lock),
        "candidate_authority_sha256": file_sha256(candidate_authority_directory / _AUTHORITY_NAME),
        "candidate_manifest_sha256": file_sha256(candidate_authority_directory / _MANIFEST_NAME),
        "candidate_manifest_schema": manifest.get("schema_version"),
        "future_dataset": _dataset_identity(published),
        "future_session_first": str(data["session_date_et"].min()),
        "future_session_last": str(data["session_date_et"].max()),
        "metrics": metrics,
    }
    _publish_future_evaluation(output_directory, evaluation, ledger)
    load_complete_intraday_future_evaluation_output(output_directory)
    return evaluation


def _validate_development_frame(
    published: PublishedIntradayDataset,
    config: IntradayDevelopmentConfig,
) -> pd.DataFrame:
    data = published.frame.copy()
    required = {
        *MODEL_FEATURE_COLUMNS,
        "dataset_row_id",
        "security_id",
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "entry_time_utc",
        "exit_bar_end_utc",
        "feature_eligible",
        "label_eligible",
        "gross_return",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "target_hit",
        "stop_hit",
        "entry_price",
        "stop_price",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataReadinessError(f"intraday development columns are missing: {missing}")
    if len(data) < config.minimum_rows or data["security_id"].nunique() < config.minimum_securities:
        raise DataReadinessError("intraday development population is below frozen minimums")
    dates = pd.to_datetime(data["session_date_et"], errors="coerce").dt.date
    if dates.isna().any():
        raise DataReadinessError("session_date_et contains invalid values")
    if dates.max() > _parse_date(config.development_end_date, "development_end_date"):
        raise DataReadinessError("development trainer refuses observations after 2026-07-08")
    data["session_date_et"] = dates.astype(str)
    if data["dataset_row_id"].isna().any() or data["dataset_row_id"].duplicated().any():
        raise DataReadinessError("dataset row identity must be complete and unique")
    if not data["feature_eligible"].map(_strict_bool).all() or not data["label_eligible"].map(_strict_bool).all():
        raise DataReadinessError("development rows must be feature- and label-eligible")
    for column in ("target_hit", "stop_hit"):
        if not data[column].map(
            lambda value: value is True or value is False or isinstance(value, np.bool_)
        ).all():
            raise DataReadinessError(f"{column} must be boolean")
        data[column] = data[column].astype(bool)
    if (data["target_hit"] & data["stop_hit"]).any():
        raise DataReadinessError("target and stop cannot both be hit")
    for column in (
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "entry_time_utc",
        "exit_bar_end_utc",
    ):
        parsed = pd.to_datetime(data[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise DataReadinessError(f"{column} contains invalid UTC timestamps")
        data[column] = parsed
    if data["feature_available_at_utc"].gt(data["decision_time_utc"]).any():
        raise DataReadinessError("feature availability occurs after decision time")
    if data["label_available_at_utc"].lt(data["exit_bar_end_utc"]).any():
        raise DataReadinessError("label availability precedes the completed path")
    horizon = data["exit_bar_end_utc"] - data["entry_time_utc"]
    if horizon.le(pd.Timedelta(0)).any() or horizon.gt(pd.Timedelta(minutes=30)).any():
        raise DataReadinessError("development labels must use executable paths of at most 30 minutes")
    numeric = [
        *MODEL_FEATURE_COLUMNS,
        "gross_return",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "entry_price",
        "stop_price",
    ]
    for column in numeric:
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"{column} must be finite")
        data[column] = values.astype("float32" if column in MODEL_FEATURE_COLUMNS else "float64")
    expected = data["gross_return"] - published.frozen_round_trip_cost_bps / 10_000.0
    if not np.allclose(expected, data["net_return"], rtol=0.0, atol=1e-10):
        raise DataReadinessError("net return does not match the frozen round-trip cost")
    for benchmark in ("spy", "qqq", "sector"):
        expected_excess = data["net_return"] - data[f"{benchmark}_return"]
        if not np.allclose(
            expected_excess,
            data[f"{benchmark}_excess_return"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise DataReadinessError(
                f"{benchmark.upper()} excess return does not match the executable interval"
            )
    if data["entry_price"].le(0.0).any() or data["stop_price"].le(0.0).any():
        raise DataReadinessError("entry and stop prices must be positive")
    return data.sort_values(
        ["decision_time_utc", "decision_group_id", "security_id"], kind="stable"
    ).reset_index(drop=True)


def _validate_future_frame(
    published: PublishedIntradayDataset,
    future_start: date,
    development_end: date,
    policy: IntradayDevelopmentConfig,
) -> pd.DataFrame:
    values = asdict(policy)
    values["development_end_date"] = "2099-12-30"
    values["future_holdout_start_date"] = "2099-12-31"
    config = IntradayDevelopmentConfig(**values)
    data = _validate_development_frame(published, config)
    dates = pd.to_datetime(data["session_date_et"]).dt.date
    if dates.min() < future_start or dates.min() <= development_end:
        raise DataReadinessError("future holdout overlaps development or starts before its frozen boundary")
    return data


def _ordered_sessions(data: pd.DataFrame) -> tuple[str, ...]:
    ordered = (
        data.groupby("session_date_et", as_index=False, observed=True)["decision_time_utc"]
        .min()
        .sort_values(["decision_time_utc", "session_date_et"], kind="stable")
    )
    return tuple(ordered["session_date_et"].astype(str))


def _stable_security_holdout(
    data: pd.DataFrame,
    fraction: float,
) -> frozenset[str]:
    securities = sorted(set(data["security_id"].astype(str)))
    threshold = int(fraction * 2**64)
    selected = frozenset(
        security
        for security in securities
        if int(hashlib.sha256(security.encode("utf-8")).hexdigest()[:16], 16)
        < threshold
    )
    if not selected or len(selected) == len(securities):
        raise DataReadinessError("stable security holdout produced an empty partition")
    return selected


def _security_set_sha256(securities: frozenset[str]) -> str:
    return hashlib.sha256("\n".join(sorted(securities)).encode("utf-8")).hexdigest()


def _walk_forward_folds(
    data: pd.DataFrame,
    sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
) -> tuple[_Fold, ...]:
    remaining = len(sessions) - config.minimum_train_sessions
    fold_size = remaining // config.validation_folds
    if fold_size < config.minimum_validation_sessions:
        raise DataReadinessError("development history is too short for walk-forward validation")
    folds: list[_Fold] = []
    for index in range(config.validation_folds):
        validation_start = config.minimum_train_sessions + index * fold_size
        validation_end = len(sessions) if index == config.validation_folds - 1 else validation_start + fold_size
        embargo_start = validation_start - config.embargo_sessions
        validation = sessions[validation_start:validation_end]
        train_candidates = sessions[:embargo_start]
        first_validation = data.loc[
            data["session_date_et"].eq(validation[0]), "decision_time_utc"
        ].min()
        safe_train = tuple(
            session
            for session in train_candidates
            if data.loc[data["session_date_et"].eq(session), "label_available_at_utc"].max()
            < first_validation
        )
        if len(safe_train) < 2:
            raise DataReadinessError(f"fold {index} has insufficient purged training history")
        folds.append(
            _Fold(
                fold=index,
                train_sessions=safe_train,
                validation_sessions=validation,
                embargo_sessions=sessions[embargo_start:validation_start],
            )
        )
    return tuple(folds)


def _candidate_specs(config: IntradayDevelopmentConfig) -> tuple[_CandidateSpec, ...]:
    ridge = tuple(
        _CandidateSpec(
            f"ridge_opportunity_alpha_{alpha:g}_logistic_downside_c_{c:g}",
            "ridge_logistic_pair",
            {"alpha": alpha, "downside_c": c},
        )
        for alpha in config.ridge_alphas
        for c in config.logistic_c_values
    )
    hgb = tuple(
        _CandidateSpec(
            f"hgb_opportunity_downside_lr_{rate:g}_leaves_{leaves}",
            "hgb_pair",
            {
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "max_iter": config.hgb_max_iter,
                "max_bins": config.hgb_max_bins,
            },
        )
        for rate in config.hgb_learning_rates
        for leaves in config.hgb_max_leaf_nodes
    )
    candidates = ridge + hgb
    if len(candidates) > 3:
        raise DataReadinessError("A4.4 permits at most three paired candidates per hypothesis")
    return candidates


def _fit_opportunity(
    spec: _CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    config: IntradayDevelopmentConfig,
) -> Any:
    if spec.family == "ridge_logistic_pair":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                ("regressor", Ridge(alpha=float(spec.hyperparameters["alpha"]), solver="cholesky")),
            ]
        )
    elif spec.family == "hgb_pair":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            random_state=config.random_seed,
        )
    else:
        raise AssertionError(f"unsupported intraday opportunity family: {spec.family}")
    estimator.fit(features, target)
    return estimator


def _fit_downside(
    spec: _CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    config: IntradayDevelopmentConfig,
) -> Any:
    if len(np.unique(target)) != 2:
        raise DataReadinessError("downside fit requires both stop-hit classes")
    if spec.family == "ridge_logistic_pair":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                (
                    "classifier",
                    LogisticRegression(
                        C=float(spec.hyperparameters["downside_c"]),
                        max_iter=500,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    elif spec.family == "hgb_pair":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            random_state=config.random_seed,
        )
    else:
        raise AssertionError(f"unsupported intraday downside family: {spec.family}")
    estimator.fit(features, target)
    return estimator


def _fit_pair(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    features: np.ndarray,
    opportunity_target: np.ndarray,
    downside_target: np.ndarray,
    training_sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
    *,
    excluded_securities: frozenset[str] = frozenset(),
) -> _FittedPair:
    fit_sessions, calibration_sessions = _split_downside_calibration(
        data, training_sessions, config
    )
    training_mask = data["session_date_et"].isin(training_sessions).to_numpy(copy=True)
    if excluded_securities:
        training_mask &= ~data["security_id"].astype(str).isin(excluded_securities).to_numpy()
    downside_fit_mask = training_mask & data["session_date_et"].isin(fit_sessions).to_numpy()
    calibration_mask = training_mask & data["session_date_et"].isin(calibration_sessions).to_numpy()
    opportunity = _fit_opportunity(
        spec, features[training_mask], opportunity_target[training_mask], config
    )
    downside = _fit_downside(
        spec, features[downside_fit_mask], downside_target[downside_fit_mask], config
    )
    calibration_target = downside_target[calibration_mask]
    if len(np.unique(calibration_target)) != 2:
        raise DataReadinessError("downside calibration requires both stop-hit classes")
    raw = _raw_stop_logit(downside, features[calibration_mask])
    calibrator = LogisticRegression(C=1_000_000.0, max_iter=500, random_state=config.random_seed)
    calibrator.fit(raw.reshape(-1, 1), calibration_target)
    return _FittedPair(
        opportunity_estimator=opportunity,
        downside_estimator=downside,
        downside_calibrator=calibrator,
        fit_sessions=fit_sessions,
        calibration_sessions=calibration_sessions,
    )


def _split_downside_calibration(
    data: pd.DataFrame,
    training_sessions: tuple[str, ...],
    config: IntradayDevelopmentConfig,
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    calibration_count = max(
        config.minimum_calibration_sessions,
        math.ceil(len(training_sessions) * config.calibration_fraction),
    )
    calibration_start = len(training_sessions) - calibration_count
    fit_end = calibration_start - config.embargo_sessions
    if fit_end < 20:
        raise DataReadinessError("training history is too short for downside calibration")
    fit_sessions = training_sessions[:fit_end]
    calibration_sessions = training_sessions[calibration_start:]
    fit = data.loc[data["session_date_et"].isin(fit_sessions)]
    calibration = data.loc[data["session_date_et"].isin(calibration_sessions)]
    if (
        fit.empty
        or calibration.empty
        or fit["label_available_at_utc"].max() >= calibration["decision_time_utc"].min()
    ):
        raise DataReadinessError("downside calibration is not causally purged")
    return fit_sessions, calibration_sessions


def _raw_stop_logit(estimator: Any, features: np.ndarray) -> np.ndarray:
    probability = np.asarray(estimator.predict_proba(features)[:, 1], dtype="float64")
    probability = np.clip(probability, 1e-6, 1.0 - 1e-6)
    return np.log(probability / (1.0 - probability))


def _predict_pair(fitted: _FittedPair, features: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    opportunity = np.asarray(
        fitted.opportunity_estimator.predict(features), dtype="float64"
    )
    raw = _raw_stop_logit(fitted.downside_estimator, features)
    downside = np.asarray(
        fitted.downside_calibrator.predict_proba(raw.reshape(-1, 1))[:, 1],
        dtype="float64",
    )
    if not np.isfinite(opportunity).all() or not np.isfinite(downside).all():
        raise DataReadinessError("intraday paired scores must be finite")
    return opportunity, downside


def _walk_forward_predictions(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    features_full: np.ndarray,
    opportunity_target: np.ndarray,
    downside_target: np.ndarray,
    folds: tuple[_Fold, ...],
    security_holdout: frozenset[str],
    config: IntradayDevelopmentConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    evidence: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = (
            data["session_date_et"].isin(fold.train_sessions)
            & ~data["security_id"].astype(str).isin(security_holdout)
        ).to_numpy()
        max_label = data.loc[train_mask, "label_available_at_utc"].max()
        validation_mask = data["session_date_et"].isin(fold.validation_sessions).to_numpy()
        min_decision = data.loc[validation_mask, "decision_time_utc"].min()
        if max_label >= min_decision:
            raise DataReadinessError(f"fold {fold.fold} violates label-time purging")

        gc.collect()

        fitted = _fit_pair(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            fold.train_sessions,
            config,
            excluded_securities=security_holdout,
        )
        opportunity_score, stop_probability = _predict_pair(
            fitted, features_full[validation_mask]
        )

        keep_columns = (
            "session_date_et",
            "decision_group_id",
            "entry_time_utc",
            "exit_bar_end_utc",
            "security_id",
            "dataset_row_id",
            "ticker",
            "net_return",
            "gross_return",
            "spy_return",
            "qqq_return",
            "sector_return",
            "spy_excess_return",
            "qqq_excess_return",
            "sector_excess_return",
            "target_hit",
            "stop_hit",
            "entry_price",
            "stop_price",
        )
        validation = data.loc[validation_mask, keep_columns].copy()
        validation["predicted_net_return"] = opportunity_score
        validation["predicted_stop_probability"] = stop_probability
        validation["validation_scope"] = np.where(
            validation["security_id"].astype(str).isin(security_holdout),
            "unseen_security",
            "seen_security",
        )
        validation["fold"] = fold.fold
        evidence.append(validation)
        records.append(
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "embargo_sessions": list(fold.embargo_sessions),
                "downside_fit_sessions": len(fitted.fit_sessions),
                "downside_calibration_sessions": len(fitted.calibration_sessions),
                "last_downside_fit_session": fitted.fit_sessions[-1],
                "first_downside_calibration_session": fitted.calibration_sessions[0],
                "max_train_label_available_at_utc": pd.Timestamp(max_label).isoformat(),
                "min_validation_decision_time_utc": pd.Timestamp(min_decision).isoformat(),
                "role": "selection" if fold.fold < len(folds) - 1 else "development_confirmation",
            }
        )
        del fitted
        gc.collect()
    return pd.concat(evidence, ignore_index=True), records


def _evaluate_spec(
    spec: _CandidateSpec,
    scored: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    config: IntradayDevelopmentConfig,
    frozen_cost_bps: float,
) -> dict[str, Any]:
    confirmation_fold = int(scored["fold"].max())
    selection_rows = scored.loc[scored["fold"].lt(confirmation_fold)].copy()
    confirmation_rows = scored.loc[scored["fold"].eq(confirmation_fold)].copy()
    threshold_records: list[dict[str, Any]] = []
    for threshold in config.expected_net_return_thresholds_bps:
        for stop_threshold in config.maximum_stop_probability_thresholds:
            scopes = _evaluate_scopes(
                selection_rows,
                threshold,
                stop_threshold,
                config,
                frozen_cost_bps,
            )
            passed, reasons = _scope_gate_result(scopes)
            threshold_records.append(
                {
                    "threshold_bps": threshold,
                    "maximum_stop_probability": stop_threshold,
                    "selection_passed": passed,
                    "failed_gate_reasons": reasons,
                    "selection_scopes": scopes,
                }
            )
    passed_thresholds = [record for record in threshold_records if bool(record["selection_passed"])]
    selected = max(passed_thresholds, key=_threshold_selection_key) if passed_thresholds else None
    confirmation_scopes: dict[str, Any] | None = None
    confirmation_passed = False
    confirmation_reasons: list[str] = []
    if selected is not None:
        confirmation_scopes = _evaluate_scopes(
            confirmation_rows,
            float(selected["threshold_bps"]),
            float(selected["maximum_stop_probability"]),
            config,
            frozen_cost_bps,
        )
        confirmation_passed, confirmation_reasons = _scope_gate_result(
            confirmation_scopes
        )
    return {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "hyperparameters": dict(spec.hyperparameters),
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "target_hit_used_as_training_target": False,
        "folds": list(folds),
        "selection_policies": threshold_records,
        "selection_passed": selected is not None,
        "validation_passed": selected is not None and confirmation_passed,
        "selected_threshold_bps": float(selected["threshold_bps"]) if selected else None,
        "selected_maximum_stop_probability": (
            float(selected["maximum_stop_probability"]) if selected else None
        ),
        "selected_selection_scopes": selected["selection_scopes"] if selected else None,
        "confirmation_scopes": confirmation_scopes,
        "confirmation_policy_frozen_before_scoring": selected is not None,
        "failed_gate_reasons": (
            confirmation_reasons
            if selected is not None and not confirmation_passed
            else []
            if confirmation_passed
            else sorted(
                {
                    reason
                    for record in threshold_records
                    for reason in record["failed_gate_reasons"]
                }
            )
        ),
    }


def _evaluate_scopes(
    scored: pd.DataFrame,
    threshold_bps: float,
    maximum_stop_probability: float,
    config: IntradayDevelopmentConfig,
    frozen_cost_bps: float,
) -> dict[str, Any]:
    output: dict[str, Any] = {}
    for scope in ("seen_security", "unseen_security"):
        frame = scored.loc[scored["validation_scope"].eq(scope)].copy()
        metrics = _evaluate_policy(
            frame,
            threshold_bps,
            maximum_stop_probability,
            config,
            frozen_cost_bps,
        )
        passed, reasons = _scope_gates(metrics, config, scope=scope)
        output[scope] = {"passed": passed, "failed_gate_reasons": reasons, "metrics": metrics}
    return output


def _scope_gate_result(scopes: Mapping[str, Any]) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    for scope, raw in scopes.items():
        record = _object(raw, f"{scope} scope")
        reasons.extend(f"{scope}:{reason}" for reason in record["failed_gate_reasons"])
    return not reasons, reasons


def _evaluate_policy(
    scored: pd.DataFrame,
    threshold_bps: float,
    maximum_stop_probability: float,
    config: IntradayDevelopmentConfig,
    frozen_cost_bps: float,
) -> dict[str, Any]:
    if scored.empty:
        raise DataReadinessError("validation scope is empty")
    primary = _position_ledger(
        scored,
        threshold_bps,
        maximum_stop_probability,
        frozen_cost_bps,
        config,
    )
    rank = _economic_ranking_metrics(scored, config)
    bootstrap = _moving_block_bootstrap(
        primary["daily_returns"],
        samples=config.bootstrap_samples,
        block_sessions=config.bootstrap_block_sessions,
        seed=config.random_seed + int(round(threshold_bps * 10.0)),
    )
    daily_frame = pd.DataFrame(primary["daily_records"])
    benchmark_bootstrap = {
        name: _moving_block_mean_interval(
            daily_frame[f"{name}_excess_return"].to_numpy(dtype="float64"),
            samples=config.bootstrap_samples,
            block_sessions=config.bootstrap_block_sessions,
            seed=config.random_seed + offset,
        )
        for offset, name in enumerate(("spy", "qqq", "sector"), start=101)
    }
    cost_curve: list[dict[str, Any]] = []
    for cost_bps in config.cost_curve_bps:
        curve = _position_ledger(
            scored, threshold_bps, maximum_stop_probability, cost_bps, config
        )
        metrics = _ledger_metrics(curve)
        cost_curve.append(
            {
                "round_trip_cost_bps": cost_bps,
                **metrics,
                "daily_return_bootstrap_95_ci": _moving_block_bootstrap(
                    curve["daily_returns"],
                    samples=config.bootstrap_samples,
                    block_sessions=config.bootstrap_block_sessions,
                    seed=config.random_seed + 300 + int(round(cost_bps)),
                ),
            }
        )
    total_groups = int(scored["decision_group_id"].nunique())
    traded_groups = len(
        {str(row["decision_group_id"]) for row in primary["position_records"]}
    )
    return {
        "rows": int(len(scored)),
        "securities": int(scored["security_id"].nunique()),
        **_ledger_metrics(primary),
        **_predictive_metrics(scored),
        **rank,
        "threshold_bps": threshold_bps,
        "maximum_stop_probability": maximum_stop_probability,
        "frozen_round_trip_cost_bps": frozen_cost_bps,
        "moving_block_bootstrap_95_ci": bootstrap,
        "benchmark_excess_bootstrap_95_ci": benchmark_bootstrap,
        "cost_curve": cost_curve,
        "position_ledger_rows": primary["positions"],
        "daily_ledger_rows": primary["daily_rows"],
        "decision_groups": total_groups,
        "decision_groups_with_entries": traded_groups,
        "no_trade_decision_rate": 1.0 - traded_groups / total_groups if total_groups else 1.0,
        "drawdown_basis": "event_time_realized_equity_with_open_positions_marked_at_frozen_stop",
        "turnover_basis": "actual_entry_and_exit_notional_divided_by_average_daily_starting_equity",
    }


def _position_ledger(
    scored: pd.DataFrame,
    threshold_bps: float,
    maximum_stop_probability: float,
    cost_bps: float,
    config: IntradayDevelopmentConfig,
) -> dict[str, Any]:
    threshold = threshold_bps / 10_000.0
    equity = 1.0
    positions: list[dict[str, Any]] = []
    daily_rows: list[dict[str, Any]] = []
    equity_marks: list[float] = [equity]
    for session, session_frame in scored.groupby("session_date_et", sort=True, observed=True):
        start_equity = equity
        cash = equity
        open_positions: list[dict[str, Any]] = []
        cooldown: dict[str, pd.Timestamp] = {}

        for _, group in session_frame.groupby("decision_group_id", sort=True, observed=True):
            entry_times = pd.to_datetime(group["entry_time_utc"], utc=True).unique()
            if len(entry_times) != 1:
                raise DataReadinessError("decision group must have one executable entry time")
            entry_time = pd.Timestamp(entry_times[0])
            cash, equity = _close_due_positions(
                open_positions,
                cutoff=entry_time,
                cash=cash,
                cost_bps=cost_bps,
                cooldown_minutes=config.per_security_cooldown_minutes,
                cooldown=cooldown,
                completed=positions,
                equity_marks=equity_marks,
            )
            candidates = group.loc[
                group["predicted_net_return"].ge(threshold)
                & group["predicted_stop_probability"].le(maximum_stop_probability)
            ].sort_values(
                ["predicted_net_return", "security_id"], ascending=[False, True], kind="stable"
            )
            candidates = candidates.head(config.maximum_candidates_per_decision)
            for row in candidates.itertuples(index=False):
                security_id = str(row.security_id)
                if len(open_positions) >= config.maximum_concurrent_positions:
                    break
                if any(str(item["security_id"]) == security_id for item in open_positions):
                    continue
                if cooldown.get(security_id, pd.Timestamp.min.tz_localize("UTC")) > entry_time:
                    continue
                notional = min(config.position_weight * equity, cash)
                if notional <= 1e-12:
                    break
                cash -= notional
                open_positions.append(
                    {
                        "dataset_row_id": str(row.dataset_row_id),
                        "ticker": str(row.ticker),
                        "security_id": security_id,
                        "session_date_et": str(session),
                        "decision_group_id": str(row.decision_group_id),
                        "entry_time_utc": entry_time,
                        "exit_time_utc": pd.Timestamp(row.exit_bar_end_utc),
                        "predicted_net_return": float(row.predicted_net_return),
                        "predicted_stop_probability": float(row.predicted_stop_probability),
                        "gross_return": float(row.gross_return),
                        "spy_return": float(row.spy_return),
                        "qqq_return": float(row.qqq_return),
                        "sector_return": float(row.sector_return),
                        "entry_price": float(row.entry_price),
                        "stop_price": float(row.stop_price),
                        "fold": int(row.fold),
                        "notional": notional,
                        "entry_weight": notional / equity,
                        "round_trip_cost_bps": cost_bps,
                    }
                )
        cash, equity = _close_due_positions(
            open_positions,
            cutoff=None,
            cash=cash,
            cost_bps=cost_bps,
            cooldown_minutes=config.per_security_cooldown_minutes,
            cooldown=cooldown,
            completed=positions,
            equity_marks=equity_marks,
        )
        if open_positions:
            raise AssertionError("session ended with open intraday positions")
        end_equity = cash
        equity = end_equity
        session_positions = [
            row for row in positions if row["session_date_et"] == str(session)
        ]
        daily_rows.append(
            {
                "session_date_et": str(session),
                "fold": int(session_frame["fold"].iloc[0]),
                "starting_equity": start_equity,
                "ending_equity": end_equity,
                "daily_return": end_equity / start_equity - 1.0,
                "spy_excess_return": sum(
                    float(row["notional"]) * float(row["realized_spy_excess_return"])
                    for row in session_positions
                ) / start_equity,
                "qqq_excess_return": sum(
                    float(row["notional"]) * float(row["realized_qqq_excess_return"])
                    for row in session_positions
                ) / start_equity,
                "sector_excess_return": sum(
                    float(row["notional"]) * float(row["realized_sector_excess_return"])
                    for row in session_positions
                ) / start_equity,
                "entries": len(session_positions),
                "entry_notional": sum(float(row["notional"]) for row in session_positions),
            }
        )
        equity_marks.append(equity)
    daily_returns = np.asarray([float(row["daily_return"]) for row in daily_rows], dtype="float64")
    return {
        "position_records": positions,
        "daily_records": daily_rows,
        "positions": len(positions),
        "daily_rows": len(daily_rows),
        "daily_returns": daily_returns,
        "equity_marks": np.asarray(equity_marks, dtype="float64"),
    }


def _close_due_positions(
    open_positions: list[dict[str, Any]],
    *,
    cutoff: pd.Timestamp | None,
    cash: float,
    cost_bps: float,
    cooldown_minutes: int,
    cooldown: dict[str, pd.Timestamp],
    completed: list[dict[str, Any]],
    equity_marks: list[float],
) -> tuple[float, float]:
    due = [item for item in open_positions if cutoff is None or item["exit_time_utc"] <= cutoff]
    equity = cash + sum(_conservative_open_value(item, cost_bps) for item in open_positions)
    if open_positions:
        equity_marks.append(equity)
    exit_times = sorted({pd.Timestamp(item["exit_time_utc"]) for item in due})
    for exit_time in exit_times:
        batch = sorted(
            (item for item in due if pd.Timestamp(item["exit_time_utc"]) == exit_time),
            key=lambda value: str(value["security_id"]),
        )
        realized: list[tuple[dict[str, Any], float, float]] = []
        for item in batch:
            realized_return = float(item["gross_return"]) - cost_bps / 10_000.0
            pnl = float(item["notional"]) * realized_return
            cash += float(item["notional"]) + pnl
            open_positions.remove(item)
            cooldown[str(item["security_id"])] = exit_time + pd.Timedelta(
                minutes=cooldown_minutes
            )
            realized.append((item, realized_return, pnl))
        equity = cash + sum(
            _conservative_open_value(opened, cost_bps) for opened in open_positions
        )
        for item, realized_return, pnl in realized:
            item["realized_net_return"] = realized_return
            item["realized_spy_excess_return"] = realized_return - float(item["spy_return"])
            item["realized_qqq_excess_return"] = realized_return - float(item["qqq_return"])
            item["realized_sector_excess_return"] = realized_return - float(item["sector_return"])
            item["pnl"] = pnl
            item["equity_after_exit"] = equity
            completed.append(item)
        equity_marks.append(equity)
    return cash, equity


def _conservative_open_value(position: Mapping[str, Any], cost_bps: float) -> float:
    entry = float(position["entry_price"])
    stop = float(position["stop_price"])
    stop_return = stop / entry - 1.0 - cost_bps / 10_000.0
    return float(position["notional"]) * (1.0 + min(0.0, stop_return))


def _ledger_metrics(ledger: Mapping[str, Any]) -> dict[str, Any]:
    positions = list(ledger["position_records"])
    daily = list(ledger["daily_records"])
    returns = np.asarray(ledger["daily_returns"], dtype="float64")
    marks = np.asarray(ledger["equity_marks"], dtype="float64")
    notionals = np.asarray([float(row["notional"]) for row in positions], dtype="float64")
    pnls = np.asarray([float(row["pnl"]) for row in positions], dtype="float64")
    average_equity = float(np.mean([float(row["starting_equity"]) for row in daily])) if daily else 1.0
    profit = float(pnls[pnls > 0.0].sum()) if len(pnls) else 0.0
    loss = float(pnls[pnls < 0.0].sum()) if len(pnls) else 0.0
    running_peak = np.maximum.accumulate(marks)
    drawdown = 1.0 - marks / running_peak
    concurrency = _maximum_observed_concurrency(positions)
    sessions = len(daily)
    round_trip_turnover = (
        float(2.0 * notionals.sum() / average_equity) if average_equity > 0.0 else 0.0
    )
    compounded = float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0
    fold_means = (
        pd.DataFrame(daily).groupby("fold", observed=True)["daily_return"].mean()
        if daily
        else pd.Series(dtype="float64")
    )
    return {
        "trade_count": len(positions),
        "sessions": sessions,
        "sessions_with_trades": sum(int(row["entries"]) > 0 for row in daily),
        "average_trade_net_return": float(pnls.sum() / notionals.sum()) if notionals.sum() > 0.0 else 0.0,
        "average_daily_net_return": float(returns.mean()) if len(returns) else 0.0,
        "compounded_net_return": compounded,
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "profit_factor": profit / abs(loss) if loss < 0.0 else (1_000_000.0 if profit > 0.0 else 0.0),
        "win_rate": float((pnls > 0.0).mean()) if len(pnls) else 0.0,
        "negative_session_rate": float((returns < 0.0).mean()) if len(returns) else 1.0,
        "return_to_drawdown": compounded / float(drawdown.max()) if len(drawdown) and drawdown.max() > 0.0 else 0.0,
        "average_spy_excess_return": _notional_weighted_position_mean(
            positions, notionals, "realized_spy_excess_return"
        ),
        "average_qqq_excess_return": _notional_weighted_position_mean(
            positions, notionals, "realized_qqq_excess_return"
        ),
        "average_sector_excess_return": _notional_weighted_position_mean(
            positions, notionals, "realized_sector_excess_return"
        ),
        "one_way_turnover": float(notionals.sum() / average_equity) if average_equity > 0.0 else 0.0,
        "round_trip_turnover": round_trip_turnover,
        "average_daily_round_trip_turnover": round_trip_turnover / sessions if sessions else math.inf,
        "profitable_fold_fraction": float(fold_means.gt(0.0).mean()) if len(fold_means) else 0.0,
        "maximum_concurrent_positions_observed": concurrency,
        "maximum_entries_per_session_observed": max(
            (int(row["entries"]) for row in daily), default=0
        ),
        "maximum_entries_per_decision_observed": _maximum_entries_per_decision(positions),
        "maximum_entry_weight_observed": max(
            (float(row["entry_weight"]) for row in positions),
            default=0.0,
        ),
        "maximum_concurrent_positions_enforced": True,
        "capital_weights_enforced": True,
        "security_cooldown_enforced": True,
    }


def _maximum_entries_per_decision(positions: Sequence[Mapping[str, Any]]) -> int:
    counts: dict[tuple[str, str], int] = {}
    for position in positions:
        key = (str(position["session_date_et"]), str(position["decision_group_id"]))
        counts[key] = counts.get(key, 0) + 1
    return max(counts.values(), default=0)


def _notional_weighted_position_mean(
    positions: Sequence[Mapping[str, Any]],
    notionals: np.ndarray,
    column: str,
) -> float:
    if not positions or notionals.sum() <= 0.0:
        return 0.0
    values = np.asarray([float(row[column]) for row in positions], dtype="float64")
    return float(np.average(values, weights=notionals))


def _maximum_observed_concurrency(positions: Sequence[Mapping[str, Any]]) -> int:
    events: list[tuple[pd.Timestamp, int]] = []
    for position in positions:
        events.append((pd.Timestamp(position["entry_time_utc"]), 1))
        events.append((pd.Timestamp(position["exit_time_utc"]), -1))
    active = 0
    maximum = 0
    for _, delta in sorted(events, key=lambda item: (item[0], item[1])):
        active += delta
        maximum = max(maximum, active)
    return maximum


def _economic_ranking_metrics(
    scored: pd.DataFrame,
    config: IntradayDevelopmentConfig,
) -> dict[str, Any]:
    gains: list[tuple[str, float]] = []
    captures: list[float] = []
    for _, group in scored.groupby("decision_group_id", sort=False, observed=True):
        if len(group) < 2:
            continue
        depth = min(config.maximum_candidates_per_decision, len(group))
        actual = group["net_return"].to_numpy(dtype="float64")
        score = group["predicted_net_return"].to_numpy(dtype="float64")
        model_mean = float(actual[np.argsort(-score, kind="stable")[:depth]].mean())
        random_expected = float(actual.mean())
        ideal_mean = float(np.sort(actual)[-depth:].mean())
        gains.append((str(group["session_date_et"].iloc[0]), model_mean - random_expected))
        denominator = ideal_mean - random_expected
        captures.append((model_mean - random_expected) / denominator if denominator > 0.0 else 0.0)
    gain_values = np.asarray([value for _, value in gains], dtype="float64")
    session_gains = (
        pd.DataFrame(gains, columns=["session", "gain"])
        .groupby("session", sort=True, observed=True)["gain"]
        .mean()
        .to_numpy(dtype="float64")
        if gains
        else np.asarray([], dtype="float64")
    )
    gain_interval = _moving_block_mean_interval(
        session_gains,
        samples=config.bootstrap_samples,
        block_sessions=config.bootstrap_block_sessions,
        seed=config.random_seed + 211,
    )
    return {
        "ranking_groups": len(gains),
        "economic_rank_gain_over_exact_random_baseline": float(gain_values.mean()) if gains else 0.0,
        "economic_rank_gain_bootstrap_95_ci": gain_interval,
        "economic_rank_capture_ratio": float(np.mean(captures)) if captures else 0.0,
        "random_baseline_method": "exact_cross_sectional_mean_return",
        "raw_ndcg_reported": False,
    }


def _predictive_metrics(scored: pd.DataFrame) -> dict[str, Any]:
    positive = scored["net_return"].gt(0.0).astype("int8").to_numpy()
    stop = scored["stop_hit"].astype("int8").to_numpy()
    opportunity = scored["predicted_net_return"].to_numpy(dtype="float64")
    downside = scored["predicted_stop_probability"].to_numpy(dtype="float64")
    opportunity_auc = _binary_auc(positive, opportunity)
    stop_auc = _binary_auc(stop, downside)
    base_rate = float(positive.mean())
    depth = max(1, math.ceil(len(scored) * 0.10))
    top = positive[np.argsort(-opportunity, kind="stable")[:depth]]
    stop_prevalence = float(stop.mean())
    brier = float(brier_score_loss(stop, downside))
    baseline_brier = stop_prevalence * (1.0 - stop_prevalence)
    return {
        "positive_net_return_roc_auc": opportunity_auc,
        "positive_net_return_pr_auc": _binary_pr_auc(positive, opportunity),
        "positive_net_return_prevalence": base_rate,
        "top_decile_positive_net_return_rate": float(top.mean()),
        "top_decile_positive_net_return_lift": float(top.mean() / base_rate) if base_rate > 0.0 else None,
        "stop_hit_roc_auc": stop_auc,
        "stop_hit_pr_auc": _binary_pr_auc(stop, downside),
        "stop_hit_prevalence": stop_prevalence,
        "stop_hit_brier": brier,
        "stop_hit_baseline_brier": baseline_brier,
        "stop_hit_brier_skill": 1.0 - brier / baseline_brier if baseline_brier > 0.0 else None,
        "stop_hit_ece": _expected_calibration_error(stop, downside),
    }


def _binary_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return float(roc_auc_score(target, score)) if len(np.unique(target)) == 2 else None


def _binary_pr_auc(target: np.ndarray, score: np.ndarray) -> float | None:
    return (
        float(average_precision_score(target, score))
        if len(np.unique(target)) == 2
        else None
    )


def _expected_calibration_error(target: np.ndarray, probability: np.ndarray) -> float:
    edges = np.linspace(0.0, 1.0, 11)
    bins = np.clip(np.searchsorted(edges, probability, side="right") - 1, 0, 9)
    error = 0.0
    for index in range(10):
        selected = bins == index
        if selected.any():
            error += float(selected.mean()) * abs(
                float(probability[selected].mean()) - float(target[selected].mean())
            )
    return error


def _moving_block_mean_interval(
    values: np.ndarray,
    *,
    samples: int,
    block_sessions: int,
    seed: int,
) -> dict[str, float | int | None]:
    finite = np.asarray(values, dtype="float64")
    finite = finite[np.isfinite(finite)]
    if len(finite) < block_sessions * 2:
        return {
            "estimate": float(finite.mean()) if len(finite) else None,
            "low": None,
            "high": None,
            "sessions": len(finite),
        }
    starts = np.arange(0, len(finite) - block_sessions + 1)
    blocks_needed = math.ceil(len(finite) / block_sessions)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate(
            [finite[start : start + block_sessions] for start in chosen]
        )[: len(finite)]
        means[index] = float(sample.mean())
    return {
        "estimate": float(finite.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "sessions": len(finite),
    }


def _moving_block_bootstrap(
    daily_returns: np.ndarray,
    *,
    samples: int,
    block_sessions: int,
    seed: int,
) -> dict[str, Any]:
    values = np.asarray(daily_returns, dtype="float64")
    if len(values) < block_sessions * 2:
        raise DataReadinessError("moving-block bootstrap has insufficient daily portfolio returns")
    starts = np.arange(0, len(values) - block_sessions + 1)
    blocks_needed = math.ceil(len(values) / block_sessions)
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    compounded = np.empty(samples, dtype="float64")
    for index in range(samples):
        chosen = rng.choice(starts, size=blocks_needed, replace=True)
        sample = np.concatenate([values[start : start + block_sessions] for start in chosen])[: len(values)]
        means[index] = float(sample.mean())
        compounded[index] = float(np.prod(1.0 + sample) - 1.0)
    return {
        "estimand": "daily_capital_weighted_portfolio_return",
        "sessions": len(values),
        "block_sessions": block_sessions,
        "bootstrap_samples": samples,
        "average_daily_net_return": _interval(values.mean(), means),
        "compounded_net_return": _interval(np.prod(1.0 + values) - 1.0, compounded),
    }


def _interval(estimate: float, samples: np.ndarray) -> dict[str, float]:
    return {
        "estimate": float(estimate),
        "low": float(np.quantile(samples, 0.025)),
        "high": float(np.quantile(samples, 0.975)),
    }


def _scope_gates(
    metrics: Mapping[str, Any],
    config: IntradayDevelopmentConfig,
    *,
    scope: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    stop_auc_gate = (
        config.minimum_seen_stop_hit_roc_auc
        if scope == "seen_security"
        else config.minimum_unseen_stop_hit_roc_auc
    )
    lift_gate = (
        config.minimum_seen_positive_net_lift
        if scope == "seen_security"
        else config.minimum_unseen_positive_net_lift
    )
    daily_interval = _object(
        _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")[
            "average_daily_net_return"
        ],
        "daily interval",
    )
    rank_interval = _object(
        metrics["economic_rank_gain_bootstrap_95_ci"], "rank interval"
    )
    benchmark_intervals = _object(
        metrics["benchmark_excess_bootstrap_95_ci"], "benchmark intervals"
    )
    checks = (
        (int(metrics["rows"]) >= config.minimum_scope_rows, "insufficient_scope_rows"),
        (
            int(metrics["securities"]) >= config.minimum_scope_securities,
            "insufficient_scope_securities",
        ),
        (
            _optional_metric_at_least(
                metrics.get("positive_net_return_roc_auc"),
                config.minimum_positive_net_return_roc_auc,
            ),
            "positive_net_return_roc_auc_below_gate",
        ),
        (
            _optional_metric_at_least(
                metrics.get("top_decile_positive_net_return_lift"), lift_gate
            ),
            "positive_net_return_lift_below_gate",
        ),
        (
            _optional_metric_at_least(metrics.get("stop_hit_roc_auc"), stop_auc_gate),
            "stop_hit_roc_auc_below_gate",
        ),
        (
            float(metrics["stop_hit_brier"]) <= config.maximum_stop_hit_brier,
            "stop_hit_brier_above_gate",
        ),
        (
            float(metrics["stop_hit_ece"]) <= config.maximum_stop_hit_ece,
            "stop_hit_ece_above_gate",
        ),
        (
            _optional_metric_at_least(metrics.get("stop_hit_brier_skill"), 0.0),
            "stop_hit_brier_skill_not_positive",
        ),
        (int(metrics["trade_count"]) >= config.minimum_validation_trades, "insufficient_validation_trades"),
        (
            int(metrics["sessions_with_trades"]) >= config.minimum_validation_sessions_with_trades,
            "insufficient_sessions_with_trades",
        ),
        (
            float(metrics["average_trade_net_return"]) * 10_000.0
            >= config.minimum_average_trade_net_return_bps,
            "average_trade_net_return_below_gate",
        ),
        (
            float(metrics["average_daily_net_return"]) * 10_000.0
            >= config.minimum_average_daily_net_return_bps,
            "average_daily_net_return_below_gate",
        ),
        (
            _optional_metric_above(
                daily_interval.get("low"),
                config.minimum_daily_return_ci_low_bps / 10_000.0,
            ),
            "daily_return_confidence_bound_below_gate",
        ),
        (float(metrics["profit_factor"]) >= config.minimum_profit_factor, "profit_factor_below_gate"),
        (float(metrics["maximum_drawdown"]) <= config.maximum_drawdown, "drawdown_above_gate"),
        (
            _optional_metric_above(rank_interval.get("low"), 0.0),
            "economic_rank_gain_confidence_bound_below_random",
        ),
        (
            _optional_metric_above(
                _object(benchmark_intervals["spy"], "SPY interval").get("low"),
                config.minimum_average_spy_excess_bps / 10_000.0,
            ),
            "spy_excess_confidence_bound_below_gate",
        ),
        (
            _optional_metric_above(
                _object(benchmark_intervals["qqq"], "QQQ interval").get("low"),
                config.minimum_average_qqq_excess_bps / 10_000.0,
            ),
            "qqq_excess_confidence_bound_below_gate",
        ),
        (
            _optional_metric_above(
                _object(benchmark_intervals["sector"], "sector interval").get("low"),
                config.minimum_average_sector_excess_bps / 10_000.0,
            ),
            "sector_excess_confidence_bound_below_gate",
        ),
        (
            float(metrics["average_daily_round_trip_turnover"])
            <= config.maximum_round_trip_turnover,
            "turnover_above_gate",
        ),
        (
            float(metrics["profitable_fold_fraction"])
            >= config.minimum_profitable_fold_fraction,
            "fold_stability_below_gate",
        ),
        (
            float(metrics["negative_session_rate"])
            <= config.maximum_negative_session_rate,
            "negative_session_rate_above_gate",
        ),
        (
            float(metrics["return_to_drawdown"])
            >= config.minimum_return_to_drawdown,
            "return_to_drawdown_below_gate",
        ),
        (
            int(metrics["maximum_entries_per_decision_observed"])
            <= config.maximum_candidates_per_decision,
            "decision_entry_capacity_breached",
        ),
        (
            int(metrics["maximum_concurrent_positions_observed"])
            <= config.maximum_concurrent_positions,
            "concurrent_position_capacity_breached",
        ),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    stress = next(
        record
        for record in metrics["cost_curve"]
        if float(record["round_trip_cost_bps"]) == config.stress_cost_bps
    )
    stress_interval = _object(
        _object(stress["daily_return_bootstrap_95_ci"], "stress bootstrap")[
            "average_daily_net_return"
        ],
        "stress daily interval",
    )
    if not _optional_metric_above(
        stress_interval.get("low"),
        config.minimum_stress_average_daily_return_bps / 10_000.0,
    ):
        reasons.append("stress_cost_average_daily_return_below_gate")
    return not reasons, reasons


def _optional_metric_at_least(value: object, minimum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) >= minimum
    )


def _optional_metric_above(value: object, minimum: float) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
        and float(value) > minimum
    )


def _threshold_selection_key(
    record: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    scope_key = _scope_selection_key(
        _object(record["selection_scopes"], "selection scopes")
    )
    return (*scope_key, -float(record["threshold_bps"]), -float(record["maximum_stop_probability"]))


def _selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    scope_key = _scope_selection_key(
        _object(record["selected_selection_scopes"], "selected selection scopes")
    )
    return (*scope_key, str(record["candidate_id"]))


def _scope_selection_key(scopes: Mapping[str, Any]) -> tuple[float, float, float]:
    keys: list[tuple[float, float, float]] = []
    for scope in ("seen_security", "unseen_security"):
        metrics = _object(
            _object(scopes[scope], f"{scope} scope")["metrics"], f"{scope} metrics"
        )
        interval = _object(
            _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")[
                "average_daily_net_return"
            ],
            "daily interval",
        )
        keys.append(
            (
                float(interval["low"]),
                float(metrics["average_daily_net_return"]),
                float(metrics["economic_rank_gain_over_exact_random_baseline"]),
            )
        )
    return min(keys)


def _audit_policy_choice(
    records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
    *,
    preferred: Mapping[str, Any] | None = None,
) -> tuple[str, float, float, bool]:
    if selected is not None:
        return (
            str(selected["candidate_id"]),
            _required_finite_number(selected["selected_threshold_bps"], "selected_threshold_bps"),
            _required_finite_number(
                selected["selected_maximum_stop_probability"],
                "selected_maximum_stop_probability",
            ),
            True,
        )
    if preferred is not None:
        return (
            str(preferred["candidate_id"]),
            _required_finite_number(
                preferred["selected_threshold_bps"], "selected_threshold_bps"
            ),
            _required_finite_number(
                preferred["selected_maximum_stop_probability"],
                "selected_maximum_stop_probability",
            ),
            False,
        )
    choices: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for candidate in records:
        thresholds = candidate.get("selection_policies")
        if not isinstance(thresholds, list):
            raise DataReadinessError("validation candidate thresholds are invalid")
        choices.extend((candidate, _object(item, "threshold record")) for item in thresholds)
    if not choices:
        raise DataReadinessError("no validation policy is available for audit")
    candidate, threshold = max(choices, key=lambda item: _threshold_selection_key(item[1]))
    return (
        str(candidate["candidate_id"]),
        _required_finite_number(threshold["threshold_bps"], "threshold_bps"),
        _required_finite_number(
            threshold["maximum_stop_probability"], "maximum_stop_probability"
        ),
        False,
    )


def _scored_frame(
    data: pd.DataFrame,
    opportunity_score: np.ndarray,
    stop_probability: np.ndarray,
) -> pd.DataFrame:
    if (
        len(opportunity_score) != len(data)
        or len(stop_probability) != len(data)
        or not np.isfinite(opportunity_score).all()
        or not np.isfinite(stop_probability).all()
    ):
        raise DataReadinessError("paired intraday scores must be finite and row-aligned")
    scored = data.copy()
    scored["predicted_net_return"] = opportunity_score
    scored["predicted_stop_probability"] = stop_probability
    return scored


def _publish_development(
    output: Path,
    candidate: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
    ledger: Mapping[str, Any],
    validation_predictions: pd.DataFrame,
) -> None:
    files: dict[str, Any] = {}
    temporary = _temporary_output(output)
    try:
        if candidate is not None:
            joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_json(temporary / _EVALUATION_NAME, evaluation)
        _write_json(temporary / _MODEL_CARD_NAME, model_card)
        _write_ledger_files(temporary, ledger)
        validation_predictions.to_parquet(
            temporary / _VALIDATION_PREDICTIONS_NAME,
            index=False,
            compression="zstd",
        )
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        state = str(evaluation["status"])
        manifest = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state": state,
            "model_family": evaluation["model_family"],
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "baseline_profile_sha256": evaluation["baseline_profile_sha256"],
            "ordered_feature_sha256": evaluation["ordered_feature_sha256"],
            "dataset": evaluation["dataset"],
            "training_config_sha256": evaluation["training_config_sha256"],
            "future_holdout_opened": False,
            "test_access_count": 0,
            "files": files,
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        _write_json(
            temporary / _AUTHORITY_NAME,
            {
                "schema_version": AUTHORITY_SCHEMA_VERSION,
                "state": state,
                "model_family": evaluation["model_family"],
                "promotion_permitted": False,
                "manifest_path": _MANIFEST_NAME,
                "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
                "baseline_profile_sha256": evaluation["baseline_profile_sha256"],
                "ordered_feature_sha256": evaluation["ordered_feature_sha256"],
                "dataset_authority_sha256": _object(
                    evaluation["dataset"], "dataset"
                )["authority_sha256"],
            },
        )
        _finish_output(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_complete_intraday_development_output(directory: Path) -> dict[str, Any]:
    """Strictly replay one A4.4 candidate or no-candidate authority."""

    root = directory.resolve()
    authority = _read_json(root / _AUTHORITY_NAME, "development authority")
    manifest = _read_json(root / _MANIFEST_NAME, "development manifest")
    evaluation = _read_json(root / _EVALUATION_NAME, "development evaluation")
    model_card = _read_json(root / _MODEL_CARD_NAME, "development model card")
    state = str(evaluation.get("status", ""))
    if (
        state not in {"candidate", "no_candidate"}
        or authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION
        or manifest.get("schema_version") != MODEL_SCHEMA_VERSION
        or authority.get("state") != state
        or manifest.get("state") != state
        or authority.get("manifest_path") != _MANIFEST_NAME
        or authority.get("manifest_sha256") != file_sha256(root / _MANIFEST_NAME)
        or manifest.get("promotion_permitted") is not False
        or evaluation.get("promotion_permitted") is not False
        or model_card.get("promotion_permitted") is not False
        or evaluation.get("future_holdout_opened") is not False
        or int(evaluation.get("test_access_count", -1)) != 0
    ):
        raise DataReadinessError("A4.4 output authority identity is invalid")
    files = _object(manifest.get("files"), "development manifest files")
    expected = {
        _EVALUATION_NAME,
        _MODEL_CARD_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
        _VALIDATION_PREDICTIONS_NAME,
    }
    if state == "candidate":
        expected.add(_CANDIDATE_NAME)
    if set(files) != expected:
        raise DataReadinessError("A4.4 output file inventory differs")
    actual = {
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_file()
    }
    if actual != expected | {_MANIFEST_NAME, _AUTHORITY_NAME}:
        raise DataReadinessError("A4.4 output immutable file set differs")
    for name, raw in files.items():
        if Path(name).name != name:
            raise DataReadinessError("A4.4 output file path is invalid")
        record = _object(raw, f"development file {name}")
        path = (root / name).resolve()
        if root not in path.parents or not path.is_file():
            raise DataReadinessError(f"A4.4 output file is missing: {name}")
        if (
            int(record.get("bytes", -1)) != path.stat().st_size
            or record.get("sha256") != file_sha256(path)
        ):
            raise DataReadinessError(f"A4.4 output file identity failed: {name}")
    profile = _object(evaluation.get("baseline_profile"), "baseline profile")
    profile_identity = BaselineProfile(
        profile_id=str(profile.get("profile_id", "")),
        description=str(profile.get("description", "")),
        population_rule={
            str(key): _required_finite_number(value, f"population rule {key}")
            for key, value in _object(
                profile.get("population_rule"), "population rule"
            ).items()
        },
    )
    profile_sha256 = profile_identity.sha256()
    dataset = _object(evaluation.get("dataset"), "evaluation dataset")
    model_family = str(evaluation.get("model_family", ""))
    event_cohort = dataset.get("research_event_cohort")
    directional_event_families = {
        f"intraday_{subtype}_confirmed_research": subtype
        for subtype in DIRECTIONAL_EVENT_SUBTYPES
    }
    event_model = model_family == "intraday_event_confirmed_research"
    directional_subtype = directional_event_families.get(model_family)
    config_payload = _object(evaluation.get("training_config"), "training config")
    if (
        model_family
        not in {
            "intraday_technical",
            "intraday_event_confirmed_research",
            *directional_event_families,
        }
        or model_card.get("model_family") != model_family
        or manifest.get("model_family") != model_family
        or authority.get("model_family") != model_family
        or (
            (event_model or directional_subtype is not None)
            and (
                not isinstance(event_cohort, dict)
                or event_cohort.get("production_eligible") is not False
                or event_cohort.get("serving_eligible") is not False
                or event_cohort.get("future_holdout_opened") is not False
                or event_cohort.get("catalyst_role")
                != "confirmation_and_population_filter_not_model_feature"
                or (
                    directional_subtype is not None
                    and event_cohort.get("event_subtype") != directional_subtype
                )
            )
        )
        or (model_family == "intraday_technical" and event_cohort is not None)
        or evaluation.get("baseline_profile_sha256") != profile_sha256
        or model_card.get("baseline_profile_sha256") != profile_sha256
        or manifest.get("baseline_profile_sha256") != profile_sha256
        or authority.get("baseline_profile_sha256") != profile_sha256
        or evaluation.get("training_config_sha256") != _json_sha256(config_payload)
        or manifest.get("training_config_sha256") != evaluation.get("training_config_sha256")
        or evaluation.get("feature_columns") != list(MODEL_FEATURE_COLUMNS)
        or model_card.get("feature_columns") != list(MODEL_FEATURE_COLUMNS)
        or evaluation.get("ordered_feature_sha256")
        != dataset.get("ordered_feature_sha256")
        or model_card.get("ordered_feature_sha256")
        != dataset.get("ordered_feature_sha256")
        or manifest.get("ordered_feature_sha256")
        != dataset.get("ordered_feature_sha256")
        or authority.get("ordered_feature_sha256")
        != dataset.get("ordered_feature_sha256")
        or manifest.get("dataset") != dataset
        or authority.get("dataset_authority_sha256") != dataset.get("authority_sha256")
    ):
        raise DataReadinessError("A4.4 profile, dataset, or policy identity differs")
    config = IntradayDevelopmentConfig(**_tuple_config_values(config_payload))
    records = evaluation.get("validation_candidates")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("A4.4 validation records are unavailable")
    selected = next(
        (
            _object(record, "selected candidate")
            for record in records
            if record.get("candidate_id") == evaluation.get("selected_candidate_id")
        ),
        None,
    )
    selection_candidates = [
        _object(record, "selection candidate")
        for record in records
        if bool(record.get("selection_passed"))
    ]
    selection_winner = (
        max(selection_candidates, key=_selection_key)
        if selection_candidates
        else None
    )
    audit_candidate, threshold, stop_threshold, passed = _audit_policy_choice(
        [_object(record, "validation candidate") for record in records],
        selected,
        preferred=selection_winner,
    )
    audit = _object(evaluation.get("auditable_policy_ledger"), "audit ledger")
    if (
        audit.get("candidate_id") != audit_candidate
        or not math.isclose(float(audit.get("threshold_bps", math.nan)), threshold)
        or not math.isclose(
            float(audit.get("maximum_stop_probability", math.nan)), stop_threshold
        )
        or audit.get("validation_passed") is not passed
    ):
        raise DataReadinessError("A4.4 selected policy replay differs")
    predictions = pd.read_parquet(root / _VALIDATION_PREDICTIONS_NAME)
    source_record = next(
        _object(record, "audit candidate")
        for record in records
        if record.get("candidate_id") == audit_candidate
    )
    hyperparameters_raw = _object(
        source_record.get("hyperparameters"), "hyperparameters"
    )
    if any(
        isinstance(value, bool) or not isinstance(value, (int, float))
        for value in hyperparameters_raw.values()
    ):
        raise DataReadinessError("A4.4 hyperparameters must be numeric")
    spec = _CandidateSpec(
        candidate_id=audit_candidate,
        family=str(source_record.get("family", "")),
        hyperparameters=cast(dict[str, float | int], hyperparameters_raw),
    )
    folds_raw = source_record.get("folds")
    if not isinstance(folds_raw, list):
        raise DataReadinessError("A4.4 fold evidence is unavailable")
    replayed = _evaluate_spec(
        spec,
        predictions,
        cast(list[Mapping[str, Any]], folds_raw),
        config,
        _required_finite_number(
            _object(evaluation.get("dataset"), "dataset").get(
                "frozen_round_trip_cost_bps"
            ),
            "dataset frozen_round_trip_cost_bps",
        ),
    )
    if _json_sha256(replayed) != _json_sha256(source_record):
        raise DataReadinessError("A4.4 validation metrics do not replay")
    if state == "candidate":
        loaded = joblib.load(root / _CANDIDATE_NAME)
        if (
            not isinstance(loaded, dict)
            or loaded.get("validation_passed") is not True
            or loaded.get("model_family") != model_family
            or loaded.get("baseline_profile_sha256") != profile_sha256
            or loaded.get("dataset") != dataset
            or loaded.get("feature_columns") != list(MODEL_FEATURE_COLUMNS)
            or loaded.get("ordered_feature_sha256")
            != dataset.get("ordered_feature_sha256")
            or loaded.get("training_config_sha256")
            != evaluation.get("training_config_sha256")
        ):
            raise DataReadinessError("A4.4 candidate payload identity differs")
    return {
        "state": state,
        "baseline_profile_sha256": profile_sha256,
        "dataset": dataset,
        "selected_candidate_id": evaluation.get("selected_candidate_id"),
        "manifest_sha256": file_sha256(root / _MANIFEST_NAME),
        "authority_sha256": file_sha256(root / _AUTHORITY_NAME),
    }


def load_complete_intraday_future_evaluation_output(directory: Path) -> Mapping[str, Any]:
    """Verify immutable future evidence and replay ledger-derived economics."""

    root = directory.resolve()
    expected_files = {
        _AUTHORITY_NAME,
        _MANIFEST_NAME,
        _FUTURE_EVALUATION_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
    }
    actual_entries = {path.name for path in root.iterdir()}
    if actual_entries != expected_files:
        raise DataReadinessError("future evidence exact-file inventory differs")
    authority = _read_json(root / _AUTHORITY_NAME, "future authority")
    manifest = _read_json(root / _MANIFEST_NAME, "future manifest")
    if (
        authority.get("schema_version") != FUTURE_AUTHORITY_SCHEMA_VERSION
        or authority.get("state") != "locked_future_evaluated"
        or manifest.get("schema_version") != FUTURE_EVALUATION_SCHEMA_VERSION
        or manifest.get("state") != "locked_future_evaluated"
    ):
        raise DataReadinessError("future evidence schema or state differs")
    if authority.get("manifest_sha256") != file_sha256(root / _MANIFEST_NAME):
        raise DataReadinessError("future authority does not bind its manifest")
    files = _object(manifest.get("files"), "future manifest files")
    expected_evidence = {
        _FUTURE_EVALUATION_NAME,
        _POSITION_LEDGER_NAME,
        _DAILY_LEDGER_NAME,
    }
    if set(files) != expected_evidence:
        raise DataReadinessError("future manifest evidence inventory differs")
    for name, raw in files.items():
        record = _object(raw, f"future file {name}")
        path = root / name
        if (
            record.get("sha256") != file_sha256(path)
            or int(record.get("bytes", -1)) != path.stat().st_size
        ):
            raise DataReadinessError(f"future evidence identity failed: {name}")
    evaluation = _read_json(root / _FUTURE_EVALUATION_NAME, "future evaluation")
    if (
        evaluation.get("schema_version") != FUTURE_EVALUATION_SCHEMA_VERSION
        or evaluation.get("status") != "locked_future_evaluated"
        or evaluation.get("selection_changed_after_future_observation") is not False
    ):
        raise DataReadinessError("future evaluation contract differs")
    metrics = _object(evaluation.get("metrics"), "future metrics")
    positions = pd.read_parquet(root / _POSITION_LEDGER_NAME)
    daily = pd.read_parquet(root / _DAILY_LEDGER_NAME)
    if int(metrics.get("position_ledger_rows", -1)) != len(positions):
        raise DataReadinessError("future position ledger row count differs")
    if int(metrics.get("daily_ledger_rows", -1)) != len(daily):
        raise DataReadinessError("future daily ledger row count differs")
    daily_returns = daily["daily_return"].to_numpy(dtype="float64")
    replay = {
        "average_daily_net_return": float(daily_returns.mean()) if len(daily_returns) else 0.0,
        "compounded_net_return": (
            float(np.prod(1.0 + daily_returns) - 1.0) if len(daily_returns) else 0.0
        ),
        "negative_session_rate": (
            float((daily_returns < 0.0).mean()) if len(daily_returns) else 1.0
        ),
        "maximum_entries_per_session_observed": (
            int(daily["entries"].max()) if len(daily) else 0
        ),
    }
    notionals = positions["notional"].to_numpy(dtype="float64")
    pnls = positions["pnl"].to_numpy(dtype="float64")
    replay["average_trade_net_return"] = (
        float(pnls.sum() / notionals.sum()) if notionals.sum() > 0.0 else 0.0
    )
    for name, expected in replay.items():
        actual = _required_finite_number(metrics.get(name), f"future metric {name}")
        if not math.isclose(actual, expected, rel_tol=0.0, abs_tol=1e-12):
            raise DataReadinessError(f"future metric does not replay: {name}")
    for identity in (
        "candidate_authority_sha256",
        "candidate_manifest_sha256",
        "future_access_lock_sha256",
    ):
        value = evaluation.get(identity)
        if not isinstance(value, str) or len(value) != 64:
            raise DataReadinessError(f"future evaluation {identity} is invalid")
    _object(evaluation.get("future_dataset"), "future dataset identity")
    return evaluation


def _publish_future_evaluation(
    output: Path,
    evaluation: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> None:
    temporary = _temporary_output(output)
    try:
        _write_json(temporary / _FUTURE_EVALUATION_NAME, evaluation)
        _write_ledger_files(temporary, ledger)
        evidence_files = (_FUTURE_EVALUATION_NAME, _POSITION_LEDGER_NAME, _DAILY_LEDGER_NAME)
        manifest = {
            "schema_version": FUTURE_EVALUATION_SCHEMA_VERSION,
            "state": "locked_future_evaluated",
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": {
                name: {
                    "sha256": file_sha256(temporary / name),
                    "bytes": (temporary / name).stat().st_size,
                }
                for name in evidence_files
            },
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        _write_json(
            temporary / _AUTHORITY_NAME,
            {
                "schema_version": FUTURE_AUTHORITY_SCHEMA_VERSION,
                "state": "locked_future_evaluated",
                "manifest_path": _MANIFEST_NAME,
                "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
            },
        )
        _finish_output(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _load_validation_passed_candidate(directory: Path) -> tuple[dict[str, Any], dict[str, Any]]:
    replay = load_complete_intraday_development_output(directory)
    if replay["state"] != "candidate":
        raise DataReadinessError("future holdout is locked until validation publishes a candidate")
    authority = _read_json(directory / _AUTHORITY_NAME, "candidate authority")
    manifest = _read_json(directory / _MANIFEST_NAME, "candidate manifest")
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise DataReadinessError("future evaluation accepts only A4.4 bar-baseline authorities")
    if authority.get("state") != "candidate" or manifest.get("state") != "candidate":
        raise DataReadinessError("future holdout is locked until validation publishes a candidate")
    if authority.get("manifest_sha256") != file_sha256(directory / _MANIFEST_NAME):
        raise DataReadinessError("candidate authority does not bind its manifest")
    files = _object(manifest.get("files"), "candidate manifest files")
    for name, raw in files.items():
        record = _object(raw, f"candidate file {name}")
        path = directory / str(name)
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"candidate file identity failed: {name}")
    if _CANDIDATE_NAME not in files:
        raise DataReadinessError("validation-passed candidate model is absent")
    loaded = joblib.load(directory / _CANDIDATE_NAME)
    if not isinstance(loaded, dict) or loaded.get("validation_passed") is not True:
        raise DataReadinessError("future holdout is locked until validation passes")
    return loaded, manifest


def _temporary_output(output: Path) -> Path:
    output.parent.mkdir(parents=True, exist_ok=True)
    return Path(tempfile.mkdtemp(prefix=f".{output.name}.", dir=output.parent))


def _require_output_isolated(output: Path, *inputs: Path) -> None:
    target = output.resolve()
    for immutable_input in inputs:
        source = immutable_input.resolve()
        if target == source or target in source.parents or source in target.parents:
            raise DataReadinessError("output overlaps an immutable input authority")
    if output.exists():
        raise FileExistsError(f"immutable output already exists: {output}")


def _consume_future_access(
    candidate: Path,
    future_dataset: Path,
    registry_directory: Path,
) -> Path:
    candidate_authority = candidate / _AUTHORITY_NAME
    candidate_authority_sha256 = file_sha256(candidate_authority)
    registry = registry_directory.expanduser().resolve()
    registry.mkdir(parents=True, exist_ok=True)
    lock = registry / f"{candidate_authority_sha256}.json"
    payload = {
        "schema_version": "edge_rebuild.intraday_future_access.v1",
        "candidate_authority_sha256": candidate_authority_sha256,
        "future_dataset_directory": str(future_dataset.resolve()),
        "future_dataset_authority_sha256": file_sha256(future_dataset / _AUTHORITY_NAME),
        "accessed_at_utc": datetime.now(UTC).isoformat(),
    }
    try:
        with lock.open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, indent=2, sort_keys=True, allow_nan=False)
            stream.write("\n")
    except FileExistsError:
        raise DataReadinessError("future holdout access was already consumed") from None
    return lock


def _write_ledger_files(directory: Path, ledger: Mapping[str, Any]) -> None:
    positions = ledger.get("position_records")
    daily = ledger.get("daily_records")
    if not isinstance(positions, list) or not isinstance(daily, list):
        raise DataReadinessError("portfolio ledger records are unavailable")
    pd.DataFrame(positions).to_parquet(directory / _POSITION_LEDGER_NAME, index=False)
    pd.DataFrame(daily).to_parquet(directory / _DAILY_LEDGER_NAME, index=False)


def _finish_output(temporary: Path, output: Path) -> None:
    try:
        temporary.rename(output)
    except FileExistsError:
        raise FileExistsError(f"immutable output already exists: {output}") from None


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _read_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataReadinessError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is invalid") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return value


def _dataset_identity(published: PublishedIntradayDataset) -> dict[str, Any]:
    return {
        "dataset_sha256": published.dataset_sha256,
        "manifest_sha256": published.manifest_sha256,
        "authority_sha256": published.authority_sha256,
        "request_sha256": published.request_sha256,
        "transformation_sha256": published.transformation_sha256,
        "session_unit_inventory_sha256": published.session_unit_inventory_sha256,
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "strategy_contract_sha256": published.strategy_contract_sha256,
        "frozen_round_trip_cost_bps": published.frozen_round_trip_cost_bps,
    }


def _gate_contract(config: IntradayDevelopmentConfig) -> dict[str, Any]:
    return {
        "minimum_scope_rows": config.minimum_scope_rows,
        "minimum_scope_securities": config.minimum_scope_securities,
        "minimum_positive_net_return_roc_auc": config.minimum_positive_net_return_roc_auc,
        "minimum_seen_positive_net_lift": config.minimum_seen_positive_net_lift,
        "minimum_unseen_positive_net_lift": config.minimum_unseen_positive_net_lift,
        "minimum_seen_stop_hit_roc_auc": config.minimum_seen_stop_hit_roc_auc,
        "minimum_unseen_stop_hit_roc_auc": config.minimum_unseen_stop_hit_roc_auc,
        "maximum_stop_hit_brier": config.maximum_stop_hit_brier,
        "maximum_stop_hit_ece": config.maximum_stop_hit_ece,
        "minimum_stop_hit_brier_skill": 0.0,
        "minimum_validation_trades": config.minimum_validation_trades,
        "minimum_validation_sessions_with_trades": config.minimum_validation_sessions_with_trades,
        "minimum_average_trade_net_return_bps": config.minimum_average_trade_net_return_bps,
        "minimum_average_daily_net_return_bps": config.minimum_average_daily_net_return_bps,
        "minimum_daily_return_ci_low_bps": config.minimum_daily_return_ci_low_bps,
        "minimum_profit_factor": config.minimum_profit_factor,
        "minimum_economic_rank_gain_bps": config.minimum_economic_rank_gain_bps,
        "minimum_average_spy_excess_bps": config.minimum_average_spy_excess_bps,
        "minimum_average_qqq_excess_bps": config.minimum_average_qqq_excess_bps,
        "minimum_average_sector_excess_bps": config.minimum_average_sector_excess_bps,
        "maximum_drawdown": config.maximum_drawdown,
        "maximum_round_trip_turnover": config.maximum_round_trip_turnover,
        "minimum_profitable_fold_fraction": config.minimum_profitable_fold_fraction,
        "maximum_negative_session_rate": config.maximum_negative_session_rate,
        "minimum_return_to_drawdown": config.minimum_return_to_drawdown,
        "maximum_entries_per_decision": config.maximum_candidates_per_decision,
        "maximum_concurrent_positions": config.maximum_concurrent_positions,
        "stress_cost_bps": config.stress_cost_bps,
        "minimum_stress_average_daily_return_bps": config.minimum_stress_average_daily_return_bps,
    }


def _future_data_contract(config: IntradayDevelopmentConfig) -> dict[str, Any]:
    return {
        "development_end_date": config.development_end_date,
        "minimum_session_date": config.future_holdout_start_date,
        "minimum_sessions": config.minimum_validation_sessions,
        "minimum_rows": config.minimum_rows,
        "minimum_securities": config.minimum_securities,
        "required_timeframe": "1Min",
        "required_price_feed": "sip",
        "required_adjustment": "all",
        "future_access_registry_directory": str(_resolved_future_access_registry(config)),
        "selection_must_remain_frozen": True,
    }


def _resolved_future_access_registry(config: IntradayDevelopmentConfig) -> Path:
    configured = Path(config.future_access_registry_directory).expanduser()
    if configured.is_absolute():
        return configured.resolve()
    return (Path.home() / ".market-predictor" / configured).resolve()


def _tuple_config_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    for name in (
        "expected_net_return_thresholds_bps",
        "maximum_stop_probability_thresholds",
        "ridge_alphas",
        "logistic_c_values",
        "hgb_learning_rates",
        "hgb_max_leaf_nodes",
        "cost_curve_bps",
    ):
        values[name] = tuple(values[name])
    return values


def _guard_memory(config: IntradayDevelopmentConfig, stage: str, *, peak: bool) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    if peak:
        assert_peak_memory_budget(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
            stage=stage,
        )


def _json_sha256(payload: Mapping[str, Any]) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _parse_date(value: str, label: str) -> date:
    try:
        return date.fromisoformat(value)
    except ValueError as exc:
        raise ValueError(f"{label} must be ISO date") from exc


def _required_finite_number(value: Any, label: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"{label} must be numeric")
    result = float(value)
    if not math.isfinite(result):
        raise DataReadinessError(f"{label} must be finite")
    return result


def _strict_bool(value: Any) -> bool:
    return value is True or isinstance(value, np.bool_) and bool(value)


def _object(value: Any, label: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise DataReadinessError(f"{label} must be an object")
    return {str(key): item for key, item in value.items()}
