from __future__ import annotations

from market_predictor.intraday.evaluation.economics import _economic_ranking_metrics, _ledger_metrics, _position_ledger
from market_predictor.intraday.evaluation.metrics import _moving_block_bootstrap, _moving_block_mean_interval, _predictive_metrics

from market_predictor.intraday.training.io import (
    _dataset_identity,
    _json_sha256,
    _load_validation_passed_candidate,
    _object,
    _parse_date,
    _publish_future_evaluation,
    _required_finite_number,
    _tuple_config_values,
    load_complete_intraday_future_evaluation_output,
)
from market_predictor.intraday.training.models import _raw_stop_logit

"""Development-only, cost-aware intraday model training and evaluation."""

import math
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import date
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.training.training import (
    MODEL_FEATURE_COLUMNS,
    PublishedIntradayDataset,
    load_published_intraday_dataset,
)

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
        if not self.expected_net_return_thresholds_bps or any(value < 0.0 for value in self.expected_net_return_thresholds_bps):
            raise ValueError("expected-return thresholds must be non-negative")
        if tuple(sorted(set(self.expected_net_return_thresholds_bps))) != self.expected_net_return_thresholds_bps:
            raise ValueError("expected-return thresholds must be unique and ordered")
        if (
            not self.maximum_stop_probability_thresholds
            or any(not 0.0 < value < 1.0 for value in self.maximum_stop_probability_thresholds)
            or tuple(sorted(set(self.maximum_stop_probability_thresholds))) != self.maximum_stop_probability_thresholds
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
            & data["session_vwap_distance_five_minute_atr"].ge(profile.population_rule["session_vwap_distance_five_minute_atr_gte"])
        )
    if profile.profile_id == "intraday_bar_long_reversion_v1":
        return (
            data["stock_return_20m"].lt(profile.population_rule["stock_return_20m_lt"])
            & data["session_vwap_distance_five_minute_atr"].le(profile.population_rule["session_vwap_distance_five_minute_atr_lte"])
            & data["volume_rsi_14"].le(profile.population_rule["volume_rsi_14_lte"])
        )
    raise DataReadinessError("intraday baseline profile identity is unsupported")


def evaluate_future_intraday_holdout(
    candidate_authority_directory: Path,
    future_dataset_authority_directory: Path,
    output_directory: Path,
) -> Mapping[str, Any]:
    """Evaluate one accepted development policy on a separate future authority.

    Candidate validation is verified before the future path is inspected. The
    future dataset is rejected unless every row starts on/after the frozen date.
    """

    from market_predictor.intraday.training.coordinator import _validate_development_frame
    from market_predictor.intraday.training.io import _consume_future_access, _finish_output, _require_output_isolated, _temporary_output, _write_json

    candidate, manifest = _load_validation_passed_candidate(candidate_authority_directory)
    if candidate.get("model_family") != "intraday_technical":
        raise DataReadinessError("research-only event-confirmed candidates cannot open the future holdout")
    contract = _object(candidate.get("future_data_contract"), "future_data_contract")
    future_start = _parse_date(str(contract.get("minimum_session_date")), "minimum_session_date")
    development_end = _parse_date(str(contract.get("development_end_date")), "development_end_date")
    policy = IntradayDevelopmentConfig(**_tuple_config_values(_object(candidate.get("training_config"), "training_config")))
    _require_output_isolated(
        output_directory,
        candidate_authority_directory,
        future_dataset_authority_directory,
    )
    if not future_dataset_authority_directory.is_dir():
        raise DataReadinessError(f"future holdout data does not exist; collect sessions from {future_start.isoformat()} onward")
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
            raise DataReadinessError(f"future holdout {identity_key} differs from development")
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
            for key, value in _object(profile_raw.get("population_rule"), "population_rule").items()
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
        raise DataReadinessError(f"future holdout has {actual_sessions} complete sessions; requires {minimum_sessions}")
    if minimum_rows < 1 or actual_rows < minimum_rows:
        raise DataReadinessError(f"future holdout has {actual_rows} profile rows; requires {minimum_rows}")
    if minimum_securities < 2 or actual_securities < minimum_securities:
        raise DataReadinessError(f"future holdout has {actual_securities} profile securities; requires {minimum_securities}")
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
    stop_probability = np.asarray(calibrator.predict_proba(raw_stop.reshape(-1, 1))[:, 1], dtype="float64")
    scored = _scored_frame(data, opportunity_score, stop_probability)
    scored["fold"] = 0
    scored["validation_scope"] = "future_holdout"
    threshold = _required_finite_number(
        candidate.get("expected_net_return_threshold_bps"),
        "expected_net_return_threshold_bps",
    )
    stop_threshold = _required_finite_number(candidate.get("maximum_stop_probability"), "maximum_stop_probability")
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
        confirmation_passed, confirmation_reasons = _scope_gate_result(confirmation_scopes)
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
        "selected_maximum_stop_probability": (float(selected["maximum_stop_probability"]) if selected else None),
        "selected_selection_scopes": selected["selection_scopes"] if selected else None,
        "confirmation_scopes": confirmation_scopes,
        "confirmation_policy_frozen_before_scoring": selected is not None,
        "failed_gate_reasons": (
            confirmation_reasons
            if selected is not None and not confirmation_passed
            else []
            if confirmation_passed
            else sorted({reason for record in threshold_records for reason in record["failed_gate_reasons"]})
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
        curve = _position_ledger(scored, threshold_bps, maximum_stop_probability, cost_bps, config)
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
    traded_groups = len({str(row["decision_group_id"]) for row in primary["position_records"]})
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


def _scope_gates(
    metrics: Mapping[str, Any],
    config: IntradayDevelopmentConfig,
    *,
    scope: str,
) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    stop_auc_gate = config.minimum_seen_stop_hit_roc_auc if scope == "seen_security" else config.minimum_unseen_stop_hit_roc_auc
    lift_gate = config.minimum_seen_positive_net_lift if scope == "seen_security" else config.minimum_unseen_positive_net_lift
    daily_interval = _object(
        _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")["average_daily_net_return"],
        "daily interval",
    )
    rank_interval = _object(metrics["economic_rank_gain_bootstrap_95_ci"], "rank interval")
    benchmark_intervals = _object(metrics["benchmark_excess_bootstrap_95_ci"], "benchmark intervals")
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
            _optional_metric_at_least(metrics.get("top_decile_positive_net_return_lift"), lift_gate),
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
            float(metrics["average_trade_net_return"]) * 10_000.0 >= config.minimum_average_trade_net_return_bps,
            "average_trade_net_return_below_gate",
        ),
        (
            float(metrics["average_daily_net_return"]) * 10_000.0 >= config.minimum_average_daily_net_return_bps,
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
            float(metrics["average_daily_round_trip_turnover"]) <= config.maximum_round_trip_turnover,
            "turnover_above_gate",
        ),
        (
            float(metrics["profitable_fold_fraction"]) >= config.minimum_profitable_fold_fraction,
            "fold_stability_below_gate",
        ),
        (
            float(metrics["negative_session_rate"]) <= config.maximum_negative_session_rate,
            "negative_session_rate_above_gate",
        ),
        (
            float(metrics["return_to_drawdown"]) >= config.minimum_return_to_drawdown,
            "return_to_drawdown_below_gate",
        ),
        (
            int(metrics["maximum_entries_per_decision_observed"]) <= config.maximum_candidates_per_decision,
            "decision_entry_capacity_breached",
        ),
        (
            int(metrics["maximum_concurrent_positions_observed"]) <= config.maximum_concurrent_positions,
            "concurrent_position_capacity_breached",
        ),
    )
    for passed, reason in checks:
        if not passed:
            reasons.append(reason)
    stress = next(record for record in metrics["cost_curve"] if float(record["round_trip_cost_bps"]) == config.stress_cost_bps)
    stress_interval = _object(
        _object(stress["daily_return_bootstrap_95_ci"], "stress bootstrap")["average_daily_net_return"],
        "stress daily interval",
    )
    if not _optional_metric_above(
        stress_interval.get("low"),
        config.minimum_stress_average_daily_return_bps / 10_000.0,
    ):
        reasons.append("stress_cost_average_daily_return_below_gate")
    return not reasons, reasons


def _optional_metric_at_least(value: object, minimum: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) >= minimum


def _optional_metric_above(value: object, minimum: float) -> bool:
    return not isinstance(value, bool) and isinstance(value, (int, float)) and math.isfinite(float(value)) and float(value) > minimum


def _threshold_selection_key(
    record: Mapping[str, Any],
) -> tuple[float, float, float, float, float]:
    scope_key = _scope_selection_key(_object(record["selection_scopes"], "selection scopes"))
    return (*scope_key, -float(record["threshold_bps"]), -float(record["maximum_stop_probability"]))


def _selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    scope_key = _scope_selection_key(_object(record["selected_selection_scopes"], "selected selection scopes"))
    return (*scope_key, str(record["candidate_id"]))


def _scope_selection_key(scopes: Mapping[str, Any]) -> tuple[float, float, float]:
    keys: list[tuple[float, float, float]] = []
    for scope in ("seen_security", "unseen_security"):
        metrics = _object(_object(scopes[scope], f"{scope} scope")["metrics"], f"{scope} metrics")
        interval = _object(
            _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")["average_daily_net_return"],
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
            _required_finite_number(preferred["selected_threshold_bps"], "selected_threshold_bps"),
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
        _required_finite_number(threshold["maximum_stop_probability"], "maximum_stop_probability"),
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
