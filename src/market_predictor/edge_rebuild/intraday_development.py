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
from typing import Any, Final

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_training import (
    MODEL_FEATURE_COLUMNS,
    PublishedIntradayDataset,
    load_published_intraday_dataset,
)
from market_predictor.resources import assert_memory_budget, assert_peak_memory_budget
from market_predictor.v3.errors import DataReadinessError

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_development_candidate.v3"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_development_evaluation.v3"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_development_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"


@dataclass(frozen=True, slots=True)
class IntradayDevelopmentConfig:
    """Frozen development policy. Future observations are not an input."""

    development_end_date: str = "2026-07-08"
    future_holdout_start_date: str = "2026-07-09"
    validation_folds: int = 3
    minimum_train_sessions: int = 120
    minimum_validation_sessions: int = 40
    embargo_sessions: int = 1
    maximum_label_horizon_minutes: int = 30
    minimum_rows: int = 1_000
    minimum_securities: int = 20
    maximum_candidates_per_decision: int = 10
    maximum_concurrent_positions: int = 10
    position_weight: float = 0.10
    per_security_cooldown_minutes: int = 30
    expected_net_return_thresholds_bps: tuple[float, ...] = (0.0, 2.0, 5.0, 10.0, 15.0)
    ridge_alphas: tuple[float, ...] = (1.0, 10.0)
    hgb_learning_rates: tuple[float, ...] = (0.05,)
    hgb_max_leaf_nodes: tuple[int, ...] = (15, 31)
    hgb_max_iter: int = 150
    hgb_max_bins: int = 127
    bootstrap_samples: int = 2_000
    bootstrap_block_sessions: int = 5
    random_seed: int = 42
    minimum_validation_trades: int = 200
    minimum_validation_sessions_with_trades: int = 40
    minimum_average_trade_net_return_bps: float = 2.0
    minimum_average_daily_net_return_bps: float = 0.0
    minimum_daily_return_ci_low_bps: float = 0.0
    minimum_profit_factor: float = 1.05
    minimum_economic_rank_gain_bps: float = 0.0
    maximum_drawdown: float = 0.15
    stress_cost_bps: float = 20.0
    minimum_stress_average_daily_return_bps: float = 0.0
    cost_curve_bps: tuple[float, ...] = (0.0, 5.0, 10.0, 20.0)
    maximum_process_memory_gib: float = 5.0
    memory_guard_headroom_gib: float = 0.75

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
        if not self.ridge_alphas or any(value <= 0.0 for value in self.ridge_alphas):
            raise ValueError("ridge alphas must be positive")
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
        if self.minimum_profit_factor < 1.0 or not 0.0 < self.maximum_drawdown < 1.0:
            raise ValueError("profit-factor or drawdown gate is invalid")
        if self.stress_cost_bps not in self.cost_curve_bps or any(value < 0.0 for value in self.cost_curve_bps):
            raise ValueError("cost curve must contain the configured stress cost")
        if tuple(sorted(set(self.cost_curve_bps))) != self.cost_curve_bps:
            raise ValueError("cost curve must be unique and ordered")
        if not 0.0 < self.maximum_process_memory_gib <= 5.0:
            raise ValueError("process memory hard limit must be in (0, 5] GiB")
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
        "ridge_alphas",
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


def train_intraday_development_candidate(
    dataset_authority_directory: Path,
    output_directory: Path,
    *,
    config: IntradayDevelopmentConfig | None = None,
) -> DevelopmentTrainingResult:
    """Use development data only and publish candidate or no-candidate evidence.

    This function has no future-authority parameter. It rejects observations
    after the frozen development boundary before fitting any estimator.
    """

    policy = config or IntradayDevelopmentConfig()
    _guard_memory(policy, "intraday development start", peak=False)
    published = load_published_intraday_dataset(dataset_authority_directory)
    data = _validate_development_frame(published, policy)
    sessions = _ordered_sessions(data)
    folds = _walk_forward_folds(data, sessions, policy)
    # Keep one compact feature matrix; candidate fits are sequential.
    features_full = data[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype="float32", copy=True)
    target_full = data["net_return"].to_numpy(dtype="float64", copy=True)
    data.drop(columns=list(MODEL_FEATURE_COLUMNS), inplace=True)

    frozen_cost_bps = published.frozen_round_trip_cost_bps
    dataset_identity_val = _dataset_identity(published)
    gc.collect()

    validation_records: list[dict[str, Any]] = []
    retained_predictions: dict[str, pd.DataFrame] = {}
    for spec in _candidate_specs(policy):
        scored, fold_records = _walk_forward_predictions(spec, data, features_full, target_full, folds, policy)
        record = _evaluate_spec(spec, scored, fold_records, policy, frozen_cost_bps)
        validation_records.append(record)
        passed = [r for r in validation_records if bool(r["validation_passed"])]
        current_selected = max(passed, key=_selection_key) if passed else None
        current_audit_candidate, _, _ = _audit_policy_choice(validation_records, current_selected)

        retained_predictions[spec.candidate_id] = scored
        keys_to_keep = {current_audit_candidate}
        if current_selected is not None:
            keys_to_keep.add(str(current_selected["candidate_id"]))

        for k in list(retained_predictions.keys()):
            if k not in keys_to_keep:
                del retained_predictions[k]

        gc.collect()

        _guard_memory(policy, f"{spec.candidate_id} validation", peak=True)

    passed = [record for record in validation_records if bool(record["validation_passed"])]
    selected = max(passed, key=_selection_key) if passed else None
    status = "candidate" if selected is not None else "no_candidate"
    selected_id = str(selected["candidate_id"]) if selected is not None else None
    config_payload = asdict(policy)
    config_hash = _json_sha256(config_payload)
    evaluation: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "promotion_permitted": False,
        "selection_basis": "development_walk_forward_validation_only",
        "objective": "expected_net_return_after_frozen_round_trip_cost",
        "target_hit_used_as_training_target": False,
        "raw_ndcg_reported": False,
        "future_holdout_opened": False,
        "future_holdout_start_date": policy.future_holdout_start_date,
        "development_end_date": policy.development_end_date,
        "dataset": dataset_identity_val,
        "training_config": config_payload,
        "training_config_sha256": config_hash,
        "validation_candidates": validation_records,
        "selected_candidate_id": selected_id,
        "minimum_economic_gates": _gate_contract(policy),
        "future_data_contract": _future_data_contract(policy),
    }
    audit_candidate, audit_threshold, audit_passed = _audit_policy_choice(
        validation_records,
        selected,
    )
    audit_ledger = _position_ledger(
        retained_predictions[audit_candidate],
        audit_threshold,
        frozen_cost_bps,
        policy,
    )
    evaluation["auditable_policy_ledger"] = {
        "candidate_id": audit_candidate,
        "threshold_bps": audit_threshold,
        "validation_passed": audit_passed,
        "selection_status": "selected_candidate" if audit_passed else "best_failed_diagnostic_only",
        "position_ledger_path": _POSITION_LEDGER_NAME,
        "daily_ledger_path": _DAILY_LEDGER_NAME,
    }
    model_card: dict[str, Any] = {
        "schema_version": "edge_rebuild.intraday_development_model_card.v1",
        "status": status,
        "promotion_permitted": False,
        "candidate_id": selected_id,
        "horizon_minutes": 30,
        "training_target": "net_return",
        "selection_target": "capital_weighted_net_economics",
        "development_rows": int(len(data)),
        "development_sessions": int(len(sessions)),
        "development_securities": int(data["security_id"].nunique()),
        "future_holdout_opened": False,
        "future_data_contract": _future_data_contract(policy),
        "limitations": [
            "candidate is development-only and cannot be promoted without a separately collected future holdout",
            "event-time equity marks open positions at cost until their exact recorded exit",
        ],
    }
    candidate: dict[str, Any] | None = None
    if selected is not None:
        spec = next(item for item in _candidate_specs(policy) if item.candidate_id == selected_id)
        fitted = _fit(spec, features_full, target_full, policy)
        candidate = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "candidate",
            "promotion_permitted": False,
            "validation_passed": True,
            "candidate_id": selected_id,
            "family": spec.family,
            "hyperparameters": dict(spec.hyperparameters),
            "expected_net_return_threshold_bps": float(selected["selected_threshold_bps"]),
            "frozen_round_trip_cost_bps": frozen_cost_bps,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "estimator": fitted,
            "dataset": dataset_identity_val,
            "training_config": config_payload,
            "training_config_sha256": config_hash,
            "future_data_contract": _future_data_contract(policy),
        }
    _publish_development(
        output_directory,
        candidate,
        evaluation,
        model_card,
        audit_ledger,
    )
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
    contract = _object(candidate.get("future_data_contract"), "future_data_contract")
    future_start = _parse_date(str(contract.get("minimum_session_date")), "minimum_session_date")
    development_end = _parse_date(str(contract.get("development_end_date")), "development_end_date")
    policy = IntradayDevelopmentConfig(
        **_tuple_config_values(_object(candidate.get("training_config"), "training_config"))
    )
    if not future_dataset_authority_directory.is_dir():
        raise DataReadinessError(
            f"future holdout data does not exist; collect sessions from {future_start.isoformat()} onward"
        )
    published = load_published_intraday_dataset(future_dataset_authority_directory)
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
    minimum_sessions = int(contract.get("minimum_sessions", 0))
    actual_sessions = int(data["session_date_et"].nunique())
    if minimum_sessions < 1 or actual_sessions < minimum_sessions:
        raise DataReadinessError(
            f"future holdout has {actual_sessions} complete sessions; requires {minimum_sessions}"
        )
    estimator = candidate.get("estimator")
    if estimator is None or not hasattr(estimator, "predict"):
        raise DataReadinessError("candidate estimator is unavailable")
    score = np.asarray(estimator.predict(data.loc[:, MODEL_FEATURE_COLUMNS]), dtype="float64")
    scored = _scored_frame(data, score)
    threshold = _required_finite_number(
        candidate.get("expected_net_return_threshold_bps"),
        "expected_net_return_threshold_bps",
    )
    metrics = _evaluate_policy(scored, threshold, policy, published.frozen_round_trip_cost_bps)
    ledger = _position_ledger(
        scored,
        threshold,
        published.frozen_round_trip_cost_bps,
        policy,
    )
    evaluation = {
        "schema_version": FUTURE_EVALUATION_SCHEMA_VERSION,
        "status": "locked_future_evaluated",
        "promotion_permitted": False,
        "selection_changed_after_future_observation": False,
        "candidate_authority_sha256": file_sha256(candidate_authority_directory / _AUTHORITY_NAME),
        "candidate_manifest_sha256": file_sha256(candidate_authority_directory / _MANIFEST_NAME),
        "candidate_manifest_schema": manifest.get("schema_version"),
        "future_dataset": _dataset_identity(published),
        "future_session_first": str(data["session_date_et"].min()),
        "future_session_last": str(data["session_date_et"].max()),
        "metrics": metrics,
    }
    _publish_future_evaluation(output_directory, evaluation, ledger)
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
    numeric = [*MODEL_FEATURE_COLUMNS, "gross_return", "net_return"]
    for column in numeric:
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"{column} must be finite")
        data[column] = values.astype("float32" if column in MODEL_FEATURE_COLUMNS else "float64")
    expected = data["gross_return"] - published.frozen_round_trip_cost_bps / 10_000.0
    if not np.allclose(expected, data["net_return"], rtol=0.0, atol=1e-10):
        raise DataReadinessError("net return does not match the frozen round-trip cost")
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
        _CandidateSpec(f"ridge_expected_net_alpha_{alpha:g}", "ridge_expected_net", {"alpha": alpha})
        for alpha in config.ridge_alphas
    )
    hgb = tuple(
        _CandidateSpec(
            f"hgb_expected_net_lr_{rate:g}_leaves_{leaves}",
            "hgb_expected_net",
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
    return ridge + hgb


def _fit(
    spec: _CandidateSpec,
    features: np.ndarray,
    target: np.ndarray,
    config: IntradayDevelopmentConfig,
) -> Any:
    if spec.family == "ridge_expected_net":
        estimator: Any = Pipeline(
            [
                ("scale", StandardScaler(copy=False)),
                ("regressor", Ridge(alpha=float(spec.hyperparameters["alpha"]), solver="cholesky")),
            ]
        )
    elif spec.family == "hgb_expected_net":
        estimator = HistGradientBoostingRegressor(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            random_state=config.random_seed,
        )
    else:
        raise AssertionError(f"unsupported intraday development family: {spec.family}")
    estimator.fit(features, target)
    return estimator


def _walk_forward_predictions(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    features_full: np.ndarray,
    target_full: np.ndarray,
    folds: tuple[_Fold, ...],
    config: IntradayDevelopmentConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    evidence: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = data["session_date_et"].isin(fold.train_sessions).to_numpy()
        max_label = data.loc[train_mask, "label_available_at_utc"].max()
        validation_mask = data["session_date_et"].isin(fold.validation_sessions).to_numpy()
        min_decision = data.loc[validation_mask, "decision_time_utc"].min()
        if max_label >= min_decision:
            raise DataReadinessError(f"fold {fold.fold} violates label-time purging")

        gc.collect()

        train_features = features_full[train_mask]
        train_target = target_full[train_mask]
        estimator = _fit(spec, train_features, train_target, config)

        del train_features, train_target
        gc.collect()

        score = np.asarray(estimator.predict(features_full[validation_mask]), dtype="float64")

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
        )
        validation = data.loc[validation_mask, keep_columns].copy()
        validation["predicted_net_return"] = score
        validation["fold"] = fold.fold
        evidence.append(validation)
        records.append(
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "embargo_sessions": list(fold.embargo_sessions),
                "max_train_label_available_at_utc": pd.Timestamp(max_label).isoformat(),
                "min_validation_decision_time_utc": pd.Timestamp(min_decision).isoformat(),
            }
        )
    return pd.concat(evidence, ignore_index=True), records


def _evaluate_spec(
    spec: _CandidateSpec,
    scored: pd.DataFrame,
    folds: Sequence[Mapping[str, Any]],
    config: IntradayDevelopmentConfig,
    frozen_cost_bps: float,
) -> dict[str, Any]:
    threshold_records: list[dict[str, Any]] = []
    for threshold in config.expected_net_return_thresholds_bps:
        metrics = _evaluate_policy(scored, threshold, config, frozen_cost_bps)
        passed, reasons = _economic_gates(metrics, config)
        threshold_records.append(
            {
                "threshold_bps": threshold,
                "validation_passed": passed,
                "failed_gate_reasons": reasons,
                "metrics": metrics,
            }
        )
    passed_thresholds = [record for record in threshold_records if bool(record["validation_passed"])]
    selected = max(passed_thresholds, key=_threshold_selection_key) if passed_thresholds else None
    return {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "hyperparameters": dict(spec.hyperparameters),
        "training_target": "net_return",
        "target_hit_used_as_training_target": False,
        "folds": list(folds),
        "thresholds": threshold_records,
        "validation_passed": selected is not None,
        "selected_threshold_bps": float(selected["threshold_bps"]) if selected else None,
        "selected_metrics": selected["metrics"] if selected else None,
        "failed_gate_reasons": [] if selected else sorted(
            {reason for record in threshold_records for reason in record["failed_gate_reasons"]}
        ),
    }


def _evaluate_policy(
    scored: pd.DataFrame,
    threshold_bps: float,
    config: IntradayDevelopmentConfig,
    frozen_cost_bps: float,
) -> dict[str, Any]:
    primary = _position_ledger(scored, threshold_bps, frozen_cost_bps, config)
    rank = _economic_ranking_metrics(scored, config.maximum_candidates_per_decision)
    bootstrap = _moving_block_bootstrap(
        primary["daily_returns"],
        samples=config.bootstrap_samples,
        block_sessions=config.bootstrap_block_sessions,
        seed=config.random_seed + int(round(threshold_bps * 10.0)),
    )
    cost_curve: list[dict[str, Any]] = []
    for cost_bps in config.cost_curve_bps:
        curve = _position_ledger(scored, threshold_bps, cost_bps, config)
        metrics = _ledger_metrics(curve)
        cost_curve.append({"round_trip_cost_bps": cost_bps, **metrics})
    total_groups = int(scored["decision_group_id"].nunique())
    traded_groups = len(
        {str(row["decision_group_id"]) for row in primary["position_records"]}
    )
    return {
        **_ledger_metrics(primary),
        **rank,
        "threshold_bps": threshold_bps,
        "frozen_round_trip_cost_bps": frozen_cost_bps,
        "moving_block_bootstrap_95_ci": bootstrap,
        "cost_curve": cost_curve,
        "position_ledger_rows": primary["positions"],
        "daily_ledger_rows": primary["daily_rows"],
        "decision_groups": total_groups,
        "decision_groups_with_entries": traded_groups,
        "no_trade_decision_rate": 1.0 - traded_groups / total_groups if total_groups else 1.0,
        "drawdown_basis": "event_time_realized_equity_with_open_positions_marked_at_cost",
        "turnover_basis": "actual_entry_and_exit_notional_divided_by_average_daily_starting_equity",
    }


def _position_ledger(
    scored: pd.DataFrame,
    threshold_bps: float,
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
            candidates = group.loc[group["predicted_net_return"].ge(threshold)].sort_values(
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
                        "gross_return": float(row.gross_return),
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
        daily_rows.append(
            {
                "session_date_et": str(session),
                "starting_equity": start_equity,
                "ending_equity": end_equity,
                "daily_return": end_equity / start_equity - 1.0,
                "entries": sum(1 for row in positions if row["session_date_et"] == str(session)),
                "entry_notional": sum(float(row["notional"]) for row in positions if row["session_date_et"] == str(session)),
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
    equity = cash + sum(float(item["notional"]) for item in open_positions)
    for item in sorted(due, key=lambda value: (value["exit_time_utc"], value["security_id"])):
        realized_return = float(item["gross_return"]) - cost_bps / 10_000.0
        pnl = float(item["notional"]) * realized_return
        cash += float(item["notional"]) + pnl
        open_positions.remove(item)
        cooldown[str(item["security_id"])] = pd.Timestamp(item["exit_time_utc"]) + pd.Timedelta(
            minutes=cooldown_minutes
        )
        equity = cash + sum(float(opened["notional"]) for opened in open_positions)
        item["realized_net_return"] = realized_return
        item["pnl"] = pnl
        item["equity_after_exit"] = equity
        completed.append(item)
        equity_marks.append(equity)
    return cash, equity


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
    return {
        "trade_count": len(positions),
        "sessions": len(daily),
        "sessions_with_trades": sum(int(row["entries"]) > 0 for row in daily),
        "average_trade_net_return": float(pnls.sum() / notionals.sum()) if notionals.sum() > 0.0 else 0.0,
        "average_daily_net_return": float(returns.mean()) if len(returns) else 0.0,
        "compounded_net_return": float(np.prod(1.0 + returns) - 1.0) if len(returns) else 0.0,
        "maximum_drawdown": float(drawdown.max()) if len(drawdown) else 0.0,
        "profit_factor": profit / abs(loss) if loss < 0.0 else (math.inf if profit > 0.0 else 0.0),
        "win_rate": float((pnls > 0.0).mean()) if len(pnls) else 0.0,
        "one_way_turnover": float(notionals.sum() / average_equity) if average_equity > 0.0 else 0.0,
        "round_trip_turnover": float(2.0 * notionals.sum() / average_equity) if average_equity > 0.0 else 0.0,
        "maximum_concurrent_positions_observed": concurrency,
        "maximum_entry_weight_observed": max(
            (float(row["entry_weight"]) for row in positions),
            default=0.0,
        ),
        "maximum_concurrent_positions_enforced": True,
        "capital_weights_enforced": True,
        "security_cooldown_enforced": True,
    }


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


def _economic_ranking_metrics(scored: pd.DataFrame, top_k: int) -> dict[str, Any]:
    gains: list[float] = []
    captures: list[float] = []
    for _, group in scored.groupby("decision_group_id", sort=False, observed=True):
        if len(group) < 2:
            continue
        depth = min(top_k, len(group))
        actual = group["net_return"].to_numpy(dtype="float64")
        score = group["predicted_net_return"].to_numpy(dtype="float64")
        model_mean = float(actual[np.argsort(-score, kind="stable")[:depth]].mean())
        random_expected = float(actual.mean())
        ideal_mean = float(np.sort(actual)[-depth:].mean())
        gains.append(model_mean - random_expected)
        denominator = ideal_mean - random_expected
        captures.append((model_mean - random_expected) / denominator if denominator > 0.0 else 0.0)
    return {
        "ranking_groups": len(gains),
        "economic_rank_gain_over_exact_random_baseline": float(np.mean(gains)) if gains else 0.0,
        "economic_rank_capture_ratio": float(np.mean(captures)) if captures else 0.0,
        "random_baseline_method": "exact_cross_sectional_mean_return",
        "raw_ndcg_reported": False,
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


def _economic_gates(metrics: Mapping[str, Any], config: IntradayDevelopmentConfig) -> tuple[bool, list[str]]:
    reasons: list[str] = []
    checks = (
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
            float(
                _object(
                    _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")["average_daily_net_return"],
                    "daily interval",
                )["low"]
            )
            * 10_000.0
            >= config.minimum_daily_return_ci_low_bps,
            "daily_return_confidence_bound_below_gate",
        ),
        (float(metrics["profit_factor"]) >= config.minimum_profit_factor, "profit_factor_below_gate"),
        (float(metrics["maximum_drawdown"]) <= config.maximum_drawdown, "drawdown_above_gate"),
        (
            float(metrics["economic_rank_gain_over_exact_random_baseline"]) * 10_000.0
            >= config.minimum_economic_rank_gain_bps,
            "economic_rank_gain_below_random",
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
    if float(stress["average_daily_net_return"]) * 10_000.0 < config.minimum_stress_average_daily_return_bps:
        reasons.append("stress_cost_average_daily_return_below_gate")
    return not reasons, reasons


def _threshold_selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, float]:
    metrics = _object(record["metrics"], "threshold metrics")
    daily_interval = _object(
        _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")["average_daily_net_return"],
        "daily interval",
    )
    return (
        float(daily_interval["low"]),
        float(metrics["average_daily_net_return"]),
        float(metrics["economic_rank_gain_over_exact_random_baseline"]),
        -float(record["threshold_bps"]),
    )


def _selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, str]:
    metrics = _object(record["selected_metrics"], "selected metrics")
    interval = _object(
        _object(metrics["moving_block_bootstrap_95_ci"], "bootstrap")["average_daily_net_return"],
        "daily interval",
    )
    return (
        float(interval["low"]),
        float(metrics["average_daily_net_return"]),
        float(metrics["economic_rank_gain_over_exact_random_baseline"]),
        str(record["candidate_id"]),
    )


def _audit_policy_choice(
    records: Sequence[Mapping[str, Any]],
    selected: Mapping[str, Any] | None,
) -> tuple[str, float, bool]:
    if selected is not None:
        return (
            str(selected["candidate_id"]),
            _required_finite_number(selected["selected_threshold_bps"], "selected_threshold_bps"),
            True,
        )
    choices: list[tuple[Mapping[str, Any], Mapping[str, Any]]] = []
    for candidate in records:
        thresholds = candidate.get("thresholds")
        if not isinstance(thresholds, list):
            raise DataReadinessError("validation candidate thresholds are invalid")
        choices.extend((candidate, _object(item, "threshold record")) for item in thresholds)
    if not choices:
        raise DataReadinessError("no validation policy is available for audit")
    candidate, threshold = max(choices, key=lambda item: _threshold_selection_key(item[1]))
    return (
        str(candidate["candidate_id"]),
        _required_finite_number(threshold["threshold_bps"], "threshold_bps"),
        False,
    )


def _scored_frame(data: pd.DataFrame, score: np.ndarray) -> pd.DataFrame:
    if len(score) != len(data) or not np.isfinite(score).all():
        raise DataReadinessError("expected-net-return scores must be finite and row-aligned")
    scored = data.copy()
    scored["predicted_net_return"] = score
    return scored


def _publish_development(
    output: Path,
    candidate: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
    ledger: Mapping[str, Any],
) -> None:
    files: dict[str, Any] = {}
    temporary = _temporary_output(output)
    try:
        if candidate is not None:
            joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_json(temporary / _EVALUATION_NAME, evaluation)
        _write_json(temporary / _MODEL_CARD_NAME, model_card)
        _write_ledger_files(temporary, ledger)
        for path in sorted(temporary.iterdir(), key=lambda item: item.name):
            files[path.name] = {"sha256": file_sha256(path), "bytes": path.stat().st_size}
        state = str(evaluation["status"])
        manifest = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state": state,
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": files,
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        _write_json(
            temporary / _AUTHORITY_NAME,
            {
                "schema_version": AUTHORITY_SCHEMA_VERSION,
                "state": state,
                "promotion_permitted": False,
                "manifest_path": _MANIFEST_NAME,
                "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
            },
        )
        _finish_output(temporary, output)
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


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
    authority = _read_json(directory / _AUTHORITY_NAME, "candidate authority")
    manifest = _read_json(directory / _MANIFEST_NAME, "candidate manifest")
    if authority.get("schema_version") != AUTHORITY_SCHEMA_VERSION:
        raise DataReadinessError("future evaluation accepts only V3 development authorities")
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


def _dataset_identity(published: PublishedIntradayDataset) -> dict[str, str]:
    return {
        "dataset_sha256": published.dataset_sha256,
        "manifest_sha256": published.manifest_sha256,
        "authority_sha256": published.authority_sha256,
    }


def _gate_contract(config: IntradayDevelopmentConfig) -> dict[str, Any]:
    return {
        "minimum_validation_trades": config.minimum_validation_trades,
        "minimum_validation_sessions_with_trades": config.minimum_validation_sessions_with_trades,
        "minimum_average_trade_net_return_bps": config.minimum_average_trade_net_return_bps,
        "minimum_average_daily_net_return_bps": config.minimum_average_daily_net_return_bps,
        "minimum_daily_return_ci_low_bps": config.minimum_daily_return_ci_low_bps,
        "minimum_profit_factor": config.minimum_profit_factor,
        "minimum_economic_rank_gain_bps": config.minimum_economic_rank_gain_bps,
        "maximum_drawdown": config.maximum_drawdown,
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
        "selection_must_remain_frozen": True,
    }


def _tuple_config_values(payload: Mapping[str, Any]) -> dict[str, Any]:
    values = dict(payload)
    for name in (
        "expected_net_return_thresholds_bps",
        "ridge_alphas",
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
