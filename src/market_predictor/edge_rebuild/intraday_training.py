"""Causal, new-only intraday candidate training for the edge rebuild."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
import os
import shutil
import tempfile
import tomllib
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final, cast

import joblib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_dataset import (
    INTRADAY_DATASET_AUTHORITY_SCHEMA,
    INTRADAY_DATASET_SCHEMA,
    load_complete_intraday_dataset,
)
from market_predictor.edge_rebuild.intraday_features import (
    CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS,
    FEATURE_SCHEMA_VERSION,
)
from market_predictor.edge_rebuild.intraday_labels import LABEL_SCHEMA_VERSION
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

DATASET_SCHEMA_VERSION: Final = INTRADAY_DATASET_SCHEMA
MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_candidate.v2"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_evaluation.v1"
MODEL_CARD_SCHEMA_VERSION: Final = "edge_rebuild.intraday_model_card.v1"
OUTPUT_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_candidate_authority.v1"
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
MODEL_FEATURE_COLUMNS: Final = CAUSAL_INTRADAY_MODEL_FEATURE_COLUMNS

_IDENTITY_COLUMNS: Final = (
    "dataset_row_id",
    "ticker",
    "security_id",
    "session_date_et",
    "decision_group_id",
    "feature_available_at_utc",
    "label_available_at_utc",
    "feature_eligible",
    "label_eligible",
    "entry_time_utc",
    "exit_bar_end_utc",
    "session_segment",
    "sector",
)
_TEXT_COLUMNS: Final = (
    "dataset_row_id",
    "ticker",
    "security_id",
    "session_date_et",
    "decision_group_id",
    "session_segment",
    "sector",
)
_PARTITION_CONCAT_CHUNK_SIZE: Final = 512


@dataclass(frozen=True, slots=True)
class IntradayTrainingConfig:
    """Frozen training controls; all estimator work is deliberately sequential."""

    validation_folds: int = 3
    final_test_fraction: float = 0.20
    minimum_train_sessions: int = 60
    minimum_validation_sessions: int = 20
    embargo_sessions: int = 1
    maximum_label_horizon_minutes: int = 30
    calibration_fraction: float = 0.20
    minimum_calibration_sessions: int = 5
    minimum_rows: int = 1_000
    minimum_securities: int = 20
    top_k: int = 10
    logistic_c_values: tuple[float, ...] = (0.1, 1.0)
    hgb_learning_rates: tuple[float, ...] = (0.05, 0.1)
    hgb_max_leaf_nodes: tuple[int, ...] = (15, 31)
    hgb_max_iter: int = 150
    hgb_max_bins: int = 127
    ranker_learning_rates: tuple[float, ...] = (0.05,)
    ranker_max_depths: tuple[int, ...] = (3,)
    ranker_n_estimators: int = 100
    ranker_max_bin: int = 127
    maximum_learned_candidates: int = 6
    bootstrap_samples: int = 500
    random_seed: int = 42
    maximum_process_memory_gib: float = 4.0
    memory_guard_headroom_gib: float = 0.75

    def __post_init__(self) -> None:
        if self.validation_folds < 2:
            raise ValueError("validation_folds must be at least 2")
        if not 0.10 <= self.final_test_fraction <= 0.40:
            raise ValueError("final_test_fraction must be between 0.10 and 0.40")
        if self.minimum_train_sessions < 4 or self.minimum_validation_sessions < 1:
            raise ValueError("walk-forward session minimums are invalid")
        if self.embargo_sessions < 1:
            raise ValueError("at least one overnight embargo session is required")
        if self.maximum_label_horizon_minutes != 30:
            raise ValueError("the frozen intraday label horizon is 30 minutes")
        if not 0.10 <= self.calibration_fraction <= 0.35 or self.minimum_calibration_sessions < 2:
            raise ValueError("calibration session controls are invalid")
        if self.minimum_rows < 1 or self.minimum_securities < 2 or self.top_k < 1:
            raise ValueError("training row, security, and top-k minimums are invalid")
        if not self.logistic_c_values or any(value <= 0 for value in self.logistic_c_values):
            raise ValueError("logistic C values must be positive")
        if not self.hgb_learning_rates or any(value <= 0 for value in self.hgb_learning_rates):
            raise ValueError("HGB learning rates must be positive")
        if not self.hgb_max_leaf_nodes or any(value < 2 for value in self.hgb_max_leaf_nodes):
            raise ValueError("HGB leaf-node limits must be at least 2")
        if self.hgb_max_iter < 10 or not 2 <= self.hgb_max_bins <= 255:
            raise ValueError("HGB iteration or bin bounds are invalid")
        if not self.ranker_learning_rates or any(value <= 0 for value in self.ranker_learning_rates):
            raise ValueError("ranker learning rates must be positive")
        if not self.ranker_max_depths or any(value < 1 or value > 8 for value in self.ranker_max_depths):
            raise ValueError("ranker depths must be between 1 and 8")
        if not 10 <= self.ranker_n_estimators <= 500 or not 2 <= self.ranker_max_bin <= 255:
            raise ValueError("ranker estimator or bin bounds are invalid")
        if not 1 <= self.maximum_learned_candidates <= 6:
            raise ValueError("learned candidate budget must be between 1 and 6")
        if self.bootstrap_samples < 100 or self.bootstrap_samples > 5_000:
            raise ValueError("session bootstrap samples must be between 100 and 5000")
        if not 0 < self.maximum_process_memory_gib <= 4.0:
            raise ValueError("maximum_process_memory_gib must be in (0, 4] GiB")
        if not 0 < self.memory_guard_headroom_gib < self.maximum_process_memory_gib:
            raise ValueError("memory headroom must be below the hard limit")


def load_intraday_training_config(path: Path) -> IntradayTrainingConfig:
    """Load a complete frozen training policy; implicit partial overrides are forbidden."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"intraday training policy is unreadable: {path}") from exc
    payload = raw.get("training")
    if not isinstance(payload, Mapping):
        raise DataReadinessError("intraday training policy requires a [training] table")
    expected = {field.name for field in fields(IntradayTrainingConfig)}
    actual = {str(key) for key in payload}
    if actual != expected:
        raise DataReadinessError(
            "intraday training policy fields differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    values = dict(payload)
    for name in (
        "logistic_c_values",
        "hgb_learning_rates",
        "hgb_max_leaf_nodes",
        "ranker_learning_rates",
        "ranker_max_depths",
    ):
        value = values[name]
        if not isinstance(value, list):
            raise DataReadinessError(f"intraday training policy {name} must be an array")
        values[name] = tuple(value)
    try:
        return IntradayTrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("intraday training policy is invalid") from exc


@dataclass(frozen=True, slots=True)
class PublishedIntradayDataset:
    frame: pd.DataFrame
    root: Path
    feature_columns: tuple[str, ...]
    deterministic_score_column: str
    target_column: str
    gross_return_column: str
    net_return_column: str
    spy_excess_return_column: str
    sector_excess_return_column: str
    frozen_round_trip_cost_bps: float
    dataset_sha256: str
    manifest_sha256: str
    authority_sha256: str


@dataclass(frozen=True, slots=True)
class WalkForwardFold:
    fold: int
    train_sessions: tuple[str, ...]
    validation_sessions: tuple[str, ...]
    embargo_sessions: tuple[str, ...]
    min_validation_decision_time_utc: pd.Timestamp


@dataclass(frozen=True, slots=True)
class IntradayTrainingResult:
    output_directory: Path
    selected_candidate_id: str
    evaluation: Mapping[str, Any]
    model_card: Mapping[str, Any]


@dataclass(frozen=True, slots=True)
class _CandidateSpec:
    candidate_id: str
    family: str
    hyperparameters: Mapping[str, float | int | str]


@dataclass(frozen=True, slots=True)
class _FittedCandidate:
    estimator: Any
    calibrator: LogisticRegression
    calibration_train_cutoff_utc: str
    fit_sessions: int
    calibration_sessions: int


@dataclass(frozen=True, slots=True)
class _CandidatePredictions:
    probability: np.ndarray
    ordering_score: np.ndarray


def train_intraday_edge_candidate(
    dataset_authority_directory: Path,
    output_directory: Path,
    *,
    config: IntradayTrainingConfig | None = None,
) -> IntradayTrainingResult:
    """Train and publish a non-promoted candidate from one trusted dataset.

    The final temporal test is materialized only after validation has selected a
    candidate. Its outcome is evaluation evidence and cannot alter selection.
    """

    policy = config or IntradayTrainingConfig()
    training_config = asdict(policy)
    training_config_sha256 = hashlib.sha256(
        json.dumps(
            training_config,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()
    _guard_memory(policy, "intraday training start", peak=False)
    if output_directory.exists():
        raise FileExistsError(f"immutable output already exists: {output_directory}")

    published = load_published_intraday_dataset(dataset_authority_directory)
    data = _validate_training_frame(published, policy)
    session_order, session_times = _ordered_sessions(data)
    development_sessions, locked_test_sessions, final_embargo_sessions = _split_locked_test_sessions(session_order, policy)
    folds = _build_purged_walk_forward_folds(data, development_sessions, session_times, policy)
    security_holdout = _deterministic_security_holdout(data, development_sessions)

    candidate_specs = _candidate_specs(policy)
    learned_candidate_count = sum(
        spec.family != "deterministic" for spec in candidate_specs
    )
    if learned_candidate_count > policy.maximum_learned_candidates:
        raise DataReadinessError(
            "intraday candidate grid exceeds the frozen learned-candidate budget: "
            f"{learned_candidate_count} > {policy.maximum_learned_candidates}"
        )
    validation_records: list[dict[str, Any]] = []
    for spec in candidate_specs:
        _guard_memory(policy, f"{spec.candidate_id} validation start", peak=False)
        validation_records.append(
            _evaluate_candidate_on_validation(
                spec,
                data,
                published,
                folds,
                security_holdout,
                policy,
            )
        )
        release_process_memory()
        _guard_memory(policy, f"{spec.candidate_id} validation end", peak=True)

    selected = _select_from_validation(validation_records)
    selected_spec = next(spec for spec in candidate_specs if spec.candidate_id == selected["candidate_id"])

    # This is the single controlled opening of the final test partition.
    final_test = data[data["session_date_et"].isin(locked_test_sessions)].copy()
    development = _purged_development_rows(data, development_sessions, final_test)
    fitted = _fit_candidate(
        selected_spec,
        development,
        published,
        policy,
        excluded_securities=frozenset(),
        stage="final development fit",
    )
    test_predictions = _predict_candidate(selected_spec, fitted, final_test, published)
    final_test_metrics = _evaluation_metrics(final_test, test_predictions, published, policy)
    audit = _overlap_audit(
        data,
        folds,
        development_sessions,
        locked_test_sessions,
        final_embargo_sessions,
        security_holdout,
    )
    release_process_memory()
    _guard_memory(policy, "final test evaluation", peak=True)

    evaluation: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": "candidate_only",
        "selection_basis": "validation_only",
        "selection_policy": {
            "name": "ER5_CONSERVATIVE_ECONOMICS_V1",
            "ordered_key": [
                "worst_scope_benchmark_excess_bootstrap_ci_low",
                "worst_scope_net_return_bootstrap_ci_low",
                "worst_scope_mean_benchmark_excess",
                "worst_scope_mean_net_return",
                "negative_worst_scope_max_drawdown",
                "negative_worst_scope_expected_calibration_error",
                "negative_worst_scope_brier_score",
                "simpler_family_tie_break",
            ],
            "auc_used_for_selection": False,
            "scope_rule": "minimum of temporal validation and deterministic unseen-security validation",
        },
        "test_access_count": 1,
        "dataset": _dataset_identity_record(published),
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "split": {
            "method": "chronological_purged_walk_forward_by_full_session_date_et",
            "validation_folds": len(folds),
            "development_sessions": len(development_sessions),
            "locked_final_test_sessions": len(locked_test_sessions),
            "final_overnight_embargo_sessions": list(final_embargo_sessions),
            "first_final_test_decision_time_utc": _iso(final_test["decision_time_utc"].min()),
            "purge_rule": "whole training session retained only when every label_available_at_utc precedes the next partition",
            "maximum_label_horizon_minutes": policy.maximum_label_horizon_minutes,
            "overnight_embargo_sessions": policy.embargo_sessions,
            "split_unit": "session_date_et",
            "random_or_row_split_used": False,
        },
        "overlap_audit": audit,
        "validation_candidates": validation_records,
        "selected_candidate_id": selected_spec.candidate_id,
        "selected_hyperparameters": dict(selected_spec.hyperparameters),
        "selected_validation_key": selected["selection_key"],
        "final_test": final_test_metrics,
        "memory": memory_audit(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
        ).to_record(),
    }
    model_card: dict[str, Any] = {
        "schema_version": MODEL_CARD_SCHEMA_VERSION,
        "model_schema_version": MODEL_SCHEMA_VERSION,
        "status": "candidate",
        "promotion_permitted": False,
        "model_family": selected_spec.family,
        "candidate_id": selected_spec.candidate_id,
        "hyperparameters": dict(selected_spec.hyperparameters),
        "feature_columns": list(published.feature_columns),
        "feature_set_sha256": feature_set_sha256(published.feature_columns),
        "deterministic_score_column": published.deterministic_score_column,
        "deterministic_score_definition": "within-decision-group percentile of causal return_1_bar; descending selection",
        "target_column": published.target_column,
        "frozen_round_trip_cost_bps": published.frozen_round_trip_cost_bps,
        "training_rows": int(len(development)),
        "training_sessions": int(development["session_date_et"].nunique()),
        "training_decision_groups": int(development["decision_group_id"].nunique()),
        "training_securities": int(development["security_id"].nunique()),
        "locked_test_rows": int(len(final_test)),
        "dataset": _dataset_identity_record(published),
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
        "selection_basis": "validation_only",
        "selection_policy": "ER5_CONSERVATIVE_ECONOMICS_V1",
        "calibration_method": "platt_sigmoid_on_prior_purged_development_sessions",
        "development_security_holdout_fraction": 0.20,
        "development_security_holdout_sha256": _security_set_sha256(security_holdout),
        "final_test_opened_once": True,
        "limitations": [
            "Candidate is not promoted and must not be used for live trading.",
            "The deterministic 20% development security holdout is validation evidence, not the locked temporal test.",
            "Publisher-provided causal features and frozen-cost returns remain part of the trust boundary.",
        ],
    }
    payload = {
        "schema_version": MODEL_SCHEMA_VERSION,
        "status": "candidate",
        "promotion_permitted": False,
        "candidate_id": selected_spec.candidate_id,
        "family": selected_spec.family,
        "hyperparameters": dict(selected_spec.hyperparameters),
        "feature_columns": published.feature_columns,
        "feature_set_sha256": feature_set_sha256(published.feature_columns),
        "deterministic_score_column": published.deterministic_score_column,
        "fitted_candidate": fitted,
        "dataset": _dataset_identity_record(published),
        "training_config": training_config,
        "training_config_sha256": training_config_sha256,
    }
    _publish_immutable(output_directory, payload, evaluation, model_card)
    return IntradayTrainingResult(
        output_directory=output_directory,
        selected_candidate_id=selected_spec.candidate_id,
        evaluation=evaluation,
        model_card=model_card,
    )


def load_published_intraday_dataset(directory: Path) -> PublishedIntradayDataset:
    """Load exactly the new edge-rebuild authority shape and verify every hash."""

    root = directory.resolve()
    authority_path = root / _AUTHORITY_NAME
    manifest_path = root / _MANIFEST_NAME
    manifest = load_complete_intraday_dataset(root)
    authority = _read_json_object(authority_path, "dataset authority")
    if authority.get("schema") != INTRADAY_DATASET_AUTHORITY_SCHEMA or manifest.get("schema") != DATASET_SCHEMA_VERSION:
        raise DataReadinessError(f"only {DATASET_SCHEMA_VERSION} with {INTRADAY_DATASET_AUTHORITY_SCHEMA} authority is accepted")
    if authority.get("state") != "complete" or manifest.get("status") != "complete":
        raise DataReadinessError("intraday dataset authority must be complete")
    manifest_sha256 = file_sha256(manifest_path)
    if authority.get("artifact") != _MANIFEST_NAME or authority.get("artifact_sha256") != manifest_sha256:
        raise DataReadinessError("intraday dataset authority does not bind the exact manifest")
    if manifest.get("feature_schema_version") != FEATURE_SCHEMA_VERSION or manifest.get("label_schema_version") != LABEL_SCHEMA_VERSION:
        raise DataReadinessError("intraday dataset uses an unrecognized causal feature or label schema")
    training_contract = _object(manifest.get("training_contract"), "manifest.training_contract")
    if training_contract.get("eligibility_column") != "dataset_eligible":
        raise DataReadinessError("intraday dataset eligibility contract is not recognized")
    excluded = set(_string_tuple(training_contract.get("feature_columns_exclude"), "feature_columns_exclude"))
    leaked_labels = {"target_hit", "gross_return", "net_return", "spy_excess_return", "sector_excess_return", "rank_label"}
    if not leaked_labels.issubset(excluded):
        raise DataReadinessError("dataset training contract does not exclude outcome columns from features")

    required_columns = tuple(
        dict.fromkeys(
            (
                *_IDENTITY_COLUMNS,
                *MODEL_FEATURE_COLUMNS,
                "target_hit",
                "gross_return",
                "cost",
                "net_return",
                "spy_excess_return",
                "sector_excess_return",
            )
        )
    )
    raw_partitions = manifest.get("partitions")
    if not isinstance(raw_partitions, list) or not raw_partitions:
        raise DataReadinessError("published intraday dataset has no partitions")
    projected_rows = sum(int(_object(raw, "manifest partition").get("rows", -1)) for raw in raw_partitions)
    summary = _object(manifest.get("summary"), "manifest.summary")
    projected_eligible_rows = int(summary.get("dataset_eligible_rows", -1))
    if projected_rows < 1 or projected_eligible_rows < 1 or projected_eligible_rows > projected_rows:
        raise DataReadinessError("published intraday dataset partition row counts are invalid")
    text_columns = 6
    timestamp_columns = 5
    boolean_columns = 2
    economic_columns = 6
    bytes_per_row = (
        len(MODEL_FEATURE_COLUMNS) * 4
        + economic_columns * 8
        + timestamp_columns * 8
        + boolean_columns
        + text_columns * 48
    )
    projected_peak = projected_eligible_rows * bytes_per_row * 3
    safety_bytes = int((4.0 - 0.75) * 1024**3)
    if projected_peak > safety_bytes:
        raise DataReadinessError(
            f"projected in-memory training frame is {projected_peak / 1024**3:.2f} GiB; exceeds the 3.25 GiB safety threshold"
        )

    parts: list[pd.DataFrame] = []
    chunks: list[pd.DataFrame] = []
    for index, raw in enumerate(raw_partitions):
        record = _object(raw, "manifest partition")
        path = _resolve_inside(root, record.get("path"))
        schema = pq.read_schema(path)  # type: ignore[no-untyped-call]
        missing = sorted(set(required_columns).difference(schema.names))
        if missing:
            raise DataReadinessError(f"intraday dataset partition is missing trainer columns: {missing}")
        part = pd.read_parquet(path, columns=list(required_columns), filters=[("dataset_eligible", "==", True)])
        if not part.empty:
            for column in MODEL_FEATURE_COLUMNS:
                part[column] = pd.to_numeric(part[column], errors="coerce").astype(
                    "float32"
                )
            for column in _TEXT_COLUMNS:
                part[column] = part[column].astype("string[pyarrow]")
            parts.append(part)
            if len(parts) >= _PARTITION_CONCAT_CHUNK_SIZE:
                chunks.append(pd.concat(parts, ignore_index=True))
                parts.clear()
        assert_memory_budget(
            hard_budget_gib=4.0,
            headroom_gib=0.75,
            stage=f"intraday dataset partition {index} load",
        )
    if parts:
        chunks.append(pd.concat(parts, ignore_index=True))
        parts.clear()
    if not chunks:
        raise DataReadinessError("published intraday dataset has no eligible training rows")
    frame = pd.concat(chunks, ignore_index=True)
    del chunks, parts
    if len(frame) != projected_eligible_rows:
        raise DataReadinessError(
            "loaded eligible rows differ from the immutable dataset summary"
        )
    release_process_memory()
    assert_peak_memory_budget(hard_budget_gib=4.0, headroom_gib=0.75, stage="intraday dataset load")
    frame["row_identity"] = frame["dataset_row_id"]
    frame["decision_time_utc"] = pd.to_datetime(frame["decision_group_id"], utc=True, errors="coerce")
    if frame["decision_time_utc"].isna().any():
        raise DataReadinessError("decision_group_id must be the UTC decision timestamp")
    frame["target"] = frame["target_hit"].astype("int8")
    costs = pd.to_numeric(frame["cost"], errors="coerce")
    if costs.isna().any() or not np.isfinite(costs.to_numpy(dtype="float64")).all() or costs.lt(0.0).any():
        raise DataReadinessError("published intraday cost must be finite and non-negative")
    unique_costs = np.unique(costs.to_numpy(dtype="float64"))
    if len(unique_costs) != 1:
        raise DataReadinessError("published intraday dataset must use one frozen round-trip cost")
    frozen_cost_bps = float(unique_costs[0] * 10_000.0)
    return PublishedIntradayDataset(
        frame=frame,
        root=root,
        feature_columns=MODEL_FEATURE_COLUMNS,
        deterministic_score_column="return_1_bar",
        target_column="target",
        gross_return_column="gross_return",
        net_return_column="net_return",
        spy_excess_return_column="spy_excess_return",
        sector_excess_return_column="sector_excess_return",
        frozen_round_trip_cost_bps=frozen_cost_bps,
        dataset_sha256=manifest_sha256,
        manifest_sha256=manifest_sha256,
        authority_sha256=file_sha256(authority_path),
    )


def _validate_training_frame(published: PublishedIntradayDataset, config: IntradayTrainingConfig) -> pd.DataFrame:
    data = published.frame
    required = {
        *_IDENTITY_COLUMNS,
        *published.feature_columns,
        published.deterministic_score_column,
        published.target_column,
        published.gross_return_column,
        published.net_return_column,
        published.spy_excess_return_column,
        published.sector_excess_return_column,
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataReadinessError(f"published intraday dataset columns are missing: {missing}")
    session_dates = pd.to_datetime(data["session_date_et"], errors="coerce")
    if session_dates.isna().any():
        raise DataReadinessError("session_date_et must contain valid exchange-session dates")
    data["session_date_et"] = session_dates.dt.date.astype(str)
    if len(data) < config.minimum_rows:
        raise DataReadinessError(f"published intraday dataset has {len(data)} rows; requires {config.minimum_rows}")
    if data["row_identity"].isna().any() or data["row_identity"].duplicated().any():
        raise DataReadinessError("row_identity must be present and unique")
    if data["decision_group_id"].isna().any() or data["security_id"].isna().any() or data["ticker"].isna().any():
        raise DataReadinessError("decision group and security identities cannot be missing")
    if data["security_id"].nunique() < config.minimum_securities:
        raise DataReadinessError("published intraday dataset has too few independent securities")
    if not data["feature_eligible"].map(_strict_bool).all() or not data["label_eligible"].map(_strict_bool).all():
        raise DataReadinessError("published training dataset may contain only feature- and label-eligible rows")

    for column in (
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "entry_time_utc",
        "exit_bar_end_utc",
    ):
        parsed = pd.to_datetime(data[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise DataReadinessError(f"{column} must contain valid UTC timestamps")
        data[column] = parsed
    if data["feature_available_at_utc"].gt(data["decision_time_utc"]).any():
        raise DataReadinessError("feature availability occurs after its decision")
    if data["label_available_at_utc"].le(data["decision_time_utc"]).any():
        raise DataReadinessError("label availability must be strictly after its decision")
    label_horizon = data["exit_bar_end_utc"] - data["entry_time_utc"]
    if label_horizon.le(pd.Timedelta(0)).any() or label_horizon.gt(pd.Timedelta(minutes=config.maximum_label_horizon_minutes)).any():
        raise DataReadinessError("label path exceeds the frozen 30-minute horizon")
    if data["label_available_at_utc"].lt(data["exit_bar_end_utc"]).any():
        raise DataReadinessError("label availability precedes the completed outcome path")

    group_time_counts = data.groupby("decision_group_id", observed=True)["decision_time_utc"].nunique()
    if group_time_counts.ne(1).any():
        raise DataReadinessError("each decision_group_id must map to exactly one decision timestamp")
    group_session_counts = data.groupby("decision_group_id", observed=True)["session_date_et"].nunique()
    if group_session_counts.ne(1).any():
        raise DataReadinessError("each decision_group_id must belong to exactly one exchange session")
    duplicate_security = data.duplicated(["decision_group_id", "security_id"], keep=False)
    if duplicate_security.any():
        raise DataReadinessError("a security may appear only once in a decision group")

    target = pd.to_numeric(data[published.target_column], errors="coerce")
    if target.isna().any() or not target.isin([0, 1]).all() or target.nunique() != 2:
        raise DataReadinessError("binary target must be complete and contain both classes")
    data[published.target_column] = target.astype("int8")

    numeric_columns = [
        *published.feature_columns,
        published.deterministic_score_column,
        published.gross_return_column,
        published.net_return_column,
        published.spy_excess_return_column,
        published.sector_excess_return_column,
    ]
    feature_columns = set(published.feature_columns)
    for column in dict.fromkeys(numeric_columns):
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"model input/economic column {column} must be finite")
        data[column] = values.astype(
            "float32" if column in feature_columns else "float64"
        )
    expected_net = data[published.gross_return_column] - published.frozen_round_trip_cost_bps / 10_000.0
    if not np.allclose(
        expected_net.to_numpy(dtype="float64"),
        data[published.net_return_column].to_numpy(dtype="float64"),
        rtol=0.0,
        atol=1e-10,
    ):
        raise DataReadinessError("net return does not include the manifest's frozen round-trip cost")
    return data.sort_values(["decision_time_utc", "decision_group_id", "security_id"], kind="stable").reset_index(drop=True)


def _ordered_sessions(data: pd.DataFrame) -> tuple[tuple[str, ...], dict[str, pd.Timestamp]]:
    sessions = (
        data.groupby("session_date_et", as_index=False, observed=True)["decision_time_utc"]
        .min()
        .sort_values(["decision_time_utc", "session_date_et"], kind="stable")
    )
    order = tuple(sessions["session_date_et"].astype(str))
    times = dict(zip(order, sessions["decision_time_utc"], strict=True))
    return order, times


def _split_locked_test_sessions(
    session_order: tuple[str, ...],
    config: IntradayTrainingConfig,
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    test_count = max(1, math.ceil(len(session_order) * config.final_test_fraction))
    test_start = len(session_order) - test_count
    embargo_start = test_start - config.embargo_sessions
    if embargo_start <= 0:
        raise DataReadinessError("history is too short for a locked test and overnight embargo")
    development = session_order[:embargo_start]
    final_embargo = session_order[embargo_start:test_start]
    locked_test = session_order[test_start:]
    minimum_development = config.minimum_train_sessions + config.validation_folds * config.minimum_validation_sessions
    if len(development) < minimum_development:
        raise DataReadinessError(
            f"development partition has {len(development)} sessions; requires at least {minimum_development} before the locked test"
        )
    return development, locked_test, final_embargo


def _build_purged_walk_forward_folds(
    data: pd.DataFrame,
    development_sessions: tuple[str, ...],
    session_times: Mapping[str, pd.Timestamp],
    config: IntradayTrainingConfig,
) -> tuple[WalkForwardFold, ...]:
    remaining = len(development_sessions) - config.minimum_train_sessions
    fold_size = remaining // config.validation_folds
    if fold_size < config.minimum_validation_sessions:
        raise DataReadinessError("development history is too short for configured walk-forward folds")
    folds: list[WalkForwardFold] = []
    for index in range(config.validation_folds):
        validation_start = config.minimum_train_sessions + index * fold_size
        validation_end = len(development_sessions) if index == config.validation_folds - 1 else validation_start + fold_size
        validation_sessions = development_sessions[validation_start:validation_end]
        validation_start_time = session_times[validation_sessions[0]]
        embargo_start = validation_start - config.embargo_sessions
        embargo_sessions = development_sessions[embargo_start:validation_start]
        potential_train = development_sessions[:embargo_start]
        train_rows = data[data["session_date_et"].isin(potential_train)]
        safe_sessions = tuple(
            session
            for session in potential_train
            if train_rows.loc[train_rows["session_date_et"].eq(session), "label_available_at_utc"].max() < validation_start_time
        )
        if len(safe_sessions) < 2:
            raise DataReadinessError(f"fold {index} has fewer than two purged training sessions")
        folds.append(
            WalkForwardFold(
                fold=index,
                train_sessions=safe_sessions,
                validation_sessions=validation_sessions,
                embargo_sessions=embargo_sessions,
                min_validation_decision_time_utc=validation_start_time,
            )
        )
    return tuple(folds)


def _purged_development_rows(
    data: pd.DataFrame,
    development_sessions: tuple[str, ...],
    final_test: pd.DataFrame,
) -> pd.DataFrame:
    first_test_time = cast(pd.Timestamp, final_test["decision_time_utc"].min())
    development = data[data["session_date_et"].isin(development_sessions)].copy()
    safe_sessions = development.groupby("session_date_et", observed=True)["label_available_at_utc"].max().lt(first_test_time)
    development = development[development["session_date_et"].isin(tuple(safe_sessions[safe_sessions].index.astype(str)))].copy()
    if development.empty or development.iloc[-1]["decision_time_utc"] >= first_test_time:
        raise DataReadinessError("final development fit is not temporally isolated from the test")
    return development


def _deterministic_security_holdout(data: pd.DataFrame, development_sessions: tuple[str, ...]) -> frozenset[str]:
    securities = sorted(set(data.loc[data["session_date_et"].isin(development_sessions), "security_id"].astype(str)))
    if len(securities) < 5:
        raise DataReadinessError("a deterministic 20% holdout requires at least five development securities")
    ordered = sorted(securities, key=lambda value: (hashlib.sha256(f"ER5_SECURITY_HOLDOUT_V1|{value}".encode()).hexdigest(), value))
    holdout_count = max(1, math.ceil(len(ordered) * 0.20))
    if len(ordered) - holdout_count < 2:
        raise DataReadinessError("security holdout leaves too few fitting securities")
    return frozenset(ordered[:holdout_count])


def _security_set_sha256(securities: frozenset[str]) -> str:
    payload = json.dumps(sorted(securities), separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _candidate_specs(config: IntradayTrainingConfig) -> tuple[_CandidateSpec, ...]:
    xgboost_version = str(importlib.import_module("xgboost").__version__)
    specs = [_CandidateSpec("deterministic_score", "deterministic", {"score_order": "descending"})]
    specs.extend(_CandidateSpec(f"logistic_c_{value:g}", "logistic", {"C": value}) for value in config.logistic_c_values)
    specs.extend(
        _CandidateSpec(
            f"hgb_lr_{rate:g}_leaves_{leaves}",
            "hist_gradient_boosting",
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
    specs.extend(
        _CandidateSpec(
            f"xgb_rank_ndcg_lr_{rate:g}_depth_{depth}",
            "xgb_ranker",
            {
                "objective": "rank:ndcg",
                "learning_rate": rate,
                "max_depth": depth,
                "n_estimators": config.ranker_n_estimators,
                "max_bin": config.ranker_max_bin,
                "n_jobs": 1,
                "library_version": xgboost_version,
            },
        )
        for rate in config.ranker_learning_rates
        for depth in config.ranker_max_depths
    )
    return tuple(specs)


def _evaluate_candidate_on_validation(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    published: PublishedIntradayDataset,
    folds: tuple[WalkForwardFold, ...],
    security_holdout: frozenset[str],
    config: IntradayTrainingConfig,
) -> dict[str, Any]:
    evidence: list[pd.DataFrame] = []
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        train = data[data["session_date_et"].isin(fold.train_sessions)].copy()
        validation = data[data["session_date_et"].isin(fold.validation_sessions)].copy()
        if train["label_available_at_utc"].max() >= validation["decision_time_utc"].min():
            raise DataReadinessError(f"fold {fold.fold} violates label purging")
        fitted = _fit_candidate(
            spec,
            train,
            published,
            config,
            excluded_securities=security_holdout,
            stage=f"{spec.candidate_id} fold {fold.fold}",
        )
        predictions = _predict_candidate(spec, fitted, validation, published)
        part = validation.copy()
        part["__probability"] = predictions.probability
        part["__ordering_score"] = predictions.ordering_score
        evidence.append(part)
        unseen = part[part["security_id"].astype(str).isin(security_holdout)].copy()
        if unseen.empty:
            raise DataReadinessError(f"fold {fold.fold} has no unseen-security validation evidence")
        fold_records.append(
            {
                "fold": fold.fold,
                "train_rows": len(train),
                "validation_rows": len(validation),
                "train_sessions": len(fold.train_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "embargo_sessions": len(fold.embargo_sessions),
                "train_session_dates": list(fold.train_sessions),
                "validation_session_dates": list(fold.validation_sessions),
                "embargo_session_dates": list(fold.embargo_sessions),
                "max_train_label_available_at_utc": _iso(train["label_available_at_utc"].max()),
                "min_validation_decision_time_utc": _iso(validation["decision_time_utc"].min()),
                "calibration_train_cutoff_utc": fitted.calibration_train_cutoff_utc,
                "fit_rows_excluding_security_holdout": int(len(train[~train["security_id"].astype(str).isin(security_holdout)])),
                "fit_securities_excluding_holdout": int(
                    train.loc[~train["security_id"].astype(str).isin(security_holdout), "security_id"].nunique()
                ),
                "unseen_security_evidence_rows": int(len(unseen)),
                "temporal_metrics": _evaluation_metrics(validation, predictions, published, config),
                "unseen_security_metrics": _evaluation_metrics(
                    unseen,
                    _CandidatePredictions(
                        probability=unseen["__probability"].to_numpy(dtype="float64"),
                        ordering_score=unseen["__ordering_score"].to_numpy(dtype="float64"),
                    ),
                    published,
                    config,
                ),
            }
        )
        del fitted, train, validation, unseen, part
        release_process_memory()
        _guard_memory(config, f"{spec.candidate_id} fold {fold.fold} complete", peak=True)
    combined = pd.concat(evidence, ignore_index=True)
    unseen_combined = combined[combined["security_id"].astype(str).isin(security_holdout)].copy()
    temporal_metrics = _evaluation_metrics(
        combined,
        _CandidatePredictions(
            probability=combined["__probability"].to_numpy(dtype="float64"),
            ordering_score=combined["__ordering_score"].to_numpy(dtype="float64"),
        ),
        published,
        config,
    )
    unseen_metrics = _evaluation_metrics(
        unseen_combined,
        _CandidatePredictions(
            probability=unseen_combined["__probability"].to_numpy(dtype="float64"),
            ordering_score=unseen_combined["__ordering_score"].to_numpy(dtype="float64"),
        ),
        published,
        config,
    )
    record: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "family": spec.family,
        "hyperparameters": dict(spec.hyperparameters),
        "folds": fold_records,
        "temporal_validation": temporal_metrics,
        "unseen_security_validation": unseen_metrics,
    }
    record["selection_key"] = list(_validation_selection_key(record))
    return record


def _fit_candidate(
    spec: _CandidateSpec,
    train: pd.DataFrame,
    published: PublishedIntradayDataset,
    config: IntradayTrainingConfig,
    *,
    excluded_securities: frozenset[str],
    stage: str,
) -> _FittedCandidate:
    _guard_memory(config, stage, peak=False)
    eligible = train[~train["security_id"].astype(str).isin(excluded_securities)].copy()
    fit, calibration = _split_fit_calibration(eligible, config)
    if fit[published.target_column].nunique() != 2 or calibration[published.target_column].nunique() != 2:
        raise DataReadinessError(f"{stage} training target has only one class")
    estimator: Any = None
    if spec.family == "logistic":
        estimator = Pipeline(
            [
                ("scale", StandardScaler()),
                (
                    "model",
                    LogisticRegression(
                        C=float(spec.hyperparameters["C"]),
                        class_weight="balanced",
                        max_iter=500,
                        random_state=config.random_seed,
                        solver="lbfgs",
                    ),
                ),
            ]
        )
    elif spec.family == "hist_gradient_boosting":
        estimator = HistGradientBoostingClassifier(
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_leaf_nodes=int(spec.hyperparameters["max_leaf_nodes"]),
            max_iter=int(spec.hyperparameters["max_iter"]),
            max_bins=int(spec.hyperparameters["max_bins"]),
            l2_regularization=1.0,
            early_stopping=False,
            random_state=config.random_seed,
        )
    elif spec.family == "xgb_ranker":
        xgboost = importlib.import_module("xgboost")
        estimator = xgboost.XGBRanker(
            objective="rank:ndcg",
            learning_rate=float(spec.hyperparameters["learning_rate"]),
            max_depth=int(spec.hyperparameters["max_depth"]),
            n_estimators=int(spec.hyperparameters["n_estimators"]),
            max_bin=int(spec.hyperparameters["max_bin"]),
            n_jobs=1,
            tree_method="hist",
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=1.0,
            random_state=config.random_seed,
            verbosity=0,
        )
    elif spec.family != "deterministic":
        raise AssertionError(f"unsupported candidate family: {spec.family}")

    if spec.family in {"logistic", "hist_gradient_boosting"}:
        x = fit.loc[:, published.feature_columns].to_numpy(dtype="float64", copy=False)
        y = fit[published.target_column].to_numpy(dtype="int8", copy=False)
        estimator.fit(x, y)
    elif spec.family == "xgb_ranker":
        ranked = fit.sort_values(["decision_group_id", "security_id"], kind="stable")
        qid = pd.factorize(ranked["decision_group_id"], sort=False)[0].astype("int32")
        estimator.fit(
            ranked.loc[:, published.feature_columns].to_numpy(dtype="float32", copy=False),
            _rank_relevance(ranked, published.net_return_column),
            qid=qid,
            verbose=False,
        )

    calibration_raw = _raw_candidate_scores(spec, estimator, calibration, published)
    calibrator = LogisticRegression(C=1.0, max_iter=300, random_state=config.random_seed, solver="lbfgs")
    calibrator.fit(calibration_raw.reshape(-1, 1), calibration[published.target_column].to_numpy(dtype="int8"))
    _guard_memory(config, f"{stage} fitted", peak=True)
    return _FittedCandidate(
        estimator=estimator,
        calibrator=calibrator,
        calibration_train_cutoff_utc=_iso(calibration["label_available_at_utc"].max()),
        fit_sessions=int(fit["session_date_et"].nunique()),
        calibration_sessions=int(calibration["session_date_et"].nunique()),
    )


def _split_fit_calibration(
    train: pd.DataFrame,
    config: IntradayTrainingConfig,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions, times = _ordered_sessions(train)
    calibration_count = max(config.minimum_calibration_sessions, math.ceil(len(sessions) * config.calibration_fraction))
    calibration_start = len(sessions) - calibration_count
    fit_end = calibration_start - config.embargo_sessions
    if fit_end < 2:
        raise DataReadinessError("training fold is too short for prior-session calibration and embargo")
    calibration_sessions = sessions[calibration_start:]
    calibration_start_time = times[calibration_sessions[0]]
    possible_fit = train[train["session_date_et"].isin(sessions[:fit_end])].copy()
    safe_sessions = possible_fit.groupby("session_date_et", observed=True)["label_available_at_utc"].max().lt(calibration_start_time)
    fit = possible_fit[possible_fit["session_date_et"].isin(tuple(safe_sessions[safe_sessions].index.astype(str)))].copy()
    calibration = train[train["session_date_et"].isin(calibration_sessions)].copy()
    if fit.empty or calibration.empty or fit["label_available_at_utc"].max() >= calibration["decision_time_utc"].min():
        raise DataReadinessError("calibration split is not causally purged")
    return fit, calibration


def _rank_relevance(frame: pd.DataFrame, return_column: str) -> np.ndarray:
    percentile = frame.groupby("decision_group_id", sort=False, observed=True)[return_column].rank(method="average", pct=True)
    return np.asarray(np.minimum(np.floor(percentile.to_numpy(dtype="float64") * 5.0), 4.0), dtype="int32")


def _raw_candidate_scores(
    spec: _CandidateSpec,
    estimator: Any,
    frame: pd.DataFrame,
    published: PublishedIntradayDataset,
) -> np.ndarray:
    if spec.family == "deterministic":
        return np.asarray(frame[published.deterministic_score_column].to_numpy(dtype="float64", copy=True), dtype="float64")
    x = frame.loc[:, published.feature_columns].to_numpy(
        dtype="float32" if spec.family == "xgb_ranker" else "float64",
        copy=False,
    )
    if spec.family == "xgb_ranker":
        return np.asarray(estimator.predict(x), dtype="float64")
    probability = np.asarray(estimator.predict_proba(x)[:, 1], dtype="float64")
    return np.log(np.clip(probability, 1e-8, 1.0 - 1e-8) / np.clip(1.0 - probability, 1e-8, 1.0))


def _predict_candidate(
    spec: _CandidateSpec,
    fitted: _FittedCandidate,
    frame: pd.DataFrame,
    published: PublishedIntradayDataset,
) -> _CandidatePredictions:
    raw = _raw_candidate_scores(spec, fitted.estimator, frame, published)
    probability = np.asarray(fitted.calibrator.predict_proba(raw.reshape(-1, 1))[:, 1], dtype="float64")
    ordering = raw if spec.family in {"deterministic", "xgb_ranker"} else probability
    return _CandidatePredictions(probability=probability, ordering_score=np.asarray(ordering, dtype="float64"))


def _select_from_validation(records: Sequence[Mapping[str, Any]]) -> Mapping[str, Any]:
    if not records:
        raise DataReadinessError("no validation candidates were evaluated")

    return max(records, key=_validation_selection_key)


def _validation_selection_key(record: Mapping[str, Any]) -> tuple[float, float, float, float, float, float, float, int, str]:
    scopes = [
        _object(record.get("temporal_validation"), "temporal validation"),
        _object(record.get("unseen_security_validation"), "unseen-security validation"),
    ]
    benchmark_ci = min(
        min(_ci_low(scope, "top_k_average_spy_excess_return"), _ci_low(scope, "top_k_average_sector_excess_return")) for scope in scopes
    )
    net_ci = min(_ci_low(scope, "top_k_average_net_return") for scope in scopes)
    benchmark_mean = min(
        min(_required_finite(scope, "top_k_average_spy_excess_return"), _required_finite(scope, "top_k_average_sector_excess_return"))
        for scope in scopes
    )
    net_mean = min(_required_finite(scope, "top_k_average_net_return") for scope in scopes)
    worst_drawdown = max(_required_finite(scope, "max_drawdown_after_costs") for scope in scopes)
    worst_ece = max(_required_finite(scope, "expected_calibration_error") for scope in scopes)
    worst_brier = max(_required_finite(scope, "brier_score") for scope in scopes)
    family_preference = {"deterministic": 3, "logistic": 2, "hist_gradient_boosting": 1, "xgb_ranker": 0}
    return (
        benchmark_ci,
        net_ci,
        benchmark_mean,
        net_mean,
        -worst_drawdown,
        -worst_ece,
        -worst_brier,
        family_preference[str(record["family"])],
        str(record["candidate_id"]),
    )


def _ci_low(metrics: Mapping[str, Any], key: str) -> float:
    intervals = _object(metrics.get("session_block_bootstrap_95_ci"), "session bootstrap intervals")
    interval = _object(intervals.get(key), f"bootstrap interval {key}")
    return _required_finite(interval, "low")


def _evaluation_metrics(
    frame: pd.DataFrame,
    predictions: _CandidatePredictions,
    published: PublishedIntradayDataset,
    config: IntradayTrainingConfig,
) -> dict[str, Any]:
    probability = np.asarray(predictions.probability, dtype="float64")
    ordering = np.asarray(predictions.ordering_score, dtype="float64")
    target = frame[published.target_column].to_numpy(dtype="int8", copy=False)
    if (
        len(probability) != len(frame)
        or len(ordering) != len(frame)
        or not np.isfinite(probability).all()
        or not np.isfinite(ordering).all()
        or ((probability < 0) | (probability > 1)).any()
    ):
        raise DataReadinessError("candidate probabilities must be finite values in [0, 1]")

    scored = frame.copy()
    scored["__probability"] = probability
    scored["__ordering_score"] = ordering
    selected = (
        scored.sort_values(
            ["decision_group_id", "__ordering_score", "security_id"],
            ascending=[True, False, True],
            kind="stable",
        )
        .groupby("decision_group_id", sort=False, observed=True)
        .head(config.top_k)
        .copy()
    )
    selected = selected.sort_values(["decision_time_utc", "decision_group_id", "security_id"], kind="stable")
    session_returns = selected.groupby("session_date_et", sort=False, observed=True)[published.net_return_column].mean()
    equity = (1.0 + session_returns).cumprod()
    running_peak = equity.cummax()
    drawdown = 1.0 - equity / running_peak

    group_sets = [set(group["security_id"].astype(str)) for _, group in selected.groupby("decision_group_id", sort=False, observed=True)]
    turnovers = [
        1.0 - len(previous.intersection(current)) / max(len(previous), len(current))
        for previous, current in zip(group_sets, group_sets[1:], strict=False)
    ]
    rank_records = _cross_sectional_rank_records(scored, published.net_return_column, config.top_k)
    bootstrap = _session_block_bootstrap(selected, rank_records, published, config)
    calibration_bins = _calibration_bins(target, probability)
    positive = selected.loc[selected[published.net_return_column].gt(0.0), published.net_return_column].sum()
    negative = selected.loc[selected[published.net_return_column].lt(0.0), published.net_return_column].sum()
    profit_factor = float(positive / abs(negative)) if negative < 0 else None
    has_two_classes = np.unique(target).size == 2
    return {
        "rows": int(len(frame)),
        "sessions": int(frame["session_date_et"].nunique()),
        "decision_groups": int(frame["decision_group_id"].nunique()),
        "securities": int(frame["security_id"].nunique()),
        "roc_auc": float(roc_auc_score(target, probability)) if has_two_classes else None,
        "pr_auc": float(average_precision_score(target, probability)) if has_two_classes else None,
        "auc_is_diagnostic_only": True,
        "brier_score": float(brier_score_loss(target, probability)),
        "calibration_bias": float(probability.mean() - target.mean()),
        "expected_calibration_error": _expected_calibration_error(calibration_bins, len(frame)),
        "calibration_bins": calibration_bins,
        "top_k": config.top_k,
        "top_k_average_gross_return": float(selected[published.gross_return_column].mean()),
        "top_k_average_net_return": float(selected[published.net_return_column].mean()),
        "top_k_average_spy_excess_return": float(selected[published.spy_excess_return_column].mean()),
        "top_k_average_sector_excess_return": float(selected[published.sector_excess_return_column].mean()),
        "top_k_win_rate_after_costs": float(selected[published.net_return_column].gt(0.0).mean()),
        "profit_factor_after_costs": profit_factor,
        "turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "trade_count": int(len(selected)),
        "max_drawdown_after_costs": float(drawdown.max()) if len(drawdown) else 0.0,
        "compounded_net_return": float(equity.iloc[-1] - 1.0) if len(equity) else 0.0,
        "decision_group_rank_ic_mean": _finite_mean(rank_records["rank_ic"]),
        "daily_session_block_rank_ic_mean": _session_metric_mean(rank_records, "rank_ic"),
        "rank_ic_stability": _stability(rank_records["rank_ic"]),
        "ndcg_at_k": _finite_mean(rank_records["ndcg_at_k"]),
        "top_minus_bottom_net_return_spread": _finite_mean(rank_records["top_minus_bottom_spread"]),
        "session_block_bootstrap_95_ci": bootstrap,
        "by_year": _economics_breakdown(selected, "__year", published),
        "by_session_segment": _economics_breakdown(selected, "session_segment", published),
        "by_sector": _economics_breakdown(selected, "sector", published),
        "frozen_round_trip_cost_bps": published.frozen_round_trip_cost_bps,
    }


def _cross_sectional_rank_records(scored: pd.DataFrame, return_column: str, top_k: int) -> pd.DataFrame:
    records: list[dict[str, Any]] = []
    for group_id, group in scored.groupby("decision_group_id", sort=False, observed=True):
        ordered = group.sort_values(["__ordering_score", "security_id"], ascending=[False, True], kind="stable")
        count = len(ordered)
        if count < 2:
            continue
        actual = ordered[return_column].to_numpy(dtype="float64")
        score = ordered["__ordering_score"].to_numpy(dtype="float64")
        rank_ic = _spearman(actual, score)
        depth = max(1, min(top_k, count // 2))
        spread = float(actual[:depth].mean() - actual[-depth:].mean())
        records.append(
            {
                "decision_group_id": str(group_id),
                "session_date_et": str(ordered["session_date_et"].iloc[0]),
                "rank_ic": rank_ic,
                "ndcg_at_k": _ndcg_at_k(actual, score, top_k),
                "top_minus_bottom_spread": spread,
            }
        )
    if not records:
        raise DataReadinessError("evaluation has no cross-sectional groups with at least two securities")
    return pd.DataFrame(records)


def _spearman(actual: np.ndarray, score: np.ndarray) -> float:
    actual_rank = pd.Series(actual).rank(method="average").to_numpy(dtype="float64")
    score_rank = pd.Series(score).rank(method="average").to_numpy(dtype="float64")
    if np.std(actual_rank) == 0.0 or np.std(score_rank) == 0.0:
        return 0.0
    return float(np.corrcoef(actual_rank, score_rank)[0, 1])


def _ndcg_at_k(actual: np.ndarray, score: np.ndarray, top_k: int) -> float:
    relevance = pd.Series(actual).rank(method="average", pct=True).to_numpy(dtype="float64")
    depth = min(top_k, len(actual))
    predicted_order = np.argsort(-score, kind="stable")[:depth]
    ideal_order = np.argsort(-relevance, kind="stable")[:depth]
    discounts = np.log2(np.arange(2, depth + 2, dtype="float64"))
    dcg = float(((2.0 ** relevance[predicted_order] - 1.0) / discounts).sum())
    ideal = float(((2.0 ** relevance[ideal_order] - 1.0) / discounts).sum())
    return dcg / ideal if ideal > 0.0 else 0.0


def _session_block_bootstrap(
    selected: pd.DataFrame,
    rank_records: pd.DataFrame,
    published: PublishedIntradayDataset,
    config: IntradayTrainingConfig,
) -> dict[str, dict[str, float | int]]:
    selected_blocks = (
        selected.groupby("session_date_et", as_index=False, observed=True)[
            [
                published.gross_return_column,
                published.net_return_column,
                published.spy_excess_return_column,
                published.sector_excess_return_column,
            ]
        ]
        .mean()
        .rename(
            columns={
                published.gross_return_column: "top_k_average_gross_return",
                published.net_return_column: "top_k_average_net_return",
                published.spy_excess_return_column: "top_k_average_spy_excess_return",
                published.sector_excess_return_column: "top_k_average_sector_excess_return",
            }
        )
    )
    rank_blocks = rank_records.groupby("session_date_et", as_index=False, observed=True)[
        ["rank_ic", "ndcg_at_k", "top_minus_bottom_spread"]
    ].mean()
    blocks = selected_blocks.merge(rank_blocks, on="session_date_et", how="inner", validate="one_to_one")
    output: dict[str, dict[str, float | int]] = {}
    for column in blocks.columns:
        if column == "session_date_et":
            continue
        values = blocks[column].to_numpy(dtype="float64")
        seed = config.random_seed + int(hashlib.sha256(column.encode()).hexdigest()[:8], 16)
        output[column] = _bootstrap_mean_interval(values, config.bootstrap_samples, seed)
    return output


def _bootstrap_mean_interval(values: np.ndarray, samples: int, seed: int) -> dict[str, float | int]:
    finite = values[np.isfinite(values)]
    if len(finite) < 2:
        raise DataReadinessError("session-block bootstrap requires at least two independent sessions")
    rng = np.random.default_rng(seed)
    means = np.empty(samples, dtype="float64")
    for index in range(samples):
        means[index] = float(rng.choice(finite, size=len(finite), replace=True).mean())
    return {
        "estimate": float(finite.mean()),
        "low": float(np.quantile(means, 0.025)),
        "high": float(np.quantile(means, 0.975)),
        "sessions": int(len(finite)),
        "bootstrap_samples": samples,
    }


def _economics_breakdown(selected: pd.DataFrame, column: str, published: PublishedIntradayDataset) -> list[dict[str, Any]]:
    data = selected.copy()
    if column == "__year":
        data[column] = data["session_date_et"].astype(str).str[:4]
    if column not in data.columns:
        return []
    records: list[dict[str, Any]] = []
    for value, group in data.groupby(column, sort=True, observed=True):
        records.append(
            {
                "value": str(value),
                "sessions": int(group["session_date_et"].nunique()),
                "trades": int(len(group)),
                "average_net_return": float(group[published.net_return_column].mean()),
                "average_spy_excess_return": float(group[published.spy_excess_return_column].mean()),
                "average_sector_excess_return": float(group[published.sector_excess_return_column].mean()),
                "win_rate_after_costs": float(group[published.net_return_column].gt(0.0).mean()),
            }
        )
    return records


def _finite_mean(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    return float(finite.mean()) if not finite.empty else 0.0


def _session_metric_mean(records: pd.DataFrame, column: str) -> float:
    return _finite_mean(records.groupby("session_date_et", observed=True)[column].mean())


def _stability(values: pd.Series) -> float:
    finite = pd.to_numeric(values, errors="coerce").dropna()
    deviation = float(finite.std(ddof=1)) if len(finite) > 1 else 0.0
    return float(finite.mean() / deviation) if deviation > 0.0 else 0.0


def _calibration_bins(target: np.ndarray, probability: np.ndarray) -> list[dict[str, float | int]]:
    edges = np.linspace(0.0, 1.0, 11)
    indices = np.minimum(np.searchsorted(edges, probability, side="right") - 1, 9)
    indices = np.maximum(indices, 0)
    records: list[dict[str, float | int]] = []
    for index in range(10):
        mask = indices == index
        if not mask.any():
            continue
        records.append(
            {
                "bin": index,
                "rows": int(mask.sum()),
                "mean_probability": float(probability[mask].mean()),
                "observed_rate": float(target[mask].mean()),
            }
        )
    return records


def _expected_calibration_error(records: Sequence[Mapping[str, float | int]], rows: int) -> float:
    return float(
        sum(int(record["rows"]) / rows * abs(float(record["mean_probability"]) - float(record["observed_rate"])) for record in records)
    )


def _overlap_audit(
    data: pd.DataFrame,
    folds: tuple[WalkForwardFold, ...],
    development_sessions: tuple[str, ...],
    test_sessions: tuple[str, ...],
    final_embargo_sessions: tuple[str, ...],
    security_holdout: frozenset[str],
) -> dict[str, Any]:
    test = data[data["session_date_et"].isin(test_sessions)]
    development = data[data["session_date_et"].isin(development_sessions)]
    fold_records: list[dict[str, Any]] = []
    for fold in folds:
        train = data[data["session_date_et"].isin(fold.train_sessions)]
        validation = data[data["session_date_et"].isin(fold.validation_sessions)]
        embargo = data[data["session_date_et"].isin(fold.embargo_sessions)]
        record = _partition_overlap_record(f"fold_{fold.fold}", train, validation)
        record["embargo_session_overlap_with_train"] = len(
            set(embargo["session_date_et"].astype(str)).intersection(train["session_date_et"].astype(str))
        )
        record["embargo_session_overlap_with_validation"] = len(
            set(embargo["session_date_et"].astype(str)).intersection(validation["session_date_et"].astype(str))
        )
        fold_records.append(record)
    final_record = _partition_overlap_record("development_vs_locked_test", development, test)
    development_securities = set(development["security_id"].astype(str))
    fitting_securities = development_securities.difference(security_holdout)
    holdout_rows = development[development["security_id"].astype(str).isin(security_holdout)]
    return {
        "row_identity_overlap_total": int(sum(record["row_identity_overlap"] for record in [*fold_records, final_record])),
        "session_date_overlap_total": int(sum(record["session_date_overlap"] for record in [*fold_records, final_record])),
        "all_rows_from_one_session_stay_in_one_temporal_split": all(
            record["session_date_overlap"] == 0
            and record["embargo_session_overlap_with_train"] == 0
            and record["embargo_session_overlap_with_validation"] == 0
            for record in fold_records
        )
        and final_record["session_date_overlap"] == 0,
        "security_overlap_is_reported_not_forbidden": True,
        "development_security_holdout": {
            "method": "sha256_ordered_security_id",
            "fraction": 0.20,
            "actual_fraction": len(security_holdout) / len(development_securities),
            "security_count": len(security_holdout),
            "security_set_sha256": _security_set_sha256(security_holdout),
            "fit_security_count": len(fitting_securities),
            "fit_holdout_security_overlap": len(fitting_securities.intersection(security_holdout)),
            "evidence_rows": int(len(holdout_rows)),
            "evidence_sessions": int(holdout_rows["session_date_et"].nunique()),
            "separate_from_locked_test": True,
        },
        "final_embargo_sessions": list(final_embargo_sessions),
        "folds": fold_records,
        "final": final_record,
    }


def _partition_overlap_record(name: str, left: pd.DataFrame, right: pd.DataFrame) -> dict[str, Any]:
    left_rows = set(left["row_identity"].astype(str))
    right_rows = set(right["row_identity"].astype(str))
    left_sessions = set(left["session_date_et"].astype(str))
    right_sessions = set(right["session_date_et"].astype(str))
    left_securities = set(left["security_id"].astype(str))
    right_securities = set(right["security_id"].astype(str))
    security_overlap = left_securities.intersection(right_securities)
    return {
        "partition": name,
        "row_identity_overlap": len(left_rows.intersection(right_rows)),
        "session_date_overlap": len(left_sessions.intersection(right_sessions)),
        "security_overlap": len(security_overlap),
        "left_securities": len(left_securities),
        "right_securities": len(right_securities),
        "right_novel_securities": len(right_securities.difference(left_securities)),
        "security_overlap_fraction_of_right": len(security_overlap) / len(right_securities) if right_securities else 0.0,
    }


def _publish_immutable(
    output_directory: Path,
    candidate: Mapping[str, Any],
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
) -> None:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output_directory.name}.", dir=output_directory.parent))
    try:
        joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_new_json(temporary / _EVALUATION_NAME, evaluation)
        _write_new_json(temporary / _MODEL_CARD_NAME, model_card)
        files = {
            name: {"sha256": file_sha256(temporary / name), "bytes": (temporary / name).stat().st_size}
            for name in (_CANDIDATE_NAME, _EVALUATION_NAME, _MODEL_CARD_NAME)
        }
        manifest = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "state": "candidate",
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": files,
        }
        _write_new_json(temporary / _MANIFEST_NAME, manifest)
        authority = {
            "schema_version": OUTPUT_AUTHORITY_SCHEMA_VERSION,
            "state": "candidate",
            "promotion_permitted": False,
            "manifest_path": _MANIFEST_NAME,
            "manifest_sha256": file_sha256(temporary / _MANIFEST_NAME),
        }
        _write_new_json(temporary / _AUTHORITY_NAME, authority)
        try:
            temporary.rename(output_directory)
        except FileExistsError:
            raise FileExistsError(f"immutable output already exists: {output_directory}") from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def _dataset_identity_record(published: PublishedIntradayDataset) -> dict[str, Any]:
    return {
        "schema_version": DATASET_SCHEMA_VERSION,
        "dataset_sha256": published.dataset_sha256,
        "manifest_sha256": published.manifest_sha256,
        "authority_sha256": published.authority_sha256,
    }


def _guard_memory(config: IntradayTrainingConfig, stage: str, *, peak: bool) -> None:
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


def _read_json_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise DataReadinessError(f"{label} is missing: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"{label} is invalid: {exc}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be a JSON object")
    return cast(dict[str, Any], value)


def _write_new_json(path: Path, value: Mapping[str, Any]) -> None:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    with path.open("xb") as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())


def _object(value: object, label: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise DataReadinessError(f"{label} must be an object")
    return cast(dict[str, Any], value)


def _string_tuple(value: object, label: str) -> tuple[str, ...]:
    if not isinstance(value, list) or not value or any(not isinstance(item, str) or not item for item in value):
        raise DataReadinessError(f"{label} must be a non-empty list of column names")
    result = tuple(cast(list[str], value))
    if len(set(result)) != len(result):
        raise DataReadinessError(f"{label} contains duplicate columns")
    return result


def _strict_bool(value: object) -> bool:
    return isinstance(value, bool | np.bool_) and bool(value)


def _required_finite(mapping: Mapping[str, Any], key: str) -> float:
    value = mapping.get(key)
    if not isinstance(value, int | float) or isinstance(value, bool) or not math.isfinite(float(value)):
        raise DataReadinessError(f"validation metric {key} is unavailable")
    return float(value)


def _resolve_inside(root: Path, raw: object) -> Path:
    if not isinstance(raw, str) or not raw or Path(raw).is_absolute():
        raise DataReadinessError("dataset partition path must be relative")
    path = (root / raw).resolve()
    if root not in path.parents or not path.is_file():
        raise DataReadinessError("dataset partition path escapes its authority or is missing")
    return path


def _iso(value: object) -> str:
    return str(pd.Timestamp(value).isoformat())


def feature_set_sha256(feature_columns: Sequence[str]) -> str:
    """Stable feature identity useful to the future dataset publisher."""

    payload = json.dumps(list(feature_columns), separators=(",", ":"), ensure_ascii=True).encode("ascii")
    return hashlib.sha256(payload).hexdigest()
