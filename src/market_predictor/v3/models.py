from __future__ import annotations

import hashlib
import importlib
import importlib.metadata
import json
import os
import subprocess
from pathlib import Path
from typing import Any, Literal, Self

import joblib
import numpy as np
import pandas as pd
from pydantic import Field, field_validator, model_validator
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, ndcg_score, roc_auc_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.process_memory import process_memory_snapshot, release_process_memory
from market_predictor.registry import feature_schema_hash, manifest_path_for, write_model_manifest
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.features import V3_FEATURE_SCHEMA_VERSION, core_feature_columns
from market_predictor.v3.partitions import DevelopmentShadowPolicy, assert_development_only
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract
from market_predictor.v3.validation import V3Fold, V3PurgedWalkForwardSplit, deterministic_ticker_holdout

ModelFamily = Literal["B0", "B1", "B2", "R1", "D1"]
MODEL_FAMILIES: tuple[ModelFamily, ...] = ("B0", "B1", "B2", "R1", "D1")
V3_MODEL_SCHEMA_VERSION = "ml_v3.model.v1"

V3_TRAINING_REQUIRED_COLUMNS = frozenset(
    {
        "ticker",
        "decision_time_utc",
        "feature_available_at_utc",
        "entry_time_utc",
        "primary_exit_time_utc",
        "session_date_et",
        "decision_group_id",
        "universe_snapshot_id",
        "price_feed",
        "ranking_target",
        "ranking_grade",
        "ranking_group_size",
        "stop_before_target",
        "overlap_weight",
        "feature_schema_version",
        "label_schema_version",
        "label_config_json",
        "label_config_hash",
    }
)
V3_PREDICTION_AUDIT_COLUMNS = frozenset(
    {
        "path_realized_return_net",
        "independent_event_id",
        "concurrent_label_count",
        "cooldown_bars",
        "market_regime",
    }
)


class V3TrainingConfig(FrozenContract):
    families: tuple[ModelFamily, ...] = MODEL_FAMILIES
    n_splits: int = Field(default=4, ge=2, le=10)
    embargo_sessions: int = Field(default=1, ge=0, le=10)
    min_train_sessions: int = Field(default=20, ge=2)
    min_train_rows: int = Field(default=500, ge=1)
    min_features: int = Field(default=15, ge=1)
    min_fold_feature_non_null_rate: float = Field(default=0.05, ge=0, le=1)
    ticker_holdout_fraction: float = Field(default=0.2, gt=0, lt=1)
    random_seed: int = 42
    top_k: int = Field(default=10, ge=1)
    max_iter: int = Field(default=150, ge=10)
    learning_rate: float = Field(default=0.05, gt=0, le=1)
    xgboost_max_depth: int = Field(default=5, ge=1, le=16)
    xgboost_max_bin: int = Field(default=64, ge=32, le=1024)
    xgboost_n_jobs: int = Field(default=1, ge=1)
    max_training_memory_gb: float = Field(default=4.0, ge=1.0, le=256.0)
    memory_guard_headroom_gb: float = Field(default=0.25, ge=0.1, le=2.0)
    training_dataset_fingerprint: str | None = Field(default=None, min_length=64, max_length=64)
    continue_on_family_error: bool = True
    schema_version: str = ML_V3_SCHEMA_VERSION

    @field_validator("families")
    @classmethod
    def validate_families(cls, value: tuple[ModelFamily, ...]) -> tuple[ModelFamily, ...]:
        normalized = tuple(dict.fromkeys(value))
        if not normalized:
            raise ValueError("at least one model family is required")
        return normalized

    @model_validator(mode="after")
    def validate_holdout_capacity(self) -> Self:
        if self.top_k > self.min_train_rows:
            raise ValueError("top_k cannot exceed min_train_rows")
        if self.memory_guard_headroom_gb >= self.max_training_memory_gb:
            raise ValueError("memory_guard_headroom_gb must be lower than max_training_memory_gb")
        return self


def training_input_columns() -> list[str]:
    """Return the bounded column projection required by the V3 trainer."""
    return sorted(V3_TRAINING_REQUIRED_COLUMNS | V3_PREDICTION_AUDIT_COLUMNS | set(core_feature_columns()))


class DeterministicTechnicalRanker:
    """B0 non-ML floor using frozen point-in-time technical ranks."""

    weights: dict[str, float] = {
        "xs_rank_return_3bar": 0.25,
        "xs_rank_relative_volume_same_minute_20d": 0.20,
        "xs_rank_rel_return_3bar_vs_qqq": 0.20,
        "xs_rank_rel_return_3bar_vs_sector": 0.15,
        "xs_rank_dist_session_vwap": 0.10,
        "xs_rank_atr_pct": -0.05,
        "regime_risk_off": -0.05,
    }

    def fit(self, x: pd.DataFrame, y: pd.Series | None = None) -> DeterministicTechnicalRanker:
        return self

    def predict(self, x: pd.DataFrame) -> np.ndarray:
        score = pd.Series(0.0, index=x.index)
        for feature, weight in self.weights.items():
            neutral = 0.0 if feature.startswith("regime_") else 0.5
            values = pd.to_numeric(x.get(feature, neutral), errors="coerce")
            if not isinstance(values, pd.Series):
                values = pd.Series(float(values), index=x.index)
            score += weight * values.fillna(neutral)
        return np.asarray(score.to_numpy(dtype=float), dtype=float)


def audit_feature_coverage(
    data: pd.DataFrame,
    folds: list[V3Fold],
    *,
    minimum_rate: float,
) -> tuple[list[str], pd.DataFrame]:
    candidates = [feature for feature in core_feature_columns() if feature in data.columns]
    records: list[dict[str, object]] = []
    selected: list[str] = []
    for feature in candidates:
        values = pd.to_numeric(data[feature], errors="coerce")
        fold_rates = [float(values.iloc[fold.train_indices].notna().mean()) for fold in folds]
        minimum = min(fold_rates)
        eligible = minimum >= minimum_rate
        records.append(
            {
                "feature": feature,
                "minimum_train_fold_non_null_rate": minimum,
                "maximum_train_fold_non_null_rate": max(fold_rates),
                "eligible": eligible,
            }
        )
        if eligible:
            selected.append(feature)
    missing = sorted(set(core_feature_columns()).difference(data.columns))
    records.extend(
        {
            "feature": feature,
            "minimum_train_fold_non_null_rate": 0.0,
            "maximum_train_fold_non_null_rate": 0.0,
            "eligible": False,
        }
        for feature in missing
    )
    return selected, pd.DataFrame(records).sort_values(["eligible", "feature"], ascending=[False, True]).reset_index(drop=True)


def train_v3_model_suite(
    dataset: pd.DataFrame,
    output_dir: Path,
    *,
    config: V3TrainingConfig = V3TrainingConfig(),
    overwrite: bool = False,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame]:
    data = _prepare_training_data(dataset)
    _assert_memory_budget(config, "validated dataset")
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=config.embargo_sessions,
        min_train_sessions=config.min_train_sessions,
        min_train_rows=config.min_train_rows,
    )
    folds = splitter.split(data)
    features, feature_audit = audit_feature_coverage(
        data,
        folds,
        minimum_rate=config.min_fold_feature_non_null_rate,
    )
    if len(features) < config.min_features:
        raise DataReadinessError(f"only {len(features)} features pass every training fold; require {config.min_features}")
    _compact_feature_storage(data, features)
    _assert_memory_budget(config, "compacted feature matrix")
    holdout = deterministic_ticker_holdout(
        data["ticker"],
        fraction=config.ticker_holdout_fraction,
        seed=config.random_seed,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    reports: dict[str, Any] = {}
    predictions: list[pd.DataFrame] = []
    for family in config.families:
        try:
            result, family_predictions = _train_family(
                family,
                data,
                features,
                folds,
                holdout,
                output_dir=output_dir,
                config=config,
                overwrite=overwrite,
            )
        except (DataReadinessError, FileExistsError) as exc:
            if not config.continue_on_family_error:
                raise
            reports[family] = {
                "family": family,
                "status": "failed",
                "error_type": type(exc).__name__,
                "error": str(exc),
            }
            continue
        reports[family] = result
        predictions.append(family_predictions)
    failed_families = [family for family, result in reports.items() if result.get("status") == "failed"]
    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
    _assert_memory_budget(config, "OOF evidence assembly")
    report: dict[str, Any] = {
        "schema": "ml_v3.training_report.v1",
        "config": config.model_dump(mode="json"),
        "feature_schema_version": V3_FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(features),
        "selected_features": features,
        "selected_feature_count": len(features),
        "ticker_holdout": sorted(holdout),
        "folds": [fold.audit_record() for fold in folds],
        "status": "partial_failure" if failed_families else "complete",
        "failed_families": failed_families,
        "models": reports,
        "memory": _memory_audit(config),
    }
    return report, combined, feature_audit


def _train_family(
    family: ModelFamily,
    data: pd.DataFrame,
    features: list[str],
    folds: list[V3Fold],
    holdout: frozenset[str],
    *,
    output_dir: Path,
    config: V3TrainingConfig,
    overwrite: bool,
) -> tuple[dict[str, Any], pd.DataFrame]:
    artifact_path = output_dir / f"{family.lower()}_candidate.joblib"
    if artifact_path.exists() and not overwrite:
        raise FileExistsError(f"candidate artifact already exists: {artifact_path}")
    target_column = _target_column(family)
    run_id = _run_id(data, family, features, target_column, config)
    eligible = data[target_column].notna()
    if family == "R1":
        eligible &= data["ranking_group_size"].ge(2)
    predictions: list[pd.DataFrame] = []
    eligible_values = eligible.to_numpy(dtype=bool)
    for fold in folds:
        train_indices = fold.train_indices[eligible_values[fold.train_indices]]
        test_indices = fold.test_indices[eligible_values[fold.test_indices]]
        if len(train_indices) == 0 or len(test_indices) == 0:
            raise DataReadinessError(f"{family} fold {fold.fold} has no eligible train or test rows")
        model = _new_model(family, config)
        _fit(model, family, data, features, target_column, config=config, row_indices=train_indices)
        predictions.append(
            _prediction_frame(
                model,
                family,
                data,
                features,
                target_column,
                fold.fold,
                "walk_forward",
                run_id,
                row_indices=test_indices,
            )
        )
        del model
        _release_training_memory()
        ticker = data["ticker"]
        holdout_train_indices = train_indices[~ticker.iloc[train_indices].isin(holdout).to_numpy()]
        holdout_test_indices = test_indices[ticker.iloc[test_indices].isin(holdout).to_numpy()]
        if len(holdout_train_indices) == 0 or len(holdout_test_indices) == 0:
            raise DataReadinessError(f"{family} fold {fold.fold} cannot produce ticker-holdout evidence")
        holdout_model = _new_model(family, config)
        _fit(
            holdout_model,
            family,
            data,
            features,
            target_column,
            config=config,
            row_indices=holdout_train_indices,
        )
        predictions.append(
            _prediction_frame(
                holdout_model,
                family,
                data,
                features,
                target_column,
                fold.fold,
                "ticker_holdout",
                run_id,
                row_indices=holdout_test_indices,
            )
        )
        del holdout_model
        _release_training_memory()
    oof = pd.concat(predictions, ignore_index=True)
    predictions.clear()
    _release_training_memory()
    _assert_memory_budget(config, f"{family} OOF evidence")
    main_metrics = _model_metrics(oof[oof["audit_scope"] == "walk_forward"], family=family, top_k=config.top_k)
    holdout_metrics = _model_metrics(oof[oof["audit_scope"] == "ticker_holdout"], family=family, top_k=config.top_k)
    final_indices = np.flatnonzero(eligible_values)
    final_model = _new_model(family, config)
    _fit(final_model, family, data, features, target_column, config=config, row_indices=final_indices)
    metadata_columns = [
        "ticker",
        "decision_time_utc",
        "label_schema_version",
        "label_config_hash",
        "label_config_json",
        "universe_snapshot_id",
        target_column,
    ]
    metadata = data.loc[:, metadata_columns].iloc[final_indices]
    payload = {
        "schema_version": V3_MODEL_SCHEMA_VERSION,
        "family": family,
        "model": final_model,
        "features": features,
        "target_col": target_column,
        "score_semantics": "downside_probability" if family == "D1" else "opportunity_score",
        "feature_schema_version": V3_FEATURE_SCHEMA_VERSION,
        "feature_schema_hash": feature_schema_hash(features),
        "run_id": run_id,
        "label_schema_version": str(metadata["label_schema_version"].iloc[0]),
        "label_config_hash": str(metadata["label_config_hash"].iloc[0]),
        "label_config": json.loads(str(metadata["label_config_json"].iloc[0])),
        "universe_snapshot_ids": sorted(metadata["universe_snapshot_id"].astype(str).unique()),
        "config": config.model_dump(mode="json"),
    }
    joblib.dump(payload, artifact_path)
    manifest = write_model_manifest(
        model_path=artifact_path,
        model_type=f"v3_{family.lower()}",
        schema_version=V3_MODEL_SCHEMA_VERSION,
        target_col=target_column,
        features=features,
        training_data=metadata[["ticker", "decision_time_utc", target_column]],
        metrics={"walk_forward": main_metrics, "ticker_holdout": holdout_metrics},
        validation_split="v3_session_grouped_purged_walk_forward",
        extra={
            "run_id": run_id,
            "family": family,
            "random_seed": config.random_seed,
            "training_dataset_fingerprint": config.training_dataset_fingerprint,
            "label_schema_version": str(metadata["label_schema_version"].iloc[0]),
            "label_config_hash": str(metadata["label_config_hash"].iloc[0]),
            "label_config": json.loads(str(metadata["label_config_json"].iloc[0])),
            "universe_snapshot_ids": sorted(metadata["universe_snapshot_id"].astype(str).unique()),
            "ticker_holdout": sorted(holdout),
            "folds": [fold.audit_record() for fold in folds],
            "dependency_versions": _dependency_versions(family),
            "execution_mode": "cpu",
            "code_commit": _code_commit(),
        },
    )
    del payload, final_model, metadata
    _release_training_memory()
    result = {
        "family": family,
        "status": "complete",
        "artifact_path": str(artifact_path),
        "manifest_path": str(manifest_path_for(artifact_path)),
        "artifact_sha256": manifest["artifact_sha256"],
        "run_id": run_id,
        "target_col": target_column,
        "walk_forward": main_metrics,
        "ticker_holdout": holdout_metrics,
    }
    return result, oof


def _new_model(family: ModelFamily, config: V3TrainingConfig) -> Any:
    if family == "B0":
        return DeterministicTechnicalRanker()
    if family == "B1":
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scaler", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(max_iter=config.max_iter, random_state=config.random_seed, class_weight="balanced"),
                ),
            ]
        )
    if family in {"B2", "D1"}:
        return Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=config.max_iter,
                        learning_rate=config.learning_rate,
                        random_state=config.random_seed,
                        class_weight="balanced",
                    ),
                ),
            ]
        )
    try:
        xgboost = importlib.import_module("xgboost")
    except ImportError as exc:
        raise DataReadinessError("R1 requires the optional dependency: python -m pip install -e '.[ranking]'") from exc
    return xgboost.XGBRanker(
        objective="rank:ndcg",
        eval_metric="ndcg@10",
        n_estimators=config.max_iter,
        learning_rate=config.learning_rate,
        max_depth=config.xgboost_max_depth,
        max_bin=config.xgboost_max_bin,
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.xgboost_n_jobs,
        callbacks=[_memory_guard_callback(xgboost, config)],
    )


def _fit(
    model: Any,
    family: ModelFamily,
    data: pd.DataFrame,
    features: list[str],
    target_column: str,
    *,
    config: V3TrainingConfig,
    row_indices: np.ndarray,
) -> None:
    _assert_memory_budget(config, f"{family} fit preflight")
    fit_columns = list(dict.fromkeys([*features, target_column, "overlap_weight", "decision_group_id", "ticker"]))
    selected = data.loc[:, fit_columns].iloc[row_indices]
    _assert_memory_budget(config, f"{family} selected rows")
    x = selected[features]
    y = pd.to_numeric(selected[target_column], errors="raise")
    weights = pd.to_numeric(selected["overlap_weight"], errors="raise").clip(lower=1e-6)
    if family == "B0":
        model.fit(x, y)
        return
    if family == "R1":
        row_order = _ranking_row_order(selected)
        if row_order is not None:
            selected = selected.iloc[row_order]
            x = selected[features]
            y = pd.to_numeric(selected[target_column], errors="raise")
            weights = pd.to_numeric(selected["overlap_weight"], errors="raise").clip(lower=1e-6)
        query_id = pd.factorize(selected["decision_group_id"], sort=False)[0].astype(np.int32, copy=False)
        starts = np.r_[0, np.flatnonzero(query_id[1:] != query_id[:-1]) + 1]
        counts = np.diff(np.r_[starts, len(query_id)])
        weight_values = weights.to_numpy(dtype=np.float32, copy=False)
        group_weights = np.add.reduceat(weight_values, starts) / counts
        matrix = x.to_numpy(dtype=np.float32, copy=False)
        _assert_memory_budget(config, "R1 compact matrix")
        try:
            model.fit(
                matrix,
                y.to_numpy(dtype=np.int16, copy=False),
                qid=query_id,
                sample_weight=group_weights.astype(np.float32, copy=False),
            )
        finally:
            model.set_params(callbacks=None)
        return
    if y.nunique() < 2:
        raise DataReadinessError(f"{family} training target has only one class")
    model.fit(x, y.astype(int), classifier__sample_weight=weights)


def _prediction_frame(
    model: Any,
    family: ModelFamily,
    data: pd.DataFrame,
    features: list[str],
    target_column: str,
    fold: int,
    audit_scope: str,
    model_run_id: str,
    *,
    row_indices: np.ndarray,
) -> pd.DataFrame:
    selected = data.iloc[row_indices]
    if family in {"B1", "B2", "D1"}:
        score = np.asarray(model.predict_proba(selected[features]))[:, 1]
    elif family == "R1":
        score = np.asarray(model.predict(selected[features].to_numpy(dtype=np.float32, copy=False)), dtype=float)
    else:
        score = np.asarray(model.predict(selected[features]), dtype=float)
    columns = [
        "ticker",
        "decision_time_utc",
        "entry_time_utc",
        "primary_exit_time_utc",
        "session_date_et",
        "decision_group_id",
        "ranking_target",
        "ranking_grade",
        "stop_before_target",
    ]
    columns.extend(
        column
        for column in (
            "path_realized_return_net",
            "independent_event_id",
            "concurrent_label_count",
            "cooldown_bars",
            "market_regime",
        )
        if column in selected.columns
    )
    output = selected[columns].copy()
    output["family"] = family
    output["fold"] = fold
    output["audit_scope"] = audit_scope
    output["model_run_id"] = model_run_id
    output["target_col"] = target_column
    output["target"] = pd.to_numeric(selected[target_column], errors="coerce").to_numpy()
    output["score"] = score
    output["opportunity_score"] = 1.0 - score if family == "D1" else score
    return output


def _model_metrics(predictions: pd.DataFrame, *, family: ModelFamily, top_k: int) -> dict[str, Any]:
    if predictions.empty:
        raise DataReadinessError(f"{family} has no predictions to audit")
    metrics: dict[str, Any] = {
        "rows": len(predictions),
        "tickers": int(predictions["ticker"].nunique()),
        "sessions": int(pd.Series(predictions["session_date_et"]).nunique()),
    }
    target = pd.to_numeric(predictions["target"], errors="coerce")
    score = pd.to_numeric(predictions["score"], errors="coerce")
    if family in {"B1", "B2", "D1"} and target.nunique() >= 2:
        metrics["roc_auc"] = float(roc_auc_score(target, score))
        metrics["average_precision"] = float(average_precision_score(target, score))
        base_rate = float(target.mean())
        top_count = max(1, int(np.ceil(len(predictions) * 0.1)))
        top_rate = float(target.loc[score.nlargest(top_count).index].mean())
        metrics["top_decile_lift"] = top_rate / base_rate if base_rate > 0 else None
    ndcg_values: list[float] = []
    top_returns: list[float] = []
    top_positive: list[float] = []
    for _, group in predictions.groupby("decision_group_id", sort=False):
        grade = pd.to_numeric(group["ranking_grade"], errors="coerce")
        opportunity = pd.to_numeric(group["opportunity_score"], errors="coerce")
        valid = grade.notna() & opportunity.notna()
        if int(valid.sum()) < 2:
            continue
        count = min(top_k, int(valid.sum()))
        ndcg_values.append(float(ndcg_score([grade[valid].to_numpy()], [opportunity[valid].to_numpy()], k=count)))
        selected = group.loc[valid].nlargest(count, "opportunity_score")
        returns = pd.to_numeric(selected["ranking_target"], errors="coerce")
        top_returns.append(float(returns.mean()))
        top_positive.append(float(returns.gt(0).mean()))
    metrics["ranking_groups"] = len(ndcg_values)
    metrics["mean_ndcg_at_k"] = float(np.mean(ndcg_values)) if ndcg_values else None
    metrics["mean_top_k_excess_return"] = float(np.mean(top_returns)) if top_returns else None
    metrics["mean_top_k_positive_rate"] = float(np.mean(top_positive)) if top_positive else None
    return metrics


def _prepare_training_data(dataset: pd.DataFrame) -> pd.DataFrame:
    missing = sorted(V3_TRAINING_REQUIRED_COLUMNS.difference(dataset.columns))
    if missing:
        raise DataReadinessError(f"V3 training dataset missing columns: {', '.join(missing)}")
    # The trainer owns this projected frame and normalizes it in place to avoid
    # retaining the original float64/string blocks alongside compact storage.
    data = dataset
    data["decision_time_utc"] = data["decision_time_utc"].map(_aware_timestamp)
    data["feature_available_at_utc"] = data["feature_available_at_utc"].map(_aware_timestamp)
    data["entry_time_utc"] = data["entry_time_utc"].map(_aware_timestamp)
    data["primary_exit_time_utc"] = data["primary_exit_time_utc"].map(_aware_timestamp)
    time_columns = ["decision_time_utc", "feature_available_at_utc", "entry_time_utc", "primary_exit_time_utc"]
    if bool(data[time_columns].isna().any(axis=None)):
        raise DataReadinessError("V3 training contains invalid decision, availability, or entry timestamps")
    if bool((data["feature_available_at_utc"] > data["decision_time_utc"]).any()):
        raise DataReadinessError("V3 training contains features unavailable at decision time")
    if bool((data["entry_time_utc"] <= data["decision_time_utc"]).any()):
        raise DataReadinessError("V3 training entry must follow the completed decision bar")
    if bool((data["primary_exit_time_utc"] < data["entry_time_utc"]).any()):
        raise DataReadinessError("V3 primary exit cannot precede entry")
    declared_session = pd.to_datetime(data["session_date_et"], errors="coerce").dt.date
    decision_session = data["decision_time_utc"].dt.tz_convert("America/New_York").dt.date
    entry_session = data["entry_time_utc"].dt.tz_convert("America/New_York").dt.date
    exit_session = data["primary_exit_time_utc"].dt.tz_convert("America/New_York").dt.date
    if bool(declared_session.isna().any() | declared_session.ne(decision_session).any()):
        raise DataReadinessError("session_date_et does not match the decision timestamp")
    if bool(pd.Series(entry_session).ne(declared_session).any()):
        raise DataReadinessError("V3 training labels cannot cross Eastern sessions")
    if bool(pd.Series(exit_session).ne(declared_session).any()):
        raise DataReadinessError("V3 primary exits cannot cross Eastern sessions")
    assert_development_only(
        data,
        policy=DevelopmentShadowPolicy(timestamp_column="decision_time_utc"),
    )
    schemas = set(data["feature_schema_version"].dropna().astype(str))
    if schemas != {V3_FEATURE_SCHEMA_VERSION}:
        raise DataReadinessError(f"unexpected V3 feature schemas: {sorted(schemas)}")
    label_schemas = set(data["label_schema_version"].dropna().astype(str))
    label_configs = set(data["label_config_json"].dropna().astype(str))
    label_hashes = set(data["label_config_hash"].dropna().astype(str))
    if len(label_schemas) != 1 or len(label_configs) != 1 or len(label_hashes) != 1:
        raise DataReadinessError("V3 training must contain one frozen label schema and configuration")
    label_config_json = next(iter(label_configs))
    if hashlib.sha256(label_config_json.encode()).hexdigest() != next(iter(label_hashes)):
        raise DataReadinessError("label_config_hash does not match label_config_json")
    try:
        decoded_label_config = json.loads(label_config_json)
    except json.JSONDecodeError as exc:
        raise DataReadinessError("label_config_json is invalid") from exc
    if not isinstance(decoded_label_config, dict):
        raise DataReadinessError("label_config_json must contain an object")
    if bool(data.duplicated(["decision_group_id", "ticker"]).any()):
        raise DataReadinessError("V3 training contains duplicate ticker/query rows")
    query_timestamp_count = data.groupby("decision_group_id")["decision_time_utc"].nunique()
    if bool(query_timestamp_count.ne(1).any()):
        raise DataReadinessError("decision_group_id must represent exactly one decision timestamp")
    feed = data["price_feed"].fillna("unknown").astype(str).str.lower().str.strip()
    if bool(feed.ne("sip").any()):
        raise DataReadinessError("V3 volume-sensitive training requires SIP price_feed provenance")
    if bool(data["universe_snapshot_id"].fillna("").astype(str).str.strip().eq("").any()):
        raise DataReadinessError("V3 training requires universe_snapshot_id on every row")
    actual_group_size = data.groupby("decision_group_id")["ticker"].transform("size")
    declared_group_size = pd.to_numeric(data["ranking_group_size"], errors="coerce")
    if bool(declared_group_size.ne(actual_group_size).any()):
        raise DataReadinessError("ranking_group_size does not match the decision query")
    overlap_weight = pd.to_numeric(data["overlap_weight"], errors="coerce")
    if bool((overlap_weight.isna() | overlap_weight.le(0) | overlap_weight.gt(1)).any()):
        raise DataReadinessError("overlap_weight must be in the interval (0, 1]")
    stop_target = pd.to_numeric(data["stop_before_target"], errors="coerce")
    if not set(stop_target.dropna().unique()).issubset({0, 1}):
        raise DataReadinessError("stop_before_target must be binary")
    ranking_target = pd.to_numeric(data["ranking_target"], errors="coerce")
    if bool(ranking_target.isna().any()):
        raise DataReadinessError("ranking_target must be numeric and complete")
    ranking_grade = pd.to_numeric(data["ranking_grade"], errors="coerce")
    valid_grade = ranking_grade.dropna()
    if bool((valid_grade.lt(0) | valid_grade.mod(1).ne(0)).any()):
        raise DataReadinessError("ranking_grade must contain non-negative integers or null")
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["target_positive_excess"] = ranking_target.gt(0).where(ranking_target.notna(), pd.NA).astype("Int64")
    for column in (
        "ticker",
        "decision_group_id",
        "universe_snapshot_id",
        "price_feed",
        "feature_schema_version",
        "label_schema_version",
        "label_config_json",
        "label_config_hash",
    ):
        data[column] = data[column].astype("category")
    if not data["decision_time_utc"].is_monotonic_increasing:
        data = data.sort_values(["session_date_et", "decision_time_utc", "ticker"])
    return data.reset_index(drop=True) if not isinstance(data.index, pd.RangeIndex) else data


def _target_column(family: ModelFamily) -> str:
    if family in {"B0", "R1"}:
        return "ranking_grade"
    if family == "D1":
        return "stop_before_target"
    return "target_positive_excess"


def _run_id(
    data: pd.DataFrame,
    family: ModelFamily,
    features: list[str],
    target_column: str,
    config: V3TrainingConfig,
) -> str:
    digest = hashlib.sha256()
    identity_columns = [
        "ticker",
        "decision_group_id",
        "universe_snapshot_id",
        "feature_schema_version",
        "label_config_hash",
        target_column,
        *features,
    ]
    for start in range(0, len(data), 50_000):
        chunk = data.loc[:, identity_columns].iloc[start : start + 50_000]
        digest.update(pd.util.hash_pandas_object(chunk, index=True).to_numpy().tobytes())
    digest.update(
        json.dumps(
            {
                "family": family,
                "features": features,
                "config": config.model_dump(mode="json", exclude={"families", "continue_on_family_error"}),
            },
            sort_keys=True,
        ).encode()
    )
    return digest.hexdigest()[:24]


def _compact_feature_storage(data: pd.DataFrame, features: list[str]) -> None:
    for feature in features:
        data[feature] = pd.to_numeric(data[feature], errors="coerce").astype(np.float32)


def _ranking_row_order(data: pd.DataFrame) -> np.ndarray | None:
    query_id = pd.factorize(data["decision_group_id"], sort=False)[0]
    starts = np.r_[0, np.flatnonzero(query_id[1:] != query_id[:-1]) + 1]
    if len(starts) == len(np.unique(query_id)):
        return None
    group = data["decision_group_id"].astype(str).to_numpy()
    ticker = data["ticker"].astype(str).to_numpy()
    return np.lexsort((ticker, group))


def _memory_guard_callback(xgboost: Any, config: V3TrainingConfig) -> Any:
    def after_iteration(self: Any, model: Any, epoch: int, evals_log: dict[str, Any]) -> bool:
        _assert_memory_budget(config, f"R1 boosting iteration {epoch}")
        return False

    callback_type = type(
        "MemoryGuard",
        (xgboost.callback.TrainingCallback,),
        {"after_iteration": after_iteration},
    )
    return callback_type()


def _assert_memory_budget(
    config: V3TrainingConfig,
    stage: str,
) -> None:
    rss = _current_process_rss_bytes()
    if rss is None:
        return
    limit = int((config.max_training_memory_gb - config.memory_guard_headroom_gb) * 1024**3)
    if rss > limit:
        raise DataReadinessError(
            f"training memory guard stopped {stage}: RSS {_gib(rss):.2f} GiB exceeds "
            f"the {_gib(limit):.2f} GiB safety threshold for the {config.max_training_memory_gb:.2f} GiB hard budget"
        )


def _current_process_rss_bytes() -> int | None:
    snapshot = process_memory_snapshot()
    return snapshot[0] if snapshot is not None else None


def _release_training_memory() -> None:
    release_process_memory()


def _memory_audit(config: V3TrainingConfig) -> dict[str, float | None]:
    snapshot = process_memory_snapshot()
    return {
        "hard_budget_gib": config.max_training_memory_gb,
        "safety_threshold_gib": config.max_training_memory_gb - config.memory_guard_headroom_gb,
        "current_working_set_gib": _gib(snapshot[0]) if snapshot is not None else None,
        "peak_working_set_gib": _gib(snapshot[1]) if snapshot is not None else None,
    }


def _gib(value: int) -> float:
    return value / 1024**3


def _dependency_versions(family: ModelFamily) -> dict[str, str | None]:
    packages = ["numpy", "pandas", "scikit-learn", "joblib"]
    if family == "R1":
        packages.append("xgboost")
    versions: dict[str, str | None] = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return versions


def _code_commit() -> str | None:
    configured = os.getenv("MARKET_PREDICTOR_CODE_COMMIT", "").strip()
    if configured:
        return configured
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _aware_timestamp(value: object) -> pd.Timestamp:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError):
        return pd.NaT
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        return pd.NaT
    return timestamp.tz_convert("UTC")
