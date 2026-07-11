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

from market_predictor.registry import feature_schema_hash, manifest_path_for, write_model_manifest
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.features import V3_FEATURE_SCHEMA_VERSION, core_feature_columns
from market_predictor.v3.partitions import DevelopmentShadowPolicy, assert_development_only
from market_predictor.v3.schema import ML_V3_SCHEMA_VERSION, FrozenContract
from market_predictor.v3.validation import V3Fold, V3PurgedWalkForwardSplit, deterministic_ticker_holdout

ModelFamily = Literal["B0", "B1", "B2", "R1", "D1"]
MODEL_FAMILIES: tuple[ModelFamily, ...] = ("B0", "B1", "B2", "R1", "D1")
V3_MODEL_SCHEMA_VERSION = "ml_v3.model.v1"


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
    xgboost_n_jobs: int = Field(default=1, ge=1)
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
        return self


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
        fold_rates = [float(pd.to_numeric(data.iloc[fold.train_indices][feature], errors="coerce").notna().mean()) for fold in folds]
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
    }
    combined = pd.concat(predictions, ignore_index=True) if predictions else pd.DataFrame()
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
    for fold in folds:
        train = data.iloc[fold.train_indices].loc[eligible.iloc[fold.train_indices]].copy()
        test = data.iloc[fold.test_indices].loc[eligible.iloc[fold.test_indices]].copy()
        if train.empty or test.empty:
            raise DataReadinessError(f"{family} fold {fold.fold} has no eligible train or test rows")
        model = _new_model(family, config)
        _fit(model, family, train, features, target_column)
        predictions.append(
            _prediction_frame(model, family, test, features, target_column, fold.fold, "walk_forward", run_id)
        )
        holdout_train = train[~train["ticker"].isin(holdout)].copy()
        holdout_test = test[test["ticker"].isin(holdout)].copy()
        if holdout_train.empty or holdout_test.empty:
            raise DataReadinessError(f"{family} fold {fold.fold} cannot produce ticker-holdout evidence")
        holdout_model = _new_model(family, config)
        _fit(holdout_model, family, holdout_train, features, target_column)
        predictions.append(
            _prediction_frame(
                holdout_model,
                family,
                holdout_test,
                features,
                target_column,
                fold.fold,
                "ticker_holdout",
                run_id,
            )
        )
    oof = pd.concat(predictions, ignore_index=True)
    main_metrics = _model_metrics(oof[oof["audit_scope"] == "walk_forward"], family=family, top_k=config.top_k)
    holdout_metrics = _model_metrics(oof[oof["audit_scope"] == "ticker_holdout"], family=family, top_k=config.top_k)
    final_data = data.loc[eligible].copy()
    final_model = _new_model(family, config)
    _fit(final_model, family, final_data, features, target_column)
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
        "label_schema_version": str(final_data["label_schema_version"].iloc[0]),
        "label_config_hash": str(final_data["label_config_hash"].iloc[0]),
        "label_config": json.loads(str(final_data["label_config_json"].iloc[0])),
        "universe_snapshot_ids": sorted(final_data["universe_snapshot_id"].astype(str).unique()),
        "config": config.model_dump(mode="json"),
    }
    joblib.dump(payload, artifact_path)
    manifest = write_model_manifest(
        model_path=artifact_path,
        model_type=f"v3_{family.lower()}",
        schema_version=V3_MODEL_SCHEMA_VERSION,
        target_col=target_column,
        features=features,
        training_data=final_data,
        metrics={"walk_forward": main_metrics, "ticker_holdout": holdout_metrics},
        validation_split="v3_session_grouped_purged_walk_forward",
        extra={
            "run_id": run_id,
            "family": family,
            "random_seed": config.random_seed,
            "training_dataset_fingerprint": config.training_dataset_fingerprint,
            "label_schema_version": str(final_data["label_schema_version"].iloc[0]),
            "label_config_hash": str(final_data["label_config_hash"].iloc[0]),
            "label_config": json.loads(str(final_data["label_config_json"].iloc[0])),
            "universe_snapshot_ids": sorted(final_data["universe_snapshot_id"].astype(str).unique()),
            "ticker_holdout": sorted(holdout),
            "folds": [fold.audit_record() for fold in folds],
            "dependency_versions": _dependency_versions(family),
            "execution_mode": "cpu",
            "code_commit": _code_commit(),
        },
    )
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
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.xgboost_n_jobs,
    )


def _fit(model: Any, family: ModelFamily, data: pd.DataFrame, features: list[str], target_column: str) -> None:
    x = data[features]
    y = pd.to_numeric(data[target_column], errors="raise")
    weights = pd.to_numeric(data["overlap_weight"], errors="raise").clip(lower=1e-6)
    if family == "B0":
        model.fit(x, y)
        return
    if family == "R1":
        ordered = data.assign(_target=y).sort_values(["decision_group_id", "ticker"])
        query_id = pd.factorize(ordered["decision_group_id"], sort=False)[0]
        group_weights = ordered.groupby("decision_group_id", sort=False)["overlap_weight"].mean().to_numpy()
        model.fit(ordered[features], ordered["_target"].astype(int), qid=query_id, sample_weight=group_weights)
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
) -> pd.DataFrame:
    if family in {"B1", "B2", "D1"}:
        score = np.asarray(model.predict_proba(data[features]))[:, 1]
    else:
        score = np.asarray(model.predict(data[features]), dtype=float)
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
        if column in data.columns
    )
    output = data[columns].copy()
    output["family"] = family
    output["fold"] = fold
    output["audit_scope"] = audit_scope
    output["model_run_id"] = model_run_id
    output["target_col"] = target_column
    output["target"] = pd.to_numeric(data[target_column], errors="coerce").to_numpy()
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
    required = {
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
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise DataReadinessError(f"V3 training dataset missing columns: {', '.join(missing)}")
    data = dataset.copy()
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
    return data.sort_values(["session_date_et", "decision_time_utc", "ticker"]).reset_index(drop=True)


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
    identity = data[
        [
            "ticker",
            "decision_group_id",
            "universe_snapshot_id",
            "feature_schema_version",
            "label_config_hash",
            target_column,
            *features,
        ]
    ]
    digest.update(pd.util.hash_pandas_object(identity, index=True).to_numpy().tobytes())
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
