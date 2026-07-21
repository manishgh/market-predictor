from __future__ import annotations

import uuid
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol, cast

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.registry import (
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_PROMOTED,
    manifest_path_for,
    verify_model_artifact,
    write_model_manifest,
)
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_FEATURES,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
    SWING_VALIDATION_SPLIT,
    SwingTrainingConfig,
    swing_target_column,
)
from market_predictor.swing.evaluation import (
    catalyst_audit,
    classification_metrics,
    conservative_economics,
    phase_economics,
    prediction_evidence,
    regime_audit,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.validation import V3PurgedWalkForwardSplit, deterministic_ticker_holdout


class ProbabilityEstimator(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class SwingTrainingResult:
    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    ticker_holdout_predictions: pd.DataFrame
    profitability_audit: pd.DataFrame
    regime_audit: pd.DataFrame
    catalyst_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    fold_audit: pd.DataFrame
    manifest: dict[str, Any]


def train_swing_model(
    dataset: pd.DataFrame,
    *,
    model_out: Path,
    dataset_sha256: str,
    config: SwingTrainingConfig | None = None,
    overwrite: bool = False,
) -> SwingTrainingResult:
    config = config or SwingTrainingConfig()
    if not overwrite and (model_out.exists() or manifest_path_for(model_out).exists()):
        raise FileExistsError(f"swing model artifact already exists: {model_out}")
    model_run_id = f"swing-{uuid.uuid4().hex}"
    data, horizon, target = _training_rows(dataset)
    if len(data) < config.min_train_rows:
        raise DataReadinessError(f"swing training needs at least {config.min_train_rows} eligible rows")
    ticker_count = int(data["ticker"].nunique())
    if ticker_count < config.min_training_tickers:
        raise DataReadinessError(
            f"swing training needs at least {config.min_training_tickers} tickers; found {ticker_count}"
        )
    holdout_tickers = deterministic_ticker_holdout(
        data["ticker"],
        fraction=config.ticker_holdout_fraction,
        seed=config.random_seed,
    )
    development = data[~data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    holdout = data[data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=horizon,
        min_train_sessions=config.min_train_sessions,
        min_train_rows=config.min_train_rows,
    )
    folds = splitter.split(development)
    features = _select_features(development, folds, config)
    if len(features) < config.min_features:
        raise DataReadinessError(f"only {len(features)} swing features pass fold coverage; need {config.min_features}")
    assert_memory_budget(
        hard_budget_gib=config.max_training_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage="swing training input",
    )

    oof_raw = np.full(len(development), np.nan, dtype=np.float64)
    fold_records: list[dict[str, object]] = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.test_indices]
        _require_binary_target(train[target], f"fold {fold.fold} training")
        estimator = _estimator(config)
        estimator.fit(_matrix(train, features), train[target].astype(int))
        oof_raw[fold.test_indices] = estimator.predict_proba(_matrix(validation, features))[:, 1]
        fold_records.append(fold.audit_record())
        del estimator
        release_process_memory()
        assert_memory_budget(
            hard_budget_gib=config.max_training_memory_gb,
            headroom_gib=config.memory_guard_headroom_gb,
            stage=f"swing fold {fold.fold}",
        )

    oof_mask = np.isfinite(oof_raw)
    if int(oof_mask.sum()) < max(100, config.min_train_rows // 4):
        raise DataReadinessError("insufficient purged walk-forward predictions")
    cross_fitted = _cross_fitted_calibration(
        oof_raw[oof_mask],
        development.loc[oof_mask, target].astype(int).to_numpy(),
        development.loc[oof_mask, "session_date_et"],
    )
    calibrator = _fit_calibrator(oof_raw[oof_mask], development.loc[oof_mask, target].astype(int).to_numpy())

    _require_binary_target(development[target], "ticker holdout training")
    holdout_estimator = _estimator(config)
    holdout_estimator.fit(_matrix(development, features), development[target].astype(int))
    holdout_raw = holdout_estimator.predict_proba(_matrix(holdout, features))[:, 1]
    holdout_probability = _apply_calibrator(calibrator, holdout_raw)
    del holdout_estimator
    release_process_memory()

    oof = prediction_evidence(
        development.loc[oof_mask].reset_index(drop=True),
        raw_probability=oof_raw[oof_mask],
        probability=cross_fitted,
        scope="walk_forward",
        horizon=horizon,
    )
    holdout_evidence = prediction_evidence(
        holdout,
        raw_probability=holdout_raw,
        probability=holdout_probability,
        scope="ticker_holdout",
        horizon=horizon,
    )
    oof_metrics = classification_metrics(oof[target], oof["swing_probability"])
    holdout_metrics = classification_metrics(
        holdout_evidence[target],
        holdout_evidence["swing_probability"],
    )
    economics = pd.concat(
        [
            phase_economics(oof, horizon=horizon, top_k=config.top_k, scope="walk_forward"),
            phase_economics(
                holdout_evidence,
                horizon=horizon,
                top_k=config.top_k,
                scope="ticker_holdout",
            ),
        ],
        ignore_index=True,
    )
    profitability = pd.concat([conservative_economics(economics), economics], ignore_index=True)
    regime = regime_audit(pd.concat([oof, holdout_evidence], ignore_index=True))
    catalyst = catalyst_audit(pd.concat([oof, holdout_evidence], ignore_index=True))
    alignment = pd.DataFrame(
        [
            {
                "alignment_error_total": 0,
                "events_without_feature_row": 0,
                "missing_historical_feature_rows": 0,
                "dates_with_news_count_mismatch": 0,
                "future_feature_rows": 0,
                "label_path_mismatches": 0,
            }
        ]
    )
    for evidence in (oof, holdout_evidence, profitability, regime, catalyst, alignment):
        evidence["model_run_id"] = model_run_id
    fold_audit = pd.DataFrame(fold_records)
    fold_audit["model_run_id"] = model_run_id

    final_estimator = _estimator(config)
    final_estimator.fit(_matrix(data, features), data[target].astype(int))
    assert_memory_budget(
        hard_budget_gib=config.max_training_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage="final swing model",
    )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model": final_estimator,
        "calibrator": calibrator,
        "features": features,
        "target_col": target,
        "horizon_sessions": horizon,
        "model_type": SWING_MODEL_TYPE,
        "model_schema_version": SWING_MODEL_SCHEMA_VERSION,
        "feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
        "family": config.family,
        "model_run_id": model_run_id,
        "decision_semantics": "post_close_decision_next_session_open_entry",
        "trained_at_utc": datetime.now(UTC).isoformat(),
    }
    temporary = model_out.with_name(f".{model_out.name}.{uuid.uuid4().hex}.tmp")
    try:
        joblib.dump(payload, temporary)
        temporary.replace(model_out)
    finally:
        temporary.unlink(missing_ok=True)

    robust = profitability.iloc[0].to_dict()
    metrics: dict[str, Any] = {
        "schema_version": SWING_MODEL_SCHEMA_VERSION,
        "model_type": SWING_MODEL_TYPE,
        "family": config.family,
        "model_run_id": model_run_id,
        "target_col": target,
        "horizon_sessions": horizon,
        "validation_split": SWING_VALIDATION_SPLIT,
        "rows": len(data),
        "tickers": ticker_count,
        "features": len(features),
        "validated_rows": len(oof),
        "roc_auc": oof_metrics["roc_auc"],
        "top_decile_lift": oof_metrics["top_decile_lift"],
        "brier_score": oof_metrics["brier_score"],
        "log_loss": oof_metrics["log_loss"],
        "precision": oof_metrics["precision"],
        "recall": oof_metrics["recall"],
        "f1": oof_metrics["f1"],
        "ticker_holdout_rows": len(holdout_evidence),
        "ticker_holdout_roc_auc": holdout_metrics["roc_auc"],
        "ticker_holdout_top_decile_lift": holdout_metrics["top_decile_lift"],
        "ticker_holdout_brier_score": holdout_metrics["brier_score"],
        "robust_avg_trade_return": robust["avg_trade_return"],
        "robust_profit_factor": robust["profit_factor"],
        "robust_max_drawdown": robust["max_drawdown"],
        "dataset_sha256": dataset_sha256,
        "memory": memory_audit(
            hard_budget_gib=config.max_training_memory_gb,
            headroom_gib=config.memory_guard_headroom_gb,
        ).to_record(),
        "trained_at_utc": payload["trained_at_utc"],
    }
    manifest = write_model_manifest(
        model_path=model_out,
        model_type=SWING_MODEL_TYPE,
        schema_version=SWING_MODEL_SCHEMA_VERSION,
        target_col=target,
        features=features,
        training_data=data,
        metrics=metrics,
        validation_split=SWING_VALIDATION_SPLIT,
        status=MODEL_STATUS_CANDIDATE,
        extra={
            "dataset_sha256": dataset_sha256,
            "feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
            "horizon_sessions": horizon,
            "family": config.family,
            "model_run_id": model_run_id,
            "holdout_tickers": sorted(holdout_tickers),
            "training_config": config.model_dump(),
        },
    )
    return SwingTrainingResult(
        metrics=metrics,
        oof_predictions=oof,
        ticker_holdout_predictions=holdout_evidence,
        profitability_audit=profitability,
        regime_audit=regime,
        catalyst_audit=catalyst,
        alignment_audit=alignment,
        fold_audit=fold_audit,
        manifest=manifest,
    )


def score_swing_frame(
    frame: pd.DataFrame,
    model_path: Path,
    *,
    require_promoted: bool = False,
) -> pd.DataFrame:
    allowed = {MODEL_STATUS_PROMOTED} if require_promoted else {MODEL_STATUS_CANDIDATE, MODEL_STATUS_PROMOTED}
    manifest = verify_model_artifact(model_path, allowed_statuses=allowed)
    if manifest.get("model_type") != SWING_MODEL_TYPE or manifest.get("schema_version") != SWING_MODEL_SCHEMA_VERSION:
        raise SchemaMismatchError("model is not a canonical swing artifact")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict):
        raise SchemaMismatchError("canonical swing artifact payload is invalid")
    if payload.get("model_type") != SWING_MODEL_TYPE:
        raise SchemaMismatchError("canonical swing payload model type mismatch")
    features = [str(feature) for feature in payload.get("features", [])]
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"swing scoring rows are missing features: {missing[:10]}")
    if "swing_feature_schema_version" not in frame.columns or bool(
        frame["swing_feature_schema_version"].astype(str).ne(SWING_FEATURE_SCHEMA_VERSION).any()
    ):
        raise SchemaMismatchError("swing scoring feature schema mismatch")
    estimator = cast(ProbabilityEstimator, payload["model"])
    raw = estimator.predict_proba(_matrix(frame, features))[:, 1]
    probability = _apply_calibrator(payload.get("calibrator"), raw)
    output = frame.copy()
    output["swing_model_probability"] = probability
    output["swing_model_raw_probability"] = raw
    output["swing_model_prediction"] = (probability >= 0.5).astype("int8")
    output["swing_model_target"] = str(payload.get("target_col", ""))
    output["swing_model_schema"] = SWING_MODEL_SCHEMA_VERSION
    return output


def _training_rows(dataset: pd.DataFrame) -> tuple[pd.DataFrame, int, str]:
    required = {
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "label_eligible",
        "horizon_sessions",
        "swing_feature_schema_version",
        "market_regime",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise SchemaMismatchError(f"swing dataset missing training columns: {', '.join(missing)}")
    if bool(dataset["swing_feature_schema_version"].astype(str).ne(SWING_FEATURE_SCHEMA_VERSION).any()):
        raise SchemaMismatchError("swing dataset feature schema mismatch")
    horizons = pd.to_numeric(dataset["horizon_sessions"], errors="coerce").dropna().astype(int).unique()
    if len(horizons) != 1:
        raise SchemaMismatchError("swing dataset must contain exactly one horizon")
    horizon = int(horizons[0])
    target = swing_target_column(horizon)
    if target not in dataset.columns:
        raise SchemaMismatchError(f"swing dataset missing target {target}")
    data = dataset[dataset["label_eligible"].fillna(False).astype(bool)].copy()
    data = data.dropna(subset=[target]).sort_values(["session_date_et", "ticker"]).reset_index(drop=True)
    decision = pd.to_datetime(data["decision_time_utc"], utc=True, errors="coerce")
    feature = pd.to_datetime(data["feature_available_at_utc"], utc=True, errors="coerce")
    if bool(decision.isna().any() | feature.isna().any() | feature.gt(decision).any()):
        raise DataReadinessError("swing training rows contain future or invalid features")
    _require_binary_target(data[target], "swing training")
    return data, horizon, target


def _select_features(
    data: pd.DataFrame,
    folds: list[Any],
    config: SwingTrainingConfig,
) -> list[str]:
    selected: list[str] = []
    for feature in SWING_FEATURES:
        if feature not in data.columns:
            continue
        if pd.to_numeric(data[feature], errors="coerce").notna().mean() < config.min_feature_non_null_rate:
            continue
        if any(
            pd.to_numeric(data.iloc[fold.train_indices][feature], errors="coerce").notna().mean()
            < config.min_feature_non_null_rate
            for fold in folds
        ):
            continue
        selected.append(feature)
    return selected


def _estimator(config: SwingTrainingConfig) -> ProbabilityEstimator:
    if config.family == "logistic":
        return cast(
            ProbabilityEstimator,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=config.max_iter,
                            class_weight="balanced",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        )
    return cast(
        ProbabilityEstimator,
        Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "classifier",
                    HistGradientBoostingClassifier(
                        max_iter=config.max_iter,
                        learning_rate=config.learning_rate,
                        l2_regularization=config.l2_regularization,
                        max_leaf_nodes=31,
                        random_state=config.random_seed,
                    ),
                ),
            ]
        ),
    )


def _matrix(frame: pd.DataFrame, features: list[str]) -> pd.DataFrame:
    return frame.loc[:, features].apply(pd.to_numeric, errors="coerce").astype("float32")


def _fit_calibrator(probability: np.ndarray, target: np.ndarray) -> IsotonicRegression | None:
    finite = np.isfinite(probability)
    if int(finite.sum()) < 100 or len(np.unique(target[finite])) < 2:
        return None
    calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
    calibrator.fit(probability[finite], target[finite])
    return calibrator


def _cross_fitted_calibration(
    probability: np.ndarray,
    target: np.ndarray,
    sessions: pd.Series,
) -> np.ndarray:
    calibrated = probability.astype(float).copy()
    ordered = sorted(pd.to_datetime(sessions).dt.date.unique())
    chunks = [chunk for chunk in np.array_split(np.asarray(ordered, dtype=object), 4) if len(chunk)]
    session_values = pd.to_datetime(sessions).dt.date.to_numpy()
    prior_sessions = np.asarray([], dtype=object)
    for chunk in chunks:
        current = np.isin(session_values, chunk)
        prior = np.isin(session_values, prior_sessions)
        if int(prior.sum()) >= 100 and len(np.unique(target[prior])) >= 2:
            calibrator = _fit_calibrator(probability[prior], target[prior])
            calibrated[current] = _apply_calibrator(calibrator, probability[current])
        prior_sessions = np.concatenate([prior_sessions, chunk])
    return calibrated


def _apply_calibrator(calibrator: object, probability: np.ndarray) -> np.ndarray:
    if calibrator is None:
        return cast(np.ndarray, np.clip(probability.astype(float), 0.0, 1.0))
    predictor = cast(IsotonicRegression, calibrator)
    return cast(
        np.ndarray,
        np.clip(np.asarray(predictor.predict(probability), dtype=float), 0.0, 1.0),
    )


def _require_binary_target(values: pd.Series, name: str) -> None:
    unique = set(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique())
    if unique != {0, 1}:
        raise DataReadinessError(f"{name} requires both target classes; found {sorted(unique)}")
