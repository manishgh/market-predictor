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

from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_FEATURES,
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
    INTRADAY_VALIDATION_SPLIT,
    IntradayTrainingConfig,
    downside_target_column,
    excess_return_column,
    opportunity_target_column,
)
from market_predictor.intraday.evaluation import (
    catalyst_audit,
    classification_metrics,
    conservative_economics,
    phase_economics,
    prediction_evidence,
    regime_audit,
)
from market_predictor.registry import (
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_PROMOTED,
    manifest_path_for,
    verify_model_artifact,
    write_model_manifest,
)
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.validation import V3PurgedWalkForwardSplit, deterministic_ticker_holdout


class ProbabilityEstimator(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True)
class IntradayTrainingResult:
    metrics: dict[str, Any]
    oof_predictions: pd.DataFrame
    ticker_holdout_predictions: pd.DataFrame
    profitability_audit: pd.DataFrame
    regime_audit: pd.DataFrame
    catalyst_audit: pd.DataFrame
    alignment_audit: pd.DataFrame
    fold_audit: pd.DataFrame
    manifest: dict[str, Any]


def train_intraday_model(
    dataset: pd.DataFrame,
    *,
    model_out: Path,
    dataset_sha256: str,
    config: IntradayTrainingConfig | None = None,
    overwrite: bool = False,
) -> IntradayTrainingResult:
    config = config or IntradayTrainingConfig()
    if not overwrite and (model_out.exists() or manifest_path_for(model_out).exists()):
        raise FileExistsError(f"intraday model artifact already exists: {model_out}")
    model_run_id = f"intraday-{uuid.uuid4().hex}"
    data, horizon, decision_interval, opportunity_target, downside_target = _training_rows(dataset)
    if len(data) < config.min_train_rows:
        raise DataReadinessError(f"intraday training needs at least {config.min_train_rows} eligible rows")
    ticker_count = int(data["ticker"].nunique())
    if ticker_count < config.min_training_tickers:
        raise DataReadinessError(f"intraday training needs at least {config.min_training_tickers} tickers; found {ticker_count}")
    holdout_tickers = deterministic_ticker_holdout(
        data["ticker"],
        fraction=config.ticker_holdout_fraction,
        seed=config.random_seed,
    )
    development = data[~data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    holdout = data[data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=config.embargo_sessions,
        min_train_sessions=config.min_train_sessions,
        min_train_rows=config.min_train_rows,
    )
    folds = splitter.split(development)
    features = _select_features(development, folds, config)
    if len(features) < config.min_features:
        raise DataReadinessError(f"only {len(features)} intraday features pass fold coverage; need {config.min_features}")
    assert_memory_budget(
        hard_budget_gib=config.max_training_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage="intraday training input",
    )

    raw_predictions = {
        opportunity_target: np.full(len(development), np.nan, dtype=np.float64),
        downside_target: np.full(len(development), np.nan, dtype=np.float64),
    }
    fold_records: list[dict[str, object]] = []
    for fold in folds:
        train = development.iloc[fold.train_indices]
        validation = development.iloc[fold.test_indices]
        for target in (opportunity_target, downside_target):
            _require_binary_target(train[target], f"fold {fold.fold} {target} training")
            estimator = _estimator(config)
            _fit_estimator(estimator, train, features, target)
            raw_predictions[target][fold.test_indices] = estimator.predict_proba(_matrix(validation, features))[:, 1]
            del estimator
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=config.max_training_memory_gb,
                headroom_gib=config.memory_guard_headroom_gb,
                stage=f"intraday fold {fold.fold} {target}",
            )
        fold_records.append(fold.audit_record())

    oof_mask = np.isfinite(raw_predictions[opportunity_target]) & np.isfinite(raw_predictions[downside_target])
    if int(oof_mask.sum()) < max(100, config.min_train_rows // 4):
        raise DataReadinessError("insufficient purged intraday walk-forward predictions")
    calibrators: dict[str, IsotonicRegression | None] = {}
    cross_fitted: dict[str, np.ndarray] = {}
    for target in (opportunity_target, downside_target):
        raw = raw_predictions[target][oof_mask]
        target_values = development.loc[oof_mask, target].astype(int).to_numpy()
        cross_fitted[target] = _cross_fitted_calibration(
            raw,
            target_values,
            development.loc[oof_mask, "session_date_et"],
        )
        calibrators[target] = _fit_calibrator(raw, target_values)

    holdout_raw: dict[str, np.ndarray] = {}
    holdout_probability: dict[str, np.ndarray] = {}
    for target in (opportunity_target, downside_target):
        _require_binary_target(development[target], f"ticker holdout {target} training")
        estimator = _estimator(config)
        _fit_estimator(estimator, development, features, target)
        holdout_raw[target] = estimator.predict_proba(_matrix(holdout, features))[:, 1]
        holdout_probability[target] = _apply_calibrator(
            calibrators[target],
            holdout_raw[target],
        )
        del estimator
        release_process_memory()

    oof = prediction_evidence(
        development.loc[oof_mask].reset_index(drop=True),
        opportunity_raw=raw_predictions[opportunity_target][oof_mask],
        opportunity_probability=cross_fitted[opportunity_target],
        downside_raw=raw_predictions[downside_target][oof_mask],
        downside_probability=cross_fitted[downside_target],
        scope="walk_forward",
        horizon_minutes=horizon,
    )
    holdout_evidence = prediction_evidence(
        holdout,
        opportunity_raw=holdout_raw[opportunity_target],
        opportunity_probability=holdout_probability[opportunity_target],
        downside_raw=holdout_raw[downside_target],
        downside_probability=holdout_probability[downside_target],
        scope="ticker_holdout",
        horizon_minutes=horizon,
    )
    opportunity_metrics = classification_metrics(
        oof[opportunity_target],
        oof["intraday_opportunity_probability"],
    )
    opportunity_holdout_metrics = classification_metrics(
        holdout_evidence[opportunity_target],
        holdout_evidence["intraday_opportunity_probability"],
    )
    downside_metrics = classification_metrics(
        oof[downside_target],
        oof["intraday_downside_probability"],
    )
    downside_holdout_metrics = classification_metrics(
        holdout_evidence[downside_target],
        holdout_evidence["intraday_downside_probability"],
    )
    economics = pd.concat(
        [
            phase_economics(
                oof,
                horizon_minutes=horizon,
                decision_interval_minutes=decision_interval,
                top_k=config.top_k,
                downside_ceiling=config.max_downside_probability,
                max_trades_per_session=config.max_trades_per_session,
                scope="walk_forward",
            ),
            phase_economics(
                holdout_evidence,
                horizon_minutes=horizon,
                decision_interval_minutes=decision_interval,
                top_k=config.top_k,
                downside_ceiling=config.max_downside_probability,
                max_trades_per_session=config.max_trades_per_session,
                scope="ticker_holdout",
            ),
        ],
        ignore_index=True,
    )
    profitability = pd.concat([conservative_economics(economics), economics], ignore_index=True)
    combined_evidence = pd.concat([oof, holdout_evidence], ignore_index=True)
    regime = regime_audit(combined_evidence)
    catalyst = catalyst_audit(combined_evidence)
    alignment = _alignment_audit(dataset)
    for evidence in (oof, holdout_evidence, profitability, regime, catalyst, alignment):
        evidence["model_run_id"] = model_run_id
    fold_audit = pd.DataFrame(fold_records)
    fold_audit["model_run_id"] = model_run_id

    final_models: dict[str, ProbabilityEstimator] = {}
    for target in (opportunity_target, downside_target):
        estimator = _estimator(config)
        _fit_estimator(estimator, data, features, target)
        final_models[target] = estimator
        assert_memory_budget(
            hard_budget_gib=config.max_training_memory_gb,
            headroom_gib=config.memory_guard_headroom_gb,
            stage=f"final intraday {target} model",
        )
    model_out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "models": final_models,
        "calibrators": calibrators,
        "features": features,
        "opportunity_target_col": opportunity_target,
        "downside_target_col": downside_target,
        "horizon_minutes": horizon,
        "decision_interval_minutes": decision_interval,
        "model_type": INTRADAY_MODEL_TYPE,
        "model_schema_version": INTRADAY_MODEL_SCHEMA_VERSION,
        "feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
        "family": config.family,
        "model_run_id": model_run_id,
        "decision_semantics": "completed_5m_decision_next_available_1m_open_entry",
        "catalyst_policy": "external_confirmation_overlay",
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
        "schema_version": INTRADAY_MODEL_SCHEMA_VERSION,
        "model_type": INTRADAY_MODEL_TYPE,
        "family": config.family,
        "model_run_id": model_run_id,
        "target_col": opportunity_target,
        "downside_target_col": downside_target,
        "horizon_minutes": horizon,
        "decision_interval_minutes": decision_interval,
        "validation_split": INTRADAY_VALIDATION_SPLIT,
        "validated_rows": len(oof),
        "ticker_holdout_rows": len(holdout_evidence),
        "tickers": ticker_count,
        "features": len(features),
        "opportunity_roc_auc": opportunity_metrics["roc_auc"],
        "opportunity_top_decile_lift": opportunity_metrics["top_decile_lift"],
        "opportunity_brier_score": opportunity_metrics["brier_score"],
        "opportunity_calibration_error": opportunity_metrics["expected_calibration_error"],
        "opportunity_holdout_roc_auc": opportunity_holdout_metrics["roc_auc"],
        "opportunity_holdout_top_decile_lift": opportunity_holdout_metrics["top_decile_lift"],
        "opportunity_holdout_brier_score": opportunity_holdout_metrics["brier_score"],
        "opportunity_holdout_calibration_error": opportunity_holdout_metrics["expected_calibration_error"],
        "downside_roc_auc": downside_metrics["roc_auc"],
        "downside_brier_score": downside_metrics["brier_score"],
        "downside_calibration_error": downside_metrics["expected_calibration_error"],
        "downside_holdout_roc_auc": downside_holdout_metrics["roc_auc"],
        "downside_holdout_brier_score": downside_holdout_metrics["brier_score"],
        "downside_holdout_calibration_error": downside_holdout_metrics["expected_calibration_error"],
        "selected_trades": robust["selected_trades"],
        "avg_trade_return": robust["avg_trade_return"],
        "avg_excess_return_vs_spy": robust["avg_excess_return_vs_spy"],
        "avg_excess_return_vs_qqq": robust["avg_excess_return_vs_qqq"],
        "avg_excess_return_vs_sector": robust["avg_excess_return_vs_sector"],
        "profit_factor": robust["profit_factor"],
        "max_drawdown": robust["max_drawdown"],
        "return_drawdown_ratio": robust["return_drawdown_ratio"],
        "negative_session_rate": robust["negative_session_rate"],
        "average_turnover": robust["average_turnover"],
        "dataset_sha256": dataset_sha256,
        "memory": memory_audit(
            hard_budget_gib=config.max_training_memory_gb,
            headroom_gib=config.memory_guard_headroom_gb,
        ).to_record(),
        "opportunity_metrics": opportunity_metrics,
        "opportunity_holdout_metrics": opportunity_holdout_metrics,
        "downside_metrics": downside_metrics,
        "downside_holdout_metrics": downside_holdout_metrics,
    }
    manifest = write_model_manifest(
        model_path=model_out,
        model_type=INTRADAY_MODEL_TYPE,
        schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
        target_col=opportunity_target,
        features=features,
        training_data=data,
        metrics=metrics,
        validation_split=INTRADAY_VALIDATION_SPLIT,
        status=MODEL_STATUS_CANDIDATE,
        extra={
            "model_run_id": model_run_id,
            "dataset_sha256": dataset_sha256,
            "downside_target_col": downside_target,
            "feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
            "ticker_holdout": sorted(holdout_tickers),
            "catalyst_policy": "external_confirmation_overlay",
            "memory": metrics["memory"],
        },
    )
    return IntradayTrainingResult(
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


def score_intraday_frame(
    frame: pd.DataFrame,
    model_path: Path,
    *,
    require_promoted: bool = False,
) -> pd.DataFrame:
    allowed = (
        {MODEL_STATUS_PROMOTED}
        if require_promoted
        else {
            MODEL_STATUS_CANDIDATE,
            MODEL_STATUS_PROMOTED,
        }
    )
    manifest = verify_model_artifact(model_path, allowed_statuses=allowed)
    if manifest.get("model_type") != INTRADAY_MODEL_TYPE or manifest.get("schema_version") != INTRADAY_MODEL_SCHEMA_VERSION:
        raise SchemaMismatchError("model is not a canonical intraday artifact")
    payload = joblib.load(model_path)
    if not isinstance(payload, dict) or payload.get("model_type") != INTRADAY_MODEL_TYPE:
        raise SchemaMismatchError("canonical intraday artifact payload is invalid")
    features = [str(feature) for feature in payload.get("features", [])]
    missing = sorted(set(features).difference(frame.columns))
    if missing:
        raise SchemaMismatchError(f"intraday scoring rows are missing features: {missing[:10]}")
    if "intraday_feature_schema_version" not in frame.columns or bool(
        frame["intraday_feature_schema_version"].astype(str).ne(INTRADAY_FEATURE_SCHEMA_VERSION).any()
    ):
        raise SchemaMismatchError("intraday scoring feature schema mismatch")
    models = payload.get("models")
    calibrators = payload.get("calibrators")
    if not isinstance(models, dict) or not isinstance(calibrators, dict):
        raise SchemaMismatchError("intraday artifact is missing models or calibrators")
    opportunity_target = str(payload.get("opportunity_target_col", ""))
    downside_target = str(payload.get("downside_target_col", ""))
    matrix = _matrix(frame, features)
    opportunity_model = cast(ProbabilityEstimator, models[opportunity_target])
    downside_model = cast(ProbabilityEstimator, models[downside_target])
    opportunity_raw = opportunity_model.predict_proba(matrix)[:, 1]
    downside_raw = downside_model.predict_proba(matrix)[:, 1]
    opportunity = _apply_calibrator(calibrators.get(opportunity_target), opportunity_raw)
    downside = _apply_calibrator(calibrators.get(downside_target), downside_raw)
    output = frame.copy()
    output["intraday_opportunity_probability"] = opportunity
    output["intraday_downside_probability"] = downside
    output["intraday_opportunity_raw_probability"] = opportunity_raw
    output["intraday_downside_raw_probability"] = downside_raw
    output["intraday_opportunity_prediction"] = (opportunity >= 0.5).astype("int8")
    output["intraday_downside_prediction"] = (downside >= 0.5).astype("int8")
    output["intraday_opportunity_target"] = opportunity_target
    output["intraday_downside_target"] = downside_target
    output["intraday_model_schema"] = INTRADAY_MODEL_SCHEMA_VERSION
    return output


def _training_rows(
    dataset: pd.DataFrame,
) -> tuple[pd.DataFrame, int, int, str, str]:
    required = {
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "entry_time_utc",
        "label_available_at_utc",
        "feature_eligible",
        "label_eligible",
        "label_window_expected",
        "label_path_exact",
        "horizon_minutes",
        "decision_bar_minutes",
        "decision_stride_bars",
        "overlap_weight",
        "intraday_feature_schema_version",
        "market_regime",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise SchemaMismatchError(f"intraday dataset missing training columns: {', '.join(missing)}")
    if bool(dataset["intraday_feature_schema_version"].astype(str).ne(INTRADAY_FEATURE_SCHEMA_VERSION).any()):
        raise SchemaMismatchError("intraday dataset feature schema mismatch")
    horizons = pd.to_numeric(dataset["horizon_minutes"], errors="coerce").dropna().astype(int).unique()
    intervals = (
        (pd.to_numeric(dataset["decision_bar_minutes"], errors="coerce") * pd.to_numeric(dataset["decision_stride_bars"], errors="coerce"))
        .dropna()
        .astype(int)
        .unique()
    )
    if len(horizons) != 1 or len(intervals) != 1:
        raise SchemaMismatchError("intraday dataset must contain one horizon and decision interval")
    horizon = int(horizons[0])
    decision_interval = int(intervals[0])
    opportunity_target = opportunity_target_column(horizon)
    downside_target = downside_target_column(horizon)
    evidence_columns = [
        opportunity_target,
        downside_target,
        excess_return_column(horizon, "spy"),
        excess_return_column(horizon, "qqq"),
        excess_return_column(horizon, "sector"),
    ]
    missing_evidence = [column for column in evidence_columns if column not in dataset.columns]
    if missing_evidence:
        raise SchemaMismatchError(f"intraday dataset missing target/evidence columns: {', '.join(missing_evidence)}")
    data = dataset[dataset["label_eligible"].fillna(False).astype(bool)].copy()
    data = (
        data.dropna(subset=[opportunity_target, downside_target])
        .sort_values(
            ["session_date_et", "decision_time_utc", "ticker"],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    decision = pd.to_datetime(data["decision_time_utc"], utc=True, errors="coerce")
    feature = pd.to_datetime(data["feature_available_at_utc"], utc=True, errors="coerce")
    entry = pd.to_datetime(data["entry_time_utc"], utc=True, errors="coerce")
    label = pd.to_datetime(data["label_available_at_utc"], utc=True, errors="coerce")
    invalid = decision.isna() | feature.isna() | entry.isna() | label.isna() | feature.gt(decision) | entry.lt(decision) | label.le(entry)
    if bool(invalid.any()):
        raise DataReadinessError("intraday training rows contain future or invalid timestamps")
    _require_binary_target(data[opportunity_target], "intraday opportunity training")
    _require_binary_target(data[downside_target], "intraday downside training")
    return data, horizon, decision_interval, opportunity_target, downside_target


def _select_features(
    data: pd.DataFrame,
    folds: list[Any],
    config: IntradayTrainingConfig,
) -> list[str]:
    selected: list[str] = []
    for feature in INTRADAY_MODEL_FEATURES:
        if feature not in data.columns:
            continue
        values = pd.to_numeric(data[feature], errors="coerce")
        if values.notna().mean() < config.min_feature_non_null_rate:
            continue
        if any(values.iloc[fold.train_indices].notna().mean() < config.min_feature_non_null_rate for fold in folds):
            continue
        selected.append(feature)
    return selected


def _estimator(config: IntradayTrainingConfig) -> ProbabilityEstimator:
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


def _fit_estimator(
    estimator: ProbabilityEstimator,
    frame: pd.DataFrame,
    features: list[str],
    target: str,
) -> None:
    weights = pd.to_numeric(frame["overlap_weight"], errors="coerce").fillna(1.0).clip(lower=1e-6)
    estimator.fit(
        _matrix(frame, features),
        frame[target].astype(int),
        classifier__sample_weight=weights.to_numpy(float),
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
        return np.asarray(np.clip(probability.astype(float), 0.0, 1.0), dtype=float)
    predictor = cast(IsotonicRegression, calibrator)
    return np.asarray(
        np.clip(np.asarray(predictor.predict(probability), dtype=float), 0.0, 1.0),
        dtype=float,
    )


def _alignment_audit(data: pd.DataFrame) -> pd.DataFrame:
    decision = pd.to_datetime(data["decision_time_utc"], utc=True, errors="coerce")
    feature = pd.to_datetime(data["feature_available_at_utc"], utc=True, errors="coerce")
    label_expected = data["label_window_expected"].fillna(False).astype(bool)
    label_exact = data["label_path_exact"].fillna(False).astype(bool)
    feature_eligible = data["feature_eligible"].fillna(False).astype(bool)
    benchmark_columns = [
        column for column in data if column.startswith("path_excess_return_") and column.endswith(("_vs_spy", "_vs_qqq", "_vs_sector"))
    ]
    benchmark_missing = data[benchmark_columns].isna().any(axis=1)
    future = int((feature > decision).fillna(True).sum())
    path_mismatch = int((feature_eligible & label_expected & ~label_exact).sum())
    benchmark_mismatch = int((feature_eligible & label_exact & benchmark_missing).sum())
    return pd.DataFrame(
        [
            {
                "alignment_error_total": future + path_mismatch + benchmark_mismatch,
                "future_feature_rows": future,
                "label_path_mismatches": path_mismatch,
                "benchmark_path_mismatches": benchmark_mismatch,
                "events_without_feature_row": 0,
                "missing_historical_feature_rows": 0,
                "dates_with_news_count_mismatch": 0,
            }
        ]
    )


def _require_binary_target(values: pd.Series, name: str) -> None:
    unique = set(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique())
    if unique != {0, 1}:
        raise DataReadinessError(f"{name} requires both target classes; found {sorted(unique)}")
