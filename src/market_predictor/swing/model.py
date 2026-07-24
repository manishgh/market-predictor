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
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.canonical.reconciliation import stamped_hash, stamped_scalar
from market_predictor.drift import build_feature_reference_profile
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    STRESS_ECONOMIC_FIELDS,
    execution_policy_identity,
    merge_stress_summary,
)
from market_predictor.label_policy import stamped_label_policy
from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    calibration_summary,
    group_ranking_metrics,
    prediction_policy_identity,
    swing_decision_scores,
)
from market_predictor.registry import (
    MODEL_STATUS_CANDIDATE,
    MODEL_STATUS_PROMOTED,
    feature_schema_hash,
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
from market_predictor.v3.calibration import (
    CausalCalibrationFit,
    apply_isotonic,
    fit_final_isotonic,
    fit_prior_isotonic,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from market_predictor.v3.validation import (
    V3Fold,
    V3PurgedWalkForwardSplit,
    causal_fold_training_indices,
    deterministic_stratified_ticker_holdout,
    identity_set_sha256,
    validation_row_identities,
)


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
    prediction_policy = PredictionSelectionPolicy(swing_top_k=config.top_k)
    if not overwrite and (model_out.exists() or manifest_path_for(model_out).exists()):
        raise FileExistsError(f"swing model artifact already exists: {model_out}")
    model_run_id = f"swing-{uuid.uuid4().hex}"
    data, horizon, target = _training_rows(dataset)
    label_policy = stamped_label_policy(dataset)
    if len(data) < config.min_train_rows:
        raise DataReadinessError(f"swing training needs at least {config.min_train_rows} eligible rows")
    ticker_count = int(data["ticker"].nunique())
    if ticker_count < config.min_training_tickers:
        raise DataReadinessError(f"swing training needs at least {config.min_training_tickers} tickers; found {ticker_count}")
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=horizon,
        min_train_sessions=config.min_train_sessions,
        min_train_rows=config.min_train_rows,
    )
    data["validation_row_identity"] = validation_row_identities(data)
    assignment_folds = splitter.split(data)
    assignment_indices, _, _ = causal_fold_training_indices(
        data,
        candidate_indices=assignment_folds[0].train_indices,
        test_indices=assignment_folds[0].test_indices,
    )
    holdout_plan = deterministic_stratified_ticker_holdout(
        data.iloc[assignment_indices],
        label_columns=[target],
        fraction=config.ticker_holdout_fraction,
        seed=config.random_seed,
    )
    holdout_tickers = holdout_plan.holdout_tickers
    development = data[~data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    holdout = data[data["ticker"].isin(holdout_tickers)].reset_index(drop=True)
    folds = splitter.split(development)
    first_train_indices, _, _ = causal_fold_training_indices(
        development,
        candidate_indices=folds[0].train_indices,
        test_indices=folds[0].test_indices,
    )
    features = _select_features(development.iloc[first_train_indices], config)
    if len(features) < config.min_features:
        raise DataReadinessError(f"only {len(features)} swing features pass fold coverage; need {config.min_features}")
    assert_memory_budget(
        hard_budget_gib=config.max_training_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage="swing training input",
    )

    feature_set_sha256 = feature_schema_hash(features)
    walk_forward_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    calibration_raw: list[np.ndarray] = []
    calibration_target: list[np.ndarray] = []
    calibration_availability: list[pd.Series] = []
    fold_records: list[dict[str, object]] = []
    calibration_seed_folds_excluded = 0
    for fold in folds:
        train_indices, max_train_label, min_test_decision = causal_fold_training_indices(
            development,
            candidate_indices=fold.train_indices,
            test_indices=fold.test_indices,
        )
        train = development.iloc[train_indices]
        validation = development.iloc[fold.test_indices].reset_index(drop=True)
        test_sessions = set(pd.to_datetime(validation["session_date_et"]).dt.date)
        ticker_validation = holdout[pd.to_datetime(holdout["session_date_et"]).dt.date.isin(test_sessions)].reset_index(drop=True)
        if ticker_validation.empty:
            raise DataReadinessError(f"fold {fold.fold} has no held-out ticker test rows")
        _require_binary_target(train[target], f"fold {fold.fold} training")
        estimator = _estimator(config)
        estimator.fit(_matrix(train, features), train[target].astype(int))
        validation_raw = estimator.predict_proba(_matrix(validation, features))[:, 1]
        ticker_raw = estimator.predict_proba(_matrix(ticker_validation, features))[:, 1]

        calibration_fit: CausalCalibrationFit | None = None
        if calibration_raw:
            calibration_fit = fit_prior_isotonic(
                np.concatenate(calibration_raw),
                np.concatenate(calibration_target),
                pd.concat(calibration_availability, ignore_index=True),
                before_utc=min_test_decision,
            )
        if calibration_fit is not None:
            walk_forward_parts.append(
                _swing_fold_evidence(
                    validation,
                    raw_probability=validation_raw,
                    probability=apply_isotonic(calibration_fit.calibrator, validation_raw),
                    scope="walk_forward",
                    horizon=horizon,
                    fold=fold.fold,
                    calibration_fit=calibration_fit,
                )
            )
            holdout_parts.append(
                _swing_fold_evidence(
                    ticker_validation,
                    raw_probability=ticker_raw,
                    probability=apply_isotonic(calibration_fit.calibrator, ticker_raw),
                    scope="ticker_holdout",
                    horizon=horizon,
                    fold=fold.fold,
                    calibration_fit=calibration_fit,
                )
            )
        else:
            calibration_seed_folds_excluded += 1
        for scope, test_frame in (
            ("walk_forward", validation),
            ("ticker_holdout", ticker_validation),
        ):
            fold_records.append(
                _fold_evidence_record(
                    fold,
                    scope=scope,
                    train=train,
                    test=test_frame,
                    max_train_label=max_train_label,
                    min_test_decision=min_test_decision,
                    feature_set_sha256=feature_set_sha256,
                    calibration_fit=calibration_fit,
                )
            )
        calibration_raw.append(validation_raw)
        calibration_target.append(validation[target].astype(int).to_numpy())
        calibration_availability.append(validation["label_available_at_utc"])
        del estimator
        release_process_memory()
        assert_memory_budget(
            hard_budget_gib=config.max_training_memory_gb,
            headroom_gib=config.memory_guard_headroom_gb,
            stage=f"swing fold {fold.fold}",
        )

    if not walk_forward_parts or not holdout_parts:
        raise DataReadinessError("no calibrated outer folds remain after the calibration seed")
    oof = pd.concat(walk_forward_parts, ignore_index=True)
    holdout_evidence = pd.concat(holdout_parts, ignore_index=True)
    oof["ticker_cohort"] = "seen"
    holdout_evidence["ticker_cohort"] = "unseen"
    full_cross_section_evidence = pd.concat(
        [oof, holdout_evidence],
        ignore_index=True,
    )
    if len(oof) < max(100, config.min_train_rows // 4):
        raise DataReadinessError("insufficient calibrated purged walk-forward predictions")
    calibrator = fit_final_isotonic(
        np.concatenate(calibration_raw),
        np.concatenate(calibration_target),
    )
    if calibrator is None:
        raise DataReadinessError("final swing calibrator lacks sufficient causal OOF evidence")
    oof_metrics = classification_metrics(oof[target], oof["swing_probability"])
    holdout_metrics = classification_metrics(
        holdout_evidence[target],
        holdout_evidence["swing_probability"],
    )
    oof_group_metrics = group_ranking_metrics(
        oof,
        target_column=target,
        score=swing_decision_scores(oof, probability_column="swing_probability"),
        group_column="decision_group_id",
        k=config.top_k,
    )
    holdout_group_metrics = group_ranking_metrics(
        holdout_evidence,
        target_column=target,
        score=swing_decision_scores(holdout_evidence, probability_column="swing_probability"),
        group_column="decision_group_id",
        k=config.top_k,
    )
    oof_calibration = calibration_summary(oof[target], oof["swing_probability"])
    holdout_calibration = calibration_summary(holdout_evidence[target], holdout_evidence["swing_probability"])
    stress_multiplier = max(DEFAULT_EXECUTION_POLICY.stress_multipliers)
    economics = phase_economics(
        full_cross_section_evidence,
        horizon=horizon,
        top_k=config.top_k,
        scope="full_cross_section",
        cohort_column="ticker_cohort",
    )
    full_economics = economics[economics["scope"].eq("full_cross_section")]
    stress_economics = phase_economics(
        full_cross_section_evidence,
        horizon=horizon,
        top_k=config.top_k,
        scope="full_cross_section",
        cost_stress=stress_multiplier,
    )
    conservative = merge_stress_summary(
        conservative_economics(full_economics),
        conservative_economics(stress_economics),
        multiplier=stress_multiplier,
        fields=STRESS_ECONOMIC_FIELDS,
    )
    profitability = pd.concat([conservative, economics], ignore_index=True)
    regime = regime_audit(
        full_cross_section_evidence,
        horizon=horizon,
        top_k=config.top_k,
        target_column=target,
        min_regime_sessions=config.min_regime_sessions,
        min_regime_trades=config.min_regime_trades,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    catalyst = catalyst_audit(full_cross_section_evidence)
    alignment = _alignment_audit(dataset)
    for evidence in (oof, holdout_evidence, profitability, regime, catalyst, alignment):
        evidence["model_run_id"] = model_run_id
    representation = holdout_plan.representation_audit.copy()
    fold_frame = pd.DataFrame(fold_records)
    scored_validation_fold_ids = sorted(
        int(fold)
        for fold in fold_frame.loc[
            fold_frame["validation_status"].eq("included"),
            "fold",
        ].unique()
    )
    folds_causally_ordered = bool(
        len(fold_records) > 0
        and pd.to_datetime(fold_frame["max_train_label_available_at_utc"], utc=True)
        .lt(pd.to_datetime(fold_frame["min_test_decision_time_utc"], utc=True))
        .all()
    )
    fold_audit = pd.concat([fold_frame, representation], ignore_index=True, sort=False)
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
        "calibration_method": "isotonic_prior_outer_folds",
        "prediction_policy": prediction_policy.specification(),
        "prediction_policy_sha256": prediction_policy.sha256(),
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
        "decision_groups": int(oof["decision_group_id"].nunique()),
        "independent_sessions": int(oof["session_date_et"].nunique()),
        "validation_folds": len(scored_validation_fold_ids),
        "configured_validation_folds": config.n_splits,
        "scored_validation_fold_ids": scored_validation_fold_ids,
        "roc_auc": oof_metrics["roc_auc"],
        "top_decile_lift": oof_metrics["top_decile_lift"],
        "group_lift_at_k": oof_group_metrics["group_lift_at_k"],
        "group_precision_at_k": oof_group_metrics["group_precision_at_k"],
        "group_ndcg_at_k": oof_group_metrics["group_ndcg_at_k"],
        "selection_k": oof_group_metrics["k"],
        "brier_score": oof_metrics["brier_score"],
        "log_loss": oof_metrics["log_loss"],
        "expected_calibration_error": oof_calibration["expected_calibration_error"],
        "calibration_bias": oof_calibration["calibration_bias"],
        "calibration_slope": oof_calibration["calibration_slope"],
        "calibration_intercept": oof_calibration["calibration_intercept"],
        "precision": oof_metrics["precision"],
        "recall": oof_metrics["recall"],
        "f1": oof_metrics["f1"],
        "ticker_holdout_rows": len(holdout_evidence),
        "full_cross_section_rows": len(full_cross_section_evidence),
        "full_cross_section_selected_trades": int(
            pd.to_numeric(
                full_economics["selected_trades"],
                errors="coerce",
            ).sum()
        ),
        "selected_seen_trades": int(
            pd.to_numeric(
                economics.loc[
                    economics["scope"].eq("full_cross_section:seen"),
                    "selected_trades",
                ],
                errors="coerce",
            ).sum()
        ),
        "selected_unseen_trades": int(
            pd.to_numeric(
                economics.loc[
                    economics["scope"].eq("full_cross_section:unseen"),
                    "selected_trades",
                ],
                errors="coerce",
            ).sum()
        ),
        "ticker_holdout_roc_auc": holdout_metrics["roc_auc"],
        "ticker_holdout_top_decile_lift": holdout_metrics["top_decile_lift"],
        "ticker_holdout_group_lift_at_k": holdout_group_metrics["group_lift_at_k"],
        "ticker_holdout_brier_score": holdout_metrics["brier_score"],
        "ticker_holdout_calibration_error": holdout_calibration["expected_calibration_error"],
        "calibration_method": "isotonic_prior_outer_folds",
        "calibration_seed_folds_excluded": calibration_seed_folds_excluded,
        "feature_set_sha256": feature_set_sha256,
        "reconciliation_sha256": stamped_hash(dataset, "reconciliation_sha256"),
        "event_assignment_sha256": stamped_hash(
            dataset,
            "event_assignment_sha256",
        ),
        "event_aggregate_sha256": stamped_hash(
            dataset,
            "event_aggregate_sha256",
        ),
        "label_material_sha256": stamped_hash(
            dataset,
            "label_material_sha256",
        ),
        "label_source_reconciliation_sha256": stamped_hash(
            dataset,
            "label_source_reconciliation_sha256",
        ),
        "dataset_label_config_sha256": stamped_hash(dataset, "dataset_label_config_sha256"),
        "universe_identity_sha256": identity_set_sha256(data["universe_snapshot_id"].astype(str).unique()),
        "universe_snapshots": int(data["universe_snapshot_id"].nunique()),
        "folds_causally_ordered": folds_causally_ordered,
        **prediction_policy_identity(prediction_policy),
        **execution_policy_identity(),
        "holdout_assignment_cutoff_utc": holdout_plan.assignment_cutoff_utc,
        "holdout_ticker_summary_sha256": holdout_plan.ticker_summary_sha256,
        "holdout_required_strata": int(holdout_plan.representation_audit["required"].astype(bool).sum()),
        "holdout_unrepresented_required_strata": int(
            (
                holdout_plan.representation_audit["required"].astype(bool) & ~holdout_plan.representation_audit["represented"].astype(bool)
            ).sum()
        ),
        "robust_avg_trade_return": robust["avg_trade_return"],
        "robust_profit_factor": robust["profit_factor"],
        "robust_max_drawdown": robust["max_drawdown"],
        "dataset_sha256": dataset_sha256,
        "feature_reference_profile": build_feature_reference_profile(data, features),
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
        extra={
            "dataset_sha256": dataset_sha256,
            "feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
            "horizon_sessions": horizon,
            "family": config.family,
            "model_run_id": model_run_id,
            "holdout_tickers": sorted(holdout_tickers),
            "holdout_assignment_cutoff_utc": holdout_plan.assignment_cutoff_utc,
            "holdout_ticker_summary_sha256": holdout_plan.ticker_summary_sha256,
            "calibration_method": "isotonic_prior_outer_folds",
            "feature_set_sha256": feature_set_sha256,
            "training_config": config.model_dump(),
            "label_policy": label_policy,
            "prediction_policy": prediction_policy.specification(),
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


def _alignment_audit(dataset: pd.DataFrame) -> pd.DataFrame:
    """Real feature/label alignment evidence (no fabricated zeros).

    Feature-row-level checks: features that postdate their decision (leakage),
    eligible rows whose expected label path is not exact, and exact-label rows
    missing a benchmark excess return. Event-to-feature reconciliation counts
    (events_without_feature_row / missing_historical_feature_rows /
    dates_with_news_count_mismatch) are populated by the R3 reconciliation
    artifact; they remain zero here until that artifact is bound in.
    """

    decision = pd.to_datetime(dataset["decision_time_utc"], utc=True, errors="coerce")
    feature = pd.to_datetime(dataset["feature_available_at_utc"], utc=True, errors="coerce")
    label_expected = dataset["label_window_expected"].fillna(False).astype(bool)
    label_exact = dataset["label_path_exact"].fillna(False).astype(bool)
    feature_eligible = dataset["feature_eligible"].fillna(False).astype(bool)
    benchmark_columns = [
        column for column in dataset if column.startswith("future_excess_return_") and column.endswith(("_vs_spy", "_vs_qqq", "_vs_sector"))
    ]
    benchmark_missing = dataset[benchmark_columns].isna().any(axis=1) if benchmark_columns else pd.Series(False, index=dataset.index)
    future = int((feature > decision).fillna(True).sum())
    path_mismatch = int((feature_eligible & label_expected & ~label_exact).sum())
    benchmark_mismatch = int((feature_eligible & label_exact & benchmark_missing).sum())
    events_without_feature_row = stamped_scalar(dataset, "reconciliation_events_without_feature_row")
    missing_historical_feature_rows = stamped_scalar(
        dataset,
        "reconciliation_missing_historical_feature_rows",
    )
    dates_with_news_count_mismatch = stamped_scalar(dataset, "reconciliation_dates_with_news_count_mismatch")
    label_source_reconciliation_errors = stamped_scalar(
        dataset,
        "label_source_reconciliation_errors",
        default=1,
    )
    return pd.DataFrame(
        [
            {
                "alignment_error_total": (
                    future
                    + path_mismatch
                    + benchmark_mismatch
                    + events_without_feature_row
                    + missing_historical_feature_rows
                    + dates_with_news_count_mismatch
                    + label_source_reconciliation_errors
                ),
                "future_feature_rows": future,
                "label_path_mismatches": path_mismatch,
                "benchmark_path_mismatches": benchmark_mismatch,
                "events_without_feature_row": events_without_feature_row,
                "missing_historical_feature_rows": missing_historical_feature_rows,
                "dates_with_news_count_mismatch": dates_with_news_count_mismatch,
                "label_source_reconciliation_errors": (label_source_reconciliation_errors),
            }
        ]
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
    return score_swing_payload(frame, payload)


def score_swing_payload(
    frame: pd.DataFrame,
    payload: object,
) -> pd.DataFrame:
    """Score with a previously verified and deserialized swing payload."""

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
    probability = apply_isotonic(payload.get("calibrator"), raw)
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
        "label_available_at_utc",
        "label_eligible",
        "horizon_sessions",
        "swing_feature_schema_version",
        "dataset_label_config_sha256",
        "label_material_sha256",
        "label_source_reconciliation_sha256",
        "label_source_reconciliation_errors",
        "universe_snapshot_id",
        "market_regime",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise SchemaMismatchError(f"swing dataset missing training columns: {', '.join(missing)}")
    if bool(dataset["swing_feature_schema_version"].astype(str).ne(SWING_FEATURE_SCHEMA_VERSION).any()):
        raise SchemaMismatchError("swing dataset feature schema mismatch")
    label_configs = dataset["dataset_label_config_sha256"].astype(str).unique()
    if len(label_configs) != 1 or not str(label_configs[0]).strip():
        raise SchemaMismatchError("swing dataset must contain exactly one label config")
    for column in (
        "label_material_sha256",
        "label_source_reconciliation_sha256",
    ):
        values = dataset[column].fillna("").astype(str).unique()
        if len(values) != 1 or len(values[0]) != 64:
            raise DataReadinessError(f"swing dataset has invalid {column} identity")
    label_reconciliation_errors = pd.to_numeric(
        dataset["label_source_reconciliation_errors"],
        errors="coerce",
    )
    if (
        label_reconciliation_errors.isna().any()
        or label_reconciliation_errors.nunique(dropna=False) != 1
        or int(label_reconciliation_errors.iloc[0]) != 0
    ):
        raise DataReadinessError("swing dataset label source reconciliation did not pass")
    horizons = pd.to_numeric(dataset["horizon_sessions"], errors="coerce").dropna().astype(int).unique()
    if len(horizons) != 1:
        raise SchemaMismatchError("swing dataset must contain exactly one horizon")
    horizon = int(horizons[0])
    target = swing_target_column(horizon)
    if target not in dataset.columns:
        raise SchemaMismatchError(f"swing dataset missing target {target}")
    data = dataset[dataset["label_eligible"].fillna(False).astype(bool)].copy()
    data = data.dropna(subset=[target]).sort_values(["session_date_et", "ticker"]).reset_index(drop=True)
    universe = data["universe_snapshot_id"].astype(str).str.strip()
    if bool(data["universe_snapshot_id"].isna().any()) or bool(universe.eq("").any()):
        raise DataReadinessError("swing training rows are missing point-in-time universe identity")
    decision = pd.to_datetime(data["decision_time_utc"], utc=True, errors="coerce")
    feature = pd.to_datetime(data["feature_available_at_utc"], utc=True, errors="coerce")
    label = pd.to_datetime(data["label_available_at_utc"], utc=True, errors="coerce")
    invalid = decision.isna() | feature.isna() | label.isna() | feature.gt(decision) | label.le(decision)
    if bool(invalid.any()):
        raise DataReadinessError("swing training rows contain future or invalid timestamps")
    _require_binary_target(data[target], "swing training")
    return data, horizon, target


def _select_features(
    training_data: pd.DataFrame,
    config: SwingTrainingConfig,
) -> list[str]:
    selected: list[str] = []
    for feature in SWING_FEATURES:
        if feature not in training_data.columns:
            continue
        if pd.to_numeric(training_data[feature], errors="coerce").notna().mean() < config.min_feature_non_null_rate:
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


def _swing_fold_evidence(
    frame: pd.DataFrame,
    *,
    raw_probability: np.ndarray,
    probability: np.ndarray,
    scope: str,
    horizon: int,
    fold: int,
    calibration_fit: CausalCalibrationFit,
) -> pd.DataFrame:
    evidence = prediction_evidence(
        frame,
        raw_probability=raw_probability,
        probability=probability,
        scope=scope,
        horizon=horizon,
    )
    evidence["validation_fold"] = fold
    evidence["calibration_method"] = calibration_fit.method
    evidence["calibration_train_cutoff_utc"] = calibration_fit.train_cutoff_utc.isoformat()
    evidence["row_identity"] = frame["validation_row_identity"].astype(str).to_numpy()
    return evidence


def _fold_evidence_record(
    fold: V3Fold,
    *,
    scope: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_train_label: pd.Timestamp,
    min_test_decision: pd.Timestamp,
    feature_set_sha256: str,
    calibration_fit: CausalCalibrationFit | None,
) -> dict[str, object]:
    return {
        **fold.audit_record(),
        "record_type": "validation_fold",
        "validation_scope": scope,
        "validation_status": ("included" if calibration_fit is not None else "calibration_seed_excluded"),
        "train_rows": len(train),
        "test_rows": len(test),
        "max_train_label_available_at_utc": max_train_label.isoformat(),
        "min_test_decision_time_utc": min_test_decision.isoformat(),
        "train_ticker_count": int(train["ticker"].nunique()),
        "test_ticker_count": int(test["ticker"].nunique()),
        "train_ticker_set_sha256": identity_set_sha256(train["ticker"].unique()),
        "test_ticker_set_sha256": identity_set_sha256(test["ticker"].unique()),
        "train_row_identity_sha256": identity_set_sha256(train["validation_row_identity"]),
        "test_row_identity_sha256": identity_set_sha256(test["validation_row_identity"]),
        "feature_set_sha256": feature_set_sha256,
        "calibration_method": calibration_fit.method if calibration_fit else "seed_only_not_scored",
        "calibration_train_cutoff_utc": (calibration_fit.train_cutoff_utc.isoformat() if calibration_fit else ""),
        "calibration_training_rows": calibration_fit.training_rows if calibration_fit else 0,
    }


def _require_binary_target(values: pd.Series, name: str) -> None:
    unique = set(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique())
    if unique != {0, 1}:
        raise DataReadinessError(f"{name} requires both target classes; found {sorted(unique)}")
