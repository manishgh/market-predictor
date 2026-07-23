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
    capacity_curve,
    execution_policy_identity,
    merge_stress_summary,
)
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
    effective_sample_size,
    phase_economics,
    prediction_evidence,
    regime_audit,
)
from market_predictor.prediction_policy import (
    INTRADAY_SELECTION_TIE_BREAKERS,
    group_ranking_metrics,
    intraday_decision_scores,
    intraday_selection_eligible,
    prediction_policy_identity,
    select_top_k_per_group,
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
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=config.embargo_sessions,
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
        label_columns=[opportunity_target, downside_target],
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
        raise DataReadinessError(f"only {len(features)} intraday features pass fold coverage; need {config.min_features}")
    assert_memory_budget(
        hard_budget_gib=config.max_training_memory_gb,
        headroom_gib=config.memory_guard_headroom_gb,
        stage="intraday training input",
    )

    feature_set_sha256 = feature_schema_hash(features)
    targets = (opportunity_target, downside_target)
    calibration_raw: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    calibration_target: dict[str, list[np.ndarray]] = {target: [] for target in targets}
    calibration_availability: list[pd.Series] = []
    walk_forward_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
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
        ticker_validation = holdout[
            pd.to_datetime(holdout["session_date_et"]).dt.date.isin(test_sessions)
        ].reset_index(drop=True)
        if ticker_validation.empty:
            raise DataReadinessError(f"fold {fold.fold} has no held-out ticker test rows")

        validation_raw: dict[str, np.ndarray] = {}
        ticker_raw: dict[str, np.ndarray] = {}
        for target in targets:
            _require_binary_target(train[target], f"fold {fold.fold} {target} training")
            estimator = _estimator(config)
            _fit_estimator(estimator, train, features, target)
            validation_raw[target] = estimator.predict_proba(_matrix(validation, features))[:, 1]
            ticker_raw[target] = estimator.predict_proba(_matrix(ticker_validation, features))[:, 1]
            del estimator
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=config.max_training_memory_gb,
                headroom_gib=config.memory_guard_headroom_gb,
                stage=f"intraday fold {fold.fold} {target}",
            )

        calibration_fits: dict[str, CausalCalibrationFit] = {}
        if calibration_availability:
            prior_availability = pd.concat(calibration_availability, ignore_index=True)
            for target in targets:
                calibration_fit = fit_prior_isotonic(
                    np.concatenate(calibration_raw[target]),
                    np.concatenate(calibration_target[target]),
                    prior_availability,
                    before_utc=min_test_decision,
                )
                if calibration_fit is not None:
                    calibration_fits[target] = calibration_fit
        included = len(calibration_fits) == len(targets)
        if included:
            walk_forward_parts.append(
                _intraday_fold_evidence(
                    validation,
                    opportunity_raw=validation_raw[opportunity_target],
                    opportunity_probability=apply_isotonic(
                        calibration_fits[opportunity_target].calibrator,
                        validation_raw[opportunity_target],
                    ),
                    downside_raw=validation_raw[downside_target],
                    downside_probability=apply_isotonic(
                        calibration_fits[downside_target].calibrator,
                        validation_raw[downside_target],
                    ),
                    scope="walk_forward",
                    horizon=horizon,
                    fold=fold.fold,
                    calibration_fits=calibration_fits,
                )
            )
            holdout_parts.append(
                _intraday_fold_evidence(
                    ticker_validation,
                    opportunity_raw=ticker_raw[opportunity_target],
                    opportunity_probability=apply_isotonic(
                        calibration_fits[opportunity_target].calibrator,
                        ticker_raw[opportunity_target],
                    ),
                    downside_raw=ticker_raw[downside_target],
                    downside_probability=apply_isotonic(
                        calibration_fits[downside_target].calibrator,
                        ticker_raw[downside_target],
                    ),
                    scope="ticker_holdout",
                    horizon=horizon,
                    fold=fold.fold,
                    calibration_fits=calibration_fits,
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
                    calibration_fits=calibration_fits if included else {},
                )
            )
        for target in targets:
            calibration_raw[target].append(validation_raw[target])
            calibration_target[target].append(validation[target].astype(int).to_numpy())
        calibration_availability.append(validation["label_available_at_utc"])

    if not walk_forward_parts or not holdout_parts:
        raise DataReadinessError("no calibrated outer folds remain after the calibration seed")
    oof = pd.concat(walk_forward_parts, ignore_index=True)
    holdout_evidence = pd.concat(holdout_parts, ignore_index=True)
    if len(oof) < max(100, config.min_train_rows // 4):
        raise DataReadinessError("insufficient calibrated purged intraday predictions")
    calibrators = {
        target: fit_final_isotonic(
            np.concatenate(calibration_raw[target]),
            np.concatenate(calibration_target[target]),
        )
        for target in targets
    }
    if any(calibrator is None for calibrator in calibrators.values()):
        raise DataReadinessError("final intraday calibrators lack sufficient causal OOF evidence")
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
    opportunity_group_metrics = group_ranking_metrics(
        oof,
        target_column=opportunity_target,
        score=intraday_decision_scores(
            oof,
            opportunity_column="intraday_opportunity_probability",
            downside_column="intraday_downside_probability",
        ),
        group_column="decision_group_id",
        k=config.top_k,
        eligible=intraday_selection_eligible(
            oof,
            downside_column="intraday_downside_probability",
            downside_ceiling=config.max_downside_probability,
        ),
    )
    opportunity_holdout_group_metrics = group_ranking_metrics(
        holdout_evidence,
        target_column=opportunity_target,
        score=intraday_decision_scores(
            holdout_evidence,
            opportunity_column="intraday_opportunity_probability",
            downside_column="intraday_downside_probability",
        ),
        group_column="decision_group_id",
        k=config.top_k,
        eligible=intraday_selection_eligible(
            holdout_evidence,
            downside_column="intraday_downside_probability",
            downside_ceiling=config.max_downside_probability,
        ),
    )
    def _intraday_phase(frame: pd.DataFrame, scope: str, cost_stress: float = 1.0) -> pd.DataFrame:
        return phase_economics(
            frame,
            horizon_minutes=horizon,
            decision_interval_minutes=decision_interval,
            top_k=config.top_k,
            downside_ceiling=config.max_downside_probability,
            max_trades_per_session=config.max_trades_per_session,
            scope=scope,
            cost_stress=cost_stress,
        )

    stress_multiplier = max(DEFAULT_EXECUTION_POLICY.stress_multipliers)
    economics = pd.concat(
        [_intraday_phase(oof, "walk_forward"), _intraday_phase(holdout_evidence, "ticker_holdout")],
        ignore_index=True,
    )
    stress_economics = pd.concat(
        [
            _intraday_phase(oof, "walk_forward", stress_multiplier),
            _intraday_phase(holdout_evidence, "ticker_holdout", stress_multiplier),
        ],
        ignore_index=True,
    )
    conservative = merge_stress_summary(
        conservative_economics(economics),
        conservative_economics(stress_economics),
        multiplier=stress_multiplier,
        fields=STRESS_ECONOMIC_FIELDS,
    )
    profitability = pd.concat([conservative, economics], ignore_index=True)
    combined_evidence = pd.concat([oof, holdout_evidence], ignore_index=True)
    regime = regime_audit(
        combined_evidence,
        horizon_minutes=horizon,
        decision_interval_minutes=decision_interval,
        top_k=config.top_k,
        downside_ceiling=config.max_downside_probability,
        max_trades_per_session=config.max_trades_per_session,
        target_column=opportunity_target,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    catalyst = catalyst_audit(combined_evidence)
    alignment = _alignment_audit(dataset)
    for evidence in (oof, holdout_evidence, profitability, regime, catalyst, alignment):
        evidence["model_run_id"] = model_run_id
    representation = holdout_plan.representation_audit.copy()
    fold_frame = pd.DataFrame(fold_records)
    folds_causally_ordered = bool(
        len(fold_records) > 0
        and pd.to_datetime(fold_frame["max_train_label_available_at_utc"], utc=True)
        .lt(pd.to_datetime(fold_frame["min_test_decision_time_utc"], utc=True))
        .all()
    )
    fold_audit = pd.concat([fold_frame, representation], ignore_index=True, sort=False)
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
        "calibration_method": "isotonic_prior_outer_folds",
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

    capacity = capacity_curve(
        select_top_k_per_group(
            oof,
            score=intraday_decision_scores(
                oof,
                opportunity_column="intraday_opportunity_probability",
                downside_column="intraday_downside_probability",
            ),
            group_column="decision_group_id",
            top_k=config.top_k,
            tie_breakers=INTRADAY_SELECTION_TIE_BREAKERS,
            eligible=intraday_selection_eligible(
                oof,
                downside_column="intraday_downside_probability",
                downside_ceiling=config.max_downside_probability,
            ),
        ),
        gross_return_column=f"path_realized_return_gross_{horizon}m",
        dollar_volume_column="entry_dollar_volume",
        price_column="entry_price",
        atr_pct_column="entry_atr_pct",
        capital_weight=1.0 / config.top_k,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    robust = profitability.iloc[0].to_dict()
    capacity_min_avg_net_return = float(pd.to_numeric(capacity["avg_net_return"], errors="coerce").min())
    if not np.isfinite(capacity_min_avg_net_return):
        # No point-in-time liquidity evidence in this dataset: fall back to the
        # marginal (base-size) economics so the gate reflects real capacity only
        # when dollar-volume evidence is present.
        capacity_min_avg_net_return = float(robust.get("avg_trade_return", float("nan")))
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
        "decision_groups": int(oof["decision_group_id"].nunique()),
        "independent_sessions": int(oof["session_date_et"].nunique()),
        "validation_folds": config.n_splits,
        "capacity_min_avg_net_return": capacity_min_avg_net_return,
        "effective_sample_size": effective_sample_size(oof["overlap_weight"]),
        "holdout_effective_sample_size": effective_sample_size(holdout_evidence["overlap_weight"]),
        "calibration_method": "isotonic_prior_outer_folds",
        "calibration_seed_folds_excluded": calibration_seed_folds_excluded,
        "feature_set_sha256": feature_set_sha256,
        "reconciliation_sha256": stamped_hash(dataset, "reconciliation_sha256"),
        "folds_causally_ordered": folds_causally_ordered,
        **prediction_policy_identity(),
        **execution_policy_identity(),
        "holdout_assignment_cutoff_utc": holdout_plan.assignment_cutoff_utc,
        "holdout_ticker_summary_sha256": holdout_plan.ticker_summary_sha256,
        "holdout_required_strata": int(
            holdout_plan.representation_audit["required"].astype(bool).sum()
        ),
        "holdout_unrepresented_required_strata": int(
            (
                holdout_plan.representation_audit["required"].astype(bool)
                & ~holdout_plan.representation_audit["represented"].astype(bool)
            ).sum()
        ),
        "tickers": ticker_count,
        "features": len(features),
        "opportunity_roc_auc": opportunity_metrics["roc_auc"],
        "opportunity_top_decile_lift": opportunity_metrics["top_decile_lift"],
        "opportunity_group_lift_at_k": opportunity_group_metrics["group_lift_at_k"],
        "opportunity_group_precision_at_k": opportunity_group_metrics["group_precision_at_k"],
        "opportunity_group_ndcg_at_k": opportunity_group_metrics["group_ndcg_at_k"],
        "selection_k": opportunity_group_metrics["k"],
        "opportunity_brier_score": opportunity_metrics["brier_score"],
        "opportunity_calibration_error": opportunity_metrics["expected_calibration_error"],
        "opportunity_holdout_roc_auc": opportunity_holdout_metrics["roc_auc"],
        "opportunity_holdout_top_decile_lift": opportunity_holdout_metrics["top_decile_lift"],
        "opportunity_holdout_group_lift_at_k": opportunity_holdout_group_metrics["group_lift_at_k"],
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
        "feature_reference_profile": build_feature_reference_profile(data, features),
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
            "holdout_assignment_cutoff_utc": holdout_plan.assignment_cutoff_utc,
            "holdout_ticker_summary_sha256": holdout_plan.ticker_summary_sha256,
            "calibration_method": "isotonic_prior_outer_folds",
            "feature_set_sha256": feature_set_sha256,
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
    opportunity = apply_isotonic(calibrators.get(opportunity_target), opportunity_raw)
    downside = apply_isotonic(calibrators.get(downside_target), downside_raw)
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
    training_data: pd.DataFrame,
    config: IntradayTrainingConfig,
) -> list[str]:
    selected: list[str] = []
    for feature in INTRADAY_MODEL_FEATURES:
        if feature not in training_data.columns:
            continue
        values = pd.to_numeric(training_data[feature], errors="coerce")
        if values.notna().mean() < config.min_feature_non_null_rate:
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


def _intraday_fold_evidence(
    frame: pd.DataFrame,
    *,
    opportunity_raw: np.ndarray,
    opportunity_probability: np.ndarray,
    downside_raw: np.ndarray,
    downside_probability: np.ndarray,
    scope: str,
    horizon: int,
    fold: int,
    calibration_fits: dict[str, CausalCalibrationFit],
) -> pd.DataFrame:
    evidence = prediction_evidence(
        frame,
        opportunity_raw=opportunity_raw,
        opportunity_probability=opportunity_probability,
        downside_raw=downside_raw,
        downside_probability=downside_probability,
        scope=scope,
        horizon_minutes=horizon,
    )
    cutoffs = [fit.train_cutoff_utc for fit in calibration_fits.values()]
    evidence["validation_fold"] = fold
    evidence["calibration_method"] = "isotonic_prior_outer_folds"
    evidence["calibration_train_cutoff_utc"] = max(cutoffs).isoformat()
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
    calibration_fits: dict[str, CausalCalibrationFit],
) -> dict[str, object]:
    cutoffs = [fit.train_cutoff_utc for fit in calibration_fits.values()]
    training_rows = min((fit.training_rows for fit in calibration_fits.values()), default=0)
    return {
        **fold.audit_record(),
        "record_type": "validation_fold",
        "validation_scope": scope,
        "validation_status": (
            "included" if calibration_fits else "calibration_seed_excluded"
        ),
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
        "calibration_method": (
            "isotonic_prior_outer_folds" if calibration_fits else "seed_only_not_scored"
        ),
        "calibration_train_cutoff_utc": max(cutoffs).isoformat() if cutoffs else "",
        "calibration_training_rows": training_rows,
    }


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
    events_without_feature_row = stamped_scalar(data, "reconciliation_events_without_feature_row")
    dates_with_news_count_mismatch = stamped_scalar(data, "reconciliation_dates_with_news_count_mismatch")
    return pd.DataFrame(
        [
            {
                "alignment_error_total": (
                    future + path_mismatch + benchmark_mismatch + events_without_feature_row + dates_with_news_count_mismatch
                ),
                "future_feature_rows": future,
                "label_path_mismatches": path_mismatch,
                "benchmark_path_mismatches": benchmark_mismatch,
                "events_without_feature_row": events_without_feature_row,
                "missing_historical_feature_rows": 0,
                "dates_with_news_count_mismatch": dates_with_news_count_mismatch,
            }
        ]
    )


def _require_binary_target(values: pd.Series, name: str) -> None:
    unique = set(pd.to_numeric(values, errors="coerce").dropna().astype(int).unique())
    if unique != {0, 1}:
        raise DataReadinessError(f"{name} requires both target classes; found {sorted(unique)}")
