"""Causal walk-forward evaluation for KS3 swing-specialist candidates."""

from __future__ import annotations

import importlib
import math
import uuid
from dataclasses import dataclass
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    calibration_summary,
    group_ranking_metrics,
    select_swing_candidates,
    swing_decision_scores,
)
from market_predictor.registry import feature_schema_hash
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.evaluation import (
    classification_metrics,
    conservative_economics,
    phase_economics,
    prediction_evidence,
)
from market_predictor.swing.specialist_contracts import (
    EstimatorFamily,
    FeatureProfile,
    SwingSpecialistResearchConfig,
)
from market_predictor.v3.calibration import (
    CausalCalibrationFit,
    apply_isotonic,
    fit_final_isotonic,
    fit_prior_isotonic,
)
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.validation import (
    V3Fold,
    V3PurgedWalkForwardSplit,
    causal_fold_training_indices,
    deterministic_stratified_ticker_holdout,
    identity_set_sha256,
)

SPECIALIST_VALIDATION_SPLIT = (
    "session_purged_walk_forward_and_ticker_holdout"
)
SPECIALIST_ACCEPTED_STATUS = "accepted_development"
SPECIALIST_REJECTED_STATUS = "rejected"


class ProbabilityEstimator(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SpecialistExperimentSpec:
    strategy_id: str
    estimator_family: EstimatorFamily
    feature_profile: FeatureProfile | None
    deterministic_score: str

    @property
    def candidate_id(self) -> str:
        profile = self.feature_profile or "deterministic_score"
        return f"{self.estimator_family}__{profile}"


@dataclass(frozen=True)
class SpecialistSplitPlan:
    strategy_id: str
    horizon_sessions: int
    data: pd.DataFrame
    development: pd.DataFrame
    ticker_holdout: pd.DataFrame
    folds: tuple[V3Fold, ...]
    holdout_tickers: frozenset[str]
    representation_audit: pd.DataFrame
    profile_features: dict[str, tuple[str, ...]]
    split_sha256: str


@dataclass(frozen=True)
class SpecialistExperimentResult:
    spec: SpecialistExperimentSpec
    status: str
    rejection_reasons: tuple[str, ...]
    metrics: dict[str, object]
    predictions: pd.DataFrame
    economics: pd.DataFrame
    regime_evidence: pd.DataFrame
    capacity_evidence: pd.DataFrame
    fold_audit: pd.DataFrame
    final_estimator: object | None
    final_calibrator: object | None


def specialist_experiment_specs(
    strategy_id: str,
    config: SwingSpecialistResearchConfig,
) -> tuple[SpecialistExperimentSpec, ...]:
    strategy = config.strategies[strategy_id]
    specs = [
        SpecialistExperimentSpec(
            strategy_id=strategy_id,
            estimator_family="deterministic_baseline",
            feature_profile=None,
            deterministic_score=strategy.deterministic_score,
        )
    ]
    specs.extend(
        SpecialistExperimentSpec(
            strategy_id=strategy_id,
            estimator_family=family,
            feature_profile=profile,
            deterministic_score=strategy.deterministic_score,
        )
        for family in strategy.estimator_families
        if family != "deterministic_baseline"
        for profile in strategy.feature_profiles
    )
    if len(specs) != strategy.experiment_count():
        raise AssertionError("specialist experiment catalog is inconsistent")
    return tuple(specs)


def build_specialist_split_plan(
    dataset: pd.DataFrame,
    *,
    strategy_id: str,
    config: SwingSpecialistResearchConfig,
) -> SpecialistSplitPlan:
    data, horizon = _training_rows(dataset, strategy_id=strategy_id)
    if len(data) < config.min_train_rows:
        raise DataReadinessError(
            f"{strategy_id} needs at least {config.min_train_rows} rows"
        )
    if int(data["ticker"].nunique()) < config.min_training_tickers:
        raise DataReadinessError(
            f"{strategy_id} needs at least "
            f"{config.min_training_tickers} tickers"
        )
    splitter = V3PurgedWalkForwardSplit(
        n_splits=config.n_splits,
        embargo_sessions=horizon,
        min_train_sessions=config.min_train_sessions,
        min_train_rows=config.min_train_rows,
    )
    assignment_folds = splitter.split(data)
    assignment_indices, _, _ = causal_fold_training_indices(
        data,
        candidate_indices=assignment_folds[0].train_indices,
        test_indices=assignment_folds[0].test_indices,
    )
    holdout_plan = deterministic_stratified_ticker_holdout(
        data.iloc[assignment_indices],
        label_columns=["strategy_target"],
        fraction=config.ticker_holdout_fraction,
        seed=config.random_seed,
    )
    holdout_tickers = frozenset(holdout_plan.holdout_tickers)
    development = data.loc[
        ~data["ticker"].isin(holdout_tickers)
    ].reset_index(drop=True)
    ticker_holdout = data.loc[
        data["ticker"].isin(holdout_tickers)
    ].reset_index(drop=True)
    folds = tuple(splitter.split(development))
    first_train_indices, _, _ = causal_fold_training_indices(
        development,
        candidate_indices=folds[0].train_indices,
        test_indices=folds[0].test_indices,
    )
    first_train = development.iloc[first_train_indices]
    profile_features: dict[str, tuple[str, ...]] = {
        profile: _select_profile_features(
            first_train,
            configured_features,
            minimum_non_null_rate=config.minimum_feature_non_null_rate,
        )
        for profile, configured_features in config.feature_profiles.items()
    }
    required_profiles = set(
        config.strategies[strategy_id].feature_profiles
    )
    empty_profiles = sorted(
        profile
        for profile in required_profiles
        if not profile_features[profile]
    )
    if empty_profiles:
        raise DataReadinessError(
            f"{strategy_id} has no eligible features for profiles: "
            + ", ".join(empty_profiles)
        )
    split_sha256 = _split_identity(
        development,
        ticker_holdout,
        folds,
        holdout_tickers,
    )
    return SpecialistSplitPlan(
        strategy_id=strategy_id,
        horizon_sessions=horizon,
        data=data,
        development=development,
        ticker_holdout=ticker_holdout,
        folds=folds,
        holdout_tickers=holdout_tickers,
        representation_audit=holdout_plan.representation_audit.copy(),
        profile_features=profile_features,
        split_sha256=split_sha256,
    )


def evaluate_specialist_experiment(
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    *,
    config: SwingSpecialistResearchConfig,
) -> SpecialistExperimentResult:
    if spec.strategy_id != plan.strategy_id:
        raise DataReadinessError("candidate strategy does not match split plan")
    features = (
        ()
        if spec.feature_profile is None
        else plan.profile_features[spec.feature_profile]
    )
    feature_set_sha256 = feature_schema_hash(list(features))
    walk_forward_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
    calibration_raw: list[np.ndarray] = []
    calibration_target: list[np.ndarray] = []
    calibration_availability: list[pd.Series] = []
    fold_records: list[dict[str, object]] = []
    calibration_seed_folds_excluded = 0

    for fold in plan.folds:
        train_indices, max_train_label, min_test_decision = (
            causal_fold_training_indices(
                plan.development,
                candidate_indices=fold.train_indices,
                test_indices=fold.test_indices,
            )
        )
        train = plan.development.iloc[train_indices]
        validation = plan.development.iloc[
            fold.test_indices
        ].reset_index(drop=True)
        test_sessions = set(
            pd.to_datetime(validation["session_date_et"]).dt.date
        )
        ticker_validation = plan.ticker_holdout.loc[
            pd.to_datetime(plan.ticker_holdout["session_date_et"])
            .dt.date.isin(test_sessions)
        ].reset_index(drop=True)
        if ticker_validation.empty:
            raise DataReadinessError(
                f"{spec.candidate_id} fold {fold.fold} has no unseen rows"
            )
        _require_binary_target(
            train["strategy_target"],
            f"{spec.candidate_id} fold {fold.fold}",
        )
        estimator = _new_estimator(spec, config)
        _fit_candidate(
            estimator,
            spec,
            train,
            features,
            config=config,
        )
        validation_raw = _raw_scores(
            estimator,
            spec,
            validation,
            features,
        )
        ticker_raw = _raw_scores(
            estimator,
            spec,
            ticker_validation,
            features,
        )
        calibration_fit: CausalCalibrationFit | None = None
        if calibration_raw:
            calibration_fit = fit_prior_isotonic(
                np.concatenate(calibration_raw),
                np.concatenate(calibration_target),
                pd.concat(calibration_availability, ignore_index=True),
                before_utc=min_test_decision,
            )
        if calibration_fit is None:
            calibration_seed_folds_excluded += 1
        else:
            walk_forward_parts.append(
                _fold_predictions(
                    validation,
                    raw_score=validation_raw,
                    probability=apply_isotonic(
                        calibration_fit.calibrator,
                        validation_raw,
                    ),
                    spec=spec,
                    fold=fold.fold,
                    scope="walk_forward",
                    cohort="seen",
                    calibration_fit=calibration_fit,
                    horizon=plan.horizon_sessions,
                )
            )
            holdout_parts.append(
                _fold_predictions(
                    ticker_validation,
                    raw_score=ticker_raw,
                    probability=apply_isotonic(
                        calibration_fit.calibrator,
                        ticker_raw,
                    ),
                    spec=spec,
                    fold=fold.fold,
                    scope="ticker_holdout",
                    cohort="unseen",
                    calibration_fit=calibration_fit,
                    horizon=plan.horizon_sessions,
                )
            )
        for scope, test_frame in (
            ("walk_forward", validation),
            ("ticker_holdout", ticker_validation),
        ):
            fold_records.append(
                _fold_record(
                    fold,
                    scope=scope,
                    train=train,
                    test=test_frame,
                    max_train_label=max_train_label,
                    min_test_decision=min_test_decision,
                    feature_set_sha256=feature_set_sha256,
                    calibration_fit=calibration_fit,
                    split_sha256=plan.split_sha256,
                )
            )
        calibration_raw.append(validation_raw)
        calibration_target.append(
            validation["strategy_target"].astype(int).to_numpy()
        )
        calibration_availability.append(
            validation["label_available_at_utc"]
        )
        del estimator
        release_process_memory()
        _assert_memory(config, f"{spec.candidate_id} fold {fold.fold}")

    if not walk_forward_parts or not holdout_parts:
        raise DataReadinessError(
            f"{spec.candidate_id} has no calibrated validation folds"
        )
    walk_forward = pd.concat(walk_forward_parts, ignore_index=True)
    ticker_holdout = pd.concat(holdout_parts, ignore_index=True)
    predictions = pd.concat(
        [walk_forward, ticker_holdout],
        ignore_index=True,
    ).sort_values(
        ["decision_time_utc", "ticker", "ticker_cohort"],
        kind="stable",
    ).reset_index(drop=True)
    if bool(predictions["row_identity"].duplicated().any()):
        raise DataReadinessError(
            f"{spec.candidate_id} emitted duplicate validation rows"
        )
    final_calibrator = fit_final_isotonic(
        np.concatenate(calibration_raw),
        np.concatenate(calibration_target),
    )
    if final_calibrator is None:
        raise DataReadinessError(
            f"{spec.candidate_id} lacks final calibration evidence"
        )

    metrics = _candidate_metrics(
        walk_forward,
        ticker_holdout,
        plan=plan,
        spec=spec,
        feature_set_sha256=feature_set_sha256,
        calibration_seed_folds_excluded=(
            calibration_seed_folds_excluded
        ),
        config=config,
    )
    economics = phase_economics(
        predictions,
        horizon=plan.horizon_sessions,
        top_k=config.top_k,
        scope="full_cross_section",
        cohort_column="ticker_cohort",
        use_stamped_net_returns=True,
    )
    rejection_reasons = _economic_rejection_reasons(
        economics,
        config=config,
    )
    status = (
        SPECIALIST_ACCEPTED_STATUS
        if not rejection_reasons
        else SPECIALIST_REJECTED_STATUS
    )
    regime = _regime_evidence(
        predictions,
        horizon=plan.horizon_sessions,
        top_k=config.top_k,
    )
    capacity = _capacity_evidence(
        predictions,
        top_k=config.top_k,
        participation_rate=config.capacity_participation_rate,
    )
    fold_audit = pd.concat(
        [
            pd.DataFrame(fold_records),
            plan.representation_audit.assign(
                record_type="ticker_representation",
                split_sha256=plan.split_sha256,
            ),
        ],
        ignore_index=True,
        sort=False,
    )
    final_estimator: object | None = None
    if status == SPECIALIST_ACCEPTED_STATUS:
        final_estimator = _new_estimator(spec, config)
        _fit_candidate(
            final_estimator,
            spec,
            plan.data,
            features,
            config=config,
        )
        _assert_memory(config, f"{spec.candidate_id} final fit")
    metrics.update(
        {
            "status": status,
            "rejection_reasons": list(rejection_reasons),
            "memory": memory_audit(
                hard_budget_gib=config.maximum_process_memory_gib,
                headroom_gib=config.memory_guard_headroom_gib,
            ).to_record(),
        }
    )
    return SpecialistExperimentResult(
        spec=spec,
        status=status,
        rejection_reasons=rejection_reasons,
        metrics=metrics,
        predictions=predictions,
        economics=economics,
        regime_evidence=regime,
        capacity_evidence=capacity,
        fold_audit=fold_audit,
        final_estimator=final_estimator,
        final_calibrator=final_calibrator,
    )


def _training_rows(
    dataset: pd.DataFrame,
    *,
    strategy_id: str,
) -> tuple[pd.DataFrame, int]:
    required = {
        "strategy_id",
        "strategy_target",
        "strategy_label_eligible",
        "strategy_horizon_sessions",
        "strategy_dataset_row_id",
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "universe_snapshot_id",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise DataReadinessError(
            "specialist training dataset is missing: " + ", ".join(missing)
        )
    observed = set(dataset["strategy_id"].astype(str))
    if observed != {strategy_id}:
        raise DataReadinessError(
            f"specialist dataset strategy mismatch: {sorted(observed)}"
        )
    eligible = dataset["strategy_label_eligible"].fillna(False).astype(bool)
    data = dataset.loc[eligible].copy()
    horizons = (
        pd.to_numeric(
            data["strategy_horizon_sessions"],
            errors="coerce",
        )
        .dropna()
        .astype(int)
        .unique()
    )
    if len(horizons) != 1:
        raise DataReadinessError(
            "specialist dataset must contain one strategy horizon"
        )
    decision = pd.to_datetime(
        data["decision_time_utc"],
        utc=True,
        errors="coerce",
    )
    feature = pd.to_datetime(
        data["feature_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    label = pd.to_datetime(
        data["label_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    invalid = (
        decision.isna()
        | feature.isna()
        | label.isna()
        | feature.gt(decision)
        | label.le(decision)
    )
    if bool(invalid.any()):
        raise DataReadinessError(
            "specialist dataset contains invalid temporal evidence"
        )
    if bool(data["strategy_dataset_row_id"].astype(str).duplicated().any()):
        raise DataReadinessError(
            "specialist dataset contains duplicate row identities"
        )
    data["validation_row_identity"] = data[
        "strategy_dataset_row_id"
    ].astype(str)
    data["strategy_target"] = pd.to_numeric(
        data["strategy_target"],
        errors="raise",
    ).astype("int8")
    _require_binary_target(data["strategy_target"], strategy_id)
    return (
        data.sort_values(
            ["session_date_et", "ticker"],
            kind="stable",
        ).reset_index(drop=True),
        int(horizons[0]),
    )


def _select_profile_features(
    training: pd.DataFrame,
    configured: tuple[str, ...],
    *,
    minimum_non_null_rate: float,
) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in configured
        if feature in training
        and pd.to_numeric(
            training[feature],
            errors="coerce",
        ).notna().mean()
        >= minimum_non_null_rate
    )


def _new_estimator(
    spec: SpecialistExperimentSpec,
    config: SwingSpecialistResearchConfig,
) -> object | None:
    if spec.estimator_family == "deterministic_baseline":
        return None
    if spec.estimator_family == "logistic":
        return cast(
            ProbabilityEstimator,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scale", StandardScaler()),
                    (
                        "classifier",
                        LogisticRegression(
                            max_iter=config.logistic_max_iter,
                            class_weight="balanced",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        )
    if spec.estimator_family == "hist_gradient_boosting":
        return cast(
            ProbabilityEstimator,
            Pipeline(
                [
                    ("imputer", SimpleImputer(strategy="median")),
                    (
                        "classifier",
                        HistGradientBoostingClassifier(
                            max_iter=config.hgb_max_iter,
                            learning_rate=config.hgb_learning_rate,
                            l2_regularization=(
                                config.hgb_l2_regularization
                            ),
                            class_weight="balanced",
                            random_state=config.random_seed,
                        ),
                    ),
                ]
            ),
        )
    try:
        xgboost = importlib.import_module("xgboost")
    except ImportError as exc:
        raise DataReadinessError(
            "direct ranker requires the ranking dependency"
        ) from exc
    return cast(
        object,
        xgboost.XGBRanker(
        objective="rank:ndcg",
        eval_metric=f"ndcg@{config.top_k}",
        n_estimators=config.ranker_max_iter,
        learning_rate=config.ranker_learning_rate,
        max_depth=config.ranker_max_depth,
        max_bin=config.ranker_max_bin,
        tree_method="hist",
        random_state=config.random_seed,
        n_jobs=config.ranker_n_jobs,
        callbacks=[_xgboost_memory_guard(xgboost, config)],
        ),
    )


def _fit_candidate(
    estimator: object | None,
    spec: SpecialistExperimentSpec,
    data: pd.DataFrame,
    features: tuple[str, ...],
    *,
    config: SwingSpecialistResearchConfig,
) -> None:
    if spec.estimator_family == "deterministic_baseline":
        return
    if estimator is None:
        raise AssertionError("learned candidate has no estimator")
    matrix = _matrix(data, features)
    target = data["strategy_target"].astype(int)
    if spec.estimator_family == "direct_ranker":
        order = np.lexsort(
            (
                data["ticker"].astype(str).to_numpy(),
                data["decision_group_id"].astype(str).to_numpy(),
            )
        )
        ordered = data.iloc[order]
        query_id = pd.factorize(
            ordered["decision_group_id"],
            sort=False,
        )[0].astype(np.int32, copy=False)
        cast(Any, estimator).fit(
            matrix.iloc[order].to_numpy(dtype=np.float32, copy=False),
            target.iloc[order].to_numpy(dtype=np.int16, copy=False),
            qid=query_id,
        )
        cast(Any, estimator).set_params(callbacks=None)
        return
    cast(ProbabilityEstimator, estimator).fit(matrix, target)


def _raw_scores(
    estimator: object | None,
    spec: SpecialistExperimentSpec,
    data: pd.DataFrame,
    features: tuple[str, ...],
) -> np.ndarray:
    if spec.estimator_family == "deterministic_baseline":
        return _deterministic_score(data, spec.deterministic_score)
    if estimator is None:
        raise AssertionError("learned candidate has no estimator")
    matrix = _matrix(data, features)
    if spec.estimator_family == "direct_ranker":
        scores: np.ndarray = np.asarray(
            cast(Any, estimator).predict(
                matrix.to_numpy(dtype=np.float32, copy=False)
            ),
            dtype=float,
        )
        return scores
    return np.asarray(
        cast(ProbabilityEstimator, estimator).predict_proba(matrix)[:, 1],
        dtype=float,
    )


def _deterministic_score(
    data: pd.DataFrame,
    score_name: str,
) -> np.ndarray:
    def number(column: str) -> pd.Series:
        if column not in data:
            raise DataReadinessError(
                f"deterministic score is missing {column}"
            )
        return pd.to_numeric(data[column], errors="coerce")

    if score_name == "xs_rank_rel_return_20d_vs_sector":
        score = number(score_name)
    elif score_name == "trend_strength":
        score = (
            0.35 * number("return_20d")
            + 0.25 * number("dist_sma_50")
            + 0.25 * number("dist_sma_200")
            + 0.15 * number("sma_200_slope_20d")
        )
    elif score_name == "catalyst_confirmation":
        score = (
            np.log1p(number("event_count_3d").clip(lower=0))
            * number("event_relevance_mean_3d")
            * number("sentiment_coverage_3d")
            + number("sentiment_mean_3d")
        )
    elif score_name == "reversal_extremity":
        score = -(
            number("return_5d")
            / number("atr_pct_14").clip(lower=1e-6)
        ) + (35.0 - number("rsi_14")) / 35.0
    elif score_name == "breakout_confirmation":
        score = np.log1p(
            number("volume_ratio_20").clip(lower=0)
        ) + number("close_location")
    else:
        raise DataReadinessError(
            f"unsupported deterministic score: {score_name}"
        )
    if bool(score.isna().any()) or not np.isfinite(score.to_numpy()).all():
        raise DataReadinessError(
            f"deterministic score {score_name} is not finite"
        )
    values: np.ndarray = score.to_numpy(dtype=float)
    return values


def _matrix(
    data: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    return data.loc[:, list(features)].apply(
        pd.to_numeric,
        errors="coerce",
    ).astype("float32")


def _fold_predictions(
    frame: pd.DataFrame,
    *,
    raw_score: np.ndarray,
    probability: np.ndarray,
    spec: SpecialistExperimentSpec,
    fold: int,
    scope: str,
    cohort: str,
    calibration_fit: CausalCalibrationFit,
    horizon: int,
) -> pd.DataFrame:
    evidence = prediction_evidence(
        frame,
        raw_probability=raw_score,
        probability=probability,
        scope=scope,
        horizon=horizon,
    )
    extra_columns = (
        "strategy_dataset_row_id",
        "strategy_target",
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "strategy_gross_return",
        "strategy_execution_cost_fraction",
        "strategy_net_return",
        "strategy_spy_return",
        "strategy_qqq_return",
        "strategy_sector_return",
        "strategy_excess_return_vs_spy",
        "strategy_excess_return_vs_qqq",
        "strategy_excess_return_vs_sector",
        "strategy_mfe",
        "strategy_mae",
        "dollar_volume",
    )
    for column in extra_columns:
        if column in frame and column not in evidence:
            evidence[column] = frame[column].to_numpy()
    evidence["row_identity"] = frame[
        "validation_row_identity"
    ].astype(str).to_numpy()
    evidence["strategy_id"] = spec.strategy_id
    evidence["candidate_id"] = spec.candidate_id
    evidence["estimator_family"] = spec.estimator_family
    evidence["feature_profile"] = (
        spec.feature_profile or "deterministic_score"
    )
    evidence["validation_fold"] = fold
    evidence["ticker_cohort"] = cohort
    evidence["calibration_method"] = calibration_fit.method
    evidence["calibration_train_cutoff_utc"] = (
        calibration_fit.train_cutoff_utc.isoformat()
    )
    evidence["calibration_training_rows"] = (
        calibration_fit.training_rows
    )
    return evidence


def _fold_record(
    fold: V3Fold,
    *,
    scope: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_train_label: pd.Timestamp,
    min_test_decision: pd.Timestamp,
    feature_set_sha256: str,
    calibration_fit: CausalCalibrationFit | None,
    split_sha256: str,
) -> dict[str, object]:
    return {
        **fold.audit_record(),
        "record_type": "validation_fold",
        "validation_scope": scope,
        "validation_status": (
            "included"
            if calibration_fit is not None
            else "calibration_seed_excluded"
        ),
        "train_rows": len(train),
        "test_rows": len(test),
        "max_train_label_available_at_utc": max_train_label.isoformat(),
        "min_test_decision_time_utc": min_test_decision.isoformat(),
        "train_ticker_count": int(train["ticker"].nunique()),
        "test_ticker_count": int(test["ticker"].nunique()),
        "train_ticker_set_sha256": identity_set_sha256(
            train["ticker"].unique()
        ),
        "test_ticker_set_sha256": identity_set_sha256(
            test["ticker"].unique()
        ),
        "train_row_identity_sha256": identity_set_sha256(
            train["validation_row_identity"]
        ),
        "test_row_identity_sha256": identity_set_sha256(
            test["validation_row_identity"]
        ),
        "feature_set_sha256": feature_set_sha256,
        "split_sha256": split_sha256,
        "calibration_method": (
            calibration_fit.method
            if calibration_fit is not None
            else "seed_only_not_scored"
        ),
        "calibration_train_cutoff_utc": (
            calibration_fit.train_cutoff_utc.isoformat()
            if calibration_fit is not None
            else ""
        ),
        "calibration_training_rows": (
            calibration_fit.training_rows
            if calibration_fit is not None
            else 0
        ),
    }


def _candidate_metrics(
    walk_forward: pd.DataFrame,
    ticker_holdout: pd.DataFrame,
    *,
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    feature_set_sha256: str,
    calibration_seed_folds_excluded: int,
    config: SwingSpecialistResearchConfig,
) -> dict[str, object]:
    temporal = classification_metrics(
        walk_forward["strategy_target"],
        walk_forward["swing_probability"],
    )
    unseen = classification_metrics(
        ticker_holdout["strategy_target"],
        ticker_holdout["swing_probability"],
    )
    temporal_group = group_ranking_metrics(
        walk_forward,
        target_column="strategy_target",
        score=swing_decision_scores(
            walk_forward,
            probability_column="swing_probability",
        ),
        group_column="decision_group_id",
        k=config.top_k,
    )
    unseen_group = group_ranking_metrics(
        ticker_holdout,
        target_column="strategy_target",
        score=swing_decision_scores(
            ticker_holdout,
            probability_column="swing_probability",
        ),
        group_column="decision_group_id",
        k=config.top_k,
    )
    temporal_calibration = calibration_summary(
        walk_forward["strategy_target"],
        walk_forward["swing_probability"],
    )
    unseen_calibration = calibration_summary(
        ticker_holdout["strategy_target"],
        ticker_holdout["swing_probability"],
    )
    return {
        "model_run_id": (
            f"ks3-{spec.candidate_id}-{uuid.uuid4().hex}"
        ),
        "strategy_id": spec.strategy_id,
        "candidate_id": spec.candidate_id,
        "estimator_family": spec.estimator_family,
        "feature_profile": (
            spec.feature_profile or "deterministic_score"
        ),
        "deterministic_score": spec.deterministic_score,
        "validation_split": SPECIALIST_VALIDATION_SPLIT,
        "split_sha256": plan.split_sha256,
        "feature_set_sha256": feature_set_sha256,
        "features": list(
            ()
            if spec.feature_profile is None
            else plan.profile_features[spec.feature_profile]
        ),
        "horizon_sessions": plan.horizon_sessions,
        "walk_forward_rows": len(walk_forward),
        "ticker_holdout_rows": len(ticker_holdout),
        "holdout_tickers": sorted(plan.holdout_tickers),
        "calibration_method": "isotonic_prior_outer_folds",
        "calibration_seed_folds_excluded": (
            calibration_seed_folds_excluded
        ),
        "walk_forward": {
            **temporal,
            **temporal_group,
            **temporal_calibration,
        },
        "ticker_holdout": {
            **unseen,
            **unseen_group,
            **unseen_calibration,
        },
    }


def _economic_rejection_reasons(
    economics: pd.DataFrame,
    *,
    config: SwingSpecialistResearchConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for cohort in ("seen", "unseen"):
        scope = f"full_cross_section:{cohort}"
        rows = economics.loc[economics["scope"].eq(scope)]
        if rows.empty:
            reasons.append(f"{scope} has no economic evidence")
            continue
        conservative = conservative_economics(rows).iloc[0]
        gates = (
            (
                "selected_trades",
                conservative["selected_trades"],
                config.minimum_selected_trades,
                "min",
            ),
            (
                "avg_trade_return",
                conservative["avg_trade_return"],
                config.minimum_avg_net_return,
                "min",
            ),
            (
                "avg_excess_return_vs_spy",
                conservative["avg_excess_return_vs_spy"],
                config.minimum_avg_excess_return_vs_spy,
                "min",
            ),
            (
                "avg_excess_return_vs_sector",
                conservative["avg_excess_return_vs_sector"],
                config.minimum_avg_excess_return_vs_sector,
                "min",
            ),
            (
                "profit_factor",
                conservative["profit_factor"],
                config.minimum_profit_factor,
                "min",
            ),
            (
                "max_drawdown",
                conservative["max_drawdown"],
                config.maximum_drawdown,
                "max",
            ),
            (
                "negative_period_rate",
                conservative["negative_period_rate"],
                config.maximum_negative_phase_rate,
                "max",
            ),
        )
        for name, raw_value, threshold, direction in gates:
            value = _finite_or_infinite(raw_value)
            passed = (
                value >= float(threshold)
                if direction == "min"
                else value <= float(threshold)
            )
            if not passed:
                reasons.append(
                    f"{scope}.{name} {value} fails "
                    f"{direction} {threshold}"
                )
    return tuple(reasons)


def _regime_evidence(
    predictions: pd.DataFrame,
    *,
    horizon: int,
    top_k: int,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for regime, rows in predictions.groupby("market_regime", sort=True):
        evidence = phase_economics(
            rows,
            horizon=horizon,
            top_k=top_k,
            scope=f"regime:{regime}",
            cohort_column="ticker_cohort",
            use_stamped_net_returns=True,
        )
        evidence["market_regime"] = str(regime)
        records.append(evidence)
    return (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame()
    )


def _capacity_evidence(
    predictions: pd.DataFrame,
    *,
    top_k: int,
    participation_rate: float,
) -> pd.DataFrame:
    selected = select_swing_candidates(
        predictions,
        policy=PredictionSelectionPolicy(swing_top_k=top_k),
        probability_column="swing_probability",
    )
    if selected.empty or "dollar_volume" not in selected:
        return pd.DataFrame(
            [
                {
                    "selected_trades": 0,
                    "participation_rate": participation_rate,
                    "median_trade_capacity_usd": None,
                    "p10_trade_capacity_usd": None,
                    "median_session_capacity_usd": None,
                }
            ]
        )
    selected = selected.copy()
    selected["_capacity"] = (
        pd.to_numeric(selected["dollar_volume"], errors="coerce")
        * participation_rate
    )
    selected = selected.loc[selected["_capacity"].gt(0)]
    session_capacity = selected.groupby("session_date_et")[
        "_capacity"
    ].sum()
    return pd.DataFrame(
        [
            {
                "selected_trades": len(selected),
                "top_k": top_k,
                "participation_rate": participation_rate,
                "median_trade_capacity_usd": float(
                    selected["_capacity"].median()
                ),
                "p10_trade_capacity_usd": float(
                    selected["_capacity"].quantile(0.10)
                ),
                "median_session_capacity_usd": float(
                    session_capacity.median()
                ),
            }
        ]
    )


def _split_identity(
    development: pd.DataFrame,
    ticker_holdout: pd.DataFrame,
    folds: tuple[V3Fold, ...],
    holdout_tickers: frozenset[str],
) -> str:
    records = [
        {
            **fold.audit_record(),
            "train_identity": identity_set_sha256(
                development.iloc[fold.train_indices][
                    "validation_row_identity"
                ]
            ),
            "test_identity": identity_set_sha256(
                development.iloc[fold.test_indices][
                    "validation_row_identity"
                ]
            ),
        }
        for fold in folds
    ]
    material = "|".join(
        (
            identity_set_sha256(sorted(holdout_tickers)),
            identity_set_sha256(
                ticker_holdout["validation_row_identity"]
            ),
            str(records),
        )
    )
    return identity_set_sha256([material])


def _xgboost_memory_guard(
    xgboost: Any,
    config: SwingSpecialistResearchConfig,
) -> object:
    def after_iteration(
        self: object,
        model: object,
        epoch: int,
        evals_log: dict[str, object],
    ) -> bool:
        del self, model, evals_log
        _assert_memory(config, f"direct ranker iteration {epoch}")
        return False

    callback_type = type(
        "KS3MemoryGuard",
        (xgboost.callback.TrainingCallback,),
        {"after_iteration": after_iteration},
    )
    return callback_type()


def _assert_memory(
    config: SwingSpecialistResearchConfig,
    stage: str,
) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )


def _require_binary_target(values: pd.Series, name: str) -> None:
    unique = set(
        pd.to_numeric(values, errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if unique != {0, 1}:
        raise DataReadinessError(
            f"{name} requires both target classes; found {sorted(unique)}"
        )


def _finite_or_infinite(value: object) -> float:
    if not isinstance(
        value,
        (str, int, float, np.integer, np.floating),
    ):
        return float("nan")
    try:
        number = float(value)
    except (TypeError, ValueError):
        return float("nan")
    if math.isnan(number):
        return float("nan")
    return number
