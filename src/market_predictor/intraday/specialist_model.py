"""Causal evaluation engine for KS4 intraday specialist candidates."""

from __future__ import annotations

import hashlib
import importlib
import json
import math
from dataclasses import dataclass
from typing import Any, Protocol, cast
from zoneinfo import ZoneInfo

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.execution_policy import DEFAULT_EXECUTION_POLICY
from market_predictor.intraday.evaluation import (
    classification_metrics,
    conservative_economics,
    phase_economics,
    prediction_evidence,
)
from market_predictor.intraday.specialist_contracts import (
    EstimatorFamily,
    IntradaySpecialistResearchConfig,
)
from market_predictor.prediction_policy import (
    INTRADAY_SELECTION_DOWNSIDE_CEILING,
    group_ranking_metrics,
    intraday_selection_eligible,
)
from market_predictor.registry import feature_schema_hash
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
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
    causal_fold_training_indices,
    deterministic_stratified_ticker_holdout,
    identity_set_sha256,
)

SPECIALIST_VALIDATION_SPLIT = (
    "session_purged_walk_forward_and_ticker_holdout"
)
SPECIALIST_ACCEPTED_STATUS = "accepted_development"
SPECIALIST_REJECTED_STATUS = "rejected"
_DECISION_INTERVAL_MINUTES = 5
_DECISION_CLOCK_OFFSET_MINUTES = 1
_NEW_YORK = ZoneInfo("America/New_York")

DETERMINISTIC_SCORE_FORMULAS = {
    "opening_breakout_confirmation": (
        "dist_opening_range_high + log1p(max(relative_volume_same_minute_20d,0)) "
        "+ close_location_5m + rel_return_3bar_vs_qqq"
    ),
    "gap_continuation_confirmation": (
        "overnight_gap + log1p(max(relative_volume_same_minute_20d,0)) "
        "+ return_1bar + dist_session_vwap"
    ),
    "gap_fade_confirmation": (
        "-overnight_gap + return_1bar + close_location_5m "
        "+ dist_opening_range_low"
    ),
    "vwap_continuation_confirmation": (
        "return_3bar + session_vwap_slope_3bar "
        "+ log1p(max(relative_volume_same_minute_20d,0)) "
        "+ rel_return_3bar_vs_sector - abs(dist_session_vwap)"
    ),
    "vwap_reversion_confirmation": (
        "-dist_session_vwap_atr_units + (50-rsi_14)/50 "
        "+ return_1bar + close_location_5m"
    ),
    "cross_sectional_momentum": (
        "xs_rank_return_3bar + xs_rank_relative_volume_same_minute_20d "
        "+ xs_rank_dollar_volume + rel_return_3bar_vs_qqq"
    ),
    "shock_reversal_confirmation": (
        "-return_3bar_atr_units + (50-rsi_14)/50 + return_1bar "
        "+ log1p(max(volume_burst_20bar,0))"
    ),
    "downside_risk": (
        "max(atr_pct,0) "
        "+ max(coalesce(volatility_12bar,volatility_6bar),0) "
        "+ max(-return_1bar,0) + max(-rel_return_3bar_vs_sector,0) "
        "+ max(1-close_location_5m,0)"
    ),
}
DETERMINISTIC_SCORE_FORMULA_SHA256 = hashlib.sha256(
    json.dumps(
        DETERMINISTIC_SCORE_FORMULAS,
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
).hexdigest()


class ProbabilityEstimator(Protocol):
    def fit(self, x: pd.DataFrame, y: pd.Series, **kwargs: Any) -> Any: ...

    def predict_proba(self, x: pd.DataFrame) -> np.ndarray: ...


@dataclass(frozen=True, slots=True)
class SpecialistExperimentSpec:
    strategy_id: str
    estimator_family: EstimatorFamily
    deterministic_score: str

    @property
    def candidate_id(self) -> str:
        return self.estimator_family


@dataclass(frozen=True)
class SpecialistSplitPlan:
    strategy_id: str
    horizon_minutes: int
    opportunity_target: str
    downside_target: str
    data: pd.DataFrame
    development: pd.DataFrame
    ticker_holdout: pd.DataFrame
    folds: tuple[V3Fold, ...]
    holdout_tickers: frozenset[str]
    representation_audit: pd.DataFrame
    features: tuple[str, ...]
    split_sha256: str


@dataclass(frozen=True)
class RetainedSpecialistModel:
    """Loadable state retained only after every acceptance gate passes."""

    estimators: dict[str, object | None]
    calibrators: dict[str, object]
    features: tuple[str, ...]
    opportunity_target: str
    downside_target: str


@dataclass(frozen=True)
class SpecialistExperimentResult:
    spec: SpecialistExperimentSpec
    status: str
    rejection_reasons: tuple[str, ...]
    metrics: dict[str, object]
    predictions: pd.DataFrame
    economics: pd.DataFrame
    regime_evidence: pd.DataFrame
    fold_audit: pd.DataFrame
    retained_model: RetainedSpecialistModel | None


def specialist_experiment_specs(
    strategy_id: str,
    config: IntradaySpecialistResearchConfig,
) -> tuple[SpecialistExperimentSpec, ...]:
    """Return the frozen candidate catalog in configured comparison order."""

    try:
        strategy = config.strategies[strategy_id]
    except KeyError as exc:
        raise DataReadinessError(
            f"unknown KS4 intraday strategy: {strategy_id}"
        ) from exc
    specs = tuple(
        SpecialistExperimentSpec(
            strategy_id=strategy_id,
            estimator_family=family,
            deterministic_score=strategy.deterministic_score,
        )
        for family in strategy.estimator_families
    )
    has_ranker = any(
        spec.estimator_family == "direct_ranker" for spec in specs
    )
    is_momentum = (
        strategy_id == "INTRADAY.MOMENTUM_CONTINUATION.60M.V1"
    )
    if has_ranker != is_momentum:
        raise DataReadinessError(
            "direct ranker is allowed only for intraday momentum"
        )
    return specs


def build_specialist_split_plan(
    dataset: pd.DataFrame,
    *,
    strategy_id: str,
    config: IntradaySpecialistResearchConfig,
) -> SpecialistSplitPlan:
    """Freeze one causal split and feature set shared by all candidates."""

    data, horizon, opportunity_target, downside_target = _training_rows(
        dataset,
        strategy_id=strategy_id,
        config=config,
    )
    if len(data) < config.min_train_rows:
        raise DataReadinessError(
            f"{strategy_id} needs at least {config.min_train_rows} rows"
        )
    if int(data["ticker"].nunique()) < config.min_training_tickers:
        raise DataReadinessError(
            f"{strategy_id} needs at least "
            f"{config.min_training_tickers} tickers"
        )
    assignment_folds = _xnys_purged_walk_forward_split(
        data,
        config=config,
    )
    if len(assignment_folds) != config.n_splits:
        raise DataReadinessError(
            f"{strategy_id} produced {len(assignment_folds)} of "
            f"{config.n_splits} assignment folds"
        )
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
    holdout_tickers = frozenset(holdout_plan.holdout_tickers)
    development = data.loc[
        ~data["ticker"].isin(holdout_tickers)
    ].reset_index(drop=True)
    ticker_holdout = data.loc[
        data["ticker"].isin(holdout_tickers)
    ].reset_index(drop=True)
    folds = _xnys_purged_walk_forward_split(
        development,
        config=config,
    )
    if len(folds) != config.n_splits:
        raise DataReadinessError(
            f"{strategy_id} produced {len(folds)} of "
            f"{config.n_splits} validation folds"
        )
    first_train_indices, _, _ = causal_fold_training_indices(
        development,
        candidate_indices=folds[0].train_indices,
        test_indices=folds[0].test_indices,
    )
    features = _select_features(
        development.iloc[first_train_indices],
        config=config,
    )
    if not features:
        raise DataReadinessError(
            f"{strategy_id} has no technical features with adequate coverage"
        )
    split_sha256 = _split_identity(
        development,
        ticker_holdout,
        folds,
        holdout_tickers,
    )
    _assert_memory(config, f"{strategy_id} split plan")
    return SpecialistSplitPlan(
        strategy_id=strategy_id,
        horizon_minutes=horizon,
        opportunity_target=opportunity_target,
        downside_target=downside_target,
        data=data,
        development=development,
        ticker_holdout=ticker_holdout,
        folds=folds,
        holdout_tickers=holdout_tickers,
        representation_audit=holdout_plan.representation_audit.copy(),
        features=features,
        split_sha256=split_sha256,
    )


def evaluate_specialist_experiment(
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    *,
    config: IntradaySpecialistResearchConfig,
) -> SpecialistExperimentResult:
    """Evaluate one candidate sequentially against a frozen strategy split."""

    if spec.strategy_id != plan.strategy_id:
        raise DataReadinessError("candidate strategy does not match split plan")
    configured = specialist_experiment_specs(spec.strategy_id, config)
    if spec not in configured:
        raise DataReadinessError("candidate is outside the frozen KS4 budget")

    feature_set_sha256 = feature_schema_hash(list(plan.features))
    targets = (plan.opportunity_target, plan.downside_target)
    calibration_raw: dict[str, list[np.ndarray]] = {
        target: [] for target in targets
    }
    calibration_targets: dict[str, list[np.ndarray]] = {
        target: [] for target in targets
    }
    calibration_availability: list[pd.Series] = []
    walk_forward_parts: list[pd.DataFrame] = []
    holdout_parts: list[pd.DataFrame] = []
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

        validation_raw: dict[str, np.ndarray] = {}
        ticker_raw: dict[str, np.ndarray] = {}
        for target in targets:
            _require_binary_target(
                train[target],
                f"{spec.candidate_id} fold {fold.fold} {target}",
            )
            estimator = _new_estimator(spec, config)
            _fit_candidate(
                estimator,
                spec,
                train,
                plan.features,
                target=target,
                config=config,
            )
            validation_raw[target] = _raw_scores(
                estimator,
                spec,
                validation,
                plan.features,
                target=target,
            )
            ticker_raw[target] = _raw_scores(
                estimator,
                spec,
                ticker_validation,
                plan.features,
                target=target,
            )
            del estimator
            release_process_memory()
            _assert_memory(
                config,
                f"{spec.candidate_id} fold {fold.fold} {target}",
            )

        calibration_fits: dict[str, CausalCalibrationFit] = {}
        if calibration_availability:
            availability = pd.concat(
                calibration_availability,
                ignore_index=True,
            )
            for target in targets:
                fitted = fit_prior_isotonic(
                    np.concatenate(calibration_raw[target]),
                    np.concatenate(calibration_targets[target]),
                    availability,
                    before_utc=min_test_decision,
                )
                if fitted is not None:
                    calibration_fits[target] = fitted
        included = len(calibration_fits) == len(targets)
        if included:
            walk_forward_parts.append(
                _fold_predictions(
                    validation,
                    raw=validation_raw,
                    calibration_fits=calibration_fits,
                    plan=plan,
                    spec=spec,
                    fold=fold.fold,
                    scope="walk_forward",
                    cohort="seen",
                )
            )
            holdout_parts.append(
                _fold_predictions(
                    ticker_validation,
                    raw=ticker_raw,
                    calibration_fits=calibration_fits,
                    plan=plan,
                    spec=spec,
                    fold=fold.fold,
                    scope="ticker_holdout",
                    cohort="unseen",
                )
            )
        else:
            calibration_seed_folds_excluded += 1
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
                    calibration_fits=(
                        calibration_fits if included else {}
                    ),
                    split_sha256=plan.split_sha256,
                )
            )
        for target in targets:
            calibration_raw[target].append(validation_raw[target])
            calibration_targets[target].append(
                validation[target].astype(int).to_numpy()
            )
        calibration_availability.append(
            validation["label_available_at_utc"]
        )

    if not walk_forward_parts or not holdout_parts:
        raise DataReadinessError(
            f"{spec.candidate_id} has no calibrated validation folds"
        )
    walk_forward = pd.concat(walk_forward_parts, ignore_index=True)
    ticker_holdout = pd.concat(holdout_parts, ignore_index=True)
    predictions = (
        pd.concat([walk_forward, ticker_holdout], ignore_index=True)
        .sort_values(
            [
                "decision_time_utc",
                "ticker",
                "ticker_cohort",
            ],
            kind="stable",
        )
        .reset_index(drop=True)
    )
    if bool(predictions["row_identity"].duplicated().any()):
        raise DataReadinessError(
            f"{spec.candidate_id} emitted duplicate validation rows"
        )

    economics = _economic_evidence(
        predictions,
        horizon_minutes=plan.horizon_minutes,
        config=config,
    )
    regime = _regime_evidence(
        predictions,
        horizon_minutes=plan.horizon_minutes,
        config=config,
    )
    rejection_reasons = _economic_rejection_reasons(
        economics,
        regime,
        config=config,
    )
    status = (
        SPECIALIST_ACCEPTED_STATUS
        if not rejection_reasons
        else SPECIALIST_REJECTED_STATUS
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

    retained_model: RetainedSpecialistModel | None = None
    if status == SPECIALIST_ACCEPTED_STATUS:
        final_calibrators = {
            target: fit_final_isotonic(
                np.concatenate(calibration_raw[target]),
                np.concatenate(calibration_targets[target]),
            )
            for target in targets
        }
        if any(value is None for value in final_calibrators.values()):
            raise DataReadinessError(
                f"{spec.candidate_id} lacks final calibration evidence"
            )
        final_estimators: dict[str, object | None] = {}
        for target in targets:
            estimator = _new_estimator(spec, config)
            _fit_candidate(
                estimator,
                spec,
                plan.data,
                plan.features,
                target=target,
                config=config,
            )
            final_estimators[target] = estimator
            _assert_memory(
                config,
                f"{spec.candidate_id} final {target}",
            )
        retained_model = RetainedSpecialistModel(
            estimators=final_estimators,
            calibrators=cast(dict[str, object], final_calibrators),
            features=plan.features,
            opportunity_target=plan.opportunity_target,
            downside_target=plan.downside_target,
        )
    else:
        release_process_memory()

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
        fold_audit=fold_audit,
        retained_model=retained_model,
    )


def _training_rows(
    dataset: pd.DataFrame,
    *,
    strategy_id: str,
    config: IntradaySpecialistResearchConfig,
) -> tuple[pd.DataFrame, int, str, str]:
    required = {
        "strategy_id",
        "setup_id",
        "ticker",
        "session_date_et",
        "decision_time_utc",
        "feature_available_at_utc",
        "entry_time_utc",
        "exit_time_utc",
        "label_available_at_utc",
        "label_window_end_utc",
        "label_eligible",
        "horizon_minutes",
    }
    missing = sorted(required.difference(dataset.columns))
    if missing:
        raise DataReadinessError(
            "KS4 training dataset is missing: " + ", ".join(missing)
        )
    observed = set(dataset["strategy_id"].astype(str))
    if observed != {strategy_id}:
        raise DataReadinessError(
            f"KS4 dataset strategy mismatch: {sorted(observed)}"
        )
    expected_horizon = config.strategies[strategy_id].horizon_minutes
    horizons = (
        pd.to_numeric(dataset["horizon_minutes"], errors="coerce")
        .dropna()
        .astype(int)
        .unique()
    )
    if len(horizons) != 1 or int(horizons[0]) != expected_horizon:
        raise DataReadinessError(
            f"{strategy_id} must contain only {expected_horizon}m labels"
        )
    horizon = int(horizons[0])
    opportunity_target = f"target_before_stop_{horizon}m"
    downside_target = f"stop_before_target_{horizon}m"
    evidence = {
        opportunity_target,
        downside_target,
        f"path_realized_return_gross_{horizon}m",
        f"path_realized_return_net_{horizon}m",
        f"path_excess_return_{horizon}m_vs_spy",
        f"path_excess_return_{horizon}m_vs_qqq",
        f"path_excess_return_{horizon}m_vs_sector",
    }
    missing_evidence = sorted(evidence.difference(dataset.columns))
    if missing_evidence:
        raise DataReadinessError(
            "KS4 training dataset lacks label evidence: "
            + ", ".join(missing_evidence)
        )
    data = dataset.loc[
        dataset["label_eligible"].fillna(False).astype(bool)
    ].copy()
    if data.empty:
        raise DataReadinessError(f"{strategy_id} has no eligible labels")
    decision = pd.to_datetime(
        data["decision_time_utc"], utc=True, errors="coerce"
    )
    feature = pd.to_datetime(
        data["feature_available_at_utc"], utc=True, errors="coerce"
    )
    entry = pd.to_datetime(
        data["entry_time_utc"], utc=True, errors="coerce"
    )
    exit_time = pd.to_datetime(
        data["exit_time_utc"], utc=True, errors="coerce"
    )
    label = pd.to_datetime(
        data["label_available_at_utc"], utc=True, errors="coerce"
    )
    window_end = pd.to_datetime(
        data["label_window_end_utc"], utc=True, errors="coerce"
    )
    invalid = (
        decision.isna()
        | feature.isna()
        | entry.isna()
        | exit_time.isna()
        | label.isna()
        | window_end.isna()
        | feature.gt(decision)
        | entry.lt(decision)
        | exit_time.le(entry)
        | exit_time.gt(window_end)
        | label.le(entry)
    )
    if bool(invalid.any()):
        raise DataReadinessError(
            "KS4 training rows contain future or invalid timestamps"
        )
    if bool(data["setup_id"].astype(str).duplicated().any()):
        raise DataReadinessError(
            "KS4 training rows contain duplicate setup identities"
        )
    gross = pd.to_numeric(
        data[f"path_realized_return_gross_{horizon}m"],
        errors="coerce",
    )
    net = pd.to_numeric(
        data[f"path_realized_return_net_{horizon}m"],
        errors="coerce",
    )
    stamped_cost = gross - net
    required_cost = config.minimum_round_trip_cost_bps / 10_000.0
    if bool(
        gross.isna().any()
        or net.isna().any()
        or stamped_cost.lt(required_cost - 1e-10).any()
    ):
        raise DataReadinessError(
            "KS4 eligible labels do not contain the minimum stamped cost"
        )
    data["decision_time_utc"] = decision
    data["feature_available_at_utc"] = feature
    data["entry_time_utc"] = entry
    data["exit_time_utc"] = exit_time
    data["label_available_at_utc"] = label
    data["label_window_end_utc"] = window_end
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    data["decision_group_id"] = (
        strategy_id
        + "|"
        + decision.dt.strftime("%Y-%m-%dT%H:%M:%S.%fZ")
    )
    data["validation_row_identity"] = data["setup_id"].astype(str)
    data["market_regime"] = _market_regime(data)
    _validate_xnys_rows(data)
    _prove_non_overlapping_intervals(data)
    data["overlap_weight"] = 1.0
    data["concurrent_label_count"] = 1
    data["independent_event_id"] = data["setup_id"].astype(str)
    for target in (opportunity_target, downside_target):
        data[target] = pd.to_numeric(
            data[target], errors="raise"
        ).astype("int8")
        _require_binary_target(data[target], f"{strategy_id} {target}")
    return (
        data.sort_values(
            ["session_date_et", "decision_time_utc", "ticker"],
            kind="stable",
        ).reset_index(drop=True),
        horizon,
        opportunity_target,
        downside_target,
    )


def _select_features(
    training: pd.DataFrame,
    *,
    config: IntradaySpecialistResearchConfig,
) -> tuple[str, ...]:
    return tuple(
        feature
        for feature in config.technical_features
        if feature in training
        and pd.to_numeric(
            training[feature], errors="coerce"
        ).notna().mean()
        >= config.minimum_feature_non_null_rate
    )


def _new_estimator(
    spec: SpecialistExperimentSpec,
    config: IntradaySpecialistResearchConfig,
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
    if (
        spec.strategy_id
        != "INTRADAY.MOMENTUM_CONTINUATION.60M.V1"
    ):
        raise DataReadinessError(
            "direct ranker is allowed only for intraday momentum"
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
    target: str,
    config: IntradaySpecialistResearchConfig,
) -> None:
    if spec.estimator_family == "deterministic_baseline":
        return
    if estimator is None:
        raise AssertionError("learned candidate has no estimator")
    matrix = _matrix(data, features)
    target_values = data[target].astype(int)
    if spec.estimator_family == "direct_ranker":
        order = np.lexsort(
            (
                data["ticker"].astype(str).to_numpy(),
                data["decision_group_id"].astype(str).to_numpy(),
            )
        )
        ordered = data.iloc[order]
        query_id = pd.factorize(
            ordered["decision_group_id"], sort=False
        )[0].astype(np.int32, copy=False)
        cast(Any, estimator).fit(
            matrix.iloc[order].to_numpy(dtype=np.float32, copy=False),
            target_values.iloc[order].to_numpy(
                dtype=np.int16, copy=False
            ),
            qid=query_id,
        )
        cast(Any, estimator).set_params(callbacks=None)
        return
    cast(ProbabilityEstimator, estimator).fit(
        matrix,
        target_values,
        classifier__sample_weight=np.ones(len(data), dtype=float),
    )


def _raw_scores(
    estimator: object | None,
    spec: SpecialistExperimentSpec,
    data: pd.DataFrame,
    features: tuple[str, ...],
    *,
    target: str,
) -> np.ndarray:
    if spec.estimator_family == "deterministic_baseline":
        return _deterministic_score(
            data,
            spec.deterministic_score,
            downside=target.startswith("stop_before_target_"),
        )
    if estimator is None:
        raise AssertionError("learned candidate has no estimator")
    matrix = _matrix(data, features)
    if spec.estimator_family == "direct_ranker":
        return np.asarray(
            cast(Any, estimator).predict(
                matrix.to_numpy(dtype=np.float32, copy=False)
            ),
            dtype=float,
        )
    return np.asarray(
        cast(ProbabilityEstimator, estimator).predict_proba(matrix)[
            :, 1
        ],
        dtype=float,
    )


def _deterministic_score(
    data: pd.DataFrame,
    score_name: str,
    *,
    downside: bool,
) -> np.ndarray:
    def number(column: str) -> pd.Series:
        if column not in data:
            raise DataReadinessError(
                f"deterministic score is missing {column}"
            )
        return pd.to_numeric(data[column], errors="coerce")

    if downside:
        volatility = number("volatility_12bar")
        if bool(volatility.isna().any()):
            volatility = volatility.fillna(number("volatility_6bar"))
        score = (
            number("atr_pct").clip(lower=0)
            + volatility.clip(lower=0)
            + (-number("return_1bar")).clip(lower=0)
            + (-number("rel_return_3bar_vs_sector")).clip(lower=0)
            + (1.0 - number("close_location_5m")).clip(lower=0)
        )
    elif score_name == "opening_breakout_confirmation":
        score = (
            number("dist_opening_range_high")
            + np.log1p(
                number("relative_volume_same_minute_20d").clip(lower=0)
            )
            + number("close_location_5m")
            + number("rel_return_3bar_vs_qqq")
        )
    elif score_name == "gap_continuation_confirmation":
        score = (
            number("overnight_gap")
            + np.log1p(
                number("relative_volume_same_minute_20d").clip(lower=0)
            )
            + number("return_1bar")
            + number("dist_session_vwap")
        )
    elif score_name == "gap_fade_confirmation":
        score = (
            -number("overnight_gap")
            + number("return_1bar")
            + number("close_location_5m")
            + number("dist_opening_range_low")
        )
    elif score_name == "vwap_continuation_confirmation":
        score = (
            number("return_3bar")
            + number("session_vwap_slope_3bar")
            + np.log1p(
                number("relative_volume_same_minute_20d").clip(lower=0)
            )
            + number("rel_return_3bar_vs_sector")
            - number("dist_session_vwap").abs()
        )
    elif score_name == "vwap_reversion_confirmation":
        score = (
            -number("dist_session_vwap_atr_units")
            + (50.0 - number("rsi_14")) / 50.0
            + number("return_1bar")
            + number("close_location_5m")
        )
    elif score_name == "cross_sectional_momentum":
        score = (
            number("xs_rank_return_3bar")
            + number("xs_rank_relative_volume_same_minute_20d")
            + number("xs_rank_dollar_volume")
            + number("rel_return_3bar_vs_qqq")
        )
    elif score_name == "shock_reversal_confirmation":
        score = (
            -number("return_3bar_atr_units")
            + (50.0 - number("rsi_14")) / 50.0
            + number("return_1bar")
            + np.log1p(number("volume_burst_20bar").clip(lower=0))
        )
    else:
        raise DataReadinessError(
            f"unsupported deterministic score: {score_name}"
        )
    values: np.ndarray = score.to_numpy(dtype=float)
    if not np.isfinite(values).all():
        raise DataReadinessError(
            f"deterministic score {score_name} is not finite"
        )
    return values


def _matrix(
    data: pd.DataFrame,
    features: tuple[str, ...],
) -> pd.DataFrame:
    return data.loc[:, list(features)].apply(
        pd.to_numeric, errors="coerce"
    ).astype("float32")


def _fold_predictions(
    frame: pd.DataFrame,
    *,
    raw: dict[str, np.ndarray],
    calibration_fits: dict[str, CausalCalibrationFit],
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    fold: int,
    scope: str,
    cohort: str,
) -> pd.DataFrame:
    opportunity_raw = raw[plan.opportunity_target]
    downside_raw = raw[plan.downside_target]
    evidence = prediction_evidence(
        frame,
        opportunity_raw=opportunity_raw,
        opportunity_probability=apply_isotonic(
            calibration_fits[plan.opportunity_target].calibrator,
            opportunity_raw,
        ),
        downside_raw=downside_raw,
        downside_probability=apply_isotonic(
            calibration_fits[plan.downside_target].calibrator,
            downside_raw,
        ),
        scope=scope,
        horizon_minutes=plan.horizon_minutes,
    )
    evidence["selection_raw_score"] = opportunity_raw
    evidence["row_identity"] = frame[
        "validation_row_identity"
    ].astype(str).to_numpy()
    evidence["strategy_id"] = spec.strategy_id
    evidence["candidate_id"] = spec.candidate_id
    evidence["estimator_family"] = spec.estimator_family
    evidence["validation_fold"] = fold
    evidence["ticker_cohort"] = cohort
    evidence["calibration_method"] = "isotonic_prior_outer_folds"
    cutoffs = [
        fitted.train_cutoff_utc for fitted in calibration_fits.values()
    ]
    evidence["calibration_train_cutoff_utc"] = max(cutoffs).isoformat()
    evidence["calibration_training_rows"] = min(
        fitted.training_rows for fitted in calibration_fits.values()
    )
    return evidence


def _selection_view(predictions: pd.DataFrame) -> pd.DataFrame:
    """Apply the calibrated downside veto, then expose raw opportunity."""

    view = predictions.copy()
    eligible = intraday_selection_eligible(
        view,
        downside_column="intraday_downside_probability",
        downside_ceiling=INTRADAY_SELECTION_DOWNSIDE_CEILING,
    )
    view = view.loc[eligible].copy()
    raw = pd.to_numeric(
        view["selection_raw_score"], errors="coerce"
    )
    if bool(raw.isna().any()) or not np.isfinite(raw.to_numpy()).all():
        raise DataReadinessError("candidate emitted a non-finite raw score")
    view["intraday_opportunity_probability"] = raw
    view["intraday_downside_probability"] = 0.0
    gross_columns = [
        column
        for column in view
        if column.startswith("path_realized_return_gross_")
    ]
    return view.drop(columns=gross_columns)


def _economic_evidence(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    config: IntradaySpecialistResearchConfig,
) -> pd.DataFrame:
    stress_multiplier = max(DEFAULT_EXECUTION_POLICY.stress_multipliers)
    records: list[pd.DataFrame] = []
    for cohort, rows in predictions.groupby("ticker_cohort", sort=True):
        stream = (
            "walk_forward" if cohort == "seen" else "ticker_holdout"
        )
        base = _clock_phase_economics(
            rows,
            horizon_minutes=horizon_minutes,
            scope=stream,
            config=config,
            cost_stress=1.0,
        )
        base["cost_stress_multiplier"] = 1.0
        records.append(base)
        stress = _clock_phase_economics(
            rows,
            horizon_minutes=horizon_minutes,
            scope=f"cost_stress:{stream}",
            config=config,
            cost_stress=stress_multiplier,
        )
        stress["cost_stress_multiplier"] = stress_multiplier
        records.append(stress)
    return pd.concat(records, ignore_index=True)


def _regime_evidence(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    config: IntradaySpecialistResearchConfig,
) -> pd.DataFrame:
    records: list[pd.DataFrame] = []
    for (regime, cohort), rows in predictions.groupby(
        ["market_regime", "ticker_cohort"],
        sort=True,
    ):
        stream = (
            "walk_forward" if cohort == "seen" else "ticker_holdout"
        )
        evidence = _clock_phase_economics(
            rows,
            horizon_minutes=horizon_minutes,
            scope=f"regime:{regime}:{stream}",
            config=config,
            cost_stress=1.0,
        )
        evidence["market_regime"] = str(regime)
        evidence["validation_stream"] = stream
        records.append(evidence)
    return (
        pd.concat(records, ignore_index=True)
        if records
        else pd.DataFrame()
    )


def _clock_phase_economics(
    predictions: pd.DataFrame,
    *,
    horizon_minutes: int,
    scope: str,
    config: IntradaySpecialistResearchConfig,
    cost_stress: float,
) -> pd.DataFrame:
    """Evaluate sparse setups on actual non-overlapping clock phases."""

    view = _selection_view(predictions)
    decision = pd.to_datetime(
        view["decision_time_utc"], utc=True, errors="coerce"
    )
    local = decision.dt.tz_convert(_NEW_YORK)
    minute = local.dt.hour * 60 + local.dt.minute
    offset = minute - 570
    aligned = (
        decision.notna()
        & offset.ge(0)
        & offset.lt(390)
        & (offset - _DECISION_CLOCK_OFFSET_MINUTES)
        .mod(_DECISION_INTERVAL_MINUTES)
        .eq(0)
        & local.dt.second.eq(0)
        & local.dt.microsecond.eq(0)
    )
    if not bool(aligned.all()):
        raise DataReadinessError(
            "KS4 decisions are not aligned to the five-minute XNYS clock"
        )
    phase_count = horizon_minutes // _DECISION_INTERVAL_MINUTES
    view["_clock_phase"] = (
        (offset - _DECISION_CLOCK_OFFSET_MINUTES)
        .floordiv(_DECISION_INTERVAL_MINUTES)
        .mod(phase_count)
    ).astype(int)
    records: list[pd.DataFrame] = []
    observed_phases = sorted(view["_clock_phase"].unique())
    for phase in observed_phases:
        phase_rows = view.loc[view["_clock_phase"].eq(phase)].drop(
            columns="_clock_phase"
        )
        evidence = phase_economics(
            phase_rows,
            horizon_minutes=horizon_minutes,
            decision_interval_minutes=horizon_minutes,
            top_k=config.top_k,
            downside_ceiling=1.0,
            max_trades_per_session=config.max_trades_per_session,
            scope=scope,
            cost_stress=cost_stress,
        )
        evidence["phase"] = phase
        records.append(evidence)
    if not records:
        return _empty_phase_economics(scope, -1)
    return pd.concat(records, ignore_index=True)


def _empty_phase_economics(scope: str, phase: int) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "scope": scope,
                "phase": phase,
                "selected_trades": 0,
                "selected_decision_groups": 0,
                "avg_trade_return": float("nan"),
                "avg_trade_return_ci_low": float("nan"),
                "avg_excess_return_vs_spy_ci_low": float("nan"),
                "avg_excess_return_vs_spy": float("nan"),
                "avg_excess_return_vs_qqq": float("nan"),
                "avg_excess_return_vs_sector": float("nan"),
                "win_rate": float("nan"),
                "profit_factor": float("nan"),
                "cumulative_return": float("nan"),
                "max_drawdown": float("nan"),
                "return_drawdown_ratio": float("nan"),
                "negative_session_rate": float("nan"),
                "average_turnover": float("nan"),
                "sessions": 0,
            }
        ]
    )


def _economic_rejection_reasons(
    economics: pd.DataFrame,
    regime_evidence: pd.DataFrame,
    *,
    config: IntradaySpecialistResearchConfig,
) -> tuple[str, ...]:
    reasons: list[str] = []
    for stream in ("walk_forward", "ticker_holdout"):
        for stressed in (False, True):
            scope = f"cost_stress:{stream}" if stressed else stream
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
                    "avg_trade_return_ci_low",
                    conservative["avg_trade_return_ci_low"],
                    config.minimum_avg_net_return_ci_low,
                    "min",
                ),
                (
                    "avg_excess_return_vs_spy_ci_low",
                    conservative["avg_excess_return_vs_spy_ci_low"],
                    config.minimum_avg_excess_return_vs_spy_ci_low,
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
                    "negative_session_rate",
                    conservative["negative_session_rate"],
                    config.maximum_negative_session_rate,
                    "max",
                ),
            )
            for name, raw, threshold, direction in gates:
                value = _finite_or_infinite(raw)
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
    for stream in ("walk_forward", "ticker_holdout"):
        for regime in config.required_market_regimes:
            scope = f"regime:{regime}:{stream}"
            rows = regime_evidence.loc[
                regime_evidence["scope"].eq(scope)
            ]
            if rows.empty:
                reasons.append(f"{scope} has no regime evidence")
                continue
            conservative = conservative_economics(rows).iloc[0]
            regime_gates = (
                (
                    "selected_trades",
                    conservative["selected_trades"],
                    config.minimum_regime_selected_trades,
                ),
                (
                    "avg_trade_return",
                    conservative["avg_trade_return"],
                    config.minimum_regime_avg_net_return,
                ),
                (
                    "avg_excess_return_vs_spy",
                    conservative["avg_excess_return_vs_spy"],
                    config.minimum_regime_avg_excess_return_vs_spy,
                ),
            )
            for name, raw, threshold in regime_gates:
                value = _finite_or_infinite(raw)
                if value < float(threshold):
                    reasons.append(
                        f"{scope}.{name} {value} fails min {threshold}"
                    )
    return tuple(reasons)


def _candidate_metrics(
    walk_forward: pd.DataFrame,
    ticker_holdout: pd.DataFrame,
    *,
    plan: SpecialistSplitPlan,
    spec: SpecialistExperimentSpec,
    feature_set_sha256: str,
    calibration_seed_folds_excluded: int,
    config: IntradaySpecialistResearchConfig,
) -> dict[str, object]:
    scopes: dict[str, dict[str, object]] = {}
    for name, frame in (
        ("walk_forward", walk_forward),
        ("ticker_holdout", ticker_holdout),
    ):
        opportunity = classification_metrics(
            frame[plan.opportunity_target],
            frame["intraday_opportunity_probability"],
        )
        downside = classification_metrics(
            frame[plan.downside_target],
            frame["intraday_downside_probability"],
        )
        ranking = group_ranking_metrics(
            frame,
            target_column=plan.opportunity_target,
            score=frame["selection_raw_score"],
            group_column="decision_group_id",
            k=config.top_k,
            eligible=intraday_selection_eligible(
                frame,
                downside_column="intraday_downside_probability",
                downside_ceiling=(
                    INTRADAY_SELECTION_DOWNSIDE_CEILING
                ),
            ),
        )
        scopes[name] = {
            "opportunity": opportunity,
            "downside": downside,
            "raw_selection_ranking": ranking,
        }
    return {
        "model_run_id": (
            f"ks4-{spec.candidate_id}-{plan.split_sha256[:16]}-"
            f"{feature_set_sha256[:16]}"
        ),
        "strategy_id": spec.strategy_id,
        "candidate_id": spec.candidate_id,
        "estimator_family": spec.estimator_family,
        "validation_split": SPECIALIST_VALIDATION_SPLIT,
        "split_sha256": plan.split_sha256,
        "feature_set_sha256": feature_set_sha256,
        "features": list(plan.features),
        "horizon_minutes": plan.horizon_minutes,
        "walk_forward_rows": len(walk_forward),
        "ticker_holdout_rows": len(ticker_holdout),
        "holdout_tickers": sorted(plan.holdout_tickers),
        "calibration_method": "isotonic_prior_outer_folds",
        "calibration_seed_folds_excluded": (
            calibration_seed_folds_excluded
        ),
        "selection_policy": (
            "raw opportunity score descending after calibrated downside "
            f"probability <= {INTRADAY_SELECTION_DOWNSIDE_CEILING}; "
            "top-k per timestamp and capped per XNYS session/clock phase"
        ),
        "deterministic_score_formula": (
            DETERMINISTIC_SCORE_FORMULAS[spec.deterministic_score]
        ),
        "deterministic_downside_formula": (
            DETERMINISTIC_SCORE_FORMULAS["downside_risk"]
        ),
        "deterministic_score_formula_sha256": (
            DETERMINISTIC_SCORE_FORMULA_SHA256
        ),
        "economic_return_policy": (
            "stamped_net_return_with_minimum_10bps_round_trip_cost"
        ),
        "overlap_weight_policy": (
            "one_only_after_per_ticker_half_open_label_interval_proof"
        ),
        "session_ordinal_policy": "actual_XNYS_calendar_sessions",
        "catalyst_overlay_status": "data_blocked",
        **scopes,
    }


def _fold_record(
    fold: V3Fold,
    *,
    scope: str,
    train: pd.DataFrame,
    test: pd.DataFrame,
    max_train_label: pd.Timestamp,
    min_test_decision: pd.Timestamp,
    feature_set_sha256: str,
    calibration_fits: dict[str, CausalCalibrationFit],
    split_sha256: str,
) -> dict[str, object]:
    cutoffs = [
        fitted.train_cutoff_utc for fitted in calibration_fits.values()
    ]
    return {
        **fold.audit_record(),
        "record_type": "validation_fold",
        "validation_scope": scope,
        "validation_status": (
            "included"
            if calibration_fits
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
            "isotonic_prior_outer_folds"
            if calibration_fits
            else "seed_only_not_scored"
        ),
        "calibration_train_cutoff_utc": (
            max(cutoffs).isoformat() if cutoffs else ""
        ),
        "calibration_training_rows": min(
            (
                fitted.training_rows
                for fitted in calibration_fits.values()
            ),
            default=0,
        ),
    }


def _market_regime(data: pd.DataFrame) -> pd.Series:
    if "market_regime" in data:
        regime = data["market_regime"].fillna("neutral").astype(str)
        if bool(regime.str.strip().eq("").any()):
            raise DataReadinessError("market regime cannot be empty")
        return regime
    required = {
        "regime_risk_on",
        "regime_risk_off",
        "regime_high_volatility",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataReadinessError(
            "KS4 rows cannot derive market regime: " + ", ".join(missing)
        )
    risk_on = pd.to_numeric(
        data["regime_risk_on"], errors="coerce"
    ).fillna(0).gt(0)
    risk_off = pd.to_numeric(
        data["regime_risk_off"], errors="coerce"
    ).fillna(0).gt(0)
    high_vol = pd.to_numeric(
        data["regime_high_volatility"], errors="coerce"
    ).fillna(0).gt(0)
    values = np.select(
        [risk_off, risk_on, high_vol],
        ["risk_off", "risk_on", "high_volatility"],
        default="neutral",
    )
    return pd.Series(values, index=data.index, dtype="string")


def _validate_xnys_rows(data: pd.DataFrame) -> None:
    sessions = pd.to_datetime(
        data["session_date_et"], errors="coerce"
    ).dt.date
    if bool(sessions.isna().any()):
        raise DataReadinessError("KS4 rows contain an invalid session date")
    calendar = xcals.get_calendar("XNYS")
    labels = calendar.sessions_in_range(min(sessions), max(sessions))
    valid_dates = {pd.Timestamp(label).date() for label in labels}
    invalid_dates = sorted(set(sessions).difference(valid_dates))
    if invalid_dates:
        raise DataReadinessError(
            "KS4 rows include non-XNYS session dates: "
            + ", ".join(value.isoformat() for value in invalid_dates[:10])
        )
    opens = {
        pd.Timestamp(label).date(): pd.Timestamp(
            calendar.session_open(label)
        ).tz_convert("UTC")
        for label in labels
    }
    closes = {
        pd.Timestamp(label).date(): pd.Timestamp(
            calendar.session_close(label)
        ).tz_convert("UTC")
        for label in labels
    }
    session_open = pd.to_datetime(sessions.map(opens), utc=True)
    session_close = pd.to_datetime(sessions.map(closes), utc=True)
    decision = pd.to_datetime(data["decision_time_utc"], utc=True)
    entry = pd.to_datetime(data["entry_time_utc"], utc=True)
    window_end = pd.to_datetime(data["label_window_end_utc"], utc=True)
    invalid_clock = (
        decision.lt(session_open)
        | decision.ge(session_close)
        | entry.lt(session_open)
        | entry.ge(session_close)
        | window_end.gt(session_close)
    )
    if bool(invalid_clock.any()):
        raise DataReadinessError(
            "KS4 row timestamps are outside their actual XNYS session"
        )


def _prove_non_overlapping_intervals(data: pd.DataFrame) -> None:
    ordered = data.sort_values(
        ["ticker", "entry_time_utc", "label_window_end_utc"],
        kind="stable",
    )
    previous_end = ordered.groupby("ticker", sort=False)[
        "label_window_end_utc"
    ].shift()
    overlap = previous_end.gt(ordered["entry_time_utc"])
    if bool(overlap.fillna(False).any()):
        sample = ordered.loc[
            overlap.fillna(False),
            ["ticker", "entry_time_utc", "label_window_end_utc"],
        ].head(5)
        raise DataReadinessError(
            "KS4 per-ticker label intervals overlap; overlap_weight=1 "
            f"is not justified: {sample.to_dict(orient='records')}"
        )


def _xnys_purged_walk_forward_split(
    frame: pd.DataFrame,
    *,
    config: IntradaySpecialistResearchConfig,
) -> tuple[V3Fold, ...]:
    """Build expanding folds on actual XNYS ordinals, not sparse row dates."""

    sessions = pd.to_datetime(
        frame["session_date_et"], errors="coerce"
    ).dt.date
    if frame.empty or bool(sessions.isna().any()):
        raise DataReadinessError(
            "XNYS walk-forward requires valid non-empty session rows"
        )
    calendar = xcals.get_calendar("XNYS")
    labels = calendar.sessions_in_range(min(sessions), max(sessions))
    ordered_sessions = [pd.Timestamp(label).date() for label in labels]
    if not set(sessions).issubset(set(ordered_sessions)):
        raise DataReadinessError(
            "XNYS walk-forward received a non-calendar session"
        )
    first_test = config.min_train_sessions + config.embargo_sessions
    remaining = len(ordered_sessions) - first_test
    if remaining < config.n_splits:
        raise DataReadinessError(
            "insufficient actual XNYS sessions for training, embargo, "
            "and requested folds"
        )
    fold_size = max(1, remaining // config.n_splits)
    folds: list[V3Fold] = []
    for fold_number in range(config.n_splits):
        test_start = first_test + fold_number * fold_size
        test_end = (
            len(ordered_sessions)
            if fold_number == config.n_splits - 1
            else min(test_start + fold_size, len(ordered_sessions))
        )
        train_end = test_start - config.embargo_sessions
        train_sessions = set(ordered_sessions[:train_end])
        test_sessions = set(ordered_sessions[test_start:test_end])
        train_indices = np.flatnonzero(
            sessions.isin(train_sessions).to_numpy()
        )
        test_indices = np.flatnonzero(
            sessions.isin(test_sessions).to_numpy()
        )
        if (
            len(train_indices) < config.min_train_rows
            or len(test_indices) == 0
        ):
            continue
        train_groups = set(
            frame.iloc[train_indices]["decision_group_id"].astype(str)
        )
        test_groups = set(
            frame.iloc[test_indices]["decision_group_id"].astype(str)
        )
        if train_groups & test_groups:
            raise DataReadinessError(
                "a KS4 decision group crosses an XNYS fold"
            )
        folds.append(
            V3Fold(
                fold=fold_number,
                train_indices=train_indices,
                test_indices=test_indices,
                train_start=ordered_sessions[0],
                train_end=ordered_sessions[train_end - 1],
                test_start=ordered_sessions[test_start],
                test_end=ordered_sessions[test_end - 1],
                embargo_sessions=config.embargo_sessions,
            )
        )
    if len(folds) != config.n_splits:
        raise DataReadinessError(
            f"only {len(folds)} of {config.n_splits} XNYS folds have "
            "adequate strategy rows"
        )
    return tuple(folds)


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
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _xgboost_memory_guard(
    xgboost: Any,
    config: IntradaySpecialistResearchConfig,
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
        "KS4MemoryGuard",
        (xgboost.callback.TrainingCallback,),
        {"after_iteration": after_iteration},
    )
    return callback_type()


def _assert_memory(
    config: IntradaySpecialistResearchConfig,
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
