"""Causal, memory-bounded evaluation for primary V2 strategies."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal

import numpy as np
import pandas as pd
from numpy.typing import NDArray
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
)
from market_predictor.intraday.specialist_model import (
    build_specialist_split_plan as build_intraday_split_plan,
)
from market_predictor.primary_v2.contracts import (
    INTRADAY_V2_ID,
    SWING_V2_ID,
    CandidateFamily,
    PrimaryV2ResearchConfig,
    SelectionPolicy,
)
from market_predictor.regime_evidence import session_block_mean_interval
from market_predictor.resources import (
    assert_memory_budget,
    release_process_memory,
)
from market_predictor.swing.specialist_contracts import (
    SwingSpecialistResearchConfig,
)
from market_predictor.swing.specialist_model import (
    build_specialist_split_plan as build_swing_split_plan,
)
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.validation import (
    V3Fold,
    causal_fold_training_indices,
)

ValidationScope = Literal["walk_forward", "ticker_holdout"]


@dataclass(frozen=True, slots=True)
class PrimaryV2ExperimentSpec:
    strategy_id: str
    candidate_family: CandidateFamily
    selection_policy: SelectionPolicy

    @property
    def candidate_id(self) -> str:
        return f"{self.candidate_family}__{self.selection_policy}"


@dataclass(frozen=True)
class PrimaryV2SplitPlan:
    strategy_id: str
    source_strategy_id: str
    data: pd.DataFrame
    development: pd.DataFrame
    ticker_holdout: pd.DataFrame
    folds: tuple[V3Fold, ...]
    holdout_tickers: frozenset[str]
    features: tuple[str, ...]
    split_sha256: str


@dataclass(frozen=True)
class PrimaryV2ExperimentResult:
    spec: PrimaryV2ExperimentSpec
    status: Literal["accepted_development", "rejected"]
    rejection_reasons: tuple[str, ...]
    predictions: pd.DataFrame
    selected_predictions: pd.DataFrame
    economics: pd.DataFrame
    regime_evidence: pd.DataFrame
    calibration_evidence: pd.DataFrame
    incremental_evidence: pd.DataFrame
    fold_audit: pd.DataFrame
    metrics: dict[str, object]
    final_candidate: object | None


@dataclass
class _FittedCandidate:
    candidate_family: CandidateFamily
    mean_model: Any | None
    quantile_models: dict[float, Any]
    event_model: Any | None
    event_classes: tuple[str, ...]
    class_returns: dict[str, float]
    resolution_model: Any | None


def primary_v2_experiment_specs(
    strategy_id: str,
) -> tuple[PrimaryV2ExperimentSpec, ...]:
    """Return the frozen, valid candidate/policy matrix."""

    if strategy_id == SWING_V2_ID:
        pairs: tuple[tuple[CandidateFamily, SelectionPolicy], ...] = (
            ("deterministic_v1_baseline", "expected_net_top_10"),
            ("hgb_mean_return", "expected_net_top_10"),
            ("hgb_quantile_return", "expected_net_top_10"),
            (
                "hgb_quantile_return",
                "positive_lower_bound_then_median_top_10",
            ),
        )
    elif strategy_id == INTRADAY_V2_ID:
        pairs = (
            ("multinomial_v1_baseline", "no_veto_expected_net_top_10"),
            ("hgb_competing_risks", "no_veto_expected_net_top_10"),
            ("hgb_quantile_return", "no_veto_expected_net_top_10"),
            ("hgb_quantile_return", "distributional_safety_top_10"),
        )
    else:
        raise DataReadinessError(f"unknown primary V2 strategy: {strategy_id}")
    return tuple(
        PrimaryV2ExperimentSpec(
            strategy_id=strategy_id,
            candidate_family=family,
            selection_policy=policy,
        )
        for family, policy in pairs
    )


def build_swing_v2_split_plan(
    dataset: pd.DataFrame,
    *,
    v1_config: SwingSpecialistResearchConfig,
    v2_config: PrimaryV2ResearchConfig,
) -> PrimaryV2SplitPlan:
    """Reuse the exact KS3 split and technical feature profile."""

    strategy = v2_config.strategies[SWING_V2_ID]
    _require_matching_split_contract(
        v2_config,
        n_splits=v1_config.n_splits,
        holdout_fraction=v1_config.ticker_holdout_fraction,
    )
    source = build_swing_split_plan(
        dataset,
        strategy_id=strategy.source_strategy_id,
        config=v1_config,
    )
    features = source.profile_features["technical_only"]
    validate_primary_v2_source_rows(
        source.data,
        strategy_id=SWING_V2_ID,
        config=v2_config,
    )
    return PrimaryV2SplitPlan(
        strategy_id=SWING_V2_ID,
        source_strategy_id=strategy.source_strategy_id,
        data=source.data,
        development=source.development,
        ticker_holdout=source.ticker_holdout,
        folds=source.folds,
        holdout_tickers=source.holdout_tickers,
        features=features,
        split_sha256=source.split_sha256,
    )


def build_intraday_v2_split_plan(
    dataset: pd.DataFrame,
    *,
    v1_config: IntradaySpecialistResearchConfig,
    v2_config: PrimaryV2ResearchConfig,
) -> PrimaryV2SplitPlan:
    """Reuse the exact KS4 XNYS split and technical feature profile."""

    strategy = v2_config.strategies[INTRADAY_V2_ID]
    _require_matching_split_contract(
        v2_config,
        n_splits=v1_config.n_splits,
        holdout_fraction=v1_config.ticker_holdout_fraction,
    )
    source = build_intraday_split_plan(
        dataset,
        strategy_id=strategy.source_strategy_id,
        config=v1_config,
    )
    validate_primary_v2_source_rows(
        source.data,
        strategy_id=INTRADAY_V2_ID,
        config=v2_config,
    )
    return PrimaryV2SplitPlan(
        strategy_id=INTRADAY_V2_ID,
        source_strategy_id=strategy.source_strategy_id,
        data=source.data,
        development=source.development,
        ticker_holdout=source.ticker_holdout,
        folds=source.folds,
        holdout_tickers=source.holdout_tickers,
        features=source.features,
        split_sha256=source.split_sha256,
    )


def evaluate_primary_v2_experiment(
    plan: PrimaryV2SplitPlan,
    spec: PrimaryV2ExperimentSpec,
    *,
    config: PrimaryV2ResearchConfig,
    baseline_selected: pd.DataFrame | None = None,
) -> PrimaryV2ExperimentResult:
    """Fit and evaluate one frozen candidate sequentially by causal fold."""

    if spec not in primary_v2_experiment_specs(plan.strategy_id):
        raise DataReadinessError("candidate is outside the frozen primary V2 matrix")
    strategy = config.strategies[plan.strategy_id]
    prediction_parts: list[pd.DataFrame] = []
    fold_records: list[dict[str, object]] = []
    for fold in plan.folds:
        train_indices, max_train_label, min_test_decision = (
            causal_fold_training_indices(
                plan.development,
                candidate_indices=fold.train_indices,
                test_indices=fold.test_indices,
            )
        )
        train = plan.development.iloc[train_indices].reset_index(drop=True)
        validation = plan.development.iloc[fold.test_indices].reset_index(drop=True)
        sessions = set(pd.to_datetime(validation[strategy.period_column]).dt.date)
        unseen = plan.ticker_holdout.loc[
            pd.to_datetime(plan.ticker_holdout[strategy.period_column]).dt.date.isin(sessions)
        ].reset_index(drop=True)
        if unseen.empty:
            raise DataReadinessError(
                f"{spec.candidate_id} fold {fold.fold} has no unseen-ticker rows"
            )
        fitted = _fit_candidate(train, plan=plan, spec=spec, config=config)
        for scope, test in (
            ("walk_forward", validation),
            ("ticker_holdout", unseen),
        ):
            predictions = _predict_candidate(
                fitted,
                test,
                plan=plan,
                spec=spec,
                config=config,
            )
            predictions["validation_scope"] = scope
            predictions["fold"] = fold.fold
            prediction_parts.append(predictions)
            fold_records.append(
                {
                    "fold": fold.fold,
                    "validation_scope": scope,
                    "train_rows": len(train),
                    "test_rows": len(test),
                    "train_tickers": int(train["ticker"].nunique()),
                    "test_tickers": int(test["ticker"].nunique()),
                    "max_train_label_available_at_utc": max_train_label.isoformat(),
                    "min_test_decision_time_utc": min_test_decision.isoformat(),
                    "causal_label_gap": bool(max_train_label < min_test_decision),
                    "split_sha256": plan.split_sha256,
                }
            )
        del fitted
        release_process_memory()
        _assert_memory(config, f"{spec.candidate_id} fold {fold.fold}")

    predictions = pd.concat(prediction_parts, ignore_index=True)
    selected = _select_predictions(predictions, spec=spec, config=config)
    economics = _economic_evidence(selected, plan=plan, config=config)
    regimes = _regime_evidence(selected, plan=plan, config=config)
    calibration = _calibration_evidence(predictions, plan=plan)
    incremental = _incremental_evidence(
        selected,
        baseline_selected=baseline_selected,
        plan=plan,
        config=config,
    )
    rejection_reasons = _promotion_failures(
        economics,
        regimes,
        calibration,
        incremental,
        plan=plan,
        spec=spec,
        config=config,
    )
    metrics: dict[str, object] = {
        "strategy_id": plan.strategy_id,
        "source_strategy_id": plan.source_strategy_id,
        "candidate_id": spec.candidate_id,
        "split_sha256": plan.split_sha256,
        "features": list(plan.features),
        "feature_count": len(plan.features),
        "prediction_rows": len(predictions),
        "selected_rows": len(selected),
        "holdout_tickers": sorted(plan.holdout_tickers),
        "status": "rejected" if rejection_reasons else "accepted_development",
        "rejection_reasons": list(rejection_reasons),
    }
    final_candidate: object | None = None
    if not rejection_reasons:
        final_candidate = _fit_candidate(
            plan.data,
            plan=plan,
            spec=spec,
            config=config,
        )
        _assert_memory(config, f"{spec.candidate_id} final fit")
    return PrimaryV2ExperimentResult(
        spec=spec,
        status="rejected" if rejection_reasons else "accepted_development",
        rejection_reasons=tuple(rejection_reasons),
        predictions=predictions,
        selected_predictions=selected,
        economics=economics,
        regime_evidence=regimes,
        calibration_evidence=calibration,
        incremental_evidence=incremental,
        fold_audit=pd.DataFrame(fold_records),
        metrics=metrics,
        final_candidate=final_candidate,
    )


def _fit_candidate(
    train: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
    spec: PrimaryV2ExperimentSpec,
    config: PrimaryV2ResearchConfig,
) -> _FittedCandidate:
    strategy = config.strategies[plan.strategy_id]
    family = spec.candidate_family
    x = _feature_frame(train, plan.features)
    target = pd.to_numeric(train[strategy.source_target], errors="raise").astype(float)
    mean_model: Any | None = None
    quantile_models: dict[float, Any] = {}
    event_model: Any | None = None
    event_classes: tuple[str, ...] = ()
    class_returns: dict[str, float] = {}
    resolution_model: Any | None = None

    if family == "hgb_mean_return":
        mean_model = _new_hgb_regressor(config)
        mean_model.fit(x, target)
    elif family == "hgb_quantile_return":
        mean_model = _new_hgb_regressor(config)
        mean_model.fit(x, target)
        quantile_models = {
            quantile: _new_hgb_regressor(config, quantile=quantile)
            for quantile in config.quantiles
        }
        for model in quantile_models.values():
            model.fit(x, target)
        if plan.strategy_id == INTRADAY_V2_ID:
            event_model, event_classes, class_returns, resolution_model = (
                _fit_intraday_path_models(
                    train,
                    x=x,
                    target=target,
                    config=config,
                    use_hgb=True,
                )
            )
    elif family in {"multinomial_v1_baseline", "hgb_competing_risks"}:
        event_model, event_classes, class_returns, resolution_model = (
            _fit_intraday_path_models(
                train,
                x=x,
                target=target,
                config=config,
                use_hgb=family == "hgb_competing_risks",
            )
        )
    elif family != "deterministic_v1_baseline":
        raise DataReadinessError(f"unsupported V2 candidate family: {family}")
    return _FittedCandidate(
        candidate_family=family,
        mean_model=mean_model,
        quantile_models=quantile_models,
        event_model=event_model,
        event_classes=event_classes,
        class_returns=class_returns,
        resolution_model=resolution_model,
    )


def _fit_intraday_path_models(
    train: pd.DataFrame,
    *,
    x: pd.DataFrame,
    target: pd.Series,
    config: PrimaryV2ResearchConfig,
    use_hgb: bool,
) -> tuple[Any, tuple[str, ...], dict[str, float], Any]:
    strategy = config.strategies[INTRADAY_V2_ID]
    assert strategy.competing_risk_targets is not None
    outcome = train[strategy.competing_risk_targets.outcome].astype(str)
    required = {"target_first", "stop_first", "timeout"}
    if set(outcome.unique()) != required:
        raise DataReadinessError("intraday training fold lacks a competing path outcome")
    if use_hgb:
        event_model: Any = HistGradientBoostingClassifier(
            max_iter=config.hgb_max_iter,
            learning_rate=config.hgb_learning_rate,
            l2_regularization=config.hgb_l2_regularization,
            random_state=config.random_seed,
        )
    else:
        event_model = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                ("scale", StandardScaler()),
                (
                    "classifier",
                    LogisticRegression(
                        max_iter=500,
                        class_weight="balanced",
                        random_state=config.random_seed,
                    ),
                ),
            ]
        )
    event_model.fit(x, outcome)
    classes = tuple(str(value) for value in event_model.classes_)
    class_returns = {
        outcome_name: float(target.loc[outcome.eq(outcome_name)].mean())
        for outcome_name in classes
    }
    resolution = pd.to_numeric(
        train[strategy.competing_risk_targets.time_to_resolution],
        errors="raise",
    ).astype(float)
    resolution_model = _new_hgb_regressor(config)
    resolution_model.fit(x, resolution)
    return event_model, classes, class_returns, resolution_model


def _predict_candidate(
    fitted: _FittedCandidate,
    frame: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
    spec: PrimaryV2ExperimentSpec,
    config: PrimaryV2ResearchConfig,
) -> pd.DataFrame:
    strategy = config.strategies[plan.strategy_id]
    keep = sorted(strategy.required_source_columns.intersection(frame.columns))
    output = frame.loc[:, keep].copy().reset_index(drop=True)
    if "market_regime" in frame:
        output["market_regime"] = frame["market_regime"].astype(str).reset_index(drop=True)
    else:
        output["market_regime"] = _derive_market_regime(frame).reset_index(drop=True)
    x = _feature_frame(frame, plan.features)
    output["candidate_family"] = spec.candidate_family
    output["selection_policy"] = spec.selection_policy
    output["predicted_q10_net_return"] = np.nan
    output["predicted_q50_net_return"] = np.nan
    output["predicted_q90_net_return"] = np.nan
    output["predicted_target_first_probability"] = np.nan
    output["predicted_stop_first_probability"] = np.nan
    output["predicted_timeout_probability"] = np.nan
    output["predicted_resolution_minutes"] = np.nan

    if fitted.candidate_family == "deterministic_v1_baseline":
        score = pd.to_numeric(
            frame["xs_rank_rel_return_20d_vs_sector"],
            errors="coerce",
        )
        output["predicted_expected_net_return"] = score.to_numpy(float)
    else:
        if fitted.mean_model is not None:
            expected = np.asarray(fitted.mean_model.predict(x)).astype(float)
        else:
            probabilities = _event_probabilities(fitted, x)
            expected = np.zeros(len(x), dtype=float)
            for outcome in fitted.event_classes:
                expected += (
                    probabilities[outcome] * fitted.class_returns[outcome]
                )
        output["predicted_expected_net_return"] = expected
        if fitted.quantile_models:
            quantile_predictions = [
                np.asarray(
                    fitted.quantile_models[quantile].predict(x)
                ).astype(float)
                for quantile in config.quantiles
            ]
            raw = np.column_stack(quantile_predictions)
            ordered = np.sort(raw, axis=1)
            output["raw_quantile_crossing"] = (
                (raw[:, 0] > raw[:, 1]) | (raw[:, 1] > raw[:, 2])
            )
            for column, values in zip(
                (
                    "predicted_q10_net_return",
                    "predicted_q50_net_return",
                    "predicted_q90_net_return",
                ),
                ordered.T,
                strict=True,
            ):
                output[column] = values
        if fitted.event_model is not None:
            probabilities = _event_probabilities(fitted, x)
            for outcome in fitted.event_classes:
                output[f"predicted_{outcome}_probability"] = probabilities[outcome]
            if fitted.resolution_model is not None:
                bars = np.asarray(
                    fitted.resolution_model.predict(x)
                ).astype(float)
                output["predicted_resolution_minutes"] = np.clip(bars, 1, 30)
    return output


def _event_probabilities(
    fitted: _FittedCandidate,
    x: pd.DataFrame,
) -> dict[str, NDArray[np.float64]]:
    if fitted.event_model is None:
        raise DataReadinessError("competing-risk model is unavailable")
    matrix = np.asarray(fitted.event_model.predict_proba(x)).astype(float)
    matrix = matrix / matrix.sum(axis=1, keepdims=True)
    return {
        outcome: matrix[:, index]
        for index, outcome in enumerate(fitted.event_classes)
    }


def _select_predictions(
    predictions: pd.DataFrame,
    *,
    spec: PrimaryV2ExperimentSpec,
    config: PrimaryV2ResearchConfig,
) -> pd.DataFrame:
    strategy = config.strategies[spec.strategy_id]
    candidates = predictions.copy()
    if spec.selection_policy in {
        "positive_lower_bound_then_median_top_10",
        "distributional_safety_top_10",
    }:
        candidates = candidates.loc[
            pd.to_numeric(
                candidates["predicted_q10_net_return"],
                errors="coerce",
            ).gt(0)
        ].copy()
    if spec.selection_policy == "distributional_safety_top_10":
        utility = (
            pd.to_numeric(
                candidates["predicted_target_first_probability"],
                errors="coerce",
            )
            - pd.to_numeric(
                candidates["predicted_stop_first_probability"],
                errors="coerce",
            )
        )
        candidates = candidates.loc[utility.gt(0)].copy()
    score_column = (
        "predicted_q50_net_return"
        if spec.selection_policy
        == "positive_lower_bound_then_median_top_10"
        else "predicted_expected_net_return"
    )
    candidates["_selection_score"] = pd.to_numeric(
        candidates[score_column],
        errors="coerce",
    )
    candidates = candidates.dropna(subset=["_selection_score"])
    selected = (
        candidates.sort_values(
            [
                "validation_scope",
                strategy.period_column,
                "_selection_score",
                "ticker",
            ],
            ascending=[True, True, False, True],
            kind="stable",
        )
        .groupby(
            ["validation_scope", strategy.period_column],
            sort=False,
            observed=True,
        )
        .head(strategy.top_k)
        .reset_index(drop=True)
    )
    selected["selected"] = True
    return selected


def _economic_evidence(
    selected: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
    config: PrimaryV2ResearchConfig,
) -> pd.DataFrame:
    strategy = config.strategies[plan.strategy_id]
    records: list[dict[str, object]] = []
    for scope in ("walk_forward", "ticker_holdout"):
        scope_rows = selected.loc[selected["validation_scope"].eq(scope)].copy()
        if plan.strategy_id == SWING_V2_ID:
            sessions = sorted(
                pd.to_datetime(scope_rows[strategy.period_column]).dt.date.unique()
            )
            for phase in range(strategy.horizon_value):
                phase_sessions = set(sessions[phase :: strategy.horizon_value])
                rows = scope_rows.loc[
                    pd.to_datetime(scope_rows[strategy.period_column]).dt.date.isin(
                        phase_sessions
                    )
                ]
                records.append(
                    _economic_record(
                        rows,
                        scope=scope,
                        phase=phase,
                        config=config,
                        strategy_id=plan.strategy_id,
                    )
                )
        else:
            records.append(
                _economic_record(
                    scope_rows,
                    scope=scope,
                    phase="all",
                    config=config,
                    strategy_id=plan.strategy_id,
                )
            )
    return pd.DataFrame(records)


def _economic_record(
    rows: pd.DataFrame,
    *,
    scope: str,
    phase: int | str,
    config: PrimaryV2ResearchConfig,
    strategy_id: str,
) -> dict[str, object]:
    strategy = config.strategies[strategy_id]
    returns = pd.to_numeric(rows[strategy.source_target], errors="coerce").dropna()
    spy = pd.to_numeric(rows[strategy.spy_excess_target], errors="coerce").dropna()
    sector = pd.to_numeric(rows[strategy.sector_excess_target], errors="coerce").dropna()
    session_ids = rows[strategy.period_column]
    period_returns = (
        rows.assign(_net=pd.to_numeric(rows[strategy.source_target], errors="coerce"))
        .groupby(strategy.period_column, observed=True)["_net"]
        .mean()
        .dropna()
    )
    equity = (1.0 + period_returns).cumprod()
    drawdown = (
        abs(float((equity / equity.cummax() - 1.0).min()))
        if not equity.empty
        else float("nan")
    )
    gains = float(returns.loc[returns.gt(0)].sum())
    losses = abs(float(returns.loc[returns.lt(0)].sum()))
    net_interval = session_block_mean_interval(
        session_ids.reindex(returns.index),
        returns,
    )
    spy_interval = session_block_mean_interval(
        session_ids.reindex(spy.index),
        spy,
    )
    return {
        "validation_scope": scope,
        "phase": phase,
        "selected_trades": len(returns),
        "periods": len(period_returns),
        "average_net_return": float(returns.mean()) if not returns.empty else float("nan"),
        "average_excess_return_vs_spy": float(spy.mean()) if not spy.empty else float("nan"),
        "average_excess_return_vs_sector": float(sector.mean()) if not sector.empty else float("nan"),
        "average_net_return_ci_low": net_interval["low"],
        "average_excess_return_vs_spy_ci_low": spy_interval["low"],
        "win_rate": float(returns.gt(0).mean()) if not returns.empty else float("nan"),
        "profit_factor": (
            gains / losses
            if losses > 0
            else float("inf")
            if gains > 0
            else float("nan")
        ),
        "maximum_drawdown": drawdown,
        "negative_period_rate": (
            float(period_returns.lt(0).mean())
            if not period_returns.empty
            else float("nan")
        ),
    }


def _regime_evidence(
    selected: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
    config: PrimaryV2ResearchConfig,
) -> pd.DataFrame:
    strategy = config.strategies[plan.strategy_id]
    records: list[dict[str, object]] = []
    for scope in ("walk_forward", "ticker_holdout"):
        scope_rows = selected.loc[selected["validation_scope"].eq(scope)]
        for regime in config.required_market_regimes:
            rows = scope_rows.loc[scope_rows["market_regime"].eq(regime)]
            net = pd.to_numeric(rows[strategy.source_target], errors="coerce").dropna()
            spy = pd.to_numeric(rows[strategy.spy_excess_target], errors="coerce").dropna()
            records.append(
                {
                    "validation_scope": scope,
                    "market_regime": regime,
                    "selected_trades": len(net),
                    "average_net_return": (
                        float(net.mean()) if not net.empty else float("nan")
                    ),
                    "average_excess_return_vs_spy": (
                        float(spy.mean()) if not spy.empty else float("nan")
                    ),
                    "evidence_sufficient": len(net)
                    >= config.minimum_regime_selected_trades,
                }
            )
    return pd.DataFrame(records)


def _calibration_evidence(
    predictions: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    target = (
        "strategy_net_return"
        if plan.strategy_id == SWING_V2_ID
        else "path_realized_return_net_30m"
    )
    for scope, scope_rows in predictions.groupby(
        "validation_scope",
        sort=True,
    ):
        slices = [("all", "all", scope_rows)]
        slices.extend(
            ("market_regime", str(regime), rows)
            for regime, rows in scope_rows.groupby(
                "market_regime",
                sort=True,
            )
        )
        for slice_kind, slice_value, rows in slices:
            records.append(
                _calibration_record(
                    rows,
                    scope=str(scope),
                    slice_kind=slice_kind,
                    slice_value=slice_value,
                    target=target,
                    intraday=plan.strategy_id == INTRADAY_V2_ID,
                )
            )
    return pd.DataFrame(records)


def _calibration_record(
    rows: pd.DataFrame,
    *,
    scope: str,
    slice_kind: str,
    slice_value: str,
    target: str,
    intraday: bool,
) -> dict[str, object]:
        actual = pd.to_numeric(rows[target], errors="coerce")
        q10 = pd.to_numeric(rows["predicted_q10_net_return"], errors="coerce")
        q50 = pd.to_numeric(rows["predicted_q50_net_return"], errors="coerce")
        q90 = pd.to_numeric(rows["predicted_q90_net_return"], errors="coerce")
        quantile_rows = q10.notna() & q50.notna() & q90.notna() & actual.notna()
        record: dict[str, object] = {
            "validation_scope": scope,
            "calibration_slice": slice_kind,
            "slice_value": slice_value,
            "rows": len(rows),
            "quantile_rows": int(quantile_rows.sum()),
            "raw_quantile_crossing_rate": (
                float(rows.loc[quantile_rows, "raw_quantile_crossing"].mean())
                if "raw_quantile_crossing" in rows and bool(quantile_rows.any())
                else float("nan")
            ),
            "q10_observed_below_rate": (
                float(actual.loc[quantile_rows].le(q10.loc[quantile_rows]).mean())
                if bool(quantile_rows.any())
                else float("nan")
            ),
            "q50_observed_below_rate": (
                float(actual.loc[quantile_rows].le(q50.loc[quantile_rows]).mean())
                if bool(quantile_rows.any())
                else float("nan")
            ),
            "q90_observed_below_rate": (
                float(actual.loc[quantile_rows].le(q90.loc[quantile_rows]).mean())
                if bool(quantile_rows.any())
                else float("nan")
            ),
            "q10_q90_interval_coverage": (
                float(
                    actual.loc[quantile_rows]
                    .between(q10.loc[quantile_rows], q90.loc[quantile_rows])
                    .mean()
                )
                if bool(quantile_rows.any())
                else float("nan")
            ),
        }
        if intraday:
            probability_columns = [
                "predicted_stop_first_probability",
                "predicted_target_first_probability",
                "predicted_timeout_probability",
            ]
            probabilities = rows[probability_columns].apply(
                pd.to_numeric,
                errors="coerce",
            )
            complete = probabilities.notna().all(axis=1)
            if bool(complete.any()):
                labels = rows.loc[complete, "path_outcome"].astype(str)
                matrix = probabilities.loc[complete].to_numpy(float)
                record["event_probability_sum_max_error"] = float(
                    np.max(np.abs(matrix.sum(axis=1) - 1.0))
                )
                record["event_log_loss"] = float(
                    log_loss(
                        labels,
                        matrix,
                        labels=["stop_first", "target_first", "timeout"],
                    )
                )
                outcomes = labels.to_numpy(str)
                truth = np.column_stack(
                    [
                        outcomes == "stop_first",
                        outcomes == "target_first",
                        outcomes == "timeout",
                    ]
                ).astype(float)
                record["event_brier_score"] = float(
                    np.mean(np.sum((matrix - truth) ** 2, axis=1))
                )
            else:
                record["event_probability_sum_max_error"] = float("nan")
                record["event_log_loss"] = float("nan")
                record["event_brier_score"] = float("nan")
        return record


def _incremental_evidence(
    selected: pd.DataFrame,
    *,
    baseline_selected: pd.DataFrame | None,
    plan: PrimaryV2SplitPlan,
    config: PrimaryV2ResearchConfig,
) -> pd.DataFrame:
    strategy = config.strategies[plan.strategy_id]
    if baseline_selected is None:
        return pd.DataFrame()
    records: list[dict[str, object]] = []
    for scope in ("walk_forward", "ticker_holdout"):
        candidate = selected.loc[selected["validation_scope"].eq(scope)]
        baseline = baseline_selected.loc[
            baseline_selected["validation_scope"].eq(scope)
        ]
        candidate_periods = (
            candidate.assign(
                _net=pd.to_numeric(
                    candidate[strategy.source_target],
                    errors="coerce",
                ),
                _spy=pd.to_numeric(
                    candidate[strategy.spy_excess_target],
                    errors="coerce",
                ),
            )
            .groupby(strategy.period_column, observed=True)[["_net", "_spy"]]
            .mean()
        )
        baseline_periods = (
            baseline.assign(
                _net=pd.to_numeric(
                    baseline[strategy.source_target],
                    errors="coerce",
                ),
                _spy=pd.to_numeric(
                    baseline[strategy.spy_excess_target],
                    errors="coerce",
                ),
            )
            .groupby(strategy.period_column, observed=True)[["_net", "_spy"]]
            .mean()
        )
        paired = candidate_periods.join(
            baseline_periods,
            how="inner",
            lsuffix="_candidate",
            rsuffix="_baseline",
        ).dropna()
        net_delta = paired["_net_candidate"] - paired["_net_baseline"]
        spy_delta = paired["_spy_candidate"] - paired["_spy_baseline"]
        net_interval = session_block_mean_interval(
            pd.Series(paired.index, index=paired.index),
            net_delta,
        )
        spy_interval = session_block_mean_interval(
            pd.Series(paired.index, index=paired.index),
            spy_delta,
        )
        records.append(
            {
                "validation_scope": scope,
                "paired_periods": len(paired),
                "average_incremental_net_return": (
                    float(net_delta.mean())
                    if not net_delta.empty
                    else float("nan")
                ),
                "incremental_net_return_ci_low": net_interval["low"],
                "average_incremental_spy_excess": (
                    float(spy_delta.mean())
                    if not spy_delta.empty
                    else float("nan")
                ),
                "incremental_spy_excess_ci_low": spy_interval["low"],
                "comparison": "candidate_minus_exact_v1_baseline",
            }
        )
    return pd.DataFrame(records)


def _promotion_failures(
    economics: pd.DataFrame,
    regimes: pd.DataFrame,
    calibration: pd.DataFrame,
    incremental: pd.DataFrame,
    *,
    plan: PrimaryV2SplitPlan,
    spec: PrimaryV2ExperimentSpec,
    config: PrimaryV2ResearchConfig,
) -> list[str]:
    failures: list[str] = []
    for scope in ("walk_forward", "ticker_holdout"):
        rows = economics.loc[economics["validation_scope"].eq(scope)]
        if rows.empty:
            failures.append(f"{scope}: no economic evidence")
            continue
        checks = (
            ("selected trades", rows["selected_trades"].min(), config.minimum_selected_trades, "min"),
            ("average net return", rows["average_net_return"].min(), config.minimum_average_net_return, "min"),
            (
                "average SPY excess",
                rows["average_excess_return_vs_spy"].min(),
                config.minimum_average_excess_return_vs_spy,
                "min",
            ),
            (
                "average sector excess",
                rows["average_excess_return_vs_sector"].min(),
                config.minimum_average_excess_return_vs_sector,
                "min",
            ),
            (
                "net return confidence lower bound",
                rows["average_net_return_ci_low"].min(),
                config.minimum_average_net_return_ci_low,
                "min",
            ),
            (
                "SPY excess confidence lower bound",
                rows["average_excess_return_vs_spy_ci_low"].min(),
                config.minimum_average_excess_return_vs_spy_ci_low,
                "min",
            ),
            ("profit factor", rows["profit_factor"].min(), config.minimum_profit_factor, "min"),
            ("maximum drawdown", rows["maximum_drawdown"].max(), config.maximum_drawdown, "max"),
            (
                "negative period rate",
                rows["negative_period_rate"].max(),
                config.maximum_negative_period_rate,
                "max",
            ),
        )
        for label, observed, threshold, direction in checks:
            value = float(observed)
            failed = (
                not np.isfinite(value)
                or (direction == "min" and value < threshold)
                or (direction == "max" and value > threshold)
            )
            if failed:
                comparator = ">=" if direction == "min" else "<="
                failures.append(
                    f"{scope}: {label} {value:.6f} must be {comparator} {threshold:.6f}"
                )
    for row in regimes.itertuples(index=False):
        if not bool(row.evidence_sufficient):
            failures.append(
                f"{row.validation_scope}: {row.market_regime} regime has "
                f"{row.selected_trades} selected rows; "
                f"{config.minimum_regime_selected_trades} required"
            )
        elif (
            float(row.average_net_return)
            < config.minimum_regime_average_net_return
            or float(row.average_excess_return_vs_spy)
            < config.minimum_regime_average_excess_return_vs_spy
        ):
            failures.append(
                f"{row.validation_scope}: {row.market_regime} regime economics are negative"
            )
    if spec.candidate_family == "hgb_quantile_return":
        for row in calibration.itertuples(index=False):
            label = (
                f"{row.validation_scope}/{row.calibration_slice}="
                f"{row.slice_value}"
            )
            if int(row.quantile_rows) < config.minimum_calibration_rows:
                failures.append(
                    f"{label}: only {row.quantile_rows} quantile rows; "
                    f"{config.minimum_calibration_rows} required"
                )
                continue
            if (
                float(row.raw_quantile_crossing_rate)
                > config.maximum_raw_quantile_crossing_rate
            ):
                failures.append(
                    f"{label}: raw quantile crossing exceeds "
                    f"{config.maximum_raw_quantile_crossing_rate:.1%}"
                )
            observed = (
                float(row.q10_observed_below_rate),
                float(row.q50_observed_below_rate),
                float(row.q90_observed_below_rate),
            )
            errors = tuple(
                abs(value - expected)
                for value, expected in zip(
                    observed,
                    config.quantiles,
                    strict=True,
                )
            )
            if max(errors) > config.maximum_quantile_calibration_error:
                failures.append(
                    f"{label}: maximum quantile calibration error "
                    f"{max(errors):.6f} exceeds "
                    f"{config.maximum_quantile_calibration_error:.6f}"
                )
            coverage = float(row.q10_q90_interval_coverage)
            if not (
                config.minimum_q10_q90_interval_coverage
                <= coverage
                <= config.maximum_q10_q90_interval_coverage
            ):
                failures.append(
                    f"{label}: q10-q90 coverage {coverage:.6f} is outside "
                    f"[{config.minimum_q10_q90_interval_coverage:.6f}, "
                    f"{config.maximum_q10_q90_interval_coverage:.6f}]"
                )
    if plan.strategy_id == INTRADAY_V2_ID:
        for row in calibration.itertuples(index=False):
            label = (
                f"{row.validation_scope}/{row.calibration_slice}="
                f"{row.slice_value}"
            )
            if int(row.rows) < config.minimum_calibration_rows:
                failures.append(
                    f"{label}: only {row.rows} event-calibration rows; "
                    f"{config.minimum_calibration_rows} required"
                )
            elif not np.isfinite(float(row.event_log_loss)):
                failures.append(
                    f"{label}: competing-risk calibration is missing"
                )
            elif float(row.event_probability_sum_max_error) > 1e-6:
                failures.append(
                    f"{label}: competing-risk probabilities do not sum to one"
                )
            elif (
                float(row.event_log_loss)
                > config.maximum_event_log_loss
            ):
                failures.append(
                    f"{label}: event log loss {float(row.event_log_loss):.6f} "
                    f"exceeds {config.maximum_event_log_loss:.6f}"
                )
            elif (
                float(row.event_brier_score)
                > config.maximum_event_brier_score
            ):
                failures.append(
                    f"{label}: event Brier score "
                    f"{float(row.event_brier_score):.6f} exceeds "
                    f"{config.maximum_event_brier_score:.6f}"
                )
    baseline_families = {
        "deterministic_v1_baseline",
        "multinomial_v1_baseline",
    }
    if spec.candidate_family not in baseline_families:
        if incremental.empty:
            failures.append("paired V1 incremental evidence is missing")
        else:
            for row in incremental.itertuples(index=False):
                if int(row.paired_periods) < 100:
                    failures.append(
                        f"{row.validation_scope}: only {row.paired_periods} paired "
                        "periods against the exact V1 baseline"
                    )
                net_low = float(row.incremental_net_return_ci_low)
                spy_low = float(row.incremental_spy_excess_ci_low)
                if (
                    not np.isfinite(net_low)
                    or net_low
                    <= config.minimum_incremental_net_return_ci_low
                ):
                    failures.append(
                        f"{row.validation_scope}: incremental net-return confidence "
                        f"lower bound {net_low:.6f} must be > "
                        f"{config.minimum_incremental_net_return_ci_low:.6f}"
                    )
                if (
                    not np.isfinite(spy_low)
                    or spy_low
                    <= config.minimum_incremental_spy_excess_ci_low
                ):
                    failures.append(
                        f"{row.validation_scope}: incremental SPY-excess confidence "
                        f"lower bound {spy_low:.6f} must be > "
                        f"{config.minimum_incremental_spy_excess_ci_low:.6f}"
                    )
    return sorted(set(failures))


def validate_primary_v2_source_rows(
    data: pd.DataFrame,
    *,
    strategy_id: str,
    config: PrimaryV2ResearchConfig,
) -> None:
    strategy = config.strategies[strategy_id]
    missing = sorted(strategy.required_source_columns.difference(data.columns))
    if missing:
        raise DataReadinessError(
            f"{strategy_id} source dataset is missing: " + ", ".join(missing)
        )
    if data[strategy.row_id_column].astype(str).duplicated().any():
        raise DataReadinessError(f"{strategy_id} source row identities are duplicated")
    for column in (
        strategy.source_target,
        strategy.spy_excess_target,
        strategy.sector_excess_target,
        strategy.mfe_target,
        strategy.mae_target,
    ):
        values = pd.to_numeric(data[column], errors="coerce")
        if values.isna().any() or not np.isfinite(values.to_numpy(float)).all():
            raise DataReadinessError(f"{strategy_id} contains non-finite {column}")
    decision = pd.to_datetime(data["decision_time_utc"], utc=True, errors="coerce")
    entry = pd.to_datetime(data["entry_time_utc"], utc=True, errors="coerce")
    exit_time = pd.to_datetime(data["exit_time_utc"], utc=True, errors="coerce")
    label = pd.to_datetime(data["label_available_at_utc"], utc=True, errors="coerce")
    invalid = (
        decision.isna()
        | entry.isna()
        | exit_time.isna()
        | label.isna()
        | entry.lt(decision)
        | exit_time.le(entry)
        | label.lt(exit_time)
    )
    if bool(invalid.any()):
        raise DataReadinessError(f"{strategy_id} contains invalid causal timestamps")
    if strategy_id == INTRADAY_V2_ID:
        if set(data["price_feed"].astype(str).str.lower().unique()) != {"sip"}:
            raise DataReadinessError("intraday V2 requires the SIP price feed")
        if set(data["adjustment"].astype(str).str.lower().unique()) != {"all"}:
            raise DataReadinessError("intraday V2 requires all corporate-action adjustments")
        same_session = (
            entry.dt.tz_convert("America/New_York").dt.date
            == exit_time.dt.tz_convert("America/New_York").dt.date
        )
        if not bool(same_session.all()):
            raise DataReadinessError("intraday V2 contains an overnight label path")


def _require_matching_split_contract(
    config: PrimaryV2ResearchConfig,
    *,
    n_splits: int,
    holdout_fraction: float,
) -> None:
    if config.n_splits != n_splits:
        raise DataReadinessError("V2 and source split counts differ")
    if not np.isclose(config.ticker_holdout_fraction, holdout_fraction):
        raise DataReadinessError("V2 and source ticker-holdout fractions differ")


def _feature_frame(frame: pd.DataFrame, features: tuple[str, ...]) -> pd.DataFrame:
    return frame.loc[:, list(features)].apply(pd.to_numeric, errors="coerce").astype("float32")


def _new_hgb_regressor(
    config: PrimaryV2ResearchConfig,
    *,
    quantile: float | None = None,
) -> HistGradientBoostingRegressor:
    return HistGradientBoostingRegressor(
        loss="quantile" if quantile is not None else "squared_error",
        quantile=quantile,
        max_iter=config.hgb_max_iter,
        learning_rate=config.hgb_learning_rate,
        l2_regularization=config.hgb_l2_regularization,
        random_state=config.random_seed,
    )


def _derive_market_regime(frame: pd.DataFrame) -> pd.Series:
    zeros = pd.Series(0.0, index=frame.index)
    risk_off = pd.to_numeric(
        frame["regime_risk_off"] if "regime_risk_off" in frame else zeros,
        errors="coerce",
    ).fillna(0)
    risk_on = pd.to_numeric(
        frame["regime_risk_on"] if "regime_risk_on" in frame else zeros,
        errors="coerce",
    ).fillna(0)
    return pd.Series(
        np.select(
            [risk_off.ge(0.5), risk_on.ge(0.5)],
            ["risk_off", "risk_on"],
            default="neutral",
        ),
        index=frame.index,
        dtype="object",
    )


def _assert_memory(config: PrimaryV2ResearchConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
