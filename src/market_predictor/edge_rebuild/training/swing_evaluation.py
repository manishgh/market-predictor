"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""

from __future__ import annotations

from collections.abc import Mapping
from datetime import date
from typing import Any, Final

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score

from market_predictor.edge_rebuild.outcome_diagnostics import (
    binary_outcome_diagnostic,
    label_permutation_control,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_EXCESS_RETURN_COLUMNS,
    MANAGED_PATH_NET_RETURN_COLUMNS,
    MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
    SWING_FEATURE_PROFILE,
)
from market_predictor.edge_rebuild.swing_selection import (
    EFFECTIVE_SECTOR_WEIGHT_COLUMN,
    select_constrained_swing_portfolio,
)
from market_predictor.edge_rebuild.training.data_io import _security_holdout_mask
from market_predictor.edge_rebuild.training.economics import (
    _daily_position_ledger,
    _economic_gate,
    _moving_block_bootstrap_mean_interval,
    _session_bootstrap,
    _session_economic_blocks,
    _stability_breakdown,
    _stability_summary,
    _year_breakdown,
)
from market_predictor.edge_rebuild.training.evaluation import (
    _calibration_bins,
    _expected_calibration_error,
)
from market_predictor.edge_rebuild.training.lgbm_models import _fit_candidate, _predict_probability
from market_predictor.edge_rebuild.training.swing_types import (
    CandidateSpec,
    SwingProfileData,
    SwingTrainingConfig,
    _iso,
)
from market_predictor.edge_rebuild.training.utils import _finite, _mapping
from market_predictor.edge_rebuild.training.walk_forward import (
    WalkForwardFold,
    _assert_label_purge,
)
from market_predictor.resources import (
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

TRAINING_SCHEMA: Final = "edge_rebuild.swing_training.v5"
MODEL_SCHEMA: Final = "edge_rebuild.swing_candidate.v5"
EVALUATION_SCHEMA: Final = "edge_rebuild.swing_evaluation.v7"
MODEL_CARD_SCHEMA: Final = "edge_rebuild.swing_model_card.v7"
OUTPUT_AUTHORITY_SCHEMA: Final = "edge_rebuild.swing_candidate_authority.v5"
SWING_BASELINE_BUNDLE_PREFIX: Final = "swing_baseline_bundle."
DECISION_START_DATE: Final = date(2019, 7, 9)
HORIZON_SESSIONS: Final = 10
ALLOWED_PROFILES: Final = (
    SWING_FEATURE_PROFILE,
)
# The learned families, per profile and per (rate, depth) point. `dual_hurdle`
# was dropped: it scored 0.452-0.462 AUC on the v12 run -- below chance -- had no
# test covering it, and its four slots pushed the grid past the contract's
# six-candidate experiment budget.
_XGB_GRID: Final = (
    ("xgbranker", "xgboost_ranker"),
    ("xgbregressor", "xgboost_regressor"),
)
_XGB_FAMILIES: Final = len(_XGB_GRID)
_MANIFEST_NAME: Final = "_manifest.json"
_AUTHORITY_NAME: Final = "_authority.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_TEXT_COLUMNS: Final = (
    "decision_id",
    "decision_group_id",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "market_regime",
)












































def _evaluation_columns() -> tuple[str, ...]:
    return tuple(
        dict.fromkeys(
            (
                "decision_id",
                "decision_group_id",
                "ticker",
                "security_id",
                "sector",
                "market_regime",
                "session_date_et",
                "decision_time_utc",
                "barrier_exit_session_date_et",
                "barrier_holding_sessions",
                "target",
                "barrier_gross_return",
                "barrier_cost",
                "barrier_net_return",
                "future_net_return_10d",
                "future_excess_return_10d_vs_spy",
                "future_excess_return_10d_vs_qqq",
                "future_excess_return_10d_vs_sector",
                *MANAGED_EXCESS_RETURN_COLUMNS,
                *MANAGED_PATH_SESSION_ORDINAL_COLUMNS,
                *MANAGED_PATH_NET_RETURN_COLUMNS,
            )
        )
    )


def _evaluate_validation_candidate(
    spec: CandidateSpec,
    profile_data: SwingProfileData,
    folds: tuple[WalkForwardFold, ...],
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    predictions: dict[str, list[pd.DataFrame]] = {
        "temporal_generalization_full_pit_cross_section": [],
        "unseen_security_generalization_stable_20pct": [],
    }
    fold_records: list[dict[str, Any]] = []
    holdout = _security_holdout_mask(profile_data.frame, strategy_contract)
    for fold in folds:
        train_columns = list(dict.fromkeys((
            "decision_id",
            "session_date_et",
            "decision_time_utc",
            "decision_group_id",
            "label_available_at_utc",
            "target",
            "barrier_net_return",
            "relevance_score",
            "ranking_reliability_weight",
            *spec.feature_columns,
        )))
        validation_columns = list(dict.fromkeys((
            *_evaluation_columns(),
            *spec.feature_columns,
        )))
        scope_records: dict[str, Any] = {}
        for scope, train_mask, validation_mask in (
            (
                "temporal_generalization_full_pit_cross_section",
                profile_data.frame["session_date_et"].isin(fold.train_sessions),
                profile_data.frame["session_date_et"].isin(
                    fold.validation_sessions
                ),
            ),
            (
                "unseen_security_generalization_stable_20pct",
                profile_data.frame["session_date_et"].isin(fold.train_sessions)
                & ~holdout,
                profile_data.frame["session_date_et"].isin(
                    fold.validation_sessions
                )
                & holdout,
            ),
        ):
            train = profile_data.frame.loc[train_mask, train_columns]
            validation = profile_data.frame.loc[validation_mask, validation_columns]
            _assert_label_purge(
                train,
                validation,
                f"{scope} validation fold {fold.fold}",
            )
            fitted = _fit_candidate(spec, train, config)
            probability = _predict_probability(
                fitted,
                validation,
                spec.feature_columns,
            )
            scored = validation.loc[:, list(_evaluation_columns())].copy()
            scored["__probability"] = probability
            predictions[scope].append(scored)
            scope_records[scope] = {
                "train_rows": len(train),
                "validation_rows": len(validation),
                "max_train_label_available_at_utc": _iso(
                    train["label_available_at_utc"].max()
                ),
                "min_validation_decision_time_utc": _iso(
                    validation["decision_time_utc"].min()
                ),
                "fit_sessions": fitted.fit_sessions,
                "calibration_sessions": fitted.calibration_sessions,
                "calibration_cutoff_utc": fitted.calibration_cutoff_utc,
                "target_prevalence": float(validation["target"].mean()),
                "probability_distribution": _probability_distribution(probability),
            }
            del fitted, train, validation, scored
            release_process_memory()
        fold_records.append(
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "purge_sessions": len(fold.purge_sessions),
                "embargo_sessions": len(fold.embargo_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "scopes": scope_records,
            }
        )
    pooled = {
        scope: pd.concat(parts, ignore_index=True)
        for scope, parts in predictions.items()
    }
    del predictions
    validation_calendar = tuple(
        session
        for fold in folds
        for session in fold.validation_sessions
    )
    threshold_records: list[dict[str, Any]] = []
    for threshold in config.probability_thresholds:
        try:
            scope_metrics = {
                scope: _evaluation_metrics(
                    frame,
                    frame["__probability"].to_numpy(dtype="float64"),
                    threshold=threshold,
                    config=config,
                    strategy_contract=strategy_contract,
                    session_calendar=validation_calendar,
                )
                for scope, frame in pooled.items()
            }
            passed = _validation_scopes_pass_economic_gates(scope_metrics)
            threshold_records.append({
                "probability_threshold": threshold,
                "eligible": passed,
                "reason": (
                    None
                    if passed
                    else "one or more frozen validation scopes failed economic gates"
                ),
                "scopes": scope_metrics,
            })
        except DataReadinessError as exc:
            threshold_records.append(
                {
                    "probability_threshold": threshold,
                    "eligible": False,
                    "reason": str(exc),
                }
            )
    eligible = [record for record in threshold_records if record["eligible"]]
    diagnostic = [record for record in threshold_records if "scopes" in record]
    if not diagnostic:
        return {
            "candidate_id": spec.candidate_id,
            "ablation_profile": spec.profile,
            "feature_group": spec.feature_group,
            "feature_columns": list(spec.feature_columns),
            "estimator_family": spec.estimator_family,
            "hyperparameters": dict(spec.hyperparameters),
            "folds": fold_records,
            "thresholds": threshold_records,
            "candidate_eligible": False,
            "reason": "no threshold selected enough validation trades",
        }
    selected = max(eligible or diagnostic, key=_threshold_selection_key)
    metrics = _mapping(selected.get("scopes"), "selected threshold scopes")
    for threshold_record in threshold_records:
        if threshold_record is selected:
            continue
        raw_scopes = threshold_record.get("scopes")
        if isinstance(raw_scopes, dict):
            for raw_metrics in raw_scopes.values():
                if isinstance(raw_metrics, dict):
                    raw_metrics.pop("paired_session_blocks", None)
    record: dict[str, Any] = {
        "candidate_id": spec.candidate_id,
        "ablation_profile": spec.profile,
        "feature_group": spec.feature_group,
        "feature_columns": list(spec.feature_columns),
        "estimator_family": spec.estimator_family,
        "hyperparameters": dict(spec.hyperparameters),
        "folds": fold_records,
        "thresholds": threshold_records,
        "selected_probability_threshold": float(selected["probability_threshold"]),
        "selected_validation_metrics": metrics,
        "candidate_eligible": bool(eligible),
    }
    if eligible:
        record["selection_key"] = list(_selection_key(record))
    return record


def _validation_scopes_pass_economic_gates(
    scope_metrics: Mapping[str, Mapping[str, Any]],
) -> bool:
    if not scope_metrics:
        return False
    return all(
        bool(_mapping(metrics.get("economic_gate"), "economic gate")["passed"])
        for metrics in scope_metrics.values()
    )


def _probability_distribution(probability: np.ndarray) -> dict[str, float]:
    if probability.ndim != 1 or probability.size < 1 or not np.isfinite(probability).all():
        raise DataReadinessError("probability diagnostics require one finite vector")
    quantiles = np.quantile(probability, [0.01, 0.10, 0.50, 0.90, 0.99])
    return {
        "minimum": float(probability.min()),
        "p01": float(quantiles[0]),
        "p10": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p90": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "maximum": float(probability.max()),
        "mean": float(probability.mean()),
    }










def _evaluation_metrics(
    frame: pd.DataFrame,
    probability: np.ndarray,
    *,
    threshold: float,
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
    session_calendar: tuple[str, ...],
) -> dict[str, Any]:
    if len(frame) != len(probability) or not np.isfinite(probability).all():
        raise DataReadinessError("prediction length or finiteness is invalid")
    scored = frame.copy()
    scored["__probability"] = probability
    candidates = scored.loc[scored["__probability"].ge(threshold)].copy()
    selected = select_constrained_swing_portfolio(
        candidates,
        maximum_trades=config.maximum_trades_per_decision,
        target_maximum_sector_weight=strategy_contract.swing.target_maximum_sector_weight,
        hard_maximum_sector_weight=strategy_contract.swing.hard_maximum_sector_weight,
        minimum_distinct_sectors=strategy_contract.swing.minimum_distinct_sectors_for_selection,
    )
    if selected.empty or selected["session_date_et"].nunique() < 2:
        raise DataReadinessError("threshold selects fewer than two independent sessions")
    selected = selected.sort_values(
        ["decision_time_utc", "decision_group_id", "security_id"], kind="stable"
    )
    target = scored["target"].to_numpy(dtype="int8", copy=False)
    has_two_classes = np.unique(target).size == 2
    base_rate = float(target.mean())
    selected_rate = float(selected["target"].mean())
    ledger = _daily_position_ledger(
        selected,
        config,
        session_calendar=session_calendar,
    )
    stress_ledger = _daily_position_ledger(
        selected,
        config,
        session_calendar=session_calendar,
        additional_round_trip_cost=(
            (strategy_contract.stress.cost_multiplier - 1.0)
            * config.expected_round_trip_cost_bps
            / 10_000.0
        ),
    )
    positive = selected.loc[selected["barrier_net_return"].gt(0), "barrier_net_return"].sum()
    negative = selected.loc[selected["barrier_net_return"].lt(0), "barrier_net_return"].sum()
    calibration_bins = _calibration_bins(target, probability)
    bootstrap = _session_bootstrap(
        selected,
        config,
        session_calendar=session_calendar,
    )
    bootstrap["portfolio_daily_return"] = _moving_block_bootstrap_mean_interval(
        np.asarray(ledger["daily_returns"], dtype="float64"),
        config.bootstrap_samples,
        config.bootstrap_block_sessions,
        config.random_seed + 10_001,
    )
    bootstrap["double_cost_portfolio_daily_return"] = (
        _moving_block_bootstrap_mean_interval(
            np.asarray(stress_ledger["daily_returns"], dtype="float64"),
            config.bootstrap_samples,
            config.bootstrap_block_sessions,
            config.random_seed + 10_002,
        )
    )
    metrics: dict[str, Any] = {
        "rows": len(scored),
        "sessions": int(scored["session_date_et"].nunique()),
        "securities": int(scored["security_id"].nunique()),
        "probability_threshold": threshold,
        "roc_auc": float(roc_auc_score(target, probability)) if has_two_classes else None,
        "pr_auc": float(average_precision_score(target, probability)) if has_two_classes else None,
        "auc_is_diagnostic_only": True,
        "binary_outcome_diagnostics": {
            "estimator_target_top_sector_quantile": binary_outcome_diagnostic(
                target,
                probability,
                definition=(
                    "published rank_label is top within the point-in-time sector "
                    "decision cohort"
                ),
            ),
            "managed_net_return_positive_after_costs": binary_outcome_diagnostic(
                scored["barrier_net_return"].gt(0.0),
                probability,
                definition="managed target/stop/timeout net return after costs is positive",
            ),
            "ten_session_net_return_positive_after_costs": binary_outcome_diagnostic(
                scored["future_net_return_10d"].gt(0.0),
                probability,
                definition="exact ten-session net return after costs is positive",
            ),
            "ten_session_spy_excess_positive": binary_outcome_diagnostic(
                scored["future_excess_return_10d_vs_spy"].gt(0.0),
                probability,
                definition="exact ten-session net return exceeds SPY over the same interval",
            ),
            "ten_session_qqq_excess_positive": binary_outcome_diagnostic(
                scored["future_excess_return_10d_vs_qqq"].gt(0.0),
                probability,
                definition="exact ten-session net return exceeds QQQ over the same interval",
            ),
            "ten_session_sector_excess_positive": binary_outcome_diagnostic(
                scored["future_excess_return_10d_vs_sector"].gt(0.0),
                probability,
                definition=(
                    "exact ten-session net return exceeds the point-in-time sector ETF "
                    "over the same interval"
                ),
            ),
        },
        "negative_controls": {
            "label_permutation": label_permutation_control(
                target,
                probability,
                random_seed=config.random_seed + 31_337,
            )
        },
        "brier_score": float(brier_score_loss(target, probability)),
        "expected_calibration_error": _expected_calibration_error(calibration_bins, len(scored)),
        "calibration_bins": calibration_bins,
        "base_positive_rate": base_rate,
        "selected_positive_rate": selected_rate,
        "selected_probability_lift": selected_rate / base_rate if base_rate > 0 else None,
        "selected_trade_count": len(selected),
        "selected_decision_count": int(selected["decision_group_id"].nunique()),
        "selected_average_managed_gross_return": float(selected["barrier_gross_return"].mean()),
        "selected_average_managed_net_return": float(
            selected["barrier_net_return"].mean()
        ),
        "selected_win_rate_after_costs": float(selected["barrier_net_return"].gt(0).mean()),
        "calendar_average_managed_net_return": float(
            bootstrap["calendar_average_managed_net_return"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_spy_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_spy_excess"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_qqq_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_qqq_excess"]["estimate"]
        ),
        "calendar_average_managed_exit_session_close_sector_excess": float(
            bootstrap["calendar_average_managed_exit_session_close_sector_excess"]["estimate"]
        ),
        "selected_average_managed_exit_session_close_spy_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_spy"].mean()
        ),
        "selected_average_managed_exit_session_close_qqq_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_qqq"].mean()
        ),
        "selected_average_managed_exit_session_close_sector_excess": float(
            selected["approx_managed_exit_session_close_excess_vs_sector"].mean()
        ),
        "managed_exit_benchmark_timestamp_policy": "entry_open_to_exit_session_close",
        "profit_factor_after_costs": float(positive / abs(negative)) if negative < 0 else None,
        "turnover": ledger["average_daily_turnover"],
        "daily_mark_to_market_max_drawdown_after_costs": ledger["max_drawdown"],
        "daily_mark_to_market_compounded_return": ledger["compounded_return"],
        "portfolio_daily_average_return": float(
            bootstrap["portfolio_daily_return"]["estimate"]
        ),
        "double_cost_portfolio_daily_average_return": float(
            bootstrap["double_cost_portfolio_daily_return"]["estimate"]
        ),
        "drawdown_has_daily_mark_to_market": True,
        "maximum_observed_sector_weight": ledger["maximum_sector_weight"],
        "target_maximum_sector_weight": (
            strategy_contract.swing.target_maximum_sector_weight
        ),
        "hard_maximum_sector_weight": (
            strategy_contract.swing.hard_maximum_sector_weight
        ),
        "minimum_distinct_sectors_for_selection": (
            strategy_contract.swing.minimum_distinct_sectors_for_selection
        ),
        "maximum_effective_sector_weight_limit": float(
            selected[EFFECTIVE_SECTOR_WEIGHT_COLUMN].max()
        ),
        "frozen_round_trip_cost_bps": config.expected_round_trip_cost_bps,
        "cost_deduction_count": 1,
        "by_regime": _stability_breakdown(selected, "market_regime"),
        "by_sector": _stability_breakdown(selected, "sector"),
        "by_year": _year_breakdown(selected),
        "moving_block_bootstrap_95_ci": bootstrap,
        "paired_session_blocks": _session_economic_blocks(
            selected,
            session_calendar=session_calendar,
        ),
    }
    metrics["economic_gate"] = _economic_gate(metrics, strategy_contract)
    metrics["regime_stability"] = _stability_summary(metrics["by_regime"])
    metrics["sector_stability"] = _stability_summary(metrics["by_sector"])
    return metrics


def _threshold_selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = _mapping(record.get("scopes"), "threshold scopes")
    keys = [
        _scope_economic_key(_mapping(metrics, f"{scope} metrics"))
        for scope, metrics in sorted(scopes.items())
    ]
    if not keys:
        raise DataReadinessError("threshold has no validation scopes")
    return tuple(min(key[index] for key in keys) for index in range(len(keys[0]))) + (
        -float(record["probability_threshold"]),
    )


def _scope_economic_key(metrics: Mapping[str, Any]) -> tuple[float, ...]:
    bootstrap = _mapping(metrics.get("moving_block_bootstrap_95_ci"), "bootstrap")
    portfolio_ci = _mapping(
        bootstrap.get("portfolio_daily_return"), "portfolio daily CI"
    )
    spy_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_spy_excess"),
        "managed SPY CI",
    )
    qqq_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_qqq_excess"),
        "managed QQQ CI",
    )
    sector_ci = _mapping(
        bootstrap.get("calendar_average_managed_exit_session_close_sector_excess"),
        "managed sector CI",
    )
    return (
        min(_finite(spy_ci, "low"), _finite(qqq_ci, "low"), _finite(sector_ci, "low")),
        _finite(portfolio_ci, "low"),
        min(
            _finite(metrics, "calendar_average_managed_exit_session_close_spy_excess"),
            _finite(metrics, "calendar_average_managed_exit_session_close_qqq_excess"),
            _finite(metrics, "calendar_average_managed_exit_session_close_sector_excess"),
        ),
        _finite(metrics, "selected_average_managed_net_return"),
        -_finite(metrics, "daily_mark_to_market_max_drawdown_after_costs"),
        -_finite(metrics, "turnover"),
    )


def _selection_key(record: Mapping[str, Any]) -> tuple[float, ...]:
    scopes = _mapping(record.get("selected_validation_metrics"), "validation scopes")
    threshold_record = {
        "probability_threshold": record.get("selected_probability_threshold"),
        "scopes": scopes,
    }
    economic = _threshold_selection_key(threshold_record)
    # Prefer the simpler logistic candidate only after all economic and risk
    # criteria tie. AUC remains diagnostic and cannot drive candidate choice.
    simplicity = 1.0 if record.get("estimator_family") == "logistic" else 0.0
    raw_columns = record.get("feature_columns")
    feature_count = len(raw_columns) if isinstance(raw_columns, list) else 10**9
    return (*economic, simplicity, -float(feature_count))


























