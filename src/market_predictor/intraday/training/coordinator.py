from __future__ import annotations

from market_predictor.intraday.evaluation.economics import _position_ledger
from market_predictor.intraday.training.config import IntradayDevelopmentConfig, _CandidateSpec
from market_predictor.intraday.evaluation.gates import (
    _audit_policy_choice,
    _evaluate_spec,
    _profile_mask,
    _selection_key,
    baseline_profile,
)
from market_predictor.intraday.training.io import (
    _dataset_identity,
    _future_data_contract,
    _gate_contract,
    _guard_memory,
    _json_sha256,
    _parse_date,
    _publish_development,
    _require_output_isolated,
    _strict_bool,
    load_complete_intraday_development_output,
)
from market_predictor.intraday.training.models import _fit_pair, _predict_pair
from market_predictor.intraday.training.validation import _Fold, _security_set_sha256, _stable_security_holdout, _walk_forward_folds

"""Development-only, cost-aware intraday model training and evaluation."""

import gc
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Final

import numpy as np
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.training.event_training import (
    DIRECTIONAL_EVENT_SUBTYPES,
    filter_to_research_event_cohort,
    load_intraday_research_event_cohort,
)
from market_predictor.intraday.training.training import (
    MODEL_FEATURE_COLUMNS,
    PublishedIntradayDataset,
    load_published_intraday_dataset,
)
from market_predictor.resources import (
    memory_audit,
    release_process_memory,
)

MODEL_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_candidate.v1"
EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_evaluation.v1"
AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_baseline_authority.v1"
FUTURE_EVALUATION_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_evaluation.v1"
FUTURE_AUTHORITY_SCHEMA_VERSION: Final = "edge_rebuild.intraday_bar_future_authority.v1"
_AUTHORITY_NAME: Final = "_authority.json"
_MANIFEST_NAME: Final = "_manifest.json"
_EVALUATION_NAME: Final = "evaluation.json"
_MODEL_CARD_NAME: Final = "model_card.json"
_CANDIDATE_NAME: Final = "candidate.joblib"
_FUTURE_EVALUATION_NAME: Final = "future_evaluation.json"
_POSITION_LEDGER_NAME: Final = "position_ledger.parquet"
_DAILY_LEDGER_NAME: Final = "daily_ledger.parquet"
_VALIDATION_PREDICTIONS_NAME: Final = "validation_predictions.parquet"


@dataclass(frozen=True, slots=True)
class DevelopmentTrainingResult:
    output_directory: Path
    status: str
    selected_candidate_id: str | None
    evaluation: Mapping[str, Any]



def train_intraday_development_candidate(
    dataset_authority_directory: Path,
    output_directory: Path,
    *,
    hypothesis: str,
    config: IntradayDevelopmentConfig | None = None,
    research_event_preflight_directory: Path | None = None,
    research_event_subtype: str | None = None,
) -> DevelopmentTrainingResult:
    """Train one technical or event-confirmed hypothesis without future data."""

    policy = config or IntradayDevelopmentConfig()
    profile = baseline_profile(hypothesis, policy)
    _guard_memory(policy, "intraday development start", peak=False)
    immutable_inputs = [dataset_authority_directory]
    if research_event_subtype is not None and research_event_preflight_directory is None:
        raise DataReadinessError("intraday event subtype requires a historical event preflight")
    if research_event_subtype is not None and research_event_subtype not in DIRECTIONAL_EVENT_SUBTYPES:
        raise DataReadinessError(f"unsupported intraday analyst-event subtype: {research_event_subtype}")
    if research_event_preflight_directory is not None:
        immutable_inputs.append(research_event_preflight_directory)
    _require_output_isolated(output_directory, *immutable_inputs)
    event_cohort = None
    if research_event_preflight_directory is not None:
        event_cohort = load_intraday_research_event_cohort(
            research_event_preflight_directory,
            event_subtype=research_event_subtype,
        )
        release_process_memory()
    published = load_published_intraday_dataset(dataset_authority_directory)
    data = _validate_development_frame(published, policy)
    if event_cohort is not None:
        data = filter_to_research_event_cohort(data, event_cohort)
    data = data.loc[_profile_mask(data, profile)].reset_index(drop=True)
    if len(data) < policy.minimum_rows or data["security_id"].nunique() < policy.minimum_securities:
        raise DataReadinessError(f"{profile.profile_id} population is too small for governed training")
    sessions = _ordered_sessions(data)
    folds = _walk_forward_folds(data, sessions, policy)
    security_holdout = _stable_security_holdout(data, policy.security_holdout_fraction)
    # Keep one compact feature matrix; candidate fits are sequential.
    features_full = data[list(MODEL_FEATURE_COLUMNS)].to_numpy(dtype="float32", copy=True)
    opportunity_target = data["net_return"].to_numpy(dtype="float64", copy=True)
    downside_target = data["stop_hit"].to_numpy(dtype="int8", copy=True)
    data.drop(columns=list(MODEL_FEATURE_COLUMNS), inplace=True)

    frozen_cost_bps = published.frozen_round_trip_cost_bps
    dataset_identity_val = _dataset_identity(published)
    if event_cohort is None:
        model_family = "intraday_technical"
    elif research_event_subtype is None:
        model_family = "intraday_event_confirmed_research"
    else:
        model_family = f"intraday_{research_event_subtype}_confirmed_research"
    if event_cohort is not None:
        dataset_identity_val["research_event_cohort"] = event_cohort.identity
    gc.collect()

    validation_records: list[dict[str, Any]] = []
    retained_predictions: dict[str, pd.DataFrame] = {}
    for spec in _candidate_specs(policy):
        scored, fold_records = _walk_forward_predictions(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            folds,
            security_holdout,
            policy,
        )
        record = _evaluate_spec(spec, scored, fold_records, policy, frozen_cost_bps)
        validation_records.append(record)
        selection_passed = [r for r in validation_records if bool(r["selection_passed"])]
        current_winner = max(selection_passed, key=_selection_key) if selection_passed else None
        current_selected = current_winner if current_winner is not None and bool(current_winner["validation_passed"]) else None
        current_audit_candidate, _, _, _ = _audit_policy_choice(
            validation_records,
            current_selected,
            preferred=current_winner,
        )

        retained_predictions[spec.candidate_id] = scored
        keys_to_keep = {current_audit_candidate}
        if current_selected is not None:
            keys_to_keep.add(str(current_selected["candidate_id"]))

        for k in list(retained_predictions.keys()):
            if k not in keys_to_keep:
                del retained_predictions[k]

        gc.collect()

        _guard_memory(policy, f"{spec.candidate_id} validation", peak=True)

    selection_passed = [record for record in validation_records if bool(record["selection_passed"])]
    selection_winner = max(selection_passed, key=_selection_key) if selection_passed else None
    selected = selection_winner if selection_winner is not None and bool(selection_winner["validation_passed"]) else None
    status = "candidate" if selected is not None else "no_candidate"
    selected_id = str(selected["candidate_id"]) if selected is not None else None
    config_payload = asdict(policy)
    config_hash = _json_sha256(config_payload)
    evaluation: dict[str, Any] = {
        "schema_version": EVALUATION_SCHEMA_VERSION,
        "status": status,
        "model_family": model_family,
        "promotion_permitted": False,
        "selection_basis": "development_walk_forward_validation_only",
        "objective": "expected_net_return_with_calibrated_stop_risk_after_frozen_cost",
        "baseline_profile": asdict(profile),
        "baseline_profile_sha256": profile.sha256(),
        "target_hit_used_as_training_target": False,
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "raw_ndcg_reported": False,
        "future_holdout_opened": False,
        "test_access_count": 0,
        "future_holdout_start_date": policy.future_holdout_start_date,
        "development_end_date": policy.development_end_date,
        "dataset": dataset_identity_val,
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "training_config": config_payload,
        "training_config_sha256": config_hash,
        "validation_candidates": validation_records,
        "selected_candidate_id": selected_id,
        "gates": _gate_contract(policy),
        "security_holdout": {
            "fraction": policy.security_holdout_fraction,
            "security_count": len(security_holdout),
            "security_set_sha256": _security_set_sha256(security_holdout),
        },
        "future_data_contract": _future_data_contract(policy),
        "memory": memory_audit(
            hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib,
        ).to_record(),
    }
    audit_candidate, audit_threshold, audit_stop_threshold, audit_passed = _audit_policy_choice(
        validation_records,
        selected,
        preferred=selection_winner,
    )
    audit_ledger = _position_ledger(
        retained_predictions[audit_candidate],
        audit_threshold,
        audit_stop_threshold,
        frozen_cost_bps,
        policy,
    )
    evaluation["auditable_policy_ledger"] = {
        "candidate_id": audit_candidate,
        "threshold_bps": audit_threshold,
        "maximum_stop_probability": audit_stop_threshold,
        "validation_passed": audit_passed,
        "selection_status": "selected_candidate" if audit_passed else "best_failed_diagnostic_only",
        "position_ledger_path": _POSITION_LEDGER_NAME,
        "daily_ledger_path": _DAILY_LEDGER_NAME,
    }
    model_card: dict[str, Any] = {
        "schema_version": "edge_rebuild.intraday_bar_baseline_model_card.v1",
        "status": status,
        "model_family": model_family,
        "promotion_permitted": False,
        "candidate_id": selected_id,
        "horizon_minutes": 30,
        "baseline_profile": asdict(profile),
        "baseline_profile_sha256": profile.sha256(),
        "opportunity_training_target": "net_return",
        "downside_training_target": "stop_hit",
        "selection_target": "capital_weighted_net_economics",
        "feature_columns": list(MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": published.ordered_feature_sha256,
        "development_rows": int(len(data)),
        "development_sessions": int(len(sessions)),
        "development_securities": int(data["security_id"].nunique()),
        "future_holdout_opened": False,
        "future_data_contract": _future_data_contract(policy),
        "limitations": [
            "candidate is development-only and cannot be promoted without a separately collected future holdout",
            "event-time equity marks open positions at their frozen stop until exact recorded exit",
            (
                "historical catalyst timestamps are provider-publication proxies; catalyst is a research-only confirmation filter"
                if event_cohort is not None
                else "catalyst and trade/quote microstructure are outside this technical estimator contract"
            ),
        ],
    }
    candidate: dict[str, Any] | None = None
    if selected is not None:
        spec = next(item for item in _candidate_specs(policy) if item.candidate_id == selected_id)
        fitted = _fit_pair(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            sessions,
            policy,
            excluded_securities=security_holdout,
        )
        candidate = {
            "schema_version": MODEL_SCHEMA_VERSION,
            "status": "candidate",
            "model_family": model_family,
            "promotion_permitted": False,
            "validation_passed": True,
            "candidate_id": selected_id,
            "baseline_profile": asdict(profile),
            "baseline_profile_sha256": profile.sha256(),
            "family": spec.family,
            "hyperparameters": dict(spec.hyperparameters),
            "expected_net_return_threshold_bps": float(selected["selected_threshold_bps"]),
            "maximum_stop_probability": float(selected["selected_maximum_stop_probability"]),
            "frozen_round_trip_cost_bps": frozen_cost_bps,
            "feature_columns": list(MODEL_FEATURE_COLUMNS),
            "ordered_feature_sha256": published.ordered_feature_sha256,
            "opportunity_estimator": fitted.opportunity_estimator,
            "downside_estimator": fitted.downside_estimator,
            "downside_calibrator": fitted.downside_calibrator,
            "downside_fit_sessions": list(fitted.fit_sessions),
            "downside_calibration_sessions": list(fitted.calibration_sessions),
            "dataset": dataset_identity_val,
            "training_config": config_payload,
            "training_config_sha256": config_hash,
            "future_data_contract": _future_data_contract(policy),
        }
    _guard_memory(policy, "intraday development publication", peak=True)
    _publish_development(
        output_directory,
        candidate,
        evaluation,
        model_card,
        audit_ledger,
        retained_predictions[audit_candidate],
    )
    load_complete_intraday_development_output(output_directory)
    return DevelopmentTrainingResult(output_directory, status, selected_id, evaluation)


def _candidate_specs(config: IntradayDevelopmentConfig) -> tuple[_CandidateSpec, ...]:
    ridge = tuple(
        _CandidateSpec(
            f"ridge_opportunity_alpha_{alpha:g}_logistic_downside_c_{c:g}",
            "ridge_logistic_pair",
            {"alpha": alpha, "downside_c": c},
        )
        for alpha in config.ridge_alphas
        for c in config.logistic_c_values
    )
    hgb = tuple(
        _CandidateSpec(
            f"hgb_opportunity_downside_lr_{rate:g}_leaves_{leaves}",
            "hgb_pair",
            {
                "learning_rate": rate,
                "max_leaf_nodes": leaves,
                "max_iter": config.hgb_max_iter,
                "max_bins": config.hgb_max_bins,
            },
        )
        for rate in config.hgb_learning_rates
        for leaves in config.hgb_max_leaf_nodes
    )
    candidates = ridge + hgb
    if len(candidates) > 3:
        raise DataReadinessError("A4.4 permits at most three paired candidates per hypothesis")
    return candidates


def _walk_forward_predictions(
    spec: _CandidateSpec,
    data: pd.DataFrame,
    features_full: np.ndarray,
    opportunity_target: np.ndarray,
    downside_target: np.ndarray,
    folds: tuple[_Fold, ...],
    security_holdout: frozenset[str],
    config: IntradayDevelopmentConfig,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    evidence: list[pd.DataFrame] = []
    records: list[dict[str, Any]] = []
    for fold in folds:
        train_mask = (
            data["session_date_et"].isin(fold.train_sessions) & ~data["security_id"].astype(str).isin(security_holdout)
        ).to_numpy()
        max_label = data.loc[train_mask, "label_available_at_utc"].max()
        validation_mask = data["session_date_et"].isin(fold.validation_sessions).to_numpy()
        min_decision = data.loc[validation_mask, "decision_time_utc"].min()
        if max_label >= min_decision:
            raise DataReadinessError(f"fold {fold.fold} violates label-time purging")

        gc.collect()

        fitted = _fit_pair(
            spec,
            data,
            features_full,
            opportunity_target,
            downside_target,
            fold.train_sessions,
            config,
            excluded_securities=security_holdout,
        )
        opportunity_score, stop_probability = _predict_pair(fitted, features_full[validation_mask])

        keep_columns = (
            "session_date_et",
            "decision_group_id",
            "entry_time_utc",
            "exit_bar_end_utc",
            "security_id",
            "dataset_row_id",
            "ticker",
            "net_return",
            "gross_return",
            "spy_return",
            "qqq_return",
            "sector_return",
            "spy_excess_return",
            "qqq_excess_return",
            "sector_excess_return",
            "target_hit",
            "stop_hit",
            "entry_price",
            "stop_price",
        )
        validation = data.loc[validation_mask, keep_columns].copy()
        validation["predicted_net_return"] = opportunity_score
        validation["predicted_stop_probability"] = stop_probability
        validation["validation_scope"] = np.where(
            validation["security_id"].astype(str).isin(security_holdout),
            "unseen_security",
            "seen_security",
        )
        validation["fold"] = fold.fold
        evidence.append(validation)
        records.append(
            {
                "fold": fold.fold,
                "train_sessions": len(fold.train_sessions),
                "validation_sessions": len(fold.validation_sessions),
                "embargo_sessions": list(fold.embargo_sessions),
                "downside_fit_sessions": len(fitted.fit_sessions),
                "downside_calibration_sessions": len(fitted.calibration_sessions),
                "last_downside_fit_session": fitted.fit_sessions[-1],
                "first_downside_calibration_session": fitted.calibration_sessions[0],
                "max_train_label_available_at_utc": pd.Timestamp(max_label).isoformat(),
                "min_validation_decision_time_utc": pd.Timestamp(min_decision).isoformat(),
                "role": "selection" if fold.fold < len(folds) - 1 else "development_confirmation",
            }
        )
        del fitted
        gc.collect()
    return pd.concat(evidence, ignore_index=True), records


def _validate_development_frame(
    published: PublishedIntradayDataset,
    config: IntradayDevelopmentConfig,
) -> pd.DataFrame:
    data = published.frame.copy()
    required = {
        *MODEL_FEATURE_COLUMNS,
        "dataset_row_id",
        "security_id",
        "ticker",
        "session_date_et",
        "decision_group_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "entry_time_utc",
        "exit_bar_end_utc",
        "feature_eligible",
        "label_eligible",
        "gross_return",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "target_hit",
        "stop_hit",
        "entry_price",
        "stop_price",
    }
    missing = sorted(required.difference(data.columns))
    if missing:
        raise DataReadinessError(f"intraday development columns are missing: {missing}")
    if len(data) < config.minimum_rows or data["security_id"].nunique() < config.minimum_securities:
        raise DataReadinessError("intraday development population is below frozen minimums")
    dates = pd.to_datetime(data["session_date_et"], errors="coerce").dt.date
    if dates.isna().any():
        raise DataReadinessError("session_date_et contains invalid values")
    if dates.max() > _parse_date(config.development_end_date, "development_end_date"):
        raise DataReadinessError("development trainer refuses observations after 2026-07-08")
    data["session_date_et"] = dates.astype(str)
    if data["dataset_row_id"].isna().any() or data["dataset_row_id"].duplicated().any():
        raise DataReadinessError("dataset row identity must be complete and unique")
    if not data["feature_eligible"].map(_strict_bool).all() or not data["label_eligible"].map(_strict_bool).all():
        raise DataReadinessError("development rows must be feature- and label-eligible")
    for column in ("target_hit", "stop_hit"):
        if not data[column].map(lambda value: value is True or value is False or isinstance(value, np.bool_)).all():
            raise DataReadinessError(f"{column} must be boolean")
        data[column] = data[column].astype(bool)
    if (data["target_hit"] & data["stop_hit"]).any():
        raise DataReadinessError("target and stop cannot both be hit")
    for column in (
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "entry_time_utc",
        "exit_bar_end_utc",
    ):
        parsed = pd.to_datetime(data[column], utc=True, errors="coerce")
        if parsed.isna().any():
            raise DataReadinessError(f"{column} contains invalid UTC timestamps")
        data[column] = parsed
    if data["feature_available_at_utc"].gt(data["decision_time_utc"]).any():
        raise DataReadinessError("feature availability occurs after decision time")
    if data["label_available_at_utc"].lt(data["exit_bar_end_utc"]).any():
        raise DataReadinessError("label availability precedes the completed path")
    horizon = data["exit_bar_end_utc"] - data["entry_time_utc"]
    if horizon.le(pd.Timedelta(0)).any() or horizon.gt(pd.Timedelta(minutes=30)).any():
        raise DataReadinessError("development labels must use executable paths of at most 30 minutes")
    numeric = [
        *MODEL_FEATURE_COLUMNS,
        "gross_return",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "entry_price",
        "stop_price",
    ]
    for column in numeric:
        values = pd.to_numeric(data[column], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype="float64")).all():
            raise DataReadinessError(f"{column} must be finite")
        data[column] = values.astype("float32" if column in MODEL_FEATURE_COLUMNS else "float64")
    expected = data["gross_return"] - published.frozen_round_trip_cost_bps / 10_000.0
    if not np.allclose(expected, data["net_return"], rtol=0.0, atol=1e-10):
        raise DataReadinessError("net return does not match the frozen round-trip cost")
    for benchmark in ("spy", "qqq", "sector"):
        expected_excess = data["net_return"] - data[f"{benchmark}_return"]
        if not np.allclose(
            expected_excess,
            data[f"{benchmark}_excess_return"],
            rtol=0.0,
            atol=1e-10,
        ):
            raise DataReadinessError(f"{benchmark.upper()} excess return does not match the executable interval")
    if data["entry_price"].le(0.0).any() or data["stop_price"].le(0.0).any():
        raise DataReadinessError("entry and stop prices must be positive")
    return data.sort_values(["decision_time_utc", "decision_group_id", "security_id"], kind="stable").reset_index(drop=True)


def _ordered_sessions(data: pd.DataFrame) -> tuple[str, ...]:
    ordered = (
        data.groupby("session_date_et", as_index=False, observed=True)["decision_time_utc"]
        .min()
        .sort_values(["decision_time_utc", "session_date_et"], kind="stable")
    )
    return tuple(ordered["session_date_et"].astype(str))
