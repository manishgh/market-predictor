"""Production-grade candidate training for the ten-session edge-rebuild swing strategy."""

from __future__ import annotations

from market_predictor.edge_rebuild.training.swing_types import (
    SwingTrainingConfig,
    CandidateSpec,
    FittedCandidate,
    SwingTrainingResult,
    SwingPanelBinding,
    SwingProfileData,
    _guard,
    _read_json,
    _write_json,
    _resolve_inside,
    _strict_bool,
    _is_unapproved_source_feature,
    _sequence_sha256,
    _json_sha256,
    _iso,
)
from market_predictor.edge_rebuild.training.data_io import (
    load_complete_swing_feature_panel,
    load_swing_panel_binding,
    load_swing_profile,
    _partition_records_for_sessions,
    _validate_profile_session_coverage,
    _validate_profile_frame,
    _projected_profile_memory_bytes,
    _security_holdout_mask,
)
from market_predictor.edge_rebuild.training.lgbm_models import (
    _fit_candidate,
    _predict_probability,
    _raw_probability,
    _linex_objective,
)
from market_predictor.edge_rebuild.training.swing_evaluation import (
    _evaluate_validation_candidate,
    _evaluation_metrics,
    _validation_scopes_pass_economic_gates,
    _probability_distribution,
    _threshold_selection_key,
    _scope_economic_key,
    _selection_key,
    _evaluation_columns,
)

import shutil
import tempfile
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, fields
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import joblib
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_artifact_contracts import (
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_PATH_COST_POLICY,
    SWING_BASELINE_ABLATION_ORDER,
    SWING_FEATURE_PROFILE,
    swing_baseline_feature_columns,
)
from market_predictor.edge_rebuild.temporal_manifest import (
    build_temporal_schedule,
    load_temporal_manifest_config,
)
from market_predictor.edge_rebuild.training.data_io import (
    _security_holdout_mask,
    load_swing_panel_binding,
    load_swing_profile,
)
from market_predictor.edge_rebuild.training.evaluation import (
    _overlap_audit,
)
from market_predictor.edge_rebuild.training.lgbm_models import (
    _fit_candidate,
    _predict_probability,
)
from market_predictor.edge_rebuild.training.swing_evaluation import (
    _evaluate_validation_candidate,
    _evaluation_columns,
    _evaluation_metrics,
    _selection_key,
)
from market_predictor.edge_rebuild.training.swing_types import (
    CandidateSpec,
    SwingPanelBinding,
    SwingTrainingConfig,
    SwingTrainingResult,
    _guard,
    _json_sha256,
    _read_json,
    _resolve_inside,
    _sequence_sha256,
    _write_json,
)
from market_predictor.edge_rebuild.training.utils import (
    _mapping,
)
from market_predictor.edge_rebuild.training.walk_forward import (
    _assert_label_purge,
    _governed_folds,
    _governed_model_sessions,
    _split_record,
)
from market_predictor.resources import (
    memory_audit,
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




def load_swing_training_config(path: Path) -> SwingTrainingConfig:
    """Load a complete policy; partial or unknown fields are rejected."""

    try:
        raw = tomllib.loads(path.read_text(encoding="utf-8"))
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise DataReadinessError(f"swing training policy is unreadable: {path}") from exc
    payload = raw.get("training")
    if not isinstance(payload, Mapping):
        raise DataReadinessError("swing training policy requires a [training] table")
    expected = {field.name for field in fields(SwingTrainingConfig)}
    actual = {str(key) for key in payload}
    if actual != expected:
        raise DataReadinessError(
            "swing training policy fields differ; "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )
    values = dict(payload)
    for name in (
        "probability_thresholds",
        "logistic_c_values",
        "xgb_learning_rates",
    ):
        value = values[name]
        if not isinstance(value, list):
            raise DataReadinessError(f"swing training policy {name} must be an array")
        values[name] = tuple(value)
    try:
        return SwingTrainingConfig(**values)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError("swing training policy is invalid") from exc


























def train_swing_edge_candidate(
    panel_authority_directory: Path,
    output_directory: Path,
    *,
    strategy_contract: StrategyContract,
    config: SwingTrainingConfig,
    temporal_policy_path: Path,
) -> SwingTrainingResult:
    """Select one candidate on validation and touch the locked final test once."""

    _guard(config, "swing training start", peak=False)
    if output_directory.exists():
        raise FileExistsError(f"immutable output already exists: {output_directory}")
    if strategy_contract.swing.horizon_sessions != HORIZON_SESSIONS:
        raise DataReadinessError("trainer accepts only the active ten-session strategy")
    if strategy_contract.swing.round_trip_cost_bps != config.expected_round_trip_cost_bps:
        raise DataReadinessError("strategy and training cost contracts differ")
    if config.maximum_trades_per_decision != strategy_contract.swing.maximum_trades_per_decision:
        raise DataReadinessError("training trade cap differs from the authoritative strategy contract")
    temporal_config = load_temporal_manifest_config(temporal_policy_path)
    if (
        strategy_contract.validation.swing_walk_forward_folds != 1
        or temporal_config.validation_embargo_expected_sessions
        != strategy_contract.validation.embargo_sessions
        or temporal_config.final_embargo_expected_sessions
        != strategy_contract.validation.embargo_sessions
        or temporal_config.label_horizon_sessions != HORIZON_SESSIONS
        or temporal_config.unseen_security_holdout_fraction
        != strategy_contract.validation.unseen_ticker_holdout_fraction
        or temporal_config.modeled_decision_start.isoformat()
        != config.decision_start_date
    ):
        raise DataReadinessError(
            "temporal manifest differs from the authoritative strategy contract"
        )
    temporal_policy_sha256 = file_sha256(temporal_policy_path)
    schedule = build_temporal_schedule(temporal_config)
    folds = _governed_folds(schedule)
    model_sessions = _governed_model_sessions(schedule)
    final_refit_sessions = tuple(
        value.isoformat() for value in schedule.final_refit_sessions
    )
    test_sessions = tuple(
        value.isoformat() for value in schedule.locked_test_sessions
    )
    final_access_sessions = tuple(sorted({*final_refit_sessions, *test_sessions}))
    binding = load_swing_panel_binding(
        panel_authority_directory,
        strategy_contract=strategy_contract,
        config=config,
    )
    config_record = asdict(config)
    config_sha256 = _json_sha256(config_record)
    specs = _candidate_specs(config, strategy_contract)
    if len(specs) > config.maximum_learned_candidates:
        raise DataReadinessError("candidate count exceeds the frozen sequential budget")

    validation_records: list[dict[str, Any]] = []
    split_record = _split_record(
        folds=folds,
        schedule=schedule,
        temporal_config=temporal_config,
        temporal_policy_sha256=temporal_policy_sha256,
        strategy_contract=strategy_contract,
    )
    profile_data = load_swing_profile(
        binding,
        SWING_FEATURE_PROFILE,
        strategy_contract=strategy_contract,
        config=config,
        sessions=model_sessions,
    )
    profile_identity = profile_data.decision_ids_sha256
    for spec in specs:
        validation_records.append(
            _evaluate_validation_candidate(
                spec,
                profile_data,
                folds,
                config,
                strategy_contract,
            )
        )
        release_process_memory()
        _guard(config, f"{spec.candidate_id} validation", peak=True)
    del profile_data
    release_process_memory()

    eligible_candidates = [
        record for record in validation_records if record.get("candidate_eligible") is True
    ]
    if not eligible_candidates:
        no_candidate_evaluation = {
            "schema": EVALUATION_SCHEMA,
            "status": "no_candidate",
            "model_family": "swing_baseline",
            "promotion_permitted": False,
            "selection_basis": "validation_only",
            "test_access_count": 0,
            "locked_test_outcomes_read": False,
            "outcome_contract": _swing_outcome_contract(config, strategy_contract),
            "dataset": _binding_record(binding, profile_identity),
            "training_config": config_record,
            "training_config_sha256": config_sha256,
            "temporal_manifest_policy_sha256": temporal_policy_sha256,
            "split": split_record,
            "validation_candidates": validation_records,
            "feature_ablation_order": list(SWING_BASELINE_ABLATION_ORDER),
            "reason": "no candidate passed the frozen validation economic gates",
        }
        no_candidate_model_card = {
            "schema": MODEL_CARD_SCHEMA,
            "model_schema": MODEL_SCHEMA,
            "status": "no_candidate",
            "model_family": "swing_baseline",
            "promotion_permitted": False,
            "candidate_id": None,
            "outcome_contract": _swing_outcome_contract(config, strategy_contract),
            "dataset": _binding_record(binding, profile_identity),
            "strategy_contract_sha256": strategy_contract.sha256(),
            "training_config_sha256": config_sha256,
            "temporal_manifest_policy_sha256": temporal_policy_sha256,
        }
        _publish_immutable(
            output_directory,
            None,
            no_candidate_evaluation,
            no_candidate_model_card,
        )
        return SwingTrainingResult(
            output_directory=output_directory,
            selected_candidate_id=None,
            evaluation=no_candidate_evaluation,
            model_card=no_candidate_model_card,
        )
    family_groups: dict[str, list[dict[str, Any]]] = {}
    for record in eligible_candidates:
        spec = next(s for s in specs if s.candidate_id == record["candidate_id"])
        fam = (
            "classifier"
            if spec.estimator_family in ("logistic", "xgboost_ranker")
            else spec.estimator_family
        )
        family_groups.setdefault(fam, []).append(record)

    selected_records = {
        fam: max(recs, key=_selection_key)
        for fam, recs in family_groups.items()
    }
    selected_specs = {
        fam: next(s for s in specs if s.candidate_id == r["candidate_id"])
        for fam, r in selected_records.items()
    }
    selected_thresholds = {
        fam: float(r["selected_probability_threshold"])
        for fam, r in selected_records.items()
    }
    bundle_candidate_id = _bundle_candidate_id(
        selected_records=selected_records,
        panel_request_sha256=binding.request_sha256,
        training_config_sha256=config_sha256,
        temporal_policy_sha256=temporal_policy_sha256,
    )

    reference_spec = next(iter(selected_specs.values()))

    # The selected profile is reloaded only after validation has frozen both
    # candidate and threshold. This is the single controlled final-test access.
    selected_data = load_swing_profile(
        binding,
        reference_spec.profile,
        strategy_contract=strategy_contract,
        config=config,
        sessions=final_access_sessions,
    )
    holdout = _security_holdout_mask(selected_data.frame, strategy_contract)
    final_test_columns = list(dict.fromkeys((
        *_evaluation_columns(),
        *selected_data.feature_columns,
    )))
    development_columns = list(dict.fromkeys((
        "decision_id",
        "security_id",
        "session_date_et",
        "decision_time_utc",
        "label_available_at_utc",
        "target",
        "barrier_net_return",
        "ranking_reliability_weight",
        *selected_data.feature_columns,
    )))
    final_test = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(test_sessions),
        final_test_columns,
    ].copy()
    development = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(final_refit_sessions),
        development_columns,
    ].copy()
    _assert_label_purge(development, final_test, "final development/test")

    unseen_development = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(final_refit_sessions)
        & ~holdout,
        development_columns,
    ].copy()
    unseen_final_test = selected_data.frame.loc[
        selected_data.frame["session_date_et"].isin(test_sessions)
        & holdout,
        final_test_columns,
    ].copy()
    _assert_label_purge(
        unseen_development,
        unseen_final_test,
        "unseen-security final development/test",
    )

    fitted_models = {}
    unseen_fitted_models = {}
    temporal_final_metrics = {}
    unseen_final_metrics = {}

    for fam, spec in selected_specs.items():
        threshold = selected_thresholds[fam]
        fitted = _fit_candidate(spec, development, config)
        probability = _predict_probability(fitted, final_test, spec.feature_columns)
        temporal_final_metrics[fam] = _evaluation_metrics(
            final_test,
            probability,
            threshold=threshold,
            config=config,
            strategy_contract=strategy_contract,
            session_calendar=test_sessions,
        )
        
        unseen_fitted = _fit_candidate(
            spec,
            unseen_development,
            config,
        )
        unseen_probability = _predict_probability(
            unseen_fitted,
            unseen_final_test,
            spec.feature_columns,
        )
        unseen_final_metrics[fam] = _evaluation_metrics(
            unseen_final_test,
            unseen_probability,
            threshold=threshold,
            config=config,
            strategy_contract=strategy_contract,
            session_calendar=test_sessions,
        )
        
        fitted_models[fam] = fitted
        unseen_fitted_models[fam] = unseen_fitted

    _guard(config, "swing final test", peak=True)

    evaluation: dict[str, Any] = {
        "schema": EVALUATION_SCHEMA,
        "status": "candidate_only",
        "model_family": "swing_baseline",
        "promotion_permitted": False,
        "selection_basis": "validation_only",
        "selection_policy": {
            "name": "SWING_CONSERVATIVE_ECONOMICS_V2",
            "auc_used_for_selection": False,
            "ordered_key": [
                "worst_holding_aligned_SPY_QQQ_sector_excess_calendar_ci_low",
                "portfolio_daily_return_bootstrap_ci_low",
                "worst_mean_holding_aligned_SPY_QQQ_sector_excess",
                "mean_managed_net_return",
                "negative_daily_mark_to_market_drawdown",
                "negative_turnover",
                "lower_probability_threshold_tie_break",
                "simpler_estimator_tie_break",
                "fewer_features_tie_break",
            ],
        },
        "test_access_count": 1,
        "locked_test_outcomes_read": True,
        "locked_test_access_policy": (
            "outcome columns loaded once only after both validation scopes passed"
        ),
        "outcome_contract": _swing_outcome_contract(config, strategy_contract),
        "strategy": {
            "horizon_trading_sessions": HORIZON_SESSIONS,
            "entry_reference": strategy_contract.swing.entry_reference,
            "exit_rule": strategy_contract.swing.exit_rule,
            "round_trip_cost_bps": config.expected_round_trip_cost_bps,
            "target": "top_sector_relative_quantile_of_managed_barrier_net_return",
        },
        "dataset": _binding_record(binding, profile_identity),
        "training_config": config_record,
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "split": split_record,
        "overlap_audit": _overlap_audit(
            selected_data.frame,
            strategy_contract=strategy_contract,
            final_refit_sessions=final_refit_sessions,
            final_test_sessions=test_sessions,
            final_embargo_sessions=tuple(
                value.isoformat() for value in schedule.final_embargo_sessions
            ),
        ),
        "validation_candidates": validation_records,
        "feature_ablation_order": list(SWING_BASELINE_ABLATION_ORDER),
        "selected_bundle_id": bundle_candidate_id,
        "selected_candidate_ids": {fam: r["candidate_id"] for fam, r in selected_records.items()},
        "selected_profile": reference_spec.profile,
        "selected_probability_thresholds": selected_thresholds,
        "selected_validation_keys": {fam: list(_selection_key(r)) for fam, r in selected_records.items()},
        "final_test": {
            "temporal_generalization_full_pit_cross_section": temporal_final_metrics,
            "unseen_security_generalization_stable_20pct": unseen_final_metrics,
        },
        "benchmark_evaluation_basis": (
            "selection uses managed stock net return plus exact fixed-ten-session "
            "SPY, QQQ, and sector excess; managed-exit-session-close benchmark "
            "comparisons are approximate diagnostics only"
        ),
        "managed_path_cost_policy": MANAGED_PATH_COST_POLICY,
        "memory": memory_audit(
            hard_budget_gib=config.maximum_process_memory_gib,
            headroom_gib=config.memory_guard_headroom_gib,
        ).to_record(),
    }
    model_card: dict[str, Any] = {
        "schema": MODEL_CARD_SCHEMA,
        "model_schema": MODEL_SCHEMA,
        "status": "candidate",
        "model_family": "swing_baseline",
        "promotion_permitted": False,
        "candidate_id": bundle_candidate_id,
        "candidate_ids": {fam: r["candidate_id"] for fam, r in selected_records.items()},
        "ablation_profile": reference_spec.profile,
        "outcome_contract": _swing_outcome_contract(config, strategy_contract),
        "models": {
            fam: {
                "estimator_family": selected_specs[fam].estimator_family,
                "feature_group": selected_specs[fam].feature_group,
                "feature_columns": list(selected_specs[fam].feature_columns),
                "hyperparameters": dict(selected_specs[fam].hyperparameters),
                "probability_threshold": selected_thresholds[fam],
            } for fam in selected_specs
        },
        "feature_columns": list(selected_data.feature_columns),
        "feature_set_sha256": _sequence_sha256(selected_data.feature_columns),
        "training_rows": len(development),
        "training_sessions": int(development["session_date_et"].nunique()),
        "training_securities": int(development["security_id"].nunique()),
        "locked_test_rows": len(final_test),
        "locked_test_unseen_security_rows": len(unseen_final_test),
        "dataset": _binding_record(binding, profile_identity),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "calibration_method": "platt_sigmoid_on_prior_purged_sessions",
        "final_test_opened_once": True,
        "limitations": [
            "Candidate is not promoted and must not be used for live trading.",
            "Drawdown and turnover use the panel's managed daily mark-to-market paths and overlapping cohorts.",
            "Managed-exit benchmark comparisons use exit-session closes and are approximate diagnostics only.",
            "SEC, global, and Finviz inputs are excluded pending independent causal ablation.",
        ],
    }
    payload: dict[str, Any] = {
        "schema": MODEL_SCHEMA,
        "status": "candidate",
        "model_family": "swing_baseline",
        "promotion_permitted": False,
        "candidate_id": bundle_candidate_id,
        "ablation_profile": reference_spec.profile,
        "feature_columns": list(selected_data.feature_columns),
        "feature_set_sha256": _sequence_sha256(selected_data.feature_columns),
        "dataset": _binding_record(binding, profile_identity),
        "training_config_sha256": config_sha256,
        "temporal_manifest_policy_sha256": temporal_policy_sha256,
        "strategy_contract_sha256": strategy_contract.sha256(),
        "fitted_models": fitted_models,
        "probability_thresholds": selected_thresholds,
    }
    _publish_immutable(
        output_directory,
        payload,
        evaluation,
        model_card,
    )
    return SwingTrainingResult(
        output_directory=output_directory,
        selected_candidate_id=bundle_candidate_id,
        evaluation=evaluation,
        model_card=model_card,
    )


def _candidate_specs(
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
) -> tuple[CandidateSpec, ...]:
    specs: list[CandidateSpec] = []
    for feature_group in SWING_BASELINE_ABLATION_ORDER:
        columns = swing_baseline_feature_columns(
            feature_group,
            contract=strategy_contract,
        )
        for value in config.logistic_c_values:
            specs.append(
                CandidateSpec(
                    candidate_id=(
                        f"swing_baseline.{feature_group}.logistic.c_{value:g}"
                    ),
                    profile=SWING_FEATURE_PROFILE,
                    feature_group=feature_group,
                    feature_columns=columns,
                    estimator_family="logistic",
                    hyperparameters={"C": value, "solver": "lbfgs", "threads": 1},
                )
            )
    full_group = SWING_BASELINE_ABLATION_ORDER[-1]
    full_columns = swing_baseline_feature_columns(
        full_group,
        contract=strategy_contract,
    )
    for rate in config.xgb_learning_rates:
        for depth in config.xgb_max_depths:
            for label, family in _XGB_GRID:
                specs.append(
                    CandidateSpec(
                        candidate_id=(
                            f"swing_baseline.{full_group}.{label}."
                            f"lr_{rate:g}.depth_{depth}"
                        ),
                        profile=SWING_FEATURE_PROFILE,
                        feature_group=full_group,
                        feature_columns=full_columns,
                        estimator_family=family,
                        hyperparameters={
                            "learning_rate": rate,
                            "max_depth": depth,
                            "n_estimators": config.xgb_n_estimators,
                            "threads": 1,
                        },
                    )
                )
    return tuple(specs)


def _bundle_candidate_id(
    *,
    selected_records: Mapping[str, Mapping[str, Any]],
    panel_request_sha256: str,
    training_config_sha256: str,
    temporal_policy_sha256: str,
) -> str:
    identity = {
        "selected_candidates": {
            family: {
                "candidate_id": record.get("candidate_id"),
                "probability_threshold": record.get("selected_probability_threshold"),
            }
            for family, record in sorted(selected_records.items())
        },
        "panel_request_sha256": panel_request_sha256,
        "training_config_sha256": training_config_sha256,
        "temporal_policy_sha256": temporal_policy_sha256,
    }
    return f"{SWING_BASELINE_BUNDLE_PREFIX}{_json_sha256(identity)[:16]}"


def _is_swing_baseline_bundle_id(value: object) -> bool:
    text = str(value)
    suffix = text.removeprefix(SWING_BASELINE_BUNDLE_PREFIX)
    return (
        text.startswith(SWING_BASELINE_BUNDLE_PREFIX)
        and len(suffix) == 16
        and all(character in "0123456789abcdef" for character in suffix)
    )


def _swing_outcome_contract(
    config: SwingTrainingConfig,
    strategy_contract: StrategyContract,
) -> dict[str, Any]:
    return {
        "schema": "edge_rebuild.swing_outcome_contract.v1",
        "horizon": "ten exact exchange sessions",
        "entry": strategy_contract.swing.entry_reference,
        "managed_exit": strategy_contract.swing.exit_rule,
        "estimator_target": {
            "column": "target",
            "definition": (
                "published top rank_label within the point-in-time sector decision cohort"
            ),
        },
        "economic_target": {
            "column": "future_excess_return_10d_vs_sector",
            "definition": (
                "exact ten-session stock net return after costs less the point-in-time "
                "sector ETF return over the identical interval"
            ),
        },
        "comparable_binary_diagnostic": (
            "exact ten-session sector excess return after costs is positive"
        ),
        "benchmark_excess_columns": {
            "SPY": "future_excess_return_10d_vs_spy",
            "QQQ": "future_excess_return_10d_vs_qqq",
            "sector": "future_excess_return_10d_vs_sector",
        },
        "benchmark_interval": "identical next-open through exact ten-session close interval",
        "round_trip_cost_bps": config.expected_round_trip_cost_bps,
    }


def _ordered_sessions(data: pd.DataFrame) -> tuple[str, ...]:
    sessions = (
        data.groupby("session_date_et", as_index=False, observed=True)["decision_time_utc"]
        .min()
        .sort_values(["decision_time_utc", "session_date_et"], kind="stable")
    )
    order = tuple(sessions["session_date_et"].astype(str))
    if len(order) != len(set(order)):
        raise DataReadinessError("exchange sessions are not unique")
    return order




























def _publish_immutable(
    output_directory: Path,
    candidate: Mapping[str, Any] | None,
    evaluation: Mapping[str, Any],
    model_card: Mapping[str, Any],
) -> None:
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(
        tempfile.mkdtemp(
            prefix=f".{output_directory.name}.", dir=output_directory.parent
        )
    )
    try:
        if candidate is not None:
            joblib.dump(dict(candidate), temporary / _CANDIDATE_NAME, compress=3)
        _write_json(temporary / _EVALUATION_NAME, evaluation)
        _write_json(temporary / _MODEL_CARD_NAME, model_card)
        artifact_names = [
            *([_CANDIDATE_NAME] if candidate is not None else []),
            _EVALUATION_NAME,
            _MODEL_CARD_NAME,
        ]
        artifacts = {
            name: {
                "sha256": file_sha256(temporary / name),
                "bytes": (temporary / name).stat().st_size,
            }
            for name in artifact_names
        }
        state = "candidate" if candidate is not None else "no_candidate"
        manifest = {
            "schema": MODEL_SCHEMA,
            "state": state,
            "promotion_permitted": False,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "files": artifacts,
        }
        _write_json(temporary / _MANIFEST_NAME, manifest)
        authority = {
            "schema": OUTPUT_AUTHORITY_SCHEMA,
            "state": state,
            "promotion_permitted": False,
            "artifact": _MANIFEST_NAME,
            "artifact_sha256": file_sha256(temporary / _MANIFEST_NAME),
        }
        _write_json(temporary / _AUTHORITY_NAME, authority)
        try:
            temporary.rename(output_directory)
        except FileExistsError:
            raise FileExistsError(
                f"immutable output already exists: {output_directory}"
            ) from None
    finally:
        if temporary.exists():
            shutil.rmtree(temporary)


def load_swing_candidate_authority(directory: Path) -> dict[str, Any]:
    """Strictly replay a candidate-only output and every artifact hash."""

    root = directory.resolve()
    manifest_path = root / _MANIFEST_NAME
    authority_path = root / _AUTHORITY_NAME
    manifest = _read_json(manifest_path, "swing candidate manifest")
    authority = _read_json(authority_path, "swing candidate authority")
    state = str(manifest.get("state"))
    if (
        manifest.get("schema") != MODEL_SCHEMA
        or state not in {"candidate", "no_candidate"}
        or manifest.get("promotion_permitted") is not False
        or authority.get("schema") != OUTPUT_AUTHORITY_SCHEMA
        or authority.get("state") != state
        or authority.get("promotion_permitted") is not False
        or authority.get("artifact") != _MANIFEST_NAME
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError("swing candidate authority does not verify")
    files = _mapping(manifest.get("files"), "candidate files")
    expected_files = {_EVALUATION_NAME, _MODEL_CARD_NAME}
    if state == "candidate":
        expected_files.add(_CANDIDATE_NAME)
    if set(files) != expected_files:
        raise DataReadinessError("swing candidate manifest has an unexpected file set")
    for name, raw in files.items():
        record = _mapping(raw, f"candidate file {name}")
        path = _resolve_inside(root, name)
        if record.get("sha256") != file_sha256(path) or int(record.get("bytes", -1)) != path.stat().st_size:
            raise DataReadinessError(f"swing candidate artifact does not verify: {name}")
    evaluation = _read_json(root / _EVALUATION_NAME, "swing evaluation")
    model_card = _read_json(root / _MODEL_CARD_NAME, "swing model card")
    if state == "no_candidate":
        if (
            evaluation.get("schema") != EVALUATION_SCHEMA
            or evaluation.get("status") != "no_candidate"
            or evaluation.get("promotion_permitted") is not False
            or evaluation.get("test_access_count") != 0
            or model_card.get("schema") != MODEL_CARD_SCHEMA
            or model_card.get("status") != "no_candidate"
            or model_card.get("promotion_permitted") is not False
            or model_card.get("candidate_id") is not None
            or evaluation.get("dataset") != model_card.get("dataset")
            or evaluation.get("training_config_sha256")
            != model_card.get("training_config_sha256")
            or evaluation.get("temporal_manifest_policy_sha256")
            != model_card.get("temporal_manifest_policy_sha256")
        ):
            raise DataReadinessError("swing no-candidate evidence is internally inconsistent")
        return {
            "status": "no_candidate",
            "candidate_id": None,
            "manifest": manifest,
            "evaluation": evaluation,
            "model_card": model_card,
        }
    payload = joblib.load(root / _CANDIDATE_NAME)
    if not isinstance(payload, Mapping):
        raise DataReadinessError("swing candidate payload is not an object")
    identities = {
        evaluation.get("selected_bundle_id"),
        model_card.get("candidate_id"),
        payload.get("candidate_id"),
    }
    if (
        evaluation.get("schema") != EVALUATION_SCHEMA
        or evaluation.get("status") != "candidate_only"
        or evaluation.get("promotion_permitted") is not False
        or evaluation.get("test_access_count") != 1
        or model_card.get("schema") != MODEL_CARD_SCHEMA
        or model_card.get("status") != "candidate"
        or model_card.get("promotion_permitted") is not False
        or payload.get("schema") != MODEL_SCHEMA
        or payload.get("status") != "candidate"
        or payload.get("promotion_permitted") is not False
        or evaluation.get("model_family") != "swing_baseline"
        or model_card.get("model_family") != "swing_baseline"
        or payload.get("model_family") != "swing_baseline"
        or len(identities) != 1
        or not _is_swing_baseline_bundle_id(next(iter(identities)))
        or evaluation.get("dataset") != model_card.get("dataset")
        or evaluation.get("dataset") != payload.get("dataset")
        or evaluation.get("training_config_sha256")
        != model_card.get("training_config_sha256")
        or evaluation.get("training_config_sha256")
        != payload.get("training_config_sha256")
        or evaluation.get("temporal_manifest_policy_sha256")
        != model_card.get("temporal_manifest_policy_sha256")
        or evaluation.get("temporal_manifest_policy_sha256")
        != payload.get("temporal_manifest_policy_sha256")
        or model_card.get("feature_columns") != list(payload.get("feature_columns", ()))
        or model_card.get("feature_set_sha256") != payload.get("feature_set_sha256")
    ):
        raise DataReadinessError("swing candidate evidence is internally inconsistent")
    return {
        "status": "candidate",
        "candidate_id": identities.pop(),
        "manifest": manifest,
        "evaluation": evaluation,
        "model_card": model_card,
    }


def _binding_record(binding: SwingPanelBinding, decision_ids_sha256: str) -> dict[str, Any]:
    return {
        "panel_manifest_schema": SWING_MATERIALIZATION_MANIFEST_SCHEMA,
        "panel_manifest_sha256": binding.manifest_sha256,
        "panel_authority_sha256": binding.authority_sha256,
        "panel_request_sha256": binding.request_sha256,
        "strategy_contract_sha256": binding.strategy_contract_sha256,
        "decision_ids_sha256": decision_ids_sha256,
    }




















