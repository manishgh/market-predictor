"""Evaluate one frozen policy against saved temporal scores; never refit a model."""
from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any, Literal

import exchange_calendars as xcals
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from pandas.testing import assert_frame_equal

from market_predictor.canonical.cutoffs import swing_prediction_cutoffs
from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.edge_rebuild.temporal_manifest import build_temporal_schedule, load_temporal_manifest_config
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.modeling.strategy_contract import StrategyContract, load_strategy_contract
from market_predictor.resources import assert_memory_budget
from market_predictor.swing.contracts.holding_accounting import HoldingSpecification
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_profile import ISSUER_REACTION_PROFILE
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.saved_evaluation_configuration import SavedEvaluationConfigurationEvidence
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets.action_evidence import CorporateActionEvidence
from market_predictor.swing.datasets.funded_policy_inputs import (
    FundedLotInput,
    unsimulated_managed_specification,
    verified_funded_policy_inputs,
)
from market_predictor.swing.datasets.saved_evaluation_configuration import verify_saved_evaluation_configuration
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.evaluation.accounting import evaluate_event_aware_swing_accounting
from market_predictor.swing.evaluation.ledger import swing_valuation_sessions
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.labels.frozen_exit import ExitPolicy
from market_predictor.swing.selection import select_constrained_swing_portfolio
from market_predictor.swing.training.return_artifacts import verify_unit
from market_predictor.swing.training.return_comparison import _run
from market_predictor.swing.training.return_validation import return_folds, security_transfer_mask

Learner = Literal["regularized_linear_return", "shallow_boosted_return"]
SCHEMA = "market_predictor.swing_saved_oof_policy_evaluation"
_IDENTITIES = ("decision_id", "security_id", "ticker", "sector", "session_date_et", "decision_time_utc")
_OPTIONAL_SHARED = (
    "parent_decision_id", "parent_ticker", "timeframe", "bar_start_utc", "prediction_cutoff_policy_id",
    "stock_component_id", "spy_component_id", "qqq_component_id", "sector_component_id",
    "training_eligible", "production_eligible", "stock_source_admitted", "fixed_comparisons_complete",
    "cross_section_eligible", "decision_group_id", "primary_benchmark", "feature_profile",
)
_REQUIRED_PARENT = (*_IDENTITIES, "feature_eligible", "cross_section_eligible", "decision_group_id", "primary_benchmark", "feature_profile")
_SCORE_COLUMNS = (*_IDENTITIES, "feature_eligible", "scope_eligible", "security_holdout", "predicted_excess_return", "prediction_status")
IMPLEMENTATION_PATHS = (
    "research/swing_oof_policy_evaluation.py", "swing/training/return_artifacts.py", "swing/training/return_comparison.py",
    "swing/training/return_validation.py", "swing/contracts/return_training.py", "swing/contracts/training_readiness.py",
    "swing/contracts/__init__.py", "swing/contracts/holding_accounting.py",
    "swing/contracts/issuer_reaction.py", "swing/contracts/issuer_reaction_profile.py",
    "swing/contracts/research.py", "swing/contracts/return_feature_profiles.py", "swing/features/panel.py",
    "swing/selection.py", "swing/datasets/funded_policy_inputs.py", "swing/labels/frozen_exit.py",
    "swing/evaluation/accounting.py", "swing/evaluation/ledger.py", "swing/evaluation/holding_accounting.py",
    "swing/evaluation/trade_simulation.py", "edge_rebuild/temporal_manifest.py", "modeling/validation.py",
    "modeling/strategy_contract.py", "modeling/resampling.py", "canonical/cutoffs.py", "canonical/store.py",
    "heavy_jobs.py", "resources.py", "evidence/hashing.py", "evidence/io.py",
    "swing/contracts/saved_evaluation_configuration.py", "swing/datasets/saved_evaluation_configuration.py",
)


@dataclass(frozen=True)
class _LedgerConfig:
    horizon_sessions: int
    expected_round_trip_cost_bps: float
    maximum_trades_per_decision: int


@dataclass(frozen=True)
class _SavedScores:
    rows: pd.DataFrame
    score_sessions: tuple[str, ...]
    entry_sessions: tuple[str, ...]
    historical_source_files: dict[str, str]
    request_sha256: str
    configuration_provenance: dict[str, Any] | None = None


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _guard() -> None:
    assert_memory_budget(stage="saved OOF policy evaluation", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=0.75, maximum_used_percent=90.0)


def _pin(root: Path, pin: SourcePin, pins: dict[str, str]) -> Path:
    path = inside(root, pin.path)
    key = path.relative_to(root).as_posix()
    _require(key not in pins or pins[key] == pin.sha256, "conflicting OOF input pins")
    _require(file_sha256(path) == pin.sha256, f"OOF input changed: {key}")
    pins[key] = pin.sha256
    return path


def _recheck(root: Path, pins: dict[str, str]) -> None:
    for name, digest in pins.items():
        _pin(root, SourcePin(path=name, sha256=digest), {})


def _object(root: Path, pin: SourcePin, pins: dict[str, str]) -> dict[str, Any]:
    return pinned_object(_pin(root, pin, pins), pin.sha256)


def _clock(value: Any) -> pd.Timestamp:
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None and stamp.utcoffset() == pd.Timedelta(0), "OOF clock is not explicit UTC")
    return stamp


def _booleans(frame: pd.DataFrame, names: tuple[str, ...]) -> None:
    for name in names:
        _require(frame[name].map(lambda value: isinstance(value, (bool, np.bool_))).all(), f"OOF {name} must be explicit boolean")


def _metadata(frame: pd.DataFrame) -> pd.DataFrame:
    _require(frame.columns.is_unique and not frame.empty and not frame.decision_id.duplicated().any(),
             "OOF metadata IDs are absent or duplicate")
    result = frame.copy()
    for name in ("decision_id", "security_id", "ticker", "sector", "decision_group_id", "primary_benchmark"):
        if name in result:
            _require(result[name].map(lambda value: isinstance(value, str) and bool(value) and value.strip() == value).all(),
                     f"OOF metadata {name} is invalid")
            result[name] = result[name].astype("string")
    _require(result.session_date_et.map(lambda value: type(value) is date).all(), "OOF sessions require canonical dates")
    result["decision_time_utc"] = result.decision_time_utc.map(_clock)
    _require(result.decision_time_utc.eq(swing_prediction_cutoffs(result.session_date_et)).all(), "OOF decision cutoff differs")
    return result


def _read_parent_metadata(
    root: Path, publication: SourcePin, request: dict[str, Any], policy: ReturnTrainingPolicy,
    strategy: StrategyContract, sessions: tuple[date, ...], pins: dict[str, str],
) -> pd.DataFrame:
    path = _pin(root, publication, pins)
    manifest = _object(root, publication, pins)
    parent_request = _object(root, SourcePin(path=(path.parent / "_request.json").relative_to(root).as_posix(),
                                          sha256=manifest["request_sha256"]), pins)
    names = tuple(swing_model_feature_columns(contract=strategy, catalyst=False))
    if policy.feature_profile in ("technical_relationships", ISSUER_REACTION_PROFILE):
        names = (*names, *RETURN_RELATIONSHIP_COLUMNS)
    if policy.feature_profile == ISSUER_REACTION_PROFILE:
        names = (*names, *REACTION_COLUMNS)
    _require(request["feature_names"] == list(names), "OOF features differ from frozen profile order")
    _require(bool(parent_request), "parent feature request is empty")
    months = sorted({day.isoformat()[:7] for day in sessions})
    _require(sorted(manifest["months"]) == months, "feature publication is not exact initial fit")
    parquet: Any = pq
    pieces = []
    for month in months:
        _guard()
        record = manifest["months"][month]
        child = record["profiles"][policy.published_profile]
        _require(child["path"] == f"{month}/{policy.published_profile}.parquet" and child["model_columns"] == list(names),
                 "parent feature child/profile order differs")
        clocks = {name: f"available_at_{name}" for name in names}
        _require(all(child["availability_columns"].get(name) == clock for name, clock in clocks.items()), "parent feature clocks differ")
        source = inside(path.parent, child["path"])
        for artifact, digest in ((source, child["sha256"]), (manifest_path_for(source), child["manifest_sha256"])):
            key = artifact.relative_to(root).as_posix()
            _require(request["source_files"].get(key) == digest, "feature child lacks saved training request pin")
            _pin(root, SourcePin(path=key, sha256=digest), pins)
        sidecar = pinned_object(manifest_path_for(source), child["manifest_sha256"])
        _require(sidecar.get("artifact_sha256") == child["sha256"] and sidecar.get("production_ready") is False
                 and sidecar.get("inputs") == {"request_sha256": manifest["request_sha256"]}, "feature sidecar lineage differs")
        available = set(parquet.ParquetFile(source).schema.names)
        _require(set(_REQUIRED_PARENT).union(clocks.values()).issubset(available), "parent selection metadata missing")
        columns = list(dict.fromkeys((*_REQUIRED_PARENT, *(name for name in _OPTIONAL_SHARED if name in available), *clocks.values())))
        frame = _metadata(pd.read_parquet(source, columns=columns))
        _booleans(frame, ("feature_eligible", "cross_section_eligible"))
        _require(len(frame) == record["rows"] == sidecar["rows"]
                 and json_sha256(sorted(frame.decision_id)) == record["decision_ids_sha256"]
                 and frame.session_date_et.map(lambda day: day.isoformat()[:7]).eq(month).all()
                 and frame.feature_profile.eq(policy.published_profile).all(), "parent row population/profile differs")
        latest = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        for column in clocks.values():
            values = frame[column].dropna().map(_clock)
            _require(values.le(frame.loc[values.index, "decision_time_utc"]).all(), "feature availability exceeds decision cutoff")
            if len(values):
                latest.loc[values.index] = pd.concat([latest.loc[values.index], values], axis=1).max(axis=1)
        frame = frame.drop(columns=list(clocks.values()))
        frame["maximum_feature_available_at_utc"] = latest
        pieces.append(frame)
    rows = pd.concat(pieces, ignore_index=True)
    _require(not rows.decision_id.duplicated().any() and set(rows.session_date_et) == set(sessions)
             and len(rows) == request["input_rows"]
             and json_sha256(sorted(rows.decision_id)) == request["input_decision_ids_sha256"], "parent full decision population differs")
    return rows


def _saved_scores(
    *, root: Path, run_manifest: SourcePin, feature_publication: SourcePin, strategy_pin: SourcePin,
    research_pin: SourcePin, strategy: StrategyContract, learner: Learner, pins: dict[str, str],
    historical_configuration_evidence: SourcePin | None = None,
) -> _SavedScores:
    parquet: Any = pq
    run_path = _pin(root, run_manifest, pins)
    manifest, request = _run(root, run_path.parent, run_manifest.sha256, pins)
    policy = ReturnTrainingPolicy.model_validate(request["policy"])
    _require(learner in ("regularized_linear_return", "shallow_boosted_return"), "unsupported saved learner")
    readiness = _object(root, policy.readiness, pins)
    readiness_policy = TrainingReadinessPolicy.model_validate(_object(root, policy.readiness_config, pins))
    _require(readiness_policy.publication == feature_publication
             and readiness_policy.research_contract == research_pin and readiness_policy.published_profile == policy.published_profile,
             "saved readiness/config/publication bindings differ")
    configuration_provenance: dict[str, Any] | None = None
    if historical_configuration_evidence is None:
        _require(readiness_policy.strategy_contract == strategy_pin, "saved readiness strategy binding differs")
        temporal_path = _pin(root, readiness_policy.temporal_contract, pins)
    else:
        # Archived hashes remain historical provenance. The separate proof binds
        # recovered original bytes to current executable configurations exactly.
        proof = SavedEvaluationConfigurationEvidence.model_validate_json(
            json.dumps(_object(root, historical_configuration_evidence, pins)))
        current_temporal = proof.temporal.current
        verified = verify_saved_evaluation_configuration(
            root=root, evidence_pin=historical_configuration_evidence, run_manifest=run_manifest,
            strategy_pin=strategy_pin, temporal_pin=current_temporal,
            original_strategy_pin=readiness_policy.strategy_contract,
            original_temporal_pin=readiness_policy.temporal_contract, pins=pins,
        )
        _require(verified.strategy_path == inside(root, strategy_pin.path), "verified strategy path differs")
        temporal_path = verified.temporal_path
        configuration_provenance = dict(verified.provenance)
    for pin in (policy.readiness, policy.readiness_config, feature_publication, readiness_policy.strategy_contract,
                research_pin, readiness_policy.temporal_contract):
        _require(request["source_files"].get(pin.path) == pin.sha256, "required metadata/config lacks original training pin")
    _require(readiness.get("publication_sha256") == feature_publication.sha256 and readiness.get("published_profile", "technical_market")
             == policy.published_profile, "saved readiness publication/profile differs")
    temporal = load_temporal_manifest_config(temporal_path)
    sessions = build_temporal_schedule(temporal).folds[0].train_sessions
    folds = return_folds(sessions, count=policy.folds, minimum_train=policy.minimum_train_sessions, embargo=policy.embargo_sessions)
    expected_folds = [{"number": fold.number, "train": [day.isoformat() for day in fold.train],
                       "embargo": [day.isoformat() for day in fold.embargo], "score": [day.isoformat() for day in fold.score]}
                      for fold in folds]
    _require(request["folds"] == expected_folds, "saved OOF fold calendar differs")
    score_days = tuple(day.isoformat() for fold in folds for day in fold.score)
    _require(tuple(len(fold.score) for fold in folds) == (179, 179, 179, 181) and len(score_days) == 718
             and score_days[0] == "2021-07-21" and score_days[-1] == "2024-05-28"
             and score_days[-11] == "2024-05-13", "OOF score/terminal calendar differs")
    parent = _read_parent_metadata(root, feature_publication, request, policy, strategy, sessions, pins)
    holdouts = set(parent.loc[security_transfer_mask(parent.security_id, policy.holdout_fraction), "security_id"])
    _require(sorted(holdouts) == request["holdout_security_ids"], "saved holdout identities differ")
    if configuration_provenance is not None:
        configuration_provenance.update(fold_calendar_sha256=json_sha256(expected_folds),
            holdout_security_ids_sha256=json_sha256(sorted(holdouts)),
            saved_fold_calendar_reproduced=True, complete_parent_holdout_assignment_reproduced=True)
    parameters = policy.linear.model_dump() if learner == "regularized_linear_return" else policy.boosted.model_dump()
    pieces = []
    for fold in folds:
        _guard()
        key = f"{learner}/temporal/fold-{fold.number}"
        record = manifest["units"][key]
        directory = inside(root, record["path"])
        _require(directory == run_path.parent / key, "saved OOF unit path differs")
        unit_pin = SourcePin(path=(directory / "_manifest.json").relative_to(root).as_posix(), sha256=record["manifest_sha256"])
        unit_manifest = _object(root, unit_pin, pins)
        unit = unit_manifest["unit"]
        cutoff = swing_prediction_cutoffs(pd.Series([fold.score[0]])).iloc[0]
        _require(unit["family"] == learner and unit["scope"] == "temporal" and unit["parameters"] == parameters
                 and unit["feature_names"] == request["feature_names"] and unit["target"] == policy.target
                 and unit["preprocessing"] == policy.missingness and _clock(unit["fit_cutoff_utc"]) == cutoff
                 and _clock(unit["maximum_training_label_maturity"]) < cutoff, "saved OOF fitting policy/cutoff/maturity differs")
        _require(type(unit["training_rows"]) is int and unit["training_rows"] > 0
                 and type(unit["training_sessions"]) is int and 0 < unit["training_sessions"] <= len(fold.train),
                 "saved OOF training counts differ")
        for field in ("training_decision_ids_sha256", "training_weights_sha256"):
            _require(isinstance(unit[field], str) and len(unit[field]) == 64 and set(unit[field]) <= set("0123456789abcdef"),
                     "saved OOF training identity/weight hash malformed")
        train_parent = parent.loc[parent.session_date_et.isin(fold.train) & parent.feature_eligible]
        _require(unit["training_rows"] <= len(train_parent)
                 and set(unit["training_security_ids"]).issubset(set(train_parent.security_id)), "saved OOF training population differs")
        verify_unit(directory, manifest_sha256=unit_pin.sha256, request_sha256=manifest["request_sha256"], unit=unit)
        for name, digest in unit_manifest["files"].items():
            _pin(root, SourcePin(path=(directory / name).relative_to(root).as_posix(), sha256=digest), pins)
        prediction_path = directory / "predictions.parquet"
        available = set(parquet.ParquetFile(prediction_path).schema.names)
        _require(set(_SCORE_COLUMNS).issubset(available), "saved OOF score metadata missing")
        shared = [name for name in _OPTIONAL_SHARED if name in available and name in parent]
        scores = _metadata(pd.read_parquet(prediction_path, columns=[*_SCORE_COLUMNS, *shared]))
        _booleans(scores, tuple(name for name in ("feature_eligible", "scope_eligible", "security_holdout", "cross_section_eligible",
                                                "training_eligible", "production_eligible", "stock_source_admitted",
                                                "fixed_comparisons_complete") if name in scores))
        _require(len(scores) == unit["scoring_rows"] and json_sha256(sorted(scores.decision_id)) == unit["scoring_decision_ids_sha256"]
                 and set(scores.session_date_et) == set(fold.score) and int(scores.scope_eligible.sum()) == unit["scope_eligible_rows"]
                 and scores.scope_eligible.equals(scores.feature_eligible)
                 and scores.security_id.isin(holdouts).equals(scores.security_holdout)
                 and scores.decision_time_utc.ge(cutoff).all(), "saved OOF score population/scope differs")
        predicted = scores.predicted_excess_return
        _require(predicted.notna().equals(scores.scope_eligible) and np.isfinite(predicted[scores.scope_eligible]).all()
                 and scores.prediction_status.eq(np.where(scores.scope_eligible, "research_prediction", "outside_eligible_scope")).all(),
                 "saved OOF predictions/status differ from eligible scope")
        expected = parent.loc[parent.session_date_et.isin(fold.score)].sort_values("decision_id").reset_index(drop=True)
        scores = scores.sort_values("decision_id").reset_index(drop=True)
        compare = [*_IDENTITIES, "feature_eligible", *shared]
        try:
            assert_frame_equal(scores[compare], expected[compare], check_dtype=False, check_exact=True)
        except AssertionError as exc:
            raise DataReadinessError("saved OOF and independent parent identities/eligibility differ") from exc
        joined = expected.merge(scores[["decision_id", "scope_eligible", "predicted_excess_return", "prediction_status"]],
                                on="decision_id", validate="one_to_one")
        joined["fold"] = fold.number
        pieces.append(joined)
    return _SavedScores(pd.concat(pieces, ignore_index=True), score_days, score_days[:-10],
                        request["source_files"], manifest["request_sha256"], configuration_provenance)


def _select(rows: pd.DataFrame, score_sessions: tuple[str, ...], entry_sessions: tuple[str, ...], strategy: StrategyContract
            ) -> tuple[pd.DataFrame, pd.DataFrame, list[dict[str, Any]]]:
    complete = rows.copy()
    _booleans(complete, ("feature_eligible", "cross_section_eligible", "scope_eligible"))
    complete["terminal_excluded"] = ~complete.session_date_et.map(date.isoformat).isin(entry_sessions)
    eligible = complete.feature_eligible & complete.cross_section_eligible & complete.scope_eligible
    candidates = complete.loc[eligible & ~complete.terminal_excluded]
    settings = strategy.swing
    selected = select_constrained_swing_portfolio(candidates, maximum_trades=settings.maximum_trades_per_decision,
        target_maximum_sector_weight=settings.target_maximum_sector_weight, hard_maximum_sector_weight=settings.hard_maximum_sector_weight,
        minimum_distinct_sectors=settings.minimum_distinct_sectors_for_selection, score_column="predicted_excess_return")
    complete["selected"] = complete.decision_id.isin(selected.decision_id)
    counts = []
    for day in score_sessions:
        mask = complete.session_date_et.eq(date.fromisoformat(day))
        counts.append({"session": day, "rows": int(mask.sum()), "scope_eligible_rows": int((mask & complete.scope_eligible).sum()),
                       "selection_eligible_rows": int((mask & eligible).sum()), "selected_rows": int((mask & complete.selected).sum()),
                       "terminal_excluded_rows": int((mask & complete.terminal_excluded).sum())})
    return complete, selected, counts


def _unsimulated_lot(lot: FundedLotInput, valuation_sessions: tuple[str, ...]) -> HoldingSpecification:
    """Retain the canonical sale; forbid only generated settlement mechanics."""
    spec, result = lot.specification, lot.exit_result
    _require(spec is not None and result is not None and result.simulation is not None,
             "funded lot lacks its canonical exit replay")
    assert spec is not None and result is not None and result.simulation is not None
    generated_events = set(result.simulation.generated_event_ids)
    generated_marks = set(result.simulation.generated_mark_keys)
    _require(not any(event.event_id in generated_events for event in spec.events)
             and not any((mark.position_id, mark.mark_at) in generated_marks for mark in spec.marks),
             "funded lot already contains generated settlement")
    original = unsimulated_managed_specification(result)
    calendar = xcals.get_calendar("XNYS")
    ends = tuple(calendar.session_close(day).to_pydatetime() for day in valuation_sessions
                 if day >= spec.initial_entry_timestamp.date().isoformat())
    # The source provider extends valuation after the ten-session exit window.
    # All source facts and the sale must still equal the canonical reconstruction.
    _require(spec.model_dump(exclude={"session_end_timestamps"})
             == original.model_dump(exclude={"session_end_timestamps"})
             and spec.session_end_timestamps == ends
             and ends[:len(original.session_end_timestamps)] == original.session_end_timestamps,
             "funded lot differs from canonical exit or full valuation calendar")
    return spec


def publish_saved_oof_policy_evaluation(
    *, root: Path, run_manifest: SourcePin, feature_publication: SourcePin, target_config: SourcePin,
    strategy_contract: SourcePin, research_contract: SourcePin, learner: Learner, exit_policy: ExitPolicy,
    evidence: CorporateActionEvidence, output: Path,
    historical_configuration_evidence: SourcePin | None = None,
) -> dict[str, Any]:
    """One immutable learner/policy result inside the provider's sole source lease.

    Only small request/config metadata is handled before entering that context.
    The provider invokes selection_loader after acquiring its canonical lease;
    the context remains open through the final report and manifest publication.
    """
    root, output = root.resolve(), inside(root.resolve(), output)
    _require(output.parent == root / "data/research" and not output.exists(), "OOF output must be a new research directory")
    package = Path(__file__).resolve().parents[1]
    implementations = {f"src/market_predictor/{name}": file_sha256(package / name) for name in IMPLEMENTATION_PATHS}
    request = {"schema": SCHEMA, "run_manifest": run_manifest.model_dump(mode="json"),
               "feature_publication": feature_publication.model_dump(mode="json"), "target_config": target_config.model_dump(mode="json"),
               "strategy_contract": strategy_contract.model_dump(mode="json"),
               "research_contract": research_contract.model_dump(mode="json"),
               "historical_configuration_evidence": (historical_configuration_evidence.model_dump(mode="json")
                    if historical_configuration_evidence is not None else None),
               "learner": learner, "exit_policy": exit_policy, "action_projection_sha256": evidence.projection_sha256,
               "action_audit_sha256": evidence.audit_sha256, "implementation_files": implementations,
               "selection": "feature_and_cross_section_and_temporal_scope_rank_without_target_or_score_threshold",
               "serving_eligible": False, "promotion_eligible": False}
    output.mkdir()
    write_json_object(output / "_request.json", request)
    request_sha256 = file_sha256(output / "_request.json")
    pins = dict(implementations)
    strategy = load_strategy_contract(_pin(root, strategy_contract, pins))
    research = load_swing_research_contract(_pin(root, research_contract, pins))
    research.assert_strategy_matches(strategy)
    _require(exit_policy in research.exit_policies and learner in research.learner_families,
             "OOF learner/exit lies outside frozen research")
    _pin(root, target_config, pins)
    state: dict[str, Any] = {}

    def selection_loader() -> tuple[pd.DataFrame, tuple[str, ...]]:
        _require(not state, "OOF selection loader invoked more than once")
        _guard()
        evidence.recheck(root)
        for name, digest in evidence.source_files.items():
            _pin(root, SourcePin(path=name, sha256=digest), pins)
        _recheck(root, pins)
        saved = _saved_scores(root=root, run_manifest=run_manifest, feature_publication=feature_publication,
            strategy_pin=strategy_contract, research_pin=research_contract, strategy=strategy, learner=learner, pins=pins,
            historical_configuration_evidence=historical_configuration_evidence)
        decisions, selected, counts = _select(saved.rows, saved.score_sessions, saved.entry_sessions, strategy)
        state.update(saved=saved, decisions=decisions, selected=selected, counts=counts)
        return selected, saved.entry_sessions

    with verified_funded_policy_inputs(root=root, target_config=target_config, evidence=evidence, strategy=strategy,
        selection_loader=selection_loader, exit_policy=exit_policy) as provider:
        _require(bool(state), "funded provider did not load the bound OOF selection")
        saved, decisions, selected, counts = state["saved"], state["decisions"], state["selected"], state["counts"]
        _require(provider.config == target_config and provider.policy.research_contract == research_contract
                 and provider.research.sha256() == research.sha256()
                 and provider.valuation_sessions == swing_valuation_sessions(saved.entry_sessions, 10)
                 and provider.valuation_sessions[-1] == "2024-05-28"
                 and set(provider.selected_ids) == set(selected.decision_id), "funded provider calendar/selection/research differs")
        holdings: list[HoldingSpecification] = []
        benchmarks: dict[str, HoldingSpecification] = {}
        gaps: list[dict[str, Any]] = []
        for decision in selected.itertuples(index=False):
            _guard()
            lot = provider.lot(decision.decision_id)
            _require(lot.decision_id == decision.decision_id and lot.canonical_security_id == decision.security_id,
                     "funded lot changed selected identity")
            if lot.missing_reasons or lot.specification is None or lot.exit_result is None or lot.decision_atr is None:
                gaps.append({"decision_id": decision.decision_id, "security_id": decision.security_id,
                             "reasons": list(lot.missing_reasons) or ["funded_lot_or_exit_unavailable"]})
            else:
                holdings.append(_unsimulated_lot(lot, provider.valuation_sessions))
        expected_benchmarks = {"SPY", "QQQ", *selected.primary_benchmark.astype(str)}
        _require(set(provider.benchmark_tickers) == expected_benchmarks, "funded benchmark identities differ")
        for ticker in sorted(expected_benchmarks):
            _guard()
            lot = provider.benchmark(ticker)
            if lot.missing_reasons or lot.specification is None:
                gaps.append({"benchmark": ticker, "reasons": list(lot.missing_reasons) or ["benchmark_valuation_unavailable"]})
            else:
                benchmarks[ticker] = lot.specification
        if gaps:
            accounting: dict[str, Any] = {"status": "valuation_unavailable", "comparisons": None,
                "eligible": False, "missing_inputs": gaps, "economic_conditions_passed": False}
        else:
            ledger_selected = selected.copy()
            ledger_selected["session_date_et"] = ledger_selected.session_date_et.map(date.isoformat)
            accounting = evaluate_event_aware_swing_accounting(ledger_selected, holdings, benchmarks,
                config=_LedgerConfig(research.horizon_sessions, research.base_round_trip_cost_bps,
                                     strategy.swing.maximum_trades_per_decision),
                strategy_contract=strategy, research_contract=research, session_calendar=saved.entry_sessions,
                simulation=provider.simulation)
        for name, digest in provider.source_files.items():
            _pin(root, SourcePin(path=name, sha256=digest), pins)
        decisions.to_parquet(output / "decisions.parquet", index=False)
        selected.to_parquet(output / "selected.parquet", index=False)
        evidence.recheck(root)
        _recheck(root, pins)
        _require(file_sha256(output / "_request.json") == request_sha256, "OOF evaluation request changed")
        report = {"schema": SCHEMA, "request_sha256": request_sha256, "status": accounting["status"], "accounting": accounting,
                  "score_sessions": list(saved.score_sessions), "entry_sessions": list(saved.entry_sessions), "sessions": counts,
                  "selected_rows": len(selected), "terminal_excluded_rows": int(decisions.terminal_excluded.sum()),
                  "historical_training_request_sha256": saved.request_sha256,
                  "historical_training_source_provenance": saved.historical_source_files,
                  "historical_configuration_verification": saved.configuration_provenance,
                  "source_files": dict(sorted(pins.items())), "serving_eligible": False, "promotion_eligible": False,
                  "training_eligible": False, "economic_eligible": False, "source_admitted": False}
        write_json_object(output / "report.json", report)
        manifest = {"schema": SCHEMA, "status": "complete_research_only", "request_sha256": request_sha256,
                    "artifacts": {name: file_sha256(output / name)
                        for name in ("_request.json", "report.json", "decisions.parquet", "selected.parquet")},
                    "serving_eligible": False, "promotion_eligible": False, "economic_eligible": False}
        write_json_object(output / "_manifest.json", manifest)
        return report | {"manifest_sha256": file_sha256(output / "_manifest.json")}
