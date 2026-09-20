"""Read only the explicitly admitted initial-fit technical research publication."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.edge_rebuild.temporal_manifest import build_temporal_schedule, load_temporal_manifest_config
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.research.swing_training_readiness import (
    IMPLEMENTATION_PATHS,
    RELATIONSHIP_IMPLEMENTATION_PATHS,
    _guard,
    _verify,
)
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_PROFILE
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets.return_relationship_verification import validate_return_relationship_receipt
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.labels.fixed_horizon_readiness import CONTEXT_COLUMNS, RETURN_COLUMNS, fixed_horizon_readiness
from market_predictor.swing.training.return_validation import security_transfer_mask


@dataclass
class ReturnInputs:
    features: pd.DataFrame
    metadata: pd.DataFrame
    sessions: tuple[date, ...]
    pins: dict[str, str]
    memory_policy: TrainingReadinessPolicy
    fit_cutoff: pd.Timestamp


def load_return_inputs(root: Path, policy: ReturnTrainingPolicy) -> ReturnInputs:
    report = pinned_object(inside(root, policy.readiness.path), policy.readiness.sha256)
    pins = dict(report["source_files"])
    if pins.get(policy.readiness_config.path) != policy.readiness_config.sha256:
        raise DataReadinessError("return request and readiness configuration differ")
    memory_policy = TrainingReadinessPolicy.model_validate(pinned_object(inside(root, policy.readiness_config.path),
        policy.readiness_config.sha256))
    _guard(memory_policy)
    _verify(root, pins)
    if memory_policy.published_profile != policy.published_profile:
        raise DataReadinessError("return request and readiness profile differ")
    relationship = policy.published_profile == RETURN_RELATIONSHIP_PROFILE
    verified = None
    if relationship:
        for name in (*IMPLEMENTATION_PATHS, *RELATIONSHIP_IMPLEMENTATION_PATHS):
            source = Path(__file__).parents[1] / name
            if pins.get(source.relative_to(root).as_posix()) != file_sha256(source):
                raise DataReadinessError("return relationship readiness lacks current executed-code pin")
        receipt_pin = memory_policy.saved_row_verification
        if pins.get(receipt_pin.path) != receipt_pin.sha256:
            raise DataReadinessError("relationship readiness omits independent saved-row receipt")
        receipt = pinned_object(inside(root, receipt_pin.path), receipt_pin.sha256)
        verified = validate_return_relationship_receipt(root, memory_policy.publication, receipt)
        if (report.get("published_profile") != policy.published_profile
                or report.get("profile_sha256") != verified.request["profile_sha256"]
                or report.get("model_columns") != list(verified.model_columns)
                or report.get("availability_columns") != verified.availability_columns
                or report.get("saved_row_verification_sha256") != receipt_pin.sha256
                or report.get("serving_eligible") is not False or report.get("additions_source_replayed") is not True
                or report.get("baseline_numerical_replayed") is not False
                or any(pins.get(name) != digest for name, digest in receipt["source_files"].items())):
            raise DataReadinessError("return relationship profile or current evidence differs")
    if (report.get("schema") != "market_predictor.swing_training_readiness" or report.get("status") != "diagnostic_complete"
            or report.get("scope") != "initial_fit_fixed_horizon_diagnostics" or report.get("rows_removed") != 0
            or report.get("training_eligible") is not False or report.get("promotion_eligible") is not False
            or report.get("managed_evaluation_eligible") is not False
            or report.get("published_target_column") != policy.target
            or report.get("publication_sha256") != memory_policy.publication.sha256
            or report.get("memory_policy") != {"maximum_system_used_percent": memory_policy.maximum_system_used_percent,
                "minimum_system_free_gib": None, "maximum_process_memory_gib": 5.0, "process_headroom_gib": 0.75}):
        raise DataReadinessError("unsupported return-training admission evidence")
    strategy = load_strategy_contract(inside(root, memory_policy.strategy_contract.path))
    research = load_swing_research_contract(inside(root, memory_policy.research_contract.path))
    research.assert_strategy_matches(strategy)
    temporal = load_temporal_manifest_config(inside(root, memory_policy.temporal_contract.path))
    schedule = build_temporal_schedule(temporal)
    sessions = schedule.folds[0].train_sessions
    if (len(sessions) != report["sessions"] or sessions[0].isoformat() != report["decision_start"]
            or sessions[-1].isoformat() != report["decision_end"] or sessions[-1] != temporal.initial_fit_end
            or sessions[0].isoformat() != research.decision_start
            or strategy.validation.unseen_ticker_holdout_fraction != policy.holdout_fraction
            or temporal.label_horizon_sessions != policy.embargo_sessions):
        raise DataReadinessError("return-training temporal or holdout boundary differs")
    names = (verified.model_columns if verified is not None else
        tuple(swing_model_feature_columns(contract=strategy, catalyst=False)))
    publication_path = inside(root, memory_policy.publication.path)
    manifest = pinned_object(publication_path, memory_policy.publication.sha256)
    request = pinned_object(publication_path.parent / "_request.json", manifest["request_sha256"])
    if report["cohort_sha256"] != request["cohort_sha256"]:
        raise DataReadinessError("return research population identity differs")
    rows = report["profiles"][policy.published_profile]["rows"]
    if type(rows) is not int or rows < 1 or rows * len(names) * 4 > 1024**3:
        raise MemoryBudgetError("return feature matrix exceeds bounded allocation")
    matrix = np.empty((rows, len(names)), dtype=np.float32)
    calendar = xcals.get_calendar(temporal.calendar)
    maturity = {day: calendar.session_close(calendar.sessions_window(pd.Timestamp(day), research.horizon_sessions + 1)[-1])
        for day in sessions}
    fit_end = calendar.session_close(sessions[-1].isoformat())
    pieces = []
    offset = 0
    months = sorted({day.isoformat()[:7] for day in sessions})
    if sorted(manifest["months"]) != months or sorted(report["months"]) != months:
        raise DataReadinessError("return training requires exact initial-fit months only")
    for month in months:
        _guard(memory_policy)
        record = manifest["months"][month]
        child = record["profiles"][policy.published_profile]
        if child["model_columns"] != list(names) or child["path"] != f"{month}/{policy.published_profile}.parquet":
            raise DataReadinessError("return model input order or month path differs")
        path = inside(publication_path.parent, child["path"])
        clocks = {name: f"available_at_{name}" for name in names}
        if any(child["availability_columns"].get(name) != clock for name, clock in clocks.items()):
            raise DataReadinessError("return input availability mapping differs")
        for source, digest in ((path, child["sha256"]), (manifest_path_for(path), child["manifest_sha256"])):
            if pins.get(source.relative_to(root).as_posix()) != digest or file_sha256(source) != digest:
                raise DataReadinessError("return input child not bound by readiness receipt")
        artifact_type = "swing_return_relationships" if relationship else "swing_research_join"
        frame, sidecar = load_canonical_artifact(path, expected_type=artifact_type, allow_research=True,
            columns=list(dict.fromkeys((*CONTEXT_COLUMNS, *RETURN_COLUMNS, *names, *clocks.values()))))
        if (sidecar["production_ready"] is not False or sidecar["inputs"] != {"request_sha256": manifest["request_sha256"]}
                or len(frame) != record["rows"] or json_sha256(sorted(frame.decision_id)) != record["decision_ids_sha256"]
                or any(day.isoformat()[:7] != month for day in frame.session_date_et)):
            raise DataReadinessError("return input population or lineage differs")
        state = fixed_horizon_readiness(frame, model_columns=names, availability_columns=clocks, maturity_by_session=maturity,
            fit_end=fit_end, minimum_warmup=temporal.warmup_sessions, round_trip_cost=research.base_round_trip_cost_bps / 10000.0)
        expected = report["months"][month][policy.published_profile]
        if any(int(state[name].sum()) != expected[name] for name in state):
            raise DataReadinessError("return input diagnostic masks differ from verified receipt")
        if offset + len(frame) > rows:
            raise DataReadinessError("return input row allocation exceeded")
        values = frame.loc[:, list(names)].to_numpy(dtype=np.float32)
        if np.isinf(values).any():
            raise DataReadinessError("return feature exceeds float32 representation")
        matrix[offset:offset + len(frame)] = values
        retained = frame.loc[:, [*CONTEXT_COLUMNS, *RETURN_COLUMNS]].copy()
        retained["fixed_horizon_supervision_available"] = state.fixed_horizon_supervision_available
        pieces.append(retained)
        offset += len(frame)
        del frame, values, state, retained
        release_process_memory()
    metadata = pd.concat(pieces, ignore_index=True)
    if (offset != rows or set(metadata.session_date_et) != set(sessions) or metadata.decision_id.duplicated().any()
            or metadata.security_id.nunique() != report["securities"]):
        raise DataReadinessError("return training population incomplete")
    metadata["security_holdout"] = security_transfer_mask(metadata.security_id, policy.holdout_fraction)
    if not metadata.security_holdout.any() or metadata.security_holdout.all():
        raise DataReadinessError("empty security-transfer partition")
    pins[policy.readiness.path] = policy.readiness.sha256
    _verify(root, pins)
    _guard(memory_policy)
    return ReturnInputs(pd.DataFrame(matrix, columns=names, copy=False), metadata, sessions, pins, memory_policy,
        metadata.decision_time_utc.max())
