"""Bounded diagnostics for the fixed-horizon research training boundary."""
from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.edge_rebuild.temporal_manifest import build_temporal_schedule, load_temporal_manifest_config
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.resources import assert_memory_budget, release_process_memory
from market_predictor.swing.contracts.research import load_swing_research_contract
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.labels.fixed_horizon_readiness import CONTEXT_COLUMNS, RETURN_COLUMNS, fixed_horizon_readiness

IMPLEMENTATION_PATHS = (
    "research/swing_training_readiness.py", "swing/labels/fixed_horizon_readiness.py",
    "swing/contracts/training_readiness.py", "swing/contracts/research.py",
    "swing/features/panel.py", "edge_rebuild/temporal_manifest.py", "modeling/strategy_contract.py",
    "canonical/store.py", "canonical/cutoffs.py",
)


def _guard() -> None:
    assert_memory_budget(stage="swing training readiness", hard_budget_gib=5.0, headroom_gib=0.75)
    assert_system_memory_available(minimum_available_gib=2.0)


def _verify(root: Path, pins: dict[str, str]) -> None:
    for name, digest in pins.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"training readiness evidence changed: {name}")


def audit_swing_training_readiness(*, root: Path, config: Path, config_sha256: str, output: Path) -> dict[str, Any]:
    """Inspect saved rows; never mutate source flags, fit models or authorize trading."""
    root = root.resolve()
    config, output = inside(root, config), inside(root, output)
    if not output.is_relative_to(root / "data/reports"):
        raise DataReadinessError("training readiness output must be below data/reports")
    if output.exists():
        raise FileExistsError(f"immutable report already exists: {output}")
    with heavy_job_lease("audit-swing-training-readiness", runtime_dir=root / "data/runtime", config_path=config):
        _guard()
        policy = TrainingReadinessPolicy.model_validate(pinned_object(config, config_sha256))
        return _audit(root, policy, config, config_sha256, output)


def _audit(root: Path, policy: TrainingReadinessPolicy, config: Path, config_sha256: str, output: Path) -> dict[str, Any]:
    pins = {config.relative_to(root).as_posix(): config_sha256}
    for source_pin in (policy.publication, policy.saved_row_verification, policy.research_contract,
            policy.strategy_contract, policy.temporal_contract):
        path = inside(root, source_pin.path)
        key = path.relative_to(root).as_posix()
        if key in pins and pins[key] != source_pin.sha256:
            raise DataReadinessError("conflicting readiness input pins")
        pins[key] = source_pin.sha256
    for name in IMPLEMENTATION_PATHS:
        source = Path(__file__).parents[1] / name
        pins[source.relative_to(root).as_posix()] = file_sha256(source)
    _verify(root, pins)
    publication_path = inside(root, policy.publication.path)
    manifest = pinned_object(publication_path, policy.publication.sha256)
    request_path = publication_path.parent / "_request.json"
    request = pinned_object(request_path, manifest["request_sha256"])
    pins[request_path.relative_to(root).as_posix()] = manifest["request_sha256"]
    provenance = request.get("source_files")
    if not isinstance(provenance, dict):
        raise DataReadinessError("publication provenance must bind source policies")
    for source_pin in (policy.research_contract, policy.strategy_contract):
        name = inside(root, source_pin.path).relative_to(root).as_posix()
        if provenance.get(name) != source_pin.sha256:
            raise DataReadinessError("readiness policy differs from publication provenance")
    if (manifest.get("schema") != "market_predictor.research_join" or manifest.get("status") != "complete_research_only"
            or manifest.get("training_eligible") is not False or manifest.get("promotion_eligible") is not False
            or manifest.get("exclusions_added") != [] or request.get("schema") != "market_predictor.research_join_request"
            or request.get("historical_first_seen_proven") is not False or request.get("managed_outcomes_available") is not False
            or request.get("profiles") != ["technical_market", "catalyst_full"]
            or type(manifest.get("rows")) is not int or manifest["rows"] != request.get("rows")):
        raise DataReadinessError("unsupported or incomplete research publication")
    receipt = pinned_object(inside(root, policy.saved_row_verification.path), policy.saved_row_verification.sha256)
    if (receipt.get("manifest_sha256") != policy.publication.sha256 or receipt.get("status") != "passed"
            or receipt.get("scope") != "published_join_population_clocks_original_targets"
            or receipt.get("matched_profile_population") is not True or receipt.get("original_outcome_values_exact") is not True
            or receipt.get("outcome_filtered_rows") != 0 or receipt.get("unique_decisions") != manifest["rows"]
            or receipt.get("training_eligible") is not False or receipt.get("promotion_eligible") is not False):
        raise DataReadinessError("saved-row verification does not bind this publication")
    research = load_swing_research_contract(inside(root, policy.research_contract.path))
    strategy = load_strategy_contract(inside(root, policy.strategy_contract.path))
    research.assert_strategy_matches(strategy)
    temporal = load_temporal_manifest_config(inside(root, policy.temporal_contract.path))
    schedule = build_temporal_schedule(temporal)
    sessions = schedule.folds[0].train_sessions
    if (sessions[0].isoformat() != research.decision_start or len(sessions) != temporal.initial_fit_expected_sessions
            or temporal.label_horizon_sessions != research.horizon_sessions
            or temporal.warmup_sessions != strategy.swing.minimum_warmup_sessions):
        raise DataReadinessError("research, strategy and temporal scopes differ")
    calendar = xcals.get_calendar(temporal.calendar)
    maturity = {day: calendar.session_close(calendar.sessions_window(pd.Timestamp(day), research.horizon_sessions + 1)[-1])
        for day in sessions}
    fit_end = calendar.session_close(sessions[-1].isoformat())
    months = sorted({day.isoformat()[:7] for day in sessions})
    if sorted(manifest["months"]) != months or receipt.get("months") != len(months):
        raise DataReadinessError("publication does not cover the frozen initial-fit month inventory")
    totals: dict[str, Counter[str]] = {}
    missing: dict[str, Counter[str]] = {}
    groups: dict[str, dict[str, Counter[str]]] = {}
    monthly: dict[str, Any] = {}
    observed: set[Any] = set()
    securities: set[str] = set()
    for month in months:
        _guard()
        record = manifest["months"][month]
        if set(record["profiles"]) != {"technical_market", "catalyst_full"}:
            raise DataReadinessError("profile inventory differs")
        monthly[month] = {}
        reference: pd.DataFrame | None = None
        masks: dict[str, pd.DataFrame] = {}
        for profile, child in record["profiles"].items():
            columns = tuple(swing_model_feature_columns(contract=strategy, catalyst=profile == "catalyst_full"))
            if (child["model_columns"] != list(columns) or child["path"] != f"{month}/{profile}.parquet"
                    or child["audit"].get("training_eligible") is not False
                    or child["audit"].get("promotion_eligible") is not False):
                raise DataReadinessError("ordered model feature contract or child scope differs")
            clocks = {name: child["availability_columns"][name] for name in columns}
            if any(clocks[name] != f"available_at_{name}" for name in columns):
                raise DataReadinessError("published model-clock mapping differs")
            path = inside(publication_path.parent, child["path"])
            child_pins = {path.relative_to(root).as_posix(): child["sha256"],
                manifest_path_for(path).relative_to(root).as_posix(): child["manifest_sha256"]}
            pins.update(child_pins)
            _verify(root, child_pins)
            projected = list(dict.fromkeys((*CONTEXT_COLUMNS, *RETURN_COLUMNS, *columns, *clocks.values())))
            frame, sidecar = load_canonical_artifact(path, expected_type="swing_research_join", allow_research=True, columns=projected)
            if (sidecar["inputs"] != {"request_sha256": manifest["request_sha256"]}
                    or sidecar["production_ready"] is not False or len(frame) != record["rows"]
                    or json_sha256(sorted(frame.decision_id)) != record["decision_ids_sha256"]):
                raise DataReadinessError("child lineage or decision population differs")
            if any(day.isoformat()[:7] != month for day in frame.session_date_et):
                raise DataReadinessError("decision assigned to wrong month")
            state = fixed_horizon_readiness(frame, model_columns=columns, availability_columns=clocks,
                maturity_by_session=maturity, fit_end=fit_end, minimum_warmup=temporal.warmup_sessions,
                round_trip_cost=research.base_round_trip_cost_bps / 10000.0)
            identity = frame.loc[:, [*CONTEXT_COLUMNS, *RETURN_COLUMNS]]
            if reference is not None:
                try:
                    pd.testing.assert_frame_equal(identity, reference, check_exact=True)
                except AssertionError as error:
                    raise DataReadinessError("profiles differ in population, target or decision eligibility") from error
            else:
                reference = identity.copy()
                observed.update(frame.session_date_et)
                securities.update(frame.security_id)
            summary = {"rows": len(frame), **{name: int(state[name].sum()) for name in state}}
            monthly[month][profile] = summary
            totals.setdefault(profile, Counter()).update(summary)
            missing.setdefault(profile, Counter()).update({name: int(frame[name].isna().sum()) for name in columns})
            profile_groups = groups.setdefault(profile, {})
            for sector, positions in frame.groupby("sector", sort=True).indices.items():
                selected = state.iloc[positions]
                profile_groups.setdefault(f"{month[:4]}:{sector}", Counter()).update(
                    {"rows": len(selected), **{name: int(selected[name].sum()) for name in selected}})
            masks[profile] = state
            _verify(root, child_pins)
            del frame, identity
            release_process_memory()
        monthly[month]["paired_complete_case_supervision"] = int(
            (masks["technical_market"].complete_case_supervision & masks["catalyst_full"].complete_case_supervision).sum())
        del masks, reference
    if observed != set(sessions) or any(total["rows"] != manifest["rows"] for total in totals.values()):
        raise DataReadinessError("initial-fit session or row population differs")
    for profile, total in totals.items():
        prior = receipt["profiles"][profile]
        if (prior["rows"] != total["rows"] or prior["feature_eligible"] != total["feature_eligible"]
                or prior["complete_model_rows"] != total["model_inputs_complete"] or prior["clock_violations"] != 0):
            raise DataReadinessError("independent saved-row receipt disagrees with readiness audit")
    _guard()
    _verify(root, pins)
    report = {"schema": "market_predictor.swing_training_readiness", "status": "diagnostic_complete",
        "scope": policy.scope, "source_files": pins, "cohort_sha256": request["cohort_sha256"],
        "decision_start": sessions[0].isoformat(), "decision_end": sessions[-1].isoformat(),
        "sessions": len(sessions), "securities": len(securities), "publication_sha256": policy.publication.sha256,
        "forecast_target": research.forecast_target, "published_target_column": "spy_fixed_horizon_excess_return",
        "rows_removed": 0, "training_eligible": False, "promotion_eligible": False,
        "managed_evaluation_eligible": False, "historical_first_seen_proven": False,
        "profiles": {key: dict(value) for key, value in totals.items()}, "months": monthly,
        "missing_model_values": {key: dict(value) for key, value in missing.items()},
        "year_sector_counts": {key: {group: dict(counts) for group, counts in value.items()} for key, value in groups.items()},
        "interpretation": "Supervised-label and complete-case counts are diagnostics, not an opportunity selection or training permission."}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json_object(output, report)
    return {**report, "report_sha256": file_sha256(output)}
