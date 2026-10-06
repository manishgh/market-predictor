"""Current-code replay of one pinned original research snapshot; no source swap.

This receipt neither changes existing consumer gates nor admits transport, fitting
or serving. Original producer hashes are provenance, never executable imports.
"""
from __future__ import annotations

import tomllib
from datetime import date
from numbers import Real
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_features import ResearchFeaturePolicy
from market_predictor.swing.contracts.return_feature_profiles import ReturnRelationshipSources
from market_predictor.swing.contracts.return_relationship_reuse import RelationshipReusePolicy
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy
from market_predictor.swing.datasets.relationship_historical_evidence import (
    historical_month,
    inspect_original_source_comparison,
    original_stock_bars,
)
from market_predictor.swing.datasets.return_relationship_integrity import (
    check_files,
    current_implementation,
    guard,
    pins,
    read_object,
)
from market_predictor.swing.datasets.return_relationship_reuse import (
    CLOSED,
    EXPECTED_ROWS,
    _historical_group,
    _historical_spy,
    _publish_report,
    compare_frames,
    inspect_reuse_evidence,
)
from market_predictor.swing.datasets.return_relationship_rows import ADDITIONS, IDENTITY_COLUMNS
from market_predictor.swing.features.adjusted_source import _bars
from market_predictor.swing.features.panel import swing_model_feature_columns
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.features.return_relationships import build_return_relationship_profile

SCHEMA = "market_predictor.original_relationship_replay"
SOURCE_START, SOURCE_END = "2018-05-29", "2024-05-28"


def _implementation(root: Path) -> dict[str, str]:
    # The existing AST closure includes historical inspection and numerical owners.
    # Pin this new research orchestrator separately; scratch execution is refused.
    path = Path(__file__).resolve()
    if path != root / "src/market_predictor/research/original_relationship_replay.py":
        raise DataReadinessError("original replay must execute from its bound repository")
    return pins(root, current_implementation(root), {path.relative_to(root).as_posix(): file_sha256(path)})


def _assign_original_groups(population: pd.DataFrame, inventory: dict[str, Any]) -> dict[str, pd.DataFrame]:
    """Preserve saved security/ticker ownership; never fabricate current query IDs."""
    result: dict[str, pd.DataFrame] = {}
    seen: set[str] = set()
    for key, item in sorted(inventory.items()):
        if (item["kind"] not in {"combined", "corrected"}
                or key != json_sha256([item["security_id"], item["source_group"]])):
            raise DataReadinessError("original source group identity differs")
        selected = population.security_id.eq(item["security_id"])
        if item["kind"] == "combined":
            selected &= population.parent_ticker.eq(item["source_group"])
        rows = population.loc[selected].reset_index(drop=True)
        identifiers = set(rows.decision_id)
        if (rows.empty or rows.decision_id.duplicated().any() or identifiers & seen
                or len(rows) != item["rows"] or json_sha256(sorted(identifiers)) != item["decision_ids_sha256"]):
            raise DataReadinessError("original source ownership is ambiguous or its decision population changed")
        seen.update(identifiers)
        result[key] = rows
    if len(seen) != len(population) or seen != set(population.decision_id):
        raise DataReadinessError("original source inventory omits decisions")
    return result


def _validate_physical_history(frame: pd.DataFrame, *, benchmark: bool) -> None:
    """Validate full usable OHLCV and exact physical clocks, beyond formula inputs."""
    _bars(frame)
    for name in ("open", "high", "low", "close", "volume"):
        if not frame[name].map(lambda value: isinstance(value, Real) and not isinstance(value, (bool, np.bool_))).all():
            raise DataReadinessError("original OHLCV must contain real numbers, never strings or booleans")
    if not frame.source.eq("alpaca").all() or not frame.availability_policy.eq("market_interval_close").all():
        raise DataReadinessError("original physical provider or availability policy differs")
    if benchmark and (frame.empty or not frame.ticker.eq("SPY").all()):
        raise DataReadinessError("original SPY is absent or substituted")
    calendar = xcals.get_calendar("XNYS")
    for name in ("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc"):
        if frame[name].isna().any() or not frame[name].map(lambda value: pd.Timestamp(value).tzinfo is not None).all():
            raise DataReadinessError("original physical clocks must be present and timezone aware")
    days = frame.session_date_et
    for name, clock in (("bar_start_utc", calendar.session_open), ("bar_end_utc", calendar.session_close)):
        expected = pd.to_datetime([clock(pd.Timestamp(day)) for day in days], utc=True)
        if not pd.to_datetime(frame[name], utc=True).eq(expected).all():
            raise DataReadinessError("original interval differs from the complete exchange session")


def _replay(*, root: Path, config: SourcePin, failed_comparison: SourcePin) -> dict[str, Any]:
    guard(90.0)
    policy = RelationshipReusePolicy.model_validate(read_object(inside(root, config.path), config.sha256))
    if policy.predictor_failure_facts is not None:
        raise DataReadinessError("original replay cannot invent current predictor failures")
    implementation = _implementation(root)
    evidence = inspect_reuse_evidence(root, policy)
    failed = inspect_original_source_comparison(root, failed_comparison, evidence.parent, evidence.relationships)
    if failed["policy"] != policy.model_dump(mode="json"):
        raise DataReadinessError("original replay request differs from its measured source comparison")
    files = pins(root, evidence.files, implementation,
        {config.path: config.sha256, failed_comparison.path: failed_comparison.sha256})
    feature = ResearchFeaturePolicy.model_validate(tomllib.loads(inside(root, policy.feature_config.path).read_text(encoding="utf-8")))
    if feature.strategy_contract != policy.strategy_contract:
        raise DataReadinessError("original replay strategy and correction configuration differ")
    correction = feature.outcome_source_config
    outcome = load_corrected_outcome_policy(root, Path(correction.path), correction.sha256)
    files = pins(root, files, {correction.path: correction.sha256})
    contract = load_strategy_contract(inside(root, policy.strategy_contract.path))
    names = tuple(swing_model_feature_columns(contract=contract, catalyst=False))
    request = evidence.relationships.request
    sources = ReturnRelationshipSources.model_validate(request["sources"])
    if (len(names) != 120 or sources.baseline_authority_sha256 != policy.historical_parent_publication.sha256
            or sources.availability_semantics != "historical_proxy" or sources.availability_policy_id != "market_interval_close"
            or sources.price_adjustment != "all" or request["source_start"] != SOURCE_START or request["source_end"] != SOURCE_END):
        raise DataReadinessError("original source basis, warmup or numerical contract differs")
    months: dict[str, Any] = {}
    failures: list[dict[str, Any]] = []
    pieces: list[pd.DataFrame] = []
    clocks: dict[str, str] = {}
    for month, record in sorted(evidence.parent.manifest["months"].items()):
        guard(90.0)
        child = record["profiles"]["technical_market"]
        if tuple(child["model_columns"]) != names or (clocks and child["availability_columns"] != clocks):
            raise DataReadinessError("original baseline feature ordering or clocks change between months")
        clocks = dict(child["availability_columns"])
        columns = list(dict.fromkeys((*IDENTITY_COLUMNS, *names, *clocks.values())))
        baseline = historical_month(evidence.parent, month, [*columns, "feature_profile"])
        original = historical_month(evidence.relationships, month, [*columns, "feature_profile"])
        if not baseline.feature_profile.eq("technical_market").all() or not original.feature_profile.eq("technical_relationships").all():
            raise DataReadinessError("original monthly profile differs")
        result = compare_frames(baseline, original, columns, scope=f"inherited_predictors/{month}", check_dtype=True)
        if not result["equal"]:
            failures.append(result)
        months[month] = {"rows": len(baseline), "decision_ids_sha256": json_sha256(sorted(baseline.decision_id)),
            "inherited_predictors_equal": result["equal"]}
        pieces.append(baseline.loc[:, IDENTITY_COLUMNS])
        del baseline, original
        release_process_memory()
    population = pd.concat(pieces, ignore_index=True)
    del pieces
    if (len(population) != EXPECTED_ROWS or population.decision_id.duplicated().any()
            or json_sha256(sorted(population.decision_id)) != failed["decision_ids_sha256"]
            or not population.session_date_et.between(date(2019, 7, 9), date(2024, 5, 28)).all()):
        raise DataReadinessError("original replay population differs from the frozen comparison")
    inventory = request["stock_inventory"]
    groups = _assign_original_groups(population, inventory)
    spy = _historical_spy(root, evidence, files)
    _validate_physical_history(spy, benchmark=True)
    sessions = tuple(stamp.date() for stamp in xcals.get_calendar("XNYS").sessions_in_range(SOURCE_START, SOURCE_END))
    comparisons: dict[str, Any] = {}
    for key, decisions in groups.items():
        guard(90.0)
        item = inventory[key]
        bars = original_stock_bars(root, evidence.relationships, key, files)
        _validate_physical_history(bars, benchmark=False)
        # Explicit already-pinned correction intervals, never alias inference.
        if item["kind"] == "corrected":
            for rule in outcome.decision_corrections:
                if rule.security_id == item["security_id"]:
                    bars.loc[bars.session_date_et.between(rule.first_session, rule.last_session), "ticker"] = rule.ticker
        columns = list(dict.fromkeys((*IDENTITY_COLUMNS, *names, *clocks.values(), "feature_profile")))
        baseline = _historical_group(evidence.parent, decisions, columns)
        saved = _historical_group(evidence.relationships, decisions, [*IDENTITY_COLUMNS, *ADDITIONS])
        replay = build_return_relationship_profile(expected_decisions=baseline.loc[:, IDENTITY_COLUMNS],
            baseline=ResearchFeaturePartition(baseline, names, clocks, {}), stock_bars=bars, spy_bars=spy,
            history_sessions=sessions, contract=contract, sources=sources).rows
        result = compare_frames(saved, replay, [*IDENTITY_COLUMNS, *ADDITIONS], scope=f"original_additions/{key}", check_dtype=True)
        comparisons[key] = {"rows": len(decisions), "decision_ids_sha256": item["decision_ids_sha256"],
            "usable_history_rows": len(bars), "quarantine": item["quarantine"], "comparison": result}
        if not result["equal"]:
            failures.append(result)
        del bars, baseline, saved, replay
        release_process_memory()
    report = {"schema": SCHEMA, "status": "passed_original_snapshot_replay" if not failures else "failed_differences",
        "config": config.model_dump(mode="json"), "failed_current_comparison": failed_comparison.model_dump(mode="json"),
        "source_files": files, "current_implementation_files": implementation,
        "historical_implementation_files": evidence.historical_implementation_files,
        "historical_implementation_disposition": "immutable_provenance_not_executed",
        "rows": len(population), "decision_ids_sha256": json_sha256(sorted(population.decision_id)),
        "months": months, "groups": comparisons, "differences": failures, "source_basis": request["source_basis"],
        "original_sources": sources.model_dump(mode="json"), "source_window_bounds": [SOURCE_START, SOURCE_END],
        "original_stock_inventory_sha256": json_sha256(inventory), "saved_rows_rewritten": False,
        "inherited_predictors_equal": all(item["inherited_predictors_equal"] for item in months.values()),
        "additions_source_replayed": not failures, "usable_ohlcv_validated": True,
        "targets_read_or_replayed": False, "baseline_price_features_replayed": False,
        "raw_news_aggregates_replayed": False, "current_archive_equivalence_proven": False,
        "provider_transport_admitted": False, "historical_first_seen_proven": False,
        "quarantined_source_inputs_admitted": False, "consumer_admission_changed": False, **CLOSED}
    check_files(root, files)
    if _implementation(root) != implementation:
        raise DataReadinessError("original replay implementation changed during verification")
    return report


def replay_original_relationships(*, root: Path, config: Path, expected_config_sha256: str,
    failed_comparison: SourcePin, output: Path,
) -> dict[str, Any]:
    """Single-lease standalone operation; publish one fresh receipt, never rows."""
    root = root.resolve()
    output = inside(root, output)
    if not output.is_relative_to(root / "data/reports") or output.exists():
        raise DataReadinessError("original replay requires a fresh report path")
    config_pin = SourcePin(path=inside(root, config).relative_to(root).as_posix(), sha256=expected_config_sha256)
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("original-relationship-replay", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        report = _replay(root=root, config=config_pin, failed_comparison=failed_comparison)
        guard(90.0)
        check_files(root, report["source_files"])
        _publish_report(output, report)
    return report


def verify_original_relationship_receipt(*, root: Path, receipt: SourcePin) -> dict[str, Any]:
    """Reproduce the entire receipt; a successful JSON declaration is insufficient."""
    root = root.resolve()
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("verify-original-relationship-replay", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        saved = read_object(inside(root, receipt.path), receipt.sha256)
        if saved.get("schema") != SCHEMA or saved.get("status") != "passed_original_snapshot_replay":
            raise DataReadinessError("original receipt is not a passed replay")
        actual = _replay(root=root, config=SourcePin.model_validate(saved["config"]),
            failed_comparison=SourcePin.model_validate(saved["failed_current_comparison"]))
        if json_sha256(actual) != json_sha256(saved):
            raise DataReadinessError("original receipt does not reproduce exactly")
        check_files(root, {receipt.path: receipt.sha256})
    return actual
