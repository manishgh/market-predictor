"""Independent bounded saved-row checks for the research-only derivative."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pyarrow.dataset as pds

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError, MemoryBudgetError
from market_predictor.core.system_memory import assert_system_memory_available, system_memory_snapshot
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import assert_memory_budget, release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS, RETURN_RELATIONSHIP_PROFILE
from market_predictor.swing.contracts.return_relationship_publication import VerifiedReturnRelationshipPublication
from market_predictor.swing.datasets.return_relationship_integrity import load_policy
from market_predictor.swing.datasets.return_relationship_parent import verify_parent
from market_predictor.swing.datasets.return_relationship_publication import verify_return_relationship_publication
from market_predictor.swing.datasets.return_relationship_rows import ADDITIONS, IDENTITY_COLUMNS, build_group, read_combined
from market_predictor.swing.datasets.return_relationship_sources import inventory_pins, relationship_sources, verify_source_context
from market_predictor.swing.labels.fixed_horizon_readiness import utc_clocks

SCOPE = "published_return_relationship_population_clocks_original_targets"
IMPLEMENTATION_PATHS = (
    "swing/datasets/return_relationship_verification.py", "swing/labels/fixed_horizon_readiness.py",
    "canonical/store.py", "evidence/io.py", "evidence/hashing.py", "heavy_jobs.py",
    "resources.py", "core/system_memory.py",
)


def _guard() -> None:
    assert_memory_budget(stage="return relationship verification", hard_budget_gib=5.0, headroom_gib=0.75)
    snapshot = system_memory_snapshot()
    if snapshot is None:
        raise MemoryBudgetError("system memory measurement unavailable")
    assert_system_memory_available(minimum_available_gib=snapshot.total_bytes * 0.1 / 1024**3,
        maximum_used_percent=90.0)


def _verify(root: Path, pins: dict[str, str]) -> None:
    for name, digest in pins.items():
        if file_sha256(inside(root, name)) != digest:
            raise DataReadinessError(f"return relationship evidence changed: {name}")


def _pins(root: Path, verified: VerifiedReturnRelationshipPublication) -> dict[str, str]:
    pins = dict(verified.source_files)
    for name in IMPLEMENTATION_PATHS:
        path = Path(__file__).parents[2] / name
        key, digest = path.relative_to(root).as_posix(), file_sha256(path)
        if key in pins and pins[key] != digest:
            raise DataReadinessError("conflicting current relationship implementation pins")
        pins[key] = digest
    return pins


def validate_return_relationship_receipt(
    root: Path, publication: SourcePin, receipt: dict[str, Any],
) -> VerifiedReturnRelationshipPublication:
    """Validate current evidence without a nested lease or historical-code bypass."""
    verified = verify_return_relationship_publication(root, publication)
    expected = _pins(root, verified)
    pins = receipt.get("source_files")
    if (receipt.get("schema") != "market_predictor.return_relationship_row_verification"
            or receipt.get("scope") != SCOPE or receipt.get("status") != "passed"
            or receipt.get("manifest_sha256") != publication.sha256
            or receipt.get("parent_manifest_sha256") != file_sha256(verified.parent_path)
            or receipt.get("rows") != verified.manifest["rows"]
            or receipt.get("unique_decisions") != verified.manifest["rows"]
            or receipt.get("months") != len(verified.months)
            or receipt.get("model_columns") != list(verified.model_columns)
            or receipt.get("availability_columns") != verified.availability_columns
            or receipt.get("profile_sha256") != verified.request["profile_sha256"]
            or any(receipt.get(name) is not True for name in (
                "matched_profile_population", "original_outcome_values_exact", "original_columns_exact", "additions_source_replayed"))
            or receipt.get("baseline_numerical_replayed") is not False
            or receipt.get("outcome_filtered_rows") != 0
            or any(receipt.get(name) is not False for name in (
                "training_eligible", "promotion_eligible", "serving_eligible"))
            or not isinstance(pins, dict) or any(pins.get(name) != digest for name, digest in expected.items())):
        raise DataReadinessError("saved relationship row verification differs from current publication")
    _verify(root, pins)
    return verified


def _replay_additions(root: Path, publication_path: Path, verified: VerifiedReturnRelationshipPublication) -> None:
    """Rebuild new values from bound physical sources, never from staged additions."""
    config = verified.request["config"]
    policy = load_policy(root, inside(root, config["path"]), config["sha256"])
    parent = verify_parent(root, policy)
    context = verify_source_context(root, policy, parent)
    inventory = verified.request["stock_inventory"]
    source_pins = inventory_pins(root, context, inventory)
    if any(verified.source_files.get(name) != digest for name, digest in source_pins.items()):
        raise DataReadinessError("relationship replay sources are not bound by publication")
    sources = relationship_sources(policy.parent_publication.sha256, context, inventory)
    if sources.model_dump(mode="json") != verified.request["sources"]:
        raise DataReadinessError("relationship replay source identity differs")
    spy = read_combined(context.combined_directory, context.combined["SPY"])
    paths = [str(inside(publication_path.parent, record["profiles"][RETURN_RELATIONSHIP_PROFILE]["path"]))
        for _, record in sorted(verified.months.items())]
    arrow: Any = pds
    dataset = arrow.dataset(paths, format="parquet")
    columns = [*IDENTITY_COLUMNS, *ADDITIONS]
    seen: set[str] = set()
    for item in inventory.values():
        _guard()
        expected = build_group(root, parent, context, item, sources, spy)
        predicate = arrow.field("security_id") == item["security_id"]
        if item["kind"] != "corrected":
            predicate = predicate & (arrow.field("parent_ticker") == item["source_group"])
        actual: pd.DataFrame = dataset.to_table(columns=columns, filter=predicate, use_threads=False).to_pandas()
        identities = set(expected.decision_id)
        if (actual.decision_id.duplicated().any() or expected.decision_id.duplicated().any()
                or len(actual) != len(expected) or set(actual.decision_id) != identities or seen.intersection(identities)):
            raise DataReadinessError("relationship source replay decision population differs")
        actual = actual.set_index("decision_id").loc[expected.decision_id].reset_index()
        try:
            pd.testing.assert_frame_equal(actual.loc[:, columns].reset_index(drop=True),
                expected.loc[:, columns].reset_index(drop=True), check_exact=True)
        except AssertionError as error:
            raise DataReadinessError("relationship source replay values, clocks or reasons differ") from error
        seen.update(identities)
        del actual, expected
        release_process_memory()
    if len(seen) != verified.manifest["rows"]:
        raise DataReadinessError("relationship source replay incomplete")
    _verify(root, source_pins)


def _check_rows(parent: pd.DataFrame, frame: pd.DataFrame, verified: VerifiedReturnRelationshipPublication) -> dict[str, int]:
    additions = (*RETURN_RELATIONSHIP_COLUMNS, *(f"available_at_{name}" for name in RETURN_RELATIONSHIP_COLUMNS),
        *(f"missing_reason_{name}" for name in RETURN_RELATIONSHIP_COLUMNS))
    if (not parent.columns.is_unique or not frame.columns.is_unique
            or list(frame.columns) != [*parent.columns, *additions]
            or "feature_profile" not in parent or not parent.feature_profile.eq("technical_market").all()
            or not frame.feature_profile.eq(RETURN_RELATIONSHIP_PROFILE).all()):
        raise DataReadinessError("relationship physical column order or profile differs")
    original = [name for name in parent.columns if name != "feature_profile"]
    try:
        pd.testing.assert_frame_equal(frame.loc[:, original], parent.loc[:, original], check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("relationship original columns, clocks, targets or nulls differ") from error
    if frame.empty or frame.decision_id.duplicated().any() or frame.duplicated(["security_id", "session_date_et"]).any():
        raise DataReadinessError("relationship decision population is empty or duplicated")
    cutoff = utc_clocks(frame.decision_time_utc, "decision cutoff")
    if cutoff.isna().any():
        raise DataReadinessError("relationship decision cutoff missing")
    complete = pd.Series(True, index=frame.index)
    for name in verified.model_columns:
        value = frame[name]
        if not pd.api.types.is_numeric_dtype(value.dtype) or pd.api.types.is_bool_dtype(value.dtype):
            raise DataReadinessError(f"relationship model input must be numeric: {name}")
        if np.isinf(value.to_numpy(dtype=float, na_value=np.nan)).any():
            raise DataReadinessError(f"relationship model input must be finite: {name}")
        clock = utc_clocks(frame[verified.availability_columns[name]], name)
        if (value.notna() & (clock.isna() | clock.gt(cutoff))).any():
            raise DataReadinessError(f"relationship future or missing clock: {name}")
        complete &= value.notna()
        if name in RETURN_RELATIONSHIP_COLUMNS:
            reason = frame[f"missing_reason_{name}"]
            if (value.dtype != np.dtype("float32") or not reason.map(lambda item: isinstance(item, str)).all()
                    or not reason.ne("").eq(value.isna()).all() or not clock.isna().eq(value.isna()).all()):
                raise DataReadinessError(f"relationship missingness, reason or dtype differs: {name}")
    if not frame.feature_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all():
        raise DataReadinessError("relationship eligibility must be boolean")
    return {"rows": len(frame), "feature_eligible": int(frame.feature_eligible.sum()),
        "complete_model_rows": int(complete.sum()), "clock_violations": 0}


def verify_return_relationship_rows(root: Path, publication: SourcePin, output: Path) -> dict[str, Any]:
    """Own one heavy lease; publish a fresh immutable receipt only after all checks."""
    root = root.resolve()
    output = inside(root, output)
    if not output.is_relative_to(root / "data/reports"):
        raise DataReadinessError("relationship verification output must be below data/reports")
    if output.exists():
        raise FileExistsError(f"immutable report already exists: {output}")
    runtime = heavy_job_runtime_dir()
    if not runtime.is_absolute():
        runtime = root / runtime
    with heavy_job_lease("verify-return-relationship-rows", runtime_dir=runtime):
        _guard()
        verified = verify_return_relationship_publication(root, publication)
        pins = _pins(root, verified)
        publication_path = inside(root, publication.path)
        pins[publication.path] = publication.sha256
        pins[verified.parent_path.relative_to(root).as_posix()] = file_sha256(verified.parent_path)
        _verify(root, pins)
        if set(verified.months) != set(verified.parent_manifest["months"]):
            raise DataReadinessError("relationship month inventory differs from parent")
        totals = dict(rows=0, feature_eligible=0, complete_model_rows=0, clock_violations=0)
        monthly: dict[str, Any] = {}
        seen: set[str] = set()
        for month, record in sorted(verified.months.items()):
            _guard()
            child = record["profiles"][RETURN_RELATIONSHIP_PROFILE]
            parent_record = verified.parent_manifest["months"][month]
            parent_child = parent_record["profiles"]["technical_market"]
            path = inside(publication_path.parent, child["path"])
            parent_path = inside(verified.parent_path.parent, parent_child["path"])
            child_pins = {}
            for artifact, descriptor in ((path, child), (parent_path, parent_child)):
                child_pins[artifact.relative_to(root).as_posix()] = descriptor["sha256"]
                child_pins[manifest_path_for(artifact).relative_to(root).as_posix()] = descriptor["manifest_sha256"]
            if any(name in pins and pins[name] != digest for name, digest in child_pins.items()):
                raise DataReadinessError("relationship child pin conflicts with source authority")
            pins.update(child_pins)
            _verify(root, child_pins)
            parent, parent_sidecar = load_canonical_artifact(parent_path, expected_type="swing_research_join", allow_research=True)
            frame, sidecar = load_canonical_artifact(path, expected_type="swing_return_relationships", allow_research=True)
            if (record["rows"] != parent_record["rows"] or len(frame) != record["rows"]
                    or record["decision_ids_sha256"] != parent_record["decision_ids_sha256"]
                    or json_sha256(sorted(frame.decision_id)) != record["decision_ids_sha256"]
                    or any(day.isoformat()[:7] != month for day in frame.session_date_et)
                    or child["model_columns"] != list(verified.model_columns)
                    or child["availability_columns"] != verified.availability_columns
                    or sidecar["production_ready"] is not False or parent_sidecar["production_ready"] is not False
                    or sidecar["inputs"] != {"request_sha256": verified.manifest["request_sha256"]}
                    or parent_sidecar["inputs"] != {"request_sha256": verified.parent_manifest["request_sha256"]}):
                raise DataReadinessError("relationship monthly population or source lineage differs")
            summary = _check_rows(parent, frame, verified)
            identities = set(frame.decision_id)
            if seen.intersection(identities):
                raise DataReadinessError("relationship cross-month duplicate decisions")
            seen.update(identities)
            monthly[month] = summary
            for name, value in summary.items():
                totals[name] += value
            _verify(root, child_pins)
            del parent, frame
            release_process_memory()
        if totals["rows"] != verified.manifest["rows"] or totals["rows"] != verified.parent_manifest["rows"]:
            raise DataReadinessError("relationship total population differs")
        _replay_additions(root, publication_path, verified)
        _guard()
        _verify(root, pins)
        report = {"schema": "market_predictor.return_relationship_row_verification", "scope": SCOPE, "status": "passed",
            "manifest_sha256": publication.sha256, "parent_manifest_sha256": file_sha256(verified.parent_path),
            "profile_sha256": verified.request["profile_sha256"], "source_files": pins,
            "model_columns": list(verified.model_columns), "availability_columns": verified.availability_columns,
            "rows": totals["rows"], "unique_decisions": len(seen), "months": len(monthly), "monthly": monthly,
            "profiles": {RETURN_RELATIONSHIP_PROFILE: totals}, "matched_profile_population": True,
            "original_outcome_values_exact": True, "original_columns_exact": True, "outcome_filtered_rows": 0,
            "additions_source_replayed": True, "baseline_numerical_replayed": False,
            "baseline_evidence_scope": "inherited_original_evidence_with_exact_saved_row_parity_not_new_numerical_replay",
            "training_eligible": False, "promotion_eligible": False, "serving_eligible": False}
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_object(output, report)
        return {**report, "report_sha256": file_sha256(output)}
