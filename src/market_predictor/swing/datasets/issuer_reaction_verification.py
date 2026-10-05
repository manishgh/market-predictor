"""Independent saved-row and source-projection verification for the 126-column profile."""
from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.resources import release_process_memory
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_publication import (
    ARTIFACT_TYPE,
    PROFILE,
    IssuerReactionPublicationPolicy,
    VerifiedIssuerReactionPublication,
)
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_PROFILE
from market_predictor.swing.datasets.issuer_reaction_publication import (
    assemble_reaction_month,
    current_implementation,
    iter_reaction_inputs,
    load_parent_month,
    load_reaction_inputs,
    verify_issuer_reaction_publication,
)
from market_predictor.swing.datasets.return_relationship_integrity import check_files
from market_predictor.swing.datasets.return_relationship_integrity import pins as merge_pins
from market_predictor.swing.datasets.return_relationship_verification import _guard
from market_predictor.swing.features.issuer_reaction_profile import build_issuer_reaction_profile
from market_predictor.swing.features.research_partition import ResearchFeaturePartition
from market_predictor.swing.labels.fixed_horizon_readiness import utc_clocks

SCHEMA = "market_predictor.issuer_reaction_row_verification"
SCOPE = "published_issuer_reaction_population_clocks_original_targets"
IMPLEMENTATION_PATHS = (
    "swing/datasets/issuer_reaction_verification.py",
    "swing/datasets/return_relationship_verification.py",
    "swing/datasets/return_relationship_integrity.py",
    "swing/labels/fixed_horizon_readiness.py",
    "canonical/store.py", "evidence/io.py", "evidence/hashing.py",
    "heavy_jobs.py", "resources.py", "core/system_memory.py",
)


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _assert_replayed_rows_equal(saved: pd.DataFrame, expected: pd.DataFrame) -> None:
    """Compare exact source replay, allowing only nullable-string storage changes."""
    replayed = expected.copy()
    for name in saved.columns.intersection(replayed.columns):
        saved_dtype, replayed_dtype = saved[name].dtype, replayed[name].dtype
        # Parquet restores nullable strings with Arrow storage; the pure kernel
        # uses Python storage for batch parity. Logical null types stay exact.
        if (isinstance(saved_dtype, pd.StringDtype)
                and isinstance(replayed_dtype, pd.StringDtype)
                and saved_dtype.storage != replayed_dtype.storage
                and type(getattr(saved_dtype, "na_value", pd.NA))
                is type(getattr(replayed_dtype, "na_value", pd.NA))):
            replayed[name] = replayed[name].astype(saved_dtype)
    pd.testing.assert_frame_equal(saved, replayed, check_exact=True)


def _pins(root: Path, verified: VerifiedIssuerReactionPublication) -> tuple[dict[str, str], dict[str, str]]:
    package = Path(__file__).resolve().parents[2]
    current = merge_pins(root, current_implementation(root), {
        (package / name).relative_to(root).as_posix(): file_sha256(package / name)
        for name in IMPLEMENTATION_PATHS
    })
    return merge_pins(root, verified.source_files, current), current


def _receipt_identity(publication: SourcePin, verified: VerifiedIssuerReactionPublication) -> dict[str, Any]:
    policy = IssuerReactionPublicationPolicy.model_validate(verified.request["policy"])
    return {
        "schema": SCHEMA, "scope": SCOPE, "status": "passed",
        "manifest_sha256": publication.sha256,
        "parent_manifest_sha256": policy.parent_publication.sha256,
        "profile_sha256": verified.request["profile_sha256"],
        "qualification_publication": policy.qualification_publication.model_dump(mode="json"),
        "event_authority_sha256": policy.qualification_authority.sha256,
        "model_columns": list(verified.model_columns), "availability_columns": verified.availability_columns,
        "rows": verified.manifest["rows"], "unique_decisions": verified.manifest["rows"],
        "months": len(verified.months), "matched_profile_population": True,
        "original_outcome_values_exact": True, "original_columns_exact": True,
        "outcome_filtered_rows": 0, "additions_source_replayed": True,
        "qualification_source_replayed": True,
        "qualification_replay_scope": "pinned_measured_authority_and_exact_version_dispositions_not_new_raw_text_extraction",
        "baseline_numerical_replayed": False,
        "baseline_evidence_scope": "inherited_verified_relationship_parent_with_exact_saved_row_parity",
        "training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
        "historical_implementation_files": verified.request.get("historical_implementation_files", {}),
    }


def validate_issuer_reaction_receipt(
    root: Path, publication: SourcePin, receipt: dict[str, Any],
) -> VerifiedIssuerReactionPublication:
    """Require the exact independently issued receipt; acquire no nested lease."""
    root = root.resolve()
    verified = verify_issuer_reaction_publication(root, publication)
    expected, current = _pins(root, verified)
    identity = _receipt_identity(publication, verified)
    _require(all(name in receipt and json_sha256(receipt[name]) == json_sha256(value)
                 for name, value in identity.items()), "issuer reaction receipt identity or replay claims differ")
    _require(receipt.get("source_files") == expected
             and receipt.get("current_implementation_files") == current,
             "issuer reaction receipt current source or consumer pins differ")
    monthly = receipt.get("monthly")
    _require(isinstance(monthly, dict) and set(monthly) == set(verified.months),
             "issuer reaction receipt month inventory differs")
    assert isinstance(monthly, dict)
    totals: dict[str, int] = {}
    for month, summary in monthly.items():
        _require(isinstance(summary, dict) and set(summary) == set(_empty_totals())
                 and all(type(value) is int and value >= 0 for value in summary.values())
                 and summary["rows"] == verified.months[month]["rows"]
                 and summary["clock_violations"] == 0
                 and all(value <= summary["rows"] for value in summary.values()),
                 "issuer reaction receipt monthly counts differ")
        for name, value in summary.items():
            totals[name] = totals.get(name, 0) + value
    _require(receipt.get("profiles") == {PROFILE: totals} and totals["rows"] == verified.manifest["rows"],
             "issuer reaction receipt totals differ")
    check_files(root, expected)
    return verified


def _empty_totals() -> dict[str, int]:
    return dict(rows=0, feature_eligible=0, complete_model_rows=0, selected_event_rows=0,
                price_reaction_rows=0, volume_reaction_rows=0, unknown_coverage_rows=0, clock_violations=0)


def _check_rows(parent: pd.DataFrame, frame: pd.DataFrame, verified: VerifiedIssuerReactionPublication) -> dict[str, int]:
    _require(parent.columns.is_unique and frame.columns.is_unique and not parent.empty
             and list(frame.columns[:len(parent.columns)]) == list(parent.columns)
             and parent.feature_profile.eq(RETURN_RELATIONSHIP_PROFILE).all()
             and frame.feature_profile.eq(PROFILE).all(), "issuer reaction physical prefix or profile differs")
    original = [name for name in parent.columns if name != "feature_profile"]
    try:
        pd.testing.assert_frame_equal(frame.loc[:, original], parent.loc[:, original], check_exact=True)
    except AssertionError as error:
        raise DataReadinessError("issuer reaction original columns, targets, eligibility, clocks or row order differ") from error
    _require(not frame.decision_id.duplicated().any() and not frame.duplicated(["security_id", "session_date_et"]).any(),
             "issuer reaction decision population is duplicated")
    _require(len(verified.model_columns) == 126 and verified.model_columns[-2:] == REACTION_COLUMNS,
             "issuer reaction model order differs")
    cutoff = utc_clocks(frame.decision_time_utc, "decision cutoff")
    _require(not cutoff.isna().any(), "issuer reaction decision clock missing")
    complete = pd.Series(True, index=frame.index)
    for name in verified.model_columns:
        value = frame[name]
        _require(pd.api.types.is_numeric_dtype(value.dtype) and not pd.api.types.is_bool_dtype(value.dtype)
                 and not np.isinf(value.to_numpy(dtype=float, na_value=np.nan)).any(),
                 f"issuer reaction model input is not finite numeric: {name}")
        clock = utc_clocks(frame[verified.availability_columns[name]], name)
        _require(not (value.notna() & (clock.isna() | clock.gt(cutoff))).any(), f"issuer reaction future or missing clock: {name}")
        complete &= value.notna()
        if name in REACTION_COLUMNS:
            reason = frame[f"missing_reason_{name}"]
            _require(value.dtype == np.dtype("float32") and reason.map(lambda item: isinstance(item, str)).all()
                     and reason.ne("").eq(value.isna()).all() and clock.isna().eq(value.isna()).all(),
                     f"issuer reaction missingness, reason or dtype differs: {name}")
    _require(frame.feature_eligible.map(lambda value: isinstance(value, (bool, np.bool_))).all(),
             "issuer reaction eligibility must be boolean")
    _require(frame.reaction_coverage_status.isin(("known", "unknown")).all(), "issuer reaction coverage status differs")
    return {"rows": len(frame), "feature_eligible": int(frame.feature_eligible.sum()),
        "complete_model_rows": int(complete.sum()), "selected_event_rows": int(frame.selected_event_id.notna().sum()),
        "price_reaction_rows": int(frame[REACTION_COLUMNS[0]].notna().sum()),
        "volume_reaction_rows": int(frame[REACTION_COLUMNS[1]].notna().sum()),
        "unknown_coverage_rows": int(frame.reaction_coverage_status.eq("unknown").sum()), "clock_violations": 0}


def verify_issuer_reaction_rows(root: Path, publication: SourcePin, output: Path) -> dict[str, Any]:
    """Rebuild additions from qualified histories and physical bars under one lease."""
    root, output = root.resolve(), inside(root.resolve(), output)
    _require(output.is_relative_to(root / "data/reports"), "issuer reaction receipt must be below data/reports")
    if output.exists():
        raise FileExistsError(f"immutable issuer reaction receipt exists: {output}")
    runtime = heavy_job_runtime_dir()
    runtime = runtime if runtime.is_absolute() else root / runtime
    with heavy_job_lease("verify-issuer-reaction-rows", runtime_dir=runtime):
        _guard()
        verified = verify_issuer_reaction_publication(root, publication)
        source_files, current = _pins(root, verified)
        policy = IssuerReactionPublicationPolicy.model_validate(verified.request["policy"])
        inputs = load_reaction_inputs(root, policy)
        fresh = merge_pins(root, inputs.source_files)
        _require(all(source_files.get(name) == digest for name, digest in fresh.items()),
                 "issuer reaction replay source is not bound by publication")
        _require(inputs.sources.model_dump(mode="json") == verified.request["sources"]
                 and inputs.sources.event_authority_sha256 == policy.qualification_authority.sha256,
                 "issuer reaction replay source semantics differ")
        check_files(root, source_files)
        _require(set(verified.months) == set(verified.parent_manifest["months"]), "issuer reaction parent month inventory differs")
        totals = _empty_totals()
        monthly: dict[str, dict[str, int]] = {}
        seen: set[str] = set()
        for month, record in sorted(verified.months.items()):
            _guard()
            parent = load_parent_month(inputs, month)
            groups: list[ResearchFeaturePartition] = []
            group_keys: set[str] = set()
            for key, baseline, kwargs in iter_reaction_inputs(root, inputs, month):
                _require(key not in group_keys, "issuer reaction source group duplicated")
                group_keys.add(key)
                groups.append(build_issuer_reaction_profile(baseline=baseline, **kwargs))
            expected = assemble_reaction_month(parent, groups)
            child = record["profiles"][PROFILE]
            path = inside(inside(root, publication.path).parent, child["path"])
            bound = {path.relative_to(root).as_posix(): child["sha256"],
                     manifest_path_for(path).relative_to(root).as_posix(): child["manifest_sha256"]}
            _require(all(source_files.get(name) == digest for name, digest in bound.items()),
                     "issuer reaction child is not bound by publication")
            check_files(root, bound)
            frame, sidecar = load_canonical_artifact(path, expected_type=ARTIFACT_TYPE, allow_research=True)
            _require(len(frame) == record["rows"] == verified.parent_manifest["months"][month]["rows"]
                     and json_sha256(sorted(frame.decision_id)) == record["decision_ids_sha256"]
                     and frame.session_date_et.map(lambda day, expected=month: day.isoformat()[:7] == expected).all()
                     and sidecar["production_ready"] is False
                     and sidecar["inputs"] == {"request_sha256": verified.manifest["request_sha256"]}
                     and expected.model_columns == verified.model_columns
                     and expected.availability_columns == verified.availability_columns,
                     "issuer reaction monthly population or lineage differs")
            summary = _check_rows(parent, frame, verified)
            try:
                    _assert_replayed_rows_equal(frame, expected.rows)
            except AssertionError as error:
                raise DataReadinessError("issuer reaction source replay values, clocks, selection or diagnostics differ") from error
            identities = set(frame.decision_id)
            _require(not seen.intersection(identities), "issuer reaction cross-month duplicate decisions")
            seen.update(identities)
            monthly[month] = summary
            for name, value in summary.items():
                totals[name] += value
            check_files(root, bound)
            del parent, frame, expected, groups
            release_process_memory()
        _require(totals["rows"] == len(seen) == verified.manifest["rows"] == verified.parent_manifest["rows"],
                 "issuer reaction total decision population differs")
        _guard()
        check_files(root, source_files)
        report = {**_receipt_identity(publication, verified), "source_files": source_files,
                  "current_implementation_files": current, "monthly": monthly, "profiles": {PROFILE: totals}}
        if output.exists():
            raise FileExistsError(f"immutable issuer reaction receipt exists: {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        write_json_object(output, report)
        return {**report, "report_sha256": file_sha256(output)}
