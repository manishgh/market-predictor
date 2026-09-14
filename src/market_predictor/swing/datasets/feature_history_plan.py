"""Two corrected adjusted feature streams through the shared daily-plan authority."""
from __future__ import annotations

import hashlib
import tomllib
from datetime import date
from pathlib import Path
from typing import Any, Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.implementation_snapshot import verify_implementation_snapshot
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.sources.official_documents import load_official_document_inventory, verify_official_document_collection
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.research_cohort import Sha256
from market_predictor.swing.datasets.history_plan_publication import (
    AUTHORITY_SCHEMA,
    PLAN_SCHEMA,
    UNIT_COLUMNS,
    publish_daily_history_plan,
)
from market_predictor.swing.datasets.symbol_corrections import pinned_object
from market_predictor.swing.labels.holding_paths import holding_calendar
from market_predictor.universe.symbol_correction_policy import load_symbol_correction_policy

FEATURE_HISTORY_SCOPE = "corrected_adjusted_feature_history"
WARMUP_START = date(2018, 5, 29)
DECISION_START = date(2019, 7, 9)
NUMERIC_END = date(2024, 5, 28)
_ENTITIES = {"cik:0001415404": ("ECHO", "SATS", DECISION_START),
    "cik:0000798354": ("FISV", "FI", date(2023, 6, 7))}


class FeatureHistoryPolicy(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    schema_version: Literal["market_predictor.corrected_feature_history_policy"]
    correction_policy: str
    correction_policy_sha256: Sha256
    warmup_start: date
    decision_start: date
    numeric_end: date
    adjustment: Literal["all"]
    purpose: Literal["source_query_only_not_membership_or_historical_first_seen"]


def _policy(root: Path, config: Path, expected_sha256: str) -> FeatureHistoryPolicy:
    path = resolve_inside_authority(root, config)
    if path.stat().st_size > 65536:
        raise DataReadinessError("feature history configuration exceeds bound")
    payload = path.read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError("feature history configuration requires its independent pin")
    policy = FeatureHistoryPolicy.model_validate(tomllib.loads(payload.decode("utf-8")))
    if (policy.warmup_start, policy.decision_start, policy.numeric_end) != (WARMUP_START, DECISION_START, NUMERIC_END):
        raise DataReadinessError("feature history warm-up, decision or numeric boundary differs")
    return policy


def feature_history_requirements(root: Path, config: Path, policy_sha256: str,
) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Reconstruct source-only requirements without opening historical price files.

    One full adjusted query per entity prevents an adjusted/raw or rename splice.
    The end-date query symbol labels the entire provider stream; it is not a claim
    about the historical exchange ticker, membership, or first-observed timing.
    Source completeness, adjustment suitability and feature admission are downstream.
    """
    root = root.resolve()
    policy = _policy(root, config, policy_sha256)
    correction = load_symbol_correction_policy(root, Path(policy.correction_policy), policy.correction_policy_sha256)
    if {item.security_id for item in correction.corrections} != set(_ENTITIES):
        raise DataReadinessError("feature history requires the two reviewed corrected issuer identities")
    for item in correction.corrections:
        ticker, provider, start = _ENTITIES[item.security_id]
        if (item.ticker, item.provider_symbol, item.start_date, item.end_date) != (ticker, provider, start, NUMERIC_END):
            raise DataReadinessError("feature history corrected provider identity or transition differs")
        if len(set(item.document_ids)) != len(item.document_ids) or len(item.document_ids) != len(item.record_locators):
            raise DataReadinessError("feature history requires distinct reviewed document locators")
    inventory = load_official_document_inventory(resolve_inside_authority(root, correction.document_inventory))
    documents = verify_official_document_collection(inside(root, correction.document_archive), inventory)
    if documents["status"] != "collected_unreviewed" or json_sha256(documents) != correction.document_report_sha256:
        raise DataReadinessError("feature history reviewed document replay pin differs")
    document_ids = {item.document_id for item in inventory.documents}
    if any(not set(item.document_ids).issubset(document_ids) for item in correction.corrections):
        raise DataReadinessError("feature history reviewed identity documents are missing")
    # Inherit universe lineage, never broaden its membership into source warm-up.
    parent = inside(root, correction.parent_plan)
    authority = pinned_object(parent / "_authority.json", correction.parent_plan_sha256)
    if authority.get("schema") != AUTHORITY_SCHEMA or authority.get("state") != "complete":
        raise DataReadinessError("feature history parent acquisition authority is incomplete")
    original = pinned_object(parent / "_request.json", authority["request_sha256"])
    membership = original.get("membership_authority")
    if (original.get("schema") != PLAN_SCHEMA or original.get("scope") != "initial_fit_raw_share_acquisition"
            or not isinstance(membership, dict) or membership.get("universe_sha256") != authority.get("universe_sha256")
            or not set(_ENTITIES).issubset(original.get("retained_security_ids", ()))):
        raise DataReadinessError("feature history parent universe/retained identities differ")
    sessions = tuple(day.isoformat() for day in holding_calendar(WARMUP_START, NUMERIC_END))
    if not sessions or sessions[0] != str(WARMUP_START) or sessions[-1] != str(NUMERIC_END):
        raise DataReadinessError("feature history bounds are not exact XNYS sessions")
    units = pd.DataFrame([{"security_id": identity, "ticker": values[1],
        "start_date": str(WARMUP_START), "end_date": str(NUMERIC_END), "role": "stock"}
        for identity, values in sorted(_ENTITIES.items())], columns=list(UNIT_COLUMNS))
    package = Path(__file__).resolve().parents[2]
    request = {"schema": PLAN_SCHEMA, "scope": FEATURE_HISTORY_SCOPE, "source_root": str(root),
        "policy_path": resolve_inside_authority(root, config).relative_to(root).as_posix(),
        "policy_sha256": policy_sha256, "policy": policy.model_dump(mode="json"),
        "reviewed_correction_policy": correction.model_dump(mode="json"),
        "membership_authority": membership,
        "provider_symbols": {values[1]: values[1] for values in _ENTITIES.values()},
        "asof_policy": "inclusive_unit_end_date_entity_mapping_not_ownership",
        "query_symbol_policy": "asof_entity_stream_label_not_historical_exchange_ticker",
        "feature_basis": "single_full_adjusted_stream_per_entity_no_raw_splice",
        "benchmark_policy": "not_requested_feature_inputs_only_not_complete_panel",
        "decision_start": str(DECISION_START), "warmup_start": str(WARMUP_START), "numeric_end": str(NUMERIC_END),
        "required_sessions": len(sessions), "required_sessions_sha256": json_sha256(sessions),
        "implementation_files": {name: file_sha256(package / name) for name in (
            "swing/datasets/feature_history_plan.py", "swing/datasets/history_plan_publication.py",
            "swing/labels/holding_paths.py")}}
    manifest = {"schema": PLAN_SCHEMA, "scope": FEATURE_HISTORY_SCOPE, "status": "ready_for_daily_history_collection",
        "outcomes_read": False, "ownership_admitted": False, "bar_coverage_verified": False,
        "feature_eligible": False, "label_eligible": False, "accounting_eligible": False,
        "promotion_eligible": False, "historical_availability_proven": False,
        "membership": {"universe_sha256": membership["universe_sha256"], "parent_lineage": membership["parent_lineage"]},
        "missing_session_ranges": [{"first_session": str(WARMUP_START), "last_session": str(NUMERIC_END), "sessions": len(sessions)}],
        "daily_bars": {"status": "ready", "source": "alpaca", "timeframe": "1Day", "price_feed": "sip",
            "adjustment": "all", "planned_units": 2, "stock_units": 2, "benchmark_units": 0}}
    _policy(root, config, policy_sha256)
    load_symbol_correction_policy(root, Path(policy.correction_policy), policy.correction_policy_sha256)
    pinned_object(parent / "_authority.json", correction.parent_plan_sha256)
    pinned_object(parent / "_request.json", authority["request_sha256"])
    return request, manifest, units


def validate_feature_history_collection_plan(*, directory: Path, request: dict[str, Any],
    manifest: dict[str, Any], units: pd.DataFrame,
    implementation_snapshot: SourcePin | None = None,
) -> dict[str, str]:
    """Collector hook: a rehashed altered plan must still reproduce its pinned scope."""
    expected_request, expected_manifest, expected_units = feature_history_requirements(
        Path(request["source_root"]), Path(request["policy_path"]), request["policy_sha256"])
    proof: dict[str, str] = {}
    comparison_request = request
    if implementation_snapshot is not None:
        old = request.get("implementation_files")
        current = expected_request["implementation_files"]
        if not isinstance(old, dict) or set(old) != set(current):
            raise DataReadinessError("feature plan implementation declaration keys differ")
        changed = {f"src/market_predictor/{name}": digest for name, digest in old.items() if digest != current[name]}
        root = Path(request["source_root"]).resolve()
        proof = verify_implementation_snapshot(root, inside(root, implementation_snapshot.path),
            implementation_snapshot.sha256, changed)
        comparison_request = {**request, "implementation_files": current}
    actual = {key: value for key, value in manifest.items() if key not in {"resources", "request_sha256"}}
    actual["daily_bars"] = {key: value for key, value in manifest["daily_bars"].items() if key != "units_artifact"}
    actual_units = units.loc[:, list(UNIT_COLUMNS)].sort_values(list(UNIT_COLUMNS)).reset_index(drop=True)
    if (comparison_request != expected_request or actual != expected_manifest
            or not actual_units.equals(expected_units.sort_values(list(UNIT_COLUMNS)).reset_index(drop=True))):
        raise DataReadinessError("feature history plan differs from pinned corrected adjusted requirements")
    return proof


def verify_feature_plan_replay(*, root: Path, authority: SourcePin,
    implementation_snapshot: SourcePin,
) -> dict[str, str]:
    """Reconstruct a pinned source-only plan and retain its explicit code evidence."""
    root = root.resolve()
    path = inside(root, authority.path)
    if path.name != "_authority.json":
        raise DataReadinessError("feature plan replay requires its independent authority")
    document = pinned_object(path, authority.sha256)
    if document.get("schema") != AUTHORITY_SCHEMA or document.get("state") != "complete":
        raise DataReadinessError("feature plan replay authority is incomplete")
    request_path, manifest_path = path.parent / "_request.json", path.parent / "_manifest.json"
    request = pinned_object(request_path, document["request_sha256"])
    manifest = pinned_object(manifest_path, document["artifact_sha256"])
    if (Path(request["source_root"]).resolve() != root or request.get("scope") != FEATURE_HISTORY_SCOPE
            or manifest.get("scope") != FEATURE_HISTORY_SCOPE):
        raise DataReadinessError("feature plan replay source root or scope differs")
    unit_record = manifest["daily_bars"]["units_artifact"]
    units_path = inside(path.parent, unit_record["path"])
    if (units_path.stat().st_size > 65536 or units_path.stat().st_size != unit_record["bytes"]
            or file_sha256(units_path) != document["units_sha256"]
            or document["units_sha256"] != unit_record["sha256"]):
        raise DataReadinessError("feature plan replay units differ from their authority")
    units = pd.read_csv(units_path, dtype=str)
    proof = validate_feature_history_collection_plan(directory=path.parent, request=request,
        manifest=manifest, units=units, implementation_snapshot=implementation_snapshot)
    proof.update({path.relative_to(root).as_posix(): authority.sha256,
        request_path.relative_to(root).as_posix(): document["request_sha256"],
        manifest_path.relative_to(root).as_posix(): document["artifact_sha256"],
        units_path.relative_to(root).as_posix(): document["units_sha256"]})
    return proof


def publish_feature_history_plan(root: Path, config: Path, policy_sha256: str, output: Path) -> dict[str, Any]:
    """Use the shared immutable publisher; collection is a separate approved job."""
    request, manifest, units = feature_history_requirements(root, config, policy_sha256)
    root = root.resolve()
    destination = inside(root, output)
    correction = request["reviewed_correction_policy"]
    for raw in (str(config), request["policy"]["correction_policy"], correction["parent_plan"],
            correction["parent_archive"], correction["document_inventory"], correction["document_archive"]):
        source = inside(root, raw)
        if source.is_relative_to(destination) or destination.is_relative_to(source):
            raise DataReadinessError("feature history output overlaps an immutable input")
    return publish_daily_history_plan(output=destination, request=request, manifest=manifest, units=units)
