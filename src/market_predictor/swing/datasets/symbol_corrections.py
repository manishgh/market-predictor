"""Evidence-pinned historical symbol corrections, not return or label admission."""
from __future__ import annotations

import hashlib
from datetime import date
from pathlib import Path
from typing import Any, Protocol

import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.json_integrity import parse_strict_json_object
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, resolve_inside_authority
from market_predictor.sources.official_documents import load_official_document_inventory, verify_official_document_collection
from market_predictor.swing.datasets.history_plan_publication import PLAN_SCHEMA, UNIT_COLUMNS, publish_daily_history_plan
from market_predictor.swing.labels.holding_paths import holding_calendar
from market_predictor.universe.symbol_correction_policy import SymbolCorrectionPolicy, load_symbol_correction_policy

CORRECTION_SCOPE = "historical_symbol_correction"


class ArchiveLoader(Protocol):
    def __call__(self, directory: Path, *, plan_directory: Path, expected_adjustment: str,
        expected_plan_authority_sha256: str | None = None) -> dict[str, Any]: ...


def pinned_object(path: Path, expected_sha256: str | None = None) -> dict[str, Any]:
    if path.stat().st_size > 8 * 1024**2:
        raise DataReadinessError("symbol correction metadata exceeds bound")
    payload = path.read_bytes()
    if expected_sha256 is not None and hashlib.sha256(payload).hexdigest() != expected_sha256:
        raise DataReadinessError("symbol correction independent file pin differs")
    return parse_strict_json_object(payload, label=str(path))


def _parent(root: Path, policy: SymbolCorrectionPolicy, loader: ArchiveLoader) -> dict[str, Any]:
    plan, archive = inside(root, policy.parent_plan), inside(root, policy.parent_archive)
    plan_authority = pinned_object(plan / "_authority.json", policy.parent_plan_sha256)
    archive_authority = pinned_object(archive / "_authority.json", policy.parent_archive_sha256)
    request = pinned_object(plan / "_request.json", plan_authority["request_sha256"])
    if request.get("scope") != "initial_fit_raw_share_acquisition":
        raise DataReadinessError("symbol correction parent must be the original initial-fit plan")
    result = loader(archive, plan_directory=plan, expected_adjustment="raw",
        expected_plan_authority_sha256=policy.parent_plan_sha256)
    pinned_object(archive / "_authority.json", policy.parent_archive_sha256)
    if result != pinned_object(archive / "_manifest.json", archive_authority["artifact_sha256"]):
        raise DataReadinessError("symbol correction parent manifest changed after replay")
    pinned_object(plan / "_request.json", plan_authority["request_sha256"])
    return result


def correction_requirements(root: Path, config: Path, policy_sha256: str, loader: ArchiveLoader,
    ) -> tuple[dict[str, Any], dict[str, Any], pd.DataFrame]:
    """Reconstruct exact replacement units from reviewed facts and immutable parents."""
    root = root.resolve()
    policy = load_symbol_correction_policy(root, config, policy_sha256)
    parent = _parent(root, policy, loader)
    plan_authority = pinned_object(inside(root, policy.parent_plan) / "_authority.json", policy.parent_plan_sha256)
    original = pinned_object(inside(root, policy.parent_plan) / "_request.json", plan_authority["request_sha256"])
    inventory = load_official_document_inventory(resolve_inside_authority(root, policy.document_inventory))
    documents = verify_official_document_collection(inside(root, policy.document_archive), inventory)
    if documents["status"] != "collected_unreviewed" or json_sha256(documents) != policy.document_report_sha256:
        raise DataReadinessError("symbol correction reviewed document collection differs")
    available_documents = {item.document_id for item in inventory.documents}
    records = {item["unit_id"]: item for item in parent["unit_artifacts"]}
    if len({c.security_id for c in policy.corrections}) != 2 or len({c.ticker for c in policy.corrections}) != 2:
        raise DataReadinessError("symbol correction needs two distinct retained security identities")
    rows: list[dict[str, str]] = []
    replacements: list[dict[str, Any]] = []
    for correction in policy.corrections:
        source = records.get(correction.parent_unit_id)
        if (source is None or source["role"] != "stock" or source["security_id"] != correction.security_id
                or source["ticker"] != correction.ticker or correction.provider_symbol == source["provider_symbol"]
                or correction.security_id not in original["retained_security_ids"]
                or not date.fromisoformat(source["start_date"]) <= correction.start_date <= correction.end_date
                    <= date.fromisoformat(source["end_date"])
                or not set(correction.document_ids).issubset(available_documents)
                or len(correction.document_ids) != len(set(correction.document_ids))
                or len(correction.record_locators) != len(correction.document_ids)):
            raise DataReadinessError("symbol correction identity, interval or reviewed source differs")
        sessions = [day.isoformat() for day in holding_calendar(correction.start_date, correction.end_date)]
        if not sessions or sessions[0] != str(correction.start_date) or sessions[-1] != str(correction.end_date):
            raise DataReadinessError("symbol correction bounds must be exact exchange sessions")
        row = {"security_id": correction.security_id, "ticker": correction.ticker,
            "start_date": str(correction.start_date), "end_date": str(correction.end_date), "role": "stock"}
        rows.append(row)
        replacements.append({**correction.model_dump(mode="json"),
            "unit_id": "swing-daily-" + json_sha256(row)[:24], "parent_artifact": source,
            "required_sessions": len(sessions), "required_sessions_sha256": json_sha256(sessions),
            "selection_policy": "corrected_source_only_inside_interval_no_parent_fallback"})
    benchmarks = [item for item in parent["unit_artifacts"] if item["role"] == "benchmark"]
    if not {"SPY", "QQQ"}.issubset({item["ticker"] for item in benchmarks}):
        raise DataReadinessError("symbol correction lacks inherited SPY/QQQ artifacts")
    for benchmark in benchmarks:
        if benchmark["status"] != "observed" or any(
            not benchmark["start_date"] <= str(c.start_date) <= str(c.end_date) <= benchmark["end_date"]
            for c in policy.corrections
        ):
            raise DataReadinessError("inherited benchmark interval does not cover corrections")
        frame = pd.read_parquet(resolve_inside_authority(inside(root, policy.parent_archive), benchmark["bars_path"]),
            columns=["session_date"])
        dates = frame.session_date.astype(str).tolist()
        if dates != [day.isoformat() for day in holding_calendar(
            date.fromisoformat(benchmark["start_date"]), date.fromisoformat(benchmark["end_date"]))]:
            raise DataReadinessError("inherited benchmark exact session coverage differs")
    membership = original["membership_authority"]
    request = {"schema": PLAN_SCHEMA, "scope": CORRECTION_SCOPE, "source_root": str(root),
        "policy_path": resolve_inside_authority(root, str(config)).relative_to(root).as_posix(),
        "policy_sha256": policy_sha256, "policy": policy.model_dump(mode="json"),
        "membership_authority": membership, "provider_symbols": {c.ticker: c.provider_symbol for c in policy.corrections},
        "asof_policy": "inclusive_unit_end_date_entity_mapping_not_ownership",
        "parent_archive_manifest_sha256": file_sha256(inside(root, policy.parent_archive) / "_manifest.json"),
        "inherited_benchmarks": benchmarks, "replacements": replacements}
    manifest = {"schema": PLAN_SCHEMA, "scope": CORRECTION_SCOPE, "status": "ready_for_daily_history_collection",
        "outcomes_read": False, "accounting_eligible": False, "label_eligible": False, "promotion_eligible": False,
        "historical_availability_proven": False, "replacements": replacements, "inherited_benchmarks": benchmarks,
        "membership": {"universe_sha256": membership["universe_sha256"], "parent_lineage": membership["parent_lineage"]},
        "missing_session_ranges": [{"first_session": str(c.start_date), "last_session": str(c.end_date)} for c in policy.corrections],
        "daily_bars": {"status": "ready", "source": "alpaca", "timeframe": "1Day", "price_feed": "sip",
            "adjustment": "raw", "planned_units": 2, "stock_units": 2, "benchmark_units": 0}}
    load_symbol_correction_policy(root, config, policy_sha256)
    return request, manifest, pd.DataFrame(rows, columns=list(UNIT_COLUMNS))


def validate_symbol_correction_collection_plan(*, directory: Path, request: dict[str, Any],
    manifest: dict[str, Any], units: pd.DataFrame, parent_archive_loader: ArchiveLoader) -> None:
    expected_request, expected_manifest, expected_units = correction_requirements(
        Path(request["source_root"]), Path(request["policy_path"]), request["policy_sha256"], parent_archive_loader)
    actual_manifest = {key: value for key, value in manifest.items() if key not in {"resources", "request_sha256"}}
    actual_manifest["daily_bars"] = {key: value for key, value in manifest["daily_bars"].items() if key != "units_artifact"}
    actual_units = units[list(UNIT_COLUMNS)].sort_values(list(UNIT_COLUMNS)).reset_index(drop=True)
    if (request != expected_request or actual_manifest != expected_manifest
            or not actual_units.equals(expected_units.sort_values(list(UNIT_COLUMNS)).reset_index(drop=True))):
        raise DataReadinessError("symbol correction plan differs from pinned reviewed replacement requirements")


def publish_symbol_correction_plan(root: Path, config: Path, policy_sha256: str, output: Path,
    loader: ArchiveLoader) -> dict[str, Any]:
    request, manifest, units = correction_requirements(root, config, policy_sha256, loader)
    destination = inside(root, output)
    policy = SymbolCorrectionPolicy.model_validate(request["policy"])
    for raw in (policy.parent_plan, policy.parent_archive, policy.document_archive, str(config)):
        source = inside(root, raw)
        if source.is_relative_to(destination) or destination.is_relative_to(source):
            raise DataReadinessError("symbol correction output overlaps an immutable input")
    return publish_daily_history_plan(output=destination, request=request, manifest=manifest, units=units)
