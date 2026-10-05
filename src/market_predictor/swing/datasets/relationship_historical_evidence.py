"""Pinned historical inspection for the explicit relationship reuse experiment only.

This owner never admits an old sidecar through the canonical reader. Archived code
and configurations are provenance, not executed instructions or current sources.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object

HISTORICAL_MANIFEST_SCHEMA = "market_data.artifact_manifest.v1"
HISTORICAL_CANONICAL_SCHEMA = "market_data.v1"


@dataclass(frozen=True)
class HistoricalFeatureEvidence:
    path: Path
    manifest: dict[str, Any]
    request: dict[str, Any]
    receipt: dict[str, Any]
    source_files: dict[str, str]
    profile: str


def _monthly_child(record: dict[str, Any], month: str, profile: str) -> dict[str, Any]:
    """Resolve the two frozen documents' distinct ownership without changing either."""
    child = record["profiles"][profile]
    if child.get("path") != f"{month}/{profile}.parquet":
        raise DataReadinessError("historical monthly child ownership differs")
    if profile == "technical_market":
        if (child.get("audit", {}).get("rows") != record["rows"]
                or "rows" in child or "decision_ids_sha256" in child):
            raise DataReadinessError("historical baseline monthly ownership or audit rows differs")
        return {**child, "rows": record["rows"], "decision_ids_sha256": record["decision_ids_sha256"]}
    if (profile != "technical_relationships" or child.get("rows") != record["rows"]
            or child.get("decision_ids_sha256") != record["decision_ids_sha256"]):
        raise DataReadinessError("historical monthly child ownership differs")
    return dict(child)


def inspect_historical_publication(root: Path, publication: SourcePin, receipt_pin: SourcePin,
    *, profile: str,
) -> HistoricalFeatureEvidence:
    if profile not in {"technical_market", "technical_relationships"}:
        raise DataReadinessError("historical inspection requires an explicit reviewed profile")
    path = inside(root, publication.path)
    manifest = read_object(path, publication.sha256)
    request_path = path.parent / "_request.json"
    request = read_object(request_path, manifest["request_sha256"])
    receipt = read_object(inside(root, receipt_pin.path), receipt_pin.sha256)
    relationship = profile == "technical_relationships"
    schema = "market_predictor.return_relationship_publication" if relationship else "market_predictor.research_join"
    request_schema = "market_predictor.return_relationship_request" if relationship else "market_predictor.research_join_request"
    scope = ("published_return_relationship_population_clocks_original_targets" if relationship
        else "published_join_population_clocks_original_targets")
    flags = ("training_eligible", "promotion_eligible")
    request_closed = (all(request.get(flag) is False for flag in flags) if relationship else
        request.get("historical_first_seen_proven") is False
        and all(request.get(flag) is None or request.get(flag) is False for flag in flags))
    months = manifest.get("months")
    expected_months = [f"{year}-{month:02}" for year in range(2019, 2025) for month in range(1, 13)
        if "2019-07" <= f"{year}-{month:02}" <= "2024-05"]
    if (path.name != "_manifest.json" or manifest.get("schema") != schema
            or request.get("schema") != request_schema or manifest.get("status") != "complete_research_only"
            or manifest.get("exclusions_added") != [] or not isinstance(months, dict)
            or sorted(months) != expected_months or request.get("rows") != manifest.get("rows")
            or receipt.get("manifest_sha256") != publication.sha256 or receipt.get("scope") != scope
            or receipt.get("status") != "passed" or receipt.get("months") != len(months)
            or receipt.get("unique_decisions") != manifest.get("rows") or receipt.get("outcome_filtered_rows") != 0
            or receipt.get("matched_profile_population") is not True
            or receipt.get("original_outcome_values_exact") is not True
            or not request_closed or any(item.get(flag) is not False for item in (manifest, receipt) for flag in flags)):
        raise DataReadinessError("historical publication or original receipt differs")
    files = pins(root, {publication.path: publication.sha256, receipt_pin.path: receipt_pin.sha256,
        request_path.relative_to(root).as_posix(): manifest["request_sha256"]})
    rows = 0
    for month, record in sorted(months.items()):
        child = _monthly_child(record, month, profile)
        artifact = inside(path.parent, child["path"])
        sidecar = inspect_sidecar(artifact, child, artifact_type="swing_return_relationships" if relationship else "swing_research_join")
        if sidecar.get("inputs") != {"request_sha256": manifest["request_sha256"]}:
            raise DataReadinessError("historical child request binding differs")
        files.update({artifact.relative_to(root).as_posix(): child["sha256"],
            manifest_path_for(artifact).relative_to(root).as_posix(): child["manifest_sha256"]})
        rows += record["rows"]
    if rows != manifest["rows"]:
        raise DataReadinessError("historical population total differs")
    check_files(root, files)
    return HistoricalFeatureEvidence(path, manifest, request, receipt, files, profile)


def inspect_sidecar(path: Path, record: dict[str, Any], *, artifact_type: str) -> dict[str, Any]:
    sidecar = read_object(manifest_path_for(path), record["manifest_sha256"])
    if (sidecar.get("schema") != HISTORICAL_MANIFEST_SCHEMA
            or sidecar.get("canonical_schema_version") != HISTORICAL_CANONICAL_SCHEMA
            or sidecar.get("artifact_type") != artifact_type or sidecar.get("artifact_sha256") != record["sha256"]
            or sidecar.get("rows") != record["rows"]
            or sidecar.get("production_ready") is not (artifact_type == "bars")
            or file_sha256(path) != record["sha256"]):
        raise DataReadinessError("pinned historical sidecar or artifact differs")
    parquet: Any = pq
    physical = parquet.ParquetFile(path)
    if physical.metadata.num_rows != record["rows"] or physical.schema_arrow.names != sidecar["columns"]:
        raise DataReadinessError("historical physical schema or row count differs")
    return sidecar


def historical_month(evidence: HistoricalFeatureEvidence, month: str,
    columns: list[str] | None = None,
) -> pd.DataFrame:
    record = _monthly_child(evidence.manifest["months"][month], month, evidence.profile)
    path = inside(evidence.path.parent, record["path"])
    check_files(evidence.path.parent, {record["path"]: record["sha256"],
        manifest_path_for(path).relative_to(evidence.path.parent).as_posix(): record["manifest_sha256"]})
    frame = pd.read_parquet(path, columns=columns)
    if (file_sha256(path) != record["sha256"] or len(frame) != record["rows"]
            or frame.decision_id.duplicated().any()
            or json_sha256(sorted(frame.decision_id)) != record["decision_ids_sha256"]):
        raise DataReadinessError("historical monthly rows changed or repeat decisions")
    return frame


def historical_source_path(root: Path, request: dict[str, Any], digest: str, suffix: str) -> Path:
    candidates = [inside(root, name) for name, declared in request["source_files"].items()
        if declared == digest and name.replace("\\", "/").endswith(suffix.replace("\\", "/"))]
    if len(candidates) != 1:
        raise DataReadinessError("historical source lacks one exact declared path/hash")
    return candidates[0]


def historical_bars(root: Path, request: dict[str, Any], item: dict[str, Any],
    files: dict[str, str],
) -> pd.DataFrame:
    """Project initial-fit rows before materialization; never read later row values."""
    record = item["artifact"]
    corrected = item["kind"] == "corrected"
    digest, suffix = (record["bars_sha256"], record["bars_path"]) if corrected else (record["sha256"], record["path"])
    path = historical_source_path(root, request, digest, suffix)
    files[path.relative_to(root).as_posix()] = digest
    if file_sha256(path) != digest:
        raise DataReadinessError("historical adjusted input changed")
    if corrected:
        manifest_path = historical_source_path(root, request, record["unit_manifest_sha256"], record["unit_manifest_path"])
        unit = read_object(manifest_path, record["unit_manifest_sha256"])
        files[manifest_path.relative_to(root).as_posix()] = record["unit_manifest_sha256"]
        if any(unit.get(key) != record[key] for key in ("bars_sha256", "security_id", "ticker", "role",
                "provider_symbol", "start_date", "end_date", "rows", "plan_unit_sha256", "unit_id")):
            raise DataReadinessError("historical corrected source unit differs")
        frame = pd.read_parquet(path, columns=RAW_COLUMNS,
            filters=[("session_date", ">=", "2018-05-29"), ("session_date", "<=", "2024-05-28")])
        if (len(frame) != record["rows"] or not frame.security_id.eq(record["security_id"]).all()
                or not frame.ticker.eq(record["ticker"]).all() or not frame.source.eq("alpaca").all()
                or not frame.price_feed.eq("sip").all() or not frame.adjustment.eq("all").all()
                or not frame.timeframe.eq("1Day").all()):
            raise DataReadinessError("historical corrected source identity/feed/basis differs")
        frame = canonicalize_bars(frame, timeframe="1d", availability_policy="market_interval_close")
    elif item["kind"] == "combined":
        sidecar = inspect_sidecar(path, {**record, "manifest_sha256": record["canonical_manifest_sha256"]}, artifact_type="bars")
        files[manifest_path_for(path).relative_to(root).as_posix()] = record["canonical_manifest_sha256"]
        columns = ["ticker", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc", "open", "high", "low",
            "close", "volume", "price_feed", "adjustment", "schema_version", "source", "availability_policy", "ingested_at_utc"]
        if not set(columns).issubset(sidecar["columns"]):
            raise DataReadinessError("historical physical bar columns missing")
        arrow: Any = pds
        predicate = ((arrow.field("bar_start_utc") >= pd.Timestamp("2018-05-29", tz="UTC"))
            & (arrow.field("bar_start_utc") < pd.Timestamp("2024-05-29", tz="UTC")))
        frame = arrow.dataset(path, format="parquet").to_table(columns=columns, filter=predicate, use_threads=False).to_pandas()
        if not frame.ticker.eq(record["ticker"]).all():
            raise DataReadinessError("historical physical ticker differs")
    else:
        raise DataReadinessError("unknown historical relationship stream")
    if file_sha256(path) != digest:
        raise DataReadinessError("historical adjusted input changed during read")
    frame["session_date_et"] = pd.to_datetime(frame.bar_start_utc, utc=True).dt.tz_convert("America/New_York").dt.date
    if not frame.session_date_et.between(pd.Timestamp("2018-05-29").date(), pd.Timestamp("2024-05-28").date()).all():
        raise DataReadinessError("historical projection escaped initial-fit bounds")
    frame["security_id"] = item["security_id"]
    return frame
