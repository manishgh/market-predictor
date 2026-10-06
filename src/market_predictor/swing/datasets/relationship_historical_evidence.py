"""Pinned historical inspection for the explicit relationship reuse experiment only.

This owner never admits an old sidecar through the canonical reader. Archived code
and configurations are provenance, not executed instructions or current sources.
"""
from __future__ import annotations

import json
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
ORIGINAL_SOURCE_COMPARISON_SHA256 = "5f480f56179e054ad4f621f780d72e49e50ac2482af74536c9a3b4aad5e83265"


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


def inspect_original_source_comparison(root: Path, pin: SourcePin,
    parent: HistoricalFeatureEvidence, relationships: HistoricalFeatureEvidence,
) -> dict[str, Any]:
    """Bind the measured source change without converting its failure into a pass."""
    if pin.sha256 != ORIGINAL_SOURCE_COMPARISON_SHA256:
        raise DataReadinessError("original replay requires the exact failed source comparison")
    report = read_object(inside(root, pin.path), pin.sha256)
    if (report.get("schema") != "market_predictor.relationship_reuse_equivalence"
            or report.get("status") != "failed_differences" or report.get("source_inputs_equal") is not False
            or report.get("additions_equal") is not False or report.get("inherited_columns_equal") is not True
            or report.get("population_equal") is not True or report.get("rows") != parent.manifest["rows"]
            or report.get("policy", {}).get("historical_parent_publication", {}).get("sha256") != file_sha256(parent.path)
            or report.get("policy", {}).get("historical_relationship_publication", {}).get("sha256") != file_sha256(relationships.path)
            or any(report.get(key) is not False for key in ("training_eligible", "promotion_eligible", "serving_eligible"))):
        raise DataReadinessError("original source comparison changed scope or admission claims")
    return report


def original_stock_bars(root: Path, evidence: HistoricalFeatureEvidence, key: str,
    files: dict[str, str],
) -> pd.DataFrame:
    """Read one original owner and preserve its independently recorded abstention."""
    from market_predictor.swing.datasets.predictor_abstention_derivation import ReviewedPredictorFailure
    from market_predictor.swing.datasets.preserved_relationship_abstentions import (
        HISTORICAL_OBSERVATION_SHA256,
        pinned_original_observation_report,
        preserved_physical_prefix,
    )

    item = evidence.request["stock_inventory"][key]
    if key != json_sha256([item["security_id"], item["source_group"]]):
        raise DataReadinessError("original stock owner key differs")
    frame = historical_bars(root, evidence.request, item, files)
    if item["quarantine"] is None:
        return frame
    fact = ReviewedPredictorFailure.model_validate_json(json.dumps(item["quarantine"]))
    if (fact.group_key != key or fact.security_id != item["security_id"] or fact.rows != item["rows"]
            or fact.symbol != item["source_group"] or len(fact.source_artifacts) != 1
            or evidence.receipt.get("additions_source_replayed") is not True):
        raise DataReadinessError("original quarantine differs from its source owner")
    observations = [pin for pin in fact.reviewed_evidence if pin.sha256 == HISTORICAL_OBSERVATION_SHA256]
    if len(observations) != 1:
        raise DataReadinessError("original quarantine lacks its exact observation")
    declared = pins(root, evidence.request["source_files"])
    for pin in (*fact.source_artifacts, *fact.reviewed_evidence):
        name = inside(root, pin.path).relative_to(root).as_posix()
        if declared.get(name) != pin.sha256:
            raise DataReadinessError("original quarantine source is not request-bound")
        files.update(pins(root, {name: pin.sha256}))
    artifact = item["artifact"]
    digest = artifact["bars_sha256"] if item["kind"] == "corrected" else artifact["sha256"]
    suffix = artifact["bars_path"] if item["kind"] == "corrected" else artifact["path"]
    source = fact.source_artifacts[0]
    if (source.sha256 != digest or inside(root, source.path) != historical_source_path(root, evidence.request, digest, suffix)):
        raise DataReadinessError("original quarantine substituted its physical bars")
    report = pinned_original_observation_report(root, observations[0])
    matches = [row for row in report.get("observations", ()) if row.get("security_id") == fact.security_id
        and row.get("ticker") == fact.symbol and row.get("source_sha256") == source.sha256
        and row.get("source_path") == inside(root, source.path).relative_to(root).as_posix()]
    if len(matches) != 1 or len(frame) != matches[0].get("bounded_rows"):
        raise DataReadinessError("original quarantine observation is incomplete or ambiguous")
    if fact.first_invalid_session is not None:
        invalid = matches[0].get("invalid_rows", [])
        first = min(invalid, key=lambda row: row["session_date_et"]) if invalid else None
        if (first is None or first["session_date_et"] != str(fact.first_invalid_session)
                or json_sha256(first) != fact.boundary_observation_sha256
                or first["invalid_fields"] != ["volume"] or first["ohlcv"]["volume"] != 0):
            raise DataReadinessError("original first-invalid boundary differs")
    check_files(root, files)
    return preserved_physical_prefix(frame, fact, matches[0], tuple(sorted(frame.session_date_et)))
