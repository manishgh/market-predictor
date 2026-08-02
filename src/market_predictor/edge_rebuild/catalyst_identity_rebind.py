"""One-time, audited catalyst identity migration onto the canonical swing spine."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.edge_rebuild.catalyst_authority import (
    COVERAGE_ARTIFACT_TYPE,
    DECISION_ARTIFACT_TYPE,
    CatalystDecisionAuthority,
    load_catalyst_decision_authority,
)
from market_predictor.v3.errors import DataReadinessError

REBIND_REQUEST_SCHEMA: Final = "edge_rebuild.catalyst_identity_rebind_request.v1"
REBIND_MANIFEST_SCHEMA: Final = "edge_rebuild.catalyst_decision_manifest.v5"
REBIND_AUTHORITY_SCHEMA: Final = "edge_rebuild.catalyst_decision_authority.v5"
DECISION_LEDGER_TYPE: Final = "edge_rebuild_catalyst_decision_identity_rebind"
COVERAGE_LEDGER_TYPE: Final = "edge_rebuild_catalyst_coverage_identity_rebind"
TARGET_PANEL_MANIFEST_SCHEMA: Final = "edge_rebuild.swing_panel_materialization.v9"
TARGET_PANEL_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.swing_panel_materialization_authority.v9"
)
TARGET_PROFILE: Final = "technical_market"
_MINIMUM_DECISION_DATE: Final = "2019-07-09"
_TARGET_COLUMNS: Final = (
    "decision_id",
    "security_id",
    "ticker",
    "decision_time_utc",
    "membership_effective_from_utc",
    "membership_effective_to_utc",
    "membership_available_at_utc",
)


@dataclass(frozen=True, slots=True)
class _TargetSpine:
    root: Path
    frame: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]
    manifest_sha256: str
    authority_sha256: str


def publish_catalyst_identity_rebind(
    *,
    parent_directory: Path,
    target_panel_directory: Path,
    output_directory: Path,
) -> CatalystDecisionAuthority:
    """Publish a V5 authority after an exact, ledgered identity migration."""

    parent = load_catalyst_decision_authority(parent_directory)
    target = _load_target_spine(target_panel_directory)
    output = output_directory.resolve()
    if output.exists():
        raise DataReadinessError(f"catalyst identity rebind is immutable: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = output.with_name(f".{output.name}.{uuid4().hex}.tmp")
    try:
        staging.mkdir(parents=False, exist_ok=False)
        decisions, decision_ledger = _rebind_decisions(parent.decisions, target.frame)
        coverage, coverage_ledger = _rebind_coverage(parent.coverage, target.frame)
        target_security_ids = set(target.frame["security_id"].astype(str))
        covered_security_ids = set(coverage["security_id"].astype(str))
        missing_coverage = sorted(target_security_ids.difference(covered_security_ids))
        if missing_coverage:
            raise DataReadinessError(
                "catalyst rebind leaves target securities without source coverage: "
                + ", ".join(missing_coverage[:10])
            )

        request = _request(parent, target)
        request_sha256 = _json_sha256(request)
        _atomic_json(staging / "_request.json", request)
        child_inputs: dict[str, str] = {
            "request_sha256": request_sha256,
            "parent_authority_sha256": file_sha256(parent.directory / "_authority.json"),
            "target_panel_authority_sha256": target.authority_sha256,
        }
        artifacts: dict[str, dict[str, object]] = {}
        artifacts["decisions"] = _write_artifact(
            decisions,
            staging / "decision_catalysts.parquet",
            artifact_type=DECISION_ARTIFACT_TYPE,
            inputs=child_inputs,
            final_directory=output,
            unique_column="decision_id",
        )
        artifacts["coverage"] = _write_artifact(
            coverage,
            staging / "source_coverage.parquet",
            artifact_type=COVERAGE_ARTIFACT_TYPE,
            inputs=child_inputs,
            final_directory=output,
            unique_column="coverage_evidence_id",
        )
        artifacts["decision_ledger"] = _write_artifact(
            decision_ledger,
            staging / "decision_identity_rebind.parquet",
            artifact_type=DECISION_LEDGER_TYPE,
            inputs=child_inputs,
            final_directory=output,
            unique_column="source_decision_id",
        )
        artifacts["coverage_ledger"] = _write_artifact(
            coverage_ledger,
            staging / "coverage_identity_rebind.parquet",
            artifact_type=COVERAGE_LEDGER_TYPE,
            inputs=child_inputs,
            final_directory=output,
            unique_column="source_coverage_evidence_id",
        )
        parent_manifest = parent.manifest
        decision_counts = _status_counts(decision_ledger)
        coverage_counts = _status_counts(coverage_ledger)
        manifest: dict[str, object] = {
            "schema": REBIND_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "source_lineage_set_sha256": str(
                parent_manifest["source_lineage_set_sha256"]
            ),
            "sentiment_scorer_identity": parent_manifest[
                "sentiment_scorer_identity"
            ],
            "sentiment_scorer_identity_sha256": parent_manifest[
                "sentiment_scorer_identity_sha256"
            ],
            "windows": parent_manifest["windows"],
            "source_families": parent_manifest["source_families"],
            "tracked_source_families": parent_manifest[
                "tracked_source_families"
            ],
            "required_model_source_families": parent_manifest[
                "required_model_source_families"
            ],
            "minimum_decision_date": _MINIMUM_DECISION_DATE,
            "decision_rows": len(decisions),
            "coverage_rows": len(coverage),
            "decision_rebind_status_counts": decision_counts,
            "coverage_rebind_status_counts": coverage_counts,
            "target_security_count": len(target_security_ids),
            "target_security_ids_sha256": _json_sha256(
                sorted(target_security_ids)
            ),
            "artifacts": artifacts,
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "production_ready": False,
            "missing_value_policy": parent_manifest["missing_value_policy"],
            "identity_policy": (
                "decision evidence requires exact ticker and decision timestamp; "
                "coverage requires a ticker assigned to exactly one canonical "
                "security across the governed target population"
            ),
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": REBIND_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "source_lineage_set_sha256": manifest[
                "source_lineage_set_sha256"
            ],
            "decision_artifact_sha256": artifacts["decisions"]["sha256"],
            "coverage_artifact_sha256": artifacts["coverage"]["sha256"],
            "decision_ledger_sha256": artifacts["decision_ledger"]["sha256"],
            "coverage_ledger_sha256": artifacts["coverage_ledger"]["sha256"],
            "sentiment_scorer_identity_sha256": manifest[
                "sentiment_scorer_identity_sha256"
            ],
            "target_panel_authority_sha256": target.authority_sha256,
            "target_security_ids_sha256": manifest["target_security_ids_sha256"],
            "production_ready": False,
            "minimum_decision_date": _MINIMUM_DECISION_DATE,
        }
        _atomic_json(staging / "_authority.json", authority)
        load_catalyst_identity_rebind(staging)
        os.replace(staging, output)
        return load_catalyst_identity_rebind(output)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_catalyst_identity_rebind(directory: Path) -> CatalystDecisionAuthority:
    """Replay every V5 lineage and artifact binding before returning data."""

    root = directory.resolve()
    request = _json_object(root / "_request.json")
    manifest = _json_object(root / "_manifest.json")
    authority = _json_object(root / "_authority.json")
    if request.get("schema") != REBIND_REQUEST_SCHEMA:
        raise DataReadinessError("catalyst identity rebind request schema is invalid")
    request_sha256 = _json_sha256(request)
    if (
        manifest.get("schema") != REBIND_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("request") != request
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("production_ready") is not False
        or manifest.get("minimum_decision_date") != _MINIMUM_DECISION_DATE
    ):
        raise DataReadinessError("catalyst identity rebind manifest does not verify")
    if (
        authority.get("schema") != REBIND_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(root / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or authority.get("source_lineage_set_sha256")
        != manifest.get("source_lineage_set_sha256")
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("catalyst identity rebind authority does not verify")
    _replay_sources(request)
    expected_files = {
        "_request.json",
        "_manifest.json",
        "_authority.json",
        "decision_catalysts.parquet",
        "decision_catalysts.parquet.manifest.json",
        "source_coverage.parquet",
        "source_coverage.parquet.manifest.json",
        "decision_identity_rebind.parquet",
        "decision_identity_rebind.parquet.manifest.json",
        "coverage_identity_rebind.parquet",
        "coverage_identity_rebind.parquet.manifest.json",
    }
    if {item.name for item in root.iterdir()} != expected_files:
        raise DataReadinessError("catalyst identity rebind inventory does not verify")
    artifacts = _required_mapping(manifest, "artifacts")
    decisions = _load_bound_artifact(
        root,
        artifacts,
        "decisions",
        "decision_catalysts.parquet",
        DECISION_ARTIFACT_TYPE,
        request_sha256,
    )
    coverage = _load_bound_artifact(
        root,
        artifacts,
        "coverage",
        "source_coverage.parquet",
        COVERAGE_ARTIFACT_TYPE,
        request_sha256,
    )
    decision_ledger = _load_bound_artifact(
        root,
        artifacts,
        "decision_ledger",
        "decision_identity_rebind.parquet",
        DECISION_LEDGER_TYPE,
        request_sha256,
    )
    coverage_ledger = _load_bound_artifact(
        root,
        artifacts,
        "coverage_ledger",
        "coverage_identity_rebind.parquet",
        COVERAGE_LEDGER_TYPE,
        request_sha256,
    )
    _validate_replayed_rows(
        decisions,
        coverage,
        decision_ledger,
        coverage_ledger,
        manifest,
        authority,
    )
    return CatalystDecisionAuthority(
        directory=root,
        decisions=decisions,
        coverage=coverage,
        manifest=manifest,
        authority=authority,
    )


def _rebind_decisions(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    parent = source.copy()
    parent["_ticker_key"] = parent["ticker"].astype(str).str.upper()
    parent["_decision_key"] = pd.to_datetime(
        parent["decision_time_utc"], utc=True, errors="raise"
    )
    spine = target.loc[:, list(_TARGET_COLUMNS)].copy()
    spine["_ticker_key"] = spine["ticker"].astype(str).str.upper()
    spine["_decision_key"] = pd.to_datetime(
        spine["decision_time_utc"], utc=True, errors="raise"
    )
    if bool(
        parent.duplicated(["_ticker_key", "_decision_key"]).any()
        or spine.duplicated(["_ticker_key", "_decision_key"]).any()
    ):
        raise DataReadinessError(
            "catalyst identity rebind requires unique ticker/decision timestamps"
        )
    lookup = spine.loc[
        :,
        ["_ticker_key", "_decision_key", "decision_id", "security_id", "ticker"],
    ].rename(
        columns={
            "decision_id": "target_decision_id",
            "security_id": "target_security_id",
            "ticker": "target_ticker",
        }
    )
    merged = parent.merge(
        lookup,
        on=["_ticker_key", "_decision_key"],
        how="left",
        validate="one_to_one",
        sort=False,
    )
    merged["match_status"] = "unmatched"
    merged["match_reason"] = "target_panel_excluded"
    matched = merged["target_decision_id"].notna()
    merged.loc[matched, "match_status"] = "rebound"
    merged.loc[matched, "match_reason"] = "exact_ticker_decision_time"

    intervals = _membership_intervals(spine)
    target_tickers = set(spine["_ticker_key"].astype(str))
    for index in merged.index[~matched]:
        ticker = str(merged.at[index, "_ticker_key"])
        if ticker not in target_tickers:
            continue
        decision_time = pd.Timestamp(merged.at[index, "_decision_key"])
        active = any(
            available <= decision_time
            and effective_from <= decision_time < effective_to
            for effective_from, effective_to, available in intervals[ticker]
        )
        merged.at[index, "match_reason"] = (
            "no_target_traded_decision" if active else "outside_canonical_membership"
        )

    rebound = merged.loc[matched].copy()
    output_columns = list(source.columns)
    rebound["decision_id"] = rebound["target_decision_id"].astype(str)
    rebound["security_id"] = rebound["target_security_id"].astype(str)
    rebound["ticker"] = rebound["target_ticker"].astype(str).str.upper()
    decisions = rebound.loc[:, output_columns].sort_values(
        ["decision_time_utc", "security_id"], kind="stable"
    ).reset_index(drop=True)
    ledger = pd.DataFrame(
        {
            "source_decision_id": merged["decision_id"].astype(str),
            "source_security_id": merged["security_id"].astype(str),
            "source_ticker": merged["ticker"].astype(str).str.upper(),
            "decision_time_utc": merged["_decision_key"],
            "target_decision_id": merged["target_decision_id"].astype("string"),
            "target_security_id": merged["target_security_id"].astype("string"),
            "target_ticker": merged["target_ticker"].astype("string"),
            "match_status": merged["match_status"].astype(str),
            "match_reason": merged["match_reason"].astype(str),
        }
    ).sort_values(["decision_time_utc", "source_decision_id"], kind="stable")
    return decisions, ledger.reset_index(drop=True)


def _rebind_coverage(
    source: pd.DataFrame,
    target: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    source_rows = source.copy()
    source_rows["_ticker_key"] = source_rows["ticker"].astype(str).str.upper()
    target_pairs = target.loc[:, ["ticker", "security_id"]].drop_duplicates()
    target_pairs["_ticker_key"] = target_pairs["ticker"].astype(str).str.upper()
    if bool(target_pairs.duplicated("_ticker_key", keep=False).any()):
        raise DataReadinessError(
            "coverage rebind found one ticker assigned to multiple target securities"
        )
    ticker_security = dict(
        zip(
            target_pairs["_ticker_key"].astype(str),
            target_pairs["security_id"].astype(str),
            strict=True,
        )
    )
    statuses: list[str] = []
    reasons: list[str] = []
    targets: list[str | None] = []
    for ticker_value in source_rows["_ticker_key"]:
        ticker = str(ticker_value)
        target_security = ticker_security.get(ticker)
        if target_security is None:
            statuses.append("unmatched")
            reasons.append("target_panel_excluded")
            targets.append(None)
            continue
        statuses.append("rebound")
        reasons.append("unique_ticker_in_target_population")
        targets.append(target_security)
    source_rows["target_security_id"] = pd.Series(targets, dtype="string")
    source_rows["match_status"] = statuses
    source_rows["match_reason"] = reasons
    matched = source_rows["match_status"].eq("rebound")
    coverage = source_rows.loc[matched, list(source.columns)].copy()
    coverage["security_id"] = source_rows.loc[matched, "target_security_id"].astype(str)
    coverage["ticker"] = coverage["ticker"].astype(str).str.upper()
    coverage["coverage_evidence_id"] = [
        _json_sha256(_normalized_coverage(row))
        for row in coverage.to_dict(orient="records")
    ]
    if bool(coverage["coverage_evidence_id"].duplicated().any()):
        raise DataReadinessError("coverage identity rebind produced duplicate evidence")
    ledger = pd.DataFrame(
        {
            "source_coverage_evidence_id": source_rows[
                "coverage_evidence_id"
            ].astype(str),
            "source_security_id": source_rows["security_id"].astype(str),
            "source_ticker": source_rows["ticker"].astype(str).str.upper(),
            "requested_start_utc": pd.to_datetime(
                source_rows["requested_start_utc"], utc=True, errors="raise"
            ),
            "requested_end_utc": pd.to_datetime(
                source_rows["requested_end_utc"], utc=True, errors="raise"
            ),
            "target_security_id": source_rows["target_security_id"],
            "match_status": source_rows["match_status"],
            "match_reason": source_rows["match_reason"],
        }
    ).sort_values(
        ["requested_start_utc", "source_coverage_evidence_id"], kind="stable"
    )
    return coverage.reset_index(drop=True), ledger.reset_index(drop=True)


def _load_target_spine(directory: Path) -> _TargetSpine:
    root = directory.resolve()
    final = root / "final"
    manifest_path = final / "_manifest.json"
    authority_path = final / "_authority.json"
    request_path = root / "_request.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    request = _json_object(request_path)
    request_payload = dict(request)
    embedded_request_sha256 = request_payload.pop("request_sha256", None)
    if (
        manifest.get("schema") != TARGET_PANEL_MANIFEST_SCHEMA
        or authority.get("schema") != TARGET_PANEL_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or manifest.get("request_sha256") != embedded_request_sha256
        or _json_sha256(request_payload) != embedded_request_sha256
        or authority.get("request_sha256") != embedded_request_sha256
    ):
        raise DataReadinessError("target swing identity spine does not verify")
    raw_files = _required_mapping(manifest, "files_by_profile").get(TARGET_PROFILE)
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("target swing panel has no technical identity spine")
    frames: list[pd.DataFrame] = []
    for raw_record in raw_files:
        if not isinstance(raw_record, dict):
            raise DataReadinessError("target swing panel file inventory is malformed")
        relative = Path(str(raw_record.get("path", "")))
        path = (final / relative).resolve()
        if not path.is_relative_to(final.resolve()) or not path.is_file():
            raise DataReadinessError("target swing panel file escapes its authority")
        if file_sha256(path) != raw_record.get("sha256"):
            raise DataReadinessError("target swing identity spine file hash mismatch")
        frame = pd.read_parquet(path, columns=list(_TARGET_COLUMNS))
        if len(frame) != int(raw_record.get("rows", -1)):
            raise DataReadinessError("target swing identity spine row count mismatch")
        frames.append(frame)
    spine = pd.concat(frames, ignore_index=True)
    if len(spine) != _integer(
        manifest.get("rows_per_ablation_panel"), "rows_per_ablation_panel"
    ):
        raise DataReadinessError("target swing identity spine total rows mismatch")
    required = set(_TARGET_COLUMNS)
    if required.difference(spine.columns):
        raise DataReadinessError("target swing identity spine is missing columns")
    decision_time = pd.to_datetime(spine["decision_time_utc"], utc=True, errors="raise")
    effective_from = pd.to_datetime(
        spine["membership_effective_from_utc"], utc=True, errors="raise"
    )
    effective_to = pd.to_datetime(
        spine["membership_effective_to_utc"], utc=True, errors="raise"
    )
    available = pd.to_datetime(
        spine["membership_available_at_utc"], utc=True, errors="raise"
    )
    if bool(
        spine["decision_id"].astype(str).duplicated().any()
        or spine["decision_id"].astype(str).eq("").any()
        or (available > decision_time).any()
        or (effective_from > decision_time).any()
        or (decision_time >= effective_to).any()
    ):
        raise DataReadinessError("target swing identity spine violates PIT identity")
    security_ids = sorted(spine["security_id"].astype(str).unique())
    if (
        len(security_ids)
        != _integer(manifest.get("modeled_security_count"), "modeled_security_count")
        or _json_sha256(security_ids)
        != manifest.get("modeled_security_ids_sha256")
    ):
        raise DataReadinessError("target swing identity population does not verify")
    return _TargetSpine(
        root=root,
        frame=spine,
        manifest=manifest,
        authority=authority,
        manifest_sha256=file_sha256(manifest_path),
        authority_sha256=file_sha256(authority_path),
    )


def _membership_intervals(
    target: pd.DataFrame,
) -> dict[str, list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]]:
    columns = [
        "ticker",
        "membership_effective_from_utc",
        "membership_effective_to_utc",
        "membership_available_at_utc",
    ]
    unique = target.loc[:, columns].drop_duplicates()
    result: dict[str, list[tuple[pd.Timestamp, pd.Timestamp, pd.Timestamp]]] = {}
    for row in unique.itertuples(index=False):
        ticker = str(row.ticker).upper()
        result.setdefault(ticker, []).append(
            (
                pd.Timestamp(row.membership_effective_from_utc),
                pd.Timestamp(row.membership_effective_to_utc),
                pd.Timestamp(row.membership_available_at_utc),
            )
        )
    return result


def _request(
    parent: CatalystDecisionAuthority,
    target: _TargetSpine,
) -> dict[str, object]:
    return {
        "schema": REBIND_REQUEST_SCHEMA,
        "parent": {
            "directory": str(parent.directory.resolve()),
            "manifest_sha256": file_sha256(parent.directory / "_manifest.json"),
            "authority_sha256": file_sha256(parent.directory / "_authority.json"),
            "request_sha256": parent.manifest["request_sha256"],
            "decision_artifact_sha256": parent.authority[
                "decision_artifact_sha256"
            ],
            "coverage_artifact_sha256": parent.authority[
                "coverage_artifact_sha256"
            ],
        },
        "target_panel": {
            "directory": str(target.root),
            "manifest_sha256": target.manifest_sha256,
            "authority_sha256": target.authority_sha256,
            "request_sha256": target.manifest["request_sha256"],
            "modeled_security_ids_sha256": target.manifest[
                "modeled_security_ids_sha256"
            ],
            "profile": TARGET_PROFILE,
        },
        "decision_match_policy": "exact_upper_ticker_and_decision_time_utc",
        "coverage_match_policy": "unique_target_ticker_in_governed_population",
        "production_ready": False,
    }


def _replay_sources(request: Mapping[str, object]) -> None:
    parent = _required_mapping(request, "parent")
    target = _required_mapping(request, "target_panel")
    loaded_parent = load_catalyst_decision_authority(Path(str(parent["directory"])))
    if (
        file_sha256(loaded_parent.directory / "_manifest.json")
        != parent.get("manifest_sha256")
        or file_sha256(loaded_parent.directory / "_authority.json")
        != parent.get("authority_sha256")
    ):
        raise DataReadinessError("catalyst rebind parent lineage changed")
    loaded_target = _load_target_spine(Path(str(target["directory"])))
    if (
        loaded_target.manifest_sha256 != target.get("manifest_sha256")
        or loaded_target.authority_sha256 != target.get("authority_sha256")
    ):
        raise DataReadinessError("catalyst rebind target lineage changed")


def _write_artifact(
    frame: pd.DataFrame,
    path: Path,
    *,
    artifact_type: str,
    inputs: Mapping[str, str],
    final_directory: Path,
    unique_column: str,
) -> dict[str, object]:
    audit = _audit(frame, artifact_type, unique_column)
    audit.raise_for_failure()
    child_manifest = write_canonical_artifact(
        frame,
        path,
        artifact_type=artifact_type,
        audit=audit,
        inputs=dict(inputs),
        production_ready=False,
    )
    path.with_suffix(path.suffix + ".lock").unlink(missing_ok=True)
    child_manifest_path = manifest_path_for(path)
    child_payload = _json_object(child_manifest_path)
    child_payload["artifact_path"] = str((final_directory / path.name).resolve())
    _atomic_json(child_manifest_path, child_payload)
    return {
        "path": path.name,
        "sha256": child_manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(child_manifest_path),
        "rows": len(frame),
        "artifact_type": artifact_type,
    }


def _load_bound_artifact(
    root: Path,
    artifacts: Mapping[str, object],
    key: str,
    filename: str,
    artifact_type: str,
    request_sha256: str,
) -> pd.DataFrame:
    record = artifacts.get(key)
    if not isinstance(record, dict):
        raise DataReadinessError(f"catalyst rebind artifact record is missing: {key}")
    path = root / filename
    frame, child_manifest = load_canonical_artifact(
        path,
        expected_type=artifact_type,
        allow_research=True,
    )
    inputs = _required_mapping(child_manifest, "inputs")
    if (
        record.get("path") != filename
        or record.get("sha256") != child_manifest.get("artifact_sha256")
        or record.get("manifest_sha256") != file_sha256(manifest_path_for(path))
        or int(record.get("rows", -1)) != len(frame)
        or inputs.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(f"catalyst rebind artifact does not verify: {key}")
    return frame


def _validate_replayed_rows(
    decisions: pd.DataFrame,
    coverage: pd.DataFrame,
    decision_ledger: pd.DataFrame,
    coverage_ledger: pd.DataFrame,
    manifest: Mapping[str, object],
    authority: Mapping[str, object],
) -> None:
    for frame, name, unique in (
        (decisions, "decisions", "decision_id"),
        (coverage, "coverage", "coverage_evidence_id"),
        (decision_ledger, "decision ledger", "source_decision_id"),
        (coverage_ledger, "coverage ledger", "source_coverage_evidence_id"),
    ):
        _audit(frame, name, unique).raise_for_failure()
    artifacts = _required_mapping(manifest, "artifacts")
    decision_record = _required_mapping(artifacts, "decisions")
    coverage_record = _required_mapping(artifacts, "coverage")
    if (
        len(decisions) != _integer(manifest.get("decision_rows"), "decision_rows")
        or len(coverage) != _integer(manifest.get("coverage_rows"), "coverage_rows")
        or _status_counts(decision_ledger)
        != manifest.get("decision_rebind_status_counts")
        or _status_counts(coverage_ledger)
        != manifest.get("coverage_rebind_status_counts")
        or authority.get("decision_artifact_sha256")
        != decision_record.get("sha256")
        or authority.get("coverage_artifact_sha256")
        != coverage_record.get("sha256")
    ):
        raise DataReadinessError("catalyst identity rebind row lineage mismatch")
    rebound_decisions = int(
        decision_ledger["match_status"].astype(str).eq("rebound").sum()
    )
    rebound_coverage = int(
        coverage_ledger["match_status"].astype(str).eq("rebound").sum()
    )
    if rebound_decisions != len(decisions) or rebound_coverage != len(coverage):
        raise DataReadinessError("catalyst identity rebind ledger does not reconcile")


def _normalized_coverage(record: Mapping[str, object]) -> dict[str, object]:
    return {
        "collection_id": str(record.get("collection_id", "")),
        "chunk_id": str(record.get("chunk_id", "")),
        "security_id": str(record.get("security_id", "")),
        "ticker": str(record.get("ticker", "")).upper(),
        "source_family": str(record.get("source_family", "")).lower(),
        "requested_start_utc": pd.Timestamp(
            record["requested_start_utc"]
        ).isoformat(),
        "requested_end_utc": pd.Timestamp(record["requested_end_utc"]).isoformat(),
        "completed_at_utc": pd.Timestamp(record["completed_at_utc"]).isoformat(),
        "status": str(record.get("status", "")),
        "row_count": _integer(record.get("row_count"), "coverage.row_count"),
        "coverage_state": str(record.get("coverage_state", "")),
        "missingness_known": bool(record.get("missingness_known")),
        "zero_event_semantics": str(record.get("zero_event_semantics", "")),
        "training_eligible": bool(record.get("training_eligible")),
        "schema_version": str(record.get("schema_version", "")),
    }


def _status_counts(frame: pd.DataFrame) -> dict[str, int]:
    counts = frame.groupby(["match_status", "match_reason"], dropna=False).size()
    return {
        f"{status}:{reason}": int(value)
        for (status, reason), value in sorted(counts.items())
    }


def _audit(
    frame: pd.DataFrame,
    name: str,
    unique_column: str,
) -> CanonicalAuditReport:
    failures = int(frame.empty or unique_column not in frame)
    if failures == 0:
        values = frame[unique_column].astype(str)
        failures += int(values.eq("").sum() + values.duplicated().sum())
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=len(frame),
                detail="identity migration rows are non-empty and unique",
            ),
        )
    )


def _required_mapping(
    payload: Mapping[str, object],
    key: str,
) -> Mapping[str, object]:
    value = payload.get(key)
    if not isinstance(value, dict):
        raise DataReadinessError(f"catalyst identity rebind {key} is malformed")
    return value


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(f"catalyst identity rebind {name} is not an integer")
    converted = int(value)
    if converted != value:
        raise DataReadinessError(f"catalyst identity rebind {name} is not an integer")
    return converted


def _json_object(path: Path) -> dict[str, object]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(payload, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): value for key, value in payload.items()}


def _json_sha256(value: object) -> str:
    compact = json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(compact.encode("utf-8")).hexdigest()


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8"
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
