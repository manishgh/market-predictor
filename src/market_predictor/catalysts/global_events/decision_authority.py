"""Immutable, decision-time global-event features with explicit source coverage."""
from __future__ import annotations

import gc
import hashlib
import json
import os
import shutil
import sqlite3
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Final
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
    audit_canonical_events,
    audit_source_collections,
)
from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)

GLOBAL_EVENT_AUTHORITY_SCHEMA: Final = "edge_rebuild.global_event_authority.v2"
GLOBAL_EVENT_MANIFEST_SCHEMA: Final = "edge_rebuild.global_event_manifest.v2"
GLOBAL_EVENT_REQUEST_SCHEMA: Final = "edge_rebuild.global_event_request.v2"
GLOBAL_DECISION_ARTIFACT_TYPE: Final = "edge_rebuild_global_event_decisions"
GLOBAL_COVERAGE_ARTIFACT_TYPE: Final = "edge_rebuild_global_source_coverage"
GLOBAL_TICKER: Final = "MARKET"
GLOBAL_SECURITY_ID: Final = "market:global"
GLOBAL_EVENT_SOURCE_FAMILIES: Final = (
    "alpaca",
    "gdelt",
)
WINDOWS: Final[Mapping[str, pd.Timedelta]] = {
    "1d": pd.Timedelta(days=1),
    "3d": pd.Timedelta(days=3),
}
MAXIMUM_PROCESS_MEMORY_GIB: Final = 4.0
MEMORY_GUARD_HEADROOM_GIB: Final = 0.5
_KNOWN_STATUSES: Final = frozenset({"observed", "observed_empty"})


@dataclass(frozen=True, slots=True)
class GlobalEventAuthority:
    directory: Path
    decisions: pd.DataFrame
    coverage: pd.DataFrame
    manifest: Mapping[str, object]
    authority: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _InputArtifact:
    path: Path
    artifact_sha256: str
    manifest_sha256: str
    rows: int
    production_ready: bool
    collection_request_sha256: str
    source_policy_sha256: str
    sentiment_scorer_identity: str
    source_families: tuple[str, ...]
    inputs: Mapping[str, object]


@dataclass(frozen=True, slots=True)
class _EventEvidence:
    collection_request_sha256: str
    source_family: str
    event_id: str
    published_ns: int
    feature_available_ns: int


@dataclass(frozen=True, slots=True)
class _CoverageEvidence:
    collection_request_sha256: str
    collection_id: str
    source_family: str
    requested_start_ns: int
    requested_end_ns: int
    completed_ns: int
    status: str
    row_count: int


def publish_global_event_authority(
    decisions: pd.DataFrame,
    event_artifacts: Sequence[Path],
    source_coverage_artifacts: Sequence[Path],
    output_directory: Path,
    *,
    required_historical_sources: Sequence[str],
    production_ready: bool,
    maximum_process_memory_gib: float = MAXIMUM_PROCESS_MEMORY_GIB,
    memory_guard_headroom_gib: float = MEMORY_GUARD_HEADROOM_GIB,
) -> GlobalEventAuthority:
    """Publish causal 1d/3d global features from hash-verified canonical inputs."""

    _validate_memory_policy(maximum_process_memory_gib, memory_guard_headroom_gib)
    sources = _normalize_sources(required_historical_sources)
    event_paths = _normalize_paths(event_artifacts, "global event")
    coverage_paths = _normalize_paths(source_coverage_artifacts, "global source coverage")
    decision_frame = _normalize_decisions(decisions)
    output_directory = output_directory.resolve()
    if output_directory.exists():
        raise DataReadinessError(f"global event authority is immutable: {output_directory}")
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging = output_directory.with_name(f".{output_directory.name}.{uuid4().hex}.tmp")
    staging.mkdir()
    database_path = staging / ".global_event_merge.sqlite3"
    event_inputs: list[_InputArtifact] = []
    coverage_inputs: list[_InputArtifact] = []
    event_evidence: list[_EventEvidence] = []
    coverage_evidence: list[_CoverageEvidence] = []
    event_rows_read = 0
    coverage_rows_read = 0
    database: sqlite3.Connection | None = None
    try:
        with sqlite3.connect(database_path) as database:
            _initialize_database(database)
            for path in event_paths:
                frame, child = load_canonical_artifact(
                    path,
                    expected_type="events",
                    allow_research=not production_ready,
                )
                CanonicalAuditReport(
                    checks=audit_canonical_events(
                        frame,
                        require_observed=production_ready,
                    )
                ).raise_for_failure()
                _validate_global_events(frame, sources)
                _require_production_input(child, path, production_ready)
                input_record = _input_record(
                    path,
                    child,
                    len(frame),
                    source_families=tuple(
                        sorted(
                            set(
                                frame["source_family"]
                                .astype(str)
                                .str.lower()
                                .str.strip()
                            )
                        )
                    ),
                )
                event_evidence.extend(_event_evidence(frame, input_record))
                _insert_events(database, frame)
                event_rows_read += len(frame)
                event_inputs.append(input_record)
                del frame
                _guard_and_release(
                    maximum_process_memory_gib,
                    memory_guard_headroom_gib,
                    f"global event input {path.name}",
                )
            for path in coverage_paths:
                frame, child = load_canonical_artifact(
                    path,
                    expected_type="source_collections",
                    allow_research=not production_ready,
                )
                CanonicalAuditReport(checks=audit_source_collections(frame, require_success=False)).raise_for_failure()
                _validate_global_coverage(frame, sources)
                _require_production_input(child, path, production_ready)
                input_record = _input_record(
                    path,
                    child,
                    len(frame),
                    source_families=tuple(
                        sorted(
                            set(
                                frame["source_family"]
                                .astype(str)
                                .str.lower()
                                .str.strip()
                            )
                        )
                    ),
                )
                coverage_evidence.extend(_coverage_evidence(frame, input_record))
                _insert_coverage(database, frame)
                coverage_rows_read += len(frame)
                coverage_inputs.append(input_record)
                del frame
                _guard_and_release(
                    maximum_process_memory_gib,
                    memory_guard_headroom_gib,
                    f"global coverage input {path.name}",
                )
            database.commit()
            _validate_cross_artifact_lineage(event_inputs, coverage_inputs)
            _reconcile_collection_evidence(event_evidence, coverage_evidence)
            aggregates = _aggregate_decisions(
                database,
                decision_frame,
                sources,
                production_ready=production_ready,
            )
            coverage = _export_coverage(database)
            unique_event_rows = int(database.execute("SELECT COUNT(*) FROM events").fetchone()[0])
            unique_coverage_rows = int(database.execute("SELECT COUNT(*) FROM coverage").fetchone()[0])
        database.close()
        database_path.unlink(missing_ok=True)
        completion = aggregates[[f"global_source_complete_{window}" for window in WINDOWS]].to_numpy(dtype=bool)
        if production_ready and not bool(completion.all()):
            raise DataReadinessError(
                "production global event authority requires complete explicit source coverage for every decision and lookback"
            )
        request = _request_payload(
            decision_frame,
            sources,
            event_inputs,
            coverage_inputs,
            production_ready,
        )
        request_sha256 = _json_sha256(request)
        source_lineage_sha256 = _json_sha256(request["source_artifacts"])
        common_inputs = {
            "request_sha256": request_sha256,
            "source_lineage_sha256": source_lineage_sha256,
        }
        decision_path = staging / "decision_global_events.parquet"
        coverage_path = staging / "source_coverage.parquet"
        decision_manifest = write_canonical_artifact(
            aggregates,
            decision_path,
            artifact_type=GLOBAL_DECISION_ARTIFACT_TYPE,
            audit=_decision_audit(aggregates, sources, production_ready),
            inputs=common_inputs,
            production_ready=production_ready,
        )
        coverage_manifest = write_canonical_artifact(
            coverage,
            coverage_path,
            artifact_type=GLOBAL_COVERAGE_ARTIFACT_TYPE,
            audit=_coverage_audit(coverage),
            inputs=common_inputs,
            production_ready=production_ready,
        )
        decision_path.with_suffix(".parquet.lock").unlink(missing_ok=True)
        coverage_path.with_suffix(".parquet.lock").unlink(missing_ok=True)
        _rewrite_artifact_path(decision_path, output_directory)
        _rewrite_artifact_path(coverage_path, output_directory)
        manifest: dict[str, object] = {
            "schema": GLOBAL_EVENT_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "required_historical_sources": list(sources),
            "windows": {name: int(duration.total_seconds()) for name, duration in WINDOWS.items()},
            "event_rows_read": event_rows_read,
            "unique_event_rows": unique_event_rows,
            "duplicate_event_rows_merged": event_rows_read - unique_event_rows,
            "coverage_rows_read": coverage_rows_read,
            "unique_coverage_rows": unique_coverage_rows,
            "duplicate_coverage_rows_merged": coverage_rows_read - unique_coverage_rows,
            "decision_rows": len(aggregates),
            "coverage_rows": len(coverage),
            "artifacts": {
                "decisions": _artifact_record(decision_path, decision_manifest),
                "coverage": _artifact_record(coverage_path, coverage_manifest),
            },
            "memory": memory_audit(
                hard_budget_gib=maximum_process_memory_gib,
                headroom_gib=memory_guard_headroom_gib,
            ).to_record(),
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "production_ready": production_ready,
            "missing_value_policy": ("zero requires full source coverage for the exact lookback; unverified source history remains null"),
        }
        _atomic_json(staging / "_manifest.json", manifest)
        authority = {
            "schema": GLOBAL_EVENT_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "source_lineage_sha256": source_lineage_sha256,
            "decision_artifact_sha256": decision_manifest["artifact_sha256"],
            "coverage_artifact_sha256": coverage_manifest["artifact_sha256"],
            "production_ready": production_ready,
        }
        _atomic_json(staging / "_authority.json", authority)
        load_global_event_authority(
            staging,
            require_production_ready=production_ready,
        )
        os.replace(staging, output_directory)
        return load_global_event_authority(
            output_directory,
            require_production_ready=production_ready,
        )
    except Exception:
        if database is not None:
            database.close()
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_global_event_authority(
    directory: Path,
    *,
    require_production_ready: bool = True,
) -> GlobalEventAuthority:
    """Strictly verify an immutable global-event authority before loading it."""

    directory = directory.resolve()
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    if manifest.get("schema") != GLOBAL_EVENT_MANIFEST_SCHEMA or manifest.get("state") != "complete":
        raise DataReadinessError("global event manifest is not complete")
    production_ready = bool(manifest.get("production_ready"))
    if require_production_ready and not production_ready:
        raise DataReadinessError("research global event authority is not production ready")
    if (
        authority.get("schema") != GLOBAL_EVENT_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("request_sha256") != manifest.get("request_sha256")
        or authority.get("source_lineage_sha256") != manifest.get("source_lineage_sha256")
        or bool(authority.get("production_ready")) != production_ready
    ):
        raise DataReadinessError("global event authority does not verify")
    request = manifest.get("request")
    if not isinstance(request, dict) or _json_sha256(request) != manifest.get("request_sha256"):
        raise DataReadinessError("global event authority request hash does not verify")
    if bool(request.get("production_ready")) != production_ready:
        raise DataReadinessError("global event production policy does not verify")
    if _json_sha256(request.get("source_artifacts")) != manifest.get("source_lineage_sha256"):
        raise DataReadinessError("global event source lineage hash does not verify")
    sources = _normalize_sources(_required_sequence(request, "required_historical_sources"))
    if list(sources) != manifest.get("required_historical_sources"):
        raise DataReadinessError("global event required source contract does not verify")
    expected_files = {
        "_authority.json",
        "_manifest.json",
        "decision_global_events.parquet",
        "decision_global_events.parquet.manifest.json",
        "source_coverage.parquet",
        "source_coverage.parquet.manifest.json",
    }
    observed_files = {path.name for path in directory.iterdir() if path.is_file()}
    if observed_files != expected_files or any(path.is_dir() for path in directory.iterdir()):
        raise DataReadinessError("global event authority inventory does not verify")
    decisions, decision_manifest = load_canonical_artifact(
        directory / "decision_global_events.parquet",
        expected_type=GLOBAL_DECISION_ARTIFACT_TYPE,
        allow_research=not require_production_ready,
    )
    coverage, coverage_manifest = load_canonical_artifact(
        directory / "source_coverage.parquet",
        expected_type=GLOBAL_COVERAGE_ARTIFACT_TYPE,
        allow_research=not require_production_ready,
    )
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict):
        raise DataReadinessError("global event artifact inventory is malformed")
    _verify_published_artifact(
        artifacts.get("decisions"),
        decision_manifest,
        directory / "decision_global_events.parquet",
        len(decisions),
    )
    _verify_published_artifact(
        artifacts.get("coverage"),
        coverage_manifest,
        directory / "source_coverage.parquet",
        len(coverage),
    )
    for child in (decision_manifest, coverage_manifest):
        inputs = child.get("inputs")
        if not isinstance(inputs, dict) or (
            inputs.get("request_sha256") != manifest.get("request_sha256")
            or inputs.get("source_lineage_sha256") != manifest.get("source_lineage_sha256")
        ):
            raise DataReadinessError("global event child artifact lineage does not verify")
    if (
        authority.get("decision_artifact_sha256") != decision_manifest.get("artifact_sha256")
        or authority.get("coverage_artifact_sha256") != coverage_manifest.get("artifact_sha256")
        or len(decisions) != _integer(manifest.get("decision_rows"), "decision_rows")
        or len(coverage) != _integer(manifest.get("coverage_rows"), "coverage_rows")
    ):
        raise DataReadinessError("global event authority artifact or row lineage mismatch")
    _decision_audit(decisions, sources, production_ready).raise_for_failure()
    _coverage_audit(coverage).raise_for_failure()
    return GlobalEventAuthority(
        directory=directory,
        decisions=decisions,
        coverage=coverage,
        manifest=manifest,
        authority=authority,
    )


def attach_global_event_features(
    decisions: pd.DataFrame,
    authority: GlobalEventAuthority | Path,
    *,
    require_production_ready: bool = True,
) -> pd.DataFrame:
    """Attach global features by exact decision timestamp; as-of fallback is forbidden."""

    directory = authority if isinstance(authority, Path) else authority.directory
    loaded = load_global_event_authority(
        directory,
        require_production_ready=require_production_ready,
    )
    if require_production_ready and not bool(loaded.manifest.get("production_ready")):
        raise DataReadinessError("research global event authority is not production ready")
    output = decisions.copy()
    if "decision_time_utc" not in output.columns:
        raise DataReadinessError("global event attachment requires decision_time_utc")
    output["decision_time_utc"] = _strict_utc_series(
        output["decision_time_utc"],
        "decision_time_utc",
    )
    authority_columns = [column for column in loaded.decisions.columns if column != "decision_time_utc"]
    collisions = sorted(set(authority_columns).intersection(output.columns))
    if collisions:
        raise DataReadinessError("global event attachment would overwrite columns: " + ", ".join(collisions))
    attached = output.merge(
        loaded.decisions,
        on="decision_time_utc",
        how="left",
        validate="many_to_one",
        indicator=True,
        sort=False,
    )
    if bool(attached["_merge"].ne("both").any()):
        raise DataReadinessError("global event authority has no exact row for one or more decision timestamps")
    attached = attached.drop(columns="_merge")
    for window in WINDOWS:
        latest = pd.to_datetime(
            attached[f"global_latest_event_feature_available_at_utc_{window}"],
            utc=True,
            errors="coerce",
        )
        if bool((latest > attached["decision_time_utc"]).fillna(False).any()):
            raise DataReadinessError("global event authority contains future evidence")
    return attached


def _initialize_database(database: sqlite3.Connection) -> None:
    database.execute("PRAGMA journal_mode=OFF")
    database.execute("PRAGMA synchronous=OFF")
    database.execute(
        """
        CREATE TABLE events (
            event_id TEXT PRIMARY KEY,
            payload_sha256 TEXT NOT NULL,
            source_family TEXT NOT NULL,
            feature_available_ns INTEGER NOT NULL,
            sentiment_numeric REAL,
            relevance REAL,
            availability_policy TEXT NOT NULL
        )
        """
    )
    database.execute("CREATE INDEX events_time_source_idx ON events(source_family, feature_available_ns)")
    database.execute(
        """
        CREATE TABLE coverage (
            collection_id TEXT PRIMARY KEY,
            payload_sha256 TEXT NOT NULL,
            source_family TEXT NOT NULL,
            requested_start_ns INTEGER NOT NULL,
            requested_end_ns INTEGER NOT NULL,
            completed_ns INTEGER NOT NULL,
            status TEXT NOT NULL,
            row_count INTEGER NOT NULL
        )
        """
    )
    database.execute("CREATE INDEX coverage_source_time_idx ON coverage(source_family, completed_ns)")


def _insert_events(database: sqlite3.Connection, frame: pd.DataFrame) -> None:
    for record in frame.to_dict(orient="records"):
        canonical = CanonicalEvent.model_validate(_none_for_missing(record)).model_dump()
        normalized = {
            "event_id": canonical["event_id"],
            "source_family": canonical["source_family"],
            "feature_available_ns": _timestamp_ns(
                canonical["feature_available_at_utc"],
                "feature_available_at_utc",
            ),
            "sentiment_numeric": _nullable_float(canonical["sentiment_numeric"]),
            "relevance": _nullable_nonnegative_float(canonical["relevance"]),
            "availability_policy": canonical["availability_policy"],
        }
        payload_sha256 = _json_sha256(_json_compatible(canonical))
        existing = database.execute(
            "SELECT payload_sha256 FROM events WHERE event_id = ?",
            (normalized["event_id"],),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_sha256:
                raise DataReadinessError("conflicting duplicate global event evidence")
            continue
        database.execute(
            "INSERT INTO events VALUES (?, ?, ?, ?, ?, ?, ?)",
            (
                normalized["event_id"],
                payload_sha256,
                normalized["source_family"],
                normalized["feature_available_ns"],
                normalized["sentiment_numeric"],
                normalized["relevance"],
                normalized["availability_policy"],
            ),
        )


def _insert_coverage(database: sqlite3.Connection, frame: pd.DataFrame) -> None:
    for record in frame.to_dict(orient="records"):
        canonical = SourceCollection.model_validate(_none_for_missing(record)).model_dump()
        normalized = {
            "collection_id": canonical["collection_id"],
            "source_family": canonical["source_family"],
            "requested_start_ns": _timestamp_ns(
                canonical["requested_start_utc"],
                "requested_start_utc",
            ),
            "requested_end_ns": _timestamp_ns(
                canonical["requested_end_utc"],
                "requested_end_utc",
            ),
            "completed_ns": _timestamp_ns(
                canonical["completed_at_utc"],
                "completed_at_utc",
            ),
            "status": canonical["status"],
            "row_count": canonical["row_count"],
        }
        payload_sha256 = _json_sha256(_json_compatible(canonical))
        existing = database.execute(
            "SELECT payload_sha256 FROM coverage WHERE collection_id = ?",
            (normalized["collection_id"],),
        ).fetchone()
        if existing is not None:
            if str(existing[0]) != payload_sha256:
                raise DataReadinessError("conflicting duplicate global coverage evidence")
            continue
        database.execute(
            "INSERT INTO coverage VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
            (
                normalized["collection_id"],
                payload_sha256,
                normalized["source_family"],
                normalized["requested_start_ns"],
                normalized["requested_end_ns"],
                normalized["completed_ns"],
                normalized["status"],
                normalized["row_count"],
            ),
        )


def _aggregate_decisions(
    database: sqlite3.Connection,
    decisions: pd.DataFrame,
    sources: tuple[str, ...],
    *,
    production_ready: bool,
) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    for row in decisions.itertuples(index=False):
        decision_time = pd.Timestamp(row.decision_time_utc)
        decision_ns = int(decision_time.value)
        output: dict[str, object] = {
            "decision_time_id": str(row.decision_time_id),
            "decision_time_utc": decision_time,
        }
        for window_name, duration in WINDOWS.items():
            start_ns = decision_ns - int(duration.value)
            source_known: list[bool] = []
            for source in sources:
                known, start, end, available = _coverage_state(
                    database,
                    source,
                    start_ns,
                    decision_ns,
                    require_causal_completion=production_ready,
                )
                source_known.append(known)
                output[f"global_source_coverage_known_{source}_{window_name}"] = known
                if known:
                    if start is None or end is None or available is None:
                        raise DataReadinessError("known global coverage has incomplete interval evidence")
                    coverage_start: object = _timestamp_from_ns(start)
                    coverage_end: object = _timestamp_from_ns(end)
                    coverage_available: object = _timestamp_from_ns(available)
                else:
                    coverage_start = pd.NaT
                    coverage_end = pd.NaT
                    coverage_available = pd.NaT
                output[f"global_source_coverage_start_utc_{source}_{window_name}"] = coverage_start
                output[f"global_source_coverage_end_utc_{source}_{window_name}"] = coverage_end
                output[f"global_source_coverage_available_at_utc_{source}_{window_name}"] = coverage_available
                count = _event_count(database, source, start_ns, decision_ns)
                output[f"global_source_count_{source}_{window_name}"] = float(count) if known else np.nan
            complete = all(source_known)
            output[f"global_source_complete_{window_name}"] = complete
            if complete:
                metrics = _event_metrics(database, sources, start_ns, decision_ns)
                output[f"global_event_count_{window_name}"] = float(metrics[0])
                output[f"global_sentiment_mean_{window_name}"] = metrics[1]
                output[f"global_sentiment_coverage_{window_name}"] = metrics[2]
                output[f"global_latest_event_feature_available_at_utc_{window_name}"] = (
                    _timestamp_from_ns(metrics[3]) if metrics[3] is not None else pd.NaT
                )
            else:
                output[f"global_event_count_{window_name}"] = np.nan
                output[f"global_sentiment_mean_{window_name}"] = np.nan
                output[f"global_sentiment_coverage_{window_name}"] = np.nan
                output[f"global_latest_event_feature_available_at_utc_{window_name}"] = pd.NaT
        records.append(output)
    return pd.DataFrame.from_records(records)


def _coverage_state(
    database: sqlite3.Connection,
    source: str,
    lookback_start_ns: int,
    decision_ns: int,
    *,
    require_causal_completion: bool,
) -> tuple[bool, int | None, int | None, int | None]:
    completion_predicate = "AND completed_ns <= ?" if require_causal_completion else ""
    parameters: tuple[object, ...] = (
        (source, lookback_start_ns, decision_ns, decision_ns)
        if require_causal_completion
        else (source, lookback_start_ns, decision_ns)
    )
    rows = database.execute(
        f"""
        SELECT requested_start_ns, requested_end_ns, completed_ns
        FROM coverage
        WHERE source_family = ?
          AND requested_end_ns >= ? AND requested_start_ns <= ?
          AND status IN ('observed', 'observed_empty')
          {completion_predicate}
        ORDER BY requested_start_ns, requested_end_ns
        """,
        parameters,
    ).fetchall()
    merged: list[list[int]] = []
    availability: list[int] = []
    for start, end, completed in rows:
        start_value, end_value, completed_value = int(start), int(end), int(completed)
        if not merged or start_value > merged[-1][1]:
            merged.append([start_value, end_value])
            availability.append(completed_value)
        else:
            merged[-1][1] = max(merged[-1][1], end_value)
            availability[-1] = max(availability[-1], completed_value)
    for interval, completed in zip(merged, availability, strict=True):
        if interval[0] <= lookback_start_ns and interval[1] >= decision_ns:
            return True, interval[0], interval[1], completed
    return False, None, None, None


def _event_count(
    database: sqlite3.Connection,
    source: str,
    start_ns: int,
    decision_ns: int,
) -> int:
    return int(
        database.execute(
            """
            SELECT COUNT(*) FROM events
            WHERE source_family = ? AND feature_available_ns >= ?
              AND feature_available_ns <= ?
            """,
            (source, start_ns, decision_ns),
        ).fetchone()[0]
    )


def _event_metrics(
    database: sqlite3.Connection,
    sources: tuple[str, ...],
    start_ns: int,
    decision_ns: int,
) -> tuple[int, float, float, int | None]:
    placeholders = ",".join("?" for _ in sources)
    rows = database.execute(
        f"""
        SELECT sentiment_numeric, relevance, feature_available_ns FROM events
        WHERE source_family IN ({placeholders}) AND feature_available_ns >= ?
          AND feature_available_ns <= ?
        """,
        (*sources, start_ns, decision_ns),
    ).fetchall()
    if not rows:
        return 0, 0.0, 0.0, None
    weighted = 0.0
    weight = 0.0
    scored = 0
    latest: int | None = None
    for sentiment, relevance, available in rows:
        latest = max(latest or int(available), int(available))
        if sentiment is None:
            continue
        scored += 1
        event_weight = max(float(relevance), 0.0) if relevance is not None else 1.0
        weighted += float(sentiment) * event_weight
        weight += event_weight
    return (
        len(rows),
        weighted / weight if weight > 0 else 0.0,
        scored / len(rows),
        latest,
    )


def _export_coverage(database: sqlite3.Connection) -> pd.DataFrame:
    rows = database.execute(
        """
        SELECT collection_id, source_family, requested_start_ns, requested_end_ns,
               completed_ns, status, row_count, payload_sha256
        FROM coverage ORDER BY source_family, requested_start_ns, collection_id
        """
    ).fetchall()
    return pd.DataFrame.from_records(
        [
            {
                "collection_id": row[0],
                "ticker": GLOBAL_TICKER,
                "security_id": GLOBAL_SECURITY_ID,
                "source_family": row[1],
                "requested_start_utc": _timestamp_from_ns(int(row[2])),
                "requested_end_utc": _timestamp_from_ns(int(row[3])),
                "completed_at_utc": _timestamp_from_ns(int(row[4])),
                "status": row[5],
                "row_count": int(row[6]),
                "coverage_evidence_sha256": row[7],
                "missingness_known": row[5] in _KNOWN_STATUSES,
                "zero_event_semantics": (
                    "known_zero_events" if row[5] == "observed_empty" else "observed_history" if row[5] == "observed" else "unknown"
                ),
            }
            for row in rows
        ]
    )


def _validate_global_events(frame: pd.DataFrame, sources: tuple[str, ...]) -> None:
    tickers = frame["ticker"].astype(str).str.upper().str.strip()
    securities = frame["security_id"].astype(str).str.lower().str.strip()
    families = frame["source_family"].astype(str).str.lower().str.strip()
    if bool(tickers.ne(GLOBAL_TICKER).any() or securities.ne(GLOBAL_SECURITY_ID).any()):
        raise DataReadinessError("global authority rejects ticker events; only MARKET/market:global is allowed")
    unexpected = sorted(set(families).difference(sources))
    if unexpected:
        raise DataReadinessError("global event source was not explicitly declared: " + ", ".join(unexpected))


def _validate_global_coverage(frame: pd.DataFrame, sources: tuple[str, ...]) -> None:
    tickers = frame["ticker"].astype(str).str.upper().str.strip()
    families = frame["source_family"].astype(str).str.lower().str.strip()
    if bool(tickers.ne(GLOBAL_TICKER).any()):
        raise DataReadinessError("global source coverage must use ticker MARKET")
    unexpected = sorted(set(families).difference(sources))
    if unexpected:
        raise DataReadinessError("global coverage source was not explicitly declared: " + ", ".join(unexpected))


def _event_evidence(
    frame: pd.DataFrame,
    artifact: _InputArtifact,
) -> list[_EventEvidence]:
    evidence: list[_EventEvidence] = []
    for record in frame.to_dict(orient="records"):
        canonical = CanonicalEvent.model_validate(
            _none_for_missing(record)
        ).model_dump()
        evidence.append(
            _EventEvidence(
                collection_request_sha256=artifact.collection_request_sha256,
                source_family=str(canonical["source_family"]),
                event_id=str(canonical["event_id"]),
                published_ns=_timestamp_ns(
                    canonical["published_at_utc"],
                    "published_at_utc",
                ),
                feature_available_ns=_timestamp_ns(
                    canonical["feature_available_at_utc"],
                    "feature_available_at_utc",
                ),
            )
        )
    return evidence


def _coverage_evidence(
    frame: pd.DataFrame,
    artifact: _InputArtifact,
) -> list[_CoverageEvidence]:
    evidence: list[_CoverageEvidence] = []
    for record in frame.to_dict(orient="records"):
        canonical = SourceCollection.model_validate(
            _none_for_missing(record)
        ).model_dump()
        evidence.append(
            _CoverageEvidence(
                collection_request_sha256=artifact.collection_request_sha256,
                collection_id=str(canonical["collection_id"]),
                source_family=str(canonical["source_family"]),
                requested_start_ns=_timestamp_ns(
                    canonical["requested_start_utc"],
                    "requested_start_utc",
                ),
                requested_end_ns=_timestamp_ns(
                    canonical["requested_end_utc"],
                    "requested_end_utc",
                ),
                completed_ns=_timestamp_ns(
                    canonical["completed_at_utc"],
                    "completed_at_utc",
                ),
                status=str(canonical["status"]),
                row_count=int(canonical["row_count"]),
            )
        )
    return evidence


def _reconcile_collection_evidence(
    events: Sequence[_EventEvidence],
    coverage: Sequence[_CoverageEvidence],
) -> None:
    events_by_request_source: dict[tuple[str, str], dict[str, _EventEvidence]] = {}
    for event in events:
        key = (event.collection_request_sha256, event.source_family)
        existing = events_by_request_source.setdefault(key, {}).get(event.event_id)
        if existing is not None and existing != event:
            raise DataReadinessError(
                "conflicting duplicate global event collection evidence"
            )
        events_by_request_source[key][event.event_id] = event

    coverage_by_request_source: dict[
        tuple[str, str], list[_CoverageEvidence]
    ] = {}
    for collection in coverage:
        key = (
            collection.collection_request_sha256,
            collection.source_family,
        )
        coverage_by_request_source.setdefault(key, []).append(collection)
        matching = {
            event.event_id
            for event in events_by_request_source.get(key, {}).values()
            if collection.requested_start_ns
            <= event.published_ns
            <= collection.requested_end_ns
            and event.feature_available_ns <= collection.completed_ns
        }
        if len(matching) != collection.row_count:
            raise DataReadinessError(
                "global source collection row_count does not reconcile with "
                f"canonical events: {collection.collection_id}"
            )
        expected_status = "observed" if matching else "observed_empty"
        if collection.status in _KNOWN_STATUSES and collection.status != expected_status:
            raise DataReadinessError(
                "global source collection status does not reconcile with "
                f"canonical events: {collection.collection_id}"
            )

    for key, request_events in events_by_request_source.items():
        collections = coverage_by_request_source.get(key, ())
        for event in request_events.values():
            if not any(
                collection.requested_start_ns
                <= event.published_ns
                <= collection.requested_end_ns
                and event.feature_available_ns <= collection.completed_ns
                for collection in collections
            ):
                raise DataReadinessError(
                    "global canonical event falls outside matching source "
                    "collection window"
                )


def _normalize_decisions(frame: pd.DataFrame) -> pd.DataFrame:
    if "decision_time_utc" not in frame.columns:
        raise DataReadinessError("global event authority decisions require decision_time_utc")
    times = _strict_utc_series(frame["decision_time_utc"], "decision_time_utc")
    unique = (
        pd.DataFrame({"decision_time_utc": times})
        .drop_duplicates()
        .sort_values(
            "decision_time_utc",
            kind="stable",
        )
    )
    if unique.empty:
        raise DataReadinessError("global event authority requires at least one decision time")
    unique["decision_time_id"] = unique["decision_time_utc"].map(
        lambda value: _json_sha256({"decision_time_utc": pd.Timestamp(value).isoformat()})
    )
    return unique[["decision_time_id", "decision_time_utc"]].reset_index(drop=True)


def _strict_utc_series(values: pd.Series, name: str) -> pd.Series:
    parsed: list[pd.Timestamp] = []
    for value in values:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError) as exc:
            raise DataReadinessError(f"{name} contains an invalid timestamp") from exc
        if pd.isna(timestamp) or timestamp.tzinfo is None:
            raise DataReadinessError(f"{name} must contain timezone-aware timestamps")
        parsed.append(timestamp.tz_convert("UTC"))
    return pd.Series(pd.DatetimeIndex(parsed), index=values.index)


def _normalize_sources(values: Sequence[object]) -> tuple[str, ...]:
    normalized = tuple(sorted({str(value).strip().lower() for value in values if str(value).strip()}))
    if not normalized:
        raise DataReadinessError("required_historical_sources must be explicit and non-empty")
    if any(not value.replace("_", "").isalnum() for value in normalized):
        raise DataReadinessError("required historical source names must be identifier-safe")
    unsupported = sorted(set(normalized).difference(GLOBAL_EVENT_SOURCE_FAMILIES))
    if unsupported:
        raise DataReadinessError(
            "unsupported global event source families: " + ", ".join(unsupported)
        )
    return normalized


def _normalize_paths(values: Sequence[Path], label: str) -> tuple[Path, ...]:
    paths = tuple(sorted({Path(value).resolve() for value in values}, key=str))
    if not paths:
        raise DataReadinessError(f"at least one {label} artifact is required")
    return paths


def _require_production_input(
    manifest: Mapping[str, object],
    path: Path,
    production_ready: bool,
) -> None:
    if production_ready and not bool(manifest.get("production_ready")):
        raise DataReadinessError(f"production authority rejects research input: {path}")


def _input_record(
    path: Path,
    manifest: Mapping[str, object],
    rows: int,
    *,
    source_families: tuple[str, ...],
) -> _InputArtifact:
    inputs = manifest.get("inputs")
    if not isinstance(inputs, dict):
        raise DataReadinessError(f"canonical input lineage is malformed: {path}")
    collection_request_sha256 = _sha256_text(
        inputs.get("collection_request_sha256"),
        "collection_request_sha256",
    )
    source_policy_sha256 = _sha256_text(
        inputs.get("source_policy_sha256"),
        "source_policy_sha256",
    )
    scorer_value = inputs.get(
        "sentiment_scorer_identity",
        inputs.get("scorer_identity"),
    )
    if not isinstance(scorer_value, str):
        raise DataReadinessError(
            "global event sentiment_scorer_identity is missing from artifact inputs"
        )
    sentiment_scorer_identity = _text(scorer_value, "sentiment_scorer_identity")
    return _InputArtifact(
        path=path.resolve(),
        artifact_sha256=_sha256_text(manifest.get("artifact_sha256"), "artifact_sha256"),
        manifest_sha256=file_sha256(manifest_path_for(path)),
        rows=rows,
        production_ready=bool(manifest.get("production_ready")),
        collection_request_sha256=collection_request_sha256,
        source_policy_sha256=source_policy_sha256,
        sentiment_scorer_identity=sentiment_scorer_identity,
        source_families=source_families,
        inputs={str(key): value for key, value in inputs.items()},
    )


def _validate_cross_artifact_lineage(
    events: Sequence[_InputArtifact],
    coverage: Sequence[_InputArtifact],
) -> None:
    coverage_sources: dict[str, set[str]] = {}
    for artifact in coverage:
        coverage_sources.setdefault(artifact.collection_request_sha256, set()).update(artifact.source_families)
    for artifact in events:
        backed_sources = coverage_sources.get(artifact.collection_request_sha256, set())
        missing = sorted(set(artifact.source_families).difference(backed_sources))
        if missing:
            raise DataReadinessError("global event artifact has no matching source-coverage lineage: " + ", ".join(missing))
    policies: dict[str, set[str]] = {}
    for artifact in (*events, *coverage):
        for source in artifact.source_families:
            policies.setdefault(source, set()).add(artifact.source_policy_sha256)
    ambiguous = sorted(source for source, values in policies.items() if len(values) != 1)
    if ambiguous:
        raise DataReadinessError(
            "global event source policy is inconsistent across artifacts: "
            + ", ".join(ambiguous)
        )
    scorer_identities: dict[str, set[str]] = {}
    for artifact in (*events, *coverage):
        for source in artifact.source_families:
            scorer_identities.setdefault(source, set()).add(
                artifact.sentiment_scorer_identity
            )
    mixed_scorers = sorted(
        source for source, values in scorer_identities.items() if len(values) != 1
    )
    if mixed_scorers:
        raise DataReadinessError(
            "global event sentiment scorer identity is inconsistent across artifacts: "
            + ", ".join(mixed_scorers)
        )


def _request_payload(
    decisions: pd.DataFrame,
    sources: tuple[str, ...],
    events: Sequence[_InputArtifact],
    coverage: Sequence[_InputArtifact],
    production_ready: bool,
) -> dict[str, object]:
    return {
        "schema": GLOBAL_EVENT_REQUEST_SCHEMA,
        "decision_times_sha256": _json_sha256([pd.Timestamp(value).isoformat() for value in decisions["decision_time_utc"]]),
        "decision_rows": len(decisions),
        "windows": {name: int(duration.total_seconds()) for name, duration in WINDOWS.items()},
        "required_historical_sources": list(sources),
        "global_identity": {
            "ticker": GLOBAL_TICKER,
            "security_id": GLOBAL_SECURITY_ID,
        },
        "production_ready": production_ready,
        "source_artifacts": {
            "events": [_input_payload(value) for value in sorted(events, key=lambda item: str(item.path))],
            "coverage": [_input_payload(value) for value in sorted(coverage, key=lambda item: str(item.path))],
        },
        "causal_policy": "feature_available_at_utc within exact decision lookback",
        "coverage_completion_policy": (
            "collection completed_at_utc must be at or before decision_time_utc"
            if production_ready
            else "retrospective research backfill may complete after decision_time_utc"
        ),
        "missing_value_policy": "known zero requires full observed source interval; otherwise null",
    }


def _input_payload(value: _InputArtifact) -> dict[str, object]:
    return {
        "path": str(value.path),
        "artifact_sha256": value.artifact_sha256,
        "manifest_sha256": value.manifest_sha256,
        "rows": value.rows,
        "production_ready": value.production_ready,
        "collection_request_sha256": value.collection_request_sha256,
        "source_policy_sha256": value.source_policy_sha256,
        "sentiment_scorer_identity": value.sentiment_scorer_identity,
        "source_families": list(value.source_families),
        "upstream_inputs": dict(value.inputs),
    }


def _decision_audit(
    frame: pd.DataFrame,
    sources: tuple[str, ...],
    production_ready: bool,
) -> CanonicalAuditReport:
    required = {"decision_time_id", "decision_time_utc"}
    for window in WINDOWS:
        required.update(
            {
                f"global_source_complete_{window}",
                f"global_event_count_{window}",
                f"global_sentiment_mean_{window}",
                f"global_sentiment_coverage_{window}",
                f"global_latest_event_feature_available_at_utc_{window}",
            }
        )
        for source in sources:
            required.update(
                {
                    f"global_source_count_{source}_{window}",
                    f"global_source_coverage_known_{source}_{window}",
                    f"global_source_coverage_start_utc_{source}_{window}",
                    f"global_source_coverage_end_utc_{source}_{window}",
                    f"global_source_coverage_available_at_utc_{source}_{window}",
                }
            )
    failures = len(required.difference(frame.columns)) + int(frame.empty)
    if failures == 0:
        decision_time = pd.to_datetime(frame["decision_time_utc"], utc=True, errors="coerce")
        failures += int(decision_time.isna().sum())
        failures += int(frame["decision_time_id"].astype(str).duplicated().sum())
        for window in WINDOWS:
            complete = frame[f"global_source_complete_{window}"].fillna(False).astype(bool)
            event_count = pd.to_numeric(frame[f"global_event_count_{window}"], errors="coerce")
            latest = pd.to_datetime(
                frame[f"global_latest_event_feature_available_at_utc_{window}"],
                utc=True,
                errors="coerce",
            )
            failures += int((latest > decision_time).fillna(False).sum())
            failures += int((complete & event_count.isna()).sum())
            failures += int((~complete & event_count.notna()).sum())
            if production_ready:
                failures += int((~complete).sum())
            for source in sources:
                known = frame[f"global_source_coverage_known_{source}_{window}"].astype(bool)
                count = pd.to_numeric(
                    frame[f"global_source_count_{source}_{window}"],
                    errors="coerce",
                )
                failures += int((known & count.isna()).sum())
                failures += int((~known & count.notna()).sum())
    return _audit_report("global_event_decision_authority", failures, len(frame))


def _coverage_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    required = {
        "collection_id",
        "ticker",
        "security_id",
        "source_family",
        "requested_start_utc",
        "requested_end_utc",
        "completed_at_utc",
        "status",
        "row_count",
        "coverage_evidence_sha256",
        "missingness_known",
        "zero_event_semantics",
    }
    failures = len(required.difference(frame.columns)) + int(frame.empty)
    if failures == 0:
        start = pd.to_datetime(frame["requested_start_utc"], utc=True, errors="coerce")
        end = pd.to_datetime(frame["requested_end_utc"], utc=True, errors="coerce")
        completed = pd.to_datetime(frame["completed_at_utc"], utc=True, errors="coerce")
        status = frame["status"].astype(str)
        known = status.isin(_KNOWN_STATUSES)
        failures += int(frame["collection_id"].astype(str).duplicated().sum())
        failures += int((start.isna() | end.isna() | completed.isna()).sum())
        failures += int((end < start).fillna(True).sum())
        failures += int((completed < end).fillna(True).sum())
        failures += int(frame["missingness_known"].astype(bool).ne(known).sum())
        failures += int(frame["ticker"].astype(str).ne(GLOBAL_TICKER).sum())
        failures += int(frame["security_id"].astype(str).ne(GLOBAL_SECURITY_ID).sum())
    return _audit_report("global_event_coverage_authority", failures, len(frame))


def _audit_report(name: str, failures: int, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass" if failures == 0 else "fail",
                failures=failures,
                rows_checked=rows,
                detail="global identity, causal timing, explicit sources, and missingness verify",
            ),
        )
    )


def _artifact_record(path: Path, manifest: Mapping[str, object]) -> dict[str, object]:
    return {
        "path": path.name,
        "sha256": manifest["artifact_sha256"],
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": manifest["rows"],
    }


def _verify_published_artifact(
    record: object,
    manifest: Mapping[str, object],
    path: Path,
    rows: int,
) -> None:
    if not isinstance(record, dict) or (
        record.get("path") != path.name
        or record.get("sha256") != manifest.get("artifact_sha256")
        or record.get("manifest_sha256") != file_sha256(manifest_path_for(path))
        or _integer(record.get("rows"), "artifact rows") != rows
    ):
        raise DataReadinessError("global event child artifact does not verify")


def _rewrite_artifact_path(path: Path, output_directory: Path) -> None:
    child_manifest_path = manifest_path_for(path)
    child = _json_object(child_manifest_path)
    child["artifact_path"] = str((output_directory / path.name).resolve())
    _atomic_json(child_manifest_path, child)


def _guard_and_release(hard_budget_gib: float, headroom_gib: float, stage: str) -> None:
    gc.collect()
    release_process_memory()
    assert_memory_budget(
        hard_budget_gib=hard_budget_gib,
        headroom_gib=headroom_gib,
        stage=stage,
    )


def _validate_memory_policy(hard_budget_gib: float, headroom_gib: float) -> None:
    if hard_budget_gib > MAXIMUM_PROCESS_MEMORY_GIB:
        raise DataReadinessError("global event authority memory budget cannot exceed 4 GiB")
    if hard_budget_gib <= 0 or headroom_gib <= 0 or headroom_gib >= hard_budget_gib:
        raise DataReadinessError("global event authority memory policy is invalid")


def _timestamp_ns(value: object, name: str) -> int:
    try:
        timestamp = pd.Timestamp(value)
    except (TypeError, ValueError) as exc:
        raise DataReadinessError(f"global event {name} is invalid") from exc
    if pd.isna(timestamp) or timestamp.tzinfo is None:
        raise DataReadinessError(f"global event {name} must be timezone-aware")
    return int(timestamp.tz_convert("UTC").value)


def _timestamp_from_ns(value: int) -> pd.Timestamp:
    return pd.Timestamp(value, unit="ns", tz="UTC")


def _nullable_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DataReadinessError("global event sentiment is invalid")
    numeric = float(value)
    if not -1.0 <= numeric <= 1.0:
        raise DataReadinessError("global event sentiment is outside [-1, 1]")
    return numeric


def _nullable_nonnegative_float(value: object) -> float | None:
    if value is None or pd.isna(value):
        return None
    if isinstance(value, bool) or not isinstance(value, (int, float, np.integer, np.floating)):
        raise DataReadinessError("global event relevance is invalid")
    numeric = float(value)
    if numeric < 0:
        raise DataReadinessError("global event relevance is negative")
    return numeric


def _none_for_missing(record: Mapping[str, object]) -> dict[str, object]:
    return {str(key): None if value is None or pd.isna(value) else value for key, value in record.items()}


def _json_compatible(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_compatible(item) for item in value]
    if isinstance(value, (pd.Timestamp, datetime)):
        timestamp = pd.Timestamp(value)
        if timestamp.tzinfo is None:
            raise DataReadinessError("global event lineage contains a naive timestamp")
        return timestamp.tz_convert("UTC").isoformat()
    if isinstance(value, np.generic):
        return value.item()
    if value is None or pd.isna(value):
        return None
    return value


def _text(value: object, name: str) -> str:
    normalized = str(value).strip()
    if not normalized or normalized.lower() == "nan":
        raise DataReadinessError(f"global event {name} is empty")
    return normalized


def _integer(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)):
        raise DataReadinessError(f"global event {name} is not an integer")
    return int(value)


def _sha256_text(value: object, name: str) -> str:
    normalized = _text(value, name).lower()
    if len(normalized) != 64 or any(character not in "0123456789abcdef" for character in normalized):
        raise DataReadinessError(f"global event {name} is not a SHA-256")
    return normalized


def _required_sequence(record: Mapping[str, object], key: str) -> Sequence[object]:
    value = record.get(key)
    if not isinstance(value, list):
        raise DataReadinessError(f"global event authority has no {key}")
    return value


def _json_object(path: Path) -> dict[str, object]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"global event JSON artifact is unreadable: {path}") from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError(f"global event JSON artifact must be an object: {path}")
    return {str(key): value for key, value in loaded.items()}


def _json_sha256(value: object) -> str:
    return hashlib.sha256(_compact_json(value).encode("utf-8")).hexdigest()


def _compact_json(value: object) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
