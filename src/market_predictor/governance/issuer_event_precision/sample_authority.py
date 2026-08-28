"""Deterministic issuer-event precision sample publication and replay."""
from __future__ import annotations

import gc
import json
import os
import shutil
import sqlite3
import tempfile
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditReport
from market_predictor.canonical.store import (
    canonical_artifact_columns,
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events import classification as issuer_event_classification
from market_predictor.catalysts.issuer_events.attribution_history import (
    EventAttributionHistory,
    load_event_attribution_history,
)
from market_predictor.catalysts.issuer_events.classification import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
)
from market_predictor.catalysts.issuer_events.family_evidence import (
    IssuerFamilyEvidence,
    load_issuer_family_evidence,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.issuer_event_precision.artifact_integrity import (
    _artifact_record,
    _artifact_records,
    _assert_frame_equal,
    _atomic_json,
    _audit_report,
    _clean_text,
    _file_record,
    _json_object,
    _json_sha256,
    _json_text_object,
    _new_staging,
    _nonnegative_int,
    _normalize_title,
    _optional_timestamp,
    _read_csv,
    _remove_lock,
    _request,
    _required_hash,
    _required_path,
    _required_text,
    _rewrite_artifact_path,
    _sha256,
    _timestamp,
    _verify_canonical_record,
    _verify_file_record,
    _verify_inventory,
    _write_csv,
)
from market_predictor.governance.issuer_event_precision.contracts import (
    _FORBIDDEN_EVIDENCE_TERMS,
    SAMPLE_ARTIFACT_TYPE,
    SAMPLE_AUTHORITY_SCHEMA,
    SAMPLE_COLUMNS,
    SAMPLE_MANIFEST_SCHEMA,
    SOURCE_AUTHORIZATION_SHA256,
    IssuerEventPrecisionPolicy,
    IssuerEventPrecisionSample,
    _guard_memory,
    load_issuer_event_precision_policy,
)
from market_predictor.governance.issuer_event_precision.review_resolution import (
    _adjudication_template,
    _review_template,
)
from market_predictor.resources import (
    memory_audit,
    release_process_memory,
)


def publish_issuer_event_precision_sample(
    *,
    authority_directory: Path,
    policy_path: Path,
    output_directory: Path,
) -> IssuerEventPrecisionSample:
    """Publish a deterministic causal evidence sample and blank review ledgers."""

    if output_directory.exists():
        raise DataReadinessError(f"issuer-event precision sample is immutable: {output_directory}")
    policy = load_issuer_event_precision_policy(policy_path)
    issuer_authority = load_issuer_family_evidence(authority_directory)
    _guard_memory(policy, "issuer-event authority publication replay")
    authority_path = authority_directory / "_authority.json"
    authority_sha256 = file_sha256(authority_path)
    sample, population, rule_variant_population = _build_deterministic_sample(
        issuer_authority,
        policy=policy,
        policy_sha256=file_sha256(policy_path),
        authority_sha256=authority_sha256,
    )
    reviewer_one = _review_template(sample, reviewer_slot=1)
    reviewer_two = _review_template(sample, reviewer_slot=2)
    adjudication = _adjudication_template(sample)
    request = {
        "schema": SAMPLE_AUTHORITY_SCHEMA,
        "issuer_event_authority_directory": str(authority_directory.resolve()),
        "issuer_event_authority_path": str(authority_path.resolve()),
        "issuer_event_authority_sha256": authority_sha256,
        "policy_path": str(policy_path.resolve()),
        "policy_sha256": file_sha256(policy_path),
        "source_authorization_sha256": SOURCE_AUTHORIZATION_SHA256,
        "production_ready": False,
    }
    request_sha256 = _json_sha256(request)
    staging = _new_staging(output_directory)
    try:
        sample_path = staging / "sample.parquet"
        sample_manifest = write_canonical_artifact(
            sample,
            sample_path,
            artifact_type=SAMPLE_ARTIFACT_TYPE,
            audit=_sample_audit(sample),
            inputs={"request_sha256": request_sha256},
            production_ready=False,
        )
        _remove_lock(sample_path)
        _write_csv(staging / "reviewer_one_template.csv", reviewer_one)
        _write_csv(staging / "reviewer_two_template.csv", reviewer_two)
        _write_csv(staging / "adjudication_template.csv", adjudication)
        manifest = {
            "schema": SAMPLE_MANIFEST_SCHEMA,
            "state": "complete",
            "request": request,
            "request_sha256": request_sha256,
            "event_families": list(EVENT_FAMILIES),
            "population": population,
            "rule_variant_population": rule_variant_population,
            "inferential_sample_counts": {
                family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("inferential")).sum())
                for family in EVENT_FAMILIES
            },
            "diagnostic_sample_counts": {
                family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("paired_wrong_issuer_diagnostic")).sum())
                for family in EVENT_FAMILIES
            },
            "artifacts": {
                "sample": _artifact_record(sample_path, sample_manifest),
                "reviewer_one_template": _file_record(staging / "reviewer_one_template.csv", len(reviewer_one)),
                "reviewer_two_template": _file_record(staging / "reviewer_two_template.csv", len(reviewer_two)),
                "adjudication_template": _file_record(staging / "adjudication_template.csv", len(adjudication)),
            },
            "completed_at_utc": datetime.now(UTC).isoformat(),
            "memory": memory_audit(
                hard_budget_gib=policy.maximum_process_memory_gib,
                headroom_gib=policy.memory_guard_headroom_gib,
            ).to_record(),
            "production_ready": False,
            "training_eligible": False,
            "alerts_eligible": False,
        }
        _atomic_json(staging / "_manifest.json", manifest)
        root = {
            "schema": SAMPLE_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(staging / "_manifest.json"),
            "request_sha256": request_sha256,
            "production_ready": False,
        }
        _atomic_json(staging / "_authority.json", root)
        _rewrite_artifact_path(sample_path, output_directory / sample_path.name)
        _load_issuer_event_precision_sample(
            staging,
            _expected_artifact_directory=output_directory,
        )
        os.replace(staging, output_directory)
        return load_issuer_event_precision_sample(output_directory)
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def load_issuer_event_precision_sample(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    retain_source_authority: bool = False,
) -> IssuerEventPrecisionSample:
    """Strictly load and causally replay an immutable precision sample."""

    return _load_issuer_event_precision_sample(
        directory,
        expected_authority_sha256=expected_authority_sha256,
        retain_source_authority=retain_source_authority,
    )


def _load_issuer_event_precision_sample(
    directory: Path,
    *,
    expected_authority_sha256: str | None = None,
    retain_source_authority: bool = False,
    _expected_artifact_directory: Path | None = None,
) -> IssuerEventPrecisionSample:
    """Load final or staged sample evidence with an internal path binding."""

    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _json_object(manifest_path)
    authority = _json_object(authority_path)
    if expected_authority_sha256 is not None and (file_sha256(authority_path) != expected_authority_sha256):
        raise DataReadinessError("issuer-event precision sample identity changed")
    if (
        manifest.get("schema") != SAMPLE_MANIFEST_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("production_ready") is not False
        or authority.get("schema") != SAMPLE_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or authority.get("production_ready") is not False
    ):
        raise DataReadinessError("issuer-event precision sample root does not verify")
    _verify_inventory(
        directory,
        {
            "_authority.json",
            "_manifest.json",
            "sample.parquet",
            "sample.parquet.manifest.json",
            "reviewer_one_template.csv",
            "reviewer_two_template.csv",
            "adjudication_template.csv",
        },
    )
    request = _request(manifest, authority)
    policy_path = _required_path(request, "policy_path")
    if file_sha256(policy_path) != _required_hash(request, "policy_sha256"):
        raise DataReadinessError("issuer-event precision sample policy changed")
    policy = load_issuer_event_precision_policy(policy_path)
    if request.get("source_authorization_sha256") != SOURCE_AUTHORIZATION_SHA256:
        raise DataReadinessError("issuer-event source authorization policy changed")
    source_directory = _required_path(request, "issuer_event_authority_directory")
    source_authority_path = _required_path(request, "issuer_event_authority_path")
    source_sha256 = _required_hash(request, "issuer_event_authority_sha256")
    if source_authority_path != (source_directory / "_authority.json").resolve():
        raise DataReadinessError("issuer-event precision source authority path differs")
    source = load_issuer_family_evidence(
        source_directory,
        expected_authority_sha256=source_sha256,
    )
    _guard_memory(policy, "issuer-event authority replay")
    sample_path = directory / "sample.parquet"
    sample, child = load_canonical_artifact(sample_path, expected_type=SAMPLE_ARTIFACT_TYPE, allow_research=True)
    artifact_records = _artifact_records(manifest)
    _verify_canonical_record(
        sample_path,
        sample,
        child,
        artifact_records,
        "sample",
        request_sha256=str(manifest["request_sha256"]),
        expected_artifact_path=(
            _expected_artifact_directory / sample_path.name
            if _expected_artifact_directory is not None
            else None
        ),
    )
    _sample_audit(sample).raise_for_failure()
    expected, expected_population, expected_variant_population = _build_deterministic_sample(
        source,
        policy=policy,
        policy_sha256=_required_hash(request, "policy_sha256"),
        authority_sha256=source_sha256,
    )
    _assert_frame_equal(sample, expected, "precision sample replay")
    templates = (
        ("reviewer_one_template", _review_template(sample, reviewer_slot=1)),
        ("reviewer_two_template", _review_template(sample, reviewer_slot=2)),
        ("adjudication_template", _adjudication_template(sample)),
    )
    for name, expected_template in templates:
        path = directory / f"{name}.csv"
        _verify_file_record(path, artifact_records, name, len(expected_template))
        observed = _read_csv(path, tuple(expected_template.columns))
        _assert_frame_equal(observed, expected_template, f"{name} replay")
    expected_inferential_counts = {
        family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("inferential")).sum())
        for family in EVENT_FAMILIES
    }
    expected_diagnostic_counts = {
        family: int((sample["proposed_event_family"].eq(family) & sample["sample_role"].eq("paired_wrong_issuer_diagnostic")).sum())
        for family in EVENT_FAMILIES
    }
    if (
        manifest.get("event_families") != list(EVENT_FAMILIES)
        or manifest.get("population") != expected_population
        or manifest.get("rule_variant_population") != expected_variant_population
        or manifest.get("inferential_sample_counts") != expected_inferential_counts
        or manifest.get("diagnostic_sample_counts") != expected_diagnostic_counts
        or manifest.get("training_eligible") is not False
        or manifest.get("alerts_eligible") is not False
    ):
        raise DataReadinessError("issuer-event precision sample summary does not verify")
    return IssuerEventPrecisionSample(
        directory=directory.resolve(),
        sample=sample,
        manifest=manifest,
        authority=authority,
        source_authority=source if retain_source_authority else None,
    )

def _build_deterministic_sample(
    authority: IssuerFamilyEvidence,
    *,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
    authority_sha256: str,
) -> tuple[
    pd.DataFrame,
    dict[str, dict[str, int]],
    dict[str, dict[str, int]],
]:
    events = authority.events
    observed_families = set(events.get("event_family", pd.Series(dtype=str)).astype(str))
    unknown = sorted(observed_families.difference(EVENT_FAMILIES))
    if unknown:
        raise DataReadinessError("issuer-event authority contains unknown families: " + ", ".join(unknown))
    with tempfile.TemporaryDirectory(prefix="market-predictor-precision-") as temporary:
        database_path = Path(temporary) / "candidate_index.sqlite3"
        connection = sqlite3.connect(database_path)
        try:
            _create_candidate_index(connection)
            _index_eligible_authority_events(connection, events, policy=policy)
            attribution, collection_manifest_path = _strict_attribution_context(authority)
            _stream_candidate_population(
                connection,
                collection_manifest_path=collection_manifest_path,
                policy=policy,
                policy_sha256=policy_sha256,
            )
            population = _population_from_candidate_index(connection)
            variant_population = _variant_population_from_candidate_index(connection)
            selected = _select_uniform_cluster_rows(
                connection,
                policy=policy,
                policy_sha256=policy_sha256,
                authority_sha256=authority_sha256,
            )
            sample = _selected_causal_evidence(
                attribution=attribution,
                collection_manifest_path=collection_manifest_path,
                selected=selected,
                policy=policy,
            )
        finally:
            connection.close()
    if sample.empty:
        sample = pd.DataFrame(columns=SAMPLE_COLUMNS)
    else:
        sample["schema_version"] = SAMPLE_MANIFEST_SCHEMA
        sample["_role_order"] = sample["sample_role"].map({"inferential": 0, "paired_wrong_issuer_diagnostic": 1})
        sample = (
            sample.sort_values(
                [
                    "proposed_event_family",
                    "cluster_selection_sha256",
                    "row_selection_sha256",
                    "_role_order",
                    "family_event_id",
                ],
                kind="stable",
            )
            .loc[:, SAMPLE_COLUMNS]
            .reset_index(drop=True)
        )
    _sample_audit(sample).raise_for_failure()
    _guard_memory(policy, "precision sample construction")
    return sample, population, variant_population


def _source_authorized_events(events: pd.DataFrame) -> pd.DataFrame:
    required = {
        "event_family",
        "source_family",
        "research_eligible",
        "family_event_id",
        "source_event_id",
        "relation_id",
        "security_id",
    }
    missing = sorted(required.difference(events.columns))
    if missing:
        raise DataReadinessError("issuer-event precision source is missing columns: " + ", ".join(missing))
    research_eligible = events["research_eligible"].astype(bool)
    source_authorized = pd.Series(False, index=events.index)
    for family in EVENT_FAMILIES:
        source_authorized |= events["event_family"].eq(family) & events["source_family"].astype(str).isin(
            ALLOWED_SOURCE_FAMILIES_BY_FAMILY[family]
        )
    eligible = events.loc[research_eligible & source_authorized]
    if bool(eligible["family_event_id"].astype(str).duplicated().any()):
        raise DataReadinessError("eligible issuer-event identities are duplicated")
    return eligible


def _create_candidate_index(connection: sqlite3.Connection) -> None:
    connection.executescript(
        """
        PRAGMA journal_mode = OFF;
        PRAGMA synchronous = OFF;
        PRAGMA temp_store = FILE;
        CREATE TABLE eligible_events (
            family_event_id TEXT PRIMARY KEY,
            source_event_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX eligible_source_event_idx
            ON eligible_events(source_event_id);
        CREATE TABLE candidates (
            family_event_id TEXT PRIMARY KEY,
            source_event_id TEXT NOT NULL,
            relation_id TEXT NOT NULL,
            security_id TEXT NOT NULL,
            event_family TEXT NOT NULL,
            source_family TEXT NOT NULL,
            chunk_id TEXT NOT NULL,
            inference_cluster_id TEXT NOT NULL,
            cluster_selection_sha256 TEXT NOT NULL,
            row_selection_sha256 TEXT NOT NULL,
            normalized_title_sha256 TEXT NOT NULL,
            calendar_quarter TEXT NOT NULL,
            rule_variant TEXT NOT NULL,
            stratum_id TEXT NOT NULL,
            payload_json TEXT NOT NULL
        );
        CREATE INDEX candidate_family_cluster_idx
            ON candidates(event_family, inference_cluster_id);
        CREATE INDEX candidate_cluster_row_idx
            ON candidates(inference_cluster_id, row_selection_sha256);
        CREATE INDEX candidate_source_event_idx
            ON candidates(source_event_id);
        """
    )


def _index_eligible_authority_events(
    connection: sqlite3.Connection,
    events: pd.DataFrame,
    *,
    policy: IssuerEventPrecisionPolicy,
) -> None:
    eligible = _source_authorized_events(events)
    columns = tuple(str(column) for column in eligible.columns)
    cursor = connection.cursor()
    try:
        for index, values in enumerate(eligible.itertuples(index=False, name=None), start=1):
            row = dict(zip(columns, values, strict=True))
            try:
                cursor.execute(
                    "INSERT INTO eligible_events VALUES (?, ?, ?)",
                    (
                        str(row["family_event_id"]),
                        str(row["source_event_id"]),
                        json.dumps(
                            row,
                            sort_keys=True,
                            separators=(",", ":"),
                            default=str,
                        ),
                    ),
                )
            except sqlite3.IntegrityError as exc:
                raise DataReadinessError("eligible issuer-event identities are duplicated") from exc
            if index % 25_000 == 0:
                _guard_memory(policy, f"authority candidate index row {index}")
    finally:
        cursor.close()
    connection.commit()


def _strict_attribution_context(
    authority: IssuerFamilyEvidence,
) -> tuple[EventAttributionHistory, Path]:
    authority_request = authority.manifest.get("request")
    if not isinstance(authority_request, dict):
        raise DataReadinessError("issuer-event authority request is malformed")
    attribution_manifest_path = _required_path(authority_request, "attribution_manifest_path")
    if file_sha256(attribution_manifest_path) != _required_hash(authority_request, "attribution_manifest_sha256"):
        raise DataReadinessError("issuer-event attribution manifest changed")
    attribution = load_event_attribution_history(attribution_manifest_path.parent)
    collection_manifest_path = _required_path(attribution.request, "collection_manifest_path")
    if collection_manifest_path != _required_path(authority_request, "collection_manifest_path"):
        raise DataReadinessError("issuer-event collection lineage differs")
    return attribution, collection_manifest_path


def _stream_candidate_population(
    connection: sqlite3.Connection,
    *,
    collection_manifest_path: Path,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
) -> None:
    collection = _json_object(collection_manifest_path)
    records = collection.get("artifacts")
    if not isinstance(records, list):
        raise DataReadinessError("precision source event inventory is malformed")
    desired_columns = {
        "event_id",
        "source_family",
        "title",
        "published_at_utc",
    }
    for index, raw in enumerate(records, start=1):
        if not isinstance(raw, dict):
            raise DataReadinessError("precision source event record is malformed")
        path = _required_path(raw, "path")
        declared = canonical_artifact_columns(path)
        columns = tuple(column for column in declared if column in desired_columns)
        if set(columns) != desired_columns:
            raise DataReadinessError("precision source event artifact lacks cluster fields")
        events, manifest = load_canonical_artifact(
            path,
            expected_type="events",
            allow_research=True,
            columns=columns,
        )
        if manifest.get("artifact_sha256") != _required_hash(raw, "sha256") or len(events) != _nonnegative_int(raw, "rows"):
            raise DataReadinessError("precision source event artifact differs")
        chunk_id = _required_text(raw, "chunk_id")
        columns_in_frame = tuple(str(column) for column in events.columns)
        insert_cursor = connection.cursor()
        for values in events.itertuples(index=False, name=None):
            source = dict(zip(columns_in_frame, values, strict=True))
            source_event_id = str(source["event_id"])
            matches = connection.execute(
                "SELECT payload_json FROM eligible_events WHERE source_event_id = ?",
                (source_event_id,),
            )
            for (payload_json,) in matches:
                payload = _json_text_object(str(payload_json), "eligible event payload")
                try:
                    insert_cursor.execute(
                        "INSERT INTO candidates VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                        _candidate_index_row(
                            payload,
                            source,
                            chunk_id=chunk_id,
                            policy_sha256=policy_sha256,
                        ),
                    )
                except sqlite3.IntegrityError as exc:
                    raise DataReadinessError("precision candidate identities are duplicated") from exc
        insert_cursor.close()
        connection.commit()
        del events
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"candidate source chunk {index}")
    eligible_count = int(connection.execute("SELECT COUNT(*) FROM eligible_events").fetchone()[0])
    candidate_count = int(connection.execute("SELECT COUNT(*) FROM candidates").fetchone()[0])
    if candidate_count != eligible_count:
        raise DataReadinessError("precision candidate population is incomplete")


def _candidate_index_row(
    family: Mapping[str, object],
    source: Mapping[str, object],
    *,
    chunk_id: str,
    policy_sha256: str,
) -> tuple[str, ...]:
    if str(family["source_event_id"]) != str(source["event_id"]) or str(family["source_family"]) != str(source["source_family"]):
        raise DataReadinessError("precision source event identity differs")
    title = _clean_text(source.get("title"))
    if not title:
        raise DataReadinessError("precision source event title is empty")
    published = _timestamp(source["published_at_utc"], "source publication time")
    if published != _timestamp(family["published_at_utc"], "family publication time"):
        raise DataReadinessError("precision source publication timing differs")
    family_name = str(family["event_family"])
    normalized_title_sha256 = _sha256(_normalize_title(title))
    publication_day = published.strftime("%Y-%m-%d")
    cluster_id = _sha256(f"{family_name}|{normalized_title_sha256}|{publication_day}")
    cluster_hash = _sha256(f"{policy_sha256}|cluster|{cluster_id}")
    family_event_id = str(family["family_event_id"])
    row_hash = _sha256(f"{policy_sha256}|row|{family_event_id}")
    feature_time = _timestamp(family["feature_available_at_utc"], "feature time")
    quarter = f"{feature_time.year}Q{feature_time.quarter}"
    variant = issuer_event_classification.issuer_event_rule_variant(family)
    stratum = "|".join(
        (
            str(family["source_family"]),
            str(family["classification_rule_id"]),
            variant,
            quarter,
        )
    )
    return (
        family_event_id,
        str(family["source_event_id"]),
        str(family["relation_id"]),
        str(family["security_id"]),
        family_name,
        str(family["source_family"]),
        chunk_id,
        cluster_id,
        cluster_hash,
        row_hash,
        normalized_title_sha256,
        quarter,
        variant,
        stratum,
        json.dumps(family, sort_keys=True, separators=(",", ":"), default=str),
    )


def _population_from_candidate_index(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        row = connection.execute(
            "SELECT COUNT(*), COUNT(DISTINCT inference_cluster_id), COUNT(DISTINCT security_id) FROM candidates WHERE event_family = ?",
            (family,),
        ).fetchone()
        output[family] = {
            "eligible_events": int(row[0]),
            "clusters": int(row[1]),
            "issuers": int(row[2]),
        }
    return output


def _variant_population_from_candidate_index(
    connection: sqlite3.Connection,
) -> dict[str, dict[str, int]]:
    output: dict[str, dict[str, int]] = {family: {} for family in EVENT_FAMILIES}
    rows = connection.execute(
        """
        WITH representatives AS (
            SELECT event_family, inference_cluster_id, rule_variant,
                   ROW_NUMBER() OVER (
                       PARTITION BY inference_cluster_id
                       ORDER BY row_selection_sha256, family_event_id
                   ) AS row_number
            FROM candidates
        )
        SELECT event_family, rule_variant, COUNT(*)
        FROM representatives
        WHERE row_number = 1
        GROUP BY event_family, rule_variant
        ORDER BY event_family, rule_variant
        """
    ).fetchall()
    for family, variant, count in rows:
        output[str(family)][str(variant)] = int(count)
    return output


def _select_uniform_cluster_rows(
    connection: sqlite3.Connection,
    *,
    policy: IssuerEventPrecisionPolicy,
    policy_sha256: str,
    authority_sha256: str,
) -> list[dict[str, object]]:
    selected: list[dict[str, object]] = []
    for family in EVENT_FAMILIES:
        clusters = connection.execute(
            "SELECT inference_cluster_id, cluster_selection_sha256 "
            "FROM candidates WHERE event_family = ? "
            "GROUP BY inference_cluster_id, cluster_selection_sha256 "
            "ORDER BY cluster_selection_sha256, inference_cluster_id LIMIT ?",
            (family, policy.families[family].sample_clusters),
        ).fetchall()
        for cluster_id, cluster_hash in clusters:
            representative_row = connection.execute(
                "SELECT payload_json, chunk_id, row_selection_sha256, "
                "normalized_title_sha256, calendar_quarter, rule_variant, stratum_id "
                "FROM candidates WHERE inference_cluster_id = ? "
                "ORDER BY row_selection_sha256, family_event_id LIMIT 1",
                (cluster_id,),
            ).fetchone()
            if representative_row is None:
                raise DataReadinessError("selected precision cluster is empty")
            representative = _selected_candidate_record(
                representative_row,
                cluster_id=str(cluster_id),
                cluster_hash=str(cluster_hash),
                sample_role="inferential",
                paired_sample_id="",
                role_index=0,
                policy_sha256=policy_sha256,
                authority_sha256=authority_sha256,
            )
            _add_candidate_population_flags(connection, representative)
            selected.append(representative)
            representative_security = str(representative["security_id"])
            diagnostic_rows = connection.execute(
                "SELECT payload_json, chunk_id, row_selection_sha256, "
                "normalized_title_sha256, calendar_quarter, rule_variant, stratum_id "
                "FROM candidates WHERE inference_cluster_id = ? AND security_id != ? "
                "ORDER BY row_selection_sha256, family_event_id LIMIT ?",
                (
                    cluster_id,
                    representative_security,
                    policy.paired_wrong_issuer_diagnostics_per_cluster,
                ),
            )
            for diagnostic_count, row in enumerate(diagnostic_rows, start=1):
                diagnostic = _selected_candidate_record(
                    row,
                    cluster_id=str(cluster_id),
                    cluster_hash=str(cluster_hash),
                    sample_role="paired_wrong_issuer_diagnostic",
                    paired_sample_id=str(representative["sample_id"]),
                    role_index=diagnostic_count,
                    policy_sha256=policy_sha256,
                    authority_sha256=authority_sha256,
                )
                _add_candidate_population_flags(connection, diagnostic)
                selected.append(diagnostic)
    return selected


def _selected_candidate_record(
    row: Sequence[object],
    *,
    cluster_id: str,
    cluster_hash: str,
    sample_role: str,
    paired_sample_id: str,
    role_index: int,
    policy_sha256: str,
    authority_sha256: str,
) -> dict[str, object]:
    payload = _json_text_object(str(row[0]), "candidate payload")
    row_hash = str(row[2])
    sample_hash = _sha256(f"{policy_sha256}|{authority_sha256}|sample|{sample_role}|{cluster_id}|{row_hash}|{role_index}")
    payload.update(
        {
            "sample_id": f"sample:{sample_hash[:32]}",
            "sample_role": sample_role,
            "inference_cluster_id": cluster_id,
            "paired_inferential_sample_id": paired_sample_id,
            "chunk_id": str(row[1]),
            "cluster_selection_sha256": cluster_hash,
            "row_selection_sha256": row_hash,
            "normalized_title_sha256": str(row[3]),
            "calendar_quarter": str(row[4]),
            "rule_variant": str(row[5]),
            "stratum_id": str(row[6]),
        }
    )
    return payload


def _add_candidate_population_flags(
    connection: sqlite3.Connection,
    candidate: dict[str, object],
) -> None:
    cluster_targets = connection.execute(
        "SELECT COUNT(DISTINCT security_id) FROM candidates WHERE inference_cluster_id = ?",
        (str(candidate["inference_cluster_id"]),),
    ).fetchone()
    source_families = connection.execute(
        "SELECT COUNT(DISTINCT event_family) FROM candidates WHERE source_event_id = ?",
        (str(candidate["source_event_id"]),),
    ).fetchone()
    candidate["multi_target_title"] = int(cluster_targets[0]) > 1
    candidate["multi_label_event"] = int(source_families[0]) > 1


def _selected_causal_evidence(
    *,
    attribution: EventAttributionHistory,
    collection_manifest_path: Path,
    selected: Sequence[Mapping[str, object]],
    policy: IssuerEventPrecisionPolicy,
) -> pd.DataFrame:
    if not selected:
        return pd.DataFrame(columns=SAMPLE_COLUMNS)
    source_by_id = _load_selected_source_events(
        collection_manifest_path,
        selected,
        policy=policy,
    )
    relation_by_id = _load_selected_relations(attribution, selected, policy=policy)
    identity_groups = _load_selected_identities(attribution, selected, policy=policy)
    rows: list[dict[str, object]] = []
    for family_row in selected:
        source_event_id = str(family_row["source_event_id"])
        relation_id = str(family_row["relation_id"])
        source = source_by_id.get(source_event_id)
        relation = relation_by_id.get(relation_id)
        if source is None or relation is None:
            raise DataReadinessError("issuer-event evidence lineage is incomplete")
        _verify_event_relation_lineage(family_row, source, relation)
        feature_time = _timestamp(family_row["feature_available_at_utc"], "feature time")
        identity = _causal_identity(identity_groups.get(str(family_row["security_id"])), feature_time)
        source_family = _clean_text(source.get("source_family"))
        source_name = _clean_text(source.get("source")) or source_family
        title = _clean_text(source.get("title"))
        published_day = _timestamp(family_row["published_at_utc"], "publication time").strftime("%Y-%m-%d")
        expected_cluster = _sha256(f"{family_row['event_family']}|{_sha256(_normalize_title(title))}|{published_day}")
        if expected_cluster != str(family_row["inference_cluster_id"]):
            raise DataReadinessError("selected precision cluster identity differs")
        rows.append(
            {
                "sample_id": str(family_row["sample_id"]),
                "sample_role": str(family_row["sample_role"]),
                "inference_cluster_id": str(family_row["inference_cluster_id"]),
                "paired_inferential_sample_id": str(family_row["paired_inferential_sample_id"]),
                "family_event_id": str(family_row["family_event_id"]),
                "source_event_id": source_event_id,
                "relation_id": relation_id,
                "source_security_id": str(family_row["source_security_id"]),
                "source_ticker": str(family_row["source_ticker"]),
                "security_id": str(family_row["security_id"]),
                "ticker": str(family_row["ticker"]),
                "source_family": str(family_row["source_family"]),
                "source": source_name,
                "proposed_event_family": str(family_row["event_family"]),
                "classification_rule_id": str(family_row["classification_rule_id"]),
                "classification_basis": str(family_row["classification_basis"]),
                "matched_text": str(family_row["matched_text"]),
                "title": title,
                "summary": _clean_text(source.get("summary")),
                "text": _clean_text(source.get("text")),
                "published_at_utc": _timestamp(family_row["published_at_utc"], "publication time"),
                "event_available_at_utc": _timestamp(family_row["event_available_at_utc"], "event availability"),
                "relation_available_at_utc": _timestamp(family_row["relation_available_at_utc"], "relation availability"),
                "feature_available_at_utc": feature_time,
                "availability_policy": str(family_row["availability_policy"]),
                "relation_channel": str(family_row["relation_channel"]),
                "issuer_company": identity["company"],
                "identity_effective_from_utc": identity["effective_from_utc"],
                "identity_effective_to_utc": identity["effective_to_utc"],
                "identity_available_at_utc": identity["available_at_utc"],
                "identity_status": identity["status"],
                "calendar_quarter": str(family_row["calendar_quarter"]),
                "rule_variant": str(family_row["rule_variant"]),
                "normalized_title_sha256": str(family_row["normalized_title_sha256"]),
                "multi_target_title": bool(family_row["multi_target_title"]),
                "multi_label_event": bool(family_row["multi_label_event"]),
                "stratum_id": str(family_row["stratum_id"]),
                "cluster_selection_sha256": str(family_row["cluster_selection_sha256"]),
                "row_selection_sha256": str(family_row["row_selection_sha256"]),
            }
        )
    frame = pd.DataFrame.from_records(rows)
    if bool(frame["title"].str.strip().eq("").any()) or bool(frame["source"].str.strip().eq("").any()):
        raise DataReadinessError("precision evidence requires causal title and source")
    if any(term in column.lower() for column in frame.columns for term in _FORBIDDEN_EVIDENCE_TERMS):
        raise DataReadinessError("precision evidence contains future outcome fields")
    _guard_memory(policy, "causal precision evidence")
    return frame


def _load_selected_source_events(
    collection_manifest_path: Path,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, dict[str, object]]:
    collection = _json_object(collection_manifest_path)
    records = collection.get("artifacts")
    if not isinstance(records, list):
        raise DataReadinessError("precision source event inventory is malformed")
    by_chunk: dict[str, set[str]] = {}
    for row in selected:
        by_chunk.setdefault(str(row["chunk_id"]), set()).add(str(row["source_event_id"]))
    output: dict[str, dict[str, object]] = {}
    desired_columns = (
        "event_id",
        "security_id",
        "ticker",
        "source_family",
        "source",
        "title",
        "summary",
        "text",
        "published_at_utc",
        "feature_available_at_utc",
    )
    records_by_chunk = {_required_text(raw, "chunk_id"): raw for raw in records if isinstance(raw, dict)}
    if set(by_chunk).difference(records_by_chunk):
        raise DataReadinessError("selected precision source chunks are missing")
    for index, chunk_id in enumerate(sorted(by_chunk), start=1):
        raw = records_by_chunk[chunk_id]
        if not isinstance(raw, dict):
            raise DataReadinessError("precision source event record is malformed")
        path = _required_path(raw, "path")
        events = _verified_filtered_artifact(
            path,
            expected_type="events",
            expected_sha256=_required_hash(raw, "sha256"),
            columns=desired_columns,
            filter_column="event_id",
            values=by_chunk[chunk_id],
        )
        for row in events.to_dict(orient="records"):
            event_id = str(row["event_id"])
            if event_id in output:
                raise DataReadinessError("precision source event identities are duplicated")
            output[event_id] = {str(key): value for key, value in row.items()}
        del events
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"source evidence chunk {index}")
    required_event_ids = {str(row["source_event_id"]) for row in selected}
    if set(output) != required_event_ids:
        raise DataReadinessError("precision source events are incomplete")
    return output


def _load_selected_relations(
    attribution: EventAttributionHistory,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, dict[str, object]]:
    by_chunk: dict[str, set[str]] = {}
    for row in selected:
        by_chunk.setdefault(str(row["chunk_id"]), set()).add(str(row["relation_id"]))
    records_by_chunk = {_required_text(record, "chunk_id"): record for record in attribution.artifact_records}
    if set(by_chunk).difference(records_by_chunk):
        raise DataReadinessError("selected precision relation chunks are missing")
    output: dict[str, dict[str, object]] = {}
    relation_columns = (
        "relation_id",
        "event_id",
        "source_security_id",
        "source_ticker",
        "target_security_id",
        "target_ticker",
        "relation_channel",
        "feature_available_at_utc",
    )
    for index, chunk_id in enumerate(sorted(by_chunk), start=1):
        record = records_by_chunk[chunk_id]
        path = _required_path(record, "path")
        relations = _verified_filtered_artifact(
            path,
            expected_type="event_security_relations",
            expected_sha256=_required_hash(record, "sha256"),
            columns=relation_columns,
            filter_column="relation_id",
            values=by_chunk[chunk_id],
        )
        for row in relations.to_dict(orient="records"):
            relation_id = str(row["relation_id"])
            if relation_id in output:
                raise DataReadinessError("precision relation identities are duplicated")
            output[relation_id] = {str(key): value for key, value in row.items()}
        del relations
        if index % 32 == 0:
            gc.collect()
            release_process_memory()
        _guard_memory(policy, f"relation evidence chunk {index}")
    required_relation_ids = {str(row["relation_id"]) for row in selected}
    if set(output) != required_relation_ids:
        raise DataReadinessError("precision event relations are incomplete")
    return output


def _load_selected_identities(
    attribution: EventAttributionHistory,
    selected: Sequence[Mapping[str, object]],
    *,
    policy: IssuerEventPrecisionPolicy,
) -> dict[str, list[dict[str, object]]]:
    identity_path = _required_path(attribution.request, "security_identities_path")
    security_ids = {str(row["security_id"]) for row in selected}
    identities = _verified_filtered_artifact(
        identity_path,
        expected_type="security_business_label_coverage",
        expected_sha256=_required_hash(attribution.request, "security_identities_sha256"),
        columns=(
            "security_id",
            "ticker",
            "company",
            "effective_from_utc",
            "effective_to_utc",
            "available_at_utc",
        ),
        filter_column="security_id",
        values=security_ids,
    )
    output = _identity_records_by_security(identities)
    del identities
    gc.collect()
    release_process_memory()
    _guard_memory(policy, "selected identity evidence")
    return output


def _verified_filtered_artifact(
    path: Path,
    *,
    expected_type: str,
    expected_sha256: str,
    columns: Sequence[str],
    filter_column: str,
    values: set[str],
) -> pd.DataFrame:
    manifest = _json_object(manifest_path_for(path))
    declared = canonical_artifact_columns(path)
    if (
        manifest.get("artifact_type") != expected_type
        or manifest.get("artifact_sha256") != expected_sha256
        or file_sha256(path) != expected_sha256
        or not set(columns).issubset(declared)
        or filter_column not in declared
    ):
        raise DataReadinessError(f"selected canonical artifact does not verify: {path}")
    if not values:
        return pd.DataFrame(columns=columns)
    frame = pd.read_parquet(
        path,
        columns=list(columns),
        filters=[(filter_column, "in", sorted(values))],
    )
    return frame.loc[frame[filter_column].astype(str).isin(values)].reset_index(drop=True)


def _verify_event_relation_lineage(
    family: Mapping[str, object],
    source: Mapping[str, object],
    relation: Mapping[str, object],
) -> None:
    comparisons = (
        (family["source_event_id"], source["event_id"]),
        (family["source_event_id"], relation["event_id"]),
        (family["source_security_id"], relation["source_security_id"]),
        (family["source_ticker"], relation["source_ticker"]),
        (family["security_id"], relation["target_security_id"]),
        (family["ticker"], relation["target_ticker"]),
        (family["relation_channel"], relation["relation_channel"]),
        (family["source_family"], source["source_family"]),
    )
    if any(str(left) != str(right) for left, right in comparisons):
        raise DataReadinessError("precision source event or relation identity differs")
    timestamp_pairs = (
        (family["published_at_utc"], source["published_at_utc"]),
        (family["event_available_at_utc"], source["feature_available_at_utc"]),
        (family["relation_available_at_utc"], relation["feature_available_at_utc"]),
    )
    if any(_timestamp(left, "lineage") != _timestamp(right, "lineage") for left, right in timestamp_pairs):
        raise DataReadinessError("precision source event or relation timing differs")
    if str(family["relation_channel"]) != "direct_issuer":
        raise DataReadinessError("precision sample cannot admit indirect issuer relations")


def _causal_identity(
    identities: Sequence[Mapping[str, object]] | None,
    feature_time: pd.Timestamp,
) -> dict[str, object]:
    empty: dict[str, object] = {
        "company": "",
        "effective_from_utc": pd.NaT,
        "effective_to_utc": pd.NaT,
        "available_at_utc": pd.NaT,
        "status": "unresolved",
    }
    if not identities:
        return empty
    eligible = [
        row
        for row in identities
        if isinstance(row["effective_from_utc"], pd.Timestamp)
        and isinstance(row["available_at_utc"], pd.Timestamp)
        and row["effective_from_utc"] <= feature_time
        and row["available_at_utc"] <= feature_time
        and (not isinstance(row["effective_to_utc"], pd.Timestamp) or row["effective_to_utc"] > feature_time)
    ]
    if len(eligible) != 1:
        empty["status"] = "ambiguous" if len(eligible) > 1 else "unresolved"
        return empty
    row = eligible[0]
    company = _clean_text(row["company"])
    if not company:
        return empty
    return {
        "company": company,
        "effective_from_utc": row["effective_from_utc"],
        "effective_to_utc": row["effective_to_utc"],
        "available_at_utc": row["available_at_utc"],
        "status": "resolved",
    }


def _identity_records_by_security(
    identities: pd.DataFrame,
) -> dict[str, list[dict[str, object]]]:
    output: dict[str, list[dict[str, object]]] = {}
    for raw in identities.to_dict(orient="records"):
        row = {str(key): value for key, value in raw.items()}
        row["effective_from_utc"] = _optional_timestamp(row.get("effective_from_utc"))
        row["effective_to_utc"] = _optional_timestamp(row.get("effective_to_utc"))
        row["available_at_utc"] = _optional_timestamp(row.get("available_at_utc"))
        output.setdefault(str(row["security_id"]), []).append(row)
    return output

def _sample_audit(frame: pd.DataFrame) -> CanonicalAuditReport:
    failures = 0
    if list(frame.columns) != list(SAMPLE_COLUMNS):
        failures += 1
    if not frame.empty:
        failures += int(frame["sample_id"].astype(str).duplicated().sum())
        failures += int(frame["family_event_id"].astype(str).duplicated().sum())
        failures += int((~frame["proposed_event_family"].astype(str).isin(EVENT_FAMILIES)).sum())
        failures += int(frame["title"].astype(str).str.strip().eq("").sum())
        failures += int(frame["source"].astype(str).str.strip().eq("").sum())
        failures += int((~frame["sample_role"].isin(("inferential", "paired_wrong_issuer_diagnostic"))).sum())
        inferential = frame.loc[frame["sample_role"].eq("inferential")]
        diagnostics = frame.loc[frame["sample_role"].eq("paired_wrong_issuer_diagnostic")]
        failures += int(inferential["inference_cluster_id"].astype(str).duplicated().sum())
        failures += int(inferential["paired_inferential_sample_id"].astype(str).ne("").sum())
        inferential_ids = set(inferential["sample_id"].astype(str))
        failures += int((~diagnostics["paired_inferential_sample_id"].astype(str).isin(inferential_ids)).sum())
        parent_clusters = inferential.set_index("sample_id")["inference_cluster_id"].astype(str).to_dict()
        failures += sum(
            parent_clusters.get(str(row["paired_inferential_sample_id"])) != str(row["inference_cluster_id"])
            for row in diagnostics.to_dict(orient="records")
        )
        availability = pd.to_datetime(frame["feature_available_at_utc"], utc=True, errors="coerce")
        identity_available = pd.to_datetime(frame["identity_available_at_utc"], utc=True, errors="coerce")
        resolved = frame["identity_status"].eq("resolved")
        failures += int((resolved & identity_available.gt(availability)).sum())
    return _audit_report("issuer_event_precision_sample", len(frame), failures)

def _population_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    raw = manifest.get("population")
    if not isinstance(raw, dict) or set(raw) != set(EVENT_FAMILIES):
        raise DataReadinessError("precision sample population summary is malformed")
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        value = raw.get(family)
        if not isinstance(value, dict):
            raise DataReadinessError("precision sample family population is malformed")
        if set(value) != {"eligible_events", "clusters", "issuers"}:
            raise DataReadinessError("precision sample family population fields differ")
        output[family] = {
            "eligible_events": _nonnegative_int(value, "eligible_events"),
            "clusters": _nonnegative_int(value, "clusters"),
            "issuers": _nonnegative_int(value, "issuers"),
        }
    return output


def _rule_variant_population_from_manifest(
    manifest: Mapping[str, object],
) -> dict[str, dict[str, int]]:
    raw = manifest.get("rule_variant_population")
    if not isinstance(raw, dict) or set(raw) != set(EVENT_FAMILIES):
        raise DataReadinessError("precision rule-variant population is malformed")
    output: dict[str, dict[str, int]] = {}
    for family in EVENT_FAMILIES:
        values = raw.get(family)
        if not isinstance(values, dict):
            raise DataReadinessError("precision family rule-variant population is malformed")
        output[family] = {str(variant): _nonnegative_int(values, str(variant)) for variant in sorted(values)}
    return output
