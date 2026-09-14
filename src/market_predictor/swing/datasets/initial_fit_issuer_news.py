"""One-time, pinned initial-fit projection of saved strict issuer evidence."""
from __future__ import annotations

import argparse
import json
import tomllib
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import TYPE_CHECKING, Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.attribution import ATTRIBUTION_POLICY_SHA256, ATTRIBUTION_POLICY_VERSION
from market_predictor.catalysts.issuer_events.attribution_history import ATTRIBUTION_SCOPE_POLICY, EventAttributionHistory
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.heavy_jobs import heavy_job_lease

if TYPE_CHECKING:
    from market_predictor.swing.datasets.issuer_event_family_cohort import SwingIssuerFamilyCohort

SCHEMA = "swing.initial_fit_issuer_derivation.v1"
FINBERT_REVISION = "4556d13015211d73dccd3fdd39d39232506f3e43"
LAST_INITIAL_FIT_CUTOFF = pd.Timestamp("2024-05-28T22:00:00Z")


@dataclass(frozen=True)
class SavedIssuerAuthority:
    attribution_dir: Path
    attribution_manifest_sha256: str
    sentiment_dir: Path
    sentiment_manifest_sha256: str
    business_labels_manifest_sha256: str
    security_identities_manifest_sha256: str


@dataclass(frozen=True)
class DerivedIssuerInputs:
    directory: Path
    manifest_sha256: str
    manifest: Mapping[str, Any]


def _json(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"issuer derivation requires a JSON object: {path}")
    return value


def _pin(path: Path, expected: str) -> None:
    if len(expected) != 64 or path.is_symlink() or file_sha256(path) != expected:
        raise DataReadinessError(f"issuer derivation source pin differs: {path}")


def _inside(root: Path, raw: object) -> Path:
    path = Path(str(raw))
    if not path.is_absolute():
        path = root / path
    if path.is_symlink() or root.resolve() not in path.resolve().parents:
        raise DataReadinessError("issuer derivation child escapes its authority")
    return path.resolve()


def _declared_path(raw: object, repository_root: Path | None) -> Path:
    """Resolve saved producer paths against the explicitly supplied repository only."""
    path = Path(str(raw))
    if repository_root is not None:
        return _inside(repository_root, path)
    if not path.is_absolute():
        raise DataReadinessError("repository-relative saved path requires an explicit repository_root")
    if path.is_symlink():
        raise DataReadinessError("saved issuer path is a symlink")
    return path.resolve()


def _request(root: Path, manifest: Mapping[str, Any]) -> dict[str, Any]:
    value = _json(root / "_request.json")
    declared = value.pop("request_sha256", None)
    if declared != manifest.get("request_sha256") or declared != json_sha256(value):
        raise DataReadinessError("saved issuer request hash differs")
    return value


def _records(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    raw = manifest.get("artifacts")
    if not isinstance(raw, list) or any(not isinstance(row, dict) for row in raw):
        raise DataReadinessError("saved issuer artifact inventory is malformed")
    result = {str(row.get("chunk_id", "")): row for row in raw}
    if "" in result or len(result) != len(raw):
        raise DataReadinessError("saved issuer chunks are missing or duplicated")
    return result


def _saved_headers(source: SavedIssuerAuthority, *, repository_root: Path | None = None) -> tuple[dict[str, Any], ...]:
    aroot, sroot = source.attribution_dir.resolve(), source.sentiment_dir.resolve()
    _pin(aroot / "_manifest.json", source.attribution_manifest_sha256)
    _pin(sroot / "_manifest.json", source.sentiment_manifest_sha256)
    attribution, sentiment = _json(aroot / "_manifest.json"), _json(sroot / "_manifest.json")
    for value, schema in ((attribution, "swing.event_attribution_manifest.v1"),
                          (sentiment, "swing.event_sentiment_manifest.v1")):
        if (value.get("schema") != schema or value.get("status") != "complete"
                or value.get("production_ready") is not False or value.get("failed_chunks") != {}):
            raise DataReadinessError("saved issuer authority is not complete research evidence")
    ar, sr = _request(aroot, attribution), _request(sroot, sentiment)
    if (ar.get("scope_policy") is not None or ar.get("schema") != "swing.event_attribution_request.v1"
            or ar.get("attribution_policy_sha256") != ATTRIBUTION_POLICY_SHA256
            or ar.get("attribution_policy_version") != ATTRIBUTION_POLICY_VERSION
            or sr.get("schema") != "swing.event_sentiment_request.v1"
            or sr.get("model_name") != "ProsusAI/finbert" or sr.get("model_revision") != FINBERT_REVISION):
        raise DataReadinessError("derivation requires the explicitly pinned old strict policy and exact FinBERT revision")
    for key in ("collection_manifest_sha256", "collection_audit_sha256", "collection_request_sha256"):
        if ar.get(key) != sr.get(key):
            raise DataReadinessError("saved attribution and sentiment have different raw source pins")
    cpath = _declared_path(ar["collection_manifest_path"], repository_root)
    audit_path = _declared_path(ar["collection_audit_path"], repository_root)
    if (cpath != _declared_path(sr["collection_manifest_path"], repository_root)
            or audit_path != _declared_path(sr["collection_audit_path"], repository_root)):
        raise DataReadinessError("saved issuer request source path association differs")
    _pin(cpath, ar["collection_manifest_sha256"])
    _pin(audit_path, ar["collection_audit_sha256"])
    collection, audit = _json(cpath), _json(audit_path)
    if (collection.get("status") != "complete" or collection.get("production_ready") is not False
            or collection.get("request_sha256") != ar["collection_request_sha256"]
            or audit.get("passed") is not True or audit.get("request_sha256") != collection["request_sha256"]):
        raise DataReadinessError("saved source collection/audit does not verify")
    _request(cpath.parent, collection)
    blindspots = audit.get("coverage_blindspot_security_ids")
    if not isinstance(blindspots, list) or any(not isinstance(v, str) or not v for v in blindspots):
        raise DataReadinessError("saved audit blindspot inventory is malformed")
    for payload in (ar, sr, attribution, sentiment):
        if payload.get("excluded_security_ids") != blindspots:
            raise DataReadinessError("saved exclusion diagnostics differ; no new whole-security exclusion is permitted")
    expected = {key for key, value in _records(collection).items() if value.get("security_id") not in blindspots}
    if set(_records(attribution)) != expected or not expected.issubset(_records(sentiment)):
        raise DataReadinessError("saved strict source inventory differs from its frozen historical exclusions")
    for key, pin in (("business_labels", source.business_labels_manifest_sha256),
                     ("security_identities", source.security_identities_manifest_sha256)):
        path = _declared_path(ar[f"{key}_path"], repository_root)
        _pin(manifest_path_for(path), pin)
        _pin(path, ar[f"{key}_sha256"])
    return ar, sr, attribution, sentiment, collection, audit


def _metadata(root: Path, record: Mapping[str, Any], kind: str, columns: list[str], *,
              repository_root: Path | None = None) -> tuple[Path, pd.DataFrame, dict[str, Any]]:
    path = _inside(root, _declared_path(record["path"], repository_root))
    if "manifest_path" in record and _declared_path(record["manifest_path"], repository_root) != manifest_path_for(path):
        raise DataReadinessError("saved issuer child manifest path association differs")
    frame, manifest = load_canonical_artifact(path, expected_type=kind, allow_research=True, columns=columns)
    if _declared_path(manifest["artifact_path"], repository_root) != path:
        raise DataReadinessError("saved issuer canonical artifact path association differs")
    if (manifest.get("artifact_sha256") != record.get("sha256") or manifest.get("production_ready") is not False
            or len(frame) != record.get("rows", len(frame))):
        raise DataReadinessError("saved issuer child hash, row count or mode differs")
    return path, frame, manifest


def _bounded_rows(path: Path, metadata: pd.DataFrame, *, clock: str, start: pd.Timestamp,
                  cutoff: pd.Timestamp, event_ids: set[str] | None = None) -> pd.DataFrame:
    """Plan eligibility from metadata, then push the predicate into numeric projection."""
    times = pd.to_datetime(metadata[clock], utc=True, errors="coerce")
    if times.isna().any():
        raise DataReadinessError("saved issuer evidence has an invalid availability clock")
    selected = times.ge(start) & times.le(cutoff)
    if event_ids is not None:
        selected &= metadata["event_id"].astype(str).isin(event_ids)
    ids = metadata.loc[selected, "event_id"].astype(str).unique().tolist()
    dataset = ds.dataset(path, format="parquet", partitioning=None)  # type: ignore[no-untyped-call]
    if not ids:
        return dataset.schema.empty_table().to_pandas()
    field: Any = ds.field  # type: ignore[attr-defined]
    predicate = ((field(clock) >= pa.scalar(start.to_pydatetime()))
                 & (field(clock) <= pa.scalar(cutoff.to_pydatetime()))
                 & field("event_id").isin(ids))
    frame = dataset.scanner(columns=dataset.schema.names, filter=predicate, use_threads=False).to_table().to_pandas()
    observed = pd.to_datetime(frame[clock], utc=True, errors="coerce")
    if (observed.isna().any() or not observed.between(start, cutoff).all()
            or not set(frame["event_id"].astype(str)).issubset(set(ids))):
        raise DataReadinessError("numeric issuer projection escaped its preselected initial-fit rows")
    return frame


def _write_json(path: Path, value: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8") as handle:
        json.dump(value, handle, sort_keys=True, indent=2)


def _validate_selected_scores(frame: pd.DataFrame) -> None:
    # event_relevance.v1 is an additive heuristic in [0.1, 2.0], not a probability.
    for column, lower, upper in (("sentiment_numeric", -1.0, 1.0), ("sentiment_confidence", 0.0, 1.0), ("relevance", 0.1, 2.0)):
        if column not in frame:
            raise DataReadinessError(f"saved selected scores lack {column}")
        numeric = pd.to_numeric(frame[column], errors="coerce")
        if numeric.isna().any() or not numeric.between(lower, upper).all():
            raise DataReadinessError(f"saved selected {column} is unavailable or outside its numeric contract")


def _verify_score_identity(events: pd.DataFrame, scores: pd.DataFrame) -> None:
    source = events.set_index("event_id")
    for column in ("security_id", "ticker"):
        expected = scores.event_id.map(source[column])
        if expected.isna().any() or not expected.eq(scores[column]).all():
            raise DataReadinessError("sentiment identity differs from its original source event")


def _write_frame(frame: pd.DataFrame, path: Path, kind: str, inputs: Mapping[str, str]) -> dict[str, Any]:
    audit = CanonicalAuditReport(checks=(
        CanonicalAuditCheck(name="verified_bounded_saved_projection", status="pass", failures=0,
                            rows_checked=len(frame), detail="pinned source bytes; metadata-first cutoff; no inference"),))
    manifest = write_canonical_artifact(frame, path, artifact_type=kind, audit=audit, inputs=inputs, production_ready=False)
    return {"path": str(path.resolve()), "sha256": manifest["artifact_sha256"], "rows": len(frame)}


def derive_initial_fit_issuer_inputs(*, source: SavedIssuerAuthority, output_directory: Path,
                                   start_utc: str = "2019-07-09T00:00:00Z",
                                   cutoff_utc: str = "2024-05-28T22:00:00Z",
                                   repository_root: Path | None = None,
                                   expected_existing_request_sha256: str | None = None) -> DerivedIssuerInputs:
    """Publish one saved generation; acquire the shared slot before any data read."""
    with heavy_job_lease("derive-initial-fit-issuer-inputs",
                         runtime_dir=repository_root.resolve() / "data/runtime" if repository_root is not None else None):
        assert_system_memory_available()
        return _derive(source, output_directory.resolve(), start_utc, cutoff_utc, repository_root=repository_root,
                       expected_existing_request_sha256=expected_existing_request_sha256)


def _derive(source: SavedIssuerAuthority, root: Path, start_utc: str, cutoff_utc: str, *,
            repository_root: Path | None = None, expected_existing_request_sha256: str | None = None) -> DerivedIssuerInputs:
    start, cutoff = pd.Timestamp(start_utc), pd.Timestamp(cutoff_utc)
    if (start.tzinfo is None or cutoff.tzinfo is None or start > cutoff or cutoff > LAST_INITIAL_FIT_CUTOFF
            or start < pd.Timestamp("2019-07-09T00:00:00Z")):
        raise DataReadinessError("issuer derivation requires explicit bounded initial-fit UTC clocks")
    repository_root = repository_root.resolve() if repository_root is not None else None
    ar, sr, attribution, sentiment, collection, audit = _saved_headers(source, repository_root=repository_root)
    old_sources, old_relations, old_scores = _records(collection), _records(attribution), _records(sentiment)
    collection_root = _declared_path(ar["collection_manifest_path"], repository_root).parent
    ledger_path = _inside(collection_root, _declared_path(collection["source_collections_path"], repository_root))
    ledger, lm = load_canonical_artifact(ledger_path, expected_type="source_collections", allow_research=True)
    if lm["artifact_sha256"] != collection["source_collections_sha256"]:
        raise DataReadinessError("saved source coverage ledger pin differs")
    starts = pd.to_datetime(ledger["requested_start_utc"], utc=True, errors="raise")
    ends = pd.to_datetime(ledger["requested_end_utc"], utc=True, errors="raise")
    ledger = ledger.loc[starts.le(cutoff) & ends.gt(start)].copy()
    if ledger["chunk_id"].duplicated().any():
        raise DataReadinessError("saved source coverage duplicates chunks")
    for column in ("requested_start_utc", "requested_end_utc", "status", "row_count"):
        ledger[f"original_{column}"] = ledger[column]
    ledger["projection_policy"] = SCHEMA
    ledger["requested_start_utc"] = pd.to_datetime(ledger["requested_start_utc"], utc=True).clip(lower=start)
    ledger["requested_end_utc"] = pd.to_datetime(ledger["requested_end_utc"], utc=True).clip(upper=cutoff)
    request = {"schema": SCHEMA, "scope_policy": ATTRIBUTION_SCOPE_POLICY,
        "source": {k: str(v.resolve()) if isinstance(v, Path) else v for k, v in asdict(source).items()},
        "start_utc": start.isoformat(), "cutoff_utc": cutoff.isoformat(), "production_ready": False,
        "source_coverage_admitted": False, "finbert_revision": FINBERT_REVISION,
        "collection_manifest_sha256": ar["collection_manifest_sha256"],
        "collection_audit_sha256": ar["collection_audit_sha256"], "original_source_ledger_sha256": lm["artifact_sha256"],
        "old_attribution_request_sha256": attribution["request_sha256"],
        "old_sentiment_request_sha256": sentiment["request_sha256"]}
    request_hash = json_sha256(request)
    if root.exists():
        if (root / "_manifest.json").exists():
            raise DataReadinessError("completed issuer derivation is immutable")
        if expected_existing_request_sha256 is None:
            raise DataReadinessError("request-only issuer derivation resume requires an external request pin")
        if {path.name for path in root.iterdir()} != {"_request.json"}:
            raise DataReadinessError("request-only resume refuses existing derived children; no files may be overwritten")
        _pin(root / "_request.json", expected_existing_request_sha256)
        if _json(root / "_request.json") != request:
            raise DataReadinessError("request-only issuer derivation resume request differs")
    else:
        if expected_existing_request_sha256 is not None:
            raise DataReadinessError("request-only issuer derivation resume directory is absent")
        root.mkdir(parents=True)
        _write_json(root / "_request.json", request)
    source_rows: list[dict[str, Any]] = []
    relation_rows: list[dict[str, Any]] = []
    score_rows: list[dict[str, Any]] = []
    gaps: list[dict[str, Any]] = []
    source_children = {str(ledger_path): str(lm["artifact_sha256"]),
        str(manifest_path_for(ledger_path)): file_sha256(manifest_path_for(ledger_path))}
    for index, row in ledger.iterrows():
        assert_system_memory_available()
        chunk = str(row["chunk_id"])
        if str(row["status"]) == "observed_empty":
            continue
        if chunk not in old_sources:
            raise DataReadinessError("saved observed ledger chunk lacks source evidence")
        if chunk not in old_relations or chunk not in old_scores:
            ledger.loc[index, "status"] = "unavailable_saved_evidence"
            gaps.append({"chunk_id": chunk, "reason": "old_attribution_or_sentiment_not_published"})
            continue
        ep, em, ec = _metadata(collection_root, old_sources[chunk], "events",
            ["event_id", "feature_available_at_utc", "security_id", "ticker"], repository_root=repository_root)
        rp, rm, rc = _metadata(source.attribution_dir, old_relations[chunk], "event_security_relations",
                               ["event_id", "feature_available_at_utc"], repository_root=repository_root)
        sp, sm, sc = _metadata(source.sentiment_dir, old_scores[chunk], "event_sentiment_research",
            ["event_id", "research_feature_available_at_utc", "sentiment_model_revision", "security_id", "ticker"],
            repository_root=repository_root)
        for path, child in ((ep, ec), (rp, rc), (sp, sc)):
            source_children[str(path)] = str(child["artifact_sha256"])
            source_children[str(manifest_path_for(path))] = file_sha256(manifest_path_for(path))
        original_hash = ec["artifact_sha256"]
        for child, key, parent_hash in ((rc, "event_attribution_request_sha256", attribution["request_sha256"]),
                                        (sc, "sentiment_request_sha256", sentiment["request_sha256"])):
            ci = child.get("inputs", {})
            if ci.get(key) != parent_hash or ci.get("source_event_artifact_sha256") != original_hash or ci.get("chunk_id") != chunk:
                raise DataReadinessError("saved issuer child transitive source/request binding differs")
        if em["event_id"].duplicated().any() or sm["event_id"].duplicated().any() or set(em.event_id) != set(sm.event_id):
            raise DataReadinessError("saved sentiment inventory differs from its original source chunk")
        if not set(rm.event_id).issubset(set(em.event_id)) or not sm.sentiment_model_revision.eq(FINBERT_REVISION).all():
            raise DataReadinessError("saved issuer relation inventory or exact score revision differs")
        _verify_score_identity(em, sm)
        eligible = pd.to_datetime(em.feature_available_at_utc, utc=True).between(start, cutoff)
        source_ids = set(em.loc[eligible, "event_id"].astype(str))
        scores = _bounded_rows(sp, sm, clock="research_feature_available_at_utc", start=start, cutoff=cutoff, event_ids=source_ids)
        ids = set(scores.event_id.astype(str))
        events = _bounded_rows(ep, em, clock="feature_available_at_utc", start=start, cutoff=cutoff, event_ids=ids)
        ids = set(events.event_id.astype(str))
        scores = scores.loc[scores.event_id.astype(str).isin(ids)].copy()
        _validate_selected_scores(scores)
        relations = _bounded_rows(rp, rm, clock="feature_available_at_utc", start=start, cutoff=cutoff, event_ids=ids)
        eligible_source_count = int(pd.to_datetime(em.feature_available_at_utc, utc=True).between(start, cutoff).sum())
        if len(events) != eligible_source_count:
            ledger.loc[index, "status"] = "unavailable_saved_evidence"
            gaps.append({"chunk_id": chunk, "reason": "source_events_not_score_available_by_cutoff",
                         "unavailable_events": eligible_source_count - len(events)})
        ledger.loc[index, "row_count"] = len(events)
        common = {"issuer_derivation_request_sha256": request_hash, "chunk_id": chunk,
                  "original_source_event_artifact_sha256": str(original_hash)}
        event = _write_frame(events, root / "collection/events" / f"{chunk}.parquet", "events", common)
        child_inputs = {**common, "source_event_artifact_sha256": str(event["sha256"])}
        relation = _write_frame(relations, root / "attribution/relations" / f"{chunk}.parquet", "event_security_relations",
                                {**child_inputs, "original_relation_artifact_sha256": str(rc["artifact_sha256"])})
        score = _write_frame(scores, root / "sentiment/sentiment" / f"{chunk}.parquet", "event_sentiment_research",
                             {**child_inputs, "original_sentiment_artifact_sha256": str(sc["artifact_sha256"])})
        identity = {"chunk_id": chunk, "security_id": str(row["security_id"]), "ticker": str(row["ticker"])}
        source_rows.append({**event, **identity})
        relation_rows.append({**relation, **identity, "source_event_sha256": event["sha256"]})
        score_rows.append({**score, **identity, "source_event_artifact_sha256": event["sha256"]})
    coverage = _write_frame(ledger, root / "collection/source_collections.parquet", "source_collections",
                            {"issuer_derivation_request_sha256": request_hash, "original_source_ledger_sha256": str(lm["artifact_sha256"])})
    shared = {"scope_policy": ATTRIBUTION_SCOPE_POLICY, "coverage_blindspot_security_ids": audit["coverage_blindspot_security_ids"],
              "source_coverage_admitted": False, "excluded_security_ids": [], "production_ready": False,
              "status": "complete", "failed_chunks": {}, "request_sha256": request_hash,
              "derivation_only": True}
    for name, rows in (("collection", source_rows), ("attribution", relation_rows), ("sentiment", score_rows)):
        value = {**shared, "schema": f"{SCHEMA}.{name}", "artifacts": rows,
                 "total_rows": sum(int(item["rows"]) for item in rows)}
        if name == "collection":
            value.update(source_collections_path=coverage["path"], source_collections_sha256=coverage["sha256"])
        _write_json(root / name / "_manifest.json", value)
    _write_json(root / "collection/_audit.json", {**shared, "passed": True,
        "audit_scope": "verified_saved_evidence_projection_not_new_provider_coverage"})
    _write_json(root / "attribution/_request.json", {**ar, **shared})
    _saved_headers(source, repository_root=repository_root)
    for original_path, pin in source_children.items():
        _pin(Path(original_path), pin)
    _write_json(root / "_source_children.json", source_children)
    inventory = {p.relative_to(root).as_posix(): file_sha256(p) for p in root.rglob("*") if p.is_file()}
    manifest = {**shared, "schema": SCHEMA, "request": request, "inventory": inventory,
                "source_path_resolution": {"policy": "explicit_repository_base_authority_containment_v1",
                    "repository_root": str(repository_root) if repository_root is not None else None},
                "unavailable_chunks": gaps, "source_events": sum(r["rows"] for r in source_rows),
                "relations": sum(r["rows"] for r in relation_rows), "scored_events": sum(r["rows"] for r in score_rows)}
    _write_json(root / "_manifest.json", manifest)
    return load_initial_fit_issuer_inputs(root, expected_manifest_sha256=file_sha256(root / "_manifest.json"))


def load_initial_fit_issuer_inputs(directory: Path, *, expected_manifest_sha256: str) -> DerivedIssuerInputs:
    root = directory.resolve()
    _pin(root / "_manifest.json", expected_manifest_sha256)
    manifest = _json(root / "_manifest.json")
    request = _json(root / "_request.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("request") != request
            or manifest.get("request_sha256") != json_sha256(request)
            or request.get("scope_policy") != ATTRIBUTION_SCOPE_POLICY
            or request.get("source_coverage_admitted") is not False
            or request.get("production_ready") is not False
            or manifest.get("status") != "complete" or manifest.get("excluded_security_ids") != []
            or manifest.get("source_coverage_admitted") is not False or manifest.get("production_ready") is not False
            or request.get("finbert_revision") != FINBERT_REVISION):
        raise DataReadinessError("issuer derivation request or scope contract differs")
    cutoff = pd.Timestamp(request["cutoff_utc"])
    if cutoff.tzinfo is None or cutoff > LAST_INITIAL_FIT_CUTOFF:
        raise DataReadinessError("issuer derivation cutoff crosses initial fit")
    inventory = manifest.get("inventory")
    if not isinstance(inventory, dict):
        raise DataReadinessError("issuer derivation inventory is malformed")
    actual = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()}
    if actual != {*inventory, "_manifest.json"}:
        raise DataReadinessError("issuer derivation file inventory differs")
    for relative, pin in inventory.items():
        _pin(_inside(root, relative), pin)
    source_children = _json(root / "_source_children.json")
    if not source_children:
        raise DataReadinessError("issuer derivation lacks original child proof")
    for path, pin in source_children.items():
        _pin(Path(path), pin)
    source = dict(request["source"])
    for key in ("attribution_dir", "sentiment_dir"):
        source[key] = Path(source[key])
    resolution = manifest.get("source_path_resolution", {})
    if resolution.get("policy") != "explicit_repository_base_authority_containment_v1":
        raise DataReadinessError("issuer derivation lacks its declared source path resolution")
    repository_root = Path(resolution["repository_root"]) if resolution.get("repository_root") is not None else None
    if repository_root is not None and not repository_root.is_absolute():
        raise DataReadinessError("issuer derivation repository path base must be absolute")
    _saved_headers(SavedIssuerAuthority(**source), repository_root=repository_root)
    start = pd.Timestamp(request["start_utc"])
    for name, kind, clock in (("collection", "events", "feature_available_at_utc"),
                              ("attribution", "event_security_relations", "feature_available_at_utc"),
                              ("sentiment", "event_sentiment_research", "research_feature_available_at_utc")):
        for record in _records(_json(root / name / "_manifest.json")).values():
            _, metadata, _ = _metadata(root, record, kind, ["event_id", clock])
            times = pd.to_datetime(metadata[clock], utc=True, errors="coerce")
            if times.isna().any() or not times.between(start, cutoff).all():
                raise DataReadinessError("derived issuer child escaped its initial-fit clock bounds")
    for path, pin in source_children.items():
        _pin(Path(path), pin)
    return DerivedIssuerInputs(root, expected_manifest_sha256, manifest)


def build_initial_fit_catalyst_lineage(*, derived_directory: Path, expected_manifest_sha256: str,
                                     decisions_path: Path, policy_path: Path, out_dir: Path) -> dict[str, object]:
    """Consume only this explicit derivation through the current lineage publisher."""
    from market_predictor.swing.catalyst_lineage import build_catalyst_lineage

    with heavy_job_lease("build-initial-fit-catalyst-lineage"):
        assert_system_memory_available()
        derived = load_initial_fit_issuer_inputs(derived_directory, expected_manifest_sha256=expected_manifest_sha256)
        _verify_decision_bounds(decisions_path, derived)
        root = derived.directory
        source_inventory = _records(_json(root / "collection/_manifest.json"))
        return build_catalyst_lineage(collection_dir=root / "collection", collection_audit_path=root / "collection/_audit.json",
            attribution_dir=root / "attribution", sentiment_dir=root / "sentiment", decisions_path=decisions_path,
            policy_path=policy_path, out_dir=out_dir, _verified_derived_source_inventory=source_inventory)


def publish_initial_fit_issuer_family(*, derived_directory: Path, expected_manifest_sha256: str,
                                    decisions_path: Path, policy_path: Path, output_directory: Path) -> SwingIssuerFamilyCohort:
    """Use the existing causal family classifier, never a newly trained model."""
    from market_predictor.swing.datasets.issuer_event_family_cohort import publish_swing_issuer_family_cohort

    with heavy_job_lease("publish-initial-fit-issuer-family"):
        assert_system_memory_available()
        derived = load_initial_fit_issuer_inputs(derived_directory, expected_manifest_sha256=expected_manifest_sha256)
        _verify_decision_bounds(decisions_path, derived)
        root = derived.directory
        manifest = _json(root / "attribution/_manifest.json")
        history = EventAttributionHistory(directory=root / "attribution", request=_json(root / "attribution/_request.json"),
            manifest=manifest, artifact_records=tuple(manifest["artifacts"]))
        return publish_swing_issuer_family_cohort(collection_dir=root / "collection", collection_audit_path=root / "collection/_audit.json",
            attribution_dir=root / "attribution", decisions_path=decisions_path, policy_path=policy_path,
            output_directory=output_directory, _verified_derived_attribution=history)


def _verify_decision_bounds(path: Path, derived: DerivedIssuerInputs) -> None:
    decisions, _ = load_canonical_artifact(path, expected_type="decisions", allow_research=True,
                                         columns=["decision_time_utc"])
    times = pd.to_datetime(decisions["decision_time_utc"], utc=True, errors="coerce")
    request = derived.manifest["request"]
    if times.isna().any() or not times.between(pd.Timestamp(request["start_utc"]), pd.Timestamp(request["cutoff_utc"])).all():
        raise DataReadinessError("canonical decisions cross the derived initial-fit interval")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    derive = subparsers.add_parser("derive")
    derive.add_argument("--root", type=Path, default=Path("."))
    derive.add_argument("--config", type=Path, required=True)
    derive.add_argument("--expected-config-sha256", required=True)
    derive.add_argument("--generation", choices=("early", "later"), required=True)
    derive.add_argument("--out-dir", type=Path, required=True)
    derive.add_argument("--expected-existing-request-sha256", help="Resume only an externally pinned request-only interrupted output")
    for name in ("lineage", "family"):
        command = subparsers.add_parser(name)
        command.add_argument("--derived-dir", type=Path, required=True)
        command.add_argument("--expected-manifest-sha256", required=True)
        command.add_argument("--decisions", type=Path, required=True)
        command.add_argument("--policy", type=Path, required=True)
        command.add_argument("--out-dir", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "derive":
        root = args.root.resolve()
        config = args.config if args.config.is_absolute() else root / args.config
        _pin(config, args.expected_config_sha256)
        policy = tomllib.loads(config.read_text(encoding="utf-8"))
        if policy.get("schema") != "swing.initial_fit_issuer_derivation_config.v1":
            raise DataReadinessError("unsupported explicit issuer derivation config")
        fields = {**policy["source_identity"], **policy["sources"][args.generation]}
        for key in ("attribution_dir", "sentiment_dir"):
            fields[key] = _inside(root, fields[key])
        output = args.out_dir if args.out_dir.is_absolute() else root / args.out_dir
        result = derive_initial_fit_issuer_inputs(source=SavedIssuerAuthority(**fields), output_directory=output,
            start_utc=policy["start_utc"], cutoff_utc=policy["cutoff_utc"], repository_root=root,
            expected_existing_request_sha256=args.expected_existing_request_sha256)
        print(json.dumps({"directory": str(result.directory), "manifest_sha256": result.manifest_sha256,
                          "source_events": result.manifest["source_events"], "scored_events": result.manifest["scored_events"],
                          "relations": result.manifest["relations"], "unavailable_chunks": len(result.manifest["unavailable_chunks"])}))
    elif args.command == "lineage":
        report = build_initial_fit_catalyst_lineage(derived_directory=args.derived_dir,
            expected_manifest_sha256=args.expected_manifest_sha256, decisions_path=args.decisions,
            policy_path=args.policy, out_dir=args.out_dir)
        print(json.dumps({key: report[key] for key in ("status", "relation_rows", "assignment_rows", "failed_chunks")}))
    else:
        family = publish_initial_fit_issuer_family(derived_directory=args.derived_dir,
            expected_manifest_sha256=args.expected_manifest_sha256, decisions_path=args.decisions,
            policy_path=args.policy, output_directory=args.out_dir)
        print(json.dumps({"family_rows": len(family.events), "assignment_rows": len(family.assignments)}))


if __name__ == "__main__":
    main()
