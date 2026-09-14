"""One-pass source projection into verified monthly issuer lineage inputs."""
from __future__ import annotations

import argparse
import json
import re
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime, timedelta
from math import ceil
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.reconciliation import (
    assignment_integrity_summary,
    build_event_assignments,
    reconciliation_sha256,
    stamp_canonical_decision_ids,
)
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.catalysts.issuer_events.attribution import ATTRIBUTION_POLICY_SHA256
from market_predictor.catalysts.issuer_events.attribution_history import ATTRIBUTION_SCOPE_POLICY
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.system_memory import assert_system_memory_available
from market_predictor.evidence.hashing import json_sha256
from market_predictor.resources import assert_memory_budget, memory_audit, release_process_memory
from market_predictor.swing.datasets.initial_fit_issuer_news import (
    FINBERT_REVISION,
    LAST_INITIAL_FIT_CUTOFF,
    _bounded_rows,
    _metadata,
    _records,
    _request,
    _validate_selected_scores,
    _verify_score_identity,
    _write_frame,
    _write_json,
    load_initial_fit_issuer_inputs,
)
from market_predictor.swing.datasets.issuer_news_publication import (
    CONFIG_SCHEMA as PUBLICATION_SCHEMA,
)
from market_predictor.swing.datasets.issuer_news_publication import (
    _check_sources,
    _checkpoint,
    _decisions,
    _month,
    _object,
    _path,
    _pin,
)

SCHEMA = "swing.initial_fit_monthly_lineage_preparation.v1"
FIRST = pd.Timestamp("2019-07-09T00:00:00Z")
BATCH_ROWS = 5000
BATCH_BYTES = 64 * 1024 * 1024
DECISION_COLUMNS = ["decision_id", "security_id", "ticker", "decision_time_utc", "timeframe",
    "bar_start_utc", "prediction_cutoff_policy_id"]


def _implementation_identity() -> dict[str, str]:
    package = Path(__file__).resolve().parents[2]
    names = ("swing/datasets/issuer_news_preparation.py", "swing/datasets/initial_fit_issuer_news.py",
        "swing/datasets/issuer_news_publication.py", "swing/catalyst_lineage.py", "canonical/reconciliation.py", "canonical/store.py")
    return {name: file_sha256(package / name) for name in names}


def _guard() -> None:
    assert_system_memory_available(minimum_available_gib=3.0, maximum_used_percent=82.0)
    assert_memory_budget(hard_budget_gib=5.0, headroom_gib=0.75, stage="monthly issuer preparation")


@contextmanager
def _decision_projection(root: Path, config: dict[str, Any]) -> Iterator[Any]:
    from market_predictor.swing.datasets.corrected_decisions import verified_corrected_decision_partitions

    # This existing context owns the root-bound heavy lease and rechecks all source pins.
    with verified_corrected_decision_partitions(root=root, config=_path(root, config["path"]),
        expected_config_sha256=config["sha256"]) as projection:
        yield projection


def _source(root: Path, spec: dict[str, Any]) -> dict[str, Any]:
    if spec["kind"] == "derived":
        directory = _path(root, spec["directory"])
        derived = load_initial_fit_issuer_inputs(directory, expected_manifest_sha256=spec["manifest_sha256"])
        roots = {name: directory / name for name in ("collection", "attribution", "sentiment")}
        manifests = {name: _object(path / "_manifest.json") for name, path in roots.items()}
        audit = _object(roots["collection"] / "_audit.json")
        parent_pins = {str(directory / "_manifest.json"): derived.manifest_sha256}
        original_index = directory / "_source_children.json"
        parent_pins[str(original_index)] = file_sha256(original_index)
        parent_pins.update(_object(original_index))
        request_hash = str(derived.manifest["request_sha256"])
    elif spec["kind"] == "corrected":
        roots = {name: _path(root, spec[name]["directory"]) for name in ("collection", "attribution", "sentiment")}
        parent_pins = {str(path / "_manifest.json"): spec[name]["manifest_sha256"] for name, path in roots.items()}
        audit_path = _path(root, spec["audit"]["path"])
        parent_pins[str(audit_path)] = spec["audit"]["sha256"]
        _check_sources(root, parent_pins)
        manifests = {name: _object(path / "_manifest.json") for name, path in roots.items()}
        audit = _object(audit_path)
        ar = _request(roots["attribution"], manifests["attribution"])
        sr = _request(roots["sentiment"], manifests["sentiment"])
        if (ar.get("attribution_policy_sha256") != ATTRIBUTION_POLICY_SHA256
                or sr.get("model_revision") != FINBERT_REVISION or sr.get("model_name") != "ProsusAI/finbert"):
            raise DataReadinessError("corrected issuer inputs differ from strict attribution or exact FinBERT revision")
        for request in (ar, sr):
            if (request.get("collection_manifest_sha256") != spec["collection"]["manifest_sha256"]
                    or request.get("collection_audit_sha256") != spec["audit"]["sha256"]
                    or request.get("collection_request_sha256") != manifests["collection"]["request_sha256"]
                    or request.get("scope_policy") != ATTRIBUTION_SCOPE_POLICY
                    or request.get("excluded_security_ids") != [] or request.get("source_coverage_admitted") is not False):
                raise DataReadinessError("corrected issuer request source or scope binding differs")
            if (_path(root, request["collection_manifest_path"]) != roots["collection"] / "_manifest.json"
                    or _path(root, request["collection_audit_path"]) != audit_path):
                raise DataReadinessError("corrected issuer request source path association differs")
        request_hash = manifests["collection"]["request_sha256"]
        if audit.get("request_sha256") != request_hash:
            raise DataReadinessError("corrected source audit request differs")
    else:
        raise DataReadinessError("monthly preparation accepts only verified derived or corrected issuer inputs")
    for manifest in manifests.values():
        if manifest.get("status") != "complete" or manifest.get("production_ready") is not False or manifest.get("failed_chunks", {}) != {}:
            raise DataReadinessError("monthly preparation source is not completed research evidence")
    if audit.get("passed") is not True:
        raise DataReadinessError("monthly preparation requires a passed source audit")
    blind = set(audit["coverage_blindspot_security_ids"])
    for name in ("attribution", "sentiment"):
        manifest = manifests[name]
        if (manifest.get("scope_policy") != ATTRIBUTION_SCOPE_POLICY or manifest.get("excluded_security_ids") != []
                or manifest.get("source_coverage_admitted") is not False
                or set(manifest.get("coverage_blindspot_security_ids", [])) != blind):
            raise DataReadinessError("monthly preparation cannot change attribution admission or blindspot diagnostics")
    records = {name: _records(manifest) for name, manifest in manifests.items()}
    if set(records["collection"]) != set(records["attribution"]) or not set(records["collection"]).issubset(records["sentiment"]):
        raise DataReadinessError("monthly preparation source inventories differ")
    ledger_path = _path(root, manifests["collection"]["source_collections_path"])
    _pin(ledger_path, manifests["collection"]["source_collections_sha256"])
    ledger, _ = load_canonical_artifact(ledger_path, expected_type="source_collections", allow_research=True)
    if ledger.chunk_id.duplicated().any():
        raise DataReadinessError("monthly preparation coverage ledger duplicates source chunks")
    parent_pins[str(ledger_path)] = manifests["collection"]["source_collections_sha256"]
    for chunk in set(records["sentiment"]) - set(records["collection"]):
        row = ledger.loc[ledger.chunk_id.astype(str).eq(chunk)]
        if len(row) != 1 or row.iloc[0].status != "observed_empty" or int(row.iloc[0].row_count) != 0:
            raise DataReadinessError("extra sentiment source chunk is not independently observed empty")
        path, frame, manifest = _metadata(roots["sentiment"], records["sentiment"][chunk], "event_sentiment_research",
            ["event_id"], repository_root=root)
        inputs = manifest.get("inputs", {})
        if (not frame.empty or inputs.get("sentiment_request_sha256") != manifests["sentiment"]["request_sha256"]
                or inputs.get("chunk_id") != chunk):
            raise DataReadinessError("empty sentiment source child request binding differs")
        parent_pins[str(path)] = manifest["artifact_sha256"]
        parent_pins[str(manifest_path_for(path))] = file_sha256(manifest_path_for(path))
    return {"roots": roots, "repository_root": root, "manifests": manifests, "records": records, "ledger": ledger,
        "blindspots": blind, "parent_pins": parent_pins, "request_sha256": request_hash, "kind": spec["kind"]}


def _chunk(source: dict[str, Any], chunk: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, str], pd.DataFrame]:
    from market_predictor.swing.catalyst_lineage import _verify_chunk_lineage

    metadata: dict[str, tuple[Path, pd.DataFrame, dict[str, Any]]] = {}
    original_pins: dict[str, str] = {}
    for name, kind, clock in (("collection", "events", "feature_available_at_utc"),
        ("attribution", "event_security_relations", "feature_available_at_utc"),
        ("sentiment", "event_sentiment_research", "research_feature_available_at_utc")):
        record = source["records"][name][chunk]
        columns = ["event_id", clock]
        if name == "sentiment":
            columns.extend(["sentiment_model_revision", "security_id", "ticker"])
        elif name == "collection":
            columns.extend(["security_id", "ticker"])
        item = _metadata(source["roots"][name], record, kind, columns, repository_root=source["repository_root"])
        metadata[name] = item
        original_pins[str(item[0])] = record["sha256"]
        original_pins[str(manifest_path_for(item[0]))] = file_sha256(manifest_path_for(item[0]))
    ep, em, ec = metadata["collection"]
    rp, rm, rc = metadata["attribution"]
    sp, sm, sc = metadata["sentiment"]
    if (em.event_id.duplicated().any() or sm.event_id.duplicated().any() or set(em.event_id) != set(sm.event_id)
            or not set(rm.event_id).issubset(em.event_id) or not sm.sentiment_model_revision.eq(FINBERT_REVISION).all()):
        raise DataReadinessError("source chunk metadata identity or scorer revision differs")
    _verify_score_identity(em, sm)
    for name, manifest in (("attribution", rc), ("sentiment", sc)):
        inputs = manifest.get("inputs", {})
        if inputs.get("source_event_artifact_sha256") != ec["artifact_sha256"] or inputs.get("chunk_id") != chunk:
            raise DataReadinessError("monthly projection source child is not bound to its actual event artifact")
        if source["kind"] == "corrected":
            key = "event_attribution_request_sha256" if name == "attribution" else "sentiment_request_sha256"
            if inputs.get(key) != source["manifests"][name]["request_sha256"]:
                raise DataReadinessError("corrected source child request pin differs")
        elif inputs.get("issuer_derivation_request_sha256") != source["request_sha256"]:
            raise DataReadinessError("derived source child request pin differs")
    eligible = pd.to_datetime(em.feature_available_at_utc, utc=True).between(FIRST, LAST_INITIAL_FIT_CUTOFF)
    availability = em.copy()
    availability["score_available_at_utc"] = _aligned_score_times(availability.event_id, sm)
    scores = _bounded_rows(sp, sm, clock="research_feature_available_at_utc", start=FIRST, cutoff=LAST_INITIAL_FIT_CUTOFF,
        event_ids=set(em.loc[eligible, "event_id"].astype(str)))
    events = _bounded_rows(ep, em, clock="feature_available_at_utc", start=FIRST, cutoff=LAST_INITIAL_FIT_CUTOFF,
        event_ids=set(scores.event_id.astype(str)))
    relations = _bounded_rows(rp, rm, clock="feature_available_at_utc", start=FIRST, cutoff=LAST_INITIAL_FIT_CUTOFF,
        event_ids=set(events.event_id.astype(str)))
    _validate_selected_scores(scores)
    _verify_chunk_lineage(chunk_id=chunk, source_record=source["records"]["collection"][chunk],
        relation_record=source["records"]["attribution"][chunk], sentiment_record=source["records"]["sentiment"][chunk],
        source_manifest=ec, relation_manifest=rc, sentiment_manifest=sc, source_events=events, relations=relations, sentiments=scores)
    for path, digest in original_pins.items():
        _pin(Path(path), digest)
    return events, relations, scores, original_pins, availability


def _aligned_score_times(event_ids: pd.Series, scores: pd.DataFrame) -> pd.Series:
    # Reindex preserves timezone-aware dtype even when the source has no events.
    times = pd.to_datetime(scores.set_index("event_id").research_feature_available_at_utc, utc=True)
    return times.reindex(event_ids).set_axis(event_ids.index)


def _select(events: pd.DataFrame, relations: pd.DataFrame, scores: pd.DataFrame, start: pd.Timestamp,
    end: pd.Timestamp) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, bool]:
    score_times = _aligned_score_times(events.event_id, scores)
    event_times = pd.to_datetime(events.feature_available_at_utc, utc=True)
    ready = pd.concat([event_times, score_times], axis=1).max(axis=1)
    relation_ready = pd.concat([pd.to_datetime(relations.feature_available_at_utc, utc=True),
        _aligned_score_times(relations.event_id, scores)], axis=1).max(axis=1)
    selected_relations = relations.loc[relation_ready.between(start, end)].copy()
    ids = set(events.loc[ready.between(start, end), "event_id"]) | set(selected_relations.event_id)
    pending = bool((event_times.between(start, end) & score_times.gt(end)).any())
    return events.loc[events.event_id.isin(ids)].copy(), selected_relations, scores.loc[scores.event_id.isin(ids)].copy(), pending


def _normalize_compact_clocks(frame: pd.DataFrame) -> None:
    # A null-only first slice must not define a naive Arrow schema that strips
    # timezone metadata from subsequent slices. Validate before normalizing.
    for column in frame.columns:
        if not column.endswith("_utc"):
            continue
        for value in frame[column].dropna():
            try:
                stamp = pd.Timestamp(value)
                if pd.isna(stamp) or stamp.tzinfo is None or stamp.utcoffset() != timedelta(0):
                    raise ValueError("not aware UTC")
            except (TypeError, ValueError, OverflowError) as exc:
                raise DataReadinessError(f"compact {column} requires aware UTC source timestamps") from exc
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit("ns")


class _CompactSink:
    """Stream bounded row groups to disk, holding no cross-month article buffers."""

    def __init__(self, view: Path, request: str) -> None:
        self.view, self.request = view, request
        self.writers: dict[str, Any] = {}
        self.schemas: dict[str, Any] = {}
        self.records: dict[str, list[dict[str, Any]]] = {name: [] for name in ("events", "relations", "scores")}
        self.rows = self.size = self.batch = 0
        self.children: dict[str, Any] = {}

    def append(self, frames: tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame], chunk: str,
        pins: dict[str, str]) -> None:
        events, relations, scores = frames
        offset = 0
        while offset < len(events):
            count = min(BATCH_ROWS - self.rows, len(events) - offset)
            part = events.iloc[offset:offset + count].copy()
            ids = set(part.event_id)
            selected = (part, relations.loc[relations.event_id.isin(ids)].copy(), scores.loc[scores.event_id.isin(ids)].copy())
            tables = {}
            for name, frame in zip(self.records, selected, strict=True):
                _normalize_compact_clocks(frame)
                frame["preparation_original_chunk_id"] = chunk
                frame["preparation_source_child_set_sha256"] = json_sha256(pins)
                table = pa.Table.from_pandas(frame, preserve_index=False)
                schema = pa.schema([pa.field(field.name, pa.string() if pa.types.is_null(field.type) else field.type)
                    for field in table.schema])
                tables[name] = table.cast(schema)
            size = sum(table.nbytes for table in tables.values())
            if size > BATCH_BYTES:
                raise DataReadinessError("one source slice exceeds compact projection byte budget")
            if self.rows and self.size + size > BATCH_BYTES:
                self.finish()
                continue
            self.view.mkdir(parents=True, exist_ok=True)
            for name, table in tables.items():
                if name not in self.writers:
                    self.schemas[name] = table.schema
                    self.writers[name] = pq.ParquetWriter(  # type: ignore[no-untyped-call]
                        self.view / f".{name}.pending", table.schema, use_dictionary=False)
                self.writers[name].write_table(table.cast(self.schemas[name]))
            self.children[chunk] = pins
            self.rows += count
            self.size += size
            offset += count
            if self.rows == BATCH_ROWS:
                self.finish()

    def finish(self) -> None:
        if not self.rows:
            return
        self.close()
        chunk = f"batch-{self.batch:06d}"
        common = {"monthly_preparation_request_sha256": self.request, "chunk_id": chunk,
            "original_source_child_set_sha256": json_sha256(self.children)}
        source_hash = ""
        for name, folder, kind in (("events", "collection/events", "events"),
            ("relations", "attribution/relations", "event_security_relations"),
            ("scores", "sentiment/sentiment", "event_sentiment_research")):
            _guard()
            temporary = self.view / f".{name}.pending"
            frame = pd.read_parquet(temporary)
            inputs = common if name == "events" else {**common, "source_event_artifact_sha256": source_hash}
            record = _write_frame(frame, self.view / folder / f"{chunk}.parquet", kind, inputs)
            record.update(chunk_id=chunk, original_source_children=self.children.copy())
            if name == "events":
                source_hash = record["sha256"]
            else:
                record["source_event_sha256" if name == "relations" else "source_event_artifact_sha256"] = source_hash
            self.records[name].append(record)
            temporary.unlink()
            del frame
        self.batch += 1
        self.rows = self.size = 0
        self.children = {}

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()
        self.schemas.clear()


def estimate_monthly_projection_files(ledger: pd.DataFrame, months: dict[str, Any],
    lookback: pd.Timedelta) -> dict[str, Any]:
    """Ledger-only sizing: whole overlapping-chunk row counts are conservative.

    Byte-limit splits are explicitly additional, not hidden in the row estimate.
    No event text, sentiment values, or held-out numeric columns are loaded.
    """
    starts = pd.to_datetime(ledger.requested_start_utc, utc=True)
    ends = pd.to_datetime(ledger.requested_end_utc, utc=True)
    records = {}
    for month, record in months.items():
        lower = max(FIRST, pd.Timestamp(record["first_decision_utc"]) - lookback)
        upper = pd.Timestamp(record["last_decision_utc"])
        selected = ledger.loc[starts.le(upper) & ends.gt(lower)]
        if selected.empty:
            continue
        rows = int(pd.to_numeric(selected.row_count, errors="raise").sum())
        batches = ceil(rows / BATCH_ROWS)
        records[month] = {"logical_chunk_overlaps": len(selected), "source_rows_upper_estimate": rows,
            "row_limited_batches": batches, "projection_files_estimate": 6 + 6 * batches}
    batches = sum(row["row_limited_batches"] for row in records.values())
    overlaps = sum(row["logical_chunk_overlaps"] for row in records.values())
    return {"months": records, "logical_chunk_overlaps": overlaps,
        "old_per_chunk_projection_files": 6 * overlaps, "row_limited_batches": batches,
        "projection_files_estimate": 2 + 6 * len(records) + 6 * batches,
        "lineage_files_estimate": 6 * len(records) + 4 * batches,
        "batch_rows": BATCH_ROWS, "batch_bytes": BATCH_BYTES,
        "estimate_policy": "overlapping_ledger_rows_upper_estimate_before_clock_filtering",
        "byte_limit_split_cost": {"projection_files": 6, "lineage_files": 4},
        "coverage_only_event_parquets": 0}


def _pending_coverage_intervals(state: dict[str, Any], chunk: str, availability: pd.DataFrame,
    lookback: pd.Timedelta) -> None:
    rows = state["ledger"].chunk_id.astype(str).eq(chunk)
    if not rows.any():
        return
    event_times = pd.to_datetime(availability.feature_available_at_utc, utc=True)
    score_times = pd.to_datetime(availability.score_available_at_utc, utc=True)
    pending = pd.Series(False, index=availability.index)
    for cutoff in state["decision_cutoffs"]:
        pending |= event_times.between(cutoff - lookback, cutoff) & score_times.gt(cutoff)
    if not pending.any():
        state["logical_chunks"].append(chunk)
        return
    gaps = sorted(zip(event_times.loc[pending], score_times.loc[pending], strict=True))
    original = state["ledger"].loc[rows].iloc[0]
    cursor, end = original.requested_start_utc, original.requested_end_utc
    segments = []
    for gap_start, gap_end in gaps:
        if gap_start > cursor:
            segments.append((cursor, min(gap_start, end)))
        cursor = max(cursor, gap_end)
        state["unavailable_score_tails"].append({"chunk_id": chunk, "from_utc": gap_start.isoformat(),
            "until_utc": gap_end.isoformat(), "reason": "source_observed_but_score_pending_at_canonical_cutoff"})
        if cursor >= end:
            break
    if cursor < end:
        segments.append((cursor, end))
    replacement = []
    for index, (start, stop) in enumerate(segments):
        if start >= stop:
            continue
        row = original.copy()
        identity = chunk if index == 0 else f"{chunk}-coverage-{index:04d}"
        row["chunk_id"] = identity
        row["preparation_original_chunk_id"] = chunk
        row["requested_start_utc"], row["requested_end_utc"] = start, stop
        # A gap stays unknown for lookbacks crossing it, even after scoring catches up.
        row["row_count"] = int(event_times.between(start, stop, inclusive="left").sum())
        replacement.append(row)
        state["logical_chunks"].append(identity)
    state["ledger"] = pd.concat([state["ledger"].loc[~rows], pd.DataFrame(replacement)], ignore_index=True)


def _project_source(root: Path, source: dict[str, Any], directory: Path, months: dict[str, Any],
    request_sha256: str, lookback: pd.Timedelta) -> dict[str, Any]:
    sinks: list[_CompactSink] = []
    try:
        return _project_source_impl(root, source, directory, months, request_sha256, lookback, sinks)
    finally:
        for sink in sinks:
            sink.close()


def _project_source_impl(root: Path, source: dict[str, Any], directory: Path, months: dict[str, Any],
    request_sha256: str, lookback: pd.Timedelta, sinks: list[_CompactSink]) -> dict[str, Any]:
    directory.mkdir(parents=True)
    ledger = source["ledger"]
    starts = pd.to_datetime(ledger.requested_start_utc, utc=True)
    ends = pd.to_datetime(ledger.requested_end_utc, utc=True)
    states: dict[str, dict[str, Any]] = {}
    for month, record in months.items():
        lower = max(FIRST, pd.Timestamp(record["first_decision_utc"]) - lookback)
        upper = pd.Timestamp(record["last_decision_utc"])
        selected = ledger.loc[starts.le(upper) & ends.gt(lower)].copy()
        for column in ("requested_start_utc", "requested_end_utc", "row_count", "status"):
            selected[f"preparation_original_{column}"] = selected[column]
        selected["requested_start_utc"] = pd.to_datetime(selected.requested_start_utc, utc=True).clip(lower=lower)
        selected["requested_end_utc"] = pd.to_datetime(selected.requested_end_utc, utc=True).clip(upper=upper)
        selected = selected.loc[selected.requested_end_utc.gt(selected.requested_start_utc)].copy()
        unknown = set(selected.loc[~selected.status.isin(["observed", "observed_empty"]), "security_id"].astype(str))
        sink = _CompactSink(directory / month, request_sha256)
        sinks.append(sink)
        states[month] = {"start": lower, "end": upper, "ledger": selected, "blind": source["blindspots"] | unknown,
            "sink": sink, "logical_chunks": [], "unavailable_score_tails": [],
            "decision_cutoffs": sorted({pd.Timestamp(value) for value in
                [record["first_decision_utc"], record["last_decision_utc"], *record.get("decision_cutoffs_utc", [])]})}
    original_pins: dict[str, str] = {}
    projected_rows = 0
    numeric_chunks = 0
    for chunk in source["records"]["collection"]:
        _guard()
        if not re.fullmatch(r"[A-Za-z0-9_.-]+", chunk):
            raise DataReadinessError("monthly preparation source chunk name is invalid")
        events, relations, scores, child_pins, availability = _chunk(source, chunk)
        numeric_chunks += 1
        original_pins.update(child_pins)
        for state in states.values():
            selected_events, selected_relations, selected_scores, _pending = _select(
                events, relations, scores, state["start"], state["end"])
            rows = state["ledger"].chunk_id.astype(str).eq(chunk)
            if rows.any():
                state["ledger"].loc[rows, "row_count"] = len(selected_events)
                _pending_coverage_intervals(state, chunk, availability, lookback)
            if selected_events.empty:
                continue
            state["sink"].append((selected_events, selected_relations, selected_scores), chunk, child_pins)
            projected_rows += len(selected_events)
        del events, relations, scores, availability
    for path, digest in original_pins.items():
        _guard()
        _pin(Path(path), digest)
    _write_json(directory / "_source_children.json", original_pins)
    views: dict[str, Any] = {}
    for month, state in states.items():
        _guard()
        view = directory / month
        sink = state["sink"]
        sink.finish()
        state["ledger"] = state["ledger"].loc[state["ledger"].requested_end_utc.gt(state["ledger"].requested_start_utc)].copy()
        if state["ledger"].empty and not sink.records["events"]:
            continue
        coverage = _write_frame(state["ledger"], view / "collection/source_collections.parquet", "source_collections",
            {"monthly_preparation_request_sha256": request_sha256,
                "original_source_ledger_sha256": source["manifests"]["collection"]["source_collections_sha256"]})
        shared = {"schema": SCHEMA, "request_sha256": request_sha256, "status": "complete", "failed_chunks": {},
            "production_ready": False, "derivation_only": True, "scope_policy": ATTRIBUTION_SCOPE_POLICY,
            "coverage_blindspot_security_ids": sorted(state["blind"]), "excluded_security_ids": [], "source_coverage_admitted": False}
        for name, key in (("collection", "events"), ("attribution", "relations"), ("sentiment", "scores")):
            manifest = {**shared, "artifacts": sink.records[key], "total_rows": sum(row["rows"] for row in sink.records[key]),
                "verified_logical_chunk_ids": sorted(state["logical_chunks"]), "physical_layout": "bounded_compact_batches_v1"}
            if name == "collection":
                manifest.update(source_collections_path=coverage["path"], source_collections_sha256=coverage["sha256"])
            _write_json(view / name / "_manifest.json", manifest)
        _write_json(view / "collection/_audit.json", {**shared, "passed": True,
            "additional_coverage_unknown_security_ids": sorted(state["blind"] - source["blindspots"]),
            "unavailable_score_tails": state["unavailable_score_tails"],
            "audit_scope": "bounded_verified_source_projection_not_provider_coverage_admission"})
        views[month] = str(view.relative_to(root).as_posix())
    inventory = {p.relative_to(directory).as_posix(): file_sha256(p) for p in directory.rglob("*") if p.is_file()}
    manifest = {"schema": SCHEMA, "request_sha256": request_sha256, "views": views, "inventory": inventory,
        "numeric_source_chunks_read": numeric_chunks, "projected_source_rows": projected_rows,
        "physical_files": len(inventory) + 1, "physical_batches": sum(sink.batch for sink in sinks),
        "parent_pins": source["parent_pins"], "production_ready": False}
    _write_json(directory / "_manifest.json", manifest)
    return {"directory": str(directory.relative_to(root).as_posix()), "manifest_sha256": file_sha256(directory / "_manifest.json")}


def _verified_projection(root: Path, record: dict[str, Any], request_sha256: str) -> dict[str, Any]:
    directory = _path(root, record["directory"])
    _pin(directory / "_manifest.json", record["manifest_sha256"])
    manifest = _object(directory / "_manifest.json")
    if (manifest.get("schema") != SCHEMA or manifest.get("request_sha256") != request_sha256
            or manifest.get("production_ready") is not False):
        raise DataReadinessError("monthly source projection request differs")
    actual = {p.relative_to(directory).as_posix() for p in directory.rglob("*") if p.is_file()}
    if actual != {*manifest["inventory"], "_manifest.json"}:
        raise DataReadinessError("monthly source projection file inventory differs")
    for name, digest in manifest["inventory"].items():
        _guard()
        _pin(_path(directory, name), digest)
    _check_sources(root, manifest["parent_pins"])
    _check_sources(root, _object(directory / "_source_children.json"))
    return manifest


def _build_compact_lineage(view: Path, decisions_path: Path, policy_path: Path, target: Path) -> None:
    """Reuse strict lineage algorithms with independently proven logical coverage IDs.

    Physical batches are not provider requests. Never substitute their IDs for the
    original coverage intervals or infer coverage from the presence of batch rows.
    The caller has verified the external projection and every original child pin.
    """
    from market_predictor.swing import catalyst_lineage as engine

    manifests = {name: _object(view / name / "_manifest.json") for name in ("collection", "attribution", "sentiment")}
    audit = _object(view / "collection/_audit.json")
    blind = engine.validate_observed_article_scope(audit, manifests["attribution"], manifests["sentiment"])
    records = {name: _records(value) for name, value in manifests.items()}
    if not (set(records["collection"]) == set(records["attribution"]) == set(records["sentiment"])):
        raise DataReadinessError("compact lineage physical inventories differ")
    logical = set(manifests["collection"]["verified_logical_chunk_ids"])
    if any(set(value["verified_logical_chunk_ids"]) != logical for value in manifests.values()):
        raise DataReadinessError("compact lineage logical inventories differ")
    decisions, decision_manifest = load_canonical_artifact(decisions_path, expected_type="decisions", allow_research=True)
    ledger, ledger_manifest = load_canonical_artifact(view / "collection/source_collections.parquet",
        expected_type="source_collections", allow_research=True)
    policy = engine.load_catalyst_lineage_policy(policy_path)
    request = {"schema": engine.CATALYST_LINEAGE_REQUEST_SCHEMA,
        **{f"{name}_manifest_sha256": file_sha256(view / name / "_manifest.json") for name in manifests},
        "collection_audit_sha256": file_sha256(view / "collection/_audit.json"),
        "decisions_sha256": decision_manifest["artifact_sha256"], "source_collections_sha256": ledger_manifest["artifact_sha256"],
        "policy_sha256": file_sha256(policy_path), "excluded_security_ids": [], "scope_policy": ATTRIBUTION_SCOPE_POLICY,
        "coverage_blindspot_security_ids": sorted(blind), "source_coverage_admitted": False, "production_ready": False}
    request_hash = json_sha256(request)
    target.mkdir(parents=True, exist_ok=True)
    engine._write_or_validate_request(target / "_request.json", request, request_hash)
    coverage = engine._coverage_frame(ledger, excluded_security_ids=blind, relation_chunk_ids=logical, sentiment_chunk_ids=logical)
    coverage_path = target / "source_coverage.parquet"
    coverage_manifest = write_canonical_artifact(coverage, coverage_path, artifact_type="catalyst_source_coverage",
        audit=engine._coverage_audit(coverage), inputs={"catalyst_lineage_request_sha256": request_hash,
            "source_collections_sha256": str(ledger_manifest["artifact_sha256"])}, production_ready=False)
    observed: list[dict[str, Any]] = []
    channel_counts = dict.fromkeys(sorted(engine._SUPPORTED_CHANNELS), 0)
    status_counts: dict[str, int] = {}
    relation_ids: set[str] = set()
    source_ids: set[str] = set()
    decision_indices = {str(key): value for key, value in decisions.groupby("security_id", sort=False).indices.items()}
    for chunk in sorted(records["collection"]):
        _guard()
        assert_memory_budget(hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib, stage="compact catalyst lineage")
        source_record, relation_record, score_record = (records[name][chunk] for name in manifests)
        event_path, assignment_path = target / "events" / f"{chunk}.parquet", target / "assignments" / f"{chunk}.parquet"
        existing = engine._load_existing_chunk(event_target=event_path, assignment_target=assignment_path,
            request_sha256=request_hash, source_record=source_record, relation_record=relation_record,
            sentiment_record=score_record, decisions=decisions, decision_indices=decision_indices, policy=policy)
        if existing is not None:
            record, event_frame, assignments = existing
        else:
            events, em = load_canonical_artifact(Path(source_record["path"]), expected_type="events", allow_research=True)
            relations, rm = load_canonical_artifact(Path(relation_record["path"]),
                expected_type="event_security_relations", allow_research=True)
            scores, sm = load_canonical_artifact(Path(score_record["path"]), expected_type="event_sentiment_research", allow_research=True)
            engine._verify_chunk_lineage(chunk_id=chunk, source_record=source_record, relation_record=relation_record,
                sentiment_record=score_record, source_manifest=em, relation_manifest=rm, sentiment_manifest=sm,
                source_events=events, relations=relations, sentiments=scores)
            for frame in (events, relations, scores):
                expected = frame.preparation_original_chunk_id.map(
                    {key: json_sha256(value) for key, value in source_record["original_source_children"].items()})
                if expected.isna().any() or not expected.eq(frame.preparation_source_child_set_sha256).all():
                    raise DataReadinessError("compact batch row-to-original-child binding differs")
            event_frame = engine._join_catalyst_events(relations, scores, policy=policy)
            direct = event_frame.loc[event_frame.training_eligible.astype(bool)].copy()
            decision_part = decisions.loc[decisions.security_id.isin(direct.security_id)]
            assignments = build_event_assignments(decision_part, direct, windows=policy.assignment_windows)
            integrity = assignment_integrity_summary(decision_part, direct, assignments, windows=policy.assignment_windows)
            inputs = engine._chunk_inputs(request_hash, source_record, relation_record, score_record, decision_manifest)
            event_manifest = write_canonical_artifact(event_frame, event_path, artifact_type="catalyst_events",
                audit=engine._catalyst_event_audit(event_frame, policy), inputs=inputs, production_ready=False)
            assignment_manifest = write_canonical_artifact(assignments, assignment_path,
                artifact_type="catalyst_event_assignments", audit=engine._assignment_audit(assignments, integrity),
                inputs={**inputs, "catalyst_events_sha256": str(event_manifest["artifact_sha256"]),
                    "assignment_sha256": reconciliation_sha256(assignments)}, production_ready=False)
            record = engine._chunk_record(chunk_id=chunk, event_path=event_path, assignment_path=assignment_path,
                event_frame=event_frame, assignments=assignments, event_manifest=event_manifest,
                assignment_manifest=assignment_manifest, source_record=source_record, relation_record=relation_record,
                sentiment_record=score_record)
            del events, relations, scores, direct, decision_part
        engine._accumulate_unique_ids(relation_ids, event_frame.relation_id, "relation", chunk)
        engine._accumulate_unique_ids(source_ids, event_frame.source_event_id, "related source event", chunk, allow_repeated=True)
        for channel, count in event_frame.relation_channel.value_counts().items():
            channel_counts[str(channel)] += int(count)
        for status, count in assignments.status.value_counts().items():
            status_counts[str(status)] = status_counts.get(str(status), 0) + int(count)
        observed.append(record)
        del event_frame, assignments
        release_process_memory()
    inventory = engine._feature_inventory(policy, request_sha256=request_hash, channel_counts=channel_counts,
        coverage=coverage, event_records=observed)
    inventory_path = target / "feature_inventory.json"
    _write_json(inventory_path, inventory)
    result = {"schema": engine.CATALYST_LINEAGE_MANIFEST_SCHEMA, "request_sha256": request_hash, "status": "complete",
        "requested_chunks": len(observed), "observed_chunks": len(observed), "skipped_chunks": 0, "failed_chunks": {},
        "excluded_security_ids": [], "scope_policy": ATTRIBUTION_SCOPE_POLICY, "coverage_blindspot_security_ids": sorted(blind),
        "source_coverage_admitted": False, "source_event_rows": manifests["sentiment"]["total_rows"],
        "related_source_events": len(source_ids), "relation_rows": sum(row["event_rows"] for row in observed),
        "training_eligible_rows": sum(row["training_eligible_rows"] for row in observed), "channel_counts": channel_counts,
        "assignment_rows": sum(row["assignment_rows"] for row in observed), "assignment_status_counts": dict(sorted(status_counts.items())),
        "coverage": {"path": str(coverage_path.resolve()), "sha256": coverage_manifest["artifact_sha256"], "rows": len(coverage),
            "states": {str(key): int(value) for key, value in coverage.coverage_state.value_counts().sort_index().items()}},
        "feature_inventory": {"path": str(inventory_path.resolve()), "sha256": file_sha256(inventory_path)},
        "artifacts": observed, "lineage_sha256": engine._lineage_sha256(request, coverage_manifest, observed, inventory),
        "memory": memory_audit(hard_budget_gib=policy.maximum_process_memory_gib,
            headroom_gib=policy.memory_guard_headroom_gib).to_record(),
        "completed_at_utc": datetime.now(UTC).isoformat(), "production_ready": False}
    engine._atomic_json(target / "_status.json", result)
    engine._atomic_json(target / "_manifest.json", result)


def prepare_initial_fit_monthly_lineages(*, root: Path, config_path: Path, expected_config_sha256: str,
    output_directory: Path, expected_existing_manifest_sha256: str | None = None,
    estimate_only: bool = False,
) -> dict[str, Any]:
    root = root.resolve()
    config_path = _path(root, str(config_path))
    _pin(config_path, expected_config_sha256)
    config = _object(config_path)
    if set(config) != {"schema", "decisions", "policy", "sources"} or config["schema"] != SCHEMA:
        raise DataReadinessError("monthly lineage preparation config differs")
    if not isinstance(config["sources"], dict) or not config["sources"]:
        raise DataReadinessError("monthly lineage preparation requires its pinned source generations")
    if any(not re.fullmatch(r"[A-Za-z0-9_-]+", name) for name in config["sources"]):
        raise DataReadinessError("monthly lineage preparation source generation name is invalid")
    pending: Path | None = None
    try:
        with _decision_projection(root, config["decisions"]) as projection:
            result = _prepare_or_estimate(root, config_path, expected_config_sha256, config, projection,
                _path(root, str(output_directory)), expected_existing_manifest_sha256, estimate_only)
            if "_pending_manifest" in result:
                pending = Path(result.pop("_pending_manifest"))
        if pending is not None:
            pending.rename(pending.parent / "_manifest.json")
        return result
    finally:
        if pending is not None:
            pending.unlink(missing_ok=True)


def _prepare_or_estimate(root: Path, config_path: Path, expected_config_sha256: str, config: dict[str, Any],
    projection: Any, output: Path, existing_pin: str | None, estimate_only: bool) -> dict[str, Any]:
        _guard()
        if estimate_only:
            from market_predictor.swing.catalyst_lineage import load_catalyst_lineage_policy

            policy_path = _path(root, config["policy"]["path"])
            _pin(policy_path, config["policy"]["sha256"])
            lookback = max(load_catalyst_lineage_policy(policy_path).assignment_windows.values())
            months = {}
            for month, frame in projection.partitions:
                _guard()
                times = pd.to_datetime(frame.decision_time_utc, utc=True)
                months[month] = {"first_decision_utc": times.min().isoformat(), "last_decision_utc": times.max().isoformat()}
            return {"status": "metadata_only_estimate", "cohort_sha256": projection.cohort_sha256,
                "sources": {name: estimate_monthly_projection_files(_source(root, spec)["ledger"], months, lookback)
                    for name, spec in config["sources"].items()}}
        return _prepare(root, config_path, expected_config_sha256, config, projection,
            output, existing_pin)


def _prepare(root: Path, config_path: Path, digest: str, config: dict[str, Any], projection: Any,
    output: Path, existing_pin: str | None,
) -> dict[str, Any]:
    from market_predictor.swing.catalyst_lineage import (
        load_catalyst_lineage_policy,
        verify_completed_catalyst_lineage,
    )

    policy_path = _path(root, config["policy"]["path"])
    _pin(policy_path, config["policy"]["sha256"])
    policy = load_catalyst_lineage_policy(policy_path)
    lookback = max(policy.assignment_windows.values())
    if lookback > pd.Timedelta(days=31):
        raise DataReadinessError("monthly preparation requires a bounded at-most-31-day lookback")
    request = {"config": config, "config_sha256": digest, "cohort_sha256": projection.cohort_sha256,
        "rows": projection.expected_rows, "decision_source_files": dict(projection.source_files),
        "implementation_identity": _implementation_identity()}
    request_hash = json_sha256(request)
    state: dict[str, Any] = {"schema": SCHEMA, "request": request, "request_sha256": request_hash,
        "months": {}, "sources": {}, "lineages": {}}
    completed = False
    if output.exists():
        if existing_pin is None:
            raise DataReadinessError("monthly preparation resume requires an external manifest/checkpoint pin")
        completed = (output / "_manifest.json").exists()
        path = output / ("_manifest.json" if completed else "_checkpoint.json")
        _pin(path, existing_pin)
        state = _object(path)
        if state.get("schema") != SCHEMA or state.get("request") != request or state.get("request_sha256") != request_hash:
            raise DataReadinessError("monthly preparation resume request differs")
        if completed and (state.get("status") != "complete_research_only" or set(state["sources"]) != set(config["sources"])):
            raise DataReadinessError("completed monthly preparation inventory differs")
    else:
        if existing_pin is not None:
            raise DataReadinessError("monthly preparation resume directory is absent")
        output.mkdir(parents=True)
    request_path = output / "_request.json"
    if request_path.exists():
        if _object(request_path) != request:
            raise DataReadinessError("monthly preparation request file differs")
    elif completed:
        raise DataReadinessError("completed monthly preparation lacks its request")
    else:
        _write_json(request_path, request)

    def save() -> None:
        def report(item: dict[str, Any]) -> None:
            item["canonical_decision_months"] = item.pop("completed_months")
            item["completed_source_generations"] = sorted(state["sources"])
            item["completed_monthly_lineages"] = sorted(state["lineages"])
            print(json.dumps(item, sort_keys=True), flush=True)
        _checkpoint(output, state, report)
    if not completed:
        save()
    seen: set[str] = set()
    for month, frame in projection.partitions:
        _guard()
        _month(month)
        if month in seen:
            raise DataReadinessError("corrected decision projection duplicates a month")
        seen.add(month)
        decisions = stamp_canonical_decision_ids(frame[DECISION_COLUMNS].copy())
        if not set(decisions.security_id).issubset(projection.retained_security_ids):
            raise DataReadinessError("monthly preparation decisions escape the frozen cohort")
        ids = json_sha256(sorted(decisions.decision_id.astype(str)))
        prior = state["months"].get(month)
        if prior is None:
            if completed:
                raise DataReadinessError("completed monthly preparation lacks a canonical month")
            target = output / "decisions" / f"{month}-{uuid4().hex}.parquet"
            record = _write_frame(decisions, target, "decisions", {"monthly_preparation_request_sha256": request_hash})
            record["path"] = target.relative_to(root).as_posix()
            record["manifest_sha256"] = file_sha256(manifest_path_for(target))
            times = pd.to_datetime(decisions.decision_time_utc, utc=True)
            prior = {"decisions": record, "rows": len(decisions), "decision_ids_sha256": ids,
                "first_decision_utc": times.min().isoformat(), "last_decision_utc": times.max().isoformat(),
                "decision_cutoffs_utc": [value.isoformat() for value in sorted(times.unique())], "lineages": []}
            state["months"][month] = prior
        actual = _decisions(root, month, prior["decisions"])
        if len(actual) != len(decisions) or prior["decision_ids_sha256"] != ids or set(actual.decision_id) != set(decisions.decision_id):
            raise DataReadinessError("monthly preparation corrected decision identity differs on replay")
        times = pd.to_datetime(decisions.decision_time_utc, utc=True)
        if prior["decision_cutoffs_utc"] != [value.isoformat() for value in sorted(times.unique())]:
            raise DataReadinessError("monthly preparation canonical cutoffs differ on replay")
        if not completed:
            save()
    if seen != set(state["months"]) or sum(row["rows"] for row in state["months"].values()) != projection.expected_rows:
        raise DataReadinessError("monthly preparation lost corrected canonical decisions")
    source_files = {str(_path(root, path).relative_to(root).as_posix()): pin for path, pin in projection.source_files.items()}
    source_files[config_path.relative_to(root).as_posix()] = digest
    source_files[request_path.relative_to(root).as_posix()] = file_sha256(request_path)
    source_files[policy_path.relative_to(root).as_posix()] = config["policy"]["sha256"]
    for name, spec in config["sources"].items():
        _guard()
        source = _source(root, spec)
        if name not in state["sources"]:
            print(json.dumps({"source": name, "physical_file_estimate":
                estimate_monthly_projection_files(source["ledger"], state["months"], lookback)}, sort_keys=True), flush=True)
            target = output / "sources" / f"{name}-{uuid4().hex}"
            state["sources"][name] = _project_source(root, source, target, state["months"], request_hash, lookback)
            save()
        prepared = _verified_projection(root, state["sources"][name], request_hash)
        generation_dir = _path(root, state["sources"][name]["directory"])
        source_files[(generation_dir / "_manifest.json").relative_to(root).as_posix()] = state["sources"][name]["manifest_sha256"]
        child_index = generation_dir / "_source_children.json"
        source_files[child_index.relative_to(root).as_posix()] = file_sha256(child_index)
        for month, view_path in prepared["views"].items():
            _guard()
            view = _path(root, view_path)
            paths = {"collection_manifest_sha256": view / "collection/_manifest.json",
                "collection_audit_sha256": view / "collection/_audit.json",
                "attribution_manifest_sha256": view / "attribution/_manifest.json",
                "sentiment_manifest_sha256": view / "sentiment/_manifest.json",
                "source_collections_sha256": view / "collection/source_collections.parquet", "policy_sha256": policy_path}
            pins = {path.relative_to(root).as_posix(): file_sha256(path) for path in paths.values()}
            source_files.update(pins)
            target = output / "lineages" / month / name
            if not (target / "_manifest.json").exists():
                if completed:
                    raise DataReadinessError("completed monthly preparation lacks a lineage")
                _build_compact_lineage(view, _path(root, state["months"][month]["decisions"]["path"]), policy_path, target)
            verified = verify_completed_catalyst_lineage(target)
            if (verified.request["decisions_sha256"] != state["months"][month]["decisions"]["sha256"]
                    or any(verified.request[key] != pins[path.relative_to(root).as_posix()] for key, path in paths.items())):
                raise DataReadinessError("monthly lineage does not reproduce its actual prepared source child pins")
            record = {"directory": target.relative_to(root).as_posix(), "manifest_sha256": verified.manifest_sha256,
                "source_paths": {key: path.relative_to(root).as_posix() for key, path in paths.items()}}
            key = f"{month}/{name}"
            if key in state["lineages"] and state["lineages"][key] != record:
                raise DataReadinessError("monthly prepared lineage changed after checkpoint")
            state["lineages"][key] = record
            if not completed:
                save()
    publication_months = {}
    for month, record in state["months"].items():
        lineages = [value for key, value in sorted(state["lineages"].items()) if key.startswith(month + "/")]
        if not lineages:
            raise DataReadinessError("monthly preparation has no source coverage lineage for a canonical month")
        publication_months[month] = {"decisions": record["decisions"], "rows": record["rows"],
            "decision_ids_sha256": record["decision_ids_sha256"], "lineages": lineages}
    publication = {"schema": PUBLICATION_SCHEMA, "cohort_sha256": projection.cohort_sha256, "rows": projection.expected_rows,
        "source_files": source_files, "months": publication_months}
    publication_path = output / "monthly-publication.json"
    _check_sources(root, source_files)
    for source_record in state["sources"].values():
        _verified_projection(root, source_record, request_hash)
    if request["implementation_identity"] != _implementation_identity():
        raise DataReadinessError("monthly preparation implementation changed during publication")
    if completed:
        _pin(publication_path, state["publication_config_sha256"])
        if _object(publication_path) != publication:
            raise DataReadinessError("completed monthly publication inputs differ from replay")
    else:
        if publication_path.exists():
            if _object(publication_path) != publication:
                raise DataReadinessError("uncommitted monthly publication inputs differ from verified reconstruction")
        else:
            _write_json(publication_path, publication)
        state.update(status="complete_research_only", publication_config_sha256=file_sha256(publication_path), source_files=source_files)
        pending = output / f".manifest-{uuid4().hex}.pending"
        _write_json(pending, state)
    return {"directory": str(output), "manifest_sha256": file_sha256(output / "_manifest.json" if completed else pending),
        **({"_pending_manifest": str(pending)} if not completed else {}),
        "publication_config": str(publication_path), "publication_config_sha256": state["publication_config_sha256"],
        "rows": projection.expected_rows, "months": len(publication_months), "sources": state["sources"]}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--expected-config-sha256", required=True)
    parser.add_argument("--out-dir", type=Path, required=True)
    parser.add_argument("--expected-existing-manifest-sha256")
    parser.add_argument("--estimate-only", action="store_true",
        help="Validate metadata and report file estimates without source projection")
    args = parser.parse_args()
    result = prepare_initial_fit_monthly_lineages(root=args.root, config_path=args.config,
        expected_config_sha256=args.expected_config_sha256, output_directory=args.out_dir,
        expected_existing_manifest_sha256=args.expected_existing_manifest_sha256, estimate_only=args.estimate_only)
    print(json.dumps(result, sort_keys=True))


if __name__ == "__main__":
    main()
