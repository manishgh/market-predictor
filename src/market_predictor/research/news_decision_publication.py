"""Immutable monthly publication of retained-parent plus observed news features.

This research publisher owns one heavy-job lease. It never reconstructs prices,
parent predictors, targets or issuer world events, and grants no admission.
"""
from __future__ import annotations

import json
import math
from collections.abc import Callable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Protocol, cast

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.news_query_scope import SourcePin
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research import news_decision_features as features
from market_predictor.research import news_parent_inputs as parent_inputs
from market_predictor.research.news_source_links import SourceLinkedNewsSlice, open_news_source_link_index
from market_predictor.swing.datasets.return_relationship_integrity import check_files, pins, read_object
from market_predictor.swing.features.research_partition import ResearchFeaturePartition

SCHEMA = "market_predictor.news_decision_publication"
CONFIG_SCHEMA = "market_predictor.news_decision_publication_config"
PROFILE = "technical_relationships_news"
ARTIFACT_TYPE = "swing_news_decision_features"
CLOSED = {"training_eligible": False, "promotion_eligible": False, "serving_eligible": False,
          "source_coverage_admitted": False, "world_actor_verified": False}
NEWS_CLOCKS = {name: f"available_at_{name}" for name in features.NEWS_COLUMNS}
NEWS_REASONS = {name: f"missing_reason_{name}" for name in features.NEWS_COLUMNS}
DIAGNOSTICS = ("news_projection_status", "news_suppressed_story_count", "news_selected_version_ids_json", "news_uncertainty_count")
ADDITIONS = (*features.NEWS_COLUMNS, *NEWS_CLOCKS.values(), *NEWS_REASONS.values(), *DIAGNOSTICS)
IMPLEMENTATION_PATHS = (
    "research/news_decision_publication.py", "research/news_decision_features.py", "research/news_source_links.py",
    "canonical/store.py", "canonical/contracts.py", "canonical/audits.py", "evidence/hashing.py", "evidence/io.py",
    "heavy_jobs.py", "locking.py", "swing/features/research_partition.py", "catalysts/issuer_events/news_query_scope.py",
)


class NewsSliceReader(Protocol):
    def for_period(self, security_id: str, start_utc: datetime, end_utc: datetime, *,
                   max_versions: int = 100000) -> SourceLinkedNewsSlice: ...


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def _implementation(root: Path) -> dict[str, str]:
    package = Path(__file__).resolve().parents[1]
    return {(package / name).relative_to(root).as_posix(): file_sha256(package / name) for name in IMPLEMENTATION_PATHS}


def _time(value: Any) -> datetime:
    stamp = pd.Timestamp(value)
    _require(not pd.isna(stamp) and stamp.tzinfo is not None, "news publication requires aware source/decision clocks")
    return cast(datetime, stamp.tz_convert("UTC").to_pydatetime())


def _decisions(frame: pd.DataFrame, clocks: Mapping[str, str]) -> tuple[features.NewsDecision, ...]:
    result = []
    for row in frame.to_dict("records"):
        technical: dict[str, features.TechnicalValue] = {}
        for name in features.TECHNICAL_COLUMNS:
            value = row[name]
            clock = row[clocks[name]]
            technical[name] = features.TechnicalValue(None if pd.isna(value) else float(value),
                                                       None if pd.isna(clock) else _time(clock), "historical_proxy")
        result.append(features.NewsDecision(row["decision_id"], row["security_id"], _time(row["decision_time_utc"]), **technical))
    return tuple(result)


def _reason(name: str) -> str:
    if name in features.NEWS_COLUMNS[-4:]:
        return "required_news_or_technical_operand_unavailable"
    if "finbert" in name and "coverage" not in name:
        return "no_available_sentiment_in_3d"
    if any(name == f"news_direct_{category}_{direction}_share_3d" for category in features.CATEGORIES
           for direction in features.DIRECTIONS):
        return "no_readable_category_documents_in_3d"
    if "present_" in name:
        return "no_readable_window_or_verified_empty_evidence"
    return "no_readable_linked_documents_in_3d"


def _parity(baseline: pd.DataFrame, result: pd.DataFrame) -> None:
    _require(list(result.columns[:len(baseline.columns)]) == list(baseline.columns), "news publication changed inherited column order")
    inherited = [name for name in baseline if name != "feature_profile"]
    try:
        pd.testing.assert_frame_equal(baseline[inherited], result[inherited], check_exact=True)
    except AssertionError as exc:
        raise DataReadinessError("news publication changed inherited values, dtypes, index or population") from exc
    _require(result.feature_profile.eq(PROFILE).all(), "news publication profile field differs")


def build_news_month(*, parent: ResearchFeaturePartition, reader: NewsSliceReader) -> ResearchFeaturePartition:
    """Append the shared kernel's columns without changing any inherited economics."""
    parent_inputs._guard()
    baseline = parent.rows
    _require(len(parent.model_columns) == 124 and len(set(parent.model_columns)) == 124 and baseline.columns.is_unique
             and not baseline.empty and not baseline.decision_id.duplicated().any(), "news publication requires exact unique parent")
    _require(not set(ADDITIONS).intersection(baseline.columns), "news publication additions already exist")
    _require(set(features.TECHNICAL_COLUMNS).issubset(parent.model_columns) and
             set(parent.model_columns).issubset(parent.availability_columns), "parent technical contract is incomplete")
    computed: dict[str, dict[str, Any]] = {}
    for security, group in baseline.groupby("security_id", sort=True):
        parent_inputs._guard()
        _require(isinstance(security, str) and bool(security), "parent stock identity is invalid")
        decisions = _decisions(group, parent.availability_columns)
        sliced = reader.for_period(security, min(row.decision_time_utc for row in decisions),
                                   max(row.decision_time_utc for row in decisions))
        _require(all(link.security_id == security for link in sliced.links) and
                 all(item.security_id == security for item in sliced.coverage), "news source slice belongs to another stock")
        _require(sliced.source_coverage_admitted is False, "research news slice cannot assert source admission")
        usable_decisions = []
        for decision in decisions:
            uncertainties = tuple(item for item in sliced.uncertainties if item.applies_at(decision.decision_time_utc))
            if uncertainties:
                computed[decision.decision_id] = {
                    **dict.fromkeys(features.NEWS_COLUMNS, np.nan), **dict.fromkeys(NEWS_CLOCKS.values(), None),
                    **dict.fromkeys(NEWS_REASONS.values(), "unresolved_source_history"),
                    "news_projection_status": "source_history_unresolved", "news_suppressed_story_count": 0,
                    "news_selected_version_ids_json": "[]", "news_uncertainty_count": len(uncertainties)}
            else:
                usable_decisions.append(decision)
        if not usable_decisions:
            continue
        results = features.aggregate_news_decisions(decisions=usable_decisions, versions=sliced.versions, links=sliced.links,
                                                     coverage=sliced.coverage, purpose="research_proxy")
        _require(len(results) == len(usable_decisions), "news kernel lost decisions")
        for decision, result in zip(usable_decisions, results, strict=True):
            _require((result.decision_id, result.security_id, result.decision_time_utc) ==
                     (decision.decision_id, decision.security_id, decision.decision_time_utc) and
                     result.columns == features.NEWS_COLUMNS and not any((result.source_admission, result.training_eligible,
                                                                         result.serving_eligible, result.promotion_eligible)),
                     "news kernel changed decision ownership, schema or admission")
            row: dict[str, Any] = {"news_projection_status": "observed_retained_documents",
                                   "news_suppressed_story_count": result.suppressed_story_count,
                                   "news_selected_version_ids_json": json.dumps(result.selected_version_ids, separators=(",", ":")),
                                   "news_uncertainty_count": 0}
            for name, value, clock in zip(features.NEWS_COLUMNS, result.values, result.available_at_utc, strict=True):
                reason = _reason(name) if value is None else None
                if value is not None:
                    _require(math.isfinite(value) and clock is not None and clock <= decision.decision_time_utc,
                             "news kernel value/clock is invalid or future")
                    with np.errstate(over="ignore"):
                        numeric = np.float32(value)
                    if not np.isfinite(numeric):
                        value, clock, reason = None, None, "float32_overflow"
                row[name], row[NEWS_CLOCKS[name]], row[NEWS_REASONS[name]] = value, clock, reason
            computed[decision.decision_id] = row
    _require(set(computed) == set(baseline.decision_id), "news publication lost or invented parent decisions")
    extra = pd.DataFrame([computed[key] for key in baseline.decision_id], index=baseline.index)
    for name in features.NEWS_COLUMNS:
        extra[name] = extra[name].astype("float32")
        extra[NEWS_CLOCKS[name]] = pd.to_datetime(extra[NEWS_CLOCKS[name]], utc=True)
        extra[NEWS_REASONS[name]] = extra[NEWS_REASONS[name]].astype(pd.StringDtype(storage="python"))
    for name in ("news_projection_status", "news_selected_version_ids_json"):
        extra[name] = extra[name].astype(pd.StringDtype(storage="python"))
    for name in ("news_suppressed_story_count", "news_uncertainty_count"):
        extra[name] = extra[name].astype("int64")
    output = baseline.copy()
    output["feature_profile"] = PROFILE
    output = pd.concat([output, extra.loc[:, ADDITIONS]], axis=1)
    _parity(baseline, output)
    return ResearchFeaturePartition(output, (*parent.model_columns, *features.NEWS_COLUMNS),
                                    {**parent.availability_columns, **NEWS_CLOCKS},
                                    {"profile": PROFILE, "parent_exact_except_feature_profile": True, "rows": len(output), **CLOSED})


def _checkpoint(path: Path, state: dict[str, Any]) -> str:
    temporary = path.with_suffix(".partial")
    _require(not temporary.exists(), "unfinished news checkpoint exists")
    write_json_object(temporary, state)
    temporary.replace(path)
    return file_sha256(path)


def _publish_month(stage: Path, month: str, part: ResearchFeaturePartition, request_sha: str) -> dict[str, Any]:
    relative = f"{month}/{PROFILE}.parquet"
    path = inside(stage, relative)
    _require(not path.exists() and not manifest_path_for(path).exists(), "uncheckpointed news month exists")
    audit = CanonicalAuditReport(checks=(CanonicalAuditCheck(name="parent_preserved_shared_news_kernel", status="pass",
                                                           rows_checked=len(part.rows), failures=0,
                                                           detail="Exact parent parity except profile; current shared cue kernel"),))
    write_canonical_artifact(part.rows, path, artifact_type=ARTIFACT_TYPE, audit=audit,
                             inputs={"request_sha256": request_sha}, production_ready=False)
    # The new unpublished sidecar owns a location-independent relative path.
    # This avoids storing a private stage pathname in the final publication.
    sidecar = manifest_path_for(path)
    metadata = json.loads(sidecar.read_text(encoding="utf-8"))
    metadata["artifact_path"] = relative
    sidecar.write_text(json.dumps(metadata, sort_keys=True, indent=2), encoding="utf-8")
    path.with_name(path.name + ".lock").unlink()
    return {"path": relative, "sha256": file_sha256(path), "manifest_sha256": file_sha256(sidecar),
            "rows": len(part.rows), "decision_ids_sha256": json_sha256(sorted(part.rows.decision_id)),
            "model_columns": list(part.model_columns), "availability_columns": dict(part.availability_columns)}


def _child(stage: Path, record: dict[str, Any], request_sha: str, parent: pd.DataFrame,
           columns: tuple[str, ...], clocks: Mapping[str, str]) -> None:
    path = inside(stage, record["path"])
    check_files(stage, {record["path"]: record["sha256"],
                       manifest_path_for(path).relative_to(stage).as_posix(): record["manifest_sha256"]})
    frame, sidecar = load_canonical_artifact(path, expected_type=ARTIFACT_TYPE, allow_research=True)
    _require(sidecar.get("inputs") == {"request_sha256": request_sha} and sidecar.get("production_ready") is False
             and sidecar.get("artifact_path") == record["path"] and record["rows"] == len(frame)
             and record["model_columns"] == list(columns) and record["availability_columns"] == dict(clocks)
             and json_sha256(sorted(frame.decision_id)) == record["decision_ids_sha256"], "news child request/schema/population differs")
    _parity(parent, frame)
    _require(list(frame.columns[len(parent.columns):]) == list(ADDITIONS), "news child addition order differs")
    for name in features.NEWS_COLUMNS:
        values, available, reasons = frame[name], frame[NEWS_CLOCKS[name]], frame[NEWS_REASONS[name]]
        _require(str(values.dtype) == "float32" and isinstance(available.dtype, pd.DatetimeTZDtype)
                 and not np.isinf(values.to_numpy()).any() and values.isna().equals(reasons.notna())
                 and values.isna().equals(available.isna()) and not available.gt(frame.decision_time_utc).any(),
                 "news child dtype, missingness or cutoff differs")


def _inventory(stage: Path, records: Mapping[str, dict[str, Any]]) -> None:
    expected = {"_request.json", "_checkpoint.json"}
    for record in records.values():
        expected.update((record["path"], record["path"] + ".manifest.json"))
    actual = {path.relative_to(stage).as_posix() for path in stage.rglob("*") if path.is_file()}
    _require(actual == expected, "news publication contains unfinished or unowned artifacts")


def materialize_news_decisions(*, root: Path, config: SourcePin, output: Path,
                               expected_checkpoint_sha256: str | None = None,
                               maximum_months_this_run: int | None = None,
                               progress: Callable[[dict[str, Any]], None] | None = None) -> dict[str, Any]:
    """Publish all months, or return an externally pinnable private checkpoint."""
    root, output = root.resolve(), inside(root.resolve(), output)
    stage = output.with_name(f".{output.name}.building")
    _require(not output.exists(), "immutable news publication already exists")
    _require(maximum_months_this_run is None or type(maximum_months_this_run) is int and maximum_months_this_run > 0,
             "news month limit must be a positive integer")
    runtime = heavy_job_runtime_dir()
    with heavy_job_lease("publish-news-decisions", runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        parent_inputs._guard()
        policy = read_object(inside(root, config.path), config.sha256)
        _require(set(policy) == {"schema", "source_link_index"} and policy["schema"] == CONFIG_SCHEMA, "news publication config differs")
        link_pin = SourcePin.model_validate(policy["source_link_index"])
        local = _implementation(root)
        with (parent_inputs.verified_news_parent_inputs(root=root) as parent,
              open_news_source_link_index(root=root, index=link_pin) as reader):
            files = pins(root, parent.source_files, reader.source_files, {config.path: config.sha256})
            implementations = pins(root, local, parent.implementation_files, reader.implementation_files)
            columns = (*parent.model_columns, *features.NEWS_COLUMNS)
            clocks = {**parent.availability_columns, **NEWS_CLOCKS}
            request = {"schema": SCHEMA + "_request", "config": config.model_dump(mode="json"), "policy": policy,
                       "profile": PROFILE, "model_columns": list(columns), "availability_columns": dict(clocks),
                       "parent_publication": parent_inputs.PUBLICATION.model_dump(mode="json"),
                       "parent_saved_row_verification": parent_inputs.SAVED_ROW_RECEIPT.model_dump(mode="json"),
                       "original_snapshot_replay": parent_inputs.ORIGINAL_REPLAY.model_dump(mode="json"),
                       "parent_profile_sha256": parent.request["profile_sha256"], "cohort_sha256": parent.request["cohort_sha256"],
                       "rows": parent_inputs.EXPECTED_ROWS, "decision_ids_sha256": parent_inputs.EXPECTED_IDS,
                       "availability_semantics": "historical_proxy", "purpose": "research_proxy",
                       "source_files": files, "implementation_files": implementations,
                       "historical_implementation_files": parent.historical_implementation_files,
                       "allowed_parent_column_changes": ["feature_profile"],
                       "uncertainty_policy": "all_news_columns_null_for_applicable_unrepresentable_source_history", **CLOSED}
            request["profile_sha256"] = json_sha256({"profile": PROFILE, "columns": columns,
                                                     "kernel": implementations["src/market_predictor/research/news_decision_features.py"],
                                                     "parent_profile": request["parent_profile_sha256"],
                                                     "source_index": link_pin.model_dump(mode="json"),
                                                     "uncertainty_policy": request["uncertainty_policy"]})
            if stage.exists():
                _require(expected_checkpoint_sha256 is not None, "news resume requires external checkpoint SHA")
                assert expected_checkpoint_sha256 is not None
                state = read_object(stage / "_checkpoint.json", expected_checkpoint_sha256)
                checkpoint_sha = expected_checkpoint_sha256
                request_sha = state["request_sha256"]
                _require(read_object(stage / "_request.json", request_sha) == request and state.get("schema") == SCHEMA + "_checkpoint"
                         and all(state.get(key) is value for key, value in CLOSED.items()), "news resume source/code/request differs")
            else:
                _require(expected_checkpoint_sha256 is None, "news checkpoint supplied without private stage")
                stage.mkdir(parents=True)
                write_json_object(stage / "_request.json", request)
                request_sha = file_sha256(stage / "_request.json")
                state = {"schema": SCHEMA + "_checkpoint", "request_sha256": request_sha, "months": {}, **CLOSED}
                checkpoint_sha = _checkpoint(stage / "_checkpoint.json", state)
            records = state["months"]
            _require(list(records) == list(parent_inputs.MONTHS[:len(records)]),
                     "news checkpoint months are not exact chronological prefix")
            _inventory(stage, records)
            written = 0
            parent_manifest = parent.manifest
            for month in parent_inputs.MONTHS:
                parent_inputs._guard()
                if month not in records and maximum_months_this_run is not None and written >= maximum_months_this_run:
                    break
                baseline = parent.read_month(month)
                expected = parent_manifest["months"][month]
                if month not in records:
                    part = build_news_month(
                        parent=ResearchFeaturePartition(baseline, parent.model_columns, parent.availability_columns, {}), reader=reader)
                    records[month] = _publish_month(stage, month, part, request_sha)
                    del part
                    written += 1
                record = records[month]
                _require(record["path"] == f"{month}/{PROFILE}.parquet" and record["rows"] == expected["rows"]
                         and record["decision_ids_sha256"] == expected["decision_ids_sha256"],
                         "news month does not preserve parent ownership")
                _child(stage, record, request_sha, baseline, columns, clocks)
                checkpoint_sha = _checkpoint(stage / "_checkpoint.json", state)
                if progress is not None:
                    progress({"month": month, "completed_months": len(records), "rows": sum(item["rows"] for item in records.values())})
                del baseline
            parent.recheck()
        # Both verified input contexts have performed their final source checks.
        check_files(root, files)
        check_files(root, implementations)
        _require(read_object(stage / "_request.json", request_sha) == request, "news publication request changed")
        _inventory(stage, records)
        for record in records.values():
            check_files(stage, {record["path"]: record["sha256"], record["path"] + ".manifest.json": record["manifest_sha256"]})
        _require(read_object(stage / "_checkpoint.json", checkpoint_sha) == state, "news checkpoint changed during publication")
        rows = sum(item["rows"] for item in records.values())
        if len(records) != len(parent_inputs.MONTHS):
            return {"status": "partial_research_only", "rows": rows, "months": len(records),
                    "checkpoint_path": (stage / "_checkpoint.json").relative_to(root).as_posix(),
                    "checkpoint_sha256": checkpoint_sha, "request_sha256": request_sha, **CLOSED}
        _require(rows == parent_inputs.EXPECTED_ROWS, "news publication total population differs")
        manifest = {"schema": SCHEMA, "status": "complete_research_only", "request_sha256": request_sha,
                    "checkpoint_sha256": checkpoint_sha, "months": records, "rows": rows, "profile": PROFILE,
                    "profile_sha256": request["profile_sha256"], "model_columns": list(columns),
                    "availability_columns": dict(clocks), "decision_ids_sha256": parent_inputs.EXPECTED_IDS,
                    "source_files": files, "implementation_files": implementations, "exclusions_added": [], **CLOSED}
        parent_inputs._guard()
        write_json_object(stage / "_manifest.json", manifest)
        stage.rename(output)
        return {**manifest, "manifest_sha256": file_sha256(output / "_manifest.json")}
