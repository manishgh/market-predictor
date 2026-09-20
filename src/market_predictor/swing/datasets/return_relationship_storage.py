"""Immutable group/month artifacts and bounded baseline staging."""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as pds
import pyarrow.parquet as pq

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import file_sha256, load_canonical_artifact, manifest_path_for, write_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import inside, write_json_object
from market_predictor.swing.contracts.return_relationship_publication import ARTIFACT_TYPE
from market_predictor.swing.datasets.return_relationship_integrity import check_files, read_object
from market_predictor.swing.datasets.return_relationship_parent import VerifiedRelationshipParent, load_parent_month
from market_predictor.swing.datasets.return_relationship_rows import ADDITIONS, IDENTITY_COLUMNS
from market_predictor.swing.datasets.return_relationship_sources import CORRECTED_SYMBOLS

GROUP_TYPE = "swing_return_relationship_additions"


def publish_rows(output: Path, relative: str, frame: pd.DataFrame, request_sha256: str, *, group: bool = False,
) -> dict[str, Any]:
    path = inside(output, relative)
    if path.exists() or manifest_path_for(path).exists():
        raise DataReadinessError("relationship publisher refuses to overwrite an uncheckpointed artifact")
    if frame.empty or frame.decision_id.duplicated().any() or not frame.columns.is_unique:
        raise DataReadinessError("relationship publication requires nonempty unique rows and columns")
    audit = CanonicalAuditReport(checks=[CanonicalAuditCheck(name="unique_decision_population", status="pass",
        rows_checked=len(frame), failures=0, detail="Verified distinct complete decision rows before publication")])
    write_canonical_artifact(frame, path, artifact_type=GROUP_TYPE if group else ARTIFACT_TYPE,
        audit=audit, inputs={"request_sha256": request_sha256}, production_ready=False)
    return {"path": relative, "sha256": file_sha256(path), "manifest_sha256": file_sha256(manifest_path_for(path)),
        "rows": len(frame), "decision_ids_sha256": json_sha256(sorted(frame.decision_id))}


def verify_child(output: Path, record: dict[str, Any], request_sha256: str, *, group: bool = False,
) -> dict[str, str]:
    path = inside(output, record["path"])
    files = {record["path"]: record["sha256"], manifest_path_for(path).relative_to(output).as_posix(): record["manifest_sha256"]}
    check_files(output, files)
    child = read_object(manifest_path_for(path), record["manifest_sha256"])
    if (child.get("artifact_type") != (GROUP_TYPE if group else ARTIFACT_TYPE)
            or child.get("artifact_sha256") != record["sha256"] or child.get("rows") != record["rows"]
            or child.get("production_ready") is not False or child.get("inputs") != {"request_sha256": request_sha256}
            or not set(ADDITIONS).issubset(child["columns"])):
        raise DataReadinessError("relationship child schema or request binding differs")
    parquet: Any = pq
    physical = parquet.ParquetFile(path)
    if physical.metadata.num_rows != record["rows"] or physical.schema_arrow.names != child["columns"]:
        raise DataReadinessError("relationship child physical schema or row count differs")
    return files


def stage_baseline(output: Path, parent: VerifiedRelationshipParent, inventory: dict[str, dict[str, Any]],
    request_sha256: str, guard: Callable[[], None],
    expected: dict[str, Any] | None = None,
) -> dict[str, Any]:
    directory = output / "_baseline_stage"
    manifest_path = directory / "_manifest.json"
    if directory.exists():
        if expected is None or expected.get("path") != "_baseline_stage/_manifest.json":
            raise DataReadinessError("relationship baseline staging lacks independently checkpointed identity")
        manifest = read_object(manifest_path, expected["sha256"])
        if manifest.get("request_sha256") != request_sha256 or manifest.get("inventory_sha256") != json_sha256(inventory):
            raise DataReadinessError("relationship baseline staging request or inventory differs")
        check_files(directory, manifest["files"])
        return expected
    if expected is not None:
        raise DataReadinessError("relationship checkpointed baseline stage is missing")
    directory.mkdir()
    writers: dict[int, Any] = {}
    arrow: Any = pa
    parquet: Any = pq
    columns = list(dict.fromkeys((*IDENTITY_COLUMNS, *parent.model_columns,
        *parent.availability_columns.values(), "feature_profile")))
    try:
        for month in sorted(parent.manifest["months"]):
            guard()
            frame = load_parent_month(parent, month, columns)
            group_keys = [json_sha256([str(identity), CORRECTED_SYMBOLS.get(str(identity), str(ticker))])
                for identity, ticker in zip(frame.security_id, frame.parent_ticker, strict=True)]
            if not set(group_keys).issubset(inventory):
                raise DataReadinessError("relationship staging encountered an unbound source group")
            frame["_source_group_key"] = group_keys
            buckets = [int(key[:2], 16) % 8 for key in group_keys]
            for bucket, rows in frame.groupby(buckets, sort=True):
                guard()
                rows = rows.sort_values(["_source_group_key", "decision_time_utc"], kind="stable")
                table = arrow.Table.from_pandas(rows, preserve_index=False)
                if bucket not in writers:
                    writers[bucket] = parquet.ParquetWriter(directory / f"bucket-{bucket}.parquet", table.schema)
                writers[bucket].write_table(table)
    finally:
        for writer in writers.values():
            writer.close()
    files = {f"bucket-{bucket}.parquet": file_sha256(directory / f"bucket-{bucket}.parquet") for bucket in writers}
    write_json_object(manifest_path, {"request_sha256": request_sha256, "inventory_sha256": json_sha256(inventory),
        "files": files, "bucket_count": 8})
    return {"path": "_baseline_stage/_manifest.json", "sha256": file_sha256(manifest_path)}


def staged_baseline(output: Path, key: str, stage: dict[str, Any], item: dict[str, Any]) -> pd.DataFrame:
    manifest = read_object(inside(output, stage["path"]), stage["sha256"])
    name = f"bucket-{int(key[:2], 16) % 8}.parquet"
    path = output / "_baseline_stage" / name
    digest = manifest["files"][name]
    if file_sha256(path) != digest:
        raise DataReadinessError("relationship staged baseline changed")
    frame = pd.read_parquet(path, filters=[("_source_group_key", "=", key)]).drop(columns="_source_group_key")
    if file_sha256(path) != digest:
        raise DataReadinessError("relationship staged baseline changed during read")
    if (len(frame) != item["rows"] or frame.decision_id.duplicated().any()
            or json_sha256(sorted(frame.decision_id)) != item["decision_ids_sha256"]):
        raise DataReadinessError("relationship baseline staging lost or duplicated decisions")
    return frame.reset_index(drop=True)


def month_additions(output: Path, groups: dict[str, dict[str, Any]], month: str) -> pd.DataFrame:
    arrow: Any = pds
    paths = [str(inside(output, record["path"])) for record in groups.values()]
    first = pd.Timestamp(month + "-01").date()
    last = (pd.Timestamp(month + "-01") + pd.offsets.MonthEnd(0)).date()
    dataset = arrow.dataset(paths, format="parquet")
    frame: pd.DataFrame = dataset.to_table(columns=[*IDENTITY_COLUMNS, *ADDITIONS],
        filter=(arrow.field("session_date_et") >= first) & (arrow.field("session_date_et") <= last),
        use_threads=False).to_pandas()
    return frame


def load_group(output: Path, record: dict[str, Any], request_sha256: str) -> pd.DataFrame:
    verify_child(output, record, request_sha256, group=True)
    frame, _ = load_canonical_artifact(inside(output, record["path"]), expected_type=GROUP_TYPE, allow_research=True)
    if frame.decision_id.duplicated().any() or json_sha256(sorted(frame.decision_id)) != record["decision_ids_sha256"]:
        raise DataReadinessError("relationship staged group identity differs")
    return frame
