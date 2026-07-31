"""Resumable, immutable materialization of the ER1A swing ranking panel."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_features import (
    SWING_FEATURE_PANEL_SCHEMA,
    SWING_FEATURE_PROFILE,
    build_swing_feature_rows,
    finalize_swing_feature_panel,
)
from market_predictor.edge_rebuild.swing_setups import (
    iter_security_batches,
    load_collection_artifact_index,
    load_daily_bars,
    load_security_batch_bars,
)
from market_predictor.locking import file_lock
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

SWING_MATERIALIZATION_REQUEST_SCHEMA: Final = (
    "edge_rebuild.swing_panel_materialization_request.v1"
)
SWING_MATERIALIZATION_MANIFEST_SCHEMA: Final = (
    "edge_rebuild.swing_panel_materialization.v1"
)
SWING_MATERIALIZATION_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.swing_panel_materialization_authority.v1"
)
SWING_STAGE_ONE_SHARD_SCHEMA: Final = "edge_rebuild.swing_panel_stage_one_shard.v1"


def materialize_swing_feature_panel(
    *,
    collection_dir: Path,
    memberships_path: Path,
    contract: StrategyContract,
    output_dir: Path,
    securities_per_shard: int = 32,
    maximum_stage_one_shards_this_run: int | None = None,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> dict[str, Any]:
    """Build bounded row shards, then finalize one complete population pass.

    A completed ``final`` directory is immutable. An interrupted stage-one run
    resumes only when its request and every existing shard still verify.
    """

    if securities_per_shard < 1:
        raise ValueError("securities_per_shard must be positive")
    if maximum_stage_one_shards_this_run is not None and maximum_stage_one_shards_this_run < 1:
        raise ValueError("maximum_stage_one_shards_this_run must be positive")
    budget = (memory_budget_gib, memory_headroom_gib)
    output_dir.parent.mkdir(parents=True, exist_ok=True)
    with file_lock(output_dir.parent / f".{output_dir.name}.materialization"):
        memberships, memberships_manifest = load_canonical_artifact(
            memberships_path,
            expected_type="memberships",
            allow_research=True,
        )
        security_ids = sorted(memberships["security_id"].astype(str).unique())
        if not security_ids:
            raise DataReadinessError("swing materialization has no securities")
        request = _build_request(
            collection_dir=collection_dir,
            memberships_path=memberships_path,
            memberships_manifest=memberships_manifest,
            contract=contract,
            security_ids=security_ids,
            securities_per_shard=securities_per_shard,
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
        )
        request_sha256 = _json_sha256(request)
        _bind_request(output_dir, request, request_sha256)
        final_dir = output_dir / "final"
        if final_dir.exists():
            return load_complete_swing_feature_panel(output_dir)

        artifacts = load_collection_artifact_index(collection_dir)
        benchmark_tickers = sorted(
            {
                contract.labels.benchmark_market.upper(),
                "QQQ",
                *memberships["primary_benchmark"].astype(str).str.upper(),
            }
        )
        benchmark_bars = pd.concat(
            [load_daily_bars(ticker, artifacts) for ticker in benchmark_tickers],
            ignore_index=True,
        )
        _guard(budget, "swing panel benchmark load")

        shard_records: list[dict[str, Any]] = []
        built_this_run = 0
        groups = list(iter_security_batches(memberships, securities_per_shard))
        for index, group in enumerate(groups):
            expected_ids = sorted(group["security_id"].astype(str).unique())
            shard_path = output_dir / "stage1" / f"shard-{index:04d}.parquet"
            record_path = shard_path.with_suffix(".json")
            existing = _load_stage_one_shard_record(
                shard_path,
                record_path,
                request_sha256=request_sha256,
                expected_security_ids=expected_ids,
            )
            if existing is not None:
                shard_records.append(existing)
                continue
            if (
                maximum_stage_one_shards_this_run is not None
                and built_this_run >= maximum_stage_one_shards_this_run
            ):
                return {
                    "status": "incomplete",
                    "request_sha256": request_sha256,
                    "completed_stage_one_shards": len(shard_records),
                    "total_stage_one_shards": len(groups),
                }
            stock_bars, zero_volume_dropped = load_security_batch_bars(
                group,
                artifacts,
            )
            if stock_bars.empty:
                raise DataReadinessError(
                    f"stage-one shard {index} has no traded bars"
                )
            rows = build_swing_feature_rows(
                stock_bars,
                benchmark_bars,
                group,
                contract=contract,
            )
            record = _publish_stage_one_shard(
                rows,
                shard_path=shard_path,
                record_path=record_path,
                request_sha256=request_sha256,
                expected_security_ids=expected_ids,
                zero_volume_bars_dropped=zero_volume_dropped,
            )
            shard_records.append(record)
            built_this_run += 1
            del stock_bars, rows
            release_process_memory()
            _guard(budget, f"swing panel stage-one shard {index}")

        del benchmark_bars
        release_process_memory()
        panel = _load_complete_stage_one_population(
            output_dir,
            shard_records=shard_records,
            request_sha256=request_sha256,
            expected_security_ids=security_ids,
        )
        _guard(budget, "swing panel stage-two input")
        finalized = finalize_swing_feature_panel(
            panel,
            contract=contract,
            expected_security_ids=security_ids,
        )
        del panel
        release_process_memory()
        _guard(budget, "swing panel stage-two finalization")
        manifest = _publish_final_panel(
            finalized,
            output_dir=output_dir,
            request_sha256=request_sha256,
            request=request,
            shard_records=shard_records,
            budget=budget,
        )
        return {"status": "complete", **manifest}


def load_complete_swing_feature_panel(output_dir: Path) -> dict[str, Any]:
    """Verify the request, manifest, authority, and every final partition."""

    request_path = output_dir / "_request.json"
    final_dir = output_dir / "final"
    manifest_path = final_dir / "_manifest.json"
    authority_path = final_dir / "_authority.json"
    request = _load_json(request_path)
    manifest = _load_json(manifest_path)
    authority = _load_json(authority_path)
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request_sha256 = _json_sha256(request_payload)
    if (
        request.get("schema") != SWING_MATERIALIZATION_REQUEST_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != SWING_MATERIALIZATION_MANIFEST_SCHEMA
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != SWING_MATERIALIZATION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
    ):
        raise DataReadinessError(
            f"swing panel authority does not verify: {output_dir}"
        )
    files = manifest.get("files")
    if not isinstance(files, list) or not files:
        raise DataReadinessError("swing panel manifest has no partitions")
    rows = 0
    for record in files:
        if not isinstance(record, dict):
            raise DataReadinessError("swing panel partition record is invalid")
        path = final_dir / str(record.get("path", ""))
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(f"swing panel partition does not verify: {path}")
        rows += int(record.get("rows", -1))
    if rows != int(manifest.get("rows", -1)):
        raise DataReadinessError("swing panel partition row count does not add up")
    return {"status": "complete", **manifest}


def _build_request(
    *,
    collection_dir: Path,
    memberships_path: Path,
    memberships_manifest: Mapping[str, object],
    contract: StrategyContract,
    security_ids: list[str],
    securities_per_shard: int,
    memory_budget_gib: float,
    memory_headroom_gib: float,
) -> dict[str, Any]:
    collection_manifest = collection_dir / "_manifest.json"
    if not collection_manifest.is_file():
        raise DataReadinessError(
            f"daily collection manifest is missing: {collection_manifest}"
        )
    return {
        "schema": SWING_MATERIALIZATION_REQUEST_SCHEMA,
        "collection_dir": str(collection_dir.resolve()),
        "collection_manifest_sha256": file_sha256(collection_manifest),
        "memberships_path": str(memberships_path.resolve()),
        "memberships_sha256": str(memberships_manifest["artifact_sha256"]),
        "memberships_manifest_sha256": file_sha256(
            manifest_path_for(memberships_path)
        ),
        "strategy_contract_sha256": contract.sha256(),
        "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
        "feature_profile": SWING_FEATURE_PROFILE,
        "security_count": len(security_ids),
        "security_ids_sha256": _json_sha256(security_ids),
        "securities_per_shard": securities_per_shard,
        "memory_budget_gib": memory_budget_gib,
        "memory_headroom_gib": memory_headroom_gib,
    }


def _bind_request(
    output_dir: Path,
    request: Mapping[str, Any],
    request_sha256: str,
) -> None:
    path = output_dir / "_request.json"
    bound = {**request, "request_sha256": request_sha256}
    if path.exists():
        if _load_json(path) != bound:
            raise DataReadinessError(
                f"swing materialization resume request differs: {path}"
            )
        return
    output_dir.mkdir(parents=True, exist_ok=True)
    _write_json_atomic(path, bound)


def _publish_stage_one_shard(
    rows: pd.DataFrame,
    *,
    shard_path: Path,
    record_path: Path,
    request_sha256: str,
    expected_security_ids: list[str],
    zero_volume_bars_dropped: int,
) -> dict[str, Any]:
    _validate_stage_one_rows(rows, expected_security_ids)
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = shard_path.with_name(f".{shard_path.name}.{uuid.uuid4().hex}.tmp")
    try:
        rows.to_parquet(temporary, index=False)
        temporary.replace(shard_path)
    finally:
        temporary.unlink(missing_ok=True)
    sessions = pd.to_datetime(rows["session_date_et"])
    record = {
        "schema": SWING_STAGE_ONE_SHARD_SCHEMA,
        "request_sha256": request_sha256,
        "path": str(shard_path.name),
        "sha256": file_sha256(shard_path),
        "rows": int(len(rows)),
        "columns": list(rows.columns),
        "security_ids": expected_security_ids,
        "sessions": int(sessions.nunique()),
        "first_session": str(sessions.min().date()),
        "last_session": str(sessions.max().date()),
        "feature_eligible_rows": int(rows["feature_eligible"].fillna(False).sum()),
        "barrier_resolved_rows": int(rows["barrier_label"].notna().sum()),
        "zero_volume_bars_dropped": int(zero_volume_bars_dropped),
    }
    _write_json_atomic(record_path, record)
    return record


def _load_stage_one_shard_record(
    shard_path: Path,
    record_path: Path,
    *,
    request_sha256: str,
    expected_security_ids: list[str],
) -> dict[str, Any] | None:
    if not shard_path.exists() and not record_path.exists():
        return None
    if not shard_path.is_file() or not record_path.is_file():
        raise DataReadinessError(f"incomplete swing stage-one shard: {shard_path}")
    record = _load_json(record_path)
    if (
        record.get("schema") != SWING_STAGE_ONE_SHARD_SCHEMA
        or record.get("request_sha256") != request_sha256
        or record.get("security_ids") != expected_security_ids
        or record.get("sha256") != file_sha256(shard_path)
    ):
        raise DataReadinessError(f"swing stage-one shard does not verify: {shard_path}")
    return record


def _load_complete_stage_one_population(
    output_dir: Path,
    *,
    shard_records: list[dict[str, Any]],
    request_sha256: str,
    expected_security_ids: list[str],
) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for index, record in enumerate(shard_records):
        path = output_dir / "stage1" / f"shard-{index:04d}.parquet"
        verified = _load_stage_one_shard_record(
            path,
            path.with_suffix(".json"),
            request_sha256=request_sha256,
            expected_security_ids=list(record["security_ids"]),
        )
        if verified is None:
            raise DataReadinessError(f"stage-one shard disappeared: {path}")
        frame = pd.read_parquet(path)
        if len(frame) != int(verified["rows"]):
            raise DataReadinessError(f"stage-one shard row count differs: {path}")
        parts.append(frame)
    panel = pd.concat(parts, ignore_index=True)
    _validate_stage_one_rows(panel, expected_security_ids)
    return panel


def _validate_stage_one_rows(
    rows: pd.DataFrame,
    expected_security_ids: list[str],
) -> None:
    if rows.empty:
        raise DataReadinessError("swing stage-one rows are empty")
    observed = sorted(rows["security_id"].astype(str).unique())
    if observed != expected_security_ids:
        raise DataReadinessError(
            "swing stage-one security population differs from its request"
        )
    if bool(rows.duplicated(["security_id", "session_date_et"]).any()):
        raise DataReadinessError("swing stage-one rows repeat a security/session")
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(rows["available_at_utc"], utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or bool((available > decision).any()):
        raise DataReadinessError("swing stage-one rows violate feature availability")


def _publish_final_panel(
    panel: pd.DataFrame,
    *,
    output_dir: Path,
    request_sha256: str,
    request: Mapping[str, Any],
    shard_records: list[dict[str, Any]],
    budget: tuple[float, float],
) -> dict[str, Any]:
    _validate_final_panel(panel)
    staging = output_dir / f".final.{uuid.uuid4().hex}.staging"
    final_dir = output_dir / "final"
    staging.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        years = pd.to_datetime(panel["session_date_et"]).dt.year
        for year in sorted(years.unique()):
            part = panel.loc[years.eq(year)]
            path = staging / "panel" / f"year={int(year)}" / "part.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.to_parquet(path, index=False)
            files.append(
                {
                    "path": str(path.relative_to(staging)).replace("\\", "/"),
                    "sha256": file_sha256(path),
                    "rows": int(len(part)),
                    "year": int(year),
                }
            )
            _guard(budget, f"swing panel publish year {year}")
        sessions = pd.to_datetime(panel["session_date_et"])
        profile = sorted(panel["feature_profile"].astype(str).unique())
        manifest: dict[str, Any] = {
            "schema": SWING_MATERIALIZATION_MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request_sha256,
            "strategy_contract_sha256": request["strategy_contract_sha256"],
            "feature_profiles": profile,
            "rows": int(len(panel)),
            "columns": list(panel.columns),
            "securities": int(panel["security_id"].nunique()),
            "sessions": int(sessions.nunique()),
            "first_session": str(sessions.min().date()),
            "last_session": str(sessions.max().date()),
            "feature_eligible_rows": int(panel["feature_eligible"].fillna(False).sum()),
            "barrier_resolved_rows": int(panel["barrier_label"].notna().sum()),
            "rank_eligible_rows": int(panel["rank_label"].notna().sum()),
            "cross_section_eligible_rows": int(
                panel["cross_section_eligible"].fillna(False).sum()
            ),
            "stage_one_shards": len(shard_records),
            "stage_one_rows": int(sum(int(item["rows"]) for item in shard_records)),
            "zero_volume_bars_dropped": int(
                sum(int(item["zero_volume_bars_dropped"]) for item in shard_records)
            ),
            "source": {
                "collection_manifest_sha256": request["collection_manifest_sha256"],
                "memberships_sha256": request["memberships_sha256"],
            },
            "memory": memory_audit(
                hard_budget_gib=budget[0],
                headroom_gib=budget[1],
            ).to_record(),
            "files": files,
        }
        _write_json_atomic(staging / "_manifest.json", manifest)
        _write_json_atomic(
            staging / "_authority.json",
            {
                "schema": SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": request_sha256,
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
            },
        )
        if final_dir.exists():
            raise DataReadinessError(f"swing panel final output already exists: {final_dir}")
        staging.replace(final_dir)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_final_panel(panel: pd.DataFrame) -> None:
    _validate_stage_one_rows(
        panel,
        sorted(panel["security_id"].astype(str).unique()),
    )
    if set(panel["feature_profile"].astype(str)) != {SWING_FEATURE_PROFILE}:
        raise DataReadinessError("swing panel mixes or changes feature profiles")
    resolved = panel["barrier_label"].notna()
    label_available = pd.to_datetime(
        panel.loc[resolved, "barrier_label_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    decision = pd.to_datetime(
        panel.loc[resolved, "decision_time_utc"],
        utc=True,
        errors="coerce",
    )
    if label_available.isna().any() or bool((label_available < decision).any()):
        raise DataReadinessError("swing panel labels are available before resolution")
    ranked = panel["rank_label"].notna()
    if bool((ranked & ~resolved).any()):
        raise DataReadinessError("swing panel ranks unresolved outcomes")


def _guard(budget: tuple[float, float], stage: str) -> None:
    hard, headroom = budget
    assert_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)
    assert_peak_memory_budget(
        hard_budget_gib=hard,
        headroom_gib=headroom,
        stage=stage,
    )


def _json_sha256(value: object) -> str:
    payload = json.dumps(value, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(payload.encode()).hexdigest()


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (FileNotFoundError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"invalid or missing JSON artifact: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"JSON artifact must contain an object: {path}")
    return {str(key): item for key, item in value.items()}


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
