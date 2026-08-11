"""Resumable, immutable materialization of the ER1A swing ranking panel."""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any, Final

import pandas as pd

from market_predictor.canonical.store import (
    file_sha256,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.swing_artifact_contracts import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)
from market_predictor.edge_rebuild.swing_daily_combination import (
    CUTOFF_DATE,
    VerifiedCombinedInputs,
    prepare_combined_daily_store,
    verify_combined_swing_inputs,
)
from market_predictor.edge_rebuild.swing_features import (
    MANAGED_PATH_COST_POLICY,
    SWING_FEATURE_PANEL_SCHEMA,
    SWING_FEATURE_PROFILE,
    build_swing_feature_rows,
    finalize_swing_feature_panel,
)
from market_predictor.edge_rebuild.swing_setups import (
    iter_security_batches,
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
from market_predictor.swing.contracts import MINIMUM_SWING_DECISION_DATE
from market_predictor.v3.errors import DataReadinessError

SWING_MATERIALIZATION_REQUEST_SCHEMA: Final = (
    "edge_rebuild.swing_panel_materialization_request.v11"
)
SWING_STAGE_ONE_SHARD_SCHEMA: Final = "edge_rebuild.swing_panel_stage_one_shard.v8"
SWING_MATERIALIZATION_PROFILES: Final = (SWING_FEATURE_PROFILE,)


def materialize_swing_feature_panel(
    *,
    pre_plan_directory: Path,
    pre_collection_directory: Path,
    post_collection_directory: Path,
    membership_directory: Path,
    raw_archive_directory: Path,
    event_directory: Path,
    transition_directory: Path,
    reviewed_transitions_path: Path,
    anchor_path: Path,
    contract: StrategyContract,
    output_dir: Path,
    security_exclusions_path: Path | None = None,
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
        verified = verify_combined_swing_inputs(
            pre_plan_directory=pre_plan_directory,
            pre_collection_directory=pre_collection_directory,
            post_collection_directory=post_collection_directory,
            membership_directory=membership_directory,
            raw_archive_directory=raw_archive_directory,
            event_directory=event_directory,
            transition_directory=transition_directory,
            reviewed_transitions_path=reviewed_transitions_path,
            anchor_path=anchor_path,
            security_exclusions_path=security_exclusions_path,
            model_decision_start=MINIMUM_SWING_DECISION_DATE,
            model_decision_cutoff=CUTOFF_DATE,
        )
        memberships = verified.memberships
        security_ids = sorted(memberships["security_id"].astype(str).unique())
        warmup_only_security_ids = list(verified.warmup_only_security_ids)
        if not security_ids:
            raise DataReadinessError("swing materialization has no securities")
        request = _build_request(
            verified=verified,
            contract=contract,
            security_ids=security_ids,
            warmup_only_security_ids=warmup_only_security_ids,
            securities_per_shard=securities_per_shard,
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
        )
        request_sha256 = _json_sha256(request)
        _bind_request(output_dir, request, request_sha256)
        final_dir = output_dir / "final"
        combined = prepare_combined_daily_store(
            verified=verified,
            output_directory=output_dir / "combined_daily",
            parent_request_sha256=request_sha256,
            memory_budget_gib=memory_budget_gib,
            memory_headroom_gib=memory_headroom_gib,
        )
        if final_dir.exists():
            return load_complete_swing_feature_panel(output_dir)
        artifacts = combined.artifacts
        sparse_missing_sessions = _sparse_missing_sessions_by_ticker(
            verified.coverage_audit
        )
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
                sparse_missing_sessions_by_ticker=sparse_missing_sessions,
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
        manifest = _finalize_and_publish_stage_one(
            output_dir=output_dir,
            request_sha256=request_sha256,
            request=request,
            shard_records=shard_records,
            expected_security_ids=security_ids,
            contract=contract,
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
    source = manifest.get("source")
    combined_authority = output_dir / "combined_daily" / "_authority.json"
    if (
        request.get("schema") != SWING_MATERIALIZATION_REQUEST_SCHEMA
        or request.get("decision_start_date")
        != MINIMUM_SWING_DECISION_DATE.isoformat()
        or request.get("swing_feature_panel_schema")
        != SWING_FEATURE_PANEL_SCHEMA
        or request.get("feature_profiles")
        != list(SWING_MATERIALIZATION_PROFILES)
        or request.get("profile_policy")
        != "technical_features_and_full_population_labels_only"
        or request.get("request_sha256") != request_sha256
        or not _population_audit_verifies(request)
        or manifest.get("schema") != SWING_MATERIALIZATION_MANIFEST_SCHEMA
        or manifest.get("decision_start_date")
        != MINIMUM_SWING_DECISION_DATE.isoformat()
        or manifest.get("swing_feature_panel_schema")
        != SWING_FEATURE_PANEL_SCHEMA
        or manifest.get("feature_profiles")
        != list(SWING_MATERIALIZATION_PROFILES)
        or manifest.get("strategy_contract_sha256")
        != request.get("strategy_contract_sha256")
        or manifest.get("request_sha256") != request_sha256
        or manifest.get("modeled_security_count")
        != request.get("modeled_security_count")
        or manifest.get("modeled_security_ids_sha256")
        != request.get("modeled_security_ids_sha256")
        or manifest.get("warmup_only_security_count")
        != request.get("warmup_only_security_count")
        or manifest.get("warmup_only_security_ids_sha256")
        != request.get("warmup_only_security_ids_sha256")
        or manifest.get("warmup_only_security_ids")
        != request.get("warmup_only_security_ids")
        or request.get("managed_path_cost_policy") != MANAGED_PATH_COST_POLICY
        or manifest.get("managed_path_cost_policy") != MANAGED_PATH_COST_POLICY
        or request.get("physical_partitioning")
        != ["feature_profile", "calendar_month"]
        or manifest.get("physical_partitioning")
        != ["feature_profile", "calendar_month"]
        or authority.get("schema") != SWING_MATERIALIZATION_AUTHORITY_SCHEMA
        or authority.get("decision_start_date")
        != MINIMUM_SWING_DECISION_DATE.isoformat()
        or authority.get("swing_feature_panel_schema")
        != SWING_FEATURE_PANEL_SCHEMA
        or authority.get("strategy_contract_sha256")
        != request.get("strategy_contract_sha256")
        or authority.get("managed_path_cost_policy") != MANAGED_PATH_COST_POLICY
        or authority.get("modeled_security_count")
        != request.get("modeled_security_count")
        or authority.get("modeled_security_ids_sha256")
        != request.get("modeled_security_ids_sha256")
        or authority.get("warmup_only_security_count")
        != request.get("warmup_only_security_count")
        or authority.get("warmup_only_security_ids_sha256")
        != request.get("warmup_only_security_ids_sha256")
        or authority.get("state") != "complete"
        or authority.get("request_sha256") != request_sha256
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or not isinstance(source, dict)
        or not combined_authority.is_file()
        or source.get("combined_daily_authority_sha256")
        != file_sha256(combined_authority)
    ):
        raise DataReadinessError(
            f"swing panel authority does not verify: {output_dir}"
        )
    files = manifest.get("files")
    files_by_profile = manifest.get("files_by_profile")
    if (
        not isinstance(files, list)
        or not files
        or not isinstance(files_by_profile, Mapping)
        or set(files_by_profile) != set(SWING_MATERIALIZATION_PROFILES)
        or files_by_profile.get(SWING_FEATURE_PROFILE) != files
    ):
        raise DataReadinessError("swing panel manifest has invalid profile partitions")
    rows_by_profile: dict[str, int] = {}
    security_ids_by_profile: dict[str, set[str]] = {}
    sessions_by_profile: dict[str, set[date]] = {}
    decision_ids_by_profile: dict[str, set[str]] = {}
    for profile in SWING_MATERIALIZATION_PROFILES:
        profile_files = files_by_profile.get(profile)
        if not isinstance(profile_files, list) or not profile_files:
            raise DataReadinessError(
                f"swing panel manifest has no {profile} partitions"
            )
        (
            rows_by_profile[profile],
            security_ids_by_profile[profile],
            sessions_by_profile[profile],
            decision_ids_by_profile[profile],
        ) = (
            _verify_final_partition_records(
                final_dir,
                profile_files,
                expected_profile=profile,
            )
        )
    modeled_population_hashes = {
        _json_sha256(sorted(values)) for values in security_ids_by_profile.values()
    }
    warmup_only = set(str(value) for value in request["warmup_only_security_ids"])
    observed_security_count = len(security_ids_by_profile[SWING_FEATURE_PROFILE])
    technical_rows = rows_by_profile[SWING_FEATURE_PROFILE]
    observed_sessions = sessions_by_profile[SWING_FEATURE_PROFILE]
    observed_decision_ids = decision_ids_by_profile[SWING_FEATURE_PROFILE]
    if (
        technical_rows != int(manifest.get("rows", -1))
        or technical_rows != int(manifest.get("stage_one_rows", -1))
        or len(observed_decision_ids) != technical_rows
        or len(observed_sessions) != int(manifest.get("sessions", -1))
        or str(manifest.get("first_session")) != min(observed_sessions).isoformat()
        or str(manifest.get("last_session")) != max(observed_sessions).isoformat()
        or observed_security_count != int(manifest.get("securities", -1))
        or observed_security_count != int(manifest.get("modeled_security_count", -1))
        or observed_security_count != int(request.get("modeled_security_count", -1))
        or modeled_population_hashes
        != {str(request.get("modeled_security_ids_sha256"))}
        or any(
            not values.isdisjoint(warmup_only)
            for values in security_ids_by_profile.values()
        )
    ):
        raise DataReadinessError(
            "swing panel profile rows or modeled security population do not verify"
        )
    return {"status": "complete", **manifest}


def _build_request(
    *,
    verified: VerifiedCombinedInputs,
    contract: StrategyContract,
    security_ids: list[str],
    warmup_only_security_ids: list[str],
    securities_per_shard: int,
    memory_budget_gib: float,
    memory_headroom_gib: float,
) -> dict[str, Any]:
    return {
        "schema": SWING_MATERIALIZATION_REQUEST_SCHEMA,
        "combined_daily_inputs": verified.request_payload,
        "strategy_contract_sha256": contract.sha256(),
        "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
        "managed_path_cost_policy": MANAGED_PATH_COST_POLICY,
        "physical_partitioning": ["feature_profile", "calendar_month"],
        "decision_start_date": MINIMUM_SWING_DECISION_DATE.isoformat(),
        "feature_profiles": list(SWING_MATERIALIZATION_PROFILES),
        "profile_policy": "technical_features_and_full_population_labels_only",
        "global_authority": None,
        "global_authority_policy": (
            "not_integrated_until_a_separate_verified_authority_is_bound"
        ),
        "security_count": len(security_ids),
        "security_ids_sha256": _json_sha256(security_ids),
        "modeled_security_count": len(security_ids),
        "modeled_security_ids_sha256": _json_sha256(security_ids),
        "warmup_only_security_count": len(warmup_only_security_ids),
        "warmup_only_security_ids": warmup_only_security_ids,
        "warmup_only_security_ids_sha256": _json_sha256(
            warmup_only_security_ids
        ),
        "securities_per_shard": securities_per_shard,
        "memory_budget_gib": memory_budget_gib,
        "memory_headroom_gib": memory_headroom_gib,
    }


def _population_audit_verifies(payload: Mapping[str, Any]) -> bool:
    warmup_only = payload.get("warmup_only_security_ids")
    if not isinstance(warmup_only, list) or any(
        not isinstance(value, str) or not value for value in warmup_only
    ):
        return False
    return bool(
        warmup_only == sorted(set(warmup_only))
        and payload.get("security_count") == payload.get("modeled_security_count")
        and payload.get("security_ids_sha256")
        == payload.get("modeled_security_ids_sha256")
        and payload.get("warmup_only_security_count") == len(warmup_only)
        and payload.get("warmup_only_security_ids_sha256")
        == _json_sha256(warmup_only)
    )


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


def _verified_stage_one_paths(
    output_dir: Path,
    *,
    shard_records: list[dict[str, Any]],
    request_sha256: str,
    expected_security_ids: list[str],
) -> list[Path]:
    paths: list[Path] = []
    observed_security_ids: set[str] = set()
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
        observed_security_ids.update(str(value) for value in verified["security_ids"])
        paths.append(path)
    if sorted(observed_security_ids) != expected_security_ids:
        raise DataReadinessError(
            "complete swing stage-one security population differs from its request"
        )
    return paths


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
    if (
        "swing_feature_panel_schema" not in rows
        or bool(
            rows["swing_feature_panel_schema"]
            .astype(str)
            .ne(SWING_FEATURE_PANEL_SCHEMA)
            .any()
        )
    ):
        raise DataReadinessError(
            "swing stage-one feature schema is stale or missing"
        )
    sessions = pd.to_datetime(rows["session_date_et"], errors="coerce").dt.date
    if sessions.isna().any() or bool(
        sessions.lt(MINIMUM_SWING_DECISION_DATE).any()
    ):
        raise DataReadinessError(
            "swing stage-one rows contain a missing or pre-"
            f"{MINIMUM_SWING_DECISION_DATE.isoformat()} decision session"
        )
    decision = pd.to_datetime(rows["decision_time_utc"], utc=True, errors="coerce")
    available = pd.to_datetime(rows["available_at_utc"], utc=True, errors="coerce")
    if decision.isna().any() or available.isna().any() or bool((available > decision).any()):
        raise DataReadinessError("swing stage-one rows violate feature availability")


def _finalize_and_publish_stage_one(
    *,
    output_dir: Path,
    request_sha256: str,
    request: Mapping[str, Any],
    shard_records: list[dict[str, Any]],
    expected_security_ids: list[str],
    contract: StrategyContract,
    budget: tuple[float, float],
) -> dict[str, Any]:
    paths = _verified_stage_one_paths(
        output_dir,
        shard_records=shard_records,
        request_sha256=request_sha256,
        expected_security_ids=expected_security_ids,
    )
    first_year = min(int(str(item["first_session"])[:4]) for item in shard_records)
    last_year = max(int(str(item["last_session"])[:4]) for item in shard_records)
    staging = output_dir / f".final.{uuid.uuid4().hex}.staging"
    final_dir = output_dir / "final"
    staging.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        columns_by_profile: dict[str, list[str]] = {}
        observed_security_ids: set[str] = set()
        observed_sessions: set[date] = set()
        totals_by_profile = {
            profile: {
                "rows": 0,
                "feature_eligible_rows": 0,
                "label_eligible_rows": 0,
                "barrier_resolved_rows": 0,
                "rank_eligible_rows": 0,
                "cross_section_eligible_rows": 0,
            }
            for profile in SWING_MATERIALIZATION_PROFILES
        }
        first_session: str | None = None
        last_session: str | None = None
        for year in range(first_year, last_year + 1):
            parts: list[pd.DataFrame] = []
            for path in paths:
                frame = pd.read_parquet(path)
                selected = pd.to_datetime(frame["session_date_et"]).dt.year.eq(year)
                if bool(selected.any()):
                    parts.append(frame.loc[selected].copy())
                del frame
            if not parts:
                continue
            rows = pd.concat(parts, ignore_index=True)
            del parts
            _guard(budget, f"swing panel stage-two input year {year}")
            for profile in SWING_MATERIALIZATION_PROFILES:
                rows["feature_profile"] = profile
                finalized = finalize_swing_feature_panel(
                    rows,
                    contract=contract,
                )
                _validate_final_panel(finalized, expected_profile=profile)
                session_values = pd.to_datetime(finalized["session_date_et"])
                if not bool(session_values.dt.year.eq(year).all()):
                    raise DataReadinessError(
                        f"swing panel stage two mixed session years in {year}"
                    )
                year_columns = list(finalized.columns)
                if profile not in columns_by_profile:
                    columns_by_profile[profile] = year_columns
                elif columns_by_profile[profile] != year_columns:
                    raise DataReadinessError(
                        f"swing panel {profile} year partitions change schema"
                    )
                month_values = session_values.dt.strftime("%Y-%m")
                for month, indices in finalized.groupby(
                    month_values,
                    sort=True,
                ).groups.items():
                    partition = finalized.loc[indices].copy()
                    partition_sessions = pd.to_datetime(
                        partition["session_date_et"]
                    )
                    path = (
                        staging
                        / "panel"
                        / f"feature_profile={profile}"
                        / f"month={month}"
                        / "part.parquet"
                    )
                    path.parent.mkdir(parents=True, exist_ok=True)
                    partition.to_parquet(path, index=False)
                    decision_ids = sorted(partition["decision_id"].astype(str))
                    files.append(
                        {
                            "path": str(path.relative_to(staging)).replace("\\", "/"),
                            "sha256": file_sha256(path),
                            "rows": int(len(partition)),
                            "partition_month": str(month),
                            "feature_profile": profile,
                            "sessions": int(partition_sessions.nunique()),
                            "first_session": str(partition_sessions.min().date()),
                            "last_session": str(partition_sessions.max().date()),
                            "securities": int(partition["security_id"].nunique()),
                            "decision_ids_sha256": _json_sha256(decision_ids),
                        }
                    )
                    observed_security_ids.update(partition["security_id"].astype(str))
                    observed_sessions.update(partition_sessions.dt.date)
                    profile_totals = totals_by_profile[profile]
                    profile_totals["rows"] += int(len(partition))
                    profile_totals["feature_eligible_rows"] += int(
                        partition["feature_eligible"].fillna(False).sum()
                    )
                    profile_totals["label_eligible_rows"] += int(
                        partition["label_eligible"].fillna(False).sum()
                    )
                    profile_totals["barrier_resolved_rows"] += int(
                        partition["barrier_label"].notna().sum()
                    )
                    profile_totals["rank_eligible_rows"] += int(
                        partition["rank_label"].notna().sum()
                    )
                    profile_totals["cross_section_eligible_rows"] += int(
                        partition["cross_section_eligible"].fillna(False).sum()
                    )
                    month_first = str(partition_sessions.min().date())
                    month_last = str(partition_sessions.max().date())
                    first_session = (
                        min(first_session, month_first)
                        if first_session
                        else month_first
                    )
                    last_session = (
                        max(last_session, month_last)
                        if last_session
                        else month_last
                    )
                    del partition
                del finalized
                release_process_memory()
                _guard(budget, f"swing panel publish {profile} year {year}")
            del rows
            release_process_memory()
            _guard(budget, f"swing panel publish year {year}")
        expected_stage_one_rows = sum(int(item["rows"]) for item in shard_records)
        expected_rows = expected_stage_one_rows
        observed_rows = sum(
            int(values["rows"]) for values in totals_by_profile.values()
        )
        if observed_rows != expected_rows:
            raise DataReadinessError(
                f"swing panel stage two retained {observed_rows} of {expected_rows} "
                "ablation rows"
            )
        technical_eligible = int(
            totals_by_profile[SWING_FEATURE_PROFILE]["feature_eligible_rows"]
        )
        if technical_eligible == 0:
            raise DataReadinessError("swing technical panel has no eligible rows")
        for profile, totals in totals_by_profile.items():
            if (
                int(totals["label_eligible_rows"]) < 1
                or int(totals["rank_eligible_rows"]) < 1
                or int(totals["cross_section_eligible_rows"]) < 1
            ):
                raise DataReadinessError(
                    f"swing {profile} panel has no trainable causal labels or "
                    "eligible ranking cross-sections"
                )
        if sorted(observed_security_ids) != expected_security_ids:
            raise DataReadinessError(
                "swing panel final security population differs from its request"
            )
        if (
            set(columns_by_profile) != set(SWING_MATERIALIZATION_PROFILES)
            or first_session is None
            or last_session is None
        ):
            raise DataReadinessError("swing panel stage two produced no partitions")
        manifest: dict[str, Any] = {
            "schema": SWING_MATERIALIZATION_MANIFEST_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request_sha256,
            "strategy_contract_sha256": request["strategy_contract_sha256"],
            "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
            "managed_path_cost_policy": MANAGED_PATH_COST_POLICY,
            "physical_partitioning": ["feature_profile", "calendar_month"],
            "decision_start_date": MINIMUM_SWING_DECISION_DATE.isoformat(),
            "feature_profiles": list(SWING_MATERIALIZATION_PROFILES),
            "rows": expected_stage_one_rows,
            "columns_by_profile": columns_by_profile,
            "securities": len(observed_security_ids),
            "modeled_security_count": request["modeled_security_count"],
            "modeled_security_ids_sha256": request[
                "modeled_security_ids_sha256"
            ],
            "warmup_only_security_count": request[
                "warmup_only_security_count"
            ],
            "warmup_only_security_ids": request[
                "warmup_only_security_ids"
            ],
            "warmup_only_security_ids_sha256": request[
                "warmup_only_security_ids_sha256"
            ],
            "sessions": len(observed_sessions),
            "first_session": first_session,
            "last_session": last_session,
            "profile_totals": totals_by_profile,
            "stage_one_shards": len(shard_records),
            "stage_one_rows": expected_stage_one_rows,
            "zero_volume_bars_dropped": int(
                sum(int(item["zero_volume_bars_dropped"]) for item in shard_records)
            ),
            "source": {
                "combined_daily_authority_sha256": file_sha256(
                    output_dir / "combined_daily" / "_authority.json"
                ),
                "pre_collection": request["combined_daily_inputs"][
                    "pre_collection"
                ],
                "post_collection": request["combined_daily_inputs"][
                    "post_collection"
                ],
                "membership_authority": request["combined_daily_inputs"][
                    "membership_authority"
                ],
                "excluded_security_ids_sha256": request["combined_daily_inputs"][
                    "excluded_security_ids_sha256"
                ],
                "coverage_audit_sha256": request["combined_daily_inputs"][
                    "coverage_audit_sha256"
                ],
                "security_exclusions": request["combined_daily_inputs"][
                    "security_exclusions"
                ],
                "benchmark_coverage": request["combined_daily_inputs"][
                    "benchmark_coverage"
                ],
            },
            "memory": memory_audit(
                hard_budget_gib=budget[0],
                headroom_gib=budget[1],
            ).to_record(),
            "files": [
                item
                for item in files
                if item["feature_profile"] == SWING_FEATURE_PROFILE
            ],
            "files_by_profile": {
                profile: [
                    item
                    for item in files
                    if item["feature_profile"] == profile
                ]
                for profile in SWING_MATERIALIZATION_PROFILES
            },
        }
        _write_json_atomic(staging / "_manifest.json", manifest)
        _write_json_atomic(
            staging / "_authority.json",
            {
                "schema": SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
                "state": "complete",
                "request_sha256": request_sha256,
                "strategy_contract_sha256": request[
                    "strategy_contract_sha256"
                ],
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "swing_feature_panel_schema": SWING_FEATURE_PANEL_SCHEMA,
                "managed_path_cost_policy": MANAGED_PATH_COST_POLICY,
                "decision_start_date": MINIMUM_SWING_DECISION_DATE.isoformat(),
                "modeled_security_count": request["modeled_security_count"],
                "modeled_security_ids_sha256": request[
                    "modeled_security_ids_sha256"
                ],
                "warmup_only_security_count": request[
                    "warmup_only_security_count"
                ],
                "warmup_only_security_ids_sha256": request[
                    "warmup_only_security_ids_sha256"
                ],
            },
        )
        if final_dir.exists():
            raise DataReadinessError(f"swing panel final output already exists: {final_dir}")
        staging.replace(final_dir)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _validate_final_panel(
    panel: pd.DataFrame,
    *,
    expected_profile: str,
) -> None:
    _validate_stage_one_rows(
        panel,
        sorted(panel["security_id"].astype(str).unique()),
    )
    if (
        "swing_feature_panel_schema" not in panel
        or bool(
            panel["swing_feature_panel_schema"]
            .astype(str)
            .ne(SWING_FEATURE_PANEL_SCHEMA)
            .any()
        )
    ):
        raise DataReadinessError("swing panel feature schema is stale or mixed")
    if set(panel["feature_profile"].astype(str)) != {expected_profile}:
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


def _verify_final_partition_records(
    final_dir: Path,
    records: list[object],
    *,
    expected_profile: str,
) -> tuple[int, set[str], set[date], set[str]]:
    rows = 0
    observed_security_ids: set[str] = set()
    observed_sessions: set[date] = set()
    observed_decision_ids: set[str] = set()
    observed_paths: set[str] = set()
    observed_months: set[str] = set()
    for record in records:
        if (
            not isinstance(record, Mapping)
            or record.get("feature_profile") != expected_profile
        ):
            raise DataReadinessError("swing panel partition record is invalid")
        relative_path = str(record.get("path", "")).replace("\\", "/")
        expected_prefix = f"panel/feature_profile={expected_profile}/"
        month = str(record.get("partition_month", ""))
        expected_path = f"{expected_prefix}month={month}/part.parquet"
        if (
            relative_path != expected_path
            or relative_path in observed_paths
            or month in observed_months
        ):
            raise DataReadinessError(
                "swing panel partition path conflicts with its feature profile"
            )
        observed_paths.add(relative_path)
        observed_months.add(month)
        path = final_dir / relative_path
        if not path.is_file() or file_sha256(path) != record.get("sha256"):
            raise DataReadinessError(
                f"swing panel partition does not verify: {path}"
            )
        contract_rows = pd.read_parquet(
            path,
            columns=[
                "session_date_et",
                "swing_feature_panel_schema",
                "feature_profile",
                "decision_id",
                "security_id",
            ],
        )
        partition_rows = int(record.get("rows", -1))
        if len(contract_rows) != partition_rows:
            raise DataReadinessError(
                f"swing panel partition row count does not verify: {path}"
            )
        sessions = pd.to_datetime(
            contract_rows["session_date_et"],
            errors="coerce",
        ).dt.date
        session_months = {
            value.strftime("%Y-%m") for value in sessions if not pd.isna(value)
        }
        decision_ids = contract_rows["decision_id"].astype("string")
        security_ids = contract_rows["security_id"].astype("string")
        if (
            sessions.isna().any()
            or bool(sessions.lt(MINIMUM_SWING_DECISION_DATE).any())
            or session_months != {month}
            or decision_ids.isna().any()
            or decision_ids.eq("").any()
            or decision_ids.duplicated().any()
            or security_ids.isna().any()
            or security_ids.eq("").any()
            or set(contract_rows["feature_profile"].astype(str)) != {expected_profile}
            or bool(
                contract_rows["swing_feature_panel_schema"]
                .astype(str)
                .ne(SWING_FEATURE_PANEL_SCHEMA)
                .any()
            )
        ):
            raise DataReadinessError(
                f"swing panel partition decision window or schema is invalid: {path}"
            )
        if (
            int(record.get("sessions", -1)) != int(pd.Series(sessions).nunique())
            or str(record.get("first_session")) != min(sessions).isoformat()
            or str(record.get("last_session")) != max(sessions).isoformat()
            or int(record.get("securities", -1)) != int(security_ids.nunique())
            or str(record.get("decision_ids_sha256"))
            != _json_sha256(sorted(decision_ids.astype(str)))
        ):
            raise DataReadinessError(
                f"swing panel partition identity or bounds do not verify: {path}"
            )
        current_decision_ids = set(decision_ids.astype(str))
        if not observed_decision_ids.isdisjoint(current_decision_ids):
            raise DataReadinessError(
                "swing panel decision identity is duplicated across partitions"
            )
        rows += partition_rows
        observed_security_ids.update(security_ids.astype(str))
        observed_sessions.update(sessions)
        observed_decision_ids.update(current_decision_ids)
    return rows, observed_security_ids, observed_sessions, observed_decision_ids


def _sparse_missing_sessions_by_ticker(
    coverage_audit: Mapping[str, Any],
) -> dict[str, tuple[date, ...]]:
    raw = coverage_audit.get("session_gap_audit")
    if not isinstance(raw, Mapping):
        raise DataReadinessError("swing materialization has no session-gap authority")
    gaps = raw.get("gaps")
    if not isinstance(gaps, list):
        raise DataReadinessError("swing session-gap authority has no gap records")
    result: dict[str, set[date]] = {}
    for item in gaps:
        if not isinstance(item, Mapping):
            raise DataReadinessError("swing session-gap authority record is invalid")
        ticker = str(item.get("ticker", "")).strip().upper()
        sessions = item.get("missing_sessions")
        if not ticker or not isinstance(sessions, list):
            raise DataReadinessError("swing session-gap authority record is malformed")
        try:
            parsed = {date.fromisoformat(str(value)) for value in sessions}
        except ValueError as exc:
            raise DataReadinessError(
                f"swing session-gap authority contains an invalid date: {ticker}"
            ) from exc
        result.setdefault(ticker, set()).update(parsed)
    observed_count = sum(len(values) for values in result.values())
    if observed_count != int(raw.get("missing_session_count", -1)):
        raise DataReadinessError(
            "swing session-gap authority missing-session count does not verify"
        )
    return {
        ticker: tuple(sorted(sessions))
        for ticker, sessions in sorted(result.items())
    }


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
