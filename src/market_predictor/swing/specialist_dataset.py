"""Causal KS3 swing-specialist dataset construction and publication."""

from __future__ import annotations

import gc
import hashlib
import json
import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

import numpy as np
import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.reconciliation import (
    ASSIGNMENT_FEATURE_COLUMNS,
    DEFAULT_EVENT_WINDOWS,
    aggregate_event_assignments,
    apply_event_assignment_features,
    event_feature_columns,
)
from market_predictor.canonical.store import (
    canonical_artifact_columns,
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.specialist_contracts import (
    SPECIALIST_DATASET_BUNDLE_SCHEMA,
    SPECIALIST_DATASET_SCHEMA,
    SwingSpecialistResearchConfig,
)
from market_predictor.swing.strategy_labels import (
    STRATEGY_IDS,
    SwingStrategyLabelPolicy,
    audit_swing_strategy_labels,
    build_swing_strategy_labels,
)
from market_predictor.v3.errors import DataReadinessError

CATALYST_LINEAGE_MANIFEST_SCHEMA = "swing.catalyst_lineage_manifest.v1"
CATALYST_WINDOWS = ("2h", "1d", "3d")
SPECIALIST_ARTIFACT_TYPE = "swing_specialist_dataset"
_BASE_FEATURE_COLUMNS = {
    "ticker",
    "security_id",
    "session_date_et",
    "decision_group_id",
    "decision_time_utc",
    "feature_available_at_utc",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "price_feed",
    "adjustment",
    "primary_benchmark",
    "feature_eligible",
    "cross_section_eligible",
    "membership_effective_to_utc",
    "universe_snapshot_id",
    "market_regime",
    "sector",
    "timeframe",
    "prediction_cutoff_policy_id",
    "atr_pct_14",
}
_LABEL_JOIN_KEYS = (
    "ticker",
    "security_id",
    "session_date_et",
    "decision_time_utc",
)


@dataclass(frozen=True)
class CatalystLineageData:
    assignments: pd.DataFrame
    coverage: pd.DataFrame
    lineage_sha256: str
    manifest_sha256: str
    request_sha256: str
    observed_chunks: int


@dataclass(frozen=True)
class CatalystAggregateLineageData:
    aggregates: pd.DataFrame
    coverage: pd.DataFrame
    lineage_sha256: str
    manifest_sha256: str
    request_sha256: str
    observed_chunks: int
    assignment_rows: int


def specialist_technical_columns(
    path: Path,
    config: SwingSpecialistResearchConfig,
) -> tuple[str, ...]:
    available = canonical_artifact_columns(path)
    selected = {
        *_BASE_FEATURE_COLUMNS,
        *(
            feature
            for features in config.feature_profiles.values()
            for feature in features
        ),
    }
    columns = tuple(column for column in available if column in selected)
    missing = sorted(_BASE_FEATURE_COLUMNS.difference(columns))
    if missing:
        raise DataReadinessError(
            "technical feature artifact is missing KS3 columns: "
            + ", ".join(missing)
        )
    return columns


def load_catalyst_lineage(
    lineage_dir: Path,
    *,
    maximum_process_memory_gib: float,
    memory_guard_headroom_gib: float,
    progress: Callable[[object], None] | None = None,
) -> CatalystLineageData:
    manifest_path = lineage_dir / "_manifest.json"
    try:
        loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(
            f"catalyst lineage manifest is unreadable: {manifest_path}"
        ) from exc
    if not isinstance(loaded, dict):
        raise DataReadinessError("catalyst lineage manifest must be an object")
    manifest = {str(key): value for key, value in loaded.items()}
    if (
        manifest.get("schema") != CATALYST_LINEAGE_MANIFEST_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("failed_chunks") != {}
        or manifest.get("observed_chunks") != manifest.get("requested_chunks")
    ):
        raise DataReadinessError("catalyst lineage bundle is incomplete")
    records = manifest.get("artifacts")
    if not isinstance(records, list) or not records:
        raise DataReadinessError("catalyst lineage has no assignment artifacts")
    coverage_record = manifest.get("coverage")
    if not isinstance(coverage_record, dict):
        raise DataReadinessError("catalyst lineage has no coverage record")
    coverage_path = lineage_dir / "source_coverage.parquet"
    coverage, coverage_manifest = load_canonical_artifact(
        coverage_path,
        expected_type="catalyst_source_coverage",
        allow_research=True,
    )
    if (
        coverage_manifest.get("artifact_sha256")
        != coverage_record.get("sha256")
        or len(coverage) != int(coverage_record.get("rows", -1))
    ):
        raise DataReadinessError("catalyst source coverage lineage mismatch")
    inventory_record = manifest.get("feature_inventory")
    if not isinstance(inventory_record, dict):
        raise DataReadinessError("catalyst lineage has no feature inventory")
    inventory_path = lineage_dir / "feature_inventory.json"
    if file_sha256(inventory_path) != str(
        inventory_record.get("sha256", "")
    ):
        raise DataReadinessError("catalyst feature inventory hash mismatch")

    frames: list[pd.DataFrame] = []
    assignment_dir = (lineage_dir / "assignments").resolve()
    for index, raw_record in enumerate(records, start=1):
        if not isinstance(raw_record, dict):
            raise DataReadinessError("invalid catalyst artifact record")
        record = {str(key): value for key, value in raw_record.items()}
        chunk_id = str(record.get("chunk_id", "")).strip()
        if not chunk_id:
            raise DataReadinessError("catalyst artifact has no chunk identity")
        path = (assignment_dir / f"{chunk_id}.parquet").resolve()
        if path.parent != assignment_dir:
            raise DataReadinessError("catalyst assignment path traversal")
        declared = Path(str(record.get("assignment_path", ""))).resolve()
        if declared != path:
            raise DataReadinessError(
                f"catalyst assignment path mismatch: {chunk_id}"
            )
        frame, child_manifest = load_canonical_artifact(
            path,
            expected_type="catalyst_event_assignments",
            allow_research=True,
            columns=ASSIGNMENT_FEATURE_COLUMNS,
        )
        if (
            child_manifest.get("artifact_sha256")
            != record.get("assignment_sha256")
            or len(frame) != int(record.get("assignment_rows", -1))
        ):
            raise DataReadinessError(
                f"catalyst assignment lineage mismatch: {chunk_id}"
            )
        frames.append(
            frame.loc[
                frame["status"].astype(str).eq("assigned")
            ].copy()
        )
        if index % 100 == 0:
            assert_memory_budget(
                hard_budget_gib=maximum_process_memory_gib,
                headroom_gib=memory_guard_headroom_gib,
                stage=f"KS3 catalyst assignment load {index}",
            )
            if progress is not None:
                progress(
                    {
                        "stage": "catalyst_assignment_load",
                        "loaded_chunks": index,
                        "total_chunks": len(records),
                    }
                )
    assignments = pd.concat(frames, ignore_index=True)
    frames.clear()
    expected_assigned = int(
        manifest.get("assignment_status_counts", {}).get("assigned", -1)
    )
    if len(assignments) != expected_assigned:
        raise DataReadinessError("catalyst assignment bundle row mismatch")
    return CatalystLineageData(
        assignments=assignments,
        coverage=coverage,
        lineage_sha256=_required_sha256(manifest, "lineage_sha256"),
        manifest_sha256=file_sha256(manifest_path),
        request_sha256=_required_sha256(manifest, "request_sha256"),
        observed_chunks=int(manifest["observed_chunks"]),
    )


def join_catalyst_lineage(
    features: pd.DataFrame,
    lineage: CatalystLineageData,
) -> tuple[pd.DataFrame, dict[str, object]]:
    return join_catalyst_aggregates(
        features,
        aggregate_catalyst_lineage(lineage),
    )


def aggregate_catalyst_lineage(
    lineage: CatalystLineageData,
) -> CatalystAggregateLineageData:
    aggregates = aggregate_event_assignments(
        lineage.assignments,
        windows=DEFAULT_EVENT_WINDOWS,
        source_families=("alpaca",),
    )
    return CatalystAggregateLineageData(
        aggregates=aggregates,
        coverage=lineage.coverage,
        lineage_sha256=lineage.lineage_sha256,
        manifest_sha256=lineage.manifest_sha256,
        request_sha256=lineage.request_sha256,
        observed_chunks=lineage.observed_chunks,
        assignment_rows=len(lineage.assignments),
    )


def join_catalyst_aggregates(
    features: pd.DataFrame,
    lineage: CatalystAggregateLineageData,
) -> tuple[pd.DataFrame, dict[str, object]]:
    event_features = apply_event_assignment_features(
        features,
        lineage.aggregates,
        windows=DEFAULT_EVENT_WINDOWS,
        source_families=("alpaca",),
        inplace=True,
    )
    event_features["catalyst_source_complete"] = _coverage_completeness(
        event_features,
        lineage.coverage,
    )
    catalyst_columns = event_feature_columns(
        DEFAULT_EVENT_WINDOWS,
        source_families=("alpaca",),
    )
    incomplete = ~event_features["catalyst_source_complete"]
    event_features.loc[
        incomplete,
        [column for column in catalyst_columns if column != "latest_event_feature_available_at_utc"],
    ] = np.nan
    decision = _strict_utc(
        event_features["decision_time_utc"],
        "decision_time_utc",
    )
    latest_event = pd.to_datetime(
        event_features["latest_event_feature_available_at_utc"],
        utc=True,
        errors="coerce",
    )
    if bool((latest_event > decision).fillna(False).any()):
        raise DataReadinessError(
            "catalyst aggregate contains post-decision evidence"
        )
    feature_available = _strict_utc(
        event_features["feature_available_at_utc"],
        "feature_available_at_utc",
    )
    event_features["feature_available_at_utc"] = pd.concat(
        [feature_available, latest_event],
        axis=1,
    ).max(axis=1)
    aggregate_columns = [
        "decision_id",
        *catalyst_columns,
        "catalyst_source_complete",
    ]
    aggregate_sha256 = _bounded_frame_sha256(
        event_features.loc[:, aggregate_columns],
        sort_by=("decision_id",),
    )
    audit = {
        "assignment_rows": lineage.assignment_rows,
        "assigned_rows": lineage.assignment_rows,
        "feature_rows": len(event_features),
        "source_complete_rows": int(
            event_features["catalyst_source_complete"].sum()
        ),
        "source_incomplete_rows": int(incomplete.sum()),
        "rows_with_event_3d": int(
            pd.to_numeric(
                event_features["event_count_3d"],
                errors="coerce",
            )
            .fillna(0)
            .gt(0)
            .sum()
        ),
        "event_aggregate_sha256": aggregate_sha256,
        "catalyst_lineage_sha256": lineage.lineage_sha256,
    }
    return event_features, audit


def build_swing_specialist_dataset(
    features: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    strategy_id: str,
    dataset_config: SwingDatasetConfig,
    strategy_policy: SwingStrategyLabelPolicy,
    research_config: SwingSpecialistResearchConfig,
    catalyst_audit: Mapping[str, object],
) -> tuple[pd.DataFrame, CanonicalAuditReport, dict[str, object]]:
    if strategy_id not in STRATEGY_IDS:
        raise DataReadinessError(f"unsupported swing strategy: {strategy_id}")
    labels = build_swing_strategy_labels(
        features,
        benchmark_bars,
        dataset_config=dataset_config,
        policy=strategy_policy,
        strategy_ids=(strategy_id,),
    )
    label_audit = audit_swing_strategy_labels(
        features,
        benchmark_bars,
        labels,
        dataset_config=dataset_config,
        policy=strategy_policy,
        strategy_ids=(strategy_id,),
    )
    label_audit.raise_for_failure()
    eligible = labels["strategy_label_eligible"].fillna(False).astype(bool)
    eligible_labels = labels.loc[eligible].copy()
    if eligible_labels.empty:
        raise DataReadinessError(
            f"{strategy_id} has no training-eligible rows"
        )
    normalized_ticker = features["ticker"].astype(str).str.upper().str.strip()
    normalized_security = features["security_id"].astype(str).str.strip()
    if (
        not normalized_ticker.equals(features["ticker"].astype(str))
        or not normalized_security.equals(features["security_id"].astype(str))
    ):
        raise DataReadinessError(
            "specialist source identities must already be normalized"
        )
    joined = eligible_labels.merge(
        features,
        on=list(_LABEL_JOIN_KEYS),
        how="left",
        validate="one_to_one",
        suffixes=("", "_feature"),
    )
    if len(joined) != len(eligible_labels):
        raise DataReadinessError(
            f"{strategy_id} feature/label join changed row count"
        )
    if bool(joined["feature_available_at_utc_feature"].isna().any()):
        raise DataReadinessError(
            f"{strategy_id} has labels without source features"
        )
    joined["source_decision_group_id"] = joined[
        "decision_group_id_feature"
    ]
    joined["decision_group_id"] = joined["strategy_decision_group_id"]
    joined["feature_available_at_utc"] = joined[
        "feature_available_at_utc_feature"
    ]
    joined["feature_eligible"] = joined["setup_eligible"]
    joined["label_eligible"] = joined["strategy_label_eligible"]
    joined["horizon_sessions"] = joined["strategy_horizon_sessions"]
    horizon = int(
        pd.to_numeric(
            joined["strategy_horizon_sessions"],
            errors="raise",
        ).iloc[0]
    )
    aliases = {
        f"future_gross_return_{horizon}d": "strategy_gross_return",
        f"future_net_return_{horizon}d": "strategy_net_return",
        f"future_spy_return_{horizon}d": "strategy_spy_return",
        f"future_qqq_return_{horizon}d": "strategy_qqq_return",
        f"future_sector_return_{horizon}d": "strategy_sector_return",
        f"future_excess_return_{horizon}d_vs_spy": "strategy_excess_return_vs_spy",
        f"future_excess_return_{horizon}d_vs_qqq": "strategy_excess_return_vs_qqq",
        f"future_excess_return_{horizon}d_vs_sector": "strategy_excess_return_vs_sector",
        f"target_net_positive_{horizon}d": "strategy_target",
    }
    for target, source_column in aliases.items():
        joined[target] = joined[source_column]
    joined["label_material_sha256"] = joined[
        "strategy_label_material_sha256"
    ]
    joined["label_source_reconciliation_sha256"] = joined[
        "strategy_label_reconciliation_sha256"
    ]
    joined["label_source_reconciliation_errors"] = joined[
        "strategy_label_reconciliation_errors"
    ]
    joined["dataset_label_config_sha256"] = joined[
        "strategy_label_policy_sha256"
    ]
    joined["dataset_label_policy_json"] = json.dumps(
        strategy_policy.strategies[strategy_id].model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
    )
    joined["specialist_research_policy_sha256"] = research_config.sha256()
    joined["specialist_dataset_schema_version"] = (
        SPECIALIST_DATASET_SCHEMA
    )
    joined["catalyst_lineage_sha256"] = str(
        catalyst_audit["catalyst_lineage_sha256"]
    )
    joined["event_aggregate_sha256"] = str(
        catalyst_audit["event_aggregate_sha256"]
    )
    joined["dollar_volume"] = (
        pd.to_numeric(joined["close"], errors="coerce")
        * pd.to_numeric(joined["volume"], errors="coerce")
    )
    joined["strategy_dataset_row_id"] = _row_identities(joined)
    joined = _drop_merge_duplicates(joined)
    audit = _audit_specialist_dataset(joined, strategy_id=strategy_id)
    audit.raise_for_failure()
    summary = {
        "strategy_id": strategy_id,
        "source_rows": len(features),
        "setup_eligible_rows": int(
            labels["setup_eligible"].fillna(False).astype(bool).sum()
        ),
        "label_eligible_rows": len(joined),
        "positive_rows": int(
            pd.to_numeric(joined["strategy_target"], errors="coerce")
            .eq(1)
            .sum()
        ),
        "tickers": int(joined["ticker"].nunique()),
        "sessions": int(joined["session_date_et"].nunique()),
        "first_decision_time_utc": _iso(
            pd.to_datetime(joined["decision_time_utc"], utc=True).min()
        ),
        "last_decision_time_utc": _iso(
            pd.to_datetime(joined["decision_time_utc"], utc=True).max()
        ),
        "label_audit": [
            check.model_dump()
            for check in label_audit.checks
        ],
    }
    return joined, audit, summary


def build_swing_specialist_dataset_bundle(
    features: pd.DataFrame,
    benchmark_bars: pd.DataFrame,
    *,
    out_dir: Path,
    dataset_config: SwingDatasetConfig,
    strategy_policy: SwingStrategyLabelPolicy,
    research_config: SwingSpecialistResearchConfig,
    catalyst_audit: Mapping[str, object],
    input_hashes: Mapping[str, str],
    progress: Callable[[object], None] | None = None,
) -> dict[str, object]:
    request = {
        "schema": SPECIALIST_DATASET_BUNDLE_SCHEMA,
        "dataset_config_sha256": dataset_config.label_config_sha256(),
        "strategy_label_policy_sha256": strategy_policy.sha256(),
        "specialist_research_policy_sha256": research_config.sha256(),
        "inputs": dict(sorted(input_hashes.items())),
    }
    request_sha256 = _json_sha256(request)
    request["request_sha256"] = request_sha256
    out_dir.mkdir(parents=True, exist_ok=True)
    _write_or_validate_json(out_dir / "_request.json", request)
    strategy_dir = out_dir / "strategies"
    strategy_dir.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    failures: dict[str, str] = {}
    for strategy_id in STRATEGY_IDS:
        target = strategy_dir / (
            strategy_id.lower().replace(".", "_") + ".parquet"
        )
        try:
            existing = _load_existing_dataset(
                target,
                strategy_id=strategy_id,
                request_sha256=request_sha256,
            )
            if existing is not None:
                records.append(existing)
                if progress is not None:
                    progress(
                        {
                            "strategy_id": strategy_id,
                            "status": "resumed",
                            "rows": existing["label_eligible_rows"],
                        }
                    )
                continue
            dataset, audit, summary = build_swing_specialist_dataset(
                features,
                benchmark_bars,
                strategy_id=strategy_id,
                dataset_config=dataset_config,
                strategy_policy=strategy_policy,
                research_config=research_config,
                catalyst_audit=catalyst_audit,
            )
            manifest = write_canonical_artifact(
                dataset,
                target,
                artifact_type=SPECIALIST_ARTIFACT_TYPE,
                audit=audit,
                inputs={
                    **dict(input_hashes),
                    "request_sha256": request_sha256,
                    "strategy_id": strategy_id,
                    "source_rows": str(summary["source_rows"]),
                    "setup_eligible_rows": str(
                        summary["setup_eligible_rows"]
                    ),
                    "strategy_label_policy_sha256": (
                        strategy_policy.strategy_sha256(strategy_id)
                    ),
                },
                production_ready=False,
            )
            record = {
                **summary,
                "path": str(target.resolve()),
                "sha256": str(manifest["artifact_sha256"]),
                "manifest_sha256": file_sha256(manifest_path_for(target)),
            }
            records.append(record)
            if progress is not None:
                progress({"status": "built", **record})
            del dataset
        except (
            DataReadinessError,
            FileNotFoundError,
            OSError,
            ValueError,
        ) as exc:
            failures[strategy_id] = f"{type(exc).__name__}: {exc}"
            if progress is not None:
                progress(
                    {
                        "strategy_id": strategy_id,
                        "status": "failed",
                        "error": failures[strategy_id],
                    }
                )
        finally:
            gc.collect()
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=research_config.maximum_process_memory_gib,
                headroom_gib=research_config.memory_guard_headroom_gib,
                stage=f"KS3 specialist dataset {strategy_id}",
            )
    try:
        assert_peak_memory_budget(
            hard_budget_gib=research_config.maximum_process_memory_gib,
            headroom_gib=research_config.memory_guard_headroom_gib,
            stage="KS3 specialist dataset bundle",
        )
    except DataReadinessError as exc:
        failures["memory_budget"] = str(exc)
    status = "complete" if not failures and len(records) == len(
        STRATEGY_IDS
    ) else "incomplete"
    result: dict[str, object] = {
        "schema": SPECIALIST_DATASET_BUNDLE_SCHEMA,
        "status": status,
        "request_sha256": request_sha256,
        "requested_strategies": len(STRATEGY_IDS),
        "observed_strategies": len(records),
        "failed_strategies": failures,
        "rows": sum(
            _object_int(record["label_eligible_rows"])
            for record in records
        ),
        "artifacts": sorted(
            records,
            key=lambda record: str(record["strategy_id"]),
        ),
        "catalyst_audit": dict(catalyst_audit),
        "memory": memory_audit(
            hard_budget_gib=research_config.maximum_process_memory_gib,
            headroom_gib=research_config.memory_guard_headroom_gib,
        ).to_record(),
        "completed_at_utc": datetime.now(UTC).isoformat(),
        "production_ready": False,
    }
    _atomic_json(out_dir / "_status.json", result)
    if status == "complete":
        _atomic_json(out_dir / "_manifest.json", result)
    return result


def _coverage_completeness(
    decisions: pd.DataFrame,
    coverage: pd.DataFrame,
) -> pd.Series:
    required = {
        "security_id",
        "requested_start_utc",
        "requested_end_utc",
        "status",
        "coverage_state",
        "missingness_known",
        "training_eligible",
    }
    missing = sorted(required.difference(coverage.columns))
    if missing:
        raise DataReadinessError(
            "catalyst coverage is missing columns: " + ", ".join(missing)
        )
    intervals = coverage.copy()
    intervals["requested_start_utc"] = _strict_utc(
        intervals["requested_start_utc"],
        "coverage.requested_start_utc",
    )
    intervals["requested_end_utc"] = _strict_utc(
        intervals["requested_end_utc"],
        "coverage.requested_end_utc",
    )
    intervals["_eligible"] = (
        intervals["status"].astype(str).eq("observed")
        & intervals["coverage_state"]
        .astype(str)
        .isin({"observed_complete", "observed_empty"})
        & intervals["missingness_known"].fillna(False).astype(bool)
        & intervals["training_eligible"].fillna(False).astype(bool)
    )
    if bool(
        intervals["requested_end_utc"].le(
            intervals["requested_start_utc"]
        ).any()
    ):
        raise DataReadinessError("catalyst coverage has invalid intervals")
    result = pd.Series(False, index=decisions.index)
    decision_time = _strict_utc(
        decisions["decision_time_utc"],
        "decision_time_utc",
    )
    decision_security = decisions["security_id"].astype(str)
    for security_id, positions in decisions.groupby(
        decision_security,
        sort=False,
    ).indices.items():
        rows = intervals.loc[
            intervals["security_id"].astype(str).eq(str(security_id))
        ].sort_values("requested_start_utc")
        if rows.empty:
            continue
        starts = pd.DatetimeIndex(rows["requested_start_utc"]).as_unit(
            "ns"
        ).asi8
        ends = pd.DatetimeIndex(rows["requested_end_utc"]).as_unit("ns").asi8
        if len(starts) > 1 and bool((starts[1:] < ends[:-1]).any()):
            raise DataReadinessError(
                f"overlapping catalyst coverage intervals: {security_id}"
            )
        selected_positions = np.asarray(positions, dtype=np.int64)
        times = (
            pd.DatetimeIndex(decision_time.iloc[selected_positions])
            .as_unit("ns")
            .asi8
        )
        interval_indices = np.searchsorted(starts, times, side="right") - 1
        valid = interval_indices >= 0
        bounded = np.zeros(len(times), dtype=bool)
        bounded[valid] = (
            times[valid] < ends[interval_indices[valid]]
        )
        eligible_values = rows["_eligible"].to_numpy(dtype=bool)
        bounded[valid] &= eligible_values[interval_indices[valid]]
        result.iloc[selected_positions] = bounded
    return result


def _audit_specialist_dataset(
    frame: pd.DataFrame,
    *,
    strategy_id: str,
) -> CanonicalAuditReport:
    required = {
        "strategy_id",
        "strategy_dataset_row_id",
        "ticker",
        "security_id",
        "decision_time_utc",
        "feature_available_at_utc",
        "label_available_at_utc",
        "setup_eligible",
        "strategy_label_eligible",
        "strategy_target",
        "strategy_gross_return",
        "strategy_execution_cost_fraction",
        "strategy_net_return",
        "strategy_spy_return",
        "strategy_qqq_return",
        "strategy_sector_return",
        "strategy_excess_return_vs_spy",
        "strategy_excess_return_vs_qqq",
        "strategy_excess_return_vs_sector",
        "strategy_label_material_sha256",
        "strategy_label_reconciliation_sha256",
        "strategy_label_reconciliation_errors",
    }
    missing = sorted(required.difference(frame.columns))
    decision = pd.to_datetime(
        frame.get("decision_time_utc"),
        utc=True,
        errors="coerce",
    )
    feature = pd.to_datetime(
        frame.get("feature_available_at_utc"),
        utc=True,
        errors="coerce",
    )
    label = pd.to_datetime(
        frame.get("label_available_at_utc"),
        utc=True,
        errors="coerce",
    )
    timestamp_errors = int(
        (
            decision.isna()
            | feature.isna()
            | label.isna()
            | feature.gt(decision)
            | label.le(decision)
        ).sum()
    )
    gross = pd.to_numeric(
        frame.get("strategy_gross_return"),
        errors="coerce",
    )
    cost = pd.to_numeric(
        frame.get("strategy_execution_cost_fraction"),
        errors="coerce",
    )
    net = pd.to_numeric(
        frame.get("strategy_net_return"),
        errors="coerce",
    )
    cost_errors = int(
        (
            ~pd.Series(
                np.isclose(
                    net,
                    gross - cost,
                    rtol=1e-10,
                    atol=1e-12,
                    equal_nan=False,
                ),
                index=frame.index,
            )
        ).sum()
    )
    benchmark_errors = int(
        frame[
            [
                "strategy_spy_return",
                "strategy_qqq_return",
                "strategy_sector_return",
            ]
        ]
        .apply(pd.to_numeric, errors="coerce")
        .isna()
        .any(axis=1)
        .sum()
    )
    identity_errors = int(
        frame["strategy_dataset_row_id"].astype(str).duplicated().sum()
    )
    strategy_errors = int(
        frame["strategy_id"].astype(str).ne(strategy_id).sum()
    )
    eligibility_errors = int(
        (
            ~frame["setup_eligible"].fillna(False).astype(bool)
            | ~frame["strategy_label_eligible"]
            .fillna(False)
            .astype(bool)
        ).sum()
    )
    target = pd.to_numeric(frame["strategy_target"], errors="coerce")
    target_errors = int((~target.isin({0, 1})).sum())
    reconciliation = pd.to_numeric(
        frame["strategy_label_reconciliation_errors"],
        errors="coerce",
    )
    reconciliation_errors = int(
        reconciliation.isna().sum()
        + reconciliation.ne(0).sum()
    )
    catalyst_errors = 0
    if strategy_id in {
        "SWING.CATALYST_DRIFT.5D.V1",
        "SWING.SHORT_TERM_REVERSAL.3D.V1",
    }:
        catalyst_errors = int(
            (~frame["catalyst_source_complete"].astype(bool)).sum()
        )
    checks = (
        _check(
            "specialist_schema",
            len(missing),
            len(frame),
            "required specialist columns are present",
        ),
        _check(
            "specialist_rows",
            int(frame.empty),
            len(frame),
            "eligible strategy dataset is non-empty",
        ),
        _check(
            "specialist_identity",
            identity_errors + strategy_errors,
            len(frame),
            "strategy row identity is unique and strategy-bound",
        ),
        _check(
            "specialist_causality",
            timestamp_errors,
            len(frame),
            "features precede decisions and labels follow decisions",
        ),
        _check(
            "specialist_eligibility",
            eligibility_errors,
            len(frame),
            "published rows pass setup and label gates",
        ),
        _check(
            "specialist_target",
            target_errors,
            len(frame),
            "published strategy targets are binary",
        ),
        _check(
            "specialist_cost_once",
            cost_errors,
            len(frame),
            "net return equals gross return minus one bound cost",
        ),
        _check(
            "specialist_benchmarks",
            benchmark_errors,
            len(frame),
            "SPY, QQQ, and sector intervals are complete",
        ),
        _check(
            "specialist_label_reconciliation",
            reconciliation_errors,
            len(frame),
            "KS2 exact-path reconciliation remains clean",
        ),
        _check(
            "specialist_catalyst_coverage",
            catalyst_errors,
            len(frame),
            "catalyst-dependent rows have verified source completeness",
        ),
    )
    return CanonicalAuditReport(checks=checks)


def _load_existing_dataset(
    path: Path,
    *,
    strategy_id: str,
    request_sha256: str,
) -> dict[str, object] | None:
    exists = path.exists()
    manifest_exists = manifest_path_for(path).exists()
    if not exists and not manifest_exists:
        return None
    if exists != manifest_exists:
        raise DataReadinessError(
            f"orphan specialist dataset artifact: {path}"
        )
    frame, manifest = load_canonical_artifact(
        path,
        expected_type=SPECIALIST_ARTIFACT_TYPE,
        allow_research=True,
    )
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("request_sha256") != request_sha256
        or inputs.get("strategy_id") != strategy_id
    ):
        raise DataReadinessError(
            f"existing specialist dataset lineage mismatch: {path}"
        )
    audit = _audit_specialist_dataset(frame, strategy_id=strategy_id)
    audit.raise_for_failure()
    manifest_inputs = manifest.get("inputs")
    if not isinstance(manifest_inputs, dict):
        raise DataReadinessError(
            f"existing specialist dataset has no inputs: {path}"
        )
    return {
        "strategy_id": strategy_id,
        "source_rows": _object_int(
            manifest_inputs.get("source_rows", 0)
        ),
        "setup_eligible_rows": _object_int(
            manifest_inputs.get("setup_eligible_rows", len(frame))
        ),
        "label_eligible_rows": len(frame),
        "positive_rows": int(
            pd.to_numeric(frame["strategy_target"], errors="coerce")
            .eq(1)
            .sum()
        ),
        "tickers": int(frame["ticker"].nunique()),
        "sessions": int(frame["session_date_et"].nunique()),
        "first_decision_time_utc": _iso(
            pd.to_datetime(frame["decision_time_utc"], utc=True).min()
        ),
        "last_decision_time_utc": _iso(
            pd.to_datetime(frame["decision_time_utc"], utc=True).max()
        ),
        "path": str(path.resolve()),
        "sha256": str(manifest["artifact_sha256"]),
        "manifest_sha256": file_sha256(manifest_path_for(path)),
        "resumed": True,
    }


def _drop_merge_duplicates(frame: pd.DataFrame) -> pd.DataFrame:
    removable = [
        column
        for column in frame
        if column.endswith("_feature")
        or column == "_strategy_source_row_id"
    ]
    return frame.drop(columns=removable).sort_values(
        ["decision_time_utc", "ticker"],
        kind="stable",
    ).reset_index(drop=True)


def _row_identities(frame: pd.DataFrame) -> pd.Series:
    return pd.Series(
        [
            hashlib.sha256(
                "|".join(
                    (
                        str(strategy_id),
                        str(security_id),
                        pd.Timestamp(decision).isoformat(),
                    )
                ).encode("utf-8")
            ).hexdigest()
            for strategy_id, security_id, decision in zip(
                frame["strategy_id"],
                frame["security_id"],
                pd.to_datetime(frame["decision_time_utc"], utc=True),
                strict=True,
            )
        ],
        index=frame.index,
        dtype="string",
    )


def _bounded_frame_sha256(
    frame: pd.DataFrame,
    *,
    sort_by: Sequence[str],
    chunk_rows: int = 50_000,
) -> str:
    ordered = frame.sort_values(list(sort_by), kind="stable").reset_index(
        drop=True
    )
    digest = hashlib.sha256()
    digest.update("\n".join(ordered.columns).encode("utf-8"))
    for start in range(0, len(ordered), chunk_rows):
        chunk = ordered.iloc[start : start + chunk_rows]
        digest.update(
            pd.util.hash_pandas_object(
                chunk,
                index=False,
            ).to_numpy(dtype=np.uint64, copy=False).tobytes()
        )
    return digest.hexdigest()


def _required_sha256(
    record: Mapping[str, object],
    key: str,
) -> str:
    value = str(record.get(key, ""))
    if len(value) != 64:
        raise DataReadinessError(f"invalid {key}")
    return value


def _object_int(value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, str)):
        raise DataReadinessError("expected integer artifact metadata")
    try:
        return int(value)
    except ValueError as exc:
        raise DataReadinessError(
            "expected integer artifact metadata"
        ) from exc


def _check(
    name: str,
    failures: int,
    rows: int,
    detail: str,
) -> CanonicalAuditCheck:
    return CanonicalAuditCheck(
        name=name,
        status="pass" if failures == 0 else "fail",
        failures=failures,
        rows_checked=rows,
        detail=detail,
    )


def _strict_utc(values: pd.Series, name: str) -> pd.Series:
    parsed = pd.to_datetime(values, utc=True, errors="coerce")
    if bool(parsed.isna().any()):
        raise DataReadinessError(f"{name} contains invalid timestamps")
    return parsed


def _json_sha256(payload: Mapping[str, object]) -> str:
    encoded = json.dumps(
        dict(payload),
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _write_or_validate_json(
    path: Path,
    payload: Mapping[str, object],
) -> None:
    if path.exists():
        try:
            existing = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise DataReadinessError(
                f"specialist request is unreadable: {path}"
            ) from exc
        if existing != dict(payload):
            raise DataReadinessError(
                f"specialist request lineage mismatch: {path}"
            )
        return
    _atomic_json(path, payload)


def _atomic_json(path: Path, payload: Mapping[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(dict(payload), indent=2, sort_keys=True),
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _iso(value: object) -> str | None:
    if value is None or pd.isna(value):
        return None
    return str(pd.Timestamp(value).isoformat())
