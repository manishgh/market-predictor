"""Resumable authority publisher for the fixed-cohort intraday bar dataset."""
from __future__ import annotations



import json
import os
import shutil
import threading
import time
import uuid
from collections.abc import Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ProcessPoolExecutor, wait
from dataclasses import dataclass
from datetime import UTC, datetime
from multiprocessing import parent_process
from pathlib import Path
from typing import Any, Final, cast

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_bar_features import (
    INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
    INTRADAY_BAR_MODEL_FEATURE_COLUMNS,
    INTRADAY_BAR_MODEL_FEATURES_SHA256,
    build_causal_intraday_bar_features,
)
from market_predictor.edge_rebuild.intraday_bar_labels import (
    INTRADAY_BAR_LABEL_SCHEMA_VERSION,
    build_exact_intraday_bar_labels,
)
from market_predictor.edge_rebuild.intraday_bar_only_five_minute import (
    load_complete_selected_session_five_minute_projection,
)
from market_predictor.edge_rebuild.intraday_contract_lineage import (
    DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
)
from market_predictor.intraday.datasets.publisher import (
    _activation_abstention_reason,
    _Artifact,
    _benchmark_artifact_index,
    _load_benchmark_session,
    _load_stock_session_batch,
    _membership_for_pair,
    _stock_artifact_index,
    _VerifiedInputs,
    _verify_inputs,
)
from market_predictor.edge_rebuild.intraday_history import json_sha256
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.edge_rebuild.volume_bars import build_causal_volume_bars
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.core.errors import DataReadinessError

INTRADAY_BAR_DATASET_SCHEMA: Final = "edge_rebuild.intraday_bar_dataset.v1"
INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.intraday_bar_dataset_authority.v1"
)
INTRADAY_BAR_TRANSFORMATION_SCHEMA: Final = (
    "edge_rebuild.intraday_bar_transformation.v1"
)
MEMORY_HARD_BUDGET_GIB: Final = 4.0
MEMORY_HEADROOM_GIB: Final = 0.75
WORKER_MEMORY_HARD_BUDGET_GIB: Final = 2.0
WORKER_MEMORY_HEADROOM_GIB: Final = 0.25
_METADATA_FILES: Final = frozenset({"_request.json", "_manifest.json", "_authority.json"})
_UNIT_FILES: Final = frozenset({"rows.parquet", "audit.json", "_unit.json"})
_LABEL_COLUMNS: Final = frozenset(
    {
        "label_schema_version",
        "label_eligible",
        "label_ineligible_reason",
        "decision_group_id",
        "entry_time_utc",
        "entry_bar_end_utc",
        "entry_price",
        "target_price",
        "stop_price",
        "exit_time_utc",
        "exit_bar_end_utc",
        "label_available_at_utc",
        "exit_price",
        "holding_minutes",
        "barrier_label",
        "label_outcome",
        "label_outcome_reason",
        "target_hit",
        "stop_hit",
        "timeout",
        "gross_return",
        "cost",
        "net_return",
        "spy_return",
        "qqq_return",
        "sector_return",
        "spy_excess_return",
        "qqq_excess_return",
        "sector_excess_return",
        "rank_label",
        "rank_percentile",
        "ranking_group_size",
    }
)


@dataclass(frozen=True, slots=True)
class _SessionWorkerContext:
    work: Path
    request_sha256: str
    transformation_sha256: str
    verified: _VerifiedInputs
    stock_index: Mapping[tuple[str, str], _Artifact]
    benchmark_index: Mapping[str, tuple[_Artifact, ...]]
    five_minute_projection_directory: Path
    five_minute_files: Mapping[str, _ProjectionArtifact]


@dataclass(frozen=True, slots=True)
class _ProjectionArtifact:
    path: Path
    sha256: str


_SESSION_WORKER_CONTEXT: _SessionWorkerContext | None = None
_SESSION_WORKER_VERIFIED_PROJECTIONS: set[Path] = set()


def publish_intraday_bar_dataset(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    five_minute_projection_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
    intraday_contract_lineage_path: Path = DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
    max_sessions_per_invocation: int | None = None,
    session_workers: int = 1,
) -> dict[str, Any]:
    """Publish one immutable, restartable unit per exchange session."""

    if (
        max_sessions_per_invocation is not None
        and max_sessions_per_invocation < 1
    ):
        raise ValueError("max_sessions_per_invocation must be positive")
    if session_workers not in {1, 2}:
        raise ValueError("session_workers must be 1 or 2")
    _require_path_isolation(
        output_directory,
        (
            selection_directory,
            stock_collection_directory,
            stock_coverage_directory,
            benchmark_collection_directory,
            membership_authority_directory,
            five_minute_projection_directory,
            strategy_contract_path,
            intraday_contract_lineage_path,
        ),
    )
    verified = _verify_inputs(
        selection_directory=selection_directory,
        stock_collection_directory=stock_collection_directory,
        stock_coverage_directory=stock_coverage_directory,
        benchmark_collection_directory=benchmark_collection_directory,
        membership_authority_directory=membership_authority_directory,
        strategy_contract=strategy_contract,
        strategy_contract_path=strategy_contract_path,
        intraday_contract_lineage_path=intraday_contract_lineage_path,
    )
    projection = load_complete_selected_session_five_minute_projection(
        five_minute_projection_directory
    )
    _validate_projection_lineage(
        projection,
        verified_parent_lineage=verified.parent_lineage,
        selection_directory=selection_directory,
        strategy_contract_path=strategy_contract_path,
    )
    parent_lineage = {
        **verified.parent_lineage,
        "five_minute_projection_authority_sha256": file_sha256(
            five_minute_projection_directory / "_authority.json"
        ),
        "five_minute_projection_manifest_sha256": file_sha256(
            five_minute_projection_directory / "_manifest.json"
        ),
        "five_minute_projection_inventory_sha256": str(
            projection["file_inventory_sha256"]
        ),
    }
    selection = verified.selection.loc[
        ~verified.selection["ticker"].isin(verified.excluded_tickers)
    ].copy()
    sessions = sorted(selection["session_date_et"].astype(str).unique())
    if not sessions:
        raise DataReadinessError("intraday bar dataset has no usable sessions")
    transformation = _transformation_identity()
    request_payload = {
        "schema": INTRADAY_BAR_DATASET_SCHEMA,
        "selection_directory": str(selection_directory.resolve()),
        "stock_collection_directory": str(stock_collection_directory.resolve()),
        "stock_coverage_directory": str(stock_coverage_directory.resolve()),
        "benchmark_collection_directory": str(benchmark_collection_directory.resolve()),
        "membership_authority_directory": str(membership_authority_directory.resolve()),
        "five_minute_projection_directory": str(
            five_minute_projection_directory.resolve()
        ),
        "strategy_contract_path": str(strategy_contract_path.resolve()),
        "intraday_contract_lineage_path": str(
            intraday_contract_lineage_path.resolve()
        ),
        "intraday_contract_lineage_file_sha256": file_sha256(
            intraday_contract_lineage_path
        ),
        "strategy_contract_sha256": verified.contract_sha256,
        "parent_lineage": parent_lineage,
        "parent_lineage_sha256": json_sha256(parent_lineage),
        "feature_schema_version": INTRADAY_BAR_FEATURE_SCHEMA_VERSION,
        "ordered_feature_names": list(INTRADAY_BAR_MODEL_FEATURE_COLUMNS),
        "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
        "label_schema_version": INTRADAY_BAR_LABEL_SCHEMA_VERSION,
        "transformation": transformation,
        "transformation_sha256": transformation["sha256"],
        "processing_unit": "one_exchange_session",
        "decision_clock": "fixed_five_minute_cohort_after_activation",
        "maximum_session_workers": 2,
        "memory_hard_budget_gib": MEMORY_HARD_BUDGET_GIB,
        "planned_sessions": sessions,
    }
    request_sha256 = json_sha256(request_payload)
    request = {**request_payload, "request_sha256": request_sha256}
    if output_directory.exists():
        manifest = load_complete_intraday_bar_dataset(output_directory)
        if manifest.get("request_sha256") != request_sha256:
            raise DataReadinessError(
                f"published intraday bar dataset is immutable: {output_directory}"
            )
        return manifest

    work = output_directory.with_name(
        f".{output_directory.name}.{request_sha256[:16]}.work"
    )
    recovered = _recover_complete_work_directory(
        work,
        output_directory=output_directory,
        request=request,
    )
    if recovered is not None:
        return recovered
    _prepare_work_directory(work, request)
    stock_index = _stock_artifact_index(verified.stock_artifacts)
    benchmark_index = _benchmark_artifact_index(verified.benchmark_artifacts)
    five_minute_files = _projection_bar_files(
        five_minute_projection_directory,
        projection,
    )
    _guard_memory("intraday bar dataset publication start")
    pending: list[tuple[str, pd.DataFrame, str]] = []
    for session_date in sessions:
        unit = work / "sessions" / f"session_date_et={session_date}"
        selected = selection.loc[
            selection["session_date_et"].astype(str).eq(session_date)
        ].copy()
        expected_identity = _session_request_sha256(
            request_sha256,
            session_date,
            selected,
        )
        if unit.exists():
            _verify_session_unit(
                unit,
                session_date=session_date,
                request_sha256=request_sha256,
                session_request_sha256=expected_identity,
                transformation_sha256=str(transformation["sha256"]),
            )
            continue
        pending.append((session_date, selected, expected_identity))
    if max_sessions_per_invocation is not None:
        pending = pending[:max_sessions_per_invocation]
    worker_context = _SessionWorkerContext(
        work=work,
        request_sha256=request_sha256,
        transformation_sha256=str(transformation["sha256"]),
        verified=verified,
        stock_index=stock_index,
        benchmark_index=benchmark_index,
        five_minute_projection_directory=five_minute_projection_directory,
        five_minute_files=five_minute_files,
    )
    worker_results: list[dict[str, Any]] = []
    if session_workers == 1:
        _initialize_session_worker(worker_context)
        for completed_index, task in enumerate(pending, start=1):
            worker_results.append(_process_session_task(task))
            if completed_index % 25 == 0:
                release_process_memory()
            _guard_memory("intraday bar dataset completed session unit")
    else:
        worker_results = _run_bounded_process_tasks(
            pending,
            context=worker_context,
            session_workers=session_workers,
        )
    publication_memory = _publication_memory_audit(
        worker_results,
        workers_are_children=session_workers > 1,
    )
    processed_this_invocation = len(pending)
    completed = sum(
        (
            work
            / "sessions"
            / f"session_date_et={planned_session}"
            / "_unit.json"
        ).is_file()
        for planned_session in sessions
    )
    if completed < len(sessions):
        return {
            "state": "work_incomplete",
            "request_sha256": request_sha256,
            "work_directory": str(work),
            "summary": {
                "planned_sessions": len(sessions),
                "completed_sessions": completed,
                "processed_this_invocation": processed_this_invocation,
                "session_workers": session_workers,
                "memory": publication_memory,
            },
        }

    _revalidate_parent_inputs(
        selection_directory=selection_directory,
        stock_collection_directory=stock_collection_directory,
        stock_coverage_directory=stock_coverage_directory,
        benchmark_collection_directory=benchmark_collection_directory,
        membership_authority_directory=membership_authority_directory,
        five_minute_projection_directory=five_minute_projection_directory,
        strategy_contract=strategy_contract,
        strategy_contract_path=strategy_contract_path,
        intraday_contract_lineage_path=intraday_contract_lineage_path,
        expected_parent_lineage=parent_lineage,
        expected_transformation=transformation,
    )

    unit_records = [
        _verify_session_unit(
            work / "sessions" / f"session_date_et={session_date}",
            session_date=session_date,
            request_sha256=request_sha256,
            transformation_sha256=str(transformation["sha256"]),
            session_request_sha256=_session_request_sha256(
                request_sha256,
                session_date,
                selection.loc[
                    selection["session_date_et"].astype(str).eq(session_date)
                ],
            ),
        )
        for session_date in sessions
    ]
    total_rows = sum(int(record["rows"]) for record in unit_records)
    eligible_rows = sum(int(record["dataset_eligible_rows"]) for record in unit_records)
    manifest = {
        **request,
        "state": "complete",
        "created_at_utc": datetime.now(UTC).isoformat(),
        "session_units": unit_records,
        "session_unit_inventory_sha256": json_sha256(unit_records),
        "summary": {
            "planned_sessions": len(sessions),
            "completed_sessions": len(unit_records),
            "selected_stock_sessions": int(len(selection)),
            "excluded_securities": len(verified.excluded_tickers),
            "source_incomplete_stock_sessions": len(verified.incomplete_pairs),
            "rows": total_rows,
            "dataset_eligible_rows": eligible_rows,
            "memory": publication_memory,
        },
        "training_contract": {
            "eligibility_column": "dataset_eligible",
            "ordered_feature_names": list(INTRADAY_BAR_MODEL_FEATURE_COLUMNS),
            "ordered_feature_sha256": INTRADAY_BAR_MODEL_FEATURES_SHA256,
            "label_columns": sorted(_LABEL_COLUMNS),
        },
    }
    _write_json(work / "_manifest.json", manifest)
    _write_json(
        work / "_authority.json",
        {
            "schema": INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(work / "_manifest.json"),
            "request_sha256": request_sha256,
            "session_unit_inventory_sha256": manifest[
                "session_unit_inventory_sha256"
            ],
            "sessions": len(unit_records),
            "rows": total_rows,
        },
    )
    load_complete_intraday_bar_dataset(work)
    assert_peak_memory_budget(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage="intraday bar dataset publication",
    )
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    work.replace(output_directory)
    return load_complete_intraday_bar_dataset(output_directory)


def load_complete_intraday_bar_dataset(directory: Path) -> dict[str, Any]:
    """Replay authority, exact unit inventory, hashes, schemas, and counts."""

    request = _read_json(directory / "_request.json")
    manifest = _read_json(directory / "_manifest.json")
    authority = _read_json(directory / "_authority.json")
    request_payload = {
        key: value for key, value in request.items() if key != "request_sha256"
    }
    request_sha256 = json_sha256(request_payload)
    sessions = request.get("planned_sessions")
    units = manifest.get("session_units")
    transformation = request.get("transformation")
    transformation_sha256 = str(request.get("transformation_sha256", ""))
    if (
        request.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != INTRADAY_BAR_DATASET_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != INTRADAY_BAR_DATASET_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or not isinstance(sessions, list)
        or not sessions
        or sessions != sorted(set(str(value) for value in sessions))
        or not isinstance(units, list)
        or len(units) != len(sessions)
        or manifest.get("session_unit_inventory_sha256") != json_sha256(units)
        or authority.get("session_unit_inventory_sha256")
        != manifest.get("session_unit_inventory_sha256")
        or not isinstance(transformation, Mapping)
        or transformation.get("sha256") != transformation_sha256
        or transformation != _transformation_identity()
    ):
        raise DataReadinessError(
            f"intraday bar dataset lacks a matching complete authority: {directory}"
        )
    replayed = []
    for session_date in cast(list[str], sessions):
        record = next(
            (
                item
                for item in cast(list[Mapping[str, Any]], units)
                if str(item.get("session_date_et")) == session_date
            ),
            None,
        )
        if record is None:
            raise DataReadinessError("intraday bar dataset omits a planned session")
        replayed.append(
            _verify_session_unit(
                directory / "sessions" / f"session_date_et={session_date}",
                session_date=session_date,
                request_sha256=request_sha256,
                session_request_sha256=str(record.get("session_request_sha256", "")),
                transformation_sha256=str(request.get("transformation_sha256", "")),
            )
        )
    if replayed != units:
        raise DataReadinessError("intraday bar dataset session inventory differs")
    actual = {
        path.relative_to(directory).as_posix()
        for path in directory.rglob("*")
        if path.is_file()
    }
    expected = set(_METADATA_FILES)
    for session_date in cast(list[str], sessions):
        prefix = f"sessions/session_date_et={session_date}"
        expected.update(f"{prefix}/{name}" for name in _UNIT_FILES)
    if actual != expected:
        raise DataReadinessError("intraday bar dataset immutable file set differs")
    total_rows = sum(int(item["rows"]) for item in replayed)
    eligible_rows = sum(int(item["dataset_eligible_rows"]) for item in replayed)
    summary = manifest.get("summary")
    if (
        not isinstance(summary, Mapping)
        or int(summary.get("completed_sessions", -1)) != len(replayed)
        or int(summary.get("rows", -1)) != total_rows
        or int(summary.get("dataset_eligible_rows", -1)) != eligible_rows
        or int(authority.get("sessions", -1)) != len(replayed)
        or int(authority.get("rows", -1)) != total_rows
    ):
        raise DataReadinessError("intraday bar dataset aggregate counts differ")
    return manifest


def _initialize_session_worker(context: _SessionWorkerContext) -> None:
    global _SESSION_WORKER_CONTEXT
    _SESSION_WORKER_CONTEXT = context
    _SESSION_WORKER_VERIFIED_PROJECTIONS.clear()
    owner = parent_process()
    if owner is not None:
        threading.Thread(
            target=_exit_when_parent_stops,
            args=(owner,),
            daemon=True,
            name="intraday-dataset-parent-watchdog",
        ).start()


def _exit_when_parent_stops(owner: Any) -> None:
    while owner.is_alive():
        time.sleep(2.0)
    os._exit(70)


def _process_session_task(task: tuple[str, pd.DataFrame, str]) -> dict[str, Any]:
    context = _SESSION_WORKER_CONTEXT
    if context is None:
        raise RuntimeError("intraday bar session worker is not initialized")
    session_date, session_selection, session_request_sha256 = task
    _build_and_publish_session(
        work=context.work,
        session_date=session_date,
        session_selection=session_selection,
        session_request_sha256=session_request_sha256,
        request_sha256=context.request_sha256,
        transformation_sha256=context.transformation_sha256,
        verified=context.verified,
        stock_index=context.stock_index,
        benchmark_index=context.benchmark_index,
        five_minute_projection_directory=(
            context.five_minute_projection_directory
        ),
        five_minute_file=context.five_minute_files.get(session_date[:7]),
    )
    assert_peak_memory_budget(
        hard_budget_gib=WORKER_MEMORY_HARD_BUDGET_GIB,
        headroom_gib=WORKER_MEMORY_HEADROOM_GIB,
        stage=f"intraday bar dataset worker for {session_date}",
    )
    return {
        "pid": os.getpid(),
        "session_date_et": session_date,
        "memory": memory_audit(
            hard_budget_gib=WORKER_MEMORY_HARD_BUDGET_GIB,
            headroom_gib=WORKER_MEMORY_HEADROOM_GIB,
        ).to_record(),
    }


def _build_and_publish_session(
    *,
    work: Path,
    session_date: str,
    session_selection: pd.DataFrame,
    session_request_sha256: str,
    request_sha256: str,
    transformation_sha256: str,
    verified: _VerifiedInputs,
    stock_index: Mapping[tuple[str, str], _Artifact],
    benchmark_index: Mapping[str, tuple[_Artifact, ...]],
    five_minute_projection_directory: Path,
    five_minute_file: _ProjectionArtifact | None,
) -> None:
    rows, audit = _build_session(
        session_date=session_date,
        session_selection=session_selection,
        verified=verified,
        stock_index=stock_index,
        benchmark_index=benchmark_index,
        five_minute_projection_directory=five_minute_projection_directory,
        five_minute_file=five_minute_file,
    )
    _publish_session_unit(
        work / "sessions" / f"session_date_et={session_date}",
        session_date=session_date,
        request_sha256=request_sha256,
        transformation_sha256=transformation_sha256,
        session_request_sha256=session_request_sha256,
        rows=rows,
        audit=audit,
    )


def _build_session(
    *,
    session_date: str,
    session_selection: pd.DataFrame,
    verified: _VerifiedInputs,
    stock_index: Mapping[tuple[str, str], _Artifact],
    benchmark_index: Mapping[str, tuple[_Artifact, ...]],
    five_minute_projection_directory: Path,
    five_minute_file: _ProjectionArtifact | None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    benchmarks = _load_benchmark_session(
        session_date,
        artifacts=benchmark_index,
        required_tickers=verified.benchmark_tickers,
    )
    activations: list[dict[str, Any]] = []
    membership_parts: list[pd.DataFrame] = []
    pair_reasons: dict[str, str] = {}
    for raw in session_selection.sort_values(
        ["activation_time_utc", "ticker"], kind="stable"
    ).to_dict(orient="records"):
        ticker = str(raw["ticker"])
        membership = _membership_for_pair(
            verified.memberships,
            ticker=ticker,
            session_date=session_date,
        )
        reason = _activation_abstention_reason(
            raw,
            membership,
            maximum_delay_seconds=verified.contract.intraday.decision_finalization_seconds,
        )
        if reason is not None:
            pair_reasons[ticker] = reason
            continue
        activations.append(raw)
        membership_parts.append(membership)
    if not activations:
        return _empty_session_rows(), _session_audit(
            session_date,
            session_selection,
            pair_reasons,
            rows=_empty_session_rows(),
            incomplete_pairs=verified.incomplete_pairs,
        )
    tickers = [str(row["ticker"]) for row in activations]
    stocks = _load_stock_session_batch(
        session_date,
        tickers,
        artifacts=stock_index,
        coverage=verified.coverage,
    )
    five = _load_five_minute_session(
        five_minute_projection_directory,
        five_minute_file,
        session_date=session_date,
        tickers=set(tickers),
    )
    observed_five = set(five["ticker"].astype(str)) if not five.empty else set()
    for ticker in tickers:
        if ticker not in observed_five:
            pair_reasons[ticker] = "missing_selected_five_minute_rows"
    usable = [row for row in activations if str(row["ticker"]) in observed_five]
    if not usable:
        rows = _empty_session_rows()
        return rows, _session_audit(
            session_date,
            session_selection,
            pair_reasons,
            rows=rows,
            incomplete_pairs=verified.incomplete_pairs,
        )
    usable_tickers = {str(row["ticker"]) for row in usable}
    stocks = stocks.loc[stocks["ticker"].astype(str).isin(usable_tickers)].copy()
    five = five.loc[five["ticker"].astype(str).isin(usable_tickers)].copy()
    memberships = pd.concat(
        [
            membership
            for membership in membership_parts
            if str(membership.iloc[0]["ticker"]) in usable_tickers
        ],
        ignore_index=True,
    )
    volume = build_causal_volume_bars(
        stocks,
        pd.DataFrame(usable),
        contract=verified.contract,
        strategy_contract_sha256=verified.contract_sha256,
    )
    completed_tickers = set(volume.bars["ticker"].astype(str))
    for ticker in usable_tickers.difference(completed_tickers):
        pair_reasons[ticker] = "no_completed_volume_bar"
    usable = [row for row in usable if str(row["ticker"]) in completed_tickers]
    if not usable:
        rows = _empty_session_rows()
    else:
        usable_tickers = {str(row["ticker"]) for row in usable}
        features = build_causal_intraday_bar_features(
            volume.bars.loc[
                volume.bars["ticker"].astype(str).isin(usable_tickers)
            ],
            five.loc[five["ticker"].astype(str).isin(usable_tickers)],
            stocks.loc[stocks["ticker"].astype(str).isin(usable_tickers)],
            benchmarks,
            memberships.loc[memberships["ticker"].astype(str).isin(usable_tickers)],
            pd.DataFrame(usable),
            contract=verified.contract,
        )
        labeled = build_exact_intraday_bar_labels(
            features,
            stocks.loc[stocks["ticker"].astype(str).isin(usable_tickers)],
            benchmarks,
            contract=verified.contract,
            strategy_contract_sha256=verified.contract_sha256,
        )
        rows = _finalize_rows(labeled)
    return rows, _session_audit(
        session_date,
        session_selection,
        pair_reasons,
        rows=rows,
        incomplete_pairs=verified.incomplete_pairs,
    )


def _finalize_rows(frame: pd.DataFrame) -> pd.DataFrame:
    rows = frame.copy()
    rows["dataset_eligible"] = (
        rows["feature_eligible"].astype(bool)
        & rows["label_eligible"].astype(bool)
        & rows["rank_label"].notna()
    )
    rows["dataset_ineligible_reason"] = pd.Series(
        pd.NA, index=rows.index, dtype="string"
    )
    feature_bad = ~rows["feature_eligible"].astype(bool)
    label_bad = ~feature_bad & ~rows["label_eligible"].astype(bool)
    rank_bad = ~feature_bad & ~label_bad & rows["rank_label"].isna()
    rows.loc[feature_bad, "dataset_ineligible_reason"] = (
        "feature:" + rows.loc[feature_bad, "feature_ineligible_reason"].astype(str)
    )
    rows.loc[label_bad, "dataset_ineligible_reason"] = (
        "label:" + rows.loc[label_bad, "label_ineligible_reason"].astype(str)
    )
    rows.loc[rank_bad, "dataset_ineligible_reason"] = (
        "rank:insufficient_exact_decision_cohort"
    )
    _validate_rows(rows)
    return rows.sort_values(["decision_time_utc", "ticker"], kind="stable").reset_index(
        drop=True
    )


def _validate_rows(rows: pd.DataFrame) -> None:
    if rows.empty:
        return
    if (
        bool(rows.duplicated(["ticker", "decision_time_utc"]).any())
        or not rows["feature_schema_version"].astype(str).eq(
            INTRADAY_BAR_FEATURE_SCHEMA_VERSION
        ).all()
        or not rows["label_schema_version"].astype(str).eq(
            INTRADAY_BAR_LABEL_SCHEMA_VERSION
        ).all()
        or not rows["ordered_feature_sha256"].astype(str).eq(
            INTRADAY_BAR_MODEL_FEATURES_SHA256
        ).all()
    ):
        raise DataReadinessError("intraday bar dataset row identity differs")
    eligible = rows.loc[rows["label_eligible"].astype(bool)]
    if eligible.empty:
        return
    decision = pd.to_datetime(eligible["decision_time_utc"], utc=True, errors="raise")
    feature = pd.to_datetime(
        eligible["feature_available_at_utc"], utc=True, errors="raise"
    )
    entry = pd.to_datetime(eligible["entry_time_utc"], utc=True, errors="raise")
    exit_end = pd.to_datetime(eligible["exit_bar_end_utc"], utc=True, errors="raise")
    label = pd.to_datetime(
        eligible["label_available_at_utc"], utc=True, errors="raise"
    )
    if (
        bool(feature.ne(decision).any())
        or bool(entry.ne(decision + pd.Timedelta(minutes=1)).any())
        or bool(exit_end.gt(label).any())
        or bool(pd.to_numeric(eligible["atr_14_5m"], errors="coerce").le(0).any())
    ):
        raise DataReadinessError("intraday bar dataset contains causal timing leakage")


def _session_audit(
    session_date: str,
    selection: pd.DataFrame,
    pair_reasons: Mapping[str, str],
    *,
    rows: pd.DataFrame,
    incomplete_pairs: frozenset[tuple[str, str]],
) -> dict[str, Any]:
    records = []
    for ticker in sorted(selection["ticker"].astype(str)):
        ticker_rows = (
            rows.loc[rows["ticker"].astype(str).eq(ticker)]
            if not rows.empty and "ticker" in rows.columns
            else rows
        )
        records.append(
            {
                "ticker": ticker,
                "session_date_et": session_date,
                "source_session_complete": (session_date, ticker)
                not in incomplete_pairs,
                "status": "abstained" if ticker in pair_reasons else "published",
                "reason": pair_reasons.get(ticker),
                "feature_rows": len(ticker_rows),
                "feature_eligible_rows": (
                    int(ticker_rows["feature_eligible"].sum())
                    if "feature_eligible" in ticker_rows
                    else 0
                ),
                "label_eligible_rows": (
                    int(ticker_rows["label_eligible"].sum())
                    if "label_eligible" in ticker_rows
                    else 0
                ),
                "dataset_eligible_rows": (
                    int(ticker_rows["dataset_eligible"].sum())
                    if "dataset_eligible" in ticker_rows
                    else 0
                ),
            }
        )
    return {"session_date_et": session_date, "stock_sessions": records}


def _empty_session_rows() -> pd.DataFrame:
    return pd.DataFrame()


def _publish_session_unit(
    unit: Path,
    *,
    session_date: str,
    request_sha256: str,
    transformation_sha256: str,
    session_request_sha256: str,
    rows: pd.DataFrame,
    audit: Mapping[str, Any],
) -> None:
    staging = unit.with_name(f".{unit.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        rows.to_parquet(staging / "rows.parquet", index=False, compression="zstd")
        _write_json(staging / "audit.json", audit)
        parquet_schema = _arrow_schema_record(
            pq.read_schema(staging / "rows.parquet")  # type: ignore[no-untyped-call]
        )
        record = {
            "schema": "edge_rebuild.intraday_bar_dataset_session_unit.v1",
            "state": "complete",
            "session_date_et": session_date,
            "request_sha256": request_sha256,
            "transformation_sha256": transformation_sha256,
            "session_request_sha256": session_request_sha256,
            "rows": len(rows),
            "dataset_eligible_rows": (
                int(rows["dataset_eligible"].sum())
                if "dataset_eligible" in rows
                else 0
            ),
            "ticker_count": (
                int(rows["ticker"].nunique()) if "ticker" in rows else 0
            ),
            "rows_sha256": file_sha256(staging / "rows.parquet"),
            "audit_sha256": file_sha256(staging / "audit.json"),
            "parquet_schema": parquet_schema,
            "parquet_schema_sha256": json_sha256(parquet_schema),
        }
        _write_json(staging / "_unit.json", record)
        unit.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(unit)
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _verify_session_unit(
    unit: Path,
    *,
    session_date: str,
    request_sha256: str,
    session_request_sha256: str,
    transformation_sha256: str,
) -> dict[str, Any]:
    record = _read_json(unit / "_unit.json")
    rows_path = unit / "rows.parquet"
    audit_path = unit / "audit.json"
    actual = {path.name for path in unit.iterdir() if path.is_file()}
    if (
        actual != _UNIT_FILES
        or record.get("schema")
        != "edge_rebuild.intraday_bar_dataset_session_unit.v1"
        or record.get("state") != "complete"
        or record.get("session_date_et") != session_date
        or record.get("request_sha256") != request_sha256
        or record.get("transformation_sha256") != transformation_sha256
        or record.get("session_request_sha256") != session_request_sha256
        or record.get("rows_sha256") != file_sha256(rows_path)
        or record.get("audit_sha256") != file_sha256(audit_path)
    ):
        raise DataReadinessError(
            f"intraday bar dataset session unit differs: {session_date}"
        )
    parquet = pq.ParquetFile(rows_path, memory_map=True)  # type: ignore[no-untyped-call]
    parquet_schema = _arrow_schema_record(parquet.schema_arrow)
    if (
        record.get("parquet_schema") != parquet_schema
        or record.get("parquet_schema_sha256") != json_sha256(parquet_schema)
    ):
        raise DataReadinessError(
            f"intraday bar dataset session schema differs: {session_date}"
        )
    physical_rows = 0 if parquet.metadata is None else parquet.metadata.num_rows
    if physical_rows != int(record.get("rows", -1)):
        raise DataReadinessError(
            f"intraday bar dataset session row count differs: {session_date}"
        )
    if physical_rows:
        rows = pd.read_parquet(
            rows_path,
            columns=[
                "session_date_et",
                "ticker",
                "decision_time_utc",
                "dataset_eligible",
            ],
        )
        if (
            not rows["session_date_et"].astype(str).eq(session_date).all()
            or bool(rows.duplicated(["ticker", "decision_time_utc"]).any())
            or int(rows["dataset_eligible"].sum())
            != int(record.get("dataset_eligible_rows", -1))
            or int(rows["ticker"].nunique()) != int(record.get("ticker_count", -1))
        ):
            raise DataReadinessError(
                f"intraday bar dataset session identities differ: {session_date}"
            )
    return dict(record)


def _prepare_work_directory(work: Path, request: Mapping[str, Any]) -> None:
    if work.exists():
        _remove_stale_json_temps(work)
        request_path = work / "_request.json"
        if not request_path.is_file():
            sessions = work / "sessions"
            if sessions.exists() and any(sessions.iterdir()):
                raise DataReadinessError(
                    "intraday bar dataset work directory lacks its request identity"
                )
            shutil.rmtree(work)
            work.mkdir(parents=True)
            _write_json(request_path, request)
            return
        existing = _read_json(request_path)
        if existing != request:
            raise DataReadinessError("intraday bar dataset work directory differs")
        _remove_stale_session_staging(work)
        manifest = work / "_manifest.json"
        authority = work / "_authority.json"
        if manifest.exists() != authority.exists():
            manifest.unlink(missing_ok=True)
            authority.unlink(missing_ok=True)
        return
    work.mkdir(parents=True)
    _write_json(work / "_request.json", request)


def _recover_complete_work_directory(
    work: Path,
    *,
    output_directory: Path,
    request: Mapping[str, Any],
) -> dict[str, Any] | None:
    if not work.exists():
        return None
    _remove_stale_json_temps(work)
    _remove_stale_session_staging(work)
    manifest = work / "_manifest.json"
    authority = work / "_authority.json"
    if not (manifest.is_file() and authority.is_file()):
        return None
    if _read_json(work / "_request.json") != request:
        raise DataReadinessError("intraday bar dataset work directory differs")
    load_complete_intraday_bar_dataset(work)
    output_directory.parent.mkdir(parents=True, exist_ok=True)
    work.replace(output_directory)
    return load_complete_intraday_bar_dataset(output_directory)


def _remove_stale_session_staging(work: Path) -> None:
    sessions = work / "sessions"
    if not sessions.is_dir():
        return
    for path in sessions.iterdir():
        if path.is_dir() and path.name.startswith(".session_date_et=") and path.name.endswith(".staging"):
            shutil.rmtree(path)


def _remove_stale_json_temps(work: Path) -> None:
    for path in work.rglob(".*.tmp"):
        if path.is_file():
            path.unlink()


def _projection_bar_files(
    root: Path,
    manifest: Mapping[str, Any],
) -> dict[str, _ProjectionArtifact]:
    output: dict[str, _ProjectionArtifact] = {}
    for raw in cast(list[Mapping[str, Any]], manifest["files"]):
        if raw.get("role") != "bars":
            continue
        month = str(raw.get("month", ""))
        path = _resolve_inside(root, str(raw.get("path", "")))
        expected_sha256 = str(raw.get("sha256", ""))
        if len(expected_sha256) != 64:
            raise DataReadinessError(
                "five-minute projection partition lacks a valid SHA-256"
            )
        if month in output:
            raise DataReadinessError("five-minute projection repeats a bar month")
        output[month] = _ProjectionArtifact(path=path, sha256=expected_sha256)
    return output


def _load_five_minute_session(
    root: Path,
    artifact: _ProjectionArtifact | None,
    *,
    session_date: str,
    tickers: set[str],
) -> pd.DataFrame:
    if artifact is None:
        return pd.DataFrame()
    path = artifact.path
    if root.resolve() not in path.resolve().parents:
        raise DataReadinessError("five-minute projection partition escapes authority")
    if path not in _SESSION_WORKER_VERIFIED_PROJECTIONS:
        if file_sha256(path) != artifact.sha256:
            raise DataReadinessError(
                f"five-minute projection partition hash differs: {path}"
            )
        _SESSION_WORKER_VERIFIED_PROJECTIONS.add(path)
    frame = pd.read_parquet(
        path,
        filters=[("session_date_et", "==", session_date)],
    )
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    return frame.loc[frame["ticker"].isin(tickers)].copy()


def _validate_projection_lineage(
    projection: Mapping[str, Any],
    *,
    verified_parent_lineage: Mapping[str, str],
    selection_directory: Path,
    strategy_contract_path: Path,
) -> None:
    if (
        Path(str(projection.get("selection_directory", ""))).resolve()
        != selection_directory.resolve()
        or projection.get("selection_authority_sha256")
        != verified_parent_lineage.get("selection_authority_sha256")
        or projection.get("selection_manifest_sha256")
        != verified_parent_lineage.get("selection_manifest_sha256")
        or projection.get("selection_table_sha256")
        != verified_parent_lineage.get("selection_table_sha256")
        or projection.get("five_minute_canonical_authority_sha256")
        != verified_parent_lineage.get("five_minute_canonical_authority_sha256")
        or projection.get("five_minute_canonical_manifest_sha256")
        != verified_parent_lineage.get("five_minute_canonical_manifest_sha256")
        or projection.get("five_minute_canonical_file_inventory_sha256")
        != verified_parent_lineage.get(
            "five_minute_canonical_file_inventory_sha256"
        )
        or projection.get("strategy_contract_file_sha256")
        != file_sha256(strategy_contract_path)
        or projection.get("strategy_contract_sha256")
        != verified_parent_lineage.get("strategy_contract_sha256")
        or projection.get("intraday_data_contract_sha256")
        != verified_parent_lineage.get("intraday_data_contract_sha256")
        or projection.get("intraday_parent_contract_sha256")
        != verified_parent_lineage.get("intraday_parent_contract_sha256")
        or str(projection.get("intraday_contract_lineage_file_sha256") or "")
        != str(
            verified_parent_lineage.get(
                "intraday_contract_lineage_file_sha256", ""
            )
        )
    ):
        raise DataReadinessError(
            "five-minute projection and verified dataset parents differ"
        )


def _session_request_sha256(
    request_sha256: str,
    session_date: str,
    selection: pd.DataFrame,
) -> str:
    identities = [
        {
            "ticker": str(row.ticker),
            "activation_time_utc": pd.Timestamp(row.activation_time_utc).isoformat(),
        }
        for row in selection.sort_values(
            ["activation_time_utc", "ticker"], kind="stable"
        ).itertuples(index=False)
    ]
    return json_sha256(
        {
            "request_sha256": request_sha256,
            "session_date_et": session_date,
            "selection": identities,
        }
    )


def _require_path_isolation(output: Path, inputs: Sequence[Path]) -> None:
    target = output.resolve()
    for source in inputs:
        resolved = source.resolve()
        if target == resolved or target in resolved.parents or resolved in target.parents:
            raise DataReadinessError("intraday bar dataset output overlaps an input")


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("intraday bar dataset artifact path is invalid")
    path = (root / relative).resolve()
    if root.resolve() not in path.parents:
        raise DataReadinessError("intraday bar dataset artifact path escapes authority")
    return path


def _transformation_identity() -> dict[str, Any]:
    root = Path(__file__).resolve().parent
    files = [
        {
            "path": name,
            "sha256": file_sha256(root / name),
        }
        for name in (
            "intraday_bar_dataset.py",
            "intraday_bar_features.py",
            "intraday_bar_labels.py",
            "volume_bars.py",
        )
    ]
    payload = {
        "schema": INTRADAY_BAR_TRANSFORMATION_SCHEMA,
        "files": files,
    }
    return {**payload, "sha256": json_sha256(payload)}


def _run_bounded_process_tasks(
    tasks: Sequence[tuple[str, pd.DataFrame, str]],
    *,
    context: _SessionWorkerContext,
    session_workers: int,
) -> list[dict[str, Any]]:
    iterator = iter(tasks)
    results: list[dict[str, Any]] = []
    with ProcessPoolExecutor(
        max_workers=session_workers,
        initializer=_initialize_session_worker,
        initargs=(context,),
    ) as executor:
        in_flight: dict[Future[dict[str, Any]], str] = {}
        for _ in range(session_workers):
            task = next(iterator, None)
            if task is None:
                break
            in_flight[executor.submit(_process_session_task, task)] = task[0]
        try:
            while in_flight:
                done, _ = wait(in_flight, return_when=FIRST_COMPLETED)
                for future in done:
                    in_flight.pop(future)
                    results.append(future.result())
                    _guard_memory("intraday bar dataset completed session unit")
                    task = next(iterator, None)
                    if task is not None:
                        in_flight[executor.submit(_process_session_task, task)] = task[0]
        except BaseException:
            for future in in_flight:
                future.cancel()
            raise
    return sorted(results, key=lambda item: str(item["session_date_et"]))


def _publication_memory_audit(
    worker_results: Sequence[Mapping[str, Any]],
    *,
    workers_are_children: bool,
) -> dict[str, Any]:
    parent = memory_audit(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
    ).to_record()
    worker_peaks: dict[int, float] = {}
    if workers_are_children:
        for result in worker_results:
            memory = result.get("memory")
            if not isinstance(memory, Mapping):
                raise DataReadinessError("intraday dataset worker omitted memory audit")
            peak = memory.get("peak_working_set_gib")
            if peak is None:
                raise DataReadinessError(
                    "intraday dataset worker memory accounting is unavailable"
                )
            pid = int(result["pid"])
            worker_peaks[pid] = max(worker_peaks.get(pid, 0.0), float(peak))
    parent_peak = parent.get("peak_working_set_gib")
    if parent_peak is None:
        raise DataReadinessError(
            "intraday dataset parent memory accounting is unavailable"
        )
    aggregate_upper_bound = float(parent_peak) + sum(worker_peaks.values())
    safety_threshold = MEMORY_HARD_BUDGET_GIB - MEMORY_HEADROOM_GIB
    if aggregate_upper_bound > safety_threshold:
        raise DataReadinessError(
            "intraday dataset aggregate peak memory upper bound exceeds "
            f"{safety_threshold:.2f} GiB"
        )
    return {
        **parent,
        "worker_peak_working_set_gib_by_pid": {
            str(pid): peak for pid, peak in sorted(worker_peaks.items())
        },
        "aggregate_peak_upper_bound_gib": aggregate_upper_bound,
    }


def _revalidate_parent_inputs(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    five_minute_projection_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    intraday_contract_lineage_path: Path,
    expected_parent_lineage: Mapping[str, str],
    expected_transformation: Mapping[str, Any],
) -> None:
    fresh = _verify_inputs(
        selection_directory=selection_directory,
        stock_collection_directory=stock_collection_directory,
        stock_coverage_directory=stock_coverage_directory,
        benchmark_collection_directory=benchmark_collection_directory,
        membership_authority_directory=membership_authority_directory,
        strategy_contract=strategy_contract,
        strategy_contract_path=strategy_contract_path,
        intraday_contract_lineage_path=intraday_contract_lineage_path,
    )
    projection = load_complete_selected_session_five_minute_projection(
        five_minute_projection_directory
    )
    _validate_projection_lineage(
        projection,
        verified_parent_lineage=fresh.parent_lineage,
        selection_directory=selection_directory,
        strategy_contract_path=strategy_contract_path,
    )
    actual_parent_lineage = {
        **fresh.parent_lineage,
        "five_minute_projection_authority_sha256": file_sha256(
            five_minute_projection_directory / "_authority.json"
        ),
        "five_minute_projection_manifest_sha256": file_sha256(
            five_minute_projection_directory / "_manifest.json"
        ),
        "five_minute_projection_inventory_sha256": str(
            projection["file_inventory_sha256"]
        ),
    }
    if (
        actual_parent_lineage != expected_parent_lineage
        or _transformation_identity() != expected_transformation
    ):
        raise DataReadinessError(
            "intraday bar dataset parents or transformation changed during publication"
        )


def _arrow_schema_record(schema: Any) -> list[dict[str, Any]]:
    return [
        {
            "name": str(field.name),
            "type": str(field.type),
            "nullable": bool(field.nullable),
        }
        for field in schema
    ]


def _guard_memory(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
        headroom_gib=MEMORY_HEADROOM_GIB,
        stage=stage,
    )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"intraday bar dataset JSON is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise DataReadinessError(f"intraday bar dataset JSON must be an object: {path}")
    return {str(key): value for key, value in raw.items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    staging = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        staging.write_text(
            json.dumps(payload, indent=2, sort_keys=True, default=str),
            encoding="utf-8",
        )
        staging.replace(path)
    finally:
        staging.unlink(missing_ok=True)
