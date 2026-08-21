from __future__ import annotations
from market_predictor.edge_rebuild.intraday_features import FEATURE_SCHEMA_VERSION
from market_predictor.edge_rebuild.intraday_labels import LABEL_SCHEMA_VERSION
import pyarrow.parquet as pq
from market_predictor.intraday.datasets.audits import _pair_abstention, _pair_audit
from market_predictor.intraday.datasets.io import _file_record
from market_predictor.edge_rebuild.intraday_selection import load_complete_intraday_selection
from market_predictor.intraday.datasets.validation import _membership_sector_exclusions, _validate_monthly_partition_records, _validate_no_leakage, _verify_inputs, _verify_monthly_partition_files
"""Atomic, lineage-bound publisher for the causal intraday training dataset."""


import shutil
import uuid
from collections.abc import Mapping
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from itertools import islice
from pathlib import Path
from typing import Any, cast

import pandas as pd
import pyarrow.parquet as pq

from market_predictor.canonical.store import (
    file_sha256,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.intraday_features import (
    FEATURE_SCHEMA_VERSION,
    build_causal_intraday_features,
)
from market_predictor.edge_rebuild.intraday_history import (
    json_sha256,
)
from market_predictor.edge_rebuild.intraday_labels import (
    LABEL_SCHEMA_VERSION,
    _add_contemporaneous_rank,
    _empty_label_columns,
    build_exact_causal_intraday_labels,
)
from market_predictor.edge_rebuild.strategy_contract import (
    StrategyContract,
)
from market_predictor.edge_rebuild.volume_bars import build_causal_volume_bars
from market_predictor.intraday.contracts.dataset_schemas import (
    _LABEL_COLUMNS,
    INTRADAY_DATASET_AUTHORITY_SCHEMA,
    INTRADAY_DATASET_SCHEMA,
    MAX_SESSION_WORKERS,
    MEMORY_HARD_BUDGET_GIB,
    MEMORY_HEADROOM_GIB,
    WORKING_SET_RELEASE_INTERVAL_SESSIONS,
    _Artifact,
    _SessionResult,
    _VerifiedInputs,
)
from market_predictor.intraday.datasets.audits import (
    _activation_abstention_reason,
    _expected_monthly_counts,
    _monthly_stock_session_counts,
    _pair_abstention,
    _pair_audit,
    _record_excluded_pairs,
    _row_abstentions,
)
from market_predictor.intraday.datasets.io import (
    _file_record,
    _guard_memory,
    _load_json,
    _MonthlyPartitionWriter,
    _StreamingAuditWriter,
    _write_json,
)
from market_predictor.intraday.datasets.transformations import (
    _benchmark_artifact_index,
    _finalize_dataset_rows,
    _load_benchmark_session,
    _load_stock_session_batch,
    _membership_for_pair,
    _resolve_inside,
    _split_decision_features,
    _stock_artifact_index,
)
from market_predictor.intraday.datasets.validation import (
    _validate_monthly_partition_records,
    _validate_no_leakage,
    _verify_inputs,
    _verify_monthly_partition_files,
)
from market_predictor.resources import (
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)


def publish_intraday_dataset(
    *,
    selection_directory: Path,
    stock_collection_directory: Path,
    stock_coverage_directory: Path,
    benchmark_collection_directory: Path,
    membership_authority_directory: Path,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
    output_directory: Path,
    session_workers: int = MAX_SESSION_WORKERS,
) -> dict[str, Any]:
    """Publish verified feature/label partitions without loading the corpus.

    Stock data is read one selected stock-session at a time. Benchmark data and
    reduced labeled rows are retained only for the current exchange session so
    cross-sectional ranks remain exact while memory stays bounded.
    """

    if session_workers < 1 or session_workers > MAX_SESSION_WORKERS:
        raise ValueError(
            f"session_workers must be between 1 and {MAX_SESSION_WORKERS}"
        )
    verified = _verify_inputs(
        selection_directory=selection_directory,
        stock_collection_directory=stock_collection_directory,
        stock_coverage_directory=stock_coverage_directory,
        benchmark_collection_directory=benchmark_collection_directory,
        membership_authority_directory=membership_authority_directory,
        strategy_contract=strategy_contract,
        strategy_contract_path=strategy_contract_path,
    )
    usable_selection = verified.selection[
        ~verified.selection["ticker"].isin(verified.excluded_tickers)
    ].copy()
    expected_selected_by_month = _monthly_stock_session_counts(verified.selection)
    expected_usable_by_month = _monthly_stock_session_counts(usable_selection)
    request = {
        "schema": INTRADAY_DATASET_SCHEMA,
        "selection_directory": str(selection_directory.resolve()),
        "stock_collection_directory": str(stock_collection_directory.resolve()),
        "stock_coverage_directory": str(stock_coverage_directory.resolve()),
        "benchmark_collection_directory": str(benchmark_collection_directory.resolve()),
        "membership_authority_directory": str(membership_authority_directory.resolve()),
        "strategy_contract_path": str(strategy_contract_path.resolve()),
        "strategy_contract_sha256": verified.contract_sha256,
        "parent_lineage": verified.parent_lineage,
        "parent_lineage_sha256": json_sha256(verified.parent_lineage),
        "membership_sector_excluded_tickers": sorted(
            verified.membership_sector_excluded_tickers
        ),
        "all_excluded_tickers": sorted(verified.excluded_tickers),
        "security_exclusion_fraction": (
            len(verified.excluded_tickers)
            / int(verified.selection["ticker"].nunique())
        ),
        "expected_selected_stock_sessions_by_month": expected_selected_by_month,
        "expected_usable_stock_sessions_by_month": expected_usable_by_month,
        "partitioning": ["session_month_et"],
        "partition_layout": "one_parquet_file_per_calendar_month",
        "partition_row_group": "one_completed_exchange_session",
        "processing_unit": "one_exchange_session",
        "ranking_unit": "one_exchange_session",
        "session_workers": session_workers,
        "working_set_release_interval_sessions": WORKING_SET_RELEASE_INTERVAL_SESSIONS,
        "memory_hard_budget_gib": MEMORY_HARD_BUDGET_GIB,
    }
    request_sha256 = json_sha256(request)
    if output_directory.exists():
        existing_manifest = load_complete_intraday_dataset(output_directory)
        if existing_manifest.get("request_sha256") != request_sha256:
            raise DataReadinessError(f"published intraday dataset is immutable: {output_directory}")
        return existing_manifest

    _guard_memory("intraday dataset publication start")
    staging = output_directory.with_name(f".{output_directory.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    partition_writer = _MonthlyPartitionWriter(staging)
    audit_writer = _StreamingAuditWriter(staging)
    try:
        _write_json(staging / "_request.json", {**request, "request_sha256": request_sha256})
        partition_records: list[dict[str, Any]] = []
        stock_index = _stock_artifact_index(verified.stock_artifacts)
        benchmark_index = _benchmark_artifact_index(verified.benchmark_artifacts)
        usable = usable_selection
        initial_pair_audits: list[dict[str, Any]] = []
        initial_abstentions: list[dict[str, Any]] = []
        _record_excluded_pairs(
            verified.selection,
            verified.excluded_tickers.difference(
                verified.membership_sector_excluded_tickers
            ),
            pair_audits=initial_pair_audits,
            abstentions=initial_abstentions,
            stage="coverage",
            reason="whole_security_coverage_exclusion",
        )
        _record_excluded_pairs(
            verified.selection,
            verified.membership_sector_excluded_tickers,
            pair_audits=initial_pair_audits,
            abstentions=initial_abstentions,
            stage="membership",
            reason="whole_security_invalid_sector_benchmark_exclusion",
        )
        if usable.empty:
            raise DataReadinessError("coverage excluded every selected security")
        audit_writer.write(initial_pair_audits, initial_abstentions)
        del initial_pair_audits, initial_abstentions

        with ThreadPoolExecutor(
            max_workers=session_workers,
            thread_name_prefix="intraday-session",
        ) as executor:
            session_groups = iter(
                usable.groupby("session_date_et", sort=True, observed=True)
            )
            completed_sessions = 0
            while True:
                batch = list(islice(session_groups, session_workers))
                if not batch:
                    break
                futures = [
                    executor.submit(
                        _publish_session,
                        session_date=str(session_date),
                        session_selection=session_selection,
                        verified=verified,
                        stock_index=stock_index,
                        benchmark_index=benchmark_index,
                        request_sha256=request_sha256,
                        parent_lineage_sha256=str(
                            request["parent_lineage_sha256"]
                        ),
                    )
                    for session_date, session_selection in batch
                ]
                for (session_date, _), future in zip(batch, futures, strict=True):
                    session_result = future.result()
                    if session_result.rows is not None:
                        completed_partition = partition_writer.write(
                            session_result.rows
                        )
                        if completed_partition is not None:
                            partition_records.append(completed_partition)
                    audit_writer.write(
                        session_result.pair_audits,
                        session_result.abstentions,
                    )
                    completed_sessions += 1
                    if (
                        completed_sessions
                        % WORKING_SET_RELEASE_INTERVAL_SESSIONS
                        == 0
                    ):
                        release_process_memory()
                    _guard_memory(
                        f"intraday dataset session {str(session_date)} complete"
                    )
        final_partition = partition_writer.close()
        if final_partition is not None:
            partition_records.append(final_partition)
        release_process_memory()

        if not partition_records:
            raise DataReadinessError("intraday dataset produced no feature-label partitions")
        _validate_monthly_partition_records(
            partition_records,
            expected_stock_sessions_by_month=expected_usable_by_month,
        )
        if audit_writer.pair_rows != len(verified.selection):
            raise DataReadinessError(
                "stock-session audit does not reconcile to the causal selection"
            )
        audit_files = audit_writer.close()
        request_record = _file_record(staging / "_request.json", staging, rows=1)
        files = sorted(
            [*partition_records, *audit_files, request_record],
            key=lambda item: str(item["path"]),
        )
        total_rows = sum(int(record["rows"]) for record in partition_records)
        eligible_rows = sum(int(record["eligible_rows"]) for record in partition_records)
        manifest: dict[str, Any] = {
            "schema": INTRADAY_DATASET_SCHEMA,
            "status": "complete",
            "created_at_utc": datetime.now(UTC).isoformat(),
            "request_sha256": request_sha256,
            "strategy_contract_sha256": verified.contract_sha256,
            "feature_schema_version": FEATURE_SCHEMA_VERSION,
            "label_schema_version": LABEL_SCHEMA_VERSION,
            "parent_lineage": verified.parent_lineage,
            "parent_lineage_sha256": request["parent_lineage_sha256"],
            "partitioning": request["partitioning"],
            "partition_layout": request["partition_layout"],
            "partition_row_group": request["partition_row_group"],
            "partitions": partition_records,
            "files": files,
            "summary": {
                "selected_stock_sessions": int(len(verified.selection)),
                "excluded_stock_sessions": int(verified.selection["ticker"].isin(verified.excluded_tickers).sum()),
                "membership_sector_excluded_securities": len(
                    verified.membership_sector_excluded_tickers
                ),
                "incomplete_stock_sessions": len(verified.incomplete_pairs),
                "published_stock_sessions": sum(
                    int(record["stock_sessions"])
                    for record in partition_records
                ),
                "partition_files": len(partition_records),
                "rows": total_rows,
                "dataset_eligible_rows": eligible_rows,
                "abstention_rows": audit_writer.abstention_rows,
                "memory": memory_audit(
                    hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
                    headroom_gib=MEMORY_HEADROOM_GIB,
                ).to_record(),
            },
            "training_contract": {
                "eligibility_column": "dataset_eligible",
                "feature_columns_exclude": sorted(
                    _LABEL_COLUMNS
                    | {
                        "dataset_eligible",
                        "dataset_ineligible_reason",
                        "dataset_row_id",
                        "dataset_request_sha256",
                        "parent_lineage_sha256",
                    }
                ),
                "label_columns": sorted(_LABEL_COLUMNS),
            },
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": INTRADAY_DATASET_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request_sha256,
                "parent_lineage_sha256": request["parent_lineage_sha256"],
                "partitions": len(partition_records),
                "rows": total_rows,
            },
        )
        assert_peak_memory_budget(
            hard_budget_gib=MEMORY_HARD_BUDGET_GIB,
            headroom_gib=MEMORY_HEADROOM_GIB,
            stage="intraday dataset publication",
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return load_complete_intraday_dataset(output_directory)
    except Exception:
        try:
            partition_writer.abort()
        except Exception:
            pass
        try:
            audit_writer.abort()
        except Exception:
            pass
        shutil.rmtree(staging, ignore_errors=True)
        raise

def _publish_session(
    *,
    session_date: str,
    session_selection: pd.DataFrame,
    verified: _VerifiedInputs,
    stock_index: Mapping[tuple[str, str], _Artifact],
    benchmark_index: Mapping[str, tuple[_Artifact, ...]],
    request_sha256: str,
    parent_lineage_sha256: str,
) -> _SessionResult:
    _guard_memory(f"intraday dataset session {session_date} start")
    benchmarks = _load_benchmark_session(
        session_date,
        artifacts=benchmark_index,
        required_tickers=verified.benchmark_tickers,
    )
    pair_audits: list[dict[str, Any]] = []
    abstentions: list[dict[str, Any]] = []
    activations = session_selection.sort_values(
        ["activation_time_utc", "ticker"], kind="stable"
    ).to_dict(orient="records")
    valid_activations: list[dict[str, Any]] = []
    membership_parts: list[pd.DataFrame] = []
    for activation in activations:
        ticker = str(activation["ticker"])
        if (session_date, ticker) in verified.incomplete_pairs:
            pair_audits.append(
                _pair_audit(
                    ticker,
                    session_date,
                    status="abstained",
                    reason="incomplete_five_minute_continuity",
                )
            )
            abstentions.append(
                _pair_abstention(
                    ticker,
                    session_date,
                    "coverage",
                    "incomplete_five_minute_continuity",
                )
            )
            continue
        membership = _membership_for_pair(
            verified.memberships,
            ticker=ticker,
            session_date=session_date,
        )
        activation_reason = _activation_abstention_reason(
            activation,
            membership,
            maximum_delay_seconds=(
                verified.contract.intraday.decision_finalization_seconds
            ),
        )
        if activation_reason is not None:
            pair_audits.append(
                _pair_audit(
                    ticker,
                    session_date,
                    status="abstained",
                    reason=activation_reason,
                )
            )
            abstentions.append(
                _pair_abstention(
                    ticker, session_date, "activation", activation_reason
                )
            )
            continue
        valid_activations.append(dict(activation))
        membership_parts.append(membership)

    if not valid_activations:
        return _SessionResult(None, tuple(pair_audits), tuple(abstentions))
    valid_tickers = [str(row["ticker"]) for row in valid_activations]
    stocks = _load_stock_session_batch(
        session_date,
        valid_tickers,
        artifacts=stock_index,
        coverage=verified.coverage,
    )
    memberships = pd.concat(membership_parts, ignore_index=True)
    volume_result = build_causal_volume_bars(
        stocks,
        pd.DataFrame(valid_activations),
        contract=verified.contract,
        strategy_contract_sha256=verified.contract_sha256,
    )
    volume_audits = {
        str(row["ticker"]): cast(Mapping[str, Any], row)
        for row in volume_result.audit.to_dict(orient="records")
    }
    for ticker, audit in volume_audits.items():
        if int(audit["completed_volume_bars"]) > 0:
            continue
        pair_audits.append(
            _pair_audit(
                ticker,
                session_date,
                status="abstained",
                reason="no_completed_volume_bars",
                source_rows=int(audit["source_rows"]),
            )
        )
        abstentions.append(
            _pair_abstention(
                ticker, session_date, "volume_bars", "no_completed_volume_bars"
            )
        )
    if volume_result.bars.empty:
        return _SessionResult(None, tuple(pair_audits), tuple(abstentions))
    features = build_causal_intraday_features(
        volume_result.bars,
        stocks,
        benchmarks,
        memberships,
        contract=verified.contract,
        strategy_contract_sha256=verified.contract_sha256,
    )
    decision_features, closed_features = _split_decision_features(features)
    if decision_features.empty:
        session_rows = _empty_label_columns(closed_features)
    else:
        session_rows = build_exact_causal_intraday_labels(
            decision_features,
            stocks,
            benchmarks,
            contract=verified.contract,
            strategy_contract_sha256=verified.contract_sha256,
        )
        if not closed_features.empty:
            session_rows = pd.concat(
                [session_rows, _empty_label_columns(closed_features)],
                ignore_index=True,
            )
    session_rows = _add_contemporaneous_rank(session_rows, verified.contract)
    session_rows = _finalize_dataset_rows(
        session_rows,
        request_sha256=request_sha256,
        parent_lineage_sha256=parent_lineage_sha256,
    )
    _validate_no_leakage(session_rows)
    for ticker, pair in session_rows.groupby("ticker", sort=True, observed=True):
        normalized = str(ticker)
        pair = pair.sort_values("volume_bar_number", kind="stable").reset_index(
            drop=True
        )
        pair_abstentions = _row_abstentions(pair)
        abstentions.extend(pair_abstentions)
        audit = volume_audits[normalized]
        pair_audits.append(
            _pair_audit(
                normalized,
                session_date,
                status="published",
                reason=None,
                source_rows=int(audit["source_rows"]),
                completed_volume_bars=len(pair),
                feature_rows=len(pair),
                feature_eligible_rows=int(pair["feature_eligible"].sum()),
                label_eligible_rows=int(pair["label_eligible"].sum()),
                dataset_eligible_rows=int(pair["dataset_eligible"].sum()),
                abstention_rows=len(pair_abstentions),
            )
        )
    return _SessionResult(
        session_rows.sort_values(
            ["session_date_et", "ticker", "volume_bar_number"],
            kind="stable",
        ).reset_index(drop=True),
        tuple(pair_audits),
        tuple(abstentions),
    )

def load_complete_intraday_dataset(directory: Path) -> dict[str, Any]:
    """Verify authority, immutable inventory, partition hashes, and row counts."""

    if not directory.is_dir():
        raise DataReadinessError(f"intraday dataset directory is missing: {directory}")
    request = _load_json(directory / "_request.json")
    manifest = _load_json(directory / "_manifest.json")
    authority = _load_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    if (
        json_sha256(request_payload) != request_sha256
        or request.get("schema") != INTRADAY_DATASET_SCHEMA
        or manifest.get("schema") != INTRADAY_DATASET_SCHEMA
        or manifest.get("status") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != INTRADAY_DATASET_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or authority.get("parent_lineage_sha256") != manifest.get("parent_lineage_sha256")
        or request.get("partitioning") != ["session_month_et"]
        or request.get("partition_layout")
        != "one_parquet_file_per_calendar_month"
        or request.get("partition_row_group")
        != "one_completed_exchange_session"
        or manifest.get("partitioning") != request.get("partitioning")
        or manifest.get("partition_layout") != request.get("partition_layout")
        or manifest.get("partition_row_group")
        != request.get("partition_row_group")
    ):
        raise DataReadinessError(f"intraday dataset lacks matching complete authority: {directory}")
    files = manifest.get("files")
    partitions = manifest.get("partitions")
    if not isinstance(files, list) or not files or not isinstance(partitions, list) or not partitions:
        raise DataReadinessError("intraday dataset manifest inventory is empty")
    partition_records = [
        cast(Mapping[str, Any], item)
        for item in partitions
        if isinstance(item, Mapping)
    ]
    if len(partition_records) != len(partitions):
        raise DataReadinessError("intraday dataset partition inventory is malformed")
    expected_selected_by_month = _expected_monthly_counts(
        request.get("expected_selected_stock_sessions_by_month"),
        label="selected",
    )
    expected_usable_by_month = _expected_monthly_counts(
        request.get("expected_usable_stock_sessions_by_month"),
        label="usable",
    )
    if (
        not set(expected_usable_by_month).issubset(expected_selected_by_month)
        or any(
            count > expected_selected_by_month[month]
            for month, count in expected_usable_by_month.items()
        )
    ):
        raise DataReadinessError(
            "intraday dataset usable coverage exceeds its causal selection"
        )
    _validate_monthly_partition_records(
        partition_records,
        expected_stock_sessions_by_month=expected_usable_by_month,
    )
    expected = {"_manifest.json", "_authority.json"}
    seen: set[str] = set()
    for raw in files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("intraday dataset file inventory is malformed")
        relative = str(raw.get("path", ""))
        if relative in seen:
            raise DataReadinessError("intraday dataset file inventory repeats a path")
        seen.add(relative)
        path = _resolve_inside(directory, relative)
        expected.add(relative)
        if not path.is_file() or path.stat().st_size != int(raw.get("bytes", -1)) or file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError(f"intraday dataset file failed integrity: {path}")
        if (
            path.suffix == ".parquet"
            and pq.ParquetFile(path).metadata.num_rows  # type: ignore[no-untyped-call]
            != int(raw.get("rows", -1))
        ):
            raise DataReadinessError(f"intraday dataset row count moved: {path}")
    if {str(item.get("path", "")) for item in partitions if isinstance(item, Mapping)} - seen:
        raise DataReadinessError("intraday dataset partition is absent from file inventory")
    _verify_monthly_partition_files(directory, partition_records)
    partition_rows = sum(int(item.get("rows", -1)) for item in partitions if isinstance(item, Mapping))
    partition_eligible = sum(int(item.get("eligible_rows", -1)) for item in partitions if isinstance(item, Mapping))
    published_stock_sessions = sum(
        int(item.get("stock_sessions", -1))
        for item in partitions
        if isinstance(item, Mapping)
    )
    inventory = {
        str(item.get("path", "")): item
        for item in files
        if isinstance(item, Mapping)
    }
    pair_audit_record = inventory.get("audit/stock_session_audit.parquet")
    abstention_record = inventory.get("audit/abstentions.parquet")
    parent_lineage = manifest.get("parent_lineage")
    summary = manifest.get("summary")
    if (
        not isinstance(parent_lineage, Mapping)
        or json_sha256(dict(parent_lineage)) != manifest.get("parent_lineage_sha256")
        or request.get("parent_lineage") != parent_lineage
        or not isinstance(summary, Mapping)
        or partition_rows != int(summary.get("rows", -1))
        or partition_eligible != int(summary.get("dataset_eligible_rows", -1))
        or published_stock_sessions
        != int(summary.get("published_stock_sessions", -1))
        or int(summary.get("selected_stock_sessions", -1))
        != sum(expected_selected_by_month.values())
        or int(summary.get("excluded_stock_sessions", -1))
        != sum(expected_selected_by_month.values())
        - sum(expected_usable_by_month.values())
        or not isinstance(pair_audit_record, Mapping)
        or int(pair_audit_record.get("rows", -1))
        != sum(expected_selected_by_month.values())
        or not isinstance(abstention_record, Mapping)
        or int(abstention_record.get("rows", -1))
        != int(summary.get("abstention_rows", -1))
        or partition_rows != int(authority.get("rows", -1))
        or len(partitions) != int(authority.get("partitions", -1))
    ):
        raise DataReadinessError("intraday dataset lineage or aggregate counts differ")
    actual = {path.relative_to(directory).as_posix() for path in directory.rglob("*") if path.is_file()}
    if actual != expected:
        raise DataReadinessError("intraday dataset immutable file set differs")
    return manifest