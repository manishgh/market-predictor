"""Hash-bound one-minute benchmark planning for selected intraday sessions."""

from __future__ import annotations

import shutil
import uuid
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_contracts import (
    REGULAR_SEGMENT,
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
    SelectedSessionBenchmarkConfig,
)
from market_predictor.edge_rebuild.intraday_history import (
    SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA,
    chunk_request_symbols,
    file_record,
    json_sha256,
    request_unit_record,
    stable_identity_hash,
    write_plan_json,
)
from market_predictor.edge_rebuild.selected_session_history import (
    verify_selected_stock_sessions,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError


def build_selected_session_benchmark_plan(
    *,
    selection_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: SelectedSessionBenchmarkConfig,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
) -> dict[str, Any]:
    """Plan SPY, QQQ, and sector-ETF minute paths for every selected session."""

    if output_directory.exists():
        raise DataReadinessError(
            f"selected-session benchmark plan output must be new: {output_directory}"
        )
    _assert_memory(config, "selected-session benchmark planning start")
    selection, selection_identity = verify_selected_stock_sessions(
        selection_directory
    )
    if (
        selection_identity["strategy_id"] != strategy_contract.intraday.strategy_id
        or selection_identity["strategy_contract_sha256"]
        != strategy_contract.sha256()
    ):
        raise DataReadinessError(
            "benchmark plan requires selection under the active intraday contract"
        )
    request: dict[str, Any] = {
        "schema": SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "strategy_contract_path": str(strategy_contract_path),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "selection": selection_identity,
        "benchmark_tickers": list(config.normalized_benchmarks()),
        "session_segments": list(config.session_segments),
        "training_performed": False,
        "download_performed": False,
    }
    plan_fingerprint = json_sha256(request)
    request["plan_fingerprint"] = plan_fingerprint
    calendar = xcals.get_calendar(config.calendar)
    selected_dates = sorted(set(selection["session_date_et"].astype(str)))
    sessions = calendar.sessions_in_range(selected_dates[0], selected_dates[-1])
    by_date = {str(pd.Timestamp(value).date()): pd.Timestamp(value) for value in sessions}
    unknown = sorted(set(selected_dates).difference(by_date))
    if unknown:
        raise DataReadinessError(
            f"selected dates are not exchange sessions: {unknown[:5]}"
        )

    session_rows: dict[str, list[dict[str, object]]] = {}
    unit_rows: dict[str, list[dict[str, object]]] = {}
    expected_rows = 0
    units = 0
    early_closes = 0
    benchmarks = sorted(config.normalized_benchmarks())
    for session_date in selected_dates:
        session = by_date[session_date]
        open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
        close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
        minutes = int((close_at - open_at).total_seconds() // 60)
        if minutes != 390:
            early_closes += 1
        month = session.strftime("%Y-%m")
        session_rows.setdefault(month, []).append(
            {
                "session_date_et": session.date(),
                "session_open_utc": open_at,
                "session_close_utc": close_at,
                "session_segment": REGULAR_SEGMENT,
            }
        )
        for chunk, mapping in chunk_request_symbols(
            benchmarks,
            expected_bars_per_symbol=minutes,
            maximum_symbols_per_unit=config.maximum_symbols_per_unit,
            maximum_expected_rows_per_unit=config.maximum_expected_rows_per_unit,
            label=f"{session_date} benchmarks",
        ):
            unit_id = stable_identity_hash(
                plan_fingerprint,
                session_date,
                REGULAR_SEGMENT,
                open_at.isoformat(),
                close_at.isoformat(),
                *sorted(mapping),
                config.history_timeframe,
                "sip",
                "all",
            )
            unit_rows.setdefault(month, []).append(
                {
                    **request_unit_record(
                        unit_id=unit_id,
                        session_date=session.date(),
                        start=open_at,
                        end=close_at,
                        chunk=chunk,
                        mapping=mapping,
                        expected_bars_per_symbol=minutes,
                        plan_fingerprint=plan_fingerprint,
                        timeframe=config.history_timeframe,
                    ),
                    "session_segment": REGULAR_SEGMENT,
                    "session_open_utc": open_at,
                    "session_close_utc": close_at,
                }
            )
            units += 1
            expected_rows += minutes * len(chunk)
    _assert_memory(config, "selected-session benchmark planning frames")

    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, object]] = []
        for month, rows in sorted(session_rows.items()):
            path = temporary / "sessions" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows).sort_values("session_open_utc")
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        for month, rows in sorted(unit_rows.items()):
            path = temporary / "units" / "1Min" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame = pd.DataFrame(rows).sort_values(
                ["requested_start_utc", "unit_id"], kind="stable"
            )
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        write_plan_json(temporary / "_request.json", request)
        files.append(file_record(temporary / "_request.json", temporary, 1))
        manifest: dict[str, Any] = {
            "schema": SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_fingerprint": plan_fingerprint,
            "policy_sha256": config.sha256(),
            "selection": selection_identity,
            "strategy_contract_sha256": strategy_contract.sha256(),
            "research_only": True,
            "promotion_eligible": False,
            "acquisition": {
                "provider": "alpaca",
                "calendar": config.calendar,
                "calendar_version": version("exchange-calendars"),
                "price_feed": "sip",
                "adjustment": "all",
                "timeframe": config.history_timeframe,
                "layer": "selected_session_benchmarks",
                "symbols": benchmarks,
                "scope": "SPY, QQQ, and all sector ETFs on every selected session",
            },
            "summary": {
                "first_history_session": selected_dates[0],
                "last_history_session": selected_dates[-1],
                "planned_history_sessions": len(selected_dates),
                "benchmark_tickers": len(benchmarks),
                "acquisition_units": units,
                "maximum_expected_rows": expected_rows,
                "early_close_sessions": early_closes,
                "memory": memory_audit(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                ).to_record(),
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        write_plan_json(temporary / "_manifest.json", manifest)
        write_plan_json(
            temporary / "_authority.json",
            {
                "schema": SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(temporary / "_manifest.json"),
                "plan_fingerprint": plan_fingerprint,
            },
        )
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def _assert_memory(config: SelectedSessionBenchmarkConfig, stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
    assert_peak_memory_budget(
        hard_budget_gib=config.maximum_process_memory_gib,
        headroom_gib=config.memory_guard_headroom_gib,
        stage=stage,
    )
