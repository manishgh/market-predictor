"""Hash-bound ER1B extended-session context acquisition planning.

The ER1A corpus holds regular-session bars only. This module plans the
separate pre-market and post-market five-minute layer for exactly the ER1A
session set, binding to the frozen ER1A plan and completed collection so the
two layers can never describe different sessions or a different universe.

Extended-hours bars stay in their own corpus. Regular-session VWAP, EMA, ATR,
and relative-volume denominators never see them, because thin extended-hours
volume varies by two orders of magnitude across the cross-section and would
silently reweight it.
"""

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
from market_predictor.edge_rebuild.history_collection import (
    load_complete_intraday_history_collection,
)
from market_predictor.edge_rebuild.history_contracts import (
    EXTENDED_CONTEXT_PLAN_SCHEMA,
    INTRADAY_HISTORY_PLAN_SCHEMA,
    POSTMARKET_SEGMENT,
    PREMARKET_SEGMENT,
    ExtendedSessionContextConfig,
)
from market_predictor.edge_rebuild.intraday_history import (
    EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA,
    chunk_request_symbols,
    expected_five_minute_bars,
    file_record,
    iter_point_in_time_sessions,
    json_sha256,
    load_complete_intraday_history_plan,
    load_plan_json,
    request_unit_record,
    stable_identity_hash,
    verify_point_in_time_memberships,
    write_plan_json,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError

EXCHANGE_TIMEZONE = "America/New_York"
SESSION_WINDOW_COLUMNS = (
    "session_date_et",
    "session_open_utc",
    "session_close_utc",
    "premarket_start_utc",
    "premarket_end_utc",
    "postmarket_start_utc",
    "postmarket_end_utc",
    "premarket_minutes",
    "postmarket_minutes",
)


def build_extended_session_context_plan(
    *,
    intraday_plan_directory: Path,
    intraday_collection_directory: Path,
    memberships_path: Path,
    membership_audit_path: Path,
    policy_path: Path,
    output_directory: Path,
    config: ExtendedSessionContextConfig,
    first_session: str | None = None,
) -> dict[str, Any]:
    """Plan pre/post-market context over the frozen ER1A session set.

    `first_session` narrows the plan to a suffix of that set. Extended context is
    only worth collecting for sessions that will reach the final dataset, and the
    regular-session corpus deliberately extends further back than the research
    window so the window can move without re-acquiring bars.
    """

    if output_directory.exists():
        raise DataReadinessError(
            f"ER1B context plan output must be new: {output_directory}"
        )
    _assert_memory(config, "ER1B context planning start")
    regular = _verify_regular_session_layer(
        intraday_plan_directory,
        intraday_collection_directory,
        memberships_path=memberships_path,
        membership_audit_path=membership_audit_path,
    )
    memberships, membership_identity = verify_point_in_time_memberships(
        memberships_path,
        membership_audit_path,
        minimum_cross_section=config.minimum_session_cross_section,
    )
    if membership_identity["sha256"] != regular["membership_sha256"]:
        raise DataReadinessError(
            "ER1B universe differs from the frozen ER1A universe"
        )
    regular_first = str(regular["first_history_session"])
    regular_last = str(regular["last_history_session"])
    window_first = first_session or regular_first
    if not regular_first <= window_first <= regular_last:
        raise DataReadinessError(
            f"ER1B first session {window_first} is outside the frozen ER1A range "
            f"{regular_first}..{regular_last}"
        )
    request = {
        "schema": EXTENDED_CONTEXT_PLAN_SCHEMA,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "regular_session_layer": regular,
        "membership": membership_identity,
        "premarket_start_et": config.premarket_start_et,
        "postmarket_end_et": config.postmarket_end_et,
        "window_first_session": window_first,
        "training_performed": False,
        "download_performed": False,
    }
    plan_fingerprint = json_sha256(request)
    calendar = xcals.get_calendar(config.calendar)
    sessions = calendar.sessions_in_range(window_first, regular_last)
    full_range = calendar.sessions_in_range(regular_first, regular_last)
    if len(full_range) != int(str(regular["planned_history_sessions"])):
        raise DataReadinessError(
            "ER1B session window does not match the frozen ER1A session count"
        )
    if not len(sessions):
        raise DataReadinessError("ER1B session window is empty")
    window_frames, unit_frames, totals = _build_context_frames(
        memberships=memberships,
        sessions=sessions,
        calendar=calendar,
        config=config,
        plan_fingerprint=plan_fingerprint,
    )
    _assert_memory(config, "ER1B context planning frames")
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        for month, frame in sorted(window_frames.items()):
            path = temporary / "session_windows" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        for month, frame in sorted(unit_frames.items()):
            path = temporary / "units" / "5Min" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        request["plan_fingerprint"] = plan_fingerprint
        write_plan_json(temporary / "_request.json", request)
        files.append(file_record(temporary / "_request.json", temporary, 1))
        manifest: dict[str, Any] = {
            "schema": EXTENDED_CONTEXT_PLAN_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_fingerprint": plan_fingerprint,
            "policy_sha256": config.sha256(),
            "research_only": True,
            "promotion_eligible": False,
            "membership_availability_policy": (
                "provider_publication_proxy_research_only"
            ),
            "acquisition": {
                "provider": "alpaca",
                "calendar": config.calendar,
                "calendar_version": version("exchange-calendars"),
                "price_feed": "sip",
                "adjustment": "all",
                "timeframe": "5Min",
                "layer": "extended_session_context",
                "segments": list(config.session_segments),
                "premarket_window_et": (
                    f"{config.premarket_start_et}-session_open"
                ),
                "postmarket_window_et": (
                    f"session_close-{config.postmarket_end_et}"
                ),
                "regular_session_rows_refetched": 0,
                "isolation": (
                    "extended bars are a separate layer and are never merged "
                    "into regular-session indicator inputs"
                ),
                "exact_path_labels": {
                    "timeframe": "1Min",
                    "scope": "selective_after_causal_setup_extraction",
                    "planned_in_this_artifact": False,
                    "missing_trade_policy": "no_trade_no_imputation",
                },
            },
            "summary": {
                "regular_layer_first_session": regular_first,
                "first_history_session": window_first,
                "last_history_session": regular_last,
                "planned_history_sessions": len(sessions),
                "regular_layer_sessions": len(full_range),
                "memory": memory_audit(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                ).to_record(),
                **totals,
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        write_plan_json(temporary / "_manifest.json", manifest)
        write_plan_json(
            temporary / "_authority.json",
            {
                "schema": EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA,
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


def _verify_regular_session_layer(
    plan_directory: Path,
    collection_directory: Path,
    *,
    memberships_path: Path,
    membership_audit_path: Path,
) -> dict[str, object]:
    """Bind ER1B to a frozen ER1A plan and a completed ER1A transport."""

    plan = load_complete_intraday_history_plan(
        plan_directory,
        accepted_schemas={
            INTRADAY_HISTORY_PLAN_SCHEMA: (
                "edge_rebuild.intraday_history_plan_authority.v1"
            )
        },
    )
    collection = load_complete_intraday_history_collection(collection_directory)
    plan_request = load_plan_json(plan_directory / "_request.json")
    fingerprint = str(plan["plan_fingerprint"])
    if collection.get("plan_fingerprint") != fingerprint:
        raise DataReadinessError(
            "ER1A collection does not belong to the supplied ER1A plan"
        )
    summary = plan.get("summary")
    membership = plan_request.get("membership")
    if not isinstance(summary, dict) or not isinstance(membership, dict):
        raise DataReadinessError("ER1A plan is missing summary or membership")
    if (
        str(membership.get("path", "")) != str(memberships_path)
        or str(membership.get("audit_path", "")) != str(membership_audit_path)
    ):
        raise DataReadinessError(
            "ER1B membership inputs differ from the frozen ER1A universe paths"
        )
    return {
        "plan_path": str(plan_directory),
        "plan_fingerprint": fingerprint,
        "plan_manifest_sha256": file_sha256(plan_directory / "_manifest.json"),
        "collection_path": str(collection_directory),
        "collection_request_sha256": str(collection["request_sha256"]),
        "collection_manifest_sha256": file_sha256(
            collection_directory / "_manifest.json"
        ),
        "collection_total_rows": int(collection["total_rows"]),
        "membership_sha256": str(membership["sha256"]),
        "universe_snapshot_id": str(membership["universe_snapshot_id"]),
        "first_history_session": str(summary["first_history_session"]),
        "last_history_session": str(summary["last_history_session"]),
        "planned_history_sessions": int(summary["planned_history_sessions"]),
    }


def _build_context_frames(
    *,
    memberships: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    calendar: Any,
    config: ExtendedSessionContextConfig,
    plan_fingerprint: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, int]]:
    window_rows: dict[str, list[dict[str, object]]] = {}
    unit_rows: dict[str, list[dict[str, object]]] = {}
    segment_units = {PREMARKET_SEGMENT: 0, POSTMARKET_SEGMENT: 0}
    segment_rows = {PREMARKET_SEGMENT: 0, POSTMARKET_SEGMENT: 0}
    all_tickers: set[str] = set()
    ticker_sessions = 0
    benchmarks = config.normalized_benchmarks()
    for entry in iter_point_in_time_sessions(
        memberships=memberships,
        sessions=sessions,
        calendar=calendar,
        minimum_cross_section=config.minimum_session_cross_section,
    ):
        session_date = entry.session.date()
        premarket_start = _exchange_moment(
            session_date,
            config.premarket_start_et,
        )
        postmarket_end = _exchange_moment(session_date, config.postmarket_end_et)
        if not premarket_start < entry.open_at <= entry.close_at < postmarket_end:
            raise DataReadinessError(
                f"ER1B extended windows do not bracket the session on "
                f"{session_date}"
            )
        month = entry.month
        window_rows.setdefault(month, []).append(
            {
                "session_date_et": session_date,
                "session_open_utc": entry.open_at,
                "session_close_utc": entry.close_at,
                "premarket_start_utc": premarket_start,
                "premarket_end_utc": entry.open_at,
                "postmarket_start_utc": entry.close_at,
                "postmarket_end_utc": postmarket_end,
                "premarket_minutes": int(
                    (entry.open_at - premarket_start).total_seconds() // 60
                ),
                "postmarket_minutes": int(
                    (postmarket_end - entry.close_at).total_seconds() // 60
                ),
            }
        )
        all_tickers.update(entry.tickers)
        ticker_sessions += len(entry.tickers)
        symbols = sorted(set(entry.tickers).union(benchmarks))
        for segment, start, end in (
            (PREMARKET_SEGMENT, premarket_start, entry.open_at),
            (POSTMARKET_SEGMENT, entry.close_at, postmarket_end),
        ):
            expected_bars = expected_five_minute_bars(start, end)
            for chunk, mapping in chunk_request_symbols(
                symbols,
                expected_bars_per_symbol=expected_bars,
                maximum_symbols_per_unit=config.maximum_symbols_per_unit,
                maximum_expected_rows_per_unit=(
                    config.maximum_expected_rows_per_unit
                ),
                label=f"{session_date} {segment}",
            ):
                unit_id = stable_identity_hash(
                    plan_fingerprint,
                    session_date.isoformat(),
                    segment,
                    start.isoformat(),
                    end.isoformat(),
                    *sorted(mapping),
                    "5Min",
                    "sip",
                    "all",
                )
                unit_rows.setdefault(month, []).append(
                    {
                        **request_unit_record(
                            unit_id=unit_id,
                            session_date=session_date,
                            start=start,
                            end=end,
                            chunk=chunk,
                            mapping=mapping,
                            expected_bars_per_symbol=expected_bars,
                            plan_fingerprint=plan_fingerprint,
                        ),
                        "session_segment": segment,
                        "session_open_utc": entry.open_at,
                        "session_close_utc": entry.close_at,
                    }
                )
                segment_units[segment] += 1
                segment_rows[segment] += expected_bars * len(chunk)
    windows = {
        month: pd.DataFrame(rows, columns=list(SESSION_WINDOW_COLUMNS))
        for month, rows in window_rows.items()
    }
    units = {
        month: pd.DataFrame(rows).sort_values(
            ["requested_start_utc", "unit_id"],
            kind="stable",
        )
        for month, rows in unit_rows.items()
    }
    return (
        windows,
        units,
        {
            "historical_tickers": len(all_tickers),
            "ticker_sessions": ticker_sessions,
            "premarket_units": segment_units[PREMARKET_SEGMENT],
            "postmarket_units": segment_units[POSTMARKET_SEGMENT],
            "acquisition_units": sum(segment_units.values()),
            "maximum_expected_premarket_rows": segment_rows[PREMARKET_SEGMENT],
            "maximum_expected_postmarket_rows": segment_rows[
                POSTMARKET_SEGMENT
            ],
            "maximum_expected_context_rows": sum(segment_rows.values()),
            "benchmark_tickers": len(benchmarks),
        },
    )


def _exchange_moment(session_date: object, clock: str) -> pd.Timestamp:
    return pd.Timestamp(
        f"{session_date} {clock}",
        tz=EXCHANGE_TIMEZONE,
    ).tz_convert("UTC")


def _assert_memory(config: ExtendedSessionContextConfig, stage: str) -> None:
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
