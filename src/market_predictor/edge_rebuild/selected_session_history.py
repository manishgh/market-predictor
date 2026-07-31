"""Hash-bound five-minute planning for the selected in-play stock-sessions.

The two existing layers plan bars for the whole point-in-time universe on every
session. This one plans bars for exactly the stock-sessions that already passed
the frozen two-layer screen, and for nothing else.

That distinction is the whole point. Full history for the 533 screened
companies over the research window would be roughly ten thousand request units;
the selection is one unit per session, about 790, because a session carries a
median of thirteen in-play names and the provider accepts fifty symbols per
request. The bars not requested are bars for symbols that were not moving that
day, which is the population the screen exists to exclude.

The requested window is the exchange session's real open and close, taken from
the XNYS calendar per session. A fixed 09:30-16:00 rule would file three hours
of genuine post-market prints as regular session on the five early closes this
selection contains.

No benchmark or sector symbol is added to a unit. Those already span the entire
research window in the regular-session corpus, so requesting them again would
spend units re-fetching bars that exist and would collide with them when the
stores are materialized.
"""

from __future__ import annotations

import shutil
import uuid
from collections.abc import Iterator
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_contracts import (
    REGULAR_SEGMENT,
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
    SELECTED_SESSION_PLAN_SCHEMA,
    SelectedSessionHistoryConfig,
    SelectedSessionOneMinuteConfig,
)
from market_predictor.edge_rebuild.intraday_history import (
    SELECTED_SESSION_ONE_MINUTE_PLAN_AUTHORITY_SCHEMA,
    SELECTED_SESSION_PLAN_AUTHORITY_SCHEMA,
    chunk_request_symbols,
    expected_five_minute_bars,
    file_record,
    json_sha256,
    request_unit_record,
    stable_identity_hash,
    write_plan_json,
)
from market_predictor.edge_rebuild.intraday_selection import (
    load_complete_intraday_selection,
)
from market_predictor.edge_rebuild.strategy_contract import StrategyContract
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError

SELECTED_SESSION_COLUMNS = (
    "session_date_et",
    "session_open_utc",
    "session_close_utc",
    "session_segment",
    "ticker",
    "session_rank",
    "relative_volume",
    "average_volume_prior_sessions",
    "session_close",
)
REQUIRED_SELECTION_COLUMNS = frozenset(
    {
        "ticker",
        "session_date_et",
        "session_rank",
        "relative_volume",
        "average_volume_prior_sessions",
        "session_close",
    }
)


@dataclass(frozen=True, slots=True)
class SelectedSession:
    """One exchange session and the in-play names selected for it."""

    session: pd.Timestamp
    open_at: pd.Timestamp
    close_at: pd.Timestamp
    selected: pd.DataFrame

    @property
    def month(self) -> str:
        return str(self.session.strftime("%Y-%m"))

    @property
    def tickers(self) -> list[str]:
        return [str(value) for value in self.selected["ticker"]]

    def selection_records(self) -> list[dict[str, object]]:
        return [
            {
                "session_date_et": self.session.date(),
                "session_open_utc": self.open_at,
                "session_close_utc": self.close_at,
                "session_segment": REGULAR_SEGMENT,
                "ticker": str(row["ticker"]),
                "session_rank": int(row["session_rank"]),
                "relative_volume": float(row["relative_volume"]),
                "average_volume_prior_sessions": float(
                    row["average_volume_prior_sessions"]
                ),
                "session_close": float(row["session_close"]),
            }
            for row in self.selected.to_dict(orient="records")
        ]


def build_selected_session_history_plan(
    *,
    selection_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: SelectedSessionHistoryConfig | SelectedSessionOneMinuteConfig,
    strategy_contract: StrategyContract,
    strategy_contract_path: Path,
) -> dict[str, Any]:
    """Plan regular-session bars for a verified selected stock-session set."""

    if output_directory.exists():
        raise DataReadinessError(
            f"selected-session plan output must be new: {output_directory}"
        )
    _assert_memory(config, "selected-session planning start")
    selection, selection_identity = verify_selected_stock_sessions(
        selection_directory
    )
    if (
        selection_identity["strategy_id"]
        != strategy_contract.intraday.strategy_id
        or selection_identity["strategy_contract_sha256"]
        != strategy_contract.sha256()
    ):
        raise DataReadinessError(
            "selected-session plan requires a selection published under the "
            "active intraday strategy contract"
        )
    plan_schema = (
        SELECTED_SESSION_PLAN_SCHEMA
        if config.history_timeframe == "5Min"
        else SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA
    )
    authority_schema = (
        SELECTED_SESSION_PLAN_AUTHORITY_SCHEMA
        if config.history_timeframe == "5Min"
        else SELECTED_SESSION_ONE_MINUTE_PLAN_AUTHORITY_SCHEMA
    )
    request = {
        "schema": plan_schema,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "strategy_contract_path": str(strategy_contract_path),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": strategy_contract.sha256(),
        "selection": selection_identity,
        "session_segments": list(config.session_segments),
        "benchmark_symbols_requested": 0,
        "training_performed": False,
        "download_performed": False,
    }
    plan_fingerprint = json_sha256(request)
    calendar = xcals.get_calendar(config.calendar)
    sessions = _verified_sessions(selection, calendar=calendar)
    selection_frames, unit_frames, totals = _build_plan_frames(
        selection=selection,
        sessions=sessions,
        calendar=calendar,
        config=config,
        plan_fingerprint=plan_fingerprint,
    )
    _assert_memory(config, "selected-session planning frames")
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        for month, frame in sorted(selection_frames.items()):
            path = temporary / "stock_sessions" / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        for month, frame in sorted(unit_frames.items()):
            path = temporary / "units" / config.history_timeframe / f"{month}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            frame.to_parquet(path, index=False)
            files.append(file_record(path, temporary, len(frame)))
        request["plan_fingerprint"] = plan_fingerprint
        write_plan_json(temporary / "_request.json", request)
        files.append(file_record(temporary / "_request.json", temporary, 1))
        manifest: dict[str, Any] = {
            "schema": plan_schema,
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
                "layer": "selected_stock_sessions",
                "segments": list(config.session_segments),
                "window_rule": (
                    "exchange session open to session close, per session, "
                    "never a fixed clock time"
                ),
                "scope": (
                    "exactly the stock-sessions that passed the frozen "
                    "two-layer intraday screen; not full history for the "
                    "selected symbols"
                ),
                "benchmarks_requested": (
                    "none; benchmark and sector bars already span the research "
                    "window in the regular-session corpus"
                ),
                "exact_path_labels": {
                    "timeframe": "1Min",
                    "scope": "all causally screened stock-sessions",
                    "planned_in_this_artifact": (
                        config.history_timeframe == "1Min"
                    ),
                    "missing_trade_policy": "no_trade_no_imputation",
                },
            },
            "summary": {
                "first_history_session": str(sessions[0].date()),
                "last_history_session": str(sessions[-1].date()),
                "planned_history_sessions": len(sessions),
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
                "schema": authority_schema,
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


def verify_selected_stock_sessions(
    directory: Path,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Verify a published screen's authority before reading its selection."""

    manifest = load_complete_intraday_selection(directory)
    table = next(
        (
            record
            for record in manifest["tables"]
            if str(record["path"]) == "selected_stock_sessions.parquet"
        ),
        None,
    )
    if table is None:
        raise DataReadinessError(
            f"published screen registers no selected stock-sessions: {directory}"
        )
    path = directory / "selected_stock_sessions.parquet"
    frame = pd.read_parquet(path)
    missing = sorted(REQUIRED_SELECTION_COLUMNS.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"selected stock-sessions lack required columns: {missing}"
        )
    frame = frame.loc[:, sorted(REQUIRED_SELECTION_COLUMNS)].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["session_date_et"] = frame["session_date_et"].astype(str)
    if (
        frame.empty
        or bool(frame["ticker"].eq("").any())
        or bool(frame.duplicated(["ticker", "session_date_et"]).any())
    ):
        raise DataReadinessError(
            "selected stock-sessions are empty or not unique per symbol-session"
        )
    return frame, {
        "path": str(directory),
        "manifest_sha256": file_sha256(directory / "_manifest.json"),
        "request_sha256": str(manifest["request_sha256"]),
        "table_sha256": str(table["sha256"]),
        "strategy_id": str(manifest["strategy_id"]),
        "strategy_contract_sha256": str(manifest["strategy_contract_sha256"]),
        "stock_sessions": int(len(frame)),
        "symbols": int(frame["ticker"].nunique()),
        "sessions": int(frame["session_date_et"].nunique()),
        "first_session_et": str(frame["session_date_et"].min()),
        "last_session_et": str(frame["session_date_et"].max()),
        "research_only": True,
    }


def _verified_sessions(
    selection: pd.DataFrame,
    *,
    calendar: Any,
) -> pd.DatetimeIndex:
    """Every selected date must be a real exchange session, not merely a date."""

    selected = sorted(set(selection["session_date_et"]))
    sessions = calendar.sessions_in_range(selected[0], selected[-1])
    known = {str(pd.Timestamp(session).date()) for session in sessions}
    unknown = sorted(set(selected).difference(known))
    if unknown:
        raise DataReadinessError(
            f"selected dates are not exchange sessions: {unknown[:5]}"
        )
    return pd.DatetimeIndex(
        [pd.Timestamp(session) for session in selected],
        name="session",
    )


def iter_selected_sessions(
    *,
    selection: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    calendar: Any,
) -> Iterator[SelectedSession]:
    """Yield each selected session with its real open, close, and names."""

    by_session = dict(list(selection.groupby("session_date_et", sort=False)))
    for session in sessions:
        key = str(session.date())
        selected = by_session[key].sort_values("ticker", kind="stable")
        open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
        close_at = pd.Timestamp(calendar.session_close(session)).tz_convert("UTC")
        if not open_at < close_at:
            raise DataReadinessError(
                f"exchange session {key} has no positive regular window"
            )
        yield SelectedSession(
            session=session,
            open_at=open_at,
            close_at=close_at,
            selected=selected,
        )


def _build_plan_frames(
    *,
    selection: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    calendar: Any,
    config: SelectedSessionHistoryConfig | SelectedSessionOneMinuteConfig,
    plan_fingerprint: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, int]]:
    selection_rows: dict[str, list[dict[str, object]]] = {}
    unit_rows: dict[str, list[dict[str, object]]] = {}
    all_tickers: set[str] = set()
    stock_sessions = 0
    unit_count = 0
    expected_rows = 0
    early_closes = 0
    for entry in iter_selected_sessions(
        selection=selection,
        sessions=sessions,
        calendar=calendar,
    ):
        month = entry.month
        selection_rows.setdefault(month, []).extend(entry.selection_records())
        symbols = sorted(set(entry.tickers))
        all_tickers.update(symbols)
        stock_sessions += len(symbols)
        session_minutes = int((entry.close_at - entry.open_at).total_seconds() // 60)
        expected_bars = (
            expected_five_minute_bars(entry.open_at, entry.close_at)
            if config.history_timeframe == "5Min"
            else session_minutes
        )
        expected_full_session_bars = 78 if config.history_timeframe == "5Min" else 390
        if expected_bars != expected_full_session_bars:
            early_closes += 1
        for chunk, mapping in chunk_request_symbols(
            symbols,
            expected_bars_per_symbol=expected_bars,
            maximum_symbols_per_unit=config.maximum_symbols_per_unit,
            maximum_expected_rows_per_unit=config.maximum_expected_rows_per_unit,
            label=f"{entry.session.date()} {REGULAR_SEGMENT}",
        ):
            unit_id = stable_identity_hash(
                plan_fingerprint,
                entry.session.date().isoformat(),
                REGULAR_SEGMENT,
                entry.open_at.isoformat(),
                entry.close_at.isoformat(),
                *sorted(mapping),
                config.history_timeframe,
                "sip",
                "all",
            )
            unit_rows.setdefault(month, []).append(
                {
                    **request_unit_record(
                        unit_id=unit_id,
                        session_date=entry.session.date(),
                        start=entry.open_at,
                        end=entry.close_at,
                        chunk=chunk,
                        mapping=mapping,
                        expected_bars_per_symbol=expected_bars,
                        plan_fingerprint=plan_fingerprint,
                    ),
                    "session_segment": REGULAR_SEGMENT,
                    "session_open_utc": entry.open_at,
                    "session_close_utc": entry.close_at,
                }
            )
            unit_count += 1
            expected_rows += expected_bars * len(chunk)
    selections = {
        month: pd.DataFrame(rows, columns=list(SELECTED_SESSION_COLUMNS))
        for month, rows in selection_rows.items()
    }
    units = {
        month: pd.DataFrame(rows).sort_values(
            ["requested_start_utc", "unit_id"],
            kind="stable",
        )
        for month, rows in unit_rows.items()
    }
    return (
        selections,
        units,
        {
            "historical_tickers": len(all_tickers),
            "stock_sessions": stock_sessions,
            "acquisition_units": unit_count,
            "maximum_expected_feature_rows": expected_rows,
            "early_close_sessions": early_closes,
            "benchmark_tickers": 0,
        },
    )


def _assert_memory(
    config: SelectedSessionHistoryConfig | SelectedSessionOneMinuteConfig,
    stage: str,
) -> None:
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
