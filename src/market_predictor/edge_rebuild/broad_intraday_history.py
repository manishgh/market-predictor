"""Plan missing broad-universe regular-session five-minute history."""

from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping
from datetime import UTC, date, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_contracts import (
    BROAD_INTRADAY_HISTORY_PLAN_SCHEMA,
    BroadIntradayHistoryConfig,
)
from market_predictor.edge_rebuild.intraday_history import (
    BROAD_INTRADAY_HISTORY_PLAN_AUTHORITY_SCHEMA,
    chunk_request_symbols,
    expected_five_minute_bars,
    file_record,
    json_sha256,
    load_complete_intraday_history_plan,
    request_unit_record,
    stable_identity_hash,
    write_plan_json,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.v3.errors import DataReadinessError

_MEMBERSHIP_COLUMNS = {
    "ticker",
    "security_id",
    "effective_from_utc",
    "effective_to_utc",
    "available_at_utc",
    "sector",
    "industry",
    "source",
    "availability_policy",
}
_CORPUS_SCHEMA = "edge_rebuild.intraday_materialization.v1"
_CORPUS_AUTHORITY_SCHEMA = "edge_rebuild.intraday_materialization_authority.v1"


def build_broad_intraday_history_plan(
    *,
    broad_memberships_path: Path,
    pit_memberships_path: Path,
    existing_corpus_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: BroadIntradayHistoryConfig,
) -> dict[str, Any]:
    """Publish only missing ticker-session requests for causal RVOL history."""

    if output_directory.exists():
        raise DataReadinessError(f"broad intraday plan output must be new: {output_directory}")
    _guard_memory(config, "broad intraday planning start")
    broad, broad_identity = _load_memberships(
        broad_memberships_path,
        role="current_snapshot_proxy",
    )
    pit, pit_identity = _load_memberships(
        pit_memberships_path,
        role="verified_sp500_point_in_time",
    )
    exclusions = set(config.normalized_fund_exclusions())
    broad, broad_excluded = _exclude_funds(broad, exclusions)
    pit, pit_excluded = _exclude_funds(pit, exclusions)
    sessions, bounds = _session_bounds(config)
    potential_tickers = set(broad["ticker"]).union(pit["ticker"])
    covered, corpus_identity = _load_existing_coverage(
        existing_corpus_directory,
        potential_tickers=potential_tickers,
        bounds=bounds,
        first_session=config.first_session,
        last_session=config.last_session,
    )
    request: dict[str, Any] = {
        "schema": BROAD_INTRADAY_HISTORY_PLAN_SCHEMA,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "window": {
            "first_session": config.first_session.isoformat(),
            "last_session": config.last_session.isoformat(),
            "calendar": config.calendar,
        },
        "broad_membership_proxy": broad_identity,
        "verified_sp500_membership": pit_identity,
        "existing_canonical_corpus": corpus_identity,
        "historical_membership_authority": False,
        "research_only_reason": (
            "the broad non-index universe is reconstructed from one current Finviz snapshot and is not point-in-time membership"
        ),
        "training_performed": False,
        "download_performed": False,
    }
    plan_fingerprint = json_sha256(request)
    missing_frames, unit_frames, totals = _build_missing_frames(
        broad=broad,
        pit=pit,
        sessions=sessions,
        bounds=bounds,
        covered=covered,
        config=config,
        plan_fingerprint=plan_fingerprint,
    )
    _guard_memory(config, "broad intraday planning frames")
    temporary = output_directory.with_name(f".{output_directory.name}.{uuid.uuid4().hex}.tmp")
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        for month, frame in sorted(missing_frames.items()):
            path = temporary / "missing_symbol_sessions" / f"{month}.parquet"
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
            "schema": BROAD_INTRADAY_HISTORY_PLAN_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "plan_fingerprint": plan_fingerprint,
            "policy_sha256": config.sha256(),
            "research_only": True,
            "promotion_eligible": False,
            "historical_membership_authority": False,
            "membership_limitations": {
                "broad_non_index": ("current_snapshot_proxy; constituents and classifications are not historically point-in-time"),
                "sp500": ("verified point-in-time intervals with provider-publication availability proxy"),
            },
            "fund_exclusion": {
                "industry_rule": "membership industry identifies ETF/fund",
                "explicit_tickers": sorted(exclusions),
                "excluded_tickers": sorted(broad_excluded.union(pit_excluded)),
            },
            "acquisition": {
                "provider": "alpaca",
                "calendar": config.calendar,
                "calendar_version": version("exchange-calendars"),
                "timeframe": "5Min",
                "session_segment": "regular",
                "price_feed": "sip",
                "adjustment": "all",
                "one_minute_paths_planned": False,
                "coverage_policy": (
                    "request a ticker-session only when the hash-verified canonical regular 5m corpus has no valid observed row"
                ),
            },
            "summary": {
                "first_session": config.first_session.isoformat(),
                "last_session": config.last_session.isoformat(),
                "calendar_sessions": len(sessions),
                "broad_proxy_symbols": int(broad["ticker"].nunique()),
                "sp500_pit_symbols": int(pit["ticker"].nunique()),
                "excluded_fund_symbols": len(broad_excluded.union(pit_excluded)),
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
                "schema": BROAD_INTRADAY_HISTORY_PLAN_AUTHORITY_SCHEMA,
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


def load_complete_broad_intraday_history_plan(directory: Path) -> dict[str, Any]:
    """Verify the generic transport authority and broad-plan restrictions."""

    manifest = load_complete_intraday_history_plan(directory)
    acquisition = manifest.get("acquisition")
    if (
        manifest.get("schema") != BROAD_INTRADAY_HISTORY_PLAN_SCHEMA
        or manifest.get("research_only") is not True
        or manifest.get("promotion_eligible") is not False
        or manifest.get("historical_membership_authority") is not False
        or not isinstance(acquisition, Mapping)
        or acquisition.get("timeframe") != "5Min"
        or acquisition.get("session_segment") != "regular"
        or acquisition.get("price_feed") != "sip"
        or acquisition.get("adjustment") != "all"
        or acquisition.get("one_minute_paths_planned") is not False
    ):
        raise DataReadinessError("broad intraday plan restrictions do not verify")
    return manifest


def _load_memberships(path: Path, *, role: str) -> tuple[pd.DataFrame, dict[str, Any]]:
    sidecar = Path(f"{path}.manifest.json")
    if not path.is_file() or not sidecar.is_file():
        raise DataReadinessError(f"{role} membership input is missing")
    manifest = _load_json(sidecar)
    frame = pd.read_parquet(path)
    if (
        manifest.get("artifact_type") != "memberships"
        or manifest.get("artifact_sha256") != file_sha256(path)
        or int(manifest.get("rows", -1)) != len(frame)
        or not _MEMBERSHIP_COLUMNS.issubset(frame.columns)
        or frame.empty
    ):
        raise DataReadinessError(f"{role} membership authority is invalid")
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="coerce")
    if (
        bool(frame["ticker"].eq("").any())
        or bool(frame["security_id"].astype(str).str.strip().eq("").any())
        or bool(frame["effective_from_utc"].isna().any())
        or bool((frame["effective_to_utc"].notna() & frame["effective_to_utc"].le(frame["effective_from_utc"])).any())
    ):
        raise DataReadinessError(f"{role} membership rows are invalid")
    if role == "current_snapshot_proxy":
        if set(frame["source"].astype(str)) != {"finviz_current_snapshot"}:
            raise DataReadinessError("broad membership must remain an explicit current-snapshot proxy")
        historical_authority = False
    else:
        if "finviz_current_snapshot" in set(frame["source"].astype(str)):
            raise DataReadinessError("S&P membership must be point-in-time evidence")
        historical_authority = True
    _reject_overlaps(frame, role)
    return frame, {
        "role": role,
        "path": str(path),
        "sha256": file_sha256(path),
        "manifest_path": str(sidecar),
        "manifest_sha256": file_sha256(sidecar),
        "manifest_created_at_utc": str(manifest.get("created_at_utc", "")),
        "rows": len(frame),
        "symbols": int(frame["ticker"].nunique()),
        "historical_membership_authority": historical_authority,
        "availability_limitation": (
            "snapshot observation time does not establish historical constituency"
            if not historical_authority
            else "membership availability uses a provider-publication proxy"
        ),
    }


def _reject_overlaps(frame: pd.DataFrame, role: str) -> None:
    maximum = pd.Timestamp.max.tz_localize("UTC")
    for ticker, group in frame.groupby("ticker", sort=False):
        previous_end: pd.Timestamp | None = None
        for row in group.sort_values("effective_from_utc").itertuples(index=False):
            start = pd.Timestamp(row.effective_from_utc)
            if previous_end is not None and start < previous_end:
                raise DataReadinessError(f"{role} membership overlaps for {ticker}")
            previous_end = maximum if pd.isna(row.effective_to_utc) else pd.Timestamp(row.effective_to_utc)


def _exclude_funds(
    frame: pd.DataFrame,
    explicit: set[str],
) -> tuple[pd.DataFrame, set[str]]:
    industry = frame["industry"].fillna("").astype(str).str.lower()
    industry_fund = industry.str.contains(
        r"\b(?:exchange[\s-]*traded[\s-]*fund|etf)\b",
        regex=True,
    )
    excluded_mask = frame["ticker"].isin(explicit) | industry_fund
    excluded = set(frame.loc[excluded_mask, "ticker"].astype(str))
    return frame.loc[~excluded_mask].copy(), excluded


def _session_bounds(
    config: BroadIntradayHistoryConfig,
) -> tuple[pd.DatetimeIndex, dict[date, tuple[pd.Timestamp, pd.Timestamp]]]:
    calendar = xcals.get_calendar(config.calendar)
    sessions = calendar.sessions_in_range(
        pd.Timestamp(config.first_session),
        pd.Timestamp(config.last_session),
    )
    if sessions.empty or sessions[0].date() != config.first_session or sessions[-1].date() != config.last_session:
        raise DataReadinessError("configured history bounds must be XNYS sessions")
    bounds = {
        session.date(): (
            pd.Timestamp(calendar.session_open(session)).tz_convert("UTC"),
            pd.Timestamp(calendar.session_close(session)).tz_convert("UTC"),
        )
        for session in sessions
    }
    return sessions, bounds


def _load_existing_coverage(
    directory: Path,
    *,
    potential_tickers: set[str],
    bounds: Mapping[date, tuple[pd.Timestamp, pd.Timestamp]],
    first_session: date,
    last_session: date,
) -> tuple[set[tuple[str, date]], dict[str, Any]]:
    manifest_path = directory / "_manifest.json"
    authority_path = directory / "_authority.json"
    manifest = _load_json(manifest_path)
    authority = _load_json(authority_path)
    integrity = manifest.get("integrity")
    if (
        manifest.get("schema") != _CORPUS_SCHEMA
        or authority.get("schema") != _CORPUS_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(manifest_path)
        or not isinstance(integrity, Mapping)
        or int(integrity.get("blocking_defect_count", -1)) != 0
        or integrity.get("identity_breaks") not in ([], None)
        or integrity.get("fabricated_bars") not in ([], None)
        or date.fromisoformat(str(manifest.get("window_first_session"))) > first_session
        or date.fromisoformat(str(manifest.get("window_last_session"))) < last_session
    ):
        raise DataReadinessError("canonical regular 5m corpus authority is invalid")
    truncated = {
        (str(row["ticker"]), date.fromisoformat(str(row["session"])))
        for row in integrity.get("truncated_ticker_sessions", [])
        if isinstance(row, Mapping) and "ticker" in row and "session" in row
    }
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list):
        raise DataReadinessError("canonical corpus file inventory is invalid")
    entries: dict[str, Mapping[str, Any]] = {}
    for raw in raw_files:
        if not isinstance(raw, Mapping) or raw.get("store") != "regular":
            continue
        relative = str(raw.get("path", ""))
        ticker = str(raw.get("ticker", "")).upper()
        if ticker not in potential_tickers:
            continue
        if not relative.replace("\\", "/").startswith("regular/5m/") or ticker in entries:
            raise DataReadinessError("canonical corpus regular timeframe is invalid")
        entries[ticker] = raw
    covered: set[tuple[str, date]] = set()
    consulted: list[dict[str, Any]] = []
    for ticker, raw in sorted(entries.items()):
        path = _resolve_inside(directory, str(raw["path"]))
        if not path.is_file() or file_sha256(path) != raw.get("sha256"):
            raise DataReadinessError(f"canonical corpus file does not verify: {path}")
        frame = pd.read_parquet(
            path,
            columns=[
                "ticker",
                "session_date_et",
                "session_segment",
                "timeframe",
                "bar_start_utc",
                "source",
                "price_feed",
                "adjustment",
            ],
        )
        if len(frame) != int(raw.get("rows", -1)):
            raise DataReadinessError(f"canonical corpus row count differs: {path}")
        _validate_corpus_frame(frame, ticker=ticker, bounds=bounds)
        dates = pd.to_datetime(frame["session_date_et"], errors="raise").dt.date
        for session in dates.unique():
            if first_session <= session <= last_session:
                covered.add((ticker, session))
        consulted.append(
            {
                "path": str(path),
                "sha256": str(raw["sha256"]),
                "rows": len(frame),
            }
        )
    replanned_truncated = len(truncated.intersection(covered))
    covered.difference_update(truncated)
    return covered, {
        "path": str(directory),
        "manifest_sha256": file_sha256(manifest_path),
        "authority_sha256": file_sha256(authority_path),
        "schema": _CORPUS_SCHEMA,
        "source": "alpaca",
        "timeframe": "5Min",
        "price_feed": "sip",
        "adjustment": "all",
        "consulted_regular_files": consulted,
        "covered_ticker_sessions": len(covered),
        "truncated_ticker_sessions_replanned": replanned_truncated,
    }


def _validate_corpus_frame(
    frame: pd.DataFrame,
    *,
    ticker: str,
    bounds: Mapping[date, tuple[pd.Timestamp, pd.Timestamp]],
) -> None:
    starts = pd.to_datetime(frame["bar_start_utc"], utc=True, errors="coerce")
    if (
        frame.empty
        or bool(starts.isna().any())
        or set(frame["ticker"].astype(str).str.upper()) != {ticker}
        or set(frame["session_segment"].astype(str)) != {"regular"}
        or set(frame["timeframe"].astype(str).str.lower()) != {"5m"}
        or set(frame["source"].astype(str).str.lower()) != {"alpaca"}
        or set(frame["price_feed"].astype(str).str.lower()) != {"sip"}
        or set(frame["adjustment"].astype(str).str.lower()) != {"all"}
        or bool((starts.dt.minute.mod(5).ne(0) | starts.dt.second.ne(0) | starts.dt.microsecond.ne(0)).any())
    ):
        raise DataReadinessError(f"canonical regular 5m identity failed for {ticker}")
    dates = pd.to_datetime(frame["session_date_et"], errors="raise").dt.date
    relevant = dates.isin(bounds)
    for session in dates[relevant].unique():
        open_at, close_at = bounds[session]
        session_starts = starts[dates.eq(session)]
        if bool(session_starts.lt(open_at).any() or session_starts.ge(close_at).any()):
            raise DataReadinessError(f"canonical regular bars exceed XNYS bounds for {ticker} {session}")


def _build_missing_frames(
    *,
    broad: pd.DataFrame,
    pit: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    bounds: Mapping[date, tuple[pd.Timestamp, pd.Timestamp]],
    covered: set[tuple[str, date]],
    config: BroadIntradayHistoryConfig,
    plan_fingerprint: str,
) -> tuple[dict[str, pd.DataFrame], dict[str, pd.DataFrame], dict[str, int]]:
    missing_rows: dict[str, list[dict[str, Any]]] = {}
    unit_rows: dict[str, list[dict[str, Any]]] = {}
    eligible_count = 0
    existing_count = 0
    expected_rows = 0
    early_closes = 0
    symbols: set[str] = set()
    for session in sessions:
        session_date = session.date()
        open_at, close_at = bounds[session_date]
        active_broad = _active_memberships(broad, open_at)
        active_pit = _active_memberships(pit, open_at)
        active = {str(row.ticker): (row, "current_snapshot_proxy") for row in active_broad.itertuples(index=False)}
        active.update({str(row.ticker): (row, "sp500_point_in_time") for row in active_pit.itertuples(index=False)})
        eligible_count += len(active)
        missing = [ticker for ticker in sorted(active) if (ticker, session_date) not in covered]
        existing_count += len(active) - len(missing)
        if not missing:
            continue
        month = session.strftime("%Y-%m")
        symbols.update(missing)
        for ticker in missing:
            row, basis = active[ticker]
            missing_rows.setdefault(month, []).append(
                {
                    "session_date_et": session_date,
                    "session_open_utc": open_at,
                    "session_close_utc": close_at,
                    "ticker": ticker,
                    "security_id": str(row.security_id),
                    "sector": str(row.sector),
                    "industry": str(row.industry),
                    "universe_basis": basis,
                    "existing_regular_5m_rows": 0,
                    "plan_fingerprint": plan_fingerprint,
                }
            )
        bars_per_symbol = expected_five_minute_bars(open_at, close_at)
        if bars_per_symbol < 78:
            early_closes += 1
        for index, (chunk, mapping) in enumerate(
            chunk_request_symbols(
                missing,
                expected_bars_per_symbol=bars_per_symbol,
                maximum_symbols_per_unit=config.maximum_symbols_per_unit,
                maximum_expected_rows_per_unit=config.maximum_expected_rows_per_unit,
                label=session_date.isoformat(),
            )
        ):
            unit_id = stable_identity_hash(
                plan_fingerprint,
                session_date.isoformat(),
                open_at.isoformat(),
                close_at.isoformat(),
                str(index),
                *sorted(mapping),
                "5Min",
                "sip",
                "all",
            )
            unit_rows.setdefault(month, []).append(
                request_unit_record(
                    unit_id=unit_id,
                    session_date=session_date,
                    start=open_at,
                    end=close_at,
                    chunk=chunk,
                    mapping=mapping,
                    expected_bars_per_symbol=bars_per_symbol,
                    plan_fingerprint=plan_fingerprint,
                    timeframe="5Min",
                )
            )
            expected_rows += bars_per_symbol * len(chunk)
    missing_frames = {
        month: pd.DataFrame(rows).sort_values(["session_open_utc", "ticker"], kind="stable") for month, rows in missing_rows.items()
    }
    units = {month: pd.DataFrame(rows).sort_values(["requested_start_utc", "unit_id"], kind="stable") for month, rows in unit_rows.items()}
    return (
        missing_frames,
        units,
        {
            "eligible_symbol_sessions": eligible_count,
            "existing_symbol_sessions_subtracted": existing_count,
            "missing_symbol_sessions": eligible_count - existing_count,
            "missing_symbols": len(symbols),
            "planned_history_sessions": sum(frame["session_date_et"].nunique() for frame in missing_frames.values()),
            "acquisition_units": sum(len(frame) for frame in units.values()),
            "maximum_expected_rows": expected_rows,
            "early_close_sessions_with_missing_coverage": early_closes,
        },
    )


def _active_memberships(frame: pd.DataFrame, open_at: pd.Timestamp) -> pd.DataFrame:
    active = frame[frame["effective_from_utc"].le(open_at) & (frame["effective_to_utc"].isna() | frame["effective_to_utc"].gt(open_at))]
    if bool(active["ticker"].duplicated().any()):
        raise DataReadinessError("membership is ambiguous at a session open")
    return active


def _resolve_inside(root: Path, relative: str) -> Path:
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise DataReadinessError("artifact path escapes the canonical corpus")
    resolved_root = root.resolve()
    resolved = (root / candidate).resolve()
    if resolved != resolved_root and resolved_root not in resolved.parents:
        raise DataReadinessError("artifact path escapes the canonical corpus")
    return resolved


def _load_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, TypeError) as exc:
        raise DataReadinessError(f"JSON artifact is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"JSON artifact is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def _guard_memory(config: BroadIntradayHistoryConfig, stage: str) -> None:
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
