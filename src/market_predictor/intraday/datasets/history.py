"""Hash-bound ER1A historical intraday acquisition planning."""
from __future__ import annotations



import hashlib
import json
import shutil
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from importlib.metadata import version
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.history_contracts import (
    BROAD_INTRADAY_HISTORY_PLAN_SCHEMA,
    EXTENDED_CONTEXT_PLAN_SCHEMA,
    INTRADAY_HISTORY_PLAN_SCHEMA,
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA,
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA,
    SELECTED_SESSION_PLAN_SCHEMA,
    IntradayHistoryConfig,
)
from market_predictor.edge_rebuild.readiness import (
    load_complete_readiness_audit,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
)
from market_predictor.symbols import provider_symbol
from market_predictor.core.errors import DataReadinessError

PLAN_AUTHORITY_SCHEMA = "edge_rebuild.intraday_history_plan_authority.v1"
EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA = (
    "edge_rebuild.extended_session_context_plan_authority.v1"
)
SELECTED_SESSION_PLAN_AUTHORITY_SCHEMA = (
    "edge_rebuild.selected_session_history_plan_authority.v1"
)
SELECTED_SESSION_ONE_MINUTE_PLAN_AUTHORITY_SCHEMA = (
    "edge_rebuild.selected_session_one_minute_plan_authority.v1"
)
SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA = (
    "edge_rebuild.selected_session_benchmark_one_minute_plan_authority.v1"
)
BROAD_INTRADAY_HISTORY_PLAN_AUTHORITY_SCHEMA = (
    "edge_rebuild.broad_intraday_history_plan_authority.v1"
)
ACCEPTED_PLAN_SCHEMAS = {
    INTRADAY_HISTORY_PLAN_SCHEMA: PLAN_AUTHORITY_SCHEMA,
    EXTENDED_CONTEXT_PLAN_SCHEMA: EXTENDED_CONTEXT_PLAN_AUTHORITY_SCHEMA,
    SELECTED_SESSION_PLAN_SCHEMA: SELECTED_SESSION_PLAN_AUTHORITY_SCHEMA,
    SELECTED_SESSION_ONE_MINUTE_PLAN_SCHEMA: (
        SELECTED_SESSION_ONE_MINUTE_PLAN_AUTHORITY_SCHEMA
    ),
    SELECTED_SESSION_BENCHMARK_PLAN_SCHEMA: (
        SELECTED_SESSION_BENCHMARK_PLAN_AUTHORITY_SCHEMA
    ),
    BROAD_INTRADAY_HISTORY_PLAN_SCHEMA: BROAD_INTRADAY_HISTORY_PLAN_AUTHORITY_SCHEMA,
}
SESSION_MEMBERSHIP_COLUMNS = (
    "session_date_et",
    "session_open_utc",
    "session_close_utc",
    "ticker",
    "security_id",
    "sector",
    "primary_benchmark",
    "universe_snapshot_id",
    "effective_from_utc",
    "effective_to_utc",
)
REQUIRED_MEMBERSHIP_COLUMNS = {
    "ticker",
    "security_id",
    "effective_from_utc",
    "effective_to_utc",
    "sector",
    "primary_benchmark",
    "universe_snapshot_id",
    "membership_source_urls",
}


def build_intraday_history_plan(
    *,
    readiness_audit_directory: Path,
    memberships_path: Path,
    membership_audit_path: Path,
    existing_stock_bars_directory: Path,
    existing_benchmark_bars_directory: Path,
    policy_path: Path,
    output_directory: Path,
    config: IntradayHistoryConfig,
) -> dict[str, Any]:
    """Plan PIT five-minute history; exact one-minute paths remain selective."""

    if output_directory.exists():
        raise DataReadinessError(
            f"ER1A history plan output must be new: {output_directory}"
        )
    _assert_memory(config, "ER1A history planning start")
    readiness = _verify_readiness_audit(readiness_audit_directory)
    existing_sessions = _existing_usable_sessions(
        readiness_audit_directory
    )
    if existing_sessions >= config.target_usable_sessions:
        raise DataReadinessError(
            "ER1A history acquisition is unnecessary at the frozen target"
        )
    memberships, membership_identity = verify_point_in_time_memberships(
        memberships_path,
        membership_audit_path,
        minimum_cross_section=config.minimum_session_cross_section,
    )
    stock_identity = verify_existing_ohlcv_identity(
        existing_stock_bars_directory,
        config=config,
    )
    benchmark_identity = verify_existing_ohlcv_identity(
        existing_benchmark_bars_directory,
        config=config,
    )
    if stock_identity["start_utc"] != benchmark_identity["start_utc"]:
        raise DataReadinessError(
            "existing stock and benchmark histories start at different times"
        )
    request = {
        "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
        "policy_path": str(policy_path),
        "policy_file_sha256": file_sha256(policy_path),
        "policy_sha256": config.sha256(),
        "readiness": readiness,
        "membership": membership_identity,
        "existing_stock_bars": stock_identity,
        "existing_benchmark_bars": benchmark_identity,
        "training_performed": False,
        "download_performed": False,
    }
    plan_fingerprint = json_sha256(request)
    calendar = xcals.get_calendar(config.calendar)
    first_existing_session = calendar.minute_to_session(
        pd.Timestamp(stock_identity["start_utc"]),
        direction="next",
    )
    final_history_session = calendar.previous_session(first_existing_session)
    required_new_usable = config.target_usable_sessions - existing_sessions
    planned_sessions = required_new_usable + config.feature_warmup_sessions
    sessions = calendar.sessions_window(
        final_history_session,
        -planned_sessions,
    )
    if len(sessions) != planned_sessions:
        raise DataReadinessError("ER1A calendar returned the wrong session count")
    session_frames, unit_frames, totals = _build_plan_frames(
        memberships=memberships,
        sessions=sessions,
        calendar=calendar,
        config=config,
        plan_fingerprint=plan_fingerprint,
    )
    _assert_memory(config, "ER1A history planning frames")
    temporary = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.tmp"
    )
    temporary.mkdir(parents=True)
    try:
        files: list[dict[str, Any]] = []
        for month, frame in sorted(session_frames.items()):
            path = temporary / "session_memberships" / f"{month}.parquet"
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
        files.append(
            file_record(
                temporary / "_request.json",
                temporary,
                1,
            )
        )
        manifest: dict[str, Any] = {
            "schema": INTRADAY_HISTORY_PLAN_SCHEMA,
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
                "feature_discovery": {
                    "timeframe": "5Min",
                    "scope": "full_point_in_time_universe",
                },
                "exact_path_labels": {
                    "timeframe": "1Min",
                    "scope": "selective_after_causal_setup_extraction",
                    "planned_in_this_artifact": False,
                    "missing_trade_policy": "no_trade_no_imputation",
                },
            },
            "summary": {
                "existing_usable_sessions": existing_sessions,
                "target_usable_sessions": config.target_usable_sessions,
                "minimum_usable_sessions": config.minimum_usable_sessions,
                "required_new_usable_sessions": required_new_usable,
                "feature_warmup_sessions": config.feature_warmup_sessions,
                "planned_history_sessions": planned_sessions,
                "first_history_session": sessions[0].date().isoformat(),
                "last_history_session": sessions[-1].date().isoformat(),
                "memory": memory_audit(
                    hard_budget_gib=config.maximum_process_memory_gib,
                    headroom_gib=config.memory_guard_headroom_gib,
                ).to_record(),
                **totals,
            },
            "files": sorted(files, key=lambda item: str(item["path"])),
        }
        write_plan_json(temporary / "_manifest.json", manifest)
        authority = {
            "schema": PLAN_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(temporary / "_manifest.json"),
            "plan_fingerprint": plan_fingerprint,
        }
        write_plan_json(temporary / "_authority.json", authority)
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        temporary.replace(output_directory)
        return manifest
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


def load_complete_intraday_history_plan(
    directory: Path,
    *,
    accepted_schemas: Mapping[str, str] = ACCEPTED_PLAN_SCHEMAS,
) -> dict[str, Any]:
    """Verify a five-minute acquisition plan against its complete authority.

    `accepted_schemas` maps each recognised plan schema to the exact authority
    schema that may sign it. An unregistered schema fails closed, so one
    layer's plan can never be replayed under another layer's identity.
    """

    request = load_plan_json(directory / "_request.json")
    manifest = load_plan_json(directory / "_manifest.json")
    authority = load_plan_json(directory / "_authority.json")
    fingerprint = str(request.get("plan_fingerprint", ""))
    plan_schema = str(manifest.get("schema", ""))
    unhashed_request = {
        key: value
        for key, value in request.items()
        if key != "plan_fingerprint"
    }
    if (
        not fingerprint
        or json_sha256(unhashed_request) != fingerprint
        or plan_schema not in accepted_schemas
        or manifest.get("plan_fingerprint") != fingerprint
        or authority.get("schema") != accepted_schemas[plan_schema]
        or authority.get("state") != "complete"
        or authority.get("plan_fingerprint") != fingerprint
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
    ):
        raise DataReadinessError(
            "five-minute acquisition plan lacks matching complete authority"
        )
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("ER1A plan has no registered files")
    expected = {"_manifest.json", "_authority.json"}
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("ER1A plan file record is malformed")
        relative = str(raw.get("path", ""))
        path = directory / relative
        expected.add(relative.replace("/", "\\"))
        if (
            Path(relative).is_absolute()
            or ".." in Path(relative).parts
            or not path.is_file()
            or path.stat().st_size != int(raw.get("bytes", -1))
            or file_sha256(path) != raw.get("sha256")
        ):
            raise DataReadinessError(
                f"ER1A plan artifact does not verify: {path}"
            )
    actual = {
        str(path.relative_to(directory))
        for path in directory.rglob("*")
        if path.is_file()
    }
    if actual != expected:
        raise DataReadinessError("ER1A plan artifact file set differs")
    return manifest


def _verify_readiness_audit(directory: Path) -> dict[str, object]:
    request = load_plan_json(directory / "_request.json")
    request_sha256 = json_sha256(request)
    load_complete_readiness_audit(
        directory,
        expected_request_sha256=request_sha256,
    )
    summary = load_plan_json(directory / "summary.json")
    acquisition = summary.get("acquisition_plan")
    if (
        summary.get("status") != "blocked_pending_targeted_acquisition"
        or summary.get("er2_authorized") is not False
        or not isinstance(acquisition, Mapping)
        or acquisition.get("authorized_by_audit") is not True
        or str(acquisition.get("feed", "")).lower() != "sip"
        or str(acquisition.get("adjustment", "")).lower() != "all"
    ):
        raise DataReadinessError(
            "ER1A requires a verified audit blocked only on acquisition"
        )
    return {
        "path": str(directory),
        "request_sha256": request_sha256,
        "manifest_sha256": file_sha256(directory / "_manifest.json"),
        "summary_sha256": file_sha256(directory / "summary.json"),
        "status": summary["status"],
    }


def _existing_usable_sessions(directory: Path) -> int:
    calendar = pd.read_csv(
        directory / "session_calendar.csv",
        usecols=["strategy_id", "session_date_et"],
    )
    intraday = calendar[
        calendar["strategy_id"].astype(str).str.startswith("INTRADAY.")
    ]
    count = int(intraday["session_date_et"].nunique())
    if count < 1:
        raise DataReadinessError("ER1 audit has no usable intraday sessions")
    return count


def verify_point_in_time_memberships(
    path: Path,
    audit_path: Path,
    *,
    minimum_cross_section: int,
) -> tuple[pd.DataFrame, dict[str, object]]:
    """Load and verify the one sourced point-in-time universe."""

    if not path.is_file() or not audit_path.is_file():
        raise DataReadinessError("ER1A membership inputs are missing")
    frame = pd.read_parquet(path)
    missing = REQUIRED_MEMBERSHIP_COLUMNS.difference(frame.columns)
    if missing:
        raise DataReadinessError(
            f"ER1A memberships lack required columns: {sorted(missing)}"
        )
    frame = frame.copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["effective_from_utc"] = pd.to_datetime(
        frame["effective_from_utc"],
        utc=True,
        errors="raise",
    )
    frame["effective_to_utc"] = pd.to_datetime(
        frame["effective_to_utc"],
        utc=True,
        errors="coerce",
    )
    if (
        frame.empty
        or bool(frame["ticker"].eq("").any())
        or bool(frame["security_id"].astype(str).str.strip().eq("").any())
        or frame["universe_snapshot_id"].nunique() != 1
        or not bool(frame["effective_to_utc"].notna().any())
        or bool(
            frame["membership_source_urls"]
            .astype(str)
            .str.strip()
            .isin({"", "[]"})
            .any()
        )
    ):
        raise DataReadinessError(
            "ER1A memberships are not a sourced point-in-time universe"
        )
    _reject_membership_overlaps(frame)
    audit = load_plan_json(audit_path)
    if (
        audit.get("universe_snapshot_id")
        != frame["universe_snapshot_id"].iloc[0]
        or int(audit.get("historical_tickers", 0))
        != int(frame["ticker"].nunique())
        or int(audit.get("membership_intervals", 0)) != len(frame)
        or audit.get("contradictions") not in ([], None)
        or int(audit.get("historical_tickers", 0))
        < minimum_cross_section
    ):
        raise DataReadinessError(
            "ER1A membership audit does not match the universe"
        )
    return frame, {
        "path": str(path),
        "sha256": file_sha256(path),
        "audit_path": str(audit_path),
        "audit_sha256": file_sha256(audit_path),
        "universe_snapshot_id": str(
            frame["universe_snapshot_id"].iloc[0]
        ),
        "membership_intervals": len(frame),
        "historical_tickers": int(frame["ticker"].nunique()),
        "first_effective_at_utc": frame["effective_from_utc"].min().isoformat(),
        "research_only": True,
    }


def _reject_membership_overlaps(frame: pd.DataFrame) -> None:
    for ticker, group in frame.groupby("ticker", sort=False):
        ordered = group.sort_values("effective_from_utc")
        previous_end: pd.Timestamp | None = None
        for row in ordered.itertuples(index=False):
            start = pd.Timestamp(row.effective_from_utc)
            if previous_end is not None and start < previous_end:
                raise DataReadinessError(
                    f"ER1A membership intervals overlap for {ticker}"
                )
            raw_end = row.effective_to_utc
            previous_end = (
                pd.Timestamp.max.tz_localize("UTC")
                if pd.isna(raw_end)
                else pd.Timestamp(raw_end)
            )


def verify_existing_ohlcv_identity(
    directory: Path,
    *,
    config: IntradayHistoryConfig,
) -> dict[str, object]:
    """Verify an existing `ohlcv.v1` store carries the frozen feed identity."""

    schema_path = directory / "_schema.json"
    manifest_path = directory / "_ohlcv_manifest.csv"
    schema = load_plan_json(schema_path)
    manifest = pd.read_csv(manifest_path)
    if (
        schema.get("schema_version") != "ohlcv.v1"
        or str(schema.get("source", "")).lower() != "alpaca"
        or str(schema.get("price_feed", "")).lower()
        != config.required_price_feed
        or str(schema.get("adjustment", "")).lower()
        != config.required_adjustment
        or schema.get("timeframes") != ["5m"]
        or manifest.empty
        or set(manifest["timeframe"].astype(str).str.lower()) != {"5m"}
    ):
        raise DataReadinessError(
            f"existing OHLCV identity is incompatible: {directory}"
        )
    return {
        "path": str(directory),
        "schema_sha256": file_sha256(schema_path),
        "manifest_sha256": file_sha256(manifest_path),
        "symbols": int(manifest["ticker"].nunique()),
        "rows": int(manifest["rows"].sum()),
        "start_utc": str(schema["start_utc"]),
        "end_utc": str(schema["end_utc"]),
        "price_feed": "sip",
        "adjustment": "all",
        "timeframe": "5Min",
    }


@dataclass(frozen=True, slots=True)
class PointInTimeSession:
    """One exchange session with its verified point-in-time cross-section."""

    session: pd.Timestamp
    open_at: pd.Timestamp
    close_at: pd.Timestamp
    active: pd.DataFrame

    @property
    def month(self) -> str:
        return str(self.session.strftime("%Y-%m"))

    @property
    def tickers(self) -> list[str]:
        return [str(value) for value in self.active["ticker"]]

    def membership_records(self) -> list[dict[str, object]]:
        return [
            {
                "session_date_et": self.session.date(),
                "session_open_utc": self.open_at,
                "session_close_utc": self.close_at,
                "ticker": row["ticker"],
                "security_id": row["security_id"],
                "sector": row["sector"],
                "primary_benchmark": row["primary_benchmark"],
                "universe_snapshot_id": row["universe_snapshot_id"],
                "effective_from_utc": row["effective_from_utc"],
                "effective_to_utc": row["effective_to_utc"],
            }
            for row in self.active.to_dict(orient="records")
        ]


def iter_point_in_time_sessions(
    *,
    memberships: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    calendar: Any,
    minimum_cross_section: int,
) -> Iterator[PointInTimeSession]:
    """Yield the one canonical point-in-time cross-section for each session."""

    for session in sessions:
        open_at = pd.Timestamp(calendar.session_open(session)).tz_convert("UTC")
        close_at = pd.Timestamp(calendar.session_close(session)).tz_convert(
            "UTC"
        )
        active = memberships[
            memberships["effective_from_utc"].le(open_at)
            & (
                memberships["effective_to_utc"].isna()
                | memberships["effective_to_utc"].gt(open_at)
            )
        ].sort_values("ticker", kind="stable")
        if len(active) < minimum_cross_section:
            raise DataReadinessError(
                f"ER1A PIT cross-section is too small on {session.date()}: "
                f"{len(active)}"
            )
        if bool(active["ticker"].duplicated().any()):
            raise DataReadinessError(
                f"ER1A PIT membership is ambiguous on {session.date()}"
            )
        yield PointInTimeSession(
            session=session,
            open_at=open_at,
            close_at=close_at,
            active=active,
        )


def expected_five_minute_bars(start: pd.Timestamp, end: pd.Timestamp) -> int:
    """Return the exact five-minute bar budget for one requested window."""

    minutes = int((end - start).total_seconds() // 60)
    if minutes < 1:
        raise DataReadinessError("requested bar window is empty")
    return (minutes + 4) // 5


def chunk_request_symbols(
    symbols: list[str],
    *,
    expected_bars_per_symbol: int,
    maximum_symbols_per_unit: int,
    maximum_expected_rows_per_unit: int,
    label: str,
) -> Iterator[tuple[list[str], dict[str, str]]]:
    """Chunk one session cross-section into provider-legal request units."""

    symbols_per_unit = min(
        maximum_symbols_per_unit,
        maximum_expected_rows_per_unit // expected_bars_per_symbol,
    )
    if symbols_per_unit < 1:
        raise DataReadinessError(f"unit row cap cannot fit {label}")
    for offset in range(0, len(symbols), symbols_per_unit):
        chunk = symbols[offset : offset + symbols_per_unit]
        mapping = {
            provider_symbol(ticker, "alpaca"): ticker for ticker in chunk
        }
        if len(mapping) != len(chunk):
            raise DataReadinessError(
                f"provider-symbol collision on {label}"
            )
        yield chunk, mapping


def request_unit_record(
    *,
    unit_id: str,
    session_date: object,
    start: pd.Timestamp,
    end: pd.Timestamp,
    chunk: list[str],
    mapping: dict[str, str],
    expected_bars_per_symbol: int,
    plan_fingerprint: str,
    timeframe: str,
) -> dict[str, object]:
    """Build one canonical SIP request-unit record."""

    if timeframe not in {"1Min", "5Min"}:
        raise ValueError("request-unit timeframe must be 1Min or 5Min")

    return {
        "unit_id": unit_id,
        "session_date_et": session_date,
        "requested_start_utc": start,
        "requested_end_utc": end,
        "canonical_symbols_json": json.dumps(chunk, separators=(",", ":")),
        "provider_symbols_json": json.dumps(
            sorted(mapping),
            separators=(",", ":"),
        ),
        "provider_to_canonical_json": json.dumps(
            mapping,
            sort_keys=True,
            separators=(",", ":"),
        ),
        "symbol_count": len(chunk),
        "expected_bars_per_symbol": expected_bars_per_symbol,
        "maximum_expected_rows": expected_bars_per_symbol * len(chunk),
        "timeframe": timeframe,
        "price_feed": "sip",
        "adjustment": "all",
        "sort": "asc",
        "limit": 10_000,
        "plan_fingerprint": plan_fingerprint,
    }


def _build_plan_frames(
    *,
    memberships: pd.DataFrame,
    sessions: pd.DatetimeIndex,
    calendar: Any,
    config: IntradayHistoryConfig,
    plan_fingerprint: str,
) -> tuple[
    dict[str, pd.DataFrame],
    dict[str, pd.DataFrame],
    dict[str, int],
]:
    session_rows: dict[str, list[dict[str, object]]] = {}
    unit_rows: dict[str, list[dict[str, object]]] = {}
    total_expected_rows = 0
    unit_count = 0
    all_tickers: set[str] = set()
    benchmark_tickers = tuple(
        ticker.strip().upper() for ticker in config.benchmark_tickers
    )
    for entry in iter_point_in_time_sessions(
        memberships=memberships,
        sessions=sessions,
        calendar=calendar,
        minimum_cross_section=config.minimum_session_cross_section,
    ):
        session, open_at, close_at = entry.session, entry.open_at, entry.close_at
        month = entry.month
        session_rows.setdefault(month, []).extend(entry.membership_records())
        all_tickers.update(entry.tickers)
        symbols = sorted(set(entry.tickers).union(benchmark_tickers))
        expected_bars_per_symbol = expected_five_minute_bars(open_at, close_at)
        for chunk, mapping in chunk_request_symbols(
            symbols,
            expected_bars_per_symbol=expected_bars_per_symbol,
            maximum_symbols_per_unit=config.maximum_symbols_per_unit,
            maximum_expected_rows_per_unit=(
                config.maximum_expected_rows_per_unit
            ),
            label=str(session.date()),
        ):
            unit_id = stable_identity_hash(
                plan_fingerprint,
                session.date().isoformat(),
                open_at.isoformat(),
                close_at.isoformat(),
                *sorted(mapping),
                "5Min",
                "sip",
                "all",
            )
            unit_rows.setdefault(month, []).append(
                request_unit_record(
                    unit_id=unit_id,
                    session_date=session.date(),
                    start=open_at,
                    end=close_at,
                    chunk=chunk,
                    mapping=mapping,
                    expected_bars_per_symbol=expected_bars_per_symbol,
                    plan_fingerprint=plan_fingerprint,
                    timeframe="5Min",
                )
            )
            total_expected_rows += expected_bars_per_symbol * len(chunk)
            unit_count += 1
    membership_frames = {
        month: pd.DataFrame(rows).sort_values(
            ["session_open_utc", "ticker"],
            kind="stable",
        )
        for month, rows in session_rows.items()
    }
    units = {
        month: pd.DataFrame(rows).sort_values(
            ["requested_start_utc", "unit_id"],
            kind="stable",
        )
        for month, rows in unit_rows.items()
    }
    return membership_frames, units, {
        "historical_tickers": len(all_tickers),
        "ticker_sessions": sum(len(frame) for frame in membership_frames.values()),
        "acquisition_units": unit_count,
        "maximum_expected_feature_rows": total_expected_rows,
        "benchmark_tickers": len(benchmark_tickers),
    }


def file_record(path: Path, root: Path, rows: int) -> dict[str, object]:
    """Describe one plan artifact the way `_manifest.json` registers it."""

    return {
        "path": path.relative_to(root).as_posix(),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": rows,
    }


def load_plan_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"ER1A JSON is unreadable: {path}") from exc
    if not isinstance(value, dict):
        raise DataReadinessError(f"ER1A JSON is not an object: {path}")
    return {str(key): item for key, item in value.items()}


def write_plan_json(path: Path, value: object) -> None:
    path.write_text(
        json.dumps(value, indent=2, sort_keys=True),
        encoding="utf-8",
    )


def json_sha256(value: object) -> str:
    return hashlib.sha256(
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
        ).encode()
    ).hexdigest()


def stable_identity_hash(*parts: str) -> str:
    """Hash ordered identity parts with an unambiguous separator."""

    return hashlib.sha256("\x1f".join(parts).encode()).hexdigest()


def _assert_memory(
    config: IntradayHistoryConfig,
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
