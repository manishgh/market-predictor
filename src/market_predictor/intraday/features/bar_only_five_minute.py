"""Immutable selected-session projection of canonical SIP five-minute bars."""
from __future__ import annotations

import json
import shutil
import uuid
from collections.abc import Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import exchange_calendars as xcals
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.contracts.lineage import (
    DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
    IntradayContractIdentity,
    require_intraday_contract_lineage,
)
from market_predictor.intraday.datasets.history import json_sha256
from market_predictor.intraday.datasets.one_minute_coverage import (
    verify_canonical_five_minute_store,
)
from market_predictor.intraday.datasets.selected_session_history import (
    verify_selected_stock_sessions,
)
from market_predictor.intraday.datasets.selection import (
    load_complete_intraday_selection,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)

PROJECTION_SCHEMA: Final = "edge_rebuild.intraday_bar_only_five_minute_projection.v1"
PROJECTION_AUTHORITY_SCHEMA: Final = (
    "edge_rebuild.intraday_bar_only_five_minute_projection_authority.v1"
)
_MEMORY_BUDGET_GIB: Final = 4.0
_MEMORY_HEADROOM_GIB: Final = 0.75
_CALENDAR: Final = "XNYS"
_BAR_COLUMNS: Final = (
    "ticker",
    "session_date_et",
    "session_segment",
    "history_era",
    "timeframe",
    "bar_start_utc",
    "bar_end_utc",
    "available_at_utc",
    "ingested_at_utc",
    "open",
    "high",
    "low",
    "close",
    "volume",
    "source",
    "price_feed",
    "adjustment",
)
_COVERAGE_COLUMNS: Final = (
    "ticker",
    "session_date_et",
    "activation_time_utc",
    "expected_rows",
    "observed_rows",
    "missing_rows",
    "coverage_status",
    "source_path",
    "source_sha256",
    "source_rows",
)
_METADATA_FILES: Final = frozenset(
    {"_request.json", "_manifest.json", "_authority.json"}
)


class _MonthlyParquetWriters:
    """Keep one deterministic writer per month without retaining all bars."""

    def __init__(self, root: Path, *, role: str) -> None:
        self._root = root
        self._role = role
        self._writers: dict[str, pq.ParquetWriter] = {}
        self._rows: dict[str, int] = {}

    def write(self, month: str, frame: pd.DataFrame) -> None:
        if frame.empty:
            return
        table = pa.Table.from_pandas(frame, preserve_index=False)
        writer = self._writers.get(month)
        if writer is None:
            path = self._path(month)
            path.parent.mkdir(parents=True, exist_ok=True)
            writer = pq.ParquetWriter(  # type: ignore[no-untyped-call]
                path, table.schema, compression="zstd"
            )
            self._writers[month] = writer
        elif writer.schema != table.schema:
            raise DataReadinessError(
                f"monthly {self._role} schema changed while writing {month}"
            )
        writer.write_table(table)  # type: ignore[no-untyped-call]
        self._rows[month] = self._rows.get(month, 0) + len(frame)

    def close(self) -> None:
        for writer in self._writers.values():
            writer.close()  # type: ignore[no-untyped-call]
        self._writers.clear()

    def records(self, root: Path, columns: Sequence[str]) -> list[dict[str, Any]]:
        return [
            _file_record(
                self._path(month),
                root,
                rows=self._rows[month],
                role=self._role,
                month=month,
                columns=columns,
            )
            for month in sorted(self._rows)
        ]

    def _path(self, month: str) -> Path:
        year, month_number = month.split("-", maxsplit=1)
        return self._root / self._role / f"year={year}" / f"month={month_number}.parquet"


def publish_selected_session_five_minute_projection(
    *,
    selection_directory: Path,
    five_minute_canonical_directory: Path,
    strategy_contract_path: Path,
    output_directory: Path,
    intraday_contract_lineage_path: Path = DEFAULT_INTRADAY_CONTRACT_LINEAGE_PATH,
) -> dict[str, Any]:
    """Atomically project selected stock-sessions without provider access."""

    _require_path_isolation(
        output_directory,
        (
            selection_directory,
            five_minute_canonical_directory,
            strategy_contract_path,
            intraday_contract_lineage_path,
        ),
    )
    if output_directory.exists():
        raise DataReadinessError(f"five-minute projection is immutable: {output_directory}")
    _guard_memory("five-minute projection start")
    selection, selection_identity = verify_selected_stock_sessions(selection_directory)
    selection_manifest = load_complete_intraday_selection(selection_directory)
    source_records, canonical_identity = verify_canonical_five_minute_store(
        five_minute_canonical_directory
    )
    contract = load_strategy_contract(strategy_contract_path)
    _validate_contract(contract)
    contract_identity = _validate_selection_lineage(
        selection_manifest,
        selection_identity=selection_identity,
        canonical_directory=five_minute_canonical_directory,
        contract=contract,
        contract_path=strategy_contract_path,
        lineage_path=intraday_contract_lineage_path,
    )
    regular_records = _regular_records(source_records)
    request_payload = {
        "schema": PROJECTION_SCHEMA,
        "selection_directory": str(selection_directory.resolve()),
        "selection_authority_sha256": file_sha256(
            selection_directory / "_authority.json"
        ),
        "selection_manifest_sha256": str(selection_identity["manifest_sha256"]),
        "selection_table_sha256": str(selection_identity["table_sha256"]),
        "five_minute_canonical_directory": str(
            five_minute_canonical_directory.resolve()
        ),
        **canonical_identity,
        "strategy_contract_path": str(strategy_contract_path.resolve()),
        "strategy_contract_file_sha256": file_sha256(strategy_contract_path),
        "strategy_contract_sha256": contract.sha256(),
        "intraday_data_contract_sha256": (
            contract_identity.intraday_data_contract_sha256
        ),
        "intraday_parent_contract_sha256": (
            contract_identity.observed_contract_sha256
        ),
        "intraday_contract_lineage_mode": contract_identity.mode,
        "intraday_contract_lineage_path": str(
            intraday_contract_lineage_path.resolve()
        ),
        "intraday_contract_lineage_file_sha256": (
            contract_identity.lineage_file_sha256
        ),
        "calendar": _CALENDAR,
        "timeframe": "5Min",
        "source": "alpaca",
        "price_feed": "sip",
        "adjustment": "all",
        "provider_download_performed": False,
        "coverage_policy": "retain_all_selected_stock_sessions",
        "memory_processing_unit": "one_ticker_source_file",
        "maximum_process_memory_gib": _MEMORY_BUDGET_GIB,
    }
    request = {
        **request_payload,
        "request_sha256": json_sha256(request_payload),
    }
    staging = output_directory.with_name(
        f".{output_directory.name}.{uuid.uuid4().hex}.staging"
    )
    if staging.exists():
        raise DataReadinessError(f"projection staging path already exists: {staging}")
    staging.mkdir(parents=True)
    bars_writers = _MonthlyParquetWriters(staging, role="bars")
    coverage_writers = _MonthlyParquetWriters(staging, role="coverage")
    try:
        _write_json(staging / "_request.json", request)
        coverage_rows: list[dict[str, Any]] = []
        calendar = xcals.get_calendar(_CALENDAR)
        selected_by_ticker = {
            str(ticker): group.sort_values("session_date_et", kind="stable")
            for ticker, group in selection.groupby("ticker", sort=True)
        }
        for ticker, selected in selected_by_ticker.items():
            _guard_memory(f"five-minute projection before {ticker}")
            record = regular_records.get(ticker)
            source = _read_source_ticker(
                five_minute_canonical_directory,
                record=record,
                ticker=ticker,
            )
            projected, ticker_coverage = _project_ticker(
                ticker=ticker,
                selected=selected,
                source=source,
                record=record,
                calendar=calendar,
            )
            for month, frame in projected.groupby("_month", sort=True):
                bars_writers.write(
                    str(month),
                    frame.drop(columns="_month").loc[:, list(_BAR_COLUMNS)],
                )
            coverage_rows.extend(ticker_coverage)
            del source, projected, ticker_coverage
            release_process_memory()
            _guard_memory(f"five-minute projection after {ticker}")
        bars_writers.close()
        coverage = pd.DataFrame(coverage_rows, columns=list(_COVERAGE_COLUMNS))
        coverage = _normalize_coverage_frame(coverage)
        coverage = coverage.sort_values(
            ["session_date_et", "ticker"], kind="stable"
        ).reset_index(drop=True)
        coverage["_month"] = coverage["session_date_et"].str.slice(0, 7)
        for month, frame in coverage.groupby("_month", sort=True):
            coverage_writers.write(
                str(month),
                frame.drop(columns="_month").loc[:, list(_COVERAGE_COLUMNS)],
            )
        coverage_writers.close()
        files = [
            _file_record(
                staging / "_request.json",
                staging,
                rows=1,
                role="request",
                month=None,
                columns=(),
            ),
            *bars_writers.records(staging, _BAR_COLUMNS),
            *coverage_writers.records(staging, _COVERAGE_COLUMNS),
        ]
        status_counts = {
            str(key): int(value)
            for key, value in coverage["coverage_status"].value_counts().items()
        }
        manifest = {
            **request,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "state": "complete",
            "selected_stock_sessions": len(coverage),
            "selected_symbols": int(coverage["ticker"].nunique()),
            "selected_sessions": int(coverage["session_date_et"].nunique()),
            "first_session_et": str(coverage["session_date_et"].min()),
            "last_session_et": str(coverage["session_date_et"].max()),
            "expected_rows": int(coverage["expected_rows"].sum()),
            "projected_rows": int(coverage["observed_rows"].sum()),
            "coverage_status_counts": status_counts,
            "source_regular_files": len(regular_records),
            "source_regular_rows": sum(
                int(record["rows"]) for record in regular_records.values()
            ),
            "files": files,
            "file_inventory_sha256": _inventory_sha256(files),
            "memory": memory_audit(
                hard_budget_gib=_MEMORY_BUDGET_GIB,
                headroom_gib=_MEMORY_HEADROOM_GIB,
            ).to_record(),
        }
        _write_json(staging / "_manifest.json", manifest)
        _write_json(
            staging / "_authority.json",
            {
                "schema": PROJECTION_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(staging / "_manifest.json"),
                "request_sha256": request["request_sha256"],
                "file_inventory_sha256": manifest["file_inventory_sha256"],
            },
        )
        load_complete_selected_session_five_minute_projection(staging)
        _guard_peak_memory("five-minute projection publication")
        output_directory.parent.mkdir(parents=True, exist_ok=True)
        staging.replace(output_directory)
        return manifest
    finally:
        bars_writers.close()
        coverage_writers.close()
        shutil.rmtree(staging, ignore_errors=True)


def load_complete_selected_session_five_minute_projection(
    directory: Path,
) -> dict[str, Any]:
    """Strictly replay source lineage, inventory, coverage, and projected rows."""

    _guard_memory("five-minute projection replay start")
    request = _read_json(directory / "_request.json")
    manifest = _read_json(directory / "_manifest.json")
    authority = _read_json(directory / "_authority.json")
    request_payload = {key: value for key, value in request.items() if key != "request_sha256"}
    request_sha256 = json_sha256(request_payload)
    selection_directory = Path(str(request.get("selection_directory", "")))
    canonical_directory = Path(
        str(request.get("five_minute_canonical_directory", ""))
    )
    contract_path = Path(str(request.get("strategy_contract_path", "")))
    lineage_path = Path(
        str(request.get("intraday_contract_lineage_path", ""))
    )
    _require_path_isolation(
        directory,
        (selection_directory, canonical_directory, contract_path, lineage_path),
    )
    selection, selection_identity = verify_selected_stock_sessions(selection_directory)
    selection_manifest = load_complete_intraday_selection(selection_directory)
    source_records, canonical_identity = verify_canonical_five_minute_store(
        canonical_directory
    )
    contract = load_strategy_contract(contract_path)
    _validate_contract(contract)
    contract_identity = _validate_selection_lineage(
        selection_manifest,
        selection_identity=selection_identity,
        canonical_directory=canonical_directory,
        contract=contract,
        contract_path=contract_path,
        lineage_path=lineage_path,
    )
    regular_records = _regular_records(source_records)
    if (
        request.get("schema") != PROJECTION_SCHEMA
        or request.get("request_sha256") != request_sha256
        or manifest.get("schema") != PROJECTION_SCHEMA
        or manifest.get("state") != "complete"
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != PROJECTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256")
        != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
        or request.get("selection_authority_sha256")
        != file_sha256(selection_directory / "_authority.json")
        or request.get("selection_manifest_sha256")
        != selection_identity["manifest_sha256"]
        or request.get("selection_table_sha256") != selection_identity["table_sha256"]
        or request.get("intraday_data_contract_sha256")
        != contract_identity.intraday_data_contract_sha256
        or request.get("intraday_parent_contract_sha256")
        != contract_identity.observed_contract_sha256
        or request.get("intraday_contract_lineage_mode") != contract_identity.mode
        or request.get("intraday_contract_lineage_file_sha256")
        != contract_identity.lineage_file_sha256
        or any(request.get(key) != value for key, value in canonical_identity.items())
        or request.get("strategy_contract_file_sha256") != file_sha256(contract_path)
        or request.get("strategy_contract_sha256") != contract.sha256()
        or request.get("provider_download_performed") is not False
        or request.get("price_feed") != "sip"
        or request.get("adjustment") != "all"
        or request.get("timeframe") != "5Min"
    ):
        raise DataReadinessError("five-minute projection authority or source lineage is invalid")
    files = _validated_inventory(directory, manifest, authority)
    bars_by_month = _role_records(files, "bars")
    coverage_by_month = _role_records(files, "coverage")
    expected_months = sorted(selection["session_date_et"].str.slice(0, 7).unique())
    if sorted(coverage_by_month) != expected_months:
        raise DataReadinessError("five-minute coverage monthly inventory is incomplete")
    _strict_source_replay(
        directory=directory,
        canonical_directory=canonical_directory,
        selection=selection,
        regular_records=regular_records,
        bars_by_month=bars_by_month,
        coverage_by_month=coverage_by_month,
    )
    coverage_rows = sum(int(record["rows"]) for record in coverage_by_month.values())
    projected_rows = sum(int(record["rows"]) for record in bars_by_month.values())
    if (
        coverage_rows != len(selection)
        or int(manifest.get("selected_stock_sessions", -1)) != coverage_rows
        or int(manifest.get("projected_rows", -1)) != projected_rows
        or int(manifest.get("selected_symbols", -1)) != selection["ticker"].nunique()
        or int(manifest.get("selected_sessions", -1))
        != selection["session_date_et"].nunique()
        or int(manifest.get("source_regular_files", -1)) != len(regular_records)
        or int(manifest.get("source_regular_rows", -1))
        != sum(int(record["rows"]) for record in regular_records.values())
    ):
        raise DataReadinessError("five-minute projection manifest summary differs")
    _guard_peak_memory("five-minute projection replay")
    return manifest


def _strict_source_replay(
    *,
    directory: Path,
    canonical_directory: Path,
    selection: pd.DataFrame,
    regular_records: Mapping[str, Mapping[str, Any]],
    bars_by_month: Mapping[str, Mapping[str, Any]],
    coverage_by_month: Mapping[str, Mapping[str, Any]],
) -> None:
    calendar = xcals.get_calendar(_CALENDAR)
    observed_status_counts: dict[str, int] = {}
    for ticker, selected in selection.groupby("ticker", sort=True):
        ticker_name = str(ticker)
        _guard_memory(f"five-minute projection replay before {ticker_name}")
        record = regular_records.get(ticker_name)
        source = _read_source_ticker(
            canonical_directory,
            record=record,
            ticker=ticker_name,
        )
        expected_bars, expected_coverage = _project_ticker(
            ticker=ticker_name,
            selected=selected.sort_values("session_date_et", kind="stable"),
            source=source,
            record=record,
            calendar=calendar,
        )
        months = sorted(selected["session_date_et"].str.slice(0, 7).unique())
        actual_bar_parts = [
            _read_filtered_partition(
                directory,
                bars_by_month[month],
                columns=_BAR_COLUMNS,
                ticker=ticker_name,
            )
            for month in months
            if month in bars_by_month
        ]
        actual_bars = (
            pd.concat(actual_bar_parts, ignore_index=True)
            if actual_bar_parts
            else pd.DataFrame(columns=list(_BAR_COLUMNS))
        )
        expected_bars = expected_bars.drop(columns="_month").loc[:, list(_BAR_COLUMNS)]
        _assert_frames_equal(expected_bars, actual_bars, label=f"bars for {ticker_name}")
        actual_coverage = pd.concat(
            [
                _read_filtered_partition(
                    directory,
                    coverage_by_month[month],
                    columns=_COVERAGE_COLUMNS,
                    ticker=ticker_name,
                )
                for month in months
            ],
            ignore_index=True,
        )
        expected_coverage_frame = pd.DataFrame(
            expected_coverage, columns=list(_COVERAGE_COLUMNS)
        )
        expected_coverage_frame = _normalize_coverage_frame(
            expected_coverage_frame
        )
        actual_coverage = _normalize_coverage_frame(actual_coverage)
        _assert_frames_equal(
            expected_coverage_frame,
            actual_coverage,
            label=f"coverage for {ticker_name}",
        )
        for status, count in expected_coverage_frame["coverage_status"].value_counts().items():
            observed_status_counts[str(status)] = observed_status_counts.get(str(status), 0) + int(count)
        del source, expected_bars, actual_bars, actual_bar_parts, actual_coverage
        release_process_memory()
        _guard_memory(f"five-minute projection replay after {ticker_name}")
    manifest = _read_json(directory / "_manifest.json")
    if manifest.get("coverage_status_counts") != observed_status_counts:
        raise DataReadinessError("five-minute projection coverage status summary differs")


def _project_ticker(
    *,
    ticker: str,
    selected: pd.DataFrame,
    source: pd.DataFrame,
    record: Mapping[str, Any] | None,
    calendar: Any,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    parts: list[pd.DataFrame] = []
    coverage: list[dict[str, Any]] = []
    source_path = None if record is None else str(record["path"])
    source_sha256 = None if record is None else str(record["sha256"])
    source_rows = 0 if record is None else int(record["rows"])
    for row in selected.itertuples(index=False):
        session = str(row.session_date_et)
        session_label = pd.Timestamp(session)
        open_at = pd.Timestamp(calendar.session_open(session_label)).tz_convert("UTC")
        close_at = pd.Timestamp(calendar.session_close(session_label)).tz_convert("UTC")
        expected_starts = pd.date_range(
            open_at, close_at, freq="5min", inclusive="left"
        )
        session_rows = source.loc[source["session_date_et"].eq(session)].copy()
        _validate_session_rows(
            session_rows,
            ticker=ticker,
            session=session,
            expected_starts=expected_starts,
        )
        observed = len(session_rows)
        expected = len(expected_starts)
        status = "missing" if observed == 0 else "complete" if observed == expected else "incomplete"
        if observed:
            session_rows["_month"] = session[:7]
            parts.append(session_rows.loc[:, [*_BAR_COLUMNS, "_month"]])
        coverage.append(
            {
                "ticker": ticker,
                "session_date_et": session,
                "activation_time_utc": pd.Timestamp(row.activation_time_utc),
                "expected_rows": expected,
                "observed_rows": observed,
                "missing_rows": expected - observed,
                "coverage_status": status,
                "source_path": source_path,
                "source_sha256": source_sha256,
                "source_rows": source_rows,
            }
        )
    projected = (
        pd.concat(parts, ignore_index=True)
        if parts
        else pd.DataFrame(columns=[*_BAR_COLUMNS, "_month"])
    )
    projected = projected.sort_values(
        ["session_date_et", "bar_start_utc"], kind="stable"
    ).reset_index(drop=True)
    return projected, coverage


def _read_source_ticker(
    directory: Path,
    *,
    record: Mapping[str, Any] | None,
    ticker: str,
) -> pd.DataFrame:
    if record is None:
        return pd.DataFrame(columns=list(_BAR_COLUMNS))
    path = _resolve_inside(directory, str(record["path"]))
    try:
        frame = pd.read_parquet(path, columns=list(_BAR_COLUMNS))
    except (OSError, ValueError, KeyError) as exc:
        raise DataReadinessError(f"canonical five-minute source is unreadable for {ticker}") from exc
    if len(frame) != int(record["rows"]):
        raise DataReadinessError(f"canonical five-minute source row count moved for {ticker}")
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["session_date_et"] = pd.to_datetime(
        frame["session_date_et"], errors="raise"
    ).dt.date.astype(str)
    for column in ("bar_start_utc", "bar_end_utc", "available_at_utc", "ingested_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    if (
        frame.empty
        or not frame["ticker"].eq(ticker).all()
        or not frame["session_segment"].astype(str).str.lower().eq("regular").all()
        or not frame["timeframe"].astype(str).str.lower().eq("5m").all()
        or not frame["source"].astype(str).str.lower().eq("alpaca").all()
        or not frame["price_feed"].astype(str).str.lower().eq("sip").all()
        or not frame["adjustment"].astype(str).str.lower().eq("all").all()
        or bool(frame.duplicated(["ticker", "bar_start_utc"]).any())
    ):
        raise DataReadinessError(f"canonical SIP/all regular five-minute identity failed for {ticker}")
    return frame


def _normalize_coverage_frame(frame: pd.DataFrame) -> pd.DataFrame:
    normalized = frame.loc[:, list(_COVERAGE_COLUMNS)].copy()
    for column in (
        "ticker",
        "session_date_et",
        "coverage_status",
        "source_path",
        "source_sha256",
    ):
        normalized[column] = normalized[column].astype("string")
    normalized["activation_time_utc"] = pd.to_datetime(
        normalized["activation_time_utc"], utc=True, errors="raise"
    )
    for column in ("expected_rows", "observed_rows", "missing_rows", "source_rows"):
        normalized[column] = pd.to_numeric(
            normalized[column], errors="raise"
        ).astype("int64")
    return normalized


def _validate_session_rows(
    frame: pd.DataFrame,
    *,
    ticker: str,
    session: str,
    expected_starts: pd.DatetimeIndex,
) -> None:
    if frame.empty:
        return
    starts = pd.DatetimeIndex(frame["bar_start_utc"])
    if (
        starts.has_duplicates
        or not starts.isin(expected_starts).all()
        or not frame["bar_end_utc"].eq(frame["bar_start_utc"] + pd.Timedelta(minutes=5)).all()
        or not frame["available_at_utc"].ge(frame["bar_end_utc"]).all()
    ):
        raise DataReadinessError(
            f"canonical five-minute bars are out-of-session, non-exact, or non-causal for {ticker} {session}"
        )


def _regular_records(records: Sequence[Mapping[str, Any]]) -> dict[str, Mapping[str, Any]]:
    result: dict[str, Mapping[str, Any]] = {}
    for record in records:
        if record.get("store") != "regular":
            continue
        ticker = str(record["ticker"]).upper().strip()
        if ticker in result:
            raise DataReadinessError(f"canonical regular inventory repeats {ticker}")
        result[ticker] = record
    return result


def _validate_contract(contract: StrategyContract) -> None:
    if (
        contract.intraday.atr_timeframe != "5Min"
        or contract.intraday_universe.activity_timeframe != "5Min"
        or contract.intraday_universe.activation_delay_seconds
        != contract.intraday.decision_finalization_seconds
    ):
        raise DataReadinessError("strategy contract does not authorize causal five-minute projection")


def _validate_selection_lineage(
    manifest: Mapping[str, Any],
    *,
    selection_identity: Mapping[str, object],
    canonical_directory: Path,
    contract: StrategyContract,
    contract_path: Path,
    lineage_path: Path,
) -> IntradayContractIdentity:
    selected_canonical = Path(str(manifest.get("canonical_dir", "")))
    if (
        selection_identity.get("strategy_id") != contract.intraday.strategy_id
        or not str(manifest.get("canonical_dir", "")).strip()
        or selected_canonical.resolve() != canonical_directory.resolve()
    ):
        raise DataReadinessError(
            "intraday selection, canonical five-minute store, and strategy contract lineage differ"
        )
    return require_intraday_contract_lineage(
        observed_contract_sha256=selection_identity.get(
            "strategy_contract_sha256"
        ),
        observed_contract_file_sha256=None,
        current_contract=contract,
        current_contract_path=contract_path,
        lineage_path=lineage_path,
    )


def _validated_inventory(
    directory: Path,
    manifest: Mapping[str, Any],
    authority: Mapping[str, Any],
) -> list[dict[str, Any]]:
    raw_files = manifest.get("files")
    if not isinstance(raw_files, list) or not raw_files:
        raise DataReadinessError("five-minute projection manifest has no files")
    files: list[dict[str, Any]] = []
    paths: set[str] = set()
    for raw in raw_files:
        if not isinstance(raw, Mapping):
            raise DataReadinessError("five-minute projection file record is malformed")
        record = {str(key): value for key, value in raw.items()}
        relative = str(record.get("path", "")).replace("\\", "/")
        if relative in paths:
            raise DataReadinessError("five-minute projection inventory repeats a path")
        path = _resolve_inside(directory, relative)
        if (
            not path.is_file()
            or file_sha256(path) != record.get("sha256")
            or path.stat().st_size != int(record.get("bytes", -1))
        ):
            raise DataReadinessError(f"five-minute projection file failed inventory: {path}")
        role = str(record.get("role", ""))
        if role in {"bars", "coverage"}:
            expected_columns = _BAR_COLUMNS if role == "bars" else _COVERAGE_COLUMNS
            if tuple(record.get("columns", [])) != expected_columns:
                raise DataReadinessError("five-minute projection partition columns differ")
            parquet = pq.ParquetFile(path, memory_map=True)  # type: ignore[no-untyped-call]
            rows = 0 if parquet.metadata is None else parquet.metadata.num_rows
            if rows != int(record.get("rows", -1)) or rows < 1:
                raise DataReadinessError("five-minute projection partition row count differs")
        elif role != "request" or relative != "_request.json":
            raise DataReadinessError("five-minute projection inventory role is invalid")
        files.append(record)
        paths.add(relative)
    inventory_hash = _inventory_sha256(files)
    if (
        manifest.get("file_inventory_sha256") != inventory_hash
        or authority.get("file_inventory_sha256") != inventory_hash
    ):
        raise DataReadinessError("five-minute projection inventory hash differs")
    actual = {
        str(path.relative_to(directory)).replace("\\", "/")
        for path in directory.rglob("*")
        if path.is_file()
    }
    declared = paths | _METADATA_FILES
    if actual != declared:
        raise DataReadinessError("five-minute projection on-disk inventory is not exact")
    return files


def _role_records(
    files: Sequence[Mapping[str, Any]], role: str
) -> dict[str, Mapping[str, Any]]:
    records: dict[str, Mapping[str, Any]] = {}
    for record in files:
        if record.get("role") != role:
            continue
        month = str(record.get("month", ""))
        if len(month) != 7 or month in records:
            raise DataReadinessError(f"five-minute {role} monthly identity is invalid")
        records[month] = record
    return records


def _read_filtered_partition(
    root: Path,
    record: Mapping[str, Any],
    *,
    columns: Sequence[str],
    ticker: str,
) -> pd.DataFrame:
    path = _resolve_inside(root, str(record["path"]))
    frame = pd.read_parquet(
        path,
        columns=list(columns),
        filters=[("ticker", "==", ticker)],
    )
    for column in (
        "bar_start_utc",
        "bar_end_utc",
        "available_at_utc",
        "ingested_at_utc",
        "activation_time_utc",
    ):
        if column in frame:
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
    sort_columns = (
        ["session_date_et", "bar_start_utc"]
        if "bar_start_utc" in frame
        else ["session_date_et", "ticker"]
    )
    return frame.sort_values(sort_columns, kind="stable").reset_index(drop=True)


def _assert_frames_equal(expected: pd.DataFrame, actual: pd.DataFrame, *, label: str) -> None:
    if expected.empty and actual.empty:
        if list(expected.columns) != list(actual.columns):
            raise DataReadinessError(
                f"five-minute projection strict replay differs: {label}"
            )
        return
    try:
        pd.testing.assert_frame_equal(
            expected.reset_index(drop=True),
            actual.reset_index(drop=True),
            check_dtype=True,
            check_exact=True,
        )
    except AssertionError as exc:
        raise DataReadinessError(f"five-minute projection strict replay differs: {label}") from exc


def _file_record(
    path: Path,
    root: Path,
    *,
    rows: int,
    role: str,
    month: str | None,
    columns: Sequence[str],
) -> dict[str, Any]:
    return {
        "path": str(path.relative_to(root)).replace("\\", "/"),
        "sha256": file_sha256(path),
        "bytes": path.stat().st_size,
        "rows": int(rows),
        "role": role,
        "month": month,
        "columns": list(columns),
    }


def _inventory_sha256(files: Sequence[Mapping[str, Any]]) -> str:
    return json_sha256(sorted((dict(record) for record in files), key=lambda item: str(item["path"])))


def _resolve_inside(root: Path, relative: str) -> Path:
    if not relative or Path(relative).is_absolute():
        raise DataReadinessError("five-minute projection inventory path is invalid")
    resolved_root = root.resolve()
    path = (resolved_root / relative).resolve()
    if resolved_root not in path.parents:
        raise DataReadinessError("five-minute projection inventory escapes its root")
    return path


def _require_path_isolation(output: Path, inputs: Sequence[Path]) -> None:
    output_resolved = output.resolve()
    for source in inputs:
        source_resolved = source.resolve()
        if (
            output_resolved == source_resolved
            or output_resolved in source_resolved.parents
            or source_resolved in output_resolved.parents
        ):
            raise DataReadinessError(
                f"five-minute projection output overlaps an input path: {source}"
            )


def _read_json(path: Path) -> dict[str, Any]:
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataReadinessError(f"five-minute projection JSON is unreadable: {path}") from exc
    if not isinstance(raw, dict):
        raise DataReadinessError(f"five-minute projection JSON must be an object: {path}")
    return {str(key): value for key, value in raw.items()}


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
    try:
        temporary.parent.mkdir(parents=True, exist_ok=True)
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True),
            encoding="utf-8",
        )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _guard_memory(stage: str) -> None:
    assert_memory_budget(
        hard_budget_gib=_MEMORY_BUDGET_GIB,
        headroom_gib=_MEMORY_HEADROOM_GIB,
        stage=stage,
    )


def _guard_peak_memory(stage: str) -> None:
    assert_peak_memory_budget(
        hard_budget_gib=_MEMORY_BUDGET_GIB,
        headroom_gib=_MEMORY_HEADROOM_GIB,
        stage=stage,
    )
