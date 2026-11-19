"""Build dense session x security matrices from retained adjusted daily bars.

The loader never reads rows dated after the development end. Panel partitions for
later years are not opened, and every parquet read applies a row filter on the
session/bar date before rows are handed to Python. The development end may not
pass the sealed final-test boundary unless the caller explicitly unseals it.
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

PANEL_DIRECTORY: Final = Path("data/features/edge_rebuild_swing_panel_20180529_20260708_v2")
BAR_DIRECTORY: Final = "combined_daily"
BAR_MANIFEST: Final = "combined_daily/_manifest.json"
PANEL_MANIFEST: Final = "final/_manifest.json"
PROFILE_DIRECTORY: Final = "final/panel/feature_profile=technical_market"
DEFAULT_DEVELOPMENT_END: Final = date(2024, 5, 28)
SEALED_PERIOD_START: Final = date(2024, 5, 29)
MARKET_TIMEZONE: Final = "America/New_York"
BENCHMARK_TICKERS: Final = ("SPY", "QQQ")
SECTOR_FUND_TICKERS: Final = ("XLK", "XLF", "XLE", "XLV", "XLY", "XLP", "XLI", "XLB", "XLU", "XLRE", "XLC")
# Technical-feature eligibility of a point-in-time S&P 500 member row, before the
# news-source completeness restriction. Built only from data at or before the session.
ELIGIBILITY_COLUMN: Final = "pre_catalyst_feature_eligible"
PANEL_COLUMNS: Final = (
    "ticker", "security_id", "sector", "session_date_et", "decision_time_utc",
    "membership_available_at_utc", "cross_section_eligible", ELIGIBILITY_COLUMN,
)
BAR_COLUMNS: Final = (
    "ticker", "timeframe", "bar_start_utc", "available_at_utc", "open", "high", "low", "close", "volume",
    "source", "price_feed", "adjustment",
)
PRICE_FIELDS: Final = ("open", "high", "low", "close", "volume")

FloatMatrix = npt.NDArray[np.float32]
SessionArray = npt.NDArray[np.datetime64]


class SealedPeriodError(ValueError):
    """Raised when a development end would read the sealed final-test period."""


def check_development_end(development_end: date, *, unseal_final_test_period: bool = False) -> None:
    if development_end >= SEALED_PERIOD_START and not unseal_final_test_period:
        raise SealedPeriodError(
            f"development end {development_end.isoformat()} reads the sealed period starting "
            f"{SEALED_PERIOD_START.isoformat()}; pass the explicit unseal-final-test-period override only "
            "after one candidate is frozen")


@dataclass(frozen=True)
class ResearchPanel:
    """Adjusted daily prices with rows = XNYS sessions (SPY bars) and columns = security_id."""

    sessions: SessionArray
    security_ids: tuple[str, ...]
    open: FloatMatrix
    high: FloatMatrix
    low: FloatMatrix
    close: FloatMatrix
    volume: FloatMatrix
    eligible: npt.NDArray[np.bool_]
    sector: npt.NDArray[np.int16]
    sector_names: tuple[str, ...]
    spy_open: FloatMatrix
    spy_close: FloatMatrix
    qqq_open: FloatMatrix
    qqq_close: FloatMatrix
    development_end: date
    load_report: dict[str, Any] = field(default_factory=dict)
    input_sha256: dict[str, str] = field(default_factory=dict)

    def __post_init__(self) -> None:
        rows, columns = len(self.sessions), len(self.security_ids)
        for name in ("open", "high", "low", "close", "volume", "eligible", "sector"):
            if getattr(self, name).shape != (rows, columns):
                raise ValueError(f"research panel {name} shape differs from sessions x securities")
        for name in ("spy_open", "spy_close", "qqq_open", "qqq_close"):
            if getattr(self, name).shape != (rows,):
                raise ValueError(f"research panel {name} length differs from sessions")
        if len(set(self.security_ids)) != columns:
            raise ValueError("research panel security columns are not unique")
        if rows and (np.any(np.diff(self.sessions) <= np.timedelta64(0, "D"))
                     or self.sessions[-1] > np.datetime64(self.development_end, "D")):
            raise ValueError("research panel sessions are unordered or pass the development end")

    def truncated(self, last_session: date) -> ResearchPanel:
        """Return the panel as it would be loaded with ``last_session`` as development end."""
        count = int(np.searchsorted(self.sessions, np.datetime64(last_session, "D"), side="right"))
        return ResearchPanel(
            sessions=self.sessions[:count].copy(), security_ids=self.security_ids,
            open=self.open[:count].copy(), high=self.high[:count].copy(), low=self.low[:count].copy(),
            close=self.close[:count].copy(), volume=self.volume[:count].copy(),
            eligible=self.eligible[:count].copy(), sector=self.sector[:count].copy(), sector_names=self.sector_names,
            spy_open=self.spy_open[:count].copy(), spy_close=self.spy_close[:count].copy(),
            qqq_open=self.qqq_open[:count].copy(), qqq_close=self.qqq_close[:count].copy(),
            development_end=last_session)


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _session_days(values: pd.Series) -> SessionArray:
    days: SessionArray = np.asarray(pd.to_datetime(values).to_numpy(), dtype="datetime64[D]")
    return days


def _read_membership_rows(profile_directory: Path, development_end: date) -> tuple[pd.DataFrame, list[int]]:
    frames: list[pd.DataFrame] = []
    opened: list[int] = []
    for part in sorted(profile_directory.glob("year=*/part.parquet")):
        year = int(part.parent.name.split("=", 1)[1])
        if year > development_end.year:
            continue  # later partitions are never opened
        table = cast(Any, pq).read_table(part, columns=list(PANEL_COLUMNS), partitioning=None,
                              filters=[("session_date_et", "<=", development_end)])
        frame = table.to_pandas()
        opened.append(year)
        frame["session"] = _session_days(frame["session_date_et"])
        if len(frame) and frame["session"].max() > np.datetime64(development_end, "D"):
            raise SealedPeriodError(f"panel rows after {development_end} reached memory from {part}")
        frames.append(frame.drop(columns=["session_date_et"]))
    if not frames:
        raise ValueError(f"no panel partitions at or before {development_end.year} under {profile_directory}")
    return pd.concat(frames, ignore_index=True), opened


def _read_bars(path: Path, ticker: str, development_end: date) -> pd.DataFrame:
    cutoff = pd.Timestamp(development_end + timedelta(days=1), tz=MARKET_TIMEZONE).tz_convert("UTC")
    table = cast(Any, pq).read_table(path, columns=list(BAR_COLUMNS), partitioning=None,
                          filters=[("bar_start_utc", "<", cutoff.to_pydatetime())])
    frame = table.to_pandas()
    frame["session"] = frame["bar_start_utc"].dt.tz_convert(MARKET_TIMEZONE).dt.tz_localize(None).to_numpy().astype(
        "datetime64[D]")
    late = frame["session"] > np.datetime64(development_end, "D")
    if bool(late.any()):
        raise SealedPeriodError(f"bar rows after {development_end} reached memory from {path}")
    contract = (frame["ticker"].eq(ticker) & frame["timeframe"].eq("1d") & frame["source"].eq("alpaca")
                & frame["price_feed"].eq("sip") & frame["adjustment"].eq("all"))
    if not bool(contract.all()):
        raise ValueError(f"{path} has {int((~contract).sum())} bar rows that are not alpaca SIP all-adjusted 1d "
                         f"bars for {ticker}")
    return frame.drop(columns=["timeframe", "source", "price_feed", "adjustment", "bar_start_utc"])


def _benchmark_series(bars: pd.DataFrame, sessions: SessionArray, ticker: str,
                      report: dict[str, Any]) -> tuple[FloatMatrix, FloatMatrix]:
    indexed = bars.set_index("session")
    if indexed.index.has_duplicates:
        raise ValueError(f"{ticker} has duplicate daily bars")
    aligned = indexed.reindex(pd.Index(sessions, name="session"))
    report[f"{ticker.lower()}_sessions_without_bar"] = int(aligned["close"].isna().sum())
    return (aligned["open"].to_numpy(dtype=np.float32), aligned["close"].to_numpy(dtype=np.float32))


def load_research_panel(
    panel_root: Path, *, development_end: date, unseal_final_test_period: bool = False,
) -> ResearchPanel:
    """Load the retained swing panel through ``development_end`` (inclusive) into dense matrices."""
    check_development_end(development_end, unseal_final_test_period=unseal_final_test_period)
    bar_manifest_path, panel_manifest_path = panel_root / BAR_MANIFEST, panel_root / PANEL_MANIFEST
    input_sha256 = {"combined_daily_manifest_sha256": file_sha256(bar_manifest_path),
                    "final_manifest_sha256": file_sha256(panel_manifest_path)}
    artifacts = json.loads(bar_manifest_path.read_text(encoding="utf-8"))["artifacts"]
    report: dict[str, Any] = {"development_end": development_end.isoformat(), "eligibility_column": ELIGIBILITY_COLUMN}

    rows, opened_years = _read_membership_rows(panel_root / PROFILE_DIRECTORY, development_end)
    report["panel_years_opened"] = opened_years
    report["membership_rows_read"] = len(rows)
    report["cross_section_eligible_true_rows"] = int(rows["cross_section_eligible"].fillna(False).astype(bool).sum())

    stock_frames: list[pd.DataFrame] = []
    benchmark_frames: dict[str, pd.DataFrame] = {}
    skipped: list[str] = []
    tickers = [str(item["ticker"]) for item in artifacts]
    if len(set(tickers)) != len(tickers):
        raise ValueError("combined daily manifest lists a ticker more than once")
    for item in artifacts:
        ticker = str(item["ticker"])
        if ticker in SECTOR_FUND_TICKERS:
            skipped.append(ticker)
            continue
        frame = _read_bars(panel_root / BAR_DIRECTORY / str(item["path"]), ticker, development_end)
        if ticker in BENCHMARK_TICKERS:
            benchmark_frames[ticker] = frame
        else:
            stock_frames.append(frame)
    missing_benchmarks = sorted(set(BENCHMARK_TICKERS).difference(benchmark_frames))
    if missing_benchmarks:
        raise ValueError(f"benchmark bars are missing: {missing_benchmarks}")
    report["sector_fund_files_not_read"] = sorted(skipped)
    report["stock_bar_files_read"] = len(stock_frames)

    sessions = np.unique(benchmark_frames["SPY"]["session"].to_numpy().astype("datetime64[D]"))
    if len(sessions) != len(benchmark_frames["SPY"]):
        raise ValueError("SPY has duplicate daily bars")
    spy_open, spy_close = _benchmark_series(benchmark_frames["SPY"], sessions, "SPY", report)
    qqq_open, qqq_close = _benchmark_series(benchmark_frames["QQQ"], sessions, "QQQ", report)
    report["sessions"] = {"count": len(sessions), "first": str(sessions[0]), "last": str(sessions[-1])}

    # Decision cutoff per session; bars must be available at or before it.
    decision_times = rows.groupby("session")["decision_time_utc"].agg(["min", "max"])
    report["sessions_with_inconsistent_decision_time"] = int((decision_times["min"] != decision_times["max"]).sum())

    # Membership rows outside the SPY session calendar cannot be placed.
    in_calendar = rows["session"].isin(sessions)
    report["membership_rows_outside_session_calendar"] = int((~in_calendar).sum())
    rows = rows.loc[in_calendar]
    report["sessions_without_membership_rows"] = int(len(sessions) - rows["session"].nunique())

    key_securities = rows.groupby(["ticker", "session"])["security_id"].transform("nunique")
    ambiguous_ticker = key_securities.gt(1)
    duplicate_security = rows.duplicated(["security_id", "session"], keep=False)
    report["membership_rows_with_ticker_shared_by_two_securities"] = int(ambiguous_ticker.sum())
    report["membership_rows_with_duplicate_security_session"] = int(duplicate_security.sum())
    ticker_counts = rows.groupby("security_id")["ticker"].nunique()
    report["securities_with_ticker_change"] = int(ticker_counts.gt(1).sum())
    usable = rows.loc[~ambiguous_ticker & ~duplicate_security]

    security_ids = tuple(sorted(rows["security_id"].astype(str).unique()))
    column_index = pd.Index(security_ids)
    shape = (len(sessions), len(security_ids))

    bars = pd.concat(stock_frames, ignore_index=True)
    report["stock_bar_rows_read"] = len(bars)
    mapped = bars.merge(usable[["ticker", "session", "security_id"]], on=["ticker", "session"], how="left")
    unmapped = mapped["security_id"].isna()
    report["stock_bar_rows_without_membership_row"] = int(unmapped.sum())
    report["stock_bar_rows_without_membership_row_by_ticker"] = {
        str(ticker): {"rows": len(group), "first": str(group["session"].min()), "last": str(group["session"].max())}
        for ticker, group in mapped.loc[unmapped].groupby("ticker")}
    mapped = mapped.loc[~unmapped]
    outside = ~mapped["session"].isin(sessions)
    report["stock_bar_rows_outside_session_calendar"] = int(outside.sum())
    mapped = mapped.loc[~outside]
    duplicate_bar = mapped.duplicated(["security_id", "session"], keep=False)
    report["stock_bar_rows_dropped_duplicate_security_session"] = int(duplicate_bar.sum())
    mapped = mapped.loc[~duplicate_bar]
    cutoff = mapped["session"].map(decision_times["min"])
    late = cutoff.isna() | (mapped["available_at_utc"] > cutoff)
    report["stock_bar_rows_dropped_available_after_decision_cutoff"] = int(late.sum())
    mapped = mapped.loc[~late]

    row_position = np.searchsorted(sessions, mapped["session"].to_numpy().astype("datetime64[D]"))
    column_position = column_index.get_indexer(mapped["security_id"].astype(str))
    matrices: dict[str, FloatMatrix] = {}
    for name in PRICE_FIELDS:
        matrix = np.full(shape, np.nan, dtype=np.float32)
        matrix[row_position, column_position] = mapped[name].to_numpy(dtype=np.float32)
        matrices[name] = matrix
    report["stock_bar_cells_placed"] = len(mapped)

    member_row = np.searchsorted(sessions, rows["session"].to_numpy().astype("datetime64[D]"))
    member_column = column_index.get_indexer(rows["security_id"].astype(str))
    flagged = rows[ELIGIBILITY_COLUMN].fillna(False).astype(bool).to_numpy()
    member_known = (rows["membership_available_at_utc"] <= rows["decision_time_utc"]).fillna(False).to_numpy()
    clean_identity = (~ambiguous_ticker & ~duplicate_security).to_numpy()
    report["membership_rows_not_technically_eligible"] = int((~flagged).sum())
    report["membership_rows_membership_known_after_decision"] = int((~member_known).sum())
    report["membership_rows_excluded_for_identity_conflict"] = int((~clean_identity).sum())
    eligible = np.zeros(shape, dtype=bool)
    keep = flagged & member_known & clean_identity
    eligible[member_row[keep], member_column[keep]] = True
    report["eligible_cells"] = int(eligible.sum())
    report["eligible_cells_without_close"] = int((eligible & ~np.isfinite(matrices["close"])).sum())

    sector_values = rows["sector"].astype("string")
    sector_names = tuple(sorted(str(name) for name in sector_values.dropna().unique()))
    report["membership_rows_without_sector"] = int(sector_values.isna().sum())
    report["sector_names"] = list(sector_names)
    sector = np.full(shape, -1, dtype=np.int16)
    codes = pd.Index(sector_names).get_indexer(sector_values.fillna("")).astype(np.int16)
    sector[member_row, member_column] = codes
    return ResearchPanel(
        sessions=sessions, security_ids=security_ids, open=matrices["open"], high=matrices["high"],
        low=matrices["low"], close=matrices["close"], volume=matrices["volume"], eligible=eligible, sector=sector,
        sector_names=sector_names, spy_open=spy_open, spy_close=spy_close, qqq_open=qqq_open, qqq_close=qqq_close,
        development_end=development_end, load_report=report, input_sha256=input_sha256)
