"""Reorganize downloaded bars into per-symbol history the model can read.

Bars arrive shaped by how they were requested: one file per network call,
holding one session for fifty symbols. Feature construction needs the opposite
shape, one file per symbol holding its whole history in time order, because a
moving average walks one symbol forward through time. Stitching that out of
thousands of request files on every read is not viable.

Thirty-four million bars cannot be held at once under the memory budget, and
regrouping request files into symbol files is a shuffle, so this runs in two
bounded passes. The first streams every source month and writes rows into a
fixed number of temporary buckets chosen by symbol. The second reads one bucket
at a time, which holds only a fraction of the corpus, and writes the final
per-symbol files.

Regular-session and extended-session bars are published into physically separate
stores. Thin pre-market volume must never reach a regular-session VWAP, moving
average, or relative-volume denominator, and separate directories make that
impossible rather than merely discouraged.

Segment is decided by the exchange session's real open and close, never by clock
time. On an early-close session the market shuts at 13:00 ET, so a fixed
09:30-16:00 rule would file three hours of genuine post-market bars as regular
session.

A bar is kept only if its symbol was eligible on that exact session. Eligibility
comes from point-in-time index membership and, when a published screen is
supplied, from the in-play stock-sessions that screen selected. The intraday
universe is deliberately not index-restricted, so membership alone would discard
every non-index name the screen exists to find.
"""

from __future__ import annotations

import hashlib
import json
import shutil
import uuid
from collections.abc import Iterator, Mapping
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Final

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.normalize import canonicalize_bars
from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.edge_rebuild.corpus_integrity import (
    IntegrityThresholds,
    verify_corpus_integrity,
)
from market_predictor.edge_rebuild.intraday_selection import (
    load_complete_intraday_selection,
)
from market_predictor.resources import (
    assert_memory_budget,
    assert_peak_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

MATERIALIZATION_SCHEMA = "edge_rebuild.intraday_materialization.v1"
MATERIALIZATION_AUTHORITY_SCHEMA = (
    "edge_rebuild.intraday_materialization_authority.v1"
)
REGULAR = "regular"
PREMARKET = "premarket"
POSTMARKET = "postmarket"
ERA_COLLECTED = "collected"
ERA_LEGACY = "legacy_ohlcv_v1"
OUTPUT_COLUMNS = (
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
_ROWS_PER_BUCKET = 1_500_000


@dataclass(frozen=True, slots=True)
class SessionBounds:
    """Real open and close for one exchange session, in UTC."""

    open_at: pd.Timestamp
    close_at: pd.Timestamp

    def segment_of(self, bar_start: pd.Timestamp) -> str:
        if bar_start < self.open_at:
            return PREMARKET
        if bar_start >= self.close_at:
            return POSTMARKET
        return REGULAR


def session_bounds_for(
    first_session: str,
    last_session: str,
    *,
    calendar_name: str = "XNYS",
) -> dict[str, SessionBounds]:
    """Real open/close per session. Early closes differ from the usual 16:00."""

    calendar = xcals.get_calendar(calendar_name)
    bounds: dict[str, SessionBounds] = {}
    for session in calendar.sessions_in_range(first_session, last_session):
        key = str(pd.Timestamp(session).date())
        bounds[key] = SessionBounds(
            open_at=pd.Timestamp(calendar.session_open(session)).tz_convert("UTC"),
            close_at=pd.Timestamp(calendar.session_close(session)).tz_convert("UTC"),
        )
    return bounds


def classify_segments(
    frame: pd.DataFrame,
    bounds: Mapping[str, SessionBounds],
) -> pd.DataFrame:
    """Attach session date and segment using real session bounds.

    Bars whose session is outside the requested window are dropped, so the
    window is applied once, here, rather than by every downstream reader.
    """

    if frame.empty:
        return frame.assign(session_date_et="", session_segment="")
    eastern = frame["bar_start_utc"].dt.tz_convert("America/New_York")
    sessions = eastern.dt.date.astype(str)
    keep = sessions.isin(bounds.keys())
    frame = frame.loc[keep].copy()
    sessions = sessions.loc[keep]
    if frame.empty:
        return frame.assign(session_date_et="", session_segment="")
    opens = sessions.map(lambda key: bounds[key].open_at)
    closes = sessions.map(lambda key: bounds[key].close_at)
    starts = frame["bar_start_utc"]
    segment = pd.Series(REGULAR, index=frame.index, dtype="object")
    segment[starts < opens] = PREMARKET
    segment[starts >= closes] = POSTMARKET
    frame["session_date_et"] = sessions
    frame["session_segment"] = segment
    return frame


def _bucket_of(ticker: str, buckets: int) -> int:
    digest = hashlib.sha256(ticker.encode()).digest()
    return int.from_bytes(digest[:4], "big") % buckets


def _iter_collection_months(collection_dir: Path) -> Iterator[tuple[str, list[Path]]]:
    bars = collection_dir / "bars"
    for month_dir in sorted(p for p in bars.iterdir() if p.is_dir()):
        parts = sorted(month_dir.glob("*.parquet"))
        if parts:
            yield month_dir.name, parts


def _read_collected(paths: list[Path]) -> pd.DataFrame:
    frame = pd.concat([pd.read_parquet(path) for path in paths], ignore_index=True)
    frame["history_era"] = ERA_COLLECTED
    return frame


def _read_legacy_symbol(path: Path, *, ingested_fallback: datetime) -> pd.DataFrame:
    """Load one `ohlcv.v1` symbol file and give it the missing time columns.

    That store predates the canonical schema and carries neither `bar_end_utc`
    nor `available_at_utc`. Both are derived through the shared canonicaliser
    under the same interval-close policy and sixty-second finalisation delay the
    collected corpus used, so one definition covers both eras. This is
    derivation, not preservation, and is recorded as such.
    """

    raw = pd.read_parquet(path)
    if raw.empty:
        return raw
    canonical = canonicalize_bars(
        raw,
        timeframe="5m",
        source="alpaca",
        price_feed="sip",
        adjustment="all",
        ingested_at_utc=ingested_fallback,
        availability_policy="market_interval_close",
        intraday_finalization_delay=pd.Timedelta(seconds=60),
    )
    canonical["history_era"] = ERA_LEGACY
    return canonical


def reorganize_intraday_history(
    *,
    collected_dirs: Mapping[str, Path],
    legacy_dirs: Mapping[str, Path],
    universe_path: Path,
    output_dir: Path,
    first_session: str,
    last_session: str,
    selected_sessions_path: Path | None = None,
    thresholds: IntegrityThresholds | None = None,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
) -> dict[str, Any]:
    """Publish per-symbol regular and extended stores for one research window."""

    if output_dir.exists():
        raise DataReadinessError(f"materialization output must be new: {output_dir}")
    bounds = session_bounds_for(first_session, last_session)
    if not bounds:
        raise DataReadinessError("materialization window contains no sessions")
    memberships, universe_manifest = load_canonical_artifact(
        universe_path,
        expected_type="memberships",
        allow_research=True,
    )
    eligible = _eligible_ticker_sessions(memberships, bounds)
    selection_identity: dict[str, Any] | None = None
    if selected_sessions_path is not None:
        selected, selection_identity = selected_ticker_sessions(
            selected_sessions_path,
            bounds,
        )
        for ticker, sessions in selected.items():
            eligible.setdefault(ticker, set()).update(sessions)
    if not eligible:
        raise DataReadinessError("no eligible ticker-sessions in the window")

    staging = output_dir.with_name(f".{output_dir.name}.{uuid.uuid4().hex}.staging")
    staging.mkdir(parents=True)
    try:
        buckets = _shuffle_into_buckets(
            collected_dirs=collected_dirs,
            legacy_dirs=legacy_dirs,
            bounds=bounds,
            eligible=eligible,
            staging=staging,
            budget=(memory_budget_gib, memory_headroom_gib),
        )
        summary = _write_symbol_stores(
            buckets=buckets,
            staging=staging,
            output_root=staging / "published",
            budget=(memory_budget_gib, memory_headroom_gib),
            expected_bars=expected_bars_per_session_segment(bounds),
        )
        report = verify_corpus_integrity(
            summary.pop("_ticker_sessions"),
            label="materialized intraday corpus",
            thresholds=thresholds,
        )
        manifest: dict[str, Any] = {
            "schema": MATERIALIZATION_SCHEMA,
            "created_at_utc": datetime.now(UTC).isoformat(),
            "window_first_session": first_session,
            "window_last_session": last_session,
            "window_sessions": len(bounds),
            "universe_path": str(universe_path),
            "universe_sha256": str(universe_manifest["artifact_sha256"]),
            "selected_stock_sessions": selection_identity,
            "sources": {
                name: {"path": str(path), "role": "collected"}
                for name, path in collected_dirs.items()
            }
            | {
                name: {"path": str(path), "role": "legacy_ohlcv_v1"}
                for name, path in legacy_dirs.items()
            },
            "availability_derivation": (
                "legacy ohlcv.v1 bars carry no bar_end_utc or available_at_utc; "
                "both are derived through canonicalize_bars under the frozen "
                "market_interval_close policy and 60-second delay"
            ),
            "segment_rule": "exchange session open/close, never clock time",
            "integrity": report.to_record(),
            "memory": memory_audit(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
            ).to_record(),
            **summary,
        }
        published = staging / "published"
        if report.defect_count:
            # A refused build must still say why, or the next run starts blind.
            rejection = output_dir.with_name(f"{output_dir.name}_rejected.json")
            _write_json(rejection, manifest)
            report.raise_if_defective(
                f"materialized intraday corpus (findings written to {rejection})"
            )
        _write_json(published / "_manifest.json", manifest)
        _write_json(
            published / "_authority.json",
            {
                "schema": MATERIALIZATION_AUTHORITY_SCHEMA,
                "state": "complete",
                "artifact": "_manifest.json",
                "artifact_sha256": file_sha256(published / "_manifest.json"),
            },
        )
        output_dir.parent.mkdir(parents=True, exist_ok=True)
        published.replace(output_dir)
        return manifest
    finally:
        shutil.rmtree(staging, ignore_errors=True)


def _eligible_ticker_sessions(
    memberships: pd.DataFrame,
    bounds: Mapping[str, SessionBounds],
) -> dict[str, set[str]]:
    """Sessions on which each security was actually an index member."""

    frame = memberships.copy()
    frame["effective_from_utc"] = pd.to_datetime(frame["effective_from_utc"], utc=True)
    frame["effective_to_utc"] = pd.to_datetime(
        frame["effective_to_utc"], utc=True, errors="coerce"
    )
    eligible: dict[str, set[str]] = {}
    for session, bound in bounds.items():
        active = frame[
            frame["effective_from_utc"].le(bound.open_at)
            & (
                frame["effective_to_utc"].isna()
                | frame["effective_to_utc"].gt(bound.open_at)
            )
        ]
        for ticker in active["ticker"].astype(str):
            eligible.setdefault(ticker, set()).add(session)
    return eligible


def selected_ticker_sessions(
    selected_sessions_path: Path,
    bounds: Mapping[str, SessionBounds],
) -> tuple[dict[str, set[str]], dict[str, Any]]:
    """Eligibility from the frozen screen, with the screen's own identity.

    Index membership answers whether a security belonged to the swing universe
    that session. It cannot answer for a security that was never in the index,
    and the intraday universe is deliberately not index-restricted. A stock-
    session that passed the frozen two-layer screen is eligible on exactly the
    session it was selected for and on no other, so bars collected for it are
    kept while any bar outside its selection is still dropped.

    The screen is read only after its published authority verifies, so an edited
    selection table cannot widen what reaches the corpus.
    """

    manifest = load_complete_intraday_selection(selected_sessions_path)
    frame = pd.read_parquet(
        selected_sessions_path / "selected_stock_sessions.parquet",
        columns=["ticker", "session_date_et"],
    )
    tickers = frame["ticker"].astype(str).str.upper().str.strip()
    sessions = frame["session_date_et"].astype(str)
    inside = sessions.isin(bounds.keys())
    eligible: dict[str, set[str]] = {}
    for ticker, session in zip(tickers[inside], sessions[inside], strict=True):
        eligible.setdefault(ticker, set()).add(session)
    return eligible, {
        "path": str(selected_sessions_path),
        "manifest_sha256": file_sha256(selected_sessions_path / "_manifest.json"),
        "request_sha256": str(manifest["request_sha256"]),
        "strategy_id": str(manifest["strategy_id"]),
        "stock_sessions": int(len(frame)),
        "stock_sessions_inside_window": int(inside.sum()),
        "symbols": int(tickers.nunique()),
    }


def _shuffle_into_buckets(
    *,
    collected_dirs: Mapping[str, Path],
    legacy_dirs: Mapping[str, Path],
    bounds: Mapping[str, SessionBounds],
    eligible: Mapping[str, set[str]],
    staging: Path,
    budget: tuple[float, float],
) -> int:
    """First pass: stream every source and split rows by symbol into buckets."""

    bucket_dir = staging / "buckets"
    bucket_dir.mkdir(parents=True, exist_ok=True)
    buckets = 24
    written = 0
    benchmarks = {"SPY", "QQQ"}

    def emit(frame: pd.DataFrame, tag: str) -> None:
        nonlocal written
        frame = classify_segments(frame, bounds)
        if frame.empty:
            return
        keep = [
            ticker in eligible and session in eligible[ticker]
            for ticker, session in zip(
                frame["ticker"], frame["session_date_et"], strict=True
            )
        ]
        benchmark_rows = frame["ticker"].isin(benchmarks)
        frame = frame.loc[pd.Series(keep, index=frame.index) | benchmark_rows]
        if frame.empty:
            return
        frame = frame.loc[:, list(OUTPUT_COLUMNS)]
        frame["_bucket"] = [_bucket_of(str(t), buckets) for t in frame["ticker"]]
        for bucket, part in frame.groupby("_bucket", sort=False):
            path = bucket_dir / f"{int(bucket):02d}" / f"{tag}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            part.drop(columns=["_bucket"]).to_parquet(path, index=False)
            written += len(part)

    for name, directory in collected_dirs.items():
        for month, paths in _iter_collection_months(directory):
            emit(_read_collected(paths), f"{name}-{month}")
            release_process_memory()
            _guard(budget, f"materialization shuffle {name} {month}")

    fallback = datetime.now(UTC)
    for name, directory in legacy_dirs.items():
        for path in sorted(directory.glob("*.parquet")):
            emit(_read_legacy_symbol(path, ingested_fallback=fallback), f"{name}-{path.stem}")
            release_process_memory()
        _guard(budget, f"materialization shuffle {name}")
    if not written:
        raise DataReadinessError("materialization produced no rows")
    return buckets


def _write_symbol_stores(
    *,
    buckets: int,
    staging: Path,
    output_root: Path,
    budget: tuple[float, float],
    expected_bars: Mapping[tuple[str, str], int],
) -> dict[str, Any]:
    """Second pass: one bucket at a time, write final per-symbol files."""

    bucket_dir = staging / "buckets"
    output_root.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, Any]] = []
    ticker_sessions: list[pd.DataFrame] = []
    totals = {REGULAR: 0, PREMARKET: 0, POSTMARKET: 0}
    for index in range(buckets):
        parts = sorted((bucket_dir / f"{index:02d}").glob("*.parquet"))
        if not parts:
            continue
        frame = pd.concat([pd.read_parquet(p) for p in parts], ignore_index=True)
        for ticker, group in frame.groupby("ticker", sort=True):
            ordered = _resolve_overlapping_sources(
                group.sort_values("bar_start_utc"), str(ticker)
            )
            for segment, store in ((REGULAR, "regular"), (None, "extended")):
                part = (
                    ordered[ordered["session_segment"] == REGULAR]
                    if segment == REGULAR
                    else ordered[ordered["session_segment"] != REGULAR]
                )
                if part.empty:
                    continue
                path = output_root / store / "5m" / f"{ticker}.parquet"
                path.parent.mkdir(parents=True, exist_ok=True)
                part.to_parquet(path, index=False)
                records.append(
                    {
                        "store": store,
                        "ticker": str(ticker),
                        "path": str(path.relative_to(output_root)).replace("\\", "/"),
                        "rows": int(len(part)),
                        "sha256": file_sha256(path),
                    }
                )
            for segment, count in ordered["session_segment"].value_counts().items():
                totals[str(segment)] = totals.get(str(segment), 0) + int(count)
            ticker_sessions.append(
                _ticker_session_rows(ordered, str(ticker), expected_bars)
            )
        release_process_memory()
        _guard(budget, f"materialization publish bucket {index}")

    combined = pd.concat(ticker_sessions, ignore_index=True)
    _assert_stores_disjoint(records)
    return {
        "files": records,
        "symbols": len({record["ticker"] for record in records}),
        "rows_by_segment": totals,
        "total_rows": int(sum(totals.values())),
        "_ticker_sessions": combined,
    }


_PRICE_COLUMNS: Final = ("open", "high", "low", "close", "volume")


def _resolve_overlapping_sources(ordered: pd.DataFrame, ticker: str) -> pd.DataFrame:
    """Collapse a bar that arrived from more than one source.

    Sources overlap by design. A symbol can be an index member with legacy
    coverage and also be selected by the in-play screen, so the same bar is
    delivered twice. Identical deliveries are the same observation counted
    twice, not a conflict, and one copy is kept.

    Deliveries that disagree on price or volume are a different matter: the same
    instant cannot have traded at two prices, so one source is wrong and the
    build refuses rather than silently choosing a winner.

    Where copies agree, the collected canonical bar is preferred over the legacy
    one because it carries observed interval and availability timestamps rather
    than derived ones.
    """

    duplicated = ordered.duplicated(["bar_start_utc"], keep=False)
    if not bool(duplicated.any()):
        return ordered

    contested = ordered.loc[duplicated]
    disagreement = (
        contested.groupby("bar_start_utc")[list(_PRICE_COLUMNS)].nunique().gt(1).any(axis=1)
    )
    if bool(disagreement.any()):
        first = disagreement.index[disagreement][0]
        raise DataReadinessError(
            f"{ticker} has conflicting bars at {first}: two sources report "
            f"different prices or volume for the same instant"
        )
    # Stable ordering puts the collected era first, so keeping the first copy
    # keeps observed timestamps over derived ones.
    ranked = ordered.assign(
        _era_rank=(ordered["history_era"] != ERA_COLLECTED).astype(int)
    ).sort_values(["bar_start_utc", "_era_rank"], kind="stable")
    return (
        ranked.drop_duplicates(["bar_start_utc"], keep="first")
        .drop(columns="_era_rank")
        .sort_values("bar_start_utc")
    )


def _ticker_session_rows(
    ordered: pd.DataFrame,
    ticker: str,
    expected_bars: Mapping[tuple[str, str], int],
) -> pd.DataFrame:
    grouped = ordered.groupby(["session_date_et", "session_segment"], sort=False)
    rows = grouped.agg(
        bars=("close", "size"),
        distinct_close=("close", "nunique"),
        zero_volume_bars=("volume", lambda values: int((values == 0).sum())),
        last_close=("close", "last"),
    ).reset_index()
    rows["ticker"] = ticker
    rows = rows.rename(
        columns={"session_date_et": "session", "session_segment": "segment"}
    )
    # An early-close session holds 42 regular bars, not 78. A fixed count would
    # report every half-day as truncated.
    rows["expected_bars"] = [
        expected_bars.get((str(session), str(segment)), 0)
        for session, segment in zip(rows["session"], rows["segment"], strict=True)
    ]
    return rows


def expected_bars_per_session_segment(
    bounds: Mapping[str, SessionBounds],
    *,
    premarket_start_et: str = "04:00",
    postmarket_end_et: str = "20:00",
) -> dict[tuple[str, str], int]:
    """Exact five-minute bar count for each session and segment."""

    expected: dict[tuple[str, str], int] = {}
    for session, bound in bounds.items():
        premarket_start = pd.Timestamp(
            f"{session} {premarket_start_et}", tz="America/New_York"
        ).tz_convert("UTC")
        postmarket_end = pd.Timestamp(
            f"{session} {postmarket_end_et}", tz="America/New_York"
        ).tz_convert("UTC")
        for segment, start, end in (
            (PREMARKET, premarket_start, bound.open_at),
            (REGULAR, bound.open_at, bound.close_at),
            (POSTMARKET, bound.close_at, postmarket_end),
        ):
            minutes = int((end - start).total_seconds() // 60)
            expected[(session, segment)] = max(0, (minutes + 4) // 5)
    return expected


def _assert_stores_disjoint(records: list[dict[str, Any]]) -> None:
    """Regular and extended stores may never describe the same symbol-bar."""

    seen: set[tuple[str, str]] = set()
    for record in records:
        key = (str(record["store"]), str(record["ticker"]))
        if key in seen:
            raise DataReadinessError(
                f"duplicate published file for {record['ticker']} in {record['store']}"
            )
        seen.add(key)


def _guard(budget: tuple[float, float], stage: str) -> None:
    hard, headroom = budget
    assert_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)
    assert_peak_memory_budget(hard_budget_gib=hard, headroom_gib=headroom, stage=stage)


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(dict(payload), indent=2, sort_keys=True, default=str),
        encoding="utf-8",
    )
