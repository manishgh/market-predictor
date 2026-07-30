"""The two-layer intraday universe screen, evaluated point-in-time.

Layer one asks whether a security could be traded intraday at all: a
twenty-session average volume floor and a price band. Layer two asks whether it
is moving today: relative volume against that same average, capped per session.
Both layers come from the frozen strategy contract, never from a literal here.

The one invariant that makes this screen usable is that the average volume a
session is measured against never contains that session's own volume. A
`rolling(20).mean()` taken on the raw volume series includes the current bar, so
a single high-volume day inflates its own baseline and depresses its own
relative volume; worse, the resulting selection has read the session it claims to
be selecting for. Every baseline here is therefore taken on a series shifted one
session forward within the symbol, and `assert_no_current_session_leak` proves
it on the produced frame rather than trusting the construction.

Finviz screener exports decide which tickers were considered for acquisition and
nothing else. No value from that snapshot reaches this module: every number
below is computed from collected SIP daily bars.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
from pandas.core.groupby.generic import SeriesGroupBy

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.edge_rebuild.intraday_history import (
    json_sha256,
    load_plan_json,
    write_plan_json,
)
from market_predictor.edge_rebuild.strategy_contract import (
    IntradayUniverseContract,
    StrategyContract,
)
from market_predictor.edge_rebuild.swing_setups import drop_placeholder_bars
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.v3.errors import DataReadinessError

INTRADAY_SELECTION_SCHEMA = "edge_rebuild.intraday_universe_selection.v1"
INTRADAY_SELECTION_AUTHORITY_SCHEMA = (
    "edge_rebuild.intraday_universe_selection_authority.v1"
)
EXCHANGE_TIMEZONE = ZoneInfo("America/New_York")
SESSION_LIQUIDITY_COLUMNS = (
    "ticker",
    "session_date_et",
    "session_volume",
    "average_volume_prior_sessions",
    "relative_volume",
    "session_close",
    "baseline_sessions",
)
SELECTION_COLUMNS = (
    *SESSION_LIQUIDITY_COLUMNS,
    "session_rank",
)
_MEMORY_BUDGET_GIB = 4.0
_MEMORY_HEADROOM_GIB = 0.75


@dataclass(frozen=True, slots=True)
class IntradaySelectionResult:
    """Both screen layers plus the counts a reviewer needs to check them."""

    liquidity: pd.DataFrame
    selection: pd.DataFrame
    audit: dict[str, Any]


def session_liquidity(bars: pd.DataFrame, *, lookback_sessions: int) -> pd.DataFrame:
    """Per symbol-session volume, its prior-session baseline, and relative volume.

    The baseline is a trailing mean over the `lookback_sessions` sessions that
    ended before the measured session. Shifting inside the symbol group is what
    keeps the measured session out of its own denominator.
    """

    if lookback_sessions < 2:
        raise ValueError("lookback_sessions must cover at least two sessions")
    required = {"ticker", "bar_start_utc", "close", "volume"}
    missing = sorted(required.difference(bars.columns))
    if missing:
        raise DataReadinessError(f"daily bars are missing columns: {missing}")
    if bars.empty:
        return pd.DataFrame(columns=list(SESSION_LIQUIDITY_COLUMNS))

    frame = pd.DataFrame(
        {
            "ticker": bars["ticker"].astype(str),
            "session_date_et": pd.to_datetime(bars["bar_start_utc"], utc=True)
            .dt.tz_convert(EXCHANGE_TIMEZONE)
            .dt.date,
            "session_volume": pd.to_numeric(bars["volume"], errors="coerce"),
            "session_close": pd.to_numeric(bars["close"], errors="coerce"),
        }
    )
    duplicates = int(frame.duplicated(["ticker", "session_date_et"]).sum())
    if duplicates:
        raise DataReadinessError(
            f"daily bars contain {duplicates} duplicate symbol-session rows"
        )
    frame = frame.sort_values(
        ["ticker", "session_date_et"],
        kind="stable",
    ).reset_index(drop=True)
    prior_by_symbol = _prior_session_volume(frame)
    frame["average_volume_prior_sessions"] = prior_by_symbol.transform(
        lambda values: values.rolling(lookback_sessions, min_periods=lookback_sessions).mean()
    )
    frame["baseline_sessions"] = prior_by_symbol.transform(
        lambda values: values.rolling(lookback_sessions, min_periods=lookback_sessions).count()
    )
    baseline = frame["average_volume_prior_sessions"]
    frame["relative_volume"] = frame["session_volume"].div(baseline.where(baseline > 0))
    return frame.loc[:, list(SESSION_LIQUIDITY_COLUMNS)].reset_index(drop=True)


def assert_no_current_session_leak(
    liquidity: pd.DataFrame,
    *,
    lookback_sessions: int,
) -> None:
    """Fail if any baseline could have contained the session it measures.

    The replay deliberately shares no code with the production path: it sums an
    explicit slice `volumes[i - lookback : i]`, which excludes position `i` by
    construction, rather than reusing the shift-then-roll helper. A guard built
    on the same helper would restate the implementation and pass in lockstep
    with its own bug; this one disagrees whenever the window moves by a session.
    """

    if liquidity.empty:
        return
    frame = liquidity.sort_values(
        ["ticker", "session_date_et"],
        kind="stable",
    ).reset_index(drop=True)
    replay = pd.Series(index=frame.index, dtype="float64")
    for _, group in frame.groupby("ticker", sort=False):
        volumes = pd.to_numeric(group["session_volume"], errors="coerce").to_numpy(dtype="float64")
        cumulative = np.concatenate(([0.0], np.cumsum(volumes)))
        window_sums = cumulative[:-1][lookback_sessions:] - cumulative[:-lookback_sessions - 1]
        values = np.full(len(volumes), np.nan)
        values[lookback_sessions:] = window_sums / lookback_sessions
        replay.iloc[group.index] = values
    published = frame["average_volume_prior_sessions"]
    both_present = replay.notna() & published.notna()
    mismatched = int(
        (
            (replay[both_present] - published[both_present]).abs()
            > 1e-6 * replay[both_present].abs().clip(lower=1.0)
        ).sum()
    )
    disagreeing_presence = int((replay.notna() != published.notna()).sum())
    if mismatched or disagreeing_presence:
        raise DataReadinessError(
            "relative-volume baseline does not match a prior-session-only replay: "
            f"{mismatched} value mismatches, {disagreeing_presence} presence mismatches"
        )
    # A baseline built from exactly `lookback_sessions` prior observations can
    # never span more; a larger count means the current session entered it.
    over_wide = int(
        pd.to_numeric(frame["baseline_sessions"], errors="coerce")
        .fillna(0)
        .gt(lookback_sessions)
        .sum()
    )
    if over_wide:
        raise DataReadinessError(
            f"{over_wide} baselines span more than {lookback_sessions} prior sessions"
        )


def _prior_session_volume(frame: pd.DataFrame) -> SeriesGroupBy[Any, Any]:
    """Group each symbol's volume series shifted one session forward.

    Shifting before grouping is what removes the measured session from its own
    baseline; every rolling statistic in this module is taken on this object.
    """

    ordered = frame.sort_index()
    shifted = ordered.groupby("ticker", sort=False)["session_volume"].shift(1)
    return shifted.groupby(ordered["ticker"], sort=False)


def layer_one_mask(
    liquidity: pd.DataFrame,
    *,
    universe: IntradayUniverseContract,
) -> pd.Series:
    """Layer one: could this security be traded intraday at all.

    Bar continuity is the third layer-one filter in the contract and is not
    applied here: it is measured on five-minute prints, which do not exist for
    these candidates yet. It is enforced when those bars are collected.
    """

    return liquidity["average_volume_prior_sessions"].ge(
        float(universe.minimum_average_volume_shares)
    ) & liquidity["session_close"].between(
        universe.minimum_price,
        universe.maximum_price,
    )


def apply_intraday_universe_layers(
    liquidity: pd.DataFrame,
    *,
    universe: IntradayUniverseContract,
) -> pd.DataFrame:
    """Apply layer one, then layer two, then the per-session candidate cap."""

    if liquidity.empty:
        return pd.DataFrame(columns=list(SELECTION_COLUMNS))
    tradable = liquidity.loc[layer_one_mask(liquidity, universe=universe)]
    in_play = tradable.loc[
        tradable["relative_volume"].ge(universe.minimum_relative_volume)
    ]
    ranked = in_play.sort_values(
        ["session_date_et", "relative_volume", "ticker"],
        ascending=[True, False, True],
        kind="stable",
    )
    ranked = ranked.assign(
        session_rank=ranked.groupby("session_date_et", sort=False).cumcount() + 1
    )
    capped = ranked.loc[
        ranked["session_rank"].le(universe.maximum_candidates_per_session)
    ]
    return capped.loc[:, list(SELECTION_COLUMNS)].reset_index(drop=True)


def build_intraday_selection(
    *,
    collection_dir: Path,
    contract: StrategyContract,
    first_session: date,
    last_session: date,
    exclude_tickers: frozenset[str] = frozenset(),
) -> IntradaySelectionResult:
    """Screen a completed daily-bar collection into selected stock-sessions."""

    if first_session > last_session:
        raise ValueError("first_session must not be after last_session")
    universe = contract.intraday_universe
    artifacts = _collection_artifacts(collection_dir)
    excluded = sorted(exclude_tickers & set(artifacts))
    frames: list[pd.DataFrame] = []
    zero_volume_dropped = 0
    rows_read = 0
    for ticker in sorted(artifacts):
        if ticker in exclude_tickers:
            continue
        bars, _ = load_canonical_artifact(
            artifacts[ticker],
            expected_type="bars",
            columns=["ticker", "bar_start_utc", "close", "volume"],
        )
        rows_read += len(bars)
        traded, dropped = drop_placeholder_bars(bars)
        zero_volume_dropped += dropped
        if not traded.empty:
            frames.append(traded)
        del bars, traded
        assert_memory_budget(
            hard_budget_gib=_MEMORY_BUDGET_GIB,
            headroom_gib=_MEMORY_HEADROOM_GIB,
            stage=f"intraday screen read {ticker}",
        )
    release_process_memory()
    if not frames:
        raise DataReadinessError(f"no traded daily bars found under {collection_dir}")
    liquidity = session_liquidity(
        pd.concat(frames, ignore_index=True),
        lookback_sessions=universe.average_volume_lookback_sessions,
    )
    del frames
    assert_no_current_session_leak(
        liquidity,
        lookback_sessions=universe.average_volume_lookback_sessions,
    )
    liquidity = liquidity.loc[
        pd.Series(liquidity["session_date_et"]).between(first_session, last_session)
    ].reset_index(drop=True)
    selection = apply_intraday_universe_layers(liquidity, universe=universe)
    per_session = selection.groupby("session_date_et", sort=True).size()
    layer_one = liquidity.loc[layer_one_mask(liquidity, universe=universe)]
    audit: dict[str, Any] = {
        "schema": INTRADAY_SELECTION_SCHEMA,
        "strategy_id": contract.intraday.strategy_id,
        "strategy_contract_sha256": contract.sha256(),
        "collection_dir": str(collection_dir.resolve()),
        "first_session_et": first_session.isoformat(),
        "last_session_et": last_session.isoformat(),
        "symbols_read": len(artifacts) - len(excluded),
        "excluded_tickers": excluded,
        "daily_rows_read": rows_read,
        "zero_volume_rows_dropped": zero_volume_dropped,
        "liquidity_rows": int(len(liquidity)),
        "sessions_in_window": int(liquidity["session_date_et"].nunique()),
        "layer_one": {
            "minimum_average_volume_shares": universe.minimum_average_volume_shares,
            "average_volume_lookback_sessions": (
                universe.average_volume_lookback_sessions
            ),
            "minimum_price": universe.minimum_price,
            "maximum_price": universe.maximum_price,
            "stock_sessions": int(len(layer_one)),
            "symbols": int(layer_one["ticker"].nunique()),
            "bar_continuity_evaluated": False,
            "bar_continuity_reason": (
                "five-minute bars do not yet exist for these candidates; the "
                "continuity gate is evaluated when they are collected"
            ),
        },
        "layer_two": {
            "minimum_relative_volume": universe.minimum_relative_volume,
            "relative_volume_lookback_sessions": (
                universe.relative_volume_lookback_sessions
            ),
            "maximum_candidates_per_session": (
                universe.maximum_candidates_per_session
            ),
            "stock_sessions": int(len(selection)),
            "symbols": int(selection["ticker"].nunique()),
            "sessions_with_candidates": int(len(per_session)),
            "per_session_minimum": int(per_session.min()) if len(per_session) else 0,
            "per_session_median": float(per_session.median()) if len(per_session) else 0.0,
            "per_session_maximum": int(per_session.max()) if len(per_session) else 0,
            "sessions_without_candidates": (
                int(liquidity["session_date_et"].nunique()) - int(len(per_session))
            ),
        },
        "memory": memory_audit(
            hard_budget_gib=_MEMORY_BUDGET_GIB,
            headroom_gib=_MEMORY_HEADROOM_GIB,
        ).to_record(),
    }
    return IntradaySelectionResult(liquidity=liquidity, selection=selection, audit=audit)


def publish_intraday_selection(
    result: IntradaySelectionResult,
    *,
    output_directory: Path,
) -> dict[str, Any]:
    """Write the screen as a hash-bound, replayable research artifact.

    The manifest carries each table's sha256 and row count; the authority binds
    the manifest's own hash to the request identity. A reader that verifies both
    cannot be handed a table that was edited after publication.
    """

    if (output_directory / "_authority.json").exists():
        raise DataReadinessError(
            f"published intraday selection is immutable: {output_directory}"
        )
    output_directory.mkdir(parents=True, exist_ok=True)
    request = {
        key: result.audit[key]
        for key in (
            "schema",
            "strategy_id",
            "strategy_contract_sha256",
            "collection_dir",
            "first_session_et",
            "last_session_et",
            "excluded_tickers",
        )
    }
    request_sha256 = json_sha256(request)
    write_plan_json(
        output_directory / "_request.json",
        {**request, "request_sha256": request_sha256},
    )

    tables = {
        "session_liquidity.parquet": result.liquidity,
        "selected_stock_sessions.parquet": result.selection,
    }
    records: list[dict[str, Any]] = []
    for name, frame in tables.items():
        path = output_directory / name
        frame.to_parquet(path, index=False)
        records.append(
            {
                "path": name,
                "rows": int(len(frame)),
                "columns": list(frame.columns),
                "sha256": file_sha256(path),
            }
        )
    manifest = {
        **result.audit,
        "request_sha256": request_sha256,
        "published_at_utc": datetime.now(UTC).isoformat(),
        "tables": records,
    }
    write_plan_json(output_directory / "_manifest.json", manifest)
    write_plan_json(
        output_directory / "_authority.json",
        {
            "schema": INTRADAY_SELECTION_AUTHORITY_SCHEMA,
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(output_directory / "_manifest.json"),
            "request_sha256": request_sha256,
        },
    )
    return manifest


def load_complete_intraday_selection(directory: Path) -> dict[str, Any]:
    """Verify a published screen against its authority before any read."""

    request = load_plan_json(directory / "_request.json")
    manifest = load_plan_json(directory / "_manifest.json")
    authority = load_plan_json(directory / "_authority.json")
    request_sha256 = str(request.get("request_sha256", ""))
    payload = {key: value for key, value in request.items() if key != "request_sha256"}
    if (
        json_sha256(payload) != request_sha256
        or manifest.get("schema") != INTRADAY_SELECTION_SCHEMA
        or manifest.get("request_sha256") != request_sha256
        or authority.get("schema") != INTRADAY_SELECTION_AUTHORITY_SCHEMA
        or authority.get("state") != "complete"
        or authority.get("artifact") != "_manifest.json"
        or authority.get("artifact_sha256") != file_sha256(directory / "_manifest.json")
        or authority.get("request_sha256") != request_sha256
    ):
        raise DataReadinessError(
            f"intraday selection lacks matching complete authority: {directory}"
        )
    tables = manifest.get("tables")
    if not isinstance(tables, list) or not tables:
        raise DataReadinessError(f"intraday selection manifest has no tables: {directory}")
    for record in tables:
        path = directory / str(record["path"])
        if not path.is_file() or file_sha256(path) != record["sha256"]:
            raise DataReadinessError(f"intraday selection table failed its hash: {path}")
        if len(pd.read_parquet(path, columns=["ticker"])) != int(record["rows"]):
            raise DataReadinessError(f"intraday selection table row count moved: {path}")
    return manifest


def _collection_artifacts(collection_dir: Path) -> dict[str, Path]:
    """Resolve per-symbol bar artifacts from a completed collection manifest."""

    manifest_path = collection_dir / "_manifest.json"
    if not manifest_path.is_file():
        raise DataReadinessError(
            f"daily collection is not finalized, no manifest at {manifest_path}"
        )
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))
    records = payload.get("artifacts")
    if not isinstance(records, list) or not records:
        raise DataReadinessError(f"daily collection manifest has no artifacts: {manifest_path}")
    artifacts: dict[str, Path] = {}
    for record in records:
        ticker = str(record["ticker"]).strip().upper()
        artifacts[ticker] = Path(str(record["path"]))
    return artifacts
