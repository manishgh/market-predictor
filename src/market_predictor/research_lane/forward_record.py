"""Append-only prospective forward score recorder (§5.6 & §6 Step 4 of Recommendation Brief).

Publishes one dated cross-sectional score snapshot (`equal_rank_24`, `reversal_short`,
`quality_slow`) for a single decision session `as_of_date` without reading any future
outcomes, allowing prospective forward evidence to mature over calendar time.

Historical sealed dates (`2024-05-29` through `2026-07-08`) are strictly refused unless
explicitly unsealed after freezing one candidate. Live prospective dates (`>= 2026-07-09`)
fetch trailing daily SIP bars from Alpaca (`[as_of_date - 400d, as_of_date]`) and never
compute or read outcomes inside the sealed window.
"""
from __future__ import annotations

import json
import os
import shutil
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from typing import Any, Final, cast
from uuid import uuid4

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.config import Settings, get_settings
from market_predictor.core.runtime_memory import guard
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.heavy_jobs import heavy_job_lease, heavy_job_runtime_dir
from market_predictor.research_lane import indicators, predictors, reversal
from market_predictor.research_lane.panel import (
    MARKET_TIMEZONE,
    PANEL_DIRECTORY,
    ResearchPanel,
    SealedPeriodError,
    check_development_end,
    load_research_panel,
)
from market_predictor.research_lane.predictors import centered_rank
from market_predictor.research_lane.trial_log import git_identity
from market_predictor.sources.alpaca import AlpacaSource

FORWARD_RECORD_DIRECTORY: Final = Path("data/research/research_lane/forward_records")
FORWARD_LOG_NAME: Final = "forward_log.jsonl"
HISTORICAL_SEALED_END: Final = date(2026, 7, 8)
RECORD_FORWARD_COMMAND: Final = "record-forward-research-scores"


def _score_panel_last_session(panel: ResearchPanel) -> pd.DataFrame:
    """Compute cross-sectional scores strictly at the final session row `-1` of `panel`."""
    if len(panel.sessions) == 0:
        raise ValueError("cannot score an empty panel")
    last_idx = len(panel.sessions) - 1
    session_str = str(panel.sessions[last_idx])
    elig_row = panel.eligible[last_idx]

    price_raw = predictors.compute_predictors(panel)
    signed_ranks = [
        predictors.EXPECTED_SIGNS[name] * centered_rank(price_raw[name], panel.eligible)[last_idx]
        for name in predictors.PREDICTOR_NAMES
    ]
    stack = np.stack(signed_ranks)
    fin = np.isfinite(stack)
    cnt = fin.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        price_eq_12 = np.where(elig_row & (cnt >= 8), np.where(fin, stack, 0.0).sum(axis=0) / np.maximum(cnt, 1), np.nan)

    ind = indicators.compute_indicators(panel)
    rev_short = reversal.reversal_signals(ind, panel.eligible)[reversal.REVERSAL_COMPOSITE][last_idx]

    rows: list[dict[str, Any]] = []
    for col, sec_id in enumerate(panel.security_ids):
        if not bool(elig_row[col]):
            continue
        sec_code = int(panel.sector[last_idx, col])
        sec_name = panel.sector_names[sec_code] if 0 <= sec_code < len(panel.sector_names) else ""
        rows.append({
            "session_date": session_str,
            "security_id": sec_id,
            "sector": sec_name,
            "close_adjusted": float(panel.close[last_idx, col]),
            "score_price_equal_12": None if not np.isfinite(price_eq_12[col]) else float(price_eq_12[col]),
            "score_reversal_short": None if not np.isfinite(rev_short[col]) else float(rev_short[col]),
        })
    return pd.DataFrame(rows)


def _fetch_live_prospective_panel(
    as_of_date: date, settings: Settings, *, unseal_final_test_period: bool = False,
) -> ResearchPanel:
    """Build a trailing live panel ending at `as_of_date` (`> 2026-07-08`), clamped after the sealed window."""
    if as_of_date <= HISTORICAL_SEALED_END:
        raise SealedPeriodError(
            f"prospective live mode requires as_of_date > {HISTORICAL_SEALED_END.isoformat()}; got {as_of_date}"
        )
    alpaca = AlpacaSource(settings)
    assets = alpaca.fetch_ticker_universe()
    symbols = sorted(set(assets["symbol"].astype(str).str.upper()) - {"SPY", "QQQ"})
    # Keep standard single-token alphabetic/dot common symbols to stay fast and memory-bounded.
    symbols = [s for s in symbols if len(s) <= 5 and s.replace(".", "").isalpha()][:1500]
    query_syms = ["SPY", "QQQ", *symbols]

    start_day = as_of_date - timedelta(days=410)
    if not unseal_final_test_period:
        # Strictly forbid reading any historical bar inside `2024-05-29..2026-07-08` while sealed!
        start_day = max(start_day, HISTORICAL_SEALED_END + timedelta(days=1))
    start_utc = datetime.combine(start_day, time(0, 0), tzinfo=UTC)
    end_utc = datetime.combine(as_of_date, time(23, 59, 59), tzinfo=UTC)

    bar_rows: list[dict[str, Any]] = []

    def _fetch_chunk(chunk: tuple[str, ...]) -> None:
        if not chunk:
            return
        token: str | None = None
        try:
            while True:
                page = alpaca.fetch_bars_page(
                    chunk, start_utc, end_utc, timeframe="1Day", page_token=token, asof=as_of_date, adjustment="all",
                )
                for sym, blist in page.bars.items():
                    for b in blist:
                        b_day = pd.Timestamp(str(b["t"])).tz_convert(MARKET_TIMEZONE).date()
                        if start_day <= b_day <= as_of_date:
                            bar_rows.append({
                                "ticker": sym,
                                "session": np.datetime64(b_day, "D"),
                                "open": float(b["o"]),
                                "high": float(b["h"]),
                                "low": float(b["l"]),
                                "close": float(b["c"]),
                                "volume": float(b["v"]),
                            })
                token = page.next_page_token
                if not token:
                    break
        except Exception as exc:
            if "status=400" in str(exc):
                if len(chunk) == 1:
                    return
                for single in chunk:
                    _fetch_chunk((single,))
                return
            raise

    for offset in range(0, len(query_syms), 45):
        _fetch_chunk(tuple(query_syms[offset : offset + 45]))
        guard("forward live panel fetch")

    df = pd.DataFrame(bar_rows).drop_duplicates(["ticker", "session"], keep="last")
    spy = df.loc[df["ticker"] == "SPY"].sort_values("session")
    qqq = df.loc[df["ticker"] == "QQQ"].sort_values("session")
    sessions = spy["session"].to_numpy().astype("datetime64[D]")
    qqq_al = pd.DataFrame({"session": sessions}).merge(qqq[["session", "open", "close"]], on="session", how="left")

    stocks_df = df.loc[~df["ticker"].isin(["SPY", "QQQ"]) & df["session"].isin(sessions)]
    sec_tickers = tuple(sorted(stocks_df["ticker"].unique()))
    sec_ids = tuple(f"ticker:{tk}" for tk in sec_tickers)
    col_idx = pd.Index(sec_tickers)

    shape = (len(sessions), len(sec_ids))
    r_pos = np.searchsorted(sessions, stocks_df["session"].to_numpy().astype("datetime64[D]"))
    c_pos = col_idx.get_indexer(stocks_df["ticker"])
    mats: dict[str, npt.NDArray[np.float32]] = {}
    for f_name in ("open", "high", "low", "close", "volume"):
        m = np.full(shape, np.nan, dtype=np.float32)
        m[r_pos, c_pos] = stocks_df[f_name].to_numpy(dtype=np.float32)
        mats[f_name] = m

    req_win = min(len(sessions), 252)
    min_valid = max(60, int(0.95 * req_win))
    fin_win = (
        (np.isfinite(mats["close"][-req_win:]) & (mats["close"][-req_win:] > 0)).sum(axis=0) >= min_valid
        if len(sessions) >= 60
        else np.zeros(len(sec_ids), dtype=bool)
    )
    dv_slice = mats["close"][-60:].astype(np.float64) * mats["volume"][-60:].astype(np.float64)
    dv_fin = np.isfinite(dv_slice)
    dv_cnt = dv_fin.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        dvol_60 = np.where(dv_cnt > 0, np.where(dv_fin, dv_slice, 0.0).sum(axis=0) / np.maximum(dv_cnt, 1), np.nan)
    elig_last = fin_win & np.isfinite(dvol_60) & np.isfinite(mats["close"][-1])
    if int(elig_last.sum()) > 1000:
        cutoff_dv = float(np.sort(dvol_60[elig_last])[-1000])
        elig_last = elig_last & (dvol_60 >= cutoff_dv)
    eligible = np.zeros(shape, dtype=bool)
    eligible[-1] = elig_last

    return ResearchPanel(
        sessions=sessions,
        security_ids=sec_ids,
        open=mats["open"],
        high=mats["high"],
        low=mats["low"],
        close=mats["close"],
        volume=mats["volume"],
        eligible=eligible,
        sector=np.zeros(shape, dtype=np.int16),
        sector_names=("Unknown",),
        spy_open=spy["open"].to_numpy(dtype=np.float32),
        spy_close=spy["close"].to_numpy(dtype=np.float32),
        qqq_open=qqq_al["open"].to_numpy(dtype=np.float32),
        qqq_close=qqq_al["close"].to_numpy(dtype=np.float32),
        development_end=as_of_date,
        load_report={"mode": "live_prospective_alpaca_sip", "as_of_date": as_of_date.isoformat()},
    )


def record_forward_scores(
    *,
    root: Path,
    as_of_date: date,
    data_root: Path | None = None,
    panel: Path = PANEL_DIRECTORY,
    output_parent: Path = FORWARD_RECORD_DIRECTORY,
    live_prospective: bool = False,
    unseal_final_test_period: bool = False,
    settings: Settings | None = None,
) -> dict[str, Any]:
    """Publish one immutable dated forward-score snapshot and append to `forward_log.jsonl`."""
    if not live_prospective:
        check_development_end(as_of_date, unseal_final_test_period=unseal_final_test_period)
    elif as_of_date <= HISTORICAL_SEALED_END and not unseal_final_test_period:
        raise SealedPeriodError(
            f"cannot record live prospective scores inside sealed window ({as_of_date.isoformat()} <= "
            f"{HISTORICAL_SEALED_END.isoformat()})"
        )

    paths = EvidencePaths(root, data_root)
    root = paths.project_root
    runtime = heavy_job_runtime_dir()

    with heavy_job_lease(RECORD_FORWARD_COMMAND, runtime_dir=runtime if runtime.is_absolute() else root / runtime):
        guard("forward score recorder start")
        if live_prospective:
            cfg = settings or get_settings()
            loaded_panel = _fetch_live_prospective_panel(
                as_of_date, cfg, unseal_final_test_period=unseal_final_test_period,
            )
        else:
            loaded_panel = load_research_panel(
                paths.directory(panel),
                development_end=as_of_date,
                unseal_final_test_period=unseal_final_test_period,
            )

        scores_df = _score_panel_last_session(loaded_panel)
        if scores_df.empty:
            raise ValueError(f"no eligible stocks scored on {as_of_date.isoformat()}")
        scored_session = str(scores_df["session_date"].iloc[0])

        dest_parent = paths.write(paths.identity(output_parent))
        dest_dir = dest_parent / scored_session
        if dest_dir.exists():
            raise FileExistsError(f"forward score snapshot already exists for session {scored_session}: {dest_dir}")

        dest_parent.mkdir(parents=True, exist_ok=True)
        staging = dest_parent / f".{scored_session}.{uuid4().hex}.partial"
        staging.mkdir(parents=True)
        manifest = {
            "recorded_at_utc": datetime.now(UTC).isoformat(),
            "requested_as_of_date": as_of_date.isoformat(),
            "scored_session": scored_session,
            "live_prospective": live_prospective,
            "eligible_stocks_scored": len(scores_df),
            "code": git_identity(root),
            "label_notice": "research forward score snapshot (zero future outcomes read)",
        }
        try:
            cast(Any, pq).write_table(pa.Table.from_pandas(scores_df, preserve_index=False), staging / "scores.parquet")
            (staging / "_manifest.json").write_text(
                json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8",
            )
            os.replace(staging, dest_dir)
        finally:
            if staging.exists():
                shutil.rmtree(staging)

        log_path = dest_parent / FORWARD_LOG_NAME
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(manifest, sort_keys=True) + "\n")

    return manifest
