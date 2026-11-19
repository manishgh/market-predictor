"""Synthetic research-lane panels for unit tests only; never real-data evidence."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from market_predictor.research_lane.panel import ResearchPanel

POST_END_PRICE = 987654.0


def synthetic_panel(*, sessions: int = 620, securities: int = 40, seed: int = 7) -> ResearchPanel:
    """Random-walk adjusted prices with gaps, eligibility holes and a sector change (unit tests only)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range("2018-05-29", periods=sessions).to_numpy().astype("datetime64[D]")
    market = rng.normal(0.0004, 0.01, sessions)
    beta = rng.uniform(0.5, 1.5, securities)
    returns = market[:, None] * beta[None, :] + rng.normal(0.0, 0.015, (sessions, securities))
    close = 50.0 * np.exp(np.cumsum(returns, axis=0))
    open_ = close * np.exp(rng.normal(0.0, 0.004, close.shape))
    high = np.maximum(open_, close) * (1.0 + np.abs(rng.normal(0.0, 0.006, close.shape)))
    low = np.minimum(open_, close) * (1.0 - np.abs(rng.normal(0.0, 0.006, close.shape)))
    volume = np.exp(rng.normal(14.0, 0.5, close.shape))
    for column, start, length in ((3, 100, 4), (7, 300, 1), (11, 450, 30), (15, 0, 280)):
        for matrix in (open_, high, low, close, volume):
            matrix[start:start + length, column] = np.nan
    eligible = rng.random(close.shape) > 0.08
    eligible[:280, 15] = False
    sector = np.tile(np.arange(securities, dtype=np.int16) % 4, (sessions, 1))
    sector[400:, 5] = 2
    spy_close = 300.0 * np.exp(np.cumsum(market))
    spy_open = spy_close * np.exp(rng.normal(0.0, 0.002, sessions))
    qqq_close = 200.0 * np.exp(np.cumsum(market * 1.2))
    qqq_open = qqq_close * np.exp(rng.normal(0.0, 0.002, sessions))
    f32 = np.float32
    return ResearchPanel(
        sessions=days, security_ids=tuple(f"sec:{index:03d}" for index in range(securities)),
        open=open_.astype(f32), high=high.astype(f32), low=low.astype(f32), close=close.astype(f32),
        volume=volume.astype(f32), eligible=eligible, sector=sector, sector_names=("A", "B", "C", "D"),
        spy_open=spy_open.astype(f32), spy_close=spy_close.astype(f32), qqq_open=qqq_open.astype(f32),
        qqq_close=qqq_close.astype(f32), development_end=pd.Timestamp(days[-1]).date())


def bar_table(ticker: str, days: list[date], base: float, end: date) -> pa.Table:
    starts = pd.to_datetime([f"{day.isoformat()} 13:30" for day in days], utc=True)
    close = np.array([POST_END_PRICE if day > end else base + index for index, day in enumerate(days)], dtype=float)
    frame = pd.DataFrame({
        "ticker": ticker, "timeframe": "1d", "bar_start_utc": starts,
        "bar_end_utc": starts + pd.Timedelta(hours=6, minutes=30),
        "available_at_utc": starts + pd.Timedelta(hours=6, minutes=45),
        "open": close - 0.5, "high": close + 1.0, "low": close - 1.0, "close": close,
        "volume": np.full(len(days), 1000, dtype=np.int64),
        "source": "alpaca", "price_feed": "sip", "adjustment": "all",
    })
    return pa.Table.from_pandas(frame, preserve_index=False)


def write_panel_files(panel_root: Path, *, end: date = date(2024, 5, 28)) -> dict[str, list[date]]:
    """Write a tiny retained-panel layout whose rows run past ``end``.

    Securities: sec:A (ticker AAA), sec:B (ticker OLD then NEW), sec:C and sec:D (ticker
    XYZ reused). The year=2025 partition and the XLK file are not parquet; opening
    either raises, proving the loader never reads them. Bars after ``end`` carry
    ``POST_END_PRICE`` so any leak is visible.
    """
    days = [day.date() for day in pd.bdate_range("2024-04-01", "2024-06-28")]
    middle = days[len(days) // 3]
    plan = {
        "AAA": [("sec:A", "Tech", day) for day in days],
        "OLD": [("sec:B", "Energy", day) for day in days if day < middle],
        "NEW": [("sec:B", "Energy", day) for day in days if day >= middle],
        "XYZ": [("sec:C" if day < middle else "sec:D", "Tech", day) for day in days],
    }
    bars_directory = panel_root / "combined_daily" / "bars"
    bars_directory.mkdir(parents=True)
    artifacts = []
    for ticker, base in (("SPY", 400.0), ("QQQ", 300.0)):
        pq.write_table(bar_table(ticker, days, base, end), bars_directory / f"{ticker}.parquet")
        artifacts.append({"ticker": ticker, "path": f"bars/{ticker}.parquet"})
    (bars_directory / "XLK.parquet").write_bytes(b"not parquet")
    artifacts.append({"ticker": "XLK", "path": "bars/XLK.parquet"})
    rows = []
    for offset, (ticker, entries) in enumerate(plan.items()):
        pq.write_table(bar_table(ticker, [day for _, _, day in entries], 10.0 * (offset + 1), end),
                       bars_directory / f"{ticker}.parquet")
        artifacts.append({"ticker": ticker, "path": f"bars/{ticker}.parquet"})
        for security, sector, day in entries:
            rows.append({"ticker": ticker, "security_id": security, "sector": sector, "session_date_et": day,
                         "decision_time_utc": pd.Timestamp(f"{day.isoformat()} 23:00", tz="UTC"),
                         "membership_available_at_utc": pd.Timestamp("2018-05-29 04:00", tz="UTC"),
                         "cross_section_eligible": False, "pre_catalyst_feature_eligible": True})
    (panel_root / "combined_daily" / "_manifest.json").write_text(json.dumps({"artifacts": artifacts}), encoding="utf-8")
    final = panel_root / "final"
    (final / "panel" / "feature_profile=technical_market" / "year=2024").mkdir(parents=True)
    (final / "panel" / "feature_profile=technical_market" / "year=2025").mkdir(parents=True)
    (final / "_manifest.json").write_text(json.dumps({"schema": "synthetic"}), encoding="utf-8")
    table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
    pq.write_table(table, final / "panel" / "feature_profile=technical_market" / "year=2024" / "part.parquet")
    (final / "panel" / "feature_profile=technical_market" / "year=2025" / "part.parquet").write_bytes(b"not parquet")
    return {"days": days, "middle": [middle]}
