"""Unadjusted ("raw") Alpaca SIP daily closes aligned to the research panel (research lane step 2a).

Market value must use unadjusted closes: provider-adjusted history divides past prices by
later splits and dividends, so an adjusted price times a historical share count would
encode future corporate actions. Every parquet read applies a session-date row filter at
the development end, and the rows are re-checked.
"""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.research_lane.panel import ResearchPanel, SealedPeriodError, file_sha256

UNADJUSTED_DIRECTORY: Final = Path("data/raw/swing_initial_fit_raw_share_canonical")
UNADJUSTED_MANIFEST: Final = "_manifest.json"
UNADJUSTED_AUTHORITY: Final = "_authority.json"
UNADJUSTED_COLUMNS: Final = ("security_id", "session_date", "close", "source", "timeframe", "price_feed", "adjustment")


def load_unadjusted_close(directory: Path, panel: ResearchPanel) -> tuple[npt.NDArray[np.float64], dict[str, Any],
                                                                            dict[str, str]]:
    """Sessions x securities unadjusted closes (NaN when absent) for the panel's securities."""
    manifest_path, authority_path = directory / UNADJUSTED_MANIFEST, directory / UNADJUSTED_AUTHORITY
    hashes = {"unadjusted_close_manifest_sha256": file_sha256(manifest_path),
              "unadjusted_close_authority_sha256": file_sha256(authority_path)}
    artifacts = json.loads(manifest_path.read_text(encoding="utf-8"))["unit_artifacts"]
    end_text = panel.development_end.isoformat()
    column_index = pd.Index(panel.security_ids)
    frames: list[pd.DataFrame] = []
    report: dict[str, Any] = {"units_listed": len(artifacts), "stock_units_read": 0}
    for item in artifacts:
        security = str(item["security_id"])
        if item.get("role") != "stock" or security not in column_index:
            continue
        path = directory / Path(str(item["bars_path"]).replace("\\", "/"))
        table = cast(Any, pq).read_table(path, columns=list(UNADJUSTED_COLUMNS), partitioning=None,
                                         filters=[("session_date", "<=", end_text)], use_threads=False)
        frame = table.to_pandas()
        del table
        if len(frame) and str(frame["session_date"].max()) > end_text:
            raise SealedPeriodError(f"unadjusted bars after {end_text} reached memory from {path}")
        contract = (frame["security_id"].eq(security) & frame["source"].eq("alpaca") & frame["timeframe"].eq("1Day")
                    & frame["price_feed"].eq("sip") & frame["adjustment"].eq("raw"))
        if not bool(contract.all()):
            raise ValueError(f"{path} has {int((~contract).sum())} rows that are not alpaca SIP raw daily bars "
                             f"for {security}")
        frames.append(frame[["security_id", "session_date", "close"]])
        report["stock_units_read"] += 1
    shape = (len(panel.sessions), len(panel.security_ids))
    matrix = np.full(shape, np.nan, dtype=np.float64)
    if not frames:
        report["securities_with_unadjusted_close"] = 0
        return matrix, report, hashes
    bars = pd.concat(frames, ignore_index=True)
    bars["session"] = pd.to_datetime(bars["session_date"]).to_numpy().astype("datetime64[D]")
    conflicting = bars.groupby(["security_id", "session"])["close"].transform("nunique").gt(1)
    report["rows_read"] = len(bars)
    report["rows_dropped_conflicting_duplicate"] = int(conflicting.sum())
    bars = bars.loc[~conflicting].drop_duplicates(["security_id", "session"])
    inside = bars["session"].isin(panel.sessions)
    report["rows_outside_session_calendar"] = int((~inside).sum())
    bars = bars.loc[inside]
    rows = np.searchsorted(panel.sessions, bars["session"].to_numpy().astype("datetime64[D]"))
    columns = column_index.get_indexer(bars["security_id"].astype(str))
    matrix[rows, columns] = bars["close"].to_numpy(dtype=np.float64)
    report["cells_placed"] = len(bars)
    report["securities_with_unadjusted_close"] = int(np.isfinite(matrix).any(axis=0).sum())
    report["first_session"] = str(bars["session"].min()) if len(bars) else None
    report["last_session"] = str(bars["session"].max()) if len(bars) else None
    return matrix, report, hashes
