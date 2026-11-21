"""Unit tests on synthetic files: sealed-period readers for SEC filings and unadjusted closes, and the command run."""
from __future__ import annotations

import json
from datetime import date
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import pytest

from market_predictor.research_lane.company_facts import (
    IssuerFacts,
)
from market_predictor.research_lane.panel import MARKET_TIMEZONE, SealedPeriodError, load_research_panel
from market_predictor.research_lane.sec_filings import (
    FILINGS_FILE,
    FILINGS_MANIFEST,
    earnings_releases,
    read_sec_filings,
    security_issuers,
)
from market_predictor.research_lane.unadjusted_prices import load_unadjusted_close
from tests.support.research_lane_panel import POST_END_PRICE, write_panel_files

END = date(2024, 5, 28)


def _utc(local: str) -> pd.Timestamp:
    return pd.Timestamp(local, tz=MARKET_TIMEZONE).tz_convert("UTC")


def write_filings(directory: Path) -> None:
    rows = [
        ("sec:A", "0000000001", "8-K", "2.02,9.01", "2024-04-25 07:00"),
        ("sec:A", "0000000001", "8-K/A", "2.02", "2024-04-26 07:00"),
        ("sec:A", "0000000001", "10-Q", None, "2024-04-30 16:30"),
        ("sec:B", "0000000002", "8-K", "7.01", "2024-05-01 09:00"),
        ("sec:B", "0000000002", "8-K", "2.02", "2024-05-28 23:30"),  # last New York day: read
        ("sec:B", "0000000002", "8-K", "2.02", "2024-05-29 00:30"),  # first sealed day: never read
        ("sec:C", "0000000003", "8-K", "2.02", "2025-02-01 08:00"),
    ]
    frame = pd.DataFrame(rows, columns=["security_id", "sec_cik", "sec_form", "item_codes", "local"])
    frame["accession_number"] = [f"acc-{i}" for i in range(len(frame))]
    frame["accepted_at_utc"] = [_utc(text) for text in frame.pop("local")]
    frame["available_at_utc"] = frame["accepted_at_utc"] + pd.Timedelta(minutes=5)
    directory.mkdir(parents=True, exist_ok=True)
    pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), directory / FILINGS_FILE)
    (directory / FILINGS_MANIFEST).write_text(json.dumps({"schema": "synthetic"}), encoding="utf-8")


def test_filings_reader_never_returns_rows_accepted_after_the_development_end(tmp_path: Path) -> None:
    write_filings(tmp_path / "inventory")
    filings = read_sec_filings(tmp_path / "inventory" / FILINGS_FILE, END)
    assert len(filings) == 5
    assert filings["accepted_at_utc"].max() == _utc("2024-05-28 23:30")
    releases, report = earnings_releases(filings)
    assert sorted(releases["accession_number"]) == ["acc-0", "acc-4"]
    assert report["amended_8k_item_2_02_rows_not_counted"] == 1
    assert security_issuers(filings) == {"sec:A": ("0000000001",), "sec:B": ("0000000002",)}


def test_later_sealed_inventory_is_refused(tmp_path: Path) -> None:
    directory = tmp_path / "swing_later_sealed_sec_form_inventory"
    write_filings(directory)
    with pytest.raises(SealedPeriodError):
        read_sec_filings(directory / FILINGS_FILE, END)


def write_unadjusted(directory: Path, units: dict[str, tuple[str, list[date]]], *, adjustment: str = "raw") -> None:
    artifacts = []
    for unit, (security, days) in units.items():
        path = directory / "units" / unit
        path.mkdir(parents=True)
        frame = pd.DataFrame({
            "security_id": security, "ticker": security[-1], "role": "stock",
            "bar_start_utc": pd.to_datetime([f"{d.isoformat()} 04:00" for d in days], utc=True),
            "session_date": [d.isoformat() for d in days],
            "open": 1.0, "high": 1.0, "low": 1.0,
            "close": [POST_END_PRICE if d > END else 20.0 + k for k, d in enumerate(days)],
            "volume": 1, "source": "alpaca", "timeframe": "1Day", "price_feed": "sip", "adjustment": adjustment,
        })
        pq.write_table(pa.Table.from_pandas(frame, preserve_index=False), path / "bars.parquet")
        artifacts.append({"security_id": security, "role": "stock", "bars_path": f"units\\{unit}\\bars.parquet"})
    artifacts.append({"security_id": "benchmark:SPY", "role": "benchmark", "bars_path": "units\\missing\\bars.parquet"})
    (directory / "_manifest.json").write_text(json.dumps({"unit_artifacts": artifacts}), encoding="utf-8")
    (directory / "_authority.json").write_text(json.dumps({"schema": "synthetic"}), encoding="utf-8")


def test_unadjusted_reader_never_loads_rows_after_the_development_end(tmp_path: Path) -> None:
    layout = write_panel_files(tmp_path / "panel", end=END)
    panel = load_research_panel(tmp_path / "panel", development_end=END)
    write_unadjusted(tmp_path / "raw", {"u1": ("sec:A", layout["days"]), "u2": ("sec:Z", layout["days"])})
    matrix, report, hashes = load_unadjusted_close(tmp_path / "raw", panel)
    assert matrix.shape == panel.close.shape
    assert not np.any(np.abs(np.nan_to_num(matrix) - POST_END_PRICE) < 10.0)
    assert np.isfinite(matrix[:, 0]).all() and np.isnan(matrix[:, 1:]).all()
    assert report["stock_units_read"] == 1 and set(hashes) == {"unadjusted_close_manifest_sha256",
                                                               "unadjusted_close_authority_sha256"}


def test_unadjusted_reader_refuses_adjusted_bars(tmp_path: Path) -> None:
    layout = write_panel_files(tmp_path / "panel", end=END)
    panel = load_research_panel(tmp_path / "panel", development_end=END)
    write_unadjusted(tmp_path / "raw", {"u1": ("sec:A", layout["days"])}, adjustment="all")
    with pytest.raises(ValueError, match="raw daily bars"):
        load_unadjusted_close(tmp_path / "raw", panel)


def companyfacts_document(facts: IssuerFacts) -> dict[str, object]:
    document: dict[str, dict[str, dict[str, object]]] = {"us-gaap": {}, "dei": {}}
    for key, items in facts.items():
        taxonomy, tag = key.split(":")
        unit = "shares" if "Shares" in tag else "USD"
        rows = [{**({"start": f.start.isoformat()} if f.start else {}), "end": f.end.isoformat(), "val": f.value,
                 "accn": f.accession, "filed": f.filed.isoformat(), "form": "10-Q"} for f in items]
        document[taxonomy][tag] = {"units": {unit: rows}}
    return {"facts": document}
