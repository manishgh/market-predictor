"""Point-in-time reader for the retained initial-fit SEC form inventory (research lane step 2a).

Only rows accepted on or before the development end (New York calendar day) are read, by a
parquet row filter, and the result is re-checked. The later-sealed inventories are refused.
An earnings release is an ``8-K`` (not ``8-K/A``) whose comma-separated item codes include 2.02.
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path
from typing import Any, Final, cast

import numpy as np
import numpy.typing as npt
import pandas as pd
import pyarrow.parquet as pq

from market_predictor.research_lane.panel import MARKET_TIMEZONE, SealedPeriodError

FILINGS_DIRECTORY: Final = Path("data/research/swing_initial_fit_sec_form_inventory_corrected_clock")
FILINGS_FILE: Final = "filings.parquet"
FILINGS_MANIFEST: Final = "_manifest.json"
SEALED_DIRECTORY_PREFIX: Final = "swing_later_sealed_"
FILING_COLUMNS: Final = ("security_id", "sec_cik", "accession_number", "sec_form", "item_codes",
                         "accepted_at_utc", "available_at_utc")
EARNINGS_FORM: Final = "8-K"
EARNINGS_ITEM: Final = "2.02"
REGULAR_CLOSE: Final = pd.Timedelta(hours=16)


def acceptance_cutoff(development_end: date) -> pd.Timestamp:
    """First instant after the development end's New York calendar day, in UTC."""
    return pd.Timestamp(development_end + timedelta(days=1), tz=MARKET_TIMEZONE).tz_convert("UTC")


def read_sec_filings(path: Path, development_end: date) -> pd.DataFrame:
    """Rows accepted before the end of the development end's New York day; later rows never reach memory."""
    if any(part.startswith(SEALED_DIRECTORY_PREFIX) for part in path.parts):
        raise SealedPeriodError(f"{path} belongs to a later-sealed SEC inventory and is not read")
    cutoff = acceptance_cutoff(development_end)
    table = cast(Any, pq).read_table(path, columns=list(FILING_COLUMNS), partitioning=None,
                                     filters=[("accepted_at_utc", "<", cutoff.to_pydatetime())])
    frame: pd.DataFrame = table.to_pandas()
    if len(frame) and (frame["accepted_at_utc"].isna().any() or bool((frame["accepted_at_utc"] >= cutoff).any())):
        raise SealedPeriodError(f"SEC filings accepted after {development_end} reached memory from {path}")
    return frame


def has_earnings_item(item_codes: pd.Series) -> npt.NDArray[np.bool_]:
    items = item_codes.fillna("").astype(str).map(lambda text: EARNINGS_ITEM in {code.strip() for code in text.split(",")})
    return np.asarray(items.to_numpy(dtype=bool), dtype=bool)


def is_earnings_release(forms: pd.Series, item_codes: pd.Series) -> npt.NDArray[np.bool_]:
    return np.asarray(forms.eq(EARNINGS_FORM).to_numpy(dtype=bool) & has_earnings_item(item_codes), dtype=bool)


def earnings_releases(filings: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, Any]]:
    """8-K item 2.02 rows with the time used for event mapping (the inventory's availability time)."""
    mask = is_earnings_release(filings["sec_form"], filings["item_codes"])
    releases = filings.loc[mask, ["security_id", "sec_cik", "accession_number", "accepted_at_utc",
                                  "available_at_utc"]].copy()
    unknown = releases["available_at_utc"].isna()
    earlier = releases["available_at_utc"] < releases["accepted_at_utc"]
    report: dict[str, Any] = {
        "earnings_release_rows": int(mask.sum()),
        "earnings_release_rows_without_availability_time": int(unknown.sum()),
        "earnings_release_rows_available_before_acceptance": int(earlier.sum()),
        "amended_8k_item_2_02_rows_not_counted": int((filings["sec_form"].eq("8-K/A").to_numpy(dtype=bool)
                                                      & has_earnings_item(filings["item_codes"])).sum()),
        "securities_with_release": int(releases["security_id"].nunique()),
    }
    if bool(earlier.any()):
        raise ValueError("an earnings release is available before its SEC acceptance time")
    releases = releases.loc[~unknown]
    if len(releases):
        report["first_acceptance_utc"] = str(releases["accepted_at_utc"].min())
        report["last_acceptance_utc"] = str(releases["accepted_at_utc"].max())
    return releases.reset_index(drop=True), report


def security_issuers(filings: pd.DataFrame) -> dict[str, tuple[str, ...]]:
    """CIKs under which each security's filings were recorded by the inventory."""
    pairs = filings[["security_id", "sec_cik"]].dropna().drop_duplicates()
    return {str(security): tuple(sorted(str(cik) for cik in group["sec_cik"]))
            for security, group in pairs.groupby("security_id")}


def session_close_times(sessions: npt.NDArray[np.datetime64]) -> npt.NDArray[np.int64]:
    """16:00 America/New_York of each session as UTC nanoseconds."""
    local = pd.DatetimeIndex(np.asarray(sessions, dtype="datetime64[ns]")) + REGULAR_CLOSE
    utc = local.tz_localize(MARKET_TIMEZONE).tz_convert("UTC").as_unit("ns")
    return np.asarray(utc.asi8, dtype=np.int64)


def event_rows(sessions: npt.NDArray[np.datetime64], times: pd.Series) -> npt.NDArray[np.int64]:
    """Index of the first session whose regular close is at or after each time (len(sessions) when none)."""
    closes = session_close_times(sessions)
    instants = np.asarray(pd.DatetimeIndex(times).tz_convert("UTC").as_unit("ns").asi8, dtype=np.int64)
    return np.asarray(np.searchsorted(closes, instants, side="left"), dtype=np.int64)
