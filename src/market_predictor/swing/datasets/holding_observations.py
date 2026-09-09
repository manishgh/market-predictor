"""Required-session raw observations, with ownership distinct from price validity."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pyarrow as pa
import pyarrow.dataset as ds

from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.labels.holding_identity import membership_session_coverage
from market_predictor.swing.labels.holding_paths import holding_calendar, validate_outcome_observations

RAW_COLUMNS = (
    "ticker", "timeframe", "bar_start_utc", "bar_end_utc", "available_at_utc",
    "open", "high", "low", "close", "volume", "price_feed", "adjustment",
)


def read_required_holding_observations(
    path: Path | None, *, ticker: str, security_id: str, sessions: tuple[date, ...],
    numeric_end: date, memberships: pd.DataFrame,
) -> pd.DataFrame:
    """Return one diagnostic row per requirement, never assign an unresolved owner.

    A missing path means the source inventory has no artifact. Malformed schemas,
    duplicate rows and clocks are corrupt evidence and raise, not missing bars.
    Numeric values are decoded through a filtered Arrow scan, never filtered after
    decoding the complete seven-year numeric history into pandas.
    """
    if (not sessions or tuple(sorted(set(sessions))) != sessions or len(sessions) > 100
            or sessions[-1] > numeric_end or not set(sessions).issubset(holding_calendar(sessions[0], sessions[-1]))):
        raise DataReadinessError("holding observations require bounded exact sessions inside the numeric scope")
    coverage = membership_session_coverage(
        memberships.loc[memberships.ticker.eq(ticker)],
        sessions=holding_calendar(sessions[0], sessions[-1]), security_ids=(security_id,),
    )["membership_covered"]
    result = pd.DataFrame({"session_date_et": sessions, "ticker": ticker, "requested_security_id": security_id})
    keys = pd.MultiIndex.from_arrays([[security_id] * len(sessions), sessions], names=coverage.index.names)
    owned = coverage.reindex(keys).to_numpy(dtype=bool)
    result["ownership_status"] = ["membership_supported" if value else "ownership_unresolved" for value in owned]
    result["security_id"] = pd.Series([security_id if value else None for value in owned], dtype="string")
    if path is None:
        result["source_present"] = False
        result["observation_status"] = "observation_missing"
        return result
    arrow: Any = ds
    dataset = arrow.dataset(path, format="parquet")
    start_type = dataset.schema.field("bar_start_utc").type
    if not pa.types.is_timestamp(start_type) or start_type.tz != "UTC":
        raise DataReadinessError("holding raw observations require an explicit UTC timestamp schema")
    if dataset.count_rows(filter=arrow.field("bar_start_utc").is_null()):
        raise DataReadinessError("holding raw observations contain unassignable null start timestamps")
    predicate = arrow.field("bar_start_utc").cast(pa.date32()).isin(list(sessions))
    if dataset.count_rows(filter=predicate) > len(sessions):
        raise DataReadinessError("duplicate holding observations or unexpected rows")
    frame = dataset.to_table(columns=list(RAW_COLUMNS), filter=predicate, use_threads=False).to_pandas()
    if len(frame) > len(sessions):
        raise DataReadinessError("duplicate holding observations or unexpected rows")
    if frame.empty:
        result["source_present"] = False
        result["observation_status"] = "observation_missing"
        return result
    frame["session_date_et"] = pd.to_datetime(frame.bar_start_utc, utc=True).dt.date
    if (frame.session_date_et.duplicated().any() or not set(frame.session_date_et).issubset(sessions)
            or not frame.ticker.eq(ticker).all() or not frame.timeframe.eq("1d").all()
            or not frame.price_feed.eq("sip").all() or not frame.adjustment.eq("all").all()):
        raise DataReadinessError("holding raw observation identity, feed or session differs")
    frame = validate_outcome_observations(frame)
    frame["source_present"] = True
    frame["observation_status"] = frame.outcome_observation_valid.map({True: "observation_valid", False: "observation_invalid"})
    output = result.merge(frame.drop(columns="ticker"), how="left", on="session_date_et", validate="one_to_one")
    output["source_present"] = output.source_present.eq(True)
    output["observation_status"] = output.observation_status.fillna("observation_missing")
    return output
