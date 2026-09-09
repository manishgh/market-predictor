"""Bounded reads of exactly selected, immutable raw daily observations."""
from __future__ import annotations

from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import resolve_inside_authority
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.contracts.holding_materialization import PositionSourceBinding
from market_predictor.swing.datasets.symbol_corrections import inside
from market_predictor.swing.labels.holding_paths import validate_outcome_observations

RAW_COLUMNS = ["security_id", "ticker", "session_date", "bar_start_utc", "open", "high", "low", "close",
    "volume", "source", "timeframe", "price_feed", "adjustment", "ingested_at_utc"]


def read_bound_observations(root: Path, selection: dict[str, Any], binding: PositionSourceBinding,
    *, first: date, last: date, interpretation_sha256: str) -> tuple[pd.DataFrame, dict[date, EvidenceReference]]:
    """Hashes cover full files; Parquet numeric decoding is restricted to requested dates."""
    matches = [segment for segment in selection["segments"]
        if segment["artifact"]["unit_id"] == binding.unit_id
        and date.fromisoformat(segment["first_session"]) <= binding.first_session
        and date.fromisoformat(segment["last_session"]) >= binding.last_session]
    if len(matches) != 1:
        raise DataReadinessError("position binding must name one exact selected segment; no parent fallback")
    segment = matches[0]
    artifact = segment["artifact"]
    if (artifact["security_id"], artifact["provider_symbol"], artifact["bars_sha256"]) != (
        binding.security_id, binding.provider_symbol, binding.bars_sha256,
    ):
        raise DataReadinessError("position binding differs from selected issuer/source artifact")
    lower, upper = max(first, binding.first_session), min(last, binding.last_session)
    if lower > upper or lower < date(2019, 7, 9) or upper > date(2024, 5, 28):
        raise DataReadinessError("raw holding read is outside the initial-fit binding window")
    archive = inside(root, Path(segment["archive"]))
    path = resolve_inside_authority(archive, artifact["bars_path"])
    if file_sha256(path) != binding.bars_sha256:
        raise DataReadinessError("raw holding artifact hash differs")
    frame = pd.read_parquet(path, columns=RAW_COLUMNS,
        filters=[("session_date", ">=", str(lower)), ("session_date", "<=", str(upper))])
    if file_sha256(path) != binding.bars_sha256:
        raise DataReadinessError("raw holding artifact changed during read")
    if frame.empty:
        return frame, {}
    if (not frame.security_id.eq(binding.security_id).all() or not frame.ticker.eq(artifact["ticker"]).all()
            or not frame.adjustment.eq("raw").all() or not frame.price_feed.eq("sip").all()
            or not frame.source.eq("alpaca").all() or not frame.timeframe.eq("1Day").all()):
        raise DataReadinessError("raw holding identity, provider, adjustment or feed differs")
    dates = pd.to_datetime(frame.session_date, errors="raise").dt.date
    if dates.duplicated().any() or not dates.between(lower, upper).all():
        raise DataReadinessError("duplicate or out-of-range holding observation")
    calendar = xcals.get_calendar("XNYS")
    if not set(dates).issubset(session.date() for session in calendar.sessions_in_range(lower, upper)):
        raise DataReadinessError("holding observation is not an exchange session")
    # Provider midnight is retained in the artifact. These clocks only locate the
    # daily observation; they do not assert historical first-observation times.
    raw_clock = pd.to_datetime(frame.bar_start_utc, utc=True).dt.tz_convert("America/New_York")
    if not raw_clock.dt.date.eq(dates).all() or not raw_clock.eq(raw_clock.dt.normalize()).all():
        raise DataReadinessError("provider daily clock differs from its session")
    schedule = calendar.schedule.reindex(pd.DatetimeIndex(dates))
    observed = frame.rename(columns={"session_date": "session_date_et"}).copy()
    observed["bar_start_utc"], observed["bar_end_utc"] = schedule.open.array, schedule.close.array
    observed["available_at_utc"] = frame.ingested_at_utc
    observed = validate_outcome_observations(observed).set_index("session_date_et")
    evidence = {day: EvidenceReference(reference=path.relative_to(root).as_posix(),
        artifact_sha256=binding.bars_sha256, record_locator=f"security_id={binding.security_id};session_date={day}",
        interpretation_policy_sha256=interpretation_sha256,
        retrieved_at=pd.Timestamp(observed.loc[day, "ingested_at_utc"]).to_pydatetime(), available_at=None)
        for day in dates}
    return observed, evidence
