"""Exact retained raw sources for investment outcomes, without swing date limits."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.swing.contracts.holding_accounting import EvidenceReference
from market_predictor.swing.datasets.holding_raw_sources import RAW_COLUMNS
from market_predictor.swing.labels.holding_paths import validate_outcome_observations


@dataclass(frozen=True)
class InvestmentObservations:
    frame: pd.DataFrame
    evidence: dict[date, EvidenceReference]
    provider_symbol: str | None
    missing_reasons: tuple[str, ...]


class InvestmentSourceReader:
    """Compose exact nonoverlapping selected segments with per-source capture clocks.

    Selection must already have been independently reconstructed by the caller.
    Ingestion is an observed label clock, never a historical feature clock.
    """

    def __init__(self, root: Path, selection: dict[str, Any], interpretation_sha256: str, *,
                 paths: EvidencePaths | None = None) -> None:
        self.paths = paths or EvidencePaths(root)
        self.selection = selection
        self.interpretation_sha256 = interpretation_sha256
        self.source_files: dict[str, str] = {}

    def recheck(self) -> None:
        for name, digest in self.source_files.items():
            if file_sha256(self.paths.read(name)) != digest:
                raise DataReadinessError("investment raw source changed during materialization")

    def read(self, security_id: str, ticker: str, sessions: tuple[date, ...], *, benchmark: bool
             ) -> InvestmentObservations:
        if not sessions or tuple(sorted(set(sessions))) != sessions:
            raise DataReadinessError("investment source sessions must be ordered, unique and nonempty")
        matches = [segment for segment in self.selection["segments"]
            if segment["artifact"]["security_id"] == security_id
            and date.fromisoformat(segment["first_session"]) <= sessions[-1]
            and date.fromisoformat(segment["last_session"]) >= sessions[0]]
        owners = {day: [segment for segment in matches
            if date.fromisoformat(segment["first_session"]) <= day <= date.fromisoformat(segment["last_session"])]
            for day in sessions}
        overlapping = [segments for segments in owners.values() if len(segments) > 1]
        if overlapping:
            # Old symbol plans retain short horizon tails past a rename. Do not
            # choose either ticker, stitch their observations, or hide another
            # genuine duplicate elsewhere in the requested window.
            duplicate_ticker = any(len({segment["artifact"]["ticker"] for segment in segments}) != len(segments)
                                   for segments in overlapping)
            if (duplicate_ticker or benchmark or any(segment["artifact"]["role"] != "stock" for segment in matches)):
                raise DataReadinessError("investment selected raw sources overlap session ownership")
            return InvestmentObservations(pd.DataFrame(), {}, None, ("raw_security_symbol_or_role_differs",))
        if any(not segments for segments in owners.values()):
            return InvestmentObservations(pd.DataFrame(), {}, None, ("missing_or_cross_segment_raw_binding",))
        if any(segment.get("source_admission") == "quarantined_unreviewed_provider_candidate" for segment in matches):
            return InvestmentObservations(pd.DataFrame(), {}, None, ("raw_stock_extension_identity_unreviewed",))
        expected_role = "benchmark" if benchmark else "stock"
        symbols = {str(segment["artifact"]["provider_symbol"]) for segment in matches}
        if (len(symbols) != 1 or any(segment["artifact"]["role"] != expected_role
                or segment["artifact"]["ticker"] != ticker for segment in matches)):
            return InvestmentObservations(pd.DataFrame(), {}, None, ("raw_security_symbol_or_role_differs",))
        symbol = next(iter(symbols))
        frames = []
        day_sources: dict[date, tuple[str, str]] = {}
        for segment in matches:
            artifact = segment["artifact"]
            covered = tuple(day for day in sessions if owners[day][0] is segment)
            if not covered:
                continue
            path = self.paths.read_child(segment["archive"], artifact["bars_path"])
            digest = str(artifact["bars_sha256"])
            if file_sha256(path) != digest:
                raise DataReadinessError("investment raw artifact hash differs")
            name = self.paths.relative(path).as_posix()
            if name in self.source_files and self.source_files[name] != digest:
                raise DataReadinessError("investment raw source has conflicting hashes")
            self.source_files[name] = digest
            part = pd.read_parquet(path, columns=RAW_COLUMNS,
                filters=[("session_date", ">=", covered[0].isoformat()),
                         ("session_date", "<=", covered[-1].isoformat())])
            if file_sha256(path) != digest:
                raise DataReadinessError("investment raw artifact changed during read")
            observed_days = pd.to_datetime(part.session_date, errors="raise").dt.date
            if not set(observed_days).issubset(covered):
                raise DataReadinessError("investment raw observations escape selected segment ownership")
            day_sources.update({day: (name, digest) for day in observed_days})
            frames.append(part)
        frame = pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
        if frame.empty:
            return InvestmentObservations(frame, {}, symbol, ("raw_observations_missing",))
        if not (frame.security_id.eq(security_id).all() and frame.ticker.eq(ticker).all()
                and frame.adjustment.eq("raw").all() and frame.price_feed.eq("sip").all()
                and frame.source.eq("alpaca").all() and frame.timeframe.eq("1Day").all()):
            raise DataReadinessError("investment observation identity, adjustment or feed differs")
        days = pd.to_datetime(frame.session_date, errors="raise").dt.date
        if days.duplicated().any() or not set(days).issubset(sessions):
            raise DataReadinessError("investment observations duplicate or escape exact sessions")
        provider_clock = pd.to_datetime(frame.bar_start_utc, utc=True).dt.tz_convert("America/New_York")
        if not provider_clock.dt.date.eq(days).all() or not provider_clock.eq(provider_clock.dt.normalize()).all():
            raise DataReadinessError("investment provider daily clock differs from session")
        calendar = xcals.get_calendar("XNYS")
        schedule = calendar.schedule.reindex(pd.DatetimeIndex(days))
        frame = frame.rename(columns={"session_date": "session_date_et"}).copy()
        frame["bar_start_utc"], frame["bar_end_utc"] = schedule.open.array, schedule.close.array
        frame["available_at_utc"] = frame.ingested_at_utc
        frame = validate_outcome_observations(frame).set_index("session_date_et")
        refs = {day: EvidenceReference(reference=day_sources[day][0], artifact_sha256=day_sources[day][1],
            record_locator=f"security_id={security_id};session_date={day}",
            interpretation_policy_sha256=self.interpretation_sha256,
            retrieved_at=pd.Timestamp(captured).to_pydatetime(),
            available_at=pd.Timestamp(available).to_pydatetime())
            for day, captured, available in zip(frame.index, frame.ingested_at_utc, frame.available_at_utc, strict=True)}
        valid_days = set(frame.index[frame.outcome_observation_valid])
        missing = tuple(f"raw_observation_unavailable:{day}" for day in sessions if day not in valid_days)
        return InvestmentObservations(frame, refs, symbol, missing)
