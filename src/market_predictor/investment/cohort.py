"""Frozen monthly investment research roster; scope filters never become exclusions."""

from __future__ import annotations

from datetime import date
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort

FIRST_DECISION = date(2019, 7, 9)
LAST_DECISION = date(2024, 5, 28)
OUT_OF_SCOPE = {"Financials", "Real Estate"}


def monthly_decisions() -> tuple[pd.Timestamp, ...]:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range(FIRST_DECISION.isoformat(), LAST_DECISION.isoformat())
    frame = pd.Series(sessions, index=sessions.strftime("%Y-%m"))
    return tuple(frame.groupby(level=0, sort=True).last())


def investment_roster(cohort: SwingResearchCohort, memberships: pd.DataFrame, identities: pd.DataFrame) -> list[dict[str, Any]]:
    """Full original population accounting, with effective membership and issuer relation joins.

    Monthly membership absence is retained as a gap/status, not interpreted as data
    exclusion. Historical effective clocks are reconstruction evidence, not observed
    historical capture. No current ticker lookup or missing-sector inference occurs.
    """
    if len(cohort.original_security_ids) != 631 or len(cohort.excluded_security_ids) != 45 or not cohort.within_cap:
        raise DataReadinessError("investment roster requires frozen original631/excluded45 approved cohort")
    required = {"security_id", "ticker", "sector", "primary_benchmark", "effective_from_utc", "effective_to_utc", "available_at_utc"}
    if not required.issubset(memberships.columns):
        raise DataReadinessError("investment membership projection lacks required canonical columns")
    identity_required = {"security_id", "sec_cik", "effective_from_utc", "effective_to_utc", "available_at_utc"}
    if not identity_required.issubset(identities.columns):
        raise DataReadinessError("investment issuer projection lacks required canonical columns")
    memberships, identities = memberships.copy(), identities.copy()
    for frame in (memberships, identities):
        for column in ("effective_from_utc", "effective_to_utc", "available_at_utc"):
            frame[column] = pd.to_datetime(frame[column], utc=True, errors="raise")
        if frame["security_id"].isna().any() or frame["effective_from_utc"].isna().any() or frame["available_at_utc"].isna().any():
            raise DataReadinessError("investment canonical identity or interval clock missing")
        if (frame["effective_to_utc"].notna() & frame["effective_to_utc"].le(frame["effective_from_utc"])).any():
            raise DataReadinessError("investment canonical interval unordered")
    calendar = xcals.get_calendar("XNYS")
    rows: list[dict[str, Any]] = []
    excluded = set(cohort.excluded_security_ids)
    for session in monthly_decisions():
        decision = calendar.session_close(session)

        def active(frame: pd.DataFrame, cutoff: pd.Timestamp) -> pd.DataFrame:
            return frame.loc[
                frame.effective_from_utc.le(cutoff)
                & (frame.effective_to_utc.isna() | frame.effective_to_utc.gt(cutoff))
                & frame.available_at_utc.le(cutoff)
            ]

        member_active, identity_active = active(memberships, decision), active(identities, decision)
        for security_id in cohort.original_security_ids:
            member = member_active.loc[member_active.security_id.eq(security_id)]
            issuer = identity_active.loc[identity_active.security_id.eq(security_id)]
            if len(member) > 1 or len(issuer) > 1:
                raise DataReadinessError(f"ambiguous effective investment identity: {security_id}")
            ticker = None if member.empty else str(member.iloc[0].ticker)
            sector_value = None if member.empty else member.iloc[0].sector
            sector = None if pd.isna(sector_value) else str(sector_value)
            benchmark = None if member.empty else str(member.iloc[0].primary_benchmark)
            cik = None if issuer.empty else str(issuer.iloc[0].sec_cik).zfill(10)
            reasons: list[str] = []
            if security_id in excluded:
                reasons.append("excluded_research_cohort")
            if member.empty:
                reasons.append("membership_unknown")
            elif sector in OUT_OF_SCOPE:
                reasons.append("out_of_scope_sector")
            elif sector in {None, "", "Unknown", "nan"}:
                reasons.append("sector_unknown")
            if cik is None or not cik.isdigit():
                cik = None
                reasons.append("identity_unknown")
            rows.append(
                {
                    "decision_id": json_sha256({"security_id": security_id, "decision_session": session.date().isoformat()}),
                    "security_id": security_id,
                    "ticker": ticker,
                    "sector": sector,
                    "sector_benchmark": benchmark,
                    "decision_session": session.date().isoformat(),
                    "decision_time_utc": decision.isoformat(),
                    "cik": cik,
                    "status": reasons[0] if reasons else "eligible",
                    "qualification_reasons": reasons,
                }
            )
    return rows
