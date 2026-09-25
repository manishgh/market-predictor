"""Saved SEC filing metadata per cohort security, using EDGAR's own form names and item codes.

Counts are filing metadata, never event meaning. Rows are every saved submissions row whose
acceptance, re-read under the issuer's published clock convention, makes it available inside
the window; an issuer of unknown convention contributes counted exclusions only. Each filing
is attributed through a pinned SEC identity relation that covers, and was available by, the
filing's availability; time outside a relation is unknown identity, not zero filings. An 8-K
accepted after its report date is a late record of an event whose first public time needs
other evidence.
"""
from __future__ import annotations

import re
from collections.abc import Collection, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import exchange_calendars as xcals
import pandas as pd

from market_predictor.catalysts.sec_filings.acceptance_clock import CONVENTIONS, UNKNOWN, UTC_LABEL, corrected_acceptance
from market_predictor.catalysts.sec_filings.collection import (
    ReplayedSecIssuer,
    conservative_sec_daily_swing_availability,
    saved_filings,
    unsaved_submission_pages,
)
from market_predictor.core.errors import DataReadinessError

NEW_YORK = "America/New_York"
CURRENT_REPORTS = frozenset({"8-K", "8-K/A"})
DOCUMENT_STATES = ("primary_document_saved", "other_filing_document_saved", "saved_without_receipt", "not_saved")
SESSION_POSITIONS = ("pre_open", "intraday", "after_close", "non_session")
# What SEC timing alone says about trading between an 8-K's report date and its availability.
REPORT_TIMINGS = ("no_post_report_session_before_sec", "report_session_may_precede_sec", "post_report_sessions_precede_sec",
                  "unknown", "not_applicable")
FILING_COLUMNS = (
    "security_id", "sec_cik", "accession_number", "sec_form", "item_codes", "report_date", "filing_date",
    "accepted_at_utc", "acceptance_raw", "acceptance_clock", "archive_event", "available_at_utc", "availability_rule",
    "acceptance_session_position", "acceptance_lag_days",
    "post_report_sessions_before_availability", "report_session_opened_before_availability", "report_timing",
    "identity_policy", "issuer_cohort_securities", "primary_document",
    "primary_document_description", "filing_size_bytes", "is_xbrl", "is_inline_xbrl", "document_status",
    "saved_document_sources", "year_new_york",
)
_XNYS = xcals.get_calendar("XNYS")
_ARCHIVE_URL = re.compile(r"https://www\.sec\.gov/Archives/edgar/data/(\d{1,10})/(\d{18})/([^/?#]+)")
_CIK_ID = re.compile(r"cik:(\d{10})(?::ticker:[A-Z0-9.-]{1,16})?")


@dataclass(frozen=True)
class SavedDocument:
    """One verified saved filing document; receipts exist only for official-document collections."""

    sec_cik: str
    accession_number: str
    filename: str
    source: str
    has_receipt: bool


@dataclass(frozen=True)
class IssuerInventory:
    filings: list[dict[str, Any]]
    outside_relation: list[dict[str, Any]]
    unrequested_forms: dict[str, int]
    available_after_window: int
    unknown_clock_rows: int
    unsaved_window_start_pages: list[str]


def saved_document(url: str, source: str, *, has_receipt: bool) -> SavedDocument | None:
    """Parse an EDGAR archive URL into its filing identity; other hosts or paths are not filings."""
    match = _ARCHIVE_URL.fullmatch(url)
    if match is None:
        return None
    digits = match[2]
    return SavedDocument(match[1].zfill(10), f"{digits[:10]}-{digits[10:12]}-{digits[12:]}", match[3], source, has_receipt)


def _integer(value: object) -> int | None:
    return None if value is None or value == "" else int(str(value))


def _flag(value: object) -> bool | None:
    return None if value is None or value == "" else bool(int(str(value)))


def session_position(accepted: pd.Timestamp) -> str:
    """Where acceptance falls in the XNYS trading day of its New York date."""
    day = accepted.tz_convert(NEW_YORK).tz_localize(None).normalize()
    if not _XNYS.is_session(day):
        return "non_session"
    if accepted < _XNYS.session_open(day):
        return "pre_open"
    return "intraday" if accepted < _XNYS.session_close(day) else "after_close"


@dataclass(frozen=True)
class ReportTiming:
    """Trading between an 8-K's report date and SEC availability; a date has no clock, so some cases stay open."""

    lag_days: int | None
    post_report_sessions: int | None
    report_session_opened: bool | None
    label: str


def report_timing(report_date: str, accepted: pd.Timestamp, available: pd.Timestamp) -> ReportTiming:
    """Sessions after the report date that opened before availability, and whether the report day itself traded first."""
    try:
        reported = pd.Timestamp(report_date)
    except ValueError:
        return ReportTiming(None, None, None, "unknown")
    if not report_date or pd.isna(reported):
        return ReportTiming(None, None, None, "unknown")
    days = int((accepted.tz_convert(NEW_YORK).tz_localize(None).normalize() - reported).days)
    available_day = available.tz_convert(NEW_YORK).tz_localize(None).normalize()
    try:
        opened = bool(_XNYS.is_session(reported)) and bool(_XNYS.session_open(reported) < available)
        later = (_XNYS.sessions_in_range(reported + pd.Timedelta(days=1), available_day) if available_day > reported
                 else pd.DatetimeIndex([]))
        sessions = sum(1 for session in later if _XNYS.session_open(session) < available)
    except ValueError:  # A report date outside the exchange calendar keeps only its calendar days.
        return ReportTiming(days, None, None, "unknown")
    label = ("post_report_sessions_precede_sec" if sessions else "report_session_may_precede_sec" if opened
             else "no_post_report_session_before_sec")
    return ReportTiming(days, sessions, opened, label)


def _cohort_cik(security_id: str) -> str | None:
    match = _CIK_ID.fullmatch(security_id)
    return None if match is None else match[1]


def document_status(documents: Sequence[SavedDocument], primary_document: str) -> str:
    receipted = [document for document in documents if document.has_receipt]
    if any(document.filename == primary_document for document in receipted):
        return "primary_document_saved"
    if receipted:
        return "other_filing_document_saved"
    return "saved_without_receipt" if documents else "not_saved"


def _available(accepted: pd.Timestamp, form: str, lag_minutes: int) -> tuple[pd.Timestamp, str]:
    when, rule = conservative_sec_daily_swing_availability(accepted.to_pydatetime(), form, lag_minutes=lag_minutes)
    return pd.Timestamp(when), rule


def _either_reading_in_window(raw: str, form: str, lag_minutes: int, window: tuple[pd.Timestamp, pd.Timestamp]) -> bool:
    for convention in CONVENTIONS:
        try:
            if window[0] <= _available(corrected_acceptance(raw, convention), form, lag_minutes)[0] <= window[1]:
                return True
        except DataReadinessError:
            continue
    return False


def issuer_inventory(issuer: ReplayedSecIssuer, *, convention: str, lag_minutes: int, requested: Collection[str],
                     relations: pd.DataFrame, cohort: frozenset[str], window: tuple[pd.Timestamp, pd.Timestamp],
                     saved: Mapping[tuple[str, str], Sequence[SavedDocument]]) -> IssuerInventory:
    """Attribute one replayed issuer's saved filings available in the window to cohort securities;
    never guess identity or clock."""
    own = relations.loc[relations.sec_cik.eq(issuer.cik) & relations.security_id.isin(cohort)].sort_values(
        ["security_id", "effective_from_utc"], kind="stable").to_dict("records")
    by_id = [security for security in cohort if _cohort_cik(security) == issuer.cik]
    archived = set(issuer.events.accession_number.astype(str))
    filings: list[dict[str, Any]] = []
    outside: list[dict[str, Any]] = []
    unrequested: dict[str, set[str]] = {}
    late = unknown = 0
    for accession, (record, row) in sorted(saved_filings(issuer.history).items()):
        raw = str(row.get("acceptanceDateTime") or "")
        if convention == UNKNOWN:
            unknown += record.form in requested and _either_reading_in_window(raw, record.form, lag_minutes, window)
            continue
        accepted_at = corrected_acceptance(raw, convention)
        if convention == UTC_LABEL and accepted_at != pd.Timestamp(record.accepted_at_utc):
            raise DataReadinessError(f"SEC acceptance re-read differs from the parsed record: {accession}")
        when, rule = _available(accepted_at, record.form, lag_minutes)
        if record.form not in requested:
            if window[0] <= when <= window[1]:
                unrequested.setdefault(record.form, set()).add(accession)
            continue
        late += window[0] <= accepted_at <= window[1] < when
        if not window[0] <= when <= window[1]:
            continue
        matches = [relation for relation in own if relation["effective_from_utc"] <= when
                   and (pd.isna(relation["effective_to_utc"]) or when < relation["effective_to_utc"])
                   and relation["available_at_utc"] <= when]
        securities = [relation["security_id"] for relation in matches]
        if len(set(securities)) != len(securities):
            raise DataReadinessError(f"SEC identity relations overlap for one security: {issuer.cik}")
        year = int(when.tz_convert(NEW_YORK).year)
        form = record.form
        for security in sorted(set(by_id) - set(securities)):
            outside.append({"security_id": security, "sec_cik": issuer.cik, "sec_form": form, "year_new_york": year})
        timing = (report_timing(record.report_date, accepted_at, when) if form in CURRENT_REPORTS
                  else ReportTiming(None, None, None, "not_applicable"))
        documents = saved.get((issuer.cik, accession), ())
        items = tuple(code.strip() for code in str(row.get("items") or "").split(",") if code.strip())
        for relation in matches:
            filings.append({
                "security_id": relation["security_id"], "sec_cik": issuer.cik, "accession_number": accession,
                "sec_form": form, "report_date": record.report_date, "filing_date": record.filing_date,
                "accepted_at_utc": accepted_at, "acceptance_raw": raw, "acceptance_clock": convention,
                "archive_event": accession in archived, "available_at_utc": when, "availability_rule": rule,
                "acceptance_session_position": session_position(accepted_at), "acceptance_lag_days": timing.lag_days,
                "post_report_sessions_before_availability": timing.post_report_sessions,
                "report_session_opened_before_availability": timing.report_session_opened, "report_timing": timing.label,
                "identity_policy": relation["identity_policy"], "issuer_cohort_securities": len(matches),
                "primary_document": record.primary_document, "item_codes": ",".join(items),
                "primary_document_description": str(row.get("primaryDocDescription") or ""),
                "filing_size_bytes": _integer(row.get("size")), "is_xbrl": _flag(row.get("isXBRL")),
                "is_inline_xbrl": _flag(row.get("isInlineXBRL")),
                "document_status": document_status(documents, record.primary_document),
                "saved_document_sources": ",".join(sorted({document.source for document in documents})),
                "year_new_york": year,
            })
    # A filing available at the window start can be dated the New York day before it; a page of that day never
    # fetched makes the start incomplete for this issuer.
    start_pages = unsaved_submission_pages(issuer.history, window[0].tz_convert(NEW_YORK).date())
    return IssuerInventory(filings, outside, {form: len(values) for form, values in unrequested.items()}, late, unknown,
                           start_pages)


def filings_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    frame = pd.DataFrame(rows, columns=FILING_COLUMNS)
    for column in ("accepted_at_utc", "available_at_utc"):
        frame[column] = pd.to_datetime(frame[column], utc=True).dt.as_unit("ns")
    for column in ("acceptance_lag_days", "post_report_sessions_before_availability", "filing_size_bytes"):
        frame[column] = frame[column].astype("Int64")
    for column in ("report_session_opened_before_availability", "is_xbrl", "is_inline_xbrl"):
        frame[column] = frame[column].astype("boolean")
    frame["year_new_york"] = frame["year_new_york"].astype("int16")
    return frame.sort_values(["security_id", "available_at_utc", "accession_number"], kind="stable").reset_index(drop=True)


def _years(window: tuple[pd.Timestamp, pd.Timestamp]) -> list[tuple[int, pd.Timestamp, pd.Timestamp]]:
    spans = []
    for year in range(window[0].tz_convert(NEW_YORK).year, window[1].tz_convert(NEW_YORK).year + 1):
        start = pd.Timestamp(year=year, month=1, day=1, tz=NEW_YORK).tz_convert("UTC")
        end = pd.Timestamp(year=year + 1, month=1, day=1, tz=NEW_YORK).tz_convert("UTC")
        spans.append((year, max(start, window[0]), min(end, window[1])))
    return spans


def security_years(filings: pd.DataFrame, outside: pd.DataFrame, relations: pd.DataFrame, cohort: Sequence[str],
                   window: tuple[pd.Timestamp, pd.Timestamp]) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per security and New York year: identity coverage and document states, plus long counts by form and item."""
    spans_by_security = {str(key): frame for key, frame in relations.loc[relations.security_id.isin(cohort)].groupby("security_id")}
    filed = filings.groupby(["security_id", "year_new_york"]).size().to_dict()
    states = filings.groupby(["security_id", "year_new_york", "document_status"]).size().to_dict()
    outside_counts = outside.groupby(["security_id", "year_new_york"]).size().to_dict()
    rows = []
    for security in cohort:
        spans = spans_by_security.get(security, relations.iloc[:0])
        for year, start, end in _years(window):
            lower = spans.effective_from_utc.clip(lower=start)
            upper = spans.effective_to_utc.fillna(end).clip(upper=end)
            covered = float(((upper - lower).dt.total_seconds() / 86_400).clip(lower=0).sum())
            total = (end - start).total_seconds() / 86_400
            rows.append({"security_id": security, "year_new_york": year, "window_days": total,
                         "sec_identity_days": covered, "unknown_identity_days": max(0.0, total - covered),
                         "filings": int(filed.get((security, year), 0)),
                         "cik_identity_outside_relation_filings": int(outside_counts.get((security, year), 0)),
                         **{f"documents_{state}": int(states.get((security, year, state), 0)) for state in DOCUMENT_STATES}})
    exploded = filings.assign(item_code=filings.item_codes.where(filings.sec_form.isin(CURRENT_REPORTS), "")
                              .str.split(",")).explode("item_code")
    exploded["item_code"] = exploded.item_code.fillna("").replace("", pd.NA).fillna(
        exploded.sec_form.map(lambda form: "none" if form in CURRENT_REPORTS else "not_applicable"))
    counts = (exploded.groupby(["security_id", "year_new_york", "sec_form", "item_code", "report_timing"])
              .accession_number.nunique().rename("filings").reset_index())
    return pd.DataFrame(rows), counts.sort_values(["security_id", "year_new_york", "sec_form", "item_code",
                                                    "report_timing"], kind="stable").reset_index(drop=True)
