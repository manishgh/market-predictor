"""EDGAR acceptance clocks per issuer.

The submissions API's ``acceptanceDateTime`` is UTC for most issuers, but for some it is New
York wall-clock time labelled UTC, uniformly across their filings. EDGAR's filing detail page
shows acceptance on EDGAR's own New York clock, so an issuer's convention is read from its
pages, never inferred, and its raw values are re-read under that convention. Counts of
filings outside EDGAR's weekday hours only corroborate the pages: genuine out-of-hours
acceptances exist, so hours are comparative evidence, never a rule.
"""
from __future__ import annotations

import re
from collections.abc import Iterable
from datetime import date, time
from typing import Final

import pandas as pd

from market_predictor.core.errors import DataReadinessError

UTC_LABEL: Final = "utc"
NEW_YORK_LABELED_UTC: Final = "new_york_wall_clock_labeled_utc"
UNKNOWN: Final = "unknown"
CONVENTIONS: Final = (UTC_LABEL, NEW_YORK_LABELED_UTC)
FORM_GROUPS: Final = ("current", "periodic", "ownership", "other")
EDGAR_OPENS: Final = time(6, 0)
EDGAR_CLOSES: Final = time(22, 0)
_NEW_YORK = "America/New_York"
_RAW = re.compile(r"(\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2})\.000Z")
_GROUPS = {"8-K": "current", "6-K": "current", "10-K": "periodic", "10-Q": "periodic", "20-F": "periodic",
           "3": "ownership", "4": "ownership", "5": "ownership"}


def _require(condition: bool, message: str) -> None:
    if not condition:
        raise DataReadinessError(message)


def form_group(form: str) -> str:
    """Current reports, periodic reports, ownership forms or other; amendments join their base form."""
    return _GROUPS.get(form.upper().removesuffix("/A"), "other")


def wall_clock(raw: str) -> pd.Timestamp:
    """The clock reading of a raw ``acceptanceDateTime``, before any convention names its zone."""
    match = _RAW.fullmatch(raw)
    _require(match is not None, f"SEC acceptanceDateTime has an unexpected form: {raw!r}")
    assert match is not None
    return pd.Timestamp(match[1])


def corrected_acceptance(raw: str, convention: str) -> pd.Timestamp:
    """The UTC acceptance instant under a decided convention; an impossible New York reading raises."""
    wall = wall_clock(raw)
    if convention == UTC_LABEL:
        return wall.tz_localize("UTC")
    _require(convention == NEW_YORK_LABELED_UTC, f"SEC acceptance clock convention is not decided: {convention}")
    try:
        return wall.tz_localize(_NEW_YORK, ambiguous="raise", nonexistent="raise").tz_convert("UTC")
    except ValueError as error:  # EDGAR is closed across both daylight-saving switches.
        raise DataReadinessError(f"SEC acceptance is not a New York instant: {raw}") from error


def page_convention(raw: str, page_accepted_utc: pd.Timestamp) -> str | None:
    """The convention under which the raw value equals EDGAR's own page acceptance, or None."""
    if corrected_acceptance(raw, UTC_LABEL) == page_accepted_utc:
        return UTC_LABEL
    try:
        return NEW_YORK_LABELED_UTC if corrected_acceptance(raw, NEW_YORK_LABELED_UTC) == page_accepted_utc else None
    except DataReadinessError:
        return None


def clock_sample(filings: pd.DataFrame) -> pd.DataFrame:
    """Each issuer's first and last filing in every form group, ordered by raw acceptance then accession."""
    ordered = filings.assign(form_group=filings.sec_form.map(form_group)).sort_values(
        ["sec_cik", "form_group", "acceptance_raw", "accession_number"], kind="stable")
    grouped = ordered.groupby(["sec_cik", "form_group"], sort=True)
    sample = pd.concat([grouped.head(1), grouped.tail(1)]).drop_duplicates(["sec_cik", "accession_number"])
    return sample.sort_values(["sec_cik", "form_group", "acceptance_raw", "accession_number"], kind="stable").loc[
        :, ["sec_cik", "form_group", "accession_number", "sec_form", "acceptance_raw"]].reset_index(drop=True)


def _seconds(moment: time) -> int:
    return moment.hour * 3600 + moment.minute * 60 + moment.second


def _outside_hours(local: pd.Series) -> pd.Series:
    seconds = local.dt.hour * 3600 + local.dt.minute * 60 + local.dt.second
    return (local.isna() | local.dt.dayofweek.ge(5) | seconds.lt(_seconds(EDGAR_OPENS))
            | seconds.gt(_seconds(EDGAR_CLOSES)))


def in_data_counts(filings: pd.DataFrame, before: date) -> pd.DataFrame:
    """Per issuer and form group, filings dated before `before` that fall outside EDGAR's weekday hours
    under each reading; an impossible New York reading counts as outside."""
    early = filings.loc[filings.filing_date.lt(before.isoformat())]
    wall = pd.to_datetime(early.acceptance_raw.map(lambda raw: wall_clock(raw).isoformat()))
    as_utc = wall.dt.tz_localize("UTC").dt.tz_convert(_NEW_YORK)
    as_new_york = wall.dt.tz_localize(_NEW_YORK, ambiguous="NaT", nonexistent="NaT")
    counts = early.assign(form_group=early.sec_form.map(form_group), outside_as_utc=_outside_hours(as_utc),
                          outside_as_new_york=_outside_hours(as_new_york))
    return (counts.groupby(["sec_cik", "form_group"], sort=True)
            .agg(filings=("accession_number", "size"), outside_as_utc=("outside_as_utc", "sum"),
                 outside_as_new_york=("outside_as_new_york", "sum")).reset_index())


def decide(sec_cik: str, groups: Iterable[str], pages: pd.DataFrame, counts: pd.DataFrame) -> tuple[str, str]:
    """One issuer's convention and why: pages decide, every form group needs one, in-data counts must not contradict.

    `pages` holds this issuer's page comparisons (`form_group`, `page_state`, `page_convention`);
    `counts` its in-data rows. A page matching neither reading, disagreeing pages or contrary
    counts mean the per-issuer model is wrong and raise; a group without a readable page is unknown.
    """
    compared = pages.loc[pages.page_state.eq("archived")]
    _require(bool(compared.page_convention.notna().all()),
             f"EDGAR page acceptance matches neither reading for issuer {sec_cik}")
    conventions = set(compared.page_convention)
    _require(len(conventions) <= 1, f"EDGAR pages disagree on the acceptance clock of issuer {sec_cik}")
    unpaged = sorted(set(groups) - set(compared.form_group))
    if unpaged:
        return UNKNOWN, f"no readable EDGAR page for form groups {','.join(unpaged)}"
    convention = conventions.pop()
    own, other = (("outside_as_utc", "outside_as_new_york") if convention == UTC_LABEL
                  else ("outside_as_new_york", "outside_as_utc"))
    contrary = counts.loc[counts[own].gt(counts[other])]
    _require(contrary.empty, f"in-data EDGAR hours contradict the pages of issuer {sec_cik}: "
                             f"{','.join(contrary.form_group)}")
    return convention, "every form group has a matching EDGAR page and no contrary in-data count"
