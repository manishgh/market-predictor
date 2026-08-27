from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime
from pathlib import Path
from typing import Any
from zoneinfo import ZoneInfo

import pandas as pd
from bs4 import BeautifulSoup

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.symbols import normalized_ticker

SECTOR_BENCHMARKS = {
    "Communication Services": "XLC",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Financials": "XLF",
    "Health Care": "XLV",
    "Industrials": "XLI",
    "Information Technology": "XLK",
    "Materials": "XLB",
    "Real Estate": "XLRE",
    "Utilities": "XLU",
}
_TABLE_COLUMNS = ("Effective Date", "Index Name", "Action", "Company Name", "Ticker", "GICS Sector")
_LEGACY_TABLE_TITLE = re.compile(r"^S&P\s+500\s+INDEX\s*[-–—]\s*(.+)$", re.IGNORECASE)
_LEGACY_ACTIONS = {"ADDED": "addition", "DELETED": "deletion"}
_LEGACY_INVALID_MIDCAP_500_TITLE = re.compile(
    r"^S&P\s+MIDCAP\s+500\s+INDEX\s*[-\u2013\u2014\u0426]\s*(.+)$",
    re.IGNORECASE,
)
_ANY_LEGACY_TABLE_TITLE = re.compile(r"^S&P\s+.+?\s+INDEX\s*[-\u2013\u2014]", re.IGNORECASE)
_EXCHANGE_TICKERS = (
    r"\((?:NYSE|NASD|NASDAQ|NYSE\s+American|OTC(?:QX|QB)?)\s*:\s*"
    r"(?P<exchange_tickers>[A-Z0-9.-]+(?:\s*[,;/]\s*[A-Z0-9.-]+)*)\)"
)
_TWITTER_2018_SOURCE_URL = (
    "https://press.spglobal.com/2018-06-04-Netflix-Set-to-Join-S-P-100-Twitter-to-Join-S-P-500-REGENXBIO-to-Join-S-P-SmallCap-600"
)
_TWITTER_2018_EFFECTIVE_DATE = "JUNE 7 2018"
_TWITTER_2018_TABLE_ROWS = (
    ("ADDED", "TWITTER", "INFORMATION TECHNOLOGY"),
    ("DELETED", "MONSANTO", "MATERIALS"),
)


@dataclass(frozen=True)
class IndexChangeSource:
    source_url: str
    source_published_date: date
    source_sha256: str

    def to_record(self) -> dict[str, str]:
        return {
            "source_url": self.source_url,
            "source_published_date": self.source_published_date.isoformat(),
            "source_sha256": self.source_sha256,
        }


@dataclass(frozen=True)
class IndexChange:
    effective_at_utc: datetime
    action: str
    ticker: str
    company: str
    sector: str
    source_url: str
    source_published_date: date
    source_sha256: str
    supporting_sources: tuple[IndexChangeSource, ...] = ()

    def source_evidence(self) -> tuple[IndexChangeSource, ...]:
        if self.supporting_sources:
            return self.supporting_sources
        return (
            IndexChangeSource(
                source_url=self.source_url,
                source_published_date=self.source_published_date,
                source_sha256=self.source_sha256,
            ),
        )

    def to_record(self) -> dict[str, Any]:
        return {
            "effective_at_utc": self.effective_at_utc.isoformat(),
            "action": self.action,
            "ticker": self.ticker,
            "company": self.company,
            "sector": self.sector,
            "source_url": self.source_url,
            "source_published_date": self.source_published_date.isoformat(),
            "source_sha256": self.source_sha256,
            "supporting_sources": [source.to_record() for source in self.source_evidence()],
        }


@dataclass(frozen=True)
class VerifiedIndexChanges:
    changes: tuple[IndexChange, ...]
    authority_sha256: str
    event_set_sha256: str


@dataclass(frozen=True)
class SymbolChange:
    effective_at_utc: datetime
    old_ticker: str
    new_ticker: str
    source_id: str
    source_url: str
    transition_type: str
    identity_continuity: bool
    old_security_id: str
    new_security_id: str

    def to_record(self) -> dict[str, object]:
        return {
            "effective_at_utc": self.effective_at_utc.isoformat(),
            "old_ticker": self.old_ticker,
            "new_ticker": self.new_ticker,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "transition_type": self.transition_type,
            "identity_continuity": self.identity_continuity,
            "old_security_id": self.old_security_id,
            "new_security_id": self.new_security_id,
        }


_REVIEWED_TRANSITION_COLUMNS = {
    "id",
    "effective_date",
    "old_symbol",
    "new_symbol",
    "transition_type",
    "identity_continuity",
    "membership_continuity",
    "old_security_id",
    "new_security_id",
    "source_url",
    "evidence_summary",
    "reviewed_by",
    "reviewed_at_utc",
}


def load_reviewed_security_transitions(path: Path) -> pd.DataFrame:
    """Load human-reviewed transition evidence that may carry index membership."""

    frame = pd.read_csv(path, dtype=str, keep_default_na=False)
    missing = sorted(_REVIEWED_TRANSITION_COLUMNS.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"Reviewed security-transition ledger is missing columns: {missing}")
    if frame.empty:
        raise DataReadinessError("Reviewed security-transition ledger must not be empty")
    frame = frame.copy()
    for column in ("id", "old_symbol", "new_symbol", "transition_type", "source_url", "evidence_summary", "reviewed_by"):
        if frame[column].astype(str).str.strip().eq("").any():
            raise DataReadinessError(f"Reviewed security-transition ledger has blank {column} values")
    for column in ("identity_continuity", "membership_continuity"):
        frame[column] = frame[column].map(lambda value, field=column: _transition_boolean(value, field=field))
    parsed_dates = pd.to_datetime(frame["effective_date"], errors="coerce")
    if parsed_dates.isna().any():
        raise DataReadinessError("Reviewed security-transition ledger has invalid effective_date values")
    frame["effective_date"] = parsed_dates.dt.date.astype(str)
    reviewed_at = pd.to_datetime(frame["reviewed_at_utc"], utc=True, errors="coerce")
    if reviewed_at.isna().any():
        raise DataReadinessError("Reviewed security-transition ledger has invalid reviewed_at_utc values")
    if not frame["source_url"].str.startswith("https://").all():
        raise DataReadinessError("Reviewed security-transition source_url values must use HTTPS")
    duplicate_keys = frame.duplicated(["effective_date", "old_symbol", "new_symbol"], keep=False)
    if duplicate_keys.any():
        raise DataReadinessError("Reviewed security-transition ledger has duplicate effective-date symbol pairs")
    return frame.reset_index(drop=True)


def merge_security_transition_evidence(provider: pd.DataFrame, reviewed: pd.DataFrame) -> pd.DataFrame:
    """Overlay reviewed decisions on provider candidates without approving provider mergers."""

    provider_frame = provider.copy()
    reviewed_frame = reviewed.copy()
    for frame, label in ((provider_frame, "provider"), (reviewed_frame, "reviewed")):
        required = {"id", "effective_date", "old_symbol", "new_symbol", "transition_type"}
        missing = sorted(required.difference(frame.columns))
        if missing:
            raise DataReadinessError(f"{label.capitalize()} security-transition data is missing columns: {missing}")
        frame["effective_date"] = pd.to_datetime(frame["effective_date"], errors="coerce").dt.date.astype(str)
        if frame["effective_date"].eq("NaT").any():
            raise DataReadinessError(f"{label.capitalize()} security-transition data has invalid effective dates")
        frame["old_symbol"] = frame["old_symbol"].astype(str).map(normalized_ticker)
        frame["new_symbol"] = frame["new_symbol"].astype(str).map(normalized_ticker)
    review_pairs = set(reviewed_frame[["old_symbol", "new_symbol"]].itertuples(index=False, name=None))
    provider_pairs = provider_frame[["old_symbol", "new_symbol"]].apply(tuple, axis=1)
    provider_frame = provider_frame.loc[~provider_pairs.isin(review_pairs)].copy()
    combined = pd.concat([provider_frame, reviewed_frame], ignore_index=True, sort=False)
    return combined.sort_values(
        ["effective_date", "old_symbol", "new_symbol", "transition_type", "id"],
        kind="stable",
    ).reset_index(drop=True)


def symbol_changes_from_transitions(frame: pd.DataFrame) -> list[SymbolChange]:
    required = {"id", "old_symbol", "new_symbol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"Security-transition data is missing columns: {missing}")
    changes: list[SymbolChange] = []
    transition_types = (
        frame["transition_type"].fillna("name_change").astype(str)
        if "transition_type" in frame.columns
        else pd.Series("name_change", index=frame.index)
    )
    membership_flags = (
        frame["membership_continuity"].fillna(False).astype(bool)
        if "membership_continuity" in frame.columns
        else transition_types.eq("name_change")
    )
    predecessor_targets = set(
        frame.loc[transition_types.ne("name_change") & membership_flags, "new_symbol"].dropna().astype(str).map(normalized_ticker)
    )
    for record in frame.to_dict(orient="records"):
        old_ticker = normalized_ticker(str(record["old_symbol"]))
        new_ticker = normalized_ticker(str(record["new_symbol"]))
        transition_type = str(record.get("transition_type") or "name_change").strip().lower()
        raw_membership_continuity = record.get("membership_continuity")
        membership_continuity = _transition_boolean(
            raw_membership_continuity,
            field="membership_continuity",
            default=transition_type == "name_change",
        )
        if not membership_continuity:
            continue
        if transition_type == "name_change" and _is_temporary_security_symbol(old_ticker) and old_ticker not in predecessor_targets:
            continue
        effective_value = record.get("effective_date") or record.get("process_date")
        if effective_value is None or pd.isna(effective_value):
            raise DataReadinessError(f"Alpaca transition {record['id']} has no effective date")
        effective_date = pd.Timestamp(effective_value).date()
        effective_at = datetime.combine(
            effective_date,
            datetime.min.time(),
            tzinfo=ZoneInfo("America/New_York"),
        ).astimezone(UTC)
        source_id = str(record["id"])
        old_cusip = _optional_transition_text(record.get("old_cusip")).upper()
        new_cusip = _optional_transition_text(record.get("new_cusip")).upper()
        raw_continuity = record.get("identity_continuity")
        identity_continuity = _transition_boolean(
            raw_continuity,
            field="identity_continuity",
            default=transition_type == "name_change" and (not old_cusip or old_cusip == new_cusip),
        )
        explicit_old_security_id = _optional_transition_text(record.get("old_security_id"))
        explicit_new_security_id = _optional_transition_text(record.get("new_security_id"))
        if identity_continuity:
            if explicit_old_security_id and explicit_new_security_id and explicit_old_security_id != explicit_new_security_id:
                raise DataReadinessError(f"Identity-continuous transition {source_id} has different security IDs")
            shared_identity = (
                explicit_old_security_id or explicit_new_security_id or (f"cusip:{old_cusip}" if old_cusip else f"transition:{source_id}")
            )
            old_security_id = shared_identity
            new_security_id = shared_identity
        else:
            old_security_id = explicit_old_security_id or (f"cusip:{old_cusip}" if old_cusip else f"transition:{source_id}:old")
            new_security_id = explicit_new_security_id or (f"cusip:{new_cusip}" if new_cusip else f"transition:{source_id}:new")
        source_url = _optional_transition_text(record.get("source_url"))
        if not source_url:
            source_url = f"https://data.alpaca.markets/v1/corporate-actions?ids={source_id}"
        changes.append(
            SymbolChange(
                effective_at_utc=effective_at,
                old_ticker=old_ticker,
                new_ticker=new_ticker,
                source_id=source_id,
                source_url=source_url,
                transition_type=transition_type,
                identity_continuity=identity_continuity,
                old_security_id=old_security_id,
                new_security_id=new_security_id,
            )
        )
    return sorted(changes, key=lambda item: (item.effective_at_utc, item.old_ticker, item.new_ticker))


def _transition_boolean(value: object, *, field: str, default: bool | None = None) -> bool:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        if default is None:
            raise DataReadinessError(f"Security-transition {field} must be true or false")
        return default
    normalized = str(value).strip().lower()
    if normalized in {"true", "1"}:
        return True
    if normalized in {"false", "0"}:
        return False
    raise DataReadinessError(f"Security-transition {field} must be true or false, got {value!r}")


def _optional_transition_text(value: object) -> str:
    if value is None or (not isinstance(value, (list, tuple, dict)) and pd.isna(value)):
        return ""
    return str(value).strip()


def _is_temporary_security_symbol(ticker: str) -> bool:
    return ticker.endswith((".WI", ".PR", ".WS", "V"))


def parse_sp500_changes(
    html: str,
    *,
    source_url: str,
    published_date: date,
    source_sha256: str | None = None,
    allow_verified_no_membership_event: bool = False,
    source_title: str = "",
) -> list[IndexChange]:
    """Parse exact S&P 500 addition/deletion rows from one official release."""
    digest = source_sha256 or hashlib.sha256(html.encode("utf-8")).hexdigest()
    if not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise DataReadinessError("Official announcement source SHA-256 is invalid")
    soup = BeautifulSoup(html, "html.parser")
    body = soup.select_one(".wd_news_body") or soup
    body_text = body.get_text(" ", strip=True)
    prose_blocks = [paragraph.get_text(" ", strip=True) for paragraph in body.find_all("p")]
    changes: list[IndexChange] = []
    deferred_rows = 0
    for table in soup.find_all("table"):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        if not rows:
            continue
        header = tuple(rows[0])
        if not set(_TABLE_COLUMNS).issubset(header):
            continue
        positions = {name: header.index(name) for name in _TABLE_COLUMNS}
        last_effective_text: str | None = None
        for row in rows[1:]:
            if len(row) < len(header):
                if not any(cell.strip() for cell in row):
                    continue
                if len(row) == 1 and row[0].lstrip().startswith("*"):
                    continue
                raise DataReadinessError(f"Malformed S&P 500 structured row in {source_url}: {row}")
            row_effective_text = row[positions["Effective Date"]].strip()
            if row_effective_text:
                last_effective_text = row_effective_text
            index_name = row[positions["Index Name"]].replace("®", "").strip()
            action = row[positions["Action"]].strip().lower()
            if index_name != "S&P 500" or action not in {"addition", "deletion"}:
                continue
            if last_effective_text is None:
                raise DataReadinessError(f"S&P 500 row has no effective date context in {source_url}: {row}")
            effective_text = last_effective_text
            if effective_text.upper() == "TBA":
                deferred_rows += 1
                continue
            effective_date = pd.Timestamp(effective_text).date()
            effective_at = datetime.combine(effective_date, datetime.min.time(), tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
            for ticker in _index_tickers(row[positions["Ticker"]]):
                changes.append(
                    IndexChange(
                        effective_at_utc=effective_at,
                        action=action,
                        ticker=ticker,
                        company=row[positions["Company Name"]].strip(),
                        sector=row[positions["GICS Sector"]].strip(),
                        source_url=source_url,
                        source_published_date=published_date,
                        source_sha256=digest,
                    )
                )
    if not changes:
        changes.extend(
            _parse_legacy_sp500_tables(
                soup,
                source_url=source_url,
                published_date=published_date,
                source_sha256=digest,
            )
        )
    if not changes:
        changes.extend(
            _parse_prose_sp500_changes(
                body_text,
                source_url=source_url,
                published_date=published_date,
                source_sha256=digest,
            )
        )
    unique: dict[tuple[datetime, str, str], IndexChange] = {}
    for item in changes:
        key = (item.effective_at_utc, item.action, item.ticker)
        existing = unique.get(key)
        if existing is not None and (existing.company != item.company or existing.sector != item.sector):
            raise DataReadinessError(f"Official announcement has conflicting duplicate S&P 500 rows: {source_url} {key}")
        unique[key] = item
    parsed = sorted(unique.values(), key=lambda item: (item.effective_at_utc, item.action, item.ticker))
    no_event_trigger = deferred_rows > 0 or _is_explicitly_deferred_sp500_continuity(body_text)
    if not parsed and no_event_trigger and _all_sp500_prose_assertions_are_deferred(prose_blocks):
        return []
    if not parsed and allow_verified_no_membership_event and not _has_sp500_membership_assertion([source_title, body_text, *prose_blocks]):
        return []
    if not parsed:
        raise DataReadinessError(f"Official announcement contains no structured S&P 500 change rows: {source_url}")
    return parsed


def _index_tickers(raw: str) -> tuple[str, ...]:
    values = tuple(normalized_ticker(value) for value in raw.split("/") if value.strip())
    if not values:
        raise DataReadinessError(f"S&P 500 change row contains no ticker: {raw!r}")
    return values


def _parse_legacy_sp500_tables(
    soup: BeautifulSoup,
    *,
    source_url: str,
    published_date: date,
    source_sha256: str,
) -> list[IndexChange]:
    body = soup.select_one(".wd_news_body") or soup
    body_text = body.get_text(" ", strip=True)
    parsed: list[IndexChange] = []
    for table in body.find_all("table"):
        rows = [[cell.get_text(" ", strip=True) for cell in row.find_all(["th", "td"])] for row in table.find_all("tr")]
        if len(rows) < 3:
            continue
        effective_date: date | None = None
        company_position: int | None = None
        sector_position: int | None = None
        current_action: str | None = None
        for row in rows:
            if not row or not any(value.strip() for value in row):
                continue
            first_cell = row[0].strip()
            title_match = _LEGACY_TABLE_TITLE.match(first_cell)
            if title_match is None:
                invalid_title_match = _LEGACY_INVALID_MIDCAP_500_TITLE.match(first_cell)
                if invalid_title_match is not None and _is_verified_twitter_2018_malformed_table(
                    rows,
                    body_text=body_text,
                    source_url=source_url,
                    effective_text=invalid_title_match.group(1),
                ):
                    title_match = invalid_title_match
            if title_match is not None:
                effective_text = title_match.group(1).strip()
                effective_date = None
                company_position = None
                sector_position = None
                current_action = None
                if effective_text.upper() == "TBA":
                    continue
                try:
                    effective_date = pd.Timestamp(effective_text).date()
                except (TypeError, ValueError) as exc:
                    raise DataReadinessError(
                        f"Legacy S&P 500 announcement has an invalid effective date {effective_text!r}: {source_url}"
                    ) from exc
                continue
            if _ANY_LEGACY_TABLE_TITLE.match(first_cell):
                effective_date = None
                company_position = None
                sector_position = None
                current_action = None
                continue
            if effective_date is None:
                continue
            normalized_row = [value.strip().upper() for value in row]
            if "COMPANY" in normalized_row and "GICS ECONOMIC SECTOR" in normalized_row:
                company_position = _legacy_column_position(normalized_row, "COMPANY")
                sector_position = _legacy_column_position(normalized_row, "GICS ECONOMIC SECTOR")
                continue
            if company_position is None or sector_position is None:
                continue
            explicit_action = _LEGACY_ACTIONS.get(first_cell.upper())
            if explicit_action is not None:
                current_action = explicit_action
            if current_action is None:
                continue
            if len(row) == 1 and first_cell.startswith("*"):
                continue
            if company_position >= len(row) or sector_position >= len(row):
                raise DataReadinessError(f"Legacy S&P 500 row is incomplete in {source_url}: {row}")
            company = row[company_position].strip()
            sector = row[sector_position].strip()
            if not company:
                continue
            tickers = _legacy_company_tickers(
                body_text,
                company,
                action=current_action,
                source_url=source_url,
            )
            effective_at = datetime.combine(
                effective_date,
                datetime.min.time(),
                tzinfo=ZoneInfo("America/New_York"),
            ).astimezone(UTC)
            for ticker in tickers:
                parsed.append(
                    IndexChange(
                        effective_at_utc=effective_at,
                        action=current_action,
                        ticker=ticker,
                        company=company,
                        sector=sector,
                        source_url=source_url,
                        source_published_date=published_date,
                        source_sha256=source_sha256,
                    )
                )
    return parsed


def _legacy_column_position(header: list[str], name: str) -> int:
    if name not in header:
        raise DataReadinessError(f"Legacy S&P 500 table is missing {name!r}")
    return header.index(name)


def _legacy_company_tickers(
    body_text: str,
    company: str,
    *,
    action: str,
    source_url: str,
) -> tuple[str, ...]:
    company_without_qualifier = re.sub(
        r"\s*\([^)]*\)\s*",
        " ",
        company,
    ).strip()
    company_tokens = re.findall(
        r"[A-Z0-9]+",
        company_without_qualifier,
        flags=re.IGNORECASE,
    )
    if not company_tokens:
        raise DataReadinessError(f"Legacy S&P 500 company name is empty in {source_url}")
    flexible_company = r"[\W_]*".join(re.escape(token) for token in company_tokens)
    bounded_extension = (
        r"(?:\s+(?:&\s+)?(?:Associates|Bank|Holding|Holdings|Group|Global|"
        r"Worldwide|Technologies|Systems))?"
    )
    corporate_suffix = r"(?:\s*,?\s*(?:Inc(?:orporated)?|Corp(?:oration)?|Co(?:mpany)?|Ltd|PLC|LLC)\.?)*"
    pattern = re.compile(
        rf"\b{flexible_company}\b{bounded_extension}{corporate_suffix}"
        rf"\s*[.,]?\s*{_EXCHANGE_TICKERS}",
        re.IGNORECASE,
    )
    candidates: list[tuple[int, tuple[str, ...]]] = []
    for match in pattern.finditer(body_text):
        before = body_text[max(0, match.start() - 100) : match.start()].casefold()
        after = body_text[match.end() : match.end() + 100].casefold()
        if action == "addition":
            score = int(any(phrase in after for phrase in ("will replace", "will be added", "will join")))
        else:
            score = int("replace " in before or "replacing " in before or "will be removed" in after)
        candidates.append((score, _normalized_ticker_group(match.group("exchange_tickers"))))

    rename_pattern = re.compile(
        rf"(?:renamed|called)\s+{flexible_company}\b[^.]*?"
        r"trade\s+under\s+(?:the\s+)?symbol\s+([A-Z0-9.-]+)",
        re.IGNORECASE,
    )
    candidates.extend((2, _normalized_ticker_group(match.group(1))) for match in rename_pattern.finditer(body_text))
    if candidates:
        highest_score = max(score for score, _ in candidates)
        ticker_groups = {tickers for score, tickers in candidates if score == highest_score}
        if len(ticker_groups) == 1:
            return ticker_groups.pop()
    raise DataReadinessError(f"Could not bind legacy S&P 500 company {company!r} to a ticker in {source_url}")


def _normalized_ticker_group(raw: str) -> tuple[str, ...]:
    tickers = tuple(normalized_ticker(value.rstrip(".")) for value in re.split(r"\s*[,;/]\s*", raw) if value.strip())
    if not tickers:
        raise DataReadinessError(f"Official ticker group is empty: {raw!r}")
    return tickers


def _invalid_midcap_500_table_is_sp500(
    rows: list[list[str]],
    body_text: str,
) -> bool:
    company_position: int | None = None
    companies: list[str] = []
    for row in rows:
        normalized_row = [value.strip().upper() for value in row]
        if "COMPANY" in normalized_row:
            company_position = normalized_row.index("COMPANY")
            continue
        if company_position is None or company_position >= len(row):
            continue
        if row[0].strip().upper() in _LEGACY_ACTIONS:
            company = row[company_position].strip()
            if company:
                companies.append(company)
    if len(companies) < 2:
        return False
    normalized_companies = [_normalized_words(company) for company in companies]
    normalized_body = _normalized_words(body_text)
    for match in re.finditer(r"\bS P 500\b", normalized_body):
        corroboration_window = normalized_body[max(0, match.start() - 400) : match.end() + 100]
        if "REPLACE" in corroboration_window and all(company in corroboration_window for company in normalized_companies):
            return True
    return False


def _is_verified_twitter_2018_malformed_table(
    rows: list[list[str]],
    *,
    body_text: str,
    source_url: str,
    effective_text: str,
) -> bool:
    if source_url != _TWITTER_2018_SOURCE_URL:
        return False
    substantive_rows = [row for row in rows if any(_normalized_words(value) for value in row)]
    if len(substantive_rows) != 4 or not substantive_rows[0]:
        return False
    title_match = _LEGACY_INVALID_MIDCAP_500_TITLE.match(substantive_rows[0][0].strip())
    if (
        title_match is None
        or _normalized_words(title_match.group(1)) != _TWITTER_2018_EFFECTIVE_DATE
        or _normalized_words(effective_text) != _TWITTER_2018_EFFECTIVE_DATE
    ):
        return False
    header = [_normalized_words(value) for value in substantive_rows[1]]
    if header.count("COMPANY") != 1 or header.count("GICS ECONOMIC SECTOR") != 1:
        return False
    company_position = header.index("COMPANY")
    sector_position = header.index("GICS ECONOMIC SECTOR")
    observed_rows: list[tuple[str, str, str]] = []
    for row in substantive_rows[2:]:
        if not row or _normalized_words(row[0]) not in _LEGACY_ACTIONS or company_position >= len(row) or sector_position >= len(row):
            return False
        observed_rows.append(
            (
                _normalized_words(row[0]),
                _normalized_words(row[company_position]),
                _normalized_words(row[sector_position]),
            )
        )
    return tuple(observed_rows) == _TWITTER_2018_TABLE_ROWS and _invalid_midcap_500_table_is_sp500(
        rows,
        body_text,
    )


def _normalized_words(value: str) -> str:
    return " ".join(re.findall(r"[A-Z0-9]+", value.upper()))


def _is_explicitly_deferred_sp500_continuity(body_text: str) -> bool:
    normalized = " ".join(body_text.split()).casefold()
    return "date to be announced" in normalized and "continue to be included in the s&p 500" in normalized


def _all_sp500_prose_assertions_are_deferred(
    prose_blocks: list[str],
) -> bool:
    assertions: list[tuple[str, str]] = []
    action_phrases = (
        "will add",
        "will be added",
        "will join",
        "will move to",
        "will be removed",
        "will be deleted",
        "will leave",
        "will replace",
        "will be replaced",
        "replacing",
        "will continue to be included",
        "will remain",
    )
    for block in prose_blocks:
        normalized = " ".join(block.split()).casefold()
        for clause in re.split(r"(?<=[.!?;])\s+", normalized):
            if "s&p 500" in clause and any(phrase in clause for phrase in action_phrases):
                assertions.append((block.casefold(), clause))
    if not assertions:
        return True
    return all(
        (
            "date to be announced" in clause
            or " tba" in clause
            or "to be named later" in clause
            or "to be named in a separate press release" in clause
            or ("date to be announced" in block and not _contains_independent_effective_timing(block))
        )
        for block, clause in assertions
    )


def _has_sp500_membership_assertion(prose_blocks: list[str]) -> bool:
    action = re.compile(
        r"\b(?:add|added|addition|change|changes|delete|deleted|deletion|join|joins|"
        r"leave|leaves|move|moves|remove|removed|removal|replace|replaced|replaces|"
        r"replacing)\b"
    )
    return any(
        "s&p 500" in normalized and action.search(normalized) is not None
        for normalized in (" ".join(block.split()).casefold() for block in prose_blocks)
    )


def _contains_independent_effective_timing(value: str) -> bool:
    months = "january|february|march|april|may|june|july|august|september|october|november|december"
    explicit_date = re.search(rf"\b(?:{months})\s+\d{{1,2}},?\s+\d{{4}}\b", value)
    relative_effective_time = re.search(r"\beffective\b(?!\s+date\b)", value)
    return explicit_date is not None or relative_effective_time is not None


def _parse_prose_sp500_changes(
    body_text: str,
    *,
    source_url: str,
    published_date: date,
    source_sha256: str,
) -> list[IndexChange]:
    company = r"(?P<company>[A-Z][A-Za-z0-9&'., -]{1,100}?)"
    effective_date = (
        r"effective\s+prior\s+to\s+the\s+open\s+of\s+trading\s+on\s+"
        r"(?:Monday|Tuesday|Wednesday|Thursday|Friday),?\s+"
        r"(?P<effective_date>[A-Z][a-z]+\s+\d{1,2},\s+\d{4})"
    )
    patterns = (
        re.compile(
            rf"will\s+add\s+{company}\s+{_EXCHANGE_TICKERS}\s+to\s+the\s+"
            rf"S&P\s+500\b.{{0,300}}?{effective_date}",
            re.IGNORECASE,
        ),
        re.compile(
            rf"{company}\s+{_EXCHANGE_TICKERS}\s+will\s+be\s+added\s+to\s+the\s+"
            rf"S&P\s+500\b.{{0,300}}?{effective_date}",
            re.IGNORECASE,
        ),
    )
    parsed: list[IndexChange] = []
    for pattern in patterns:
        for match in pattern.finditer(body_text):
            try:
                parsed_date = pd.Timestamp(match.group("effective_date")).date()
            except (TypeError, ValueError) as exc:
                raise DataReadinessError(f"Prose S&P 500 announcement has an invalid effective date in {source_url}") from exc
            effective_at = datetime.combine(
                parsed_date,
                datetime.min.time(),
                tzinfo=ZoneInfo("America/New_York"),
            ).astimezone(UTC)
            for ticker in _normalized_ticker_group(match.group("exchange_tickers")):
                parsed.append(
                    IndexChange(
                        effective_at_utc=effective_at,
                        action="addition",
                        ticker=ticker,
                        company=match.group("company").strip(),
                        sector="",
                        source_url=source_url,
                        source_published_date=published_date,
                        source_sha256=source_sha256,
                    )
                )
    return parsed


def _ticker_at(
    ticker: str,
    published_date: date,
    effective_at: datetime,
    aliases: list[SymbolChange],
) -> str:
    current = ticker
    aliases_by_time: dict[datetime, list[SymbolChange]] = {}
    for alias in aliases:
        alias_date = alias.effective_at_utc.astimezone(ZoneInfo("America/New_York")).date()
        if alias.identity_continuity and published_date < alias_date and alias.effective_at_utc <= effective_at:
            aliases_by_time.setdefault(alias.effective_at_utc, []).append(alias)
    for alias_time in sorted(aliases_by_time):
        matches = [alias for alias in aliases_by_time[alias_time] if alias.old_ticker == current]
        if len(matches) > 1:
            destinations = sorted({alias.new_ticker for alias in matches})
            raise DataReadinessError(f"Ticker {current} has ambiguous destinations at {alias_time.isoformat()}: {destinations}")
        if matches:
            current = matches[0].new_ticker
    return current


def build_point_in_time_sp500_universe(
    *,
    current_snapshot: pd.DataFrame,
    event_directory: Path,
    archive_directory: Path,
    symbol_changes: list[SymbolChange] | None = None,
    start_date: date,
    cutoff_date: date,
    anchor_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Verify official event authority and reconstruct effective intervals."""

    from market_predictor.universe.sp500.index_change_events import (
        require_spglobal_event_reconstruction_ready,
    )

    verified_changes = require_spglobal_event_reconstruction_ready(
        event_directory,
        archive_directory=archive_directory,
    )
    return _build_point_in_time_sp500_universe(
        current_snapshot=current_snapshot,
        verified_changes=verified_changes,
        symbol_changes=symbol_changes,
        start_date=start_date,
        cutoff_date=cutoff_date,
        anchor_source=anchor_source,
    )


def _build_point_in_time_sp500_universe(
    *,
    current_snapshot: pd.DataFrame,
    verified_changes: VerifiedIndexChanges,
    symbol_changes: list[SymbolChange] | None = None,
    start_date: date,
    cutoff_date: date,
    anchor_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reverse official changes from a frozen current snapshot into effective intervals."""
    if start_date > cutoff_date:
        raise ValueError("start_date must not be after cutoff_date")
    if not re.fullmatch(r"[0-9a-f]{64}", verified_changes.authority_sha256):
        raise DataReadinessError("S&P event authority SHA-256 is invalid")
    if not re.fullmatch(r"[0-9a-f]{64}", verified_changes.event_set_sha256):
        raise DataReadinessError("S&P event-set SHA-256 is invalid")
    changes = list(verified_changes.changes)
    current = _normalize_current_snapshot(current_snapshot)
    aliases = symbol_changes or []
    relevant = [
        item for item in changes if start_date <= item.effective_at_utc.astimezone(ZoneInfo("America/New_York")).date() <= cutoff_date
    ]
    resolved_changes = [
        replace(
            item,
            ticker=_ticker_at(
                item.ticker,
                item.source_published_date,
                item.effective_at_utc,
                aliases,
            ),
        )
        for item in relevant
    ]
    identity_payload = {
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "anchor_source": anchor_source,
        "event_authority_sha256": verified_changes.authority_sha256,
        "event_set_sha256": verified_changes.event_set_sha256,
        "current_tickers": sorted(current.index.astype(str)),
        "changes": [item.to_record() for item in relevant],
        "symbol_changes": [item.to_record() for item in aliases],
    }
    snapshot_hash = hashlib.sha256(json.dumps(identity_payload, sort_keys=True).encode("utf-8")).hexdigest()
    snapshot_id = f"sp500-pit-{snapshot_hash[:20]}"
    metadata = _metadata_by_ticker(current, relevant, aliases)
    states = {ticker: True for ticker in current.index.astype(str)}
    interval_ends: dict[str, datetime | None] = {ticker: None for ticker in states}
    interval_sources: dict[str, set[str]] = {ticker: {anchor_source} for ticker in states}
    intervals: list[dict[str, Any]] = []
    contradictions: list[str] = []
    applied_aliases: list[SymbolChange] = []
    grouped: dict[datetime, list[IndexChange]] = {}
    for change in resolved_changes:
        grouped.setdefault(change.effective_at_utc, []).append(change)
    aliases_by_time: dict[datetime, list[SymbolChange]] = {}
    for alias in aliases:
        alias_date = alias.effective_at_utc.astimezone(ZoneInfo("America/New_York")).date()
        if start_date <= alias_date <= cutoff_date:
            aliases_by_time.setdefault(alias.effective_at_utc, []).append(alias)
    for effective_at in sorted(set(grouped).union(aliases_by_time), reverse=True):
        additions = [item for item in grouped.get(effective_at, []) if item.action == "addition"]
        deletions = [item for item in grouped.get(effective_at, []) if item.action == "deletion"]
        for change in additions:
            if not states.get(change.ticker, False):
                contradictions.append(f"{change.ticker} addition at {effective_at.isoformat()} is not present immediately after the event")
                continue
            interval_sources.setdefault(change.ticker, set()).update(source.source_url for source in change.source_evidence())
            intervals.append(
                _membership_record(
                    change.ticker,
                    effective_at,
                    interval_ends.get(change.ticker),
                    metadata,
                    snapshot_id,
                    interval_sources[change.ticker],
                )
            )
            states[change.ticker] = False
            interval_ends[change.ticker] = None
            interval_sources[change.ticker] = set()
        for change in deletions:
            if states.get(change.ticker, False):
                contradictions.append(f"{change.ticker} deletion at {effective_at.isoformat()} is present immediately after the event")
                continue
            states[change.ticker] = True
            interval_ends[change.ticker] = effective_at
            interval_sources[change.ticker] = {
                anchor_source,
                *(source.source_url for source in change.source_evidence()),
            }
        for alias in aliases_by_time.get(effective_at, []):
            if alias.old_ticker == alias.new_ticker or not states.get(alias.new_ticker, False):
                continue
            if states.get(alias.old_ticker, False):
                if not alias.identity_continuity:
                    continue
                contradictions.append(f"{alias.old_ticker}->{alias.new_ticker} at {effective_at.isoformat()} has both tickers active")
                continue
            applied_aliases.append(alias)
            new_sources = interval_sources.setdefault(alias.new_ticker, {anchor_source})
            new_sources.add(alias.source_url)
            intervals.append(
                _membership_record(
                    alias.new_ticker,
                    effective_at,
                    interval_ends.get(alias.new_ticker),
                    metadata,
                    snapshot_id,
                    new_sources,
                )
            )
            states[alias.new_ticker] = False
            interval_ends[alias.new_ticker] = None
            interval_sources[alias.new_ticker] = set()
            states[alias.old_ticker] = True
            interval_ends[alias.old_ticker] = effective_at
            interval_sources[alias.old_ticker] = {anchor_source, alias.source_url}
    start_at = datetime.combine(start_date, datetime.min.time(), tzinfo=ZoneInfo("America/New_York")).astimezone(UTC)
    for ticker, present in states.items():
        if present:
            intervals.append(
                _membership_record(
                    ticker,
                    start_at,
                    interval_ends.get(ticker),
                    metadata,
                    snapshot_id,
                    interval_sources.get(ticker, {anchor_source}),
                )
            )
    if contradictions:
        raise DataReadinessError("Point-in-time S&P reconstruction contradictions: " + " | ".join(contradictions[:20]))
    universe = pd.DataFrame(intervals).sort_values(["effective_from_utc", "ticker"], kind="stable").reset_index(drop=True)
    universe, security_audit = _attach_security_identities(
        universe,
        current=current,
        aliases=aliases,
        cutoff_date=cutoff_date,
    )
    audit = {
        "schema": "ml_v3.sp500_point_in_time_universe.v1",
        "generated_at_utc": datetime.now(UTC).isoformat(),
        "start_date": start_date.isoformat(),
        "cutoff_date": cutoff_date.isoformat(),
        "universe_snapshot_id": snapshot_id,
        "snapshot_sha256": snapshot_hash,
        "anchor_source": anchor_source,
        "current_tickers": len(current),
        "historical_tickers": int(universe["ticker"].nunique()),
        "membership_intervals": len(universe),
        "change_events": len(relevant),
        "symbol_change_events": len(applied_aliases),
        "symbol_changes": [item.to_record() for item in applied_aliases],
        "resolved_announcement_tickers": [
            {
                "effective_at_utc": original.effective_at_utc.isoformat(),
                "source_ticker": original.ticker,
                "effective_ticker": resolved.ticker,
            }
            for original, resolved in zip(relevant, resolved_changes, strict=True)
            if original.ticker != resolved.ticker
        ],
        "security_identity": security_audit,
        "source_urls": sorted({source.source_url for item in relevant for source in item.source_evidence()}),
        "source_evidence": [
            {"source_url": source_url, "source_sha256": source_sha256}
            for source_url, source_sha256 in sorted(
                {(source.source_url, source.source_sha256) for item in relevant for source in item.source_evidence()}
            )
        ],
        "contradictions": contradictions,
    }
    return universe, audit


def _normalize_current_snapshot(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {column.lower().strip(): column for column in frame.columns}
    ticker_column = aliases.get("ticker") or aliases.get("symbol")
    if ticker_column is None:
        raise DataReadinessError("Current S&P snapshot requires a ticker column")
    normalized = frame.copy()
    normalized["ticker"] = normalized[ticker_column].astype(str).map(normalized_ticker)
    if normalized["ticker"].duplicated().any():
        duplicates = sorted(normalized.loc[normalized["ticker"].duplicated(keep=False), "ticker"].unique())
        raise DataReadinessError(f"Current S&P snapshot contains duplicate tickers: {duplicates[:20]}")
    return normalized.set_index("ticker", drop=False)


def _metadata_by_ticker(
    current: pd.DataFrame,
    changes: list[IndexChange],
    symbol_changes: list[SymbolChange],
) -> dict[str, dict[str, str]]:
    aliases = {column.lower().strip(): column for column in current.columns}
    company_column, sector_column, industry_column = aliases.get("company"), aliases.get("sector"), aliases.get("industry")
    metadata: dict[str, dict[str, str]] = {}
    for ticker, row in current.iterrows():
        metadata[str(ticker)] = {
            "company": str(row[company_column]).strip() if company_column else str(ticker),
            "sector": str(row[sector_column]).strip() if sector_column else "Unknown",
            "industry": str(row[industry_column]).strip() if industry_column else "Unknown",
        }
    for change in changes:
        item = metadata.setdefault(change.ticker, {"company": change.company, "sector": change.sector, "industry": "Unknown"})
        if not item.get("company") or item["company"] == change.ticker:
            item["company"] = change.company
        if not item.get("sector") or item["sector"] == "Unknown":
            item["sector"] = change.sector
    for _ in range(len(symbol_changes) + 1):
        propagated = False
        for alias in symbol_changes:
            if alias.new_ticker in metadata and alias.old_ticker not in metadata:
                source = metadata[alias.new_ticker]
                metadata[alias.old_ticker] = (
                    dict(source)
                    if alias.identity_continuity
                    else {
                        "company": alias.old_ticker,
                        "sector": source["sector"],
                        "industry": source["industry"],
                    }
                )
                propagated = True
            elif alias.old_ticker in metadata and alias.new_ticker not in metadata:
                source = metadata[alias.old_ticker]
                metadata[alias.new_ticker] = (
                    dict(source)
                    if alias.identity_continuity
                    else {
                        "company": alias.new_ticker,
                        "sector": source["sector"],
                        "industry": source["industry"],
                    }
                )
                propagated = True
        if not propagated:
            break
    for alias in symbol_changes:
        metadata.setdefault(
            alias.old_ticker,
            {"company": alias.old_ticker, "sector": "Unknown", "industry": "Unknown"},
        )
        metadata.setdefault(
            alias.new_ticker,
            {"company": alias.new_ticker, "sector": "Unknown", "industry": "Unknown"},
        )
    return metadata


def _attach_security_identities(
    universe: pd.DataFrame,
    *,
    current: pd.DataFrame,
    aliases: list[SymbolChange],
    cutoff_date: date,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    output = universe.copy()
    identities = [
        _security_identity_for_interval(
            ticker=str(record["ticker"]),
            company=str(record["company"]),
            effective_from=pd.Timestamp(record["effective_from_utc"]),
            effective_to=pd.Timestamp(record["effective_to_utc"]) if not pd.isna(record["effective_to_utc"]) else None,
            current=current,
            aliases=aliases,
        )
        for record in output.to_dict(orient="records")
    ]
    output.insert(1, "security_id", identities)

    active_at_cutoff = datetime.combine(
        cutoff_date,
        datetime.min.time(),
        tzinfo=ZoneInfo("America/New_York"),
    ).astimezone(UTC)
    active = output[
        output["effective_from_utc"].le(active_at_cutoff)
        & (output["effective_to_utc"].isna() | output["effective_to_utc"].gt(active_at_cutoff))
    ]
    if bool(active["ticker"].duplicated().any()):
        duplicates = sorted(active.loc[active["ticker"].duplicated(keep=False), "ticker"].unique())
        raise DataReadinessError(f"Current membership has duplicate ticker identities: {duplicates[:20]}")

    current_columns = {column.lower().strip(): column for column in current.columns}
    metadata_columns = {
        "company": current_columns.get("company"),
        "sector": current_columns.get("sector"),
        "industry": current_columns.get("industry"),
    }
    current_metadata_by_identity: dict[str, dict[str, str]] = {}
    for record in active.to_dict(orient="records"):
        ticker = str(record["ticker"])
        if ticker not in current.index:
            continue
        source = current.loc[ticker]
        current_metadata_by_identity[str(record["security_id"])] = {
            target: str(source[column]).strip()
            for target, column in metadata_columns.items()
            if column is not None and not pd.isna(source[column]) and str(source[column]).strip()
        }
    for security_id, values in current_metadata_by_identity.items():
        mask = output["security_id"].eq(security_id)
        for column, value in values.items():
            output.loc[mask, column] = value

    _validate_security_identity_intervals(output)
    reused = output.groupby("ticker", sort=True)["security_id"].nunique().loc[lambda values: values.gt(1)].index.astype(str).tolist()
    return output, {
        "security_identities": int(output["security_id"].nunique()),
        "reused_tickers": reused,
        "current_metadata_identities": len(current_metadata_by_identity),
    }


def _security_identity_for_interval(
    *,
    ticker: str,
    company: str,
    effective_from: pd.Timestamp,
    effective_to: pd.Timestamp | None,
    current: pd.DataFrame,
    aliases: list[SymbolChange],
) -> str:
    old_matches = [
        alias
        for alias in aliases
        if alias.old_ticker == ticker and effective_to is not None and alias.effective_at_utc == effective_to.to_pydatetime()
    ]
    if old_matches:
        security_ids = {alias.old_security_id for alias in old_matches}
        if len(security_ids) != 1:
            effective_to_text = effective_to.isoformat() if effective_to is not None else "unknown"
            raise DataReadinessError(f"Ticker {ticker} has ambiguous security identities at {effective_to_text}")
        return security_ids.pop()

    new_matches = [alias for alias in aliases if alias.new_ticker == ticker and alias.effective_at_utc <= effective_from.to_pydatetime()]
    if new_matches:
        latest = max(alias.effective_at_utc for alias in new_matches)
        security_ids = {alias.new_security_id for alias in new_matches if alias.effective_at_utc == latest}
        if len(security_ids) != 1:
            raise DataReadinessError(f"Ticker {ticker} has ambiguous security identities at {latest.isoformat()}")
        return security_ids.pop()

    if effective_to is None and ticker in current.index:
        current_columns = {column.lower().strip(): column for column in current.columns}
        cik_column = current_columns.get("cik")
        if cik_column is not None:
            cik = str(current.loc[ticker, cik_column]).strip().removesuffix(".0")
            if cik and cik.lower() != "nan":
                return f"cik:{cik.zfill(10)}:ticker:{ticker}"

    identity_payload = f"{ticker}|{company.strip().lower()}|{effective_from.isoformat()}"
    return f"sp500-historical:{hashlib.sha256(identity_payload.encode('utf-8')).hexdigest()[:24]}"


def _validate_security_identity_intervals(universe: pd.DataFrame) -> None:
    for security_id, group in universe.sort_values(["security_id", "effective_from_utc"]).groupby(
        "security_id",
        sort=False,
    ):
        previous_end: pd.Timestamp | None = None
        previous_is_open = False
        for record in group.itertuples(index=False):
            start = pd.Timestamp(record.effective_from_utc)
            if previous_is_open or (previous_end is not None and start < previous_end):
                raise DataReadinessError(f"Security identity {security_id} has overlapping membership intervals")
            previous_is_open = pd.isna(record.effective_to_utc)
            previous_end = None if previous_is_open else pd.Timestamp(record.effective_to_utc)


def _membership_record(
    ticker: str,
    effective_from: datetime,
    effective_to: datetime | None,
    metadata: dict[str, dict[str, str]],
    snapshot_id: str,
    sources: set[str],
) -> dict[str, Any]:
    item = metadata[ticker]
    sector = item["sector"]
    return {
        "ticker": ticker,
        "company": item["company"],
        "effective_from_utc": effective_from,
        "effective_to_utc": effective_to,
        "sector": sector,
        "industry": item["industry"],
        "market_cap_bucket": "large_cap_sp500",
        "liquidity_bucket": "sp500_constituent",
        "primary_benchmark": SECTOR_BENCHMARKS.get(sector, "SPY"),
        "universe_snapshot_id": snapshot_id,
        "membership_source_urls": json.dumps(sorted(sources)),
    }
