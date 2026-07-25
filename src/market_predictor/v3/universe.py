from __future__ import annotations

import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, replace
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from urllib.parse import urljoin
from zoneinfo import ZoneInfo

import pandas as pd
import requests
from bs4 import BeautifulSoup

from market_predictor.v3.contracts import normalized_ticker
from market_predictor.v3.errors import DataReadinessError

SP_GLOBAL_ARCHIVE_URL = "https://press.spglobal.com/index.php"
ARCHIVE_QUERY = {"keywords": "s & p 500 index", "l": "100", "s": "2429"}
USER_AGENT = "market-predictor/0.1 point-in-time-universe-audit"
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
_PUBLISHED_DATE = re.compile(r"/(20\d{2}-\d{2}-\d{2})-")
_CHANGE_TITLE = re.compile(r"\b(?:set|sets)\s+to\s+join\s+S&P\s+500\b", re.IGNORECASE)
_TABLE_COLUMNS = ("Effective Date", "Index Name", "Action", "Company Name", "Ticker", "GICS Sector")
_LEGACY_TABLE_TITLE = re.compile(r"^S&P\s+500\s+INDEX\s*[-–—]\s*(.+)$", re.IGNORECASE)
_LEGACY_ACTIONS = {"ADDED": "addition", "DELETED": "deletion"}
_ANY_LEGACY_TABLE_TITLE = re.compile(r"^S&P\s+.+?\s+INDEX\s*[-\u2013\u2014]", re.IGNORECASE)
_EXCHANGE_TICKER = r"\((?:NYSE|NASD|NASDAQ|NYSE\s+American|OTC(?:QX|QB)?)\s*:\s*([A-Z0-9.-]+)\)"


@dataclass(frozen=True)
class AnnouncementLink:
    published_date: date
    title: str
    url: str


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
        }


@dataclass(frozen=True)
class SymbolChange:
    effective_at_utc: datetime
    old_ticker: str
    new_ticker: str
    source_id: str
    source_url: str
    security_id: str

    def to_record(self) -> dict[str, str]:
        return {
            "effective_at_utc": self.effective_at_utc.isoformat(),
            "old_ticker": self.old_ticker,
            "new_ticker": self.new_ticker,
            "source_id": self.source_id,
            "source_url": self.source_url,
            "security_id": self.security_id,
        }


def symbol_changes_from_alpaca(frame: pd.DataFrame) -> list[SymbolChange]:
    required = {"id", "process_date", "old_symbol", "new_symbol"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(f"Alpaca name-change data is missing columns: {missing}")
    changes: list[SymbolChange] = []
    for record in frame.to_dict(orient="records"):
        old_ticker = normalized_ticker(str(record["old_symbol"]))
        new_ticker = normalized_ticker(str(record["new_symbol"]))
        if _is_temporary_security_symbol(old_ticker) or _is_temporary_security_symbol(new_ticker):
            continue
        effective_date = pd.Timestamp(record["process_date"]).date()
        effective_at = datetime.combine(
            effective_date,
            datetime.min.time(),
            tzinfo=ZoneInfo("America/New_York"),
        ).astimezone(UTC)
        source_id = str(record["id"])
        old_cusip = str(record.get("old_cusip") or "").strip().upper()
        new_cusip = str(record.get("new_cusip") or "").strip().upper()
        security_id = (
            f"cusip:{old_cusip}"
            if old_cusip and old_cusip == new_cusip
            else f"alpaca-name-change:{source_id}"
        )
        changes.append(
            SymbolChange(
                effective_at_utc=effective_at,
                old_ticker=old_ticker,
                new_ticker=new_ticker,
                source_id=source_id,
                source_url=f"https://data.alpaca.markets/v1/corporate-actions?ids={source_id}",
                security_id=security_id,
            )
        )
    return sorted(changes, key=lambda item: (item.effective_at_utc, item.old_ticker, item.new_ticker))


def _is_temporary_security_symbol(ticker: str) -> bool:
    return ticker.endswith((".WI", ".PR", ".WS"))


def discover_sp500_change_announcements(
    *,
    start_date: date,
    end_date: date,
    timeout_seconds: float = 30.0,
    maximum_pages: int = 10,
) -> list[AnnouncementLink]:
    """Discover official constituent-change releases covering an effective-date window."""
    if start_date > end_date:
        raise ValueError("start_date must not be after end_date")
    earliest_publication = start_date - timedelta(days=45)
    discovered: dict[str, AnnouncementLink] = {}
    for page in range(maximum_pages):
        params = dict(ARCHIVE_QUERY)
        params["o"] = str(page * 100)
        response = _request(SP_GLOBAL_ARCHIVE_URL, params=params, timeout_seconds=timeout_seconds)
        soup = BeautifulSoup(response.text, "html.parser")
        page_dates: list[date] = []
        for anchor in soup.find_all("a", href=True):
            url = urljoin(SP_GLOBAL_ARCHIVE_URL, str(anchor["href"]))
            match = _PUBLISHED_DATE.search(url)
            if match is None:
                continue
            published = date.fromisoformat(match.group(1))
            page_dates.append(published)
            title = anchor.get_text(" ", strip=True)
            if _CHANGE_TITLE.search(title) and earliest_publication <= published <= end_date:
                discovered[url] = AnnouncementLink(published_date=published, title=title, url=url)
        if page_dates and min(page_dates) < earliest_publication:
            break
    links = sorted(discovered.values(), key=lambda item: (item.published_date, item.url))
    if not links:
        raise DataReadinessError("No official S&P 500 change announcements were discovered for the requested window")
    return links


def parse_sp500_changes(html: str, *, source_url: str, published_date: date) -> list[IndexChange]:
    """Parse exact S&P 500 addition/deletion rows from one official release."""
    digest = hashlib.sha256(html.encode("utf-8")).hexdigest()
    soup = BeautifulSoup(html, "html.parser")
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
                continue
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
    unique = {(item.effective_at_utc, item.action, item.ticker): item for item in changes}
    parsed = sorted(unique.values(), key=lambda item: (item.effective_at_utc, item.action, item.ticker))
    if not parsed and deferred_rows:
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
            if company_position >= len(row) or sector_position >= len(row):
                raise DataReadinessError(f"Legacy S&P 500 row is incomplete in {source_url}: {row}")
            company = row[company_position].strip()
            sector = row[sector_position].strip()
            if not company:
                continue
            ticker = _legacy_company_ticker(body_text, company, source_url=source_url)
            effective_at = datetime.combine(
                effective_date,
                datetime.min.time(),
                tzinfo=ZoneInfo("America/New_York"),
            ).astimezone(UTC)
            parsed.append(
                IndexChange(
                    effective_at_utc=effective_at,
                    action=current_action,
                    ticker=normalized_ticker(ticker),
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


def _legacy_company_ticker(body_text: str, company: str, *, source_url: str) -> str:
    company_tokens = re.findall(r"[A-Z0-9]+", company, flags=re.IGNORECASE)
    if not company_tokens:
        raise DataReadinessError(f"Legacy S&P 500 company name is empty in {source_url}")
    flexible_company = r"[\W_]*".join(re.escape(token) for token in company_tokens)
    legal_descriptor = r"(?:\s+(?:Holding|Holdings|Group|Global|Worldwide|Technologies|Systems))?"
    corporate_suffix = r"(?:\s*,?\s*(?:Inc(?:orporated)?|Corp(?:oration)?|Co(?:mpany)?|Ltd|PLC|LLC)\.?)*"
    pattern = re.compile(
        rf"\b{flexible_company}\b{legal_descriptor}{corporate_suffix}\s*[.,]?\s*{_EXCHANGE_TICKER}",
        re.IGNORECASE,
    )
    tickers = {match.group(1).upper() for match in pattern.finditer(body_text)}
    if len(tickers) != 1:
        raise DataReadinessError(f"Could not bind legacy S&P 500 company {company!r} to a ticker in {source_url}")
    return tickers.pop()


def collect_sp500_changes(
    *,
    start_date: date,
    end_date: date,
    raw_directory: Path,
    workers: int = 6,
    timeout_seconds: float = 30.0,
) -> tuple[list[IndexChange], dict[str, Any]]:
    """Persist each source independently, then fail closed if any required release fails."""
    links = discover_sp500_change_announcements(start_date=start_date, end_date=end_date, timeout_seconds=timeout_seconds)
    raw_directory.mkdir(parents=True, exist_ok=True)
    changes: list[IndexChange] = []
    sources: list[dict[str, Any]] = []
    failures: list[dict[str, str]] = []
    with ThreadPoolExecutor(max_workers=max(1, workers)) as executor:
        futures = {executor.submit(_download_announcement, link, raw_directory, timeout_seconds): link for link in links}
        for future in as_completed(futures):
            link = futures[future]
            try:
                parsed, source = future.result()
                changes.extend(parsed)
                sources.append(source)
            except Exception as exc:
                failures.append({"url": link.url, "error": f"{type(exc).__name__}: {exc}"})
    manifest = {
        "schema": "ml_v3.sp500_change_sources.v1",
        "collected_at_utc": datetime.now(UTC).isoformat(),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "sources": sorted(sources, key=lambda item: str(item["url"])),
        "failures": sorted(failures, key=lambda item: item["url"]),
    }
    (raw_directory / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    if failures:
        raise DataReadinessError(f"{len(failures)} official S&P announcement downloads failed; see {raw_directory / 'manifest.json'}")
    deduplicated = {
        (item.effective_at_utc, item.action, item.ticker): item
        for item in changes
        if start_date <= item.effective_at_utc.astimezone(ZoneInfo("America/New_York")).date() <= end_date
    }
    return sorted(deduplicated.values(), key=lambda item: (item.effective_at_utc, item.action, item.ticker)), manifest


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
        if published_date < alias_date and alias.effective_at_utc <= effective_at:
            aliases_by_time.setdefault(alias.effective_at_utc, []).append(alias)
    for alias_time in sorted(aliases_by_time):
        matches = [alias for alias in aliases_by_time[alias_time] if alias.old_ticker == current]
        if len(matches) > 1:
            destinations = sorted({alias.new_ticker for alias in matches})
            raise DataReadinessError(
                f"Ticker {current} has ambiguous destinations at {alias_time.isoformat()}: {destinations}"
            )
        if matches:
            current = matches[0].new_ticker
    return current


def build_point_in_time_sp500_universe(
    *,
    current_snapshot: pd.DataFrame,
    changes: list[IndexChange],
    symbol_changes: list[SymbolChange] | None = None,
    start_date: date,
    cutoff_date: date,
    anchor_source: str,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Reverse official changes from a frozen current snapshot into effective intervals."""
    if start_date > cutoff_date:
        raise ValueError("start_date must not be after cutoff_date")
    current = _normalize_current_snapshot(current_snapshot)
    aliases = symbol_changes or []
    relevant = [
        item
        for item in changes
        if start_date <= item.effective_at_utc.astimezone(ZoneInfo("America/New_York")).date() <= cutoff_date
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
            interval_sources.setdefault(change.ticker, set()).add(change.source_url)
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
            interval_sources[change.ticker] = {anchor_source, change.source_url}
        for alias in aliases_by_time.get(effective_at, []):
            if alias.old_ticker == alias.new_ticker or not states.get(alias.new_ticker, False):
                continue
            if states.get(alias.old_ticker, False):
                contradictions.append(
                    f"{alias.old_ticker}->{alias.new_ticker} at {effective_at.isoformat()} has both tickers active"
                )
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
        "source_urls": sorted({item.source_url for item in relevant}),
        "contradictions": contradictions,
    }
    return universe, audit


def _download_announcement(link: AnnouncementLink, raw_directory: Path, timeout_seconds: float) -> tuple[list[IndexChange], dict[str, Any]]:
    response = _request(link.url, timeout_seconds=timeout_seconds)
    digest = hashlib.sha256(response.text.encode("utf-8")).hexdigest()
    path = raw_directory / f"{link.published_date.isoformat()}_{digest[:12]}.html"
    path.write_text(response.text, encoding="utf-8")
    changes = parse_sp500_changes(response.text, source_url=link.url, published_date=link.published_date)
    return changes, {
        "url": link.url,
        "title": link.title,
        "published_date": link.published_date.isoformat(),
        "sha256": digest,
        "raw_path": str(path),
        "change_rows": len(changes),
    }


def _request(url: str, *, timeout_seconds: float, params: dict[str, str] | None = None) -> requests.Response:
    last_error: Exception | None = None
    for attempt in range(3):
        try:
            response = requests.get(url, params=params, headers={"User-Agent": USER_AGENT}, timeout=timeout_seconds)
            response.raise_for_status()
            return response
        except requests.RequestException as exc:
            last_error = exc
            if attempt < 2:
                time.sleep(0.5 * (2**attempt))
    raise DataReadinessError(f"Failed to fetch {url}: {last_error}")


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
                metadata[alias.old_ticker] = dict(metadata[alias.new_ticker])
                propagated = True
            elif alias.old_ticker in metadata and alias.new_ticker not in metadata:
                metadata[alias.new_ticker] = dict(metadata[alias.old_ticker])
                propagated = True
        if not propagated:
            break
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
    reused = (
        output.groupby("ticker", sort=True)["security_id"]
        .nunique()
        .loc[lambda values: values.gt(1)]
        .index.astype(str)
        .tolist()
    )
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
        if alias.old_ticker == ticker
        and effective_to is not None
        and alias.effective_at_utc == effective_to.to_pydatetime()
    ]
    if old_matches:
        security_ids = {alias.security_id for alias in old_matches}
        if len(security_ids) != 1:
            effective_to_text = effective_to.isoformat() if effective_to is not None else "unknown"
            raise DataReadinessError(f"Ticker {ticker} has ambiguous security identities at {effective_to_text}")
        return security_ids.pop()

    new_matches = [
        alias
        for alias in aliases
        if alias.new_ticker == ticker and alias.effective_at_utc <= effective_from.to_pydatetime()
    ]
    if new_matches:
        latest = max(alias.effective_at_utc for alias in new_matches)
        security_ids = {alias.security_id for alias in new_matches if alias.effective_at_utc == latest}
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
