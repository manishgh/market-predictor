from __future__ import annotations

import re
from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import requests
from bs4 import BeautifulSoup

from market_predictor.schemas import NewsEvent


class FinvizSource:
    base_url = "https://finviz.com/quote.ashx"

    def __init__(self, *, user_agent: str = "market-predictor/0.1") -> None:
        self.headers = {"User-Agent": f"Mozilla/5.0 {user_agent}"}

    def fetch_news(
        self,
        ticker: str,
        start: datetime,
        *,
        end: datetime | None = None,
        limit: int = 100,
    ) -> list[NewsEvent]:
        symbol = ticker.upper().strip()
        response = requests.get(
            self.base_url,
            params={"t": symbol, "p": "d"},
            headers=self.headers,
            timeout=30,
        )
        response.raise_for_status()
        soup = BeautifulSoup(response.text, "html.parser")
        rows = soup.select("table.fullview-news-outer tr")
        events: list[NewsEvent] = []
        last_date: datetime | None = None
        start_utc = start.astimezone(UTC)
        end_utc = (end or datetime.now(UTC)).astimezone(UTC)
        for row in rows[:limit]:
            cells = row.select("td")
            link = row.select_one("a")
            if len(cells) < 2 or link is None:
                continue
            time_text = cells[0].get_text(" ", strip=True)
            title = link.get_text(" ", strip=True)
            url = link.get("href")
            published, last_date = self._parse_finviz_time(time_text, last_date)
            if published is None:
                continue
            if published < start_utc or published > end_utc:
                continue
            source_name = self._provider_from_title(title)
            events.append(
                NewsEvent(
                    ticker=symbol,
                    timestamp=published,
                    source="finviz",
                    title=title,
                    url=str(url) if url else None,
                    summary=None,
                    text=title,
                    raw={"finviz_time": time_text, "provider": source_name},
                )
            )
        return events

    @staticmethod
    def _parse_finviz_time(value: str, last_date: datetime | None) -> tuple[datetime | None, datetime | None]:
        eastern = ZoneInfo("America/New_York")
        value = value.strip()
        try:
            if re.match(r"^[A-Z][a-z]{2}-\d{2}-\d{2}\s+", value):
                parsed = datetime.strptime(value, "%b-%d-%y %I:%M%p").replace(tzinfo=eastern)
                return parsed.astimezone(UTC), parsed
            if last_date is None:
                return None, last_date
            parsed_time = datetime.strptime(value, "%I:%M%p").time()
            parsed = datetime.combine(last_date.date(), parsed_time, tzinfo=eastern)
            return parsed.astimezone(UTC), last_date
        except ValueError:
            return None, last_date

    @staticmethod
    def _provider_from_title(title: str) -> str | None:
        match = re.search(r"\(([^()]+)\)\s*$", title)
        return match.group(1) if match else None
