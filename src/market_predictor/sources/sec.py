from __future__ import annotations

from datetime import UTC, datetime
from zoneinfo import ZoneInfo

import pandas as pd

from market_predictor.config import Settings
from market_predictor.schemas import NewsEvent
from market_predictor.sources.http import HttpClient


class SecSource:
    ticker_map_url = "https://www.sec.gov/files/company_tickers.json"
    submissions_url = "https://data.sec.gov/submissions/CIK{cik}.json"

    def __init__(self, settings: Settings) -> None:
        self.client = HttpClient(user_agent=settings.sec_user_agent)

    def cik_for_ticker(self, ticker: str) -> str:
        payload = self.client.get_json(self.ticker_map_url)
        ticker_upper = ticker.upper()
        for item in payload.values():
            if item.get("ticker", "").upper() == ticker_upper:
                return str(item["cik_str"]).zfill(10)
        raise ValueError(f"CIK not found for ticker {ticker_upper}")

    def fetch_filings(
        self,
        ticker: str,
        start: datetime,
        end: datetime | None = None,
        *,
        forms: set[str] | None = None,
        limit: int = 100,
    ) -> list[NewsEvent]:
        """Fetch recent SEC submissions as timestamped events.

        SEC acceptance timestamps are Eastern clock times without a timezone suffix.
        The collector records first-seen time separately because SEC does not publish
        an exact first-available-on-sec.gov timestamp.
        """
        end = end or datetime.now(UTC)
        ticker_upper = ticker.upper()
        cik = self.cik_for_ticker(ticker_upper)
        payload = self.client.get_json(self.submissions_url.format(cik=cik))
        recent = payload.get("filings", {}).get("recent", {})
        if not recent:
            return []
        frame = pd.DataFrame(recent)
        if frame.empty or "form" not in frame.columns:
            return []
        if forms:
            wanted = {form.upper() for form in forms}
            frame = frame[frame["form"].astype(str).str.upper().isin(wanted)]
        if "acceptanceDateTime" in frame.columns:
            timestamps = frame["acceptanceDateTime"].map(self._acceptance_time_utc)
        else:
            timestamps = pd.to_datetime(frame.get("filingDate"), errors="coerce", utc=True)
        frame = frame.assign(_timestamp=timestamps).dropna(subset=["_timestamp"])
        frame = frame[(frame["_timestamp"] >= pd.Timestamp(start)) & (frame["_timestamp"] <= pd.Timestamp(end))]
        frame = frame.sort_values("_timestamp", ascending=False).head(limit)
        events: list[NewsEvent] = []
        cik_int = str(int(cik))
        for row in frame.to_dict(orient="records"):
            form = str(row.get("form", "") or "").upper()
            accession = str(row.get("accessionNumber", "") or "")
            accession_path = accession.replace("-", "")
            primary_doc = str(row.get("primaryDocument", "") or "")
            filing_date = str(row.get("filingDate", "") or "")
            report_date = str(row.get("reportDate", "") or "")
            document_url = None
            if accession_path and primary_doc:
                document_url = f"https://www.sec.gov/Archives/edgar/data/{cik_int}/{accession_path}/{primary_doc}"
            title = f"{ticker_upper} SEC {form}"
            if report_date:
                title = f"{title} report {report_date}"
            summary = f"SEC filing {form}"
            if filing_date:
                summary = f"{summary}, filed {filing_date}"
            events.append(
                NewsEvent(
                    ticker=ticker_upper,
                    timestamp=pd.Timestamp(row["_timestamp"]).to_pydatetime(),
                    source=f"sec:{form.lower()}",
                    title=title,
                    url=document_url,
                    summary=summary,
                    text=f"{title}. {summary}.",
                    raw={
                        "cik": cik,
                        "form": form,
                        "accession_number": accession,
                        "primary_document": primary_doc,
                        "filing_date": filing_date,
                        "report_date": report_date,
                    },
                )
            )
        return events

    @staticmethod
    def _acceptance_time_utc(value: object) -> pd.Timestamp:
        try:
            timestamp = pd.Timestamp(value)
        except (TypeError, ValueError):
            return pd.NaT
        if pd.isna(timestamp):
            return pd.NaT
        if timestamp.tzinfo is None:
            timestamp = timestamp.tz_localize(ZoneInfo("America/New_York"))
        return timestamp.tz_convert("UTC")
