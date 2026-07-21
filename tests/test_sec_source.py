from __future__ import annotations

import unittest
from datetime import UTC, datetime

import pandas as pd

from market_predictor.config import Settings
from market_predictor.sources.sec import SecSource


class _FakeSecClient:
    def get_json(self, url: str, **_: object) -> dict[str, object]:
        if url.endswith("company_tickers.json"):
            return {"0": {"ticker": "MSFT", "cik_str": 789019}}
        return {
            "filings": {
                "recent": {
                    "form": ["8-K"],
                    "acceptanceDateTime": ["2026-07-21T16:05:00"],
                    "filingDate": ["2026-07-21"],
                    "reportDate": ["2026-07-21"],
                    "accessionNumber": ["0000789019-26-000001"],
                    "primaryDocument": ["msft-8k.htm"],
                }
            }
        }


class SecSourceTests(unittest.TestCase):
    def test_acceptance_datetime_is_interpreted_as_eastern_clock_time(self) -> None:
        source = SecSource(Settings())
        source.client = _FakeSecClient()  # type: ignore[assignment]
        events = source.fetch_filings(
            "MSFT",
            datetime(2026, 7, 21, 0, 0, tzinfo=UTC),
            end=datetime(2026, 7, 22, 0, 0, tzinfo=UTC),
        )
        self.assertEqual(len(events), 1)
        self.assertEqual(pd.Timestamp(events[0].timestamp), pd.Timestamp("2026-07-21T20:05:00Z"))


if __name__ == "__main__":
    unittest.main()
