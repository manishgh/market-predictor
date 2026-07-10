from __future__ import annotations

from datetime import datetime, timezone
import unittest

from market_predictor.sources.gdelt import GdeltSource


class GdeltSourceTests(unittest.TestCase):
    def test_parses_doc_articles_into_market_events(self) -> None:
        payload = {
            "articles": [
                {
                    "title": "Hormuz blockade threat raises oil tanker risk",
                    "url": "https://example.com/hormuz",
                    "seendate": "20260708081500",
                    "domain": "example.com",
                    "language": "English",
                    "sourceCountry": "US",
                    "socialimage": "https://example.com/image.jpg",
                }
            ]
        }

        events = GdeltSource.events_from_payload(payload, query="hormuz oil")

        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event.ticker, "MARKET")
        self.assertEqual(event.source, "gdelt:doc")
        self.assertEqual(event.title, "Hormuz blockade threat raises oil tanker risk")
        self.assertEqual(event.raw["query"], "hormuz oil")
        self.assertEqual(event.raw["source_country"], "US")

    def test_skips_articles_without_title_url_or_timestamp(self) -> None:
        payload = {
            "articles": [
                {"title": "", "url": "https://example.com/a", "seendate": "20260708081500"},
                {"title": "No URL", "url": "", "seendate": "20260708081500"},
                {"title": "No timestamp", "url": "https://example.com/b", "seendate": ""},
            ]
        }

        self.assertEqual(GdeltSource.events_from_payload(payload, query="risk"), [])

    def test_query_failures_are_isolated(self) -> None:
        client = _FlakyClient()
        source = GdeltSource(client=client, endpoint="https://example.com/gdelt", request_pause_seconds=0)

        events, errors = source.fetch_context_events_with_errors(
            datetime(2026, 7, 8, tzinfo=timezone.utc),
            queries=("ok", "bad"),
            max_records_per_query=1,
        )

        self.assertEqual(len(events), 1)
        self.assertEqual(len(errors), 1)
        self.assertIn("bad", errors[0])


class _FlakyClient:
    def get_json(self, url: str, *, params: dict, retries: int, pause: float):
        if params["query"] == "bad":
            raise RuntimeError("rate limited")
        return {
            "articles": [
                {
                    "title": "Taiwan Strait blockade risk hits semiconductors",
                    "url": "https://example.com/taiwan",
                    "seendate": "20260708081500",
                }
            ]
        }


if __name__ == "__main__":
    unittest.main()
