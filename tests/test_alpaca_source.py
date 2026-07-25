from __future__ import annotations

import unittest
from datetime import UTC, date, datetime
from unittest.mock import Mock

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource


class AlpacaSourceTests(unittest.TestCase):
    def test_news_pages_preserve_provider_timestamps_and_page_tokens(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json.side_effect = [
            {
                "news": [
                    {
                        "id": 1,
                        "created_at": "2026-07-01T12:00:00Z",
                        "updated_at": "2026-07-01T12:05:00Z",
                        "headline": "First",
                        "source": "benzinga",
                    }
                ],
                "next_page_token": "page-2",
            },
            {
                "news": [
                    {
                        "id": 2,
                        "created_at": "2026-07-01T13:00:00Z",
                        "updated_at": "2026-07-01T13:02:00Z",
                        "headline": "Second",
                        "source": "benzinga",
                    }
                ],
                "next_page_token": None,
            },
        ]
        source.client = client

        events = source.fetch_news(
            "MSFT",
            datetime(2026, 7, 1, tzinfo=UTC),
            datetime(2026, 7, 2, tzinfo=UTC),
        )

        self.assertEqual([event.title for event in events], ["First", "Second"])
        self.assertEqual(events[0].timestamp, datetime(2026, 7, 1, 12, tzinfo=UTC))
        self.assertEqual(
            client.get_json.call_args_list[1].kwargs["params"]["page_token"],
            "page-2",
        )

    def test_news_pagination_rejects_repeated_token(self) -> None:
        source = AlpacaSource(
            Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret")
        )
        client = Mock()
        client.get_json.return_value = {
            "news": [],
            "next_page_token": "same-token",
        }
        source.client = client

        with self.assertRaisesRegex(RuntimeError, "repeated"):
            source.fetch_news(
                "MSFT",
                datetime(2026, 7, 1, tzinfo=UTC),
                datetime(2026, 7, 2, tzinfo=UTC),
            )

    def test_security_transitions_normalize_renames_and_deduplicate_mergers(self) -> None:
        source = AlpacaSource(Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret"))
        client = Mock()
        client.get_json.return_value = {
            "corporate_actions": {
                "name_changes": [
                    {
                        "id": "rename",
                        "process_date": "2022-02-17",
                        "old_symbol": "VIAC",
                        "new_symbol": "PARA",
                        "old_cusip": "92556H206",
                        "new_cusip": "92556H206",
                    }
                ],
                "cash_mergers": [
                    {
                        "id": "cash",
                        "process_date": "2025-08-07",
                        "effective_date": "2025-08-07",
                        "acquiree_symbol": "PARA",
                        "acquirer_symbol": "PSKY",
                        "acquiree_cusip": "92556H206",
                        "acquirer_cusip": "69932A204",
                    }
                ],
                "stock_mergers": [
                    {
                        "id": "stock",
                        "process_date": "2025-08-07",
                        "effective_date": "2025-08-07",
                        "acquiree_symbol": "PARA",
                        "acquirer_symbol": "PSKY",
                        "acquiree_cusip": "92556H206",
                        "acquirer_cusip": "69932A204",
                    }
                ],
            },
            "next_page_token": None,
        }
        source.client = client

        frame = source.fetch_security_transitions(date(2022, 1, 1), date(2026, 1, 1))

        self.assertEqual(list(frame["old_symbol"]), ["VIAC", "PARA"])
        self.assertEqual(list(frame["new_symbol"]), ["PARA", "PSKY"])
        self.assertEqual(list(frame["identity_continuity"]), [True, False])
        self.assertEqual(list(frame["membership_continuity"]), [True, False])


if __name__ == "__main__":
    unittest.main()
