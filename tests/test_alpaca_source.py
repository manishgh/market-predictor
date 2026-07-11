from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource


class AlpacaSourceTests(unittest.TestCase):
    def test_name_changes_are_paginated_and_normalized(self) -> None:
        source = AlpacaSource(Settings(ALPACA_API_KEY_ID="key", ALPACA_API_SECRET_KEY="secret"))
        client = Mock()
        client.get_json.side_effect = [
            {
                "corporate_actions": {
                    "name_changes": [
                        {
                            "id": "change-1",
                            "process_date": "2026-06-24",
                            "old_symbol": "SATS",
                            "new_symbol": "ECHO",
                            "old_cusip": "278768106",
                            "new_cusip": "278768106",
                        }
                    ]
                },
                "next_page_token": "next",
            },
            {"corporate_actions": {"name_changes": []}, "next_page_token": None},
        ]
        source.client = client

        frame = source.fetch_name_changes(date(2026, 6, 1), date(2026, 6, 30))

        self.assertEqual(frame.iloc[0]["old_symbol"], "SATS")
        self.assertEqual(frame.iloc[0]["new_symbol"], "ECHO")
        self.assertEqual(client.get_json.call_count, 2)
        second_params = client.get_json.call_args_list[1].kwargs["params"]
        self.assertEqual(second_params["page_token"], "next")


if __name__ == "__main__":
    unittest.main()
