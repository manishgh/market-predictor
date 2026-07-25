from __future__ import annotations

import unittest
from datetime import date
from unittest.mock import Mock

from market_predictor.config import Settings
from market_predictor.sources.alpaca import AlpacaSource


class AlpacaSourceTests(unittest.TestCase):
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
