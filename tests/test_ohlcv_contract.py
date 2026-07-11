from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.cli import _normalize_ohlcv


class OhlcvContractTests(unittest.TestCase):
    def test_normalized_bars_persist_feed_provenance(self) -> None:
        frame = pd.DataFrame(
            {
                "timestamp": ["2026-07-08T14:30:00Z"],
                "open": [100],
                "high": [101],
                "low": [99],
                "close": [100.5],
                "volume": [1_000],
            }
        )
        normalized = _normalize_ohlcv("msft", frame, "5m", price_feed="SIP")
        self.assertEqual(normalized.iloc[0]["symbol"], "MSFT")
        self.assertEqual(normalized.iloc[0]["price_feed"], "sip")


if __name__ == "__main__":
    unittest.main()
