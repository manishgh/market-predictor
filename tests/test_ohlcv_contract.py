from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.cli import _merge_ohlcv_manifest, _normalize_ohlcv


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

    def test_incremental_manifest_replaces_only_requested_ticker_timeframe(self) -> None:
        existing = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "BBB"],
                "timeframe": ["1d", "5m", "5m"],
                "rows": [10, 20, 30],
                "path": ["aaa-1d", "aaa-5m-old", "bbb-5m"],
            }
        )
        update = pd.DataFrame({"ticker": ["AAA"], "timeframe": ["5m"], "rows": [25], "path": ["aaa-5m-new"]})
        merged = _merge_ohlcv_manifest(existing, update, symbols=["AAA"], timeframes={"5m"})
        self.assertEqual(len(merged), 3)
        self.assertEqual(merged.loc[(merged["ticker"] == "AAA") & (merged["timeframe"] == "5m"), "rows"].item(), 25)
        self.assertEqual(merged.loc[(merged["ticker"] == "AAA") & (merged["timeframe"] == "1d"), "rows"].item(), 10)


if __name__ == "__main__":
    unittest.main()
