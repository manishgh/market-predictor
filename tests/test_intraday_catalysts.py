from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.intraday_catalysts import add_intraday_catalyst_features


class IntradayCatalystTests(unittest.TestCase):
    def test_adds_asof_ticker_and_market_context_features_without_future_leakage(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            event_dir = Path(tmp) / "events"
            event_dir.mkdir()
            events = pd.DataFrame(
                [
                    {
                        "ticker": "MSFT",
                        "timestamp": "2026-07-08T13:40:00Z",
                        "source": "seeking_alpha:rapidapi_news",
                        "title": "MSFT wins large AI cloud contract",
                        "summary": "",
                        "text": "MSFT wins large AI cloud contract",
                        "sentiment_numeric": 0.8,
                    },
                    {
                        "ticker": "MSFT",
                        "timestamp": "2026-07-08T14:20:00Z",
                        "source": "alpaca:benzinga",
                        "title": "MSFT future headline",
                        "summary": "",
                        "text": "MSFT future headline",
                        "sentiment_numeric": 0.4,
                    },
                ]
            )
            events.to_parquet(event_dir / "MSFT_events.parquet", index=False)
            market = pd.DataFrame(
                [
                    {
                        "ticker": "MARKET",
                        "timestamp": "2026-07-08T13:45:00Z",
                        "source": "gdelt:doc",
                        "title": "Global chip supply disruption risk",
                        "summary": "",
                        "text": "Global chip supply disruption risk",
                        "sentiment_numeric": -0.7,
                    }
                ]
            )
            market_path = Path(tmp) / "market.parquet"
            market.to_parquet(market_path, index=False)
            bars = pd.DataFrame(
                {
                    "ticker": ["MSFT", "MSFT"],
                    "timestamp": pd.to_datetime(["2026-07-08T13:35:00Z", "2026-07-08T14:00:00Z"]),
                    "date": pd.to_datetime(["2026-07-08T13:35:00Z", "2026-07-08T14:00:00Z"]),
                    "volume_z20": [0.0, 1.0],
                }
            )

            enriched, audit = add_intraday_catalyst_features(bars, event_dirs=[event_dir], market_context_path=market_path)

            self.assertEqual(enriched["news_count_2h"].tolist(), [0.0, 1.0])
            self.assertEqual(enriched["source_count_seeking_alpha_2h"].tolist(), [0.0, 1.0])
            self.assertEqual(enriched["source_count_alpaca_2h"].tolist(), [0.0, 0.0])
            self.assertEqual(enriched["market_context_news_count_2h"].tolist(), [0.0, 1.0])
            self.assertIn("catalyst_attention_score_2h", enriched.columns)
            self.assertEqual(int(audit.iloc[0]["ticker_events"]), 2)


if __name__ == "__main__":
    unittest.main()
