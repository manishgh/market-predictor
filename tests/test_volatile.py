from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from market_predictor.features import events_to_frame, source_family_for_source
from market_predictor.volatile import VolatileLabelConfig, build_volatile_dataset


class VolatileDatasetTests(unittest.TestCase):
    def test_source_family_normalizes_seeking_alpha_and_finviz(self) -> None:
        self.assertEqual(source_family_for_source("seeking_alpha:rapidapi_news"), "seeking_alpha")
        self.assertEqual(source_family_for_source("finviz"), "finviz")
        events = events_to_frame(
            [
                {
                    "ticker": "MXL",
                    "timestamp": "2026-07-01T12:00:00Z",
                    "source": "seeking_alpha:rapidapi_news",
                    "title": "MXL news",
                    "summary": "",
                    "text": "",
                }
            ]
        )
        self.assertEqual(events.iloc[0]["source"], "seeking_alpha:rapidapi_news")

    def test_builds_news_and_big_move_labels_without_filling_missing_future(self) -> None:
        rows = []
        start = date(2026, 1, 1)
        for idx in range(8):
            rows.append(
                {
                    "ticker": "RGTI",
                    "date": start + timedelta(days=idx),
                    "close": 10 + idx,
                    "return_1d": 0.02,
                    "return_5d_past": 0.08,
                    "volume_z20": 1.5,
                    "news_count": 1 if idx % 2 == 0 else 0,
                    "news_count_z30": 1.0,
                    "event_count": 1,
                    "sentiment_mean": 0.2,
                    "future_return_1d": 0.04 if idx == 0 else (-0.04 if idx == 1 else None),
                }
            )
        one = pd.DataFrame(rows)
        universe = pd.DataFrame({"ticker": ["RGTI"], "theme_bucket": ["seed_high_beta_mover"]})
        dataset, audit = build_volatile_dataset(
            one,
            universe=universe,
            config=VolatileLabelConfig(min_rows_per_ticker=5, min_news_rows_per_ticker=2),
        )
        self.assertEqual(len(dataset), 8)
        self.assertIn("news_volume_attention", dataset.columns)
        self.assertIn("volatile_setup_score", dataset.columns)
        self.assertEqual(int(dataset.loc[dataset["date"].eq(start), "target_next_day_big_up"].iloc[0]), 1)
        self.assertEqual(int(dataset.loc[dataset["date"].eq(start + timedelta(days=1)), "target_next_day_big_down"].iloc[0]), 1)
        self.assertTrue(pd.isna(dataset.loc[dataset["date"].eq(start + timedelta(days=2)), "target_next_day_big_up"].iloc[0]))
        self.assertTrue(bool(audit.loc[audit["ticker"].eq("RGTI"), "model_eligible"].iloc[0]))

    def test_readiness_gate_removes_tickers_without_news(self) -> None:
        rows = []
        start = date(2026, 1, 1)
        for idx in range(6):
            rows.append(
                {
                    "ticker": "QUIET",
                    "date": start + timedelta(days=idx),
                    "close": 20,
                    "return_1d": 0.0,
                    "return_5d_past": 0.0,
                    "volume_z20": 0.0,
                    "news_count": 0,
                    "future_return_1d": 0.01,
                }
            )
        dataset, audit = build_volatile_dataset(
            pd.DataFrame(rows),
            config=VolatileLabelConfig(min_rows_per_ticker=5, min_news_rows_per_ticker=1),
        )
        self.assertTrue(dataset.empty)
        self.assertFalse(bool(audit.loc[audit["ticker"].eq("QUIET"), "model_eligible"].iloc[0]))


if __name__ == "__main__":
    unittest.main()
