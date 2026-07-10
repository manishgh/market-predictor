from __future__ import annotations

import unittest

from market_predictor.readiness import (
    INVALID,
    VALID,
    WARN,
    assess_daily_readiness,
    assess_intraday_readiness,
)


class PredictionReadinessTests(unittest.TestCase):
    def test_promoted_model_with_required_sources_is_valid(self) -> None:
        readiness = assess_daily_readiness(
            daily_bar_count=260,
            latest_price_date="2026-07-07",
            price_feed="sip",
            benchmark_present=True,
            market_context_present=True,
            model_status="promoted",
            required_sources={"alpaca", "finviz"},
            available_sources={"alpaca", "finviz", "sec"},
        )

        self.assertEqual(readiness.status, VALID)
        self.assertEqual(readiness.source_status, "present")

    def test_unpromoted_model_and_mismatched_news_invalidates_prediction(self) -> None:
        readiness = assess_daily_readiness(
            daily_bar_count=260,
            latest_price_date="2026-07-07",
            price_feed="sip",
            benchmark_present=True,
            market_context_present=True,
            model_status="candidate",
            required_sources={"alpaca", "reddit"},
            available_sources={"alpaca"},
            news_candle_mismatch_count=2,
            stale_cache=True,
        )

        self.assertEqual(readiness.status, INVALID)
        reasons = "; ".join(readiness.reasons)
        self.assertIn("model status is candidate", reasons)
        self.assertIn("missing required source families: reddit", reasons)
        self.assertIn("news/candle mismatches detected: 2", reasons)
        self.assertIn("stale cache", reasons)

    def test_intraday_readiness_uses_intraday_warmup(self) -> None:
        readiness = assess_intraday_readiness(
            intraday_bar_count=130,
            latest_price_timestamp="2026-07-09T15:55:00-04:00",
            price_feed="sip",
            benchmark_present=True,
            market_context_present=True,
            model_status="promoted",
        )

        self.assertEqual(readiness.status, VALID)
        self.assertEqual(readiness.timeframe, "intraday")
        self.assertEqual(readiness.intraday_bar_count, 130)
        self.assertEqual(readiness.daily_bar_count, 0)

    def test_unknown_feed_tier_warns_and_iex_invalidates_volume_features(self) -> None:
        unknown = assess_daily_readiness(
            daily_bar_count=260,
            latest_price_date="2026-07-07",
            price_feed="alpaca",
            benchmark_present=True,
            market_context_present=True,
        )
        partial = assess_intraday_readiness(
            intraday_bar_count=130,
            latest_price_timestamp="2026-07-09T15:55:00-04:00",
            price_feed="iex",
            benchmark_present=True,
            market_context_present=True,
        )

        self.assertEqual(unknown.status, WARN)
        self.assertEqual(partial.status, INVALID)


if __name__ == "__main__":
    unittest.main()
