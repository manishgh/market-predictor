from __future__ import annotations

from datetime import date, timedelta
import unittest

import pandas as pd

from market_predictor.alerts import AlertConfig, backtest_indicator_alerts, generate_indicator_alerts


class AlertRuleTests(unittest.TestCase):
    def test_macd_bullish_cross_emits_up_alert(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "MSFT",
                    "date": date(2026, 1, 2),
                    "close": 100.0,
                    "volume": 1_000_000,
                    "macd_signal_diff": -0.05,
                    "volume_z20": 0.0,
                },
                {
                    "ticker": "MSFT",
                    "date": date(2026, 1, 3),
                    "close": 102.0,
                    "volume": 1_500_000,
                    "macd_signal_diff": 0.10,
                    "volume_z20": 1.0,
                },
            ]
        )
        alerts = generate_indicator_alerts(frame, AlertConfig(min_score=2.0), latest_only=True)
        self.assertIn("macd_bullish_cross", set(alerts["alert_type"]))
        self.assertIn("up", set(alerts["direction"]))

    def test_volume_breakout_requires_volume_confirmation(self) -> None:
        rows = []
        start = date(2026, 1, 1)
        for index in range(25):
            rows.append(
                {
                    "ticker": "NVDA",
                    "date": start + timedelta(days=index),
                    "close": 100.0 + index * 0.1,
                    "volume": 1_000_000,
                    "volume_z20": 0.0,
                }
            )
        rows.append(
            {
                "ticker": "NVDA",
                "date": start + timedelta(days=25),
                "close": 110.0,
                "volume": 3_000_000,
                "volume_z20": 2.0,
            }
        )
        alerts = generate_indicator_alerts(pd.DataFrame(rows), AlertConfig(min_score=2.0), latest_only=True)
        self.assertIn("volume_confirmed_breakout", set(alerts["alert_type"]))

    def test_backtest_summary_scores_directional_wins(self) -> None:
        frame = pd.DataFrame(
            [
                {
                    "ticker": "AAPL",
                    "date": date(2026, 1, 1),
                    "close": 100.0,
                    "volume": 1_000_000,
                    "macd_signal_diff": -0.2,
                    "volume_z20": 0.0,
                    "future_return_1d": 0.01,
                    "target_up_1d": 1,
                },
                {
                    "ticker": "AAPL",
                    "date": date(2026, 1, 2),
                    "close": 101.0,
                    "volume": 1_500_000,
                    "macd_signal_diff": 0.2,
                    "volume_z20": 1.2,
                    "future_return_1d": 0.02,
                    "target_up_1d": 1,
                },
            ]
        )
        alerts, summary = backtest_indicator_alerts(frame, horizon_days=1, config=AlertConfig(min_score=2.0))
        self.assertGreaterEqual(len(alerts), 1)
        self.assertIn("macd_bullish_cross", set(alerts["alert_type"]))
        overall = summary[summary["alert_type"].eq("ALL")].iloc[0]
        self.assertEqual(overall["count"], len(alerts))
        self.assertEqual(overall["direction_win_rate"], 1.0)


if __name__ == "__main__":
    unittest.main()
