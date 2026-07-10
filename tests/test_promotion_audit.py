from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.promotion_audit import (
    ProfitabilityAuditConfig,
    build_catalyst_news_audit,
    build_market_regime_audit,
    build_walk_forward_profitability_audit,
)


class PromotionAuditTests(unittest.TestCase):
    def test_profitability_audit_joins_predictions_to_path_returns(self) -> None:
        dates = pd.date_range("2026-07-01 13:30:00Z", periods=10, freq="5min")
        dataset = pd.DataFrame(
            {
                "ticker": ["MSFT"] * 10,
                "date": dates,
                "target_entry_success_12b": [0, 1] * 5,
                "horizon_return_from_entry_12b": [-0.002, 0.004, -0.001, 0.006, 0.001, 0.005, -0.003, 0.007, 0.002, 0.008],
                "entry_exit_outcome_12b": ["stop_first", "target_first"] * 5,
                "spy_return_6bar": [0.002] * 10,
                "qqq_return_6bar": [0.002] * 10,
            }
        )
        predictions = pd.DataFrame(
            {
                "ticker": ["MSFT"] * 10,
                "date": dates.tz_convert(None).astype(str),
                "target_entry_success_12b": [0, 1] * 5,
                "oos_probability": [0.1, 0.95, 0.2, 0.90, 0.3, 0.85, 0.4, 0.80, 0.5, 0.75],
            }
        )

        summary, trades, regime = build_walk_forward_profitability_audit(
            dataset=dataset,
            predictions=predictions,
            config=ProfitabilityAuditConfig(top_fraction=0.2),
        )

        self.assertEqual(int(summary.iloc[0]["selected_trades"]), 2)
        self.assertGreater(float(summary.iloc[0]["avg_trade_return"]), 0)
        self.assertFalse(trades.empty)
        self.assertEqual(regime.iloc[0]["market_regime"], "risk_on")

    def test_profitability_audit_can_cap_selected_trades_per_session(self) -> None:
        dates = pd.to_datetime(
            [
                "2026-07-01 13:30:00Z",
                "2026-07-01 13:35:00Z",
                "2026-07-01 13:40:00Z",
                "2026-07-02 13:30:00Z",
                "2026-07-02 13:35:00Z",
                "2026-07-02 13:40:00Z",
            ]
        )
        dataset = pd.DataFrame(
            {
                "ticker": ["MSFT"] * 6,
                "date": dates,
                "_mp_session_date": ["2026-07-01"] * 3 + ["2026-07-02"] * 3,
                "target_entry_success_12b": [1] * 6,
                "horizon_return_from_entry_12b": [0.01, 0.005, -0.003, 0.02, 0.001, -0.004],
                "spy_return_6bar": [0.002] * 6,
                "qqq_return_6bar": [0.002] * 6,
            }
        )
        predictions = pd.DataFrame(
            {
                "ticker": ["MSFT"] * 6,
                "date": dates.tz_convert(None).astype(str),
                "oos_probability": [0.99, 0.98, 0.97, 0.96, 0.95, 0.94],
            }
        )

        summary, trades, _ = build_walk_forward_profitability_audit(
            dataset=dataset,
            predictions=predictions,
            config=ProfitabilityAuditConfig(top_fraction=1.0, max_trades_per_period=1),
        )

        self.assertEqual(int(summary.iloc[0]["selected_trades"]), 2)
        self.assertEqual(trades.groupby("_mp_session_date").size().max(), 1)
        self.assertIn("return_drawdown_ratio", summary.columns)

    def test_regime_audit_reports_coverage(self) -> None:
        dataset = pd.DataFrame(
            {
                "ticker": ["A", "B", "C"],
                "date": pd.date_range("2026-07-01", periods=3),
                "spy_return_6bar": [0.003, -0.004, 0.0],
                "qqq_return_6bar": [0.002, -0.003, 0.0],
            }
        )

        audit = build_market_regime_audit(dataset=dataset)

        self.assertEqual(int(audit.iloc[0]["regimes_present"]), 3)
        self.assertAlmostEqual(float(audit.iloc[0]["max_single_regime_share"]), 1 / 3)

    def test_catalyst_audit_counts_sources_and_alignment_errors(self) -> None:
        dataset = pd.DataFrame(
            {
                "ticker": ["MSFT", "MSFT"],
                "date": pd.date_range("2026-07-01", periods=2),
                "news_count": [2, 0],
                "source_count_seeking_alpha": [1, 0],
                "event_relevance_score": [0.8, 0.2],
            }
        )
        alignment = pd.DataFrame([{"ticker": "MSFT", "events_without_feature_row": 1}])

        audit = build_catalyst_news_audit(dataset=dataset, alignment_audit=alignment)

        self.assertTrue(bool(audit.iloc[0]["has_catalyst_features"]))
        self.assertEqual(int(audit.iloc[0]["alignment_error_total"]), 1)
        self.assertEqual(int(audit.iloc[0]["seeking_alpha_rows"]), 1)


if __name__ == "__main__":
    unittest.main()
