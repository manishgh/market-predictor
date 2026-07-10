from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.catalyst_overlay import assess_catalyst_overlay, overlay_decision_score


class CatalystOverlayTests(unittest.TestCase):
    def test_positive_relevant_multi_source_catalyst_confirms_bullish_model(self) -> None:
        assessment = assess_catalyst_overlay(
            pd.Series(
                {
                    "news_count_2h": 3,
                    "sentiment_mean_2h": 0.40,
                    "event_relevance_mean_2h": 1.2,
                    "source_count_alpaca_2h": 2,
                    "source_count_sec_2h": 1,
                    "event_contract_count_2h": 1,
                    "minutes_since_last_catalyst": 20,
                }
            ),
            model_probability=0.75,
        )

        self.assertEqual(assessment.status, "confirmed")
        self.assertEqual(assessment.direction, "positive")
        self.assertEqual(assessment.source_diversity, 2)
        self.assertGreater(overlay_decision_score(0.75, assessment), 0.75)

    def test_negative_catalyst_conflicts_without_changing_model_probability(self) -> None:
        assessment = assess_catalyst_overlay(
            pd.Series(
                {
                    "news_count_2h": 2,
                    "sentiment_mean_2h": -0.25,
                    "event_relevance_mean_2h": 1.1,
                    "source_count_alpaca_2h": 1,
                }
            ),
            model_probability=0.74,
        )

        self.assertEqual(assessment.status, "conflicting")
        self.assertEqual(assessment.direction, "negative")
        self.assertAlmostEqual(overlay_decision_score(0.74, assessment), 0.66)

    def test_strong_negative_material_catalyst_vetoes_long_entry(self) -> None:
        assessment = assess_catalyst_overlay(
            pd.Series(
                {
                    "news_count_2h": 2,
                    "sentiment_mean_2h": -0.60,
                    "event_relevance_mean_2h": 1.5,
                    "source_count_alpaca_2h": 1,
                    "source_count_seeking_alpha_2h": 1,
                    "event_guidance_count_2h": 1,
                }
            ),
            model_probability=0.80,
        )

        self.assertEqual(assessment.status, "veto")
        self.assertAlmostEqual(overlay_decision_score(0.80, assessment), 0.60)

    def test_generic_low_relevance_headlines_do_not_confirm(self) -> None:
        assessment = assess_catalyst_overlay(
            pd.Series(
                {
                    "news_count_2h": 4,
                    "sentiment_mean_2h": 0.50,
                    "event_relevance_mean_2h": 0.2,
                    "generic_movers_count_2h": 4,
                }
            ),
            model_probability=0.80,
        )

        self.assertEqual(assessment.status, "mixed")
        self.assertEqual(assessment.score, 0.0)
        self.assertAlmostEqual(overlay_decision_score(0.80, assessment), 0.80)


if __name__ == "__main__":
    unittest.main()
