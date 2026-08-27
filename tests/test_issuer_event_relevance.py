from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.catalysts.issuer_events.relevance import (
    RELEVANCE_POLICY_VERSION,
    SecurityMetadata,
    add_event_relevance,
)


class SwingEventRelevanceTests(unittest.TestCase):
    def test_relevance_policy_identity_is_frozen(self) -> None:
        self.assertEqual(RELEVANCE_POLICY_VERSION, "swing.event_relevance.v1")

    def test_lunr_space_theme_outweighs_unrelated_oil_headline(self) -> None:
        events = pd.DataFrame(
            {
                "security_id": ["security:lunr", "security:lunr"],
                "ticker": ["LUNR", "LUNR"],
                "title": [
                    "Space stocks sold off on Friday; should you buy the dip?",
                    "Oil executives warn gasoline prices will get worse",
                ],
                "summary": ["Space sector update", "Energy market update"],
                "text": ["", ""],
            }
        )
        metadata = SecurityMetadata(
            security_id="security:lunr",
            ticker="LUNR",
            company="Intuitive Machines, Inc.",
            sector="Industrials",
            industry="Aerospace & Defense",
        )

        scored = add_event_relevance(events, metadata)

        self.assertGreater(float(scored.loc[0, "relevance"]), 0.5)
        self.assertLess(float(scored.loc[1, "relevance"]), 0.5)
        self.assertIn("industry_theme_title", scored.loc[0, "relevance_basis"])

    def test_direct_company_headline_is_high_relevance(self) -> None:
        events = pd.DataFrame(
            {
                "security_id": ["security:lunr"],
                "ticker": ["LUNR"],
                "title": ["Intuitive Machines wins new lunar contract"],
                "summary": [""],
                "text": [""],
            }
        )
        metadata = SecurityMetadata(
            security_id="security:lunr",
            ticker="LUNR",
            company="Intuitive Machines, Inc.",
            sector="Industrials",
            industry="Aerospace & Defense",
        )

        scored = add_event_relevance(events, metadata)

        self.assertGreaterEqual(float(scored.loc[0, "relevance"]), 1.0)
        self.assertIn("company_title", scored.loc[0, "relevance_basis"])

    def test_generic_multi_stock_roundup_stays_low_without_identity_match(
        self,
    ) -> None:
        events = pd.DataFrame(
            {
                "security_id": ["security:lunr"],
                "ticker": ["LUNR"],
                "title": ["Key deals this week: GSK, Incyte, and OpenAI"],
                "summary": [""],
                "text": [""],
            }
        )
        metadata = SecurityMetadata(
            security_id="security:lunr",
            ticker="LUNR",
            company="Intuitive Machines, Inc.",
            sector="Industrials",
            industry="Aerospace & Defense",
        )

        scored = add_event_relevance(events, metadata)

        self.assertEqual(float(scored.loc[0, "relevance"]), 0.1)
        self.assertIn("generic_penalty", scored.loc[0, "relevance_basis"])

    def test_single_letter_ticker_does_not_match_indefinite_article(self) -> None:
        events = pd.DataFrame(
            {
                "security_id": ["security:a", "security:a"],
                "ticker": ["A", "A"],
                "title": [
                    "A new clinical trial reports mixed results",
                    "Agilent rises after NYSE: A reports stronger earnings",
                ],
                "summary": ["", ""],
                "text": ["", ""],
            }
        )
        metadata = SecurityMetadata(
            security_id="security:a",
            ticker="A",
            company="Agilent Technologies, Inc.",
            sector="Health Care",
            industry="Life Sciences Tools & Services",
        )

        scored = add_event_relevance(events, metadata)

        self.assertNotIn("ticker_title", scored.loc[0, "relevance_basis"])
        self.assertIn("ticker_title", scored.loc[1, "relevance_basis"])


if __name__ == "__main__":
    unittest.main()
