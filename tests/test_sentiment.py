from __future__ import annotations

import unittest
from unittest.mock import patch

import pandas as pd

from market_predictor.features import add_finbert_with_scorer
from market_predictor.sentiment import _load_optional_dependency, build_sentiment_inputs


class _FakeScorer:
    def __init__(self) -> None:
        self.texts: list[str] = []
        self.batch_size = 0

    def score_texts(self, texts: list[str], batch_size: int = 16) -> pd.DataFrame:
        self.texts = texts
        self.batch_size = batch_size
        return pd.DataFrame(
            {
                "sentiment_label": ["positive"] * len(texts),
                "sentiment_score": [0.9] * len(texts),
                "sentiment_numeric": [0.9] * len(texts),
            }
        )


class SentimentInputTests(unittest.TestCase):
    def setUp(self) -> None:
        self.events = pd.DataFrame(
            {
                "title": ["Raises guidance", "Duplicate", ""],
                "summary": ["Revenue beats estimates", "Duplicate", ""],
                "text": ["Long article one", "Long article two", "Fallback article"],
            }
        )

    def test_title_summary_uses_provider_fields_and_falls_back_to_text(self) -> None:
        inputs = build_sentiment_inputs(self.events, mode="title_summary")

        self.assertEqual(inputs.tolist(), ["Raises guidance. Revenue beats estimates", "Duplicate", "Fallback article"])

    def test_title_mode_uses_summary_then_text_as_fallbacks(self) -> None:
        frame = self.events.copy()
        frame.loc[1, "title"] = ""

        inputs = build_sentiment_inputs(frame, mode="title")

        self.assertEqual(inputs.tolist(), ["Raises guidance", "Duplicate", "Fallback article"])

    def test_unknown_mode_is_rejected(self) -> None:
        with self.assertRaisesRegex(ValueError, "Unsupported sentiment input mode"):
            build_sentiment_inputs(self.events, mode="unknown")

    def test_scoring_can_use_a_temporary_input_without_overwriting_raw_text(self) -> None:
        frame = self.events.copy()
        frame["_sentiment_input"] = build_sentiment_inputs(frame)
        scorer = _FakeScorer()

        scored = add_finbert_with_scorer(
            frame,
            scorer,
            batch_size=8,
            text_column="_sentiment_input",
        )

        self.assertEqual(scorer.batch_size, 8)
        self.assertEqual(scorer.texts[0], "Raises guidance. Revenue beats estimates")
        self.assertEqual(scored["text"].tolist(), self.events["text"].tolist())
        self.assertTrue(scored["sentiment_numeric"].eq(0.9).all())

    def test_missing_optional_dependency_has_actionable_error(self) -> None:
        missing = ModuleNotFoundError("No module named 'torch'", name="torch")

        with (
            patch("market_predictor.sentiment.importlib.import_module", side_effect=missing),
            self.assertRaisesRegex(RuntimeError, "install the market-predictor 'training' dependency group"),
        ):
            _load_optional_dependency("torch", extra="training")


if __name__ == "__main__":
    unittest.main()
