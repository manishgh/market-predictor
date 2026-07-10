from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from market_predictor.global_context import build_sector_theme_monitor, classify_universe_themes, score_flashpoints
from market_predictor.registry import write_model_manifest


class GlobalContextTests(unittest.TestCase):
    def test_scores_oil_chokepoint_flashpoint(self) -> None:
        now = datetime(2026, 7, 8, tzinfo=timezone.utc)
        events = pd.DataFrame(
            [
                {
                    "timestamp": now - timedelta(hours=1),
                    "title": "Hormuz blockade threat disrupts oil shipment routes",
                    "summary": "Tanker traffic in the Persian Gulf faces missile attack risk.",
                    "sentiment_numeric": -0.6,
                },
                {
                    "timestamp": now - timedelta(hours=2),
                    "title": "Persian Gulf tanker seizure raises oil supply risk",
                    "summary": "",
                    "sentiment_numeric": -0.4,
                },
            ]
        )

        scored = score_flashpoints(events, now=now, lookback_hours=24)

        self.assertFalse(scored.empty)
        first = scored.iloc[0]
        self.assertEqual(first["commodity_channel"], "oil")
        self.assertGreater(first["shock_score"], 0.0)
        self.assertIn("energy_oil_gas", first["positive_themes"])

    def test_classifies_requested_sector_themes(self) -> None:
        universe = pd.DataFrame(
            [
                {"ticker": "MRNA", "company": "Moderna", "sector": "Health Care", "industry": "Biotechnology"},
                {"ticker": "MSFT", "company": "Microsoft", "sector": "Information Technology", "industry": "Systems Software"},
                {"ticker": "GOOGL", "company": "Alphabet", "sector": "Communication Services", "industry": "Interactive Media"},
            ]
        )

        themes = classify_universe_themes(universe).set_index("ticker")["monitor_theme"].to_dict()

        self.assertEqual(themes["MRNA"], "healthcare_biotech")
        self.assertEqual(themes["MSFT"], "software")
        self.assertEqual(themes["GOOGL"], "communication_services")

    def test_sector_monitor_requires_promoted_model_and_applies_flashpoint_impact(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            model_path = Path(tmp) / "model.joblib"
            model = _FakeProbabilityModel()
            joblib.dump({"model": model, "features": ["feature"], "target_col": "target_next_week_big_up"}, model_path)
            training = pd.DataFrame(
                [
                    {"ticker": "XOM", "date": "2026-07-01", "feature": 0.1, "target": 1},
                    {"ticker": "DAL", "date": "2026-07-01", "feature": 0.2, "target": 0},
                ]
            )
            write_model_manifest(
                model_path=model_path,
                model_type="volatile_mover",
                schema_version="volatile_mover.v1",
                target_col="target",
                features=["feature"],
                training_data=training,
                metrics={"roc_auc": 0.7},
                validation_split="date_grouped_purged_walk_forward",
                status="promoted",
            )
            dataset = pd.DataFrame(
                [
                    {"ticker": "XOM", "date": "2026-07-08", "feature": 0.1, "volume_z20": 1.0, "news_count": 2},
                    {"ticker": "DAL", "date": "2026-07-08", "feature": 0.2, "volume_z20": 1.0, "news_count": 2},
                ]
            )
            universe = pd.DataFrame(
                [
                    {"ticker": "XOM", "company": "Exxon Mobil", "sector": "Energy", "industry": "Integrated Oil & Gas"},
                    {"ticker": "DAL", "company": "Delta Air Lines", "sector": "Industrials", "industry": "Passenger Airlines"},
                ]
            )
            flashpoints = pd.DataFrame(
                [
                    {
                        "flashpoint": "oil_chokepoint_middle_east",
                        "shock_score": 0.8,
                        "positive_themes": "energy_oil_gas",
                        "negative_themes": "airlines_travel",
                    }
                ]
            )

            sector_report, ticker_report = build_sector_theme_monitor(
                dataset=dataset,
                universe=universe,
                model_path=model_path,
                flashpoints=flashpoints,
            )

            by_ticker = ticker_report.set_index("ticker")
            self.assertGreater(by_ticker.loc["XOM", "global_net_impact"], 0)
            self.assertLess(by_ticker.loc["DAL", "global_net_impact"], 0)
            self.assertIn("energy_oil_gas", set(sector_report["monitor_theme"]))


class _FakeProbabilityModel:
    def predict_proba(self, frame: pd.DataFrame) -> np.ndarray:
        return np.array([[0.8, 0.2] for _ in range(len(frame))])


if __name__ == "__main__":
    unittest.main()
