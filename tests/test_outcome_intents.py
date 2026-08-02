from __future__ import annotations

import tempfile
import unittest
from datetime import timedelta
from pathlib import Path

import pandas as pd

from market_predictor.feature_store import LiveFeatureStore
from market_predictor.outcome_intents import register_snapshot_intents
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.prediction_contracts import PredictionRequest
from tests.test_prediction_service import (
    _intraday_frame,
    _intraday_service,
    _publish_live_intraday,
    _write_intraday_model,
)


class OutcomeIntentIntegrationTests(unittest.TestCase):
    def test_registers_identity_complete_live_snapshot_for_maturation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "intraday.joblib"
            frame = _intraday_frame("MSFT", rows=150)
            _write_intraday_model(model)
            generated = (
                pd.to_datetime(frame["decision_time_utc"], utc=True)
                .max()
                .to_pydatetime()
                + timedelta(minutes=1)
            )
            store = LiveFeatureStore(root)
            _publish_live_intraday(store, frame, generated)
            service = _intraday_service(
                root,
                dataset=None,
                model=model,
                data_source="live",
                live_feature_store=store,
            )
            response = service.predict(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="intraday",
                    as_of=generated,
                )
            )
            assert response.snapshot_id is not None
            repository = OutcomeRepository(root / "data/outcomes")

            intents = register_snapshot_intents(
                service.snapshot_store,
                repository,
                response.snapshot_id,
            )

            self.assertEqual(len(intents), 1)
            intent = intents[0]
            self.assertEqual(intent.ticker, "MSFT")
            self.assertEqual(intent.view, "intraday")
            self.assertEqual(intent.model_release_id, "e" * 64)
            self.assertEqual(
                repository.load_intent(intent.maturation_key),
                intent,
            )


if __name__ == "__main__":
    unittest.main()
