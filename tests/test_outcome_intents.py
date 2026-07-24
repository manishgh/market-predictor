from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from market_predictor.feature_store import LiveFeatureStore
from market_predictor.outcome_intents import register_snapshot_intents
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.prediction_contracts import PredictionRequest
from tests.test_prediction_service import (
    _publish_live_swing,
    _service,
    _swing_frame,
    _write_model,
)


class OutcomeIntentIntegrationTests(unittest.TestCase):
    def test_registers_identity_complete_live_snapshot_for_maturation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            _write_model(
                model,
                features,
                target_col="target_net_positive_5d",
                status="promoted",
                probability=0.73,
            )
            generated = datetime(2025, 9, 17, 22, 5, tzinfo=UTC)
            store = LiveFeatureStore(root)
            _publish_live_swing(store, frame, generated)
            service = _service(
                root,
                swing=(None, model),
                data_source="live",
                live_feature_store=store,
            )
            response = service.predict(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
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
            self.assertEqual(intent.view, "swing")
            self.assertEqual(intent.model_release_id, "e" * 64)
            self.assertEqual(
                repository.load_intent(intent.maturation_key),
                intent,
            )


if __name__ == "__main__":
    unittest.main()
