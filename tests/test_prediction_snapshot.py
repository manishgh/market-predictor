from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from market_predictor.prediction_contracts import PredictionRequest, PredictionResponse
from market_predictor.prediction_snapshot import PredictionSnapshotStore


class PredictionSnapshotStoreTests(unittest.TestCase):
    def test_records_and_loads_content_addressed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            request = PredictionRequest(tickers=["MSFT"], mode="swing")
            response = PredictionResponse(mode="swing", horizon="auto")

            recorded = store.record(request, response)
            loaded_request, loaded_response, envelope = store.load(recorded.snapshot_id or "")

            self.assertEqual(len(recorded.snapshot_id or ""), 64)
            self.assertEqual(recorded.snapshot_id, recorded.snapshot_sha256)
            self.assertEqual(loaded_request.tickers, ["MSFT"])
            self.assertEqual(loaded_response.snapshot_id, recorded.snapshot_id)
            self.assertEqual(envelope["content_sha256"], recorded.snapshot_id)
            self.assertTrue(store.path_for(recorded.snapshot_id or "").exists())

    def test_detects_snapshot_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            recorded = store.record(
                PredictionRequest(tickers=["MSFT"], mode="swing"),
                PredictionResponse(mode="swing", horizon="auto"),
            )
            path = store.path_for(recorded.snapshot_id or "")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["content"]["request"]["tickers"] = ["AAPL"]
            path.write_text(json.dumps(envelope), encoding="utf-8")

            with self.assertRaisesRegex(ValueError, "integrity check failed"):
                store.load(recorded.snapshot_id or "")


if __name__ == "__main__":
    unittest.main()
