from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from pydantic import ValidationError

from market_predictor.core.prediction_contracts import (
    GlobalContextInfo,
    PredictionConflictError,
    PredictionEvidenceV3,
    PredictionRequest,
    PredictionResponse,
)
from market_predictor.serving.snapshot_store import PredictionSnapshotStore


class PredictionSnapshotStoreTests(unittest.TestCase):
    def test_prediction_contracts_reject_non_finite_numbers(self) -> None:
        with self.assertRaises(ValidationError):
            GlobalContextInfo(net_impact=float("nan"))

    def test_writer_rejects_non_finite_bypassed_payload(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            response = _response(
                datetime(2026, 7, 21, 20, 15, tzinfo=UTC)
            ).model_copy(update={"errors": [float("nan")]})

            with self.assertRaises(PredictionConflictError):
                store.record(
                    PredictionRequest(tickers=["MSFT"], mode="swing"),
                    response,
                )
    def test_records_and_loads_content_addressed_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            request = PredictionRequest(tickers=["MSFT"], mode="swing")
            cutoff = datetime(2026, 7, 21, 20, 15, tzinfo=UTC)
            response = _response(cutoff)

            recorded = store.record(request, response)
            loaded_request, loaded_response, envelope = store.load(recorded.snapshot_id or "")

            self.assertEqual(len(recorded.snapshot_id or ""), 64)
            self.assertEqual(recorded.snapshot_id, recorded.snapshot_sha256)
            self.assertEqual(loaded_request.tickers, ["MSFT"])
            self.assertEqual(loaded_request.as_of, cutoff)
            self.assertEqual(loaded_response.snapshot_id, recorded.snapshot_id)
            self.assertEqual(loaded_response.evidence, response.evidence)
            self.assertEqual(envelope["content_sha256"], recorded.snapshot_id)
            self.assertEqual(
                envelope["content"]["response"]["evidence"]["prediction_cutoff_utc"],
                cutoff.isoformat().replace("+00:00", "Z"),
            )
            self.assertTrue(store.path_for(recorded.snapshot_id or "").exists())

    def test_detects_snapshot_tampering(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            recorded = store.record(
                PredictionRequest(tickers=["MSFT"], mode="swing"),
                _response(datetime(2026, 7, 21, 20, 15, tzinfo=UTC)),
            )
            path = store.path_for(recorded.snapshot_id or "")
            envelope = json.loads(path.read_text(encoding="utf-8"))
            envelope["content"]["request"]["tickers"] = ["AAPL"]
            path.write_text(json.dumps(envelope), encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                store.load(recorded.snapshot_id or "")

    def test_rejects_duplicate_snapshot_keys(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            recorded = store.record(
                PredictionRequest(tickers=["MSFT"], mode="swing"),
                _response(datetime(2026, 7, 21, 20, 15, tzinfo=UTC)),
            )
            path = store.path_for(recorded.snapshot_id or "")
            ambiguous = path.read_text(encoding="utf-8").replace(
                '"schema":',
                '"schema": "untrusted", "schema":',
                1,
            )
            path.write_text(ambiguous, encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                store.load(recorded.snapshot_id or "")


def _response(cutoff: datetime) -> PredictionResponse:
    request_id = "request-1"
    return PredictionResponse(
        request_id=request_id,
        mode="swing",
        horizon="10b",
        evidence=PredictionEvidenceV3(
            request_id=request_id,
            correlation_id="correlation-1",
            prediction_cutoff_utc=cutoff,
            view_prediction_policy_sha256={"swing": "b" * 64},
            serving_policy_id="market_predictor.serving_policy_bundle.v2",
            serving_policy_sha256="a" * 64,
            identity_status="research_only",
        ),
    )


if __name__ == "__main__":
    unittest.main()
