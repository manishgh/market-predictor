from __future__ import annotations

import unittest
from unittest.mock import patch

from market_predictor.admission import InferenceAdmissionController
from market_predictor.prediction_contracts import (
    PredictionCapacityError,
    PredictionMemoryPressureError,
)


class InferenceAdmissionControllerTests(unittest.TestCase):
    def test_rejects_projected_memory_before_admission(self) -> None:
        controller = InferenceAdmissionController(
            max_concurrent_requests=1,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.25,
        )
        with (
            patch(
                "market_predictor.admission.process_memory_snapshot",
                return_value=(int(3.5 * 1024**3), int(3.5 * 1024**3)),
            ),
            self.assertRaises(PredictionMemoryPressureError),
        ):
            with controller.lease(estimated_incremental_gib=0.5):
                self.fail("memory-pressure request must not be admitted")
        self.assertEqual(controller.snapshot().active_requests, 0)

    def test_rejects_unknown_memory_when_policy_requires_measurement(self) -> None:
        controller = InferenceAdmissionController(
            max_concurrent_requests=1,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.25,
            reject_unknown_memory=True,
        )
        with (
            patch(
                "market_predictor.admission.process_memory_snapshot",
                return_value=None,
            ),
            self.assertRaises(PredictionMemoryPressureError),
        ):
            with controller.lease(estimated_incremental_gib=0.5):
                self.fail("unknown-memory request must not be admitted")

    def test_rejects_concurrent_work_without_queueing(self) -> None:
        controller = InferenceAdmissionController(
            max_concurrent_requests=1,
            memory_budget_gib=4.0,
            memory_headroom_gib=0.25,
        )
        with (
            patch(
                "market_predictor.admission.process_memory_snapshot",
                return_value=(1024**2, 1024**2),
            ),
            controller.lease(estimated_incremental_gib=0.25),
        ):
            with self.assertRaises(PredictionCapacityError):
                with controller.lease(estimated_incremental_gib=0.25):
                    self.fail("concurrent request must not be queued")
        self.assertEqual(controller.snapshot().active_requests, 0)


if __name__ == "__main__":
    unittest.main()
