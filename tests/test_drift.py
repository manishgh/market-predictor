from __future__ import annotations

import unittest
from datetime import UTC, datetime, timedelta

import numpy as np
import pandas as pd
from pydantic import ValidationError

from market_predictor.governance.drift.features import (
    audit_feature_drift,
    validate_feature_drift_report,
)
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.modeling.feature_reference import build_feature_reference_profile


class FeatureDriftTests(unittest.TestCase):
    def test_reports_stable_and_severe_live_distributions(self) -> None:
        training = pd.DataFrame(
            {
                "momentum": np.linspace(-1.0, 1.0, 101),
                "volume": np.linspace(10.0, 20.0, 101),
            }
        )
        reference = build_feature_reference_profile(training, ["momentum", "volume"])

        stable = self._audit(training.tail(20), reference)
        shifted = audit_feature_drift(
            pd.DataFrame({"momentum": [20.0, 21.0], "volume": [None, None]}),
            reference,
            **self._identity(),
        )

        self.assertEqual(stable["status"], "stable")
        self.assertEqual(shifted["status"], "severe")
        self.assertGreaterEqual(int(shifted["severe_feature_count"]), 1)

    def test_reports_unavailable_without_comparable_reference(self) -> None:
        self.assertEqual(
            self._audit(pd.DataFrame({"x": [1.0]}), None)["status"],
            "unavailable",
        )
        result = audit_feature_drift(
            pd.DataFrame({"x": [1.0]}),
            {"different": {"mean": 0.0, "std": 1.0, "missing_rate": 0.0}},
            **self._identity(),
        )
        self.assertEqual(result["status"], "unavailable")

    def test_requires_complete_model_feature_coverage(self) -> None:
        reference = build_feature_reference_profile(
            pd.DataFrame({"momentum": [1.0], "volume": [2.0]}),
            ["momentum", "volume"],
        )

        result = self._audit(pd.DataFrame({"momentum": [1.0]}), reference)

        self.assertEqual(result["status"], "unavailable")
        self.assertIn("volume:missing_live_feature", str(result["reason"]))

    def test_report_rejects_status_that_disagrees_with_bound_thresholds(self) -> None:
        training = pd.DataFrame({"momentum": np.linspace(-1.0, 1.0, 101)})
        reference = build_feature_reference_profile(training, ["momentum"])
        severe = self._audit(pd.DataFrame({"momentum": [20.0]}), reference)
        tampered = dict(severe)
        tampered["status"] = "stable"
        tampered["report_id"] = content_sha256(
            {key: value for key, value in tampered.items() if key != "report_id"}
        )

        with self.assertRaises(ValidationError):
            validate_feature_drift_report(tampered)

    def test_report_rejects_duplicate_or_renamed_feature_rows(self) -> None:
        reference = build_feature_reference_profile(
            pd.DataFrame({"momentum": [1.0], "volume": [2.0]}),
            ["momentum", "volume"],
        )
        valid = self._audit(
            pd.DataFrame({"momentum": [1.0], "volume": [2.0]}),
            reference,
        )
        for names in (("momentum", "momentum"), ("momentum", "renamed")):
            tampered = dict(valid)
            rows = [dict(row) for row in valid["feature_rows"]]
            for row, name in zip(rows, names, strict=True):
                row["feature"] = name
            tampered["feature_rows"] = rows
            tampered["report_id"] = content_sha256(
                {key: value for key, value in tampered.items() if key != "report_id"}
            )
            with self.assertRaises(ValidationError):
                validate_feature_drift_report(tampered)

    def _audit(
        self,
        frame: pd.DataFrame,
        reference: dict[str, object] | None,
    ) -> dict[str, object]:
        return audit_feature_drift(frame, reference, **self._identity())

    @staticmethod
    def _identity() -> dict[str, object]:
        now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        return {
            "mode": "swing",
            "horizon": "10b",
            "model_release_id": "a" * 64,
            "model_artifact_sha256": "b" * 64,
            "feature_artifact_set_sha256": "f" * 64,
            "window_start": now - timedelta(hours=1),
            "window_end": now,
            "generated_at": now,
            "source_artifact_sha256": "c" * 64,
        }


if __name__ == "__main__":
    unittest.main()
