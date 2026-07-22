from __future__ import annotations

import unittest

import numpy as np
import pandas as pd

from market_predictor.drift import audit_feature_drift, build_feature_reference_profile


class FeatureDriftTests(unittest.TestCase):
    def test_reports_stable_and_severe_live_distributions(self) -> None:
        training = pd.DataFrame(
            {
                "momentum": np.linspace(-1.0, 1.0, 101),
                "volume": np.linspace(10.0, 20.0, 101),
            }
        )
        reference = build_feature_reference_profile(training, ["momentum", "volume"])

        stable = audit_feature_drift(training.tail(20), reference)
        shifted = audit_feature_drift(
            pd.DataFrame({"momentum": [20.0, 21.0], "volume": [None, None]}),
            reference,
        )

        self.assertEqual(stable["status"], "stable")
        self.assertEqual(shifted["status"], "severe")
        self.assertGreaterEqual(int(shifted["severe_feature_count"]), 1)

    def test_reports_unavailable_without_comparable_reference(self) -> None:
        self.assertEqual(audit_feature_drift(pd.DataFrame({"x": [1.0]}), None)["status"], "unavailable")
        result = audit_feature_drift(
            pd.DataFrame({"x": [1.0]}),
            {"different": {"mean": 0.0, "std": 1.0, "missing_rate": 0.0}},
        )
        self.assertEqual(result["status"], "unavailable")


if __name__ == "__main__":
    unittest.main()
