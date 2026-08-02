"""R3 P1-7: one frozen label/cost config per dataset, content-addressed."""

from __future__ import annotations

import unittest

from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.swing.contracts import SwingDatasetConfig


class LabelConfigHashTest(unittest.TestCase):
    def test_hash_is_deterministic_and_config_sensitive(self) -> None:
        self.assertEqual(SwingDatasetConfig().label_config_sha256(), SwingDatasetConfig().label_config_sha256())
        self.assertNotEqual(
            SwingDatasetConfig().label_config_sha256(),
            SwingDatasetConfig(round_trip_cost_bps=20.0).label_config_sha256(),
        )
        self.assertNotEqual(
            IntradayDatasetConfig().label_config_sha256(),
            IntradayDatasetConfig(target_atr=2.0).label_config_sha256(),
        )

if __name__ == "__main__":
    unittest.main()
