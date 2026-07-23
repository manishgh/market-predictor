"""R3 P1-7: one frozen label/cost config per dataset, content-addressed."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.swing.contracts import SwingDatasetConfig, SwingTrainingConfig
from market_predictor.swing.model import train_swing_model
from market_predictor.v3.errors import SchemaMismatchError
from tests.test_swing_model import _training_dataset


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

    def test_mixed_label_config_fails_training(self) -> None:
        dataset = _training_dataset()
        dataset.loc[dataset.index[0], "dataset_label_config_sha256"] = "c" * 64  # a second, conflicting config
        config = SwingTrainingConfig(
            family="logistic",
            n_splits=3,
            min_train_sessions=30,
            min_train_rows=100,
            min_training_tickers=6,
            min_features=25,
            top_k=3,
            max_iter=150,
        )
        with TemporaryDirectory() as temp_dir, self.assertRaises(SchemaMismatchError):
            train_swing_model(
                dataset,
                model_out=Path(temp_dir) / "swing.joblib",
                dataset_sha256="a" * 64,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
