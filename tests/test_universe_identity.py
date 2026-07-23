"""R3 P1-8: swing rows must carry point-in-time universe identity."""

from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.swing.contracts import SwingTrainingConfig
from market_predictor.swing.model import train_swing_model
from market_predictor.v3.errors import DataReadinessError
from tests.test_swing_model import _training_dataset


class SwingUniverseIdentityTest(unittest.TestCase):
    def test_missing_universe_identity_fails_training(self) -> None:
        dataset = _training_dataset()
        dataset["universe_snapshot_id"] = None  # a present-survivor universe with no snapshot identity
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
        with TemporaryDirectory() as temp_dir, self.assertRaises(DataReadinessError):
            train_swing_model(
                dataset,
                model_out=Path(temp_dir) / "swing.joblib",
                dataset_sha256="a" * 64,
                config=config,
            )


if __name__ == "__main__":
    unittest.main()
