from __future__ import annotations

import hashlib
import importlib.util
import json
import tempfile
import unittest
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

from market_predictor.registry import manifest_path_for
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.features import V3_FEATURE_SCHEMA_VERSION
from market_predictor.v3.models import V3TrainingConfig, audit_feature_coverage, train_v3_model_suite
from market_predictor.v3.validation import V3PurgedWalkForwardSplit


class V3ModelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.data = _training_data()
        self.config = V3TrainingConfig(
            families=("B0", "B1", "B2", "D1"),
            n_splits=3,
            embargo_sessions=1,
            min_train_sessions=6,
            min_train_rows=80,
            min_features=5,
            min_fold_feature_non_null_rate=0.2,
            ticker_holdout_fraction=0.2,
            top_k=3,
            max_iter=20,
        )

    def test_fold_feature_coverage_removes_early_sparse_feature(self) -> None:
        folds = V3PurgedWalkForwardSplit(
            n_splits=3,
            embargo_sessions=1,
            min_train_sessions=6,
            min_train_rows=80,
        ).split(self.data)
        selected, audit = audit_feature_coverage(self.data, folds, minimum_rate=0.2)
        self.assertIn("return_1bar", selected)
        self.assertNotIn("ema_50", selected)
        sparse = audit[audit["feature"] == "ema_50"].iloc[0]
        self.assertFalse(bool(sparse["eligible"]))

    def test_baseline_suite_writes_candidate_manifests_and_holdout_evidence(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            report, predictions, _ = train_v3_model_suite(
                self.data,
                Path(directory),
                config=self.config,
            )
            self.assertEqual(set(report["models"]), {"B0", "B1", "B2", "D1"})
            self.assertEqual(len(report["ticker_holdout"]), 2)
            self.assertEqual(set(predictions["audit_scope"]), {"walk_forward", "ticker_holdout"})
            for family in self.config.families:
                artifact = Path(report["models"][family]["artifact_path"])
                self.assertTrue(artifact.exists())
                manifest = json.loads(manifest_path_for(artifact).read_text(encoding="utf-8"))
                self.assertEqual(manifest["status"], "candidate")
                self.assertEqual(manifest["validation_split"], "v3_session_grouped_purged_walk_forward")
                self.assertEqual(manifest["extra"]["random_seed"], 42)
                self.assertEqual(manifest["extra"]["label_config"]["bar_minutes"], 5)
                self.assertEqual(len(manifest["extra"]["universe_snapshot_ids"]), 16)

    def test_seeded_run_id_and_oof_scores_are_deterministic(self) -> None:
        config = self.config.model_copy(update={"families": ("B1",)})
        with tempfile.TemporaryDirectory() as first_directory, tempfile.TemporaryDirectory() as second_directory:
            first_report, first_predictions, _ = train_v3_model_suite(self.data, Path(first_directory), config=config)
            second_report, second_predictions, _ = train_v3_model_suite(self.data, Path(second_directory), config=config)
        self.assertEqual(first_report["models"]["B1"]["run_id"], second_report["models"]["B1"]["run_id"])
        np.testing.assert_allclose(first_predictions["score"], second_predictions["score"], rtol=0, atol=1e-12)

    def test_one_family_failure_does_not_discard_other_family_results(self) -> None:
        config = self.config.model_copy(update={"families": ("B0", "B1")})
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory)
            (output / "b1_candidate.joblib").write_text("existing", encoding="utf-8")
            report, predictions, _ = train_v3_model_suite(self.data, output, config=config)
        self.assertEqual(report["status"], "partial_failure")
        self.assertEqual(report["failed_families"], ["B1"])
        self.assertEqual(report["models"]["B0"]["status"], "complete")
        self.assertEqual(set(predictions["family"]), {"B0"})

    def test_training_rejects_naive_or_future_feature_timestamps(self) -> None:
        naive = self.data.copy()
        naive["decision_time_utc"] = naive["decision_time_utc"].map(lambda value: value.replace(tzinfo=None))
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(DataReadinessError):
            train_v3_model_suite(naive, Path(directory), config=self.config)
        future = self.data.copy()
        future["feature_available_at_utc"] = future["decision_time_utc"] + pd.Timedelta(minutes=1)
        with tempfile.TemporaryDirectory() as directory, self.assertRaises(DataReadinessError):
            train_v3_model_suite(future, Path(directory), config=self.config)

    @unittest.skipUnless(importlib.util.find_spec("xgboost") is not None, "xgboost ranking extra is not installed")
    def test_optional_r1_ranker_produces_grouped_oof_evidence(self) -> None:
        config = self.config.model_copy(update={"families": ("R1",), "max_iter": 10})
        with tempfile.TemporaryDirectory() as directory:
            report, predictions, _ = train_v3_model_suite(self.data, Path(directory), config=config)
        self.assertGreater(report["models"]["R1"]["walk_forward"]["ranking_groups"], 0)
        self.assertTrue(predictions["score"].notna().all())


def _training_data() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 1, 5)
    tickers = [f"T{index:02d}" for index in range(10)]
    label_config_json = '{"bar_minutes":5,"round_trip_cost_bps":10.0}'
    label_config_hash = hashlib.sha256(label_config_json.encode()).hexdigest()
    for session_offset in range(16):
        session = start + timedelta(days=session_offset)
        for query in range(2):
            decision = datetime.combine(session, time(15, query * 5), tzinfo=UTC)
            query_id = decision.isoformat()
            for ticker_index, ticker in enumerate(tickers):
                centered = ticker_index - 4.5
                momentum = centered / 100 + session_offset / 10_000
                rows.append(
                    {
                        "ticker": ticker,
                        "decision_time_utc": decision,
                        "feature_available_at_utc": decision,
                        "entry_time_utc": decision + timedelta(minutes=5),
                        "session_date_et": session,
                        "decision_group_id": query_id,
                        "universe_snapshot_id": f"snapshot-{session.isoformat()}",
                        "price_feed": "sip",
                        "ranking_target": momentum,
                        "ranking_grade": min(4, ticker_index // 2),
                        "ranking_group_size": len(tickers),
                        "stop_before_target": int(ticker_index < 3),
                        "overlap_weight": 0.5 if query else 1.0,
                        "feature_schema_version": V3_FEATURE_SCHEMA_VERSION,
                        "label_schema_version": "ml_v3.v1",
                        "label_config_json": label_config_json,
                        "label_config_hash": label_config_hash,
                        "return_1bar": momentum / 2,
                        "return_3bar": momentum,
                        "relative_volume_same_minute_20d": 1 + ticker_index / 10,
                        "dollar_volume": 1_000_000 + ticker_index * 100_000,
                        "rel_return_3bar_vs_qqq": momentum - 0.001,
                        "xs_rank_return_1bar": ticker_index / 9,
                        "xs_rank_return_3bar": ticker_index / 9,
                        "xs_rank_relative_volume_same_minute_20d": ticker_index / 9,
                        "xs_rank_rel_return_3bar_vs_qqq": ticker_index / 9,
                        "ema_50": momentum if session_offset >= 12 else np.nan,
                    }
                )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
