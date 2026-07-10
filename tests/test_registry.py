from __future__ import annotations

from datetime import date, timedelta
import json
from pathlib import Path
import tempfile
import unittest

import joblib
import pandas as pd

from market_predictor.registry import feature_schema_hash, manifest_path_for, promote_model_manifest, write_model_manifest


class ModelRegistryTests(unittest.TestCase):
    def test_writes_manifest_with_artifact_hash_and_feature_schema(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            features = ["return_1d", "volume_z20"]
            rows = [
                {
                    "ticker": "MSFT",
                    "date": date(2026, 1, 1) + timedelta(days=idx),
                    "return_1d": 0.01,
                    "volume_z20": 1.0,
                    "target": idx % 2,
                }
                for idx in range(6)
            ]

            manifest = write_model_manifest(
                model_path=path,
                model_type="unit_test",
                schema_version="unit.v1",
                target_col="target",
                features=features,
                training_data=pd.DataFrame(rows),
                metrics={"roc_auc": 0.6},
                validation_split="date_grouped_purged_walk_forward",
            )

            loaded = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))
            self.assertEqual(loaded["artifact_sha256"], manifest["artifact_sha256"])
            self.assertEqual(loaded["dataset"]["feature_schema_hash"], feature_schema_hash(features))
            self.assertEqual(loaded["status"], "candidate")

    def test_promotes_candidate_when_all_gates_pass(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            report_path = Path(tmp) / "promotion.json"
            joblib.dump({"model": "placeholder"}, path)
            features = ["return_1d", "volume_z20"]
            training = _training_frame(features, rows=100, tickers=20)
            write_model_manifest(
                model_path=path,
                model_type="unit_test",
                schema_version="unit.v1",
                target_col="target",
                features=features,
                training_data=training,
                metrics={"roc_auc": 0.7, "top_decile_lift": 2.2, "validated_rows": 80, "tickers": 20},
                validation_split="date_grouped_purged_walk_forward",
            )

            result = promote_model_manifest(
                model_path=path,
                metrics={
                    "roc_auc": 0.7,
                    "top_decile_lift": 2.2,
                    "validated_rows": 80,
                    "tickers": 20,
                    "validation_split": "date_grouped_purged_walk_forward",
                },
                alignment_audit=pd.DataFrame(
                    [
                        {
                            "ticker": "MSFT",
                            "events_without_feature_row": 0,
                            "pending_after_latest_feature_date": 0,
                            "missing_historical_feature_rows": 0,
                            "dates_with_news_count_mismatch": 0,
                        }
                    ]
                ),
                profitability_audit=_profitability_audit(),
                regime_audit=_regime_audit(),
                catalyst_audit=_catalyst_audit(),
                min_roc_auc=0.65,
                min_top_decile_lift=2.0,
                min_validated_rows=50,
                min_tickers=10,
                min_selected_trades=10,
                report_path=report_path,
            )

            self.assertTrue(result["passed"])
            loaded = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "promoted")
            self.assertTrue(report_path.exists())

    def test_rejects_dirty_alignment_without_changing_candidate_status(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            report_path = Path(tmp) / "rejection.json"
            joblib.dump({"model": "placeholder"}, path)
            features = ["return_1d", "volume_z20"]
            write_model_manifest(
                model_path=path,
                model_type="unit_test",
                schema_version="unit.v1",
                target_col="target",
                features=features,
                training_data=_training_frame(features, rows=100, tickers=20),
                metrics={"roc_auc": 0.7, "top_decile_lift": 2.2, "validated_rows": 80, "tickers": 20},
                validation_split="date_grouped_purged_walk_forward",
            )

            result = promote_model_manifest(
                model_path=path,
                metrics={
                    "roc_auc": 0.7,
                    "top_decile_lift": 2.2,
                    "validated_rows": 80,
                    "tickers": 20,
                    "validation_split": "date_grouped_purged_walk_forward",
                },
                alignment_audit=pd.DataFrame([{"ticker": "MSFT", "events_without_feature_row": 1}]),
                profitability_audit=_profitability_audit(),
                regime_audit=_regime_audit(),
                catalyst_audit=_catalyst_audit(),
                min_roc_auc=0.65,
                min_top_decile_lift=2.0,
                min_validated_rows=50,
                min_tickers=10,
                min_selected_trades=10,
                report_path=report_path,
            )

            self.assertFalse(result["passed"])
            self.assertIn("alignment errors", result["failures"][0])
            loaded = json.loads(manifest_path_for(path).read_text(encoding="utf-8"))
            self.assertEqual(loaded["status"], "candidate")

    def test_rejects_model_with_poor_profitability_even_when_auc_passes(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "model.joblib"
            joblib.dump({"model": "placeholder"}, path)
            features = ["return_1d", "volume_z20"]
            write_model_manifest(
                model_path=path,
                model_type="unit_test",
                schema_version="unit.v1",
                target_col="target",
                features=features,
                training_data=_training_frame(features, rows=100, tickers=20),
                metrics={"roc_auc": 0.7, "top_decile_lift": 2.2, "validated_rows": 80, "tickers": 20},
                validation_split="date_grouped_purged_walk_forward",
            )

            result = promote_model_manifest(
                model_path=path,
                metrics={
                    "roc_auc": 0.7,
                    "top_decile_lift": 2.2,
                    "validated_rows": 80,
                    "tickers": 20,
                    "validation_split": "date_grouped_purged_walk_forward",
                },
                alignment_audit=pd.DataFrame([{"ticker": "MSFT", "events_without_feature_row": 0}]),
                profitability_audit=pd.DataFrame(
                    [{"selected_trades": 50, "avg_trade_return": -0.002, "profit_factor": 0.8, "max_drawdown": 0.05}]
                ),
                regime_audit=_regime_audit(),
                catalyst_audit=_catalyst_audit(),
                min_roc_auc=0.65,
                min_top_decile_lift=2.0,
                min_validated_rows=50,
                min_tickers=10,
                min_selected_trades=10,
            )

            self.assertFalse(result["passed"])
            self.assertTrue(any("avg_trade_return" in failure for failure in result["failures"]))


def _training_frame(features: list[str], *, rows: int, tickers: int) -> pd.DataFrame:
    start = date(2026, 1, 1)
    records = []
    for idx in range(rows):
        row = {
            "ticker": f"T{idx % tickers:03d}",
            "date": start + timedelta(days=idx),
            "target": idx % 2,
        }
        for feature in features:
            row[feature] = float(idx % 7)
        records.append(row)
    return pd.DataFrame(records)


def _profitability_audit() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "selected_trades": 25,
                "avg_trade_return": 0.004,
                "profit_factor": 1.4,
                "max_drawdown": 0.04,
                "return_drawdown_ratio": 1.2,
                "negative_period_rate": 0.35,
            }
        ]
    )


def _regime_audit() -> pd.DataFrame:
    return pd.DataFrame([{"regimes_present": 3, "max_single_regime_share": 0.6}])


def _catalyst_audit() -> pd.DataFrame:
    return pd.DataFrame([{"has_catalyst_features": True, "alignment_error_total": 0, "low_relevance_event_rate": 0.0}])


if __name__ == "__main__":
    unittest.main()
