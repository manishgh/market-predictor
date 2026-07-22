from __future__ import annotations

import json
import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import write_canonical_artifact
from market_predictor.cli import app
from market_predictor.registry import load_model_manifest
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_FEATURES,
    SWING_MODEL_TYPE,
    SwingPromotionConfig,
    SwingTrainingConfig,
)
from market_predictor.swing.model import score_swing_frame, train_swing_model
from market_predictor.swing.promotion import (
    load_swing_training_evidence,
    promote_swing_model,
    promotion_evidence_from_result,
    write_swing_training_evidence,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError


class SwingModelTests(unittest.TestCase):
    def test_cli_trains_and_rejects_tampered_evidence(self) -> None:
        dataset = _training_dataset()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset_path = root / "swing_dataset.parquet"
            model_path = root / "swing_candidate.joblib"
            evidence_dir = root / "evidence"
            training_config = root / "training.json"
            promotion_config = root / "promotion.json"
            write_canonical_artifact(
                dataset,
                dataset_path,
                artifact_type="swing_dataset",
                audit=_passing_audit(len(dataset)),
            )
            training_config.write_text(json.dumps(_training_config().model_dump()), encoding="utf-8")
            promotion_config.write_text(json.dumps(_permissive_promotion_config().model_dump()), encoding="utf-8")
            runner = CliRunner()
            trained = runner.invoke(
                app,
                [
                    "train-swing-model",
                    "--dataset",
                    str(dataset_path),
                    "--model-out",
                    str(model_path),
                    "--evidence-dir",
                    str(evidence_dir),
                    "--config",
                    str(training_config),
                ],
            )
            self.assertEqual(trained.exit_code, 0, msg=f"{trained.output}\n{trained.exception}")
            evidence = load_swing_training_evidence(evidence_dir, model_path)
            self.assertEqual(evidence.provenance, "hash_verified_evidence_bundle")
            profitability_path = evidence_dir / "profitability.csv"
            profitability = pd.read_csv(profitability_path)
            profitability.loc[0, "return_drawdown_ratio"] = 1.0
            profitability.to_csv(profitability_path, index=False)
            promoted = runner.invoke(
                app,
                [
                    "promote-swing-model",
                    "--model",
                    str(model_path),
                    "--evidence-dir",
                    str(evidence_dir),
                    "--config",
                    str(promotion_config),
                ],
            )
            self.assertNotEqual(promoted.exit_code, 0)
            self.assertIsInstance(promoted.exception, DataReadinessError)
            self.assertEqual(load_model_manifest(model_path)["status"], "candidate")

    def test_trains_walk_forward_and_unseen_ticker_candidate(self) -> None:
        dataset = _training_dataset()
        config = SwingTrainingConfig(
            family="logistic",
            n_splits=3,
            min_train_sessions=30,
            min_train_rows=100,
            min_training_tickers=6,
            min_features=25,
            ticker_holdout_fraction=0.2,
            top_k=3,
            max_iter=150,
            max_training_memory_gb=4.0,
        )
        with TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "swing_candidate.joblib"
            result = train_swing_model(
                dataset,
                model_out=model_path,
                dataset_sha256="a" * 64,
                config=config,
            )
            manifest = load_model_manifest(model_path)
            self.assertEqual(manifest["status"], "candidate")
            self.assertEqual(manifest["model_type"], SWING_MODEL_TYPE)
            self.assertGreater(result.metrics["validated_rows"], 100)
            self.assertGreater(result.metrics["ticker_holdout_rows"], 100)
            self.assertIn("feature_reference_profile", result.metrics)
            self.assertTrue(result.oof_predictions["swing_probability"].between(0, 1).all())
            self.assertTrue(result.ticker_holdout_predictions["swing_probability"].between(0, 1).all())
            self.assertEqual(result.profitability_audit.iloc[0]["phase"], "conservative")
            paths = write_swing_training_evidence(result, Path(temp_dir) / "evidence")
            self.assertTrue(all(path.exists() for path in paths.values()))
            with self.assertRaises(FileExistsError):
                write_swing_training_evidence(result, Path(temp_dir) / "evidence")

            mismatched_alignment = result.alignment_audit.copy()
            mismatched_alignment["model_run_id"] = "different-run"
            evidence = promotion_evidence_from_result(result)
            rejected = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, alignment_audit=mismatched_alignment),
                config=_permissive_promotion_config(),
            )
            self.assertFalse(rejected["passed"])
            self.assertTrue(any("model_run_id" in failure for failure in rejected["failures"]))
            self.assertEqual(load_model_manifest(model_path)["status"], "candidate")

            promotable_profitability = result.profitability_audit.copy()
            promotable_profitability.loc[0, "return_drawdown_ratio"] = 1.0
            promoted = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, profitability_audit=promotable_profitability),
                config=_permissive_promotion_config(),
            )
            self.assertTrue(promoted["passed"], promoted["failures"])
            self.assertEqual(load_model_manifest(model_path)["status"], "promoted")

            latest = dataset.sort_values("decision_time_utc").groupby("ticker", as_index=False).tail(1)
            scored = score_swing_frame(latest, model_path, require_promoted=True)
            self.assertTrue(scored["swing_model_probability"].between(0, 1).all())
            self.assertEqual(set(scored["swing_model_schema"]), {"swing.model.v1"})

            invalid = latest.copy()
            invalid["swing_feature_schema_version"] = "wrong"
            with self.assertRaises(SchemaMismatchError):
                score_swing_frame(invalid, model_path)

    def test_rejects_future_feature_timestamp(self) -> None:
        dataset = _training_dataset()
        dataset.loc[dataset.index[0], "feature_available_at_utc"] = (
            dataset.loc[dataset.index[0], "decision_time_utc"] + pd.Timedelta(seconds=1)
        )
        with TemporaryDirectory() as temp_dir, self.assertRaises(DataReadinessError):
            train_swing_model(
                dataset,
                model_out=Path(temp_dir) / "invalid.joblib",
                dataset_sha256="b" * 64,
                config=SwingTrainingConfig(
                    family="logistic",
                    n_splits=2,
                    min_train_sessions=30,
                    min_train_rows=100,
                    min_training_tickers=6,
                    min_features=25,
                    max_iter=100,
                ),
            )


def _training_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    sessions = pd.bdate_range("2024-01-02", periods=180)
    tickers = [f"T{index:02d}" for index in range(12)]
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        decision_time = pd.Timestamp(session, tz="America/New_York") + pd.Timedelta(hours=16, minutes=15)
        decision_time = decision_time.tz_convert("UTC")
        regime = ("risk_on", "neutral", "risk_off")[session_index % 3]
        for ticker_index, ticker in enumerate(tickers):
            feature_values = rng.normal(0.0, 1.0, len(SWING_FEATURES)).astype(float)
            feature_map = dict(zip(SWING_FEATURES, feature_values, strict=True))
            signal = 0.9 * feature_map["return_20d"] + 0.6 * feature_map["volume_z20"] + rng.normal(0, 0.8)
            target = int(signal > 0)
            net_return = (0.012 if target else -0.010) + float(rng.normal(0, 0.004))
            rows.append(
                {
                    "ticker": ticker,
                    "session_date_et": session.date(),
                    "decision_group_id": decision_time.isoformat(),
                    "decision_time_utc": decision_time,
                    "feature_available_at_utc": decision_time,
                    "label_eligible": True,
                    "horizon_sessions": 5,
                    "swing_feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
                    "market_regime": regime,
                    "sector": "Technology" if ticker_index % 2 == 0 else "Healthcare",
                    "primary_benchmark": "XLK" if ticker_index % 2 == 0 else "XLV",
                    "event_count_3d": max(0.0, feature_map["event_count_3d"]),
                    "event_relevance_mean_3d": 0.8,
                    "low_relevance_event_fraction_3d": 0.05,
                    "target_net_positive_5d": target,
                    "future_net_return_5d": net_return,
                    "future_excess_return_5d_vs_spy": net_return - 0.001,
                    "future_excess_return_5d_vs_qqq": net_return - 0.0015,
                    "future_excess_return_5d_vs_sector": net_return - 0.0005,
                    **feature_map,
                }
            )
    return pd.DataFrame(rows)


def _permissive_promotion_config() -> SwingPromotionConfig:
    return SwingPromotionConfig(
        min_roc_auc=0.5,
        min_ticker_holdout_roc_auc=0.5,
        min_top_decile_lift=1.0,
        min_ticker_holdout_lift=1.0,
        min_validated_rows=100,
        min_tickers=2,
        min_selected_trades=1,
        min_avg_trade_return=-1.0,
        min_avg_excess_return_vs_spy=-1.0,
        min_avg_excess_return_vs_qqq=-1.0,
        min_avg_excess_return_vs_sector=-1.0,
        min_profit_factor=0.0,
        max_drawdown=1.0,
        min_return_drawdown_ratio=0.0,
        max_negative_period_rate=1.0,
        min_regimes=1,
        max_single_regime_share=1.0,
        min_catalyst_row_rate=0.0,
        max_low_relevance_event_rate=1.0,
        max_peak_working_set_gib=4.0,
    )


def _training_config() -> SwingTrainingConfig:
    return SwingTrainingConfig(
        family="logistic",
        n_splits=3,
        min_train_sessions=30,
        min_train_rows=100,
        min_training_tickers=6,
        min_features=25,
        ticker_holdout_fraction=0.2,
        top_k=3,
        max_iter=150,
        max_training_memory_gb=4.0,
    )


def _passing_audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="fixture",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="test fixture",
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
