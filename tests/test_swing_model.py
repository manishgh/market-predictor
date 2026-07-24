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
from market_predictor.prediction_policy import parse_prediction_policy
from market_predictor.registry import load_model_manifest, verify_model_artifact
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_FEATURES,
    SWING_MODEL_TYPE,
    SwingDatasetConfig,
    SwingPromotionConfig,
    SwingTrainingConfig,
)
from market_predictor.swing.model import score_swing_frame, train_swing_model
from market_predictor.swing.promotion import (
    load_swing_training_evidence,
    promote_swing_model,
    write_swing_training_evidence,
)
from market_predictor.v3.errors import DataReadinessError, SchemaMismatchError
from tests.r4_fixtures import test_signing_material, trust_context_for_candidate


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
            signing_key, trust_store, signer_id = test_signing_material()
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
                    "--hypothesis-registry",
                    str(root / "trust"),
                    "--hypothesis-id",
                    "swing-cli-test",
                    "--shadow-bundle",
                    str(root / "trust" / "shadow" / "missing-shadow.json"),
                    "--build-identity",
                    "ci:test",
                    "--approver-identity",
                    "reviewer:test",
                    "--signing-private-key",
                    str(signing_key),
                    "--attestation-trust-store",
                    str(trust_store),
                    "--signer-id",
                    signer_id,
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
            bound_policy = parse_prediction_policy(
                result.metrics["prediction_policy"],
                expected_sha256=result.metrics["prediction_policy_sha256"],
            )
            self.assertEqual(bound_policy.swing_top_k, config.top_k)
            self.assertTrue(result.oof_predictions["swing_probability"].between(0, 1).all())
            self.assertTrue(result.ticker_holdout_predictions["swing_probability"].between(0, 1).all())
            for evidence in (result.oof_predictions, result.ticker_holdout_predictions):
                self.assertTrue(
                    {
                        "validation_fold",
                        "validation_scope",
                        "calibration_method",
                        "calibration_train_cutoff_utc",
                        "row_identity",
                    }.issubset(evidence.columns)
                )
                cutoff = pd.to_datetime(evidence["calibration_train_cutoff_utc"], utc=True)
                decision = pd.to_datetime(evidence["decision_time_utc"], utc=True)
                self.assertTrue(cutoff.lt(decision).all())
            validation_folds = result.fold_audit[
                result.fold_audit["record_type"].eq("validation_fold")
            ]
            self.assertTrue(
                pd.to_datetime(validation_folds["max_train_label_available_at_utc"], utc=True)
                .lt(pd.to_datetime(validation_folds["min_test_decision_time_utc"], utc=True))
                .all()
            )
            self.assertIn("calibration_seed_excluded", set(validation_folds["validation_status"]))
            excluded_folds = validation_folds.loc[
                validation_folds["validation_status"].eq("calibration_seed_excluded"),
                "fold",
            ].nunique()
            self.assertEqual(result.metrics["calibration_seed_folds_excluded"], excluded_folds)
            self.assertEqual(
                set(validation_folds["feature_set_sha256"]),
                {result.manifest["dataset"]["feature_schema_hash"]},
            )
            representation = result.fold_audit[
                result.fold_audit["record_type"].eq("holdout_representation")
            ]
            required = representation["required"].fillna(False).astype(bool)
            self.assertTrue(representation.loc[required, "represented"].astype(bool).all())
            self.assertEqual(result.profitability_audit.iloc[0]["phase"], "conservative")
            evidence_dir = Path(temp_dir) / "evidence"
            paths = write_swing_training_evidence(result, evidence_dir)
            self.assertTrue(all(path.exists() for path in paths.values()))
            with self.assertRaises(FileExistsError):
                write_swing_training_evidence(result, Path(temp_dir) / "evidence")

            mismatched_alignment = result.alignment_audit.copy()
            mismatched_alignment["model_run_id"] = "different-run"
            evidence = load_swing_training_evidence(evidence_dir, model_path)
            rejected = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, alignment_audit=mismatched_alignment),
                config=_permissive_promotion_config(),
            )
            self.assertFalse(rejected["passed"])
            self.assertTrue(any("model_run_id" in failure for failure in rejected["failures"]))
            self.assertEqual(load_model_manifest(model_path)["status"], "candidate")

            trust_context = trust_context_for_candidate(
                Path(temp_dir) / "trust",
                model_path=model_path,
                metrics=result.metrics,
                model_type=SWING_MODEL_TYPE,
            )
            promoted = promote_swing_model(
                model_path=model_path,
                evidence=evidence,
                config=_permissive_promotion_config(),
                trust_context=trust_context,
            )
            self.assertTrue(promoted["passed"], promoted["failures"])
            self.assertEqual(load_model_manifest(model_path)["status"], "candidate")
            self.assertEqual(verify_model_artifact(model_path, allowed_statuses={"promoted"})["status"], "promoted")

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

    def test_future_poison_cannot_change_earlier_causal_probabilities_or_features(self) -> None:
        baseline = _training_dataset()
        poison_start = sorted(baseline["session_date_et"].unique())[-25]
        poisoned = baseline.copy()
        future = poisoned["session_date_et"].ge(poison_start)
        poisoned.loc[future, "target_net_positive_5d"] = (
            1 - poisoned.loc[future, "target_net_positive_5d"].astype(int)
        )
        poisoned.loc[future, SWING_FEATURES[0]] = (
            pd.to_numeric(poisoned.loc[future, SWING_FEATURES[0]]) + 10_000.0
        )
        poisoned.loc[future, SWING_FEATURES[1]] = np.nan

        with TemporaryDirectory() as first_dir, TemporaryDirectory() as second_dir:
            first = train_swing_model(
                baseline,
                model_out=Path(first_dir) / "baseline.joblib",
                dataset_sha256="c" * 64,
                config=_training_config(),
            )
            second = train_swing_model(
                poisoned,
                model_out=Path(second_dir) / "poisoned.joblib",
                dataset_sha256="d" * 64,
                config=_training_config(),
            )

        self.assertEqual(
            first.manifest["extra"]["feature_set_sha256"],
            second.manifest["extra"]["feature_set_sha256"],
        )
        for left, right in (
            (first.oof_predictions, second.oof_predictions),
            (first.ticker_holdout_predictions, second.ticker_holdout_predictions),
        ):
            earlier = left[left["session_date_et"].lt(poison_start)]
            compare = earlier[["row_identity", "swing_probability"]].merge(
                right[["row_identity", "swing_probability"]],
                on="row_identity",
                suffixes=("_baseline", "_poisoned"),
                validate="one_to_one",
            )
            self.assertFalse(compare.empty)
            np.testing.assert_allclose(
                compare["swing_probability_baseline"],
                compare["swing_probability_poisoned"],
                rtol=0.0,
                atol=0.0,
            )


def _training_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    label_config = SwingDatasetConfig()
    label_policy_json = json.dumps(
        label_config.label_policy(),
        sort_keys=True,
        separators=(",", ":"),
    )
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
            net_return = (0.012 if target else -0.002) + float(rng.normal(0, 0.001))
            rows.append(
                {
                    "ticker": ticker,
                    "session_date_et": session.date(),
                    "decision_group_id": decision_time.isoformat(),
                    "decision_time_utc": decision_time,
                    "feature_available_at_utc": decision_time,
                    "label_available_at_utc": decision_time + pd.offsets.BDay(5),
                    "label_eligible": True,
                    "feature_eligible": True,
                    "label_window_expected": True,
                    "label_path_exact": True,
                    "horizon_sessions": 5,
                    "swing_feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
                    "reconciliation_sha256": "a" * 64,
                    "dataset_label_config_sha256": label_config.label_config_sha256(),
                    "dataset_label_policy_json": label_policy_json,
                    "market_regime": regime,
                    "sector": "Technology" if ticker_index % 2 == 0 else "Healthcare",
                    "market_cap_bucket": "large" if ticker_index < 6 else "mid",
                    "liquidity_bucket": "high" if ticker_index % 3 else "medium",
                    "primary_benchmark": "XLK" if ticker_index % 2 == 0 else "XLV",
                    "universe_snapshot_id": "snapshot-1",
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
        min_group_lift_at_k=0.0,
        min_ticker_holdout_group_lift_at_k=0.0,
        min_decision_groups=1,
        min_independent_sessions=1,
        min_validation_folds=1,
        min_stress_avg_trade_return=-1.0,
        min_stress_avg_excess_return_vs_spy=-1.0,
        min_worst_regime_avg_excess_return_vs_spy=-1.0,
        max_worst_regime_drawdown=1.0,
        max_worst_regime_calibration_error=1.0,
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
        max_calibration_error=1.0,
        max_ticker_holdout_calibration_error=1.0,
        max_calibration_bias=1.0,
        min_calibration_slope=0.0,
        max_calibration_slope=100.0,
        max_abs_calibration_intercept=1.0,
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
