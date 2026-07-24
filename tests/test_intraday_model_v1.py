from __future__ import annotations

import json
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd

from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_FEATURES,
    INTRADAY_MODEL_TYPE,
    IntradayDatasetConfig,
    IntradayPromotionConfig,
    IntradayTrainingConfig,
    downside_target_column,
    excess_return_column,
    net_return_column,
    opportunity_target_column,
)
from market_predictor.intraday.model import score_intraday_frame, train_intraday_model
from market_predictor.intraday.promotion import (
    load_intraday_training_evidence,
    promote_intraday_model,
    promotion_evidence_from_result,
    write_intraday_training_evidence,
)
from market_predictor.prediction_policy import parse_prediction_policy
from market_predictor.registry import verify_model_artifact
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import trust_context_for_candidate


class IntradayModelV1Tests(unittest.TestCase):
    def test_trains_atomic_opportunity_and_downside_candidate(self) -> None:
        dataset = _training_dataset()
        config = _training_config()
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "intraday_60m.joblib"

            result = train_intraday_model(
                dataset,
                model_out=model_path,
                dataset_sha256="a" * 64,
                config=config,
            )
            manifest = verify_model_artifact(model_path, allowed_statuses={"candidate"})
            scored = score_intraday_frame(dataset.tail(10), model_path)

        self.assertEqual(manifest["model_type"], "canonical_intraday")
        self.assertEqual(result.metrics["validation_split"], "session_purged_walk_forward_and_ticker_holdout")
        self.assertIn("feature_reference_profile", result.metrics)
        bound_policy = parse_prediction_policy(
            result.metrics["prediction_policy"],
            expected_sha256=result.metrics["prediction_policy_sha256"],
        )
        self.assertEqual(bound_policy.intraday_top_k, config.top_k)
        self.assertEqual(
            bound_policy.intraday_downside_ceiling,
            config.max_downside_probability,
        )
        self.assertEqual(
            bound_policy.intraday_max_trades_per_session,
            config.max_trades_per_session,
        )
        self.assertFalse(result.oof_predictions.empty)
        self.assertFalse(result.ticker_holdout_predictions.empty)
        self.assertIn("intraday_opportunity_probability", scored.columns)
        self.assertIn("intraday_downside_probability", scored.columns)
        self.assertTrue(scored["intraday_opportunity_probability"].between(0, 1).all())
        self.assertTrue(scored["intraday_downside_probability"].between(0, 1).all())
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
        excluded_folds = validation_folds.loc[
            validation_folds["validation_status"].eq("calibration_seed_excluded"),
            "fold",
        ].nunique()
        scored_fold_ids = sorted(
            int(fold)
            for fold in validation_folds.loc[
                validation_folds["validation_status"].eq("included"),
                "fold",
            ].unique()
        )
        self.assertEqual(result.metrics["calibration_seed_folds_excluded"], excluded_folds)
        self.assertEqual(result.metrics["validation_folds"], len(scored_fold_ids))
        self.assertEqual(result.metrics["scored_validation_fold_ids"], scored_fold_ids)
        self.assertEqual(
            result.metrics["configured_validation_folds"],
            result.metrics["validation_folds"] + excluded_folds,
        )
        self.assertEqual(
            set(validation_folds["feature_set_sha256"]),
            {result.manifest["dataset"]["feature_schema_hash"]},
        )
        self.assertEqual(result.catalyst_audit.iloc[0]["included_in_estimators"], False)
        self.assertLess(float(result.metrics["memory"]["peak_working_set_gib"]), 4.0)

    def test_rejects_future_feature_timestamp(self) -> None:
        dataset = _training_dataset()
        dataset.loc[dataset.index[-1], "feature_available_at_utc"] = dataset.loc[dataset.index[-1], "decision_time_utc"] + pd.Timedelta(
            minutes=1
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            with self.assertRaisesRegex(DataReadinessError, "future or invalid timestamps"):
                train_intraday_model(
                    dataset,
                    model_out=Path(temp_dir) / "intraday.joblib",
                    dataset_sha256="b" * 64,
                    config=_training_config(),
                )

    def test_hash_bound_evidence_and_atomic_promotion(self) -> None:
        dataset = _training_dataset()
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model_path = root / "intraday.joblib"
            result = train_intraday_model(
                dataset,
                model_out=model_path,
                dataset_sha256="c" * 64,
                config=_training_config(),
            )
            evidence_dir = root / "evidence"
            paths = write_intraday_training_evidence(result, evidence_dir)
            original_profitability = paths["profitability"].read_bytes()
            with paths["profitability"].open("a", encoding="utf-8") as handle:
                handle.write("\nmodified")
            with self.assertRaisesRegex(DataReadinessError, "integrity check failed"):
                load_intraday_training_evidence(evidence_dir, model_path)
            paths["profitability"].write_bytes(original_profitability)
            loaded = load_intraday_training_evidence(evidence_dir, model_path)
            substituted_profitability = loaded.profitability_audit.copy()
            substituted_profitability.loc[0, "avg_trade_return"] = 1.0
            rejected = promote_intraday_model(
                model_path=model_path,
                evidence=replace(
                    loaded,
                    profitability_audit=substituted_profitability,
                ),
                config=IntradayPromotionConfig(
                    min_validated_rows=100,
                    min_tickers=6,
                    min_selected_trades=1,
                    min_catalyst_coverage_rate=0.10,
                ),
            )
            self.assertFalse(rejected["passed"])
            self.assertTrue(
                any(
                    "differs from its persisted bundle" in failure
                    for failure in rejected["failures"]
                )
            )

            report = promote_intraday_model(
                model_path=model_path,
                evidence=loaded,
                trust_context=trust_context_for_candidate(
                    root / "trust",
                    model_path=model_path,
                    metrics=result.metrics,
                    model_type=INTRADAY_MODEL_TYPE,
                ),
                config=IntradayPromotionConfig(
                    min_validated_rows=100,
                    min_tickers=6,
                    min_selected_trades=1,
                    min_catalyst_coverage_rate=0.10,
                    min_opportunity_group_lift_at_k=0.0,
                    min_opportunity_holdout_group_lift_at_k=0.0,
                    min_decision_groups=1,
                    min_independent_sessions=1,
                    min_validation_folds=1,
                    min_effective_sample_size=0.0,
                    min_stress_avg_trade_return=-1.0,
                    min_stress_avg_excess_return_vs_spy=-1.0,
                    min_worst_regime_avg_excess_return_vs_spy=-1.0,
                    max_worst_regime_drawdown=1.0,
                    max_worst_regime_calibration_error=1.0,
                    min_capacity_avg_net_return=-1.0,
                ),
            )

            self.assertTrue(report["passed"], report["failures"])
            verify_model_artifact(model_path, allowed_statuses={"promoted"})

    def test_default_promotion_rejects_small_synthetic_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            model_path = Path(temp_dir) / "intraday.joblib"
            result = train_intraday_model(
                _training_dataset(),
                model_out=model_path,
                dataset_sha256="d" * 64,
                config=_training_config(),
            )

            report = promote_intraday_model(
                model_path=model_path,
                evidence=promotion_evidence_from_result(result),
            )

            self.assertFalse(report["passed"])
            self.assertTrue(any("validated_rows" in failure for failure in report["failures"]))
            verify_model_artifact(model_path, allowed_statuses={"candidate"})

    def test_alignment_evidence_includes_rows_excluded_from_training(self) -> None:
        dataset = _training_dataset()
        excluded = dataset.index[-1]
        dataset.loc[excluded, "label_eligible"] = False
        dataset.loc[excluded, "label_path_exact"] = False

        with tempfile.TemporaryDirectory() as temp_dir:
            result = train_intraday_model(
                dataset,
                model_out=Path(temp_dir) / "intraday.joblib",
                dataset_sha256="e" * 64,
                config=_training_config(),
            )

        self.assertEqual(int(result.alignment_audit.iloc[0]["label_path_mismatches"]), 1)
        self.assertEqual(int(result.alignment_audit.iloc[0]["alignment_error_total"]), 1)

    def test_future_poison_cannot_change_earlier_causal_probabilities_or_features(self) -> None:
        baseline = _training_dataset()
        poison_start = sorted(baseline["session_date_et"].unique())[-3]
        poisoned = baseline.copy()
        future = poisoned["session_date_et"].ge(poison_start)
        for target in (opportunity_target_column(60), downside_target_column(60)):
            poisoned.loc[future, target] = 1 - poisoned.loc[future, target].astype(int)
        poisoned.loc[future, INTRADAY_MODEL_FEATURES[0]] = (
            pd.to_numeric(poisoned.loc[future, INTRADAY_MODEL_FEATURES[0]]) + 10_000.0
        )
        poisoned.loc[future, INTRADAY_MODEL_FEATURES[1]] = np.nan

        with tempfile.TemporaryDirectory() as first_dir, tempfile.TemporaryDirectory() as second_dir:
            first = train_intraday_model(
                baseline,
                model_out=Path(first_dir) / "baseline.joblib",
                dataset_sha256="f" * 64,
                config=_training_config(),
            )
            second = train_intraday_model(
                poisoned,
                model_out=Path(second_dir) / "poisoned.joblib",
                dataset_sha256="0" * 64,
                config=_training_config(),
            )

        self.assertEqual(
            first.manifest["extra"]["feature_set_sha256"],
            second.manifest["extra"]["feature_set_sha256"],
        )
        probability_columns = [
            "intraday_opportunity_probability",
            "intraday_downside_probability",
        ]
        for left, right in (
            (first.oof_predictions, second.oof_predictions),
            (first.ticker_holdout_predictions, second.ticker_holdout_predictions),
        ):
            earlier = left[left["session_date_et"].lt(poison_start)]
            compare = earlier[["row_identity", *probability_columns]].merge(
                right[["row_identity", *probability_columns]],
                on="row_identity",
                suffixes=("_baseline", "_poisoned"),
                validate="one_to_one",
            )
            self.assertFalse(compare.empty)
            for column in probability_columns:
                np.testing.assert_allclose(
                    compare[f"{column}_baseline"],
                    compare[f"{column}_poisoned"],
                    rtol=0.0,
                    atol=0.0,
                )


def _training_config() -> IntradayTrainingConfig:
    return IntradayTrainingConfig(
        family="logistic",
        n_splits=2,
        min_train_sessions=20,
        min_train_rows=100,
        min_training_tickers=6,
        min_features=8,
        ticker_holdout_fraction=0.2,
        top_k=3,
        max_trades_per_session=6,
        max_iter=150,
    )


def _training_dataset() -> pd.DataFrame:
    rng = np.random.default_rng(42)
    label_config = IntradayDatasetConfig()
    label_policy_json = json.dumps(
        label_config.label_policy(),
        sort_keys=True,
        separators=(",", ":"),
    )
    sessions = pd.bdate_range("2025-01-02", periods=30)
    tickers = [f"T{index:02d}" for index in range(10)]
    rows: list[dict[str, object]] = []
    horizon = 60
    for session_index, session in enumerate(sessions):
        for group_index in range(6):
            decision = pd.Timestamp(session, tz="America/New_York") + pd.Timedelta(
                hours=10,
                minutes=group_index * 15,
            )
            decision = decision.tz_convert("UTC")
            group = decision.isoformat()
            for ticker_index, ticker in enumerate(tickers):
                technical = (
                    0.8 * np.sin((session_index + ticker_index) / 4.0)
                    + 0.6 * np.cos((group_index + ticker_index) / 3.0)
                    + rng.normal(0, 0.5)
                )
                opportunity = int(technical + rng.normal(0, 0.7) > 0.15)
                downside = int(technical + rng.normal(0, 0.7) < -0.25)
                if opportunity and downside:
                    downside = 0
                net_return = 0.004 * technical + rng.normal(0, 0.002)
                row: dict[str, object] = {
                    "ticker": ticker,
                    "session_date_et": session.date(),
                    "decision_group_id": group,
                    "decision_time_utc": decision,
                    "feature_available_at_utc": decision,
                    "entry_time_utc": decision,
                    "exit_time_utc": decision + pd.Timedelta(minutes=horizon),
                    "label_available_at_utc": decision + pd.Timedelta(minutes=horizon),
                    "label_window_end_utc": decision + pd.Timedelta(minutes=horizon),
                    "feature_eligible": True,
                    "label_eligible": True,
                    "label_window_expected": True,
                    "label_path_exact": True,
                    "horizon_minutes": horizon,
                    "decision_bar_minutes": 5,
                    "decision_stride_bars": 3,
                    "overlap_weight": 1.0,
                    "concurrent_label_count": 1,
                    "independent_event_id": f"{ticker}:{session.date()}:{group_index}",
                    "intraday_feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
                    "reconciliation_sha256": "a" * 64,
                    "dataset_label_config_sha256": label_config.label_config_sha256(),
                    "dataset_label_policy_json": label_policy_json,
                    "market_regime": ["risk_on", "risk_off", "neutral"][session_index % 3],
                    "sector": "Technology",
                    "market_cap_bucket": "large" if ticker_index < 5 else "mid",
                    "liquidity_bucket": "high" if ticker_index % 2 else "medium",
                    "primary_benchmark": "XLK",
                    "catalyst_eligible": session_index % 2 == 0,
                    "event_count_2h": int(session_index % 5 == 0),
                    "event_relevance_mean_2h": 1.0,
                    "low_relevance_event_fraction_2h": 0.0,
                    opportunity_target_column(horizon): opportunity,
                    downside_target_column(horizon): downside,
                    net_return_column(horizon): net_return,
                    excess_return_column(horizon, "spy"): net_return - 0.0002,
                    excess_return_column(horizon, "qqq"): net_return - 0.0003,
                    excess_return_column(horizon, "sector"): net_return - 0.0001,
                }
                for feature_index, feature in enumerate(INTRADAY_MODEL_FEATURES):
                    row[feature] = technical + 0.01 * feature_index + rng.normal(0, 0.03)
                rows.append(row)
    return pd.DataFrame(rows)
