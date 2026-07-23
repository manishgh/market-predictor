"""R2 honest-evaluation and economics: the handoff's required scenarios."""

from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from market_predictor import prediction_service
from market_predictor.execution_policy import executable_fill_price, executable_fill_prices
from market_predictor.intraday.labels import add_overlap_metadata
from market_predictor.prediction_policy import (
    INTRADAY_SELECTION_TIE_BREAKERS,
    calibration_summary,
    intraday_action,
    intraday_decision_score,
    intraday_decision_scores,
    select_top_k_per_group,
    swing_action,
    swing_decision_score,
)
from market_predictor.swing.contracts import SwingPromotionConfig, SwingTrainingConfig, swing_target_column
from market_predictor.swing.evaluation import regime_audit as swing_regime_audit
from market_predictor.swing.model import train_swing_model
from market_predictor.swing.promotion import promote_swing_model, promotion_evidence_from_result
from tests.test_swing_model import _permissive_promotion_config, _training_dataset

_TARGET = swing_target_column(5)


class ServingPolicyIdentityTest(unittest.TestCase):
    def test_oof_replay_scores_ranks_and_actions_match_serving(self) -> None:
        # Score identity: the policy score equals opportunity*(1-downside) and equals
        # the exact value the serving path computes.
        for opportunity, downside in [(0.72, 0.20), (0.50, 0.50), (0.90, 0.10)]:
            expected = opportunity * (1.0 - downside)
            self.assertAlmostEqual(intraday_decision_score(opportunity, downside), expected)
            row = pd.Series({"opp": opportunity, "down": downside})
            self.assertAlmostEqual(prediction_service._risk_adjusted_intraday_score(row, "opp", "down"), expected)
            self.assertEqual(prediction_service._intraday_signal(opportunity, downside), intraday_action(opportunity, downside))
        for probability in [0.70, 0.55, 0.40, 0.30]:
            self.assertEqual(swing_decision_score(probability), probability)
            self.assertEqual(prediction_service._swing_signal(probability), swing_action(probability))

        # Rank identity: selection ranks by the shared risk-adjusted score, not opportunity alone.
        frame = pd.DataFrame(
            {
                "decision_group_id": ["g", "g", "g"],
                "ticker": ["A", "B", "C"],
                "intraday_opportunity_probability": [0.60, 0.90, 0.80],
                "intraday_downside_probability": [0.10, 0.50, 0.20],
            }
        )
        scores = intraday_decision_scores(
            frame,
            opportunity_column="intraday_opportunity_probability",
            downside_column="intraday_downside_probability",
        )
        selected = select_top_k_per_group(
            frame,
            score=scores,
            group_column="decision_group_id",
            top_k=1,
            tie_breakers=INTRADAY_SELECTION_TIE_BREAKERS,
        )
        # A=0.54, B=0.45, C=0.64 -> C wins even though B has the highest opportunity.
        self.assertEqual(list(selected["ticker"]), ["C"])


class AverageUniquenessTest(unittest.TestCase):
    def test_hand_calculated_staggered_intervals(self) -> None:
        base = pd.Timestamp("2026-01-05 14:30:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA", "AAA"],
                "session_date_et": ["2026-01-05", "2026-01-05", "2026-01-05"],
                "label_path_exact": [True, True, True],
                "entry_time_utc": [base, base + pd.Timedelta(minutes=1), base + pd.Timedelta(minutes=4)],
                "label_window_end_utc": [
                    base + pd.Timedelta(minutes=2),
                    base + pd.Timedelta(minutes=3),
                    base + pd.Timedelta(minutes=6),
                ],
            }
        )
        out = add_overlap_metadata(frame)
        # A spans bars {0,1}: concurrency {1,2} -> mean(1, 0.5) = 0.75
        # B spans bars {1,2}: concurrency {2,1} -> mean(0.5, 1) = 0.75
        # C spans bars {4,5}: concurrency {1,1} -> 1.0
        self.assertAlmostEqual(float(out.iloc[0]["overlap_weight"]), 0.75)
        self.assertAlmostEqual(float(out.iloc[1]["overlap_weight"]), 0.75)
        self.assertAlmostEqual(float(out.iloc[2]["overlap_weight"]), 1.0)
        self.assertEqual(list(out["concurrent_label_count"]), [2, 2, 1])
        # Greedy non-overlapping independent events: A and C.
        self.assertEqual(int(out["independent_event_id"].notna().sum()), 2)


class GapThroughFillTest(unittest.TestCase):
    def test_gap_through_stop_fills_at_worse_open(self) -> None:
        # Bar opens through the stop: fill at the worse open, not the barrier.
        self.assertAlmostEqual(
            executable_fill_price(outcome="stop_first", target_price=110.0, stop_price=95.0, trigger_open=90.0, final_price=100.0),
            90.0,
        )
        # Bar opens above the stop, trades down through it: fill at the stop.
        self.assertAlmostEqual(
            executable_fill_price(outcome="stop_first", target_price=110.0, stop_price=95.0, trigger_open=98.0, final_price=100.0),
            95.0,
        )
        # Target: a favorable gap above the target is not credited.
        self.assertAlmostEqual(
            executable_fill_price(outcome="target_first", target_price=110.0, stop_price=95.0, trigger_open=115.0, final_price=112.0),
            110.0,
        )
        # Timeout exits at the final path close.
        self.assertAlmostEqual(
            executable_fill_price(outcome="timeout", target_price=110.0, stop_price=95.0, trigger_open=100.0, final_price=103.0),
            103.0,
        )
        fills = executable_fill_prices(
            outcome=np.array(["stop_first", "target_first", "timeout"]),
            target_price=np.array([110.0, 110.0, 110.0]),
            stop_price=np.array([95.0, 95.0, 95.0]),
            trigger_open=np.array([90.0, 115.0, 100.0]),
            final_price=np.array([100.0, 112.0, 103.0]),
        )
        np.testing.assert_allclose(fills, [90.0, 110.0, 103.0])


class CalibrationGateTest(unittest.TestCase):
    def test_biased_probabilities_preserving_auc_fail_calibration(self) -> None:
        rng = np.random.default_rng(0)
        n = 4000
        p_true = rng.uniform(0.05, 0.95, n)
        y = (rng.uniform(size=n) < p_true).astype(int)
        good = calibration_summary(pd.Series(y), pd.Series(p_true))
        # A monotone transform preserves ranking (and AUC) but distorts probability levels.
        p_biased = p_true * 0.5
        biased = calibration_summary(pd.Series(y), pd.Series(p_biased))
        self.assertAlmostEqual(roc_auc_score(y, p_true), roc_auc_score(y, p_biased), places=6)
        self.assertLess(good["expected_calibration_error"], 0.10)
        self.assertGreater(biased["expected_calibration_error"], 0.10)
        self.assertGreater(abs(biased["calibration_bias"]), abs(good["calibration_bias"]))


class SparseRegimeTest(unittest.TestCase):
    def test_sparse_regime_reports_insufficient_evidence(self) -> None:
        frame = _regime_frame()
        audit = swing_regime_audit(
            frame,
            horizon=5,
            top_k=3,
            target_column=_TARGET,
            min_regime_sessions=1,
            min_regime_trades=1,
        )
        dense = audit[audit["scope"] == "regime:dense"].iloc[0]
        sparse = audit[audit["scope"] == "regime:sparse"].iloc[0]
        self.assertEqual(dense["evidence_status"], "sufficient")
        self.assertEqual(sparse["evidence_status"], "insufficient_evidence")


class PromotionGateTest(unittest.TestCase):
    def _train(self, temp_dir: str) -> tuple[Path, object, object]:
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
        model_path = Path(temp_dir) / "swing_candidate.joblib"
        result = train_swing_model(dataset, model_out=model_path, dataset_sha256="a" * 64, config=config)
        return model_path, result, promotion_evidence_from_result(result)

    def _config(self, **overrides: object) -> SwingPromotionConfig:
        return _permissive_promotion_config().model_copy(update=overrides)

    def test_positive_point_return_but_non_positive_stress_economics_fails(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path, result, evidence = self._train(temp_dir)
            profitability = result.profitability_audit.copy()
            profitability.loc[0, "avg_trade_return"] = 0.05
            profitability.loc[0, "avg_excess_return_vs_spy"] = 0.03
            profitability.loc[0, "avg_excess_return_vs_qqq"] = 0.03
            profitability.loc[0, "avg_excess_return_vs_sector"] = 0.03
            profitability.loc[0, "return_drawdown_ratio"] = 1.0
            profitability.loc[0, "stress_avg_trade_return"] = -0.02
            profitability.loc[0, "stress_avg_excess_return_vs_spy"] = -0.02
            report = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, profitability_audit=profitability),
                config=self._config(min_avg_trade_return=0.0, min_stress_avg_trade_return=0.0, min_stress_avg_excess_return_vs_spy=0.0),
            )
            self.assertFalse(report["passed"])
            self.assertTrue(any("stress_avg_trade_return" in failure for failure in report["failures"]))

    def test_overall_profit_with_populated_losing_regime_fails(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path, result, evidence = self._train(temp_dir)
            regime = result.regime_audit.copy()
            summary = regime.iloc[0]
            losing = {
                "scope": "regime:synthetic_bear",
                "regimes_present": summary["regimes_present"],
                "max_single_regime_share": summary["max_single_regime_share"],
                "rows": 400,
                "evidence_status": "sufficient",
                "selected_trades": 80,
                "sessions": 30,
                "avg_trade_return": -0.05,
                "avg_excess_return_vs_spy": -0.05,
                "max_drawdown": 0.20,
                "calibration_error": 0.05,
            }
            regime = pd.concat([regime, pd.DataFrame([losing])], ignore_index=True)
            regime["model_run_id"] = result.metrics["model_run_id"]
            report = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, regime_audit=regime),
                config=self._config(min_worst_regime_avg_excess_return_vs_spy=-0.01),
            )
            self.assertFalse(report["passed"])
            self.assertTrue(
                any("synthetic_bear" in failure and "avg_excess_return_vs_spy" in failure for failure in report["failures"])
            )

    def test_missing_reconciliation_hash_fails_promotion(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path, result, evidence = self._train(temp_dir)
            metrics = dict(result.metrics)
            metrics["reconciliation_sha256"] = ""
            report = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, metrics=metrics),
                config=self._config(),
            )
            self.assertFalse(report["passed"])
            self.assertTrue(any("reconciliation_sha256" in failure for failure in report["failures"]))

    def test_too_few_sessions_fails_even_with_many_rows(self) -> None:
        with TemporaryDirectory() as temp_dir:
            model_path, result, evidence = self._train(temp_dir)
            metrics = dict(result.metrics)
            metrics["validated_rows"] = 100_000
            metrics["independent_sessions"] = 3
            report = promote_swing_model(
                model_path=model_path,
                evidence=replace(evidence, metrics=metrics),
                config=self._config(min_validated_rows=100, min_independent_sessions=60),
            )
            self.assertFalse(report["passed"])
            self.assertTrue(any("independent_sessions" in failure for failure in report["failures"]))


def _regime_frame() -> pd.DataFrame:
    records: list[dict[str, object]] = []

    def add(regime: str, base_date: pd.Timestamp, n_sessions: int, tickers: list[str]) -> None:
        for session_offset in range(n_sessions):
            date = base_date + pd.Timedelta(days=session_offset)
            group = f"{regime}-{session_offset}"
            for index, ticker in enumerate(tickers):
                records.append(
                    {
                        "ticker": ticker,
                        "session_date_et": date.strftime("%Y-%m-%d"),
                        "decision_group_id": group,
                        "market_regime": regime,
                        "swing_probability": 0.50 + 0.03 * index,
                        _TARGET: index % 2,
                        "future_net_return_5d": 0.010,
                        "future_gross_return_5d": 0.012,
                        "future_excess_return_5d_vs_spy": 0.005,
                        "future_excess_return_5d_vs_qqq": 0.004,
                        "future_excess_return_5d_vs_sector": 0.003,
                        "close": 50.0,
                        "atr_pct_14": 0.02,
                    }
                )

    add("dense", pd.Timestamp("2026-01-05"), 8, ["A", "B", "C", "D", "E"])
    add("sparse", pd.Timestamp("2026-03-05"), 1, ["F", "G", "H"])
    return pd.DataFrame(records)


if __name__ == "__main__":
    unittest.main()
