from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, time, timedelta

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.evaluation import (
    RankingAuditConfig,
    V3PromotionGateConfig,
    build_multi_output_evidence,
    evaluate_ranking_economics,
    evaluate_v3_promotion_evidence,
    fit_disjoint_calibrator,
)


class V3EvaluationTests(unittest.TestCase):
    def setUp(self) -> None:
        self.predictions = _prediction_frame()

    def test_calibration_fit_and_evaluation_sessions_are_disjoint(self) -> None:
        first, first_report, first_evaluation = fit_disjoint_calibrator(
            self.predictions,
            family="D1",
            method="sigmoid",
            minimum_sessions=6,
        )
        second, second_report, second_evaluation = fit_disjoint_calibrator(
            self.predictions,
            family="D1",
            method="sigmoid",
            minimum_sessions=6,
        )
        self.assertLess(first_report["fit_end"], first_report["evaluation_start"])
        self.assertEqual(first.model_run_id, second.model_run_id)
        pd.testing.assert_series_equal(
            first_evaluation["calibrated_probability"].reset_index(drop=True),
            second_evaluation["calibrated_probability"].reset_index(drop=True),
        )
        self.assertEqual(first_report, second_report)

    def test_ranker_scores_cannot_be_mislabeled_as_probabilities(self) -> None:
        with self.assertRaises(DataReadinessError):
            fit_disjoint_calibrator(self.predictions, family="R1")

    def test_session_blocked_ranking_audit_is_reproducible(self) -> None:
        _, _, calibration = fit_disjoint_calibrator(self.predictions, family="D1", minimum_sessions=6)
        evidence = build_multi_output_evidence(
            self.predictions,
            opportunity_family="R1",
            downside_calibration=calibration,
        )
        config = RankingAuditConfig(
            top_k=2,
            maximum_downside_probability=0.8,
            bootstrap_iterations=100,
            bootstrap_seed=17,
            minimum_sessions=2,
        )
        first, selected = evaluate_ranking_economics(evidence, config=config)
        second, _ = evaluate_ranking_economics(evidence, config=config)
        self.assertEqual(first["average_trade_return_interval"], second["average_trade_return_interval"])
        self.assertGreater(first["ranking_groups"], 0)
        self.assertTrue(selected["independent_event_id"].notna().all())
        self.assertTrue(selected["downside_calibrated"].eq(1).all())

    def test_raw_downside_scores_cannot_drive_production_economics(self) -> None:
        evidence = build_multi_output_evidence(self.predictions, opportunity_family="R1")
        with self.assertRaises(DataReadinessError):
            evaluate_ranking_economics(
                evidence,
                config=RankingAuditConfig(bootstrap_iterations=100, minimum_sessions=2),
            )

    def test_overlapping_decision_groups_are_not_double_counted(self) -> None:
        _, _, calibration = fit_disjoint_calibrator(self.predictions, family="D1", minimum_sessions=6)
        evidence = build_multi_output_evidence(
            self.predictions,
            opportunity_family="R1",
            downside_calibration=calibration,
        )
        overlapping = evidence.copy()
        overlapping["decision_time_utc"] = pd.to_datetime(overlapping["decision_time_utc"], utc=True) + pd.Timedelta(minutes=5)
        overlapping["entry_time_utc"] = pd.to_datetime(overlapping["entry_time_utc"], utc=True) + pd.Timedelta(minutes=5)
        overlapping["primary_exit_time_utc"] = pd.to_datetime(
            overlapping["primary_exit_time_utc"], utc=True
        ) + pd.Timedelta(minutes=5)
        overlapping["decision_group_id"] = overlapping["decision_group_id"].astype(str) + ":overlap"
        overlapping["independent_event_id"] = overlapping["independent_event_id"].astype(str) + ":overlap"
        report, _ = evaluate_ranking_economics(
            pd.concat([evidence, overlapping], ignore_index=True),
            config=RankingAuditConfig(
                top_k=2,
                maximum_downside_probability=0.8,
                bootstrap_iterations=100,
                minimum_sessions=2,
            ),
        )
        self.assertEqual(report["ranking_groups"], report["selected_decision_groups"] * 2)

    def test_promotion_evidence_rejects_missing_and_unstable_results(self) -> None:
        missing = evaluate_v3_promotion_evidence(
            ranking_audit=None,
            holdout_metrics=None,
            calibration_audits=None,
        )
        self.assertFalse(missing["passed"])
        weak_ranking = {
            "readiness_failures": [],
            "selected_sessions": 30,
            "selected_trades": 200,
            "mean_ndcg_at_k": 0.7,
            "average_trade_return": 0.01,
            "average_trade_return_interval": {"low": -0.005},
            "profit_factor": 1.5,
            "max_drawdown": 0.1,
        }
        result = evaluate_v3_promotion_evidence(
            ranking_audit=weak_ranking,
            holdout_metrics={"mean_ndcg_at_k": 0.6},
            calibration_audits={"D1": {"after": {"expected_calibration_error": 0.05}}},
            config=V3PromotionGateConfig(),
        )
        self.assertFalse(result["passed"])
        self.assertTrue(any("average return CI" in failure for failure in result["failures"]))


def _prediction_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    start = date(2026, 2, 2)
    for session_offset in range(12):
        session = start + timedelta(days=session_offset)
        decision = datetime.combine(session, time(15), tzinfo=UTC)
        query = decision.isoformat()
        for ticker_index in range(4):
            ticker = f"T{ticker_index}"
            grade = ticker_index
            downside_target = int(ticker_index == 0)
            common = {
                "ticker": ticker,
                "decision_time_utc": decision,
                "session_date_et": session,
                "decision_group_id": query,
                "entry_time_utc": decision + timedelta(minutes=5),
                "primary_exit_time_utc": decision + timedelta(minutes=15),
                "audit_scope": "walk_forward",
                "fold": session_offset // 4,
                "ranking_target": (ticker_index - 1.5) / 100,
                "ranking_grade": grade,
                "stop_before_target": downside_target,
                "path_realized_return_net": (ticker_index - 1.0) / 100,
                "independent_event_id": f"{ticker}:{session.isoformat()}",
            }
            rows.append(
                {
                    **common,
                    "family": "R1",
                    "model_run_id": "r1-run",
                    "target": grade,
                    "score": ticker_index / 3,
                    "opportunity_score": ticker_index / 3,
                }
            )
            rows.append(
                {
                    **common,
                    "family": "D1",
                    "model_run_id": "d1-run",
                    "target": downside_target,
                    "score": 0.75 if downside_target else 0.2 + session_offset / 200,
                    "opportunity_score": 0.25 if downside_target else 0.8 - session_offset / 200,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
