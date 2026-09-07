from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.drift.features import FEATURE_DRIFT_REPORT_VERSION
from market_predictor.governance.drift.policy import (
    DriftAssessmentV2,
    DriftPolicyV2,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.modeling.feature_reference import feature_reference_names_sha256


class DriftPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        self.release_id = "a" * 64
        self.model_sha = "b" * 64
        self.prediction_policy_sha = "c" * 64
        self.label_policy_sha = "d" * 64
        self.execution_policy_sha = "e" * 64
        self.feature_names_sha = feature_reference_names_sha256(["x"])
        self.policy = DriftPolicyV2(
            minimum_matured_samples=10,
            minimum_independent_decision_groups=5,
        )

    def test_stable_and_warning_performance_remain_actionable(self) -> None:
        stable = self._evaluate(self._report(samples=20))
        warning = self._evaluate(
            self._report(
                samples=20,
                opportunity_brier=0.30,
                view="intraday",
                horizon="60m",
            ),
            mode="intraday",
            horizon="60m",
        )

        self.assertEqual(
            (stable.state, stable.actionability),
            ("stable", "actionable"),
        )
        self.assertEqual(
            (warning.state, warning.actionability),
            ("warning", "actionable"),
        )

    def test_insufficient_severe_stale_and_unavailable_fail_closed(self) -> None:
        insufficient = self._evaluate(self._report(samples=5))
        severe = self._evaluate(self._report(samples=20, drawdown=0.30))
        stale = self._evaluate(
            self._report(
                samples=20,
                generated_at=self.now - timedelta(days=8),
            )
        )
        unavailable = self._evaluate(
            self._report(samples=20),
            feature_status="unavailable",
        )

        self.assertEqual(
            (insufficient.state, insufficient.actionability),
            ("warming", "rank_only"),
        )
        self.assertEqual(
            (severe.state, severe.actionability),
            ("severe", "not_ready"),
        )
        self.assertEqual(
            (stale.state, stale.actionability),
            ("stale", "not_ready"),
        )
        self.assertEqual(
            (unavailable.state, unavailable.actionability),
            ("unavailable", "not_ready"),
        )

    def test_intraday_downside_degradation_is_severe(self) -> None:
        assessment = self._evaluate(
            self._report(
                samples=20,
                view="intraday",
                horizon="60m",
                downside_brier=0.40,
            ),
            mode="intraday",
            horizon="60m",
        )

        self.assertEqual(
            (assessment.state, assessment.actionability),
            ("severe", "not_ready"),
        )

    def test_overdue_unmatured_predictions_fail_closed(self) -> None:
        assessment = self._evaluate(
            self._report(samples=20, pending_age_minutes=40_000)
        )

        self.assertEqual(
            (assessment.state, assessment.actionability),
            ("unavailable", "not_ready"),
        )
        self.assertIn("selected_policy_outcomes_overdue", assessment.reasons)

    def test_future_performance_evidence_is_rejected_without_clock_tolerance(self) -> None:
        assessment = self._evaluate(
            self._report(
                samples=20,
                generated_at=self.now + timedelta(microseconds=1),
            )
        )

        self.assertEqual(
            (assessment.state, assessment.actionability),
            ("stale", "not_ready"),
        )
        self.assertIn("performance_report_from_future", assessment.reasons)

    def test_identity_mismatch_is_not_ready(self) -> None:
        report = self._report(samples=20)

        assessment = evaluate_drift(
            mode="swing",
            horizon="10b",
            model_release_id=self.release_id,
            model_artifact_sha256="f" * 64,
            prediction_policy_sha256=self.prediction_policy_sha,
            label_policy_sha256=self.label_policy_sha,
            execution_policy_sha256=self.execution_policy_sha,
            feature_reference_profile_sha256="9" * 64,
            feature_reference_names_sha256=self.feature_names_sha,
            feature_drift=self._feature_report("stable"),
            performance_report=report,
            policy=self.policy,
            evaluated_at=self.now,
        )

        self.assertEqual(assessment.state, "unavailable")
        self.assertEqual(assessment.actionability, "not_ready")
        self.assertIn(
            "selected_policy_identity_mismatch",
            assessment.reasons,
        )

    def test_policy_rejects_inverted_thresholds(self) -> None:
        with self.assertRaises(ValidationError):
            DriftPolicyV2(
                warning_opportunity_brier_score=0.4,
                severe_opportunity_brier_score=0.3,
            )

    def test_state_store_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DriftStateStore(Path(temp_dir))
            assessment = self._evaluate(self._report(samples=20))
            store.publish(assessment)

            self.assertEqual(
                store.load("swing", "10b", self.release_id),
                assessment,
            )

            path = Path(temp_dir) / "swing" / "10b" / f"{self.release_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["state"] = "warning"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                store.load("swing", "10b", self.release_id)

    def test_assessment_rejects_impossible_state_actionability_pair(self) -> None:
        valid = self._evaluate(self._report(samples=20)).model_dump(
            mode="python", exclude={"assessment_id"}
        )
        valid["state"] = "severe"
        valid["actionability"] = "actionable"

        with self.assertRaisesRegex(ValidationError, "state and actionability"):
            DriftAssessmentV2.model_validate(
                {**valid, "assessment_id": content_sha256(valid)}
            )

    def test_state_store_rejects_time_regression_and_equal_time_conflict(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DriftStateStore(Path(temp_dir))
            latest = self._evaluate(
                self._report(samples=20),
                evaluated_at=self.now,
            )
            older = self._evaluate(
                self._report(samples=20),
                evaluated_at=self.now - timedelta(minutes=1),
            )
            conflicting = self._evaluate(
                self._report(samples=20),
                feature_status="warning",
                evaluated_at=self.now,
            )
            store.publish(latest)

            with self.assertRaises(PredictionConflictError):
                store.publish(older)
            with self.assertRaises(PredictionConflictError):
                store.publish(conflicting)

    def test_each_benchmark_can_trigger_excess_return_degradation(self) -> None:
        for field in ("excess", "qqq_excess", "sector_excess"):
            assessment = self._evaluate(
                self._report(samples=20, **{field: -0.006})
            )

            self.assertEqual(
                (assessment.state, assessment.actionability),
                ("severe", "not_ready"),
            )
            self.assertTrue(
                any(
                    reason.endswith("_excess_return_severe")
                    for reason in assessment.reasons
                )
            )

    def test_feature_drift_route_time_and_feature_set_are_enforced(self) -> None:
        route_mismatch = self._feature_report("stable")
        route_mismatch["horizon"] = "5b"
        route_mismatch["report_id"] = content_sha256(
            {key: value for key, value in route_mismatch.items() if key != "report_id"}
        )
        future = self._feature_report("stable")
        future_time = self.now + timedelta(microseconds=1)
        future["generated_at_utc"] = future_time.isoformat().replace("+00:00", "Z")
        future["window_end_utc"] = future["generated_at_utc"]
        future["report_id"] = content_sha256(
            {key: value for key, value in future.items() if key != "report_id"}
        )
        feature_set_mismatch = self._feature_report("stable")
        feature_set_mismatch["feature_artifact_set_sha256"] = "8" * 64
        feature_set_mismatch["report_id"] = content_sha256(
            {
                key: value
                for key, value in feature_set_mismatch.items()
                if key != "report_id"
            }
        )
        reference_mismatch = self._feature_report("stable")
        reference_mismatch["reference_profile_sha256"] = "7" * 64
        reference_mismatch["report_id"] = content_sha256(
            {
                key: value
                for key, value in reference_mismatch.items()
                if key != "report_id"
            }
        )
        feature_names_mismatch = self._feature_report("stable")
        feature_names_mismatch["feature_rows"] = (
            {
                "feature": "different",
                "standardized_mean_shift": 0.0,
                "missing_rate_delta": 0.0,
            },
        )
        feature_names_mismatch["feature_names_sha256"] = feature_reference_names_sha256(
            ["different"]
        )
        feature_names_mismatch["report_id"] = content_sha256(
            {
                key: value
                for key, value in feature_names_mismatch.items()
                if key != "report_id"
            }
        )
        stale_observations = self._feature_report("stable")
        stale_observations["window_start_utc"] = (
            self.now - timedelta(days=3)
        ).isoformat().replace("+00:00", "Z")
        stale_observations["window_end_utc"] = (
            self.now - timedelta(days=2)
        ).isoformat().replace("+00:00", "Z")
        stale_observations["report_id"] = content_sha256(
            {
                key: value
                for key, value in stale_observations.items()
                if key != "report_id"
            }
        )
        threshold_mismatch = self._feature_report("stable")
        threshold_mismatch["standardized_shift_warning"] = 2_000.0
        threshold_mismatch["standardized_shift_severe"] = 3_000.0
        threshold_mismatch["report_id"] = content_sha256(
            {
                key: value
                for key, value in threshold_mismatch.items()
                if key != "report_id"
            }
        )
        insufficient_sample = self._feature_report("stable")
        insufficient_sample["live_rows"] = 1
        insufficient_sample["report_id"] = content_sha256(
            {
                key: value
                for key, value in insufficient_sample.items()
                if key != "report_id"
            }
        )

        cases = (
            (route_mismatch, "feature_drift_identity_mismatch"),
            (future, "feature_drift_from_future"),
            (feature_set_mismatch, "feature_drift_feature_set_mismatch"),
            (reference_mismatch, "feature_drift_reference_identity_mismatch"),
            (
                feature_names_mismatch,
                "feature_drift_feature_names_identity_mismatch",
            ),
            (stale_observations, "feature_drift_observations_stale"),
            (threshold_mismatch, "feature_drift_threshold_identity_mismatch"),
            (insufficient_sample, "feature_drift_sample_insufficient"),
        )
        for feature_report, expected_reason in cases:
            assessment = evaluate_drift(
                mode="swing",
                horizon="10b",
                model_release_id=self.release_id,
                model_artifact_sha256=self.model_sha,
                prediction_policy_sha256=self.prediction_policy_sha,
                label_policy_sha256=self.label_policy_sha,
                execution_policy_sha256=self.execution_policy_sha,
                feature_reference_profile_sha256="9" * 64,
                feature_reference_names_sha256=self.feature_names_sha,
                feature_drift=feature_report,
                performance_report=self._report(samples=20),
                policy=self.policy,
                evaluated_at=self.now,
            )
            self.assertEqual(assessment.actionability, "not_ready")
            self.assertIn(expected_reason, assessment.reasons)

    def _evaluate(
        self,
        report: dict[str, object],
        *,
        feature_status: str = "stable",
        mode: str = "swing",
        horizon: str = "10b",
        evaluated_at: datetime | None = None,
    ):
        return evaluate_drift(
            mode=mode,
            horizon=horizon,
            model_release_id=self.release_id,
            model_artifact_sha256=self.model_sha,
            prediction_policy_sha256=self.prediction_policy_sha,
            label_policy_sha256=self.label_policy_sha,
            execution_policy_sha256=self.execution_policy_sha,
            feature_reference_profile_sha256="9" * 64,
            feature_reference_names_sha256=self.feature_names_sha,
            feature_drift=self._feature_report(
                feature_status,
                mode=mode,
                horizon=horizon,
            ),
            performance_report=report,
            policy=self.policy,
            evaluated_at=evaluated_at or self.now,
        )

    def _feature_report(
        self,
        status: str,
        *,
        mode: str = "swing",
        horizon: str = "10b",
    ) -> dict[str, object]:
        available = status != "unavailable"
        content: dict[str, object] = {
            "contract_version": FEATURE_DRIFT_REPORT_VERSION,
            "mode": mode,
            "horizon": horizon,
            "model_release_id": self.release_id,
            "model_artifact_sha256": self.model_sha,
            "feature_artifact_set_sha256": "f" * 64,
            "reference_profile_sha256": "9" * 64,
            "feature_names_sha256": self.feature_names_sha,
            "window_start_utc": (self.now - timedelta(hours=1))
            .isoformat()
            .replace("+00:00", "Z"),
            "window_end_utc": self.now.isoformat().replace("+00:00", "Z"),
            "generated_at_utc": self.now.isoformat().replace("+00:00", "Z"),
            "source_artifact_sha256": "7" * 64,
            "live_rows": 30,
            "status": status,
            "features_compared": 1 if available else 0,
            "warning_feature_count": 1 if status in {"warning", "severe"} else 0,
            "severe_feature_count": 1 if status == "severe" else 0,
            "max_standardized_mean_shift": (
                4.0 if status == "severe" else 2.0 if status == "warning" else 0.0
            ),
            "max_missing_rate_delta": 0.0,
            "standardized_shift_warning": 2.0,
            "standardized_shift_severe": 4.0,
            "missing_rate_delta_warning": 0.2,
            "missing_rate_delta_severe": 0.5,
            "feature_rows": (
                (
                    {
                        "feature": "x",
                        "standardized_mean_shift": (
                            4.0
                            if status == "severe"
                            else 2.0
                            if status == "warning"
                            else 0.0
                        ),
                        "missing_rate_delta": 0.0,
                    },
                )
                if available
                else ()
            ),
            "reason": None if available else "feature reference unavailable",
        }
        return {**content, "report_id": content_sha256(content)}

    def _report(
        self,
        *,
        samples: int,
        opportunity_brier: float = 0.20,
        downside_brier: float = 0.20,
        excess: float = 0.01,
        qqq_excess: float = 0.01,
        sector_excess: float = 0.01,
        drawdown: float = 0.05,
        generated_at: datetime | None = None,
        view: str = "swing",
        horizon: str = "10b",
        pending_age_minutes: int | None = None,
    ) -> dict[str, object]:
        generated = generated_at or self.now
        window_start = generated - timedelta(days=60)
        pending_count = int(pending_age_minutes is not None)
        oldest_pending = (
            generated - timedelta(minutes=pending_age_minutes)
            if pending_age_minutes is not None
            else None
        )
        total_predictions = samples + pending_count
        row_identity: dict[str, object] = {
            "model_release_id": self.release_id,
            "model_artifact_sha256": self.model_sha,
            "prediction_policy_sha256": self.prediction_policy_sha,
            "label_policy_sha256": self.label_policy_sha,
            "execution_policy_sha256": self.execution_policy_sha,
            "feature_artifact_set_sha256": "f" * 64,
            "source_intent_ids_sha256": "1" * 64,
            "source_observation_ids_sha256": "6" * 64,
            "source_outcome_ids_sha256": "2" * 64,
            "view": view,
            "horizon": horizon,
            "cohort_type": "all",
            "cohort_value": "all",
            "window_start_utc": window_start.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "window_end_utc": generated.isoformat().replace("+00:00", "Z"),
            "total_predictions": total_predictions,
            "eligible_predictions": total_predictions,
            "selected_predictions": total_predictions,
            "actionable_predictions": total_predictions,
            "matured_selected_samples": samples,
            "pending_selected_samples": pending_count,
            "oldest_pending_decision_time_utc": (
                oldest_pending.isoformat().replace("+00:00", "Z")
                if oldest_pending is not None
                else None
            ),
            "independent_decision_groups": samples,
            "evidence_status": (
                "sufficient" if samples >= 10 else "insufficient_evidence"
            ),
            "selection_rate": 1.0,
            "actionable_rate": 1.0,
            "mean_probability": 0.60,
            "probability_p10": 0.50,
            "probability_p50": 0.60,
            "probability_p90": 0.70,
            "mean_decision_score": 0.60,
            "decision_score_p10": 0.50,
            "decision_score_p50": 0.60,
            "decision_score_p90": 0.70,
            "mean_selected_rank": 1.0,
            "selected_rank_p90": 1.0,
            "opportunity_observed_rate": 0.55 if view == "intraday" else None,
            "opportunity_brier_score": opportunity_brier if view == "intraday" else None,
            "opportunity_calibration_error": 0.05 if view == "intraday" else None,
            "mean_downside_probability": 0.30 if view == "intraday" else None,
            "downside_observed_rate": 0.25 if view == "intraday" else None,
            "downside_brier_score": (
                downside_brier if view == "intraday" else None
            ),
            "downside_calibration_error": (
                0.05 if view == "intraday" else None
            ),
            "average_net_return": 0.01,
            "average_excess_return_vs_spy": excess,
            "average_excess_return_vs_qqq": qqq_excess,
            "average_excess_return_vs_sector": sector_excess,
            "cumulative_net_return": 0.10,
            "win_rate": 0.55,
            "max_drawdown": drawdown,
            "first_decision_time_utc": (oldest_pending or generated).isoformat().replace(
                "+00:00",
                "Z",
            ),
            "last_decision_time_utc": generated.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "last_matured_outcome_utc": generated.isoformat().replace(
                "+00:00",
                "Z",
            ),
        }
        row = {
            **row_identity,
            "cohort_id": content_sha256(row_identity),
        }
        report_identity: dict[str, object] = {
            "contract_version": (
                "market_predictor.selected_policy_performance.v2"
            ),
            "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
            "lookback_days": 60,
            "minimum_matured_samples": 10,
            "window_start_utc": window_start.isoformat().replace(
                "+00:00",
                "Z",
            ),
            "window_end_utc": generated.isoformat().replace("+00:00", "Z"),
            "source_intent_ids": ["3" * 64],
            "source_observation_ids": ["6" * 64],
            "source_outcome_ids": ["4" * 64],
            "rows": [row],
        }
        return {
            **report_identity,
            "report_id": content_sha256(report_identity),
        }


if __name__ == "__main__":
    unittest.main()
