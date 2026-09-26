from __future__ import annotations

import json
import tempfile
import tomllib
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from zoneinfo import ZoneInfo

from pydantic import ValidationError
from typer.testing import CliRunner

from market_predictor.commands.configuration import load_typed_config
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.drift.features import (
    FEATURE_DRIFT_REPORT_VERSION,
    validate_feature_drift_report,
)
from market_predictor.governance.drift.policy import (
    DriftAssessmentV3,
    DriftPolicyV3,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.governance.outcomes.contracts import content_sha256
from market_predictor.modeling.feature_reference import feature_reference_names_sha256
from market_predictor.production_cli import app

ROOT = Path(__file__).resolve().parents[1]
NEW_YORK = ZoneInfo("America/New_York")


class DriftPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        self.release_id = "a" * 64
        self.model_sha = "b" * 64
        self.prediction_policy_sha = "c" * 64
        self.label_policy_sha = "d" * 64
        self.execution_policy_sha = "e" * 64
        self.feature_names_sha = feature_reference_names_sha256(["x"])
        self.policy = DriftPolicyV3(
            minimum_matured_samples=10,
            minimum_independent_decision_groups=5,
        )

    def test_stable_and_warning_performance_remain_actionable(self) -> None:
        stable = self._evaluate(self._report(samples=20))
        warning = self._evaluate(self._report(samples=20, drawdown=0.20))

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

    def test_overdue_unmatured_predictions_fail_closed(self) -> None:
        assessment = self._evaluate(
            self._report(
                samples=20,
                oldest_pending=self.now - timedelta(minutes=40_000),
            )
        )

        self.assertEqual(
            (assessment.state, assessment.actionability),
            ("unavailable", "not_ready"),
        )
        self.assertIn("selected_policy_outcomes_overdue", assessment.reasons)

    def test_pending_deadline_counts_exchange_sessions_across_holidays(self) -> None:
        # 3 July 2026 is an XNYS holiday. The tenth session after 1 July closes on
        # 16 July and after 2 July on 17 July; each deadline adds the 7-day grace.
        before_holiday = datetime(2026, 7, 1, 22, 0, tzinfo=UTC)
        on_holiday_eve = datetime(2026, 7, 2, 22, 0, tzinfo=UTC)
        tenth_close = datetime(2026, 7, 17, 20, 0, tzinfo=UTC)

        overdue = self._evaluate(self._report(samples=20, oldest_pending=before_holiday))
        pending = self._evaluate(self._report(samples=20, oldest_pending=on_holiday_eve))

        self.assertIn("selected_policy_outcomes_overdue", overdue.reasons)
        self.assertEqual((pending.state, pending.actionability), ("stable", "actionable"))
        deadline = tenth_close + timedelta(days=7)
        self.assertFalse(self.policy.outcome_overdue("10b", date(2026, 7, 2), deadline))
        self.assertTrue(
            self.policy.outcome_overdue("10b", date(2026, 7, 2), deadline + timedelta(microseconds=1))
        )

    def test_pending_deadline_scales_to_investment_horizons(self) -> None:
        decision = datetime(2025, 7, 24, 22, 0, tzinfo=UTC)
        # An annual horizon needs a performance window longer than a year.
        ten_session = self._evaluate(
            self._report(samples=20, oldest_pending=decision, lookback_days=400)
        )
        annual = self._evaluate(
            self._report(samples=20, horizon="252b", oldest_pending=decision, lookback_days=400),
            horizon="252b",
        )

        self.assertIn("selected_policy_outcomes_overdue", ten_session.reasons)
        self.assertEqual((annual.state, annual.actionability), ("stable", "actionable"))
        # The 252nd session after 24 July 2025 closes on 27 July 2026. The weekday
        # estimate's limit, 353 days plus the grace, ends 19 July: eight days too early.
        deadline = datetime(2026, 7, 27, 20, 0, tzinfo=UTC) + timedelta(days=7)
        self.assertFalse(self.policy.outcome_overdue("252b", date(2025, 7, 24), deadline))
        self.assertTrue(
            self.policy.outcome_overdue("252b", date(2025, 7, 24), deadline + timedelta(microseconds=1))
        )

    def test_pending_deadline_requires_a_session_horizon_and_calendar(self) -> None:
        with self.assertRaisesRegex(ValueError, "not a session count"):
            self.policy.outcome_overdue("10d", date(2026, 6, 24), self.now)
        with self.assertRaisesRegex(ValueError, "timezone-aware"):
            self.policy.outcome_overdue("10b", date(2026, 6, 24), self.now.replace(tzinfo=None))
        with self.assertRaisesRegex(DataReadinessError, "not an XNYS session"):
            self.policy.outcome_overdue("10b", date(2026, 7, 3), self.now)
        with self.assertRaisesRegex(DataReadinessError, "not an XNYS session"):
            self.policy.outcome_overdue("10b", date(1990, 1, 2), self.now)
        with self.assertRaisesRegex(DataReadinessError, "XNYS calendar ends"):
            self.policy.outcome_overdue("252b", date(2026, 6, 24), datetime(2099, 1, 2, tzinfo=UTC))

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
            DriftPolicyV3(warning_max_drawdown=0.3, severe_max_drawdown=0.2)
        with self.assertRaises(ValidationError):
            DriftPolicyV3.model_validate({"maximum_pending_age_minutes_swing": 30_240})

    def test_configured_policy_matches_serving_pin(self) -> None:
        policy = load_typed_config(ROOT / "configs" / "drift_policy.toml", DriftPolicyV3)
        default = tomllib.loads((ROOT / "configs" / "default.toml").read_text(encoding="utf-8"))

        self.assertEqual(policy, DriftPolicyV3())
        self.assertEqual(policy.sha256(), default["prediction_serving"]["drift_policy_sha256"])

    def test_retired_intraday_and_day_horizons_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DriftStateStore(Path(temp_dir))
            with self.assertRaisesRegex(ValueError, "intraday drift state is retired"):
                store.load("intraday", "60m", self.release_id)
            with self.assertRaisesRegex(ValueError, "horizon is invalid"):
                store.load("swing", "10d", self.release_id)
        with self.assertRaisesRegex(ValidationError, "intraday"):
            validate_feature_drift_report(
                self._feature_report("stable", mode="intraday", horizon="60m")
            )
        superseded = self._feature_report("stable")
        superseded["contract_version"] = "market_predictor.feature_drift_report.v1"
        superseded["report_id"] = content_sha256(
            {key: value for key, value in superseded.items() if key != "report_id"}
        )
        with self.assertRaises(ValidationError):
            validate_feature_drift_report(superseded)
        for horizon in ("60m", "10d"):
            with self.subTest(horizon=horizon), self.assertRaises(ValidationError):
                self._evaluate(self._report(samples=20, horizon=horizon), horizon=horizon)

    def test_cli_assesses_swing_routes_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            result = CliRunner().invoke(
                app,
                [
                    "publish-drift-assessment", "--mode", "intraday", "--horizon", "60m",
                    "--model-release-id", self.release_id, "--model-artifact-sha256", self.model_sha,
                    "--prediction-policy-sha256", self.prediction_policy_sha,
                    "--label-policy-sha256", self.label_policy_sha,
                    "--execution-policy-sha256", self.execution_policy_sha,
                    "--feature-reference-profile-sha256", "9" * 64,
                    "--feature-reference-names-sha256", self.feature_names_sha,
                    "--feature-drift-report", str(root / "absent.json"),
                    "--drift-dir", str(root / "drift"),
                ],
            )

            self.assertEqual(result.exit_code, 2)
            self.assertIn("only swing routes are assessed", result.output)
            self.assertFalse((root / "drift").exists())

    def test_superseded_assessment_versions_are_refused(self) -> None:
        current = self._evaluate(self._report(samples=20)).model_dump(mode="json")
        for version in (
            "market_predictor.drift_assessment.v1",
            "market_predictor.drift_assessment.v2",
        ):
            content = {
                **{key: value for key, value in current.items() if key != "assessment_id"},
                "contract_version": version,
            }
            with self.subTest(version=version), self.assertRaises(ValidationError):
                DriftAssessmentV3.model_validate(
                    {**content, "assessment_id": content_sha256(content)}
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
            DriftAssessmentV3.model_validate(
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
        horizon: str = "10b",
        evaluated_at: datetime | None = None,
    ):
        return evaluate_drift(
            mode="swing",
            horizon=horizon,
            model_release_id=self.release_id,
            model_artifact_sha256=self.model_sha,
            prediction_policy_sha256=self.prediction_policy_sha,
            label_policy_sha256=self.label_policy_sha,
            execution_policy_sha256=self.execution_policy_sha,
            feature_reference_profile_sha256="9" * 64,
            feature_reference_names_sha256=self.feature_names_sha,
            feature_drift=self._feature_report(feature_status, horizon=horizon),
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
        excess: float = 0.01,
        qqq_excess: float = 0.01,
        sector_excess: float = 0.01,
        drawdown: float = 0.05,
        generated_at: datetime | None = None,
        horizon: str = "10b",
        oldest_pending: datetime | None = None,
        lookback_days: int = 60,
    ) -> dict[str, object]:
        generated = generated_at or self.now
        window_start = generated - timedelta(days=lookback_days)
        pending_count = int(oldest_pending is not None)
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
            "view": "swing",
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
            "oldest_pending_decision_session_et": (
                oldest_pending.astimezone(NEW_YORK).date().isoformat()
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
                "market_predictor.selected_policy_performance.v3"
            ),
            "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
            "lookback_days": lookback_days,
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
