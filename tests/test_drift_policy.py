from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from market_predictor.drift_policy import (
    DriftPolicyV2,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.outcome_contracts import content_sha256
from market_predictor.prediction_contracts import PredictionConflictError


class DriftPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        self.release_id = "a" * 64
        self.model_sha = "b" * 64
        self.prediction_policy_sha = "c" * 64
        self.label_policy_sha = "d" * 64
        self.execution_policy_sha = "e" * 64
        self.policy = DriftPolicyV2(
            minimum_matured_samples=10,
            minimum_independent_decision_groups=5,
        )

    def test_stable_and_warning_performance_remain_actionable(self) -> None:
        stable = self._evaluate(self._report(samples=20))
        warning = self._evaluate(
            self._report(samples=20, opportunity_brier=0.30)
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
            feature_drift={"status": "stable"},
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

    def _evaluate(
        self,
        report: dict[str, object],
        *,
        feature_status: str = "stable",
        mode: str = "swing",
        horizon: str = "10b",
    ):
        return evaluate_drift(
            mode=mode,
            horizon=horizon,
            model_release_id=self.release_id,
            model_artifact_sha256=self.model_sha,
            prediction_policy_sha256=self.prediction_policy_sha,
            label_policy_sha256=self.label_policy_sha,
            execution_policy_sha256=self.execution_policy_sha,
            feature_drift={"status": feature_status},
            performance_report=report,
            policy=self.policy,
            evaluated_at=self.now,
        )

    def _report(
        self,
        *,
        samples: int,
        opportunity_brier: float = 0.20,
        downside_brier: float = 0.20,
        excess: float = 0.01,
        drawdown: float = 0.05,
        generated_at: datetime | None = None,
        view: str = "swing",
        horizon: str = "10b",
    ) -> dict[str, object]:
        generated = generated_at or self.now
        window_start = generated - timedelta(days=60)
        row_identity: dict[str, object] = {
            "model_release_id": self.release_id,
            "model_artifact_sha256": self.model_sha,
            "prediction_policy_sha256": self.prediction_policy_sha,
            "label_policy_sha256": self.label_policy_sha,
            "execution_policy_sha256": self.execution_policy_sha,
            "feature_artifact_set_sha256": "f" * 64,
            "source_intent_ids_sha256": "1" * 64,
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
            "total_predictions": samples,
            "eligible_predictions": samples,
            "selected_predictions": samples,
            "actionable_predictions": samples,
            "matured_selected_samples": samples,
            "pending_selected_samples": 0,
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
            "opportunity_observed_rate": 0.55,
            "opportunity_brier_score": opportunity_brier,
            "opportunity_calibration_error": 0.05,
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
            "cumulative_net_return": 0.10,
            "win_rate": 0.55,
            "max_drawdown": drawdown,
            "first_decision_time_utc": generated.isoformat().replace(
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
            "source_outcome_ids": ["4" * 64],
            "rows": [row],
        }
        return {
            **report_identity,
            "report_id": content_sha256(report_identity),
        }


if __name__ == "__main__":
    unittest.main()
