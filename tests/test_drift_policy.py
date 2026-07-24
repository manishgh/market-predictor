from __future__ import annotations

import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from market_predictor.drift_policy import (
    DriftPolicyV1,
    DriftStateStore,
    evaluate_drift,
)
from market_predictor.outcome_contracts import content_sha256
from market_predictor.prediction_contracts import PredictionConflictError


class DriftPolicyTests(unittest.TestCase):
    def setUp(self) -> None:
        self.now = datetime(2026, 7, 24, 12, 0, tzinfo=UTC)
        self.release_id = "a" * 64
        self.policy = DriftPolicyV1(minimum_matured_samples=10)

    def test_stable_and_warning_performance_remain_actionable(self) -> None:
        stable = self._evaluate(self._report(samples=20))
        warning = self._evaluate(self._report(samples=20, brier=0.30))

        self.assertEqual((stable.state, stable.actionability), ("stable", "actionable"))
        self.assertEqual(
            (warning.state, warning.actionability),
            ("warning", "actionable"),
        )

    def test_warming_severe_stale_and_unavailable_fail_closed(self) -> None:
        warming = self._evaluate(self._report(samples=5))
        severe = self._evaluate(self._report(samples=20, drawdown=0.30))
        stale = self._evaluate(
            self._report(samples=20, generated_at=self.now - timedelta(days=2))
        )
        unavailable = self._evaluate(
            self._report(samples=20),
            feature_status="unavailable",
        )

        self.assertEqual((warming.state, warming.actionability), ("warming", "rank_only"))
        self.assertEqual((severe.state, severe.actionability), ("severe", "not_ready"))
        self.assertEqual((stale.state, stale.actionability), ("stale", "not_ready"))
        self.assertEqual(
            (unavailable.state, unavailable.actionability),
            ("unavailable", "not_ready"),
        )

    def test_policy_rejects_inverted_thresholds(self) -> None:
        with self.assertRaises(ValidationError):
            DriftPolicyV1(
                warning_brier_score=0.4,
                severe_brier_score=0.3,
            )

    def test_state_store_round_trip_and_tamper_detection(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            store = DriftStateStore(Path(temp_dir))
            assessment = self._evaluate(self._report(samples=20))
            store.publish(assessment)

            self.assertEqual(
                store.load("swing", "5d", self.release_id),
                assessment,
            )

            path = Path(temp_dir) / "swing" / "5d" / f"{self.release_id}.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["state"] = "warning"
            path.write_text(json.dumps(payload), encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                store.load("swing", "5d", self.release_id)

    def _evaluate(
        self,
        report: dict[str, object],
        *,
        feature_status: str = "stable",
    ):
        return evaluate_drift(
            mode="swing",
            horizon="5d",
            model_release_id=self.release_id,
            feature_drift={"status": feature_status},
            performance_report=report,
            policy=self.policy,
            evaluated_at=self.now,
        )

    def _report(
        self,
        *,
        samples: int,
        brier: float = 0.20,
        excess: float = 0.01,
        drawdown: float = 0.05,
        generated_at: datetime | None = None,
    ) -> dict[str, object]:
        generated = generated_at or self.now
        identity = {
            "contract_version": "market_predictor.performance_cohorts.v1",
            "generated_at_utc": generated.isoformat().replace("+00:00", "Z"),
            "minimum_samples": 10,
            "source_outcome_ids": ["c" * 64],
            "rows": [
                {
                    "model_release_id": self.release_id,
                    "view": "swing",
                    "horizon": "5d",
                    "cohort_type": "all",
                    "cohort_value": "all",
                    "samples": samples,
                    "evidence_status": (
                        "sufficient" if samples >= 10 else "insufficient_evidence"
                    ),
                    "mean_probability": 0.60,
                    "observed_rate": 0.55,
                    "brier_score": brier,
                    "calibration_error": 0.05,
                    "average_net_return": 0.01,
                    "average_excess_return_vs_spy": excess,
                    "win_rate": 0.55,
                    "max_drawdown": drawdown,
                    "first_exit_time_utc": generated.isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                    "last_exit_time_utc": generated.isoformat().replace(
                        "+00:00",
                        "Z",
                    ),
                }
            ],
        }
        return {**identity, "report_id": content_sha256(identity)}


if __name__ == "__main__":
    unittest.main()
