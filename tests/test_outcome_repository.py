from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.outcome_contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV1,
    PredictionMaturationIntentV1,
    content_sha256,
    maturation_key_sha256,
    semantic_prediction_sha256,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.prediction_policy import PREDICTION_POLICY_SHA256
from market_predictor.swing.contracts import SwingDatasetConfig


class OutcomeRepositoryTests(unittest.TestCase):
    def test_records_intent_attempt_and_outcome_idempotently(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            attempt = _attempt(intent)
            evidence = [{"ticker": "MSFT", "bar_start_utc": "2026-07-27T13:30:00+00:00"}]
            outcome = _outcome(intent, evidence)

            repository.record_intent(intent)
            repository.record_attempt(attempt)
            first = repository.record_outcome(outcome, evidence_rows=evidence)
            second = repository.record_outcome(outcome, evidence_rows=evidence)

            self.assertEqual(first, second)
            self.assertEqual(repository.load_intent(intent.maturation_key), intent)
            self.assertEqual(repository.load_outcome(intent.maturation_key), outcome)
            self.assertEqual(
                repository.semantic_canonical_key(intent.semantic_prediction_id),
                intent.maturation_key,
            )

    def test_concurrent_equal_outcome_writers_converge(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            evidence = [{"ticker": "MSFT", "bar_start_utc": "2026-07-27T13:30:00+00:00"}]
            outcome = _outcome(intent, evidence)

            with ThreadPoolExecutor(max_workers=8) as executor:
                results = list(
                    executor.map(
                        lambda _: repository.record_outcome(
                            outcome,
                            evidence_rows=evidence,
                        ),
                        range(24),
                    )
                )

            self.assertTrue(all(result == outcome for result in results))

    def test_repeated_snapshot_occurrences_share_one_semantic_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intent(snapshot_id="1" * 64)
            second = _intent(snapshot_id="2" * 64)

            repository.record_intent(first)
            repository.record_intent(second)

            self.assertNotEqual(first.maturation_key, second.maturation_key)
            self.assertEqual(
                repository.semantic_canonical_key(first.semantic_prediction_id),
                first.maturation_key,
            )


def _intent(snapshot_id: str = "1" * 64) -> PredictionMaturationIntentV1:
    config = SwingDatasetConfig()
    decision = datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
    base: dict[str, object] = {
        "contract_version": "market_predictor.maturation_intent.v1",
        "ticker": "MSFT",
        "canonical_security_id": "security:MSFT",
        "view": "swing",
        "horizon": "5d",
        "decision_time_utc": decision,
        "decision_session_et": date(2026, 7, 24),
        "decision_group_id": decision.isoformat(),
        "model_release_id": "a" * 64,
        "model_artifact_sha256": "b" * 64,
        "feature_artifact_sha256": "c" * 64,
        "serving_policy_sha256": PREDICTION_POLICY_SHA256,
        "label_policy_sha256": config.label_config_sha256(),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "label_policy": config.label_policy(),
        "primary_benchmark": "XLK",
        "market_regime": "risk_on",
        "sector": "Technology",
        "market_cap_bucket": "large",
        "liquidity_bucket": "high",
        "price_feed": "SIP",
        "probability": 0.7,
        "downside_probability": None,
        "calibration_bin": 7,
        "signal": "strong_bullish_watch",
        "actionable": True,
        "catalyst_status": "confirmed",
        "decision_atr": None,
    }
    semantic = semantic_prediction_sha256(base)
    return PredictionMaturationIntentV1.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic),
        }
    )


def _attempt(intent: PredictionMaturationIntentV1) -> MaturationAttemptV1:
    base = {
        "contract_version": "market_predictor.maturation_attempt.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "observed_as_of_utc": datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        "status": "pending",
        "reasons": ("horizon_not_complete",),
        "missing_intervals": (),
    }
    return MaturationAttemptV1.model_validate(
        {**base, "attempt_id": content_sha256(base)}
    )


def _outcome(
    intent: PredictionMaturationIntentV1,
    evidence: list[dict[str, object]],
) -> MaturedOutcomeV1:
    entry = datetime(2026, 7, 27, 13, 30, tzinfo=UTC)
    exit_time = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)
    available = exit_time + timedelta(minutes=15)
    base = {
        "contract_version": "market_predictor.matured_outcome.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "ticker": intent.ticker,
        "view": intent.view,
        "horizon": intent.horizon,
        "entry_time_utc": entry,
        "exit_time_utc": exit_time,
        "label_available_at_utc": available,
        "matured_at_utc": available,
        "entry_price": 100.0,
        "exit_price": 105.0,
        "gross_return": 0.05,
        "net_return": 0.049,
        "mfe": 0.07,
        "mae": -0.02,
        "path_outcome": "positive",
        "opportunity_target": 1,
        "downside_target": None,
        "spy_return": 0.01,
        "qqq_return": 0.012,
        "sector_return": 0.008,
        "excess_return_vs_spy": 0.039,
        "excess_return_vs_qqq": 0.037,
        "excess_return_vs_sector": 0.041,
        "evidence_sha256": content_sha256(evidence),
    }
    return MaturedOutcomeV1.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )


if __name__ == "__main__":
    unittest.main()
