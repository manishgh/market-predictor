from __future__ import annotations

import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    EXECUTION_POLICY_SHA256,
    round_trip_cost_bps,
)
from market_predictor.governance.outcomes.contracts import (
    MaturationAttemptV1,
    MaturedOutcomeV2,
    PredictionMaturationIntentV2,
    PredictionMonitoringObservationV1,
    content_sha256,
    maturation_key_sha256,
    monitoring_observation_from_intent,
    semantic_prediction_sha256,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.label_policy import policy_sha256
from market_predictor.modeling.prediction_selection import SwingPredictionPolicy
from market_predictor.modeling.strategy_contract import load_strategy_contract
from market_predictor.swing.contracts.outcome_policy import (
    swing_outcome_policy,
    swing_outcome_policy_sha256,
)

ROOT = Path(__file__).resolve().parents[1]


class OutcomeRepositoryTests(unittest.TestCase):
    def test_swing_intent_rejects_intraday_policy_contract(self) -> None:
        valid = _intent().model_dump(
            mode="python",
            exclude={"maturation_key", "semantic_prediction_id"},
        )
        valid["prediction_policy"] = {
            "contract_version": "market_predictor.prediction_policy.v2"
        }
        semantic = semantic_prediction_sha256(valid)
        with self.assertRaisesRegex(ValidationError, "swing prediction policy"):
            PredictionMaturationIntentV2.model_validate(
                {
                    **valid,
                    "semantic_prediction_id": semantic,
                    "maturation_key": maturation_key_sha256(
                        str(valid["snapshot_id"]), semantic
                    ),
                }
            )

    def test_swing_outcome_rejects_legacy_path_and_calibration_target(self) -> None:
        intent = _intent()
        evidence = [{"ticker": "MSFT"}]
        valid = _outcome(intent, evidence).model_dump(
            mode="python",
            exclude={"outcome_id"},
        )
        valid.update({"path_outcome": "positive", "opportunity_target": 1})
        with self.assertRaisesRegex(ValidationError, "managed barrier semantics"):
            MaturedOutcomeV2.model_validate(
                {**valid, "outcome_id": content_sha256(valid)}
            )

    def test_swing_intent_rejects_mismatched_label_horizon(self) -> None:
        valid = _intent().model_dump(
            mode="python",
            exclude={"maturation_key", "semantic_prediction_id"},
        )
        label_policy = dict(valid["label_policy"])
        label_policy["horizon_sessions"] = 9
        valid["label_policy"] = label_policy
        valid["label_policy_sha256"] = policy_sha256(label_policy)
        semantic = semantic_prediction_sha256(valid)

        with self.assertRaisesRegex(ValidationError, "policy horizons"):
            PredictionMaturationIntentV2.model_validate(
                {
                    **valid,
                    "semantic_prediction_id": semantic,
                    "maturation_key": maturation_key_sha256(
                        str(valid["snapshot_id"]), semantic
                    ),
                }
            )

    def test_outcome_rejects_availability_before_exit(self) -> None:
        intent = _intent()
        valid = _outcome(intent, [{"ticker": "MSFT"}]).model_dump(
            mode="python",
            exclude={"outcome_id"},
        )
        invalid_time = valid["exit_time_utc"] - timedelta(microseconds=1)
        valid["label_available_at_utc"] = invalid_time
        valid["matured_at_utc"] = invalid_time

        with self.assertRaisesRegex(ValidationError, "before its exit"):
            MaturedOutcomeV2.model_validate(
                {**valid, "outcome_id": content_sha256(valid)}
            )

    def test_outcome_rejects_inconsistent_return_arithmetic(self) -> None:
        intent = _intent()
        evidence = [{"ticker": "MSFT"}]
        valid = _outcome(intent, evidence).model_dump(
            mode="python",
            exclude={"outcome_id"},
        )
        valid["excess_return_vs_spy"] = 0.50

        with self.assertRaisesRegex(ValidationError, "benchmark excess return"):
            MaturedOutcomeV2.model_validate(
                {**valid, "outcome_id": content_sha256(valid)}
            )

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

    def test_conflicting_existing_outcome_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            evidence = [{"ticker": "MSFT"}]
            outcome = _outcome(intent, evidence)
            repository.record_intent(intent)
            repository.record_outcome(outcome, evidence_rows=evidence)

            conflicting_content = outcome.model_dump(
                mode="python",
                exclude={"outcome_id"},
            )
            conflicting_content["mfe"] = 0.08
            conflicting = MaturedOutcomeV2.model_validate(
                {
                    **conflicting_content,
                    "outcome_id": content_sha256(conflicting_content),
                }
            )

            with self.assertRaises(PredictionConflictError):
                repository.record_outcome(conflicting, evidence_rows=evidence)

    def test_outcome_cost_policy_must_match_its_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            evidence = [{"ticker": "MSFT"}]
            repository.record_intent(intent)
            content = _outcome(intent, evidence).model_dump(
                mode="python",
                exclude={"outcome_id"},
            )
            content["label_round_trip_cost_bps"] = 5.0
            content["label_net_return"] = float(content["gross_return"]) - 0.0005
            outcome = MaturedOutcomeV2.model_validate(
                {**content, "outcome_id": content_sha256(content)}
            )

            with self.assertRaises(PredictionConflictError):
                repository.record_outcome(outcome, evidence_rows=evidence)

    def test_repository_rejects_duplicate_json_keys_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            path = (
                Path(temp_dir)
                / "intents"
                / intent.maturation_key[:2]
                / f"{intent.maturation_key}.json"
            )
            path.parent.mkdir(parents=True)

            path.write_text('{"ticker":"MSFT","ticker":"AAPL"}', encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                repository.load_intent(intent.maturation_key)

            path.write_text('{"probability":NaN}', encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                repository.load_intent(intent.maturation_key)

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

    def test_semantic_index_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            path = (
                Path(temp_dir)
                / "semantic"
                / intent.semantic_prediction_id[:2]
                / f"{intent.semantic_prediction_id}.json"
            )
            payload = path.read_text(encoding="utf-8").replace(
                "market_predictor.semantic_prediction.v1",
                "market_predictor.semantic_prediction.corrupt",
            )
            path.write_text(payload, encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.semantic_canonical_key(intent.semantic_prediction_id)
            with self.assertRaises(PredictionConflictError):
                repository.record_intent(intent)

    def test_observation_must_match_its_persisted_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            content = monitoring_observation_from_intent(intent).model_dump(
                mode="python",
                exclude={"observation_id"},
            )
            content["probability"] = 0.99
            content["calibration_bin"] = 9
            tampered = PredictionMonitoringObservationV1.model_validate(
                {**content, "observation_id": content_sha256(content)}
            )

            with self.assertRaises(PredictionConflictError):
                repository.record_observation(tampered)

    def test_observation_read_rebinds_to_intent_and_storage_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            observation = monitoring_observation_from_intent(intent)
            content = observation.model_dump(mode="python", exclude={"observation_id"})
            content["probability"] = 0.99
            content["calibration_bin"] = 9
            tampered = PredictionMonitoringObservationV1.model_validate(
                {**content, "observation_id": content_sha256(content)}
            )
            path = (
                Path(temp_dir)
                / "observations"
                / observation.observation_id[:2]
                / f"{observation.observation_id}.json"
            )
            path.write_text(tampered.model_dump_json(indent=2), encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.observations()

    def test_outcome_read_rebinds_execution_inputs_to_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            evidence = [{"ticker": "MSFT"}]
            outcome = _outcome(intent, evidence)
            repository.record_intent(intent)
            repository.record_outcome(outcome, evidence_rows=evidence)
            content = outcome.model_dump(mode="python", exclude={"outcome_id"})
            content["decision_atr"] = 0.01
            execution_cost_bps = round_trip_cost_bps(
                price=float(content["entry_price"]),
                atr_pct=0.01 / float(content["entry_price"]),
                participation=0.0,
                policy=DEFAULT_EXECUTION_POLICY,
            )
            net_return = float(content["gross_return"]) - execution_cost_bps / 10_000.0
            content["execution_cost_bps"] = execution_cost_bps
            content["net_return"] = net_return
            content["excess_return_vs_spy"] = net_return - float(content["spy_return"])
            content["excess_return_vs_qqq"] = net_return - float(content["qqq_return"])
            content["excess_return_vs_sector"] = net_return - float(content["sector_return"])
            tampered = MaturedOutcomeV2.model_validate(
                {**content, "outcome_id": content_sha256(content)}
            )
            path = (
                Path(temp_dir)
                / "outcomes"
                / intent.maturation_key[:2]
                / f"{intent.maturation_key}.json"
            )
            path.write_text(tampered.model_dump_json(indent=2), encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.load_outcome(intent.maturation_key)


def _intent(snapshot_id: str = "1" * 64) -> PredictionMaturationIntentV2:
    strategy = load_strategy_contract(
        ROOT / "configs" / "edge_rebuild_strategy_contract.toml"
    )
    swing_contract = strategy.swing
    prediction_policy = SwingPredictionPolicy(
        horizon_sessions=swing_contract.horizon_sessions,
        minimum_probability=0.5,
        maximum_predictions_per_decision=10,
        target_maximum_sector_weight=1.0,
        hard_maximum_sector_weight=1.0,
        minimum_distinct_sectors=1,
    )
    decision = datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
    base: dict[str, object] = {
        "contract_version": "market_predictor.maturation_intent.v2",
        "ticker": "MSFT",
        "canonical_security_id": "security:MSFT",
        "view": "swing",
        "horizon": "10b",
        "decision_time_utc": decision,
        "decision_session_et": date(2026, 7, 24),
        "decision_group_id": decision.isoformat(),
        "model_release_id": "a" * 64,
        "model_artifact_sha256": "b" * 64,
        "feature_artifact_sha256": "c" * 64,
        "prediction_policy_sha256": prediction_policy.sha256(),
        "label_policy_sha256": swing_outcome_policy_sha256(swing_contract),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "prediction_policy": prediction_policy.specification(),
        "label_policy": swing_outcome_policy(swing_contract),
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
        "rank": 1,
        "selection_eligible": True,
        "selected_for_policy": True,
        "actionable": True,
        "catalyst_status": "confirmed",
        "decision_atr": 1.0,
    }
    semantic = semantic_prediction_sha256(base)
    return PredictionMaturationIntentV2.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic),
        }
    )


def _attempt(intent: PredictionMaturationIntentV2) -> MaturationAttemptV1:
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
    intent: PredictionMaturationIntentV2,
    evidence: list[dict[str, object]],
) -> MaturedOutcomeV2:
    entry = datetime(2026, 7, 27, 13, 30, tzinfo=UTC)
    exit_time = datetime(2026, 7, 31, 20, 0, tzinfo=UTC)
    available = exit_time + timedelta(minutes=15)
    execution_cost_bps = round_trip_cost_bps(
        price=100.0,
        atr_pct=float(intent.decision_atr or 0.0) / 100.0,
        participation=0.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    label_cost_bps = float(intent.label_policy["round_trip_cost_bps"])
    net_return = 0.05 - execution_cost_bps / 10_000.0
    base = {
        "contract_version": "market_predictor.matured_outcome.v2",
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
        "label_round_trip_cost_bps": label_cost_bps,
        "label_net_return": 0.05 - label_cost_bps / 10_000.0,
        "execution_policy_sha256": intent.execution_policy_sha256,
        "decision_atr": intent.decision_atr,
        "execution_participation_fraction": 0.0,
        "execution_cost_bps": execution_cost_bps,
        "net_return": net_return,
        "mfe": 0.07,
        "mae": -0.02,
        "path_outcome": "timeout",
        "opportunity_target": 0 if intent.view == "intraday" else None,
        "downside_target": 0 if intent.view == "intraday" else None,
        "spy_return": 0.01,
        "qqq_return": 0.012,
        "sector_return": 0.008,
        "excess_return_vs_spy": net_return - 0.01,
        "excess_return_vs_qqq": net_return - 0.012,
        "excess_return_vs_sector": net_return - 0.008,
        "evidence_sha256": content_sha256(evidence),
    }
    return MaturedOutcomeV2.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )


if __name__ == "__main__":
    unittest.main()
