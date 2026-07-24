from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.outcome_contracts import (
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
    content_sha256,
    maturation_key_sha256,
    semantic_prediction_sha256,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.performance_monitoring import (
    build_performance_cohorts,
    load_performance_report,
    write_performance_report,
)
from tests.test_outcome_repository import _intent, _outcome


class PerformanceMonitoringTests(unittest.TestCase):
    def test_aggregates_calibration_economics_and_drawdown(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intent_variant(
                "MSFT",
                "1",
                probability=0.8,
                decision_time=datetime(2026, 7, 23, 22, 0, tzinfo=UTC),
            )
            second = _intent_variant(
                "AAPL",
                "2",
                probability=0.2,
                decision_time=datetime(2026, 7, 24, 22, 0, tzinfo=UTC),
            )
            unselected = _intent_variant(
                "TSLA",
                "3",
                probability=0.99,
                decision_time=datetime(2026, 7, 24, 22, 0, tzinfo=UTC),
                selected=False,
            )
            _record(
                repository,
                first,
                target=1,
                net_return=0.10,
                excess_return=0.08,
            )
            _record(
                repository,
                second,
                target=0,
                net_return=-0.05,
                excess_return=-0.06,
            )
            _record(
                repository,
                unselected,
                target=1,
                net_return=0.90,
                excess_return=0.80,
            )

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=2,
            )
            row = next(
                item
                for item in report["rows"]
                if item["cohort_type"] == "all"
            )

            self.assertEqual(row["total_predictions"], 3)
            self.assertEqual(row["selected_predictions"], 2)
            self.assertEqual(row["actionable_predictions"], 2)
            self.assertEqual(row["matured_selected_samples"], 2)
            self.assertEqual(row["pending_selected_samples"], 0)
            self.assertEqual(row["evidence_status"], "sufficient")
            self.assertAlmostEqual(row["selection_rate"], 2 / 3)
            self.assertAlmostEqual(row["opportunity_brier_score"], 0.04)
            self.assertAlmostEqual(row["opportunity_calibration_error"], 0.20)
            self.assertAlmostEqual(row["average_net_return"], 0.025)
            self.assertAlmostEqual(row["average_excess_return_vs_spy"], 0.01)
            self.assertAlmostEqual(row["win_rate"], 0.5)
            self.assertAlmostEqual(row["max_drawdown"], 0.05)
            self.assertEqual(len(report["source_intent_ids"]), 3)
            self.assertEqual(len(report["source_outcome_ids"]), 2)

    def test_excludes_noncanonical_repeated_snapshot_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            canonical = _intent_variant("MSFT", "1", probability=0.8)
            repeated = _intent_variant("MSFT", "2", probability=0.8)
            self.assertEqual(
                canonical.semantic_prediction_id,
                repeated.semantic_prediction_id,
            )
            _record(
                repository,
                canonical,
                target=1,
                net_return=0.10,
                excess_return=0.08,
            )
            _record(
                repository,
                repeated,
                target=0,
                net_return=-0.50,
                excess_return=-0.60,
            )

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
            row = next(
                item
                for item in report["rows"]
                if item["cohort_type"] == "all"
            )

            self.assertEqual(row["matured_selected_samples"], 1)
            self.assertAlmostEqual(row["average_net_return"], 0.10)
            self.assertEqual(len(report["source_outcome_ids"]), 1)

    def test_drawdown_equal_weights_predictions_in_one_decision_group(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intent_variant("MSFT", "1", probability=0.8)
            second = _intent_variant("AAPL", "2", probability=0.2)
            _record(
                repository,
                first,
                target=0,
                net_return=-0.10,
                excess_return=-0.10,
            )
            _record(
                repository,
                second,
                target=1,
                net_return=0.10,
                excess_return=0.10,
            )

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=2,
            )
            row = next(
                item
                for item in report["rows"]
                if item["cohort_type"] == "all"
            )

            self.assertAlmostEqual(row["max_drawdown"], 0.0)

    def test_pending_selection_and_rolling_window_are_explicit(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            pending = _intent_variant(
                "MSFT",
                "1",
                probability=0.8,
                decision_time=datetime(2026, 7, 24, 22, 0, tzinfo=UTC),
            )
            old = _intent_variant(
                "AAPL",
                "2",
                probability=0.7,
                decision_time=datetime(2026, 5, 1, 22, 0, tzinfo=UTC),
            )
            repository.record_intent(pending)
            _record(
                repository,
                old,
                target=1,
                net_return=0.20,
                excess_return=0.15,
            )

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
                lookback_days=30,
            )
            row = next(
                item
                for item in report["rows"]
                if item["cohort_type"] == "all"
            )

            self.assertEqual(row["total_predictions"], 1)
            self.assertEqual(row["pending_selected_samples"], 1)
            self.assertEqual(row["matured_selected_samples"], 0)
            self.assertEqual(row["evidence_status"], "insufficient_evidence")
            self.assertIsNone(row["opportunity_brier_score"])
            self.assertEqual(report["source_intent_ids"], [pending.maturation_key])
            self.assertEqual(report["source_outcome_ids"], [])

    def test_intraday_reports_downside_calibration_separately(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intraday_intent_variant(
                "NVDA",
                "1",
                opportunity_probability=0.8,
                downside_probability=0.2,
            )
            second = _intraday_intent_variant(
                "AMD",
                "2",
                opportunity_probability=0.6,
                downside_probability=0.4,
            )
            _record(
                repository,
                first,
                target=1,
                downside_target=0,
                net_return=0.03,
                excess_return=0.02,
            )
            _record(
                repository,
                second,
                target=0,
                downside_target=1,
                net_return=-0.02,
                excess_return=-0.03,
            )

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=2,
            )
            row = next(
                item
                for item in report["rows"]
                if item["cohort_type"] == "all"
            )

            self.assertEqual(row["view"], "intraday")
            self.assertAlmostEqual(row["opportunity_brier_score"], 0.20)
            self.assertAlmostEqual(row["downside_brier_score"], 0.20)
            self.assertAlmostEqual(row["mean_downside_probability"], 0.30)
            self.assertAlmostEqual(row["downside_observed_rate"], 0.50)

    def test_persisted_report_round_trip_rejects_mutation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository = OutcomeRepository(root / "outcomes")
            intent = _intent_variant("MSFT", "1", probability=0.8)
            _record(
                repository,
                intent,
                target=1,
                net_return=0.10,
                excess_return=0.08,
            )
            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
            path = root / "performance.json"
            write_performance_report(path, report)
            self.assertEqual(load_performance_report(path), report)

            mutated = path.read_text(encoding="utf-8").replace(
                '"total_predictions": 1',
                '"total_predictions": 2',
                1,
            )
            path.write_text(mutated, encoding="utf-8")
            with self.assertRaises(ValueError):
                load_performance_report(path)


def _intent_variant(
    ticker: str,
    snapshot_character: str,
    *,
    probability: float,
    decision_time: datetime | None = None,
    selected: bool = True,
) -> PredictionMaturationIntentV2:
    base = _intent().model_dump(
        mode="python",
        exclude={"maturation_key", "semantic_prediction_id", "snapshot_id"},
    )
    decision = decision_time or datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
    base.update(
        {
            "ticker": ticker,
            "canonical_security_id": f"security:{ticker}",
            "probability": probability,
            "calibration_bin": min(9, int(probability * 10)),
            "decision_time_utc": decision,
            "decision_session_et": decision.date(),
            "decision_group_id": decision.isoformat(),
            "signal": (
                "strong_bullish_watch" if selected else "neutral"
            ),
            "rank": 1 if selected else 2,
            "selected_for_policy": selected,
            "actionable": selected,
        }
    )
    semantic_id = semantic_prediction_sha256(base)
    snapshot_id = snapshot_character * 64
    return PredictionMaturationIntentV2.model_validate(
        {
            **base,
            "semantic_prediction_id": semantic_id,
            "snapshot_id": snapshot_id,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic_id),
        }
    )


def _intraday_intent_variant(
    ticker: str,
    snapshot_character: str,
    *,
    opportunity_probability: float,
    downside_probability: float,
) -> PredictionMaturationIntentV2:
    config = IntradayDatasetConfig()
    base = _intent().model_dump(
        mode="python",
        exclude={"maturation_key", "semantic_prediction_id", "snapshot_id"},
    )
    decision = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)
    base.update(
        {
            "ticker": ticker,
            "canonical_security_id": f"security:{ticker}",
            "view": "intraday",
            "horizon": "60m",
            "decision_time_utc": decision,
            "decision_session_et": decision.date(),
            "decision_group_id": decision.isoformat(),
            "probability": opportunity_probability,
            "downside_probability": downside_probability,
            "calibration_bin": min(9, int(opportunity_probability * 10)),
            "label_policy": config.label_policy(),
            "label_policy_sha256": config.label_config_sha256(),
            "decision_atr": 1.0,
            "signal": "entry_candidate",
        }
    )
    semantic_id = semantic_prediction_sha256(base)
    snapshot_id = snapshot_character * 64
    return PredictionMaturationIntentV2.model_validate(
        {
            **base,
            "semantic_prediction_id": semantic_id,
            "snapshot_id": snapshot_id,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic_id),
        }
    )


def _record(
    repository: OutcomeRepository,
    intent: PredictionMaturationIntentV2,
    *,
    target: int,
    downside_target: int | None = None,
    net_return: float,
    excess_return: float,
) -> None:
    evidence = [{"ticker": intent.ticker, "maturation_key": intent.maturation_key}]
    base = _outcome(intent, evidence).model_dump(
        mode="python",
        exclude={"outcome_id"},
    )
    base.update(
        {
            "opportunity_target": target,
            "downside_target": downside_target,
            "net_return": net_return,
            "gross_return": net_return + 0.001,
            "path_outcome": "positive" if target else "negative",
            "excess_return_vs_spy": excess_return,
            "evidence_sha256": content_sha256(evidence),
        }
    )
    outcome = MaturedOutcomeV1.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )
    repository.record_intent(intent)
    repository.record_outcome(outcome, evidence_rows=evidence)


if __name__ == "__main__":
    unittest.main()
