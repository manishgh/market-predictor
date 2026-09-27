from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.governance.outcomes.contracts import (
    RETIRED_INTRADAY,
    MaturedOutcome,
    PredictionMaturationIntent,
    PredictionMonitoringObservation,
    content_sha256,
    maturation_key_sha256,
    monitoring_observation_from_intent,
    monitoring_semantic_sha256,
    semantic_prediction_sha256,
)
from market_predictor.governance.outcomes.performance import (
    build_performance_cohorts,
    load_performance_report,
    validate_performance_report,
    write_performance_report,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from tests.test_outcome_repository import _intent, _outcome

RETIRED_CALIBRATION_FIELDS = (
    "opportunity_observed_rate",
    "opportunity_brier_score",
    "opportunity_calibration_error",
    "mean_downside_probability",
    "downside_observed_rate",
    "downside_brier_score",
    "downside_calibration_error",
)


class PerformanceMonitoringTests(unittest.TestCase):
    def test_rejects_outcome_entered_before_its_prediction_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent_variant("MSFT", "1", probability=0.8)
            evidence = [{"ticker": intent.ticker}]
            base = _outcome(intent, evidence).model_dump(
                mode="python",
                exclude={"outcome_id"},
            )
            base["entry_time_utc"] = intent.decision_time_utc
            outcome = MaturedOutcome.model_validate(
                {**base, "outcome_id": content_sha256(base)}
            )
            repository.record_intent(intent)
            repository.record_outcome(intent, outcome, evidence_rows=evidence)

            with self.assertRaisesRegex(DataReadinessError, "does not match"):
                build_performance_cohorts(
                    repository,
                    generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                    minimum_samples=1,
                )

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
            self.assertEqual(row["mean_decision_score"], row["mean_probability"])
            self.assertFalse(set(RETIRED_CALIBRATION_FIELDS).intersection(row))
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
            self.assertEqual(row["total_predictions"], 1)
            self.assertEqual(row["pending_selected_samples"], 0)
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
            self.assertEqual(row["oldest_pending_decision_session_et"], "2026-07-24")
            self.assertEqual(row["matured_selected_samples"], 0)
            self.assertEqual(row["evidence_status"], "insufficient_evidence")
            self.assertEqual(report["source_intent_ids"], [pending.maturation_key])
            self.assertEqual(report["source_outcome_ids"], [])

    def test_window_is_aligned_to_when_outcomes_finish(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            # Window 3 Jul - 2 Aug. A 26 June decision finishes its ten sessions on 13 July;
            # a 1 June decision finished on 15 June, before the window.
            finishing = _intent_variant(
                "MSFT", "1", probability=0.8, decision_time=datetime(2026, 6, 26, 22, 0, tzinfo=UTC)
            )
            finished = _intent_variant(
                "AAPL", "2", probability=0.8, decision_time=datetime(2026, 6, 1, 22, 0, tzinfo=UTC)
            )
            for intent in (finishing, finished):
                _record(repository, intent, target=1, net_return=0.02, excess_return=0.01)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
                lookback_days=30,
            )
            row = next(item for item in report["rows"] if item["cohort_type"] == "all")

            self.assertEqual(row["total_predictions"], 1)
            self.assertEqual(report["source_intent_ids"], [finishing.maturation_key])
            self.assertEqual(report["window_start_utc"], "2026-07-03T00:00:00Z")
            # Every cohort of the route spans the route's included decisions.
            self.assertEqual({item["window_start_utc"] for item in report["rows"]}, {"2026-06-26T22:00:00Z"})

    def test_window_includes_a_horizon_ending_exactly_at_its_start(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            # The tenth session after 2 July 2026 closes 17 July 20:00 UTC (3 July is a holiday).
            on_edge = _intent_variant("MSFT", "1", probability=0.8, decision_time=datetime(2026, 7, 2, 22, 0, tzinfo=UTC))
            before = _intent_variant("AAPL", "2", probability=0.8, decision_time=datetime(2026, 7, 1, 22, 0, tzinfo=UTC))
            repository.record_intent(on_edge)
            repository.record_intent(before)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 7, 17, 20, 0, tzinfo=UTC) + timedelta(days=30),
                minimum_samples=1,
                lookback_days=30,
            )

            self.assertEqual(report["source_intent_ids"], [on_edge.maturation_key])

    def test_backdated_report_counts_later_outcomes_as_pending(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            # Its outcome matures on 31 July, after the report time.
            matured_later = _intent_variant("MSFT", "1", probability=0.8)
            _record(repository, matured_later, target=1, net_return=0.02, excess_return=0.01)
            still_pending = _intent_variant(
                "AAPL", "2", probability=0.8, decision_time=datetime(2026, 7, 27, 22, 0, tzinfo=UTC)
            )
            repository.record_intent(still_pending)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 7, 30, tzinfo=UTC),
                minimum_samples=1,
            )
            row = next(item for item in report["rows"] if item["cohort_type"] == "all")

            self.assertEqual(row["pending_selected_samples"], 2)
            self.assertEqual(row["route_oldest_pending_decision_session_et"], "2026-07-24")

    def test_route_oldest_pending_counts_selected_decisions_before_the_report_per_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            unselected = _intent_variant(
                "TSLA", "1", probability=0.3, selected=False, decision_time=datetime(2026, 3, 2, 22, 0, tzinfo=UTC)
            )
            selected = _intent_variant(
                "NVDA", "2", probability=0.8, decision_time=datetime(2026, 5, 4, 22, 0, tzinfo=UTC)
            )
            future = _intent_variant(
                "AMD", "3", probability=0.8, decision_time=datetime(2026, 8, 10, 22, 0, tzinfo=UTC)
            )
            other_release = _intent_variant(
                "META", "4", probability=0.8, release="b" * 64, decision_time=datetime(2026, 4, 6, 22, 0, tzinfo=UTC)
            )
            for intent in (unselected, selected, future, other_release):
                repository.record_intent(intent)
            for release in ("a", "b"):
                recent = _intent_variant(
                    f"R{release.upper()}", "5", probability=0.8, release=release * 64,
                    decision_time=datetime(2026, 7, 10, 22, 0, tzinfo=UTC),
                )
                _record(repository, recent, target=1, net_return=0.02, excess_return=0.01)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
                lookback_days=30,
            )
            oldest = {
                row["model_release_id"]: row["route_oldest_pending_decision_session_et"]
                for row in report["rows"]
                if row["cohort_type"] == "all"
            }

            self.assertEqual(oldest, {"a" * 64: "2026-05-04", "b" * 64: "2026-04-06"})

    def test_report_refuses_a_row_window_that_breaks_the_route_rule(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            _record(repository, _intent_variant("MSFT", "1", probability=0.8), target=1, net_return=0.02, excess_return=0.01)
            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
        row = next(item for item in report["rows"] if item["cohort_type"] == "sector")
        changed = {key: value for key, value in row.items() if key != "cohort_id"}
        changed["window_start_utc"] = "2026-07-01T00:00:00Z"
        changed["cohort_id"] = content_sha256(changed)
        candidate = {key: value for key, value in report.items() if key != "report_id"}
        candidate["rows"] = [changed if item is row else item for item in report["rows"]]
        candidate["report_id"] = content_sha256(candidate)

        with self.assertRaisesRegex(ValidationError, "does not follow the route"):
            validate_performance_report(candidate)

    def test_route_oldest_pending_spans_every_stored_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            stuck = _intent_variant(
                "TSLA", "1", probability=0.8, decision_time=datetime(2026, 3, 2, 22, 0, tzinfo=UTC)
            )
            repository.record_intent(stuck)
            recent = _intent_variant(
                "MSFT", "2", probability=0.8, decision_time=datetime(2026, 7, 10, 22, 0, tzinfo=UTC)
            )
            _record(repository, recent, target=1, net_return=0.02, excess_return=0.01)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
                lookback_days=30,
            )
            rows = {row["cohort_type"]: row for row in report["rows"] if row["cohort_type"] in {"all", "sector"}}

            self.assertEqual(rows["all"]["pending_selected_samples"], 0)
            self.assertEqual(rows["all"]["route_oldest_pending_decision_session_et"], "2026-03-02")
            self.assertIsNone(rows["sector"]["route_oldest_pending_decision_session_et"])

    def test_report_reads_only_the_partitions_its_window_can_reach(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            sessions = [
                day
                for day in (date(2026, 1, 2) + timedelta(days=offset) for offset in range(212))
                if day.weekday() < 5 and day not in {date(2025, 9, 1), date(2025, 11, 27), date(2025, 12, 25),
                                                     date(2026, 1, 1), date(2026, 1, 19), date(2026, 2, 16),
                                                     date(2026, 4, 3), date(2026, 5, 25), date(2026, 6, 19),
                                                     date(2026, 7, 3)}
            ]
            for index, session in enumerate(sessions):
                decision = datetime.combine(session, datetime.min.time(), tzinfo=UTC) + timedelta(hours=22)
                repository.record_intent(
                    _intent_variant(f"T{index:03d}", "1", probability=0.8, decision_time=decision)
                )
            opened: list[date] = []
            original = repository.session_intents

            def recording(session: date) -> list[PredictionMaturationIntent]:
                opened.append(session)
                return original(session)

            repository.session_intents = recording  # type: ignore[method-assign]
            build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
                lookback_days=30,
            )

            # 30 days of outcomes plus the ten-session horizon: about 30 of 145 partitions.
            self.assertLess(len(opened), 40)
            self.assertGreaterEqual(min(opened), date(2026, 6, 15))
            self.assertEqual(len(repository.sessions()), len(sessions))

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

    def test_invalid_observation_remains_in_population_denominator(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            valid = _intent_variant("MSFT", "1", probability=0.8)
            _record(
                repository,
                valid,
                target=1,
                net_return=0.10,
                excess_return=0.08,
            )
            base = monitoring_observation_from_intent(valid).model_dump(
                mode="python",
                exclude={"observation_id"},
            )
            base.update(
                {
                    "ticker": "TSLA",
                    "probability": None,
                    "calibration_bin": None,
                    "signal": "not_ready",
                    "rank": None,
                    "selection_eligible": False,
                    "selected_for_policy": False,
                    "actionable": False,
                    "readiness_status": "invalid",
                    "maturation_key": None,
                }
            )
            base.pop("semantic_prediction_id")
            base["semantic_prediction_id"] = monitoring_semantic_sha256(base)
            invalid = PredictionMonitoringObservation.model_validate(
                {**base, "observation_id": content_sha256(base)}
            )
            repository.record_observation(invalid)

            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
            row = next(
                item for item in report["rows"] if item["cohort_type"] == "all"
            )

            self.assertEqual(row["total_predictions"], 2)
            self.assertEqual(row["eligible_predictions"], 1)
            self.assertEqual(row["selected_predictions"], 1)
            self.assertEqual(row["matured_selected_samples"], 1)
            self.assertEqual(len(report["source_observation_ids"]), 2)
            self.assertEqual(len(report["source_intent_ids"]), 1)

    def test_performance_report_rejects_time_regression_and_conflict(self) -> None:
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
            latest = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 3, tzinfo=UTC),
                minimum_samples=1,
            )
            older = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
            conflicting = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 3, tzinfo=UTC),
                minimum_samples=2,
            )
            path = root / "performance.json"
            write_performance_report(path, latest)

            with self.assertRaises(PredictionConflictError):
                write_performance_report(path, older)
            with self.assertRaises(PredictionConflictError):
                write_performance_report(path, conflicting)

    def test_retired_views_fields_and_versions_are_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            _record(
                repository,
                _intent_variant("MSFT", "1", probability=0.8),
                target=1,
                net_return=0.10,
                excess_return=0.08,
            )
            report = build_performance_cohorts(
                repository,
                generated_at=datetime(2026, 8, 2, tzinfo=UTC),
                minimum_samples=1,
            )
        row = report["rows"][0]
        for changes in (
            {"view": "intraday", "horizon": "60m"},
            {"horizon": "10d"},
            {"opportunity_brier_score": 0.2},
        ):
            # Re-hash both identities so only the retired value is wrong.
            changed = {key: value for key, value in {**row, **changes}.items() if key != "cohort_id"}
            changed["cohort_id"] = content_sha256(changed)
            candidate = {key: value for key, value in report.items() if key != "report_id"}
            candidate["rows"] = [changed]
            candidate["report_id"] = content_sha256(candidate)
            with self.subTest(changes=changes), self.assertRaises(ValidationError) as raised:
                validate_performance_report(candidate)
            self.assertNotIn("identity", str(raised.exception))
            if changes.get("view") == "intraday":
                self.assertIn(RETIRED_INTRADAY, str(raised.exception))
        with self.assertRaises(ValidationError):
            validate_performance_report(
                {**report, "contract": "market_predictor.selected_policy_performance.v2"}
            )


def _intent_variant(
    ticker: str,
    snapshot_character: str,
    *,
    probability: float,
    decision_time: datetime | None = None,
    selected: bool = True,
    release: str = "a" * 64,
) -> PredictionMaturationIntent:
    base = _intent().model_dump(
        mode="python",
        exclude={"maturation_key", "semantic_prediction_id", "snapshot_id"},
    )
    decision = decision_time or datetime(2026, 7, 24, 22, 0, tzinfo=UTC)
    base.update(
        {
            "ticker": ticker,
            "canonical_security_id": f"security:{ticker}",
            "model_release_id": release,
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
    return PredictionMaturationIntent.model_validate(
        {
            **base,
            "semantic_prediction_id": semantic_id,
            "snapshot_id": snapshot_id,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic_id),
        }
    )


def _record(
    repository: OutcomeRepository,
    intent: PredictionMaturationIntent,
    *,
    target: int,
    net_return: float,
    excess_return: float,
) -> None:
    evidence = [{"ticker": intent.ticker, "maturation_key": intent.maturation_key}]
    base = _outcome(intent, evidence).model_dump(
        mode="python",
        exclude={"outcome_id"},
    )
    execution_cost_fraction = float(base["execution_cost_bps"]) / 10_000.0
    gross_return = net_return + execution_cost_fraction
    label_cost_fraction = float(base["label_round_trip_cost_bps"]) / 10_000.0
    base.update(
        {
            "net_return": net_return,
            "gross_return": gross_return,
            "label_net_return": gross_return - label_cost_fraction,
            "exit_price": float(base["entry_price"]) * (1.0 + gross_return),
            "path_outcome": "target_first" if target else "stop_first",
            "spy_return": net_return - excess_return,
            "qqq_return": net_return - excess_return,
            "sector_return": net_return - excess_return,
            "excess_return_vs_spy": excess_return,
            "excess_return_vs_qqq": excess_return,
            "excess_return_vs_sector": excess_return,
            "evidence_sha256": content_sha256(evidence),
        }
    )
    outcome = MaturedOutcome.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )
    repository.record_intent(intent)
    repository.record_outcome(intent, outcome, evidence_rows=evidence)


if __name__ == "__main__":
    unittest.main()
