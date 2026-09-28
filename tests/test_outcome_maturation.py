from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pydantic import ValidationError

from market_predictor.collection.outcome_bars import (
    CorporateActionUnit,
    OutcomeBarUnit,
    collect_bars,
    collect_corporate_actions,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import (
    RETIRED_INTRADAY,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
)
from market_predictor.governance.outcomes.evidence import EvidenceTerms
from market_predictor.governance.outcomes.maturation import mature_prediction
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.worker import mature_pending_intents, record_operator_resolution
from market_predictor.modeling.maturation import PendingPath
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.labels import add_exact_swing_labels
from market_predictor.swing.labels.barrier_and_rank import (
    BarrierSpec,
    apply_triple_barrier,
)
from tests.test_outcome_bar_collection import _clock as receipt_clock
from tests.test_outcome_bar_collection import _source as receipt_source
from tests.test_outcome_repository import _intent as swing_intent


class OutcomeMaturationTests(unittest.TestCase):
    def test_swing_zero_volume_observation_remains_pending(self) -> None:
        bars = _swing_bars()
        bars.loc[bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)), "volume"] = 0
        result, evidence = mature_prediction(
            swing_intent(), bars, observed_as_of=datetime(2026, 8, 8, 12, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )
        self.assertIsInstance(result, PendingPath)
        self.assertEqual(evidence, [])

    def test_swing_rejects_non_daily_timeframe(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        bars["timeframe"] = "1m"

        with self.assertRaises(DataReadinessError):
            mature_prediction(
                intent,
                bars,
                observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
                proven_stock_gaps=frozenset(),
            )

    def test_swing_remains_pending_then_matures_on_exact_session_path(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()

        pending, pending_evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 7, 30, 22, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )
        matured, evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )

        self.assertIsInstance(pending, PendingPath)
        self.assertEqual(pending_evidence, [])
        self.assertIsInstance(matured, MaturedOutcome)
        assert isinstance(matured, MaturedOutcome)
        self.assertEqual(matured.path_outcome, "target_first")
        self.assertAlmostEqual(matured.gross_return, 0.03)
        self.assertAlmostEqual(
            matured.label_net_return,
            0.03 - float(intent.label_policy["round_trip_cost_bps"]) / 10_000.0,
        )
        self.assertGreater(len(evidence), 5)
        self.assertEqual(matured.label_available_at_utc, matured.matured_at_utc)

    def test_swing_maturation_matches_offline_label_builder(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        stock = bars[bars["ticker"].eq("MSFT")].copy()
        offline = apply_triple_barrier(
            stock.loc[
                :, ["session_date_et", "open", "high", "low", "close"]
            ].rename(columns={"session_date_et": "session"}),
            pd.DataFrame(
                {
                    "session": [intent.decision_session_et],
                    "atr": [intent.decision_atr_fraction * 100.25],
                }
            ),
            spec=BarrierSpec(
                target_atr_multiple=float(intent.label_policy["target_atr_multiple"]),
                stop_atr_multiple=float(intent.label_policy["stop_atr_multiple"]),
                horizon_sessions=int(intent.label_policy["horizon_sessions"]),
                same_bar_resolution=str(
                    intent.label_policy["same_bar_barrier_resolution"]
                ),
            ),
        ).iloc[0]
        matured, _ = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )

        self.assertIsInstance(matured, MaturedOutcome)
        assert isinstance(matured, MaturedOutcome)
        self.assertAlmostEqual(
            matured.gross_return,
            float(offline["exit_price"]) / 100.0 - 1.0,
        )
        self.assertAlmostEqual(
            matured.label_net_return,
            float(offline["exit_price"]) / 100.0
            - 1.0
            - float(intent.label_policy["round_trip_cost_bps"]) / 10_000.0,
        )
        self.assertEqual(matured.path_outcome, "target_first")

    def test_stop_before_a_gap_waits_until_the_gap_is_proven(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        msft = bars["ticker"].eq("MSFT")
        bars.loc[msft & bars["session_date_et"].eq(date(2026, 7, 28)), "low"] = 50.0
        bars = bars.loc[~(msft & bars["session_date_et"].ge(date(2026, 7, 30)))].copy()
        observed = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

        # The missing sessions may simply not be collected yet, and the outcome is immutable.
        pending, _ = mature_prediction(
            intent, bars, observed_as_of=observed, proven_stock_gaps=frozenset()
        )
        self.assertEqual(pending.reasons, ("required_bar_path_incomplete",))
        self.assertIn("MSFT:2026-07-30", pending.missing_intervals)

        matured, _ = mature_prediction(
            intent,
            bars,
            observed_as_of=observed,
            proven_stock_gaps=frozenset({date(2026, 7, 30)}),
        )
        assert isinstance(matured, MaturedOutcome)
        self.assertEqual((matured.path_outcome, matured.holding_sessions), ("stop_first", 2))
        self.assertIsNone(matured.fixed_horizon_net_return)
        self.assertIsNone(matured.fixed_horizon_excess_return_vs_sector)

    def test_unusable_bars_after_an_early_exit_do_not_hold_the_outcome(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        bars.loc[bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 28)), "low"] = 50.0
        # Benchmarks the managed return does not use, after the exit on July 28.
        for ticker, session in (("QQQ", date(2026, 7, 31)), ("SPY", date(2026, 8, 4))):
            bars.loc[bars["ticker"].eq(ticker) & bars["session_date_et"].eq(session), "volume"] = 0

        matured, _ = mature_prediction(
            intent, bars, observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )

        assert isinstance(matured, MaturedOutcome)
        self.assertEqual((matured.path_outcome, matured.holding_sessions), ("stop_first", 2))
        self.assertIsNotNone(matured.fixed_horizon_net_return)

    def test_an_unusable_stock_bar_ends_the_path_like_a_missing_one(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        msft = bars["ticker"].eq("MSFT")
        bars.loc[msft & bars["session_date_et"].eq(date(2026, 7, 28)), "low"] = 50.0
        bars.loc[msft & bars["session_date_et"].eq(date(2026, 8, 4)), "volume"] = 0
        observed = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

        pending, _ = mature_prediction(
            intent, bars, observed_as_of=observed, proven_stock_gaps=frozenset()
        )
        self.assertEqual(pending.missing_intervals, ("MSFT:2026-08-04", "MSFT:2026-08-05", "MSFT:2026-08-06", "MSFT:2026-08-07"))
        matured, _ = mature_prediction(
            intent,
            bars,
            observed_as_of=observed,
            proven_stock_gaps=frozenset({date(2026, 8, 4)}),
        )
        assert isinstance(matured, MaturedOutcome)
        self.assertIsNone(matured.fixed_horizon_net_return)

    def test_a_later_split_adjustment_leaves_the_outcome_unchanged(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        split = bars.copy()
        # Bars collected after a 2-for-1 split are all halved; the served ATR fraction is not.
        split.loc[split["ticker"].eq("MSFT"), ["open", "high", "low", "close"]] /= 2.0
        observed = datetime(2026, 8, 8, 12, 0, tzinfo=UTC)

        results = [
            mature_prediction(
                intent, frame, observed_as_of=observed, proven_stock_gaps=frozenset()
            )[0]
            for frame in (bars, split)
        ]

        original, adjusted = results
        assert isinstance(original, MaturedOutcome) and isinstance(adjusted, MaturedOutcome)
        self.assertEqual(
            (adjusted.path_outcome, adjusted.holding_sessions), (original.path_outcome, original.holding_sessions)
        )
        self.assertAlmostEqual(adjusted.gross_return, original.gross_return)
        self.assertAlmostEqual(adjusted.execution_cost_bps, original.execution_cost_bps)

    def test_fixed_horizon_returns_equal_the_trainer_labels(self) -> None:
        intent = swing_intent()
        bars = _swing_bars().assign(security_id=lambda frame: "security:" + frame["ticker"])
        matured, _ = mature_prediction(
            intent, bars, observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )
        stock = bars.loc[bars["ticker"].eq("MSFT")]
        decisions = stock.loc[stock["session_date_et"].eq(intent.decision_session_et)].assign(
            decision_group_id="group", feature_eligible=True, primary_benchmark="XLK",
            membership_effective_to_utc=pd.NaT, atr_pct_14=intent.decision_atr_fraction,
        )
        labels = add_exact_swing_labels(
            decisions,
            bars.loc[~bars["ticker"].eq("MSFT")],
            SwingDatasetConfig(horizon_sessions=10, round_trip_cost_bps=float(intent.label_policy["round_trip_cost_bps"])),
            outcome_bars=stock,
        ).iloc[0]

        assert isinstance(matured, MaturedOutcome)
        self.assertAlmostEqual(matured.fixed_horizon_net_return or 0.0, float(labels["future_net_return_10d"]), places=12)
        self.assertAlmostEqual(
            matured.fixed_horizon_excess_return_vs_sector or 0.0,
            float(labels["future_excess_return_10d_vs_sector"]),
            places=12,
        )

    def test_gap_before_any_barrier_stays_pending_on_the_stock(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        msft = bars["ticker"].eq("MSFT")
        bars = bars.loc[~(msft & bars["session_date_et"].ge(date(2026, 7, 29)))].copy()

        pending, evidence = mature_prediction(
            intent, bars, observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )

        self.assertIsInstance(pending, PendingPath)
        self.assertEqual(pending.reasons, ("required_bar_path_incomplete",))
        self.assertIn("MSFT:2026-07-29", pending.missing_intervals)
        self.assertEqual(evidence, [])

    def test_fixed_horizon_returns_follow_the_trainer_target(self) -> None:
        intent = swing_intent()
        matured, _ = mature_prediction(
            intent, _swing_bars(), observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            proven_stock_gaps=frozenset(),
        )

        assert isinstance(matured, MaturedOutcome)
        cost = float(intent.label_policy["round_trip_cost_bps"]) / 10_000.0
        # MSFT opens at 100 on the entry session and closes at 105 on the tenth; the sector
        # ETF (XLK) opens at 201 and closes at 211 over the same interval.
        self.assertAlmostEqual(matured.fixed_horizon_net_return or 0.0, 105.0 / 100.0 - 1.0 - cost)
        self.assertAlmostEqual(
            matured.fixed_horizon_excess_return_vs_sector or 0.0,
            105.0 / 100.0 - 1.0 - cost - (211.0 / 201.0 - 1.0),
        )

    def test_worker_matures_only_canonical_semantic_occurrence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            first = swing_intent(snapshot_id="1" * 64)
            duplicate = swing_intent(snapshot_id="2" * 64)
            repository.record_intent(first)
            repository.record_intent(duplicate)

            summary = _mature(repository, _collected(Path(temp_dir)))

            # A repeated occurrence is never indexed as pending, so the worker never visits it.
            self.assertEqual((summary["index_entries"], summary["matured"]), (1, 1))
            self.assertEqual(summary["not_canonical_dropped"], 0)
            self.assertTrue(repository.has_outcome(first.maturation_key, first.decision_session_et))
            self.assertFalse(repository.has_outcome(duplicate.maturation_key, duplicate.decision_session_et))
            self.assertEqual(repository.pending(), [])

    def test_worker_drops_an_index_entry_whose_outcome_is_already_durable(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            receipts = _collected(Path(temp_dir))
            _mature(repository, receipts)
            # Simulate a crash between writing the outcome and removing its index entry.
            repository._write_pending(intent.maturation_key, intent.decision_session_et)

            summary = _mature(repository, receipts, observed=OBSERVED + timedelta(days=1))

            self.assertEqual((summary["already_matured"], summary["matured"]), (1, 0))
            self.assertEqual(repository.pending(), [])

    def test_worker_waits_for_the_horizon_to_close_without_recording_attempts(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)

            # The tenth session after July 24 closes on August 7.
            summary = _mature(repository, Path(temp_dir) / "receipts", observed=datetime(2026, 8, 7, 19, 0, tzinfo=UTC))

            self.assertEqual((summary["horizon_open"], summary["pending"], summary["matured"]), (1, 0, 0))
            self.assertEqual(repository.pending(), [(intent.maturation_key, intent.decision_session_et)])
            self.assertIsNone(repository.latest_attempt(intent.maturation_key, intent.decision_session_et))

    def test_worker_leaves_an_unfinished_registration_and_drops_an_entry_that_lost_the_race(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            first = swing_intent(snapshot_id="1" * 64)
            repository.record_intent(first)
            session = first.decision_session_et
            # Registration stopped after the index entry, before the semantic record.
            (Path(temp_dir) / "outcomes" / "sessions" / session.isoformat() / "semantic" / f"{first.semantic_prediction_id}.json").unlink()
            receipts = _collected(Path(temp_dir))

            summary = _mature(repository, receipts)
            self.assertEqual((summary["registration_incomplete"], summary["matured"]), (1, 0))
            self.assertEqual(repository.pending(), [(first.maturation_key, session)])

            # Another occurrence registers before the rerun and becomes canonical.
            later = swing_intent(snapshot_id="2" * 64)
            repository.record_intent(later)
            summary = _mature(repository, receipts)

            self.assertEqual((summary["not_canonical_dropped"], summary["matured"]), (1, 1))
            self.assertTrue(repository.has_outcome(later.maturation_key, session))
            self.assertFalse(repository.has_outcome(first.maturation_key, session))
            self.assertEqual(repository.pending(), [])

    def test_a_settled_gap_lets_an_early_exit_mature_without_the_fixed_horizon(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            msft = bars["ticker"].eq("MSFT")
            bars.loc[msft & bars["session_date_et"].eq(date(2026, 7, 28)), "low"] = 50.0
            bars = bars.loc[~(msft & bars["session_date_et"].ge(date(2026, 7, 30)))]

            # Before the settlement period the missing sessions may still arrive.
            unsettled = _collected(Path(temp_dir) / "early", bars, retrieved=datetime(2026, 8, 9, 12, tzinfo=UTC))
            self.assertEqual(_mature(repository, unsettled)["pending"], 1)

            summary = _mature(repository, _collected(Path(temp_dir) / "late", bars))

            self.assertEqual(summary["matured"], 1)
            outcome = repository.load_outcome(intent.maturation_key, intent.decision_session_et)
            self.assertEqual((outcome.path_outcome, outcome.holding_sessions), ("stop_first", 2))
            self.assertIsNone(outcome.fixed_horizon_net_return)

    def test_a_merged_stock_becomes_unresolvable_and_a_bare_gap_waits_for_an_operator(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            # No target or stop is reached by July 28, the last session before the gap.
            bars = bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].ge(date(2026, 7, 29)))]
            receipts = _collected(Path(temp_dir), bars)

            self.assertEqual(_latest(repository, intent, receipts).reasons, ("stock_gap_without_cessation",))

            merger = {"id": "merger-1", "acquiree_symbol": "MSFT", "rate": "104.00",
                      "effective_date": "2026-07-29", "process_date": "2026-08-03"}
            _collected_actions(receipts, {"cash_mergers": [merger]})
            latest = _latest(repository, intent, receipts)

            self.assertEqual((latest.status, latest.reasons), ("unresolvable", ("cash_merger",)))
            self.assertEqual(len(latest.receipt_ids), 2)
            # It stays indexed, so later evidence can still mature it.
            self.assertEqual(repository.pending(), [(intent.maturation_key, intent.decision_session_et)])

    def test_a_rename_and_a_worthless_removal_are_followed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            receipts = _collected(Path(temp_dir), bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].ge(date(2026, 7, 29)))])
            rename = {"id": "rename-1", "old_symbol": "MSFT", "new_symbol": "MSFX", "process_date": "2026-07-28"}
            _collected_actions(receipts, {"name_changes": [rename]})
            removal = {"id": "removal-1", "symbol": "MSFX", "process_date": "2026-11-30"}
            _collected_actions(receipts, {"worthless_removals": [removal]}, symbol="MSFX")

            latest = _latest(repository, intent, receipts)

            self.assertEqual((latest.status, latest.reasons), ("unresolvable", ("worthless_removal",)))

    def test_an_interior_gap_is_unresolvable_only_when_minute_bars_confirm_the_halt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            receipts = _collected(Path(temp_dir), bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)))])

            self.assertEqual(_latest(repository, intent, receipts).reasons, ("interior_gap_needs_minute_bars",))

            _collected_minutes(receipts, traded=True)
            self.assertEqual(_latest(repository, intent, receipts).reasons, ("interior_gap_contradicted",))

        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            repository.record_intent(intent)
            receipts = _collected(Path(temp_dir), bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)))])
            _collected_minutes(receipts, traded=False)

            latest = _latest(repository, intent, receipts)

            self.assertEqual((latest.status, latest.reasons), ("unresolvable", ("interior_gap",)))

    def test_a_minute_receipt_confirms_a_halt_only_once_settled(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            receipts = _collected(Path(temp_dir), bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)))])
            # Retrieved the next morning: final, but before the settlement period ended.
            _collected_minutes(receipts, traded=False, retrieved=datetime(2026, 7, 28, 12, tzinfo=UTC))

            self.assertEqual(_latest(repository, intent, receipts).reasons, ("interior_gap_needs_minute_bars",))

    def test_a_backdated_run_reads_only_receipts_collected_by_its_time(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            receipts = _collected(Path(temp_dir))

            summary = _mature(repository, receipts, observed=RETRIEVED - timedelta(hours=1))

            self.assertEqual((summary["matured"], summary["pending"]), (0, 1))
            latest = repository.latest_attempt(intent.maturation_key, intent.decision_session_et)
            assert latest is not None
            self.assertEqual((latest.reasons, latest.receipt_ids), (("decision_session_not_observed",), ()))
            # A run observing an earlier time than the latest attempt adds nothing to the log.
            summary = _mature(repository, receipts, observed=RETRIEVED - timedelta(days=1))
            self.assertEqual(summary["observed_before_latest_attempt"], 1)

    def test_an_operator_resolution_stands_until_the_evidence_matures_the_outcome(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            bars = _swing_bars()
            gap = bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].ge(date(2026, 7, 29)))]
            receipts = _collected(Path(temp_dir), gap)
            self.assertEqual(_latest(repository, intent, receipts).reasons, ("stock_gap_without_cessation",))

            resolution = record_operator_resolution(
                repository, maturation_key=intent.maturation_key, decision_session=intent.decision_session_et,
                operator_id="operator@example", reference="exchange delisting notice, filing 0001",
                observed_as_of=OBSERVED + timedelta(hours=1), terms=TERMS,
            )
            summary = _mature(repository, receipts, observed=OBSERVED + timedelta(days=1))

            self.assertEqual((resolution.status, resolution.reasons), ("unresolvable", ("operator_verified",)))
            self.assertEqual(summary["unresolvable"], 1)
            self.assertEqual(repository.latest_attempt(intent.maturation_key, intent.decision_session_et), resolution)
            # The late bars arrive after all: the outcome matures and supersedes the resolution.
            _collected(Path(temp_dir), retrieved=OBSERVED + timedelta(days=2))
            summary = _mature(repository, receipts, observed=OBSERVED + timedelta(days=3))
            self.assertEqual(summary["matured"], 1)

    def test_a_new_drift_policy_writes_a_new_attempt(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            receipts = Path(temp_dir) / "receipts"
            _mature(repository, receipts)

            mature_pending_intents(repository, receipts_root=receipts, observed_as_of=OBSERVED + timedelta(days=1),
                                   terms=EvidenceTerms(grace_days=7, settlement_days=2, drift_policy_sha256="e" * 64))

            attempts = repository.attempts(intent.maturation_key, intent.decision_session_et)
            self.assertEqual([attempt.drift_policy_sha256 for attempt in attempts], ["d" * 64, "e" * 64])

    def test_a_session_any_receipt_returned_is_never_a_proven_gap(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            receipts = _collected(Path(temp_dir))
            # A later, settled receipt transiently omits the last sessions the first one returned.
            bars = _swing_bars()
            _collected(Path(temp_dir), bars.loc[~(bars["ticker"].eq("MSFT") & bars["session_date_et"].ge(date(2026, 8, 5)))],
                       retrieved=datetime(2026, 8, 13, 12, tzinfo=UTC))

            summary = _mature(repository, receipts, observed=OBSERVED + timedelta(days=1))

            self.assertEqual(summary["matured"], 1)
            outcome = repository.load_outcome(intent.maturation_key, intent.decision_session_et)
            self.assertIsNotNone(outcome.fixed_horizon_net_return)

    def test_an_unchanged_attempt_is_not_written_again(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir) / "outcomes")
            intent = swing_intent()
            repository.record_intent(intent)
            receipts = Path(temp_dir) / "receipts"

            _mature(repository, receipts)
            _mature(repository, receipts, observed=OBSERVED + timedelta(days=1))

            attempts = repository.attempts(intent.maturation_key, intent.decision_session_et)
            self.assertEqual([attempt.reasons for attempt in attempts], [("decision_session_not_observed",)])
            self.assertEqual((attempts[0].grace_days, attempts[0].settlement_days), (7, 3))

    def test_retired_intraday_and_superseded_intents_are_refused(self) -> None:
        payload = swing_intent().model_dump(mode="python")
        for changes in (
            {"view": "intraday", "horizon": "5m"},
            {"horizon": "10d"},
            {"contract": "market_predictor.maturation_intent.v2"},
            {"downside_probability": 0.2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError) as raised:
                PredictionMaturationIntent.model_validate({**payload, **changes})
            if changes.get("view") == "intraday":
                self.assertIn(RETIRED_INTRADAY, str(raised.exception))


TERMS = EvidenceTerms(grace_days=7, settlement_days=3, drift_policy_sha256="d" * 64)
DECISION = date(2026, 7, 24)
LAST = date(2026, 8, 7)
# Settled: at least three days after the tenth session's close on August 7.
RETRIEVED = datetime(2026, 8, 12, 12, 0, tzinfo=UTC)
OBSERVED = datetime(2026, 8, 13, 12, 0, tzinfo=UTC)


def _mature(repository: OutcomeRepository, receipts: Path, *, observed: datetime = OBSERVED) -> dict[str, int]:
    return mature_pending_intents(repository, receipts_root=receipts, observed_as_of=observed, terms=TERMS)


def _latest(repository: OutcomeRepository, intent: PredictionMaturationIntent, receipts: Path) -> MaturationAttempt:
    _mature(repository, receipts)
    latest = repository.latest_attempt(intent.maturation_key, intent.decision_session_et)
    assert latest is not None
    return latest


def _collected(root: Path, bars: pd.DataFrame | None = None, *, retrieved: datetime = RETRIEVED) -> Path:
    """Collect the fixture's bars as Alpaca returns them into receipts under `root / receipts`."""
    frame = _swing_bars() if bars is None else bars

    def respond(_url: str, params: dict[str, object]) -> tuple[int, object]:
        rows = {
            symbol: [
                {"t": f"{row.session_date_et.isoformat()}T04:00:00Z", "o": row.open, "h": row.high,
                 "l": row.low, "c": row.close, "v": row.volume}
                for row in frame.loc[frame["ticker"].eq(symbol)].itertuples()
            ]
            for symbol in str(params["symbols"]).split(",")
            if frame["ticker"].eq(symbol).any()
        }
        return 200, {"bars": rows, "next_page_token": None}

    unit = OutcomeBarUnit(DECISION, DECISION, LAST, ("MSFT", "QQQ", "SPY", "XLK"))
    collect_bars(receipt_source(respond, retrieved), [unit], root=root / "receipts", clock=receipt_clock(retrieved))
    return root / "receipts"


def _collected_actions(receipts: Path, actions: dict[str, object], *, symbol: str = "MSFT") -> None:
    unit = CorporateActionUnit(DECISION, symbol, DECISION, date(2026, 8, 12))
    source = receipt_source(lambda _url, _params: (200, {"corporate_actions": actions, "next_page_token": None}))
    collect_corporate_actions(source, [unit], root=receipts, clock=receipt_clock())


def _collected_minutes(receipts: Path, *, traded: bool, retrieved: datetime = RETRIEVED) -> None:
    minute = {"t": "2026-07-27T13:30:00Z", "o": 100.0, "h": 100.0, "l": 100.0, "c": 100.0, "v": 10}
    bars = {"MSFT": [minute]} if traded else {}
    source = receipt_source(lambda _url, _params: (200, {"bars": bars, "next_page_token": None}), retrieved)
    unit = OutcomeBarUnit(DECISION, date(2026, 7, 27), date(2026, 7, 27), ("MSFT",), "1Min")
    collect_bars(source, [unit], root=receipts, clock=receipt_clock(retrieved))


def _swing_bars() -> pd.DataFrame:
    sessions = [
        date(2026, 7, 24),
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
        date(2026, 8, 3),
        date(2026, 8, 4),
        date(2026, 8, 5),
        date(2026, 8, 6),
        date(2026, 8, 7),
    ]
    rows: list[dict[str, object]] = []
    for ticker, base in (("SPY", 500.0), ("MSFT", 100.0)):
        ticker_sessions = sessions
        for offset, session in enumerate(ticker_sessions):
            open_price = base + (offset if ticker == "SPY" else max(offset - 1, 0))
            close_price = open_price + (1.0 if ticker == "SPY" else 0.25)
            if ticker == "MSFT" and session == sessions[-1]:
                close_price = 105.0
            rows.append(_daily_row(ticker, session, open_price, close_price))
    for ticker, base in (("QQQ", 400.0), ("XLK", 200.0)):
        for offset, session in enumerate(sessions):
            open_price = base + offset
            rows.append(_daily_row(ticker, session, open_price, open_price + 1.0))
    return pd.DataFrame(rows)


def _daily_row(
    ticker: str,
    session: date,
    open_price: float,
    close_price: float,
) -> dict[str, object]:
    start = datetime.combine(session, time(13, 30), tzinfo=UTC)
    end = datetime.combine(session, time(20, 0), tzinfo=UTC)
    return {
        "ticker": ticker,
        "session_date_et": session,
        "bar_start_utc": start,
        "bar_end_utc": end,
        "available_at_utc": end + timedelta(minutes=15),
        "open": open_price,
        "high": max(open_price, close_price) + 1.0,
        "low": min(open_price, close_price) - 1.0,
        "close": close_price,
        "volume": 1_000_000.0,
        "price_feed": "sip",
        "adjustment": "all",
        "timeframe": "1d",
        "source_artifact_sha256": "9" * 64,
    }


if __name__ == "__main__":
    unittest.main()
