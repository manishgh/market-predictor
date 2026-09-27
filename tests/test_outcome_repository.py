from __future__ import annotations

import itertools
import json
import tempfile
import unittest
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, date, datetime, timedelta
from pathlib import Path
from typing import Any
from unittest import mock

import exchange_calendars as xcals
import pandas as pd
from pydantic import ValidationError

import market_predictor.governance.outcomes.repository as repository_module
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import PredictionConflictError
from market_predictor.execution_policy import (
    DEFAULT_EXECUTION_POLICY,
    EXECUTION_POLICY_SHA256,
    round_trip_cost_bps,
)
from market_predictor.governance.outcomes.contracts import (
    RETIRED_INTRADAY,
    MaturationAttempt,
    MaturedOutcome,
    PredictionMaturationIntent,
    PredictionMonitoringObservation,
    content_sha256,
    maturation_key_sha256,
    monitoring_observation_from_intent,
    semantic_prediction_sha256,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.sessions import session_after
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
            "contract": "market_predictor.prediction_policy.v2"
        }
        semantic = semantic_prediction_sha256(valid)
        with self.assertRaisesRegex(ValidationError, "swing prediction policy"):
            PredictionMaturationIntent.model_validate(
                {
                    **valid,
                    "semantic_prediction_id": semantic,
                    "maturation_key": maturation_key_sha256(
                        str(valid["snapshot_id"]), semantic
                    ),
                }
            )

    def test_superseded_observation_and_outcome_versions_are_refused(self) -> None:
        intent = _intent()
        observation = monitoring_observation_from_intent(intent).model_dump(mode="python", exclude={"observation_id"})
        outcome = _outcome(intent, [{"ticker": "MSFT"}]).model_dump(mode="python", exclude={"outcome_id"})
        for model, content, key, version in (
            (PredictionMonitoringObservation, observation, "observation_id", "market_predictor.prediction_observation.v1"),
            (MaturedOutcome, outcome, "outcome_id", "market_predictor.matured_outcome.v2"),
        ):
            changed = {**content, "contract": version}
            with self.subTest(version=version), self.assertRaises(ValidationError):
                model.model_validate({**changed, key: content_sha256(changed)})

    def test_loaders_name_stored_retired_intraday_records(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            path = next((Path(temp_dir) / "sessions").glob("*/intents/*.json"))
            stored = json.loads(path.read_text(encoding="utf-8"))
            path.write_text(json.dumps({**stored, "view": "intraday", "horizon": "5m"}), encoding="utf-8")

            for load in (
                lambda: repository.session_intents(intent.decision_session_et),
                lambda: repository.load_intent(intent.maturation_key, intent.decision_session_et),
            ):
                with self.assertRaisesRegex(DataReadinessError, RETIRED_INTRADAY):
                    load()

    def test_pending_index_follows_canonical_intents_until_their_outcome(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intent(snapshot_id="1" * 64)
            repeated = _intent(snapshot_id="2" * 64)
            repository.record_intent(first)
            repository.record_intent(repeated)
            session = first.decision_session_et

            # A repeated occurrence of the same prediction is never canonical, so never pending.
            self.assertEqual(repository.pending(), [(first.maturation_key, session)])
            self.assertEqual(repository.sessions(), (session,))
            self.assertEqual(repository.session_horizons(session), frozenset({"10b"}))

            # A canonical entry without an outcome is never dropped.
            with self.assertRaises(PredictionConflictError):
                repository.drop_pending(first.maturation_key, session)

            evidence = [{"ticker": "MSFT"}]
            repository.record_outcome(first, _outcome(first, evidence), evidence_rows=evidence)
            self.assertEqual(repository.pending(), [])

            # A later occurrence, or a rerun of the canonical one, never indexes it again.
            repository.record_intent(first)
            repository.record_intent(_intent(snapshot_id="3" * 64))
            self.assertEqual(repository.pending(), [])

    def test_registration_rerun_repairs_a_crash_at_every_write(self) -> None:
        real_write = repository_module._write_json_durable
        written: list[Path] = []

        def counting(path: Path, value: object) -> None:
            written.append(path)
            real_write(path, value)

        with tempfile.TemporaryDirectory() as temp_dir, mock.patch.object(
            repository_module, "_write_json_durable", counting
        ):
            OutcomeRepository(Path(temp_dir)).record_intent(_intent())
        # Intent, horizons, observation, index entry and semantic record.
        self.assertEqual([path.parent.name for path in written], ["intents", "2026-07-24", "observations", "pending", "semantic"])

        for failing in range(len(written)):
            with self.subTest(crash_before=written[failing].parent.name), tempfile.TemporaryDirectory() as temp_dir:
                repository = OutcomeRepository(Path(temp_dir))
                intent = _intent()
                session = intent.decision_session_et

                with mock.patch.object(
                    repository_module, "_write_json_durable", _crash_before(real_write, failing)
                ), self.assertRaises(OSError):
                    repository.record_intent(intent)
                repository.record_intent(intent)

                self.assertEqual(repository.pending(), [(intent.maturation_key, session)])
                self.assertEqual(
                    repository.semantic_canonical_key(intent.semantic_prediction_id, session), intent.maturation_key
                )
                self.assertEqual(repository.session_horizons(session), frozenset({"10b"}))
                self.assertEqual(
                    repository.session_observations(session, {intent.maturation_key: intent}),
                    [monitoring_observation_from_intent(intent)],
                )

        # A semantic record without its index entry (the order before this repair) is also restored.
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            (Path(temp_dir) / "pending" / f"{intent.maturation_key}.json").unlink()

            repository.record_intent(intent)

            self.assertEqual(repository.pending(), [(intent.maturation_key, intent.decision_session_et)])

    def test_index_entry_that_vanishes_during_a_rerun_is_written_again(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            entry = Path(temp_dir) / "pending" / f"{intent.maturation_key}.json"
            real_read = Path.read_bytes

            def vanishing(path: Path) -> bytes:
                # The outcome writer removes the entry between the rerun's existence check and its read.
                if path == entry:
                    path.unlink()
                    raise FileNotFoundError(path)
                return real_read(path)

            with mock.patch.object(Path, "read_bytes", vanishing):
                repository.record_intent(intent)

            self.assertEqual(repository.pending(), [(intent.maturation_key, intent.decision_session_et)])

    def test_outcome_entry_must_open_the_session_after_the_decision(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            evidence = [{"ticker": "MSFT"}]
            late = _outcome(intent, evidence, entry_offset=2)

            with self.assertRaises(PredictionConflictError):
                repository.record_outcome(intent, late, evidence_rows=evidence)

    def test_outcome_holding_period_and_timeout_bind_their_fixed_horizon_values(self) -> None:
        intent = _intent()
        content = _outcome(intent, [{"ticker": "MSFT"}]).model_dump(mode="python", exclude={"outcome_id"})
        cases = {
            "exit before its holding period": {"exit_time_utc": content["exit_time_utc"] - timedelta(days=1)},
            "timeout without the fixed horizon": {
                "fixed_horizon_net_return": None,
                "fixed_horizon_excess_return_vs_sector": None,
            },
            "timeout with another fixed-horizon return": {"fixed_horizon_net_return": 0.01},
        }
        for name, changes in cases.items():
            changed = {**content, **changes}
            with self.subTest(name), self.assertRaises(ValidationError):
                MaturedOutcome.model_validate({**changed, "outcome_id": content_sha256(changed)})

    def test_pending_index_holds_only_current_entries(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            evidence = [{"ticker": "MSFT"}]
            repository.record_outcome(intent, _outcome(intent, evidence), evidence_rows=evidence)

            self.assertEqual(list((Path(temp_dir) / "pending").iterdir()), [])

    def test_concurrent_registration_indexes_one_canonical_occurrence(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            occurrences = [_intent(snapshot_id=f"{digit:x}" * 64) for digit in range(1, 9)]
            session = occurrences[0].decision_session_et

            with ThreadPoolExecutor(max_workers=8) as executor:
                list(executor.map(repository.record_intent, occurrences))
                list(executor.map(lambda horizon: repository._record_horizon(session, horizon), [f"{n}b" for n in range(1, 17)]))

            canonical = repository.semantic_canonical_key(occurrences[0].semantic_prediction_id, session)
            self.assertIn(canonical, {intent.maturation_key for intent in occurrences})
            self.assertEqual(repository.pending(), [(canonical, session)])
            self.assertEqual(len(repository.session_intents(session)), len(occurrences))
            self.assertEqual(repository.session_horizons(session), frozenset(f"{n}b" for n in range(1, 17)))

    def test_flat_layout_is_refused(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            (Path(temp_dir) / "outcomes").mkdir()

            with self.assertRaisesRegex(DataReadinessError, "flat layout"):
                OutcomeRepository(Path(temp_dir))

    def test_sessions_skip_plain_files_and_refuse_unknown_directories(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            (Path(temp_dir) / "sessions" / "desktop.ini").write_text("", encoding="utf-8")

            self.assertEqual(repository.sessions(), (intent.decision_session_et,))

            (Path(temp_dir) / "sessions" / "misc").mkdir()
            with self.assertRaises(PredictionConflictError):
                repository.sessions()

    def test_attempt_is_filed_only_under_its_intent_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            attempt = _attempt(intent)

            with self.assertRaises(PredictionConflictError):
                repository.record_attempt(attempt, decision_session=intent.decision_session_et)
            repository.record_intent(intent)
            with self.assertRaises(PredictionConflictError):
                repository.record_attempt(attempt, decision_session=intent.decision_session_et + timedelta(days=3))
            repository.record_attempt(attempt, decision_session=intent.decision_session_et)

    def test_reads_and_replacements_retry_a_file_held_open_elsewhere(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            session = intent.decision_session_et

            with mock.patch.object(repository_module.time, "sleep"):
                with mock.patch.object(repository_module.os, "replace", _held_open(repository_module.os.replace, 2)):
                    repository.record_intent(intent)
                with mock.patch.object(Path, "read_bytes", _held_open(Path.read_bytes, 2)):
                    self.assertEqual(repository.session_horizons(session), frozenset({"10b"}))
                # A file still held open after every retry is a conflict, never a partial read.
                with mock.patch.object(Path, "read_bytes", _held_open(Path.read_bytes, 5)), self.assertRaises(
                    PredictionConflictError
                ):
                    repository.session_horizons(session)

    def test_lookups_are_bound_to_the_decision_session(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            other_session = intent.decision_session_et + timedelta(days=3)

            with self.assertRaises(FileNotFoundError):
                repository.load_intent(intent.maturation_key, other_session)
            self.assertIsNone(
                repository.semantic_canonical_key(intent.semantic_prediction_id, other_session)
            )
            self.assertFalse(repository.has_outcome(intent.maturation_key, other_session))

    def test_swing_outcome_rejects_legacy_path_and_calibration_target(self) -> None:
        intent = _intent()
        evidence = [{"ticker": "MSFT"}]
        valid = _outcome(intent, evidence).model_dump(
            mode="python",
            exclude={"outcome_id"},
        )
        for legacy in ({"path_outcome": "positive"}, {"opportunity_target": 1}, {"downside_target": 0}):
            changed = {**valid, **legacy}
            with self.subTest(legacy=legacy), self.assertRaises(ValidationError):
                MaturedOutcome.model_validate(
                    {**changed, "outcome_id": content_sha256(changed)}
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
            PredictionMaturationIntent.model_validate(
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
            MaturedOutcome.model_validate(
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
            MaturedOutcome.model_validate(
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
            repository.record_attempt(attempt, decision_session=intent.decision_session_et)
            first = repository.record_outcome(intent, outcome, evidence_rows=evidence)
            second = repository.record_outcome(intent, outcome, evidence_rows=evidence)

            self.assertEqual(first, second)
            self.assertEqual(repository.load_intent(intent.maturation_key, intent.decision_session_et), intent)
            self.assertEqual(repository.load_outcome(intent.maturation_key, intent.decision_session_et), outcome)
            self.assertEqual(
                repository.semantic_canonical_key(intent.semantic_prediction_id, intent.decision_session_et),
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
                            intent,
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
            repository.record_outcome(intent, outcome, evidence_rows=evidence)

            conflicting_content = outcome.model_dump(
                mode="python",
                exclude={"outcome_id"},
            )
            conflicting_content["mfe"] = 0.08
            conflicting = MaturedOutcome.model_validate(
                {
                    **conflicting_content,
                    "outcome_id": content_sha256(conflicting_content),
                }
            )

            with self.assertRaises(PredictionConflictError):
                repository.record_outcome(intent, conflicting, evidence_rows=evidence)

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
            content["fixed_horizon_net_return"] = content["label_net_return"]
            content["fixed_horizon_excess_return_vs_sector"] = float(content["label_net_return"]) - float(content["sector_return"])
            outcome = MaturedOutcome.model_validate(
                {**content, "outcome_id": content_sha256(content)}
            )

            with self.assertRaises(PredictionConflictError):
                repository.record_outcome(intent, outcome, evidence_rows=evidence)

    def test_repository_rejects_duplicate_json_keys_and_nonfinite_values(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            path = (
                Path(temp_dir)
                / "sessions"
                / intent.decision_session_et.isoformat()
                / "intents"
                / f"{intent.maturation_key}.json"
            )
            path.parent.mkdir(parents=True)

            path.write_text('{"ticker":"MSFT","ticker":"AAPL"}', encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                repository.load_intent(intent.maturation_key, intent.decision_session_et)

            path.write_text('{"probability":NaN}', encoding="utf-8")
            with self.assertRaises(PredictionConflictError):
                repository.load_intent(intent.maturation_key, intent.decision_session_et)

    def test_repeated_snapshot_occurrences_share_one_semantic_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = _intent(snapshot_id="1" * 64)
            second = _intent(snapshot_id="2" * 64)

            repository.record_intent(first)
            repository.record_intent(second)

            self.assertNotEqual(first.maturation_key, second.maturation_key)
            self.assertEqual(
                repository.semantic_canonical_key(first.semantic_prediction_id, first.decision_session_et),
                first.maturation_key,
            )

    def test_semantic_index_corruption_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            repository.record_intent(intent)
            path = (
                Path(temp_dir)
                / "sessions"
                / intent.decision_session_et.isoformat()
                / "semantic"
                / f"{intent.semantic_prediction_id}.json"
            )
            payload = path.read_text(encoding="utf-8").replace(
                "market_predictor.semantic_prediction",
                "market_predictor.semantic_prediction.corrupt",
            )
            path.write_text(payload, encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.semantic_canonical_key(intent.semantic_prediction_id, intent.decision_session_et)
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
            tampered = PredictionMonitoringObservation.model_validate(
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
            tampered = PredictionMonitoringObservation.model_validate(
                {**content, "observation_id": content_sha256(content)}
            )
            path = (
                Path(temp_dir)
                / "sessions"
                / observation.decision_session_et.isoformat()
                / "observations"
                / f"{observation.observation_id}.json"
            )
            path.write_text(tampered.model_dump_json(indent=2), encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.session_observations(intent.decision_session_et, {intent.maturation_key: intent})

    def test_outcome_read_rebinds_execution_inputs_to_intent(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            intent = _intent()
            evidence = [{"ticker": "MSFT"}]
            outcome = _outcome(intent, evidence)
            repository.record_intent(intent)
            repository.record_outcome(intent, outcome, evidence_rows=evidence)
            content = outcome.model_dump(mode="python", exclude={"outcome_id"})
            content["decision_atr_fraction"] = 0.02
            execution_cost_bps = round_trip_cost_bps(
                price=float(content["entry_price"]),
                atr_pct=0.02 * float(content["decision_close"]) / float(content["entry_price"]),
                participation=0.0,
                policy=DEFAULT_EXECUTION_POLICY,
            )
            net_return = float(content["gross_return"]) - execution_cost_bps / 10_000.0
            content["execution_cost_bps"] = execution_cost_bps
            content["net_return"] = net_return
            content["excess_return_vs_spy"] = net_return - float(content["spy_return"])
            content["excess_return_vs_qqq"] = net_return - float(content["qqq_return"])
            content["excess_return_vs_sector"] = net_return - float(content["sector_return"])
            tampered = MaturedOutcome.model_validate(
                {**content, "outcome_id": content_sha256(content)}
            )
            path = (
                Path(temp_dir)
                / "sessions"
                / intent.decision_session_et.isoformat()
                / "outcomes"
                / f"{intent.maturation_key}.json"
            )
            path.write_text(tampered.model_dump_json(indent=2), encoding="utf-8")

            with self.assertRaises(PredictionConflictError):
                repository.load_outcome(intent.maturation_key, intent.decision_session_et)


def _crash_before(write: Callable[[Path, object], None], failing: int) -> Callable[[Path, object], None]:
    """`write`, stopped as a crash would stop it before its `failing`-th call (counted from zero)."""
    calls = itertools.count()

    def call(path: Path, value: object) -> None:
        if next(calls) == failing:
            raise OSError("simulated crash")
        write(path, value)

    return call


def _held_open(operation: Callable[..., Any], refusals: int) -> Callable[..., Any]:
    """`operation`, refused as Windows refuses a file another process holds open, `refusals` times."""
    remaining = [refusals]

    def call(*args: Any, **kwargs: Any) -> Any:
        if remaining[0] > 0:
            remaining[0] -= 1
            raise PermissionError("file is held open by another process")
        return operation(*args, **kwargs)

    return call


def _intent(snapshot_id: str = "1" * 64) -> PredictionMaturationIntent:
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
        "contract": "market_predictor.maturation_intent",
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
        "calibration_bin": 7,
        "signal": "strong_bullish_watch",
        "rank": 1,
        "selection_eligible": True,
        "selected_for_policy": True,
        "actionable": True,
        "catalyst_status": "confirmed",
        # An ATR of 1.0 on MSFT's decision close of 100.25 in the maturation bar fixture.
        "decision_atr_fraction": 1.0 / 100.25,
    }
    semantic = semantic_prediction_sha256(base)
    return PredictionMaturationIntent.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic),
        }
    )


def _attempt(intent: PredictionMaturationIntent) -> MaturationAttempt:
    base = {
        "contract": "market_predictor.maturation_attempt",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "observed_as_of_utc": datetime(2026, 7, 26, 12, 0, tzinfo=UTC),
        "status": "pending",
        "reasons": ("horizon_not_complete",),
        "missing_intervals": (),
    }
    return MaturationAttempt.model_validate(
        {**base, "attempt_id": content_sha256(base)}
    )


def _outcome(
    intent: PredictionMaturationIntent,
    evidence: list[dict[str, object]],
    *,
    entry_offset: int = 1,
) -> MaturedOutcome:
    """A ten-session timeout entered `entry_offset` sessions after the decision."""
    calendar = xcals.get_calendar("XNYS")
    entry_session = session_after(intent.decision_session_et, entry_offset)
    entry = calendar.session_open(pd.Timestamp(entry_session)).to_pydatetime()
    exit_time = calendar.session_close(pd.Timestamp(session_after(entry_session, 9))).to_pydatetime()
    available = exit_time + timedelta(minutes=15)
    decision_close = 100.25
    execution_cost_bps = round_trip_cost_bps(
        price=100.0,
        atr_pct=intent.decision_atr_fraction * decision_close / 100.0,
        participation=0.0,
        policy=DEFAULT_EXECUTION_POLICY,
    )
    label_cost_bps = float(intent.label_policy["round_trip_cost_bps"])
    net_return = 0.05 - execution_cost_bps / 10_000.0
    base = {
        "contract": "market_predictor.matured_outcome",
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
        "decision_atr_fraction": intent.decision_atr_fraction,
        "decision_close": decision_close,
        "execution_participation_fraction": 0.0,
        "execution_cost_bps": execution_cost_bps,
        "net_return": net_return,
        "mfe": 0.07,
        "mae": -0.02,
        "path_outcome": "timeout",
        "spy_return": 0.01,
        "qqq_return": 0.012,
        "sector_return": 0.008,
        "excess_return_vs_spy": net_return - 0.01,
        "excess_return_vs_qqq": net_return - 0.012,
        "excess_return_vs_sector": net_return - 0.008,
        "holding_sessions": 10,
        "fixed_horizon_net_return": 0.05 - label_cost_bps / 10_000.0,
        "fixed_horizon_excess_return_vs_sector": 0.05 - label_cost_bps / 10_000.0 - 0.008,
        "evidence_sha256": content_sha256(evidence),
    }
    return MaturedOutcome.model_validate(
        {**base, "outcome_id": content_sha256(base)}
    )


if __name__ == "__main__":
    unittest.main()
