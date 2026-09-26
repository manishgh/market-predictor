from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from pydantic import ValidationError

from market_predictor.core.errors import DataReadinessError
from market_predictor.governance.outcomes.contracts import (
    RETIRED_INTRADAY,
    MaturedOutcomeV3,
    PredictionMaturationIntentV3,
)
from market_predictor.governance.outcomes.maturation import mature_prediction
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.worker import mature_pending_intents
from market_predictor.swing.labels.barrier_and_rank import (
    BarrierSpec,
    apply_triple_barrier,
)
from tests.test_outcome_repository import _intent as swing_intent


class OutcomeMaturationTests(unittest.TestCase):
    def test_swing_zero_volume_observation_remains_pending(self) -> None:
        bars = _swing_bars()
        bars.loc[bars["ticker"].eq("MSFT") & bars["session_date_et"].eq(date(2026, 7, 27)), "volume"] = 0
        result, evidence = mature_prediction(
            swing_intent(), bars, observed_as_of=datetime(2026, 8, 8, 12, tzinfo=UTC),
            source_artifact_sha256="9" * 64,
        )
        self.assertEqual(result.status, "pending")
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
                source_artifact_sha256="9" * 64,
            )

    def test_swing_remains_pending_then_matures_on_exact_session_path(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()

        pending, pending_evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 7, 30, 22, 0, tzinfo=UTC),
            source_artifact_sha256="9" * 64,
        )
        matured, evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
            source_artifact_sha256="9" * 64,
        )

        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending_evidence, [])
        self.assertIsInstance(matured, MaturedOutcomeV3)
        assert isinstance(matured, MaturedOutcomeV3)
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
                    "atr": [intent.decision_atr],
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
            source_artifact_sha256="9" * 64,
        )

        self.assertIsInstance(matured, MaturedOutcomeV3)
        assert isinstance(matured, MaturedOutcomeV3)
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

    def test_worker_matures_only_canonical_semantic_occurrence(self) -> None:
        with TemporaryDirectory() as temp_dir:
            repository = OutcomeRepository(Path(temp_dir))
            first = swing_intent(snapshot_id="1" * 64)
            duplicate = swing_intent(snapshot_id="2" * 64)
            repository.record_intent(first)
            repository.record_intent(duplicate)

            summary = mature_pending_intents(
                repository,
                _swing_bars(),
                observed_as_of=datetime(2026, 8, 8, 12, 0, tzinfo=UTC),
                source_artifact_sha256="9" * 64,
            )

            self.assertEqual(summary["matured"], 1)
            self.assertEqual(summary["duplicate_semantic"], 1)
            self.assertTrue(repository.has_outcome(first.maturation_key))
            self.assertFalse(repository.has_outcome(duplicate.maturation_key))

    def test_retired_intraday_and_superseded_intents_are_refused(self) -> None:
        payload = swing_intent().model_dump(mode="python")
        for changes in (
            {"view": "intraday", "horizon": "5m"},
            {"horizon": "10d"},
            {"contract_version": "market_predictor.maturation_intent.v2"},
            {"downside_probability": 0.2},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError) as raised:
                PredictionMaturationIntentV3.model_validate({**payload, **changes})
            if changes.get("view") == "intraday":
                self.assertIn(RETIRED_INTRADAY, str(raised.exception))


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
    }


if __name__ == "__main__":
    unittest.main()
