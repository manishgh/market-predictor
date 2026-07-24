from __future__ import annotations

import unittest
from datetime import UTC, date, datetime, time, timedelta
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd

from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.intraday.labels import add_exact_one_minute_labels
from market_predictor.outcome_contracts import (
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
    maturation_key_sha256,
    semantic_prediction_sha256,
)
from market_predictor.outcome_maturation import mature_prediction
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.outcome_worker import mature_pending_intents
from market_predictor.prediction_policy import (
    DEFAULT_PREDICTION_POLICY,
    PREDICTION_POLICY_SHA256,
)
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.dataset import _add_exact_labels
from tests.test_outcome_repository import _intent as swing_intent


class OutcomeMaturationTests(unittest.TestCase):
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
            observed_as_of=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            source_artifact_sha256="9" * 64,
        )

        self.assertEqual(pending.status, "pending")
        self.assertEqual(pending_evidence, [])
        self.assertIsInstance(matured, MaturedOutcomeV1)
        assert isinstance(matured, MaturedOutcomeV1)
        self.assertEqual(matured.path_outcome, "positive")
        self.assertAlmostEqual(matured.gross_return, 0.05)
        self.assertAlmostEqual(matured.net_return, 0.049)
        self.assertGreater(len(evidence), 5)
        self.assertEqual(matured.label_available_at_utc, matured.matured_at_utc)

    def test_intraday_uses_exact_entry_and_stop_first_ambiguity(self) -> None:
        intent = _intraday_intent()
        bars = _intraday_bars(ambiguous=True)

        result, evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 7, 24, 14, 10, tzinfo=UTC),
            source_artifact_sha256="8" * 64,
        )

        self.assertIsInstance(result, MaturedOutcomeV1)
        assert isinstance(result, MaturedOutcomeV1)
        self.assertEqual(result.entry_time_utc, intent.decision_time_utc)
        self.assertEqual(result.path_outcome, "stop_first")
        self.assertEqual(result.opportunity_target, 0)
        self.assertEqual(result.downside_target, 1)
        self.assertAlmostEqual(result.exit_price, 99.25)
        self.assertTrue(evidence)

    def test_intraday_missing_entry_bar_never_shifts_forward(self) -> None:
        intent = _intraday_intent()
        bars = _intraday_bars(ambiguous=False)
        bars = bars[
            ~(
                bars["ticker"].eq("MSFT")
                & bars["bar_start_utc"].eq(pd.Timestamp(intent.decision_time_utc))
            )
        ].copy()

        result, evidence = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 7, 24, 14, 10, tzinfo=UTC),
            source_artifact_sha256="8" * 64,
        )

        self.assertEqual(result.status, "pending")
        self.assertIn("required_bar_path_incomplete", result.reasons)
        self.assertTrue(
            any(intent.decision_time_utc.isoformat() in item for item in result.missing_intervals)
        )
        self.assertEqual(evidence, [])

    def test_intraday_maturation_matches_offline_label_builder(self) -> None:
        intent = _intraday_intent()
        bars = _intraday_bars(ambiguous=True)
        config = IntradayDatasetConfig(horizon_minutes=5)
        decision = pd.DataFrame(
            [
                {
                    "ticker": intent.ticker,
                    "decision_time_utc": intent.decision_time_utc,
                    "session_date_et": intent.decision_session_et,
                    "session_minute_et": 10 * 60,
                    "atr_14_price_5m": intent.decision_atr,
                    "primary_benchmark": intent.primary_benchmark,
                    "feature_eligible": True,
                }
            ]
        )
        offline = add_exact_one_minute_labels(decision, bars, config).iloc[0]
        matured, _ = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 7, 24, 14, 10, tzinfo=UTC),
            source_artifact_sha256="8" * 64,
        )

        self.assertIsInstance(matured, MaturedOutcomeV1)
        assert isinstance(matured, MaturedOutcomeV1)
        self.assertEqual(matured.path_outcome, offline["path_outcome"])
        self.assertAlmostEqual(
            matured.gross_return,
            float(offline["path_realized_return_gross_5m"]),
        )
        self.assertAlmostEqual(
            matured.net_return,
            float(offline["path_realized_return_net_5m"]),
        )
        self.assertAlmostEqual(matured.mfe, float(offline["path_mfe_5m"]))
        self.assertAlmostEqual(matured.mae, float(offline["path_mae_5m"]))

    def test_swing_maturation_matches_offline_label_builder(self) -> None:
        intent = swing_intent()
        bars = _swing_bars()
        stock = bars[bars["ticker"].eq("MSFT")].copy()
        stock["feature_eligible"] = True
        stock["primary_benchmark"] = "XLK"
        stock["decision_group_id"] = stock["session_date_et"].astype(str)
        benchmarks = bars[bars["ticker"].isin({"SPY", "QQQ", "XLK"})].copy()
        offline = _add_exact_labels(
            stock,
            benchmarks,
            SwingDatasetConfig(),
        ).iloc[0]
        matured, _ = mature_prediction(
            intent,
            bars,
            observed_as_of=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
            source_artifact_sha256="9" * 64,
        )

        self.assertIsInstance(matured, MaturedOutcomeV1)
        assert isinstance(matured, MaturedOutcomeV1)
        self.assertAlmostEqual(
            matured.gross_return,
            float(offline["future_gross_return_5d"]),
        )
        self.assertAlmostEqual(
            matured.net_return,
            float(offline["future_net_return_5d"]),
        )
        self.assertAlmostEqual(matured.mfe, float(offline["future_mfe_5d"]))
        self.assertAlmostEqual(matured.mae, float(offline["future_mae_5d"]))

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
                observed_as_of=datetime(2026, 8, 1, 12, 0, tzinfo=UTC),
                source_artifact_sha256="9" * 64,
            )

            self.assertEqual(summary["matured"], 1)
            self.assertEqual(summary["duplicate_semantic"], 1)
            self.assertTrue(repository.has_outcome(first.maturation_key))
            self.assertFalse(repository.has_outcome(duplicate.maturation_key))


def _intraday_intent() -> PredictionMaturationIntentV2:
    config = IntradayDatasetConfig(horizon_minutes=5)
    decision = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)
    base: dict[str, object] = {
        "contract_version": "market_predictor.maturation_intent.v2",
        "ticker": "MSFT",
        "canonical_security_id": "security:MSFT",
        "view": "intraday",
        "horizon": "5m",
        "decision_time_utc": decision,
        "decision_session_et": date(2026, 7, 24),
        "decision_group_id": decision.isoformat(),
        "model_release_id": "a" * 64,
        "model_artifact_sha256": "b" * 64,
        "feature_artifact_sha256": "c" * 64,
        "prediction_policy_sha256": PREDICTION_POLICY_SHA256,
        "label_policy_sha256": config.label_config_sha256(),
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "prediction_policy": DEFAULT_PREDICTION_POLICY.specification(),
        "label_policy": config.label_policy(),
        "primary_benchmark": "XLK",
        "market_regime": "risk_on",
        "sector": "Technology",
        "market_cap_bucket": "large",
        "liquidity_bucket": "high",
        "price_feed": "SIP",
        "probability": 0.72,
        "downside_probability": 0.2,
        "calibration_bin": 7,
        "signal": "entry_candidate",
        "rank": 1,
        "selection_eligible": True,
        "selected_for_policy": True,
        "actionable": True,
        "catalyst_status": "confirmed",
        "decision_atr": 1.0,
    }
    semantic = semantic_prediction_sha256(base)
    snapshot_id = "1" * 64
    return PredictionMaturationIntentV2.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic),
        }
    )


def _swing_bars() -> pd.DataFrame:
    sessions = [
        date(2026, 7, 24),
        date(2026, 7, 27),
        date(2026, 7, 28),
        date(2026, 7, 29),
        date(2026, 7, 30),
        date(2026, 7, 31),
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
    for ticker, entry, exit_price in (("QQQ", 400.0, 408.0), ("XLK", 200.0, 202.0)):
        rows.append(_daily_row(ticker, sessions[1], entry, entry + 1.0))
        rows.append(_daily_row(ticker, sessions[-1], exit_price - 1.0, exit_price))
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
    }


def _intraday_bars(*, ambiguous: bool) -> pd.DataFrame:
    start = datetime(2026, 7, 24, 14, 0, tzinfo=UTC)
    rows: list[dict[str, object]] = []
    for ticker, base in (("MSFT", 100.0), ("SPY", 500.0), ("QQQ", 400.0), ("XLK", 200.0)):
        for offset in range(5):
            moment = start + timedelta(minutes=offset)
            high = base + 0.2
            low = base - 0.2
            open_price = base
            close = base + 0.05
            if ticker == "MSFT" and offset == 0 and ambiguous:
                high = 101.25
                low = 99.0
                open_price = 100.0
                close = 100.0
            rows.append(
                {
                    "ticker": ticker,
                    "session_date_et": date(2026, 7, 24),
                    "bar_start_utc": moment,
                    "bar_end_utc": moment + timedelta(minutes=1),
                    "available_at_utc": moment + timedelta(minutes=1),
                    "open": open_price,
                    "high": high,
                    "low": low,
                    "close": close,
                    "volume": 100_000.0,
                    "price_feed": "sip",
                    "adjustment": "all",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
