from __future__ import annotations

import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import pandas as pd

from market_predictor.core.prediction_contracts import (
    GlobalContextInfo,
    InvestmentReplayRequest,
    ModelInfo,
    PredictionEvidence,
    PredictionRequest,
    PredictionResponse,
    PredictionRowEvidence,
    ReadinessInfo,
    SwingPrediction,
    TickerPrediction,
)
from market_predictor.serving.investment_replay import InvestmentReplayService, simulate_investment_leg
from market_predictor.serving.snapshot_store import PredictionSnapshotStore


class StaticPriceProvider:
    def __init__(self, frames: dict[str, pd.DataFrame]) -> None:
        self.frames = frames
        self.calls: list[str] = []

    def fetch(
        self,
        ticker: str,
        start: datetime,
        end: datetime,
        *,
        timeframe: str,
    ) -> pd.DataFrame:
        self.calls.append(ticker)
        return self.frames[ticker].copy()


class InvestmentReplayTests(unittest.TestCase):
    def test_replays_stock_against_exact_spy_and_qqq_window(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(store, signal="positive_setup")
            provider = StaticPriceProvider(
                {
                    "MSFT": _daily_bars([100.0, 100.0], [105.0, 110.0]),
                    "SPY": _daily_bars([100.0, 100.0], [101.0, 102.0]),
                    "QQQ": _daily_bars([100.0, 100.0], [101.5, 103.0]),
                }
            )
            service = InvestmentReplayService(
                snapshot_store=store,
                price_provider=provider,
                now=lambda: datetime.fromisoformat("2026-07-06T21:00:00+00:00"),
            )

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00"),
                    slippage_bps=0,
                    commission_bps=0,
                )
            )

            self.assertEqual(result.status, "completed")
            assert result.stock is not None
            self.assertAlmostEqual(result.stock.ending_value, 11_000.0)
            self.assertAlmostEqual(result.benchmarks["SPY"].ending_value, 10_200.0)
            self.assertAlmostEqual(result.benchmarks["QQQ"].ending_value, 10_300.0)
            self.assertAlmostEqual(result.excess_return_vs_spy or 0.0, 0.08)
            self.assertAlmostEqual(result.excess_return_vs_qqq or 0.0, 0.07)
            self.assertEqual(provider.calls, ["MSFT", "SPY", "QQQ"])

    def test_rejects_replay_when_model_was_created_after_decision(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(
                store,
                signal="positive_setup",
                model_created_at="2026-07-02T12:00:00+00:00",
            )
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00"),
                )
            )

            self.assertEqual(result.status, "invalid")
            self.assertIn("model was created after", " | ".join(result.reasons))
            self.assertEqual(provider.calls, [])

    def test_non_actionable_prediction_does_not_invest_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(store, signal="positive_setup", selected=False)
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00"),
                )
            )

            self.assertEqual(result.status, "not_entered")
            self.assertEqual(provider.calls, [])

    def test_force_entry_cannot_override_invalid_prediction_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(store, signal="positive_setup", readiness_status="invalid")
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00"),
                    force_entry=True,
                )
            )

            self.assertEqual(result.status, "invalid")
            self.assertIn("data-readiness status is invalid", " | ".join(result.reasons))
            self.assertEqual(provider.calls, [])


    def test_label_boundary_is_required_and_strictly_before_row_decision(self) -> None:
        for boundary in [None, "2026-07-01T20:00:00+00:00", "2026-07-02T20:00:00+00:00"]:
            with self.subTest(boundary=boundary), tempfile.TemporaryDirectory() as tmp:
                store = PredictionSnapshotStore(Path(tmp))
                snapshot_id = _snapshot(store, signal="positive_setup", label_boundary=boundary,
                    request_as_of="2026-07-06T20:00:00+00:00")
                provider = StaticPriceProvider({})
                result = InvestmentReplayService(snapshot_store=store, price_provider=provider).replay(
                    InvestmentReplayRequest(snapshot_id=snapshot_id, ticker="MSFT", force_entry=True,
                        evaluation_as_of=datetime.fromisoformat("2026-07-07T20:00:00+00:00")))
                self.assertEqual(result.status, "invalid")
                self.assertIn("training-label availability", " | ".join(result.reasons))
                self.assertEqual(result.decision_time, datetime.fromisoformat("2026-07-01T20:00:00+00:00"))
                self.assertEqual(provider.calls, [])

    def test_boundary_immediately_before_row_time_is_accepted_and_exposed(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            boundary = "2026-07-01T19:59:59.999999+00:00"
            snapshot_id = _snapshot(store, signal="positive_setup", label_boundary=boundary,
                request_as_of="2026-07-06T20:00:00+00:00")
            provider = StaticPriceProvider({ticker: _daily_bars([100., 100.], [105., 110.])
                for ticker in ["MSFT", "SPY", "QQQ"]})
            result = InvestmentReplayService(snapshot_store=store, price_provider=provider).replay(
                InvestmentReplayRequest(snapshot_id=snapshot_id, ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00")))
            self.assertEqual(result.status, "completed")
            self.assertEqual(result.model_training_labels_available_through_utc, datetime.fromisoformat(boundary))
            assert result.stock is not None
            self.assertEqual(result.stock.entry_time, datetime.fromisoformat("2026-07-02T13:30:00+00:00"))

    def test_missing_or_duplicate_row_time_refuses_replay(self) -> None:
        for count in [0, 2]:
            with self.subTest(count=count), tempfile.TemporaryDirectory() as tmp:
                store = PredictionSnapshotStore(Path(tmp))
                snapshot_id = _snapshot(store, signal="positive_setup", row_count=count)
                provider = StaticPriceProvider({})
                result = InvestmentReplayService(snapshot_store=store, price_provider=provider).replay(
                    InvestmentReplayRequest(snapshot_id=snapshot_id, ticker="MSFT",
                        evaluation_as_of=datetime.fromisoformat("2026-07-06T21:00:00+00:00")))
                self.assertEqual(result.status, "invalid")
                self.assertIn("exactly one matching", " | ".join(result.reasons))
                self.assertEqual(provider.calls, [])

    def test_daily_bar_uses_early_close(self) -> None:
        bars = pd.DataFrame({"date": ["2026-11-27"], "open": [100.], "close": [105.]})
        result = simulate_investment_leg(bars, ticker="MSFT", timeframe="1Day", initial_capital=10_000.,
            slippage_bps=0., commission_bps=0., decision_time=datetime.fromisoformat("2026-11-25T22:00:00+00:00"),
            evaluation_time=datetime.fromisoformat("2026-11-27T18:00:00+00:00"))
        self.assertEqual(result.entry_time, datetime.fromisoformat("2026-11-27T14:30:00+00:00"))
        self.assertEqual(result.exit_time, datetime.fromisoformat("2026-11-27T18:00:00+00:00"))
        self.assertAlmostEqual(result.ending_value, 10_500.)

    def test_holiday_bar_is_refused(self) -> None:
        bars = pd.DataFrame({"date": ["2026-07-03"], "open": [100.], "close": [105.]})
        with self.assertRaisesRegex(ValueError, "non-XNYS"):
            simulate_investment_leg(bars, ticker="MSFT", timeframe="1Day", initial_capital=10_000.,
                slippage_bps=0., commission_bps=0., decision_time=datetime.fromisoformat("2026-07-02T22:00:00+00:00"),
                evaluation_time=datetime.fromisoformat("2026-07-06T20:00:00+00:00"))


def _snapshot(
    store: PredictionSnapshotStore,
    *,
    signal: str,
    model_created_at: str = "2026-06-30T12:00:00+00:00",
    readiness_status: str = "valid",
    selected: bool = True,
    label_boundary: str | None = "2026-06-29T20:00:00+00:00",
    row_decision: str = "2026-07-01T20:00:00+00:00",
    request_as_of: str = "2026-07-01T20:00:00+00:00",
    row_count: int = 1,
) -> str:
    request = PredictionRequest(
        tickers=["MSFT"],
        mode="swing",
        as_of=datetime.fromisoformat(request_as_of),
    )
    readiness = ReadinessInfo(
        status=readiness_status,  # type: ignore[arg-type]
        timeframe="daily",
        daily_bar_count=260,
        required_bar_count=250,
        latest_price_date="2026-07-01",
        price_feed="sip",
        benchmark_status="present",
        market_context_status="present",
        model_status="promoted",
        source_status="present",
    )
    prediction = SwingPrediction(
        ticker="MSFT",
        date="2026-07-01",
        probability=0.72,
        model_prediction=1,
        signal=signal,
        selected_for_policy=selected and readiness_status == "valid",
        selection_eligible=selected and readiness_status == "valid",
        action="watch_for_entry" if selected and readiness_status == "valid" else "hold_off",
        readiness=readiness,
        global_context=GlobalContextInfo(),
    )
    model = ModelInfo(
        path="models/edge-swing-10b.joblib",
        status="promoted",
        target="target_next_week_big_up",
        artifact_sha256="a" * 64,
        resolved_horizon="10b",
        bar_timeframe="1Day",
        created_at_utc=model_created_at,
        training_data_start="2025-01-01",
        training_data_end="2026-06-29",
        training_labels_available_through_utc=datetime.fromisoformat(label_boundary) if label_boundary else None,
    )
    response = PredictionResponse(
        mode="swing",
        horizon="auto",
        resolved_horizons={"swing": "10b"},
        models={"swing": model},
        evidence=PredictionEvidence(
            request_id="replay-fixture", correlation_id="replay-fixture",
            prediction_cutoff_utc=datetime.fromisoformat(request_as_of),
            row_feature_availability=[PredictionRowEvidence(
                ticker="MSFT", view="swing", decision_time_utc=datetime.fromisoformat(row_decision),
                feature_available_at_utc=datetime.fromisoformat(row_decision),
            ) for _ in range(row_count)],
            serving_policy_id="market_predictor.swing_prediction_policy", serving_policy_sha256="b" * 64,
            identity_status="complete",
        ),
        predictions=[
            TickerPrediction(
                ticker="MSFT",
                final_signal=signal,
                readiness_status=readiness_status,  # type: ignore[arg-type]
                swing=prediction,
            )
        ],
    )
    recorded = store.record(request, response)
    return recorded.snapshot_id or ""


def _daily_bars(opens: list[float], closes: list[float]) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": ["2026-07-02", "2026-07-06"],
            "open": opens,
            "high": [max(open_price, close_price) for open_price, close_price in zip(opens, closes, strict=True)],
            "low": [min(open_price, close_price) for open_price, close_price in zip(opens, closes, strict=True)],
            "close": closes,
            "volume": [1_000_000, 1_000_000],
        }
    )


if __name__ == "__main__":
    unittest.main()
