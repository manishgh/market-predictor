from __future__ import annotations

from datetime import datetime
from pathlib import Path
import tempfile
import unittest

import pandas as pd

from market_predictor.investment_replay import InvestmentReplayService
from market_predictor.prediction_contracts import (
    GlobalContextInfo,
    InvestmentReplayRequest,
    ModelInfo,
    PredictionRequest,
    PredictionResponse,
    ReadinessInfo,
    SwingPrediction,
    UnifiedTickerPrediction,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore


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
            snapshot_id = _snapshot(store, signal="bullish_watch")
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
                now=lambda: datetime.fromisoformat("2026-07-03T21:00:00+00:00"),
            )

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-03T21:00:00+00:00"),
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
                signal="bullish_watch",
                model_created_at="2026-07-02T12:00:00+00:00",
            )
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-03T21:00:00+00:00"),
                )
            )

            self.assertEqual(result.status, "invalid")
            self.assertIn("model was created after", " | ".join(result.reasons))
            self.assertEqual(provider.calls, [])

    def test_non_actionable_prediction_does_not_invest_without_override(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(store, signal="neutral")
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-03T21:00:00+00:00"),
                )
            )

            self.assertEqual(result.status, "not_entered")
            self.assertEqual(provider.calls, [])

    def test_force_entry_cannot_override_invalid_prediction_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            store = PredictionSnapshotStore(Path(tmp))
            snapshot_id = _snapshot(store, signal="bullish_watch", readiness_status="invalid")
            provider = StaticPriceProvider({})
            service = InvestmentReplayService(snapshot_store=store, price_provider=provider)

            result = service.replay(
                InvestmentReplayRequest(
                    snapshot_id=snapshot_id,
                    ticker="MSFT",
                    evaluation_as_of=datetime.fromisoformat("2026-07-03T21:00:00+00:00"),
                    force_entry=True,
                )
            )

            self.assertEqual(result.status, "invalid")
            self.assertIn("data-readiness status is invalid", " | ".join(result.reasons))
            self.assertEqual(provider.calls, [])


def _snapshot(
    store: PredictionSnapshotStore,
    *,
    signal: str,
    model_created_at: str = "2026-06-30T12:00:00+00:00",
    readiness_status: str = "valid",
) -> str:
    request = PredictionRequest(
        tickers=["MSFT"],
        mode="swing",
        as_of=datetime.fromisoformat("2026-07-01T20:00:00+00:00"),
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
        readiness=readiness,
        global_context=GlobalContextInfo(),
    )
    model = ModelInfo(
        path="models/swing.joblib",
        status="promoted",
        target="target_next_week_big_up",
        artifact_sha256="a" * 64,
        resolved_horizon="5d",
        bar_timeframe="1Day",
        created_at_utc=model_created_at,
        training_data_start="2025-01-01",
        training_data_end="2026-06-29",
    )
    response = PredictionResponse(
        mode="swing",
        horizon="auto",
        resolved_horizons={"swing": "5d"},
        models={"swing": model},
        predictions=[
            UnifiedTickerPrediction(
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
            "date": ["2026-07-02", "2026-07-03"],
            "open": opens,
            "high": [max(open_price, close_price) for open_price, close_price in zip(opens, closes)],
            "low": [min(open_price, close_price) for open_price, close_price in zip(opens, closes)],
            "close": closes,
            "volume": [1_000_000, 1_000_000],
        }
    )


if __name__ == "__main__":
    unittest.main()
