from __future__ import annotations

from datetime import date, datetime, timedelta
from pathlib import Path
import tempfile
import unittest

import joblib
import numpy as np
import pandas as pd

from market_predictor.prediction_contracts import PredictionRequest
from market_predictor.prediction_service import PredictionService
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.registry import write_model_manifest


class FixedProbabilityModel:
    def __init__(self, probability: float) -> None:
        self.probability = probability

    def predict_proba(self, data: pd.DataFrame) -> np.ndarray:
        return np.column_stack(
            [
                np.full(len(data), 1.0 - self.probability),
                np.full(len(data), self.probability),
            ]
        )


class PredictionServiceTests(unittest.TestCase):
    def test_swing_prediction_uses_promoted_model_and_returns_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_next_week_big_up", status="promoted", probability=0.73)

            response = PredictionService(root).predict_swing(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    swing_dataset=dataset,
                    swing_model=model,
                )
            )

            self.assertEqual(response.mode, "swing")
            self.assertEqual(response.models["swing"].status, "promoted")
            prediction = response.predictions[0].swing
            self.assertIsNotNone(prediction)
            assert prediction is not None
            self.assertEqual(prediction.ticker, "MSFT")
            self.assertAlmostEqual(prediction.probability or 0.0, 0.73)
            self.assertEqual(prediction.signal, "strong_bullish_watch")
            self.assertEqual(prediction.readiness.status, "valid")

    def test_swing_prediction_rejects_candidate_model_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_next_week_big_up", status="candidate", probability=0.73)

            with self.assertRaisesRegex(ValueError, "model must be promoted"):
                PredictionService(root).predict_swing(
                    PredictionRequest(
                        tickers=["MSFT"],
                        mode="swing",
                        swing_dataset=dataset,
                        swing_model=model,
                    )
                )

    def test_unified_response_keeps_swing_when_intraday_model_is_not_promoted(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            swing_dataset = root / "swing_features.parquet"
            intraday_dataset = root / "intraday_features.parquet"
            swing_model = root / "swing.joblib"
            intraday_model = root / "intraday.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(swing_dataset, index=False)
            _swing_frame(["MSFT"], features, rows=260).to_parquet(intraday_dataset, index=False)
            _write_model(swing_model, features, target_col="target_next_week_big_up", status="promoted", probability=0.31)
            _write_model(intraday_model, features, target_col="entry_success", status="candidate", probability=0.80)

            response = PredictionService(root).predict_unified(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="unified",
                    swing_dataset=swing_dataset,
                    swing_model=swing_model,
                    intraday_dataset=intraday_dataset,
                    intraday_model=intraday_model,
                )
            )

            self.assertEqual(response.mode, "unified")
            self.assertTrue(response.errors)
            row = response.predictions[0]
            self.assertIsNotNone(row.swing)
            self.assertIsNone(row.intraday)
            self.assertEqual(row.final_signal, "high_conviction_watch")

    def test_daily_as_of_does_not_use_close_before_market_close(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            frame.to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_next_week_big_up", status="promoted", probability=0.73)
            final_date = frame["date"].iloc[-1]
            cutoff = datetime.fromisoformat(f"{final_date.isoformat()}T15:59:00-04:00")

            response = PredictionService(root).predict_swing(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    as_of=cutoff,
                    swing_dataset=dataset,
                    swing_model=model,
                )
            )

            prediction = response.predictions[0].swing
            assert prediction is not None
            self.assertTrue((prediction.date or "").startswith(str(frame["date"].iloc[-2])))
            self.assertEqual(response.resolved_horizons, {"swing": "5d"})

    def test_explicit_horizon_rejects_incompatible_model_target(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_next_week_big_up", status="promoted", probability=0.73)

            with self.assertRaisesRegex(ValueError, "incompatible with model target horizon 5d"):
                PredictionService(root).predict_swing(
                    PredictionRequest(
                        tickers=["MSFT"],
                        mode="swing",
                        horizon="1d",
                        swing_dataset=dataset,
                        swing_model=model,
                    )
                )

    def test_intraday_as_of_waits_for_bar_close_and_uses_intraday_readiness(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "intraday.parquet"
            model = root / "intraday.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _intraday_frame("MSFT", rows=150)
            frame.to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_entry_success_12b", status="promoted", probability=0.72)
            cutoff = pd.Timestamp(frame["date"].iloc[-1], tz="UTC") + pd.Timedelta(minutes=2)

            response = PredictionService(root).predict_intraday(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="intraday",
                    as_of=cutoff.to_pydatetime(),
                    intraday_dataset=dataset,
                    intraday_model=model,
                )
            )

            prediction = response.predictions[0].intraday
            assert prediction is not None
            self.assertEqual(
                pd.to_datetime(prediction.date, utc=True),
                pd.Timestamp(frame["date"].iloc[-2]).tz_localize("UTC"),
            )
            self.assertEqual(prediction.readiness.timeframe, "intraday")
            self.assertGreaterEqual(prediction.readiness.intraday_bar_count, 130)
            self.assertEqual(prediction.readiness.daily_bar_count, 0)
            self.assertEqual(response.resolved_horizons, {"intraday": "12b"})

    def test_top_level_predict_persists_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_next_week_big_up", status="promoted", probability=0.73)
            service = PredictionService(root)

            response = service.predict(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    swing_dataset=dataset,
                    swing_model=model,
                )
            )

            self.assertIsNotNone(response.snapshot_id)
            self.assertEqual(response.snapshot_id, response.snapshot_sha256)
            self.assertTrue(service.snapshot_store.path_for(response.snapshot_id or "").exists())

    def test_intraday_catalyst_overlay_changes_decision_not_model_probability(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "intraday.parquet"
            model = root / "intraday.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _intraday_frame("MSFT", rows=150)
            frame["news_count_2h"] = 2
            frame["sentiment_mean_2h"] = 0.40
            frame["event_relevance_mean_2h"] = 1.2
            frame["source_count_alpaca_2h"] = 1
            frame["source_count_sec_2h"] = 1
            frame["event_contract_count_2h"] = 1
            frame.to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_entry_success_12b", status="promoted", probability=0.72)

            response = PredictionService(root).predict_intraday(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="intraday",
                    intraday_dataset=dataset,
                    intraday_model=model,
                )
            )

            prediction = response.predictions[0].intraday
            assert prediction is not None
            self.assertAlmostEqual(prediction.probability or 0.0, 0.72)
            self.assertAlmostEqual(prediction.decision_score or 0.0, 0.76)
            self.assertEqual(prediction.catalyst.status, "confirmed")
            self.assertEqual(prediction.signal, "entry_candidate_confirmed")

    def test_live_data_source_uses_registered_feature_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            _write_model(model, features, target_col="target_next_week_big_up", status="promoted", probability=0.73)
            store = LiveFeatureStore(root)
            store.publish("swing", frame, price_feed="sip")

            response = PredictionService(root, live_feature_store=store).predict_swing(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    data_source="live",
                    swing_model=model,
                )
            )

            self.assertEqual(response.data_source, "live")
            prediction = response.predictions[0].swing
            assert prediction is not None
            self.assertAlmostEqual(prediction.probability or 0.0, 0.73)


def _write_model(
    path: Path,
    features: list[str],
    *,
    target_col: str,
    status: str,
    probability: float,
) -> None:
    payload = {
        "model": FixedProbabilityModel(probability),
        "features": features,
        "target_col": target_col,
        "schema_version": "unit.v1",
    }
    joblib.dump(payload, path)
    training = _swing_frame(["MSFT", "AAPL"], features, rows=300)
    training["target"] = [idx % 2 for idx in range(len(training))]
    write_model_manifest(
        model_path=path,
        model_type="unit_test",
        schema_version="unit.v1",
        target_col=target_col,
        features=features,
        training_data=training.assign(**{target_col: training["target"]}),
        metrics={
            "roc_auc": 0.7,
            "top_decile_lift": 2.1,
            "validated_rows": 250,
            "tickers": 2,
        },
        validation_split="date_grouped_purged_walk_forward",
        status=status,
    )


def _swing_frame(tickers: list[str], features: list[str], *, rows: int) -> pd.DataFrame:
    start = date(2025, 1, 1)
    records = []
    for ticker in tickers:
        for idx in range(rows):
            record = {
                "ticker": ticker,
                "date": start + timedelta(days=idx),
                "close": 100.0 + idx,
                "price_feed": "sip",
                "return_1d": 0.01,
                "volume_z20": 1.5,
                "news_count": 2,
                "event_count": 3,
                "sentiment_mean": 0.2,
                "sector_return_1d": 0.005,
                "global_net_impact": 0.0,
            }
            for feature in features:
                record.setdefault(feature, float(idx % 5))
            records.append(record)
    return pd.DataFrame(records)


def _intraday_frame(ticker: str, *, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-08T13:30:00Z", periods=rows, freq="5min")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "date": timestamps.tz_convert(None),
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 100_000.0,
            "price_feed": "sip",
            "return_1d": 0.01,
            "volume_z20": 1.5,
            "qqq_return_1bar": 0.001,
            "market_context_intraday_shock_score_2h": 0.0,
        }
    )


if __name__ == "__main__":
    unittest.main()
