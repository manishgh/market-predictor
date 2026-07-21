from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, date, datetime, timedelta
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
)
from market_predictor.prediction_contracts import PredictionDataSource, PredictionRequest
from market_predictor.prediction_service import (
    PredictionService,
    ServingRoute,
    serving_routes_from_config,
)
from market_predictor.registry import write_model_manifest
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)


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
    def test_serving_routes_are_loaded_from_server_configuration(self) -> None:
        routes = serving_routes_from_config(
            {
                "prediction_serving": {
                    "routes": {
                        "swing": {
                            "5d": {
                                "model": "models/swing.joblib",
                                "bar_timeframe": "1Day",
                            }
                        }
                    }
                }
            }
        )

        self.assertEqual(routes["swing"]["5d"].model, Path("models/swing.joblib"))

    def test_swing_prediction_uses_promoted_model_and_returns_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)

            response = _service(root, swing=(dataset, model)).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing"))

            self.assertEqual(response.mode, "swing")
            self.assertEqual(response.models["swing"].status, "promoted")
            prediction = response.predictions[0].swing
            self.assertIsNotNone(prediction)
            assert prediction is not None
            self.assertEqual(prediction.ticker, "MSFT")
            self.assertAlmostEqual(prediction.probability or 0.0, 0.73)
            self.assertEqual(prediction.signal, "strong_bullish_watch_confirmed")
            self.assertEqual(prediction.readiness.status, "valid")

    def test_swing_prediction_rejects_candidate_model_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_net_positive_5d", status="candidate", probability=0.73)

            with self.assertRaisesRegex(ValueError, "status candidate is not allowed"):
                _service(root, swing=(dataset, model)).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing"))

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
            _write_model(swing_model, features, target_col="target_net_positive_5d", status="promoted", probability=0.70)
            _write_model(
                intraday_model,
                features,
                target_col="target_before_stop_60m",
                status="candidate",
                probability=0.80,
            )

            response = _service(
                root,
                swing=(swing_dataset, swing_model),
                intraday=(intraday_dataset, intraday_model),
            ).predict_unified(PredictionRequest(tickers=["MSFT"], mode="unified"))

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
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)
            final_date = frame["date"].iloc[-1]
            cutoff = datetime.fromisoformat(f"{final_date.isoformat()}T15:59:00-04:00")

            response = _service(root, swing=(dataset, model)).predict_swing(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    as_of=cutoff,
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
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)

            with self.assertRaisesRegex(ValueError, "incompatible with model target horizon 5d"):
                _service(root, swing=(dataset, model), swing_horizon="1d").predict_swing(
                    PredictionRequest(
                        tickers=["MSFT"],
                        mode="swing",
                        horizon="1d",
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
            _write_model(
                model,
                features,
                target_col="target_before_stop_60m",
                status="promoted",
                probability=0.72,
            )
            cutoff = pd.Timestamp(frame["date"].iloc[-1], tz="UTC") + pd.Timedelta(minutes=2)

            response = _service(root, intraday=(dataset, model)).predict_intraday(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="intraday",
                    as_of=cutoff.to_pydatetime(),
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
            self.assertEqual(response.resolved_horizons, {"intraday": "60m"})

    def test_top_level_predict_persists_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)
            service = _service(root, swing=(dataset, model))

            response = service.predict(PredictionRequest(tickers=["MSFT"], mode="swing"))

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
            _write_model(
                model,
                features,
                target_col="target_before_stop_60m",
                status="promoted",
                probability=0.72,
            )

            response = _service(root, intraday=(dataset, model)).predict_intraday(PredictionRequest(tickers=["MSFT"], mode="intraday"))

            prediction = response.predictions[0].intraday
            assert prediction is not None
            self.assertAlmostEqual(prediction.opportunity_probability or 0.0, 0.72)
            self.assertAlmostEqual(prediction.downside_probability or 0.0, 0.20)
            self.assertAlmostEqual(prediction.decision_score or 0.0, 0.608)
            self.assertEqual(prediction.catalyst.status, "confirmed")
            self.assertEqual(prediction.signal, "entry_candidate_confirmed")

    def test_live_data_source_uses_registered_feature_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)
            store = LiveFeatureStore(root)
            generated = datetime(2025, 9, 17, 20, 30, tzinfo=UTC)
            store.publish("swing", frame, price_feed="sip", generated_at=generated)

            response = _service(
                root,
                swing=(None, model),
                data_source="live",
                live_feature_store=store,
            ).predict_swing(
                PredictionRequest(
                    tickers=["MSFT"],
                    mode="swing",
                    as_of=generated,
                )
            )

            self.assertEqual(response.data_source, "live")
            prediction = response.predictions[0].swing
            assert prediction is not None
            self.assertAlmostEqual(prediction.probability or 0.0, 0.73)

    def test_rejects_model_when_artifact_no_longer_matches_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(
                model,
                features,
                target_col="target_net_positive_5d",
                status="promoted",
                probability=0.73,
            )
            with model.open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(ValueError, "artifact integrity check failed"):
                _service(root, swing=(dataset, model)).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing"))

    def test_warning_readiness_is_never_returned_as_actionable(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            frame["price_feed"] = "unknown"
            frame.to_parquet(dataset, index=False)
            _write_model(
                model,
                features,
                target_col="target_net_positive_5d",
                status="promoted",
                probability=0.73,
            )

            response = _service(root, swing=(dataset, model)).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing"))

            prediction = response.predictions[0].swing
            assert prediction is not None
            self.assertEqual(prediction.readiness.status, "warn")
            self.assertEqual(prediction.signal, "not_ready")
            self.assertIsNone(prediction.decision_score)
            self.assertIsNone(prediction.model_prediction)
            self.assertEqual(response.predictions[0].final_signal, "not_ready")

    def test_health_checks_registered_model_and_live_feature_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            _write_model(
                model,
                features,
                target_col="target_net_positive_5d",
                status="promoted",
                probability=0.73,
            )
            generated = datetime(2025, 9, 17, 20, 30, tzinfo=UTC)
            store = LiveFeatureStore(root)
            store.publish("swing", frame, price_feed="sip", generated_at=generated)
            service = _service(
                root,
                swing=(None, model),
                data_source="live",
                live_feature_store=store,
            )

            result = service.health(as_of=generated)

            self.assertEqual(result["status"], "ready")
            self.assertEqual(result["data_source"], "live")


def _service(
    root: Path,
    *,
    swing: tuple[Path | None, Path] | None = None,
    swing_horizon: str = "5d",
    intraday: tuple[Path | None, Path] | None = None,
    data_source: PredictionDataSource = "curated",
    live_feature_store: LiveFeatureStore | None = None,
) -> PredictionService:
    routes: dict[str, dict[str, ServingRoute]] = {}
    if swing is not None:
        dataset, model = swing
        routes["swing"] = {
            swing_horizon: ServingRoute(
                model=model,
                curated_dataset=dataset,
                bar_timeframe="1Day",
            )
        }
    if intraday is not None:
        dataset, model = intraday
        routes["intraday"] = {
            "60m": ServingRoute(
                model=model,
                curated_dataset=dataset,
                bar_timeframe="5Min",
            )
        }
    return PredictionService(
        root,
        routes=routes,
        data_source=data_source,
        live_feature_store=live_feature_store,
    )


def _write_model(
    path: Path,
    features: list[str],
    *,
    target_col: str,
    status: str,
    probability: float,
) -> None:
    is_swing = target_col.startswith("target_net_positive_")
    model_type = SWING_MODEL_TYPE if is_swing else INTRADAY_MODEL_TYPE
    schema_version = SWING_MODEL_SCHEMA_VERSION if is_swing else INTRADAY_MODEL_SCHEMA_VERSION
    payload: dict[str, object] = {
        "features": features,
        "model_type": model_type,
    }
    if is_swing:
        payload.update(
            {
                "model": FixedProbabilityModel(probability),
                "target_col": target_col,
                "calibrator": None,
                "horizon_sessions": 5,
                "feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
            }
        )
    else:
        downside_target = "stop_before_target_60m"
        payload.update(
            {
                "models": {
                    target_col: FixedProbabilityModel(probability),
                    downside_target: FixedProbabilityModel(0.20),
                },
                "calibrators": {target_col: None, downside_target: None},
                "opportunity_target_col": target_col,
                "downside_target_col": downside_target,
                "horizon_minutes": 60,
                "feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
            }
        )
    joblib.dump(payload, path)
    training = _swing_frame(["MSFT", "AAPL"], features, rows=300)
    training["target"] = [idx % 2 for idx in range(len(training))]
    write_model_manifest(
        model_path=path,
        model_type=model_type,
        schema_version=schema_version,
        target_col=target_col,
        features=features,
        training_data=training.assign(**{target_col: training["target"]}),
        metrics={
            "roc_auc": 0.7,
            "top_decile_lift": 2.1,
            "validated_rows": 250,
            "tickers": 2,
        },
        validation_split=(
            "session_purged_walk_forward_and_ticker_holdout" if is_swing else "session_purged_walk_forward_and_ticker_holdout"
        ),
        status=status,
    )


def _swing_frame(tickers: list[str], features: list[str], *, rows: int) -> pd.DataFrame:
    start = date(2025, 1, 1)
    records = []
    for ticker in tickers:
        for idx in range(rows):
            session_date = start + timedelta(days=idx)
            feature_available = (
                pd.Timestamp(session_date).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
            ).tz_convert("UTC")
            record = {
                "ticker": ticker,
                "date": session_date,
                "session_date_et": session_date,
                "feature_available_at_utc": feature_available,
                "swing_feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
                "close": 100.0 + idx,
                "price_feed": "sip",
                "return_1d": 0.01,
                "volume_z20": 1.5,
                "news_count": 2,
                "event_count": 3,
                "sentiment_mean": 0.2,
                "sector_return_1d": 0.005,
                "global_net_impact": 0.0,
                "global_event_count_1d": 1.0,
                "event_count_3d": 3.0,
                "sentiment_mean_3d": 0.2,
                "event_relevance_mean_3d": 0.8,
                "source_count_alpaca_3d": 1.0,
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
            "bar_start_utc": timestamps,
            "feature_available_at_utc": timestamps + pd.Timedelta(minutes=5),
            "intraday_feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
            "five_minute_bar_count": np.arange(1, rows + 1),
            "open": 100.0,
            "high": 101.0,
            "low": 99.5,
            "close": 100.5,
            "volume": 100_000.0,
            "price_feed": "sip",
            "return_1d": 0.01,
            "volume_z20": 1.5,
            "qqq_return_1bar_5m": 0.001,
            "global_event_count_2h": 1.0,
        }
    )


if __name__ == "__main__":
    unittest.main()
