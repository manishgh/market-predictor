from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
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
from market_predictor.live_features import live_feature_columns
from market_predictor.prediction_contracts import (
    PredictionCapacityError,
    PredictionDataSource,
    PredictionReadinessError,
    PredictionRequest,
    PredictionValidationError,
)
from market_predictor.prediction_service import (
    SERVING_POLICY_ID,
    SERVING_POLICY_SHA256,
    PredictionService,
    ServingRoute,
    serving_routes_from_config,
)
from market_predictor.registry import write_model_manifest
from market_predictor.serving_context import (
    ActiveModelContext,
    ActiveReleaseRoute,
    verify_serving_model_artifact,
)
from market_predictor.swing.contracts import (
    SWING_FEATURE_SCHEMA_VERSION,
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from tests.r4_fixtures import authorize_candidate_for_test, synthetic_identity_metrics


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


class StaticModelContextProvider:
    """Test-only provider; production always resolves an active release pointer."""

    def __init__(self, root: Path) -> None:
        self.root = root
        self.contexts: dict[tuple[str, str, Path], ActiveModelContext] = {}
        self.load_count = 0

    def get(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> ActiveModelContext:
        model_path = route.repository if route.repository.is_absolute() else self.root / route.repository
        key = (mode, horizon, model_path)
        cached = self.contexts.get(key)
        if cached is not None:
            return cached
        expected_type = SWING_MODEL_TYPE if mode == "swing" else INTRADAY_MODEL_TYPE
        expected_schema = SWING_MODEL_SCHEMA_VERSION if mode == "swing" else INTRADAY_MODEL_SCHEMA_VERSION
        manifest = verify_serving_model_artifact(
            model_path,
            resolved_horizon=horizon,
            expected_model_type=expected_type,
            expected_schema_version=expected_schema,
        )
        payload = joblib.load(model_path)
        self.load_count += 1
        context = ActiveModelContext(
            mode=mode,
            horizon=horizon,
            release_id="e" * 64,
            pointer_sha256="d" * 64,
            model_path=model_path,
            manifest=manifest,
            payload=payload,
        )
        self.contexts[key] = context
        return context

    def snapshot(self) -> dict[str, object]:
        return {"loaded_contexts": len(self.contexts), "contexts": []}

    def cached(self, mode: str, horizon: str) -> ActiveModelContext | None:
        for (cached_mode, cached_horizon, _), context in self.contexts.items():
            if cached_mode == mode and cached_horizon == horizon:
                return context
        return None

    def is_current(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> bool:
        del route
        return self.cached(mode, horizon) is not None


class BlockingModelContextProvider(StaticModelContextProvider):
    def __init__(self, root: Path) -> None:
        super().__init__(root)
        self.entered = threading.Event()
        self.release = threading.Event()

    def get(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> ActiveModelContext:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release blocked model context")
        return super().get(mode, horizon, route)


class PredictionServiceTests(unittest.TestCase):
    def test_serving_routes_are_loaded_from_server_configuration(self) -> None:
        routes = serving_routes_from_config(
            {
                "prediction_serving": {
                    "attestation_trust_store": "configs/trust.json",
                    "routes": {
                        "swing": {
                            "5d": {
                                "release_repository": "data/releases/swing_5d",
                                "bar_timeframe": "1Day",
                            }
                        }
                    }
                }
            }
        )

        self.assertEqual(
            routes["swing"]["5d"].repository,
            Path("data/releases/swing_5d"),
        )

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
            self.assertEqual(prediction.signal, "strong_bullish_watch")
            self.assertAlmostEqual(prediction.decision_score or 0.0, 0.73)
            self.assertEqual(prediction.readiness.status, "valid")
            self.assertEqual(response.horizon, "5d")
            self.assertIsNotNone(response.evidence)
            assert response.evidence is not None
            self.assertEqual(response.evidence.serving_policy_id, SERVING_POLICY_ID)
            self.assertEqual(response.evidence.serving_policy_sha256, SERVING_POLICY_SHA256)
            self.assertEqual(response.evidence.identity_status, "research_only")

    def test_prediction_rejects_oversized_ticker_batch_before_scoring(self) -> None:
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
            service = _service(
                root,
                swing=(dataset, model),
                max_tickers_per_request=1,
            )

            with self.assertRaises(PredictionValidationError):
                service.predict(
                    PredictionRequest(tickers=["MSFT", "AAPL"], mode="swing")
                )

    def test_concurrent_request_is_rejected_instead_of_queued(self) -> None:
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
            provider = BlockingModelContextProvider(root)
            service = _service(
                root,
                swing=(dataset, model),
                model_context_cache=provider,
                max_concurrent_inference=1,
            )
            request = PredictionRequest(tickers=["MSFT"], mode="swing")

            with ThreadPoolExecutor(max_workers=1) as executor:
                first = executor.submit(service.predict, request)
                self.assertTrue(provider.entered.wait(timeout=2))
                with self.assertRaises(PredictionCapacityError):
                    service.predict(request)
                provider.release.set()
                self.assertEqual(first.result(timeout=5).mode, "swing")

    def test_swing_prediction_rejects_candidate_model_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_net_positive_5d", status="candidate", probability=0.73)

            with self.assertRaises(PredictionReadinessError):
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

            with self.assertRaises(PredictionReadinessError):
                _service(root, swing=(dataset, model), swing_horizon="1d").predict_swing(
                    PredictionRequest(
                        tickers=["MSFT"],
                        mode="swing",
                        horizon="1d",
                    )
                )

    def test_intraday_60m_wire_horizon_accepts_60m_1h_and_auto(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "intraday.parquet"
            model = root / "intraday.joblib"
            features = ["return_1d", "volume_z20"]
            _intraday_frame("MSFT", rows=150).to_parquet(dataset, index=False)
            _write_model(
                model,
                features,
                target_col="target_before_stop_60m",
                status="promoted",
                probability=0.72,
            )
            service = _service(root, intraday=(dataset, model))

            for requested in ("60m", "1h", "auto"):
                with self.subTest(requested=requested):
                    response = service.predict_intraday(
                        PredictionRequest(tickers=["MSFT"], mode="intraday", horizon=requested)
                    )
                    self.assertEqual(response.horizon, "60m")
                    self.assertEqual(response.resolved_horizons, {"intraday": "60m"})
                    self.assertIsNotNone(response.evidence)
                    assert response.evidence is not None
                    self.assertEqual(response.evidence.resolved_horizons, {"intraday": "60m"})

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

    def test_intraday_catalyst_is_metadata_only_and_exact_policy_is_served(self) -> None:
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
            self.assertAlmostEqual(prediction.decision_score or 0.0, 0.72 * (1.0 - 0.20))
            self.assertEqual(prediction.catalyst.status, "confirmed")
            self.assertEqual(prediction.signal, "entry_candidate")

    def test_live_data_source_uses_registered_feature_store(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)
            store = LiveFeatureStore(root)
            generated = datetime(2025, 9, 17, 22, 5, tzinfo=UTC)
            _publish_live_swing(store, frame, generated)

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
            self.assertEqual(prediction.readiness.daily_bar_count, 260)
            self.assertEqual(prediction.readiness.status, "valid")
            self.assertEqual(prediction.signal, "strong_bullish_watch")
            self.assertIsNotNone(response.evidence)
            assert response.evidence is not None
            self.assertEqual(response.evidence.identity_status, "complete")
            self.assertIn("swing", response.evidence.feature_artifacts)
            self.assertEqual(response.evidence.model_artifact_sha256["swing"], response.models["swing"].artifact_sha256)
            expected_cutoff = pd.to_datetime(frame["decision_time_utc"], utc=True).max().to_pydatetime()
            self.assertEqual(response.evidence.prediction_cutoff_utc, expected_cutoff)
            self.assertEqual(response.evidence.view_prediction_cutoffs_utc, {"swing": expected_cutoff})
            self.assertIn("ticker:alpaca", response.evidence.source_watermarks["swing"])

    def test_live_swing_fails_closed_for_under_warm_or_missing_audited_count(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            daily_bar_count = 249
            with self.subTest(daily_bar_count=daily_bar_count):
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
                store = LiveFeatureStore(root)
                generated = datetime(2025, 9, 17, 22, 5, tzinfo=UTC)
                _publish_live_swing(store, frame, generated, daily_bar_count=daily_bar_count)

                response = _service(
                    root,
                    swing=(None, model),
                    data_source="live",
                    live_feature_store=store,
                ).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing", as_of=generated))

                prediction = response.predictions[0].swing
                assert prediction is not None
                self.assertEqual(prediction.readiness.status, "warn")
                self.assertEqual(prediction.signal, "not_ready")
                self.assertIsNone(prediction.rank)
                self.assertIsNone(prediction.decision_score)

        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            store = LiveFeatureStore(root)
            frame = _swing_frame(["MSFT"], ["return_1d", "volume_z20"], rows=260)
            with self.assertRaisesRegex(ValueError, "daily_bar_count"):
                _publish_live_swing(
                    store,
                    frame,
                    datetime(2025, 9, 17, 22, 5, tzinfo=UTC),
                    daily_bar_count=None,
                )

    def test_negative_catalyst_does_not_modify_swing_score_signal_or_rank(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            frame = _swing_frame(["MSFT"], features, rows=260)
            frame["sentiment_mean_3d"] = -0.60
            frame["event_relevance_mean_3d"] = 1.5
            frame["event_offering_count_1d"] = 1.0
            frame.to_parquet(dataset, index=False)
            _write_model(model, features, target_col="target_net_positive_5d", status="promoted", probability=0.73)

            response = _service(root, swing=(dataset, model)).predict_swing(
                PredictionRequest(tickers=["MSFT"], mode="swing")
            )

            prediction = response.predictions[0].swing
            assert prediction is not None
            self.assertEqual(prediction.catalyst.status, "veto")
            self.assertAlmostEqual(prediction.decision_score or 0.0, 0.73)
            self.assertEqual(prediction.signal, "strong_bullish_watch")
            self.assertEqual(prediction.rank, 1)

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

            with self.assertRaises(PredictionReadinessError):
                _service(root, swing=(dataset, model)).predict_swing(PredictionRequest(tickers=["MSFT"], mode="swing"))

    def test_tampered_promotion_attestation_fails_as_not_ready(self) -> None:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            dataset = root / "features.parquet"
            model = root / "swing.joblib"
            features = ["return_1d", "volume_z20"]
            _swing_frame(["MSFT"], features, rows=260).to_parquet(
                dataset,
                index=False,
            )
            _write_model(
                model,
                features,
                target_col="target_net_positive_5d",
                status="promoted",
                probability=0.73,
            )
            attestation = model.with_suffix(
                model.suffix + ".promotion.attestation.json"
            )
            payload = json.loads(attestation.read_text(encoding="utf-8"))
            payload["build_identity"] = "tampered"
            attestation.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaises(PredictionReadinessError):
                _service(root, swing=(dataset, model)).predict_swing(
                    PredictionRequest(tickers=["MSFT"], mode="swing")
                )

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
            generated = datetime(2025, 9, 17, 22, 5, tzinfo=UTC)
            store = LiveFeatureStore(root)
            _publish_live_swing(store, frame, generated)
            service = _service(
                root,
                swing=(None, model),
                data_source="live",
                live_feature_store=store,
            )
            service.preload()

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
    model_context_cache: StaticModelContextProvider | None = None,
    max_concurrent_inference: int = 1,
    max_tickers_per_request: int = 100,
) -> PredictionService:
    routes: dict[str, dict[str, ServingRoute]] = {}
    if swing is not None:
        dataset, model = swing
        routes["swing"] = {
            swing_horizon: ServingRoute(
                repository=model,
                attestation_trust_store=Path("unused-test-trust.json"),
                curated_dataset=dataset,
                bar_timeframe="1Day",
            )
        }
    if intraday is not None:
        dataset, model = intraday
        routes["intraday"] = {
            "60m": ServingRoute(
                repository=model,
                attestation_trust_store=Path("unused-test-trust.json"),
                curated_dataset=dataset,
                bar_timeframe="5Min",
            )
        }
    return PredictionService(
        root,
        routes=routes,
        data_source=data_source,
        live_feature_store=live_feature_store,
        model_context_cache=model_context_cache or StaticModelContextProvider(root),
        max_concurrent_inference=max_concurrent_inference,
        max_tickers_per_request=max_tickers_per_request,
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
    model_run_id = f"prediction-service-{path.stem}"
    metrics = {
        **synthetic_identity_metrics(model_type=model_type, model_run_id=model_run_id),
        "roc_auc": 0.7,
        "top_decile_lift": 2.1,
        "validated_rows": 250,
        "tickers": 2,
    }
    write_model_manifest(
        model_path=path,
        model_type=model_type,
        schema_version=schema_version,
        target_col=target_col,
        features=features,
        training_data=training.assign(**{target_col: training["target"]}),
        metrics=metrics,
        validation_split=(
            "session_purged_walk_forward_and_ticker_holdout" if is_swing else "session_purged_walk_forward_and_ticker_holdout"
        ),
        extra={"model_run_id": model_run_id},
    )
    if status == "promoted":
        authorize_candidate_for_test(path, metrics)
    elif status != "candidate":
        raise ValueError(f"unsupported test model status: {status}")


def _publish_live_swing(
    store: LiveFeatureStore,
    frame: pd.DataFrame,
    generated: datetime,
    *,
    daily_bar_count: int | None = 260,
) -> dict[str, object]:
    complete = frame.copy()
    latest_decision = pd.to_datetime(complete["decision_time_utc"], utc=True).max()
    complete = complete[pd.to_datetime(complete["decision_time_utc"], utc=True).eq(latest_decision)].copy()
    missing = {
        column: pd.Series(0.0, index=complete.index)
        for column in live_feature_columns("swing")
        if column not in complete
    }
    complete = pd.concat([complete, pd.DataFrame(missing)], axis=1)
    if daily_bar_count is None:
        complete = complete.drop(columns=["daily_bar_count"], errors="ignore")
    else:
        complete["daily_bar_count"] = daily_bar_count
    return store.publish(
        "swing",
        complete,
        price_feed="sip",
        feature_schema_version=SWING_FEATURE_SCHEMA_VERSION,
        source_artifact_sha256="a" * 64,
        source_artifact_type="swing_inference_features",
        generated_at=generated,
    )


def _swing_frame(tickers: list[str], features: list[str], *, rows: int) -> pd.DataFrame:
    start = date(2025, 1, 1)
    records = []
    for ticker in tickers:
        for idx in range(rows):
            session_date = start + timedelta(days=idx)
            bar_available = (
                pd.Timestamp(session_date).tz_localize("America/New_York") + pd.Timedelta(hours=16, minutes=15)
            ).tz_convert("UTC")
            decision_time = (
                pd.Timestamp(session_date).tz_localize("America/New_York") + pd.Timedelta(hours=18)
            ).tz_convert("UTC")
            feature_available = decision_time - pd.Timedelta(minutes=10)
            record = {
                "ticker": ticker,
                "date": session_date,
                "session_date_et": session_date,
                "decision_time_utc": decision_time,
                "feature_available_at_utc": feature_available,
                "bar_available_at_utc": bar_available,
                "prediction_cutoff_policy_id": "xnys_1800_america_new_york_v1",
                "source_coverage_end_utc_alpaca": decision_time - pd.Timedelta(minutes=15),
                "swing_feature_schema_version": SWING_FEATURE_SCHEMA_VERSION,
                "daily_bar_count": idx + 1,
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
            "decision_time_utc": timestamps + pd.Timedelta(minutes=5),
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
