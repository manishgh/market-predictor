from __future__ import annotations

import json
import tempfile
import threading
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from market_predictor.drift_policy import (
    DriftAssessmentV2,
    DriftPolicyV2,
    DriftStateStore,
)
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.contracts import (
    INTRADAY_FEATURE_SCHEMA_VERSION,
    INTRADAY_MODEL_SCHEMA_VERSION,
    INTRADAY_MODEL_TYPE,
    IntradayDatasetConfig,
)
from market_predictor.live_features import live_feature_columns
from market_predictor.outcome_contracts import content_sha256
from market_predictor.prediction_contracts import (
    PredictionCapacityError,
    PredictionDataSource,
    PredictionDriftBlockedError,
    PredictionModelUnavailableError,
    PredictionReadinessError,
    PredictionRequest,
    PredictionValidationError,
)
from market_predictor.prediction_policy import (
    PredictionSelectionPolicy,
    prediction_policy_identity,
)
from market_predictor.prediction_service import (
    PredictionService,
    ServingRoute,
    serving_routes_from_config,
)
from market_predictor.registry import load_model_manifest, write_model_manifest
from market_predictor.serving_context import (
    ActiveModelContext,
    ActiveReleaseRoute,
    verify_serving_model_artifact,
)
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
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


class IdentityCalibrator:
    def predict(self, probability: np.ndarray) -> np.ndarray:
        return probability


class StaticModelContextProvider:
    """Test-only intraday provider; production resolves signed release pointers."""

    def __init__(
        self,
        root: Path,
        live_feature_store: LiveFeatureStore | None = None,
    ) -> None:
        self.root = root
        self.live_feature_store = live_feature_store
        self.contexts: dict[tuple[str, str, Path], ActiveModelContext] = {}

    def get(
        self,
        mode: str,
        horizon: str,
        route: ActiveReleaseRoute,
    ) -> ActiveModelContext:
        if mode != "intraday":
            raise PredictionModelUnavailableError
        model_path = (
            route.repository
            if route.repository.is_absolute()
            else self.root / route.repository
        )
        key = (mode, horizon, model_path)
        cached = self.contexts.get(key)
        if cached is not None:
            return cached
        manifest = verify_serving_model_artifact(
            model_path,
            resolved_horizon=horizon,
            expected_model_type=INTRADAY_MODEL_TYPE,
            expected_schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
        )
        payload = joblib.load(model_path)
        feature_frame: pd.DataFrame | None = None
        feature_manifest: dict[str, object] = {}
        serving_bundle_id: str | None = None
        if self.live_feature_store is not None:
            _, manifest_path = self.live_feature_store.paths("intraday")
            loaded = json.loads(manifest_path.read_text(encoding="utf-8"))
            if not isinstance(loaded, dict):
                raise ValueError("test live feature manifest is invalid")
            feature_manifest = {str(name): value for name, value in loaded.items()}
            generated = datetime.fromisoformat(
                str(feature_manifest["generated_at_utc"])
            )
            feature_frame = self.live_feature_store.load(
                "intraday",
                as_of=generated,
            )
            serving_bundle_id = "f" * 64
        context = ActiveModelContext(
            mode=mode,
            horizon=horizon,
            release_id="e" * 64,
            pointer_sha256="d" * 64,
            model_path=model_path,
            manifest=manifest,
            payload=payload,
            serving_bundle_id=serving_bundle_id,
            serving_bundle_generated_at_utc=(
                str(feature_manifest.get("generated_at_utc"))
                if feature_manifest
                else None
            ),
            feature_frame=feature_frame,
            feature_manifest=feature_manifest,
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
    def test_serving_routes_load_signed_ten_session_swing_configuration(self) -> None:
        routes = serving_routes_from_config(
            {
                "prediction_serving": {
                    "attestation_trust_store": "configs/trust.json",
                    "promotion_gate_policy_sha256": "a" * 64,
                    "routes": {
                        "swing": {
                            "10b": {
                                "release_repository": "models/edge_rebuild/swing/promoted",
                                "bar_timeframe": "1Day",
                            }
                        }
                    },
                }
            }
        )

        route = routes["swing"]["10b"]
        self.assertEqual(route.repository, Path("models/edge_rebuild/swing/promoted"))
        self.assertEqual(route.promotion_gate_policy_sha256, "a" * 64)

    def test_serving_routes_reject_retired_five_day_swing(self) -> None:
        with self.assertRaisesRegex(ValueError, "ten-session"):
            serving_routes_from_config(
                {
                    "prediction_serving": {
                        "attestation_trust_store": "configs/trust.json",
                        "promotion_gate_policy_sha256": "a" * 64,
                        "routes": {
                            "swing": {
                                "5d": {
                                    "release_repository": "data/releases/retired"
                                }
                            }
                        },
                    }
                }
            )

    def test_missing_signed_swing_generation_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            service = PredictionService(
                root,
                routes={
                    "swing": {
                        "10b": ServingRoute(
                            repository=root / "missing-generation",
                            attestation_trust_store=root / "trust.json",
                            promotion_gate_policy_sha256="a" * 64,
                        )
                    }
                },
            )

            with self.assertRaises(PredictionModelUnavailableError):
                service.predict_swing(
                    PredictionRequest(tickers=["MSFT"], mode="swing")
                )

    def test_oversized_batch_is_rejected_before_model_loading(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            provider = StaticModelContextProvider(root)
            service = _intraday_service(
                root,
                dataset=root / "missing.parquet",
                model=root / "missing.joblib",
                provider=provider,
                max_tickers_per_request=1,
            )

            with self.assertRaises(PredictionValidationError):
                service.predict(
                    PredictionRequest(tickers=["MSFT", "AAPL"], mode="intraday")
                )
            self.assertEqual(provider.snapshot()["loaded_contexts"], 0)

    def test_concurrent_request_is_rejected_instead_of_queued(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            provider = BlockingModelContextProvider(root)
            service = _intraday_service(
                root,
                dataset=dataset,
                model=model,
                provider=provider,
            )
            request = PredictionRequest(tickers=["MSFT"], mode="intraday")
            with ThreadPoolExecutor(max_workers=1) as pool:
                pending = pool.submit(service.predict, request)
                self.assertTrue(provider.entered.wait(timeout=5))
                with self.assertRaises(PredictionCapacityError):
                    service.predict(request)
                provider.release.set()
                response = pending.result(timeout=5)
            self.assertEqual(response.resolved_horizons, {"intraday": "60m"})

    def test_intraday_wire_horizon_accepts_aliases(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            service = _intraday_service(root, dataset=dataset, model=model)

            for requested in ("60m", "1h", "auto"):
                with self.subTest(requested=requested):
                    response = service.predict_intraday(
                        PredictionRequest(
                            tickers=["MSFT"],
                            mode="intraday",
                            horizon=requested,
                        )
                    )
                    self.assertEqual(response.horizon, "60m")
                    self.assertEqual(
                        response.resolved_horizons,
                        {"intraday": "60m"},
                    )

    def test_intraday_as_of_excludes_unclosed_bar(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = _intraday_frame("MSFT", rows=150)
            dataset = root / "intraday.parquet"
            model = root / "intraday.joblib"
            frame.to_parquet(dataset, index=False)
            _write_intraday_model(model)
            cutoff = pd.Timestamp(frame["date"].iloc[-1], tz="UTC") + pd.Timedelta(
                minutes=2
            )

            response = _intraday_service(
                root,
                dataset=dataset,
                model=model,
            ).predict_intraday(
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
            self.assertGreaterEqual(prediction.readiness.intraday_bar_count, 130)

    def test_top_level_predict_persists_immutable_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            service = _intraday_service(root, dataset=dataset, model=model)

            response = service.predict(
                PredictionRequest(tickers=["MSFT"], mode="intraday")
            )

            self.assertIsNotNone(response.snapshot_id)
            self.assertEqual(response.snapshot_id, response.snapshot_sha256)
            assert response.snapshot_id is not None
            self.assertTrue(service.snapshot_store.path_for(response.snapshot_id).exists())

    def test_intraday_catalyst_is_confirmation_metadata_only(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            frame = _intraday_frame("MSFT", rows=150)
            frame["news_count_2h"] = 2
            frame["sentiment_mean_2h"] = 0.40
            frame["event_relevance_mean_2h"] = 1.2
            frame["source_count_alpaca_2h"] = 1
            frame["source_count_sec_2h"] = 1
            frame["event_contract_count_2h"] = 1
            dataset = root / "intraday.parquet"
            model = root / "intraday.joblib"
            frame.to_parquet(dataset, index=False)
            _write_intraday_model(model)

            response = _intraday_service(
                root,
                dataset=dataset,
                model=model,
            ).predict_intraday(
                PredictionRequest(tickers=["MSFT"], mode="intraday")
            )

            prediction = response.predictions[0].intraday
            assert prediction is not None
            self.assertAlmostEqual(prediction.opportunity_probability or 0.0, 0.72)
            self.assertAlmostEqual(prediction.downside_probability or 0.0, 0.20)
            self.assertEqual(prediction.catalyst.status, "confirmed")

    def test_unified_response_keeps_intraday_but_abstains_without_swing(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            response = _intraday_service(
                root,
                dataset=dataset,
                model=model,
            ).predict_unified(
                PredictionRequest(tickers=["MSFT"], mode="unified")
            )

            self.assertIn("intraday", response.models)
            self.assertNotIn("swing", response.models)
            self.assertTrue(response.errors)
            self.assertEqual(response.predictions[0].final_signal, "not_ready")

    def test_model_artifact_mutation_is_rejected(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            model.write_bytes(model.read_bytes() + b"tampered")

            with self.assertRaises(PredictionReadinessError):
                _intraday_service(
                    root,
                    dataset=dataset,
                    model=model,
                ).predict_intraday(
                    PredictionRequest(tickers=["MSFT"], mode="intraday")
                )

    def test_direct_prediction_rejects_severe_drift(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            dataset, model = _intraday_inputs(root)
            store = DriftStateStore(root / "drift")
            evaluated = datetime.now(UTC)
            store.publish(
                _drift_assessment(
                    "intraday",
                    "60m",
                    "severe",
                    evaluated,
                    model,
                )
            )
            service = _intraday_service(
                root,
                dataset=dataset,
                model=model,
                drift_state_store=store,
                enforce_drift=True,
            )

            with self.assertRaises(PredictionDriftBlockedError):
                service.predict_intraday(
                    PredictionRequest(tickers=["MSFT"], mode="intraday")
                )


def _intraday_service(
    root: Path,
    *,
    dataset: Path | None,
    model: Path,
    provider: StaticModelContextProvider | None = None,
    data_source: PredictionDataSource = "curated",
    live_feature_store: LiveFeatureStore | None = None,
    max_tickers_per_request: int = 100,
    drift_state_store: DriftStateStore | None = None,
    enforce_drift: bool = False,
) -> PredictionService:
    return PredictionService(
        root,
        routes={
            "intraday": {
                "60m": ServingRoute(
                    repository=model,
                    attestation_trust_store=Path("unused-test-trust.json"),
                    curated_dataset=dataset,
                    bar_timeframe="5Min",
                )
            }
        },
        data_source=data_source,
        live_feature_store=live_feature_store,
        model_context_cache=provider
        or StaticModelContextProvider(
            root,
            live_feature_store if data_source == "live" else None,
        ),
        max_tickers_per_request=max_tickers_per_request,
        drift_state_store=drift_state_store,
        enforce_drift=enforce_drift,
    )


def _intraday_inputs(root: Path) -> tuple[Path, Path]:
    dataset = root / "intraday.parquet"
    model = root / "intraday.joblib"
    _intraday_frame("MSFT", rows=150).to_parquet(dataset, index=False)
    _write_intraday_model(model)
    return dataset, model


def _write_intraday_model(path: Path) -> None:
    features = ["return_1d", "volume_z20"]
    opportunity_target = "target_before_stop_60m"
    downside_target = "stop_before_target_60m"
    policy = PredictionSelectionPolicy()
    payload: dict[str, object] = {
        "model_type": INTRADAY_MODEL_TYPE,
        "model_schema_version": INTRADAY_MODEL_SCHEMA_VERSION,
        "feature_schema_version": INTRADAY_FEATURE_SCHEMA_VERSION,
        "features": features,
        "models": {
            opportunity_target: FixedProbabilityModel(0.72),
            downside_target: FixedProbabilityModel(0.20),
        },
        "calibrators": {
            opportunity_target: IdentityCalibrator(),
            downside_target: IdentityCalibrator(),
        },
        "opportunity_target_col": opportunity_target,
        "downside_target_col": downside_target,
        "horizon_minutes": 60,
        "calibration_method": "isotonic_prior_fold_only",
        "prediction_policy": policy.specification(),
        "prediction_policy_sha256": policy.sha256(),
    }
    joblib.dump(payload, path)
    training = _intraday_frame("MSFT", rows=150)
    training[opportunity_target] = np.arange(len(training)) % 2
    model_run_id = f"prediction-service-{path.stem}"
    label_config = IntradayDatasetConfig()
    metrics = {
        **synthetic_identity_metrics(
            model_type=INTRADAY_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        **prediction_policy_identity(policy),
        "dataset_label_config_sha256": label_config.label_config_sha256(),
        "roc_auc": 0.7,
        "top_decile_lift": 2.1,
        "validated_rows": len(training),
        "tickers": 1,
    }
    write_model_manifest(
        model_path=path,
        model_type=INTRADAY_MODEL_TYPE,
        schema_version=INTRADAY_MODEL_SCHEMA_VERSION,
        target_col=opportunity_target,
        features=features,
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={
            "model_run_id": model_run_id,
            "label_policy": label_config.label_policy(),
            "prediction_policy": policy.specification(),
        },
    )
    authorize_candidate_for_test(path, metrics)


def _publish_live_intraday(
    store: LiveFeatureStore,
    frame: pd.DataFrame,
    generated_at: datetime,
) -> dict[str, object]:
    latest = pd.to_datetime(frame["decision_time_utc"], utc=True).max()
    complete = frame.loc[
        pd.to_datetime(frame["decision_time_utc"], utc=True).eq(latest)
    ].copy()
    missing = {
        column: pd.Series(0.0, index=complete.index)
        for column in live_feature_columns("intraday")
        if column not in complete
    }
    complete = pd.concat([complete, pd.DataFrame(missing)], axis=1)
    return store.publish(
        "intraday",
        complete,
        price_feed="sip",
        feature_schema_version=INTRADAY_FEATURE_SCHEMA_VERSION,
        source_artifact_sha256="a" * 64,
        source_artifact_type="intraday_inference_features",
        source_watermarks={"ticker:alpaca": latest.isoformat()},
        generated_at=generated_at,
    )


def _intraday_frame(ticker: str, *, rows: int) -> pd.DataFrame:
    timestamps = pd.date_range("2026-07-08T13:30:00Z", periods=rows, freq="5min")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "canonical_security_id": f"security:{ticker}",
            "date": timestamps.tz_convert(None),
            "session_date_et": timestamps.tz_convert("America/New_York").date,
            "decision_group_id": (timestamps + pd.Timedelta(minutes=5)).astype(str),
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
            "primary_benchmark": "XLK",
            "market_regime": "risk_on",
            "sector": "Technology",
            "market_cap_bucket": "large",
            "liquidity_bucket": "high",
            "atr_14_price_5m": 1.25,
            "return_1d": 0.01,
            "volume_z20": 1.5,
            "qqq_return_1bar_5m": 0.001,
            "global_event_count_2h": 1.0,
        }
    )


def _drift_assessment(
    mode: str,
    horizon: str,
    state: str,
    evaluated_at: datetime,
    model_path: Path,
) -> DriftAssessmentV2:
    manifest = load_model_manifest(model_path)
    metrics = manifest["metrics"]
    if not isinstance(metrics, dict):
        raise AssertionError("test model metrics are unavailable")
    content = {
        "contract_version": "market_predictor.drift_assessment.v2",
        "mode": mode,
        "horizon": horizon,
        "model_release_id": "e" * 64,
        "model_artifact_sha256": manifest["artifact_sha256"],
        "prediction_policy_sha256": metrics["prediction_policy_sha256"],
        "label_policy_sha256": metrics["dataset_label_config_sha256"],
        "execution_policy_sha256": metrics["execution_policy_sha256"],
        "policy_sha256": DriftPolicyV2().sha256(),
        "performance_report_id": "1" * 64,
        "performance_cohort_id": "2" * 64,
        "feature_artifact_set_sha256": "3" * 64,
        "evaluated_at_utc": evaluated_at.isoformat().replace("+00:00", "Z"),
        "state": state,
        "actionability": "not_ready",
        "reasons": ("selected_policy_performance_severe",),
        "feature_drift_status": "stable",
        "total_predictions": 50,
        "selected_predictions": 10,
        "matured_samples": 10,
        "independent_decision_groups": 10,
        "last_matured_outcome_utc": evaluated_at.isoformat().replace(
            "+00:00",
            "Z",
        ),
    }
    return DriftAssessmentV2.model_validate(
        {**content, "assessment_id": content_sha256(content)}
    )


if __name__ == "__main__":
    unittest.main()
