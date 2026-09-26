from __future__ import annotations

import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import timedelta
from pathlib import Path

import pytest
from pydantic import ValidationError

from market_predictor.core.prediction_contracts import (
    InvestmentReplayRequest,
    PredictionCapacityError,
    PredictionDriftBlockedError,
    PredictionEvidenceV4,
    PredictionModelUnavailableError,
    PredictionRequest,
    PredictionResponse,
    PredictionValidationError,
    ReadinessInfo,
)
from market_predictor.governance.drift.policy import DriftStateStore
from market_predictor.serving.prediction_service import (
    PredictionService,
    serving_routes_from_config,
)
from market_predictor.serving.routes import ServingRoute
from market_predictor.serving.swing_inference import LoadedSwingModelGeneration
from tests.support.swing_serving import (
    NOW,
    GenerationCache,
    drift_assessment,
    swing_model_info,
    swing_serving,
)


class BlockingGenerationCache(GenerationCache):
    def __init__(self, generation: LoadedSwingModelGeneration) -> None:
        super().__init__(generation)
        self.entered = threading.Event()
        self.release = threading.Event()

    def get(self, *args: object, **kwargs: object) -> LoadedSwingModelGeneration:
        self.entered.set()
        if not self.release.wait(timeout=5):
            raise TimeoutError("test did not release the blocked generation load")
        return super().get(*args, **kwargs)


def _route_config(horizon: str) -> dict[str, object]:
    return {
        "prediction_serving": {
            "attestation_trust_store": "configs/trust.json",
            "promotion_gate_policy_sha256": "a" * 64,
            "drift_policy_sha256": "b" * 64,
            "routes": {
                "swing": {
                    horizon: {
                        "release_repository": "models/edge_rebuild/swing/promoted",
                        "bar_timeframe": "1Day",
                    }
                }
            },
        }
    }


def test_serving_routes_load_signed_ten_session_swing_configuration() -> None:
    route = serving_routes_from_config(_route_config("10b"))["swing"]["10b"]

    assert route.repository == Path("models/edge_rebuild/swing/promoted")
    assert route.promotion_gate_policy_sha256 == "a" * 64
    assert route.drift_policy_sha256 == "b" * 64


@pytest.mark.parametrize("horizon", ("5d", "10d", "63b", "1h"))
def test_serving_routes_reject_every_horizon_but_ten_sessions(horizon: str) -> None:
    with pytest.raises(ValueError, match="ten-session"):
        serving_routes_from_config(_route_config(horizon))


def test_serving_routes_name_retired_intraday_configuration() -> None:
    config = _route_config("10b")
    routes = config["prediction_serving"]["routes"]  # type: ignore[index]
    routes["intraday"] = {"60m": {"release_repository": "models/intraday"}}  # type: ignore[index]

    with pytest.raises(ValueError, match="intraday prediction is retired"):
        serving_routes_from_config(config)


def test_service_accepts_swing_routes_only(tmp_path: Path) -> None:
    route = ServingRoute(repository=tmp_path, attestation_trust_store=tmp_path / "trust.json", drift_policy_sha256="b" * 64)
    for routes in ({"intraday": {"60m": route}}, {"swing": {"10b": route}, "intraday": {"60m": route}}):
        with pytest.raises(ValueError, match="swing routes only"):
            PredictionService(tmp_path, routes=routes)


@pytest.mark.parametrize("horizon", ("1d", "5d", "10d", "tomorrow", "next_week", "1h", "60m"))
def test_request_horizon_is_a_session_count(horizon: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(tickers=["MSFT"], horizon=horizon)
    assert PredictionRequest(tickers=["MSFT"], horizon=" 10B ").horizon == "10b"
    assert PredictionRequest(tickers=["MSFT"], horizon="252b").horizon == "252b"


@pytest.mark.parametrize("mode", ("intraday", "unified"))
def test_request_mode_is_swing_only(mode: str) -> None:
    with pytest.raises(ValidationError):
        PredictionRequest(tickers=["MSFT"], mode=mode)


def test_retired_readiness_and_replay_views_are_refused() -> None:
    with pytest.raises(ValidationError):
        ReadinessInfo(status="valid", timeframe="intraday")
    with pytest.raises(ValidationError):
        InvestmentReplayRequest(snapshot_id="a" * 64, ticker="MSFT", model_view="intraday")


def test_retired_fields_are_refused_not_dropped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False)
    payload = serving.service.predict(PredictionRequest(tickers=["T000"], as_of=NOW)).model_dump(mode="json")
    [ticker] = payload["predictions"]
    swing = ticker["swing"]

    for changed in (
        {**payload, "predictions": [{**ticker, "intraday": None}]},
        {**payload, "predictions": [{**ticker, "unified_score": 0.5}]},
        {**payload, "predictions": [{**ticker, "swing": {**swing, "unified_score": 0.5}}]},
        {**payload, "predictions": [{**ticker, "swing": {**swing, "readiness": {**swing["readiness"], "intraday_bar_count": 0}}}]},
    ):
        with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
            PredictionResponse.model_validate(changed)
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        ReadinessInfo(status="valid", intraday_bar_count=10)


def test_superseded_response_and_evidence_versions_are_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False)
    response = serving.service.predict(PredictionRequest(tickers=["T000"], as_of=NOW))
    assert response.evidence is not None
    payload = response.model_dump(mode="json")
    evidence = response.evidence.model_dump(mode="json")

    assert PredictionResponse.model_validate(payload) == response
    for version in ("market_predictor.prediction.v1", "market_predictor.prediction.v2"):
        with pytest.raises(ValidationError):
            PredictionResponse.model_validate({**payload, "contract_version": version})
    for version in ("market_predictor.prediction_evidence.v2", "market_predictor.prediction_evidence.v3"):
        with pytest.raises(ValidationError):
            PredictionEvidenceV4.model_validate({**evidence, "contract_version": version})


def test_unconfigured_session_horizon_is_rejected_before_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False)

    with pytest.raises(PredictionValidationError):
        serving.service.predict_swing(PredictionRequest(tickers=["T000"], horizon="63b", as_of=NOW))
    assert serving.generation_cache.loads == 0


def test_missing_signed_swing_generation_fails_closed(tmp_path: Path) -> None:
    service = PredictionService(
        tmp_path,
        routes={
            "swing": {
                "10b": ServingRoute(
                    repository=tmp_path / "missing-generation",
                    attestation_trust_store=tmp_path / "trust.json",
                    promotion_gate_policy_sha256="a" * 64,
                    drift_policy_sha256="b" * 64,
                )
            }
        },
    )

    with pytest.raises(PredictionModelUnavailableError):
        service.predict_swing(PredictionRequest(tickers=["MSFT"]))


def test_oversized_batch_is_rejected_before_model_loading(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, max_tickers_per_request=1)

    with pytest.raises(PredictionValidationError):
        serving.service.predict(PredictionRequest(tickers=["T000", "T001"], as_of=NOW))
    assert serving.generation_cache.loads == 0


def test_concurrent_request_is_rejected_instead_of_queued(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(
        tmp_path, monkeypatch, generation_cache=BlockingGenerationCache, enforce_drift=False
    )
    cache = serving.generation_cache
    assert isinstance(cache, BlockingGenerationCache)
    request = PredictionRequest(tickers=["T000"], as_of=NOW)

    with ThreadPoolExecutor(max_workers=1) as pool:
        pending = pool.submit(serving.service.predict, request)
        assert cache.entered.wait(timeout=5)
        with pytest.raises(PredictionCapacityError):
            serving.service.predict(request)
        cache.release.set()
        response = pending.result(timeout=5)

    assert response.resolved_horizons == {"swing": "10b"}
    assert cache.loads == 1


def test_top_level_predict_persists_immutable_snapshot(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, persist_snapshots=True)

    response = serving.service.predict(PredictionRequest(tickers=["T000"], as_of=NOW))

    assert response.snapshot_id is not None
    assert response.snapshot_id == response.snapshot_sha256
    assert serving.service.snapshot_store.path_for(response.snapshot_id).exists()
    assert response.data_source == "live"


def test_stable_matching_drift_assessment_allows_prediction(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    store = DriftStateStore(tmp_path / "drift")
    serving = swing_serving(tmp_path, monkeypatch, drift_state_store=store)
    store.publish(drift_assessment(swing_model_info(serving, monkeypatch), state="stable", evaluated_at=NOW))

    response = serving.service.predict_swing(PredictionRequest(tickers=["T000"], as_of=NOW))

    assert response.predictions[0].swing is not None


@pytest.mark.parametrize(
    ("state", "evaluated_at", "policy_sha256", "cause"),
    (
        ("severe", NOW, None, None),
        ("stable", NOW + timedelta(microseconds=1), None, "from the future"),
        ("stable", NOW - timedelta(minutes=1_441), None, "is stale"),
        ("stable", NOW, "f" * 64, "policy identity mismatch"),
    ),
    ids=("severe", "from_future", "stale", "unapproved_policy"),
)
def test_unactionable_drift_blocks_prediction(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    state: str,
    evaluated_at: object,
    policy_sha256: str | None,
    cause: str | None,
) -> None:
    store = DriftStateStore(tmp_path / "drift")
    serving = swing_serving(tmp_path, monkeypatch, drift_state_store=store)
    model = swing_model_info(serving, monkeypatch)
    options = {} if policy_sha256 is None else {"policy_sha256": policy_sha256}
    store.publish(drift_assessment(model, state=state, evaluated_at=evaluated_at, **options))

    with pytest.raises(PredictionDriftBlockedError) as blocked:
        serving.service.predict_swing(PredictionRequest(tickers=["T000"], as_of=NOW))
    if cause is None:
        assert blocked.value.__cause__ is None
    else:
        assert cause in str(blocked.value.__cause__)
