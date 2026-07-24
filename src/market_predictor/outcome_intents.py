from __future__ import annotations

from datetime import date
from zoneinfo import ZoneInfo

from market_predictor.outcome_contracts import (
    PredictionMaturationIntentV1,
    maturation_key_sha256,
    semantic_prediction_sha256,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.prediction_contracts import (
    IntradayPrediction,
    PredictionResponse,
    PredictionRowEvidenceV1,
    SwingPrediction,
)
from market_predictor.prediction_snapshot import PredictionSnapshotStore
from market_predictor.v3.errors import DataReadinessError

_EASTERN = ZoneInfo("America/New_York")


def register_snapshot_intents(
    snapshot_store: PredictionSnapshotStore,
    outcome_repository: OutcomeRepository,
    snapshot_id: str,
) -> list[PredictionMaturationIntentV1]:
    _, response, _ = snapshot_store.load(snapshot_id)
    intents = maturation_intents_from_response(response, snapshot_id=snapshot_id)
    return [outcome_repository.record_intent(intent) for intent in intents]


def maturation_intents_from_response(
    response: PredictionResponse,
    *,
    snapshot_id: str,
) -> list[PredictionMaturationIntentV1]:
    evidence = response.evidence
    if evidence is None:
        raise DataReadinessError("prediction snapshot has no point-in-time evidence")
    if evidence.identity_status != "complete":
        raise DataReadinessError("only identity-complete live predictions can mature")
    intents: list[PredictionMaturationIntentV1] = []
    for prediction in response.predictions:
        if prediction.swing is not None:
            intents.append(
                _intent(
                    response,
                    snapshot_id=snapshot_id,
                    ticker=prediction.ticker,
                    view="swing",
                    prediction=prediction.swing,
                )
            )
        if prediction.intraday is not None:
            intents.append(
                _intent(
                    response,
                    snapshot_id=snapshot_id,
                    ticker=prediction.ticker,
                    view="intraday",
                    prediction=prediction.intraday,
                )
            )
    if not intents:
        raise DataReadinessError("prediction snapshot has no model views to mature")
    return intents


def _intent(
    response: PredictionResponse,
    *,
    snapshot_id: str,
    ticker: str,
    view: str,
    prediction: SwingPrediction | IntradayPrediction,
) -> PredictionMaturationIntentV1:
    evidence = response.evidence
    assert evidence is not None
    model = response.models.get(view)
    feature = evidence.feature_artifacts.get(view)
    row = _row_evidence(evidence.row_feature_availability, ticker=ticker, view=view)
    if model is None or feature is None:
        raise DataReadinessError(f"{view} maturation model/feature identity is missing")
    required_strings = {
        "canonical_security_id": row.canonical_security_id,
        "decision_group_id": row.decision_group_id,
        "primary_benchmark": row.primary_benchmark,
        "market_regime": row.market_regime,
        "sector": row.sector,
        "market_cap_bucket": row.market_cap_bucket,
        "liquidity_bucket": row.liquidity_bucket,
        "price_feed": row.price_feed,
        "model_release_id": model.release_id,
        "model_artifact_sha256": model.artifact_sha256,
        "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256,
    }
    missing = sorted(name for name, value in required_strings.items() if not value)
    if missing or model.label_policy is None:
        raise DataReadinessError(
            f"{view} maturation identity is incomplete: {', '.join(missing)}"
        )
    probability, downside = _probabilities(prediction)
    horizon = model.resolved_horizon or response.resolved_horizons.get(view)
    if not horizon:
        raise DataReadinessError(f"{view} maturation horizon is missing")
    decision_session = (
        date.fromisoformat(row.session_date_et)
        if row.session_date_et
        else row.decision_time_utc.astimezone(_EASTERN).date()
    )
    base: dict[str, object] = {
        "contract_version": "market_predictor.maturation_intent.v1",
        "ticker": ticker,
        "canonical_security_id": str(row.canonical_security_id),
        "view": view,
        "horizon": horizon,
        "decision_time_utc": row.decision_time_utc,
        "decision_session_et": decision_session,
        "decision_group_id": str(row.decision_group_id),
        "model_release_id": str(model.release_id),
        "model_artifact_sha256": str(model.artifact_sha256),
        "feature_artifact_sha256": feature.artifact_sha256,
        "serving_policy_sha256": evidence.serving_policy_sha256,
        "label_policy_sha256": str(model.label_policy_sha256),
        "execution_policy_sha256": str(model.execution_policy_sha256),
        "label_policy": model.label_policy,
        "primary_benchmark": str(row.primary_benchmark),
        "market_regime": str(row.market_regime),
        "sector": str(row.sector),
        "market_cap_bucket": str(row.market_cap_bucket),
        "liquidity_bucket": str(row.liquidity_bucket),
        "price_feed": str(row.price_feed).upper(),
        "probability": probability,
        "downside_probability": downside,
        "calibration_bin": min(int(probability * 10), 9),
        "signal": prediction.signal,
        "actionable": prediction.readiness.status == "valid"
        and prediction.signal != "not_ready",
        "catalyst_status": prediction.catalyst.status,
        "decision_atr": row.decision_atr,
    }
    semantic_id = semantic_prediction_sha256(base)
    return PredictionMaturationIntentV1.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic_id,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic_id),
        }
    )


def _row_evidence(
    rows: list[PredictionRowEvidenceV1],
    *,
    ticker: str,
    view: str,
) -> PredictionRowEvidenceV1:
    matches = [row for row in rows if row.ticker == ticker and row.view == view]
    if len(matches) != 1:
        raise DataReadinessError(
            f"expected one {view} evidence row for {ticker}; found {len(matches)}"
        )
    return matches[0]


def _probabilities(
    prediction: SwingPrediction | IntradayPrediction,
) -> tuple[float, float | None]:
    if isinstance(prediction, SwingPrediction):
        probability = prediction.probability
        downside = None
    else:
        probability = prediction.opportunity_probability
        downside = prediction.downside_probability
    if probability is None:
        raise DataReadinessError("maturation requires a model probability")
    return probability, downside
