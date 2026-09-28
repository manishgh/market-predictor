from __future__ import annotations

from dataclasses import dataclass
from datetime import date
from zoneinfo import ZoneInfo

from market_predictor.core.cross_section_contracts import CrossSectionMember
from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import (
    PredictionResponse,
    PredictionRowEvidence,
    SwingPrediction,
)
from market_predictor.governance.outcomes.contracts import (
    PredictionMaturationIntent,
    PredictionMonitoringObservation,
    content_sha256,
    maturation_key_sha256,
    monitoring_semantic_sha256,
    semantic_prediction_sha256,
)
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.governance.outcomes.session_records import SessionRecord
from market_predictor.monitoring_lease import monitoring_lease
from market_predictor.serving.snapshot_store import PredictionSnapshotStore

_EASTERN = ZoneInfo("America/New_York")
# Abstentions the model never scored: outside the live universe, or a member whose inputs were incomplete.
UNMONITORED_ABSTENTIONS = (["out_of_universe"], ["live_inputs_incomplete"])


@dataclass(frozen=True, slots=True)
class SnapshotRegistration:
    intents: list[PredictionMaturationIntent]
    # Requested tickers the model never scored, by abstention reason; they have no observation.
    unmonitored_tickers: dict[str, list[str]]
    session_record: SessionRecord | None = None


def register_snapshot_intents(
    snapshot_store: PredictionSnapshotStore,
    outcome_repository: OutcomeRepository,
    snapshot_id: str,
) -> SnapshotRegistration:
    from market_predictor.serving.session_registration import register_cross_section_snapshot
    with monitoring_lease("register-outcome-intents"):
        return register_cross_section_snapshot(snapshot_store, outcome_repository, snapshot_id)


def maturation_intents_from_response(
    response: PredictionResponse,
    *,
    snapshot_id: str,
) -> list[PredictionMaturationIntent]:
    evidence = response.evidence
    if evidence is None:
        raise DataReadinessError("prediction snapshot has no point-in-time evidence")
    if evidence.identity_status != "complete":
        raise DataReadinessError("only identity-complete live predictions can mature")
    intents: list[PredictionMaturationIntent] = []
    for prediction in response.predictions:
        if (
            prediction.swing is not None
            and prediction.swing.probability is not None
            and prediction.swing.readiness.status == "valid"
        ):
            intents.append(
                _intent(
                    response,
                    snapshot_id=snapshot_id,
                    ticker=prediction.ticker,
                    view="swing",
                    prediction=prediction.swing,
                )
            )
    return intents


def monitoring_observations_from_response(
    response: PredictionResponse,
    *,
    snapshot_id: str,
    intents: dict[tuple[str, str], PredictionMaturationIntent] | None = None,
    members: tuple[CrossSectionMember, ...] | None = None,
) -> list[PredictionMonitoringObservation]:
    evidence = response.evidence
    if evidence is None or evidence.identity_status != "complete":
        raise DataReadinessError(
            "only identity-complete live predictions can be monitored"
        )
    intent_map = (
        intents
        if intents is not None
        else {
            (intent.ticker, intent.view): intent
            for intent in maturation_intents_from_response(response, snapshot_id=snapshot_id)
        }
    )
    member_map = {member.ticker: member for member in members or ()}
    scored = {(row.ticker, row.view) for row in evidence.row_feature_availability}
    observations: list[PredictionMonitoringObservation] = []
    for prediction in response.predictions:
        if prediction.swing is None or response.models.get("swing") is None:
            continue
        if (prediction.ticker, "swing") not in scored:
            if members is not None:
                member = member_map.get(prediction.ticker)
                if member is None or member.abstention_reason is None:
                    raise DataReadinessError("unscored monitoring member has no verified abstention")
                observations.append(_unscored_observation(response, member, snapshot_id=snapshot_id))
                continue
            # An unscored abstention carries no model decision; the caller reports it as unmonitored.
            if prediction.swing.abstention_reasons not in UNMONITORED_ABSTENTIONS:
                raise DataReadinessError(f"scored swing prediction for {prediction.ticker} has no evidence row")
            continue
        observations.append(
            _observation(
                response,
                snapshot_id=snapshot_id,
                ticker=prediction.ticker,
                view="swing",
                prediction=prediction.swing,
                intent=intent_map.get((prediction.ticker, "swing")),
            )
        )
    if not observations:
        raise DataReadinessError("prediction snapshot has no model views to monitor")
    return observations


def _observation(
    response: PredictionResponse,
    *,
    snapshot_id: str,
    ticker: str,
    view: str,
    prediction: SwingPrediction,
    intent: PredictionMaturationIntent | None,
) -> PredictionMonitoringObservation:
    evidence = response.evidence
    assert evidence is not None
    model = response.models[view]
    feature = evidence.feature_artifacts.get(view)
    row = _row_evidence(evidence.row_feature_availability, ticker=ticker, view=view)
    if feature is None:
        raise DataReadinessError(f"{view} monitoring feature identity is missing")
    required_strings = {
        "decision_group_id": row.decision_group_id,
        "market_regime": row.market_regime,
        "sector": row.sector,
        "market_cap_bucket": row.market_cap_bucket,
        "liquidity_bucket": row.liquidity_bucket,
        "model_release_id": model.release_id,
        "model_artifact_sha256": model.artifact_sha256,
        "prediction_policy_sha256": model.prediction_policy_sha256,
        "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256,
    }
    missing = sorted(name for name, value in required_strings.items() if not value)
    if missing:
        raise DataReadinessError(
            f"{view} monitoring identity is incomplete: {', '.join(missing)}"
        )
    probability = prediction.probability
    horizon = model.resolved_horizon or response.resolved_horizons.get(view)
    if not horizon:
        raise DataReadinessError(f"{view} monitoring horizon is missing")
    decision_session = (
        date.fromisoformat(row.session_date_et)
        if row.session_date_et
        else row.decision_time_utc.astimezone(_EASTERN).date()
    )
    content: dict[str, object] = {
        "contract": "market_predictor.prediction_observation",
        "snapshot_id": snapshot_id,
        "ticker": ticker,
        "view": view,
        "horizon": horizon,
        "decision_time_utc": row.decision_time_utc,
        "decision_session_et": decision_session,
        "decision_group_id": str(row.decision_group_id),
        "model_release_id": str(model.release_id),
        "model_artifact_sha256": str(model.artifact_sha256),
        "feature_artifact_sha256": feature.artifact_sha256,
        "prediction_policy_sha256": str(model.prediction_policy_sha256),
        "label_policy_sha256": str(model.label_policy_sha256),
        "execution_policy_sha256": str(model.execution_policy_sha256),
        "market_regime": str(row.market_regime),
        "sector": str(row.sector),
        "market_cap_bucket": str(row.market_cap_bucket),
        "liquidity_bucket": str(row.liquidity_bucket),
        "probability": probability,
        "calibration_bin": min(int(probability * 10), 9) if probability is not None else None,
        "signal": prediction.signal,
        "rank": prediction.rank,
        "selection_eligible": prediction.selection_eligible,
        "selected_for_policy": prediction.selected_for_policy,
        "actionable": prediction.readiness.status == "valid"
        and prediction.selected_for_policy
        and prediction.signal != "not_ready",
        "readiness_status": prediction.readiness.status,
        "catalyst_status": prediction.catalyst.status,
        "maturation_key": intent.maturation_key if intent is not None else None,
    }
    content["semantic_prediction_id"] = (
        intent.semantic_prediction_id
        if intent is not None
        else monitoring_semantic_sha256(content)
    )
    return PredictionMonitoringObservation.model_validate(
        {**content, "observation_id": content_sha256(content)}
    )


def _intent(
    response: PredictionResponse,
    *,
    snapshot_id: str,
    ticker: str,
    view: str,
    prediction: SwingPrediction,
) -> PredictionMaturationIntent:
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
        "prediction_policy_sha256": model.prediction_policy_sha256,
    }
    missing = sorted(name for name, value in required_strings.items() if not value)
    if (
        missing
        or model.label_policy is None
        or model.prediction_policy is None
        or model.prediction_policy_sha256
        != evidence.view_prediction_policy_sha256.get(view)
    ):
        raise DataReadinessError(
            f"{view} maturation identity is incomplete: {', '.join(missing)}"
        )
    probability = prediction.probability
    if probability is None:
        raise DataReadinessError("maturation requires a model probability")
    horizon = model.resolved_horizon or response.resolved_horizons.get(view)
    if not horizon:
        raise DataReadinessError(f"{view} maturation horizon is missing")
    decision_session = (
        date.fromisoformat(row.session_date_et)
        if row.session_date_et
        else row.decision_time_utc.astimezone(_EASTERN).date()
    )
    base: dict[str, object] = {
        "contract": "market_predictor.maturation_intent",
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
        "prediction_policy_sha256": str(model.prediction_policy_sha256),
        "label_policy_sha256": str(model.label_policy_sha256),
        "execution_policy_sha256": str(model.execution_policy_sha256),
        "prediction_policy": model.prediction_policy,
        "label_policy": model.label_policy,
        "primary_benchmark": str(row.primary_benchmark),
        "market_regime": str(row.market_regime),
        "sector": str(row.sector),
        "market_cap_bucket": str(row.market_cap_bucket),
        "liquidity_bucket": str(row.liquidity_bucket),
        "price_feed": str(row.price_feed).upper(),
        "probability": probability,
        "calibration_bin": min(int(probability * 10), 9),
        "signal": prediction.signal,
        "rank": prediction.rank,
        "selection_eligible": prediction.selection_eligible,
        "selected_for_policy": prediction.selected_for_policy,
        "actionable": prediction.readiness.status == "valid"
        and prediction.selected_for_policy
        and prediction.signal != "not_ready",
        "catalyst_status": prediction.catalyst.status,
        "decision_atr_fraction": _decision_atr_fraction(prediction),
    }
    semantic_id = semantic_prediction_sha256(base)
    return PredictionMaturationIntent.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic_id,
            "maturation_key": maturation_key_sha256(snapshot_id, semantic_id),
        }
    )


def _row_evidence(
    rows: list[PredictionRowEvidence],
    *,
    ticker: str,
    view: str,
) -> PredictionRowEvidence:
    matches = [row for row in rows if row.ticker == ticker and row.view == view]
    if len(matches) != 1:
        raise DataReadinessError(
            f"expected one {view} evidence row for {ticker}; found {len(matches)}"
        )
    return matches[0]


def _decision_atr_fraction(prediction: SwingPrediction) -> float:
    if prediction.managed_risk is None:
        raise DataReadinessError(f"maturation requires the managed risk context for {prediction.ticker}")
    return prediction.managed_risk.atr_fraction_of_latest_close


def _unscored_observation(
    response: PredictionResponse, member: CrossSectionMember, *, snapshot_id: str,
) -> PredictionMonitoringObservation:
    evidence = response.evidence
    assert evidence is not None
    model = response.models["swing"]
    decision = evidence.prediction_cutoff_utc
    content: dict[str, object] = {
        "contract": "market_predictor.prediction_observation", "snapshot_id": snapshot_id,
        "ticker": member.ticker, "view": "swing", "horizon": model.resolved_horizon,
        "decision_time_utc": decision, "decision_session_et": decision.astimezone(_EASTERN).date(),
        "decision_group_id": decision.isoformat(), "model_release_id": model.release_id,
        "model_artifact_sha256": model.artifact_sha256,
        "feature_artifact_sha256": evidence.feature_artifacts["swing"].artifact_sha256,
        "prediction_policy_sha256": model.prediction_policy_sha256, "label_policy_sha256": model.label_policy_sha256,
        "execution_policy_sha256": model.execution_policy_sha256,
        "market_regime": None, "sector": member.sector, "market_cap_bucket": None, "liquidity_bucket": None,
        "probability": None, "calibration_bin": None, "signal": "abstain", "rank": None,
        "selection_eligible": False, "selected_for_policy": False, "actionable": False,
        "readiness_status": "invalid", "catalyst_status": "unavailable", "maturation_key": None,
    }
    content["semantic_prediction_id"] = monitoring_semantic_sha256(content)
    return PredictionMonitoringObservation.model_validate({**content, "observation_id": content_sha256(content)})
