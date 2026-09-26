from __future__ import annotations

from pathlib import Path

import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.core.prediction_contracts import PredictionRequest
from market_predictor.governance.outcomes.repository import OutcomeRepository
from market_predictor.serving.outcome_intents import (
    monitoring_observations_from_response,
    register_snapshot_intents,
)
from tests.support.swing_serving import NOW, swing_serving


def test_registers_identity_complete_swing_snapshot_for_maturation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False, persist_snapshots=True)
    response = serving.service.predict(
        PredictionRequest(tickers=["T000", "T059", "MISSING"], as_of=NOW)
    )
    assert response.snapshot_id is not None
    repository = OutcomeRepository(tmp_path / "data/outcomes")

    registration = register_snapshot_intents(
        serving.service.snapshot_store,
        repository,
        response.snapshot_id,
    )

    assert {intent.ticker for intent in registration.intents} == {"T000", "T059"}
    assert registration.unmonitored_tickers == {"out_of_universe": ["MISSING"]}
    for intent in registration.intents:
        assert (intent.view, intent.horizon) == ("swing", "10b")
        assert intent.model_release_id == response.models["swing"].release_id
        assert repository.load_intent(intent.maturation_key) == intent
    # MISSING is outside the live universe: it abstains unscored, so nothing is monitored.
    observations = monitoring_observations_from_response(response, snapshot_id=response.snapshot_id)
    assert {(observation.ticker, observation.view) for observation in observations} == {
        ("T000", "swing"),
        ("T059", "swing"),
    }


def test_members_with_incomplete_inputs_are_reported_not_called_out_of_universe(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(
        tmp_path, monkeypatch, enforce_drift=False, persist_snapshots=True, excluded_tickers=("T060",)
    )
    response = serving.service.predict(PredictionRequest(tickers=["T000", "T060", "MISSING"], as_of=NOW))
    assert response.snapshot_id is not None
    reasons = {
        prediction.ticker: prediction.swing.abstention_reasons
        for prediction in response.predictions
        if prediction.swing is not None
    }

    registration = register_snapshot_intents(
        serving.service.snapshot_store, OutcomeRepository(tmp_path / "data/outcomes"), response.snapshot_id
    )

    assert reasons == {"T000": [], "T060": ["live_inputs_incomplete"], "MISSING": ["out_of_universe"]}
    assert registration.unmonitored_tickers == {
        "live_inputs_incomplete": ["T060"],
        "out_of_universe": ["MISSING"],
    }


def test_scored_prediction_without_evidence_row_is_refused(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    serving = swing_serving(tmp_path, monkeypatch, enforce_drift=False)
    response = serving.service.predict(PredictionRequest(tickers=["T000"], as_of=NOW))
    assert response.evidence is not None
    stripped = response.model_copy(
        update={"evidence": response.evidence.model_copy(update={"row_feature_availability": []})}
    )

    with pytest.raises(DataReadinessError, match="has no evidence row"):
        monitoring_observations_from_response(stripped, snapshot_id="a" * 64, intents={})
