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

    intents = register_snapshot_intents(
        serving.service.snapshot_store,
        repository,
        response.snapshot_id,
    )

    assert {intent.ticker for intent in intents} == {"T000", "T059"}
    for intent in intents:
        assert (intent.view, intent.horizon) == ("swing", "10b")
        assert intent.model_release_id == response.models["swing"].release_id
        assert repository.load_intent(intent.maturation_key) == intent
    # MISSING is outside the live universe: it abstains unscored, so nothing is monitored.
    observations = monitoring_observations_from_response(response, snapshot_id=response.snapshot_id)
    assert {(observation.ticker, observation.view) for observation in observations} == {
        ("T000", "swing"),
        ("T059", "swing"),
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
