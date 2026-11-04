"""Deterministic UNIT estimators only: no fits, artifacts or release admission."""
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from market_predictor.core.errors import DataReadinessError
from market_predictor.edge_rebuild.training.lgbm_models import _predict_probability
from market_predictor.governance.promotion.bundle_contracts import validate_promoted_bundle
from market_predictor.modeling.candidate_probabilities import candidate_calibrated_probabilities
from market_predictor.modeling.feature_reference import feature_reference_names_sha256, feature_reference_profile_sha256
from market_predictor.serving.swing_inference import LoadedSwingModelGeneration, SwingInferenceEngine
from market_predictor.swing.contracts.model_artifact import SWING_CANDIDATE_MODEL_SCHEMA
from tests.test_swing_inference import CONTRACT, _base_bundle, _feature_reference


class Classifier:
    def __init__(self):
        self.scores = np.array([0.73, 0.18], dtype=np.float64)
        self.seen = None

    def predict_proba(self, features):
        self.seen = features.copy()
        return np.column_stack((1 - self.scores, self.scores))


class Ranker:
    def __init__(self):
        self.scores = np.array([-4., 2.], dtype=np.float64)
        self.seen = None

    def predict(self, features):
        self.seen = features.copy()
        return self.scores.copy()


class Regressor(Ranker):
    def __init__(self):
        super().__init__()
        self.scores = np.array([-0.04, 0.12], dtype=np.float64)


class Calibrator:
    def __init__(self):
        self.seen = None

    def predict_proba(self, scores):
        self.seen = scores.copy()
        probability = 1 / (1 + np.exp(-scores[:, 0]))
        return np.column_stack((1 - probability, probability))


def _engine(fitted, *, model_name):
    bundle = validate_promoted_bundle(_base_bundle(), strategy_contract=CONTRACT)
    reference = _feature_reference(bundle.ordered_feature_columns)
    models = {"classifier": fitted} if model_name == "classifier" else {
        "classifier": SimpleNamespace(estimator=Classifier(), calibrator=Calibrator(), feature_columns=fitted.feature_columns),
        model_name: fitted}
    payload = {"schema": SWING_CANDIDATE_MODEL_SCHEMA, "status": "candidate", "promotion_permitted": False,
        "candidate_id": bundle.model_id, "model_family": bundle.model_family,
        "strategy_contract_sha256": bundle.strategy_contract_sha256, "execution_policy_sha256": bundle.execution_policy_sha256,
        "training_decisions_end_session": bundle.training_decisions_end_session,
        "training_labels_available_through_utc": bundle.training_labels_available_through_utc,
        "feature_columns": bundle.ordered_feature_columns, "feature_reference_profile": reference,
        "feature_reference_profile_sha256": feature_reference_profile_sha256(reference),
        "feature_reference_names_sha256": feature_reference_names_sha256(reference),
        "ablation_profile": bundle.feature_profile, "fitted_models": models,
        "probability_thresholds": dict.fromkeys(models, 0.6)}
    return SwingInferenceEngine(LoadedSwingModelGeneration(generation_id=bundle.sha256(), pointer_sha256="a" * 64,
        bundle=bundle, model_payload=payload))


@pytest.mark.parametrize("estimator_class,model_name", [(Classifier, "classifier"), (Ranker, "classifier"),
                                                        (Regressor, "xgboost_regressor")])
def test_training_and_serving_keep_identical_raw_score_calibration_and_feature_order(estimator_class, model_name):
    bundle = validate_promoted_bundle(_base_bundle(), strategy_contract=CONTRACT)
    columns = tuple(bundle.ordered_feature_columns[index] for index in (0, 2))
    frame = pd.DataFrame(np.arange(2 * len(bundle.ordered_feature_columns), dtype=float).reshape(2, -1),
                         columns=bundle.ordered_feature_columns)
    estimator, calibrator = estimator_class(), Calibrator()
    fitted = SimpleNamespace(estimator=estimator, calibrator=calibrator, feature_columns=columns)
    expected = 1 / (1 + np.exp(-estimator.scores))
    matrix = frame.loc[:, list(columns)].to_numpy(dtype="float32")
    direct = candidate_calibrated_probabilities(estimator, calibrator, matrix)
    training = _predict_probability(fitted, frame, columns)
    serving = _engine(fitted, model_name=model_name).predict(frame)[model_name]
    np.testing.assert_array_equal(direct, expected)
    np.testing.assert_array_equal(training, expected)
    np.testing.assert_array_equal(serving, expected)
    np.testing.assert_array_equal(estimator.seen, matrix)
    np.testing.assert_array_equal(calibrator.seen, estimator.scores.reshape(-1, 1))


@pytest.mark.parametrize("classifier,result", [
    (True, np.array([0.2, 0.3])), (True, np.ones((2, 3))),
    (False, np.ones((2, 1))), (False, np.ones(1)), (False, np.array([np.nan, 0.])),
])
def test_invalid_estimator_shapes_and_nonfinite_scores_cannot_reach_calibration(classifier, result):
    estimator = SimpleNamespace(**{"predict_proba" if classifier else "predict": lambda features: result})
    calibrator = SimpleNamespace(predict_proba=lambda scores: pytest.fail("invalid scores reached calibration"))
    with pytest.raises(DataReadinessError, match="raw scores"):
        candidate_calibrated_probabilities(estimator, calibrator, np.ones((2, 1)))


@pytest.mark.parametrize("result", [np.ones(2), np.ones((2, 3)), np.ones((1, 2)),
    np.array([[0., np.inf], [1., 0.]]), np.array([[0., -0.1], [0., 1.1]])])
def test_invalid_calibrator_shape_or_probability_is_refused(result):
    calibrator = SimpleNamespace(predict_proba=lambda scores: result)
    with pytest.raises(DataReadinessError, match="invalid probabilities"):
        candidate_calibrated_probabilities(Regressor(), calibrator, np.ones((2, 1)))
