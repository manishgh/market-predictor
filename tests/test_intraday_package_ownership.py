from __future__ import annotations

import hashlib
import json
import pickle
from pathlib import Path

import pandas as pd
import pytest
from pydantic import ValidationError

import market_predictor.intraday.contracts as contracts
import market_predictor.intraday.evaluation as evaluation


def test_intraday_public_apis_resolve_to_canonical_packages() -> None:
    package_root = Path(contracts.__file__).resolve().parents[1]

    assert Path(contracts.__file__).resolve() == package_root / "contracts" / "__init__.py"
    assert Path(evaluation.__file__).resolve() == package_root / "evaluation" / "__init__.py"
    assert contracts.IntradayDatasetConfig.__module__ == (
        "market_predictor.intraday.contracts.configs"
    )


def test_intraday_contract_identity_and_label_policy_are_frozen() -> None:
    config = contracts.IntradayDatasetConfig()
    feature_sha256 = hashlib.sha256(
        json.dumps(
            contracts.INTRADAY_MODEL_FEATURES,
            separators=(",", ":"),
        ).encode("utf-8")
    ).hexdigest()

    assert contracts.INTRADAY_FEATURE_SCHEMA_VERSION == "intraday.features.v2"
    assert contracts.INTRADAY_MODEL_SCHEMA_VERSION == "intraday.model.v1"
    assert len(contracts.INTRADAY_MODEL_FEATURES) == 95
    assert feature_sha256 == "a8ae2e7e759631d44a7accbf07b650c4813ed335b97a439418ac240566b3a5d8"
    assert config.label_policy() == {
        "policy": "intraday_label.v2",
        "horizon_minutes": 60,
        "decision_bar_minutes": 5,
        "execution_bar_minutes": 1,
        "decision_stride_bars": 3,
        "target_atr": 1.0,
        "stop_atr": 0.75,
        "round_trip_cost_bps": 10.0,
        "ambiguous_barrier_policy": "stop",
        "entry_rule": "exact_bar_start_at_decision_time",
        "stop_fill_rule": "worse_of_stop_or_trigger_open",
        "target_fill_rule": "target_price",
        "timeout_fill_rule": "final_horizon_bar_close",
        "broad_benchmark": "SPY",
        "growth_benchmark": "QQQ",
    }
    assert config.label_config_sha256() == (
        "0a753b03a83188dac0c9daedb41927310653c4641b03078e2664784bcaec6ac1"
    )


def test_intraday_config_validation_and_pickle_owner_are_stable() -> None:
    config = contracts.IntradayDatasetConfig()
    restored = pickle.loads(pickle.dumps(config))

    assert restored == config
    assert type(restored).__module__ == "market_predictor.intraday.contracts.configs"
    with pytest.raises(ValidationError, match="frozen"):
        config.horizon_minutes = 30
    with pytest.raises(ValidationError, match="Extra inputs are not permitted"):
        contracts.IntradayDatasetConfig(unknown_setting=True)
    with pytest.raises(ValidationError, match="memory guard headroom"):
        contracts.IntradayDatasetConfig(
            max_build_memory_gb=1.0,
            memory_guard_headroom_gb=1.0,
        )


def test_intraday_evaluation_characterization_is_stable() -> None:
    metrics = evaluation.classification_metrics(
        pd.Series([0, 0, 1, 1]),
        pd.Series([0.1, 0.4, 0.6, 0.9]),
    )
    overlap = evaluation.overlap_evidence_summary(
        pd.Series([1.0, 0.5, 0.5]),
        pd.Series(["event-a", "event-b", "event-b"]),
    )

    assert metrics == pytest.approx(
        {
            "roc_auc": 1.0,
            "average_precision": 1.0,
            "accuracy": 1.0,
            "precision": 1.0,
            "recall": 1.0,
            "f1": 1.0,
            "brier_score": 0.085,
            "log_loss": 0.30809306971190853,
            "expected_calibration_error": 0.25,
            "base_positive_rate": 0.5,
            "top_decile_positive_rate": 1.0,
            "top_decile_lift": 2.0,
        }
    )
    assert overlap == {
        "summed_label_uniqueness": 2.0,
        "independent_event_count": 2,
        "effective_sample_size": 2.0,
    }
