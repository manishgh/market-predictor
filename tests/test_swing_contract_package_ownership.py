from __future__ import annotations

import hashlib
import importlib
import json
import pickle
from pathlib import Path

import pytest

import market_predictor.swing.contracts as contracts
from market_predictor.swing.contracts.materialization import (
    SWING_MATERIALIZATION_AUTHORITY_SCHEMA,
    SWING_MATERIALIZATION_MANIFEST_SCHEMA,
)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def test_swing_contract_api_resolves_to_the_descriptive_package() -> None:
    swing_root = Path(contracts.__file__).resolve().parents[1]

    assert Path(contracts.__file__).resolve() == swing_root / "contracts" / "__init__.py"
    assert not (swing_root / "contracts.py").exists()


@pytest.mark.parametrize(
    "model",
    (
        contracts.FrozenConfig,
        contracts.SwingDatasetConfig,
        contracts.SwingTrainingConfig,
        contracts.SwingPromotionConfig,
    ),
)
def test_swing_config_owner_and_pickle_identity_are_stable(
    model: type[contracts.FrozenConfig],
) -> None:
    config = model()
    restored = pickle.loads(pickle.dumps(config))

    assert model.__module__ == "market_predictor.swing.contracts"
    assert restored == config
    assert type(restored).__module__ == "market_predictor.swing.contracts"


def test_swing_feature_and_default_config_hashes_are_stable() -> None:
    feature_sha256 = _sha256(
        json.dumps(contracts.SWING_FEATURES, separators=(",", ":"))
    )
    technical_feature_sha256 = _sha256(
        json.dumps(contracts.TECHNICAL_MARKET_FEATURES, separators=(",", ":"))
    )

    assert len(contracts.SWING_FEATURES) == 99
    assert feature_sha256 == (
        "a841554e6edb6e63e6571cf653e064f51fb9c67a893aac63b266b6e0dfe3792f"
    )
    assert len(contracts.TECHNICAL_MARKET_FEATURES) == 53
    assert technical_feature_sha256 == (
        "4d68fd5327f1cc535ba1458a1138cd4faac866a4c129c686c2a48bede0de81fb"
    )
    assert _sha256(contracts.SwingDatasetConfig().model_dump_json()) == (
        "b09ef6f2d22b48fe31d3e3af1ae16d5ea306ad6276d4f5711c54c992551a189d"
    )
    assert _sha256(contracts.SwingTrainingConfig().model_dump_json()) == (
        "094c8ae89831338a6e64abc317063852b445546d8fc5619317e16528c8147812"
    )
    assert _sha256(contracts.SwingPromotionConfig().model_dump_json()) == (
        "491edfa821bd96459690d00cc4a9fdfb7b8f16628314f22d881fdc0d57ccaae5"
    )
    assert contracts.SwingDatasetConfig().label_config_sha256() == (
        "20a8bfdf233102f702eca13141cdeddc45afd5aa3db38703bb8f6edbeffe281e"
    )


def test_swing_materialization_schema_identities_are_stable() -> None:
    assert SWING_MATERIALIZATION_MANIFEST_SCHEMA == (
        "edge_rebuild.swing_panel_materialization.v12"
    )
    assert SWING_MATERIALIZATION_AUTHORITY_SCHEMA == (
        "edge_rebuild.swing_panel_materialization_authority.v12"
    )


@pytest.mark.parametrize(
    "module_name",
    (
        "market_predictor.edge_rebuild.swing_materialization",
        "market_predictor.edge_rebuild.swing_training",
        "market_predictor.edge_rebuild.training.data_io",
    ),
)
def test_materialization_schema_constants_have_no_accidental_aliases(
    module_name: str,
) -> None:
    module = importlib.import_module(module_name)

    assert not hasattr(module, "SWING_MATERIALIZATION_MANIFEST_SCHEMA")
    assert not hasattr(module, "SWING_MATERIALIZATION_AUTHORITY_SCHEMA")
