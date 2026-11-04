"""Historical bundle verification remains read-only after writer retirement."""
from __future__ import annotations

import json
from pathlib import Path

import pytest
from typer.testing import CliRunner

from market_predictor.core.errors import DataReadinessError
from market_predictor.production_cli import app
from market_predictor.serving import bundle as historical
from market_predictor.swing.contracts import SWING_FEATURE_SCHEMA_VERSION
from tests.support.historical_serving_bundle import (
    historical_bundle_fixture,
    historical_bundle_inputs,
    historical_bundle_timestamp,
)

RETIRED_BUNDLE_COMMANDS = ("publish-serving-bundle", "activate-serving-bundle", "rollback-serving-bundle",
                          "verify-serving-bundle", "show-active-serving-bundle")


@pytest.fixture
def recorded(tmp_path: Path) -> tuple[Path, Path, dict[str, object]]:
    repository, trust, release_id, features = historical_bundle_inputs(tmp_path, "recorded")
    return repository, trust, historical_bundle_fixture(repository, trust, release_id, features)


@pytest.mark.parametrize("command", RETIRED_BUNDLE_COMMANDS)
def test_obsolete_bundle_commands_cannot_change_recorded_pointer(recorded, command: str) -> None:
    repository, _, _ = recorded
    pointer = repository / "active_serving_bundle.json"
    before = pointer.read_bytes()
    result = CliRunner().invoke(app, [command])
    assert result.exit_code == 2
    assert "No such command" in result.output
    assert pointer.read_bytes() == before


def test_historical_owner_has_no_publication_or_activation_path() -> None:
    for name in ("publish_serving_bundle", "activate_serving_bundle", "rollback_serving_bundle"):
        assert not hasattr(historical, name)


def test_complete_historical_bundle_is_verified_without_mutation(recorded) -> None:
    repository, trust, bundle = recorded
    before = {path.relative_to(repository): path.read_bytes() for path in repository.rglob("*") if path.is_file()}
    active = historical.load_active_serving_bundle(repository, attestation_trust_store_path=trust,
                                                   as_of=historical_bundle_timestamp())
    assert active["bundle"]["bundle_id"] == bundle["bundle_id"]
    assert active["bundle"]["feature_schema_version"] == SWING_FEATURE_SCHEMA_VERSION
    assert active["bundle"]["calibration_method"] == "isotonic_prior_fold_only"
    assert {path.relative_to(repository): path.read_bytes() for path in repository.rglob("*") if path.is_file()} == before


def test_feature_mutation_invalidates_historical_bundle(recorded) -> None:
    repository, trust, bundle = recorded
    feature = repository / "serving_bundles" / str(bundle["bundle_id"]) / "features/features.parquet"
    feature.write_bytes(b"mutated")
    with pytest.raises(DataReadinessError, match="feature artifact integrity"):
        historical.verify_serving_bundle(repository, str(bundle["bundle_id"]), attestation_trust_store_path=trust,
                                         as_of=historical_bundle_timestamp())


def test_manifest_policy_mutation_breaks_historical_content_identity(recorded) -> None:
    repository, trust, bundle = recorded
    path = repository / "serving_bundles" / str(bundle["bundle_id"]) / "bundle.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    manifest["prediction_policy_sha256"] = "0" * 64
    path.write_text(json.dumps(manifest), encoding="utf-8")
    with pytest.raises(DataReadinessError, match="content hash"):
        historical.verify_serving_bundle(repository, str(bundle["bundle_id"]), attestation_trust_store_path=trust,
                                         as_of=historical_bundle_timestamp())


def test_partial_historical_bundle_cannot_replace_recorded_pointer(recorded) -> None:
    repository, trust, _ = recorded
    pointer = repository / "active_serving_bundle.json"
    before = pointer.read_bytes()
    directory = repository / "serving_bundles" / ("f" * 64)
    directory.mkdir()
    (directory / "bundle.json").write_text("{}", encoding="utf-8")
    with pytest.raises(DataReadinessError):
        historical.verify_serving_bundle(repository, "f" * 64, attestation_trust_store_path=trust,
                                         as_of=historical_bundle_timestamp())
    assert pointer.read_bytes() == before
