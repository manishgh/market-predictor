"""Explicit reaction-profile consumption; fixtures contain synthetic evidence only."""
from __future__ import annotations

import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import Mock

import numpy as np
import pandas as pd
import pytest
from pydantic import ValidationError

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import swing_return_inputs as loader
from market_predictor.research import swing_training_readiness as readiness
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.issuer_reaction import REACTION_COLUMNS
from market_predictor.swing.contracts.issuer_reaction_profile import ISSUER_REACTION_PROFILE
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.contracts.training_readiness import TrainingReadinessPolicy
from market_predictor.swing.datasets import issuer_reaction_verification as verifier
from market_predictor.swing.labels.fixed_horizon_readiness import CONTEXT_COLUMNS, RETURN_COLUMNS
from market_predictor.swing.training.return_validation import date_balanced_weights
from tests.test_swing_issuer_reaction_publication import qualification_fixture as qualification_fixture
from tests.test_swing_issuer_reaction_publication import reaction_publication as reaction_publication
from tests.test_swing_return_relationship_publication import publication_fixture as publication_fixture
from tests.test_swing_training_readiness import _json, _run
from tests.test_swing_training_readiness import publication as publication

REPO = Path(__file__).resolve().parents[1]


def test_reaction_profile_changes_no_fitting_setting() -> None:
    original = json.loads((REPO / "configs/swing_return_training.json").read_text())
    baseline = ReturnTrainingPolicy.model_validate(original)
    reaction = ReturnTrainingPolicy.model_validate({**original, "feature_profile": ISSUER_REACTION_PROFILE,
                                                    "published_profile": ISSUER_REACTION_PROFILE})
    excluded = {"feature_profile", "published_profile"}
    assert reaction.model_dump(exclude=excluded) == baseline.model_dump(exclude=excluded)
    readiness_payload = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    assert TrainingReadinessPolicy.model_validate({**readiness_payload,
        "published_profile": ISSUER_REACTION_PROFILE}).published_profile == ISSUER_REACTION_PROFILE


@pytest.mark.parametrize("feature,published", [
    (ISSUER_REACTION_PROFILE, "technical_market"), (ISSUER_REACTION_PROFILE, "technical_relationships"),
    ("existing_technical", ISSUER_REACTION_PROFILE), ("technical_relationships", ISSUER_REACTION_PROFILE),
    ("catalyst_full", "catalyst_full"),
])
def test_reaction_profile_cannot_substitute_a_different_saved_profile(feature: str, published: str) -> None:
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    with pytest.raises(ValidationError):
        ReturnTrainingPolicy.model_validate({**payload, "feature_profile": feature, "published_profile": published})


def test_generic_catalyst_is_not_a_reaction_readiness_profile() -> None:
    payload = json.loads((REPO / "configs/swing_training_readiness.json").read_text())
    with pytest.raises(ValidationError):
        TrainingReadinessPolicy.model_validate({**payload, "published_profile": "catalyst_full"})


def test_reaction_readiness_requires_independent_receipt_before_loading_rows(
    publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    # Deliberately relabel a synthetic generic publication. The dedicated verifier,
    # not a generic profile branch, must reject it before any numerical row load.
    publication["policy"]["published_profile"] = ISSUER_REACTION_PROFILE
    publication["config_sha256"] = _json(publication["config"], publication["policy"])
    verifier = Mock(side_effect=DataReadinessError("synthetic reaction receipt missing"))
    read = Mock(side_effect=AssertionError("rows loaded before reaction receipt"))
    monkeypatch.setattr(readiness, "validate_issuer_reaction_receipt", verifier)
    monkeypatch.setattr(readiness, "load_canonical_artifact", read)
    with pytest.raises(DataReadinessError, match="reaction receipt missing"):
        _run(publication)
    verifier.assert_called_once()
    read.assert_not_called()
    assert not publication["output"].exists()


def test_real_reaction_publication_receipt_readiness_and_126_loader_chain(
    reaction_publication: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    state = reaction_publication
    root, relationship = state["root"], state["relationship"]
    package = root / "src/market_predictor"
    for name in (*readiness.IMPLEMENTATION_PATHS, *readiness.REACTION_IMPLEMENTATION_PATHS, *verifier.IMPLEMENTATION_PATHS):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, path)
    for module, name in ((readiness, "research/swing_training_readiness.py"),
                         (loader, "research/swing_return_inputs.py"),
                         (verifier, "swing/datasets/issuer_reaction_verification.py")):
        monkeypatch.setattr(module, "__file__", str(package / name))
        monkeypatch.setattr(module, "release_process_memory", lambda: None)
    monkeypatch.setattr(verifier, "_guard", lambda: None)
    sessions = relationship["sessions"]
    temporal = SimpleNamespace(calendar="XNYS", initial_fit_expected_sessions=len(sessions),
        label_horizon_sessions=10, warmup_sessions=250, initial_fit_end=sessions[-1])
    for module in (readiness, loader):
        monkeypatch.setattr(module, "_guard", lambda policy: None)
        monkeypatch.setattr(module, "load_temporal_manifest_config", lambda path: temporal)
        monkeypatch.setattr(module, "build_temporal_schedule", lambda config: SimpleNamespace(
            folds=(SimpleNamespace(train_sessions=sessions),)))
    parent_receipt_pin = state["policy"].parent_saved_row_verification
    parent_receipt = json.loads((root / parent_receipt_pin.path).read_text())
    config124 = root / "configs/current_relationship_readiness.json"
    memory124 = {**relationship["parent"]["policy"], "published_profile": "technical_relationships",
        "publication": relationship["publication"].model_dump(mode="json"),
        "saved_row_verification": parent_receipt_pin.model_dump(mode="json")}
    config124_hash = _json(config124, memory124)
    output124 = root / "data/reports/current_relationship_readiness.json"
    report124 = readiness.audit_swing_training_readiness(root=root, config=config124,
        config_sha256=config124_hash, output=output124)
    payload124 = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload124.update(feature_profile="technical_relationships", published_profile="technical_relationships",
        readiness={"path": output124.relative_to(root).as_posix(), "sha256": report124["report_sha256"]},
        readiness_config={"path": config124.relative_to(root).as_posix(), "sha256": config124_hash})
    policy124 = ReturnTrainingPolicy.model_validate(payload124)
    data124 = loader.load_return_inputs(root, policy124)
    assert file_sha256(root / parent_receipt_pin.path) == parent_receipt_pin.sha256
    assert report124["saved_row_verification_source_provenance"] == parent_receipt["source_files"]
    original_report124 = output124.read_bytes()
    changed_report124 = _json(output124, {**report124, "saved_row_verification_source_provenance": {}})
    changed_policy124 = policy124.model_copy(update={"readiness": SourcePin(
        path=policy124.readiness.path, sha256=changed_report124)})
    with pytest.raises(DataReadinessError, match="saved receipt provenance differs"):
        loader.load_return_inputs(root, changed_policy124)
    output124.write_bytes(original_report124)
    receipt_path = root / "data/reports/reaction_rows_for_training.json"
    receipt = verifier.verify_issuer_reaction_rows(root, state["publication"], receipt_path)
    config = root / "configs/reaction_readiness.json"
    memory_policy = {**relationship["parent"]["policy"], "published_profile": ISSUER_REACTION_PROFILE,
        "publication": state["publication"].model_dump(mode="json"),
        "saved_row_verification": {"path": receipt_path.relative_to(root).as_posix(), "sha256": receipt["report_sha256"]}}
    config_hash = _json(config, memory_policy)
    output = root / "data/reports/reaction_readiness.json"
    report = readiness.audit_swing_training_readiness(root=root, config=config, config_sha256=config_hash, output=output)
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload.update(feature_profile=ISSUER_REACTION_PROFILE, published_profile=ISSUER_REACTION_PROFILE,
        readiness={"path": output.relative_to(root).as_posix(), "sha256": report["report_sha256"]},
        readiness_config={"path": config.relative_to(root).as_posix(), "sha256": config_hash})
    policy = ReturnTrainingPolicy.model_validate(payload)
    data = loader.load_return_inputs(root, policy)
    verified = verifier.validate_issuer_reaction_receipt(root, state["publication"],
        json.loads(receipt_path.read_text()))
    assert tuple(data.features.columns) == verified.model_columns and data.features.shape[1] == 126
    assert tuple(data.features.columns[-2:]) == REACTION_COLUMNS
    assert data.features.dtypes.eq(np.dtype("float32")).all()
    pd.testing.assert_frame_equal(data.features.iloc[:, :124], data124.features, check_exact=True)
    pd.testing.assert_frame_equal(data.metadata, data124.metadata, check_exact=True)
    originals = []
    published = []
    for month, record in sorted(verified.months.items()):
        parent_child = verified.parent_manifest["months"][month]["profiles"]["technical_relationships"]
        originals.append(pd.read_parquet(verified.parent_path.parent / parent_child["path"]))
        child = record["profiles"][ISSUER_REACTION_PROFILE]
        published.append(pd.read_parquet((root / state["publication"].path).parent / child["path"]))
    baseline = pd.concat(originals, ignore_index=True)
    derivative = pd.concat(published, ignore_index=True)
    metadata = [*CONTEXT_COLUMNS, *RETURN_COLUMNS]
    pd.testing.assert_frame_equal(data.metadata.loc[:, metadata], baseline.loc[:, metadata], check_exact=True)
    pd.testing.assert_frame_equal(data.features.iloc[:, :124], baseline.loc[:, list(verified.model_columns[:124])], check_exact=True)
    pd.testing.assert_frame_equal(data.features, derivative.loc[:, list(verified.model_columns)].astype("float32"), check_exact=True)
    np.testing.assert_array_equal(date_balanced_weights(data.metadata.session_date_et), date_balanced_weights(baseline.session_date_et))
    assert data.metadata.security_holdout.any() and not data.metadata.security_holdout.all()
    assert report["qualification_source_replayed"] is True and report["baseline_numerical_replayed"] is False
    assert report["training_eligible"] is False and report["promotion_eligible"] is False
    read = Mock(side_effect=AssertionError("numeric loader called before receipt identity check"))
    monkeypatch.setattr(loader, "load_canonical_artifact", read)
    for field, value in (("event_authority_sha256", "f" * 64), ("qualification_publication", {}),
                         ("qualification_source_replayed", False), ("profile_sha256", "f" * 64),
                         ("saved_row_verification_sha256", "f" * 64)):
        poisoned = {**report, field: value}
        digest = _json(output, poisoned)
        changed_policy = policy.model_copy(update={"readiness": SourcePin(path=policy.readiness.path, sha256=digest)})
        with pytest.raises(DataReadinessError):
            loader.load_return_inputs(root, changed_policy)
        read.assert_not_called()
