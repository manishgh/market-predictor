"""Real synthetic publication admission, source replay, readiness and 124-column loading."""
from __future__ import annotations

import copy
import json
import shutil
from datetime import date
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.research import swing_return_inputs as loader
from market_predictor.research import swing_training_readiness as readiness
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.contracts.return_feature_profiles import RETURN_RELATIONSHIP_COLUMNS
from market_predictor.swing.contracts.return_training import ReturnTrainingPolicy
from market_predictor.swing.datasets import return_relationship_verification as verifier
from market_predictor.swing.labels.fixed_horizon_readiness import CONTEXT_COLUMNS, RETURN_COLUMNS
from tests import test_swing_return_relationship_publication as publication_tests
from tests.test_swing_feature_history_plan import evidence as evidence
from tests.test_swing_return_relationship_publication import publication_fixture as publication_fixture
from tests.test_swing_training_readiness import REPO, _json


@pytest.fixture
def relationship_admission(publication_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    return _admit_relationship(publication_fixture, monkeypatch)


def _admit_relationship(state: dict[str, Any], monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    root = state["root"]
    package = root / "src/market_predictor"
    for name in (*readiness.IMPLEMENTATION_PATHS, *readiness.RELATIONSHIP_IMPLEMENTATION_PATHS, *verifier.IMPLEMENTATION_PATHS):
        path = package / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, path)
    for module, name in ((verifier, "swing/datasets/return_relationship_verification.py"),
            (readiness, "research/swing_training_readiness.py"), (loader, "research/swing_return_inputs.py")):
        monkeypatch.setattr(module, "__file__", str(package / name))
        monkeypatch.setattr(module, "release_process_memory", lambda: None)
    monkeypatch.setattr(verifier, "_guard", lambda: None)
    monkeypatch.setattr(readiness, "_guard", lambda policy: None)
    monkeypatch.setattr(loader, "_guard", lambda policy: None)
    sessions = state["sessions"]
    # The fixture's smaller calendar is a test boundary, not a production scope override.
    temporal = SimpleNamespace(calendar="XNYS", initial_fit_expected_sessions=len(sessions), label_horizon_sessions=10,
        warmup_sessions=250, initial_fit_end=sessions[-1])
    for module in (readiness, loader):
        monkeypatch.setattr(module, "load_temporal_manifest_config", lambda path: temporal)
        monkeypatch.setattr(module, "build_temporal_schedule", lambda config: SimpleNamespace(
            folds=(SimpleNamespace(train_sessions=sessions),)))
    receipt_path = root / "data/reports/relationship_rows.json"
    receipt = verifier.verify_return_relationship_rows(root, state["publication"], receipt_path)
    policy = copy.deepcopy(state["parent"]["policy"])
    policy.update(published_profile="technical_relationships", publication=state["publication"].model_dump(mode="json"),
        saved_row_verification=dict(path=receipt_path.relative_to(root).as_posix(), sha256=receipt["report_sha256"]))
    config = root / "configs/relationship_readiness.json"
    config_hash = _json(config, policy)
    output = root / "data/reports/relationship_readiness.json"
    report = readiness.audit_swing_training_readiness(root=root, config=config, config_sha256=config_hash, output=output)
    payload = json.loads((REPO / "configs/swing_return_training.json").read_text())
    payload.update(feature_profile="technical_relationships", published_profile="technical_relationships",
        readiness=dict(path=output.relative_to(root).as_posix(), sha256=report["report_sha256"]),
        readiness_config=dict(path=config.relative_to(root).as_posix(), sha256=config_hash))
    return {**state, "receipt": receipt, "report": report, "training_policy": ReturnTrainingPolicy.model_validate(payload)}


def test_real_relationship_publication_to_124_column_loader(relationship_admission: dict[str, Any]) -> None:
    state = relationship_admission
    data = loader.load_return_inputs(state["root"], state["training_policy"])
    verified = verifier.verify_return_relationship_publication(state["root"], state["publication"])
    assert tuple(data.features.columns) == verified.model_columns
    assert data.features.shape == (verified.manifest["rows"], 124)
    assert data.features.dtypes.eq(np.dtype("float32")).all()
    assert tuple(data.features.columns[-4:]) == RETURN_RELATIONSHIP_COLUMNS
    assert data.metadata.security_holdout.any() and not data.metadata.security_holdout.all()
    assert state["receipt"]["additions_source_replayed"] is True
    assert state["report"]["baseline_numerical_replayed"] is False
    originals = []
    published = []
    directory = (state["root"] / state["publication"].path).parent
    for month, record in sorted(verified.months.items()):
        originals.append(pd.read_parquet(verified.parent_path.parent
            / verified.parent_manifest["months"][month]["profiles"]["technical_market"]["path"]))
        published.append(pd.read_parquet(directory / record["profiles"]["technical_relationships"]["path"]))
    baseline = pd.concat(originals, ignore_index=True)
    derivative = pd.concat(published, ignore_index=True)
    columns = [*CONTEXT_COLUMNS, *RETURN_COLUMNS]
    pd.testing.assert_frame_equal(data.metadata.loc[:, columns], baseline.loc[:, columns], check_exact=True)
    pd.testing.assert_frame_equal(data.features, derivative.loc[:, list(verified.model_columns)].astype("float32"), check_exact=True)
    assert any(name.endswith("/return_relationship_verification.py") for name in data.pins)
    assert any(name.endswith("/swing_return_inputs.py") for name in data.pins)


def test_missing_source_clock_publication_replays_and_loads_124_columns(
    evidence: dict[str, Any], monkeypatch: pytest.MonkeyPatch,
) -> None:
    original_bars = publication_tests._bars

    def missing_stock_ingestion(sessions: tuple[date, ...], ticker: str) -> pd.DataFrame:
        frame = original_bars(sessions, ticker)
        if ticker != "SPY":
            frame["ingested_at_utc"] = pd.Series(pd.NaT, index=frame.index, dtype="datetime64[ns, UTC]")
        return frame

    # Generate missing-clock source bytes before the real fixture hashes its authorities.
    monkeypatch.setattr(publication_tests, "_bars", missing_stock_ingestion)
    state = _admit_relationship(publication_fixture.__wrapped__(evidence, monkeypatch), monkeypatch)
    data = loader.load_return_inputs(state["root"], state["training_policy"])
    verified = verifier.verify_return_relationship_publication(state["root"], state["publication"])
    assert data.features.shape == (354, 124)
    assert tuple(data.features.columns) == verified.model_columns
    assert data.features.dtypes.eq(np.dtype("float32")).all()
    assert data.features[list(RETURN_RELATIONSHIP_COLUMNS)].isna().all(axis=None)
    assert state["receipt"]["status"] == "passed"
    assert state["receipt"]["additions_source_replayed"] is True
    assert state["report"]["baseline_numerical_replayed"] is False
    assert len(verified.months) == 59
    directory = (state["root"] / state["publication"].path).parent
    for record in verified.months.values():
        frame = pd.read_parquet(directory / record["profiles"]["technical_relationships"]["path"])
        for name in RETURN_RELATIONSHIP_COLUMNS:
            clock = frame[f"available_at_{name}"]
            assert clock.dtype == pd.DatetimeTZDtype(unit="ns", tz="UTC")
            assert clock.isna().all()
            assert frame[f"missing_reason_{name}"].eq("missing_required_bar_clock").all()


def test_relationship_loader_rejects_current_code_changed_after_readiness(relationship_admission: dict[str, Any]) -> None:
    state = relationship_admission
    source = state["root"] / "src/market_predictor/research/swing_return_inputs.py"
    with source.open("ab") as handle:
        handle.write(b"\n# changed after readiness\n")
    with pytest.raises(DataReadinessError):
        loader.load_return_inputs(state["root"], state["training_policy"])


@pytest.mark.parametrize("mutation", ["value", "future_clock"])
def test_rehashed_addition_value_rejected_by_fresh_source_replay(
    publication_fixture: dict[str, Any], monkeypatch: pytest.MonkeyPatch, mutation: str,
) -> None:
    state = publication_fixture
    root = state["root"]
    for name in verifier.IMPLEMENTATION_PATHS:
        path = root / "src/market_predictor" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(REPO / "src/market_predictor" / name, path)
    monkeypatch.setattr(verifier, "__file__", str(root / "src/market_predictor/swing/datasets/return_relationship_verification.py"))
    monkeypatch.setattr(verifier, "_guard", lambda: None)
    monkeypatch.setattr(verifier, "release_process_memory", lambda: None)
    manifest_path = root / state["publication"].path
    manifest = json.loads(manifest_path.read_text())
    child = next(iter(manifest["months"].values()))["profiles"]["technical_relationships"]
    path = manifest_path.parent / child["path"]
    frame = pd.read_parquet(path)
    name = RETURN_RELATIONSHIP_COLUMNS[0]
    positions = frame.index[frame[name].notna()]
    assert len(positions), "source fixture must expose at least one verifiable relationship value"
    if mutation == "value":
        frame.loc[positions[0], name] += np.float32(0.125)
    else:
        frame.loc[positions[0], f"available_at_{name}"] = (
            frame.loc[positions[0], "decision_time_utc"] + pd.Timedelta(nanoseconds=1))
    frame.to_parquet(path, index=False)
    sidecar_path = manifest_path_for(path)
    sidecar = json.loads(sidecar_path.read_text())
    sidecar["artifact_sha256"] = file_sha256(path)
    child["sha256"] = file_sha256(path)
    child["manifest_sha256"] = _json(sidecar_path, sidecar)
    publication = SourcePin(path=state["publication"].path, sha256=_json(manifest_path, manifest))
    diagnostic = ("relationship source replay values, clocks or reasons differ" if mutation == "value"
        else f"relationship future or missing clock: {name}")
    with pytest.raises(DataReadinessError, match=f"^{diagnostic}$"):
        verifier.verify_return_relationship_rows(root, publication, root / "data/reports/poison.json")
    assert not (root / "data/reports/poison.json").exists()
