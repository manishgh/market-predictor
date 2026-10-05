"""Unit-only historical ownership checks using synthetic, pinned Parquet files."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.io import write_json_object
from market_predictor.swing.contracts.holding_materialization import SourcePin
from market_predictor.swing.datasets import relationship_historical_evidence as owner
from tests.test_swing_return_relationship_reuse import _historical_parent_fixture

MONTH = "2019-07"


def _object(path: Path) -> dict[str, Any]:
    return dict(json.loads(path.read_text(encoding="utf-8")))


def _rewrite(path: Path, value: dict[str, Any]) -> None:
    """Deliberately mutate existing synthetic unit evidence before resealing it."""
    assert path.is_file(), "unit mutation requires an existing synthetic file"
    path.write_text(json.dumps(value, sort_keys=True), encoding="utf-8")


def _repin(root: Path, manifest: dict[str, Any], receipt: SourcePin) -> tuple[SourcePin, SourcePin]:
    path = root / "historical-parent/_manifest.json"
    _rewrite(path, manifest)
    publication = SourcePin(path=path.relative_to(root).as_posix(), sha256=file_sha256(path))
    record = _object(root / receipt.path)
    record["manifest_sha256"] = publication.sha256
    record["unique_decisions"] = manifest["rows"]
    _rewrite(root / receipt.path, record)
    return publication, SourcePin(path=receipt.path, sha256=file_sha256(root / receipt.path))


@pytest.fixture(params=["technical_market", "technical_relationships"])
def publication(tmp_path: Path, request: pytest.FixtureRequest) -> tuple[Path, SourcePin, SourcePin, str]:
    pin, receipt = _historical_parent_fixture(tmp_path)
    profile = str(request.param)
    if profile == "technical_relationships":
        directory = tmp_path / "historical-parent"
        manifest = _object(directory / "_manifest.json")
        manifest["schema"] = "market_predictor.return_relationship_publication"
        original_request = _object(directory / "_request.json")
        original_request.update(schema="market_predictor.return_relationship_request",
            training_eligible=False, promotion_eligible=False)
        _rewrite(directory / "_request.json", original_request)
        manifest["request_sha256"] = file_sha256(directory / "_request.json")
        for month, record in manifest["months"].items():
            child = record["profiles"].pop("technical_market")
            previous = directory / child["path"]
            child["path"] = f"{month}/{profile}.parquet"
            path = directory / child["path"]
            previous.rename(path)
            sidecar = _object(manifest_path_for(previous))
            sidecar.update(artifact_type="swing_return_relationships",
                inputs={"request_sha256": manifest["request_sha256"]})
            write_json_object(manifest_path_for(path), sidecar)
            child.update(rows=record["rows"], decision_ids_sha256=record["decision_ids_sha256"],
                manifest_sha256=file_sha256(manifest_path_for(path)), audit={})
            record["profiles"][profile] = child
        original_receipt = _object(tmp_path / receipt.path)
        original_receipt["scope"] = "published_return_relationship_population_clocks_original_targets"
        _rewrite(tmp_path / receipt.path, original_receipt)
        pin, receipt = _repin(tmp_path, manifest, receipt)
    return tmp_path, pin, receipt, profile


def test_exact_profile_ownership_reads_physical_month_without_mutating_manifest(publication: Any) -> None:
    root, pin, receipt, profile = publication
    before = (root / pin.path).read_bytes()
    evidence = owner.inspect_historical_publication(root, pin, receipt, profile=profile)
    child = evidence.manifest["months"][MONTH]["profiles"][profile]
    assert ("rows" in child) is (profile == "technical_relationships")
    assert ("decision_ids_sha256" in child) is (profile == "technical_relationships")
    if profile == "technical_market":
        assert set(child) == {"audit", "availability_columns", "manifest_sha256", "model_columns", "path", "sha256"}
    frame = owner.historical_month(evidence, MONTH)
    assert frame.decision_id.tolist() == [f"decision-{MONTH}"]
    assert (root / pin.path).read_bytes() == before


@pytest.mark.parametrize("fault", ["missing_rows", "wrong_rows", "wrong_owner"])
def test_rehashed_profile_ownership_contradictions_reject(publication: Any, fault: str) -> None:
    root, pin, receipt, profile = publication
    manifest = _object(root / pin.path)
    child = manifest["months"][MONTH]["profiles"][profile]
    row_owner = child["audit"] if profile == "technical_market" else child
    if fault == "missing_rows":
        row_owner.pop("rows")
    elif fault == "wrong_rows":
        row_owner["rows"] = 2
    elif profile == "technical_market":
        child["decision_ids_sha256"] = manifest["months"][MONTH]["decision_ids_sha256"]
    else:
        child.pop("decision_ids_sha256")
    pin, receipt = _repin(root, manifest, receipt)
    with pytest.raises(DataReadinessError, match="ownership"):
        owner.inspect_historical_publication(root, pin, receipt, profile=profile)


@pytest.mark.parametrize("align_metadata", [False, True])
def test_sidecar_rows_bound_to_month_and_physical_parquet(publication: Any, align_metadata: bool) -> None:
    root, pin, receipt, profile = publication
    manifest = _object(root / pin.path)
    month = manifest["months"][MONTH]
    child = month["profiles"][profile]
    sidecar_path = manifest_path_for(root / "historical-parent" / child["path"])
    sidecar = _object(sidecar_path)
    sidecar["rows"] = 2
    _rewrite(sidecar_path, sidecar)
    child["manifest_sha256"] = file_sha256(sidecar_path)
    if align_metadata:
        month["rows"] = 2
        (child["audit"] if profile == "technical_market" else child)["rows"] = 2
        manifest["rows"] += 1
        request_path = root / "historical-parent/_request.json"
        request = _object(request_path)
        request["rows"] = manifest["rows"]
        _rewrite(request_path, request)
        manifest["request_sha256"] = file_sha256(request_path)
        # First month fails its physical row check before request-input binding.
    pin, receipt = _repin(root, manifest, receipt)
    with pytest.raises(DataReadinessError, match="row count|sidecar"):
        owner.inspect_historical_publication(root, pin, receipt, profile=profile)


def test_rehashed_monthly_decision_hash_is_bound_to_physical_ids(publication: Any) -> None:
    root, pin, receipt, profile = publication
    manifest = _object(root / pin.path)
    month = manifest["months"][MONTH]
    month["decision_ids_sha256"] = json_sha256(["different-decision"])
    if profile == "technical_relationships":
        month["profiles"][profile]["decision_ids_sha256"] = month["decision_ids_sha256"]
    pin, receipt = _repin(root, manifest, receipt)
    evidence = owner.inspect_historical_publication(root, pin, receipt, profile=profile)
    with pytest.raises(DataReadinessError, match="monthly rows"):
        owner.historical_month(evidence, MONTH)


@pytest.mark.parametrize("target", ["artifact", "sidecar"])
def test_month_read_rejects_saved_bytes_changed_after_inspection(publication: Any, target: str) -> None:
    root, pin, receipt, profile = publication
    evidence = owner.inspect_historical_publication(root, pin, receipt, profile=profile)
    path = root / "historical-parent" / MONTH / f"{profile}.parquet"
    if target == "sidecar":
        path = manifest_path_for(path)
    path.write_bytes(path.read_bytes() + b"tamper")
    with pytest.raises(DataReadinessError, match="source changed"):
        owner.historical_month(evidence, MONTH)
