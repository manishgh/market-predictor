"""Saved compact-view integration, with synthetic news and no provider calls."""
from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import patch

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.datasets.issuer_news_identity_alignment import (
    SCHEMA,
    _canonical,
    _immutable_json,
    _mapped_view,
    _record_path,
    align_monthly_news,
)
from market_predictor.swing.datasets.issuer_news_preparation import _build_compact_lineage
from market_predictor.swing.features.catalyst_decision_authority import publish_catalyst_decision_authority
from market_predictor.universe.issuer_news_identity import BRIDGE_COLUMNS
from tests.test_monthly_issuer_preparation import _run_preparation
from tests.test_monthly_issuer_preparation import prepared_inputs as prepared_inputs


def test_compact_identity_view_reuses_scores_and_preserves_unknown_coverage(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    prepared = _run_preparation(tmp_path, prepared_inputs)
    source = tmp_path / prepared["sources"]["corrected"]["directory"] / "2024-04"
    score_manifest = json.loads((source / "sentiment/_manifest.json").read_text())
    score_pins = {record["path"]: file_sha256(Path(record["path"])) for record in score_manifest["artifacts"]}
    bridge = pd.DataFrame(columns=BRIDGE_COLUMNS)
    target = tmp_path / "mapped"
    with patch("market_predictor.swing.datasets.issuer_news_identity_alignment._guard"):
        counts = _mapped_view(source, target, bridge, "a" * 64)
    assert counts["mapped_relation_rows"] == 0
    assert counts["relation_rows"] > 0
    assert json.loads((target / "sentiment/_manifest.json").read_text())["artifacts"] == score_manifest["artifacts"]
    assert all(file_sha256(Path(path)) == digest for path, digest in score_pins.items())
    coverage, _ = load_canonical_artifact(target / "collection/source_collections.parquet", allow_research=True)
    original, _ = load_canonical_artifact(source / "collection/source_collections.parquet", allow_research=True)
    pd.testing.assert_frame_equal(coverage.loc[:, original.columns], original)
    assert set(coverage.identity_translation_status) == {"unmapped"}
    config = json.loads(Path(prepared["publication_config"]).read_text())
    record = config["months"]["2024-04"]
    decisions = tmp_path / record["decisions"]["path"]
    with patch("market_predictor.swing.datasets.issuer_news_preparation._guard"):
        _build_compact_lineage(target, decisions, tmp_path / "lineage.toml", tmp_path / "mapped-lineage")
    authority = publish_catalyst_decision_authority([tmp_path / "mapped-lineage"], tmp_path / "mapped-authority",
        canonical_decisions={"path": str(decisions), "sha256": record["decisions"]["sha256"],
            "manifest_sha256": record["decisions"]["manifest_sha256"]})
    assert len(authority.decisions) == 1
    assert authority.manifest["production_ready"] is False


def test_identity_artifact_pin_failure_before_read(tmp_path: Path) -> None:
    path = tmp_path / "source.json"
    path.write_text("{}")
    with pytest.raises(DataReadinessError, match="pin differs"):
        _record_path(tmp_path, {"path": path.name, "sha256": "a" * 64}, {})


def test_canonical_identity_sidecar_required(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="artifact and manifest"):
        _canonical(tmp_path, {"path": "identity.parquet", "sha256": "a" * 64}, "memberships", {})


def test_immutable_publication_rejects_changed_content(tmp_path: Path) -> None:
    path = tmp_path / "manifest.json"
    _immutable_json(path, {"verified": True})
    digest = file_sha256(path)
    _immutable_json(path, {"verified": True})
    assert file_sha256(path) == digest
    with pytest.raises(DataReadinessError, match="changed"):
        _immutable_json(path, {"verified": False})


def test_aligned_monthly_publication_resumes_and_rejects_source_tamper(prepared_inputs, tmp_path: Path) -> None:  # type: ignore[no-untyped-def]
    path, _, projection = prepared_inputs
    config = json.loads(path.read_text())
    config["sources"]["early"] = config["sources"].pop("corrected")
    path.write_text(json.dumps(config))
    prepared = _run_preparation(tmp_path, (path, file_sha256(path), projection))
    marker = tmp_path / "synthetic-owner.txt"
    marker.write_text("synthetic test implementation identity")
    alignment_config = {"schema": SCHEMA,
        "preparation": {"path": "prepared/_manifest.json", "sha256": prepared["manifest_sha256"]},
        **{name: {} for name in ("source_registry", "source_memberships", "target_membership_authority",
            "target_sec_identity_authority", "symbol_corrections")}}
    config_path = tmp_path / "alignment.json"
    config_path.write_text(json.dumps(alignment_config))
    output = tmp_path / "data/research/aligned"
    namespace = "market_predictor.swing.datasets.issuer_news_identity_alignment."
    with patch(namespace + "_bridge", return_value=pd.DataFrame(columns=BRIDGE_COLUMNS)), \
            patch(namespace + "_implementation", return_value={marker.name: file_sha256(marker)}), \
            patch(namespace + "_guard"), patch("market_predictor.swing.datasets.issuer_news_preparation._guard"):
        result = align_monthly_news(root=tmp_path, config_path=config_path,
            expected_config_sha256=file_sha256(config_path), output=output)
        assert result["months"] == 2
        checkpoint = file_sha256(output / "_checkpoint.json")
        repeated = align_monthly_news(root=tmp_path, config_path=config_path,
            expected_config_sha256=file_sha256(config_path), output=output, expected_checkpoint_sha256=checkpoint)
        assert repeated == result
        published = json.loads((output / "monthly-publication.json").read_text())
        assert published["rows"] == 9
        assert all("aligned/lineages" in row["lineages"][0]["directory"] for row in published["months"].values())
        record = next(iter(json.loads((output / "_manifest.json").read_text())["lineages"].values()))
        original_pin = record["record"]["manifest_sha256"]
        original_manifest = tmp_path / record["record"]["directory"] / "_manifest.json"
        assert file_sha256(original_manifest) == original_pin
        original_manifest.write_text("{}")
        with pytest.raises(DataReadinessError, match="pin differs"):
            align_monthly_news(root=tmp_path, config_path=config_path,
                expected_config_sha256=file_sha256(config_path), output=output, expected_checkpoint_sha256=checkpoint)
