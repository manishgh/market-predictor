"""Synthetic storage relocation tests; no retained market data is modified."""
from __future__ import annotations

import json
import shutil
from datetime import date
from pathlib import Path
from typing import Any

import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.evidence.paths import EvidencePaths
from market_predictor.swing.contracts.research_cohort import SwingResearchCohort, load_swing_research_cohort
from market_predictor.swing.datasets.initial_fit_raw_share_plan import run_initial_fit_raw_share_plan
from market_predictor.swing.datasets.symbol_corrected_sources import _segment
from market_predictor.universe.symbol_correction_policy import load_symbol_correction_policy
from tests.test_initial_fit_raw_share_plan import plan_fixture as plan_fixture
from tests.test_swing_holding_identity_preflight import inventory as inventory
from tests.test_swing_research_cohort import _cohort_inputs
from tests.test_swing_symbol_corrected_sources import _bars


def test_symbol_policy_uses_selected_storage_without_old_root_fallback(tmp_path: Path) -> None:
    root, storage = tmp_path / "project", tmp_path / "storage"
    logical = root / "data/research/corrections.toml"
    logical.parent.mkdir(parents=True)
    source = Path(__file__).resolve().parents[1] / "configs/swing_symbol_corrections.toml"
    logical.write_bytes(source.read_bytes())
    expected = load_symbol_correction_policy(root, logical, file_sha256(logical))
    physical = storage / "research/corrections.toml"
    physical.parent.mkdir(parents=True)
    shutil.copyfile(logical, physical)
    paths = EvidencePaths(root, storage)
    assert load_symbol_correction_policy(root, logical, file_sha256(logical), paths=paths) == expected
    physical.unlink()
    with pytest.raises(DataReadinessError):
        load_symbol_correction_policy(root, logical, file_sha256(logical), paths=paths)


def test_corrected_segment_keeps_logical_archive_and_exact_observations(tmp_path: Path) -> None:
    root, storage = tmp_path / "project", tmp_path / "storage"
    logical = root / "data/raw/archive"
    logical.mkdir(parents=True)
    artifact = _bars(logical)
    expected = _segment(logical, artifact, date(2023, 6, 7), date(2023, 6, 8), replacement=True,
        paths=EvidencePaths(root))
    shutil.copytree(root / "data", storage)
    shutil.rmtree(root / "data")
    paths = EvidencePaths(root, storage)
    assert _segment(paths.directory(logical), artifact, date(2023, 6, 7), date(2023, 6, 8),
        replacement=True, paths=paths) == expected
    (storage / "raw/archive/bars.parquet").write_bytes(b"changed")
    with pytest.raises(DataReadinessError, match="bar file differs"):
        _segment(paths.directory(logical), artifact, date(2023, 6, 7), date(2023, 6, 8),
            replacement=True, paths=paths)


def test_cohort_verifies_relocated_source_pin_bytes(tmp_path: Path) -> None:
    root, storage = tmp_path / "project", tmp_path / "storage"
    source = root / "data/raw/source.json"
    source.parent.mkdir(parents=True)
    source.write_text('{"synthetic": true}', encoding="utf-8")
    payload, _ = _cohort_inputs()
    payload["source_files"] = {"data/raw/source.json": file_sha256(source)}
    cohort = SwingResearchCohort.model_validate_json(json.dumps(payload))
    envelope = {"cohort": payload, "cohort_sha256": cohort.sha256(), "summary": cohort.summary(), "coverage": []}
    envelope["audit_sha256"] = json_sha256(envelope)
    audit = root / "data/research/cohort.json"
    audit.parent.mkdir(parents=True)
    audit.write_text(json.dumps(envelope), encoding="utf-8")
    shutil.copytree(root / "data", storage)
    shutil.rmtree(root / "data")
    paths = EvidencePaths(root, storage)
    assert load_swing_research_cohort(audit, source_root=root, paths=paths) == cohort
    (storage / "raw/source.json").write_text("changed", encoding="utf-8")
    with pytest.raises(DataReadinessError, match="source hash differs"):
        load_swing_research_cohort(audit, source_root=root, paths=paths)


def test_initial_fit_plan_replays_relocated_output_without_changing_request(plan_fixture: dict[str, Any]) -> None:
    root = plan_fixture["root"]
    logical = root / "data/research/raw-plan"
    original = run_initial_fit_raw_share_plan(root, plan_fixture["plan_config"], logical)
    pin = file_sha256(logical / "_authority.json")
    storage = root.parent / (root.name + "-storage")
    shutil.copytree(root / "data", storage)
    shutil.rmtree(root / "data")
    paths = EvidencePaths(root, storage)
    assert run_initial_fit_raw_share_plan(root, plan_fixture["plan_config"], logical,
        expected_plan_sha256=pin, paths=paths) == original
