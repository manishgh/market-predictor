"""Public metadata projection orchestration; independent verifiers are fixture doubles."""
from __future__ import annotations

import shutil
from contextlib import contextmanager
from datetime import date
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.hashing import json_sha256
from market_predictor.swing.datasets import corrected_decisions
from market_predictor.swing.datasets.corrected_outcomes import load_corrected_outcome_policy


@pytest.fixture
def metadata(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    repo = Path(__file__).resolve().parents[1]
    (tmp_path / "configs").mkdir()
    for name in ("swing_corrected_outcomes.toml", "swing_initial_fit_raw_share_plan.toml"):
        shutil.copyfile(repo / "configs" / name, tmp_path / "configs" / name)
    config = tmp_path / "configs/swing_corrected_outcomes.toml"
    pin = file_sha256(config)
    policy = load_corrected_outcome_policy(tmp_path, config, pin)
    document_ids = {name for rule in policy.decision_corrections for name in rule.document_ids}
    documents = {"status": "collected_unreviewed", "documents": [
        {"document_id": name, "attempts": []} for name in sorted(document_ids)]}
    mapping = SimpleNamespace(parent_plan_sha256="f" * 64, document_inventory="documents/inventory.toml",
        document_archive="documents/archive", document_report_sha256=json_sha256(documents))
    preflight = "data/reports/swing_research_cohort/holding_identity_preflight.json"
    request = {"source_files": {preflight: "f" * 64, "metadata/manifest.json": "f" * 64},
        "retained_security_ids": [f"fixture:{index}" for index in range(586)],
        "excluded_security_ids": [f"excluded:{index}" for index in range(45)],
        "decision_start": "2019-07-09", "initial_fit_end": "2024-05-28", "cohort_sha256": "c" * 64}
    objects = {
        policy.source_selection.path: {"correction_plan": "plans/corrected", "correction_plan_sha256": "f" * 64,
            "policy_sha256": policy.symbol_corrections.sha256},
        "plans/corrected/_authority.json": {"request_sha256": "f" * 64},
        "plans/corrected/_request.json": {"policy": {"parent_plan": "plans/parent", "parent_plan_sha256": "f" * 64}},
        "plans/parent/_authority.json": {"request_sha256": "f" * 64},
        "plans/parent/_request.json": request,
        preflight: {"request": {"parent_manifest_path": "metadata/manifest.json"}},
        "metadata/manifest.json": {"files": [
            {"partition_month": "2024-01", "first_session": "2024-01-02", "last_session": "2024-01-31"},
            {"partition_month": "2025-01", "first_session": "2025-01-02", "last_session": "2025-01-31"}]},
    }
    calls: list[str] = []

    @contextmanager
    def verified_plan(root: Path, config: Path, directory: Path, *, expected_plan_sha256: str) -> Any:
        assert root == tmp_path and directory == tmp_path / "plans/parent" and expected_plan_sha256 == "f" * 64
        calls.append("lease_enter")
        yield {"requirements": {"in_window_decisions": 2}}
        calls.append("lease_exit")

    frame = pd.DataFrame({"decision_id": ["corrected-a", "corrected-b"], "parent_decision_id": ["parent-a", "parent-b"],
        "security_id": ["fixture:0", "fixture:1"], "ticker": ["AAA", "BBB"], "parent_ticker": ["AAA", "BBB"],
        "session_date_et": [date(2024, 1, 2)] * 2})

    def project(root: Path, source: dict[str, Any], record: dict[str, Any], policy: Any) -> pd.DataFrame:
        assert calls[-1] != "lease_exit" and record["partition_month"] == "2024-01"
        calls.append("metadata_projected")
        return frame.copy()

    monkeypatch.setattr(corrected_decisions, "verified_initial_fit_raw_share_plan", verified_plan)
    monkeypatch.setattr(corrected_decisions, "pinned_object", lambda path, expected=None: objects[path.relative_to(tmp_path).as_posix()])
    monkeypatch.setattr(corrected_decisions, "load_symbol_correction_policy", lambda *args: mapping)
    monkeypatch.setattr(corrected_decisions, "load_official_document_inventory", lambda path: object())
    monkeypatch.setattr(corrected_decisions, "verify_official_document_collection", lambda *args: documents)
    monkeypatch.setattr(corrected_decisions, "file_sha256", lambda path: "f" * 64)
    monkeypatch.setattr(corrected_decisions, "_metadata_implementation", lambda root: {"semantic.py": "f" * 64})
    monkeypatch.setattr(corrected_decisions, "_check", lambda *args: calls.append("source_recheck"))
    monkeypatch.setattr(corrected_decisions, "_guard", lambda: None)
    monkeypatch.setattr(corrected_decisions, "load_corrected_decision_partition", project)
    monkeypatch.setattr(pd, "read_parquet", lambda *args, **kwargs: pytest.fail("no price/full-panel read permitted"))
    return {"root": tmp_path, "config": config, "expected_config_sha256": pin, "calls": calls}


def test_public_projection_holds_lease_preserves_parents_and_excludes_heldout(metadata: dict[str, Any]) -> None:
    arguments = {key: metadata[key] for key in ("root", "config", "expected_config_sha256")}
    with corrected_decisions.verified_corrected_decision_partitions(**arguments) as projection:
        assert projection.expected_rows == 2 and len(projection.retained_security_ids) == 586
        assert projection.cohort_sha256 == "c" * 64 and "semantic.py" in projection.source_files
        frames = list(projection.partitions)
        assert len(frames) == 1 and frames[0][0] == "2024-01"
        assert frames[0][1].parent_decision_id.tolist() == ["parent-a", "parent-b"]
        assert "lease_exit" not in metadata["calls"]
    assert metadata["calls"][-2:] == ["source_recheck", "lease_exit"]


def test_partial_metadata_consumption_cannot_claim_complete_population(metadata: dict[str, Any]) -> None:
    arguments = {key: metadata[key] for key in ("root", "config", "expected_config_sha256")}
    with pytest.raises(DataReadinessError, match="complete frozen population"):
        with corrected_decisions.verified_corrected_decision_partitions(**arguments) as projection:
            next(projection.partitions)


def test_exported_source_pins_are_read_only(metadata: dict[str, Any]) -> None:
    arguments = {key: metadata[key] for key in ("root", "config", "expected_config_sha256")}
    with corrected_decisions.verified_corrected_decision_partitions(**arguments) as projection:
        with pytest.raises(TypeError):
            projection.source_files["semantic.py"] = "0" * 64  # type: ignore[index]
        list(projection.partitions)
