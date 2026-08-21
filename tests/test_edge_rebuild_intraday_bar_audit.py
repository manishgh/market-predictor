from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild import intraday_bar_audit as module
from market_predictor.edge_rebuild.intraday_bar_audit import (
    publish_intraday_bar_dataset_audit,
)
from market_predictor.core.errors import DataReadinessError


def test_audit_is_reproducible_and_bound_to_both_authorities(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    dataset = tmp_path / "dataset"
    projection = tmp_path / "projection"
    report_path = tmp_path / "reports" / "audit.json"
    session = "2024-01-03"
    unit = dataset / "sessions" / f"session_date_et={session}"
    unit.mkdir(parents=True)
    projection.mkdir()
    rows = _eligible_rows(session)
    rows.to_parquet(unit / "rows.parquet", index=False)
    (dataset / "_manifest.json").write_text("{}", encoding="utf-8")
    (dataset / "_authority.json").write_text("{}", encoding="utf-8")
    coverage_path = projection / "coverage.parquet"
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date_et": [session],
            "coverage_status": ["incomplete"],
        }
    ).to_parquet(coverage_path, index=False)
    (projection / "_manifest.json").write_text("{}", encoding="utf-8")
    (projection / "_authority.json").write_text("{}", encoding="utf-8")
    dataset_manifest = {
        "request_sha256": "1" * 64,
        "transformation_sha256": "2" * 64,
        "session_unit_inventory_sha256": "3" * 64,
        "planned_sessions": [session],
        "training_contract": {
            "ordered_feature_names": ["volume_return_1_bar"],
            "ordered_feature_sha256": "4" * 64,
        },
        "summary": {"rows": 1, "dataset_eligible_rows": 1},
    }
    projection_manifest = {
        "file_inventory_sha256": "5" * 64,
        "files": [
            {
                "role": "coverage",
                "path": coverage_path.name,
                "sha256": file_sha256(coverage_path),
            }
        ],
    }
    monkeypatch.setattr(
        module,
        "load_complete_intraday_bar_dataset",
        lambda _path: dataset_manifest,
    )
    monkeypatch.setattr(
        module,
        "load_complete_selected_session_five_minute_projection",
        lambda _path: projection_manifest,
    )

    first = publish_intraday_bar_dataset_audit(
        dataset_directory=dataset,
        five_minute_projection_directory=projection,
        output_path=report_path,
    )
    second = publish_intraday_bar_dataset_audit(
        dataset_directory=dataset,
        five_minute_projection_directory=projection,
        output_path=report_path,
    )

    assert first == second
    assert first["status"] == "pass"
    assert first["dataset_manifest_sha256"] == file_sha256(
        dataset / "_manifest.json"
    )
    assert first["dataset_authority_sha256"] == file_sha256(
        dataset / "_authority.json"
    )
    assert first["session_unit_inventory_sha256"] == "3" * 64
    assert first["five_minute_projection_manifest_sha256"] == file_sha256(
        projection / "_manifest.json"
    )
    assert json.loads(report_path.read_text(encoding="utf-8")) == first


def test_audit_output_cannot_modify_an_authority(tmp_path: Path) -> None:
    dataset = tmp_path / "dataset"
    projection = tmp_path / "projection"
    dataset.mkdir()
    projection.mkdir()

    with pytest.raises(DataReadinessError, match="overlaps an authority"):
        publish_intraday_bar_dataset_audit(
            dataset_directory=dataset,
            five_minute_projection_directory=projection,
            output_path=dataset / "audit.json",
        )


def _eligible_rows(session: str) -> pd.DataFrame:
    decision = pd.Timestamp(f"{session}T15:00:00Z")
    exit_end = decision + pd.Timedelta(minutes=30)
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date_et": [session],
            "decision_time_utc": [decision],
            "feature_available_at_utc": [decision],
            "label_available_at_utc": [exit_end],
            "exit_bar_end_utc": [exit_end],
            "feature_eligible": [True],
            "label_eligible": [True],
            "dataset_eligible": [True],
            "atr_14_5m": [1.0],
            "five_minute_bar_observed": [True],
            "ordered_feature_sha256": ["4" * 64],
            "feature_schema_version": ["edge_rebuild.intraday_bar_features.v1"],
            "label_schema_version": ["edge_rebuild.intraday_bar_labels.v1"],
        }
    )
