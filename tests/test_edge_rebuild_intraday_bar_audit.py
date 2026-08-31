from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

import market_predictor.intraday.datasets.bar_audit as module
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.bar_audit import (
    INTRADAY_BAR_DATASET_AUDIT_SCHEMA,
    publish_intraday_bar_dataset_audit,
)


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
    dataset_manifest["five_minute_projection_directory"] = str(projection.resolve())
    dataset_manifest["parent_lineage"] = {
        "five_minute_projection_authority_sha256": file_sha256(projection / "_authority.json"),
        "five_minute_projection_manifest_sha256": file_sha256(projection / "_manifest.json"),
        "five_minute_projection_inventory_sha256": projection_manifest["file_inventory_sha256"],
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
    assert first["schema"] == INTRADAY_BAR_DATASET_AUDIT_SCHEMA
    assert first["status"] == "pass"
    assert first["five_minute_projection_lineage_verified"] is True
    assert first["source_feature_cutoff_violations"] == 0
    assert first["incomplete_five_minute_prefix_feature_eligible_violations"] == 0
    assert first["dataset_manifest_sha256"] == file_sha256(dataset / "_manifest.json")
    assert first["dataset_authority_sha256"] == file_sha256(dataset / "_authority.json")
    assert first["session_unit_inventory_sha256"] == "3" * 64
    assert first["five_minute_projection_manifest_sha256"] == file_sha256(projection / "_manifest.json")
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


@pytest.mark.parametrize(
    "changed_key",
    [
        "five_minute_projection_authority_sha256",
        "five_minute_projection_manifest_sha256",
        "five_minute_projection_inventory_sha256",
    ],
)
def test_audit_rejects_projection_lineage_mismatch(
    tmp_path: Path,
    changed_key: str,
) -> None:
    projection = tmp_path / "projection"
    projection.mkdir()
    (projection / "_manifest.json").write_text("{}", encoding="utf-8")
    (projection / "_authority.json").write_text("{}", encoding="utf-8")
    projection_manifest = {"file_inventory_sha256": "3" * 64}
    parent = {
        "five_minute_projection_authority_sha256": file_sha256(projection / "_authority.json"),
        "five_minute_projection_manifest_sha256": file_sha256(projection / "_manifest.json"),
        "five_minute_projection_inventory_sha256": "3" * 64,
    }
    parent[changed_key] = "f" * 64

    with pytest.raises(DataReadinessError, match="projection lineage differ"):
        module._require_projection_binding(
            dataset_manifest={
                "five_minute_projection_directory": str(projection.resolve()),
                "parent_lineage": parent,
            },
            projection_manifest=projection_manifest,
            projection_directory=projection,
        )


def test_audit_rejects_projection_path_mismatch(tmp_path: Path) -> None:
    projection = tmp_path / "projection"
    other = tmp_path / "other"
    projection.mkdir()
    (projection / "_manifest.json").write_text("{}", encoding="utf-8")
    (projection / "_authority.json").write_text("{}", encoding="utf-8")

    with pytest.raises(DataReadinessError, match="projection path differ"):
        module._require_projection_binding(
            dataset_manifest={
                "five_minute_projection_directory": str(other.resolve()),
                "parent_lineage": {},
            },
            projection_manifest={"file_inventory_sha256": "3" * 64},
            projection_directory=projection,
        )


@pytest.mark.parametrize(
    ("source_timestamp", "prefix_complete", "feature_eligible", "expected_source", "expected_prefix"),
    [
        ("equal", True, True, 0, 0),
        ("late", True, True, 1, 0),
        ("missing", True, True, 1, 0),
        ("equal", False, True, 0, 1),
        ("late", False, False, 0, 0),
    ],
)
def test_audit_checks_raw_source_cutoff_and_five_minute_prefix(
    source_timestamp: str,
    prefix_complete: bool,
    feature_eligible: bool,
    expected_source: int,
    expected_prefix: int,
) -> None:
    frame = _eligible_rows("2024-01-03")
    decision = frame.loc[0, "decision_time_utc"]
    if source_timestamp == "late":
        frame.loc[0, "source_feature_available_at_utc"] = decision + pd.Timedelta(seconds=1)
    elif source_timestamp == "missing":
        frame.loc[0, "source_feature_available_at_utc"] = pd.NaT
    frame.loc[0, "five_minute_prefix_complete"] = prefix_complete
    frame.loc[0, "feature_eligible"] = feature_eligible
    frame.loc[0, "dataset_eligible"] = feature_eligible
    counters = module._empty_counters()

    module._audit_session(
        frame,
        session_date="2024-01-03",
        expected_feature_hash="4" * 64,
        counters=counters,
        tickers=set(),
        incomplete_stats={},
    )

    assert counters["source_feature_cutoff_violations"] == expected_source
    assert counters["incomplete_five_minute_prefix_feature_eligible_violations"] == expected_prefix


def test_existing_audit_report_remains_immutable(tmp_path: Path) -> None:
    report = tmp_path / "audit.json"
    report.write_text(
        json.dumps({"schema": "edge_rebuild.intraday_bar_dataset_audit.v1"}),
        encoding="utf-8",
    )

    with pytest.raises(DataReadinessError, match="immutable and differs"):
        module._publish_report(
            report,
            {"schema": INTRADAY_BAR_DATASET_AUDIT_SCHEMA},
        )


def _eligible_rows(session: str) -> pd.DataFrame:
    decision = pd.Timestamp(f"{session}T15:00:00Z")
    exit_end = decision + pd.Timedelta(minutes=30)
    return pd.DataFrame(
        {
            "ticker": ["AAA"],
            "session_date_et": [session],
            "decision_time_utc": [decision],
            "source_feature_available_at_utc": [decision],
            "feature_available_at_utc": [decision],
            "label_available_at_utc": [exit_end],
            "exit_bar_end_utc": [exit_end],
            "feature_eligible": [True],
            "label_eligible": [True],
            "dataset_eligible": [True],
            "atr_14_5m": [1.0],
            "five_minute_bar_observed": [True],
            "five_minute_prefix_complete": [True],
            "ordered_feature_sha256": ["4" * 64],
            "feature_schema_version": ["edge_rebuild.intraday_bar_features.v1"],
            "label_schema_version": ["edge_rebuild.intraday_bar_labels.v1"],
        }
    )
