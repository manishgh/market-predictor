from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256, manifest_path_for
from market_predictor.edge_rebuild.swing_history_acquisition import (
    AUTHORITY_SCHEMA,
    PLAN_SCHEMA,
    publish_swing_history_acquisition_plan,
)
from market_predictor.v3.errors import DataReadinessError


def test_plan_blocks_bar_units_until_membership_is_extended(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, membership_start="2019-07-09")

    manifest = publish_swing_history_acquisition_plan(
        repository_root=tmp_path,
        output_directory=Path("out"),
        **inputs,
    )

    assert manifest["schema"] == PLAN_SCHEMA
    assert manifest["status"] == "official_source_archive_authority_required"
    assert manifest["outcomes_read"] is False
    assert manifest["daily_bars"]["planned_units"] == 0
    assert manifest["daily_bars"]["status"] == "blocked_until_archive_authority"
    assert not (tmp_path / "out" / "daily_bar_units.csv").exists()
    assert manifest["membership"]["required_start"] == "2018-05-29"
    assert manifest["membership"]["reusable_official_sources"] == 1
    authority = _read_json(tmp_path / "out" / "_authority.json")
    assert authority["schema"] == AUTHORITY_SCHEMA
    assert authority["artifact_sha256"] == file_sha256(
        tmp_path / "out" / "_manifest.json"
    )


def test_extended_membership_cannot_bypass_missing_archive_authority(
    tmp_path: Path,
) -> None:
    inputs = _fixture(tmp_path, membership_start="2018-05-29")

    manifest = publish_swing_history_acquisition_plan(
        repository_root=tmp_path,
        output_directory=Path("out"),
        **inputs,
    )

    assert manifest["status"] == "official_source_archive_authority_required"
    assert manifest["membership"]["membership_dates_cover_required_start"] is True
    assert manifest["daily_bars"]["planned_units"] == 0
    assert manifest["daily_bars"]["status"] == "blocked_until_archive_authority"
    assert not (tmp_path / "out" / "daily_bar_units.csv").exists()


def test_official_source_tampering_refuses_bar_units(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, membership_start="2019-07-09")
    (tmp_path / "evidence" / "release.html").write_text(
        "tampered",
        encoding="utf-8",
    )

    manifest = publish_swing_history_acquisition_plan(
        repository_root=tmp_path,
        output_directory=Path("out"),
        **inputs,
    )

    assert manifest["status"] == "official_source_reacquisition_required"
    assert manifest["daily_bars"]["planned_units"] == 0
    assert (
        manifest["daily_bars"]["status"]
        == "blocked_until_source_reacquisition"
    )
    assert "SHA-256" in manifest["daily_bars"]["refusal_reason"]
    assert manifest["membership"]["reusable_official_sources"] == 0
    assert manifest["membership"]["invalid_official_sources"] == 1


def test_output_must_be_new(tmp_path: Path) -> None:
    inputs = _fixture(tmp_path, membership_start="2019-07-09")
    (tmp_path / "out").mkdir()

    with pytest.raises(DataReadinessError, match="output must be new"):
        publish_swing_history_acquisition_plan(
            repository_root=tmp_path,
            output_directory=Path("out"),
            **inputs,
        )


def _fixture(tmp_path: Path, *, membership_start: str) -> dict[str, Path]:
    temporal = tmp_path / "temporal"
    temporal.mkdir()
    temporal_manifest = {
        "schema": "edge_rebuild.temporal_manifest.v1",
        "status": "insufficient_history",
        "coverage": {
            "outcomes_read": False,
            "target_sessions_missing": 279,
            "missing_ranges": [
                {
                    "first_session": "2018-05-29",
                    "last_session": "2019-07-08",
                    "sessions": 279,
                }
            ],
        },
    }
    _write_json(temporal / "_manifest.json", temporal_manifest)
    _write_json(
        temporal / "_authority.json",
        {
            "schema": "edge_rebuild.temporal_manifest_authority.v1",
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(temporal / "_manifest.json"),
        },
    )

    memberships = tmp_path / "memberships.parquet"
    frame = pd.DataFrame(
        {
            "security_id": ["security:AAA"],
            "ticker": ["AAA"],
            "effective_from_utc": [
                pd.Timestamp(f"{membership_start}T04:00:00Z")
            ],
            "effective_to_utc": [pd.NaT],
            "primary_benchmark": ["XLK"],
        }
    )
    frame.to_parquet(memberships, index=False)
    _write_json(
        manifest_path_for(memberships),
        {
            "schema": "market_data.artifact_manifest.v1",
            "artifact_type": "memberships",
            "artifact_sha256": file_sha256(memberships),
            "rows": len(frame),
        },
    )

    evidence = tmp_path / "evidence"
    evidence.mkdir()
    release = evidence / "release.html"
    release.write_text("official", encoding="utf-8")
    anchor = evidence / "anchor.csv"
    anchor.write_text("ticker\nAAA\n", encoding="utf-8")
    transitions = evidence / "transitions.parquet"
    pd.DataFrame({"id": ["transition-1"]}).to_parquet(transitions, index=False)
    review = evidence / "review.csv"
    review.write_text("id\n", encoding="utf-8")
    universe_audit = tmp_path / "universe_audit.json"
    _write_json(
        universe_audit,
        {
            "schema": "ml_v3.sp500_point_in_time_universe.v1",
            "start_date": membership_start,
            "cutoff_date": "2026-07-08",
            "anchor_source": "evidence/anchor.csv",
            "snapshot_sha256": file_sha256(anchor),
            "source_manifest": {
                "schema": "ml_v3.sp500_change_sources.v1",
                "sources": [
                    {
                        "raw_path": "evidence/release.html",
                        "sha256": file_sha256(release),
                    }
                ],
            },
            "security_transition_evidence": {
                "provider_path": "evidence/transitions.parquet",
                "provider_sha256": file_sha256(transitions),
                "reviewed_path": "evidence/review.csv",
                "reviewed_sha256": file_sha256(review),
            },
        },
    )

    daily = tmp_path / "daily"
    daily.mkdir()
    ledger = daily / "ledger.parquet"
    pd.DataFrame({"collection_id": ["daily-1"]}).to_parquet(ledger, index=False)
    request_payload = {
        "schema": "swing.daily_history_collection.v1",
        "start_date": "2019-07-09",
        "end_date": "2026-07-08",
        "source": "alpaca",
        "price_feed": "sip",
        "adjustment": "all",
        "timeframe": "1d",
        "symbols": ["AAA", "SPY", "QQQ", "XLK"],
    }
    request_hash = hashlib.sha256(
        json.dumps(request_payload, sort_keys=True).encode("utf-8")
    ).hexdigest()
    _write_json(daily / "_request.json", {**request_payload, "request_sha256": request_hash})
    _write_json(
        daily / "_status.json",
        {
            "schema": "swing.daily_history_manifest.v1",
            "status": "complete",
            "request_sha256": request_hash,
            "source_collections_path": "daily/ledger.parquet",
            "source_collections_sha256": file_sha256(ledger),
        },
    )
    _write_json(
        daily / "_manifest.json",
        {
            "schema": "swing.daily_history_manifest.v1",
            "status": "complete",
            "request_sha256": request_hash,
            "total_rows": 1_000,
        },
    )
    return {
        "temporal_manifest_directory": Path("temporal"),
        "memberships_path": Path("memberships.parquet"),
        "universe_audit_path": Path("universe_audit.json"),
        "current_daily_collection_directory": Path("daily"),
    }


def _read_json(path: Path) -> dict[str, object]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
