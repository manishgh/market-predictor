from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from datetime import date
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

import market_predictor.edge_rebuild.sp500_memberships as membership_module
from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.swing_history_acquisition import (
    AUTHORITY_SCHEMA,
    DAILY_BAR_UNITS_FILE,
    PLAN_SCHEMA,
    publish_swing_history_acquisition_plan,
)
from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.universe import VerifiedIndexChanges


def test_verified_membership_authority_publishes_stock_and_benchmark_units(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)

    manifest = publish_swing_history_acquisition_plan(
        repository_root=tmp_path,
        output_directory=Path("out"),
        **inputs,
    )

    assert manifest["schema"] == PLAN_SCHEMA
    assert manifest["status"] == "ready_for_daily_history_collection"
    assert manifest["outcomes_read"] is False
    assert manifest["membership"]["authority_start"] == "2018-05-29"
    assert manifest["membership"]["authority_cutoff"] == "2026-07-08"
    assert manifest["membership"]["benchmark_session_exclusions"] == 0
    assert len(manifest["membership"]["universe_sha256"]) == 64
    assert manifest["daily_bars"]["status"] == "ready"
    assert manifest["daily_bars"]["stock_units"] == 500
    assert manifest["daily_bars"]["benchmark_units"] == 3
    units_path = tmp_path / "out" / DAILY_BAR_UNITS_FILE
    units = pd.read_csv(units_path, dtype=str)
    assert len(units) == manifest["daily_bars"]["planned_units"]
    assert set(units.loc[units["role"].eq("benchmark"), "ticker"]) == {
        "QQQ",
        "SPY",
        "XLK",
    }
    spy = units.loc[units["ticker"].eq("SPY")].iloc[0]
    assert spy["start_date"] == "2018-05-29"
    assert spy["end_date"] == "2019-07-08"
    assert manifest["daily_bars"]["units_artifact"]["sha256"] == file_sha256(units_path)
    request = _read_json(tmp_path / "out" / "_request.json")
    lineage = request["membership_authority"]
    assert lineage["parent_lineage"] == manifest["membership"]["parent_lineage"]
    assert lineage["universe_sha256"] == manifest["membership"]["universe_sha256"]
    authority = _read_json(tmp_path / "out" / "_authority.json")
    assert authority["schema"] == AUTHORITY_SCHEMA
    assert authority["artifact_sha256"] == file_sha256(tmp_path / "out" / "_manifest.json")


def test_parent_lineage_tampering_fails_closed(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    _write_json(
        tmp_path / "raw" / "_authority.json",
        {"artifact_sha256": "f" * 64},
    )

    with pytest.raises(DataReadinessError, match="request identity"):
        publish_swing_history_acquisition_plan(
            repository_root=tmp_path,
            output_directory=Path("out"),
            **inputs,
        )


def test_authority_start_after_missing_window_fails(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(
        tmp_path,
        monkeypatch,
        authority_start=date(2019, 1, 2),
    )

    with pytest.raises(DataReadinessError, match="does not cover the missing-history start"):
        publish_swing_history_acquisition_plan(
            repository_root=tmp_path,
            output_directory=Path("out"),
            **inputs,
        )


def test_output_must_be_new(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    inputs = _fixture(tmp_path, monkeypatch)
    (tmp_path / "out").mkdir()

    with pytest.raises(DataReadinessError, match="output must be new"):
        publish_swing_history_acquisition_plan(
            repository_root=tmp_path,
            output_directory=Path("out"),
            **inputs,
        )


def _fixture(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    *,
    authority_start: date = date(2018, 5, 29),
) -> dict[str, Path]:
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

    raw = tmp_path / "raw"
    events = tmp_path / "events"
    transitions = tmp_path / "transitions"
    for directory, payload in (
        (raw, {"artifact_sha256": "a" * 64}),
        (events, {"event_set_sha256": "b" * 64}),
        (transitions, {"transition_set_sha256": "c" * 64}),
    ):
        directory.mkdir()
        _write_json(directory / "_authority.json", payload)
    reviewed = tmp_path / "reviewed.csv"
    reviewed.write_text("bound input\n", encoding="utf-8")
    anchor = tmp_path / "anchor.csv"
    _anchor(500).to_csv(anchor, index=False)
    empty_transitions = _empty_transitions()
    verified_events = VerifiedIndexChanges(
        changes=(),
        authority_sha256="d" * 64,
        event_set_sha256="e" * 64,
    )
    monkeypatch.setattr(
        membership_module,
        "require_sp500_transition_authority",
        lambda *_, **__: empty_transitions,
    )
    monkeypatch.setattr(
        membership_module,
        "require_spglobal_event_reconstruction_ready",
        lambda *_, **__: verified_events,
    )
    membership_authority = tmp_path / "membership_authority"
    membership_module.publish_sp500_membership_authority(
        archive_directory=raw,
        event_directory=events,
        transition_directory=transitions,
        reviewed_transitions_path=reviewed,
        anchor_path=anchor,
        start_date=authority_start,
        cutoff_date=date(2026, 7, 8),
        output_directory=membership_authority,
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
        "symbols": ["SPY", "QQQ", "XLK"],
    }
    request_hash = hashlib.sha256(json.dumps(request_payload, sort_keys=True).encode("utf-8")).hexdigest()
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
        "membership_authority_directory": Path("membership_authority"),
        "raw_archive_directory": Path("raw"),
        "event_authority_directory": Path("events"),
        "transition_authority_directory": Path("transitions"),
        "reviewed_transitions_path": Path("reviewed.csv"),
        "anchor_path": Path("anchor.csv"),
        "current_daily_collection_directory": Path("daily"),
    }


def _anchor(count: int) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": [f"T{number:03d}" for number in range(count)],
            "company": [f"Company {number}" for number in range(count)],
            "sector": ["Information Technology"] * count,
            "industry": ["Software"] * count,
            "cik": [f"{number + 1:010d}" for number in range(count)],
        }
    )


def _empty_transitions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "transition_id": pd.Series(dtype="string"),
            "effective_at_utc": pd.Series(dtype="datetime64[ns, UTC]"),
            "old_ticker": pd.Series(dtype="string"),
            "new_ticker": pd.Series(dtype="string"),
            "identity_continuity": pd.Series(dtype="bool"),
            "membership_continuity": pd.Series(dtype="bool"),
            "old_security_id": pd.Series(dtype="string"),
            "new_security_id": pd.Series(dtype="string"),
            "source_url": pd.Series(dtype="string"),
        }
    )


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(payload, dict)
    return payload


def _write_json(path: Path, payload: Mapping[str, object]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
