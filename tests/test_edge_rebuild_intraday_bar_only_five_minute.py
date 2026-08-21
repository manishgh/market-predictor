from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import exchange_calendars as xcals
import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.intraday_bar_only_five_minute import (
    PROJECTION_AUTHORITY_SCHEMA,
    load_complete_selected_session_five_minute_projection,
    publish_selected_session_five_minute_projection,
)
from market_predictor.edge_rebuild.intraday_selection import (
    INTRADAY_SELECTION_SCHEMA,
    IntradaySelectionResult,
    publish_intraday_selection,
)
from market_predictor.edge_rebuild.strategy_contract import load_strategy_contract
from market_predictor.core.errors import DataReadinessError

CONTRACT = Path("configs/edge_rebuild_strategy_contract.toml")
SESSIONS = ("2024-01-03", "2024-02-01", "2024-02-02")


def test_publishes_monthly_complete_incomplete_and_missing_projection(
    tmp_path: Path,
) -> None:
    selection, canonical = _inputs(tmp_path)
    output = tmp_path / "projection"

    manifest = publish_selected_session_five_minute_projection(
        selection_directory=selection,
        five_minute_canonical_directory=canonical,
        strategy_contract_path=CONTRACT,
        output_directory=output,
    )

    assert manifest["provider_download_performed"] is False
    assert manifest["selected_stock_sessions"] == 3
    assert manifest["coverage_status_counts"] == {
        "complete": 1,
        "incomplete": 1,
        "missing": 1,
    }
    coverage_records = [
        record for record in manifest["files"] if record["role"] == "coverage"
    ]
    bar_records = [record for record in manifest["files"] if record["role"] == "bars"]
    assert [record["month"] for record in coverage_records] == ["2024-01", "2024-02"]
    assert [record["month"] for record in bar_records] == ["2024-01", "2024-02"]
    coverage = pd.concat(
        [pd.read_parquet(output / str(record["path"])) for record in coverage_records],
        ignore_index=True,
    )
    statuses = coverage.set_index(["ticker", "session_date_et"])["coverage_status"]
    assert statuses[("AAA", "2024-01-03")] == "complete"
    assert statuses[("AAA", "2024-02-01")] == "incomplete"
    assert statuses[("BBB", "2024-02-02")] == "missing"
    bars = pd.concat(
        [pd.read_parquet(output / str(record["path"])) for record in bar_records],
        ignore_index=True,
    )
    assert len(bars) == 155
    assert set(bars["session_date_et"].astype(str)) == {"2024-01-03", "2024-02-01"}
    assert "2024-02-05" not in set(bars["session_date_et"].astype(str))
    assert load_complete_selected_session_five_minute_projection(output) == manifest


def test_replay_rejects_source_and_projection_authority_tampering(
    tmp_path: Path,
) -> None:
    selection, canonical = _inputs(tmp_path)
    output = tmp_path / "projection"
    publish_selected_session_five_minute_projection(
        selection_directory=selection,
        five_minute_canonical_directory=canonical,
        strategy_contract_path=CONTRACT,
        output_directory=output,
    )
    authority_path = output / "_authority.json"
    authority = _read_json(authority_path)
    authority["schema"] = "tampered"
    _write_json(authority_path, authority)
    with pytest.raises(DataReadinessError, match="authority or source lineage"):
        load_complete_selected_session_five_minute_projection(output)

    selection_two, canonical_two = _inputs(tmp_path / "source_tamper")
    output_two = tmp_path / "projection_source_tamper"
    publish_selected_session_five_minute_projection(
        selection_directory=selection_two,
        five_minute_canonical_directory=canonical_two,
        strategy_contract_path=CONTRACT,
        output_directory=output_two,
    )
    source_path = canonical_two / "regular" / "5m" / "AAA.parquet"
    source = pd.read_parquet(source_path)
    source.loc[0, "close"] = float(source.loc[0, "close"]) + 1.0
    source.to_parquet(source_path, index=False)
    with pytest.raises(DataReadinessError, match="canonical 5m file failed hash"):
        load_complete_selected_session_five_minute_projection(output_two)


def test_future_rows_are_excluded_and_selected_out_of_session_rows_fail(
    tmp_path: Path,
) -> None:
    selection, canonical = _inputs(tmp_path, add_out_of_session=True)
    with pytest.raises(DataReadinessError, match="out-of-session"):
        publish_selected_session_five_minute_projection(
            selection_directory=selection,
            five_minute_canonical_directory=canonical,
            strategy_contract_path=CONTRACT,
            output_directory=tmp_path / "projection",
        )


def test_rejects_output_path_overlap(tmp_path: Path) -> None:
    selection, canonical = _inputs(tmp_path)
    with pytest.raises(DataReadinessError, match="overlaps an input path"):
        publish_selected_session_five_minute_projection(
            selection_directory=selection,
            five_minute_canonical_directory=canonical,
            strategy_contract_path=CONTRACT,
            output_directory=canonical / "projection",
        )


def test_rejects_selection_bound_to_another_canonical_store(tmp_path: Path) -> None:
    selection, canonical = _inputs(tmp_path)
    manifest_path = selection / "_manifest.json"
    manifest = _read_json(manifest_path)
    manifest["canonical_dir"] = str(tmp_path / "different_canonical")
    _write_json(manifest_path, manifest)
    authority_path = selection / "_authority.json"
    authority = _read_json(authority_path)
    authority["artifact_sha256"] = file_sha256(manifest_path)
    _write_json(authority_path, authority)
    with pytest.raises(DataReadinessError, match="lineage differ"):
        publish_selected_session_five_minute_projection(
            selection_directory=selection,
            five_minute_canonical_directory=canonical,
            strategy_contract_path=CONTRACT,
            output_directory=tmp_path / "projection",
        )


def test_replay_requires_exact_on_disk_inventory(tmp_path: Path) -> None:
    selection, canonical = _inputs(tmp_path)
    output = tmp_path / "projection"
    manifest = publish_selected_session_five_minute_projection(
        selection_directory=selection,
        five_minute_canonical_directory=canonical,
        strategy_contract_path=CONTRACT,
        output_directory=output,
    )
    declared = {
        str(record["path"]) for record in manifest["files"]
    } | {"_manifest.json", "_authority.json"}
    actual = {
        str(path.relative_to(output)).replace("\\", "/")
        for path in output.rglob("*")
        if path.is_file()
    }
    assert actual == declared
    (output / "undeclared.parquet").write_bytes(b"not parquet")
    with pytest.raises(DataReadinessError, match="on-disk inventory is not exact"):
        load_complete_selected_session_five_minute_projection(output)


def test_authority_is_written_last_and_is_hash_bound(tmp_path: Path) -> None:
    selection, canonical = _inputs(tmp_path)
    output = tmp_path / "projection"
    publish_selected_session_five_minute_projection(
        selection_directory=selection,
        five_minute_canonical_directory=canonical,
        strategy_contract_path=CONTRACT,
        output_directory=output,
    )
    authority = _read_json(output / "_authority.json")
    assert authority["schema"] == PROJECTION_AUTHORITY_SCHEMA
    assert authority["state"] == "complete"
    assert authority["artifact_sha256"] == file_sha256(output / "_manifest.json")


def _inputs(
    root: Path,
    *,
    add_out_of_session: bool = False,
) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    contract = load_strategy_contract(CONTRACT)
    selection_directory = root / "selection"
    selection = pd.DataFrame(
        {
            "ticker": ["AAA", "AAA", "BBB"],
            "session_date_et": list(SESSIONS),
            "activation_time_utc": [
                "2024-01-03T15:01:00Z",
                "2024-02-01T15:01:00Z",
                "2024-02-02T15:01:00Z",
            ],
            "activation_rank": [1, 1, 1],
            "relative_volume_at_activation": [2.1, 2.2, 2.3],
            "average_volume_prior_sessions": [2_000_000.0] * 3,
            "median_volume_prior_sessions": [1_900_000.0] * 3,
            "price_at_activation": [25.0, 26.0, 27.0],
        }
    )
    audit: dict[str, Any] = {
        "schema": INTRADAY_SELECTION_SCHEMA,
        "strategy_id": contract.intraday.strategy_id,
        "strategy_contract_sha256": contract.sha256(),
        "canonical_dir": str(root / "canonical"),
        "membership_authority_dir": "fixture",
        "membership_authority_sha256": "a" * 64,
        "membership_manifest_sha256": "b" * 64,
        "membership_table_sha256": "c" * 64,
        "membership_universe_sha256": "d" * 64,
        "membership_universe_snapshot_id": "fixture",
        "membership_parent_lineage": {},
        "membership_cold_start_policy": "fixture",
        "first_session_et": SESSIONS[0],
        "last_session_et": SESSIONS[-1],
        "excluded_tickers": [],
    }
    publish_intraday_selection(
        IntradaySelectionResult(
            liquidity=pd.DataFrame({"ticker": ["AAA"]}),
            selection=selection,
            audit=audit,
        ),
        output_directory=selection_directory,
    )
    canonical_directory = root / "canonical"
    source = pd.concat(
        [
            _session_bars("AAA", "2024-01-03"),
            _session_bars("AAA", "2024-02-01").iloc[:-1],
            _session_bars("AAA", "2024-02-05"),
        ],
        ignore_index=True,
    )
    if add_out_of_session:
        invalid = _session_bars("AAA", "2024-02-01").iloc[[-1]].copy()
        invalid["bar_start_utc"] = invalid["bar_start_utc"] + pd.Timedelta(minutes=5)
        invalid["bar_end_utc"] = invalid["bar_end_utc"] + pd.Timedelta(minutes=5)
        invalid["available_at_utc"] = invalid["available_at_utc"] + pd.Timedelta(minutes=5)
        invalid["ingested_at_utc"] = invalid["ingested_at_utc"] + pd.Timedelta(minutes=5)
        source = pd.concat([source, invalid], ignore_index=True)
    _publish_canonical_store(canonical_directory, source)
    return selection_directory, canonical_directory


def _session_bars(ticker: str, session: str) -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    label = pd.Timestamp(session)
    open_at = pd.Timestamp(calendar.session_open(label)).tz_convert("UTC")
    close_at = pd.Timestamp(calendar.session_close(label)).tz_convert("UTC")
    starts = pd.date_range(open_at, close_at, freq="5min", inclusive="left")
    offsets = pd.Series(range(len(starts)), dtype="float64")
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": session,
            "session_segment": "regular",
            "history_era": "collected",
            "timeframe": "5m",
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=5),
            "available_at_utc": starts + pd.Timedelta(minutes=6),
            "ingested_at_utc": starts + pd.Timedelta(minutes=7),
            "open": 20.0 + offsets,
            "high": 20.5 + offsets,
            "low": 19.5 + offsets,
            "close": 20.25 + offsets,
            "volume": 1000 + offsets.astype("int64"),
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _publish_canonical_store(directory: Path, frame: pd.DataFrame) -> None:
    path = directory / "regular" / "5m" / "AAA.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(path, index=False)
    record = {
        "path": "regular/5m/AAA.parquet",
        "rows": len(frame),
        "sha256": file_sha256(path),
        "store": "regular",
        "ticker": "AAA",
    }
    manifest = {
        "schema": "edge_rebuild.intraday_materialization.v1",
        "total_rows": len(frame),
        "integrity": {
            "blocking_defect_count": 0,
            "identity_breaks": [],
            "fabricated_bars": [],
        },
        "files": [record],
    }
    _write_json(directory / "_manifest.json", manifest)
    _write_json(
        directory / "_authority.json",
        {
            "schema": "edge_rebuild.intraday_materialization_authority.v1",
            "state": "complete",
            "artifact": "_manifest.json",
            "artifact_sha256": file_sha256(directory / "_manifest.json"),
        },
    )


def _read_json(path: Path) -> dict[str, Any]:
    loaded = json.loads(path.read_text(encoding="utf-8"))
    assert isinstance(loaded, dict)
    return loaded


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
