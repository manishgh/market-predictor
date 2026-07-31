from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from market_predictor.canonical.store import file_sha256
from market_predictor.edge_rebuild.broad_intraday_history import (
    build_broad_intraday_history_plan,
    load_complete_broad_intraday_history_plan,
)
from market_predictor.edge_rebuild.history_contracts import (
    BroadIntradayHistoryConfig,
    load_broad_intraday_history_config,
    load_collection_transport_config,
)
from market_predictor.v3.errors import DataReadinessError

POLICY = Path("configs/edge_rebuild_broad_intraday_history.toml")
EARLY_CLOSE = "2024-07-03"
FULL_SESSION = "2024-07-05"


def test_plan_subtracts_coverage_excludes_funds_and_bounds_units(
    tmp_path: Path,
) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    output = tmp_path / "plan"

    manifest = build_broad_intraday_history_plan(
        broad_memberships_path=broad,
        pit_memberships_path=pit,
        existing_corpus_directory=corpus,
        policy_path=POLICY,
        output_directory=output,
        config=_config(),
    )
    verified = load_complete_broad_intraday_history_plan(output)
    units = pd.read_parquet(output / "units" / "5Min" / "2024-07.parquet")
    missing = pd.read_parquet(output / "missing_symbol_sessions" / "2024-07.parquet")

    assert verified == manifest
    assert manifest["research_only"] is True
    assert manifest["promotion_eligible"] is False
    assert manifest["historical_membership_authority"] is False
    assert "current_snapshot_proxy" in manifest["membership_limitations"]["broad_non_index"]
    assert manifest["acquisition"]["one_minute_paths_planned"] is False
    assert manifest["summary"]["existing_symbol_sessions_subtracted"] == 3
    assert not {"SPY", "QQQ", "FUND"}.intersection(missing["ticker"])
    assert not (
        missing["ticker"].eq("AAA") & missing["session_date_et"].isin([pd.Timestamp(EARLY_CLOSE).date(), pd.Timestamp(FULL_SESSION).date()])
    ).any()
    assert int(units["symbol_count"].max()) <= 50
    assert int(units["maximum_expected_rows"].max()) <= 10_000
    assert set(units["timeframe"]) == {"5Min"}
    early = units[units["session_date_et"].eq(pd.Timestamp(EARLY_CLOSE).date())]
    full = units[units["session_date_et"].eq(pd.Timestamp(FULL_SESSION).date())]
    assert set(early["expected_bars_per_symbol"]) == {42}
    assert set(full["expected_bars_per_symbol"]) == {78}
    assert set(early["requested_end_utc"]) == {pd.Timestamp("2024-07-03 17:00", tz="UTC")}
    assert set(full["requested_end_utc"]) == {pd.Timestamp("2024-07-05 20:00", tz="UTC")}


def test_current_snapshot_cannot_masquerade_as_point_in_time(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    frame = pd.read_parquet(broad)
    frame["source"] = "sp_global_primary_evidence"
    _publish_memberships(broad, frame)

    with pytest.raises(DataReadinessError, match="current-snapshot proxy"):
        build_broad_intraday_history_plan(
            broad_memberships_path=broad,
            pit_memberships_path=pit,
            existing_corpus_directory=corpus,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=_config(),
        )


@pytest.mark.parametrize(
    ("column", "poison"),
    [("price_feed", "iex"), ("timeframe", "1m")],
)
def test_plan_rejects_wrong_existing_feed_or_timeframe(
    tmp_path: Path,
    column: str,
    poison: str,
) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    path = corpus / "regular" / "5m" / "AAA.parquet"
    frame = pd.read_parquet(path)
    frame[column] = poison
    frame.to_parquet(path, index=False)
    _resign_corpus_file(corpus, ticker="AAA")

    with pytest.raises(DataReadinessError, match="regular 5m identity"):
        build_broad_intraday_history_plan(
            broad_memberships_path=broad,
            pit_memberships_path=pit,
            existing_corpus_directory=corpus,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=_config(),
        )


def test_plan_rejects_corrupt_existing_file_hash(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    path = corpus / "regular" / "5m" / "AAA.parquet"
    frame = pd.read_parquet(path)
    frame.loc[0, "source"] = "corrupt"
    frame.to_parquet(path, index=False)

    with pytest.raises(DataReadinessError, match="file does not verify"):
        build_broad_intraday_history_plan(
            broad_memberships_path=broad,
            pit_memberships_path=pit,
            existing_corpus_directory=corpus,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=_config(),
        )


def test_plan_rejects_manifest_row_count_mismatch(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    manifest_path = corpus / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    manifest["files"][0]["rows"] += 1
    manifest_path.write_text(
        json.dumps(manifest, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    _write_corpus_authority(corpus)

    with pytest.raises(DataReadinessError, match="row count differs"):
        build_broad_intraday_history_plan(
            broad_memberships_path=broad,
            pit_memberships_path=pit,
            existing_corpus_directory=corpus,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=_config(),
        )


def test_plan_rejects_regular_bar_outside_session(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    path = corpus / "regular" / "5m" / "AAA.parquet"
    frame = pd.read_parquet(path)
    frame.loc[frame["session_date_et"].eq(pd.Timestamp(FULL_SESSION).date()), "bar_start_utc"] = pd.Timestamp(
        "2024-07-05 12:00",
        tz="UTC",
    )
    frame.to_parquet(path, index=False)
    _resign_corpus_file(corpus, ticker="AAA")

    with pytest.raises(DataReadinessError, match="exceed XNYS bounds"):
        build_broad_intraday_history_plan(
            broad_memberships_path=broad,
            pit_memberships_path=pit,
            existing_corpus_directory=corpus,
            policy_path=POLICY,
            output_directory=tmp_path / "plan",
            config=_config(),
        )


def test_plan_replans_exact_missing_session(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    _publish_corpus(
        corpus,
        {
            "AAA": [EARLY_CLOSE],
            "PITCO": [EARLY_CLOSE],
        },
    )
    output = tmp_path / "plan"

    build_broad_intraday_history_plan(
        broad_memberships_path=broad,
        pit_memberships_path=pit,
        existing_corpus_directory=corpus,
        policy_path=POLICY,
        output_directory=output,
        config=_config(),
    )
    missing = pd.read_parquet(output / "missing_symbol_sessions" / "2024-07.parquet")
    aaa = missing.loc[missing["ticker"].eq("AAA"), "session_date_et"]

    assert set(aaa) == {pd.Timestamp(FULL_SESSION).date()}


def test_policy_and_generic_collector_registration_prohibit_one_minute() -> None:
    config = load_broad_intraday_history_config(POLICY)
    generic = load_collection_transport_config(POLICY)
    payload = config.model_dump(mode="python")
    payload["history_timeframe"] = "1Min"

    assert isinstance(generic, BroadIntradayHistoryConfig)
    assert generic.history_timeframe == "5Min"
    with pytest.raises(ValueError, match="five-minute"):
        BroadIntradayHistoryConfig.model_validate(payload)


def test_plan_loader_detects_modified_output(tmp_path: Path) -> None:
    broad, pit, corpus = _inputs(tmp_path)
    output = tmp_path / "plan"
    build_broad_intraday_history_plan(
        broad_memberships_path=broad,
        pit_memberships_path=pit,
        existing_corpus_directory=corpus,
        policy_path=POLICY,
        output_directory=output,
        config=_config(),
    )
    path = output / "missing_symbol_sessions" / "2024-07.parquet"
    frame = pd.read_parquet(path)
    frame.iloc[:-1].to_parquet(path, index=False)

    with pytest.raises(DataReadinessError, match="does not verify"):
        load_complete_broad_intraday_history_plan(output)


def _config() -> BroadIntradayHistoryConfig:
    payload = load_broad_intraday_history_config(POLICY).model_dump(mode="python")
    payload["first_session"] = EARLY_CLOSE
    payload["last_session"] = FULL_SESSION
    return BroadIntradayHistoryConfig.model_validate(payload)


def _inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    effective_from = pd.Timestamp("2024-07-01", tz="UTC")
    effective_to = pd.Timestamp("2024-07-06", tz="UTC")
    broad_records = [
        _membership("AAA", effective_from, effective_to),
        *[_membership(f"B{index:03d}", effective_from, effective_to) for index in range(60)],
        _membership("SPY", effective_from, effective_to, industry="ETF"),
        _membership(
            "FUND",
            effective_from,
            effective_to,
            industry="Exchange Traded Fund",
        ),
    ]
    broad = tmp_path / "broad.parquet"
    _publish_memberships(broad, pd.DataFrame(broad_records))
    pit_frame = pd.DataFrame(
        [
            _membership(
                "PITCO",
                effective_from,
                effective_to,
                source="sp_global_primary_evidence",
            )
        ]
    )
    pit = tmp_path / "pit.parquet"
    _publish_memberships(pit, pit_frame)
    corpus = tmp_path / "corpus"
    _publish_corpus(
        corpus,
        {
            "AAA": [EARLY_CLOSE, FULL_SESSION],
            "PITCO": [EARLY_CLOSE],
        },
    )
    return broad, pit, corpus


def _membership(
    ticker: str,
    effective_from: pd.Timestamp,
    effective_to: pd.Timestamp,
    *,
    industry: str = "Software",
    source: str = "finviz_current_snapshot",
) -> dict[str, object]:
    return {
        "ticker": ticker,
        "security_id": f"SEC-{ticker}",
        "effective_from_utc": effective_from,
        "effective_to_utc": effective_to,
        "available_at_utc": effective_from,
        "sector": "Technology",
        "industry": industry,
        "market_cap_bucket": "mid",
        "liquidity_bucket": "liquid",
        "primary_benchmark": "XLK",
        "universe_snapshot_id": "test-snapshot",
        "source": source,
        "availability_policy": "provider_publication_proxy",
        "schema_version": "market_data.v1",
    }


def _publish_memberships(path: Path, frame: pd.DataFrame) -> None:
    frame.to_parquet(path, index=False)
    manifest = {
        "schema": "market_data.artifact_manifest.v1",
        "artifact_type": "memberships",
        "artifact_path": str(path),
        "artifact_sha256": file_sha256(path),
        "rows": len(frame),
        "columns": list(frame.columns),
        "created_at_utc": "2026-07-30T19:39:32+00:00",
        "production_ready": False,
    }
    Path(f"{path}.manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")


def _publish_corpus(directory: Path, sessions: dict[str, list[str]]) -> None:
    files = []
    for ticker, dates in sessions.items():
        rows = []
        for raw_date in dates:
            start = pd.Timestamp("2024-07-03 13:30", tz="UTC") if raw_date == EARLY_CLOSE else pd.Timestamp("2024-07-05 13:30", tz="UTC")
            rows.append(
                {
                    "ticker": ticker,
                    "session_date_et": pd.Timestamp(raw_date).date(),
                    "session_segment": "regular",
                    "timeframe": "5m",
                    "bar_start_utc": start,
                    "source": "alpaca",
                    "price_feed": "sip",
                    "adjustment": "all",
                }
            )
        path = directory / "regular" / "5m" / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        pd.DataFrame(rows).to_parquet(path, index=False)
        files.append(
            {
                "path": path.relative_to(directory).as_posix(),
                "rows": len(rows),
                "sha256": file_sha256(path),
                "store": "regular",
                "ticker": ticker,
            }
        )
    manifest = {
        "schema": "edge_rebuild.intraday_materialization.v1",
        "window_first_session": EARLY_CLOSE,
        "window_last_session": FULL_SESSION,
        "integrity": {
            "blocking_defect_count": 0,
            "identity_breaks": [],
            "fabricated_bars": [],
            "truncated_ticker_sessions": [],
        },
        "files": files,
    }
    manifest_path = directory / "_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_corpus_authority(directory)


def _resign_corpus_file(directory: Path, *, ticker: str) -> None:
    manifest_path = directory / "_manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    path = directory / "regular" / "5m" / f"{ticker}.parquet"
    for record in manifest["files"]:
        if record["ticker"] == ticker:
            record["sha256"] = file_sha256(path)
            record["rows"] = len(pd.read_parquet(path))
    manifest_path.write_text(json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8")
    _write_corpus_authority(directory)


def _write_corpus_authority(directory: Path) -> None:
    authority = {
        "schema": "edge_rebuild.intraday_materialization_authority.v1",
        "state": "complete",
        "artifact": "_manifest.json",
        "artifact_sha256": file_sha256(directory / "_manifest.json"),
    }
    (directory / "_authority.json").write_text(json.dumps(authority, indent=2, sort_keys=True), encoding="utf-8")
