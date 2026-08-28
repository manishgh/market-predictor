from __future__ import annotations

from dataclasses import replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

import market_predictor.intraday.datasets.dataset_v2 as dataset_module
from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.intraday.datasets.dataset_v2 import (
    INTRADAY_DATASET_SCHEMA,
    _Artifact,
    _VerifiedInputs,
    load_complete_intraday_dataset,
    publish_intraday_dataset,
)
from market_predictor.intraday.datasets.selection import (
    INTRADAY_SELECTION_SCHEMA,
)
from market_predictor.intraday.features.labels import (
    build_exact_causal_intraday_labels,
)
from market_predictor.modeling.strategy_contract import (
    StrategyContract,
    load_strategy_contract,
)

ROOT = Path(__file__).resolve().parents[1]
CONTRACT_PATH = ROOT / "configs" / "edge_rebuild_strategy_contract.toml"
DAY = "2026-07-08"
BENCHMARKS = (
    "SPY",
    "QQQ",
    "XLB",
    "XLC",
    "XLE",
    "XLF",
    "XLI",
    "XLK",
    "XLP",
    "XLRE",
    "XLU",
    "XLV",
    "XLY",
)


def _contract() -> StrategyContract:
    return load_strategy_contract(CONTRACT_PATH)


def _minute_bars(ticker: str, *, offset: float = 0.0) -> pd.DataFrame:
    starts = pd.date_range(f"{DAY}T13:30:00Z", periods=390, freq="1min")
    phase = np.arange(390, dtype="float64")
    opens = 100.0 + offset + phase * 0.002 + np.sin(phase / 8.0) * 0.08
    closes = opens + np.sin(phase / 3.0) * 0.03 + 0.01
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1m",
            "bar_start_utc": starts,
            "bar_end_utc": starts + pd.Timedelta(minutes=1),
            "available_at_utc": starts + pd.Timedelta(minutes=2),
            "open": opens,
            "high": np.maximum(opens, closes) + 0.08,
            "low": np.minimum(opens, closes) - 0.08,
            "close": closes,
            "volume": 1_000.0,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _verified_inputs(tmp_path: Path, *, tickers: tuple[str, ...] = ("AAA",)) -> _VerifiedInputs:
    contract = _contract()
    selection_rows: list[dict[str, object]] = []
    membership_rows: list[dict[str, object]] = []
    coverage_rows: list[dict[str, object]] = []
    stock_artifacts: list[_Artifact] = []
    for rank, ticker in enumerate(tickers, start=1):
        selection_rows.append(
            {
                "ticker": ticker,
                "session_date_et": DAY,
                "activation_time_utc": pd.Timestamp(f"{DAY}T13:31:00Z"),
                "activation_rank": rank,
                "relative_volume_at_activation": 2.1,
                "average_volume_prior_sessions": 390_000.0,
                "median_volume_prior_sessions": 390_000.0,
                "price_at_activation": 100.0,
            }
        )
        membership_rows.append(
            {
                "ticker": ticker,
                "security_id": f"SEC-{ticker}",
                "effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "effective_to_utc": pd.NaT,
                "available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
                "sector": "Information Technology",
                "industry": "Software",
                "market_cap_bucket": "large_cap_sp500",
                "liquidity_bucket": "sp500_constituent",
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "pit-test",
                "source": "spglobal_official_point_in_time",
                "availability_policy": "provider_publication_proxy",
                "schema_version": "market_data.v1",
            }
        )
        stock = _minute_bars(ticker, offset=float(rank))
        path = tmp_path / "stock" / f"{ticker}.parquet"
        path.parent.mkdir(parents=True, exist_ok=True)
        stock.to_parquet(path, index=False)
        stock_artifacts.append(
            _Artifact(
                path=path,
                session_date_et=DAY,
                symbol_rows={ticker: len(stock)},
                sha256=file_sha256(path),
            )
        )
        coverage_rows.append(
            {
                "ticker": ticker,
                "session_date_et": DAY,
                "observed_rows": len(stock),
                "coverage_status": "complete",
            }
        )

    benchmark = pd.concat(
        [_minute_bars(ticker, offset=10.0 + index) for index, ticker in enumerate(BENCHMARKS)],
        ignore_index=True,
    )
    benchmark_path = tmp_path / "benchmark" / "session.parquet"
    benchmark_path.parent.mkdir(parents=True, exist_ok=True)
    benchmark.to_parquet(benchmark_path, index=False)
    benchmark_artifact = _Artifact(
        path=benchmark_path,
        session_date_et=DAY,
        symbol_rows={ticker: 390 for ticker in BENCHMARKS},
        sha256=file_sha256(benchmark_path),
    )
    lineage = {
        "selection_authority_sha256": "1" * 64,
        "selection_manifest_sha256": "2" * 64,
        "selection_table_sha256": "3" * 64,
        "stock_collection_authority_sha256": "4" * 64,
        "stock_collection_manifest_sha256": "5" * 64,
        "stock_coverage_authority_sha256": "6" * 64,
        "stock_coverage_manifest_sha256": "7" * 64,
        "benchmark_collection_authority_sha256": "8" * 64,
        "benchmark_collection_manifest_sha256": "9" * 64,
        "membership_authority_sha256": "a" * 64,
        "membership_manifest_sha256": "b" * 64,
        "membership_table_sha256": "c" * 64,
        "strategy_contract_file_sha256": "d" * 64,
        "strategy_contract_sha256": contract.sha256(),
    }
    return _VerifiedInputs(
        selection=pd.DataFrame(selection_rows),
        coverage=pd.DataFrame(coverage_rows),
        excluded_tickers=frozenset(),
        membership_sector_excluded_tickers=frozenset(),
        incomplete_pairs=frozenset(),
        memberships=pd.DataFrame(membership_rows),
        stock_artifacts=tuple(stock_artifacts),
        benchmark_artifacts=(benchmark_artifact,),
        benchmark_tickers=frozenset(BENCHMARKS),
        parent_lineage=lineage,
        contract=contract,
        contract_sha256=contract.sha256(),
    )


def _publish(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    verified: _VerifiedInputs,
    *,
    output_name: str = "dataset",
    session_workers: int = 4,
) -> dict[str, Any]:
    monkeypatch.setattr(dataset_module, "_verify_inputs", lambda **_: verified)
    return publish_intraday_dataset(
        selection_directory=tmp_path / "selection",
        stock_collection_directory=tmp_path / "stock_collection",
        stock_coverage_directory=tmp_path / "coverage",
        benchmark_collection_directory=tmp_path / "benchmark_collection",
        membership_authority_directory=tmp_path / "memberships",
        strategy_contract=verified.contract,
        strategy_contract_path=CONTRACT_PATH,
        output_directory=tmp_path / output_name,
        session_workers=session_workers,
    )


def test_publishes_hash_bound_partition_and_complete_abstention_audit(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    manifest = _publish(tmp_path, monkeypatch, verified)
    output = tmp_path / "dataset"

    assert manifest["schema"] == INTRADAY_DATASET_SCHEMA
    assert manifest["status"] == "complete"
    assert manifest["summary"]["published_stock_sessions"] == 1
    assert len(manifest["partitions"]) == 1
    partition = output / manifest["partitions"][0]["path"]
    rows = pd.read_parquet(partition)
    abstentions = pd.read_parquet(output / "audit" / "abstentions.parquet")

    assert rows["dataset_request_sha256"].eq(manifest["request_sha256"]).all()
    assert rows["parent_lineage_sha256"].eq(manifest["parent_lineage_sha256"]).all()
    assert set(abstentions["dataset_row_id"].dropna()).issubset(set(rows["dataset_row_id"]))
    assert (output / "_authority.json").is_file()
    assert load_complete_intraday_dataset(output) == manifest


def test_monthly_writer_uses_one_file_with_session_row_groups(
    tmp_path: Path,
) -> None:
    writer = dataset_module._MonthlyPartitionWriter(tmp_path)

    def rows(session: str, ticker: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session_date_et": [session],
                "ticker": [ticker],
                "volume_bar_number": [1],
                "dataset_eligible": [True],
                "feature_available_at_utc": [
                    pd.Timestamp(f"{session}T15:00:00Z")
                ],
                "label_available_at_utc": [
                    pd.Timestamp(f"{session}T15:30:00Z")
                ],
            }
        )

    assert writer.write(rows("2026-01-05", "AAA")) is None
    assert writer.write(rows("2026-01-06", "BBB")) is None
    january = writer.write(rows("2026-02-02", "AAA"))
    february = writer.close()

    assert january is not None
    assert february is not None
    assert january["stock_sessions"] == 2
    january_path = tmp_path / january["path"]
    assert dataset_module.pq.ParquetFile(january_path).metadata.num_row_groups == 2
    assert len(list((tmp_path / "partitions").rglob("*.parquet"))) == 2
    dataset_module._validate_monthly_partition_records(
        [january, february],
        expected_stock_sessions_by_month={"2026-01": 2, "2026-02": 1},
    )
    dataset_module._verify_monthly_partition_files(
        tmp_path, [january, february]
    )


def test_sparse_interior_month_uses_causal_expected_coverage() -> None:
    def record(month: str, stock_sessions: int = 1) -> dict[str, Any]:
        return {
            "path": (
                f"partitions/session_month_et={month}/part-00000.parquet"
            ),
            "session_month_et": month,
            "first_session_date_et": f"{month}-02",
            "last_session_date_et": f"{month}-03",
            "rows": stock_sessions,
            "eligible_rows": stock_sessions,
            "stock_sessions": stock_sessions,
            "ticker_count": 1,
        }

    records = [record("2026-01"), record("2026-02"), record("2026-03")]
    dataset_module._validate_monthly_partition_records(
        records,
        expected_stock_sessions_by_month={
            "2026-01": 1,
            "2026-02": 1,
            "2026-03": 1,
        },
    )

    with pytest.raises(DataReadinessError, match="layout contract"):
        dataset_module._validate_monthly_partition_records(
            [record("2026-01"), record("2026-02", stock_sessions=2)],
            expected_stock_sessions_by_month={"2026-01": 1, "2026-02": 1},
        )


def test_monthly_writer_rejects_cross_month_schema_drift(tmp_path: Path) -> None:
    writer = dataset_module._MonthlyPartitionWriter(tmp_path)
    january = pd.DataFrame(
        {
            "session_date_et": ["2026-01-30"],
            "ticker": ["AAA"],
            "volume_bar_number": [1],
            "dataset_eligible": [True],
            "feature_available_at_utc": [pd.Timestamp("2026-01-30T15:00:00Z")],
            "label_available_at_utc": [pd.Timestamp("2026-01-30T15:30:00Z")],
        }
    )
    february = january.copy()
    february["session_date_et"] = "2026-02-02"
    february["volume_bar_number"] = february["volume_bar_number"].astype(float)

    writer.write(january)
    with pytest.raises(DataReadinessError, match="schema changed across"):
        writer.write(february)
    writer.abort()


def test_monthly_writer_requires_strictly_increasing_sessions(tmp_path: Path) -> None:
    writer = dataset_module._MonthlyPartitionWriter(tmp_path)

    def rows(session: str) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "session_date_et": [session],
                "ticker": ["AAA"],
                "volume_bar_number": [1],
                "dataset_eligible": [True],
                "feature_available_at_utc": [pd.Timestamp(f"{session}T15:00:00Z")],
                "label_available_at_utc": [pd.Timestamp(f"{session}T15:30:00Z")],
            }
        )

    writer.write(rows("2026-01-06"))
    with pytest.raises(DataReadinessError, match="strictly increasing"):
        writer.write(rows("2026-01-05"))
    writer.abort()


def test_streaming_audit_writer_flushes_bounded_row_groups(tmp_path: Path) -> None:
    writer = dataset_module._StreamingAuditWriter(tmp_path)
    writer.write(
        [dataset_module._pair_audit("AAA", "2026-01-05", status="published", reason=None)],
        [],
    )
    writer.write(
        [dataset_module._pair_audit("BBB", "2026-01-06", status="published", reason=None)],
        [
            dataset_module._pair_abstention(
                "BBB", "2026-01-06", "label", "missing_future_bars"
            )
        ],
    )
    records = writer.close()

    assert writer.pair_rows == 2
    assert writer.abstention_rows == 1
    assert len(records) == 2
    pair_file = dataset_module.pq.ParquetFile(
        tmp_path / "audit" / "stock_session_audit.parquet"
    )
    assert pair_file.metadata.num_row_groups == 2


def test_physical_replay_rejects_multiple_sessions_in_one_row_group(
    tmp_path: Path,
) -> None:
    path = (
        tmp_path
        / "partitions"
        / "session_month_et=2026-01"
        / "part-00000.parquet"
    )
    path.parent.mkdir(parents=True)
    frame = pd.DataFrame(
        {
            "session_date_et": ["2026-01-05", "2026-01-06"],
            "ticker": ["AAA", "AAA"],
            "volume_bar_number": [1, 1],
            "dataset_eligible": [True, True],
        }
    )
    frame.to_parquet(path, index=False)
    record = {
        **dataset_module._file_record(path, tmp_path, rows=2),
        "session_month_et": "2026-01",
        "first_session_date_et": "2026-01-05",
        "last_session_date_et": "2026-01-06",
        "stock_sessions": 2,
        "ticker_count": 1,
        "eligible_rows": 2,
    }

    with pytest.raises(DataReadinessError, match="exactly one session"):
        dataset_module._verify_monthly_partition_files(tmp_path, [record])


def test_parallel_session_processing_matches_single_worker_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path, tickers=("AAA", "BBB", "CCC", "DDD"))
    single_manifest = _publish(
        tmp_path,
        monkeypatch,
        verified,
        output_name="single",
        session_workers=1,
    )
    parallel_manifest = _publish(
        tmp_path,
        monkeypatch,
        verified,
        output_name="parallel",
        session_workers=4,
    )

    def rows(root: Path, manifest: dict[str, Any]) -> pd.DataFrame:
        frames = [pd.read_parquet(root / item["path"]) for item in manifest["partitions"]]
        frame = pd.concat(frames, ignore_index=True).sort_values(
            ["ticker", "volume_bar_number"], kind="stable"
        )
        return frame.drop(
            columns=["dataset_row_id", "dataset_request_sha256"]
        ).reset_index(drop=True)

    pd.testing.assert_frame_equal(
        rows(tmp_path / "single", single_manifest),
        rows(tmp_path / "parallel", parallel_manifest),
    )
    assert single_manifest["summary"]["rows"] == parallel_manifest["summary"]["rows"]
    assert single_manifest["request_sha256"] != parallel_manifest["request_sha256"]


def test_session_worker_limit_is_enforced_before_input_loading(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path)
    monkeypatch.setattr(dataset_module, "_verify_inputs", lambda **_: verified)

    with pytest.raises(ValueError, match="between 1 and 4"):
        _publish(tmp_path, monkeypatch, verified, session_workers=5)


def test_incomplete_five_minute_pair_is_audited_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path, tickers=("AAA", "BBB"))
    verified = replace(
        verified,
        incomplete_pairs=frozenset({(DAY, "AAA")}),
    )

    manifest = _publish(tmp_path, monkeypatch, verified)
    output = tmp_path / "dataset"
    pair_audit = pd.read_parquet(output / "audit" / "stock_session_audit.parquet")

    assert manifest["summary"]["incomplete_stock_sessions"] == 1
    assert manifest["summary"]["published_stock_sessions"] == 1
    row = pair_audit.loc[pair_audit["ticker"].eq("AAA")].iloc[0]
    assert row["status"] == "abstained"
    assert row["reason"] == "incomplete_five_minute_continuity"


def test_invalid_membership_sector_benchmark_excludes_whole_security(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path, tickers=("AAA", "BBB"))
    verified = replace(
        verified,
        excluded_tickers=frozenset({"AAA"}),
        membership_sector_excluded_tickers=frozenset({"AAA"}),
    )

    manifest = _publish(tmp_path, monkeypatch, verified)
    audit = pd.read_parquet(
        tmp_path / "dataset" / "audit" / "stock_session_audit.parquet"
    )

    assert manifest["summary"]["membership_sector_excluded_securities"] == 1
    assert manifest["summary"]["published_stock_sessions"] == 1
    row = audit.loc[audit["ticker"].eq("AAA")].iloc[0]
    assert row["status"] == "excluded"
    assert row["reason"] == "whole_security_invalid_sector_benchmark_exclusion"


def test_close_plus_delay_activation_is_audited_abstention(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path, tickers=("AAA", "BBB"))
    selection = verified.selection.copy()
    selection.loc[selection["ticker"].eq("AAA"), "activation_time_utc"] = (
        pd.Timestamp(f"{DAY}T20:01:00Z")
    )
    verified = replace(verified, selection=selection)

    manifest = _publish(tmp_path, monkeypatch, verified)
    audit = pd.read_parquet(
        tmp_path / "dataset" / "audit" / "stock_session_audit.parquet"
    )

    assert manifest["summary"]["published_stock_sessions"] == 1
    row = audit.loc[audit["ticker"].eq("AAA")].iloc[0]
    assert row["status"] == "abstained"
    assert row["reason"] == "activation_not_executable_before_session_close"


def test_legacy_or_tampered_selection_parent_is_rejected_before_read(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        dataset_module,
        "load_complete_intraday_selection",
        lambda _: {"schema": "edge_rebuild.intraday_universe_selection.v2"},
    )

    with pytest.raises(DataReadinessError, match="legacy or leaked"):
        dataset_module._verify_inputs(
            selection_directory=tmp_path / "selection",
            stock_collection_directory=tmp_path / "stock",
            stock_coverage_directory=tmp_path / "coverage",
            benchmark_collection_directory=tmp_path / "benchmarks",
            membership_authority_directory=tmp_path / "memberships",
            strategy_contract=_contract(),
            strategy_contract_path=CONTRACT_PATH,
        )


def test_missing_benchmark_path_fails_closed_without_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    verified.benchmark_artifacts[0].path.unlink()

    with pytest.raises(DataReadinessError, match="benchmark one-minute path is missing"):
        _publish(tmp_path, monkeypatch, verified)

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.*.staging"))


def test_sparse_observed_benchmark_minutes_are_not_imputed_or_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path)
    artifact = verified.benchmark_artifacts[0]
    frame = pd.read_parquet(artifact.path)
    xlb_last = frame.index[frame["ticker"].eq("XLB")][-1]
    frame = frame.drop(index=xlb_last).reset_index(drop=True)
    frame.to_parquet(artifact.path, index=False)
    symbol_rows = dict(artifact.symbol_rows)
    symbol_rows["XLB"] -= 1
    verified = replace(
        verified,
        benchmark_artifacts=(
            replace(
                artifact,
                symbol_rows=symbol_rows,
                sha256=file_sha256(artifact.path),
            ),
        ),
    )

    manifest = _publish(tmp_path, monkeypatch, verified)

    assert manifest["status"] == "complete"
    assert manifest["summary"]["published_stock_sessions"] == 1


def test_stock_artifact_same_row_count_tamper_is_rejected(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path)
    artifact = verified.stock_artifacts[0]
    frame = pd.read_parquet(artifact.path)
    frame.loc[0, "close"] = float(frame.loc[0, "close"]) + 1.0
    frame.to_parquet(artifact.path, index=False)

    with pytest.raises(DataReadinessError, match="artifact hash differs"):
        _publish(tmp_path, monkeypatch, verified)


def test_final_minute_label_may_finalize_after_session_close() -> None:
    frame = pd.DataFrame(
        {
            "feature_schema_version": [dataset_module.FEATURE_SCHEMA_VERSION],
            "label_schema_version": [dataset_module.LABEL_SCHEMA_VERSION],
            "label_eligible": [True],
            "feature_available_at_utc": [pd.Timestamp(f"{DAY}T19:28:00Z")],
            "entry_time_utc": [pd.Timestamp(f"{DAY}T19:29:00Z")],
            "exit_bar_end_utc": [pd.Timestamp(f"{DAY}T20:00:00Z")],
            "label_available_at_utc": [pd.Timestamp(f"{DAY}T20:01:00Z")],
            "session_close_utc": [pd.Timestamp(f"{DAY}T20:00:00Z")],
        }
    )

    dataset_module._validate_no_leakage(frame)

    frame.loc[0, "exit_bar_end_utc"] = pd.Timestamp(f"{DAY}T20:01:00Z")
    with pytest.raises(DataReadinessError, match="leakage"):
        dataset_module._validate_no_leakage(frame)


def test_tampered_parent_authority_failure_is_not_bypassed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    def reject(_: Path) -> dict[str, Any]:
        raise DataReadinessError("intraday selection table failed its hash")

    monkeypatch.setattr(dataset_module, "load_complete_intraday_selection", reject)
    with pytest.raises(DataReadinessError, match="failed its hash"):
        dataset_module._verify_inputs(
            selection_directory=tmp_path / "selection",
            stock_collection_directory=tmp_path / "stock",
            stock_coverage_directory=tmp_path / "coverage",
            benchmark_collection_directory=tmp_path / "benchmarks",
            membership_authority_directory=tmp_path / "memberships",
            strategy_contract=_contract(),
            strategy_contract_path=CONTRACT_PATH,
        )


def test_leakage_timestamp_rejects_complete_publication(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    original = build_exact_causal_intraday_labels

    def poisoned(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original(*args, **kwargs)
        eligible = frame["label_eligible"].astype(bool)
        frame.loc[eligible, "label_available_at_utc"] = frame.loc[eligible, "exit_bar_end_utc"] - pd.Timedelta(seconds=1)
        return frame

    monkeypatch.setattr(dataset_module, "build_exact_causal_intraday_labels", poisoned)
    with pytest.raises(DataReadinessError, match="leakage"):
        _publish(tmp_path, monkeypatch, verified)

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.*.staging"))


def test_partial_publication_is_removed_and_never_gets_authority(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    original = dataset_module._MonthlyPartitionWriter.write

    def interrupted(
        writer: dataset_module._MonthlyPartitionWriter,
        *args: Any,
        **kwargs: Any,
    ) -> dict[str, Any] | None:
        original(writer, *args, **kwargs)
        raise RuntimeError("simulated interruption")

    monkeypatch.setattr(
        dataset_module._MonthlyPartitionWriter,
        "write",
        interrupted,
    )
    with pytest.raises(RuntimeError, match="simulated interruption"):
        _publish(tmp_path, monkeypatch, verified)

    assert not (tmp_path / "dataset").exists()
    assert not list(tmp_path.glob(".dataset.*.staging"))


def test_abort_suppresses_close_failure_and_resets_writer(tmp_path: Path) -> None:
    class BrokenWriter:
        def close(self) -> None:
            raise OSError("simulated close failure")

    writer = dataset_module._MonthlyPartitionWriter(tmp_path)
    writer._writer = BrokenWriter()  # type: ignore[assignment]
    writer._month = "2026-01"
    writer._path = tmp_path / "partitions" / "broken.parquet"

    writer.abort()

    assert writer._writer is None
    assert writer._month is None
    assert writer._path is None


def test_idempotent_replay_preserves_immutable_dataset(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    first = _publish(tmp_path, monkeypatch, verified)
    manifest_mtime = (tmp_path / "dataset" / "_manifest.json").stat().st_mtime_ns
    second = _publish(tmp_path, monkeypatch, verified)

    assert first == second
    assert (tmp_path / "dataset" / "_manifest.json").stat().st_mtime_ns == manifest_mtime

    changed_lineage = {**verified.parent_lineage, "selection_manifest_sha256": "f" * 64}
    changed = replace(verified, parent_lineage=changed_lineage)
    with pytest.raises(DataReadinessError, match="immutable"):
        _publish(tmp_path, monkeypatch, changed)


def test_tampered_published_partition_is_detected(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    verified = _verified_inputs(tmp_path)
    manifest = _publish(tmp_path, monkeypatch, verified)
    partition = tmp_path / "dataset" / manifest["partitions"][0]["path"]
    with partition.open("ab") as handle:
        handle.write(b"tamper")

    with pytest.raises(DataReadinessError, match="failed integrity"):
        load_complete_intraday_dataset(tmp_path / "dataset")


def test_dataset_loader_enforces_partition_metadata_verification(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path)
    _publish(tmp_path, monkeypatch, verified)

    def reject_partition_metadata(*_: object, **__: object) -> None:
        raise DataReadinessError("partition metadata rejected")

    monkeypatch.setattr(
        dataset_module,
        "_verify_monthly_partition_files",
        reject_partition_metadata,
    )
    with pytest.raises(DataReadinessError, match="partition metadata rejected"):
        load_complete_intraday_dataset(tmp_path / "dataset")


def test_stock_loading_is_bounded_to_one_exchange_session(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    verified = _verified_inputs(tmp_path, tickers=("AAA", "BBB", "CCC"))
    observed: list[tuple[int, int]] = []
    original = dataset_module._load_stock_session_batch

    def tracked(*args: Any, **kwargs: Any) -> pd.DataFrame:
        frame = original(*args, **kwargs)
        observed.append((frame["ticker"].nunique(), len(frame)))
        return frame

    monkeypatch.setattr(dataset_module, "_load_stock_session_batch", tracked)
    manifest = _publish(tmp_path, monkeypatch, verified)

    assert observed == [(3, 1_170)]
    assert manifest["summary"]["published_stock_sessions"] == 3


def test_selection_schema_constant_is_current_causal_version() -> None:
    assert INTRADAY_SELECTION_SCHEMA.endswith(".v3")


def test_membership_sector_exclusions_are_scoped_to_selected_securities() -> None:
    memberships = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB", "CCC"],
            "primary_benchmark": ["XLK", "SPY", "SPY"],
        }
    )

    assert dataset_module._membership_sector_exclusions(
        memberships,
        selected_tickers={"AAA", "BBB"},
    ) == frozenset({"BBB"})
