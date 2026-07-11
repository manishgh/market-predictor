from __future__ import annotations

import tempfile
import unittest
from datetime import date
from pathlib import Path

import pandas as pd

from market_predictor.v3.development import (
    DevelopmentDatasetConfig,
    build_monthly_development_dataset,
    load_verified_development_dataset,
)
from market_predictor.v3.errors import DataReadinessError


class V3DevelopmentDatasetTests(unittest.TestCase):
    def test_builds_monthly_point_in_time_dataset_without_loading_whole_universe(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars_dir = root / "bars"
            benchmark_dir = root / "benchmarks"
            bars_dir.mkdir()
            benchmark_dir.mkdir()
            times = pd.date_range("2026-07-08 13:00:00Z", periods=40, freq="5min")
            for ticker, move in {"AAA": 0.2, "BBB": -0.1}.items():
                _raw_bars(ticker, times, move).to_parquet(bars_dir / f"{ticker}.parquet", index=False)
            for ticker, move in {"SPY": 0.03, "QQQ": 0.05, "XLK": 0.04}.items():
                _raw_bars(ticker, times, move).to_parquet(benchmark_dir / f"{ticker}.parquet", index=False)
            memberships = pd.DataFrame(
                {
                    "ticker": ["AAA", "BBB"],
                    "effective_from_utc": ["2026-01-01T00:00:00Z"] * 2,
                    "effective_to_utc": [None, None],
                    "sector": ["Information Technology"] * 2,
                    "industry": ["Software"] * 2,
                    "market_cap_bucket": ["large_cap_sp500"] * 2,
                    "liquidity_bucket": ["sp500_constituent"] * 2,
                    "primary_benchmark": ["XLK"] * 2,
                    "universe_snapshot_id": ["snapshot-1"] * 2,
                }
            )
            memberships_path = root / "memberships.parquet"
            memberships.to_parquet(memberships_path, index=False)
            output_dir = root / "output"
            report = build_monthly_development_dataset(
                bars_directory=bars_dir,
                benchmark_directory=benchmark_dir,
                memberships_path=memberships_path,
                technical_directory=root / "technical",
                output_directory=output_dir,
                config=DevelopmentDatasetConfig(
                    minimum_cross_section=2,
                    workers=2,
                    horizons_bars=(1, 2),
                    primary_horizon_bars=2,
                    decision_stride_bars=3,
                    decision_start_date=date(2026, 7, 8),
                ),
            )
            dataset = pd.read_parquet(output_dir)
            resumed = build_monthly_development_dataset(
                bars_directory=bars_dir,
                benchmark_directory=benchmark_dir,
                memberships_path=memberships_path,
                technical_directory=root / "technical",
                output_directory=root / "output-resumed",
                reuse_technical=True,
                config=DevelopmentDatasetConfig(
                    minimum_cross_section=2,
                    workers=2,
                    horizons_bars=(1, 2),
                    primary_horizon_bars=2,
                    decision_stride_bars=3,
                    decision_start_date=date(2026, 7, 8),
                ),
            )
            resumed_in_place = build_monthly_development_dataset(
                bars_directory=bars_dir,
                benchmark_directory=benchmark_dir,
                memberships_path=memberships_path,
                technical_directory=root / "technical",
                output_directory=output_dir,
                reuse_technical=True,
                resume_output=True,
                config=DevelopmentDatasetConfig(
                    minimum_cross_section=2,
                    workers=2,
                    horizons_bars=(1, 2),
                    primary_horizon_bars=2,
                    decision_stride_bars=3,
                    decision_start_date=date(2026, 7, 8),
                ),
            )
            verified, verified_manifest = load_verified_development_dataset(output_dir)
        self.assertGreater(len(dataset), 0)
        self.assertEqual(report["summary"]["tickers"], 2)
        self.assertTrue(dataset.groupby("decision_group_id")["ticker"].nunique().eq(2).all())
        self.assertTrue(dataset["price_feed"].eq("sip").all())
        self.assertTrue(resumed["technical_reused"])
        self.assertEqual(resumed["dataset_fingerprint"], report["dataset_fingerprint"])
        self.assertEqual(resumed_in_place["dataset_fingerprint"], report["dataset_fingerprint"])
        self.assertEqual(len(verified), report["summary"]["label_rows"])
        self.assertEqual(verified_manifest["dataset_fingerprint"], report["dataset_fingerprint"])

    def test_verified_loader_rejects_modified_month(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars_dir = root / "bars"
            benchmark_dir = root / "benchmarks"
            bars_dir.mkdir()
            benchmark_dir.mkdir()
            times = pd.date_range("2026-07-08 13:30:00Z", periods=40, freq="5min")
            for ticker in ("AAA", "BBB"):
                _raw_bars(ticker, times, 0.1).to_parquet(bars_dir / f"{ticker}.parquet", index=False)
            for ticker in ("SPY", "QQQ", "XLK"):
                _raw_bars(ticker, times, 0.03).to_parquet(benchmark_dir / f"{ticker}.parquet", index=False)
            memberships_path = root / "memberships.parquet"
            _memberships().to_parquet(memberships_path, index=False)
            output = root / "output"
            build_monthly_development_dataset(
                bars_directory=bars_dir,
                benchmark_directory=benchmark_dir,
                memberships_path=memberships_path,
                technical_directory=root / "technical",
                output_directory=output,
                config=DevelopmentDatasetConfig(
                    minimum_cross_section=2,
                    workers=2,
                    horizons_bars=(1, 2),
                    primary_horizon_bars=2,
                    decision_stride_bars=3,
                    decision_start_date=date(2026, 7, 8),
                ),
            )
            shard = next(output.glob("*.parquet"))
            shard.write_bytes(shard.read_bytes() + b"tampered")
            with self.assertRaisesRegex(DataReadinessError, "hash-invalid"):
                load_verified_development_dataset(output)

    def test_equity_prints_after_early_close_are_excluded_by_market_grid(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars_dir = root / "bars"
            benchmark_dir = root / "benchmarks"
            bars_dir.mkdir()
            benchmark_dir.mkdir()
            stock_times = pd.date_range("2024-11-29 14:30:00Z", periods=44, freq="5min")
            benchmark_times = stock_times[:42]
            for ticker in ("AAA", "BBB"):
                _raw_bars(ticker, stock_times, 0.1).to_parquet(bars_dir / f"{ticker}.parquet", index=False)
            for ticker in ("SPY", "QQQ", "XLK"):
                _raw_bars(ticker, benchmark_times, 0.03).to_parquet(benchmark_dir / f"{ticker}.parquet", index=False)
            memberships_path = root / "memberships.parquet"
            _memberships().to_parquet(memberships_path, index=False)
            report = build_monthly_development_dataset(
                bars_directory=bars_dir,
                benchmark_directory=benchmark_dir,
                memberships_path=memberships_path,
                technical_directory=root / "technical",
                output_directory=root / "output",
                config=DevelopmentDatasetConfig(
                    minimum_cross_section=2,
                    workers=2,
                    horizons_bars=(1, 2),
                    primary_horizon_bars=2,
                    decision_stride_bars=3,
                    decision_start_date=date(2024, 11, 29),
                ),
            )
            dataset = pd.read_parquet(root / "output")
        self.assertEqual(report["months"][0]["off_grid_rows_removed"], 4)
        self.assertLessEqual(dataset["primary_exit_time_utc"].max(), benchmark_times.max())

    def test_internal_shared_benchmark_gap_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars_dir = root / "bars"
            benchmark_dir = root / "benchmarks"
            bars_dir.mkdir()
            benchmark_dir.mkdir()
            times = pd.date_range("2026-07-08 13:30:00Z", periods=40, freq="5min")
            for ticker in ("AAA", "BBB"):
                _raw_bars(ticker, times, 0.1).to_parquet(bars_dir / f"{ticker}.parquet", index=False)
            _raw_bars("SPY", times, 0.03).to_parquet(benchmark_dir / "SPY.parquet", index=False)
            _raw_bars("QQQ", times, 0.03).to_parquet(benchmark_dir / "QQQ.parquet", index=False)
            _raw_bars("XLK", times.delete(20), 0.03).to_parquet(benchmark_dir / "XLK.parquet", index=False)
            memberships_path = root / "memberships.parquet"
            _memberships().to_parquet(memberships_path, index=False)
            with self.assertRaisesRegex(DataReadinessError, "benchmark market grid is incomplete"):
                build_monthly_development_dataset(
                    bars_directory=bars_dir,
                    benchmark_directory=benchmark_dir,
                    memberships_path=memberships_path,
                    technical_directory=root / "technical",
                    output_directory=root / "output",
                    config=DevelopmentDatasetConfig(
                        minimum_cross_section=2,
                        workers=2,
                        horizons_bars=(1, 2),
                        primary_horizon_bars=2,
                        decision_stride_bars=3,
                        decision_start_date=date(2026, 7, 8),
                    ),
                )


def _raw_bars(ticker: str, times: pd.DatetimeIndex, move: float) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(times):
        close = 100 + move * index
        rows.append(
            {
                "symbol": ticker,
                "timeframe": "5m",
                "timestamp": timestamp,
                "open": close,
                "high": close + 0.5,
                "low": close - 0.5,
                "close": close,
                "volume": 10_000 + index,
                "source": "alpaca",
                "price_feed": "sip",
                "adjustment": "all",
                "ingested_at_utc": pd.Timestamp("2026-07-09T00:00:00Z"),
            }
        )
    return pd.DataFrame(rows)


def _memberships() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "effective_from_utc": ["2024-01-01T00:00:00Z"] * 2,
            "effective_to_utc": [None, None],
            "sector": ["Information Technology"] * 2,
            "industry": ["Software"] * 2,
            "market_cap_bucket": ["large_cap_sp500"] * 2,
            "liquidity_bucket": ["sp500_constituent"] * 2,
            "primary_benchmark": ["XLK"] * 2,
            "universe_snapshot_id": ["snapshot-1"] * 2,
        }
    )


if __name__ == "__main__":
    unittest.main()
