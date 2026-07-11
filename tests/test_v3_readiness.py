from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.v3.readiness import DevelopmentReadinessConfig, audit_development_readiness


class V3DevelopmentReadinessTests(unittest.TestCase):
    def test_complete_development_inputs_pass(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars = _bars()
            bars_path = root / "bars.parquet"
            bars.to_parquet(bars_path, index=False)
            universe_path = root / "universe.csv"
            _universe().to_csv(universe_path, index=False)
            benchmark_dir = root / "benchmarks"
            benchmark_dir.mkdir()
            for symbol in ("SPY", "QQQ"):
                benchmark = bars[bars["ticker"] == "AAA"].copy()
                benchmark["ticker"] = symbol
                benchmark.to_parquet(benchmark_dir / f"{symbol}.parquet", index=False)
            report = audit_development_readiness(
                bars_path=bars_path,
                universe_path=universe_path,
                benchmark_dir=benchmark_dir,
                config=DevelopmentReadinessConfig(
                    minimum_tickers=2,
                    minimum_sessions=2,
                    required_benchmarks=("SPY", "QQQ"),
                ),
            )
        self.assertTrue(report["ready"], report["checks"])

    def test_current_snapshot_partial_feed_and_missing_benchmarks_fail(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            bars_path = root / "bars.parquet"
            _bars().drop(columns="price_feed").to_parquet(bars_path, index=False)
            universe_path = root / "universe.csv"
            pd.DataFrame({"ticker": ["AAA", "BBB"], "sector": ["Technology", "Healthcare"]}).to_csv(
                universe_path,
                index=False,
            )
            benchmark_dir = root / "benchmarks"
            benchmark_dir.mkdir()
            report = audit_development_readiness(
                bars_path=bars_path,
                universe_path=universe_path,
                benchmark_dir=benchmark_dir,
                config=DevelopmentReadinessConfig(
                    minimum_tickers=3,
                    minimum_sessions=3,
                    required_benchmarks=("SPY", "QQQ"),
                ),
            )
        self.assertFalse(report["ready"])
        self.assertIn("sip_feed_provenance", report["failures"])
        self.assertIn("point_in_time_universe_schema", report["failures"])
        self.assertIn("required_benchmarks", report["failures"])


def _bars() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for timestamp in pd.to_datetime(["2026-07-07T14:30:00Z", "2026-07-08T14:30:00Z"]):
        for ticker in ("AAA", "BBB"):
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100,
                    "high": 101,
                    "low": 99,
                    "close": 100.5,
                    "volume": 1_000,
                    "price_feed": "sip",
                }
            )
    return pd.DataFrame(rows)


def _universe() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "effective_from_utc": ["2026-01-01T00:00:00Z"] * 2,
            "effective_to_utc": [None, None],
            "sector": ["Technology", "Healthcare"],
            "industry": ["Software", "Biotech"],
            "market_cap_bucket": ["large", "large"],
            "liquidity_bucket": ["high", "high"],
            "primary_benchmark": ["QQQ", "SPY"],
            "universe_snapshot_id": ["snapshot-1", "snapshot-1"],
        }
    )


if __name__ == "__main__":
    unittest.main()
