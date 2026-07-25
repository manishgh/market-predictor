from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.swing.market_history import collect_swing_daily_history
from market_predictor.v3.errors import DataReadinessError


class SwingMarketHistoryTests(unittest.TestCase):
    def test_failure_is_isolated_and_resume_retries_only_missing_symbol(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _write_memberships(root / "memberships.parquet")
            out_dir = root / "history"
            first_calls: list[str] = []

            def first_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
                del start, end
                first_calls.append(symbol)
                if symbol == "SPY":
                    raise RuntimeError("temporary provider failure")
                return _bars()

            first = collect_swing_daily_history(
                memberships_path=memberships,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 2),
                out_dir=out_dir,
                fetcher=first_fetch,
                price_feed="sip",
                workers=2,
                benchmarks=("SPY",),
            )

            self.assertEqual(first.status, "incomplete")
            self.assertEqual(first.unavailable_symbols, ())
            self.assertEqual(first.failed_symbols, ("SPY",))
            self.assertEqual(set(first_calls), {"AAA", "SPY"})
            aaa, _ = load_canonical_artifact(out_dir / "bars" / "1d" / "AAA.parquet", expected_type="bars")
            self.assertEqual(len(aaa), 2)
            self.assertFalse((out_dir / "_manifest.json").exists())

            second_calls: list[str] = []

            def second_fetch(symbol: str, start: datetime, end: datetime) -> pd.DataFrame:
                del start, end
                second_calls.append(symbol)
                return _bars()

            second = collect_swing_daily_history(
                memberships_path=memberships,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 2),
                out_dir=out_dir,
                fetcher=second_fetch,
                price_feed="sip",
                workers=2,
                benchmarks=("SPY",),
            )

            self.assertEqual(second.status, "complete")
            self.assertEqual(second_calls, ["SPY"])
            self.assertEqual(second.skipped_symbols, 1)
            manifest = json.loads((out_dir / "_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["artifact_count"], 2)
            self.assertEqual(manifest["total_rows"], 4)
            ledger = pd.read_parquet(out_dir / "_source_collections.parquet")
            self.assertEqual(set(ledger["status"]), {"observed"})

            with self.assertRaises(DataReadinessError):
                collect_swing_daily_history(
                    memberships_path=memberships,
                    start_date=date(2026, 7, 1),
                    end_date=date(2026, 7, 2),
                    out_dir=out_dir,
                    fetcher=second_fetch,
                    price_feed="sip",
                    workers=2,
                    benchmarks=("SPY",),
                )

    def test_observed_empty_finalizes_as_an_explicit_coverage_gap(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _write_memberships(root / "memberships.parquet")

            result = collect_swing_daily_history(
                memberships_path=memberships,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 2),
                out_dir=root / "history",
                fetcher=lambda symbol, start, end: pd.DataFrame() if symbol == "SPY" else _bars(),
                price_feed="sip",
                workers=2,
                benchmarks=("SPY",),
            )

            self.assertEqual(result.status, "complete_with_gaps")
            self.assertEqual(result.unavailable_symbols, ("SPY",))
            manifest = json.loads((root / "history" / "_manifest.json").read_text(encoding="utf-8"))
            self.assertEqual(manifest["unavailable_symbols"], ["SPY"])

    def test_rejects_non_sip_collection(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _write_memberships(root / "memberships.parquet")
            with self.assertRaises(DataReadinessError):
                collect_swing_daily_history(
                    memberships_path=memberships,
                    start_date=date(2026, 7, 1),
                    end_date=date(2026, 7, 2),
                    out_dir=root / "history",
                    fetcher=lambda symbol, start, end: _bars(),
                    price_feed="iex",
                    benchmarks=("SPY",),
                )


def _write_memberships(path: Path) -> Path:
    pd.DataFrame(
        {
            "ticker": ["AAA"],
            "security_id": ["test:aaa"],
            "effective_from_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "effective_to_utc": [pd.NaT],
            "primary_benchmark": ["SPY"],
        }
    ).to_parquet(path, index=False)
    return path


def _bars() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "date": [date(2026, 7, 1), date(2026, 7, 2)],
            "open": [100.0, 101.0],
            "high": [102.0, 103.0],
            "low": [99.0, 100.0],
            "close": [101.0, 102.0],
            "volume": [1_000_000, 1_100_000],
        }
    )


if __name__ == "__main__":
    unittest.main()
