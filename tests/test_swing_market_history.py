from __future__ import annotations

import json
import tempfile
import unittest
from datetime import date, datetime
from pathlib import Path

import pandas as pd

from market_predictor.canonical.store import file_sha256, load_canonical_artifact
from market_predictor.swing.market_history import collect_swing_daily_history
from market_predictor.swing.market_history_audit import audit_swing_daily_history
from market_predictor.swing.panel_inputs import build_swing_market_panel_inputs
from market_predictor.v3.errors import DataReadinessError


class SwingMarketHistoryTests(unittest.TestCase):
    def test_builds_hash_bound_point_in_time_market_panel_inputs(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _write_memberships(root / "memberships.parquet")
            history = root / "history"
            collect_swing_daily_history(
                memberships_path=memberships,
                start_date=date(2026, 7, 1),
                end_date=date(2026, 7, 2),
                out_dir=history,
                fetcher=lambda symbol, start, end: _bars(),
                price_feed="sip",
                workers=2,
                benchmarks=("SPY",),
            )
            report_path, summary_path = _write_coverage_evidence(
                root=root,
                memberships=memberships,
                history=history,
            )

            stock_bars, benchmark_bars, canonical_memberships, audit = (
                build_swing_market_panel_inputs(
                    memberships_path=memberships,
                    collection_dir=history,
                    coverage_report_path=report_path,
                    coverage_summary_path=summary_path,
                    benchmarks=("SPY",),
                )
            )

            self.assertEqual(len(stock_bars), 2)
            self.assertEqual(set(stock_bars["ticker"]), {"AAA"})
            self.assertEqual(len(benchmark_bars), 2)
            self.assertEqual(set(benchmark_bars["ticker"]), {"SPY"})
            self.assertEqual(canonical_memberships.loc[0, "security_id"], "test:aaa")
            self.assertEqual(
                canonical_memberships.loc[0, "availability_policy"],
                "provider_publication_proxy",
            )
            self.assertTrue(audit["training_ready"])
            self.assertFalse(audit["production_ready"]["memberships"])

            report_path.write_text(
                report_path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )
            with self.assertRaises(DataReadinessError):
                build_swing_market_panel_inputs(
                    memberships_path=memberships,
                    collection_dir=history,
                    coverage_report_path=report_path,
                    coverage_summary_path=summary_path,
                    benchmarks=("SPY",),
                )

    def test_coverage_audit_allows_terminal_nontrading_but_blocks_interior_gaps(self) -> None:
        cases = (
            ([date(2026, 7, 6), date(2026, 7, 7)], "terminal_nontrading_gap", True),
            ([date(2026, 7, 6), date(2026, 7, 8)], "interior_gap", False),
        )
        for observed_dates, expected_gap, expected_ready in cases:
            with self.subTest(expected_gap=expected_gap), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                memberships = _write_memberships(root / "memberships.parquet")
                history = root / "history"

                collect_swing_daily_history(
                    memberships_path=memberships,
                    start_date=date(2026, 7, 6),
                    end_date=date(2026, 7, 8),
                    out_dir=history,
                    fetcher=lambda symbol, start, end, observed=tuple(observed_dates): (
                        _bars_for_dates([date(2026, 7, 6), date(2026, 7, 7), date(2026, 7, 8)])
                        if symbol == "SPY"
                        else _bars_for_dates(list(observed))
                    ),
                    price_feed="sip",
                    workers=2,
                    benchmarks=("SPY",),
                )

                report, summary = audit_swing_daily_history(
                    memberships_path=memberships,
                    collection_dir=history,
                    benchmarks=("SPY",),
                )

                self.assertEqual(report.iloc[0]["gap_class"], expected_gap)
                self.assertEqual(summary["training_ready"], expected_ready)
                self.assertEqual(summary["missing_member_sessions"], 1)

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
            "company": ["AAA Test Company"],
            "effective_from_utc": [pd.Timestamp("2026-01-01", tz="UTC")],
            "effective_to_utc": [pd.NaT],
            "sector": ["Technology"],
            "industry": ["Software"],
            "market_cap_bucket": ["large_cap_sp500"],
            "liquidity_bucket": ["sp500_constituent"],
            "primary_benchmark": ["SPY"],
            "universe_snapshot_id": ["test-snapshot-1"],
        }
    ).to_parquet(path, index=False)
    return path


def _write_coverage_evidence(
    *,
    root: Path,
    memberships: Path,
    history: Path,
) -> tuple[Path, Path]:
    report, summary = audit_swing_daily_history(
        memberships_path=memberships,
        collection_dir=history,
        benchmarks=("SPY",),
    )
    report_path = root / "coverage.csv"
    summary_path = root / "coverage.json"
    report.to_csv(report_path, index=False)
    summary["report"] = {
        "path": str(report_path.resolve()),
        "rows": len(report),
        "sha256": file_sha256(report_path),
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return report_path, summary_path


def _bars() -> pd.DataFrame:
    return _bars_for_dates([date(2026, 7, 1), date(2026, 7, 2)])


def _bars_for_dates(dates: list[date]) -> pd.DataFrame:
    count = len(dates)
    return pd.DataFrame(
        {
            "date": dates,
            "open": [100.0 + value for value in range(count)],
            "high": [102.0 + value for value in range(count)],
            "low": [99.0 + value for value in range(count)],
            "close": [101.0 + value for value in range(count)],
            "volume": [1_000_000 + value * 100_000 for value in range(count)],
        }
    )


if __name__ == "__main__":
    unittest.main()
