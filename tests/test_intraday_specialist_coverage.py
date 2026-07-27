from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.intraday.specialist_coverage import (
    aggregate_setup_coverage,
    audit_requirement_coverage,
)
from market_predictor.v3.errors import DataReadinessError


class IntradaySpecialistCoverageTests(unittest.TestCase):
    def test_requirement_and_setup_coverage_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bars.parquet"
            bars = []
            for ticker in ("AAA", "SPY", "QQQ", "XLK"):
                for timestamp in pd.date_range(
                    "2026-06-01T13:30:00Z",
                    periods=4,
                    freq="1min",
                ):
                    if (
                        ticker == "XLK"
                        and timestamp
                        == pd.Timestamp("2026-06-01T13:31:00Z")
                    ):
                        continue
                    bars.append(
                        {
                            "ticker": ticker,
                            "bar_start_utc": timestamp,
                        }
                    )
            pd.DataFrame(bars).to_parquet(path, index=False)
            requirements = _requirements()
            index = {
                (ticker, "2026-06-01"): path
                for ticker in ("AAA", "SPY", "QQQ", "XLK")
            }

            audited = audit_requirement_coverage(
                requirements,
                artifact_index=index,
            )
            setups = aggregate_setup_coverage(audited).set_index("setup_id")

            failed = audited.loc[~audited["coverage_exact"]]
            self.assertEqual(len(failed), 1)
            self.assertEqual(
                failed.iloc[0]["coverage_reason"],
                "missing_minutes",
            )
            self.assertEqual(int(failed.iloc[0]["missing_bars"]), 1)
            self.assertFalse(bool(setups.loc["setup-1", "grid_complete"]))
            self.assertFalse(
                bool(setups.loc["setup-1", "sector_benchmark_complete"])
            )
            self.assertTrue(bool(setups.loc["setup-2", "grid_complete"]))

    def test_duplicate_requirement_identity_is_rejected(self) -> None:
        requirements = _requirements().iloc[[0, 0]].copy()

        with self.assertRaisesRegex(
            DataReadinessError,
            "duplicate requirement",
        ):
            audit_requirement_coverage(
                requirements,
                artifact_index={},
            )


def _requirements() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    roles = {
        "AAA": '["stock"]',
        "SPY": '["spy"]',
        "QQQ": '["qqq"]',
        "XLK": '["sector_benchmark"]',
    }
    for setup, start in (
        ("setup-1", "2026-06-01T13:30:00Z"),
        ("setup-2", "2026-06-01T13:32:00Z"),
    ):
        start_at = pd.Timestamp(start)
        for ticker, role in roles.items():
            rows.append(
                {
                    "requirement_id": f"{setup}-{ticker}",
                    "setup_id": setup,
                    "strategy_id": "INTRADAY.TEST.60M.V1",
                    "ticker": ticker,
                    "roles_json": role,
                    "segment_kind": "label",
                    "session_date_et": pd.Timestamp(
                        "2026-06-01"
                    ).date(),
                    "requested_start_utc": start_at,
                    "requested_end_utc": start_at
                    + pd.Timedelta(minutes=2),
                    "decision_time_utc": start_at,
                    "price_feed": "sip",
                    "adjustment": "all",
                    "timeframe": "1m",
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
