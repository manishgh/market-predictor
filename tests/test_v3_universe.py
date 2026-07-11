from __future__ import annotations

import unittest
from datetime import date

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.universe import (
    build_point_in_time_sp500_universe,
    parse_sp500_changes,
    symbol_changes_from_alpaca,
)


class V3PointInTimeUniverseTests(unittest.TestCase):
    def test_parses_only_sp500_rows_from_official_table(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html(),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        self.assertEqual([(item.action, item.ticker) for item in changes], [("addition", "NEW"), ("deletion", "OLD")])
        self.assertEqual(changes[0].effective_at_utc.isoformat(), "2026-01-05T05:00:00+00:00")

    def test_reverses_changes_into_non_overlapping_intervals(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html(),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        current = pd.DataFrame(
            {
                "ticker": ["AAA", "NEW"],
                "company": ["Always", "New Company"],
                "sector": ["Industrials", "Information Technology"],
                "industry": ["Services", "Software"],
            }
        )
        universe, audit = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            start_date=date(2025, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )
        old = universe[universe["ticker"] == "OLD"].iloc[0]
        new = universe[universe["ticker"] == "NEW"].iloc[0]
        self.assertEqual(pd.Timestamp(old["effective_to_utc"]), pd.Timestamp("2026-01-05T05:00:00Z"))
        self.assertEqual(pd.Timestamp(new["effective_from_utc"]), pd.Timestamp("2026-01-05T05:00:00Z"))
        self.assertEqual(new["primary_benchmark"], "XLK")
        self.assertEqual(audit["current_tickers"], 2)
        self.assertEqual(audit["historical_tickers"], 3)

    def test_fails_on_transition_contradiction(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html().replace("<td>OLD</td>", "<td>AAA</td>"),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        current = pd.DataFrame({"ticker": ["AAA", "NEW"], "sector": ["Industrials", "Information Technology"]})
        with self.assertRaises(DataReadinessError):
            build_point_in_time_sp500_universe(
                current_snapshot=current,
                changes=changes,
                start_date=date(2025, 7, 1),
                cutoff_date=date(2026, 7, 1),
                anchor_source="anchor.csv",
            )

    def test_symbol_change_preserves_membership_continuity(self) -> None:
        changes = parse_sp500_changes(
            _announcement_html().replace("NEW", "OLD"),
            source_url="https://press.spglobal.com/2026-01-01-example",
            published_date=date(2026, 1, 1),
        )
        changes = [item for item in changes if item.action == "addition"]
        aliases = symbol_changes_from_alpaca(
            pd.DataFrame(
                {
                    "id": ["change-1"],
                    "process_date": ["2026-02-01"],
                    "old_symbol": ["OLD"],
                    "new_symbol": ["NEW"],
                }
            )
        )
        current = pd.DataFrame({"ticker": ["NEW"], "sector": ["Information Technology"]})
        universe, _ = build_point_in_time_sp500_universe(
            current_snapshot=current,
            changes=changes,
            symbol_changes=aliases,
            start_date=date(2025, 7, 1),
            cutoff_date=date(2026, 7, 1),
            anchor_source="anchor.csv",
        )
        old = universe[universe["ticker"] == "OLD"].iloc[0]
        new = universe[universe["ticker"] == "NEW"].iloc[0]
        self.assertEqual(pd.Timestamp(old["effective_to_utc"]), pd.Timestamp(new["effective_from_utc"]))


def _announcement_html() -> str:
    return """
    <html><body><table>
      <tr><th>Effective Date</th><th>Index Name</th><th>Action</th><th>Company Name</th><th>Ticker</th><th>GICS Sector</th></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P 500</td><td>Addition</td><td>New Company</td><td>NEW</td><td>Information Technology</td></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P 500</td><td>Deletion</td><td>Old Company</td><td>OLD</td><td>Industrials</td></tr>
      <tr><td>January 5, 2026</td><td>S&amp;P MidCap 400</td><td>Addition</td><td>Other</td><td>OTHER</td><td>Industrials</td></tr>
    </table></body></html>
    """


if __name__ == "__main__":
    unittest.main()
