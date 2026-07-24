from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from typer.testing import CliRunner

from market_predictor.canonical.contracts import CanonicalUniverseMembership, SourceCollection
from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.cli import app


class CanonicalCliTests(unittest.TestCase):
    def test_production_decision_pipeline_uses_verified_point_in_time_inputs(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            raw_bars = root / "raw_bars.parquet"
            raw_events = root / "raw_events.parquet"
            raw_sources = root / "raw_sources.parquet"
            raw_memberships = root / "raw_memberships.parquet"
            bars = root / "bars.parquet"
            events = root / "events.parquet"
            sources = root / "sources.parquet"
            memberships = root / "memberships.parquet"
            decisions = root / "decisions.parquet"

            pd.DataFrame(
                {
                    "symbol": ["MSFT"],
                    "timestamp": [pd.Timestamp("2026-07-21T04:00:00Z")],
                    "ingested_at_utc": [pd.Timestamp("2026-07-21T20:16:00Z")],
                    "open": [100.0],
                    "high": [101.0],
                    "low": [99.0],
                    "close": [100.5],
                    "volume": [1_000.0],
                }
            ).to_parquet(raw_bars, index=False)
            pd.DataFrame(
                {
                    "ticker": ["MSFT"],
                    "timestamp": [pd.Timestamp("2026-07-21T20:30:00Z")],
                    "ingested_at_utc": [pd.Timestamp("2026-07-21T21:30:00Z")],
                    "sentiment_scored_at_utc": [pd.Timestamp("2026-07-21T21:31:00Z")],
                    "source": ["alpaca:benzinga"],
                    "title": ["Microsoft announces an update"],
                    "url": ["https://example.test/msft"],
                    "summary": ["Summary"],
                    "text": ["Text"],
                    "sentiment_numeric": [0.7],
                    "raw": ['{"id": 1, "created_at": "2026-07-21T13:30:00Z"}'],
                }
            ).to_parquet(raw_events, index=False)
            collection = SourceCollection(
                collection_id="alpaca-collection-1",
                ticker="MSFT",
                source_family="alpaca",
                requested_start_utc=datetime(2026, 7, 20, tzinfo=UTC),
                requested_end_utc=datetime(2026, 7, 21, 21, 30, tzinfo=UTC),
                started_at_utc=datetime(2026, 7, 21, 21, 30, tzinfo=UTC),
                completed_at_utc=datetime(2026, 7, 21, 21, 31, tzinfo=UTC),
                status="observed",
                row_count=1,
            )
            pd.DataFrame([collection.model_dump()]).to_parquet(raw_sources, index=False)
            membership = CanonicalUniverseMembership(
                ticker="MSFT",
                effective_from_utc=datetime(2026, 1, 1, tzinfo=UTC),
                available_at_utc=datetime(2026, 1, 1, tzinfo=UTC),
                sector="Technology",
                industry="Software",
                market_cap_bucket="mega",
                liquidity_bucket="high",
                primary_benchmark="XLK",
                universe_snapshot_id="snapshot-1",
                source="finviz",
                availability_policy="observed",
            )
            pd.DataFrame([membership.model_dump()]).to_parquet(raw_memberships, index=False)

            invocations = [
                [
                    "canonicalize-bars",
                    "--input-path",
                    str(raw_bars),
                    "--out",
                    str(bars),
                    "--timeframe",
                    "1d",
                    "--price-feed",
                    "sip",
                ],
                ["canonicalize-events", "--input-path", str(raw_events), "--out", str(events)],
                [
                    "canonicalize-source-collections",
                    "--input-path",
                    str(raw_sources),
                    "--out",
                    str(sources),
                ],
                [
                    "canonicalize-memberships",
                    "--input-path",
                    str(raw_memberships),
                    "--out",
                    str(memberships),
                ],
                [
                    "build-canonical-decisions",
                    "--bars",
                    str(bars),
                    "--events",
                    str(events),
                    "--source-collections",
                    str(sources),
                    "--memberships",
                    str(memberships),
                    "--required-sources",
                    "alpaca",
                    "--decision-mode",
                    "swing-nightly",
                    "--out",
                    str(decisions),
                ],
            ]
            for arguments in invocations:
                result = runner.invoke(app, arguments)
                self.assertEqual(result.exit_code, 0, msg=f"{arguments}: {result.output}\n{result.exception}")

            decision_frame, manifest = load_canonical_artifact(decisions, expected_type="decisions")
            self.assertTrue(manifest["production_ready"])
            self.assertEqual(decision_frame.loc[0, "event_count_2h"], 1)
            self.assertEqual(decision_frame.loc[0, "source_status_alpaca"], "observed")
            self.assertEqual(decision_frame.loc[0, "primary_benchmark"], "XLK")
            self.assertEqual(
                decision_frame.loc[0, "bar_available_at_utc"],
                pd.Timestamp("2026-07-21T20:15:00Z"),
            )
            self.assertEqual(decision_frame.loc[0, "decision_time_utc"], pd.Timestamp("2026-07-21T22:00:00Z"))
            self.assertEqual(
                decision_frame.loc[0, "prediction_cutoff_policy_id"],
                "xnys_1800_america_new_york_v1",
            )
            self.assertLessEqual(
                decision_frame.loc[0, "latest_event_feature_available_at_utc"],
                decision_frame.loc[0, "decision_time_utc"],
            )

    def test_production_decision_command_requires_explicit_mode(self) -> None:
        result = CliRunner().invoke(
            app,
            ["build-canonical-decisions", "--help"],
            color=False,
            terminal_width=240,
        )
        self.assertEqual(result.exit_code, 0, msg=result.output)
        self.assertIn("--decision-mode", result.output)
        self.assertIn("required", result.output.lower())


if __name__ == "__main__":
    unittest.main()
