from __future__ import annotations

import json
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import pandas as pd
from typer.testing import CliRunner

from market_predictor.cli import app
from market_predictor.swing.inventory import (
    SwingResearchInventoryConfig,
    build_swing_research_inventory,
)


class SwingResearchInventoryTests(unittest.TestCase):
    def test_inventory_separates_technical_and_catalyst_eligibility(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            features = root / "features"
            events.mkdir()
            features.mkdir()
            _write_events(events / "MSFT_events.parquet", observed=True)
            _write_features(features, ticker="MSFT", rows=6, news_counts=[0, 0, 1, 0, 0, 0])
            memberships = root / "memberships.parquet"
            pd.DataFrame(
                {
                    "ticker": ["MSFT"],
                    "effective_from_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
                    "effective_to_utc": [pd.NaT],
                    "available_at_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
                }
            ).to_parquet(memberships, index=False)
            collections = root / "collections.parquet"
            pd.DataFrame(
                {
                    "ticker": ["MSFT"],
                    "source_family": ["alpaca"],
                    "status": ["observed"],
                }
            ).to_parquet(collections, index=False)

            report, summary = build_swing_research_inventory(
                raw_event_directory=events,
                feature_directory=features,
                memberships_path=memberships,
                source_collections_path=collections,
                config=SwingResearchInventoryConfig(
                    minimum_daily_bars=250,
                    minimum_feature_rows_5d=5,
                    minimum_news_months=0,
                    minimum_first_observed_rate=0.95,
                ),
            )

            self.assertEqual(len(report), 1)
            row = report.iloc[0]
            self.assertEqual(row["technical_eligibility"], "ineligible")
            self.assertEqual(row["catalyst_research_eligibility"], "eligible")
            self.assertEqual(row["catalyst_promotion_eligibility"], "eligible")
            self.assertEqual(row["model_eligibility"], "ineligible")
            self.assertEqual(row["news_candle_alignment_status"], "pass")
            self.assertEqual(int(row["dates_with_news_count_mismatch"]), 0)
            self.assertEqual(float(row["first_observed_event_rate"]), 1.0)
            self.assertEqual(summary["ticker_count"], 1)

    def test_inventory_fails_catalyst_when_observation_and_alignment_evidence_are_missing(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            features = root / "features"
            events.mkdir()
            features.mkdir()
            _write_events(events / "MSFT_events.parquet", observed=False)
            _write_features(features, ticker="MSFT", rows=300, news_counts=[0] * 300)

            report, _ = build_swing_research_inventory(
                raw_event_directory=events,
                feature_directory=features,
                config=SwingResearchInventoryConfig(
                    minimum_daily_bars=250,
                    minimum_feature_rows_5d=250,
                    minimum_news_months=0,
                    require_point_in_time_membership=False,
                ),
            )

            row = report.iloc[0]
            self.assertEqual(row["catalyst_research_eligibility"], "ineligible")
            self.assertEqual(row["catalyst_promotion_eligibility"], "ineligible")
            self.assertEqual(row["model_eligibility"], "warn")
            self.assertIn("first_observed_rate=0.0000<0.9500", row["eligibility_reasons"])
            self.assertIn("source_collection_evidence=missing", row["eligibility_reasons"])
            self.assertEqual(row["news_candle_alignment_status"], "fail")

    def test_promotion_collection_evidence_fails_when_any_attempt_is_not_observed(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            features = root / "features"
            events.mkdir()
            features.mkdir()
            _write_events(events / "MSFT_events.parquet", observed=True)
            _write_features(features, ticker="MSFT", rows=300, news_counts=[0, 0, 1, *([0] * 297)])
            collections = root / "collections.parquet"
            pd.DataFrame(
                {
                    "ticker": ["MSFT", "MSFT"],
                    "source_family": ["alpaca", "alpaca"],
                    "status": ["observed", "failed"],
                }
            ).to_parquet(collections, index=False)

            report, _ = build_swing_research_inventory(
                raw_event_directory=events,
                feature_directory=features,
                source_collections_path=collections,
                config=SwingResearchInventoryConfig(
                    minimum_daily_bars=250,
                    minimum_feature_rows_5d=250,
                    minimum_news_months=0,
                    require_point_in_time_membership=False,
                ),
            )

            row = report.iloc[0]
            self.assertEqual(row["technical_eligibility"], "eligible")
            self.assertEqual(row["catalyst_research_eligibility"], "eligible")
            self.assertEqual(row["catalyst_promotion_eligibility"], "ineligible")
            self.assertEqual(row["model_eligibility"], "warn")
            self.assertEqual(row["source_collection_evidence_status"], "incomplete")

    def test_corrupt_ticker_file_does_not_discard_other_ticker_results(self) -> None:
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            features = root / "features"
            events.mkdir()
            features.mkdir()
            _write_events(events / "MSFT_events.parquet", observed=True)
            _write_features(features, ticker="MSFT", rows=300, news_counts=[0, 0, 1, *([0] * 297)])
            (events / "BAD_events.parquet").write_text("not parquet", encoding="utf-8")
            _write_features(features, ticker="BAD", rows=300, news_counts=[0] * 300)

            report, summary = build_swing_research_inventory(
                raw_event_directory=events,
                feature_directory=features,
                config=SwingResearchInventoryConfig(
                    minimum_daily_bars=250,
                    minimum_feature_rows_5d=250,
                    minimum_news_months=0,
                    require_point_in_time_membership=False,
                    require_source_collection_evidence=False,
                ),
            )

            self.assertEqual(set(report["ticker"]), {"BAD", "MSFT"})
            bad = report.loc[report["ticker"].eq("BAD")].iloc[0]
            msft = report.loc[report["ticker"].eq("MSFT")].iloc[0]
            self.assertIn("events:ArrowInvalid:", bad["audit_error"])
            self.assertEqual(bad["model_eligibility"], "ineligible")
            self.assertEqual(msft["audit_error"], "")
            self.assertEqual(summary["ticker_count"], 2)

    def test_cli_writes_hash_bound_inventory_and_rejects_overwrite(self) -> None:
        runner = CliRunner()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            events = root / "events"
            features = root / "features"
            events.mkdir()
            features.mkdir()
            _write_events(events / "MSFT_events.parquet", observed=False)
            _write_features(features, ticker="MSFT", rows=6, news_counts=[0, 0, 1, 0, 0, 0])
            output = root / "inventory.csv"
            summary = root / "inventory.json"
            config = root / "config.toml"
            config.write_text(
                "\n".join(
                    [
                        "minimum_daily_bars = 250",
                        "minimum_feature_rows_5d = 5",
                        "minimum_news_months = 0",
                        "require_point_in_time_membership = false",
                        "require_source_collection_evidence = false",
                    ]
                ),
                encoding="utf-8",
            )
            args = [
                "audit-swing-research-inventory",
                "--raw-event-dir",
                str(events),
                "--feature-dir",
                str(features),
                "--out",
                str(output),
                "--summary-out",
                str(summary),
                "--config",
                str(config),
            ]

            result = runner.invoke(app, args)
            self.assertEqual(result.exit_code, 0, msg=f"{result.output}\n{result.exception}")
            payload = json.loads(summary.read_text(encoding="utf-8"))
            self.assertEqual(payload["report"]["rows"], 1)
            self.assertEqual(len(payload["report"]["sha256"]), 64)
            self.assertEqual(payload["schema_version"], "swing.research_inventory.v1")

            repeated = runner.invoke(app, args)
            self.assertNotEqual(repeated.exit_code, 0)
            self.assertIn("already exists", repeated.output)


def _write_events(path: Path, *, observed: bool) -> None:
    payload: dict[str, list[object]] = {
        "ticker": ["MSFT"],
        "timestamp": [pd.Timestamp("2026-01-05T15:00:00Z")],
        "source": ["alpaca:benzinga"],
        "title": ["Microsoft announces an update"],
        "url": ["https://example.test/msft"],
        "summary": ["Summary"],
        "text": ["Text"],
        "raw": ["{}"],
    }
    if observed:
        payload["ingested_at_utc"] = [pd.Timestamp("2026-01-05T15:01:00Z")]
        payload["availability_policy"] = ["observed"]
    pd.DataFrame(payload).to_parquet(path, index=False)


def _write_features(directory: Path, *, ticker: str, rows: int, news_counts: list[int]) -> None:
    dates = pd.bdate_range("2026-01-01", periods=rows)
    base = pd.DataFrame(
        {
            "date": dates,
            "open": [100.0] * rows,
            "high": [101.0] * rows,
            "low": [99.0] * rows,
            "close": [100.5] * rows,
            "volume": [1_000_000] * rows,
            "news_count": news_counts,
            "future_return_1d": [0.01] * rows,
            "future_return_5d": [0.02] * rows,
            "price_feed": ["sip"] * rows,
        }
    )
    base.to_parquet(directory / f"{ticker}_daily_1d.parquet", index=False)
    base.to_parquet(directory / f"{ticker}_daily_5d.parquet", index=False)


if __name__ == "__main__":
    unittest.main()
