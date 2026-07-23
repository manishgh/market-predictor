from __future__ import annotations

import unittest
from datetime import UTC, datetime
from pathlib import Path
from tempfile import TemporaryDirectory

import exchange_calendars as xcals
import numpy as np
import pandas as pd
from typer.testing import CliRunner

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF, swing_prediction_cutoffs
from market_predictor.canonical.store import load_canonical_artifact, write_canonical_artifact
from market_predictor.cli import app
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.dataset import build_swing_dataset, build_swing_inference_features
from market_predictor.v3.errors import DataReadinessError


class SwingDatasetTests(unittest.TestCase):
    def test_builds_latest_label_free_swing_inference_group(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        features, audit = build_swing_inference_features(
            decisions,
            benchmarks,
            global_events=events,
            global_source_collections=sources,
            config=SwingDatasetConfig(
                min_daily_bars=250,
                minimum_cross_section=2,
                required_global_sources=("alpaca",),
            ),
        )

        self.assertTrue(audit.passed, audit.to_frame().to_dict(orient="records"))
        self.assertEqual(len(features), 2)
        self.assertEqual(features["decision_time_utc"].nunique(), 1)
        self.assertFalse(any(column.startswith(("future_", "target_", "label_")) for column in features))

    def test_cli_publishes_hash_verified_swing_dataset(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = {
                "decisions": (decisions, root / "decisions.parquet"),
                "bars": (benchmarks, root / "benchmarks.parquet"),
                "events": (events, root / "global_events.parquet"),
                "source_collections": (sources, root / "global_sources.parquet"),
            }
            for artifact_type, (frame, path) in inputs.items():
                write_canonical_artifact(
                    frame,
                    path,
                    artifact_type=artifact_type,
                    audit=_passing_audit(len(frame)),
                )
            config = root / "dataset.json"
            config.write_text(
                '{"required_global_sources":["alpaca"],"minimum_cross_section":2}',
                encoding="utf-8",
            )
            output = root / "swing_dataset.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "build-swing-dataset",
                    "--decisions",
                    str(inputs["decisions"][1]),
                    "--benchmark-bars",
                    str(inputs["bars"][1]),
                    "--global-events",
                    str(inputs["events"][1]),
                    "--global-source-collections",
                    str(inputs["source_collections"][1]),
                    "--config",
                    str(config),
                    "--out",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=f"{result.output}\n{result.exception}")
            frame, manifest = load_canonical_artifact(output, expected_type="swing_dataset")
            self.assertGreater(int(frame["label_eligible"].sum()), 0)
            self.assertTrue(manifest["production_ready"])
            self.assertEqual(len(manifest["inputs"]), 4)

            live_output = root / "swing_live_features.parquet"
            live_result = CliRunner().invoke(
                app,
                [
                    "build-swing-live-features",
                    "--decisions",
                    str(inputs["decisions"][1]),
                    "--benchmark-bars",
                    str(inputs["bars"][1]),
                    "--global-events",
                    str(inputs["events"][1]),
                    "--global-source-collections",
                    str(inputs["source_collections"][1]),
                    "--config",
                    str(config),
                    "--out",
                    str(live_output),
                ],
            )
            self.assertEqual(
                live_result.exit_code,
                0,
                msg=f"{live_result.output}\n{live_result.exception}",
            )
            live_frame, live_manifest = load_canonical_artifact(
                live_output,
                expected_type="swing_inference_features",
            )
            self.assertEqual(live_frame["decision_time_utc"].nunique(), 1)
            self.assertNotIn("future_net_return_5d", live_frame)
            self.assertTrue(live_manifest["production_ready"])

    def test_builds_warm_exact_point_in_time_swing_rows(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        config = SwingDatasetConfig(
            horizon_sessions=5,
            min_daily_bars=250,
            minimum_cross_section=2,
            required_global_sources=("alpaca",),
        )
        dataset, audit = build_swing_dataset(
            decisions,
            benchmarks,
            global_events=events,
            global_source_collections=sources,
            config=config,
        )
        self.assertTrue(audit.passed, msg=audit.to_frame().to_string(index=False))
        eligible = dataset[dataset["label_eligible"]]
        self.assertGreater(len(eligible), 0)
        self.assertTrue(eligible["daily_bar_count"].ge(250).all())
        self.assertTrue((eligible["feature_available_at_utc"] <= eligible["decision_time_utc"]).all())
        self.assertTrue((eligible["entry_time_utc"] > eligible["decision_time_utc"]).all())
        self.assertTrue((eligible["exit_time_utc"] > eligible["entry_time_utc"]).all())
        self.assertTrue(eligible["future_net_return_5d"].notna().all())
        self.assertTrue(eligible["future_excess_return_5d_vs_spy"].notna().all())
        self.assertTrue(eligible["future_excess_return_5d_vs_sector"].notna().all())
        self.assertTrue(set(eligible["global_source_status_alpaca"]).issubset({"observed", "observed_empty"}))

    def test_missing_sector_bar_fails_dataset_audit(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        missing_session = benchmarks.loc[
            (benchmarks["ticker"] == "XLK") & (benchmarks["session_date_et"].notna()),
            "session_date_et",
        ].iloc[-8]
        benchmarks = benchmarks[~((benchmarks["ticker"] == "XLK") & (benchmarks["session_date_et"] == missing_session))].copy()
        config = SwingDatasetConfig(
            horizon_sessions=5,
            min_daily_bars=250,
            minimum_cross_section=2,
            required_global_sources=("alpaca",),
        )
        _, audit = build_swing_dataset(
            decisions,
            benchmarks,
            global_events=events,
            global_source_collections=sources,
            config=config,
        )
        check = next(item for item in audit.checks if item.name == "swing_benchmark_coverage")
        self.assertEqual(check.status, "fail")

    def test_rejects_research_proxy_global_events(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        events["availability_policy"] = "provider_publication_proxy"
        with self.assertRaises(DataReadinessError):
            build_swing_dataset(
                decisions,
                benchmarks,
                global_events=events,
                global_source_collections=sources,
                config=SwingDatasetConfig(
                    min_daily_bars=250,
                    minimum_cross_section=2,
                    required_global_sources=("alpaca",),
                ),
            )

    def test_stale_global_collection_status_fails_dataset_audit(self) -> None:
        decisions, benchmarks, events, sources = _inputs()
        _, audit = build_swing_dataset(
            decisions,
            benchmarks,
            global_events=events,
            global_source_collections=sources.iloc[:1].copy(),
            config=SwingDatasetConfig(
                min_daily_bars=250,
                minimum_cross_section=2,
                required_global_sources=("alpaca",),
            ),
        )
        check = next(item for item in audit.checks if item.name == "swing_global_source_coverage")
        self.assertEqual(check.status, "fail")
        self.assertGreater(check.failures, 0)


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2025-01-02", "2026-03-31")[:265].tz_localize("UTC")
    decisions = pd.concat(
        [_daily_rows(ticker, sessions, offset) for ticker, offset in (("AAA", 0.0), ("BBB", 8.0))],
        ignore_index=True,
    )
    benchmarks = pd.concat(
        [
            _daily_rows("SPY", sessions, 300.0, decision=False),
            _daily_rows("QQQ", sessions, 400.0, decision=False),
            _daily_rows("XLK", sessions, 200.0, decision=False),
        ],
        ignore_index=True,
    )
    published = datetime(2025, 1, 2, 13, 0, tzinfo=UTC)
    event = CanonicalEvent(
        event_id="a" * 64,
        ticker="MARKET",
        source_family="alpaca",
        source="alpaca:benzinga",
        published_at_utc=published,
        first_seen_at_utc=published,
        available_at_utc=published,
        sentiment_scored_at_utc=published,
        feature_available_at_utc=published,
        title="Global market context",
        sentiment_numeric=0.2,
        relevance=1.0,
        availability_policy="observed",
        raw_sha256="b" * 64,
    )
    collections = []
    for index, decision_time in enumerate(sorted(decisions["decision_time_utc"].unique())):
        completed = pd.Timestamp(decision_time).to_pydatetime()
        collections.append(
            SourceCollection(
                collection_id=f"global-alpaca-{index:04d}",
                ticker="MARKET",
                source_family="alpaca",
                requested_start_utc=completed - pd.Timedelta(days=3),
                requested_end_utc=completed - pd.Timedelta(minutes=1),
                started_at_utc=completed - pd.Timedelta(minutes=1),
                completed_at_utc=completed,
                status="observed" if index == 0 else "observed_empty",
                row_count=1 if index == 0 else 0,
            ).model_dump()
        )
    return decisions, benchmarks, pd.DataFrame([event.model_dump()]), pd.DataFrame(collections)


def _daily_rows(
    ticker: str,
    sessions: pd.DatetimeIndex,
    offset: float,
    *,
    decision: bool = True,
) -> pd.DataFrame:
    positions = np.arange(len(sessions), dtype=float)
    base = 100.0 + offset + positions * 0.15 + np.sin(positions / 7.0)
    open_price = base * (1.0 + 0.001 * np.sin(positions / 3.0))
    close = base * (1.0 + 0.002 * np.cos(positions / 5.0))
    start = sessions + pd.Timedelta(hours=14, minutes=30)
    end = sessions + pd.Timedelta(hours=21)
    available = end + pd.Timedelta(minutes=15)
    frame = pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1d",
            "bar_start_utc": start,
            "bar_end_utc": end,
            "available_at_utc": available,
            "ingested_at_utc": available + pd.Timedelta(hours=1),
            "open": open_price,
            "high": np.maximum(open_price, close) * 1.01,
            "low": np.minimum(open_price, close) * 0.99,
            "close": close,
            "volume": 1_000_000 + positions * 1_000,
            "source": "alpaca",
            "price_feed": "sip",
            "adjustment": "all",
            "availability_policy": "market_interval_close",
            "schema_version": "market_data.v1",
            "session_date_et": sessions.date,
        }
    )
    if not decision:
        return frame
    cutoffs = swing_prediction_cutoffs(pd.Series(sessions.date, index=frame.index))
    frame["bar_available_at_utc"] = available
    frame["decision_time_utc"] = cutoffs
    frame["feature_available_at_utc"] = available
    frame["prediction_cutoff_policy_id"] = SWING_NIGHTLY_CUTOFF.policy_id
    frame["decision_group_id"] = cutoffs.astype(str)
    frame["primary_benchmark"] = "XLK"
    frame["sector"] = "Technology"
    frame["industry"] = "Software"
    frame["market_cap_bucket"] = "large"
    frame["liquidity_bucket"] = "high"
    frame["universe_snapshot_id"] = "snapshot-1"
    frame["membership_available_at_utc"] = pd.Timestamp("2025-01-01T00:00:00Z")
    frame["membership_effective_from_utc"] = pd.Timestamp("2024-01-01T00:00:00Z")
    frame["membership_effective_to_utc"] = pd.NaT
    frame["event_count_3d"] = 0
    frame["sentiment_mean_3d"] = 0.0
    frame["latest_event_feature_available_at_utc"] = pd.NaT
    frame["source_status_alpaca"] = "observed"
    frame["source_status_available_at_utc_alpaca"] = cutoffs
    frame["source_coverage_end_utc_alpaca"] = cutoffs - pd.Timedelta(minutes=1)
    return frame


def _passing_audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="fixture",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="test fixture",
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
