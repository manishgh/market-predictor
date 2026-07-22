from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import pandas as pd
from typer.testing import CliRunner

from market_predictor.canonical.audits import CanonicalAuditCheck, CanonicalAuditReport
from market_predictor.canonical.store import load_canonical_artifact, write_canonical_artifact
from market_predictor.cli import app
from market_predictor.intraday.contracts import (
    INTRADAY_MODEL_FEATURES,
    IntradayDatasetConfig,
    downside_target_column,
    opportunity_target_column,
)
from market_predictor.intraday.dataset import (
    build_intraday_dataset,
    build_intraday_inference_features,
)


class IntradayDatasetV1Tests(unittest.TestCase):
    def test_builds_latest_label_free_intraday_inference_group(self) -> None:
        decisions, one_minute, benchmarks, events, collections = _inputs()
        features, audit = build_intraday_inference_features(
            decisions,
            one_minute,
            benchmarks,
            global_events=events,
            global_source_collections=collections,
            config=_config(),
        )

        self.assertTrue(audit.passed, audit.to_frame().to_dict(orient="records"))
        self.assertEqual(len(features), 2)
        self.assertEqual(features["decision_time_utc"].nunique(), 1)
        self.assertFalse(any(column.startswith(("target_", "path_", "label_", "future_")) for column in features))

    def test_cli_publishes_hash_verified_intraday_dataset(self) -> None:
        decisions, one_minute, benchmarks, events, collections = _inputs()
        with TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            inputs = {
                "decisions": (decisions, root / "decisions.parquet"),
                "one_minute": (one_minute, root / "one_minute.parquet"),
                "benchmarks": (benchmarks, root / "benchmarks.parquet"),
                "events": (events, root / "events.parquet"),
                "collections": (collections, root / "collections.parquet"),
            }
            artifact_types = {
                "decisions": "decisions",
                "one_minute": "bars",
                "benchmarks": "bars",
                "events": "events",
                "collections": "source_collections",
            }
            for name, (frame, path) in inputs.items():
                write_canonical_artifact(
                    frame,
                    path,
                    artifact_type=artifact_types[name],
                    audit=_passing_audit(len(frame)),
                )
            config_path = root / "config.json"
            config_path.write_text(
                """{
                    "horizon_minutes": 5,
                    "decision_stride_bars": 1,
                    "min_five_minute_bars": 50,
                    "min_one_minute_bars": 50,
                    "minimum_cross_section": 2,
                    "first_decision_minute_et": 600,
                    "last_decision_minute_et": 945
                }""",
                encoding="utf-8",
            )
            output = root / "intraday_dataset.parquet"
            result = CliRunner().invoke(
                app,
                [
                    "build-intraday-dataset",
                    "--decisions",
                    str(inputs["decisions"][1]),
                    "--one-minute-bars",
                    str(inputs["one_minute"][1]),
                    "--benchmark-bars",
                    str(inputs["benchmarks"][1]),
                    "--global-events",
                    str(inputs["events"][1]),
                    "--global-source-collections",
                    str(inputs["collections"][1]),
                    "--config",
                    str(config_path),
                    "--out",
                    str(output),
                ],
            )
            self.assertEqual(result.exit_code, 0, msg=f"{result.output}\n{result.exception}")
            frame, manifest = load_canonical_artifact(output, expected_type="intraday_dataset")
            self.assertGreater(int(frame["label_eligible"].sum()), 0)
            self.assertTrue(manifest["production_ready"])
            self.assertEqual(len(manifest["inputs"]), 5)

            live_output = root / "intraday_live_features.parquet"
            live_result = CliRunner().invoke(
                app,
                [
                    "build-intraday-live-features",
                    "--decisions",
                    str(inputs["decisions"][1]),
                    "--one-minute-bars",
                    str(inputs["one_minute"][1]),
                    "--benchmark-bars",
                    str(inputs["benchmarks"][1]),
                    "--global-events",
                    str(inputs["events"][1]),
                    "--global-source-collections",
                    str(inputs["collections"][1]),
                    "--config",
                    str(config_path),
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
                expected_type="intraday_inference_features",
            )
            self.assertEqual(live_frame["decision_time_utc"].nunique(), 1)
            self.assertNotIn("path_outcome", live_frame)
            self.assertTrue(live_manifest["production_ready"])

    def test_builds_exact_completed_bar_features_and_one_minute_labels(self) -> None:
        decisions, one_minute, benchmarks, events, collections = _inputs()
        config = _config()

        dataset, audit = build_intraday_dataset(
            decisions,
            one_minute,
            benchmarks,
            global_events=events,
            global_source_collections=collections,
            config=config,
        )

        self.assertTrue(audit.passed, audit.to_frame().to_dict(orient="records"))
        eligible = dataset[dataset["label_eligible"]]
        self.assertFalse(eligible.empty)
        self.assertTrue((eligible["feature_available_at_utc"] <= eligible["decision_time_utc"]).all())
        self.assertTrue((eligible["entry_time_utc"] >= eligible["decision_time_utc"]).all())
        outcomes = (
            eligible[opportunity_target_column(config.horizon_minutes)]
            + eligible[downside_target_column(config.horizon_minutes)]
            + eligible[f"path_timeout_{config.horizon_minutes}m"]
        )
        self.assertTrue(outcomes.eq(1).all())
        self.assertTrue(eligible["independent_event_id"].notna().any())
        self.assertTrue(set(INTRADAY_MODEL_FEATURES).issubset(dataset.columns))
        self.assertFalse(any(feature.startswith("event_") for feature in INTRADAY_MODEL_FEATURES))

    def test_missing_future_one_minute_bar_fails_exact_path_audit(self) -> None:
        decisions, one_minute, benchmarks, events, collections = _inputs()
        missing_start = pd.Timestamp("2025-01-08 14:02", tz="America/New_York").tz_convert("UTC")
        one_minute = one_minute[~(one_minute["ticker"].eq("AAA") & one_minute["bar_start_utc"].eq(missing_start))].copy()

        _, audit = build_intraday_dataset(
            decisions,
            one_minute,
            benchmarks,
            global_events=events,
            global_source_collections=collections,
            config=_config(),
        )

        checks = audit.to_frame().set_index("name")
        self.assertEqual(checks.loc["intraday_exact_label_path", "status"], "fail")
        self.assertGreater(int(checks.loc["intraday_exact_label_path", "failures"]), 0)

    def test_missing_exact_benchmark_execution_bar_fails_audit(self) -> None:
        decisions, one_minute, benchmarks, events, collections = _inputs()
        missing_start = pd.Timestamp("2025-01-08 14:00", tz="America/New_York").tz_convert("UTC")
        one_minute = one_minute[~(one_minute["ticker"].eq("SPY") & one_minute["bar_start_utc"].eq(missing_start))].copy()

        _, audit = build_intraday_dataset(
            decisions,
            one_minute,
            benchmarks,
            global_events=events,
            global_source_collections=collections,
            config=_config(),
        )

        checks = audit.to_frame().set_index("name")
        self.assertEqual(checks.loc["intraday_label_benchmarks", "status"], "fail")


def _config() -> IntradayDatasetConfig:
    return IntradayDatasetConfig(
        horizon_minutes=5,
        decision_stride_bars=1,
        min_five_minute_bars=50,
        min_one_minute_bars=50,
        minimum_cross_section=2,
        first_decision_minute_et=10 * 60,
        last_decision_minute_et=15 * 60 + 45,
    )


def _inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    sessions = [pd.Timestamp(value).date() for value in ("2025-01-06", "2025-01-07", "2025-01-08")]
    tickers = ["AAA", "BBB", "SPY", "QQQ", "XLK"]
    one_minute_parts = [
        _one_minute_session(ticker, session, ticker_index) for ticker_index, ticker in enumerate(tickers) for session in sessions
    ]
    one_minute = pd.concat(one_minute_parts, ignore_index=True)
    five_minute = _aggregate_five_minute(one_minute)
    decisions = five_minute[five_minute["ticker"].isin({"AAA", "BBB"})].copy()
    decisions["decision_time_utc"] = decisions["available_at_utc"]
    decisions["feature_available_at_utc"] = decisions["available_at_utc"]
    decisions["membership_available_at_utc"] = pd.Timestamp("2024-01-01", tz="UTC")
    decisions["primary_benchmark"] = "XLK"
    decisions["universe_snapshot_id"] = "synthetic-v1"
    decisions["market_cap_bucket"] = "large"
    decisions["liquidity_bucket"] = "high"
    decisions["source_status_alpaca"] = "observed_empty"
    decisions["source_observed_rows_alpaca"] = 0
    decisions["source_status_available_at_utc_alpaca"] = decisions["decision_time_utc"]
    decisions["source_coverage_end_utc_alpaca"] = decisions["decision_time_utc"]
    benchmarks = five_minute[five_minute["ticker"].isin({"SPY", "QQQ", "XLK"})].copy()
    market_times = sorted(decisions["decision_time_utc"].unique())
    collection_rows = []
    for moment in market_times:
        timestamp = pd.Timestamp(moment)
        for source in ("alpaca", "gdelt"):
            collection_rows.append(
                {
                    "ticker": "MARKET",
                    "source_family": source,
                    "requested_end_utc": timestamp,
                    "completed_at_utc": timestamp,
                    "status": "observed_empty",
                    "row_count": 0,
                }
            )
    collections = pd.DataFrame(collection_rows)
    event_rows = []
    for session in sessions:
        available = pd.Timestamp(f"{session} 09:00", tz="America/New_York").tz_convert("UTC")
        event_rows.append(
            {
                "ticker": "MARKET",
                "event_id": f"market-{session}",
                "feature_available_at_utc": available,
                "availability_policy": "observed",
                "sentiment_numeric": 0.1,
                "relevance": 1.0,
            }
        )
    return decisions, one_minute, benchmarks, pd.DataFrame(event_rows), collections


def _one_minute_session(ticker: str, session: object, ticker_index: int) -> pd.DataFrame:
    start = pd.Timestamp(f"{session} 09:30", tz="America/New_York")
    eastern = pd.date_range(start, periods=390, freq="1min")
    utc = eastern.tz_convert("UTC")
    minute = np.arange(390, dtype=float)
    base = 80.0 + ticker_index * 20.0
    drift = 0.003 if ticker in {"AAA", "SPY", "QQQ", "XLK"} else -0.001
    close = base + drift * minute + 0.03 * np.sin(minute / 13.0)
    open_price = np.concatenate(([close[0] - drift], close[:-1]))
    high = np.maximum(open_price, close) + 0.04
    low = np.minimum(open_price, close) - 0.04
    volume = 10_000.0 + (minute % 30) * 100.0 + ticker_index * 500.0
    return pd.DataFrame(
        {
            "ticker": ticker,
            "timeframe": "1m",
            "bar_start_utc": utc,
            "bar_end_utc": utc + pd.Timedelta(minutes=1),
            "available_at_utc": utc + pd.Timedelta(minutes=1),
            "open": open_price,
            "high": high,
            "low": low,
            "close": close,
            "volume": volume,
            "price_feed": "sip",
            "adjustment": "all",
        }
    )


def _aggregate_five_minute(one_minute: pd.DataFrame) -> pd.DataFrame:
    data = one_minute.copy()
    eastern = data["bar_start_utc"].dt.tz_convert("America/New_York")
    data["session"] = eastern.dt.date
    data["slot"] = ((eastern.dt.hour * 60 + eastern.dt.minute - (9 * 60 + 30)) // 5).astype(int)
    grouped = data.groupby(["ticker", "session", "slot"], sort=False)
    output = grouped.agg(
        bar_start_utc=("bar_start_utc", "first"),
        bar_end_utc=("bar_end_utc", "last"),
        available_at_utc=("available_at_utc", "last"),
        open=("open", "first"),
        high=("high", "max"),
        low=("low", "min"),
        close=("close", "last"),
        volume=("volume", "sum"),
        price_feed=("price_feed", "first"),
        adjustment=("adjustment", "first"),
    ).reset_index()
    output["timeframe"] = "5m"
    return output.drop(columns=["session", "slot"])


def _passing_audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="synthetic_input",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="synthetic canonical input",
            ),
        )
    )
