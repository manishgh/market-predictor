from __future__ import annotations

import unittest
from datetime import UTC, datetime

import exchange_calendars as xcals
import numpy as np
import pandas as pd

from market_predictor.canonical.contracts import CanonicalEvent, SourceCollection
from market_predictor.canonical.cutoffs import SWING_NIGHTLY_CUTOFF, swing_prediction_cutoffs
from market_predictor.swing.audits import audit_swing_dataset
from market_predictor.swing.contracts import (
    CATALYST_FEATURES,
    FUNDAMENTAL_FEATURES,
    SwingDatasetConfig,
)
from market_predictor.swing.dataset import (
    _add_technical_features,
    build_swing_dataset,
    build_swing_feature_history,
    build_swing_inference_features,
)
from market_predictor.swing.labels import (
    _benchmark_label_return,
    add_exact_swing_labels,
)
from market_predictor.core.errors import DataReadinessError


class SwingDatasetTests(unittest.TestCase):
    def test_batched_technical_features_match_identity_complete_replay(
        self,
    ) -> None:
        sessions = pd.bdate_range("2025-01-02", periods=80, tz="UTC")
        parts: list[pd.DataFrame] = []
        for index in range(49):
            part = _daily_rows(
                f"T{index:02d}",
                sessions,
                float(index),
                decision=False,
            )
            part["security_id"] = f"test:t{index:02d}"
            parts.append(part)
        source = pd.concat(parts, ignore_index=True)
        expected = pd.concat(
            [
                _add_technical_features(
                    part.copy(),
                    identity_column="security_id",
                )
                for _, part in source.groupby("security_id", sort=True)
            ],
            ignore_index=True,
        )
        observed = _add_technical_features(
            source.copy(),
            identity_column="security_id",
        )
        derived = [
            column
            for column in observed.columns
            if column not in source.columns
        ]

        pd.testing.assert_frame_equal(
            observed[derived],
            expected[derived],
            check_exact=True,
        )

    def test_label_window_stops_at_exclusive_membership_end(self) -> None:
        sessions = pd.date_range("2026-07-06", periods=4, tz="UTC")
        decisions = _daily_rows("EXIT", sessions, 0.0)
        decisions["membership_effective_to_utc"] = pd.Timestamp(
            "2026-07-08T04:00:00Z"
        )
        decisions = _add_technical_features(
            decisions,
            identity_column="security_id",
        )
        decisions["feature_eligible"] = True
        benchmarks = pd.concat(
            [
                _daily_rows(ticker, sessions, offset, decision=False)
                for ticker, offset in (
                    ("SPY", 0.0),
                    ("QQQ", 2.0),
                    ("XLK", 4.0),
                )
            ],
            ignore_index=True,
        )

        labeled = add_exact_swing_labels(
            decisions,
            benchmarks,
            SwingDatasetConfig(horizon_sessions=1),
        )

        before_exit = labeled.loc[
            labeled["session_date_et"].eq(sessions[1].date())
        ].iloc[0]
        self.assertFalse(before_exit["label_window_expected"])
        self.assertFalse(before_exit["label_eligible"])

    def test_benchmark_label_return_vectorizes_exact_paths_and_missing_rows(
        self,
    ) -> None:
        lookup = pd.DataFrame(
            {
                "ticker": ["SPY", "SPY", "QQQ", "QQQ"],
                "session_date_et": [
                    datetime(2026, 7, 1).date(),
                    datetime(2026, 7, 2).date(),
                    datetime(2026, 7, 1).date(),
                    datetime(2026, 7, 2).date(),
                ],
                "open": [100.0, 101.0, 200.0, 202.0],
                "close": [101.0, 110.0, 201.0, 220.0],
            }
        ).set_index(["ticker", "session_date_et"])
        decisions = pd.DataFrame(
            {
                "entry_session_date_et": [
                    datetime(2026, 7, 1).date(),
                    datetime(2026, 7, 1).date(),
                    datetime(2026, 6, 30).date(),
                ],
                "exit_session_date_et": [
                    datetime(2026, 7, 2).date(),
                    datetime(2026, 7, 2).date(),
                    datetime(2026, 7, 2).date(),
                ],
            }
        )

        returns = _benchmark_label_return(
            decisions,
            lookup,
            pd.Series(["SPY", "QQQ", "SPY"]),
        )

        np.testing.assert_allclose(
            returns.iloc[:2],
            np.array([0.10, 0.10]),
            rtol=0,
            atol=1e-12,
        )
        self.assertTrue(pd.isna(returns.iloc[2]))

    def test_builds_technical_market_without_event_inputs(self) -> None:
        decisions, benchmarks, _, _ = _inputs()
        decisions["feature_profile"] = "technical_market"
        event_or_source_columns = [
            column
            for column in decisions.columns
            if column in {*CATALYST_FEATURES, *FUNDAMENTAL_FEATURES}
            or column.startswith(
                (
                    "source_",
                    "latest_event_",
                    "event_",
                    "reconciliation_",
                )
            )
        ]
        decisions = decisions.drop(columns=event_or_source_columns)
        config = SwingDatasetConfig(
            feature_profile="technical_market",
            min_daily_bars=250,
            minimum_cross_section=2,
            required_ticker_sources=(),
            required_global_sources=(),
        )

        dataset, audit = build_swing_dataset(
            decisions,
            benchmarks,
            config=config,
        )

        self.assertTrue(audit.passed, audit.to_frame().to_dict(orient="records"))
        self.assertEqual(set(dataset["feature_profile"]), {"technical_market"})
        self.assertTrue(set(CATALYST_FEATURES).isdisjoint(dataset.columns))
        self.assertTrue(
            all(
                not column.startswith(("source_status_", "global_source_"))
                for column in dataset.columns
            )
        )

    def test_catalyst_completeness_is_recomputed_from_source_coverage(
        self,
    ) -> None:
        decisions, benchmarks, events, sources = _inputs()
        decisions["catalyst_source_complete"] = True
        decisions = decisions.drop(
            columns="source_coverage_end_utc_alpaca"
        )
        config = SwingDatasetConfig(
            min_daily_bars=250,
            minimum_cross_section=2,
            required_global_sources=("alpaca",),
        )

        features, _ = build_swing_feature_history(
            decisions,
            benchmarks,
            global_events=events,
            global_source_collections=sources,
            config=config,
        )

        self.assertFalse(features["catalyst_source_complete"].any())

        renamed, benchmarks, events, sources = _inputs()
        renamed["catalyst_source_complete"] = True
        renamed = renamed.rename(
            columns={
                "source_status_alpaca": "source_status_unregistered",
                "source_status_available_at_utc_alpaca": (
                    "source_status_available_at_utc_unregistered"
                ),
                "source_coverage_end_utc_alpaca": (
                    "source_coverage_end_utc_unregistered"
                ),
            }
        )
        renamed_features, _ = build_swing_feature_history(
            renamed,
            benchmarks,
            global_events=events,
            global_source_collections=sources,
            config=config,
        )
        self.assertFalse(
            renamed_features["catalyst_source_complete"].any()
        )

    def test_reused_ticker_does_not_cross_security_feature_or_label_boundary(self) -> None:
        sessions = pd.date_range("2026-07-06", periods=4, tz="UTC")
        old_security = _daily_rows("REUSE", sessions[:2], 0.0)
        new_security = _daily_rows("REUSE", sessions[2:], 10.0)
        old_security["security_id"] = "security:old"
        new_security["security_id"] = "security:new"
        decisions = _add_technical_features(
            pd.concat([old_security, new_security], ignore_index=True),
            identity_column="security_id",
        )
        decisions["feature_eligible"] = True
        benchmarks = pd.concat(
            [
                _daily_rows(ticker, sessions, offset, decision=False)
                for ticker, offset in (("SPY", 0.0), ("QQQ", 2.0), ("XLK", 4.0))
            ],
            ignore_index=True,
        )

        labeled = add_exact_swing_labels(
            decisions,
            benchmarks,
            SwingDatasetConfig(horizon_sessions=1),
        )

        counts = labeled.groupby("security_id")["daily_bar_count"].apply(list).to_dict()
        self.assertEqual(counts, {"security:new": [1, 2], "security:old": [1, 2]})
        old_last = labeled.loc[
            labeled["security_id"].eq("security:old")
            & labeled["session_date_et"].eq(sessions[1].date())
        ].iloc[0]
        self.assertFalse(old_last["label_path_exact"])
        self.assertTrue(pd.isna(old_last["entry_time_utc"]))

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

    def test_source_replay_rejects_stock_and_benchmark_mutations(self) -> None:
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
        self.assertTrue(audit.passed, audit.to_frame().to_dict(orient="records"))
        row = dataset.loc[dataset["label_eligible"]].iloc[0]

        stock_source = dataset.copy()
        stock_exit = stock_source["ticker"].eq(row["ticker"]) & stock_source["session_date_et"].eq(row["exit_session_date_et"])
        stock_source.loc[stock_exit, "close"] *= 1.05
        stock_audit = audit_swing_dataset(
            dataset,
            config,
            source_frame=stock_source,
            benchmark_bars=benchmarks,
        )
        stock_check = stock_audit.to_frame().set_index("name").loc["swing_label_source_reconciliation"]
        self.assertEqual(stock_check["status"], "fail")
        self.assertGreater(int(stock_check["failures"]), 0)

        benchmark_source = benchmarks.copy()
        benchmark_entry = benchmark_source["ticker"].eq("SPY") & benchmark_source["session_date_et"].eq(row["entry_session_date_et"])
        benchmark_source.loc[benchmark_entry, "open"] *= 1.05
        benchmark_audit = audit_swing_dataset(
            dataset,
            config,
            source_frame=dataset,
            benchmark_bars=benchmark_source,
        )
        benchmark_check = benchmark_audit.to_frame().set_index("name").loc["swing_label_source_reconciliation"]
        self.assertEqual(benchmark_check["status"], "fail")
        self.assertGreater(int(benchmark_check["failures"]), 0)

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
        security_id="market:global",
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
    frame["security_id"] = f"test:{ticker.lower()}"
    frame["sector"] = "Technology"
    frame["industry"] = "Software"
    frame["market_cap_bucket"] = "large"
    frame["liquidity_bucket"] = "high"
    frame["universe_snapshot_id"] = "snapshot-1"
    frame["feature_profile"] = "catalyst_full"
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


if __name__ == "__main__":
    unittest.main()
