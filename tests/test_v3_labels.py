from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.v3.errors import DataReadinessError, LeakageAuditError
from market_predictor.v3.labels import V3LabelConfig, build_v3_labels


class V3LabelTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = pd.date_range("2026-07-08 14:30:00Z", periods=7, freq="5min")
        self.bars = _ticker_bars(self.times)
        self.benchmarks = _benchmark_bars(self.times)
        self.config = V3LabelConfig(
            horizons_bars=(1, 2),
            primary_horizon_bars=2,
            bar_minutes=5,
            round_trip_cost_bps=10,
            target_atr=1,
            stop_atr=1,
            minimum_ranking_group=3,
        )

    def test_exact_interval_cost_benchmark_and_path_math(self) -> None:
        labeled = build_v3_labels(self.bars, self.benchmarks, config=self.config)
        first = labeled[(labeled["ticker"] == "AAA")].sort_values("decision_time_utc").iloc[0]
        self.assertAlmostEqual(float(first["net_return_10m"]), 0.019, places=8)
        self.assertAlmostEqual(float(first["qqq_return_10m"]), 0.01, places=8)
        self.assertAlmostEqual(float(first["net_excess_qqq_10m"]), 0.009, places=8)
        self.assertIn("net_excess_qqq_to_close", first.index)
        self.assertEqual(first["path_outcome"], "target_first")
        self.assertAlmostEqual(float(first["path_realized_return_net"]), 0.009, places=8)
        self.assertEqual(int(first["bars_to_mfe_10m"]), 2)

    def test_ranking_grades_and_overlap_metadata_are_deterministic(self) -> None:
        labeled = build_v3_labels(self.bars, self.benchmarks, config=self.config)
        first_group = labeled[labeled["decision_time_utc"] == self.times[0]]
        grades = dict(zip(first_group["ticker"], first_group["ranking_grade"], strict=True))
        self.assertEqual(grades["AAA"], 4)
        self.assertEqual(grades["BBB"], 2)
        self.assertEqual(grades["CCC"], 0)
        self.assertTrue(labeled["overlap_weight"].between(0, 1, inclusive="right").all())
        self.assertTrue(labeled["concurrent_label_count"].ge(1).all())
        independent = labeled[labeled["ticker"] == "AAA"].dropna(subset=["independent_event_id"])
        source_times = pd.to_datetime(independent["decision_time_utc"], utc=True).sort_values()
        self.assertTrue(source_times.diff().dropna().ge(pd.Timedelta(minutes=10)).all())

    def test_labels_never_cross_eastern_sessions(self) -> None:
        prior_day = pd.date_range("2026-07-07 14:30:00Z", periods=3, freq="5min")
        bars = pd.concat([_ticker_bars(prior_day), self.bars], ignore_index=True)
        benchmarks = pd.concat([_benchmark_bars(prior_day), self.benchmarks], ignore_index=True)
        labeled = build_v3_labels(bars, benchmarks, config=self.config)
        entry_session = pd.to_datetime(labeled["entry_time_utc"], utc=True).dt.tz_convert("America/New_York").dt.date
        self.assertTrue((entry_session == labeled["session_date_et"]).all())

    def test_development_builder_rejects_shadow_rows(self) -> None:
        bars = self.bars.copy()
        bars["timestamp"] = pd.date_range("2026-07-09 14:30:00Z", periods=len(bars), freq="5min")
        with self.assertRaises(LeakageAuditError):
            build_v3_labels(bars, self.benchmarks, config=self.config)

    def test_missing_exact_benchmark_bar_fails_closed(self) -> None:
        benchmarks = self.benchmarks[self.benchmarks["timestamp"] != self.times[2]].copy()
        with self.assertRaises(DataReadinessError):
            build_v3_labels(self.bars, benchmarks, config=self.config)

    def test_ticker_gap_drops_non_exact_horizon(self) -> None:
        bars = self.bars[~((self.bars["ticker"] == "AAA") & (self.bars["timestamp"] == self.times[1]))].copy()

        labeled = build_v3_labels(bars, self.benchmarks, config=self.config)
        elapsed = pd.to_datetime(labeled["primary_exit_time_utc"], utc=True) - pd.to_datetime(
            labeled["decision_time_utc"],
            utc=True,
        )

        self.assertTrue(elapsed.eq(pd.Timedelta(minutes=10)).all())
        self.assertFalse(
            bool(((labeled["ticker"] == "AAA") & (labeled["decision_time_utc"] == self.times[0])).any())
        )

    def test_label_builder_preserves_frozen_feature_schema(self) -> None:
        bars = self.bars.copy()
        bars["feature_schema_version"] = "ml_v3.features.v1"
        labeled = build_v3_labels(bars, self.benchmarks, config=self.config)
        self.assertTrue(labeled["feature_schema_version"].eq("ml_v3.features.v1").all())
        self.assertEqual(labeled["label_config_hash"].nunique(), 1)
        self.assertEqual(labeled["label_config_json"].nunique(), 1)

    def test_regular_session_decisions_never_use_after_hours_bars(self) -> None:
        times = pd.date_range("2026-07-08 19:50:00Z", periods=5, freq="5min")
        bars = _ticker_bars(times)
        benchmarks = _benchmark_bars(times)
        config = V3LabelConfig(
            horizons_bars=(1,),
            primary_horizon_bars=1,
            bar_minutes=5,
            minimum_ranking_group=3,
        )
        labeled = build_v3_labels(bars, benchmarks, config=config)
        decision_minute = pd.to_datetime(labeled["decision_time_utc"], utc=True).dt.tz_convert("America/New_York")
        entry_minute = pd.to_datetime(labeled["entry_time_utc"], utc=True).dt.tz_convert("America/New_York")
        exit_minute = pd.to_datetime(labeled["primary_exit_time_utc"], utc=True).dt.tz_convert("America/New_York")
        self.assertTrue((decision_minute.dt.hour * 60 + decision_minute.dt.minute < 16 * 60).all())
        self.assertTrue((entry_minute.dt.hour * 60 + entry_minute.dt.minute < 16 * 60).all())
        self.assertTrue((exit_minute.dt.hour * 60 + exit_minute.dt.minute < 16 * 60).all())

    def test_rotating_stride_keeps_cross_section_groups_and_reduces_overlap(self) -> None:
        times = pd.date_range("2026-07-08 14:30:00Z", periods=20, freq="5min")
        bars = _ticker_bars_extended(times)
        benchmarks = _benchmark_bars_extended(times)
        config = V3LabelConfig(
            horizons_bars=(2,),
            primary_horizon_bars=2,
            bar_minutes=5,
            minimum_ranking_group=3,
            decision_stride_bars=3,
            rotate_decision_offset_by_session=True,
        )
        labeled = build_v3_labels(bars, benchmarks, config=config)
        self.assertTrue(labeled.groupby("decision_group_id")["ticker"].nunique().eq(3).all())
        aaa = labeled[labeled["ticker"] == "AAA"].sort_values("decision_time_utc")
        gaps = pd.to_datetime(aaa["decision_time_utc"], utc=True).diff().dropna()
        self.assertTrue(gaps.ge(pd.Timedelta(minutes=15)).all())


def _ticker_bars(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    closes = {
        "AAA": [100, 100, 102, 103, 104, 105, 106],
        "BBB": [100, 100, 101, 101.5, 102, 102.5, 103],
        "CCC": [100, 100, 99, 98.5, 98, 97.5, 97],
    }
    for ticker, values in closes.items():
        for index, timestamp in enumerate(times):
            close = values[index]
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100.0 if index == 1 else close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000,
                    "atr_14": 1.0,
                    "primary_benchmark": "XLK",
                    "universe_snapshot_id": "snapshot-1",
                    "price_feed": "sip",
                    "technical_feature": index / 10,
                }
            )
    return pd.DataFrame(rows)


def _benchmark_bars(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, closes in {"QQQ": [100, 100, 101, 101, 102, 102, 103], "XLK": [100, 100, 100.5, 101, 101, 102, 102]}.items():
        for index, timestamp in enumerate(times):
            close = closes[index]
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": 100.0 if index == 1 else close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 10_000,
                }
            )
    return pd.DataFrame(rows)


def _ticker_bars_extended(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, move in {"AAA": 0.2, "BBB": 0.1, "CCC": -0.1}.items():
        for index, timestamp in enumerate(times):
            close = 100 + move * index
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": close,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1_000,
                    "atr_14": 1.0,
                    "primary_benchmark": "XLK",
                    "universe_snapshot_id": "snapshot-1",
                    "price_feed": "sip",
                }
            )
    return pd.DataFrame(rows)


def _benchmark_bars_extended(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for ticker, move in {"QQQ": 0.05, "XLK": 0.04}.items():
        for index, timestamp in enumerate(times):
            close = 100 + move * index
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": close,
                    "high": close,
                    "low": close,
                    "close": close,
                    "volume": 10_000,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
