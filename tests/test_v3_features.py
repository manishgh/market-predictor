from __future__ import annotations

import unittest
from time import perf_counter

import pandas as pd

from market_predictor.v3.errors import DataReadinessError
from market_predictor.v3.features import (
    V3_MICROSTRUCTURE_FEATURES,
    build_v3_features,
    build_v3_ticker_features,
    core_feature_columns,
    finalize_v3_cross_sectional_features,
)


class V3FeatureTests(unittest.TestCase):
    def setUp(self) -> None:
        self.times = pd.date_range("2026-07-08 14:30:00Z", periods=8, freq="5min")
        self.bars = _ticker_bars(self.times)
        self.benchmarks = _benchmark_bars(self.times)

    def test_cross_sectional_ranks_use_only_current_decision_group(self) -> None:
        features = build_v3_features(self.bars, self.benchmarks, minimum_cross_section=3)
        group = features[features["decision_time_utc"] == self.times[1]]
        ranks = dict(zip(group["ticker"], group["xs_rank_return_1bar"], strict=True))
        self.assertEqual(ranks, {"AAA": 1.0, "BBB": 0.5, "CCC": 0.0})
        self.assertTrue(group["cross_section_eligible"].eq(1).all())

    def test_future_bars_cannot_change_prior_features(self) -> None:
        baseline = build_v3_features(self.bars, self.benchmarks, minimum_cross_section=3)
        changed = self.bars.copy()
        changed.loc[changed["timestamp"] == self.times[-1], ["high", "low", "close", "volume"]] = [900, 1, 500, 9_000_000]
        rescored = build_v3_features(changed, self.benchmarks, minimum_cross_section=3)
        prior = baseline[baseline["decision_time_utc"] < self.times[-1]].reset_index(drop=True)
        prior_rescored = rescored[rescored["decision_time_utc"] < self.times[-1]].reset_index(drop=True)
        pd.testing.assert_frame_equal(prior[list(core_feature_columns())], prior_rescored[list(core_feature_columns())])

    def test_batch_and_truncated_live_history_have_feature_parity(self) -> None:
        batch = build_v3_features(self.bars, self.benchmarks, minimum_cross_section=3)
        cutoff = self.times[5]
        live = build_v3_features(
            self.bars[self.bars["timestamp"] <= cutoff],
            self.benchmarks[self.benchmarks["timestamp"] <= cutoff],
            minimum_cross_section=3,
        )
        batch_row = batch[batch["decision_time_utc"] == cutoff].reset_index(drop=True)
        live_row = live[live["decision_time_utc"] == cutoff].reset_index(drop=True)
        pd.testing.assert_frame_equal(batch_row[list(core_feature_columns())], live_row[list(core_feature_columns())])

    def test_source_availability_is_joined_as_of_collection_time(self) -> None:
        collected = self.times[0] + pd.Timedelta(minutes=2)
        availability = pd.DataFrame(
            {
                "ticker": ["AAA"],
                "source_family": ["reddit"],
                "available": [True],
                "row_count": [25],
                "first_available_at_utc": [self.times[0] - pd.Timedelta(days=1)],
                "last_available_at_utc": [self.times[0]],
                "collected_at_utc": [collected],
            }
        )
        features = build_v3_features(
            self.bars,
            self.benchmarks,
            source_availability=availability,
            minimum_cross_section=3,
        )
        aaa = features[features["ticker"] == "AAA"].sort_values("decision_time_utc")
        self.assertEqual(int(aaa.iloc[0]["source_available_reddit"]), 0)
        self.assertEqual(int(aaa.iloc[1]["source_available_reddit"]), 1)
        self.assertEqual(int(aaa.iloc[1]["source_rows_reddit"]), 25)

    def test_missing_microstructure_is_explicit_and_not_in_core(self) -> None:
        features = build_v3_features(self.bars, self.benchmarks, minimum_cross_section=3)
        self.assertTrue(features["microstructure_available"].eq(0).all())
        self.assertTrue(features["spread_pct"].isna().all())
        self.assertTrue(set(V3_MICROSTRUCTURE_FEATURES).isdisjoint(core_feature_columns()))

    def test_returns_reset_at_session_boundary(self) -> None:
        next_day = pd.date_range("2026-07-09 14:30:00Z", periods=2, freq="5min")
        bars = pd.concat([self.bars, _ticker_bars(next_day)], ignore_index=True)
        benchmarks = pd.concat([self.benchmarks, _benchmark_bars(next_day)], ignore_index=True)
        features = build_v3_features(bars, benchmarks, minimum_cross_section=3)
        opening = features[features["decision_time_utc"] == next_day[0]]
        self.assertTrue(opening["return_1bar"].isna().all())

    def test_missing_exact_sector_bar_fails_closed(self) -> None:
        benchmarks = self.benchmarks[
            ~((self.benchmarks["ticker"] == "XLK") & (self.benchmarks["timestamp"] == self.times[3]))
        ]
        with self.assertRaises(DataReadinessError):
            build_v3_features(self.bars, benchmarks, minimum_cross_section=3)

    def test_feature_build_performance_smoke(self) -> None:
        pieces = []
        for index in range(20):
            piece = self.bars.copy()
            piece["ticker"] = piece["ticker"].map(lambda ticker, suffix=index: f"{ticker}{suffix:02d}")
            pieces.append(piece)
        bars = pd.concat(pieces, ignore_index=True)
        started = perf_counter()
        features = build_v3_features(bars, self.benchmarks, minimum_cross_section=20)
        elapsed = perf_counter() - started
        self.assertEqual(len(features), len(bars))
        self.assertLess(elapsed, 8.0, f"480-row V3 feature smoke build took {elapsed:.2f}s")

    def test_sharded_feature_stages_match_single_pass(self) -> None:
        expected = build_v3_features(self.bars, self.benchmarks, minimum_cross_section=3)
        shards = [build_v3_ticker_features(group) for _, group in self.bars.groupby("ticker", sort=False)]
        actual = finalize_v3_cross_sectional_features(
            pd.concat(shards, ignore_index=True),
            self.benchmarks,
            minimum_cross_section=3,
        )
        pd.testing.assert_frame_equal(expected, actual)


def _ticker_bars(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    moves = {"AAA": 2.0, "BBB": 1.0, "CCC": -1.0}
    for ticker, move in moves.items():
        for index, timestamp in enumerate(times):
            close = 100.0 + move * index
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": close - move / 2,
                    "high": close + 0.5,
                    "low": close - 0.5,
                    "close": close,
                    "volume": 1000 + index * 100,
                    "primary_benchmark": "XLK",
                    "universe_snapshot_id": "snapshot-1",
                    "price_feed": "sip",
                }
            )
    return pd.DataFrame(rows)


def _benchmark_bars(times: pd.DatetimeIndex) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    moves = {"QQQ": 0.5, "SPY": 0.3, "XLK": 0.4}
    for ticker, move in moves.items():
        for index, timestamp in enumerate(times):
            close = 100.0 + move * index
            rows.append(
                {
                    "ticker": ticker,
                    "timestamp": timestamp,
                    "open": close,
                    "high": close + 0.1,
                    "low": close - 0.1,
                    "close": close,
                    "volume": 100_000,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
