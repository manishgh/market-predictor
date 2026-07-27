from __future__ import annotations

import copy
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_dataset import (
    build_one_minute_requirements,
    build_requirement_window_bridge,
    extract_specialist_setups,
    merge_one_minute_requirements,
    restrict_to_complete_benchmark_grid,
    specialist_source_projection,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]


class IntradaySpecialistDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.config = load_intraday_specialist_research_config(
            ROOT / "configs" / "intraday_specialist_research.toml"
        )

    def test_projection_excludes_every_future_defined_column(self) -> None:
        projection = specialist_source_projection(self.config)
        prohibited = (
            "entry_",
            "exit_",
            "future",
            "label",
            "target",
            "stop",
            "outcome",
            "mfe",
            "mae",
            "net_return",
            "path_",
            "bars_to_",
            "ranking_",
        )
        self.assertFalse(
            [
                column
                for column in projection
                if any(marker in column.lower() for marker in prohibited)
            ]
        )

    def test_completed_bar_cutoff_is_five_minutes_after_source_timestamp(
        self,
    ) -> None:
        frame = _source_frame(self.config)
        setups = extract_specialist_setups(
            frame,
            config=self.config,
            source_dataset_fingerprint="f" * 64,
        )
        selected = setups["INTRADAY.GAP_CONTINUATION.60M.V1"]
        self.assertEqual(len(selected), 1)
        self.assertEqual(
            selected.iloc[0]["decision_time_utc"],
            pd.Timestamp("2026-06-01T14:01:00Z"),
        )
        self.assertEqual(
            selected.iloc[0]["feature_available_at_utc"],
            pd.Timestamp("2026-06-01T14:00:30Z"),
        )
        self.assertEqual(
            selected.iloc[0]["source_bar_start_utc"],
            pd.Timestamp("2026-06-01T13:55:00Z"),
        )

    def test_future_column_poison_cannot_change_setup_identity(self) -> None:
        frame = _source_frame(self.config)
        baseline = extract_specialist_setups(
            frame,
            config=self.config,
            source_dataset_fingerprint="f" * 64,
        )
        poisoned = frame.copy()
        poisoned["net_return_60m"] = [999.0, -999.0]
        poisoned["path_outcome"] = ["target", "stop"]
        replay = extract_specialist_setups(
            poisoned,
            config=self.config,
            source_dataset_fingerprint="f" * 64,
        )
        for strategy_id in baseline:
            self.assertEqual(
                baseline[strategy_id]["setup_id"].tolist(),
                replay[strategy_id]["setup_id"].tolist(),
            )

    def test_source_bar_start_convention_is_fail_closed(self) -> None:
        frame = _source_frame(self.config)
        frame["decision_time_utc"] = (
            frame["decision_time_utc"] + pd.Timedelta(minutes=5)
        )
        with self.assertRaises(DataReadinessError):
            extract_specialist_setups(
                frame,
                config=self.config,
                source_dataset_fingerprint="f" * 64,
            )

    def test_exact_requirements_include_stock_spy_qqq_and_sector(self) -> None:
        frame = _source_frame(self.config)
        setup = extract_specialist_setups(
            frame,
            config=self.config,
            source_dataset_fingerprint="f" * 64,
        )["INTRADAY.GAP_CONTINUATION.60M.V1"]
        grid = _minute_grid()
        requirements = build_one_minute_requirements(
            setup,
            minimum_warmup_bars=130,
            regular_minute_grid=grid,
        )
        self.assertEqual(
            set(requirements["ticker"]),
            {"AAA", "QQQ", "SPY", "XLK"},
        )
        self.assertTrue(requirements["price_feed"].eq("sip").all())
        self.assertTrue(requirements["adjustment"].eq("all").all())
        self.assertTrue(requirements["timeframe"].eq("1m").all())
        label_requirements = requirements[
            requirements["segment_kind"].eq("label")
        ]
        self.assertTrue(
            label_requirements["requested_end_utc"].eq(
                pd.Timestamp("2026-06-01T15:01:00Z")
            ).all()
        )
        self.assertGreaterEqual(
            int(requirements["planned_warmup_bars"].min()),
            130,
        )

    def test_overlapping_requirements_merge_per_ticker(self) -> None:
        requirements = pd.DataFrame(
            {
                "requirement_id": ["a", "b", "c"],
                "ticker": ["AAA", "AAA", "AAA"],
                "roles_json": ['["stock"]'] * 3,
                "session_date_et": [
                    pd.Timestamp("2026-06-01").date(),
                    pd.Timestamp("2026-06-01").date(),
                    pd.Timestamp("2026-06-03").date(),
                ],
                "requested_start_utc": pd.to_datetime(
                    [
                        "2026-06-01T13:00:00Z",
                        "2026-06-01T13:30:00Z",
                        "2026-06-03T13:00:00Z",
                    ],
                    utc=True,
                ),
                "requested_end_utc": pd.to_datetime(
                    [
                        "2026-06-01T14:00:00Z",
                        "2026-06-01T15:00:00Z",
                        "2026-06-03T14:00:00Z",
                    ],
                    utc=True,
                ),
            }
        )
        merged = merge_one_minute_requirements(requirements)
        self.assertEqual(len(merged), 2)
        self.assertEqual(merged.iloc[0]["requirement_count"], 2)
        bridge = build_requirement_window_bridge(merged)
        self.assertEqual(set(bridge["requirement_id"]), {"a", "b", "c"})
        self.assertFalse(bridge["requirement_id"].duplicated().any())

    def test_incomplete_benchmark_grid_is_removed_not_imputed(self) -> None:
        timestamps = pd.to_datetime(
            ["2026-06-01T14:00:00Z", "2026-06-01T14:05:00Z"],
            utc=True,
        )
        technical = pd.DataFrame(
            {
                "ticker": ["AAA", "AAA"],
                "primary_benchmark": ["XLK", "XLK"],
                "timestamp": timestamps,
            }
        )
        benchmarks = pd.DataFrame(
            {
                "ticker": ["SPY", "QQQ", "XLK", "SPY", "XLK"],
                "timestamp": [
                    timestamps[0],
                    timestamps[0],
                    timestamps[0],
                    timestamps[1],
                    timestamps[1],
                ],
            }
        )
        selected, removed = restrict_to_complete_benchmark_grid(
            technical,
            benchmarks,
        )
        self.assertEqual(len(selected), 1)
        self.assertEqual(removed, 1)

    def test_setup_spacing_suppresses_overlapping_same_strategy_rows(self) -> None:
        payload = self.config.model_dump(mode="python")
        payload["strategies"] = copy.deepcopy(payload["strategies"])
        strategy = payload["strategies"][
            "INTRADAY.GAP_CONTINUATION.60M.V1"
        ]
        strategy["first_decision_minute_et"] = 570
        strategy["last_decision_minute_et_exclusive"] = 700
        config = IntradaySpecialistResearchConfig.model_validate(payload)
        frame = _source_frame(config)
        second = frame.iloc[[0]].copy()
        for column in (
            "timestamp",
            "decision_time_utc",
            "feature_available_at_utc",
        ):
            second[column] = second[column] + pd.Timedelta(minutes=30)
        frame = pd.concat([frame.iloc[[0]], second], ignore_index=True)
        selected = extract_specialist_setups(
            frame,
            config=config,
            source_dataset_fingerprint="f" * 64,
        )["INTRADAY.GAP_CONTINUATION.60M.V1"]
        self.assertEqual(len(selected), 1)


def _source_frame(config: IntradaySpecialistResearchConfig) -> pd.DataFrame:
    projection = specialist_source_projection(config)
    timestamps = pd.to_datetime(
        ["2026-06-01T13:55:00Z", "2026-06-01T16:55:00Z"],
        utc=True,
    )
    rows: list[dict[str, object]] = []
    for index, timestamp in enumerate(timestamps):
        row: dict[str, object] = {
            column: 1.0 for column in projection
        }
        row.update(
            {
                "ticker": "AAA",
                "timestamp": timestamp,
                "decision_time_utc": timestamp,
                "feature_available_at_utc": timestamp,
                "session_date_et": timestamp.tz_convert(
                    "America/New_York"
                ).date(),
                "primary_benchmark": "XLK",
                "universe_snapshot_id": "snapshot",
                "sector": "Technology",
                "industry": "Software",
                "market_cap_bucket": "large",
                "liquidity_bucket": "high",
                "open": 100.0,
                "high": 101.0,
                "low": 99.0,
                "close": 100.8,
                "volume": 100_000,
                "atr_14": 1.0,
                "atr_pct": 0.01,
                "session_vwap": 100.0,
                "price_feed": "sip",
                "adjustment": "all",
                "overnight_gap": 0.03 if index == 0 else 0.0,
                "relative_volume_same_minute_20d": 1.5,
                "dist_session_vwap": 0.005,
                "return_1bar": 0.002,
                "return_3bar": 0.004,
                "close_location_5m": 0.9,
                "cross_section_eligible": 1,
            }
        )
        rows.append(row)
    return pd.DataFrame(rows)


def _minute_grid() -> pd.DatetimeIndex:
    prior = pd.date_range(
        "2026-05-29T13:30:00Z",
        periods=390,
        freq="1min",
    )
    current = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=390,
        freq="1min",
    )
    return prior.append(current)


if __name__ == "__main__":
    unittest.main()
