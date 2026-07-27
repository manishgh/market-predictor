from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.intraday.specialist_contracts import (
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_training_data import (
    SPECIALIST_TRAINING_ROW_SCHEMA,
    _validate_strategy_training_shard,
    build_clock_grid_features,
    build_strategy_training_rows,
    load_clock_grid_for_requirements,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "intraday_specialist_research.toml"


class IntradaySpecialistTrainingDataTests(unittest.TestCase):
    def test_clock_grid_marks_and_causally_fills_no_trade_minute(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bars.parquet"
            bars = _observed_bars(
                tickers=("AAA",),
                periods=3,
            )
            bars = bars[
                bars["bar_start_utc"].ne(
                    pd.Timestamp("2026-06-01T13:31:00Z")
                )
            ]
            bars.to_parquet(path, index=False)
            requirements = _requirements(("AAA",), periods=3)
            records = {
                ("AAA", "2026-06-01"): {
                    "path": str(path),
                    "requested_start_utc": "2026-06-01T13:30:00+00:00",
                    "requested_end_utc": "2026-06-01T13:33:00+00:00",
                }
            }

            dense = load_clock_grid_for_requirements(
                requirements,
                artifact_records=records,
                finalization_delay_seconds=30,
            )

            missing = dense.iloc[1]
            self.assertFalse(bool(missing["observed_eligible_trade"]))
            self.assertEqual(float(missing["close"]), 100.0)
            self.assertEqual(float(missing["volume"]), 0.0)
            self.assertEqual(
                missing["available_at_utc"],
                pd.Timestamp("2026-06-01T13:32:30Z"),
            )

    def test_executable_label_requires_observed_entry_and_benchmarks(
        self,
    ) -> None:
        config = load_intraday_specialist_research_config(POLICY)
        observed = _observed_bars(
            tickers=("AAA", "SPY", "QQQ", "XLK"),
            periods=240,
        )
        requirements = _requirements(
            ("AAA", "SPY", "QQQ", "XLK"),
            periods=240,
        )
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "bars.parquet"
            observed.to_parquet(path, index=False)
            records = {
                (ticker, "2026-06-01"): {
                    "path": str(path),
                    "requested_start_utc": "2026-06-01T13:30:00+00:00",
                    "requested_end_utc": "2026-06-01T17:30:00+00:00",
                }
                for ticker in ("AAA", "SPY", "QQQ", "XLK")
            }
            dense = load_clock_grid_for_requirements(
                requirements,
                artifact_records=records,
                finalization_delay_seconds=30,
            )
        features = build_clock_grid_features(
            dense,
            minimum_warmup_bars=130,
        )
        setup = pd.DataFrame(
            [
                {
                    "setup_id": "setup",
                    "strategy_id": "INTRADAY.VWAP_REVERSION.30M.V1",
                    "ticker": "AAA",
                    "session_date_et": pd.Timestamp(
                        "2026-06-01"
                    ).date(),
                    "session_minute_et": 12 * 60,
                    "decision_time_utc": pd.Timestamp(
                        "2026-06-01T16:00:00Z"
                    ),
                    "feature_available_at_utc": pd.Timestamp(
                        "2026-06-01T15:59:30Z"
                    ),
                    "atr_14_price_5m": 1.0,
                    "primary_benchmark": "XLK",
                }
            ]
        )

        labeled = build_strategy_training_rows(
            setup,
            bars=dense,
            one_minute_features=features,
            horizon_minutes=30,
            config=config,
        )

        self.assertTrue(bool(labeled.iloc[0]["label_eligible"]))
        self.assertEqual(
            labeled.iloc[0]["label_ineligible_reason"],
            "eligible",
        )

    def test_resume_shard_must_match_expected_setup_ids(self) -> None:
        expected = pd.DataFrame({"setup_id": ["expected"]})
        shard = pd.DataFrame(
            {
                "training_schema_version": [
                    SPECIALIST_TRAINING_ROW_SCHEMA
                ],
                "setup_id": ["stale"],
                "strategy_id": [
                    "INTRADAY.VWAP_REVERSION.30M.V1"
                ],
                "label_eligible": [True],
                "label_ineligible_reason": ["eligible"],
            }
        )

        with self.assertRaises(DataReadinessError):
            _validate_strategy_training_shard(
                shard,
                expected_setups=expected,
                strategy_id="INTRADAY.VWAP_REVERSION.30M.V1",
                path=Path("stale.parquet"),
            )


def _observed_bars(
    *,
    tickers: tuple[str, ...],
    periods: int,
) -> pd.DataFrame:
    rows = []
    timestamps = pd.date_range(
        "2026-06-01T13:30:00Z",
        periods=periods,
        freq="1min",
    )
    for ticker in tickers:
        for timestamp in timestamps:
            rows.append(
                {
                    "ticker": ticker,
                    "bar_start_utc": timestamp,
                    "bar_end_utc": timestamp + pd.Timedelta(minutes=1),
                    "available_at_utc": timestamp
                    + pd.Timedelta(minutes=1, seconds=30),
                    "open": 100.0,
                    "high": 100.2,
                    "low": 99.8,
                    "close": 100.0,
                    "volume": 10_000,
                    "price_feed": "sip",
                    "adjustment": "all",
                }
            )
    return pd.DataFrame(rows)


def _requirements(
    tickers: tuple[str, ...],
    *,
    periods: int,
) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "setup_id": "setup",
                "ticker": ticker,
                "session_date_et": pd.Timestamp("2026-06-01").date(),
                "requested_start_utc": pd.Timestamp(
                    "2026-06-01T13:30:00Z"
                ),
                "requested_end_utc": pd.Timestamp(
                    "2026-06-01T13:30:00Z"
                )
                + pd.Timedelta(minutes=periods),
            }
            for ticker in tickers
        ]
    )


if __name__ == "__main__":
    unittest.main()
