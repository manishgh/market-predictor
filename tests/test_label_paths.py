from __future__ import annotations

import unittest

import numpy as np

from market_predictor.label_paths import (
    evaluate_intraday_barrier_paths,
    evaluate_swing_paths,
    open_close_return,
)


class LabelPathTests(unittest.TestCase):
    def test_swing_path_uses_next_open_horizon_close_and_cost(self) -> None:
        result = evaluate_swing_paths(
            entry_price=np.array([100.0]),
            exit_price=np.array([110.0]),
            path_high=np.array([[102.0, 115.0, 111.0]]),
            path_low=np.array([[98.0, 99.0, 105.0]]),
            round_trip_cost_bps=17.0,
        )

        self.assertAlmostEqual(float(result.gross_return[0]), 0.10)
        self.assertAlmostEqual(float(result.net_return[0]), 0.0983)
        self.assertAlmostEqual(float(result.mfe[0]), 0.15)
        self.assertAlmostEqual(float(result.mae[0]), -0.02)

    def test_intraday_target_stop_collision_is_stop_first(self) -> None:
        result = _intraday_result(
            high=[[102.0, 101.0]],
            low=[[98.0, 99.0]],
            close=[[100.0, 100.5]],
        )

        self.assertEqual(result.outcome[0], "stop_first")
        self.assertTrue(bool(result.stop_first[0]))
        self.assertEqual(int(result.outcome_offset[0]), 0)
        self.assertAlmostEqual(float(result.realized_price[0]), 99.0)

    def test_intraday_target_stop_and_timeout_paths(self) -> None:
        result = _intraday_result(
            high=[
                [101.0, 102.1, 102.2],
                [100.5, 100.8, 101.0],
                [100.5, 100.8, 101.0],
            ],
            low=[
                [99.5, 99.4, 99.8],
                [99.5, 98.9, 99.0],
                [99.5, 99.4, 99.2],
            ],
            close=[
                [100.5, 102.0, 102.1],
                [100.2, 99.0, 99.1],
                [100.2, 100.3, 100.4],
            ],
        )

        self.assertEqual(result.outcome.tolist(), ["target_first", "stop_first", "timeout"])
        self.assertEqual(result.outcome_offset.tolist(), [1, 1, 2])
        self.assertAlmostEqual(float(result.realized_price[0]), 102.0)
        self.assertAlmostEqual(float(result.realized_price[1]), 99.0)
        self.assertAlmostEqual(float(result.realized_price[2]), 100.4)
        self.assertAlmostEqual(float(result.net_return[2]), 0.0023)

    def test_open_close_return_rejects_invalid_prices(self) -> None:
        result = open_close_return(
            np.array([100.0, 0.0, np.nan]),
            np.array([101.0, 10.0, 10.0]),
        )

        self.assertAlmostEqual(float(result[0]), 0.01)
        self.assertTrue(np.isnan(result[1]))
        self.assertTrue(np.isnan(result[2]))


def _intraday_result(
    *,
    high: list[list[float]],
    low: list[list[float]],
    close: list[list[float]],
):
    shape = np.asarray(high, dtype=float).shape
    return evaluate_intraday_barrier_paths(
        path_open=np.full(shape, 100.0),
        path_high=np.asarray(high, dtype=float),
        path_low=np.asarray(low, dtype=float),
        path_close=np.asarray(close, dtype=float),
        entry_atr=np.ones(shape[0]),
        target_atr=2.0,
        stop_atr=1.0,
        round_trip_cost_bps=17.0,
    )
