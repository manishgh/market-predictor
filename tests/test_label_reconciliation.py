from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.label_reconciliation import (
    label_material_sha256,
    replay_mismatch_count,
    stamp_label_reconciliation,
    stamped_material_hash_is_valid,
)


class LabelReconciliationTests(unittest.TestCase):
    def test_material_hash_is_order_independent_and_content_sensitive(self) -> None:
        frame = _material_frame()
        reordered = frame.iloc[::-1].reset_index(drop=True)
        mutated = frame.copy()
        mutated.loc[0, "net_return"] = 0.03

        expected = _material_hash(frame)

        self.assertEqual(_material_hash(reordered), expected)
        self.assertNotEqual(_material_hash(mutated), expected)

    def test_stamp_and_replay_fail_on_material_mutation(self) -> None:
        frame = _material_frame()
        stamped = stamp_label_reconciliation(
            frame,
            identity_columns=("ticker", "decision_time_utc"),
            material_columns=("exit_time_utc", "net_return", "outcome"),
            label_policy_sha256="a" * 64,
        )
        reordered = frame.iloc[::-1].reset_index(drop=True)
        mutated = reordered.copy()
        mutated.loc[0, "outcome"] = "stop_first"

        self.assertTrue(
            stamped_material_hash_is_valid(
                stamped,
                identity_columns=("ticker", "decision_time_utc"),
                material_columns=("exit_time_utc", "net_return", "outcome"),
            )
        )
        self.assertEqual(
            replay_mismatch_count(
                frame,
                reordered,
                identity_columns=("ticker", "decision_time_utc"),
                material_columns=("exit_time_utc", "net_return", "outcome"),
            ),
            0,
        )
        self.assertGreater(
            replay_mismatch_count(
                frame,
                mutated,
                identity_columns=("ticker", "decision_time_utc"),
                material_columns=("exit_time_utc", "net_return", "outcome"),
            ),
            0,
        )


def _material_hash(frame: pd.DataFrame) -> str:
    return label_material_sha256(
        frame,
        identity_columns=("ticker", "decision_time_utc"),
        material_columns=("exit_time_utc", "net_return", "outcome"),
    )


def _material_frame() -> pd.DataFrame:
    decisions = pd.to_datetime(
        ["2026-01-05T15:00:00Z", "2026-01-05T15:05:00Z"],
        utc=True,
    )
    return pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "decision_time_utc": decisions,
            "exit_time_utc": decisions + pd.Timedelta(minutes=5),
            "net_return": [0.01, -0.02],
            "outcome": ["target_first", "timeout"],
        }
    )
