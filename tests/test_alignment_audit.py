"""R3 P0-3: the swing alignment audit is computed, not fabricated zeros."""

from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.swing.model import _alignment_audit


class SwingAlignmentAuditTest(unittest.TestCase):
    def test_alignment_audit_detects_leakage_and_path_mismatch(self) -> None:
        base = pd.Timestamp("2026-01-05 21:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "decision_time_utc": [base, base, base],
                # row 1's feature is available after its decision -> leakage.
                "feature_available_at_utc": [base, base + pd.Timedelta(seconds=1), base],
                "feature_eligible": [True, True, True],
                "label_window_expected": [True, True, True],
                # row 2's expected label path is not exact -> path mismatch.
                "label_path_exact": [True, True, False],
                "future_excess_return_5d_vs_spy": [0.01, 0.01, 0.01],
                "future_excess_return_5d_vs_qqq": [0.01, 0.01, 0.01],
                "future_excess_return_5d_vs_sector": [0.01, 0.01, None],
            }
        )
        audit = _alignment_audit(frame).iloc[0]
        self.assertEqual(int(audit["future_feature_rows"]), 1)
        self.assertEqual(int(audit["label_path_mismatches"]), 1)
        # row 2 has a missing benchmark but is not exact, so it is not a benchmark mismatch.
        self.assertEqual(int(audit["benchmark_path_mismatches"]), 0)
        self.assertEqual(
            int(audit["alignment_error_total"]),
            int(audit["future_feature_rows"])
            + int(audit["label_path_mismatches"])
            + int(audit["benchmark_path_mismatches"]),
        )

    def test_clean_frame_has_no_alignment_errors(self) -> None:
        base = pd.Timestamp("2026-01-05 21:00", tz="UTC")
        frame = pd.DataFrame(
            {
                "decision_time_utc": [base, base],
                "feature_available_at_utc": [base, base],
                "feature_eligible": [True, True],
                "label_window_expected": [True, True],
                "label_path_exact": [True, True],
                "future_excess_return_5d_vs_spy": [0.01, 0.02],
                "future_excess_return_5d_vs_qqq": [0.01, 0.02],
                "future_excess_return_5d_vs_sector": [0.01, 0.02],
            }
        )
        audit = _alignment_audit(frame).iloc[0]
        self.assertEqual(int(audit["alignment_error_total"]), 0)


if __name__ == "__main__":
    unittest.main()
