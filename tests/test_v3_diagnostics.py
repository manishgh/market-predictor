from __future__ import annotations

import unittest

import pandas as pd

from market_predictor.v3.diagnostics import FailureAttributionConfig, build_failure_attribution
from market_predictor.v3.errors import DataReadinessError, LeakageAuditError


class V3FailureAttributionTests(unittest.TestCase):
    def test_reports_identical_top_k_across_horizons_and_strata(self) -> None:
        predictions, development = _evidence()

        report, strata, selected = build_failure_attribution(
            predictions,
            development,
            dataset_fingerprint="a" * 64,
            config=FailureAttributionConfig(
                top_k=2,
                bootstrap_iterations=100,
                minimum_stratum_rows=1,
                minimum_stratum_sessions=2,
            ),
        )

        self.assertTrue(report["diagnostic_only"])
        self.assertFalse(report["promotion_eligible"])
        self.assertFalse(report["shadow_data_accessed"])
        self.assertEqual(set(report["scope_summary"]), {"walk_forward", "ticker_holdout"})
        self.assertEqual(len(selected), 16)
        self.assertEqual(set(strata["audit_scope"]), {"walk_forward", "ticker_holdout"})
        walk = report["scope_summary"]["walk_forward"]
        self.assertGreater(walk["outcomes"]["net_excess_qqq_60m"]["selection_delta"], 0)
        self.assertEqual(len(walk["score_deciles"]), 5)

    def test_rejects_prediction_target_mismatch(self) -> None:
        predictions, development = _evidence()
        development.loc[0, "ranking_target"] += 0.01

        with self.assertRaisesRegex(DataReadinessError, "ranking targets differ"):
            build_failure_attribution(
                predictions,
                development,
                dataset_fingerprint="a" * 64,
                config=FailureAttributionConfig(top_k=2, bootstrap_iterations=100),
            )

    def test_rejects_shadow_rows(self) -> None:
        predictions, development = _evidence()
        predictions.loc[0, "decision_time_utc"] = pd.Timestamp("2026-07-09T14:00:00Z")

        with self.assertRaises(LeakageAuditError):
            build_failure_attribution(
                predictions,
                development,
                dataset_fingerprint="a" * 64,
                config=FailureAttributionConfig(top_k=2, bootstrap_iterations=100),
            )


def _evidence() -> tuple[pd.DataFrame, pd.DataFrame]:
    prediction_rows: list[dict[str, object]] = []
    development_rows: list[dict[str, object]] = []
    for session_index, day in enumerate(("2026-01-05", "2026-01-06")):
        for group_index, hour in enumerate((15, 19)):
            decision = pd.Timestamp(f"{day}T{hour:02d}:00:00Z")
            group_id = f"{day}-{hour}"
            for ticker_index in range(5):
                ticker = f"T{ticker_index}"
                excess = (ticker_index - 2) / 1_000
                development_rows.append(
                    {
                        "ticker": ticker,
                        "decision_time_utc": decision,
                        "decision_group_id": group_id,
                        "session_date_et": day,
                        "sector": "technology" if ticker_index < 3 else "healthcare",
                        "industry": "test",
                        "market_cap_bucket": "large",
                        "liquidity_bucket": "liquid",
                        "session_progress": 0.2 + group_index * 0.5,
                        "regime_risk_on": 1 if session_index == 0 else 0,
                        "regime_risk_off": 1 if session_index == 1 else 0,
                        "regime_high_volatility": 0,
                        "xs_rank_dollar_volume": (ticker_index + 1) / 5,
                        "xs_rank_atr_pct": (5 - ticker_index) / 5,
                        "mfe_60m": abs(excess) + 0.002,
                        "mae_60m": -abs(excess),
                        "ranking_target": excess,
                        "net_excess_qqq_30m": excess / 2,
                        "net_excess_qqq_60m": excess,
                        "net_excess_qqq_120m": excess * 1.5,
                        "net_excess_qqq_to_close": excess * 2,
                        "net_excess_sector_30m": excess / 2,
                        "net_excess_sector_60m": excess,
                        "net_excess_sector_120m": excess * 1.5,
                        "net_excess_sector_to_close": excess * 2,
                        "net_return_30m": excess / 2 + 0.0001,
                        "net_return_60m": excess + 0.0001,
                        "net_return_120m": excess * 1.5 + 0.0001,
                        "net_return_to_close": excess * 2 + 0.0001,
                    }
                )
                for scope in ("walk_forward", "ticker_holdout"):
                    prediction_rows.append(
                        {
                            "ticker": ticker,
                            "decision_time_utc": decision,
                            "decision_group_id": group_id,
                            "session_date_et": day,
                            "audit_scope": scope,
                            "family": "R1",
                            "model_run_id": "r1-run",
                            "score": ticker_index / 5,
                            "ranking_target": excess,
                            "fold": group_index,
                        }
                    )
    return pd.DataFrame(prediction_rows), pd.DataFrame(development_rows)


if __name__ == "__main__":
    unittest.main()
