from __future__ import annotations

import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import pandas as pd

from market_predictor.canonical.reconciliation import ASSIGNMENT_COLUMNS
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.dataset import prepare_swing_benchmark_bars
from market_predictor.swing.specialist_contracts import (
    load_swing_specialist_research_config,
)
from market_predictor.swing.specialist_dataset import (
    CatalystLineageData,
    build_swing_specialist_dataset,
    build_swing_specialist_dataset_bundle,
    join_catalyst_lineage,
)
from market_predictor.swing.strategy_labels import (
    load_swing_strategy_label_policy,
)
from market_predictor.v3.errors import DataReadinessError
from tests.test_swing_strategy_labels import _fixture

ROOT = Path(__file__).resolve().parents[1]


class SwingSpecialistContractTests(unittest.TestCase):
    def test_frozen_policy_respects_catalog_and_experiment_budget(self) -> None:
        policy = load_swing_specialist_research_config(
            ROOT / "configs" / "swing_specialist_research.toml"
        )

        self.assertEqual(len(policy.strategies), 6)
        self.assertEqual(len(policy.feature_profiles), 3)
        self.assertEqual(len(policy.sha256()), 64)
        self.assertLessEqual(
            max(
                strategy.experiment_count()
                for strategy in policy.strategies.values()
            ),
            12,
        )
        self.assertEqual(
            policy.strategies[
                "SWING.CATALYST_DRIFT.5D.V1"
            ].experiment_count(),
            7,
        )


class SwingSpecialistDatasetTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.research = load_swing_specialist_research_config(
            ROOT / "configs" / "swing_specialist_research.toml"
        )
        cls.strategy_policy = load_swing_strategy_label_policy(
            ROOT / "configs" / "swing_strategy_labels.toml"
        )
        cls.dataset_config = SwingDatasetConfig(
            feature_profile="technical_market",
            required_ticker_sources=(),
            required_global_sources=(),
        )

    def test_canonical_benchmarks_are_normalized_for_strategy_labels(
        self,
    ) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["spy", "SPY"],
                "timeframe": ["1d", "1d"],
                "bar_start_utc": [
                    "2025-01-02T05:00:00Z",
                    "2025-01-03T05:00:00Z",
                ],
                "bar_end_utc": [
                    "2025-01-03T04:59:59Z",
                    "2025-01-04T04:59:59Z",
                ],
                "available_at_utc": [
                    "2025-01-03T05:05:00Z",
                    "2025-01-04T05:05:00Z",
                ],
                "open": [100.0, 101.0],
                "high": [102.0, 103.0],
                "low": [99.0, 100.0],
                "close": [101.0, 102.0],
                "volume": [1_000, 1_100],
                "price_feed": ["sip", "sip"],
                "adjustment": ["all", "all"],
            }
        )

        result = prepare_swing_benchmark_bars(frame)

        self.assertEqual(result["ticker"].tolist(), ["SPY", "SPY"])
        self.assertEqual(
            [str(value) for value in result["session_date_et"]],
            ["2025-01-02", "2025-01-03"],
        )
        self.assertIsNotNone(result["bar_start_utc"].dt.tz)

    def test_bundle_resume_reports_verified_eligible_rows(self) -> None:
        progress: list[object] = []
        record = {
            "strategy_id": "SWING.TEST.5D.V1",
            "label_eligible_rows": 17,
        }
        with (
            TemporaryDirectory() as directory,
            patch(
                "market_predictor.swing.specialist_dataset.STRATEGY_IDS",
                ("SWING.TEST.5D.V1",),
            ),
            patch(
                "market_predictor.swing.specialist_dataset._load_existing_dataset",
                return_value=record,
            ),
            patch(
                "market_predictor.swing.specialist_dataset.assert_memory_budget"
            ),
            patch(
                "market_predictor.swing.specialist_dataset.assert_peak_memory_budget"
            ),
            patch(
                "market_predictor.swing.specialist_dataset._validate_dataset_bundle_files"
            ),
        ):
            result = build_swing_specialist_dataset_bundle(
                pd.DataFrame(),
                pd.DataFrame(),
                out_dir=Path(directory),
                dataset_config=self.dataset_config,
                strategy_policy=self.strategy_policy,
                research_config=self.research,
                catalyst_audit={},
                input_hashes={"fixture": "a" * 64},
                progress=progress.append,
            )

        self.assertEqual(result["status"], "complete")
        self.assertEqual(result["rows"], 17)
        self.assertEqual(
            progress,
            [
                {
                    "strategy_id": "SWING.TEST.5D.V1",
                    "status": "resumed",
                    "rows": 17,
                }
            ],
        )

    def test_verified_coverage_distinguishes_zero_events_from_missing(self) -> None:
        decisions = _decision_rows()
        coverage = pd.DataFrame(
            {
                "security_id": ["security:aaa"],
                "requested_start_utc": [
                    pd.Timestamp("2024-12-29", tz="UTC")
                ],
                "requested_end_utc": [
                    pd.Timestamp("2025-01-03", tz="UTC")
                ],
                "status": ["observed"],
                "coverage_state": ["observed_empty"],
                "missingness_known": [True],
                "training_eligible": [True],
            }
        )
        lineage = CatalystLineageData(
            assignments=pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            coverage=coverage,
            lineage_sha256="a" * 64,
            manifest_sha256="b" * 64,
            request_sha256="c" * 64,
            observed_chunks=1,
        )

        joined, audit = join_catalyst_lineage(decisions, lineage)

        self.assertEqual(
            joined["catalyst_source_complete"].tolist(),
            [True, True, False],
        )
        self.assertEqual(joined.loc[0, "event_count_3d"], 0)
        self.assertTrue(pd.isna(joined.loc[2, "event_count_3d"]))
        self.assertEqual(audit["source_complete_rows"], 2)

    def test_overlapping_source_coverage_fails_closed(self) -> None:
        decisions = _decision_rows().iloc[:1].copy()
        coverage = pd.DataFrame(
            {
                "security_id": ["security:aaa", "security:aaa"],
                "requested_start_utc": [
                    pd.Timestamp("2025-01-01", tz="UTC"),
                    pd.Timestamp("2025-01-02", tz="UTC"),
                ],
                "requested_end_utc": [
                    pd.Timestamp("2025-01-04", tz="UTC"),
                    pd.Timestamp("2025-01-05", tz="UTC"),
                ],
                "status": ["observed", "observed"],
                "coverage_state": [
                    "observed_complete",
                    "observed_complete",
                ],
                "missingness_known": [True, True],
                "training_eligible": [True, True],
            }
        )
        lineage = CatalystLineageData(
            assignments=pd.DataFrame(columns=ASSIGNMENT_COLUMNS),
            coverage=coverage,
            lineage_sha256="a" * 64,
            manifest_sha256="b" * 64,
            request_sha256="c" * 64,
            observed_chunks=2,
        )

        with self.assertRaisesRegex(
            DataReadinessError,
            "overlapping catalyst coverage",
        ):
            join_catalyst_lineage(decisions, lineage)

    def test_catalyst_and_verified_no_event_setups_become_eligible(self) -> None:
        features, benchmarks = _fixture()
        features["universe_snapshot_id"] = "fixture-universe"
        features["market_regime"] = "neutral"
        features["sector"] = "Technology"
        features["timeframe"] = "1d"
        features["prediction_cutoff_policy_id"] = "fixture-cutoff"
        catalyst_audit = {
            "catalyst_lineage_sha256": "a" * 64,
            "event_aggregate_sha256": "b" * 64,
        }

        catalyst, catalyst_checks, catalyst_summary = (
            build_swing_specialist_dataset(
                features,
                benchmarks,
                strategy_id="SWING.CATALYST_DRIFT.5D.V1",
                dataset_config=self.dataset_config,
                strategy_policy=self.strategy_policy,
                research_config=self.research,
                catalyst_audit=catalyst_audit,
            )
        )
        reversal, reversal_checks, reversal_summary = (
            build_swing_specialist_dataset(
                features,
                benchmarks,
                strategy_id="SWING.SHORT_TERM_REVERSAL.3D.V1",
                dataset_config=self.dataset_config,
                strategy_policy=self.strategy_policy,
                research_config=self.research,
                catalyst_audit=catalyst_audit,
            )
        )

        self.assertTrue(catalyst_checks.passed)
        self.assertTrue(reversal_checks.passed)
        self.assertGreater(catalyst_summary["label_eligible_rows"], 0)
        self.assertGreater(reversal_summary["label_eligible_rows"], 0)
        self.assertTrue(catalyst["catalyst_source_complete"].all())
        self.assertTrue(reversal["catalyst_source_complete"].all())
        self.assertTrue(catalyst["event_count_3d"].gt(0).all())
        self.assertTrue(reversal["event_count_3d"].eq(0).all())
        self.assertTrue(
            catalyst["feature_available_at_utc"].le(
                catalyst["decision_time_utc"]
            ).all()
        )


def _decision_rows() -> pd.DataFrame:
    decision = pd.to_datetime(
        [
            "2025-01-01T22:00:00Z",
            "2025-01-02T22:00:00Z",
            "2025-01-03T22:00:00Z",
        ]
    )
    start = decision - pd.Timedelta(hours=7.5)
    end = decision - pd.Timedelta(hours=1)
    return pd.DataFrame(
        {
            "ticker": ["AAA"] * 3,
            "security_id": ["security:aaa"] * 3,
            "session_date_et": decision.date,
            "decision_group_id": [
                f"group-{index}" for index in range(3)
            ],
            "decision_time_utc": decision,
            "feature_available_at_utc": end,
            "bar_start_utc": start,
            "bar_end_utc": end,
            "available_at_utc": end,
            "timeframe": ["1d"] * 3,
            "prediction_cutoff_policy_id": ["fixture"] * 3,
        }
    )


if __name__ == "__main__":
    unittest.main()
