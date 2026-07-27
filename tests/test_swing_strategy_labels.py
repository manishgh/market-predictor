from __future__ import annotations

import tomllib
import unittest
from json import loads
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

import numpy as np
import pandas as pd

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.swing import strategy_labels as strategy_label_module
from market_predictor.swing.contracts import SwingDatasetConfig
from market_predictor.swing.strategy_labels import (
    STRATEGY_IDS,
    SwingStrategyLabelPolicy,
    audit_swing_strategy_labels,
    build_strategy_setups,
    build_swing_strategy_label_bundle,
    build_swing_strategy_labels,
    load_swing_strategy_label_policy,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "swing_strategy_labels.toml"
DECISION_INDEX = 25


class SwingStrategyLabelPolicyTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_swing_strategy_label_policy(POLICY_PATH)
        cls.config = SwingDatasetConfig(
            feature_profile="technical_market",
            required_ticker_sources=(),
            required_global_sources=(),
        )

    def test_policy_freezes_distinct_strategy_contracts(self) -> None:
        self.assertEqual(set(self.policy.strategies), set(STRATEGY_IDS))
        self.assertEqual(
            self.policy.execution_policy_sha256,
            EXECUTION_POLICY_SHA256,
        )
        self.assertEqual(
            self.policy.strategies[
                "SWING.SHORT_TERM_REVERSAL.3D.V1"
            ].horizon_sessions,
            3,
        )
        self.assertEqual(
            self.policy.strategies[
                "SWING.BREAKOUT_EXPANSION.5D.V1"
            ].family,
            "breakout_expansion",
        )
        hashes = {
            self.policy.strategy_sha256(strategy_id)
            for strategy_id in STRATEGY_IDS
        }
        self.assertEqual(len(hashes), len(STRATEGY_IDS))

        with POLICY_PATH.open("rb") as handle:
            payload = tomllib.load(handle)
        payload["strategies"][
            "SWING.TIME_SERIES_MOMENTUM.5D.V1"
        ]["target_rule"] = "top_quintile_spy_excess_and_positive_net"
        with self.assertRaisesRegex(ValueError, "strategy contract mismatch"):
            SwingStrategyLabelPolicy.model_validate(payload)

        with POLICY_PATH.open("rb") as handle:
            execution_payload = tomllib.load(handle)
        execution_payload["entry_rule"] = "same_day_close"
        with self.assertRaises(ValueError):
            SwingStrategyLabelPolicy.model_validate(execution_payload)

    def test_setup_selection_is_invariant_to_future_bar_mutation(self) -> None:
        frame, _ = _fixture()
        original = build_strategy_setups(frame, self.policy)
        mutated = frame.copy()
        decision_date = frame.loc[
            (frame["ticker"].eq("AAA"))
            & (frame.groupby("ticker").cumcount().eq(DECISION_INDEX)),
            "session_date_et",
        ].iloc[0]
        future = pd.to_datetime(mutated["session_date_et"]).dt.date > decision_date
        mutated.loc[future, ["open", "high", "low", "close"]] *= 5.0
        replay = build_strategy_setups(mutated, self.policy)
        identity = (
            original["ticker"].eq("AAA")
            & original["session_date_et"].eq(decision_date)
        )
        replay_identity = (
            replay["ticker"].eq("AAA")
            & replay["session_date_et"].eq(decision_date)
        )
        columns = [
            "strategy_id",
            "setup_eligible",
            "setup_abstention_reason",
            "strategy_setup_policy_sha256",
        ]
        pd.testing.assert_frame_equal(
            original.loc[identity, columns].reset_index(drop=True),
            replay.loc[replay_identity, columns].reset_index(drop=True),
        )

    def test_future_feature_and_generic_label_inputs_fail_closed(self) -> None:
        frame, _ = _fixture()
        poisoned = frame.copy()
        poisoned.loc[0, "feature_available_at_utc"] = (
            poisoned.loc[0, "decision_time_utc"]
            + pd.Timedelta(seconds=1)
        )
        with self.assertRaisesRegex(
            DataReadinessError,
            "post-decision evidence",
        ):
            build_strategy_setups(poisoned, self.policy)

        generic = frame.copy()
        generic["target_net_positive_5d"] = 1
        with self.assertRaisesRegex(
            DataReadinessError,
            "generic or future labels",
        ):
            build_swing_strategy_labels(
                generic,
                _fixture()[1],
                dataset_config=self.config,
                policy=self.policy,
            )

    def test_distinct_labels_replay_and_charge_cost_once(self) -> None:
        frame, benchmarks = _fixture()
        labels = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )

        self.assertEqual(
            set(labels["strategy_id"]),
            set(STRATEGY_IDS),
        )
        eligible = labels["strategy_label_eligible"].astype(bool)
        np.testing.assert_allclose(
            labels.loc[eligible, "strategy_net_return"],
            labels.loc[eligible, "strategy_gross_return"]
            - labels.loc[
                eligible,
                "strategy_execution_cost_fraction",
            ],
            rtol=1e-10,
            atol=1e-12,
        )
        self.assertTrue(
            labels.loc[eligible, "strategy_excess_return_vs_spy"].notna().all()
        )
        self.assertTrue(
            labels.loc[
                eligible,
                "strategy_excess_return_vs_sector",
            ].notna().all()
        )
        audit = audit_swing_strategy_labels(
            frame,
            benchmarks,
            labels,
            dataset_config=self.config,
            policy=self.policy,
        )
        self.assertTrue(audit.passed)

        aaa_date = _decision_date(frame, "AAA")
        aaa = labels.loc[
            labels["ticker"].eq("AAA")
            & labels["session_date_et"].eq(aaa_date)
        ].set_index("strategy_id")
        self.assertTrue(
            bool(
                aaa.loc[
                    "SWING.CATALYST_DRIFT.5D.V1",
                    "setup_eligible",
                ]
            )
        )
        self.assertTrue(
            bool(
                aaa.loc[
                    "SWING.BREAKOUT_EXPANSION.5D.V1",
                    "setup_eligible",
                ]
            )
        )
        self.assertEqual(
            aaa.loc[
                "SWING.BREAKOUT_EXPANSION.5D.V1",
                "barrier_outcome",
            ],
            "target_first",
        )
        breakout = aaa.loc["SWING.BREAKOUT_EXPANSION.5D.V1"]
        self.assertAlmostEqual(
            float(breakout["exit_price"]),
            float(breakout["barrier_realized_price"]),
        )
        self.assertAlmostEqual(
            float(breakout["strategy_gross_return"]),
            float(breakout["exit_price"])
            / float(breakout["entry_price"])
            - 1.0,
        )
        self.assertFalse(bool(breakout["breakout_failed"]))
        aaa_indices = frame.index[frame["ticker"].eq("AAA")]
        entry_session = frame.loc[
            aaa_indices[DECISION_INDEX + 1],
            "session_date_et",
        ]
        exit_session = pd.Timestamp(
            breakout["exit_time_utc"]
        ).date()
        spy = benchmarks.loc[benchmarks["ticker"].eq("SPY")]
        expected_spy = (
            float(
                spy.loc[
                    spy["session_date_et"].eq(exit_session),
                    "close",
                ].iloc[0]
            )
            / float(
                spy.loc[
                    spy["session_date_et"].eq(entry_session),
                    "open",
                ].iloc[0]
            )
            - 1.0
        )
        self.assertAlmostEqual(
            float(breakout["strategy_spy_return"]),
            expected_spy,
            places=12,
        )
        self.assertEqual(
            pd.Timestamp(breakout["exit_time_utc"]),
            pd.Timestamp(
                frame.loc[
                    aaa_indices[
                        DECISION_INDEX
                        + int(breakout["barrier_outcome_session"])
                    ],
                    "bar_end_utc",
                ]
            ),
        )

        cross = labels.loc[
            labels["strategy_id"].eq(
                "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"
            )
            & labels["strategy_label_eligible"].fillna(False)
            & labels["strategy_target"].eq(1)
        ]
        self.assertTrue(
            cross["strategy_excess_return_vs_spy"].gt(0).all()
        )

        bbb_date = _decision_date(frame, "BBB")
        bbb = labels.loc[
            labels["ticker"].eq("BBB")
            & labels["session_date_et"].eq(bbb_date)
            & labels["strategy_id"].eq(
                "SWING.SHORT_TERM_REVERSAL.3D.V1"
            )
        ].iloc[0]
        self.assertTrue(bool(bbb["setup_eligible"]))
        self.assertEqual(bbb["strategy_outcome"], "overreaction_reversal")

    def test_date_chunking_is_exactly_equivalent_to_complete_replay(
        self,
    ) -> None:
        frame, benchmarks = _fixture()
        complete = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
            _chunk_sessions=None,
        )
        chunked = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
            _chunk_sessions=10,
        )

        pd.testing.assert_frame_equal(
            chunked,
            complete,
            check_exact=True,
        )

    def test_bounded_audit_replay_matches_complete_labels(self) -> None:
        frame, benchmarks = _fixture()
        strategy_ids = ("SWING.CATALYST_DRIFT.5D.V1",)
        labels = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
            strategy_ids=strategy_ids,
        )

        mismatches = (
            strategy_label_module._bounded_strategy_replay_mismatch_count(
                frame,
                benchmarks,
                labels,
                dataset_config=self.config,
                policy=self.policy,
                strategy_ids=strategy_ids,
                chunk_sessions=10,
            )
        )

        self.assertEqual(mismatches, 0)

    def test_same_session_barrier_collision_is_stop_first(self) -> None:
        frame, benchmarks = _fixture(collision=True)
        labels = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )
        row = labels.loc[
            labels["ticker"].eq("AAA")
            & labels["session_date_et"].eq(_decision_date(frame, "AAA"))
            & labels["strategy_id"].eq(
                "SWING.BREAKOUT_EXPANSION.5D.V1"
            )
        ].iloc[0]
        self.assertEqual(row["barrier_outcome"], "stop_first")
        self.assertTrue(bool(row["breakout_failed"]))
        self.assertEqual(int(row["strategy_target"]), 0)

    def test_non_positive_breakout_stop_abstains_instead_of_crashing(
        self,
    ) -> None:
        frame, benchmarks = _fixture()
        aaa = frame.index[frame["ticker"].eq("AAA")]
        first_future = aaa[DECISION_INDEX + 1]
        frame.loc[
            first_future,
            ["open", "high", "low", "close"],
        ] = [1.0, 1.1, 0.9, 1.0]

        labels = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )
        row = labels.loc[
            labels["ticker"].eq("AAA")
            & labels["session_date_et"].eq(_decision_date(frame, "AAA"))
            & labels["strategy_id"].eq(
                "SWING.BREAKOUT_EXPANSION.5D.V1"
            )
        ].iloc[0]
        self.assertFalse(bool(row["strategy_label_eligible"]))
        self.assertEqual(
            row["label_abstention_reason"],
            "invalid_barrier_prices",
        )
        self.assertTrue(pd.isna(row["strategy_target"]))

    def test_missing_atr_cost_evidence_abstains(self) -> None:
        frame, benchmarks = _fixture()
        aaa = frame.index[frame["ticker"].eq("AAA")]
        frame.loc[aaa[DECISION_INDEX], "atr_pct_14"] = np.nan

        labels = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )
        row = labels.loc[
            labels["ticker"].eq("AAA")
            & labels["session_date_et"].eq(_decision_date(frame, "AAA"))
            & labels["strategy_id"].eq(
                "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"
            )
        ].iloc[0]
        self.assertFalse(bool(row["strategy_label_eligible"]))
        self.assertEqual(
            row["label_abstention_reason"],
            "missing_execution_cost_evidence",
        )

    def test_missing_path_abstains_and_poisoned_output_fails_replay(self) -> None:
        frame, benchmarks = _fixture()
        missing_session = frame.loc[
            frame["ticker"].eq("AAA")
        ].iloc[DECISION_INDEX + 2]["session_date_et"]
        incomplete = frame.loc[
            ~(
                frame["ticker"].eq("AAA")
                & frame["session_date_et"].eq(missing_session)
            )
        ].copy()
        labels = build_swing_strategy_labels(
            incomplete,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )
        row = labels.loc[
            labels["ticker"].eq("AAA")
            & labels["session_date_et"].eq(_decision_date(frame, "AAA"))
            & labels["strategy_id"].eq(
                "SWING.TIME_SERIES_MOMENTUM.5D.V1"
            )
        ].iloc[0]
        self.assertFalse(bool(row["strategy_label_eligible"]))
        self.assertEqual(
            row["label_abstention_reason"],
            "missing_exact_stock_path",
        )

        complete = build_swing_strategy_labels(
            frame,
            benchmarks,
            dataset_config=self.config,
            policy=self.policy,
        )
        poisoned = complete.copy()
        index = poisoned.index[
            poisoned["strategy_label_eligible"].astype(bool)
        ][0]
        poisoned.loc[index, "strategy_target"] = (
            1 - int(poisoned.loc[index, "strategy_target"])
        )
        audit = audit_swing_strategy_labels(
            frame,
            benchmarks,
            poisoned,
            dataset_config=self.config,
            policy=self.policy,
        )
        replay = audit.to_frame().set_index("name")
        self.assertEqual(
            replay.loc["strategy_label_replay", "status"],
            "fail",
        )

    def test_bundle_publishes_independent_immutable_strategy_artifacts(
        self,
    ) -> None:
        frame, benchmarks = _fixture()
        with TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "strategy-labels"
            observed = build_swing_strategy_label_bundle(
                frame,
                benchmarks,
                dataset_config=self.config,
                policy=self.policy,
                out_dir=out_dir,
                input_hashes={"fixture": "a" * 64},
            )

            self.assertEqual(observed["status"], "complete")
            self.assertEqual(observed["observed_strategies"], 6)
            self.assertEqual(observed["skipped_strategies"], 0)
            memory = observed["memory"]
            self.assertIsInstance(memory, dict)
            self.assertLess(
                float(memory["current_working_set_gib"]),
                4.0,
            )
            self.assertLessEqual(
                float(memory["peak_working_set_gib"]),
                float(memory["safety_threshold_gib"]),
            )
            manifest = loads(
                (out_dir / "_manifest.json").read_text(encoding="utf-8")
            )
            self.assertEqual(
                manifest["request_sha256"],
                observed["request_sha256"],
            )
            self.assertEqual(len(manifest["artifacts"]), len(STRATEGY_IDS))
            for record in manifest["artifacts"]:
                artifact, artifact_manifest = load_canonical_artifact(
                    Path(record["path"]),
                    expected_type="swing_strategy_labels",
                    allow_research=True,
                )
                self.assertEqual(artifact["strategy_id"].nunique(), 1)
                self.assertEqual(
                    artifact["strategy_id"].iloc[0],
                    record["strategy_id"],
                )
                self.assertEqual(
                    artifact_manifest["inputs"]["strategy_id"],
                    record["strategy_id"],
                )

            validated = build_swing_strategy_label_bundle(
                frame,
                benchmarks,
                dataset_config=self.config,
                policy=self.policy,
                out_dir=out_dir,
                input_hashes={"fixture": "a" * 64},
            )
            self.assertEqual(
                validated["request_sha256"],
                observed["request_sha256"],
            )
            artifact_path = Path(manifest["artifacts"][0]["path"])
            artifact_path.write_bytes(
                artifact_path.read_bytes() + b"corruption"
            )
            with self.assertRaisesRegex(
                DataReadinessError,
                "integrity check failed",
            ):
                build_swing_strategy_label_bundle(
                    frame,
                    benchmarks,
                    dataset_config=self.config,
                    policy=self.policy,
                    out_dir=out_dir,
                    input_hashes={"fixture": "a" * 64},
                )

    def test_incomplete_bundle_isolated_and_resumes_by_request_hash(
        self,
    ) -> None:
        frame, benchmarks = _fixture()
        original_writer = strategy_label_module.write_canonical_artifact
        calls = 0

        def fail_third_write(*args: object, **kwargs: object) -> object:
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError("injected isolated publisher failure")
            return original_writer(*args, **kwargs)

        with TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "strategy-labels"
            with patch.object(
                strategy_label_module,
                "write_canonical_artifact",
                side_effect=fail_third_write,
            ):
                incomplete = build_swing_strategy_label_bundle(
                    frame,
                    benchmarks,
                    dataset_config=self.config,
                    policy=self.policy,
                    out_dir=out_dir,
                    input_hashes={"fixture": "b" * 64},
                )
            self.assertEqual(incomplete["status"], "incomplete")
            self.assertEqual(incomplete["observed_strategies"], 5)
            self.assertIn(
                "SWING.CATALYST_DRIFT.5D.V1",
                incomplete["failed_strategies"],
            )
            self.assertFalse((out_dir / "_manifest.json").exists())

            with (
                patch.object(
                    strategy_label_module,
                    "swing_strategy_evaluator_sha256",
                    return_value="d" * 64,
                ),
                self.assertRaisesRegex(
                    DataReadinessError,
                    "request lineage mismatch",
                ),
            ):
                build_swing_strategy_label_bundle(
                    frame,
                    benchmarks,
                    dataset_config=self.config,
                    policy=self.policy,
                    out_dir=out_dir,
                    input_hashes={"fixture": "b" * 64},
                )

            complete = build_swing_strategy_label_bundle(
                frame,
                benchmarks,
                dataset_config=self.config,
                policy=self.policy,
                out_dir=out_dir,
                input_hashes={"fixture": "b" * 64},
            )
            self.assertEqual(complete["status"], "complete")
            self.assertEqual(complete["skipped_strategies"], 5)
            self.assertTrue((out_dir / "_manifest.json").exists())

    def test_orphaned_uncommitted_parquet_is_rebuilt_on_resume(self) -> None:
        frame, benchmarks = _fixture()
        original_writer = strategy_label_module.write_canonical_artifact
        calls = 0

        def orphan_first_write(
            labels: pd.DataFrame,
            path: Path,
            **kwargs: object,
        ) -> object:
            nonlocal calls
            calls += 1
            if calls == 1:
                labels.to_parquet(path, index=False)
                raise OSError("injected crash before manifest publication")
            return original_writer(labels, path, **kwargs)

        with TemporaryDirectory() as temporary:
            out_dir = Path(temporary) / "strategy-labels"
            with patch.object(
                strategy_label_module,
                "write_canonical_artifact",
                side_effect=orphan_first_write,
            ):
                incomplete = build_swing_strategy_label_bundle(
                    frame,
                    benchmarks,
                    dataset_config=self.config,
                    policy=self.policy,
                    out_dir=out_dir,
                    input_hashes={"fixture": "c" * 64},
                )
            self.assertEqual(incomplete["status"], "incomplete")
            complete = build_swing_strategy_label_bundle(
                frame,
                benchmarks,
                dataset_config=self.config,
                policy=self.policy,
                out_dir=out_dir,
                input_hashes={"fixture": "c" * 64},
            )
            self.assertEqual(complete["status"], "complete")


def _fixture(
    *,
    collision: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    sessions = pd.bdate_range("2025-01-02", periods=36).date
    stock_parts = [
        _stock_rows("AAA", "security:aaa", sessions, rising=True),
        _stock_rows("BBB", "security:bbb", sessions, rising=False),
    ]
    stocks = pd.concat(stock_parts, ignore_index=True)
    benchmarks = pd.concat(
        [
            _benchmark_rows("SPY", sessions, 500.0, 0.0010),
            _benchmark_rows("QQQ", sessions, 400.0, 0.0012),
            _benchmark_rows("XLK", sessions, 200.0, 0.0008),
        ],
        ignore_index=True,
    )

    aaa = stocks.index[stocks["ticker"].eq("AAA")]
    aaa_decision = aaa[DECISION_INDEX]
    stocks.loc[aaa_decision, "close"] = 102.0
    stocks.loc[aaa_decision, "high"] = 102.4
    stocks.loc[aaa_decision, "low"] = 99.8
    stocks.loc[aaa_decision, "volume_ratio_20"] = 2.0
    stocks.loc[aaa_decision, "close_location"] = 0.9
    stocks.loc[aaa_decision, "event_count_3d"] = 2.0
    stocks.loc[aaa_decision, "sentiment_coverage_3d"] = 1.0
    stocks.loc[aaa_decision, "event_relevance_mean_3d"] = 0.99
    stocks.loc[aaa_decision, "return_5d"] = 0.08
    stocks.loc[aaa_decision, "return_20d"] = 0.12
    stocks.loc[aaa_decision, "rel_return_20d_vs_sector"] = 0.08
    stocks.loc[aaa_decision, "xs_rank_rel_return_20d_vs_sector"] = 0.95
    for offset, close in enumerate((103.0, 105.0, 106.0, 107.0, 108.0), start=1):
        index = aaa[DECISION_INDEX + offset]
        stocks.loc[index, ["open", "close"]] = [close, close + 0.5]
        stocks.loc[index, "high"] = close + 2.0
        stocks.loc[index, "low"] = close - 0.5
    if collision:
        first = aaa[DECISION_INDEX + 1]
        stocks.loc[first, "high"] = 110.0
        stocks.loc[first, "low"] = 95.0

    bbb = stocks.index[stocks["ticker"].eq("BBB")]
    bbb_decision = bbb[DECISION_INDEX]
    stocks.loc[bbb_decision, "return_5d"] = -0.10
    stocks.loc[bbb_decision, "rsi_14"] = 25.0
    stocks.loc[bbb_decision, "event_count_3d"] = 0.0
    stocks.loc[bbb_decision, "return_20d"] = -0.12
    stocks.loc[bbb_decision, "rel_return_20d_vs_sector"] = -0.10
    stocks.loc[bbb_decision, "xs_rank_rel_return_20d_vs_sector"] = 0.05
    for offset, close in enumerate((90.0, 93.0, 95.0, 96.0, 97.0), start=1):
        index = bbb[DECISION_INDEX + offset]
        stocks.loc[index, ["open", "close"]] = [close, close + 0.5]
        stocks.loc[index, "high"] = close + 1.0
        stocks.loc[index, "low"] = close - 0.5
    return stocks, benchmarks


def _stock_rows(
    ticker: str,
    security_id: str,
    sessions: np.ndarray,
    *,
    rising: bool,
) -> pd.DataFrame:
    count = len(sessions)
    close = np.full(count, 100.0 if rising else 95.0)
    timestamps = pd.to_datetime(sessions, utc=True)
    bar_start = timestamps + pd.Timedelta(hours=14, minutes=30)
    bar_end = timestamps + pd.Timedelta(hours=21)
    decision = timestamps + pd.Timedelta(hours=22)
    return pd.DataFrame(
        {
            "ticker": ticker,
            "security_id": security_id,
            "session_date_et": sessions,
            "decision_group_id": [
                f"group-{session.isoformat()}" for session in sessions
            ],
            "decision_time_utc": decision,
            "feature_available_at_utc": bar_end,
            "bar_start_utc": bar_start,
            "bar_end_utc": bar_end,
            "available_at_utc": bar_end,
            "open": close,
            "high": close + 0.5,
            "low": close - 0.5,
            "close": close,
            "volume": np.full(count, 2_000_000),
            "price_feed": "sip",
            "adjustment": "all",
            "primary_benchmark": "XLK",
            "feature_eligible": True,
            "cross_section_eligible": True,
            "atr_pct_14": 0.02 if rising else 0.03,
            "return_5d": 0.04 if rising else -0.02,
            "return_20d": 0.08 if rising else -0.04,
            "dist_sma_50": 0.05 if rising else -0.02,
            "dist_sma_200": 0.10 if rising else -0.05,
            "sma_200_slope_20d": 0.01 if rising else -0.01,
            "dist_ema_20": 0.03 if rising else -0.02,
            "rsi_14": 60.0 if rising else 45.0,
            "event_count_3d": 0.0,
            "sentiment_coverage_3d": 1.0,
            "event_relevance_mean_3d": 0.0,
            "catalyst_source_complete": True,
            "volume_ratio_20": 1.0,
            "close_location": 0.5,
            "rel_return_20d_vs_sector": 0.06 if rising else -0.04,
            "xs_rank_rel_return_20d_vs_sector": 0.9 if rising else 0.1,
        }
    )


def _benchmark_rows(
    ticker: str,
    sessions: np.ndarray,
    start: float,
    daily_return: float,
) -> pd.DataFrame:
    ordinal = np.arange(len(sessions))
    open_price = start * (1.0 + daily_return) ** ordinal
    return pd.DataFrame(
        {
            "ticker": ticker,
            "session_date_et": sessions,
            "open": open_price,
            "close": open_price * (1.0 + daily_return),
        }
    )


def _decision_date(frame: pd.DataFrame, ticker: str) -> object:
    return frame.loc[frame["ticker"].eq(ticker)].iloc[DECISION_INDEX][
        "session_date_et"
    ]


if __name__ == "__main__":
    unittest.main()
