from __future__ import annotations

import unittest
from dataclasses import replace
from pathlib import Path
from shutil import copytree
from tempfile import TemporaryDirectory

import pandas as pd

from market_predictor.swing import specialist_experiments
from market_predictor.swing.evaluation import phase_economics
from market_predictor.swing.specialist_contracts import (
    load_swing_specialist_research_config,
)
from market_predictor.swing.specialist_model import (
    SPECIALIST_ACCEPTED_STATUS,
    build_specialist_split_plan,
    evaluate_specialist_experiment,
    specialist_experiment_specs,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ID = "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1"


class SwingSpecialistModelTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        base = load_swing_specialist_research_config(
            ROOT / "configs" / "swing_specialist_research.toml"
        )
        cls.config = base.model_copy(
            update={
                "n_splits": 3,
                "min_train_sessions": 20,
                "min_train_rows": 100,
                "min_training_tickers": 10,
                "ticker_holdout_fraction": 0.20,
                "top_k": 3,
                "minimum_selected_trades": 1,
                "minimum_avg_net_return": -1.0,
                "minimum_avg_excess_return_vs_spy": -1.0,
                "minimum_avg_excess_return_vs_sector": -1.0,
                "minimum_avg_net_return_ci_low": -1.0,
                "minimum_avg_excess_return_vs_spy_ci_low": -1.0,
                "minimum_profit_factor": 0.0,
                "maximum_drawdown": 1.0,
                "maximum_negative_phase_rate": 1.0,
                "required_market_regimes": ("risk_on", "risk_off"),
                "minimum_regime_selected_trades": 1,
                "minimum_regime_avg_net_return": -1.0,
                "minimum_regime_avg_excess_return_vs_spy": -1.0,
            }
        )

    def test_experiment_catalog_matches_frozen_budget(self) -> None:
        counts = {
            strategy_id: len(
                specialist_experiment_specs(strategy_id, self.config)
            )
            for strategy_id in self.config.strategies
        }

        self.assertEqual(
            counts,
            {
                "SWING.CROSS_SECTIONAL_MOMENTUM.5D.V1": 4,
                "SWING.TIME_SERIES_MOMENTUM.5D.V1": 3,
                "SWING.CATALYST_DRIFT.5D.V1": 7,
                "SWING.SHORT_TERM_REVERSAL.3D.V1": 3,
                "SWING.BREAKOUT_EXPANSION.5D.V1": 3,
                "SWING.SECTOR_RESIDUAL_MOMENTUM.5D.V1": 4,
            },
        )

    def test_deterministic_candidate_is_causal_and_economically_audited(
        self,
    ) -> None:
        dataset = _dataset()
        first = build_specialist_split_plan(
            dataset,
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        second = build_specialist_split_plan(
            dataset,
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        spec = specialist_experiment_specs(
            STRATEGY_ID,
            self.config,
        )[0]

        result = evaluate_specialist_experiment(
            first,
            spec,
            config=self.config,
        )

        self.assertEqual(first.split_sha256, second.split_sha256)
        self.assertEqual(result.status, SPECIALIST_ACCEPTED_STATUS)
        self.assertFalse(result.predictions.empty)
        self.assertFalse(result.economics.empty)
        self.assertFalse(result.regime_evidence.empty)
        self.assertFalse(result.capacity_evidence.empty)
        self.assertEqual(
            set(result.predictions["ticker_cohort"]),
            {"seen", "unseen"},
        )
        included = result.fold_audit.loc[
            result.fold_audit["validation_status"].eq("included")
        ]
        self.assertTrue(
            (
                pd.to_datetime(
                    included["calibration_train_cutoff_utc"],
                    utc=True,
                )
                < pd.to_datetime(
                    included["min_test_decision_time_utc"],
                    utc=True,
                )
            ).all()
        )

    def test_learned_candidates_share_exact_validation_rows(self) -> None:
        plan = build_specialist_split_plan(
            _dataset(),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        specs = specialist_experiment_specs(STRATEGY_ID, self.config)
        baseline = evaluate_specialist_experiment(
            plan,
            specs[0],
            config=self.config,
        )
        logistic = evaluate_specialist_experiment(
            plan,
            specs[1],
            config=self.config,
        )

        identity_columns = [
            "row_identity",
            "validation_fold",
            "ticker_cohort",
        ]
        pd.testing.assert_frame_equal(
            baseline.predictions[identity_columns].reset_index(drop=True),
            logistic.predictions[identity_columns].reset_index(drop=True),
        )
        self.assertEqual(
            baseline.metrics["split_sha256"],
            logistic.metrics["split_sha256"],
        )

    def test_stamped_net_economics_does_not_recharge_cost(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "BBB"],
                "session_date_et": [
                    pd.Timestamp("2025-01-02").date(),
                    pd.Timestamp("2025-01-02").date(),
                ],
                "decision_group_id": ["g1", "g1"],
                "swing_probability": [0.9, 0.8],
                "strategy_target": [1, 1],
                "future_gross_return_1d": [0.20, 0.20],
                "future_net_return_1d": [0.10, 0.10],
                "future_spy_return_1d": [0.01, 0.01],
                "future_qqq_return_1d": [0.02, 0.02],
                "future_sector_return_1d": [0.03, 0.03],
                "future_excess_return_1d_vs_spy": [0.09, 0.09],
                "future_excess_return_1d_vs_qqq": [0.08, 0.08],
                "future_excess_return_1d_vs_sector": [0.07, 0.07],
                "close": [100.0, 100.0],
                "atr_pct_14": [0.02, 0.02],
            }
        )

        result = phase_economics(
            frame,
            horizon=1,
            top_k=2,
            scope="test",
            use_stamped_net_returns=True,
        )

        self.assertAlmostEqual(
            float(result.iloc[0]["avg_trade_return"]),
            0.10,
        )
        self.assertAlmostEqual(
            float(result.iloc[0]["avg_excess_return_vs_spy"]),
            0.09,
        )

    def test_raw_score_breaks_calibration_plateaus_for_selection(self) -> None:
        frame = pd.DataFrame(
            {
                "ticker": ["AAA", "ZZZ"],
                "session_date_et": [
                    pd.Timestamp("2025-01-02").date(),
                    pd.Timestamp("2025-01-02").date(),
                ],
                "decision_group_id": ["g1", "g1"],
                "swing_probability": [0.5, 0.5],
                "raw_probability": [0.1, 0.9],
                "strategy_target": [0, 1],
                "future_net_return_1d": [-0.10, 0.20],
                "future_excess_return_1d_vs_spy": [-0.11, 0.19],
                "future_excess_return_1d_vs_qqq": [-0.12, 0.18],
                "future_excess_return_1d_vs_sector": [-0.13, 0.17],
            }
        )

        result = phase_economics(
            frame,
            horizon=1,
            top_k=1,
            scope="test",
            use_stamped_net_returns=True,
            selection_score_column="raw_probability",
        )

        self.assertAlmostEqual(
            float(result.iloc[0]["avg_trade_return"]),
            0.20,
        )

    def test_candidate_evidence_is_hash_verified_and_resumable(self) -> None:
        plan = build_specialist_split_plan(
            _dataset(),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        spec = specialist_experiment_specs(
            STRATEGY_ID,
            self.config,
        )[0]
        result = evaluate_specialist_experiment(
            plan,
            spec,
            config=self.config,
        )
        request = specialist_experiments._candidate_request(
            spec,
            dataset_sha256="a" * 64,
            split_sha256=plan.split_sha256,
            bundle_request_sha256="b" * 64,
            config=self.config,
        )
        with TemporaryDirectory() as directory:
            out_dir = Path(directory) / "candidate"
            written = specialist_experiments._write_candidate_evidence(
                out_dir,
                result=result,
                request=request,
                dataset_sha256="a" * 64,
                config=self.config,
            )
            resumed = specialist_experiments._load_existing_candidate(
                out_dir,
                expected_request_sha256=str(request["request_sha256"]),
            )

            self.assertIsNotNone(resumed)
            assert resumed is not None
            self.assertEqual(written["candidate_id"], resumed["candidate_id"])
            self.assertTrue((out_dir / "model.joblib").is_file())
            manifest = specialist_experiments._load_json(
                out_dir / "_manifest.json"
            )
            files = manifest["files"]
            assert isinstance(files, dict)
            model_record = files["model"]
            assert isinstance(model_record, dict)
            self.assertFalse(
                Path(str(model_record["path"])).is_absolute()
            )
            self.assertEqual(
                written["manifest_sha256"],
                resumed["manifest_sha256"],
            )
            copied = out_dir.parent / "copied-candidate"
            copytree(out_dir, copied)
            copied_resume = (
                specialist_experiments._load_existing_candidate(
                    copied,
                    expected_request_sha256=str(
                        request["request_sha256"]
                    ),
                )
            )
            self.assertIsNotNone(copied_resume)
            (copied / "_request.json").write_text(
                "{}",
                encoding="utf-8",
            )
            with self.assertRaisesRegex(
                DataReadinessError,
                "request identity mismatch",
            ):
                specialist_experiments._load_existing_candidate(
                    copied,
                    expected_request_sha256=str(
                        request["request_sha256"]
                    ),
                )

    def test_rejected_candidate_cannot_retain_stale_model(self) -> None:
        plan = build_specialist_split_plan(
            _dataset(),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        spec = specialist_experiment_specs(
            STRATEGY_ID,
            self.config,
        )[0]
        accepted = evaluate_specialist_experiment(
            plan,
            spec,
            config=self.config,
        )
        rejected = replace(
            accepted,
            status="rejected",
            rejection_reasons=("fixture rejection",),
            final_estimator=None,
            final_calibrator=None,
        )
        request = specialist_experiments._candidate_request(
            spec,
            dataset_sha256="a" * 64,
            split_sha256=plan.split_sha256,
            bundle_request_sha256="b" * 64,
            config=self.config,
        )
        with TemporaryDirectory() as directory:
            out_dir = Path(directory) / "candidate"
            specialist_experiments._write_candidate_evidence(
                out_dir,
                result=rejected,
                request=request,
                dataset_sha256="a" * 64,
                config=self.config,
            )

            self.assertFalse((out_dir / "model.joblib").exists())
            (out_dir / "model.joblib").write_bytes(b"stale")
            with self.assertRaisesRegex(
                DataReadinessError,
                "file set mismatch",
            ):
                specialist_experiments._load_existing_candidate(
                    out_dir,
                    expected_request_sha256=str(
                        request["request_sha256"]
                    ),
                )

    def test_implementation_identity_binds_sources_and_runtime(self) -> None:
        identity = specialist_experiments._implementation_identity()

        self.assertEqual(len(str(identity["implementation_sha256"])), 64)
        self.assertIn(
            "specialist_model.py",
            identity["source_sha256"],
        )
        self.assertIn("scikit-learn", identity["runtime_versions"])


def _dataset() -> pd.DataFrame:
    sessions = pd.bdate_range("2024-01-02", periods=90)
    tickers = [f"T{index:02d}" for index in range(20)]
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        decision = session.tz_localize("UTC") + pd.Timedelta(hours=22)
        label_available = decision + pd.Timedelta(days=7)
        for ticker_index, ticker in enumerate(tickers):
            target = int((session_index + ticker_index) % 4 == 0)
            rank = (ticker_index + 1) / len(tickers)
            net = 0.03 if target else -0.015
            cost = 0.002
            spy = 0.004
            qqq = 0.005
            sector = 0.003
            row_id = f"{session.date().isoformat()}-{ticker}"
            rows.append(
                {
                    "strategy_id": STRATEGY_ID,
                    "strategy_target": target,
                    "strategy_label_eligible": True,
                    "strategy_horizon_sessions": 5,
                    "strategy_dataset_row_id": row_id,
                    "ticker": ticker,
                    "security_id": f"security:{ticker.lower()}",
                    "session_date_et": session.date(),
                    "decision_group_id": f"group-{session.date()}",
                    "decision_time_utc": decision,
                    "feature_available_at_utc": decision
                    - pd.Timedelta(minutes=1),
                    "label_available_at_utc": label_available,
                    "entry_time_utc": decision + pd.Timedelta(hours=16),
                    "exit_time_utc": label_available
                    - pd.Timedelta(hours=1),
                    "universe_snapshot_id": f"universe-{session.date()}",
                    "market_regime": (
                        "risk_on" if session_index % 2 == 0 else "risk_off"
                    ),
                    "sector": "Technology",
                    "primary_benchmark": "XLK",
                    "strategy_gross_return": net + cost,
                    "strategy_execution_cost_fraction": cost,
                    "strategy_net_return": net,
                    "strategy_spy_return": spy,
                    "strategy_qqq_return": qqq,
                    "strategy_sector_return": sector,
                    "strategy_excess_return_vs_spy": net - spy,
                    "strategy_excess_return_vs_qqq": net - qqq,
                    "strategy_excess_return_vs_sector": net - sector,
                    "strategy_mfe": max(net, 0) + 0.01,
                    "strategy_mae": min(net, 0) - 0.01,
                    "future_gross_return_5d": net + cost,
                    "future_net_return_5d": net,
                    "future_spy_return_5d": spy,
                    "future_qqq_return_5d": qqq,
                    "future_sector_return_5d": sector,
                    "future_excess_return_5d_vs_spy": net - spy,
                    "future_excess_return_5d_vs_qqq": net - qqq,
                    "future_excess_return_5d_vs_sector": net - sector,
                    "target_net_positive_5d": target,
                    "close": 50.0 + ticker_index,
                    "atr_pct_14": 0.02,
                    "dollar_volume": 5_000_000.0,
                    "return_20d": rank * 0.10,
                    "dist_sma_50": rank * 0.05,
                    "dist_sma_200": rank * 0.08,
                    "sma_200_slope_20d": rank * 0.01,
                    "xs_rank_rel_return_20d_vs_sector": rank,
                    "event_count_3d": 0,
                    "event_relevance_mean_3d": 0.0,
                    "low_relevance_event_fraction_3d": 0.0,
                }
            )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    unittest.main()
