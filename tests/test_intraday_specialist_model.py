from __future__ import annotations

import unittest
from pathlib import Path
from typing import ClassVar

import exchange_calendars as xcals
import pandas as pd

from market_predictor.intraday.specialist_contracts import (
    IntradaySpecialistResearchConfig,
    load_intraday_specialist_research_config,
)
from market_predictor.intraday.specialist_model import (
    DETERMINISTIC_SCORE_FORMULA_SHA256,
    SPECIALIST_ACCEPTED_STATUS,
    SPECIALIST_REJECTED_STATUS,
    _clock_phase_economics,
    build_specialist_split_plan,
    evaluate_specialist_experiment,
    specialist_experiment_specs,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
STRATEGY_ID = "INTRADAY.OPENING_RANGE_BREAKOUT.60M.V1"
MOMENTUM_ID = "INTRADAY.MOMENTUM_CONTINUATION.60M.V1"


class IntradaySpecialistModelTests(unittest.TestCase):
    config: ClassVar[IntradaySpecialistResearchConfig]

    @classmethod
    def setUpClass(cls) -> None:
        base = load_intraday_specialist_research_config(
            ROOT / "configs" / "intraday_specialist_research.toml"
        )
        cls.config = base.model_copy(
            update={
                "n_splits": 3,
                "min_train_sessions": 15,
                "min_train_rows": 100,
                "min_training_tickers": 8,
                "ticker_holdout_fraction": 0.20,
                "top_k": 3,
                "max_trades_per_session": 3,
                "logistic_max_iter": 100,
                "hgb_max_iter": 50,
                "minimum_selected_trades": 1,
                "minimum_avg_net_return": -1.0,
                "minimum_avg_excess_return_vs_spy": -1.0,
                "minimum_avg_excess_return_vs_sector": -1.0,
                "minimum_avg_net_return_ci_low": -1.0,
                "minimum_avg_excess_return_vs_spy_ci_low": -1.0,
                "minimum_profit_factor": 0.0,
                "maximum_drawdown": 1.0,
                "maximum_negative_session_rate": 1.0,
                "required_market_regimes": ("risk_on", "risk_off"),
                "minimum_regime_selected_trades": 1,
                "minimum_regime_avg_net_return": -1.0,
                "minimum_regime_avg_excess_return_vs_spy": -1.0,
                "technical_features": (
                    "atr_pct",
                    "volatility_12bar",
                    "return_1bar",
                    "rel_return_3bar_vs_sector",
                    "close_location_5m",
                    "dist_opening_range_high",
                    "relative_volume_same_minute_20d",
                    "rel_return_3bar_vs_qqq",
                ),
            }
        )

    def test_catalog_limits_direct_ranker_to_momentum(self) -> None:
        ordinary = specialist_experiment_specs(
            STRATEGY_ID, self.config
        )
        momentum = specialist_experiment_specs(
            MOMENTUM_ID, self.config
        )

        self.assertNotIn(
            "direct_ranker",
            {spec.estimator_family for spec in ordinary},
        )
        self.assertIn(
            "direct_ranker",
            {spec.estimator_family for spec in momentum},
        )

    def test_split_is_deterministic_xnys_purged_and_shared(self) -> None:
        dataset = _dataset()
        first = build_specialist_split_plan(
            dataset,
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        second = build_specialist_split_plan(
            dataset.sample(frac=1.0, random_state=17),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )

        self.assertEqual(first.split_sha256, second.split_sha256)
        self.assertEqual(first.holdout_tickers, second.holdout_tickers)
        for fold in first.folds:
            train = first.development.iloc[fold.train_indices]
            test = first.development.iloc[fold.test_indices]
            self.assertLess(
                pd.to_datetime(
                    train["label_available_at_utc"], utc=True
                ).max(),
                pd.to_datetime(
                    test["decision_time_utc"], utc=True
                ).min(),
            )

    def test_baseline_retains_model_only_after_all_gates_pass(self) -> None:
        plan = build_specialist_split_plan(
            _dataset(),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        spec = specialist_experiment_specs(
            STRATEGY_ID, self.config
        )[0]

        accepted = evaluate_specialist_experiment(
            plan, spec, config=self.config
        )
        rejected_config = self.config.model_copy(
            update={"minimum_avg_net_return": 1.0}
        )
        rejected = evaluate_specialist_experiment(
            plan, spec, config=rejected_config
        )

        self.assertEqual(accepted.status, SPECIALIST_ACCEPTED_STATUS)
        self.assertIsNotNone(accepted.retained_model)
        self.assertEqual(rejected.status, SPECIALIST_REJECTED_STATUS)
        self.assertIsNone(rejected.retained_model)
        self.assertTrue(rejected.rejection_reasons)
        self.assertEqual(
            accepted.metrics["deterministic_score_formula_sha256"],
            DETERMINISTIC_SCORE_FORMULA_SHA256,
        )
        self.assertEqual(
            accepted.metrics["catalyst_overlay_status"],
            "data_blocked",
        )

    def test_learned_candidate_models_opportunity_and_downside_separately(
        self,
    ) -> None:
        plan = build_specialist_split_plan(
            _dataset(),
            strategy_id=STRATEGY_ID,
            config=self.config,
        )
        spec = specialist_experiment_specs(
            STRATEGY_ID, self.config
        )[1]

        result = evaluate_specialist_experiment(
            plan, spec, config=self.config
        )

        self.assertEqual(result.status, SPECIALIST_ACCEPTED_STATUS)
        retained = result.retained_model
        self.assertIsNotNone(retained)
        assert retained is not None
        self.assertEqual(
            set(retained.estimators),
            {plan.opportunity_target, plan.downside_target},
        )
        self.assertIn(
            "intraday_opportunity_probability",
            result.predictions,
        )
        self.assertIn(
            "intraday_downside_probability",
            result.predictions,
        )

    def test_raw_opportunity_orders_after_calibrated_downside_veto(
        self,
    ) -> None:
        config = self.config.model_copy(
            update={"top_k": 1, "max_trades_per_session": 1}
        )
        predictions = _selection_fixture()

        evidence = _clock_phase_economics(
            predictions,
            horizon_minutes=60,
            scope="walk_forward",
            config=config,
            cost_stress=1.0,
        )

        selected = evidence.loc[evidence["selected_trades"].gt(0)]
        self.assertEqual(int(selected["selected_trades"].sum()), 1)
        self.assertAlmostEqual(
            float(selected.iloc[0]["avg_trade_return"]),
            0.02,
        )

    def test_overlap_and_non_xnys_rows_fail_closed(self) -> None:
        overlapping = pd.concat(
            [_dataset(), _dataset().iloc[[0]].copy()],
            ignore_index=True,
        )
        overlapping.loc[
            overlapping.index[-1], "setup_id"
        ] = "overlapping-setup"
        with self.assertRaisesRegex(
            DataReadinessError, "intervals overlap"
        ):
            build_specialist_split_plan(
                overlapping,
                strategy_id=STRATEGY_ID,
                config=self.config,
            )

        invalid = _dataset()
        invalid.loc[0, "session_date_et"] = pd.Timestamp(
            "2025-01-04"
        ).date()
        with self.assertRaisesRegex(DataReadinessError, "non-XNYS"):
            build_specialist_split_plan(
                invalid,
                strategy_id=STRATEGY_ID,
                config=self.config,
            )

    def test_cost_below_ten_bps_fails_closed(self) -> None:
        invalid = _dataset()
        invalid.loc[
            0, "path_realized_return_net_60m"
        ] = invalid.loc[0, "path_realized_return_gross_60m"] - 0.0005

        with self.assertRaisesRegex(DataReadinessError, "minimum stamped cost"):
            build_specialist_split_plan(
                invalid,
                strategy_id=STRATEGY_ID,
                config=self.config,
            )


def _dataset() -> pd.DataFrame:
    calendar = xcals.get_calendar("XNYS")
    sessions = calendar.sessions_in_range("2025-01-02", "2025-03-31")[
        :55
    ]
    tickers = [f"T{index:02d}" for index in range(12)]
    rows: list[dict[str, object]] = []
    for session_index, session in enumerate(sessions):
        session_date = pd.Timestamp(session).date()
        decision = pd.Timestamp(
            calendar.session_open(session)
        ).tz_convert("UTC") + pd.Timedelta(minutes=31)
        entry = decision
        window_end = entry + pd.Timedelta(minutes=60)
        for ticker_index, ticker in enumerate(tickers):
            opportunity = int(
                (session_index + ticker_index) % 3 != 0
            )
            downside = 1 - opportunity
            net = 0.01 if opportunity else -0.006
            feature_signal = (
                0.02 if opportunity else -0.01
            ) + ticker_index / 10_000.0
            regime_index = session_index % 3
            rows.append(
                {
                    "strategy_id": STRATEGY_ID,
                    "setup_id": f"{session_date}:{ticker}",
                    "ticker": ticker,
                    "session_date_et": session_date,
                    "decision_time_utc": decision,
                    "feature_available_at_utc": (
                        decision - pd.Timedelta(seconds=30)
                    ),
                    "entry_time_utc": entry,
                    "exit_time_utc": window_end,
                    "label_available_at_utc": (
                        window_end + pd.Timedelta(seconds=30)
                    ),
                    "label_window_end_utc": window_end,
                    "label_eligible": True,
                    "horizon_minutes": 60,
                    "target_before_stop_60m": opportunity,
                    "stop_before_target_60m": downside,
                    "path_realized_return_gross_60m": net + 0.001,
                    "path_realized_return_net_60m": net,
                    "path_spy_return_60m": 0.0,
                    "path_qqq_return_60m": 0.0,
                    "path_sector_return_60m": 0.0,
                    "path_excess_return_60m_vs_spy": net,
                    "path_excess_return_60m_vs_qqq": net,
                    "path_excess_return_60m_vs_sector": net,
                    "entry_price": 20.0,
                    "entry_dollar_volume": 2_000_000.0,
                    "entry_atr_pct": 0.02,
                    "sector": (
                        "Technology" if ticker_index % 2 else "Healthcare"
                    ),
                    "primary_benchmark": (
                        "XLK" if ticker_index % 2 else "XLV"
                    ),
                    "market_cap_bucket": (
                        "mid" if ticker_index % 2 else "small"
                    ),
                    "liquidity_bucket": (
                        "high" if ticker_index % 2 else "medium"
                    ),
                    "regime_risk_on": int(regime_index == 0),
                    "regime_risk_off": int(regime_index == 1),
                    "regime_high_volatility": 0,
                    "atr_pct": 0.02 + downside * 0.01,
                    "volatility_12bar": 0.01 + downside * 0.01,
                    "return_1bar": feature_signal,
                    "rel_return_3bar_vs_sector": feature_signal,
                    "close_location_5m": 0.8 if opportunity else 0.55,
                    "dist_opening_range_high": max(feature_signal, 0.0),
                    "relative_volume_same_minute_20d": (
                        2.0 if opportunity else 1.3
                    ),
                    "rel_return_3bar_vs_qqq": feature_signal,
                }
            )
    return pd.DataFrame(rows)


def _selection_fixture() -> pd.DataFrame:
    session = pd.Timestamp("2025-01-02").date()
    decision = pd.Timestamp("2025-01-02T15:01:00Z")
    return pd.DataFrame(
        {
            "ticker": ["VETO", "WIN", "LOW"],
            "session_date_et": [session] * 3,
            "decision_group_id": ["g1"] * 3,
            "decision_time_utc": [decision] * 3,
            "entry_time_utc": [decision] * 3,
            "exit_time_utc": [decision + pd.Timedelta(minutes=60)] * 3,
            "label_window_end_utc": [
                decision + pd.Timedelta(minutes=60)
            ]
            * 3,
            "market_regime": ["risk_on"] * 3,
            "sector": ["Technology"] * 3,
            "primary_benchmark": ["XLK"] * 3,
            "target_before_stop_60m": [1, 1, 0],
            "stop_before_target_60m": [0, 0, 1],
            "path_realized_return_net_60m": [0.50, 0.02, -0.01],
            "path_excess_return_60m_vs_spy": [0.50, 0.02, -0.01],
            "path_excess_return_60m_vs_qqq": [0.50, 0.02, -0.01],
            "path_excess_return_60m_vs_sector": [0.50, 0.02, -0.01],
            "intraday_opportunity_probability": [0.5, 0.5, 0.5],
            "intraday_downside_probability": [0.60, 0.20, 0.10],
            "selection_raw_score": [0.90, 0.80, 0.10],
        }
    )


if __name__ == "__main__":
    unittest.main()
