from __future__ import annotations

import tomllib
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

from market_predictor.intraday.specialist_contracts import (
    INTRADAY_CATALYST_OVERLAY_FEATURES,
    INTRADAY_SPECIALIST_IDS,
    INTRADAY_SPECIALIST_RESEARCH_SCHEMA,
    IntradaySpecialistResearchConfig,
    intraday_specialist_policy_identity,
    load_intraday_specialist_research_config,
)
from market_predictor.v3.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY = ROOT / "configs" / "intraday_specialist_research.toml"


class IntradaySpecialistContractTests(unittest.TestCase):
    def test_repository_policy_freezes_complete_strategy_order(self) -> None:
        config = load_intraday_specialist_research_config(POLICY)

        self.assertEqual(config.schema_version, INTRADAY_SPECIALIST_RESEARCH_SCHEMA)
        self.assertEqual(tuple(config.strategies), INTRADAY_SPECIALIST_IDS)
        self.assertEqual(config.required_price_feed, "sip")
        self.assertEqual(config.required_adjustment, "all")
        self.assertGreaterEqual(config.minimum_round_trip_cost_bps, 10.0)
        self.assertLess(
            config.memory_guard_headroom_gib,
            config.maximum_process_memory_gib,
        )

    def test_every_strategy_is_long_only_and_session_bounded(self) -> None:
        config = load_intraday_specialist_research_config(POLICY)

        for strategy in config.strategies.values():
            self.assertEqual(strategy.direction, "long")
            self.assertLess(
                strategy.first_decision_minute_et,
                strategy.last_decision_minute_et_exclusive,
            )
            self.assertLessEqual(
                strategy.last_decision_minute_et_exclusive
                - 1
                + strategy.horizon_minutes,
                960,
            )
            self.assertGreaterEqual(
                strategy.minimum_setup_spacing_minutes,
                strategy.horizon_minutes,
            )

    def test_catalyst_is_overlay_only(self) -> None:
        config = load_intraday_specialist_research_config(POLICY)

        self.assertEqual(
            set(config.catalyst_overlay_features),
            INTRADAY_CATALYST_OVERLAY_FEATURES,
        )
        self.assertTrue(
            set(config.technical_features).isdisjoint(
                config.catalyst_overlay_features
            )
        )
        for strategy in config.strategies.values():
            self.assertEqual(
                strategy.selection_policies,
                (
                    "model_score_only",
                    "catalyst_confirmation_overlay",
                ),
            )

    def test_ranker_is_limited_to_cross_sectional_momentum(self) -> None:
        config = load_intraday_specialist_research_config(POLICY)

        rankers = {
            strategy_id
            for strategy_id, strategy in config.strategies.items()
            if "direct_ranker" in strategy.estimator_families
        }
        self.assertEqual(
            rankers,
            {"INTRADAY.MOMENTUM_CONTINUATION.60M.V1"},
        )

    def test_policy_identity_is_deterministic(self) -> None:
        first = intraday_specialist_policy_identity(POLICY)
        second = intraday_specialist_policy_identity(POLICY)

        self.assertEqual(first, second)
        self.assertEqual(len(first["file_sha256"]), 64)
        self.assertEqual(len(first["policy_sha256"]), 64)

    def test_unknown_setup_feature_fails_closed(self) -> None:
        with POLICY.open("rb") as handle:
            payload = tomllib.load(handle)
        strategy = payload["strategies"][INTRADAY_SPECIALIST_IDS[0]]
        strategy["setup_rules"][0]["feature"] = "future_return"

        with self.assertRaises(ValueError):
            IntradaySpecialistResearchConfig.model_validate(payload)

    def test_missing_policy_fails_closed(self) -> None:
        with TemporaryDirectory() as directory:
            missing = Path(directory) / "missing.toml"
            with self.assertRaisesRegex(
                DataReadinessError,
                "missing KS4 research policy",
            ):
                load_intraday_specialist_research_config(missing)


if __name__ == "__main__":
    unittest.main()
