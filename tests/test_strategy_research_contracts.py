from __future__ import annotations

import json
import shutil
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from market_predictor.strategy_research_contracts import (
    ReferenceModelInventory,
    ResearchHypothesisRegistry,
    StrategyResearchGovernance,
    validate_strategy_research_contracts,
)
from market_predictor.v3.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]


class StrategyResearchContractTests(unittest.TestCase):
    def test_repository_ks0_contracts_are_complete(self) -> None:
        report = _validate(REPOSITORY_ROOT)

        self.assertTrue(report["valid"])
        self.assertEqual(report["catalog_count"], 25)
        self.assertEqual(report["hypothesis_count"], 25)
        self.assertEqual(report["reference_model_count"], 5)
        self.assertEqual(report["reference_models_serving_eligible"], 0)
        self.assertEqual(
            report["validation_scopes"],
            ["purged_walk_forward", "unseen_ticker_holdout"],
        )

    def test_garch_hypotheses_are_bounded_risk_claims(self) -> None:
        registry = ResearchHypothesisRegistry.model_validate_json(
            (
                REPOSITORY_ROOT / "docs" / "strategy_hypothesis_registry.json"
            ).read_text(encoding="utf-8")
        )
        by_id = {hypothesis.item_id: hypothesis for hypothesis in registry.hypotheses}

        intraday = by_id["RISK.GARCH.60M.V1"]
        swing = by_id["RISK.GARCH.5D.V1"]
        self.assertIn("conditional-variance", intraday.claim)
        self.assertIn("QLIKE", intraday.falsified_when)
        self.assertIn("drawdown", swing.falsified_when)
        self.assertNotIn("direction", intraday.primary_outcome.lower())

    def test_missing_hypothesis_fails_catalog_coverage(self) -> None:
        with _repository_copy() as root:
            path = root / "docs" / "strategy_hypothesis_registry.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["hypotheses"] = payload["hypotheses"][:-1]
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                DataReadinessError,
                "coverage differs",
            ):
                _validate(root)

    def test_hypothesis_state_cannot_diverge_from_ledger(self) -> None:
        with _repository_copy() as root:
            path = root / "docs" / "strategy_hypothesis_registry.json"
            payload = json.loads(path.read_text(encoding="utf-8"))
            payload["hypotheses"][0]["state"] = "planned"
            path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                DataReadinessError,
                "state mismatch",
            ):
                _validate(root)

    def test_research_policy_mutation_fails_registry_binding(self) -> None:
        with _repository_copy() as root:
            path = root / "configs" / "strategy_research_governance.toml"
            path.write_text(
                path.read_text(encoding="utf-8") + "\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "research policy hash mismatch",
            ):
                _validate(root)

    def test_bound_training_contract_mutation_fails(self) -> None:
        with _repository_copy() as root:
            path = root / "configs" / "intraday_training.toml"
            path.write_text(
                path.read_text(encoding="utf-8").replace(
                    "embargo_sessions = 1",
                    "embargo_sessions = 2",
                ),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "intraday_training contract hash mismatch",
            ):
                _validate(root)

    def test_experiment_budget_cannot_exceed_declared_dimensions(self) -> None:
        payload = _policy_payload()
        payload["maximum_development_experiments_per_strategy_version"] = 100

        with self.assertRaisesRegex(
            ValidationError,
            "experiment budget exceeds",
        ):
            StrategyResearchGovernance.model_validate(payload)

    def test_reference_model_cannot_be_assigned_to_strategy_or_serving(self) -> None:
        path = REPOSITORY_ROOT / "docs" / "reference_model_inventory.json"
        payload = json.loads(path.read_text(encoding="utf-8"))
        payload["models"][0]["strategy_id"] = "SWING.CATALYST_DRIFT.5D.V1"
        payload["models"][0]["serving_eligible"] = True

        with self.assertRaises(ValidationError):
            ReferenceModelInventory.model_validate(payload)


def _validate(root: Path) -> dict[str, object]:
    return validate_strategy_research_contracts(
        ledger_path=Path("docs/strategy_execution_ledger.json"),
        hypothesis_registry_path=Path("docs/strategy_hypothesis_registry.json"),
        policy_path=Path("configs/strategy_research_governance.toml"),
        reference_inventory_path=Path("docs/reference_model_inventory.json"),
        repository_root=root,
        verify_git=False,
    )


def _policy_payload() -> dict[str, object]:
    import tomllib

    path = REPOSITORY_ROOT / "configs" / "strategy_research_governance.toml"
    return tomllib.loads(path.read_text(encoding="utf-8"))


class _repository_copy:
    def __init__(self) -> None:
        self._temporary: tempfile.TemporaryDirectory[str] | None = None

    def __enter__(self) -> Path:
        self._temporary = tempfile.TemporaryDirectory()
        root = Path(self._temporary.name)
        docs = root / "docs"
        configs = root / "configs"
        evidence = docs / "evidence"
        model_cards = docs / "model_cards"
        docs.mkdir()
        configs.mkdir()
        evidence.mkdir()
        model_cards.mkdir()

        for name in (
            "strategy_execution_ledger.json",
            "known_strategy_expansion_sequence_2026-07-26.md",
            "catalyst_confirmation_architecture.md",
            "strategy_hypothesis_registry.json",
            "reference_model_inventory.json",
        ):
            shutil.copy2(REPOSITORY_ROOT / "docs" / name, docs / name)
        for name in (
            "ks0_strategy_research_contracts.json",
            "ks0_verification_20260726.json",
            "ks1_catalyst_lineage_replay_20260726.json",
            "ks1_verification_20260726.json",
            "ks2_strategy_label_replay_20260726.json",
            "ks2_verification_20260726.json",
            "ks3_swing_specialist_replay_20260727.json",
            "ks3_verification_20260727.json",
        ):
            shutil.copy2(
                REPOSITORY_ROOT / "docs" / "evidence" / name,
                evidence / name,
            )
        for name in (
            "swing_technical_5d_logistic_20260725.md",
            "swing_technical_5d_hgb_20260725.md",
            "v3_c8_r1_20260720.md",
            "v4_h1_120m_20260721.md",
        ):
            shutil.copy2(
                REPOSITORY_ROOT / "docs" / "model_cards" / name,
                model_cards / name,
            )
        for name in (
            "strategy_research_governance.toml",
            "catalyst_lineage.toml",
            "swing_dataset.toml",
            "swing_training.toml",
            "swing_promotion.toml",
            "intraday_dataset.toml",
            "intraday_training.toml",
            "intraday_promotion.toml",
        ):
            shutil.copy2(REPOSITORY_ROOT / "configs" / name, configs / name)
        return root

    def __exit__(
        self,
        exc_type: object,
        exc_value: object,
        traceback: object,
    ) -> None:
        if self._temporary is not None:
            self._temporary.cleanup()


if __name__ == "__main__":
    unittest.main()
