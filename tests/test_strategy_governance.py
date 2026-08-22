from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from pydantic import ValidationError

from market_predictor.strategy_governance import (
    StrategyExecutionLedger,
    validate_strategy_execution_ledger,
)
from market_predictor.core.errors import (
    ArtifactIntegrityError,
    DataReadinessError,
)

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
LEDGER_PATH = REPOSITORY_ROOT / "docs" / "strategy_execution_ledger.json"


class StrategyGovernanceTests(unittest.TestCase):
    def test_repository_ledger_covers_the_frozen_plan(self) -> None:
        report = validate_strategy_execution_ledger(
            LEDGER_PATH,
            repository_root=REPOSITORY_ROOT,
            verify_git=False,
        )

        self.assertTrue(report["valid"])
        self.assertEqual(report["checkpoint_count"], 10)
        self.assertEqual(report["catalog_count"], 25)
        self.assertEqual(report["next_checkpoint"], "KS5")

    def test_garch_family_is_explicit_and_owned_by_ks6(self) -> None:
        ledger = StrategyExecutionLedger.model_validate_json(
            LEDGER_PATH.read_text(encoding="utf-8")
        )
        by_id = {entry.item_id: entry for entry in ledger.catalog}

        self.assertEqual(by_id["RISK.GARCH.60M.V1"].checkpoint_id, "KS6")
        self.assertEqual(by_id["RISK.GARCH.5D.V1"].checkpoint_id, "KS6")
        self.assertEqual(by_id["RISK.EGARCH.60M.V1"].checkpoint_id, "KS6")
        self.assertIn("QLIKE", " ".join(by_id["RISK.GARCH.60M.V1"].required_evidence))

    def test_completed_checkpoint_cannot_omit_evidence(self) -> None:
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        checkpoint = next(
            item
            for item in payload["checkpoints"]
            if item["checkpoint_id"] == "KS0"
        )
        checkpoint["evidence"] = []
        for gate in checkpoint["exit_gates"]:
            gate["evidence_ids"] = []
        for verification in checkpoint["verification"]:
            verification["evidence_ids"] = []

        with self.assertRaisesRegex(
            ValidationError,
            "completed checkpoint requires evidence artifacts",
        ):
            StrategyExecutionLedger.model_validate(payload)

    def test_catalog_omission_fails_plan_alignment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_governance_files(root)
            ledger_path = root / "docs" / "strategy_execution_ledger.json"
            payload = json.loads(ledger_path.read_text(encoding="utf-8"))
            payload["catalog"] = payload["catalog"][:-1]
            ledger_path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(
                DataReadinessError,
                "strategy catalog mismatch",
            ):
                validate_strategy_execution_ledger(
                    ledger_path,
                    repository_root=root,
                    verify_git=False,
                )

    def test_plan_mutation_fails_hash_binding(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self._copy_governance_files(root)
            plan_path = (
                root
                / "docs"
                / "known_strategy_expansion_sequence_2026-07-26.md"
            )
            plan_path.write_text(
                plan_path.read_text(encoding="utf-8") + "\nmutation\n",
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                ArtifactIntegrityError,
                "strategy plan hash",
            ):
                validate_strategy_execution_ledger(
                    root / "docs" / "strategy_execution_ledger.json",
                    repository_root=root,
                    verify_git=False,
                )

    def test_repository_path_traversal_is_rejected(self) -> None:
        payload = json.loads(LEDGER_PATH.read_text(encoding="utf-8"))
        payload["plan"]["path"] = "../outside.md"

        with self.assertRaisesRegex(
            ValidationError,
            "normalized and relative",
        ):
            StrategyExecutionLedger.model_validate(payload)

    @staticmethod
    def _copy_governance_files(root: Path) -> None:
        docs = root / "docs"
        evidence = docs / "evidence"
        docs.mkdir(parents=True)
        evidence.mkdir()
        for name in (
            "strategy_execution_ledger.json",
            "known_strategy_expansion_sequence_2026-07-26.md",
            "catalyst_confirmation_architecture.md",
        ):
            source = REPOSITORY_ROOT / "docs" / name
            (docs / name).write_bytes(source.read_bytes())
        for name in (
            "ks0_strategy_research_contracts.json",
            "ks0_verification_20260726.json",
            "ks1_catalyst_lineage_replay_20260726.json",
            "ks1_verification_20260726.json",
            "ks2_strategy_label_replay_20260726.json",
            "ks2_verification_20260726.json",
            "ks3_swing_specialist_replay_20260727.json",
            "ks3_verification_20260727.json",
            "ks4_intraday_specialist_replay_20260728.json",
            "ks4_verification_20260728.json",
        ):
            source = REPOSITORY_ROOT / "docs" / "evidence" / name
            (evidence / name).write_bytes(source.read_bytes())


if __name__ == "__main__":
    unittest.main()
