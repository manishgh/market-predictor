from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from market_predictor.causal_shadow import load_causal_shadow_bundle
from market_predictor.hypothesis_registry import load_hypothesis
from market_predictor.outcome_contracts import content_sha256
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.promotion_attestation import file_sha256
from market_predictor.shadow_ledger import shadow_gate_failures
from market_predictor.v3.errors import DataReadinessError
from scripts.promotion_fixture import (
    synthetic_identity_metrics,
    trust_context_for_candidate,
)


class CausalShadowTests(unittest.TestCase):
    def test_bundle_reproduces_from_paired_matured_rows(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "candidate.joblib"
            model.write_bytes(b"candidate")
            metrics = synthetic_identity_metrics(
                model_type="canonical_swing",
                model_run_id="causal-shadow-test",
            )
            context = trust_context_for_candidate(
                root / "governance",
                model_path=model,
                metrics=metrics,
                model_type="canonical_swing",
                improvements=[0.02, 0.01, 0.03, 0.015],
            )
            hypothesis = load_hypothesis(
                context.hypothesis_registry_root,
                context.hypothesis_id,
            )

            bundle = load_causal_shadow_bundle(
                context.shadow_bundle_path,
                repository=OutcomeRepository(
                    context.outcome_repository_root
                ),
                hypothesis=hypothesis,
            )

            self.assertEqual(
                bundle["candidate_artifact_sha256"],
                file_sha256(model),
            )
            self.assertEqual(bundle["independent_sessions"], 4)
            np.testing.assert_allclose(
                [
                    row["candidate_benchmark_excess_return"]
                    for row in bundle["session_returns"]
                ],
                [0.02, 0.01, 0.03, 0.015],
                rtol=0.0,
                atol=1e-15,
            )
            self.assertEqual(
                shadow_gate_failures(
                    bundle,
                    minimum_independent_sessions=4,
                    minimum_paired_improvement_ci_low=0.0,
                ),
                [],
            )

    def test_outcome_mutation_invalidates_existing_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "candidate.joblib"
            model.write_bytes(b"candidate")
            metrics = synthetic_identity_metrics(
                model_type="canonical_swing",
                model_run_id="causal-shadow-poison",
            )
            context = trust_context_for_candidate(
                root / "governance",
                model_path=model,
                metrics=metrics,
                model_type="canonical_swing",
            )
            hypothesis = load_hypothesis(
                context.hypothesis_registry_root,
                context.hypothesis_id,
            )
            outcome_path = next(
                context.outcome_repository_root.glob("outcomes/*/*.json")
            )
            outcome = json.loads(outcome_path.read_text(encoding="utf-8"))
            outcome["net_return"] = float(outcome["net_return"]) + 0.50
            outcome["excess_return_vs_spy"] = (
                float(outcome["net_return"])
                - float(outcome["spy_return"])
            )
            identity = dict(outcome)
            identity.pop("outcome_id")
            outcome["outcome_id"] = content_sha256(identity)
            outcome_path.write_text(
                json.dumps(outcome, sort_keys=True),
                encoding="utf-8",
            )

            with self.assertRaisesRegex(
                DataReadinessError,
                "do not reproduce",
            ):
                load_causal_shadow_bundle(
                    context.shadow_bundle_path,
                    repository=OutcomeRepository(
                        context.outcome_repository_root
                    ),
                    hypothesis=hypothesis,
                )

    def test_missing_frozen_group_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model = root / "candidate.joblib"
            model.write_bytes(b"candidate")
            metrics = synthetic_identity_metrics(
                model_type="canonical_swing",
                model_run_id="causal-shadow-gap",
            )
            context = trust_context_for_candidate(
                root / "governance",
                model_path=model,
                metrics=metrics,
                model_type="canonical_swing",
            )
            hypothesis = load_hypothesis(
                context.hypothesis_registry_root,
                context.hypothesis_id,
            )
            intent_path = next(
                context.outcome_repository_root.glob("intents/*/*.json")
            )
            intent_path.unlink()

            with self.assertRaisesRegex(
                DataReadinessError,
                "every frozen decision group",
            ):
                load_causal_shadow_bundle(
                    context.shadow_bundle_path,
                    repository=OutcomeRepository(
                        context.outcome_repository_root
                    ),
                    hypothesis=hypothesis,
                )
