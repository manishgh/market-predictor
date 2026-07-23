from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import joblib
import pandas as pd
from typer.testing import CliRunner

from market_predictor.cli import app
from market_predictor.registry import write_model_manifest
from market_predictor.release import (
    activate_local_release,
    load_active_local_release,
    publish_local_release,
    rollback_local_release,
    verify_local_release,
)
from market_predictor.swing.contracts import (
    SWING_MODEL_SCHEMA_VERSION,
    SWING_MODEL_TYPE,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import (
    authorize_candidate_for_test,
    synthetic_identity_metrics,
    test_signing_material,
)


class LocalReleaseTests(unittest.TestCase):
    def test_cli_publishes_without_activation_when_requested(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, evidence = _promoted_candidate(root / "source", "cli")
            release_root = root / "repository"
            _, trust_store, _ = test_signing_material()

            result = CliRunner().invoke(
                app,
                [
                    "publish-local-release",
                    "--model",
                    str(model),
                    "--evidence-manifest",
                    str(evidence),
                    "--release-root",
                    str(release_root),
                    "--attestation-trust-store",
                    str(trust_store),
                    "--no-activate",
                ],
            )

            self.assertEqual(
                result.exit_code,
                0,
                msg=f"{result.output}\n{result.exception}",
            )
            self.assertFalse((release_root / "active_release.json").exists())
            self.assertEqual(len(list((release_root / "releases").glob("*/release.json"))), 1)

    def test_publishes_complete_release_before_switching_active_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, evidence = _promoted_candidate(root / "source", "first")
            release_root = root / "repository"

            published = publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence,
            )
            release_id = str(published["release_id"])
            active = load_active_local_release(release_root)
            repeated = publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence,
            )

            self.assertEqual(active["pointer"]["release_id"], release_id)
            self.assertEqual(repeated["release_id"], release_id)
            self.assertEqual(len(list((release_root / "releases").iterdir())), 1)
            self.assertEqual(active["release"]["attestation_id"], published["attestation_id"])
            self.assertTrue(
                (release_root / "releases" / release_id / "model" / model.name).is_file()
            )
            self.assertTrue(
                (
                    release_root
                    / "releases"
                    / release_id
                    / "evidence"
                    / "metrics.json"
                ).is_file()
            )

    def test_partial_release_never_replaces_active_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, evidence = _promoted_candidate(root / "source", "complete")
            release_root = root / "repository"
            complete = publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence,
            )
            complete_id = str(complete["release_id"])
            partial_id = "f" * 64
            partial = release_root / "releases" / partial_id
            partial.mkdir(parents=True)
            (partial / "release.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(DataReadinessError):
                activate_local_release(release_root, partial_id)

            active = load_active_local_release(release_root)
            self.assertEqual(active["pointer"]["release_id"], complete_id)

    def test_release_mutation_invalidates_verification_and_activation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, evidence = _promoted_candidate(root / "source", "mutated")
            release_root = root / "repository"
            published = publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence,
                activate=False,
            )
            release_id = str(published["release_id"])
            metrics = (
                release_root
                / "releases"
                / release_id
                / "evidence"
                / "metrics.json"
            )
            metrics.write_text('{"mutated":true}', encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "asset integrity"):
                verify_local_release(release_root, release_id)
            with self.assertRaises(DataReadinessError):
                activate_local_release(release_root, release_id)
            self.assertFalse((release_root / "active_release.json").exists())

    def test_release_manifest_metadata_mutation_invalidates_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            model, evidence = _promoted_candidate(root / "source", "manifest")
            release_root = root / "repository"
            published = publish_local_release(
                release_root,
                model_path=model,
                evidence_manifest_path=evidence,
                activate=False,
            )
            release_id = str(published["release_id"])
            manifest_path = (
                release_root / "releases" / release_id / "release.json"
            )
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["generated_at_utc"] = (
                _timestamp() + timedelta(days=1)
            ).isoformat()
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "content hash"):
                verify_local_release(release_root, release_id)

    def test_rollback_only_targets_a_verified_prior_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_root = root / "repository"
            first_model, first_evidence = _promoted_candidate(
                root / "source",
                "first",
            )
            second_model, second_evidence = _promoted_candidate(
                root / "source",
                "second",
            )
            first = publish_local_release(
                release_root,
                model_path=first_model,
                evidence_manifest_path=first_evidence,
            )
            second = publish_local_release(
                release_root,
                model_path=second_model,
                evidence_manifest_path=second_evidence,
            )

            rolled_back = rollback_local_release(
                release_root,
                str(first["release_id"]),
                activated_at=_timestamp() + timedelta(minutes=2),
            )

            self.assertEqual(rolled_back["release_id"], first["release_id"])
            self.assertEqual(rolled_back["previous_release_id"], second["release_id"])
            self.assertEqual(
                load_active_local_release(release_root)["pointer"]["release_id"],
                first["release_id"],
            )

    def test_concurrent_activation_keeps_one_complete_valid_release(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            release_root = root / "repository"
            releases: list[str] = []
            for marker in ("one", "two"):
                model, evidence = _promoted_candidate(root / "source", marker)
                published = publish_local_release(
                    release_root,
                    model_path=model,
                    evidence_manifest_path=evidence,
                    activate=False,
                )
                releases.append(str(published["release_id"]))

            with ThreadPoolExecutor(max_workers=2) as pool:
                pointers = list(
                    pool.map(
                        lambda release_id: activate_local_release(
                            release_root,
                            release_id,
                            activated_at=_timestamp(),
                        ),
                        releases,
                    )
                )

            active = load_active_local_release(release_root)
            active_id = str(active["pointer"]["release_id"])
            self.assertIn(active_id, releases)
            self.assertEqual({str(pointer["release_id"]) for pointer in pointers}, set(releases))
            self.assertEqual(
                active["pointer"]["previous_release_id"],
                next(release_id for release_id in releases if release_id != active_id),
            )


def _promoted_candidate(root: Path, marker: str) -> tuple[Path, Path]:
    root.mkdir(parents=True, exist_ok=True)
    model = root / f"swing-{marker}.joblib"
    joblib.dump({"marker": marker}, model)
    model_run_id = f"release-{marker}"
    metrics = {
        **synthetic_identity_metrics(
            model_type=SWING_MODEL_TYPE,
            model_run_id=model_run_id,
        ),
        "roc_auc": 0.75,
    }
    training = pd.DataFrame(
        {
            "ticker": ["AAA", "BBB"],
            "date": pd.date_range("2026-01-01", periods=2),
            "return_1d": [0.01, -0.01],
            "target_net_positive_5d": [1, 0],
        }
    )
    write_model_manifest(
        model_path=model,
        model_type=SWING_MODEL_TYPE,
        schema_version=SWING_MODEL_SCHEMA_VERSION,
        target_col="target_net_positive_5d",
        features=["return_1d"],
        training_data=training,
        metrics=metrics,
        validation_split="session_purged_walk_forward_and_ticker_holdout",
        extra={"model_run_id": model_run_id},
    )
    evidence = authorize_candidate_for_test(model, metrics)
    return model, evidence


def _timestamp() -> datetime:
    return datetime(2026, 7, 23, 12, 0, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
