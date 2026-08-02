from __future__ import annotations

import json
import tempfile
import unittest
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

from typer.testing import CliRunner

from market_predictor.feature_store import LiveFeatureStore
from market_predictor.intraday.contracts import INTRADAY_FEATURE_SCHEMA_VERSION
from market_predictor.production_cli import app
from market_predictor.release import publish_local_release
from market_predictor.serving_bundle import (
    activate_serving_bundle,
    load_active_serving_bundle,
    publish_serving_bundle,
    rollback_serving_bundle,
    verify_serving_bundle,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import test_signing_material as signing_material_for_test
from tests.test_serving_context import _promoted_intraday_model, _publish_intraday


class ServingBundleTests(unittest.TestCase):
    def test_cli_publishes_complete_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, release_id, feature_path = _inputs(root, "cli")

            result = CliRunner().invoke(
                app,
                [
                    "publish-serving-bundle",
                    "--mode",
                    "intraday",
                    "--horizon",
                    "60m",
                    "--model-release-id",
                    release_id,
                    "--feature-snapshot",
                    str(feature_path),
                    "--release-root",
                    str(repository),
                    "--attestation-trust-store",
                    str(trust_store),
                    "--generated-at",
                    _timestamp().isoformat(),
                ],
            )

            self.assertEqual(
                result.exit_code,
                0,
                msg=f"{result.output}\n{result.exception}",
            )
            self.assertTrue((repository / "active_serving_bundle.json").is_file())

    def test_publishes_and_loads_one_complete_atomic_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, release_id, feature_path = _inputs(root, "one")

            published = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=release_id,
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp(),
            )
            active = load_active_serving_bundle(
                repository,
                attestation_trust_store_path=trust_store,
                as_of=_timestamp(),
            )

            self.assertEqual(active["bundle"]["bundle_id"], published["bundle_id"])
            self.assertEqual(active["bundle"]["model_release_id"], release_id)
            self.assertEqual(
                active["bundle"]["feature_schema_version"],
                INTRADAY_FEATURE_SCHEMA_VERSION,
            )
            self.assertEqual(
                active["bundle"]["calibration_method"],
                "isotonic_prior_fold_only",
            )

    def test_partial_bundle_never_replaces_active_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, release_id, feature_path = _inputs(root, "complete")
            complete = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=release_id,
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp(),
            )
            partial_id = "f" * 64
            partial = repository / "serving_bundles" / partial_id
            partial.mkdir(parents=True)
            (partial / "bundle.json").write_text("{}", encoding="utf-8")

            with self.assertRaises(DataReadinessError):
                activate_serving_bundle(
                    repository,
                    partial_id,
                    attestation_trust_store_path=trust_store,
                    activated_at=_timestamp(),
                )

            active = load_active_serving_bundle(
                repository,
                attestation_trust_store_path=trust_store,
                as_of=_timestamp(),
            )
            self.assertEqual(active["bundle"]["bundle_id"], complete["bundle_id"])

    def test_feature_mutation_invalidates_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, release_id, feature_path = _inputs(root, "mutation")
            published = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=release_id,
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp(),
                activate=False,
            )
            bundle_id = str(published["bundle_id"])
            bundled_feature = (
                repository
                / "serving_bundles"
                / bundle_id
                / "features"
                / "features.parquet"
            )
            bundled_feature.write_bytes(b"mutated")

            with self.assertRaisesRegex(DataReadinessError, "feature artifact integrity"):
                verify_serving_bundle(
                    repository,
                    bundle_id,
                    attestation_trust_store_path=trust_store,
                    as_of=_timestamp(),
                )

    def test_rollback_targets_verified_immediately_previous_bundle(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, first_release, feature_path = _inputs(root, "first")
            first = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=first_release,
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp(),
            )
            second_model, second_evidence = _promoted_intraday_model(root / "second", "second")
            second_release = publish_local_release(
                repository,
                model_path=second_model,
                evidence_manifest_path=second_evidence,
                attestation_trust_store_path=trust_store,
                activate=False,
            )
            second = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=str(second_release["release_id"]),
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp() + timedelta(minutes=1),
            )

            rolled_back = rollback_serving_bundle(
                repository,
                str(first["bundle_id"]),
                attestation_trust_store_path=trust_store,
                activated_at=_timestamp() + timedelta(minutes=2),
            )

            self.assertEqual(rolled_back["bundle_id"], first["bundle_id"])
            self.assertEqual(rolled_back["previous_bundle_id"], second["bundle_id"])

    def test_concurrent_activation_leaves_one_verified_generation(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, first_release, feature_path = _inputs(root, "one")
            bundle_ids: list[str] = []
            for marker, release_id in (("one", first_release), ("two", "")):
                if not release_id:
                    model, evidence = _promoted_intraday_model(root / marker, marker)
                    release = publish_local_release(
                        repository,
                        model_path=model,
                        evidence_manifest_path=evidence,
                        attestation_trust_store_path=trust_store,
                        activate=False,
                    )
                    release_id = str(release["release_id"])
                bundle = publish_serving_bundle(
                    repository,
                    mode="intraday",
                    horizon="60m",
                    model_release_id=release_id,
                    feature_path=feature_path,
                    attestation_trust_store_path=trust_store,
                    generated_at=_timestamp(),
                    activate=False,
                )
                bundle_ids.append(str(bundle["bundle_id"]))

            with ThreadPoolExecutor(max_workers=2) as pool:
                pointers = list(
                    pool.map(
                        lambda bundle_id: activate_serving_bundle(
                            repository,
                            bundle_id,
                            attestation_trust_store_path=trust_store,
                            activated_at=_timestamp(),
                        ),
                        bundle_ids,
                    )
                )

            active = load_active_serving_bundle(
                repository,
                attestation_trust_store_path=trust_store,
                as_of=_timestamp(),
            )
            self.assertIn(active["bundle"]["bundle_id"], bundle_ids)
            self.assertEqual(
                {str(pointer["bundle_id"]) for pointer in pointers},
                set(bundle_ids),
            )

    def test_manifest_policy_mutation_breaks_content_identity(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            repository, trust_store, release_id, feature_path = _inputs(root, "policy")
            published = publish_serving_bundle(
                repository,
                mode="intraday",
                horizon="60m",
                model_release_id=release_id,
                feature_path=feature_path,
                attestation_trust_store_path=trust_store,
                generated_at=_timestamp(),
                activate=False,
            )
            bundle_id = str(published["bundle_id"])
            manifest_path = repository / "serving_bundles" / bundle_id / "bundle.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["prediction_policy_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "content hash"):
                verify_serving_bundle(
                    repository,
                    bundle_id,
                    attestation_trust_store_path=trust_store,
                    as_of=_timestamp(),
                )


def _inputs(root: Path, marker: str) -> tuple[Path, Path, str, Path]:
    repository = root / "repository"
    _, trust_store, _ = signing_material_for_test()
    model, evidence = _promoted_intraday_model(root / "models" / marker, marker)
    release = publish_local_release(
        repository,
        model_path=model,
        evidence_manifest_path=evidence,
        attestation_trust_store_path=trust_store,
        activate=False,
    )
    store = LiveFeatureStore(root)
    _publish_intraday(store, _timestamp())
    feature_path, _ = store.paths("intraday")
    return repository, trust_store, str(release["release_id"]), feature_path


def _timestamp() -> datetime:
    return datetime(2026, 7, 10, 22, 5, tzinfo=UTC)


if __name__ == "__main__":
    unittest.main()
