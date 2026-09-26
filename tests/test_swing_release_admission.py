"""Current CLI admission cannot activate historical day-trading evidence."""

from __future__ import annotations

import hashlib
import json
import shutil
from datetime import datetime, tzinfo
from pathlib import Path
from unittest.mock import patch

import pytest
from typer.testing import CliRunner

import market_predictor.commands.release as release_commands
import market_predictor.release as release_module
import market_predictor.serving.admission as admission
from market_predictor.core.errors import DataReadinessError
from market_predictor.feature_store import LiveFeatureStore
from market_predictor.production_cli import app
from market_predictor.promotion_attestation import promotion_attestation_path_for
from market_predictor.release import publish_local_release, verify_local_release
from market_predictor.serving.bundle import SERVING_BUNDLE_MANIFEST, publish_serving_bundle
from tests.r4_fixtures import test_signing_material as signing_material_for_test
from tests.support.swing_release import promoted_swing_candidate, retired_intraday_candidate
from tests.test_feature_store import _frame as swing_frame
from tests.test_feature_store import _publish as publish_swing_features
from tests.test_serving_bundle import _inputs as historical_bundle_inputs
from tests.test_serving_bundle import _timestamp as bundle_timestamp


def _publish_args(model: Path, evidence: Path, repository: Path) -> list[str]:
    _, trust, _ = signing_material_for_test()
    return [
        "publish-local-release",
        "--model",
        str(model),
        "--evidence-manifest",
        str(evidence),
        "--release-root",
        str(repository),
        "--attestation-trust-store",
        str(trust),
    ]


@pytest.mark.parametrize("mode", ("intraday", "unified", "unknown"))
@pytest.mark.parametrize("command", ("publish-live-features", "publish-serving-bundle"))
def test_unsupported_mode_rejected_before_io(tmp_path: Path, mode: str, command: str) -> None:
    args = [command, "--mode", mode]
    if command == "publish-live-features":
        args += ["--input-path", str(tmp_path / "absent.parquet"), "--live-dir", str(tmp_path / "live")]
    else:
        args += [
            "--horizon",
            "10b",
            "--model-release-id",
            "a" * 64,
            "--feature-snapshot",
            str(tmp_path / "absent.parquet"),
            "--attestation-trust-store",
            str(tmp_path / "absent.json"),
            "--release-root",
            str(tmp_path / "releases"),
        ]
    result = CliRunner().invoke(app, args)
    assert result.exit_code == 2
    assert "mode must be swing" in result.output
    assert not list(tmp_path.iterdir())


def test_serving_horizon_is_checked_before_io(tmp_path: Path) -> None:
    result = CliRunner().invoke(
        app,
        [
            "publish-serving-bundle",
            "--mode",
            "swing",
            "--horizon",
            "60m",
            "--model-release-id",
            "a" * 64,
            "--feature-snapshot",
            str(tmp_path / "missing.parquet"),
            "--attestation-trust-store",
            str(tmp_path / "missing.json"),
            "--release-root",
            str(tmp_path / "release"),
        ],
    )
    assert result.exit_code == 2
    assert "horizon must be 10b" in result.output
    assert not list(tmp_path.iterdir())


@pytest.mark.parametrize("activation", ("--activate", "--no-activate"))
def test_intraday_publication_rejected_without_output(tmp_path: Path, activation: str) -> None:
    model, evidence = retired_intraday_candidate(tmp_path / "source", "rejected")
    repository = tmp_path / "repository"
    result = CliRunner().invoke(app, [*_publish_args(model, evidence, repository), activation])
    assert isinstance(result.exception, DataReadinessError)
    assert "requires a swing model" in str(result.exception)
    assert not repository.exists()


@pytest.mark.parametrize("command", ("activate-local-release", "rollback-local-release"))
def test_historical_release_cannot_change_active_pointer(
    tmp_path: Path, command: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    model, evidence = retired_intraday_candidate(tmp_path / "source", "historical")
    _, trust, _ = signing_material_for_test()
    repository = tmp_path / "repository"
    release = _pre_retirement_release(repository, model, evidence, trust, monkeypatch)
    pointer = repository / "active_release.json"
    before = pointer.read_bytes()
    result = CliRunner().invoke(
        app,
        [command, "--release-id", str(release["release_id"]), "--release-root", str(repository), "--attestation-trust-store", str(trust)],
    )
    assert isinstance(result.exception, DataReadinessError)
    assert "intraday model releases are retired" in str(result.exception)
    assert pointer.read_bytes() == before
    with pytest.raises(DataReadinessError, match="intraday model releases are retired"):
        verify_local_release(repository, str(release["release_id"]), attestation_trust_store_path=trust)


def _pre_retirement_release(
    repository: Path, model: Path, evidence: Path, trust: Path, monkeypatch: pytest.MonkeyPatch
) -> dict[str, object]:
    """Publish as the code did before retirement, when day-trading evidence was accepted."""
    with monkeypatch.context() as patched:
        patched.setattr(release_module, "_validate_evidence_schema", lambda *_args, **_kwargs: None)
        return publish_local_release(
            repository, model_path=model, evidence_manifest_path=evidence, attestation_trust_store_path=trust
        )


@pytest.mark.parametrize("command", ("activate-serving-bundle", "rollback-serving-bundle"))
def test_historical_bundle_cannot_change_active_pointer(tmp_path: Path, command: str) -> None:
    repository, trust, release_id, features = historical_bundle_inputs(tmp_path, "historical")
    swing = publish_serving_bundle(
        repository,
        mode="swing",
        horizon="10b",
        model_release_id=release_id,
        feature_path=features,
        attestation_trust_store_path=trust,
        generated_at=bundle_timestamp(),
    )
    historical = _rehashed_intraday_bundle(repository, str(swing["bundle_id"]))
    pointer = repository / "active_serving_bundle.json"
    before = pointer.read_bytes()
    result = CliRunner().invoke(
        app, [command, "--bundle-id", historical, "--release-root", str(repository), "--attestation-trust-store", str(trust)]
    )
    assert isinstance(result.exception, DataReadinessError)
    assert "intraday serving bundles are retired" in str(result.exception)
    assert pointer.read_bytes() == before


def test_intraday_bundle_publication_is_refused_before_io(tmp_path: Path) -> None:
    with pytest.raises(DataReadinessError, match="intraday serving bundles are retired"):
        publish_serving_bundle(
            tmp_path / "repository",
            mode="intraday",  # type: ignore[arg-type]
            horizon="60m",
            model_release_id="a" * 64,
            feature_path=tmp_path / "absent.parquet",
            attestation_trust_store_path=tmp_path / "absent.json",
        )
    assert not list(tmp_path.iterdir())


def _rehashed_intraday_bundle(repository: Path, bundle_id: str) -> str:
    """A self-consistent bundle as a pre-retirement day-trading publication recorded it."""
    bundles = repository / "serving_bundles"
    bundle = json.loads((bundles / bundle_id / SERVING_BUNDLE_MANIFEST).read_bytes())
    identity = {key: value for key, value in bundle.items() if key != "bundle_id"}
    identity.update({"mode": "intraday", "horizon": "60m"})
    historical = hashlib.sha256(
        json.dumps(identity, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    shutil.copytree(bundles / bundle_id, bundles / historical)
    (bundles / historical / SERVING_BUNDLE_MANIFEST).write_text(
        json.dumps({**identity, "bundle_id": historical}), encoding="utf-8"
    )
    return historical


def test_source_replacement_cannot_activate_retired_model(tmp_path: Path) -> None:
    swing, swing_evidence = promoted_swing_candidate(tmp_path / "swing", "before-race")
    intraday, intraday_evidence = retired_intraday_candidate(tmp_path / "intraday", "after-race")
    repository = tmp_path / "repository"

    def replaced_source(root, **kwargs):
        assert kwargs["activate"] is False
        with patch.object(release_module, "_validate_evidence_schema"):
            return publish_local_release(
                root,
                model_path=intraday,
                evidence_manifest_path=intraday_evidence,
                activate=False,
                attestation_trust_store_path=kwargs["attestation_trust_store_path"],
            )

    with patch.object(release_commands, "publish_local_release", side_effect=replaced_source):
        result = CliRunner().invoke(app, _publish_args(swing, swing_evidence, repository))
    assert isinstance(result.exception, DataReadinessError)
    assert "intraday model releases are retired" in str(result.exception)
    assert not (repository / "active_release.json").exists()
    # Rejected immutable evidence remains inspectable; no automatic cleanup.
    assert len(list((repository / "releases").glob("*/release.json"))) == 1


def test_release_manifest_replacement_is_detected_on_exact_parsed_bytes(tmp_path: Path) -> None:
    model, evidence = promoted_swing_candidate(tmp_path / "source", "manifest-race")
    _, trust, _ = signing_material_for_test()
    repository = tmp_path / "repository"
    published = publish_local_release(
        repository, model_path=model, evidence_manifest_path=evidence, activate=False, attestation_trust_store_path=trust
    )
    release_id = str(published["release_id"])
    verified = verify_local_release(repository, release_id, attestation_trust_store_path=trust)
    path = repository / "releases" / release_id / str(verified["candidate_manifest_path"])
    changed = json.loads(path.read_bytes())
    changed["model_type"] = "canonical_intraday"
    path.write_text(json.dumps(changed), encoding="utf-8")
    with patch.object(admission, "verify_local_release", return_value=verified):
        with pytest.raises(DataReadinessError, match="changed during release admission"):
            admission.require_swing_release(repository, release_id, trust)


@pytest.mark.parametrize("model_type", (None, "", "unknown", "canonical_intraday"))
def test_missing_or_unknown_model_type_is_not_swing(model_type: str | None) -> None:
    with patch.object(admission, "verify_model_artifact", return_value={"model_type": model_type}):
        with pytest.raises(DataReadinessError, match="requires a swing model"):
            admission.require_swing_candidate(Path("unused"), Path("unused"))


def test_swing_publish_activate_and_rollback_preserve_existing_rules(tmp_path: Path) -> None:
    _, trust, _ = signing_material_for_test()
    repository = tmp_path / "repository"
    ids = []
    for marker in ("first", "second", "third"):
        model, evidence = promoted_swing_candidate(tmp_path / marker, marker)
        result = CliRunner().invoke(app, _publish_args(model, evidence, repository))
        assert result.exit_code == 0, (result.output, result.exception)
        pointer = json.loads((repository / "active_release.json").read_bytes())
        ids.append(pointer["release_id"])
    pointer_path = repository / "active_release.json"
    before = pointer_path.read_bytes()
    base = ["--release-root", str(repository), "--attestation-trust-store", str(trust)]
    invalid = CliRunner().invoke(app, ["rollback-local-release", "--release-id", ids[0], *base])
    assert isinstance(invalid.exception, DataReadinessError)
    assert "immediately previous" in str(invalid.exception)
    assert pointer_path.read_bytes() == before
    valid = CliRunner().invoke(app, ["rollback-local-release", "--release-id", ids[1], *base])
    assert valid.exit_code == 0, (valid.output, valid.exception)
    assert json.loads(pointer_path.read_bytes())["release_id"] == ids[1]
    activate = CliRunner().invoke(app, ["activate-local-release", "--release-id", ids[2], *base])
    assert activate.exit_code == 0, (activate.output, activate.exception)
    assert json.loads(pointer_path.read_bytes())["release_id"] == ids[2]


def test_strict_manifest_parse_does_not_accept_duplicate_keys(tmp_path: Path) -> None:
    _, trust, _ = signing_material_for_test()
    release_id = "a" * 64
    relative = "model/candidate.json"
    path = tmp_path / "releases" / release_id / relative
    path.parent.mkdir(parents=True)
    payload = b'{"model_type":"canonical_intraday","model_type":"canonical_swing"}'
    path.write_bytes(payload)
    bound_release = {
        "candidate_manifest_path": relative,
        "assets": [
            {
                "kind": "candidate_manifest",
                "destination": relative,
                "sha256": hashlib.sha256(payload).hexdigest(),
            }
        ],
    }
    with patch.object(admission, "verify_local_release", return_value=bound_release):
        with pytest.raises(DataReadinessError, match="duplicate key"):
            admission.require_swing_release(tmp_path, release_id, trust)


def test_swing_bundle_activation_and_rollback_keep_previous_generation_rule(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    class FixtureClock(datetime):
        @classmethod
        def now(cls, tz: tzinfo | None = None) -> datetime:
            return bundle_timestamp().astimezone(tz)

    monkeypatch.setattr("market_predictor.serving.bundle.datetime", FixtureClock)
    _, trust, _ = signing_material_for_test()
    repository = tmp_path / "repository"
    store = LiveFeatureStore(tmp_path)
    publish_swing_features(store, swing_frame(), bundle_timestamp())
    features, _ = store.paths("swing")
    ids = []
    for marker in ("first", "second", "third"):
        model, evidence = promoted_swing_candidate(tmp_path / marker, marker)
        release = publish_local_release(
            repository,
            model_path=model,
            evidence_manifest_path=evidence,
            activate=False,
            attestation_trust_store_path=trust,
        )
        bundle = publish_serving_bundle(
            repository,
            mode="swing",
            horizon="10b",
            model_release_id=str(release["release_id"]),
            feature_path=features,
            attestation_trust_store_path=trust,
            generated_at=bundle_timestamp(),
        )
        ids.append(str(bundle["bundle_id"]))
    pointer_path = repository / "active_serving_bundle.json"
    before = pointer_path.read_bytes()
    base = ["--release-root", str(repository), "--attestation-trust-store", str(trust)]
    invalid = CliRunner().invoke(app, ["rollback-serving-bundle", "--bundle-id", ids[0], *base])
    assert isinstance(invalid.exception, DataReadinessError)
    assert "immediately previous" in str(invalid.exception)
    assert pointer_path.read_bytes() == before
    valid = CliRunner().invoke(app, ["rollback-serving-bundle", "--bundle-id", ids[1], *base])
    assert valid.exit_code == 0, (valid.output, valid.exception)
    assert json.loads(pointer_path.read_bytes())["bundle_id"] == ids[1]
    activate = CliRunner().invoke(app, ["activate-serving-bundle", "--bundle-id", ids[2], *base])
    assert activate.exit_code == 0, (activate.output, activate.exception)
    assert json.loads(pointer_path.read_bytes())["bundle_id"] == ids[2]


@pytest.mark.parametrize("failure", ("untrusted_signer", "tampered_attestation"))
def test_cli_does_not_weaken_signature_verification(tmp_path: Path, failure: str) -> None:
    model, evidence = promoted_swing_candidate(tmp_path / "source", failure)
    repository = tmp_path / "repository"
    args = _publish_args(model, evidence, repository)
    if failure == "untrusted_signer":
        trust = tmp_path / "untrusted.json"
        trust.write_text(json.dumps({
            "schema": "market_predictor.attestation_trust_store.v1", "issuers": {},
        }), encoding="utf-8")
        args[args.index("--attestation-trust-store") + 1] = str(trust)
    else:
        path = promotion_attestation_path_for(model)
        payload = json.loads(path.read_bytes())
        payload["attestation_id"] = "0" * 64
        path.write_text(json.dumps(payload), encoding="utf-8")
    result = CliRunner().invoke(app, args)
    assert result.exit_code != 0
    assert "attestation" in str(result.exception)
    assert not repository.exists()
