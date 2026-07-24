"""Deterministic synthetic promotion material for tests and CI smoke releases."""

from __future__ import annotations

import json
import os
import tempfile
from base64 import b64encode
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any

import jwt
import pandas as pd
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from pydantic import SecretStr

from market_predictor.causal_shadow import write_causal_shadow_bundle
from market_predictor.execution_policy import EXECUTION_POLICY_SHA256
from market_predictor.hypothesis_registry import declare_hypothesis
from market_predictor.intraday.contracts import IntradayDatasetConfig
from market_predictor.outcome_contracts import (
    MaturedOutcomeV1,
    PredictionMaturationIntentV2,
    content_sha256,
    maturation_key_sha256,
    semantic_prediction_sha256,
)
from market_predictor.outcome_repository import OutcomeRepository
from market_predictor.prediction_policy import (
    parse_prediction_policy,
    prediction_policy_identity,
)
from market_predictor.promotion_attestation import (
    ATTESTATION_TRUST_STORE_ENV,
    ATTESTATION_TRUST_STORE_SCHEMA,
    SIGNATURE_ALGORITHM,
    file_sha256,
)
from market_predictor.promotion_identity import (
    APPROVER_ROLE,
    BUILD_ROLE,
    DEFAULT_APPROVER_TOKEN_ENV,
    DEFAULT_BUILD_TOKEN_ENV,
    PromotionIdentityAuthenticator,
    PromotionIdentityConfig,
    PromotionTokens,
)
from market_predictor.promotion_workflow import (
    PromotionTrustContext,
    evaluate_shadow_and_attest,
)
from market_predictor.registry import load_model_manifest
from market_predictor.swing.contracts import SwingDatasetConfig


def synthetic_identity_metrics(
    *,
    model_type: str,
    model_run_id: str,
    validation_split: str = "session_purged_walk_forward_and_ticker_holdout",
) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "model_run_id": model_run_id,
        "validation_split": validation_split,
        "holdout_assignment_cutoff_utc": "2026-01-30T23:00:00+00:00",
        "holdout_ticker_summary_sha256": "1" * 64,
        "feature_set_sha256": "2" * 64,
        "reconciliation_sha256": "3" * 64,
        "event_assignment_sha256": "a" * 64,
        "event_aggregate_sha256": "b" * 64,
        "label_material_sha256": "c" * 64,
        "label_source_reconciliation_sha256": "d" * 64,
        "dataset_label_config_sha256": "4" * 64,
        "calibration_method": "isotonic_prior_fold_only",
        "folds_causally_ordered": True,
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "dataset_sha256": "9" * 64,
        **prediction_policy_identity(),
    }
    if model_type == "canonical_swing":
        metrics["universe_identity_sha256"] = "5" * 64
    return metrics


def trust_context_for_candidate(
    root: Path,
    *,
    model_path: Path,
    metrics: dict[str, Any],
    model_type: str,
    hypothesis_suffix: str = "001",
    improvements: list[float] | None = None,
) -> PromotionTrustContext:
    signing_key, trust_store, signer_id = test_signing_material()
    run_id = str(metrics["model_run_id"])
    safe_run_id = "".join(character if character.isalnum() or character in "._-" else "-" for character in run_id)
    hypothesis_id = f"{safe_run_id}-{hypothesis_suffix}"
    family = f"{model_type.replace('_', '-')}-family"
    declared_at = datetime(2026, 1, 1, 12, 0, tzinfo=UTC)
    baseline_artifact = root / "baseline" / "baseline.joblib"
    baseline_artifact.parent.mkdir(parents=True, exist_ok=True)
    baseline_artifact.write_bytes(b"synthetic frozen baseline")
    baseline_sha = file_sha256(baseline_artifact)
    values = improvements or [0.02, 0.01, 0.03, 0.015]
    decisions = tuple(
        pd.Timestamp(value)
        .tz_localize("America/New_York")
        .tz_convert("UTC")
        .isoformat()
        for value in pd.date_range(
            "2026-02-02 16:00",
            periods=len(values),
            freq="B",
        )
    )
    hypothesis = declare_hypothesis(
        root,
        hypothesis_id=hypothesis_id,
        hypothesis_family=family,
        model_type=model_type,
        candidate_artifact_sha256=file_sha256(model_path),
        baseline_id="frozen-baseline-v1",
        baseline_artifact_sha256=baseline_sha,
        prediction_policy_sha256=str(metrics["prediction_policy_sha256"]),
        execution_policy_sha256=str(metrics["execution_policy_sha256"]),
        shadow_view=(
            "swing" if model_type == "canonical_swing" else "intraday"
        ),
        shadow_horizon=("5d" if model_type == "canonical_swing" else "60m"),
        shadow_decision_group_ids=decisions,
        shadow_minimum_tickers_per_group=1,
        objective="Synthetic test declaration for the immutable promotion trust path.",
        declared_at=declared_at,
    )
    outcome_repository = OutcomeRepository(root / "outcomes")
    _write_synthetic_shadow_outcomes(
        outcome_repository,
        model_type=model_type,
        candidate_sha=file_sha256(model_path),
        baseline_sha=baseline_sha,
        decisions=decisions,
        candidate_returns=values,
        prediction_policy=parse_prediction_policy(
            metrics["prediction_policy"],
            expected_sha256=str(metrics["prediction_policy_sha256"]),
        ).specification(),
        prediction_policy_sha=str(metrics["prediction_policy_sha256"]),
    )
    bundle = write_causal_shadow_bundle(
        root,
        outcome_repository,
        hypothesis=hypothesis,
        generated_at=declared_at + timedelta(days=60),
        bootstrap_iterations=200,
        bootstrap_seed=17,
    )
    identity_config, identity_tokens = test_promotion_identity_material()
    return PromotionTrustContext(
        hypothesis_registry_root=root,
        hypothesis_id=hypothesis_id,
        shadow_bundle_path=bundle,
        outcome_repository_root=outcome_repository.root,
        baseline_artifact_path=baseline_artifact,
        identity_config=identity_config,
        identity_tokens=identity_tokens,
        signing_private_key_path=signing_key,
        attestation_trust_store_path=trust_store,
        signer_id=signer_id,
        minimum_shadow_sessions=len(values),
        minimum_paired_improvement_ci_low=0.0,
    )


def _write_synthetic_shadow_outcomes(
    repository: OutcomeRepository,
    *,
    model_type: str,
    candidate_sha: str,
    baseline_sha: str,
    decisions: tuple[str, ...],
    candidate_returns: list[float],
    prediction_policy: dict[str, object],
    prediction_policy_sha: str,
) -> None:
    view = "swing" if model_type == "canonical_swing" else "intraday"
    horizon = "5d" if view == "swing" else "60m"
    label_config = (
        SwingDatasetConfig()
        if view == "swing"
        else IntradayDatasetConfig()
    )
    label_policy = label_config.label_policy()
    label_policy_sha = label_config.label_config_sha256()
    for index, (group_id, candidate_return) in enumerate(
        zip(decisions, candidate_returns, strict=True)
    ):
        decision = datetime.fromisoformat(group_id)
        for side, artifact_sha, net_return in (
            ("candidate", candidate_sha, candidate_return),
            ("baseline", baseline_sha, 0.0),
        ):
            snapshot_id = content_sha256(
                {
                    "side": side,
                    "decision_group_id": group_id,
                }
            )
            intent = _synthetic_intent(
                snapshot_id=snapshot_id,
                artifact_sha=artifact_sha,
                view=view,
                horizon=horizon,
                decision=decision,
                label_policy=label_policy,
                label_policy_sha=label_policy_sha,
                prediction_policy=prediction_policy,
                prediction_policy_sha=prediction_policy_sha,
            )
            repository.record_intent(intent)
            evidence = [
                {
                    "side": side,
                    "decision_group_id": group_id,
                    "ordinal": index,
                }
            ]
            repository.record_outcome(
                _synthetic_outcome(
                    intent,
                    net_return=net_return,
                    evidence=evidence,
                ),
                evidence_rows=evidence,
            )


def _synthetic_intent(
    *,
    snapshot_id: str,
    artifact_sha: str,
    view: str,
    horizon: str,
    decision: datetime,
    label_policy: dict[str, object],
    label_policy_sha: str,
    prediction_policy: dict[str, object],
    prediction_policy_sha: str,
) -> PredictionMaturationIntentV2:
    base: dict[str, object] = {
        "contract_version": "market_predictor.maturation_intent.v2",
        "ticker": "TEST",
        "canonical_security_id": "security:TEST",
        "view": view,
        "horizon": horizon,
        "decision_time_utc": decision,
        "decision_session_et": decision.date(),
        "decision_group_id": decision.isoformat(),
        "model_release_id": artifact_sha,
        "model_artifact_sha256": artifact_sha,
        "feature_artifact_sha256": "f" * 64,
        "prediction_policy_sha256": prediction_policy_sha,
        "label_policy_sha256": label_policy_sha,
        "execution_policy_sha256": EXECUTION_POLICY_SHA256,
        "prediction_policy": prediction_policy,
        "label_policy": label_policy,
        "primary_benchmark": "SPY",
        "market_regime": "neutral",
        "sector": "Technology",
        "market_cap_bucket": "large",
        "liquidity_bucket": "high",
        "price_feed": "SIP",
        "probability": 0.75,
        "downside_probability": 0.1 if view == "intraday" else None,
        "calibration_bin": 7,
        "signal": "bullish_watch",
        "rank": 1,
        "selection_eligible": True,
        "selected_for_policy": True,
        "actionable": True,
        "catalyst_status": "confirmed",
        "decision_atr": 1.0 if view == "intraday" else None,
    }
    semantic = semantic_prediction_sha256(base)
    return PredictionMaturationIntentV2.model_validate(
        {
            **base,
            "snapshot_id": snapshot_id,
            "semantic_prediction_id": semantic,
            "maturation_key": maturation_key_sha256(
                snapshot_id,
                semantic,
            ),
        }
    )


def _synthetic_outcome(
    intent: PredictionMaturationIntentV2,
    *,
    net_return: float,
    evidence: list[dict[str, object]],
) -> MaturedOutcomeV1:
    entry = intent.decision_time_utc + timedelta(minutes=5)
    exit_time = entry + timedelta(minutes=30)
    available = exit_time + timedelta(minutes=1)
    base = {
        "contract_version": "market_predictor.matured_outcome.v1",
        "maturation_key": intent.maturation_key,
        "semantic_prediction_id": intent.semantic_prediction_id,
        "snapshot_id": intent.snapshot_id,
        "ticker": intent.ticker,
        "view": intent.view,
        "horizon": intent.horizon,
        "entry_time_utc": entry,
        "exit_time_utc": exit_time,
        "label_available_at_utc": available,
        "matured_at_utc": available,
        "entry_price": 100.0,
        "exit_price": 100.0 * (1.0 + net_return),
        "gross_return": net_return,
        "net_return": net_return,
        "mfe": max(net_return, 0.0),
        "mae": min(net_return, 0.0),
        "path_outcome": "positive" if net_return > 0 else "negative",
        "opportunity_target": int(net_return > 0),
        "downside_target": int(net_return < 0) if intent.view == "intraday" else None,
        "spy_return": 0.0,
        "qqq_return": 0.0,
        "sector_return": 0.0,
        "excess_return_vs_spy": net_return,
        "excess_return_vs_qqq": net_return,
        "excess_return_vs_sector": net_return,
        "evidence_sha256": content_sha256(evidence),
    }
    return MaturedOutcomeV1.model_validate(
        {
            **base,
            "outcome_id": content_sha256(base),
        }
    )


def authorize_candidate_for_test(model_path: Path, metrics: dict[str, Any]) -> Path:
    root = model_path.parent / f".{model_path.name}.promotion-test"
    manifest = load_model_manifest(model_path)
    context = trust_context_for_candidate(
        root,
        model_path=model_path,
        metrics=metrics,
        model_type=str(manifest["model_type"]),
    )
    evidence_manifest = root / "evidence.manifest.json"
    evidence_manifest.parent.mkdir(parents=True, exist_ok=True)
    metrics_path = root / "metrics.json"
    metrics_path.write_text(json.dumps(metrics, sort_keys=True), encoding="utf-8")
    evidence_manifest.write_text(
        json.dumps(
            {
                "schema": (
                    "intraday_training_evidence.v1" if manifest["model_type"] == "canonical_intraday" else "swing_training_evidence.v1"
                ),
                "model_run_id": metrics["model_run_id"],
                "model_artifact_sha256": manifest["artifact_sha256"],
                "files": {
                    "metrics": {
                        "path": metrics_path.name,
                        "sha256": file_sha256(metrics_path),
                    }
                },
            },
            sort_keys=True,
        ),
        encoding="utf-8",
    )
    outcome = evaluate_shadow_and_attest(
        model_path=model_path,
        evidence_manifest_path=evidence_manifest,
        metrics=metrics,
        gate_config={"test_fixture": True},
        context=context,
    )
    if outcome.attestation is None:
        raise AssertionError(f"synthetic promotion fixture failed: {outcome.failures}")
    return evidence_manifest


def test_signing_material() -> tuple[Path, Path, str]:
    root = Path(tempfile.gettempdir()) / f"market-predictor-r4-signing-{os.getpid()}"
    root.mkdir(parents=True, exist_ok=True)
    key_path = root / "test-ed25519-private.pem"
    trust_store_path = root / "test-attestation-trust.json"
    signer_id = "test-ci-signer"
    if not key_path.exists():
        private_key = Ed25519PrivateKey.generate()
        key_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        public_key = private_key.public_key().public_bytes(
            encoding=serialization.Encoding.Raw,
            format=serialization.PublicFormat.Raw,
        )
        trust_store_path.write_text(
            json.dumps(
                {
                    "schema": ATTESTATION_TRUST_STORE_SCHEMA,
                    "issuers": {
                        signer_id: {
                            "algorithm": SIGNATURE_ALGORITHM,
                            "public_key_base64": b64encode(public_key).decode("ascii"),
                        }
                    },
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    os.environ[ATTESTATION_TRUST_STORE_ENV] = str(trust_store_path)
    os.environ["MARKET_PREDICTOR_ALLOW_TEST_CLOCK"] = "1"
    return key_path, trust_store_path, signer_id


def test_promotion_identity_material() -> tuple[
    PromotionIdentityConfig,
    PromotionTokens,
]:
    root = Path(tempfile.gettempdir()) / (
        f"market-predictor-promotion-identity-{os.getpid()}"
    )
    root.mkdir(parents=True, exist_ok=True)
    key_path = root / "test-rsa-private.pem"
    jwks_path = root / "test-jwks.json"
    issuer = "https://issuer.market-predictor.test"
    audience = "market-predictor-promotion"
    if not key_path.exists():
        private_key = rsa.generate_private_key(
            public_exponent=65_537,
            key_size=2_048,
        )
        key_path.write_bytes(
            private_key.private_bytes(
                encoding=serialization.Encoding.PEM,
                format=serialization.PrivateFormat.PKCS8,
                encryption_algorithm=serialization.NoEncryption(),
            )
        )
        numbers = private_key.public_key().public_numbers()
        jwks_path.write_text(
            json.dumps(
                {
                    "keys": [
                        {
                            "kty": "RSA",
                            "kid": "promotion-test-key",
                            "use": "sig",
                            "alg": "RS256",
                            "n": _base64url_uint(numbers.n),
                            "e": _base64url_uint(numbers.e),
                        }
                    ]
                },
                sort_keys=True,
            ),
            encoding="utf-8",
        )
    loaded = serialization.load_pem_private_key(
        key_path.read_bytes(),
        password=None,
    )
    if not isinstance(loaded, rsa.RSAPrivateKey):
        raise AssertionError("test promotion identity key is not RSA")
    now = datetime.now(UTC)
    build_token = _promotion_token(
        loaded,
        issuer=issuer,
        audience=audience,
        principal="test-build-principal",
        role=BUILD_ROLE,
        token_id=f"build-{os.getpid()}",
        now=now,
    )
    approver_token = _promotion_token(
        loaded,
        issuer=issuer,
        audience=audience,
        principal="test-approver-principal",
        role=APPROVER_ROLE,
        token_id=f"approver-{os.getpid()}",
        now=now,
    )
    os.environ[DEFAULT_BUILD_TOKEN_ENV] = build_token
    os.environ[DEFAULT_APPROVER_TOKEN_ENV] = approver_token
    return (
        PromotionIdentityConfig(
            issuer=issuer,
            audience=audience,
            jwks_path=jwks_path,
        ),
        PromotionTokens(
            build=SecretStr(build_token),
            approver=SecretStr(approver_token),
        ),
    )


def test_authenticated_promotion_principals() -> tuple[
    dict[str, Any],
    dict[str, Any],
]:
    config, tokens = test_promotion_identity_material()
    return PromotionIdentityAuthenticator(config).authenticate_pair(tokens)


def _promotion_token(
    private_key: rsa.RSAPrivateKey,
    *,
    issuer: str,
    audience: str,
    principal: str,
    role: str,
    token_id: str,
    now: datetime,
) -> str:
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "sub": principal,
            "oid": principal,
            "tid": "test-tenant",
            "azp": "promotion-test-client",
            "roles": [role],
            "jti": token_id,
            "iat": int(now.timestamp()) - 1,
            "exp": int((now + timedelta(minutes=10)).timestamp()),
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "promotion-test-key"},
    )


def _base64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return b64encode(raw).rstrip(b"=").replace(b"+", b"-").replace(
        b"/",
        b"_",
    ).decode("ascii")
