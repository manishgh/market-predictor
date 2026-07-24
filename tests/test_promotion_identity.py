from __future__ import annotations

import hashlib
import json
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
from cryptography.hazmat.primitives import serialization
from cryptography.hazmat.primitives.asymmetric import rsa
from pydantic import SecretStr

from market_predictor.promotion_identity import (
    APPROVER_ROLE,
    BUILD_ROLE,
    PromotionIdentityAuthenticator,
    PromotionTokens,
)
from market_predictor.v3.errors import DataReadinessError
from tests.r4_fixtures import test_promotion_identity_material


class PromotionIdentityTests(unittest.TestCase):
    def test_valid_pair_is_authenticated_and_token_material_is_not_exposed(
        self,
    ) -> None:
        config, tokens = test_promotion_identity_material()

        build, approver = PromotionIdentityAuthenticator(
            config
        ).authenticate_pair(tokens)

        self.assertEqual(build["required_role"], BUILD_ROLE)
        self.assertEqual(approver["required_role"], APPROVER_ROLE)
        self.assertNotEqual(build["principal_id"], approver["principal_id"])
        self.assertEqual(
            build["jwks_sha256"],
            hashlib.sha256(config.jwks_path.read_bytes()).hexdigest(),
        )
        serialized = json.dumps([build, approver], sort_keys=True)
        self.assertNotIn(tokens.build.get_secret_value(), serialized)
        self.assertNotIn(tokens.approver.get_secret_value(), serialized)

    def test_swapped_roles_fail_closed(self) -> None:
        config, tokens = test_promotion_identity_material()

        with self.assertRaisesRegex(DataReadinessError, "required role"):
            PromotionIdentityAuthenticator(config).authenticate_pair(
                PromotionTokens(
                    build=tokens.approver,
                    approver=tokens.build,
                )
            )

    def test_same_authenticated_principal_cannot_build_and_approve(self) -> None:
        config, _ = test_promotion_identity_material()
        key = _private_key(config.jwks_path.parent / "test-rsa-private.pem")
        now = datetime.now(UTC)

        with self.assertRaisesRegex(DataReadinessError, "must be distinct"):
            PromotionIdentityAuthenticator(config).authenticate_pair(
                PromotionTokens(
                    build=SecretStr(
                        _token(
                            key,
                            issuer=config.issuer,
                            audience=config.audience,
                            principal="same-principal",
                            role=BUILD_ROLE,
                            token_id="same-build",
                            now=now,
                        )
                    ),
                    approver=SecretStr(
                        _token(
                            key,
                            issuer=config.issuer,
                            audience=config.audience,
                            principal="same-principal",
                            role=APPROVER_ROLE,
                            token_id="same-approver",
                            now=now,
                        )
                    ),
                )
            )

    def test_wrong_audience_and_missing_jti_fail_closed(self) -> None:
        config, tokens = test_promotion_identity_material()
        key = _private_key(config.jwks_path.parent / "test-rsa-private.pem")
        now = datetime.now(UTC)
        wrong_audience = SecretStr(
            _token(
                key,
                issuer=config.issuer,
                audience="wrong-promotion-audience",
                principal="test-build-principal",
                role=BUILD_ROLE,
                token_id="wrong-audience",
                now=now,
            )
        )
        missing_jti = SecretStr(
            _token(
                key,
                issuer=config.issuer,
                audience=config.audience,
                principal="test-build-principal",
                role=BUILD_ROLE,
                token_id=None,
                now=now,
            )
        )

        for invalid in (wrong_audience, missing_jti):
            with self.subTest(token=invalid):
                with self.assertRaisesRegex(
                    DataReadinessError,
                    "authentication failed",
                ):
                    PromotionIdentityAuthenticator(config).authenticate_pair(
                        PromotionTokens(
                            build=invalid,
                            approver=tokens.approver,
                        )
                    )


def _private_key(path: Path) -> rsa.RSAPrivateKey:
    loaded = serialization.load_pem_private_key(
        path.read_bytes(),
        password=None,
    )
    if not isinstance(loaded, rsa.RSAPrivateKey):
        raise AssertionError("test private key is not RSA")
    return loaded


def _token(
    private_key: rsa.RSAPrivateKey,
    *,
    issuer: str,
    audience: str,
    principal: str,
    role: str,
    token_id: str | None,
    now: datetime,
) -> str:
    claims: dict[str, object] = {
        "iss": issuer,
        "aud": audience,
        "sub": principal,
        "oid": principal,
        "tid": "test-tenant",
        "roles": [role],
        "iat": int(now.timestamp()) - 1,
        "exp": int((now + timedelta(minutes=10)).timestamp()),
    }
    if token_id is not None:
        claims["jti"] = token_id
    return jwt.encode(
        claims,
        private_key,
        algorithm="RS256",
        headers={"kid": "promotion-test-key"},
    )
