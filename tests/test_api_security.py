from __future__ import annotations

import base64
import json
import tempfile
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path

import jwt
from cryptography.hazmat.primitives.asymmetric import rsa
from fastapi.testclient import TestClient

from market_predictor.api import create_app
from market_predictor.api_security import (
    API_SCOPES,
    ApiAuthenticator,
    ApiSecurityConfig,
    ApiSecurityError,
    PrincipalRateLimiter,
)
from market_predictor.telemetry import RuntimeTelemetry
from tests.test_api import StubPredictionService, StubReplayService


class ApiAuthenticatorTests(unittest.TestCase):
    def test_development_token_authenticates_and_enforces_scope(self) -> None:
        token = "development-token-value-1234567890"
        authenticator = ApiAuthenticator(
            ApiSecurityConfig(
                mode="development",
                development_token=token,
                development_scopes=frozenset({"predictions.read"}),
            )
        )

        principal = authenticator.authenticate(
            f"Bearer {token}",
            required_scope="predictions.read",
        )

        self.assertEqual(principal.principal_id, "development-client")
        with self.assertRaises(ApiSecurityError) as missing:
            authenticator.authenticate(None, required_scope="predictions.read")
        self.assertEqual(missing.exception.status_code, 401)
        with self.assertRaises(ApiSecurityError) as forbidden:
            authenticator.authenticate(
                f"Bearer {token}",
                required_scope="metrics.read",
            )
        self.assertEqual(forbidden.exception.status_code, 403)

    def test_entra_jwt_validates_signature_audience_issuer_and_roles(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            private_key, jwks_path = _write_test_jwks(root)
            issuer = "https://login.microsoftonline.com/test-tenant/v2.0"
            audience = "market-predictor-api"
            authenticator = ApiAuthenticator(
                ApiSecurityConfig(
                    mode="entra",
                    issuer=issuer,
                    audience=audience,
                    jwks_path=jwks_path,
                )
            )
            valid = _access_token(
                private_key,
                issuer=issuer,
                audience=audience,
                roles=["predictions.read"],
            )

            principal = authenticator.authenticate(
                f"Bearer {valid}",
                required_scope="predictions.read",
            )

            self.assertEqual(principal.principal_id, "test-tenant:principal-object")
            self.assertEqual(principal.actor_id, "trading-flow-client")

            wrong_audience = _access_token(
                private_key,
                issuer=issuer,
                audience="different-api",
                roles=["predictions.read"],
            )
            with self.assertRaises(ApiSecurityError) as invalid:
                authenticator.authenticate(
                    f"Bearer {wrong_audience}",
                    required_scope="predictions.read",
                )
            self.assertEqual(invalid.exception.status_code, 401)

            wrong_scope = _access_token(
                private_key,
                issuer=issuer,
                audience=audience,
                roles=["metrics.read"],
            )
            with self.assertRaises(ApiSecurityError) as forbidden:
                authenticator.authenticate(
                    f"Bearer {wrong_scope}",
                    required_scope="predictions.read",
                )
            self.assertEqual(forbidden.exception.status_code, 403)


class PrincipalRateLimiterTests(unittest.TestCase):
    def test_token_bucket_refills_and_bounds_principal_state(self) -> None:
        now = [100.0]
        rates = {scope: 1 for scope in API_SCOPES}
        limiter = PrincipalRateLimiter(
            requests_per_minute=rates,
            maximum_principals=2,
            clock=lambda: now[0],
        )
        limiter.acquire("principal-a", "predictions.read")
        with self.assertRaises(ApiSecurityError) as limited:
            limiter.acquire("principal-a", "predictions.read")
        self.assertEqual(limited.exception.status_code, 429)

        now[0] += 60.0
        limiter.acquire("principal-a", "predictions.read")
        for principal in ["principal-b", "principal-c", "principal-d"]:
            for scope in API_SCOPES:
                limiter.acquire(principal, scope)
        self.assertLessEqual(
            limiter.tracked_bucket_count(),
            2 * len(API_SCOPES),
        )


class SecuredPredictionApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.token = "development-token-value-1234567890"

    def test_prediction_auth_scope_body_limit_and_rate_limit(self) -> None:
        config = ApiSecurityConfig(
            mode="development",
            development_token=self.token,
            development_scopes=frozenset({"predictions.read"}),
        )
        limiter = PrincipalRateLimiter(
            requests_per_minute={
                **{scope: 10 for scope in API_SCOPES},
                "predictions.read": 1,
            }
        )
        client = TestClient(
            create_app(
                StubPredictionService(),  # type: ignore[arg-type]
                security_config=config,
                rate_limiter=limiter,
                maximum_body_bytes=1_024,
            )
        )

        missing = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"]},
        )
        valid = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"]},
            headers=self._authorization(),
        )
        limited = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["MSFT"]},
            headers=self._authorization(),
        )
        oversized = client.post(
            "/v1/predictions/swing",
            content=b"{" + b"x" * 2_000 + b"}",
            headers={
                **self._authorization(),
                "content-type": "application/json",
            },
        )

        self.assertEqual(missing.status_code, 401)
        self.assertEqual(missing.headers["www-authenticate"], "Bearer")
        self.assertEqual(valid.status_code, 200)
        self.assertEqual(limited.status_code, 429)
        self.assertIn("retry-after", limited.headers)
        self.assertEqual(oversized.status_code, 413)

    def test_scoped_operations_minimal_readiness_and_disabled_replay(self) -> None:
        config = ApiSecurityConfig(
            mode="development",
            development_token=self.token,
            development_scopes=frozenset({"operations.read"}),
        )
        client = TestClient(
            create_app(
                StubPredictionService(),  # type: ignore[arg-type]
                replay_service=StubReplayService(),  # type: ignore[arg-type]
                security_config=config,
                replay_enabled=False,
            )
        )

        ready = client.get("/v1/health/ready")
        operations_missing = client.get("/v1/operations/health")
        operations = client.get(
            "/v1/operations/health",
            headers=self._authorization(),
        )
        metrics = client.get("/v1/metrics", headers=self._authorization())
        replay = client.post(
            "/v1/replays/investment",
            json={"snapshot_id": "a" * 64, "ticker": "MSFT"},
            headers=self._authorization(),
        )

        self.assertEqual(ready.status_code, 200)
        self.assertEqual(set(ready.json()), {"status", "checked_at_utc"})
        self.assertEqual(operations_missing.status_code, 401)
        self.assertEqual(operations.status_code, 200)
        self.assertIn("components", operations.json())
        self.assertEqual(metrics.status_code, 403)
        self.assertEqual(replay.status_code, 404)

    def test_ticker_contract_rejects_noncanonical_and_oversized_lists(self) -> None:
        config = ApiSecurityConfig(
            mode="development",
            development_token=self.token,
            development_scopes=frozenset({"predictions.read"}),
        )
        client = TestClient(
            create_app(
                StubPredictionService(),  # type: ignore[arg-type]
                security_config=config,
            )
        )

        invalid = client.post(
            "/v1/predictions/swing",
            json={"tickers": ["../../MSFT"]},
            headers=self._authorization(),
        )
        too_many = client.post(
            "/v1/predictions/swing",
            json={"tickers": [f"A{index:03d}" for index in range(101)]},
            headers=self._authorization(),
        )

        self.assertEqual(invalid.status_code, 422)
        self.assertEqual(too_many.status_code, 422)

    def test_structured_audit_contains_identity_but_never_bearer_token(self) -> None:
        config = ApiSecurityConfig(
            mode="development",
            development_token=self.token,
            development_scopes=frozenset({"predictions.read"}),
        )
        telemetry = RuntimeTelemetry()
        client = TestClient(
            create_app(
                StubPredictionService(),  # type: ignore[arg-type]
                telemetry=telemetry,
                security_config=config,
            )
        )

        with self.assertLogs("market_predictor.telemetry", level="INFO") as captured:
            response = client.post(
                "/v1/predictions/swing",
                json={"tickers": ["MSFT"]},
                headers={
                    **self._authorization(),
                    "x-correlation-id": "trading-flow:security-test",
                },
            )

        logs = "\n".join(captured.output)
        self.assertEqual(response.status_code, 200)
        self.assertIn('"principal_id": "development-client"', logs)
        self.assertIn('"correlation_id": "trading-flow:security-test"', logs)
        self.assertNotIn(self.token, logs)

    def _authorization(self) -> dict[str, str]:
        return {"authorization": f"Bearer {self.token}"}


def _write_test_jwks(root: Path) -> tuple[rsa.RSAPrivateKey, Path]:
    private_key = rsa.generate_private_key(public_exponent=65_537, key_size=2_048)
    numbers = private_key.public_key().public_numbers()
    path = root / "jwks.json"
    path.write_text(
        json.dumps(
            {
                "keys": [
                    {
                        "kty": "RSA",
                        "use": "sig",
                        "alg": "RS256",
                        "kid": "test-key",
                        "n": _base64url_uint(numbers.n),
                        "e": _base64url_uint(numbers.e),
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    return private_key, path


def _access_token(
    private_key: rsa.RSAPrivateKey,
    *,
    issuer: str,
    audience: str,
    roles: list[str],
) -> str:
    now = datetime.now(UTC)
    return jwt.encode(
        {
            "iss": issuer,
            "aud": audience,
            "iat": now,
            "nbf": now - timedelta(seconds=1),
            "exp": now + timedelta(minutes=5),
            "tid": "test-tenant",
            "oid": "principal-object",
            "sub": "principal-subject",
            "azp": "trading-flow-client",
            "roles": roles,
        },
        private_key,
        algorithm="RS256",
        headers={"kid": "test-key"},
    )


def _base64url_uint(value: int) -> str:
    raw = value.to_bytes((value.bit_length() + 7) // 8, "big")
    return base64.urlsafe_b64encode(raw).rstrip(b"=").decode("ascii")


if __name__ == "__main__":
    unittest.main()
