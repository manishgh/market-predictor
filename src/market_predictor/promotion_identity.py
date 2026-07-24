from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

from market_predictor.jwt_verification import (
    JwtVerificationError,
    LocalJwksVerifier,
)
from market_predictor.v3.errors import DataReadinessError

BUILD_ROLE = "promotion.build"
APPROVER_ROLE = "promotion.approve"
PROMOTION_PRINCIPAL_SCHEMA = "market_predictor.authenticated_principal.v1"
DEFAULT_BUILD_TOKEN_ENV = "MARKET_PREDICTOR_PROMOTION_BUILD_TOKEN"
DEFAULT_APPROVER_TOKEN_ENV = "MARKET_PREDICTOR_PROMOTION_APPROVER_TOKEN"
_ENVIRONMENT_NAME = re.compile(r"^[A-Z][A-Z0-9_]{0,127}$")


class PromotionIdentityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    issuer: str
    audience: str
    jwks_path: Path
    clock_skew_seconds: int = Field(default=60, ge=0, le=300)
    maximum_token_bytes: int = Field(default=16_384, ge=1_024, le=65_536)

    @model_validator(mode="after")
    def validate_trust_contract(self) -> PromotionIdentityConfig:
        if not self.issuer.startswith("https://"):
            raise ValueError("promotion identity issuer must use HTTPS")
        if not self.audience.strip():
            raise ValueError("promotion identity audience is required")
        return self


@dataclass(frozen=True, slots=True)
class PromotionTokens:
    build: SecretStr
    approver: SecretStr


class PromotionIdentityAuthenticator:
    def __init__(self, config: PromotionIdentityConfig) -> None:
        self.config = config
        self._verifier = LocalJwksVerifier(
            issuer=config.issuer,
            audience=config.audience,
            jwks_path=config.jwks_path,
            clock_skew_seconds=config.clock_skew_seconds,
            maximum_token_bytes=config.maximum_token_bytes,
        )

    def authenticate_pair(
        self,
        tokens: PromotionTokens,
        *,
        authenticated_at: datetime | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        timestamp = _utc(authenticated_at or datetime.now(UTC))
        build = self._authenticate(
            tokens.build,
            required_role=BUILD_ROLE,
            authenticated_at=timestamp,
        )
        approver = self._authenticate(
            tokens.approver,
            required_role=APPROVER_ROLE,
            authenticated_at=timestamp,
        )
        if build["principal_id"] == approver["principal_id"]:
            raise DataReadinessError(
                "promotion build and approver principals must be distinct"
            )
        return build, approver

    def _authenticate(
        self,
        token: SecretStr,
        *,
        required_role: str,
        authenticated_at: datetime,
    ) -> dict[str, Any]:
        raw_token = token.get_secret_value()
        try:
            claims, kid, jwks_sha256 = self._verifier.verify(
                raw_token,
                required_claims=("exp", "iat", "iss", "aud", "jti"),
            )
        except JwtVerificationError as exc:
            raise DataReadinessError(
                "promotion identity token authentication failed"
            ) from exc
        roles = _roles(claims)
        if required_role not in roles:
            raise DataReadinessError(
                f"promotion identity is missing required role: {required_role}"
            )
        tenant_id = _bounded_optional_claim(claims.get("tid"), "tenant")
        object_id = _bounded_optional_claim(claims.get("oid"), "object")
        subject = _bounded_optional_claim(claims.get("sub"), "subject")
        actor_id = _bounded_optional_claim(
            claims.get("azp") or claims.get("appid"),
            "actor",
        )
        identity = object_id or subject
        if identity is None:
            raise DataReadinessError(
                "promotion identity token has no principal identity"
            )
        principal_id = f"{tenant_id}:{identity}" if tenant_id else identity
        token_id = _bounded_required_claim(claims.get("jti"), "token")
        issued_at = _numeric_date(claims.get("iat"), "issued-at")
        expires_at = _numeric_date(claims.get("exp"), "expiry")
        if expires_at <= issued_at:
            raise DataReadinessError(
                "promotion identity token lifetime is invalid"
            )
        evidence = {
            "schema": PROMOTION_PRINCIPAL_SCHEMA,
            "principal_id": principal_id,
            "actor_id": actor_id,
            "tenant_id": tenant_id,
            "issuer": self.config.issuer,
            "audience": self.config.audience,
            "required_role": required_role,
            "token_id": token_id,
            "signing_key_id": kid,
            "jwks_sha256": jwks_sha256,
            "issued_at_utc": datetime.fromtimestamp(
                issued_at,
                tz=UTC,
            ).isoformat(),
            "expires_at_utc": datetime.fromtimestamp(
                expires_at,
                tz=UTC,
            ).isoformat(),
            "authenticated_at_utc": authenticated_at.isoformat(),
            "claims_sha256": _json_sha256(claims),
            "token_sha256": hashlib.sha256(
                raw_token.encode("utf-8")
            ).hexdigest(),
            "authentication_method": "oidc_rs256_jwks",
        }
        validate_promotion_principal(
            evidence,
            expected_role=required_role,
        )
        return evidence


def promotion_tokens_from_environment(
    *,
    build_token_env: str = DEFAULT_BUILD_TOKEN_ENV,
    approver_token_env: str = DEFAULT_APPROVER_TOKEN_ENV,
) -> PromotionTokens:
    for name in (build_token_env, approver_token_env):
        if _ENVIRONMENT_NAME.fullmatch(name) is None:
            raise DataReadinessError(
                "promotion token environment variable name is invalid"
            )
    if build_token_env == approver_token_env:
        raise DataReadinessError(
            "build and approver tokens require separate environment variables"
        )
    build = os.environ.get(build_token_env, "")
    approver = os.environ.get(approver_token_env, "")
    if not build or not approver:
        raise DataReadinessError(
            "promotion identity token environment variables are not populated"
        )
    return PromotionTokens(
        build=SecretStr(build),
        approver=SecretStr(approver),
    )


def validate_promotion_principal(
    value: object,
    *,
    expected_role: str,
) -> dict[str, Any]:
    expected_fields = {
        "schema",
        "principal_id",
        "actor_id",
        "tenant_id",
        "issuer",
        "audience",
        "required_role",
        "token_id",
        "signing_key_id",
        "jwks_sha256",
        "issued_at_utc",
        "expires_at_utc",
        "authenticated_at_utc",
        "claims_sha256",
        "token_sha256",
        "authentication_method",
    }
    if not isinstance(value, dict) or set(value) != expected_fields:
        raise DataReadinessError(
            "promotion authenticated principal fields are invalid"
        )
    principal = {str(key): item for key, item in value.items()}
    if (
        principal.get("schema") != PROMOTION_PRINCIPAL_SCHEMA
        or principal.get("required_role") != expected_role
        or principal.get("authentication_method") != "oidc_rs256_jwks"
    ):
        raise DataReadinessError(
            "promotion authenticated principal contract is invalid"
        )
    for field in (
        "principal_id",
        "issuer",
        "audience",
        "token_id",
        "signing_key_id",
    ):
        _bounded_required_claim(principal.get(field), field)
    if not str(principal["issuer"]).startswith("https://"):
        raise DataReadinessError(
            "promotion authenticated principal issuer is invalid"
        )
    for field in ("claims_sha256", "token_sha256", "jwks_sha256"):
        digest = str(principal.get(field) or "")
        if len(digest) != 64:
            raise DataReadinessError(
                f"promotion authenticated principal {field} is invalid"
            )
        try:
            int(digest, 16)
        except ValueError as exc:
            raise DataReadinessError(
                f"promotion authenticated principal {field} is invalid"
            ) from exc
    issued_at = _utc(datetime.fromisoformat(str(principal["issued_at_utc"])))
    expires_at = _utc(datetime.fromisoformat(str(principal["expires_at_utc"])))
    authenticated_at = _utc(
        datetime.fromisoformat(str(principal["authenticated_at_utc"]))
    )
    if not issued_at <= authenticated_at < expires_at:
        raise DataReadinessError(
            "promotion authenticated principal timing is invalid"
        )
    for optional_field in ("actor_id", "tenant_id"):
        value = principal.get(optional_field)
        if value is not None:
            _bounded_required_claim(value, optional_field)
    return principal


def _roles(claims: dict[str, Any]) -> frozenset[str]:
    raw = claims.get("roles")
    if isinstance(raw, list):
        values = raw
    elif isinstance(raw, str):
        values = [raw]
    else:
        values = []
    return frozenset(
        role
        for value in values
        if isinstance(value, str)
        for role in [value.strip()]
        if role
    )


def _bounded_optional_claim(value: object, name: str) -> str | None:
    if value is None:
        return None
    return _bounded_required_claim(value, name)


def _bounded_required_claim(value: object, name: str) -> str:
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise DataReadinessError(
            f"promotion identity {name} claim is invalid"
        )
    return value


def _numeric_date(value: object, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise DataReadinessError(
            f"promotion identity {name} claim is invalid"
        )
    return int(value)


def _json_sha256(value: object) -> str:
    payload = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise DataReadinessError(
            "promotion identity timestamp must be timezone-aware"
        )
    return value.astimezone(UTC)
