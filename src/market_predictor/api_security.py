from __future__ import annotations

import hmac
import json
import re
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import jwt
from pydantic import BaseModel, ConfigDict, Field, SecretStr, model_validator

API_SCOPES = frozenset(
    {
        "predictions.read",
        "operations.read",
        "metrics.read",
        "replay.execute",
    }
)
_BEARER_PATTERN = re.compile(r"^Bearer[ \t]+([^ \t]+)$", re.IGNORECASE)


class ApiSecurityConfig(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    mode: Literal["disabled", "development", "entra"] = "disabled"
    issuer: str | None = None
    audience: str | None = None
    jwks_path: Path | None = None
    development_token: SecretStr | None = None
    development_principal_id: str = "development-client"
    development_scopes: frozenset[str] = API_SCOPES
    clock_skew_seconds: int = Field(default=60, ge=0, le=300)
    maximum_token_bytes: int = Field(default=16_384, ge=1_024, le=65_536)

    @model_validator(mode="after")
    def validate_mode_requirements(self) -> ApiSecurityConfig:
        unknown = set(self.development_scopes).difference(API_SCOPES)
        if unknown:
            raise ValueError(f"unknown API scopes: {sorted(unknown)}")
        if self.mode == "development":
            token = (
                self.development_token.get_secret_value()
                if self.development_token is not None
                else ""
            )
            if len(token) < 32:
                raise ValueError("development bearer token must contain at least 32 characters")
        if self.mode == "entra":
            if not self.issuer or not self.audience or self.jwks_path is None:
                raise ValueError("Entra auth requires issuer, audience, and a local JWKS path")
            if not self.issuer.startswith("https://"):
                raise ValueError("Entra issuer must use HTTPS")
        return self


@dataclass(frozen=True, slots=True)
class ApiPrincipal:
    principal_id: str
    actor_id: str | None
    tenant_id: str | None
    scopes: frozenset[str]
    authentication_method: str


class ApiSecurityError(Exception):
    def __init__(
        self,
        *,
        status_code: int,
        code: str,
        message: str,
        retry_after_seconds: int | None = None,
    ) -> None:
        super().__init__(message)
        self.status_code = status_code
        self.code = code
        self.message = message
        self.retry_after_seconds = retry_after_seconds


class ApiAuthenticator:
    def __init__(self, config: ApiSecurityConfig) -> None:
        self.config = config
        self._jwks_lock = threading.Lock()
        self._jwks_mtime_ns: int | None = None
        self._keys: dict[str, Any] = {}

    @classmethod
    def disabled_for_tests(cls) -> ApiAuthenticator:
        return cls(ApiSecurityConfig(mode="disabled"))

    def authenticate(
        self,
        authorization: str | None,
        *,
        required_scope: str,
    ) -> ApiPrincipal:
        if required_scope not in API_SCOPES:
            raise ValueError("route requires an unknown API scope")
        if self.config.mode == "disabled":
            return ApiPrincipal(
                principal_id="test-client",
                actor_id=None,
                tenant_id=None,
                scopes=API_SCOPES,
                authentication_method="disabled_test_only",
            )
        token = _bearer_token(authorization, self.config.maximum_token_bytes)
        principal = (
            self._development_principal(token)
            if self.config.mode == "development"
            else self._entra_principal(token)
        )
        if required_scope not in principal.scopes:
            raise ApiSecurityError(
                status_code=403,
                code="insufficient_scope",
                message="The caller is not authorized for this operation.",
            )
        return principal

    def _development_principal(self, token: str) -> ApiPrincipal:
        expected = self.config.development_token
        expected_value = expected.get_secret_value() if expected is not None else ""
        if not hmac.compare_digest(token, expected_value):
            raise _authentication_error()
        return ApiPrincipal(
            principal_id=self.config.development_principal_id,
            actor_id=self.config.development_principal_id,
            tenant_id=None,
            scopes=self.config.development_scopes,
            authentication_method="development_static_token",
        )

    def _entra_principal(self, token: str) -> ApiPrincipal:
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise _authentication_error()
            kid = str(header.get("kid") or "")
            key = self._signing_key(kid)
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self.config.audience,
                issuer=self.config.issuer,
                leeway=self.config.clock_skew_seconds,
                options={"require": ["exp", "iat", "iss", "aud"]},
            )
        except ApiSecurityError:
            raise
        except jwt.PyJWTError as exc:
            raise _authentication_error() from exc
        tenant_id = _bounded_claim(claims.get("tid"), "tenant")
        object_id = _bounded_claim(claims.get("oid"), "object")
        subject = _bounded_claim(claims.get("sub"), "subject")
        actor_id = _bounded_claim(
            claims.get("azp") or claims.get("appid"),
            "actor",
        )
        identity = object_id or subject
        if identity is None:
            raise _authentication_error()
        principal_id = f"{tenant_id}:{identity}" if tenant_id else identity
        scopes = _token_scopes(claims)
        return ApiPrincipal(
            principal_id=principal_id,
            actor_id=actor_id,
            tenant_id=tenant_id,
            scopes=scopes,
            authentication_method="entra_jwt",
        )

    def _signing_key(self, kid: str) -> Any:
        if not kid or len(kid) > 256:
            raise _authentication_error()
        path = self.config.jwks_path
        if path is None:
            raise _authentication_error()
        try:
            mtime_ns = path.stat().st_mtime_ns
        except OSError as exc:
            raise _authentication_error() from exc
        with self._jwks_lock:
            if self._jwks_mtime_ns != mtime_ns:
                self._keys = _load_jwks(path)
                self._jwks_mtime_ns = mtime_ns
            key = self._keys.get(kid)
        if key is None:
            raise _authentication_error()
        return key


@dataclass(slots=True)
class _Bucket:
    tokens: float
    updated_at: float


class PrincipalRateLimiter:
    def __init__(
        self,
        *,
        requests_per_minute: dict[str, int],
        maximum_principals: int = 10_000,
        clock: Any = time.monotonic,
    ) -> None:
        if maximum_principals < 1:
            raise ValueError("maximum rate-limit principals must be positive")
        if set(requests_per_minute) != API_SCOPES:
            raise ValueError("rate limits must be configured for every API scope")
        if any(value < 1 for value in requests_per_minute.values()):
            raise ValueError("rate limits must be positive")
        self.requests_per_minute = dict(requests_per_minute)
        self.maximum_principals = maximum_principals
        self.clock = clock
        self._lock = threading.Lock()
        self._buckets: OrderedDict[tuple[str, str], _Bucket] = OrderedDict()

    def acquire(self, principal_id: str, scope: str) -> None:
        capacity = self.requests_per_minute[scope]
        refill_per_second = capacity / 60.0
        key = (principal_id, scope)
        now = float(self.clock())
        with self._lock:
            bucket = self._buckets.pop(key, None)
            if bucket is None:
                bucket = _Bucket(tokens=float(capacity), updated_at=now)
            else:
                elapsed = max(0.0, now - bucket.updated_at)
                bucket.tokens = min(
                    float(capacity),
                    bucket.tokens + elapsed * refill_per_second,
                )
                bucket.updated_at = now
            if bucket.tokens < 1.0:
                self._buckets[key] = bucket
                retry_after = max(1, int((1.0 - bucket.tokens) / refill_per_second) + 1)
                raise ApiSecurityError(
                    status_code=429,
                    code="rate_limit_exceeded",
                    message="The caller has exceeded the API rate limit.",
                    retry_after_seconds=retry_after,
                )
            bucket.tokens -= 1.0
            self._buckets[key] = bucket
            while len(self._buckets) > self.maximum_principals * len(API_SCOPES):
                self._buckets.popitem(last=False)

    def tracked_bucket_count(self) -> int:
        with self._lock:
            return len(self._buckets)


def _bearer_token(authorization: str | None, maximum_bytes: int) -> str:
    value = authorization or ""
    if len(value.encode("utf-8")) > maximum_bytes:
        raise _authentication_error()
    match = _BEARER_PATTERN.fullmatch(value)
    if match is None:
        raise _authentication_error()
    return match.group(1)


def _token_scopes(claims: dict[str, Any]) -> frozenset[str]:
    values: set[str] = set()
    delegated = claims.get("scp")
    if isinstance(delegated, str):
        values.update(delegated.split())
    roles = claims.get("roles")
    if isinstance(roles, list):
        values.update(str(role) for role in roles)
    elif isinstance(roles, str):
        values.add(roles)
    return frozenset(value for value in values if value in API_SCOPES)


def _bounded_claim(value: object, name: str) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str) or not 1 <= len(value) <= 256:
        raise ApiSecurityError(
            status_code=401,
            code="invalid_access_token",
            message=f"The access token {name} identity is invalid.",
        )
    return value


def _load_jwks(path: Path) -> dict[str, Any]:
    try:
        loaded = json.loads(path.read_text(encoding="utf-8"))
        keys = loaded.get("keys") if isinstance(loaded, dict) else None
        if not isinstance(keys, list):
            raise ValueError("JWKS keys are missing")
        parsed: dict[str, Any] = {}
        for raw in keys:
            if not isinstance(raw, dict):
                continue
            kid = str(raw.get("kid") or "")
            if not kid or kid in parsed:
                raise ValueError("JWKS key identity is invalid")
            pyjwk = jwt.PyJWK.from_dict(raw, algorithm="RS256")
            parsed[kid] = pyjwk.key
        if not parsed:
            raise ValueError("JWKS contains no usable signing keys")
        return parsed
    except (OSError, ValueError, jwt.PyJWTError) as exc:
        raise _authentication_error() from exc


def _authentication_error() -> ApiSecurityError:
    return ApiSecurityError(
        status_code=401,
        code="invalid_access_token",
        message="A valid bearer access token is required.",
    )
