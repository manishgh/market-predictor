from __future__ import annotations

import hashlib
import json
import threading
from pathlib import Path
from typing import Any

import jwt


class JwtVerificationError(ValueError):
    """Raised when a JWT cannot be verified against the pinned trust contract."""


class LocalJwksVerifier:
    """Verify RS256 JWTs against a reloadable, deployment-owned JWKS file."""

    def __init__(
        self,
        *,
        issuer: str,
        audience: str,
        jwks_path: Path,
        clock_skew_seconds: int,
        maximum_token_bytes: int,
    ) -> None:
        self.issuer = issuer
        self.audience = audience
        self.jwks_path = jwks_path
        self.clock_skew_seconds = clock_skew_seconds
        self.maximum_token_bytes = maximum_token_bytes
        self._lock = threading.Lock()
        self._mtime_ns: int | None = None
        self._keys: dict[str, Any] = {}
        self._jwks_sha256: str | None = None

    def verify(
        self,
        token: str,
        *,
        required_claims: tuple[str, ...] = ("exp", "iat", "iss", "aud"),
    ) -> tuple[dict[str, Any], str, str]:
        if (
            not token
            or len(token.encode("utf-8")) > self.maximum_token_bytes
        ):
            raise JwtVerificationError("JWT size is invalid")
        try:
            header = jwt.get_unverified_header(token)
            if header.get("alg") != "RS256":
                raise JwtVerificationError("JWT algorithm is invalid")
            kid = str(header.get("kid") or "")
            key, jwks_sha256 = self._signing_key(kid)
            claims = jwt.decode(
                token,
                key=key,
                algorithms=["RS256"],
                audience=self.audience,
                issuer=self.issuer,
                leeway=self.clock_skew_seconds,
                options={"require": list(required_claims)},
            )
        except JwtVerificationError:
            raise
        except jwt.PyJWTError as exc:
            raise JwtVerificationError("JWT validation failed") from exc
        if not isinstance(claims, dict):
            raise JwtVerificationError("JWT claims are invalid")
        return (
            {str(key): value for key, value in claims.items()},
            kid,
            jwks_sha256,
        )

    def _signing_key(self, kid: str) -> tuple[Any, str]:
        if not kid or len(kid) > 256:
            raise JwtVerificationError("JWT signing key identity is invalid")
        try:
            mtime_ns = self.jwks_path.stat().st_mtime_ns
        except OSError as exc:
            raise JwtVerificationError("JWKS is unavailable") from exc
        with self._lock:
            if self._mtime_ns != mtime_ns:
                self._keys, self._jwks_sha256 = _load_jwks(
                    self.jwks_path
                )
                self._mtime_ns = mtime_ns
            key = self._keys.get(kid)
            jwks_sha256 = self._jwks_sha256
        if key is None or jwks_sha256 is None:
            raise JwtVerificationError("JWT signing key is not trusted")
        return key, jwks_sha256


def _load_jwks(path: Path) -> tuple[dict[str, Any], str]:
    try:
        raw_bytes = path.read_bytes()
        loaded = json.loads(raw_bytes)
        keys = loaded.get("keys") if isinstance(loaded, dict) else None
        if not isinstance(keys, list):
            raise ValueError("JWKS keys are missing")
        parsed: dict[str, Any] = {}
        for raw_key in keys:
            if not isinstance(raw_key, dict):
                continue
            kid = str(raw_key.get("kid") or "")
            if not kid or kid in parsed:
                raise ValueError("JWKS key identity is invalid")
            parsed[kid] = jwt.PyJWK.from_dict(
                raw_key,
                algorithm="RS256",
            ).key
        if not parsed:
            raise ValueError("JWKS contains no usable signing keys")
        return parsed, hashlib.sha256(raw_bytes).hexdigest()
    except (OSError, ValueError, jwt.PyJWTError) as exc:
        raise JwtVerificationError("JWKS is invalid") from exc
