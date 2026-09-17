"""Portable raw news receipts, not normalized events or model admission evidence."""

from __future__ import annotations

import hashlib
import math
import re
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal, Self

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from market_predictor.core.json_integrity import parse_strict_json_object

MAX_MANIFEST_BYTES = 16_384
MAX_PAYLOAD_BYTES = 8_388_608
_UTC_CLOCK = re.compile(r"^[0-9]{4}-[0-9]{2}-[0-9]{2}T[0-9]{2}:[0-9]{2}:[0-9]{2}\.[0-9]{6}Z$")
_PAGE_TOKEN = re.compile(r"[\x21-\x7e]{1,2048}")


class _ReceiptContract(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True, strict=True)


def _utc_clock(value: object) -> datetime:
    if isinstance(value, str) and _UTC_CLOCK.fullmatch(value):
        return datetime.strptime(value, "%Y-%m-%dT%H:%M:%S.%fZ").replace(tzinfo=UTC)
    if isinstance(value, datetime) and value.utcoffset() == timedelta(0):
        return value.astimezone(UTC)
    raise ValueError("receipt clocks require UTC with six fractional digits and Z")


class NewsPageRequest(_ReceiptContract):
    symbol: str = Field(pattern=r"^[A-Z][A-Z0-9.-]{0,14}$")
    start_utc: datetime
    end_utc: datetime
    page_token: str | None = Field(max_length=2048)
    include_content: bool
    limit: int = Field(ge=1, le=50)

    @field_validator("start_utc", "end_utc", mode="before")
    @classmethod
    def clocks(cls, value: object) -> datetime:
        return _utc_clock(value)

    @model_validator(mode="after")
    def bounds(self) -> Self:
        if self.start_utc >= self.end_utc:
            raise ValueError("news request start must precede end")
        if self.page_token is not None and _PAGE_TOKEN.fullmatch(self.page_token) is None:
            raise ValueError("page token must be null or printable ASCII without spaces")
        return self


class NewsHttpReceipt(_ReceiptContract):
    schema_version: Literal["alpaca.news_http_receipt.v1"]
    producer: Literal["market_predictor", "trading_flow"]
    producer_revision: str = Field(min_length=40, max_length=40, pattern=r"^[0-9a-f]{40}$")
    endpoint: Literal["alpaca.news"]
    transport: Literal["http"]
    status_code: Literal[200]
    byte_representation: Literal["decoded_http_body_utf8"]
    availability_basis: Literal["observed_receipt"]
    request: NewsPageRequest
    received_at_utc: datetime
    payload_sha256: str = Field(pattern=r"^[0-9a-f]{64}$")
    payload_bytes: int = Field(ge=1, le=MAX_PAYLOAD_BYTES)

    @field_validator("status_code", mode="before")
    @classmethod
    def status_is_integer(cls, value: object) -> object:
        if type(value) is not int:
            raise ValueError("HTTP status must be an integer")
        return value

    @field_validator("received_at_utc", mode="before")
    @classmethod
    def clock(cls, value: object) -> datetime:
        return _utc_clock(value)

    @model_validator(mode="after")
    def receipt_follows_request(self) -> Self:
        if self.received_at_utc < self.request.end_utc:
            raise ValueError("receipt precedes the requested coverage end")
        return self


@dataclass(frozen=True, slots=True)
class ValidatedNewsReceipt:
    manifest: NewsHttpReceipt
    manifest_bytes: bytes
    payload_bytes: bytes

    @property
    def receipt_id(self) -> str:
        return hashlib.sha256(self.manifest_bytes).hexdigest()

    def observable_at(self, cutoff: datetime) -> bool:
        return self.manifest.received_at_utc <= _utc_clock(cutoff)


def validate_news_receipt(manifest_bytes: bytes, payload_bytes: bytes) -> ValidatedNewsReceipt:
    if not 0 < len(manifest_bytes) <= MAX_MANIFEST_BYTES:
        raise ValueError("news receipt manifest exceeds its byte limit")
    if not 0 < len(payload_bytes) <= MAX_PAYLOAD_BYTES:
        raise ValueError("news receipt payload exceeds its byte limit")
    receipt = NewsHttpReceipt.model_validate(_parse_object(manifest_bytes, label="news receipt"))
    if len(payload_bytes) != receipt.payload_bytes or hashlib.sha256(payload_bytes).hexdigest() != receipt.payload_sha256:
        raise ValueError("news receipt payload integrity mismatch")
    payload = _parse_object(payload_bytes, label="news HTTP body")
    news = payload.get("news")
    if not isinstance(news, list) or len(news) > receipt.request.limit or any(not isinstance(row, dict) for row in news):
        raise ValueError("news HTTP body must contain a bounded news array")
    token = payload.get("next_page_token")
    if token is not None and (not isinstance(token, str) or _PAGE_TOKEN.fullmatch(token) is None):
        raise ValueError("news HTTP body has an invalid next-page token")
    return ValidatedNewsReceipt(receipt, manifest_bytes, payload_bytes)


def _parse_object(raw: bytes, *, label: str) -> dict[str, object]:
    try:
        value = parse_strict_json_object(raw, label=label)
        _require_bounded_depth(value, 1)
        return value
    except RecursionError as exc:
        raise ValueError("news JSON exceeds its nesting limit") from exc


def _require_bounded_depth(value: object, depth: int) -> None:
    if isinstance(value, (dict, list)):
        if depth > 64:
            raise ValueError("news JSON exceeds its nesting limit")
        for child in value.values() if isinstance(value, dict) else value:
            _require_bounded_depth(child, depth + 1)
        if isinstance(value, dict):
            for key in value:
                key.encode("utf-8", errors="strict")
    elif isinstance(value, str):
        value.encode("utf-8", errors="strict")
    elif isinstance(value, int):
        try:
            finite = math.isfinite(float(value))
        except OverflowError as exc:
            raise ValueError("news JSON number exceeds the finite binary64 range") from exc
        if not finite:
            raise ValueError("news JSON number exceeds the finite binary64 range")


def read_news_receipt(manifest_path: Path, payload_path: Path, *, expected_receipt_sha256: str) -> ValidatedNewsReceipt:
    # Bounded reads also protect against files growing between stat and read.
    with manifest_path.open("rb") as handle:
        manifest = handle.read(MAX_MANIFEST_BYTES + 1)
    if (
        re.fullmatch(r"[0-9a-f]{64}", expected_receipt_sha256) is None
        or hashlib.sha256(manifest).hexdigest() != expected_receipt_sha256
    ):
        raise ValueError("news receipt manifest does not match its independent integrity pin")
    with payload_path.open("rb") as handle:
        payload = handle.read(MAX_PAYLOAD_BYTES + 1)
    return validate_news_receipt(manifest, payload)
