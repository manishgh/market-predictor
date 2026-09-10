"""Evidence-pinned provider query intervals, never fabricated index membership."""
from __future__ import annotations

from pathlib import Path
from typing import Annotated, Any, Literal, Self

import pandas as pd
from pydantic import AwareDatetime, BaseModel, ConfigDict, Field, model_validator

from market_predictor.canonical.store import file_sha256
from market_predictor.core.errors import DataReadinessError
from market_predictor.evidence.io import resolve_inside_authority


class _Contract(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, frozen=True)


class SourcePin(_Contract):
    path: Annotated[str, Field(min_length=1)]
    sha256: Annotated[str, Field(pattern=r"^[0-9a-f]{64}$")]


class IssuerQueryInterval(_Contract):
    ticker: Annotated[str, Field(pattern=r"^[A-Z0-9][A-Z0-9.-]{0,14}$")]
    security_id: Annotated[str, Field(min_length=1)]
    start_utc: AwareDatetime
    end_exclusive_utc: AwareDatetime
    interpretation: Annotated[str, Field(min_length=1)]
    source_files: Annotated[tuple[SourcePin, ...], Field(min_length=1)]

    @model_validator(mode="after")
    def ordered(self) -> Self:
        if self.start_utc >= self.end_exclusive_utc:
            raise ValueError("issuer query interval is empty or reversed")
        return self


class NewsQueryScope(_Contract):
    schema_version: Literal["market_predictor.issuer_news_query_scope"] = Field(alias="schema")
    purpose: Literal["historical_source_collection_only"]
    production_ready: Literal[False]
    intervals: Annotated[tuple[IssuerQueryInterval, ...], Field(min_length=1, max_length=2000)]


def load_news_query_scope(path: Path, *, root: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    root = root.resolve()
    path = resolve_inside_authority(root, str(path))
    if path.stat().st_size > 1024**2:
        raise DataReadinessError("issuer query scope exceeds size limit")
    digest = file_sha256(path)
    scope = NewsQueryScope.model_validate_json(path.read_bytes())
    rows: list[dict[str, Any]] = []
    pins: dict[str, str] = {}
    prior: list[IssuerQueryInterval] = []
    for interval in scope.intervals:
        for old in prior:
            if (old.ticker == interval.ticker or old.security_id == interval.security_id) and (
                old.start_utc < interval.end_exclusive_utc and interval.start_utc < old.end_exclusive_utc
            ):
                raise DataReadinessError("overlapping or ambiguous issuer query intervals")
        prior.append(interval)
        for pin in interval.source_files:
            if file_sha256(resolve_inside_authority(root, pin.path)) != pin.sha256:
                raise DataReadinessError("issuer query source pin differs")
            pins[pin.path] = pin.sha256
        rows.append({"ticker": interval.ticker, "security_id": interval.security_id,
            "effective_from_utc": interval.start_utc, "effective_to_utc": interval.end_exclusive_utc})
    if file_sha256(path) != digest:
        raise DataReadinessError("issuer query scope changed while loading")
    return pd.DataFrame(rows), {"query_scope_path": str(path), "query_scope_sha256": digest,
        "query_scope_root": str(root), "query_source_files": pins,
        "scope_kind": "explicit_issuer_query_intervals_not_index_membership"}
