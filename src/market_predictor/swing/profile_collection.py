from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.v3.errors import DataReadinessError

PROFILE_COLLECTION_SCHEMA = "security_profile_collection.v1"
PROFILE_SCHEMA_VERSION = "security_profile.v1"
PROFILE_AVAILABILITY_POLICY = "first_observed_at_collection"
ProfileBatchFetcher = Callable[[list[str]], Any]


@dataclass(frozen=True, slots=True)
class ProfileCollectionResult:
    status: str
    requested_batches: int
    observed_batches: int
    failed_batches: tuple[str, ...]
    requested_current_tickers: int
    observed_profiles: int
    manifest_path: Path | None


def collect_current_security_profiles(
    *,
    memberships_path: Path,
    out_dir: Path,
    fetch_batch: ProfileBatchFetcher,
    batch_size: int = 4,
    now: Callable[[], datetime] | None = None,
) -> ProfileCollectionResult:
    """Collect current profile evidence without backdating it into training."""

    if batch_size < 1 or batch_size > 4:
        raise ValueError("profile batch_size must be between 1 and 4; the provider silently truncates larger batches")
    clock = now or (lambda: datetime.now(UTC))
    memberships, membership_manifest = load_canonical_artifact(
        memberships_path,
        expected_type="memberships",
        allow_research=True,
    )
    required = {
        "security_id",
        "ticker",
        "effective_from_utc",
        "effective_to_utc",
    }
    missing = sorted(required.difference(memberships.columns))
    if missing:
        raise DataReadinessError(f"profile memberships are missing columns: {missing}")
    collection_time = _utc(clock())
    current = _current_memberships(memberships, collection_time)
    symbols = sorted(current["ticker"].astype(str).unique())
    if not symbols:
        raise DataReadinessError("profile collection has no current securities")
    batches = [symbols[index : index + batch_size] for index in range(0, len(symbols), batch_size)]
    work_units = [
        {
            "batch_id": _batch_id(batch),
            "symbols": batch,
        }
        for batch in batches
    ]
    request = {
        "schema": PROFILE_COLLECTION_SCHEMA,
        "memberships_path": str(memberships_path.resolve()),
        "memberships_sha256": str(membership_manifest["artifact_sha256"]),
        "endpoint": "/symbols/get-profile",
        "batch_size": batch_size,
        "work_units": work_units,
        "knowledge_scope": "current_inference_only",
        "availability_policy": PROFILE_AVAILABILITY_POLICY,
        "production_ready": False,
    }
    request_hash = _sha256_json(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "_manifest.json"
    if final_path.exists():
        raise DataReadinessError(f"completed profile collection is immutable: {final_path}")
    _write_or_validate_request(out_dir / "_request.json", request, request_hash)
    raw_dir = out_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    observed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    for unit in work_units:
        batch_id = str(unit["batch_id"])
        batch_symbols = [str(value) for value in unit["symbols"]]
        path = raw_dir / f"{batch_id}.json"
        try:
            page = _load_profile_page(
                path,
                request_hash=request_hash,
                batch_id=batch_id,
                symbols=batch_symbols,
            )
            if page is None:
                payload = fetch_batch(batch_symbols)
                fetched_at = _utc(clock())
                page = _profile_page(
                    request_hash=request_hash,
                    batch_id=batch_id,
                    symbols=batch_symbols,
                    fetched_at=fetched_at,
                    payload=payload,
                )
                _atomic_json(path, page)
            observed[batch_id] = page
        except Exception as exc:
            failures[batch_id] = f"{type(exc).__name__}: {str(exc)[:500]}"

    if failures or len(observed) != len(work_units):
        status = {
            "schema": PROFILE_COLLECTION_SCHEMA,
            "request_sha256": request_hash,
            "status": "incomplete",
            "requested_batches": len(work_units),
            "observed_batches": len(observed),
            "failed_batches": failures,
            "updated_at_utc": _utc(clock()).isoformat(),
            "production_ready": False,
        }
        _atomic_json(out_dir / "_status.json", status)
        return ProfileCollectionResult(
            status="incomplete",
            requested_batches=len(work_units),
            observed_batches=len(observed),
            failed_batches=tuple(sorted(failures)),
            requested_current_tickers=len(symbols),
            observed_profiles=0,
            manifest_path=None,
        )

    profiles = _normalize_profiles(observed.values(), current)
    coverage = _profile_coverage(memberships, current, profiles)
    _audit_profiles(profiles, current)
    profile_path = out_dir / "profiles.parquet"
    coverage_path = out_dir / "coverage.parquet"
    profile_manifest = write_canonical_artifact(
        profiles,
        profile_path,
        artifact_type="security_profiles_current",
        audit=_passing_audit("current_security_profiles", len(profiles)),
        inputs={
            "profile_collection_request_sha256": request_hash,
            "memberships_sha256": str(membership_manifest["artifact_sha256"]),
        },
        production_ready=False,
    )
    coverage_manifest = write_canonical_artifact(
        coverage,
        coverage_path,
        artifact_type="security_profile_coverage",
        audit=_passing_audit("security_profile_coverage", len(coverage)),
        inputs={
            "profile_collection_request_sha256": request_hash,
            "profiles_sha256": str(profile_manifest["artifact_sha256"]),
        },
        production_ready=False,
    )
    status = {
        "schema": PROFILE_COLLECTION_SCHEMA,
        "request_sha256": request_hash,
        "status": "complete",
        "requested_batches": len(work_units),
        "observed_batches": len(observed),
        "failed_batches": {},
        "requested_current_tickers": len(symbols),
        "observed_profiles": len(profiles),
        "coverage_rows": len(coverage),
        "coverage_dispositions": {str(key): int(value) for key, value in coverage["disposition"].value_counts().items()},
        "profiles_path": str(profile_path),
        "profiles_sha256": str(profile_manifest["artifact_sha256"]),
        "coverage_path": str(coverage_path),
        "coverage_sha256": str(coverage_manifest["artifact_sha256"]),
        "updated_at_utc": _utc(clock()).isoformat(),
        "knowledge_scope": "current_inference_only",
        "availability_policy": PROFILE_AVAILABILITY_POLICY,
        "production_ready": False,
    }
    _atomic_json(out_dir / "_status.json", status)
    _atomic_json(final_path, status)
    return ProfileCollectionResult(
        status="complete",
        requested_batches=len(work_units),
        observed_batches=len(observed),
        failed_batches=(),
        requested_current_tickers=len(symbols),
        observed_profiles=len(profiles),
        manifest_path=final_path,
    )


def _current_memberships(
    memberships: pd.DataFrame,
    observed_at: datetime,
) -> pd.DataFrame:
    data = memberships.copy()
    data["ticker"] = data["ticker"].astype(str).str.upper().str.strip()
    start = pd.to_datetime(data["effective_from_utc"], utc=True)
    end = pd.to_datetime(data["effective_to_utc"], utc=True)
    current = data.loc[start.le(observed_at) & (end.isna() | end.gt(observed_at))].copy()
    if bool(current["ticker"].duplicated().any()):
        duplicates = sorted(current.loc[current["ticker"].duplicated(False), "ticker"].unique())
        raise DataReadinessError(f"current profile ticker maps to multiple securities: {duplicates}")
    return current


def _normalize_profiles(
    pages: Any,
    current: pd.DataFrame,
) -> pd.DataFrame:
    by_ticker = current.set_index("ticker", drop=False)
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    for page in pages:
        fetched_at = _utc(datetime.fromisoformat(str(page["fetched_at_utc"])))
        payload = page["payload"]
        items = payload.get("data", []) if isinstance(payload, dict) else []
        if not isinstance(items, list):
            raise DataReadinessError("profile payload data must be a list")
        for item in items:
            if not isinstance(item, dict):
                continue
            ticker = str(item.get("id", "")).upper().strip()
            attributes = item.get("attributes")
            if ticker not in by_ticker.index or not isinstance(attributes, dict):
                continue
            if ticker in seen:
                raise DataReadinessError(f"profile provider returned duplicate ticker {ticker}")
            description = str(attributes.get("longDesc") or "").strip()
            if not description:
                continue
            membership = by_ticker.loc[ticker]
            if isinstance(membership, pd.DataFrame):
                raise DataReadinessError(f"ambiguous current membership for {ticker}")
            source_material = json.dumps(
                item,
                sort_keys=True,
                separators=(",", ":"),
                ensure_ascii=True,
                default=str,
            )
            rows.append(
                {
                    "security_id": str(membership["security_id"]),
                    "ticker": ticker,
                    "company": str(attributes.get("companyName") or ""),
                    "sector": str(attributes.get("sectorname") or ""),
                    "industry": str(attributes.get("primaryname") or ""),
                    "long_description": description,
                    "provider_ticker_id": str(item.get("tickerId") or ""),
                    "source_family": "seeking_alpha",
                    "source_document_id": (f"rapidapi:seeking-alpha:profile:{ticker}"),
                    "source_content_sha256": hashlib.sha256(source_material.encode("utf-8")).hexdigest(),
                    "observed_at_utc": fetched_at,
                    "available_at_utc": fetched_at,
                    "effective_from_utc": fetched_at,
                    "effective_to_utc": pd.NaT,
                    "knowledge_scope": "current_inference_only",
                    "availability_policy": PROFILE_AVAILABILITY_POLICY,
                    "schema_version": PROFILE_SCHEMA_VERSION,
                }
            )
            seen.add(ticker)
    columns = [
        "security_id",
        "ticker",
        "company",
        "sector",
        "industry",
        "long_description",
        "provider_ticker_id",
        "source_family",
        "source_document_id",
        "source_content_sha256",
        "observed_at_utc",
        "available_at_utc",
        "effective_from_utc",
        "effective_to_utc",
        "knowledge_scope",
        "availability_policy",
        "schema_version",
    ]
    return pd.DataFrame(rows, columns=columns).sort_values(["security_id", "ticker"]).reset_index(drop=True)


def _profile_coverage(
    memberships: pd.DataFrame,
    current: pd.DataFrame,
    profiles: pd.DataFrame,
) -> pd.DataFrame:
    identities = memberships[["security_id", "ticker", "effective_from_utc", "effective_to_utc"]].drop_duplicates()
    current_keys = set(
        zip(
            current["security_id"].astype(str),
            current["ticker"].astype(str),
            strict=False,
        )
    )
    observed_keys = set(
        zip(
            profiles["security_id"].astype(str),
            profiles["ticker"].astype(str),
            strict=False,
        )
    )
    rows = []
    for row in identities.itertuples(index=False):
        key = (str(row.security_id), str(row.ticker))
        disposition = (
            "observed_current_profile"
            if key in observed_keys
            else ("provider_missing_current_profile" if key in current_keys else "not_current_at_collection")
        )
        rows.append(
            {
                "security_id": key[0],
                "ticker": key[1],
                "effective_from_utc": row.effective_from_utc,
                "effective_to_utc": row.effective_to_utc,
                "disposition": disposition,
                "profile_eligible_for_historical_training": False,
            }
        )
    return pd.DataFrame(rows).sort_values(["security_id", "ticker", "effective_from_utc"]).reset_index(drop=True)


def _audit_profiles(
    profiles: pd.DataFrame,
    current: pd.DataFrame,
) -> None:
    if bool(profiles["security_id"].astype(str).duplicated().any()):
        raise DataReadinessError("current profiles duplicate security IDs")
    current_ids = set(current["security_id"].astype(str))
    if not set(profiles["security_id"].astype(str)).issubset(current_ids):
        raise DataReadinessError("profile does not map to a current security")
    available = pd.to_datetime(profiles["available_at_utc"], utc=True)
    effective = pd.to_datetime(profiles["effective_from_utc"], utc=True)
    if bool(available.isna().any() or effective.ne(available).any()):
        raise DataReadinessError("profile availability/effectivity is invalid")
    if set(profiles["knowledge_scope"].astype(str)) - {"current_inference_only"}:
        raise DataReadinessError("profile knowledge scope is invalid")


def _profile_page(
    *,
    request_hash: str,
    batch_id: str,
    symbols: list[str],
    fetched_at: datetime,
    payload: Any,
) -> dict[str, Any]:
    payload_hash = _sha256_json(payload)
    content = {
        "schema": PROFILE_COLLECTION_SCHEMA,
        "request_sha256": request_hash,
        "batch_id": batch_id,
        "symbols": symbols,
        "fetched_at_utc": fetched_at.isoformat(),
        "payload_sha256": payload_hash,
        "payload": payload,
    }
    return {**content, "page_sha256": _sha256_json(content)}


def _load_profile_page(
    path: Path,
    *,
    request_hash: str,
    batch_id: str,
    symbols: list[str],
) -> dict[str, Any] | None:
    if not path.exists():
        return None
    page = _json_object(path)
    page_hash = str(page.pop("page_sha256", ""))
    if page_hash != _sha256_json(page):
        raise DataReadinessError(f"profile page self-hash mismatch: {path}")
    page["page_sha256"] = page_hash
    if (
        page.get("request_sha256") != request_hash
        or page.get("batch_id") != batch_id
        or page.get("symbols") != symbols
        or page.get("payload_sha256") != _sha256_json(page.get("payload"))
    ):
        raise DataReadinessError(f"profile page lineage mismatch: {path}")
    return page


def _batch_id(symbols: list[str]) -> str:
    return hashlib.sha256(",".join(symbols).encode("utf-8")).hexdigest()[:24]


def _passing_audit(name: str, rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name=name,
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="identity, availability, evidence, and scope validated",
            ),
        )
    )


def _write_or_validate_request(
    path: Path,
    request: dict[str, Any],
    request_hash: str,
) -> None:
    payload = {**request, "request_sha256": request_hash}
    if path.exists():
        if _json_object(path) != payload:
            raise DataReadinessError(f"profile resume request does not match {path}")
        return
    _atomic_json(path, payload)


def _json_object(path: Path) -> dict[str, Any]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _sha256_json(value: Any) -> str:
    encoded = json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(value, indent=2, sort_keys=True, default=str) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def _utc(value: datetime) -> datetime:
    if value.tzinfo is None:
        raise ValueError("profile timestamps must be timezone-aware")
    return value.astimezone(UTC)
