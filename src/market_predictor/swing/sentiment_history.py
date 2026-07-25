from __future__ import annotations

import hashlib
import json
import os
from collections.abc import Callable
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from uuid import uuid4

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    file_sha256,
    load_canonical_artifact,
    manifest_path_for,
    write_canonical_artifact,
)
from market_predictor.resources import (
    assert_memory_budget,
    memory_audit,
    release_process_memory,
)
from market_predictor.sentiment import build_sentiment_inputs
from market_predictor.swing.event_relevance import (
    RELEVANCE_POLICY_VERSION,
    SecurityMetadata,
    add_event_relevance,
)
from market_predictor.v3.errors import DataReadinessError

SENTIMENT_REQUEST_SCHEMA = "swing.event_sentiment_request.v1"
SENTIMENT_MANIFEST_SCHEMA = "swing.event_sentiment_manifest.v1"
SENTIMENT_SCHEMA_VERSION = "swing.event_sentiment.v1"
SENTIMENT_AVAILABILITY_POLICY = (
    "provider_publication_proxy_plus_fixed_inference_latency"
)


class TextScorer(Protocol):
    def score_texts(
        self,
        texts: list[str],
        batch_size: int = 16,
    ) -> pd.DataFrame: ...


def score_alpaca_news_history(
    *,
    collection_dir: Path,
    collection_audit_path: Path,
    universe_path: Path,
    out_dir: Path,
    scorer: TextScorer,
    model_name: str,
    model_revision: str,
    execution_device: str,
    text_mode: str = "title_summary",
    max_length: int = 128,
    batch_size: int = 32,
    fixed_latency_minutes: int = 5,
    memory_budget_gib: float = 4.0,
    memory_headroom_gib: float = 0.75,
    progress: Callable[[dict[str, Any]], None] | None = None,
) -> dict[str, Any]:
    """Score audited event chunks sequentially and publish research-only evidence."""

    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    if max_length < 1 or max_length > 512:
        raise ValueError("max_length must be between 1 and 512")
    if fixed_latency_minutes < 0 or fixed_latency_minutes > 60:
        raise ValueError("fixed_latency_minutes must be between 0 and 60")
    collection_manifest_path = collection_dir / "_manifest.json"
    if not collection_manifest_path.exists():
        raise FileNotFoundError(collection_manifest_path)
    collection = _json_object(collection_manifest_path)
    audit = _json_object(collection_audit_path)
    if (
        collection.get("status") != "complete"
        or bool(collection.get("production_ready"))
        or not bool(audit.get("passed"))
        or audit.get("request_sha256") != collection.get("request_sha256")
    ):
        raise DataReadinessError(
            "sentiment scoring requires a passed research-only collection audit"
        )
    excluded_security_ids = tuple(
        sorted(
            str(value)
            for value in audit.get("coverage_blindspot_security_ids", [])
        )
    )
    universe = _read_universe(universe_path)
    metadata = _metadata_by_security_and_ticker(universe)
    artifacts_raw = collection.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise DataReadinessError("collection manifest has no artifact inventory")
    artifacts = [
        {str(key): value for key, value in record.items()}
        for record in artifacts_raw
        if isinstance(record, dict)
    ]

    request = {
        "schema": SENTIMENT_REQUEST_SCHEMA,
        "collection_manifest_path": str(collection_manifest_path.resolve()),
        "collection_manifest_sha256": file_sha256(collection_manifest_path),
        "collection_request_sha256": str(collection["request_sha256"]),
        "collection_audit_path": str(collection_audit_path.resolve()),
        "collection_audit_sha256": file_sha256(collection_audit_path),
        "universe_path": str(universe_path.resolve()),
        "universe_sha256": file_sha256(universe_path),
        "model_name": model_name,
        "model_revision": model_revision,
        "execution_device": execution_device,
        "text_mode": text_mode,
        "max_length": max_length,
        "batch_size": batch_size,
        "fixed_latency_minutes": fixed_latency_minutes,
        "sentiment_availability_policy": SENTIMENT_AVAILABILITY_POLICY,
        "relevance_policy_version": RELEVANCE_POLICY_VERSION,
        "excluded_security_ids": list(excluded_security_ids),
        "production_ready": False,
    }
    request_hash = _sha256_json(request)
    out_dir.mkdir(parents=True, exist_ok=True)
    final_path = out_dir / "_manifest.json"
    if final_path.exists():
        raise DataReadinessError(
            f"completed sentiment history is immutable: {final_path}"
        )
    _write_or_validate_request(out_dir / "_request.json", request, request_hash)
    scored_dir = out_dir / "sentiment"
    attempts_dir = out_dir / "attempts"
    scored_dir.mkdir(parents=True, exist_ok=True)
    attempts_dir.mkdir(parents=True, exist_ok=True)
    assert_memory_budget(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
        stage="historical sentiment start",
    )

    observed: dict[str, dict[str, Any]] = {}
    failures: dict[str, str] = {}
    skipped = 0
    eligible_artifacts = [
        artifact
        for artifact in artifacts
        if str(artifact.get("security_id", "")) not in excluded_security_ids
    ]
    for index, artifact in enumerate(eligible_artifacts):
        chunk_id = str(artifact.get("chunk_id", ""))
        security_id = str(artifact.get("security_id", ""))
        ticker = str(artifact.get("ticker", ""))
        target = scored_dir / f"{chunk_id}.parquet"
        existing = _load_existing_sentiment(
            path=target,
            request_hash=request_hash,
            chunk_id=chunk_id,
            source_artifact_sha256=str(artifact.get("sha256", "")),
            ticker=ticker,
            security_id=security_id,
        )
        if existing is not None:
            observed[chunk_id] = existing
            skipped += 1
            _progress(
                progress,
                index=index + 1,
                total=len(eligible_artifacts),
                chunk_id=chunk_id,
                status="skipped",
                rows=int(existing["rows"]),
            )
            continue

        started_at = datetime.now(UTC)
        try:
            source_path = Path(str(artifact["path"]))
            events, event_manifest = load_canonical_artifact(
                source_path,
                expected_type="events",
                allow_research=True,
            )
            if str(event_manifest["artifact_sha256"]) != str(
                artifact.get("sha256", "")
            ):
                raise DataReadinessError(
                    f"source event hash mismatch for {chunk_id}"
                )
            security_metadata = metadata.get((security_id, ticker))
            if security_metadata is None:
                raise DataReadinessError(
                    f"no point-in-time company metadata for {security_id}/{ticker}"
                )
            relevant = add_event_relevance(events, security_metadata)
            inputs = build_sentiment_inputs(relevant, mode=text_mode)
            scores = scorer.score_texts(
                inputs.astype(str).tolist(),
                batch_size=batch_size,
            )
            if len(scores) != len(events):
                raise DataReadinessError(
                    f"FinBERT row count mismatch for {chunk_id}"
                )
            completed_at = datetime.now(UTC)
            sentiment = _sentiment_frame(
                relevant,
                scores,
                inputs=inputs,
                model_name=model_name,
                model_revision=model_revision,
                text_mode=text_mode,
                max_length=max_length,
                fixed_latency_minutes=fixed_latency_minutes,
                computed_at_utc=completed_at,
            )
            _audit_sentiment_frame(sentiment)
            manifest = write_canonical_artifact(
                sentiment,
                target,
                artifact_type="event_sentiment_research",
                audit=_passing_audit(len(sentiment)),
                inputs={
                    "sentiment_request_sha256": request_hash,
                    "source_event_artifact_sha256": str(
                        event_manifest["artifact_sha256"]
                    ),
                    "chunk_id": chunk_id,
                },
                production_ready=False,
            )
            record = _artifact_record(
                artifact,
                target,
                sentiment,
                manifest,
                started_at=started_at,
                completed_at=completed_at,
            )
            observed[chunk_id] = record
            _write_attempt(
                attempts_dir,
                chunk_id=chunk_id,
                request_hash=request_hash,
                status="observed",
                started_at=started_at,
                completed_at=completed_at,
                rows=len(sentiment),
                error=None,
            )
            _progress(
                progress,
                index=index + 1,
                total=len(eligible_artifacts),
                chunk_id=chunk_id,
                status="observed",
                rows=len(sentiment),
            )
        except Exception as exc:
            completed_at = datetime.now(UTC)
            failures[chunk_id] = f"{type(exc).__name__}: {str(exc)[:500]}"
            _write_attempt(
                attempts_dir,
                chunk_id=chunk_id,
                request_hash=request_hash,
                status="failed",
                started_at=started_at,
                completed_at=completed_at,
                rows=0,
                error=failures[chunk_id],
            )
            _progress(
                progress,
                index=index + 1,
                total=len(eligible_artifacts),
                chunk_id=chunk_id,
                status="failed",
                rows=0,
            )
        finally:
            release_process_memory()
            assert_memory_budget(
                hard_budget_gib=memory_budget_gib,
                headroom_gib=memory_headroom_gib,
                stage=f"historical sentiment {chunk_id}",
            )

    status = (
        "complete"
        if not failures and len(observed) == len(eligible_artifacts)
        else "incomplete"
    )
    memory = memory_audit(
        hard_budget_gib=memory_budget_gib,
        headroom_gib=memory_headroom_gib,
    ).to_record()
    status_payload: dict[str, Any] = {
        "schema": SENTIMENT_MANIFEST_SCHEMA,
        "request_sha256": request_hash,
        "status": status,
        "updated_at_utc": datetime.now(UTC).isoformat(),
        "requested_chunks": len(eligible_artifacts),
        "observed_chunks": len(observed),
        "failed_chunks": failures,
        "skipped_chunks": skipped,
        "excluded_security_ids": list(excluded_security_ids),
        "excluded_chunks": len(artifacts) - len(eligible_artifacts),
        "total_rows": sum(int(record["rows"]) for record in observed.values()),
        "memory": memory,
        "production_ready": False,
        "sentiment_availability_policy": SENTIMENT_AVAILABILITY_POLICY,
    }
    _atomic_json(out_dir / "_status.json", status_payload)
    if status != "complete":
        return status_payload
    final_payload = {
        **status_payload,
        "artifacts": [
            observed[str(artifact["chunk_id"])]
            for artifact in eligible_artifacts
        ],
    }
    _atomic_json(final_path, final_payload)
    return final_payload


def _sentiment_frame(
    events: pd.DataFrame,
    scores: pd.DataFrame,
    *,
    inputs: pd.Series,
    model_name: str,
    model_revision: str,
    text_mode: str,
    max_length: int,
    fixed_latency_minutes: int,
    computed_at_utc: datetime,
) -> pd.DataFrame:
    required_scores = {
        "sentiment_label",
        "sentiment_score",
        "sentiment_numeric",
    }
    missing = sorted(required_scores.difference(scores.columns))
    if missing:
        raise DataReadinessError(
            f"FinBERT output is missing columns: {missing}"
        )
    event_available = pd.to_datetime(events["available_at_utc"], utc=True)
    output = pd.DataFrame(
        {
            "event_id": events["event_id"].astype(str).to_numpy(),
            "security_id": events["security_id"].astype(str).to_numpy(),
            "ticker": events["ticker"].astype(str).to_numpy(),
            "source_family": events["source_family"].astype(str).to_numpy(),
            "published_at_utc": pd.to_datetime(
                events["published_at_utc"],
                utc=True,
            ).to_numpy(),
            "event_available_at_utc": event_available.to_numpy(),
            "research_feature_available_at_utc": (
                event_available + pd.Timedelta(minutes=fixed_latency_minutes)
            ).to_numpy(),
            "inference_computed_at_utc": computed_at_utc,
            "sentiment_label": scores["sentiment_label"]
            .astype(str)
            .str.lower()
            .to_numpy(),
            "sentiment_confidence": pd.to_numeric(
                scores["sentiment_score"],
                errors="coerce",
            ).to_numpy(),
            "sentiment_numeric": pd.to_numeric(
                scores["sentiment_numeric"],
                errors="coerce",
            ).to_numpy(),
            "relevance": pd.to_numeric(
                events["relevance"],
                errors="coerce",
            ).to_numpy(),
            "relevance_basis": events["relevance_basis"].astype(str).to_numpy(),
            "relevance_policy_version": RELEVANCE_POLICY_VERSION,
            "sentiment_input_sha256": inputs.astype(str).map(
                lambda value: hashlib.sha256(
                    value.encode("utf-8")
                ).hexdigest()
            ).to_numpy(),
            "sentiment_model": model_name,
            "sentiment_model_revision": model_revision,
            "sentiment_input_mode": text_mode,
            "sentiment_max_length": max_length,
            "sentiment_availability_policy": SENTIMENT_AVAILABILITY_POLICY,
            "schema_version": SENTIMENT_SCHEMA_VERSION,
        }
    )
    return output


def _audit_sentiment_frame(frame: pd.DataFrame) -> None:
    if frame.empty:
        raise DataReadinessError("sentiment artifact cannot be empty")
    if bool(frame["event_id"].astype(str).duplicated().any()):
        raise DataReadinessError("sentiment artifact has duplicate event IDs")
    confidence = pd.to_numeric(frame["sentiment_confidence"], errors="coerce")
    numeric = pd.to_numeric(frame["sentiment_numeric"], errors="coerce")
    relevance = pd.to_numeric(frame["relevance"], errors="coerce")
    if bool(
        confidence.isna().any()
        or confidence.lt(0).any()
        or confidence.gt(1).any()
        or numeric.isna().any()
        or numeric.lt(-1).any()
        or numeric.gt(1).any()
        or relevance.isna().any()
        or relevance.lt(0).any()
    ):
        raise DataReadinessError("sentiment/relevance values are out of bounds")
    available = pd.to_datetime(
        frame["event_available_at_utc"],
        utc=True,
    )
    feature_available = pd.to_datetime(
        frame["research_feature_available_at_utc"],
        utc=True,
    )
    computed = pd.to_datetime(
        frame["inference_computed_at_utc"],
        utc=True,
    )
    if bool(
        feature_available.lt(available).any()
        or computed.lt(available).any()
    ):
        raise DataReadinessError(
            "sentiment timing precedes source evidence availability"
        )


def _metadata_by_security_and_ticker(
    universe: pd.DataFrame,
) -> dict[tuple[str, str], SecurityMetadata]:
    result: dict[tuple[str, str], SecurityMetadata] = {}
    for (security_id, ticker), part in universe.groupby(
        ["security_id", "ticker"],
        sort=True,
    ):
        values = {
            column: sorted(
                set(part[column].fillna("").astype(str).str.strip())
                - {""}
            )
            for column in ("company", "sector", "industry")
        }
        if any(len(items) != 1 for items in values.values()):
            raise DataReadinessError(
                f"ambiguous company metadata for {security_id}/{ticker}"
            )
        result[(str(security_id), str(ticker))] = SecurityMetadata(
            security_id=str(security_id),
            ticker=str(ticker),
            company=values["company"][0],
            sector=values["sector"][0],
            industry=values["industry"][0],
        )
    return result


def _read_universe(path: Path) -> pd.DataFrame:
    if path.suffix.lower() == ".parquet":
        frame = pd.read_parquet(
            path,
            columns=["security_id", "ticker", "company", "sector", "industry"],
        )
    elif path.suffix.lower() == ".csv":
        frame = pd.read_csv(path)
    else:
        raise ValueError(f"unsupported universe format: {path.suffix}")
    required = {"security_id", "ticker", "company", "sector", "industry"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise DataReadinessError(
            f"sentiment metadata universe is missing columns: {missing}"
        )
    frame = frame.loc[:, sorted(required)].copy()
    frame["ticker"] = frame["ticker"].astype(str).str.upper().str.strip()
    frame["security_id"] = frame["security_id"].astype(str).str.strip()
    return frame


def _load_existing_sentiment(
    *,
    path: Path,
    request_hash: str,
    chunk_id: str,
    source_artifact_sha256: str,
    ticker: str,
    security_id: str,
) -> dict[str, Any] | None:
    if not path.exists() and not manifest_path_for(path).exists():
        return None
    frame, manifest = load_canonical_artifact(
        path,
        expected_type="event_sentiment_research",
        allow_research=True,
    )
    inputs = manifest.get("inputs")
    if (
        not isinstance(inputs, dict)
        or inputs.get("sentiment_request_sha256") != request_hash
        or inputs.get("chunk_id") != chunk_id
        or inputs.get("source_event_artifact_sha256")
        != source_artifact_sha256
    ):
        raise DataReadinessError(
            f"existing sentiment artifact has another request identity: {path}"
        )
    _audit_sentiment_frame(frame)
    return {
        "chunk_id": chunk_id,
        "ticker": ticker,
        "security_id": security_id,
        "source_event_artifact_sha256": source_artifact_sha256,
        "path": str(path),
        "manifest_path": str(manifest_path_for(path)),
        "sha256": str(manifest["artifact_sha256"]),
        "rows": len(frame),
        "first_feature_available_at_utc": pd.to_datetime(
            frame["research_feature_available_at_utc"],
            utc=True,
        ).min().isoformat(),
        "last_feature_available_at_utc": pd.to_datetime(
            frame["research_feature_available_at_utc"],
            utc=True,
        ).max().isoformat(),
    }


def _artifact_record(
    source_artifact: dict[str, Any],
    path: Path,
    frame: pd.DataFrame,
    manifest: dict[str, object],
    *,
    started_at: datetime,
    completed_at: datetime,
) -> dict[str, Any]:
    feature_time = pd.to_datetime(
        frame["research_feature_available_at_utc"],
        utc=True,
    )
    return {
        "chunk_id": str(source_artifact["chunk_id"]),
        "ticker": str(source_artifact["ticker"]),
        "security_id": str(source_artifact["security_id"]),
        "source_event_artifact_sha256": str(source_artifact["sha256"]),
        "path": str(path),
        "manifest_path": str(manifest_path_for(path)),
        "sha256": str(manifest["artifact_sha256"]),
        "rows": len(frame),
        "first_feature_available_at_utc": feature_time.min().isoformat(),
        "last_feature_available_at_utc": feature_time.max().isoformat(),
        "started_at_utc": started_at.isoformat(),
        "completed_at_utc": completed_at.isoformat(),
    }


def _passing_audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="event_sentiment_research",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="sentiment, relevance, timing, and identity validated",
            ),
        )
    )


def _write_attempt(
    attempts_dir: Path,
    *,
    chunk_id: str,
    request_hash: str,
    status: str,
    started_at: datetime,
    completed_at: datetime,
    rows: int,
    error: str | None,
) -> None:
    _atomic_json(
        attempts_dir
        / f"{chunk_id}_{completed_at.strftime('%Y%m%dT%H%M%S%f')}.json",
        {
            "schema": SENTIMENT_REQUEST_SCHEMA,
            "request_sha256": request_hash,
            "chunk_id": chunk_id,
            "status": status,
            "started_at_utc": started_at.isoformat(),
            "completed_at_utc": completed_at.isoformat(),
            "rows": rows,
            "error": error,
        },
    )


def _progress(
    callback: Callable[[dict[str, Any]], None] | None,
    **payload: Any,
) -> None:
    if callback is not None:
        callback(dict(payload))


def _write_or_validate_request(
    path: Path,
    request: dict[str, Any],
    request_hash: str,
) -> None:
    payload = {**request, "request_sha256": request_hash}
    if path.exists():
        if _json_object(path) != payload:
            raise DataReadinessError(
                f"sentiment resume request does not match {path}"
            )
        return
    _atomic_json(path, payload)


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(
        payload,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
        default=str,
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{uuid4().hex}.tmp")
    try:
        temporary.write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)
