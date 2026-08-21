"""Verified source-news shard inventory shared by sentiment and lineage."""
from __future__ import annotations



import hashlib
import json
from pathlib import Path
from typing import Any

from market_predictor.canonical.store import load_canonical_artifact
from market_predictor.core.errors import DataReadinessError


def build_source_news_shard_inventory(
    collection_dir: Path,
    collection: dict[str, Any],
) -> list[dict[str, Any]]:
    artifacts_raw = collection.get("artifacts")
    if not isinstance(artifacts_raw, list):
        raise DataReadinessError("collection manifest has no artifact inventory")
    artifacts = {
        str(record.get("chunk_id", "")): {
            str(key): value for key, value in record.items()
        }
        for record in artifacts_raw
        if isinstance(record, dict)
    }
    if "" in artifacts or len(artifacts) != len(artifacts_raw):
        raise DataReadinessError("collection event artifact identities are invalid")

    request = _json_object(collection_dir / "_request.json")
    if request.get("request_sha256") != collection.get("request_sha256"):
        raise DataReadinessError("collection request and manifest identities do not match")
    work_units_raw = request.get("work_units")
    if not isinstance(work_units_raw, list):
        raise DataReadinessError("collection request has no work-unit inventory")
    work_units = [
        {str(key): value for key, value in record.items()}
        for record in work_units_raw
        if isinstance(record, dict)
    ]
    if len(work_units) != len(work_units_raw):
        raise DataReadinessError("collection work-unit inventory is invalid")

    ledger_path = _recorded_artifact_path(
        str(collection.get("source_collections_path", "")),
        collection_dir=collection_dir,
    )
    ledger, ledger_manifest = load_canonical_artifact(
        ledger_path,
        expected_type="source_collections",
        allow_research=True,
    )
    ledger_sha256 = str(ledger_manifest.get("artifact_sha256", ""))
    if (
        ledger_sha256 != str(collection.get("source_collections_sha256", ""))
        or bool(ledger_manifest.get("production_ready"))
    ):
        raise DataReadinessError("collection source-ledger identity is invalid")
    if "chunk_id" not in ledger.columns or bool(
        ledger["chunk_id"].astype(str).duplicated().any()
    ):
        raise DataReadinessError("collection source-ledger chunk identities are invalid")
    ledger_by_chunk = {
        str(row["chunk_id"]): row
        for row in ledger.to_dict(orient="records")
    }
    unit_ids = [str(unit.get("chunk_id", "")) for unit in work_units]
    if (
        "" in unit_ids
        or len(set(unit_ids)) != len(unit_ids)
        or set(ledger_by_chunk) != set(unit_ids)
    ):
        raise DataReadinessError("collection work units and source ledger do not match")

    inventory: list[dict[str, Any]] = []
    for unit in work_units:
        chunk_id = str(unit["chunk_id"])
        ticker = str(unit.get("ticker", ""))
        security_id = str(unit.get("security_id", ""))
        ledger_row = ledger_by_chunk[chunk_id]
        if (
            str(ledger_row.get("ticker", "")) != ticker
            or str(ledger_row.get("security_id", "")) != security_id
        ):
            raise DataReadinessError(f"collection source identity mismatch for {chunk_id}")
        status = str(ledger_row.get("status", ""))
        artifact = artifacts.get(chunk_id)
        if status == "observed":
            if artifact is None:
                raise DataReadinessError(f"observed source shard has no artifact: {chunk_id}")
            inventory.append({**artifact, "source_empty": False})
            continue
        if status != "observed_empty" or int(ledger_row.get("row_count", -1)) != 0:
            raise DataReadinessError(f"source shard is not terminal for sentiment: {chunk_id}")
        if artifact is not None:
            raise DataReadinessError(f"empty source shard unexpectedly has events: {chunk_id}")
        empty_evidence_sha256 = _sha256_json(
            {
                "schema": "swing.empty_news_source_evidence.v1",
                "collection_request_sha256": collection["request_sha256"],
                "source_collections_sha256": ledger_sha256,
                "chunk_id": chunk_id,
                "ticker": ticker,
                "security_id": security_id,
                "start_utc": str(unit.get("start_utc", "")),
                "end_exclusive_utc": str(unit.get("end_exclusive_utc", "")),
                "status": status,
                "row_count": 0,
            }
        )
        inventory.append(
            {
                "chunk_id": chunk_id,
                "ticker": ticker,
                "security_id": security_id,
                "sha256": empty_evidence_sha256,
                "source_empty": True,
            }
        )
    if set(artifacts).difference(unit_ids):
        raise DataReadinessError("collection manifest has artifacts outside its work units")
    return inventory


def _recorded_artifact_path(value: str, *, collection_dir: Path) -> Path:
    path = Path(value)
    if path.exists():
        return path
    local = collection_dir / path.name
    if local.exists():
        return local
    raise FileNotFoundError(path)


def _json_object(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise DataReadinessError(f"expected JSON object: {path}")
    return {str(key): value for key, value in payload.items()}


def _sha256_json(payload: object) -> str:
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()
