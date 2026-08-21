from __future__ import annotations

import json
import tempfile
import unittest
from collections.abc import Callable
from pathlib import Path
from typing import cast

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
from market_predictor.swing.event_attribution_history import (
    ATTRIBUTION_MANIFEST_SCHEMA,
    attribute_alpaca_news_history,
    load_event_attribution_history,
)
from market_predictor.swing.news_history import NEWS_HISTORY_MANIFEST_SCHEMA
from market_predictor.core.errors import DataReadinessError


class SwingEventAttributionHistoryTests(unittest.TestCase):
    def test_publishes_hash_bound_relation_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            authority_dir, result = _publish_authority(root)

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["relation_rows"], 1)
            self.assertEqual(
                result["channel_counts"],
                {
                    "direct_issuer": 1,
                    "business_exposure": 0,
                    "sector_context": 0,
                },
            )
            loaded = load_event_attribution_history(
                authority_dir,
                expected_manifest_sha256=file_sha256(
                    authority_dir / "_manifest.json"
                ),
            )
            self.assertEqual(loaded.manifest["schema"], ATTRIBUTION_MANIFEST_SCHEMA)
            self.assertEqual(len(loaded.artifact_records), 1)
            relations, child_manifest = load_canonical_artifact(
                authority_dir / "relations" / "chunk-1.parquet",
                expected_type="event_security_relations",
                allow_research=True,
            )
            self.assertEqual(
                relations["relation_channel"].tolist(),
                ["direct_issuer"],
            )
            self.assertFalse(child_manifest["production_ready"])
            self.assertFalse((authority_dir / "relations" / "chunk-1.parquet.lock").exists())
            with self.assertRaisesRegex(DataReadinessError, "research-only"):
                load_event_attribution_history(
                    authority_dir,
                    require_production_ready=True,
                )
            with self.assertRaisesRegex(
                DataReadinessError,
                "immutable",
            ):
                attribute_alpaca_news_history(
                    collection_dir=root / "collection",
                    collection_audit_path=root / "audit.json",
                    business_labels_path=root / "labels.parquet",
                    security_identities_path=root / "identities.parquet",
                    out_dir=authority_dir,
                )

    def test_rejects_root_contract_request_and_expected_identity_tampering(self) -> None:
        cases: tuple[tuple[str, str, Callable[[dict[str, object]], None]], ...] = (
            ("schema", "manifest", lambda value: value.__setitem__("schema", "stale.schema")),
            ("status", "manifest", lambda value: value.__setitem__("status", "incomplete")),
            ("production", "manifest", lambda value: value.__setitem__("production_ready", True)),
            (
                "request",
                "request",
                lambda value: value.__setitem__("excluded_security_ids", ["security:other"]),
            ),
        )
        for name, target, mutation in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                authority_dir, _ = _publish_authority(Path(temporary))
                path = authority_dir / f"_{target}.json"
                payload = _read_json(path)
                mutation(payload)
                _write_json(path, payload)
                with self.assertRaises(DataReadinessError):
                    load_event_attribution_history(authority_dir)

        with tempfile.TemporaryDirectory() as temporary:
            authority_dir, _ = _publish_authority(Path(temporary))
            wrong_identity = "f" * 64
            self.assertNotEqual(file_sha256(authority_dir / "_manifest.json"), wrong_identity)
            with self.assertRaisesRegex(DataReadinessError, "manifest identity"):
                load_event_attribution_history(
                    authority_dir,
                    expected_manifest_sha256=wrong_identity,
                )

    def test_rejects_source_and_child_tampering(self) -> None:
        cases = ("source_manifest", "child_data", "child_lineage")
        for name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                authority_dir, _ = _publish_authority(root)
                if name == "source_manifest":
                    collection_manifest = root / "collection" / "_manifest.json"
                    payload = _read_json(collection_manifest)
                    payload["total_rows"] = 2
                    _write_json(collection_manifest, payload)
                elif name == "child_data":
                    child = authority_dir / "relations" / "chunk-1.parquet"
                    child.write_bytes(child.read_bytes() + b"tampered")
                else:
                    child_manifest_path = manifest_path_for(
                        authority_dir / "relations" / "chunk-1.parquet"
                    )
                    payload = _read_json(child_manifest_path)
                    inputs = _object(payload.get("inputs"), "child inputs")
                    inputs["chunk_id"] = "another-chunk"
                    payload["inputs"] = inputs
                    _write_json(child_manifest_path, payload)
                with self.assertRaises(DataReadinessError):
                    load_event_attribution_history(authority_dir)

    def test_rejects_path_escape_partial_inventory_and_count_tampering(self) -> None:
        cases = ("path_escape", "partial_inventory", "row_count", "channel_count")
        for name in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as temporary:
                root = Path(temporary)
                authority_dir, _ = _publish_authority(root)
                if name == "partial_inventory":
                    manifest_path_for(
                        authority_dir / "relations" / "chunk-1.parquet"
                    ).unlink()
                else:
                    manifest = _read_json(authority_dir / "_manifest.json")
                    artifact = _first_object(
                        manifest.get("artifacts"),
                        "manifest artifacts",
                    )
                    if name == "path_escape":
                        artifact["path"] = str(root / "outside.parquet")
                    elif name == "row_count":
                        artifact["rows"] = 2
                    else:
                        channel_counts = _object(
                            artifact.get("channel_counts"),
                            "artifact channel counts",
                        )
                        channel_counts["direct_issuer"] = 2
                    _write_root_payloads(authority_dir, manifest)
                with self.assertRaises(DataReadinessError):
                    load_event_attribution_history(authority_dir)


def _publish_authority(root: Path) -> tuple[Path, dict[str, object]]:
    collection = root / "collection"
    events_path = collection / "events" / "chunk-1.parquet"
    labels_path = root / "labels.parquet"
    identities_path = root / "identities.parquet"
    collection_request_sha256 = "a" * 64
    event_manifest = write_canonical_artifact(
        _events(),
        events_path,
        artifact_type="events",
        audit=_audit(1),
        inputs={
            "collection_request_sha256": collection_request_sha256,
            "chunk_id": "chunk-1",
        },
        production_ready=False,
    )
    write_canonical_artifact(
        _labels(),
        labels_path,
        artifact_type="security_business_labels",
        audit=_audit(1),
        production_ready=False,
    )
    write_canonical_artifact(
        _identities(),
        identities_path,
        artifact_type="security_business_label_coverage",
        audit=_audit(1),
        production_ready=False,
    )
    collection.mkdir(parents=True, exist_ok=True)
    _write_json(
        collection / "_manifest.json",
        {
            "schema": NEWS_HISTORY_MANIFEST_SCHEMA,
            "status": "complete",
            "production_ready": False,
            "availability_policy": "provider_publication_proxy",
            "request_sha256": collection_request_sha256,
            "requested_chunks": 1,
            "observed_chunks": 1,
            "empty_chunks": 0,
            "failed_chunks": {},
            "artifacts": [
                {
                    "chunk_id": "chunk-1",
                    "security_id": "security:wdc",
                    "ticker": "WDC",
                    "path": str(events_path.resolve()),
                    "manifest_path": str(manifest_path_for(events_path).resolve()),
                    "sha256": event_manifest["artifact_sha256"],
                    "rows": 1,
                }
            ],
            "artifact_count": 1,
            "total_rows": 1,
        },
    )
    audit_path = root / "audit.json"
    _write_json(
        audit_path,
        {
            "passed": True,
            "request_sha256": collection_request_sha256,
            "coverage_blindspot_security_ids": [],
        },
    )
    authority_dir = root / "relations"
    result = attribute_alpaca_news_history(
        collection_dir=collection,
        collection_audit_path=audit_path,
        business_labels_path=labels_path,
        security_identities_path=identities_path,
        out_dir=authority_dir,
    )
    return authority_dir, result


def _read_json(path: Path) -> dict[str, object]:
    value = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(value, dict):
        raise AssertionError(f"expected JSON object: {path}")
    return {str(key): item for key, item in value.items()}


def _write_json(path: Path, value: dict[str, object]) -> None:
    path.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _write_root_payloads(authority_dir: Path, value: dict[str, object]) -> None:
    _write_json(authority_dir / "_manifest.json", value)
    _write_json(authority_dir / "_status.json", value)


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise AssertionError(f"expected object for {label}")
    return cast(dict[str, object], value)


def _first_object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, list) or not value:
        raise AssertionError(f"expected non-empty list for {label}")
    return _object(value[0], label)


def _events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["event-1"],
            "security_id": ["security:wdc"],
            "ticker": ["WDC"],
            "feature_available_at_utc": [pd.Timestamp("2025-01-02T14:00:00Z")],
            "title": ["$WDC reports quarterly results"],
            "summary": [""],
            "text": [""],
        }
    )


def _labels() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["security:wdc"],
            "ticker": ["WDC"],
            "company": ["Western Digital"],
            "business_tag": ["offering.hardware.storage"],
            "label_type": ["offering"],
            "match_terms": ['["hard disk drives"]'],
            "tag_rank": [1],
            "confidence": [0.6],
            "relation_use": ["context"],
            "effective_from_utc": [pd.Timestamp("2021-01-01T00:00:00Z")],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [pd.Timestamp("2021-01-01T00:00:00Z")],
        }
    )


def _identities() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "security_id": ["security:wdc"],
            "ticker": ["WDC"],
            "company": ["Western Digital"],
            "effective_from_utc": [
                pd.Timestamp("2021-01-01T00:00:00Z")
            ],
            "effective_to_utc": [pd.NaT],
            "available_at_utc": [
                pd.Timestamp("2021-01-01T00:00:00Z")
            ],
        }
    )


def _audit(rows: int) -> CanonicalAuditReport:
    return CanonicalAuditReport(
        checks=(
            CanonicalAuditCheck(
                name="fixture",
                status="pass",
                failures=0,
                rows_checked=rows,
                detail="test fixture",
            ),
        )
    )


if __name__ == "__main__":
    unittest.main()
