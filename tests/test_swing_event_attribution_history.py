from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.swing.event_attribution_history import (
    attribute_alpaca_news_history,
)
from market_predictor.v3.errors import DataReadinessError


class SwingEventAttributionHistoryTests(unittest.TestCase):
    def test_publishes_hash_bound_relation_chunks(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            collection = root / "collection"
            events_dir = collection / "events"
            events_path = events_dir / "chunk-1.parquet"
            labels_path = root / "labels.parquet"
            identities_path = root / "identities.parquet"
            event_manifest = write_canonical_artifact(
                _events(),
                events_path,
                artifact_type="events",
                audit=_audit(1),
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
            (collection / "_manifest.json").write_text(
                json.dumps(
                    {
                        "status": "complete",
                        "production_ready": False,
                        "request_sha256": "request-1",
                        "artifacts": [
                            {
                                "chunk_id": "chunk-1",
                                "security_id": "security:wdc",
                                "ticker": "WDC",
                                "path": str(events_path),
                                "sha256": event_manifest["artifact_sha256"],
                            }
                        ],
                    }
                ),
                encoding="utf-8",
            )
            audit_path = root / "audit.json"
            audit_path.write_text(
                json.dumps(
                    {
                        "passed": True,
                        "request_sha256": "request-1",
                        "coverage_blindspot_security_ids": [],
                    }
                ),
                encoding="utf-8",
            )

            result = attribute_alpaca_news_history(
                collection_dir=collection,
                collection_audit_path=audit_path,
                business_labels_path=labels_path,
                security_identities_path=identities_path,
                out_dir=root / "relations",
            )

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
            relations, manifest = load_canonical_artifact(
                root / "relations" / "relations" / "chunk-1.parquet",
                expected_type="event_security_relations",
                allow_research=True,
            )
            self.assertEqual(
                relations["relation_channel"].tolist(),
                ["direct_issuer"],
            )
            self.assertFalse(manifest["production_ready"])
            with self.assertRaisesRegex(
                DataReadinessError,
                "immutable",
            ):
                attribute_alpaca_news_history(
                    collection_dir=collection,
                    collection_audit_path=audit_path,
                    business_labels_path=labels_path,
                    security_identities_path=identities_path,
                    out_dir=root / "relations",
                )


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
