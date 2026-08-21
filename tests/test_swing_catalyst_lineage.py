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
from market_predictor.swing.catalyst_lineage import (
    _reconcile_sentiment_inventory,
    build_catalyst_lineage,
)
from market_predictor.swing.event_attribution import (
    ATTRIBUTION_POLICY_SHA256,
    ATTRIBUTION_POLICY_VERSION,
)
from market_predictor.core.errors import DataReadinessError


class SwingCatalystLineageTests(unittest.TestCase):
    def test_replays_direct_event_and_excludes_unrelated_sentiment(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish()

            result = fixture.build()

            self.assertEqual(result["status"], "complete")
            self.assertEqual(result["source_event_rows"], 2)
            self.assertEqual(result["related_source_events"], 1)
            self.assertEqual(result["relation_rows"], 1)
            self.assertEqual(result["training_eligible_rows"], 1)
            self.assertEqual(result["assignment_rows"], 2)
            self.assertEqual(
                result["assignment_status_counts"],
                {"assigned": 2},
            )
            events, event_manifest = load_canonical_artifact(
                fixture.output / "events" / "chunk-1.parquet",
                expected_type="catalyst_events",
                allow_research=True,
            )
            self.assertEqual(events["source_event_id"].tolist(), ["event-direct"])
            self.assertEqual(events["event_id"].tolist(), ["relation-direct"])
            self.assertEqual(events["relation_channel"].tolist(), ["direct_issuer"])
            self.assertTrue(events["training_eligible"].all())
            self.assertFalse(event_manifest["production_ready"])
            assignments, _ = load_canonical_artifact(
                fixture.output / "assignments" / "chunk-1.parquet",
                expected_type="catalyst_event_assignments",
                allow_research=True,
            )
            self.assertEqual(
                assignments["window_name"].tolist(),
                ["1d", "3d"],
            )
            self.assertTrue(
                (
                    pd.to_datetime(assignments["feature_available_at_utc"], utc=True)
                    <= pd.to_datetime(assignments["decision_time_utc"], utc=True)
                ).all()
            )
            coverage, _ = load_canonical_artifact(
                fixture.output / "source_coverage.parquet",
                expected_type="catalyst_source_coverage",
                allow_research=True,
            )
            self.assertEqual(coverage["coverage_state"].tolist(), ["observed_complete"])
            inventory = json.loads(
                (fixture.output / "feature_inventory.json").read_text(
                    encoding="utf-8"
                )
            )
            self.assertIn("catalyst_only", inventory["profiles"])
            self.assertIn("technical_plus_catalyst", inventory["profiles"])
            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                fixture.build()

    def test_missing_sentiment_fails_chunk_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish(sentiment_event_ids=["event-direct"])

            result = fixture.build()

            self.assertEqual(result["status"], "incomplete")
            self.assertIn("sentiment event inventory mismatch", result["failed_chunks"]["chunk-1"])
            self.assertFalse((fixture.output / "_manifest.json").exists())

    def test_backdated_relation_fails_chunk_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish(
                relation_feature_available_at=pd.Timestamp(
                    "2025-01-02T13:59:00Z"
                )
            )

            result = fixture.build()

            self.assertEqual(result["status"], "incomplete")
            self.assertIn("backdated availability", result["failed_chunks"]["chunk-1"])

    def test_zero_row_sentiment_for_observed_empty_source_is_ignored(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentiment_dir = Path(temporary) / "sentiment-output"
            empty_path = sentiment_dir / "sentiment" / "empty.parquet"
            empty_manifest = write_canonical_artifact(
                pd.DataFrame(),
                empty_path,
                artifact_type="event_sentiment_research",
                audit=_audit(0),
                inputs={
                    "chunk_id": "empty",
                    "sentiment_request_sha256": "sentiment-request",
                    "source_event_artifact_sha256": "empty-source-sha",
                },
                production_ready=False,
            )
            records = {
                "observed": {"rows": 2},
                "empty": {
                    "path": str(empty_path),
                    "rows": 0,
                    "sha256": empty_manifest["artifact_sha256"],
                    "security_id": "security:wdc",
                    "ticker": "WDC",
                    "source_event_artifact_sha256": "empty-source-sha",
                },
            }
            source_collections = pd.DataFrame(
                {
                    "chunk_id": ["observed", "empty"],
                    "security_id": ["security:wdc", "security:wdc"],
                    "ticker": ["WDC", "WDC"],
                    "status": ["observed", "observed_empty"],
                    "row_count": [2, 0],
                }
            )

            reconciled = _reconcile_sentiment_inventory(
                records,
                eligible_chunk_ids={"observed"},
                source_collections=source_collections,
                excluded_security_ids=set(),
                source_inventory={
                    "empty": {"source_empty": True, "sha256": "empty-source-sha"}
                },
                sentiment_dir=sentiment_dir,
                sentiment_request_sha256="sentiment-request",
            )

            self.assertEqual(reconciled, {"observed": {"rows": 2}})

    def test_nonempty_extra_sentiment_chunk_fails_reconciliation(self) -> None:
        source_collections = pd.DataFrame(
            {
                "chunk_id": ["observed", "extra"],
                "security_id": ["security:wdc", "security:wdc"],
                "ticker": ["WDC", "WDC"],
                "status": ["observed", "observed_empty"],
                "row_count": [2, 0],
            }
        )

        with self.assertRaisesRegex(DataReadinessError, "sentiment chunk inventory"):
            _reconcile_sentiment_inventory(
                {"observed": {"rows": 2}, "extra": {"rows": 1}},
                eligible_chunk_ids={"observed"},
                source_collections=source_collections,
                excluded_security_ids=set(),
                source_inventory={
                    "extra": {"source_empty": True, "sha256": "empty-source-sha"}
                },
                sentiment_dir=Path("sentiment"),
                sentiment_request_sha256="sentiment-request",
            )

    def test_empty_sentiment_identity_mismatches_fail_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            sentiment_dir = root / "sentiment-output"
            empty_path = sentiment_dir / "sentiment" / "empty.parquet"
            manifest = write_canonical_artifact(
                pd.DataFrame(),
                empty_path,
                artifact_type="event_sentiment_research",
                audit=_audit(0),
                inputs={
                    "chunk_id": "empty",
                    "sentiment_request_sha256": "sentiment-request",
                    "source_event_artifact_sha256": "empty-source-sha",
                },
                production_ready=False,
            )
            base_record = {
                "path": str(empty_path),
                "rows": 0,
                "sha256": manifest["artifact_sha256"],
                "security_id": "security:wdc",
                "ticker": "WDC",
                "source_event_artifact_sha256": "empty-source-sha",
            }
            source_collections = pd.DataFrame(
                {
                    "chunk_id": ["observed", "empty"],
                    "security_id": ["security:wdc", "security:wdc"],
                    "ticker": ["WDC", "WDC"],
                    "status": ["observed", "observed_empty"],
                    "row_count": [2, 0],
                }
            )
            cases = (
                ("artifact hash", {"sha256": "0" * 64}, "sentiment-request", set()),
                ("security", {"security_id": "security:other"}, "sentiment-request", set()),
                (
                    "source evidence",
                    {"source_event_artifact_sha256": "wrong-source"},
                    "sentiment-request",
                    set(),
                ),
                ("request", {}, "wrong-request", set()),
                ("excluded", {}, "sentiment-request", {"security:wdc"}),
            )
            for name, mutation, request_sha256, excluded in cases:
                with self.subTest(name=name):
                    with self.assertRaises(DataReadinessError):
                        _reconcile_sentiment_inventory(
                            {
                                "observed": {"rows": 2},
                                "empty": {**base_record, **mutation},
                            },
                            eligible_chunk_ids={"observed"},
                            source_collections=source_collections,
                            excluded_security_ids=excluded,
                            source_inventory={
                                "empty": {
                                    "source_empty": True,
                                    "sha256": "empty-source-sha",
                                }
                            },
                            sentiment_dir=sentiment_dir,
                            sentiment_request_sha256=request_sha256,
                        )

    def test_tampered_empty_sentiment_artifact_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            sentiment_dir = Path(temporary) / "sentiment-output"
            empty_path = sentiment_dir / "sentiment" / "empty.parquet"
            manifest = write_canonical_artifact(
                pd.DataFrame(),
                empty_path,
                artifact_type="event_sentiment_research",
                audit=_audit(0),
                inputs={
                    "chunk_id": "empty",
                    "sentiment_request_sha256": "sentiment-request",
                    "source_event_artifact_sha256": "empty-source-sha",
                },
                production_ready=False,
            )
            empty_path.write_bytes(empty_path.read_bytes() + b"tampered")

            with self.assertRaises(DataReadinessError):
                _reconcile_sentiment_inventory(
                    {
                        "observed": {"rows": 2},
                        "empty": {
                            "path": str(empty_path),
                            "rows": 0,
                            "sha256": manifest["artifact_sha256"],
                            "security_id": "security:wdc",
                            "ticker": "WDC",
                            "source_event_artifact_sha256": "empty-source-sha",
                        },
                    },
                    eligible_chunk_ids={"observed"},
                    source_collections=pd.DataFrame(
                        {
                            "chunk_id": ["observed", "empty"],
                            "security_id": ["security:wdc", "security:wdc"],
                            "ticker": ["WDC", "WDC"],
                            "status": ["observed", "observed_empty"],
                            "row_count": [2, 0],
                        }
                    ),
                    excluded_security_ids=set(),
                    source_inventory={
                        "empty": {"source_empty": True, "sha256": "empty-source-sha"}
                    },
                    sentiment_dir=sentiment_dir,
                    sentiment_request_sha256="sentiment-request",
                )

    def test_collection_source_ledger_hash_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish()
            manifest_path = fixture.collection / "_manifest.json"
            manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            manifest["source_collections_sha256"] = "0" * 64
            manifest_path.write_text(json.dumps(manifest), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "source-ledger identity"):
                fixture.build()

    def test_collection_audit_request_mismatch_fails(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish()
            audit = json.loads(fixture.audit_path.read_text(encoding="utf-8"))
            audit["request_sha256"] = "wrong-request"
            fixture.audit_path.write_text(json.dumps(audit), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "passed collection audit"):
                fixture.build()


class _Fixture:
    def __init__(self, root: Path) -> None:
        self.root = root
        self.collection = root / "collection"
        self.attribution = root / "attribution"
        self.sentiment = root / "sentiment"
        self.output = root / "lineage"
        self.decisions = root / "decisions.parquet"
        self.policy = root / "catalyst_lineage.toml"
        self.audit_path = root / "collection_audit.json"

    def publish(
        self,
        *,
        sentiment_event_ids: list[str] | None = None,
        relation_feature_available_at: pd.Timestamp | None = None,
    ) -> None:
        source_path = self.collection / "events" / "chunk-1.parquet"
        source_manifest = write_canonical_artifact(
            _source_events(),
            source_path,
            artifact_type="events",
            audit=_audit(2),
            production_ready=False,
        )
        source_sha256 = str(source_manifest["artifact_sha256"])
        source_collections_path = self.collection / "_source_collections.parquet"
        source_collections_manifest = write_canonical_artifact(
            _source_collections(),
            source_collections_path,
            artifact_type="source_collections",
            audit=_audit(1),
            production_ready=False,
        )
        self.collection.mkdir(parents=True, exist_ok=True)
        (self.collection / "_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "production_ready": False,
                    "request_sha256": "collection-request",
                    "source_collections_path": str(source_collections_path),
                    "source_collections_sha256": source_collections_manifest[
                        "artifact_sha256"
                    ],
                    "artifacts": [
                        {
                            "chunk_id": "chunk-1",
                            "security_id": "security:wdc",
                            "ticker": "WDC",
                            "path": str(source_path),
                            "sha256": source_sha256,
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        (self.collection / "_request.json").write_text(
            json.dumps(
                {
                    "request_sha256": "collection-request",
                    "work_units": [
                        {
                            "chunk_id": "chunk-1",
                            "security_id": "security:wdc",
                            "ticker": "WDC",
                            "start_utc": "2025-01-01T00:00:00+00:00",
                            "end_exclusive_utc": "2025-02-01T00:00:00+00:00",
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        self.audit_path.write_text(
            json.dumps(
                {
                    "passed": True,
                    "request_sha256": "collection-request",
                    "coverage_blindspot_security_ids": [],
                }
            ),
            encoding="utf-8",
        )

        relation_path = self.attribution / "relations" / "chunk-1.parquet"
        relation_manifest = write_canonical_artifact(
            _relations(
                relation_feature_available_at
                or pd.Timestamp("2025-01-02T14:00:00Z")
            ),
            relation_path,
            artifact_type="event_security_relations",
            audit=_audit(1),
            inputs={"source_event_artifact_sha256": source_sha256},
            production_ready=False,
        )
        self.attribution.mkdir(parents=True, exist_ok=True)
        (self.attribution / "_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "production_ready": False,
                    "excluded_security_ids": [],
                    "artifacts": [
                        {
                            "chunk_id": "chunk-1",
                            "source_event_sha256": source_sha256,
                            "path": str(relation_path),
                            "sha256": relation_manifest["artifact_sha256"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )

        selected_sentiment_ids = sentiment_event_ids or [
            "event-direct",
            "event-unrelated",
        ]
        sentiment_path = self.sentiment / "sentiment" / "chunk-1.parquet"
        sentiment_frame = _sentiments().loc[
            lambda frame: frame["event_id"].isin(selected_sentiment_ids)
        ].reset_index(drop=True)
        sentiment_manifest = write_canonical_artifact(
            sentiment_frame,
            sentiment_path,
            artifact_type="event_sentiment_research",
            audit=_audit(len(sentiment_frame)),
            inputs={"source_event_artifact_sha256": source_sha256},
            production_ready=False,
        )
        self.sentiment.mkdir(parents=True, exist_ok=True)
        (self.sentiment / "_manifest.json").write_text(
            json.dumps(
                {
                    "status": "complete",
                    "production_ready": False,
                    "request_sha256": "sentiment-request",
                    "total_rows": len(sentiment_frame),
                    "excluded_security_ids": [],
                    "artifacts": [
                        {
                            "chunk_id": "chunk-1",
                            "source_event_artifact_sha256": source_sha256,
                            "path": str(sentiment_path),
                            "sha256": sentiment_manifest["artifact_sha256"],
                        }
                    ],
                }
            ),
            encoding="utf-8",
        )
        write_canonical_artifact(
            _decisions(),
            self.decisions,
            artifact_type="decisions",
            audit=_audit(1),
            production_ready=False,
        )
        self.policy.write_text(_policy_text(), encoding="utf-8")

    def build(self) -> dict[str, object]:
        return build_catalyst_lineage(
            collection_dir=self.collection,
            collection_audit_path=self.audit_path,
            attribution_dir=self.attribution,
            sentiment_dir=self.sentiment,
            decisions_path=self.decisions,
            policy_path=self.policy,
            out_dir=self.output,
        )


def _source_events() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": ["event-direct", "event-unrelated"],
            "security_id": ["security:wdc", "security:wdc"],
            "ticker": ["WDC", "WDC"],
            "source_family": ["alpaca", "alpaca"],
            "published_at_utc": [
                pd.Timestamp("2025-01-02T14:00:00Z"),
                pd.Timestamp("2025-01-02T15:00:00Z"),
            ],
            "available_at_utc": [
                pd.Timestamp("2025-01-02T14:00:00Z"),
                pd.Timestamp("2025-01-02T15:00:00Z"),
            ],
            "feature_available_at_utc": [
                pd.Timestamp("2025-01-02T14:00:00Z"),
                pd.Timestamp("2025-01-02T15:00:00Z"),
            ],
        }
    )


def _relations(feature_available_at: pd.Timestamp) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "relation_id": ["relation-direct"],
            "event_id": ["event-direct"],
            "source_security_id": ["security:wdc"],
            "source_ticker": ["WDC"],
            "target_security_id": ["security:wdc"],
            "target_ticker": ["WDC"],
            "relation_channel": ["direct_issuer"],
            "relation_score": [0.99],
            "relation_basis": ["explicit_ticker"],
            "matched_business_labels": ["[]"],
            "matched_label_types": ["[]"],
            "matched_terms": ['["$wdc"]'],
            "event_feature_available_at_utc": [
                pd.Timestamp("2025-01-02T14:00:00Z")
            ],
            "identity_available_at_utc": [
                pd.Timestamp("2024-01-01T00:00:00Z")
            ],
            "label_available_at_utc": [pd.NaT],
            "feature_available_at_utc": [feature_available_at],
            "attribution_policy_version": [ATTRIBUTION_POLICY_VERSION],
            "attribution_policy_sha256": [ATTRIBUTION_POLICY_SHA256],
            "business_label_assignment_sha256": ["labels-sha"],
            "security_identity_registry_sha256": ["identity-sha"],
        }
    )


def _sentiments() -> pd.DataFrame:
    event_ids = ["event-direct", "event-unrelated"]
    times = [
        pd.Timestamp("2025-01-02T14:00:00Z"),
        pd.Timestamp("2025-01-02T15:00:00Z"),
    ]
    return pd.DataFrame(
        {
            "event_id": event_ids,
            "security_id": ["security:wdc", "security:wdc"],
            "ticker": ["WDC", "WDC"],
            "source_family": ["alpaca", "alpaca"],
            "published_at_utc": times,
            "event_available_at_utc": times,
            "research_feature_available_at_utc": [
                value + pd.Timedelta(minutes=5) for value in times
            ],
            "sentiment_label": ["positive", "negative"],
            "sentiment_confidence": [0.9, 0.8],
            "sentiment_numeric": [0.9, -0.8],
            "relevance": [1.0, 0.1],
            "relevance_basis": ["provider_tag+ticker", "provider_tag"],
            "sentiment_input_sha256": ["input-1", "input-2"],
            "sentiment_model": ["ProsusAI/finbert", "ProsusAI/finbert"],
            "sentiment_model_revision": ["revision", "revision"],
        }
    )


def _source_collections() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "collection_id": ["collection-1"],
            "chunk_id": ["chunk-1"],
            "security_id": ["security:wdc"],
            "ticker": ["WDC"],
            "source_family": ["alpaca"],
            "requested_start_utc": [pd.Timestamp("2025-01-01T00:00:00Z")],
            "requested_end_utc": [pd.Timestamp("2025-02-01T00:00:00Z")],
            "status": ["observed"],
            "row_count": [2],
        }
    )


def _decisions() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ticker": ["WDC"],
            "security_id": ["security:wdc"],
            "decision_time_utc": [pd.Timestamp("2025-01-02T21:00:00Z")],
            "prediction_cutoff_policy_id": ["swing-nightly"],
            "timeframe": ["1Day"],
            "bar_start_utc": [pd.Timestamp("2025-01-02T14:30:00Z")],
        }
    )


def _policy_text() -> str:
    return """
schema_version = "market_predictor.catalyst_lineage.v1"
production_ready = false
availability_policy = "provider_publication_proxy_plus_fixed_inference_latency"
training_eligible_channels = ["direct_issuer"]
research_only_channels = ["business_exposure", "sector_context"]
maximum_process_memory_gib = 4.0
memory_guard_headroom_gib = 0.75

[assignment_windows]
"2h" = "2h"
"1d" = "1D"
"3d" = "3D"

[feature_profiles.catalyst_only]
features = ["event_count_{window}", "sentiment_mean_{window}"]

[feature_profiles.technical_plus_catalyst]
features = ["event_count_{window}", "sentiment_mean_{window}"]
""".strip()


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
