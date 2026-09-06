from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path
from typing import cast
from unittest.mock import patch

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.catalysts.issuer_events.attribution import (
    ATTRIBUTION_POLICY_SHA256,
    ATTRIBUTION_POLICY_VERSION,
)
from market_predictor.core.errors import DataReadinessError
from market_predictor.swing.catalyst_lineage import (
    CATALYST_COVERAGE_COLUMNS,
    _reconcile_sentiment_inventory,
    _verify_assignment_semantics,
    _verify_coverage_semantics,
    _verify_feature_inventory,
    build_catalyst_lineage,
    verify_completed_catalyst_lineage,
)


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
            inventory = json.loads((fixture.output / "feature_inventory.json").read_text(encoding="utf-8"))
            self.assertIn("catalyst_only", inventory["profiles"])
            self.assertIn("technical_plus_catalyst", inventory["profiles"])
            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                fixture.build()

    def test_verifies_complete_bundle_with_bounded_semantic_projections(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish()
            result = fixture.build()

            with patch(
                "market_predictor.swing.catalyst_lineage.pd.read_parquet",
                wraps=pd.read_parquet,
            ) as reader:
                verified = verify_completed_catalyst_lineage(fixture.output)

            coverage_record = cast(dict[str, object], result["coverage"])
            self.assertEqual(verified.manifest, result)
            self.assertEqual(verified.request_sha256, result["request_sha256"])
            self.assertEqual(verified.coverage_sha256, coverage_record["sha256"])
            self.assertEqual(verified.lineage_sha256, result["lineage_sha256"])
            self.assertEqual(tuple(verified.coverage.columns), CATALYST_COVERAGE_COLUMNS)
            self.assertEqual(len(verified.coverage), 1)
            projections = [call.kwargs["columns"] for call in reader.call_args_list]
            self.assertEqual(reader.call_count, 4)
            self.assertEqual(projections.count([]), 1)
            self.assertIn(list(CATALYST_COVERAGE_COLUMNS), projections)

    def test_verifier_rejects_request_hash_mismatch(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            request_path = fixture.output / "_request.json"
            request = json.loads(request_path.read_text(encoding="utf-8"))
            request["policy_sha256"] = "0" * 64
            request_path.write_text(json.dumps(request), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "request hash binding"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_duplicate_json_keys(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            request_path = fixture.output / "_request.json"
            raw = request_path.read_text(encoding="utf-8").rstrip()
            request_path.write_text(
                raw[:-1] + ',"request_sha256":"' + "0" * 64 + '"}',
                encoding="utf-8",
            )

            with self.assertRaisesRegex(DataReadinessError, "JSON artifact is invalid"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_nonfinite_json(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            for name in ("_manifest.json", "_status.json"):
                path = fixture.output / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["memory"]["peak_working_set_gib"] = float("nan")
                path.write_text(json.dumps(payload, allow_nan=True), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "JSON artifact is invalid"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_feature_inventory_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            inventory_path = fixture.output / "feature_inventory.json"
            inventory = json.loads(inventory_path.read_text(encoding="utf-8"))
            inventory["event_artifact_count"] = 99
            inventory_path.write_text(json.dumps(inventory), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "feature inventory identity"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_coverage_artifact_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            with (fixture.output / "source_coverage.parquet").open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(DataReadinessError, "integrity check failed"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_feature_inventory_rejects_overlapping_channel_policy(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            inventory = json.loads(
                (fixture.output / "feature_inventory.json").read_text(encoding="utf-8")
            )
            manifest = json.loads(
                (fixture.output / "_manifest.json").read_text(encoding="utf-8")
            )
            inventory["training_eligible_channels"].append("business_exposure")

            with self.assertRaisesRegex(DataReadinessError, "channel policy differs"):
                _verify_feature_inventory(inventory, manifest=manifest)

    def test_coverage_semantics_reject_unknown_missingness_as_eligible(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            coverage, _ = load_canonical_artifact(
                fixture.output / "source_coverage.parquet",
                expected_type="catalyst_source_coverage",
                allow_research=True,
            )
            manifest = json.loads((fixture.output / "_manifest.json").read_text(encoding="utf-8"))
            coverage.loc[0, "missingness_known"] = False

            with self.assertRaisesRegex(DataReadinessError, "contradicts missingness"):
                _verify_coverage_semantics(coverage, manifest=manifest)

    def test_coverage_semantics_reconciles_manifest_state_counts(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            coverage, _ = load_canonical_artifact(
                fixture.output / "source_coverage.parquet",
                expected_type="catalyst_source_coverage",
                allow_research=True,
            )
            manifest = json.loads((fixture.output / "_manifest.json").read_text(encoding="utf-8"))
            manifest["coverage"]["states"] = {"observed_complete": 2}

            with self.assertRaisesRegex(DataReadinessError, "state counts"):
                _verify_coverage_semantics(coverage, manifest=manifest)

    def test_assignment_semantics_reject_future_information(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            assignments, _ = load_canonical_artifact(
                fixture.output / "assignments" / "chunk-1.parquet",
                expected_type="catalyst_event_assignments",
                allow_research=True,
            )
            events, _ = load_canonical_artifact(
                fixture.output / "events" / "chunk-1.parquet",
                expected_type="catalyst_events",
                allow_research=True,
            )
            assignments.loc[:, "decision_time_utc"] = pd.Timestamp("2025-01-02T13:00:00Z")

            with self.assertRaisesRegex(DataReadinessError, "causal window"):
                _verify_assignment_semantics(
                    assignments,
                    event_ids=set(events["event_id"].astype(str)),
                    expected_material_sha256="0" * 64,
                )

    def test_verifier_rejects_event_artifact_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            with (fixture.output / "events" / "chunk-1.parquet").open("ab") as handle:
                handle.write(b"tampered")

            with self.assertRaisesRegex(DataReadinessError, "integrity check failed"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_assignment_sidecar_identity_tamper(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            sidecar_path = fixture.output / "assignments" / "chunk-1.parquet.manifest.json"
            sidecar = json.loads(sidecar_path.read_text(encoding="utf-8"))
            sidecar["inputs"]["catalyst_events_sha256"] = "0" * 64
            sidecar_path.write_text(json.dumps(sidecar), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "sidecar identity"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_rejects_unexpected_artifact(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            (fixture.output / "unexpected.json").write_text("{}", encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "inventory does not match"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_verifier_recomputes_lineage_hash(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _completed_fixture(Path(temporary))
            for name in ("_manifest.json", "_status.json"):
                path = fixture.output / name
                payload = json.loads(path.read_text(encoding="utf-8"))
                payload["lineage_sha256"] = "0" * 64
                path.write_text(json.dumps(payload), encoding="utf-8")

            with self.assertRaisesRegex(DataReadinessError, "lineage hash does not verify"):
                verify_completed_catalyst_lineage(fixture.output)

    def test_missing_sentiment_fails_chunk_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish(sentiment_event_ids=["event-direct"])

            result = fixture.build()

            self.assertEqual(result["status"], "incomplete")
            failures = cast(dict[str, str], result["failed_chunks"])
            self.assertIn("sentiment event inventory mismatch", failures["chunk-1"])
            self.assertFalse((fixture.output / "_manifest.json").exists())

    def test_backdated_relation_fails_chunk_reconciliation(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            fixture = _Fixture(Path(temporary))
            fixture.publish(relation_feature_available_at=pd.Timestamp("2025-01-02T13:59:00Z"))

            result = fixture.build()

            self.assertEqual(result["status"], "incomplete")
            failures = cast(dict[str, str], result["failed_chunks"])
            self.assertIn("backdated availability", failures["chunk-1"])

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
            records: dict[str, dict[str, object]] = {
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
                source_inventory={"empty": {"source_empty": True, "sha256": "empty-source-sha"}},
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
                source_inventory={"extra": {"source_empty": True, "sha256": "empty-source-sha"}},
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
            cases: tuple[tuple[str, dict[str, object], str, set[str]], ...] = (
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
                    source_inventory={"empty": {"source_empty": True, "sha256": "empty-source-sha"}},
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


def _completed_fixture(root: Path) -> _Fixture:
    fixture = _Fixture(root)
    fixture.publish()
    fixture.build()
    return fixture


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
                    "source_collections_sha256": source_collections_manifest["artifact_sha256"],
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
            _relations(relation_feature_available_at or pd.Timestamp("2025-01-02T14:00:00Z")),
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
        sentiment_frame = _sentiments().loc[lambda frame: frame["event_id"].isin(selected_sentiment_ids)].reset_index(drop=True)
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
            "event_feature_available_at_utc": [pd.Timestamp("2025-01-02T14:00:00Z")],
            "identity_available_at_utc": [pd.Timestamp("2024-01-01T00:00:00Z")],
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
            "research_feature_available_at_utc": [value + pd.Timedelta(minutes=5) for value in times],
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
