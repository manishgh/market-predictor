from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditCheck,
    CanonicalAuditReport,
)
from market_predictor.canonical.store import write_canonical_artifact
from market_predictor.catalysts.issuer_events.attribution import (
    build_event_security_relations,
)
from market_predictor.swing.security_label_artifact import (
    build_security_label_artifact,
)

_START = pd.Timestamp("2021-01-04T05:00:00Z")
_OBSERVED = pd.Timestamp("2026-07-26T06:00:00Z")


class SwingSecurityLabelArtifactTests(unittest.TestCase):
    def test_builds_context_history_and_prospective_profile_exposure(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = root / "memberships.parquet"
            profiles = root / "profiles.parquet"
            training = root / "training.parquet"
            universe = root / "universe.parquet"
            _write_inputs(
                memberships=memberships,
                profiles=profiles,
                training=training,
                universe=universe,
            )

            artifact = build_security_label_artifact(
                memberships_path=memberships,
                universe_path=universe,
                profiles_path=profiles,
                training_dataset_path=training,
                policy_path=(Path(__file__).parents[1] / "configs" / "security_business_labels.toml"),
            )

            self.assertEqual(
                artifact.summary["training_security_ids"],
                3,
            )
            self.assertEqual(
                artifact.summary["historical_insufficient_security_ids"],
                1,
            )
            wdc = artifact.assignments.loc[artifact.assignments["ticker"].eq("WDC")]
            historical = wdc.loc[wdc["effective_from_utc"].eq(_START)]
            current_start = wdc.loc[
                wdc["effective_from_utc"].gt(_OBSERVED),
                "effective_from_utc",
            ].min()
            current = wdc.loc[
                wdc["effective_from_utc"].eq(current_start)
            ]
            resumed = wdc.loc[
                wdc["effective_from_utc"].gt(current_start)
            ]
            self.assertEqual(
                historical["business_tag"].tolist(),
                ["offering.hardware.storage"],
            )
            self.assertEqual(
                historical["relation_use"].tolist(),
                ["context"],
            )
            self.assertEqual(
                historical["effective_to_utc"].tolist(),
                [current_start],
            )
            self.assertEqual(
                set(current["business_tag"]),
                {
                    "offering.hardware.storage",
                    "end_market.data_center",
                },
            )
            self.assertEqual(
                current.set_index("business_tag").loc[
                    "offering.hardware.storage",
                    "relation_use",
                ],
                "exposure",
            )
            self.assertTrue(current["available_at_utc"].ge(current_start).all())
            self.assertEqual(
                resumed["business_tag"].tolist(),
                ["offering.hardware.storage"],
            )
            self.assertEqual(
                resumed["relation_use"].tolist(),
                ["context"],
            )
            unknown = artifact.coverage.loc[artifact.coverage["ticker"].eq("UNK")].iloc[0]
            self.assertEqual(
                unknown["historical_disposition"],
                "insufficient_historical_evidence",
            )
            self.assertEqual(unknown["historical_label_count"], 0)
            self.assertFalse(bool(artifact.coverage["historical_exposure_training_eligible"].any()))
            relations = build_event_security_relations(
                pd.DataFrame(
                    {
                        "event_id": ["storage-demand-1"],
                        "security_id": ["security:market"],
                        "ticker": ["MARKET"],
                        "feature_available_at_utc": [
                            current_start + pd.Timedelta(seconds=1)
                        ],
                        "title": [
                            "Hard disk drives demand accelerates"
                        ],
                        "summary": [""],
                        "text": [""],
                    }
                ),
                artifact.assignments,
            )
            self.assertEqual(
                relations.loc[
                    relations["target_security_id"].eq(
                        "security:wdc"
                    ),
                    "relation_channel",
                ].tolist(),
                ["business_exposure"],
            )


def _write_inputs(
    *,
    memberships: Path,
    profiles: Path,
    training: Path,
    universe: Path,
) -> None:
    identity_rows = [
        (
            "security:wdc",
            "WDC",
            "Western Digital",
            "Information Technology",
            "Technology Hardware, Storage & Peripherals",
        ),
        (
            "security:mu",
            "MU",
            "Micron Technology",
            "Information Technology",
            "Semiconductors",
        ),
        (
            "security:unknown",
            "UNK",
            "Unknown Industries",
            "Industrials",
            "unknown",
        ),
    ]
    membership_frame = pd.DataFrame(
        {
            "security_id": [row[0] for row in identity_rows],
            "ticker": [row[1] for row in identity_rows],
            "sector": [row[3] for row in identity_rows],
            "industry": [row[4] for row in identity_rows],
            "effective_from_utc": [_START] * 3,
            "effective_to_utc": [pd.NaT] * 3,
            "available_at_utc": [_START] * 3,
            "availability_policy": ["provider_publication_proxy"] * 3,
        }
    )
    universe_frame = pd.DataFrame(
        {
            "security_id": [row[0] for row in identity_rows],
            "ticker": [row[1] for row in identity_rows],
            "company": [row[2] for row in identity_rows],
            "effective_from_utc": [_START] * 3,
            "effective_to_utc": [pd.NaT] * 3,
        }
    )
    profile_frame = pd.DataFrame(
        {
            "security_id": ["security:wdc", "security:mu"],
            "ticker": ["WDC", "MU"],
            "long_description": [
                ("Makes hard disk drives and data storage devices for data center customers."),
                ("Makes dynamic random access memory, NAND flash, and high bandwidth memory for data center systems."),
            ],
            "source_document_id": [
                "external:profile:wdc",
                "external:profile:mu",
            ],
            "source_content_sha256": ["a" * 64, "b" * 64],
            "observed_at_utc": [_OBSERVED, _OBSERVED],
            "available_at_utc": [_OBSERVED, _OBSERVED],
            "knowledge_scope": [
                "current_inference_only",
                "current_inference_only",
            ],
        }
    )
    training_frame = pd.DataFrame(
        {
            "security_id": [row[0] for row in identity_rows],
            "ticker": [row[1] for row in identity_rows],
            "label_eligible": [True, True, True],
        }
    )
    write_canonical_artifact(
        membership_frame,
        memberships,
        artifact_type="memberships",
        audit=_audit(len(membership_frame)),
        production_ready=False,
    )
    write_canonical_artifact(
        profile_frame,
        profiles,
        artifact_type="security_profiles_current",
        audit=_audit(len(profile_frame)),
        production_ready=False,
    )
    write_canonical_artifact(
        training_frame,
        training,
        artifact_type="swing_dataset",
        audit=_audit(len(training_frame)),
        production_ready=False,
    )
    universe_frame.to_parquet(universe, index=False)


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
