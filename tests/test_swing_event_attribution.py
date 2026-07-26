from __future__ import annotations

import json
import unittest

import pandas as pd

from market_predictor.swing.event_attribution import (
    ATTRIBUTION_POLICY_SHA256,
    ATTRIBUTION_POLICY_VERSION,
    build_event_security_relations,
)
from market_predictor.v3.errors import DataReadinessError

_EVENT_TIME = pd.Timestamp("2026-01-20T15:00:00Z")


class SwingEventAttributionTests(unittest.TestCase):
    def test_lunr_oil_headline_has_no_false_issuer_association(self) -> None:
        events = _events(
            security_id="security:lunr",
            ticker="LUNR",
            title="Oil executives warn gasoline prices will get worse",
        )
        labels = _labels(
            _label(
                "security:lunr",
                "LUNR",
                "Intuitive Machines",
                "lunar_services",
                "offering",
                ["lunar services", "moon mission"],
                1,
            ),
            _label(
                "security:lunr",
                "LUNR",
                "Intuitive Machines",
                "space_economy",
                "end_market",
                ["space economy", "space stocks"],
                2,
            ),
        )

        relations = build_event_security_relations(events, labels)

        self.assertTrue(relations.empty)

    def test_wdc_is_direct_and_stx_is_storage_exposure(self) -> None:
        events = _events(
            security_id="security:wdc",
            ticker="WDC",
            title="Western Digital sees enterprise storage demand accelerate",
        )
        labels = _labels(
            _label(
                "security:wdc",
                "WDC",
                "Western Digital",
                "data_storage",
                "offering",
                ["enterprise storage", "data storage"],
                1,
            ),
            _label(
                "security:stx",
                "STX",
                "Seagate Technology",
                "data_storage",
                "offering",
                ["enterprise storage", "data storage"],
                1,
            ),
        )

        relations = build_event_security_relations(events, labels)
        by_target = relations.set_index("target_security_id")

        self.assertEqual(
            by_target.loc["security:wdc", "relation_channel"],
            "direct_issuer",
        )
        self.assertIn(
            "company_text",
            by_target.loc["security:wdc", "relation_basis"],
        )
        self.assertEqual(
            by_target.loc["security:stx", "relation_channel"],
            "business_exposure",
        )
        self.assertEqual(
            json.loads(
                by_target.loc[
                    "security:stx",
                    "matched_business_labels",
                ]
            ),
            ["data_storage"],
        )

    def test_mu_memory_and_data_center_terms_establish_exposure(self) -> None:
        events = _events(
            security_id="security:market",
            ticker="MARKET",
            title="AI data centers drive stronger DRAM memory demand",
        )
        labels = _labels(
            _label(
                "security:mu",
                "MU",
                "Micron Technology",
                "memory_semiconductors",
                "offering",
                ["dram", "memory"],
                1,
            ),
            _label(
                "security:mu",
                "MU",
                "Micron Technology",
                "data_center_supplier",
                "end_market",
                ["data center", "data centers"],
                2,
            ),
        )

        relations = build_event_security_relations(events, labels)

        self.assertEqual(len(relations), 1)
        self.assertEqual(relations.loc[0, "target_security_id"], "security:mu")
        self.assertEqual(
            relations.loc[0, "relation_channel"],
            "business_exposure",
        )
        self.assertEqual(
            json.loads(relations.loc[0, "matched_label_types"]),
            ["end_market", "offering"],
        )

    def test_end_market_only_is_sector_context(self) -> None:
        events = _events(
            security_id="security:market",
            ticker="MARKET",
            title="Data center spending accelerates across the technology sector",
        )
        labels = _labels(
            _label(
                "security:mu",
                "MU",
                "Micron Technology",
                "memory_semiconductors",
                "offering",
                ["dram", "memory chip"],
                1,
            ),
            _label(
                "security:mu",
                "MU",
                "Micron Technology",
                "data_center_supplier",
                "end_market",
                ["data center", "data centers"],
                2,
            ),
        )

        relations = build_event_security_relations(events, labels)

        self.assertEqual(relations["relation_channel"].tolist(), ["sector_context"])
        self.assertNotIn(
            "direct_issuer",
            set(relations["relation_channel"]),
        )
        self.assertNotIn(
            "business_exposure",
            set(relations["relation_channel"]),
        )

    def test_source_tag_without_identity_uses_relation_channel_not_direct(
        self,
    ) -> None:
        context_relations = build_event_security_relations(
            _events(
                security_id="security:mu",
                ticker="MU",
                title="Semiconductors demand rises across the sector",
            ),
            _labels(
                _label(
                    "security:mu",
                    "MU",
                    "Micron Technology",
                    "semiconductor_industry",
                    "offering",
                    ["semiconductors"],
                    1,
                    relation_use="context",
                )
            ),
        )
        exposure_relations = build_event_security_relations(
            _events(
                security_id="security:mu",
                ticker="MU",
                title="DRAM memory demand rises",
            ),
            _labels(
                _label(
                    "security:mu",
                    "MU",
                    "Micron Technology",
                    "memory_semiconductors",
                    "offering",
                    ["dram memory"],
                    1,
                )
            ),
        )

        self.assertEqual(
            context_relations["relation_channel"].tolist(),
            ["sector_context"],
        )
        self.assertEqual(
            exposure_relations["relation_channel"].tolist(),
            ["business_exposure"],
        )

    def test_future_profile_tag_is_excluded(self) -> None:
        events = _events(
            security_id="security:market",
            ticker="MARKET",
            title="Enterprise storage demand accelerates",
        )
        labels = _labels(
            _label(
                "security:stx",
                "STX",
                "Seagate Technology",
                "data_storage",
                "offering",
                ["enterprise storage"],
                1,
                available_at=pd.Timestamp("2026-02-01T00:00:00Z"),
            )
        )

        relations = build_event_security_relations(events, labels)

        self.assertTrue(relations.empty)

    def test_ticker_reuse_does_not_redirect_direct_relation(self) -> None:
        events = _events(
            security_id="security:old_xyz",
            ticker="XYZ",
            title="XYZ reports quarterly earnings",
        )
        labels = _labels(
            _label(
                "security:new_xyz",
                "XYZ",
                "Xylophone Yield Systems",
                "data_storage",
                "offering",
                ["enterprise storage"],
                1,
            )
        )

        relations = build_event_security_relations(events, labels)

        self.assertEqual(
            relations["target_security_id"].tolist(),
            ["security:old_xyz"],
        )
        self.assertEqual(
            relations["relation_channel"].tolist(),
            ["direct_issuer"],
        )
        self.assertNotIn(
            "security:new_xyz",
            set(relations["target_security_id"]),
        )

    def test_half_open_interval_and_feature_availability_are_enforced(
        self,
    ) -> None:
        events = pd.concat(
            [
                _events(
                    event_id="event_before_end_0001",
                    security_id="security:market",
                    ticker="MARKET",
                    title="Enterprise storage demand accelerates",
                    feature_available_at=_EVENT_TIME - pd.Timedelta(seconds=1),
                ),
                _events(
                    event_id="event_at_end_000002",
                    security_id="security:market",
                    ticker="MARKET",
                    title="Enterprise storage demand accelerates",
                ),
            ],
            ignore_index=True,
        )
        labels = _labels(
            _label(
                "security:stx",
                "STX",
                "Seagate Technology",
                "data_storage",
                "offering",
                ["enterprise storage"],
                1,
                available_at=_EVENT_TIME - pd.Timedelta(days=2),
                effective_to=_EVENT_TIME,
            )
        )

        relations = build_event_security_relations(events, labels)

        self.assertEqual(relations["event_id"].tolist(), ["event_before_end_0001"])
        self.assertEqual(
            relations.loc[0, "feature_available_at_utc"],
            _EVENT_TIME - pd.Timedelta(seconds=1),
        )

    def test_policy_and_relation_hashes_are_deterministic(self) -> None:
        events = _events(
            security_id="security:wdc",
            ticker="WDC",
            title="Western Digital sees enterprise storage demand accelerate",
        )
        labels = _labels(
            _label(
                "security:wdc",
                "WDC",
                "Western Digital",
                "data_storage",
                "offering",
                ["enterprise storage"],
                1,
            ),
            _label(
                "security:stx",
                "STX",
                "Seagate Technology",
                "data_storage",
                "offering",
                ["enterprise storage"],
                1,
            ),
        )

        first = build_event_security_relations(events, labels)
        second = build_event_security_relations(
            events.sample(frac=1, random_state=3),
            labels.sample(frac=1, random_state=5),
        )

        self.assertEqual(first["relation_id"].tolist(), second["relation_id"].tolist())
        self.assertTrue(first["attribution_policy_version"].eq(ATTRIBUTION_POLICY_VERSION).all())
        self.assertTrue(first["attribution_policy_sha256"].eq(ATTRIBUTION_POLICY_SHA256).all())
        self.assertEqual(
            first["business_label_assignment_sha256"].tolist(),
            second["business_label_assignment_sha256"].tolist(),
        )

    def test_more_than_three_active_tags_is_rejected(self) -> None:
        labels = _labels(
            *[
                _label(
                    "security:mu",
                    "MU",
                    "Micron Technology",
                    f"tag_{rank}",
                    "offering",
                    [f"term {rank}"],
                    rank if rank <= 3 else 3,
                )
                for rank in range(1, 5)
            ]
        )

        with self.assertRaisesRegex(DataReadinessError, "more than three"):
            build_event_security_relations(
                _events(
                    security_id="security:market",
                    ticker="MARKET",
                    title="Term 1",
                ),
                labels,
            )


def _events(
    *,
    event_id: str = "canonical_event_0001",
    security_id: str,
    ticker: str,
    title: str,
    feature_available_at: pd.Timestamp = _EVENT_TIME,
) -> pd.DataFrame:
    return pd.DataFrame(
        {
            "event_id": [event_id],
            "security_id": [security_id],
            "ticker": [ticker],
            "feature_available_at_utc": [feature_available_at],
            "title": [title],
            "summary": [""],
            "text": [""],
        }
    )


def _labels(*records: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame.from_records(records)


def _label(
    security_id: str,
    ticker: str,
    company: str,
    business_tag: str,
    label_type: str,
    match_terms: list[str],
    tag_rank: int,
    *,
    available_at: pd.Timestamp = pd.Timestamp("2025-01-01T00:00:00Z"),
    effective_to: pd.Timestamp | None = None,
    relation_use: str | None = None,
) -> dict[str, object]:
    return {
        "security_id": security_id,
        "ticker": ticker,
        "company": company,
        "business_tag": business_tag,
        "label_type": label_type,
        "match_terms": match_terms,
        "tag_rank": tag_rank,
        "confidence": 0.9,
        "relation_use": relation_use or ("context" if label_type == "end_market" else "exposure"),
        "effective_from_utc": pd.Timestamp("2025-01-01T00:00:00Z"),
        "effective_to_utc": effective_to,
        "available_at_utc": available_at,
    }


if __name__ == "__main__":
    unittest.main()
