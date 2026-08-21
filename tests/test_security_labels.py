from __future__ import annotations

import hashlib
import unittest
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import ClassVar

from market_predictor.security_labels import (
    CurrentProfileEvidence,
    MembershipEvidence,
    SecurityBusinessLabelSet,
    SecurityLabelPolicy,
    assign_security_business_labels,
    load_security_label_policy,
    profile_terms_from_text,
    validate_business_label_ids,
)
from market_predictor.core.errors import DataReadinessError

ROOT = Path(__file__).resolve().parents[1]
POLICY_PATH = ROOT / "configs" / "security_business_labels.toml"
AS_OF = datetime(2026, 7, 26, tzinfo=UTC)
MEMBERSHIP_START = datetime(2021, 7, 9, tzinfo=UTC)
MEMBERSHIP_AVAILABLE = datetime(2021, 7, 10, tzinfo=UTC)
PROFILE_OBSERVED = datetime(2026, 7, 20, 14, 0, tzinfo=UTC)


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def _membership(
    security_id: str,
    industry: str,
    *,
    sector: str = "Information Technology",
    start: datetime = MEMBERSHIP_START,
    end: datetime | None = None,
) -> MembershipEvidence:
    return MembershipEvidence(
        security_id=security_id,
        sector=sector,
        industry=industry,
        effective_from_utc=start,
        effective_to_utc=end,
        available_at_utc=MEMBERSHIP_AVAILABLE,
        availability_policy="provider_publication_proxy",
        source_uri=f"artifact://membership/{security_id}",
        source_published_at_utc=MEMBERSHIP_AVAILABLE,
        source_content_sha256=_sha256(f"membership:{security_id}:{industry}:{start.isoformat()}"),
    )


def _profile(
    security_id: str,
    *terms: str,
    observed: datetime = PROFILE_OBSERVED,
    membership_label_ids: tuple[str, ...] | None = None,
) -> CurrentProfileEvidence:
    if membership_label_ids is None:
        membership_label_ids = ("offering.semiconductor.general",) if security_id == "security:mu" else ("offering.hardware.storage",)
    return CurrentProfileEvidence(
        security_id=security_id,
        profile_terms=tuple(terms),
        membership_label_ids=membership_label_ids,
        observed_at_utc=observed,
        source_uri=f"artifact://profile/{security_id}",
        source_published_at_utc=observed,
        source_content_sha256=_sha256(f"profile:{security_id}"),
    )


def _result(
    results: tuple[SecurityBusinessLabelSet, ...],
    security_id: str,
    scope: str,
) -> SecurityBusinessLabelSet:
    matches = [item for item in results if item.security_id == security_id and item.knowledge_scope == scope]
    if len(matches) != 1:
        raise AssertionError(f"expected one result for {security_id}/{scope}, found {len(matches)}")
    return matches[0]


class SecurityBusinessLabelTests(unittest.TestCase):
    policy: ClassVar[SecurityLabelPolicy]

    @classmethod
    def setUpClass(cls) -> None:
        cls.policy = load_security_label_policy(POLICY_PATH)

    def test_wdc_stx_and_mu_receive_only_exact_evidence_labels(self) -> None:
        memberships = (
            _membership("security:wdc", "Technology Hardware, Storage & Peripherals"),
            _membership("security:stx", "Technology Hardware, Storage & Peripherals"),
            _membership("security:mu", "Semiconductors"),
        )
        profiles = (
            _profile("security:wdc", "hard disk drives", "data center", "storage demand"),
            _profile("security:stx", "storage hardware", "hyperscale data center", "enterprise storage demand"),
            _profile("security:mu", "memory semiconductors", "data center", "memory pricing"),
        )

        results = assign_security_business_labels(
            self.policy,
            memberships,
            profiles,
            as_of_utc=AS_OF,
        )

        for security_id in ("security:wdc", "security:stx"):
            historical = _result(results, security_id, "historical_research_proxy")
            self.assertEqual(
                [item.label_id for item in historical.assignments],
                ["offering.hardware.storage"],
            )
            self.assertEqual(historical.disposition, "assigned")
        mu_historical = _result(results, "security:mu", "historical_research_proxy")
        self.assertEqual(
            [item.label_id for item in mu_historical.assignments],
            ["offering.semiconductor.general"],
        )

        self.assertEqual(
            [item.label_id for item in _result(results, "security:wdc", "current_inference_only").assignments],
            [
                "offering.hardware.storage",
                "end_market.data_center",
                "driver.storage_demand",
            ],
        )
        self.assertEqual(
            [item.label_id for item in _result(results, "security:stx", "current_inference_only").assignments],
            [
                "offering.hardware.storage",
                "end_market.data_center",
                "driver.storage_demand",
            ],
        )
        self.assertEqual(
            [item.label_id for item in _result(results, "security:mu", "current_inference_only").assignments],
            [
                "offering.semiconductor.memory",
                "end_market.data_center",
                "driver.memory_pricing",
            ],
        )

    def test_unknown_industry_has_explicit_insufficient_evidence(self) -> None:
        results = assign_security_business_labels(
            self.policy,
            (_membership("security:unknown", "Unknown"),),
            as_of_utc=AS_OF,
        )

        result = results[0]
        self.assertEqual(result.disposition, "insufficient_evidence")
        self.assertEqual(result.assignments, ())
        self.assertRegex(result.assignment_set_sha256, r"^[0-9a-f]{64}$")

    def test_unconfigured_known_industry_gets_exact_industry_fallback(self) -> None:
        results = assign_security_business_labels(
            self.policy,
            (
                _membership(
                    "security:restaurant",
                    "Restaurants",
                    sector="Consumer Discretionary",
                ),
            ),
            as_of_utc=AS_OF,
        )

        self.assertEqual(
            [item.label_id for item in results[0].assignments],
            ["offering.industry.restaurants"],
        )

    def test_profile_text_extracts_only_controlled_exact_phrases(self) -> None:
        terms = profile_terms_from_text(
            self.policy,
            ("The company sells hard disk drive products and data center platforms. Its name also contains unrelated words."),
        )

        self.assertEqual(terms, ("data center", "hard disk drive"))

    def test_data_center_is_not_inferred_without_exact_profile_evidence(self) -> None:
        results = assign_security_business_labels(
            self.policy,
            (_membership("security:wdc", "Technology Hardware, Storage & Peripherals"),),
            (_profile("security:wdc", "hard disk drives", "enterprise customers"),),
            as_of_utc=AS_OF,
        )

        current = _result(results, "security:wdc", "current_inference_only")
        labels = {item.label_id for item in current.assignments}
        self.assertEqual(labels, {"offering.hardware.storage"})
        self.assertNotIn("end_market.data_center", labels)

    def test_current_profile_labels_are_unavailable_before_observation(self) -> None:
        results = assign_security_business_labels(
            self.policy,
            (),
            (_profile("security:mu", "memory semiconductors"),),
            as_of_utc=AS_OF,
        )

        result = results[0]
        self.assertEqual(result.knowledge_scope, "current_inference_only")
        self.assertEqual(result.available_at_utc, PROFILE_OBSERVED)
        self.assertEqual(result.effective_from_utc, PROFILE_OBSERVED)
        self.assertFalse(result.is_available_at(datetime(2025, 7, 20, tzinfo=UTC)))
        self.assertTrue(result.is_available_at(PROFILE_OBSERVED))
        self.assertFalse(
            result.is_available_at(
                PROFILE_OBSERVED + timedelta(days=91)
            )
        )

    def test_assignment_hashes_are_deterministic_across_profile_term_order(self) -> None:
        first = assign_security_business_labels(
            self.policy,
            (),
            (_profile("security:mu", "memory semiconductors", "data center", "memory pricing"),),
            as_of_utc=AS_OF,
        )[0]
        second = assign_security_business_labels(
            load_security_label_policy(POLICY_PATH),
            (),
            (_profile("security:mu", "memory pricing", "data center", "memory semiconductors"),),
            as_of_utc=AS_OF,
        )[0]

        self.assertEqual(first.evidence_sha256, second.evidence_sha256)
        self.assertEqual(first.assignment_set_sha256, second.assignment_set_sha256)
        self.assertEqual(
            [item.assignment_sha256 for item in first.assignments],
            [item.assignment_sha256 for item in second.assignments],
        )
        self.assertEqual(first.taxonomy_sha256, second.taxonomy_sha256)

    def test_unknown_label_is_rejected(self) -> None:
        with self.assertRaisesRegex(DataReadinessError, "unknown security business labels"):
            validate_business_label_ids(self.policy, ("offering.not_registered",))

    def test_incompatible_profile_phrases_do_not_consume_label_slots(
        self,
    ) -> None:
        profile = _profile(
            "security:overloaded",
            "hard disk drives",
            "data center",
            "storage demand",
            "memory pricing",
        )

        result = assign_security_business_labels(
            self.policy,
            (),
            (profile,),
            as_of_utc=AS_OF,
        )[0]

        self.assertEqual(
            [item.label_id for item in result.assignments],
            [
                "offering.hardware.storage",
                "end_market.data_center",
                "driver.storage_demand",
            ],
        )

    def test_customer_or_investment_mentions_do_not_become_offerings(
        self,
    ) -> None:
        profiles = (
            _profile(
                "security:kkr",
                "semiconductors",
                membership_label_ids=("offering.industry.asset_management_custody_banks",),
            ),
            _profile(
                "security:amd",
                "public cloud",
                membership_label_ids=("offering.semiconductor.general",),
            ),
            _profile(
                "security:storage",
                "public cloud",
                membership_label_ids=("offering.hardware.storage",),
            ),
        )

        results = assign_security_business_labels(
            self.policy,
            (),
            profiles,
            as_of_utc=AS_OF,
        )

        self.assertTrue(all(result.disposition == "insufficient_evidence" for result in results))

    def test_overlapping_membership_intervals_are_rejected_but_touching_are_valid(self) -> None:
        boundary = datetime(2024, 1, 1, tzinfo=UTC)
        touching = (
            _membership(
                "security:wdc",
                "Technology Hardware, Storage & Peripherals",
                end=boundary,
            ),
            _membership(
                "security:wdc",
                "Technology Hardware, Storage & Peripherals",
                start=boundary,
            ),
        )
        self.assertEqual(
            len(assign_security_business_labels(self.policy, touching, as_of_utc=AS_OF)),
            2,
        )

        overlapping = (
            touching[0],
            _membership(
                "security:wdc",
                "Technology Hardware, Storage & Peripherals",
                start=datetime(2023, 12, 31, tzinfo=UTC),
            ),
        )
        with self.assertRaisesRegex(DataReadinessError, "intervals overlap"):
            assign_security_business_labels(
                self.policy,
                overlapping,
                as_of_utc=AS_OF,
            )

    def test_future_and_invalid_timestamps_are_rejected(self) -> None:
        with self.assertRaisesRegex(DataReadinessError, "future timestamp"):
            assign_security_business_labels(
                self.policy,
                (),
                (_profile("security:mu", "memory semiconductors", observed=datetime(2027, 1, 1, tzinfo=UTC)),),
                as_of_utc=AS_OF,
            )

        invalid = _membership(
            "security:wdc",
            "Technology Hardware, Storage & Peripherals",
            start=datetime(2025, 1, 1, tzinfo=UTC),
            end=datetime(2024, 1, 1, tzinfo=UTC),
        )
        with self.assertRaisesRegex(DataReadinessError, "must be later"):
            assign_security_business_labels(
                self.policy,
                (invalid,),
                as_of_utc=AS_OF,
            )


if __name__ == "__main__":
    unittest.main()
