from __future__ import annotations

import tempfile
import unittest
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from market_predictor.canonical.audits import (
    CanonicalAuditReport,
    audit_universe_memberships,
)
from market_predictor.canonical.normalize import (
    canonicalize_universe_memberships,
)
from market_predictor.canonical.store import (
    load_canonical_artifact,
    write_canonical_artifact,
)
from market_predictor.swing.profile_collection import (
    collect_current_security_profiles,
)
from market_predictor.v3.errors import DataReadinessError


class SwingProfileCollectionTests(unittest.TestCase):
    def test_rejects_provider_truncating_batch_size(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            memberships = _memberships(Path(temporary) / "memberships.parquet")
            with self.assertRaisesRegex(ValueError, "silently truncates"):
                collect_current_security_profiles(
                    memberships_path=memberships,
                    out_dir=Path(temporary) / "profiles",
                    fetch_batch=lambda symbols: {"data": []},
                    batch_size=5,
                )

    def test_collects_current_profiles_and_disposes_missing_and_historical(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")
            calls: list[list[str]] = []

            def fetch(symbols: list[str]) -> dict[str, object]:
                calls.append(symbols)
                return {
                    "data": [
                        _profile(
                            "MU",
                            "Micron makes DRAM, NAND, and data center memory.",
                        )
                    ]
                }

            result = collect_current_security_profiles(
                memberships_path=memberships,
                out_dir=root / "profiles",
                fetch_batch=fetch,
                batch_size=2,
                now=lambda: datetime(2026, 7, 26, tzinfo=UTC),
            )

            self.assertEqual(result.status, "complete")
            self.assertEqual(calls, [["MU", "WDC"]])
            self.assertEqual(result.observed_profiles, 1)
            profiles, manifest = load_canonical_artifact(
                root / "profiles" / "profiles.parquet",
                expected_type="security_profiles_current",
                allow_research=True,
            )
            self.assertEqual(profiles.loc[0, "ticker"], "MU")
            self.assertEqual(
                profiles.loc[0, "knowledge_scope"],
                "current_inference_only",
            )
            self.assertEqual(
                profiles.loc[0, "available_at_utc"],
                pd.Timestamp("2026-07-26T00:00:00Z"),
            )
            self.assertFalse(manifest["production_ready"])
            coverage, _ = load_canonical_artifact(
                root / "profiles" / "coverage.parquet",
                expected_type="security_profile_coverage",
                allow_research=True,
            )
            dispositions = dict(
                zip(
                    coverage["ticker"],
                    coverage["disposition"],
                    strict=True,
                )
            )
            self.assertEqual(
                dispositions,
                {
                    "OLD": "not_current_at_collection",
                    "MU": "observed_current_profile",
                    "WDC": "provider_missing_current_profile",
                },
            )
            self.assertFalse(coverage["profile_eligible_for_historical_training"].any())
            with self.assertRaisesRegex(DataReadinessError, "immutable"):
                collect_current_security_profiles(
                    memberships_path=memberships,
                    out_dir=root / "profiles",
                    fetch_batch=fetch,
                    now=lambda: datetime(2026, 7, 26, tzinfo=UTC),
                )

    def test_failed_batch_resumes_from_verified_raw_page(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")
            calls = 0

            def first_fetch(symbols: list[str]) -> dict[str, object]:
                nonlocal calls
                calls += 1
                if "WDC" in symbols:
                    raise RuntimeError("temporary failure")
                return {"data": [_profile("MU", "Memory products")]}

            first = collect_current_security_profiles(
                memberships_path=memberships,
                out_dir=root / "profiles",
                fetch_batch=first_fetch,
                batch_size=1,
                now=lambda: datetime(2026, 7, 26, tzinfo=UTC),
            )
            self.assertEqual(first.status, "incomplete")
            self.assertEqual(calls, 2)

            resumed_calls: list[list[str]] = []

            def resumed_fetch(symbols: list[str]) -> dict[str, object]:
                resumed_calls.append(symbols)
                return {"data": [_profile("WDC", "Data storage drives")]}

            resumed = collect_current_security_profiles(
                memberships_path=memberships,
                out_dir=root / "profiles",
                fetch_batch=resumed_fetch,
                batch_size=1,
                now=lambda: datetime(2026, 7, 26, tzinfo=UTC),
            )
            self.assertEqual(resumed.status, "complete")
            self.assertEqual(resumed_calls, [["WDC"]])
            self.assertEqual(resumed.observed_profiles, 2)

    def test_profile_availability_is_response_received_time(self) -> None:
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            memberships = _memberships(root / "memberships.parquet")
            times = iter(
                (
                    datetime(2026, 7, 26, 12, 0, 0, tzinfo=UTC),
                    datetime(2026, 7, 26, 12, 0, 7, tzinfo=UTC),
                    datetime(2026, 7, 26, 12, 0, 8, tzinfo=UTC),
                )
            )

            collect_current_security_profiles(
                memberships_path=memberships,
                out_dir=root / "profiles",
                fetch_batch=lambda symbols: {"data": [_profile(symbol, "Memory products") for symbol in symbols]},
                batch_size=2,
                now=lambda: next(times),
            )

            profiles, _ = load_canonical_artifact(
                root / "profiles" / "profiles.parquet",
                expected_type="security_profiles_current",
                allow_research=True,
            )
            self.assertTrue(profiles["available_at_utc"].eq(pd.Timestamp("2026-07-26T12:00:07Z")).all())


def _profile(ticker: str, description: str) -> dict[str, object]:
    return {
        "id": ticker,
        "tickerId": 123,
        "attributes": {
            "longDesc": description,
            "sectorname": "Information Technology",
            "primaryname": "Semiconductors",
            "companyName": f"{ticker} Company",
        },
    }


def _memberships(path: Path) -> Path:
    raw = pd.DataFrame(
        {
            "ticker": ["MU", "WDC", "OLD"],
            "security_id": ["security:mu", "security:wdc", "security:old"],
            "effective_from_utc": [
                pd.Timestamp("2020-01-01T00:00:00Z"),
                pd.Timestamp("2020-01-01T00:00:00Z"),
                pd.Timestamp("2020-01-01T00:00:00Z"),
            ],
            "effective_to_utc": [
                pd.NaT,
                pd.NaT,
                pd.Timestamp("2025-01-01T00:00:00Z"),
            ],
            "available_at_utc": [
                pd.Timestamp("2020-01-01T00:00:00Z"),
                pd.Timestamp("2020-01-01T00:00:00Z"),
                pd.Timestamp("2020-01-01T00:00:00Z"),
            ],
            "sector": [
                "Information Technology",
                "Information Technology",
                "Information Technology",
            ],
            "industry": [
                "Semiconductors",
                "Technology Hardware, Storage & Peripherals",
                "Unknown",
            ],
            "market_cap_bucket": ["large", "large", "large"],
            "liquidity_bucket": ["high", "high", "high"],
            "primary_benchmark": ["XLK", "XLK", "XLK"],
            "universe_snapshot_id": ["test", "test", "test"],
            "source": ["test", "test", "test"],
            "availability_policy": [
                "observed",
                "observed",
                "observed",
            ],
        }
    )
    canonical = canonicalize_universe_memberships(raw)
    audit = CanonicalAuditReport(
        checks=audit_universe_memberships(
            canonical,
            require_observed=False,
        )
    )
    write_canonical_artifact(
        canonical,
        path,
        artifact_type="memberships",
        audit=audit,
        production_ready=False,
    )
    return path


if __name__ == "__main__":
    unittest.main()
