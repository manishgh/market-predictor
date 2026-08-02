from __future__ import annotations

from typing import Any

import pandas as pd
import pytest

from market_predictor.edge_rebuild.catalyst_identity_rebind import (
    _rebind_coverage,
    _rebind_decisions,
)
from market_predictor.v3.errors import DataReadinessError

DECISION_TIME = pd.Timestamp("2024-01-10T23:00:00Z")


def _target(*, duplicate_ticker: bool = False) -> pd.DataFrame:
    rows: list[dict[str, Any]] = [
        {
            "decision_id": "canonical-decision",
            "security_id": "cik:0000000123",
            "ticker": "ABC",
            "decision_time_utc": DECISION_TIME,
            "membership_effective_from_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
            "membership_effective_to_utc": pd.Timestamp("2030-01-01T00:00:00Z"),
            "membership_available_at_utc": pd.Timestamp("2020-01-01T00:00:00Z"),
        }
    ]
    if duplicate_ticker:
        rows.append(
            {
                **rows[0],
                "decision_id": "ambiguous-decision",
                "security_id": "cik:0000000999",
                "decision_time_utc": DECISION_TIME + pd.Timedelta(days=1),
            }
        )
    return pd.DataFrame.from_records(rows)


def test_decision_rebind_requires_exact_ticker_and_decision_time() -> None:
    source = pd.DataFrame(
        {
            "decision_id": ["legacy-exact", "legacy-off-session", "legacy-excluded"],
            "security_id": [
                "cik:0000000123:ticker:ABC",
                "cik:0000000123:ticker:ABC",
                "cik:0000000456:ticker:XYZ",
            ],
            "ticker": ["abc", "ABC", "XYZ"],
            "decision_time_utc": [
                DECISION_TIME,
                DECISION_TIME + pd.Timedelta(hours=1),
                DECISION_TIME,
            ],
            "event_count_3d": [2.0, 1.0, 1.0],
        }
    )

    decisions, ledger = _rebind_decisions(source, _target())

    assert decisions["decision_id"].tolist() == ["canonical-decision"]
    assert decisions["security_id"].tolist() == ["cik:0000000123"]
    assert decisions["event_count_3d"].tolist() == [2.0]
    reasons = ledger.set_index("source_decision_id")["match_reason"].to_dict()
    assert reasons == {
        "legacy-exact": "exact_ticker_decision_time",
        "legacy-off-session": "no_target_traded_decision",
        "legacy-excluded": "target_panel_excluded",
    }


def test_coverage_rebind_uses_unique_ticker_in_governed_population() -> None:
    source = pd.DataFrame(
        {
            "coverage_evidence_id": ["coverage-1", "coverage-2"],
            "collection_id": ["collection", "collection"],
            "chunk_id": ["ABC", "XYZ"],
            "security_id": [
                "cik:0000000123:ticker:ABC",
                "cik:0000000456:ticker:XYZ",
            ],
            "ticker": ["ABC", "XYZ"],
            "source_family": ["alpaca", "alpaca"],
            "requested_start_utc": [
                pd.Timestamp("2023-01-01T00:00:00Z"),
                pd.Timestamp("2023-01-01T00:00:00Z"),
            ],
            "requested_end_utc": [
                pd.Timestamp("2025-01-01T00:00:00Z"),
                pd.Timestamp("2025-01-01T00:00:00Z"),
            ],
            "completed_at_utc": [
                pd.Timestamp("2025-01-02T00:00:00Z"),
                pd.Timestamp("2025-01-02T00:00:00Z"),
            ],
            "status": ["complete", "complete"],
            "row_count": [4, 1],
            "coverage_state": ["observed_complete", "observed_complete"],
            "missingness_known": [True, True],
            "zero_event_semantics": ["observed_history", "observed_history"],
            "training_eligible": [True, True],
            "schema_version": ["v1", "v1"],
            "source_lineage_sha256s": ["[]", "[]"],
        }
    )

    coverage, ledger = _rebind_coverage(source, _target())

    assert coverage["security_id"].tolist() == ["cik:0000000123"]
    assert coverage["coverage_evidence_id"].str.len().tolist() == [64]
    assert ledger["match_reason"].tolist() == [
        "unique_ticker_in_target_population",
        "target_panel_excluded",
    ]


def test_coverage_rebind_rejects_ambiguous_target_ticker() -> None:
    source = pd.DataFrame(
        {
            "coverage_evidence_id": ["coverage-1"],
            "ticker": ["ABC"],
        }
    )

    with pytest.raises(DataReadinessError, match="multiple target securities"):
        _rebind_coverage(source, _target(duplicate_ticker=True))
