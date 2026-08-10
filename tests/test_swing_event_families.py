from __future__ import annotations

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_predictor.swing.event_families import (
    EVENT_FAMILIES,
    EVENT_FAMILY_COLUMNS,
    classify_event_families,
)
from market_predictor.v3.errors import DataReadinessError

_AVAILABLE_AT = "2026-07-20T14:00:00Z"


def _events(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": f"event-{index}",
                "source_family": "alpaca",
                "feature_available_at_utc": _AVAILABLE_AT,
                "title": "Unclassified issuer update",
                **row,
            }
            for index, row in enumerate(rows, start=1)
        ]
    )


@pytest.mark.parametrize(
    ("family", "event"),
    (
        ("earnings", {"title": "Acme reports Q2 earnings"}),
        ("guidance", {"title": "Acme raises full-year guidance"}),
        (
            "sec_material_event",
            {
                "source_family": "sec",
                "sec_form": "8-K",
                "title": "Acme files current report",
            },
        ),
        ("analyst_revision", {"title": "Morgan Stanley upgrades Acme"}),
        ("offering", {"title": "Acme prices public offering"}),
        ("merger_acquisition", {"title": "Acme agrees to acquire Target"}),
        ("regulatory_decision", {"title": "FDA approves Acme therapy"}),
        ("product_event", {"title": "Acme launches new product"}),
    ),
)
def test_classifies_each_supported_family(
    family: str,
    event: dict[str, object],
) -> None:
    classified = classify_event_families(_events(event))

    assert set(EVENT_FAMILIES) == {
        "earnings",
        "guidance",
        "sec_material_event",
        "analyst_revision",
        "offering",
        "merger_acquisition",
        "regulatory_decision",
        "product_event",
    }
    assert family in set(classified["event_family"])
    assert classified["event_feature_available_at_utc"].eq(
        pd.Timestamp(_AVAILABLE_AT)
    ).all()


def test_one_event_can_be_both_earnings_and_guidance() -> None:
    classified = classify_event_families(
        _events(
            {
                "title": (
                    "Acme reports Q2 earnings and raises full-year guidance"
                )
            }
        )
    )

    assert set(classified["event_family"]) == {"earnings", "guidance"}
    assert classified["event_id"].nunique() == 1


def test_unmatched_event_remains_absent_instead_of_becoming_neutral() -> None:
    classified = classify_event_families(
        _events({"title": "Acme schedules its annual shareholder meeting"})
    )

    assert classified.empty
    assert tuple(classified.columns) == EVENT_FAMILY_COLUMNS


def test_structured_sec_form_is_used_without_title_keyword_inference() -> None:
    classified = classify_event_families(
        _events(
            {
                "source_family": "sec",
                "sec_form": "10-Q/A",
                "title": "Amendment filed",
            }
        )
    )

    row = classified.loc[
        classified["event_family"].eq("sec_material_event")
    ].iloc[0]
    assert row["classification_basis"] == "structured_sec_form"
    assert row["classification_rule_id"] == "sec_material_form"
    assert row["matched_text"] == "10-Q/A"


def test_duplicate_event_ids_are_rejected() -> None:
    events = _events(
        {"event_id": "duplicate", "title": "Acme reports Q2 earnings"},
        {"event_id": "duplicate", "title": "Acme raises guidance"},
    )

    with pytest.raises(DataReadinessError, match="unique non-empty event IDs"):
        classify_event_families(events)


@pytest.mark.parametrize("invalid", [None, "not-a-timestamp"])
def test_invalid_feature_availability_is_rejected(invalid: object) -> None:
    events = _events(
        {
            "title": "Acme reports Q2 earnings",
            "feature_available_at_utc": invalid,
        }
    )

    with pytest.raises(DataReadinessError, match="invalid feature availability"):
        classify_event_families(events)


def test_classification_is_deterministic_under_input_shuffle() -> None:
    events = _events(
        {"event_id": "z", "title": "Acme raises guidance"},
        {"event_id": "a", "title": "Acme reports Q2 earnings"},
        {"event_id": "m", "title": "FDA approves Acme therapy"},
    )

    expected = classify_event_families(events)
    shuffled = classify_event_families(
        events.sample(frac=1.0, random_state=17).reset_index(drop=True)
    )

    assert_frame_equal(shuffled, expected)


@pytest.mark.parametrize(
    "title",
    (
        "Acme earnings preview: what investors should expect",
        "Acme discusses its long-term acquisition strategy",
        "Acme presentation includes a general FDA regulatory overview",
    ),
)
def test_weak_context_without_completed_event_is_not_classified(title: str) -> None:
    classified = classify_event_families(_events({"title": title}))

    assert classified.empty
