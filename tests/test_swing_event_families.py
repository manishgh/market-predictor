from __future__ import annotations

import tomllib
from pathlib import Path

import pandas as pd
import pytest
from pandas.testing import assert_frame_equal

from market_predictor.swing.event_families import (
    ALLOWED_SOURCE_FAMILIES_BY_FAMILY,
    EVENT_FAMILIES,
    EVENT_FAMILY_COLUMNS,
    EVENT_FAMILY_POLICY_VERSION,
    classify_event_families,
)
from market_predictor.core.errors import DataReadinessError

_AVAILABLE_AT = "2026-07-20T14:00:00Z"
_POLICY_PATH = Path(__file__).parents[1] / "configs" / "swing_event_family_policy.toml"


def _events(*rows: dict[str, object]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "event_id": f"event-{index}",
                "security_id": "security-acme",
                "ticker": "ACME",
                "issuer_company": "Acme Inc",
                "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
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
    assert EVENT_FAMILY_POLICY_VERSION == "swing.issuer_event_family.v2"
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


def test_structured_sec_form_requires_bound_issuer_identity() -> None:
    events = _events(
        {
            "source_family": "sec",
            "security_id": "",
            "ticker": "",
            "sec_form": "8-K",
            "title": "Current report filed",
        }
    )

    with pytest.raises(DataReadinessError, match="issuer security_id"):
        classify_event_families(events)


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


def test_family_source_allowlists_are_frozen() -> None:
    expected = {
        "earnings": ("alpaca", "finviz"),
        "guidance": ("alpaca", "finviz"),
        "sec_material_event": ("sec",),
        "analyst_revision": ("alpaca", "finviz"),
        "offering": ("alpaca", "finviz", "sec"),
        "merger_acquisition": ("alpaca", "finviz"),
        "regulatory_decision": ("alpaca", "finviz"),
        "product_event": ("alpaca", "finviz"),
    }
    policy = tomllib.loads(_POLICY_PATH.read_text(encoding="utf-8"))

    assert ALLOWED_SOURCE_FAMILIES_BY_FAMILY == expected
    assert policy["event_family_policy_version"] == EVENT_FAMILY_POLICY_VERSION
    assert policy["allowed_source_families"] == {
        family: list(sources) for family, sources in expected.items()
    }


def test_title_family_is_rejected_from_disallowed_source() -> None:
    classified = classify_event_families(
        _events(
            {
                "source_family": "sec",
                "sec_form": "",
                "title": "Acme reports Q2 earnings",
            }
        )
    )

    assert classified.empty


def test_micron_company_anchor_targets_analyst_action() -> None:
    classified = classify_event_families(
        _events(
            {
                "security_id": "security-mu",
                "ticker": "MU",
                "issuer_company": "Micron Technology Inc",
                "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
                "title": "BofA upgrades Micron to Buy",
            }
        )
    )

    assert classified["event_family"].tolist() == ["analyst_revision"]
    assert classified["classification_basis"].tolist() == [
        "issuer_targeted_title_rule"
    ]


def test_post_event_issuer_company_identity_fails_closed() -> None:
    events = _events(
        {
            "security_id": "security-mu",
            "ticker": "MU",
            "issuer_company": "Micron Technology Inc",
            "issuer_company_available_at_utc": "2026-07-21T14:00:00Z",
            "title": "BofA upgrades Micron to Buy",
        }
    )

    with pytest.raises(DataReadinessError, match="company availability"):
        classify_event_families(events)


@pytest.mark.parametrize("ticker", ["A", "APP", "NOW"])
def test_bare_ambiguous_ticker_word_is_not_an_issuer_anchor(ticker: str) -> None:
    classified = classify_event_families(
        _events(
            {
                "security_id": f"security-{ticker.lower()}",
                "ticker": ticker,
                "issuer_company": None,
                "issuer_company_available_at_utc": None,
                "title": "Now a company upgrades its internal app platform",
            }
        )
    )

    assert classified.empty


def test_explicit_ticker_notation_is_an_issuer_anchor() -> None:
    classified = classify_event_families(
        _events(
            {
                "security_id": "security-app",
                "ticker": "APP",
                "issuer_company": None,
                "issuer_company_available_at_utc": None,
                "title": "Morgan Stanley upgrades $APP to Buy",
            }
        )
    )

    assert classified["event_family"].tolist() == ["analyst_revision"]


def test_multi_ticker_microsoft_upgrade_targets_only_microsoft() -> None:
    title = "Morgan Stanley upgrades Microsoft; Apple shares rise"
    events = _events(
        {
            "event_id": "microsoft",
            "security_id": "security-msft",
            "ticker": "MSFT",
            "issuer_company": "Microsoft Corporation",
            "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
            "title": title,
        },
        {
            "event_id": "apple",
            "security_id": "security-aapl",
            "ticker": "AAPL",
            "issuer_company": "Apple Inc",
            "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
            "title": title,
        },
    )

    classified = classify_event_families(events)

    assert classified["event_id"].tolist() == ["microsoft"]


@pytest.mark.parametrize(
    "event",
    (
        {
            "security_id": "security-ups",
            "ticker": "UPS",
            "title": "UPS facility upgrades sorting equipment",
        },
        {
            "security_id": "security-tsla",
            "ticker": "TSLA",
            "issuer_company": "Tesla Inc",
            "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
            "title": "Morgan Stanley upgrades Ford; Tesla shares unchanged",
        },
        {
            "security_id": "security-mu",
            "ticker": "MU",
            "issuer_company": "Micron Technology Inc",
            "issuer_company_available_at_utc": "2026-07-19T14:00:00Z",
            "title": "Micron Q2 earnings preview: what investors should expect",
        },
        {
            "title": "Acme could acquire Target if financing is secured",
        },
    ),
)
def test_issuer_targeting_poison_cases_remain_unclassified(
    event: dict[str, object],
) -> None:
    classified = classify_event_families(_events(event))

    assert classified.empty


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
